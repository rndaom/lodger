"""lodger serve's front door, found in the 2026-11-03 audit: request checks, fair waiting between two models, the PC
sleeping, growing back while a card engine fails. No graphics card or llama-server: the engines are small fakes."""
import http.client
import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

from lodger import serve
from tests.test_serve import FakeEng, lodger


class Engine:
    """Stands in for llama-server: answers every POST with `whole` (JSON, `status`), or streams `events` (strings
    sent as they are, dicts as `data:` lines) and then closes; cut=True: the engine counts as gone after that."""

    def __init__(self, whole=None, events=None, status=200, cut=False, where="processor", done=True):
        self.whole, self.events, self.status, self.cut, self.done = whole, list(events or []), status, cut, done
        self.where, self.dead, self.calls = where, False, []
        self.stopping, self.wd = False, SimpleNamespace(reason=None)
        eng = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                eng.calls.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                if eng.events is None or eng.whole is not None or eng.status != 200:
                    body = json.dumps(eng.whole).encode()
                    self.send_response(eng.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    return self.wfile.write(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for ev in eng.events:
                    self.wfile.write(ev.encode() if isinstance(ev, str) else b"data: " + json.dumps(ev).encode() + b"\n\n")
                    self.wfile.flush()
                if eng.cut:
                    eng.dead = True
                elif eng.done:
                    self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def auth(self):
        return {}

    def alive(self):
        return not self.dead

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Front:
    """lodger's real handler (main port, or Ollama's with ollama_api) in front of engines handed out one per
    wait_engine()."""

    def __init__(self, engines, key=None, ollama_api=False):
        self.engines, self.engine, self.waits = list(engines), None, []
        self.entry = SimpleNamespace(name="m", path="/nowhere/m.gguf", skip=None, ctx=4096, reasoning=False, settings={})
        self.asleep, self.ready, self.active, self.last_used, self.why = False, True, 0, 0, "ready"
        self.cv, self.force_off_card = threading.Condition(), False
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(self, key, ollama_api))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def available(self):
        return [self.entry]

    def status(self):
        return {"ready": True}

    def find(self, name):
        return self.entry if name in (None, "", "m") else None

    def cooling(self, entry):
        return None

    def loaded(self):
        return None

    def wait_engine(self, entry, limit=None, ping=None):
        self.waits.append(ping)
        self.engine = self.engines.pop(0) if self.engines else None
        return self.engine

    def release(self, e):
        pass

    def using(self, e):
        return serve.contextlib.nullcontext()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        for e in self.engines:
            e.close()

    def raw(self, data, until=b"\r\n\r\n"):
        """Send bytes, return what came back once `until` has arrived (or the connection closed)."""
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as s:
            s.sendall(data)
            out = b""
            while until not in out:
                b = s.recv(65536)
                if not b:
                    break
                out += b
            return out

    def post(self, path, body, headers=None):
        """(status, headers, body bytes)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", path, body if isinstance(body, (bytes, str)) else json.dumps(body),
                  {"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        out = r.status, dict(r.getheaders()), r.read()
        c.close()
        return out


class FrontCase(unittest.TestCase):
    def setUp(self):
        self.fronts = []

    def tearDown(self):
        for f in self.fronts:
            f.close()

    def front(self, *engines, **kw):
        f = Front(engines, **kw)
        self.fronts.append(f)
        return f


ASK = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


class RequestTest(FrontCase):
    def test_stream_must_be_true_or_false(self):
        e = Engine(whole={"choices": []})
        status, _, data = self.front(e).post("/v1/chat/completions", dict(ASK, stream="false"))
        self.assertEqual(status, 400)
        self.assertIn(b"'stream' must be true or false", data)
        self.assertEqual(e.calls, [])
        status, _, _ = self.front(Engine(whole={"choices": []})).post("/v1/chat/completions", dict(ASK, stream=None))
        self.assertEqual(status, 200)

    def test_only_streaming_answers_are_streamed(self):
        e = Engine(whole={"tokens": [1, 2]})
        status, headers, data = self.front(e).post("/tokenize", {"content": "hi", "stream": True})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data), {"tokens": [1, 2]})       # not an event stream of a JSON answer
        self.assertNotEqual(headers.get("Content-Type"), "text/event-stream")

    def test_refused_request_gets_no_go_ahead(self):
        f = self.front()
        out = f.raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: evil.example\r\nContent-Length: 10\r\n"
                    b"Expect: 100-continue\r\n\r\n")
        self.assertTrue(out.startswith(b"HTTP/1.1 403"), out[:40])
        out = f.raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 2\r\n"
                    b"Expect: 100-continue\r\n\r\n")
        self.assertTrue(out.startswith(b"HTTP/1.1 100"), out[:40])


class Waiter(threading.Thread):
    """A request waiting for a model's engine (wait_engine) in its own thread; it releases the engine at once."""

    def __init__(self, l, name, **kw):
        super().__init__(daemon=True)
        self.l, self.entry, self.kw, self.got = l, l.models[name], kw, None
        self.start()

    def run(self):
        self.got = self.l.wait_engine(self.entry, **{"limit": 5, **self.kw})
        if self.got:
            self.l.release(self.got)


class FairnessTest(unittest.TestCase):
    def setUp(self):
        self.l = lodger()
        self.l.note = lambda text: None

    def loads(self, delay=0.3):
        """start_or_give_up() that installs a fake engine for the model being switched to, after `delay`."""
        def start():
            time.sleep(delay)
            self.l.install(FakeEng(self.l.cur))
            return True
        return mock.patch.object(self.l, "start_or_give_up", side_effect=start)

    def test_requests_for_two_models_are_both_served(self):
        l = self.l
        l.install(FakeEng(l.models["a"]))
        wb = Waiter(l, "b")
        time.sleep(0.1)
        self.assertIs(l.switch_to, l.models["b"])
        with self.loads():
            threading.Timer(0.1, lambda: setattr(self, "wa", Waiter(l, "a"))).start()
            l.switch()                                # loads b; meanwhile a request for a comes in
            wb.join(2)
            self.assertIs(wb.got and wb.got.entry, l.models["b"])   # the request that asked for b gets b first
            self.assertIs(l.switch_to, l.models["a"])
            l.switch()                                # then a, for the request that waited for it
            self.wa.join(2)
        self.assertIs(self.wa.got and self.wa.got.entry, l.models["a"])

    def test_switch_waits_for_requests_that_came_first(self):
        l = self.l
        l.install(FakeEng(l.models["a"]))
        with l.cv:                                # a request for a, not yet served when b is asked for
            l.ticket += 1
            l.waiting[object()] = (l.models["a"], l.ticket)
            l.switch_to, l.switch_ticket = l.models["b"], l.ticket + 1
        with mock.patch.object(l, "start_or_give_up") as start:
            l.switch()
            start.assert_not_called()

    def test_pings_keep_the_place_in_line(self):
        l = self.l
        pings = []
        with mock.patch.object(serve, "PING", 0.2):
            w = Waiter(l, "b", limit=1, ping=lambda: pings.append(1))
            w.join(2)
        self.assertGreaterEqual(len(pings), 3)
        self.assertEqual(l.waiting, {})

    def test_app_gone_ends_the_wait(self):
        l = self.l

        def gone():
            raise OSError("the app went away")
        with mock.patch.object(serve, "PING", 0.1):
            with self.assertRaises(OSError):
                l.wait_engine(l.models["b"], limit=2, ping=gone)
        self.assertEqual(l.waiting, {})


QUIET = SimpleNamespace(update=lambda: None)


class MonitorCase(unittest.TestCase):
    def setUp(self):
        self.l = lodger()
        self.notes = []
        self.l.note = self.notes.append
        self.quiet = [mock.patch.object(self.l, "others_coming", return_value=None),
                      mock.patch.object(self.l, "note_desktop"), mock.patch.object(self.l, "unpause")]
        for p in self.quiet:
            p.start()

    def tearDown(self):
        for p in self.quiet:
            p.stop()


class SleepTest(MonitorCase):
    def test_wake_from_sleep_does_not_stop_a_healthy_model(self):
        l = self.l
        e = FakeEng(l.models["a"], where="card")
        beat = time.time()
        e.watched = lambda: time.time() - beat < 10      # the watchdog's heartbeat runs on wall time
        e.fence = mock.Mock()
        l.install(e)
        l.clock_off -= 3600                             # as after an hour of sleep: the wall clock moved, monotonic didn't
        beat -= 3600
        l.quiet_since = 0
        with mock.patch.object(l, "bring_up", return_value=True):
            l.tick(QUIET)
        e.fence.stop.assert_not_called()
        self.assertIsNone(e.wd.reason)
        self.assertTrue(any("slept" in n for n in self.notes), self.notes)
        self.assertGreater(l.quiet_since, time.time() - 5)     # the card wasn't watched meanwhile: no grow-back yet

    def test_a_dead_watchdog_still_stops_the_model(self):
        l = self.l
        e = FakeEng(l.models["a"], where="card")
        e.watched = lambda: False
        e.fence = mock.Mock(stop=lambda: setattr(e, "_alive", False))
        l.install(e)
        l.clock_off -= 3600
        with mock.patch.object(l, "bring_up", return_value=True):
            l.tick(QUIET)                               # grace after waking ...
            self.assertTrue(e.alive())
            l.grace_until = 0                           # ... but not longer than that
            l.tick(QUIET)
        self.assertFalse(e.alive())


class GrowBackTest(MonitorCase):
    def test_card_engine_stopped_during_the_hand_over_keeps_the_processor(self):
        l = self.l
        l.a.allow_gpu = True
        old = FakeEng(l.models["a"])
        l.install(old)
        l.quiet_since = l.last_aside = 0
        new = FakeEng(l.models["a"], where="card")
        got = l.wait_engine(l.models["a"], limit=1)     # an answer is being written on the processor
        self.assertIs(got, old)

        def game_starts():
            new.wd.reason = "an app needs the graphics card"
            new._alive = False
            time.sleep(0.3)
            l.release(old)
        with mock.patch.object(l, "make_plan", return_value={"refuse": None, "ngl": 99}), \
                mock.patch.object(l, "launch", return_value=new), mock.patch.object(serve, "GROW_WAIT", 5):
            threading.Timer(0.2, game_starts).start()
            l.tick(QUIET)
        self.assertIs(l.engine, old)
        self.assertFalse(old.stopped)
        self.assertEqual(l.why_off_card, "stepped aside: an app needs the graphics card")


if __name__ == "__main__":
    unittest.main()
