"""lodger serve's behaviour without a graphics card, llama-server or systemd: request framing, answers that
continue after a move, waiting for a model, switching, failed starts, pi's files, paused services."""
import http.client
import json
import os
import socket
import stat
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

from lodger import serve


def chunk(text=None, finish=None, tool=False, role=False, cid="chatcmpl-1"):
    delta = {"role": "assistant"} if role else {}
    if text is not None:
        delta["content"] = text
    if tool:
        delta["tool_calls"] = [{"index": 0, "function": {"name": "f", "arguments": "{"}}]
    return {"id": cid, "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class StreamEngine:
    """Stands in for llama-server: streams `chunks`; then [DONE], or with cut=True the connection just closes
    and the engine counts as gone (as if lodger had moved it)."""

    def __init__(self, chunks, cut=False, where="card"):
        self.chunks, self.cut, self.where, self.dead, self.bodies = chunks, cut, where, False, []
        self.stopping, self.wd = False, SimpleNamespace(reason=None)
        eng = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                eng.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for c in eng.chunks:
                    self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                    self.wfile.flush()
                if eng.cut:
                    eng.dead = True
                else:
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
    """lodger's real request handler in front of a list of engines, handed out one per wait_engine()."""

    def __init__(self, engines):
        self.engines, self.engine = list(engines), None
        self.entry = SimpleNamespace(name="m", skip=None, ctx=4096, reasoning=False)
        self.asleep, self.ready, self.active, self.last_used, self.why = False, True, 0, 0, "ready"
        self.cv, self.force_off_card = threading.Condition(), False
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(self))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def available(self):
        return [self.entry]

    def status(self):
        return {"ready": True}

    def find(self, name):
        return self.entry

    def cooling(self, entry):
        return None

    def wait_engine(self, entry, limit=None, ping=None, turn=None):
        self.engine = self.engines.pop(0) if self.engines else None
        return self.engine

    def using(self, e):
        return serve.contextlib.nullcontext()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        for e in self.engines:
            e.close()

    def raw(self, data):
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as s:
            s.sendall(data)
            out = b""
            while True:
                b = s.recv(65536)
                if not b:
                    return out
                out += b
                if b"\r\n\r\n" in out and b"Content-Length: " in out:
                    head, _, rest = out.partition(b"\r\n\r\n")
                    n = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
                    if len(rest) >= n:
                        return out

    def chat(self, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        events = [l.decode().rstrip("\n") for l in r if l.strip()]
        c.close()
        return events


def data(events):
    return [json.loads(e[6:]) for e in events if e.startswith("data: {")]


def text(events):
    return "".join((d.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for d in data(events))


class StreamTest(unittest.TestCase):
    def setUp(self):
        self.fronts = []

    def tearDown(self):
        for f in self.fronts:
            f.close()

    def front(self, *engines):
        f = Front(engines)
        self.fronts.append(f)
        return f

    ASK = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "count"}]}

    def test_answer_continues_after_a_move(self):
        a = StreamEngine([chunk("one ", role=True, cid="first"), chunk("two ", cid="first")], cut=True)
        b = StreamEngine([chunk(None, role=True, cid="second"), chunk("three", cid="second"),
                          chunk(None, finish="stop", cid="second")])
        ev = self.front(a, b).chat(dict(self.ASK, max_tokens=10))
        self.assertEqual(text(ev), "one two three")
        self.assertEqual(ev[-1], "data: [DONE]")
        self.assertTrue(any(e.startswith(": lodger restarted") for e in ev))
        self.assertEqual({d["id"] for d in data(ev)}, {"first"})          # one answer, as far as the app can tell
        self.assertEqual(sum(1 for d in data(ev) if d["choices"][0]["delta"].get("role")), 1)
        resent = b.bodies[0]
        self.assertEqual(resent["messages"][-1], {"role": "assistant", "content": "one two "})
        self.assertEqual(resent["max_tokens"], 8)

    def test_app_prefill_is_continued_not_repeated(self):
        a = StreamEngine([chunk("lo", cid="x")], cut=True)
        b = StreamEngine([chunk(" world", cid="y"), chunk(None, finish="stop", cid="y")])
        ask = dict(self.ASK, messages=self.ASK["messages"] + [{"role": "assistant", "content": "Hel"}])
        self.front(a, b).chat(ask)
        msgs = b.bodies[0]["messages"]
        self.assertEqual(msgs[-1]["content"], "Hello")
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])

    def test_limit_reached_ends_without_a_new_engine(self):
        a = StreamEngine([chunk("a"), chunk("b")], cut=True)
        b = StreamEngine([chunk("never")])
        ev = self.front(a, b).chat(dict(self.ASK, max_tokens=2))
        self.assertEqual(text(ev), "ab")
        self.assertEqual(data(ev)[-1]["choices"][0]["finish_reason"], "length")
        self.assertEqual(ev[-1], "data: [DONE]")
        self.assertEqual(b.bodies, [])

    def test_finished_answer_is_not_continued(self):
        a = StreamEngine([chunk("done"), chunk(None, finish="stop")], cut=True)
        b = StreamEngine([chunk("extra")])
        ev = self.front(a, b).chat(self.ASK)
        self.assertEqual(text(ev), "done")
        self.assertEqual(ev[-1], "data: [DONE]")
        self.assertEqual(b.bodies, [])

    def test_tool_call_cut_asks_for_a_retry(self):
        a = StreamEngine([chunk(None, tool=True)], cut=True)
        ev = self.front(a, StreamEngine([chunk("x")])).chat(self.ASK)
        self.assertIn("tool call", data(ev)[-1]["error"]["message"])

    def test_structured_answer_cut_asks_for_a_retry(self):
        a = StreamEngine([chunk('{"a": ')], cut=True)
        ask = dict(self.ASK, response_format={"type": "json_object"})
        ev = self.front(a, StreamEngine([chunk("1}")])).chat(ask)
        self.assertIn("structured", data(ev)[-1]["error"]["message"])

    def test_broken_line_is_not_passed_on(self):
        a = StreamEngine([chunk("ok ")], cut=True)
        orig = a.httpd.RequestHandlerClass.do_POST

        def do_POST(h):
            orig(h)
            h.wfile.write(b'data: {"choi')        # cut in the middle of a line
        a.httpd.RequestHandlerClass.do_POST = do_POST
        b = StreamEngine([chunk("fine"), chunk(None, finish="stop")])
        ev = self.front(a, b).chat(self.ASK)
        self.assertFalse(any('"choi' in e and not e.startswith("data: {\"id") for e in ev))
        self.assertEqual(text(ev), "ok fine")


class FramingTest(unittest.TestCase):
    def setUp(self):
        self.f = Front([])

    def tearDown(self):
        self.f.close()

    def test_chunked_body_is_read(self):
        body = json.dumps({"model": ["x"]}).encode()
        req = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n"
               + f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n")
        out = self.f.raw(req)
        self.assertIn(b" 400 ", out.split(b"\r\n")[0])
        self.assertIn(b"'model' must be a string", out)

    def test_bad_lengths(self):
        for n, code in (("-1", b" 400 "), ("abc", b" 400 "), (str(serve.MAX_BODY + 1), b" 413 ")):
            out = self.f.raw(f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {n}\r\n\r\n".encode())
            self.assertIn(code, out.split(b"\r\n")[0], n)

    def test_bad_bodies(self):
        for body, why in ((b"[]", b"JSON object"), (b'{"messages": {}}', b"must be a list"), (b"{", b"not JSON")):
            out = self.f.raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: "
                             + str(len(body)).encode() + b"\r\n\r\n" + body)
            self.assertIn(b" 400 ", out.split(b"\r\n")[0], body)
            self.assertIn(why, out)

    def test_query_string_and_head(self):
        out = self.f.raw(b"GET /v1/models?x=1 HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        self.assertIn(b" 200 ", out.split(b"\r\n")[0])
        out = self.f.raw(b"HEAD /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        self.assertIn(b" 200 ", out.split(b"\r\n")[0])
        self.assertTrue(out.endswith(b"\r\n\r\n"))         # headers only


def lodger(**kw):
    a = SimpleNamespace(allow_gpu=False, idle=0, pause=[], threads=None, leave_gpu=0, min_ctx=4096, port=9999,
                        api_key=None, api_key_file=None, files=[], default=None)
    vars(a).update(kw)
    models = {n: SimpleNamespace(name=n, skip=None, ctx=4096, path=f"/{n}.gguf", reasoning=False,
                                 planning_args=lambda: [], args=lambda: []) for n in ("a", "b")}
    with mock.patch.object(serve, "RUNS", tempfile.mkdtemp()):
        l = serve.Lodger(a, models, "a")
    l.models_file = l.models_file + ".not-in-tests"      # the monitor's look at models.toml finds nothing to read
    l.models_seen = None
    return l


class FakeEng:
    def __init__(self, entry, where="processor"):
        self.entry, self.where, self.stopping, self.ready_at, self.out = entry, where, False, 1, "/run"
        self.stopped, self._alive = False, True
        self.wd = SimpleNamespace(reason=None, is_alive=lambda: True, stop_flag=threading.Event(),
                                  now={"vram": 0, "gtt": 0})
        self.plan = {"ngl": 0, "n_cpu_moe": 0, "ctx": 4096, "gpu_est": 0}

    def alive(self):
        return self._alive and not self.stopped

    def watched(self):
        return True

    def stop(self):
        self.stopped = True

    def finish(self):
        pass


class LodgerTest(unittest.TestCase):
    def setUp(self):
        self.l = lodger()
        self.l.note = lambda text: None

    def test_busy_engine_is_not_switched_away(self):
        l = self.l
        e = FakeEng(l.models["a"])
        l.install(e)
        got = l.wait_engine(l.models["a"], limit=1)
        self.assertIs(got, e)
        self.assertEqual(l.busy[e], 1)
        threading.Thread(target=l.wait_engine, args=(l.models["b"], 2), daemon=True).start()
        time.sleep(0.2)
        self.assertIs(l.switch_to, l.models["b"])
        with mock.patch.object(l, "start_or_give_up") as start:
            l.switch()                                     # an answer is being written: not now
            start.assert_not_called()
            self.assertIs(l.engine, e)
            l.release(e)
            l.switch()
            start.assert_called_once()
        self.assertTrue(e.stopped)
        self.assertIs(l.cur, l.models["b"])

    def test_requests_for_the_current_model_queue_behind_a_switch(self):
        l = self.l
        l.install(FakeEng(l.models["a"]))
        l.switch_to = l.models["b"]
        self.assertIsNone(l.wait_engine(l.models["a"], limit=0.3))

    def test_failed_start_cools_down_instead_of_disabling(self):
        l = self.l
        l.cur = l.models["b"]
        with mock.patch.object(l, "launch", return_value=None):
            l.last_fail = "no room"
            self.assertFalse(l.start_or_give_up())
        self.assertIsNone(l.models["b"].skip)               # not left out for good
        self.assertIn("no room", l.cooling(l.models["b"]))
        self.assertIs(l.cur, l.models["a"])
        t0 = time.time()
        self.assertIsNone(l.wait_engine(l.models["b"], limit=5))
        self.assertLess(time.time() - t0, 1)                # says so at once
        self.assertTrue(l.status()["models"]["b"].startswith("didn't start"))
        l.failed["b"] = (time.time() - 1, "old")
        self.assertIsNone(l.cooling(l.models["b"]))         # tried again after the cool-down
        self.assertEqual(l.status()["models"]["b"], "ok")

    def test_engine_the_watchdog_is_stopping_is_not_handed_out(self):
        l = self.l
        e = FakeEng(l.models["a"])
        l.install(e)
        e.wd.reason = "a game needs the card"                # still alive for a moment while it's stopped
        self.assertIsNone(l.wait_engine(l.models["a"], limit=0.3))
        self.assertEqual(l.busy, {})

    def test_default_request_skips_a_skipped_model(self):
        l = self.l
        l.models["a"].skip = "can't be read"
        self.assertIs(l.find(None), l.models["b"])
        self.assertIs(l.find("B"), l.models["b"])
        self.assertIsNone(l.find("a"))

    def test_idle_unload_waits_for_answers(self):
        l = lodger(idle=0.0001)
        l.note = lambda text: None
        l.asleep = False
        e = FakeEng(l.models["a"])
        l.install(e)
        l.last_used = 0
        nb = SimpleNamespace(update=lambda: None)
        with mock.patch.object(l, "others_coming", return_value=None), mock.patch.object(l, "note_desktop"):
            got = l.wait_engine(l.models["a"], limit=1)
            l.last_used = 0
            l.tick(nb)
            self.assertFalse(e.stopped)                     # being written: stays loaded
            l.release(got)
            l.tick(nb)
        self.assertTrue(e.stopped)
        self.assertTrue(l.asleep)

    def test_engine_lost_and_processor_refused_retries_later(self):
        l = self.l
        e = FakeEng(l.models["a"])
        l.install(e)
        e._alive = False
        nb = SimpleNamespace(update=lambda: None)
        with mock.patch.object(l, "others_coming", return_value=None), mock.patch.object(l, "note_desktop"), \
                mock.patch.object(l, "launch", return_value=None) as launch, mock.patch.object(serve.time, "sleep"):
            l.last_fail = "not enough memory"
            l.tick(nb)
            self.assertIsNone(l.engine)
            self.assertGreater(l.retry_at, time.time())
            l.tick(nb)                                      # too soon: no new try
            self.assertEqual(launch.call_count, 1)
            l.retry_at = 0
            l.tick(nb)
            self.assertEqual(launch.call_count, 2)
        self.assertIsNone(l.wait_engine(l.models["a"], limit=5))   # it keeps failing: no 10-minute wait

    def test_monitor_survives_an_error(self):
        l = self.l
        calls = []

        def tick(nb):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("sysfs went away")
            l.closing = True
        with mock.patch.object(l, "tick", side_effect=tick), mock.patch.object(serve, "Neighbors"), \
                mock.patch.object(serve.time, "sleep"):
            l.monitor()
        self.assertEqual(len(calls), 2)
        with open(os.path.join(l.out, "monitor-errors.log")) as f:
            self.assertIn("sysfs went away", f.read())

    def test_dead_watchdog_stops_the_model(self):
        l = self.l
        e = FakeEng(l.models["a"], where="card")
        e.watched = lambda: False
        e.fence = mock.Mock(stop=lambda: setattr(e, "_alive", False))
        l.install(e)
        nb = SimpleNamespace(update=lambda: None)
        with mock.patch.object(l, "others_coming", return_value=None), mock.patch.object(l, "note_desktop"), \
                mock.patch.object(l, "bring_up", return_value=True), mock.patch.object(l, "unpause"):
            l.tick(nb)
        self.assertFalse(e.alive())
        self.assertIn("watchdog", e.wd.reason)

    def test_asked_move_does_not_double_the_quiet_wait(self):
        l = self.l
        e = FakeEng(l.models["a"], where="card")
        l.install(e)
        l.force_off_card = True
        nb = SimpleNamespace(update=lambda: None)
        with mock.patch.object(l, "others_coming", return_value=None), mock.patch.object(l, "note_desktop"), \
                mock.patch.object(l, "bring_up", return_value=True), mock.patch.object(l, "unpause"):
            l.tick(nb)
        self.assertTrue(e.stopped)
        self.assertEqual(l.quiet_needed, serve.QUIET_SECONDS)
        self.assertGreater(l.last_aside, 0)


class PiConfigTest(unittest.TestCase):
    MODELS = [{"id": "m", "context_window": 8192}]

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.p = mock.patch.object(serve, "PI_DIR", self.d)
        self.p.start()

    def tearDown(self):
        self.p.stop()

    def path(self, name):
        return os.path.join(self.d, name)

    def test_unreadable_models_file_is_left_alone(self):
        with open(self.path("models.json"), "w") as f:
            f.write('{"providers": {"other": {"apiKey": "k"},}}')     # a trailing comma: half-edited
        with mock.patch("sys.stderr"):
            self.assertFalse(serve.write_pi_config(8181, self.MODELS, "m"))
        with open(self.path("models.json")) as f:
            self.assertIn('"other"', f.read())

    def test_other_providers_kept_and_links_followed(self):
        real = self.path("real-models.json")
        with open(real, "w") as f:
            json.dump({"providers": {"other": {"apiKey": "k"}}}, f)
        os.symlink(real, self.path("models.json"))
        self.assertTrue(serve.write_pi_config(8181, self.MODELS, "m"))
        self.assertTrue(os.path.islink(self.path("models.json")))
        with open(real) as f:
            cfg = json.load(f)
        self.assertEqual(set(cfg["providers"]), {"other", "lodger"})
        self.assertEqual(stat.S_IMODE(os.stat(real).st_mode), 0o600)
        self.assertEqual([n for n in os.listdir(self.d) if n.endswith(".tmp")], [])

    def test_settings(self):
        with open(self.path("settings.json"), "w") as f:
            json.dump({"defaultProvider": "lodger", "defaultModel": "m2", "theme": "dark"}, f)
        serve.write_pi_config(8181, self.MODELS + [{"id": "m2", "context_window": 8192}], "m")
        with open(self.path("settings.json")) as f:
            self.assertEqual(json.load(f)["defaultModel"], "m2")      # your pick among lodger's models stays
        with open(self.path("settings.json"), "w") as f:
            json.dump({"defaultProvider": "other", "defaultModel": "x"}, f)
        serve.write_pi_config(8181, self.MODELS, "m")
        with open(self.path("settings.json")) as f:
            self.assertEqual(json.load(f)["defaultProvider"], "other")
        with open(self.path("settings.json"), "w") as f:
            f.write("{oops")
        with mock.patch("sys.stderr"):
            serve.write_pi_config(8181, self.MODELS, "m")
        with open(self.path("settings.json")) as f:
            self.assertEqual(f.read(), "{oops")


class PauseTest(unittest.TestCase):
    def test_only_a_gone_owners_services_are_started(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(serve, "RUNTIME_DIR", d), \
                mock.patch.object(serve.subprocess, "run") as run, mock.patch.object(serve, "log"):
            with open(os.path.join(d, "lodger.paused-mine"), "w") as f:
                f.write(f"{os.getpid()}\n")
            with open(os.path.join(d, "lodger.paused-gone"), "w") as f:
                f.write("999999999\n")
            open(os.path.join(d, "lodger.paused-old"), "w").close()
            live = 1
            with mock.patch.object(serve, "lodger_alive", side_effect=lambda pid: pid == live):
                with open(os.path.join(d, "lodger.paused-theirs"), "w") as f:
                    f.write(f"{live}\n")
                serve.unpause_all()
            started = sorted(c.args[0][-1] for c in run.call_args_list)
            self.assertEqual(started, ["gone", "mine", "old"])
            self.assertEqual(os.listdir(d), ["lodger.paused-theirs"])


class PortTest(unittest.TestCase):
    def test_port_owned(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            port = s.getsockname()[1]
            self.assertTrue(serve.port_owned(port, [os.getpid()]))
            self.assertFalse(serve.port_owned(port, [1]))
            self.assertFalse(serve.port_owned(port, []))


if __name__ == "__main__":
    unittest.main()
