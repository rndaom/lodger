"""Who may use `lodger serve`: Host, Origin and API key checks, and the engine's own key.
No graphics card, llama-server or systemd needed: the engine is a small fake."""
import contextlib
import http.client
import json
import os
import stat
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

from lodger import serve

KEY = "s3cret-key"


class HostTest(unittest.TestCase):
    def test_local(self):
        for h in ("127.0.0.1", "127.0.0.1:8181", "localhost", "LocalHost:8181", "[::1]", "[::1]:8181"):
            self.assertTrue(serve.local_host(h), h)

    def test_not_local(self):
        for h in (None, "", "evil.example", "evil.example:8181", "localhost.evil.example", "127.0.0.1.nip.io",
                  "localhost:", "localhost:abc", "localhost:８１", "[::1]x", "[::2]:8181", "::1", "0.0.0.0:8181",
                  "127.0.0.2", "localhost.", "user@localhost"):
            self.assertFalse(serve.local_host(h), h)


class OriginTest(unittest.TestCase):
    def test_local(self):
        for o in ("http://localhost", "http://localhost:3000", "https://127.0.0.1:8443", "http://[::1]:5173",
                  "HTTP://LOCALHOST", "app://whatever", "file://", "tauri://localhost", "vscode-webview://abc"):
            self.assertTrue(serve.local_origin(o), o)

    def test_not_local(self):
        for o in ("null", "", "https://evil.example", "http://localhost.evil.example", "http://127.0.0.1.nip.io",
                  "ftp://localhost", "localhost", "http://localhost/path", "chrome-extension://abc",
                  "http://evil.example#localhost"):
            self.assertFalse(serve.local_origin(o), o)


class CheckTest(unittest.TestCase):
    def test_plain_local_request(self):
        self.assertIsNone(serve.check("POST", "/v1/chat/completions", {"Host": "127.0.0.1:8181"}))

    def test_bad_host(self):
        code, msg = serve.check("GET", "/health", {"Host": "evil.example:8181"})
        self.assertEqual(code, 403)
        self.assertIn("evil.example", msg)

    def test_no_host(self):
        self.assertEqual(serve.check("GET", "/v1/models", {})[0], 403)

    def test_foreign_origin(self):
        h = {"Host": "localhost:8181", "Origin": "https://evil.example"}
        self.assertEqual(serve.check("POST", "/v1/chat/completions", h)[0], 403)
        self.assertEqual(serve.check("OPTIONS", "/v1/chat/completions", h)[0], 403)
        self.assertEqual(serve.check("GET", "/health", h)[0], 403)

    def test_local_origin(self):
        h = {"Host": "localhost:8181", "Origin": "http://localhost:3000"}
        self.assertIsNone(serve.check("POST", "/v1/chat/completions", h))

    def test_key(self):
        host = {"Host": "127.0.0.1:8181"}
        for path in ("/v1/chat/completions", "/chat/completions", "/v1/models", "/lodger/status", "/lodger/move",
                     "/nothing"):
            self.assertEqual(serve.check("POST", path, host, KEY)[0], 401, path)
            for auth in ("Bearer wrong", f"Basic {KEY}", KEY, f"Bearer {KEY}x", "Bearer "):
                self.assertEqual(serve.check("POST", path, {**host, "Authorization": auth}, KEY)[0], 401, auth)
            for auth in (f"Bearer {KEY}", f"bearer {KEY}", f"  Bearer  {KEY} "):
                self.assertIsNone(serve.check("POST", path, {**host, "Authorization": auth}, KEY), auth)

    def test_health_and_preflight_need_no_key(self):
        host = {"Host": "127.0.0.1:8181"}
        self.assertIsNone(serve.check("GET", "/health", host, KEY))
        self.assertIsNone(serve.check("GET", "/health?x=1", host, KEY))
        self.assertIsNone(serve.check("OPTIONS", "/v1/chat/completions", {**host, "Origin": "http://localhost"}, KEY))

    def test_host_checked_before_key(self):
        h = {"Host": "evil.example", "Authorization": f"Bearer {KEY}"}
        self.assertEqual(serve.check("GET", "/v1/models", h, KEY)[0], 403)


class KeyTest(unittest.TestCase):
    def test_read_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "key")
            with open(p, "w") as f:
                f.write("# lodger's key\n\n  abc123  \nignored\n")
            with mock.patch.dict(os.environ, {"LODGER_API_KEY": "from-env"}):
                self.assertEqual(serve.read_key(p), "abc123")       # the file wins
                self.assertEqual(serve.read_key(), "from-env")
            with mock.patch.dict(os.environ, {"LODGER_API_KEY": ""}):
                self.assertIsNone(serve.read_key())
            open(p, "w").close()
            with self.assertRaises(SystemExit):
                serve.read_key(p)
            with self.assertRaises(SystemExit):
                serve.read_key(os.path.join(d, "missing"))

    def test_pi_key(self):
        self.assertEqual(serve.pi_key(None), "lodger")
        self.assertEqual(serve.pi_key("abc"), "abc")
        self.assertEqual(serve.pi_key("a$b"), "a$$b")
        self.assertEqual(serve.pi_key("!abc"), "$!abc")
        self.assertEqual(serve.pi_key("abc", "/tmp/my key"), "!cat '/tmp/my key'")

    def test_pi_config_private(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(serve, "PI_DIR", d):
            serve.write_pi_config(8181, [{"id": "m", "context_window": 4096}], "m", "abc")
            path = os.path.join(d, "models.json")
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            with open(path) as f:
                self.assertEqual(json.load(f)["providers"]["lodger"]["apiKey"], "abc")


class FakeEngine:
    """Stands in for llama-server: answers only with its key."""

    def __init__(self):
        self.key, self.seen = "engine-key", []
        eng = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                eng.seen.append(self.headers.get("Authorization"))
                ok = self.headers.get("Authorization") == f"Bearer {eng.key}"
                body = json.dumps({"choices": [{"message": {"content": "hi"}}]} if ok else {"error": "key"}).encode()
                self.send_response(200 if ok else 401)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def auth(self):
        return {"Authorization": f"Bearer {self.key}"}

    def alive(self):
        return True


class FakeLodger:
    def __init__(self, engine):
        self.engine, self.entry = engine, SimpleNamespace(name="m", skip=None)
        self.asleep, self.ready, self.active, self.last_used = False, True, 0, 0
        self.cv, self.why, self.force_off_card = threading.Condition(), "ready", False

    def available(self):
        return []

    def status(self):
        return {"ready": True}

    def find(self, name):
        return self.entry

    def wait_engine(self, entry):
        return self.engine

    @contextlib.contextmanager
    def using(self, e):
        yield


class HandlerTest(unittest.TestCase):
    """The real request handler on a free port, in front of the fake engine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()
        cls.servers = []

    @classmethod
    def tearDownClass(cls):
        for s in cls.servers + [cls.engine.httpd]:
            s.shutdown()
            s.server_close()

    def start(self, key=None):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(FakeLodger(self.engine), key))
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.servers.append(httpd)
        return httpd.server_address[1]

    def call(self, port, method, path, headers=None, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        headers = dict(headers or {})
        c.putrequest(method, path, skip_host="Host" in headers)
        data = json.dumps(body).encode() if body is not None else b""
        for k, v in {**headers, "Content-Length": str(len(data))}.items():
            c.putheader(k, v)
        c.endheaders(data)
        r = c.getresponse()
        out = r.status, dict(r.getheaders()), r.read()
        c.close()
        return out

    CHAT = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}

    def test_chat_reaches_engine_with_its_key(self):
        port = self.start()
        code, _, body = self.call(port, "POST", "/v1/chat/completions", body=self.CHAT)
        self.assertEqual(code, 200, body)
        self.assertEqual(self.engine.seen[-1], "Bearer engine-key")

    def test_bad_host(self):
        port = self.start()
        code, headers, body = self.call(port, "POST", "/v1/chat/completions", {"Host": "evil.example"}, self.CHAT)
        self.assertEqual(code, 403)
        self.assertIn("evil.example", json.loads(body)["error"]["message"])
        self.assertEqual(headers.get("Connection"), "close")

    def test_foreign_origin_and_preflight(self):
        port = self.start()
        evil = {"Origin": "https://evil.example"}
        self.assertEqual(self.call(port, "POST", "/v1/chat/completions", evil, self.CHAT)[0], 403)
        code, headers, _ = self.call(port, "OPTIONS", "/v1/chat/completions",
                                     {**evil, "Access-Control-Request-Method": "POST"})
        self.assertEqual(code, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_local_origin_gets_cors(self):
        port = self.start()
        local = {"Origin": "http://localhost:3000"}
        code, headers, _ = self.call(port, "OPTIONS", "/v1/chat/completions",
                                     {**local, "Access-Control-Request-Method": "POST"})
        self.assertEqual(code, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://localhost:3000")
        self.assertIn("Authorization", headers.get("Access-Control-Allow-Headers"))
        code, headers, _ = self.call(port, "POST", "/v1/chat/completions", local, self.CHAT)
        self.assertEqual(code, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://localhost:3000")

    def test_key(self):
        port = self.start(KEY)
        code, headers, _ = self.call(port, "POST", "/v1/chat/completions", body=self.CHAT)
        self.assertEqual(code, 401)
        self.assertIn("Bearer", headers.get("WWW-Authenticate"))
        self.assertEqual(self.call(port, "GET", "/lodger/status")[0], 401)
        self.assertEqual(self.call(port, "GET", "/health")[0], 200)
        auth = {"Authorization": f"Bearer {KEY}"}
        self.assertEqual(self.call(port, "POST", "/v1/chat/completions", auth, self.CHAT)[0], 200)
        self.assertEqual(self.call(port, "GET", "/lodger/status", auth)[0], 200)

    def test_keep_alive_after_refusal(self):
        """A refused request's unread body must not be taken for the next request."""
        port = self.start()
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.putrequest("POST", "/v1/chat/completions", skip_host=True)
        c.putheader("Host", "evil.example")
        c.putheader("Content-Length", "5")
        c.endheaders(b"hello")
        r = c.getresponse()
        r.read()
        self.assertEqual(r.status, 403)
        self.assertTrue(r.will_close)
        c.close()


class EngineKeyTest(unittest.TestCase):
    """The engine gets its own key in a private file, never on its command line."""

    def test_engine_key(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(serve, "RUNTIME_DIR", d), \
                mock.patch.object(serve.subprocess, "Popen") as popen, \
                mock.patch.object(serve, "Watchdog"):
            plan = {"model": "/m.gguf", "ctx": 4096, "threads": 4, "ngl": 0, "n_cpu_moe": 0, "ram_cap": 1 << 30}
            entry = SimpleNamespace(name="m", args=lambda: [])
            e = serve.Engine(plan, entry, d)
            cmd = popen.call_args[0][0]
            self.assertNotIn(e.key, " ".join(cmd))
            self.assertEqual(cmd[cmd.index("--api-key-file") + 1], e.keyfile)
            self.assertEqual(stat.S_IMODE(os.stat(e.keyfile).st_mode), 0o600)
            with open(e.keyfile) as f:
                self.assertEqual(f.read().strip(), e.key)
            with open(os.path.join(d, "plan.json")) as f:
                self.assertNotIn(e.key, f.read())
            self.assertEqual(e.auth(), {"Authorization": f"Bearer {e.key}"})
            self.assertGreaterEqual(len(e.key), 32)
            e.finish()
            self.assertFalse(os.path.exists(e.keyfile))
            serve.Engine.all.remove(e)


if __name__ == "__main__":
    unittest.main()
