"""Ollama's API in front of lodger (--ollama): requests translated for llama-server, answers turned back into
Ollama's shape, streamed answers that continue after a move, the listener and its checks. No graphics card,
llama-server or Ollama needed: the engines are small fakes."""
import http.client
import json
import os
import socket
import struct
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

from lodger import ollama, serve

KEY = "s3cret-key"


def chunk(text=None, think=None, finish=None, tool=None, cid="chatcmpl-1", **extra):
    delta = {}
    if text is not None:
        delta["content"] = text
    if think is not None:
        delta["reasoning_content"] = think
    if tool is not None:
        delta["tool_calls"] = [tool]
    return {"id": cid, "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}


class Engine:
    """Stands in for llama-server. Streams `lines` (dicts become `data:` events; then [DONE] unless done=False),
    or with cut=True just closes and counts as gone (as if lodger had moved it); or answers `whole` as JSON."""

    def __init__(self, lines=(), cut=False, done=True, whole=None, status=200, where="processor"):
        self.lines, self.cut, self.done, self.whole, self.status = list(lines), cut, done, whole, status
        self.where, self.dead, self.calls = where, False, []
        self.stopping, self.wd = False, SimpleNamespace(reason=None)
        eng = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                eng.calls.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                if eng.whole is not None:
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
                for c in eng.lines:
                    self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
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
    """lodger's real handler, as the Ollama listener, in front of engines handed out one per wait_engine()."""

    def __init__(self, engines, key=None, entry=None):
        self.engines, self.engine, self.waits = list(engines), None, 0
        self.entry = entry or SimpleNamespace(name="m", path="/nowhere/m.gguf", skip=None, ctx=4096,
                                              reasoning=True, settings={})
        self.asleep, self.ready, self.active, self.last_used, self.why = False, True, 0, 0, "ready"
        self.cv, self.force_off_card = threading.Condition(), False
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(self, key, ollama_api=True))
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
        return (self.entry, 0, None) if self.engine else None

    def wait_engine(self, entry, limit=None, ping=None):
        self.waits += 1
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

    def call(self, method, path, body=None, headers=None):
        """(status, headers, body bytes)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(method, path, json.dumps(body) if body is not None else None,
                  {"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, r, data

    def post(self, path, body, **kw):
        status, r, data = self.call("POST", path, body, **kw)
        return status, json.loads(data)

    def lines(self, path, body):
        status, r, data = self.call("POST", path, body)
        self.ctype = r.getheader("Content-Type")
        return [json.loads(line) for line in data.splitlines() if line.strip()]


def said(lines, kind="chat"):
    return "".join((o["message"]["content"] if kind == "chat" else o["response"]) for o in lines if "error" not in o)


class TranslateTest(unittest.TestCase):
    def test_options(self):
        _, req, _ = ollama.request("/api/chat", {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                                 "options": {"num_ctx": 65536, "num_predict": 50, "temperature": 0.2,
                                                             "top_p": 0.9, "top_k": 20, "seed": 7, "stop": "END",
                                                             "num_gpu": 99, "repeat_penalty": 1.1}}, "m")
        self.assertEqual(req["max_tokens"], 50)
        self.assertEqual((req["temperature"], req["top_p"], req["top_k"], req["seed"]), (0.2, 0.9, 20, 7))
        self.assertEqual(req["stop"], ["END"])
        self.assertEqual(req["repeat_penalty"], 1.1)
        for k in ("num_ctx", "n_ctx", "num_gpu", "options"):
            self.assertNotIn(k, req)                      # lodger decides the context and the card
        self.assertTrue(req["stream"])                    # Ollama streams unless told not to
        _, req, _ = ollama.request("/api/chat", {"messages": [{"role": "user", "content": "x"}], "stream": False,
                                                 "options": {"num_predict": -1}}, "m")
        self.assertNotIn("max_tokens", req)
        self.assertFalse(req["stream"])
        _, req, _ = ollama.request("/api/chat", {"messages": [{"role": "user", "content": "x"}], "stream": None}, "m")
        self.assertTrue(req["stream"])                    # null: the default, as Ollama reads it

    def test_format_and_think(self):
        ask = {"messages": [{"role": "user", "content": "x"}]}
        _, req, _ = ollama.request("/api/chat", dict(ask, format="json", think=False), "m")
        self.assertEqual(req["response_format"], {"type": "json_object"})
        self.assertEqual(req["chat_template_kwargs"], {"enable_thinking": False})
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        _, req, _ = ollama.request("/api/chat", dict(ask, format=schema, think="high"), "m")
        self.assertEqual(req["response_format"]["json_schema"]["schema"], schema)
        self.assertEqual(req["reasoning_effort"], "high")
        self.assertNotIn("chat_template_kwargs", req)
        _, req, _ = ollama.request("/api/generate", {"prompt": "x", "raw": True, "format": schema}, "m")
        self.assertEqual(req["json_schema"], schema)
        with self.assertRaises(ValueError):
            ollama.request("/api/chat", dict(ask, format="yaml"), "m")

    def test_messages(self):
        msgs = [{"role": "system", "content": "be brief"},
                {"role": "user", "content": "what's this?", "images": ["iVBORw0KGgo="]},
                {"role": "assistant", "content": "", "thinking": "hmm",
                 "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Tokyo"}}},
                                {"function": {"name": "time", "arguments": {}}}]},
                {"role": "tool", "content": "sunny", "tool_name": "weather"},
                {"role": "tool", "content": "noon", "tool_name": "time"}]
        _, req, _ = ollama.request("/api/chat", {"messages": msgs}, "m")
        out = req["messages"]
        self.assertEqual(out[0], {"role": "system", "content": "be brief"})
        self.assertEqual(out[1]["content"][1], {"type": "image_url",
                                                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}})
        calls = out[2]["tool_calls"]
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Tokyo"})
        self.assertEqual(out[2]["reasoning_content"], "hmm")
        self.assertTrue(all(len(c["id"]) == 9 and c["id"].isalnum() for c in calls))   # Mistral's templates
        self.assertEqual([m["tool_call_id"] for m in out[3:]], [c["id"] for c in calls])
        self.assertEqual(out[3]["name"], "weather")
        with self.assertRaises(ValueError):
            ollama.request("/api/chat", {"messages": [{"role": "user", "content": ["parts"]}]}, "m")

    def test_generate(self):
        path, req, _ = ollama.request("/api/generate", {"prompt": "why?", "system": "be brief"}, "m")
        self.assertEqual(path, "/v1/chat/completions")    # so it continues after a move, as chat does
        self.assertEqual(req["messages"], [{"role": "system", "content": "be brief"},
                                           {"role": "user", "content": "why?"}])
        path, req, _ = ollama.request("/api/generate", {"prompt": "[INST] x [/INST]", "raw": True}, "m")
        self.assertEqual((path, req["prompt"]), ("/v1/completions", "[INST] x [/INST]"))
        path, req, _ = ollama.request("/api/generate", {"prompt": "def f(", "suffix": "\n    return x"}, "m")
        self.assertEqual((path, req["input_prefix"], req["input_suffix"]), ("/infill", "def f(", "\n    return x"))
        self.assertIsNone(ollama.request("/api/generate", {"model": "m"}, "m")[1])   # load the model
        self.assertIsNone(ollama.request("/api/chat", {"model": "m", "messages": []}, "m")[1])

    def test_embed(self):
        path, req, _ = ollama.request("/api/embed", {"input": ["a", {"text": "b"}]}, "m")
        self.assertEqual((path, req["input"]), ("/v1/embeddings", ["a", "b"]))
        self.assertEqual(ollama.request("/api/embeddings", {"prompt": "a"}, "m")[1]["input"], "a")
        _, req, ans = ollama.request("/api/embed", {"model": "m"}, "m")
        self.assertIsNone(req)                            # load the model
        self.assertEqual(ans.loaded(), {"model": "m", "embeddings": []})
        with self.assertRaises(ValueError):
            ollama.request("/api/embed", {"input": 3}, "m")

    def test_names(self):
        self.assertEqual(ollama.tagged("qwen"), "qwen:latest")
        self.assertEqual(ollama.tagged("qwen:9b"), "qwen:9b")
        self.assertEqual(ollama.plain("qwen:latest"), "qwen")
        self.assertEqual(ollama.plain("qwen:9b"), "qwen:9b")
        self.assertIsNone(ollama.plain(None))


def gguf(path, settings):
    """A GGUF header with these settings (text or uint32), then the tokenizer's list."""
    def text(s):
        return struct.pack("<Q", len(s.encode())) + s.encode()
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(settings) + 1))
        for k, v in settings.items():
            f.write(text(k) + (struct.pack("<I", 8) + text(v) if isinstance(v, str) else struct.pack("<II", 4, v)))
        f.write(text("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + text("a") + text("b"))
        f.write(b"\0" * 64)
    return path


class ModelInfoTest(unittest.TestCase):
    def test_details_from_the_header(self):
        with tempfile.TemporaryDirectory() as d:
            p = gguf(os.path.join(d, "Qwen-9B.gguf"), {"general.architecture": "qwen35", "general.size_label": "9B",
                                                       "general.file_type": 15, "qwen35.context_length": 262144})
            e = SimpleNamespace(name="qwen", path=p, ctx=32768, reasoning=False, settings={"mmproj": "/x"}, skip=None)
            item = ollama.listed(e)
            self.assertEqual(item["name"], "qwen:latest")
            self.assertEqual(item["size"], os.path.getsize(p))
            self.assertEqual(len(item["digest"]), 64)
            self.assertEqual(item["details"], {"parent_model": "", "format": "gguf", "family": "qwen35",
                                               "families": ["qwen35"], "parameter_size": "9B",
                                               "quantization_level": "Q4_K_M"})
            info = ollama.show(e)
            self.assertEqual(info["model_info"]["qwen35.context_length"], 32768)   # lodger's, not the trained one
            self.assertNotIn("tokenizer.ggml.tokens", info["model_info"])
            self.assertEqual(info["capabilities"], ["completion", "tools", "vision"])
            self.assertEqual(info["parameters"], "num_ctx 32768")

    def test_quantization_from_the_name(self):
        with tempfile.TemporaryDirectory() as d:
            p = gguf(os.path.join(d, "Model-IQ4_XS.gguf"), {"general.architecture": "llama"})
            e = SimpleNamespace(name="x", path=p, ctx=None, reasoning=True, settings={}, skip=None)
            self.assertEqual(ollama.details(e)["quantization_level"], "IQ4_XS")
            self.assertEqual(ollama.listed(SimpleNamespace(name="y", path=os.path.join(d, "gone.gguf")))["size"], 0)
            info = ollama.show(e)                         # context not decided yet: left out, not null
            self.assertEqual((info["model_info"], info["parameters"]), ({"general.architecture": "llama"}, ""))
            self.assertNotIn("context_length", ollama.running(e, 0, None))

    def test_bad_engine_output_is_left_out(self):
        a = ollama.Answer("chat", "m")
        self.assertEqual(a.feed('data: {"choices": [{"delta": {"tool_calls": ["x"]}}]}'), [])
        self.assertEqual(a.whole(200, b'{"choices": ["x"]}')[0], 502)
        self.assertEqual(a.whole(200, b'[1]')[0], 502)


class OllamaFrontTest(unittest.TestCase):
    def setUp(self):
        self.fronts = []

    def tearDown(self):
        for f in self.fronts:
            f.close()

    def front(self, *engines, **kw):
        f = Front(engines, **kw)
        self.fronts.append(f)
        return f

    ASK = {"model": "m:latest", "messages": [{"role": "user", "content": "count"}]}

    def test_stream(self):
        a = Engine([chunk("one ", think="hm"), chunk("two"),
                    chunk(None, finish="stop", timings={"prompt_n": 9, "prompt_ms": 5.0, "predicted_n": 2,
                                                         "predicted_ms": 40.0})])
        f = self.front(a)
        out = f.lines("/api/chat", self.ASK)
        self.assertEqual(f.ctype, "application/x-ndjson")
        self.assertEqual(said(out), "one two")
        self.assertEqual(out[0]["message"]["thinking"], "hm")
        self.assertEqual(out[0]["model"], "m:latest")      # the name the app used
        self.assertTrue(out[0]["created_at"].endswith("Z"))
        self.assertEqual([o["done"] for o in out], [False, False, True])
        last = out[-1]
        self.assertEqual(last["done_reason"], "stop")
        self.assertEqual((last["prompt_eval_count"], last["eval_count"], last["eval_duration"]), (9, 2, 40_000_000))
        path, sent = a.calls[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(sent["model"], "m")

    def test_think_false_asks_and_passes_thinking_on(self):
        a = Engine([chunk(think="hm"), chunk("hi"), chunk(None, finish="stop")])
        out = self.front(a).lines("/api/chat", dict(self.ASK, think=False))
        self.assertEqual(a.calls[0][1]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(out[0]["message"]["thinking"], "hm")    # a template that can't turn it off

    def test_answer_continues_after_a_move(self):
        a = Engine([chunk("one ", cid="first"), chunk("two ", cid="first")], cut=True)
        b = Engine([chunk("three", cid="second"), chunk(None, finish="length", cid="second")])
        out = self.front(a, b).lines("/api/chat", dict(self.ASK, options={"num_predict": 10}))
        self.assertEqual(said(out), "one two three")
        self.assertEqual(out[-1]["done_reason"], "length")
        self.assertTrue(all("error" not in o for o in out))
        resent = b.calls[0][1]
        self.assertEqual(resent["messages"][-1], {"role": "assistant", "content": "one two "})
        self.assertEqual(resent["max_tokens"], 8)

    def test_generate_stream_continues_after_a_move(self):
        a = Engine([chunk("Hel")], cut=True)
        b = Engine([chunk("lo"), chunk(None, finish="stop")])
        out = self.front(a, b).lines("/api/generate", {"model": "m", "prompt": "say hello"})
        self.assertEqual(said(out, "generate"), "Hello")
        self.assertTrue(out[-1]["done"])

    def test_tool_call_in_pieces(self):
        call = {"index": 0, "id": "x", "type": "function", "function": {"name": "weather", "arguments": '{"ci'}}
        more = {"index": 0, "function": {"arguments": 'ty": "Tokyo"}'}}
        out = self.front(Engine([chunk(tool=call), chunk(tool=more), chunk(None, finish="tool_calls")])).lines(
            "/api/chat", dict(self.ASK, tools=[{"type": "function", "function": {"name": "weather"}}]))
        calls = [o for o in out if o["message"].get("tool_calls")]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["message"]["tool_calls"],
                         [{"function": {"name": "weather", "arguments": {"city": "Tokyo"}}}])
        self.assertEqual(out[-1]["done_reason"], "stop")

    def test_error_in_a_stream(self):
        out = self.front(Engine([{"error": {"code": 400, "message": "too long"}}], done=False)).lines(
            "/api/chat", self.ASK)
        self.assertEqual(out, [{"error": "too long"}])
        self.assertEqual(ollama.error_text(json.dumps({"error": {"message": '{"error": {"message": "inner"}}'}})),
                         "inner")                        # an engine's error that lodger passed on
        out = self.front().lines("/api/chat", self.ASK)    # no engine comes up
        self.assertIn("not available", out[-1]["error"])

    def test_whole_chat(self):
        whole = {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": "", "reasoning_content": "hm",
                    "tool_calls": [{"id": "1", "type": "function",
                                    "function": {"name": "weather", "arguments": '{"city": "Tokyo"}'}}]}}],
                 "usage": {"prompt_tokens": 12, "completion_tokens": 5}}
        status, out = self.front(Engine(whole=whole)).post("/api/chat", dict(self.ASK, stream=False))
        self.assertEqual(status, 200)
        self.assertTrue(out["done"])
        self.assertEqual(out["message"]["tool_calls"], [{"function": {"name": "weather",
                                                                        "arguments": {"city": "Tokyo"}}}])
        self.assertEqual(out["message"]["thinking"], "hm")
        self.assertEqual((out["prompt_eval_count"], out["eval_count"]), (12, 5))

    def test_whole_generate_and_errors(self):
        whole = {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}}]}
        f = self.front(Engine(whole=whole), Engine(whole={"error": {"code": 400, "message": "bad"}}, status=400))
        status, out = f.post("/api/generate", {"model": "m", "prompt": "hello", "stream": False})
        self.assertEqual((status, out["response"], out["done_reason"]), (200, "Hi", "stop"))
        self.assertEqual(f.post("/api/generate", {"model": "m", "prompt": "x", "stream": False}), (400, {"error": "bad"}))

    def test_raw_and_infill(self):
        raw = Engine([{"choices": [{"index": 0, "text": "abc", "finish_reason": None}]},
                      {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}])
        infill = Engine([{"content": "x = 1", "stop": False}, {"content": "", "stop": True, "stop_type": "limit"}],
                        done=False)                       # llama-server's own kind ends by closing
        f = self.front(raw, infill)
        out = f.lines("/api/generate", {"model": "m", "prompt": "a", "raw": True})
        self.assertEqual((said(out, "generate"), out[-1]["done"]), ("abc", True))
        self.assertEqual(raw.calls[0][0], "/v1/completions")
        out = f.lines("/api/generate", {"model": "m", "prompt": "def f(", "suffix": ")"})
        self.assertEqual(said(out, "generate"), "x = 1")
        self.assertEqual((out[-1]["done"], out[-1]["done_reason"]), (True, "length"))
        self.assertEqual(infill.calls[0][0], "/infill")

    def test_embeddings(self):
        whole = {"data": [{"index": 1, "embedding": [0.3]}, {"index": 0, "embedding": [0.1, 0.2]}],
                 "usage": {"prompt_tokens": 4}}
        f = self.front(Engine(whole=whole), Engine(whole={"data": [{"index": 0, "embedding": [0.5]}]}))
        status, out = f.post("/api/embed", {"model": "m", "input": ["a", "b"]})
        self.assertEqual((status, out["embeddings"], out["prompt_eval_count"]), (200, [[0.1, 0.2], [0.3]], 4))
        self.assertEqual(f.post("/api/embeddings", {"model": "m", "prompt": "a"}), (200, {"embedding": [0.5]}))

    def test_load_and_unload(self):
        f = self.front(Engine())
        status, out = f.post("/api/generate", {"model": "m"})
        self.assertEqual((status, out["done"], out["done_reason"], f.waits), (200, True, "load", 1))
        status, out = f.post("/api/chat", {"model": "m", "messages": [], "keep_alive": 0})
        self.assertEqual((status, out["done"], f.waits), (200, True, 1))      # lodger unloads by itself
        self.assertNotIn("done_reason", out)

    def test_lists(self):
        f = self.front(Engine())
        self.assertEqual(f.call("GET", "/")[2], b"Ollama is running")
        self.assertEqual(f.call("HEAD", "/")[0], 200)
        status, _, data = f.call("GET", "/api/version")
        self.assertEqual(json.loads(data), {"version": ollama.VERSION})
        tags = json.loads(f.call("GET", "/api/tags")[2])["models"]
        self.assertEqual([m["name"] for m in tags], ["m:latest"])
        self.assertEqual(json.loads(f.call("GET", "/api/ps")[2]), {"models": []})
        f.post("/api/generate", {"model": "m"})
        ps = json.loads(f.call("GET", "/api/ps")[2])["models"]
        self.assertEqual((ps[0]["name"], ps[0]["size_vram"], ps[0]["expires_at"]), ("m:latest", 0, ollama.FOREVER))
        status, out = f.post("/api/show", {"name": "m:latest"})
        self.assertEqual((status, out["model_info"]["unknown.context_length"]), (200, 4096))
        self.assertEqual(f.call("GET", "/v1/models")[0], 200)     # the OpenAI-style API is there too

    def test_what_lodger_doesnt_do(self):
        f = self.front()
        for path in ("/api/pull", "/api/push", "/api/create", "/api/copy", "/api/delete", "/api/blobs/sha256:00"):
            status, out = f.post(path, {"model": "m"})
            self.assertEqual(status, 501, path)
            self.assertIn("not supported by lodger", out["error"])
        status, out = f.post("/api/chat", {"model": "llama3.2", "messages": []})
        self.assertEqual(status, 404)
        self.assertIn("llama3.2", out["error"])
        self.assertEqual(f.post("/api/chat", dict(self.ASK, options="fast"))[0], 400)
        self.assertEqual(f.post("/api/nothing", {})[0], 404)

    def test_the_same_checks_as_the_main_port(self):
        f = self.front(Engine(whole={"data": []}), key=KEY)
        status, out = f.post("/api/embed", {"model": "m", "input": "a"}, headers={"Host": "evil.example:11434"})
        self.assertEqual(status, 403)
        self.assertIn("evil.example", out["error"])      # Ollama's shape: the message as text
        status, out = f.post("/api/embed", {"model": "m", "input": "a"},
                             headers={"Origin": "https://evil.example", "Authorization": f"Bearer {KEY}"})
        self.assertEqual(status, 403)
        self.assertEqual(f.call("GET", "/api/tags")[0], 401)
        self.assertEqual(f.call("GET", "/")[0], 401)
        self.assertEqual(f.call("GET", "/api/tags", headers={"Authorization": f"Bearer {KEY}"})[0], 200)
        status, out = f.post("/api/embed", {"model": "m", "input": "a"}, headers={"Authorization": f"Bearer {KEY}"})
        self.assertEqual(status, 200)

    def test_main_port_has_no_ollama_api(self):
        f = Front([])
        main = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(f))
        threading.Thread(target=main.serve_forever, daemon=True).start()
        try:
            c = http.client.HTTPConnection("127.0.0.1", main.server_address[1], timeout=5)
            c.request("GET", "/api/tags")
            r = c.getresponse()
            self.assertEqual((r.status, json.loads(r.read())["error"]["message"]), (404, "not found"))
            c.close()
        finally:
            main.shutdown()
            main.server_close()
            f.close()


@mock.patch.object(serve, "log", lambda msg: None)
class ListenerTest(unittest.TestCase):
    def lodger(self, port):
        return SimpleNamespace(ollama=None), SimpleNamespace(ollama_port=port, api_key=None)

    def test_port_taken(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            port = s.getsockname()[1]
            lodger, a = self.lodger(port)
            self.assertTrue(serve.port_taken(port))
            self.assertIsNone(serve.ollama_listener(lodger, a))
            self.assertIn("not started", lodger.ollama)

    def test_free_port(self):
        port = serve.free_port()
        lodger, a = self.lodger(port)
        self.assertFalse(serve.port_taken(port))
        h = serve.ollama_listener(lodger, a)
        try:
            self.assertEqual(lodger.ollama, f"http://127.0.0.1:{port}")
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            c.request("GET", "/api/version")
            self.assertEqual(json.loads(c.getresponse().read()), {"version": ollama.VERSION})
            c.close()
        finally:
            h.shutdown()
            h.server_close()


if __name__ == "__main__":
    unittest.main()
