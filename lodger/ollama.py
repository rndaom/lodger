"""Ollama's API (https://github.com/ollama/ollama/blob/main/docs/api.md) in front of lodger, for apps that only
speak it (Raycast, Home Assistant, Open WebUI's Ollama mode, editor plugins). `lodger serve --ollama` answers it
on 127.0.0.1:11434; serve.py's handler does the work, so answers still continue after a move. This file only
translates: Ollama requests to the OpenAI-style ones llama-server knows, and its answers back.

Not here: pull, push, create, copy, delete (lodger serves the GGUF files in models.toml) and the model's
context (`num_ctx`: lodger decides it). `keep_alive` is ignored too: lodger loads and unloads models itself."""
import datetime
import hashlib
import json
import math
import os
import time

from .model import header, parts

VERSION = "0.12.0"          # the Ollama version lodger answers like (apps check it before using newer fields)
FOREVER = "2318-01-01T00:00:00Z"
UNSUPPORTED = ("/api/pull", "/api/push", "/api/create", "/api/copy", "/api/delete", "/api/blobs/")
# Ollama's options that llama-server knows (most by the same name); the others (num_ctx, num_gpu, num_thread,
# num_batch, use_mmap, ...) are lodger's to decide and are ignored
OPTIONS = {"num_predict": "max_tokens", "num_keep": "n_keep", "temperature": "temperature", "top_k": "top_k",
           "top_p": "top_p", "min_p": "min_p", "typical_p": "typical_p", "seed": "seed", "stop": "stop",
           "repeat_penalty": "repeat_penalty", "repeat_last_n": "repeat_last_n",
           "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty",
           "mirostat": "mirostat", "mirostat_tau": "mirostat_tau", "mirostat_eta": "mirostat_eta"}
# llama.cpp's file types (llama.h), shown as Ollama's quantization_level
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
              12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
              19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
              26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
              36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}


def now(t=None):
    """Ollama's time stamps: RFC 3339 in UTC."""
    t = datetime.datetime.fromtimestamp(time.time() if t is None else t, datetime.timezone.utc)
    return t.isoformat(timespec="microseconds").replace("+00:00", "Z")


def tagged(name):
    """A model's name as Ollama shows it (name:tag; lodger's models have only `latest`)."""
    return name if ":" in name else name + ":latest"


def plain(name):
    """The lodger model an Ollama name means: "qwen:latest" is "qwen"."""
    return name[:-len(":latest")] if isinstance(name, str) and name.endswith(":latest") else name


_files = {}


def file_info(entry):
    """(size, modified, header settings) of a model's file(s), read once per change of the file."""
    try:
        paths = parts(entry.path)
        st = [os.stat(p) for p in paths]
    except (OSError, SystemExit):
        return 0, 0, {}
    key = (entry.path, st[0].st_mtime, st[0].st_size)
    if _files.get(entry.path, (None,))[0] != key:
        _files[entry.path] = (key, (sum(s.st_size for s in st), st[0].st_mtime, header(entry.path)))
    return _files[entry.path][1]


def details(entry):
    size, _, h = file_info(entry)
    arch = h.get("general.architecture") or ""
    quant = FILE_TYPES.get(h.get("general.file_type"), "")
    if not quant:
        up = os.path.basename(entry.path).upper()
        quant = next((q for q in sorted(FILE_TYPES.values(), key=len, reverse=True) if q in up), "")
    return {"parent_model": "", "format": "gguf", "family": arch, "families": [arch] if arch else None,
            "parameter_size": h.get("general.size_label") or "", "quantization_level": quant}


def digest(entry):
    """A stable id for the file (Ollama's is the content's sha256; hashing gigabytes per listing is too slow)."""
    size, mtime, _ = file_info(entry)
    return hashlib.sha256(f"{entry.path}\0{size}\0{mtime}".encode()).hexdigest()


def listed(entry):
    size, mtime, _ = file_info(entry)
    return {"name": tagged(entry.name), "model": tagged(entry.name), "modified_at": now(mtime), "size": size,
            "digest": digest(entry), "details": details(entry)}


def capabilities(entry):
    s = entry.settings if isinstance(getattr(entry, "settings", None), dict) else {}
    if s.get("embedding") or s.get("embeddings"):
        return ["embedding"]
    return (["completion", "tools"] + (["thinking"] if entry.reasoning else [])
            + (["vision"] if s.get("mmproj") else []))


def show(entry):
    """POST /api/show. The context is lodger's (what apps may send), not the one the model was trained for."""
    _, mtime, h = file_info(entry)
    arch = h.get("general.architecture") or "unknown"
    info = {k: v for k, v in h.items() if not (isinstance(v, float) and not math.isfinite(v))}   # JSON has no NaN
    info["general.architecture"] = arch
    if entry.ctx:                                # not decided yet just after lodger started
        info[f"{arch}.context_length"] = entry.ctx
    return {"modelfile": f"# served by lodger from {entry.path}\nFROM {entry.path}\n",
            "parameters": f"num_ctx {entry.ctx}" if entry.ctx else "", "template": "",
            "details": details(entry), "model_info": info, "capabilities": capabilities(entry),
            "modified_at": now(mtime)}


def running(entry, vram, expires):
    """One model in GET /api/ps."""
    out = {**listed(entry), "expires_at": now(expires) if expires else FOREVER, "size_vram": vram}
    if entry.ctx:
        out["context_length"] = entry.ctx
    return out


def message(m, ids):
    """One Ollama chat message as llama-server wants it. Ollama's tool calls carry no ids, and some chat templates
    want them (Mistral's: 9 letters or digits): `ids` keeps a count and the ids of the last assistant message's
    calls, handed to the tool answers that follow it in order."""
    if not isinstance(m, dict):
        raise ValueError("each message must be an object")
    content = m.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("a message's 'content' must be text")
    out = {"role": m.get("role") or "user", "content": content}
    images = m.get("images") or []
    if not isinstance(images, list) or not all(isinstance(i, str) for i in images):
        raise ValueError("'images' must be a list of base64 texts")
    if images:
        out["content"] = [{"type": "text", "text": content}] + [
            {"type": "image_url", "image_url": {"url": i if i.startswith("data:") else "data:image/png;base64," + i}}
            for i in images]
    if out["role"] == "assistant":
        if m.get("thinking"):
            out["reasoning_content"] = m["thinking"]
        calls = m.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("'tool_calls' must be a list")
        ids["last"] = []
        if calls:
            out["tool_calls"] = []
            for c in calls:
                c = c if isinstance(c, dict) else {}
                f = c.get("function") if isinstance(c.get("function"), dict) else {}
                args = f.get("arguments", {})
                ids["n"] = ids.get("n", 0) + 1
                cid = c.get("id") or f"{ids['n']:09d}"
                ids["last"].append(cid)
                out["tool_calls"].append({"id": cid, "type": "function", "function": {
                    "name": f.get("name") or "", "arguments": args if isinstance(args, str) else json.dumps(args)}})
    elif out["role"] == "tool":
        last = ids.get("last") or []
        cid = m.get("tool_call_id") or (last.pop(0) if last else None)
        if cid:
            out["tool_call_id"] = cid
        if m.get("tool_name"):
            out["name"] = m["tool_name"]
    return out


def sampling(body, req, chat=True):
    """Ollama's options, format and think, as llama-server fields in `req`."""
    opts = body.get("options") or {}
    if not isinstance(opts, dict):
        raise ValueError("'options' must be an object")
    for k, v in opts.items():
        if k not in OPTIONS or v is None:
            continue
        if k == "num_predict":
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:   # -1, -2: as long as it takes
                req["max_tokens"] = v
        elif k == "stop":
            req["stop"] = [v] if isinstance(v, str) else v
        else:
            req[OPTIONS[k]] = v
    fmt = body.get("format")
    if fmt == "json":
        if chat:
            req["response_format"] = {"type": "json_object"}
        else:
            req["json_schema"] = {"type": "object"}
    elif isinstance(fmt, dict):
        if chat:
            req["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": fmt}}
        else:
            req["json_schema"] = fmt
    elif fmt not in (None, ""):
        raise ValueError("'format' must be \"json\" or a JSON schema")
    think = body.get("think")
    if chat and isinstance(think, bool):
        req["chat_template_kwargs"] = {"enable_thinking": think}
    elif chat and isinstance(think, str) and think:
        req["reasoning_effort"] = think                  # "low", "medium", "high"
    return req


def request(path, body, name):
    """(llama-server path, OpenAI-style request, Answer) for an Ollama request; the request is None when it
    only asks for the model to be loaded (an empty prompt). Raises ValueError for a request that's wrong."""
    model = body.get("model") or tagged(name)
    if path in ("/api/embed", "/api/embeddings"):
        inp = body.get("input") if path == "/api/embed" else body.get("prompt")
        if inp is None and path == "/api/embed":
            return "/v1/embeddings", None, Answer("embed", model)       # Ollama's way to load the model
        if isinstance(inp, list):
            inp = [i.get("text", "") if isinstance(i, dict) else i for i in inp]
        if not (isinstance(inp, str) or isinstance(inp, list) and all(isinstance(i, str) for i in inp)):
            raise ValueError("'input' must be text or a list of texts" if path == "/api/embed" else
                             "'prompt' must be text")
        return "/v1/embeddings", {"model": name, "input": inp}, Answer(path[5:], model)
    stream = True if body.get("stream") is None else body["stream"]
    if not isinstance(stream, bool):
        raise ValueError("'stream' must be true or false")
    req = {"model": name, "stream": stream}
    if path == "/api/chat":
        msgs = body.get("messages") or []
        if not isinstance(msgs, list):
            raise ValueError("'messages' must be a list")
        tools = body.get("tools")
        if tools is not None and not isinstance(tools, list):
            raise ValueError("'tools' must be a list")
        ids = {}
        req["messages"] = [message(m, ids) for m in msgs]
        if tools:
            req["tools"] = tools
        return "/v1/chat/completions", sampling(body, req) if msgs else None, Answer("chat", model)
    prompt, suffix, system = body.get("prompt") or "", body.get("suffix") or "", body.get("system") or ""
    images = body.get("images") or []
    if not all(isinstance(v, str) for v in (prompt, suffix, system)):
        raise ValueError("'prompt', 'suffix' and 'system' must be text")
    if not (prompt or suffix or images):
        return "/v1/chat/completions", None, Answer("generate", model)
    if suffix:                                  # fill in the middle (code models)
        req.update(input_prefix=prompt, input_suffix=suffix, prompt="")
        return "/infill", sampling(body, req, chat=False), Answer("generate", model)
    if body.get("raw"):                         # the app wrote the whole prompt, template and all
        req["prompt"] = prompt
        return "/v1/completions", sampling(body, req, chat=False), Answer("generate", model)
    req["messages"] = ([{"role": "system", "content": system}] if system else []) + [
        message({"role": "user", "content": prompt, "images": images}, {})]
    return "/v1/chat/completions", sampling(body, req), Answer("generate", model)


def error_text(data, depth=0):
    """The message in an error answer from llama-server (or the text itself; lodger may have wrapped one)."""
    try:
        d = json.loads(data)
    except ValueError:
        return data.decode(errors="replace") if isinstance(data, bytes) else str(data)
    e = d.get("error", d) if isinstance(d, dict) else d
    msg = str(e.get("message", e) if isinstance(e, dict) else e)
    return error_text(msg, 1) if not depth and msg.lstrip().startswith("{") else msg


def arguments(a):
    """A tool call's arguments as Ollama gives them: an object (llama-server writes JSON text)."""
    if isinstance(a, str):
        try:
            v = json.loads(a or "{}")
            return v if isinstance(v, dict) else a
        except ValueError:
            return a
    return a


class Answer:
    """One answer in Ollama's shape: kind is "chat", "generate", "embed" or "embeddings". feed() takes the
    engine's stream line by line (as lodger passes it on) and gives the NDJSON objects for the app; whole()
    turns a whole answer around. Thinking is passed on even after think: false (a model whose template can't
    turn it off still thinks, and its answer comes after that)."""

    def __init__(self, kind, model):
        self.kind, self.model = kind, model
        self.t0, self.first, self.count = time.time(), None, 0
        self.calls, self.reason, self.timings, self.usage = {}, None, {}, {}
        self.ended = False

    def piece(self, text="", thinking="", calls=None, done=False):
        out = {"model": self.model, "created_at": now()}
        if self.kind == "chat":
            out["message"] = {"role": "assistant", "content": text}
            if thinking:
                out["message"]["thinking"] = thinking
            if calls:
                out["message"]["tool_calls"] = calls
        else:
            out["response"] = text
            if thinking:
                out["thinking"] = thinking
        out["done"] = done
        return out

    def stats(self):
        t, u, end = self.timings, self.usage, time.time()
        first = self.first or end
        n = int(t.get("predicted_n") or u.get("completion_tokens") or 0)
        return {"total_duration": int((end - self.t0) * 1e9), "load_duration": 0,
                "prompt_eval_count": int(t.get("prompt_n") or u.get("prompt_tokens") or 0),
                "prompt_eval_duration": int(t["prompt_ms"] * 1e6 if t.get("prompt_ms") else (first - self.t0) * 1e9),
                "eval_count": max(self.count, n),
                # after a move the engine's own numbers cover only the last part
                "eval_duration": int(t["predicted_ms"] * 1e6 if t.get("predicted_ms") and self.count <= n
                                     else (end - first) * 1e9)}

    def end(self):
        self.ended = True
        out = []
        if self.calls:
            out.append(self.piece(calls=[{"function": {"name": c["name"], "arguments": arguments(c["args"])}}
                                         for _, c in sorted(self.calls.items())]))
        final = self.piece(done=True)
        final["done_reason"] = "length" if self.reason == "length" else "stop"
        return out + [{**final, **self.stats()}]

    def feed(self, s):
        """One line of the engine's stream (as serve.py sends it on) -> the objects to send to the app."""
        if self.ended:
            return []
        if s.startswith(":"):                    # lodger's pings: keep the app's connection open
            return [self.piece()]
        if s == "data: [DONE]":
            return self.end()
        if not s.startswith("data:"):
            return []
        try:
            d = json.loads(s[5:])
        except ValueError:
            return []
        if not isinstance(d, dict):
            return []
        if "error" in d:
            self.ended = True
            return [{"error": error_text(json.dumps(d))}]
        try:
            return self.read(d)
        except (AttributeError, TypeError, ValueError):     # not what llama-server sends: left out
            return []

    def read(self, d):
        self.timings = d.get("timings") or self.timings
        self.usage = d.get("usage") or self.usage
        if "choices" in d:
            choice = (d.get("choices") or [{}])[0]
            choice = choice if isinstance(choice, dict) else {}
            delta = choice.get("delta") or {}
            text = delta.get("content") or choice.get("text") or ""
            think = delta.get("reasoning_content") or ""
            for c in delta.get("tool_calls") or []:
                call = self.calls.setdefault(c.get("index", len(self.calls)), {"name": "", "args": ""})
                f = c.get("function") or {}
                call["name"] = f.get("name") or call["name"]
                call["args"] += f.get("arguments") or ""
            self.reason = choice.get("finish_reason") or self.reason
        else:                                    # llama-server's own kind (/infill)
            text, think = d.get("content") or "", ""
            if d.get("stop"):
                self.reason = "length" if d.get("stop_type") == "limit" else "stop"
        if not (text or think):
            return []
        self.first = self.first or time.time()
        self.count += 1
        return [self.piece(text, think)]

    def close(self):
        """The engine's stream ended without [DONE] (llama-server's own kinds end so): the last object."""
        return [] if self.ended else self.end()

    def whole(self, status, data):
        """(HTTP status, object) for the app from a whole answer of llama-server."""
        if status != 200:
            return status, {"error": error_text(data)}
        try:
            d = json.loads(data)
        except ValueError:
            return 502, {"error": "the model's engine sent an answer lodger can't read"}
        try:
            return self.read_whole(d)
        except (AttributeError, TypeError, ValueError):
            return 502, {"error": "the model's engine sent an answer lodger can't read"}

    def read_whole(self, d):
        self.usage, self.timings = d.get("usage") or {}, d.get("timings") or {}
        if self.kind in ("embed", "embeddings"):
            vecs = [x.get("embedding") for x in sorted(d.get("data") or [], key=lambda x: x.get("index", 0))]
            if self.kind == "embeddings":
                return 200, {"embedding": vecs[0] if vecs else []}
            return 200, {"model": self.model, "embeddings": vecs, "total_duration": int((time.time() - self.t0) * 1e9),
                         "load_duration": 0, "prompt_eval_count": int(self.usage.get("prompt_tokens") or 0)}
        if "choices" in d:
            choice = (d.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or choice.get("text") or ""
            think = msg.get("reasoning_content") or ""
            calls = [{"function": {"name": (c.get("function") or {}).get("name", ""),
                                   "arguments": arguments((c.get("function") or {}).get("arguments", "{}"))}}
                     for c in msg.get("tool_calls") or []]
            self.reason = choice.get("finish_reason")
        else:                                    # /infill
            text, think, calls = d.get("content") or "", "", []
            self.reason = "length" if d.get("stop_type") == "limit" else "stop"
        self.first = self.t0 + (self.timings.get("prompt_ms") or 0) / 1000
        out = self.piece(text, think, calls, done=True)
        out["done_reason"] = "length" if self.reason == "length" else "stop"
        return 200, {**out, **self.stats()}

    def loaded(self, load=True):
        """The answer to a request with nothing to write: Ollama's way to load a model, or with keep_alive 0 to
        unload it (lodger unloads models itself, with --idle)."""
        if self.kind == "embed":
            return {"model": self.model, "embeddings": []}
        out = self.piece(done=True)
        if load:
            out["done_reason"] = "load"
        return out
