"""One answer in the shape of the API an app speaks, kept whole across moves of the model.

lodger always asks llama-server for a streamed answer (also when the app wants it whole), so that what was written
before a move is known: the next engine is asked to continue it (the partial answer goes in as the last assistant
message, which llama-server continues), and its events are joined onto the first engine's so the app sees one
answer. Three shapes: OpenAI's chat completions (Chat), Anthropic's Messages (Messages) and OpenAI's Responses
(Responses). Without a move the engine's events reach a streaming app as they are.

An answer object is fed the engine's events (feed) and gives what to send to the app: event texts for a streaming
app, or at the end the whole answer (whole). serve.py's handler does the talking; nothing here touches a socket.
Token counts after a move are lodger's count of the pieces written before it plus the engine's own after it (one
piece is about one token)."""
import json
import time

LIMITS = ("max_tokens", "max_completion_tokens", "n_predict")


def positive_int(v):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class Event:
    """One server-sent event: `name` (None for a plain `data:` event, ":" for a comment) and its `data`."""
    __slots__ = ("name", "data")

    def __init__(self, name, data):
        self.name, self.data = name, data

    def text(self):
        """As it is sent (without the blank line that ends it)."""
        if self.name == ":":
            return self.data
        return (f"event: {self.name}\n" if self.name else "") + "\n".join("data: " + d for d in self.data.split("\n"))


def events(readline):
    """The engine's server-sent events, read with `readline()`, until it closes. A line cut off in the middle (the
    engine went away while writing it) raises OSError: it is never passed on."""
    name, data = None, []
    while True:
        line = readline()
        if not line:
            if name is not None or data:
                raise OSError("stream ended early")
            return
        if not line.endswith(b"\n"):
            raise OSError("stream ended early")
        s = line.decode(errors="replace").rstrip("\r\n")
        if not s:
            if name is not None or data:
                yield Event(name, "\n".join(data))
            name, data = None, []
        elif s.startswith(":"):
            yield Event(":", s)
        else:
            field, _, value = s.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)


def parse(data):
    """An event's JSON object; OSError for anything else (a broken engine, never passed on as is)."""
    try:
        d = json.loads(data)
    except ValueError:
        raise OSError("broken event from the engine") from None
    if not isinstance(d, dict):
        raise OSError("broken event from the engine")
    return d


def error_message(body):
    """The message in an engine's error answer (bytes or text, any of the APIs' shapes), else the text itself."""
    text = body.decode(errors="replace") if isinstance(body, bytes) else str(body)
    try:
        d = json.loads(text)
    except ValueError:
        return text.strip()[:500]
    e = d.get("error", d) if isinstance(d, dict) else d
    if isinstance(e, dict):
        return str(e.get("message") or e)
    return str(e)


OPENAI_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
                404: "not_found_error", 413: "invalid_request_error", 501: "not_supported_error",
                503: "unavailable_error"}


def openai_error(status, msg):
    return {"error": {"message": msg, "type": OPENAI_TYPES.get(status, "server_error"), "code": status}}


ANTHROPIC_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
                   404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
                   503: "overloaded_error", 529: "overloaded_error"}


def anthropic_error(status, msg):
    return {"type": "error", "error": {"type": ANTHROPIC_TYPES.get(status, "api_error"), "message": msg}}


class Answer:
    """What the three shapes share. Subclasses set `path` (llama-server's) and the shape's parts."""
    path = None
    limit_keys = LIMITS

    def __init__(self, body, stream):
        self.body, self.stream = body, stream
        self.limit = next((positive_int(body[k]) for k in self.limit_keys if k in body), None)
        self.pieces = 0             # pieces of text or thinking written so far, over all engines
        self.before = 0             # ... before the engine now writing started
        self.attempt = 0
        self.done = False           # the engine's last event came (or an error): nothing more to read
        self.finished = False       # the answer is complete; only the end of the stream may be missing
        self.tooling = False        # a tool call is being written (half a call can't be continued)
        self.failure = None         # (status, error object) the engine sent in the middle of a whole answer
        self.t0 = time.time()
        self.writing_ms = 0.0       # time the earlier engines spent writing their part
        self.t_first = self.t_last = None

    def wrote(self):
        """A piece of text or thinking came: count it and its writing time."""
        self.pieces += 1
        now = time.time()
        self.t_first = self.t_first or now
        self.t_last = now

    def request(self, attempt):
        """The body for the engine (always streamed) for this attempt; None if the answer already reached its
        length limit (then close())."""
        if attempt:
            if self.limit and self.limit - self.pieces <= 0:
                return None
            if self.t_first:
                self.writing_ms += (self.t_last - self.t_first) * 1000
            self.t_first = self.t_last = None
        self.attempt, self.before = attempt, self.pieces
        req = dict(self.body)
        req["stream"] = True
        if attempt:
            if self.limit:
                for k in self.limit_keys:
                    if k in req:
                        req[k] = self.limit - self.pieces
            self.continue_from(req)
        return req

    def reset(self):
        """Start over (a whole answer that can't be continued: the app hasn't seen any of it)."""
        type(self).__init__(self, self.body, self.stream)

    def cut(self):
        """Why the answer can't be continued after a move (None: it can)."""
        if self.tooling:
            return "the model moved while writing a tool call; please retry"
        return None

    def usage(self, u, prompt_key, answer_key, total_key="total_tokens"):
        """Token counts of a continued answer: the pieces written before the move count as the answer's, not as
        the conversation the next engine read."""
        if not (isinstance(u, dict) and self.before):
            return u
        u = dict(u)
        if isinstance(u.get(answer_key), int):
            u[answer_key] += self.before
        if isinstance(u.get(prompt_key), int):
            u[prompt_key] = max(0, u[prompt_key] - self.before)
            if total_key and isinstance(u.get(total_key), int):
                u[total_key] = u[prompt_key] + u.get(answer_key, 0)
        return u

    def timings(self, t):
        """llama-server's timings for a continued answer: the writing numbers cover the whole answer."""
        if not (isinstance(t, dict) and self.before and isinstance(t.get("predicted_n"), (int, float))):
            return t
        t = dict(t)
        n = t["predicted_n"] + self.before
        ms = (t.get("predicted_ms") or 0) + self.writing_ms
        t.update(predicted_n=n, predicted_ms=ms)
        if n and ms:
            t.update(predicted_per_token_ms=ms / n, predicted_per_second=n * 1000 / ms)
        return t


class Chat(Answer):
    """OpenAI's chat completions (/v1/chat/completions)."""
    path = "/v1/chat/completions"

    def __init__(self, body, stream):
        super().__init__(body, stream)
        rf = body.get("response_format")
        self.structured = bool(body.get("json_schema") or body.get("grammar")
                               or isinstance(rf, dict) and rf.get("type", "text") != "text")
        self.several = (positive_int(body.get("n")) or 1) > 1
        self.ended = set()          # several answers: the ones that are complete
        msgs = list(body.get("messages") or [])
        last = msgs[-1] if msgs and isinstance(msgs[-1], dict) else {}
        # the app's own start of the answer (a trailing assistant message) is continued, not repeated
        self.prefill = msgs.pop() if (last.get("role") == "assistant" and isinstance(last.get("content"), str)
                                      and not last.get("tool_calls")) else None
        self.msgs = msgs
        self.content = self.reasoning = ""
        self.first = {}             # id, created, model and system_fingerprint of the answer's first piece
        self.calls = {}             # tool calls by index: id, name, arguments
        self.reason = None
        self.last_usage = self.last_timings = None

    @staticmethod
    def wanted_plain(body):
        """Answers lodger passes on as they are (no continuation): several at once, or with token probabilities."""
        return ((positive_int(body.get("n")) or 1) > 1 or bool(body.get("logprobs")) or bool(body.get("n_probs"))
                or body.get("top_logprobs") is not None)

    def request(self, attempt):
        req = super().request(attempt)
        if req is not None and not self.stream:
            req["stream_options"] = {"include_usage": True}    # a whole answer has its token counts
        return req

    def continue_from(self, req):
        if self.content or self.reasoning:
            pre = self.prefill or {}
            partial = {"role": "assistant", "content": pre.get("content", "") + self.content}
            if self.reasoning or pre.get("reasoning_content"):
                partial["reasoning_content"] = (pre.get("reasoning_content") or "") + self.reasoning
            req["messages"] = self.msgs + [partial]

    def cut(self):
        if self.several and self.first:
            return "the model moved while writing several answers; please retry"
        if self.structured and (self.content or self.reasoning):
            return "the model moved while writing a structured answer; please retry"
        return super().cut()

    def moved_note(self, where):
        return f": lodger restarted the model on the {where}; continuing the answer"

    def feed(self, ev):
        if ev.name == ":":
            return [ev.data] if self.stream else []          # pings
        if ev.data == "[DONE]":
            self.done = True
            return ["data: [DONE]"] if self.stream else []
        d = parse(ev.data)
        if "error" in d:
            self.done = True
            if not self.stream:
                e = d["error"]
                code = e.get("code") if isinstance(e, dict) else None
                self.failure = (code if isinstance(code, int) and 400 <= code < 600 else 500, d)
            return ["data: " + ev.data] if self.stream else []
        changed = False
        choices = d.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        if self.several:                            # several answers: passed on as they are, never continued
            self.first = self.first or {k: d.get(k) for k in ("id", "created", "model", "system_fingerprint")}
            if choice.get("finish_reason"):
                self.ended.add(choice.get("index"))
                self.finished = len(self.ended) >= positive_int(self.body.get("n"))
            return ["data: " + ev.data] if self.stream else []
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        text = delta.get("content") if isinstance(delta.get("content"), str) else ""
        think = delta.get("reasoning_content") if isinstance(delta.get("reasoning_content"), str) else ""
        if not self.first:
            self.first = {k: d.get(k) for k in ("id", "created", "model", "system_fingerprint")}
        elif self.attempt:
            if set(delta) <= {"role"} and not choice.get("finish_reason") and not d.get("usage"):
                return []                           # the next engine's "assistant here" piece: said already
            d.update({k: v for k, v in self.first.items() if v is not None})   # one answer to the app
            changed = True
        if self.before and (d.get("usage") or d.get("timings")):
            d["usage"] = self.usage(d.get("usage"), "prompt_tokens", "completion_tokens")
            d["timings"] = self.timings(d.get("timings"))
            d = {k: v for k, v in d.items() if v is not None}
            changed = True
        if text or think:
            self.wrote()
        self.content += text
        self.reasoning += think
        for c in delta.get("tool_calls") or []:
            if isinstance(c, dict):
                self.tooling = True
                call = self.calls.setdefault(c.get("index", len(self.calls)), {"id": None, "name": "", "args": ""})
                f = c.get("function") if isinstance(c.get("function"), dict) else {}
                call["id"] = c.get("id") or call["id"]
                call["name"] += f.get("name") or ""
                call["args"] += f.get("arguments") or ""
        if choice.get("finish_reason"):
            self.finished, self.reason = True, choice["finish_reason"]
        self.last_usage = d.get("usage") or self.last_usage
        self.last_timings = d.get("timings") or self.last_timings
        if not self.stream:
            return []
        return ["data: " + (compact(d) if changed else ev.data)]

    def close(self, entry_name, why=None):
        """End a streamed answer now: at its length limit (why="length"), or after an engine stopped once the
        answer was complete."""
        if self.done:
            return []
        self.done = True
        if why:
            self.reason = self.reason or why
            if not self.stream:
                return []
            return ["data: " + compact({"id": self.first.get("id") or "chatcmpl-lodger",
                                        "object": "chat.completion.chunk",
                                        "created": self.first.get("created") or int(time.time()),
                                        "model": self.first.get("model") or entry_name,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": why}]}),
                    "data: [DONE]"]
        return ["data: [DONE]"] if self.stream else []

    def whole(self, entry_name):
        """The whole answer, as llama-server gives one."""
        msg = {"role": "assistant", "content": self.content}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.calls:
            msg["tool_calls"] = []
            for _, c in sorted(self.calls.items()):
                tc = {"type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
                if c["id"]:
                    tc["id"] = c["id"]
                msg["tool_calls"].append(tc)
        out = {"choices": [{"finish_reason": self.reason or "length", "index": 0, "message": msg}],
               "created": self.first.get("created") or int(self.t0), "model": self.first.get("model") or entry_name,
               "system_fingerprint": self.first.get("system_fingerprint"), "object": "chat.completion",
               "usage": self.last_usage, "id": self.first.get("id") or "chatcmpl-lodger"}
        if self.last_timings:
            out["timings"] = self.last_timings
        return {k: v for k, v in out.items() if v is not None}

    def error(self, status, msg):
        return openai_error(status, msg)

    def stream_error(self, msg):
        return ["data: " + compact({"error": {"message": msg, "type": "unavailable_error", "code": 503}})]


KINDS = {"text_delta": "text", "thinking_delta": "thinking", "input_json_delta": "tool_use"}   # delta -> block


class Messages(Answer):
    """Anthropic's Messages API (/v1/messages): content blocks (thinking, text, tool_use) in numbered events.
    After a move the next engine's message_start is left out, its first block joins the cut one if it is of the
    same kind, its block numbers follow on from the first engine's, and blocks it didn't close are closed."""
    path = "/v1/messages"
    limit_keys = ("max_tokens",)

    def __init__(self, body, stream):
        super().__init__(body, stream)
        msgs = list(body.get("messages") or [])
        last = msgs[-1] if msgs and isinstance(msgs[-1], dict) else {}
        self.pre_text, self.pre_thinking, self.pre_other = "", "", False
        if last.get("role") == "assistant":           # the app's own start of the answer
            msgs.pop()
            c = last.get("content")
            for b in ([{"type": "text", "text": c}] if isinstance(c, str) else c if isinstance(c, list) else []):
                t = b.get("type") if isinstance(b, dict) else None
                if t == "text" and isinstance(b.get("text"), str):
                    self.pre_text += b["text"]
                elif t == "thinking" and isinstance(b.get("thinking"), str):
                    self.pre_thinking += b["thinking"]
                else:
                    self.pre_other = True              # e.g. a tool call: can't be joined to a continuation
            if self.pre_other:
                msgs.append(last)
        self.msgs = msgs
        self.start = None           # the first engine's message (message_start)
        self.blocks = []            # our blocks: type, open, text / thinking / json, id, name
        self.map = {}               # the engine now writing: its block number -> ours
        self.cont = None            # our block the next engine's first block continues
        self.stop = {}              # stop_reason and stop_sequence
        self.out_tokens = None

    def text_of(self, kind):
        return "".join(b.get(kind, "") for b in self.blocks if b["type"] == kind)

    def continue_from(self, req):
        last = self.blocks[-1] if self.blocks else None
        self.cont = len(self.blocks) - 1 if last and last["open"] and last["type"] in ("text", "thinking") else None
        self.map = {}
        thinking, text = self.text_of("thinking"), self.text_of("text")
        if not (thinking or text):
            return
        content = []
        if thinking or self.pre_thinking:
            content.append({"type": "thinking", "thinking": self.pre_thinking + thinking, "signature": ""})
        content.append({"type": "text", "text": self.pre_text + text})
        req["messages"] = self.msgs + [{"role": "assistant", "content": content if len(content) > 1 else
                                        content[0]["text"]}]

    def cut(self):
        if self.pre_other and self.blocks:
            return "the model moved while answering after a tool call of the app's own; please retry"
        return super().cut()

    def moved_note(self, where):
        return f": lodger restarted the model on the {where}; continuing the answer"

    @staticmethod
    def event(name, d):
        return f"event: {name}\ndata: {compact(d)}"

    def latest(self, kind):
        """Our newest block of this kind the engine now writing has started or continued (None: none)."""
        mine = [o for o in self.map.values() if self.blocks[o]["type"] == kind]
        return max(mine) if mine else None

    def closing(self, our):
        """Events that close our block `our` (a thinking block gets its empty signature first, as llama-server
        sends it)."""
        b = self.blocks[our]
        b["open"] = False
        out = []
        if b["type"] == "thinking":
            out.append(self.event("content_block_delta", {"type": "content_block_delta", "index": our,
                                                          "delta": {"type": "signature_delta", "signature": ""}}))
        return out + [self.event("content_block_stop", {"type": "content_block_stop", "index": our})]

    def feed(self, ev):
        if ev.name == ":":
            return [ev.data] if self.stream else []
        d = parse(ev.data)
        kind = ev.name or d.get("type")
        same = [ev.text()] if self.stream else []
        if kind == "error":
            self.done = True
            if not self.stream:
                self.failure = (500, d if d.get("type") == "error" else anthropic_error(500, error_message(ev.data)))
            return same
        if kind == "message_start":
            if self.start is None:
                self.start = d.get("message") if isinstance(d.get("message"), dict) else {}
                return same
            return []                                   # a continuation: the app has its start already
        if kind == "content_block_start":
            i, cb = d.get("index"), d.get("content_block") if isinstance(d.get("content_block"), dict) else {}
            t, out = cb.get("type"), []
            if self.attempt and self.cont is not None and not self.map and self.blocks[self.cont]["type"] == t:
                self.map[i] = self.cont                 # the cut block goes on
                return []
            if self.attempt and self.cont is not None and self.cont not in self.map.values() \
                    and self.blocks[self.cont]["open"]:
                out += self.closing(self.cont)          # the cut block ended there
                self.cont = None
            our = len(self.blocks)
            self.blocks.append({"type": t, "open": True, "id": cb.get("id"), "name": cb.get("name")})
            self.map[i] = our
            if t == "tool_use":
                self.tooling = True
            if our != i and self.attempt:
                d["index"] = our
                out.append(self.event(ev.name or kind, d))
            else:
                out.append(ev.text())
            return out if self.stream else []
        if kind in ("content_block_delta", "content_block_stop"):
            delta = d.get("delta") if isinstance(d.get("delta"), dict) else {}
            closes = kind == "content_block_stop" or delta.get("type") == "signature_delta"
            if self.attempt and closes:
                # after a move lodger closes the blocks itself (at message_delta): the engine numbers its closing
                # events counting the thinking it was given, not the blocks it streamed
                return []
            our = self.map.get(d.get("index"))
            if our is None or (not closes and self.blocks[our]["type"] != KINDS.get(delta.get("type"))):
                our = self.latest(KINDS.get(delta.get("type")))
            if our is None:
                return same if not self.attempt else []
            b = self.blocks[our]
            if closes:
                b["open"] = False
            else:
                field = {"text_delta": "text", "thinking_delta": "thinking", "input_json_delta": "partial_json"}.get(
                    delta.get("type"))
                if field and isinstance(delta.get(field), str):
                    name = "json" if field == "partial_json" else field
                    b[name] = b.get(name, "") + delta[field]
                    if field != "partial_json" and delta[field]:
                        self.wrote()
            if our != d.get("index") and self.attempt:
                d["index"] = our
                return [self.event(ev.name or kind, d)] if self.stream else []
            return same
        if kind == "message_delta":
            out = []
            if self.attempt:
                for our, b in enumerate(self.blocks):
                    if b["open"]:
                        out += self.closing(our)        # in order, as llama-server closes them
            self.finished = True
            self.stop = d.get("delta") if isinstance(d.get("delta"), dict) else {}
            u = d.get("usage")
            if self.before and isinstance(u, dict) and isinstance(u.get("output_tokens"), int):
                d["usage"] = dict(u, output_tokens=u["output_tokens"] + self.before)
                self.out_tokens = d["usage"]["output_tokens"]
                return (out + [self.event(ev.name or kind, d)]) if self.stream else []
            self.out_tokens = u.get("output_tokens") if isinstance(u, dict) else None
            return (out + same) if self.stream else []
        if kind == "message_stop":
            self.done = True
        return same

    def close(self, entry_name, why=None):
        if self.done:
            return []
        self.done = True
        out = []
        if not self.finished:
            for our, b in enumerate(self.blocks):
                if b["open"]:
                    out += self.closing(our)
            self.stop = {"stop_reason": "max_tokens" if why == "length" else "end_turn", "stop_sequence": None}
            self.out_tokens = self.pieces
            out.append(self.event("message_delta", {"type": "message_delta", "delta": self.stop,
                                                    "usage": {"output_tokens": self.pieces}}))
        out.append(self.event("message_stop", {"type": "message_stop"}))
        return out if self.stream else []

    def whole(self, entry_name):
        start = self.start or {}
        content = []
        for b in self.blocks:
            if b["type"] == "thinking":
                content.append({"type": "thinking", "thinking": b.get("thinking", ""), "signature": ""})
            elif b["type"] == "text":
                content.append({"type": "text", "text": b.get("text", "")})
            elif b["type"] == "tool_use":
                try:
                    args = json.loads(b.get("json") or "{}")
                except ValueError:
                    args = {}
                content.append({"type": "tool_use", "id": b.get("id"), "name": b.get("name"),
                                "input": args if isinstance(args, dict) else {}})
        usage = dict(start.get("usage") or {})
        usage["output_tokens"] = self.out_tokens if self.out_tokens is not None else self.pieces
        return {"id": start.get("id") or "msg_lodger", "type": "message", "role": "assistant", "content": content,
                "model": start.get("model") or entry_name, "stop_reason": self.stop.get("stop_reason") or "max_tokens",
                "stop_sequence": self.stop.get("stop_sequence"), "usage": usage}

    def error(self, status, msg):
        return anthropic_error(status, msg)

    def stream_error(self, msg):
        return [self.event("error", anthropic_error(503, msg))]


DELTAS = {"response.output_text.delta": "message", "response.reasoning_text.delta": "reasoning",
          "response.function_call_arguments.delta": "function_call"}   # delta event -> item type


class Responses(Answer):
    """OpenAI's Responses API (/v1/responses): output items (reasoning, message, function_call) in events named
    response.*. After a move the next engine's response.created/in_progress are left out, its first item joins
    the cut one if it is of the same kind (its ids and numbers become ours), and the closing events carry the
    whole answer's text."""
    path = "/v1/responses"
    limit_keys = ("max_output_tokens",)

    def __init__(self, body, stream):
        super().__init__(body, stream)
        inp = body.get("input")
        items = [{"type": "message", "role": "user", "content": inp}] if isinstance(inp, str) else \
            list(inp) if isinstance(inp, list) else []
        last = items[-1] if items and isinstance(items[-1], dict) else {}
        self.pre_text = ""
        if last.get("role") == "assistant" and last.get("type", "message") == "message":
            c = last.get("content")
            parts = [{"type": "output_text", "text": c}] if isinstance(c, str) else c if isinstance(c, list) else []
            if all(isinstance(p, dict) and p.get("type") in ("output_text", "input_text")
                   and isinstance(p.get("text"), str) for p in parts):
                items.pop()                             # the app's own start of the answer: continued, not repeated
                self.pre_text = "".join(p["text"] for p in parts)
        self.items = items
        self.resp = None            # the first engine's response (id, ...)
        self.out = []               # our output items: type, id, text, done, and the engine's item fields
        self.map = {}               # the engine now writing: its output_index -> ours
        self.ids = {}               # ... its item ids -> ours
        self.cont = None
        self.completed = None

    def continue_from(self, req):
        last = self.out[-1] if self.out else None
        self.cont = len(self.out) - 1 if last and not last["done"] and last["type"] in ("reasoning", "message") else None
        self.map, self.ids = {}, {}
        reasoning = "".join(o["text"] for o in self.out if o["type"] == "reasoning")
        text = "".join(o["text"] for o in self.out if o["type"] == "message")
        if not (reasoning or text):
            return
        extra = []
        if reasoning:
            extra.append({"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": reasoning}]})
        extra.append({"type": "message", "role": "assistant",
                      "content": [{"type": "output_text", "text": self.pre_text + text}]})
        req["input"] = self.items + extra

    def moved_note(self, where):
        return f": lodger restarted the model on the {where}; continuing the answer"

    @staticmethod
    def event(name, d):
        return f"event: {name}\ndata: {compact(d)}"

    def full(self, o):
        """Our item `o` as a finished output item, with all of its text."""
        item = dict(o["item"])
        item["id"], item["status"] = o["id"], "completed"
        if o["type"] == "reasoning":
            item["content"] = [{"text": o["text"], "type": "reasoning_text"}]
        elif o["type"] == "message":
            item["content"] = [{"type": "output_text", "annotations": [], "logprobs": [], "text": o["text"]}]
        elif o["type"] == "function_call":
            item["arguments"] = o["text"]
        return item

    def finishing(self, our):
        """The closing events of our item `our` (as llama-server sends them at the end)."""
        o = self.out[our]
        o["done"] = True
        out = []
        if o["type"] == "message":
            part = {"type": "output_text", "annotations": [], "logprobs": [], "text": o["text"]}
            out.append(self.event("response.output_text.done", {
                "type": "response.output_text.done", "item_id": o["id"], "output_index": our, "content_index": 0,
                "text": o["text"]}))
            out.append(self.event("response.content_part.done", {
                "type": "response.content_part.done", "item_id": o["id"], "output_index": our, "content_index": 0,
                "part": part}))
        out.append(self.event("response.output_item.done", {"type": "response.output_item.done",
                                                            "output_index": our, "item": self.full(o)}))
        return out

    def ours(self, d, t=None):
        """Our item for an event of the engine now writing: by its output_index, else by its item id, else (after a
        move) its newest item of type `t`. None: no such item."""
        def fits(o):
            return o is not None and (not t or self.out[o]["type"] == t)
        our = self.map.get(d.get("output_index"))
        if not fits(our):
            iid = self.ids.get(d.get("item_id"), d.get("item_id"))
            our = next((o for o in self.map.values() if iid and self.out[o]["id"] == iid), None)
        if not fits(our) and self.attempt and t:
            mine = [o for o in self.map.values() if self.out[o]["type"] == t]
            our = max(mine) if mine else None
        return our if fits(our) else None

    def feed(self, ev):
        if ev.name == ":":
            return [ev.data] if self.stream else []
        d = parse(ev.data)
        kind = ev.name or d.get("type")
        same = [ev.text()] if self.stream else []
        if kind == "error" or "error" in d and not ev.name:
            self.done = True
            if not self.stream:
                self.failure = (500, openai_error(500, error_message(ev.data)))
            return same
        if kind in ("response.created", "response.in_progress"):
            if self.resp is None and isinstance(d.get("response"), dict):
                self.resp = d["response"]
            return same if not self.attempt else []
        changed = bool(self.attempt)
        if kind == "response.output_item.added":
            item = d.get("item") if isinstance(d.get("item"), dict) else {}
            t = item.get("type")
            if self.cont is not None and not self.map and self.out[self.cont]["type"] == t:
                self.map[d.get("output_index")] = self.cont
                self.ids[item.get("id")] = self.out[self.cont]["id"]
                return []                               # the cut item goes on
            our = len(self.out)
            self.out.append({"type": t, "id": item.get("id"), "text": "", "done": False, "item": item})
            self.map[d.get("output_index")] = our
            if t == "function_call":
                self.tooling = True
                self.out[our]["text"] = item.get("arguments") or ""
            d["output_index"] = our
        elif kind == "response.content_part.added":
            our = self.ours(d)
            if our is None:
                return same if not self.attempt else []
            if self.attempt and our == self.cont:
                return []
            d["output_index"], d["item_id"] = our, self.out[our]["id"]
        elif kind in DELTAS:
            our = self.ours(d, DELTAS[kind])
            if our is None:
                return same if not self.attempt else []
            delta = d.get("delta") if isinstance(d.get("delta"), str) else ""
            self.out[our]["text"] += delta
            if delta and kind != "response.function_call_arguments.delta":
                self.wrote()
            d["output_index"], d["item_id"] = our, self.out[our]["id"]
        elif kind in ("response.output_text.done", "response.reasoning_text.done", "response.content_part.done",
                      "response.output_item.done", "response.function_call_arguments.done"):
            self.finished = True                        # the closing events: everything is written
            if self.attempt:
                # after a move lodger finishes the items itself (at response.completed): the engine numbers its
                # closing events counting the reasoning it was given, not the items it streamed
                return []
            our = self.ours(d)
            if our is not None and kind == "response.output_item.done":
                self.out[our]["done"] = True
            return same
        elif kind == "response.completed":
            out = []
            for our, o in enumerate(self.out):
                if not o["done"]:
                    out += self.finishing(our)          # items the next engine didn't finish
            self.finished = self.done = True
            if not self.attempt:
                self.completed = d.get("response") if isinstance(d.get("response"), dict) else {}
                return same
            r = dict(d.get("response") or {})
            if self.resp and self.resp.get("id"):
                r["id"] = self.resp["id"]
            r["output"] = [self.full(o) for o in self.out]
            r["usage"] = self.usage(r.get("usage"), "input_tokens", "output_tokens")
            if self.before and isinstance(d.get("timings"), dict):
                d["timings"] = self.timings(d["timings"])
            d["response"] = r
            self.completed = r
            return (out + [self.event(ev.name or kind, d)]) if self.stream else []
        else:
            return same
        if not self.stream:
            return []
        return [self.event(ev.name or kind, d)] if changed else same

    def close(self, entry_name, why=None):
        if self.done:
            return []
        self.done = True
        out = []
        for our, o in enumerate(self.out):
            if not o["done"]:
                out += self.finishing(our)
        r = {"id": (self.resp or {}).get("id") or "resp_lodger", "object": "response", "created_at": int(self.t0),
             "status": "incomplete" if why == "length" else "completed", "model": entry_name,
             "output": [self.full(o) for o in self.out],
             "usage": {"input_tokens": 0, "output_tokens": self.pieces, "total_tokens": self.pieces}}
        if why == "length":
            r["incomplete_details"] = {"reason": "max_output_tokens"}
        self.completed = r
        out.append(self.event("response.completed", {"type": "response.completed", "response": r}))
        return out if self.stream else []

    def whole(self, entry_name):
        r = dict(self.completed or {"id": (self.resp or {}).get("id") or "resp_lodger", "object": "response",
                                     "created_at": int(self.t0), "status": "completed", "model": entry_name,
                                     "output": [self.full(o) for o in self.out]})
        r.setdefault("completed_at", r.get("created_at"))
        return r

    def error(self, status, msg):
        return openai_error(status, msg)

    def stream_error(self, msg):
        return [self.event("response.failed", {"type": "response.failed", "response": {
            "id": (self.resp or {}).get("id") or "resp_lodger", "object": "response", "status": "failed",
            "error": {"code": "server_error", "message": msg}}})]
