"""Answers that continue after a move, in each API's shape (lodger/answers.py through serve.py's handler): OpenAI chat
(streamed and whole), Anthropic Messages and OpenAI Responses; token counts over the whole answer; errors in each
API's own shape. The engines are fakes that stream what llama-server would."""
import json
import unittest

from tests.test_serve_front import ASK, Engine, FrontCase


def chunk(text=None, think=None, finish=None, tool=None, cid="chatcmpl-1", role=False, **extra):
    delta = {"role": "assistant"} if role else {}
    if text is not None:
        delta["content"] = text
    if think is not None:
        delta["reasoning_content"] = think
    if tool is not None:
        delta["tool_calls"] = [tool]
    return {"id": cid, "object": "chat.completion.chunk", "created": 7, "model": "m", "system_fingerprint": "b1",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}


def ev(name, d):
    return f"event: {name}\ndata: {json.dumps(d)}\n\n"


def sse(data):
    """(event name or None, data object) for each event of a streamed reply."""
    out = []
    for block in data.decode().split("\n\n"):
        name, lines = None, []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                lines.append(line[6:])
        if lines and lines != ["[DONE]"]:
            out.append((name, json.loads("\n".join(lines))))
    return out


USAGE = {"completion_tokens": 1, "prompt_tokens": 12, "total_tokens": 13, "prompt_tokens_details": {"cached_tokens": 0}}
TIMINGS = {"prompt_n": 12, "prompt_ms": 10.0, "predicted_n": 1, "predicted_ms": 5.0}


class ChatTest(FrontCase):
    def test_whole_answer_continues_after_a_move(self):
        a = Engine(events=[chunk(role=True, cid="first"), chunk("one ", cid="first"), chunk("two ", cid="first")], cut=True)
        b = Engine(events=[chunk(role=True, cid="second"), chunk("three", cid="second"), chunk(finish="stop", cid="second"),
                           {"id": "second", "object": "chat.completion.chunk", "created": 9, "model": "m", "choices": [],
                            "usage": USAGE, "timings": TIMINGS}])
        status, _, data = self.front(a, b).post("/v1/chat/completions", dict(ASK, max_tokens=10))
        self.assertEqual(status, 200)
        out = json.loads(data)
        self.assertEqual(out["object"], "chat.completion")
        self.assertEqual(out["id"], "first")
        self.assertEqual(out["choices"][0]["message"], {"role": "assistant", "content": "one two three"})
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertEqual(out["usage"]["completion_tokens"], 3)              # the whole answer, not only the end
        self.assertEqual(out["usage"]["prompt_tokens"], 10)
        self.assertEqual(out["usage"]["total_tokens"], 13)
        self.assertEqual(out["timings"]["predicted_n"], 3)
        self.assertTrue(a.calls[0][1]["stream"])                            # streamed inside lodger
        self.assertEqual(a.calls[0][1]["stream_options"], {"include_usage": True})
        self.assertEqual(b.calls[0][1]["messages"][-1], {"role": "assistant", "content": "one two "})
        self.assertEqual(b.calls[0][1]["max_tokens"], 8)

    def test_whole_tool_call_cut_starts_over(self):
        call = {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\""}}
        a = Engine(events=[chunk(tool=call)], cut=True)
        b = Engine(events=[chunk(tool=call), chunk(tool={"index": 0, "function": {"arguments": ": 1}"}}),
                           chunk(finish="tool_calls")])
        status, _, data = self.front(a, b).post("/v1/chat/completions", ASK)
        self.assertEqual(status, 200)
        msg = json.loads(data)["choices"][0]["message"]
        self.assertEqual(msg["tool_calls"], [{"type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"},
                                              "id": "c1"}])
        self.assertEqual(b.calls[0][1]["messages"], ASK["messages"])        # asked again from the start

    def test_stream_usage_counts_the_whole_answer(self):
        a = Engine(events=[chunk("a"), chunk("b")], cut=True)
        b = Engine(events=[chunk("c"), chunk(finish="stop"),
                           {"id": "x", "object": "chat.completion.chunk", "created": 9, "model": "m", "choices": [],
                            "usage": USAGE, "timings": TIMINGS}])
        status, _, data = self.front(a, b).post("/v1/chat/completions",
                                                dict(ASK, stream=True, stream_options={"include_usage": True}))
        last = sse(data)[-1][1]
        self.assertEqual(last["usage"]["completion_tokens"], 3)
        self.assertEqual(last["timings"]["predicted_n"], 3)
        self.assertEqual(last["id"], "chatcmpl-1")

    def test_engine_error_before_the_answer_keeps_its_status(self):
        err = {"error": {"code": 400, "message": "the request exceeds the available context size",
                         "type": "exceed_context_size_error"}}
        for stream in (False, True):
            status, headers, data = self.front(Engine(whole=err, status=400)).post(
                "/v1/chat/completions", dict(ASK, stream=stream))
            self.assertEqual(status, 400, stream)
            self.assertEqual(json.loads(data), err)

    def test_several_answers(self):
        whole = {"choices": [{"index": 0}, {"index": 1}]}
        e = Engine(whole=whole)
        status, _, data = self.front(e).post("/v1/chat/completions", dict(ASK, n=2))
        self.assertEqual(json.loads(data), whole)                           # passed on as it is
        self.assertNotIn("stream", e.calls[0][1])
        a = Engine(events=[chunk("x")], cut=True)
        status, _, data = self.front(a, Engine(events=[chunk("y")])).post("/v1/chat/completions",
                                                                         dict(ASK, n=2, stream=True))
        self.assertIn("several answers", sse(data)[-1][1]["error"]["message"])

    def test_logprobs_answer_passed_on_whole(self):
        e = Engine(whole={"choices": [{"logprobs": {"content": []}}]})
        status, _, data = self.front(e).post("/v1/chat/completions", dict(ASK, logprobs=True))
        self.assertIn("logprobs", json.loads(data)["choices"][0])

    def test_unavailable_model(self):
        status, _, data = self.front().post("/v1/chat/completions", dict(ASK, stream=True))
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(data)["error"]["type"], "unavailable_error")


def msg_start(mid="msg_1", input_tokens=20):
    return ev("message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "content": [], "model": "m", "stop_reason": None,
        "stop_sequence": None, "usage": {"cache_read_input_tokens": 0, "input_tokens": input_tokens, "output_tokens": 0}}})


def block_start(i, kind, **extra):
    return ev("content_block_start", {"type": "content_block_start", "index": i,
                                      "content_block": {"type": kind, kind if kind != "tool_use" else "id": "", **extra}})


def block_delta(i, kind, text):
    field = {"text_delta": "text", "thinking_delta": "thinking", "input_json_delta": "partial_json",
             "signature_delta": "signature"}[kind]
    return ev("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": kind, field: text}})


def block_stop(i):
    return ev("content_block_stop", {"type": "content_block_stop", "index": i})


def msg_end(n, reason="end_turn"):
    return [ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
                                 "usage": {"output_tokens": n}}),
            ev("message_stop", {"type": "message_stop"})]


MSG = {"model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "count"}]}


class MessagesTest(FrontCase):
    def test_stream_without_a_move_is_passed_on_as_it_is(self):
        events = [msg_start(), block_start(0, "text"), block_delta(0, "text_delta", "hi"), block_stop(0), *msg_end(1)]
        status, headers, data = self.front(Engine(events=events, done=False)).post("/v1/messages", dict(MSG, stream=True))
        self.assertEqual(status, 200)
        self.assertEqual(data.decode(), "".join(events))

    def test_stream_continues_after_a_move(self):
        a = Engine(events=[msg_start(), block_start(0, "thinking"), block_delta(0, "thinking_delta", "hmm"),
                           block_start(1, "text"), block_delta(1, "text_delta", "one "),
                           block_delta(1, "text_delta", "two ")], cut=True)
        b = Engine(events=[msg_start("msg_2", 30), block_start(0, "text"), block_delta(0, "text_delta", "three"),
                           block_stop(0), *msg_end(1)], done=False)
        status, _, data = self.front(a, b).post("/v1/messages", dict(MSG, stream=True))
        out = sse(data)
        kinds = [n for n, _ in out]
        self.assertEqual(kinds.count("message_start"), 1)
        self.assertEqual(out[0][1]["message"]["id"], "msg_1")
        self.assertEqual(kinds.count("content_block_start"), 2)
        text = "".join(d["delta"].get("text", "") for n, d in out if n == "content_block_delta" and d["index"] == 1)
        self.assertEqual(text, "one two three")
        stops = [d["index"] for n, d in out if n == "content_block_stop"]
        self.assertEqual(sorted(stops), [0, 1])                             # each block closed once
        sig = [d for n, d in out if n == "content_block_delta" and d["delta"]["type"] == "signature_delta"]
        self.assertEqual([d["index"] for d in sig], [0])
        self.assertEqual(kinds[-2:], ["message_delta", "message_stop"])
        self.assertEqual(out[-2][1]["usage"]["output_tokens"], 4)           # 3 pieces before the move + 1
        resent = b.calls[0][1]
        self.assertEqual(resent["messages"][-1], {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": ""}, {"type": "text", "text": "one two "}]})
        self.assertEqual(resent["max_tokens"], 97)

    def test_whole_answer_continues_after_a_move(self):
        a = Engine(events=[msg_start(), block_start(0, "text"), block_delta(0, "text_delta", "Hel")], cut=True)
        b = Engine(events=[msg_start("msg_2"), block_start(0, "text"), block_delta(0, "text_delta", "lo"),
                           block_stop(0), *msg_end(1)], done=False)
        ask = dict(MSG, messages=MSG["messages"] + [{"role": "assistant", "content": "Say: "}])
        status, _, data = self.front(a, b).post("/v1/messages", ask)
        self.assertEqual(status, 200)
        out = json.loads(data)
        self.assertEqual(out["id"], "msg_1")
        self.assertEqual(out["content"], [{"type": "text", "text": "Hello"}])
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["usage"]["output_tokens"], 2)
        self.assertEqual(out["usage"]["input_tokens"], 20)
        self.assertEqual(b.calls[0][1]["messages"][-1], {"role": "assistant", "content": "Say: Hel"})
        self.assertEqual(len(b.calls[0][1]["messages"]), 2)                 # the app's own start, continued

    def test_cut_while_thinking_continues_the_thinking(self):
        """llama-server continues the thinking only when the assistant message it is given has no text part, even an
        empty one; with one it writes the rest of the thinking as text, with a literal </think> in it."""
        a = Engine(events=[msg_start(), block_start(0, "thinking"), block_delta(0, "thinking_delta", "hmm")], cut=True)
        b = Engine(events=[msg_start("msg_2"), block_start(0, "thinking"), block_delta(0, "thinking_delta", " more"),
                           block_stop(0), block_start(1, "text"), block_delta(1, "text_delta", "ok"), block_stop(1),
                           *msg_end(2)], done=False)
        status, _, data = self.front(a, b).post("/v1/messages", MSG)
        self.assertEqual(status, 200)
        self.assertEqual(b.calls[0][1]["messages"][-1], {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": ""}]})
        self.assertEqual(json.loads(data)["content"], [{"type": "thinking", "thinking": "hmm more", "signature": ""},
                                                       {"type": "text", "text": "ok"}])

    def after_thinking(self):
        """A cut in the answer after its thinking; llama-server numbers the next engine's closing events counting the
        thinking it was given (thinking 0, text 1), not the text block it streamed as 0."""
        a = Engine(events=[msg_start(), block_start(0, "thinking"), block_delta(0, "thinking_delta", "hmm"),
                           block_delta(0, "signature_delta", ""), block_stop(0), block_start(1, "text"),
                           block_delta(1, "text_delta", "one ")], cut=True)
        b = Engine(events=[msg_start("msg_2"), block_start(0, "text"), block_delta(0, "text_delta", "two"),
                           block_delta(0, "signature_delta", ""), block_stop(0), block_stop(1), *msg_end(1)],
                   done=False)
        return a, b

    def test_closing_numbers_after_a_move_dont_matter_whole(self):
        status, _, data = self.front(*self.after_thinking()).post("/v1/messages", MSG)
        self.assertEqual(status, 200)
        out = json.loads(data)
        self.assertEqual(out["content"], [{"type": "thinking", "thinking": "hmm", "signature": ""},
                                          {"type": "text", "text": "one two"}])
        self.assertEqual(out["usage"]["output_tokens"], 3)

    def test_closing_numbers_after_a_move_dont_matter_stream(self):
        status, _, data = self.front(*self.after_thinking()).post("/v1/messages", dict(MSG, stream=True))
        out = sse(data)
        self.assertEqual([n for n, _ in out if n == "error"], [])
        stops = [d["index"] for n, d in out if n == "content_block_stop"]
        self.assertEqual(stops, [0, 1])
        sig = [d["index"] for n, d in out if n == "content_block_delta" and d["delta"]["type"] == "signature_delta"]
        self.assertEqual(sig, [0])
        text = "".join(d["delta"].get("text", "") for n, d in out if n == "content_block_delta" and d["index"] == 1)
        self.assertEqual(text, "one two")
        self.assertEqual([n for n, _ in out][-2:], ["message_delta", "message_stop"])

    def test_usage_when_the_engine_never_started_the_message(self):
        a = Engine(events=[], cut=True)
        b = Engine(events=[msg_start("msg_2", 30)], cut=True,
                   paths={"/v1/messages/count_tokens": {"input_tokens": 9}})
        c = Engine(events=[], paths={"/v1/messages/count_tokens": {"input_tokens": 9}})
        msgs = [block_start(0, "text"), *[block_delta(0, "text_delta", f"{i} ") for i in range(100)]]
        b.events += msgs
        status, _, data = self.front(a, b, c).post("/v1/messages", MSG)
        out = json.loads(data)
        self.assertEqual(out["stop_reason"], "max_tokens")
        self.assertEqual(out["usage"]["input_tokens"], 30)
        self.assertEqual(out["usage"]["output_tokens"], 100)

    def test_tool_call_cut_ends_with_an_anthropic_error(self):
        a = Engine(events=[msg_start(), block_start(0, "tool_use", name="ls"),
                           block_delta(0, "input_json_delta", "{\"p")], cut=True)
        status, _, data = self.front(a, Engine(events=[])).post("/v1/messages", dict(MSG, stream=True))
        name, d = sse(data)[-1]
        self.assertEqual(name, "error")
        self.assertEqual(d["type"], "error")
        self.assertIn("tool call", d["error"]["message"])

    def test_errors_in_anthropic_shape(self):
        f = self.front()
        status, _, data = f.post("/v1/messages", dict(MSG, model="claude-sonnet-4-5"))
        self.assertEqual(status, 404)
        d = json.loads(data)
        self.assertEqual(d["type"], "error")
        self.assertEqual(d["error"]["type"], "not_found_error")
        self.assertIn("claude-sonnet-4-5", d["error"]["message"])
        status, _, data = f.post("/v1/messages", MSG)
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(data)["error"]["type"], "overloaded_error")
        status, _, data = self.front(Engine(whole={"error": {"code": 400, "message": "too long"}}, status=400)).post(
            "/v1/messages/count_tokens", MSG)
        self.assertEqual((status, json.loads(data)), (400, {"type": "error", "error": {
            "type": "invalid_request_error", "message": "too long"}}))

    def test_count_tokens_passed_on(self):
        e = Engine(whole={"input_tokens": 9})
        status, _, data = self.front(e).post("/v1/messages/count_tokens", MSG)
        self.assertEqual((status, json.loads(data)), (200, {"input_tokens": 9}))
        self.assertEqual(e.calls[0][0], "/v1/messages/count_tokens")


def rev(name, **d):
    return ev(name, {"type": name, **d})


def resp_start(rid="resp_1"):
    r = {"id": rid, "object": "response", "status": "in_progress"}
    return [rev("response.created", response=r), rev("response.in_progress", response=r)]


def msg_added(i, mid):
    return [rev("response.output_item.added", output_index=i, item={
                "content": [], "id": mid, "role": "assistant", "status": "in_progress", "type": "message"}),
            rev("response.content_part.added", item_id=mid, output_index=i, content_index=0,
                part={"type": "output_text", "text": ""})]


def text_delta(i, mid, t):
    return rev("response.output_text.delta", item_id=mid, output_index=i, content_index=0, delta=t)


def resp_end(i, mid, text, rid, n):
    part = {"type": "output_text", "annotations": [], "logprobs": [], "text": text}
    item = {"type": "message", "status": "completed", "id": mid, "content": [part], "role": "assistant"}
    return [rev("response.output_text.done", item_id=mid, output_index=i, content_index=0, text=text),
            rev("response.content_part.done", item_id=mid, output_index=i, content_index=0, part=part),
            rev("response.output_item.done", output_index=i, item=item),
            rev("response.completed", response={"id": rid, "object": "response", "created_at": 5, "status": "completed",
                                                "model": "m", "output": [item], "usage": {
                                                    "input_tokens": 30, "output_tokens": n, "total_tokens": 30 + n}})]


RESP = {"model": "m", "input": "count", "max_output_tokens": 50}


class ResponsesTest(FrontCase):
    def test_stream_continues_after_a_move(self):
        a = Engine(events=[*resp_start(), *msg_added(0, "msg_a"), text_delta(0, "msg_a", "one "),
                           text_delta(0, "msg_a", "two ")], cut=True)
        b = Engine(events=[*resp_start("resp_2"), *msg_added(0, "msg_b"), text_delta(0, "msg_b", "three"),
                           *resp_end(0, "msg_b", "three", "resp_2", 1)], done=False)
        status, _, data = self.front(a, b).post("/v1/responses", dict(RESP, stream=True))
        out = sse(data)
        kinds = [n for n, _ in out]
        self.assertEqual(kinds.count("response.created"), 1)
        self.assertEqual(kinds.count("response.output_item.added"), 1)
        self.assertEqual({d.get("item_id") for n, d in out if "item_id" in d}, {"msg_a"})
        done = dict(out)["response.output_text.done"]
        self.assertEqual(done["text"], "one two three")
        completed = out[-1][1]["response"]
        self.assertEqual(completed["id"], "resp_1")
        self.assertEqual(completed["output"][0]["content"][0]["text"], "one two three")
        self.assertEqual(completed["output"][0]["id"], "msg_a")
        self.assertEqual(completed["usage"], {"input_tokens": 28, "output_tokens": 3, "total_tokens": 31})
        resent = b.calls[0][1]
        self.assertEqual(resent["input"], [{"type": "message", "role": "user", "content": "count"},
                                           {"type": "message", "role": "assistant",
                                            "content": [{"type": "output_text", "text": "one two "}]}])
        self.assertEqual(resent["max_output_tokens"], 48)

    def test_whole_answer(self):
        a = Engine(events=[*resp_start(), *msg_added(0, "msg_a"), text_delta(0, "msg_a", "Hel")], cut=True)
        b = Engine(events=[*resp_start("resp_2"), *msg_added(0, "msg_b"), text_delta(0, "msg_b", "lo"),
                           *resp_end(0, "msg_b", "lo", "resp_2", 1)], done=False)
        status, _, data = self.front(a, b).post("/v1/responses", RESP)
        out = json.loads(data)
        self.assertEqual(out["id"], "resp_1")
        self.assertEqual(out["object"], "response")
        self.assertEqual(out["output"][0]["content"][0]["text"], "Hello")
        self.assertIn("completed_at", out)

    def test_without_a_move_events_are_as_they_were(self):
        events = [*resp_start(), *msg_added(0, "msg_a"), text_delta(0, "msg_a", "hi"), *resp_end(0, "msg_a", "hi", "resp_1", 1)]
        status, _, data = self.front(Engine(events=events, done=False)).post("/v1/responses", dict(RESP, stream=True))
        self.assertEqual(data.decode(), "".join(events))

    def test_cut_while_reasoning_continues_the_reasoning(self):
        r_item = {"id": "rs_a", "type": "reasoning", "summary": [], "content": [], "status": "in_progress"}
        a = Engine(events=[*resp_start(), rev("response.output_item.added", output_index=0, item=r_item),
                           rev("response.reasoning_text.delta", item_id="rs_a", output_index=0, content_index=0,
                               delta="hmm")], cut=True)
        b = Engine(events=[*resp_start("resp_2"), *msg_added(0, "msg_b"), text_delta(0, "msg_b", "ok"),
                           *resp_end(0, "msg_b", "ok", "resp_2", 1)], done=False)
        self.front(a, b).post("/v1/responses", RESP)
        self.assertEqual(b.calls[0][1]["input"][-1], {"type": "reasoning", "summary": [], "content": [
            {"type": "reasoning_text", "text": "hmm"}]})                    # no empty message after it

    def after_reasoning(self):
        """A cut in the answer after its reasoning; the next engine's closing events number the reasoning it was
        given as item 0 and its message as item 1, though it streamed the message as item 0."""
        r_item = {"id": "rs_a", "type": "reasoning", "summary": [], "content": [], "status": "in_progress"}
        a = Engine(events=[*resp_start(), rev("response.output_item.added", output_index=0, item=r_item),
                           rev("response.reasoning_text.delta", item_id="rs_a", output_index=0, content_index=0,
                               delta="hmm"),
                           *msg_added(1, "msg_a"), text_delta(1, "msg_a", "one ")], cut=True)
        r_b = {"id": "rs_b", "type": "reasoning", "summary": [], "status": "completed",
               "content": [{"type": "reasoning_text", "text": "hmm"}]}
        end = resp_end(1, "msg_b", "two", "resp_2", 1)
        b = Engine(events=[*resp_start("resp_2"), *msg_added(0, "msg_b"), text_delta(0, "msg_b", "two"),
                           rev("response.output_item.done", output_index=0, item=r_b), *end], done=False)
        return a, b

    def test_closing_numbers_after_a_move_dont_matter_whole(self):
        status, _, data = self.front(*self.after_reasoning()).post("/v1/responses", RESP)
        self.assertEqual(status, 200)
        out = json.loads(data)
        self.assertEqual([o["type"] for o in out["output"]], ["reasoning", "message"])
        self.assertEqual(out["output"][1]["content"][0]["text"], "one two")
        self.assertEqual(out["usage"], {"input_tokens": 28, "output_tokens": 3, "total_tokens": 31})

    def test_closing_numbers_after_a_move_dont_matter_stream(self):
        status, _, data = self.front(*self.after_reasoning()).post("/v1/responses", dict(RESP, stream=True))
        out = sse(data)
        kinds = [n for n, _ in out]
        self.assertNotIn("response.failed", kinds)
        done = [(d["output_index"], d["item"]["id"]) for n, d in out if n == "response.output_item.done"]
        self.assertEqual(done, [(0, "rs_a"), (1, "msg_a")])                 # each item finished once
        self.assertEqual([d["text"] for n, d in out if n == "response.output_text.done"], ["one two"])
        completed = out[-1][1]["response"]
        self.assertEqual(completed["usage"]["output_tokens"], 3)
        self.assertEqual(completed["output"][1]["content"][0]["text"], "one two")

    def test_move_before_the_first_event_still_creates_the_response(self):
        a = Engine(events=[], cut=True)
        b = Engine(events=[*resp_start("resp_2"), *msg_added(0, "msg_b"), text_delta(0, "msg_b", "hi"),
                           *resp_end(0, "msg_b", "hi", "resp_2", 1)], done=False)
        status, _, data = self.front(a, b).post("/v1/responses", dict(RESP, stream=True))
        kinds = [n for n, _ in sse(data)]
        self.assertEqual(kinds[:2], ["response.created", "response.in_progress"])
        self.assertEqual(kinds.count("response.created"), 1)

    def test_structured_answer_is_not_continued(self):
        ask = dict(RESP, stream=True, text={"format": {"type": "json_schema", "name": "n", "schema": {}}})
        a = Engine(events=[*resp_start(), *msg_added(0, "msg_a"), text_delta(0, "msg_a", '{"a": ')], cut=True)
        b = Engine(events=[*resp_start("resp_2"), *msg_added(0, "msg_b"), text_delta(0, "msg_b", "1}"),
                           *resp_end(0, "msg_b", "1}", "resp_2", 1)], done=False)
        status, _, data = self.front(a, b).post("/v1/responses", ask)
        name, d = sse(data)[-1]
        self.assertEqual(name, "response.failed")
        self.assertIn("structured", d["response"]["error"]["message"])
        self.assertEqual(b.calls, [])

    def test_usage_of_an_answer_ended_at_its_limit_after_a_move(self):
        a = Engine(events=[*resp_start(), *msg_added(0, "msg_a"), *[text_delta(0, "msg_a", f"{i} ") for i in range(50)]],
                   cut=True)
        b = Engine(events=[], paths={"/v1/responses/input_tokens": {"object": "response.input_tokens", "input_tokens": 12}})
        status, _, data = self.front(a, b).post("/v1/responses", RESP)
        out = json.loads(data)
        self.assertEqual(out["status"], "incomplete")
        self.assertEqual(out["usage"], {"input_tokens": 12, "output_tokens": 50, "total_tokens": 62})

    def test_tool_call_cut_fails_the_response(self):
        a = Engine(events=[*resp_start(), rev("response.output_item.added", output_index=0, item={
            "id": "fc_1", "arguments": "", "call_id": "call_1", "name": "ls", "type": "function_call",
            "status": "in_progress"}), rev("response.function_call_arguments.delta", delta="{", item_id="fc_1",
                                           output_index=0)], cut=True)
        status, _, data = self.front(a, Engine(events=[])).post("/v1/responses", dict(RESP, stream=True))
        name, d = sse(data)[-1]
        self.assertEqual(name, "response.failed")
        self.assertEqual(d["response"]["status"], "failed")
        self.assertIn("tool call", d["response"]["error"]["message"])


class PassTest(FrontCase):
    def test_rerank_and_input_tokens(self):
        for path in ("/v1/rerank", "/rerank", "/v1/responses/input_tokens"):
            e = Engine(whole={"results": []})
            status, _, data = self.front(e).post(path, {"model": "m", "query": "q", "documents": ["a"]})
            self.assertEqual((status, e.calls[0][0]), (200, path))


if __name__ == "__main__":
    unittest.main()
