"""lodger serve: keep a model ready for apps on this PC, and move it out of your way and back.

  lodger serve MODEL [--allow-gpu] [--port 8181] [--ctx 16384] [--think]
  lodger ask "question" [--port 8181]

Apps talk to lodger at http://127.0.0.1:8181/v1/chat/completions (the usual OpenAI-style API); only
this PC can reach it. Behind it, lodger runs our llama.cpp build's server on a private port, with the
same plan, fence and watchdog as `lodger run`.
- Steps aside: when another app needs the graphics card, the model is stopped on the card and started
  again on the processor at once. An answer being written continues where it stopped.
- Grows back: when the card has been quiet for a while and no answer is being written, the model moves
  back onto the card.
- Sleeps (with --idle MIN): unloads the model after MIN minutes unused and loads it again on the next
  request, so it holds no memory while you don't use it.
GET /lodger/status says where the model is and why. POST /lodger/move moves it off the card now
(it grows back later as usual)."""
import argparse
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import plan as planner, sysmem
from .launch import RUNS, Watchdog, gpu_blockers, router_coming, unit_props
from .neighbors import Neighbors

MiB, GiB = sysmem.MiB, sysmem.GiB
HOST = "127.0.0.1"          # only this PC; part of the approved exception (../CLAUDE.md)
QUIET_SECONDS = 60          # the card must be quiet this long before the model grows back
MOVE_WAIT = 300             # how long a request waits for the model to be ready


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def free_port():
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


class Engine:
    """One llama-server process in its own fenced systemd unit, with a watchdog."""
    count = 0
    all = []

    def __init__(self, plan, a, out):
        Engine.count += 1
        self.plan, self.out = plan, out
        self.port = free_port()
        self.unit = f"lodger-{os.getpid()}-{Engine.count}"
        self.where = "card" if plan["ngl"] else "processor"
        p = plan
        cmd = [os.path.join(planner.BIN, "llama-server"), "-m", p["model"], "-c", str(p["ctx"]),
               "-t", str(p["threads"]), "--fit", "off", "-np", "1", "--host", HOST, "--port", str(self.port),
               "--no-webui", "-a", a.name]
        if p["ngl"]:
            cmd += ["--device", "ROCm0", "-ngl", str(p["ngl"])] + (["-ncmoe", str(p["n_cpu_moe"])] if p["n_cpu_moe"] else [])
            env = {"HIP_VISIBLE_DEVICES": "0"}
        else:
            cmd += ["--device", "none", "-ngl", "0"]
            env = {"HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"}
        if not a.think:
            cmd += ["--reasoning", "off"]
        sr = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={self.unit}", "--pipe", "--wait"]
        sr += [f"--property={k}={v}" for k, v in unit_props(p).items()]
        sr += [f"--setenv={k}={v}" for k, v in env.items()]
        with open(os.path.join(out, "plan.json"), "w") as f:
            json.dump({**p, "cmd": cmd, "env": env}, f, indent=1)
        self.logf = open(os.path.join(out, "engine.log"), "w")
        # own session: Ctrl-C in the terminal reaches lodger only, which then stops the engine itself
        self.proc = subprocess.Popen(sr + cmd, stdin=subprocess.DEVNULL, stdout=self.logf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.wd = Watchdog(self.unit, bool(p["ngl"]), out)
        self.wd.start()
        self.started = time.time()
        Engine.all.append(self)

    def alive(self):
        return self.proc.poll() is None

    def wait_ready(self, limit=600):
        while time.time() - self.started < limit and self.alive():
            try:
                c = http.client.HTTPConnection(HOST, self.port, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    return True
            except OSError:
                pass
            time.sleep(0.5)
        return False

    def stop(self):
        """Planned stop (not the watchdog): the watchdog is told first so it records no reason."""
        self.wd.stop_flag.set()
        subprocess.run(["systemctl", "--user", "kill", "-s", "TERM", self.unit + ".service"], capture_output=True)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            subprocess.run(["systemctl", "--user", "kill", "-s", "KILL", self.unit + ".service"], capture_output=True)
            self.proc.wait(timeout=10)
        subprocess.run(["systemctl", "--user", "stop", self.unit + ".service"], capture_output=True)  # in case it outlived its client
        self.finish()

    def finish(self):
        if self.logf.closed:
            return
        self.wd.stop_flag.set()
        self.wd.join(timeout=5)
        self.logf.close()
        # learn: planned vs really used on the card (only once it had loaded)
        if self.plan["ngl"] and time.time() - self.started > 20:
            with open(os.path.join(RUNS, "calibration.jsonl"), "a") as f:
                f.write(json.dumps({"time": time.strftime("%F %T"), "model": self.plan["name"], "ngl": self.plan["ngl"],
                                    "n_cpu_moe": self.plan["n_cpu_moe"], "ctx": self.plan["ctx"],
                                    "gpu_est": self.plan["gpu_est"], "llama_gpu_est": self.plan.get("llama_gpu_est"),
                                    "gpu_real": self.wd.peak["model_vram"] or None, "stopped": self.wd.reason,
                                    "run": os.path.basename(self.out), "via": "serve"}) + "\n")


class Lodger:
    def __init__(self, a):
        self.a = a
        self.cv = threading.Condition()
        self.engine = None
        self.ready = False
        self.active = 0
        self.why = "starting"
        self.why_off_card = None if a.allow_gpu else "the graphics card is only used when you allow it (--allow-gpu)"
        self.last_aside = 0.0
        self.quiet_since = time.time()
        self.quiet_needed = QUIET_SECONDS     # doubles each time the model has to step aside soon after growing back
        self.card_since = None
        self.events = []                      # (time, text) shown in /lodger/status
        self.force_off_card = False
        self.closing = False
        self.asleep = bool(a.idle)            # with --idle: load the model on the first request, not at start
        self.wake = False
        self.last_used = time.time()
        self.out = os.path.join(RUNS, time.strftime("%Y-%m-%d-%H%M%S") + "-serve-" + a.name)
        os.makedirs(self.out)

    def note(self, text):
        log(text)
        self.events = (self.events + [(time.strftime("%H:%M:%S"), text)])[-20:]

    def make_plan(self, on_card):
        try:
            return planner.make(self.a.model, target_ctx=self.a.ctx, min_ctx=self.a.min_ctx, cpu_only=not on_card,
                                leave_gpu=self.a.leave_gpu * MiB, threads=self.a.threads, device_check=on_card,
                                why_gpu_off=None if on_card else (self.why_off_card or "the card is not wanted now"))
        except (Exception, SystemExit) as ex:   # a failed llama.cpp probe must not take the server down
            return {"refuse": f"planning failed: {ex}"}

    def launch(self, want_card, card_only=False):
        """Start an engine (card if wanted and possible, else processor) and wait until it answers.
        card_only: give up instead of falling back to the processor. Returns the Engine, or None."""
        if want_card:
            self.why_off_card = gpu_blockers(self.a.allow_gpu, sysmem.dgpu())
        if card_only and self.why_off_card:
            return None
        p = self.make_plan(want_card and not self.why_off_card)
        if p["refuse"] or (card_only and not p["ngl"]):
            if p["refuse"]:
                self.note(f"can't start the model: {p['refuse']}")
            return None
        n = Engine.count + 1
        out = os.path.join(self.out, f"engine-{n}-{'card' if p['ngl'] else 'processor'}")
        if self.closing:
            return None
        os.makedirs(out)
        e = Engine(p, self.a, out)
        self.note(f"starting on the {e.where}" + (f" ({self.why_off_card})" if self.why_off_card and not p["ngl"] else "")
                  + f", context {p['ctx']}" + (f", {p['ngl']} layers on the card" if p["ngl"] else ""))
        if not e.wait_ready():
            reason = e.wd.reason or "it exited while loading (see engine.log)"
            if e.alive():
                e.stop()
            else:
                e.finish()
            self.note(f"the {e.where} engine didn't come up: {reason}")
            return None
        self.note(f"ready on the {e.where} after {time.time() - e.started:.0f} s")
        return e

    def install(self, e):
        with self.cv:
            old, self.engine, self.ready = self.engine, e, True
            self.why = f"on the {e.where}" + (f": {self.why_off_card}" if e.where == "processor" and self.why_off_card else "")
            self.card_since = time.time() if e.where == "card" else None
            self.cv.notify_all()
        return old

    def bring_up(self, want_card):
        """Start an engine and keep trying (on the processor) until one is up."""
        delay = 10
        e = self.launch(want_card)
        while not e and not self.closing:
            time.sleep(delay)
            delay = min(delay * 2, 300)
            e = self.launch(False)
        if e:
            self.install(e)

    def wait_engine(self, limit=MOVE_WAIT):
        with self.cv:
            self.last_used = time.time()
            if self.asleep:
                self.wake = True
            self.cv.wait_for(lambda: self.ready and self.engine and self.engine.alive(), timeout=limit)
            return self.engine if self.ready and self.engine and self.engine.alive() else None

    def stepped_aside(self):
        self.last_aside = time.time()
        if self.card_since and time.time() - self.card_since < 600:
            self.quiet_needed = min(self.quiet_needed * 2, 1800)   # it came back too soon: wait longer next time
        elif self.card_since:
            self.quiet_needed = QUIET_SECONDS
        self.card_since = None

    def monitor(self):
        """Steps aside / grows back. Runs for the life of `lodger serve`."""
        nb = Neighbors()
        while not self.closing:
            time.sleep(1)
            if self.closing:
                return
            if nb.update() or router_coming():
                self.quiet_since = time.time()
            if self.asleep:
                if self.wake:
                    self.wake = False
                    self.note("a request came in: loading the model")
                    self.bring_up(want_card=self.a.allow_gpu)
                    with self.cv:
                        self.asleep = False
                continue
            e = self.engine
            # unused for a while: unload, so the memory is yours again until the next request
            if (self.a.idle and e and e.alive() and self.active == 0
                    and time.time() - self.last_used >= self.a.idle * 60):
                with self.cv:
                    self.ready, self.asleep, self.engine = False, True, None
                    self.why = f"asleep: unloaded after {self.a.idle} min unused (loads again on the next request)"
                self.note(f"unused for {self.a.idle} min: unloading the model until the next request")
                e.stop()
                self.card_since = None
                continue
            if e and not e.alive():
                reason = e.wd.reason
                e.finish()
                with self.cv:
                    self.ready = False
                if reason and e.where == "card":
                    self.stepped_aside()
                    self.why_off_card = "stepped aside: " + reason
                    self.note(f"stepped aside from the card: {reason}")
                else:
                    self.note(f"the {e.where} engine stopped" + (f": {reason}" if reason else
                              f" unexpectedly (see {e.out}/engine.log)"))
                    time.sleep(3)
                self.bring_up(want_card=False)          # straight back up on the processor
                continue
            if self.force_off_card and e and e.where == "card":
                self.force_off_card = False
                self.stepped_aside()
                self.note("moving off the card (asked to)")
                self.why_off_card = "asked to move off the card"
                with self.cv:
                    self.ready = False
                e.stop()
                self.bring_up(want_card=False)
                continue
            # grow back: on the processor, card allowed, nothing being written, card quiet for a while
            now = time.time()
            if (e and e.where == "processor" and self.a.allow_gpu and self.active == 0
                    and now - self.quiet_since >= self.quiet_needed and now - self.last_aside >= self.quiet_needed):
                trial = self.make_plan(True)
                if trial["refuse"] or not trial["ngl"]:
                    self.quiet_since = time.time()      # no room on the card yet; look again in a while
                    continue
                self.note(f"the card has been quiet for {now - self.quiet_since:.0f} s: growing back onto it")
                # the processor engine keeps answering while the card engine loads; then they swap
                new = self.launch(want_card=True, card_only=True)
                if not new:
                    self.quiet_since = time.time()
                    self.note("not growing back after all: " + (self.why_off_card or "the card engine didn't come up"))
                    continue
                old = self.install(new)
                if old:
                    old.stop()                          # an answer still being written there continues on the card

    def status(self):
        e = self.engine
        return {"model": self.a.name, "ready": self.ready, "where": "asleep" if self.asleep else e.where if e else None,
                "why": self.why,
                "answers_in_progress": self.active, "quiet_needed_s": self.quiet_needed,
                "card_quiet_for_s": round(time.time() - self.quiet_since),
                "plan": {k: e.plan[k] for k in ("ngl", "n_cpu_moe", "ctx", "gpu_est")} if e else None,
                "events": self.events}


def make_handler(lodger):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                if lodger.asleep:                     # a request wakes it, so apps may go ahead
                    return self.send_json(200, {"status": "ok", "asleep": True})
                return self.send_json(200 if lodger.ready else 503, {"status": "ok" if lodger.ready else "loading"})
            if self.path == "/v1/models":
                return self.send_json(200, {"object": "list", "data": [{"id": lodger.a.name, "object": "model", "owned_by": "lodger"}]})
            if self.path == "/lodger/status":
                return self.send_json(200, lodger.status())
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            if self.path == "/lodger/move":
                on_card = bool(lodger.engine and lodger.engine.where == "card" and lodger.ready)
                lodger.force_off_card = on_card
                return self.send_json(200, {"moving": on_card, "where": lodger.engine.where if lodger.engine else None})
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                return self.send_json(404, {"error": "not found"})
            try:
                body = json.loads(raw)
            except ValueError:
                return self.send_json(400, {"error": "body is not JSON"})
            with lodger.cv:
                lodger.active += 1
            try:
                if body.get("stream"):
                    self.stream(body)
                else:
                    self.whole(body)
            finally:
                with lodger.cv:
                    lodger.active -= 1
                    lodger.last_used = time.time()

        def upstream(self, e, body):
            c = http.client.HTTPConnection(HOST, e.port, timeout=600)
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            return c, c.getresponse()

        def whole(self, body):
            for _ in range(4):
                e = lodger.wait_engine()
                if not e:
                    return self.send_json(503, {"error": "the model is not available right now: " + lodger.why})
                try:
                    c, r = self.upstream(e, body)
                    data = r.read()
                    if r.status == 200 or e.alive():
                        self.send_response(r.status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(1)                       # the engine moved underneath us: ask again on the new one
            self.send_json(503, {"error": "the model kept moving; try again"})

        def stream(self, body):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            content, reasoning, pieces = "", "", 0
            limit = body.get("max_tokens") or body.get("max_completion_tokens") or body.get("n_predict")
            for attempt in range(6):
                e = lodger.wait_engine()
                if not e:
                    self.event(f'data: {json.dumps({"error": {"message": "the model is not available: " + lodger.why}})}')
                    return
                req = dict(body)
                if content or reasoning:
                    # continue the answer where the last engine stopped (llama-server continues a trailing
                    # assistant message)
                    partial = {"role": "assistant", "content": content}
                    if reasoning:
                        partial["reasoning_content"] = reasoning
                    req["messages"] = list(body.get("messages", [])) + [partial]
                    if limit:
                        for k in ("max_tokens", "max_completion_tokens", "n_predict"):
                            if k in req:
                                req[k] = max(1, limit - pieces)
                    self.event(f": lodger restarted the model on the {e.where}; continuing the answer")
                c = None
                try:
                    c, r = self.upstream(e, req)
                    if r.status != 200:
                        err = r.read().decode(errors="replace")
                        if e.alive() and lodger.engine is e:
                            self.event(f"data: {json.dumps({'error': {'message': err[:500]}})}")
                            return
                        raise OSError("engine went away")
                    while True:
                        line = r.readline()
                        if not line:
                            raise OSError("stream ended early")
                        s = line.decode(errors="replace").rstrip("\r\n")
                        if not s:
                            continue
                        if s.startswith("data: ") and s != "data: [DONE]":
                            try:
                                d = json.loads(s[6:])
                                delta = (d.get("choices") or [{}])[0].get("delta") or {}
                                if delta.get("content") or delta.get("reasoning_content"):
                                    pieces += 1
                                content += delta.get("content") or ""
                                reasoning += delta.get("reasoning_content") or ""
                            except ValueError:
                                pass
                        self.event(s)
                        if s == "data: [DONE]":
                            return
                except (OSError, http.client.HTTPException):
                    if self.client_gone:
                        return
                    # the engine moved: wait until it is gone or replaced (a short hiccup retries the same one)
                    t0 = time.time()
                    while time.time() - t0 < 5 and lodger.engine is e and e.alive():
                        time.sleep(0.2)
                finally:
                    if c:
                        c.close()                   # stops the engine writing if the app went away
            self.event(f'data: {json.dumps({"error": {"message": "the model kept moving; answer cut short"}})}')

        client_gone = False

        def event(self, s):
            try:
                self.wfile.write((s + "\n\n").encode())
                self.wfile.flush()
            except OSError:
                self.client_gone = True
                raise

    return H


def serve(argv):
    ap = argparse.ArgumentParser(prog="lodger serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--allow-gpu", action="store_true", help="the graphics card may be used")
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--name", help="model name apps see (default: from the file)")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--min-ctx", type=int, default=planner.MIN_CTX)
    ap.add_argument("--leave-gpu", type=int, default=0, metavar="MiB")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--think", action="store_true", help="let the model reason before answering")
    ap.add_argument("--idle", type=float, default=0, metavar="MIN",
                    help="unload the model after MIN minutes unused, and load it on the next request "
                         "(it then also waits for the first request before loading); 0 = keep it loaded")
    a = ap.parse_args(argv)
    a.name = a.name or re.sub(r"(?i)\.gguf$", "", os.path.basename(a.model))
    lodger = Lodger(a)
    httpd = ThreadingHTTPServer((HOST, a.port), make_handler(lodger))
    httpd.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    log(f"lodger serving {a.name} at http://{HOST}:{a.port}/v1 (only this PC); logs in {lodger.out}")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        if lodger.asleep:
            lodger.why = "asleep until the first request"
            log("waiting for the first request before loading the model")
        else:
            lodger.bring_up(want_card=a.allow_gpu)
        threading.Thread(target=lodger.monitor, daemon=True).start()
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log("stopping")
    finally:
        lodger.closing = True
        httpd.shutdown()
        for e in Engine.all:                     # also one still loading for a move
            e.stop()
    return 0


def ask(argv):
    ap = argparse.ArgumentParser(prog="lodger ask")
    ap.add_argument("question")
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("-n", type=int, default=512, help="most words in the answer")
    a = ap.parse_args(argv)
    body = {"model": "lodger", "stream": True, "max_tokens": a.n,
            "messages": [{"role": "user", "content": a.question}]}
    c = http.client.HTTPConnection(HOST, a.port, timeout=900)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        print(f"error {r.status}: {r.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    t0, words, timings = time.time(), 0, None
    for line in r:
        s = line.decode(errors="replace").strip()
        if s.startswith(": lodger"):
            print(f"\n[{s[2:]}]", file=sys.stderr, flush=True)
        elif s.startswith("data: ") and s != "data: [DONE]":
            d = json.loads(s[6:])
            if "error" in d:
                print(f"\n[error: {d['error'].get('message')}]", file=sys.stderr)
                return 1
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            text = delta.get("content") or ""
            words += bool(text)
            print(text, end="", flush=True)
            timings = d.get("timings") or timings
    print(flush=True)
    if timings:
        print(f"[{timings.get('predicted_per_second', 0):.1f} words/s writing, {time.time() - t0:.0f} s total]", file=sys.stderr)
    return 0
