"""lodger serve: keep your local models ready for apps on this PC, and move them out of your way and back.

  lodger serve [MODEL.gguf ...] [--allow-gpu] [--port 8181] [--idle MIN] [--api-key-file PATH]
  lodger ask "question" [--model NAME] [--port 8181] [--api-key-file PATH]
  lodger pi [pi options]

Apps talk to lodger at http://127.0.0.1:8181/v1/chat/completions (the usual OpenAI-style API); only
this PC can reach it, and web pages in your browser can't use it (only pages served from this PC can).
With --api-key-file or LODGER_API_KEY, apps must also send the key (Authorization: Bearer KEY). It serves every model in ~/.config/lodger/models.toml (or the files given), one at
a time: a request naming another model switches to it once nothing is being written. Behind it, lodger
runs our llama.cpp build's server on a private port, with the same plan, fence and watchdog as `lodger run`.
pi lists all of them: `lodger serve` writes a "lodger" provider into pi's models.json.
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
import contextlib
import hmac
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import catalog, plan as planner, sysmem
from .launch import GAME_GUARD_FLAG, RUNS, Watchdog, gpu_blockers, router_coming, unit_cgroup, unit_props
from .neighbors import Neighbors

MiB, GiB = sysmem.MiB, sysmem.GiB
HOST = "127.0.0.1"          # only this PC; part of the approved exception (../CLAUDE.md)
QUIET_SECONDS = 60          # the card must be quiet this long before the model grows back
MOVE_WAIT = 600             # how long a request waits for its model (a switch may wait for an answer)
PAUSE_IDLE_VRAM = 128 * MiB  # a --pause service holding less graphics memory than this counts as idle
RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
LOCAL_NAMES = ("127.0.0.1", "localhost", "[::1]")
APP_SCHEMES = ("app", "file", "tauri", "vscode-webview", "vscode-file")


def unit_active(unit):
    return subprocess.run(["systemctl", "--user", "is-active", "-q", unit]).returncode == 0


def unit_vram(unit):
    """Graphics memory the processes of a systemd user service hold on the card."""
    cg = unit_cgroup(unit)
    try:
        with open(cg + "/cgroup.procs") as f:
            pids = [int(x) for x in f.read().split()]
    except (OSError, TypeError):
        return 0
    pci = (sysmem.dgpu() or {}).get("pci")
    seen = {}
    for pid in pids:
        seen.update(sysmem.gpu_clients(pid))
    return sum(v["vram"] for k, v in seen.items() if k[0] == pci)


def pause_flag(unit):
    return os.path.join(RUNTIME_DIR, f"lodger.paused-{unit}")


def unpause_all(argv=()):
    """Start again every service lodger paused (also after a crash: `lodger unpause`)."""
    for name in sorted(os.listdir(RUNTIME_DIR)):
        if name.startswith("lodger.paused-"):
            unit = name[len("lodger.paused-"):]
            subprocess.run(["systemctl", "--user", "start", unit], capture_output=True)
            os.remove(os.path.join(RUNTIME_DIR, name))
            log(f"started {unit} again")
    return 0


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

    def __init__(self, plan, entry, out):
        Engine.count += 1
        self.plan, self.entry, self.out = plan, entry, out
        self.port = free_port()
        self.unit = f"lodger-{os.getpid()}-{Engine.count}"
        self.where = "card" if plan["ngl"] else "processor"
        # its own key, so other programs on this PC can't use its private port; handed over in a file only
        # we can read (a command line is visible to every user), removed once the engine has read it
        self.key = secrets.token_urlsafe(32)
        self.keyfile = os.path.join(RUNTIME_DIR, self.unit + ".key")
        fd = os.open(self.keyfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(self.key + "\n")
        p = plan
        cmd = [os.path.join(planner.BIN, "llama-server"), "-m", p["model"], "-c", str(p["ctx"]),
               "-t", str(p["threads"]), "--fit", "off", "-np", "1", "--host", HOST, "--port", str(self.port),
               "--api-key-file", self.keyfile, "--no-webui", "-a", entry.name] + entry.args()
        if p["ngl"]:
            cmd += ["--device", "ROCm0", "-ngl", str(p["ngl"])] + (["-ncmoe", str(p["n_cpu_moe"])] if p["n_cpu_moe"] else [])
            env = {"HIP_VISIBLE_DEVICES": "0"}
        else:
            cmd += ["--device", "none", "-ngl", "0"]
            env = {"HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"}
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

    def auth(self):
        return {"Authorization": f"Bearer {self.key}"}

    def forget_keyfile(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.keyfile)

    def wait_ready(self, limit=600):
        while time.time() - self.started < limit and self.alive():
            try:
                c = http.client.HTTPConnection(HOST, self.port, timeout=2)
                c.request("GET", "/health", headers=self.auth())
                if c.getresponse().status == 200:
                    self.forget_keyfile()
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
        self.forget_keyfile()
        if self.logf.closed:
            return
        self.wd.stop_flag.set()
        self.wd.join(timeout=5)
        self.logf.close()
        # learn: planned vs really used on the card (only once it had loaded)
        if self.plan["ngl"] and time.time() - self.started > 20:
            with open(os.path.join(RUNS, "calibration.jsonl"), "a") as f:
                f.write(json.dumps({"time": time.strftime("%F %T"), "model": self.entry.name, "ngl": self.plan["ngl"],
                                    "n_cpu_moe": self.plan["n_cpu_moe"], "ctx": self.plan["ctx"],
                                    "gpu_est": self.plan["gpu_est"], "llama_gpu_est": self.plan.get("llama_gpu_est"),
                                    "gpu_real": self.wd.peak["model_vram"] or None, "stopped": self.wd.reason,
                                    "run": os.path.basename(self.out), "via": "serve"}) + "\n")


class Lodger:
    def __init__(self, a, models, default):
        self.a = a
        self.models = models                  # name -> catalog.Entry
        self.cur = models[default]            # the model loaded (or to load next)
        self.default = default
        self.switch_to = None                 # a request is waiting for this model
        self.busy = {}                        # engine -> answers being written on it now
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
        self.asleep = bool(a.idle)            # with --idle: load a model on the first request, not at start
        self.wake = False
        self.last_used = time.time()
        self.out = os.path.join(RUNS, time.strftime("%Y-%m-%d-%H%M%S") + "-serve")
        os.makedirs(self.out)

    def note(self, text):
        log(text)
        self.events = (self.events + [(time.strftime("%H:%M:%S"), text)])[-20:]

    def find(self, name):
        """The model a request names: exact or any case; none/"lodger" = the loaded one."""
        if not name or name == "lodger":
            return self.cur
        e = self.models.get(name) or next((m for n, m in self.models.items() if n.lower() == str(name).lower()), None)
        return e if e and not e.skip else None

    def available(self):
        return [e for e in self.models.values() if not e.skip]

    def decide_ctx(self, entry):
        """A model's context for this session: automatic (see plan.CTX_CAP/CTX_SHARE) for where it may run,
        the shorter of card and processor, so a conversation that fits on the card still fits after a move
        to the processor. Planning only, nothing loads."""
        if entry.ctx or entry.skip:
            return
        ctxs = []
        try:
            p = planner.make(entry.path, cpu_only=True, threads=self.a.threads, why_gpu_off="deciding the context",
                             extra=entry.planning_args())
            if not p["refuse"]:
                ctxs.append(p["ctx"])
            if self.a.allow_gpu:
                # as if lodger's own model weren't on the card: that memory is freed before this one loads
                e, card = self.engine, sysmem.dgpu()
                mine = (e.wd.peak["model_vram"] or e.plan["gpu_bytes"]) if e and e.where == "card" and e.alive() else 0
                p = planner.make(entry.path, leave_gpu=self.a.leave_gpu * MiB, threads=self.a.threads,
                                 extra=entry.planning_args(), vram_free=sysmem.vram_free(card) + mine if card else None)
                if not p["refuse"] and p["ngl"]:
                    ctxs.append(p["ctx"])
        except (Exception, SystemExit) as ex:   # e.g. a file type this llama.cpp can't read
            entry.skip = f"can't be read by this llama.cpp build ({str(ex)[:120]})"
            log(f"{entry.name}: {entry.skip}")
            return
        entry.ctx = min(ctxs) if ctxs else planner.MIN_CTX

    def prepare(self):
        """At start: work out every model's context (cached after the first time), then tell pi about them."""
        for e in list(self.models.values()):
            self.decide_ctx(e)
        ok = self.available()
        log(f"{len(ok)} models ready to serve: " + ", ".join(f"{e.name} ({e.ctx})" for e in ok))
        for e in self.models.values():
            if e.skip:
                log(f"not served: {e.name}: {e.skip}")
        if self.a.port == 8181:
            write_pi_config(self.a.port, [model_info(e) for e in ok], self.default,
                            pi_key(self.a.api_key, self.a.api_key_file))

    def make_plan(self, on_card):
        e = self.cur
        self.decide_ctx(e)
        if e.skip:
            return {"refuse": e.skip}
        try:
            return planner.make(e.path, target_ctx=e.ctx, min_ctx=e.ctx if on_card else self.a.min_ctx,
                                cpu_only=not on_card, extra=e.planning_args(),
                                leave_gpu=self.a.leave_gpu * MiB, threads=self.a.threads, device_check=on_card,
                                why_gpu_off=None if on_card else (self.why_off_card or "the card is not wanted now"))
        except (Exception, SystemExit) as ex:   # a failed llama.cpp probe must not take the server down
            return {"refuse": f"planning failed: {ex}"}

    def pausable_busy(self):
        for u in self.a.pause:
            if unit_active(u) and unit_vram(u) >= PAUSE_IDLE_VRAM:
                return f"{u} has a model loaded on the graphics card (it's being used)"
        return None

    def others_coming(self):
        """Is another AI server about to need the card? Services in --pause count only while busy."""
        if os.path.exists(GAME_GUARD_FLAG):
            return "llama-game-guard paused llama-router for a game"
        busy = self.pausable_busy()
        if busy:
            return busy
        return None if "llama-router" in self.a.pause else router_coming()

    def pause_others(self):
        """Stop idle --pause services before the model takes the card (they could load a model at any moment)."""
        for u in self.a.pause:
            if unit_active(u):
                subprocess.run(["systemctl", "--user", "stop", u], capture_output=True)
                open(pause_flag(u), "w").close()
                self.note(f"paused {u} (idle) while the model uses the graphics card")

    def unpause(self):
        for u in self.a.pause:
            if os.path.exists(pause_flag(u)):
                subprocess.run(["systemctl", "--user", "start", u], capture_output=True)
                os.remove(pause_flag(u))
                self.note(f"started {u} again (the model left the graphics card)")

    def launch(self, want_card, card_only=False):
        e = self._launch(want_card, card_only)
        if not (e and e.where == "card") and not (self.engine and self.engine.where == "card" and self.engine.alive()):
            self.unpause()                       # the card isn't ours after all
        return e

    def _launch(self, want_card, card_only=False):
        """Start an engine for the current model (card if wanted and possible, else processor) and wait until
        it answers. card_only: give up instead of falling back to the processor. Returns the Engine, or None."""
        if want_card:
            self.why_off_card = self.a.allow_gpu and self.pausable_busy()
            if not self.why_off_card:
                if self.a.allow_gpu:
                    self.pause_others()
                self.why_off_card = gpu_blockers(self.a.allow_gpu, sysmem.dgpu())
        if card_only and self.why_off_card:
            return None
        p = self.make_plan(want_card and not self.why_off_card)
        if p["refuse"] or (card_only and not p["ngl"]):
            if p["refuse"]:
                self.note(f"can't start {self.cur.name}: {p['refuse']}")
            return None
        n = Engine.count + 1
        out = os.path.join(self.out, f"engine-{n}-{self.cur.name}-{'card' if p['ngl'] else 'processor'}")
        if self.closing:
            return None
        os.makedirs(out)
        e = Engine(p, self.cur, out)
        self.note(f"starting {self.cur.name} on the {e.where}"
                  + (f" ({self.why_off_card})" if self.why_off_card and not p["ngl"] else "")
                  + f", context {p['ctx']}" + (f", {p['ngl']} layers on the card" if p["ngl"] else ""))
        if not e.wait_ready():
            reason = e.wd.reason or "it exited while loading (see engine.log)"
            if e.alive():
                e.stop()
            else:
                e.finish()
            self.note(f"the {e.where} engine didn't come up: {reason}")
            return None
        self.note(f"{self.cur.name} ready on the {e.where} after {time.time() - e.started:.0f} s")
        return e

    def install(self, e):
        with self.cv:
            old, self.engine, self.ready = self.engine, e, True
            self.why = f"{e.entry.name} on the {e.where}" + (
                f": {self.why_off_card}" if e.where == "processor" and self.why_off_card else "")
            self.card_since = time.time() if e.where == "card" else None
            self.cv.notify_all()
        return old

    def bring_up(self, want_card, tries=None):
        """Start an engine for the current model, retrying on the processor until one is up (or `tries`
        attempts failed: then the model is marked as failed and False returned)."""
        delay, n = 10, 1
        e = self.launch(want_card)
        while not e and not self.closing:
            if tries and n >= tries:
                return False
            time.sleep(delay)
            delay, n = min(delay * 2, 300), n + 1
            e = self.launch(False)
        if e:
            self.install(e)
        return bool(e)

    def wait_engine(self, entry, limit=MOVE_WAIT):
        """The engine for `entry`, once it's loaded and ready (asks for a switch or a wake-up if needed);
        None if that doesn't happen within `limit` seconds or the model can't run."""
        end = time.time() + limit
        with self.cv:
            while True:
                self.last_used = time.time()
                if entry.skip:
                    return None
                if (self.cur is entry and self.ready and self.engine and self.engine.alive()
                        and self.switch_to in (None, entry)):
                    return self.engine
                if self.cur is not entry and self.switch_to is None:
                    self.switch_to = entry            # first in line: the monitor switches once nothing is written
                if self.asleep:
                    self.wake = True
                left = end - time.time()
                if left <= 0:
                    return None
                self.cv.wait(min(1.0, left))

    @contextlib.contextmanager
    def using(self, e):
        with self.cv:
            self.busy[e] = self.busy.get(e, 0) + 1
        try:
            yield
        finally:
            with self.cv:
                self.busy[e] -= 1
                self.cv.notify_all()

    def stepped_aside(self):
        self.last_aside = time.time()
        if self.card_since and time.time() - self.card_since < 600:
            self.quiet_needed = min(self.quiet_needed * 2, 1800)   # it came back too soon: wait longer next time
        elif self.card_since:
            self.quiet_needed = QUIET_SECONDS
        self.card_since = None

    def switch(self):
        """Unload the current model and load the one a request is waiting for."""
        with self.cv:
            target, old = self.switch_to, self.engine
            self.ready, self.engine, self.cur = False, None, target
            self.switch_to, self.asleep, self.wake = None, False, False
            self.why = f"switching to {target.name}"
        self.note(f"switching to {target.name}" + (f" (from {old.entry.name})" if old else ""))
        if old:
            old.stop()
            if old.where == "card":
                self.unpause()
        self.card_since, self.quiet_needed = None, QUIET_SECONDS
        if not self.bring_up(want_card=self.a.allow_gpu, tries=2):
            with self.cv:
                target.skip = target.skip or "it didn't start (see the engine logs)"
                self.asleep, self.cur = True, self.models[self.default]
                self.why = f"{target.name} couldn't start: {target.skip}"
                self.cv.notify_all()
            self.note(self.why)

    def monitor(self):
        """Switches models, steps aside / grows back, sleeps. Runs for the life of `lodger serve`."""
        nb = Neighbors()
        while not self.closing:
            time.sleep(1)
            if self.closing:
                return
            if nb.update() or self.others_coming():
                self.quiet_since = time.time()
            e = self.engine
            # another model wanted: switch once nothing is being written with this one
            if self.switch_to and self.switch_to is not self.cur and (not e or not self.busy.get(e)):
                self.switch()
                continue
            if self.asleep:
                if self.wake:
                    self.wake = False
                    with self.cv:
                        if self.switch_to:
                            self.cur, self.switch_to = self.switch_to, None
                    self.note(f"a request came in: loading {self.cur.name}")
                    with self.cv:
                        self.asleep = False
                    if not self.bring_up(want_card=self.a.allow_gpu, tries=2):
                        with self.cv:
                            self.cur.skip = self.cur.skip or "it didn't start (see the engine logs)"
                            self.asleep, self.why = True, f"{self.cur.name} couldn't start: {self.cur.skip}"
                            self.cur = self.models[self.default]
                            self.cv.notify_all()
                        self.note(self.why)
                continue
            # unused for a while: unload, so the memory is yours again until the next request
            if (self.a.idle and e and e.alive() and self.active == 0
                    and time.time() - self.last_used >= self.a.idle * 60):
                with self.cv:
                    self.ready, self.asleep, self.engine = False, True, None
                    self.why = f"asleep: unloaded after {self.a.idle:g} min unused (loads again on the next request)"
                self.note(f"unused for {self.a.idle:g} min: unloading {e.entry.name} until the next request")
                e.stop()
                self.card_since = None
                self.unpause()
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
                if e.where == "card":
                    self.unpause()
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
                self.unpause()
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
        return {"model": self.cur.name, "default": self.default, "ready": self.ready,
                "where": "asleep" if self.asleep else e.where if e else None, "why": self.why,
                "context": self.cur.ctx, "answers_in_progress": self.active,
                "switching_to": self.switch_to.name if self.switch_to else None,
                "quiet_needed_s": self.quiet_needed, "card_quiet_for_s": round(time.time() - self.quiet_since),
                "plan": {k: e.plan[k] for k in ("ngl", "n_cpu_moe", "ctx", "gpu_est")} if e else None,
                "models": {n: m.skip or "ok" for n, m in self.models.items()},
                "events": self.events}


def model_info(e):
    return {"id": e.name, "object": "model", "owned_by": "lodger", "context_window": e.ctx,
            "reasoning": e.reasoning, "meta": {"n_ctx": e.ctx}}


def local_host(value):
    """Is this Host header this PC (127.0.0.1, localhost or [::1], with or without a port)? A web page that
    points its own name at 127.0.0.1 (DNS rebinding) still sends its own name here."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        name, _, rest = v.partition("]")
        name += "]"
        if rest and not rest.startswith(":"):
            return False
        port = rest[1:] if rest else None
    else:
        name, colon, port = v.partition(":")
        port = port if colon else None
    return name in LOCAL_NAMES and (port is None or (port.isascii() and port.isdigit()))


def local_origin(value):
    """Is this Origin header a page served from this PC (http(s)://localhost etc.) or a desktop app's own
    page (app://, file://, tauri://, vscode-webview://, as Ollama allows; a browser never sends these)?
    "null" (a file opened in a browser, a sandboxed frame) is not."""
    scheme, sep, rest = (value or "").strip().partition("://")
    scheme = scheme.lower()
    return bool(sep) and (scheme in APP_SCHEMES or scheme in ("http", "https") and local_host(rest))


def check(method, path, headers, key=None):
    """May this request reach lodger? None if so, else (HTTP status, why in plain words). `headers` is any
    mapping with .get(); `key` is the API key apps must send (None: no key needed)."""
    host = headers.get("Host")
    if not local_host(host):
        return 403, (f"lodger only answers requests addressed to 127.0.0.1, localhost or [::1], not {host!r} "
                     "(a web page may be trying to reach it through its own name)")
    origin = headers.get("Origin")
    if origin is not None and not local_origin(origin):
        return 403, f"lodger doesn't answer web pages from other sites (this request came from {origin!r})"
    if method == "OPTIONS":                     # a browser asking first; it never sends the key with that
        return None
    if key and path.split("?")[0] != "/health":
        scheme, _, given = (headers.get("Authorization") or "").strip().partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(given.strip().encode(), key.encode()):
            return 401, ("lodger wants its API key: send the header 'Authorization: Bearer KEY' "
                         "(the key in lodger's --api-key-file or LODGER_API_KEY)")
    return None


def make_handler(lodger, key=None):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        cors = None                              # the local page's origin to allow in the answer, if any

        def log_message(self, *args):
            pass

        def send_json(self, code, obj, headers=()):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            if self.cors:
                self.send_header("Access-Control-Allow-Origin", self.cors)
                self.send_header("Vary", "Origin")
            super().end_headers()

        def refused(self):
            """Answer a request that may not reach lodger (see check()); True if it was refused."""
            origin = self.headers.get("Origin")
            self.cors = origin if origin and local_origin(origin) else None
            why = check(self.command, self.path, self.headers, key)
            if not why:
                return False
            code, msg = why
            extra = [("Connection", "close")]   # its body (if any) is left unread
            if code == 401:
                extra.append(("WWW-Authenticate", 'Bearer realm="lodger"'))
            self.send_json(code, {"error": {"message": msg, "type": "forbidden" if code == 403 else "unauthorized"}},
                           extra)
            return True

        def do_OPTIONS(self):
            if self.refused():
                return
            self.send_response(204)
            self.send_header("Allow", "GET, POST, OPTIONS")
            if self.cors:
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.refused():
                return
            if self.path == "/health":
                if lodger.asleep:                     # a request wakes it, so apps may go ahead
                    return self.send_json(200, {"status": "ok", "asleep": True})
                return self.send_json(200 if lodger.ready else 503, {"status": "ok" if lodger.ready else "loading"})
            if self.path == "/v1/models":
                return self.send_json(200, {"object": "list", "data": [model_info(e) for e in lodger.available()]})
            if self.path == "/lodger/status":
                return self.send_json(200, lodger.status())
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.refused():
                return
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
            entry = lodger.find(body.get("model"))
            if not entry:
                names = ", ".join(e.name for e in lodger.available())
                return self.send_json(404, {"error": {"message": f"lodger has no model called {body.get('model')!r}; "
                                                                 f"it has: {names}"}})
            with lodger.cv:
                lodger.active += 1
            try:
                if body.get("stream"):
                    self.stream(body, entry)
                else:
                    self.whole(body, entry)
            finally:
                with lodger.cv:
                    lodger.active -= 1
                    lodger.last_used = time.time()

        def upstream(self, e, body):
            c = http.client.HTTPConnection(HOST, e.port, timeout=600)
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json", **e.auth()})
            return c, c.getresponse()

        def unavailable(self, entry):
            return f"{entry.name} is not available right now: " + (entry.skip or lodger.why)

        def whole(self, body, entry):
            for _ in range(4):
                e = lodger.wait_engine(entry)
                if not e:
                    return self.send_json(503, {"error": {"message": self.unavailable(entry)}})
                try:
                    with lodger.using(e):
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
            self.send_json(503, {"error": {"message": "the model kept moving; try again"}})

        def stream(self, body, entry):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            content, reasoning, pieces, tooling = "", "", 0, False
            limit = body.get("max_tokens") or body.get("max_completion_tokens") or body.get("n_predict")
            for attempt in range(6):
                e = lodger.wait_engine(entry)
                if not e:
                    self.event(f'data: {json.dumps({"error": {"message": self.unavailable(entry)}})}')
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
                    with lodger.using(e):
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
                                    tooling = tooling or bool(delta.get("tool_calls"))
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
                    if tooling:                 # half a tool call can't be continued; the app asks again
                        self.event(f'data: {json.dumps({"error": {"message": "the model moved while writing a tool call; please retry"}})}')
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
    ap.add_argument("files", nargs="*", metavar="MODEL.gguf",
                    help=f"serve only these files (default: the models in {catalog.CONFIG})")
    ap.add_argument("--default", help="model to load when a request doesn't name one")
    ap.add_argument("--allow-gpu", action="store_true", help="the graphics card may be used")
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--min-ctx", type=int, default=planner.MIN_CTX)
    ap.add_argument("--leave-gpu", type=int, default=0, metavar="MiB")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--idle", type=float, default=0, metavar="MIN",
                    help="unload the model after MIN minutes unused, and load it on the next request "
                         "(it then also waits for the first request before loading); 0 = keep it loaded")
    ap.add_argument("--pause", action="append", default=[], metavar="SERVICE",
                    help="a systemd user service running another AI server (e.g. llama-router): while it is idle, "
                         "lodger stops it before using the graphics card and starts it again when it leaves the "
                         "card; while it has a model loaded, lodger stays off the card. Repeatable.")
    ap.add_argument("--api-key-file", metavar="PATH",
                    help="apps must send the key in this file (Authorization: Bearer KEY); LODGER_API_KEY does the "
                         "same. Default: no key (web pages still can't use lodger; other programs on this PC can)")
    a = ap.parse_args(argv)
    a.api_key = read_key(a.api_key_file)
    models, default = catalog.load(a.files or None)
    if not models:
        sys.exit(f"no models found: add folders or files to {catalog.CONFIG}")
    default = a.default or (default if default in models else None) or next(iter(models))
    lodger = Lodger(a, models, default)
    unpause_all()                                # left over from a crash
    httpd = ThreadingHTTPServer((HOST, a.port), make_handler(lodger, a.api_key))
    httpd.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    signal.signal(signal.SIGINT, signal.default_int_handler)   # also when started in the background of a script
    log(f"lodger serving {len(models)} models at http://{HOST}:{a.port}/v1 (only this PC"
        + (", API key required" if a.api_key else "") + f"); default {default}; logs in {lodger.out}")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        lodger.decide_ctx(lodger.cur)
        if lodger.asleep:
            lodger.why = "asleep until the first request"
            log("waiting for the first request before loading a model")
        else:
            lodger.bring_up(want_card=a.allow_gpu)
        threading.Thread(target=lodger.prepare, daemon=True).start()
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
        lodger.unpause()
    return 0


def ask(argv):
    ap = argparse.ArgumentParser(prog="lodger ask")
    ap.add_argument("question")
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--model", default="", help="which model (default: the one loaded)")
    ap.add_argument("-n", type=int, default=512, help="most words in the answer")
    ap.add_argument("--api-key-file", metavar="PATH", help="the key lodger serve wants, if any (or LODGER_API_KEY)")
    a = ap.parse_args(argv)
    body = {"model": a.model, "stream": True, "max_tokens": a.n,
            "messages": [{"role": "user", "content": a.question}]}
    c = http.client.HTTPConnection(HOST, a.port, timeout=900)
    c.request("POST", "/v1/chat/completions", json.dumps(body),
              {"Content-Type": "application/json", **auth_header(read_key(a.api_key_file))})
    r = c.getresponse()
    if r.status != 200:
        print(f"error {r.status}: {r.read().decode(errors='replace')}", file=sys.stderr)
        if r.status == 401:
            print("lodger serve was started with an API key: pass --api-key-file PATH or set LODGER_API_KEY",
                  file=sys.stderr)
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
            if delta.get("reasoning_content"):          # thinking: shown dimmed, apart from the answer
                print(f"\033[2m{delta['reasoning_content']}\033[0m", end="", file=sys.stderr, flush=True)
            text = delta.get("content") or ""
            words += bool(text)
            print(text, end="", flush=True)
            timings = d.get("timings") or timings
    print(flush=True)
    if timings:
        print(f"[{timings.get('predicted_per_second', 0):.1f} words/s writing, {time.time() - t0:.0f} s total]", file=sys.stderr)
    return 0

def read_key(path=None):
    """The API key: the first line of `path` that isn't empty or a #comment, else $LODGER_API_KEY, else None."""
    if path:
        try:
            with open(os.path.expanduser(path)) as f:
                lines = [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]
        except OSError as ex:
            sys.exit(f"can't read the API key file: {ex}")
        if not lines:
            sys.exit(f"the API key file {path} is empty")
        return lines[0]
    return os.environ.get("LODGER_API_KEY", "").strip() or None


def auth_header(key):
    return {"Authorization": f"Bearer {key}"} if key else {}


PI_DIR = os.environ.get("PI_CODING_AGENT_DIR") or os.path.expanduser("~/.pi/agent")


def pi_key(key, path=None):
    """What to put as the lodger provider's apiKey in pi's models.json: a key from a file is read by pi itself
    (`!cat FILE`, so the key isn't copied); one from LODGER_API_KEY is written as is (escaped for pi);
    no key: a placeholder, which lodger ignores."""
    if not key:
        return "lodger"
    if path:
        return "!cat " + shlex.quote(os.path.abspath(os.path.expanduser(path)))
    k = key.replace("$", "$$")
    return "$" + k if k.startswith("!") else k


def write_pi_config(port, models, default, api_key="lodger"):
    """Tell pi about lodger's models: a "lodger" provider in pi's models.json (other providers are kept),
    and lodger's default model as pi's default unless you chose another one in pi's settings.
    `api_key`: see pi_key()."""
    path = os.path.join(PI_DIR, "models.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    cfg.setdefault("providers", {})["lodger"] = {
        "baseUrl": f"http://{HOST}:{port}/v1", "apiKey": api_key, "api": "openai-completions",
        "models": [{"id": m["id"], "name": f"{m['id']} (local)", "reasoning": bool(m.get("reasoning")),
                    "contextWindow": m["context_window"], "maxTokens": min(32768, m["context_window"] // 4),
                    "input": ["text"], "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "compat": {"supportsDeveloperRole": False}} for m in models if m.get("context_window")]}
    os.makedirs(PI_DIR, exist_ok=True)
    # only you may read it (it can hold API keys, lodger's and other providers')
    with os.fdopen(os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(path + ".tmp", path)
    spath = os.path.join(PI_DIR, "settings.json")
    try:
        with open(spath) as f:
            settings = json.load(f)
    except (OSError, ValueError):
        settings = {}
    if settings.get("defaultProvider") in (None, "lodger"):
        settings.update(defaultProvider="lodger", defaultModel=default)
        with open(spath + ".tmp", "w") as f:
            json.dump(settings, f, indent=2)
        os.replace(spath + ".tmp", spath)


def pi(argv):
    """`lodger pi [pi options]`: pi with lodger's models (refreshes pi's list of them first). Plain `pi`
    works too: `lodger serve` writes the same list when it starts. With an API key, set LODGER_API_KEY
    (or rely on the list `lodger serve` wrote)."""
    key = read_key()
    try:
        c = http.client.HTTPConnection(HOST, 8181, timeout=5)
        c.request("GET", "/v1/models", headers=auth_header(key))
        r = c.getresponse()
        data = r.read()
        if r.status == 401:
            data = None
            print("lodger wants an API key (set LODGER_API_KEY to refresh the model list); "
                  "using the list lodger serve wrote", file=sys.stderr)
        else:
            models = json.loads(data)["data"]
            c.request("GET", "/lodger/status", headers=auth_header(key))
            default = json.loads(c.getresponse().read())["default"]
    except (OSError, ValueError, KeyError):
        print("lodger isn't running. Start it:  systemctl --user start lodger", file=sys.stderr)
        return 1
    if data is not None:
        write_pi_config(8181, models, default, pi_key(key))
    try:
        os.execvp("pi", ["pi"] + argv)
    except FileNotFoundError:
        print("pi isn't installed (https://github.com/earendil-works/pi)", file=sys.stderr)
        return 1
