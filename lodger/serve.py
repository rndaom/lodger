"""lodger serve: keep your local models ready for apps on this PC, and move them out of your way and back.

  lodger serve [MODEL.gguf ...] [--allow-gpu] [--port 8181] [--idle MIN] [--api-key-file PATH]
  lodger ask "question" [--model NAME] [--port 8181] [--api-key-file PATH]
  lodger pi [pi options]

Apps talk to lodger at http://127.0.0.1:8181/v1/chat/completions (the usual OpenAI-style API); only
this PC can reach it, and web pages in your browser can't use it (only pages served from this PC can).
With --api-key-file or LODGER_API_KEY, apps must also send the key (Authorization: Bearer KEY).
It serves every model in ~/.config/lodger/models.toml (or the files given), one at a time:
a request naming another model switches to it once nothing is being written. Behind it, lodger runs
llama.cpp's server on a private port, with the same plan, fence and watchdog as `lodger run`.
pi (https://github.com/earendil-works/pi) can list all of them: `lodger pi` writes a "lodger" provider into
pi's models.json (and `lodger serve` keeps it current when extras.pi is on in config.toml).
- Steps aside: when another app needs the graphics card, the model is stopped on the card and started
  again on the processor at once. An answer being written continues where it stopped.
- Grows back: when the card has been quiet for a while and no answer is being written, the model moves
  back onto the card.
- Sleeps (with --idle MIN): unloads the model after MIN minutes unused and loads it again on the next
  request, so it holds no memory while you don't use it.
GET /lodger/status says where the model is and why. POST /lodger/move moves it off the card now
(it grows back later as usual). With --ollama (or `ollama = true` in config.toml), apps that only speak
Ollama's API find lodger at http://127.0.0.1:11434 too."""
import argparse
import http.client
import json
import os
import contextlib
import hmac
import secrets
import shlex
import signal
import stat
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import answers, catalog, config, engine, fence, gpu as gpus, ollama, plan as planner, reserve, sysmem
from .launch import GAME_GUARD_FLAG, RUNS, Watchdog, card_fields, gpu_blockers, router_coming, unit_cgroup, unit_props
from .neighbors import Neighbors

MiB, GiB = sysmem.MiB, sysmem.GiB
MOVING = 256 * MiB        # lodger memory that changed this much between two looks is on the move
HOST = "127.0.0.1"          # only this PC; part of the approved exception (../CLAUDE.md)
QUIET_SECONDS = 60          # the card must be quiet this long before the model grows back
MOVE_WAIT = 600             # how long a request waits for its model (a switch may wait for an answer)
GROW_WAIT = 60              # growing back: how long an answer on the processor may finish before it moves to the card
FAIL_COOLDOWN = 300         # a model that didn't start is tried again on a request after this long
PING = 15                   # while the engine reads a long conversation: keep the app's connection open this often
SILENCE = 300               # an engine that started answering and then says nothing for this long is stuck
MAX_BODY = 64 * MiB         # largest request lodger reads
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
    return sysmem.gpu_use(sysmem.dgpu(), pids)["vram"]


def pause_flag(unit):
    return os.path.join(RUNTIME_DIR, f"lodger.paused-{unit}")


def lodger_alive(pid):
    """Is `pid` a lodger that is still running (not a dead one's number, reused by another program)?"""
    if pid == os.getpid():
        return True
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"lodger" in f.read()
    except OSError:
        return False


def pause_owner(unit):
    """The pid of the lodger that paused `unit` (None: nobody, or an old flag without one)."""
    try:
        with open(pause_flag(unit)) as f:
            return int(f.read().strip() or 0) or None
    except (OSError, ValueError):
        return None


def unpause_all(argv=()):
    """Start again every service a lodger paused and can't start again itself, because it is gone (after a
    crash: `lodger unpause`). Services another running lodger paused stay paused: its model is on the card."""
    for name in sorted(os.listdir(RUNTIME_DIR)):
        if name.startswith("lodger.paused-"):
            unit = name[len("lodger.paused-"):]
            owner = pause_owner(unit)
            if owner and owner != os.getpid() and lodger_alive(owner):
                log(f"{unit} stays paused: lodger {owner} uses the graphics card and starts it again itself")
                continue
            subprocess.run(["systemctl", "--user", "start", unit], capture_output=True)
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(RUNTIME_DIR, name))
            log(f"started {unit} again")
    return 0


def log(msg):
    """A line on lodger's output. Never raises: a closed terminal (EIO) must not stop the monitor halfway."""
    with contextlib.suppress(OSError, ValueError):
        print(time.strftime("%H:%M:%S ") + msg, flush=True)


def free_port():
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def port_owned(port, pids):
    """Is the program listening on HOST:port one of `pids`? (The port was free when lodger picked it; another
    program could have taken it before the engine did, and must not get the engine's key or your chats.)"""
    want = f"{socket.inet_aton(HOST)[::-1].hex().upper()}:{port:04X}"
    inodes = set()
    try:
        with open("/proc/net/tcp") as f:
            for line in f.readlines()[1:]:
                x = line.split()
                if len(x) > 9 and x[1] == want and x[3] == "0A":      # 0A: listening
                    inodes.add(f"socket:[{x[9]}]")
    except OSError:
        return False
    for pid in pids:
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            with contextlib.suppress(OSError):
                if os.readlink(f"/proc/{pid}/fd/{fd}") in inodes:
                    return True
    return False


class Engine:
    """One llama-server process in its own fence (see fence.py), with a watchdog."""
    count = 0
    all = []

    def __init__(self, plan, entry, out, card=planner.AUTO):
        Engine.count += 1
        card = sysmem.dgpu() if card is planner.AUTO else card     # one card for the command, watchdog and notes
        self.plan, self.entry, self.out = plan, entry, out
        self.port = free_port()
        self.where = "card" if plan["ngl"] else "processor"
        self.name = f"lodger-{os.getpid()}-{Engine.count}"
        self.stopping, self.ready_at, self.logf = False, None, None
        # its own key, so other programs on this PC can't use its private port; handed over in a file only
        # we can read (a command line is visible to every user), removed once the engine has read it
        self.key = secrets.token_urlsafe(32)
        self.keyfile = os.path.join(RUNTIME_DIR, self.name + ".key")
        fd = os.open(self.keyfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(self.key + "\n")
        p = plan
        try:
            cmd = [engine.tool("llama-server"), "-m", p["model"], "-c", str(p["ctx"]),
                   "-t", str(p["threads"]), "--fit", "off", "-np", "1", "--host", HOST, "--port", str(self.port),
                   "--api-key-file", self.keyfile, "--no-webui", "-a", entry.name,
                   "--cache-ram", str(p.get("cache_ram_mib", 0))] + entry.args()
            dargs, env = engine.device_args(p["ngl"], card, p["n_cpu_moe"])
            cmd += dargs
            with open(os.path.join(out, "plan.json"), "w") as f:
                json.dump({**p, "cmd": cmd, "env": env, "fence": fence.mode()[0]}, f, indent=1)
            self.logf = open(os.path.join(out, "engine.log"), "w")
            self.started = time.time()
            # own session: Ctrl-C in the terminal reaches lodger only, which then stops the engine itself
            self.fence = fence.Fenced(cmd, env, unit_props(p), stdout=self.logf, stderr=subprocess.STDOUT,
                                      new_session=True, name=self.name)
        except BaseException:
            self.forget_keyfile()
            if self.logf:
                self.logf.close()
            raise
        Engine.all.append(self)                  # from here on, lodger's exit stops it whatever happens next
        self.proc = self.fence.proc
        try:
            self.wd = Watchdog(self.fence, bool(p["ngl"]), out, log=os.path.join(out, "engine.log"), keep=p["keep"],
                               card=card)
            self.wd.start()
        except BaseException:                    # never leave a model running without its watchdog
            self.fence.stop()
            Engine.all.remove(self)
            self.forget_keyfile()
            self.logf.close()
            raise
        try:
            self.card = card_fields(bool(p["ngl"]), card)   # which card and backend, for calibration
        except (Exception, SystemExit):
            self.card = {}

    def alive(self):
        return self.proc.poll() is None

    def watched(self):
        """Is the watchdog still on duty? (If its thread died, nothing would stop the model in time.)"""
        return self.wd.healthy(stale=10) or bool(self.wd.reason) or self.wd.stop_flag.is_set()

    def auth(self):
        return {"Authorization": f"Bearer {self.key}"}

    def forget_keyfile(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.keyfile)

    def wait_ready(self, limit=600, abort=lambda: False):
        """Wait until the engine answers. /health needs no key, and the key is only sent once the program on the
        port is known to be the engine (see port_owned)."""
        while time.time() - self.started < limit and self.alive() and not abort():
            try:
                c = http.client.HTTPConnection(HOST, self.port, timeout=2)
                c.request("GET", "/health")
                ok = c.getresponse().status == 200
                c.close()
                if ok:
                    if not port_owned(self.port, self.fence.pids()):
                        self.wd.reason = self.wd.reason or f"another program took the engine's port {self.port}"
                        return False
                    self.forget_keyfile()
                    self.ready_at = time.time()
                    return True
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.5)
        return False

    def stop(self):
        """Planned stop (not the watchdog). The watchdog keeps watching until the model is really gone: if the
        stop fails, it still stops the model when memory runs low."""
        self.stopping = True
        self.fence.stop()
        self.wd.stop_flag.set()
        self.finish()

    def finish(self):
        self.forget_keyfile()
        if self.logf.closed:
            return
        self.wd.stop_flag.set()
        self.wd.join(timeout=5)
        self.logf.close()
        if self.fence.pids():
            self.fence.stop()
        else:
            self.fence.close()
        with contextlib.suppress(ValueError):
            Engine.all.remove(self)
        # learn: planned vs really used on the card (only once it had loaded and run a little)
        if self.plan["ngl"] and self.ready_at and time.time() - self.ready_at > 10:
            with open(os.path.join(RUNS, "calibration.jsonl"), "a") as f:
                f.write(json.dumps({"time": time.strftime("%F %T"), "model": self.entry.name, "ngl": self.plan["ngl"],
                                    "n_cpu_moe": self.plan["n_cpu_moe"], "ctx": self.plan["ctx"],
                                    "gpu_est": self.plan["gpu_est"], "llama_gpu_est": self.plan.get("llama_gpu_est"),
                                    "gpu_real": self.wd.peak["model_vram"] or None, "stopped": self.wd.reason,
                                    "run": os.path.basename(self.out), "via": "serve", **self.card}) + "\n")


class Lodger:
    def __init__(self, a, models, default):
        self.a = a
        self.models = models                  # name -> catalog.Entry
        self.cur = models[default]            # the model loaded (or to load next)
        self.default = default
        self.switch_to = None                 # a request is waiting for this model
        self.switch_ticket = 0                # ... and its place in line: requests that came before it go first
        self.ticket = 0                       # requests take numbers as they come in
        self.waiting = {}                     # requests waiting for an engine: token -> (model, ticket)
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
        self.last_note = 0.0
        self.last_fail = None                 # why the last engine didn't start
        self.failed = {}                      # name -> (time it may be tried again, why it didn't start)
        self.failures = 0                     # the current model's starts that failed in a row
        self.retry_at, self.retry_delay = 0.0, 10
        self.ollama = None                    # where Ollama's API is answered, or why not
        self.t_start = time.time()
        self.last_power = 0.0                 # when the power source was last looked at
        self.reload_wanted = None             # why the model list is to be read again (SIGHUP), or None
        self.reloading = threading.Lock()
        self.conf = {n: self.fingerprint(e) for n, e in models.items()}   # each model as models.toml had it
        self.models_file = catalog.CONFIG
        self.models_seen, self.models_changed_at = self.models_stat(), None
        self.models_read, self.models_error = time.time(), None
        self.monitor_error = None
        self.clock_off = time.time() - time.monotonic()   # moves when the PC sleeps or the clock is set (clock())
        self.grace_until = 0.0                # monotonic: until then a stale watchdog heartbeat may be the clock's doing
        self.out = os.path.join(RUNS, time.strftime("%Y-%m-%d-%H%M%S") + "-serve")
        os.makedirs(self.out)

    def clock(self):
        """Did the PC sleep, or the clock get set, since the last look? The monotonic clock stops while the PC is
        suspended, the wall clock doesn't. After such a jump the watchdog's heartbeat (wall time) looks stale until
        its next round, so it gets a short grace instead of its model being stopped; and the card wasn't watched
        meanwhile, so its quiet time starts again."""
        off = time.time() - time.monotonic()
        jump, self.clock_off = off - self.clock_off, off
        if abs(jump) > 5:
            self.grace_until = time.monotonic() + 15
            self.quiet_since = time.time()
            self.note(f"the PC slept or its clock was set ({jump / 60:+.0f} min): carrying on")
        return jump

    def watched(self, e):
        """Is engine `e`'s watchdog on duty (or, just after the clock jumped, not yet known to be off it)?"""
        self.clock()
        return e.watched() or time.monotonic() < self.grace_until

    @staticmethod
    def fingerprint(e):
        """A model as the model list gives it (file, settings, context, skip) and its file's size and time: when
        this changes, lodger plans the model afresh."""
        try:
            st = os.stat(e.path)
            stamp = (st.st_mtime_ns, st.st_size)
        except (OSError, TypeError):
            stamp = None
        return (e.path, json.dumps(getattr(e, "settings", None), sort_keys=True, default=str), e.ctx, e.skip, stamp)

    def models_stat(self):
        try:
            st = os.stat(self.models_file)
            return st.st_mtime_ns, st.st_size
        except OSError:
            return None

    def watch_models(self, now):
        """Read the model list again once models.toml changed and then stayed the same for 3 s (an editor may
        still be writing it)."""
        seen = self.models_stat()
        if seen != self.models_seen:
            self.models_seen, self.models_changed_at = seen, now
        elif self.models_changed_at and now - self.models_changed_at >= 3:
            self.models_changed_at = None
            self.reload(f"{self.models_file} changed")

    def reload(self, why):
        """Read the model list again (models.toml, or the files lodger was started with): new models are served,
        removed ones no longer (an answer being written finishes first), and a model whose settings or file changed
        is planned afresh from its next start (the same object, so requests in flight keep it). A list that can't
        be read leaves the models as they were. True if it was read."""
        with self.reloading:
            try:
                models, default = catalog.load(self.a.files or None, self.models_file)
                if not models:
                    raise ValueError("it lists no models")
            except (OSError, ValueError, SystemExit) as ex:   # tomllib's errors are ValueErrors
                self.models_error = f"{self.models_file} couldn't be read again ({ex}); serving the models read before"
                self.note(self.models_error)
                return False
            changes = []
            with self.cv:
                out = {}
                for name, new in models.items():
                    cur, fp = self.models.get(name), self.fingerprint(new)
                    if cur is None:
                        out[name] = new
                        changes.append(f"added {name}")
                    else:
                        out[name] = cur
                        if self.conf.get(name) != fp:
                            loaded = self.engine and self.engine.entry is cur
                            cur.__dict__.update(new.__dict__)
                            changes.append(f"changed {name}" + (" (from its next start)" if loaded else ""))
                    self.conf[name] = fp
                for name, cur in self.models.items():
                    if name not in out:
                        cur.skip = "removed from the model list"
                        self.conf.pop(name, None)
                        changes.append(f"removed {name}")
                self.models = out
                self.default = (self.a.default if self.a.default in out else default if default in out
                                else next(iter(out)))
                if self.switch_to is not None and self.switch_to.skip:
                    self.switch_to = None
                if self.cur is not out.get(self.cur.name):        # the loaded model was removed
                    if self.engine:
                        self.switch_to = self.switch_to or out[self.default]   # once its answers are written
                        self.switch_ticket = 0
                    else:
                        self.cur = out[self.default]
                self.models_read, self.models_error = time.time(), None
                self.cv.notify_all()
            self.note(f"read the model list again ({why}): " + (", ".join(changes) or "nothing changed"))
        threading.Thread(target=self.prepare, daemon=True).start()   # new models' contexts, pi's list
        return True

    def unplugged(self):
        """Why the model should leave the graphics card now: the PC runs on battery and gpu_on_battery is off
        (lodger doesn't start work on the card then, and work already there would drain the battery). None if not."""
        if config.get().get("gpu_on_battery"):
            return None
        card = sysmem.dgpu()
        if card is None or card.integrated:
            return None
        src, detail = gpus.power_source()
        if src != "battery":
            return None
        return f"{detail}: the graphics card is used only when plugged in (gpu_on_battery in config.toml)"

    def note(self, text):
        log(text)
        self.events = (self.events + [(time.strftime("%H:%M:%S"), text)])[-20:]

    def find(self, name):
        """The model a request names: exact or any case; none/"lodger" = the loaded one."""
        if not name or name == "lodger":
            return self.cur if not self.cur.skip else next(iter(self.available()), None)
        e = self.models.get(name) or next((m for n, m in self.models.items() if n.lower() == name.lower()), None)
        return e if e and not e.skip else None

    def available(self):
        return [e for e in self.models.values() if not e.skip]

    def cooling(self, entry):
        """Why `entry` didn't start a little while ago (lodger tries again after FAIL_COOLDOWN), or None."""
        until, why = self.failed.get(entry.name, (0, None))
        return why if time.time() < until else None

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
        except (Exception, SystemExit) as ex:   # e.g. a file type this llama.cpp can't read
            entry.skip = f"can't be read by this llama.cpp build ({str(ex)[:120]})"
            log(f"{entry.name}: {entry.skip}")
            return
        if not p["refuse"]:
            ctxs.append(p["ctx"])
        if self.a.allow_gpu:
            # as if lodger's own model weren't on the card: that memory is freed before this one loads
            try:
                e, card = self.engine, sysmem.dgpu()
                mine = (e.wd.peak["model_vram"] or e.plan["gpu_bytes"]) if e and e.where == "card" and e.alive() else 0
                free = card.free() if card else None    # None: can't be read, so plan from what's there now
                p = planner.make(entry.path, leave_gpu=self.a.leave_gpu * MiB, threads=self.a.threads,
                                 extra=entry.planning_args(), vram_free=free + mine if free is not None else None)
                if not p["refuse"] and p["ngl"]:
                    ctxs.append(p["ctx"])
            except (Exception, SystemExit) as ex:   # the processor plan worked: the model runs there at least
                log(f"{entry.name}: planning for the graphics card failed, context from the processor's plan ({str(ex)[:120]})")
        entry.ctx = min(ctxs) if ctxs else planner.MIN_CTX

    def prepare(self):
        """At start: work out every model's context (cached after the first time), then tell pi about them."""
        for e in list(self.models.values()):
            if self.closing:
                return
            self.decide_ctx(e)
        ok = self.available()
        log(f"{len(ok)} models ready to serve: " + ", ".join(f"{e.name} ({e.ctx})" for e in ok))
        for e in self.models.values():
            if e.skip:
                log(f"not served: {e.name}: {e.skip}")
        if self.a.port == 8181 and config.extra("pi"):
            write_pi_config(self.a.port, [model_info(e) for e in ok], self.default,
                            pi_key(self.a.api_key, self.a.api_key_file))

    def make_plan(self, on_card, card=planner.AUTO):
        e = self.cur
        self.decide_ctx(e)
        if e.skip:
            return {"refuse": e.skip}
        try:
            return planner.make(e.path, target_ctx=e.ctx, min_ctx=e.ctx if on_card else self.a.min_ctx,
                                cpu_only=not on_card, extra=e.planning_args(),
                                leave_gpu=self.a.leave_gpu * MiB, threads=self.a.threads, device_check=on_card, card=card,
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
        if config.extra("llama_router") and os.path.exists(GAME_GUARD_FLAG):
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
                with open(pause_flag(u), "w") as f:
                    f.write(f"{os.getpid()}\n")        # who starts it again (see unpause_all)
                self.note(f"paused {u} (idle) while the model uses the graphics card")

    def unpause(self):
        for u in self.a.pause:
            if os.path.exists(pause_flag(u)) and pause_owner(u) in (None, os.getpid()):
                subprocess.run(["systemctl", "--user", "start", u], capture_output=True)
                with contextlib.suppress(FileNotFoundError):
                    os.remove(pause_flag(u))
                self.note(f"started {u} again (the model left the graphics card)")

    def launch(self, want_card, card_only=False):
        e = self._launch(want_card, card_only)
        if not (e and e.where == "card") and not (self.engine and self.engine.where == "card" and self.engine.alive()):
            self.unpause()                       # the card isn't ours after all
        return e

    def _launch(self, want_card, card_only=False):
        """Start an engine for the current model (card if wanted and possible, else processor) and wait until
        it answers. card_only: give up instead of falling back to the processor. Returns the Engine, or None
        (self.last_fail says why)."""
        card = sysmem.dgpu()                     # one card for the checks, the plan, the command and the watchdog
        if want_card:
            self.why_off_card = self.a.allow_gpu and self.pausable_busy()
            if not self.why_off_card:
                if self.a.allow_gpu:
                    self.pause_others()
                self.why_off_card = gpu_blockers(self.a.allow_gpu, card)
        if card_only and self.why_off_card:
            self.last_fail = self.why_off_card
            return None
        p = self.make_plan(want_card and not self.why_off_card, card)
        if p["refuse"] or (card_only and not p["ngl"]):
            self.last_fail = p["refuse"] or "it doesn't fit on the graphics card now"
            if p["refuse"]:
                self.note(f"can't start {self.cur.name}: {p['refuse']}")
            return None
        if self.closing:
            return None
        n = Engine.count + 1
        out = os.path.join(self.out, f"engine-{n}-{self.cur.name}-{'card' if p['ngl'] else 'processor'}")
        try:
            os.makedirs(out, exist_ok=True)
            e = Engine(p, self.cur, out, card)
        except (Exception, SystemExit) as ex:   # e.g. llama.cpp's files were moved, the fence can't be made
            self.last_fail = f"it couldn't be started ({ex})"
            self.note(f"can't start {self.cur.name}: {self.last_fail}")
            return None
        self.note(f"starting {self.cur.name} on the {e.where}"
                  + (f" ({self.why_off_card})" if self.why_off_card and not p["ngl"] else "")
                  + f", context {p['ctx']}" + (f", {p['ngl']} layers on the card" if p["ngl"] else ""))
        if not e.wait_ready(abort=lambda: self.closing or not self.watched(e)) or self.closing or not self.watched(e):
            reason = e.wd.reason or ("lodger is stopping" if self.closing else "its watchdog stopped working"
                                     if not self.watched(e) else "it exited while loading (see engine.log)")
            if e.alive():
                e.stop()
            else:
                e.finish()
            self.last_fail = reason
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
            self.failed.pop(e.entry.name, None)
            self.failures, self.retry_delay = 0, 10
            self.cv.notify_all()
        return old

    def bring_up(self, want_card):
        """One try at starting an engine for the current model; True once it is up and answering."""
        e = self.launch(want_card)
        if e:
            self.install(e)
        else:
            self.failures += 1
        return bool(e)

    def retry_later(self):
        """No engine is up for the current model: the monitor tries again in a while (10 s, doubling to 5 min)."""
        self.retry_at = time.time() + self.retry_delay
        self.why = f"{self.cur.name} couldn't start ({self.last_fail}); trying again in {self.retry_delay} s"
        self.note(self.why)
        self.retry_delay = min(self.retry_delay * 2, 300)

    def start_or_give_up(self):
        """Load the model a request asked for (switch, wake-up): card if allowed, else (or if that fails) the
        processor. If neither starts, it waits FAIL_COOLDOWN before a request may try again."""
        target = self.cur
        if self.bring_up(want_card=self.a.allow_gpu) or (self.a.allow_gpu and not self.closing and self.bring_up(False)):
            return True
        with self.cv:
            why = self.last_fail or "see the engine logs"
            self.failed[target.name] = (time.time() + FAIL_COOLDOWN, why)
            self.asleep, self.cur = True, self.models[self.default] if not self.models[self.default].skip else target
            self.why = (f"{target.name} couldn't start ({why}); a request may try it again after "
                        f"{FAIL_COOLDOWN // 60} min")
            self.failures = 0
            self.cv.notify_all()
        self.note(self.why)
        return False

    def wait_engine(self, entry, limit=MOVE_WAIT, ping=None):
        """The engine for `entry`, once it's loaded and ready (asks for a switch or a wake-up if needed), counted
        as busy until release(); None if that doesn't happen within `limit` seconds or the model can't run.
        Requests are served in the order they came: when another model is asked for, the requests for this one
        that came first still get it before the switch, and the request that asked for the switch gets the new
        model before the next one (two apps using two models would otherwise keep switching, serving neither).
        ping(): called every PING seconds while waiting, without the lock (it may raise OSError: the app left)."""
        end = time.time() + limit
        token, next_ping = object(), time.monotonic() + PING
        with self.cv:
            self.ticket += 1
            ticket = self.ticket
            self.waiting[token] = (entry, ticket)
            try:
                while True:
                    self.last_used = time.time()
                    if entry.skip or self.closing:
                        return None
                    e = self.engine
                    if (self.cur is entry and self.ready and e and e.alive() and not e.stopping and not e.wd.reason
                            and (self.switch_to in (None, entry) or ticket < self.switch_ticket)):
                        self.busy[e] = self.busy.get(e, 0) + 1     # under the lock: no switch or sleep can slip in
                        return e
                    if self.cooling(entry):
                        return None
                    if self.cur is entry and not e and not self.asleep and self.failures >= 2:
                        return None                       # it keeps failing to start: say so now, not in 10 minutes
                    if self.cur is not entry and self.switch_to is None:
                        # first in line: the monitor switches once nothing is written and the requests that came
                        # before this one have their answers
                        self.switch_to, self.switch_ticket = entry, ticket
                    if self.asleep:
                        self.wake = True
                    left = end - time.time()
                    if left <= 0:
                        return None
                    if ping and time.monotonic() >= next_ping:
                        next_ping = time.monotonic() + PING
                        self.cv.release()
                        try:
                            ping()
                        finally:
                            self.cv.acquire()
                        continue
                    self.cv.wait(max(0.01, min(1.0, left, next_ping - time.monotonic() if ping else 1.0)))
            finally:
                del self.waiting[token]
                self.cv.notify_all()

    def may_switch(self):
        """(Under the lock.) May the monitor switch to the model asked for now? Not while an answer is being
        written, nor while requests for the loaded model that came before the switch request wait for it."""
        e = self.engine
        if self.switch_to is None or self.switch_to is self.cur or (e and self.busy.get(e)):
            return False
        if e and self.ready and e.alive():
            return not any(m is self.cur and t < self.switch_ticket for m, t in self.waiting.values())
        return True

    def release(self, e):
        with self.cv:
            self.busy[e] -= 1
            if not self.busy[e]:
                del self.busy[e]
            self.cv.notify_all()

    @contextlib.contextmanager
    def using(self, e):
        """An engine from wait_engine(), released at the end."""
        try:
            yield
        finally:
            self.release(e)

    def stepped_aside(self):
        self.last_aside = time.time()
        if self.card_since and time.time() - self.card_since < 600:
            self.quiet_needed = min(self.quiet_needed * 2, 1800)   # it came back too soon: wait longer next time
        elif self.card_since:
            self.quiet_needed = QUIET_SECONDS
        self.card_since = None

    def card_quiet(self, now=None):
        now = now or time.time()
        return now - self.quiet_since >= self.quiet_needed and now - self.last_aside >= self.quiet_needed

    def drop(self, e):
        with self.cv:
            if self.engine is e:
                self.engine, self.ready = None, False

    def switch(self):
        """Unload the current model and load the one a request is waiting for."""
        with self.cv:
            target, old = self.switch_to, self.engine
            if not self.may_switch():
                return                                # an answer started meanwhile: switch after it
            self.ready, self.engine, self.cur = False, None, target
            self.switch_to, self.asleep, self.wake = None, False, False
            self.why = f"switching to {target.name}"
        self.note(f"switching to {target.name}" + (f" (from {old.entry.name})" if old else ""))
        if old:
            old.stop()
            if old.where == "card":
                self.unpause()
        self.card_since, self.quiet_needed = None, QUIET_SECONDS
        self.start_or_give_up()

    def wake_up(self):
        with self.cv:
            self.wake = False
            if self.switch_to:
                self.cur, self.switch_to = self.switch_to, None
            self.asleep = False
        self.note(f"a request came in: loading {self.cur.name}")
        self.start_or_give_up()

    def monitor(self):
        """Switches models, steps aside / grows back, sleeps. Runs for the life of `lodger serve`; an error in
        one round is written down and the next round goes on (a dead monitor would strand every request)."""
        nb = Neighbors(ours=fence.our_pids)
        last_error = None
        while not self.closing:
            time.sleep(1)
            if self.closing:
                return
            try:
                self.tick(nb)
            except (Exception, SystemExit) as ex:
                # nothing here may end the monitor either (a note to a closed terminal, a full disk)
                with contextlib.suppress(Exception):
                    with open(os.path.join(self.out, "monitor-errors.log"), "a") as f:
                        f.write(time.strftime("%F %T\n"))
                        traceback.print_exc(file=f)
                if repr(ex) != last_error:
                    with contextlib.suppress(Exception):
                        self.note(f"lodger's monitor hit an error and carries on (see monitor-errors.log): {ex!r}")
                self.monitor_error = f"{time.strftime('%H:%M:%S')} {ex!r}"
                last_error = repr(ex)
                time.sleep(5)

    def tick(self, nb):
        """One round of the monitor."""
        self.clock()
        now = time.time()
        if self.reload_wanted:
            why, self.reload_wanted = self.reload_wanted, None
            self.reload(why)
        self.watch_models(now)
        if nb.update() or self.others_coming():
            self.quiet_since = now
        elif now - self.last_note >= 10:
            self.last_note = now
            self.note_desktop()
        e = self.engine
        # another model wanted: switch once nothing is being written with this one
        with self.cv:
            switching = self.may_switch()
        if switching:
            return self.switch()
        if self.asleep:
            if self.wake:
                self.wake_up()
            return
        if e and e.alive() and not self.watched(e):
            # nothing would stop the model if memory ran out: stop it now (planned stops tell the watchdog first)
            e.wd.reason = "lodger's watchdog stopped working, so the model was stopped to be safe"
            self.note(e.wd.reason)
            e.fence.stop()
        # unused for a while: unload, so the memory is yours again until the next request
        if self.a.idle and e and e.alive():
            with self.cv:
                unused = (self.active == 0 and not self.busy.get(e)
                          and time.time() - self.last_used >= self.a.idle * 60)
                if unused:
                    self.ready, self.asleep, self.engine = False, True, None
                    self.why = f"asleep: unloaded after {self.a.idle:g} min unused (loads again on the next request)"
            if unused:
                self.note(f"unused for {self.a.idle:g} min: unloading {e.entry.name} until the next request")
                e.stop()
                self.card_since = None
                self.unpause()
                return
        if e and not e.alive():
            reason = e.wd.reason
            e.finish()
            self.drop(e)
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
            if not self.bring_up(want_card=False) and not self.closing:   # straight back up on the processor
                self.retry_later()
            return
        if e and e.where == "card" and not self.force_off_card and now - self.last_power >= 10:
            self.last_power = now
            self.force_off_card = self.unplugged() or False
        if self.force_off_card:
            why = self.force_off_card if isinstance(self.force_off_card, str) else "asked to move off the card"
            self.force_off_card = False
            if e and e.where == "card":
                # asked to (POST /lodger/move) or unplugged: not a sign the card is contested, so the quiet wait
                # doesn't grow
                self.last_aside, self.card_since = time.time(), None
                self.note(f"moving off the card ({why})")
                self.why_off_card = why
                with self.cv:
                    self.ready = False
                e.stop()
                self.drop(e)
                self.unpause()
                if not self.bring_up(want_card=False) and not self.closing:
                    self.retry_later()
                return
        if not e:
            # no engine (it stepped aside and the processor didn't start, or the first start failed): try again
            if now >= self.retry_at and not self.closing:
                self.note(f"trying to start {self.cur.name} again")
                if not self.bring_up(want_card=self.a.allow_gpu and self.card_quiet(now)) and not self.closing:
                    self.retry_later()
            return
        # grow back: on the processor, card allowed, nothing being written, card quiet for a while
        if e.where == "processor" and self.a.allow_gpu and self.active == 0 and self.card_quiet(now):
            trial = self.make_plan(True)
            if trial["refuse"] or not trial["ngl"]:
                self.quiet_since = time.time()      # no room on the card yet; look again in a while
                return
            self.note(f"the card has been quiet for {now - self.quiet_since:.0f} s: growing back onto it")
            # the processor engine keeps answering while the card engine loads; then they swap
            new = self.launch(want_card=True, card_only=True)
            if not new:
                self.quiet_since = time.time()
                self.note("not growing back after all: " + (self.last_fail or "the card engine didn't come up"))
                return
            old = self.install(new)
            if old:
                # new answers go to the card now; one still being written on the processor may finish there
                # (a tool call cut by a move must be asked again), then the rest continues on the card
                end = time.time() + GROW_WAIT
                with self.cv:
                    while (time.time() < end and self.busy.get(old) and not self.closing and new.alive()
                           and not new.wd.reason and self.watched(new) and self.watched(old)):   # watch them here
                        self.cv.wait(1)
                if not new.alive() or new.wd.reason:
                    # the card engine had to stop before the hand-over was done: the processor one carries on
                    reason = new.wd.reason or "it stopped (see engine.log)"
                    self.stepped_aside()
                    self.why_off_card = "stepped aside: " + reason
                    self.install(old)
                    if new.alive():
                        new.stop()
                    else:
                        new.finish()
                    self.note(f"stepped aside from the card while growing back: {reason}")
                    self.unpause()
                    return
                old.stop()

    def note_desktop(self):
        """While no app needs the card: note how much of it everything but lodger's models uses (reserve.py).
        Any lodger's model is left out, ours and another's (lodger.service next to a test lodger), and nothing is
        noted while lodger memory is on the move (an engine loading or stopping)."""
        card = sysmem.dgpu()
        if not card or card.asleep():
            return
        e = self.engine
        if e and (e.stopping or not (e.alive() and e.ready_at)):
            return                                # loading or stopping: its memory changes faster than it's logged
        key = "gtt" if card.integrated else "vram"
        before = sysmem.lodger_use(card)[key]
        m = card.memory()
        after = sysmem.lodger_use(card)[key]
        if not m or abs(after - before) > MOVING:
            return                                # another lodger's model is loading or stopping
        ours = e.wd.now[key] if e and e.where == "card" else 0
        used = (m["gtt_used"] or 0) if card.integrated else m["used"]
        reserve.note(card.pci, max(0, used - max(ours, before, after)))

    def status(self):
        e = self.engine
        models = {}
        for n, m in self.models.items():
            why = m.skip or self.cooling(m)
            models[n] = ("not served: " if m.skip else "didn't start: ") + why if why else "ok"
        now = time.time()
        with self.cv:
            waiting = {}
            for m, _ in self.waiting.values():
                waiting[m.name] = waiting.get(m.name, 0) + 1
            writing = sum(self.busy.values())
        started = getattr(e, "started", None)
        return {"model": self.cur.name, "default": self.default, "ready": self.ready,
                "where": "asleep" if self.asleep else e.where if e else None, "why": self.why,
                "context": self.cur.ctx, "answers_in_progress": self.active, "answers_being_written": writing,
                "waiting": waiting, "switching_to": self.switch_to.name if self.switch_to else None,
                "off_card": self.why_off_card,
                "quiet_needed_s": self.quiet_needed, "card_quiet_for_s": round(now - self.quiet_since),
                "engine": {"model": e.entry.name, "where": e.where, "up_s": round(now - started) if started else None,
                           "answering": bool(e.ready_at), "stopping": e.wd.reason or (e.stopping or None),
                           "log": os.path.join(e.out, "engine.log")} if e else None,
                "retry_in_s": round(self.retry_at - now) if not e and self.retry_at > now else None,
                "last_fail": self.last_fail,
                "plan": {k: e.plan[k] for k in ("ngl", "n_cpu_moe", "ctx", "gpu_est")} if e else None,
                "models": models, "models_file": {"path": self.models_file, "read": time.strftime(
                    "%F %T", time.localtime(self.models_read)), "error": self.models_error},
                "monitor_error": self.monitor_error, "uptime_s": round(now - self.t_start),
                "ollama": self.ollama, "events": self.events}

    def loaded(self):
        """(model, graphics memory it uses, when it unloads or None) for the model loaded now, or None."""
        e = self.engine
        if not (e and self.ready and e.alive()):
            return None
        vram = (e.wd.peak["model_vram"] or e.plan["gpu_bytes"]) if e.where == "card" else 0
        return e.entry, vram, self.last_used + self.a.idle * 60 if self.a.idle else None


def model_info(e):
    """One model in GET /v1/models: OpenAI's fields, and Anthropic's (type, display_name, created_at) for its clients."""
    return {"id": e.name, "object": "model", "owned_by": "lodger", "context_window": e.ctx,
            "reasoning": e.reasoning, "meta": {"n_ctx": e.ctx}, "type": "model", "display_name": e.name,
            "created_at": ollama.now(mtime(getattr(e, "path", None)))}


def mtime(path):
    try:
        return os.path.getmtime(path)
    except (OSError, TypeError):
        return 0


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
    if key and path.split("?")[0] not in ("/health", "/v1/health"):
        scheme, _, given = (headers.get("Authorization") or "").strip().partition(" ")
        if scheme.lower() != "bearer":
            given = headers.get("X-Api-Key") or ""     # how Anthropic's clients send it (Claude Code's ANTHROPIC_API_KEY)
        if not (given.strip() and hmac.compare_digest(given.strip().encode(), key.encode())):
            return 401, ("lodger wants its API key: send the header 'Authorization: Bearer KEY' (or 'x-api-key: KEY') "
                         "(the key in lodger's --api-key-file or LODGER_API_KEY)")
    return None


CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")
MESSAGES_PATHS = ("/v1/messages",)            # Anthropic's API (Claude Code and other Anthropic clients)
RESPONSES_PATHS = ("/v1/responses", "/responses")   # OpenAI's Responses API (Codex and other newer OpenAI clients)
# llama-server's other answers, passed through as they are (a streamed one cut by a move ends with an error, a whole
# one is asked again of the next engine)
PASS_PATHS = ("/v1/completions", "/completions", "/completion", "/v1/embeddings", "/embeddings", "/embedding",
              "/infill", "/tokenize", "/detokenize", "/apply-template", "/v1/rerank", "/rerank", "/v1/reranking",
              "/reranking", "/v1/messages/count_tokens", "/v1/responses/input_tokens", "/responses/input_tokens",
              "/v1/chat/completions/input_tokens", "/chat/completions/input_tokens")
STREAM_PATHS = ("/v1/completions", "/completions", "/completion", "/infill")   # of those, the ones that can stream
ANTHROPIC_PATHS = ("/v1/messages", "/v1/messages/count_tokens")   # errors in Anthropic's shape
positive_int = answers.positive_int


OLLAMA_PATHS = ("/api/chat", "/api/generate", "/api/embed", "/api/embeddings", "/api/show")


def make_handler(lodger, key=None, ollama_api=False):
    """The request handler. ollama_api: the Ollama-style listener, which answers /api/... (ollama.py) as well."""
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 300                            # a connection that sends nothing for this long is closed
        cors = None                              # the local page's origin to allow in the answer, if any
        api = "openai"                           # the shape of this request's answers: openai, anthropic or ollama
        started = False                          # a streamed reply's headers went out
        head = False                             # HEAD: headers only
        client_gone = False
        oreq = False                             # this request is Ollama's kind: answered in its shape
        nd = None                                # an Ollama answer being streamed (ollama.Answer)
        conv = None                              # turns a whole answer into Ollama's shape

        def log_message(self, *args):
            pass

        def handle_expect_100(self):
            """Say "go ahead" only to a request that may reach lodger; the others get their refusal at once."""
            if check(self.command, self.path, self.headers, key):
                return True                      # do_POST refuses it (and closes: the body is never sent)
            return super().handle_expect_100()

        def send_json(self, code, obj, headers=()):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            if not self.head:
                self.wfile.write(body)

        def send_error_json(self, code, msg, close=False):
            if close:
                self.close_connection = True
            body = ({"error": msg} if self.api == "ollama" else answers.anthropic_error(code, msg)
                    if self.api == "anthropic" else answers.openai_error(code, msg))
            self.send_json(code, body, [("Connection", "close")] if close else ())

        def send_text(self, code, text):
            body = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not self.head:
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
            route = self.route()
            self.oreq = ollama_api and route.startswith("/api/")
            self.api = "ollama" if self.oreq else "anthropic" if route in ANTHROPIC_PATHS else "openai"
            why = check(self.command, self.path, self.headers, key)
            if not why:
                return False
            code, msg = why
            extra = [("Connection", "close")]   # its body (if any) is left unread
            self.close_connection = True
            if code == 401:
                extra.append(("WWW-Authenticate", 'Bearer realm="lodger"'))
            self.send_json(code, {"error": msg} if self.oreq else answers.anthropic_error(code, msg)
                           if self.api == "anthropic" else
                           {"error": {"message": msg, "type": "forbidden" if code == 403 else "unauthorized"}}, extra)
            return True

        def route(self):
            return urlsplit(self.path).path

        def do_OPTIONS(self):
            if self.refused():
                return
            self.send_response(204)
            self.send_header("Allow", "GET, HEAD, POST, OPTIONS")
            if self.cors:
                self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Authorization, Content-Type, X-Api-Key, Anthropic-Version, Anthropic-Beta")
                self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):
            self.head = True                     # this connection's next request may be a GET
            try:
                self.do_GET()
            finally:
                self.head = False

        def do_GET(self):
            if self.refused():
                return
            path = self.route()
            if path in ("/health", "/v1/health"):
                if lodger.asleep:                     # a request wakes it, so apps may go ahead
                    return self.send_json(200, {"status": "ok", "asleep": True})
                return self.send_json(200 if lodger.ready else 503, {"status": "ok" if lodger.ready else "loading"})
            if path in ("/v1/models", "/models"):
                data = [model_info(e) for e in lodger.available()]
                return self.send_json(200, {"object": "list", "data": data, "has_more": False,   # Anthropic's too
                                            "first_id": data[0]["id"] if data else None,
                                            "last_id": data[-1]["id"] if data else None})
            if path == "/lodger/status":
                return self.send_json(200, lodger.status())
            if ollama_api and path == "/":
                return self.send_text(200, "Ollama is running")     # what apps look for
            if self.oreq and path == "/api/version":
                return self.send_json(200, {"version": ollama.VERSION})
            if self.oreq and path == "/api/tags":
                return self.send_json(200, {"models": [ollama.listed(e) for e in lodger.available()]})
            if self.oreq and path == "/api/ps":
                now = lodger.loaded()
                return self.send_json(200, {"models": [ollama.running(*now)] if now else []})
            self.send_error_json(404, "not found")

        def body(self):
            """The request's body (bytes), or None after answering a broken one. Chunked bodies are read too."""
            te = (self.headers.get("Transfer-Encoding") or "").strip().lower()
            if te and te != "identity":
                if te != "chunked":
                    self.send_error_json(501, f"lodger can't read a {te!r} body", close=True)
                    return None
                parts, total = [], 0
                try:
                    while True:
                        size = int(self.rfile.readline(1024).split(b";")[0].strip(), 16)
                        if size < 0:
                            raise ValueError
                        if size == 0:
                            while self.rfile.readline(1024).strip():   # trailers
                                pass
                            return b"".join(parts)
                        total += size
                        if total > MAX_BODY:
                            self.send_error_json(413, f"the request is bigger than {MAX_BODY // MiB} MiB", close=True)
                            return None
                        parts.append(self.rfile.read(size))
                        self.rfile.readline(8)
                except (ValueError, OSError):
                    self.send_error_json(400, "the request's chunked body is broken", close=True)
                    return None
            n = (self.headers.get("Content-Length") or "0").strip()
            if not (n.isascii() and n.isdigit()):
                self.send_error_json(400, f"bad Content-Length {n!r}", close=True)
                return None
            if int(n) > MAX_BODY:
                self.send_error_json(413, f"the request is bigger than {MAX_BODY // MiB} MiB", close=True)
                return None
            raw = self.rfile.read(int(n)) if int(n) else b""
            if len(raw) < int(n):
                self.close_connection = True     # the app went away in the middle of sending it
                return None
            return raw

        def json_body(self, raw):
            """The request's JSON object, or None after answering a broken one."""
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self.send_error_json(400, "body is not JSON")
            if not isinstance(body, dict):
                return self.send_error_json(400, "body must be a JSON object")
            return body

        def no_model(self, name):
            names = ", ".join(e.name for e in lodger.available())
            self.send_error_json(404, f"lodger has no model called {name!r}; it has: {names}")

        def do_POST(self):
            if self.refused():
                return
            self.nd = self.conv = None
            self.started = False
            raw = self.body()
            if raw is None:
                return
            path = self.route()
            if path == "/lodger/reload":
                ok = lodger.reload("asked to")
                return self.send_json(200 if ok else 422, {"reloaded": ok, "error": lodger.models_error,
                                                           "models": [e.name for e in lodger.available()]})
            if path == "/lodger/move":
                on_card = bool(lodger.engine and lodger.engine.where == "card" and lodger.ready)
                lodger.force_off_card = on_card
                return self.send_json(200, {"moving": on_card, "where": lodger.engine.where if lodger.engine else None})
            if self.oreq:
                return self.ollama_post(raw, path)
            if path not in CHAT_PATHS + MESSAGES_PATHS + RESPONSES_PATHS + PASS_PATHS:
                return self.send_error_json(404, "not found")
            body = self.json_body(raw)
            if body is None:
                return
            name = body.get("model")
            if name is not None and not isinstance(name, str):
                return self.send_error_json(400, "'model' must be a string")
            if path in CHAT_PATHS + MESSAGES_PATHS and not isinstance(body.get("messages", []), list):
                return self.send_error_json(400, "'messages' must be a list")
            if path in RESPONSES_PATHS and not isinstance(body.get("input", ""), (str, list)):
                return self.send_error_json(400, "'input' must be text or a list")
            if body.get("stream") is not None and not isinstance(body["stream"], bool):
                return self.send_error_json(400, "'stream' must be true or false")
            entry = lodger.find(name)
            if not entry:
                return self.no_model(name)
            self.answer(body, entry, "/v1/chat/completions" if path in CHAT_PATHS else
                        "/v1/responses" if path in RESPONSES_PATHS else path)

        def answer(self, body, entry, path):
            """Answer `body` with `entry`'s engine at llama-server's `path`. Chat, Messages and Responses answers
            continue after a move (converse), whole ones too; the others are passed through."""
            with lodger.cv:
                lodger.active += 1
            try:
                stream = bool(body.get("stream"))
                kind = (answers.Chat if path == "/v1/chat/completions" else answers.Messages if path == "/v1/messages"
                        else answers.Responses if path == "/v1/responses" else None)
                if kind and not (kind is answers.Chat and not stream and answers.Chat.wanted_plain(body)):
                    self.converse(kind(body, stream), entry)
                elif not stream or path not in STREAM_PATHS + ("/v1/chat/completions",):
                    self.whole(body, entry, path)
                else:
                    self.relay(body, entry, path)
                if self.nd and stream and self.started and not self.client_gone:
                    self.send_lines(self.nd.close())   # llama-server's own kinds of stream end without [DONE]
            except OSError:                          # the app went away while lodger answered it
                self.close_connection = True
            finally:
                with lodger.cv:
                    lodger.active -= 1
                    lodger.last_used = time.time()

        def ollama_post(self, raw, path):
            """Ollama's POST /api/...: translated to llama-server's kind and answered as the others are."""
            if path.startswith(ollama.UNSUPPORTED):
                return self.send_error_json(501, f"{path} is not supported by lodger: it serves the GGUF files "
                                                 f"listed in {catalog.CONFIG} (add models there)")
            if path not in OLLAMA_PATHS:
                return self.send_error_json(404, "not found")
            body = self.json_body(raw)
            if body is None:
                return
            name = body.get("model") or body.get("name")    # /api/show: older apps send "name"
            if name is not None and not isinstance(name, str):
                return self.send_error_json(400, "'model' must be a string")
            entry = lodger.find(name) or lodger.find(ollama.plain(name))
            if not entry:
                return self.no_model(name)
            if path == "/api/show":
                return self.send_json(200, ollama.show(entry))
            try:
                target, req, ans = ollama.request(path, body, entry.name)
            except ValueError as ex:
                return self.send_error_json(400, str(ex))
            if req is None:                          # nothing to write: load the model (or, keep_alive 0, nothing)
                if body.get("keep_alive") in (0, "0", "0s", "0m"):
                    return self.send_json(200, ans.loaded(load=False))
                try:
                    e = lodger.wait_engine(entry, ping=self.check_gone)
                except OSError:
                    return
                if not e:
                    return self.send_error_json(503, self.unavailable(entry))
                lodger.release(e)
                return self.send_json(200, ans.loaded())
            self.conv, self.nd = ans.whole, ans if req.get("stream") else None
            self.answer(req, entry, target)

        def app_gone(self):
            """Has the app closed its connection? (Checked while it waits for an answer without streaming.)"""
            try:
                r, _, _ = select.select([self.connection], [], [], 0)
                gone = bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except (OSError, ValueError):
                gone = True
            if gone:
                self.client_gone = self.close_connection = True
            return gone

        def upstream(self, e, body, path, ping=None):
            """Send the request to the engine and wait until its answer starts, however long it takes to read the
            conversation (on the processor, a long one can take many minutes). Meanwhile ping() is called every
            PING seconds: it pings a streaming app, or checks that an app waiting for a whole answer is still there
            (an app that went away ends the wait, and closing the connection stops the engine working on it).
            Raises OSError if the engine or the app goes away."""
            c = http.client.HTTPConnection(HOST, e.port, timeout=10)
            try:
                c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", **e.auth()})
                while not select.select([c.sock], [], [], PING)[0]:
                    if not e.alive():
                        raise OSError("the engine went away")
                    if ping:
                        ping()
                c.sock.settimeout(SILENCE)      # once it answers, llama-server sends something every 30 s
                return c, c.getresponse()
            except BaseException:
                c.close()
                raise

        def running(self, e):
            """Is `e` still the engine, running and not being stopped?"""
            return lodger.engine is e and e.alive() and not self.stopping(e)

        def moved(self, e):
            """After an error talking to engine `e`: was it stopped or replaced (a move: ask again on the next one),
            rather than failing while it runs?"""
            t0 = time.time()
            while time.time() - t0 < 5 and lodger.engine is e and e.alive() and not self.stopping(e):
                time.sleep(0.2)
            return lodger.engine is not e or not e.alive() or self.stopping(e)

        @staticmethod
        def stopping(e):
            return getattr(e, "stopping", False) or bool(getattr(getattr(e, "wd", None), "reason", None))

        def unavailable(self, entry):
            return f"{entry.name} is not available right now: " + (entry.skip or lodger.cooling(entry) or lodger.why)

        def whole(self, body, entry, path):
            """A whole answer passed through as llama-server gives it (asked again of the next engine after a
            move: nothing of it was sent yet)."""
            for _ in range(4):
                e = lodger.wait_engine(entry, ping=self.check_gone)
                if not e:
                    return self.send_error_json(503, self.unavailable(entry))
                c = None
                try:
                    with lodger.using(e):
                        c, r = self.upstream(e, body, path, ping=self.check_gone)
                        data = r.read()
                except (OSError, http.client.HTTPException) as ex:
                    if self.client_gone:
                        return
                    if not self.moved(e):
                        return self.send_error_json(502, f"the model's engine stopped answering ({ex})")
                    time.sleep(1)                   # the engine moved underneath us: ask again on the new one
                    continue
                finally:
                    if c:
                        c.close()
                if r.status != 200 and not self.running(e):
                    time.sleep(1)                   # an error from an engine being stopped: ask the next one
                    continue
                if self.conv:
                    return self.send_json(*self.conv(r.status, data))
                if r.status != 200 and self.api == "anthropic":
                    return self.send_json(r.status, answers.anthropic_error(r.status, answers.error_message(data)))
                self.send_response(r.status)
                self.send_header("Content-Type", r.getheader("Content-Type") or "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error_json(503, "the model kept moving; try again")

        def converse(self, ans, entry):
            """An answer that continues after a move (answers.py: chat, Messages, Responses). The engine always
            streams; a streaming app gets the events as they come (the reply's headers go out with the first one,
            so an error before that keeps its HTTP status), a whole answer is sent once it's complete."""
            ping = (lambda: self.event(":")) if ans.stream else self.check_gone
            for attempt in range(6):
                e = lodger.wait_engine(entry, ping=ping)
                if not e:
                    return self.fail(ans, 503, self.unavailable(entry))
                c = None
                try:
                    with lodger.using(e):
                        req = ans.request(attempt)
                        if req is None:             # the length limit was reached just before the move
                            return self.complete(ans, entry, "length", e)
                        if attempt and ans.stream:
                            self.event(ans.moved_note(e.where))
                        c, r = self.upstream(e, req, ans.path, ping=ping)
                        if r.status != 200:
                            err = r.read()
                            if self.running(e):
                                return self.fail(ans, r.status, err)
                            raise OSError("the engine went away")
                        if not (r.getheader("Content-Type") or "").startswith("text/event-stream"):
                            data = r.read()                 # not a stream after all: passed on whole, as it is
                            if ans.stream:
                                return self.fail(ans, 502, "the model's engine didn't stream its answer")
                            return self.send_json(*self.conv(200, data)) if self.conv else self.send_raw(200, data)
                        look = time.monotonic() + 1
                        for ev in answers.events(r.readline):
                            for s in ans.feed(ev):
                                self.event(s)
                            if ans.done:
                                return self.complete(ans, entry)
                            if not ans.stream and time.monotonic() > look:
                                look = time.monotonic() + 1
                                self.check_gone()
                        raise OSError("stream ended early")
                except (OSError, http.client.HTTPException):
                    if self.client_gone:
                        return
                    if ans.finished:                # only the end was cut off
                        return self.complete(ans, entry)
                    if not self.moved(e):
                        return self.fail(ans, 502, "the model's engine stopped answering; please retry")
                    why = ans.cut()
                    if why and ans.stream:
                        return self.fail(ans, 503, why)
                    if why:
                        ans.reset()                 # the app has seen none of it: start over
                finally:
                    if c:
                        c.close()                   # stops the engine writing if the app went away
            return self.fail(ans, 503, "the model kept moving; answer cut short")

        def complete(self, ans, entry, why=None, e=None):
            """The answer is complete (or reached its length limit): end the stream, or send the whole answer.
            `e`: an engine to count the request's tokens with, if the answer's own end never said."""
            path = e and ans.counting()
            if path:
                c = None
                with contextlib.suppress(Exception):
                    c, r = self.upstream(e, ans.body, path)
                    n = json.loads(r.read()).get("input_tokens") if r.status == 200 else None
                    ans.input_tokens = n if isinstance(n, int) and not isinstance(n, bool) else None
                if c:
                    c.close()
            for s in ans.close(entry.name, why):
                self.event(s)
            if ans.stream:
                return
            status, obj = ans.failure or (200, ans.whole(entry.name))
            if self.conv:
                return self.send_json(*self.conv(status, json.dumps(obj).encode()))
            self.send_json(status, obj)

        def fail(self, ans, status, err):
            """End the answer with an error: in the stream if one was started, else as the HTTP reply (an
            engine's own error answer for OpenAI's shapes as it is). `err`: lodger's words, or the engine's reply."""
            msg = answers.error_message(err) if isinstance(err, bytes) else err
            if self.started:
                with contextlib.suppress(OSError):
                    for s in ans.stream_error(msg):
                        self.event(s)
                return
            if self.oreq or self.api == "anthropic" or not isinstance(err, bytes):
                return self.send_error_json(status, msg)
            self.send_raw(status, err)

        def send_raw(self, status, data):
            """An engine's JSON reply, as it is."""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if not self.head:
                self.wfile.write(data)

        def plain_fail(self, status, err):
            """fail() for a passed-through stream (llama-server's own kinds)."""
            msg = answers.error_message(err) if isinstance(err, bytes) else err
            if self.started:
                with contextlib.suppress(OSError):
                    self.event("data: " + answers.compact({"error": {"message": msg, "code": status}}))
                return
            if self.oreq or not isinstance(err, bytes):
                return self.send_error_json(status, msg)
            self.send_raw(status, err)

        def start_stream(self):
            self.started = True
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson" if self.nd else "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def check_gone(self):
            """The ping for an app waiting for a whole answer: one that went away ends the wait (OSError)."""
            if self.app_gone():
                raise OSError("the app went away")

        def relay(self, body, entry, path):
            """A streaming answer of the other kinds (e.g. /v1/completions): passed through; a move ends it."""
            ping = lambda: self.event(":")  # noqa: E731
            e = lodger.wait_engine(entry, ping=ping)
            if not e:
                return self.plain_fail(503, self.unavailable(entry))
            c = None
            try:
                with lodger.using(e):
                    c, r = self.upstream(e, body, path, ping=ping)
                    if r.status != 200:
                        return self.plain_fail(r.status, r.read())
                    for ev in answers.events(r.readline):
                        self.event(ev.text())
                        if ev.data == "[DONE]":
                            return
                    if not self.running(e):
                        raise OSError("the engine went away")
            except (OSError, http.client.HTTPException):
                if not self.client_gone:
                    self.plain_fail(503, "the model moved while writing; please retry")
            finally:
                if c:
                    c.close()

        def event(self, s):
            """One server-sent event to the app (an Ollama app gets what it turns into instead); the first one
            sends the reply's headers."""
            if not self.started:
                self.start_stream()
            if self.nd:
                return self.send_lines(self.nd.feed(s))
            self.write((s + "\n\n").encode())

        def send_lines(self, objs):
            if objs:
                self.write(b"".join(json.dumps(o, ensure_ascii=False).encode() + b"\n" for o in objs))

        def write(self, data):
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except OSError:
                self.client_gone = True
                raise

    return H


def ctx_arg(v):
    n = int(v)
    if n < planner.SMALLEST_CTX:
        raise argparse.ArgumentTypeError(f"at least {planner.SMALLEST_CTX}")
    return n


def serve(argv):
    ap = argparse.ArgumentParser(prog="lodger serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", metavar="MODEL.gguf",
                    help=f"serve only these files (default: the models in {catalog.CONFIG})")
    ap.add_argument("--default", help="model to load when a request doesn't name one")
    ap.add_argument("--allow-gpu", action="store_true", help="the graphics card may be used")
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--min-ctx", type=ctx_arg, default=planner.MIN_CTX)
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
    ap.add_argument("--ollama", action="store_true",
                    help="also answer Ollama's API, for apps that only speak it, at http://127.0.0.1:11434 "
                         "(`ollama = true` in config.toml does the same); not if that port is taken, by Ollama say")
    ap.add_argument("--ollama-port", type=int, default=11434, metavar="PORT")
    a = ap.parse_args(argv)
    a.api_key = read_key(a.api_key_file)
    missing = [p for p in a.files if not os.path.exists(os.path.expanduser(p))]
    if missing:
        sys.exit(f"no such file: {', '.join(missing)} (lodger serve takes model files; to start with one of the "
                 f"models in {catalog.CONFIG}, name it with --default NAME)")
    models, default = catalog.load(a.files or None)
    if not models:
        sys.exit(f"no models found: add folders or files to {catalog.CONFIG}, or download one: `lodger models`")
    default = a.default or (default if default in models else None) or next(iter(models))
    lodger = Lodger(a, models, default)
    for line in fence.reap_orphans():            # models a crashed lodger left running: they hold memory, unwatched
        log(f"cleaned up after an earlier lodger: {line}")
    unpause_all()                                # left over from a crash
    httpd = ThreadingHTTPServer((HOST, a.port), make_handler(lodger, a.api_key))
    httpd.daemon_threads = True
    ohttpd = ollama_listener(lodger, a) if a.ollama or config.get().get("ollama") else None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    signal.signal(signal.SIGINT, signal.default_int_handler)   # also when started in the background of a script
    signal.signal(signal.SIGHUP, on_hangup(lodger))
    log(f"lodger serving {len(models)} models at http://{HOST}:{a.port}/v1 (only this PC"
        + (", API key required" if a.api_key else "") + f"); default {default}; logs in {lodger.out}")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monitor = threading.Thread(target=lodger.monitor, daemon=True)
    try:
        lodger.decide_ctx(lodger.cur)
        if lodger.asleep:
            lodger.why = "asleep until the first request"
            log("waiting for the first request before loading a model")
        elif not lodger.bring_up(want_card=a.allow_gpu):
            lodger.retry_later()                 # the monitor tries again
        threading.Thread(target=lodger.prepare, daemon=True).start()
        monitor.start()
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log("stopping")
    finally:
        # a second Ctrl-C or TERM must not cut this short: the models run outside lodger and must be stopped
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        lodger.closing = True
        lodger.why = "lodger is stopping"            # what a request still waiting is told
        with lodger.cv:
            lodger.cv.notify_all()
        httpd.shutdown()
        if ohttpd:
            ohttpd.shutdown()
        stop_all()
        if monitor.is_alive():
            monitor.join(20)                     # it may be starting one right now
            stop_all()
        lodger.unpause()
    return 0


def on_hangup(lodger, tty=None):
    """SIGHUP. Run as a service (`systemctl --user reload lodger`, `kill -HUP`): read the model list again (the
    monitor does it). Run in a terminal, SIGHUP means the terminal closed: stop, with the models, as Ctrl-C does."""
    if tty is None:
        try:
            tty = os.isatty(sys.stdout.fileno())
        except (OSError, ValueError, AttributeError):
            tty = False

    def handler(sig, frame):
        if tty:
            raise KeyboardInterrupt
        lodger.reload_wanted = "SIGHUP"
    return handler


def port_taken(port):
    """Does a program answer on this port of this PC (IPv4 or IPv6)?"""
    for family, addr in ((socket.AF_INET, HOST), (socket.AF_INET6, "::1")):
        with contextlib.suppress(OSError), socket.socket(family) as s:
            s.settimeout(1)
            if s.connect_ex((addr, port)) == 0:
                return True
    return False


def ollama_listener(lodger, a):
    """Ollama's API (ollama.py) on HOST:a.ollama_port, with the same checks as the main port; None if the port
    is taken (by Ollama itself, say: its apps then keep talking to it)."""
    try:
        if port_taken(a.ollama_port):
            raise OSError("another program answers there (Ollama itself?)")
        h = ThreadingHTTPServer((HOST, a.ollama_port), make_handler(lodger, a.api_key, ollama_api=True))
    except OSError as ex:
        lodger.ollama = f"not started: port {a.ollama_port} is taken ({ex.strerror or ex})"
        log(f"Ollama's API {lodger.ollama}")
        return None
    h.daemon_threads = True
    threading.Thread(target=h.serve_forever, daemon=True).start()
    lodger.ollama = f"http://{HOST}:{a.ollama_port}"
    log(f"Ollama's API at {lodger.ollama} too")
    return h


def stop_all():
    for e in list(Engine.all):                   # also one still loading for a move
        try:
            e.stop()
        except Exception as ex:                  # stop the others anyway
            log(f"couldn't stop {e.name}: {ex!r}")


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


def read_json(path):
    """The JSON object in `path`: {} if there is no file, None if it isn't a JSON object (then it's left alone)."""
    try:
        with open(path) as f:
            obj = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def write_json(path, obj, mode):
    """Replace `path` (or, if it's a link, the file it points to) with `obj` in one step, with these permissions."""
    real = os.path.realpath(path)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(real), prefix=".lodger-", suffix=".tmp")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, real)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise


def write_pi_config(port, models, default, api_key="lodger"):
    """Tell pi about lodger's models: a "lodger" provider in pi's models.json (other providers are kept),
    and lodger's default model as pi's default unless you chose another provider or lodger model in pi's settings.
    `api_key`: see pi_key(). A file lodger can't read as JSON is left as it is (it may be half-edited), and
    False returned."""
    path = os.path.join(PI_DIR, "models.json")
    cfg = read_json(path)
    if cfg is None or not isinstance(cfg.get("providers", {}), dict):
        print(f"lodger: not updating {path}: it isn't the JSON pi expects (fix it, then run lodger pi)", file=sys.stderr)
        return False
    cfg.setdefault("providers", {})["lodger"] = {
        "baseUrl": f"http://{HOST}:{port}/v1", "apiKey": api_key, "api": "openai-completions",
        "models": [{"id": m["id"], "name": f"{m['id']} (local)", "reasoning": bool(m.get("reasoning")),
                    "contextWindow": m["context_window"], "maxTokens": min(32768, m["context_window"] // 4),
                    "input": ["text"], "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "compat": {"supportsDeveloperRole": False}} for m in models if m.get("context_window")]}
    os.makedirs(PI_DIR, exist_ok=True)
    write_json(path, cfg, 0o600)              # only you may read it (it can hold API keys, lodger's and others')
    spath = os.path.join(PI_DIR, "settings.json")
    settings = read_json(spath)
    if settings is None:
        print(f"lodger: not updating {spath}: it isn't JSON lodger can read", file=sys.stderr)
    elif (settings.get("defaultProvider") is None or settings.get("defaultProvider") == "lodger"
          and settings.get("defaultModel") not in [m["id"] for m in models]):   # keep a lodger model you picked
        settings.update(defaultProvider="lodger", defaultModel=default)
        try:
            mode = stat.S_IMODE(os.stat(spath).st_mode)
        except OSError:
            mode = 0o644
        write_json(spath, settings, mode)
    return True


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
