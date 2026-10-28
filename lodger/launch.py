"""lodger plan / lodger run: run a model on what is free right now, fenced in and watched.

  lodger plan MODEL [--ctx N] [--cpu-only] [--leave-gpu MiB] [--allow-gpu] [--pretend-vram-free MiB] [--pretend-ram-avail MiB]
  lodger run  MODEL [same options] [--prompt TEXT] [-n 256] [--runtime SEC]

plan: say what would go where, and why, without starting anything.
run:  start our llama.cpp (llama-cli) with that plan in its own cgroup (RAM cap, no swap, no crash
      dumps, low CPU/disk priority) and watch it several times a second. It steps aside (stops) at once
      if free VRAM or RAM gets too low, or if another app needs the graphics card: keeps it busy or
      takes a lot of its memory. Without --prompt it opens an interactive chat in this terminal.
The graphics card is only used when you allow it: --allow-gpu (or answer the question on a terminal).
Otherwise the model runs on the processor only."""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

from . import config, engine, fence, plan as planner, reserve, sysmem
from .neighbors import Neighbors

MiB, GiB = sysmem.MiB, sysmem.GiB
RUNS = config.runs_dir()
# watchdog: stop the model if RAM available falls under this (graphics memory: reserve.kill_line).
# LODGER_KILL_*_MIB can only raise these (used to test the stop path), never lower them.
KILL_RAM_AVAIL = max(1536, int(os.environ.get("LODGER_KILL_RAM_AVAIL_MIB", 0))) * MiB
LOAD_SECONDS = 90          # watch faster while the model loads (its biggest allocations happen then)
# The author's own setup (extras.llama_router in config.toml): ~/.local/bin/llama-game-guard leaves this while it
# has llama-router stopped for a game; the router comes back after
GAME_GUARD_FLAG = os.path.join(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"),
                               "llama-game-guard.router-stopped")


def router_active():
    return subprocess.run(["systemctl", "--user", "is-active", "-q", "llama-router"]).returncode == 0


def router_coming():
    if not config.extra("llama_router"):
        return None
    if router_active():
        return ("llama-router is running and can load a big model at any moment; "
                "stop it first (systemctl --user stop llama-router) and start it again afterwards")
    if os.path.exists(GAME_GUARD_FLAG):
        return "llama-game-guard stopped llama-router for a game and will start it again when the game ends"
    return None


def settle(card, seconds=2.0, limit=30.0):
    """Before taking memory on the card: is another app busy on it, or still loading (free memory
    falling)? Returns a reason the card is taken, or None once it has been calm for `seconds`."""
    nb = Neighbors(card, ours=fence.our_pids)
    t0 = time.time()
    while time.time() - t0 < limit:
        start = sysmem.vram_free(card)
        busy = nb.watch(seconds)
        if busy:
            return busy
        if start - sysmem.vram_free(card) < 256 * MiB:
            return None
        print("(memory on the graphics card is still changing: waiting for it to settle)", flush=True)
    return "memory on the graphics card kept changing for 30 s (another app is loading)"


def gpu_blockers(allowed, card):
    """Reasons the graphics card must not be used right now (None = fine)."""
    if not allowed:
        return "the graphics card is only used when you allow it (pass --allow-gpu)"
    return router_coming() or settle(card)


def fmt_cmd(cmd, env):
    e = " ".join(f"{k}={v}" for k, v in env.items())
    return (e + " " if e else "") + " ".join(shlex.quote(c) for c in cmd)


def llama_cmd(p, a):
    cmd = [engine.tool("llama-cli"), "-m", p["model"], "-c", str(p["ctx"]), "-t", str(p["threads"]),
           "--fit", "off"]
    # only the chosen card is visible to llama.cpp, so it's always device 0 (see engine.py)
    dargs, env = engine.device_args(p["ngl"], sysmem.dgpu(), p["n_cpu_moe"])
    cmd += dargs
    if not a.think:
        cmd += ["--reasoning", "off"]
    if a.prompt is not None:
        cmd += ["-st", "-n", str(a.n), "-p", a.prompt]
    return cmd, env


def show(p, cmd=None, env=None):
    print("\n".join("- " + w for w in p["why"]))
    if p["refuse"]:
        print(f"\nREFUSED: {p['refuse']}")
        return
    where = (f"graphics card: {p['ngl_layers']} of {p['layers']} layers"
             + (f", experts of {p['layers'] - p['n_cpu_moe']} layers" if p["moe"] and p["ngl"] else "")
             + f", ~{p['gpu_est'] / GiB:.1f} GiB (planned with margin {p['gpu_bytes'] / GiB:.1f} GiB)") if p["ngl"] else "processor only"
    print(f"\nPlan: {where}; context {p['ctx']} words; RAM fence {p['ram_cap'] / GiB:.1f} GiB.")
    if cmd:
        print("Command: " + fmt_cmd(cmd, env))


def unit_props(p, runtime=None):
    """The fence around the model (as systemd unit properties; see fence.py for PCs without systemd)."""
    props = {
        "MemoryMax": str(p["ram_cap"]), "MemorySwapMax": "0",
        "LimitCORE": "0",                       # no crash dump: a dumping process keeps its VRAM for seconds
        "CPUWeight": "20", "IOWeight": "20",    # your apps get the processor and disks first
        "Nice": "10",
    }
    if not p["ngl"]:
        props["PrivateDevices"] = "yes"         # processor only: the graphics card's device files don't exist for it
    if runtime:
        props["RuntimeMaxSec"] = str(runtime)
    return props


def unit_cgroup(unit):
    """The cgroup of a systemd user service (for --pause services)."""
    r = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", unit + ".service"],
                       capture_output=True, text=True)
    cg = r.stdout.strip()
    return "/sys/fs/cgroup" + cg if cg else None


class Watchdog(threading.Thread):
    def __init__(self, fenced, on_gpu, out, log=None, keep=None):
        super().__init__(daemon=True)
        self.kill_vram = reserve.kill_line(keep if keep is not None else reserve.floor(0))
        self.fenced, self.on_gpu, self.out = fenced, on_gpu, out
        self.log, self.device_seen = log, False   # the engine's log: is it on the card lodger chose?
        self.stop_flag = threading.Event()
        self.reason = None
        self.peak = {"model_vram": 0, "model_gtt": 0, "cgroup_ram": 0}
        self.now = {"vram": 0, "gtt": 0}
        self.low = {"vram_free": None, "ram_avail": None}
        self.card = sysmem.dgpu()
        self.neighbors = Neighbors(self.card, ours=fence.our_pids) if on_gpu else None

    def kill(self, why, graceful=False):
        if self.reason:
            return
        self.reason = why
        print(f"\n[lodger] STOPPING THE MODEL: {why}", file=sys.stderr, flush=True)
        if graceful:
            self.fenced.kill(signal.SIGTERM)
            t0 = time.time()
            while time.time() - t0 < 1.5 and not self.fenced.gone():
                time.sleep(0.1)
        self.fenced.kill(signal.SIGKILL)

    def check_device(self):
        """Stop the engine if llama.cpp says it uses another card than the one lodger chose. True if stopped."""
        try:
            with open(self.log, errors="replace") as f:
                text = f.read(1 << 20)
        except OSError:
            return False
        if self.card and engine.USING.search(text):
            self.device_seen = True
            why = engine.check_log(text, self.card)
            if why:
                self.kill(why)
                return True
        return False

    def run(self):
        t0 = time.time()
        last_nb = last_log = 0.0
        with open(os.path.join(self.out, "watch.tsv"), "w") as f:
            f.write("t\tvram_free_MiB\tram_avail_MiB\tmodel_vram_MiB\tmodel_gtt_MiB\tmodel_ram_MiB\n")
            while not self.stop_flag.is_set():
                now = time.time()
                loading = now - t0 < LOAD_SECONDS
                vf = sysmem.vram_free(self.card)                       # cheap: checked every tick
                ra = sysmem.meminfo()["MemAvailable"]
                self.low["vram_free"] = vf if self.low["vram_free"] is None else min(self.low["vram_free"], vf)
                self.low["ram_avail"] = ra if self.low["ram_avail"] is None else min(self.low["ram_avail"], ra)
                if self.on_gpu and self.card and vf < self.kill_vram:   # stopping a processor-only model frees no VRAM
                    self.kill(f"only {vf // MiB} MiB of graphics memory left (limit {self.kill_vram // MiB})")
                elif ra < KILL_RAM_AVAIL:
                    self.kill(f"only {ra // MiB} MiB of RAM left (limit {KILL_RAM_AVAIL // MiB})")
                elif self.fenced.over_cap():                           # no kernel fence on this PC: we are the fence
                    self.kill(f"the model went over its RAM cap ({self.fenced.cap // MiB} MiB)")
                elif self.on_gpu and self.log and not self.device_seen and self.check_device():
                    pass
                elif self.on_gpu and now - last_nb >= 0.25:
                    last_nb = now
                    why = self.neighbors.update() or router_coming()
                    if why:
                        self.kill(why + "; giving the graphics card back", graceful=True)
                if now - last_log >= 0.5:                              # model's own use, logged twice a second
                    last_log = now
                    mv = mg = 0
                    ram = None
                    pids = self.fenced.pids()
                    if pids:
                        use = sysmem.gpu_use(self.card, pids)
                        mv, mg = use["vram"], use["gtt"]
                        ram, peak = self.fenced.memory()
                        self.peak["cgroup_ram"] = max(self.peak["cgroup_ram"], peak or ram or 0)
                    self.now = {"vram": mv, "gtt": mg}
                    self.peak["model_vram"] = max(self.peak["model_vram"], mv)
                    self.peak["model_gtt"] = max(self.peak["model_gtt"], mg)
                    f.write(f"{now - t0:.1f}\t{vf // MiB}\t{ra // MiB}\t{mv // MiB}\t{mg // MiB}\t{(ram or 0) // MiB}\n")
                    f.flush()
                self.stop_flag.wait(0.1 if loading else 0.25)


def build_plan(a, for_run):
    pretend = a.pretend_vram_free is not None or a.pretend_ram_avail is not None
    if for_run and pretend:
        sys.exit("--pretend-* is for 'plan' only: a real run always uses the real free memory.")
    why_off = None
    card = sysmem.dgpu()
    if not a.cpu_only and not pretend:
        allowed = a.allow_gpu
        if for_run and not allowed and sys.stdin.isatty():
            allowed = input("Use the graphics card? [y/N] ").strip().lower() == "y"
        if for_run or allowed:
            why_off = gpu_blockers(allowed, card)
        else:
            b = router_coming()
            print("(note: this plan assumes the graphics card is allowed; a real run needs --allow-gpu"
                  + (f", and now: {b}" if b else "") + ")\n")
    return planner.make(
        a.model, target_ctx=a.ctx, min_ctx=a.min_ctx, cpu_only=a.cpu_only, leave_gpu=a.leave_gpu * MiB,
        vram_free=None if a.pretend_vram_free is None else a.pretend_vram_free * MiB,
        ram_avail=None if a.pretend_ram_avail is None else a.pretend_ram_avail * MiB,
        threads=a.threads, why_gpu_off=why_off, device_check=bool(a.allow_gpu and not why_off and not pretend))


def do_run(a):
    p = build_plan(a, True)
    cmd, env = llama_cmd(p, a) if not p["refuse"] else (None, None)
    show(p, cmd, env)
    if p["refuse"]:
        return 2
    if p["ngl"]:
        # last look right before starting: if another app took memory while we planned, plan again on the processor
        drop = p["vram_free"] - sysmem.vram_free()
        if drop > 256 * MiB:
            print(f"\n{drop // MiB} MiB of graphics memory was taken while planning: running on the processor instead.")
            a.cpu_only = True
            p = build_plan(a, True)
            cmd, env = llama_cmd(p, a) if not p["refuse"] else (None, None)
            show(p, cmd, env)
            if p["refuse"]:
                return 2
    name = re.sub(r"[^A-Za-z0-9.]+", "-", p["name"] or "model")[:40]
    out = os.path.join(RUNS, time.strftime("%Y-%m-%d-%H%M%S") + "-run-" + name)
    os.makedirs(out)
    props = unit_props(p, a.runtime)
    interactive = a.prompt is None
    kind, why = fence.mode()
    if kind != "systemd":
        print(f"\n(fence: {why})")
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump({**p, "cmd": cmd, "env": env, "cgroup": props, "fence": kind}, f, indent=1)

    t0 = time.time()
    if interactive:
        o = e = None
        fenced = fence.Fenced(cmd, env, props, stdin=None, pty=True, name=f"lodger-{os.getpid()}")
    else:
        o, e = open(os.path.join(out, "output.txt"), "w"), open(os.path.join(out, "log.txt"), "w")
        fenced = fence.Fenced(cmd, env, props, stdout=o, stderr=e, name=f"lodger-{os.getpid()}")
    wd = Watchdog(fenced, bool(p["ngl"]), out, log=None if interactive else os.path.join(out, "log.txt"), keep=p["keep"])
    wd.start()
    signal.signal(signal.SIGINT, lambda *_: wd.kill("you pressed Ctrl+C", graceful=True))
    rc = fenced.wait()
    secs = time.time() - t0
    wd.stop_flag.set()
    wd.join()
    fenced.stop() if fenced.pids() else fenced.close()
    for fh in (o, e):
        if fh:
            fh.close()

    speed, text = None, ""
    if not interactive:
        text = open(os.path.join(out, "output.txt"), errors="ignore").read()
        log = open(os.path.join(out, "log.txt"), errors="ignore").read()
        speed = [l.strip() for l in (text + log).splitlines() if "t/s" in l][-3:]
        answer = text[text.find("\n> ") + 1:] if "\n> " in text else text
        print("\n--- answer ---\n" + "\n".join(l for l in answer.strip().splitlines() if "t/s" not in l and l != "Exiting...").strip())
        if speed:
            print("--- speed ---\n" + "\n".join(speed))
    res = {
        "exit": rc, "seconds": round(secs, 1), "stopped_by_watchdog": wd.reason,
        "model_vram_peak": wd.peak["model_vram"], "model_gtt_peak": wd.peak["model_gtt"],
        "model_ram_peak": wd.peak["cgroup_ram"],
        "vram_free_lowest": wd.low["vram_free"], "ram_avail_lowest": wd.low["ram_avail"], "speed": speed,
    }
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    # learn: what we planned vs what it really used (a run stopped while loading hasn't shown its full use)
    loaded = not wd.reason or secs > LOAD_SECONDS or (not interactive and "\n> " in text)
    with open(os.path.join(RUNS, "calibration.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.strftime("%F %T"), "model": p["name"], "ngl": p["ngl"],
                            "n_cpu_moe": p["n_cpu_moe"], "ctx": p["ctx"],
                            "gpu_est": p["gpu_est"], "llama_gpu_est": p.get("llama_gpu_est"),
                            "gpu_real": wd.peak["model_vram"] if loaded else None,
                            "ram_anon_est": p["ram_anon_est"], "ram_cap": p["ram_cap"],
                            "ram_real_peak": wd.peak["cgroup_ram"], "exit": rc, "stopped": wd.reason,
                            "run": os.path.basename(out)}) + "\n")
    lo_v = wd.low["vram_free"]
    print(f"\nexit {rc} after {secs:.0f} s"
          + (f"; STOPPED: {wd.reason}" if wd.reason else "")
          + f". Model used at most {wd.peak['model_vram'] // MiB} MiB VRAM (planned {p['gpu_est'] // MiB}),"
          f" {wd.peak['cgroup_ram'] // MiB} MiB RAM incl. cached weights (fence {p['ram_cap'] // MiB})."
          + (f" Lowest free VRAM {lo_v // MiB} MiB," if lo_v is not None else "")
          + f" lowest free RAM {(wd.low['ram_avail'] or 0) // MiB} MiB.\nLogs: {out}")
    return 0 if rc == 0 and not wd.reason else 1


def main(argv):
    ap = argparse.ArgumentParser(prog="lodger", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "run"])
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, help="context length to aim for (words/tokens; default: automatic)")
    ap.add_argument("--min-ctx", type=int, default=planner.MIN_CTX)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true", help="the graphics card may be used")
    ap.add_argument("--leave-gpu", type=int, default=0, metavar="MiB", help="extra graphics memory to leave free")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--pretend-vram-free", type=int, metavar="MiB", help="plan only: pretend this much VRAM is free")
    ap.add_argument("--pretend-ram-avail", type=int, metavar="MiB", help="plan only: pretend this much RAM is available")
    ap.add_argument("--prompt")
    ap.add_argument("-n", type=int, default=256, help="most words to write for --prompt")
    ap.add_argument("--think", action="store_true", help="let the model reason before answering")
    ap.add_argument("--runtime", type=int, help="stop after this many seconds")
    a = ap.parse_args(argv)
    if a.action == "plan":
        p = build_plan(a, False)
        cmd, env = llama_cmd(p, a) if not p["refuse"] else (None, None)
        show(p, cmd, env)
        return 2 if p["refuse"] else 0
    return do_run(a)
