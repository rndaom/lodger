"""lodger stress: does anything crash when the model shares the PC? Uses the graphics card, so it needs
--allow-gpu. Best run while nobody is using the PC (the user's choice, 2026-10-28).

  lodger stress MODEL --allow-gpu [--game-appid 440] [--game-seconds 150] [--only-game]

With extras.llama_router on: stop llama-router first (systemctl --user stop llama-router; start it again after).

1. answer:   the model answers on the graphics card while your usual apps are open.
2. memory:   while the model writes, a test program takes graphics memory the way a game loads its
             textures (never leaving under 3 GiB free). The model must step aside. Needs build/vram-eater
             (tools/vram-eater.hip, AMD cards with ROCm only): skipped without it.
With --game-appid (a Steam game; Steam is started and closed again if it wasn't running):
3. no false alarm: Steam starts while the model writes. Steam is not using the card hard, so the
             model must keep going.
4. game:     while the model writes, the game starts. The model must give the card back. Then, with the
             game running, the model answers again (on the processor if the game is busy on the card).
5. race:     the model and the game start at the same moment. Either may get the card; nothing may crash.
After each phase: kernel log checked for graphics-card errors, crash dumps counted, and your desktop's
compositor must still be running. A per-app memory log runs the whole time (lodger measure)."""
import argparse
import os
import signal
import subprocess
import sys
import time

from . import config, sysmem
from .gpu import amd

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
LODGER = os.path.join(ROOT, "bin", "lodger")
EATER = os.path.join(ROOT, "build", "vram-eater")
RUNS = config.runs_dir()
LONG = "Write a long, detailed story about a lighthouse keeper and a storm. At least 3000 words."
SHORT = "In two short sentences, explain what a mixture-of-experts language model is."
BAD_KERNEL = ("MES failed", "ring timeout", "GPU reset", "VRAM is lost", "Not enough memory for command",
              "page fault", "job timedout", "GPU recovery")
STEPPED_ASIDE = "giving the graphics card back"


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def running(name):
    return subprocess.run(["pgrep", "-x", name], capture_output=True).returncode == 0


# XDG_CURRENT_DESKTOP -> the compositor's process name
COMPOSITORS = {"hyprland": "Hyprland", "sway": "sway", "niri": "niri", "gnome": "gnome-shell", "kde": "kwin_wayland",
               "cosmic": "cosmic-comp", "wayfire": "wayfire", "river": "river", "labwc": "labwc"}


def compositor():
    """The process name of the desktop's compositor, or None if lodger doesn't know this desktop."""
    for d in (os.environ.get("XDG_CURRENT_DESKTOP") or "").lower().split(":"):
        if d in COMPOSITORS:
            return COMPOSITORS[d]
    return None


COMPOSITOR = compositor()


def router_running():
    """The author's own llama-router service (extras.llama_router) would compete for the card."""
    return config.extra("llama_router") and \
        subprocess.run(["systemctl", "--user", "is-active", "-q", "llama-router"]).returncode == 0


def eater_why():
    """None if the test program that takes graphics memory can run on the chosen card, else why not."""
    if not os.access(EATER, os.X_OK):
        return f"no {EATER} (build it with  hipcc --offload-arch=<your card> -O2 -o build/vram-eater tools/vram-eater.hip)"
    dev = sysmem.dgpu()
    if dev is None or dev.vendor != "AMD" or amd.kfd_index(dev) is None:
        return "build/vram-eater needs an AMD card that ROCm lists"
    return None


def eater_env():
    return dict(os.environ, HIP_VISIBLE_DEVICES=str(amd.kfd_index(sysmem.dgpu())))


def health(since):
    """Problems since `since` (a 'YYYY-MM-DD HH:MM:SS' time), as a list of plain sentences."""
    bad = []
    klog = sh(["journalctl", "-k", "--since", since, "--no-pager", "-o", "cat"])
    dev = sysmem.dgpu()
    driver = dev.driver if dev and dev.driver else None
    hits = [l for l in klog.splitlines()
            if (driver is None or driver in l) and any(b.lower() in l.lower() for b in BAD_KERNEL)]
    if hits:
        bad.append("graphics-card errors in the kernel log:\n    " + "\n    ".join(hits[:8]))
    dumps = [l for l in sh(["coredumpctl", "list", "--since", since, "--no-pager", "--no-legend"]).splitlines() if l.strip()]
    if dumps:
        bad.append("programs crashed:\n    " + "\n    ".join(dumps[:8]))
    if COMPOSITOR and not running(COMPOSITOR):
        bad.append(f"{COMPOSITOR} (your desktop) is not running")
    return bad


def launch(model, prompt, n, env=None, background=False, extra=()):
    cmd = [LODGER, "run", model, "--allow-gpu", "--prompt", prompt, "-n", str(n), "--runtime", "900", *extra]
    e = dict(os.environ, **(env or {}))
    if background:
        return subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return subprocess.run(cmd, env=e, capture_output=True, text=True)


def wait_writing(t0, proc, limit=300):
    """Wait until the run started after t0 has loaded and is writing (llama-cli echoes the prompt after
    loading). Returns False if it ended or didn't get there in time."""
    while time.time() - t0 < limit and proc.poll() is None:
        for d in os.listdir(RUNS):
            out = os.path.join(RUNS, d, "output.txt")
            if "-run-" in d and os.path.exists(out) and os.path.getmtime(os.path.join(RUNS, d)) >= t0 - 1:
                with open(out, errors="ignore") as f:
                    if "\n> " in f.read():
                        return True
        time.sleep(1)
    return False


def finish(p, timeout=120):
    try:
        return p.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        p.send_signal(signal.SIGINT)                  # lodger stops its model on Ctrl+C
        return p.communicate(timeout=30)[0]


def stop_line(out):
    return next((l.split("STOPPING THE MODEL:", 1)[1].strip() for l in out.splitlines() if "STOPPING THE MODEL:" in l), "")


def used_card(out):
    return "Graphics card not used" not in out


# ---------- Steam game helpers (test harness only; lodger itself knows no game names) ----------

STARTED_STEAM = []


def game_procs():
    """The game's processes: everything under Steam's game wrapper (reaper), deepest first."""
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(d))
            except (OSError, IndexError, ValueError):
                pass
    out = []

    def walk(pid):
        for k in kids.get(pid, []):
            walk(k)
        out.append(pid)
    for r in (int(x) for x in sh(["pgrep", "-x", "reaper"]).split()):
        walk(r)
    return out


def game_vram():
    return sysmem.gpu_use(sysmem.dgpu(), game_procs())["vram"]


def start_steam(limit=240):
    """Start Steam and wait until it has settled (its start-up Wine processes gone)."""
    subprocess.Popen(["uwsm-app", "--", "steam", "-silent"], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    STARTED_STEAM.append(1)
    t0 = time.time()
    time.sleep(20)
    while time.time() - t0 < limit and (running("wineserver") or running("reaper")):
        time.sleep(2)
    time.sleep(10)
    return time.time() - t0


def start_game(appid, limit=300):
    """Start a Steam game; return None once it runs, else why not. Never types anything into Steam."""
    if not running("steam"):
        start_steam()
    subprocess.Popen(["steam", "-applaunch", str(appid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < limit:
        if running("reaper"):
            return None
        time.sleep(1)
    return f"Steam did not start game {appid} within {limit} s (signed out, update, or a dialog waiting?)"


def wait_drawing(limit=180, least=512 * sysmem.MiB):
    t0, peak = time.time(), 0
    while time.time() - t0 < limit and peak < least:
        peak = max(peak, game_vram())
        time.sleep(1)
    return peak


def close_game():
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        procs = game_procs()
        if not procs:
            break
        for pid in procs:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        t0 = time.time()
        while game_procs() and time.time() - t0 < wait:
            time.sleep(1)
    time.sleep(5)


# ---------- phases ----------

def phase(name, fn, results):
    since = time.strftime("%F %T")
    print(f"\n=== {name} ===", flush=True)
    ok, note = fn()
    time.sleep(3)
    bad = health(since)
    results.append((name, ok and not bad, note, bad))
    print(("PASS" if results[-1][1] else "FAIL") + f": {note}", flush=True)
    for b in bad:
        print("  ! " + b)
    return not bad


def main(argv):
    ap = argparse.ArgumentParser(prog="lodger stress", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--allow-gpu", action="store_true", help="required: the graphics card may be used")
    ap.add_argument("--game-appid", help="Steam app id to start and close automatically, e.g. 440 = TF2, 1422450 = Deadlock")
    ap.add_argument("--game-seconds", type=int, default=150, help="how long the game runs in phase 4")
    ap.add_argument("--only-game", action="store_true", help="skip phases 1 and 2")
    a = ap.parse_args(argv)
    if not a.allow_gpu:
        sys.exit("The stress test uses the graphics card: pass --allow-gpu.")
    if router_running():
        sys.exit("Stop llama-router first: systemctl --user stop llama-router (and start it again afterwards).")
    no_eater = eater_why()
    if COMPOSITOR is None:
        print(f"(note: lodger doesn't know your desktop ({os.environ.get('XDG_CURRENT_DESKTOP') or 'none set'}), "
              "so it can't check that its compositor survives)", flush=True)

    meas = subprocess.Popen([LODGER, "measure", "--seconds", "3600", "--interval", "1", "--label", "stress", "--quiet"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    results = []
    try:
        if not a.only_game:
            def answer():
                r = launch(a.model, SHORT, 120)
                return r.returncode == 0 and used_card(r.stdout), \
                    "model answered on the card. " + " ".join(r.stdout.strip().splitlines()[-2:])[:300]
            if not phase("1. answer with your apps open", answer, results):
                return report(results, meas)

            def memory():
                t0 = time.time()
                # the model leaves 4 GiB extra free so the eater has room to work
                p = launch(a.model, LONG, 6000, env={"LODGER_KILL_VRAM_FREE_MIB": "4096"}, background=True,
                           extra=["--leave-gpu", "4096"])
                if not wait_writing(t0, p):
                    p.kill()
                    return False, "model did not start writing within 5 minutes"
                time.sleep(5)
                e = subprocess.run([EATER, "3072", "20"], capture_output=True, text=True, timeout=300, env=eater_env())
                out = finish(p)
                why = stop_line(out)
                return STEPPED_ASIDE in out or "graphics memory left" in out, \
                    (f"model stepped aside ({why}). " if why else "model did NOT step aside. ") + e.stdout.strip().replace("\n", " ")
            if no_eater:
                print(f"\n=== 2. another app takes graphics memory ===\nSKIPPED: {no_eater}", flush=True)
            elif not phase("2. another app takes graphics memory", memory, results):
                return report(results, meas)

        if a.game_appid:
            if not running("steam"):
                def steam():
                    t0 = time.time()
                    p = launch(a.model, LONG, 6000, background=True)
                    if not wait_writing(t0, p):
                        p.kill()
                        return False, "model did not start writing within 5 minutes"
                    secs = start_steam()
                    alive = p.poll() is None
                    out = finish(p, timeout=5) if not alive else ""
                    if alive:
                        p.send_signal(signal.SIGINT)
                        p.communicate(timeout=30)
                    return alive, (f"Steam started and settled in {secs:.0f} s; the model kept writing (no false alarm)."
                                   if alive else f"model stopped while Steam started: {stop_line(out) or 'ended early'}")
                if not phase("3. Steam starts (not a game): no false alarm", steam, results):
                    return report(results, meas)

            def game():
                t0 = time.time()
                p = launch(a.model, LONG, 6000, background=True)
                if not wait_writing(t0, p):
                    p.kill()
                    return False, "model did not start writing within 5 minutes"
                g = start_game(a.game_appid)
                if g:
                    p.kill()
                    return False, g
                tg = time.time()
                out = finish(p)
                why = stop_line(out)
                stepped = STEPPED_ASIDE in out
                note = (f"model gave the card back {time.time() - tg:.0f} s after the game started ({why}). " if stepped
                        else "model did NOT step aside for the game. ")
                peak = wait_drawing()
                note += f"Game drawing ({peak // sysmem.MiB} MiB). "
                r = launch(a.model, SHORT, 120)
                note += ("Asked again while the game ran: answered on the "
                         + ("card (the game wasn't busy on it at that moment). " if used_card(r.stdout) else "processor. ")
                         if r.returncode == 0 or STEPPED_ASIDE in r.stdout else "Asked again while the game ran: FAILED. ")
                if STEPPED_ASIDE in r.stdout:
                    note += f"(it then stepped aside: {stop_line(r.stdout)}) "
                while time.time() - tg < a.game_seconds:
                    peak = max(peak, game_vram())
                    time.sleep(1)
                alive = game_procs() != []
                note += f"Game used up to {peak // sysmem.MiB} MiB; " + ("still running at the end." if alive else "HAD STOPPED by itself.")
                close_game()
                return stepped and alive and (r.returncode == 0 or STEPPED_ASIDE in r.stdout), note
            if not phase("4. a game starts while the model writes", game, results):
                return report(results, meas)

            def race():
                t0 = time.time()
                p = launch(a.model, LONG, 6000, background=True)
                g = start_game(a.game_appid)
                peak = wait_drawing() if not g else 0
                out = finish(p, timeout=60)
                card = used_card(out)
                why = stop_line(out)
                alive = game_procs() != []
                close_game()
                note = (("model took the card, then stepped aside: " + why if why else "model kept the card") if card
                        else "model saw the card was busy and ran on the processor") + \
                       f". Game drew ({peak // sysmem.MiB} MiB) and " + ("kept running." if alive else "HAD STOPPED.")
                return not g and alive, note
            phase("5. model and game start together", race, results)
    finally:
        if game_procs():
            close_game()
        if STARTED_STEAM:
            subprocess.run(["steam", "-shutdown"], capture_output=True, timeout=60)
    return report(results, meas)


def report(results, meas):
    meas.terminate()
    out = meas.communicate(timeout=60)[0] if meas.stdout else ""
    print("\n=== summary ===")
    for name, ok, note, bad in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    loc = [l for l in out.splitlines() if l.startswith("(saved in")]
    if loc:
        print("Memory log " + loc[-1].strip("()"))
    return 0 if results and all(r[1] for r in results) else 1
