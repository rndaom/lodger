"""lodger doctor: what lodger sees on this PC, and what it would do with it. Only reads: it starts no model,
writes nothing, and never wakes a graphics card that is asleep (its memory isn't read then).

  lodger doctor

Paste its output into a bug report: it holds no personal files, only the hardware, the settings and paths
(your home folder is written as ~)."""
import contextlib
import glob
import http.client
import io
import os
import platform
import re
import shutil
import subprocess
import sys

from . import __version__, config, engine, fence, gpu, host, neighbors, reserve
from .gpu import nvidia

GiB = 1 << 30


def gib(b):
    return f"{b / GiB:.1f} GiB"


def system():
    try:
        osr = platform.freedesktop_os_release()
        name = osr.get("PRETTY_NAME") or osr.get("NAME", "?")
    except OSError:
        name = "unknown Linux"
    u = os.uname()
    return f"{name}, Linux {u.release} ({u.machine}), Python {platform.python_version()}"


def memory():
    m = host.meminfo()
    swap = f", swap {gib(m['SwapTotal'])}" if m.get("SwapTotal") else ", no swap"
    lim = host.cgroup_limit()
    capped = f"; the cgroups lodger runs in cap it at {gib(lim[0])} ({gib(lim[1])} of that free)" if lim else ""
    return f"RAM {gib(m.get('MemTotal', 0))}, {gib(m.get('MemAvailable', 0))} available now{swap}{capped}"


def private(text, home=None):
    """The report with the home folder written as ~ (it names the user)."""
    home = (os.path.expanduser("~") if home is None else home).rstrip("/")
    if not home:
        return text
    return re.sub(re.escape(home) + r"(?=/|\b|$)", "~", text)


def missing_libs(bin_dir):
    """{library: [programs or backends that need it]} that the system's loader can't find, for llama.cpp's
    programs and the backend libraries it loads later (ldd reads them; nothing of llama.cpp runs)."""
    if not bin_dir or not shutil.which("ldd"):
        return {}
    files = [os.path.join(bin_dir, n) for n in ("llama-server", "llama-fit-params")]
    for sub in (bin_dir, os.path.join(bin_dir, "..", "lib"), os.path.join(bin_dir, "..", "lib64")):
        files += sorted(glob.glob(os.path.join(sub, "libggml*.so*")) + glob.glob(os.path.join(sub, "libllama*.so*")))
    gone = {}
    for f in dict.fromkeys(os.path.realpath(f) for f in files if os.path.isfile(f)):
        try:
            r = subprocess.run(["ldd", f], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in r.stdout.splitlines():
            m = re.match(r"\s*(\S+) => not found", line)
            if m:
                gone.setdefault(m.group(1), []).append(os.path.basename(f))
    return gone


def serving(port=8181):
    """lodger serve on this PC: (systemd unit state or None, what /health on `port` answered or None)."""
    unit = None
    if shutil.which("systemctl"):
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            r = subprocess.run(["systemctl", "--user", "show", "-P", "ActiveState,UnitFileState", "lodger.service"],
                               capture_output=True, text=True, timeout=10)
            got = [x.strip() for x in r.stdout.splitlines() if x.strip()]
            if r.returncode == 0 and len(got) == 2 and got[1]:
                unit = f"{got[0]}, {got[1]}"
    health = None
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        c.request("GET", "/health")
        resp = c.getresponse()
        body = resp.read(4096).decode(errors="replace")
        c.close()
        health = f"{resp.status} {body.strip()}"
    except (OSError, http.client.HTTPException):
        pass
    return unit, health


def card_lines(dev, chosen):
    name = dev.name() or "unknown model"
    kind = "built-in (shares RAM)" if dev.integrated else "own memory"
    out = [f"  {dev.pci}  {dev.vendor} {name} [{dev.vendor_id:04x}:{dev.device_id:04x}]"
           + ("   <- lodger uses this one" if chosen else "")]
    out.append(f"      driver {dev.driver or 'none'}, {kind}, power {dev.power()}"
               + ("" if dev.known else ", lodger has no backend for this driver"))
    if dev.asleep():
        out.append("      memory: not read (the card is asleep; lodger doesn't wake it to look)")
    else:
        m = dev.memory()
        if m is None:
            out.append(f"      memory: unknown ({dev.why_unknown()})")
        elif dev.integrated:
            gtt = f" (the driver lets it map up to {gib(m['gtt_total'])})" if m.get("gtt_total") else ""
            out.append(f"      memory: can borrow {gib(dev.free(m))} of RAM now{gtt}")
        else:
            out.append(f"      memory: {gib(m['total'])}, {gib(m['used'])} used, {gib(m['total'] - m['used'])} free")
    screens = dev.outputs()
    if screens:
        out.append("      screens: " + ", ".join(screens))
    return out


def main(argv):
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            return report()
    finally:
        sys.stdout.write(private(out.getvalue()))
        sys.stdout.flush()


def report():
    checked, errors, warnings = config.check()
    try:
        cfg = config.get()
    except SystemExit:
        cfg = checked                       # a broken setting keeps its default here, and is listed
        config.use(cfg)
    problems = list(errors)
    print(f"lodger doctor {__version__} (reads only; wakes nothing)\n")
    print("System:   " + system())
    print("Memory:   " + memory())
    src, detail = gpu.power_source()
    print("Power:    " + detail)
    cfgnote = "" if os.path.exists(config.PATH) else " (not there: defaults)"
    print(f"Settings: {config.PATH}{cfgnote}")
    for e in errors:
        print("          problem: " + e)
    for w in warnings:
        print("          warning: " + w)
    on = [k for k, v in cfg["extras"].items() if v]
    if on:
        print("          extras on: " + ", ".join(on))

    devs = gpu.devices()
    chosen = gpu.choose(cfg["gpu"], devs)
    print("\nGraphics:")
    if not devs:
        print("  none found")
    for d in devs:
        print("\n".join(card_lines(d, chosen is not None and d.pci == chosen.pci)))
    print(f"  setting: gpu = \"{cfg['gpu']}\"")
    if chosen is None:
        why = gpu.why_none(cfg["gpu"], devs)
        print(f"  lodger runs models on the processor only: {why}")
        if devs and cfg["gpu"] != "none":
            problems.append(why)
    else:
        if chosen.asleep() and not cfg["wake_gpu"]:
            print("  the chosen card is asleep: lodger leaves it alone until something else wakes it (wake_gpu = true to use it anyway)")
        if src == "battery" and not chosen.integrated and not cfg["gpu_on_battery"]:
            print("  on battery: lodger keeps work off the card (gpu_on_battery = true to allow it)")
        total = None if chosen.asleep() else chosen.total()
        if total:
            keep, why = reserve.reserve(chosen.pci, total)
            print(f"  left free for your desktop: {gib(keep)}" + (f" ({why})" if why else ""))
            r = reserve.report(chosen.pci, total)
            if r["warning"]:
                print("  note: " + r["warning"] + (f"; lodger keeps {gib(r['keep'])} at most" if r["capped"] else ""))
                problems.append("the learned desktop reserve looks too high")
        sees_all, how = neighbors.coverage(chosen)
        print("  apps lodger sees on it: " + ("all" if sees_all else "yours") + f" ({how})")
        if chosen.crash_report():
            print("  the card reset or hung in the last few minutes (the kernel holds its driver's crash report): "
                  "lodger keeps off it for now")
            problems.append("the graphics card reset or hung recently")
        hot = chosen.too_hot()
        if hot:
            print(f"  at its critical temperature: {hot} (lodger keeps off it while it is)")
            problems.append("the graphics card is at its critical temperature")

    e = engine.describe()
    print("\nllama.cpp:")
    if not e["root"]:
        print("  not found. `lodger setup` downloads it; or set llama_cpp in the settings, or LODGER_LLAMA_CPP")
        problems.append("no llama.cpp")
    else:
        print(f"  {e['root']} (from {e['from']})")
        missing = [t for t in ("llama-server", "llama-fit-params") if not e[t]]
        if missing:
            print("  missing: " + ", ".join(missing))
            problems.append("llama.cpp is missing " + ", ".join(missing))
        print("  graphics backends: " + (", ".join(e["backends"]) or "none (processor only)"))
        if chosen is not None:
            t = engine.for_device(chosen, devs)
            if "why" in t:
                print(f"  can't use the chosen card: {t['why']}")
                problems.append(t["why"])
            else:
                env = " ".join(f"{k}={v}" for k, v in t["env"].items())
                print(f"  reaches the chosen card as {t['name']} ({env})")
        gone = missing_libs(e["bin"])
        for lib, who in sorted(gone.items()):
            print(f"  MISSING library {lib} (needed by {', '.join(sorted(set(who)))})")
        if gone:
            problems.append("llama.cpp can't load " + ", ".join(sorted(gone)) + " (install them, or `lodger setup "
                            "--backend` another build)")
        if "vulkan" in e["backends"]:
            layer = engine.vulkan_layer()
            print("  Vulkan: " + (f"Mesa's device-select layer found ({layer})" if layer else
                                  "Mesa's device-select layer NOT found (VkLayer_MESA_device_select.json): lodger uses "
                                  "Vulkan only where one card is listed"))
        if "cuda" in e["backends"]:
            drv = nvidia.driver_version()
            ok = nvidia.cuda_ok(e["cuda"], drv) if e["cuda"] else None
            print(f"  CUDA: build for CUDA {e['cuda'] or '? (uses the system CUDA)'}, NVIDIA driver "
                  + (".".join(map(str, drv)) if drv else "not loaded")
                  + {True: " (fine)", False: " (TOO OLD for this build)", None: ""}[ok])
            if ok is False:
                problems.append(f"NVIDIA driver too old for this llama.cpp's CUDA {e['cuda']} build")

    print()
    kind, why = fence.mode()
    print(f"\nFence:    {kind}: {why}")
    if kind == "systemd":
        capped, cwhy = fence.systemd_memory()
        print("          memory cap: " + ("the user manager has the memory controller (each unit's cap is checked "
                                         "again once it runs)" if capped else f"NOT applied ({cwhy}): the watchdog enforces it"))
        if not capped:
            problems.append("no hard memory fence")
        d, nwhy = fence.delegated()
        print("          without systemd lodger would use " + (f"its own cgroups under {d}" if d else f"no hard fence ({nwhy})"))
    elif kind == "none":
        problems.append("no hard memory fence")
    left = fence.find_orphans()
    for o in left:
        where = o["where"] if isinstance(o["where"], str) else "processes " + ", ".join(map(str, o["where"]))
        print(f"          LEFT BEHIND: {o['name']} ({o['kind']}: {where}); the lodger that started it ({o['owner']}) is gone")
    if left:
        problems.append(f"{len(left)} model fence(s) left behind by a lodger that is gone (the next `lodger serve` "
                        "or `lodger run` stops them)")

    from . import catalog
    try:
        models, default = catalog.load()
        n = len([m for m in models.values() if not getattr(m, "skip", None)])
        print(f"Models:   {n} in {catalog.CONFIG}" + (f", default {default}" if default else ""))
    except Exception as ex:                  # a broken models.toml is a finding, not a crash
        print(f"Models:   can't read {catalog.CONFIG}: {ex}")
        problems.append("models.toml unreadable")

    unit, health = serving()
    print("Serving:  " + ("lodger.service " + unit if unit else "no lodger.service")
          + "; 127.0.0.1:8181 " + (f"answers /health: {health}" if health else "doesn't answer"))

    print("\n" + ("All fine." if not problems else "To look at: " + "; ".join(problems)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
