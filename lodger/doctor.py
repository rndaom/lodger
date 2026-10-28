"""lodger doctor: what lodger sees on this PC, and what it would do with it. Only reads: it starts no model,
writes nothing, and never wakes a graphics card that is asleep (its memory isn't read then).

  lodger doctor

Paste its output into a bug report: it holds no personal files, only the hardware, the settings and paths."""
import os
import platform
import sys

from . import config, engine, fence, gpu, host, reserve

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
    return f"RAM {gib(m.get('MemTotal', 0))}, {gib(m.get('MemAvailable', 0))} available now{swap}"


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
    cfg = config.get()
    problems = []
    print("lodger doctor (reads only; wakes nothing)\n")
    print("System:   " + system())
    print("Memory:   " + memory())
    src, detail = gpu.power_source()
    print("Power:    " + detail)
    cfgnote = "" if os.path.exists(config.PATH) else " (not there: defaults)"
    print(f"Settings: {config.PATH}{cfgnote}")
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

    from . import model
    try:
        model.gguf_reader()
        print(f"\nModel files: read with {model.GGUF_FROM}")
    except SystemExit as ex:
        print(f"\nModel files: can't read them: {ex}")
        problems.append("no gguf package")

    kind, why = fence.mode()
    print(f"\nFence:    {kind}: {why}")
    if kind == "systemd":
        d, nwhy = fence.delegated()
        print("          without systemd lodger would use " + (f"its own cgroups under {d}" if d else f"no hard fence ({nwhy})"))
    elif kind == "none":
        problems.append("no hard memory fence")

    from . import catalog
    try:
        models, default = catalog.load()
        n = len([m for m in models.values() if not getattr(m, "skip", None)])
        print(f"Models:   {n} in {catalog.CONFIG}" + (f", default {default}" if default else ""))
    except Exception as ex:                  # a broken models.toml is a finding, not a crash
        print(f"Models:   can't read {catalog.CONFIG}: {ex}")
        problems.append("models.toml unreadable")

    print("\n" + ("All fine." if not problems else "To look at: " + "; ".join(problems)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
