"""Read live VRAM and RAM use from sysfs and /proc. Read-only; safe to call any time."""
import os
import re

from . import config, gpu, host

MiB = 1 << 20
GiB = 1 << 30
OURS = "lodger (model)"      # the model's own processes (its unit or cgroup is called lodger-*)


_read = host.read


# ---------- graphics cards (any vendor: see lodger/gpu) ----------

def cards():
    """Every card lodger can read memory from: {dev, card, pci, total, used, gtt_total, gtt_used, outputs},
    bytes. Built-in graphics report their own small "VRAM" here, like the kernel does."""
    out = []
    for d in gpu.devices():
        m = d.memory() if d.known else None
        if not m or not m["total"]:
            continue
        out.append({"dev": d, "card": d.card, "pci": d.pci, "total": m["total"], "used": m["used"],
                    "gtt_total": m["gtt_total"] or 0, "gtt_used": m["gtt_used"] or 0, "outputs": d.outputs()})
    return out


def dgpu():
    """The card lodger uses (see gpu.choose and gpu in config.toml), or None."""
    return gpu.choose(config.get()["gpu"])


def vram_free(dev=None):
    """Free memory on the card lodger uses (or `dev`), bytes; 0 if there is none or it can't be read."""
    d = dev or dgpu()
    f = d.free() if d else None
    return f or 0


def gpu_use(dev, pids):
    """What these processes hold on `dev` together: {vram, gtt} bytes."""
    tot = {"vram": 0, "gtt": 0}
    if dev:
        for rec in dev.usage(pids).values():
            tot["vram"] += rec["vram"]
            tot["gtt"] += rec["gtt"]
    return tot


# ---------- RAM ----------

def meminfo():
    """/proc/meminfo in bytes."""
    return host.meminfo()


# ---------- per process ----------

def app_name(pid):
    """Group processes by the systemd scope/service the desktop started them in:
    app-vesktop-2328.scope -> vesktop, wayland-wm@hyprland.desktop.service -> hyprland.
    Without a systemd user session, a user's processes are grouped by program name."""
    cg = _read(f"/proc/{pid}/cgroup", "") or ""
    path = cg.strip().split("::", 1)[-1]
    leaf = path.rsplit("/", 1)[-1]
    if leaf.startswith("lodger-"):
        return OURS
    if not leaf or leaf == "init.scope" or "user.slice" not in path:
        return comm(pid) if "user.slice" not in path and _own(pid) else "system"
    if leaf.startswith("wayland-wm@"):                       # the compositor, as uwsm starts it
        return re.split(r"[.@]", leaf[len("wayland-wm@"):])[0].lower()
    name = re.sub(r"\.(service|scope|slice)$", "", leaf)
    name = name.encode().decode("unicode_escape")          # gtk\x2dlaunch -> gtk-launch
    if name.startswith("app-"):
        name = name[4:]
        name = re.sub(r"^(Hyprland|gnome|kde|niri|sway)-", "", name)   # launched by the desktop's own launcher
        name = re.sub(r"@.*$", "", name)                    # vesktop@autostart -> vesktop
        name = re.sub(r"-[0-9a-f]{4,}$", "", name)          # trailing launch id
    return name


def _own(pid):
    """Is this process the current user's?"""
    for line in (_read(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith("Uid:"):
            return line.split()[1] == str(os.getuid())
    return False


def pids():
    return [int(p) for p in host.listdir("/proc") if p.isdigit()]


def comm(pid):
    return (_read(f"/proc/{pid}/comm", "?") or "?").strip()


def pss(pid):
    """Proportional RAM use (shared pages split between their users), bytes. Needs no root
    for our own processes; others read as 0."""
    s = _read(f"/proc/{pid}/smaps_rollup")
    if not s:
        return 0
    m = re.search(r"^Pss:\s+(\d+) kB", s, re.M)
    return int(m.group(1)) * 1024 if m else 0


def snapshot(with_pss=True):
    """Everything at one moment. Each GPU client is counted once, for the first process seen
    holding it (a client can be passed between processes)."""
    apps = {}
    all_pids = pids()
    names = {pid: app_name(pid) for pid in all_pids}
    for pid in all_pids:
        a = apps.setdefault(names[pid], {"vram": {}, "gtt": {}, "pss": 0, "procs": 0, "top": {}})
        a["procs"] += 1
        if with_pss:
            p = pss(pid)
            a["pss"] += p
            c = comm(pid)
            a["top"][c] = a["top"].get(c, 0) + p
    for d in gpu.devices():
        for pid, rec in d.usage(all_pids).items():
            a = apps.get(names.get(pid))
            if a is None:
                continue
            a["vram"][d.pci] = a["vram"].get(d.pci, 0) + rec["vram"]
            a["gtt"][d.pci] = a["gtt"].get(d.pci, 0) + rec["gtt"]
    apps = {k: v for k, v in apps.items() if v["procs"]}
    return {"cards": cards(), "mem": meminfo(), "apps": apps}
