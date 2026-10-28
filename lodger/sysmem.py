"""Read live VRAM and RAM use from sysfs and /proc. Read-only; safe to call any time."""
import os
import re

MiB = 1 << 20
GiB = 1 << 30


def _read(path, default=None):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def _int(path, default=0):
    s = _read(path)
    return int(s) if s and s.strip().isdigit() else default


# ---------- graphics cards ----------

def cards():
    """Every amdgpu card with VRAM: {pci, card, total, used, gtt_total, gtt_used} in bytes."""
    out = []
    for name in sorted(os.listdir("/sys/class/drm")):
        if not re.fullmatch(r"card\d+", name):
            continue
        dev = f"/sys/class/drm/{name}/device"
        total = _int(f"{dev}/mem_info_vram_total")
        if not total:
            continue
        out.append({
            "card": name,
            "pci": os.path.basename(os.path.realpath(dev)),
            "total": total,
            "used": _int(f"{dev}/mem_info_vram_used"),
            "gtt_total": _int(f"{dev}/mem_info_gtt_total"),
            "gtt_used": _int(f"{dev}/mem_info_gtt_used"),
            "outputs": connected_outputs(name),
        })
    return out


def dgpu():
    """The card with the most VRAM (same rule as ~/.local/bin/llama-server)."""
    cs = cards()
    return max(cs, key=lambda c: c["total"]) if cs else None


def vram_free(card=None):
    c = card or dgpu()
    if not c:
        return 0
    dev = f"/sys/class/drm/{c['card']}/device"
    return _int(f"{dev}/mem_info_vram_total") - _int(f"{dev}/mem_info_vram_used")


def connected_outputs(card):
    outs = []
    for name in os.listdir("/sys/class/drm"):
        if name.startswith(card + "-") and (_read(f"/sys/class/drm/{name}/status", "") or "").strip() == "connected":
            outs.append(name[len(card) + 1:])
    return sorted(outs)


# ---------- RAM ----------

def meminfo():
    """/proc/meminfo in bytes."""
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, v = line.split(":", 1)
        parts = v.split()
        out[k] = int(parts[0]) * (1024 if len(parts) > 1 else 1)
    return out


# ---------- per process ----------

def app_name(pid):
    """Group processes by the systemd scope/service the desktop started them in:
    app-vesktop-2328.scope -> vesktop, wayland-wm@hyprland.desktop.service -> hyprland."""
    cg = _read(f"/proc/{pid}/cgroup", "") or ""
    path = cg.strip().split("::", 1)[-1]
    leaf = path.rsplit("/", 1)[-1]
    if not leaf or leaf == "init.scope" or "user.slice" not in path:
        return "system"
    if leaf.startswith("wayland-wm@"):
        return "hyprland"
    if leaf.startswith("lodger-"):
        return "lodger (model)"
    name = re.sub(r"\.(service|scope|slice)$", "", leaf)
    name = name.encode().decode("unicode_escape")          # gtk\x2dlaunch -> gtk-launch
    if name.startswith("app-"):
        name = name[4:]
        name = re.sub(r"^Hyprland-", "", name)              # launched by a Hyprland keybind
        name = re.sub(r"@.*$", "", name)                    # vesktop@autostart -> vesktop
        name = re.sub(r"-[0-9a-f]{4,}$", "", name)          # trailing launch id
    return name


def pids():
    return [int(p) for p in os.listdir("/proc") if p.isdigit()]


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


_UNITS = {"": 1, "KiB": 1024, "MiB": MiB, "GiB": GiB}


def gpu_clients(pid):
    """amdgpu clients a process holds: {(pci, client_id): {vram, gtt, busy_ns}}. vram/gtt are resident
    bytes; busy_ns is the time the card's 3D or compute engine has spent on this client so far."""
    out = {}
    fddir = f"/proc/{pid}/fdinfo"
    try:
        fds = os.listdir(fddir)
    except OSError:
        return out
    for fd in fds:
        s = _read(f"{fddir}/{fd}")
        if not s or "drm-driver:\tamdgpu" not in s:
            continue
        kv = dict(line.split(":\t", 1) for line in s.splitlines() if ":\t" in line)
        key = (kv.get("drm-pdev", "?"), kv.get("drm-client-id", fd))
        if key in out:
            continue
        rec = {"busy_ns": max(int(kv.get(f"drm-engine-{e}", "0 ns").split()[0]) for e in ("gfx", "compute"))}
        for mem in ("vram", "gtt"):
            v = kv.get(f"drm-resident-{mem}") or kv.get(f"drm-memory-{mem}") or "0"
            num, *unit = v.split()
            rec[mem] = int(num) * _UNITS.get(unit[0] if unit else "", 1)
        out[key] = rec
    return out


def snapshot(with_pss=True):
    """Everything at one moment. Each GPU client is counted once, for the first process seen
    holding it (a client can be passed between processes)."""
    seen = set()
    apps = {}
    for pid in pids():
        name = app_name(pid)
        a = apps.setdefault(name, {"vram": {}, "gtt": {}, "pss": 0, "procs": 0, "top": {}})
        a["procs"] += 1
        if with_pss:
            p = pss(pid)
            a["pss"] += p
            c = comm(pid)
            a["top"][c] = a["top"].get(c, 0) + p
        for key, rec in gpu_clients(pid).items():
            if key in seen:
                continue
            seen.add(key)
            pci = key[0]
            a["vram"][pci] = a["vram"].get(pci, 0) + rec["vram"]
            a["gtt"][pci] = a["gtt"].get(pci, 0) + rec["gtt"]
    apps = {k: v for k, v in apps.items() if v["procs"]}
    return {"cards": cards(), "mem": meminfo(), "apps": apps}
