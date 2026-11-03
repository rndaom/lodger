"""Where lodger reads the PC's state from (/sys and /proc). Tests point ROOT at recorded copies."""
import os

ROOT = ""


def path(p):
    return ROOT + p


def read(p, default=None):
    """A file's text. Bytes that aren't UTF-8 (any process's environment, name or cgroup may hold them) are
    replaced, never an error."""
    try:
        with open(ROOT + p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return default


def read_int(p, default=0):
    s = read(p)
    return int(s) if s and s.strip().isdigit() else default


def meminfo():
    """/proc/meminfo in bytes. It shows the whole machine, so where lodger runs inside a memory limit (a
    container, a capped user slice), MemAvailable and MemTotal are lowered to that limit's room (cgroup_limit).
    Kernels before 3.14 have no MemAvailable: MemFree stands in (smaller, so on the safe side)."""
    out = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        parts = v.split()
        if parts and parts[0].isdigit():
            out[k] = int(parts[0]) * (1024 if len(parts) > 1 else 1)
    if "MemAvailable" not in out and "MemFree" in out:
        out["MemAvailable"] = out["MemFree"]
    lim = cgroup_limit()
    if lim and "MemAvailable" in out:
        out["MemTotal"] = min(out.get("MemTotal", lim[0]), lim[0])
        out["MemAvailable"] = min(out["MemAvailable"], lim[1])
    return out


UNLIMITED = 1 << 60     # cgroup v1 writes "no limit" as a number near 2^63


def _stat(path, *keys):
    """The first of `keys` found in a memory.stat file, bytes (0 if none)."""
    vals = dict(line.split(None, 1) for line in (read(path) or "").splitlines() if len(line.split()) == 2)
    for k in keys:
        if vals.get(k, "").strip().isdigit():
            return int(vals[k])
    return 0


def _levels(path):
    """The cgroup folders a model lodger starts is inside too: the root and every folder down to lodger's own,
    leaving out lodger's own when it is a systemd service or scope (the model runs in a unit of its own)."""
    parts = [p for p in path.split("/") if p]
    if parts and parts[-1].endswith((".service", ".scope")):
        parts = parts[:-1]
    return ["/".join(parts[:n]) for n in range(len(parts) + 1)]


def cgroup_limit():
    """(limit, room) in bytes for the memory limits on the cgroups lodger runs in (cgroup v2 memory.max and
    memory.high, v1 memory.limit_in_bytes), or None without any: limit is the smallest, room the least left
    under one (limit - charged + file cache the kernel can drop)."""
    lim = room = None
    for line in (read("/proc/self/cgroup") or "").splitlines():
        hid, ctl, path = (line.split(":", 2) + ["", ""])[:3]
        if hid == "0" and ctl == "":                                      # cgroup v2
            base, files = "/sys/fs/cgroup", ("memory.max", "memory.high")
            used_f, stat_keys = "memory.current", ("inactive_file",)
        elif "memory" in ctl.split(","):                                  # cgroup v1
            base, files = "/sys/fs/cgroup/memory", ("memory.limit_in_bytes",)
            used_f, stat_keys = "memory.usage_in_bytes", ("total_inactive_file", "inactive_file")
        else:
            continue
        for level in _levels(path.strip()):
            d = base + ("/" + level if level else "")
            caps = [int(v) for v in ((read(f"{d}/{f}") or "").strip() for f in files) if v.isdigit()]
            caps = [c for c in caps if c < UNLIMITED]
            if not caps:
                continue
            cap = min(caps)
            left = max(0, cap - read_int(f"{d}/{used_f}") + _stat(f"{d}/memory.stat", *stat_keys))
            lim = cap if lim is None else min(lim, cap)
            room = min(left, cap) if room is None else min(room, left, cap)
    return (lim, room) if lim is not None else None


def listdir(p):
    try:
        return os.listdir(ROOT + p)
    except OSError:
        return []


def realname(p):
    """Last part of where a /sys link points (a PCI address, a driver name), or None."""
    try:
        return os.path.basename(os.path.realpath(ROOT + p)) if os.path.lexists(ROOT + p) else None
    except OSError:
        return None
