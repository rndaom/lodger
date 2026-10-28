"""Where lodger reads the PC's state from (/sys and /proc). Tests point ROOT at recorded copies."""
import os

ROOT = ""


def path(p):
    return ROOT + p


def read(p, default=None):
    try:
        with open(ROOT + p) as f:
            return f.read()
    except OSError:
        return default


def read_int(p, default=0):
    s = read(p)
    return int(s) if s and s.strip().isdigit() else default


def meminfo():
    """/proc/meminfo in bytes."""
    out = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        parts = v.split()
        if parts:
            out[k] = int(parts[0]) * (1024 if len(parts) > 1 else 1)
    return out


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
