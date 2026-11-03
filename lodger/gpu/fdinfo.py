"""What each process does on a graphics card, from /proc/<pid>/fdinfo, for every driver that follows the
kernel's DRM usage stats (https://docs.kernel.org/gpu/drm-usage-stats.html): amdgpu, i915, xe, msm, panthor ...
Read-only. Only file descriptors that point at /dev/dri/* are read (a browser holds thousands of others), and
never those of a card that is asleep when its driver's fdinfo may wake it (xe takes a runtime power reference
to read its engine counters; amdgpu and i915 don't; drivers lodger doesn't know are treated like xe).

A driver only needs a row in DRIVERS: which engines mean "the card is working for this app" (3D and
compute, not video or copy) and which memory regions are on the card itself vs in system RAM."""
import collections
import os
import re
import time

from .. import host

Table = collections.namedtuple("Table", "engines card system")

DRIVERS = {
    "amdgpu": Table(engines=("gfx", "compute"), card=("vram",), system=("gtt",)),
    "i915": Table(engines=("render", "compute"), card=("local",), system=("system",)),
    "xe": Table(engines=("rcs", "ccs"), card=("vram",), system=("system", "gtt")),
}
# any other driver: every engine counts, and regions are guessed from their names ("memory" is the spec's
# name for system RAM, used by drm_show_memory_stats in msm, panfrost, panthor, v3d ...)
GENERIC = Table(engines=None, card=("vram", "local"), system=("gtt", "system", "memory"))
# drivers whose fdinfo is safe to read while the card sleeps (checked in their sources, kernel 7.x)
NO_WAKE = {"amdgpu", "i915"}

UNITS = {"": 1, "B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}
_REGION = re.compile(r"^drm-(resident|memory|total)-(.+)$")
_NODE = re.compile(r"^(?:card|renderD)\d+$")


def parse(text):
    """The drm-* keys of one fdinfo file: {key: value}, or None if it isn't a DRM client.
    Split on the first ':' (older amdgpu prints 'drm-memory-gtt: \\t2048 KiB', with a space before the tab)."""
    if not text or "drm-driver" not in text:
        return None
    kv = {}
    for line in text.splitlines():
        k, sep, v = line.partition(":")
        if sep and k.startswith("drm-"):
            kv[k.strip()] = v.strip()
    return kv if "drm-driver" in kv else None


def size(value):
    """'2048 KiB' -> 2097152; unknown units count as 0 rather than a wrong number."""
    num, _, unit = (value or "0").partition(" ")
    try:
        return int(num) * UNITS[unit.strip()] if unit.strip() in UNITS else 0
    except ValueError:
        return 0


def _num(value):
    try:
        return int((value or "0").split()[0])
    except (ValueError, IndexError):
        return 0


def _region_kind(table, region):
    """'card', 'system' or None for a memory region name like vram, vram0, local0, gtt, stolen-system0."""
    if region.startswith("stolen-"):
        return None
    for kind, prefixes in (("card", table.card), ("system", table.system)):
        for p in prefixes:
            if region == p or (region.startswith(p) and region[len(p):].isdigit()):
                return kind
    return None


def client(kv):
    """One DRM client as a record:
    {driver, pci, id, vram: bytes on the card, gtt: bytes of system RAM the card uses,
     busy_ns: time the busiest 3D/compute engine spent on it (drivers that count nanoseconds),
     cycles: {engine: (busy cycles, total cycles)} (drivers that count cycles, e.g. xe)}."""
    driver = kv["drm-driver"]
    table = DRIVERS.get(driver, GENERIC)
    rec = {"driver": driver, "pci": kv.get("drm-pdev", "?"), "id": kv.get("drm-client-id"),
           "vram": 0, "gtt": 0, "busy_ns": 0, "cycles": {}}
    # resident memory is what is really there; drm-memory-* is the older name for the same number; a driver
    # that prints neither for a region (resident is optional in the spec) is counted by its total
    regions, rank = {}, {"resident": 3, "memory": 2, "total": 1}
    for k, v in kv.items():
        m = _REGION.match(k)
        if m and rank[m.group(1)] > regions.get(m.group(2), (0, 0))[0]:
            regions[m.group(2)] = (rank[m.group(1)], size(v))
    regions = {r: n for r, (_, n) in regions.items()}
    for region, n in regions.items():
        kind = _region_kind(table, region)
        if kind == "card":
            rec["vram"] += n
        elif kind == "system":
            rec["gtt"] += n
    busy = []
    for k, v in kv.items():
        if k.startswith("drm-engine-") and not k.startswith("drm-engine-capacity-"):
            e = k[len("drm-engine-"):]
            if table.engines is None or e in table.engines:
                busy.append(_num(v) // max(1, _num(kv.get(f"drm-engine-capacity-{e}", "1"))))
        elif k.startswith("drm-cycles-"):
            e = k[len("drm-cycles-"):]
            if table.engines is None or e in table.engines:
                cap = max(1, _num(kv.get(f"drm-engine-capacity-{e}", "1")))
                rec["cycles"][e] = (_num(v) // cap, _num(kv.get(f"drm-total-cycles-{e}")))
    rec["busy_ns"] = max(busy, default=0)
    return rec


_nodes = {}             # (host.ROOT, "renderD128") -> (time, pci, driver)


def node_card(node):
    """(pci, driver) of a /dev/dri node like renderD128, from sysfs (cached for a minute), or (None, None)."""
    key = (host.ROOT, node)
    hit = _nodes.get(key)
    if hit and time.monotonic() - hit[0] < 60:
        return hit[1:]
    pci = host.realname(f"/sys/class/drm/{node}/device")
    driver = host.realname(f"/sys/bus/pci/devices/{pci}/driver") if pci else None
    _nodes[key] = (time.monotonic(), pci, driver)
    return pci, driver


def asleep(pci):
    return (host.read(f"/sys/bus/pci/devices/{pci}/power/runtime_status") or "").strip() in ("suspended", "suspending")


def drm_fds(pid, pci=None):
    """The process's file descriptors that point at a graphics card (/dev/dri/*), skipping cards other than
    `pci` (when given) and sleeping cards whose driver could be woken by reading. If the fd links can't be
    listed, every fdinfo entry (the kernel's access rules for both are the same, so this is rare)."""
    fds = host.listdir(f"/proc/{pid}/fd")
    if not fds:
        return host.listdir(f"/proc/{pid}/fdinfo")
    out = []
    for fd in fds:
        try:
            target = os.readlink(host.path(f"/proc/{pid}/fd/{fd}"))
        except OSError:
            continue
        node = target.rsplit("/", 1)[-1]
        if not (target.startswith("/dev/dri/") or "/dri/" in target) or not _NODE.match(node):
            continue
        card, driver = node_card(node)
        if card and pci and card != pci:
            continue
        if card and driver not in NO_WAKE and asleep(card):
            continue
        out.append(fd)
    return out


def holds(pid, path):
    """Does the process have `path` open (say /dev/kfd, which every ROCm program opens)?"""
    for fd in host.listdir(f"/proc/{pid}/fd"):
        try:
            if os.readlink(host.path(f"/proc/{pid}/fd/{fd}")) == path:
                return True
        except OSError:
            pass
    return False


def clients(pid, pci=None):
    """DRM clients a process holds: {(pci, client_id): record} (only on `pci` when given). An app often opens
    the same client through several file descriptors; it is counted once."""
    out = {}
    base = f"/proc/{pid}/fdinfo"
    for fd in drm_fds(pid, pci):
        kv = parse(host.read(f"{base}/{fd}"))
        if not kv:
            continue
        rec = client(kv)
        key = (rec["pci"], rec["id"] if rec["id"] is not None else f"{pid}/{fd}")
        out.setdefault(key, rec)
    return out


def per_card(pids):
    """Graphics memory (on the card + system RAM it maps) the processes hold on each card: {pci: bytes}."""
    seen, out = set(), {}
    for pid in pids:
        for key, rec in clients(pid).items():
            if key not in seen:
                seen.add(key)
                out[key[0]] = out.get(key[0], 0) + rec["vram"] + rec["gtt"]
    return out


def usage(pci, pids):
    """What each process does on the card at `pci`: {pid: {vram, gtt, busy_ns, cycles}}. A client passed
    between processes is counted once, for the first process holding it."""
    seen, out = set(), {}
    for pid in pids:
        for key, rec in clients(pid, pci).items():
            if key[0] != pci or key in seen:
                continue
            seen.add(key)
            add(out.setdefault(pid, empty()), rec)
    return out


def empty():
    return {"vram": 0, "gtt": 0, "busy_ns": 0, "cycles": {}}


def add(total, rec):
    """Add one client's (or process's) record into a running total."""
    total["vram"] += rec["vram"]
    total["gtt"] += rec["gtt"]
    total["busy_ns"] += rec["busy_ns"]
    for e, (c, t) in rec["cycles"].items():
        oc, ot = total["cycles"].get(e, (0, 0))
        total["cycles"][e] = (oc + c, max(ot, t))      # same clock for every client on one card
    return total
