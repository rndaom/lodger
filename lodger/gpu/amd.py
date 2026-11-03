"""AMD cards (amdgpu). Memory comes from the driver's sysfs files, per-app use from fdinfo. Both are safe
to read while the card sleeps."""
import os

from .. import host
from . import fdinfo


# kernels from 6.7 show board_info on every card but APUs (amdgpu hides it when adev->flags & AMD_IS_APU)
BOARD_INFO_SINCE = (6, 7)


def _kernel():
    rel = (host.read("/proc/sys/kernel/osrelease") or "").strip()
    try:
        return tuple(int(x) for x in rel.split("-")[0].split(".")[:2])
    except ValueError:
        return None


def integrated(dev):
    """A processor's built-in graphics (APU) shares the PC's RAM; on APUs "VRAM" is just a piece of RAM the
    BIOS set aside. Signs of a card with its own memory chips, sturdiest first:
    - board_info: amdgpu shows it on every card but APUs (kernels since 6.7, where its absence decides;
      only its presence is checked: reading it asks the card);
    - mem_info_vram_vendor: only on cards with their own memory, but hidden when the card doesn't report the
      chips' maker (some older GCN cards);
    - on an older kernel showing neither: a "VRAM" under 1 GiB is an APU's carve-out (cards had 2 GiB+).
    Taking an APU for a card is the safe mistake (lodger then uses only the carve-out); the other way round
    would budget a card's memory from RAM, so the size rule only calls tiny ones built-in."""
    if os.path.exists(host.path(f"{dev.sys}/board_info")) or host.read(f"{dev.sys}/mem_info_vram_vendor"):
        return False
    k = _kernel()
    if k and k >= BOARD_INFO_SINCE:
        return True
    total = host.read_int(f"{dev.sys}/mem_info_vram_total")
    return 0 < total < (1 << 30)


def memory(dev):
    """{total, used, gtt_total, gtt_used} in bytes (GTT: system RAM the card may borrow), or None unless every
    number reads (a missing "used" must never look like a free card)."""
    vals = {}
    for k, f in (("total", "vram_total"), ("used", "vram_used"), ("gtt_total", "gtt_total"), ("gtt_used", "gtt_used")):
        s = (host.read(f"{dev.sys}/mem_info_{f}") or "").strip()
        if not s.isdigit():
            return None
        vals[k] = int(s)
    return vals if vals["total"] else None


def why_unknown(dev):
    return None if memory(dev) else "amdgpu doesn't show all of this card's memory numbers in sysfs (mem_info_*)"


def usage(dev, pids):
    return fdinfo.usage(dev.pci, pids)


def has_busy(dev):
    return os.path.exists(host.path(f"{dev.sys}/gpu_busy_percent"))


def busy(dev):
    """How busy the card's graphics engine is, 0..1 (gpu_busy_percent: the power firmware's own average), or
    None. Only asked while the card is awake: older kernels may wake it to answer."""
    s = (host.read(f"{dev.sys}/gpu_busy_percent") or "").strip()
    return int(s) / 100 if s.isdigit() and int(s) <= 100 else None


def kfd_index(dev):
    """ROCm's number for this card (HIP_VISIBLE_DEVICES): its place among the GPU nodes in the kernel's KFD
    topology, matched by PCI address. None if ROCm can't see it."""
    base = "/sys/class/kfd/kfd/topology/nodes"
    nodes = sorted((int(n) for n in host.listdir(base) if n.isdigit()))
    i = 0
    bus, slot = dev.pci.rsplit(":", 1)[0], dev.pci.rsplit(":", 1)[1]
    domain, busnum = (int(x, 16) for x in bus.split(":"))
    devnum, fn = (int(x, 16) for x in slot.split("."))
    location = (busnum << 8) | (devnum << 3) | fn
    for n in nodes:
        if not host.read_int(f"{base}/{n}/gpu_id"):
            continue                                    # a processor node
        props = dict(line.split(None, 1) for line in (host.read(f"{base}/{n}/properties") or "").splitlines()
                     if len(line.split(None, 1)) == 2)
        if int(props.get("location_id", -1)) == location and int(props.get("domain", 0)) == domain:
            return i
        i += 1
    return None
