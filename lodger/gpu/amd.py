"""AMD cards (amdgpu). Memory comes from the driver's sysfs files, per-app use from fdinfo. Both are safe
to read while the card sleeps."""
from .. import host
from . import fdinfo


def integrated(dev):
    """A processor's built-in graphics (APU) shares the PC's RAM. amdgpu shows mem_info_vram_vendor only on
    cards with their own memory chips; on APUs "VRAM" is just a small piece of RAM the BIOS set aside."""
    return not host.read(f"{dev.sys}/mem_info_vram_vendor")


def memory(dev):
    """{total, used, gtt_total, gtt_used} in bytes (GTT: system RAM the card may borrow), or None."""
    total = host.read_int(f"{dev.sys}/mem_info_vram_total")
    if not total:
        return None
    return {"total": total, "used": host.read_int(f"{dev.sys}/mem_info_vram_used"),
            "gtt_total": host.read_int(f"{dev.sys}/mem_info_gtt_total"),
            "gtt_used": host.read_int(f"{dev.sys}/mem_info_gtt_used")}


def why_unknown(dev):
    return None if memory(dev) else "amdgpu doesn't show this card's memory in sysfs"


def usage(dev, pids):
    return fdinfo.usage(dev.pci, pids)


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
