"""Graphics cards, for any vendor. Everything here reads /sys and /proc only and never opens /dev/dri or
/dev/nvidia*, so looking never wakes a sleeping card (NVIDIA's NVML is only called while the card is awake).

Adding a kind of card: write lodger/gpu/<vendor>.py with
  integrated(dev)     True for a processor's built-in graphics (shares the PC's RAM)
  memory(dev)         {total, used, gtt_total, gtt_used} in bytes, or None (unknown, or asleep: never wake it)
  why_unknown(dev)    plain words for why memory() is None
  usage(dev, pids)    {pid: {vram, gtt, busy_ns, cycles}} (for drivers with DRM fdinfo: fdinfo.usage, plus
                      a row in fdinfo.DRIVERS)
add its kernel driver to BACKENDS, and tell lodger/engine.py how llama.cpp finds the card."""
import re

from .. import host
from . import amd, generic, intel, nvidia

BACKENDS = {"amdgpu": amd, "nvidia": nvidia, "i915": intel, "xe": intel}
VENDORS = {0x1002: "AMD", 0x10de: "NVIDIA", 0x8086: "Intel"}
PCI_IDS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids")


def _hex(s):
    try:
        return int((s or "0").strip(), 16)
    except ValueError:
        return 0


class Device:
    """One graphics card (or built-in graphics), named by its PCI address."""

    def __init__(self, pci):
        self.pci = pci
        self.sys = f"/sys/bus/pci/devices/{pci}"
        self.vendor_id = _hex(host.read(f"{self.sys}/vendor"))
        self.device_id = _hex(host.read(f"{self.sys}/device"))
        self.vendor = VENDORS.get(self.vendor_id, f"vendor {self.vendor_id:04x}")
        self.driver = host.realname(f"{self.sys}/driver")
        self.backend = BACKENDS.get(self.driver, generic)
        cards = sorted((n for n in host.listdir(f"{self.sys}/drm") if re.fullmatch(r"card\d+", n)),
                       key=lambda n: int(n[4:]))
        self.card = cards[0] if cards else None
        self.boot_vga = host.read_int(f"{self.sys}/boot_vga") == 1
        self.integrated = self.backend.integrated(self)

    def __repr__(self):
        return f"<{self.vendor} {self.pci} {self.driver}{' integrated' if self.integrated else ''}>"

    @property
    def known(self):
        """lodger has a backend for this card's driver."""
        return self.backend is not generic

    def power(self):
        """The kernel's runtime power state: active, suspended, ... ('unsupported' when it never sleeps)."""
        return (host.read(f"{self.sys}/power/runtime_status") or "unsupported").strip()

    def asleep(self):
        return self.power() in ("suspended", "suspending")

    def memory(self):
        return self.backend.memory(self)

    def why_unknown(self):
        return self.backend.why_unknown(self)

    def usage(self, pids):
        return self.backend.usage(self, pids)

    def free(self, mem=None, meminfo=None):
        """Graphics memory free right now in bytes, or None if unknown. Built-in graphics share RAM: what
        it can still get is the smaller of RAM available and its GTT limit (the most RAM the driver lets the
        card map) minus what it has mapped."""
        m = self.memory() if mem is None else mem
        if m is None:
            return None
        if self.integrated:
            avail = (meminfo or host.meminfo()).get("MemAvailable", 0)
            return min(avail, m["gtt_total"] - m["gtt_used"]) if m.get("gtt_total") else avail
        return m["total"] - m["used"]

    def total(self, mem=None, meminfo=None):
        """The card's memory size (built-in graphics: its GTT limit, or all RAM), or None if unknown."""
        m = self.memory() if mem is None else mem
        if m is None:
            return None
        if self.integrated:
            return m.get("gtt_total") or (meminfo or host.meminfo()).get("MemTotal", 0)
        return m["total"]

    def outputs(self):
        """Connected screens, e.g. ['DP-1', 'HDMI-A-1']."""
        if not self.card:
            return []
        base = f"{self.sys}/drm/{self.card}"
        return sorted(n[len(self.card) + 1:] for n in host.listdir(base)
                      if n.startswith(self.card + "-") and (host.read(f"{base}/{n}/status") or "").strip() == "connected")

    def name(self):
        """Marketing name from the PCI ID list, if the system has one (reading a file; touches no card)."""
        vid, did = f"{self.vendor_id:04x}", f"{self.device_id:04x}"
        for path in PCI_IDS:
            try:
                with open(path, errors="replace") as f:
                    in_vendor = False
                    for line in f:
                        if line.startswith("#") or not line.strip():
                            continue
                        if not line.startswith("\t"):
                            in_vendor = line.startswith(vid + "  ")
                        elif in_vendor and line.startswith(f"\t{did}  "):
                            return line.strip()[len(did):].strip()
            except OSError:
                continue
        return None


def devices():
    """Every display controller on the PCI bus (PCI class 03xxxx), in address order."""
    out = []
    for pci in sorted(host.listdir("/sys/bus/pci/devices")):
        if (host.read(f"/sys/bus/pci/devices/{pci}/class") or "").strip().startswith("0x03"):
            out.append(Device(pci))
    return out


def normalize(pci):
    """'03:00.0' -> '0000:03:00.0'."""
    pci = (pci or "").strip().lower()
    return pci if pci.count(":") == 2 else "0000:" + pci


def choose(want="auto", devs=None):
    """The card lodger uses: want = 'auto', 'none' or a PCI address. Auto prefers a card with its own memory
    over built-in graphics, then the most memory (where sysfs tells without waking it), then the one the PC
    started on. Only cards lodger has a backend for. None: processor only."""
    devs = devices() if devs is None else devs
    want = (want or "auto").strip().lower()
    if want == "none":
        return None
    if want != "auto":
        return next((d for d in devs if d.pci == normalize(want)), None)
    cands = [d for d in devs if d.known]
    if not cands:
        return None
    return max(cands, key=lambda d: (not d.integrated, host.read_int(f"{d.sys}/mem_info_vram_total"), d.boot_vga))


def why_none(want="auto", devs=None):
    """Plain words for why choose() found no card."""
    devs = devices() if devs is None else devs
    want = (want or "auto").strip().lower()
    if want == "none":
        return "the settings say not to use a graphics card (gpu = \"none\")"
    if want != "auto":
        return f"no graphics card at {normalize(want)} (the settings' gpu = \"{want}\")"
    if not devs:
        return "no graphics card found"
    return "no graphics card lodger knows how to use (" + ", ".join(f"{d.pci}: driver {d.driver or 'none'}" for d in devs) + ")"


def power_source():
    """('ac' | 'battery', detail). Batteries of a mouse or headset (scope=Device) don't count. A PC without
    a battery is on mains power."""
    base = "/sys/class/power_supply"
    batteries, mains = [], []
    for n in host.listdir(base):
        kind = (host.read(f"{base}/{n}/type") or "").strip()
        if (host.read(f"{base}/{n}/scope") or "").strip().lower() == "device":
            continue
        if kind == "Battery":
            batteries.append((n, (host.read(f"{base}/{n}/status") or "").strip()))
        elif host.read_int(f"{base}/{n}/online") == 1:
            mains.append(n)
    if not batteries:
        return "ac", "no battery"
    if mains:
        return "ac", "plugged in (" + ", ".join(mains) + ")"
    if any(s == "Discharging" for _, s in batteries):
        return "battery", "on battery (" + ", ".join(f"{n} {s.lower()}" for n, s in batteries) + ")"
    return "ac", "battery " + ", ".join(s.lower() or "?" for _, s in batteries)
