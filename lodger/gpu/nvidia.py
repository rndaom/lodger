"""NVIDIA cards with NVIDIA's own driver. That driver has no fdinfo, so lodger asks NVML
(libnvidia-ml.so.1, part of the driver) through ctypes: free memory, memory per process, and how busy the
card is for each process. NVML wakes a sleeping card, so it is only called while the card is awake
(a sleeping card has nobody using it anyway). Without NVML, /proc/<pid>/fd links to /dev/nvidiaN still say
which processes hold the card.

Untested on real NVIDIA hardware so far (tests use a fake NVML)."""
import ctypes
import os
import threading
import time

from .. import host

NVML_SUCCESS = 0
NVML_ERROR_NOT_FOUND = 6
NVML_ERROR_INSUFFICIENT_SIZE = 7
NOT_AVAILABLE = (1 << 64) - 1


def integrated(dev):
    return False


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _Process(ctypes.Structure):        # nvmlProcessInfo_t, as the *_v3 calls fill it
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class _Sample(ctypes.Structure):         # nvmlProcessUtilizationSample_t
    _fields_ = [("pid", ctypes.c_uint), ("timeStamp", ctypes.c_ulonglong), ("smUtil", ctypes.c_uint),
                ("memUtil", ctypes.c_uint), ("encUtil", ctypes.c_uint), ("decUtil", ctypes.c_uint)]


class NVML:
    """The few NVML calls lodger needs. Raises OSError when the library or a call fails."""

    def __init__(self, libname="libnvidia-ml.so.1"):
        self.lib = ctypes.CDLL(libname)
        self.handles = {}
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")

    def _check(self, rc, what):
        if rc != NVML_SUCCESS:
            raise OSError(f"NVML {what} failed (error {rc})")

    def _handle(self, pci):
        if pci not in self.handles:
            h = ctypes.c_void_p()
            self._check(self.lib.nvmlDeviceGetHandleByPciBusId_v2(pci.encode(), ctypes.byref(h)), "GetHandleByPciBusId")
            self.handles[pci] = h
        return self.handles[pci]

    def memory(self, pci):
        """(total, free, used) bytes."""
        m = _Memory()
        self._check(self.lib.nvmlDeviceGetMemoryInfo(self._handle(pci), ctypes.byref(m)), "GetMemoryInfo")
        return m.total, m.free, m.used

    def processes(self, pci):
        """{pid: graphics memory bytes} for compute and graphics processes."""
        out = {}
        for fn in ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses_v3"):
            call = getattr(self.lib, fn)
            n = ctypes.c_uint(0)
            rc = call(self._handle(pci), ctypes.byref(n), None)
            if rc == NVML_SUCCESS:
                continue
            if rc != NVML_ERROR_INSUFFICIENT_SIZE:
                self._check(rc, fn)
            n = ctypes.c_uint(n.value + 8)
            buf = (_Process * n.value)()
            self._check(call(self._handle(pci), ctypes.byref(n), buf), fn)
            for p in buf[:n.value]:
                used = 0 if p.usedGpuMemory == NOT_AVAILABLE else p.usedGpuMemory
                out[p.pid] = out.get(p.pid, 0) + used
        return out

    def utilization(self, pci, since_us):
        """[(pid, timestamp µs, 3D/compute busy %)] for samples newer than since_us."""
        n = ctypes.c_uint(0)
        rc = self.lib.nvmlDeviceGetProcessUtilization(self._handle(pci), None, ctypes.byref(n), ctypes.c_ulonglong(since_us))
        if rc in (NVML_SUCCESS, NVML_ERROR_NOT_FOUND) or not n.value:
            return []
        if rc != NVML_ERROR_INSUFFICIENT_SIZE:
            self._check(rc, "GetProcessUtilization")
        buf = (_Sample * n.value)()
        rc = self.lib.nvmlDeviceGetProcessUtilization(self._handle(pci), buf, ctypes.byref(n), ctypes.c_ulonglong(since_us))
        if rc == NVML_ERROR_NOT_FOUND:
            return []
        self._check(rc, "GetProcessUtilization")
        return [(s.pid, s.timeStamp, s.smUtil) for s in buf[:n.value]]


_nvml = None            # tests put a fake here
_nvml_error = None
_lock = threading.Lock()


def nvml():
    """The NVML connection, opened on first use (only ever called while the card is awake)."""
    global _nvml, _nvml_error
    with _lock:
        if _nvml is None and _nvml_error is None:
            try:
                _nvml = NVML()
            except (OSError, AttributeError) as ex:
                _nvml_error = str(ex) or "libnvidia-ml.so.1 not found"
        return _nvml


def why_unknown(dev):
    if dev.asleep():
        return "the card is asleep (power saving); lodger doesn't wake it just to look"
    if nvml() is None:
        return f"NVML isn't available ({_nvml_error})"
    return None


def memory(dev):
    """{total, used, gtt_total, gtt_used} or None (asleep, or no NVML)."""
    if dev.asleep() or nvml() is None:
        return None
    try:
        total, free, used = nvml().memory(dev.pci)
    except OSError:
        return None
    return {"total": total, "used": total - free, "gtt_total": None, "gtt_used": None}


_busy = {}              # pci -> {"since_us": last sample seen, "ns": {pid: busy ns so far}}


def usage(dev, pids):
    """{pid: {vram, gtt, busy_ns, cycles}} for the given processes. NVML gives busy as a share per sample;
    lodger adds those up into a running busy time, like the kernel's fdinfo counters."""
    if dev.asleep() or nvml() is None:
        return {}
    want = set(pids)
    try:
        procs = nvml().processes(dev.pci)
        st = _busy.setdefault(dev.pci, {"since_us": int(time.time() * 1e6) - 1_000_000, "ns": {}, "seen": {}})
        for pid, ts, sm in sorted(nvml().utilization(dev.pci, st["since_us"]), key=lambda s: s[1]):
            prev = st["seen"].get(pid, ts - 1_000_000 // 6)           # NVML samples about 6 times a second
            st["ns"][pid] = st["ns"].get(pid, 0) + int(sm / 100 * max(0, ts - prev) * 1000)
            st["seen"][pid] = ts
            st["since_us"] = max(st["since_us"], ts)
    except OSError:
        return {}
    out = {}
    for pid in want & (set(procs) | set(st["ns"])):
        out[pid] = {"vram": procs.get(pid, 0), "gtt": 0, "busy_ns": st["ns"].get(pid, 0), "cycles": {}}
    return out


def minor(dev):
    """N in /dev/nvidiaN for this card (from the driver's /proc file; doesn't wake the card)."""
    for line in (host.read(f"/proc/driver/nvidia/gpus/{dev.pci}/information") or "").splitlines():
        if line.startswith("Device Minor:"):
            return line.split(":", 1)[1].strip()
    return None


def holders(dev, pids):
    """Processes with /dev/nvidiaN open: the cheap check that needs no NVML."""
    n = minor(dev)
    if n is None:
        return []
    target = f"/dev/nvidia{n}"
    out = []
    for pid in pids:
        for fd in host.listdir(f"/proc/{pid}/fd"):
            try:
                if os.readlink(host.path(f"/proc/{pid}/fd/{fd}")) == target:
                    out.append(pid)
                    break
            except OSError:
                pass
    return out


def cuda_index(dev, devices):
    """CUDA's number for this card with CUDA_DEVICE_ORDER=PCI_BUS_ID: its place among NVIDIA cards by PCI address."""
    ours = sorted(d.pci for d in devices if d.driver == "nvidia")
    return ours.index(dev.pci) if dev.pci in ours else None
