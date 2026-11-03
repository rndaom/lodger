"""NVIDIA cards with NVIDIA's own driver. That driver has no fdinfo, so lodger asks NVML
(libnvidia-ml.so.1, part of the driver) through ctypes: free memory, memory per process, and how busy the
card is for each process. NVML wakes a sleeping card, so it is only called while the card is awake
(a sleeping card has nobody using it anyway), and an open NVML session holds the card's device files, which
can keep it from going to sleep: where the card may sleep (power/control "auto"), lodger closes the session
a few seconds after its last question, unless lodger's own model is on that card. If NVML can't be opened,
lodger tries again later (5 s, doubling up to 5 min) rather than giving up for good. Without NVML,
/proc/<pid>/fd links to /dev/nvidiaN still say which processes hold the card.

Untested on real NVIDIA hardware so far (tests use a fake NVML)."""
import ctypes
import os
import re
import threading
import time

from .. import host

NVML_SUCCESS = 0
NVML_ERROR_NOT_FOUND = 6
NVML_ERROR_INSUFFICIENT_SIZE = 7
NOT_AVAILABLE = (1 << 64) - 1


def integrated(dev):
    return False


reads_when_awake = True     # memory() is None while the card sleeps, not for good (see gpu/__init__.py)


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

    def shutdown(self):
        self.handles = {}
        self.lib.nvmlShutdown()

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
_nvml_error = None      # after a failure: (why, when to try again, the wait before that try)
_lock = threading.Lock()
_closer = None
LINGER = 3.0            # seconds an unused session stays open where the card may sleep
RETRY = (5.0, 300.0)    # first and longest wait before trying to open NVML again


def nvml():
    """The NVML connection, opened on first use (only ever called while the card is awake). After a failure
    it is tried again once the wait has passed, so a driver that was still loading doesn't stay unknown."""
    global _nvml, _nvml_error
    with _lock:
        if _nvml is None and (_nvml_error is None or time.monotonic() >= _nvml_error[1]):
            try:
                _nvml = NVML()
                _nvml_error = None
            except (OSError, AttributeError) as ex:
                wait = min(RETRY[1], _nvml_error[2] * 2) if _nvml_error else RETRY[0]
                _nvml_error = (str(ex) or "libnvidia-ml.so.1 not found", time.monotonic() + wait, wait)
        return _nvml


def _close():
    global _nvml
    with _lock:
        if isinstance(_nvml, NVML):
            try:
                _nvml.shutdown()
            except (OSError, AttributeError):
                pass
            _nvml = None


def may_sleep(dev):
    """The kernel may put this card to sleep (runtime power management on)."""
    return (host.read(f"{dev.sys}/power/control") or "").strip() == "auto"


def _done(dev):
    """After a question: where the card may sleep and lodger's model isn't on it, close the session soon."""
    global _closer
    if not may_sleep(dev):
        return
    from .. import fence
    if holders(dev, fence.our_pids()):
        return
    with _lock:
        if _closer:
            _closer.cancel()
        _closer = threading.Timer(LINGER, _close)
        _closer.daemon = True
        _closer.start()


def why_unknown(dev):
    if dev.asleep():
        return "the card is asleep (power saving); lodger doesn't wake it just to look"
    if nvml() is None:
        return f"NVML isn't available ({_nvml_error[0] if _nvml_error else 'unknown error'}; lodger tries again later)"
    return None


def memory(dev):
    """{total, used, gtt_total, gtt_used} or None (asleep, or no NVML)."""
    if dev.asleep() or nvml() is None:
        return None
    try:
        total, free, used = nvml().memory(dev.pci)
    except OSError:
        return None
    finally:
        _done(dev)
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
        now_us = int(time.time() * 1e6)
        st = _busy.setdefault(dev.pci, {"since_us": now_us - 1_000_000, "ns": {}, "seen": {}})
        if st["since_us"] > now_us + 1_000_000:
            # NVML stamps its samples with the wall clock: after the clock was set back, asking for samples newer
            # than the last one seen would return none until the clock caught up (blind for that long)
            st["since_us"], st["seen"] = now_us - 1_000_000, {}
        for pid, ts, sm in sorted(nvml().utilization(dev.pci, st["since_us"]), key=lambda s: s[1]):
            prev = st["seen"].get(pid, ts - 1_000_000 // 6)           # NVML samples about 6 times a second
            st["ns"][pid] = st["ns"].get(pid, 0) + int(sm / 100 * max(0, ts - prev) * 1000)
            st["seen"][pid] = ts
            st["since_us"] = max(st["since_us"], ts)
    except OSError:
        return {}
    finally:
        _done(dev)
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


# CUDA builds and the oldest Linux driver each needs (NVIDIA's CUDA toolkit release notes): CUDA 12.x runs
# on 525.60.13 and newer (minor version compatibility); CUDA 13.x needs 580 and newer, and dropped
# Maxwell, Pascal and Volta cards.
CUDA_DRIVER = {12: (525, 60, 13), 13: (580, 0, 0)}
_VERSION = re.compile(r"(?:Module|x86_64|aarch64|arm64)\s+(\d+)\.(\d+)(?:\.(\d+))?")


def driver_version():
    """NVIDIA's kernel driver version as a tuple, e.g. (550, 54, 14), from /proc/driver/nvidia/version
    ("NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.14  ..."; the open module says "... for x86_64"),
    or None."""
    first = (host.read("/proc/driver/nvidia/version") or "").split("\n", 1)[0]
    m = _VERSION.search(first)
    return tuple(int(x or 0) for x in m.groups()) if m else None


def cuda_ok(major, version=None):
    """Can this PC's NVIDIA driver run a CUDA `major` build? None when the driver version is unknown."""
    version = driver_version() if version is None else version
    if not version or major not in CUDA_DRIVER:
        return None
    return version >= CUDA_DRIVER[major]
