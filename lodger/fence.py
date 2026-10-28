"""The fence around a model: a RAM cap with no swap, no crash dumps (a dumping process keeps its graphics
memory for seconds), low processor and disk priority, and on the processor only, no device files at all.

Three kinds, the strongest the PC offers:
- systemd: a transient unit of the systemd user session (systemd-run --user). The usual case.
- cgroup:  no systemd user session, but a cgroup v2 folder this user may write to (a delegated subtree):
           lodger makes a lodger-* cgroup there with the same memory cap (and CPU/IO weights where the
           controllers are on). No PrivateDevices: the cards are hidden through the engine's environment only.
- none:    neither: no hard cap. The program gets oom_score_adj 1000 (the kernel stops it first when memory
           runs out), no crash dumps and nice 10, and the watchdog stops it when it goes over its RAM cap.
           lodger says so when it starts. (Never RLIMIT_AS: it breaks memory-mapped models and GPU runtimes.)
LODGER_FENCE=cgroup|none picks a weaker kind on purpose (to test it)."""
import os
import shutil
import signal
import subprocess
import threading
import time

from . import host

CGROOT = "/sys/fs/cgroup"
KINDS = ("systemd", "cgroup", "none")
# the start of every program in a cgroup/none fence: join the cgroup (refuse to run if that fails), be the
# first the kernel stops when memory runs out, no crash dumps, low priority
WRAP = ('cg="$1"; shift; if [ -n "$cg" ]; then echo $$ > "$cg/cgroup.procs" || exit 70; fi; '
        'echo 1000 > /proc/self/oom_score_adj 2>/dev/null; ulimit -c 0; exec nice -n 10 "$@"')

_mode = None
_live = set()
_count = 0
_lock = threading.Lock()


def _systemd_user():
    if not shutil.which("systemd-run"):
        return False
    try:
        return subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def own_cgroup():
    for line in (host.read("/proc/self/cgroup") or "").splitlines():
        if line.startswith("0::"):
            return line[3:].strip()
    return None


def delegated():
    """(a cgroup v2 folder lodger may make its own cgroups in, or None, why not)."""
    if not os.path.exists(CGROOT + "/cgroup.controllers"):
        return None, "no cgroup v2"
    own = own_cgroup()
    if own is None:
        return None, "this process isn't in a cgroup v2"
    parts = [p for p in own.split("/") if p]
    # our own cgroup holds processes (us), so it can't hand controllers to children: start at its parent
    for n in range(len(parts) - 1, -1, -1):
        d = os.path.join(CGROOT, *parts[:n])
        if not all(os.access(os.path.join(d, f), os.W_OK) for f in ("cgroup.procs", "cgroup.subtree_control")):
            continue
        if not os.access(d, os.W_OK):
            continue
        if "memory" in (host.read(d + "/cgroup.controllers", "") or "").split():
            return d, None
    return None, "no cgroup this user may write to with the memory controller (not delegated)"


def mode():
    """(kind, plain words). Decided once per process."""
    global _mode
    if _mode is None:
        forced = os.environ.get("LODGER_FENCE")
        if forced and forced not in KINDS:
            raise SystemExit(f"LODGER_FENCE must be one of {', '.join(KINDS)}")
        if forced in (None, "systemd") and _systemd_user():
            _mode = ("systemd", "systemd user session")
        else:
            d, why = delegated() if forced in (None, "systemd", "cgroup") else (None, "LODGER_FENCE=none")
            if d:
                _mode = ("cgroup", f"own cgroups under {d} ({'LODGER_FENCE=cgroup' if forced == 'cgroup' else 'no systemd user session'})")
            else:
                _mode = ("none", f"NO HARD MEMORY FENCE ({why}): the watchdog stops the model over its RAM cap, "
                                 "and the kernel stops it first if memory runs out")
    return _mode


def our_pids():
    """Every process inside a fence lodger runs now (so lodger never mistakes its model for another app)."""
    with _lock:
        live = list(_live)
    out = set()
    for f in live:
        out.update(f.pids())
    return out


def _size(v):
    return str(int(v)) if str(v).isdigit() else str(v)


class Fenced:
    """One program inside a fence. props are systemd unit properties (MemoryMax, MemorySwapMax, LimitCORE,
    CPUWeight, IOWeight, Nice, PrivateDevices, RuntimeMaxSec); the other kinds apply what they can."""

    def __init__(self, cmd, env, props, stdin=subprocess.DEVNULL, stdout=None, stderr=None, pty=False,
                 new_session=False, name=None):
        global _count
        with _lock:
            _count += 1
            n = _count
        self.kind, self.why = mode()
        self.name = name or f"lodger-{os.getpid()}-{n}"
        self.props = dict(props)
        self.cap = int(self.props["MemoryMax"]) if str(self.props.get("MemoryMax", "")).isdigit() else None
        self.cg = None
        self.peak = 0
        self._timer = None
        if self.kind == "systemd":
            sr = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={self.name}", "--pty" if pty else "--pipe", "--wait"]
            sr += [f"--property={k}={v}" for k, v in self.props.items()]
            sr += [f"--setenv={k}={v}" for k, v in env.items()]
            full, penv = sr + list(cmd), None
        else:
            cgpath = self._make_cgroup() if self.kind == "cgroup" else ""
            full = ["sh", "-c", WRAP, "lodger-fence", cgpath or ""] + list(cmd)
            penv = dict(os.environ, **env)
            new_session = True                      # its own process group: how 'none' finds and stops it
        self.proc = subprocess.Popen(full, stdin=stdin, stdout=stdout, stderr=stderr, env=penv,
                                     start_new_session=new_session)
        if self.kind != "systemd" and self.props.get("RuntimeMaxSec"):
            self._timer = threading.Timer(float(self.props["RuntimeMaxSec"]), self.kill, (signal.SIGKILL,))
            self._timer.daemon = True
            self._timer.start()
        with _lock:
            _live.add(self)

    # ---- cgroup kind ----
    def _make_cgroup(self):
        parent, why = delegated()
        if not parent:
            raise SystemExit(f"can't fence the model: {why}")
        ctl = (host.read(parent + "/cgroup.subtree_control", "") or "").split()
        if "memory" not in ctl:
            with open(parent + "/cgroup.subtree_control", "w") as f:
                f.write("+memory")
        d = os.path.join(parent, self.name)
        os.mkdir(d)
        limits = {"memory.max": self.props.get("MemoryMax"), "memory.swap.max": "0" if "MemorySwapMax" in self.props else None,
                  "cpu.weight": self.props.get("CPUWeight"), "io.weight": self.props.get("IOWeight")}
        for k, v in limits.items():
            if v is not None and os.path.exists(os.path.join(d, k)):
                with open(os.path.join(d, k), "w") as f:
                    f.write(_size(v))
        if self.cap and not os.path.exists(os.path.join(d, "memory.max")):
            os.rmdir(d)
            raise SystemExit("can't fence the model: the memory controller isn't on in " + parent)
        self.cg = d
        return d

    # ---- all kinds ----
    def cgroup(self):
        """The cgroup folder, or None ('none' kind, or a systemd unit that is gone)."""
        if self.kind == "systemd" and not self.cg:
            r = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", self.name + ".service"],
                               capture_output=True, text=True)
            cg = r.stdout.strip()
            self.cg = CGROOT + cg if cg else None
        return self.cg

    def pids(self):
        cg = self.cgroup()
        if cg:
            try:
                with open(cg + "/cgroup.procs") as f:
                    return [int(x) for x in f.read().split()]
            except OSError:
                return []
        if self.kind == "none" and self.proc.poll() is None:
            return _group(self.proc.pid)
        return []

    def memory(self):
        """(current, peak) RAM of everything in the fence, bytes (cgroup numbers: page cache counts too)."""
        cg = self.cgroup()
        if cg:
            cur = _first_int(cg + "/memory.current")
            peak = _first_int(cg + "/memory.peak") or cur
            return cur, peak
        cur = sum(_rss(p) for p in self.pids())
        self.peak = max(self.peak, cur)
        return cur, self.peak

    def over_cap(self):
        """Only the 'none' kind has no kernel cap: the watchdog enforces it from this."""
        return self.kind == "none" and self.cap is not None and self.memory()[0] > self.cap

    def gone(self):
        return self.proc.poll() is not None and not self.pids()

    def kill(self, sig=signal.SIGKILL):
        if self.kind == "systemd":
            subprocess.run(["systemctl", "--user", "kill", "-s", signal.Signals(sig).name[3:], self.name + ".service"],
                           capture_output=True)
            return
        if self.kind == "cgroup" and sig == signal.SIGKILL and self.cg and os.path.exists(self.cg + "/cgroup.kill"):
            try:
                with open(self.cg + "/cgroup.kill", "w") as f:
                    f.write("1")
                return
            except OSError:
                pass
        for pid in self.pids() or ([self.proc.pid] if self.proc.poll() is None else []):
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    def wait(self, timeout=None):
        return self.proc.wait(timeout)

    def stop(self):
        """TERM, then KILL after 10 s; returns once everything in the fence is gone."""
        self.kill(signal.SIGTERM)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.kill(signal.SIGKILL)
            self.proc.wait(timeout=10)
        if self.kind == "systemd":
            subprocess.run(["systemctl", "--user", "stop", self.name + ".service"], capture_output=True)  # in case it outlived its client
        else:
            t0 = time.time()
            while self.pids() and time.time() - t0 < 10:
                self.kill(signal.SIGKILL)
                time.sleep(0.1)
        self.close()

    def close(self):
        """Forget the fence once its program has ended (removes a cgroup lodger made)."""
        if self._timer:
            self._timer.cancel()
        with _lock:
            _live.discard(self)
        if self.kind == "cgroup" and self.cg:
            t0 = time.time()
            while time.time() - t0 < 5:
                try:
                    os.rmdir(self.cg)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    time.sleep(0.1)


def run(cmd, env, props, timeout=300):
    """Run a short program in a fence and capture its output (like subprocess.run(capture_output=True, text=True))."""
    f = Fenced(cmd, env, props, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = f.proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        f.kill()
        out, err = f.proc.communicate()
    finally:
        if f.pids():
            f.stop()
        else:
            f.close()
    return subprocess.CompletedProcess(f.proc.args, f.proc.returncode, out.decode(errors="replace"), err.decode(errors="replace"))


def _first_int(path):
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0


def _rss(pid):
    for line in (host.read(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _group(pgid):
    """Processes in process group `pgid` (a 'none' fence's program and everything it started)."""
    out = []
    for p in host.listdir("/proc"):
        if p.isdigit():
            s = host.read(f"/proc/{p}/stat") or ""
            try:
                if int(s.rsplit(")", 1)[1].split()[3]) == pgid:
                    out.append(int(p))
            except (IndexError, ValueError):
                pass
    return out
