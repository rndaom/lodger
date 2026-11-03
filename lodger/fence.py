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
LODGER_FENCE=cgroup|none picks a weaker kind on purpose (to test it).

systemd only caps memory where the user manager has the memory controller (cgroup v2, delegated to it):
elsewhere MemoryMax is silently ignored. So each Fenced checks its unit's memory.max once it runs, and if
the cap isn't there the watchdog enforces it like for 'none' (Fenced.hard(), over_cap()).

The engine's environment is the same under every kind (engine.child_env: card-picking variables set or
removed, driver and runtime settings passed on from where lodger runs).

Left behind (fence.find_orphans, reap_orphans): a model outlives lodger if lodger is killed or crashes (a
systemd unit runs on; cgroup and 'none' programs run in their own session). Units and cgroups are named
lodger-<pid> or lodger-<pid>-<n> after the lodger process that made them, and cgroup/'none' programs carry
LODGER_FENCE_NAME, so a model whose lodger is gone can be found and stopped: `lodger serve` and `lodger run`
should call reap_orphans() before they start a model; `lodger doctor` lists them. Where util-linux's setpriv
can, cgroup/'none' programs also get a parent-death signal: the kernel kills them when lodger dies. That
signal follows the *thread* that started the program, and lodger serve starts engines from threads that end,
so every fenced program is started from one spawner thread that lives as long as lodger."""
import contextlib
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time

from . import engine, host

CGROOT = "/sys/fs/cgroup"
KINDS = ("systemd", "cgroup", "none")
# the start of every program in a cgroup/none fence: join the cgroup (refuse to run if that fails), be the
# first the kernel stops when memory runs out, no crash dumps, low priority
WRAP = ('cg="$1"; shift; if [ -n "$cg" ]; then echo $$ > "$cg/cgroup.procs" || exit 70; fi; '
        'echo 1000 > /proc/self/oom_score_adj 2>/dev/null; ulimit -c 0; exec nice -n 10 "$@"')

SYSTEMCTL_WAIT = 5          # seconds: a systemd that hasn't answered by then counts as failed (it may hang)
ASK_AGAIN = 30              # seconds before a systemd that didn't answer is asked again
_mode = None
_live = set()
_count = 0
_lock = threading.Lock()
NAME = re.compile(r"^lodger-(\d+)(?:-\d+)?$")       # a fence lodger made: lodger-<pid> or lodger-<pid>-<n>


def _systemd_user():
    if not shutil.which("systemd-run"):
        return False
    try:
        return subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _systemctl(*args):
    """systemctl --user ARGS (output captured), or None if it can't start or doesn't answer in time. Called
    from the watchdog: it must never wait forever."""
    try:
        return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=SYSTEMCTL_WAIT)
    except (OSError, subprocess.TimeoutExpired):
        return None


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


def _manager():
    """The systemd user manager's cgroup folder: its units' cgroups are under it."""
    uid = os.getuid()
    m = re.match(rf"^(.*/user@{uid}\.service)(?:/|$)", own_cgroup() or "")
    return CGROOT + m.group(1) if m else f"{CGROOT}/user.slice/user-{uid}.slice/user@{uid}.service"


def systemd_memory():
    """(True, None) if the systemd user manager can cap its services' memory (it has the memory controller),
    else (False, why). Each unit is checked again once it runs (Fenced.hard)."""
    if not os.path.exists(host.path(CGROOT + "/cgroup.controllers")):
        return False, "cgroup v1 or hybrid: the systemd user manager can't cap memory"
    ctl = host.read(_manager() + "/cgroup.controllers")
    if ctl is None:
        return True, None                   # an unusual layout: each unit's own check decides
    if "memory" not in ctl.split():
        return False, "the memory controller isn't delegated to the systemd user manager"
    return True, None


def mode():
    """(kind, plain words). Decided once per process."""
    global _mode
    if _mode is None:
        forced = os.environ.get("LODGER_FENCE")
        if forced and forced not in KINDS:
            raise SystemExit(f"LODGER_FENCE must be one of {', '.join(KINDS)}")
        if forced in (None, "systemd") and _systemd_user():
            capped, why = systemd_memory()
            _mode = ("systemd", "systemd user session" if capped else
                     f"systemd user session, but NO HARD MEMORY CAP ({why}): the watchdog stops the model over "
                     "its RAM cap, and the kernel stops it first if memory runs out")
        else:
            d, why = delegated() if forced in (None, "systemd", "cgroup") else (None, "LODGER_FENCE=none")
            if d:
                _mode = ("cgroup", f"own cgroups under {d} ({'LODGER_FENCE=cgroup' if forced == 'cgroup' else 'no systemd user session'})")
            else:
                _mode = ("none", f"NO HARD MEMORY FENCE ({why}): the watchdog stops the model over its RAM cap, "
                                 "and the kernel stops it first if memory runs out")
    return _mode


def hard_cap():
    """Does this PC's fence kind make the kernel hold the model to its RAM cap? (Per program: Fenced.hard.)"""
    kind, _ = mode()
    return kind == "cgroup" or (kind == "systemd" and systemd_memory()[0])


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


_pdeath = None
_jobs = queue.Queue()
_spawner = None


def pdeathsig():
    """['setpriv', '--pdeathsig', 'KILL'] where util-linux's setpriv has it (2.33 and newer), else []."""
    global _pdeath
    if _pdeath is None:
        _pdeath = []
        if shutil.which("setpriv"):
            try:
                r = subprocess.run(["setpriv", "--help"], capture_output=True, text=True, timeout=10)
                if "--pdeathsig" in r.stdout:
                    _pdeath = ["setpriv", "--pdeathsig", "KILL"]
            except (OSError, subprocess.TimeoutExpired):
                pass
    return _pdeath


def _spawn_loop():
    while True:
        fn, box, done = _jobs.get()
        try:
            box["value"] = fn()
        except BaseException as ex:            # handed back to the caller's thread
            box["error"] = ex
        done.set()


def _spawn(fn):
    """Run fn (it starts a program) on the spawner thread, which lives as long as lodger (see the top)."""
    global _spawner
    with _lock:
        if _spawner is None or not _spawner.is_alive():
            _spawner = threading.Thread(target=_spawn_loop, name="lodger-spawner", daemon=True)
            _spawner.start()
    box, done = {}, threading.Event()
    _jobs.put((fn, box, done))
    done.wait()
    if "error" in box:
        raise box["error"]
    return box["value"]


class Fenced:
    """One program inside a fence. props are systemd unit properties (MemoryMax, MemorySwapMax, LimitCORE,
    CPUWeight, IOWeight, Nice, PrivateDevices, RuntimeMaxSec); the other kinds apply what they can.
    env: what lodger sets for it (see engine.child_env for the rest of its environment)."""
    _ask_after = 0.0            # monotonic time before which systemd isn't asked for the unit's cgroup again

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
        self._hard = True if self.kind == "cgroup" else (False if self.kind == "none" else None)
        self.hard_why = None if self.kind == "cgroup" else ("no hard fence" if self.kind == "none" else "not checked yet")
        put, drop = engine.child_env(env)
        if self.kind == "systemd":
            sr = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={self.name}", "--pty" if pty else "--pipe", "--wait"]
            sr += [f"--property={k}={v}" for k, v in self.props.items()]
            sr += [f"--setenv={k}={v}" for k, v in put.items()]
            if drop:
                sr.append("--property=UnsetEnvironment=" + " ".join(drop))
            full, penv = sr + list(cmd), None
        else:
            cgpath = self._make_cgroup() if self.kind == "cgroup" else ""
            full = pdeathsig() + ["sh", "-c", WRAP, "lodger-fence", cgpath or ""] + list(cmd)
            penv = {k: v for k, v in os.environ.items() if k not in drop}
            penv.update(put)
            penv["LODGER_FENCE_NAME"] = self.name   # how find_orphans knows it if lodger dies
            new_session = True                      # its own process group: how 'none' finds and stops it
        self.proc = _spawn(lambda: subprocess.Popen(full, stdin=stdin, stdout=stdout, stderr=stderr, env=penv,
                                                    start_new_session=new_session))
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
        """The cgroup folder, or None ('none' kind, or a systemd unit that is gone). A unit's is looked for where
        systemd-run puts it; systemd is asked only where it isn't there (an unusual layout), and a systemd
        that didn't answer isn't asked again for a while: the watchdog calls this every tick."""
        if self.kind == "systemd" and not self.cg:
            d = f"{_manager()}/app.slice/{self.name}.service"
            if os.path.isdir(d):
                self.cg = d
            elif time.monotonic() >= self._ask_after:
                r = _systemctl("show", "-p", "ControlGroup", "--value", self.name + ".service")
                if r is None:
                    self._ask_after = time.monotonic() + ASK_AGAIN
                cg = r.stdout.strip() if r and r.returncode == 0 else ""
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
        """(current, peak) RAM of everything in the fence, bytes (cgroup numbers: page cache counts too;
        where the cgroup has no memory controller, the processes' resident memory)."""
        cg = self.cgroup()
        cur = _read_int(cg + "/memory.current") if cg else None
        if cur is not None:
            peak = _read_int(cg + "/memory.peak") or cur
            return cur, peak
        cur = sum(_rss(p) for p in self.pids())
        self.peak = max(self.peak, cur)
        return cur, self.peak

    def hard(self):
        """Does the kernel hold this program to its RAM cap? cgroup kind: yes (lodger wrote memory.max and
        refuses to run without it); none: no; systemd: its unit's memory.max is read once it runs (until then,
        no). self.hard_why says why not."""
        if self._hard is not None:
            return self._hard
        cg = self.cgroup()
        if not cg:
            return False
        mx = (_text(cg + "/memory.max") or "").strip()
        if mx.isdigit() and self.cap and 0 < int(mx) <= self.cap:
            self._hard, self.hard_why = True, None
            swap = _text(cg + "/memory.swap.max")
            if "MemorySwapMax" in self.props and swap is not None and swap.strip() != "0":
                self.hard_why = "RAM capped, but swap isn't (memory.swap.max " + swap.strip() + ")"
        elif self.cap is None:
            self._hard, self.hard_why = True, None
        elif self.pids():                       # it runs, so systemd has set what it could
            self._hard = False
            self.hard_why = (f"systemd didn't apply MemoryMax (memory.max is {mx or 'missing'}): "
                             + (systemd_memory()[1] or "no memory controller for this unit"))
        return bool(self._hard)

    def own_memory(self):
        """RAM only the program holds (not model file pages, which the kernel can drop and read again), bytes."""
        cg = self.cgroup()
        stat = _text(cg + "/memory.stat") if cg else None
        if stat:
            v = {k: int(n) for k, n in (l.split() for l in stat.splitlines() if l.count(" ") == 1) if n.isdigit()}
            if "anon" in v:
                return v["anon"] + v.get("shmem", 0)
        return sum(_anon(p) for p in self.pids())

    def over_cap(self):
        """Over its RAM cap where the kernel doesn't enforce it ('none', or a systemd unit whose cap didn't
        apply): the watchdog enforces it from this. Like the kernel would, it counts the program's own memory:
        weights read from the model file are page cache, given back under pressure (a "slow" plan relies on it)."""
        return self.cap is not None and not self.hard() and self.own_memory() > self.cap

    def gone(self):
        return self.proc.poll() is not None and not self.pids()

    def _cgroup_kill(self):
        """SIGKILL everything in the fence's cgroup with one write to cgroup.kill (kernel 5.14+). True if done."""
        cg = self.cgroup() if self.kind != "none" else None
        if not cg or not os.path.exists(cg + "/cgroup.kill"):
            return False
        try:
            with open(cg + "/cgroup.kill", "w") as f:
                f.write("1")
            return True
        except OSError:
            return False

    def kill(self, sig=signal.SIGKILL):
        """Signal everything in the fence. Never through systemctl: a systemd round trip can take seconds (or
        hang) exactly when memory runs low and the watchdog needs the model gone. SIGKILL goes through
        cgroup.kill where it can, else to each process in the fence's cgroup."""
        if sig == signal.SIGKILL and self._cgroup_kill():
            return
        pids = self.pids()
        if not pids and self.kind != "systemd" and self.proc.poll() is None:
            pids = [self.proc.pid]              # (a systemd unit's client isn't the model: killing it does nothing)
        for pid in pids:
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
            _systemctl("stop", self.name + ".service")         # in case it outlived its client
        else:
            t0 = time.monotonic()
            while self.pids() and time.monotonic() - t0 < 10:
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
            t0 = time.monotonic()
            while time.monotonic() - t0 < 5:
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
    return _read_int(path) or 0


def _text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _read_int(path):
    """The first number in a file, or None if it isn't there."""
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


# ---------- models lodger left behind ----------

def owner_alive(pid):
    """Is the lodger process that made fence lodger-<pid>-... still running? (A reused pid running something
    else doesn't count.)"""
    if pid == os.getpid():
        return True
    cmd = host.read(f"/proc/{pid}/cmdline")
    if cmd is None:
        return False
    args = [a for a in cmd.split("\0") if a][:4]
    return any(os.path.basename(a) == "lodger" or a == "lodger.cli" for a in args)


def named():
    """Programs started in a cgroup/'none' fence by any lodger, found by the LODGER_FENCE_NAME they carry (they have
    no lodger-* systemd unit or cgroup to tell them by): {fence name: [pids]}. Only this user's are readable."""
    procs = {}
    for p in host.listdir("/proc"):
        if p.isdigit():
            name = fence_name(p)
            if name:
                procs.setdefault(name, []).append(int(p))
    return procs


def fence_name(pid):
    """The lodger fence a process was started in, from the LODGER_FENCE_NAME it carries, or None."""
    env = host.read(f"/proc/{pid}/environ") or ""
    if "LODGER_FENCE_NAME=" not in env:
        return None
    for var in env.split("\0"):
        if var.startswith("LODGER_FENCE_NAME="):
            name = var[len("LODGER_FENCE_NAME="):]
            return name if NAME.match(name) else None
    return None


def find_orphans():
    """Fences whose lodger is gone, with the model maybe still in them: [{kind, name, owner, where}]. Reads
    only (systemctl list-units, the delegated cgroup folder, /proc)."""
    out, seen = [], set()
    if shutil.which("systemctl"):
        try:
            r = subprocess.run(["systemctl", "--user", "list-units", "--all", "--plain", "--no-legend", "--full",
                                "lodger-*.service"], capture_output=True, text=True, timeout=10)
            lines = r.stdout.splitlines() if r.returncode == 0 else []
        except (OSError, subprocess.TimeoutExpired):
            lines = []
        for line in lines:
            parts = [p for p in line.split() if p not in ("●", "*")]
            if len(parts) < 3 or not parts[0].endswith(".service"):
                continue
            unit, active = parts[0], parts[2]
            m = NAME.match(unit[:-len(".service")])        # never lodger.service itself
            if m and active in ("active", "activating", "deactivating", "reloading") and not owner_alive(int(m.group(1))):
                out.append({"kind": "systemd", "name": unit[:-len(".service")], "owner": int(m.group(1)), "where": unit})
                seen.add(unit[:-len(".service")])
    parent, _ = delegated()
    for n in (os.listdir(parent) if parent and os.path.isdir(parent) else []):
        m = NAME.match(n)
        if m and os.path.isdir(os.path.join(parent, n)) and not owner_alive(int(m.group(1))):
            out.append({"kind": "cgroup", "name": n, "owner": int(m.group(1)), "where": os.path.join(parent, n)})
            seen.add(n)
    for name, pids in sorted(named().items()):
        m = NAME.match(name)
        if m and name not in seen and not owner_alive(int(m.group(1))):
            out.append({"kind": "process", "name": name, "owner": int(m.group(1)), "where": sorted(pids)})
    return out


def reap_orphans(log=None):
    """Stop every model a lodger that is gone left running (see find_orphans); returns plain words for each.
    `lodger serve` and `lodger run` should call this before starting a model: an orphan holds memory and
    has no watchdog. log(text) is called for each one stopped."""
    done = []
    for o in find_orphans():
        if o["kind"] == "systemd":
            unit = o["where"]
            for cmd in (["kill", "-s", "KILL", unit], ["stop", unit]):
                try:
                    subprocess.run(["systemctl", "--user"] + cmd, capture_output=True, timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        elif o["kind"] == "cgroup":
            _kill_cgroup(o["where"])
        else:
            for pid in o["where"]:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
        msg = f"stopped a model lodger {o['owner']} left running ({o['kind']} {o['name']})"
        done.append(msg)
        if log:
            log(msg)
    return done


def _kill_cgroup(d):
    """Kill everything in a lodger-* cgroup and remove it."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5:
        pids = [int(x) for x in (_text(d + "/cgroup.procs") or "").split()]
        if not pids:
            break
        if os.path.exists(d + "/cgroup.kill"):
            with contextlib.suppress(OSError), open(d + "/cgroup.kill", "w") as f:
                f.write("1")
        else:
            for pid in pids:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)
    with contextlib.suppress(OSError):
        os.rmdir(d)


def _anon(pid):
    n = 0
    for line in (host.read(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith(("RssAnon:", "RssShmem:")):
            n += int(line.split()[1]) * 1024
    return n


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
