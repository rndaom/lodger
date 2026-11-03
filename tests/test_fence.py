"""The fence without systemd: the 'none' kind runs real (harmless) programs; no graphics card involved."""
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from lodger import fence
from tests.support import machine, write

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEAD = 4194305                  # above any pid_max: no process has it


def zombie_or_gone(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


class FenceTest(unittest.TestCase):
    def setUp(self):
        fence._mode = None

    def tearDown(self):
        fence._mode = None

    def test_forced_kinds(self):
        with mock.patch.dict(os.environ, {"LODGER_FENCE": "none"}):
            self.assertEqual(fence.mode()[0], "none")
            self.assertIn("NO HARD MEMORY FENCE", fence.mode()[1])
        fence._mode = None
        with mock.patch.dict(os.environ, {"LODGER_FENCE": "bogus"}), self.assertRaises(SystemExit):
            fence.mode()

    def test_no_systemd_and_no_delegation_means_none(self):
        with mock.patch.dict(os.environ, {"LODGER_FENCE": ""}), \
                mock.patch("lodger.fence._systemd_user", return_value=False), \
                mock.patch("lodger.fence.delegated", return_value=(None, "not delegated")):
            self.assertEqual(fence.mode()[0], "none")

    def test_none_kind_runs_and_applies_what_it_can(self):
        with mock.patch.dict(os.environ, {"LODGER_FENCE": "none"}):
            r = fence.run(["sh", "-c", "cat /proc/self/oom_score_adj; ulimit -c; echo $X"], {"X": "hello"},
                          {"MemoryMax": "1000000000", "LimitCORE": "0"})
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.split(), ["1000", "0", "hello"])

    def test_none_kind_stops_the_whole_group(self):
        with mock.patch.dict(os.environ, {"LODGER_FENCE": "none"}):
            f = fence.Fenced(["sh", "-c", "sleep 30 & sleep 30"], {}, {"MemoryMax": "1000000000"},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.3)
            self.assertGreaterEqual(len(f.pids()), 2)
            self.assertTrue(set(f.pids()) <= fence.our_pids())
            self.assertFalse(f.over_cap())
            f.stop()
            self.assertEqual(f.pids(), [])
            self.assertNotIn(f, fence._live)

    def test_none_kind_watchdog_cap(self):
        with mock.patch.dict(os.environ, {"LODGER_FENCE": "none"}):
            f = fence.Fenced(["sleep", "30"], {}, {"MemoryMax": "1024"}, stdout=subprocess.DEVNULL)
            time.sleep(0.2)
            self.assertTrue(f.over_cap())                 # any program is over 1 KiB
            f.kill(signal.SIGKILL)
            f.wait()
            f.close()

    def test_runtime_limit(self):
        with mock.patch.dict(os.environ, {"LODGER_FENCE": "none"}):
            f = fence.Fenced(["sleep", "30"], {}, {"RuntimeMaxSec": "0.3"}, stdout=subprocess.DEVNULL)
            t0 = time.time()
            f.wait(timeout=10)
            f.close()
        self.assertLess(time.time() - t0, 5)

    @unittest.skipUnless(fence.delegated()[0] and os.environ.get("LODGER_TEST_CGROUP"),
                         "needs a delegated cgroup v2 (set LODGER_TEST_CGROUP=1 to try)")
    def test_cgroup_kind(self):
        with mock.patch.dict(os.environ, {"LODGER_FENCE": "cgroup"}):
            f = fence.Fenced(["sleep", "30"], {}, {"MemoryMax": str(256 << 20), "MemorySwapMax": "0"},
                             stdout=subprocess.DEVNULL)
            time.sleep(0.3)
            try:
                self.assertTrue(os.path.basename(f.cg).startswith("lodger-"))
                with open(f.cg + "/memory.max") as m:
                    self.assertEqual(int(m.read()), 256 << 20)
                self.assertEqual(len(f.pids()), 1)
                self.assertGreater(f.memory()[0], 0)
            finally:
                f.stop()
            self.assertFalse(os.path.exists(f.cg))


class EnvTest(unittest.TestCase):
    """The engine's environment is the same under every fence (engine.child_env)."""
    LEAKY = {"ROCR_VISIBLE_DEVICES": "1", "CUDA_VISIBLE_DEVICES": "1", "MESA_VK_DEVICE_SELECT": "10de:2684",
             "HSA_OVERRIDE_GFX_VERSION": "11.0.0", "LLAMA_ARG_N_GPU_LAYERS": "5", "LODGER_TEST_KEEP": "yes"}
    SHOW = ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "MESA_VK_DEVICE_SELECT", "HSA_OVERRIDE_GFX_VERSION",
            "LLAMA_ARG_N_GPU_LAYERS", "HIP_VISIBLE_DEVICES", "LODGER_TEST_KEEP", "LODGER_FENCE_NAME")

    def setUp(self):
        fence._mode = None

    def tearDown(self):
        fence._mode = None

    def test_none_kind_drops_card_picking_and_keeps_runtime(self):
        code = "import os; print(repr({k: os.environ.get(k) for k in %r}))" % (self.SHOW,)
        with mock.patch.dict(os.environ, dict(self.LEAKY, LODGER_FENCE="none")):
            r = fence.run([sys.executable, "-c", code], {"HIP_VISIBLE_DEVICES": "0"}, {})
        got = eval(r.stdout)
        self.assertIsNone(got["ROCR_VISIBLE_DEVICES"])
        self.assertIsNone(got["CUDA_VISIBLE_DEVICES"])               # HIP reads it too
        self.assertIsNone(got["MESA_VK_DEVICE_SELECT"])
        self.assertIsNone(got["LLAMA_ARG_N_GPU_LAYERS"])
        self.assertEqual(got["HSA_OVERRIDE_GFX_VERSION"], "11.0.0")
        self.assertEqual(got["HIP_VISIBLE_DEVICES"], "0")
        self.assertEqual(got["LODGER_TEST_KEEP"], "yes")              # everything else as before
        self.assertTrue(got["LODGER_FENCE_NAME"].startswith(f"lodger-{os.getpid()}-"))

    def test_systemd_kind_passes_the_same(self):
        with mock.patch.dict(os.environ, self.LEAKY), \
                mock.patch("lodger.fence.mode", return_value=("systemd", "systemd user session")), \
                mock.patch("subprocess.Popen") as popen:
            f = fence.Fenced(["llama-server"], {"HIP_VISIBLE_DEVICES": "0"}, {"MemoryMax": "1000"}, name="lodger-1-1")
            f.close()
        argv = popen.call_args[0][0]
        self.assertIn("--setenv=HSA_OVERRIDE_GFX_VERSION=11.0.0", argv)
        self.assertIn("--setenv=HIP_VISIBLE_DEVICES=0", argv)
        self.assertFalse([a for a in argv if a.startswith(("--setenv=ROCR", "--setenv=LLAMA_ARG", "--setenv=LODGER"))])
        unset = next(a for a in argv if a.startswith("--property=UnsetEnvironment=")).split("=", 2)[2].split()
        for v in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "MESA_VK_DEVICE_SELECT", "LLAMA_ARG_N_GPU_LAYERS"):
            self.assertIn(v, unset)
        self.assertNotIn("HIP_VISIBLE_DEVICES", unset)
        self.assertEqual(argv[-1], "llama-server")


class SystemdCapTest(unittest.TestCase):
    """MemoryMax only works where the user manager has the memory controller: each unit is checked."""

    def fenced(self, cg_files, pids=(123,)):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for k, v in cg_files.items():
            with open(os.path.join(tmp.name, k), "w") as fh:
                fh.write(v)
        with mock.patch("lodger.fence.mode", return_value=("systemd", "systemd user session")), \
                mock.patch("subprocess.Popen"):
            f = fence.Fenced(["x"], {}, {"MemoryMax": str(1 << 30), "MemorySwapMax": "0"}, name="lodger-1-2")
        f.close()
        f.cg = tmp.name
        f.pids = lambda: list(pids)
        return f

    def test_cap_applied(self):
        f = self.fenced({"memory.max": str(1 << 30), "memory.swap.max": "0", "memory.current": str(2 << 30)})
        self.assertTrue(f.hard())
        self.assertFalse(f.over_cap())                     # the kernel enforces it, not the watchdog

    def test_cap_ignored_means_the_watchdog_enforces_it(self):
        f = self.fenced({"memory.max": "max", "memory.current": str(2 << 30),
                         "memory.stat": f"anon {2 << 30}\nfile 0\nshmem 0\n"})
        with mock.patch("lodger.fence.systemd_memory", return_value=(False, "not delegated")):
            self.assertFalse(f.hard())
            self.assertIn("not delegated", f.hard_why)
            self.assertTrue(f.over_cap())

    def test_model_file_pages_dont_count_against_an_unenforced_cap(self):
        # a "slow" plan: weights bigger than the room are read again as needed; the kernel would drop them
        f = self.fenced({"memory.max": "max", "memory.current": str(3 << 30),
                         "memory.stat": f"anon {512 << 20}\nfile {(3 << 30) - (512 << 20)}\nshmem 0\n"})
        with mock.patch("lodger.fence.systemd_memory", return_value=(False, "not delegated")):
            self.assertFalse(f.over_cap())

    def test_no_memory_controller_counts_the_processes(self):
        f = self.fenced({}, pids=(os.getpid(),))
        with mock.patch("lodger.fence.systemd_memory", return_value=(False, "cgroup v1")):
            cur, _ = f.memory()
            self.assertGreater(cur, 0)                     # not 0 from a missing memory.current
            self.assertFalse(f.hard())

    def test_not_judged_before_it_runs(self):
        f = self.fenced({"memory.max": "max"}, pids=())
        self.assertFalse(f.hard())
        self.assertIsNone(f._hard)                         # asked again once the unit has processes

    def test_user_manager_without_memory_controller(self):
        uid = os.getuid()
        with machine("desktop-amd", writable=True) as root:
            write(root, "/sys/fs/cgroup/cgroup.controllers", "cpu io memory pids\n")
            write(root, "/proc/self/cgroup", f"0::/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/x.scope\n")
            mgr = f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/cgroup.controllers"
            write(root, mgr, "cpu memory pids\n")
            self.assertEqual(fence.systemd_memory(), (True, None))
            write(root, mgr, "cpu pids\n")
            self.assertFalse(fence.systemd_memory()[0])
            os.remove(os.path.join(root, "sys/fs/cgroup/cgroup.controllers"))
            self.assertIn("cgroup v1", fence.systemd_memory()[1])


class SystemdKillTest(unittest.TestCase):
    """Stopping a model must not wait on systemd: under memory pressure (when the watchdog acts) starting
    systemctl and the D-Bus round trip can take seconds, or hang."""

    def fenced(self):
        with mock.patch("lodger.fence.mode", return_value=("systemd", "systemd user session")), \
                mock.patch("subprocess.Popen"):
            f = fence.Fenced(["x"], {}, {"MemoryMax": str(1 << 30)}, name="lodger-1-3")
        f.close()
        return f

    @staticmethod
    def hung(cmd, **kw):
        if "timeout" not in kw:
            raise AssertionError(f"{cmd} would wait forever on a hung systemd")
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    def test_kill_writes_the_units_cgroup_kill(self):
        f = self.fenced()
        with tempfile.TemporaryDirectory() as cg:
            open(os.path.join(cg, "cgroup.kill"), "w").close()
            f.cg = cg
            with mock.patch("lodger.fence.subprocess.run", side_effect=self.hung) as run:
                f.kill(signal.SIGKILL)
            with open(os.path.join(cg, "cgroup.kill")) as fh:
                self.assertEqual(fh.read(), "1")
        run.assert_not_called()

    def test_a_hung_systemctl_never_hangs_the_watchdog(self):
        f = self.fenced()
        with mock.patch("lodger.fence.subprocess.run", side_effect=self.hung):
            self.assertIsNone(f.cgroup())
            self.assertEqual(f.pids(), [])
            f.kill(signal.SIGKILL)
            f.kill(signal.SIGTERM)

    def test_sigkill_with_no_cgroup_known_asks_systemd_once_at_most(self):
        f = self.fenced()
        calls = []
        with mock.patch.object(fence, "_manager", return_value="/nonexistent"), \
                mock.patch.object(fence, "_systemctl", side_effect=lambda *a: calls.append(a[0])):
            f.kill(signal.SIGKILL)
            f.kill(signal.SIGKILL)                     # a systemd that didn't answer isn't asked every tick
            f.kill(signal.SIGTERM)
        self.assertEqual(calls, ["show"])

    def test_signals_go_to_the_units_processes_not_through_systemd(self):
        f = self.fenced()
        with tempfile.TemporaryDirectory() as cg:
            with open(os.path.join(cg, "cgroup.procs"), "w") as fh:
                fh.write("4001\n4002\n")
            f.cg = cg
            with mock.patch.object(fence, "_systemctl") as ctl, mock.patch("lodger.fence.os.kill") as kill:
                f.kill(signal.SIGTERM)
                f.kill(signal.SIGKILL)                 # no cgroup.kill (a kernel before 5.14): one by one
        ctl.assert_not_called()
        self.assertEqual(kill.call_args_list, [mock.call(4001, signal.SIGTERM), mock.call(4002, signal.SIGTERM),
                                               mock.call(4001, signal.SIGKILL), mock.call(4002, signal.SIGKILL)])

    def test_the_units_cgroup_is_found_without_systemctl(self):
        f = self.fenced()
        with tempfile.TemporaryDirectory() as mgr:
            os.makedirs(os.path.join(mgr, "app.slice", "lodger-1-3.service"))
            with mock.patch.object(fence, "_manager", return_value=mgr), mock.patch.object(fence, "_systemctl") as ctl:
                self.assertEqual(f.cgroup(), os.path.join(mgr, "app.slice", "lodger-1-3.service"))
        ctl.assert_not_called()

    def test_stop_gives_up_on_a_hung_systemctl(self):
        f = self.fenced()
        f.proc.wait.return_value = 0
        with mock.patch("lodger.fence.subprocess.run", side_effect=self.hung):
            f.stop()


class OrphanTest(unittest.TestCase):
    """Models a lodger that is gone left running: found by name and by LODGER_FENCE_NAME, then stopped."""
    UNITS = ("lodger.service loaded active running lodger: local AI that steps aside for your apps\n"
             "lodger-4242-3.service loaded active running model engine\n"
             "lodger-5151.service loaded active running model chat\n"
             "● lodger-4243-1.service loaded failed failed model engine\n")

    def setUp(self):
        self.cg = tempfile.TemporaryDirectory()
        for n in ("lodger-4242-2", "lodger-5151-1", "lodger-77.service", "app-foo.scope"):
            os.mkdir(os.path.join(self.cg.name, n))

    def tearDown(self):
        self.cg.cleanup()

    def found(self, runs=None):
        def run(cmd, **kw):
            if runs is not None:
                runs.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, self.UNITS if "list-units" in cmd else "", "")
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/100/environ", "HOME=/x\0LODGER_FENCE_NAME=lodger-4242-1\0")
            write(root, "/proc/300/environ", "LODGER_FENCE_NAME=lodger-5151-2\0")
            write(root, "/proc/400/environ", "X_LODGER_FENCE_NAME=lodger-4242-9\0")      # not the variable
            write(root, "/proc/4242/cmdline", "bash\0")                                   # pid reused by another program
            write(root, "/proc/5151/cmdline", "python3\0/home/u/.local/bin/lodger\0serve\0")
            with mock.patch("lodger.fence.shutil.which", return_value="/usr/bin/systemctl"), \
                    mock.patch("lodger.fence.subprocess.run", side_effect=run), \
                    mock.patch("lodger.fence.delegated", return_value=(self.cg.name, None)), \
                    mock.patch("lodger.fence.os.kill") as kill, \
                    mock.patch("lodger.fence._kill_cgroup") as killcg:
                if runs is None:
                    return fence.find_orphans()
                return fence.reap_orphans(), kill, killcg

    def test_find(self):
        got = {(o["kind"], o["name"]): o for o in self.found()}
        self.assertEqual(set(got), {("systemd", "lodger-4242-3"), ("cgroup", "lodger-4242-2"), ("process", "lodger-4242-1")})
        self.assertEqual(got[("process", "lodger-4242-1")]["where"], [100])

    def test_reap(self):
        runs = []
        done, kill, killcg = self.found(runs)
        self.assertEqual(len(done), 3)
        self.assertIn(["systemctl", "--user", "kill", "-s", "KILL", "lodger-4242-3.service"], runs)
        self.assertIn(["systemctl", "--user", "stop", "lodger-4242-3.service"], runs)
        self.assertFalse([c for c in runs if "lodger.service" in c or "lodger-5151.service" in c])
        killcg.assert_called_once_with(os.path.join(self.cg.name, "lodger-4242-2"))
        kill.assert_called_once_with(100, signal.SIGKILL)

    def test_owner_alive(self):
        self.assertTrue(fence.owner_alive(os.getpid()))
        self.assertFalse(fence.owner_alive(DEAD))

    def test_real_left_behind_program_is_stopped(self):
        p = subprocess.Popen(["sleep", "30"], env=dict(os.environ, LODGER_FENCE_NAME=f"lodger-{DEAD}-1"))
        try:
            time.sleep(0.2)
            with mock.patch("lodger.fence.shutil.which", return_value=None), \
                    mock.patch("lodger.fence.delegated", return_value=(None, "test")):
                self.assertIn(f"lodger-{DEAD}-1", [o["name"] for o in fence.find_orphans()])
                self.assertTrue(fence.reap_orphans())
            self.assertEqual(p.wait(timeout=5), -signal.SIGKILL)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()


class ParentDeathTest(unittest.TestCase):
    def setUp(self):
        fence._mode = None

    def tearDown(self):
        fence._mode = None

    def test_program_outlives_the_thread_that_started_it(self):
        box = {}
        with mock.patch.dict(os.environ, {"LODGER_FENCE": "none"}):
            t = threading.Thread(target=lambda: box.setdefault(
                "f", fence.Fenced(["sleep", "30"], {}, {}, stdout=subprocess.DEVNULL)))
            t.start()
            t.join()
            time.sleep(0.3)
            f = box["f"]
            try:
                self.assertIsNone(f.proc.poll())          # serve starts engines from threads that end
            finally:
                f.stop()

    @unittest.skipUnless(fence.pdeathsig(), "needs util-linux setpriv --pdeathsig")
    def test_program_dies_with_lodger(self):
        code = ("import subprocess, time\n"
                "from lodger import fence\n"
                "f = fence.Fenced(['sleep', '30'], {}, {}, stdout=subprocess.DEVNULL)\n"
                "time.sleep(0.2)\n"
                "print(f.proc.pid, flush=True)\n"
                "time.sleep(30)\n")
        p = subprocess.Popen([sys.executable, "-c", code], cwd=REPO, stdout=subprocess.PIPE, text=True,
                             env=dict(os.environ, LODGER_FENCE="none"))
        child = int(p.stdout.readline())
        p.kill()                                           # lodger is killed: no chance to stop its model
        p.wait()
        p.stdout.close()
        t0 = time.time()
        while time.time() - t0 < 5 and not zombie_or_gone(child):
            time.sleep(0.05)
        gone = zombie_or_gone(child)
        if not gone:
            os.kill(child, signal.SIGKILL)
        self.assertTrue(gone)


if __name__ == "__main__":
    unittest.main()
