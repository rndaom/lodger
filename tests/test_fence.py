"""The fence without systemd: the 'none' kind runs real (harmless) programs; no graphics card involved."""
import os
import signal
import subprocess
import time
import unittest
from unittest import mock

from lodger import fence


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


if __name__ == "__main__":
    unittest.main()
