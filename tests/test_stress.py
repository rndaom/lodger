"""The stress test's own helpers (nothing is started on the card)."""
import os
import signal
import subprocess
import sys
import unittest
from unittest import mock

from lodger import stress

KLOG = """[drm:amdgpu_job_timedout [amdgpu]] *ERROR* ring gfx_0.0.0 timeout, signaled seq=1, emitted seq=2
NVRM: Xid (PCI:0000:01:00): 79, pid=1234, name=llama-server, GPU has fallen off the bus.
NVRM: loading NVIDIA UNIX Open Kernel Module for x86_64  570.153.02
i915 0000:00:02.0: [drm] GPU HANG: ecode 12:1:85dffffb, in llama-server [1234]
usb 1-2: device descriptor read/64, error -71 timeout
"""


class KernelLogTest(unittest.TestCase):
    def test_per_driver(self):
        self.assertEqual(stress.kernel_trouble(KLOG, "amdgpu"), [KLOG.splitlines()[0]])
        self.assertEqual(stress.kernel_trouble(KLOG, "nvidia"), [KLOG.splitlines()[1]])   # "NVRM: Xid", no driver name
        self.assertEqual(stress.kernel_trouble(KLOG, "i915"), [KLOG.splitlines()[3]])
        self.assertEqual(len(stress.kernel_trouble(KLOG, None)), 3)                         # never the usb line


class CommandTest(unittest.TestCase):
    def test_runs_this_lodger_even_when_installed(self):
        self.assertEqual(stress.LODGER, [sys.executable, "-m", "lodger.cli"])
        env = stress.child_env({"X": "1"})
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], stress.ROOT)
        r = subprocess.run(stress.LODGER + ["--help"], env=env, capture_output=True, text=True, cwd="/")
        self.assertEqual(r.returncode, 0)
        self.assertIn("lodger doctor", r.stdout)

    def test_end_asks_lodger_to_stop_its_model(self):
        p = subprocess.Popen([sys.executable, "-c", "import signal, time\n"
                              "signal.signal(signal.SIGINT, lambda *a: exit(0))\ntime.sleep(30)"],
                             stdout=subprocess.PIPE, text=True)
        import time
        time.sleep(0.3)
        with mock.patch("lodger.fence.reap_orphans") as reap:
            stress.end(p)
        self.assertEqual(p.returncode, 0)                   # Ctrl+C was enough: no SIGKILL, nothing to reap
        reap.assert_not_called()

    def test_end_kills_only_as_a_last_resort(self):
        p = mock.Mock()
        p.poll.return_value = None
        p.communicate.side_effect = [subprocess.TimeoutExpired("x", 1), subprocess.TimeoutExpired("x", 1), ("", None)]
        with mock.patch("lodger.fence.reap_orphans", return_value=[]) as reap:
            stress.end(p)
        self.assertEqual([c.args[0] for c in p.send_signal.call_args_list], [signal.SIGINT, signal.SIGTERM])
        p.kill.assert_called_once()
        reap.assert_called_once()


if __name__ == "__main__":
    unittest.main()
