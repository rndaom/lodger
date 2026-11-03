"""Clock steps and suspend: lodger's timing must use the monotonic clock. The wall clock jumps (NTP, the user
setting the time, a resume from suspend), and a jump must neither blind the watching nor hide a stuck watchdog."""
import tempfile
import unittest
from unittest import mock

from lodger import config, launch
from lodger.gpu import nvidia
from lodger.neighbors import Neighbors
from tests.support import FakeNVML, machine, write

MiB = 1 << 20
HOUR = 3600.0


class Clocks:
    """A monotonic clock that only moves forward, and a wall clock that can be stepped."""

    def __init__(self):
        self.mono, self.wall = 1000.0, 1_700_000_000.0

    def tick(self, s):
        self.mono += s
        self.wall += s

    def patch(self, module):
        return mock.patch.multiple(f"{module}.time", time=lambda: self.wall, monotonic=lambda: self.mono)


class BusyCard:
    pci = "0000:03:00.0"
    integrated = False

    def __init__(self, clocks, busy):
        self.clocks, self.busy = clocks, busy

    def usage(self, pids):
        return {p: {"vram": 300 * MiB, "gtt": 0, "busy_ns": int(self.clocks.mono * 1e9 * self.busy), "cycles": {}}
                for p in pids}


class NeighborsClockTest(unittest.TestCase):
    def test_an_app_started_after_the_wall_clock_steps_back_is_seen(self):
        c = Clocks()
        running = []
        with c.patch("lodger.neighbors"), mock.patch("lodger.sysmem.pids", side_effect=lambda: list(running)), \
                mock.patch("lodger.sysmem.app_name", return_value="game"):
            nb = Neighbors(BusyCard(c, 0.6))
            nb.update()
            c.tick(0.25)
            c.wall -= HOUR                                      # NTP (or the user) sets the clock back
            running.append(4242)                                # then a game starts
            reason = None
            for _ in range(20):
                c.tick(0.25)
                reason = nb.update() or reason
        self.assertIn("game is keeping the graphics card busy", reason or "")


class WatchdogClockTest(unittest.TestCase):
    def setUp(self):
        config.use(config.load("/dev/null"))
        self.addCleanup(config.use, None)

    def test_a_stuck_watchdog_isnt_hidden_by_a_clock_step_back(self):
        c = Clocks()

        class F:
            cap = None
        with tempfile.TemporaryDirectory() as tmp, machine("desktop-amd"), c.patch("lodger.launch"):
            w = launch.Watchdog(F(), False, tmp)
            with mock.patch.object(w, "is_alive", return_value=True):
                self.assertTrue(w.healthy())
                c.tick(60)                                      # no check finished for a minute: stuck
                c.wall -= HOUR
                self.assertFalse(w.healthy())

    def test_a_resume_from_suspend_isnt_a_stuck_watchdog(self):
        c = Clocks()

        class F:
            cap = None
        with tempfile.TemporaryDirectory() as tmp, machine("desktop-amd"), c.patch("lodger.launch"):
            w = launch.Watchdog(F(), False, tmp)
            with mock.patch.object(w, "is_alive", return_value=True):
                c.wall += 8 * HOUR                              # the night in suspend: the monotonic clock stood still
                self.assertTrue(w.healthy())


class NvidiaClockTest(unittest.TestCase):
    def test_busy_time_still_counts_after_the_wall_clock_steps_back(self):
        with machine("laptop-nvidia", writable=True) as root:
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "active\n")
            t = 1_700_000_000_000_000
            fake = FakeNVML(procs={500: 1 << 30}, samples=[(500, t, 60)])
            nvidia._nvml = fake
            from lodger import gpu
            card = gpu.choose()
            with mock.patch("time.time", return_value=t / 1e6):
                before = card.usage([500])[500]["busy_ns"]
            back = t - int(HOUR * 1e6)                          # the clock went back an hour; NVML stamps follow it
            fake.samples += [(500, back + 1_000_000, 60), (500, back + 2_000_000, 60)]
            with mock.patch("time.time", return_value=(back + 2_000_000) / 1e6):
                card.usage([500])
                b = card.usage([500])[500]["busy_ns"]
            self.assertGreater(b - before, 0)
            # NVML still holds the sample stamped an hour ahead: it must not count as an hour of work
            self.assertLess(b - before, 1_000_000_000)
            fake.samples += [(500, back + 3_000_000, 60)]
            with mock.patch("time.time", return_value=(back + 3_000_000) / 1e6):
                b2 = card.usage([500])[500]["busy_ns"]
            self.assertAlmostEqual(b2 - b, 600_000_000, delta=1_000_000)   # a second at 60 %, as before the step

    def test_a_long_gap_between_samples_counts_one_sample_period_at_most(self):
        with machine("laptop-nvidia", writable=True) as root:
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "active\n")
            t = 1_700_000_000_000_000
            fake = FakeNVML(procs={500: 1 << 30}, samples=[(500, t, 100)])
            nvidia._nvml = fake
            from lodger import gpu
            card = gpu.choose()
            with mock.patch("time.time", return_value=t / 1e6):
                before = card.usage([500])[500]["busy_ns"]
            fake.samples += [(500, t + 60_000_000, 100)]       # one busy sample after a minute without any
            with mock.patch("time.time", return_value=(t + 60_000_000) / 1e6):
                b = card.usage([500])[500]["busy_ns"]
            self.assertLessEqual(b - before, 1_000_000_000)


if __name__ == "__main__":
    unittest.main()
