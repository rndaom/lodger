"""The watchdog's wrong-card check, from the kernel's fdinfo (recorded desktop: pids 100/200 share a client
on the RX 7800 XT at 03:00.0 with ~413 MiB; pid 300 has ~2 MiB on the built-in graphics at 11:00.0)."""
import argparse
import contextlib
import io
import os
import signal
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from lodger import config, launch
from lodger.gpu import fdinfo
from lodger.neighbors import Neighbors
from tests.support import machine

MiB = 1 << 20


class FakeFence:
    def __init__(self, pids, gone=True):
        self._pids, self.signals, self._gone = pids, [], gone
        self.cap = None

    def pids(self):
        return self._pids

    def kill(self, sig):
        self.signals.append(sig)

    def gone(self):
        return self._gone

    def over_cap(self):
        return False

    def memory(self):
        return 0, 0


def wait_for(cond, limit=3.0):
    t0 = time.time()
    while time.time() - t0 < limit and not cond():
        time.sleep(0.01)
    return cond()


class FailClosedTest(unittest.TestCase):
    """The watchdog never leaves a model unwatched: a check it can't make stops the model."""

    def setUp(self):
        config.use(config.load(os.devnull))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(config.use, None)
        self.quiet = contextlib.redirect_stderr(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def start(self, fence, out=None):
        w = launch.Watchdog(fence, False, out or self.tmp.name)
        w.start()
        self.addCleanup(w.join, 2)
        self.addCleanup(w.stop_flag.set)
        return w

    def test_an_error_in_a_check_stops_the_model_and_keeps_watching(self):
        f = FakeFence([], gone=False)
        with machine("desktop-amd"), mock.patch.object(launch.sysmem, "meminfo", return_value={}):
            w = self.start(f)
            self.assertTrue(wait_for(lambda: w.reason and f.signals.count(signal.SIGKILL) >= 3))
            self.assertIn("couldn't check", w.reason)
            self.assertIn("KeyError", w.reason)
            self.assertTrue(w.is_alive())
            self.assertTrue(w.healthy())

    def test_kill_is_retried_until_the_fence_is_empty(self):
        f = FakeFence([], gone=False)
        with machine("desktop-amd"):
            w = launch.Watchdog(f, False, self.tmp.name)
            w.kill("first")
            w.kill("second")
        self.assertEqual(w.reason, "first")
        self.assertEqual(f.signals, [signal.SIGKILL, signal.SIGKILL])

    def test_a_closed_terminal_never_stops_a_kill(self):
        f = FakeFence([], gone=False)
        closed = io.StringIO()
        closed.close()                                         # lodger's terminal went away
        with machine("desktop-amd"), contextlib.redirect_stderr(closed), contextlib.redirect_stdout(closed):
            w = launch.Watchdog(f, False, self.tmp.name)
            w.kill("only 100 MiB of RAM left")
        self.assertEqual(f.signals, [signal.SIGKILL])
        self.assertEqual(w.reason, "only 100 MiB of RAM left")

    def test_a_closed_terminal_never_ends_the_watching(self):
        f = FakeFence([], gone=False)
        closed = io.StringIO()
        closed.close()
        with machine("desktop-amd"), contextlib.redirect_stderr(closed), \
                mock.patch.object(launch.sysmem, "meminfo", return_value={}):
            w = self.start(f)
            self.assertTrue(wait_for(lambda: f.signals.count(signal.SIGKILL) >= 3))
            self.assertTrue(w.is_alive())

    def test_watching_is_false_once_the_thread_is_gone_with_the_model_alive(self):
        f = FakeFence([], gone=False)
        with machine("desktop-amd"):
            w = launch.Watchdog(f, False, self.tmp.name)
            w.reason = "only 100 MiB of RAM left"              # decided to stop it, then the thread died
            self.assertFalse(w.watching())
            f._gone = True
            self.assertTrue(w.watching())                      # nothing left to watch

    def test_logging_trouble_never_stops_the_watching(self):
        f = FakeFence([1234])
        low = {"MemTotal": 32 << 30, "MemAvailable": 100 * MiB}
        with machine("desktop-amd"), mock.patch.object(launch.sysmem, "gpu_use", side_effect=OSError("disk full")):
            w = self.start(f, out=os.path.join(self.tmp.name, "missing", "folder"))     # watch.tsv can't be made
            time.sleep(0.3)
            self.assertTrue(w.healthy())
            self.assertIsNone(w.reason)
            with mock.patch.object(launch.sysmem, "meminfo", return_value=low):
                self.assertTrue(wait_for(lambda: w.reason))
        self.assertIn("RAM left", w.reason)

    def test_healthy_says_when_nobody_is_watching(self):
        with machine("desktop-amd"):
            w = launch.Watchdog(FakeFence([]), False, self.tmp.name)
            self.assertFalse(w.healthy())                       # not started
            w.start()
            self.assertTrue(w.healthy())
            w.beat -= 60                                        # stuck
            self.assertFalse(w.healthy())
            self.assertTrue(wait_for(w.healthy))                # it comes back once a check finishes
            w.stop_flag.set()
            w.join(2)
            self.assertFalse(w.healthy())

    def test_run_stops_the_model_when_the_watchdog_is_dead(self):
        class Proc(FakeFence):
            def wait(self, timeout=None):
                if signal.SIGKILL in self.signals:
                    return -9
                time.sleep(timeout)
                raise subprocess.TimeoutExpired("llama", timeout)
        f = Proc([])
        with machine("desktop-amd"):
            w = launch.Watchdog(f, False, self.tmp.name)        # never started: as good as dead
            self.assertEqual(launch.wait_watched(f, w, every=0.01), -9)
        self.assertIn("stopped watching", w.reason)


class SettleTest(unittest.TestCase):
    """An app already busy on the card is seen before lodger loads anything."""

    class Clock:
        def __init__(self):
            self.t = 1000.0

        def time(self):
            return self.t

        monotonic = time

        def sleep(self, s):
            self.t += s

    class Card:
        pci = "0000:03:00.0"

        def __init__(self, clock, busy):
            self.clock, self.busy = clock, busy

        def usage(self, pids):
            return {p: {"vram": 300 * MiB, "gtt": 0, "busy_ns": int(self.clock.t * 1e9 * self.busy), "cycles": {}}
                    for p in pids}

        def free(self, meminfo=None):
            return 12 << 30

    def settle(self, busy):
        clock = self.Clock()
        with mock.patch("lodger.launch.time", clock), mock.patch("lodger.neighbors.time", clock), \
                mock.patch.object(launch.sysmem, "pids", return_value=[4242]), \
                mock.patch.object(launch.sysmem, "app_name", return_value="game"), \
                mock.patch.object(launch.fence, "our_pids", return_value=[]):
            return launch.settle(self.Card(clock, busy))

    def test_a_busy_app_is_seen(self):
        self.assertIn("game is keeping the graphics card busy", self.settle(0.6) or "")

    def test_a_calm_card_is_free(self):
        self.assertIsNone(self.settle(0.01))


class AllowGpuTest(unittest.TestCase):
    def test_yes_at_the_question_also_asks_llama_cpp(self):
        a = argparse.Namespace(model="/m.gguf", ctx=None, min_ctx=4096, cpu_only=False, allow_gpu=False, leave_gpu=0,
                               threads=None, pretend_vram_free=None, pretend_ram_avail=None)
        with mock.patch.object(launch.sysmem, "dgpu", return_value=None), \
                mock.patch.object(launch.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value="y"), \
                mock.patch.object(launch, "gpu_blockers", return_value=None) as blockers, \
                mock.patch.object(launch.planner, "make", return_value={}) as make:
            launch.build_plan(a, True)
        self.assertTrue(blockers.call_args[0][0])
        self.assertTrue(make.call_args.kwargs["device_check"])


class NeighborsOrderTest(unittest.TestCase):
    def test_partial_scans_go_in_pid_order(self):
        seen = []

        class Card:
            pci = "0000:03:00.0"

            def usage(self, pids):
                seen.append(list(pids))
                return {p: fdinfo.empty() for p in pids}
        nb = Neighbors(Card(), ours=lambda: ())
        nb.gpu_pids = {200: "b", 100: "a", 30: "c"}            # found in this order by earlier full scans
        nb.scanned = time.monotonic()
        nb._sample()
        self.assertEqual(seen, [[30, 100, 200]])


class WrongCardTest(unittest.TestCase):
    def setUp(self):
        config.use(config.load(os.devnull))
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        config.use(None)
        self.tmp.cleanup()

    def check(self, pids, on_gpu, log=None):
        f = FakeFence(pids)
        w = launch.Watchdog(f, on_gpu, self.tmp.name, log=log)
        with contextlib.redirect_stderr(io.StringIO()):
            stopped = w.check_device()
        return stopped, w.reason, f.signals

    def test_per_card_counts_a_shared_client_once(self):
        with machine("desktop-amd"):
            card = sum(r["vram"] + r["gtt"] for r in fdinfo.clients("100").values())
            igpu = sum(r["vram"] + r["gtt"] for r in fdinfo.clients("300").values())
            self.assertEqual(fdinfo.per_card(["100", "200", "300"]), {"0000:03:00.0": card, "0000:11:00.0": igpu})
            self.assertGreater(card, 400 * MiB)

    def test_on_the_chosen_card(self):
        with machine("desktop-amd"):
            self.assertEqual(self.check(["100", "300"], True)[:2], (False, None))

    def test_on_another_card(self):
        with machine("desktop-amd"), mock.patch.object(launch, "OTHER_CARD", MiB):
            stopped, why, signals = self.check(["300"], True)
        self.assertTrue(stopped)
        self.assertIn("0000:11:00.0", why)
        self.assertIn("0000:03:00.0", why)
        self.assertTrue(signals)

    def test_processor_only_model_on_a_card(self):
        with machine("desktop-amd"):
            stopped, why, _ = self.check(["100"], False)
        self.assertTrue(stopped)
        self.assertIn("processor only", why)

    def test_log_line_still_counts(self):
        log = os.path.join(self.tmp.name, "engine.log")
        with open(log, "w") as f:
            f.write("llama_model_load: using device Vulkan0 (AMD Radeon 610M (RADV RAPHAEL)) (0000:11:00.0) - 500 MiB free\n")
        with machine("desktop-amd"):
            stopped, why, _ = self.check([], True, log=log)
        self.assertTrue(stopped)
        self.assertIn("0000:11:00.0", why)


class FakeCard:
    """A card whose free memory the test sets (None: it can't be read, as during a reset or after unplugging)."""
    pci = "0000:03:00.0"
    integrated = False
    known = True

    def __init__(self, free, crashed=False, hot=None):
        self._free, self.crashed, self.hot = free, crashed, hot

    def free(self, mem=None, meminfo=None):
        return self._free

    def usage(self, pids):
        return {}

    def asleep(self):
        return False

    def crash_report(self):
        return self.crashed

    def too_hot(self, starting=False):
        return self.hot


class StopTest(unittest.TestCase):
    def setUp(self):
        config.use(config.load(os.devnull))
        self.addCleanup(config.use, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.quiet = contextlib.redirect_stderr(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def test_the_kernel_stops_the_model_first_when_memory_runs_out(self):
        props = launch.unit_props({"ram_cap": 8 << 30, "ngl": 0})
        self.assertEqual(props["OOMScoreAdjust"], "1000")

    def test_it_watches_the_card_the_model_was_planned_for(self):
        planned = FakeCard(10 << 30)
        with machine("desktop-amd"):                               # sysmem.dgpu() would pick the recorded card
            w = launch.Watchdog(FakeFence([]), True, self.tmp.name, keep=2 << 30, card=planned)
        self.assertIs(w.card, planned)

    def test_card_memory_that_cant_be_read_stops_the_model_in_plain_words(self):
        f = FakeFence([], gone=False)
        with machine("desktop-amd"):
            w = launch.Watchdog(f, True, self.tmp.name, keep=2 << 30, card=FakeCard(None))
            w.check(0.0, False)
        self.assertIn("can't be read", w.reason)
        self.assertNotIn("only 0 MiB", w.reason)

    def test_a_graceful_stop_turns_hard_when_memory_runs_out_meanwhile(self):
        f = FakeFence([], gone=False)                          # the model ignores SIGTERM
        card = FakeCard(10 << 30)
        with machine("desktop-amd"):
            w = launch.Watchdog(f, True, self.tmp.name, keep=2 << 30, card=card)
            card._free = 200 * MiB                             # a game loads fast while the model winds down
            t0 = time.monotonic()
            w.kill("a game needs the card; giving the graphics card back", graceful=True)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(f.signals[:2], [signal.SIGTERM, signal.SIGKILL])


class CardTroubleTest(unittest.TestCase):
    """While the model is on the card: the card resets or hangs, the laptop is unplugged, the card overheats."""

    def setUp(self):
        config.use(config.load(os.devnull))
        self.addCleanup(config.use, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.quiet = contextlib.redirect_stderr(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def watch(self, card, seconds=1.0, power=("ac", "plugged in")):
        f = FakeFence([], gone=False)
        with machine("desktop-amd"), mock.patch.object(launch.gpu, "power_source", return_value=power), \
                mock.patch.object(launch.Neighbors, "update", return_value=None):
            w = launch.Watchdog(f, True, self.tmp.name, keep=2 << 30, card=card)
            t = 100.0
            while t < 100.0 + seconds and not w.reason:
                w.check(t, False)
                t += 0.25
        return w, f

    def test_a_card_reset_stops_the_model(self):
        w, f = self.watch(FakeCard(10 << 30, crashed=True))
        self.assertIn("reset or hung", w.reason or "")
        self.assertNotIn(signal.SIGTERM, f.signals)             # no waiting for a model on a card in trouble

    def test_going_on_battery_steps_aside(self):
        w, f = self.watch(FakeCard(10 << 30), power=("battery", "on battery (BAT0 discharging)"))
        self.assertIn("on battery", w.reason or "")
        self.assertEqual(f.signals[0], signal.SIGTERM)

    def test_battery_allowed_in_the_settings(self):
        config.get()["gpu_on_battery"] = True
        w, _ = self.watch(FakeCard(10 << 30), power=("battery", "on battery"))
        self.assertIsNone(w.reason)

    def test_built_in_graphics_dont_care_about_the_battery(self):
        card = FakeCard(10 << 30)
        card.integrated = True
        w, _ = self.watch(card, power=("battery", "on battery"))
        self.assertIsNone(w.reason)

    def test_an_overheating_card_steps_aside_after_a_few_seconds(self):
        w, _ = self.watch(FakeCard(10 << 30, hot="junction 111 C (critical 110 C)"), seconds=3)
        self.assertIsNone(w.reason)                             # a moment over the line isn't enough
        w, _ = self.watch(FakeCard(10 << 30, hot="junction 111 C (critical 110 C)"), seconds=8)
        self.assertIn("too hot", w.reason or "")

    def test_no_start_on_a_card_in_trouble(self):
        with mock.patch.object(launch, "settle", side_effect=AssertionError("not needed")):
            self.assertIn("reset or hung", launch.gpu_blockers(True, FakeCard(10 << 30, crashed=True)))
            self.assertIn("too hot", launch.gpu_blockers(True, FakeCard(10 << 30, hot="edge 101 C (critical 100 C)")))
            self.assertIsNone(launch.gpu_blockers(True, FakeCard(10 << 30), settled=True))   # serve watched it already


if __name__ == "__main__":
    unittest.main()
