"""The watchdog's wrong-card check, from the kernel's fdinfo (recorded desktop: pids 100/200 share a client
on the RX 7800 XT at 03:00.0 with ~413 MiB; pid 300 has ~2 MiB on the built-in graphics at 11:00.0)."""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from lodger import config, launch
from lodger.gpu import fdinfo
from tests.support import machine

MiB = 1 << 20


class FakeFence:
    def __init__(self, pids):
        self._pids, self.signals = pids, []

    def pids(self):
        return self._pids

    def kill(self, sig):
        self.signals.append(sig)

    def gone(self):
        return True


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


if __name__ == "__main__":
    unittest.main()
