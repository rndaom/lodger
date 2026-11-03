"""Apps lodger can't look into (another user's game, a root compositor, a system container): the kernel hides
their fdinfo, so they are found from the whole card's numbers minus every app lodger can see."""
import unittest
from unittest import mock

from lodger import gpu, neighbors
from lodger.gpu import fdinfo
from lodger.neighbors import UNSEEN, Neighbors
from tests.support import machine, write

MiB, GiB = 1 << 20, 1 << 30
DEAD = 4194305                  # above any pid_max: a pid no process has


class Card:
    """A card where the test sets each app's busy share and memory, and what the whole card shows."""
    pci = "0000:03:00.0"
    integrated = False

    def __init__(self):
        self.t = 1000.0
        self.apps = {}          # pid -> [busy share, vram, gtt, busy ns so far]
        self.hidden_busy = 0.0  # what apps lodger can't see add to the whole card
        self.hidden_vram = 0
        self.lag = 0            # memory an app freed that the card still shows
        self.base = 300 * MiB   # the kernel's own

    def advance(self, dt):
        self.t += dt
        for a in self.apps.values():
            a[3] += int(a[0] * dt * 1e9)

    def usage(self, pids):
        return {p: {"vram": a[1], "gtt": a[2], "busy_ns": a[3], "cycles": {}} for p, a in self.apps.items() if p in pids}

    def whole(self):
        busy = min(1.0, sum(a[0] for a in self.apps.values()) + self.hidden_busy)
        return {"busy": busy, "used": self.base + self.lag + self.hidden_vram + sum(a[1] for a in self.apps.values())}


class Run:
    def __init__(self, card, ours=(), names=None):
        self.card, self.names = card, names or {}
        self.nb = Neighbors(card, ours=lambda: list(ours))

    def __enter__(self):
        c = self.card
        self.patches = [mock.patch("lodger.neighbors.time.monotonic", side_effect=lambda: c.t),
                        mock.patch("lodger.sysmem.pids", side_effect=lambda: sorted(c.apps)),
                        mock.patch("lodger.sysmem.app_name", side_effect=lambda p: self.names.get(p, f"app{p}"))]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()

    def run(self, seconds, step=0.25, during=None):
        """Sample every `step` s; the first reason, or None."""
        n = int(seconds / step)
        for i in range(n):
            if during:
                during(i * step)
            r = self.nb.update()
            if r:
                return r
            self.card.advance(step)
        return None


class UnseenBusyTest(unittest.TestCase):
    def test_another_users_game_keeping_the_card_busy_is_seen(self):
        c = Card()
        c.apps = {10: [0.05, 300 * MiB, 0, 0]}      # the compositor
        c.hidden_busy = 0.6
        with Run(c) as r:
            why = r.run(6)
        self.assertIn(UNSEEN, why or "")
        self.assertIn("busy (60%)", why)

    def test_apps_lodger_sees_dont_leave_a_remainder(self):
        c = Card()
        c.apps = {10: [0.20, 300 * MiB, 0, 0], 11: [0.10, 200 * MiB, 0, 0]}
        c.hidden_busy = 0.03                         # what the card's own average adds on top (seen here: ~3%)
        with Run(c) as r:
            self.assertIsNone(r.run(10))

    def test_lodgers_own_busy_model_isnt_compared(self):
        # the card's average and the model's counters aren't measured alike: only judged while the model rests
        c = Card()
        c.apps = {10: [0.05, 300 * MiB, 0, 0], DEAD: [0.70, 8 * GiB, 0, 0]}
        c.hidden_busy = 0.3
        with Run(c, ours=[DEAD]) as r:
            self.assertIsNone(r.run(10))

    def test_lodgers_resting_model_and_a_hidden_game(self):
        c = Card()
        c.apps = {10: [0.05, 300 * MiB, 0, 0], DEAD: [0.0, 8 * GiB, 0, 0]}
        c.hidden_busy = 0.5
        with Run(c, ours=[DEAD]) as r:
            self.assertIn(UNSEEN, r.run(6) or "")

    def test_lodgers_own_rocm_model_isnt_a_stranger(self):
        # ROCm's compute queues bypass the kernel's per-app busy counters: the model's work shows only card-wide
        c = Card()
        c.apps = {10: [0.05, 300 * MiB, 0, 0], DEAD: [0.0, 8 * GiB, 0, 0]}
        c.hidden_busy = 0.9
        with Run(c, ours=[DEAD]) as r, mock.patch.object(fdinfo, "holds", side_effect=lambda pid, path: path == "/dev/kfd"):
            self.assertIsNone(r.run(10))

    def test_a_rocm_app_lodger_can_see_counts_through_the_whole_card(self):
        c = Card()
        c.apps = {10: [0.05, 300 * MiB, 0, 0], 20: [0.0, 2 * GiB, 0, 0]}     # its compute shows nowhere per app
        c.hidden_busy = 0.8
        with Run(c, names={20: "blender"}) as r:
            self.assertIn(UNSEEN, r.run(6) or "")

    def test_a_kernel_without_usage_stats_never_counts_lodgers_own_work(self):
        # lodger's model holds the card, but its fdinfo shows nothing: the whole card's numbers can't be split
        c = Card()
        c.hidden_busy = 0.9
        with Run(c, ours=[DEAD]) as r, mock.patch.object(fdinfo, "drm_fds", return_value=["5"]):
            self.assertIsNone(r.run(10))

    def test_a_new_app_lodger_can_see_is_named_not_called_unseen(self):
        c = Card()
        c.apps = {10: [0.05, 300 * MiB, 0, 0]}

        def start(t):
            if t == 0.5:                             # starts between two full scans
                c.apps[20] = [0.6, 900 * MiB, 0, 0]
        with Run(c, names={20: "game"}) as r:
            why = r.run(8, during=start)
        self.assertTrue(why and why.startswith("game "), why)


class UnseenMemoryTest(unittest.TestCase):
    def test_a_hidden_app_loading_is_seen(self):
        c = Card()
        c.apps = {10: [0.0, 300 * MiB, 0, 0]}

        def load(t):
            c.hidden_vram = min(GiB, int(t * 400 * MiB))
        with Run(c) as r:
            why = r.run(8, during=load)
        self.assertIn(UNSEEN, why or "")
        self.assertIn("MiB of graphics memory", why)

    def test_memory_freed_a_moment_late_isnt_a_stranger(self):
        c = Card()
        c.apps = {10: [0.0, 300 * MiB, 0, 0], 11: [0.0, 4 * GiB, 0, 0]}

        def close(t):
            if t == 3.0:                             # the game exits: its fdinfo goes at once...
                del c.apps[11]
                c.lag = 4 * GiB
            elif t == 3.5:                           # ...its memory half a second later
                c.lag = 0
        with Run(c) as r:
            self.assertIsNone(r.run(10, during=close))


class BuiltInGraphicsTest(unittest.TestCase):
    def test_an_app_loading_into_shared_memory_needs_the_card(self):
        c = Card()
        c.integrated = True
        c.apps = {10: [0.0, 64 * MiB, 200 * MiB, 0]}

        def load(t):
            c.apps[10][2] = 200 * MiB + min(GiB, int(t * 400 * MiB))
        with Run(c, names={10: "game"}) as r:
            why = r.run(6, during=load)
        self.assertIn("game took", why or "")


class CardNumbersTest(unittest.TestCase):
    def test_amd_whole_card(self):
        with machine("desktop-amd"):
            d = gpu.choose()
            self.assertTrue(d.has_busy())
            self.assertEqual(d.whole()["busy"], 0.23)
            self.assertEqual(d.whole()["used"], d.memory()["used"])
            sees, words = neighbors.coverage(d)
            self.assertFalse(sees)
            self.assertIn("how busy the whole card is", words)

    def test_asleep_card_isnt_asked(self):
        with machine("desktop-amd", writable=True) as root:
            write(root, "/sys/bus/pci/devices/0000:03:00.0/power/runtime_status", "suspended\n")
            self.assertIsNone(gpu.choose("03:00.0").busy())

    def test_nvidia_lists_everyone(self):
        with machine("laptop-nvidia"):
            d = gpu.choose("01:00.0")
            self.assertIsNone(d.whole())
            self.assertTrue(neighbors.coverage(d)[0])

    def test_intel_says_what_it_cant_see(self):
        with machine("desktop-arc"):
            sees, words = neighbors.coverage(gpu.choose("03:00.0"))
        self.assertFalse(sees)
        self.assertIn("not at all", words)


if __name__ == "__main__":
    unittest.main()
