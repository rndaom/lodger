"""The desktop reserve learned from notes (reserve.py): a damaged notes file must never make it absurd."""
import shutil
import tempfile
import unittest

from lodger import config, reserve

GiB = 1 << 30


class ReserveNotesTest(unittest.TestCase):
    def setUp(self):
        runs = tempfile.mkdtemp(prefix="lodger-runs-")
        self.addCleanup(shutil.rmtree, runs)
        cfg = config.load("/nonexistent")
        cfg["runs_dir"] = runs
        config.use(cfg)
        self.addCleanup(config.use, None)
        self.addCleanup(reserve._notes.clear)

    def test_infinite_or_nan_notes_count_as_no_notes(self):
        # Python's json reads Infinity and NaN: an infinite reserve would keep the model off the card for good,
        # and kill_line(inf) would stop every model on the card at once
        for i, bad in enumerate(["Infinity", "NaN", "1e400"]):
            pci = f"0000:0{i}:00.0"
            with open(reserve._path(pci), "w") as f:
                f.write("{" + ", ".join(f'"{m}": [0, {bad if m == 5 else 1}]' for m in range(90)) + "}")
            self.assertEqual(reserve._load(pci), {}, bad)
            self.assertEqual(reserve.reserve(pci, 16 * GiB), (2 * GiB, None))

    def test_never_under_the_floor(self):
        notes = reserve._load("0000:09:00.0")
        for m in range(120):
            notes[m] = [GiB, GiB]
        self.assertEqual(reserve.reserve("0000:09:00.0", 16 * GiB), (2 * GiB, None))


    def test_learned_part_is_at_most_a_quarter_of_the_card(self):
        # an idle game or a test's memory grabber noted as "desktop" (seen on the project's PC: 5.8 GiB learned on
        # a 16 GiB card whose desktop uses ~1 GiB) must not take the card away for a week
        notes = reserve._load("0000:09:00.0")
        for m in range(120):
            notes[m] = [GiB, GiB] if m % 20 < 10 else [GiB, 12 * GiB]
        keep, why = reserve.reserve("0000:09:00.0", 16 * GiB)
        self.assertEqual(keep, 4 * GiB)
        self.assertIn("4.0 GiB", why)

    def test_a_small_learned_rise_is_kept(self):
        notes = reserve._load("0000:09:00.0")
        for m in range(120):
            notes[m] = [GiB, GiB] if m % 20 < 10 else [GiB, 3 * GiB]
        self.assertEqual(reserve.reserve("0000:09:00.0", 16 * GiB)[0], 2 * GiB + reserve.HEADROOM)

    def test_report_for_doctor(self):
        notes = reserve._load("0000:09:00.0")
        for m in range(120):
            notes[m] = [GiB, GiB] if m % 20 < 10 else [GiB, 12 * GiB]
        r = reserve.report("0000:09:00.0", 16 * GiB)
        self.assertEqual((r["floor"], r["keep"], r["notes"]), (2 * GiB, 4 * GiB, 120))
        self.assertEqual(r["measured"], 11 * GiB + reserve.HEADROOM)
        self.assertTrue(r["capped"])
        self.assertIn("lodger's own model", r["warning"])
        quiet = reserve.report("0000:08:00.0", 16 * GiB)
        self.assertEqual((quiet["keep"], quiet["measured"], quiet["warning"]), (2 * GiB, None, None))

if __name__ == "__main__":
    unittest.main()
