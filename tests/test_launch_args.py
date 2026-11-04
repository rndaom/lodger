import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from lodger import launch


class ModelArgTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.file = os.path.join(self.d.name, "m.gguf")
        open(self.file, "w").close()
        entry = SimpleNamespace(path=self.file, skip=None, planning_args=lambda: ["--cache-type-k", "q8_0"])
        skipped = SimpleNamespace(path=self.file, skip="unreadable", planning_args=lambda: [])
        p = mock.patch("lodger.catalog.load", return_value=({"m": entry, "bad": skipped}, "m"))
        p.start()
        self.addCleanup(p.stop)

    def test_a_file_is_taken_as_it_is(self):
        self.assertEqual(launch.model_arg(self.file), (self.file, []))

    def test_a_name_brings_its_memory_settings(self):
        self.assertEqual(launch.model_arg("m"), (self.file, ["--cache-type-k", "q8_0"]))

    def test_unknown_and_skipped_names_are_plain_errors(self):
        with self.assertRaises(SystemExit) as cm:
            launch.model_arg("nope")
        self.assertIn("names there: bad, m", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            launch.model_arg("bad")
        self.assertIn("not served: unreadable", str(cm.exception))

    def test_the_command_uses_what_was_planned(self):
        a = SimpleNamespace(think=False, prompt=None, n=8, extra=["--cache-type-k", "q8_0"])
        p = {"model": self.file, "ctx": 4096, "threads": 2, "ngl": 0, "n_cpu_moe": 0}
        with mock.patch("lodger.engine.tool", return_value="/x/llama-cli"), \
                mock.patch("lodger.engine.device_args", return_value=([], {})):
            cmd, _ = launch.llama_cmd(p, a, card=None)
        self.assertEqual(cmd[-2:], ["--cache-type-k", "q8_0"])


if __name__ == "__main__":
    unittest.main()
