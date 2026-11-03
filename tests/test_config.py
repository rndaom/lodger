"""Settings: wrong kinds and broken files stop lodger with a plain message; unknown names warn."""
import contextlib
import io
import os
import tempfile
import unittest

from lodger import config


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "config.toml")
        config._warned.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text):
        with open(self.path, "w") as f:
            f.write(text)

    def test_defaults_and_good_file(self):
        self.assertEqual(config.load(os.devnull), config.DEFAULTS)
        self.write('gpu = "0000:03:00.0"\ndesktop_reserve_mib = 2048\nwake_gpu = true\n[extras]\npi = true\n')
        c = config.load(self.path)
        self.assertEqual((c["gpu"], c["desktop_reserve_mib"], c["wake_gpu"], c["extras"]["pi"]), ("0000:03:00.0", 2048, True, True))
        self.assertFalse(c["extras"]["llama_router"])

    def test_wrong_kinds_stop_with_a_plain_message(self):
        for text, word in (("gpu = 3\n", "gpu must be text"), ("extras = true\n", "extras must be a [table]"),
                           ("wake_gpu = \"yes\"\n", "wake_gpu must be true or false"),
                           ("desktop_reserve_mib = true\n", "desktop_reserve_mib must be a whole number"),
                           ("[extras]\npi = 1\n", "extras.pi must be true or false"),
                           ('gpu = "atuo"\n', "gpu must be \"auto\"")):
            self.write(text)
            with self.assertRaises(SystemExit) as ex:
                config.load(self.path)
            self.assertIn(word, str(ex.exception))
            self.assertIn(self.path, str(ex.exception))

    def test_broken_toml(self):
        self.write("gpu = \n")
        with self.assertRaises(SystemExit) as ex:
            config.load(self.path)
        self.assertIn("isn't valid TOML", str(ex.exception))

    def test_unknown_names_warn_once(self):
        self.write("gpu_on_batery = true\n[extras]\nrouter = true\n")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            c = config.load(self.path)
            config.load(self.path)
        self.assertFalse(c["gpu_on_battery"])
        self.assertNotIn("gpu_on_batery", c)
        self.assertEqual(err.getvalue().count("gpu_on_batery"), 1)
        self.assertIn("unknown extra 'router'", err.getvalue())

    def test_check_lists_without_stopping(self):
        self.write("gpu = 3\nwake_gpu = true\n")
        c, errors, warnings = config.check(self.path)
        self.assertEqual(c["gpu"], "auto")                  # the broken one keeps its default
        self.assertTrue(c["wake_gpu"])
        self.assertEqual(len(errors), 1)

    def test_paths_must_be_full(self):
        self.write('runs_dir = "runs"\nllama_cpp = "~/llama.cpp"\n')
        c, errors, _ = config.check(self.path)
        self.assertEqual(len(errors), 1)
        self.assertIn("runs_dir must be a full path", errors[0])
        self.assertIsNone(c["runs_dir"])
        self.assertEqual(c["llama_cpp"], "~/llama.cpp")


if __name__ == "__main__":
    unittest.main()
