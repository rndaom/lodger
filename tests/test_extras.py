"""The author's personal extras stay off unless config.toml turns them on."""
import os
import unittest
from unittest import mock

from lodger import config, launch, stress


def cfg(**extras):
    c = config.load(os.devnull)
    c["extras"].update(extras)
    return c


class ExtrasTest(unittest.TestCase):
    def tearDown(self):
        config.use(None)

    def test_router_off_by_default(self):
        config.use(cfg())
        with mock.patch("lodger.launch.subprocess.run", side_effect=AssertionError("systemctl called")):
            self.assertIsNone(launch.router_coming())
            self.assertFalse(stress.router_running())

    def test_router_when_on(self):
        config.use(cfg(llama_router=True))
        with mock.patch("lodger.launch.router_active", return_value=True):
            self.assertIn("llama-router is running", launch.router_coming())

    def test_pi_off_by_default(self):
        config.use(cfg())
        self.assertFalse(config.extra("pi"))
        config.use(cfg(pi=True))
        self.assertTrue(config.extra("pi"))

    def test_compositor(self):
        for desk, name in (("Hyprland", "Hyprland"), ("ubuntu:GNOME", "gnome-shell"), ("KDE", "kwin_wayland"), ("", None)):
            with mock.patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": desk}):
                self.assertEqual(stress.compositor(), name)


if __name__ == "__main__":
    unittest.main()
