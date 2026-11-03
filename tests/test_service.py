import argparse
import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

from lodger import service

REPO_UNIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "systemd", "lodger.service")


def settings(text):
    return [l for l in text.splitlines() if l and not l.startswith("#")]


class FakeSystemd:
    """Answers `systemctl --user ...` like a running user manager; records what was asked."""

    def __init__(self, fragment="", dropins=""):
        self.calls = []
        self.fragment, self.dropins = fragment, dropins

    def __call__(self, *args):
        self.calls.append(args)
        if args[0] == "is-system-running":
            return 0, "running"
        if args[:3] == ("show", "-P", "FragmentPath"):
            return 0, self.fragment
        if args[:3] == ("show", "-P", "DropInPaths"):
            return 0, self.dropins
        if args[0] == "is-active":
            return 0, "active"
        return 0, ""


def opts(**kw):
    d = {"gpu": True, "idle": 10, "ollama": False, "force": False, "no_start": False}
    d.update(kw)
    return argparse.Namespace(**d)


class RenderTest(unittest.TestCase):
    def test_default_unit_matches_the_repos_file(self):
        text = service.render(["/home/u/.local/bin/lodger"], home="/home/u", mark=False)
        with open(REPO_UNIT) as f:
            self.assertEqual(settings(text), settings(f.read()))

    def test_options_reach_exec_start(self):
        text = service.render(["/usr/bin/lodger"], gpu=False, idle=2.5, ollama=True, home="/home/u")
        self.assertIn("ExecStart=/usr/bin/lodger serve --idle 2.5 --ollama\n", text)
        self.assertIn(service.MARK, text)
        never = service.render(["/usr/bin/lodger"], idle=0, home="/home/u")
        self.assertIn("ExecStart=/usr/bin/lodger serve --allow-gpu\n", never)
        self.assertNotIn("0 minutes", never)

    def test_exec_line_quotes_and_escapes(self):
        self.assertEqual(service.exec_line(["/home/u/.local/bin/lodger"], home="/home/u/"), "%h/.local/bin/lodger")
        self.assertEqual(service.exec_line(["/home/u/my apps/lodger"], home="/home/u"), '"%h/my apps/lodger"')
        self.assertEqual(service.exec_line(["/opt/50%/lodger"], home="/home/u"), "/opt/50%%/lodger")
        self.assertEqual(service.exec_line(["/home/user2/lodger"], home="/home/u"), "/home/user2/lodger")
        self.assertEqual(service.exec_line(["/usr/bin/python3", "-m", "lodger.cli"], home="/"),
                         "/usr/bin/python3 -m lodger.cli")

    def test_command_prefers_the_running_lodger_script(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        script = os.path.join(d, "lodger")
        open(script, "w").close()
        self.assertEqual(service.command(script, which=lambda n: "/usr/bin/lodger"), [script])
        self.assertEqual(service.command("/x/cli.py", which=lambda n: "/usr/bin/lodger"), ["/usr/bin/lodger"])
        self.assertEqual(service.command("/x/cli.py", which=lambda n: None, exe="/py"), ["/py", "-m", "lodger.cli"])


class InstallTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.home)
        env = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": self.home})
        env.start()
        self.addCleanup(env.stop)
        self.sd = FakeSystemd()
        p = mock.patch.object(service, "systemctl", self.sd)
        p.start()
        self.addCleanup(p.stop)
        c = mock.patch.object(service, "command", lambda: ["/usr/bin/lodger"])
        c.start()
        self.addCleanup(c.stop)
        self.path = os.path.join(self.home, "systemd", "user", "lodger.service")

    def run_quiet(self, fn, a):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = fn(a)
        return code, out.getvalue()

    def test_install_writes_enables_and_starts(self):
        code, _ = self.run_quiet(service.install, opts())
        self.assertEqual(code, 0)
        with open(self.path) as f:
            self.assertIn("ExecStart=/usr/bin/lodger serve --allow-gpu --idle 10", f.read())
        self.assertIn(("daemon-reload",), self.sd.calls)
        self.assertIn(("enable", "--now", "lodger.service"), self.sd.calls)

    def test_install_twice_is_quiet_and_no_start_does_not_start(self):
        self.run_quiet(service.install, opts())
        _, out = self.run_quiet(service.install, opts(no_start=True))
        self.assertIn("up to date", out)
        self.assertIn(("enable", "lodger.service"), self.sd.calls)

    def test_a_unit_lodger_did_not_write_is_left_alone(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as f:
            f.write("[Service]\nExecStart=/my/own serve\n")
        with self.assertRaises(SystemExit) as cm:
            self.run_quiet(service.install, opts())
        self.assertIn("left alone", str(cm.exception))
        with open(self.path) as f:
            self.assertIn("/my/own", f.read())
        self.run_quiet(service.install, opts(force=True))
        with open(self.path) as f:
            self.assertIn(service.MARK, f.read())

    def test_a_linked_unit_is_left_alone(self):
        target = os.path.join(self.home, "elsewhere.service")
        with open(target, "w") as f:
            f.write("[Service]\nExecStart=/x\n")
        os.makedirs(os.path.dirname(self.path))
        os.symlink(target, self.path)
        with self.assertRaises(SystemExit) as cm:
            self.run_quiet(service.install, opts())
        self.assertIn("link", str(cm.exception))
        self.run_quiet(service.install, opts(force=True))
        self.assertFalse(os.path.islink(self.path))
        with open(target) as f:
            self.assertEqual(f.read(), "[Service]\nExecStart=/x\n")     # the link's target is untouched

    def test_a_packaged_unit_is_only_enabled(self):
        self.sd.fragment = "/usr/lib/systemd/user/lodger.service"
        _, out = self.run_quiet(service.install, opts())
        self.assertFalse(os.path.exists(self.path))
        self.assertIn("package", out)
        self.assertIn(("enable", "--now", "lodger.service"), self.sd.calls)

    def test_drop_ins_are_mentioned(self):
        self.sd.dropins = "/home/u/.config/systemd/user/lodger.service.d/x.conf"
        _, out = self.run_quiet(service.install, opts())
        self.assertIn("x.conf", out)

    def test_remove_deletes_only_its_own_unit(self):
        self.run_quiet(service.install, opts())
        self.run_quiet(service.remove, opts())
        self.assertFalse(os.path.exists(self.path))
        self.assertIn(("disable", "--now", "lodger.service"), self.sd.calls)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            f.write("[Service]\nExecStart=/mine\n")
        _, out = self.run_quiet(service.remove, opts())
        self.assertTrue(os.path.exists(self.path))
        self.assertIn("stays", out)

    def test_no_user_manager_is_a_plain_message(self):
        with mock.patch.object(service, "systemctl", lambda *a: (None, "systemctl not found")):
            with self.assertRaises(SystemExit) as cm:
                service.install(opts())
        self.assertIn("autostart", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
