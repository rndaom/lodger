import contextlib
import io
import unittest

from lodger import cli


def run(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(list(args))
    return code, out.getvalue()


class CliTest(unittest.TestCase):
    def test_help_lists_the_commands(self):
        for flag in ("", "-h", "--help", "help"):
            code, text = run(*([flag] if flag else []))
            self.assertEqual(code, 0)
            for word in ("plan", "run", "serve", "ask", "stress-serve"):
                self.assertIn(f"lodger {word}", text)

    def test_unknown_command_is_an_error(self):
        code, text = run("no-such-command")
        self.assertEqual(code, 2)
        self.assertIn("lodger serve", text)

    def test_a_typo_gets_a_suggestion(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, _ = run("serv")
        self.assertEqual(code, 2)
        self.assertIn("did you mean `lodger serve`", err.getvalue())

    def test_version(self):
        from lodger import __version__
        for flag in ("--version", "-V", "version"):
            code, text = run(flag)
            self.assertEqual((code, text.strip()), (0, f"lodger {__version__}"))

    def test_every_command_in_the_help_is_known(self):
        _, text = run("--help")
        for line in text.splitlines():
            if line.startswith("  lodger ") and not line.startswith("  lodger --"):
                self.assertIn(line.split()[1], cli.COMMANDS)


if __name__ == "__main__":
    unittest.main()
