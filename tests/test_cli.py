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


if __name__ == "__main__":
    unittest.main()
