"""The `lodger` command: picks the right part of lodger for what you typed."""
import difflib
import sys

from lodger import __version__

COMMANDS = ("plan", "run", "measure", "stress", "serve", "ask", "pi", "unpause", "stress-serve", "doctor", "setup",
            "models", "service", "version")

HELP = """lodger: local AI that lives on your PC as a guest. Your apps come first.

  lodger plan MODEL ...     what would go where, and why
  lodger run MODEL ...      run it, fenced in and watched
  lodger measure ...        log what each app uses (read-only)
  lodger stress MODEL ...   check nothing crashes while sharing the PC
  lodger serve MODEL ...    keep a model ready for apps on this PC (steps aside, grows back)
  lodger ask "question"     ask the model lodger is serving
  lodger pi [pi options]    talk to it in pi (your pi setup, untouched)
  lodger unpause            start again any service lodger paused (after a crash)
  lodger stress-serve MODEL check serve steps aside and grows back without crashing anything
  lodger doctor             what lodger sees on this PC (reads only)
  lodger setup              download llama.cpp (Vulkan by default) into ~/.local/share/lodger
  lodger models             models to start with: what fits this PC; `lodger models get NAME` downloads one
  lodger service install    start `lodger serve` with your desktop session (systemd user unit)
  lodger --version          which lodger this is

`lodger <command> --help` for the options."""


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    cmd = args[0] if args else ""
    if cmd in ("plan", "run"):
        from lodger import launch
        return launch.main(args)
    if cmd == "measure":
        from lodger import measure
        return measure.main(args[1:])
    if cmd == "stress":
        from lodger import stress
        return stress.main(args[1:])
    if cmd == "stress-serve":
        from lodger import servetest
        return servetest.main(args[1:])
    if cmd == "doctor":
        from lodger import doctor
        return doctor.main(args[1:])
    if cmd == "setup":
        from lodger import setup
        return setup.main(args[1:])
    if cmd == "models":
        from lodger import picker
        return picker.main(args[1:])
    if cmd == "service":
        from lodger import service
        return service.main(args[1:])
    if cmd in ("--version", "-V", "version"):
        print(f"lodger {__version__}")
        return 0
    if cmd in ("serve", "ask", "pi", "unpause"):
        from lodger import serve
        return getattr(serve, {"unpause": "unpause_all"}.get(cmd, cmd))(args[1:])
    if cmd in ("", "-h", "--help", "help"):
        print(HELP)
        return 0
    near = difflib.get_close_matches(cmd, COMMANDS, n=1)
    print(f"lodger: no command {cmd!r}" + (f"; did you mean `lodger {near[0]}`?" if near else ""), file=sys.stderr)
    print(HELP)
    return 2


if __name__ == "__main__":
    sys.exit(main())
