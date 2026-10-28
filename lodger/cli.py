"""The `lodger` command: picks the right part of lodger for what you typed."""
import sys

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
    if cmd in ("serve", "ask", "pi", "unpause"):
        from lodger import serve
        return getattr(serve, {"unpause": "unpause_all"}.get(cmd, cmd))(args[1:])
    print(HELP)
    return 0 if cmd in ("", "-h", "--help", "help") else 2


if __name__ == "__main__":
    sys.exit(main())
