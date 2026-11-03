"""lodger service: start `lodger serve` with your desktop session, through a systemd user unit.

  lodger service install [--no-gpu] [--idle MIN] [--ollama] [--force] [--no-start]
  lodger service remove
  lodger service status

install writes ~/.config/systemd/user/lodger.service for the lodger you ran it with (so it works however lodger
was installed: pipx, uv tool, pip, a checkout's bin/lodger), then enables and starts it. Starting holds no
memory: the model loads on the first request and unloads after --idle minutes unused (10 by default). A unit
that a package installed (in /usr/lib/systemd/user) is used as it is: install only enables it. A unit file you
wrote or linked yourself is left alone unless you add --force.

remove stops and disables the service, and deletes the unit file if `lodger service install` wrote it."""
import argparse
import os
import re
import shutil
import subprocess
import sys

from . import config

UNIT = "lodger.service"
MARK = "# Written by `lodger service install`"

TEMPLATE = """# lodger serve: http://127.0.0.1:8181/v1 (only this PC can reach it).
# Starts with your desktop session and stops when you log out. Serves every model in
# ~/.config/lodger/models.toml, one at a time. {idle}
{mark}[Unit]
Description=lodger: local AI that steps aside for your apps
PartOf=graphical-session.target
After=graphical-session.target

[Service]
ExecStart={exe} serve{flags}
Environment=PYTHONUNBUFFERED=1
# lodger stops its model itself on SIGTERM; give it time to
TimeoutStopSec=40
Restart=on-failure
RestartSec=30

[Install]
WantedBy=graphical-session.target
"""


def unit_dir():
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "systemd", "user")


def unit_path():
    return os.path.join(unit_dir(), UNIT)


def command(argv0=None, which=shutil.which, exe=None):
    """The lodger command to put in the unit: the one running now when it is a `lodger` script (bin/lodger
    in a checkout, ~/.local/bin/lodger), else `lodger` on PATH, else this Python with `-m lodger.cli`.
    Links are kept as they are, so an upgrade that moves the real file doesn't break the unit."""
    argv0 = sys.argv[0] if argv0 is None else argv0
    if os.path.basename(argv0) == "lodger" and os.path.isfile(argv0):
        return [os.path.abspath(argv0)]
    found = which("lodger")
    if found:
        return [os.path.abspath(found)]
    return [exe or sys.executable, "-m", "lodger.cli"]


def exec_line(cmd, home=None):
    """ExecStart's command: quoted where needed, % doubled (systemd specifiers), the home folder as %h."""
    home = (os.path.expanduser("~") if home is None else home).rstrip("/")
    out = []
    for i, part in enumerate(cmd):
        lead = ""
        if i == 0 and home and part.startswith(home + "/"):
            lead, part = "%h", part[len(home):]
        part = lead + part.replace("%", "%%")
        if not part or re.search(r"[\s\"'\\;]", part):
            part = '"' + part.replace("\\", "\\\\").replace('"', '\\"') + '"'
        out.append(part)
    return " ".join(out)


def render(cmd, gpu=True, idle=10, ollama=False, mark=True, home=None):
    flags = (" --allow-gpu" if gpu else "") + (f" --idle {idle:g}" if idle else "") + (" --ollama" if ollama else "")
    said = (f"A model loads on the first request for it and unloads\n# after {idle:g} minutes unused, so lodger "
            "holds no memory while you don't use it." if idle else "A model stays loaded once a request loaded it.")
    return TEMPLATE.format(exe=exec_line(cmd, home), flags=flags, idle=said,
                           mark=(MARK + "; `lodger service remove` deletes it.\n") if mark else "")


def systemctl(*args, run=subprocess.run):
    """(exit code, output) of `systemctl --user ARGS`; (None, why) if there is no systemctl."""
    if not shutil.which("systemctl"):
        return None, "systemctl not found"
    try:
        r = run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as ex:
        return None, str(ex)
    return r.returncode, (r.stdout + r.stderr).strip()


def user_manager():
    """Whether a systemd user manager answers (None if yes, else why not)."""
    code, out = systemctl("is-system-running")
    if code is None:
        return out
    if out.splitlines() and out.splitlines()[0].strip() in ("running", "degraded", "starting", "initializing", "maintenance"):
        return None
    return out or "no systemd user session"


def fragment():
    """Where systemd reads lodger.service from now ('' if it doesn't know one)."""
    code, out = systemctl("show", "-P", "FragmentPath", UNIT)
    return out.strip() if code == 0 else ""


def ours(path):
    try:
        with open(path) as f:
            return MARK in f.read()
    except OSError:
        return False


def install(a):
    why = user_manager()
    if why:
        raise SystemExit(f"lodger service needs a systemd user session ({why}). Without one, start `lodger serve"
                         f"{' --allow-gpu' if a.gpu else ''}` from your desktop's autostart instead.")
    path = unit_path()
    frag = fragment()
    if frag and os.path.realpath(frag) != os.path.realpath(path) and not os.path.lexists(path):
        print(f"Using the installed unit {frag} (from a package) as it is.")
    else:
        text = render(command(), a.gpu, a.idle, a.ollama)
        if os.path.lexists(path) and not a.force:
            if os.path.islink(path) or not ours(path):
                raise SystemExit(f"{path} is already there and `lodger service install` didn't write it"
                                 f"{' (it is a link to ' + os.path.realpath(path) + ')' if os.path.islink(path) else ''}: "
                                 "it is left alone. --force replaces it.")
        os.makedirs(unit_dir(), exist_ok=True)
        old = None
        if os.path.isfile(path):
            with open(path) as f:
                old = f.read()
        if old != text:
            tmp = path + ".new"
            with open(tmp, "w") as f:
                f.write(text)
            if os.path.islink(path):
                os.remove(path)
            os.replace(tmp, path)
            print(f"Wrote {path}")
        else:
            print(f"{path} is up to date")
        code, out = systemctl("daemon-reload")
        if code:
            raise SystemExit(f"systemctl --user daemon-reload failed: {out}")
    code, drop = systemctl("show", "-P", "DropInPaths", UNIT)
    if code == 0 and drop:
        print(f"Note: drop-in files change this unit and win over it: {drop}")
    code, out = systemctl("enable", *([] if a.no_start else ["--now"]), UNIT)
    if code:
        raise SystemExit(f"systemctl --user enable failed: {out}")
    code, active = systemctl("is-active", UNIT)
    print(f"lodger serve starts with your desktop session; now {active or 'unknown'}.")
    if not a.no_start and active == "active":
        print("If it was already running with other settings: systemctl --user restart lodger")
    print("Logs: journalctl --user -u lodger -f")
    return 0


def remove(a):
    why = user_manager()
    if why:
        raise SystemExit(f"no systemd user session ({why}): nothing to remove")
    systemctl("disable", "--now", UNIT)
    path = unit_path()
    if os.path.isfile(path) and not os.path.islink(path) and ours(path):
        os.remove(path)
        systemctl("daemon-reload")
        print(f"Stopped, disabled and removed {path}")
    elif os.path.lexists(path):
        print(f"Stopped and disabled; {path} was not written by `lodger service install`, so it stays")
    else:
        print("Stopped and disabled" + (f" ({fragment()} stays: a package installed it)" if fragment() else ""))
    return 0


def status(a):
    why = user_manager()
    if why:
        print(f"no systemd user session ({why})")
        return 1
    _, enabled = systemctl("is-enabled", UNIT)
    _, active = systemctl("is-active", UNIT)
    frag = fragment()
    print(f"{UNIT}: {enabled or 'not installed'}, {active or 'unknown'}" + (f" ({frag})" if frag else ""))
    code, out = systemctl("show", "-P", "ExecStart", UNIT)
    if code == 0 and out:
        argv = out.split("argv[]=", 1)[-1].split(" ;", 1)[0] if "argv[]=" in out else out
        print("runs: " + argv)
    return 0 if active == "active" else 3


def main(argv):
    ap = argparse.ArgumentParser(prog="lodger service", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install", help="write, enable and start the user unit")
    i.add_argument("--no-gpu", dest="gpu", action="store_false", help="processor only (default: use the card while it's free)")
    i.add_argument("--idle", type=float, default=10, metavar="MIN", help="unload a model after MIN minutes unused (0: never; default 10)")
    i.add_argument("--ollama", action="store_true", help="also answer Ollama's API on 127.0.0.1:11434")
    i.add_argument("--force", action="store_true", help="replace a lodger.service that lodger didn't write")
    i.add_argument("--no-start", action="store_true", help="enable it, but start it only at the next login")
    sub.add_parser("remove", help="stop and disable it; delete the unit if lodger wrote it")
    sub.add_parser("status", help="installed, enabled, running?")
    a = ap.parse_args(argv)
    if a.cmd == "install" and a.idle < 0:
        ap.error("--idle can't be negative")
    config.get()                                     # a broken config.toml stops here, not in the service
    return {"install": install, "remove": remove, "status": status}[a.cmd](a)
