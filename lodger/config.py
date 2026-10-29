"""lodger's settings: ~/.config/lodger/config.toml (LODGER_CONFIG points elsewhere). Every key is optional;
see systemd/config.example.toml. The model list is a separate file (models.toml, see catalog.py).

A file that isn't valid TOML, or a setting of the wrong kind (gpu = 3, extras = true), stops lodger with a
plain message saying which; a setting lodger doesn't know (a typo like gpu_on_batery) is ignored with a
warning on stderr. `lodger doctor` lists both."""
import os
import re
import sys
import tomllib

PATH = os.environ.get("LODGER_CONFIG") or os.path.join(
    os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "lodger", "config.toml")
STATE = os.path.join(os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"), "lodger")
DATA = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "lodger")

DEFAULTS = {
    "llama_cpp": None,              # llama.cpp folder (its programs in it, in bin/ or in build/bin/); env LODGER_LLAMA_CPP
    "gpu": "auto",                  # "auto", "none", or a PCI address like "0000:03:00.0"
    "desktop_reserve_mib": None,    # graphics memory always left for the desktop, at least; None: by card size
    "gpu_on_battery": False,        # start work on a graphics card while the laptop runs on battery
    "wake_gpu": False,              # use a card that is asleep (power saving), waking it
    "runs_dir": None,               # logs, learned numbers, caches; None: ~/.local/state/lodger
    "extras": {
        "llama_router": False,      # watch a llama-router service and llama-game-guard's marker (the author's PC)
        "pi": False,                # `lodger serve` on port 8181 writes a "lodger" provider into pi's config
    },
}

# what each setting may be (None: may be left unset); extras are all true/false
KINDS = {"llama_cpp": (str, None), "gpu": (str,), "desktop_reserve_mib": (int, None), "gpu_on_battery": (bool,),
         "wake_gpu": (bool,), "runs_dir": (str, None), "extras": (dict,)}
_NAMES = {str: "text in quotes", int: "a whole number", bool: "true or false", dict: "a [table]"}
GPU = re.compile(r"^(auto|none|([0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])$")

_cache = None
_warned = set()


def _kind_ok(v, kinds):
    if v is None:
        return None in kinds
    return any(k is not None and isinstance(v, k) and not (k is int and isinstance(v, bool)) for k in kinds)


def check(path=None):
    """(settings, errors, warnings) from `path` (default PATH) on top of the defaults. Errors: the file can't
    be read, or a setting has the wrong kind (that setting keeps its default); warnings: unknown settings."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    errors, warnings = [], []
    p = path or PATH
    if not os.path.exists(p):
        return cfg, errors, warnings
    try:
        with open(p, "rb") as f:
            got = tomllib.load(f)
    except tomllib.TOMLDecodeError as ex:
        return cfg, [f"{p} isn't valid TOML: {ex}"], warnings
    except OSError as ex:
        return cfg, [f"can't read {p}: {ex.strerror or ex}"], warnings
    for k, v in got.items():
        if k not in KINDS:
            warnings.append(f"{p}: unknown setting {k!r} (ignored; the settings are {', '.join(DEFAULTS)})")
        elif not _kind_ok(v, KINDS[k]):
            want = " or ".join(_NAMES[t] for t in KINDS[k] if t is not None)
            errors.append(f"{p}: {k} must be {want}, not {v!r}")
        elif k == "extras":
            for e, ev in v.items():
                if e not in DEFAULTS["extras"]:
                    warnings.append(f"{p}: unknown extra {e!r} (ignored; the extras are {', '.join(DEFAULTS['extras'])})")
                elif not isinstance(ev, bool):
                    errors.append(f"{p}: extras.{e} must be true or false, not {ev!r}")
                else:
                    cfg["extras"][e] = ev
        elif k == "gpu" and not GPU.match(v.strip().lower()):
            errors.append(f"{p}: gpu must be \"auto\", \"none\" or a PCI address like \"0000:03:00.0\", not {v!r}")
        elif k == "desktop_reserve_mib" and v is not None and v < 0:
            errors.append(f"{p}: desktop_reserve_mib can't be negative")
        else:
            cfg[k] = v
    return cfg, errors, warnings


def load(path=None):
    """Settings from `path` (default PATH) on top of the defaults. Stops with a plain message if the file is
    broken; warns once about settings lodger doesn't know."""
    cfg, errors, warnings = check(path)
    for w in warnings:
        if w not in _warned:
            _warned.add(w)
            print("lodger: warning: " + w, file=sys.stderr)
    if errors:
        raise SystemExit("lodger: problem in the settings: " + "; ".join(errors))
    return cfg


def get():
    global _cache
    if _cache is None:
        _cache = load()
    return _cache


def use(cfg):
    """Replace the settings (tests); None reads the file again on next use."""
    global _cache
    _cache = cfg


def extra(name):
    return bool(get()["extras"].get(name))


def runs_dir():
    d = get()["runs_dir"]
    return os.path.realpath(os.path.expanduser(d)) if d else STATE
