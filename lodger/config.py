"""lodger's settings: ~/.config/lodger/config.toml (LODGER_CONFIG points elsewhere). Every key is optional;
see systemd/config.example.toml. The model list is a separate file (models.toml, see catalog.py)."""
import os
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

_cache = None


def load(path=None):
    """Settings from `path` (default PATH) on top of the defaults."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    p = path or PATH
    if os.path.exists(p):
        with open(p, "rb") as f:
            got = tomllib.load(f)
        for k, v in got.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
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
