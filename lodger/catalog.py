"""The models lodger serves: every .gguf file in the folders named in ~/.config/lodger/models.toml (see
that file), with names and llama-server settings from it."""
import fnmatch
import os
import re
import tomllib

CONFIG = os.environ.get("LODGER_MODELS") or os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "lodger", "models.toml")
# settings that change how much memory a model needs: llama.cpp's own estimate must see them too
MEMORY_SETTINGS = {"cache-type-k", "cache-type-v", "flash-attn", "ubatch-size", "batch-size"}
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


class Entry:
    def __init__(self, name, path, settings, skip=None):
        self.name, self.path, self.skip = name, path, skip
        self.settings = dict(settings)               # llama-server options, e.g. {"temp": 0.7}
        self.ctx = self.settings.pop("ctx", None)    # None: decided automatically
        self.reasoning = str(self.settings.get("reasoning", "on")) != "off"

    def args(self, keys=None):
        """llama-server command-line options for these settings (only `keys`, if given)."""
        out = []
        for k, v in self.settings.items():
            if keys is not None and k not in keys:
                continue
            if v is True:
                out.append("--" + k)
            elif v is not False:
                out += ["--" + k, ",".join(map(str, v)) if isinstance(v, list) else str(v)]
        return out

    def planning_args(self):
        return self.args(MEMORY_SETTINGS)


def slug(path):
    name = SPLIT.sub(".gguf", os.path.basename(path))
    return re.sub(r"[^a-z0-9.]+", "-", name[:-5].lower()).strip("-")


def load(paths=None, config=CONFIG):
    """Entries by name, in the config's order, then the files found. `paths`: serve only these files."""
    cfg = {}
    if os.path.exists(config):
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
    defaults = cfg.get("defaults", {})
    ignore = cfg.get("ignore", [])
    named = {}
    for name, sec in cfg.get("models", {}).items():
        path = os.path.abspath(os.path.expanduser(sec["file"]))
        named[path] = (name, {k: v for k, v in sec.items() if k not in ("file", "skip")}, sec.get("skip"))
    files = [os.path.abspath(os.path.expanduser(p)) for p in paths] if paths else []
    if not paths:
        files = list(named)
        for folder in cfg.get("folders", []):
            for root, dirs, names in os.walk(os.path.expanduser(folder)):
                dirs.sort()
                for n in sorted(names):
                    p = os.path.abspath(os.path.join(root, n))
                    m = SPLIT.search(n)
                    if (not n.endswith(".gguf") or (m and m.group(1) != "00001") or p in files
                            or any(fnmatch.fnmatch(p, g) for g in ignore)):
                        continue
                    files.append(p)
    out = {}
    for p in files:
        if not os.path.exists(p):
            continue
        name, own, skip = named.get(p, (slug(p), {}, None))
        out[name] = Entry(name, p, {**defaults, **own}, skip)
    return out, cfg.get("default")
