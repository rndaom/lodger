"""The models lodger serves: every .gguf file in the folders named in ~/.config/lodger/models.toml (see
that file), with names and llama-server settings from it."""
import fnmatch
import os
import re
import tomllib

from .model import architecture

CONFIG = os.environ.get("LODGER_MODELS") or os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "lodger", "models.toml")
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
SMALLEST_CTX = 256                                   # plan.SMALLEST_CTX

# llama-server's other names (llama.cpp common/arg.cpp) for the options lodger looks at, so a setting is
# known whichever name it is written with; it is passed on under the long name
ALIASES = {
    "c": "ctx-size", "ngl": "n-gpu-layers", "gpu-layers": "n-gpu-layers", "ncmoe": "n-cpu-moe", "cmoe": "cpu-moe",
    "t": "threads", "tb": "threads-batch", "m": "model", "a": "alias", "np": "parallel", "fitt": "fit-target",
    "fitc": "fit-ctx", "fitp": "fit-print", "dev": "device", "sm": "split-mode", "ts": "tensor-split", "mg": "main-gpu",
    "ot": "override-tensor", "mu": "model-url", "hf": "hf-repo", "hfr": "hf-repo", "hff": "hf-file", "hft": "hf-token",
    "dr": "docker-repo", "ui": "webui", "no-ui": "no-webui", "mmdev": "mmproj-device", "mmu": "mmproj-url",
    "md": "model-draft", "spec-draft-model": "model-draft", "hfd": "hf-repo-draft",
    "hfrd": "hf-repo-draft", "spec-draft-hf": "hf-repo-draft", "nkvo": "no-kv-offload", "kvo": "kv-offload",
    "ctk": "cache-type-k", "ctv": "cache-type-v", "fa": "flash-attn", "ub": "ubatch-size", "b": "batch-size",
    "nr": "no-repack", "cram": "cache-ram", "mm": "mmproj", "lm": "load-mode", "lzm": "lazy-mode", "kvu": "kv-unified",
    "no-kvu": "no-kv-unified", "rea": "reasoning", "ctkd": "cache-type-k-draft", "ctvd": "cache-type-v-draft",
    "otd": "override-tensor-draft", "cmoed": "cpu-moe-draft", "ncmoed": "n-cpu-moe-draft", "devd": "device-draft",
    "ngld": "n-gpu-layers-draft", "td": "threads-draft", "tbd": "threads-batch-draft",
}
# set by lodger itself for every run (its plan, its card, its server); a model's settings can't change them
OWNED = {"n-gpu-layers", "n-cpu-moe", "cpu-moe", "host", "port", "api-key", "api-key-file", "threads",
         "threads-batch", "model", "alias", "fit", "fit-target", "fit-ctx", "fit-print", "parallel", "device",
         "split-mode", "tensor-split", "main-gpu", "override-tensor", "rpc", "model-url", "hf-repo", "hf-file",
         "hf-token", "docker-repo", "webui", "no-webui", "models-dir", "models-preset", "mmproj-device", "mmproj-url",
         "cache-ram"}
# change memory in ways lodger can't plan yet: a second model, extra weights, or the context moved to RAM
UNPLANNED = {"model-draft": "a draft model (speculative decoding)", "hf-repo-draft": "a draft model (speculative decoding)",
             "spec-type": "speculative decoding", "lora": "LoRA adapters", "lora-scaled": "LoRA adapters",
             "control-vector": "control vectors", "control-vector-scaled": "control vectors",
             "override-kv": "changed model settings (override-kv)", "no-kv-offload": "the context kept in RAM (no-kv-offload)"}
# change memory, and llama.cpp's own estimate (llama-fit-params) sizes them: the planner passes them on
PLANNED = {"cache-type-k", "cache-type-v", "flash-attn", "ubatch-size", "batch-size", "swa-full", "no-host", "repack",
           "no-repack", "op-offload", "no-op-offload", "kv-offload", "load-mode", "lazy-mode"}
# change memory, but llama-fit-params can't size them: plan.own_costs counts them itself
COUNTED = {"mmproj", "mmproj-offload", "no-mmproj-offload", "no-mmap", "mlock", "mmap"}
MEMORY_SETTINGS = PLANNED | COUNTED


def name_of(key):
    """The long llama-server name of a setting written as `key` (with or without dashes, or a short name)."""
    k = str(key).lstrip("-")
    return ALIASES.get(k, k)


def problem(settings, ctx):
    """Why lodger can't run a model with these settings (plain words), or None."""
    if ctx is not None and (type(ctx) is not int or ctx < SMALLEST_CTX):
        return f"ctx must be a whole number of words, at least {SMALLEST_CTX} (it is {ctx!r})"
    for k, v in settings.items():
        if v is False:
            continue                                 # not passed on at all
        if k == "ctx-size":
            return "its settings set ctx-size; write ctx = N instead, so lodger plans with it"
        if k in OWNED:
            return f"its settings set {k}, which lodger sets itself for every run; remove it from models.toml"
        if k in UNPLANNED or k.startswith("spec-draft-") or k.endswith("-draft"):
            return (f"its settings ask for {UNPLANNED.get(k, 'a draft model (speculative decoding)')} ({k}), which "
                    "lodger can't plan memory for yet")
        if k == "mmproj" and not (isinstance(v, str) and os.path.isfile(v)):
            return f"its vision projector (mmproj) isn't there: {v}"
        if v is True and k in ("cache-type-k", "cache-type-v", "ubatch-size", "batch-size", "load-mode", "lazy-mode", "mmproj"):
            return f"its setting {k} needs a value, not true"
    return None


class Entry:
    def __init__(self, name, path, settings, skip=None):
        self.name, self.path = name, path
        self.settings = {}                           # llama-server options by long name, e.g. {"temp": 0.7}
        for k, v in dict(settings).items():
            k = k if k == "ctx" else name_of(k)
            self.settings[k] = os.path.abspath(os.path.expanduser(v)) if k == "mmproj" and isinstance(v, str) else v
        self.ctx = self.settings.pop("ctx", None)    # None: decided automatically
        self.skip = skip or problem(self.settings, self.ctx)
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
        """The settings that change memory, for plan.make(extra=...)."""
        return self.args(MEMORY_SETTINGS)


def slug(path):
    name = SPLIT.sub(".gguf", os.path.basename(path))
    return re.sub(r"[^a-z0-9.]+", "-", name[:-5].lower()).strip("-")


def listed(cfg, key):
    """A list setting of models.toml; one string counts as a list of one."""
    v = cfg.get(key, [])
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"{key} in models.toml must be a list of strings, like {key} = [\"...\"]")
    return v


def load(paths=None, config=CONFIG):
    """Entries by name, in the config's order, then the files found. `paths`: serve only these files.
    A name used twice (the same file name in two folders, or a file named like a [models.NAME] section)
    stays with the first; the later one is listed as name-2 (-3, ...) and not served, with why."""
    cfg = {}
    if os.path.exists(config):
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
    defaults = cfg.get("defaults", {})
    ignore = listed(cfg, "ignore")
    named = {}
    for name, sec in cfg.get("models", {}).items():
        if not isinstance(sec, dict) or not isinstance(sec.get("file"), str):
            raise ValueError(f'[models.{name}] in models.toml needs file = "path/to/model.gguf"')
        path = os.path.abspath(os.path.expanduser(sec["file"]))
        named[path] = (name, {k: v for k, v in sec.items() if k not in ("file", "skip")}, sec.get("skip"))
    files = [os.path.abspath(os.path.expanduser(p)) for p in paths] if paths else []
    if not paths:
        files = list(named)
        for folder in listed(cfg, "folders"):
            for root, dirs, names in os.walk(os.path.expanduser(folder)):
                dirs.sort()
                for n in sorted(names):
                    p = os.path.abspath(os.path.join(root, n))
                    m = SPLIT.search(n)
                    if (not n.endswith(".gguf") or (m and m.group(1) != "00001") or p in files
                            or any(fnmatch.fnmatch(p, g) for g in ignore) or architecture(p) == "clip"):
                        continue
                    files.append(p)
    out, first = {}, {}
    for p in files:
        if not os.path.exists(p):
            continue
        name, own, skip = named.get(p, (slug(p), {}, None))
        if architecture(p) == "clip":
            skip = skip or "it is a vision projector, not a model: give it to its model as mmproj = \"...\" instead"
        if name in out:
            n = 2
            while f"{name}-{n}" in out:
                n += 1
            skip = skip or f"its name {name} is already used by {first[name]}; give it its own [models.NAME] section"
            name = f"{name}-{n}"
        first.setdefault(name, p)
        out[name] = Entry(name, p, {**defaults, **own}, skip)
    return out, cfg.get("default")
