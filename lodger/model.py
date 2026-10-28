"""What a model file holds: the exact size of every piece, grouped the way the planner places them.
Reads only the file headers with the `gguf` package (PyPI; or, if that isn't installed, the gguf-py folder of
the llama.cpp source tree lodger uses). Split models are read from all parts."""
import glob
import os
import re
import sys

_gguf = None
GGUF_FROM = None


def gguf_reader():
    """GGUFReader from the gguf package, or from the llama.cpp source tree lodger uses if that isn't installed.
    Loaded on first use, so lodger's other commands work without it."""
    global _gguf, GGUF_FROM
    if _gguf is None:
        try:
            from gguf import GGUFReader
            GGUF_FROM = "the gguf package"
        except ImportError:
            from . import engine
            tree = os.path.join(engine.root()[0] or "", "gguf-py")
            if not os.path.isdir(tree):
                raise SystemExit("lodger needs the gguf package to read model files: pip install gguf "
                                 "(installing lodger with pip, pipx or uv brings it)")
            sys.path.insert(0, tree)
            from gguf import GGUFReader
            GGUF_FROM = tree
        _gguf = GGUFReader
    return _gguf


EXPERT = re.compile(r"^blk\.(\d+)\.ffn_.*_(?:ch)?exps(?:\.|$)")
LAYER = re.compile(r"^blk\.(\d+)\.")


def parts(path):
    """All files of a split model (name-00001-of-00006.gguf), or just the one file."""
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    found = sorted(glob.glob(f"{glob.escape(m.group(1))}-*-of-{m.group(2)}.gguf"))
    if len(found) != int(m.group(2)):
        raise SystemExit(f"split model is missing parts: found {len(found)} of {int(m.group(2))}")
    return found


def _field(reader, key):
    f = reader.fields.get(key)
    if f is None:
        return None
    try:
        return f.contents()
    except Exception:
        return None


class Model:
    """Sizes in bytes.
    layers[i] = {"dense": bytes, "experts": bytes}; also token_embd, output (on the card only when
    every layer is there), other (anything not in a layer)."""

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(path))   # not realpath: split parts may be links to two drives
        self.files = parts(self.path)
        self.layers = {}
        self.token_embd = 0
        self.output = 0
        self.other = 0
        self.meta = {}
        has_output = False
        for i, fp in enumerate(self.files):
            r = gguf_reader()(fp, "r")
            if i == 0:
                arch = _field(r, "general.architecture")
                self.meta = {
                    "arch": arch,
                    "name": _field(r, "general.name"),
                    "block_count": _field(r, f"{arch}.block_count"),
                    "ctx_train": _field(r, f"{arch}.context_length"),
                    "expert_count": _field(r, f"{arch}.expert_count"),
                    "expert_used": _field(r, f"{arch}.expert_used_count"),
                }
            for t in r.tensors:
                n, size = t.name, int(t.n_bytes)
                m = EXPERT.match(n)
                if m:
                    self.layers.setdefault(int(m.group(1)), {"dense": 0, "experts": 0})["experts"] += size
                    continue
                m = LAYER.match(n)
                if m:
                    self.layers.setdefault(int(m.group(1)), {"dense": 0, "experts": 0})["dense"] += size
                elif n == "token_embd.weight":
                    self.token_embd += size
                elif n.startswith("output"):
                    self.output += size
                    has_output |= n == "output.weight"
                else:
                    self.other += size
            del r
        if not has_output:
            # tied embeddings: llama.cpp makes a copy of token_embd for the output layer
            self.output += self.token_embd
        self.n_layer = max(self.layers) + 1 if self.layers else 0

    @property
    def is_moe(self):
        return any(l["experts"] for l in self.layers.values())

    @property
    def total(self):
        return sum(l["dense"] + l["experts"] for l in self.layers.values()) + self.token_embd + self.output + self.other

    def dense(self, layers=None):
        ids = self.layers if layers is None else layers
        return sum(self.layers[i]["dense"] for i in ids if i in self.layers)

    def experts(self, layers=None):
        ids = self.layers if layers is None else layers
        return sum(self.layers[i]["experts"] for i in ids if i in self.layers)

    def file_bytes(self):
        return sum(os.path.getsize(f) for f in self.files)
