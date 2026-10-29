"""What a model file holds: the exact size of every piece, grouped the way the planner places them.
Reads only the file headers with the `gguf` package (PyPI; or, if that isn't installed, the gguf-py folder of
the llama.cpp source tree lodger uses). Split models are read from all parts."""
import glob
import os
import re
import struct
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
    found = sorted(glob.glob(f"{glob.escape(m.group(1))}-{'[0-9]' * 5}-of-{m.group(2)}.gguf"))
    if [int(f[-19:-14]) for f in found] != list(range(1, int(m.group(2)) + 1)):     # 00001 .. N, each once
        raise SystemExit(f"split model is missing parts: found {len(found)} of {int(m.group(2))}")
    return found


_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # GGUF value types -> bytes


_KINDS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def header(path, keys=64, until=None):
    """A GGUF file's plain settings (numbers and text, not lists) from its header, reading only the first few
    and stopping at `until` or at the tokenizer's (long lists); {} if the file isn't GGUF or it isn't there.
    Needs no gguf package (the catalog uses it)."""
    def num(f, fmt):
        b = f.read(struct.calcsize(fmt))
        return struct.unpack(fmt, b)[0]

    def text(f):
        n = num(f, "<Q")
        if n > 1 << 16:
            raise ValueError("not a sensible GGUF header")
        return f.read(n).decode("utf-8", "replace")

    def skip(f, kind):
        if kind == 8:
            f.seek(num(f, "<Q"), 1)
        elif kind == 9:
            sub, n = num(f, "<I"), num(f, "<Q")
            if sub in _SIZES:
                f.seek(_SIZES[sub] * n, 1)
            else:
                for _ in range(n):
                    skip(f, sub)
        else:
            f.seek(_SIZES[kind], 1)
    out = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return out
            wide = "<I" if num(f, "<I") == 1 else "<Q"          # version 1 counted in 32 bits
            num(f, wide)                                        # number of tensors
            for _ in range(min(num(f, wide), keys)):
                k, kind = text(f), num(f, "<I")
                if k.startswith("tokenizer."):
                    break
                if kind == 8:
                    out[k] = text(f)
                elif kind in _KINDS:
                    out[k] = num(f, _KINDS[kind])
                else:
                    skip(f, kind)
                if k == until:
                    break
    except (OSError, struct.error, KeyError, ValueError, RecursionError):
        pass
    return out


def architecture(path, keys=64):
    """general.architecture from a GGUF file's header ("clip" for a vision projector); None if the file isn't
    GGUF or it isn't there."""
    v = header(path, keys, until="general.architecture").get("general.architecture")
    return v if isinstance(v, str) else None


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
