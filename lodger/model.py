"""What a model file holds: the exact size of every piece, grouped the way the planner places them.
Reads only the file headers, itself (read_gguf: a few milliseconds; the `gguf` package took seconds for a big
tokenizer). Split models are read from all parts."""
import bisect
import copy
import glob
import os
import re
import struct

# the experts llama.cpp's --n-cpu-moe keeps in RAM (common.h LLM_FFN_EXPS_REGEX)
EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(?:up|down|gate|gate_up)_(?:ch)?exps")
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


# ggml's tensor types: (elements per block, bytes per block), as in gguf-py's GGML_QUANT_SIZES
TYPES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
         16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
         23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
         34: (256, 54), 35: (256, 66), 39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18)}
REPACKABLE = {2, 8, 10, 12, 13, 14, 20, 39, 41}   # Q4_0 Q8_0 Q2_K Q4_K Q5_K Q6_K IQ4_NL MXFP4 Q1_0 (see plan.cpu_repacks)
SHORT_LIST = 4096            # lists in the header up to this long are kept (per-layer settings); longer ones skipped
LONG_TEXT = 1 << 20          # a text setting longer than this is skipped (a chat template is ~10 KiB) ...
NAME_MAX = 1 << 16           # ... and a setting's or tensor's name can't be longer than this
MAX_DIMS = 4                 # ggml's GGML_MAX_DIMS

# llama.cpp's sliding-window layouts by design (src/models/*.cpp, load_swa_pattern / set_swa_pattern), for files
# that give the window but not which layers use it: (period, whether the first layer of each period is the one that
# sees everything, the window llama.cpp uses when the file gives none). Period 0: every layer.
SWA_LAYOUT = {
    "gemma2": (2, False, 4096), "gemma3": (6, False, None), "gemma3n": (5, False, None),
    "gemma-embedding": (6, False, None), "gemma-embedding2": (6, False, None), "gpt-oss": (2, False, None),
    "cohere2": (4, False, None), "cohere2moe": (4, True, None), "exaone4": (4, False, 4096),
    "exaone-moe": (4, False, 128), "afmoe": (4, False, None), "llama4": (4, False, 8192), "olmo2": (4, False, None),
    "mellum": (4, False, None), "plamo3": (8, False, None), "laguna": (4, True, None), "smallthinker": (4, True, None),
    "muse-glimmer": (4, False, None), "modern-bert": (3, True, None), "deepseek4": (0, False, None),
    "deepseek41": (0, False, None),
}
NO_SWA = {"phi3"}            # llama.cpp ignores these designs' window


def read_gguf(path):
    """(settings, [(tensor name, bytes, ggml type, shape)]) from a GGUF file's header, without the gguf package: numbers, text and
    short lists of numbers (per-layer settings); the tokenizer's long lists are skipped. A tensor's size comes
    from its type and shape; for a type lodger doesn't know (or a size that doesn't fit where the next tensor
    starts), from where the next one starts, which is how much of the file it really takes."""
    import mmap
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if size else b""
    try:
        return _parse(buf, size, path)
    except (struct.error, IndexError, UnicodeDecodeError, OverflowError, RecursionError) as ex:
        raise ValueError(f"{path}: not a readable GGUF file ({type(ex).__name__}: {ex})") from None
    finally:
        if size:
            buf.close()


def _parse(buf, size, path):
    pos = 0

    def num(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, buf, pos)[0]
        pos += struct.calcsize(fmt)
        return v

    def bad(why):
        return ValueError(f"{path}: not a sensible GGUF header ({why})")

    def text(limit=NAME_MAX):
        """A text, or None if it is longer than `limit` (then skipped unread)."""
        nonlocal pos
        n = num("<Q")
        if n > size - pos:
            raise bad("a text runs past the end of the file")
        pos += n
        return bytes(buf[pos - n:pos]).decode("utf-8", "replace") if n <= limit else None

    def name():
        n = text()
        if n is None:
            raise bad(f"a name longer than {NAME_MAX} bytes")
        return n

    def value(kind):
        nonlocal pos
        if kind == 8:
            return text(LONG_TEXT)
        if kind == 9:
            sub, n = num("<I"), num("<Q")
            if sub in _KINDS and n <= SHORT_LIST:
                fmt = _KINDS[sub]
                out = list(struct.unpack_from("<" + fmt[1] * n, buf, pos))
                pos += struct.calcsize(fmt) * n
                return out
            if sub in _SIZES:
                if _SIZES[sub] * n > size - pos:
                    raise bad("a list runs past the end of the file")
                pos += _SIZES[sub] * n
            elif sub == 8:
                if 8 * n > size - pos:
                    raise bad("a list runs past the end of the file")
                for _ in range(n):
                    text(0)
            else:
                raise bad(f"a list of type {sub}")     # lists of lists: llama.cpp doesn't read them either
            return None
        if kind not in _KINDS:
            raise ValueError(f"{path}: unknown GGUF value type {kind}")
        return num(_KINDS[kind])

    if bytes(buf[:4]) != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    pos = 4
    if num("<I") < 2:
        raise ValueError(f"{path}: GGUF version 1 isn't supported")
    n_tensors, n_kv = num("<Q"), num("<Q")
    if n_kv * 12 + n_tensors * 24 > size - pos:     # the least each setting and each tensor's entry takes
        raise bad(f"{n_kv} settings and {n_tensors} tensors don't fit in the file")
    kv = {}
    for _ in range(n_kv):
        k = name()
        v = value(num("<I"))
        if v is not None:
            kv[k] = v
    align = kv.get("general.alignment")
    align = align if type(align) is int and align > 0 else 32
    infos = []
    for _ in range(n_tensors):
        t = name()
        n_dims = num("<I")
        if n_dims > MAX_DIMS:
            raise bad(f"a tensor with {n_dims} dimensions")
        dims = [num("<Q") for _ in range(n_dims)]
        kind, offset = num("<I"), num("<Q")
        infos.append((t, dims, kind, offset))
    data = (pos + align - 1) // align * align
    ends = sorted({o for _, _, _, o in infos} | {size - data})
    tensors = []
    for name, dims, kind, offset in infos:
        room = ends[bisect.bisect_right(ends, offset)] - offset if offset < ends[-1] else 0
        n = 1
        for d in dims:
            n *= d
        block, nbytes = TYPES.get(kind, (0, 0))
        exact = n // block * nbytes if block and n % block == 0 else None
        tensors.append((name, exact if exact is not None and exact <= room < exact + align else max(room, exact or 0),
                        kind, dims))
    return kv, tensors


_cache = {}


def load(path):
    """A Model, read once per change of its files (serve plans the same model again and again)."""
    path = os.path.abspath(os.path.expanduser(path))
    key = tuple((f, st.st_size, st.st_mtime_ns) for f in parts(path) for st in [os.stat(f)])
    hit = _cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    m = Model(path)
    if len(_cache) >= 32:
        _cache.clear()
    _cache[path] = (key, m)
    return m


class Model:
    """Sizes in bytes.
    layers[i] = {"dense": bytes, "experts": bytes}; also token_embd, output (on the card only when
    every layer is there), other (anything not in a layer).
    Layers that only multi-token prediction (MTP) uses are loaded by llama.cpp only when asked to: they are
    counted in mtp (same form as layers), and in layers as empty, so the layer numbers stay llama.cpp's.
    kinds[i]: what a layer keeps for the context: "attn" (grows with it), "swa" (a sliding window: fixed once
    the context is longer), "recr" (a fixed state), or None (nothing)."""
    kv, mtp, mtp_quant, kinds, quant, tied = {}, {}, (), {}, (), False    # what a model made by hand (tests) lacks

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(path))   # not realpath: split parts may be links to two drives
        self.files = parts(self.path)
        self.layers = {}
        self.token_embd = 0
        self.output = 0
        self.other = 0
        self.meta = {}
        self.kv = {}
        self.mtp = {}
        self.quant = []           # (group, layer, ggml type, rows, bytes) of weights llama.cpp may repack
        names = {}
        has_output = False
        for i, fp in enumerate(self.files):
            kv, tensors = read_gguf(fp)
            if i == 0:
                self.kv = kv
                arch = kv.get("general.architecture")
                self.meta = {
                    "arch": arch,
                    "name": kv.get("general.name"),
                    "block_count": kv.get(f"{arch}.block_count"),
                    "ctx_train": kv.get(f"{arch}.context_length"),
                    "expert_count": kv.get(f"{arch}.expert_count"),
                    "expert_used": kv.get(f"{arch}.expert_used_count"),
                }
            for n, size, kind, dims in tensors:
                group, layer = None, -1
                m = EXPERT.match(n)
                if m:
                    layer, group = int(m.group(1)), "experts"
                    self.layers.setdefault(layer, {"dense": 0, "experts": 0})["experts"] += size
                elif LAYER.match(n):
                    m = LAYER.match(n)
                    layer, group = int(m.group(1)), "dense"
                    self.layers.setdefault(layer, {"dense": 0, "experts": 0})["dense"] += size
                    names.setdefault(layer, set()).add(n[m.end():].split(".")[0])
                elif n == "token_embd.weight":
                    self.token_embd += size
                    group = "token_embd"
                elif n.startswith("output"):
                    self.output += size
                    has_output |= n == "output.weight"
                    group = "output" if n == "output.weight" else None
                else:
                    self.other += size
                if group and kind in REPACKABLE and len(dims) in (2, 3):
                    self.quant.append((group, layer, kind, dims[1], size))
        self.tied = not has_output
        if self.tied:
            # tied embeddings: llama.cpp makes a copy of token_embd for the output layer
            self.output += self.token_embd
            self.quant += [("output", -1, k, r, b) for g, _, k, r, b in self.quant if g == "token_embd"]
        blocks = self.meta["block_count"]
        self.n_layer = max([max(self.layers) + 1 if self.layers else 0] + ([blocks] if type(blocks) is int and self.layers else []))
        arch = self.meta["arch"]
        nextn = self.kv.get(f"{arch}.nextn_predict_layers")
        if type(nextn) is int and 0 < nextn < self.n_layer:
            for i in range(self.n_layer - nextn, self.n_layer):
                if i in self.layers:
                    self.mtp[i] = self.layers[i]
                    self.layers[i] = {"dense": 0, "experts": 0}
            self.mtp_quant = [q for q in self.quant if q[1] in self.mtp]
            self.quant = [q for q in self.quant if q[1] not in self.mtp]
        self.kinds = {i: None if i in self.mtp else _kind(names.get(i, ()), self._swa(i)) for i in range(self.n_layer)}

    @property
    def window(self):
        """The sliding window llama.cpp uses for this model (words), or None."""
        arch = self.meta.get("arch")
        w = self.kv.get(f"{arch}.attention.sliding_window")
        if arch in NO_SWA:
            return None
        if type(w) is int and w > 0:
            return w
        return SWA_LAYOUT.get(arch, (0, False, None))[2] if w is None else None

    def _swa(self, i):
        """Is layer i a sliding-window layer: True, False, or None (the model has sliding windows, but neither the
        file nor lodger knows which layers)."""
        if not self.window:
            return False
        pattern = self.kv.get(f"{self.meta['arch']}.attention.sliding_window_pattern")
        if isinstance(pattern, list):
            return bool(pattern[i]) if i < len(pattern) else None
        layout = SWA_LAYOUT.get(self.meta["arch"])
        n = pattern if type(pattern) is int and pattern >= 0 else layout[0] if layout else None
        if n is None:
            return None
        if n == 0:
            return True
        # llama.cpp's set_swa_pattern: one layer in n sees everything, the first or the last of each n
        return i % n != 0 if layout and layout[1] else i % n < n - 1

    def with_mtp(self):
        """This model as llama.cpp loads it for multi-token prediction drafting (--spec-type draft-mtp): its MTP
        layers' weights are loaded too (mtp still says which layers they are)."""
        m = copy.copy(self)
        m.layers = {**self.layers, **self.mtp}
        m.quant = list(self.quant) + list(self.mtp_quant)
        return m

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

    def repacked(self, rows, dense_layers, expert_layers, output):
        """(bytes, of which the output layer's) that llama.cpp copies into its own memory in a faster layout when
        these pieces run on the processor; rows: {ggml type: rows it must be a multiple of} (plan.cpu_repacks)."""
        total = out = 0
        for group, i, kind, n, size in self.quant:
            if kind not in rows or n % rows[kind]:
                continue
            if group == "output" and output:
                total, out = total + size, out + size
            elif (group == "dense" and i in dense_layers) or (group == "experts" and i in expert_layers):
                total += size
        return total, out

    def file_bytes(self):
        return sum(os.path.getsize(f) for f in self.files)


RECURRENT = ("ssm_", "shortconv", "time_mix", "conv1d")
ATTENTION = ("attn_q", "attn_k", "attn_v", "attn_kv", "wkv")


def _kind(names, swa):
    """What a layer with these tensors keeps for the context (see Model.kinds)."""
    if any(n.startswith(RECURRENT) for n in names):
        return "recr"
    if any(n.startswith(ATTENTION) for n in names):
        return "swa" if swa else "attn"     # swa None (unknown): counted like the others, as before
    return None
