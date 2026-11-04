"""Decide where each piece of a model goes, from live free memory minus margins.
Every number is in bytes. Each choice records a plain-words reason in plan["why"]."""
import bisect
import contextlib
import hashlib
import json
import os
import re
import signal
import threading
import time

from . import config, engine, fence, host, reserve, sysmem
from . import gpu as gpus
from . import model as models
from .model import Model

MiB, GiB = sysmem.MiB, sysmem.GiB
AUTO = object()                  # make(card=AUTO): the card lodger chose (gpus.choose)

# Margins (see NOTES.md "How lodger decides"). Graphics memory kept for the desktop: reserve.py.
EST_MARGIN = 0.10                # our estimate can be this much too low ...
EST_MARGIN_FIXED = 256 * MiB     # ... plus this
HIP_OVERHEAD = 512 * MiB         # HIP runtime, memory pools, staging (grew ~0.35 GiB at runtime on 2026-10-25)
RAM_KEEP_FREE_MIN = 4 * GiB      # RAM left for everything else: at least this ...
RAM_KEEP_FREE_FRAC = 0.15        # ... or this share of all RAM
PROC_OVERHEAD = 512 * MiB        # the program itself, tokenizer, buffers outside llama.cpp's estimate
MIN_CTX = 4096
SMALLEST_CTX = 256               # shorter than this isn't a usable context (and 0 would never end the ladder)
# Context when you don't name one: as long as the model can take, up to CTX_CAP, as long as it costs no
# more than CTX_SHARE of the memory the model may use there (the rest is better spent on the model itself).
CTX_CAP = 131072
CTX_SHARE = 0.25
MMPROJ_SCRATCH = GiB             # a vision projector's working space for pictures, when its file doesn't say
PICTURE_TOKENS = 4096            # the most words a picture becomes in llama.cpp's projectors that take any size,
PICTURE_TOKENS_BY_TYPE = {"youtuvl": 62500, "hunyuanvl": 16384, "deepseek4v": 384, "deepseek41v": 1024}  # or these
CHECKPOINTS = 32                 # llama-server's --ctx-checkpoints: copies of the state a model can't roll back
UBATCH = 512                     # llama.cpp's default --ubatch-size
DRAFT_MAX = 3                    # llama.cpp's default --spec-draft-n-max
# --load-mode values that keep the weights in the program's own memory instead of the page cache (older builds:
# --no-mmap, --mlock): then Linux can't drop them again, so they must fit in RAM
HELD = {"none", "mlock", "mmap+mlock", "dio"}
PROBE_KEYS = ("model", "context", "compute")
PROBE_TIMEOUT = 120              # llama-fit-params answers in well under a second on the processor, seconds with a card


def cache_dir():
    return os.path.join(config.runs_dir(), "probe-cache")


def llama_errors(text, most=8, no_card=False):
    """llama.cpp's own error lines in its output (log level E, an exception's what()), the last few; "" if none.
    no_card: leave out the backends finding no card (expected when the card is hidden)."""
    errs = [l.strip() for l in text.splitlines()
            if (re.match(r"\s*[\d.]+ E ", l) or l.lstrip().startswith(("what():", "error:")))
            and not (no_card and re.search(r"ggml_\w+_init:", l))]
    return "\n".join(dict.fromkeys(errs[-most:]))


def probe(model, ctx, gpu=None, extra=(), dev=None):
    """Ask llama.cpp what this model needs, without loading it (llama-fit-params --fit-print).
    gpu=None: processor only, in a unit with no device files, so the graphics card is never touched.
    gpu=(ngl, n_cpu_moe): llama.cpp's own estimate for the card `dev` with that placement. This starts the
    card's driver briefly (no big allocations), so only when the graphics card is allowed.
    extra: llama.cpp options that change memory use (e.g. --cache-type-k q8_0).
    Returns {model, context, compute} in bytes for the device. Cached (per llama.cpp build)."""
    cache = cache_dir()
    os.makedirs(cache, exist_ok=True)
    fit = engine.tool("llama-fit-params")
    if gpu is not None:
        dev = dev or sysmem.dgpu()
        dargs, denv = engine.device_args(gpu[0], dev, gpu[1])
        name = dargs[1]
    # every part of the model and every library of the build: a changed one means a new estimate
    key = hashlib.sha1((f"{model.path}:{stamp(model.files)}:{ctx}:{gpu}" + (f":{list(extra)}" if extra else "")
                        + f":{fit}:{stamp([fit] + build_libs(fit))}" + (f":{name}" if gpu is not None else "")).encode()).hexdigest()[:16]
    cf = os.path.join(cache, key + ".json")
    hit = read_cached(cf)
    if hit:
        return hit
    props = {"MemoryMax": "4G", "MemorySwapMax": "0", "LimitCORE": "0"}
    args = [fit, "-m", model.files[0], "-c", str(ctx), "--fit-print", "on"] + list(extra)
    if gpu is None:
        dargs, denv = engine.device_args(0, None)
        props["PrivateDevices"] = "yes"
        args += dargs
        prefix = "Host "
    else:
        args += dargs
        prefix = name + " "                     # --fit-print lines start with the device's name
    started = time.monotonic()
    r = fence.run(args, dict(denv, GGML_NO_BACKTRACE="1"), props, timeout=PROBE_TIMEOUT)  # no debugger on a crash
    # the device's line: model, context and compute in MiB (llama.cpp's common_fit_print)
    row = re.compile(re.escape(prefix) + r"\s*(\d+)\s+(\d+)\s+(\d+)(?:\s|$)")
    found = next((m for l in (r.stdout + "\n" + r.stderr).splitlines() if (m := row.match(l))), None)
    if r.returncode or not found:
        if r.returncode and time.monotonic() - started >= PROBE_TIMEOUT - 1:
            why = f"didn't answer within {PROBE_TIMEOUT} s and was stopped"
        elif r.returncode and r.returncode < 0:
            why = f"crashed ({signal.Signals(-r.returncode).name if -r.returncode in signal.valid_signals() else r.returncode})"
        elif r.returncode:
            why = f"failed (exit {r.returncode})"
        else:
            why = f"gave no estimate for {prefix.strip()} (its --fit-print output isn't what lodger reads)"
        tail = llama_errors(r.stderr, no_card=gpu is None) or (r.stderr.strip()[-2000:] + ("\n" + r.stdout.strip()[-500:] if not found
                                                                         else "")).strip()
        raise SystemExit(f"llama-fit-params {why} for {os.path.basename(model.files[0])}" + (f":\n{tail}" if tail else "."))
    out = {k: int(v) * MiB for k, v in zip(PROBE_KEYS, found.groups())}
    tmp = f"{cf}.{os.getpid()}-{threading.get_ident()}.tmp"     # two plans may probe at once
    try:
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, cf)                     # never a half-written file under the real name
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(tmp)
    return out


def stamp(paths):
    """Size and change time of each file: what a cached probe depends on."""
    out = []
    for p in paths:
        st = os.stat(p)
        out.append(f"{st.st_size}:{st.st_mtime}")
    return ",".join(out)


def build_libs(tool):
    """The llama.cpp libraries next to a program (or in ../lib): most of a build's code lives in them."""
    d = os.path.dirname(tool)
    out = []
    for lib in (d, os.path.join(os.path.dirname(d), "lib")):
        with contextlib.suppress(OSError):
            out += sorted(os.path.join(lib, n) for n in os.listdir(lib)
                          if n.startswith("lib") and ".so" in n and os.path.isfile(os.path.join(lib, n)))
    return out


def read_cached(path):
    """A cached probe result, or None if there is none or it is damaged (then llama.cpp is asked again)."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not all(type(d.get(k)) is int and d[k] >= 0 for k in PROBE_KEYS):
        return None
    return {k: d[k] for k in PROBE_KEYS}


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def learned_overhead(pci=None, backend=None):
    """What the card really used beyond llama.cpp's own estimate in recent runs (HIP runtime, memory pools,
    staging). Until there is data: HIP_OVERHEAD. Runs that say which card (pci) and engine backend they used
    count only for that card and backend; older runs that don't say count for every one. A damaged line is
    skipped on its own (stopping at it would drop the newest runs, and with them the highest overhead)."""
    path = os.path.join(config.runs_dir(), "calibration.jsonl")
    seen = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(d, dict) or not (_number(d.get("llama_gpu_est")) and _number(d.get("gpu_real"))):
                    continue
                if (pci and d.get("pci", pci) != pci) or (backend and d.get("backend", backend) != backend):
                    continue
                seen.append(d["gpu_real"] - d["llama_gpu_est"])
    except OSError:
        pass
    if not seen:
        return HIP_OVERHEAD, "no runs measured yet"
    worst = max(seen[-5:])
    return min(1536 * MiB, max(256 * MiB, worst + 128 * MiB)), f"learned from {len(seen[-5:])} recent runs"


class Costs:
    """Context memory grows in a straight line with the context length (plus a fixed part, e.g. the
    state of recurrent layers), so two probes give it for any length. Sliding-window layers grow only up to
    their window: then llama.cpp is also asked at the bend (`points`), and lengths between are read off the
    straight pieces."""

    def __init__(self, model, lo, hi, extra=(), points=()):
        self.lo, self.hi = lo, hi
        got = {c: probe(model, c, extra=extra) for c in sorted({lo, hi} | {p for p in points if lo < p < hi})}
        a, b = got[lo], got[hi]
        grow = b["context"] - a["context"]
        self.per_tok = grow / (hi - lo) if hi > lo and grow > 0 else 0.0
        # no growth measured: the larger of the two, so neither length is planned below llama.cpp's own number
        self.base = a["context"] - self.per_tok * lo if self.per_tok else max(a["context"], b["context"])
        self.compute = max(g["compute"] for g in got.values())
        self.points = sorted((c, g["context"]) for c, g in got.items()) if len(got) > 2 else None
        if self.points:
            # what grows with every word once the windows are full: the slope of the last piece
            (c1, v1), (c2, v2) = self.points[-2:]
            self.per_tok = max(0.0, (v2 - v1) / (c2 - c1))

    draft_tok = 0.0                 # what a multi-token prediction draft context adds per word (add_draft) ...
    draft_compute = 0               # ... its scratch ...
    snapshots = 0                   # ... and the recurrent state's copies for rolling back a draft

    def context(self, ctx):
        return self._trunk(ctx) + self.draft(ctx)

    def draft(self, ctx):
        """The part of context(ctx) that drafting adds."""
        return int(self.draft_tok * ctx) + self.snapshots

    def _trunk(self, ctx):
        if not self.points:
            return int(self.base + self.per_tok * ctx)
        pts = self.points
        i = max(0, min(len(pts) - 2, bisect.bisect_right([c for c, _ in pts], ctx) - 1))
        (c1, v1), (c2, v2) = pts[i], pts[i + 1]
        return max(v1, v2) if v1 >= v2 else int(v1 + (v2 - v1) * (ctx - c1) / (c2 - c1))

    def fixed(self, ctx):
        """The part of context(ctx) that doesn't grow with more words (recurrent state, full sliding windows)."""
        return max(0, self._trunk(ctx) - int(self.per_tok * ctx))

    def add_draft(self, m, n_max=DRAFT_MAX):
        """Multi-token prediction drafting (--spec-type draft-mtp): llama-server makes a second context as long as
        the main one, with a plain attention cache for the MTP layers only and its own scratch (llama.cpp's
        common/speculative.cpp), and a model with recurrent layers keeps n_max more copies of their state to roll
        a draft back (common.cpp: n_rs_seq = the draft length; llama-memory-recurrent.cpp). llama-fit-params
        sizes none of it."""
        kinds = list(m.kinds.values())
        attn = kinds.count("attn") or sum(1 for k in kinds if k) or max(1, m.n_layer - len(m.mtp))
        self.draft_tok = len(m.mtp) * self.per_tok / attn
        self.draft_compute = self.compute
        self.compute *= 2
        self.snapshots = n_max * self.fixed(self.hi) if "recr" in kinds else 0


def window_bend(m, extra):
    """The context length where a model's sliding windows stop growing (llama.cpp: window + ubatch, in steps
    of 256), or None."""
    w = m.window
    if not w or "--swa-full" in extra:
        return None
    ub = UBATCH
    args = list(extra)
    for i, a in enumerate(args[:-1]):
        if a == "--ubatch-size" and str(args[i + 1]).isdigit():
            ub = int(args[i + 1])
    return (w + ub + 255) // 256 * 256


def layer_ctx(m, costs, ctx):
    """Each layer's part of the context memory at `ctx` (they add up to costs.context(ctx)): what grows with the
    context belongs to the layers that see all of it, the fixed rest (recurrent state, sliding windows) to the
    layers that keep it. Where the file doesn't say which layers those are, each layer that keeps anything gets
    the same (layers that keep nothing, e.g. multi-token prediction, none)."""
    L = m.n_layer
    total = costs.context(ctx) - costs.draft(ctx)
    kinds = m.kinds or {}
    grows = [i for i in range(L) if kinds.get(i) == "attn"]
    keeps = [i for i in range(L) if kinds.get(i) in ("swa", "recr")]
    out = dict.fromkeys(range(L), 0.0)
    if not grows and not keeps:
        grows = list(range(L))
    if not grows or not keeps:
        for i in grows or keeps:
            out[i] = total / len(grows or keeps)
    else:
        tok = min(total, costs.per_tok * ctx)
        for i in grows:
            out[i] = tok / len(grows)
        for i in keeps:
            out[i] = (total - tok) / len(keeps)
    for i, part in draft_ctx(m, costs, ctx).items():
        out[i] += part
    return out


def draft_ctx(m, costs, ctx):
    """Each layer's part of what drafting adds to the context (costs.draft): the draft cache with the MTP layers,
    the state's copies with the recurrent layers."""
    out = {}
    for i in m.mtp if costs.draft_tok else ():
        out[i] = out.get(i, 0) + costs.draft_tok * ctx / len(m.mtp)
    recr = [i for i, k in (m.kinds or {}).items() if k == "recr"]
    for i in recr if costs.snapshots else ():
        out[i] = out.get(i, 0) + costs.snapshots / len(recr)
    return out


def cpu_repacks():
    """The weights llama.cpp copies into a faster layout in its own memory when they run on this processor
    (ggml-cpu/repack.cpp): {ggml type: rows a tensor must be a multiple of}. Such copies can't be dropped and read
    again like the model file's pages, so they count as the model's own memory."""
    flags, arm = set(), False
    for line in (host.read("/proc/cpuinfo") or "").splitlines():
        k, _, v = line.partition(":")
        if k.strip() in ("flags", "Features"):
            flags |= set(v.split())
            arm |= k.strip() == "Features"
    if "avx2" in flags:                                 # Q4_0 Q4_K IQ4_NL MXFP4, and Q2_K with AVX-512
        return {2: 8, 12: 8, 20: 8, 39: 8} | ({10: 8} if "avx512f" in flags else {})
    if arm and flags & {"asimddp", "i8mm"}:             # Q4_0 Q8_0 Q1_0 IQ4_NL MXFP4 Q4_K Q5_K Q6_K
        return {2: 4, 8: 4, 41: 4, 20: 4, 39: 4, 12: 8, 13: 8, 14: 8}
    return {}


def physical_cores():
    cores = set()
    base = "/sys/devices/system/cpu"
    for d in os.listdir(base):
        if d.startswith("cpu") and d[3:].isdigit():
            try:
                with open(f"{base}/{d}/topology/core_id") as f, open(f"{base}/{d}/topology/physical_package_id") as g:
                    cores.add((g.read().strip(), f.read().strip()))
            except OSError:
                pass
    return len(cores) or os.cpu_count() or 4


def ladder(target, lo, ctx_train):
    """Context lengths to try, longest first: `target` halved step by step, ending at exactly `lo`, never
    longer than the model was trained for."""
    if target < SMALLEST_CTX or lo < SMALLEST_CTX:
        raise ValueError(f"a context shorter than {SMALLEST_CTX} words isn't usable (asked for {min(target, lo)})")
    top = min(target, int(ctx_train)) if ctx_train and int(ctx_train) > 0 else target
    lo = min(lo, top)
    out, c = [], top
    while c > lo:
        out.append(c)
        c //= 2
    return out + [lo]


def own_costs(extra):
    """llama-server options that change memory but that llama-fit-params can't size (see catalog.py) are
    counted here. Returns {fit: options for llama-fit-params, proj: vision projector bytes, proj_work: its working
    memory, proj_card: projector on the card, held: weights held in RAM, repack: weights may be repacked,
    checkpoints: --ctx-checkpoints, mtp: multi-token prediction drafting, moe_cache: --moe-cache-mib in bytes}."""
    out = {"fit": [], "proj": 0, "proj_work": 0, "proj_card": True, "held": False, "repack": True,
           "checkpoints": CHECKPOINTS, "mtp": False, "draft_max": DRAFT_MAX, "moe_cache": 0, "no_host": False}
    proj, tokens, fa = None, None, True
    args = iter(extra)
    for a in args:
        if a == "--mmproj":
            proj = next(args)
            out["proj"] = os.path.getsize(proj)
        elif a == "--image-max-tokens":
            tokens = int(next(args))
        elif a == "--spec-type":
            out["mtp"] = "draft-mtp" in str(next(args)).split(",")
        elif a == "--spec-draft-n-max":
            out["draft_max"] = max(0, int(next(args)))
        elif a == "--moe-cache-mib":
            out["moe_cache"] = max(0, int(next(args))) * MiB
        elif a in ("--mmproj-offload", "--no-mmproj-offload"):
            out["proj_card"] = a == "--mmproj-offload"
        elif a in ("--no-mmap", "--mlock"):
            out["held"] = True
        elif a == "--mmap":
            pass
        elif a == "--ctx-checkpoints":
            out["checkpoints"] = max(0, int(next(args)))
        elif a == "--load-mode":
            v = next(args)
            out["held"] = v in HELD
            out["fit"] += [a, v]
        else:
            if a in ("--repack", "--no-repack"):
                out["repack"] = a == "--repack"
            out["no_host"] |= a == "--no-host"
            if a == "--flash-attn":
                v = next(args)
                fa = str(v).lower() not in ("off", "0", "false", "disabled")
                out["fit"] += [a, v]
                continue
            out["fit"].append(a)
    if proj:
        out["proj_work"] = projector_work(proj, tokens, fa)
    return out


def projector_work(path, tokens=None, flash_attn=True):
    """A vision projector's working memory for the largest picture it takes: clip.cpp sizes it for a test picture
    and grows it for a bigger one. About 4 bytes x (3 x its width + its ffn width) per patch with flash attention
    (on the processor 120 MiB at 4096 patches and 248 MiB at 8464 for Qwen3-VL's, within 2 % of this), plus a
    patches x patches table per head without. Half again on top (measured on the processor, not yet on a card),
    at least 256 MiB; MMPROJ_SCRATCH when the file doesn't say enough. tokens: --image-max-tokens."""
    try:
        kv, _ = models.read_gguf(path)
    except (OSError, ValueError):
        return MMPROJ_SCRATCH

    def num(k):
        v = kv.get("clip.vision." + k)
        return v if type(v) is int and v > 0 else None
    width, ffn, heads, size, patch = (num(k) for k in ("embedding_length", "feed_forward_length",
                                                      "attention.head_count", "image_size", "patch_size"))
    if not kv.get("clip.has_vision_encoder") or not all((width, ffn, heads, size, patch)) or kv.get("clip.has_audio_encoder"):
        return MMPROJ_SCRATCH
    merge = num("spatial_merge_size") or num("projector.scale_factor") or 1
    kind = kv.get("clip.projector_type")
    if tokens and tokens > 0:
        patches = tokens * merge * merge
    elif kind in PICTURE_TOKENS_BY_TYPE:                # clip.cpp's own limit for these, whatever the file says
        patches = PICTURE_TOKENS_BY_TYPE[kind] * merge * merge
    elif num("image_max_pixels"):
        patches = num("image_max_pixels") // (patch * patch)
    elif merge > 1:
        patches = PICTURE_TOKENS * merge * merge       # pictures of any size, up to llama.cpp's usual limit
    else:
        patches = (size // patch) ** 2                 # every picture is scaled to image_size
    work = patches * (3 * width + ffn) * 4 + (0 if flash_attn else heads * patches * patches * 4)
    return max(256 * MiB, int(1.5 * work))


def with_margin(x):
    return int(x * (1 + EST_MARGIN) + EST_MARGIN_FIXED)


def make(model_path, target_ctx=None, min_ctx=MIN_CTX, cpu_only=False, leave_gpu=0,
         vram_free=None, ram_avail=None, threads=None, why_gpu_off=None, device_check=False, extra=(),
         card=AUTO, mem=None, keep=None):
    """Return a plan dict, or one with plan["refuse"] set to the reason.
    target_ctx: context to aim for; None = automatic (see CTX_CAP, CTX_SHARE).
    device_check: also ask llama.cpp for its own estimate on the card (starts the card's driver briefly).
    extra: llama-server options that change memory (catalog.Entry.planning_args()); see own_costs for the ones
    lodger counts itself.
    card: the graphics card (default: the one lodger chose; None: there is none); mem: /proc/meminfo numbers;
    keep: graphics memory to leave for the desktop (default: reserve.py). model_path may also be a Model."""
    m = model_path if isinstance(model_path, Model) else models.load(model_path)
    L = m.n_layer
    why = []

    def refused(reason):
        return {"model": m.path, "name": m.meta.get("name"), "why": why, "refuse": reason}
    if not L:
        return refused(f"lodger can't place this model: it has no layers it knows ({m.meta.get('arch') or 'unknown'} design)")
    leave_gpu = max(0, leave_gpu)
    own_extra = own_costs(extra)
    extra, proj, proj_card, held = (own_extra[k] for k in ("fit", "proj", "proj_card", "held"))
    moe_cache = own_extra["moe_cache"]
    if own_extra["mtp"]:
        if not m.mtp:
            return refused("its settings ask for multi-token prediction drafting (spec-type draft-mtp), but this model "
                           "file has no MTP layers")
        mtp_bytes = sum(l["dense"] + l["experts"] for l in m.mtp.values())
        m = m.with_mtp()
    if moe_cache:
        if not m.is_moe:
            return refused("its settings ask for an expert cache (moe-cache-mib), but this isn't a mixture-of-experts model")
        least = min_moe_cache(m)
        if moe_cache < least:
            return refused(f"its expert cache (moe-cache-mib {moe_cache // MiB}) is too small for one word's experts: "
                           f"llama.cpp needs at least {-(-least // MiB)}")
    card = sysmem.dgpu() if card is AUTO else card
    mem = sysmem.meminfo() if mem is None else mem
    cfg = config.get()
    live = vram_free is None                    # not a --pretend plan
    backend = None
    if card is None:
        no_card = gpus.why_none(cfg["gpu"])
    elif live and card.asleep() and not cfg["wake_gpu"]:
        no_card = "the graphics card is asleep (power saving); lodger doesn't wake it (wake_gpu in config.toml)"
    elif live and not card.integrated and gpus.power_source()[0] == "battery" and not cfg["gpu_on_battery"]:
        no_card = "the PC is on battery; lodger starts work on the graphics card only when plugged in (gpu_on_battery)"
    elif live and card.free(meminfo=mem) is None:
        no_card = card.why_unknown() or "its free memory can't be read"
    else:
        t = engine.for_device(card)
        no_card, backend = t.get("why"), t.get("backend")
    if live:
        vram_free = (card.free(meminfo=mem) or 0) if card else 0
    ram_avail = mem["MemAvailable"] if ram_avail is None else ram_avail
    ram_keep = max(RAM_KEEP_FREE_MIN, int(mem["MemTotal"] * RAM_KEEP_FREE_FRAC))
    ram_budget = ram_avail - ram_keep
    shared = bool(card and card.integrated)
    if shared:
        # built-in graphics take their memory from RAM: never more than RAM can spare
        vram_free = max(0, min(vram_free, ram_budget))
    keep_why = None
    if keep is None:
        keep, keep_why = reserve.reserve(card.pci, card.total(meminfo=mem)) if card else (reserve.floor(0), None)
    vram_budget = vram_free - keep - leave_gpu

    auto = target_ctx is None
    ctxs = ladder(CTX_CAP if auto else target_ctx, min(min_ctx, CTX_CAP if auto else target_ctx), m.meta.get("ctx_train"))
    bend = window_bend(m, extra)
    costs = Costs(m, ctxs[-1], ctxs[0], extra, [bend] if bend else ())
    if own_extra["mtp"]:
        costs.add_draft(m, own_extra["draft_max"])

    def unseen(g):
        """What llama-fit-params doesn't know about on the card with placement g: MTP drafting's layers, cache,
        state copies and scratch (it isn't told --spec-type)."""
        if not own_extra["mtp"]:
            return 0
        on = range(L - g["ngl_layers"], L)
        weights = sum(m.mtp[i]["dense"] + (m.mtp[i]["experts"] if i >= g["n_cpu_moe"] else 0) for i in m.mtp if i in on)
        return weights + int(sum(v for i, v in draft_ctx(m, costs, g["ctx"]).items() if i in on)) + costs.draft_compute

    def fair(budget):
        """Automatic context: the lengths that cost at most CTX_SHARE of `budget` (at least the shortest)."""
        if not auto:
            return ctxs
        return [c for c in ctxs if with_margin(costs.context(c)) <= CTX_SHARE * budget] or ctxs[-1:]

    plan = {
        "model": m.path, "name": m.meta.get("name"), "arch": m.meta.get("arch"),
        "layers": L, "moe": m.is_moe, "model_bytes": m.total,
        "vram_free": vram_free, "vram_budget": vram_budget, "ram_avail": ram_avail, "ram_budget": ram_budget,
        "keep": keep, "shared_memory": shared,
        "threads": threads or max(2, physical_cores() - 2),
        "why": why, "refuse": None,
    }
    why.append(f"Model: {m.meta.get('name')} ({m.meta.get('arch')}), {m.total / GiB:.1f} GiB, {L} layers"
               + (f", mixture of experts ({m.meta.get('expert_used')} of {m.meta.get('expert_count')} experts per word)"
                  if m.is_moe else "") + ".")
    why.append(f"Free right now: {vram_free / GiB:.1f} GiB on the graphics card, {ram_avail / GiB:.1f} GiB RAM."
               if not shared else f"Free right now: {ram_avail / GiB:.1f} GiB RAM, shared by the processor and its built-in "
               f"graphics; the graphics can take {vram_free / GiB:.1f} GiB of it (what RAM can spare, within its GTT limit).")
    if keep_why:
        why.append(f"Desktop reserve: {keep_why}.")
    why.append(f"Kept back: {keep / GiB:.1f} GiB of graphics memory for the desktop"
               + (f" + {leave_gpu / GiB:.1f} GiB you asked to leave" if leave_gpu else "")
               + f", {ram_keep / GiB:.1f} GiB RAM for everything else. "
               f"Our estimates get +{EST_MARGIN:.0%} + {EST_MARGIN_FIXED // MiB} MiB extra on top.")
    trained = m.meta.get("ctx_train")
    if not (type(trained) is int and trained > 0):
        why.append("The model file doesn't say how long a context it was trained for; lodger doesn't limit it to that.")
    elif not auto and target_ctx > trained:
        why.append(f"Context at most {trained} words, what this model was trained for (asked for {target_ctx}).")

    if own_extra["mtp"]:
        why.append(f"Multi-token prediction drafting (spec-type draft-mtp): its {len(m.mtp)} MTP layer(s) are loaded "
                   f"too ({mtp_bytes / GiB:.1f} GiB), with a second context as long as the main one and its own "
                   "scratch, counted.")
    # a vision projector (mmproj) is loaded whole, on the card with the model unless told otherwise, and needs
    # room for the pictures; llama-fit-params doesn't count it
    proj_need = proj + own_extra["proj_work"] if proj else 0
    if proj:
        why.append(f"Vision projector: {proj / GiB:.1f} GiB + {own_extra['proj_work'] / GiB:.1f} GiB for pictures, counted "
                   + ("on the graphics card when the model is there, else in RAM." if proj_card else "in RAM (no-mmproj-offload)."))
    gpu = None
    if cpu_only or no_card or why_gpu_off:
        why.append("Graphics card not used: " + (why_gpu_off or ("you asked for processor only" if cpu_only else no_card)) + ".")
    else:
        overhead, src = learned_overhead(card.pci, backend)
        more = overhead + (proj_need if proj_card else 0)      # on the card besides llama.cpp's own estimate
        budget, w = vram_budget, []
        gpu = place_on_gpu(m, costs, fair(budget), budget, w, more, moe_cache)
        for _ in range(3):
            if not (gpu and device_check):
                break
            d = probe(m, gpu["ctx"], (gpu["ngl"], gpu["n_cpu_moe"]), extra, dev=card)
            llama = d["model"] + d["context"] + d["compute"]
            plan["llama_gpu_est"] = llama
            beside = more + (moe_cache if gpu["n_cpu_moe"] else 0) + unseen(gpu)
            if llama + beside <= gpu["gpu_est"]:
                break                                   # our own estimate is the higher one: keep it
            short = with_margin(llama + beside) - gpu["gpu_bytes"]
            if with_margin(llama + beside) <= vram_budget:   # the real limit, not the shrunk working budget
                gpu.update(gpu_est=llama + beside, gpu_bytes=with_margin(llama + beside))
                break
            budget -= short                             # llama.cpp says more than we thought: plan smaller
            w = [f"llama.cpp's own estimate for the card was {short // MiB} MiB higher than ours; planned smaller."]
            gpu = place_on_gpu(m, costs, fair(vram_budget), budget, w, more, moe_cache)
        else:
            if gpu and device_check:
                w.append("Couldn't get our plan and llama.cpp's estimate to agree; not using the graphics card.")
                gpu = None
        why.extend(w)
        if gpu and moe_cache and gpu["n_cpu_moe"]:
            why.append(f"Expert cache (moe-cache-mib): {moe_cache // MiB} MiB on the card for experts kept in RAM, counted.")
        if gpu:
            why.append(f"Card estimate: " + (f"llama.cpp says {plan['llama_gpu_est'] / GiB:.1f} GiB" if "llama_gpu_est" in plan
                       else "our own (llama.cpp asked only for real runs)")
                       + f", plus {overhead // MiB} MiB for the card's runtime ({src}).")
        plan["overhead"] = overhead

    if gpu is None:
        plan.update(ngl=0, ngl_layers=0, n_cpu_moe=0, ctx=None, output_on_gpu=False, gpu_bytes=0, gpu_est=0)
        if moe_cache:
            plan["refuse"] = ("Its settings ask for an expert cache on the graphics card (moe-cache-mib), but the card "
                              "isn't used for this model; llama.cpp would stop ('MoE cache requires a GPU backend').")
            return plan
    else:
        plan.update(gpu)
        if shared:
            ram_budget -= gpu["gpu_bytes"]          # what the built-in graphics take is gone from RAM
            plan["ram_budget"] = ram_budget
            why.append(f"The built-in graphics' part ({gpu['gpu_bytes'] / GiB:.1f} GiB) comes out of RAM; "
                       f"{max(0, ram_budget) / GiB:.1f} GiB RAM is left for the rest.")
    on_cpu_layers = set(range(L - plan["ngl_layers"]))
    experts_on_cpu = on_cpu_layers | set(range(plan["n_cpu_moe"]))
    out_cpu = not plan["output_on_gpu"]
    cpu_weights = (m.token_embd + m.other + m.dense(on_cpu_layers) + m.experts(experts_on_cpu)
                   + (m.output if out_cpu else 0))

    # RAM: weights not on the card are read from the model file through the page cache. Linux can drop
    # those pages again at any time, so they can't crash anything (if they don't fit they are read from
    # the SSD again: slow, not dangerous). Memory llama.cpp allocates itself can't be dropped: that part
    # must fit or we refuse. Weights held in RAM (no-mmap, mlock) and a projector off the card count as that,
    # and so do weights llama.cpp repacks for the processor (a copy in a faster layout).
    if held:
        copied, cached = cpu_weights, 0
    else:
        # with a card, the weights left in RAM go to its host buffer first, which mmap swaps for the file's pages
        # (llama-model.cpp make_cpu_buft_list, llama-model-loader.cpp): only a processor-only run, or one with
        # --no-host, repacks them
        repacks = cpu_repacks() if own_extra["repack"] and (gpu is None or own_extra["no_host"]) else {}
        copied, copied_out = m.repacked(repacks, on_cpu_layers, experts_on_cpu, out_cpu)
        # a tied output layer reads token_embd's own pages (unless it is copied): counted once
        cached = cpu_weights - copied - (m.token_embd - copied_out if m.tied and out_cpu else 0)
    # llama-server's context checkpoints (one per answer, up to --ctx-checkpoints) copy what a model can't roll
    # back, its recurrent state or sliding windows, into RAM
    # (llama-server makes them for any model with a sliding window, whichever layers use it, and keeps one more
    # for a draft when it can't roll a draft back otherwise)
    keeps = m.window or any(k in ("swa", "recr") for k in m.kinds.values())
    n_ckpt = own_extra["checkpoints"] + (1 if own_extra["mtp"] and m.window else 0) if keeps else 0

    def ckpt(c):
        """One checkpoint at context c: the recurrent state and the sliding windows. A window longer than the
        context holds all of it, which the probes (all shorter than the window) can't tell from growth."""
        f = costs.fixed(c)
        if bend and bend > ctxs[0]:
            kinds = list(m.kinds.values())
            swa, attn = kinds.count("swa"), kinds.count("attn")
            f += int(costs.per_tok * c * (swa / (swa + attn) if swa else 1))
        return f

    def own_at(c):
        return (PROC_OVERHEAD + (proj_need if gpu is None or not proj_card else 0) + copied
                + n_ckpt * ckpt(c))

    def less(c):
        """What a setting in models.toml would take off the model's own memory, in words."""
        ways = ([f"no-repack = true leaves out the {copied / GiB:.1f} GiB of repacked weights (the processor gets "
                 "slower)"] if copied and not held else [])
        ways += ([f"a smaller ctx-checkpoints (now {n_ckpt}) saves {ckpt(c) / GiB:.2f} GiB per checkpoint"]
                 if n_ckpt and ckpt(c) else [])
        return (" In models.toml, " + "; ".join(ways) + ".") if ways else ""
    if held:
        why.append(f"The weights kept in RAM ({cpu_weights / GiB:.1f} GiB) are held there (no-mmap or mlock), "
                   "so they count as the model's own memory.")
    elif copied:
        why.append(f"llama.cpp copies {copied / GiB:.1f} GiB of the weights the processor runs into a faster layout "
                   "(repack) in its own memory: counted as the model's own.")
    if gpu is None:
        def need(c):
            return costs.context(c) + costs.compute + own_at(c)
        ctx = next((c for c in fair(ram_budget) if need(c) <= ram_budget), None)
        if ctx is None:
            plan["refuse"] = (f"Not enough RAM even on the processor: context, scratch and the model's own copies "
                              f"need {need(ctxs[-1]) / GiB:.1f} GiB, "
                              f"{max(0, ram_budget) / GiB:.1f} GiB can be spared." + less(ctxs[-1]))
            return plan
        plan["ctx"] = ctx
        anon = need(ctx)
        why.append(f"Running on the processor only, context {ctx} words.")
    else:
        share = layer_ctx(m, costs, plan["ctx"])
        cpu_ctx_share = sum(share[i] for i in on_cpu_layers) if plan["ngl_layers"] < L else 0
        anon = int(cpu_ctx_share) + (costs.compute if plan["n_cpu_moe"] or plan["ngl_layers"] < L else 0) + own_at(plan["ctx"])
    if n_ckpt and ckpt(plan["ctx"]):
        why.append(f"Context checkpoints: llama-server keeps up to {n_ckpt} copies of the state this model can't roll "
                   f"back ({ckpt(plan['ctx']) / MiB:.0f} MiB each) in RAM, counted as the model's own.")
    if auto:
        why.append(f"Context chosen automatically: {plan['ctx']} words (this model can take up to "
                   f"{m.meta.get('ctx_train') or '?'}; lodger aims for at most {CTX_CAP} and lets the context use "
                   f"at most {CTX_SHARE:.0%} of the memory it may take, so the model itself keeps the rest).")
    plan["ram_anon_est"] = anon
    plan["ram_weights"] = cpu_weights
    plan["ram_copied"] = copied
    plan["checkpoint_bytes"] = n_ckpt * ckpt(plan["ctx"])
    if gpu and plan["n_cpu_moe"] and not moe_cache:
        # what llama.cpp's expert cache (--moe-cache-mib) could use on the card here; not passed on (yet)
        room = int((vram_budget - gpu["gpu_bytes"]) / (1 + EST_MARGIN)) // MiB
        plan["moe_cache_room_mib"] = room if room * MiB >= min_moe_cache(m) else 0

    if with_margin(anon) > ram_budget:
        plan["refuse"] = (f"Not enough RAM: the model needs {with_margin(anon) / GiB:.1f} GiB of its own memory "
                          f"(with margin) and only {max(0, ram_budget) / GiB:.1f} GiB can be spared." + less(plan["ctx"]))
        return plan
    room_for_weights = ram_budget - with_margin(anon)
    if cached > room_for_weights:
        plan["slow"] = True
        why.append(f"Weights kept in RAM: {cached / GiB:.1f} GiB, but only {room_for_weights / GiB:.1f} GiB fits "
                   "next to your apps. The rest is read from the SSD again as needed: safe, but slower.")
    else:
        plan["slow"] = False
        why.append(f"Weights kept in RAM: {cached / GiB:.1f} GiB (fits).")
    # RAM fence: own memory + as much of the weights as we can spare; never more than the budget
    plan["ram_cap"] = max(with_margin(anon), min(ram_budget, with_margin(anon) + cached + GiB))
    # llama-server keeps earlier conversations in RAM to answer them faster later (8 GiB by default, counted
    # nowhere else): only in the room the fence has left over, so it never pushes the model over its cap
    plan["cache_ram_mib"] = max(0, min(8192, (plan["ram_cap"] - with_margin(anon) - cached) // MiB))
    why.append(f"RAM fence for the model: {plan['ram_cap'] / GiB:.1f} GiB, no swap. "
               f"Processor threads: {plan['threads']} (leaves cores for your apps).")
    return plan


def place_on_gpu(m, costs, ctxs, budget, why, overhead, moe_cache=0):
    """Best placement on the card, or None. Order of preference (NOTES.md step 4). overhead: what the card holds
    beyond the model's pieces, context and scratch (its runtime, a vision projector). moe_cache: llama.cpp's
    expert cache, made on the card only when some experts stay in RAM."""
    L = m.n_layer
    # 1. everything, longest context first
    for c in ctxs:
        need = m.dense() + m.experts() + m.output + m.other + costs.context(c) + costs.compute + overhead
        if with_margin(need) <= budget:
            why.append(f"Everything fits on the graphics card with a {c}-word context "
                       f"({with_margin(need) / GiB:.1f} GiB of {budget / GiB:.1f} GiB usable, margin included).")
            return gpu_plan(L, L, 0, c, need, True)
    # 2. dense parts + context on the card, experts of as many layers as fit (the last layers first:
    #    llama.cpp's --n-cpu-moe N keeps the experts of the FIRST N layers in RAM)
    if m.is_moe:
        for c in ctxs:
            fixed = m.dense() + m.output + m.other + costs.context(c) + costs.compute + overhead + moe_cache
            if with_margin(fixed) > budget:
                continue
            k, need = 0, fixed
            while k < L and with_margin(need + m.layers.get(L - 1 - k, {"experts": 0})["experts"]) <= budget:
                need += m.layers.get(L - 1 - k, {"experts": 0})["experts"]
                k += 1
            why.append(f"Too big for the card as a whole. On the card: every layer's dense part, the {c}-word context"
                       f" and the experts of {k} of {L} layers ({with_margin(need) / GiB:.1f} GiB with margin). "
                       f"Experts of the other {L - k} layers stay in RAM and the processor runs them; each word "
                       "only uses a few experts, so this costs less speed than it sounds.")
            if c != ctxs[0]:
                why.append(f"Context shortened from {ctxs[0]} to {c} words to make room.")
            return gpu_plan(L, L, L - k, c, need, True)
    # 3. fewer whole layers on the card (all experts in RAM). llama.cpp's -ngl N puts the output layer on the
    #    card first, then the last N - 1 layers
    for c in ctxs:
        share = layer_ctx(m, costs, c)
        need = costs.compute + overhead + m.other + m.output + (moe_cache if m.is_moe else 0)
        # (pieces outside the layers: counted on the card to be safe)
        if with_margin(need) > budget:
            continue
        n = 0
        while n < L:
            i = L - 1 - n
            add = m.layers.get(i, {"dense": 0})["dense"] + share[i]
            if with_margin(need + add) > budget:
                break
            need += add
            n += 1
        if n >= max(1, L // 2) or c == ctxs[-1]:
            if n < max(1, L // 4):
                why.append(f"Only {n} of {L} layers would fit on the graphics card: too little to be worth "
                           "taking memory from your apps.")
                return None
            why.append(f"Only {n} of {L} layers fit on the card, with the output layer (context {c} words, "
                       f"{with_margin(need) / GiB:.1f} GiB with margin); the rest runs on the processor.")
            return gpu_plan(L, n, L if m.is_moe else 0, c, int(need), True)
    why.append(f"Nothing useful fits on the graphics card ({budget / GiB:.1f} GiB usable after margins).")
    return None


def min_moe_cache(m):
    """The smallest --moe-cache-mib llama.cpp takes: room for the experts one word uses in one layer, plus one
    (llama.cpp's llama-moe-cache.cpp)."""
    n, used = m.meta.get("expert_count"), m.meta.get("expert_used")
    one = max((l["experts"] for l in m.layers.values()), default=0)
    if type(n) is not int or n <= 0 or type(used) is not int or used <= 0:
        return 0
    return -(-one // n) * (used + 1)


def gpu_plan(L, n_layers, n_cpu_moe, ctx, need, output_on_gpu):
    return {
        "ngl": n_layers + 1 if output_on_gpu else n_layers,   # -ngl counts the output layer as one more
        "ngl_layers": n_layers,
        "n_cpu_moe": n_cpu_moe,
        "ctx": ctx,
        "output_on_gpu": output_on_gpu,
        "gpu_est": need,
        "gpu_bytes": with_margin(need),
    }
