"""Decide where each piece of a model goes, from live free memory minus margins.
Every number is in bytes. Each choice records a plain-words reason in plan["why"]."""
import hashlib
import json
import os
import subprocess

from . import sysmem
from .model import LLAMA_CPP, Model

MiB, GiB = sysmem.MiB, sysmem.GiB
BIN = os.path.join(LLAMA_CPP, "build", "bin")
CACHE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "runs", "probe-cache")

# Margins (see NOTES.md "How the launcher decides")
VRAM_KEEP_FREE = 2048 * MiB      # never plan to leave less than this free on the card
EST_MARGIN = 0.10                # our estimate can be this much too low ...
EST_MARGIN_FIXED = 256 * MiB     # ... plus this
HIP_OVERHEAD = 512 * MiB         # HIP runtime, memory pools, staging (grew ~0.35 GiB at runtime on 2026-10-25)
RAM_KEEP_FREE_MIN = 4 * GiB      # RAM left for everything else: at least this ...
RAM_KEEP_FREE_FRAC = 0.15        # ... or this share of all RAM
PROC_OVERHEAD = 512 * MiB        # the program itself, tokenizer, buffers outside llama.cpp's estimate
MIN_CTX = 4096
# Context when you don't name one: as long as the model can take, up to CTX_CAP, as long as it costs no
# more than CTX_SHARE of the memory the model may use there (the rest is better spent on the model itself).
CTX_CAP = 131072
CTX_SHARE = 0.25


def probe(model, ctx, gpu=None, extra=()):
    """Ask llama.cpp what this model needs, without loading it (llama-fit-params --fit-print).
    gpu=None: processor only, in a unit with no device files, so the graphics card is never touched.
    gpu=(ngl, n_cpu_moe): llama.cpp's own estimate for the RX 7800 XT with that placement. This starts the
    card's driver briefly (no big allocations), so only when the graphics card is allowed.
    extra: llama.cpp options that change memory use (e.g. --cache-type-k q8_0).
    Returns {model, context, compute} in bytes for the device. Cached."""
    os.makedirs(CACHE, exist_ok=True)
    st = os.stat(model.files[0])
    key = hashlib.sha1((f"{model.path}:{st.st_size}:{st.st_mtime}:{ctx}:{gpu}" + (f":{list(extra)}" if extra else "")).encode()).hexdigest()[:16]
    cf = os.path.join(CACHE, key + ".json")
    if os.path.exists(cf):
        with open(cf) as f:
            return json.load(f)
    unit = ["systemd-run", "--user", "--quiet", "--collect", "--wait", "--pipe",
            "-p", "MemoryMax=4G", "-p", "MemorySwapMax=0", "-p", "LimitCORE=0"]
    args = [os.path.join(BIN, "llama-fit-params"), "-m", model.files[0], "-c", str(ctx), "--fit-print", "on"] + list(extra)
    if gpu is None:
        unit += ["-p", "PrivateDevices=yes", "--setenv=HIP_VISIBLE_DEVICES=-1", "--setenv=ROCR_VISIBLE_DEVICES=-1"]
        args += ["--device", "none", "-ngl", "0"]
        dev = "Host "
    else:
        unit += ["--setenv=HIP_VISIBLE_DEVICES=0"]
        args += ["--device", "ROCm0", "-ngl", str(gpu[0])] + (["-ncmoe", str(gpu[1])] if gpu[1] else [])
        dev = "ROCm0 "
    r = subprocess.run(unit + args, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith(dev)), None)
    if r.returncode or not line:
        raise SystemExit(f"llama-fit-params failed (exit {r.returncode}):\n{r.stderr[-2000:]}")
    _, mdl, ctx_mib, comp_mib = line.split()
    out = {"model": int(mdl) * MiB, "context": int(ctx_mib) * MiB, "compute": int(comp_mib) * MiB}
    with open(cf, "w") as f:
        json.dump(out, f)
    return out


def learned_overhead():
    """What the card really used beyond llama.cpp's own estimate in recent runs (HIP runtime, memory pools,
    staging). Until there is data: HIP_OVERHEAD."""
    path = os.path.join(CACHE, "..", "calibration.jsonl")
    seen = []
    try:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("llama_gpu_est") and d.get("gpu_real"):
                    seen.append(d["gpu_real"] - d["llama_gpu_est"])
    except (OSError, ValueError):
        pass
    if not seen:
        return HIP_OVERHEAD, "no runs measured yet"
    worst = max(seen[-5:])
    return min(1536 * MiB, max(256 * MiB, worst + 128 * MiB)), f"learned from {len(seen[-5:])} recent runs"


class Costs:
    """Context memory grows in a straight line with the context length (plus a fixed part, e.g. the
    state of recurrent layers), so two probes give it for any length."""

    def __init__(self, model, lo, hi, extra=()):
        a, b = probe(model, lo, extra=extra), probe(model, hi, extra=extra)
        self.per_tok = max(0.0, (b["context"] - a["context"]) / (hi - lo)) if hi > lo else 0.0
        self.base = a["context"] - self.per_tok * lo
        self.compute = max(a["compute"], b["compute"])

    def context(self, ctx):
        return int(self.base + self.per_tok * ctx)


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
    target = min(target, ctx_train or target)
    out = []
    c = target
    while c >= lo:
        out.append(c)
        c //= 2
    return out or [lo]


def with_margin(x):
    return int(x * (1 + EST_MARGIN) + EST_MARGIN_FIXED)


def make(model_path, target_ctx=None, min_ctx=MIN_CTX, cpu_only=False, leave_gpu=0,
         vram_free=None, ram_avail=None, threads=None, why_gpu_off=None, device_check=False, extra=()):
    """Return a plan dict, or one with plan["refuse"] set to the reason.
    target_ctx: context to aim for; None = automatic (see CTX_CAP, CTX_SHARE).
    device_check: also ask llama.cpp for its own estimate on the card (starts the card's driver briefly)."""
    m = Model(model_path)
    L = m.n_layer
    why = []
    card = sysmem.dgpu()
    mem = sysmem.meminfo()
    vram_free = sysmem.vram_free(card) if vram_free is None else vram_free
    ram_avail = mem["MemAvailable"] if ram_avail is None else ram_avail
    ram_keep = max(RAM_KEEP_FREE_MIN, int(mem["MemTotal"] * RAM_KEEP_FREE_FRAC))
    ram_budget = ram_avail - ram_keep
    vram_budget = vram_free - VRAM_KEEP_FREE - leave_gpu

    auto = target_ctx is None
    ctxs = ladder(CTX_CAP if auto else target_ctx, min(min_ctx, CTX_CAP if auto else target_ctx), m.meta.get("ctx_train"))
    costs = Costs(m, ctxs[-1], ctxs[0], extra)

    def fair(budget):
        """Automatic context: the lengths that cost at most CTX_SHARE of `budget` (at least the shortest)."""
        if not auto:
            return ctxs
        return [c for c in ctxs if with_margin(costs.context(c)) <= CTX_SHARE * budget] or ctxs[-1:]

    plan = {
        "model": m.path, "name": m.meta.get("name"), "arch": m.meta.get("arch"),
        "layers": L, "moe": m.is_moe, "model_bytes": m.total,
        "vram_free": vram_free, "vram_budget": vram_budget, "ram_avail": ram_avail, "ram_budget": ram_budget,
        "threads": threads or max(2, physical_cores() - 2),
        "why": why, "refuse": None,
    }
    why.append(f"Model: {m.meta.get('name')} ({m.meta.get('arch')}), {m.total / GiB:.1f} GiB, {L} layers"
               + (f", mixture of experts ({m.meta.get('expert_used')} of {m.meta.get('expert_count')} experts per word)"
                  if m.is_moe else "") + ".")
    why.append(f"Free right now: {vram_free / GiB:.1f} GiB on the graphics card, {ram_avail / GiB:.1f} GiB RAM.")
    why.append(f"Kept back: {VRAM_KEEP_FREE / GiB:.1f} GiB of graphics memory for the desktop"
               + (f" + {leave_gpu / GiB:.1f} GiB you asked to leave" if leave_gpu else "")
               + f", {ram_keep / GiB:.1f} GiB RAM for everything else. "
               f"Our estimates get +{EST_MARGIN:.0%} + {EST_MARGIN_FIXED // MiB} MiB extra on top.")

    gpu = None
    if cpu_only or card is None or why_gpu_off:
        why.append("Graphics card not used: " + (why_gpu_off or ("you asked for processor only" if cpu_only else "no card found")) + ".")
    else:
        overhead, src = learned_overhead()
        budget, w = vram_budget, []
        gpu = place_on_gpu(m, costs, fair(budget), budget, w, overhead)
        for _ in range(3):
            if not (gpu and device_check):
                break
            d = probe(m, gpu["ctx"], (gpu["ngl"], gpu["n_cpu_moe"]), extra)
            llama = d["model"] + d["context"] + d["compute"]
            plan["llama_gpu_est"] = llama
            if llama + overhead <= gpu["gpu_est"]:
                break                                   # our own estimate is the higher one: keep it
            short = with_margin(llama + overhead) - gpu["gpu_bytes"]
            if with_margin(llama + overhead) <= vram_budget:   # the real limit, not the shrunk working budget
                gpu.update(gpu_est=llama + overhead, gpu_bytes=with_margin(llama + overhead))
                break
            budget -= short                             # llama.cpp says more than we thought: plan smaller
            w = [f"llama.cpp's own estimate for the card was {short // MiB} MiB higher than ours; planned smaller."]
            gpu = place_on_gpu(m, costs, fair(vram_budget), budget, w, overhead)
        else:
            if gpu and device_check:
                w.append("Couldn't get our plan and llama.cpp's estimate to agree; not using the graphics card.")
                gpu = None
        why.extend(w)
        if gpu:
            why.append(f"Card estimate: " + (f"llama.cpp says {plan['llama_gpu_est'] / GiB:.1f} GiB" if "llama_gpu_est" in plan
                       else "our own (llama.cpp asked only for real runs)")
                       + f", plus {overhead // MiB} MiB for the card's runtime ({src}).")
        plan["overhead"] = overhead

    if gpu is None:
        plan.update(ngl=0, ngl_layers=0, n_cpu_moe=0, ctx=None, output_on_gpu=False, gpu_bytes=0, gpu_est=0)
    else:
        plan.update(gpu)
    on_cpu_layers = set(range(L - plan["ngl_layers"]))
    experts_on_cpu = on_cpu_layers | set(range(plan["n_cpu_moe"]))

    # RAM: weights not on the card are read from the model file through the page cache. Linux can drop
    # those pages again at any time, so they can't crash anything (if they don't fit they are read from
    # the SSD again: slow, not dangerous). Memory llama.cpp allocates itself can't be dropped: that part
    # must fit or we refuse.
    if gpu is None:
        ctx = next((c for c in fair(ram_budget) if costs.context(c) + costs.compute + PROC_OVERHEAD <= ram_budget), None)
        if ctx is None:
            plan["refuse"] = (f"Not enough RAM even on the processor: context + scratch need "
                              f"{(costs.context(ctxs[-1]) + costs.compute + PROC_OVERHEAD) / GiB:.1f} GiB, "
                              f"{max(0, ram_budget) / GiB:.1f} GiB can be spared.")
            return plan
        plan["ctx"] = ctx
        anon = costs.context(ctx) + costs.compute + PROC_OVERHEAD
        why.append(f"Running on the processor only, context {ctx} words.")
    else:
        cpu_ctx_share = costs.context(plan["ctx"]) * len(on_cpu_layers) / L if plan["ngl_layers"] < L else 0
        anon = int(cpu_ctx_share) + (costs.compute if plan["n_cpu_moe"] or plan["ngl_layers"] < L else 0) + PROC_OVERHEAD
    cpu_weights = (m.token_embd + m.other + m.dense(on_cpu_layers) + m.experts(experts_on_cpu)
                   + (0 if plan["output_on_gpu"] else m.output))
    if auto:
        why.append(f"Context chosen automatically: {plan['ctx']} words (this model can take up to "
                   f"{m.meta.get('ctx_train') or '?'}; lodger aims for at most {CTX_CAP} and lets the context use "
                   f"at most {CTX_SHARE:.0%} of the memory it may take, so the model itself keeps the rest).")
    plan["ram_anon_est"] = anon
    plan["ram_weights"] = cpu_weights

    if with_margin(anon) > ram_budget:
        plan["refuse"] = (f"Not enough RAM: the model needs {with_margin(anon) / GiB:.1f} GiB of its own memory "
                          f"(with margin) and only {max(0, ram_budget) / GiB:.1f} GiB can be spared.")
        return plan
    room_for_weights = ram_budget - with_margin(anon)
    if cpu_weights > room_for_weights:
        plan["slow"] = True
        why.append(f"Weights kept in RAM: {cpu_weights / GiB:.1f} GiB, but only {room_for_weights / GiB:.1f} GiB fits "
                   "next to your apps. The rest is read from the SSD again as needed: safe, but slower.")
    else:
        plan["slow"] = False
        why.append(f"Weights kept in RAM: {cpu_weights / GiB:.1f} GiB (fits).")
    # RAM fence: own memory + as much of the weights as we can spare; never more than the budget
    plan["ram_cap"] = max(with_margin(anon), min(ram_budget, with_margin(anon) + cpu_weights + GiB))
    why.append(f"RAM fence for the model: {plan['ram_cap'] / GiB:.1f} GiB, no swap. "
               f"Processor threads: {plan['threads']} (leaves cores for your apps).")
    return plan


def place_on_gpu(m, costs, ctxs, budget, why, overhead):
    """Best placement on the card, or None. Order of preference (NOTES.md step 4)."""
    L = m.n_layer
    # 1. everything, longest context first
    for c in ctxs:
        need = m.dense() + m.experts() + m.output + costs.context(c) + costs.compute + overhead
        if with_margin(need) <= budget:
            why.append(f"Everything fits on the graphics card with a {c}-word context "
                       f"({with_margin(need) / GiB:.1f} GiB of {budget / GiB:.1f} GiB usable, margin included).")
            return gpu_plan(L, L, 0, c, need, True)
    # 2. dense parts + context on the card, experts of as many layers as fit (the last layers first:
    #    llama.cpp's --n-cpu-moe N keeps the experts of the FIRST N layers in RAM)
    if m.is_moe:
        for c in ctxs:
            fixed = m.dense() + m.output + costs.context(c) + costs.compute + overhead
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
    # 3. fewer whole layers on the card (all experts in RAM)
    for c in ctxs:
        per_layer_ctx = costs.context(c) / L
        need = costs.compute + overhead
        n = 0
        while n < L:
            i = L - 1 - n                       # llama.cpp puts the LAST layers on the card
            add = m.layers.get(i, {"dense": 0})["dense"] + per_layer_ctx
            if with_margin(need + add) > budget:
                break
            need += add
            n += 1
        if n >= max(1, L // 2) or c == ctxs[-1]:
            if n < max(1, L // 4):
                why.append(f"Only {n} of {L} layers would fit on the graphics card: too little to be worth "
                           "taking memory from your apps.")
                return None
            why.append(f"Only {n} of {L} layers fit on the card (context {c} words, "
                       f"{with_margin(need) / GiB:.1f} GiB with margin); the rest runs on the processor.")
            return gpu_plan(L, n, L if m.is_moe else 0, c, int(need), False)
    why.append(f"Nothing useful fits on the graphics card ({budget / GiB:.1f} GiB usable after margins).")
    return None


def gpu_plan(L, n_layers, n_cpu_moe, ctx, need, output_on_gpu):
    return {
        "ngl": L + 1 if output_on_gpu else n_layers,   # -ngl counts the output layer as one more
        "ngl_layers": n_layers,
        "n_cpu_moe": n_cpu_moe,
        "ctx": ctx,
        "output_on_gpu": output_on_gpu,
        "gpu_est": need,
        "gpu_bytes": with_margin(need),
    }
