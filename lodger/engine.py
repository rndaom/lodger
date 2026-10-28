"""The engine: which llama.cpp lodger runs, what kind of build it is (ROCm, CUDA, Vulkan, processor only), and
how to point it at exactly the card lodger chose.

Where llama.cpp is, first found wins:
  1. LODGER_LLAMA_CPP (environment)
  2. llama_cpp in ~/.config/lodger/config.toml
  3. ~/.local/share/lodger/llama.cpp/current (what `lodger setup` downloads)
  4. llama-server on your PATH (with llama-fit-params next to it)
The folder may hold the programs itself, or bin/ or build/bin/ (a source build).

Picking the card: lodger makes only the chosen card visible to the engine (HIP_VISIBLE_DEVICES,
CUDA_VISIBLE_DEVICES, Mesa's Vulkan device selection), so it is always device 0 (ROCm0, CUDA0, Vulkan0), and
checks llama.cpp's "using device ... (PCI address)" line when it starts: a different card means stop at once."""
import os
import re
import shutil

from . import config, host
from .gpu import amd, nvidia

# backend -> (llama.cpp's device name prefix, library that marks a build with it, vendors it can drive)
BACKENDS = {
    "rocm": ("ROCm", "libggml-hip", {"AMD"}),
    "cuda": ("CUDA", "libggml-cuda", {"NVIDIA"}),
    "vulkan": ("Vulkan", "libggml-vulkan", {"AMD", "NVIDIA", "Intel"}),
}
# a vendor's own backend first, Vulkan otherwise
PREFER = {"AMD": ("rocm", "vulkan"), "NVIDIA": ("cuda", "vulkan"), "Intel": ("vulkan",)}
USING = re.compile(r"using device (\S+) \((.*)\) \(([0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7])\)")


def root():
    """The llama.cpp folder lodger uses, and where that came from."""
    env = os.environ.get("LODGER_LLAMA_CPP")
    if env:
        return os.path.expanduser(env), "LODGER_LLAMA_CPP"
    cfg = config.get().get("llama_cpp")
    if cfg:
        return os.path.expanduser(cfg), f"llama_cpp in {config.PATH}"
    setup = os.path.join(config.DATA, "llama.cpp", "current")
    if os.path.isdir(setup):
        return setup, "lodger setup"
    found = shutil.which("llama-fit-params")
    if found:
        return os.path.dirname(os.path.realpath(found)), "PATH"
    return None, "not found"


def bin_dir():
    r, _ = root()
    if not r:
        return None
    for d in (os.path.join(r, "build", "bin"), os.path.join(r, "bin"), r):
        if os.path.exists(os.path.join(d, "llama-server")):
            return d
    return None


def tool(name):
    """Full path of a llama.cpp program (llama-server, llama-fit-params, llama-cli)."""
    d = bin_dir()
    if not d or not os.path.exists(os.path.join(d, name)):
        r, src = root()
        raise SystemExit(f"lodger can't find llama.cpp's {name} (llama.cpp: {r or 'none'}, from {src}). "
                         "Run `lodger setup` to download llama.cpp, or set llama_cpp in "
                         f"{config.PATH} or LODGER_LLAMA_CPP to your own build.")
    return os.path.join(d, name)


def backends(d=None):
    """The graphics backends this llama.cpp build has, e.g. {'rocm'} or {'vulkan'}; empty: processor only."""
    d = d or bin_dir()
    if not d:
        return set()
    names = set()
    for sub in (d, os.path.join(d, "..", "lib"), os.path.join(d, "..", "lib64")):
        try:
            names.update(os.listdir(sub))
        except OSError:
            pass
    return {b for b, (_, lib, _) in BACKENDS.items() if any(n.startswith(lib + ".so") or n.startswith(lib + "-") for n in names)}


def for_device(dev, devices=None, have=None):
    """How the engine reaches `dev`: {backend, name, env}, or {why} if it can't."""
    have = backends() if have is None else have
    if not have:
        return {"why": "this llama.cpp build has no graphics backend (processor only)"}
    b = next((b for b in PREFER.get(dev.vendor, ()) if b in have), None)
    if b is None:
        return {"why": f"this llama.cpp build ({', '.join(sorted(have))}) can't drive a {dev.vendor} card; "
                       "`lodger setup` can download one that can (Vulkan works on all)"}
    from . import gpu
    devices = gpu.devices() if devices is None else devices
    env = {}
    if b == "rocm":
        i = amd.kfd_index(dev)
        if i is None:
            return {"why": "ROCm doesn't list this card (no matching KFD node)"}
        env["HIP_VISIBLE_DEVICES"] = str(i)
    elif b == "cuda":
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(nvidia.cuda_index(dev, devices))
    elif b == "vulkan":
        env.update(vulkan_env(dev, devices))
    return {"backend": b, "name": BACKENDS[b][0] + "0", "env": env}


def vulkan_env(dev, devices):
    """Show the Vulkan engine only this card: Mesa's device-select layer by PCI address (DRI_PRIME) and by
    vendor:device id when that is unique, then device 0. If another vendor's card is NVIDIA, its driver
    isn't loaded at all (listing devices through it would wake it)."""
    env = {"GGML_VK_VISIBLE_DEVICES": "0",
           "DRI_PRIME": "pci-" + dev.pci.replace(":", "_").replace(".", "_"),
           "MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE": "1"}
    vd = (dev.vendor_id, dev.device_id)
    if sum((d.vendor_id, d.device_id) == vd for d in devices) == 1:
        env["MESA_VK_DEVICE_SELECT"] = f"{dev.vendor_id:04x}:{dev.device_id:04x}!"
    if dev.vendor != "NVIDIA" and any(d.vendor == "NVIDIA" for d in devices):
        env["VK_LOADER_DRIVERS_DISABLE"] = "*nvidia*"
    return env


def cpu_env(have=None):
    """Hide every card from a processor-only engine (on top of systemd's PrivateDevices where there is one)."""
    have = backends() if have is None else have
    env = {}
    if "rocm" in have:
        env.update(HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1")
    if "cuda" in have:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if "vulkan" in have:
        env["GGML_VK_VISIBLE_DEVICES"] = ""
    return env


def device_args(plan_ngl, dev, n_cpu_moe=0):
    """llama.cpp options and environment for a plan: on the card (`dev`) or processor only."""
    if plan_ngl and dev is not None:
        t = for_device(dev)
        if "why" in t:
            raise SystemExit(f"can't put the model on the graphics card: {t['why']}")
        return ["--device", t["name"], "-ngl", str(plan_ngl)] + (["-ncmoe", str(n_cpu_moe)] if n_cpu_moe else []), t["env"]
    return ["--device", "none", "-ngl", "0"], cpu_env()


def check_log(text, dev):
    """llama.cpp's 'using device NAME (description) (PCI)' lines: None if they match `dev` (or haven't come
    yet), else why not. The safety net for the visibility settings above."""
    for name, desc, pci in USING.findall(text or ""):
        if pci.lower()[-12:] != dev.pci.lower()[-12:]:
            return f"llama.cpp used {name} ({desc}, {pci}), not the card lodger chose ({dev.pci})"
    return None


def describe():
    """For `lodger doctor`: where llama.cpp is and what it has."""
    r, src = root()
    d = bin_dir()
    return {"root": r, "from": src, "bin": d, "backends": sorted(backends(d)) if d else [],
            "llama-server": bool(d and os.path.exists(os.path.join(d, "llama-server"))),
            "llama-fit-params": bool(d and os.path.exists(os.path.join(d, "llama-fit-params"))),
            "kfd": bool(host.listdir("/sys/class/kfd/kfd/topology/nodes"))}
