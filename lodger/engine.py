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
checks llama.cpp's "using device ... (PCI address)" line when it starts: a different card means stop at once.

The engine's environment is the same under every fence (fence.py applies child_env()):
- DEVICE_VARS pick the card. lodger sets the ones it needs and removes the rest, so a ROCR_VISIBLE_DEVICES,
  CUDA_VISIBLE_DEVICES, MESA_VK_DEVICE_SELECT ... in your shell can't move the model to another card.
- RUNTIME_VARS (by name or prefix) tune the drivers and runtimes: HSA_OVERRIDE_GFX_VERSION, RADV_*, ROCm,
  CUDA and Vulkan settings, LD_LIBRARY_PATH. They come from the environment lodger was started in, also
  under systemd (whose services otherwise get the user manager's environment instead).
- llama.cpp's own LLAMA_ARG_* settings are removed: lodger sets llama.cpp's options on its command line."""
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
DEVICE_VARS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES",
               "CUDA_DEVICE_ORDER", "GGML_VK_VISIBLE_DEVICES", "MESA_VK_DEVICE_SELECT",
               "MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE", "DRI_PRIME", "VK_LOADER_DRIVERS_DISABLE",
               "NODEVICE_SELECT", "VK_LOADER_LAYERS_DISABLE",
               # the Vulkan loader's own ways to choose drivers and devices: they can change which card is Vulkan0
               "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES", "VK_DRIVER_FILES", "VK_LOADER_DRIVERS_SELECT",
               "VK_LOADER_DEVICE_SELECT", "VK_LOADER_DEVICE_ID_FILTER", "VK_LOADER_VENDOR_ID_FILTER",
               "VK_LOADER_DRIVER_ID_FILTER", "VK_INSTANCE_LAYERS", "VK_LOADER_LAYERS_ENABLE")
RUNTIME_VARS = ("HSA_", "HIP_", "ROCM_", "ROCBLAS_", "HIPBLASLT_", "AMD_", "GPU_", "RADV_", "ACO_", "MESA_", "VK_",
                "ANV_", "INTEL_", "CUDA_", "__GL_", "GGML_", "LD_LIBRARY_PATH")
CLEARED = ("LLAMA_ARG_", "LLAMA_API_KEY")
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


def _libs(d):
    names = set()
    for sub in (d, os.path.join(d, "..", "lib"), os.path.join(d, "..", "lib64")):
        try:
            names.update(os.listdir(sub))
        except OSError:
            pass
    return names


def backends(d=None):
    """The graphics backends this llama.cpp build has, e.g. {'rocm'} or {'vulkan'}; empty: processor only."""
    d = d or bin_dir()
    if not d:
        return set()
    names = _libs(d)
    return {b for b, (_, lib, _) in BACKENDS.items() if any(n.startswith(lib + ".so") or n.startswith(lib + "-") for n in names)}


def cuda_major(d=None):
    """The CUDA version (12, 13 ...) a CUDA build's own runtime libraries are for, or None if it doesn't
    carry them (it then uses the system's CUDA)."""
    d = d or bin_dir()
    found = [re.match(r"^lib(?:cudart|cublas)\.so\.(\d+)", n) for n in (_libs(d) if d else ())]
    majors = {int(m.group(1)) for m in found if m}
    return max(majors) if majors else None


def _match(name, names):
    return any(name == n or (n.endswith("_") and name.startswith(n)) for n in names)


def child_env(env, environ=None):
    """(variables to set, names to remove) for a fenced engine given lodger's `env` (device_args): `env`
    itself, plus the runtime settings of the environment lodger runs in; removed: every card-picking
    variable lodger didn't set, and llama.cpp's own settings."""
    environ = os.environ if environ is None else environ
    put = {k: v for k, v in environ.items()
           if _match(k, RUNTIME_VARS) and k not in DEVICE_VARS and not _match(k, CLEARED)}
    put.update(env)
    drop = [k for k in DEVICE_VARS if k not in put]
    drop += sorted(k for k in environ if _match(k, CLEARED) and k not in put)
    return put, drop


def for_device(dev, devices=None, have=None):
    """How the engine reaches `dev`: {backend, name, env}, or {why} if it can't."""
    have = backends() if have is None else have
    if not dev.known:
        return {"why": f"lodger has no backend for this card's driver ({dev.driver or 'none'})"}
    if not have:
        return {"why": "this llama.cpp build has no graphics backend (processor only)"}
    order = [b for b in PREFER.get(dev.vendor, ()) if b in have]
    old_driver = None
    if "cuda" in order:
        major = cuda_major()
        if major and nvidia.cuda_ok(major) is False:
            need = ".".join(str(x) for x in nvidia.CUDA_DRIVER[major] if x) or "?"
            old_driver = (f"this llama.cpp's CUDA {major} build needs NVIDIA's driver {need} or newer (this PC has "
                          f"{'.'.join(map(str, nvidia.driver_version()))}); `lodger setup --backend cuda` picks one it can run")
            order.remove("cuda")
    b = order[0] if order else None
    if b is None:
        return {"why": old_driver or f"this llama.cpp build ({', '.join(sorted(have))}) can't drive a {dev.vendor} card; "
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
        i = nvidia.cuda_index(dev, devices)
        if i is None:
            return {"why": f"CUDA doesn't list this card (driver {dev.driver or 'none'}, not NVIDIA's)"}
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(i)
    elif b == "vulkan":
        why = vulkan_ambiguous(dev, devices)
        if why:
            return {"why": why}
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


def _vulkan_dirs(kind):
    """The Vulkan loader's search folders for implicit layers ('implicit_layer.d') or drivers ('icd.d')."""
    home = os.path.expanduser("~")
    dirs = [os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")]
    dirs += (os.environ.get("XDG_CONFIG_DIRS") or "/etc/xdg").split(":")
    dirs += ["/etc", os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")]
    dirs += (os.environ.get("XDG_DATA_DIRS") or "").split(":") + ["/usr/local/share", "/usr/share"]   # always: a superset
    if kind == "implicit_layer.d" and os.environ.get("VK_IMPLICIT_LAYER_PATH"):
        dirs, found = [], os.environ["VK_IMPLICIT_LAYER_PATH"].split(":")
    else:
        found = []
    found = found + [os.path.join(d, "vulkan", kind) for d in dirs if d]
    if kind == "implicit_layer.d":
        found += (os.environ.get("VK_ADD_IMPLICIT_LAYER_PATH") or "").split(":")
    return [d for d in found if d]


def vulkan_layer():
    """Path of Mesa's device-select layer (VkLayer_MESA_device_select.json), or None if it isn't installed."""
    for d in _vulkan_dirs("implicit_layer.d"):
        for n in host.listdir(d):
            if n.startswith("VkLayer_MESA_device_select") and n.endswith(".json"):
                return os.path.join(d, n)
    return None


def software_vulkan():
    """A processor 'card' (lavapipe, SwiftShader) is installed: without the layer it is listed like a card."""
    files = [f for v in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES") for f in (os.environ.get(v) or "").split(":") if f]
    if not files:
        files = [n for d in _vulkan_dirs("icd.d") for n in host.listdir(d)]
    files += (os.environ.get("VK_ADD_DRIVER_FILES") or "").split(":")
    return any(("lvp" in n or "swiftshader" in n.lower()) for n in (os.path.basename(f) for f in files))


def vulkan_ambiguous(dev, devices):
    """Why GGML_VK_VISIBLE_DEVICES=0 might not be `dev`, or None. Mesa's device-select layer leaves only the
    chosen card in the list; without it, Vulkan lists every card (and lavapipe) in no set order, and NVIDIA's
    driver has no fdinfo to catch a wrong pick, so lodger uses the card only when it is the only one listed."""
    if vulkan_layer():
        return None
    # (for another vendor's card, NVIDIA's driver is kept off: vulkan_env)
    listed = [d for d in devices if dev.vendor == "NVIDIA" or d.vendor != "NVIDIA"]
    if len(listed) > 1 or software_vulkan():
        return ("Vulkan can't be pointed at one card here: Mesa's device-select layer isn't installed "
                "(VkLayer_MESA_device_select.json, in vulkan-mesa-layers or mesa-vulkan-drivers), so device 0 "
                "could be another card; install it, or use the card's own backend (ROCm or CUDA build)")
    return None


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
    return {"root": r, "from": src, "bin": d, "backends": sorted(backends(d)) if d else [], "cuda": cuda_major(d) if d else None,
            "llama-server": bool(d and os.path.exists(os.path.join(d, "llama-server"))),
            "llama-fit-params": bool(d and os.path.exists(os.path.join(d, "llama-fit-params"))),
            "kfd": bool(host.listdir("/sys/class/kfd/kfd/topology/nodes"))}
