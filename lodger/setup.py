"""lodger setup: download a ready-made llama.cpp from its GitHub releases into ~/.local/share/lodger.

  lodger setup [--backend vulkan|rocm|cuda|cpu] [--tag b12345] [--cuda 12|13] [--rocm 7.2] [--keep 2]

Vulkan by default: it drives AMD, NVIDIA and Intel cards (processor only when the PC has no card lodger
can use). rocm (AMD) and cuda (NVIDIA, with NVIDIA's runtime libraries) are bigger downloads that are
sometimes faster. Every file is checked against the SHA-256 that GitHub publishes for it before anything is
unpacked; the build is unpacked next to its final place and moved there in one step, with a note that it is
complete, so a download cut short is never taken for a finished one. It lands in
~/.local/share/lodger/llama.cpp/<tag>-<backend>, and `current` points at it. Older downloads beyond the
last --keep are removed. Your own llama.cpp (llama_cpp in the settings, or LODGER_LLAMA_CPP) still wins.

CUDA: releases carry a CUDA 12 and a CUDA 13 build. CUDA 13 needs NVIDIA's driver 580 or newer and no
longer runs Maxwell, Pascal or Volta cards, so lodger takes the newest CUDA your driver can run (from
/proc/driver/nvidia/version), CUDA 12 when it can't tell; --cuda picks one. --rocm picks a ROCm version
when a release has more than one (default: the newest; older cards may need an older one).
GITHUB_TOKEN, if set, is sent to GitHub's API (not to the downloads), for a higher rate limit."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

from . import config, gpu
from .gpu import nvidia

API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
ARCH = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
BACKENDS = ("vulkan", "rocm", "cuda", "cpu")
NEEDED = ("llama-server", "llama-fit-params", "llama-cli")
MARKER = ".lodger-setup.json"           # written last: the folder is complete
LOOKUP = object()                       # pick(): read the NVIDIA driver's version from this PC


def asset_pattern(backend, arch):
    """Release file names, e.g. llama-b11972-bin-ubuntu-vulkan-x64.tar.gz; group 1 = tag, 2 = version."""
    mid = {"vulkan": "vulkan-()", "cpu": "()", "rocm": r"rocm-([\d.]+)-", "cuda": r"cuda-([\d.]+)-"}[backend]
    return re.compile(rf"^llama-(b\d+)-bin-ubuntu-{mid}{arch}\.tar\.gz$")


def _ver(s):
    return [int(x) for x in (s or "0").split(".") if x.isdigit()]


def _choose(hits, backend, want, driver):
    """The (asset, match) to take among one release's builds of `backend` (several CUDA or ROCm versions)."""
    if want:
        return [h for h in hits if h[1].group(2) == want or h[1].group(2).startswith(want + ".")]
    hits = sorted(hits, key=lambda h: _ver(h[1].group(2)), reverse=True)       # newest first
    if backend != "cuda":
        return hits
    if not driver:
        return sorted(hits, key=lambda h: _ver(h[1].group(2)))                # can't tell: most compatible first
    return [h for h in hits if nvidia.cuda_ok(_ver(h[1].group(2))[0], driver) is not False]


def pick(releases, backend, arch, tag=None, want=None, driver=LOOKUP):
    """(release, asset, extra assets) for the newest build (or `tag`) that has a file for this backend.
    cuda also needs NVIDIA's runtime (cudart-...) of the same CUDA version, and a version this PC's driver
    can run (driver: its version tuple, None if unknown); want: a CUDA/ROCm version like 12 or 7.2."""
    pat = asset_pattern(backend, arch)
    if driver is LOOKUP:
        driver = nvidia.driver_version() if backend == "cuda" else None
    for r in releases:
        if r.get("draft") or not re.fullmatch(r"b\d+", r.get("tag_name", "")) or (tag and r["tag_name"] != tag):
            continue
        hits = [(a, pat.match(a["name"])) for a in r.get("assets", [])]
        hits = [(a, m) for a, m in hits if m]
        for a, m in _choose(hits, backend, want, driver):
            extra = []
            if backend == "cuda":
                rt = f"cudart-{a['name']}"
                extra = [x for x in r["assets"] if x["name"] == rt]
                if not extra:
                    continue
            return r, a, extra
    return None


def get_json(url):
    """GitHub's API, with a plain message when the network or GitHub says no."""
    headers = {"User-Agent": "lodger-setup", "Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN") and url.startswith("https://api.github.com/"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as f:
            return json.load(f)
    except urllib.error.HTTPError as ex:
        if ex.code in (403, 429) and (ex.headers.get("X-RateLimit-Remaining") == "0" or ex.code == 429):
            raise SystemExit("GitHub's rate limit for downloads without an account is used up for now: try again "
                             "in an hour, or set GITHUB_TOKEN to a GitHub token") from None
        if ex.code == 404:
            raise SystemExit(f"GitHub has no such llama.cpp release ({url})") from None
        raise SystemExit(f"GitHub answered {ex.code} {ex.reason} for {url}") from None
    except (urllib.error.URLError, OSError, ValueError) as ex:
        why = getattr(ex, "reason", None) or ex
        raise SystemExit(f"can't reach GitHub ({why}): check the network connection and try again") from None


def download(asset, folder):
    """Download one release file into `folder`, checking its SHA-256. Returns the path."""
    digest = asset.get("digest") or ""
    if not digest.startswith("sha256:"):
        raise SystemExit(f"GitHub publishes no SHA-256 for {asset['name']}: not installing it unchecked")
    want = digest.split(":", 1)[1].lower()
    path = os.path.join(folder, asset["name"])
    h, done, size, shown = hashlib.sha256(), 0, asset.get("size") or 0, -1
    req = urllib.request.Request(asset["browser_download_url"], headers={"User-Agent": "lodger-setup"})
    try:
        with urllib.request.urlopen(req, timeout=60) as src, open(path, "wb") as out:
            while True:
                block = src.read(1 << 20)
                if not block:
                    break
                out.write(block)
                h.update(block)
                done += len(block)
                if size and done * 10 // size != shown:
                    shown = done * 10 // size
                    print(f"\r  {asset['name']}: {done >> 20} of {size >> 20} MiB", end="", file=sys.stderr, flush=True)
    except (urllib.error.URLError, OSError) as ex:
        raise SystemExit(f"downloading {asset['name']} failed ({getattr(ex, 'reason', None) or ex}); nothing was installed") from None
    print(file=sys.stderr)
    if h.hexdigest() != want:
        raise SystemExit(f"{asset['name']}: the download doesn't match its published SHA-256 (got {h.hexdigest()}, "
                         f"expected {want}); nothing was installed")
    return path


def safe_tar():
    """Python 3.11.0-3.11.3 can't filter what an archive unpacks; lodger never unpacks without that."""
    if not hasattr(tarfile, "data_filter"):
        raise SystemExit(f"Python {platform.python_version()} can't unpack archives safely (tarfile's filters "
                         "came in 3.11.4): update Python, or build llama.cpp yourself and set llama_cpp")


def _merge(src, dst):
    """Move everything in src into dst; folders both have are merged (two archives may share one)."""
    os.makedirs(dst, exist_ok=True)
    for n in os.listdir(src):
        s, d = os.path.join(src, n), os.path.join(dst, n)
        if os.path.isdir(s) and not os.path.islink(s) and os.path.isdir(d) and not os.path.islink(d):
            _merge(s, d)
            continue
        if os.path.isdir(d) and not os.path.islink(d):
            shutil.rmtree(d)
        os.replace(s, d)


def unpack(archive, dest):
    """Unpack into `dest` (merging with what is there); the archive's one top folder's contents land in
    `dest` itself. Unpacks next to `dest` first, in a fresh folder."""
    safe_tar()
    parent = os.path.dirname(os.path.abspath(dest))
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=parent, prefix=".unpack-")
    try:
        with tarfile.open(archive) as t:
            t.extractall(tmp, filter="data")   # no absolute paths, no links out of the folder, no devices
        tops = os.listdir(tmp)
        inner = os.path.join(tmp, tops[0]) if len(tops) == 1 and os.path.isdir(os.path.join(tmp, tops[0])) else tmp
        _merge(inner, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def complete(dest, assets):
    """The folder holds a finished install of exactly these files (the marker is written last)."""
    try:
        with open(os.path.join(dest, MARKER)) as f:
            got = json.load(f)
    except (OSError, ValueError):
        return False
    return got.get("assets") == [a["name"] for a in assets] and all(os.path.exists(os.path.join(dest, n)) for n in NEEDED)


def install(assets, base, name):
    """Download and check every file, unpack them all into one staging folder, then move it into place.
    Returns the final folder."""
    dest = os.path.join(base, name)
    with tempfile.TemporaryDirectory(dir=base, prefix=".download-") as tmp:
        paths = [download(a, tmp) for a in assets]          # all checked before anything is unpacked
        stage = os.path.join(tmp, "stage")
        for p in paths:
            unpack(p, stage)
        missing = [n for n in NEEDED if not os.path.exists(os.path.join(stage, n))]
        if missing:
            raise SystemExit(f"{assets[0]['name']} has no {', '.join(missing)}: not installed")
        with open(os.path.join(stage, MARKER), "w") as f:
            json.dump({"assets": [a["name"] for a in assets], "digests": [a.get("digest") for a in assets]}, f)
        if os.path.lexists(dest):
            os.replace(dest, os.path.join(tmp, "old"))     # removed with tmp
        os.replace(stage, dest)
    return dest


def clean_stale(base):
    """Remove what a setup cut short left behind (call with the lock held)."""
    for n in os.listdir(base):
        if n.startswith((".download-", ".unpack-")) or n.endswith(".unpack") or n == "current.new":
            p = os.path.join(base, n)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    os.remove(p)


@contextlib.contextmanager
def locked(base):
    """One `lodger setup` at a time."""
    with open(os.path.join(base, ".setup.lock"), "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit("another `lodger setup` is running") from None
        yield


def default_backend():
    dev = gpu.choose(config.get()["gpu"])
    return "vulkan" if dev is not None else "cpu"


def main(argv):
    ap = argparse.ArgumentParser(prog="lodger setup", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=BACKENDS, help="default: vulkan (cpu when there is no card lodger can use)")
    ap.add_argument("--tag", help="a llama.cpp build, e.g. b11972 (default: the newest)")
    ap.add_argument("--cuda", metavar="VERSION", help="CUDA build to take, e.g. 12 or 13.4 (default: the newest your driver runs)")
    ap.add_argument("--rocm", metavar="VERSION", help="ROCm build to take when there are several (default: the newest)")
    ap.add_argument("--keep", type=int, default=2, help="downloads to keep, the new one included (default 2)")
    a = ap.parse_args(argv)
    safe_tar()
    arch = ARCH.get(platform.machine().lower())
    if not arch:
        raise SystemExit(f"llama.cpp publishes no Linux build for {platform.machine()}: build it yourself and set llama_cpp")
    backend = a.backend or default_backend()
    want = a.cuda if backend == "cuda" else a.rocm if backend == "rocm" else None
    print(f"Looking for llama.cpp {a.tag or '(newest)'} for {backend} on {arch} ...", flush=True)
    if a.tag:
        found = pick([get_json(f"{API}/tags/{a.tag}")], backend, arch, a.tag, want)
    else:
        found = pick(get_json(API + "?per_page=30"), backend, arch, None, want)
    if not found:
        drv = nvidia.driver_version() if backend == "cuda" else None
        hint = (f" that NVIDIA driver {'.'.join(map(str, drv))} can run (--cuda picks one anyway)" if drv else "")
        raise SystemExit(f"no llama.cpp release{' ' + a.tag if a.tag else ' among the newest 30'} with a {backend}"
                         f"{' ' + want if want else ''} build for Linux {arch}{hint} "
                         "(see https://github.com/ggml-org/llama.cpp/releases)")
    rel, asset, extra = found
    if backend == "cuda":
        v = asset_pattern(backend, arch).match(asset["name"]).group(2)
        drv = nvidia.driver_version()
        print(f"CUDA {v} build" + (f" (NVIDIA driver {'.'.join(map(str, drv))})" if drv else
                                    " (NVIDIA's driver version not found: taking the most compatible)"), flush=True)
    base = os.path.join(config.DATA, "llama.cpp")
    name = f"{rel['tag_name']}-{backend}"
    os.makedirs(base, exist_ok=True)
    with locked(base):
        clean_stale(base)
        dest = os.path.join(base, name)
        if complete(dest, [asset] + extra):
            print(f"Already downloaded: {dest}")
        else:
            install([asset] + extra, base, name)
        link = os.path.join(base, "current")
        tmp_link = link + ".new"
        if os.path.lexists(tmp_link):
            os.remove(tmp_link)
        os.symlink(name, tmp_link)
        os.replace(tmp_link, link)
        print(f"Installed llama.cpp {rel['tag_name']} ({backend}) in {dest}; lodger uses it from {link}")

        old = sorted((d for d in os.listdir(base) if re.fullmatch(r"b\d+-\w+", d) and d != name),
                     key=lambda d: os.path.getmtime(os.path.join(base, d)), reverse=True)
        for d in old[max(0, a.keep - 1):]:
            shutil.rmtree(os.path.join(base, d))
            print(f"Removed an older download: {d}")

    if os.environ.get("LODGER_LLAMA_CPP") or config.get().get("llama_cpp"):
        print(f"Note: your settings point lodger at your own llama.cpp ({os.environ.get('LODGER_LLAMA_CPP') or config.get()['llama_cpp']}); "
              f"remove llama_cpp from {config.PATH} (and LODGER_LLAMA_CPP) to use this download instead.")
    print("Check it with `lodger doctor`.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
