"""lodger setup: download a ready-made llama.cpp from its GitHub releases into ~/.local/share/lodger.

  lodger setup [--backend vulkan|rocm|cuda|cpu] [--tag b12345] [--keep 2]

Vulkan by default: it drives AMD, NVIDIA and Intel cards (processor only when the PC has no card lodger
can use). rocm (AMD) and cuda (NVIDIA, with NVIDIA's runtime libraries) are bigger downloads that are
sometimes faster. The download is checked against the SHA-256 that GitHub publishes for it, unpacked to
~/.local/share/lodger/llama.cpp/<tag>-<backend>, and `current` points at it. Older downloads beyond the
last --keep are removed. Your own llama.cpp (llama_cpp in the settings, or LODGER_LLAMA_CPP) still wins."""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request

from . import config, gpu

API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
ARCH = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
BACKENDS = ("vulkan", "rocm", "cuda", "cpu")
NEEDED = ("llama-server", "llama-fit-params", "llama-cli")


def asset_pattern(backend, arch):
    """Release file names, e.g. llama-b11972-bin-ubuntu-vulkan-x64.tar.gz; group 1 = tag, 2 = version."""
    mid = {"vulkan": "vulkan-()", "cpu": "()", "rocm": r"rocm-([\d.]+)-", "cuda": r"cuda-([\d.]+)-"}[backend]
    return re.compile(rf"^llama-(b\d+)-bin-ubuntu-{mid}{arch}\.tar\.gz$")


def pick(releases, backend, arch, tag=None):
    """(release, asset, extra assets) for the newest build (or `tag`) that has a file for this backend.
    cuda also needs NVIDIA's runtime (cudart-...) of the same CUDA version."""
    pat = asset_pattern(backend, arch)
    for r in releases:
        if r.get("draft") or not re.fullmatch(r"b\d+", r.get("tag_name", "")) or (tag and r["tag_name"] != tag):
            continue
        hits = [(a, pat.match(a["name"])) for a in r.get("assets", [])]
        hits = [(a, m) for a, m in hits if m]
        if not hits:
            continue
        # the newest CUDA/ROCm version first
        a, m = max(hits, key=lambda h: [int(x) for x in (h[1].group(2) or "0").split(".")])
        extra = []
        if backend == "cuda":
            rt = f"cudart-{a['name']}"
            extra = [x for x in r["assets"] if x["name"] == rt]
            if not extra:
                continue
        return r, a, extra
    return None


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "lodger-setup", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as f:
        return json.load(f)


def download(asset, folder):
    """Download one release file into `folder`, checking its SHA-256. Returns the path."""
    digest = asset.get("digest") or ""
    if not digest.startswith("sha256:"):
        raise SystemExit(f"GitHub publishes no SHA-256 for {asset['name']}: not installing it unchecked")
    want = digest.split(":", 1)[1].lower()
    path = os.path.join(folder, asset["name"])
    h, done, size, shown = hashlib.sha256(), 0, asset.get("size") or 0, -1
    req = urllib.request.Request(asset["browser_download_url"], headers={"User-Agent": "lodger-setup"})
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
    print(file=sys.stderr)
    if h.hexdigest() != want:
        raise SystemExit(f"{asset['name']}: the download doesn't match its published SHA-256 (got {h.hexdigest()}, "
                         f"expected {want}); nothing was installed")
    return path


def unpack(archive, dest):
    """Unpack into `dest`; the archive's one top folder's contents land in `dest` itself."""
    tmp = dest + ".unpack"
    os.makedirs(tmp)
    with tarfile.open(archive) as t:
        t.extractall(tmp, filter="data")       # no absolute paths, no links out of the folder, no devices
    tops = os.listdir(tmp)
    inner = os.path.join(tmp, tops[0]) if len(tops) == 1 and os.path.isdir(os.path.join(tmp, tops[0])) else tmp
    os.makedirs(dest, exist_ok=True)
    for n in os.listdir(inner):
        os.replace(os.path.join(inner, n), os.path.join(dest, n))
    shutil.rmtree(tmp)


def default_backend():
    dev = gpu.choose(config.get()["gpu"])
    return "vulkan" if dev is not None else "cpu"


def main(argv):
    ap = argparse.ArgumentParser(prog="lodger setup", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=BACKENDS, help="default: vulkan (cpu when there is no card lodger can use)")
    ap.add_argument("--tag", help="a llama.cpp build, e.g. b11972 (default: the newest)")
    ap.add_argument("--keep", type=int, default=2, help="downloads to keep, the new one included (default 2)")
    a = ap.parse_args(argv)
    arch = ARCH.get(platform.machine().lower())
    if not arch:
        raise SystemExit(f"llama.cpp publishes no Linux build for {platform.machine()}: build it yourself and set llama_cpp")
    backend = a.backend or default_backend()
    print(f"Looking for llama.cpp {a.tag or '(newest)'} for {backend} on {arch} ...", flush=True)
    found = pick(get_json(API + "?per_page=30"), backend, arch, a.tag)
    if not found:
        raise SystemExit(f"no llama.cpp release{' ' + a.tag if a.tag else ''} with a {backend} build for Linux {arch} "
                         "among the newest 30 (see https://github.com/ggml-org/llama.cpp/releases)")
    rel, asset, extra = found
    base = os.path.join(config.DATA, "llama.cpp")
    name = f"{rel['tag_name']}-{backend}"
    dest = os.path.join(base, name)
    os.makedirs(base, exist_ok=True)
    if os.path.isdir(dest) and all(os.path.exists(os.path.join(dest, n)) for n in NEEDED):
        print(f"Already downloaded: {dest}")
    else:
        if os.path.exists(dest):
            shutil.rmtree(dest)
        with tempfile.TemporaryDirectory(dir=base, prefix=".download-") as tmp:
            for x in [asset] + extra:
                unpack(download(x, tmp), dest)
        missing = [n for n in NEEDED if not os.path.exists(os.path.join(dest, n))]
        if missing:
            shutil.rmtree(dest)
            raise SystemExit(f"{asset['name']} has no {', '.join(missing)}: not installed")
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
