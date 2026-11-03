"""lodger models: a few good models to start with, for a PC that has none yet.

  lodger models              the models downloaded here, and which of lodger's list fit this PC right now
  lodger models get NAME     download one into ~/.local/share/lodger/models
  lodger models rm NAME      move a downloaded model to the Trash

lodger's list is small on purpose: general chat models under the Apache-2.0 or MIT licence (checked on the
file's repository and on the model it was made from), in the makers' own files or well-known conversions
(llama.cpp's ggml-org, Unsloth), sized for about 4 to 24 GiB of free graphics memory and RAM together, dense
and mixture of experts (MoE: only a few experts work per word, so it runs fast with most of it in RAM).
Each one is pinned to one exact file on Hugging Face (repository, commit, size and SHA-256), so what you get
is what was checked.

Whether a model fits is lodger's own plan (the one `lodger plan` shows) for the memory free right now. To plan
before downloading, lodger fetches only the start of each file (its header, a few MB) into
~/.cache/lodger/headers. Without llama.cpp (`lodger setup`) or the network the fit is a rough guess, and says so.

Nothing downloads until you run `lodger models get NAME`. It shows the size and licence first, continues an
interrupted download when run again, and counts the file as downloaded only once its SHA-256 matches (a
NAME.gguf.ok note next to it). `lodger serve` then serves it as NAME: the folder is in lodger's default
folders, or added to `folders` in models.toml when that lists its own. `lodger models rm NAME` moves the file
to the Trash (with `gio trash` when there is one); you empty the Trash yourself."""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import http.client
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from typing import NamedTuple

from . import catalog, host

MiB, GiB = 1 << 20, 1 << 30
HF = "https://huggingface.co"
CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "lodger", "headers")
HEADER_FIRST, HEADER_MOST = 4 * MiB, 256 * MiB     # header fetch: first try, and the most it may take
SPARE_DISK = 512 * MiB                             # left free on the disk after a download
LICENCES = {"Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0", "MIT": "https://opensource.org/license/mit"}


class Pick(NamedTuple):
    name: str           # lodger's name: the file is saved as NAME.gguf, so `lodger serve` uses the same name
    repo: str           # Hugging Face repository
    revision: str       # its commit: the file can't change under the same address
    file: str
    size: int           # bytes
    sha256: str
    licence: str        # SPDX id, checked on the repository and on the model it was made from
    moe: bool           # mixture of experts
    params: str         # e.g. "30B, 3B active"
    about: str

    @property
    def url(self):
        return f"{HF}/{self.repo}/resolve/{self.revision}/{urllib.parse.quote(self.file)}"

    @property
    def page(self):
        return f"{HF}/{self.repo}"


# smallest first; checked on Hugging Face 2026-10-29 and 2026-11-03 (sizes and SHA-256 from its API, licences on both repos)
LIST = (
    Pick("granite-4.2-3b", "ibm-granite/granite-4.2-3b-GGUF", "c40945d71cd90f249a56985e8155551a9188dc30",
         "granite-4.2-3b-Q4_K_M.gguf", 2244011552, "e0406663965846ae22a403456eb826ccce5f450840491f71952f18a7cb78e7d5",
         "Apache-2.0", False, "3B", "IBM's small Granite 4.2, Q4_K_M; context up to 128K"),
    Pick("qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523",
         "Qwen3.5-4B-Q4_K_M.gguf", 2740937888, "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
         "Apache-2.0", False, "4B", "Qwen3.5 4B, Q4_K_M (Unsloth's conversion); thinks first unless told not to; up to 256K"),
    Pick("ling-3.0-tiny", "inclusionAI/Ling-3.0-tiny-GGUF", "01b850e29ed5b44e2b810160ff980a9bb353296f",
         "Ling-3.0-tiny-Q4_K_M.gguf", 4823894944, "246d67d45f5b0c7447ca0ce0bdb8b46b8a47b593c5068635dc16422c051d7be9",
         "MIT", True, "7.9B, 1.3B active", "inclusionAI's Ling 3.0 tiny, Q4_K_M: quick even on the processor; up to 128K"),
    Pick("qwen3.5-9b", "unsloth/Qwen3.5-9B-GGUF", "3885219b6810b007914f3a7950a8d1b469d598a5",
         "Qwen3.5-9B-Q4_K_M.gguf", 5680522464, "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
         "Apache-2.0", False, "9B", "Qwen3.5 9B, Q4_K_M (Unsloth's conversion); thinks first unless told not to; up to 256K"),
    Pick("gemma-4-12b", "google/gemma-4-12B-it-qat-q4_0-gguf", "29d097773436b69ff9feafd636ab4cf873786537",
         "gemma-4-12b-it-qat-q4_0.gguf", 6975879296, "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b",
         "Apache-2.0", False, "12B", "Google's Gemma 4 12B, trained for 4-bit (QAT q4_0), Google's own file; up to 256K"),
    Pick("ministral-3-14b", "mistralai/Ministral-3-14B-Instruct-2512-GGUF", "74fac473c43357d7fb2671713608183cc72496d0",
         "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf", 8239593024,
         "824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613",
         "Apache-2.0", False, "14B", "Mistral's Ministral 3 14B Instruct (2512), Q4_K_M, Mistral's own file; up to 256K"),
    Pick("gpt-oss-20b", "ggml-org/gpt-oss-20b-GGUF", "15dcfc7e7c608c6161507211ed43c5b590801b4b",
         "gpt-oss-20b-MXFP4.gguf", 12109566624, "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
         "Apache-2.0", True, "21B, 3.6B active", "OpenAI's gpt-oss 20B in its own 4-bit format (MXFP4), llama.cpp's conversion; up to 128K"),
    Pick("gemma-4-26b-a4b", "google/gemma-4-26B-A4B-it-qat-q4_0-gguf", "d1c082be9cf3c8a514acf63b8761f4b41935842e",
         "gemma-4-26B_q4_0-it.gguf", 14439363584, "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d",
         "Apache-2.0", True, "25B, 4B active", "Google's Gemma 4 26B-A4B, trained for 4-bit (QAT q4_0), Google's own file; up to 256K"),
    Pick("qwen3.6-35b-a3b", "unsloth/Qwen3.6-35B-A3B-GGUF", "a483e9e6cbd595906af30beda3187c2663a1118c",
         "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", 17730509792, "649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3",
         "Apache-2.0", True, "35B, 3B active", "Qwen3.6 35B-A3B, Unsloth's 4-bit IQ4_XS: fast for its size with experts in RAM; up to 256K"),
    Pick("qwen3.8-27b", "ggml-org/Qwen3.8-27B-GGUF", "a316ea527fde32c12487be7574e2db5ac2ad2a7e",
         "Qwen3.8-27B-Q4_K_M.gguf", 18973870528, "c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747",
         "Apache-2.0", False, "27B", "Qwen3.8 27B, Q4_K_M, llama.cpp's conversion; the strongest here, wants a big card; up to 256K"),
)


def folder():
    """Where downloads go (catalog.DOWNLOADS, one of lodger serve's default folders)."""
    return catalog.DOWNLOADS


def find(name):
    for p in LIST:
        if p.name == name:
            return p
    raise SystemExit(f"lodger's list has no model {name!r}: `lodger models` shows the list")


def gib(n):
    return f"{n / GiB:.1f} GiB"


# ---------- what is downloaded ----------

def marker(path):
    return path + ".ok"


def read_marker(path):
    """The note `get` wrote next to a checked file, or None."""
    try:
        with open(marker(path)) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def done(pick, base=None):
    """The pick's file is here, complete and checked."""
    path = os.path.join(base or folder(), pick.name + ".gguf")
    note = read_marker(path)
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    return bool(note) and note.get("sha256") == pick.sha256 and size == pick.size


def installed(base=None):
    """[(name, path, note)] for every model file in the download folder, by name; note is None for a file `get`
    didn't check, and "part" for a download not finished yet (NAME.gguf.part)."""
    base = base or folder()
    out = []
    with contextlib.suppress(OSError):
        for n in sorted(os.listdir(base)):
            p = os.path.join(base, n)
            if n.endswith(".gguf") and os.path.isfile(p):
                out.append((n[:-5], p, read_marker(p)))
            elif n.endswith(".gguf.part") and os.path.isfile(p) and not os.path.exists(p[:-5]):
                out.append((n[:-10], p, "part"))
    return out


# ---------- downloading ----------

def _open(url, start=0, end=None):
    """GET url (from byte `start`, through byte `end`); urllib follows Hugging Face's redirect to its file store."""
    req = urllib.request.Request(url, headers={"User-Agent": "lodger-models"})
    if start or end is not None:
        req.add_header("Range", f"bytes={start}-{'' if end is None else end}")
    return urllib.request.urlopen(req, timeout=60)


def _resumed_at(resp):
    """The first byte of a 206 answer (Content-Range: bytes N-M/T), or 0 for a whole-file answer."""
    if resp.status != 206:
        return 0
    m = re.match(r"bytes (\d+)-", resp.headers.get("Content-Range") or "")
    return int(m.group(1)) if m else -1


def hash_file(path, h=None, limit=None):
    h = h or hashlib.sha256()
    left = limit
    with open(path, "rb") as f:
        while left is None or left > 0:
            block = f.read(1 << 20 if left is None else min(1 << 20, left))
            if not block:
                break
            h.update(block)
            if left is not None:
                left -= len(block)
    return h


def download(pick, base=None):
    """Download the pick into `base` (default: folder()), continuing a .part left by an earlier try, and check
    its SHA-256. The file gets its real name only after that, with its .ok note. Returns the path."""
    base = base or folder()
    dest = os.path.join(base, pick.name + ".gguf")
    part = dest + ".part"
    note = read_marker(dest) if os.path.lexists(dest) else None
    with contextlib.suppress(OSError):
        if note and os.path.getsize(dest) != note.get("size"):
            note = None                             # a note left from before, next to another file
    if os.path.lexists(dest) and note is None:
        raise SystemExit(f"{dest} is already there and lodger didn't download it: move or rename it first, so it "
                         "isn't replaced")
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > pick.size:                            # not this file: start again
        os.remove(part)
        have = 0
    free = shutil.disk_usage(base).free
    if free < pick.size - have + SPARE_DISK:
        raise SystemExit(f"not enough disk space in {base}: {pick.name} needs {gib(pick.size - have)} more, "
                         f"{gib(free)} is free (and lodger leaves {SPARE_DISK // MiB} MiB)")
    h = hashlib.sha256()
    if have:
        print(f"Continuing an earlier download ({gib(have)} of {gib(pick.size)}): checking that part first ...",
              file=sys.stderr, flush=True)
        hash_file(part, h)
    done_bytes, shown = have, -1
    try:
        if have < pick.size:
            with _open(pick.url, have) as src:
                at = _resumed_at(src)
                if at != have:
                    if at != 0:
                        raise SystemExit(f"Hugging Face sent {pick.file} from an unexpected place; try again")
                    h, done_bytes = hashlib.sha256(), 0    # the whole file came instead: start over
                with open(part, "ab" if done_bytes else "wb") as out:
                    while True:
                        block = src.read(1 << 20)
                        if not block:
                            break
                        if done_bytes + len(block) > pick.size:
                            raise SystemExit(f"{pick.file} is bigger than it should be ({gib(pick.size)}); "
                                             "nothing was installed")
                        out.write(block)
                        h.update(block)
                        done_bytes += len(block)
                        if done_bytes * 100 // pick.size != shown:
                            shown = done_bytes * 100 // pick.size
                            print(f"\r  {pick.name}: {done_bytes >> 20} of {pick.size >> 20} MiB ({shown}%)", end="",
                                  file=sys.stderr, flush=True)
            print(file=sys.stderr)
    except (urllib.error.URLError, http.client.HTTPException, OSError) as ex:
        print(file=sys.stderr)
        why = getattr(ex, "reason", None) or ex
        raise SystemExit(f"downloading {pick.name} stopped ({why}) at {gib(done_bytes)} of {gib(pick.size)}; "
                         f"run `lodger models get {pick.name}` again to continue") from None
    if done_bytes != pick.size:
        raise SystemExit(f"the download of {pick.name} ended early ({gib(done_bytes)} of {gib(pick.size)}); "
                         f"run `lodger models get {pick.name}` again to continue")
    if h.hexdigest() != pick.sha256:
        os.remove(part)
        raise SystemExit(f"{pick.name}: the download doesn't match its SHA-256 (got {h.hexdigest()}, expected "
                         f"{pick.sha256}); removed it, nothing was installed. Run the command again to download it again.")
    note = {"name": pick.name, "repo": pick.repo, "revision": pick.revision, "file": pick.file,
            "size": pick.size, "sha256": pick.sha256, "licence": pick.licence,
            "checked": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    tmp = marker(dest) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(note, f, indent=1)
    os.replace(tmp, marker(dest))                   # the note first: a file under its real name is always checked
    os.replace(part, dest)
    return dest


@contextlib.contextmanager
def locked(base):
    """One `lodger models get` or `rm` at a time."""
    with open(os.path.join(base, ".lock"), "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit("another `lodger models` is downloading or removing a model") from None
        yield


# ---------- lodger serve finds them ----------

def _home_form(path):
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path == home or path.startswith(home + os.sep) else path


def _array_end(text, i):
    """For a TOML array starting at text[i] == "[": (index of its closing "]", index just after its last value
    or comma, or after "[" when it is empty, that last character)."""
    depth, last, ch, n = 0, i + 1, "[", len(text)
    while i < n:
        c = text[i]
        if c == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c in "\"'":
            triple = text.startswith(c * 3, i)
            q = c * 3 if triple else c
            j = i + len(q)
            while j < n and not text.startswith(q, j):
                j += 2 if c == '"' and text[j] == "\\" else 1
            i = j + len(q)
            last, ch = i, c
            continue
        if c == "[":
            depth += 1
            if depth > 1:
                last, ch = i + 1, c
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i, last, ch
            last, ch = i + 1, c
        elif not c.isspace():
            last, ch = i + 1, c
        i += 1
    raise ValueError("unclosed array")


def with_folder(text, entry):
    """models.toml's text with `entry` added to its top-level `folders`, or None if lodger can't do that safely
    (the result must read back as exactly the same settings plus that one folder)."""
    try:
        before = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    if "folders" not in before:
        return None
    m = re.search(r"(?m)^[ \t]*folders[ \t]*=[ \t]*", text)
    first_table = re.search(r"(?m)^[ \t]*\[", text)
    if not m or (first_table and first_table.start() < m.start()):
        return None
    lit = json.dumps(entry)                         # a JSON string is a TOML basic string
    i = m.end()
    try:
        if text.startswith("[", i):
            _, last, ch = _array_end(text, i)
            add = lit if ch == "[" else f" {lit}," if ch == "," else f", {lit}"
            new = text[:last] + add + text[last:]
        else:
            s = re.match(r"\"(?:[^\"\\\n]|\\.)*\"|'[^'\n]*'", text[i:])
            if not s:
                return None
            new = text[:i] + "[" + s.group(0) + ", " + lit + "]" + text[i + s.end():]
        after = tomllib.loads(new)
    except (ValueError, tomllib.TOMLDecodeError):
        return None
    old_folders = catalog.listed(before, "folders")
    if catalog.listed(after, "folders") != old_folders + [entry]:
        return None
    if {k: v for k, v in after.items() if k != "folders"} != {k: v for k, v in before.items() if k != "folders"}:
        return None
    return new


def register(base=None, config=None):
    """Make sure `lodger serve` looks in the download folder. Returns what was done, in words."""
    base = os.path.abspath(base or folder())
    config = config or catalog.CONFIG
    if not os.path.exists(config):
        return f"lodger serve finds it there (one of its default folders; there is no {config})"
    entry = _home_form(base)
    try:
        with open(config, "rb") as f:
            text = f.read().decode()
        cfg = tomllib.loads(text)
        folders = catalog.listed(cfg, "folders")
    except (OSError, UnicodeDecodeError, ValueError) as ex:      # TOMLDecodeError is a ValueError
        return (f"NOTE: can't read {config} ({ex}); once it is fixed, make sure its `folders` (if any) include "
                f"\"{entry}\" so lodger serve finds it")
    if "folders" not in cfg:
        return f"lodger serve finds it there ({config} sets no folders, so lodger's default folders count)"
    if any(os.path.abspath(os.path.expanduser(x)) == base for x in folders):
        return f"lodger serve finds it there ({config} lists that folder)"
    new = with_folder(text, entry)
    if new is None:
        return (f"NOTE: lodger couldn't add the folder to {config} by itself; add \"{entry}\" to its `folders` "
                "so lodger serve finds it")
    target = os.path.realpath(config)           # a link (e.g. to a dotfiles folder) stays a link
    tmp = target + ".lodger-tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:     # as it was, \r\n included
            f.write(new)
        shutil.copymode(target, tmp)
        os.replace(tmp, target)
    except OSError as ex:                       # e.g. a link into a read-only folder: the download is fine
        with contextlib.suppress(OSError):
            os.remove(tmp)
        return (f"NOTE: lodger couldn't write {target} ({ex.strerror or ex}); add \"{entry}\" to its `folders` "
                "so lodger serve finds it")
    return f"Added \"{entry}\" to `folders` in {config}, so lodger serve finds it"


# ---------- the Trash ----------

def in_use(path):
    """Processes (pids) that have `path` mapped or open, e.g. a running lodger serve's llama-server."""
    try:
        st = os.stat(path)
    except OSError:
        return []
    real, out = os.path.realpath(path), []
    for pid in host.listdir("/proc"):
        if not pid.isdigit():
            continue
        maps = host.read(f"/proc/{pid}/maps") or ""
        mapped = any(len(f := line.split(maxsplit=5)) == 6 and f[4] == str(st.st_ino) and f[5] == real
                     for line in maps.splitlines())
        if not mapped:
            fds = host.path(f"/proc/{pid}/fd")
            with contextlib.suppress(OSError):
                mapped = any(os.readlink(os.path.join(fds, fd)) == real for fd in os.listdir(fds))
        if mapped:
            out.append(int(pid))
    return out


def trash(path):
    """Move `path` to the desktop's Trash: `gio trash` when there is one, else the freedesktop.org Trash in
    the home folder (~/.local/share/Trash) by hand."""
    if shutil.which("gio"):
        r = subprocess.run(["gio", "trash", "--", path], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"`gio trash {path}` failed: {(r.stderr or r.stdout).strip()}")
        return
    base = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "Trash")
    files, info = os.path.join(base, "files"), os.path.join(base, "info")
    os.makedirs(files, exist_ok=True)
    os.makedirs(info, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(path))
    n = 0
    while True:
        name = f"{stem}{'.' + str(n) if n else ''}{ext}"
        note = os.path.join(info, name + ".trashinfo")
        try:
            fd = os.open(note, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            n += 1
            continue
        if os.path.lexists(os.path.join(files, name)):
            os.close(fd)
            os.remove(note)
            n += 1
            continue
        break
    with os.fdopen(fd, "w") as f:
        f.write(f"[Trash Info]\nPath={urllib.parse.quote(os.path.abspath(path))}\n"
                f"DeletionDate={datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}\n")
    try:
        os.rename(path, os.path.join(files, name))
    except OSError as ex:
        os.remove(note)
        raise SystemExit(f"can't move {path} to the Trash ({ex.strerror or ex}); nothing was removed") from None


# ---------- does it fit ----------

class Short(Exception):
    """The bytes so far end before the header does."""


def header_end(buf):
    """Where a GGUF file's tensor data starts (the header ends, aligned), from its first bytes. Raises Short when
    `buf` ends before the header, ValueError when it isn't GGUF."""
    pos = 0

    def take(n):
        nonlocal pos
        if pos + n > len(buf):
            raise Short
        pos += n
        return buf[pos - n:pos]

    def num(fmt):
        return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]

    def text():
        n = num("<Q")
        if n > 1 << 20:
            raise ValueError("not a sensible GGUF header")
        return take(n).decode("utf-8", "replace")

    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def value(kind):
        if kind == 8:
            return text()
        if kind == 9:
            sub, n = num("<I"), num("<Q")
            if sub in scalar:
                take(struct.calcsize(scalar[sub]) * n)
            else:
                for _ in range(n):
                    value(sub)
            return None
        if kind not in scalar:
            raise ValueError(f"unknown GGUF value type {kind}")
        return num(scalar[kind])

    if take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    if num("<I") < 2:
        raise ValueError("GGUF version 1 isn't supported")
    n_tensors, n_kv = num("<Q"), num("<Q")
    align = 32
    for _ in range(n_kv):
        k, kind = text(), num("<I")
        v = value(kind)
        if k == "general.alignment" and isinstance(v, int) and v > 0:
            align = v
    for _ in range(n_tensors):
        text()
        take(8 * num("<I"))                         # dimensions
        num("<I")                                   # type
        num("<Q")                                   # offset in the data
    return (pos + align - 1) // align * align


def header_stub(pick):
    """A local stand-in for the pick's file: its real header, then nothing (a sparse file of the full size, using
    no disk space), so lodger's planner and llama-fit-params read it as the real model. Cached. Returns the path."""
    path = os.path.join(CACHE, f"{pick.name}-{pick.sha256[:12]}.gguf")
    if os.path.exists(path) and os.path.getsize(path) == pick.size:
        return path
    os.makedirs(CACHE, exist_ok=True)
    want = HEADER_FIRST
    while True:
        with _open(pick.url, 0, want - 1) as src:
            buf = src.read(want)                    # a server that ignores Range sends it all: read only this
        try:
            end = header_end(buf)
            break
        except Short:
            if len(buf) < want or want >= HEADER_MOST:
                raise ValueError(f"{pick.file}: its header doesn't end within {gib(len(buf))}") from None
            want *= 4
    if end > pick.size:
        raise ValueError(f"{pick.file}: its header is bigger than the file")
    fd, tmp = tempfile.mkstemp(dir=CACHE, prefix=".header-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(buf[:end])
            f.truncate(pick.size)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    return path


def free_now(card, mem):
    """(free graphics memory, available RAM) in bytes, right now."""
    vram = (card.free(meminfo=mem) or 0) if card is not None and not card.integrated else 0
    return vram, mem["MemAvailable"]


def rough(pick, vram, ram):
    """A guess without a plan: the file plus room for context and the desktop, against all free memory."""
    return pick.size * 1.15 + 2 * GiB <= vram + ram


def where(p):
    """A plan in a few words."""
    L = p["layers"]
    if not p["ngl"]:
        return "on the processor"
    if p["ngl_layers"] < L:
        return f"{p['ngl_layers']} of {L} layers on the graphics card"
    if p["moe"] and p["n_cpu_moe"]:
        return f"on the graphics card, experts of {p['n_cpu_moe']} of {L} layers in RAM"
    return "all on the graphics card"


def judge(picks, plan_one, vram, ram, rough_why=None):
    """{name: (fits, words, every layer on the card)} for each pick. plan_one(pick) -> a plan dict; None (with
    rough_why), or a pick whose plan fails, gets a rough guess instead; no network means no more plans. Returns
    (verdicts, [why a guess was rough], the note from the first plan on why the card isn't used, or None)."""
    out, notes, card_note = {}, [rough_why] if rough_why else [], None
    for pk in picks:
        if pk.size > vram + ram:
            out[pk.name] = (False, f"no: {gib(pk.size)} file, {gib(vram + ram)} free in all", False)
            continue
        p = None
        if plan_one:
            print(f"\r  planning {pk.name} ...{' ' * 20}", end="", file=sys.stderr, flush=True)
            try:
                p = plan_one(pk)
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError) as ex:
                notes.append(f"can't reach Hugging Face for the files' headers ({getattr(ex, 'reason', None) or ex})")
                plan_one = None
            except (SystemExit, Exception) as ex:          # e.g. a design this llama.cpp doesn't know yet
                notes.append(f"{pk.name}: {(str(ex) or type(ex).__name__).strip().splitlines()[0]}")
        if p is None:
            ok = rough(pk, vram, ram)
            out[pk.name] = (ok, ("probably" if ok else "probably not") + " (rough guess)", False)
            continue
        card_note = card_note or next((w for w in p["why"] if w.startswith("Graphics card not used")), None)
        whole = bool(p["ngl"]) and p["ngl_layers"] >= p["layers"]
        if p["refuse"]:
            out[pk.name] = (False, "no: " + p["refuse"].split(":")[0].rstrip("."), False)
        elif p.get("slow"):
            out[pk.name] = (False, f"slowly: {where(p)}, but RAM can't hold the rest (read from disk again and again)",
                            False)
        else:
            out[pk.name] = (True, f"yes: {where(p)}, {p['ctx'] // 1024}K context", whole)
    if plan_one or len(notes) > bool(rough_why):
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr, flush=True)
    return out, notes, card_note


def suggest(picks, verdicts):
    """Of each kind (dense, mixture of experts), the largest pick that fits with every layer on the graphics
    card (a mixture of experts may keep experts in RAM), else the largest that fits."""
    out = []
    for moe in (False, True):
        fit = [p for p in picks if p.moe == moe and verdicts[p.name][0]]
        if fit:
            out.append(max(fit, key=lambda p: (verdicts[p.name][2], p.size)))
    return out


def planner(base):
    """(plan_one for judge(), or None and why not, the card, /proc/meminfo). plan_one is lodger's planner on the
    downloaded file, else on a header stub, with the [defaults] of models.toml (q8_0 context and so on)."""
    from . import engine, model, plan, sysmem
    card, mem = sysmem.dgpu(), sysmem.meminfo()
    try:
        engine.tool("llama-fit-params")
        model.gguf_reader()
    except SystemExit as ex:
        return None, str(ex), card, mem
    cfg = {}
    with contextlib.suppress(OSError, ValueError, tomllib.TOMLDecodeError):
        with open(catalog.CONFIG, "rb") as f:
            cfg = tomllib.load(f).get("defaults", {})
    cfg = cfg if isinstance(cfg, dict) else {}

    def plan_one(pk):
        path = os.path.join(base, pk.name + ".gguf")
        path = path if done(pk, base) else header_stub(pk)
        extra = catalog.Entry(pk.name, path, cfg).planning_args()
        return plan.make(path, extra=extra, card=card, mem=mem)
    return plan_one, None, card, mem


# ---------- the command ----------

def show_list(a):
    base = folder()
    have = installed(base)
    print(f"Downloaded, in {base}:")
    if not have:
        print("  none yet")
    for name, path, note in have:
        size = os.path.getsize(path)
        if note == "part":
            print(f"  {name:<22} {gib(size):>9}  (partly downloaded: `lodger models get {name}` continues it)")
        elif note:
            print(f"  {name:<22} {gib(size):>9}  {note.get('licence', '?'):<11} {note.get('repo', '')}")
        else:
            print(f"  {name:<22} {gib(size):>9}  (not downloaded by lodger: not checked)")
    picks = [p for p in LIST if not done(p, base)] if not a.all else list(LIST)
    if not picks:
        print("\nEvery model on lodger's list is downloaded.")
        return 0
    plan_one, no_plan, card, mem = planner(base)
    vram, ram = free_now(card, mem)
    print(f"\nlodger's list, and what fits right now ({gib(vram)} free on the graphics card, {gib(ram)} RAM; "
          "a model lodger is serving counts as used). Planning ...", flush=True)
    verdicts, notes, card_note = judge(picks, plan_one, vram, ram, no_plan)
    print(f"  {'NAME':<22} {'KIND':<7} {'SIZE':>9}  {'LICENCE':<11} FITS")
    for p in picks:
        kind = "MoE" if p.moe else "dense"
        print(f"  {p.name:<22} {kind:<7} {gib(p.size):>9}  {p.licence:<11} {verdicts[p.name][1]}")
    if card_note:
        print(f"\n{card_note}")
    for n in notes:
        print(f"\nNo plan, so a rough guess: {n.splitlines()[0]}")
    best = suggest(picks, verdicts)
    print()
    if best:
        print("Suggested: " + ", ".join(f"{p.name} ({'mixture of experts, ' + p.params if p.moe else p.params})"
                                        for p in best) + ".")
    else:
        print("None of them fits in the memory free right now. Close some apps, or try again later.")
    print("Nothing is downloaded until you ask: `lodger models get NAME` (`lodger models info NAME` for details).")
    return 0


def show_info(p):
    print(f"{p.name}: {p.about}")
    print(f"  {'mixture of experts' if p.moe else 'dense'}, {p.params} parameters")
    print(f"  licence: {p.licence} ({LICENCES.get(p.licence, '')}; see {p.page})")
    print(f"  file:    {p.file}, {gib(p.size)} ({p.size} bytes)")
    print(f"  from:    {p.repo} at {p.revision[:12]}")
    print(f"  sha256:  {p.sha256}")


def get(name):
    p = find(name)
    base = folder()
    os.makedirs(base, exist_ok=True)
    show_info(p)
    with locked(base):
        if done(p, base):
            print(f"\nAlready downloaded and checked: {os.path.join(base, p.name + '.gguf')}")
        else:
            print(f"\nDownloading {gib(p.size)} into {base} ...", flush=True)
            path = download(p, base)
            print(f"Downloaded and checked (SHA-256 matches): {path}")
    print(register(base) + f"; it is served as {p.name}.")
    print(f"If lodger serve is running, restart it to see it (systemctl --user restart lodger). Try: lodger plan "
          f"{os.path.join(base, p.name + '.gguf')}")
    return 0


def rm(name):
    base = folder()
    path = os.path.join(base, name + ".gguf")
    if "/" in name or name.startswith(".") or not any(os.path.lexists(x) for x in (path, path + ".part")):
        raise SystemExit(f"no downloaded model {name!r} in {base}: `lodger models` lists them")
    with locked(base):
        for x in (path, path + ".part"):
            if os.path.lexists(x):
                users = [p for p in in_use(x) if p != os.getpid()]
                trash(x)
                print(f"Moved to the Trash: {x}")
                if users:
                    print(f"It is in use right now (process {', '.join(map(str, users))}): that keeps running from the "
                          "Trash until it stops, and the disk space comes back only then.")
        with contextlib.suppress(FileNotFoundError):
            os.remove(marker(path))
    print("If lodger serve is running, restart it so it stops offering it (systemctl --user restart lodger).")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(prog="lodger models", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", choices=("list", "get", "rm", "info"), default="list")
    ap.add_argument("name", nargs="?", help="a model's name from `lodger models`")
    ap.add_argument("--all", action="store_true", help="list: plan the downloaded ones too")
    a = ap.parse_args(argv)
    if a.action != "list" and not a.name:
        ap.error(f"`lodger models {a.action}` needs a model's name")
    if a.action == "get":
        return get(a.name)
    if a.action == "rm":
        return rm(a.name)
    if a.action == "info":
        show_info(find(a.name))
        return 0
    return show_list(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
