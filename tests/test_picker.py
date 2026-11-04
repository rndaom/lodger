"""lodger models: the built-in list, downloads (from a fake Hugging Face on this PC), models.toml and the Trash."""
import contextlib
import hashlib
import http.server
import io
import json
import mmap
import os
import re
import shutil
import struct
import tempfile
import threading
import tomllib
import unittest
from unittest import mock

from lodger import catalog, cli, picker

MiB, GiB = 1 << 20, 1 << 30


def _s(t):
    b = t.encode()
    return struct.pack("<Q", len(b)) + b


def gguf_bytes(layers=2, align=32, vocab=50, data_size=64):
    """A small, valid GGUF file: a few settings (a string array like a tokenizer's), `layers` layers of F32
    tensors, the data aligned to `align`."""
    kv = [("general.architecture", 8, _s("llama")),
          ("general.alignment", 4, struct.pack("<I", align)),
          ("general.name", 8, _s("Tiny")),
          ("llama.block_count", 4, struct.pack("<I", layers)),
          ("llama.context_length", 4, struct.pack("<I", 4096)),
          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, vocab) + b"".join(_s(f"tok{i}") for i in range(vocab))),
          ("tokenizer.ggml.scores", 9, struct.pack("<IQ", 6, vocab) + b"\0" * 4 * vocab)]
    names = ["token_embd.weight"] + [f"blk.{i}.attn_q.weight" for i in range(layers)] + ["output.weight"]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(names), len(kv))
    for k, kind, v in kv:
        head += _s(k) + struct.pack("<I", kind) + v
    n = data_size // 4
    for i, name in enumerate(names):
        head += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<I", 0) + struct.pack("<Q", i * data_size)
    pad = (-len(head)) % align
    return head + b"\0" * pad, len(head) + pad, bytes(range(256)) * (len(names) * data_size // 256 + 1)


class FakeHF(http.server.BaseHTTPRequestHandler):
    """Hugging Face's two steps: /resolve/... answers with a redirect to the file store, which answers Range."""
    body = b""
    cut_after = None          # send only this many bytes of the next answer, then hang up
    ignore_range = False
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        FakeHF.seen.append((self.path, self.headers.get("Range"), self.headers.get("User-Agent")))
        if "/resolve/" in self.path:
            self.send_response(302)
            self.send_header("Location", "/store/file?sig=1")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body, start = FakeHF.body, 0
        m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range") or "")
        if m and not FakeHF.ignore_range:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else len(body) - 1
            part = body[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{start + len(part) - 1}/{len(body)}")
        else:
            part = body
            self.send_response(200)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        if FakeHF.cut_after is not None:
            part, FakeHF.cut_after = part[:FakeHF.cut_after], None
            self.wfile.write(part)
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(part)


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lodger-picker-")
        self.addCleanup(shutil.rmtree, self.dir)
        self.models = os.path.join(self.dir, "models")
        os.makedirs(self.models)
        st = contextlib.ExitStack()
        self.addCleanup(st.close)
        st.enter_context(mock.patch.object(catalog, "DOWNLOADS", self.models))
        st.enter_context(mock.patch.object(catalog, "CONFIG", os.path.join(self.dir, "models.toml")))
        st.enter_context(mock.patch.object(picker, "CACHE", os.path.join(self.dir, "cache")))
        st.enter_context(mock.patch("sys.stderr", io.StringIO()))
        self.st = st

    def pick(self, body, name="tiny-1b", **kw):
        d = dict(name=name, repo="org/Tiny-GGUF", revision="a" * 40, file="Tiny-Q4_K_M.gguf", size=len(body),
                 sha256=hashlib.sha256(body).hexdigest(), licence="MIT", moe=False, params="1B", about="a test")
        d.update(kw)
        return picker.Pick(**d)


class ServerBase(Base):
    def setUp(self):
        super().setUp()
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHF)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        FakeHF.seen, FakeHF.cut_after, FakeHF.ignore_range = [], None, False
        self.st.enter_context(mock.patch.object(picker, "HF", f"http://127.0.0.1:{srv.server_address[1]}"))


class ListTest(unittest.TestCase):
    def test_every_entry_is_complete_and_pinned(self):
        names = [p.name for p in picker.LIST]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(picker.LIST), 6)
        for p in picker.LIST:
            with self.subTest(p.name):
                self.assertEqual(catalog.slug(p.name + ".gguf"), p.name)     # lodger serve uses the same name
                self.assertRegex(p.revision, r"^[0-9a-f]{40}$")
                self.assertRegex(p.sha256, r"^[0-9a-f]{64}$")
                self.assertIn(p.licence, picker.LICENCES)
                self.assertTrue(p.file.endswith(".gguf") and not catalog.SPLIT.search(p.file))
                self.assertGreater(p.size, 1 << 30)
                self.assertTrue(p.url.startswith("https://huggingface.co/" + p.repo + "/resolve/" + p.revision + "/"))

    def test_smallest_first_with_both_kinds_for_4_to_24_gib(self):
        sizes = [p.size for p in picker.LIST]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], 3 * GiB)
        self.assertTrue(any(p.moe for p in picker.LIST) and any(not p.moe for p in picker.LIST))
        self.assertTrue(all(p.size < 18 * GiB for p in picker.LIST))

    def test_unknown_name_is_a_plain_message(self):
        with self.assertRaises(SystemExit) as ex:
            picker.find("no-such-model")
        self.assertIn("lodger models", str(ex.exception))


class HeaderTest(ServerBase):
    def test_header_end_is_the_aligned_start_of_the_data(self):
        head, end, _ = gguf_bytes(align=64)
        self.assertEqual(picker.header_end(head + b"\1" * 100), end)
        self.assertEqual(end % 64, 0)
        with self.assertRaises(picker.Short):
            picker.header_end(head[:len(head) // 2])
        with self.assertRaises(ValueError):
            picker.header_end(b"NOPE" + head[4:])

    def stub_of(self, body, **kw):
        FakeHF.body = body
        p = self.pick(body, **kw)
        return p, picker.header_stub(p)

    def test_stub_is_the_header_then_a_hole_of_the_full_size(self):
        head, end, data = gguf_bytes()
        body = head + data
        with mock.patch.object(picker, "HEADER_FIRST", 256):         # several tries, growing
            p, path = self.stub_of(body)
        self.assertEqual(os.path.getsize(path), p.size)
        with open(path, "rb") as f:
            got = f.read()
        self.assertEqual(got[:end], head)
        self.assertEqual(got[end:], b"\0" * (len(body) - end))
        ranges = [r for path_, r, _ in FakeHF.seen if path_.startswith("/store")]
        self.assertEqual(ranges[0], "bytes=0-255")
        self.assertGreater(len(ranges), 1)
        n = len(FakeHF.seen)
        self.assertEqual(picker.header_stub(p), path)                 # cached: not fetched again
        self.assertEqual(len(FakeHF.seen), n)

    def test_a_server_that_ignores_range_still_gives_a_stub(self):
        head, end, data = gguf_bytes()
        FakeHF.ignore_range = True
        _, path = self.stub_of(head + data)
        with open(path, "rb") as f:
            self.assertEqual(f.read(end), head)

    def test_not_gguf_is_refused(self):
        with self.assertRaises(ValueError):
            self.stub_of(b"<html>not found</html>" * 10)

    def test_the_model_reader_reads_a_stub_like_the_real_file(self):
        from lodger.model import Model
        head, end, data = gguf_bytes(layers=3)
        body = head + data
        _, path = self.stub_of(body)
        real = os.path.join(self.dir, "real.gguf")
        with open(real, "wb") as f:
            f.write(body)
        a, b = Model(path), Model(real)
        self.assertEqual((a.n_layer, a.total, a.meta), (b.n_layer, b.total, b.meta))
        self.assertEqual(a.n_layer, 3)


class DownloadTest(ServerBase):
    def setUp(self):
        super().setUp()
        FakeHF.body = os.urandom(3 * MiB + 12345)
        self.p = self.pick(FakeHF.body)
        self.dest = os.path.join(self.models, self.p.name + ".gguf")

    def test_downloads_checks_and_notes_it(self):
        self.assertEqual(picker.download(self.p, self.models), self.dest)
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), FakeHF.body)
        self.assertTrue(picker.done(self.p, self.models))
        self.assertFalse(os.path.exists(self.dest + ".part"))
        self.assertEqual(picker.read_marker(self.dest)["sha256"], self.p.sha256)
        self.assertEqual(picker.read_marker(self.dest)["licence"], "MIT")
        first = [x for x in FakeHF.seen if "/resolve/" in x[0]][0]
        self.assertEqual(first[0], f"/org/Tiny-GGUF/resolve/{'a' * 40}/Tiny-Q4_K_M.gguf")
        self.assertEqual(first[2], "lodger-models")

    def test_a_file_lodger_didnt_download_is_never_replaced(self):
        with open(self.dest, "wb") as f:
            f.write(b"my own model")
        with self.assertRaises(SystemExit) as ex:
            picker.download(self.p, self.models)
        self.assertIn("didn't download it", str(ex.exception))
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), b"my own model")
        self.assertFalse([x for x in FakeHF.seen if "/resolve/" in x[0]])     # nothing fetched
        # a note left from an earlier download, next to the user's own file: still not lodger's
        with open(picker.marker(self.dest), "w") as f:
            json.dump({"sha256": "0" * 64, "size": 123}, f)
        with self.assertRaises(SystemExit):
            picker.download(self.p, self.models)
        # one lodger did download (an older revision, say) is replaced
        with open(picker.marker(self.dest), "w") as f:
            json.dump({"sha256": "0" * 64, "size": len(b"my own model")}, f)
        self.assertEqual(picker.download(self.p, self.models), self.dest)

    def test_an_interrupted_download_continues_where_it_stopped(self):
        FakeHF.cut_after = MiB + 7
        with self.assertRaises(SystemExit) as ex:
            picker.download(self.p, self.models)
        self.assertIn("again to continue", str(ex.exception))
        self.assertFalse(os.path.exists(self.dest))                  # not done, not served
        got = os.path.getsize(self.dest + ".part")
        self.assertTrue(0 < got <= MiB + 7)
        self.assertEqual(picker.installed(self.models), [(self.p.name, self.dest + ".part", "part")])
        FakeHF.seen = []
        picker.download(self.p, self.models)
        self.assertIn(f"bytes={got}-", [r for _, r, _ in FakeHF.seen])
        self.assertTrue(picker.done(self.p, self.models))
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), FakeHF.body)

    def test_a_server_that_sends_the_whole_file_again_starts_over(self):
        with open(self.dest + ".part", "wb") as f:
            f.write(FakeHF.body[:1000])
        FakeHF.ignore_range = True
        picker.download(self.p, self.models)
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), FakeHF.body)

    def test_a_wrong_part_is_found_by_the_hash_and_removed(self):
        with open(self.dest + ".part", "wb") as f:
            f.write(b"x" * 1000)                                    # not the start of this file
        with self.assertRaises(SystemExit) as ex:
            picker.download(self.p, self.models)
        self.assertIn("doesn't match its SHA-256", str(ex.exception))
        self.assertFalse(os.path.exists(self.dest))
        self.assertFalse(os.path.exists(self.dest + ".part"))
        self.assertFalse(os.path.exists(self.dest + ".ok"))
        picker.download(self.p, self.models)                          # the next try starts clean
        self.assertTrue(picker.done(self.p, self.models))

    def test_a_bigger_file_than_listed_is_refused(self):
        p = self.p._replace(size=len(FakeHF.body) - 10)
        with self.assertRaises(SystemExit) as ex:
            picker.download(p, self.models)
        self.assertIn("bigger than it should be", str(ex.exception))
        self.assertFalse(os.path.exists(self.dest))

    def test_not_enough_disk_space_downloads_nothing(self):
        usage = shutil.disk_usage(self.models)._replace(free=len(FakeHF.body))
        with mock.patch("shutil.disk_usage", return_value=usage), self.assertRaises(SystemExit) as ex:
            picker.download(self.p, self.models)
        self.assertIn("not enough disk space", str(ex.exception))
        self.assertEqual(FakeHF.seen, [])

    def test_a_file_without_its_note_is_not_done(self):
        with open(self.dest, "wb") as f:
            f.write(FakeHF.body)
        self.assertFalse(picker.done(self.p, self.models))
        self.assertEqual([n for n, _, note in picker.installed(self.models) if note is None], [self.p.name])

    def test_get_shows_licence_and_size_then_downloads_and_registers(self):
        with mock.patch.object(picker, "LIST", (self.p,)), mock.patch("sys.stdout", new=io.StringIO()) as out:
            self.assertEqual(cli.main(["models", "get", self.p.name]), 0)
        text = out.getvalue()
        self.assertLess(text.index("licence: MIT"), text.index("Downloading"))
        self.assertIn(f"{self.p.size} bytes", text)
        self.assertIn("default folders", text)
        self.assertTrue(picker.done(self.p, self.models))
        with mock.patch.object(picker, "LIST", (self.p,)), mock.patch("sys.stdout", new=io.StringIO()) as out:
            cli.main(["models", "get", self.p.name])
        self.assertIn("Already downloaded", out.getvalue())
        models, _ = catalog.load(config=catalog.CONFIG)
        self.assertEqual(list(models), [self.p.name])               # lodger serve finds it under its name


class RegisterTest(Base):
    def edit(self, text):
        entry = "~/.local/share/lodger/models"
        new = picker.with_folder(text, entry)
        self.assertIsNotNone(new, text)
        before, after = tomllib.loads(text), tomllib.loads(new)
        self.assertEqual(catalog.listed(after, "folders"), catalog.listed(before, "folders") + [entry])
        return new

    def test_arrays_of_every_shape_keep_their_comments(self):
        self.edit('folders = []\n')
        self.edit('folders = ["~/models"]\n[defaults]\ntemp = 0.7\n')
        new = self.edit('# mine\nfolders = [\n  "~/models",  # the big drive\n  \'~/b]c\',\n]\nignore = ["*x*"]\n')
        self.assertIn("# the big drive", new)
        self.assertIn("# mine", new)
        self.edit('folders = [ "~/a" , # c ]\n  "~/b" # d\n]\n')
        self.edit('folders = "~/models"\n')
        self.edit("folders = '~/models' # one\n")

    def test_what_it_cant_do_safely_it_leaves_alone(self):
        self.assertIsNone(picker.with_folder('[defaults]\nfolders = ["~/x"]\n', "~/y"))   # not the top-level key
        self.assertIsNone(picker.with_folder('ignore = []\n', "~/y"))
        self.assertIsNone(picker.with_folder('folders = [\n', "~/y"))
        self.assertIsNone(picker.with_folder('folders = """~/odd"""\n', "~/y"))

    def test_register(self):
        cfg = catalog.CONFIG
        self.assertIn("default folders", picker.register(self.models, cfg))       # no models.toml
        with open(cfg, "w") as f:
            f.write('default = "x"\n[models.x]\nfile = "/m/x.gguf"\n')
        self.assertIn("default folders", picker.register(self.models, cfg))       # no folders: the default
        with open(cfg, "w") as f:
            f.write('folders = ["/elsewhere"]  # mine\n')
        os.chmod(cfg, 0o640)
        self.assertIn("Added", picker.register(self.models, cfg))
        with open(cfg, "rb") as f:
            self.assertEqual(tomllib.load(f)["folders"], ["/elsewhere", self.models])
        self.assertEqual(os.stat(cfg).st_mode & 0o777, 0o640)
        self.assertIn("lists that folder", picker.register(self.models, cfg))     # only once
        with open(cfg, "w") as f:
            f.write("folders = [\n")
        self.assertIn("NOTE: can't read", picker.register(self.models, cfg))     # broken: left alone
        with open(cfg) as f:
            self.assertEqual(f.read(), "folders = [\n")

    def test_register_keeps_a_linked_config_a_link_and_its_line_ends(self):
        real = os.path.join(self.dir, "dotfiles-models.toml")
        with open(real, "wb") as f:
            f.write(b'# m\xc3\xafne\r\nfolders = ["/elsewhere"]\r\n')
        cfg = os.path.join(self.dir, "linked.toml")
        os.symlink(real, cfg)
        self.assertIn("Added", picker.register(self.models, cfg))
        self.assertTrue(os.path.islink(cfg))
        with open(real, "rb") as f:
            self.assertEqual(f.read(), b'# m\xc3\xafne\r\nfolders = ["/elsewhere", ' + json.dumps(self.models).encode()
                             + b']\r\n')
        self.assertFalse([n for n in os.listdir(self.dir) if "lodger-tmp" in n])

    def test_register_into_a_folder_it_cant_write_says_so(self):
        d = os.path.join(self.dir, "ro")
        os.mkdir(d)
        real = os.path.join(d, "models.toml")
        with open(real, "w") as f:
            f.write('folders = ["/elsewhere"]\n')
        cfg = os.path.join(self.dir, "linked-ro.toml")
        os.symlink(real, cfg)
        os.chmod(d, 0o555)
        self.addCleanup(os.chmod, d, 0o755)
        if os.access(d, os.W_OK):
            self.skipTest("running as root")
        msg = picker.register(self.models, cfg)                 # after a download: a note, not a traceback
        self.assertIn("NOTE: lodger couldn't write", msg)
        self.assertIn(self.models, msg)

    def test_catalog_scans_the_download_folder_unless_folders_are_set(self):
        open(os.path.join(self.models, "tiny-1b.gguf"), "wb").close()
        open(os.path.join(self.models, "tiny-1b.gguf.part"), "wb").close()
        missing = os.path.join(self.dir, "missing.toml")
        self.assertEqual(list(catalog.load(config=missing)[0]), ["tiny-1b"])
        cfg = os.path.join(self.dir, "own.toml")
        with open(cfg, "w") as f:
            f.write(f'folders = ["{self.dir}/elsewhere"]\n')
        self.assertEqual(catalog.load(config=cfg)[0], {})


class TrashTest(Base):
    def setUp(self):
        super().setUp()
        self.p = self.pick(b"x" * 100)
        self.path = os.path.join(self.models, self.p.name + ".gguf")
        for x in (self.path, self.path + ".ok"):
            with open(x, "w") as f:
                f.write("{}")

    def test_without_gio_it_uses_the_home_trash(self):
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": self.dir}), mock.patch("shutil.which", return_value=None):
            picker.trash(self.path)
            other = os.path.join(self.dir, "x", self.p.name + ".gguf")
            os.makedirs(os.path.dirname(other))
            open(other, "w").close()
            picker.trash(other)                                     # same name: a second slot
        files = sorted(os.listdir(os.path.join(self.dir, "Trash", "files")))
        self.assertEqual(files, [self.p.name + ".1.gguf", self.p.name + ".gguf"])
        with open(os.path.join(self.dir, "Trash", "info", self.p.name + ".gguf.trashinfo")) as f:
            info = f.read()
        self.assertIn(f"Path={self.path}\n", info)
        self.assertRegex(info, r"DeletionDate=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")
        self.assertFalse(os.path.exists(self.path))

    def test_with_gio(self):
        with mock.patch("shutil.which", return_value="/usr/bin/gio"), mock.patch("subprocess.run") as run:
            run.return_value.returncode = 0
            picker.trash(self.path)
        self.assertEqual(run.call_args[0][0], ["gio", "trash", "--", self.path])

    def test_rm_trashes_the_model_and_its_part_and_drops_the_note(self):
        open(self.path + ".part", "w").close()
        moved = []
        with mock.patch.object(picker, "trash", side_effect=lambda p: (moved.append(p), os.remove(p))), \
                mock.patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(cli.main(["models", "rm", self.p.name]), 0)
        self.assertEqual(moved, [self.path, self.path + ".part"])
        self.assertFalse(os.path.exists(self.path + ".ok"))
        for bad in ("nothing-here", "../models/" + self.p.name, ".lock"):
            with self.subTest(bad), self.assertRaises(SystemExit):
                picker.rm(bad)


class InUseTest(unittest.TestCase):
    def test_a_mapped_or_open_file_is_in_use(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.gguf")
            with open(path, "wb") as f:
                f.write(b"\0" * 4096)
            self.assertEqual(picker.in_use(path), [])
            with open(path, "rb") as f:
                self.assertEqual(picker.in_use(path), [os.getpid()])          # open
                mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                self.assertEqual(picker.in_use(path), [os.getpid()])          # mapped, the file closed
            finally:
                mapped.close()
            self.assertEqual(picker.in_use(path), [])
            self.assertEqual(picker.in_use(os.path.join(d, "gone.gguf")), [])

    def test_rm_says_when_the_model_is_in_use(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(picker, "folder", return_value=d), \
                mock.patch.object(picker, "in_use", return_value=[4242]), \
                mock.patch.object(picker, "trash", side_effect=os.remove), \
                mock.patch("sys.stdout", new=io.StringIO()) as out:
            open(os.path.join(d, "x.gguf"), "w").close()
            picker.rm("x")
        self.assertIn("in use right now (process 4242)", out.getvalue())


class FitTest(Base):
    def plan(self, refuse=None, ngl=0, layers=40, n_cpu_moe=0, moe=False, ctx=32768, slow=False):
        why = [] if ngl else ["Graphics card not used: the PC is on battery."]
        return {"refuse": refuse, "ngl": ngl, "ngl_layers": min(ngl, layers), "layers": layers, "moe": moe,
                "n_cpu_moe": n_cpu_moe, "ctx": ctx, "slow": slow, "why": why}

    def picks(self):
        return [self.pick(b"", name="small", size=2 * GiB), self.pick(b"", name="moe", size=5 * GiB, moe=True),
                self.pick(b"", name="mid", size=6 * GiB), self.pick(b"", name="big", size=18 * GiB)]

    def test_plans_say_where_and_the_largest_fitting_of_each_kind_is_suggested(self):
        plans = {"small": self.plan(ngl=41), "moe": self.plan(ngl=41, n_cpu_moe=10, moe=True),
                 "mid": self.plan(refuse="Not enough RAM: the model needs 9 GiB")}
        v, notes, card = picker.judge(self.picks(), lambda p: plans[p.name], 8 * GiB, 8 * GiB)
        self.assertEqual(v["small"], (True, "yes: all on the graphics card, 32K context", True))
        self.assertEqual(v["moe"][1], "yes: on the graphics card, experts of 10 of 40 layers in RAM, 32K context")
        self.assertEqual(v["mid"], (False, "no: Not enough RAM", False))
        self.assertFalse(v["big"][0])                                 # bigger than all free memory: not planned
        self.assertEqual(notes, [])
        self.assertEqual([p.name for p in picker.suggest(self.picks(), v)], ["small", "moe"])

    def test_a_model_wholly_on_the_card_is_suggested_over_a_bigger_split_one(self):
        plans = {"small": self.plan(ngl=41), "mid": self.plan(ngl=20), "moe": self.plan(ngl=41, n_cpu_moe=30, moe=True)}
        v, _, _ = picker.judge(self.picks()[:3], lambda p: plans[p.name], 8 * GiB, 16 * GiB)
        self.assertEqual(v["mid"][1], "yes: 20 of 40 layers on the graphics card, 32K context")
        self.assertEqual([p.name for p in picker.suggest(self.picks()[:3], v)], ["small", "moe"])
        v["small"] = (True, "yes: on the processor", False)          # no card at all: simply the largest
        v["mid"] = (True, "yes: on the processor", False)
        self.assertEqual(picker.suggest(self.picks()[:3], v)[0].name, "mid")

    def test_without_llama_cpp_the_guesses_are_rough_and_say_so(self):
        v, notes, _ = picker.judge(self.picks(), None, 4 * GiB, 8 * GiB, "lodger can't find llama.cpp")
        self.assertEqual(v["small"], (True, "probably (rough guess)", False))
        self.assertEqual(v["mid"], (True, "probably (rough guess)", False))
        self.assertFalse(v["big"][0])
        self.assertEqual(notes, ["lodger can't find llama.cpp"])

    def test_one_failed_plan_doesnt_spoil_the_others_but_no_network_stops_asking(self):
        import urllib.error
        calls = []

        def plan_one(p):
            calls.append(p.name)
            if p.name == "small":
                raise SystemExit("llama-fit-params failed (exit 1):\nunknown model architecture")
            if p.name == "moe":
                raise urllib.error.URLError("no network")
            return self.plan(ngl=41)
        v, notes, _ = picker.judge(self.picks(), plan_one, 16 * GiB, 8 * GiB)
        self.assertEqual(calls, ["small", "moe"])
        self.assertEqual(v["small"][1], "probably (rough guess)")
        self.assertEqual(notes[0], "small: llama-fit-params failed (exit 1):")
        self.assertIn("can't reach Hugging Face", notes[1])
        self.assertIn("rough guess", v["mid"][1])

    def test_the_list_plans_and_suggests_without_downloading(self):
        p = self.pick(b"", name="small", size=2 * GiB)
        with mock.patch.object(picker, "LIST", (p,)), \
                mock.patch.object(picker, "planner", return_value=(lambda pk: self.plan(), None, None, {"MemAvailable": 8 * GiB})), \
                mock.patch("sys.stdout", new=io.StringIO()) as out:
            self.assertEqual(cli.main(["models"]), 0)
        text = out.getvalue()
        self.assertIn("none yet", text)
        self.assertIn("yes: on the processor, 32K context", text)
        self.assertIn("Graphics card not used: the PC is on battery.", text)
        self.assertIn("Suggested: small (1B).", text)
        self.assertEqual(os.listdir(self.models), [])


if __name__ == "__main__":
    unittest.main()
