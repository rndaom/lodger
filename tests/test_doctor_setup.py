"""lodger doctor on recorded machines (never waking a sleeping card), and lodger setup's choice of file,
checksum and unpacking (no network)."""
import contextlib
import hashlib
import io
import os
import tarfile
import tempfile
import unittest
from unittest import mock

from lodger import config, doctor, setup
from lodger.gpu import nvidia

from tests.support import FakeNVML, machine


def run_doctor(orphans=(), capped=(True, None), describe=None, settings=os.devnull):
    out = io.StringIO()
    with contextlib.ExitStack() as st:
        st.enter_context(contextlib.redirect_stdout(out))
        st.enter_context(mock.patch("lodger.fence.mode", return_value=("systemd", "systemd user session")))
        st.enter_context(mock.patch("lodger.fence.systemd_memory", return_value=capped))
        st.enter_context(mock.patch("lodger.fence.delegated", return_value=(None, "not delegated")))
        st.enter_context(mock.patch("lodger.fence.find_orphans", return_value=list(orphans)))
        st.enter_context(mock.patch("lodger.catalog.load", return_value=({}, None)))
        st.enter_context(mock.patch.object(config, "PATH", settings))
        st.enter_context(mock.patch("lodger.engine.root", return_value=(None, "not found")))
        if describe:
            st.enter_context(mock.patch("lodger.engine.describe", return_value=describe))
        doctor.main([])
    return out.getvalue()


class DoctorTest(unittest.TestCase):
    def setUp(self):
        config.use(config.load(os.devnull))

    def tearDown(self):
        config.use(None)

    def test_desktop(self):
        with machine("desktop-amd"):
            text = run_doctor()
        self.assertIn("0000:03:00.0", text)
        self.assertIn("<- lodger uses this one", text)
        self.assertIn("no llama.cpp", text)

    def test_sleeping_nvidia_is_not_woken(self):
        fake = FakeNVML()
        with machine("laptop-nvidia"):
            nvidia._nvml = fake
            text = run_doctor()
        self.assertEqual(fake.calls, 0)
        self.assertIn("not read (the card is asleep", text)
        self.assertIn("lodger leaves it alone", text)

    def test_left_behind_and_fence_cap(self):
        with machine("desktop-amd"):
            text = run_doctor(orphans=[{"kind": "systemd", "name": "lodger-4242-1", "owner": 4242,
                                        "where": "lodger-4242-1.service"}], capped=(False, "cgroup v1 or hybrid"))
        self.assertIn("LEFT BEHIND: lodger-4242-1", text)
        self.assertIn("NOT applied (cgroup v1 or hybrid)", text)
        self.assertIn("no hard memory fence", text)

    def test_settings_problems_are_listed(self):
        with tempfile.TemporaryDirectory() as tmp, machine("desktop-amd"):
            p = os.path.join(tmp, "config.toml")
            with open(p, "w") as f:
                f.write("gpu = 3\nwake_gpuu = true\n")
            text = run_doctor(settings=p)
        self.assertIn("problem: " + p + ": gpu must be text", text)
        self.assertIn("unknown setting 'wake_gpuu'", text)

    def test_vulkan_layer_and_cuda_driver(self):
        d = {"root": "/x", "from": "test", "bin": "/x", "backends": ["cuda", "vulkan"], "cuda": 13,
             "llama-server": True, "llama-fit-params": True, "kfd": False}
        with machine("laptop-nvidia"), mock.patch("lodger.engine.vulkan_layer", return_value=None), \
                mock.patch("lodger.engine.for_device", return_value={"why": "test"}):
            text = run_doctor(describe=d)
        self.assertIn("device-select layer NOT found", text)
        self.assertIn("CUDA: build for CUDA 13, NVIDIA driver 570.153.2 (TOO OLD", text)

    def test_apu_and_arc(self):
        with machine("laptop-apu"):
            self.assertIn("can borrow", run_doctor())
        with machine("desktop-arc"):
            self.assertIn("can't read free memory on Intel Arc", run_doctor())


def release(tag, *names):
    return {"tag_name": tag, "draft": False, "prerelease": True,
            "assets": [{"name": n, "size": 1, "digest": "sha256:00", "browser_download_url": "x"} for n in names]}


class SetupTest(unittest.TestCase):
    RELS = [
        {"tag_name": "v0.8.0", "draft": False, "assets": [{"name": "nightly-tag.txt"}]},
        release("b200", "llama-b200-bin-ubuntu-x64.tar.gz"),
        release("b199", "llama-b199-bin-ubuntu-vulkan-x64.tar.gz", "llama-b199-bin-ubuntu-x64.tar.gz",
                "llama-b199-bin-ubuntu-rocm-7.2-x64.tar.gz", "llama-b199-bin-ubuntu-rocm-10.0-x64.tar.gz",
                "llama-b199-bin-ubuntu-cuda-12.8-x64.tar.gz", "llama-b199-bin-ubuntu-cuda-13.4-x64.tar.gz",
                "cudart-llama-b199-bin-ubuntu-cuda-13.4-x64.tar.gz", "llama-b199-bin-ubuntu-26.04-s390x.tar.gz"),
    ]

    def name(self, *a, **kw):
        r = setup.pick(self.RELS, *a, **kw)
        return r and (r[0]["tag_name"], r[1]["name"], [x["name"] for x in r[2]])

    def test_pick(self):
        self.assertEqual(self.name("cpu", "x64"), ("b200", "llama-b200-bin-ubuntu-x64.tar.gz", []))
        self.assertEqual(self.name("vulkan", "x64")[1], "llama-b199-bin-ubuntu-vulkan-x64.tar.gz")
        self.assertEqual(self.name("rocm", "x64")[1], "llama-b199-bin-ubuntu-rocm-10.0-x64.tar.gz")
        self.assertEqual(self.name("cuda", "x64")[2], ["cudart-llama-b199-bin-ubuntu-cuda-13.4-x64.tar.gz"])
        self.assertEqual(self.name("cpu", "x64", tag="b199")[0], "b199")
        self.assertIsNone(self.name("vulkan", "arm64"))

    def archive(self, folder, data):
        path = os.path.join(folder, "a.tar.gz")
        with tarfile.open(path, "w:gz") as t:
            for n in ("llama-server", "llama-fit-params"):        # in one top folder, like the real archives
                info = tarfile.TarInfo(os.path.join("llama-b1", n))
                info.size, info.mode = len(data), 0o755
                t.addfile(info, io.BytesIO(data))
            link = tarfile.TarInfo("llama-b1/libx.so")
            link.type, link.linkname = tarfile.SYMTYPE, "llama-server"
            t.addfile(link)
        return path

    def test_download_checks_and_unpacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self.archive(tmp, b"#!/bin/sh\n")
            with open(src, "rb") as f:
                body = f.read()
            good = {"name": "a.tar.gz", "size": len(body), "browser_download_url": "https://example.invalid/a.tar.gz",
                    "digest": "sha256:" + hashlib.sha256(body).hexdigest()}
            dl = os.path.join(tmp, "dl")
            os.makedirs(dl)
            with mock.patch("urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(body)), \
                    contextlib.redirect_stderr(io.StringIO()):
                path = setup.download(good, dl)
                with self.assertRaises(SystemExit):
                    setup.download(dict(good, digest="sha256:" + "0" * 64), dl)
                with self.assertRaises(SystemExit):
                    setup.download(dict(good, digest=None), dl)
            dest = os.path.join(tmp, "b1-cpu")
            setup.unpack(path, dest)
            self.assertTrue(os.access(os.path.join(dest, "llama-server"), os.X_OK))
            self.assertEqual(os.readlink(os.path.join(dest, "libx.so")), "llama-server")

    def test_unpack_refuses_escapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.tar.gz")
            with tarfile.open(path, "w:gz") as t:
                info = tarfile.TarInfo("../escaped")
                t.addfile(info, io.BytesIO(b""))
            with self.assertRaises(tarfile.TarError):
                setup.unpack(path, os.path.join(tmp, "dest"))
            self.assertFalse(os.path.exists(os.path.join(tmp, "escaped")))

    def test_refuses_without_tar_filters(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(tarfile.__dict__):
            del tarfile.__dict__["data_filter"]                  # Python 3.11.0-3.11.3
            with self.assertRaises(SystemExit) as ex:
                setup.unpack(self.archive(tmp, b"x"), os.path.join(tmp, "dest"))
            self.assertIn("3.11.4", str(ex.exception))


CUDA = [release("b300", "llama-b300-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b300-bin-ubuntu-cuda-12.8-x64.tar.gz",
                "llama-b300-bin-ubuntu-cuda-13.4-x64.tar.gz", "cudart-llama-b300-bin-ubuntu-cuda-13.4-x64.tar.gz")]


def tar(folder, name, files, top="llama-b300"):
    path = os.path.join(folder, name)
    with tarfile.open(path, "w:gz") as t:
        for n, data in files.items():
            info = tarfile.TarInfo(os.path.join(top, n))
            info.size, info.mode = len(data), 0o755
            t.addfile(info, io.BytesIO(data))
    with open(path, "rb") as f:
        body = f.read()
    return {"name": name, "size": len(body), "browser_download_url": "file://" + path,
            "digest": "sha256:" + hashlib.sha256(body).hexdigest(), "body": body}


class SetupInstallTest(unittest.TestCase):
    def cuda(self, **kw):
        return setup.pick(CUDA, "cuda", "x64", **kw)[1]["name"]

    def test_cuda_by_driver(self):
        self.assertIn("cuda-12.8", self.cuda(driver=(570, 153, 2)))      # CUDA 13 needs 580+
        self.assertIn("cuda-13.4", self.cuda(driver=(580, 65, 6)))
        self.assertIn("cuda-12.8", self.cuda(driver=None))               # can't tell: the most compatible
        self.assertIn("cuda-13.4", self.cuda(driver=(570, 1, 1), want="13"))
        self.assertIsNone(setup.pick(CUDA, "cuda", "x64", driver=(510, 1, 1)))

    def test_rocm_version(self):
        self.assertIn("rocm-7.2", setup.pick(SetupTest.RELS, "rocm", "x64", want="7.2")[1]["name"])

    def fetch(self, assets):
        bodies = {a["browser_download_url"]: a["body"] for a in assets}
        return mock.patch("urllib.request.urlopen", side_effect=lambda req, **k: io.BytesIO(bodies[req.full_url]))

    def test_two_archives_with_the_same_folder_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            main = tar(tmp, "main.tar.gz", {"llama-server": b"s", "llama-fit-params": b"f", "llama-cli": b"c", "lib/libggml.so": b"g"})
            rt = tar(tmp, "cudart.tar.gz", {"lib/libcudart.so.12": b"r"})
            base = os.path.join(tmp, "base")
            os.makedirs(base)
            with self.fetch([main, rt]), contextlib.redirect_stderr(io.StringIO()):
                dest = setup.install([main, rt], base, "b300-cuda")
            self.assertEqual(sorted(os.listdir(os.path.join(dest, "lib"))), ["libcudart.so.12", "libggml.so"])
            self.assertTrue(setup.complete(dest, [main, rt]))
            self.assertFalse(setup.complete(dest, [main]))              # another set of files isn't this install
            self.assertEqual([n for n in os.listdir(base) if n.startswith(".")], [])

    def test_a_bad_second_file_installs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            main = tar(tmp, "main.tar.gz", {"llama-server": b"s", "llama-fit-params": b"f", "llama-cli": b"c"})
            rt = dict(tar(tmp, "cudart.tar.gz", {"lib/libcudart.so.12": b"r"}), digest="sha256:" + "0" * 64)
            base = os.path.join(tmp, "base")
            os.makedirs(base)
            with self.fetch([main, rt]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                setup.install([main, rt], base, "b300-cuda")
            self.assertEqual(os.listdir(base), [])                     # nothing unpacked, nothing to mistake for done

    def test_leftovers_of_a_crash_are_cleaned(self):
        with tempfile.TemporaryDirectory() as base:
            for n in ("b300-cuda.unpack", ".download-abc", ".unpack-xyz"):
                os.makedirs(os.path.join(base, n, "inner"))
            os.makedirs(os.path.join(base, "b299-vulkan"))
            os.symlink("b299-vulkan", os.path.join(base, "current.new"))
            with setup.locked(base):
                setup.clean_stale(base)
                with self.assertRaises(SystemExit):
                    with setup.locked(base):
                        pass
            self.assertEqual(sorted(os.listdir(base)), [".setup.lock", "b299-vulkan"])

    def test_a_folder_without_the_marker_is_not_done(self):
        with tempfile.TemporaryDirectory() as dest:
            for n in setup.NEEDED:
                open(os.path.join(dest, n), "w").close()
            self.assertFalse(setup.complete(dest, [{"name": "a.tar.gz"}]))

    def test_tag_is_asked_for_by_name(self):
        urls = []

        def get(url):
            urls.append(url)
            return release("b199", "llama-b199-bin-ubuntu-x64.tar.gz")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "DATA", tmp), \
                mock.patch("lodger.setup.get_json", side_effect=get), \
                mock.patch("lodger.setup.install", side_effect=lambda assets, base, name: os.makedirs(os.path.join(base, name))), \
                contextlib.redirect_stdout(io.StringIO()):
            config.use(config.load(os.devnull))
            try:
                setup.main(["--backend", "cpu", "--tag", "b199"])
            finally:
                config.use(None)
            self.assertEqual(urls, [setup.API + "/tags/b199"])
            self.assertEqual(os.readlink(os.path.join(tmp, "llama.cpp", "current")), "b199-cpu")

    def test_github_errors_are_plain(self):
        import email.message
        import urllib.error
        hdr = email.message.Message()
        hdr["X-RateLimit-Remaining"] = "0"
        limit = urllib.error.HTTPError(setup.API, 403, "rate limit exceeded", hdr, io.BytesIO(b""))
        self.addCleanup(limit.close)
        with mock.patch("urllib.request.urlopen", side_effect=limit), self.assertRaises(SystemExit) as ex:
            setup.get_json(setup.API)
        self.assertIn("GITHUB_TOKEN", str(ex.exception))
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no network")), \
                self.assertRaises(SystemExit) as ex:
            setup.get_json(setup.API)
        self.assertIn("can't reach GitHub", str(ex.exception))

    def test_token_only_for_the_api(self):
        seen = []

        def opener(req, **kw):
            seen.append((req.full_url, req.get_header("Authorization")))
            return io.BytesIO(b"[]")
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "t0ken"}), mock.patch("urllib.request.urlopen", side_effect=opener):
            setup.get_json(setup.API)
            setup.get_json("https://example.invalid/x.json")
        self.assertEqual(seen, [(setup.API, "Bearer t0ken"), ("https://example.invalid/x.json", None)])


if __name__ == "__main__":
    unittest.main()
