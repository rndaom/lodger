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


def run_doctor():
    out = io.StringIO()
    with contextlib.redirect_stdout(out), \
            mock.patch("lodger.fence.mode", return_value=("systemd", "systemd user session")), \
            mock.patch("lodger.fence.delegated", return_value=(None, "not delegated")), \
            mock.patch("lodger.catalog.load", return_value=({}, None)), \
            mock.patch("lodger.engine.root", return_value=(None, "not found")):
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


if __name__ == "__main__":
    unittest.main()
