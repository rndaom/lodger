"""The planner, with no graphics card and no llama.cpp: model sizes and llama.cpp's estimates are made up."""
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from lodger import config, engine, gpu, plan
from lodger.model import Model
from tests.support import machine

MiB, GiB = 1 << 20, 1 << 30


def fake_model(layers=40, dense=100 * MiB, experts=0, embd=500 * MiB, ctx_train=262144):
    m = object.__new__(Model)
    m.path, m.files = "/models/fake.gguf", ["/models/fake.gguf"]
    m.layers = {i: {"dense": dense, "experts": experts} for i in range(layers)}
    m.token_embd, m.output, m.other = embd, embd, 0
    m.n_layer = layers
    m.meta = {"name": "Fake", "arch": "fake", "ctx_train": ctx_train,
              "expert_count": 128 if experts else None, "expert_used": 8 if experts else None}
    return m


def fake_probe(model, ctx, gpu=None, extra=(), dev=None):
    # context: 64 KiB per token plus 100 MiB; scratch 300 MiB
    return {"model": 0, "context": 100 * MiB + ctx * 64 * 1024, "compute": 300 * MiB}


def fake_build(*libs):
    """A llama.cpp folder with empty programs and the given backend libraries."""
    d = tempfile.mkdtemp(prefix="lodger-build-")
    os.makedirs(os.path.join(d, "build", "bin"))
    for n in ("llama-server", "llama-fit-params", "llama-cli") + libs:
        open(os.path.join(d, "build", "bin", n), "w").close()
    return d


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.runs = tempfile.mkdtemp(prefix="lodger-runs-")
        self.build = fake_build("libggml-hip.so")
        cfg = config.load("/nonexistent")
        cfg.update(runs_dir=self.runs, llama_cpp=self.build)
        config.use(cfg)
        self.mem = {"MemTotal": 32 * GiB, "MemAvailable": 20 * GiB}
        p = mock.patch("lodger.plan.probe", side_effect=fake_probe)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        config.use(None)
        for d in (self.runs, self.build):
            subprocess.run(["rm", "-rf", d])

    def make(self, m, **kw):
        kw.setdefault("mem", self.mem)
        kw.setdefault("threads", 8)
        return plan.make(m, **kw)

    def test_small_model_goes_on_the_card_whole(self):
        with machine("desktop-amd"):
            p = self.make(fake_model(), vram_free=15 * GiB)
        self.assertIsNone(p["refuse"])
        self.assertEqual(p["ngl"], 41)                    # every layer + the output layer
        self.assertEqual(p["n_cpu_moe"], 0)
        self.assertTrue(p["output_on_gpu"])
        self.assertLessEqual(p["gpu_bytes"], 15 * GiB - p["keep"])

    def test_big_moe_keeps_some_experts_in_ram(self):
        m = fake_model(dense=60 * MiB, experts=400 * MiB)
        with machine("desktop-amd"):
            p = self.make(m, vram_free=12 * GiB)
        self.assertIsNone(p["refuse"])
        self.assertEqual(p["ngl"], 41)
        self.assertGreater(p["n_cpu_moe"], 0)
        self.assertLess(p["n_cpu_moe"], 40)
        self.assertLessEqual(p["gpu_bytes"], 12 * GiB - p["keep"])

    def test_no_card_runs_on_the_processor_and_says_why(self):
        p = self.make(fake_model(), card=None)
        self.assertEqual(p["ngl"], 0)
        self.assertTrue(any("no graphics card" in w for w in p["why"]), p["why"])

    def test_card_memory_unknown_means_processor(self):
        with machine("desktop-arc"):
            p = self.make(fake_model())
        self.assertEqual(p["ngl"], 0)
        self.assertTrue(any("Arc" in w for w in p["why"]), p["why"])

    def test_engine_that_cant_drive_the_card(self):
        with machine("laptop-nvidia", writable=True) as root:
            from tests.support import FakeNVML, write
            from lodger.gpu import nvidia
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "active\n")
            nvidia._nvml = FakeNVML()
            p = self.make(fake_model(), card=gpu.choose())
        self.assertEqual(p["ngl"], 0)
        self.assertTrue(any("can't drive a NVIDIA card" in w for w in p["why"]), p["why"])

    def test_llama_cpp_check_uses_the_chosen_device(self):
        seen = []

        def probe(model, ctx, gpu=None, extra=(), dev=None):
            seen.append((gpu, dev))
            return fake_probe(model, ctx)
        with machine("desktop-amd"), mock.patch("lodger.plan.probe", side_effect=probe):
            p = self.make(fake_model(), vram_free=15 * GiB, device_check=True)
        self.assertIsNotNone(p.get("llama_gpu_est"))
        self.assertTrue(any(g and d is not None and d.pci == "0000:03:00.0" for g, d in seen))


class ProbeTest(unittest.TestCase):
    """--fit-print lines start with the device's name: whatever build, never a fixed 'ROCm0'."""

    def setUp(self):
        self.runs = tempfile.mkdtemp(prefix="lodger-runs-")

    def tearDown(self):
        config.use(None)
        subprocess.run(["rm", "-rf", self.runs])

    def run_probe(self, libs, gpu_place, out):
        build = fake_build(*libs)
        self.addCleanup(subprocess.run, ["rm", "-rf", build])
        cfg = config.load("/nonexistent")
        cfg.update(runs_dir=self.runs, llama_cpp=build)
        config.use(cfg)
        m = fake_model()
        m.files = [os.path.join(build, "build", "bin", "llama-cli")]     # any file: only its size and time are read
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        with machine("desktop-amd"), mock.patch("lodger.plan.subprocess.run", side_effect=run):
            r = plan.probe(m, 4096, gpu_place, dev=gpu.choose() if gpu_place else None)
        return r, calls[0]

    def test_vulkan_build(self):
        r, cmd = self.run_probe(["libggml-vulkan.so"], (41, 0), "Vulkan0 1000 200 300 \nHost 10 0 5 \n")
        self.assertEqual(r["model"], 1000 * MiB)
        self.assertIn("Vulkan0", cmd)
        self.assertIn("--setenv=GGML_VK_VISIBLE_DEVICES=0", cmd)
        self.assertIn("--setenv=DRI_PRIME=pci-0000_03_00_0", cmd)

    def test_rocm_build(self):
        r, cmd = self.run_probe(["libggml-hip.so"], (41, 3), "ROCm0 900 100 200 \nHost 10 0 5 \n")
        self.assertEqual(r["compute"], 200 * MiB)
        self.assertIn("--setenv=HIP_VISIBLE_DEVICES=0", cmd)
        self.assertEqual(cmd[cmd.index("-ncmoe") + 1], "3")

    def test_processor_only_hides_the_cards(self):
        r, cmd = self.run_probe(["libggml-vulkan.so"], None, "Host 10 20 30 \n")
        self.assertEqual(r["context"], 20 * MiB)
        self.assertIn("PrivateDevices=yes", cmd)
        self.assertIn("--setenv=GGML_VK_VISIBLE_DEVICES=", cmd)
        self.assertEqual(cmd[cmd.index("--device") + 1], "none")


class EngineTest(unittest.TestCase):
    def tearDown(self):
        config.use(None)

    def test_build_kinds(self):
        for libs, want in ((("libggml-hip.so",), {"rocm"}), (("libggml-cuda.so.0",), {"cuda"}),
                           (("libggml-vulkan.so",), {"vulkan"}), ((), set())):
            b = fake_build(*libs)
            try:
                self.assertEqual(engine.backends(os.path.join(b, "build", "bin")), want)
            finally:
                subprocess.run(["rm", "-rf", b])

    def test_nvidia_gets_cuda_by_pci_order(self):
        with machine("laptop-nvidia"):
            devs = gpu.devices()
            t = engine.for_device(devs[1], devs, have={"cuda", "vulkan"})
        self.assertEqual(t["name"], "CUDA0")
        self.assertEqual(t["env"], {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"})

    def test_vulkan_on_intel_keeps_the_sleeping_nvidia_driver_out(self):
        with machine("laptop-nvidia"):
            devs = gpu.devices()
            t = engine.for_device(devs[0], devs, have={"vulkan"})
        self.assertEqual(t["name"], "Vulkan0")
        self.assertEqual(t["env"]["VK_LOADER_DRIVERS_DISABLE"], "*nvidia*")
        self.assertEqual(t["env"]["MESA_VK_DEVICE_SELECT"], "8086:a7a0!")

    def test_wrong_card_in_the_log(self):
        with machine("desktop-amd"):
            d = gpu.choose()
            ok = "load: using device ROCm0 (AMD Radeon RX 7800 XT) (0000:03:00.0) - 16000 MiB free\n"
            bad = "load: using device Vulkan0 (AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)) (0000:11:00.0) - 400 MiB free\n"
            self.assertIsNone(engine.check_log(ok, d))
            self.assertIsNone(engine.check_log("still loading", d))
            self.assertIn("0000:11:00.0", engine.check_log(bad, d))

    def test_missing_llama_cpp_says_what_to_do(self):
        cfg = config.load("/nonexistent")
        cfg["llama_cpp"] = "/nonexistent/llama.cpp"
        config.use(cfg)
        with self.assertRaises(SystemExit) as e:
            engine.tool("llama-server")
        self.assertIn("lodger setup", str(e.exception))


if __name__ == "__main__":
    unittest.main()
