"""The planner, with no graphics card and no llama.cpp: model sizes and llama.cpp's estimates are made up."""
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from lodger import config, engine, gpu, plan
from lodger.model import Model
from tests.support import machine
from tests.test_model import write_gguf

MiB, GiB = 1 << 20, 1 << 30


def fake_model(layers=40, dense=100 * MiB, experts=0, embd=500 * MiB, ctx_train=262144, other=0, output=None,
               kinds=None, quant=(), tied=False, kv=None):
    m = object.__new__(Model)
    m.path, m.files = "/models/fake.gguf", ["/models/fake.gguf"]
    m.layers = {i: {"dense": dense, "experts": experts} for i in range(layers)}
    m.token_embd, m.output, m.other = embd, embd if output is None else output, other
    m.n_layer = layers
    m.meta = {"name": "Fake", "arch": "fake", "ctx_train": ctx_train,
              "expert_count": 128 if experts else None, "expert_used": 8 if experts else None}
    if kinds:
        m.kinds = dict(enumerate(kinds))
    m.quant, m.tied, m.kv = list(quant), tied, kv or {}
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


class PlanBase(unittest.TestCase):
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


class PlanTest(PlanBase):
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
            config.get()["gpu_on_battery"] = True
            p = self.make(fake_model(), card=gpu.choose())
        self.assertEqual(p["ngl"], 0)
        self.assertTrue(any("can't drive a NVIDIA card" in w for w in p["why"]), p["why"])

    def test_no_card_work_on_battery(self):
        with machine("laptop-nvidia", writable=True) as root:
            from tests.support import FakeNVML, write
            from lodger.gpu import nvidia
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "active\n")
            nvidia._nvml = FakeNVML()
            p = self.make(fake_model(), card=gpu.choose())
        self.assertEqual(p["ngl"], 0)
        self.assertTrue(any("on battery" in w for w in p["why"]), p["why"])

    def test_sleeping_card_is_left_asleep(self):
        with machine("desktop-amd", writable=True) as root:
            from tests.support import write
            write(root, "/sys/bus/pci/devices/0000:03:00.0/power/runtime_status", "suspended\n")
            p = self.make(fake_model())
            self.assertEqual(p["ngl"], 0)
            self.assertTrue(any("asleep" in w for w in p["why"]), p["why"])
            config.get()["wake_gpu"] = True
            self.assertGreater(self.make(fake_model())["ngl"], 0)

    def test_built_in_graphics_share_ram(self):
        vk = fake_build("libggml-vulkan.so")
        self.addCleanup(subprocess.run, ["rm", "-rf", vk])
        config.get()["llama_cpp"] = vk
        with machine("laptop-apu"):
            mem = {"MemTotal": 28 * GiB, "MemAvailable": 18 * GiB}
            p = self.make(fake_model(), mem=mem)
        ram_keep = max(plan.RAM_KEEP_FREE_MIN, int(28 * GiB * plan.RAM_KEEP_FREE_FRAC))
        # min(RAM available - kept (13.8 GiB), GTT left (12 GiB))
        self.assertEqual(p["vram_free"], min(18 * GiB - ram_keep, 12 * GiB))
        self.assertTrue(p["shared_memory"])
        self.assertGreater(p["ngl"], 0)
        # what the graphics take is gone from RAM
        self.assertEqual(p["ram_budget"], 18 * GiB - ram_keep - p["gpu_bytes"])
        self.assertLessEqual(p["ram_cap"], p["ram_budget"])

    def test_reserve_follows_the_desktop(self):
        from lodger import reserve
        with machine("desktop-amd"):
            card = gpu.choose()
            base = self.make(fake_model(dense=250 * MiB), vram_free=15 * GiB)
            self.assertEqual(base["keep"], 2 * GiB)              # 1/8 of 16 GiB
            t0 = 1_000_000 * 60
            for i in range(120):                                  # two hours; every 30 min the desktop jumps 3 GiB
                reserve.note(card.pci, GiB + (3 * GiB if i % 30 == 5 else 0), now=t0 + i * 60)
            p = self.make(fake_model(dense=250 * MiB), vram_free=15 * GiB)
        reserve._notes.clear()
        self.assertEqual(p["keep"], 3 * GiB + reserve.HEADROOM)
        self.assertTrue(any("Desktop reserve" in w for w in p["why"]))
        self.assertLess(p["gpu_bytes"], base["gpu_bytes"])

    def test_refuses_without_enough_ram(self):
        p = self.make(fake_model(), card=None, ram_avail=3 * GiB)
        self.assertIsNotNone(p["refuse"])

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

    def run_probe(self, libs, gpu_place, out, code=0, err="", took=0):
        build = fake_build(*libs)
        self.addCleanup(subprocess.run, ["rm", "-rf", build])
        cfg = config.load("/nonexistent")
        cfg.update(runs_dir=self.runs, llama_cpp=build)
        config.use(cfg)
        m = fake_model()
        m.files = [os.path.join(build, "build", "bin", "llama-cli")]     # any file: only its size and time are read
        calls = []

        def run(cmd, env, props, timeout=300):
            calls.append((cmd, env, props))
            return subprocess.CompletedProcess(cmd, code, stdout=out, stderr=err)
        with machine("desktop-amd"), mock.patch("lodger.fence.run", side_effect=run), \
                mock.patch("lodger.plan.time.monotonic", side_effect=[0, took]):
            r = plan.probe(m, 4096, gpu_place, dev=gpu.choose() if gpu_place else None)
        return r, calls[0]

    def test_vulkan_build(self):
        r, (cmd, env, props) = self.run_probe(["libggml-vulkan.so"], (41, 0), "Vulkan0 1000 200 300 \nHost 10 0 5 \n")
        self.assertEqual(r["model"], 1000 * MiB)
        self.assertEqual(cmd[cmd.index("--device") + 1], "Vulkan0")
        self.assertEqual(env["GGML_VK_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["DRI_PRIME"], "pci-0000_03_00_0")
        self.assertNotIn("PrivateDevices", props)

    def test_rocm_build(self):
        r, (cmd, env, props) = self.run_probe(["libggml-hip.so"], (41, 3), "ROCm0 900 100 200 \nHost 10 0 5 \n")
        self.assertEqual(r["compute"], 200 * MiB)
        self.assertEqual(env, {"HIP_VISIBLE_DEVICES": "0", "GGML_NO_BACKTRACE": "1"})
        self.assertEqual(cmd[cmd.index("-ncmoe") + 1], "3")

    def test_processor_only_hides_the_cards(self):
        r, (cmd, env, props) = self.run_probe(["libggml-vulkan.so"], None, "Host 10 20 30 \n")
        self.assertEqual(r["context"], 20 * MiB)
        self.assertEqual(props["PrivateDevices"], "yes")
        self.assertEqual(env, {"GGML_VK_VISIBLE_DEVICES": "", "GGML_NO_BACKTRACE": "1"})
        self.assertEqual(cmd[cmd.index("--device") + 1], "none")

    def test_output_with_more_columns_or_on_stderr_is_still_read(self):
        r, _ = self.run_probe([], None, "Host 10 20 30 40\n")
        self.assertEqual((r["model"], r["context"], r["compute"]), (10 * MiB, 20 * MiB, 30 * MiB))
        r, _ = self.run_probe([], None, "", err="llama_fit: ...\nHost 11 21 31\n")
        self.assertEqual(r["model"], 11 * MiB)

    def test_failures_say_what_happened_in_plain_words(self):
        # a SystemExit with why, never a ValueError from a line lodger didn't expect
        for out, code, err, took, want in (
                ("Host 10 20\n", 0, "", 0, "gave no estimate for Host"),
                ("Hostile 1 2 3\n", 0, "", 0, "gave no estimate"),
                ("Host ten 20 30\n", 0, "", 0, "gave no estimate"),
                ("", -11, "", 1, "crashed (SIGSEGV)"),
                ("", -9, "", plan.PROBE_TIMEOUT, f"didn't answer within {plan.PROBE_TIMEOUT} s"),
                ("", 1, "error: unknown argument: --spec-type", 0, "failed (exit 1)")):
            with self.assertRaises(SystemExit) as e:
                self.run_probe([], None, out, code, err, took)
            self.assertIn(want, str(e.exception), out)
            self.assertIn("llama-cli", str(e.exception))      # names the model file (here: any file)
        with self.assertRaises(SystemExit) as e:
            self.run_probe([], None, "", 1, "error: unknown argument: --spec-type", 0)
        self.assertIn("unknown argument", str(e.exception))
        self.assertEqual(os.listdir(os.path.join(self.runs, "probe-cache")), [])          # nothing cached


    def test_a_failure_shows_llama_cpps_own_error_not_a_backtrace(self):
        err = ("0.00.000.670 E ggml_cuda_init: failed to initialize ROCm: no ROCm-capable device is detected\n"
               "0.00.001.751 I llama_fit_params: printing estimated memory\n"
               "0.00.033.678 E gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142. should be in [0, 43)\n"
               "0.00.037.853 E llama_model_load_from_file_impl: failed to load model\n"
               "terminate called after throwing an instance of 'std::runtime_error'\n  what():  failed to load model\n"
               + "".join(f"#{i} 0x00007fb0958 in frame{i} () from /x/libllama-common.so.0\n" for i in range(60)))
        with self.assertRaises(SystemExit) as e:
            self.run_probe([], None, "", -6, err, 0)
        msg = str(e.exception)
        self.assertIn("invalid ggml type 142", msg)
        self.assertNotIn("frame59", msg)
        self.assertNotIn("ROCm", msg)                                   # the card is hidden from it on purpose
        _, (cmd, env, props) = self.run_probe([], None, "Host 1 2 3\n")
        self.assertEqual(env.get("GGML_NO_BACKTRACE"), "1")            # no debugger attached to a crash

    def test_a_file_this_build_cant_read_is_not_probed_again(self):
        # llama-fit-params aborts on it: each abort is a crash report and a desktop notice
        build = fake_build()
        self.addCleanup(subprocess.run, ["rm", "-rf", build])
        cfg = config.load("/nonexistent")
        cfg.update(runs_dir=self.runs, llama_cpp=build)
        config.use(cfg)
        m = fake_model()
        m.files = [os.path.join(build, "build", "bin", "llama-cli")]
        calls = []
        results = {"bad": (-6, "0.00.03 E gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142\n"),
                   "slow": (-9, "")}

        def probe(kind, ctx):
            def run(cmd, env, props, timeout=300):
                calls.append(kind)
                code, err = results[kind]
                return subprocess.CompletedProcess(cmd, code, stdout="", stderr=err)
            with machine("desktop-amd"), mock.patch("lodger.fence.run", side_effect=run), \
                    mock.patch("lodger.plan.time.monotonic", side_effect=[0, plan.PROBE_TIMEOUT if kind == "slow" else 0]):
                with self.assertRaises(SystemExit) as e:
                    plan.probe(m, ctx)
            return str(e.exception)

        self.assertIn("invalid ggml type", probe("bad", 4096))
        again = probe("bad", 8192)                                  # any context, any card: the file is the problem
        self.assertEqual(calls, ["bad"])
        self.assertIn("invalid ggml type", again)
        self.assertIn("remembered", again)
        os.utime(m.files[0], (1, 1))                               # the file changed: asked again
        probe("bad", 4096)
        self.assertEqual(calls, ["bad", "bad"])
        for f in os.listdir(plan.cache_dir()):
            os.remove(os.path.join(plan.cache_dir(), f))
        probe("slow", 4096)
        probe("slow", 4096)                                         # a timeout isn't the file's fault
        self.assertEqual(calls[2:], ["slow", "slow"])

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


class ReserveTest(unittest.TestCase):
    def tearDown(self):
        config.use(None)

    def test_floor_by_card_size(self):
        from lodger import reserve
        config.use(config.load("/nonexistent"))
        self.assertEqual(reserve.floor(16 * GiB), 2 * GiB)
        self.assertEqual(reserve.floor(8 * GiB), GiB)
        self.assertEqual(reserve.floor(12 * GiB), 1536 * MiB)
        self.assertEqual(reserve.floor(4 * GiB), GiB)
        cfg = config.load("/nonexistent")
        cfg["desktop_reserve_mib"] = 3000
        config.use(cfg)
        self.assertEqual(reserve.floor(8 * GiB), 3000 * MiB)

    def test_rise_needs_an_hour(self):
        from lodger import reserve
        self.assertIsNone(reserve.rise({m: [0, 5 * GiB] for m in range(30)}))
        self.assertEqual(reserve.rise({m: [GiB, GiB] for m in range(90)}), 0)

    def test_kill_line_stays_under_the_reserve(self):
        from lodger import reserve
        self.assertEqual(reserve.kill_line(2 * GiB), 1536 * MiB)     # this PC: as before
        with mock.patch.dict(os.environ, {"LODGER_KILL_VRAM_FREE_MIB": "4096"}):
            self.assertEqual(reserve.kill_line(2 * GiB), 4 * GiB)    # tests may raise it

    def test_config_can_raise_the_floor_but_never_lower_it(self):
        from lodger import reserve
        for low in (-4096, 0, 512):
            cfg = config.load("/nonexistent")
            cfg["desktop_reserve_mib"] = low
            config.use(cfg)
            self.assertEqual(reserve.floor(16 * GiB), 2 * GiB, low)
        cfg["desktop_reserve_mib"] = "lots"
        with self.assertRaises(ValueError):
            reserve.floor(16 * GiB)

    def test_strange_notes_file_counts_as_no_notes(self):
        from lodger import reserve
        runs = tempfile.mkdtemp(prefix="lodger-runs-")
        self.addCleanup(subprocess.run, ["rm", "-rf", runs])
        cfg = config.load("/nonexistent")
        cfg["runs_dir"] = runs
        config.use(cfg)
        for i, text in enumerate(['[1, 2]', '{"5": 3}', '{"5": [1]}', '{"x": [1, 2]}', '{"5": ["a", "b"]}', 'not json']):
            pci = f"0000:0{i}:00.0"
            with open(reserve._path(pci), "w") as f:
                f.write(text)
            self.assertEqual(reserve._load(pci), {}, text)
            self.assertEqual(reserve.reserve(pci, 16 * GiB), (2 * GiB, None))
        reserve._notes.clear()


class LadderTest(unittest.TestCase):
    def test_the_usual_ladder_is_unchanged(self):
        self.assertEqual(plan.ladder(131072, 4096, 262144), [131072, 65536, 32768, 16384, 8192, 4096])
        self.assertEqual(plan.ladder(16384, 16384, 262144), [16384])

    def test_the_minimum_is_always_tried(self):
        self.assertEqual(plan.ladder(6000, 4096, None), [6000, 4096])
        self.assertEqual(plan.ladder(131072, 4096, 128000)[-1], 4096)

    def test_never_past_what_the_model_was_trained_for(self):
        self.assertEqual(plan.ladder(131072, 4096, 2048), [2048])
        self.assertEqual(plan.ladder(131072, 4096, 128000)[0], 128000)

    def test_zero_or_tiny_contexts_are_refused(self):
        for target, lo in ((0, 0), (8192, 0), (100, 100)):
            with self.assertRaises(ValueError):
                plan.ladder(target, lo, 262144)


class PlanSafetyTest(PlanBase):
    """Things that must make a plan bigger (or refuse), never smaller."""

    def test_no_layers_is_refused(self):
        p = self.make(fake_model(layers=0), vram_free=15 * GiB, card=None)
        self.assertIn("no layers", p["refuse"])

    def test_a_context_past_the_training_is_capped_and_says_so(self):
        p = self.make(fake_model(ctx_train=32768), card=None, target_ctx=65536)
        self.assertEqual(p["ctx"], 32768)
        self.assertTrue(any("trained for (asked for 65536)" in w for w in p["why"]), p["why"])
        p = self.make(fake_model(ctx_train=None), card=None, target_ctx=8192)
        self.assertTrue(any("doesn't say how long" in w for w in p["why"]), p["why"])

    def test_zero_min_ctx_is_refused_not_looped(self):
        with self.assertRaises(ValueError):
            self.make(fake_model(), card=None, min_ctx=0)

    def test_pieces_outside_the_layers_count_on_the_card(self):
        with machine("desktop-amd"):
            a = self.make(fake_model(), vram_free=15 * GiB, target_ctx=8192)
            b = self.make(fake_model(other=GiB), vram_free=15 * GiB, target_ctx=8192)
            c = self.make(fake_model(dense=60 * MiB, experts=400 * MiB, other=GiB), vram_free=12 * GiB)
        self.assertEqual(b["gpu_est"] - a["gpu_est"], GiB)
        self.assertGreater(c["n_cpu_moe"], 0)
        self.assertLessEqual(c["gpu_bytes"], 12 * GiB - c["keep"])

    def test_negative_leave_gpu_doesnt_add_room(self):
        with machine("desktop-amd"):
            a = self.make(fake_model(), vram_free=15 * GiB)
            b = self.make(fake_model(), vram_free=15 * GiB, leave_gpu=-8 * GiB)
        self.assertEqual(a["vram_budget"], b["vram_budget"])

    def test_context_that_doesnt_grow_is_never_planned_below_llamas_number(self):
        def probe(model, ctx, gpu=None, extra=(), dev=None):
            return {"model": 0, "context": (900 if ctx == 4096 else 500) * MiB, "compute": 0}
        with mock.patch("lodger.plan.probe", side_effect=probe):
            c = plan.Costs(fake_model(), 4096, 65536)
        self.assertGreaterEqual(c.context(65536), 900 * MiB)

    def test_one_length_still_tells_what_grows_from_what_is_fixed(self):
        # serve plans the card for exactly the context it chose: with one probe, all of the context looked
        # fixed, and a hybrid model's 32 checkpoints were planned as 32 copies of its whole cache
        asked = []

        def probe(model, ctx, gpu=None, extra=(), dev=None):
            asked.append(ctx)
            return fake_probe(model, ctx)
        with mock.patch("lodger.plan.probe", side_effect=probe):
            c = plan.Costs(fake_model(), 131072, 131072)
        self.assertEqual(c.per_tok, 64 * 1024)
        self.assertEqual(c.fixed(131072), 100 * MiB)
        self.assertEqual(c.context(131072), 100 * MiB + 131072 * 64 * 1024)
        self.assertEqual(sorted(set(asked)), [65536, 131072])
        with mock.patch("lodger.plan.probe", side_effect=probe):
            self.assertEqual(plan.Costs(fake_model(), 256, 256).per_tok, 64 * 1024)    # the shortest: one longer

    def projector(self, size):
        path = os.path.join(self.runs, "mmproj.gguf")
        with open(path, "wb") as f:
            f.truncate(size)
        return path

    def test_vision_projector_counts_on_the_card(self):
        seen = []

        def probe(model, ctx, gpu=None, extra=(), dev=None):
            seen.append(list(extra))
            return fake_probe(model, ctx)
        proj = self.projector(800 * MiB)
        with machine("desktop-amd"), mock.patch("lodger.plan.probe", side_effect=probe):
            a = self.make(fake_model(), vram_free=15 * GiB, target_ctx=8192)
            b = self.make(fake_model(), vram_free=15 * GiB, target_ctx=8192, extra=["--mmproj", proj, "--flash-attn", "on"])
            c = self.make(fake_model(), vram_free=15 * GiB, target_ctx=8192, extra=["--mmproj", proj, "--no-mmproj-offload"])
        self.assertEqual(b["gpu_est"] - a["gpu_est"], 800 * MiB + plan.MMPROJ_SCRATCH)
        self.assertEqual(c["gpu_est"], a["gpu_est"])
        self.assertEqual(c["ram_anon_est"] - a["ram_anon_est"], 800 * MiB + plan.MMPROJ_SCRATCH)
        self.assertTrue(all("--mmproj" not in e and "--no-mmproj-offload" not in e for e in seen), seen)
        self.assertIn(["--flash-attn", "on"], seen)                  # the rest still reaches llama.cpp

    def test_vision_projector_counts_in_ram_on_the_processor(self):
        proj = self.projector(800 * MiB)
        a = self.make(fake_model(), card=None, target_ctx=8192)
        b = self.make(fake_model(), card=None, target_ctx=8192, extra=["--mmproj", proj])
        self.assertEqual(b["ram_anon_est"] - a["ram_anon_est"], 800 * MiB + plan.MMPROJ_SCRATCH)

    def test_weights_held_in_ram_must_fit(self):
        m = fake_model()
        a = self.make(m, card=None, target_ctx=8192)
        for extra in (["--no-mmap"], ["--mlock"], ["--load-mode", "mlock"], ["--load-mode", "none"]):
            b = self.make(m, card=None, target_ctx=8192, extra=extra)
            self.assertEqual(b["ram_anon_est"] - a["ram_anon_est"], m.total, extra)
            self.assertFalse(b["slow"])
        tight = self.make(m, card=None, target_ctx=8192, ram_avail=a["ram_anon_est"] + 6 * GiB, extra=["--no-mmap"])
        self.assertIsNotNone(tight["refuse"])
        self.assertIsNone(self.make(m, card=None, target_ctx=8192, ram_avail=a["ram_anon_est"] + 6 * GiB)["refuse"])
        o = plan.own_costs(["--load-mode", "mmap", "--no-mmap"])
        self.assertEqual((o["fit"], o["proj"], o["proj_card"], o["held"]), (["--load-mode", "mmap"], 0, True, True))
        self.assertFalse(plan.own_costs(["--load-mode", "mmap"])["held"])


class PlacementTest(PlanBase):
    """Where llama.cpp really puts things, and what really takes memory."""

    def test_part_of_the_layers_on_the_card_brings_the_output_layer(self):
        # llama.cpp's -ngl N puts the output layer on the card first, then the last N - 1 layers: a plan that
        # counted N layers and left the output in RAM put a whole output layer (here 1 GiB) on the card unplanned
        m = fake_model(layers=40, dense=300 * MiB, output=GiB)
        with machine("desktop-amd"):
            p = self.make(m, vram_free=8 * GiB, target_ctx=8192)
        self.assertIsNone(p["refuse"])
        self.assertLess(p["ngl_layers"], 40)
        self.assertTrue(p["output_on_gpu"])
        self.assertEqual(p["ngl"], p["ngl_layers"] + 1)
        on_card = range(40 - p["ngl_layers"], 40)
        ctx = fake_probe(m, p["ctx"])["context"] * p["ngl_layers"] / 40
        self.assertEqual(p["gpu_est"], int(m.dense(on_card) + m.output + ctx + 300 * MiB + p["overhead"]))
        self.assertLessEqual(p["gpu_bytes"], p["vram_budget"])
        # and the output layer isn't counted in RAM too
        self.assertEqual(p["ram_weights"], m.token_embd + m.dense(range(40 - p["ngl_layers"])))

    def test_context_of_hybrid_layers_goes_where_it_is_kept(self):
        # 3 recurrent layers to 1 attention layer: the context's growing part is all in the attention layers,
        # the fixed part (recurrent state) in the recurrent ones
        def probe(model, ctx, gpu=None, extra=(), dev=None):
            return {"model": 0, "context": 400 * MiB + ctx * 64 * 1024, "compute": 300 * MiB}
        m = fake_model(layers=40, dense=300 * MiB, kinds=["recr", "recr", "recr", "attn"] * 10)
        with machine("desktop-amd"), mock.patch("lodger.plan.probe", side_effect=probe):
            c = plan.Costs(m, 4096, 16384)
            share = plan.layer_ctx(m, c, 16384)
            p = self.make(m, vram_free=8 * GiB, target_ctx=16384, min_ctx=16384)
        self.assertAlmostEqual(share[3], 16384 * 64 * 1024 / 10)
        self.assertAlmostEqual(share[0], 400 * MiB / 30)
        self.assertAlmostEqual(sum(share.values()), c.context(16384), delta=1)   # bytes; 3.11 sums floats less exactly
        self.assertEqual(p["ctx"], 16384)
        n = p["ngl_layers"]
        self.assertLess(n, 40)
        card = sum(share[i] for i in range(40 - n, 40))
        self.assertEqual(p["gpu_est"], int(m.dense(range(40 - n, 40)) + m.output + card + 300 * MiB + p["overhead"]))

    def test_sliding_windows_are_asked_at_their_bend(self):
        # sliding-window layers grow only up to window + ubatch: a straight line between 4096 and 131072 would
        # put 32768 words far below llama.cpp's own number
        def ctx_bytes(c):
            return c * 16 * 1024 + min(c, 8192 + 512) * 64 * 1024
        asked = []

        def probe(model, ctx, gpu=None, extra=(), dev=None):
            asked.append(ctx)
            return {"model": 0, "context": ctx_bytes(ctx), "compute": 300 * MiB}
        m = fake_model(kinds=["swa", "swa", "swa", "attn"] * 10, kv={"fake.attention.sliding_window": 8192})
        with mock.patch("lodger.plan.probe", side_effect=probe):
            bend = plan.window_bend(m, [])
            c = plan.Costs(m, 4096, 131072, [], [bend])
            self.assertEqual(plan.window_bend(m, ["--ubatch-size", "1024"]), 9216)
            self.assertIsNone(plan.window_bend(m, ["--swa-full"]))
        self.assertEqual(bend, 8704)
        self.assertEqual(sorted(asked), [4096, 8704, 131072])
        for ctx in (4096, 8192, 16384, 32768, 65536, 131072):
            self.assertGreaterEqual(c.context(ctx), ctx_bytes(ctx) - 1, ctx)
        self.assertEqual(c.per_tok, 16 * 1024)
        self.assertEqual(c.fixed(65536), 8704 * 64 * 1024)
        with machine("desktop-amd"), mock.patch("lodger.plan.probe", side_effect=probe):
            p = self.make(m, card=None, target_ctx=32768)
        self.assertIn(8704, asked[3:])

    def test_weights_repacked_for_the_processor_are_its_own_memory(self):
        # on x86 with AVX2, llama.cpp copies Q4_0 weights the processor runs into its own memory (a faster layout):
        # they can't be dropped like file pages (seen: LFM2.5 2.6B, 1.3 GiB of 1.7 GiB in RssAnon)
        quant = [("dense", i, 2, 4096, 100 * MiB) for i in range(40)] + [("output", -1, 2, 4096, 500 * MiB)]
        m = fake_model(quant=quant)
        with mock.patch("lodger.plan.cpu_repacks", return_value={}):
            a = self.make(m, card=None, target_ctx=8192)
        with mock.patch("lodger.plan.cpu_repacks", return_value={2: 8}):
            b = self.make(m, card=None, target_ctx=8192)
            c = self.make(m, card=None, target_ctx=8192, extra=["--no-repack"])
            with machine("desktop-amd"):
                d = self.make(m, vram_free=6 * GiB, target_ctx=8192)
                e = self.make(m, vram_free=6 * GiB, target_ctx=8192, extra=["--no-host"])
        self.assertEqual(b["ram_copied"], 40 * 100 * MiB + 500 * MiB)
        self.assertEqual(b["ram_anon_est"] - a["ram_anon_est"], b["ram_copied"])
        self.assertEqual(c["ram_anon_est"], a["ram_anon_est"])
        # with a card, the layers left in RAM go to its host buffer, which mmap swaps for the file's own pages
        # (llama-model.cpp make_cpu_buft_list): nothing is repacked, unless --no-host leaves the host buffer out
        self.assertLess(d["ngl_layers"], 40)
        self.assertEqual(d["ram_copied"], 0)
        self.assertEqual(e["ram_copied"], (40 - e["ngl_layers"]) * 100 * MiB)     # only what stays on the processor
        self.assertTrue(any("repack" in w for w in b["why"]), b["why"])
        with mock.patch("lodger.plan.cpu_repacks", return_value={2: 8}):
            tight = self.make(m, card=None, target_ctx=8192, ram_avail=a["ram_anon_est"] + 6 * GiB)
        self.assertIn("no-repack = true leaves out the 4.4 GiB", tight["refuse"] or "")

    def test_cpu_features_decide_what_is_repacked(self):
        cases = {"flags\t\t: fpu sse2 avx avx2 fma\n": {2: 8, 12: 8, 20: 8, 39: 8},
                 "flags\t\t: fpu avx2 avx512f\n": {2: 8, 12: 8, 20: 8, 39: 8, 10: 8},
                 "flags\t\t: fpu sse2 avx\n": {},
                 "Features\t: fp asimd asimddp\n": {2: 4, 8: 4, 41: 4, 20: 4, 39: 4, 12: 8, 13: 8, 14: 8},
                 "": {}}
        for text, want in cases.items():
            with mock.patch("lodger.host.read", return_value=text):
                self.assertEqual(plan.cpu_repacks(), want, text)

    def test_tied_output_on_the_processor_is_read_once(self):
        m = fake_model(tied=True, output=500 * MiB)
        with mock.patch("lodger.plan.cpu_repacks", return_value={}):
            p = self.make(m, card=None, target_ctx=8192)
        self.assertEqual(p["ram_copied"], 0)
        self.assertEqual(p["ram_weights"], m.token_embd + m.dense() + m.output)
        self.assertTrue(any("Weights kept in RAM: 4.4 GiB" in w for w in p["why"]), p["why"])   # 40 x 100 + 500 MiB

    def test_context_checkpoints_are_counted_in_ram(self):
        # llama-server keeps up to 32 copies of a hybrid model's recurrent state (or sliding windows) per slot
        def probe(model, ctx, gpu=None, extra=(), dev=None):
            return {"model": 0, "context": 150 * MiB + ctx * 64 * 1024, "compute": 300 * MiB}
        hybrid = fake_model(kinds=["recr", "recr", "recr", "attn"] * 10)
        with mock.patch("lodger.plan.probe", side_effect=probe):
            a = self.make(fake_model(), card=None, target_ctx=8192)
            b = self.make(hybrid, card=None, target_ctx=8192)
            c = self.make(hybrid, card=None, target_ctx=8192, extra=["--ctx-checkpoints", "4"])
        self.assertEqual(b["ram_anon_est"] - a["ram_anon_est"], 32 * 150 * MiB)
        self.assertEqual(c["ram_anon_est"] - a["ram_anon_est"], 4 * 150 * MiB)
        self.assertEqual(a["checkpoint_bytes"], 0)
        self.assertEqual(b["checkpoint_bytes"], 32 * 150 * MiB)
        with mock.patch("lodger.plan.probe", side_effect=probe):
            tight = self.make(hybrid, card=None, target_ctx=8192, ram_avail=b["ram_anon_est"] + 2 * GiB)
        self.assertIn("a smaller ctx-checkpoints (now 32)", tight["refuse"] or "")      # says what would help


def projector_file(d, **vision):
    """A vision projector's header: Qwen3-VL's by default."""
    kv = {"general.architecture": "clip", "clip.has_vision_encoder": True}
    hp = {"embedding_length": 1152, "feed_forward_length": 4304, "attention.head_count": 16, "image_size": 768,
          "patch_size": 16, "spatial_merge_size": 2, **vision}
    kv.update({f"clip.vision.{k}": v for k, v in hp.items() if v is not None})
    return write_gguf(os.path.join(d, "mmproj.gguf"), kv, [("v.blk.0.attn_q.weight", 0, [1152, 1152], 1152 * 1152 * 4)])


def mtp_model(**kw):
    """A model whose last layer only multi-token prediction uses (like Qwen3.5's)."""
    m = fake_model(kinds=["attn"] * 39 + [None], **kw)
    m.mtp = {39: {"dense": 200 * MiB, "experts": kw.get("experts", 0)}}
    m.layers[39] = {"dense": 0, "experts": 0}
    return m


class GapTest(PlanBase):
    """What llama-fit-params doesn't size: multi-token prediction drafting, llama.cpp's expert cache, a vision
    projector's working memory."""

    def test_mtp_drafting_loads_its_layers_and_a_second_context(self):
        seen = []

        def probe(model, ctx, gpu=None, extra=(), dev=None):
            seen.append(list(extra))
            return fake_probe(model, ctx)
        m = mtp_model()
        with mock.patch("lodger.plan.probe", side_effect=probe):
            a = self.make(m, card=None, target_ctx=8192)
            b = self.make(m, card=None, target_ctx=8192, extra=["--spec-type", "draft-mtp"])
        draft = int(8192 * 64 * 1024 / 39)          # one more attention layer's cache, as long as the main one
        self.assertEqual(b["ram_weights"] - a["ram_weights"], 200 * MiB)
        self.assertEqual(b["ram_anon_est"] - a["ram_anon_est"], draft + 300 * MiB)
        self.assertTrue(all("--spec-type" not in e for e in seen), seen)    # llama-fit-params doesn't take it
        self.assertTrue(any("draft-mtp" in w for w in b["why"]), b["why"])
        with machine("desktop-amd"):
            c = self.make(m, vram_free=15 * GiB, target_ctx=8192)
            d = self.make(m, vram_free=15 * GiB, target_ctx=8192, extra=["--spec-type", "ngram-mod,draft-mtp"])
        self.assertEqual(d["gpu_est"] - c["gpu_est"], 200 * MiB + draft + 300 * MiB)
        self.assertEqual(d["ngl"], 41)

    def test_mtp_drafting_without_mtp_layers_is_refused(self):
        p = self.make(fake_model(), card=None, target_ctx=8192, extra=["--spec-type", "draft-mtp"])
        self.assertIn("no MTP layers", p["refuse"] or "")
        self.assertIsNone(self.make(fake_model(), card=None, target_ctx=8192, extra=["--spec-type", "ngram-mod"])["refuse"])

    def test_mtp_draft_cache_sits_with_the_mtp_layer(self):
        m = mtp_model().with_mtp()
        costs = plan.Costs(m, 4096, 8192)
        costs.add_draft(m)
        share = plan.layer_ctx(m, costs, 8192)
        self.assertAlmostEqual(sum(share.values()), costs.context(8192), delta=2)
        self.assertAlmostEqual(share[39], 8192 * 64 * 1024 / 39, delta=2)
        self.assertEqual(costs.fixed(8192), 100 * MiB)                       # the draft cache grows with the context

    def test_expert_cache_is_counted_on_the_card_when_experts_stay_in_ram(self):
        m = fake_model(dense=60 * MiB, experts=400 * MiB)
        with machine("desktop-amd"):
            a = self.make(m, vram_free=12 * GiB, target_ctx=8192, extra=["--moe-cache-mib", "0"])
            b = self.make(m, vram_free=12 * GiB, target_ctx=8192, extra=["--moe-cache-mib", "1024"])
            whole = self.make(fake_model(dense=60 * MiB, experts=10 * MiB), vram_free=12 * GiB, target_ctx=8192)
            whole_c = self.make(fake_model(dense=60 * MiB, experts=10 * MiB), vram_free=12 * GiB, target_ctx=8192,
                                extra=["--moe-cache-mib", "1024"])
        self.assertIsNone(b["refuse"])
        self.assertGreater(b["n_cpu_moe"], a["n_cpu_moe"])                   # room for the cache, fewer experts
        self.assertLessEqual(b["gpu_bytes"], b["vram_budget"])
        self.assertTrue(any("Expert cache" in w for w in b["why"]), b["why"])
        self.assertEqual(whole_c["gpu_est"], whole["gpu_est"])              # every expert on the card: no cache made
        # with no cache asked for, the plan says how much room one could have
        self.assertGreaterEqual(a["moe_cache_room_mib"], 0)
        self.assertLess(a["moe_cache_room_mib"] * MiB, 400 * MiB * 1.2)
        self.assertNotIn("moe_cache_room_mib", b)
        self.assertNotIn("moe_cache_mib", a)                                 # 0 in models.toml: no cache of lodger's
        self.assertNotIn("moe_cache_mib", b)                                 # set in models.toml: passed on as it is

    def test_lodger_makes_an_expert_cache_when_experts_stay_in_ram(self):
        m = fake_model(dense=60 * MiB, experts=400 * MiB)
        with machine("desktop-amd"):
            off = self.make(m, vram_free=12 * GiB, target_ctx=8192, extra=["--moe-cache-mib", "0"])
            auto = self.make(m, vram_free=12 * GiB, target_ctx=8192)
            small = self.make(m, vram_free=8 * GiB, target_ctx=8192)
            small_off = self.make(m, vram_free=8 * GiB, target_ctx=8192, extra=["--moe-cache-mib", "0"])
            whole = self.make(fake_model(dense=60 * MiB, experts=10 * MiB), vram_free=12 * GiB, target_ctx=8192)
            cpu = self.make(m, vram_free=12 * GiB, target_ctx=8192, cpu_only=True)
        # in place of some experts on the card, never at the cost of context or whole layers
        self.assertEqual(auto["moe_cache_mib"], plan.AUTO_MOE_CACHE // MiB)
        self.assertEqual((auto["ctx"], auto["ngl"]), (off["ctx"], off["ngl"]))
        self.assertGreater(auto["n_cpu_moe"], off["n_cpu_moe"])
        self.assertLessEqual(auto["gpu_bytes"], auto["vram_budget"])
        self.assertTrue(any("chosen by lodger" in w for w in auto["why"]), auto["why"])
        # a smaller card: half the experts' room there (2 of 40 layers' 400 MiB)
        self.assertEqual(small_off["n_cpu_moe"], 38)
        self.assertEqual(small["moe_cache_mib"], 400)
        for p in (whole, cpu):                                              # nothing in RAM, or no card
            self.assertNotIn("moe_cache_mib", p)
            self.assertIsNone(p["refuse"])
        self.assertEqual(plan.auto_moe_cache(m, {"n_cpu_moe": 40}), 0)     # no experts on the card: no room
        self.assertEqual(plan.auto_moe_cache(m, {"n_cpu_moe": 39}), 200 * MiB)
        many = fake_model(dense=60 * MiB, experts=400 * MiB)
        many.meta["expert_used"] = 100                                     # half a layer is less than llama.cpp takes
        self.assertEqual(plan.auto_moe_cache(many, {"n_cpu_moe": 39}), 0)

    def test_the_cache_lodger_chose_goes_on_the_command_line(self):
        dev = mock.Mock()
        with mock.patch.object(engine, "for_device", return_value={"name": "ROCm0", "env": {}}):
            self.assertEqual(engine.device_args(41, dev, 30, 2048)[0],
                             ["--device", "ROCm0", "-ngl", "41", "-ncmoe", "30", "--moe-cache-mib", "2048"])
            self.assertEqual(engine.device_args(41, dev, 0, 2048)[0], ["--device", "ROCm0", "-ngl", "41"])
        self.assertEqual(engine.device_args(0, None, 0, 2048)[0], ["--device", "none", "-ngl", "0"])

    def test_expert_cache_that_cant_work_is_refused(self):
        moe = fake_model(dense=60 * MiB, experts=400 * MiB)
        p = self.make(moe, card=None, target_ctx=8192, extra=["--moe-cache-mib", "1024"])
        self.assertIn("MoE cache requires a GPU backend", p["refuse"] or "")
        p = self.make(fake_model(), card=None, target_ctx=8192, extra=["--moe-cache-mib", "1024"])
        self.assertIn("mixture-of-experts", p["refuse"] or "")
        # one word uses 8 of 128 experts: llama.cpp needs 9 slots of 400/128 MiB
        self.assertEqual(plan.min_moe_cache(moe), -(-400 * MiB // 128) * 9)
        with machine("desktop-amd"):
            p = self.make(moe, vram_free=12 * GiB, target_ctx=8192, extra=["--moe-cache-mib", "16"])
        self.assertIn("too small", p["refuse"] or "")
        self.assertEqual(plan.own_costs(["--moe-cache-mib", "16"])["fit"], [])

    def test_vision_projector_working_memory_from_its_header(self):
        d = tempfile.mkdtemp(prefix="lodger-proj-")
        self.addCleanup(subprocess.run, ["rm", "-rf", d])
        per_patch = (3 * 1152 + 4304) * 4
        # measured on the processor with Qwen3-VL's projector: 120.06 MiB at 4096 patches, 248.10 MiB at 8464
        self.assertAlmostEqual(4096 * per_patch / MiB, 120.06, delta=120.06 * 0.02)
        self.assertAlmostEqual(8464 * per_patch / MiB, 248.10, delta=248.10 * 0.02)
        p = projector_file(d)
        self.assertEqual(plan.projector_work(p), int(1.5 * 4096 * 4 * per_patch))     # pictures up to 4096 words
        self.assertEqual(plan.projector_work(p, tokens=1024), max(256 * MiB, int(1.5 * 1024 * 4 * per_patch)))
        self.assertEqual(plan.projector_work(p, tokens=1024, flash_attn=False),
                         int(1.5 * (4096 * per_patch + 16 * 4096 * 4096 * 4)))
        p = projector_file(d, image_max_pixels=2048 * 2048)
        self.assertEqual(plan.projector_work(p), int(1.5 * 16384 * per_patch))
        p = projector_file(d, spatial_merge_size=None, image_size=896, patch_size=14)      # every picture at 896 x 896
        self.assertEqual(plan.projector_work(p), 256 * MiB)
        p = projector_file(d, patch_size=None)
        self.assertEqual(plan.projector_work(p), plan.MMPROJ_SCRATCH)        # the file doesn't say enough
        o = plan.own_costs(["--mmproj", projector_file(d), "--image-max-tokens", "2048", "--flash-attn", "off"])
        self.assertEqual(o["fit"], ["--flash-attn", "off"])
        self.assertEqual(o["proj_work"], int(1.5 * (8192 * per_patch + 16 * 8192 * 8192 * 4)))
        a = self.make(fake_model(), card=None, target_ctx=8192)
        b = self.make(fake_model(), card=None, target_ctx=8192, extra=["--mmproj", projector_file(d)])
        self.assertEqual(b["ram_anon_est"] - a["ram_anon_est"], os.path.getsize(projector_file(d)) + plan.projector_work(projector_file(d)))


class LlamaCppDetailTest(PlanBase):
    """Memory llama.cpp takes in ways its own estimate doesn't show (each checked in its source)."""

    def test_any_sliding_window_makes_checkpoints(self):
        # llama-server checkpoints any model with n_swa > 0, whichever layers slide (server-context.cpp)
        m = fake_model(kinds=["attn"] * 40, kv={"fake.attention.sliding_window": 128})
        with mock.patch("lodger.plan.cpu_repacks", return_value={}):
            p = self.make(m, card=None, target_ctx=8192)
        self.assertGreater(p["checkpoint_bytes"], 0)

    def test_a_window_longer_than_the_context_is_copied_whole(self):
        # the probes are all shorter than the window: growth they see is partly the windows', which a checkpoint
        # copies too
        m = fake_model(kinds=["swa", "swa", "swa", "attn"] * 10, kv={"fake.attention.sliding_window": 1 << 20})
        with mock.patch("lodger.plan.cpu_repacks", return_value={}):
            p = self.make(m, card=None, target_ctx=8192, extra=["--ctx-checkpoints", "1"])
        self.assertEqual(p["checkpoint_bytes"], 100 * MiB + int(8192 * 64 * 1024 * 0.75))

    def test_a_card_plan_doesnt_repack(self):
        # an ordinary Q4_K MoE: experts in RAM are the file's pages with a card, so a 16 GB PC can run it
        quant = [("experts", i, 12, 768, 300 * MiB) for i in range(40)]
        m = fake_model(dense=60 * MiB, experts=300 * MiB, quant=quant)
        mem = {"MemTotal": 16 * GiB, "MemAvailable": 10 * GiB}
        with machine("desktop-amd"), mock.patch("lodger.plan.cpu_repacks", return_value={12: 8}):
            b = self.make(m, vram_free=10 * GiB, target_ctx=8192, mem=mem)
        self.assertIsNone(b["refuse"], b["refuse"])
        self.assertEqual(b["ram_copied"], 0)

    def test_mtp_on_a_hybrid_model_keeps_copies_of_its_state(self):
        # common.cpp: n_rs_seq = the draft length (3) for draft-mtp; the recurrent cache gets 1 + n_rs_seq rows
        def probe(model, ctx, gpu=None, extra=(), dev=None):
            return {"model": 0, "context": 150 * MiB + ctx * 64 * 1024, "compute": 300 * MiB}
        m = fake_model(kinds=["recr", "recr", "recr", "attn"] * 9 + ["recr", "recr", "recr", None])
        m.mtp = {39: {"dense": 200 * MiB, "experts": 0}}
        m.layers[39] = {"dense": 0, "experts": 0}
        base = ["--ctx-checkpoints", "0"]
        with mock.patch("lodger.plan.probe", side_effect=probe), mock.patch("lodger.plan.cpu_repacks", return_value={}):
            a = self.make(m, card=None, target_ctx=8192, extra=base)
            b = self.make(m, card=None, target_ctx=8192, extra=base + ["--spec-type", "draft-mtp"])
            c = self.make(m, card=None, target_ctx=8192, extra=base + ["--spec-type", "draft-mtp", "--spec-draft-n-max", "6"])
        draft = int(8192 * 64 * 1024 / 9)
        self.assertEqual(b["ram_anon_est"] - a["ram_anon_est"], draft + 3 * 150 * MiB + 300 * MiB)
        self.assertEqual(c["ram_anon_est"] - b["ram_anon_est"], 3 * 150 * MiB)

    def test_mtp_on_a_sliding_window_model_keeps_one_more_checkpoint(self):
        m = mtp_model(kv={"fake.attention.sliding_window": 128})
        with mock.patch("lodger.plan.cpu_repacks", return_value={}):
            a = self.make(m, card=None, target_ctx=8192, extra=["--ctx-checkpoints", "2"])
            b = self.make(m, card=None, target_ctx=8192, extra=["--ctx-checkpoints", "2", "--spec-type", "draft-mtp"])
        self.assertGreater(a["checkpoint_bytes"], 0)
        self.assertEqual(b["checkpoint_bytes"] * 2, a["checkpoint_bytes"] * 3)

    def test_llamas_card_estimate_doesnt_drop_what_it_isnt_told(self):
        # llama-fit-params isn't told --spec-type: when its card estimate is higher than lodger's, the MTP layer,
        # its cache and scratch still count on top
        def probe(model, ctx, gpu=None, extra=(), dev=None):
            if gpu is None:
                return fake_probe(model, ctx)
            return {"model": 4500 * MiB, "context": 100 * MiB + ctx * 64 * 1024, "compute": 900 * MiB}
        with machine("desktop-amd"), mock.patch("lodger.plan.probe", side_effect=probe), \
                mock.patch("lodger.plan.cpu_repacks", return_value={}):
            a = self.make(mtp_model(), vram_free=15 * GiB, target_ctx=8192, device_check=True)
            b = self.make(mtp_model(), vram_free=15 * GiB, target_ctx=8192, extra=["--spec-type", "draft-mtp"],
                          device_check=True)
        self.assertEqual((a["ngl"], b["ngl"]), (41, 41))
        self.assertEqual(a["llama_gpu_est"], b["llama_gpu_est"])
        self.assertEqual(b["gpu_est"] - a["gpu_est"], 200 * MiB + int(8192 * 64 * 1024 / 39) + 300 * MiB)

    def test_projectors_with_their_own_picture_limit(self):
        d = tempfile.mkdtemp(prefix="lodger-proj-")
        self.addCleanup(subprocess.run, ["rm", "-rf", d])
        per_patch = (3 * 1152 + 4304) * 4
        kv = {"general.architecture": "clip", "clip.has_vision_encoder": True, "clip.projector_type": "youtuvl"}
        kv.update({f"clip.vision.{k}": v for k, v in {"embedding_length": 1152, "feed_forward_length": 4304,
                   "attention.head_count": 16, "image_size": 768, "patch_size": 16, "spatial_merge_size": 2}.items()})
        p = write_gguf(os.path.join(d, "youtu.gguf"), kv, [("v.blk.0.attn_q.weight", 0, [4, 4], 64)])
        self.assertEqual(plan.projector_work(p), int(1.5 * 62500 * 4 * per_patch))     # 8000 x 8000 pictures
        kv["clip.projector_type"] = "deepseek4v"
        del kv["clip.vision.spatial_merge_size"]
        kv["clip.vision.projector.scale_factor"] = 2
        p = write_gguf(os.path.join(d, "ds4.gguf"), kv, [("v.blk.0.attn_q.weight", 0, [4, 4], 64)])
        self.assertEqual(plan.projector_work(p), 256 * MiB)                            # 384 words: small
        self.assertEqual(plan.projector_work(p, tokens=4096), int(1.5 * 4096 * 4 * per_patch))


class ProbeCacheTest(unittest.TestCase):
    def setUp(self):
        self.runs = tempfile.mkdtemp(prefix="lodger-runs-")
        self.build = fake_build("libggml-hip.so", "libllama.so")
        self.addCleanup(subprocess.run, ["rm", "-rf", self.runs, self.build])
        cfg = config.load("/nonexistent")
        cfg.update(runs_dir=self.runs, llama_cpp=self.build)
        config.use(cfg)
        self.addCleanup(config.use, None)
        self.m = fake_model()
        self.m.files = []
        for i in (1, 2):
            p = os.path.join(self.runs, f"part{i}.gguf")
            open(p, "w").close()
            self.m.files.append(p)
        self.calls = []

    def run_probe(self):
        def run(cmd, env, props, timeout=300):
            self.calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="Host 10 20 30 \n", stderr="")
        with machine("desktop-amd"), mock.patch("lodger.fence.run", side_effect=run):
            return plan.probe(self.m, 4096)

    def cached(self):
        return [os.path.join(plan.cache_dir(), n) for n in os.listdir(plan.cache_dir())]

    def test_damaged_cache_file_is_a_miss(self):
        self.assertEqual(self.run_probe()["context"], 20 * MiB)
        (cf,) = self.cached()
        for bad in ('{"model": 1', '[1, 2]', '{"model": 1, "context": 2}', '{"model": "1", "context": 2, "compute": 3}'):
            with open(cf, "w") as f:
                f.write(bad)
            self.assertEqual(self.run_probe(), {"model": 10 * MiB, "context": 20 * MiB, "compute": 30 * MiB}, bad)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(self.cached(), [cf])                       # rewritten in place, no temp files left
        self.run_probe()
        self.assertEqual(len(self.calls), 5)                        # a good file is used

    def test_any_part_or_library_changing_asks_again(self):
        self.run_probe()
        self.run_probe()
        self.assertEqual(len(self.calls), 1)
        os.utime(self.m.files[1], (1, 1))
        self.run_probe()
        self.assertEqual(len(self.calls), 2)
        os.utime(os.path.join(self.build, "build", "bin", "libllama.so"), (1, 1))
        self.run_probe()
        self.assertEqual(len(self.calls), 3)


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.runs = tempfile.mkdtemp(prefix="lodger-runs-")
        self.addCleanup(subprocess.run, ["rm", "-rf", self.runs])
        cfg = config.load("/nonexistent")
        cfg["runs_dir"] = self.runs
        config.use(cfg)
        self.addCleanup(config.use, None)

    def write(self, *lines):
        with open(os.path.join(self.runs, "calibration.jsonl"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def test_bad_lines_are_skipped_one_by_one(self):
        good = '{"llama_gpu_est": %d, "gpu_real": %d}' % (10 * GiB, 10 * GiB + 900 * MiB)
        self.write("{not json", "[1, 2]", '"text"', '{"llama_gpu_est": "big", "gpu_real": 5}',
                   '{"llama_gpu_est": true, "gpu_real": 5}', '{"llama_gpu_est": null, "gpu_real": 5}', good)
        over, src = plan.learned_overhead()
        self.assertEqual(over, 900 * MiB + 128 * MiB)              # the good run still counts
        self.assertIn("1 recent", src)

    def test_runs_on_another_card_or_backend_dont_count(self):
        self.write('{"llama_gpu_est": %d, "gpu_real": %d, "pci": "0000:03:00.0", "backend": "rocm"}' % (GiB, GiB + 300 * MiB),
                   '{"llama_gpu_est": %d, "gpu_real": %d, "pci": "0000:09:00.0", "backend": "rocm"}' % (GiB, GiB + 1200 * MiB),
                   '{"llama_gpu_est": %d, "gpu_real": %d, "pci": "0000:03:00.0", "backend": "vulkan"}' % (GiB, GiB + 1100 * MiB),
                   '{"llama_gpu_est": %d, "gpu_real": %d}' % (GiB, GiB + 200 * MiB))
        self.assertEqual(plan.learned_overhead("0000:03:00.0", "rocm")[0], 300 * MiB + 128 * MiB)
        self.assertEqual(plan.learned_overhead()[0], 1200 * MiB + 128 * MiB)


if __name__ == "__main__":
    unittest.main()
