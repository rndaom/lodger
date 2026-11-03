import os
import struct
import tempfile
import unittest

from lodger import catalog, model, plan


def touch(*parts):
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GGUF")
    return path


def gguf(path, arch, version=3):
    """A GGUF header with two settings: general.alignment, then general.architecture."""
    def text(s):
        return struct.pack("<Q", len(s)) + s.encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", version, 0, 2))
        f.write(text("general.alignment") + struct.pack("<II", 4, 32))
        f.write(text("general.architecture") + struct.pack("<I", 8) + text(arch))
    return path


class SlugTest(unittest.TestCase):
    def test_file_name_becomes_a_plain_name(self):
        self.assertEqual(catalog.slug("/m/Qwen3.5-9B-UD-Q4_K_XL.gguf"), "qwen3.5-9b-ud-q4-k-xl")

    def test_split_models_share_one_name(self):
        self.assertEqual(catalog.slug("/m/Big-Model-00001-of-00006.gguf"), "big-model")


class EntryTest(unittest.TestCase):
    def test_settings_become_llama_server_options(self):
        e = catalog.Entry("m", "/m.gguf", {"temp": 0.7, "flash-attn": True, "no-warmup": False, "ctx": 8192, "dry": ["a", "b"]})
        self.assertEqual(e.ctx, 8192)
        self.assertEqual(e.args(), ["--temp", "0.7", "--flash-attn", "--dry", "a,b"])

    def test_planning_args_are_only_the_ones_that_change_memory(self):
        e = catalog.Entry("m", "/m.gguf", {"temp": 0.7, "cache-type-k": "q8_0", "flash-attn": "on"})
        self.assertEqual(e.planning_args(), ["--cache-type-k", "q8_0", "--flash-attn", "on"])

    def test_reasoning_is_on_unless_turned_off(self):
        self.assertTrue(catalog.Entry("m", "/m.gguf", {}).reasoning)
        self.assertFalse(catalog.Entry("m", "/m.gguf", {"reasoning": "off"}).reasoning)
        self.assertFalse(catalog.Entry("m", "/m.gguf", {"rea": "off"}).reasoning)

    def test_this_pcs_settings_pass_through_unchanged(self):
        s = {"flash-attn": "on", "cache-type-k": "q8_0", "cache-type-v": "q8_0", "no-warmup": True, "reasoning": "off",
             "temp": 0.7, "top-p": 0.8, "top-k": 20, "min-p": 0, "presence-penalty": 1.5}
        e = catalog.Entry("m", "/m.gguf", s)
        self.assertIsNone(e.skip)
        self.assertEqual(e.planning_args(), ["--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
        self.assertEqual(e.args()[:7], ["--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--no-warmup"])

    def test_short_names_reach_the_planner(self):
        e = catalog.Entry("m", "/m.gguf", {"ctk": "q8_0", "-fa": "on", "--ub": 512, "b": 1024, "temp": 0.7, "swa-full": True})
        self.assertIsNone(e.skip)
        self.assertEqual(e.planning_args(), ["--cache-type-k", "q8_0", "--flash-attn", "on", "--ubatch-size", "512",
                                             "--batch-size", "1024", "--swa-full"])
        self.assertIn("--temp", e.args())

    def test_a_models_own_setting_wins_over_the_defaults_whatever_its_name(self):
        e = catalog.Entry("m", "/m.gguf", {**{"cache-type-k": "f16"}, **{"ctk": "q4_0"}})
        self.assertEqual(e.planning_args(), ["--cache-type-k", "q4_0"])

    def test_settings_lodger_sets_itself_are_refused(self):
        for k, v, word in (("ngl", 99, "n-gpu-layers"), ("gpu-layers", 10, "n-gpu-layers"), ("ncmoe", 3, "n-cpu-moe"),
                           ("np", 4, "parallel"), ("port", 1, "port"), ("host", "0.0.0.0", "host"), ("t", 32, "threads"),
                           ("api-key", "x", "api-key"), ("m", "/x.gguf", "model"), ("fit", "on", "fit"),
                           ("dev", "ROCm0", "device"), ("ot", "exps=CPU", "override-tensor"), ("webui", True, "webui"),
                           ("models-preset", "/x.ini", "models-preset")):
            e = catalog.Entry("m", "/m.gguf", {k: v})
            self.assertIn(word, e.skip or "", k)
            self.assertIn("lodger sets itself", e.skip, k)
        self.assertIn("ctx = N", catalog.Entry("m", "/m.gguf", {"c": 8192}).skip)
        self.assertIsNone(catalog.Entry("m", "/m.gguf", {"webui": False}).skip)      # false: not passed on at all

    def test_settings_lodger_cant_plan_are_refused(self):
        for k, v in (("md", "/draft.gguf"), ("model-draft", "/d.gguf"), ("ctkd", "q8_0"), ("spec-draft-ngl", 99),
                     ("spec-type", "mtp"), ("lora", "/a.gguf"), ("override-kv", "x=int:1"), ("nkvo", True)):
            e = catalog.Entry("m", "/m.gguf", {k: v})
            self.assertIn("can't plan memory for", e.skip or "", k)

    def test_speculative_decoding_without_a_draft_model_is_planned(self):
        for v in ("draft-mtp", "ngram-mod", "ngram-simple,draft-mtp", ["draft-mtp", "ngram-cache"], "none"):
            e = catalog.Entry("m", "/m.gguf", {"spec-type": v, "spec-draft-n-max": 3, "draft-p-min": 0.8,
                                               "spec-ngram-mod-n-max": 24})
            self.assertIsNone(e.skip, v)
            self.assertEqual(e.planning_args()[0], "--spec-type")       # the planner counts it
        e = catalog.Entry("m", "/m.gguf", {"spec-type": "draft-mtp", "spec-draft-n-max": 6})
        self.assertEqual(e.planning_args(), ["--spec-type", "draft-mtp", "--spec-draft-n-max", "6"])   # counted
        self.assertIsNotNone(catalog.Entry("m", "/m.gguf", {"spec-draft-n-max": "lots"}).skip)
        for v in ("draft-eagle3", "draft-simple", "mtp", "ngram-mod,draft-dflash"):
            self.assertIn("draft model", catalog.Entry("m", "/m.gguf", {"spec-type": v}).skip or "", v)
        self.assertIn("needs a value", catalog.Entry("m", "/m.gguf", {"spec-type": True}).skip or "")

    def test_counted_settings_need_whole_numbers(self):
        e = catalog.Entry("m", "/m.gguf", {"ctxcp": 4, "moe-cache-mib": 2048, "image-max-tokens": 1024})
        self.assertIsNone(e.skip)
        self.assertEqual(e.planning_args(), ["--ctx-checkpoints", "4", "--moe-cache-mib", "2048",
                                             "--image-max-tokens", "1024"])
        self.assertEqual(catalog.Entry("m", "/m.gguf", {"swa-checkpoints": 2}).planning_args(), ["--ctx-checkpoints", "2"])
        for k, v in (("ctx-checkpoints", "lots"), ("moe-cache-mib", -1), ("image-max-tokens", 1.5), ("moe-cache-mib", True)):
            self.assertIsNotNone(catalog.Entry("m", "/m.gguf", {k: v}).skip, (k, v))

    def test_dense_layers_on_the_processor_are_lodgers_to_set(self):
        for k in ("ncffn", "n-cpu-ffn"):
            self.assertIn("lodger sets itself", catalog.Entry("m", "/m.gguf", {k: 4}).skip or "", k)

    def test_a_table_is_not_a_setting(self):
        self.assertIn("is a table", catalog.Entry("m", "/m.gguf", {"temp": {"a": 1}}).skip or "")

    def test_bad_ctx_is_refused(self):
        for ctx in (0, -1, 100, "big", True, 8192.5):
            self.assertIn("ctx must be", catalog.Entry("m", "/m.gguf", {"ctx": ctx}).skip or "", ctx)
        self.assertIsNone(catalog.Entry("m", "/m.gguf", {"ctx": 8192}).skip)
        self.assertEqual(catalog.SMALLEST_CTX, plan.SMALLEST_CTX)

    def test_vision_projector_goes_to_the_planner_with_a_full_path(self):
        with tempfile.TemporaryDirectory() as d:
            proj = touch(d, "mmproj.gguf")
            e = catalog.Entry("m", "/m.gguf", {"mm": proj, "no-mmproj-offload": True})
            self.assertIsNone(e.skip)
            self.assertEqual(e.planning_args(), ["--mmproj", proj, "--no-mmproj-offload"])
            gone = catalog.Entry("m", "/m.gguf", {"mmproj": os.path.join(d, "missing.gguf")})
            self.assertIn("isn't there", gone.skip)
        home = catalog.Entry("m", "/m.gguf", {"mmproj": "~/x/mmproj.gguf"})
        self.assertEqual(home.settings["mmproj"], os.path.expanduser("~/x/mmproj.gguf"))

    def test_weights_in_ram_go_to_the_planner(self):
        e = catalog.Entry("m", "/m.gguf", {"no-mmap": True, "lm": "mlock"})
        self.assertEqual(e.planning_args(), ["--no-mmap", "--load-mode", "mlock"])


class LoadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def write_config(self, text):
        path = os.path.join(self.dir, "models.toml")
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_no_config_means_no_models(self):
        models, default = catalog.load(config=os.path.join(self.dir, "missing.toml"))
        self.assertEqual((models, default), ({}, None))

    def test_folders_are_scanned_and_named_models_get_their_settings(self):
        named = touch(self.dir, "models", "a", "Alpha.gguf")
        touch(self.dir, "models", "b", "Beta-7B.gguf")
        touch(self.dir, "models", "notes.txt")
        config = self.write_config(f"""
default = "alpha"
folders = ["{self.dir}/models"]
[defaults]
flash-attn = "on"
[models.alpha]
file = "{named}"
temp = 0.5
[models.gone]
file = "{self.dir}/does-not-exist.gguf"
""")
        models, default = catalog.load(config=config)
        self.assertEqual(default, "alpha")
        self.assertEqual(list(models), ["alpha", "beta-7b"])
        self.assertEqual(models["alpha"].path, named)
        self.assertEqual(models["alpha"].settings, {"flash-attn": "on", "temp": 0.5})
        self.assertEqual(models["beta-7b"].settings, {"flash-attn": "on"})

    def test_ignored_files_and_extra_split_parts_are_left_out(self):
        touch(self.dir, "m", "Big-00001-of-00002.gguf")
        touch(self.dir, "m", "Big-00002-of-00002.gguf")
        touch(self.dir, "m", "mmproj-vision.gguf")
        touch(self.dir, "m", "Small.gguf")
        config = self.write_config(f'folders = ["{self.dir}/m"]\nignore = ["*mmproj*"]\n')
        models, _ = catalog.load(config=config)
        self.assertEqual(sorted(models), ["big", "small"])

    def test_skip_reason_is_kept(self):
        path = touch(self.dir, "huge.gguf")
        config = self.write_config(f'[models.huge]\nfile = "{path}"\nskip = "needs special setup"\n')
        models, _ = catalog.load(config=config)
        self.assertEqual(models["huge"].skip, "needs special setup")
        self.assertNotIn("skip", models["huge"].settings)

    def test_serving_chosen_files_ignores_the_folders(self):
        chosen = touch(self.dir, "x", "Chosen.gguf")
        touch(self.dir, "x", "Other.gguf")
        config = self.write_config(f'folders = ["{self.dir}/x"]\n')
        models, _ = catalog.load([chosen], config=config)
        self.assertEqual(list(models), ["chosen"])

    def test_one_string_counts_as_a_list(self):
        touch(self.dir, "m", "Small.gguf")
        touch(self.dir, "m", "mmproj-vision.gguf")
        config = self.write_config(f'folders = "{self.dir}/m"\nignore = "*mmproj*"\n')
        models, _ = catalog.load(config=config)
        self.assertEqual(list(models), ["small"])
        with self.assertRaises(ValueError):
            catalog.load(config=self.write_config("folders = 5\n"))
        with self.assertRaises(ValueError):
            catalog.load(config=self.write_config("[models.x]\ntemp = 1\n"))

    def test_one_file_in_two_sections_keeps_the_first_and_says_so(self):
        f = touch(self.dir, "m", "M.gguf")
        config = self.write_config(f'[models.fast]\nfile = "{f}"\ntemp = 0.2\n[models.slow]\nfile = "{f}"\ntemp = 1.0\n')
        models, _ = catalog.load(config=config)
        self.assertEqual(list(models), ["fast", "slow"])
        self.assertEqual((models["fast"].settings, models["fast"].skip), ({"temp": 0.2}, None))
        self.assertIn("already served as fast", models["slow"].skip or "")
        self.assertEqual(list(catalog.load([f], config=config)[0]), ["fast", "slow"])

    def test_a_name_used_twice_keeps_the_first(self):
        a = touch(self.dir, "a", "Same.gguf")
        b = touch(self.dir, "b", "Same.gguf")
        named = touch(self.dir, "c", "Mine.gguf")
        c = touch(self.dir, "c", "Small.gguf")
        config = self.write_config(f'folders = ["{self.dir}/a", "{self.dir}/b", "{self.dir}/c"]\n'
                                   f'[models.small]\nfile = "{named}"\n')
        for _ in range(2):                                         # the same answer every time
            models, _ = catalog.load(config=config)
            self.assertEqual(list(models), ["small", "same", "same-2", "small-2"])
            self.assertEqual((models["small"].path, models["same"].path), (named, a))
            self.assertIsNone(models["same"].skip)
            self.assertEqual(models["same-2"].path, b)
            self.assertIn("already used", models["same-2"].skip)
            self.assertEqual(models["small-2"].path, c)
            self.assertIn("[models.NAME]", models["small-2"].skip)

    def test_vision_projectors_arent_models(self):
        gguf(os.path.join(self.dir, "m", "Model.gguf"), "llama")
        proj = gguf(os.path.join(self.dir, "m", "vision-f16.gguf"), "clip")
        models, _ = catalog.load(config=self.write_config(f'folders = ["{self.dir}/m"]\n'))
        self.assertEqual(list(models), ["model"])
        models, _ = catalog.load(config=self.write_config(f'[models.v]\nfile = "{proj}"\n'))
        self.assertIn("vision projector", models["v"].skip)

    def test_architecture_from_the_header(self):
        self.assertEqual(model.architecture(gguf(os.path.join(self.dir, "a.gguf"), "clip")), "clip")
        self.assertEqual(model.architecture(gguf(os.path.join(self.dir, "b.gguf"), "qwen35", version=2)), "qwen35")
        self.assertIsNone(model.architecture(touch(self.dir, "c.gguf")))
        self.assertIsNone(model.architecture(os.path.join(self.dir, "missing.gguf")))
        with open(os.path.join(self.dir, "d.gguf"), "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1 << 62))
        self.assertIsNone(model.architecture(os.path.join(self.dir, "d.gguf")))


class SplitPartsTest(unittest.TestCase):
    def test_only_five_digit_parts_count(self):
        with tempfile.TemporaryDirectory() as d:
            parts = [touch(d, f"Big-0000{i}-of-00003.gguf") for i in (1, 2, 3)]
            touch(d, "Big-extra-of-00003.gguf")
            touch(d, "Big-000010-of-00003.gguf")
            self.assertEqual(model.parts(parts[0]), parts)
            os.remove(parts[1])
            with self.assertRaises(SystemExit):
                model.parts(parts[0])


if __name__ == "__main__":
    unittest.main()
