import os
import tempfile
import unittest

from lodger import catalog


def touch(*parts):
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GGUF")
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


if __name__ == "__main__":
    unittest.main()
