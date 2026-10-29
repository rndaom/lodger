"""How the engine is pointed at one card, on recorded machines (no llama.cpp, no card used)."""
import os
import unittest
from unittest import mock

from lodger import engine, gpu
from tests.support import machine, write

LAYER = "/usr/share/vulkan/implicit_layer.d/VkLayer_MESA_device_select.json"


class EnvTest(unittest.TestCase):
    def test_this_pc_unchanged(self):
        with machine("desktop-amd"):
            card, igpu = gpu.devices()
            self.assertEqual(engine.for_device(card, have={"rocm"}),
                             {"backend": "rocm", "name": "ROCm0", "env": {"HIP_VISIBLE_DEVICES": "0"}})
            v = engine.for_device(card, have={"vulkan"})["env"]
            self.assertEqual(v, {"GGML_VK_VISIBLE_DEVICES": "0", "DRI_PRIME": "pci-0000_03_00_0",
                                 "MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE": "1", "MESA_VK_DEVICE_SELECT": "1002:747e!"})

    def test_child_env(self):
        environ = {"PATH": "/usr/bin", "ROCR_VISIBLE_DEVICES": "1", "GPU_DEVICE_ORDINAL": "1", "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
                   "RADV_PERFTEST": "sam", "LD_LIBRARY_PATH": "/opt/rocm/lib", "LLAMA_ARG_CTX_SIZE": "99", "LLAMA_API_KEY": "k"}
        put, drop = engine.child_env({"HIP_VISIBLE_DEVICES": "0"}, environ)
        self.assertEqual(put, {"HSA_OVERRIDE_GFX_VERSION": "11.0.0", "RADV_PERFTEST": "sam",
                               "LD_LIBRARY_PATH": "/opt/rocm/lib", "HIP_VISIBLE_DEVICES": "0"})
        self.assertNotIn("PATH", put)                         # not a runtime setting: each fence keeps its own
        for v in ("ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES", "LLAMA_ARG_CTX_SIZE", "LLAMA_API_KEY"):
            self.assertIn(v, drop)
        self.assertNotIn("HIP_VISIBLE_DEVICES", drop)


class UnknownDriverTest(unittest.TestCase):
    def test_nouveau_is_not_used(self):
        with machine("laptop-nvidia", writable=True) as root:
            os.remove(os.path.join(root, "sys/bus/pci/devices/0000:01:00.0/driver"))
            os.symlink("../../../../bus/pci/drivers/nouveau", os.path.join(root, "sys/bus/pci/devices/0000:01:00.0/driver"))
            card = gpu.choose("01:00.0")                       # an explicit gpu = <pci> still finds it
            t = engine.for_device(card, have={"cuda", "vulkan"})
            self.assertIn("nouveau", t["why"])
            self.assertNotIn("env", t)

    def test_cuda_without_an_index(self):
        with machine("laptop-nvidia"):
            card = gpu.choose("01:00.0")
            self.assertIn("CUDA doesn't list", engine.for_device(card, devices=[], have={"cuda"})["why"])


class VulkanLayerTest(unittest.TestCase):
    def test_layer_found_in_the_recording(self):
        with machine("desktop-amd"):
            self.assertTrue(engine.vulkan_layer().endswith("VkLayer_MESA_device_select.json"))

    def test_without_the_layer_two_cards_are_ambiguous(self):
        with machine("desktop-amd", writable=True) as root:
            os.remove(os.path.join(root, LAYER.lstrip("/")))
            with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/nonexistent", "XDG_CONFIG_HOME": "/nonexistent"}):
                self.assertIsNone(engine.vulkan_layer())
                card = gpu.choose()
                self.assertIn("device-select layer", engine.for_device(card, have={"vulkan"})["why"])
                self.assertEqual(engine.for_device(card, have={"rocm", "vulkan"})["backend"], "rocm")

    def test_without_the_layer_one_card_is_fine_unless_lavapipe(self):
        with machine("laptop-nvidia", writable=True) as root:
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "active\n")
            card = gpu.Device("0000:01:00.0")
            with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/nonexistent", "XDG_CONFIG_HOME": "/nonexistent"}), \
                    mock.patch("lodger.engine.vulkan_layer", return_value=None):
                self.assertIn("why", engine.for_device(card, have={"vulkan"}))          # the Intel graphics is listed too
                self.assertIsNone(engine.vulkan_ambiguous(card, [card]))
                write(root, "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json", "{}")
                self.assertIsNotNone(engine.vulkan_ambiguous(card, [card]))

    def test_nvidia_kept_off_for_another_card(self):
        with machine("laptop-nvidia"), mock.patch("lodger.engine.vulkan_layer", return_value=None):
            devs = gpu.devices()
            igpu = devs[0]
            self.assertIsNone(engine.vulkan_ambiguous(igpu, devs))   # NVIDIA's driver is disabled for it
            self.assertEqual(engine.vulkan_env(igpu, devs)["VK_LOADER_DRIVERS_DISABLE"], "*nvidia*")


class CudaDriverTest(unittest.TestCase):
    def test_driver_too_old_for_the_build_falls_back_to_vulkan(self):
        with machine("laptop-nvidia", writable=True) as root:
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "active\n")
            devs = gpu.devices()
            card = devs[1]
            with mock.patch("lodger.engine.cuda_major", return_value=13), \
                    mock.patch("lodger.engine.vulkan_layer", return_value="/x"):
                t = engine.for_device(card, devs, have={"cuda", "vulkan"})
                self.assertEqual(t["backend"], "vulkan")
                self.assertIn("580", engine.for_device(card, devs, have={"cuda"})["why"])
            with mock.patch("lodger.engine.cuda_major", return_value=12):
                self.assertEqual(engine.for_device(card, devs, have={"cuda"})["env"],
                                 {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"})


if __name__ == "__main__":
    unittest.main()
