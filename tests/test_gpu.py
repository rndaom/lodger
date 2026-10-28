import unittest
from unittest import mock

from lodger import gpu, sysmem
from lodger.gpu import amd, nvidia
from lodger.neighbors import Neighbors
from tests.support import FakeNVML, machine, read, write

GiB, MiB = 1 << 30, 1 << 20


class DesktopAmdTest(unittest.TestCase):
    """Recorded on the project's PC: RX 7800 XT + built-in Radeon graphics."""

    def test_devices(self):
        with machine("desktop-amd"):
            devs = gpu.devices()
            self.assertEqual([d.pci for d in devs], ["0000:03:00.0", "0000:11:00.0"])   # the bridge isn't a card
            card, igpu = devs
            self.assertEqual((card.vendor, card.driver, card.card, card.integrated), ("AMD", "amdgpu", "card1", False))
            self.assertTrue(igpu.integrated)
            self.assertEqual(card.outputs(), ["DP-1", "DP-2"])
            self.assertEqual(card.power(), "active")

    def test_choose_picks_the_card_with_its_own_memory(self):
        with machine("desktop-amd"):
            self.assertEqual(gpu.choose().pci, "0000:03:00.0")
            self.assertEqual(gpu.choose("11:00.0").pci, "0000:11:00.0")
            self.assertIsNone(gpu.choose("none"))
            self.assertIsNone(gpu.choose("0000:99:00.0"))
            self.assertIn("99:00.0", gpu.why_none("0000:99:00.0"))

    def test_free_memory_on_the_card_is_total_minus_used(self):
        with machine("desktop-amd") as root:
            d = gpu.choose()
            total = int(read(root, "sys/bus/pci/devices/0000:03:00.0/mem_info_vram_total"))
            used = int(read(root, "sys/bus/pci/devices/0000:03:00.0/mem_info_vram_used"))
            self.assertEqual(d.free(), total - used)
            self.assertEqual(sysmem.vram_free(), total - used)

    def test_rocm_number_from_kfd_topology(self):
        with machine("desktop-amd"):
            card, igpu = gpu.devices()
            self.assertEqual(amd.kfd_index(card), 0)
            self.assertEqual(amd.kfd_index(igpu), 1)

    def test_headset_battery_is_not_the_pcs(self):
        with machine("desktop-amd"):
            self.assertEqual(gpu.power_source()[0], "ac")

    def test_app_names(self):
        with machine("desktop-amd"):
            self.assertEqual(sysmem.app_name(100), "hyprland")
            self.assertEqual(sysmem.app_name(300), "vesktop")
            self.assertEqual(sysmem.app_name(400), "system")


class LaptopNvidiaTest(unittest.TestCase):
    """Intel built-in graphics + an NVIDIA card that is asleep, on battery."""

    def test_devices_and_choice(self):
        with machine("laptop-nvidia"):
            igpu, card = gpu.devices()
            self.assertTrue(igpu.integrated)
            self.assertFalse(card.integrated)
            self.assertTrue(card.asleep())
            self.assertEqual(gpu.choose().pci, "0000:01:00.0")

    def test_never_wakes_a_sleeping_nvidia_card(self):
        with machine("laptop-nvidia"):
            fake = FakeNVML()
            nvidia._nvml = fake
            card = gpu.choose()
            self.assertIsNone(card.memory())
            self.assertEqual(card.usage([1, 2]), {})
            self.assertIn("asleep", card.why_unknown())
            self.assertEqual(fake.calls, 0)

    def test_battery(self):
        with machine("laptop-nvidia"):
            src, detail = gpu.power_source()
            self.assertEqual(src, "battery")
            self.assertIn("BAT0", detail)
            self.assertNotIn("hidpp", detail)               # a mouse's battery

    def test_built_in_graphics_budget_comes_from_ram(self):
        with machine("laptop-nvidia"):
            igpu = gpu.choose("00:02.0")
            self.assertEqual(igpu.free(), 9 * GiB)            # MemAvailable; i915 shows no GTT numbers

    def test_nvidia_awake(self):
        with machine("laptop-nvidia", writable=True) as root:
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "active\n")
            nvidia._nvml = FakeNVML(total=8 * GiB, used=3 * GiB, procs={500: 2 * GiB, 900: GiB},
                                    samples=[(500, 10_000_000_000_000_000, 50)])
            card = gpu.choose()
            self.assertEqual(card.free(), 5 * GiB)
            u = card.usage([500, 501])
            self.assertEqual(set(u), {500})
            self.assertEqual(u[500]["vram"], 2 * GiB)

    def test_nvidia_busy_adds_up_like_a_counter(self):
        with machine("laptop-nvidia", writable=True) as root:
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "active\n")
            t = 10_000_000_000_000_000
            fake = FakeNVML(procs={500: GiB}, samples=[(500, t, 60)])
            nvidia._nvml = fake
            card = gpu.choose()
            with mock.patch("time.time", return_value=t / 1e6):
                card.usage([500])
            fake.samples.append((500, t + 1_000_000, 60))     # one second later, 60% busy
            b = card.usage([500])[500]["busy_ns"]
            self.assertAlmostEqual(b - 100_000_000, 600_000_000, delta=1_000_000)

    def test_cuda_number(self):
        with machine("laptop-nvidia"):
            devs = gpu.devices()
            self.assertEqual(nvidia.cuda_index(devs[1], devs), 0)
            self.assertEqual(nvidia.minor(devs[1]), "0")


class LaptopApuTest(unittest.TestCase):
    def test_shared_memory_budget(self):
        with machine("laptop-apu"):
            d = gpu.choose()
            self.assertTrue(d.integrated)
            # smaller of RAM available (18 GiB) and GTT left (14 - 2 GiB)
            self.assertEqual(d.free(), 12 * GiB)
            self.assertEqual(d.total(), 14 * GiB)

    def test_plugged_in(self):
        with machine("laptop-apu"):
            self.assertEqual(gpu.power_source()[0], "ac")


class ArcTest(unittest.TestCase):
    def test_arc_memory_is_a_todo_and_says_so(self):
        with machine("desktop-arc"):
            d = gpu.choose()
            self.assertFalse(d.integrated)
            self.assertIsNone(d.free())
            self.assertIn("Arc", d.why_unknown())

    def test_without_systemd_apps_are_grouped_by_program(self):
        with machine("desktop-arc"), mock.patch("lodger.sysmem._own", return_value=True):
            self.assertEqual(sysmem.app_name(700), "game.exe")


class NeighborsTest(unittest.TestCase):
    def test_busy_app_on_xe_counted_in_cycles(self):
        with machine("desktop-arc", writable=True) as root:
            nb = Neighbors(gpu.choose())
            fd = "/proc/700/fdinfo/5"
            text = read(root, fd)
            clock = [1000.0]
            with mock.patch("lodger.neighbors.time.time", side_effect=lambda: clock[0]), \
                    mock.patch("lodger.sysmem.app_name", return_value="game"):
                total, rcs = 7655183225, 28257900
                reason = None
                for step in range(6):
                    # each second the clock moves 1e9 cycles and the game keeps 3D busy 60% of that
                    write(root, fd, text.replace(f"drm-cycles-rcs:\t{28257900}", f"drm-cycles-rcs:\t{rcs}")
                          .replace(f"drm-total-cycles-rcs:\t{7655183225}", f"drm-total-cycles-rcs:\t{total}"))
                    reason = nb.update() or reason
                    clock[0] += 1.0
                    total += 1_000_000_000
                    rcs += 600_000_000
            self.assertIsNotNone(reason)
            self.assertIn("game is keeping the graphics card busy (60%)", reason)

    def test_own_model_never_counts(self):
        with machine("desktop-amd"):
            nb = Neighbors(gpu.choose(), ours=lambda: [100])
            self.assertEqual(nb._sample(), {})


if __name__ == "__main__":
    unittest.main()
