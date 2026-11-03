import os
import shutil
import time
import unittest
from unittest import mock

from lodger import gpu, sysmem
from lodger.gpu import amd, nvidia
from lodger.neighbors import Neighbors
from tests.support import FIXTURES, FakeNVML, machine, read, write

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
            with mock.patch("time.time", return_value=t / 1e6 + 1):
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
            d = gpu.choose("03:00.0")
            self.assertFalse(d.integrated)
            self.assertIsNone(d.free())
            self.assertIn("Arc", d.why_unknown())

    def test_without_systemd_apps_are_grouped_by_program(self):
        with machine("desktop-arc"), mock.patch("lodger.sysmem._own", return_value=True):
            self.assertEqual(sysmem.app_name(700), "game.exe")


class NeighborsTest(unittest.TestCase):
    def test_busy_app_on_xe_counted_in_cycles(self):
        with machine("desktop-arc", writable=True) as root:
            nb = Neighbors(gpu.choose("03:00.0"))
            fd = "/proc/700/fdinfo/5"
            text = read(root, fd)
            clock = [1000.0]
            with mock.patch("lodger.neighbors.time.monotonic", side_effect=lambda: clock[0]), \
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


def add_card(root, src_machine, src_pci, pci):
    """Copy a recorded card into another recorded machine at `pci`."""
    shutil.copytree(os.path.join(FIXTURES, src_machine, "sys/bus/pci/devices", src_pci),
                    os.path.join(root, "sys/bus/pci/devices", pci), symlinks=True)


class ChoiceTest(unittest.TestCase):
    """gpu = "auto" ranks only cards whose memory lodger can read, by the same size for every vendor."""

    def test_unreadable_arc_does_not_beat_the_built_in_graphics(self):
        with machine("desktop-arc", writable=True) as root:
            add_card(root, "laptop-nvidia", "0000:00:02.0", "0000:00:02.0")
            self.assertEqual(gpu.choose().pci, "0000:00:02.0")
            self.assertIn("Arc", gpu.why_none("auto", [gpu.Device("0000:03:00.0")]))

    def test_bigger_nvidia_beats_smaller_amd(self):
        with machine("laptop-nvidia", writable=True) as root:
            add_card(root, "desktop-amd", "0000:03:00.0", "0000:02:00.0")
            write(root, "/sys/bus/pci/devices/0000:02:00.0/mem_info_vram_total", str(8 * GiB))
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "active\n")
            nvidia._nvml = FakeNVML(total=24 * GiB, used=GiB)
            self.assertEqual(gpu.choose().pci, "0000:01:00.0")
            nvidia._nvml = FakeNVML(total=6 * GiB, used=GiB)
            self.assertEqual(gpu.choose().pci, "0000:02:00.0")

    def test_sleeping_nvidia_still_counts(self):
        with machine("laptop-nvidia", writable=True) as root:
            add_card(root, "desktop-amd", "0000:03:00.0", "0000:02:00.0")
            write(root, "/sys/bus/pci/devices/0000:02:00.0/mem_info_vram_total", str(8 * GiB))
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "active\n")
            nvidia._nvml = FakeNVML(total=24 * GiB, used=GiB)
            gpu.choose()                                            # seen awake once: 24 GiB
            write(root, "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status", "suspended\n")
            self.assertEqual(gpu.choose().pci, "0000:01:00.0")      # its size is remembered, it isn't woken

    def test_this_pc_unchanged(self):
        with machine("desktop-amd"):
            self.assertEqual(gpu.choose().pci, "0000:03:00.0")


class AmdTest(unittest.TestCase):
    def test_unreadable_used_is_unknown_not_free(self):
        with machine("desktop-amd", writable=True) as root:
            os.remove(os.path.join(root, "sys/bus/pci/devices/0000:03:00.0/mem_info_vram_used"))
            card = gpu.Device("0000:03:00.0")
            self.assertIsNone(card.memory())
            self.assertIsNone(card.free())
            self.assertIn("mem_info", card.why_unknown())
            self.assertEqual(sysmem.vram_free(card), 0)             # the watchdog reads 0 free: it stops the model
            self.assertEqual(gpu.choose().pci, "0000:11:00.0")      # not planned for

    def test_garbage_is_unknown(self):
        with machine("desktop-amd", writable=True) as root:
            write(root, "/sys/bus/pci/devices/0000:03:00.0/mem_info_gtt_used", "oops\n")
            self.assertIsNone(gpu.Device("0000:03:00.0").memory())

    def test_integrated(self):
        with machine("desktop-amd", writable=True) as root:
            card = "/sys/bus/pci/devices/0000:03:00.0"
            os.remove(os.path.join(root, card.lstrip("/"), "mem_info_vram_vendor"))
            self.assertFalse(gpu.Device("0000:03:00.0").integrated)  # board_info: never on an APU (6.7+)
            os.remove(os.path.join(root, card.lstrip("/"), "board_info"))
            self.assertTrue(gpu.Device("0000:03:00.0").integrated)   # 6.7+ and no board_info: an APU
            write(root, "/proc/sys/kernel/osrelease", "6.1.0-18-amd64\n")
            self.assertFalse(gpu.Device("0000:03:00.0").integrated)  # older kernel: 16 GiB is no carve-out
            self.assertTrue(gpu.Device("0000:11:00.0").integrated)   # 512 MiB is
        with machine("laptop-apu"):
            self.assertTrue(gpu.choose().integrated)


class NvmlLifeTest(unittest.TestCase):
    def test_failure_is_retried_later(self):
        clock = [100.0]
        with machine("laptop-nvidia"), mock.patch("lodger.gpu.nvidia.NVML", side_effect=OSError("no lib")) as make, \
                mock.patch("lodger.gpu.nvidia.time.monotonic", side_effect=lambda: clock[0]):
            self.assertIsNone(nvidia.nvml())
            clock[0] += 1
            self.assertIsNone(nvidia.nvml())                   # within 5 s: not tried again
            self.assertEqual(make.call_count, 1)
            clock[0] += 5
            self.assertIsNone(nvidia.nvml())                   # after 5 s: tried again, next wait 10 s
            self.assertEqual(make.call_count, 2)
            self.assertEqual(nvidia._nvml_error[2], 10.0)
            self.assertIn("no lib", nvidia.why_unknown(gpu.Device("0000:00:02.0")))

    def test_session_closed_where_the_card_may_sleep(self):
        class Real(nvidia.NVML):
            def __init__(self):
                self.handles, self.closed = {}, False

            def shutdown(self):
                self.closed = True

            def memory(self, pci):
                return 8 * GiB, 6 * GiB, 2 * GiB
        with machine("laptop-nvidia", writable=True) as root, mock.patch.object(nvidia, "LINGER", 0.05):
            dev = "/sys/bus/pci/devices/0000:01:00.0"
            write(root, dev + "/power/runtime_status", "active\n")
            card = gpu.Device("0000:01:00.0")
            for control, closed in (("on\n", False), ("auto\n", True)):
                write(root, dev + "/power/control", control)
                s = nvidia._nvml = Real()
                self.assertEqual(card.free(), 6 * GiB)
                time.sleep(0.2)
                self.assertEqual(s.closed, closed)
                self.assertEqual(nvidia._nvml is None, closed)

    def test_driver_version_and_cuda(self):
        with machine("laptop-nvidia"):
            self.assertEqual(nvidia.driver_version(), (570, 153, 2))
            self.assertTrue(nvidia.cuda_ok(12))
            self.assertFalse(nvidia.cuda_ok(13))
        with machine("desktop-amd"):
            self.assertIsNone(nvidia.driver_version())
            self.assertIsNone(nvidia.cuda_ok(13))
        self.assertTrue(nvidia.cuda_ok(13, (580, 65, 6)))


class CardTroubleSignsTest(unittest.TestCase):
    def test_crash_report_names_the_card(self):
        with machine("desktop-amd", writable=True) as root:
            d = gpu.choose()
            self.assertFalse(d.crash_report())
            os.makedirs(os.path.join(root, "sys/class/devcoredump/devcd3"))
            os.symlink(os.path.join(root, "sys/bus/pci/devices/0000:11:00.0"),
                       os.path.join(root, "sys/class/devcoredump/devcd3/failing_device"))
            self.assertFalse(d.crash_report())                  # the built-in graphics' report
            os.makedirs(os.path.join(root, "sys/class/devcoredump/devcd4"))
            os.symlink(os.path.join(root, "sys/bus/pci/devices/0000:03:00.0"),
                       os.path.join(root, "sys/class/devcoredump/devcd4/failing_device"))
            self.assertTrue(d.crash_report())

    def test_heat_from_hwmon(self):
        with machine("desktop-amd", writable=True) as root:
            h = "/sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon2"
            for name, now, crit in (("temp1", 43000, 100000), ("temp2", 111000, 110000), ("temp3", 66000, 0)):
                write(root, f"{h}/{name}_input", f"{now}\n")
                write(root, f"{h}/{name}_crit", f"{crit}\n")
            write(root, f"{h}/temp2_label", "junction\n")
            d = gpu.choose()
            self.assertEqual(d.heat()[:2], [("temp1", 43.0, 100.0), ("junction", 111.0, 110.0)])
            self.assertIn("junction 111", d.too_hot())
            write(root, f"{h}/temp2_input", "90000\n")
            self.assertIsNone(d.too_hot())
            write(root, "/sys/bus/pci/devices/0000:03:00.0/power/runtime_status", "suspended\n")
            self.assertEqual(d.heat(), [])                      # never woken to look


if __name__ == "__main__":
    unittest.main()
