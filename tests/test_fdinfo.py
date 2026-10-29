import os
import shutil
import unittest
from unittest import mock

from lodger import host
from lodger.gpu import fdinfo
from tests.support import FIXTURES, machine, write

MiB, KiB = 1 << 20, 1 << 10


def load(machine_name, pid, fd):
    with open(os.path.join(FIXTURES, machine_name, "proc", str(pid), "fdinfo", str(fd))) as f:
        return fdinfo.parse(f.read())


class ParseTest(unittest.TestCase):
    def test_not_a_drm_file(self):
        self.assertIsNone(fdinfo.parse("pos:\t0\nflags:\t02\n"))
        self.assertIsNone(fdinfo.parse(""))

    def test_split_on_first_colon_keeps_a_space_before_the_tab(self):
        kv = fdinfo.parse("drm-driver:\tamdgpu\ndrm-memory-gtt: \t2048 KiB\ndrm-pdev:\t0000:03:00.0\n")
        self.assertEqual(kv["drm-memory-gtt"], "2048 KiB")
        self.assertEqual(kv["drm-pdev"], "0000:03:00.0")      # the value's own colons stay

    def test_units(self):
        self.assertEqual(fdinfo.size("12 KiB"), 12 * KiB)
        self.assertEqual(fdinfo.size("3 MiB"), 3 * MiB)
        self.assertEqual(fdinfo.size("2 GiB"), 2 << 30)
        self.assertEqual(fdinfo.size("4096"), 4096)
        self.assertEqual(fdinfo.size("7 PiB"), 0)            # unknown unit: 0, not a wrong number


class AmdgpuTest(unittest.TestCase):
    def test_current_format_recorded(self):
        r = fdinfo.client(load("desktop-amd", 100, 38))
        self.assertEqual((r["driver"], r["pci"], r["id"]), ("amdgpu", "0000:03:00.0", "22"))
        self.assertEqual(r["vram"], 369612 * KiB)
        self.assertEqual(r["busy_ns"], 368068247665)          # gfx (busier than compute)
        self.assertEqual(r["cycles"], {})

    def test_current_format_prefers_resident_over_old_names(self):
        r = fdinfo.client(load("desktop-amd", 300, 58))
        self.assertEqual(r["gtt"], 2 * MiB)                   # drm-resident-gtt, same as drm-memory-gtt
        self.assertEqual(r["vram"], 12 * KiB)

    def test_old_format_with_space_before_tab(self):
        r = fdinfo.client(load("laptop-apu", 600, 9))
        self.assertEqual(r["vram"], 1024 * KiB)
        self.assertEqual(r["gtt"], 30720 * KiB)               # the old ':\t' split read 0 here
        self.assertEqual(r["busy_ns"], 123456789)             # dma (copy engine) doesn't count


class IntelTest(unittest.TestCase):
    def test_i915(self):
        r = fdinfo.client(load("laptop-nvidia", 500, 12))
        self.assertEqual(r["vram"], 0)                        # built-in graphics: no memory of its own
        self.assertEqual(r["gtt"], 23 * MiB)                  # system0 counts, stolen-system0 doesn't
        self.assertEqual(r["busy_ns"], 25662044495)           # render; video doesn't count

    def test_xe_counts_cycles(self):
        r = fdinfo.client(load("desktop-arc", 700, 5))
        self.assertEqual(r["vram"], 524288 * KiB)
        self.assertEqual(r["gtt"], 2052 * KiB)
        self.assertEqual(r["busy_ns"], 0)
        self.assertEqual(set(r["cycles"]), {"rcs", "ccs"})    # 3D and compute only
        self.assertEqual(r["cycles"]["rcs"], (28257900, 7655183225))


class GenericTest(unittest.TestCase):
    def test_unknown_driver_guesses_from_names(self):
        kv = fdinfo.parse("drm-driver:\tpanthor\ndrm-client-id:\t3\ndrm-pdev:\tfb000000.gpu\n"
                          "drm-engine-panthor:\t1000 ns\ndrm-resident-memory:\t4 MiB\n")
        r = fdinfo.client(kv)
        self.assertEqual(r["busy_ns"], 1000)
        self.assertEqual((r["vram"], r["gtt"]), (0, 4 * MiB))    # "memory" is the spec's name for system RAM


class UsageTest(unittest.TestCase):
    def test_one_client_counted_once_across_fds_and_processes(self):
        with machine("desktop-amd"):
            u = fdinfo.usage("0000:03:00.0", [100, 200, 300, 400])
        self.assertEqual(set(u), {100})                       # 200 holds the same client; 300 is on the other card
        self.assertEqual(u[100]["vram"], 369612 * KiB)

    def test_other_card(self):
        with machine("desktop-amd"):
            u = fdinfo.usage("0000:11:00.0", [100, 200, 300, 400])
        self.assertEqual(set(u), {300})


class FdFilterTest(unittest.TestCase):
    """Only fds pointing at /dev/dri are read, and never those of a sleeping card whose driver could wake."""

    def test_only_card_fds_are_read(self):
        with machine("desktop-amd"):
            self.assertEqual(sorted(fdinfo.drm_fds(100)), ["38", "39"])            # not fd 0 (/dev/null)
            self.assertEqual(fdinfo.drm_fds(400), [])                              # a pipe
            self.assertEqual(fdinfo.drm_fds(300, "0000:03:00.0"), [])              # its card is the other one
            self.assertEqual(fdinfo.node_card("renderD129"), ("0000:11:00.0", "amdgpu"))

    def test_without_fd_links_every_fdinfo_is_read(self):
        with machine("desktop-amd", writable=True) as root:
            shutil.rmtree(os.path.join(root, "proc/100/fd"))
            self.assertEqual(sorted(fdinfo.drm_fds(100)), ["0", "38", "39"])

    def test_sleeping_xe_card_is_not_read(self):
        with machine("desktop-arc", writable=True) as root:
            self.assertEqual(set(fdinfo.usage("0000:03:00.0", [700])), {700})
            write(root, "/sys/bus/pci/devices/0000:03:00.0/power/runtime_status", "suspended\n")
            with mock.patch("lodger.host.read", wraps=host.read) as rd:
                self.assertEqual(fdinfo.usage("0000:03:00.0", [700]), {})
            self.assertFalse([c for c in rd.call_args_list if "fdinfo" in c.args[0]])

    def test_sleeping_amd_card_is_still_read(self):
        with machine("desktop-amd", writable=True) as root:
            write(root, "/sys/bus/pci/devices/0000:03:00.0/power/runtime_status", "suspended\n")
            self.assertEqual(set(fdinfo.usage("0000:03:00.0", [100])), {100})   # amdgpu's fdinfo doesn't wake it


class RegionTest(unittest.TestCase):
    def test_total_when_no_resident(self):
        kv = fdinfo.parse("drm-driver:\tv3d\ndrm-client-id:\t1\ndrm-total-memory:\t8 MiB\ndrm-shared-memory:\t0\n")
        self.assertEqual(fdinfo.client(kv)["gtt"], 8 * MiB)

    def test_resident_wins_over_total(self):
        kv = fdinfo.parse("drm-driver:\tmsm\ndrm-client-id:\t1\ndrm-total-memory:\t8 MiB\ndrm-resident-memory:\t3 MiB\n")
        self.assertEqual(fdinfo.client(kv)["gtt"], 3 * MiB)

    def test_amdgpu_unchanged(self):
        r = fdinfo.client(load("desktop-amd", 300, 58))
        self.assertEqual((r["vram"], r["gtt"]), (12 * KiB, 2 * MiB))


if __name__ == "__main__":
    unittest.main()
