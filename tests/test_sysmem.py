"""RAM numbers: /proc/meminfo shows the whole machine, but lodger may run inside a memory limit (a container,
a capped user slice): available RAM is then the room under the tightest limit."""
import unittest

from lodger import host, sysmem
from tests.support import machine, write

GiB = 1 << 30
SLICE = "/user.slice/user-1000.slice"
OWN = SLICE + "/user@1000.service/app.slice/lodger.service"


class MeminfoTest(unittest.TestCase):
    def test_no_limits_means_the_kernels_numbers(self):
        with machine("desktop-amd"):
            m = sysmem.meminfo()
        self.assertEqual(m["MemAvailable"], 12365144 * 1024)
        self.assertEqual(m["MemTotal"], 31722872 * 1024)

    def test_without_memavailable_free_memory_stands_in(self):
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/meminfo", "MemTotal:       31722872 kB\nMemFree:         1710076 kB\n")
            self.assertEqual(sysmem.meminfo()["MemAvailable"], 1710076 * 1024)

    def test_a_capped_slice_limits_available_ram(self):
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/self/cgroup", "0::" + OWN + "\n")
            write(root, "/sys/fs/cgroup" + SLICE + "/memory.max", str(8 * GiB) + "\n")
            write(root, "/sys/fs/cgroup" + SLICE + "/memory.current", str(6 * GiB) + "\n")
            write(root, "/sys/fs/cgroup" + SLICE + "/memory.stat", f"anon {5 * GiB}\ninactive_file {GiB}\n")
            m = sysmem.meminfo()
        self.assertEqual(m["MemAvailable"], 3 * GiB)            # 8 - 6 + 1 the kernel can drop
        self.assertEqual(m["MemTotal"], 8 * GiB)

    def test_memory_high_counts_like_a_limit(self):
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/self/cgroup", "0::" + OWN + "\n")
            write(root, "/sys/fs/cgroup" + SLICE + "/memory.high", str(4 * GiB) + "\n")
            write(root, "/sys/fs/cgroup" + SLICE + "/memory.current", str(3 * GiB) + "\n")
            self.assertEqual(sysmem.meminfo()["MemAvailable"], GiB)

    def test_lodgers_own_service_limit_doesnt_hold_the_model(self):
        # under systemd the model runs in a unit beside lodger.service, not inside it
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/self/cgroup", "0::" + OWN + "\n")
            write(root, "/sys/fs/cgroup" + OWN + "/memory.max", str(GiB) + "\n")
            write(root, "/sys/fs/cgroup" + OWN + "/memory.current", str(GiB // 2) + "\n")
            self.assertEqual(sysmem.meminfo()["MemAvailable"], 12365144 * 1024)

    def test_container_limit_cgroup_v2(self):
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/self/cgroup", "0::/\n")              # a cgroup namespace: the container is the root
            write(root, "/sys/fs/cgroup/memory.max", str(6 * GiB) + "\n")
            write(root, "/sys/fs/cgroup/memory.current", str(2 * GiB) + "\n")
            self.assertEqual(sysmem.meminfo()["MemAvailable"], 4 * GiB)

    def test_container_limit_cgroup_v1(self):
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/self/cgroup", "12:memory:/docker/abc\n11:cpu,cpuacct:/docker/abc\n")
            write(root, "/sys/fs/cgroup/memory/memory.limit_in_bytes", str(6 * GiB) + "\n")
            write(root, "/sys/fs/cgroup/memory/memory.usage_in_bytes", str(3 * GiB) + "\n")
            write(root, "/sys/fs/cgroup/memory/memory.stat", f"cache {GiB}\ntotal_inactive_file {GiB // 2}\n")
            self.assertEqual(sysmem.meminfo()["MemAvailable"], 3 * GiB + GiB // 2)

    def test_v1_unlimited_is_no_limit(self):
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/self/cgroup", "12:memory:/user.slice\n")
            write(root, "/sys/fs/cgroup/memory/user.slice/memory.limit_in_bytes", "9223372036854771712\n")
            write(root, "/sys/fs/cgroup/memory/user.slice/memory.usage_in_bytes", str(3 * GiB) + "\n")
            self.assertIsNone(host.cgroup_limit())

    def test_built_in_graphics_see_the_limit_too(self):
        with machine("laptop-apu", writable=True) as root:
            write(root, "/proc/self/cgroup", "0::/\n")
            write(root, "/sys/fs/cgroup/memory.max", str(2 * GiB) + "\n")
            write(root, "/sys/fs/cgroup/memory.current", "0\n")
            from lodger import gpu
            card = gpu.choose()
            self.assertTrue(card.integrated)
            self.assertLessEqual(card.free(), 2 * GiB)



class LodgerUseTest(unittest.TestCase):
    """What every lodger model holds on a card (any lodger's, a test's too): the desktop notes leave it out."""

    def test_models_in_any_lodger_fence_count(self):
        from lodger import gpu
        with machine("desktop-amd", writable=True) as root:
            card = gpu.choose()
            self.assertEqual(sysmem.lodger_use(card, ours=()), {"vram": 0, "gtt": 0})
            # pids 100 and 200 share one client on the card: as if another lodger's engine held it
            write(root, "/proc/100/cgroup", "0::" + SLICE + "/user@1000.service/app.slice/lodger-4242-3.service\n")
            use = sysmem.lodger_use(card, ours=())
            self.assertEqual(use, sysmem.gpu_use(card, [100]))
            self.assertGreater(use["vram"], 300 * (1 << 20))       # ~361 MiB on the card

    def test_this_lodgers_own_unfenced_model_counts(self):
        from lodger import gpu
        with machine("desktop-amd"):
            card = gpu.choose()
            self.assertEqual(sysmem.lodger_use(card, ours=[100]), sysmem.gpu_use(card, [100]))

    def test_no_card(self):
        self.assertEqual(sysmem.lodger_use(None, ours=()), {"vram": 0, "gtt": 0})

if __name__ == "__main__":
    unittest.main()
