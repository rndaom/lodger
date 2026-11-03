"""RAM numbers: /proc/meminfo shows the whole machine, but lodger may run inside a memory limit (a container,
a capped user slice): available RAM is then the room under the tightest limit."""
import os
import unittest
from unittest import mock

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

    def test_a_none_fence_model_shares_lodgers_own_limit(self):
        with machine("desktop-amd", writable=True) as root, mock.patch.dict(os.environ, {"LODGER_FENCE": "none"}):
            write(root, "/proc/self/cgroup", "0::" + OWN + "\n")
            write(root, "/sys/fs/cgroup" + OWN + "/memory.max", str(4 * GiB) + "\n")
            write(root, "/sys/fs/cgroup" + OWN + "/memory.current", str(GiB) + "\n")
            self.assertEqual(sysmem.meminfo()["MemAvailable"], 3 * GiB)

    def test_a_containers_scope_limit_holds(self):
        # docker without a cgroup namespace: the container's limit is on its scope, lodger's own folder
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/self/cgroup", "0::/system.slice/docker-0123abcd.scope\n")
            write(root, "/sys/fs/cgroup/system.slice/docker-0123abcd.scope/memory.max", str(6 * GiB) + "\n")
            write(root, "/sys/fs/cgroup/system.slice/docker-0123abcd.scope/memory.current", str(2 * GiB) + "\n")
            self.assertEqual(sysmem.meminfo()["MemAvailable"], 4 * GiB)

    def test_under_systemd_the_models_own_slice_counts(self):
        # lodger started from somewhere else in the user manager: its model goes to app.slice all the same
        mgr = SLICE + "/user@1000.service"
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/self/cgroup", "0::" + mgr + "/session.slice/lodger.service\n")
            write(root, "/sys/fs/cgroup" + mgr + "/session.slice/memory.max", str(GiB) + "\n")
            write(root, "/sys/fs/cgroup" + mgr + "/session.slice/memory.current", "0\n")
            self.assertEqual(sysmem.meminfo()["MemAvailable"], 12365144 * 1024)
            write(root, "/sys/fs/cgroup" + mgr + "/app.slice/memory.max", str(4 * GiB) + "\n")
            write(root, "/sys/fs/cgroup" + mgr + "/app.slice/memory.current", str(3 * GiB) + "\n")
            self.assertEqual(sysmem.meminfo()["MemAvailable"], GiB)

    def test_above_memory_high_there_is_no_room(self):
        # the kernel throttles everything in a cgroup over memory.high (the desktop too, when it is the user's
        # slice): no room, so the watchdog gives the memory back, as it would near memory.max
        with machine("desktop-amd", writable=True) as root:
            write(root, "/proc/self/cgroup", "0::" + OWN + "\n")
            write(root, "/sys/fs/cgroup" + SLICE + "/memory.high", str(4 * GiB) + "\n")
            write(root, "/sys/fs/cgroup" + SLICE + "/memory.current", str(5 * GiB) + "\n")
            write(root, "/sys/fs/cgroup" + SLICE + "/memory.stat", f"inactive_file {GiB // 2}\n")
            self.assertEqual(sysmem.meminfo()["MemAvailable"], 0)

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

    def test_another_lodgers_unfenced_model_counts(self):
        # a 'none' fence has no lodger-* cgroup: its programs carry LODGER_FENCE_NAME instead (fence.find_orphans)
        from lodger import gpu
        with machine("desktop-amd", writable=True) as root:
            card = gpu.choose()
            write(root, "/proc/100/environ", "PATH=/usr/bin\0LODGER_FENCE_NAME=lodger-4242-1\0")
            write(root, "/proc/300/environ", "PATH=/usr/bin\0NOT_LODGER_FENCE_NAME=lodger-1-1\0")
            self.assertEqual(sysmem.lodger_use(card, ours=()), sysmem.gpu_use(card, [100]))

    def test_no_card(self):
        self.assertEqual(sysmem.lodger_use(None, ours=()), {"vram": 0, "gtt": 0})

class StrangeBytesTest(unittest.TestCase):
    """Any process may carry bytes that aren't UTF-8 (its environment, its name, a cgroup someone named): reading
    them must never stop orphan reaping, lodger doctor or the desktop notes."""

    def test_proc_files_that_arent_utf8_are_read(self):
        from lodger import fence
        with machine("desktop-amd", writable=True) as root:
            def raw(rel, data):
                path = os.path.join(root, rel.lstrip("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(data)
            raw("/proc/900/environ", b"LANG=C\0NAME=caf\xe9\0")
            raw("/proc/900/comm", b"caf\xe9\n")
            raw("/proc/900/cmdline", b"caf\xe9\0")
            raw("/proc/900/stat", b"900 (caf\xe9) S 1 900 900 0")
            raw("/proc/900/cgroup", b"0::" + SLICE.encode() + b"/user@1000.service/app.slice/app-caf\xe9.scope\n")
            raw("/proc/901/environ", b"LODGER_FENCE_NAME=lodger-4242-1\0")
            self.assertEqual(fence.named(), {"lodger-4242-1": [901]})
            self.assertFalse(fence.owner_alive(900))
            self.assertIn(900, fence._group(900))
            self.assertTrue(sysmem.app_name(900).startswith("caf"))
            self.assertTrue(sysmem.comm(900).startswith("caf"))
            # a cgroup named with a broken escape (systemd escapes as \x2d; anyone may name their own)
            raw("/proc/902/cgroup", b"0::" + SLICE.encode() + b"/user@1000.service/app.slice/app-x\\x2.scope\n")
            self.assertEqual(sysmem.app_name(902), "x\\x2")
            raw("/proc/903/cgroup", b"0::" + SLICE.encode() + b"/user@1000.service/app.slice/app-gtk\\x2dlaunch-ab12.scope\n")
            self.assertEqual(sysmem.app_name(903), "gtk-launch")


if __name__ == "__main__":
    unittest.main()
