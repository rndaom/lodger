"""Shared helpers: run lodger's readers against a recorded machine in tests/fixtures."""
import contextlib
import os
import shutil
import tempfile

from lodger import host
from lodger.gpu import nvidia

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


@contextlib.contextmanager
def machine(name, writable=False):
    """Read /sys and /proc from tests/fixtures/<name> (a temporary copy when the test changes files)."""
    src = os.path.join(FIXTURES, name)
    tmp = None
    if writable:
        tmp = tempfile.mkdtemp(prefix="lodger-test-")
        root = os.path.join(tmp, name)
        shutil.copytree(src, root, symlinks=True)
    else:
        root = src
    old = host.ROOT
    host.ROOT = root
    nvidia._nvml, nvidia._nvml_error, nvidia._busy = None, None, {}
    try:
        yield root
    finally:
        host.ROOT = old
        nvidia._nvml, nvidia._nvml_error, nvidia._busy = None, None, {}
        if tmp:
            shutil.rmtree(tmp)


def write(root, rel, text):
    p = os.path.join(root, rel.lstrip("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


class FakeNVML:
    """Stands in for libnvidia-ml: memory, processes and utilization samples per PCI address."""

    def __init__(self, total=8 << 30, used=1 << 30, procs=None, samples=None):
        self.total, self.used = total, used
        self.procs = procs or {}
        self.samples = samples or []
        self.calls = 0

    def memory(self, pci):
        self.calls += 1
        return self.total, self.total - self.used, self.used

    def processes(self, pci):
        self.calls += 1
        return dict(self.procs)

    def utilization(self, pci, since_us):
        self.calls += 1
        return [s for s in self.samples if s[1] > since_us]


def read(root, rel):
    with open(os.path.join(root, rel.lstrip("/"))) as f:
        return f.read()
