"""Any card lodger has no backend for (nouveau, virtio, a new driver ...). Apps' use is still read from
fdinfo when the driver follows the kernel's spec, so lodger can watch it, but it can't tell how much of the
card's memory is free, so it never puts the model there: it runs on the processor and says why."""
from . import fdinfo


def integrated(dev):
    return False


def memory(dev):
    return None


def why_unknown(dev):
    return (f"lodger doesn't know how to read free memory on this card (driver {dev.driver or 'none'}); "
            "adding a backend is described in lodger/gpu/__init__.py")


def usage(dev, pids):
    return fdinfo.usage(dev.pci, pids)
