"""Intel graphics (i915 and xe). Per-app use comes from fdinfo. Built-in graphics share the PC's RAM.

TODO: free memory on Intel Arc cards (their own memory) needs an ioctl (xe: DRM_XE_DEVICE_QUERY_MEM_REGIONS;
i915: DRM_I915_QUERY_MEMORY_REGIONS, which also needs CAP_PERFMON for the free number). Until someone adds
it, lodger doesn't use Arc cards and says so."""
from . import fdinfo


def integrated(dev):
    """Intel's built-in graphics always sits at PCI 00:02.0; Arc cards are elsewhere."""
    return dev.pci.endswith(":00:02.0")


def memory(dev):
    if integrated(dev):
        # shares RAM; the planner budgets it from RAM (no GTT numbers in sysfs for i915/xe)
        return {"total": 0, "used": 0, "gtt_total": None, "gtt_used": None}
    return None


def why_unknown(dev):
    return None if integrated(dev) else ("lodger can't read free memory on Intel Arc cards yet "
                                         "(see lodger/gpu/intel.py; a contribution is welcome)")


def usage(dev, pids):
    return fdinfo.usage(dev.pci, pids)
