"""Intel graphics (i915 and xe). Per-app use comes from fdinfo. Built-in graphics share the PC's RAM.

Free memory on Intel Arc cards (their own memory) isn't used yet, and here is why (checked 2026-11-03):
- sysfs has no "memory used" file for i915 or xe (amdgpu's mem_info_vram_used has no Intel twin; xe even
  dropped its vram size file in favour of the query ioctl), and hwmon gives power and heat, not memory;
- the ioctls (xe: DRM_XE_DEVICE_QUERY_MEM_REGIONS; i915: DRM_I915_QUERY_MEMORY_REGIONS) need /dev/dri opened,
  which can wake a sleeping card, and report the used/unallocated number only with CAP_PERFMON;
- adding up fdinfo (drm-resident-vram0 / local0) sees only this user's apps, so a root compositor or another
  user's game would be missed: a number that is too high, the unsafe direction (CONTRIBUTING rule 6).
A way in: debugfs (vram0_mm, root only), or an ioctl helper run while the card is known to be awake, with
CAP_PERFMON. Until then lodger runs on the processor next to an Arc card and says why."""
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
