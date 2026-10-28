# Recorded machines

Copies of the files lodger reads under `/sys` and `/proc`, so the tests run anywhere without a graphics card.
`lodger.host.ROOT` is pointed at one of these folders.

- `desktop-amd`: **recorded** on the project's PC (2026-10-28, kernel 7.2): RX 7800 XT + Ryzen built-in graphics,
  a headset battery (`scope=Device`), the KFD topology, and real amdgpu fdinfo (current format: `drm-resident-*`
  plus the old `drm-memory-gtt: \t` lines). Process ids renamed to 100-400.
- `laptop-nvidia`, `laptop-apu`, `desktop-arc`: **written by hand** in the kernel's documented formats
  (https://docs.kernel.org/gpu/drm-usage-stats.html and the drivers' sources), because nobody on the project has
  that hardware yet: Intel i915 + NVIDIA asleep on battery; an AMD APU with an older kernel's amdgpu fdinfo
  (`drm-memory-*` only); an Intel Arc card on xe, which counts busy time in cycles.

Real recordings from those machines are very welcome (see CONTRIBUTING): `lodger doctor` says what is read.
