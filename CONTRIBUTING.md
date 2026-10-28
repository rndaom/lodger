# Contributing to lodger

Thank you for helping. lodger is for everyone's ordinary Linux PC, but it has only been seen on one, so reports
from other hardware are as valuable as code. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first.

## Dev setup

lodger needs Python 3.11 or newer and has one dependency (`gguf`, to read model files). Everything else is the
standard library.

```bash
git clone https://github.com/rndaom/lodger
cd lodger
uv venv && source .venv/bin/activate        # or: python -m venv .venv
uv pip install -e . ruff                    # or: pip install -e . ruff
```

Run it from the checkout without installing: `bin/lodger --help`. After the install above, `lodger --help` works too.

Run the checks (the same two CI runs, on Python 3.11, 3.12 and 3.13, with no graphics card):

```bash
ruff check .
python -m unittest discover tests
```

Tests must not need a GPU, a model, systemd or a llama.cpp build. Anything that touches those goes behind a
small function you can replace in a test, or reads files you recorded (a copy of a `/sys` folder or an fdinfo
file under `tests/`). Recorded files from other people's machines are very welcome.

## Safety rules for contributors

lodger exists so that a model cannot take down somebody's desktop, and it has happened (a model that filled a
16 GB card crashed the whole desktop). These rules are not optional, for code or for how you test it.

1. **Never fill the card.** Always keep at least **2 GiB of graphics memory free** for the desktop, on top of
   whatever lodger's own margins say. Don't lower the floor in code to make a test or a small card work; make
   the plan smaller instead.
2. **Test on the processor first.** Run with `--cpu-only` (or no `--allow-gpu`) and `lodger plan` before you let
   anything near the card. `lodger plan` only reads and prints.
3. **Start small on the card.** Use a small model and a short context for your first card run, and watch free
   graphics memory while it loads.
4. **Don't start `llama-server` by hand with a model that fills the card.** Use `lodger serve` or `lodger run`,
   which plan, fence and watch it. Tools that lodger doesn't guard (`llama-bench`, `llama-cli`, containers) need
   a model small enough to leave the desktop its 2 GiB.
5. **Stress tests run when you are away from the PC.** `lodger stress` and `lodger stress-serve` grab graphics
   memory and start games on purpose. Save your work, and run them when a crash costs the least.
6. **Refuse rather than guess.** If a piece of code can't tell how much memory is free, or whether an app is
   using the card, it must act as if it is not free and step aside. A model that is too careful is a bug; one
   that is too bold is a worse one.
7. **No app names.** lodger knows what apps *use*, not what they are. Don't add lists of games or programs.

## Adding a GPU vendor

Today the card-specific code is mixed into `sysmem.py`, `neighbors.py`, `plan.py` and `launch.py`, and is
written for AMD. A refactor that moves it into a `lodger/gpu/` package, **with one module per vendor**, is
landing now. This is the planned shape, so you can start reading the kernel and vendor docs for your card. Names
may still change a little, so check the code on `main` before you build on it.

Each backend (`lodger/gpu/<vendor>.py`) provides:

| Function | What it does |
|---|---|
| `devices()` | The cards of this vendor, found from sysfs only. **Must never wake a sleeping card** (laptop GPUs). |
| `power(dev)` | Whether the card is awake, asleep or on battery-saving, read without waking it. |
| `memory(dev)` | Total, used and free graphics memory in bytes, from the kernel's numbers; `None` while the card sleeps. |
| `clients(pid)` | What one app holds on the card: memory, and how long the card's 3D/compute engine has worked for it. |
| `llama_device(dev, backend)` | llama.cpp's name for that card (`Vulkan0`, `CUDA0`, `ROCm0`), matched by PCI id, never by index. |
| `launch_env(dev, backend)` | The environment variables the engine needs to see only that card. |

Plus **engine and memory-region tables** for the shared parser of the kernel's
[DRM fdinfo](https://docs.kernel.org/gpu/drm-usage-stats.html) files, which covers amdgpu, i915, xe and others:
which `drm-engine-*` entries are 3D/compute, and which memory regions live on the card rather than in system
RAM. A vendor whose driver gives no fdinfo (NVIDIA) provides its own `clients()` instead, for example through
NVML.

Things to get right for your vendor:

- **Never wake a sleeping GPU** with a status check. Opening `/dev/dri/*`, `lspci -vv`, NVML or `nvidia-smi`,
  and listing llama.cpp devices can all wake a laptop's big card.
- **Shared-memory cards (APUs, integrated)** have a tiny dedicated carve-out but can use system RAM. Budget with
  both, and watch other apps' shared-memory use too.
- **Free memory comes from the kernel**, not the compute runtime. On the tested AMD card the runtime's own number
  was about 700 MiB too high.
- Add recorded sample files under `tests/` and a test that reads them, so CI covers your backend without your card.

Open an issue before you start, so we can agree the interface and avoid double work.

## Filing a hardware report

If you run lodger on a card or machine that isn't listed as tested, please file a
[hardware report](https://github.com/rndaom/lodger/issues/new?template=hardware_report.yml). It asks for the output
of `lodger doctor` (coming; until it lands, give your GPU, driver and kernel versions by hand), your distro, RAM
and what happened. A report that says "it didn't work, and here is the log" helps as much as one that says "it
did".

## Style

- **Plain words in comments and messages.** The code explains itself to someone who knows Python but not
  graphics drivers: say what a thing is for and why, in everyday language ("the card's own memory", "steps
  aside"). Match the density of the comments around you: short, mostly on the why, no banner comments.
- Error messages say what happened and what to do, in plain words, with no stack trace for a user mistake.
- Follow the file you're in: naming, layout and idiom. `ruff check .` must pass; don't reformat code you aren't
  changing.
- Standard library first. A new dependency needs a good reason in the pull request.
- Don't break the four promises (see the [README](README.md)); when they disagree, the earlier one wins.

## Pull requests

- One change per pull request, with a short description of what and why.
- Say how you tested it, and on which hardware. For anything that touches the card, say how you stayed
  inside the safety rules above.
- Add or update a test for new behaviour, and a line in [CHANGELOG.md](CHANGELOG.md) under "Unreleased".
- Don't include model files, logs from `runs/`, or anything personal.

By contributing you agree that your work is released under the [MIT licence](LICENSE).
