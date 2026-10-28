# lodger

Local AI that lives on your PC as a **guest**. Your apps come first.

Most tools for running AI models on your own computer act as if the computer belongs to the AI: they fill the
graphics card and use every core. That is fine on a dedicated AI box. On the PC you also browse, chat, work and
game on, it ends with a frozen game, a crashed desktop, or a model you have to remember to switch off.

lodger assumes the opposite. The model is a lodger in a house that belongs to you. It lives in the room that is
free, keeps to itself, steps aside when you need the space, and never takes the house down with it.

- **It watches what every app really uses** on the graphics card (memory and busy time). It knows no app
  names: a game, a browser drawing 3D, a video editor and another AI tool are all treated the same.
- **When an app needs the card, the model steps aside to the processor in the middle of an answer**, and the
  answer carries on where it stopped. The app talking to lodger sees one answer, not an error.
- **When the card is quiet again, the model grows back onto it**, while the processor keeps answering until
  the swap is done.

Nothing else I could find does this. Other tools either use lists of game names or kill the answer when the
card is needed.

> **Status: early, Linux only.** lodger was built and tested on one PC (AMD RX 7800 XT). See
> [supported hardware](#supported-hardware) for what that means for yours.

## The four promises

When two of them disagree, the earlier one wins.

1. **Never crash the desktop.** Before loading, lodger looks at how much graphics memory (VRAM) and normal
   memory (RAM) is free *right now*, keeps a safety margin for everything else (always at least 2 GiB of VRAM),
   and refuses rather than guesses. While running it keeps watching and stops the model at once if memory gets
   too low.
2. **Share.** It never competes with the apps you have open. When one needs the card or memory, the model
   shrinks or steps aside. When the PC is quiet again, it grows back.
3. **Fast on ordinary hardware.** Models bigger than your card still run: lodger keeps the parts used for every
   word on the card and lets the rest wait in RAM.
4. **Sets itself up.** It measures the PC it is on and picks its own settings. Nobody should need to know what
   `-ngl` or `--n-cpu-moe` means.

## How it differs from Ollama and llama.cpp

lodger does not replace them. It is built on [llama.cpp](https://github.com/ggml-org/llama.cpp) and runs
llama.cpp's own `llama-server` as its engine, so it reads the same GGUF model files and speaks the same
OpenAI-style API. What lodger adds around the engine:

| | Ollama / plain llama.cpp | lodger |
|---|---|---|
| Where the model goes | Wherever it fits at start-up | Planned from what is free *now*, with margins kept for your desktop |
| An app needs the card | Both fight; one loses, often your app or the whole desktop | The model steps aside to the processor, mid-answer, without losing the answer |
| The card is free again | You restart the model yourself | The model grows back onto the card by itself |
| If the model misbehaves | It can take the card or all of RAM | It runs in a fence (own cgroup with a RAM cap and no swap, low priority, crash dumps off), watched ten times a second |
| Idle | Stays loaded | Optional: unloads after N idle minutes, loads on the next request (`--idle`) |

## Supported hardware

| Hardware | Status |
|---|---|
| AMD Radeon, discrete (tested: **RX 7800 XT**, 16 GB, ROCm) | Works. This is the one machine it has been tested on. |
| AMD Radeon, other models | Expected to work, **not tested**. Reports welcome. |
| NVIDIA | **In progress, help wanted.** Needs a backend that reads memory and per-app use through NVML. |
| Intel Arc / integrated | **In progress, help wanted.** The kernel exposes what is needed; the backend is not written. |
| AMD APUs / laptops with shared memory | **In progress, help wanted.** Budgets work differently when the card shares RAM with the system. |
| Processor only | Works (`--cpu-only`), without the stepping aside. |
| Windows, macOS | Not planned for now. Contributions welcome, see [CONTRIBUTING](CONTRIBUTING.md). |

"Works" means tested by hand on one machine, with stress tests that start games and grab graphics memory while
the model is answering (see [Safety](#safety)). It is not a guarantee for your setup. If you try it on other
hardware, please file a [hardware report](https://github.com/rndaom/lodger/issues/new?template=hardware_report.yml),
whether it worked or not.

## Requirements

- **Linux** with a systemd user session (lodger uses `systemd-run --user` for its fence) and cgroup v2.
  Flatpak is not supported: its private `/proc` hides what other apps use, and lodger needs to see that.
- **Python 3.11 or newer.** [uv](https://docs.astral.sh/uv/) brings its own.
- **llama.cpp** with `llama-server` and `llama-fit-params` built or downloaded. Today lodger looks for a build
  in a fixed place (`lodger/model.py`); a configurable location and `lodger setup` to fetch one for you are
  *coming*.
- **A model** as a `.gguf` file, from [Hugging Face](https://huggingface.co/models?library=gguf) for example.
  Check the model's licence yourself; lodger never downloads or redistributes models for you.
- **RAM:** 16 GB or more is comfortable. The only machine tested has 30 GiB. 8 GB works for small models on the
  processor, but is untested.

## Quickstart

```bash
uv tool install lodger        # or: pipx install lodger
lodger serve --allow-gpu      # keep a model ready for apps on this PC
```

Not on PyPI yet? Install from source instead:

```bash
uv tool install git+https://github.com/rndaom/lodger
```

lodger serves every model it finds in the folders listed in `~/.config/lodger/models.toml`. Copy
[`systemd/models.example.toml`](systemd/models.example.toml) there and point `folders` at your models. You can
also name files directly: `lodger serve ~/models/MyModel.gguf --allow-gpu`.

Without `--allow-gpu` lodger runs on the processor only. With it, lodger uses the card while it is free and
steps aside when it is not.

Then connect an app at **`http://127.0.0.1:8181/v1`** (see below), or just try it:

```bash
lodger ask "Why is the sky blue?"
```

Coming, not there yet:

```bash
lodger doctor     # check this PC: card, driver, systemd, llama.cpp, and say what will work
lodger setup      # download the right llama.cpp, write a systemd user unit that starts lodger at login
```

To start lodger at login today, use [`systemd/lodger.service`](systemd/lodger.service) as a starting point for
a user unit in `~/.config/systemd/user/` (it is the author's own, with paths for the author's PC).

## Connect an app

lodger speaks the usual OpenAI-style API on this PC only:

| | |
|---|---|
| Base URL | `http://127.0.0.1:8181/v1` |
| API key | none needed (type anything if the app insists) |
| Model name | any name from `GET /v1/models`; asking for another one switches models (5-20 seconds) |

```bash
curl http://127.0.0.1:8181/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.5-9b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Any app or library that lets you set an OpenAI base URL works: add it as a "custom OpenAI-compatible provider"
and use the URL above. Streaming and non-streaming both work. The API has `/v1/chat/completions`,
`/v1/models` and `/health`.

lodger's own endpoints:

- `GET /lodger/status`: where the model is (card or processor), why, and recent events.
- `POST /lodger/move`: move the model off the card now. It grows back later as usual.

If you use [pi](https://github.com/earendil-works/pi), `lodger pi` adds lodger as a provider listing every model
and opens pi (your other providers are kept).

An answer being written when the model moves continues where it stopped. The exception is a tool call in the
middle of a move: that answer ends with an error, and well-behaved clients retry.

## Commands

| Command | What it does |
|---|---|
| `lodger serve [MODEL.gguf ...]` | Keep models ready for apps on this PC. Steps aside, grows back. `--allow-gpu`, `--idle MIN`, `--port` |
| `lodger ask "question"` | Ask the model lodger is serving |
| `lodger pi` | Open [pi](https://github.com/earendil-works/pi) with lodger's models |
| `lodger plan MODEL` | Show what would go where and why, without running anything |
| `lodger run MODEL` | Run one model once, fenced in and watched |
| `lodger measure` | Log what each app uses, read-only |
| `lodger unpause` | Start again any service lodger paused, after a crash |
| `lodger stress MODEL`, `lodger stress-serve MODEL` | Tests that grab the card on purpose. See [Safety](#safety) |
| `lodger doctor` | *Coming.* Check this PC and say what will work |
| `lodger setup` | *Coming.* Download llama.cpp and set up the login service |

`lodger <command> --help` shows the options. `lodger plan` is the safest first step: it only reads and prints.

## Safety

lodger is built so that a model cannot take your desktop down, because that has happened: a model started by
hand filled a 16 GB card, and the desktop crashed with everything open.

- It keeps **at least 2 GiB of graphics memory free** for your desktop, and 4 GiB (or 15%) of RAM.
- It refuses to start rather than guess, and says why in plain words.
- The model runs in a **fence**: its own cgroup with a RAM cap and no swap, low CPU and disk priority, fewer
  threads than the processor has, and crash dumps off.
- A **watchdog** checks memory ten times a second while loading and four times a second after, and stops the
  model at once if free graphics memory or RAM gets too low.
- The server listens on **127.0.0.1 only**. See [SECURITY.md](SECURITY.md).

What lodger cannot promise: a bug in a graphics driver can still hang a card, and lodger is early software
tested on one machine. **Do not run `lodger stress` or `lodger stress-serve` while you have unsaved work.** They
grab graphics memory and start games on purpose to check that nothing crashes. Run them when you are away.

## Help wanted

lodger is for everyone's ordinary Linux PC, but it has seen one. The most useful things you can do:

- **Try it and file a [hardware report](https://github.com/rndaom/lodger/issues/new?template=hardware_report.yml)**,
  especially on NVIDIA, Intel, an APU or laptop, or an 8 GB card.
- **Add a graphics-card backend** for your vendor. [CONTRIBUTING.md](CONTRIBUTING.md) explains how.
- Fix a bug, improve these docs, or port the idea to another OS.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) first.

## Licence

[MIT](LICENSE). Models you run with it have their own licences.
