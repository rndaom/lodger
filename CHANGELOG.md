# Changelog

All notable changes to lodger are written here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (while below 1.0, minor versions may change things).

## [Unreleased]

### Added
- `lodger serve`: a local server for apps on this PC at `http://127.0.0.1:8181/v1` (OpenAI-style
  `/v1/chat/completions` with or without streaming, `/v1/models`, `/health`), plus `GET /lodger/status` and
  `POST /lodger/move`.
- Serves every model in `~/.config/lodger/models.toml` (folders to scan, names, per-model llama-server settings),
  one at a time; a request naming another model switches to it.
- Steps aside: when an app needs the graphics card or memory runs low, the model moves to the processor in the
  middle of an answer, and the answer continues where it stopped.
- Grows back: when the card has been quiet for a while, the model returns to it while the processor keeps
  answering until the swap. Repeated moves make it wait longer before trying again.
- Automatic context length: as long as the model supports (up to 128K) and as memory allows, decided once per
  session so a conversation that fits before a move fits after it.
- `--idle MIN`: unload the model after MIN minutes unused and load it again on the next request.
- `--pause SERVICE`: stop another AI server's systemd user service while lodger uses the card, and start it
  again afterwards. `lodger unpause` restores it after a crash.
- Who needs the card, without app names: per-app graphics memory and engine busy time from `/proc/<pid>/fdinfo`.
- `lodger plan` and `lodger run`: placement planned from the free graphics memory and RAM right now (keeping
  2 GiB of graphics memory free for the desktop), with experts of mixture-of-experts models moved to RAM first,
  checked against llama.cpp's own estimate and a learned overhead; run fenced in a systemd cgroup with a RAM
  cap, no swap, low priority and crash dumps off, and a watchdog that stops the model if memory gets low.
- `lodger measure`: read-only log of what each app uses.
- `lodger stress` and `lodger stress-serve`: tests that grab graphics memory and start games while a model
  answers, to check nothing crashes.
- `lodger ask` and `lodger pi`: a small client, and a pi provider listing every model.
- `lodger serve` also passes `/v1/completions`, `/v1/embeddings`, `/infill`, `/tokenize`, `/detokenize` and
  `/apply-template` through to the model (an answer cut by a move ends with an error), answers HEAD, and reads
  chunked request bodies.
- `lodger serve` and `lodger run` stop models that a crashed or killed lodger left running before they start one;
  `lodger doctor` lists any left behind.
- Repository files for contributors: README, MIT licence, CONTRIBUTING, Code of Conduct, SECURITY, issue
  forms, pull request template, CI (ruff and unit tests on Python 3.11 to 3.13), `pyproject.toml` for
  installing with pip, pipx and uv, and a `lodger` command (`lodger.cli`) that `bin/lodger` also calls.

### Fixed
- Nothing runs unwatched: an error in the watchdog's checks stops the model, a watchdog that stops beating gets the
  model stopped, and `lodger serve`'s monitor carries on after an error.
- llama-server settings in `models.toml` that change memory are now planned for; settings lodger sets itself, or
  can't plan for yet, make that model "not served" with the reason. llama-server's RAM prompt cache is sized to fit
  inside the model's fence.
- A model that fails to start is tried again after a while instead of being left out until restart.
- Long prompts on the processor no longer time out; a continued answer keeps one id and its `max_tokens`, continues
  an app's own start of the answer, and ends with an error when a tool call or JSON answer was cut.
- pi's `models.json` and `settings.json` are left alone when they aren't valid JSON, instead of being replaced.
- `--ctx 0` / `--min-ctx 0` no longer loop forever; a damaged probe cache or calibration line no longer breaks
  planning.
- Unreadable AMD card memory counts as no room instead of all free; card choice ranks cards by their real size;
  polling no longer wakes sleeping Intel xe or NVIDIA cards; card-choosing variables from your shell no longer reach
  the engine; the systemd memory cap is checked to be in force.
- `lodger setup` installs all-or-nothing, picks CUDA 12 or 13 by the driver, and finds any release by `--tag`.
- Bad `config.toml` values give a plain message instead of a traceback; unknown keys a warning.
- `lodger stress` and `stress-serve` work when lodger is installed, and see NVIDIA's `Xid` faults.
