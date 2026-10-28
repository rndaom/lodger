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
- Repository files for contributors: README, MIT licence, CONTRIBUTING, Code of Conduct, SECURITY, issue
  forms, pull request template, CI (ruff and unit tests on Python 3.11 to 3.13), `pyproject.toml` for
  installing with pip, pipx and uv, and a `lodger` command (`lodger.cli`) that `bin/lodger` also calls.
