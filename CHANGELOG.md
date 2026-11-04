# Changelog

All notable changes to lodger are written here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (while below 1.0, minor versions may change things).

## [Unreleased]

### Added
- `lodger serve` answers Anthropic's Messages API (`/v1/messages`, `/v1/messages/count_tokens`, `x-api-key`), so tools
  such as Claude Code can use a local model, and OpenAI's Responses API (`/v1/responses`); both continue an answer
  after a move like `/v1/chat/completions`. Rerank, token counts and llama-server's own `/completion` are passed on.
- Whole (non-streaming) answers continue after a move too, and `usage` counts the whole answer.
- `models.toml` is read again while serving: when it changes, on SIGHUP (`systemctl --user reload lodger`) or
  `POST /lodger/reload`. `/lodger/status` shows the engine, its log, waiting requests, retries and recent errors.
- `lodger service install|status|remove`: a systemd user unit for the lodger you run it with, so pip, pipx and uv
  installs start at login too; a unit it didn't write is left alone.
- `lodger --version`; a mistyped command gets a suggestion.
- No dependencies beyond Python: lodger reads model files itself (the `gguf` package is no longer needed).
- `lodger doctor` also shows lodger's version, whether `lodger serve` is up, libraries llama.cpp can't load, which
  apps lodger can see on the card, a learned desktop reserve that looks wrong, a recent card reset, a card at its
  critical temperature and cgroup memory caps; your home folder is written as `~`.
- AUR package `lodger-git` (`packaging/aur`), built, tested and installed on Arch in CI; a release workflow that
  publishes tagged versions to GitHub and PyPI. CI runs on Python 3.14 too.
- Apps lodger can't look into (another user's game, a root compositor, a system container, ROCm compute) count as
  needing the card on AMD cards, from the whole card's busy and memory numbers.
- The model leaves the card when it resets or hangs (the driver's crash report), when a laptop is unplugged
  (`gpu_on_battery` off) or when the card reaches its critical temperature, and doesn't start there meanwhile.
- Planned settings: `spec-type draft-mtp` and the n-gram types, `moe-cache-mib`, `ctx-checkpoints`,
  `image-max-tokens`. Model headers are read by lodger itself, about 20 times faster: the plans before a first
  answer take a fraction of a second instead of ~13 s.
- `lodger serve --ollama` (or `ollama = true` in config.toml): Ollama's API on `127.0.0.1:11434` for apps that
  only speak it (Raycast, Home Assistant, Open WebUI, editor plugins): `/api/chat` and `/api/generate` (NDJSON
  streaming or whole; `options`, `format`, tools, images, `think`), `/api/embed`, `/api/embeddings`,
  `/api/show`, `/api/tags`, `/api/ps`, `/api/version`. Answers continue after a move as on port 8181; the same
  Host, Origin and API key checks apply. Not started if the port is taken (by Ollama, say). `num_ctx` and
  `keep_alive` are lodger's to decide; pull, push, create, copy and delete answer that lodger doesn't do them.
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
- `lodger models`: a built-in list of ten Apache-2.0 and MIT chat models (dense and mixture of experts, about 2 to
  18 GiB) and which of them fit this PC right now, planned by lodger's own planner from each file's header before
  anything is downloaded. `lodger models get NAME` downloads one into `~/.local/share/lodger/models` (size and
  licence shown first; resumable; SHA-256 checked before it counts as done), `lodger models info NAME` shows
  where it comes from, and `lodger models rm NAME` moves it to the Trash.
- `lodger serve` serves the models `lodger models get` downloaded when `models.toml` lists no `folders` (or there
  is none); when it lists its own, `lodger models get` adds the download folder to them.

### Fixed
- Plans count the output layer on partial card placements (up to ~1 GiB more on the card than planned), the
  weights llama.cpp repacks when it runs on the processor only, llama-server's context checkpoints for hybrid and
  sliding-window models (2-8 GiB of RAM; llama.cpp's own sliding-window layouts for files that give only the
  window), MTP's extra copies of a hybrid model's state, and where those models keep their context. A damaged or
  hostile model file is a plain error, read with bounded memory.
- The learned desktop reserve could take lodger's own or another lodger's model, or an idle game, for desktop use
  (5.8 GiB on a 16 GiB card): no lodger model is noted any more, and the learned part is capped at a quarter of the
  card.
- The watchdog, "who needs the card" and the settle wait use the monotonic clock: setting the clock back or waking
  from suspend no longer blinds them or stops a healthy model. The watchdog stops a model with one write to its
  cgroup and no `systemctl` call can hang it; the kernel picks the model first if memory runs out.
- One card for the plan, the engine's command and its watchdog, also in `lodger serve`.
- RAM available respects container and user-slice memory limits.
- Two apps asking for different models take turns instead of making lodger switch back and forth; a waiting app
  that leaves ends its wait; `stream` must be true or false.
- `lodger models get` never replaces a file it didn't download, and a symlinked models.toml stays a link; a file
  named in two `[models.X]` sections is listed with why.
- `llama_cpp` and `runs_dir` must be full paths; `lodger setup --keep 0` is refused.
- Closing the terminal of `lodger serve` stops it (as a service, SIGHUP reloads models.toml); a closed terminal can't
  stop the monitor or the watchdog. The watchdog kills first and logs after, and never waits on `systemctl`.
- Structured answers (JSON schema, JSON mode, grammar) end with an error after a move instead of continuing into
  broken JSON; a Responses stream that moves before its first event still opens with `response.created`.
- A process whose `/proc` text isn't UTF-8 no longer breaks orphan reaping, doctor or the desktop notes; another
  lodger's model is never taken for a stranger's app; NVIDIA busy time stays sane after the clock is set back.
- `lodger plan` and `lodger run` take a model's name from models.toml (with its memory settings), and an unknown
  name is a plain error instead of a traceback.
- `lodger stress-serve` takes a model's name from `models.toml` as well as a file (a name used to end in "no models
  found"), and `lodger serve` says which file it was given doesn't exist. A request cut off because lodger is
  stopping is told so, instead of where the model was.
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
