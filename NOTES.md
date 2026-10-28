# lodger: local AI that lives on your PC as a guest

## Goal
Run a local AI model on the PC people already own (a gaming PC, a work PC, a laptop), **next to**
whatever they use every day: browsers, chat and music apps, things running in the background, games.
No new hardware, no moving monitor cables, nothing to tune. The model is a lodger: it lives in the
room that's free, keeps to itself, steps aside when the owner needs the space, and never takes the
house down with it.

It shouldn't favour one kind of user. Lodger knows no app names: it watches what every app actually
uses (memory, graphics card time) and gives way to whatever needs it, whether that's a game, a
browser drawing 3D, video editing or another AI tool.

Other tools (llama.cpp's own defaults, Strata, people with AI rigs) assume the machine belongs to the
AI: fill the graphics card, use every core. We assume the opposite. That is the whole idea.

Built on our patched llama.cpp (`../ds41-local/src/llama.cpp`, tag b11936 + expert prefetch in
`ggml/src/ggml-cpu/ggml-cpu.c` + the two-drive stripe loader in `src/llama-mmap.cpp`). Not on Strata:
Strata hung this graphics card on 2026-10-28 and the desktop crashed. llama.cpp never has here.
Our own tool and fork; nothing goes upstream to llama.cpp. (Named "good-neighbor" until 2026-10-28.)

## The four promises, in order
When two promises disagree, the earlier one wins.

1. **Never crash the desktop.** Before loading, look at how much graphics memory (VRAM) and normal
   memory (RAM) is free *right now*, keep a safety margin for everything else, and refuse rather than
   guess. While running, keep watching and stop the model at once if free memory gets too low.
   Avoid the things that have hung this card (Strata's engine).
2. **Share.** Never compete with the apps you have open. When one of them needs the graphics card or
   memory, the model shrinks (fewer parts on the card, shorter memory of the chat) or steps aside;
   when the PC is quiet again, it grows back.
3. **Fast on ordinary hardware.** Use the tricks from ds41-local: read model files from both SSDs
   at once and ask the disk for the next pieces before they are needed. A second SSD is a bonus,
   never a requirement.
4. **Set itself up.** It measures the PC it is on and picks its own settings. Nobody should need to
   know what "-ngl" or "--n-cpu-moe" means.

## Words used here (plain meaning)
- **VRAM**: the graphics card's own memory (16 GB on the RX 7800 XT). The desktop, your apps and the
  model all share it. If it runs out, programs on the card can crash, and on 2026-10-25 that took
  the whole desktop (Hyprland) down.
- **RAM**: the PC's main memory (30 GiB here). Running out makes Linux kill programs or crawl.
- **Mixture of experts (MoE)**: a model split into many small "experts"; each word only uses a few.
  So a 35B model does the work of a ~3B one per word. The experts are the big part of the file and
  the easiest part to move off the graphics card into RAM, which makes MoE models good guests.
- **Layers**: a model is a stack of layers. Each layer has "dense" parts (used for every word) and,
  in an MoE model, experts.
- **Context**: how much of the conversation the model can keep in mind. Every word of context
  costs memory (the "KV cache"), so a longer context means more VRAM.
- **Compute buffer**: scratch space the engine needs while it works.
- **cgroup cap**: a fence Linux puts around a program: "this program may use at most X GB of RAM
  and no swap". If it goes over, only it gets stopped, not the desktop.

## How lodger decides
1. **Is the card free to use?** Only if you allow it (`--allow-gpu`). Then lodger watches every app
   for two seconds (see "Who needs the card" below). If one is busy on the card or still loading
   (free memory falling), it waits for things to settle, or runs on the processor instead.
2. **Measure what is free now.** VRAM free = card total minus used (`/sys/class/drm/card*/device`;
   the kernel's number: the card runtime's own number said ~700 MiB more than really was free).
   RAM free = `MemAvailable` from `/proc/meminfo`.
3. **Subtract margins.**
   - Keep at least 2 GiB of VRAM free for the desktop at all times (from `../CLAUDE.md`).
   - Add 10% of the estimate + 256 MiB, because estimates are never exact.
   - Keep at least 4 GiB of RAM (or 15%, whichever is bigger) for everything else.
4. **Ask what it costs.** Exact piece sizes come from the model file (`gguf` reader); context and
   scratch sizes from llama.cpp's own estimator (`llama-fit-params --fit-print`). File sizes let us
   try many placements instantly; the estimator knows each model design's details.
5. **Place the pieces, best first.**
   - Everything on the graphics card, with the longest context that fits? Done (fastest).
   - Otherwise: all dense parts + context on the card, and as many layers' experts as fit; the rest
     of the experts stay in RAM and the processor works on them. (llama.cpp's `--n-cpu-moe N`.)
     Experts are the right thing to move because each word only touches a few of them.
   - Still too big? Shorten the context step by step down to a minimum (4096 words).
   - Still too big? Put fewer whole layers on the card (only if at least a quarter fit; less isn't
     worth taking memory from your apps).
   - Nothing fits on the card? Run on the processor only, if RAM allows.
   - Nothing fits at all? **Refuse**, and say why in plain words.
6. **Check with llama.cpp.** For a real run on the card, lodger asks llama.cpp for its own estimate
   of the chosen placement on the card and adds the runtime overhead learned from earlier runs
   (`runs/calibration.jsonl`; 512 MiB until there's data). If that's more than planned, it plans
   smaller. (First check: llama.cpp said 11.0 GiB, the model really used 11.1 GiB.)
7. **Last look, then start.** Right before starting, if free VRAM dropped since planning (another
   app took some), it runs on the processor instead.
8. **Run it fenced in.** Own cgroup with a RAM cap and no swap, low CPU and disk priority (your apps
   first), fewer threads than the CPU has, crash dumps off. (On 2026-10-25 a crashing model kept
   15 GB of VRAM busy for ~5 s while Linux saved its crash dump; that is what killed the desktop.)
   On the processor only, the card's device files don't even exist for it (`PrivateDevices=yes`).
9. **Watch while it runs.** Ten times a second while loading (its big memory grabs happen then), four
   times a second after. Stop at once if free VRAM < 1.5 GiB or free RAM < 1.5 GiB; step aside
   (stop gracefully) if another app needs the card, or llama-router is (or will be) back.
10. **Learn.** Log planned vs really used, so the margins can shrink when the estimates prove good.

## Who needs the card (`lodger/neighbors.py`)
No app names. Linux reports, per app, how much of the card's memory it holds and how long the card's
3D/compute engine has worked for it (`/proc/<pid>/fdinfo`). An app needs the card when it:
- keeps the engine busy 25% of the time or more, for 3 seconds in a row (measured over 1-second
  windows: frames come in bursts), or
- takes 512 MiB or more of graphics memory within 5 seconds (it's loading something).
The idle desktop uses under 0.1%. Lodger's own model never counts. One check costs ~17 ms.

## Extra rules
- Stress tests run while the user is *away* (their choice, 2026-10-28): a crash then costs the least.
  The user also allowed starting games for them.
- `llama-router` (serves the Steam Deck) can load a 15 GB model at any moment, so lodger doesn't use
  the card while it runs, or while `llama-game-guard` has it paused for a game (that script now leaves
  `$XDG_RUNTIME_DIR/llama-game-guard.router-stopped` while it waits to start the router again).
- Always pin to the RX 7800 XT (`--device ROCm0`); llama.cpp also sees the processor's built-in
  graphics as ROCm1 and must never use it.

## Server: `lodger serve` (user approved 2026-10-28) - this PC only
Why: grow-back needs the conversation to survive moving the model between card and processor. With a
server, every request carries the whole conversation, so lodger can restart the engine anywhere and
nothing is lost.
- **All your models** (since 2026-10-28 14:00): `lodger serve` serves every model in
  `~/.config/lodger/models.toml` (folders to scan, names, per-model llama-server settings, `skip` with a
  reason; `LODGER_MODELS` points elsewhere for tests), one at a time. A request naming another model
  waits until nothing is being written, then lodger unloads the current one and loads that one (~5-20 s).
  New requests for the current model queue behind a waiting switch, so nobody waits forever. Each model
  gets its own context (decided at start, as if lodger's own loaded model weren't on the card) and plan.
  A model that can't be read or fails to start twice is marked and left out.
- **Front door**: `lodger serve [--allow-gpu]` listens on `127.0.0.1:8181` (only this PC, no key).
  Apps use the usual OpenAI-style API: `/v1/chat/completions` (streaming or not), `/v1/models`, `/health`.
  Lodger's own: `GET /lodger/status` (where the model is, why, recent events), `POST /lodger/move` (move
  off the card now; it grows back later as usual). `lodger ask "question"` is a small client for it.
- **Engine**: our build's `llama-server` on a random private port (127.0.0.1), one at a time, in the same
  fenced unit with the same plan and watchdog as `lodger run`.
- **Steps aside**: when the watchdog stops the card engine (an app needs the card, memory low, router
  coming), a processor engine starts at once (~5 s). An answer being written continues where it stopped:
  lodger resends the conversation with the partial answer as the last assistant message, and
  llama-server continues it. The app sees one answer (plus an SSE comment line saying it moved).
- **Grows back**: when on the processor, no answer is being written, and the card has been quiet for
  60 s, lodger makes a fresh plan; if it fits, it loads the card engine *while the processor engine keeps
  answering*, then swaps. If the model has to step aside again within 10 min of growing back, the quiet
  wait doubles (60, 120, 240 ... up to 30 min), so a game sitting in its menu doesn't make it bounce.
- **Context is automatic** (`plan.CTX_CAP`, `CTX_SHARE`): as long as the model can take, up to 128K, as
  long as it costs at most a quarter of the memory the model may use there. It is decided once per session
  (the shorter of the card's and the processor's), so a conversation that fits on the card still fits
  after a move. `--ctx N` overrides. /v1/models reports it (`context_window`).
- **Other AI servers** (`--pause SERVICE`, e.g. `llama-router`): while that service is idle (holds no
  graphics memory), lodger stops it before taking the card and starts it again when the model leaves
  the card (step aside, sleep, exit; `lodger unpause` after a crash). While it has a model loaded,
  lodger stays on the processor. Without `--pause`, a running router keeps lodger off the card.
- **pi**: on start (port 8181 only) lodger writes a `lodger` provider listing every model into
  `~/.pi/agent/models.json` (other providers kept) and makes it pi's default in `settings.json` unless
  another provider was chosen there. Plain `pi` then lists them all; `/model` or `--model lodger/NAME`
  switches. `lodger pi` refreshes that list and opens pi.
  A move in the middle of a tool call ends that answer with an error (pi retries); plain text continues.
- The safety exception (lodger may start our build's `llama-server`, local only) is written in
  `../CLAUDE.md` and the Claude hook's comment.
The Steam Deck is **not** part of this. Since 2026-10-28 the PC doesn't serve the Deck at all
(llama-router and llama-game-guard disabled, Hermes on the Deck uses Grok then Codex), so nothing else on
this PC loads models onto the card and lodger uses it freely.

## Gaps fixed on 2026-10-28 (were "known gaps" in version 1)
- **Steam's start-up looked like a game.** Game detection by process names (`reaper`, `wineserver`,
  `gamescope`) is gone; "who needs the card" above replaces it and works for any app.
- **Loading race** (an app grabbing memory while the model loads): settle check before planning, last
  look before starting, 0.1 s watchdog while loading, and any app taking 512 MiB in 5 s makes the model
  step aside. Narrower, not gone: two programs can still ask for big blocks within the same tenth of a
  second. The fenced, dump-free model then fails cleanly, which is the case to verify in the stress test.
- **Router coming back after a game** while the model is on the card: the game guard's marker file.
- **Card scratch space was guessed**: now llama.cpp's own estimate + learned overhead.

## Later (promise 2 and 3)
- Shrink in place without a restart (grow-back is done, but every move restarts the engine).
- `lodger serve --think`: continuing a half-written *reasoning* part after a move is untested.
- Save the stress test's answers, to check the seam of a moved answer by eye on the card too (checked on
  the processor: counting 1-60 across a kill came out clean).
- Shrink the RAM fence while another app is busy (with the model on the processor next to Deadlock,
  Linux compressed ~2 GB of other apps' idle memory).
- Lower graphics-card priority for the model's work, so other apps' frames come first.
- Keep the user's most-used experts on the card: b11936 has `--moe-cache-mib N` ("GPU cache for the MoE
  experts kept in the CPU"), found 2026-10-28; try it in the plan's spare card memory.
- Adaptive desktop reserve: the 2 GiB kept free is a fixed number (this PC's house rule, the minimum here).
  Measure what the desktop really uses and how much it jumps, and keep that plus headroom (matters on
  cards with 4-8 GB).
- Two-drive striping for any model too big for RAM (from ds41-local).
- Auto-setup: a one-time benchmark that writes the PC's profile.

## Layout
- `bin/lodger` - the one command: `lodger plan|run|measure|stress|serve|ask|stress-serve` (`--help` on each).
- `systemd/models.example.toml` - the model list used here (live copy: `~/.config/lodger/models.toml`).
- `lodger/` - the code: `catalog.py` (the model list), `launch.py` (plan/run, watchdog), `plan.py` (placement), `neighbors.py` (who
  needs the card), `sysmem.py` (memory readers), `model.py` (model file reader), `measure.py`, `stress.py`,
  `serve.py` (serve, ask, step aside, grow back), `servetest.py` (stress-serve).
- `systemd/lodger.service` - starts `lodger serve --allow-gpu --idle 10` (all models) with the desktop session;
  linked into `~/.config/systemd/user/` and enabled.
- `tools/vram-eater.hip` - test program that takes graphics memory like a game loading; built into
  `build/` (not for git).
- `runs/` - logs from measurements and runs, `calibration.jsonl`, `probe-cache/` (local only, not for git).

## Status
(newest last)
- 2026-10-28 09:51-10:01 **baseline measurement** (`runs/2026-10-28-0951-measure-baseline/`), 10 min, Discord
  (Vesktop), Claude, 1Password open; Chrome was not open. The desktop uses **0.8-1.1 GiB** of the card's 16 GiB
  (Hyprland up to 772 MiB, the wallpaper engine `owed` ~250 MiB, Claude up to 110 MiB). Discord draws on the
  built-in graphics (67 MiB there), as set up on 2026-10-25. RAM: 17.9 GiB was the lowest available (my
  processor-only test runs ran during it; the desktop alone leaves ~20 GiB).
- 2026-10-28 launcher v1 tested on the processor (Qwen3.5-9B, the only suitable model on disk): correct
  answer, **9.3 words/s writing, 80 words/s reading**, RAM inside its fence. Emergency stop and Ctrl+C both
  stop the model at once and leave nothing behind.
- 2026-10-28 10:01 and 10:03 **GPU stress test** (user away, by their choice; router stopped and restarted,
  ~1 min each): both phases PASS twice, no graphics-card errors in the kernel log, no crashes, Hyprland fine.
  - Answer on the card: Qwen3.5-9B fully on the card, **74 words/s writing, 229 reading**. Used 6.0 GiB,
    planned 6.6 GiB (estimate ~10% high: safe side).
  - Memory eater: the model was stopped when free VRAM hit 4.0 GiB (test line), the eater then took 12 GiB.
  - Found: HIP's own "free memory" number (`hipMemGetInfo`) said ~700 MiB more than the kernel's
    (sysfs). The first run's eater meant to stop at 3 GiB free but the card went to 2.5 GiB. The eater now
    reads sysfs; second run's lowest was 3.3 GiB. Lesson: always use sysfs for the 2 GiB line.
- Next: download a mid-size MoE (proposed: Qwen3.6-35B-A3B UD-IQ4_XS, 17.7 GB, needs the user's OK), run
  the planner on it (it's bigger than the card, so it exercises expert placement), stress test it, then the
  `--game` phase when the user can start a game. Decide the server question (see "Not decided yet").
- 2026-10-28 10:13 **model downloaded** (user approved): `~/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
  (unsloth, 17.7 GB, sha256 checked, `.ok` file). 35B mixture of experts, 8 of 256 experts per word (~3B work).
- **Plan for it** (desktop alone, 15.2 GiB free): everything on the card except the experts of 17 of 40 layers
  (those 6.4 GiB stay in RAM), 16K context, planned 11.75 GiB on the card, really used **11.3 GiB** (4% under
  the estimate; the margin was not needed). Lowest free VRAM ~4 GiB.
- Speed: **52.8 words/s writing, 144 reading** on the card; **16.7 / 59** on the processor only (faster than
  the 9B on the processor, because each word only uses ~3B of the 35B).
- **Stress test 35B** (user away; router stopped ~4 min per run, then restarted; the user allowed launching
  games): phases 1 and 2 PASS (eater phase: the model leaves 4 GiB extra so the eater can push it).
  Game phase, first try: not a real result. Steam's own start-up runs Wine (`wineserver`), which the
  watchdog took for a game, and TF2 only ran ~10 s. Test fixed (start Steam first and let it settle, wait
  until the game really draws). Then:
  - **TF2** PASS: model stepped aside the moment the game started (`reaper`), game drew (767 MiB), model
    answered on the processor during the game (15.9 words/s), game kept running, closed cleanly.
  - **Deadlock** PASS: same, game used 2.0 GiB in its menu, model answered on the processor in 24 s.
    No graphics-card errors in the kernel log, no crashes, Hyprland fine, Steam closed again afterwards.
  - Seen: with the model on the processor next to Deadlock, Linux compressed ~2 GB of other apps' idle
    memory into zram (compressed RAM "swap", swappiness 150 here; 3.2 GB was there before). Harmless,
    but it shows the RAM fence (which counts cached model file pages) is generous; promise 2 should
    shrink it while a game runs.
  - Seen: a Docker container running `llama-server` (another project) holds ~1.5 GB, swapped out.
- Known gap added: `wineserver` also runs when Steam starts or a Proton prefix updates, so "game started"
  can fire without a game. Only too careful, never unsafe; refine later (e.g. look for a game window).
- Next: decide the server question (to serve the Steam Deck through good-neighbor); a match in Deadlock
  (more VRAM than its menu); promise 2's "grow back after the game" (restart on the card when the game ends).
- 2026-10-28 ~10:40 **renamed to lodger** (user's pick) and the version-1 gaps fixed (see "Gaps fixed").
  Old status lines above still say `gn-measure`/`gn-launch`/`gn-stress` and `--attended`/game names:
  those are now `lodger measure|plan|run|stress`. Tested on the processor: 35B through `lodger run`
  answered correctly at 16.8 words/s, falling back to the processor because the router was on.
  llama.cpp's own card estimate checked: 11.0 GiB for the 35B placement vs 11.1 GiB really used.
- 2026-10-28 11:24-11:35 **GPU stress test of lodger** (user away; router stopped ~5 min per run, then
  restarted). From now on: assume the user is away unless they say they're at the PC (their rule).
  - TF2 run, all five phases PASS: (1) answer on the card; (2) memory grab: stepped aside; (3) Steam
    starting: no false alarm, kept writing; (4) game: stepped aside 19 s after launch, when the game
    actually loaded 635 MiB; asked again while TF2 sat in its menu, it answered *on the card* (TF2
    barely worked the card: sharing, as intended); (5) model and game started together: model got the
    card first, then stepped aside when the game loaded. Game kept running every time.
  - Deadlock run, phases 3-5 PASS: stepped aside 13 s after launch; asked again during the game, it
    chose the processor by itself (the game had just taken 551 MiB); race: same as TF2.
  - Lowest free graphics memory in both runs: 3.0 GiB. No crashes, Hyprland fine, Steam closed after.
  - Kernel log: `REG_WAIT timeout ... dcn32_program_compbuf_size` twice (when the games closed). That's
    the display engine (monitor output), not the model: it also appeared at 09:51 and 09:54 today with
    nothing on the card, and 18 times in the previous boot. Not counted as a failure.
  - Learning works: llama.cpp estimate 11.57 GiB, really used 11.69 GiB (+113 MiB), learned overhead now
    256 MiB (the floor) instead of 512, so the plan is 11.85 GiB instead of ~12.1.
  - Label quirk: the test's memory grabber showed up as "com.anthropic.Claude", because it was started
    from inside the Claude app and Linux groups it with that app. Detection was right, only the name.
- Next: the server question (serve the Steam Deck through lodger); "grow back" after stepping aside.
- 2026-10-28 ~11:45 user decided: lodger serves **this PC only** (the Deck stays on llama-router), and may
  start our build's `llama-server` on 127.0.0.1 (exception written into `../CLAUDE.md` and the hook).
- 2026-10-28 11:58 **`lodger serve` on the processor** (9B): streaming and plain answers fine. Killed the
  engine in the middle of "count from 1 to 60": a new one was up in 8 s and the count finished with no
  repeats or gaps. Found and fixed: a Ctrl-C to the whole terminal group killed `systemd-run`'s client
  first, so lodger thought the engine was gone and left it running (cleaned up). The client now runs in
  its own session and shutdown always stops the units; checked with TERM and Ctrl-C to the group.
- 2026-10-28 12:05-12:21 **`lodger stress-serve` on the card** (35B, TF2; user away; router stopped ~17 min,
  then restarted), all five phases PASS, no graphics-card errors, no crashes, Hyprland fine:
  1. on the card in 8 s (41 layers, 16K context), answered;
  2. asked to move mid-answer: cut at 11.4 s, finished on the processor (3906 characters);
  3. grew back onto the card after 88 s (quiet wait 120 s because of the move in 2);
  4. memory grab mid-answer (1.4 GiB in 5 s): stepped aside at 11.9 s, answer finished on the processor,
     grew back 212 s later (quiet wait 240 s);
  5. TF2: stepped aside when the game loaded 636 MiB; a question during the game was answered on the
     processor in 11 s; grew back 445 s after (quiet wait 480 s). Not covered: the answer had already
     finished before Steam got the game going, so a game interrupting an answer mid-sentence was
     not seen here (2 and 4 cover mid-answer moves).
  Lowest free graphics memory 2.18 GiB (during 4: the test's eater stops at 2 GiB free).
  A first run was stopped early: the test itself waited for the whole answer before asking for the move.
- 2026-10-28 12:37 **in git and starting at login** (user's request). `lodger serve --idle MIN`: the model
  loads on the first request and unloads after MIN minutes unused, so lodger holds no memory while you
  don't use it (tested: asleep at start, ~5 s to wake for the 9B, unloads, wakes again). The service runs
  the 35B with `--idle 10`; first question after login: 16 s (planning 4 s, loading 9 s, answer).
  While llama-router runs (always, for the Deck) lodger stays on the processor (~16 words/s), by design.

- 2026-10-28 13:00 user asked for: automatic context, a fix for "stuck on the processor", and pi as the
  way to talk to models. Done (see Server): 35B gets 128K context on this PC (17 of 40 expert layers on
  the card instead of 23 at 16K; 48-51 words/s), the 9B on the processor 64K.
  Tested on the card with llama-router running: router paused when the model took the card, started
  again when it unloaded, stepped aside (memory grab mid-answer: answer finished on the processor) or
  was moved, paused again on grow-back, started again on exit. pi through lodger read a file with its
  tool. No graphics-card errors, no crashes.
  Not yet in the login service: `--pause llama-router` changes what the Deck sees (its server is off
  while the PC uses the card), so it waits for the user's OK. Until then the service stays on the
  processor while the router runs.
  Found: a background job started from a script ignores Ctrl-C, so `lodger serve` now sets its own
  Ctrl-C handler.
- Project home: lodger stays its own project, separate from 0xSero's Local AI (different job: sharing the
  PC, not picking and launching recipes). It speaks the standard API, so any of those tools can use it.
- 2026-10-28 13:15 user: "get local AI off the Deck" (the router conflicted with lodger). Done:
  llama-router and llama-game-guard disabled and stopped on the PC (files kept); on the Deck, Hermes'
  `local` provider and its fallback to the PC's Qwen 9B removed (backup `config.yaml.bak-2026-10-28`,
  checked that nothing else changed), Hermes restarted and running (Grok first, then Codex). The login
  service now loads onto the card: 35B, 128K context, 50 words/s.
- 2026-10-28 13:30-14:10 **lodger runs all local models** (user: "I want to see all my local AIs through
  lodger"; 0xSero's Local AI should not run models itself). Done:
  - 0xSero's Local AI: its LFM2.5 2.6B containers stopped with its own `omarchy-local-ai stop` (its code
    untouched); the model file is now served by lodger.
  - `~/.config/lodger/models.toml`: 8 models, settings from the old router presets (sampling, thinking
    off for the 9B and the 35B), q8_0 KV cache + flash attention for all (A/B on the 35B: 52-53 words/s
    with q8_0 vs 50-51 with f16, and room for 4 more expert layers on the card).
  - All 6 servable models loaded and answered on the card, no graphics-card errors or crashes:
    LFM 2.6B 165 words/s (128K), Qwen3.5-9B 65 (128K), Saluki 27B 35 (64K, all on the card),
    Qwen3.6-35B 52 (128K, 21 of 40 expert layers on the card), Swift 27B 12 and Heretic 27B 8.5 (64K,
    53 and 47 of 65 layers on the card: these dense 13 GB models don't fit beside 2 GiB kept free).
  - Not served: Bonsai 27B (PQ2_0, a file type this llama.cpp and gguf-py can't read; needs the Prism
    fork) and DeepSeek V4.1 (248 GB + Engram file; needs ds41-local's own setup).
  - Fixed on the way: the planner gave up on the card when llama.cpp's estimate was higher (models with
    an MTP layer), because it compared against its shrunk working budget instead of the real limit.
  - Plain `pi` lists all 6 (settings backup `~/.pi/agent/settings.json.bak-2026-10-28`); tested reading a
    file with the 35B and listing files with the 9B (lodger switched by itself).
  - Later: vision (the projector files are there, not planned yet); MTP / n-gram speculation for the
    27Bs (the old router got ~58 vs 32 words/s on Heretic with it; llama-fit-params can't size it yet).
