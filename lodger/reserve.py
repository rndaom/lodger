"""How much graphics memory lodger always leaves free for your desktop (the compositor, browsers, the
apps you have open), beyond what they use right now.

It is the larger of
- a floor: desktop_reserve_mib in config.toml, else 1/8 of the card, between 1 and 2 GiB, and
- what the desktop showed it needs: while `lodger serve` runs and no app is busy on the card, it notes how
  much memory everything but lodger's model uses, minute by minute (kept 7 days, per card). The reserve is
  how far that use rises within 10 minutes, at worst 1 time in 20, plus 256 MiB. It counts once there is at
  least an hour of notes, and at most a quarter of the card (MAX_SHARE).
Apps that really need the card (a game loading) aren't covered by this: lodger steps aside for those. But an
app that holds a lot while idle (a game in its menu, a test's memory grabber) looks like the desktop in the
notes: the quarter-of-the-card limit keeps such notes from taking the card away for a week."""
import json
import math
import os
import time

from . import config

MiB, GiB = 1 << 20, 1 << 30
WINDOW = 10            # minutes a rise is measured over
KEEP = 7 * 24 * 60     # minutes of notes kept
MIN_NOTES = 60         # minutes of notes before the measured part counts
HEADROOM = 256 * MiB
SHARE = 0.95           # rises at or below this share of all windows are covered
MAX_SHARE = 0.25       # the measured part never asks for more than this share of the card

_notes = {}            # pci -> {minute: [lowest, highest]}


def floor(total):
    """desktop_reserve_mib from config.toml can raise the reserve above the one for the card's size, never
    lower it: a smaller number (or 0, or a negative one) leaves the reserve for the card's size."""
    step = 256 * MiB                                   # a "16 GB" card has 15.98 GiB: round up
    auto = min(2 * GiB, max(GiB, -(-(total or 0) // 8 // step) * step))
    cfg = config.get().get("desktop_reserve_mib")
    if cfg is None:
        return auto
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)):
        raise ValueError(f"desktop_reserve_mib in config.toml must be a number of MiB, like 2048 (it is {cfg!r})")
    return max(auto, int(cfg * MiB))


def _path(pci):
    return os.path.join(config.runs_dir(), f"desktop-{pci}.json")


def _valid(v):
    # json reads Infinity and NaN too: an endless reserve would keep the model off the card for good
    return (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                        and math.isfinite(x) and x >= 0 for x in v))


def _load(pci):
    """The notes for a card; a damaged or strange file counts as no notes (the floor still holds)."""
    if pci not in _notes:
        try:
            with open(_path(pci)) as f:
                d = json.load(f)
            notes = {int(k): v for k, v in d.items()}
            if not all(_valid(v) for v in notes.values()):
                raise ValueError("not [lowest, highest] pairs")
            _notes[pci] = notes
        except (OSError, ValueError, AttributeError, TypeError):
            _notes[pci] = {}
    return _notes[pci]


def note(pci, used, now=None):
    """Remember that everything but lodger's model uses `used` bytes on the card now. Saved once a minute."""
    minute = int((now or time.time()) // 60)
    notes = _load(pci)
    new_minute = minute not in notes
    lo, hi = notes.get(minute, [used, used])
    notes[minute] = [min(lo, used), max(hi, used)]
    if new_minute:
        for m in [m for m in notes if m < minute - KEEP]:
            del notes[m]
        try:
            os.makedirs(os.path.dirname(_path(pci)), exist_ok=True)
            with open(_path(pci) + ".tmp", "w") as f:
                json.dump(notes, f)
            os.replace(_path(pci) + ".tmp", _path(pci))
        except OSError:
            pass


def rise(notes):
    """How far use rose within WINDOW minutes, at worst for SHARE of the windows; None without enough notes."""
    if len(notes) < MIN_NOTES:
        return None
    mins = sorted(notes)
    rises = []
    for i, m in enumerate(mins):
        top = max(notes[k][1] for k in mins[i:i + WINDOW + 1] if k - m <= WINDOW)
        rises.append(max(0, top - notes[m][0]))
    rises.sort()
    return rises[int(SHARE * (len(rises) - 1))]


def reserve(pci, total):
    """(bytes to keep free, plain words about where the number came from)."""
    f = floor(total)
    r = rise(_load(pci)) if pci else None
    if r is None or r + HEADROOM <= f:
        return f, None
    keep = int(r + HEADROOM)
    if total and keep > total * MAX_SHARE:
        keep = max(f, int(total * MAX_SHARE))
        if keep <= f:
            return f, None
        return keep, (f"your desktop's graphics memory use rose by up to {r / GiB:.1f} GiB within {WINDOW} minutes "
                      f"lately (likely an app holding memory while idle), so lodger keeps {keep / GiB:.1f} GiB, "
                      f"a quarter of the card, instead of its usual {f / GiB:.1f} GiB")
    return keep, (f"your desktop's graphics memory use rose by up to {r / GiB:.1f} GiB within "
                  f"{WINDOW} minutes lately, so lodger keeps more than its usual {f / GiB:.1f} GiB")


def kill_line(keep):
    """Stop the model at once below this much free graphics memory: 3/4 of the reserve (1.5 GiB for 2 GiB),
    so a plan that leaves the reserve free never trips it. LODGER_KILL_VRAM_FREE_MIB can only raise it."""
    return max(int(os.environ.get("LODGER_KILL_VRAM_FREE_MIB", 0)) * MiB, keep * 3 // 4)
