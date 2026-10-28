"""How much graphics memory lodger always leaves free for your desktop (the compositor, browsers, the
apps you have open), beyond what they use right now.

It is the larger of
- a floor: desktop_reserve_mib in config.toml, else 1/8 of the card, between 1 and 2 GiB, and
- what the desktop showed it needs: while `lodger serve` runs and no app is busy on the card, it notes how
  much memory everything but lodger's model uses, minute by minute (kept 7 days, per card). The reserve is
  how far that use rises within 10 minutes, at worst 1 time in 20, plus 256 MiB. It counts once there is at
  least an hour of notes.
Apps that really need the card (a game loading) aren't covered by this: lodger steps aside for those."""
import json
import os
import time

from . import config

MiB, GiB = 1 << 20, 1 << 30
WINDOW = 10            # minutes a rise is measured over
KEEP = 7 * 24 * 60     # minutes of notes kept
MIN_NOTES = 60         # minutes of notes before the measured part counts
HEADROOM = 256 * MiB
SHARE = 0.95           # rises at or below this share of all windows are covered

_notes = {}            # pci -> {minute: [lowest, highest]}


def floor(total):
    cfg = config.get().get("desktop_reserve_mib")
    if cfg:
        return int(cfg) * MiB
    step = 256 * MiB                                   # a "16 GB" card has 15.98 GiB: round up
    return min(2 * GiB, max(GiB, -(-(total or 0) // 8 // step) * step))


def _path(pci):
    return os.path.join(config.runs_dir(), f"desktop-{pci}.json")


def _load(pci):
    if pci not in _notes:
        try:
            with open(_path(pci)) as f:
                _notes[pci] = {int(k): v for k, v in json.load(f).items()}
        except (OSError, ValueError, AttributeError):
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
    return r + HEADROOM, (f"your desktop's graphics memory use rose by up to {r / GiB:.1f} GiB within "
                          f"{WINDOW} minutes lately, so lodger keeps more than its usual {f / GiB:.1f} GiB")


def kill_line(keep):
    """Stop the model at once below this much free graphics memory: 3/4 of the reserve (1.5 GiB for 2 GiB),
    so a plan that leaves the reserve free never trips it. LODGER_KILL_VRAM_FREE_MIB can only raise it."""
    return max(int(os.environ.get("LODGER_KILL_VRAM_FREE_MIB", 0)) * MiB, keep * 3 // 4)
