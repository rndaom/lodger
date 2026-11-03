"""Does another app need the graphics card right now? Works for any app (a game, a browser drawing 3D,
another AI tool), with no list of names: it watches what each app does on the card.

An app "needs the card" when, on the card the model uses, it either
- keeps the card's 3D or compute engine busy BUSY_FRAC of the time for BUSY_SECONDS in a row, or
- takes GROWTH more graphics memory within GROWTH_SECONDS (something is loading; on built-in graphics,
  which keep apps' memory in shared RAM, that counts too).
The model's own processes (systemd unit lodger-*) never count. Times are monotonic: the wall clock can jump.

Apps lodger can't look into: the kernel shows /proc/<pid>/fdinfo only to the app's own user, so another
user's game, a compositor or Xorg running as root, or a system container using the card show no per-app
numbers. Where the driver gives the whole card's numbers (amdgpu: gpu_busy_percent, memory used), lodger
compares them with the sum of every app it can see, its own model included: what is left over is those apps,
judged by the same rules (busy only while lodger's own model leaves the card mostly idle, as the two
numbers aren't measured the same way). NVIDIA's NVML lists every user's apps, so it needs none of this."""
import collections
import os
import time

from . import sysmem
from .gpu import fdinfo

MiB = sysmem.MiB
BUSY_FRAC = float(os.environ.get("LODGER_BUSY_FRAC", 0.25))
BUSY_SECONDS = 3.0
GROWTH = 512 * MiB
GROWTH_SECONDS = 5.0
OURS = sysmem.OURS
UNSEEN = "an app lodger can't look into (another user's, or the system's)"
OURS_QUIET = 0.10      # the whole card's busy is compared only while lodger's own model keeps it under this


def coverage(card):
    """What lodger can see of the apps on `card`, for lodger doctor: (sees every app, plain words)."""
    if card is None:
        return False, "no graphics card"
    if card.sees_every_app:
        return True, "the driver lists every app on the card, any user's"
    wide = []
    if card.has_busy():
        wide.append("how busy the whole card is")
    if not card.integrated and card.known and card.memory() is not None:
        wide.append("the whole card's memory use")
    return False, ("per-app use comes from /proc/<pid>/fdinfo, which the kernel shows only for your own apps; "
                   "apps of other users or the system (a root Xorg, gamescope, a system container) are seen "
                   + ("only through " + " and ".join(wide) if wide else "not at all on this card"))


class Neighbors:
    def __init__(self, card=None, ours=None):
        self.card = card or sysmem.dgpu()
        self.pci = self.card.pci if self.card else None
        self.ours = ours or (lambda: ())     # pids of lodger's own model (when it runs without a lodger-* cgroup)
        self.hist = collections.deque()      # (time, {app: (busy_ns, vram, cycles)}, the whole card: see _whole)
        self.busy_since = {}                 # app -> time it went over BUSY_FRAC
        self.gpu_pids, self.scanned = {}, 0  # pid -> app, for processes that hold the card (rescanned every 2 s)
        self.shared = bool(getattr(self.card, "integrated", False))
        self.whole = None                    # the whole card at the last sample

    def _sample(self):
        now = time.monotonic()
        full = now - self.scanned > 2
        if full:
            self.scanned = now
        self.whole = None
        if not self.card:
            return {}
        # always in pid order, as a full scan is: a client shared by two processes then always counts for the
        # same one, and never looks like one app suddenly taking another's memory
        pids = sysmem.pids() if full else sorted(self.gpu_pids, key=int)
        use = self.card.usage(pids)
        ours = set(self.ours())
        seen = {}                            # every app on the card, lodger's own model too (as OURS)
        for pid in pids:
            if pid not in use:
                self.gpu_pids.pop(pid, None)
                continue
            name = self.gpu_pids.get(pid) or sysmem.app_name(pid)
            self.gpu_pids[pid] = name
            fdinfo.add(seen.setdefault(OURS if pid in ours else name, fdinfo.empty()), use[pid])
        self.whole = self._whole(full, use, ours, seen)
        # built-in graphics keep apps' memory in shared RAM (GTT): a game loading there shows up in that
        return {app: (r["busy_ns"], r["vram"] + (r["gtt"] if self.shared else 0), r["cycles"])
                for app, r in seen.items() if app != OURS}

    def _whole(self, full, use, ours, seen):
        """The whole card's numbers next to the sum of what lodger sees: {full, busy (0..1), used (bytes),
        seen: {app: (busy_ns, vram)}}, or None where the driver gives none (or lists every app itself)."""
        whole = getattr(self.card, "whole", None)
        w = whole() if whole else None
        if not w or (w["busy"] is None and w["used"] is None):
            return None
        # lodger's model holds the card yet its fdinfo shows nothing (a kernel without DRM usage stats): the
        # card's numbers would count lodger's own work as a stranger's
        if any(pid not in use and fdinfo.drm_fds(pid, self.pci) for pid in ours):
            return None
        return {"full": full, "busy": w["busy"], "used": w["used"],
                "seen": {app: (r["busy_ns"], r["vram"]) for app, r in seen.items()}}

    def update(self):
        """Take a sample; return the reason another app needs the card, or None."""
        now = time.monotonic()
        cur = self._sample()
        self.hist.append((now, cur, self.whole))
        while self.hist and now - self.hist[0][0] > max(GROWTH_SECONDS, BUSY_SECONDS) + 1:
            self.hist.popleft()
        if len(self.hist) < 2:
            return None
        # busy share over the last second or more (frames come in bursts; shorter windows are too jumpy)
        ref = next(((t, s, w) for t, s, w in reversed(self.hist) if now - t >= 1.0), None)
        if ref is None:
            return None
        t_prev, prev, _ = ref
        reason = None
        for app, (b, v, cyc) in cur.items():
            # an app seen for the first time starts counting from its next sample
            b0, _, cyc0 = prev.get(app, (b, 0, cyc))
            frac = (b - b0) / max(1e-3, now - t_prev) / 1e9
            for e, (c, tot) in cyc.items():                  # drivers that count cycles (xe)
                c0, tot0 = cyc0.get(e, (c, tot))
                if tot > tot0:
                    frac = max(frac, (c - c0) / (tot - tot0))
            if frac >= BUSY_FRAC:
                self.busy_since.setdefault(app, t_prev)
                if now - self.busy_since[app] >= BUSY_SECONDS and not reason:
                    reason = f"{app} is keeping the graphics card busy ({frac:.0%})"
            else:
                self.busy_since.pop(app, None)
            old = next(((t, s) for t, s, _ in self.hist if now - t <= GROWTH_SECONDS), None)
            if old and not reason:
                grew = v - old[1].get(app, (0, 0, {}))[1]
                if grew >= GROWTH:
                    reason = f"{app} took {grew // MiB} MiB of graphics memory in {now - old[0]:.0f} s"
        return reason or self._unseen(now, ref)

    def _unseen(self, now, ref):
        """Apps lodger can't look into, from what the whole card does or holds beyond every app lodger sees.
        Only said after a full scan (an app that just started is found by one); a sign between scans brings
        the next full scan forward."""
        w = self.whole
        if not w:
            self.busy_since.pop(UNSEEN, None)
            return None
        t_prev, _, wp = ref
        why = None
        busy = None
        # an app that came or went in between makes its share of the window unknown: no judgement this round
        if wp and set(wp["seen"]) == set(w["seen"]):
            dt = max(1e-3, now - t_prev)
            cards = [x["busy"] for t, _, x in self.hist if x and t_prev < t <= now and x["busy"] is not None]
            seen = sum(max(0, b - wp["seen"][app][0]) for app, (b, _) in w["seen"].items()) / dt / 1e9
            ours = (w["seen"].get(OURS, (0, 0))[0] - wp["seen"].get(OURS, (0, 0))[0]) / dt / 1e9
            if cards and ours < OURS_QUIET:
                busy = sum(cards) / len(cards) - seen
        if busy is not None and busy >= BUSY_FRAC:
            self.busy_since.setdefault(UNSEEN, t_prev)
            if now - self.busy_since[UNSEEN] >= BUSY_SECONDS:
                why = f"{UNSEEN} is keeping the graphics card busy ({busy:.0%})"
        else:
            self.busy_since.pop(UNSEEN, None)
        if not why and w["used"] is not None:
            def rest(x):
                return x["used"] - sum(v for _, v in x["seen"].values())
            known = [(t, x) for t, _, x in self.hist if x and x["used"] is not None]
            # still there after a second: memory an app freed shows in the card's use a moment after its
            # fdinfo is gone, which would look like a stranger taking it
            recent = [rest(x) for t, x in known if now - t <= 1.5]
            old = next(((t, x) for t, x in known if now - t <= GROWTH_SECONDS), None)
            if old and len(recent) >= 2 and now - old[0] > 1.5:
                grew = min(recent) - rest(old[1])
                if grew >= GROWTH:
                    why = f"{UNSEEN} took {grew // MiB} MiB of graphics memory in {now - old[0]:.0f} s"
        if why and not w["full"]:
            self.scanned = 0
            return None
        return why

    def watch(self, seconds, interval=0.25):
        """Sample for `seconds`; return the first reason seen, or None."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            r = self.update()
            if r:
                return r
            time.sleep(interval)
        return None
