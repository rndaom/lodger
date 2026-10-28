"""Does another app need the graphics card right now? Works for any app (a game, a browser drawing 3D,
another AI tool), with no list of names: it watches what each app does on the card.

An app "needs the card" when, on the card the model uses, it either
- keeps the card's 3D or compute engine busy BUSY_FRAC of the time for BUSY_SECONDS in a row, or
- takes GROWTH more graphics memory within GROWTH_SECONDS (something is loading).
The model's own processes (systemd unit lodger-*) never count."""
import collections
import os
import time

from . import sysmem

MiB = sysmem.MiB
BUSY_FRAC = float(os.environ.get("LODGER_BUSY_FRAC", 0.25))
BUSY_SECONDS = 3.0
GROWTH = 512 * MiB
GROWTH_SECONDS = 5.0
OURS = "lodger (model)"


class Neighbors:
    def __init__(self, card=None):
        self.card = card or sysmem.dgpu()
        self.pci = self.card["pci"] if self.card else None
        self.hist = collections.deque()      # (time, {app: (busy_ns, vram)})
        self.busy_since = {}                 # app -> time it went over BUSY_FRAC
        self.gpu_pids, self.scanned = {}, 0  # pid -> app, for processes that hold the card (rescanned every 2 s)

    def _sample(self):
        now = time.time()
        full = now - self.scanned > 2
        if full:
            self.scanned = now
        apps = {}
        for pid in (sysmem.pids() if full else list(self.gpu_pids)):
            clients = sysmem.gpu_clients(pid)
            if not clients:
                self.gpu_pids.pop(pid, None)
                continue
            name = self.gpu_pids.get(pid) or sysmem.app_name(pid)
            self.gpu_pids[pid] = name
            if name == OURS:
                continue
            b, v = apps.get(name, (0, 0))
            for (pci, _), rec in clients.items():
                if pci == self.pci:
                    b += rec["busy_ns"]
                    v += rec["vram"]
            apps[name] = (b, v)
        return apps

    def update(self):
        """Take a sample; return the reason another app needs the card, or None."""
        now = time.time()
        cur = self._sample()
        self.hist.append((now, cur))
        while self.hist and now - self.hist[0][0] > max(GROWTH_SECONDS, BUSY_SECONDS) + 1:
            self.hist.popleft()
        if len(self.hist) < 2:
            return None
        # busy share over the last second or more (frames come in bursts; shorter windows are too jumpy)
        ref = next(((t, s) for t, s in reversed(self.hist) if now - t >= 1.0), None)
        if ref is None:
            return None
        t_prev, prev = ref
        reason = None
        for app, (b, v) in cur.items():
            # an app seen for the first time starts counting from its next sample
            frac = (b - prev.get(app, (b, 0))[0]) / max(1e-3, now - t_prev) / 1e9
            if frac >= BUSY_FRAC:
                self.busy_since.setdefault(app, t_prev)
                if now - self.busy_since[app] >= BUSY_SECONDS and not reason:
                    reason = f"{app} is keeping the graphics card busy ({frac:.0%})"
            else:
                self.busy_since.pop(app, None)
            old = next(((t, s) for t, s in self.hist if now - t <= GROWTH_SECONDS), None)
            if old and not reason:
                grew = v - old[1].get(app, (0, 0))[1]
                if grew >= GROWTH:
                    reason = f"{app} took {grew // MiB} MiB of graphics memory in {now - old[0]:.0f} s"
        return reason

    def watch(self, seconds, interval=0.25):
        """Sample for `seconds`; return the first reason seen, or None."""
        end = time.time() + seconds
        while time.time() < end:
            r = self.update()
            if r:
                return r
            time.sleep(interval)
        return None
