"""lodger stress-serve: check that `lodger serve` steps aside and grows back without crashing anything.

  lodger stress-serve MODEL [--game-appid 440] [--port 8182]

With extras.llama_router on: stop llama-router first (systemctl --user stop llama-router; start it again after).
Phases (each checked for graphics-card errors in the kernel log, crashed programs and a live desktop):
  1. the model starts on the card and answers
  2. asked to move off the card in the middle of an answer: the answer finishes on the processor
  3. grows back onto the card once the card has been quiet
  4. another program takes graphics memory in the middle of an answer: the model steps aside, the answer
     finishes, the model grows back after the program lets go (needs build/vram-eater: AMD with ROCm only;
     skipped without it)
  5. (with --game-appid) a game starts in the middle of an answer; a question asked during the game is
     answered on the processor; the model grows back after the game closes"""
import argparse
import http.client
import json
import os
import subprocess
import sys
import threading
import time

from . import sysmem
from .launch import RUNS
from .stress import EATER, STARTED_STEAM, close_game, eater_env, eater_why, health, router_running, start_game, wait_drawing

MiB, GiB = sysmem.MiB, sysmem.GiB
LONG = "Write a detailed, numbered list of 40 practical tips for keeping a home vegetable garden healthy."


class Server:
    def __init__(self, model, port, out):
        self.port = port
        self.log = open(os.path.join(out, "serve.log"), "w")
        self.proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "..", "bin", "lodger"),
                                      "serve", model, "--allow-gpu", "--port", str(port)],
                                     stdout=self.log, stderr=subprocess.STDOUT)

    def get(self, path):
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            c.request("GET", path)
            r = c.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        except (OSError, ValueError):
            return None, {}

    def status(self):
        return self.get("/lodger/status")[1]

    def where(self):
        s = self.status()
        return s.get("where") if s.get("ready") else None

    def wait_where(self, place, limit):
        t0 = time.time()
        while time.time() - t0 < limit:
            if self.where() == place:
                return time.time() - t0
            if self.proc.poll() is not None:
                return None
            time.sleep(1)
        return None

    def move(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request("POST", "/lodger/move", "{}", {"Content-Type": "application/json"})
        return json.loads(c.getresponse().read())

    def ask(self, question, n=900, res=None):
        """Stream an answer; returns {text, moves, error, seconds, started (time first words came)}.
        res: a dict to fill in as the answer comes (for watching it from another thread)."""
        res = {} if res is None else res
        res.update({"text": "", "moves": [], "error": None, "started": None})
        t0 = time.time()
        body = {"model": "", "stream": True, "max_tokens": n, "messages": [{"role": "user", "content": question}]}
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            for line in r:
                s = line.decode(errors="replace").strip()
                if s.startswith(": lodger"):
                    res["moves"].append(round(time.time() - t0, 1))
                elif s.startswith("data: ") and s != "data: [DONE]":
                    d = json.loads(s[6:])
                    if "error" in d:
                        res["error"] = d["error"].get("message")
                        break
                    piece = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                    if piece and res["started"] is None:
                        res["started"] = time.time()
                    res["text"] += piece
        except (OSError, ValueError, http.client.HTTPException) as ex:
            res["error"] = str(ex)
        res["seconds"] = round(time.time() - t0, 1)
        return res

    def ask_bg(self, question):
        box = {}
        t = threading.Thread(target=self.ask, args=(question, 900, box), daemon=True)
        t.start()
        return t, box

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(2)            # Ctrl-C: stops its engine cleanly
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


class LowWatch(threading.Thread):
    """Lowest free graphics memory seen."""
    def __init__(self):
        super().__init__(daemon=True)
        self.card, self.low, self.go = sysmem.dgpu(), None, True

    def run(self):
        while self.go:
            v = sysmem.vram_free(self.card)
            self.low = v if self.low is None else min(self.low, v)
            time.sleep(0.1)


def wait_started(box, limit=120):
    t0 = time.time()
    while time.time() - t0 < limit and not box.get("started") and "seconds" not in box:
        time.sleep(0.5)


def answered(box, why_ok=""):
    text = box.get("text", "")
    if box.get("error"):
        return False, f"answer failed: {box['error']}"
    if len(text) < 400:
        return False, f"answer too short ({len(text)} characters)"
    return True, f"answer finished ({len(text)} characters, {box.get('seconds')} s, restarted at {box.get('moves')} s){why_ok}"


def main(argv):
    ap = argparse.ArgumentParser(prog="lodger stress-serve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8182)
    ap.add_argument("--game-appid", type=int)
    a = ap.parse_args(argv)
    if router_running():
        sys.exit("Stop llama-router first: systemctl --user stop llama-router (and start it again afterwards).")
    out = os.path.join(RUNS, time.strftime("%Y-%m-%d-%H%M%S") + "-stress-serve")
    os.makedirs(out)
    low = LowWatch()
    low.start()
    srv = Server(a.model, a.port, out)
    results, children = [], []

    def phase(name, fn):
        since = time.strftime("%F %T")
        print(f"\n=== {name} ===", flush=True)
        try:
            ok, note = fn()
        except Exception as ex:                  # report and carry on to the next check
            ok, note = False, f"test error: {ex!r}"
        time.sleep(3)
        bad = health(since)
        results.append((name, ok and not bad, note, bad))
        print(("PASS" if results[-1][1] else "FAIL") + f": {note}", flush=True)
        for b in bad:
            print("  ! " + b, flush=True)
        return ok and not bad

    def p1():
        t = srv.wait_where("card", 600)
        if t is None:
            return False, f"not on the card: {srv.status().get('why')}"
        r = srv.ask("In two sentences, what does a lodger do?", 120)
        return (not r["error"] and len(r["text"]) > 40), f"on the card after {t:.0f} s; answered in {r['seconds']} s"

    def p2():
        t, box = srv.ask_bg(LONG)
        wait_started(box)
        time.sleep(5)
        m = srv.move()
        t.join(900)
        ok, note = answered(box)
        where = srv.where()
        return ok and m.get("moving") and where == "processor" and box.get("moves"), f"{note}; now on the {where}"

    def p3():
        t = srv.wait_where("card", 900)
        if t is None:
            return False, f"did not grow back: {srv.status().get('why')}"
        r = srv.ask("Name three colours.", 60)
        return not r["error"], f"grew back onto the card after {t:.0f} s; answered in {r['seconds']} s"

    def p4():
        t, box = srv.ask_bg(LONG)
        wait_started(box)
        time.sleep(4)
        eater = subprocess.Popen([EATER, "2048", "30"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=eater_env())
        children.append(eater)
        t.join(900)
        ok, note = answered(box)
        stepped = srv.where() == "processor" or any("stepped aside" in e[1] for e in srv.status().get("events", []))
        eater.wait(300)
        g = srv.wait_where("card", 900)
        return (ok and stepped and g is not None,
                f"{note}; stepped aside: {stepped}; grew back {'after %.0f s' % g if g is not None else 'NO'}")

    def p5():
        t, box = srv.ask_bg(LONG)
        wait_started(box)
        time.sleep(4)
        why = start_game(a.game_appid)
        if why:
            t.join(900)
            return False, why
        peak = wait_drawing()
        t.join(900)
        ok, note = answered(box)
        during = srv.ask("Name three animals.", 60)
        where_during = srv.status().get("where")
        time.sleep(30)                           # let the game run a little longer
        close_game()
        g = srv.wait_where("card", 1500)
        return (ok and not during["error"] and where_during == "processor" and g is not None,
                f"{note}; game drew with {peak // MiB} MiB; question during the game answered on the {where_during} "
                f"in {during['seconds']} s; grew back {'after %.0f s' % g if g is not None else 'NO'}")

    try:
        if phase("1. starts on the card", p1):
            phase("2. move off the card mid-answer", p2)
            phase("3. grow back", p3)
            no_eater = eater_why()
            if no_eater:
                print(f"\n=== 4. another program takes graphics memory mid-answer ===\nSKIPPED: {no_eater}", flush=True)
            else:
                phase("4. another program takes graphics memory mid-answer", p4)
            if a.game_appid:
                phase(f"5. game {a.game_appid} starts mid-answer", p5)
    finally:
        status = srv.status()
        for c in children:
            if c.poll() is None:
                c.kill()
        srv.stop()
        low.go = False
        if STARTED_STEAM:
            subprocess.run(["steam", "-shutdown"], capture_output=True)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(f"# lodger stress-serve {time.strftime('%F %T')}\n\nmodel: {a.model}\n"
                f"lowest free graphics memory: {(low.low or 0) / GiB:.2f} GiB\n\n")
        for name, ok, note, bad in results:
            f.write(f"- {'PASS' if ok else 'FAIL'} {name}: {note}\n" + "".join(f"  - ! {b}\n" for b in bad))
        f.write("\n## lodger events\n\n" + "".join(f"- {t} {e}\n" for t, e in status.get("events", [])))
    print(f"\nlowest free graphics memory: {(low.low or 0) / GiB:.2f} GiB; summary in {out}/summary.md")
    return 0 if results and all(r[1] for r in results) else 1
