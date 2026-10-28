"""lodger measure: log live VRAM and RAM use per app while you use the PC normally, then summarize
the peaks. Read-only (sysfs + /proc); safe to run any time, even unattended.

  lodger measure [--seconds 600] [--interval 2] [--label everyday]

Writes <runs folder>/<date>-measure-<label>/samples.jsonl (one line per sample) and summary.md.
Ctrl+C stops early and still writes the summary."""
import argparse
import json
import os
import signal
import time

from . import config, sysmem

MiB = sysmem.MiB


def main(argv):
    ap = argparse.ArgumentParser(prog="lodger measure", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--interval", type=float, default=2)
    ap.add_argument("--label", default="desktop")
    ap.add_argument("--quiet", action="store_true", help="no live table")
    a = ap.parse_args(argv)

    root = config.runs_dir()
    out = os.path.realpath(os.path.join(root, time.strftime("%Y-%m-%d-%H%M") + "-measure-" + a.label))
    os.makedirs(out, exist_ok=True)
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))

    dev = sysmem.dgpu()
    pci = dev.pci if dev else None
    d = next((c for c in sysmem.cards() if c["pci"] == pci), None)
    peaks = {}          # app -> {vram, gtt, vram_igpu, pss}
    totals = []         # (t, vram_used, gtt_used, ram_avail, swap_used)
    t0 = time.time()
    with open(os.path.join(out, "samples.jsonl"), "w") as f:
        while not stop and time.time() - t0 < a.seconds:
            snap = sysmem.snapshot()
            t = round(time.time() - t0, 1)
            card = next((c for c in snap["cards"] if c["pci"] == pci), None)
            mem = snap["mem"]
            row = {
                "t": t,
                "vram_used": card["used"] if card else 0,
                "vram_total": card["total"] if card else 0,
                "gtt_used": card["gtt_used"] if card else 0,
                "ram_avail": mem["MemAvailable"],
                "ram_total": mem["MemTotal"],
                "swap_used": mem["SwapTotal"] - mem["SwapFree"],
                "apps": {},
            }
            for name, ap_ in snap["apps"].items():
                v = ap_["vram"].get(pci, 0)
                g = ap_["gtt"].get(pci, 0)
                vi = sum(x for k, x in ap_["vram"].items() if k != pci)
                if v or g or vi or ap_["pss"] >= 20 * MiB:
                    row["apps"][name] = {"vram": v, "gtt": g, "vram_igpu": vi, "pss": ap_["pss"], "procs": ap_["procs"]}
                    p = peaks.setdefault(name, {"vram": 0, "gtt": 0, "vram_igpu": 0, "pss": 0})
                    for k in p:
                        p[k] = max(p[k], row["apps"][name][k])
            f.write(json.dumps(row) + "\n")
            f.flush()
            totals.append((t, row["vram_used"], row["gtt_used"], row["ram_avail"], row["swap_used"]))
            if not a.quiet:
                live(row)
            time.sleep(max(0.0, a.interval - (time.time() - t0 - t)))

    summary = summarize(out, d, peaks, totals)
    print(summary)
    return 0


def live(row):
    top = sorted(row["apps"].items(), key=lambda kv: -(kv[1]["vram"] * 4 + kv[1]["pss"]))[:6]
    apps = "  ".join(f"{n[:14]} {v['vram'] // MiB}V/{v['pss'] // MiB}R" for n, v in top)
    print(f"{row['t']:7.1f}s  VRAM used {row['vram_used'] // MiB:5} MiB  RAM avail {row['ram_avail'] // MiB:6} MiB  | {apps}",
          flush=True)


def summarize(out, d, peaks, totals):
    if not totals:
        return "no samples"
    vram_peak = max(x[1] for x in totals)
    vram_min = min(x[1] for x in totals)
    gtt_peak = max(x[2] for x in totals)
    avail_min = min(x[3] for x in totals)
    swap_peak = max(x[4] for x in totals)
    total = d["total"] if d else 0
    counted_peak = 0
    lines = [
        f"# Measurement {os.path.basename(out)}",
        "",
        f"{len(totals)} samples over {totals[-1][0]:.0f} s. Graphics card: {d['card']} ({d['pci']}), "
        f"{total // MiB} MiB VRAM, screens on it: {', '.join(d['outputs']) or 'none'}." if d else "No graphics card found.",
        "",
        "## Whole PC",
        f"- VRAM used on the graphics card: lowest {vram_min // MiB} MiB, highest {vram_peak // MiB} MiB "
        f"(so at worst {(total - vram_peak) // MiB} MiB was free).",
        f"- GTT (normal RAM the card borrows) highest: {gtt_peak // MiB} MiB.",
        f"- RAM available, lowest: {avail_min // MiB} MiB. Swap used, highest: {swap_peak // MiB} MiB.",
        "",
        "## Per app (highest value seen for each; peaks may come at different times)",
        "| app | VRAM on card (MiB) | GTT (MiB) | VRAM on built-in graphics (MiB) | RAM (MiB, PSS) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, p in sorted(peaks.items(), key=lambda kv: -(kv[1]["vram"] * 4 + kv[1]["pss"])):
        if p["vram"] < MiB and p["gtt"] < MiB and p["vram_igpu"] < MiB and p["pss"] < 50 * MiB:
            continue
        counted_peak += p["vram"]
        lines.append(f"| {name} | {p['vram'] // MiB} | {p['gtt'] // MiB} | {p['vram_igpu'] // MiB} | {p['pss'] // MiB} |")
    lines += [
        "",
        f"Apps' own VRAM adds up to at most {counted_peak // MiB} MiB; the rest of the card's \"used\" is the "
        "kernel driver, firmware and the screens' frame buffers that no app owns.",
        "RAM is PSS: memory shared between processes is split between them, so the column adds up.",
    ]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(text)
    return text + f"\n(saved in {out})"

