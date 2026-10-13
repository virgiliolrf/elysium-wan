"""SVG charts for one collected WAN run (standard library only, so it runs anywhere and renders on GitHub).

    python plots.py <run-dir>
Writes <run-dir>/plots/{loss,step_time,throughput,network,gpu}.svg from the files analyze.py reads.
"""

import csv
import glob
import json
import os
import statistics
import sys

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 820, 340, 64, 24, 40, 48
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]


def _nice(lo, hi):
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def svg_lines(path, title, series, xlabel, ylabel, ylog=False):
    """series: [(label, [(x, y), ...])]."""
    pts = [p for _, s in series for p in s if p[1] is not None]
    if not pts:
        return
    x0, x1 = _nice(min(p[0] for p in pts), max(p[0] for p in pts))
    y0, y1 = _nice(min(0.0, min(p[1] for p in pts)), max(p[1] for p in pts) * 1.05)
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def X(x):
        return PAD_L + (x - x0) / (x1 - x0) * pw

    def Y(y):
        return PAD_T + ph - (y - y0) / (y1 - y0) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        'font-family="Helvetica,Arial,sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="15" font-weight="bold">{title}</text>',
    ]
    for i in range(6):
        yv = y0 + (y1 - y0) * i / 5
        out.append(
            f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="#e5e7eb"/>'
        )
        out.append(f'<text x="{PAD_L - 6}" y="{Y(yv) + 4:.1f}" text-anchor="end">{yv:.3g}</text>')
        xv = x0 + (x1 - x0) * i / 5
        out.append(
            f'<text x="{X(xv):.1f}" y="{H - PAD_B + 16}" text-anchor="middle">{xv:.4g}</text>'
        )
    out.append(f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle">{xlabel}</text>')
    out.append(
        f'<text x="14" y="{PAD_T + ph / 2}" text-anchor="middle" transform="rotate(-90 14 {PAD_T + ph / 2})">{ylabel}</text>'
    )
    out.append(
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{pw}" height="{ph}" fill="none" stroke="#9ca3af"/>'
    )
    for i, (label, s) in enumerate(series):
        s = [p for p in s if p[1] is not None]
        if not s:
            continue
        c = COLORS[i % len(COLORS)]
        d = " ".join(f"{X(x):.1f},{Y(min(y, y1)):.1f}" for x, y in s)
        out.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6" points="{d}"/>')
        lx = PAD_L + 10 + i * 190
        out.append(f'<rect x="{lx}" y="{PAD_T + 8}" width="14" height="4" fill="{c}"/>')
        out.append(f'<text x="{lx + 20}" y="{PAD_T + 14}">{label}</text>')
    out.append("</svg>")
    open(path, "w").write("\n".join(out))


def _telem(path):
    out = []
    for line in open(path, errors="replace"):
        if line.startswith("WT_TELEM "):
            try:
                out.append(json.loads(line[9:]))
            except ValueError:
                pass
    return out


def _smooth(points, k):
    out = []
    for i in range(len(points)):
        w = points[max(0, i - k + 1) : i + 1]
        out.append((points[i][0], statistics.mean(y for _, y in w)))
    return out


def main():
    run = sys.argv[1]
    os.makedirs(os.path.join(run, "plots"), exist_ok=True)
    P = lambda name: os.path.join(run, "plots", name)  # noqa: E731
    spec = json.load(open(sorted(glob.glob(os.path.join(run, "*", "spec-*.json")))[0]))
    tokens_step = spec["microbatches"] * spec["mb_size"] * spec["model"]["seq_len"]

    loss = []
    for p in glob.glob(os.path.join(run, "*", "*_loss_stream.jsonl")):
        loss = [json.loads(x) for x in open(p) if x.strip()]
    if loss:
        by = {}
        for r in loss:
            by.setdefault(r["step"], []).append(r["loss"])
        pts = [(s, statistics.mean(v)) for s, v in sorted(by.items())]
        svg_lines(
            P("loss.svg"),
            "Training loss (mean over the step's microbatches)",
            [("loss", pts), ("loss, 20-step mean", _smooth(pts, 20))],
            "step",
            "cross-entropy",
        )

    stages = {}
    for log in glob.glob(os.path.join(run, "*", "stage-r*p*.log")):
        node = os.path.basename(os.path.dirname(log))
        rank = int(os.path.basename(log).split("-r")[1].split("p")[0])
        stages[rank] = (node, _telem(log))
    if stages:
        last = max(stages)
        node, tel = stages[last]
        step_pts = [(t["step"], t["step_ms"] / 1e3) for t in tel]
        stall_pts = [(t["step"], t.get("stall_ms", 0) / 1e3) for t in tel]
        svg_lines(
            P("step_time.svg"),
            f"Step time and network wait (stage {last}, {node}; capped at 10 s)",
            [
                ("step time (s)", [(x, min(y, 10)) for x, y in step_pts]),
                ("waiting on the network (s)", [(x, min(y, 10)) for x, y in stall_pts]),
            ],
            "step",
            "seconds",
        )
        tps = [(t["step"], tokens_step / (t["step_ms"] / 1e3)) for t in tel]
        svg_lines(
            P("throughput.svg"),
            "Throughput (tokens per second, from each step's time)",
            [("tokens/s", tps), ("tokens/s, 20-step mean", _smooth(tps, 20))],
            "step",
            "tokens/s",
        )

    net, gpu = [], []
    for p in sorted(glob.glob(os.path.join(run, "*", "sysmon.csv"))):
        node = os.path.basename(os.path.dirname(p))
        rows = list(csv.DictReader(open(p)))
        if len(rows) < 3:
            continue
        t0 = float(rows[0]["t"])
        rx, tx, util = [], [], []
        for a, b in zip(rows, rows[1:]):
            dt = max(1e-3, float(b["t"]) - float(a["t"]))
            m = (float(b["t"]) - t0) / 60
            rx.append((m, (int(b["rx_bytes"]) - int(a["rx_bytes"])) * 8 / 1e6 / dt))
            tx.append((m, (int(b["tx_bytes"]) - int(a["tx_bytes"])) * 8 / 1e6 / dt))
            if b["gpu_util"]:
                util.append((m, float(b["gpu_util"])))
        net += [(f"{node} received", _smooth(rx, 10)), (f"{node} sent", _smooth(tx, 10))]
        gpu.append((f"{node} GPU utilization", _smooth(util, 10)))
    if net:
        svg_lines(
            P("network.svg"),
            "Network traffic per node (Mbit/s, 10 s mean)",
            net,
            "minutes",
            "Mbit/s",
        )
    if gpu:
        svg_lines(P("gpu.svg"), "GPU utilization per node (%, 10 s mean)", gpu, "minutes", "%")
    print("\n".join(sorted(os.listdir(os.path.join(run, "plots")))))


if __name__ == "__main__":
    main()
