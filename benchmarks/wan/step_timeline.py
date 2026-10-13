"""One measured training step of a 2-stage run, both GPUs on one clock: when each one computes, waits, or runs the
optimizer. The figure the README leads with.

    python step_timeline.py <run-dir> <step> <out-prefix> [--title ...] [--subtitle ...]
<run-dir> holds one sub-directory per node with the stages' *_telemetry.jsonl[.gz] (bench_report.py's layout).
Writes <out-prefix>-light.png and <out-prefix>-dark.png (GitHub picks one with <picture>).

The window is one step on stage 0: from its first forward of <step> to its first forward of <step>+1. Stage 1's clock
is moved onto stage 0's from message pairs (the minimum of send-to-arrival each way is offset + latency and
latency - offset). Computing is the union of the stage's F, B and W ops; the optimizer is its background AdamW
(opt_bg); everything else is waiting for a message.
Needs matplotlib (not a project dependency).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COMPUTE = ("fwd", "bwd_b", "bwd_w")
THEMES = {
    # Validated with the dataviz skill's checker (categorical, both modes: CVD dE 24.7 light / 26.8 dark).
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        wait="#e1e0d9",
        compute="#2a78d6",
        opt="#eb6834",
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        wait="#383835",
        compute="#3987e5",
        opt="#d95926",
    ),
}


def _rows(path):
    fh = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    out = []
    for line in fh:
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # a line cut short when the file was copied mid-write
    return out


def load(run_dir):
    ev = {}
    for f in glob.glob(os.path.join(run_dir, "*", "*_telemetry.jsonl*")):
        ev[int(f.split("_rank")[1][:1])] = _rows(f)
    return ev


def clock_offset(ev):
    def first_send(rows, hop):
        out = {}
        for e in rows:
            if e["ev"] == "send" and e.get("hop") == hop:
                k = (e["step"], e["mb"])
                out[k] = min(out.get(k, 1e18), e["t0"])
        return out

    def recv(rows, kind):
        return {
            (e["step"], e["mb"]): e["t0"]
            for e in rows
            if e["ev"] == "recv" and e.get("kind") == kind
        }

    s0, r1 = first_send(ev[0], "next0"), recv(ev[1], "fwd")
    s1, r0 = first_send(ev[1], "prev0"), recv(ev[0], "bwd")
    a = min(r1[k] - s0[k] for k in s0 if k in r1)
    b = min(r0[k] - s1[k] for k in s1 if k in r0)
    return (a - b) / 2, (a + b) / 2  # rank-1 clock minus rank-0 clock; one-way latency


def union(iv, eps=1e-4):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + eps:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def clip(iv, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in iv if b > lo and a < hi]


def measure(ev, step):
    off, lat = clock_offset(ev)
    shift = {0: 0.0, 1: off}
    f0 = [e for e in ev[0] if e["ev"] == "fwd" and e["mb"] == 0]
    t_lo = min(e["t0"] for e in f0 if e["step"] == step)
    t_hi = min(e["t0"] for e in f0 if e["step"] == step + 1)
    lanes = {}
    for r in (0, 1):
        comp = union(
            (e["t0"] - shift[r], e["t1"] - shift[r])
            for e in ev[r]
            if e["ev"] in COMPUTE and e.get("step") in (step - 1, step, step + 1)
        )
        opt = [(e["t0"] - shift[r], e["t1"] - shift[r]) for e in ev[r] if e["ev"] == "opt_bg"]
        comp = clip(comp, t_lo, t_hi)
        opt = [
            (a, b) for a, b in clip(opt, t_lo, t_hi) if b - a >= 0.05
        ]  # no sliver of the next step's
        # Stage 0's first forward of a step waits, layer by layer, for its optimizer (RUNTIME-I23): that span is
        # the optimizer's, not compute.
        if r == 0:
            f = next(e for e in ev[0] if e["ev"] == "fwd" and e["step"] == step and e["mb"] == 0)
            fmed = sorted(
                e["t1"] - e["t0"] for e in ev[0] if e["ev"] == "fwd" and e["step"] == step
            )
            fmed = fmed[len(fmed) // 2]
            cut = (f["t0"], f["t1"] - fmed)
            comp = [
                seg
                for a, b in comp
                for seg in ((a, min(b, cut[0])), (max(a, cut[1]), b))
                if seg[1] > seg[0]
            ]
        busy = sum(b - a for a, b in comp)
        lanes[r] = dict(
            comp=[(a - t_lo, b - t_lo) for a, b in comp],
            opt=[(a - t_lo, b - t_lo) for a, b in opt],
            busy=busy / (t_hi - t_lo),
        )
    return lanes, t_hi - t_lo, lat


def draw(lanes, span, theme, out, title, subtitle, notes):
    c = THEMES[theme]
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": c["ink"],
            "axes.labelcolor": c["ink2"],
            "xtick.color": c["muted"],
            "ytick.color": c["ink"],
        }
    )
    fig, ax = plt.subplots(figsize=(10.0, 3.55), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    rows = [
        (1, "Stage 0 · Virginia\nH100 NVL", lanes[0]),
        (0, "Stage 1 · Czechia\nH100 NVL", lanes[1]),
    ]
    h, oh = 0.42, 0.13  # compute bar and optimizer bar heights (data units)
    for y, _label, lane in rows:
        ax.broken_barh([(0, span)], (y - h / 2, h), facecolors=c["wait"], linewidth=0)
        ax.broken_barh(
            [(a, b - a) for a, b in lane["comp"]],
            (y - h / 2, h),
            facecolors=c["compute"],
            linewidth=0,
        )
        ax.broken_barh(
            [(a, b - a) for a, b in lane["opt"]],
            (y - h / 2 - oh - 0.05, oh),
            facecolors=c["opt"],
            linewidth=0,
        )
        ax.text(
            span + 0.35,
            y,
            f"{100 * lane['busy']:.0f}% computing",
            va="center",
            ha="left",
            fontsize=10,
            color=c["ink"],
            fontweight="bold",
        )
    for x, y, text, where in notes:
        if where == "beside-optimizer":  # a direct label to the right of the lane's optimizer bar
            ax.text(
                x,
                y - h / 2 - oh / 2 - 0.05,
                text,
                ha="left",
                va="center",
                fontsize=8.4,
                color=c["ink2"],
            )
            continue
        up = where == "above"
        ax.annotate(
            text,
            xy=(x, y + (h / 2 if up else -h / 2)),
            xytext=(x, y + (0.36 if up else -0.55)),
            ha="left",
            va="bottom" if up else "top",
            fontsize=8.4,
            color=c["ink2"],
            # the connector leaves the text at its left corner, so it stands vertical over the marked moment
            arrowprops=dict(
                arrowstyle="-",
                color=c["muted"],
                lw=0.8,
                shrinkA=1,
                shrinkB=1,
                relpos=(0.0, 0.0 if up else 1.0),
            ),
        )
    ax.set_yticks([y for y, _, _ in rows], [label for _, label, _ in rows], fontsize=9.5)
    ax.set_ylim(-0.95, 1.75)
    ax.set_xlim(0, span)
    ax.set_xlabel("seconds into the step (both clocks aligned)", fontsize=9)
    ax.xaxis.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=c["muted"])
    fig.text(0.012, 0.965, title, fontsize=12.5, fontweight="bold", color=c["ink"], va="top")
    fig.text(0.012, 0.885, subtitle, fontsize=9.2, color=c["ink2"], va="top")
    handles = [
        Patch(facecolor=c["compute"], label="computing (forward, backward, weight gradients)"),
        Patch(facecolor=c["opt"], label="AdamW, moments streamed from host memory"),
        Patch(facecolor=c["wait"], label="idle: waiting for a message or the optimizer (bubble)"),
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.6,
        bbox_to_anchor=(0.5, -0.01),
        handlelength=1.4,
    )
    for t in leg.get_texts():
        t.set_color(c["ink2"])
    fig.subplots_adjust(left=0.13, right=0.86, top=0.74, bottom=0.25)
    fig.savefig(out, facecolor=c["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("step", type=int)
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--subtitle", default="")
    ap.add_argument(
        "--notes",
        default="",
        help='JSON list of [x_seconds, lane (1 = stage 0, 0 = stage 1), text, "above" | "below" | "beside-optimizer"]',
    )
    a = ap.parse_args()
    lanes, span, lat = measure(load(a.run), a.step)
    print(f"step {a.step}: {span:.2f} s, one-way latency {lat * 1e3:.0f} ms")
    notes = json.loads(a.notes) if a.notes else []
    for theme in THEMES:
        draw(lanes, span, theme, f"{a.out}-{theme}.png", a.title, a.subtitle, notes)
        print("wrote", f"{a.out}-{theme}.png")
    for r, lane in lanes.items():
        waits = (
            [b2 - a2 for (_, a2), (b2, _) in zip(lane["comp"], lane["comp"][1:])]
            if lane["comp"]
            else []
        )
        lead = lane["comp"][0][0] if lane["comp"] else span
        print(
            f"  stage {r}: computing {100 * lane['busy']:.1f}%, first op at {lead:.2f} s, "
            f"gaps over 0.2 s: {sum(1 for w in waits if w > 0.2)}, optimizer spans {[(round(x, 2), round(y, 2)) for x, y in lane['opt']]}"
        )


if __name__ == "__main__":
    main()
