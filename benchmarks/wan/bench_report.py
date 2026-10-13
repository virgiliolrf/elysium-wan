"""Charts and a report for one 2-stage WAN run, from what the nodes logged.

    python bench_report.py <run-dir> [--baseline <run-dir>] [--transport transport.json] [--gpu-profile gpu.json]
                           [--title "..."] [--skip-steps 2]

<run-dir> holds one sub-directory per node (va/, cz/, ...) with what `cluster.py collect` (or a copy loop) pulled:
stage-r*.log (WT_TELEM, WT_MEM), *_telemetry.jsonl (per-op events), *_loss_stream.jsonl (per-microbatch losses,
on the last stage), spec-*.json, sysmon.csv, coord.log. Writes <run-dir>/charts/*.png and <run-dir>/REPORT.md.

Definitions (every figure is measured, none modelled):
- step time: the stage's WT_TELEM step_ms.
- idle (stall share): WT_TELEM stall_ms / step_ms: time the stage had no runnable op, waiting for a message.
- GPU-busy share: the union of the stage's F, B and W op intervals over the step, from the event log. 1 minus it is
  the bubble a GPU sees: waits for messages, the optimizer's wait folded inside the first forward, and gaps.
- tokens/s: microbatches x mb_size x seq_len / step time.
Needs matplotlib (not a project dependency: run it from any venv that has it).
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1f2933"
GRID = "#d9dee4"
STAGE_COLORS = ["#1d6fa5", "#d1495b", "#2e8b57", "#8e6c8a"]
OPS = ("F", "B", "W")
EV_OPS = {"fwd": "F", "bwd_b": "B", "bwd_w": "W"}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)


# ------------------------------------------------------------------ loading


def _telem(path):
    out = []
    for line in open(path, errors="replace"):
        if line.startswith("WT_TELEM "):
            try:
                out.append(json.loads(line[9:]))
            except ValueError:
                pass
    return out


def _mem(path):
    out = {}
    for line in open(path, errors="replace"):
        if line.startswith("WT_MEM "):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            try:
                out[int(kv["step"])] = {k: float(v) for k, v in kv.items() if k != "step"}
            except (KeyError, ValueError):
                pass
    return out


def _jsonl(path):
    """Rows of a JSON-lines file; a line cut short (a file copied while being written) is skipped."""
    out = []
    fh = (
        gzip.open(path, "rt", errors="replace")
        if path.endswith(".gz")
        else open(path, errors="replace")
    )
    for x in fh:
        if x.strip():
            try:
                out.append(json.loads(x))
            except ValueError:
                pass
    return out


def load_run(d):
    """-> {"stages": {rank: {...}}, "losses": {step: [loss by mb]}, "spec": {...}, "sysmon": {node: rows}}."""
    run = {"stages": {}, "losses": {}, "spec": None, "sysmon": {}, "coord": ""}
    for sp in sorted(glob.glob(os.path.join(d, "*", "spec-*.json"))):
        run["spec"] = run["spec"] or json.load(open(sp))
    # A node's out/ can still hold an earlier run's files: only this run's tag counts.
    tag = run["spec"]["tag"] if run["spec"] else "*"
    for node in sorted(os.listdir(d)):
        nd = os.path.join(d, node)
        if not os.path.isdir(nd) or node == "charts":
            continue
        for log in glob.glob(os.path.join(nd, "stage-r*.log")):
            rank = int(os.path.basename(log)[7:].split("p")[0])
            st = run["stages"].setdefault(rank, {"node": node})
            st["telem"] = {t["step"]: t for t in _telem(log)}
            st["mem"] = _mem(log)
        for tel in glob.glob(os.path.join(nd, f"{tag}_rank*_telemetry.jsonl*")):
            rank = int(tel.split("_rank")[1][:1])
            st = run["stages"].setdefault(rank, {"node": node})
            st["events"] = _jsonl(tel)
        for ls in glob.glob(os.path.join(nd, f"{tag}_loss_stream.jsonl*")):
            for r in _jsonl(ls):
                run["losses"].setdefault(r["step"], {})[r["mb"]] = r["loss"]
        sm = os.path.join(nd, "sysmon.csv")
        if os.path.exists(sm):
            run["sysmon"][node] = [
                {k: float(v) for k, v in r.items() if v not in ("", None)}
                for r in csv.DictReader(open(sm))
            ]
        cl = os.path.join(nd, "coord.log")
        if os.path.exists(cl):
            run["coord"] = open(cl, errors="replace").read()
    return run


def tokens_per_step(spec):
    return spec["microbatches"] * spec.get("mb_size", 1) * spec["model"]["seq_len"]


def params_of(spec):
    m = spec["model"]
    d, n, v = m["d_model"], m["n_layer"], m["vocab_size"]
    return 12 * d * d * n + 2 * v * d


# ------------------------------------------------------------------ per-step measures


def _union(iv):
    total, end = 0.0, float("-inf")
    for a, b in sorted(iv):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def step_measures(stage):
    """Per step: step_s, idle, busy (GPU-busy share from F/B/W intervals), op medians, optimizer wait in the
    first forward, boundary wait (stall before the step's first op) and mid-step stall."""
    out = {}
    ev = stage.get("events") or []
    by_step: dict = {}
    for e in ev:
        if e.get("ev") in EV_OPS or e.get("ev") == "stall":
            by_step.setdefault(e.get("step"), []).append(e)
    starts = {}
    for s, es in by_step.items():
        comp = [e for e in es if e["ev"] in EV_OPS]
        if comp:
            starts[s] = min(e["t0"] for e in comp)
    for s, t in sorted(stage.get("telem", {}).items()):
        m = {
            "step_s": t["step_ms"] / 1e3,
            "idle": t["stall_ms"] / t["step_ms"],
            "ops": {k: v["med_ms"] for k, v in t["ops"].items()},
        }
        es = by_step.get(s)
        if es and s in starts and s + 1 in starts:
            comp = [e for e in es if e["ev"] in EV_OPS]
            span = starts[s + 1] - starts[s]
            busy = _union([(e["t0"], e["t1"]) for e in comp])
            f0 = [e for e in comp if e["ev"] == "fwd" and e.get("mb") == 0]
            fmed = statistics.median([e["t1"] - e["t0"] for e in comp if e["ev"] == "fwd"])
            optwait = max(0.0, (f0[0]["t1"] - f0[0]["t0"]) - fmed) if f0 else 0.0
            first_b = min((e["t0"] for e in comp if e["ev"] == "bwd_b"), default=starts[s])
            stalls = [e for e in es if e["ev"] == "stall"]
            fill = sum(e["t1"] - e["t0"] for e in stalls if e["t0"] < first_b)
            m.update(
                span_s=span,
                gpu_busy=(busy - optwait) / span,
                optwait_s=optwait,
                fill_stall_s=fill,
                mid_stall_s=sum(e["t1"] - e["t0"] for e in stalls) - fill,
                compute_s=busy - optwait,
            )
        out[s] = m
    return out


def recv_spreads(stage, kind):
    return [
        (e["t1"] - e["t0"]) * 1e3
        for e in stage.get("events") or []
        if e.get("ev") == "recv" and e.get("kind") == kind
    ]


# ------------------------------------------------------------------ charts


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_loss(run, out, title):
    steps = sorted(run["losses"])
    if not steps:
        return None
    mean = [statistics.mean(run["losses"][s].values()) for s in steps]
    lo = [min(run["losses"][s].values()) for s in steps]
    hi = [max(run["losses"][s].values()) for s in steps]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.fill_between(
        steps,
        lo,
        hi,
        color=STAGE_COLORS[0],
        alpha=0.15,
        lw=0,
        label="min-max over the step's microbatches",
    )
    ax.plot(steps, mean, color=STAGE_COLORS[0], lw=1.8, label="mean loss of the step")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_title(f"Training loss: {title}")
    ax.annotate(
        f"{mean[0]:.2f}", (steps[0], mean[0]), textcoords="offset points", xytext=(6, 4), color=INK
    )
    ax.annotate(
        f"{mean[-1]:.2f}",
        (steps[-1], mean[-1]),
        textcoords="offset points",
        xytext=(-14, 8),
        color=INK,
    )
    ax.legend(loc="upper right")
    return _save(fig, out, "loss.png")


def chart_bubble(meas, out, target=None, skip=2):
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for rank, m in sorted(meas.items()):
        steps = [s for s in sorted(m) if s >= skip]
        ax.plot(
            steps,
            [100 * m[s]["idle"] for s in steps],
            color=STAGE_COLORS[rank],
            lw=1.6,
            label=f"stage {rank}: waiting for a message (stall)",
        )
        bs = [s for s in steps if "gpu_busy" in m[s]]
        ax.plot(
            bs,
            [100 * (1 - m[s]["gpu_busy"]) for s in bs],
            color=STAGE_COLORS[rank],
            lw=1.0,
            ls="--",
            label=f"stage {rank}: GPU not computing F/B/W",
        )
    if target:
        ax.axhline(100 * target, color=INK, lw=0.8, ls=":")
        ax.text(
            ax.get_xlim()[1],
            100 * target,
            f" target {100 * target:.0f}%",
            va="center",
            color=INK,
            fontsize=8.5,
        )
    ax.set_ylim(bottom=0)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("share of the step (%)")
    ax.set_title("Pipeline bubble per stage")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out, "bubble.png")


def chart_throughput(meas, tok, out, skip=2, params=None, gpus=2, peak_tflops=None):
    m = meas[min(meas)]
    steps = [s for s in sorted(m) if s >= skip]
    ts = [tok / max(meas[r][s]["step_s"] for r in meas if s in meas[r]) for s in steps]
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(steps, ts, color=STAGE_COLORS[2], lw=1.6)
    med = statistics.median(ts)
    ax.axhline(med, color=INK, lw=0.8, ls=":")
    label = f" median {med:,.0f} tok/s"
    if params and peak_tflops:
        mfu = 6 * params * med / (gpus * peak_tflops * 1e12)
        label += f"  (MFU {100 * mfu:.1f}% of {peak_tflops:.0f} TFLOPs x {gpus})"
    ax.text(steps[0], med, label, va="bottom", color=INK, fontsize=8.5)
    ax.set_ylim(0, max(ts) * 1.15)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("tokens / s")
    ax.set_title("Training throughput")
    return _save(fig, out, "throughput.png")


def chart_breakdown(meas, out, skip=2):
    """Mean seconds per step, steady state, per stage: GPU computing, optimizer wait, waits before the first B
    (fill and boundary), mid-step waits, other."""
    ranks = sorted(meas)
    parts = {
        "computing F/B/W": [],
        "optimizer (in 1st forward)": [],
        "waiting at the step boundary": [],
        "waiting mid-step": [],
        "other": [],
    }
    for r in ranks:
        ss = [s for s in meas[r] if s >= skip and "compute_s" in meas[r][s]]
        if not ss:
            for k in parts:
                parts[k].append(0.0)
            continue

        def avg(k, r=r, ss=ss):
            return statistics.mean(meas[r][s][k] for s in ss)

        span = avg("span_s")
        comp, opt, fill, mid = (
            avg("compute_s"),
            avg("optwait_s"),
            avg("fill_stall_s"),
            avg("mid_stall_s"),
        )
        for k, v in zip(parts, (comp, opt, fill, mid, max(0.0, span - comp - opt - fill - mid))):
            parts[k].append(v)
    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    left = [0.0] * len(ranks)
    colors = ["#2e8b57", "#e0a458", "#d1495b", "#f08a8a", "#b8c2cc"]
    for (k, vals), c in zip(parts.items(), colors):
        ax.barh([f"stage {r}" for r in ranks], vals, left=left, color=c, label=k, height=0.55)
        for i, v in enumerate(vals):
            if v > 2.5:  # narrow segments are named in the legend and in REPORT.md's table
                ax.text(
                    left[i] + v / 2,
                    i,
                    f"{v:.1f}s",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
        left = [a + b for a, b in zip(left, vals)]
    ax.invert_yaxis()
    ax.set_xlabel("seconds per step (mean, steady state)")
    ax.set_title("Where each step's time goes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "step_breakdown.png")


def chart_ops(meas_by_label, out, skip=2):
    labels = list(meas_by_label)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
    for rank, ax in enumerate(axes):
        width = 0.8 / max(1, len(labels))
        for j, lab in enumerate(labels):
            meas = meas_by_label[lab]
            if rank not in meas:
                continue
            ss = [s for s in meas[rank] if s >= skip] or list(meas[rank])
            if not ss:
                continue
            vals = [statistics.median(meas[rank][s]["ops"].get(op, 0.0) for s in ss) for op in OPS]
            xs = [i + (j - (len(labels) - 1) / 2) * width for i in range(len(OPS))]
            bars = ax.bar(xs, vals, width=width * 0.92, color=STAGE_COLORS[j % 4], label=lab)
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        ax.set_xticks(range(len(OPS)), ["F", "B (incl. recompute)", "W"])
        ax.set_title(f"stage {rank}: ms per microbatch")
    axes[0].set_ylabel("median ms")
    axes[1].legend(loc="upper right", fontsize=8)
    return _save(fig, out, "ops.png")


def chart_recv(spreads_by_label, out):
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    for j, (lab, xs) in enumerate(spreads_by_label.items()):
        if not xs:
            continue
        xs = sorted(xs)
        ys = [100 * (i + 1) / len(xs) for i in range(len(xs))]
        ax.plot(
            xs,
            ys,
            color=STAGE_COLORS[j % 4],
            lw=1.6,
            label=f"{lab}: median {statistics.median(xs):.0f} ms, p90 {xs[int(0.9 * (len(xs) - 1))]:.0f} ms",
        )
    ax.set_xscale("log")
    ax.set_xlabel("first fragment to whole boundary (ms, log scale)")
    ax.set_ylabel("% of boundaries")
    ax.set_title("How long an activation takes to arrive whole (stage 1 receiving)")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out, "recv_spread.png")


def chart_transport(tr, out):
    if not tr:
        return None
    labels = [r["label"] for r in tr]
    p50 = [r["p50"] for r in tr]
    p90 = [r["p90"] for r in tr]
    idle = [r.get("idle") for r in tr]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    xs = range(len(tr))
    ax.bar([x - 0.27 for x in xs], p50, width=0.26, color=STAGE_COLORS[0], label="median")
    ax.bar(list(xs), p90, width=0.26, color=STAGE_COLORS[1], label="p90")
    ax.bar(
        [x + 0.27 for x in xs],
        [v or 0 for v in idle],
        width=0.26,
        color="#b8c2cc",
        label="first after 3 s idle",
    )
    for x, v in zip(xs, p50):
        ax.text(x - 0.27, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(list(xs), labels, fontsize=8)
    ax.set_ylabel("ms, send start to receiver's ack")
    ax.set_title("7.9 MB boundary every 357 ms, Virginia to Czechia: latency by flows and pacing")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out, "transport.png")


def chart_gpu(run, out, t0=None):
    if not run["sysmon"]:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.6), sharex=True)
    for j, (node, rows) in enumerate(sorted(run["sysmon"].items())):
        rows = [r for r in rows if "t" in r]
        if len(rows) < 3:
            continue
        base = t0 or rows[0]["t"]
        ts = [(r["t"] - base) / 60 for r in rows]
        util = [r.get("gpu_util", 0) for r in rows]
        k = 15
        sm = [statistics.mean(util[max(0, i - k) : i + 1]) for i in range(len(util))]
        axes[0].plot(
            ts, sm, color=STAGE_COLORS[j], lw=1.2, label=f"{node} GPU utilization (15 s mean)"
        )
        tx = []
        for a, b in zip(rows, rows[1:]):
            dt = b["t"] - a["t"]
            tx.append((b.get("tx_bytes", 0) - a.get("tx_bytes", 0)) * 8 / dt / 1e6 if dt > 0 else 0)
        txs = [statistics.mean(tx[max(0, i - k) : i + 1]) for i in range(len(tx))]
        axes[1].plot(
            ts[1:], txs, color=STAGE_COLORS[j], lw=1.2, label=f"{node} sent (Mbps, 15 s mean)"
        )
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("GPU utilization and network, sampled every second")
    axes[1].set_ylabel("Mbps")
    axes[1].set_xlabel("minutes since the first sample")
    axes[1].legend(loc="upper right", fontsize=8)
    return _save(fig, out, "gpu_network.png")


def chart_history(rows, out):
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    names = [r["name"] for r in rows]
    worst = [r["idle_max"] for r in rows]
    best = [r["idle_min"] for r in rows]
    xs = range(len(rows))
    ax.bar(list(xs), worst, color="#f08a8a", width=0.55, label="most idle stage")
    ax.bar(list(xs), best, color=STAGE_COLORS[0], width=0.55, label="least idle stage")
    for x, r in zip(xs, rows):
        ax.text(
            x,
            r["idle_max"] + 1,
            f"{r['tok_s']:,} tok/s" if r.get("tok_s") else "",
            ha="center",
            fontsize=7.5,
        )
    ax.set_xticks(list(xs), names, fontsize=7.5)
    ax.set_ylabel("idle share of the step (%)")
    ax.set_title("Elysium WAN runs so far: bubble and throughput")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out, "history.png")


def boundary_rows(run, step):
    """The critical path from stage 1's last B of step-1 to its first F of `step`, both clocks aligned on rank 0's
    (offset from message pairs: the minimum of sender's first fragment out to receiver's first fragment in, both
    directions, is offset + latency and latency - offset). -> [(label, t0_ms, t1_ms, color)], offset, latency."""
    e0 = run["stages"][0].get("events") or []
    e1 = run["stages"][1].get("events") or []

    def first_send(evs, hop):
        out = {}
        for e in evs:
            if e["ev"] == "send" and e.get("hop") == hop:
                k = (e["step"], e["mb"])
                out[k] = min(out.get(k, 1e18), e["t0"])
        return out

    def recv(evs, kind):
        return {(e["step"], e["mb"]): e for e in evs if e["ev"] == "recv" and e.get("kind") == kind}

    s0, r1 = first_send(e0, "next0"), recv(e1, "fwd")
    s1, r0 = first_send(e1, "prev0"), recv(e0, "bwd")
    a = min(r1[k]["t0"] - s0[k] for k in s0 if k in r1)
    b = min(r0[k]["t0"] - s1[k] for k in s1 if k in r0)
    off, lat = (a - b) / 2, (a + b) / 2
    p = step - 1

    def ops(evs, st, name, mb=None):
        return [
            e
            for e in evs
            if e["ev"] == name and e.get("step") == st and (mb is None or e.get("mb") == mb)
        ]

    last_b1 = max(ops(e1, p, "bwd_b"), key=lambda e: e["t1"])
    mbl = last_b1["mb"]
    T = last_b1["t1"] - off
    g_send, g_recv = s1[(p, mbl)] - off, r0[(p, mbl)]
    tail0 = [e for e in e0 if e.get("step") == p and e["ev"] in EV_OPS and e["t1"] > T]
    w_end = max(e["t1"] for e in tail0)
    opt0 = [e for e in e0 if e["ev"] == "opt_bg" and e["step"] == p][0]
    opt1 = [e for e in e1 if e["ev"] == "opt_bg" and e["step"] == p][0]
    f0 = ops(e0, step, "fwd", 0)[0]
    a_send, a_recv = s0[(step, 0)], r1[(step, 0)]
    f1 = ops(e1, step, "fwd", 0)[0]
    ms = lambda x: (x - T) * 1e3  # noqa: E731
    rows = [
        ("stage 1: optimizer (background)", ms(opt1["t0"] - off), ms(opt1["t1"] - off), "#e0a458"),
        ("last gradient: encode + send", 0.0, ms(g_send), "#b8c2cc"),
        ("last gradient: on the wire", ms(g_send), ms(g_recv["t1"]), "#d1495b"),
        ("stage 0: remaining B and W", ms(min(e["t0"] for e in tail0)), ms(w_end), "#2e8b57"),
        ("stage 0: optimizer", ms(opt0["t0"]), ms(opt0["t1"]), "#e0a458"),
        ("stage 0: first forward (waits per layer)", ms(f0["t0"]), ms(f0["t1"]), "#2e8b57"),
        ("first activation: encode", ms(f0["t1"]), ms(a_send), "#b8c2cc"),
        ("first activation: on the wire", ms(a_send), ms(a_recv["t1"] - off), "#d1495b"),
        ("stage 1: first forward", ms(f1["t0"] - off), ms(f1["t1"] - off), "#2e8b57"),
    ]
    return rows, off, lat


def chart_boundary(run, step, out, name="boundary.png", title=""):
    try:
        rows, _off, lat = boundary_rows(run, step)
    except (KeyError, ValueError, IndexError):
        return None
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for i, (_lab, a, b, c) in enumerate(rows):
        ax.barh(i, max(b - a, 4), left=a, color=c, height=0.6)
        ax.text(b + 25, i, f"{b - a:,.0f} ms", va="center", fontsize=7.5, color=INK)
    end = max(r[2] for r in rows)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(min(0, min(r[1] for r in rows)) - 50, end * 1.12)
    ax.set_xlabel(
        f"ms after stage 1's last backward of step {step - 1} (one clock; one-way latency {lat * 1e3:.0f} ms)"
    )
    ax.set_title(title or f"Step boundary {step - 1} to {step}: stage 1 idle {end / 1e3:.2f} s")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, name)


# ------------------------------------------------------------------ report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline")
    ap.add_argument("--baseline-label", default="before")
    ap.add_argument("--label", default="this run")
    ap.add_argument("--transport")
    ap.add_argument("--history")
    ap.add_argument("--title", default="")
    ap.add_argument("--skip-steps", type=int, default=2)
    ap.add_argument("--target", type=float, default=0.13)
    ap.add_argument("--peak-tflops", type=float, default=835.0)
    a = ap.parse_args()

    run = load_run(a.run)
    out = os.path.join(a.run, "charts")
    spec = run["spec"]
    tok = tokens_per_step(spec)
    params = params_of(spec)
    meas = {r: step_measures(st) for r, st in run["stages"].items()}
    title = a.title or f"{params / 1e9:.1f}B, {spec['stages']} stages"
    charts = [
        chart_loss(run, out, title),
        chart_bubble(meas, out, a.target, a.skip_steps),
        chart_throughput(meas, tok, out, a.skip_steps, params, spec["stages"], a.peak_tflops),
        chart_breakdown(meas, out, a.skip_steps),
        chart_gpu(run, out),
    ]
    base = load_run(a.baseline) if a.baseline else None
    if base:
        bmeas = {r: step_measures(st) for r, st in base["stages"].items()}
        charts.append(chart_ops({a.baseline_label: bmeas, a.label: meas}, out, a.skip_steps))
        charts.append(
            chart_recv(
                {
                    a.baseline_label: recv_spreads(base["stages"].get(1, {}), "fwd"),
                    a.label: recv_spreads(run["stages"].get(1, {}), "fwd"),
                },
                out,
            )
        )
    else:
        charts.append(chart_ops({a.label: meas}, out, a.skip_steps))
    steady = sorted(s for s in meas.get(1, {}) if s >= a.skip_steps + 1)
    if steady:
        charts.append(chart_boundary(run, steady[len(steady) // 2], out))
        if base:
            bsteady = sorted(s for s in bmeas.get(1, {}) if s >= a.skip_steps + 1)
            if bsteady:
                charts.append(
                    chart_boundary(
                        base,
                        bsteady[len(bsteady) // 2],
                        out,
                        "boundary_before.png",
                        f"{a.baseline_label}: step boundary",
                    )
                )
    if a.transport:
        charts.append(chart_transport(json.load(open(a.transport)), out))
    if a.history:
        charts.append(chart_history(json.load(open(a.history)), out))
    summary = summarize(run, meas, tok, params, a)
    json.dump(summary, open(os.path.join(a.run, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("charts:", ", ".join(os.path.basename(c) for c in charts if c))


def summarize(run, meas, tok, params, a):
    spec = run["spec"]
    m = spec["model"]
    steady = {r: [s for s in sorted(mm) if s >= a.skip_steps] for r, mm in meas.items()}
    step_s = [max(meas[r][s]["step_s"] for r in meas if s in meas[r]) for s in steady[min(steady)]]
    tok_s = [tok / x for x in step_s]
    losses = run["losses"]
    ls = sorted(losses)
    per_stage = {}
    for r, ss in steady.items():
        idle = [meas[r][s]["idle"] for s in ss]
        busy = [meas[r][s]["gpu_busy"] for s in ss if "gpu_busy" in meas[r][s]]
        per_stage[r] = {
            "node": run["stages"][r]["node"],
            "idle_median": statistics.median(idle) if idle else None,
            "idle_mean": statistics.mean(idle) if idle else None,
            "gpu_not_computing_median": 1 - statistics.median(busy) if busy else None,
            "peak_mem_gib": max(
                (v.get("peak", 0) for v in run["stages"][r].get("mem", {}).values()), default=None
            ),
            "ops_median_ms": {
                op: statistics.median(meas[r][s]["ops"].get(op, 0.0) for s in ss) for op in OPS
            }
            if ss
            else {},
        }
    med_tok = statistics.median(tok_s) if tok_s else None
    return {
        "model": {
            "params_b": round(params / 1e9, 2),
            "d_model": m["d_model"],
            "n_layer": m["n_layer"],
            "n_head": m["n_head"],
            "seq_len": m["seq_len"],
            "vocab": m["vocab_size"],
            "compute_dtype": m["compute_dtype"],
            "attn": f"{m.get('attn_impl')}/{m.get('sdpa_backend')}",
        },
        "partition": spec["partition"],
        "microbatches": spec["microbatches"],
        "tokens_per_step": tok,
        "steps_logged": len(step_s) + a.skip_steps,
        "steady_steps": len(step_s),
        "step_s_median": statistics.median(step_s) if step_s else None,
        "tokens_per_s_median": med_tok,
        "mfu_vs_peak": 6 * params * med_tok / (spec["stages"] * a.peak_tflops * 1e12)
        if med_tok
        else None,
        "loss_first": statistics.mean(losses[ls[0]].values()) if ls else None,
        "loss_last": statistics.mean(losses[ls[-1]].values()) if ls else None,
        "loss_steps": len(ls),
        "stages": per_stage,
    }


if __name__ == "__main__":
    main()
