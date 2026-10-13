"""The dense GPT shape a WAN fleet runs best (SCHEDULER-I23): the square-cube law, priced.

A stage computes ~8 x (its parameters) FLOPs per token (forward, recomputed forward, backward), so its compute per
microbatch grows with d^2 x layers; its boundary moves seq x d values per microbatch per direction, growing with d
alone. Over a slow hop the stage waits whenever moving a microbatch takes longer than computing it. The bubble that
leaves is about 1 - compute / max(compute, transfer), plus the pipeline's fill and drain, which are proportional to
the stage count over the microbatches. The 2026-10-12 runs measured exactly that: a 2.8B model across 45 Mbps
waited 75% of each step, and the planner had promised far less.

advise() picks, among standard shapes (d a multiple of 128, 128-dim heads), the one with the most parameters
whose stage fits memory (`bytes_per_param` of state plus a measured-calibrated activation charge) and whose priced
bubble stays under `bubble_target`. If none does, it picks the one with the least bubble. Candidates keep the
width/depth ratio d_model / n_layer within `aspect` (20-200 by default: Kaplan et al. 2020 measured the loss within
a few percent of optimal across that range; GPT-3 175B is 128): the advice trades no quality for speed, and every
candidate is a standard dense decoder, trained exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

VOCAB = 50257


@dataclass(frozen=True)
class Shape:
    d_model: int
    n_head: int
    layers_per_stage: int
    stages: int
    params: float  # total, embedding and head included
    compute_s: float  # per microbatch per stage
    transfer_s: float  # per microbatch over the slowest hop
    bubble: float  # priced fraction of a step the GPUs wait
    stage_bytes: float  # priced device memory of the heaviest stage

    @property
    def n_layer(self) -> int:
        return self.layers_per_stage * self.stages


def price(
    d: int,
    layers_per_stage: int,
    stages: int,
    *,
    tflops: float,
    link_mbps: float,
    seq: int = 1024,
    microbatches: int = 32,
    bytes_per_param: float = 8.0,
    wire_ratio: float = 0.52,
    act_bytes_per_layer_token: float | None = None,
    pending: int = 3,
) -> Shape:
    """One candidate's figures. tflops: the stage GPU's effective dense bf16 rate (measured, e.g. the coordinator's
    probe x achieved utilization); link_mbps: the slowest hop's per-direction rate as the data plane sees it;
    wire_ratio: bytes on the wire per raw bf16 boundary byte (lossless planes codec, 0.52 measured on 2026-10-12);
    act_bytes_per_layer_token: activation bytes a layer holds per token per in-flight microbatch (default 64 d, the
    16 d values of the deferred W store at 4 bytes, as measured beside the analytic 32 d)."""
    per_layer = 12 * d * d
    stage_params = (
        per_layer * layers_per_stage + VOCAB * d
    )  # the heaviest stage holds the table or the head
    compute = 8.0 * stage_params * seq / (tflops * 1e12)
    transfer = seq * d * 2 * wire_ratio * 8 / (link_mbps * 1e6) if link_mbps > 0 else 0.0
    stall = 1.0 - compute / max(compute, transfer) if compute > 0 else 1.0
    fill = (stages - 1) / (microbatches + stages - 1)
    bubble = 1.0 - (1.0 - stall) * (1.0 - fill)
    act = act_bytes_per_layer_token if act_bytes_per_layer_token is not None else 64.0 * d
    stage_bytes = stage_params * bytes_per_param + act * seq * layers_per_stage * (pending + 1)
    total = per_layer * layers_per_stage * stages + 2 * VOCAB * d
    return Shape(
        d_model=d,
        n_head=max(1, d // 128),
        layers_per_stage=layers_per_stage,
        stages=stages,
        params=float(total),
        compute_s=compute,
        transfer_s=transfer,
        bubble=bubble,
        stage_bytes=stage_bytes,
    )


def advise(
    stages: int,
    memory_bytes: float,
    *,
    tflops: float,
    link_mbps: float,
    bubble_target: float = 0.2,
    d_choices: tuple = tuple(range(1024, 12288 + 1, 512)),
    max_layers_per_stage: int = 128,
    aspect: tuple = (20.0, 200.0),
    **kw,
) -> dict:
    """-> {"best": Shape, "fits": [Shape...]}: the largest standard shape whose heaviest stage fits `memory_bytes`
    (the fit budget) with a priced bubble at most `bubble_target`, else the fitting shape with the least bubble."""
    fits = []
    for d in d_choices:
        for lps in range(1, max_layers_per_stage + 1):
            s = price(d, lps, stages, tflops=tflops, link_mbps=link_mbps, **kw)
            if s.stage_bytes > memory_bytes:
                break
            if aspect[0] <= d / s.n_layer <= aspect[1]:
                fits.append(s)
    if not fits:
        raise ValueError("no standard shape fits one stage's memory within the width/depth range")
    ok = [s for s in fits if s.bubble <= bubble_target]
    best = max(ok, key=lambda s: s.params) if ok else min(fits, key=lambda s: (s.bubble, -s.params))
    return {"best": best, "fits": fits, "met_target": bool(ok)}


def needed_mbps(
    d: int, layers_per_stage: int, *, tflops: float, seq: int = 1024, wire_ratio: float = 0.52
) -> float:
    """The hop rate at which this stage stops waiting: its boundary moves in its compute time."""
    s = price(d, layers_per_stage, 2, tflops=tflops, link_mbps=1e9, seq=seq, wire_ratio=wire_ratio)
    return seq * d * 2 * wire_ratio * 8 / s.compute_s / 1e6 if s.compute_s > 0 else math.inf
