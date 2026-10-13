"""E4 (ADR-0006 F3, F5, F6): the exposed exchange at the scaled B', judged
against the as-built SCHEDULER-I17 simulation.

The predictions are those of the granted `docs/experiments/m2b-e4/PLAN.json`.
This module reads them from that file and refuses a configuration that is
not the PLAN's (`check_configuration`, `check_split`). Before any arm is
measured, the simulation must be anchored (`precheck`):

- at round 5's configuration (F' 50 GFLOP/s, op split 1:2:1, the model's
  HBM') it reproduces the PLAN's `round5_anchor` within its tolerance;
- at the E4 configuration (the PLAN's measured F', op split and fold rate)
  it reproduces every arm of the PLAN's pinned table within 5%.

An arm is one stage of width w alone (`simulate_arm`), laid out as the
PLAN's `derive/e4_spec.py`: each peer runs F, B, W for its M/w microbatches;
the units are the Embed (tok + pos, ONE fp32 unit, no W compute) on stage 0
and each block's four SplitLinear modules; NIC and PCIe at cap'; one host per
peer with unconstrained DRAM (the fold is not paced); the chunk f' and the hop
latency 2f'/PCIe'. Stage 0 is padded per microbatch slot to C_B'
(`scaled.stage0_padding`, window_s 0); the 'today' arms run the whole-stage
flush (`scaled.today_mode`). Exposure is in % of the arm's C, scale-free.

The simulation is pure: no I/O beyond reading the PLAN, no randomness.

The measured side on this tree:
- F5 (`measure_exchange`): a standalone exchange harness. w processes move the
  stage's exchange bytes through the shared-pacer shim (TRANSPORT-I10) and
  never touch training state; it measures the streaming left fold and the
  residue-group tree at the same configuration.
- The sweep rule (`sweep_threshold`), applied alike to measured and simulated
  points.
- The training arms (F3, F6, the controls, today, the emission and RTT arms;
  `training_run`, `measure_training`): the runtime itself (stage.py) over the
  shared-pacer shim at the PLAN's scaled configuration, through timing-only
  seams (WANTRAIN_TEST_CHUNK_BYTES, WANTRAIN_TEST_INTRA_LATENCY_MS,
  WANTRAIN_TEST_SLOT_PAD). The network is set from the stage's in-situ F'
  (`check_fprime`); E is read from the measured stage's telemetry as the
  simulation defines it (`stage_late`) and judged against the I17 simulation
  at the arm's in-situ op times; every state must be the reference's.
- PLAN section 5's summary (`summary_line`) from every arm's result line.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import socket
import struct
import tempfile
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from wantrain.launch.local import ensure_reference, reference_state_path, run_local
from wantrain.model.gpt import make_config
from wantrain.runtime.exchange import GATHER, GRAD, NeedFirstQueue, NeedKey
from wantrain.runtime.spec import JobSpec
from wantrain.scheduler.exchange_sim import (
    Fabric,
    Peer,
    StageLayout,
    Unit,
    pieces_owners,
    simulate_exchange,
    stride_owners,
)
from wantrain.scheduler.zero_bubble import B, F, W, ZBConfig
from wantrain.transport.shim import DelayedDelivery, LinkSpec, SenderPacer, SharedBudget
from wantrain.validation.state import diff_manifests, distributed_manifest

PLAN_PATH = Path(__file__).resolve().parents[3] / "docs/experiments/m2b-e4/PLAN.json"
ANCHOR_TOLERANCE = 0.05  # relative: the simulation against a pinned table

# B (round 4): H100 400 TFLOP/s, 400 Gbps = 50 GB/s, HBM 3.35 TB/s, 16 MiB chunk.
F_B = 400e12
T_MB_B = 4096
H_B = 4096
CAP_B = 50e9
HBM_B = 3.35e12
CHUNK_B = 16 * 2**20

# round5.md section 6's configuration: F' 50 GFLOP/s, split 1:2:1, HBM' paced.
# The pinned percentages are the PLAN's `round5_anchor.arms` (ROUND5_PCT).
ROUND5_FPRIME = 50e9
ROUND5_SPLIT = (0.25, 0.5, 0.25)

# Arm prefix -> (unit kind, blocks, w, M/w), PLAN.json `scaled`.
STAGES = {
    "stage0": ("first", 1, 4, 32),
    "interior_w4": ("mid", 4, 4, 8),
    "interior_w8": ("mid", 4, 8, 4),
}
_MODULES = (
    (3, 2),
    (1, 4),
    (4, 8),
    (4, 10),
)  # (multiple of h^2, today's stride index): qkv, proj, fc1, fc2


@dataclass(frozen=True)
class Shape:
    h: int
    vocab: int
    seq: int
    t_mb: int


@dataclass(frozen=True)
class Arm:
    """One E4 arm as the I17 simulation takes it. `layout` is pieces, step1
    (today's stride ZeRO-1, barrier on) or lpt (whole tensors placed longest
    first, explicit bounds). `stage_flush` is today's whole-stage flush
    (`scaled.today_mode`: every unit released at W's end)."""

    stage: str
    layout: str = "pieces"
    emission: str = "forward"
    policy: str = "need"
    barrier: bool | None = None
    rtt: float = 0.0
    sites: tuple[str, ...] | None = None
    tree: bool = False
    cap: float | None = None  # NIC cap' in B/s; None: the scaling map's
    wire_fp32: bool = False
    stage_flush: bool = False


@dataclass(frozen=True)
class Simulated:
    pct: float  # exposure in % of C
    exposure: float  # seconds
    period: float  # C, seconds


def load_plan(path: str | Path = PLAN_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    for key in ("scaled", "arms", "ratios", "ratio_tolerance_rel", "expected", "round5_anchor"):
        if key not in plan:
            raise ValueError(f"E4 PLAN {path}: missing {key!r}")
    return plan


def round5_pct(plan: Mapping) -> dict[str, float]:
    """The pinned round-5 anchor, % of C per arm (PLAN.json `round5_anchor.arms`)."""
    return {k: float(v) for k, v in plan["round5_anchor"]["arms"].items()}


ROUND5_PCT = round5_pct(load_plan())


def scaled_shape(plan: Mapping) -> Shape:
    sc = plan["scaled"]
    return Shape(sc["h"], sc["V"], sc["seq"], sc["T_mb"])


def _wkey(w: int) -> str:
    return f"w{w}"


def arm_of(name: str, plan: Mapping) -> Arm:
    """The simulation inputs of a PLAN arm, by its name."""
    parts = name.split(".")
    by = {a["arm"]: a for a in plan["arms"]}
    if name not in by:
        raise ValueError(f"E4 PLAN has no arm {name!r}")
    row = by[name]
    if parts[0] == "sweep":
        return Arm("interior_w4", cap=row["cap_MBps"] * 1e6)
    stage, kind = parts[0], parts[1]
    if stage not in STAGES:
        raise ValueError(f"E4 arm {name!r}: unknown stage {stage!r}")
    if kind == "today":
        return Arm(stage, "step1", emission="backward", wire_fp32=True, stage_flush=True)
    if kind in ("step1", "lpt"):
        return Arm(stage, kind)
    if kind == "tree":
        return Arm(stage, tree=True)
    rtt = row.get("rtt_ms", 0.0) / 1e3
    sites = None
    if kind.startswith("uplink"):
        sites = tuple(f"site{s}" for s in plan["scaled"]["uplink"]["sites"])
    return Arm(
        stage,
        emission="backward" if kind.endswith("bwd") else "forward",
        policy="fifo" if kind.startswith("fifo") else "need",
        barrier=True if kind.startswith("barrier") else None,
        rtt=rtt,
        sites=sites,
    )


def stage_units(kind: str, n_blocks: int, shape: Shape, wire_fp32: bool = False):
    """The I17 units of a stage and, per unit, its whole tensors as (elements,
    today's stride index): the Embed as ONE fp32 unit (tok, index 0, and pos,
    index 1; no W compute), then each block's four SplitLinear modules."""
    h = shape.h
    units: list[Unit] = []
    tensors: list[tuple[tuple[int, int], ...]] = []
    base = 0
    if kind == "first":
        units.append(Unit((shape.vocab + shape.seq) * h, 4, 4, 0, 0.0))
        tensors.append(((shape.vocab * h, 0), (shape.seq * h, 1)))
        base = 2
    b = 4 if wire_fp32 else 2
    for blk in range(n_blocks):
        for mult, i in _MODULES:
            p = mult * h * h
            units.append(Unit(p, b, b, len(units), float(p)))
            tensors.append(((p, base + 12 * blk + i),))
    return tuple(units), tensors


def lpt_owners(units: Sequence[Unit], tensors, w: int) -> tuple[tuple[float, ...], ...]:
    """Whole tensors, longest first onto the least loaded peer (r4.py
    `own_of(..., "lpt")`); a unit of several tensors gets the share of its
    elements on each peer. The simulation runs them as explicit bounds."""
    load = [0] * w
    own = [[0.0] * w for _ in units]
    order = sorted(
        ((n, ui, ti) for ui, ts in enumerate(tensors) for ti, (n, _i) in enumerate(ts)),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    for n, ui, _ti in order:
        j = load.index(min(load))
        load[j] += n
        own[ui][j] += n / units[ui].params
    return tuple(tuple(r) for r in own)


@dataclass(frozen=True)
class _Setup:
    """One arm at the scaling map: what the simulation and the F5 harness
    both run. `period` is C: C_B' on stage 0, (M/w) c_s on an interior stage."""

    kind: str
    w: int
    mw: int
    units: tuple[Unit, ...]
    tensors: list
    port: float  # cap' (B/s)
    nic: float
    pcie: float
    hbm: float
    chunk: int  # f' (bytes)
    boundary: int  # pipeline-boundary bytes per microbatch per direction
    c: float  # c_s, one microbatch's F + B + W (s)
    split: tuple[float, float, float]
    period: float

    @property
    def padded(self) -> bool:
        return self.kind == "first"


def _setup(
    arm: Arm,
    shape: Shape,
    fprime: float,
    split: Sequence[float],
    hbm: float | None,
    net_fprime: float | None = None,
) -> _Setup:
    """round5.md section 6's scaling map: cap' = PCIe' = 50 GB/s x s_rate,
    f' = 16 MiB x (h'/h)^2, HBM' = 3.35 TB/s x s_rate unless `hbm` (a measured
    fold rate) is given, s_rate = (F'/F)(T_mb/T_mb'); stage 0's C_B' = 32 c_last'.
    `net_fprime` (default `fprime`) is the F' the network and stage 0's period
    were set from: a training arm sets them from the PLAN's F' and computes at
    its in-situ F' (`fprime`)."""
    kind, n_blocks, w, mw = STAGES[arm.stage]
    if len(split) != 3 or min(split) <= 0 or abs(sum(split) - 1.0) > 1e-3:
        raise ValueError(f"op split must be 3 positive fractions summing to 1, got {split}")
    net = fprime if net_fprime is None else net_fprime
    s_rate = (net / F_B) * (T_MB_B / shape.t_mb)
    port = CAP_B * s_rate
    nic = port if arm.cap is None else arm.cap
    units, tensors = stage_units(kind, n_blocks, shape, arm.wire_fp32)
    if arm.tree:
        units = tuple(Unit(u.params, 4, u.b_p, u.fwd_index, u.flops) for u in units)
    c = 8 * sum(u.flops for u in units) * shape.t_mb / fprime
    c_last = 8 * (12 * shape.h**2 + shape.vocab * shape.h) * shape.t_mb / net
    return _Setup(
        kind=kind,
        w=w,
        mw=mw,
        units=units,
        tensors=tensors,
        port=port,
        nic=nic,
        pcie=max(port, nic),  # PCIe' = cap'; a swept NIC below it leaves the hop latency 2f'/PCIe'
        hbm=HBM_B * s_rate if hbm is None else hbm,
        chunk=max(1, int(CHUNK_B * (shape.h / H_B) ** 2)),
        boundary=(4 if kind == "mid" else 2) * shape.t_mb * shape.h,
        c=c,
        split=(split[0], split[1], split[2]),
        period=32 * c_last if kind == "first" else mw * c,
    )


def peer_order(n: int, ahead: int = 1) -> list[tuple[str, int]]:
    """One peer's op order over its n owned microbatches (local index i):
    F of the first `ahead`, then B(i), W(i), F(i + ahead). ahead 1 is F, B, W
    per microbatch (the PLAN's layout); a training arm's interior stage runs
    `ahead` > 1 so that B(i) never waits for the downstream backward."""
    out = [("F", i) for i in range(min(ahead, n))]
    for i in range(n):
        out += [("B", i), ("W", i)]
        if i + ahead < n:
            out.append(("F", i + ahead))
    return out


def simulate_arm(
    arm: Arm,
    shape: Shape,
    fprime: float,
    split: Sequence[float],
    hbm: float | None = None,
    net_fprime: float | None = None,
    ahead: int = 1,
) -> Simulated:
    """The I17 exposure of one arm at the scaling map (`_setup`). Stage 0 runs
    each microbatch in its slot of C_B' (`scaled.stage0_padding`); E is read
    against the slot. `ahead`: the peers' op order (`peer_order`)."""
    su = _setup(arm, shape, fprime, split, hbm, net_fprime)
    w, mw, units, tensors, c = su.w, su.mw, su.units, su.tensors, su.c
    owners, layout = {
        "pieces": (pieces_owners(len(units), w), "pieces"),
        "step1": (stride_owners(tensors, w), "zero1"),
        "lpt": (lpt_owners(units, tensors, w), "bounds"),
    }[arm.layout]
    sites = arm.sites or ("site0",) * w
    if len(sites) != w:
        raise ValueError(f"{arm.stage}: {len(sites)} sites for {w} peers")
    peers = tuple(Peer(f"p{q}", su.nic, su.pcie, su.hbm, f"host{q}", sites[q]) for q in range(w))
    stage = StageLayout(
        units,
        owners,
        peers,
        layout=layout,
        barrier=arm.barrier,
        boundary_bytes=su.boundary,
        flush="stage" if arm.stage_flush else "module",
    )
    m = w * mw
    ff, fb, fw = su.split
    period = su.period if su.padded else None
    ops = {"F": F, "B": B, "W": W}
    orders = [[[(ops[o], i * w + q) for o, i in peer_order(mw, ahead)] for q in range(w)]]
    if period is None:
        cfg = ZBConfig(1, m, [c * ff / w], [c * fb / w], [c * fw / w])
        weights = [[1.0] * w]
    else:
        # scaled.stage0_padding: microbatch i of a peer in the slot [i tau, (i + 1) tau),
        # F at its start, B then W ending it. A stand-in stage of one lane per peer (no
        # exchange) holds each B back by tau - c_s after its F.
        tau = period / mw
        cfg = ZBConfig(
            2, m, [c * ff / w, 0.0], [c * fb / w, (tau - c) / w], [c * fw / w, 0.0], comm=[0.0]
        )
        orders.append(
            [[op for mb in range(q, m, w) for op in ((F, mb), (B, mb))] for q in range(w)]
        )
        weights = [[1.0] * w, [1.0] * w]
    uplinks = {(x, y): su.port for x in set(sites) for y in set(sites) if x != y}
    fabric = Fabric(
        host_dram={p.host: math.inf for p in peers},
        uplinks=uplinks,
        rtt=arm.rtt,
        chunk=su.chunk,
    )
    got = simulate_exchange(
        cfg,
        orders,
        weights,
        {0: stage},
        fabric,
        emission=arm.emission,
        policy=arm.policy,
        period=period,
    )
    pred = got.stages[0]
    exposure = pred.tail if arm.tree else pred.exposure
    return Simulated(100 * exposure / su.period, exposure, su.period)


def plan_inputs(plan: Mapping, stage: str) -> tuple[float, tuple[float, float, float], float]:
    """(F', op split, fold rate) the PLAN pins for a stage's width."""
    sc = plan["scaled"]
    key = _wkey(STAGES[stage][2])
    return (
        float(sc["F_prime_flops"][key]),
        tuple(sc["op_split_F_B_W"][key]),
        float(sc["hbm_prime_gbs"]) * 1e9,
    )


def simulate_plan_arm(plan: Mapping, name: str) -> Simulated:
    arm = arm_of(name, plan)
    fprime, split, hbm = plan_inputs(plan, arm.stage)
    return simulate_arm(arm, scaled_shape(plan), fprime, split, hbm)


def ratios(shape: Shape, fprime: float) -> dict[str, float]:
    """round5.md section 6's kept ratios (PLAN.json `ratios`) at a shape and F'."""
    h, v, seq, t = shape.h, shape.vocab, shape.seq, shape.t_mb
    units, _ = stage_units("first", 1, shape)
    p0 = sum(u.params for u in units) + 13 * h  # + the block's biases and norms
    tables = (v + seq) * h
    grad_bytes = sum(u.b_g * u.params for u in units) + 2 * 13 * h
    s_rate = (fprime / F_B) * (T_MB_B / t)
    cap = CAP_B * s_rate
    chunk = max(1, int(CHUNK_B * (h / H_B) ** 2))
    c = 8 * sum(u.flops for u in units) * t / fprime
    w = STAGES["stage0"][2]
    per_mb = sum(u.b_g * u.params * (w - 1) / w for u in units)
    return {
        "V_over_h": v / h,
        "seq_over_h": seq / h,
        "Tmb_over_h": t / h,
        "embed_share_stage0": tables / p0,
        "tok_share_stage0": v * h / p0,
        "fp32_share_stage0_grad_bytes": 4 * tables / grad_bytes,
        "nic_util_stage0": (per_mb + 2 * t * h) / cap / c,
        "M_over_w_stage0": STAGES["stage0"][3],
        "fold_over_c_HBM_model": 18 * (sum(u.params for u in units) / w) / (HBM_B * s_rate) / c,
        "chunk_time_over_c": chunk / cap / c,
        "hop_latency_over_c": 2 * chunk / cap / c,
        "C_B_over_c_stage0": 32 * (1 + v / (12 * h)),
    }


def check_ratios(plan: Mapping, fprime: float) -> dict[str, float]:
    """Every kept ratio at the PLAN's B' and this F' against full B, within
    the PLAN's tolerance; raises naming the first ratio off. Returns the
    relative deviation of each."""
    ref = plan["ratios"]["full_B"]
    got = ratios(scaled_shape(plan), fprime)
    tol = float(plan["ratio_tolerance_rel"])
    out = {}
    for key, want in ref.items():
        if key not in got:
            raise ValueError(f"E4 ratio {key!r}: the harness does not compute it")
        rel = got[key] / want - 1
        out[key] = rel
        if abs(rel) > tol:
            raise ValueError(
                f"E4 ratio {key!r}: {got[key]:.6g} at B' against {want:.6g} at B ({100 * rel:+.3f}%)"
            )
    return out


def check_configuration(plan: Mapping, run: Mapping) -> None:
    """Refuse a run off the PLAN's scaled configuration, naming the field.
    `run` holds the harness's h, V, seq, T_mb, n_head, intra_op_threads,
    chunk_bytes, fold_paced and, per stage, w and M_over_w."""
    sc = plan["scaled"]
    for key in ("h", "V", "seq", "T_mb", "n_head", "intra_op_threads", "chunk_bytes", "fold_paced"):
        if run.get(key) != sc[key]:
            raise ValueError(
                f"E4 configuration: {key} is {run.get(key)!r}, the PLAN pins {sc[key]!r}"
            )
    for stage, (_kind, _nb, w, mw) in STAGES.items():
        if sc["w"][stage] != w or sc["M_over_w"][stage] != mw:
            raise ValueError(f"E4 PLAN: {stage} is not w {w}, M/w {mw}")
        got = run.get("stages", {}).get(stage, {})
        if got.get("w") != w or got.get("M_over_w") != mw:
            raise ValueError(
                f"E4 configuration: {stage} w/M_over_w {got!r}, the PLAN pins {w}/{mw}"
            )


def check_fprime(plan: Mapping, fprime_net: float, fprime_insitu: float) -> None:
    """PLAN section 7: cap' = PCIe' = 2e-3 B/s per FLOP/s x the in-situ F'. A
    run whose network was set from an F' more than the PLAN's in-situ
    tolerance (op_split_tolerance_rel, the one it states) off the F' it then
    computed at is off the configuration."""
    tol = float(plan["scaled"]["op_split_tolerance_rel"])
    rel = fprime_net / fprime_insitu - 1
    if abs(rel) > tol:
        raise ValueError(
            f"E4 configuration: cap' set from F' {fprime_net / 1e9:.3f} GFLOP/s, in-situ F' "
            f"{fprime_insitu / 1e9:.3f} ({100 * rel:+.1f}%, tolerance {100 * tol:.0f}%)"
        )


def check_split(plan: Mapping, w: int, split: Sequence[float]) -> None:
    """The in-situ op split within the PLAN's relative tolerance of the pinned
    one, per op; otherwise the run is off the configuration."""
    sc = plan["scaled"]
    pinned = sc["op_split_F_B_W"][_wkey(w)]
    tol = float(sc["op_split_tolerance_rel"])
    for op, got, want in zip("FBW", split, pinned, strict=True):
        if abs(got / want - 1) > tol:
            raise ValueError(
                f"E4 configuration: op split {op} {got:.4f} at w {w}, the PLAN pins {want:.4f} "
                f"(tolerance {100 * tol:.0f}%)"
            )


def precheck(plan: Mapping) -> list[dict]:
    """The simulation against both pinned tables. Returns one record per arm
    (config, arm, pinned, simulated, rel, ok)."""
    out: list[dict] = []
    shape = scaled_shape(plan)
    for name, pinned in round5_pct(plan).items():
        arm = arm_of(name, plan)
        got = simulate_arm(arm, shape, ROUND5_FPRIME, ROUND5_SPLIT)
        out.append(_record("round5", name, pinned, got.pct))
    for row in plan["arms"]:
        name = row["arm"]
        got_pct = simulate_plan_arm(plan, name).pct
        out.append(_record("e4", name, row["predicted_pct_of_C"], got_pct))
    return out


def _record(config: str, arm: str, pinned: float, simulated: float | None) -> dict:
    rel = None if simulated is None else simulated / pinned - 1
    return {
        "config": config,
        "arm": arm,
        "pinned_pct": pinned,
        "simulated_pct": None if simulated is None else round(simulated, 4),
        "rel": None if rel is None else round(rel, 5),
        "ok": rel is not None and abs(rel) <= ANCHOR_TOLERANCE,
        "label": "SIMULATED (as-built SCHEDULER-I17) against DERIVED pins",
    }


# ------------------------------------------------------------------ the sweep rule (F3)


def _line(pts: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Least squares y = a + b x."""
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx <= 0:
        raise ValueError("a sweep line needs two distinct cap' values")
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return my - b * mx, b


def sweep_threshold(points: Sequence[tuple[float, float, float]]) -> dict:
    """PLAN `scaled.sweep.threshold_rule`, the same on measured and simulated
    points: each point is (utilization, cap' MB/s, E % of C). One least-squares
    line E% = a + b/cap' through the points below S1 (utilization < 1), one
    through those above; the threshold is the cap' where they meet."""
    below = [(1 / cap, e) for u, cap, e in points if u < 1]
    above = [(1 / cap, e) for u, cap, e in points if u > 1]
    if len(below) < 3 or len(above) < 3:
        raise ValueError(
            f"the sweep needs >= 3 points on each side of S1, got {len(below)} and {len(above)}"
        )
    lo, hi = _line(below), _line(above)
    if hi[1] == lo[1]:
        raise ValueError("the two sweep lines are parallel: no threshold")
    x = (lo[0] - hi[0]) / (hi[1] - lo[1])
    if x <= 0:
        raise ValueError(f"the sweep lines meet at 1/cap' = {x:.6g}, not a positive cap'")
    return {"below": lo, "above": hi, "cap_MBps": 1 / x}


# ------------------------------------------------------------------ the training arms

# Every training arm runs the runtime itself (stage.py over the shared-pacer shim,
# TRANSPORT-I10) at the PLAN's scaled configuration, through the timing-only seams
# WANTRAIN_TEST_CHUNK_BYTES (f'), WANTRAIN_TEST_INTRA_LATENCY_MS (RTT/2 + 2f'/PCIe' on
# the intra hops) and, on stage 0, WANTRAIN_TEST_SLOT_PAD (scaled.stage0_padding); the
# controls add WANTRAIN_TEST_QUEUE=fifo or WANTRAIN_TEST_BARRIER=always.
FAST_MBPS = 1e5  # a neighbour stage's port (12.5 GB/s): never the bottleneck
HEAD_WIDTH = 4  # stage-0 arms: one Head peer per stage-0 peer, so B(i) is back within its slot
AHEAD = 4  # an interior stage's forwards ahead of its first B (peer_order)
TRAIN_STEPS = 5  # step 0 waits on nothing, step 1 warms up, E is the mean over steps 2..4
_SEAMS = (
    "WANTRAIN_TEST_CHUNK_BYTES",
    "WANTRAIN_TEST_INTRA_LATENCY_MS",
    "WANTRAIN_TEST_SLOT_PAD",
    "WANTRAIN_TEST_QUEUE",
    "WANTRAIN_TEST_BARRIER",
    "WANTRAIN_TEST_UPDATES",
    "WANTRAIN_SIM_SLOW",
)


@dataclass
class TrainingRun:
    """One training arm as the runtime runs it: the spec, its timing seams
    (environment), the measured stage and its peers' op order."""

    name: str
    arm: Arm
    spec: JobSpec
    env: dict[str, str]
    rank: int  # the measured stage
    ahead: int
    fprime_net: float  # the F' the network and stage 0's period were set from


def lpt_bounds(shapes: Mapping[str, Sequence[int]], w: int) -> dict[str, list[int]]:
    """Whole tensors, longest first onto the least loaded peer (ties: the
    earlier tensor, the lower peer), as explicit piece bounds: the owner's
    piece is the whole tensor, every other peer's is empty."""
    numel = {n: math.prod(s) for n, s in shapes.items()}
    pos = {n: i for i, n in enumerate(shapes)}
    load = [0] * w
    out = {}
    for n in sorted(shapes, key=lambda n: (-numel[n], pos[n])):
        k = load.index(min(load))
        load[k] += numel[n]
        out[n] = [0] * (k + 1) + [numel[n]] * (w - k)
    return out


def training_run(
    plan: Mapping,
    name: str,
    out_dir: str | Path,
    port_base: int,
    steps: int = TRAIN_STEPS,
    fprime_net: float | None = None,
) -> TrainingRun:
    """The runtime job of a PLAN training arm at the scaled configuration.
    Stage 0 arms: [Embed + 1 block at w 4, Head at HEAD_WIDTH (ZeRO-1 arms: at
    most its 3 tensors)], stage 0 padded
    per microbatch slot to C_B'. Interior arms: [Embed, 4 blocks at w, Head],
    the neighbours at width 1. The measured stage's peers are paced at cap'
    (egress and ingress, shared by every connection of the peer), the
    neighbours' ports at FAST_MBPS; the uplink arm puts the measured peers on
    the PLAN's two sites (uplink cap' each way) and the neighbours on a third.
    bf16 compute and boundary, recompute (B includes its forward, as the
    PLAN's profile), the planes codec, every manifest the reference's.
    `fprime_net`: the in-situ F' (a previous arm's) that cap', PCIe', the hop
    latency and C_B' are set from by the scaling map (PLAN section 4: the
    percentages stand at any F'); a sweep point keeps its utilization. Default:
    the PLAN's pinned F'."""
    arm = arm_of(name, plan)
    if arm.tree:
        raise ValueError(f"E4 arm {name}: the tree is F5's exchange harness, not a training arm")
    kind, n_blocks, w, mw = STAGES[arm.stage]
    shape = scaled_shape(plan)
    if shape.t_mb % shape.seq:
        raise ValueError(f"E4: T_mb' {shape.t_mb} is not a whole number of sequences {shape.seq}")
    fprime, split, hbm = plan_inputs(plan, arm.stage)
    net = fprime if fprime_net is None else fprime_net
    if arm.cap is not None:
        arm = replace(arm, cap=arm.cap * net / fprime)
    su = _setup(arm, shape, net, split, hbm)
    first = kind == "first"
    model = make_config("toy", compute_dtype="bf16", boundary_dtype="bf16")
    model.d_model, model.vocab_size, model.seq_len = shape.h, shape.vocab, shape.seq
    model.n_head, model.n_layer = plan["scaled"]["n_head"], n_blocks
    rank = 0 if first else 1
    last = 2 + n_blocks if not first else 3
    partition = [[0, 2], [2, 3]] if first else [[0, 1], [1, 1 + n_blocks], [1 + n_blocks, last]]
    zero1 = arm.layout not in ("pieces", "lpt")
    head = HEAD_WIDTH
    if first and zero1:
        # ZeRO-1 strides whole tensors over the peers: a Head of 3 trainable tensors
        # runs at most 3 wide (a peer with an empty stride cannot build its AdamW).
        shapes = JobSpec(model=model, stages=2, partition=partition).stage_param_shapes()
        head = min(HEAD_WIDTH, len(shapes[1]))
    widths = [w, head] if first else [1, w, 1]
    port = [[FAST_MBPS] * x for x in widths]
    port[rank] = [su.nic * 8 / 1e6] * w
    sites = [["site0"] * x for x in widths]
    if arm.sites:
        sites = [["edge"] * x for x in widths]
        sites[rank] = list(arm.sites)
    emission = ["store"] * len(widths)
    emission[rank] = "forward" if arm.emission == "forward" else "store"
    spec = JobSpec(
        model=model,
        steps=steps,
        microbatches=w * mw,
        mb_size=shape.t_mb // shape.seq,
        recompute=True,
        stages=len(widths),
        stage_widths=widths,
        partition=partition,
        wire="fp32" if arm.wire_fp32 else "native",
        shard="zero1" if zero1 else "pieces",
        stream_grads=True,
        wire_codec="planes",
        emission_order=emission,
        pacing="shared",
        peer_egress_mbps=port,
        peer_ingress_mbps=[list(r) for r in port],
        sites=sites,
        out_dir=str(out_dir),
        tag="e4",
        port_base=port_base,
        dial_timeout_s=60,
    )
    pair = set(arm.sites or ())
    spec.uplink_mbps = {
        f"{a}->{b}": su.port * 8 / 1e6 if {a, b} <= pair else FAST_MBPS
        for a, b in spec.site_pairs()
    }
    if arm.layout == "lpt":
        spec.piece_bounds = [None] * len(widths)
        spec.piece_bounds[rank] = lpt_bounds(spec.stage_param_shapes()[rank], w)
    spec.finalize()
    ahead = 1 if first else AHEAD
    spec.schedule_order[rank] = [
        [[op, spec.owned(rank, p)[i]] for op, i in peer_order(mw, ahead)] for p in range(w)
    ]
    for p in range(w):
        if len(spec.owned(rank, p)) != mw:
            raise ValueError(f"E4 arm {name}: peer {p} owns {spec.owned(rank, p)}, not M/w {mw}")
    spec.finalize()
    hop = arm.rtt / 2 + 2 * su.chunk / su.pcie
    env = {
        "WANTRAIN_TEST_CHUNK_BYTES": str(su.chunk),
        "WANTRAIN_TEST_INTRA_LATENCY_MS": repr(hop * 1e3),
    }
    if first:
        tau, cb, cw = su.period / mw, su.c * su.split[1], su.c * su.split[2]
        env["WANTRAIN_TEST_SLOT_PAD"] = f"0:{tau * 1e3!r}:{cb * 1e3!r}:{cw * 1e3!r}"
    if arm.policy == "fifo":
        env["WANTRAIN_TEST_QUEUE"] = "fifo"
    if arm.barrier:
        env["WANTRAIN_TEST_BARRIER"] = "always"
    return TrainingRun(name, arm, spec, env, rank, ahead, net)


def training_configuration(plan: Mapping, run: TrainingRun) -> dict:
    """What a training arm runs, for check_configuration; the stages other
    than the measured one carry the PLAN's values (each arm measures one)."""
    s, m = run.spec, run.spec.model
    cfg = {
        "h": m.d_model,
        "V": m.vocab_size,
        "seq": m.seq_len,
        "T_mb": s.mb_size * m.seq_len,
        "n_head": m.n_head,
        "intra_op_threads": 1,  # runtime.determinism.setup_determinism
        "chunk_bytes": int(run.env["WANTRAIN_TEST_CHUNK_BYTES"]),
        "fold_paced": False,
    }
    stages = {st: {"w": w, "M_over_w": mw} for st, (_k, _nb, w, mw) in STAGES.items()}
    w = s.stage_widths[run.rank]
    stages[run.arm.stage] = {"w": w, "M_over_w": s.microbatches // w}
    return {**cfg, "stages": stages}


@contextmanager
def _environ(env: Mapping[str, str]):
    """The run's seams, exactly: every other timing seam is unset."""
    saved = {k: os.environ.get(k) for k in _SEAMS}
    try:
        for k in _SEAMS:
            os.environ.pop(k, None)
        os.environ.update(env)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _events(spec: JobSpec, rank: int, peer: int) -> list[dict]:
    path = os.path.join(spec.out_dir, f"{spec.tag}_rank{rank}p{peer}_telemetry.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _dur(e: Mapping) -> float:
    return float(e["t1"]) - float(e["t0"])


def peer_timing(events: Sequence[Mapping], steps: int) -> dict:
    """One measured peer's timeline from its telemetry events, per step t:
    `first_f`, when its first forward began; `gather`, the seconds its
    forwards waited for gathers (_await_layer's stall events); `end`, when its
    last op ended. Also the op times of steps 2.. (F net of its gather waits,
    B, W), per op the list."""
    first_f = [math.inf] * steps
    end = [-math.inf] * steps
    gather = [0.0] * steps
    stalls = [e for e in events if e["ev"] == "stall" and e.get("wait") == "gather"]
    for e in stalls:
        gather[e["step"]] += _dur(e)
    ops: dict[str, list[float]] = {"F": [], "B": [], "W": []}
    for e in events:
        op = {"fwd": "F", "bwd_b": "B", "bwd_w": "W"}.get(e["ev"])
        if op is None:
            continue
        t = e["step"]
        end[t] = max(end[t], float(e["t1"]))
        if op == "F":
            first_f[t] = min(first_f[t], float(e["t0"]))
        if t < 2:
            continue
        x = _dur(e)
        if op == "F":
            x -= sum(
                _dur(g)
                for g in stalls
                if g["step"] == t and e["t0"] <= g["t0"] and g["t1"] <= e["t1"]
            )
        ops[op].append(x)
    if not all(map(math.isfinite, first_f + end)):
        raise ValueError("a measured step has no forward or no op on this peer")
    return {"first_f": first_f, "gather": gather, "end": end, "ops": ops}


def stage_late(peers: Sequence[Mapping]) -> list[float]:
    """SCHEDULER-I17's E per step t >= 1 (exchange_sim `delay`): how late the
    stage's step-t forward is against step t - 1's stage end (the latest peer
    end; a peer waiting for a slower one is the lane bubble, in the period,
    never exposure), on the latest peer: max over peers of first_f + gather
    waits - stage end, at least 0. A barrier kept, a ZeRO-1 optimizer step or
    a gather wait all delay first_f or add to the waits. Step 0 is 0."""
    steps = len(peers[0]["end"])
    late = [0.0] * steps
    for t in range(1, steps):
        due = max(p["end"][t - 1] for p in peers)
        late[t] = max(0.0, max(p["first_f"][t] + p["gather"][t] - due for p in peers))
    return late


def late_parts(peers: Sequence[Mapping]) -> list[tuple[float, float]]:
    """stage_late's two parts per step t >= 1 (step 0: zeros), on the peer
    that sets it: (its first forward's start against step t - 1's stage end,
    its gather waits). Diagnostic only: under pieces without the barrier the
    exchange delays a forward through the gather waits, so a late start is
    the forward's input arriving late (the neighbour stage's turn-around), a
    kept barrier or a ZeRO-1 step, none of which an arm's I17 simulation of
    the stage alone has."""
    steps = len(peers[0]["end"])
    out = [(0.0, 0.0)] * steps
    for t in range(1, steps):
        due = max(p["end"][t - 1] for p in peers)
        p = max(peers, key=lambda p: p["first_f"][t] + p["gather"][t])
        out[t] = (p["first_f"][t] - due, p["gather"][t])
    return out


def _median(xs: Sequence[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        raise ValueError("no op events to take a median of")
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def measure_training(
    plan: Mapping,
    name: str,
    out_dir: str | Path,
    port_base: int,
    ref_dir: str | Path,
    steps: int = TRAIN_STEPS,
    fprime_net: float | None = None,
) -> dict:
    """Run one training arm and measure it. -> the measured E (% of C), its
    I17 simulation at the in-situ configuration (the op times the run
    measured, the network and stage 0's period as set) and the PLAN's pinned
    prediction, the in-situ op split (None in `off_configuration` when within
    the PLAN's tolerance), and whether the state equals the reference's."""
    if steps < 3:
        raise ValueError(f"E4 needs a warm-up and at least one measured step, got steps={steps}")
    run = training_run(plan, name, out_dir, port_base, steps, fprime_net)
    check_configuration(plan, training_configuration(plan, run))
    spec = run.spec
    ref = ensure_reference(JobSpec.from_json(spec.to_json()), str(ref_dir))
    kind, _nb, w, mw = STAGES[run.arm.stage]
    shape = scaled_shape(plan)
    _pinned, split, hbm = plan_inputs(plan, run.arm.stage)
    su = _setup(run.arm, shape, run.fprime_net, split, hbm)
    timeout = 120 + steps * su.period * 4
    with _environ(run.env):
        res = run_local(spec, timeout=timeout)
    if not res["ok"]:
        raise RuntimeError(f"E4 arm {name}: the run failed {res} (logs under {spec.out_dir})")
    state = distributed_manifest(spec)
    with open(reference_state_path(ref), encoding="utf-8") as fh:
        want = json.load(fh)
    diff = diff_manifests(state, want)
    peers = [peer_timing(_events(spec, run.rank, p), steps) for p in range(w)]
    med = [[_median(t["ops"][op]) for op in "FBW"] for t in peers]
    ops = [sum(m[i] for m in med) / w for i in range(3)]
    c = sum(ops)
    split_in = (ops[0] / c, ops[1] / c, ops[2] / c)
    flops = sum(u.flops for u in su.units)
    fprime_in = 8 * flops * shape.t_mb / c
    off = None
    try:
        check_split(plan, w, split_in)
        check_fprime(plan, run.fprime_net, fprime_in)
    except ValueError as e:
        off = str(e)
    sim = simulate_arm(
        run.arm, shape, fprime_in, split_in, hbm, net_fprime=run.fprime_net, ahead=run.ahead
    )
    late = stage_late(peers)
    e = sum(late[2:]) / (steps - 2)
    parts = late_parts(peers)
    return {
        "arm": name,
        "pct": 100 * e / sim.period,
        "exposure_s": e,
        "period_s": sim.period,
        "per_step_pct": [round(100 * x / sim.period, 4) for x in late],
        "per_step_start_gather_pct": [
            [round(100 * x / sim.period, 4) for x in pair] for pair in parts
        ],
        "predicted_insitu_pct": sim.pct,
        "pinned_pct": {a["arm"]: a["predicted_pct_of_C"] for a in plan["arms"]}[name],
        "c_s_ms": round(c * 1e3, 3),
        "c_s_set_ms": round(su.c * 1e3, 3),
        "fprime_insitu": fprime_in,
        "fprime_net": run.fprime_net,
        "cap_MBps": su.nic / 1e6,
        "split_insitu": [round(x, 4) for x in split_in],
        "off_configuration": off,
        "identity": not diff and state == want,
        "manifest_diff": diff,
        "manifest_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
        "wall_s": res["wall_seconds"],
        "seams": dict(run.env),
    }


FINGERPRINT_FILES = (
    "src/wantrain/validation/exposure.py",
    "src/wantrain/scheduler/exchange_sim.py",
    "src/wantrain/runtime/stage.py",
    "src/wantrain/runtime/exchange.py",
    "src/wantrain/transport/shim.py",
    "docs/experiments/m2b-e4/PLAN.json",
)


def tree_fingerprint() -> str:
    """sha256 over the files an E4 result depends on: a result line recorded
    under another fingerprint is evidence for another tree, never this one."""
    root = Path(__file__).resolve().parents[3]
    h = hashlib.sha256()
    for rel in FINGERPRINT_FILES:
        h.update(rel.encode() + b"\0" + (root / rel).read_bytes())
    return h.hexdigest()


def in_band(measured: float, predicted: float) -> bool:
    """F3's band: measured within [0.8, 1.2]x its simulation."""
    return predicted > 0 and 0.8 <= measured / predicted <= 1.2


# E4's gate (HANDOUT-M2b D-3, the human's decision 2026-10-10: on the Mac, measure, do not
# block). The checks that move or certify bits, or that ADR-0006 rests on, fail on every
# machine: a missing or invalid measurement, a state that is not the reference's, F6, F5 and
# BA-43's emission order at RTT_hi. F3's band [0.8, 1.2]x (and the sweep threshold within
# 25%) and an arm off the PLAN's configuration (interior w8, M2B-E4-W8-OFFCONFIG) are
# timing-model checks this CPU cannot hold to the PLAN's hardware: with the hardware switch at
# gpu false they are measured and printed, never failing; with gpu true (M1) they fail exactly
# as before. Only the switch decides; no other input can turn an enforced check into a
# measured one, so this is never a mask.
E4_ENFORCED = ("measurement", "identity", "f6", "f5", "emission_order")
E4_MEASURED_WITHOUT_GPU = ("f3_band", "off_configuration")


def e4_gate_fails(check: str, ok: bool, gpu: bool) -> bool:
    """Whether a check that came out `ok` fails E4's gate under the hardware
    switch `gpu` (anything but a literal true reads as false upstream)."""
    if check not in E4_ENFORCED + E4_MEASURED_WITHOUT_GPU:
        raise ValueError(f"E4 gate: unknown check {check!r}")
    if ok:
        return False
    return gpu is True or check in E4_ENFORCED


def f3_band_mode(gpu: bool) -> str:
    """The field an F3 result line carries: whether its band decides."""
    if gpu is True:
        return "enforced (gpu true)"
    return "measured-not-enforced (gpu false, HANDOUT-M2b D-3)"


def measurement_problems(line: Mapping) -> list[str]:
    """Why a result line is not a valid measurement ([] when it is): every
    field the classification reads present, predicted, measured and ratio
    finite numbers, predicted > 0 and measured, ratio >= 0 (E is at least 0)."""
    bad = [
        f"no {k}"
        for k in ("arm", "predicted", "measured", "ratio", "unit", "label")
        if k not in line
    ]
    for k in ("predicted", "measured", "ratio"):
        if k not in line:
            continue
        v = line[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            bad.append(f"{k} is not a finite number: {v!r}")
        elif v < 0 or (k == "predicted" and v == 0):
            bad.append(f"{k} out of range: {v!r}")
    return bad


def f6_fractions(plan: Mapping, measured: Mapping[str, float]) -> dict[str, float]:
    """PLAN section 5: per F6 arm, the measured margin E(alternative) -
    E(pieces_fwd), % of C, over the predicted margin (twice the threshold).
    `measured`: arm name -> % of C."""
    out = {}
    for row in plan["arms"]:
        if row.get("f6_min_margin_pct") is None:
            continue
        alt, vs = row["arm"], row["f6_vs"]
        stage, kind = alt.split(".")
        key = f"f6_{stage}_vs_{kind}_fraction"
        if key not in plan["expected"]:
            raise ValueError(f"E4 PLAN: no expected metric {key!r} for the F6 arm {alt}")
        out[key] = (measured[alt] - measured[vs]) / (2 * row["f6_min_margin_pct"])
    return out


def missing_arms(plan: Mapping, lines: Mapping[str, Mapping]) -> list[str]:
    """The result lines `summary_line` needs that `lines` lacks."""
    need = set(SUMMARY_ARMS)
    for r in plan["arms"]:
        if r.get("f6_vs"):
            need |= {r["arm"], r["f6_vs"]}
    return sorted(need - set(lines))


def summary_line(plan: Mapping, lines: Mapping[str, Mapping]) -> dict:
    """PLAN section 5's last line from every arm's result line (by arm name):
    the F6 fractions and their minimum, f3_violations (the F3 arms outside
    the band, plus a sweep threshold more than 25% off), f5_violations and
    emission_order_violations; `off_configuration` names every arm whose line
    ran off the PLAN's configuration (its numbers above are not evidence).
    Raises naming the arms not measured."""
    missing = missing_arms(plan, lines)
    if missing:
        raise ValueError(f"E4 summary: arms not measured: {missing}")
    fr = f6_fractions(plan, {k: v["measured"] for k, v in lines.items()})
    f3 = sum(1 for k, v in lines.items() if v.get("f3") and not v["in_f3_band"])
    f3 += int(not lines["sweep.threshold"]["within_25pct"])
    out = {
        "f6_margin_fraction_min": min(fr.values()),
        **fr,
        "f3_violations": f3,
        "f5_violations": int(lines["interior_w4.tree"]["fired"]),
        "emission_order_violations": int(not lines["interior_w4.emission.rtt_2cw"]["agrees"]),
        "off_configuration": sorted(k for k, v in lines.items() if v.get("off_configuration")),
        "label": LABEL_TRAINING,
    }
    if set(out) - {"label", "off_configuration"} != set(plan["expected"]):
        raise ValueError(
            f"E4 summary keys {sorted(out)} are not the PLAN's {sorted(plan['expected'])}"
        )
    return out


# The result lines the summary reads besides the F6 arms: the F3 judges (every sweep
# point and the threshold, both uplink arms, the barrier controls, the emission arms),
# F5's decision and BA-43's order at RTT_hi.
SUMMARY_ARMS = (
    *(f"sweep.util_{u}" for u in ("0.5", "0.7", "0.9", "1.1", "1.3", "1.6", "2.0")),
    "sweep.threshold",
    "interior_w4.uplink_pieces_fwd",
    "interior_w4.uplink_pieces_bwd",
    "stage0.barrier_pieces_fwd",
    "interior_w4.barrier_pieces_fwd",
    "interior_w8.barrier_pieces_fwd",
    *(
        f"{s}.emission_{o}.{r}"
        for s in ("interior_w4", "stage0")
        for o in ("fwd", "bwd")
        for r in ("rtt0", "rtt_2cw")
    ),
    "interior_w4.emission.rtt_2cw",
    "interior_w4.tree",
)


# ------------------------------------------------------------------ F5: the exchange harness

_HDR = struct.Struct(
    "<BHIiII"
)  # kind, unit, step, mb (a partial: its sender; a gather: -1), offset, length
_WAIT_S = 120.0  # a gather or a peer this late is a stalled harness, never a measurement
_DTYPES = {2: torch.bfloat16, 4: torch.float32}
LABEL = (
    "SIMULATED (gpu false: localhost processes, shared-pacer shim, compute at the PLAN's op times)"
)
LABEL_TRAINING = (
    "SIMULATED (gpu false: the runtime's localhost processes on the shared-pacer shim, real "
    "bf16 compute on this CPU)"
)


@dataclass(frozen=True)
class HarnessRun:
    """One F5 run: the w peers of one stage. Compute is the PLAN's op times
    (sleeps: the harness computes no gradient and holds no training state);
    every exchange byte is real and crosses the shim: each peer's egress and
    ingress SharedBudget at cap', chunks of at most f' served need-first by the
    runtime's NeedFirstQueue, the one-way delay 2f'/PCIe' after the ingress.
    An owner folds on arrival, on its own thread, as SCHEDULER-I17 has it."""

    w: int
    mw: int
    params: tuple[int, ...]  # elements per unit, in forward order
    b_g: tuple[int, ...]  # bytes per gradient element on the wire (a tree partial: 4)
    b_p: tuple[int, ...]  # bytes per gathered element
    flops: tuple[float, ...]
    ops: tuple[float, float, float]  # c_F, c_B, c_W (s)
    tau: float  # one microbatch's slot: c_s, or C_B'/(M/w) on stage 0
    mbps: float  # cap' (Mbit/s)
    hop_ms: float
    chunk: int
    boundary: int
    tree: bool
    steps: int
    budgets: tuple[tuple[str, str], ...]  # per peer: (egress, ingress) budget file
    token: bytes


def harness_configuration(plan: Mapping) -> dict:
    """What the F5 harness runs, for check_configuration: the PLAN's shapes
    size its units, f' comes from the scaling map, one intra-op thread, the
    fold not paced, the PLAN's widths."""
    shape = scaled_shape(plan)
    return {
        "h": shape.h,
        "V": shape.vocab,
        "seq": shape.seq,
        "T_mb": shape.t_mb,
        "n_head": plan["scaled"]["n_head"],
        "intra_op_threads": 1,
        "chunk_bytes": max(1, int(CHUNK_B * (shape.h / H_B) ** 2)),
        "fold_paced": False,
        "stages": {s: {"w": w, "M_over_w": mw} for s, (_k, _nb, w, mw) in STAGES.items()},
    }


def measure_exchange(
    plan: Mapping, stage: str, tree: bool, steps: int = 4, timeout_s: float = 600.0
) -> dict:
    """F5: one stage's exchange, measured on the harness: the streaming left
    fold of every microbatch (`tree` False, I17's pieces arm) or the
    residue-group tree (each peer's fp32 partial per unit piece, released as its
    last W produces the unit). E per step is the largest D over the peers: how
    late the step's first forward is, accumulated over the units in forward
    order (SCHEDULER-I17). Step 0 waits on nothing and step 1 is the warm-up;
    E is the mean over steps 2..`steps`. Also returns the I17 simulation of
    the same arm."""
    if steps < 3:
        raise ValueError(f"F5 needs a warm-up and at least one measured step, got steps={steps}")
    check_configuration(plan, harness_configuration(plan))
    arm = Arm(stage, tree=tree)
    fprime, split, hbm = plan_inputs(plan, stage)
    shape = scaled_shape(plan)
    su = _setup(arm, shape, fprime, split, hbm)
    sim = simulate_arm(arm, shape, fprime, split, hbm)
    for u in su.units:
        if u.params % su.w:
            raise ValueError(f"unit of {u.params} elements does not split in {su.w} pieces")
    tmp = tempfile.mkdtemp(prefix="e4-f5-")
    try:
        run = HarnessRun(
            w=su.w,
            mw=su.mw,
            params=tuple(u.params for u in su.units),
            b_g=tuple(u.b_g for u in su.units),
            b_p=tuple(u.b_p for u in su.units),
            flops=tuple(u.flops for u in su.units),
            ops=(su.c * su.split[0], su.c * su.split[1], su.c * su.split[2]),
            tau=su.period / su.mw,
            mbps=su.port * 8 / 1e6,
            hop_ms=2 * su.chunk / su.pcie * 1e3,
            chunk=su.chunk,
            boundary=su.boundary,
            tree=tree,
            steps=steps,
            budgets=tuple(
                (os.path.join(tmp, f"egress{q}"), os.path.join(tmp, f"ingress{q}"))
                for q in range(su.w)
            ),
            token=os.urandom(16),
        )
        late = _launch(run, timeout_s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    per_step = [max(d[t] for d in late) for t in range(steps + 1)]
    measured = per_step[2:]
    e = sum(measured) / len(measured)
    return {
        "stage": stage,
        "tree": tree,
        "exposure_s": e,
        "pct": 100 * e / su.period,
        "per_step_pct": [round(100 * x / su.period, 4) for x in per_step],
        "per_peer_s": [[round(x, 5) for x in d] for d in late],
        "predicted_pct": sim.pct,
        "period_s": su.period,
    }


def arm_line(arm: str, predicted: float, measured: float, criterion: str, **extra) -> dict:
    """One result line per arm, the fields the /wan-experiment classification
    reads (arm, predicted, measured, unit, criterion), labelled SIMULATED."""
    return {
        "arm": arm,
        "predicted": round(predicted, 4),
        "measured": round(measured, 4),
        "unit": "% of C",
        "criterion": criterion,
        "label": LABEL,
        **extra,
    }


def _expect(conn, kind: str, deadline: float):
    left = deadline - time.monotonic()
    if left <= 0 or not conn.poll(left):
        raise TimeoutError(f"E4 F5 harness: no {kind!r} from a peer within the run's bound")
    try:
        msg = conn.recv()
    except EOFError:
        raise RuntimeError(f"E4 F5 harness: a peer exited before {kind!r}") from None
    if msg[0] == "error":
        raise RuntimeError(f"E4 F5 harness failed: {msg[1]}")
    if msg[0] != kind:
        raise RuntimeError(f"E4 F5 harness: expected {kind!r} from a peer, got {msg[0]!r}")
    return msg[1] if len(msg) > 1 else None


def _launch(run: HarnessRun, timeout_s: float) -> list[list[float]]:
    """Run the w peer processes; -> each peer's D per step."""
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    deadline = time.monotonic() + timeout_s
    try:
        for q in range(run.w):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_harness_peer, args=(run, q, child), daemon=True)
            p.start()
            child.close()
            conns.append(parent)
            procs.append(p)
        ports = [_expect(c, "port", deadline) for c in conns]
        for c in conns:
            c.send(("ports", ports))
        for c in conns:
            _expect(c, "connected", deadline)
        start = time.monotonic() + 1.0
        for c in conns:
            c.send(("start", start))
        late = [_expect(c, "done", deadline) for c in conns]
        for c in conns:
            c.send(("stop",))
        for c in conns:
            _expect(c, "exit", deadline)
        return late
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        for c in conns:
            c.close()


def _harness_peer(run: HarnessRun, q: int, conn) -> None:
    peer = None
    try:
        peer = _Peer(run, q, conn)
        peer.main()
    except BaseException as e:
        try:
            conn.send(("error", f"peer {q}: {type(e).__name__}: {e}\n{traceback.format_exc()}"))
        except OSError:
            pass
    finally:
        if peer is not None:
            peer.close()


def _recv_exact(sock: socket.socket, n: int) -> bytearray | None:
    """n bytes, or None on a clean end of stream before the first byte."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:])
        if k == 0:
            if got == 0:
                return None
            raise ConnectionError("connection closed mid-message")
        got += k
    return buf


def _sleep_until(t: float) -> None:
    dt = t - time.monotonic()
    if dt > 0:
        time.sleep(dt)


class _Peer:
    """One harness peer: the compute timeline on the main thread, a sender and
    a reader per remote peer, an exchange thread (reassembly, folds, AdamW,
    gathers) and two boundary pacers."""

    def __init__(self, run: HarnessRun, q: int, conn):
        torch.set_num_threads(1)
        self.run, self.q, self.conn = run, q, conn
        self.n = [p // run.w for p in run.params]
        self.egress = SharedBudget(run.mbps, run.budgets[q][0])
        self.ingress = SharedBudget(run.mbps, run.budgets[q][1])
        self.inbox: queue.Queue = queue.Queue()
        self.cv = threading.Condition()
        self.ready = {(0, u): run.w for u in range(len(run.params))}
        self.ready_at = {(0, u): 0.0 for u in range(len(run.params))}
        self.err: str | None = None
        self.sendq = {r: NeedFirstQueue() for r in range(run.w) if r != q}
        self.bq: dict[str, queue.Queue] = {"out": queue.Queue(), "in": queue.Queue()}
        self.seq = itertools.count()
        self.parts: dict = {}
        self.folds: dict = {}
        self.opt: dict = {}
        self.socks: dict[int, socket.socket] = {}
        self.threads: list[threading.Thread] = []
        self.grad = [bytes(b * n) for b, n in zip(run.b_g, self.n)]
        self.own = [torch.zeros(n, dtype=_DTYPES[b]) for b, n in zip(run.b_g, self.n)]

    def main(self) -> None:
        self._connect()
        for r, s in self.socks.items():
            self._spawn(self._sender, s, self.sendq[r])
            self._spawn(self._reader, s, r)
        self._spawn(self._pace_boundary, self.bq["out"], self.egress)
        self._spawn(self._pace_boundary, self.bq["in"], self.ingress)
        self._spawn(self._exchange)
        self.conn.send(("connected",))
        late = self._timeline(self._recv("start"))
        self.conn.send(("done", late))
        self._recv("stop")
        self._shutdown()
        self.conn.send(("exit",))

    def close(self) -> None:
        for s in self.socks.values():
            s.close()
        self.egress.close()
        self.ingress.close()

    def _recv(self, kind: str):
        msg = self.conn.recv()
        if msg[0] != kind:
            raise RuntimeError(
                f"peer {self.q}: expected {kind!r} from the launcher, got {msg[0]!r}"
            )
        return msg[1] if len(msg) > 1 else None

    def _connect(self) -> None:
        """Every pair once: dial the lower peers, accept the higher ones, each
        proving the run's token (a stray local connection is fatal)."""
        run, q, w = self.run, self.q, self.run.w
        lst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            lst.bind(("127.0.0.1", 0))
            lst.listen(w)
            lst.settimeout(_WAIT_S)
            self.conn.send(("port", lst.getsockname()[1]))
            ports = self._recv("ports")
            for r in range(q):
                s = socket.create_connection(("127.0.0.1", ports[r]), timeout=_WAIT_S)
                self.socks[r] = s
                s.sendall(run.token + struct.pack("<I", q))
            while len(self.socks) < w - 1:
                s, _addr = lst.accept()
                s.settimeout(_WAIT_S)
                hello = _recv_exact(s, len(run.token) + 4)
                r = struct.unpack("<I", bytes(hello[-4:]))[0] if hello is not None else -1
                if (
                    hello is None
                    or not hmac.compare_digest(bytes(hello[:-4]), run.token)
                    or not q < r < w
                    or r in self.socks
                ):
                    s.close()
                    raise ConnectionError(f"peer {q}: refused a connection that is not a peer")
                self.socks[r] = s
        finally:
            lst.close()
        for s in self.socks.values():
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _spawn(self, fn, *args) -> None:
        def guarded():
            try:
                fn(*args)
            except BaseException as e:
                with self.cv:
                    self.err = self.err or f"{fn.__name__}: {type(e).__name__}: {e}"
                    self.cv.notify_all()

        t = threading.Thread(target=guarded, daemon=True)
        t.start()
        self.threads.append(t)

    # ---------------- wire

    def _sender(self, sock: socket.socket, q: NeedFirstQueue) -> None:
        pacer = SenderPacer(LinkSpec(egress=self.egress))
        while True:
            item = q.pop()
            if item is None:
                sock.shutdown(socket.SHUT_WR)
                return
            hdr, part = item
            pacer.pace(len(hdr) + len(part))
            sock.sendall(hdr)
            sock.sendall(part)

    def _reader(self, sock: socket.socket, r: int) -> None:
        delay = DelayedDelivery(
            LinkSpec(latency_ms=self.run.hop_ms, ingress=self.ingress), self.inbox.put
        )
        while True:
            hdr = _recv_exact(sock, _HDR.size)
            if hdr is None:
                return
            kind, unit, step, mb, off, ln = _HDR.unpack(hdr)
            total = self._check(r, kind, unit, step, mb, off, ln)
            body = _recv_exact(sock, ln)
            if body is None:
                raise ConnectionError(f"peer {self.q}: intra{r} ended inside a chunk")
            delay.push(("x", r, kind, unit, step, mb, off, total, body), nbytes=_HDR.size + ln)

    def _check(self, r: int, kind: int, unit: int, step: int, mb: int, off: int, ln: int) -> int:
        """A received chunk header, validated before its body is read."""
        run = self.run
        if unit >= len(run.params) or step > run.steps or not 0 < ln <= run.chunk:
            raise ValueError(f"peer {self.q}: bad chunk from intra{r}: unit {unit} step {step}")
        if kind == GATHER:
            total, ok = run.b_p[unit] * self.n[unit], mb == -1
        elif kind == GRAD:
            total = run.b_g[unit] * self.n[unit]
            ok = mb == r if run.tree else (0 <= mb < run.w * run.mw and mb % run.w == r)
        else:
            raise ValueError(f"peer {self.q}: chunk of unknown kind {kind} from intra{r}")
        if not ok or off % run.chunk or off + ln > total:
            raise ValueError(
                f"peer {self.q}: chunk from intra{r} off its piece (mb {mb}, off {off})"
            )
        return total

    def _push(self, dest: int, kind: int, unit: int, step: int, mb: int, payload) -> None:
        view = memoryview(payload).cast("B")
        for off in range(0, len(view), self.run.chunk):
            part = view[off : off + self.run.chunk]
            key = NeedKey(unit, kind, step, mb, next(self.seq))
            self.sendq[dest].push(key, (_HDR.pack(kind, unit, step, mb, off, len(part)), part))

    def _pace_boundary(self, q: queue.Queue, budget: SharedBudget) -> None:
        """Pipeline-boundary bytes: they load this peer's port, in chunks of f'."""
        while True:
            nb = q.get()
            if nb is None:
                return
            for off in range(0, nb, self.run.chunk):
                budget.pace(min(self.run.chunk, nb - off))

    # ---------------- exchange: reassembly, left fold, AdamW, gathers

    def _exchange(self) -> None:
        while True:
            m = self.inbox.get()
            if m is None:
                return
            if m[0] == "local":
                _tag, unit, step, idx, g = m
                self._contribute(unit, step, idx, g)
                continue
            _tag, r, kind, unit, step, mb, off, total, body = m
            key = (r, kind, unit, step, mb)
            got = self.parts.get(key)
            if got is None:
                got = self.parts[key] = (bytearray(total), set())
            if off in got[1]:
                raise ValueError(f"peer {self.q}: duplicate chunk {key} at {off}")
            got[0][off : off + len(body)] = body
            got[1].add(off)
            if len(got[1]) < -(-total // self.run.chunk):
                continue
            del self.parts[key]
            if kind == GATHER:
                self._install(step, unit)
            else:
                g = torch.frombuffer(got[0], dtype=_DTYPES[self.run.b_g[unit]])
                self._contribute(unit, step, mb, g)

    def _contribute(self, unit: int, step: int, idx: int, g: torch.Tensor) -> None:
        """Left fold in canonical order: microbatches (left fold) or senders
        (tree), each fold as soon as the previous one and its contribution are there."""
        run = self.run
        f = self.folds.get((step, unit))
        if f is None:
            f = self.folds[(step, unit)] = [torch.zeros(self.n[unit]), 0, {}]
        f[2][idx] = g
        while f[1] in f[2]:
            f[0].add_(f[2].pop(f[1]))
            f[1] += 1
        if f[1] == (run.w if run.tree else run.w * run.mw):
            del self.folds[(step, unit)]
            self._update(step, unit, f[0])

    def _update(self, step: int, unit: int, g: torch.Tensor) -> None:
        st = self.opt.get(unit)
        if st is None:
            st = self.opt[unit] = [torch.zeros(self.n[unit]) for _ in range(3)]
        p, m, v = st
        m.mul_(0.9).add_(g, alpha=0.1)
        v.mul_(0.95).addcmul_(g, g, value=0.05)
        p.addcdiv_(m, v.sqrt().add_(1e-8), value=-1e-3)
        wire = p.to(_DTYPES[self.run.b_p[unit]]).view(torch.uint8).numpy().tobytes()
        for d in self.sendq:
            self._push(d, GATHER, unit, step + 1, -1, wire)
        self._install(step + 1, unit)

    def _install(self, step: int, unit: int) -> None:
        with self.cv:
            n = self.ready[(step, unit)] = self.ready.get((step, unit), 0) + 1
            if n == self.run.w:
                self.ready_at[(step, unit)] = time.monotonic()
                self.cv.notify_all()

    def _await(self, step: int, unit: int) -> float:
        """Block until every piece of the unit serves `step`; -> when the last
        one was installed (the compute thread's own wake-up is not exposure)."""
        deadline = time.monotonic() + _WAIT_S
        with self.cv:
            while self.ready.get((step, unit), 0) < self.run.w:
                if self.err is not None:
                    raise RuntimeError(f"peer {self.q}: {self.err}")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"peer {self.q}: step {step} unit {unit}: gathers missing")
                self.cv.wait(min(left, 1.0))
            return self.ready_at[(step, unit)]

    # ---------------- compute timeline

    def _timeline(self, t0: float) -> list[float]:
        """Microbatch i of this peer (global i*w + q) in its slot [i tau, (i+1)
        tau): F at the start, W ending it, B before W (scaled.stage0_padding;
        tau = c_s on an interior stage). Step t+1 is due one period after step
        t's origin; its first F waits unit by unit for the gathers. -> D per step."""
        run = self.run
        cf, _cb, cw = run.ops
        pfl = sum(run.flops)
        late = []
        s = t0
        for t in range(run.steps + 1):
            cursor, d = s, 0.0
            for u in range(len(run.params)):
                rdy = self._await(t, u)
                if rdy > cursor:
                    d += rdy - cursor
                    cursor = rdy
                cursor += cf * run.flops[u] / pfl
                _sleep_until(cursor)
            late.append(d)
            if t == run.steps:
                return late
            o = s + d
            for i in range(run.mw):
                slot = o + i * run.tau
                if i:
                    _sleep_until(slot + cf)
                if run.boundary:
                    self.bq["out"].put(run.boundary)
                    self.bq["in"].put(run.boundary)
                w_start = slot + run.tau - cw
                _sleep_until(w_start)
                self._w(t, i, w_start, w_start + cw, pfl)
            s = o + run.mw * run.tau
        return late

    def _w(self, t: int, i: int, ws: float, we: float, pfl: float) -> None:
        """W releases each unit as it computes it, in forward order (a unit
        without W compute, the Embed, when W starts): to every owner of a
        piece, the own piece locally. The tree releases only the last W's,
        each peer's fp32 partial."""
        run = self.run
        if run.tree and i < run.mw - 1:
            _sleep_until(we)
            return
        idx = self.q if run.tree else i * run.w + self.q
        cum = 0.0
        for u in range(len(run.params)):
            cum += run.flops[u]
            _sleep_until(ws + (we - ws) * cum / pfl)
            for d in self.sendq:
                self._push(d, GRAD, u, t, idx, self.grad[u])
            self.inbox.put(("local", u, t, idx, self.own[u]))
        _sleep_until(we)

    def _shutdown(self) -> None:
        for q in self.sendq.values():
            q.put(None)
        for bq in self.bq.values():
            bq.put(None)
        self.inbox.put(None)
        for t in self.threads:
            t.join(timeout=_WAIT_S)
        if self.err is not None:
            raise RuntimeError(f"peer {self.q}: {self.err}")
