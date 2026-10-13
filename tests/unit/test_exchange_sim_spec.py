"""SCHEDULER-I17 and SCHEDULER-I16 clauses of the exchange simulation and the
planner, on small cases whose answer is computed by hand from the SPEC text
(docs/design/SPEC-scheduler.md; m2b-e4 PLAN `scaled.fold_rule`,
`scaled.today_mode`)."""

import functools
import math
import types

import pytest

from wantrain.scheduler.exchange_sim import (
    ADAMW_BYTES,
    Fabric,
    Peer,
    StageLayout,
    StagePrediction,
    Unit,
    fold_bytes,
    pieces_owners,
    simulate_exchange,
    stride_owners,
)
from wantrain.scheduler.zero_bubble import B, F, W, ZBConfig

# CONTROL-I25: every fleet here is one device class (speeds may differ).
ONE_CLASS = "cpu:x86_64:AVX2"

FAST = 1e15  # links far faster than HBM: transfers cost ~nothing
HBM = 1e10


def _peers():
    return tuple(Peer(f"p{q}", FAST, FAST, HBM, f"h{q}", "s") for q in range(2))


def _two_peers(units, owners, weights, *, barrier=None, layout="pieces", flush="module"):
    """Two peers, one microbatch each (peer q runs mb q), op costs 1:2:1 ms
    at equal weights. -> (cfg, orders, weights, {0: layout}, fabric)."""
    stage = StageLayout(units, owners, _peers(), layout=layout, barrier=barrier, flush=flush)
    cfg = ZBConfig(1, 2, [0.0005], [0.001], [0.0005])
    orders = [[[(F, q), (B, q), (W, q)] for q in range(2)]]
    fabric = Fabric(host_dram={"h0": math.inf, "h1": math.inf}, chunk=1 << 20)
    return cfg, orders, [weights], {0: stage}, fabric


def _fold(n, b_g):
    return n * fold_bytes(b_g) / HBM


def _adam(n):
    return n * ADAMW_BYTES / HBM


@pytest.mark.parametrize("slow", [0, 1])
def test_owner_left_folds_every_contribution_in_microbatch_order(slow):
    """One unit in two pieces; the slow peer's W ends 3 ms after the fast
    one's. Each owner folds mb 0 then mb 1, each fold starting at max(the
    previous fold's end, the contribution's arrival), then AdamW. When mb 0
    (peer 0) is the late one, mb 1 waits for mb 0's fold: two folds after the
    stage's end; when mb 1 is the late one, one fold."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the gathers left at max(arrivals) + one fold + AdamW (one
    # fold, both cases), or the owner folded in arrival order instead of the
    # canonical microbatch order (one fold when mb 0 is late).
    params = 1_000_000
    unit = Unit(params, 2, 2, 0, 1.0)
    weights = [0.5, 1.0] if slow == 0 else [1.0, 0.5]
    got = simulate_exchange(*_two_peers((unit,), pieces_owners(1, 2), weights)).stages[0]
    n = params / 2
    folds_after_end = 2 if slow == 0 else 1
    want = folds_after_end * _fold(n, 2) + _adam(n)
    assert got.tail == pytest.approx(want, rel=1e-5)
    assert got.delay == pytest.approx(want, rel=1e-5)


def test_barrier_releases_every_gather_at_the_latest_chain_end():
    """Stride ZeRO-1 with the step barrier, both units released at W's end
    (flush="stage"): a small unit owned by peer 0 (needed first) and a large
    one owned by peer 1. Tb is the large unit's chain end (two folds, both
    contributions at the stage's end, then AdamW); every gather leaves at Tb
    and step t+1's first unit waits for Tb, so X = Tb - end. Without the
    barrier the small unit is ready after its own chain and the large one
    is needed c_F x (the small unit's share of the F FLOPs) later."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: Tb were max(arrivals) + one fold + AdamW, or step t+1's
    # first unit did not wait for Tb under the barrier.
    small, big = 100_000, 2_000_000
    units = (Unit(small, 2, 2, 0, float(small)), Unit(big, 2, 2, 1, float(big)))
    owners = stride_owners((((small, 0),), ((big, 1),)), 2)
    assert owners == ((1.0, 0.0), (0.0, 1.0))
    kept = simulate_exchange(
        *_two_peers(units, owners, [1.0, 1.0], layout="zero1", flush="stage")
    ).stages[0]
    tb = 2 * _fold(big, 2) + _adam(big)
    assert 2 * _fold(small, 2) + _adam(small) < tb
    assert kept.tail == pytest.approx(tb, rel=1e-5)
    free = simulate_exchange(
        *_two_peers(units, owners, [1.0, 1.0], barrier=False, layout="zero1", flush="stage")
    ).stages[0]
    c_f = 0.001  # 0.5 ms at weight 1 of 2
    assert free.tail == pytest.approx(tb - c_f * small / (small + big), rel=1e-5)


def test_stage_flush_releases_every_unit_at_w_end():
    """flush="stage" (today's whole-stage flush, PLAN scaled.today_mode): the
    Embed-like unit (no W compute) and the module are both released at W's
    end, so the emission order changes nothing and the Embed's chain ends
    2 folds + AdamW after the stage's end. Under flush="module" the Embed is
    released at W's start, c_W = 1 ms earlier."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the stage flush released a unit before W's end (the
    # Embed at W start, or modules in emission order), or the flag were ignored.
    pe, pl = 2_000_000, 200_000
    units = (Unit(pe, 4, 4, 0, 0.0), Unit(pl, 2, 2, 1, 1.0))
    own = pieces_owners(2, 2)
    embed = 2 * _fold(pe / 2, 4) + _adam(pe / 2)
    module = 2 * _fold(pl / 2, 2) + _adam(pl / 2)
    assert embed > module + 0.001
    stage = {
        e: simulate_exchange(*_two_peers(units, own, [1.0, 1.0], flush="stage"), emission=e)
        for e in ("forward", "backward")
    }
    assert stage["forward"].stages == stage["backward"].stages
    assert stage["forward"].stages[0].tail == pytest.approx(embed, rel=1e-5)
    per_mb = simulate_exchange(*_two_peers(units, own, [1.0, 1.0])).stages[0]
    c_w = 0.001  # 0.5 ms at weight 1 of 2
    assert per_mb.tail == pytest.approx(embed - c_w, rel=1e-5)
    with pytest.raises(ValueError, match="flush"):
        StageLayout(units, own, _peers(), flush="unit")


# ------------------------------------------------------------------ planner


def _small_cfg():
    from wantrain.model.gpt import GPTConfig

    return GPTConfig(
        vocab_size=1000, n_layer=2, n_head=4, d_model=64, seq_len=128, compute_dtype="bf16"
    )


def test_zero1_embed_owners_follow_the_runtime_stride():
    """Under zero1 the planner's stage 0 gives the Embed unit's owners as the
    runtime shards its tensors (runtime/stage.py params[peer::w]): tok,
    parameter 0, on peer 0 and pos, parameter 1, on peer 1, in proportion
    to their elements; the block's modules sit at stride 2 + {2, 4, 8, 10}."""
    # covers: SCHEDULER-I16, SCHEDULER-I17
    # WOULD-FAIL-IF: the merged Embed unit were placed whole on stride index 0
    # (peer 0), as the planner did before.
    from wantrain.scheduler import planner

    cfg = _small_cfg()
    ns = types.SimpleNamespace(
        cfg=cfg,
        native_bytes=2,
        bnd_bytes=2,
        tokens=128,
        tensor=128 * 64 * 2,
        recompute=True,
        gpus={i: types.SimpleNamespace(name=f"g{i}") for i in range(4)},
    )
    ns.costs = functools.partial(planner._Ctx.costs, ns)
    st = planner._Ctx.stage(ns, 0, 3, (0, 2), [0, 1, 2, 3], "zero1", "native")
    h, v, seq = 64, 1000, 128
    tok, pos = v * h / ((v + seq) * h), seq * h / ((v + seq) * h)
    assert st.units[0].params == (v + seq) * h and st.units[0].b_g == 4
    assert st.owners[0] == pytest.approx((tok, pos, 0.0, 0.0), abs=1e-12)
    # qkv 2+2, proj 2+4, fc1 2+8, fc2 2+10 -> peers 0, 2, 2, 0
    assert [row.index(1.0) for row in st.owners[1:]] == [0, 2, 2, 0]
    pieces = planner._Ctx.stage(ns, 0, 3, (0, 2), [0, 1, 2, 3], "pieces", "native")
    assert pieces.owners[0] == (0.25,) * 4


@pytest.mark.parametrize(
    "tail,delay,window,want",
    [
        (0.010, 0.050, 0.020, 0.050),  # delay-bound: already against the slot
        (0.050, 0.010, 0.020, 0.030),  # tail-bound: the window hides part of X_s
        (0.010, 0.000, 0.020, 0.000),  # hidden entirely
    ],
)
def test_planner_subtracts_the_window_from_the_tail_only(tail, delay, window, want):
    """SCHEDULER-I16/I17: step = period + max_s exposed_s + T_ovh, where the
    window_s before step t+1's forward hides part of the tail X_s; the delay
    of that forward is measured against its slot, which already holds the
    window, so it is never reduced by the window a second time."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the planner priced max(X_s, delay) - window_s (a
    # delay-bound stage with window_s > 0 under-priced by the window).
    from wantrain.scheduler import planner

    p = StagePrediction(
        exposure=max(tail, delay), tail=tail, delay=delay, window=window, overhead=0.0
    )
    assert planner._exposed(p) == pytest.approx(want, abs=1e-15)


def _cpu_fleet(ram_bytes):
    measured = ("host_ram_bytes", "host_dram_gbps", "nic_up_gbps", "nic_down_gbps")
    nodes = [
        dict(
            rank=i,
            host="127.0.0.1",
            tflops=0.2,
            vram_bytes=16 << 30,
            dial_port=7000 + i,
            listen_port=8000 + i,
            device="cpu",
            hostname="mac",
            site="",
            gpu_name="cpu",
            host_ram_bytes=ram_bytes,
            host_dram_gbps=20.0,
            nic_up_gbps=20.0,
            nic_down_gbps=20.0,
            sources={k: "measured" for k in (*measured, "device", "gpu_name")}
            | {"pcie_h2d_gbps": "absent", "pcie_d2h_gbps": "absent"},
            device_class=ONE_CLASS,
        )
        for i in range(4)
    ]
    job = {"model": "toy", "steps": 3, "device": "cpu", "mb_size": 64, "microbatches": 8}
    return {
        "run_id": "toy",
        "nodes": nodes,
        "link": {"mbps": 20000, "rtt_ms": 0.2, "duplex": "per_direction"},
        "job": job,
    }


def test_mem_host_is_charged_on_pieces_candidates_only():
    """SCHEDULER-I16 admits shard "pieces" under MEM-HOST; zero1 wide stages
    keep today's admission (SPEC task 4). A host whose RAM is all reserve
    (4 GiB, the floor) refuses every wide pieces candidate for MEM-HOST and
    no zero1 candidate."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: MEM-HOST were charged on zero1 candidates too.
    from wantrain.scheduler import fleetplan

    out = fleetplan.build_spec(_cpu_fleet(4 << 30), allow_pieces=True)
    wide = [c for c in out["candidates"] if c["w"] > 1]
    pieces = [c for c in wide if c["shard"] == "pieces"]
    zero1 = [c for c in wide if c["shard"] == "zero1"]
    assert pieces and zero1
    assert all(any(r["kind"] == "MEM-HOST" for r in c["reasons"]) for c in pieces)
    assert not any(r["kind"] == "MEM-HOST" for c in zero1 for r in c["reasons"])


def test_s2_takes_the_window_off_the_tail_only():
    """covers: SCHEDULER-I16
    WOULD-FAIL-IF: S2 subtracts window_s from a delay that is already measured
    against the slot, admitting a delay-bound stage whose forward is late."""
    import dataclasses

    from wantrain.scheduler import admission
    from wantrain.scheduler.exchange_sim import StagePrediction

    fields = {f.name for f in dataclasses.fields(StagePrediction)}
    base = {n: 0.0 for n in fields}
    late = StagePrediction(**{**base, "exposure": 10.0, "tail": 4.0, "delay": 10.0, "window": 6.0})
    tail_bound = StagePrediction(
        **{**base, "exposure": 10.0, "tail": 10.0, "delay": 2.0, "window": 6.0}
    )
    # epsilon*C = 5: the late forward (10 beyond its slot) is refused, the tail-bound stage (10 - 6 = 4) admitted
    reasons = admission.s2({0: late, 1: tail_bound}, compute=5.0 / admission.EPSILON)
    assert [r.stage for r in reasons] == [0]
    assert reasons[0].value == 10.0


def test_counted_k_prices_the_fragmented_exchange_of_adaptive_microbatching():
    """covers: SCHEDULER-I16
    WOULD-FAIL-IF: S1 priced a job with adaptive microbatching (fragmented
    exchange frames, SC-5: 14 + 20 passes) at the default path's k 15, or a
    job without it at 17."""
    import re
    from pathlib import Path

    from wantrain.scheduler import admission

    assert admission.counted_k(0.0) == admission.K_HOST_DRAM_COUNTED == 15
    assert admission.counted_k(None) == 15
    assert admission.counted_k(8.0) == admission.K_HOST_DRAM_COUNTED_ADAPTIVE == 17
    spec = (Path(__file__).resolve().parents[2] / "docs/design/SPEC-scheduler.md").read_text()
    m = re.search(
        r"\(15 today on the sealed M2 data plane, (\d+) when adaptive microbatching", spec
    )
    assert m and int(m.group(1)) == admission.K_HOST_DRAM_COUNTED_ADAPTIVE
