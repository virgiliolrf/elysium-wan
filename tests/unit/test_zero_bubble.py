"""Zero-bubble scheduler: paper formulas, the reference implementation's
schedules, heterogeneity, memory, and the deferred weight-gradient split.

WOULD-FAIL-IF: the port drifted from sail-sg's heuristic, a schedule broke
a dependency or a memory limit, or deferring dW changed a single bit.
"""

import pytest
import torch
import torch.nn as nn

from wantrain.model import wgrad
from wantrain.model.gpt import GPTConfig, build_layers
from wantrain.scheduler.zero_bubble import (
    B,
    F,
    W,
    ZBConfig,
    _with_pending_budget,
    analytic_costs,
    bridge_orders,
    default_schedule,
    iwrr,
    peak_memory,
    schedule_for_spec,
    simulate,
    simulate_slots,
    slot_orders,
    solve,
    stage_memory_model,
    zero_bubble,
)

# CONTROL-I25: every fleet here is one device class (speeds may differ).
ONE_CLASS = "cpu:x86_64:AVX2"


def one_f_one_b_baseline(cfg):
    """Classic 1F1B with each W right after its B. Not part of the engine:
    only the yardstick the paper measures zero bubble against."""
    p, m = cfg.n_stages, cfg.n_micro
    orders = []
    for s in range(p):
        warm = min(p - s - 1, m)
        o = [(F, mb) for mb in range(warm)]
        for i in range(m - warm):
            o += [(F, warm + i), (B, i), (W, i)]
        for i in range(warm):
            o += [(B, m - warm + i), (W, m - warm + i)]
        orders.append(o)
    return simulate(cfg, orders)[1]


def uniform(p, m, tf=1.0, tb=1.0, tw=1.0, comm=0.0, **kw):
    return ZBConfig(p, m, [tf] * p, [tb] * p, [tw] * p, comm=comm, **kw)


@pytest.mark.parametrize("p,m", [(4, 12), (8, 24), (3, 9)])
def test_paper_bubble_formulas(p, m):
    """ZB-H1 bubble (p-1)(TF+TB-TW) at 1F1B memory; zero bubble at 2x
    memory when TF = TB = TW (Qi et al., Sec. 2)."""
    # covers: SCHEDULER-I2
    cfg = uniform(p, m)
    work = m * 3.0
    assert zero_bubble(cfg, "1p").makespan == pytest.approx(work + (p - 1) * 1.0)
    z2 = zero_bubble(cfg, "2p")
    assert z2.makespan == pytest.approx(work) and z2.bubble_rate == pytest.approx(0.0)
    assert one_f_one_b_baseline(cfg) > zero_bubble(cfg, "1p").makespan


# Produced by sail-sg/zero-bubble-pipeline-parallelism's own
# scheduler/zb.py (Graph.build_graph + initial_solution) for these inputs.
GOLDEN = [
    (
        dict(p=4, m=8, tf=1.0, tb=1.0, tw=1.0, c=0.0, mf=2.0, mb=-1.0, mm=8.0),
        27.0,
        [
            "F0 F1 F2 F3 B0 W0 F4 B1 W1 F5 B2 W2 F6 B3 W3 F7 B4 W4 B5 W5 B6 W6 B7 W7",
            "F0 F1 F2 B0 F3 B1 W0 F4 B2 W1 F5 B3 W2 F6 B4 W3 F7 B5 W4 B6 W5 B7 W6 W7",
            "F0 F1 B0 F2 B1 F3 B2 W0 F4 B3 W1 F5 B4 W2 F6 B5 W3 F7 B6 W4 B7 W5 W6 W7",
            "F0 B0 F1 B1 F2 B2 F3 B3 W0 F4 B4 W1 F5 B5 W2 F6 B6 W3 F7 B7 W4 W5 W6 W7",
        ],
    ),
    (
        dict(p=3, m=6, tf=2.0, tb=3.0, tw=1.0, c=1.0, mf=2.0, mb=-1.5, mm=12.0),
        45.0,
        [
            "F0 F1 F2 F3 F4 F5 B0 W0 B1 W1 B2 W2 B3 W3 B4 W4 B5 W5",
            "F0 F1 F2 F3 B0 F4 B1 F5 B2 W0 W1 B3 W2 W3 B4 W4 B5 W5",
            "F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 W0 W1 W2 W3 W4 W5",
        ],
    ),
    (
        dict(p=5, m=10, tf=1.0, tb=2.0, tw=1.0, c=3.0, mf=2.0, mb=-1.0, mm=20.0),
        67.0,
        [
            "F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 B0 W0 B1 W1 B2 W2 B3 W3 B4 W4 B5 W5 B6 W6 B7 W7 B8 W8 B9 W9"
        ]
        * 4
        + [
            "F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8 F9 B9 W0 W1 W2 W3 W4 W5 W6 W7 W8 W9"
        ],
    ),
]


@pytest.mark.parametrize("case,best,orders", GOLDEN)
def test_matches_the_reference_implementation(case, best, orders):
    # covers: SCHEDULER-I1
    p, m = case["p"], case["m"]
    cfg = ZBConfig(
        p,
        m,
        [case["tf"]] * p,
        [case["tb"]] * p,
        [case["tw"]] * p,
        comm=case["c"],
        mem_f=[case["mf"]] * p,
        mem_b=[case["mb"]] * p,
        mem_w=[-(case["mf"] + case["mb"])] * p,
        max_mem=[case["mm"]] * p,
    )
    z = zero_bubble(cfg, "given")
    got = [" ".join(f"{t}{mb}" for t, mb in o) for o in z.orders]
    assert got == orders
    assert z.makespan == pytest.approx(best)


def _replay_memory(cfg, orders):
    worst = []
    for s, order in enumerate(orders):
        m = peak = 0.0
        for t, _mb in order:
            m += {F: cfg.mem_f, B: cfg.mem_b, W: cfg.mem_w}[t][s]
            peak = max(peak, m)
        worst.append(peak)
    return worst


def test_heterogeneous_stages_and_wan_hops():
    """Per-stage costs (a slow card in the middle) and per-hop links (one
    slow WAN hop): the schedule is valid, within memory, and beats 1F1B."""
    # covers: SCHEDULER-I3
    cfg = ZBConfig(
        4,
        16,
        cost_f=[1.0, 2.5, 1.0, 1.2],
        cost_b=[1.2, 3.0, 1.2, 1.4],
        cost_w=[0.8, 2.0, 0.8, 0.9],
        comm=[0.1, 2.0, 0.1],
        mem_f=[1.0] * 4,
        mem_b=[-0.5] * 4,
        mem_w=[-0.5] * 4,
        max_mem=[8.0] * 4,
    )
    z = zero_bubble(cfg, "given")
    _, ms = simulate(cfg, z.orders)
    assert ms == pytest.approx(z.makespan)
    assert all(peak <= 8.0 + 1e-9 for peak in _replay_memory(cfg, z.orders))
    assert z.makespan < one_f_one_b_baseline(cfg)
    for s, o in enumerate(z.orders):
        assert sorted(mb for t, mb in o if t == W) == list(range(16)), s


def test_simulator_rejects_a_deadlocked_order():
    # covers: SCHEDULER-I4
    cfg = uniform(2, 2)
    bad = [
        [(F, 0), (B, 0), (W, 0), (F, 1), (B, 1), (W, 1)],
        [(B, 0), (F, 0), (W, 0), (F, 1), (B, 1), (W, 1)],
    ]
    with pytest.raises(ValueError, match="deadlock"):
        simulate(cfg, bad)


# ------------------------------------------------------------------ wgrad


def _grads(layers, x, tgt, deferred):
    params = [p for p in layers.parameters() if p.requires_grad]
    h = x
    for layer in layers:
        h = layer(h)
    loss = nn.functional.cross_entropy(h.reshape(-1, h.size(-1)), tgt.reshape(-1))
    if not deferred:
        return torch.autograd.grad(loss, params)
    owned = wgrad.owned_params(layers)
    others = [p for p in params if id(p) not in owned]
    with wgrad.capture() as store:
        og = torch.autograd.grad(loss, others, allow_unused=True)
    dws = wgrad.flush(store)
    by = {id(p): g for p, g in zip(others, og)}
    return [by[id(p)] if id(p) in by else dws[id(p)] for p in params]


@pytest.mark.parametrize("compute", ["fp32", "bf16"])
def test_deferred_weight_grads_are_bit_identical(compute):
    # covers: MODEL-I1
    cfg = GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        d_model=32,
        seq_len=8,
        compute_dtype=compute,
        boundary_dtype=compute,
    )
    layers = build_layers(cfg, 7)
    assert wgrad.owned_params(layers)
    x = torch.randint(0, 64, (2, 8))
    tgt = torch.randint(0, 64, (2, 8))
    now = _grads(layers, x, tgt, deferred=False)
    later = _grads(layers, x, tgt, deferred=True)
    assert all(torch.equal(a, b) for a, b in zip(now, later))


def test_split_linear_matches_stock_linear_numerically():
    # covers: MODEL-I1
    torch.manual_seed(0)
    a = nn.Linear(16, 8)
    b = nn.Linear(16, 8)
    b.load_state_dict(a.state_dict())
    wgrad.convert(nn.ModuleList([b]))
    x = torch.randn(3, 5, 16, requires_grad=True)
    ga = torch.autograd.grad(a(x).square().sum(), [x, a.weight, a.bias])
    gb = torch.autograd.grad(b(x).square().sum(), [x, b.weight, b.bias])
    for u, v in zip(ga, gb):
        torch.testing.assert_close(u, v)


def test_convert_leaves_tied_weights_to_autograd():
    # covers: MODEL-I3
    emb = nn.Embedding(10, 4)
    head = nn.Linear(4, 10, bias=False)
    head.weight = emb.weight
    other = nn.Linear(4, 4)
    n = wgrad.convert(nn.ModuleList([emb, head, other]))
    assert n == 1 and type(head) is nn.Linear and isinstance(other, wgrad.SplitLinear)


def test_store_is_seen_by_device_backward_threads():
    """CUDA and MPS run backward on an autograd device thread; the store must
    be visible there, or dW is silently returned to a call that discards it."""
    # covers: MODEL-I2
    from conftest import GPU_OFF, cuda_available, gpu_allowed, mps_available

    dev = "cuda" if cuda_available() else ("mps" if mps_available() else None)
    if dev is None:
        pytest.skip("needs a CUDA or MPS device" if gpu_allowed() else GPU_OFF)
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, seq_len=8)
    layers = build_layers(cfg, 7).to(dev)
    x = torch.randint(0, 64, (2, 8), device=dev)
    tgt = torch.randint(0, 64, (2, 8), device=dev)
    now = _grads(layers, x, tgt, deferred=False)
    later = _grads(layers, x, tgt, deferred=True)
    assert all(torch.equal(a, b) for a, b in zip(now, later))


def test_heterogeneous_inputs_always_give_valid_schedules():
    """Per-stage costs and memories the reference never sees (it uses one
    median cost): every schedule must still respect every dependency."""
    # covers: SCHEDULER-I3
    import random

    rng = random.Random(7)
    for _ in range(200):
        p, m = rng.randint(2, 6), rng.randint(1, 18)
        mf = [rng.uniform(0.5, 3) for _ in range(p)]
        mb = [rng.uniform(-x, 2) for x in mf]
        cfg = ZBConfig(
            p,
            m,
            [rng.uniform(0.1, 5) for _ in range(p)],
            [rng.uniform(0.1, 5) for _ in range(p)],
            [rng.uniform(0.05, 5) for _ in range(p)],
            comm=[rng.choice([0, 0.1, 1, 5]) for _ in range(p - 1)],
            mem_f=mf,
            mem_b=mb,
            mem_w=[-(a + b) for a, b in zip(mf, mb)],
        )
        try:
            z = zero_bubble(cfg, rng.choice(["1p", "2p"]))
        except ValueError as e:  # a budget too small for any schedule is a refusal, not a bug
            assert "fit" in str(e)
            continue
        simulate(cfg, z.orders)


def test_iwrr_is_proportional_interleaved_and_round_robin_when_equal():
    # covers: SCHEDULER-I6
    assert iwrr(6, [1, 1]) == [0, 1, 0, 1, 0, 1]
    own = iwrr(12, [3, 1])
    assert own.count(0) == 9 and own.count(1) == 3
    assert max(len(run) for run in "".join(map(str, own)).split("1")) <= 4  # interleaved
    with pytest.raises(ValueError):
        iwrr(4, [1, 0])


def test_wide_stage_slot_orders_cannot_deadlock():
    """A stage-level order split by owner keeps every peer's relative order,
    so the slot-level schedule is deadlock-free by construction."""
    # covers: SCHEDULER-I5
    cfg = ZBConfig(3, 9, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], comm=[0.3, 0.3])
    z = zero_bubble(cfg, "2p")
    owner = [[0] * 9, iwrr(9, [2.0, 1.0]), [0] * 9]
    orders = slot_orders(z.orders, owner, [1, 2, 1])
    assert sorted(mb for t, mb in orders[1][1] if t == F) == [i for i in range(9) if owner[1][i]]
    _, ms = simulate_slots(cfg, orders, owner, [[1.0], [1.0, 0.5], [1.0]])
    assert ms > 0


def test_default_schedule_is_deterministic_and_complete():
    # covers: SCHEDULER-I7
    from wantrain.model.gpt import make_config
    from wantrain.runtime.spec import JobSpec

    spec = JobSpec(model=make_config("toy"), stages=4, microbatches=10, stage_widths=[1, 2, 1, 1])
    spec.finalize()
    again = JobSpec.from_json(spec.to_json())
    again.schedule_order = None
    assert again.finalize().schedule_order == spec.schedule_order
    assert default_schedule(spec) == spec.schedule_order


# ------------------------------------------------------------------ lane orders (R14)


def _stage_level(cfg, owner, widths, weights, k):
    """Today's bridge, written out: one server per stage, split per peer."""
    st = zero_bubble(_with_pending_budget(cfg, k), "given")
    orders = slot_orders(st.orders, owner, widths)
    return orders, simulate_slots(cfg, orders, owner, weights)[1]


def _derivation_cfg(S, w, M):
    """derivation.md section 5's depth.py inputs: 7B, H100, T_mb 4096, recompute
    memory model, aggregate (per-peer / w) cost per stage, zero comm."""
    h, tmb = 4096, 4096
    c = 8 * (12 * h * h * 32 / S) * tmb / 4e14
    bnd, wpin = tmb * h * 2, (32 // S) * tmb * 16 * h * 4
    cfg = ZBConfig(
        S,
        M,
        [c / 4 / w] * S,
        [c / 2 / w] * S,
        [c / 4 / w] * S,
        comm=[0.0] * (S - 1),
        mem_f=[float(bnd)] * S,
        mem_b=[float(wpin)] * S,
        mem_w=[-float(bnd + wpin)] * S,
    )
    return cfg, (M / w) * c


@pytest.mark.parametrize(
    "S,w,lane,stage", [(16, 2, 46.9, 46.9), (8, 4, 43.8, 109.4), (4, 8, 37.5, 187.5)]
)
def test_lane_order_reproduces_the_derivation_figures(S, w, lane, stage):
    """Equal weights, uniform costs, ZB-2p, M 32: the bridge's orders idle as
    the lane solve of derivation.md section 5, today's split as its control."""
    # covers: SCHEDULER-I19
    # WOULD-FAIL-IF: the bridge kept the stage-level split on wide stages
    # (109.4 / 187.5% instead of 43.8 / 37.5%). (Equal weights and zero comm
    # make a lane's cost scale order-neutral; test_lane_order_solves_each_lane_
    # at_its_peers_op_time pins the scale.)
    cfg, C = _derivation_cfg(S, w, 32)
    owner = [iwrr(32, [1.0] * w) for _ in range(S)]
    weights = [[1.0] * w for _ in range(S)]
    orders, _sched, ms = bridge_orders(cfg, owner, [w] * S, weights, "2p")
    assert simulate_slots(cfg, orders, owner, weights)[1] == ms
    assert abs(100 * (ms / C - 1) - lane) <= 0.5
    _o, base = _stage_level(cfg, owner, [w] * S, weights, 2)
    assert abs(100 * (base / C - 1) - stage) <= 0.5


def _peaks(cfg, s, orders):
    out = []
    for order in orders:
        m = peak = 0.0
        for t, _mb in order:
            m += {F: cfg.mem_f, B: cfg.mem_b, W: cfg.mem_w}[t][s]
            peak = max(peak, m)
        out.append(peak)
    return out


def test_lane_order_keeps_each_peer_within_its_own_budget():
    """S 4, w 4, M 32 under "auto": per-peer budgets bound every lane order;
    handing each lane the stage sum (the control) overruns the peer's
    budget; a scalar (summed) budget keeps today's stage-level orders."""
    # covers: SCHEDULER-I19
    # WOULD-FAIL-IF: a lane were solved against the stage's summed budget, or a
    # scalar budget on a wide stage produced lane orders.
    S, w, M, b = 4, 4, 32, 3.0

    def cfg():
        return ZBConfig(S, M, [0.25] * S, [0.5] * S, [0.25] * S, comm=[0.0] * (S - 1))

    owner = [iwrr(M, [1.0] * w) for _ in range(S)]
    weights = [[1.0] * w for _ in range(S)]
    c = cfg()
    orders, sched, _ms = bridge_orders(c, owner, [w] * S, weights, "auto", [[b] * w] * S)
    assert sched.params["lane_stages"] == [0, 1, 2, 3]
    assert all(pk <= b + 1e-9 for s in range(S) for pk in _peaks(c, s, orders[s]))
    c = cfg()
    summed, _s, _ms = bridge_orders(c, owner, [w] * S, weights, "auto", [[b * w] * w] * S)
    assert max(pk for s in range(S) for pk in _peaks(c, s, summed[s])) > b
    c = cfg()
    kept, sched, _ms = bridge_orders(c, owner, [w] * S, weights, "auto", [b * w] * S)
    assert "lane_stages" not in sched.params
    c2 = cfg()
    c2.max_mem = list(c.max_mem)
    assert kept == slot_orders(zero_bubble(c2, "given").orders, owner, [w] * S)


def test_lane_order_is_never_slower_on_heterogeneous_configs():
    """The generator of test_heterogeneous_inputs_always_give_valid_schedules
    with widths 1-4 and weighted IWRR: the bridge never returns a longer
    makespan than the stage-level split, its orders cover exactly each slot's
    microbatches and simulate without deadlock, and lanes do win somewhere."""
    # covers: SCHEDULER-I19
    # WOULD-FAIL-IF: lane orders were returned without comparing them, a lane
    # order dropped or duplicated a microbatch, or the lane solve never ran.
    import random

    rng = random.Random(7)
    wins = 0
    for _ in range(200):
        p, m = rng.randint(2, 6), rng.randint(1, 18)
        mf = [rng.uniform(0.5, 3) for _ in range(p)]
        mb = [rng.uniform(-x, 2) for x in mf]
        costs = [[rng.uniform(lo, 5) for _ in range(p)] for lo in (0.1, 0.1, 0.05)]
        comm = [rng.choice([0, 0.1, 1, 5]) for _ in range(p - 1)]
        widths = [rng.randint(1, 4) for _ in range(p)]
        weights = [[rng.uniform(0.25, 1.0) for _ in range(w)] for w in widths]
        owner = [iwrr(m, wt) for wt in weights]
        k = rng.choice([1, 2])
        mem = dict(comm=comm, mem_f=mf, mem_b=mb, mem_w=[-(a + b) for a, b in zip(mf, mb)])
        try:
            _o, base = _stage_level(ZBConfig(p, m, *costs, **mem), owner, widths, weights, k)
        except ValueError as e:  # a budget too small for any schedule is a refusal
            assert "fit" in str(e)
            continue
        c = ZBConfig(p, m, *costs, **mem)
        orders, _sched, ms = bridge_orders(c, owner, widths, weights, f"{k}p")
        assert simulate_slots(c, orders, owner, weights)[1] == ms <= base
        for s in range(p):
            for q in range(widths[s]):
                own = [i for i in range(m) if owner[s][i] == q]
                for typ in (F, B, W):
                    assert sorted(mb_ for t, mb_ in orders[s][q] if t == typ) == own
        wins += ms < base
    assert wins > 0


def test_lane_order_through_schedule_for_spec_and_default_schedule():
    """The engine's entry points: a wide spec gets lane orders and a shorter
    slot makespan than the stage-level split; width 1 gets today's orders;
    default_schedule stays deterministic across a JSON round trip."""
    # covers: SCHEDULER-I19
    # WOULD-FAIL-IF: schedule_for_spec bypassed the lane solve, the default
    # schedule changed across a JSON round trip, or a width-1 pipeline's
    # orders changed.
    from wantrain.model.gpt import make_config
    from wantrain.runtime.spec import JobSpec

    spec = JobSpec(model=make_config("toy"), stages=4, microbatches=16, stage_widths=[4] * 4)
    spec.finalize()
    prof = [[{"t_f": 1.0, "t_b": 2.0, "t_w": 1.0}] * 4 for _ in range(4)]
    out = schedule_for_spec(spec, prof)
    assert out["schedule"].params["lane_stages"] == [0, 1, 2, 3]
    cfg = out["config"]
    split = slot_orders(out["schedule"].orders, spec.mb_owner, spec.stage_widths)
    base = simulate_slots(cfg, split, spec.mb_owner, spec.peer_weights)[1]
    assert out["slot_makespan"] < base
    again = JobSpec.from_json(spec.to_json())
    again.mb_owner = again.schedule_order = None  # type: ignore[assignment]
    assert again.finalize().schedule_order == default_schedule(again)

    flat = JobSpec(model=make_config("toy"), stages=4, microbatches=10).finalize()
    out = schedule_for_spec(flat, [[{"t_f": 1.0, "t_b": 2.0, "t_w": 1.0}]] * 4)
    assert "lane_stages" not in out["schedule"].params
    assert flat.schedule_order == [[[[t, mb] for t, mb in o]] for o in out["schedule"].orders]


def test_lane_order_solves_each_lane_at_its_peers_op_time():
    """Unequal weights in wide stages with nonzero comm: each peer's order is
    the zero-bubble solve of its own microbatches with every cost scaled by
    sum(w) / w_q, and the aggregate-cost control (scale 1) orders differently."""
    # covers: SCHEDULER-I19
    # WOULD-FAIL-IF: a lane were solved at the aggregate stage cost (or any
    # scale but its peer's op time) instead of cost x sum(w) / w_q.
    S, M, wts = 4, 12, [1.0, 0.5]
    cfg = ZBConfig(S, M, [1.0] * S, [2.0] * S, [1.0] * S, comm=[1.0] * (S - 1))
    owner = [iwrr(M, wts) for _ in range(S)]
    weights = [list(wts) for _ in range(S)]
    orders, sched, _ms = bridge_orders(cfg, owner, [2] * S, weights, "2p")
    assert sched.params["lane_stages"] == [0, 1, 2, 3]

    def lane(own, scale, s):
        c = ZBConfig(
            S,
            len(own),
            [x * scale for x in cfg.cost_f],
            [x * scale for x in cfg.cost_b],
            [x * scale for x in cfg.cost_w],
            comm=cfg.comm,
        )
        return [[t, own[i]] for t, i in zero_bubble(_with_pending_budget(c, 2), "given").orders[s]]

    differs = False
    for s in range(S):
        for q, wq in enumerate(wts):
            own = [i for i in range(M) if owner[s][i] == q]
            expected = lane(own, sum(wts) / wq, s)
            assert orders[s][q] == expected
            differs |= lane(own, 1.0, s) != expected
    assert differs  # the control discriminates


def test_lane_order_from_default_schedule_beats_the_stage_level_split():
    """A spec that arrives without a schedule, 4 stages of width 4, M 16: the
    default orders' slot makespan is strictly below today's stage-level split
    of the same analytic costs (ZB-2p, no comm)."""
    # covers: SCHEDULER-I19
    # WOULD-FAIL-IF: default_schedule (and so JobSpec.finalize) returned the
    # stage-level split on wide stages instead of the lane solve.
    from wantrain.model.gpt import make_config
    from wantrain.runtime.spec import JobSpec

    spec = JobSpec(model=make_config("toy"), stages=4, microbatches=16, stage_widths=[4] * 4)
    spec.finalize()
    cf, cb, cw = analytic_costs(spec)
    agg = [sum(w) for w in spec.peer_weights]
    mems = [stage_memory_model(spec, b - a) for a, b in spec.partition]
    cfg = ZBConfig(
        4,
        16,
        [c / a for c, a in zip(cf, agg)],
        [c / a for c, a in zip(cb, agg)],
        [c / a for c, a in zip(cw, agg)],
        comm=[0.0] * 3,
        mem_f=[x[0] for x in mems],
        mem_b=[x[1] for x in mems],
        mem_w=[x[2] for x in mems],
    )
    _split, base = _stage_level(cfg, spec.mb_owner, spec.stage_widths, spec.peer_weights, 2)
    got = default_schedule(spec)
    assert simulate_slots(cfg, got, spec.mb_owner, spec.peer_weights)[1] < base
    assert spec.schedule_order == got


# ------------------------------------------------- SCHEDULER-I17: exchange simulation

_ES_MODULES = (
    ("qkv", 3, 2),
    ("proj", 1, 4),
    ("fc1", 4, 8),
    ("fc2", 4, 10),
)  # h^2 multiple, stride index


def _exchange_sim_units(kind, n_blocks, h, vocab, seq):
    """SCHEDULER-I17's stage units: the Embed as ONE fp32 unit (tok + pos, no
    W compute) on the first stage, then SplitLinear modules in forward order;
    with each unit's whole tensors as (elements, today's stride index) (gpt.py,
    tok 0, pos 1, 12 parameters per block)."""
    from wantrain.scheduler.exchange_sim import Unit

    us, tensors = [], []
    base = 0
    if kind == "first":
        us.append(Unit((vocab + seq) * h, 4, 4, 0, 0.0))
        tensors.append(((vocab * h, 0), (seq * h, 1)))
        base = 2
    for b in range(n_blocks):
        for _n, s, i in _ES_MODULES:
            us.append(Unit(s * h * h, 2, 2, len(us), float(s * h * h)))
            tensors.append(((s * h * h, base + 12 * b + i),))
    return tuple(us), tensors


def _exchange_sim_arm(
    kind,
    n_blocks,
    w,
    mw,
    lay,
    scale,
    emission="forward",
    policy="need",
    rtt=0.0,
    pad=False,
    tmb=None,
):
    """One E4 arm: a stage of width w alone, each peer running F, B, W per owned
    microbatch (M/w of them), at point B (H100, 400 Gbps) or at round5.md §6's
    scaled B' (F' 50 GFLOP/s; cap', PCIe', HBM' and the chunk scaled so every
    ratio is kept). pad: the step is padded to C_B = 32 x the last stage's c."""
    import math

    from wantrain.scheduler.exchange_sim import (
        Fabric,
        Peer,
        StageLayout,
        pieces_owners,
        simulate_exchange,
        stride_owners,
    )
    from wantrain.validation.exposure import lpt_owners

    if scale == "B'":
        h, vocab, seq, t_mb, rate = 256, 2000, 256, 256, 50e9
    else:
        h, vocab, seq, t_mb, rate = 4096, 32000, 4096, 4096, 400e12
    t_mb = tmb or t_mb
    s_rate = (rate / 400e12) * (4096 / t_mb)
    units, tensors = _exchange_sim_units(kind, n_blocks, h, vocab, seq)
    c = 8 * sum(u.flops for u in units) * t_mb / rate
    owners, layout = {
        "pieces": (pieces_owners(len(units), w), "pieces"),
        "step1": (stride_owners(tensors, w), "zero1"),
        "lpt": (lpt_owners(units, tensors, w), "bounds"),
    }[lay]
    peers = tuple(
        Peer(f"p{q}", 50e9 * s_rate, 50e9 * s_rate, 3.35e12 * s_rate, f"h{q}", "s")
        for q in range(w)
    )
    a = (4 if kind == "mid" else 2) * t_mb * h  # boundary bytes per microbatch per direction
    stage = StageLayout(units, owners, peers, layout=layout, boundary_bytes=a)
    m = w * mw
    cfg = ZBConfig(1, m, [c / 4 / w], [c / 2 / w], [c / 4 / w])
    orders = [
        [[op for mb in range(q, m, w) for op in ((F, mb), (B, mb), (W, mb))] for q in range(w)]
    ]
    fabric = Fabric(
        host_dram={p.host: math.inf for p in peers},
        rtt=rtt,
        chunk=max(1, int(16 * 2**20 * (h / 4096) ** 2)),
    )
    period = 32 * 8 * (12 * h * h + vocab * h) * t_mb / rate if pad else mw * c
    got = simulate_exchange(
        cfg,
        orders,
        [[1.0] * w],
        {0: stage},
        fabric,
        emission=emission,
        policy=policy,
        period=period if pad else None,
    )
    return got, period, (cfg, orders, stage, fabric)


def _e4_plan():
    """The m2b-e4 PLAN of record (amended by SC-4): the pins are read, never restated."""
    from wantrain.validation import exposure

    return exposure.load_plan()


_E4 = _e4_plan()
_ROUND5 = _E4["round5_anchor"]


@pytest.mark.parametrize("arm", sorted(_ROUND5["arms"]))
def test_exchange_sim_reproduces_the_round5_anchor(arm):
    """At round 5's configuration (F' 50 GFLOP/s, op split 1:2:1, HBM'
    paced) E as % of C at B' is the PLAN's `round5_anchor` within its
    tolerance (stage 0 padded per slot to C_B', interior stages against their
    own C), and the same percentage at full point B within 0.5% of B' (scale
    invariance)."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the simulation dropped the max-min sharing, the need-first
    # queues, the left fold or the barrier, laid stage 0 out packed instead of
    # per slot, or mixed a byte or time unit between the scales (e.g. a fixed
    # 16 MiB hop copy instead of the fabric's chunk).
    from wantrain.validation import exposure

    a = exposure.arm_of(arm, _E4)
    full_b = exposure.Shape(4096, 32000, 4096, 4096)
    pct = {
        "B'": exposure.simulate_arm(
            a, exposure.scaled_shape(_E4), exposure.ROUND5_FPRIME, exposure.ROUND5_SPLIT
        ).pct,
        "B": exposure.simulate_arm(a, full_b, 400e12, exposure.ROUND5_SPLIT).pct,
    }
    assert pct["B"] == pytest.approx(pct["B'"], rel=0.005)
    pinned = _ROUND5["arms"][arm]
    assert pct["B'"] == pytest.approx(pinned, rel=_ROUND5["tolerance_rel"]), pct


@pytest.mark.parametrize("row", _E4["arms"], ids=[r["arm"] for r in _E4["arms"]])
def test_exchange_sim_reproduces_the_e4_plan_arms(row):
    """The E4 pre-check, arm by arm: at the PLAN's measured F', op split and
    fold rate the simulation gives every pinned arm, the 'today' arms (the
    whole-stage flush) and the tree included, within the PLAN's 5%."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: an arm were laid out off the PLAN (stage 0 packed, the
    # Embed as two units, today's flush per module) or the simulation drifted
    # from the SPEC's fold, barrier or queue rules.
    from wantrain.validation import exposure

    got = exposure.simulate_plan_arm(_E4, row["arm"]).pct
    assert exposure.ANCHOR_TOLERANCE == 0.05  # PLAN.md: every arm above within 5%
    assert got == pytest.approx(row["predicted_pct_of_C"], rel=exposure.ANCHOR_TOLERANCE), (
        row["arm"],
        got,
    )


@pytest.mark.parametrize("bind", ["nic", "host-dram", "uplink"])
def test_exchange_sim_volume_bound_matches_the_link_period(bind):
    """When one link sets the period, E = max(X_s, V/cap + RTT + t_gap - C)
    within 2% (derivation §2: t_gap = c_B + c_W/U; V the link's bytes per
    step, boundary bytes included, at k passes per side on host DRAM)."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: a resource were not shared among the chunks crossing it
    # (NIC per direction, host DRAM at k passes per byte per side, the uplink
    # per direction), the boundary bytes or RTT were left out, or a link idled
    # while bytes waited.
    import math

    from wantrain.scheduler.exchange_sim import (
        Fabric,
        Peer,
        StageLayout,
        Unit,
        pieces_owners,
        simulate_exchange,
    )

    w, mw, n_units, params, c, rtt, k = 4, 8, 4, 50_000_000, 0.04, 0.05, 3.0
    units = tuple(Unit(params // n_units, 2, 2, i, 1.0) for i in range(n_units))
    big = 1e12
    a = (
        0.0 if bind == "uplink" else 4_000_000.0
    )  # boundary bytes leave the stage, not across its uplink
    sites = ["a", "a", "b", "b"] if bind == "uplink" else ["a"] * w
    nic = 1e9 if bind == "nic" else big
    peers = tuple(Peer(f"p{q}", nic, big, 1e12, f"h{q}", sites[q]) for q in range(w))
    dram = {f"h{q}": (4.5e9 if bind == "host-dram" else math.inf) for q in range(w)}
    uplinks = {("a", "b"): 0.8e9, ("b", "a"): 0.8e9}
    m = w * mw
    cfg = ZBConfig(1, m, [c / 4 / w], [c / 2 / w], [c / 4 / w])
    orders = [
        [[op for mb in range(q, m, w) for op in ((F, mb), (B, mb), (W, mb))] for q in range(w)]
    ]
    fabric = Fabric(host_dram=dram, uplinks=uplinks, k_passes=k, rtt=rtt, chunk=1 << 20)
    got = simulate_exchange(
        cfg,
        orders,
        [[1.0] * w],
        {0: StageLayout(units, pieces_owners(n_units, w), peers, boundary_bytes=a)},
        fabric,
    ).stages[0]
    share = {"nic": (w - 1) / w, "host-dram": (w - 1) / w, "uplink": 2 * 2 / w}[bind]
    per_dir = mw * (2 * params * share + a) + 2 * params * share
    volume, cap = {
        "nic": (per_dir, 1e9),
        "host-dram": (k * 2 * per_dir, 4.5e9),
        "uplink": (per_dir, 0.8e9),
    }[bind]
    bound = volume / cap + rtt + c / 2 + (c / 4) / n_units - mw * c
    assert bound > 2 * got.tail  # the case is volume-bound
    assert got.exposure == pytest.approx(max(got.tail, bound), rel=0.02)
    assert got.window == 0.0


def test_exchange_sim_fifo_exposes_more_than_need_first_at_point_a():
    """Point A (S 8, w 4, M/w 8, 400 Gbps, pieces, backward emission): the
    need-first queues expose 20.127 ms and FIFO 33.871 ms, as the independent
    SCHEDULER-I17 oracle (docs/experiments/m2b-e4/derive/i17_oracle.py) gives
    this arm. round3.md §7's 18.8 ms predates the left fold (SC-4)."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the policy flag were ignored, or need-first did not serve
    # the unit the next forward needs first (gathers before gradients).
    need, _, _ = _exchange_sim_arm("mid", 4, 4, 8, "pieces", "B", emission="backward")
    fifo, _, _ = _exchange_sim_arm(
        "mid", 4, 4, 8, "pieces", "B", emission="backward", policy="fifo"
    )
    e_need, e_fifo = need.stages[0].exposure, fifo.stages[0].exposure
    assert e_need == pytest.approx(0.020127, rel=1e-3)
    assert e_fifo == pytest.approx(0.033871, rel=1e-3)
    assert e_fifo >= 1.5 * e_need


def test_exchange_sim_barrier_is_a_flag_independent_of_the_layout():
    """E4's barrier-kept pieces control: pieces with the barrier on time
    exactly as a zero1 layout (barrier on by default) given the same owners;
    pieces default to no barrier; explicit bounds take the flag too (it
    changes their timing). Backward emission at point A, both values from the
    independent SCHEDULER-I17 oracle (m2b-e4 derive/i17_oracle.py): with the
    left fold the barrier exposes slightly less here, so no sign is implied."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the barrier were tied to the layout name (pieces could
    # not keep it, or zero1 lost it by default), or the flag changed nothing.
    from dataclasses import replace

    from wantrain.scheduler.exchange_sim import bounds_owners, simulate_exchange

    _, _, (cfg, orders, stage, fabric) = _exchange_sim_arm("mid", 4, 4, 8, "pieces", "B")

    def run(st):
        return simulate_exchange(cfg, orders, [[1.0] * 4], {0: st}, fabric, emission="backward")

    assert not stage.step_barrier
    kept = run(replace(stage, barrier=True))
    zero1 = run(replace(stage, layout="zero1"))
    assert replace(stage, layout="zero1").step_barrier
    assert kept == zero1
    free = run(stage)
    # the I17 oracle at this arm: 20.127 ms free, 19.935 ms with the barrier
    # (with the left fold, one release of every gather lets need-first order them)
    assert free.stages[0].exposure == pytest.approx(0.020127, rel=1e-3)
    assert kept.stages[0].exposure == pytest.approx(0.019935, rel=1e-3)
    params = [u.params for u in stage.units]
    total = sum(params)
    edges = [0, total // 4, total // 2, 3 * total // 4, total]
    bounds = replace(stage, layout="bounds", owners=bounds_owners(params, edges))
    # the flag acts on explicit bounds too; here it helps (46.3 -> 45.1 ms): one
    # release of every gather lets need-first order them, so no sign is implied
    assert run(bounds).stages[0].exposure != run(replace(bounds, barrier=True)).stages[0].exposure


def test_exchange_sim_names_the_faster_emission_order():
    """Both emission orders are predicted and the smaller step time is named
    (BA-43). The independent SCHEDULER-I17 oracle (m2b-e4 derive/i17_oracle.py)
    gives, with the left fold: forward at point A (E 18.409 against 20.127 ms)
    and backward at T_mb 16k, RTT 10 ms (80.012 against 80.312 ms)."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the emission order were ignored, fixed, or the choice did
    # not follow the predicted step time.
    from wantrain.scheduler.exchange_sim import predict_exchange

    picks = {}
    for tag, kw in (("A", {}), ("rtt", dict(tmb=16384, rtt=0.010))):
        _, _, (cfg, orders, stage, fabric) = _exchange_sim_arm("mid", 4, 4, 8, "pieces", "B", **kw)
        got = predict_exchange(cfg, orders, [[1.0] * 4], {0: stage}, fabric)
        assert got.forward.emission == "forward" and got.backward.emission == "backward"
        assert got.chosen.step_time == min(got.forward.step_time, got.backward.step_time)
        assert got.forward.step_time != got.backward.step_time
        picks[tag] = got.best
    assert picks == {"A": "forward", "rtt": "backward"}


def test_exchange_sim_is_pure_and_validates_its_inputs():
    """Same inputs, same floats, inputs untouched; malformed inputs refused."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the simulation kept state between calls, mutated an
    # input, or silently accepted owners that do not sum to 1, an unknown
    # emission, a host without DRAM rate or a cross-site pair without uplink.
    import copy
    from dataclasses import replace

    from wantrain.scheduler.exchange_sim import Fabric, simulate_exchange

    _, _, (cfg, orders, stage, fabric) = _exchange_sim_arm("first", 1, 4, 4, "lpt", "B'")
    before = copy.deepcopy((cfg, orders, stage, fabric))
    one = simulate_exchange(cfg, orders, [[1.0] * 4], {0: stage}, fabric, emission="backward")
    two = simulate_exchange(cfg, orders, [[1.0] * 4], {0: stage}, fabric, emission="backward")
    assert one == two
    assert (cfg, orders, stage, fabric) == before
    with pytest.raises(ValueError, match="summing to 1"):
        replace(stage, owners=((0.5, 0.0, 0.0, 0.0),) + stage.owners[1:])
    with pytest.raises(ValueError, match="emission"):
        simulate_exchange(cfg, orders, [[1.0] * 4], {0: stage}, fabric, emission="sideways")
    with pytest.raises(ValueError, match="DRAM"):
        simulate_exchange(cfg, orders, [[1.0] * 4], {0: stage}, Fabric(host_dram={}))
    split = replace(
        stage, peers=tuple(replace(p, site="b") if i else p for i, p in enumerate(stage.peers))
    )
    with pytest.raises(ValueError, match="uplink"):
        simulate_exchange(cfg, orders, [[1.0] * 4], {0: split}, fabric)


@pytest.mark.parametrize("barrier", [False, True])
def test_exchange_sim_gathers_wait_for_the_owners_fold_and_adamw(barrier):
    """Links far faster than HBM, both peers' W ending together: step t+1's
    F on the unit starts after the owner's left fold of both microbatches'
    contributions and one AdamW, at their HBM op times ((b_g + 16) B per
    folded element for a 2-byte wire, 12 for fp32; 30 B per updated element),
    with or without the barrier."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: a gather left before the owner's last fold and AdamW, the
    # fold or AdamW bytes per element were wrong, or F(t+1) did not wait for
    # the unit's gather.
    import math

    from wantrain.scheduler.exchange_sim import (
        Fabric,
        Peer,
        StageLayout,
        Unit,
        pieces_owners,
        simulate_exchange,
    )

    fast, hbm, params = 1e15, 1e9, 1_000_000
    for b_g, per_elem in ((2, 2 * (2 + 16) + 30), (4, 2 * 12 + 30)):
        unit = Unit(params, b_g, b_g, 0, 1.0)
        peers = tuple(Peer(f"p{q}", fast, fast, hbm, f"h{q}", "s") for q in range(2))
        stage = StageLayout((unit,), pieces_owners(1, 2), peers, barrier=barrier)
        cfg = ZBConfig(1, 2, [0.001], [0.002], [0.001])
        orders = [[[(F, q), (B, q), (W, q)] for q in range(2)]]
        fabric = Fabric(host_dram={"h0": math.inf, "h1": math.inf}, chunk=1 << 20)
        got = simulate_exchange(cfg, orders, [[1.0, 1.0]], {0: stage}, fabric).stages[0]
        assert got.tail == pytest.approx(params / 2 * per_elem / hbm, rel=1e-6)
        assert got.exposure == got.tail


# ------------------------------------------------------------------ SCHEDULER-I13


def _complete(cfg, orders):
    for s, o in enumerate(orders):
        for typ in (F, B, W):
            assert sorted(mb for t, mb in o if t == typ) == list(range(cfg.n_micro)), (s, typ)


def _seven_b(budget_gb):
    """SPEC-scheduler task 1's example: a 7B-shaped GPT (32 x 4096, seq 2048,
    bf16 boundary, recompute) on 8 H100 stages at 400 TFLOP/s effective
    (DESIGN.md's assumption), 200 ms one-way per hop, mb 1 and 1024
    microbatches (~2M tokens per step), `budget_gb` of activations per stage."""
    model = GPTConfig(
        vocab_size=32000,
        n_layer=32,
        n_head=32,
        d_model=4096,
        seq_len=2048,
        compute_dtype="bf16",
        boundary_dtype="bf16",
    )
    from wantrain.runtime.spec import JobSpec

    spec = JobSpec(model=model, stages=8, microbatches=1024, mb_size=1, recompute=True)
    spec.partition = [[0, 5]] + [[a, a + 4] for a in range(5, 29, 4)] + [[29, 34]]
    costs = [[x * 2048 / 400e12 for x in c] for c in analytic_costs(spec)]
    mems = [stage_memory_model(spec, b - a) for a, b in spec.partition]
    return ZBConfig(
        8,
        1024,
        *costs,
        comm=[0.2] * 7,
        mem_f=[x[0] for x in mems],
        mem_b=[x[1] for x in mems],
        mem_w=[x[2] for x in mems],
        max_mem=[budget_gb * 1e9] * 8,
    )


def test_tight_budget_degrades_instead_of_refusing():
    """The SPEC's example refused at 8 GB before M2 ("no zero-bubble schedule
    fits the memory limit"); now it degrades to a valid schedule that fits,
    with the bubble reported and no smaller than the 16 GB solve's."""
    # covers: SCHEDULER-I13
    # WOULD-FAIL-IF: zero_bubble raises again when no grid candidate fits, the
    # degraded schedule over-commits a stage, its warm-ups break the
    # non-increasing >= 1 rule, the reported bubble is not the simulated one,
    # or it takes the largest warm-ups as they are (equal warm-ups make every
    # microbatch wait a round trip: ~98% bubble here, against ~17%).
    tight = _seven_b(8)
    z = zero_bubble(tight, "given")
    assert z.params["degraded"] is True
    warm = z.params["warmups"]
    assert len(warm) == 8 and warm[-1] >= 1
    assert all(a >= b for a, b in zip(warm, warm[1:])), warm
    _end, ms = simulate(tight, z.orders)
    assert ms == pytest.approx(z.makespan)
    assert z.params["bubble"] == z.bubble_rate > 0.0
    _complete(tight, z.orders)
    assert all(pk <= 8e9 for pk in peak_memory(tight, z.orders))

    # the best that fits: no worse than 1F1B's warm-ups (p - j) or the
    # largest warm-ups that fit 8 GB, both built here independently
    def warmups(ks):
        out = []
        for k in ks:
            o = [(F, mb) for mb in range(k)]
            for i in range(1024):
                o += [(B, i), (W, i)] + ([(F, k + i)] if k + i < 1024 else [])
            out.append(o)
        return out

    largest = []
    for s in range(8):
        k = 1
        while tight.mem_f[s] * (k + 1) + tight.mem_b[s] <= 8e9 and k < 1024:
            k += 1
        largest.append(min([k] + largest))
    for ks in ([8 - j for j in range(8)], largest):
        orders = warmups(ks)
        assert all(pk <= 8e9 for pk in peak_memory(tight, orders)), ks
        assert z.makespan <= simulate(tight, orders)[1] + 1e-9, ks
    roomy = zero_bubble(_seven_b(16), "given")
    assert roomy.params["degraded"] is False
    assert z.bubble_rate >= roomy.bubble_rate
    # 8 GB holds ~300 microbatches on stage 0; ~85 cover 7 hops of 400 ms
    # round trip at ~36 ms per microbatch, so the latency stays hidden
    assert z.bubble_rate <= roomy.bubble_rate + 0.01, (z.bubble_rate, roomy.bubble_rate)


def test_degraded_schedule_fits_every_budget():
    """Heterogeneous costs, links and per-stage budgets from one microbatch
    upward: every returned schedule, degraded or not, is complete, deadlock-
    free and within every stage's budget."""
    # covers: SCHEDULER-I13
    # WOULD-FAIL-IF: a grid candidate is returned without the memory replay
    # (the heuristic's first F and B are placed unchecked), or the degraded
    # warm-ups are chosen without replaying each stage's memory.
    import random

    rng = random.Random(13)
    seen = {True: 0, False: 0}
    for _ in range(150):
        p, m = rng.randint(2, 6), rng.randint(1, 24)
        mf = [rng.uniform(0.05, 1.0) for _ in range(p)]
        mb = [rng.uniform(-x, 3.0) for x in mf]
        need = [f + max(0.0, b) for f, b in zip(mf, mb)]
        cfg = ZBConfig(
            p,
            m,
            [rng.uniform(0.1, 5) for _ in range(p)],
            [rng.uniform(0.1, 5) for _ in range(p)],
            [rng.uniform(0.05, 5) for _ in range(p)],
            comm=[rng.choice([0, 0.1, 1, 5]) for _ in range(p - 1)],
            mem_f=mf,
            mem_b=mb,
            mem_w=[-(a + b) for a, b in zip(mf, mb)],
            max_mem=[n * rng.uniform(1.0, 2 * p) for n in need],
        )
        z = zero_bubble(cfg, "given")
        seen[z.params["degraded"]] += 1
        simulate(cfg, z.orders)
        _complete(cfg, z.orders)
        for s, (pk, lim) in enumerate(zip(peak_memory(cfg, z.orders), cfg.max_mem)):
            assert pk <= lim + 1e-9, (s, pk, lim, z.params)
    assert seen[True] >= 10 and seen[False] >= 10, seen


def test_bubble_never_grows_with_the_budget():
    """Regression check on pinned grids, not a general property: on the
    4-stage grids of budgets from one microbatch to ample, the no-recompute
    comm-2.0 grid and the 7B example's budgets, the solved bubble is
    non-increasing and the grid crosses from degraded to not. SCHEDULER-I13
    states that monotonicity in the budget is NOT guaranteed (SPEC-scheduler
    section 2.1 Selection, section 5 task 1)."""
    # covers: SCHEDULER-I13
    # WOULD-FAIL-IF: a larger budget picks a worse schedule: the degraded path
    # stops searching warm-ups or caps them below what fits, the grid is
    # skipped when it fits, or the 7B example's tight budgets stop degrading
    # gracefully: 3 GB must cost more than 4 GB (more memory is used) and
    # more than twice the 16 GB bubble; or a grid candidate that barely fits
    # is returned although the warm-up family is better (the no-recompute
    # comm 2.0 grid: 34.4% at budget 5.8, 36.0% at 6.1 before the union).
    # Not claimed: monotonicity on every config. SCHEDULER-I13's stated
    # non-guarantee and its counterexample (costs 1/1/1, mem_f 1, mem_b -0.5,
    # comm 1.0, 4 x 32: budget 5.95 -> 6.1 grows the bubble): both candidate
    # sets are built from the budget.
    shapes = [((0.1, 1.0, -1.1), comm, 1.1) for comm in (0.0, 0.5, 2.0)]
    shapes.append(((1.0, -0.5, -0.5), 2.0, 1.0))
    for (mf, mb, mw), comm, first in shapes:
        cfg = ZBConfig(
            4,
            32,
            [1.0] * 4,
            [2.0] * 4,
            [1.0] * 4,
            comm=[comm] * 3,
            mem_f=[mf] * 4,
            mem_b=[mb] * 4,
            mem_w=[mw] * 4,
        )
        bubbles, kinds = [], set()
        for budget in [first + 0.3 * i for i in range(21)]:
            cfg.max_mem = [budget] * 4
            z = zero_bubble(cfg, "given")
            bubbles.append(z.bubble_rate)
            kinds.add(z.params["degraded"])
        assert kinds == {True, False}, comm
        assert all(a >= b - 1e-12 for a, b in zip(bubbles, bubbles[1:])), (comm, bubbles)
        assert bubbles[0] > bubbles[-1], (comm, bubbles)
    seven = [zero_bubble(_seven_b(gb), "given").bubble_rate for gb in (3, 4, 8, 16)]
    assert all(a >= b - 1e-12 for a, b in zip(seven, seven[1:])), seven
    assert seven[0] > seven[1] and seven[0] > 2 * seven[-1], seven


def test_below_one_microbatch_refuses_with_the_numbers():
    """The only refusal left: a stage whose budget cannot hold one
    microbatch's F plus what its B keeps for W, naming stage, need, budget."""

    # covers: SCHEDULER-I13
    # WOULD-FAIL-IF: the refusal drops the numbers, counts only F (B's pending
    # W share forgotten), or refuses a budget that holds exactly one microbatch.
    def cfg(budget2):
        return ZBConfig(
            4,
            8,
            [1.0] * 4,
            [1.0] * 4,
            [1.0] * 4,
            comm=0.5,
            mem_f=[1000.0, 1000.0, 1500.0, 1000.0],
            mem_b=[-500.0, 200.0, 3700.0, -500.0],
            mem_w=[-500.0, -1200.0, -5200.0, -500.0],
            max_mem=[2000.0, 2000.0, budget2, 2000.0],
        )

    with pytest.raises(ValueError) as e:
        zero_bubble(cfg(5199.0), "given")
    msg = str(e.value)
    assert "stage 2" in msg and "5200 bytes" in msg and "budget 5199 bytes" in msg, msg
    assert "fit" in msg
    exact = cfg(5200.0)
    z = zero_bubble(exact, "given")
    _complete(exact, z.orders)
    assert peak_memory(exact, z.orders)[2] == pytest.approx(5200.0)


def test_measured_budget_is_never_raised_to_the_floor():
    """solve() hands the measured per-stage budgets to the solver as they
    are; the old 1F1B floor (p - r + 1) x (|mem_f| + |mem_b|) silently
    over-committed a tight device."""
    # covers: SCHEDULER-I13
    # WOULD-FAIL-IF: solve() raises a budget to the floor again (cfg.max_mem
    # differs from the measured budgets, or a stage's peak exceeds its budget),
    # or one unmeasured (0) stage discards the other stages' measured budgets.
    from wantrain.model.gpt import make_config
    from wantrain.runtime.spec import JobSpec

    spec = JobSpec(model=make_config("toy"), stages=4, microbatches=12, recompute=True)
    spec.finalize()
    p = spec.stages
    mems = [stage_memory_model(spec, b - a) for a, b in spec.partition]
    need = [f + max(0.0, b) for f, b, _w in mems]
    floor_ = [(p - r + 1) * (abs(f) + abs(b)) for r, (f, b, _w) in enumerate(mems)]
    budgets = [int(1.5 * n) for n in need]
    assert all(b < f_ for b, f_ in zip(budgets, floor_))
    cf, cb, cw = analytic_costs(spec)
    _orders, sched, cfg, _ms = solve(spec, cf, cb, cw, [max(cf)] * (p - 1), budgets)
    assert cfg.max_mem == [float(b) for b in budgets]
    assert all(pk <= b for pk, b in zip(peak_memory(cfg, sched.orders), budgets))
    _complete(cfg, sched.orders)
    # one unmeasured stage (0) takes its ZB-2p default alone; the measured
    # stages keep theirs (it used to send the whole pipeline to ZB-2p)
    mixed = budgets[:3] + [0]
    _orders, sched, cfg, _ms = solve(spec, cf, cb, cw, [max(cf)] * (p - 1), mixed)
    two_p = 2 * p * (abs(mems[3][0]) + max(0.0, mems[3][1]))
    assert cfg.max_mem == [float(b) for b in budgets[:3]] + [two_p]
    assert two_p > budgets[0]
    peaks = peak_memory(cfg, sched.orders)
    assert all(pk <= b for pk, b in zip(peaks, budgets[:3])), (peaks, budgets)
    _complete(cfg, sched.orders)


# ------------------------------------------------------------------ SCHEDULER-I15


def _replan_spec(widths=(1, 2, 1), m=8):
    from wantrain.model.gpt import make_config
    from wantrain.runtime.spec import JobSpec

    return JobSpec(
        model=make_config("toy"), stages=len(widths), microbatches=m, stage_widths=list(widths)
    ).finalize()


def _windows(spec, cost, hop_s, steps=5, noise=None):
    """{(rank, peer): [summary, ...]} as the coordinator keeps them, each
    summary made by the runtime's own summarize_step from stage events.
    cost(r, q) -> (t_f, t_b, t_w) seconds; hop_s[j] seconds per frame on hop
    j; noise(step, r, q) -> multiplier. A slot owning no microbatch runs no
    op, as in the runtime."""
    from wantrain.runtime.telemetry import summarize_step

    out = {}
    for r in range(spec.stages):
        for q in range(spec.stage_widths[r]):
            win = []
            runs = 2 if q in spec.mb_owner[r] else 0
            for step in range(steps):
                k = noise(step, r, q) if noise else 1.0
                ev, t = [], 0.0
                for name, c in zip(("fwd", "bwd_b", "bwd_w"), cost(r, q)):
                    for _ in range(runs):
                        ev.append({"ev": name, "step": step, "t0": t, "t1": t + c * k})
                        t += c * k
                links = [("next", r)] if r + 1 < spec.stages else []
                links += [("prev", r - 1)] if r > 0 else []
                for head, j in links:
                    for nq in range(spec.stage_widths[j + 1 if head == "next" else j]):
                        for _ in range(3):
                            ev.append(
                                {
                                    "ev": "send",
                                    "hop": f"{head}{nq}",
                                    "nbytes": 64,
                                    "t0": t,
                                    "t1": t + hop_s[j] * k,
                                }
                            )
                            t += hop_s[j] * k
                win.append(summarize_step(ev, step, r, q, t * 1e3, 10))
            out[(r, q)] = win
    return out


def _apply(spec, res):
    import copy

    s = copy.deepcopy(spec)
    s.peer_weights = res["peer_weights"]
    s.mb_owner = res["mb_owner"]
    s.schedule_order = res["schedule_order"]
    return s.finalize()


def test_hops_from_telemetry_counts_transfers_not_fragments():
    # covers: SCHEDULER-I15
    # WOULD-FAIL-IF: the measured hop term divides busy_ms by frames when
    # p2p_streams > 1, where each send event is one fragment, so the hop
    # reads 8x too fast for an 8-fragment transfer (or the single-stream
    # path stops counting one transfer per frame).
    import dataclasses

    from wantrain.runtime.telemetry import summarize_step
    from wantrain.scheduler.zero_bubble import _boundary_bytes, hops_from_telemetry

    frag_s, nfrag, transfers = 0.002, 8, 3
    for streams in (1, 2):
        spec = dataclasses.replace(_replan_spec((1, 1), m=4), p2p_streams=streams).finalize()
        whole = _boundary_bytes(spec)
        win = {}
        for r in range(2):
            head = "next0" if r == 0 else "prev0"
            ev, t = [], 0.0
            for _ in range(transfers):
                # one stream: one send per transfer; else one per fragment
                parts = (
                    [(whole, frag_s * nfrag)] if streams == 1 else [(whole / nfrag, frag_s)] * nfrag
                )
                for nbytes, d in parts:
                    ev.append(
                        {"ev": "send", "hop": head, "nbytes": int(nbytes), "t0": t, "t1": t + d}
                    )
                    t += d
            win[(r, 0)] = [summarize_step(ev, 0, r, 0, t * 1e3, 10)]
        frames = win[(0, 0)][0]["hops"]["next0"]["frames"]
        assert frames == (transfers if streams == 1 else transfers * nfrag)
        assert hops_from_telemetry(spec, win) == pytest.approx([frag_s * nfrag], rel=1e-6)


def test_replan_shifts_microbatches_away_from_a_slow_peer():
    """A width-2 stage with one peer 4x slower: the re-plan gives it fewer
    microbatches and predicts a step at least 10% shorter (HANDOUT-M2's
    rebalancing example)."""
    # covers: SCHEDULER-I15
    # WOULD-FAIL-IF: profiles ignore the measured times (weights stay equal),
    # the prediction of the current step uses the spec's weights instead of
    # the measured speeds, or decision ignores the gain.
    from wantrain.scheduler.zero_bubble import (
        hops_from_telemetry,
        profiles_from_telemetry,
        replan,
    )

    spec = _replan_spec(m=12)
    base = (0.010, 0.012, 0.008)

    def cost(r, q):
        return tuple(4 * c for c in base) if (r, q) == (1, 1) else base

    win = _windows(spec, cost, [0.001, 0.002])
    prof = profiles_from_telemetry(spec, win)
    assert prof[1][1] == pytest.approx({"t_f": 0.040, "t_b": 0.048, "t_w": 0.032})
    assert prof[0][0] == pytest.approx({"t_f": 0.010, "t_b": 0.012, "t_w": 0.008})
    assert hops_from_telemetry(spec, win) == pytest.approx([0.001, 0.002])
    res = replan(spec, win)
    assert res["decision"] is True and res["owner_changed"] is True, res["why"]
    assert res["pred_new_s"] < 0.9 * res["pred_cur_s"]
    assert spec.mb_owner[1].count(1) == 6
    assert res["mb_owner"][1].count(1) < 6
    assert res["peer_weights"][1] == pytest.approx([1.0, 0.25])


def test_replan_keeps_a_balanced_fleet():
    """A balanced fleet re-measured with a few percent of noise: nothing clears
    the 10% threshold, the decision is false. Planned from its own profiles,
    the re-solve gives the same plan; launched on the analytic schedule, the
    re-solve reorders for a ~5% gain, which is below the threshold."""
    # covers: SCHEDULER-I15
    # WOULD-FAIL-IF: decision is true on any changed order regardless of the
    # gain, or the threshold is compared the wrong way.
    from wantrain.scheduler.zero_bubble import replan, schedule_for_spec

    base = (0.010, 0.012, 0.008)

    def noise(step, r, q):
        return 1.0 + 0.03 * (((step * 7 + r * 3 + q * 5) % 5) - 2) / 2

    planned = _replan_spec()
    prof = [[dict(t_f=base[0], t_b=base[1], t_w=base[2])] * w for w in planned.stage_widths]
    schedule_for_spec(planned, prof, [0.001, 0.001])
    analytic = _replan_spec()
    for spec, reordered in ((planned, False), (analytic, True)):
        res = replan(spec, _windows(spec, lambda r, q: base, [0.001, 0.001], noise=noise))
        assert res["decision"] is False, res["why"]
        assert res["owner_changed"] is False
        assert (res["schedule_order"] != spec.schedule_order) is reordered
        assert 0.9 * res["pred_cur_s"] < res["pred_new_s"] <= res["pred_cur_s"]
    # a 4x slower peer in the same fleet does clear it: the false above comes
    # from the gain, not from a comparison that never holds
    slow = _windows(planned, lambda r, q: tuple(4 * c for c in base) if q else base, [0.001] * 2)
    assert replan(planned, slow)["decision"] is True


def test_replan_keeps_the_link_latency():
    """Sends measure only the bytes leaving (the emulated one-way latency is
    applied on the receiving side): the re-plan adds the link's latency back,
    so a fleet launched on 200 ms links and re-measured as launched keeps its
    latency-aware warm-ups and predicts no gain."""
    # covers: SCHEDULER-I15
    # WOULD-FAIL-IF: a measured hop replaces the whole comm term (latency
    # dropped), so the re-solve picks the shallower warm-ups of a ~1 ms link.
    import copy

    from wantrain.scheduler.zero_bubble import (
        hop_comm_seconds,
        hops_from_telemetry,
        replan,
        schedule_for_spec,
    )

    spec = _replan_spec(widths=(1, 1, 1, 1), m=8)
    spec.links = [{"latency_ms": 200.0, "bw_mbps": 1e5}] * 4
    base = (0.010, 0.012, 0.008)
    prof = [[dict(t_f=base[0], t_b=base[1], t_w=base[2])] * w for w in spec.stage_widths]
    schedule_for_spec(spec, prof)
    win = _windows(spec, lambda r, q: base, [0.001] * 3)
    hops = hops_from_telemetry(spec, win)
    assert hops == pytest.approx([0.001] * 3)
    assert hop_comm_seconds(spec, hops) == pytest.approx([0.201] * 3)
    res = replan(spec, win)
    assert res["schedule_order"] == spec.schedule_order, res["why"]
    assert res["decision"] is False
    # the check discriminates: the same solve without the latency reorders
    blind = copy.deepcopy(spec)
    blind.links = [{"latency_ms": 0.0, "bw_mbps": 1e5}] * 4
    schedule_for_spec(blind, prof, [0.001] * 3)
    assert blind.schedule_order != spec.schedule_order


def test_replan_keeps_every_peer_measured():
    """A peer slow enough that weighted round robin would give it nothing
    still owns a microbatch, so the next window measures it and a second
    re-plan runs. A non-positive op time is refused at the boundary."""
    # covers: SCHEDULER-I15
    # WOULD-FAIL-IF: the re-plan starves a slot (iwrr(4, [1, 0.25]) owns all
    # four on peer 0), so its next window has no op sample and every later
    # re-plan raises; or a 0 ms report is taken as an infinitely fast peer.
    from wantrain.scheduler.zero_bubble import replan

    spec = _replan_spec(widths=(1, 2, 1), m=4)
    base = (0.010, 0.012, 0.008)

    def cost(r, q):
        return tuple(4 * c for c in base) if (r, q) == (1, 1) else base

    res = replan(spec, _windows(spec, cost, [0.001, 0.001]))
    assert res["peer_weights"][1] == pytest.approx([1.0, 0.25])
    assert res["mb_owner"][1].count(1) == 1, res["mb_owner"]
    nxt = _apply(spec, res)
    again = replan(nxt, _windows(nxt, cost, [0.001, 0.001]))
    assert again["mb_owner"][1].count(1) == 1
    assert again["decision"] is False, again["why"]
    bad = _windows(spec, cost, [0.001, 0.001])
    bad[(1, 0)][0]["ops"]["F"]["med_ms"] = 0.0
    with pytest.raises(ValueError, match=r"slot \(1, 0\)"):
        replan(spec, bad)


def test_replan_touches_only_ownership_and_order():
    """The input spec is not modified, and the re-planned spec differs from it
    only in peer_weights, mb_owner and schedule_order: the recipe pins (and
    with them the fold order and the bytes) are those of the input."""
    # covers: SCHEDULER-I15
    # WOULD-FAIL-IF: replan mutates its input, or the re-solve changes a
    # recipe field (partition, microbatches, widths) through finalize.
    from dataclasses import asdict

    from wantrain.scheduler.zero_bubble import replan

    spec = _replan_spec(widths=(2, 1, 3), m=10)
    before = spec.to_json()
    base = (0.010, 0.012, 0.008)
    win = _windows(spec, lambda r, q: tuple(c * (1 + q) for c in base), [0.003, 0.001])
    res = replan(spec, win)
    assert spec.to_json() == before
    new = _apply(spec, res)
    a, b = asdict(spec), asdict(new)
    assert sorted(k for k in a if a[k] != b[k]) == ["mb_owner", "peer_weights", "schedule_order"]
    assert new.pins() == spec.pins()
    with pytest.raises(ValueError, match=r"slot \(1, 0\)"):
        replan(spec, {k: v for k, v in win.items() if k != (1, 0)})
    with pytest.raises(ValueError, match="min_gain"):
        replan(spec, win, min_gain=1.0)


def test_replan_is_deterministic():
    """The same windows give byte-identical output across two calls and after
    a JSON round trip of the summaries and of the result."""
    # covers: SCHEDULER-I15
    # WOULD-FAIL-IF: the output depends on dict or set iteration order, holds
    # non-JSON values (tuples, Schedule objects), or reads a clock.
    import json

    from wantrain.scheduler.zero_bubble import replan

    spec = _replan_spec(widths=(1, 3, 2), m=9)
    base = (0.010, 0.012, 0.008)
    win = _windows(spec, lambda r, q: tuple(c * (1 + 0.5 * q) for c in base), [0.002, 0.004])
    one = json.dumps(replan(spec, win), sort_keys=True)
    two = json.dumps(replan(spec, win), sort_keys=True)
    trip = {k: json.loads(json.dumps(v)) for k, v in win.items()}
    three = json.dumps(replan(spec, trip), sort_keys=True)
    assert one == two == three
    assert json.dumps(json.loads(one), sort_keys=True) == one


def test_replanned_orders_are_valid_for_any_window():
    """Random fleets and windows: the returned weights, owners and orders pass
    JobSpec.finalize (every slot covers exactly its microbatches) and
    simulate_slots, and pred_new_s is that simulation's step time."""
    # covers: SCHEDULER-I15
    # WOULD-FAIL-IF: the returned orders are not split by the returned owners,
    # deadlock, or the prediction is not the simulated step of the new plan.
    import random

    from wantrain.scheduler.zero_bubble import (
        hops_from_telemetry,
        predict_step,
        profiles_from_telemetry,
        replan,
    )

    for seed in range(12):
        rng = random.Random(seed)
        widths = [rng.randint(1, 3) for _ in range(rng.randint(2, 4))]
        spec = _replan_spec(widths=widths, m=rng.randint(4, 12))
        costs = {
            (r, q): tuple(rng.uniform(0.001, 0.05) for _ in range(3))
            for r, w in enumerate(widths)
            for q in range(w)
        }
        hop_s = [rng.uniform(0.0, 0.02) for _ in range(len(widths) - 1)]
        win = _windows(
            spec,
            lambda r, q, c=costs: c[(r, q)],
            hop_s,
            noise=lambda s, r, q, g=rng: g.uniform(0.9, 1.1),
        )
        gain = rng.choice([0.0, 0.1, 0.5])
        res = replan(spec, win, min_gain=gain)
        new = _apply(spec, res)
        prof, hops = profiles_from_telemetry(spec, win), hops_from_telemetry(spec, win)
        assert predict_step(new, prof, hops) == res["pred_new_s"]
        cfg = ZBConfig(
            len(widths),
            spec.microbatches,
            [1.0] * len(widths),
            [1.0] * len(widths),
            [1.0] * len(widths),
            comm=hop_s,
        )
        _end, ms = simulate_slots(cfg, new.schedule_order, new.mb_owner, new.peer_weights)
        assert ms > 0
        if res["decision"]:
            assert res["pred_new_s"] <= (1 - gain) * res["pred_cur_s"]


# ------------------------------------------------- interface I-10 (SCHEDULER-I19)


def test_lane_order_interface_signatures_and_scalar_budgets():
    """Interface I-10 (HANDOUT-M2b): the entry points keep the merged tree's
    signatures; a width-1 pipeline gets today's stage-level orders; a scalar
    mem_budgets on a wide stage keeps the stage-level split (no lane is ever
    solved against a stage sum); outputs are deterministic."""
    # covers: SCHEDULER-I19
    # WOULD-FAIL-IF: an entry point's signature changed under stage-pieces, a
    # width-1 or scalar-budget wide pipeline got lane orders (or any order but
    # the stage-level solve's split), or two equal calls returned different orders.
    import inspect

    from wantrain.model.gpt import make_config
    from wantrain.runtime.spec import JobSpec

    empty = inspect.Parameter.empty
    sigs = {
        solve: [
            ("spec", empty),
            ("cost_f", empty),
            ("cost_b", empty),
            ("cost_w", empty),
            ("comm", empty),
            ("mem_budgets", None),
            ("mem_mode", "auto"),
        ],
        schedule_for_spec: [
            ("spec", empty),
            ("profiles", empty),
            ("hops", None),
            ("mem_budgets", None),
            ("mem_mode", "auto"),
            ("every_peer", False),
        ],
        default_schedule: [("spec", empty)],
        analytic_costs: [("spec", empty)],
        simulate_slots: [
            ("cfg", empty),
            ("slot_orders", empty),
            ("mb_owner", empty),
            ("peer_weights", empty),
        ],
    }
    for fn, want in sigs.items():
        got = [(p.name, p.default) for p in inspect.signature(fn).parameters.values()]
        assert got == want, fn.__name__

    def stage_level(spec, cost, comm, max_mem):
        mems = [stage_memory_model(spec, b - a) for a, b in spec.partition]
        cfg = ZBConfig(
            spec.stages,
            spec.microbatches,
            *cost,
            comm=comm,
            mem_f=[x[0] for x in mems],
            mem_b=[x[1] for x in mems],
            mem_w=[x[2] for x in mems],
        )
        if max_mem is None:
            sched = zero_bubble(_with_pending_budget(cfg, 2), "given")
        else:
            cfg.max_mem = list(max_mem)
            sched = zero_bubble(cfg, "given")
        return slot_orders(sched.orders, spec.mb_owner, spec.stage_widths)

    cost = ([1.0, 1.2, 0.8, 1.0], [2.0, 2.1, 1.9, 2.4], [1.0, 0.9, 1.1, 1.0])
    flat = JobSpec(model=make_config("toy"), stages=4, microbatches=12).finalize()
    got = solve(flat, *cost, [0.1] * 3, mem_mode="2p")[0]
    assert got == stage_level(flat, cost, [0.1] * 3, None)
    assert got == solve(flat, *cost, [0.1] * 3, mem_mode="2p")[0]

    wide = JobSpec(model=make_config("toy"), stages=4, microbatches=16, stage_widths=[4] * 4)
    wide.finalize()
    m = stage_memory_model(wide, 2)
    scalar = [8 * (abs(m[0]) + max(0.0, m[1]))] * 4
    got, sched, _cfg, _ms = solve(wide, *cost, [0.0] * 3, mem_budgets=scalar)
    assert "lane_stages" not in sched.params
    assert got == stage_level(wide, cost, [0.0] * 3, scalar)
    # control: the same budgets as per-peer lists do give lanes
    per_peer = [[b / w] * w if w > 1 else b for b, w in zip(scalar, wide.stage_widths)]
    _o, lanes, _c, _m = solve(wide, *cost, [0.0] * 3, mem_budgets=per_peer)
    assert lanes.params.get("lane_stages")


# ------------------------------------------------- SCHEDULER-I17: groups, site RTT


def _exchange_sim_pipeline(hosts, sites=None, site_rtt=None):
    """Three wide stages (w 2) of four 50M-parameter units; peers on `hosts`."""
    import math

    from wantrain.scheduler.exchange_sim import Fabric, Peer, StageLayout, Unit, pieces_owners

    S, w, M = 3, 2, 8
    units = tuple(Unit(50_000_000, 2, 2, i, 1.0) for i in range(4))
    sites = sites or ["a"] * (S * w)
    layouts = {
        s: StageLayout(
            units,
            pieces_owners(4, w),
            tuple(
                Peer(f"s{s}p{q}", 1e9, 2e9, 1e12, hosts[s * w + q], sites[s * w + q])
                for q in range(w)
            ),
            boundary_bytes=2e6,
        )
        for s in range(S)
    }
    cfg = ZBConfig(S, M, [0.01] * S, [0.02] * S, [0.01] * S)
    owner = [iwrr(M, [1.0, 1.0]) for _ in range(S)]
    weights = [[1.0, 1.0]] * S
    orders, _sched, _ms = bridge_orders(cfg, owner, [w] * S, weights, "2p")
    fabric = Fabric(
        host_dram={h: 6e9 for h in hosts},
        uplinks={("a", "b"): 0.5e9, ("b", "a"): 0.5e9, ("a", "a"): math.inf},
        k_passes=2.0,
        rtt=0.001,
        site_rtt=site_rtt or {},
    )
    return cfg, orders, weights, layouts, fabric


def test_exchange_sim_groups_stages_that_share_a_resource_and_no_others():
    """One call over every stage equals each resource group simulated on its
    own (max-min sharing decomposes over disjoint resources), and stages that
    share a host still contend: alone, the shared-host stages are faster."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the per-group decomposition split stages that share a host
    # DRAM (losing their contention) or changed the result of independent stages.
    from wantrain.scheduler.exchange_sim import simulate_exchange

    hosts = ["h0", "h1", "h0", "h1", "h2", "h3"]  # stages 0 and 1 share h0 and h1
    cfg, orders, weights, layouts, fabric = _exchange_sim_pipeline(hosts)
    every = simulate_exchange(cfg, orders, weights, layouts, fabric).stages
    pair = simulate_exchange(cfg, orders, weights, {0: layouts[0], 1: layouts[1]}, fabric).stages
    lone = simulate_exchange(cfg, orders, weights, {2: layouts[2]}, fabric).stages
    for s in (0, 1):
        assert every[s].exposure == pytest.approx(pair[s].exposure, rel=1e-9)
    assert every[2] == lone[2]
    alone0 = simulate_exchange(cfg, orders, weights, {0: layouts[0]}, fabric).stages[0]
    assert alone0.exposure < every[0].exposure  # the shared DRAM is modelled


def test_exchange_sim_site_rtt_applies_between_sites_only():
    """A pair's site_rtt replaces the intra-site rtt on transfers between
    those sites only: a stage inside one site is unchanged by it, a stage
    straddling the sites is delayed by it."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: site_rtt were ignored, or applied to transfers inside a site, or a site_rtt
    # below the intra-site rtt were accepted (it would subtract latency in _lat).
    from wantrain.scheduler.exchange_sim import simulate_exchange

    hosts = [f"h{i}" for i in range(6)]
    sites = ["a", "a", "a", "b", "a", "a"]  # stage 1 straddles a and b
    base = _exchange_sim_pipeline(hosts, sites)
    far = _exchange_sim_pipeline(hosts, sites, {("a", "b"): 0.2, ("b", "a"): 0.2})
    got0 = simulate_exchange(*base).stages
    got1 = simulate_exchange(*far).stages
    assert got1[0] == got0[0] and got1[2] == got0[2]
    assert got1[1].tail > got0[1].tail + 0.05
    # a site pair closer than the intra-site rtt (0.001) would shorten transfers: refused
    near = _exchange_sim_pipeline(hosts, sites, {("a", "b"): 0.0005, ("b", "a"): 0.2})
    with pytest.raises(ValueError, match="site_rtt must be >= rtt"):
        simulate_exchange(*near)


def test_exchange_sim_tail_excludes_the_lane_bubble():
    """X_s counts exchange after the stage's compute ends (its last peer's),
    never a faster peer's wait for a slower one: that wait is the lane bubble
    β, already in the period (ADR-0006 T_step = C·(1 + β) + E − window_s +
    T_ovh). Two peers at speeds 1 : 0.9 and 1 : 0.5 owning 4 microbatches each
    expose what equal peers expose less one fold, while their lanes end 34 and
    304 ms apart: the owner left-folds mb 6 then mb 7 (SCHEDULER-I17), and with
    the peers skewed mb 6 is folded before mb 7 arrives, so one fold (not two)
    follows the stage's end. This is what made S2 refuse the toy pieces plan
    whenever the probed CPU speeds differed by ~10% (pieces_plan_oracle, M2b
    round-1 review)."""
    # covers: SCHEDULER-I17, SCHEDULER-I16
    # WOULD-FAIL-IF: the tail run measured each peer's lateness from its own
    # compute end (the faster peer's idle wait for the slower one counted as
    # exposure, so S2 refused any speed skew larger than ε).
    import math

    from wantrain.scheduler.exchange_sim import (
        Fabric,
        Peer,
        StageLayout,
        Unit,
        pieces_owners,
        simulate_exchange,
    )

    w, m, c = 2, 8, 0.04
    units = tuple(Unit(2_000_000, 2, 2, i, 1.0) for i in range(4))
    peers = tuple(Peer(f"p{q}", 1e9, math.inf, 1e12, f"h{q}", "a") for q in range(w))
    cfg = ZBConfig(1, m, [c / 4], [c / 2], [c / 4])
    orders = [
        [[op for mb in range(q, m, w) for op in ((F, mb), (B, mb), (W, mb))] for q in range(w)]
    ]
    fabric = Fabric(
        host_dram={f"h{q}": math.inf for q in range(w)}, uplinks={}, k_passes=0.0, rtt=0.01
    )
    layouts = {0: StageLayout(units, pieces_owners(4, w), peers)}

    def tail(weights):
        got = simulate_exchange(cfg, orders, [weights], layouts, fabric, emission="backward")
        return got.stages[0].tail

    equal = tail([1.0, 1.0])
    assert equal > 0.005  # the exchange itself is exposed: the tail is measured
    for slow in (0.9, 0.5):
        bubble = (m // w) * c * (1.0 + slow) * (1.0 / slow - 1.0)  # lane ends apart
        assert bubble > 2 * equal
        fold = 1_000_000 * 18 / 1e12  # one piece: 2e6 / 2 elements at 18 B, HBM 1 TB/s
        assert tail([1.0, slow]) == pytest.approx(equal - fold, rel=1e-9)


# ============================================================ M4: MoE costs (SCHEDULER-I21)
# MoE dims written here (no GPTConfig field of I-1 needed): h 32, 4 heads,
# S 16, E 8 experts of SwiGLU f 48, k 2, a shared expert f_sh 24 with its
# sigmoid gate, no router bias. Per MoE MLP: router 32·8 = 256, an expert
# 3·32·48 = 4608, the shared expert 3·32·24 = 2304, its gate 32:
# total 256 + 8·4608 + 2304 + 32 = 39,456, active 256 + 2·4608 + 2304 + 32 =
# 11,808. Attention (full multi-head) 4·32² = 4096.
_MOE = dict(
    moe_experts=8,
    moe_top_k=2,
    moe_d_ff=48,
    moe_shared_d_ff=24,
    moe_shared_gate="sigmoid",
    moe_router_bias=False,
    moe_expert="swiglu",
    moe_layers=None,
)


def _moe_cfg(**dense):
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, seq_len=16, **dense)
    for k, v in _MOE.items():
        setattr(cfg, k, v)
    return cfg


def test_m4_plan_moe_active_vs_total():
    """Compute at active parameters (k of E experts), bytes and memory at
    total: moe_layer_costs, layer_costs' 6·P^F + 12·h·s, analytic F/B/W and
    the stage's parameters (all E experts)."""
    # covers: SCHEDULER-I21
    # WOULD-FAIL-IF: compute were charged at all E experts, or memory at k.
    import types

    from wantrain.scheduler import autoconfig
    from wantrain.scheduler.partition import layer_costs
    from wantrain.scheduler.planner import stage_param_split

    c = autoconfig.moe_layer_costs(_MOE, 32, 16, 1)
    assert (c["params_total"], c["params_active"]) == (39456, 11808)
    assert c["rho"] == pytest.approx(39456 / 11808)
    cfg = _moe_cfg()
    assert layer_costs(cfg)[1:3] == [6 * (4096 + 11808) + 12 * 32 * 16] * 2
    f, b, w = analytic_costs(types.SimpleNamespace(model=cfg, recompute=False, partition=[(1, 2)]))
    # F 2·(attn + active) + 4hs; B adds 4hs more and the router's dW (never
    # converted, so in B); W the rest
    assert (f, b, w) == ([8192 + 2048 + 23616], [8192 + 4096 + 23616 + 512], [8192 + 23616 - 512])
    assert stage_param_split(cfg, 1, 3) == (2 * (4096 + 39456), 0)
    dense = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, seq_len=16)
    assert layer_costs(dense)[1] == 32 * (72 * 32 + 12 * 16)  # dense unchanged


def test_m4_plan_moe_count_lane_and_pending_w():
    """The count lane is L·E·8 B per microbatch per peer (8 int64 counts per
    MoE layer here); the pending W of an MoE block is its attention's 6h
    plus k·(2h + 3f) plus 2h + 3f_sh values per token (SPEC-model §2)."""
    # covers: SCHEDULER-I21
    # WOULD-FAIL-IF: the lane were left out, or the pending W charged the dense
    # block's 16h or every expert's rows.
    import types

    from wantrain.scheduler import autoconfig

    c = autoconfig.moe_layer_costs(_MOE, 32, 16, 2)
    assert c["count_lane_bytes"] == 8 * 8
    assert c["pending_w_per_token"] == 2 * (64 + 144) + (64 + 72)
    ns = types.SimpleNamespace(model=_moe_cfg(), mb_size=2, recompute=True)
    f, b, w = stage_memory_model(ns, 2, (1, 3))
    per = 6 * 32 + 2 * (64 + 144) + (64 + 72)  # 744 values per token
    assert b == 2 * 32 * per * 4 and f + b + w == 0
    assert b != 2 * 32 * 16 * 32 * 4  # the dense figure


def test_m4_plan_moe_hf_bank_mem():
    """Under HF banks the routed experts' dW runs in B (their pending W is 0)
    and B leaves each bank's gradient b_m·E·3hf held to W's start:
    stage_memory_model charges it in mem_b and frees it in mem_w, in both
    branches; a census's 3-D parameters are the bank."""
    # covers: SCHEDULER-I21
    # WOULD-FAIL-IF: the bank hold were left out of mem_b or never released, or
    # an HF bank's experts were priced at no compute (outside the linear MACs).
    import types

    from wantrain.scheduler import autoconfig

    c = autoconfig.moe_layer_costs(dict(_MOE, hf_banks=True), 32, 16, 2)
    assert c["bank_bytes"] == 4 * 8 * 3 * 32 * 48 == 147456
    assert c["pending_w_per_token"] == 64 + 72
    assert (c["b"], c["w"]) == (2 * 11808 + 2 * (256 + 2 * 4608), 2 * (11808 - 256 - 2 * 4608))
    shapes = [
        ["self_attn.q_proj.weight", [32, 32]],
        ["mlp.experts.gate_up_proj", [8, 96, 32]],
        ["mlp.experts.down_proj", [8, 32, 48]],
    ]
    layer = dict(params=0, linear_macs_per_token=0, attn_heads=0, head_dim=0, param_shapes=shapes)
    census = [
        dict(params=64 * 32, linear_macs_per_token=0, param_shapes=[["emb", [64, 32]]]),
        layer,
        layer,
        dict(params=32, linear_macs_per_token=2048, param_shapes=[["lm_head", [64, 32]]]),
    ]
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, seq_len=16)
    cfg.arch, cfg.hf_census = "hf", census
    # a bank's compute is k of its E experts (no nn.Linear: outside the
    # census's linear MACs), dX and dW both in B; unknown k is refused
    cfg.hf_config = {"model_type": "mixtral"}
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        analytic_costs(types.SimpleNamespace(model=cfg, recompute=False, partition=[(1, 2)]))
    cfg.hf_config = {"model_type": "mixtral", "num_experts_per_tok": 2}
    active = 2 * (96 * 32 + 32 * 48)  # 9216 MACs per token
    got = analytic_costs(types.SimpleNamespace(model=cfg, recompute=False, partition=[(1, 2)]))
    assert got == ([2.0 * active], [4.0 * active], [0.0])
    bank = 4 * (8 * 96 * 32 + 8 * 32 * 48)
    assert bank == 147456
    pend = 32 * 64 * 4  # q's X and dY per token, 32 tokens (mb 2 x S 16)
    for recompute in (True, False):
        ns = types.SimpleNamespace(model=cfg, mb_size=2, recompute=recompute)
        f, b, w = stage_memory_model(ns, 1, (1, 2))
        assert w == -(pend + bank) if not recompute else w == -(f + pend + bank)
        assert f + b + w == 0
        without = types.SimpleNamespace(model=cfg, mb_size=2, recompute=recompute)
        layer["param_shapes"] = shapes[:1]
        assert stage_memory_model(without, 1, (1, 2))[1] == b - bank
        layer["param_shapes"] = shapes


def test_m4_plan_moe_units_per_projection_and_bias_unit():
    """SCHEDULER-I17's MoE units, in the runtime's order (pieces.exchange_units,
    BA-466): the small-tensor unit first (the norms 4·32, router 256 and the 8
    selection biases: 392, a count lane of 64 B, the biases at ZeRO-1 index
    n_params + j), the attention's linears with their biases (3072 + 96,
    1024 + 32), one unit per expert projection holding all E experts (12,288
    elements each, compute at k of E), the shared expert's projections (768
    each) and its gate (32). A MoE config whose dense part is not the builtin
    block is refused 'unpriced'."""
    # covers: SCHEDULER-I21
    # WOULD-FAIL-IF: a MoE config is priced with the dense block's fc1/fc2 units.
    from wantrain.scheduler.planner import _arch_priced, _stage_units

    cfg = _moe_cfg()
    assert _arch_priced(cfg) is None
    units, idx, blocks = _stage_units(cfg, 1, 2, 2, 2)
    assert blocks == 1
    assert [u.params for u in units] == [392, 3168, 1056, 12288, 12288, 12288, 768, 768, 768, 32]
    assert [u.lane for u in units] == [64, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert [u.flops for u in units[3:6]] == [2 * 12288 / 8] * 3
    small = units[0]
    assert small.grad_bytes() == 2 * 392 + 64
    # ln1 0-1, qkv 2-3, proj 4-5, ln2 6-7, router 8, experts 9..32, shared
    # 33-35, gate 36; the selection biases after the 37 parameters
    assert idx[0] == ((32, 0), (32, 1), (32, 6), (32, 7), (256, 8), (8, 37))
    assert idx[1:3] == [((3072, 2), (96, 3)), ((1024, 4), (32, 5))]
    assert idx[3] == tuple((1536, 9 + 3 * e) for e in range(8))
    assert idx[5] == tuple((1536, 11 + 3 * e) for e in range(8))
    assert idx[6:] == [((768, 33),), ((768, 34),), ((768, 35),), ((32, 36),)]
    dense_units, _i, _b = _stage_units(GPTConfig(n_layer=2, d_model=32, n_head=4), 1, 2, 2, 2)
    assert [u.params for u in dense_units] == [3072, 1024, 4096, 4096]
    assert _arch_priced(_moe_cfg(norm="rmsnorm")) == "norm='rmsnorm'"


def test_m4_plan_moe_i16_bank_device_term():
    """SCHEDULER-I16's device memory gains m_BW,k·b_m·P_bank,s on a stage
    with HF banks: m_BW,k from the emitted order (the most microbatches
    between B's end and W's start), M_k before an order is solved; it pushes
    a peer over its budget into MEM where the same stage without banks fits."""
    # covers: SCHEDULER-I21
    # WOULD-FAIL-IF: the bank hold is left out of the device-memory condition.
    import dataclasses

    from wantrain.scheduler import admission
    from wantrain.scheduler.exchange_sim import Unit, pieces_owners

    one_by_one = [(F, 0), (B, 0), (W, 0), (F, 1), (B, 1), (W, 1)]
    two_held = [(F, 0), (F, 1), (B, 0), (B, 1), (W, 0), (W, 1)]
    assert admission.m_bw_from_order(one_by_one) == 1
    assert admission.m_bw_from_order(two_held) == 2
    P = 10**9
    peers = tuple(
        admission.Gpu(f"p{q}", f"h{q}", "s0", "cuda", "X", 100.0, 80 * 10**9) for q in range(2)
    )
    st = admission.Stage(
        0,
        (Unit(10**6, 4, 4, 0, 1.0),),
        pieces_owners(1, 2),
        peers,
        1.0,
        0.0,
        16,
        32,
        1,
        4,
        bank_elems=P,
    )
    for m_bw, want in ((1, 4 * P), (2, 8 * P), (None, 3 * 4 * P)):
        assert admission.device_terms(st, 0, 3, 2, 6, m_bw)["hf_bank"] == want
    plain = dataclasses.replace(st, bank_elems=0)
    assert "hf_bank" not in admission.device_terms(plain, 0, 3, 2, 6)
    base = sum(admission.device_terms(plain, 0, 3, 2, 6).values())
    vram = int((base + 6 * P) / 0.88) + 1  # room for one held bank, not two
    small = dataclasses.replace(
        st, peers=tuple(dataclasses.replace(g, vram_bytes=vram) for g in peers)
    )
    owned = {0: [3, 3]}
    assert admission.device_memory([dataclasses.replace(small, bank_elems=0)], owned, 2, 6) == []
    assert admission.device_memory([small], owned, 2, 6, {0: [1, 1]}) == []
    refused = admission.device_memory([small], owned, 2, 6, {0: [2, 1]})
    assert [(r.kind, r.where) for r in refused] == [("MEM", "stage 0 peer p0")]


def test_m4_plan_moe_rho_rationale():
    """A plan's rationale names ρ_s = P_s/P^F_s for every stage of every
    candidate pieces are refused for (SCHEDULER-I21)."""
    # covers: SCHEDULER-I21
    # WOULD-FAIL-IF: ρ were computed at total over total (1.0) or left out.
    from wantrain.scheduler.partition import proportional_partition
    from wantrain.scheduler.planner import Fleet, Node, _choose, _plan_depth

    nodes = [
        Node(f"n{i}", tflops=80.0, vram_bytes=24 << 30, compute_cap=(8, 9), device_class=ONE_CLASS)
        for i in range(4)
    ]
    # a toy block at 80 TFLOP/s needs ~138 Gbps for the 2x margin: on a slower
    # link the precision ladder admits nothing and there is no recipe to price
    fleet = Fleet(nodes=nodes, link_mbps=200_000.0, rtt_ms=5.0, duplex="per_direction")
    dense = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, seq_len=16)
    base = _plan_depth(dense, fleet, steps=10, device="cuda")
    assert base["spec"] is not None, base["rationale"]
    cfg = _moe_cfg()
    assert proportional_partition(cfg, [160.0, 160.0]) == [[0, 2], [2, 4]]
    out = _choose(cfg, fleet, base, [0, 1, 2, 3], "cuda", True, None)
    rho = out["rationale"]["moe_rho"]
    assert rho and set(rho) == {
        c["name"] for c in out["candidates"] if c["shard"] == "pieces" and c["w"] > 1
    }
    # each stage: the Embed (or the Head) 64·32 and one MoE block
    want = round((2048 + 4096 + 39456) / (2048 + 4096 + 11808), 4)
    assert rho["S 2 w 2 pieces + native"] == [want, want]


def test_m4_plan_moe_default_expert_width_matches_the_module():
    """GPTConfig's default moe_d_ff (None) prices the experts at the width
    arch.MoEMLP builds (moe_d_ff, else d_ff, else the SwiGLU default): the
    MLP's total parameters equal the real module's, and the planner's
    layer_costs and stage parameters run on such a config."""
    # covers: SCHEDULER-I21
    # WOULD-FAIL-IF: moe_d_ff None reached moe_layer_costs (int(None) raised
    # for every MoE job left at the default width) or the experts were priced
    # at another width than the module holds.
    from wantrain.model.arch import MoEMLP
    from wantrain.scheduler import autoconfig
    from wantrain.scheduler.partition import layer_costs
    from wantrain.scheduler.planner import stage_param_split

    for d_ff, router_bias, shared in ((None, False, 0), (64, True, 24)):
        cfg = GPTConfig(
            vocab_size=64,
            n_layer=2,
            n_head=4,
            d_model=32,
            seq_len=16,
            d_ff=d_ff,
            moe_experts=4,
            moe_top_k=2,
            moe_router_bias=router_bias,
            moe_shared_d_ff=shared,
        )
        moe = autoconfig.moe_fields(cfg)
        assert moe["moe_d_ff"] == (256 if d_ff is None else 64)
        c = autoconfig.moe_layer_costs(moe, 32, 16, 1)
        real = sum(p.numel() for p in MoEMLP(cfg, 0).parameters())
        assert c["params_total"] == real
        assert stage_param_split(cfg, 1, 2) == (4 * 32 * 32 + real, 0)
        assert len(layer_costs(cfg)) == 4
