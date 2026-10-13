"""A measured fit overshoot feeds back into the cut (coordinator.shrink_for_overshoot): the overflowing node plans
with less memory, so the planner moves layers off it instead of halving a microbatch it cannot halve.

WOULD-FAIL-IF: the feedback left the planner's view of the node unchanged, shrank a node that fit, or the planner
kept the same cut for the shrunk node (the 2026-10-12 MoE run: three 0.64B-parameter MoE layers measured 22.49 GiB
against a 19.66 GiB budget on a 24 GB RTX 4090).
"""

from wantrain.coordinator.coordinator import MEM_FEEDBACK, shrink_for_overshoot
from wantrain.model.gpt import GPTConfig
from wantrain.scheduler.planner import Fleet, Node, Overrides, plan_job

GIB = 2**30
GB = [23.2, 23.1, 47.0, 23.1, 23.1, 47.0, 47.0]  # the run's chain, in GB as the workers reported


def _moe():
    return GPTConfig(
        vocab_size=50257,
        d_model=3072,
        n_layer=28,
        n_head=24,
        seq_len=1024,
        compute_dtype="bf16",
        norm="rmsnorm",
        mlp="swiglu",
        bias=False,
        pos="rope",
        moe_experts=8,
        moe_top_k=2,
        moe_d_ff=8192,
        moe_score="sigmoid",
        moe_select="score_bias",
        moe_norm=True,
        moe_bias_rate=0.001,
    )


def _cut(vram):
    fleet = Fleet(
        nodes=[
            Node(
                f"n{i}",
                tflops=150.0,
                vram_bytes=v,
                compute_cap=(8, 9),
                device_class="cuda:RTX 4090",
            )
            for i, v in enumerate(vram)
        ],
        link_mbps=173.0,
        rtt_ms=35.0,
        duplex="per_direction",
    )
    ov = Overrides(
        recompute=True,
        compute_dtype="bf16",
        boundary_dtype="bf16",
        mb_size=1,
        microbatches=32,
        extras={"optimizer": "adamw-host"},
    )
    spec = plan_job(_moe(), fleet, steps=2, device="cuda", require_margin=False, overrides=ov)[
        "spec"
    ]
    return [b - a for a, b in spec.partition]


def test_a_measured_overshoot_moves_layers_off_the_node():
    # covers: CONTROL-I26
    nodes = [{"vram_bytes": int(g * 1e9), "label": f"n{i}"} for i, g in enumerate(GB)]
    before = _cut([n["vram_bytes"] for n in nodes])
    assert before[3] == 3  # what the run planned for its 4th node
    over = {"id3": (int(22.49 * GIB), int(19.66 * GIB))}
    notes = shrink_for_overshoot(nodes, [f"id{i}" for i in range(7)], over)
    assert len(notes) == 1 and "n3" in notes[0]
    shrunk = int(23.1e9) - int(MEM_FEEDBACK * (22.49 - 19.66) * GIB)
    assert nodes[3]["vram_bytes"] == shrunk
    assert [n["vram_bytes"] for i, n in enumerate(nodes) if i != 3] == [
        int(g * 1e9) for i, g in enumerate(GB) if i != 3
    ]
    after = _cut([n["vram_bytes"] for n in nodes])
    assert after[3] < before[3] and sum(after) == sum(before)


def _moe_spec(microbatches=32):
    from wantrain.runtime.spec import JobSpec

    cfg = GPTConfig(
        vocab_size=50257,
        d_model=2048,
        n_layer=28,
        n_head=16,
        seq_len=1024,
        compute_dtype="bf16",
        norm="rmsnorm",
        mlp="swiglu",
        bias=False,
        pos="rope",
        moe_experts=8,
        moe_top_k=2,
        moe_d_ff=5632,
        moe_score="sigmoid",
        moe_select="score_bias",
        moe_norm=True,
        moe_bias_rate=0.001,
    )
    return JobSpec(
        model=cfg,
        stages=7,
        microbatches=microbatches,
        recompute=True,
        optimizer="adamw-host",
        partition=[[0, 5], [5, 9], [9, 13], [13, 17], [17, 21], [21, 25], [25, 30]],
        device="cuda",
    ).finalize()


def test_a_measured_overshoot_calibrates_that_slot_s_schedule_budget():
    # covers: CONTROL-I26
    # The 2026-10-12 run: an 8.4B MoE (d 2048, 4 MoE layers per stage, 32 microbatches) OOMed stage 0 on a
    # 24 GB card; a cut taken off its budget did not move the schedule (the model priced its pending microbatches
    # at about half their size). The calibrated budget does: fewer microbatches in flight, under the new budget.
    from types import SimpleNamespace

    from wantrain.coordinator.coordinator import (
        _cap_budgets,
        calibrated_budget,
        scheduled_pending_peak,
    )
    from wantrain.resources.fit import peak_pending
    from wantrain.scheduler import zero_bubble

    spec = _moe_spec()
    card = int(23.54 * GIB)
    fit_budget = int(0.85 * card)
    before_peak = scheduled_pending_peak(spec, 0, 0)
    before_hold = peak_pending(spec.schedule_order[0][0])
    cap = calibrated_budget(spec, 0, 0, card, fit_budget, oom=True)
    assert 0 < cap < before_peak
    cf, cb, cw = zero_bubble.analytic_costs(spec)
    budgets = [int(40e9)] * spec.stages
    nodes = [[SimpleNamespace(node_id=f"n{r}")] for r in range(spec.stages)]
    held = _cap_budgets(budgets, nodes, {"n0": cap})
    assert held[0] == cap and held[1:] == budgets[1:]
    orders, _s, _cfg, _ms = zero_bubble.solve(spec, cf, cb, cw, [0.0] * (spec.stages - 1), held)
    spec.schedule_order = orders
    assert scheduled_pending_peak(spec, 0, 0) <= cap + 1
    assert peak_pending(orders[0][0]) < before_hold
    assert _cap_budgets(budgets, nodes, {}) is budgets
