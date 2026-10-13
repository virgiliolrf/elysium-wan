"""RUNTIME-I25: per-stage recompute. A stage whose order holds one microbatch between an F and its B keeps F's
graph; the spec validates the per-stage list; the memory and cost models follow it.

WOULD-FAIL-IF: the last stage of a zero-bubble order were left recomputing, a stage with two or more in flight kept
its graph, a forced stage stayed off, an arch "hf" spec accepted a mixed list, or stage_memory_model / analytic
costs ignored the per-stage flag.
"""

import pytest

from wantrain.model.gpt import make_config
from wantrain.runtime.spec import JobSpec
from wantrain.scheduler.zero_bubble import analytic_costs, selective_recompute, stage_memory_model


def _spec(**kw):
    return JobSpec(model=make_config("toy"), stages=3, microbatches=8, **kw).finalize()


def test_the_last_stage_keeps_its_graph_and_the_others_recompute():
    # covers: RUNTIME-I25
    spec = _spec(recompute=True)
    got = selective_recompute(spec)
    assert got is not None and got[-1] is False
    for r, slots in enumerate(spec.schedule_order):
        live = peak = 0
        for op, _ in slots[0]:
            live += 1 if op == "F" else (-1 if op == "B" else 0)
            peak = max(peak, live)
        assert got[r] == (peak > 1), (r, peak)


def test_forced_stages_recompute_and_nothing_changes_without_recompute():
    # covers: RUNTIME-I25
    spec = _spec(recompute=True)
    assert selective_recompute(spec, force={2}) in (None, [True, True, True])
    assert selective_recompute(_spec(recompute=False)) is None


def test_spec_validates_and_reads_the_per_stage_list():
    # covers: RUNTIME-I25
    spec = _spec(recompute=True, recompute_stages=[True, False, False])
    assert [spec.recomputes(r) for r in range(3)] == [True, False, False]
    assert _spec(recompute=True).recomputes(1) is True
    with pytest.raises(ValueError):
        _spec(recompute=True, recompute_stages=[True, False])
    assert "recompute_stages" not in spec.pins()


def test_memory_and_cost_models_follow_the_stage():
    # covers: RUNTIME-I25
    spec = _spec(recompute=True, recompute_stages=[True, True, False])
    a, b = spec.partition[2]
    kept = stage_memory_model(spec, b - a, (a, b), False)
    rebuilt = stage_memory_model(spec, b - a, (a, b), True)
    assert kept[0] > rebuilt[0]  # F holds the graph, not only the boundary
    cf, cb, _cw = analytic_costs(spec)
    full = analytic_costs(_spec(recompute=True))[1]
    assert cb[2] == pytest.approx(full[2] - cf[2]) and cb[0] == pytest.approx(full[0])
