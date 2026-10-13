"""RUNTIME-I26: recut_states moves each entry's tensors and optimizer state to the stage that now holds it, the same
objects renumbered, and refuses what it cannot re-cut.

WOULD-FAIL-IF: a tensor were copied or reordered across entries, a moment landed on another parameter, the pins kept
the old partition, losses were lost, or a mismatched stage count, step, chunk or parameter count passed.
"""

import pytest
import torch

from wantrain.runtime.recut import recut_states

NAMES = [["e"], ["w", "b"], ["w", "b"], ["w", "b"], ["h"]]  # embed, 3 blocks, head


def _wave(r, part, step=4):
    a, b = part[r]
    chunk, params = {}, []
    for g in range(a, b):
        for n in NAMES[g]:
            chunk[f"{g - a}.{n}"] = torch.full((2,), float(10 * g + len(n)))
            params.append((g, n))
    state = {
        i: {"step": torch.tensor(5.0), "exp_avg": torch.full((2,), float(100 * g + i))}
        for i, (g, _) in enumerate(params)
    }
    return {
        "step": step,
        "chunks": {r: chunk},
        "opt": {"state": state, "param_groups": [{"lr": 0.1, "params": list(range(len(params)))}]},
        "rng": torch.tensor([r]),
        "losses": {f"{step}:{r}": float(r)},
        "pins": {"partition": [list(x) for x in part], "lr": 0.1},
    }


def test_entries_and_their_moments_move_with_their_layers():
    old, new = [[0, 2], [2, 4], [4, 5]], [[0, 1], [1, 3], [3, 5]]
    states = [_wave(r, old) for r in range(3)]
    out = recut_states(states, old, new, NAMES)
    # entry 1 (block "w","b") moved from stage 0 to stage 1 as its local entry 0: the same tensor objects
    assert out[1]["chunks"][1]["0.w"] is states[0]["chunks"][0]["1.w"]
    assert out[2]["chunks"][2]["0.w"] is states[1]["chunks"][1]["1.w"]
    assert set(out[0]["chunks"][0]) == {"0.e"}
    # stage 1's first parameter is entry 1's "w", whose moments were stage 0's index 1
    assert out[1]["opt"]["state"][0] is states[0]["opt"]["state"][1]
    assert out[1]["opt"]["param_groups"][0]["params"] == [0, 1, 2, 3]
    assert out[2]["opt"]["param_groups"][0]["lr"] == 0.1
    assert all(o["pins"]["partition"] == new and o["step"] == 4 for o in out)
    assert out[0]["losses"] == {"4:0": 0.0, "4:1": 1.0, "4:2": 2.0}
    assert torch.equal(out[2]["rng"], torch.tensor([2]))


def test_what_cannot_be_re_cut_is_refused():
    old = [[0, 2], [2, 4], [4, 5]]
    states = [_wave(r, old) for r in range(3)]
    with pytest.raises(ValueError):
        recut_states(states, old, [[0, 3], [3, 5]], NAMES)  # another stage count
    with pytest.raises(ValueError):
        recut_states(states[:2] + [_wave(2, old, step=5)], old, old, NAMES)  # mixed steps
    bad = [_wave(r, old) for r in range(3)]
    bad[1]["opt"]["param_groups"][0]["params"] = [0]
    with pytest.raises(ValueError):
        recut_states(bad, old, old, NAMES)  # parameter count
    with pytest.raises(ValueError):
        recut_states(states, old, [[0, 2], [3, 4], [4, 5]], NAMES)  # not contiguous


def test_refined_partition_moves_entries_off_the_slow_stage_and_needs_a_gain():
    from wantrain.coordinator.coordinator import refined_partition
    from wantrain.model.gpt import make_config
    from wantrain.runtime.spec import JobSpec

    spec = JobSpec(model=make_config("toy"), stages=3, microbatches=8).finalize()
    even = [[{"t_f": 0.003, "t_b": 0.005, "t_w": 0.002}] for _ in range(3)]
    slow = [even[0], [{"t_f": 0.012, "t_b": 0.020, "t_w": 0.008}], even[2]]
    always = lambda s, a, b: True  # noqa: E731
    new, before, after = refined_partition(spec, slow, always, 0.10)
    assert new[1][1] - new[1][0] < spec.partition[1][1] - spec.partition[1][0]
    assert after < 0.9 * before
    assert refined_partition(spec, slow, always, 0.99) is None  # not that much better
    assert refined_partition(spec, slow, lambda s, a, b: False, 0.10) is None  # nothing fits


def test_slow_slots_and_recut_refusal():
    # covers: CONTROL-I27
    from wantrain.coordinator.coordinator import recut_refusal, slow_slots, stage_compute_ms
    from wantrain.model.gpt import make_config
    from wantrain.runtime.spec import JobSpec

    hist = {(0, 0): [10, 10, 14, 14, 14], (1, 0): [10, 14, 10, 14, 14]}
    assert slow_slots(hist, {(0, 0): 10.0, (1, 0): 10.0}, 1.3, 3) == [((0, 0), 1.4)]
    assert slow_slots(hist, {}, 1.3, 3) == []
    assert (
        stage_compute_ms({"ops": {"F": {"med_ms": 3}, "B": {"med_ms": 5}, "W": {"med_ms": 2}}})
        == 10
    )
    toy = make_config("toy")
    assert recut_refusal(JobSpec(model=toy, stages=3, microbatches=8).finalize()) is None
    wide = JobSpec(model=toy, stages=3, microbatches=8, stage_widths=[1, 2, 1]).finalize()
    assert recut_refusal(wide) == "a wide stage"


def test_the_store_swaps_a_wave_whole_and_drops_the_newer_ones(tmp_path):
    from wantrain.fault_tolerance.checkpoint import CheckpointStore, validate_wave

    store = CheckpointStore(str(tmp_path))
    for step in (3, 4):
        store.replace_wave((0, 0), step, {"step": step, "v": "old"})
    store.replace_wave((0, 0), 3, {"step": 3, "v": "new"})
    m = validate_wave(store.wave_path((0, 0), 3), 3)
    assert [f["name"] for f in m["files"]] == ["state.pt"]
    got = torch.load(store.wave_path((0, 0), 3) + "state.pt", weights_only=True)
    assert got == {"step": 3, "v": "new"}
    store.drop_after([(0, 0)], 3)
    assert store.index() == {(0, 0): [3]}


class _Node:
    def __init__(self, nid, mem):
        self.node_id, self._mem = nid, mem

    def memory_bytes(self):
        return self._mem


def test_dp_feasibility_uses_measured_peaks_and_the_optimizers_state_bytes():
    # covers: SCHEDULER-I24
    # WOULD-FAIL-IF: a measured peak were ignored (the analytic model priced instead), a stage near its budget took
    # more blocks, or adamw-host ranges were priced at adamw's 16 bytes per element
    from wantrain.coordinator.coordinator import _stage_fits
    from wantrain.model.gpt import make_config
    from wantrain.runtime import optim
    from wantrain.runtime.spec import JobSpec
    from wantrain.scheduler.planner import stage_param_split

    spec = JobSpec(
        model=make_config("toy"), stages=3, microbatches=8, optimizer="adamw-host"
    ).finalize()
    groups = [[_Node("a", 1 << 34)], [_Node("b", 1 << 34)], [_Node("c", 1 << 34)]]
    (a1, b1) = spec.partition[1]
    bpp = optim.state_bytes_per_param("adamw-host")
    state1 = bpp * stage_param_split(spec.model, a1, b1)[0]
    block = bpp * stage_param_split(spec.model, a1, a1 + 1)[0]
    roomy = {"b": ((a1, b1), state1 + 1000, int(10 * (state1 + 1000)))}
    tight = {"b": ((a1, b1), state1 + 1000, int((state1 + 1000) / 0.97) + 1)}
    assert _stage_fits(spec, groups, roomy)(1, a1, b1 + 1) is True
    assert _stage_fits(spec, groups, tight)(1, a1, b1) is True
    assert _stage_fits(spec, groups, tight)(1, a1, b1 + 1) is False  # one more block: over
    assert block > 0
    # no peak: the planner's model, at the optimizer's bytes per element
    import wantrain.scheduler.planner as planner

    seen = {}
    orig = planner.stage_fits

    def spy(*a, **k):
        seen["state_bpp"] = k.get("state_bpp")
        return orig(*a, **k)

    planner.stage_fits = spy
    try:
        _stage_fits(spec, groups, {})(0, 0, 2)
    finally:
        planner.stage_fits = orig
    assert seen["state_bpp"] == 8
