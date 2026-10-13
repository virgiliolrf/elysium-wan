"""Fault tolerance under real failures, on a real local cluster.

Worker disappears -> failure detected -> task reassigned to a spare -> every
slot resumes from the newest common checkpoint wave -> training continues.
On a CPU fleet the healed run's loss sequence must be bit-identical to the
single-device reference: recovery may cost time, never correctness.

WOULD-FAIL-IF: a heal resumed from mismatched waves, re-planned the
partition, lost the loss history, or a slot restarted at step 0 while its
peers resumed.
"""

import os
import time

import pytest
from helpers.cluster import LocalCluster

from wantrain.coordinator.coordinator import Job
from wantrain.launch.local import ensure_reference
from wantrain.model.gpt import make_config
from wantrain.runtime.spec import JobSpec
from wantrain.validation.compare import compare

pytestmark = pytest.mark.slow


def _recipe(steps=40, **kw):
    base = dict(
        model=make_config("toy"),
        steps=steps,
        microbatches=8,
        recompute=True,
        wire_codec="planes",
        ckpt_every=2,
    )
    base.update(kw)
    spec = JobSpec(**base)
    # A little emulated WAN latency keeps the run long enough to kill mid-way.
    spec.links = [
        dict(latency_ms=15.0, bw_mbps=0.0, jitter_ms=0.0, jitter_seed=0) for _ in range(spec.stages)
    ]
    return spec


def _why(res):
    return f"{res.reason}; heals={res.heals}; events=" + " | ".join(e[2] for e in res.events[-25:])


def _bitwise(res, recipe, tmp_path):
    ref = ensure_reference(JobSpec.from_json(recipe.to_json()), str(tmp_path / "ref"))
    r = compare(ref, os.path.join(res.run_dir, "losses.json"))
    assert r["ok"] and r["compared"] == recipe.steps * recipe.microbatches, r


def _cluster(tmp_path, port_base, n, **cfg):
    c = LocalCluster(tmp_path, port_base, probe=False, **cfg)
    for i in range(n):
        c.add_worker(f"w{i}")
    c.wait_registered(n)
    return c


def test_dead_worker_is_replaced_and_the_run_stays_exact(tmp_path, port_base):
    # covers: CONTROL-I10
    c = _cluster(tmp_path, port_base, 3)  # 2 slots + 1 spare
    try:
        recipe = _recipe(stages=2)
        c.run_async(Job(spec=recipe))
        c.wait_step(5)
        victim = c.name_of((1, 0))
        c.kill_worker(victim)
        res = c.join()
        assert res.ok, _why(res)
        assert len(res.heals) == 1 and res.heals[0]["from"] == victim, _why(res)
        assert res.heals[0]["resumed_from_step"] is not None
        assert res.heals[0]["resumed_from_step"] >= 1
        assert res.attempts == 2
        _bitwise(res, recipe, tmp_path)
    finally:
        c.close()


def test_killed_stage_process_is_convicted_and_healed(tmp_path, port_base):
    # covers: CONTROL-I10
    c = _cluster(tmp_path, port_base, 3)
    try:
        recipe = _recipe(stages=2)
        c.run_async(Job(spec=recipe))
        c.wait_step(5)
        victim = c.name_of((0, 0))
        c.kill_stage(victim)  # the agent survives, its stage does not
        res = c.join()
        assert res.ok, _why(res)
        assert len(res.heals) == 1 and res.heals[0]["from"] == victim, _why(res)
        assert "no done marker" in res.heals[0]["reason"] or "gone" in res.heals[0]["reason"]
        _bitwise(res, recipe, tmp_path)
    finally:
        c.close()


def test_frozen_stage_is_restarted_in_place(tmp_path, port_base):
    """Every process alive and heartbeating, yet no progress (what a broken WAN
    path between two workers looks like from the control plane): restart all
    slots from the last common wave on the same nodes, no replacement."""
    # covers: CONTROL-I10
    c = _cluster(tmp_path, port_base, 2, stall_floor_s=5.0)
    try:
        recipe = _recipe(stages=2)
        c.run_async(Job(spec=recipe))
        c.wait_step(5)
        c.freeze_stage(c.name_of((1, 0)))
        res = c.join()
        assert res.ok, _why(res)
        assert res.restarts == 1 and res.heals == []
        _bitwise(res, recipe, tmp_path)
    finally:
        c.close()


def test_control_plane_blip_reconnects_without_recovery(tmp_path, port_base):
    """The coordinator's connection to a worker drops; training on the data
    plane never notices, the agent dials back inside the grace, and the run
    finishes with no heal and no restart."""
    # covers: CONTROL-I10
    c = _cluster(tmp_path, port_base, 2, reconnect_grace_s=20.0)
    try:
        recipe = _recipe(stages=2)
        c.run_async(Job(spec=recipe))
        c.wait_step(4)
        w = c.coord.workers[c.coord.run.slots[(0, 0)]]
        w.ch.close()  # sever the control connection
        deadline = time.time() + 20
        while time.time() < deadline and not (w.connected and w.ch is not None and not w.ch.closed):
            time.sleep(0.1)
        res = c.join()
        assert res.ok, _why(res)
        assert res.heals == [] and res.restarts == 0 and res.attempts == 1
        assert any("reconnected" in e[2] for e in res.events)
        _bitwise(res, recipe, tmp_path)
    finally:
        c.close()


def test_no_spare_fails_with_the_reason(tmp_path, port_base):
    # covers: CONTROL-I10
    c = _cluster(tmp_path, port_base, 2)
    try:
        c.run_async(Job(spec=_recipe(stages=2)))
        c.wait_step(3)
        c.kill_worker(c.name_of((1, 0)))
        res = c.join()
        assert not res.ok
        assert "no registered spare" in res.reason
    finally:
        c.close()


def test_wide_stage_peer_death_heals_every_shard(tmp_path, port_base):
    """A peer of a ZeRO-1 stage dies: it held one optimizer shard. Every slot,
    including its surviving stage-mate, resumes from the common wave."""
    # covers: CONTROL-I10
    c = _cluster(tmp_path, port_base, 4)  # 3 slots + 1 spare
    try:
        recipe = _recipe(stages=2, stage_widths=[2, 1], wire_codec="raw")
        c.run_async(Job(spec=recipe))
        c.wait_step(5)
        victim = c.name_of((0, 1))
        c.kill_worker(victim)
        res = c.join()
        assert res.ok, _why(res)
        assert len(res.heals) == 1 and res.heals[0]["slot"] == [0, 1]
        _bitwise(res, recipe, tmp_path)
    finally:
        c.close()
