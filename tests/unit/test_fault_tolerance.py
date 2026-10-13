"""Failure detection, checkpoint waves, replacement choice."""

# WOULD-FAIL-IF: the liveness judge convicts a healthy or reconnecting node or misses a killed or silent one, the checkpoint store accepts a corrupt shard or prunes a wave still needed, or replacement ignores the recipe or the dead list

import hashlib
import json
import os
import re
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from helpers import wave_experiment

import wantrain.coordinator.coordinator as coord_mod
import wantrain.worker.agent as agent_mod
from wantrain.coordinator.coordinator import Coordinator, CoordinatorConfig, Worker, _Run
from wantrain.fault_tolerance import checkpoint
from wantrain.fault_tolerance.checkpoint import (
    CheckpointStore,
    engine_shard_path,
    patch_spec_for_resume,
    sha256_file,
)
from wantrain.fault_tolerance.liveness import PEER_LOST, LivenessJudge, SlotState
from wantrain.fault_tolerance.replacement import find_replacement
from wantrain.resources.node import NodeInfo
from wantrain.runtime.spec import JobSpec
from wantrain.transport.control import ChannelClosed
from wantrain.worker.agent import WorkerAgent

T0 = 1_000_000.0


def slots(n=3, **kw):
    out = []
    for r in range(n):
        st = SlotState(
            slot=(r, 0),
            node_id=f"n{r}",
            launched=True,
            last_heartbeat=T0,
            started_at=T0,
            stage_alive=True,
        )
        for k, v in kw.items():
            setattr(st, k, v)
        out.append(st)
    return out


def judge(**kw):
    base = dict(
        heartbeat_timeout_s=10,
        reconnect_grace_s=20,
        stall_floor_s=30,
        strike_gap_s=5,
        first_step_timeout_s=60,
    )
    base.update(kw)
    return LivenessJudge(**base)


def test_healthy_fleet_is_ok_and_all_clean_exits_are_done():
    # covers: CONTROL-I7
    j = judge()
    s = slots()
    for st in s:
        st.progress = [(0, T0 + 1)]
    assert j.judge(s, T0 + 2).kind == "ok"
    for st in s:
        st.exit_code, st.done_marker = 0, True
    assert j.judge(s, T0 + 3).kind == "done"


def test_process_killed_from_outside_convicts_at_once():
    # covers: CONTROL-I7
    s = slots()
    s[1].exit_code, s[1].done_marker = -9, False
    v = judge().judge(s, T0 + 1)
    assert v.kind == "convict" and v.slot == (1, 0)


def test_engine_exception_is_a_run_failure_not_a_dead_box():
    # covers: CONTROL-I7
    s = slots()
    s[2].exit_code, s[2].done_marker, s[2].log_tail = 1, True, "ValueError: bad recipe\nWT_DONE 1"
    v = judge().judge(s, T0 + 1)
    assert v.kind == "failed" and v.slot == (2, 0)


def test_lost_connection_waits_for_the_reconnect_grace():
    # covers: CONTROL-I7
    j = judge()
    s = slots()
    s[0].connected, s[0].disconnected_at = False, T0
    assert j.judge(s, T0 + 5).kind == "suspect"  # a WAN blip: wait
    v = j.judge(s, T0 + 25)
    assert v.kind == "convict" and v.slot == (0, 0)


def test_two_simultaneous_losses_are_not_two_dead_boxes():
    # covers: CONTROL-I7
    s = slots()
    for st in s[:2]:
        st.connected, st.disconnected_at = False, T0
    v = judge().judge(s, T0 + 25)
    assert v.kind == "failed" and "network" in v.why


def test_heartbeat_silence_needs_two_strikes():
    # covers: CONTROL-I7
    j = judge()
    s = slots()
    s[1].last_heartbeat = T0 - 30
    v1 = j.judge(s, T0)
    assert v1.kind == "suspect" and v1.slot == (1, 0)
    assert j.judge(s, T0 + 2).kind == "suspect"  # too soon for a second strike
    v3 = j.judge(s, T0 + 6)
    assert v3.kind == "convict" and v3.slot == (1, 0)


def test_peer_lost_points_at_the_silent_slot():
    # covers: CONTROL-I7
    j = judge()
    s = slots()
    s[0].exit_code, s[0].done_marker = 1, True
    s[0].log_tail = "RuntimeError: reader: peer connection lost (ConnectionError)\nWT_DONE 1"
    s[2].last_heartbeat = T0 - 60  # silent
    assert j.judge(s, T0).kind == "suspect"
    v = j.judge(s, T0 + 6)
    assert v.kind == "convict" and v.slot == (2, 0)


def test_a_dead_slots_late_exit_inside_the_grace_convicts_it_not_the_fleet():
    # covers: CONTROL-I7
    # WOULD-FAIL-IF: S3 declares the fleet "stalled" (restart in place) the moment a peer-lost exit
    # is in, so the killed slot's own EXIT, in flight on another worker's channel, lands too late
    # and the dead process is never convicted (S2); or the grace never ends and a real data-plane
    # break with every worker alive is never restarted.
    j = judge()
    s = slots()
    for st in s:
        st.last_heartbeat = T0
    s[1].exit_code, s[1].done_marker = 1, True
    s[1].log_tail = "RuntimeError: reader: peer connection lost\nWT_DONE 1"
    v = j.judge(s, T0)
    assert v.kind == "suspect" and sorted(v.slots) == [(0, 0), (2, 0)], v
    # the killed slot's EXIT arrives 2 s later: S2 convicts it, the fleet is not restarted
    s[0].exit_code, s[0].done_marker, s[0].log_tail = -9, False, ""
    for st in s:
        st.last_heartbeat = T0 + 2
    v = j.judge(s, T0 + 2)
    assert v.kind == "convict" and v.slot == (0, 0), v
    # no late EXIT: past the grace, every worker alive -> the data plane is broken
    s = slots()
    s[1].exit_code, s[1].done_marker = 1, True
    s[1].log_tail = "RuntimeError: reader: peer connection lost\nWT_DONE 1"
    for t in (T0, T0 + 10):
        for st in s:
            st.last_heartbeat = t
        assert j.judge(s, t).kind == "suspect"
    for st in s:
        st.last_heartbeat = T0 + 11
    v = j.judge(s, T0 + 11)
    assert v.kind == "stalled" and "lost its peer" in v.why, v


def test_alive_but_not_progressing_is_a_stall_not_a_conviction():
    # covers: CONTROL-I7
    s = slots()
    for st in s:
        st.progress = [(i, T0 + i) for i in range(5)]  # 1 s per step
        st.last_heartbeat = T0 + 100
    v = judge().judge(s, T0 + 100)  # 96 s > max(30, 6 x 1)
    assert v.kind == "stalled"


def test_checkpoint_store_common_step_and_prune(tmp_path):
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: common_step is not the newest step every slot holds a committed wave of,
    # index reads an uncommitted or staged directory as a wave, or prune drops a wave at or
    # above the common step (or keeps one below it)
    st = CheckpointStore(str(tmp_path / "store"))
    sl = [(0, 0), (1, 0)]
    for slot, steps in (((0, 0), [1, 3, 5]), ((1, 0), [1, 3])):
        for s in steps:
            _st_wave(st, slot, s)
    os.makedirs(os.path.join(st.root, "r1p0", ".tmp-step-00000005-x"))  # not committed
    body = str(tmp_path / "staged.part")
    _c1_saver(b"staged only")(body)
    st.stage_part((1, 0), 1, 7, "state.pt", "0" * 64, body)  # staged, never committed
    assert st.common_step(sl) == 3
    st.prune(sl)
    idx = st.index()
    assert idx[(0, 0)] == [3, 5] and idx[(1, 0)] == [3]  # older than common dropped
    assert st.common_step([(0, 0), (2, 0)]) is None
    assert sorted(os.listdir(os.path.join(st.root, "r0p0"))) == ["step-00000003", "step-00000005"]
    for slot, s in (((0, 0), 3), ((0, 0), 5), ((1, 0), 3)):
        assert _c1_bytes(os.path.join(st.wave_path(slot, s), "state.pt")) == _c1_wave(s, slot)


def test_corrupt_shard_is_refused(tmp_path):
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: the store commits a wave whose part's bytes do not match the sha256 the part
    # and the manifest claim, or keeps the refused part staged
    st = CheckpointStore(str(tmp_path / "s"))
    p = str(tmp_path / "a")
    _c1_saver(b"payload")(p)
    st.stage_part((0, 0), 1, 1, "state.pt", "0" * 64, p)
    m = {"v": 1, "step": 1, "files": [{"name": "state.pt", "sha256": "0" * 64, "bytes": 7}]}
    with pytest.raises(ValueError):
        st.commit_staged((0, 0), 1, 1, m)
    assert not os.path.exists(p) and st.index() == {}
    assert os.listdir(os.path.join(st.root, ".incoming", "r0p0")) == []


def test_one_slots_commit_never_holds_another_slots_reader(tmp_path, monkeypatch):
    # covers: CONTROL-I20
    # WOULD-FAIL-IF: commit_staged hashes a wave's files (or a held wave's) while holding the
    # store's lock, so the slots committing one step serialize and each connection reader, with
    # the heartbeats and progress behind it, waits for every other slot's bytes (~10 s per
    # 10.5 GB slot at 1 GB/s, HANDOUT-M2 §3, against a 15 s heartbeat timeout)
    st = CheckpointStore(str(tmp_path / "store"))
    hashing, release = threading.Event(), threading.Event()
    real = checkpoint.sha256_file

    def gated(path, *a, **kw):
        # Holds slot r0p0's commit inside its hash until the test releases it.
        if f"{os.sep}r0p0{os.sep}.tmp-step-" in path:
            hashing.set()
            assert release.wait(30)
        return real(path, *a, **kw)

    monkeypatch.setattr(checkpoint, "sha256_file", gated)
    errs: list = []
    slow = _c1_thread(errs, _st_wave, st, (0, 0), 1)
    slow.start()
    try:
        assert hashing.wait(10)
        other = _c1_thread(errs, _st_wave, st, (1, 0), 1)
        other.start()
        other.join(10)
        assert not other.is_alive(), "slot r1p0's commit waited for r0p0's hash"
        assert slow.is_alive()  # r0p0 still hashing: the two commits really overlapped
    finally:
        release.set()
        slow.join(30)
    assert not errs and not slow.is_alive()
    assert st.index() == {(0, 0): [1], (1, 0): [1]}
    for slot in ((0, 0), (1, 0)):
        assert _c1_bytes(os.path.join(st.wave_path(slot, 1), "state.pt")) == _c1_wave(1, slot)


def test_resume_spec_pins_everything_but_addresses_and_slot_keys():
    # covers: CONTROL-I8
    # covers: CONTROL-I22
    # WOULD-FAIL-IF: patch_spec_for_resume changes a field other than endpoints, devices, slot
    # keys and resume_step, keeps the old slot_peers or trusted_peers when given new ones, sets
    # trusted_peers to anything but the sorted set of slot_peers, or accepts a malformed id
    spec = {
        "partition": [[0, 4], [4, 8]],
        "mb_size": 4,
        "init_seed": 1,
        "resume": False,
        "endpoints": [[["a", 1]], [["b", 2]]],
    }
    out = json.loads(patch_spec_for_resume(json.dumps(spec), [[["a", 1]], [["c", 9]]]))
    assert out["resume"] is True and out["endpoints"][1] == [["c", 9]]
    assert out["partition"] == spec["partition"] and out["mb_size"] == 4
    assert "resume_step" not in out
    out = json.loads(patch_spec_for_resume(json.dumps(spec), resume_step=6))
    assert out["resume_step"] == 6 and out["resume"] is True
    assert {k: v for k, v in out.items() if k not in ("resume", "resume_step")} == {
        k: v for k, v in spec.items() if k != "resume"
    }
    for bad in (-1, True, "6", 6.0):
        with pytest.raises(ValueError):
            patch_spec_for_resume(json.dumps(spec), resume_step=bad)

    a, b, c = "a" * 32, "0123456789abcdef" * 2, "f" * 32
    keyed = dict(spec, slot_peers=[[a], [b]], trusted_peers=sorted([a, b]))
    out = json.loads(
        patch_spec_for_resume(
            json.dumps(keyed), [[["a", 1]], [["c", 9]]], slot_peers=[[c], [b]], resume_step=4
        )
    )
    assert out["slot_peers"] == [[c], [b]] and out["trusted_peers"] == [b, c]
    assert out["resume_step"] == 4 and out["endpoints"][1] == [["c", 9]]
    assert {
        k: v
        for k, v in out.items()
        if k not in ("resume", "resume_step", "endpoints", "slot_peers", "trusted_peers")
    } == {
        k: v
        for k, v in keyed.items()
        if k not in ("resume", "endpoints", "slot_peers", "trusted_peers")
    }
    # a wide stage whose two slots share one node's key: the set holds it once
    out = json.loads(patch_spec_for_resume(json.dumps(spec), slot_peers=[[c, a], [c]]))
    assert out["slot_peers"] == [[c, a], [c]] and out["trusted_peers"] == [a, c]
    # without slot_peers the stored pins stay as they are
    out = json.loads(patch_spec_for_resume(json.dumps(keyed), resume_step=2))
    assert out["slot_peers"] == [[a], [b]] and out["trusted_peers"] == sorted([a, b])
    for bad in ([[a], [b.upper()]], [[a], ["b" * 31]], [[a], []], [a, b], "x", [[a], [7]]):
        with pytest.raises(ValueError, match="slot_peers"):
            patch_spec_for_resume(json.dumps(spec), slot_peers=bad)


def test_ownership_patch_pins_everything_else():
    # covers: CONTROL-I21
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: patch_spec_for_resume changes peer_weights, mb_owner or schedule_order
    # without the explicit ownership argument, changes any other field with it, keeps the stored
    # ownership when given a new one, or accepts an ownership with a missing, extra or non-list
    # field
    spec = JobSpec(stages=3, stage_widths=[1, 2, 1], microbatches=8, steps=6).finalize()
    stored = spec.to_json()
    new = JobSpec.from_json(stored)
    new.peer_weights = [[1.0], [1.0, 0.25], [1.0]]
    new.mb_owner = None
    new.schedule_order = None
    new.finalize()
    own = {
        "peer_weights": new.peer_weights,
        "mb_owner": new.mb_owner,
        "schedule_order": new.schedule_order,
    }
    assert new.mb_owner != spec.mb_owner and new.schedule_order != spec.schedule_order
    base = json.loads(stored)
    out = json.loads(patch_spec_for_resume(stored, resume_step=3, ownership=own))
    assert out["resume"] is True and out["resume_step"] == 3
    for k, v in own.items():
        assert out[k] == json.loads(json.dumps(v)), k
    keep = ("resume", "resume_step", *checkpoint.OWNERSHIP_FIELDS)
    assert {k: v for k, v in out.items() if k not in keep} == {
        k: v for k, v in base.items() if k not in keep
    }
    JobSpec.from_json(json.dumps(out)).finalize()  # a valid spec under the new ownership
    out = json.loads(patch_spec_for_resume(stored, resume_step=3))
    for k in checkpoint.OWNERSHIP_FIELDS:
        assert out[k] == base[k], k
    bad = [
        {k: own[k] for k in ("peer_weights", "mb_owner")},
        dict(own, partition=[[0, 1]]),
        dict(own, mb_owner="0,1"),
        dict(own, schedule_order=[]),
        [own["peer_weights"]],
    ]
    for b in bad:
        with pytest.raises(ValueError, match="ownership"):
            patch_spec_for_resume(stored, resume_step=3, ownership=b)


def test_replan_decision_rule():
    # covers: CONTROL-I21
    # WOULD-FAIL-IF: a rebalance is applied below the 10% predicted gain, when the gain over the
    # remaining steps does not exceed twice the restart cost (or equals it), or after max_replans
    # rebalances; or a plan that pays is refused
    from wantrain.coordinator.coordinator import REPLAN_MIN_GAIN, replan_rule
    from wantrain.scheduler.zero_bubble import replan

    assert REPLAN_MIN_GAIN == 0.10
    assert (CoordinatorConfig().max_replans, CoordinatorConfig().replan_restart_cost_s) == (2, 30)
    spec = JobSpec(stages=3, stage_widths=[1, 2, 1], microbatches=8, steps=24).finalize()

    def window(slow):
        out = {}
        for r, w in enumerate(spec.stage_widths):
            for p in range(w):
                k = slow if (r, p) == (1, 1) else 1.0
                m = len(spec.owned(r, p))
                ops = {op: {"n": m, "med_ms": t * k} for op, t in (("F", 3), ("B", 5), ("W", 2))}
                ops["OPT"] = {"n": 1, "med_ms": 1.0}
                out[(r, p)] = [{"ops": ops, "hops": {}} for _ in range(4)]
        return out

    small = replan(spec, window(1.05), REPLAN_MIN_GAIN)  # gain below 10%
    big = replan(spec, window(4.0), REPLAN_MIN_GAIN)
    assert small["decision"] is False and big["decision"] is True, (small, big)
    assert big["pred_new_s"] <= 0.9 * big["pred_cur_s"]
    gain = big["pred_cur_s"] - big["pred_new_s"]
    ok, why = replan_rule(small, 10_000, 0.0, 0, 2)
    assert not ok and why == small["why"]
    ok, why = replan_rule(big, 100, gain * 100 / 2 * 0.99, 0, 2)
    assert ok and "pays" in why, why
    # too few remaining steps: gain x steps == 2R exactly, and below it
    ok, why = replan_rule(big, 4, gain * 4 / 2, 0, 2)
    assert not ok and "does not pay" in why, why
    ok, _ = replan_rule(big, 1, gain, 0, 2)
    assert not ok
    ok, _ = replan_rule(big, 0, 0.0, 0, 2)
    assert not ok
    # budget spent
    ok, why = replan_rule(big, 10_000, 0.0, 2, 2)
    assert not ok and "budget (2) is spent" in why
    ok, _ = replan_rule(big, 10_000, 0.0, 1, 2)
    assert ok


def n(i, gpu="RTX 3090", device="cuda", gb=24, host=None, region="eu", cap=(8, 6), tf=30.0):
    return NodeInfo(
        node_id=f"n{i}",
        name=f"w{i}",
        host=host or f"10.0.0.{i}",
        data_port=1,
        device=device,
        gpu_name=gpu,
        compute_cap=cap,
        vram_total_bytes=gb << 30,
        vram_free_bytes=gb << 30,
        tflops_fp32=tf,
        region=region,
        device_class=f"cuda:{gpu}" if device == "cuda" else f"{device}:x86_64:AVX2",
    )


def test_replacement_prefers_identical_model_then_region():
    # covers: CONTROL-I9
    dead = n(0)
    survivors = [n(1)]
    spares = [
        n(2, gpu="RTX 3080", region="eu", gb=24),  # same series, same region
        n(3, gpu="RTX 3090", region="us"),  # same model, other region
        n(4, gpu="RTX 3090", region="eu", gb=12),  # too small
        n(5, gpu="RTX 3090", region="eu", host="10.0.0.1"),
    ]  # survivor's address
    best, note = find_replacement(spares, dead, survivors)
    assert best.node_id == "n3" and "bit-comparable" in note
    best, _ = find_replacement([s for s in spares if s.node_id != "n3"], dead, survivors)
    assert best.node_id == "n2"


def test_replacement_respects_the_recipe_and_the_dead_list():
    # covers: CONTROL-I9
    dead = n(0)
    pre_ampere = n(2, gpu="RTX 2080", cap=(7, 5))
    assert find_replacement([pre_ampere], dead, [], compute_dtype="bf16", match="any") is None
    assert find_replacement([n(3)], dead, [], dead_keys={"w3@10.0.0.3"}) is None
    cpu = n(4, gpu="cpu", device="cpu", cap=None)
    assert find_replacement([cpu], dead, [], match="series") is None
    assert find_replacement([cpu], dead, [], match="any")[0].node_id == "n4"


def test_pieces_heal_find_replacement_refuses_a_declared_class():
    # covers: CONTROL-I9
    # WOULD-FAIL-IF: probed_class accepts a spare (or a dead node) whose gpu_name or device
    # was declared by the operator, so a pieces slot is healed by a class nobody probed
    dead = n(0)
    declared_gpu = n(2, region="eu")
    declared_gpu.declared = ["gpu_name"]
    declared_dev = n(3, region="eu")
    declared_dev.declared = ["device", "region"]
    probed = n(4, region="us", tf=10.0)
    spares = [declared_gpu, declared_dev, probed]
    assert find_replacement(spares, dead, [], match="model")[0].node_id == "n2"
    best, note = find_replacement(spares, dead, [], match="model", probed_class=True)
    assert best.node_id == "n4" and "bit-comparable" in note
    assert find_replacement(spares[:2], dead, [], match="model", probed_class=True) is None
    declared_region = n(5)
    declared_region.declared = ["region"]
    assert find_replacement([declared_region], dead, [], "fp32", "model", probed_class=True)
    dead.declared = ["gpu_name"]
    assert find_replacement([probed], dead, [], match="model", probed_class=True) is None


def _pieces_heal_coordinator(tmp_path, shard, widths, spares, match="any"):
    """A real Coordinator whose run has the slots of `widths`, every slot on an
    RTX 3090. FAKE at the worker-network boundary, named as such: `_request`
    records each request and answers {"ok": True}; the relaunch is recorded
    instead of run (`_resume_spec`, `_restore_and_relaunch`), since it would
    read the spec through JobSpec fields another task adds."""
    coord = Coordinator(
        config=CoordinatorConfig(state_dir=str(tmp_path / "coord"), replacement_match=match)
    )
    sent, relaunched = [], []

    def fake_request(w, msg, body=b"", timeout=None, file=None, reply_kind=None):
        sent.append((w.node_id, msg["kind"]))
        return {"ok": True}

    coord._request = fake_request
    coord._resume_spec = lambda run, step: "resume-spec"
    coord._restore_and_relaunch = lambda run, step, spec_json: relaunched.append(step)
    os.makedirs(tmp_path / "run")
    run = _Run("run-ph", str(tmp_path / "run"))
    d = json.loads(JobSpec(steps=4, stages=len(widths), stage_widths=list(widths)).to_json())
    d["shard"] = shard
    run.spec_json = json.dumps(d)
    i = 0
    for r, wd in enumerate(widths):
        for p in range(wd):
            info = n(i)
            coord.workers[info.node_id] = Worker(info, _C1FakeChannel(), f"s{i}", None)
            coord.workers[info.node_id].run_id = run.run_id  # a slot's worker, as _setup marks it
            run.slots[(r, p)] = info.node_id
            i += 1
    for info in spares:
        coord.workers[info.node_id] = Worker(info, _C1FakeChannel(), info.node_id, None)
    coord.run = run
    return coord, run, sent, relaunched


def _pieces_heal_spares():
    series = n(20, gpu="RTX 3080", region="eu", tf=50.0)  # same series, faster, dead's region
    declared = n(21, region="eu", tf=40.0)  # same model by the operator's word only
    declared.declared = ["gpu_name"]
    identical = n(22, region="us", tf=10.0)  # the identical probed model, farther, slower
    return series, declared, identical


def test_pieces_heal_takes_only_the_identical_probed_model(tmp_path):
    # covers: CONTROL-I9
    # WOULD-FAIL-IF: _heal passes cfg.replacement_match (here 'any') for a slot of a pieces
    # stage, or does not ask for a probed class, so the faster RTX 3080 or the spare that only
    # declared an RTX 3090 takes the slot instead of the identical probed RTX 3090
    coord, run, _sent, relaunched = _pieces_heal_coordinator(
        tmp_path, "pieces", [1, 2], list(_pieces_heal_spares())
    )
    coord._heal(run, (1, 1), "killed")
    assert run.slots[(1, 1)] == "n22"
    assert run.heals[0]["to"] == "w22" and relaunched == [None]


def test_pieces_heal_without_an_identical_spare_fails_naming_the_class(tmp_path):
    # covers: CONTROL-I9
    # WOULD-FAIL-IF: a pieces slot with no identical probed spare is healed by another class,
    # or the failure relaunches, or its reason does not name the required model
    series, declared, _ = _pieces_heal_spares()
    coord, run, sent, relaunched = _pieces_heal_coordinator(
        tmp_path, "pieces", [2], [series, declared]
    )
    with pytest.raises(RuntimeError) as e:
        coord._heal(run, (0, 0), "killed")
    assert "identical probed device model 'RTX 3090' on 'cuda'" in str(e.value)
    assert run.slots[(0, 0)] == "n0" and run.heals == [] and relaunched == []
    assert {kind for _, kind in sent} <= {"ABORT"}


def test_pieces_heal_keeps_the_configured_match_for_zero1_and_width_1(tmp_path):
    # covers: CONTROL-I9, CONTROL-I25
    # WOULD-FAIL-IF: the identical-probed-model rule leaks to a zero1 slot or a width-1 slot of
    # a pieces job, which keep cfg.replacement_match (the spare that only declared an RTX 3090,
    # of the run's device class, heals them); or a spare of another device class (the RTX
    # 3080, which 'series' matched before CONTROL-I25) heals any slot
    for shard, widths, slot in (("zero1", [2], (0, 1)), ("pieces", [1, 2], (0, 0))):
        series, declared, _ = _pieces_heal_spares()
        sub = tmp_path / f"{shard}{len(widths)}"
        coord, run, _sent, relaunched = _pieces_heal_coordinator(
            sub, shard, widths, [declared], match="series"
        )
        coord._heal(run, slot, "killed")
        assert run.slots[slot] == "n21" and relaunched == [None]
        coord, run, _sent, relaunched = _pieces_heal_coordinator(
            sub / "series", shard, widths, [series], match="series"
        )
        with pytest.raises(RuntimeError) as e:
            coord._heal(run, slot, "killed")
        assert (
            "MIXED-FLEET: no spare of the run's device class cuda:RTX 3090 "
            "(pool: cuda:RTX 3080 x1)" in str(e.value)
        )
        assert run.slots[slot] != "n20" and run.heals == [] and relaunched == []


# ---------------------------------------------------------------- CONTROL-I17


def _c1_clock(monkeypatch, t):
    """Injects the coordinator's clock: `time.time()` inside coordinator.py
    reads clock[0]; nothing else in the process is affected."""
    clock = [t]
    monkeypatch.setattr(coord_mod, "time", SimpleNamespace(time=lambda: clock[0]))
    return clock


def _c1_coordinator(tmp_path, steps, **cfg):
    """A real Coordinator and a run of two slots, (0, 0) and (1, 0). FAKE at the
    worker-network boundary, named as such: `_request` records the request and
    answers {"ok": True} (no agent, no socket) and `_sync_all_connections` does
    nothing; every coordinator method the tests call is the real one."""
    coord = Coordinator(config=CoordinatorConfig(state_dir=str(tmp_path / "coord"), **cfg))
    sent = []

    def fake_request(w, msg, body=b"", timeout=None, file=None, reply_kind=None):
        sent.append((w.node_id, msg["kind"]))
        return {"ok": True}

    coord._request = fake_request
    coord._sync_all_connections = lambda run: None
    os.makedirs(tmp_path / "run")
    run = _Run("run-c1", str(tmp_path / "run"))
    spec = JobSpec(steps=steps, stages=2)
    run.spec, run.spec_json = spec, spec.to_json()
    for r in range(2):
        info = n(r, gpu="cpu", device="cpu", cap=None)
        # The RESTORE_PARTs go to a FAKE channel (same boundary as fake_request).
        coord.workers[info.node_id] = Worker(info, _C1FakeChannel(), f"session{r}", None)
        run.slots[(r, 0)] = info.node_id
        _st_wave(run.store, (r, 0), 1)  # the wave a resume restores
    coord.run = run
    return coord, run, sent


def _c1_progress(coord, run, slot, step, src):
    """One PROGRESS message from the slot's worker, through the real _dispatch."""
    msg = {
        "kind": "PROGRESS",
        "run_id": run.run_id,
        "attempt": run.attempt,
        "step": step,
        "loss": None,
        "t": 0.0,
        "src": src,
    }
    coord._dispatch(coord.workers[run.slots[slot]], msg, None)


def _c1_at(j, states, t):
    """Judge at time t with every heartbeat fresh, so strikes stay out of it."""
    for st in states:
        st.last_heartbeat = t
    return j.judge(states, t)


def _c1_fleet(last_steps):
    """One slot per entry; slot i reported steps 0..last_steps[i], step s at T0 + s."""
    s = slots(n=len(last_steps))
    for st, last in zip(s, last_steps):
        st.progress = [(i, T0 + i) for i in range(last + 1)]
    return s


def test_restart_cost_is_timed_from_every_relaunch(tmp_path, monkeypatch):
    # covers: CONTROL-I21
    # WOULD-FAIL-IF: a relaunch from step 0 (no common wave) leaves restore_t0 at an earlier
    # restore whose first step never came, so R is measured from that older restore and the
    # rebalance rule (G > 2R) compares the gain with an overstated restart cost
    clock = _c1_clock(monkeypatch, T0)
    coord, run, _sent = _c1_coordinator(tmp_path, 8)
    coord._restore_and_relaunch(run, 1, run.spec_json)
    assert run.restore_t0 == T0
    clock[0] = T0 + 100  # its first step never came; the next relaunch has no wave
    coord._restore_and_relaunch(run, None, run.spec_json)
    assert run.restore_t0 == T0 + 100


def test_replayed_metric_progress_never_feeds_the_stall_clock(tmp_path, monkeypatch):
    # covers: CONTROL-I17
    # WOULD-FAIL-IF: the coordinator stamps WT_METRIC (src "metric") progress, or a step outside
    # [start_step, steps - 1], a non-integer step or a src-less PROGRESS, into the stall clock, so
    # a resume's replayed loss stream moves the last-progress time and collapses the median step;
    # or add_progress records a repeat or a lower step (ADR-0003: a src "step" stream 2, 3, 2, 3, 2
    # would keep renewing the last-progress time, so S4 never fires)
    st = SlotState(slot=(0, 0), node_id="x", start_step=2)
    assert st.add_progress(1, T0, 3) is False  # before the attempt's first step
    assert st.add_progress(2, T0 + 1, 3) is True
    assert st.add_progress(2, T0 + 2, 3) is False  # a repeat of the last step
    assert st.add_progress(4, T0 + 3, 3) is False  # past the final step
    assert st.add_progress(3, T0 + 4, 3) is True
    assert st.progress == [(2, T0 + 1), (3, T0 + 4)]
    assert st.add_progress(2, T0 + 5, 3) is False  # lower than the last step
    assert st.progress == [(2, T0 + 1), (3, T0 + 4)]

    # ADR-0003: a src "step" stream 2, 3, 2, 3, 2 through the real _dispatch records 2 and 3
    # once; a quiet past the limit measured from step 3 reaches S4 although messages kept coming
    clock = _c1_clock(monkeypatch, T0)
    coord, run, _sent = _c1_coordinator(
        tmp_path / "loop", 4, stall_floor_s=5.0, first_step_timeout_s=60.0
    )
    coord._restore_and_relaunch(run, 1, run.spec_json)  # start_step 2, final step 3
    for t, s in ((1, 2), (2, 3), (40, 2), (50, 3), (60, 2)):
        clock[0] = T0 + t
        _c1_progress(coord, run, (0, 0), s, "step")
    assert run.states[(0, 0)].progress == [(2, T0 + 1), (3, T0 + 2)]
    j = coord._judge(run)
    states = list(run.states.values())
    assert _c1_at(j, states, T0 + 2 + 59).kind == "ok"  # limit: first_step_timeout_s (no median)
    v = _c1_at(j, states, T0 + 2 + 61)
    assert v.kind == "stalled" and "(limit 60s)" in v.why, v

    clock[0] = T0
    coord, run, _sent = _c1_coordinator(tmp_path, 8, stall_floor_s=5.0, first_step_timeout_s=60.0)
    coord._restore_and_relaunch(run, None, run.spec_json)
    for s in range(5):  # 2 s per step: the S4 limit is max(5, 6 x 2) = 12 s
        clock[0] = T0 + 2 * (s + 1)
        for slot in run.slots:
            _c1_progress(coord, run, slot, s, "step")
    before = {slot: list(st.progress) for slot, st in run.states.items()}
    assert before[(0, 0)] == [(s, T0 + 2 * (s + 1)) for s in range(5)]
    assert before[(1, 0)] == before[(0, 0)]

    clock[0] = T0 + 11  # a resume's burst, all at once
    for slot in run.slots:
        # Steps 5 and 6 pass the range and strictly-greater rules, so only the src filter
        # rejects them: a coordinator that also stamps src "metric" records them.
        _c1_progress(coord, run, slot, 5, "metric")
        _c1_progress(coord, run, slot, 6, "metric")
        for s in range(5):
            _c1_progress(coord, run, slot, s, "metric")
        _c1_progress(coord, run, slot, 8, "step")  # = steps: past the final step 7
        _c1_progress(coord, run, slot, -1, "step")  # before the attempt's start step
        _c1_progress(coord, run, slot, "5", "step")  # not an integer
        _c1_progress(coord, run, slot, 5, None)  # no source: not a WT_STEP
    after = {slot: list(st.progress) for slot, st in run.states.items()}
    assert after == before
    assert any("dropped progress with step '5'" in e[2] for e in coord.events)

    j = coord._judge(run)
    assert j.final_step == 7
    states = list(run.states.values())
    assert _c1_at(j, states, T0 + 10 + 11).kind == "ok"
    v = _c1_at(j, states, T0 + 10 + 13)
    assert v.kind == "stalled" and "(limit 12s)" in v.why, v


def test_resumed_attempt_is_judged_by_the_first_step_timeout(tmp_path, monkeypatch):
    # covers: CONTROL-I17
    # WOULD-FAIL-IF: _launch sets start_step to steps or final_step + 1 (shutdown from the start,
    # every real WT_STEP dropped: S4 off for a resumed attempt) or leaves it at 0 on a mid-run
    # relaunch, or a resumed attempt without real progress is judged by the stall floor
    j = judge(final_step=7)  # floor 30 s, first step 60 s
    s = slots(start_step=5)
    assert _c1_at(j, s, T0 + 31).kind == "ok"  # past the floor, inside the first-step timeout
    assert _c1_at(j, s, T0 + 61).kind == "stalled"

    clock = _c1_clock(monkeypatch, T0)
    coord, run, sent = _c1_coordinator(tmp_path, 4, stall_floor_s=30.0, first_step_timeout_s=60.0)
    coord._restore_and_relaunch(run, 1, run.spec_json)  # resume after wave 1 of 4 steps
    assert sorted(k for _n, k in sent) == ["ASSIGN", "ASSIGN", "RESTORE", "RESTORE"]
    assert run.attempt == 1
    assert {slot: st.start_step for slot, st in run.states.items()} == {(0, 0): 2, (1, 0): 2}

    jr = coord._judge(run)
    assert jr.final_step == 3
    states = list(run.states.values())
    assert _c1_at(jr, states, T0 + 31).kind == "ok"
    v = _c1_at(jr, states, T0 + 61)
    assert v.kind == "stalled", v  # not shutdown: the attempt owes steps 2 and 3

    clock[0] = T0 + 62
    _c1_progress(coord, run, (0, 0), 1, "step")  # the wave itself: not this attempt's step
    assert run.states[(0, 0)].progress == []
    _c1_progress(coord, run, (0, 0), 2, "step")
    assert run.states[(0, 0)].progress == [(2, T0 + 62)]

    sent.clear()
    coord._restore_and_relaunch(run, None, run.spec_json)  # no wave: from step 0
    assert sorted(k for _n, k in sent) == ["ASSIGN", "ASSIGN"]
    assert run.attempt == 2
    assert {slot: st.start_step for slot, st in run.states.items()} == {(0, 0): 0, (1, 0): 0}


def test_quiet_after_the_final_step_is_shutdown_not_a_stall():
    # covers: CONTROL-I17
    # WOULD-FAIL-IF: quiet after every slot finished the final step is judged a stall, the
    # shutdown is unbounded, the judge only raises the floor, shutdown starts when any (not
    # every) slot finished, a peer loss after the final step with no stage left running is "ok"
    # forever, or final_step=None changes today's verdicts
    assert LivenessJudge().shutdown_limit_s == 600.0
    assert CoordinatorConfig().shutdown_limit_s == 600.0
    j = judge(final_step=7, shutdown_limit_s=300)

    # 1. before the final step, quiet past the limit (max(30, 6 x 1 s)) is a stall
    v = _c1_at(j, _c1_fleet([5, 5, 5]), T0 + 5 + 31)
    assert v.kind == "stalled" and "(limit 30s)" in v.why, v

    # 2. every slot reported the final step: quiet is shutdown, bounded
    assert _c1_at(j, _c1_fleet([7, 7, 7]), T0 + 7 + 31).kind == "ok"
    assert _c1_at(j, _c1_fleet([7, 7, 7]), T0 + 7 + 299).kind == "ok"
    v = _c1_at(j, _c1_fleet([7, 7, 7]), T0 + 7 + 301)
    assert v.kind == "failed" and "shutdown exceeded 300s after the final step" in v.why, v

    # 3. one slot still owes the final step
    v = _c1_at(j, _c1_fleet([7, 6, 7]), T0 + 7 + 31)
    assert v.kind == "stalled", v

    # 4. a slot that exited 0 owes nothing; one that still owes a step keeps S4 on
    s = _c1_fleet([5, 7, 7])
    s[0].exit_code, s[0].done_marker = 0, True
    assert _c1_at(j, s, T0 + 7 + 31).kind == "ok"
    s = _c1_fleet([7, 6, 6])
    s[0].exit_code, s[0].done_marker = 0, True
    v = _c1_at(j, s, T0 + 7 + 31)
    assert v.kind == "stalled", v

    # 5. an attempt that started past the final step (relaunched from the final wave)
    s = slots(start_step=8)
    assert _c1_at(j, s, T0 + 61).kind == "ok"  # past first_step_timeout_s (60)
    assert _c1_at(j, s, T0 + 301).kind == "failed"

    # 6. strikes still apply in shutdown
    s = _c1_fleet([7, 7, 7])
    for st in s:
        st.last_heartbeat = T0 + 38
    s[1].last_heartbeat = T0
    v = j.judge(s, T0 + 38)
    assert v.kind == "suspect" and v.slot == (1, 0), v

    # 6b. a peer lost after the final step: a silent slot is still S3's suspect, and once every
    # stage exited (none left silent) the shutdown bound still ends the attempt, never "ok" forever
    s = _c1_fleet([7, 7, 7])
    s[0].exit_code, s[0].done_marker = 1, True
    s[0].log_tail = "RuntimeError: reader: peer connection lost\nWT_DONE 1"
    # inside exit_report_grace_s (10): the silent slots may yet report their own exits
    v = _c1_at(j, s, T0 + 7 + 31)
    assert v.kind == "suspect" and v.slot is None and "lost its peer" in v.why, v
    v = _c1_at(j, s, T0 + 7 + 31 + 11)
    assert v.kind == "stalled" and "lost its peer" in v.why, v
    for st in s[1:]:
        st.exit_code, st.done_marker = 0, True
    assert _c1_at(j, s, T0 + 7 + 31).kind == "ok"
    v = _c1_at(j, s, T0 + 7 + 301)
    assert v.kind == "failed" and "shutdown exceeded 300s after the final step" in v.why, v

    # 7. without a final step the judge is today's: the same quiet is a stall
    v = _c1_at(judge(), _c1_fleet([7, 7, 7]), T0 + 7 + 31)
    assert v.kind == "stalled", v


# ---------------------------------------------------------------- CONTROL-I18


def _c2_coordinator(tmp_path):
    """A real Coordinator and a run at attempt 1 of two slots whose rank and peer
    both differ, (0, 0) on worker n0 and (1, 1) on worker n1, 4 steps. No
    socket: messages go through the real _dispatch / _conn_loop."""
    coord = Coordinator(config=CoordinatorConfig(state_dir=str(tmp_path / "coord")))
    os.makedirs(tmp_path / "run")
    run = _Run("run-c2", str(tmp_path / "run"))
    spec = JobSpec(steps=4, stages=2)
    run.spec, run.spec_json = spec, spec.to_json()
    run.attempt = 1
    for slot in ((0, 0), (1, 1)):
        info = n(slot[0], gpu="cpu", device="cpu", cap=None)
        coord.workers[info.node_id] = Worker(info, None, f"session{slot[0]}", None)
        run.slots[slot] = info.node_id
        run.states[slot] = SlotState(slot=slot, node_id=info.node_id, launched=True)
    coord.run = run
    return coord, run


def _c2_scripted_channel(script):
    """FAKE at the control-socket boundary, named as such: recv() hands out the
    scripted (msg, body) pairs in order, then raises ChannelClosed as a peer
    that hung up would. Records how many times it was read."""
    reads = []

    def recv(file_dir=None):
        reads.append(len(reads))
        if len(reads) > len(script):
            raise ChannelClosed("end of script")
        return script[len(reads) - 1]

    return SimpleNamespace(recv=recv, close=lambda: None, last_rx=0.0), reads


def _c2_file(path, data):
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def _c2_read(path):
    with open(path, "rb") as f:
        return f.read()


def _c2_summary(rank, peer, step):
    """A valid per-step summary in RUNTIME-I9's shape (distinct values)."""
    return {
        "v": 1,
        "step": step,
        "rank": rank,
        "peer": peer,
        "step_ms": 120.5,
        "emit_us": 80.25,
        "ops": {
            "F": {"n": 4, "med_ms": 3.5},
            "B": {"n": 4, "med_ms": 5.75},
            "W": {"n": 4, "med_ms": 2.25},
            "OPT": {"n": 1, "med_ms": 1.125},
        },
        "stall_ms": 0.5,
        "hops": {"next0": {"frames": 8, "bytes": 65536, "busy_ms": 12.5}},
    }


def _c2_telem(coord, run, slot, summary, attempt=None):
    """One TELEM_STEP from the connection bound to `slot`, through the real _dispatch."""
    msg = {
        "kind": "TELEM_STEP",
        "run_id": run.run_id,
        "attempt": run.attempt if attempt is None else attempt,
        "summary": summary,
    }
    coord._dispatch(coord.workers[run.slots[slot]], msg, b"")


def _c2_window(run, slot):
    return [s["step"] for s in run.telem_window.get(slot, ())]


def test_malformed_messages_never_stop_the_reader(tmp_path):
    # covers: CONTROL-I18
    # WOULD-FAIL-IF: an exception from one message ends the connection's reader (later heartbeats
    # go unread and a live worker is convicted), a dropped message's file body is left behind or
    # moved outside run_dir, or the TELEMETRY branch builds its destination from the message's
    # rank, peer or attempt instead of the connection's slot (slot (0, 0)'s file is overwritten)
    # or moves a stale attempt's body
    coord, run = _c2_coordinator(tmp_path)
    w = coord.workers[run.slots[(1, 1)]]  # the connection bound to slot (1, 1)
    inbox = tmp_path / "inbox"
    os.makedirs(inbox)
    own = _c2_file(tmp_path / "run" / "telemetry_r0p0_a1.jsonl", b'{"ev":"own-of-r0p0"}\n')
    rid, a = run.run_id, run.attempt
    tel = {"kind": "TELEMETRY", "run_id": rid, "file": True}
    ckpt_body = _c2_file(inbox / "ckpt.part", b"not a shard")
    tel_body = _c2_file(inbox / "tel.part", b'{"ev":"sent-by-r1p1"}\n')
    stale_body = _c2_file(inbox / "stale.part", b'{"ev":"stale"}\n')
    script = [
        ({"run_id": rid, "attempt": a}, b""),  # no kind
        ({"kind": "TELEMETRY", "run_id": rid, "attempt": a, "rank": 1, "peer": 1}, b""),
        (
            {
                "kind": "CKPT",
                "run_id": rid,
                "attempt": a,
                "rank": "r1",  # not an integer
                "peer": 1,
                "step": 0,
                "sha256": "0" * 64,
                "file": True,
            },
            ckpt_body,
        ),
        ({"kind": "HEARTBEAT", "stage_alive": True}, b""),
        # Names the other slot: must land at the connection's own slot (1, 1).
        (dict(tel, attempt=a, rank=0, peer=0), tel_body),
        # A stale attempt: dropped and its body unlinked, never moved.
        (dict(tel, attempt=0, rank=1, peer=1), stale_body),
        ({"kind": "HEARTBEAT", "stage_alive": True}, b""),
    ]
    ch, reads = _c2_scripted_channel(script)
    w.ch, w.last_heartbeat = ch, 0.0
    coord._conn_loop(w, ch)

    assert _c2_read(own) == b'{"ev":"own-of-r0p0"}\n'  # slot (0, 0)'s file is untouched
    mine = tmp_path / "run" / "telemetry_r1p1_a1.jsonl"
    assert _c2_read(mine) == b'{"ev":"sent-by-r1p1"}\n'
    assert not (tmp_path / "run" / "telemetry_r1p1_a0.jsonl").exists()
    files = sorted(f for f in os.listdir(tmp_path / "run") if f.startswith("telemetry_"))
    assert files == ["telemetry_r0p0_a1.jsonl", "telemetry_r1p1_a1.jsonl"]
    assert os.listdir(inbox) == []  # every body moved to run_dir or unlinked
    assert len(reads) == len(script) + 1  # read to the end: only the channel error stopped it
    assert w.last_heartbeat > 0.0  # the heartbeats after the bad messages were read
    malformed = [e[2] for e in coord.events if e[1] == "warn" and e[2].startswith("malformed ")]
    kinds = [m.split(" from ")[0] for m in malformed]
    want = ["malformed None", "malformed TELEMETRY", "malformed CKPT", "malformed TELEMETRY"]
    assert kinds == want, malformed
    assert all(" from w1: " in m for m in malformed), malformed


def test_telem_step_identity_and_bounds_are_checked_at_the_boundary(tmp_path):
    # covers: CONTROL-I18
    # WOULD-FAIL-IF: the TELEM_STEP branch indexes the window by the summary's rank/peer instead of
    # the connection's slot (the summary lands in (0, 0)'s window), or accepts a summary of a
    # stale attempt, a step outside [start step, steps - 1], a non-finite number, more than 4096
    # bytes or an unknown key, or the telemetry log grows past telem_log_max_bytes
    coord, run = _c2_coordinator(tmp_path)
    _c2_telem(coord, run, (0, 0), _c2_summary(0, 0, 0))
    _c2_telem(coord, run, (1, 1), _c2_summary(1, 1, 2))
    assert (_c2_window(run, (0, 0)), _c2_window(run, (1, 1))) == ([0], [2])

    _c2_telem(coord, run, (1, 1), _c2_summary(0, 0, 1))  # names the other slot
    assert _c2_window(run, (0, 0)) == [0]
    assert run.telem_stats["accepted"] == 2

    out_of_range = _c2_summary(1, 1, 4)  # steps = 4: the last step is 3
    nan = _c2_summary(1, 1, 1)
    nan["ops"]["B"]["med_ms"] = float("nan")
    big = _c2_summary(1, 1, 1)
    big["hops"] = {f"intra{k}": {"frames": 1, "bytes": 1, "busy_ms": 1.0} for k in range(120)}
    unknown = dict(_c2_summary(1, 1, 1), extra=1)
    for bad in (out_of_range, nan, big, unknown):
        _c2_telem(coord, run, (1, 1), bad)
    _c2_telem(coord, run, (1, 1), _c2_summary(1, 1, 1), attempt=0)  # stale attempt
    assert (_c2_window(run, (0, 0)), _c2_window(run, (1, 1))) == ([0], [2])
    assert (run.telem_stats["accepted"], run.telem_stats["dropped"]) == (2, 6)

    log = tmp_path / "run" / "telem_steps.jsonl"
    with open(log) as f:
        recs = [json.loads(line) for line in f]
    ids = [(r["rank"], r["peer"], r["step"], r["attempt"]) for r in recs]
    assert ids == [(0, 0, 0, 1), (1, 1, 2, 1)]
    assert recs[1] == dict(_c2_summary(1, 1, 2), attempt=1)

    coord.cfg.telem_log_max_bytes = os.path.getsize(log)  # full: the next record does not fit
    _c2_telem(coord, run, (1, 1), _c2_summary(1, 1, 3))
    _c2_telem(coord, run, (0, 0), _c2_summary(0, 0, 1))
    assert (_c2_window(run, (0, 0)), _c2_window(run, (1, 1))) == ([0, 1], [2, 3])
    with open(log) as f:
        assert len(f.readlines()) == 2
    full = [e for e in coord.events if "telem_log_max_bytes" in e[2]]
    assert len(full) == 1, full

    tel = run.rationale["telemetry"]
    assert (tel["accepted"], tel["dropped"]) == (4, 6)
    assert 0.0 < tel["handle_us_max"] < 1e6


def test_orphan_bodies_are_unlinked_and_peer_text_is_bounded(tmp_path, monkeypatch):
    # covers: CONTROL-I18
    # WOULD-FAIL-IF: a file body on a HEARTBEAT, a PROGRESS or an unknown kind of the current run
    # is left in the inbox (a worker fills the coordinator's disk without raising), a value the
    # worker controls is logged unclipped (one 16 MiB header pins 16 MiB of events), or the
    # coordinator keeps every event record forever
    coord, run = _c2_coordinator(tmp_path)
    w = coord.workers[run.slots[(1, 1)]]
    inbox = tmp_path / "inbox"
    os.makedirs(inbox)
    rid, a = run.run_id, run.attempt
    big = "A" * 100_000
    script = [
        (
            {"kind": "HEARTBEAT", "stage_alive": True, "file": True},
            _c2_file(inbox / "hb.part", b"x"),
        ),
        (
            {"kind": "PROGRESS", "run_id": rid, "attempt": a, "src": "step", "step": 0, "file": 1},
            _c2_file(inbox / "progress.part", b"xx"),
        ),
        ({"kind": "BOGUS", "run_id": rid, "file": True}, _c2_file(inbox / "bogus.part", b"xxx")),
        ({"kind": "PROGRESS", "run_id": rid, "attempt": a, "src": "step", "step": big}, b""),
        ({"kind": [big], "rid": [1]}, b""),  # an unhashable rid: TypeError in _dispatch
    ]
    ch, reads = _c2_scripted_channel(script)
    w.ch = ch
    coord._conn_loop(w, ch)

    assert os.listdir(inbox) == []
    assert len(reads) == len(script) + 1
    assert run.states[(1, 1)].progress[0][0] == 0
    texts = [e[2] for e in coord.events]
    assert any(t.startswith("slot (1, 1): dropped progress with step 'AAA") for t in texts), texts
    assert any(t.startswith("malformed ['AAA") for t in texts), texts
    assert max(len(t) for t in texts) < 400, max(len(t) for t in texts)

    monkeypatch.setattr(coord_mod, "EVENTS_MAX", 3)
    for i in range(5):
        coord.log("info", f"event {i}")
    assert [e[2] for e in coord.events] == ["event 2", "event 3", "event 4"]


# ---- C1: a wave's label is the step its bytes hold (ship, restore, export) ----


def _c1_wave(step, slot=(1, 0)):
    """Distinct bytes per wave and per slot."""
    head = f"wave-{step}-slot-r{slot[0]}p{slot[1]}|".encode()
    return head * 2048 + bytes([step, slot[0], slot[1]])


def _c1_saver(data):
    def save(tmp):
        with open(tmp, "wb") as f:
            f.write(data)

    return save


def _c1_bytes(path):
    with open(path, "rb") as f:
        return f.read()


class _C1FakeChannel:
    """Fake of the transport boundary (Channel): records every send in order,
    reading the shipped file at send time and returning its sha256 (also put
    in the header) as Channel.send_file does. With a gate, the first send_file
    blocks on it after signalling `entered`; `on_send`, when set, runs first in
    every send_file. `msgs` holds the plain messages other than CKPT."""

    def __init__(self, gate=None):
        self.gate = gate
        self.entered = threading.Event()
        self.on_send = None
        self.files = []
        self.msgs = []
        self.stream = []
        self._calls = 0
        self._lock = threading.Lock()

    def send_file(self, msg, path):
        if self.on_send is not None:
            self.on_send()
        with self._lock:
            self._calls += 1
            first = self._calls == 1
        if first and self.gate is not None:
            self.entered.set()
            self.gate.wait(30)
        data = _c1_bytes(path)
        sha = hashlib.sha256(data).hexdigest()
        m = dict(msg, sha256=sha, size=len(data), file=True)
        with self._lock:
            self.files.append((m, data))
            self.stream.append((m, data))
        return sha

    def send(self, msg, body=b""):
        with self._lock:
            self.stream.append((dict(msg), None))
            if msg["kind"] != "CKPT":
                self.msgs.append(dict(msg))


def _c1_agent(tmp_path, ch):
    a = WorkerAgent("127.0.0.1:9", "127.0.0.1", 0, workdir=str(tmp_path / "w"), bench=False)
    out = os.path.join(a.workdir, "runs", "run1", "out")
    os.makedirs(out)
    a.assignment = dict(run_id="run1", attempt=1, rank=1, peer=0, tag="t", out_dir=out, final=False)
    a.ch = ch
    return a, engine_shard_path(out, "t", 1, 0)


def _c1_commit(a, step, data=None, keep=checkpoint.WAVE_KEEP):
    """The stage's commit of wave `step` of slot r1p0 (distinct bytes per step)."""
    return checkpoint.commit_wave(
        a.assignment["out_dir"],
        "t",
        1,
        0,
        step,
        _c1_saver(_c1_wave(step) if data is None else data),
        keep=keep,
    )[0]


def _c1_thread(errs, fn, *args):
    def run():
        try:
            fn(*args)
        except BaseException as e:  # noqa: BLE001 -- re-raised by the test as a failure
            errs.append(e)

    t = threading.Thread(target=run, daemon=True)
    return t


def _c1_shipped(ch):
    """[(step, manifest, {name: bytes})] in send order. Asserts the wire shape:
    each wave is its CKPT_PARTs, contiguous, then one CKPT whose manifest lists
    exactly those parts with the sha256 of the bytes sent; no part is left
    without its CKPT."""
    out, parts = [], []
    for msg, data in ch.stream:
        if msg["kind"] == "CKPT_PART":
            assert set(msg) == {
                "kind",
                "run_id",
                "attempt",
                "step",
                "name",
                "sha256",
                "size",
                "file",
            }
            assert not parts or parts[0][0]["step"] == msg["step"], "parts of two waves interleave"
            parts.append((msg, data))
        elif msg["kind"] == "CKPT":
            assert set(msg) == {"kind", "run_id", "attempt", "step", "manifest"}, msg
            m = msg["manifest"]
            assert m["step"] == msg["step"]
            got = {p["name"]: d for p, d in parts}
            assert len(got) == len(parts) and all(p["step"] == msg["step"] for p, _ in parts)
            assert {e["name"]: e["sha256"] for e in m["files"]} == {
                n: hashlib.sha256(d).hexdigest() for n, d in got.items()
            }
            assert all(
                (p["run_id"], p["attempt"]) == (msg["run_id"], msg["attempt"]) for p, _ in parts
            )
            out.append((msg["step"], m, got))
            parts = []
    assert parts == [], "CKPT_PARTs without their CKPT"
    return out


def _c1_assert_labels_true(ch):
    for step, _m, files in _c1_shipped(ch):
        assert files["state.pt"] == _c1_wave(step), (
            f"CKPT labeled step {step} carries other bytes ({files['state.pt'][:12]!r})"
        )


def _c1_steps(ch):
    return [s for s, _, _ in _c1_shipped(ch)]


def test_ship_labels_each_wave_from_its_manifest(tmp_path):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: the shipper labels a wave from anything but its own wave.json (the alias's
    # sidecar, which names the newest wave, or the scan position): wave 3's bytes then go out as
    # step 7; or it skips a committed wave, ships one out of step order, or ships one twice
    gate = threading.Event()
    ch = _C1FakeChannel(gate=gate)
    a, pt = _c1_agent(tmp_path, ch)
    held = {s: checkpoint.validate_wave(_c1_commit(a, s), s) for s in (3, 5, 7)}
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(pt)) == 7  # the alias names the newest
    errs = []
    ship = _c1_thread(errs, a._ship_waves, 1)
    ship.start()
    assert ch.entered.wait(10), "the ship never reached send_file"
    held[9] = checkpoint.validate_wave(_c1_commit(a, 9), 9)  # while wave 3 is on the wire;
    # its commit prunes wave 3 (keep 3), which still ships whole
    gate.set()
    ship.join(10)
    assert not ship.is_alive() and not errs, errs
    _c1_assert_labels_true(ch)
    assert _c1_steps(ch) == [3, 5, 7]
    a._ship_waves(1)
    _c1_assert_labels_true(ch)
    assert _c1_steps(ch) == [3, 5, 7, 9]
    assert {step: m for step, m, _ in _c1_shipped(ch)} == held


class _C1WatchedLock:
    """The agent's real _ship_lock, wrapped: `waiting` is set when a caller
    finds it held and blocks for it."""

    def __init__(self, lock, waiting=None):
        self._lock = lock
        self.waiting = waiting or threading.Event()

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self.waiting.set()
            self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()


def test_concurrent_ships_never_cross_waves(tmp_path, monkeypatch):
    # covers: CONTROL-I10
    # WOULD-FAIL-IF: _ship_waves runs without _ship_lock (a second shipper sends wave 3 again
    # while the first is still sending it, so parts of two ships interleave and a wave goes out
    # twice), or a wave is never shipped
    gate = threading.Event()
    ch = _C1FakeChannel(gate=gate)
    a, _pt = _c1_agent(tmp_path, ch)
    # Set once ship 2 has gone as far as it can while ship 1 sends: blocked on
    # the held lock, or (without the lock) at its own first send. An Event,
    # never a timed join.
    ship2_done_copy = threading.Event()
    monkeypatch.setattr(a, "_ship_lock", _C1WatchedLock(a._ship_lock, ship2_done_copy))
    _c1_commit(a, 3)
    errs = []
    ship1 = _c1_thread(errs, a._ship_waves, 1)
    ship1.start()
    assert ch.entered.wait(10), "ship 1 never reached send_file"
    _c1_commit(a, 5)
    ship2 = _c1_thread(errs, a._ship_waves, 1)

    def mark():
        if threading.current_thread() is ship2:
            ship2_done_copy.set()

    ch.on_send = mark
    ship2.start()
    assert ship2_done_copy.wait(10), "ship 2 neither waited for the lock nor sent"
    gate.set()
    ship1.join(10)
    ship2.join(10)
    assert not ship1.is_alive() and not ship2.is_alive()
    assert not errs, errs
    _c1_assert_labels_true(ch)
    assert sorted(_c1_steps(ch)) == [3, 5]


def test_restore_writes_the_wave_and_its_alias(tmp_path):
    # covers: CONTROL-I10
    # WOULD-FAIL-IF: _on_restore leaves the alias or its sidecar naming wave 5 over the restored
    # wave 3, writes no committed wave 3, or rewrites the slot's own wave 5
    ch = _C1FakeChannel()
    a, pt = _c1_agent(tmp_path, ch)
    d5 = _c1_commit(a, 5)
    before = _wv_tree(d5)
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(pt)) == 5
    msg = _c1_restore(a, tmp_path / "w" / "incoming", 3)
    a._on_restore(msg, b"")
    assert ch.msgs == [{"kind": "RESTORE_ACK", "rid": 7, "ok": True, "error": None}]
    d3 = checkpoint.wave_dir(a.assignment["out_dir"], "t", 1, 0, 3)
    assert checkpoint.validate_wave(d3, 3) == msg["manifest"]
    assert _c1_bytes(os.path.join(d3, "state.pt")) == _c1_wave(3)
    assert _c1_bytes(pt) == _c1_wave(3)
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(pt)) == 3
    assert _wv_tree(d5) == before


def test_export_wave_writes_each_wave_and_alias(tmp_path):
    # covers: CONTROL-I10
    # WOULD-FAIL-IF: export_wave writes an alias without its sidecar, or a sidecar naming a step
    # other than the exported wave's (a resume from the export is then refused), or writes no
    # committed wave under out/waves/ that validates as that step with the store's manifest
    store = CheckpointStore(str(tmp_path / "store"))
    slots_ = [(0, 0), (1, 0)]
    for slot in slots_:
        for step in (3, 5):
            _st_wave(store, slot, step)
    out = str(tmp_path / "out")
    paths = store.export_wave(5, slots_, out, "t")
    assert paths == [engine_shard_path(out, "t", r, p) for r, p in slots_]
    for slot, path in zip(slots_, paths):
        assert _c1_bytes(path) == _c1_wave(5, slot)
        assert checkpoint.read_sidecar(checkpoint.sidecar_of(path)) == 5
        checkpoint.check_resume_step(checkpoint.sidecar_of(path), 5)
        d = checkpoint.wave_dir(out, "t", slot[0], slot[1], 5)
        assert checkpoint.validate_wave(d, 5) == store.manifest(slot, 5)
        assert os.path.samefile(path, os.path.join(d, "state.pt"))
    assert sorted(os.listdir(out)) == sorted(
        [os.path.basename(p) for p in paths]
        + [os.path.basename(checkpoint.sidecar_of(p)) for p in paths]
        + ["waves"]
    )  # no export temp left


# ---- CONTROL-I23: a RESTORE is exclusive, leak-free and recorded ----


def _c1_restore(a, inbox, step=3, rid=7, data=None, part_sha=None):
    """Deliver wave `step`'s RESTORE_PART to the agent (its body written into
    the inbox, as the channel would) and return the RESTORE that commits it.
    part_sha replaces the sha256 the part's header claims."""
    data = _c1_wave(step) if data is None else data
    os.makedirs(inbox, exist_ok=True)
    body = os.path.join(inbox, f"part-{rid}.part")
    with open(body, "wb") as f:
        f.write(data)
    sha = hashlib.sha256(data).hexdigest()
    part = {
        "kind": "RESTORE_PART",
        "run_id": "run1",
        "rank": 1,
        "peer": 0,
        "step": step,
        "name": "state.pt",
        "sha256": sha if part_sha is None else part_sha,
        "size": len(data),
        "file": True,
    }
    a._on_restore_part(part, body)
    return {
        "kind": "RESTORE",
        "rid": rid,
        "run_id": "run1",
        "tag": "t",
        "rank": 1,
        "peer": 0,
        "step": step,
        "manifest": _wv_manifest(data, step),
    }


_C1_SLOW_EXIT = (
    "import os, signal, time\n"
    "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(1), os._exit(0)))\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)


def _c1_long_stage(tmp_path):
    """A real supervised child that stays alive until killed and takes a second to exit on
    SIGTERM, so only a kill that waits for the exit sees it dead. Returns once its SIGTERM
    handler is installed."""
    ready = threading.Event()
    stage = agent_mod.StageProcess(
        [sys.executable, "-c", _C1_SLOW_EXIT],
        str(tmp_path / "stage" / "stage.log"),
        dict(os.environ),
        lambda line: ready.set() if line == "ready" else None,
    )
    if not ready.wait(30):
        stage.kill()
        stage.close()
        raise AssertionError("stage child never reported ready")
    return stage


def test_restore_is_refused_while_the_stage_is_alive(tmp_path):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: _on_restore has no alive check (the live stage's alias and sidecar are moved
    # under it to a committed wave 3), or an ABORT of the live stage does not wait for its exit
    # (no stage.kill() in _on_abort, or StageProcess.kill without proc.wait), so the next
    # RESTORE of a heal is refused
    ch = _C1FakeChannel()
    a, pt = _c1_agent(tmp_path, ch)
    inbox = tmp_path / "w" / "incoming"
    _c1_commit(a, 5)
    stage = a.stage = _c1_long_stage(tmp_path)
    try:
        assert stage.alive()
        a._on_restore(_c1_restore(a, inbox, 3), b"")
        assert len(ch.msgs) == 1
        ack = ch.msgs[0]
        assert (ack["kind"], ack["rid"], ack["ok"]) == ("RESTORE_ACK", 7, False)
        assert "alive" in ack["error"] and f"pid {stage.proc.pid}" in ack["error"], ack
        assert "r1p0 attempt 1" in ack["error"], ack
        assert _c1_bytes(pt) == _c1_wave(5)
        assert checkpoint.read_sidecar(checkpoint.sidecar_of(pt)) == 5
        assert not os.path.exists(checkpoint.wave_dir(a.assignment["out_dir"], "t", 1, 0, 3))
        assert os.listdir(inbox) == []
        assert ("run1", 1, 0, 3) not in a._shipped

        # A heal ABORTs the still-live stage first; the ABORT waits for the stage's exit, so the
        # RESTORE that follows succeeds. Nothing kills the stage before this ABORT.
        assert stage.alive()
        out = a.assignment["out_dir"]
        a._on_abort({"kind": "ABORT", "rid": 8}, b"")
        assert not stage.alive()
        assert _c1_steps(ch) == [5]  # the ABORT shipped the committed wave
        a.assignment = dict(run_id="run1", attempt=1, rank=1, peer=0, tag="t", out_dir=out)
        a._on_restore(_c1_restore(a, inbox, 3, rid=9), b"")
        assert ch.msgs[-1] == {"kind": "RESTORE_ACK", "rid": 9, "ok": True, "error": None}
        assert _c1_bytes(pt) == _c1_wave(3)
        assert checkpoint.read_sidecar(checkpoint.sidecar_of(pt)) == 3
        assert os.listdir(inbox) == []
    finally:
        stage.kill()
        stage.close()


def test_restore_and_ship_never_interleave(tmp_path, monkeypatch):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: _on_restore runs without _ship_lock (the restore commits wave 3, moves the
    # alias and acknowledges while a ship of wave 5 is still inside send)
    gate = threading.Event()
    ch = _C1FakeChannel(gate=gate)
    a, pt = _c1_agent(tmp_path, ch)
    # Set when the restore blocks on the held lock, or (without the lock) when it has finished.
    restore_progressed = threading.Event()
    monkeypatch.setattr(a, "_ship_lock", _C1WatchedLock(a._ship_lock, restore_progressed))
    _c1_commit(a, 5)
    errs = []
    ship = _c1_thread(errs, a._ship_waves, 1)
    ship.start()
    assert ch.entered.wait(10), "the ship never reached send_file"
    msg = _c1_restore(a, tmp_path / "w" / "incoming", 3)

    def restore():
        try:
            a._on_restore(msg, b"")
        finally:
            restore_progressed.set()

    restorer = _c1_thread(errs, restore)
    restorer.start()
    assert restore_progressed.wait(10), "the restore neither waited for the lock nor finished"
    assert ch.msgs == []  # no RESTORE_ACK while the ship holds the lock
    assert _c1_bytes(pt) == _c1_wave(5)
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(pt)) == 5
    gate.set()
    ship.join(10)
    restorer.join(10)
    assert not ship.is_alive() and not restorer.is_alive()
    assert not errs, errs
    _c1_assert_labels_true(ch)
    assert _c1_steps(ch) == [5]
    assert ch.msgs == [{"kind": "RESTORE_ACK", "rid": 7, "ok": True, "error": None}]
    assert _c1_bytes(pt) == _c1_wave(3)
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(pt)) == 3


def test_restore_unlinks_the_body_on_every_path(tmp_path):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: _on_restore has no try/finally (a missing or non-integer step, or a message
    # that raises, leaves the received parts in the inbox), accepts a step that is not an int,
    # leaves the parts after a sha256 mismatch, or commits a wave whose part does not match the
    # manifest
    ch = _C1FakeChannel()
    a, pt = _c1_agent(tmp_path, ch)
    _c1_commit(a, 5)
    inbox = tmp_path / "w" / "incoming"
    os.makedirs(inbox)
    out = a.assignment["out_dir"]

    for i, step in enumerate((None, "3", 3.0, True, [3])):
        msg = _c1_restore(a, inbox, 3, rid=i)
        if step is None:
            del msg["step"]
        else:
            msg["step"] = step
        a._on_restore(msg, b"")
        ack = ch.msgs[-1]
        assert (ack["kind"], ack["rid"], ack["ok"]) == ("RESTORE_ACK", i, False), ack
        assert ack["error"].startswith("malformed step"), ack
        assert os.listdir(inbox) == [], step

    msg = _c1_restore(a, inbox, 3, rid=20)
    msg["rank"] = "r1"  # int() raises: _guard reports it as an ERROR
    with pytest.raises(ValueError):
        a._on_restore(msg, b"")
    assert os.listdir(inbox) == []

    msg = _c1_restore(a, inbox, 3, rid=21, part_sha="0" * 64)  # the part claims other bytes
    a._on_restore(msg, b"")
    ack = ch.msgs[-1]
    assert (ack["rid"], ack["ok"]) == (21, False) and "sha256 mismatch" in ack["error"], ack
    assert os.listdir(inbox) == []

    flipped = bytearray(_c1_wave(3))
    flipped[-1] ^= 1
    msg = _c1_restore(a, inbox, 3, rid=22, data=bytes(flipped))
    msg["manifest"] = _wv_manifest(_c1_wave(3), 3)
    msg = dict(msg)
    a._on_restore(msg, b"")  # header and manifest agree; the bytes do not
    ack = ch.msgs[-1]
    assert (ack["rid"], ack["ok"]) == (22, False) and "sha256" in ack["error"], ack
    assert os.listdir(inbox) == []

    assert _c1_bytes(pt) == _c1_wave(5)  # no refused restore touched the alias
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(pt)) == 5
    assert [s for s, _ in checkpoint.slot_waves(out, "t", 1, 0)] == [5]

    a._on_restore(_c1_restore(a, inbox, 3, rid=23), b"")
    assert ch.msgs[-1] == {"kind": "RESTORE_ACK", "rid": 23, "ok": True, "error": None}
    assert os.listdir(inbox) == []
    assert _c1_bytes(pt) == _c1_wave(3)
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(pt)) == 3
    assert ch.files == []
    run_dir = os.path.join(a.workdir, "runs", "run1")
    assert sorted(os.listdir(run_dir)) == ["out"]  # no restore staging left


def test_restored_wave_is_not_reshipped(tmp_path):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: _on_restore does not enter the restored (run_id, rank, peer, step) in
    # _shipped (the next ship sends wave 3 back to the coordinator that already holds it), or
    # it marks a wave the stage has not saved since (wave 5 is then never shipped)
    ch = _C1FakeChannel()
    a, _pt = _c1_agent(tmp_path, ch)
    a._on_restore(_c1_restore(a, tmp_path / "w" / "incoming", 3), b"")
    assert ch.msgs[-1]["ok"] is True
    assert ("run1", 1, 0, 3) in a._shipped
    a._ship_waves(1)
    assert ch.files == []
    _c1_commit(a, 5)
    a._ship_waves(1)
    _c1_assert_labels_true(ch)
    assert _c1_steps(ch) == [5]


def test_restore_refuses_a_local_wave_with_another_manifest(tmp_path):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: a RESTORE replaces (or silently keeps and acknowledges) the slot's own
    # committed wave of the same step whose manifest differs, or rewrites an equal one instead of
    # keeping it untouched and pointing the alias at it
    ch = _C1FakeChannel()
    a, pt = _c1_agent(tmp_path, ch)
    inbox = tmp_path / "w" / "incoming"
    other = _c1_wave(3, (7, 7))  # same step, other bytes
    d3 = _c1_commit(a, 3, data=other)
    _c1_commit(a, 5)
    held = _wv_tree(d3)
    a._on_restore(_c1_restore(a, inbox, 3, rid=1), b"")
    ack = ch.msgs[-1]
    assert (ack["rid"], ack["ok"]) == (1, False) and "another manifest" in ack["error"], ack
    assert _wv_tree(d3) == held and _c1_bytes(pt) == _c1_wave(5)
    assert ("run1", 1, 0, 3) not in a._shipped
    assert os.listdir(inbox) == []

    a2, pt2 = _c1_agent(tmp_path / "equal", _C1FakeChannel())
    d3 = _c1_commit(a2, 3)
    _c1_commit(a2, 5)
    held = _wv_tree(d3)
    a2._on_restore(_c1_restore(a2, tmp_path / "equal" / "w" / "incoming", 3, rid=2), b"")
    assert a2.ch.msgs[-1] == {"kind": "RESTORE_ACK", "rid": 2, "ok": True, "error": None}
    assert _wv_tree(d3) == held  # kept untouched
    assert os.path.samefile(pt2, os.path.join(d3, "state.pt"))
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(pt2)) == 3


# ---- CONTROL-I23: every committed wave ships, in order, as a unit ----

_SHIP_STAGE = (
    "import sys, time\n"
    "from wantrain.fault_tolerance import checkpoint\n"
    "out = sys.argv[1]\n"
    "for step in range(10):\n"
    "    data = (f'wave-{step}-slot-r1p0|'.encode() * 2048) + bytes([step, 1, 0])\n"
    "    def save(tmp, data=data):\n"
    "        with open(tmp, 'wb') as f:\n"
    "            f.write(data)\n"
    "    _, new = checkpoint.commit_wave(out, 't', 1, 0, step, save)\n"
    "    if new:\n"
    "        print(f'WT_WAVE {step}', flush=True)\n"
    "    time.sleep(0.02)\n"
)


def test_every_committed_wave_ships_in_order_under_fast_saves(tmp_path, capsys):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: the agent samples the slot every 0.5 s (the pre-task _supervise: it sees
    # about one of the 10 waves the stage commits 20 ms apart and keeps 3 of, so the rest are
    # pruned unshipped), ships only the newest wave of a scan, or is not woken by WT_WAVE: a wave
    # is then missing, shipped twice or out of step order, or a ship gap is logged
    ch = _C1FakeChannel()
    a, _pt = _c1_agent(tmp_path, ch)
    stage = agent_mod.StageProcess(
        [sys.executable, "-c", _SHIP_STAGE, a.assignment["out_dir"]],
        str(tmp_path / "stage" / "stage.log"),
        a._env(),
        lambda line: a._on_stage_line(1, line),
    )
    a.stage = stage
    a._supervise(stage, 1)  # returns once the stage exited and its exit was reported
    assert stage.rc == 0, stage.tail()
    _c1_assert_labels_true(ch)
    assert _c1_steps(ch) == list(range(10))
    lines = capsys.readouterr().out.splitlines()
    assert [ln for ln in lines if wave_experiment.GAP_RE.match(ln)] == []
    shipped = [wave_experiment.SHIP_RE.match(ln) for ln in lines]
    assert [int(m.group(4)) for m in shipped if m] == list(range(10))
    assert ch.msgs[0]["kind"] == "EXIT" and ch.msgs[0]["rc"] == 0


def test_a_wire_step_past_int64_is_refused_as_a_bad_step(tmp_path):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: a step is any non-negative int: step 10**250 then names a directory past
    # NAME_MAX and the store raises OSError (ENAMETOOLONG) from mkdtemp/makedirs instead of
    # refusing the part as a malformed step
    store = CheckpointStore(str(tmp_path / "store"))
    body = tmp_path / "part"
    body.write_bytes(b"x")
    for step in (1 << 63, 10**250):
        with pytest.raises(ValueError, match="bad wave step"):
            checkpoint.wave_dir(str(tmp_path), "t", 0, 0, step)
        with pytest.raises(ValueError, match="bad wave step"):
            store.stage_part((0, 0), 1, step, "state.pt", "0" * 64, str(body))
    assert body.exists()
    assert checkpoint.wave_dir(str(tmp_path), "t", 0, 0, (1 << 63) - 1).endswith(os.sep)


_DYING_STAGE = (
    "import os, sys, time\n"
    "while not os.path.exists(sys.argv[1]):\n"
    "    time.sleep(0.01)\n"
    "print('RuntimeError: reader: peer connection lost (ConnectionError: peer closed)')\n"
    "print('WT_DONE 1', flush=True)\n"
    "sys.exit(1)\n"
)


def test_a_dead_stage_reports_its_exit_while_a_wave_is_on_the_wire(tmp_path):
    # covers: CONTROL-I10
    # WOULD-FAIL-IF: the agent reports a stage's non-zero exit only from its supervisor, after
    # the next scan and after the wave it is shipping (the pre-fix _supervise): a dead stage's
    # EXIT then reaches the judge after its peer's "peer lost" EXIT, and the judge restarts in
    # place instead of convicting it (tests/chaos test_killed_stage_process_is_convicted_and_healed
    # failed 2 of 12 runs that way); or the reaper reports before the stage's output is drained,
    # so its done marker or "peer lost" tail is missing; or the exit is reported twice
    gate = threading.Event()
    ch = _C1FakeChannel(gate=gate)
    a, _pt = _c1_agent(tmp_path, ch)
    _c1_commit(a, 3)  # committed, not shipped: the supervisor's first scan ships it
    go = tmp_path / "die"
    stage = agent_mod.StageProcess(
        [sys.executable, "-c", _DYING_STAGE, str(go)],
        str(tmp_path / "stage" / "stage.log"),
        a._env(),
        lambda line: a._on_stage_line(1, line),
        on_exit=lambda st: a._on_stage_exit(1, st),
    )
    a.stage = stage
    errs = []
    sup = _c1_thread(errs, a._supervise, stage, 1)
    sup.start()
    try:
        assert ch.entered.wait(30), "the supervisor never started shipping wave 3"
        go.write_text("")  # the stage dies while wave 3 is on the wire
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not any(m["kind"] == "EXIT" for m in ch.msgs):
            time.sleep(0.01)
        exits = [m for m in ch.msgs if m["kind"] == "EXIT"]
        assert len(exits) == 1, ch.msgs
        assert (exits[0]["rc"], exits[0]["done_marker"]) == (1, True), exits[0]
        assert PEER_LOST in exits[0]["log_tail"], exits[0]
        assert ch.files == [], "the wave finished before the exit was reported"
    finally:
        gate.set()
        sup.join(30)
        if stage.alive():
            stage.kill()
    assert not sup.is_alive() and not errs, errs
    assert [m["kind"] for m in ch.msgs].count("EXIT") == 1, ch.msgs
    assert _c1_steps(ch) == [3]
    assert stage.reported


def test_ship_logs_each_wave_in_the_experiment_format(tmp_path, capsys):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: a shipped wave's log line does not parse with wave_experiment.SHIP_RE (the
    # experiment then counts it unshipped), a 'seen' or 'race-skip' line is still printed, a
    # wave announced but pruned before its ship is not logged as a GAP_RE line, or is logged
    # again on every pass
    ch = _C1FakeChannel()
    a, _pt = _c1_agent(tmp_path, ch)
    slot_dir = os.path.dirname(
        os.path.normpath(checkpoint.wave_dir(a.assignment["out_dir"], "t", 1, 0, 0))
    )
    d1 = _c1_commit(a, 1)
    a._on_stage_line(1, "WT_WAVE 1")
    checkpoint._remove_wave(slot_dir, d1)  # pruned before any ship
    _c1_commit(a, 3)
    a._on_stage_line(1, "WT_WAVE 3")
    a._ship_waves(1)
    _c1_commit(a, 5)
    a._on_stage_line(1, "WT_WAVE 5")
    a._ship_waves(1)
    a._ship_waves(1)
    assert _c1_steps(ch) == [3, 5]
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[agent ")]
    assert lines, "no agent line"
    ship = [wave_experiment.SHIP_RE.match(ln) for ln in lines]
    gap = [wave_experiment.GAP_RE.match(ln) for ln in lines]
    assert all(s or g for s, g in zip(ship, gap)), lines
    assert [(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)) for m in ship if m] == [
        (a.name, "1", "0", "3", "shipped"),
        (a.name, "1", "0", "5", "shipped"),
    ]
    assert [(m.group(2), m.group(3), m.group(4)) for m in gap if m] == [("1", "0", "1")]
    assert not any(wave_experiment.SEEN_RE.match(ln) for ln in lines)


def test_ship_never_opens_the_alias(tmp_path, monkeypatch):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: the shipper reads the alias <tag>_ckpt_rank<r>p<p>.pt or its .step.json
    # (the mutable pair a save or a RESTORE replaces) instead of the committed wave directory:
    # the alias, replaced here by other bytes, would then be what ships
    import builtins

    ch = _C1FakeChannel()
    a, pt = _c1_agent(tmp_path, ch)
    _c1_commit(a, 3)
    stray = str(tmp_path / "stray.pt")
    _c1_saver(b"not-the-wave")(stray)
    os.replace(stray, pt)  # the alias now holds other bytes
    touched = []
    real_open, real_os_open, real_link = builtins.open, os.open, os.link

    def rec(path):
        p = os.path.abspath(os.fspath(path)) if isinstance(path, (str, os.PathLike)) else path
        if p in (pt, checkpoint.sidecar_of(pt)):
            touched.append(p)

    monkeypatch.setattr(builtins, "open", lambda f, *x, **k: (rec(f), real_open(f, *x, **k))[1])
    monkeypatch.setattr(os, "open", lambda f, *x, **k: (rec(f), real_os_open(f, *x, **k))[1])
    monkeypatch.setattr(os, "link", lambda s, d, *x, **k: (rec(s), real_link(s, d, *x, **k))[1])
    a._ship_waves(1)
    monkeypatch.undo()
    assert touched == []
    _c1_assert_labels_true(ch)
    assert _c1_steps(ch) == [3]


# ---- the coordinator's wave store: staged parts, committed waves (CONTROL-I20, BA-43) ----


def _st_wave(store, slot, step, data=None, attempt=1):
    """Commit a wave into the store through the real staging path: its one
    part (staged as a received body would be), then its manifest."""
    data = _c1_wave(step, slot) if data is None else data
    body = os.path.join(os.path.dirname(store.root), f"in-r{slot[0]}p{slot[1]}-{step}.part")
    _c1_saver(data)(body)
    sha = hashlib.sha256(data).hexdigest()
    store.stage_part(slot, attempt, step, "state.pt", sha, body)
    assert store.commit_staged(slot, attempt, step, _wv_manifest(data, step)) is True


def _st_part(run, step, name, data, attempt=None, sha=None, **extra):
    """One CKPT_PART as the coordinator's reader hands it over: header and the
    path of its received body (in the inbox next to the run)."""
    inbox = os.path.join(os.path.dirname(run.run_dir), "inbox")
    os.makedirs(inbox, exist_ok=True)
    body = os.path.join(inbox, f"{name}-{step}-{len(os.listdir(inbox))}-{os.urandom(4).hex()}.part")
    _c1_saver(data)(body)
    msg = {
        "kind": "CKPT_PART",
        "run_id": run.run_id,
        "attempt": run.attempt if attempt is None else attempt,
        "step": step,
        "name": name,
        "sha256": hashlib.sha256(data).hexdigest() if sha is None else sha,
        "size": len(data),
        "file": True,
        **extra,
    }
    return msg, body


def _st_ckpt(run, step, manifest, attempt=None):
    msg = {
        "kind": "CKPT",
        "run_id": run.run_id,
        "attempt": run.attempt if attempt is None else attempt,
        "step": step,
        "manifest": manifest,
    }
    return msg, b""


def _st_feed(coord, run, slot_or_node, script):
    """Through the real _conn_loop of the connection (FAKE scripted channel at
    the socket boundary, _c2_scripted_channel). Returns the read count."""
    nid = run.slots.get(slot_or_node, slot_or_node)
    w = coord.workers[nid]
    ch, reads = _c2_scripted_channel(script)
    w.ch = ch
    coord._conn_loop(w, ch)
    return len(reads)


def _st_left(run):
    """Files left in the inbox and in the store's staging."""
    inbox = os.path.join(os.path.dirname(run.run_dir), "inbox")
    staged = os.path.join(run.store.root, ".incoming")
    return [
        os.path.relpath(os.path.join(d, f), os.path.dirname(run.run_dir))
        for root in (inbox, staged)
        for d, _, fs in os.walk(root)
        for f in fs
    ]


def _st_warn(coord, worker, kind):
    return [
        e[2]
        for e in coord.events
        if e[1] == "warn" and e[2].startswith(f"malformed {kind} from {worker}: ")
    ]


def _st_wave_script(run, step, data, slot=(1, 1)):
    part = _st_part(run, step, "state.pt", data)
    return [part, _st_ckpt(run, step, _wv_manifest(data, step))]


def test_store_refuses_a_wave_with_a_mislabeled_manifest_or_file(tmp_path):
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: the store commits a wave whose CKPT manifest names another step than the
    # CKPT (a wave of step 2 held as step 4), whose manifest lists a file with another sha256
    # than its staged part, or whose part's bytes do not match the hash both claim; or it keeps
    # a refused wave's staged parts
    coord, run = _c2_coordinator(tmp_path)
    data = _c1_wave(4, (1, 1))
    flipped = bytearray(data)
    flipped[-1] ^= 1
    m4 = _wv_manifest(data, 4)
    script = [
        _st_part(run, 4, "state.pt", data),
        _st_ckpt(run, 4, _wv_manifest(data, 2)),  # names step 2
        _st_part(run, 4, "state.pt", data),
        _st_ckpt(run, 4, _wv_manifest(b"other bytes", 4)),  # another sha256
        _st_part(run, 4, "state.pt", bytes(flipped), sha=m4["files"][0]["sha256"]),
        _st_ckpt(run, 4, m4),  # header and manifest agree; the bytes do not
        _st_part(run, 4, "state.pt", data),
        _st_ckpt(run, 4, dict(m4, step=True)),
    ]
    assert _st_feed(coord, run, (1, 1), script) == len(script) + 1
    assert run.store.index() == {}
    assert _st_left(run) == []
    assert len(_st_warn(coord, "w1", "CKPT")) == 4, coord.events

    assert _st_feed(coord, run, (1, 1), _st_wave_script(run, 4, data)) == 3
    assert run.store.index() == {(1, 1): [4]}
    d = run.store.wave_path((1, 1), 4)
    assert checkpoint.validate_wave(d, 4) == m4
    assert _c1_bytes(os.path.join(d, "state.pt")) == data
    assert _st_left(run) == []


def test_ckpt_part_from_another_slots_connection_is_refused(tmp_path):
    # covers: CONTROL-I20
    # WOULD-FAIL-IF: CKPT_PART or CKPT take the slot from the message (a rank/peer naming slot
    # (0, 0) on (1, 1)'s connection stages or commits a wave as (0, 0)'s), a worker holding no
    # slot of the run stages a part at all, or a refused part's body is left in the inbox
    coord, run = _c2_coordinator(tmp_path)
    spare = n(2, gpu="cpu", device="cpu", cap=None)
    coord.workers[spare.node_id] = Worker(spare, None, "session2", None)
    forged = _c1_wave(2, (0, 0))
    own = _c1_wave(4, (1, 1))
    script = [
        _st_part(run, 2, "state.pt", forged, rank=0, peer=0),
        _st_ckpt(run, 2, _wv_manifest(forged, 2)),
        (dict(_st_ckpt(run, 2, _wv_manifest(forged, 2))[0], rank=0, peer=0), b""),
        _st_part(run, 3, "state.pt", forged, peer=0),
        *_st_wave_script(run, 4, own),
    ]
    assert _st_feed(coord, run, (1, 1), script) == len(script) + 1
    assert _st_feed(coord, run, spare.node_id, _st_wave_script(run, 6, forged)) == 3
    assert run.store.index() == {(1, 1): [4]}, run.store.index()
    assert _c1_bytes(os.path.join(run.store.wave_path((1, 1), 4), "state.pt")) == own
    assert _st_left(run) == []
    assert len(_st_warn(coord, "w1", "CKPT_PART")) == 2, coord.events
    assert len(_st_warn(coord, "w1", "CKPT")) == 2, coord.events
    assert len(_st_warn(coord, "w2", "CKPT_PART")) == 1 and len(_st_warn(coord, "w2", "CKPT")) == 1


def test_ckpt_part_of_another_attempt_is_refused(tmp_path):
    # covers: CONTROL-I20
    # WOULD-FAIL-IF: CKPT_PART or CKPT do not check the attempt (the previous or a later one, or
    # True or "1" coerced by int()), or accept a step that is not a non-negative int ("6", 7.5,
    # True, -1): any of those parts is then staged and its wave sits in the store next to step 5
    coord, run = _c2_coordinator(tmp_path)
    data = _c1_wave(5, (1, 1))
    m5 = _wv_manifest(data, 5)
    script = []
    for attempt in (0, 2, True, "1"):
        script += [
            _st_part(run, 5, "state.pt", data, attempt=attempt),
            _st_ckpt(run, 5, m5, attempt=attempt),
        ]
    for step in (-1, True, "6", 7.5, None):
        script += [_st_part(run, step, "state.pt", data), _st_ckpt(run, step, m5)]
    script += _st_wave_script(run, 5, data)
    assert _st_feed(coord, run, (1, 1), script) == len(script) + 1
    assert run.store.index() == {(1, 1): [5]}, run.store.index()
    assert _st_left(run) == []
    assert len(_st_warn(coord, "w1", "CKPT_PART")) == 9, coord.events
    assert len(_st_warn(coord, "w1", "CKPT")) == 9, coord.events


def test_third_open_step_drops_staged_parts(tmp_path):
    # covers: CONTROL-I20
    # WOULD-FAIL-IF: staging is unbounded (a slot's parts of a third uncommitted step are staged
    # next to the other two, and a worker that never sends CKPT fills the coordinator's disk), a
    # part of a third step drops less than every staged part of the slot, or the reader stops
    coord, run = _c2_coordinator(tmp_path)
    w2, w4, w6 = (_c1_wave(s, (1, 1)) for s in (2, 4, 6))
    script = [
        _st_part(run, 2, "state.pt", w2),
        _st_part(run, 4, "state.pt", w4),
        _st_part(run, 6, "state.pt", w6),  # a third open step
        _st_ckpt(run, 2, _wv_manifest(w2, 2)),
        _st_ckpt(run, 4, _wv_manifest(w4, 4)),
        *_st_wave_script(run, 6, w6),
    ]
    staged_before_third = []
    real = run.store.stage_part

    def watch(slot, attempt, step, *a):
        if step == 6 and not staged_before_third:
            staged_before_third.append(sorted(_st_left(run)))
        return real(slot, attempt, step, *a)

    run.store.stage_part = watch
    assert _st_feed(coord, run, (1, 1), script) == len(script) + 1
    assert len([p for p in staged_before_third[0] if ".incoming" in p]) == 2
    assert run.store.index() == {(1, 1): [6]}
    assert _st_left(run) == []
    warns = _st_warn(coord, "w1", "CKPT_PART")
    assert len(warns) == 1 and "third uncommitted step" in warns[0], warns
    assert len(_st_warn(coord, "w1", "CKPT")) == 2  # steps 2 and 4: their parts were dropped


def test_oversized_or_duplicate_part_is_refused_and_unlinked(tmp_path):
    # covers: CONTROL-I20
    # WOULD-FAIL-IF: a part larger than the bytes its manifest declares is committed (or kept
    # staged), a second part of one name replaces or joins the first, or a refused part or the
    # parts it spoils stay on disk
    coord, run = _c2_coordinator(tmp_path)
    data = _c1_wave(2, (1, 1))
    m2 = _wv_manifest(data, 2)
    script = [
        _st_part(run, 2, "state.pt", data),
        _st_part(run, 2, "state.pt", data),  # duplicate: the step's parts are dropped
        _st_ckpt(run, 2, m2),
        # Oversized: the header claims the manifest's sha256, the body carries one more byte.
        _st_part(run, 2, "state.pt", data + b"x", sha=m2["files"][0]["sha256"]),
        _st_ckpt(run, 2, m2),
        _st_part(run, 2, "../escape", data),  # not a wave file name
        *_st_wave_script(run, 2, data),
    ]
    assert _st_feed(coord, run, (1, 1), script) == len(script) + 1
    assert run.store.index() == {(1, 1): [2]}
    assert _c1_bytes(os.path.join(run.store.wave_path((1, 1), 2), "state.pt")) == data
    assert _st_left(run) == []
    assert not os.path.exists(os.path.join(run.store.root, "escape"))
    parts = _st_warn(coord, "w1", "CKPT_PART")
    assert len(parts) == 2 and "staged twice" in parts[0], parts
    ckpts = _st_warn(coord, "w1", "CKPT")
    assert len(ckpts) == 2 and "bytes" in ckpts[1], ckpts


# ---- wave layout: immutable step-named waves with a hashed manifest ----


def _wv_commit(out, step, slot=(1, 0), **kw):
    return checkpoint.commit_wave(
        out, "t", slot[0], slot[1], step, _c1_saver(_c1_wave(step, slot)), **kw
    )


def _wv_tree(root):
    """{relative path: (inode, sha256)} of every file under root."""
    out = {}
    for d, _, names in os.walk(root):
        for n in names:
            p = os.path.join(d, n)
            out[os.path.relpath(p, root)] = (os.stat(p).st_ino, sha256_file(p))
    return out


def _wv_manifest(data, step):
    import hashlib

    return {
        "v": 1,
        "step": step,
        "files": [
            {"name": "state.pt", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        ],
    }


def _wv_stage(tmp_path, name, data, extra=None):
    staged = tmp_path / name
    staged.mkdir()
    (staged / "state.pt").write_bytes(data)
    for k, v in (extra or {}).items():
        (staged / k).write_bytes(v)
    return str(staged)


def test_wave_commit_is_immutable_and_wave_json_last(tmp_path, monkeypatch):
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: commit_wave writes wave.json before state.pt is complete, renames the
    # directory into place before wave.json exists, writes into the final directory directly,
    # uses another directory name than step-<step:08d>, or rewrites a committed wave later
    out = str(tmp_path / "out")
    final = checkpoint.wave_dir(out, "t", 1, 0, 3)
    assert final == os.path.join(out, "waves", "t", "r1p0", "step-00000003") + os.sep
    seen = []

    def save(path):
        seen.append(("save", os.path.dirname(path) != os.path.normpath(final)))
        assert not os.path.exists(os.path.join(os.path.dirname(path), "wave.json"))
        assert not os.path.exists(final)
        _c1_saver(_c1_wave(3))(path)

    real_rename = os.rename

    def rename(src, dst, *a, **kw):
        if os.path.normpath(str(dst)) == os.path.normpath(final):
            with open(os.path.join(src, "wave.json")) as f:
                m = json.load(f)
            assert m["files"][0]["sha256"] == sha256_file(os.path.join(src, "state.pt"))
            seen.append(("rename", sorted(os.listdir(src))))
        return real_rename(src, dst, *a, **kw)

    monkeypatch.setattr(os, "rename", rename)
    d, committed = checkpoint.commit_wave(out, "t", 1, 0, 3, save)
    monkeypatch.setattr(os, "rename", real_rename)
    assert (d, committed) == (final, True)
    assert seen == [("save", True), ("rename", ["state.pt", "wave.json"])]
    with open(os.path.join(d, "wave.json")) as f:
        assert json.load(f) == _wv_manifest(_c1_wave(3), 3)
    assert checkpoint.validate_wave(d, 3)["step"] == 3
    before = _wv_tree(d)
    for step in (4, 5):
        _wv_commit(out, step)
    _wv_commit(out, 3)  # the same step again: refused without a write
    assert _wv_tree(d) == before
    assert sorted(os.listdir(os.path.dirname(os.path.normpath(d)))) == [
        "step-00000003",
        "step-00000004",
        "step-00000005",
    ]


def test_final_step_is_committed_once(tmp_path):
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: a second commit of a step the slot already holds calls save_fn, rewrites the
    # wave, relinks the alias or returns committed=True (the last step is then announced twice)
    out = str(tmp_path / "out")
    d, committed = _wv_commit(out, 7)
    assert committed is True
    alias = engine_shard_path(out, "t", 1, 0)
    before = (_wv_tree(out), os.stat(alias).st_ino, os.stat(alias).st_mtime_ns)
    side_mtime = os.stat(checkpoint.sidecar_of(alias)).st_mtime_ns

    def never(path):
        raise AssertionError("save_fn called for a committed step")

    d2, committed2 = checkpoint.commit_wave(out, "t", 1, 0, 7, never)
    assert (d2, committed2) == (d, False)
    assert (_wv_tree(out), os.stat(alias).st_ino, os.stat(alias).st_mtime_ns) == before
    assert os.stat(checkpoint.sidecar_of(alias)).st_mtime_ns == side_mtime
    # a corrupt directory of that step is never overwritten nor reported as committed
    with open(os.path.join(d, "state.pt"), "r+b") as f:
        f.write(b"X")
    with pytest.raises(ValueError, match="sha256"):
        checkpoint.commit_wave(out, "t", 1, 0, 7, never)


def test_a_commit_interrupted_after_the_rename_converges_on_retry(tmp_path, monkeypatch):
    # covers: CONTROL-I10
    # WOULD-FAIL-IF: a retry of a step whose commit failed after the rename (ENOSPC in the
    # sidecar write, the hard link or the pruning) takes the committed=False early return
    # without finishing the alias and the pruning, so the alias stays on the previous step or
    # without a sidecar (check_resume_step then refuses the resume) and old waves pile up
    import errno

    out = str(tmp_path / "out")
    alias = engine_shard_path(out, "t", 1, 0)
    side = checkpoint.sidecar_of(alias)
    for step in (1, 2, 3):
        _wv_commit(out, step)

    def once(name, real):
        state = {"armed": True}

        def fail(*a, **kw):
            if state["armed"]:
                state["armed"] = False
                raise OSError(errno.ENOSPC, "No space left on device")
            return real(*a, **kw)

        monkeypatch.setattr(*name, fail)

    def retried(step, fault):
        once(*fault)
        with pytest.raises(OSError, match="No space"):
            _wv_commit(out, step)
        d = checkpoint.wave_dir(out, "t", 1, 0, step)
        checkpoint.validate_wave(d, step)  # the wave itself is committed
        assert _wv_commit(out, step) == (d, False)
        assert os.path.samefile(alias, os.path.join(d, "state.pt"))
        assert checkpoint.read_sidecar(side) == step
        checkpoint.check_resume_step(side, step)
        assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [
            step - 2,
            step - 1,
            step,
        ]

    retried(4, ((checkpoint, "write_sidecar"), checkpoint.write_sidecar))
    retried(5, ((os, "link"), os.link))
    retried(6, ((checkpoint, "_remove_wave"), checkpoint._remove_wave))
    # a retry of an older step never moves the alias off the newest wave
    newest = (os.stat(alias).st_ino, os.stat(side).st_mtime_ns)
    assert _wv_commit(out, 5)[1] is False
    assert (os.stat(alias).st_ino, os.stat(side).st_mtime_ns) == newest


def test_validate_wave_refuses_every_mislabel(tmp_path):
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: validate_wave accepts a wave without wave.json, an unparsable one, one naming
    # another step or version, a listed file missing or with another sha256 or size, or a file
    # present but not listed; or committed_waves lists such a directory
    import shutil

    out = str(tmp_path / "out")
    good, _ = _wv_commit(out, 3)
    assert checkpoint.validate_wave(good, 3) == _wv_manifest(_c1_wave(3), 3)
    data = _c1_wave(3)

    def broken(i, edit):
        d = str(tmp_path / f"bad{i}")
        shutil.copytree(good, d)
        edit(d)
        return d

    def manifest(m):
        def edit(d):
            with open(os.path.join(d, "wave.json"), "w") as f:
                json.dump(m, f)

        return edit

    def unlink(name):
        return lambda d: os.unlink(os.path.join(d, name))

    def write(name, b):
        def edit(d):
            with open(os.path.join(d, name), "wb") as f:
                f.write(b)

        return edit

    flipped = bytearray(data)
    flipped[0] ^= 1
    cases = [
        (unlink("wave.json"), 3, "wave.json is missing"),
        (write("wave.json", b"{not json"), 3, "unparsable"),
        (manifest(_wv_manifest(data, 4)), 3, "names step 4"),
        (manifest({**_wv_manifest(data, 3), "v": 2}), 3, "version 2"),
        (manifest({**_wv_manifest(data, 3), "v": True}), 3, "version True"),
        (lambda d: None, 4, "names step 3"),
        (unlink("state.pt"), 3, "state.pt is missing"),
        (write("state.pt", bytes(flipped)), 3, "sha256"),
        (write("state.pt", data + b"x"), 3, "bytes"),
        (write("extra.bin", b"x"), 3, "extra.bin is present but not listed"),
    ]
    for i, (edit, step, reason) in enumerate(cases):
        d = broken(i, edit)
        with pytest.raises(ValueError, match=reason):
            checkpoint.validate_wave(d, step)
    with pytest.raises(ValueError, match="no such directory"):
        checkpoint.validate_wave(str(tmp_path / "absent"), 3)

    # committed_waves lists only directories that validate
    _wv_commit(out, 4)
    _wv_commit(out, 5)
    slot_dir = os.path.join(out, "waves", "t", "r1p0")
    os.unlink(os.path.join(slot_dir, "step-00000004", "wave.json"))
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [3, 5]


def test_received_wave_commits_only_when_every_hash_matches(tmp_path):
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: commit_received_wave commits a file whose sha256 or size differs from the
    # manifest, a manifest of another version or a non-int step, a staged directory holding an
    # unlisted or missing file, or replaces a committed wave of the same step with another
    # manifest; or it rewrites a committed wave whose manifest is equal
    out = str(tmp_path / "out")
    data = _c1_wave(4)
    m = _wv_manifest(data, 4)
    wave_root = os.path.join(out, "waves")
    flipped = bytearray(data)
    flipped[-1] ^= 1
    refused = [
        (_wv_stage(tmp_path, "s1", bytes(flipped)), m, "sha256"),
        (_wv_stage(tmp_path, "s2", data + b"x"), m, "bytes"),
        (_wv_stage(tmp_path, "s3", data, {"junk": b"x"}), m, "junk is staged but not listed"),
        (_wv_stage(tmp_path, "s4", data), {**m, "v": 2}, "version 2"),
        (_wv_stage(tmp_path, "s5", data), {**m, "step": True}, "bad step"),
        (_wv_stage(tmp_path, "s6", data), {**m, "step": "4"}, "bad step"),
        (
            _wv_stage(tmp_path, "s7", data),
            {**m, "files": m["files"] + [{**m["files"][0], "name": "opt.pt"}]},
            "opt.pt is missing",
        ),
    ]
    for staged, man, reason in refused:
        with pytest.raises(ValueError, match=reason):
            checkpoint.commit_received_wave(out, "t", 1, 0, man, staged)
        assert checkpoint.committed_waves(out, "t", 1, 0) == []
        assert not os.path.exists(engine_shard_path(out, "t", 1, 0))
    # no temp directory survives a refused commit
    assert all(n == [] for _, _, n in os.walk(wave_root))

    staged = _wv_stage(tmp_path, "ok", data)
    d, committed = checkpoint.commit_received_wave(out, "t", 1, 0, m, staged)
    assert committed is True
    assert d == checkpoint.wave_dir(out, "t", 1, 0, 4)
    assert checkpoint.validate_wave(d, 4) == m
    assert _c1_bytes(engine_shard_path(out, "t", 1, 0)) == data
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(engine_shard_path(out, "t", 1, 0))) == 4

    before = _wv_tree(d)
    again = _wv_stage(tmp_path, "again", data)
    assert checkpoint.commit_received_wave(out, "t", 1, 0, m, again) == (d, False)
    assert _wv_tree(d) == before
    other = _wv_stage(tmp_path, "other", _c1_wave(9))
    with pytest.raises(ValueError, match="another manifest"):
        checkpoint.commit_received_wave(out, "t", 1, 0, {**_wv_manifest(_c1_wave(9), 4)}, other)
    assert _wv_tree(d) == before


def test_alias_follows_the_newest_wave(tmp_path, monkeypatch):
    # covers: CONTROL-I10
    # WOULD-FAIL-IF: the alias stays on an older wave, is not a hard link to the newest wave's
    # state.pt, its sidecar names another step than wave.json, or the copy fallback on EXDEV is
    # missing (a cross-device out_dir then fails the commit)
    import errno

    out = str(tmp_path / "out")
    alias = engine_shard_path(out, "t", 1, 0)
    for step in (1, 2, 3):
        d, _ = _wv_commit(out, step)
        assert os.path.samefile(alias, os.path.join(d, "state.pt"))
        assert _c1_bytes(alias) == _c1_wave(step)
        assert checkpoint.read_sidecar(checkpoint.sidecar_of(alias)) == step
        checkpoint.check_resume_step(checkpoint.sidecar_of(alias), step)

    def cross_device(src, dst, *a, **kw):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", cross_device)
    d, _ = _wv_commit(out, 4)
    assert not os.path.samefile(alias, os.path.join(d, "state.pt"))
    assert _c1_bytes(alias) == _c1_wave(4)
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(alias)) == 4


def test_commit_keeps_the_newest_three_waves(tmp_path):
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: commit_wave keeps more or fewer than the newest three waves, prunes a wave it
    # just committed (a restored wave older than the slot's newest), counts waves newer than the
    # committed step in the keep window (a later commit of step 3 prunes the restored step 2,
    # or a commit removes the abandoned 5 and 6), or drop_waves_after keeps a
    # wave newer than its step or removes one at or below it
    out = str(tmp_path / "out")
    for step in range(1, 7):
        _wv_commit(out, step)
        want = list(range(max(1, step - 2), step + 1))
        assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == want
    # another slot is never touched
    _wv_commit(out, 1, slot=(0, 0))
    staged = _wv_stage(tmp_path, "old", _c1_wave(2))
    d, committed = checkpoint.commit_received_wave(
        out, "t", 1, 0, _wv_manifest(_c1_wave(2), 2), staged
    )
    assert committed is True
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [2, 4, 5, 6]
    # a later commit below the slot's newest, with no drop in between, keeps the restored wave:
    # the window counts only waves at or below the committed step (SPEC-control §2.3)
    assert _wv_commit(out, 3)[1] is True
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [2, 3, 4, 5, 6]
    assert checkpoint.drop_waves_after(out, "t", 1, 0, 4) == [5, 6]
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [2, 3, 4]
    assert checkpoint.drop_waves_after(out, "t", 1, 0, 4) == []
    # back in the ascending case the window is exactly the newest three
    _wv_commit(out, 5)
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [3, 4, 5]
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 0, 0)] == [1]
    with pytest.raises(ValueError):
        _wv_commit(out, 8, keep=0)


def test_wave_cap_refuses_a_descending_flood(tmp_path):
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: a peer shipping descending steps grows a slot past keep + MAX_WAVES_AHEAD
    # (no cap in commit_received_wave or commit_wave), the cap counts only one of its two bounds,
    # a refusal writes or removes anything (a temp directory survives, a wave is pruned, the alias
    # moves), or the cap refuses a re-commit of a step the slot already holds
    out = str(tmp_path / "out")
    slot = os.path.join(out, "waves", "t", "r1p0")
    alias = engine_shard_path(out, "t", 1, 0)

    def recv(step, name):
        data = _c1_wave(step)
        staged = _wv_stage(tmp_path, name, data)
        return checkpoint.commit_received_wave(out, "t", 1, 0, _wv_manifest(data, step), staged)

    assert checkpoint.MAX_WAVES_AHEAD == 4
    for step in (20, 19, 18, 17):
        assert recv(step, f"d{step}")[1] is True
    held = [17, 18, 19, 20]
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == held
    before = _wv_tree(out)
    with pytest.raises(ValueError, match=r"r1p0 step 16.*holds 4 waves, 4 above step 16"):
        recv(16, "d16")
    with pytest.raises(ValueError, match=r"holds 4 waves, 4 above step 16"):
        _wv_commit(out, 16)
    assert _wv_tree(out) == before  # nothing written, nothing removed, alias unmoved
    assert sorted(os.listdir(slot)) == [f"step-{s:08d}" for s in held]
    # a held step is never refused by the cap, even with 3 waves above it
    assert recv(17, "again17")[1] is False
    # a step above every held one is not refused by the "above" bound
    assert _wv_commit(out, 21)[1] is True
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [19, 20, 21]

    # the total bound: keep + MAX_WAVES_AHEAD step directories refuse a new step with none above
    out2 = str(tmp_path / "out2")
    for step in range(1, 8):
        _wv_commit(out2, step, keep=7)
    before = _wv_tree(out2)
    with pytest.raises(ValueError, match=r"holds 7 waves, 0 above step 8"):
        _wv_commit(out2, 8, keep=3)
    assert _wv_tree(out2) == before
    # a re-commit of a held step with 6 waves above it is not refused
    assert _wv_commit(out2, 1, keep=7) == (checkpoint.wave_dir(out2, "t", 1, 0, 1), False)
    assert _wv_tree(out2) == before
    assert os.path.exists(alias)


def test_committed_waves_skips_a_wave_pruned_while_it_is_read(tmp_path, monkeypatch):
    # covers: CONTROL-I8
    # WOULD-FAIL-IF: validate_wave lets a FileNotFoundError out when a concurrent commit prunes the
    # directory it is reading, so committed_waves raises instead of returning the waves that remain
    out = str(tmp_path / "out")
    for step in range(1, 4):
        _wv_commit(out, step)
    real = checkpoint._read_manifest
    fired = []

    def racing(d, step):
        m = real(d, step)
        if step == 1 and not fired:
            fired.append(1)
            _wv_commit(out, 4)  # keep=3 prunes step 1 between its manifest and its files
        return m

    monkeypatch.setattr(checkpoint, "_read_manifest", racing)
    # the listing predates step 4; step 1 vanished while read and is skipped, not raised
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [2, 3]
    assert fired == [1]
    monkeypatch.setattr(checkpoint, "_read_manifest", real)
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [2, 3, 4]
    with pytest.raises(ValueError, match="no such directory"):
        checkpoint.validate_wave(checkpoint.wave_dir(out, "t", 1, 0, 1), 1)


def test_wave_files_never_match_the_oracle_glob(tmp_path):
    # covers: CONTROL-I10
    # WOULD-FAIL-IF: a wave file, a wave temp directory or a link temp lands where the oracle's
    # out_dir/*_ckpt_rank*p*.pt glob (tests/helpers/exact.py) reads it as a slot's shard
    import glob

    out = str(tmp_path / "out")
    for slot in ((0, 0), (1, 0)):
        for step in (1, 2):
            _wv_commit(out, step, slot)

    def dies(path):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("killed mid-save")

    with pytest.raises(RuntimeError):
        checkpoint.commit_wave(out, "t", 0, 0, 3, dies)
    staged = _wv_stage(tmp_path, "rx", _c1_wave(5, (1, 0)))
    checkpoint.commit_received_wave(out, "t", 1, 0, _wv_manifest(_c1_wave(5, (1, 0)), 5), staged)
    hits = sorted(glob.glob(os.path.join(out, "*_ckpt_rank*p*.pt")))
    assert hits == [engine_shard_path(out, "t", 0, 0), engine_shard_path(out, "t", 1, 0)]
    wave_files = [
        os.path.join(d, n) for d, _, ns in os.walk(os.path.join(out, "waves")) for n in ns
    ]
    assert wave_files and not set(wave_files) & set(hits)
    leftovers = glob.glob(os.path.join(out, "**", "*"), recursive=True)
    assert not [p for p in leftovers if "tmp" in os.path.basename(p)], leftovers


def test_alias_writers_never_write_into_a_wave(tmp_path):
    # covers: CONTROL-I10
    # WOULD-FAIL-IF: an alias writer opens the alias (a hard link to the wave's state.pt) for
    # writing in place instead of replacing the directory entry: save_shard writing through a
    # stale temp that is itself a hard link to the wave, export_wave copying onto the alias, or
    # link_legacy linking or copying over the alias directly (BA-46)
    out = str(tmp_path / "out")
    d, _ = _wv_commit(out, 3)
    state = os.path.join(d, "state.pt")
    want = sha256_file(state)
    alias = engine_shard_path(out, "t", 1, 0)
    assert os.path.samefile(alias, state)

    def unchanged():
        assert sha256_file(state) == want
        checkpoint.validate_wave(d, 3)

    checkpoint.save_shard(alias, 9, _c1_saver(_c1_wave(9)))
    unchanged()
    assert _c1_bytes(alias) == _c1_wave(9)

    checkpoint.link_legacy(d, out, "t", 1, 0)
    os.link(state, alias + ".tmp")  # a crash left the link temp behind
    checkpoint.save_shard(alias, 10, _c1_saver(_c1_wave(10)))
    unchanged()

    checkpoint.link_legacy(d, out, "t", 1, 0)
    other = str(tmp_path / "other.pt")
    _c1_saver(_c1_wave(11))(other)
    os.replace(other, alias)  # the agent's RESTORE
    unchanged()

    checkpoint.link_legacy(d, out, "t", 1, 0)
    store = CheckpointStore(str(tmp_path / "store"))
    _st_wave(store, (1, 0), 12)
    store.export_wave(12, [(1, 0)], out, "t")
    unchanged()
    assert _c1_bytes(alias) == _c1_wave(12, (1, 0))

    checkpoint.link_legacy(d, out, "t", 1, 0)
    checkpoint.link_legacy(d, out, "t", 1, 0)
    unchanged()
    assert os.path.samefile(alias, state)


# ---- I-15 / BA-460: a wave lists extra files beside state.pt (hf_buffers.json) ----

# Today's dense manifest of b"dense-golden-state-pt" at step 3, byte for byte.
_WV_DENSE_GOLDEN = (
    b'{"v": 1, "step": 3, "files": [{"name": "state.pt", "sha256": '
    b'"0126f3338dd7af16df9d008f01963f2657581cf5fa166867399cd4b63df8edb5", "bytes": 21}]}'
)
_WV_EXTRAS = {"zz-code.txt": b"code-identity-9", "hf_buffers.json": b'{"rope.inv_freq": "ab12"}'}


def _wv_entry(name, data):
    return {"name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _wv_extra_commit(out, step, extras, save=None):
    save = _c1_saver(_c1_wave(step)) if save is None else save
    return checkpoint.commit_wave(out, "t", 1, 0, step, save, extra_files=extras)


def test_wave_extra_dense_manifest_bytes_are_unchanged(tmp_path):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: a dense manifest's bytes changed: extra_files None or {} adds a key, an
    # empty extras list, another key order or other separators to wave.json, so every dense
    # wave a resume or the store compares by bytes no longer matches today's
    for i, extras in enumerate((None, {})):
        out = str(tmp_path / f"out{i}")
        d, committed = _wv_extra_commit(out, 3, extras, _c1_saver(b"dense-golden-state-pt"))
        assert committed is True
        assert _c1_bytes(os.path.join(d, "wave.json")) == _WV_DENSE_GOLDEN
        assert sorted(os.listdir(d)) == ["state.pt", "wave.json"]
    # the default argument writes the same bytes
    d, _ = checkpoint.commit_wave(
        str(tmp_path / "out2"), "t", 1, 0, 3, _c1_saver(b"dense-golden-state-pt")
    )
    assert _c1_bytes(os.path.join(d, "wave.json")) == _WV_DENSE_GOLDEN


def test_wave_extra_is_written_before_wave_json_and_listed_after_state(tmp_path, monkeypatch):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: an extra not listed (the wave then fails validation as an unlisted file),
    # listed before state.pt or in insertion order rather than by name, listed with another
    # sha256 or size than its bytes, or written after wave.json or into the final directory
    out = str(tmp_path / "out")
    final = checkpoint.wave_dir(out, "t", 1, 0, 4)
    seen = []
    real_rename = os.rename

    def rename(src, dst, *a, **kw):
        if os.path.normpath(str(dst)) == os.path.normpath(final):
            seen.append(sorted(os.listdir(src)))
            for name, data in _WV_EXTRAS.items():
                assert _c1_bytes(os.path.join(src, name)) == data
        return real_rename(src, dst, *a, **kw)

    monkeypatch.setattr(os, "rename", rename)
    d, committed = _wv_extra_commit(out, 4, dict(_WV_EXTRAS))
    monkeypatch.setattr(os, "rename", real_rename)
    assert (d, committed) == (final, True)
    assert seen == [["hf_buffers.json", "state.pt", "wave.json", "zz-code.txt"]]
    want = {
        "v": 1,
        "step": 4,
        "files": [
            _wv_entry("state.pt", _c1_wave(4)),
            _wv_entry("hf_buffers.json", _WV_EXTRAS["hf_buffers.json"]),
            _wv_entry("zz-code.txt", _WV_EXTRAS["zz-code.txt"]),
        ],
    }
    with open(os.path.join(d, "wave.json")) as f:
        assert json.load(f) == want
    assert checkpoint.validate_wave(d, 4) == want
    for name, data in _WV_EXTRAS.items():
        assert _c1_bytes(os.path.join(d, name)) == data
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [4]
    alias = engine_shard_path(out, "t", 1, 0)
    assert os.path.samefile(alias, os.path.join(d, "state.pt"))
    assert checkpoint.read_sidecar(checkpoint.sidecar_of(alias)) == 4


def test_wave_extra_bad_name_is_refused_before_anything_is_written(tmp_path):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: an extra whose name is empty, '.', '..', wave.json or state.pt, or holds
    # '/', '\\' or NUL is accepted (it escapes the wave directory or shadows the wave's own
    # files), refused with another exception, refused without naming it, or refused only after
    # save_fn ran or a directory was created; or a name equal to another (or to the wave's own
    # files) under case folding, or over 255 bytes in UTF-8, is accepted
    def never(path):
        raise AssertionError("save_fn called before the extra names were checked")

    bad = ["", ".", "..", "wave.json", "state.pt", "../x", "a/b", "a\\b", "a\0b", "/abs"]
    for i, name in enumerate(bad):
        out = str(tmp_path / f"out{i}")
        extras = {"hf_buffers.json": b"{}", name: b"x"}
        with pytest.raises(ValueError, match=re.escape(repr(name))):
            _wv_extra_commit(out, 2, extras, never)
        assert not os.path.exists(out), name
    out = str(tmp_path / "typed")
    with pytest.raises(ValueError, match="hf_buffers.json: content is not bytes"):
        _wv_extra_commit(out, 2, {"hf_buffers.json": "{}"}, never)
    assert not os.path.exists(out)
    # a case-insensitive file system (APFS, NTFS) resolves 'STATE.PT' to the wave's own
    # file and two names differing only in case to one file: refused before save_fn runs
    for i, extras in enumerate(
        (
            {"STATE.PT": b"x"},
            {"Wave.Json": b"x"},
            {"hf_buffers.json": b"{}", "HF_BUFFERS.json": b"y"},
            {"\u00e9" * 128: b"x"},  # 128 characters, 256 bytes in UTF-8
        )
    ):
        out = str(tmp_path / f"fold{i}")
        name = sorted(extras)[-1] if len(extras) > 1 else next(iter(extras))
        with pytest.raises(ValueError, match="is not a wave file name"):
            _wv_extra_commit(out, 2, extras, never)
        assert not os.path.exists(out), name
    # a name refused even when the slot already holds the step: checked first
    held = str(tmp_path / "held")
    _wv_extra_commit(held, 2, None)
    with pytest.raises(ValueError, match=re.escape(repr("state.pt"))):
        _wv_extra_commit(held, 2, {"state.pt": b"x"}, never)


def test_wave_extra_held_wave_with_other_extras_is_refused(tmp_path):
    # covers: CONTROL-I23
    # WOULD-FAIL-IF: a held wave with different extras accepted: a second commit of a step whose
    # wave lists another hf_buffers.json (other bytes), lacks an extra the commit carries, or
    # carries one the commit lacks returns committed=False, so an old buffer file is resumed
    # beside a new commit's state; or an equal one is rewritten
    out = str(tmp_path / "out")
    d, _ = _wv_extra_commit(out, 5, dict(_WV_EXTRAS))
    before = _wv_tree(out)

    def never(path):
        raise AssertionError("save_fn called for a committed step")

    assert _wv_extra_commit(out, 5, dict(reversed(list(_WV_EXTRAS.items()))), never) == (d, False)
    assert _wv_tree(out) == before
    cases = [
        ({**_WV_EXTRAS, "hf_buffers.json": b'{"rope.inv_freq": "cd34"}'}, "hf_buffers.json"),
        ({"zz-code.txt": _WV_EXTRAS["zz-code.txt"]}, "hf_buffers.json"),
        ({**_WV_EXTRAS, "moe.json": b"[]"}, "moe.json"),
        (None, "hf_buffers.json"),
    ]
    for extras, name in cases:
        with pytest.raises(ValueError, match=f"extra file {re.escape(name)} "):
            _wv_extra_commit(out, 5, extras, never)
        assert _wv_tree(out) == before
    # a held dense wave refuses a commit of the same step that carries an extra
    _wv_extra_commit(out, 6, None)
    dense = _wv_tree(out)
    with pytest.raises(ValueError, match="extra file hf_buffers.json "):
        _wv_extra_commit(out, 6, {"hf_buffers.json": b"{}"}, never)
    assert _wv_tree(out) == dense


def test_wave_extra_edited_deleted_or_unlisted_is_refused(tmp_path):
    # covers: CONTROL-I23, CONTROL-I10
    # WOULD-FAIL-IF: an edited extra accepted: validate_wave or committed_waves (what a resume
    # reads) accept a wave whose listed extra was edited after the commit (same size or not),
    # deleted or replaced by a symlink, or that holds a file it does not list
    import shutil

    out = str(tmp_path / "out")
    good, _ = _wv_extra_commit(out, 3, dict(_WV_EXTRAS))
    buf = _WV_EXTRAS["hf_buffers.json"]
    flipped = bytearray(buf)
    flipped[0] ^= 1

    def write(name, b):
        return lambda d: open(os.path.join(d, name), "wb").write(b)

    def symlink(d):
        os.unlink(os.path.join(d, "hf_buffers.json"))
        os.symlink(os.path.join(good, "hf_buffers.json"), os.path.join(d, "hf_buffers.json"))

    cases = [
        (write("hf_buffers.json", bytes(flipped)), "hf_buffers.json has sha256"),
        (write("zz-code.txt", _WV_EXTRAS["zz-code.txt"] + b"!"), "zz-code.txt has 16 bytes"),
        (lambda d: os.unlink(os.path.join(d, "hf_buffers.json")), "hf_buffers.json is missing"),
        (symlink, "hf_buffers.json is not a regular file"),
        (write("hf_buffers.old.json", buf), "hf_buffers.old.json is present but not listed"),
    ]
    for i, (edit, reason) in enumerate(cases):
        d = str(tmp_path / f"bad{i}")
        shutil.copytree(good, d, symlinks=True)
        edit(d)
        with pytest.raises(ValueError, match=re.escape(reason)):
            checkpoint.validate_wave(d, 3)
    _wv_extra_commit(out, 4, dict(_WV_EXTRAS))
    write("hf_buffers.json", bytes(flipped))(checkpoint.wave_dir(out, "t", 1, 0, 3))
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [4]


def test_wave_extra_is_carried_by_the_store_and_validates_after_export(tmp_path):
    # covers: CONTROL-I23, CONTROL-I10
    # WOULD-FAIL-IF: the store or export_wave drops or rewrites an extra (the exported wave then
    # fails validation or lists no hf_buffers.json, and a local resume loses the run's buffers),
    # or a received wave whose staged directory lacks a listed extra is committed
    src_out = str(tmp_path / "src")
    d, _ = _wv_extra_commit(src_out, 7, dict(_WV_EXTRAS))
    m = checkpoint.validate_wave(d, 7)
    store = CheckpointStore(str(tmp_path / "store"))
    for e in m["files"]:
        body = str(tmp_path / f"in-{e['name']}.part")
        _c1_saver(_c1_bytes(os.path.join(d, e["name"])))(body)
        store.stage_part((1, 0), 1, 7, e["name"], e["sha256"], body)
    assert store.commit_staged((1, 0), 1, 7, m) is True
    assert checkpoint.validate_wave(store.wave_path((1, 0), 7), 7) == m
    out = str(tmp_path / "out")
    (alias,) = store.export_wave(7, [(1, 0)], out, "t")
    dst = checkpoint.wave_dir(out, "t", 1, 0, 7)
    assert checkpoint.validate_wave(dst, 7) == m
    for name, data in _WV_EXTRAS.items():
        assert _c1_bytes(os.path.join(dst, name)) == data
    assert _c1_bytes(alias) == _c1_wave(7)
    checkpoint.check_resume_step(checkpoint.sidecar_of(alias), 7)
    # a received wave whose staged directory lacks the listed extra is refused
    staged = _wv_stage(tmp_path, "short", _c1_wave(8), {"zz-code.txt": _WV_EXTRAS["zz-code.txt"]})
    m8 = {
        "v": 1,
        "step": 8,
        "files": [
            _wv_entry("state.pt", _c1_wave(8)),
            _wv_entry("hf_buffers.json", _WV_EXTRAS["hf_buffers.json"]),
            _wv_entry("zz-code.txt", _WV_EXTRAS["zz-code.txt"]),
        ],
    }
    with pytest.raises(ValueError, match="hf_buffers.json is missing"):
        checkpoint.commit_received_wave(out, "t", 1, 0, m8, staged)
    assert [s for s, _ in checkpoint.committed_waves(out, "t", 1, 0)] == [7]


def test_fresh_launch_resets_the_loss_stream_even_when_a_sibling_slot_removed_it(
    tmp_path, monkeypatch
):
    """Every slot process of a spec resets the same shared loss stream at a fresh
    launch; another slot can remove it between a check and the unlink. The reset
    must treat absence as done, keep the stream on a resume, and remove it
    otherwise (found as a FileNotFoundError in an aead_cost run, 2026-10-10)."""
    # covers: CONTROL-I10
    # WOULD-FAIL-IF: the reset checks os.path.exists and then unlinks without tolerating
    # FileNotFoundError, so a slot whose sibling removed the stream first dies at launch
    from wantrain.worker import stage_main

    stream = tmp_path / "t_loss_stream.jsonl"
    stream.write_text('{"step": 0}\n')
    stage_main.reset_loss_stream(str(stream), resume=True)
    assert stream.exists()
    stage_main.reset_loss_stream(str(stream), resume=False)
    assert not stream.exists()
    # The sibling already removed it, yet a check would still report it present.
    monkeypatch.setattr(stage_main.os.path, "exists", lambda p: True)
    stage_main.reset_loss_stream(str(stream), resume=False)
    assert not stream.exists()
