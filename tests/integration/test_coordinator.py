"""End-to-end control plane: real agents, real stages, real sockets.

Node lifecycle: start -> register (measured resources) -> available ->
assigned -> executes -> reports progress and exit -> completes -> idle again.
Scheduler: N workers -> discovery -> registration -> selection -> routing ->
placement -> launch -> monitoring -> completion, with the loss sequence of a
CPU fleet diffed bit for bit against the single-device reference.
"""

# WOULD-FAIL-IF: the coordinator's register -> select -> route -> place -> profile -> schedule -> launch -> monitor path drops a node, ignores declared capacity, link measurements or profiled costs, or a fleet run's losses differ by one bit from the single-device reference

import json
import math
import os
import re
import statistics
import time

import pytest
from conftest import GPU_OFF, gpu_allowed, mps_available
from helpers.cluster import LocalCluster
from helpers.exact import fleet_manifest, reference_manifest_for

from wantrain.coordinator.coordinator import Coordinator, CoordinatorConfig, Job, Worker, _Run
from wantrain.fault_tolerance.liveness import Verdict
from wantrain.launch.local import ensure_reference
from wantrain.model.gpt import make_config
from wantrain.resources.node import NodeInfo
from wantrain.runtime.spec import JobSpec
from wantrain.scheduler.zero_bubble import replan
from wantrain.validation.compare import compare
from wantrain.validation.state import diff_manifests

pytestmark = pytest.mark.slow


def _recipe(**kw):
    base = dict(
        model=make_config("toy"),
        steps=6,
        microbatches=8,
        recompute=True,
        wire_codec="planes",
    )
    base.update(kw)
    return JobSpec(**base)


def _assert_bitwise(res, recipe, tmp_path):
    assert res.ok, res.reason
    ref = ensure_reference(JobSpec.from_json(recipe.to_json()), str(tmp_path / "ref"))
    got = os.path.join(res.run_dir, "losses.json")
    r = compare(ref, got)
    assert r["ok"] and r["compared"] == recipe.steps * recipe.microbatches, r


def test_node_lifecycle_and_two_stage_run(tmp_path, port_base):
    # covers: CONTROL-I1
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        c.add_worker("a")
        c.add_worker("b")
        c.wait_registered(2)
        infos = [w.info for w in c.coord.workers.values()]
        assert all(
            i.tflops_fp32 > 0 and i.memory_bytes() > 0 and i.device == "cpu" for i in infos
        )  # measured, not declared
        recipe = _recipe(stages=2)
        res = c.run(Job(spec=recipe))
        _assert_bitwise(res, recipe, tmp_path)
        msgs = [e[2] for e in res.events]
        assert any("registered a" in m for m in msgs) and any(
            "launching attempt 1" in m for m in msgs
        )
        assert any("complete" in m for m in msgs)
        assert all(w.run_id is None for w in c.coord.workers.values())  # idle again
        # a second job reuses the same registered workers
        res2 = c.run(Job(spec=_recipe(stages=2, steps=2)))
        assert res2.ok and res2.run_id != res.run_id
    finally:
        c.close()


def test_scheduler_cuts_layers_by_declared_heterogeneous_capacity(tmp_path, port_base):
    """Three nodes declaring 4:2:1 compute and different memory: the fast node
    gets the most layers, the largest memory gets the hungriest rank, and the
    run is still exactly the reference (the recipe is partition-independent)."""
    # covers: CONTROL-I2
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        c.add_worker("fast", declare={"tflops": 4.0, "vram_bytes": 4 << 30, "gpu_name": "sim-A"})
        c.add_worker("mid", declare={"tflops": 2.0, "vram_bytes": 16 << 30, "gpu_name": "sim-B"})
        c.add_worker("slow", declare={"tflops": 1.0, "vram_bytes": 8 << 30, "gpu_name": "sim-C"})
        c.wait_registered(3)
        recipe = _recipe(stages=3, model=make_config("toy"))
        recipe.model.n_layer = 12
        res = c.run(Job(spec=recipe))
        _assert_bitwise(res, recipe, tmp_path)
        rat = res.rationale
        placed = [row[0].split("(")[0] for row in rat["placement"]]
        layers = dict(zip(placed, rat["layers_per_stage"]))
        assert layers["fast"] > layers["mid"] > layers["slow"], rat
        assert sorted(rat["stage_tflops"]) == [1.0, 2.0, 4.0]
        assert "heterogeneous fleet" in rat["selection"]
    finally:
        c.close()


WAN = {"intra": {"mbps": 400, "rtt_ms": 2}, "inter": {"mbps": 60, "rtt_ms": 60}}


def test_wan_aware_routing_crosses_the_slow_link_once(tmp_path, port_base):
    """Two emulated regions joined by a slow, high-latency link. The probe
    matrix and the per-hop measurements are real transfers shaped by the
    profile; the chain must cross between regions exactly once, the WAN shim
    must carry the same profile into training, and the result stays exact."""
    # covers: CONTROL-I3
    c = LocalCluster(
        tmp_path, port_base, probe=True, probe_bulk_bytes=1 << 20, measure_bulk_bytes=1 << 20
    )
    try:
        for name, region in (("eu1", "eu"), ("us1", "us"), ("eu2", "eu"), ("us2", "us")):
            c.add_worker(name, region=region, wan_profile=WAN)
        c.wait_registered(4)
        c.coord.cfg.selection.same_region = False  # the job spans both regions
        recipe = _recipe(stages=4, steps=4)
        recipe.model.n_layer = 6
        res = c.run(Job(spec=recipe))
        _assert_bitwise(res, recipe, tmp_path)
        names = [row[0].split("(")[0] for row in res.rationale["placement"]]
        regions = [n[:2] for n in names]
        crossings = sum(a != b for a, b in zip(regions, regions[1:]))
        assert crossings == 1, names
        hops = res.rationale["hops"]
        slow = [h for h in hops if h["rtt_ms"] > 30]
        assert len(slow) == 1 and slow[0]["mbps"] < 100
        # Profiled F/B/W per stage + the measured hops -> a zero-bubble schedule
        sched = res.rationale["schedule"]
        assert sched["source"] == "profiled", sched
        # a step cannot be shorter than one round trip across the slow hop
        assert sched["step_s"] * 1e3 >= 2 * max(sched["comm_ms"])
        assert max(sched["comm_ms"]) > 20  # the WAN hop is in the model
        links = res.rationale["emulated_links"]
        assert sum(1 for lk in links[:-1] if lk["bw_mbps"] == 60) == 1
        assert len(hops) == 3 and all(h["duplex"] == "per_direction" for h in hops)
        # Link memory is keyed by machine address; every worker here shares one
        # address, so nothing is remembered (a same-host pair says nothing about
        # a WAN path). The unit tests cover the memory itself.
        assert c.coord.memory.paths() == {}
    finally:
        c.close()


def test_planner_mode_solves_the_job_from_the_measured_fleet(tmp_path, port_base):
    # covers: CONTROL-I4
    c = LocalCluster(
        tmp_path, port_base, probe=True, probe_bulk_bytes=1 << 20, measure_bulk_bytes=1 << 20
    )
    try:
        c.add_worker("p0")
        c.add_worker("p1")
        c.wait_registered(2)
        res = c.run(
            Job(
                plan={
                    "model": "toy",
                    "steps": 3,
                    "device": "cpu",
                    "mb_size": 4,
                    "bubble_target": 0.05,
                },
                stages=2,
            )
        )
        assert res.ok, res.reason
        rat = res.rationale
        assert rat["precision"]["boundary"] in ("fp32", "bf16")
        assert rat["microbatches"] >= 1 and rat["k_inflight"] >= 1
        spec = JobSpec.load(os.path.join(res.run_dir, "spec.json"))
        assert spec.recompute and spec.schedule_order
        with open(os.path.join(res.run_dir, "losses.json")) as f:
            losses = json.load(f)["losses"]
        assert len(losses) == 3 * spec.microbatches
        assert all(math.isfinite(x["loss"]) for x in losses)
    finally:
        c.close()


@pytest.mark.mps
@pytest.mark.skipif(
    not mps_available(), reason="needs an Apple MPS device" if gpu_allowed() else GPU_OFF
)
def test_real_heterogeneous_fleet_cpu_and_gpu(tmp_path, port_base):
    """Genuinely different hardware on one host: one stage on the CPU, one on
    the Apple GPU. CONTROL-I5 is SUSPENDED by CONTROL-I25 (the human,
    2026-10-11) until the heterogeneous track returns: such a fleet spans two
    device classes, so the run is refused MIXED-FLEET and nothing launches."""
    # covers: CONTROL-I5, CONTROL-I25
    # WOULD-FAIL-IF: a CPU + MPS fleet (two device classes reported in HELLO) is selected
    # and launched instead of refused MIXED-FLEET naming both classes
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        c.add_worker("cpu0", device="cpu")
        c.add_worker("gpu0", device="mps")
        c.wait_registered(2)
        infos = sorted((w.info for w in c.coord.workers.values()), key=lambda i: i.device)
        assert [i.device for i in infos] == ["cpu", "mps"]
        classes = [i.device_class for i in infos]
        assert classes[0].startswith("cpu:") and classes[1].startswith("mps:")
        res = c.run(Job(spec=_recipe(stages=2, steps=8)))
        assert not res.ok
        assert "MIXED-FLEET: no single device class can field 2 nodes" in res.reason, res.reason
        assert all(f"{cls} x1" in res.reason for cls in classes), res.reason
        assert not os.path.exists(os.path.join(res.run_dir, "spec.json"))
    finally:
        c.close()


def test_wide_stage_with_weighted_ownership_via_the_coordinator(tmp_path, port_base):
    """A stage of two peers (ZeRO-1 sharded optimizer, microbatch ownership
    weighted by each peer's profiled throughput) followed by a single-peer
    stage, placed and launched by the coordinator: still exactly the reference."""
    # covers: CONTROL-I6
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(3):
            c.add_worker(f"w{i}")
        c.wait_registered(3)
        recipe = _recipe(stages=2, stage_widths=[2, 1], wire_codec="raw")
        res = c.run(Job(spec=recipe))
        _assert_bitwise(res, recipe, tmp_path)
        spec = JobSpec.load(os.path.join(res.run_dir, "spec.json"))
        assert [len(row) for row in spec.endpoints] == [2, 1]
        assert sorted(set(spec.mb_owner[0])) == [0, 1]  # both peers own work
        assert len(spec.schedule_order[0]) == 2  # one order per peer
    finally:
        c.close()


def _ck_key_id(workdir):
    """The peer id of the key an agent keeps in its workdir (CONTROL-I22)."""
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    from wantrain.transport import identity

    path = os.path.join(workdir, "peer_key")
    assert os.path.exists(path), path
    key = identity.load_or_create_key(path)
    return identity.peer_id(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def _ck_spec(c, name, run_id, attempt, slot):
    """The spec the agent `name` was handed for `slot` at `attempt`."""
    path = os.path.join(
        c.workdirs[name], "runs", run_id, f"spec-a{attempt}-r{slot[0]}p{slot[1]}.json"
    )
    with open(path) as f:
        return json.load(f)


def _ck_holder(c, run_id, attempt, slot):
    """The one agent that was handed `slot` at `attempt` (after the run, when
    the coordinator no longer holds it)."""
    names = [
        name
        for name, wd in c.workdirs.items()
        if os.path.exists(
            os.path.join(wd, "runs", run_id, f"spec-a{attempt}-r{slot[0]}p{slot[1]}.json")
        )
    ]
    assert len(names) == 1, (slot, names)
    return names[0]


def test_launched_spec_pins_each_slots_key(tmp_path, port_base):
    """Each slot of a launched spec is pinned to the key its own agent proved
    in its control handshake: two agents, two distinct ids, slot_peers[r][0]
    = the id of the agent holding (r, 0), trusted_peers = their sorted set,
    in the stored spec and in what every agent received; the run is exact."""
    # covers: CONTROL-I22
    # WOULD-FAIL-IF: the first launch leaves slot_peers or trusted_peers unset, pins one host-wide
    # id for every slot, pins a slot to another agent's key or ids the handshake did not prove,
    # or the agents are handed a spec other than the stored one
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        c.add_worker("a")
        c.add_worker("b")
        c.wait_registered(2)
        recipe = _recipe(stages=2, steps=2)
        res = c.run(Job(spec=recipe))
        _assert_bitwise(res, recipe, tmp_path)
        ids = {name: _ck_key_id(c.workdirs[name]) for name in ("a", "b")}
        assert ids["a"] != ids["b"]
        assert sorted(w.peer_id for w in c.coord.workers.values()) == sorted(ids.values())
        holder = {slot: _ck_holder(c, res.run_id, 1, slot) for slot in ((0, 0), (1, 0))}
        assert sorted(holder.values()) == ["a", "b"]
        want = [[ids[holder[(0, 0)]]], [ids[holder[(1, 0)]]]]
        spec = JobSpec.load(os.path.join(res.run_dir, "spec.json"))
        assert spec.slot_peers == want
        assert spec.trusted_peers == sorted(ids.values())
        for slot, name in holder.items():
            got = _ck_spec(c, name, res.run_id, 1, slot)
            assert got["slot_peers"] == want and got["trusted_peers"] == sorted(ids.values())
    finally:
        c.close()


def test_stage_gets_its_agents_key(tmp_path, port_base, monkeypatch):
    """The agent hands its stage its own key (<workdir>/peer_key), overriding
    the key path the agent inherited: across the chain, each stage proves the
    key of the agent that runs it, the one the coordinator pinned."""
    # covers: CONTROL-I22
    # WOULD-FAIL-IF: the agent's _env keeps the inherited WANTRAIN_PEER_KEY (every stage on the
    # host proves one shared key: the pinned trusted_peers refuse the chain, or each stage's
    # p2p_auth names that shared id instead of its neighbour agent's key)
    from wantrain.worker.agent import WorkerAgent

    inherited = str(tmp_path / "host-wide" / "peer_key")
    monkeypatch.setenv("WANTRAIN_PEER_KEY", inherited)
    agent = WorkerAgent("127.0.0.1:1", "127.0.0.1", 1, workdir=str(tmp_path / "solo"))
    assert agent._env()["WANTRAIN_PEER_KEY"] == str(tmp_path / "solo" / "peer_key")
    assert os.environ["WANTRAIN_PEER_KEY"] == inherited  # the agent's own env is untouched

    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        c.add_worker("a")
        c.add_worker("b")
        c.wait_registered(2)
        recipe = _recipe(stages=2, steps=2)
        res = c.run(Job(spec=recipe))
        _assert_bitwise(res, recipe, tmp_path)
        assert not os.path.exists(inherited)  # no stage ever loaded the inherited key
        holder = {r: _ck_holder(c, res.run_id, 1, (r, 0)) for r in (0, 1)}
        tag = JobSpec.load(os.path.join(res.run_dir, "spec.json")).tag
        for r in (0, 1):
            out = os.path.join(c.workdirs[holder[r]], "runs", res.run_id, "out")
            events = _c2_lines(os.path.join(out, f"{tag}_rank{r}p0_telemetry.jsonl"))
            auth = [e for e in events if e["ev"] == "p2p_auth"]
            assert len(auth) == 1, auth
            assert auth[0]["peer"] == _ck_key_id(c.workdirs[holder[1 - r]]), (r, auth)
    finally:
        c.close()


def test_heal_repins_the_replacement_key(tmp_path, port_base, monkeypatch):
    """A healed slot is re-pinned to its replacement's key: the relaunch spec
    every slot receives names the replacement's id at the healed slot and the
    survivor's at its own, and the dead node's id is gone from slot_peers and
    from trusted_peers. The healed run stays the reference."""
    # covers: CONTROL-I22
    # WOULD-FAIL-IF: heal relaunches under the stored spec's slot_peers or trusted_peers (the dead
    # node's key still trusted, the replacement's refused on the chain), or pins the replacement's
    # key at a slot other than the one it took
    monkeypatch.setenv("WANTRAIN_SIM_EXIT_DELAY_S", "10")
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(3):  # 2 slots + 1 spare
            c.add_worker(f"w{i}")
        c.wait_registered(3)
        recipe = _recipe(stages=2, steps=8, ckpt_every=2)
        c.run_async(Job(spec=recipe))
        deadline = time.time() + 300
        while True:
            run = c.coord.run
            if run is not None and (run.store.common_step(list(run.slots)) or -1) >= 3:
                break
            assert time.time() < deadline, "the coordinator never held a common wave >= 3"
            time.sleep(0.05)
        victim, survivor = c.name_of((0, 0)), c.name_of((1, 0))
        c.kill_worker(victim)
        res = c.join()
        _assert_bitwise(res, recipe, tmp_path)
        assert len(res.heals) == 1 and res.heals[0]["from"] == victim, res.heals
        replacement = res.heals[0]["to"]
        ids = {name: _ck_key_id(c.workdirs[name]) for name in (victim, survivor, replacement)}
        assert len(set(ids.values())) == 3
        first = _ck_spec(c, survivor, res.run_id, 1, (1, 0))
        assert first["slot_peers"] == [[ids[victim]], [ids[survivor]]]
        want = [[ids[replacement]], [ids[survivor]]]
        for name, slot in ((replacement, (0, 0)), (survivor, (1, 0))):
            got = _ck_spec(c, name, res.run_id, 2, slot)
            assert got["resume"] is True and got["slot_peers"] == want, (name, got)
            assert got["trusted_peers"] == sorted([ids[replacement], ids[survivor]])
            assert ids[victim] not in got["trusted_peers"]
    finally:
        c.close()


def _c1_assert_exact(c, res, recipe, tmp_path):
    state = fleet_manifest(c, res, tmp_path / "collected")
    reference = reference_manifest_for(recipe, tmp_path / "ref")
    assert diff_manifests(state, reference) == []
    assert state == reference


def _c1_every_slot_at(c, step):
    run = c.coord.run
    if run is None:
        return False
    states = list(run.states.values())
    return bool(states) and all(st.last_step() == step for st in states)


def test_slow_shutdown_after_the_last_step_completes(tmp_path, port_base, monkeypatch):
    """Every slot finished its last optimizer step; the exit reports then take
    45 s (a slow final-wave ship), longer than any limit S4 can use here (floor
    8 s, first-step timeout 30 s). That quiet is shutdown, not a stall: the run
    completes without a restart and stays exactly the reference."""
    # covers: CONTROL-I17
    # WOULD-FAIL-IF: quiet after every slot reported step steps - 1 feeds the S4 stall clock (the
    # fleet is restarted in place from the final wave, the replayed loss lines collapse the limit
    # to the floor, and the restart budget is spent), or the shutdown changes a byte of the state
    monkeypatch.setenv("WANTRAIN_SIM_EXIT_DELAY_S", "45")
    first_step_s = 30
    c = LocalCluster(tmp_path, port_base, probe=False, first_step_timeout_s=first_step_s)
    try:
        c.add_worker("a")
        c.add_worker("b")
        c.wait_registered(2)
        recipe = _recipe(stages=2, steps=4)
        c.run_async(Job(spec=recipe))
        last_step_at = None
        while c._thread.is_alive():
            if last_step_at is None and _c1_every_slot_at(c, recipe.steps - 1):
                last_step_at = time.time()  # no earlier than the fleet's last WT_STEP
            time.sleep(0.05)
        res = c.join()
        exits = [
            e[0]
            for e in res.events
            if re.match(r"slot \(\d+, \d+\) exited ", e[2])
            and last_step_at is not None
            and e[0] >= last_step_at
        ]
        quiet_s = min(exits) - last_step_at if exits else 0.0
        assert res.ok, f"RUN_FAILED: {res.reason}"
        assert res.restarts == 0, f"RESTARTED: {res.restarts} restart(s)"
        assert quiet_s > first_step_s, f"QUIET_TOO_SHORT: {quiet_s:.1f}s"
        _c1_assert_exact(c, res, recipe, tmp_path)
    finally:
        c.close()


def test_relaunch_from_the_final_wave_runs_no_step_and_is_not_restarted(
    tmp_path, port_base, monkeypatch
):
    """A worker dies after every slot holds the final wave and before the exit
    reports (held 30 s). The heal relaunches from wave steps - 1: that attempt
    runs no optimizer step, only the shutdown, so it is judged by the shutdown
    bound, not by the first-step timeout (20 s), and is never restarted."""
    # covers: CONTROL-I17
    # WOULD-FAIL-IF: _launch leaves start_step at 0 on a relaunch from the final wave: the
    # relaunched attempt runs no step, its 30 s quiet exceeds first_step_timeout_s and it is
    # restarted until the budget is spent; or the relaunch changes a byte of the final state
    monkeypatch.setenv("WANTRAIN_SIM_EXIT_DELAY_S", "30")
    c = LocalCluster(tmp_path, port_base, probe=False, first_step_timeout_s=20)
    try:
        for i in range(3):  # 2 slots + 1 spare
            c.add_worker(f"w{i}")
        c.wait_registered(3)
        recipe = _recipe(stages=2, steps=4)
        final = recipe.steps - 1
        c.run_async(Job(spec=recipe))
        deadline = time.time() + 300
        while True:
            run = c.coord.run
            if (
                run is not None
                and _c1_every_slot_at(c, final)
                and run.store.common_step(list(run.slots)) == final
            ):
                break
            assert time.time() < deadline, "the coordinator never held the final wave of every slot"
            time.sleep(0.1)
        victim = c.name_of((0, 0))
        c.kill_worker(victim)
        res = c.join()
        assert res.ok, res.reason
        assert len(res.heals) == 1, res.heals
        assert res.heals[0]["from"] == victim and res.heals[0]["resumed_from_step"] == final
        assert res.restarts == 0, res.restarts
        # The victim's pre-kill step-3 shard ties with the replacement's, and
        # fleet_manifest keeps the first one it walks: compare only the
        # workers that ran the final attempt, so the healed slot's bytes are
        # the replacement's own.
        import glob
        from types import SimpleNamespace

        replacement = res.heals[0]["to"]
        assert replacement != victim
        shard = glob.glob(
            os.path.join(c.workdirs[replacement], "runs", res.run_id, "out", "*_ckpt_rank0p0.pt")
        )
        assert len(shard) == 1, shard
        with open(shard[0][:-3] + ".step.json") as f:
            assert json.load(f)["step"] == final
        survivors = SimpleNamespace(workdirs={k: v for k, v in c.workdirs.items() if k != victim})
        _c1_assert_exact(survivors, res, recipe, tmp_path)
    finally:
        c.close()


def test_healed_replacement_holds_the_restored_wave(tmp_path, port_base, monkeypatch):
    """A worker dies once the coordinator holds a common wave (step >= 3 of 8,
    a wave every 2 steps; exits held 10 s so the kill lands before the end).
    The heal restores that wave to every slot: the replacement holds it as a
    committed wave of resume_step, byte for byte the dead worker's, its stage
    resumes from the alias naming that same step, and the relaunch spec says
    resume_step = the heal's resumed_from_step."""
    # covers: CONTROL-I10
    # WOULD-FAIL-IF: RESTORE writes only the alias (no committed wave of resume_step on the
    # replacement) or a wave other than the dead worker's shipped one, the alias the stage
    # resumes from names another step than the restored wave, the relaunch spec does not carry
    # resume_step = resumed_from_step, or the healed run changes a byte of the losses
    from wantrain.fault_tolerance import checkpoint

    monkeypatch.setenv("WANTRAIN_SIM_EXIT_DELAY_S", "10")
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(3):  # 2 slots + 1 spare
            c.add_worker(f"w{i}")
        c.wait_registered(3)
        recipe = _recipe(stages=2, steps=8, ckpt_every=2)
        c.run_async(Job(spec=recipe))
        deadline = time.time() + 300
        while True:
            run = c.coord.run
            if run is not None and (run.store.common_step(list(run.slots)) or -1) >= 3:
                break
            assert time.time() < deadline, "the coordinator never held a common wave >= 3"
            time.sleep(0.05)
        victim = c.name_of((0, 0))
        c.kill_worker(victim)
        res = c.join()
        _assert_bitwise(res, recipe, tmp_path)
        assert len(res.heals) == 1 and res.heals[0]["from"] == victim, res.heals
        step = res.heals[0]["resumed_from_step"]
        assert step in (3, 5, 7), res.heals
        replacement = res.heals[0]["to"]
        run_dir = os.path.join(c.workdirs[replacement], "runs", res.run_id)
        with open(os.path.join(run_dir, "spec-a2-r0p0.json")) as f:
            spec = json.load(f)
        assert spec["resume"] is True and spec["resume_step"] == step
        got = checkpoint.validate_wave(
            checkpoint.wave_dir(os.path.join(run_dir, "out"), spec["tag"], 0, 0, step), step
        )
        dead_out = os.path.join(c.workdirs[victim], "runs", res.run_id, "out")
        shipped = checkpoint.validate_wave(
            checkpoint.wave_dir(dead_out, spec["tag"], 0, 0, step), step
        )
        assert got == shipped
        with open(os.path.join(run_dir, "stage-r0p0.log")) as f:
            resumes = [ln for ln in f.read().splitlines() if ln.startswith("WT_RESUME ")]
        assert len(resumes) == 1 and f" start_step={step + 1} " in resumes[0], resumes
    finally:
        c.close()


def _c2_lines(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _c2_check_slot(spec, run_dir, attempt, r, p, sums):
    """Ops complete and per-hop frame accounting for one slot's summaries as the
    coordinator logged them, against the slot's telemetry file shipped at exit."""
    assert [s["step"] for s in sums] == list(range(spec.steps)), (r, p, sums)
    # What the slot's schedule runs per step, from the spec, never from the events.
    want = {op: sum(1 for o in spec.schedule_order[r][p] if o[0] == op) for op in "FBW"}
    assert want["F"] == len(spec.owned(r, p)), (r, p, want)
    for s in sums:
        assert (s["v"], s["rank"], s["peer"], s["attempt"]) == (1, r, p, attempt), s
        for op in "FBW":
            assert s["ops"][op]["n"] == want[op], (r, p, op, s)
        assert s["ops"]["OPT"]["n"] == 1, (r, p, s)
        assert all(s["ops"][op]["med_ms"] > 0 for op in ("F", "B", "W", "OPT")), (r, p, s)
    events = _c2_lines(os.path.join(run_dir, f"telemetry_r{r}p{p}_a{attempt}.jsonl"))
    sends = [e for e in events if e["ev"] == "send"]
    assert sends and all("hop" in e for e in sends), (r, p)
    for hop in {e["hop"] for e in sends} | {h for s in sums for h in s["hops"]}:
        early = sum(1 for e in sends if e["hop"] == hop and e["step"] <= spec.steps - 2)
        framed = sum(s["hops"].get(hop, {}).get("frames", 0) for s in sums)
        total = sum(1 for e in sends if e["hop"] == hop)
        assert early <= framed <= total, (r, p, hop, early, framed, total)


def test_live_telemetry_reaches_the_coordinator_byte_exact(tmp_path, port_base):
    """Every slot of a wide-stage fleet streams one validated per-step summary
    to the coordinator (TELEM_STEP), whose op counts match the slot's schedule
    and whose hop frames match the slot's event log; nothing is dropped, the
    cost of measuring is printed, and the training state is the reference's."""
    # covers: CONTROL-I18, RUNTIME-I9
    # WOULD-FAIL-IF: the agent does not forward WT_TELEM as TELEM_STEP, the coordinator drops or
    # mislabels a valid summary (wrong slot, attempt or step), a summary's op counts or hop frames
    # disagree with the schedule and the event log, or telemetry moves one bit of the state
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(3):
            c.add_worker(f"t{i}")
        c.wait_registered(3)
        recipe = _recipe(stages=2, stage_widths=[2, 1], wire_codec="raw", steps=4)
        res = c.run(Job(spec=recipe))
        assert res.ok, res.reason
        assert (res.attempts, res.restarts, res.heals) == (1, 0, []), res.events[-5:]
        spec = JobSpec.load(os.path.join(res.run_dir, "spec.json"))
        tel = res.rationale["telemetry"]
        recs = _c2_lines(os.path.join(res.run_dir, "telem_steps.jsonl"))
        slots = [(r, p) for r, row in enumerate(spec.endpoints) for p in range(len(row))]
        assert len(slots) == 3, slots
        by_slot = {slot: [] for slot in slots}
        for rec in recs:
            by_slot[(rec["rank"], rec["peer"])].append(rec)
        assert tel["dropped"] == 0, tel
        assert tel["accepted"] == len(recs) == len(slots) * spec.steps, (tel, len(recs))
        checked = 0
        for r, p in slots:
            _c2_check_slot(spec, res.run_dir, res.attempts, r, p, by_slot[(r, p)])
            checked += len(by_slot[(r, p)])
        _c1_assert_exact(c, res, recipe, tmp_path)

        emit_us = [rec["emit_us"] for rec in recs]
        step_ms = [rec["step_ms"] for rec in recs]
        assert all(x >= 0 for x in emit_us) and all(x > 0 for x in step_ms)
        frac = statistics.median(emit_us) / 1000 / statistics.median(step_ms)
        print(f"TELEM_US_MAX={max(emit_us):.6f}")
        print(f"TELEM_FRAC={frac:.6f}")
        print(f"TELEM_INTAKE_US_MAX={tel['handle_us_max']:.6f}")
        print(f"OPS_CHECKED={checked:.6f}")
    finally:
        c.close()


# ---------------------------------------------------------------- CONTROL-I21


class _RbChannel:
    """FAKE of the control channel at the worker-network boundary, named as
    such: RESTORE_PART sends are recorded, nothing is transmitted."""

    last_rx = 0.0

    def __init__(self):
        self.sent = []

    def send_file(self, msg, path):
        self.sent.append(dict(msg))

    def send(self, msg, body=b""):
        self.sent.append(dict(msg))

    def close(self):
        pass


_RB_WIDTHS = [1, 2, 1]
_RB_SLOTS = [(r, p) for r, w in enumerate(_RB_WIDTHS) for p in range(w)]


def _rb_node(i):
    return NodeInfo(
        node_id=f"n{i}",
        name=f"w{i}",
        host=f"10.0.0.{i}",
        data_port=1,
        device="cpu",
        gpu_name="cpu",
        vram_total_bytes=24 << 30,
        vram_free_bytes=24 << 30,
        tflops_fp32=1.0,
        region="eu",
        device_class="cpu:arm64:DEFAULT",
    )


def _rb_coordinator(tmp_path, steps=24, spare=False, on_abort=None, **cfg):
    """A real Coordinator and a toy 1-2-1 run launched as attempt 1 on four
    workers (plus one idle spare), restart cost 0.1 s unless given. FAKE at the worker-network boundary, named
    as such: `_request` records (node, kind, message, body) and answers
    {"ok": True}, calling `on_abort` on every ABORT first (as an agent ships
    its waves before its ABORT_ACK); `_sync_all_connections` does nothing.
    Every coordinator method the tests call is the real one."""
    cfg.setdefault("replan_restart_cost_s", 0.1)
    coord = Coordinator(
        config=CoordinatorConfig(state_dir=str(tmp_path / "coord"), poll_s=0.01, **cfg)
    )
    sent = []

    def fake_request(w, msg, body=b"", timeout=None, file=None, reply_kind=None):
        if msg["kind"] == "ABORT" and on_abort is not None:
            on_abort()
        sent.append((w.node_id, msg["kind"], dict(msg), body))
        return {"ok": True}

    coord._request = fake_request
    coord._sync_all_connections = lambda run: None
    os.makedirs(tmp_path / "run")
    run = _Run("run-rb", str(tmp_path / "run"))
    spec = _recipe(stages=3, stage_widths=list(_RB_WIDTHS), wire_codec="raw", steps=steps)
    spec.finalize()
    run.spec, run.spec_json = spec, spec.to_json()
    for i, slot in enumerate(_RB_SLOTS + ([None] if spare else [])):
        info = _rb_node(i)
        w = Worker(info, _RbChannel(), f"session{i}", None, peer_id=f"{i:032x}")
        coord.workers[info.node_id] = w
        if slot is not None:
            w.run_id = run.run_id
            run.slots[slot] = info.node_id
    coord.run = run
    coord._launch(run)
    return coord, run, sent


def _rb_waves(run, *steps):
    """Commit wave `step` of every slot through the store's real staging path,
    then prune as the CKPT intake does."""
    import hashlib

    for step in steps:
        for slot in run.slots:
            data = f"wave-{step}-r{slot[0]}p{slot[1]}|".encode() * 64
            body = os.path.join(run.run_dir, f"in-r{slot[0]}p{slot[1]}-{step}.part")
            with open(body, "wb") as f:
                f.write(data)
            sha = hashlib.sha256(data).hexdigest()
            run.store.stage_part(slot, run.attempt, step, "state.pt", sha, body)
            manifest = {
                "v": 1,
                "step": step,
                "files": [{"name": "state.pt", "sha256": sha, "bytes": len(data)}],
            }
            assert run.store.commit_staged(slot, run.attempt, step, manifest) is True
        run.store.prune(list(run.slots))


def _rb_summary(spec, slot, step, slow=1.0):
    """A valid TELEM_STEP summary of `slot` at `step` under `spec`'s ownership;
    op times scaled by `slow`."""
    r, p = slot
    n = len(spec.owned(r, p))
    hops = {}
    for head, nb in (("next", r + 1), ("prev", r - 1)):
        if 0 <= nb < len(_RB_WIDTHS):
            for q in range(_RB_WIDTHS[nb]):
                hops[f"{head}{q}"] = {"frames": 2 * n, "bytes": 4096 * n, "busy_ms": 1.0 * n}
    return {
        "v": 1,
        "step": step,
        "rank": r,
        "peer": p,
        "step_ms": 100.0,
        "emit_us": 10.0,
        "ops": {
            "F": {"n": n, "med_ms": 3.0 * slow},
            "B": {"n": n, "med_ms": 5.0 * slow},
            "W": {"n": n, "med_ms": 2.0 * slow},
            "OPT": {"n": 1, "med_ms": 1.0},
        },
        "stall_ms": 0.0,
        "hops": hops,
    }


def _rb_telem(coord, run, steps, slow_slot=None, node=None, attempt=None):
    """One TELEM_STEP per slot and step through the real _dispatch; the slot
    `slow_slot` is 4x slower. `node` sends from that worker's connection."""
    spec = coord._owned_spec(run)
    for step in steps:
        for slot in run.slots:
            w = coord.workers[node or run.slots[slot]]
            summary = _rb_summary(spec, slot, step, 4.0 if slot == slow_slot else 1.0)
            msg = {
                "kind": "TELEM_STEP",
                "run_id": run.run_id,
                "attempt": run.attempt if attempt is None else attempt,
                "summary": summary,
            }
            coord._dispatch(w, msg, b"")


def _rb_assigns(sent, attempt):
    return [(m, body) for _n, kind, m, body in sent if kind == "ASSIGN" and m["attempt"] == attempt]


def test_telemetry_window_starts_empty_each_attempt(tmp_path):
    """Every attempt (first launch, restart, heal, rebalance) starts with an
    empty telemetry window, and a summary of an earlier attempt that arrives
    after the relaunch never enters it."""
    # covers: CONTROL-I21
    # WOULD-FAIL-IF: _launch keeps run.telem_window across attempts (a restart, heal or rebalance
    # window still holds the earlier attempt's or the dead owner's summaries), or a late summary
    # of an earlier attempt is accepted into the new window
    coord, run, sent = _rb_coordinator(tmp_path, spare=True)
    run.telem_window = {(1, 1): ["stale"]}
    coord._launch(run)  # a first launch of a run whose window was not empty
    assert run.attempt == 2 and run.telem_window == {}
    _rb_telem(coord, run, range(4))
    assert {slot: len(w) for slot, w in run.telem_window.items()} == {s: 4 for s in _RB_SLOTS}
    _rb_waves(run, 3)

    coord._restart_in_place(run, "test stall")
    assert run.attempt == 3 and run.restarts == 1 and run.telem_window == {}
    dropped = run.telem_stats["dropped"]
    _rb_telem(coord, run, [4], attempt=2)  # the aborted attempt's late summaries
    assert run.telem_window == {} and run.telem_stats["dropped"] == dropped + len(_RB_SLOTS)
    _rb_telem(coord, run, range(4, 8))

    coord._heal(run, (1, 1), "test conviction")
    assert run.attempt == 4 and len(run.heals) == 1 and run.telem_window == {}
    _rb_telem(coord, run, range(4, 8), slow_slot=(1, 1))
    _rb_waves(run, 4)

    assert coord._maybe_rebalance(run) is True
    assert run.attempt == 5 and len(run.replans) == 1 and run.telem_window == {}
    assert (run.restarts, len(run.heals)) == (1, 1)  # a rebalance consumes neither


def test_rebalance_restores_the_common_wave_after_the_abort(tmp_path):
    """The rule decides at common wave 3; during the abort every slot ships
    wave 4 and the intake's prune drops wave 3. The rebalance re-reads the
    common step and restores and relaunches from wave 4 under the re-planned
    ownership. With no common wave left after the abort it is abandoned: no
    relaunch, never from step 0."""
    # covers: CONTROL-I21
    # WOULD-FAIL-IF: the rebalance restores the step it decided at (pruned during the abort: the
    # restore fails or relaunches from a wave the store no longer holds), relaunches without the
    # re-planned ownership or without resume_step, consumes a restart or a heal, or a missing
    # common wave relaunches the fleet from step 0
    import threading

    shipped: list = []
    once = threading.Lock()

    def ship_during_abort():  # one ABORT per worker, each on its own thread
        with once:
            if not shipped:
                shipped.append(4)
                _rb_waves(run, 4)

    coord, run, sent = _rb_coordinator(tmp_path / "a", on_abort=ship_during_abort)
    _rb_waves(run, 2, 3)
    _rb_telem(coord, run, range(4), slow_slot=(1, 1))
    old = coord._owned_spec(run)
    assert coord._maybe_rebalance(run) is True
    assert shipped == [4] and not run.store.has((0, 0), 3)
    assert run.attempt == 2 and (run.restarts, run.heals) == (0, [])
    assert len(run.replans) == 1
    rec = run.replans[0]
    assert set(rec) == {"step", "pred_cur_s", "pred_new_s", "owner_changed", "peer_weights"}
    assert rec["step"] == 4 and rec["owner_changed"] is True
    assert rec["pred_new_s"] <= 0.9 * rec["pred_cur_s"], rec
    restores = [m for _n, kind, m, _b in sent if kind == "RESTORE"]
    assert sorted(m["step"] for m in restores) == [4] * len(_RB_SLOTS)
    assigns = _rb_assigns(sent, 2)
    assert len(assigns) == len(_RB_SLOTS)
    for _m, body in assigns:
        got = json.loads(body)
        assert got["resume"] is True and got["resume_step"] == 4
        assert got["peer_weights"] == rec["peer_weights"] == run.ownership["peer_weights"]
        assert got["mb_owner"][1].count(1) < old.mb_owner[1].count(1)
    assert {s: st.start_step for s, st in run.states.items()} == {s: 5 for s in _RB_SLOTS}
    assert any(e[2].startswith("REPLAN {") for e in coord.events)

    import shutil

    def lose_waves():
        shutil.rmtree(run2.store._slot_dir((2, 0)), ignore_errors=True)

    coord2, run2, sent2 = _rb_coordinator(tmp_path / "b", on_abort=lose_waves)
    _rb_waves(run2, 3)
    _rb_telem(coord2, run2, range(4), slow_slot=(1, 1))
    with pytest.raises(RuntimeError, match="rebalance abandoned"):
        coord2._maybe_rebalance(run2)
    assert run2.attempt == 1 and run2.replans == []
    assert [k for _n, k, _m, _b in sent2].count("ASSIGN") == len(_RB_SLOTS)  # attempt 1 only
    assert any("REPLAN abandoned" in e[2] for e in coord2.events)


class _RbStop(Exception):
    pass


def _rb_scripted_judge(script):
    """FAKE liveness judge (the judge has its own tests): hands out the
    scripted verdicts in order (a callable runs first), then stops the
    monitor."""

    def judge(states, now=None):
        if not script:
            raise _RbStop()
        step = script.pop(0)
        if callable(step):
            step = step()
        return step

    return type("ScriptedJudge", (), {"judge": staticmethod(judge)})()


def test_rebalance_and_a_heal_in_one_window_make_one_attempt(tmp_path):
    """A window that would rebalance meets a conviction in the same monitor
    pass: the heal runs alone (one new attempt), and since the heal's attempt
    starts a new window, the passes after it never rebalance, even once a
    newer common wave arrives. The control arm, without the conviction,
    rebalances from the same window."""
    # covers: CONTROL-I21
    # WOULD-FAIL-IF: the rule is evaluated before or alongside a convict verdict (two attempts
    # from one window), or the healed attempt is rebalanced from the summaries of the attempt
    # before the heal
    coord, run, sent = _rb_coordinator(tmp_path / "heal", spare=True)
    _rb_waves(run, 3)
    _rb_telem(coord, run, range(4), slow_slot=(1, 1))
    script = [
        Verdict("convict", slot=(0, 0), why="killed"),
        lambda: _rb_waves(run, 4) or Verdict("ok"),
        Verdict("ok"),
    ]
    judge = _rb_scripted_judge(script)
    coord._judge = lambda run: judge
    with pytest.raises(_RbStop):
        coord._monitor(run)
    assert run.attempt == 2 and len(run.heals) == 1 and run.replans == []
    assert len(_rb_assigns(sent, 2)) == len(_RB_SLOTS) and not _rb_assigns(sent, 3)

    coord, run, sent = _rb_coordinator(tmp_path / "control")
    _rb_waves(run, 3)
    _rb_telem(coord, run, range(4), slow_slot=(1, 1))
    judge = _rb_scripted_judge([Verdict("ok"), Verdict("ok")])
    coord._judge = lambda run: judge
    with pytest.raises(_RbStop):
        coord._monitor(run)
    assert run.attempt == 2 and len(run.replans) == 1 and run.heals == []


def test_rebalance_after_a_heal_ignores_the_dead_nodes_telemetry(tmp_path):
    """The 4x slow peer is convicted and healed onto a spare. Its window is
    gone with its attempt, and its late summaries are refused: the healed
    attempt's window (all peers equal) never rebalances, although the window
    with the dead node's summaries kept in it would."""
    # covers: CONTROL-I21
    # WOULD-FAIL-IF: the window survives the heal (the dead node's 4x summaries decide a
    # rebalance of a healed, balanced fleet), or a late summary from the dead node's connection
    # enters the healed attempt's window
    coord, run, sent = _rb_coordinator(tmp_path, spare=True)
    _rb_waves(run, 3)
    _rb_telem(coord, run, range(4), slow_slot=(1, 1))
    before = {slot: list(w) for slot, w in run.telem_window.items()}
    dead = run.slots[(1, 1)]
    coord._heal(run, (1, 1), "test conviction")
    assert run.slots[(1, 1)] != dead and run.attempt == 2 and run.telem_window == {}
    _rb_telem(coord, run, range(4, 8), node=dead, attempt=1)  # late, old attempt
    _rb_telem(coord, run, range(4, 8), node=dead)  # late, from the dead node's connection
    assert run.telem_window == {}
    _rb_telem(coord, run, range(4, 8))
    _rb_waves(run, 4)
    assert coord._maybe_rebalance(run) is False
    assert run.attempt == 2 and run.replans == []
    kept = {slot: before[slot] + list(w) for slot, w in run.telem_window.items()}
    assert replan(coord._owned_spec(run), kept)["decision"] is True


_RB_SLOW_FROM = 6  # WANTRAIN_SIM_SLOW's from_step in the test below


def _rb_measured_step_s(run_dir, attempt, first_step):
    """Mean wall time per step (s) of `attempt` over its steps >= first_step,
    from the per-step summaries the coordinator logged (telem_steps.jsonl).
    A step's time is the max step_ms over the slots: the slots of a pipeline
    step wait for each other, so the slowest one is the fleet's step. Only
    steps every slot reported count. -> (mean s, the steps used)."""
    by_step: dict = {}
    for r in _c2_lines(os.path.join(run_dir, "telem_steps.jsonl")):
        if r["attempt"] == attempt and r["step"] >= first_step:
            by_step.setdefault(r["step"], {})[(r["rank"], r["peer"])] = r["step_ms"]
    steps = sorted(s for s, slots in by_step.items() if len(slots) == len(_RB_SLOTS))
    assert len(steps) >= 3, (attempt, first_step, sorted(by_step))
    return statistics.mean(max(by_step[s].values()) for s in steps) / 1e3, steps


def test_slowed_peer_is_rebalanced_and_the_state_stays_exact(tmp_path, port_base, monkeypatch):
    """A toy 1-2-1 fleet whose peer (1, 1) turns 4x slower from step 6: the
    coordinator re-plans once from the current attempt's telemetry, moves
    microbatches off the slow peer, restores the common wave and relaunches
    it as its own attempt; losses and the final state equal the reference.
    The real step gets faster: attempt 2's measured step is at most 0.75x
    attempt 1's slowed step. The planner's prediction error (gain and absolute
    step) is measured and printed, not asserted: the human's decision of
    2026-10-09 (HANDOUT-M2 §2c), with calibrating predict_step for CPU
    contention left to M3."""
    # covers: CONTROL-I21
    # WOULD-FAIL-IF: the coordinator never re-plans a slowed peer, re-plans more than once, the
    # rebalance consumes a restart or heal, the relaunch keeps the old ownership or does not
    # resume from the recorded wave, or the rebalance moves one bit of the losses or the state
    # WOULD-FAIL-IF: the new ownership does not make the measured step at least 25% faster
    # than the slowed step before the replan (a re-plan that moves too little work, or none,
    # off the slow peer)
    monkeypatch.setenv("WANTRAIN_SIM_SLOW", f"1:1:4:{_RB_SLOW_FROM}")
    c = LocalCluster(tmp_path, port_base, probe=False, replan_restart_cost_s=0.5)
    try:
        for i in range(4):
            c.add_worker(f"r{i}")
        c.wait_registered(4)
        recipe = _recipe(stages=3, stage_widths=[1, 2, 1], wire_codec="raw", steps=40)
        res = c.run(Job(spec=recipe))
        _assert_bitwise(res, recipe, tmp_path)
        assert len(res.replans) == 1, res.replans
        rec = res.replans[0]
        assert rec["owner_changed"] is True, rec
        assert (res.attempts, res.restarts, res.heals) == (2, 0, []), res.events[-5:]
        holder = _ck_holder(c, res.run_id, 2, (1, 1))
        assert holder == _ck_holder(c, res.run_id, 1, (1, 1))
        first = _ck_spec(c, holder, res.run_id, 1, (1, 1))
        second = _ck_spec(c, holder, res.run_id, 2, (1, 1))
        assert second["resume"] is True and second["resume_step"] == rec["step"]
        assert second["peer_weights"] == rec["peer_weights"]
        assert second["mb_owner"][1].count(1) < first["mb_owner"][1].count(1), (first, second)
        assert any(e[2].startswith("REPLAN {") for e in res.events)
        print(f"REPLAN_STEP={rec['step']}")
        print(f"REPLAN_PRED={rec['pred_cur_s']:.6f}->{rec['pred_new_s']:.6f}")
        _c1_assert_exact(c, res, recipe, tmp_path)
        # Before: attempt 1's steps from the first slowed one (every op of the slow peer
        # stretched) until the abort. New: attempt 2's steps after its first (resume warm-up),
        # and never one before the slowdown starts.
        before, before_steps = _rb_measured_step_s(res.run_dir, 1, _RB_SLOW_FROM)
        new, new_steps = _rb_measured_step_s(res.run_dir, 2, max(rec["step"] + 2, _RB_SLOW_FROM))
        measured_gain = before - new
        pred_gain = rec["pred_cur_s"] - rec["pred_new_s"]
        print(f"MEASURED_STEPS before={before_steps} new={new_steps}")
        print(
            f"MEASURED_SPEEDUP before={before:.6f} new={new:.6f} ratio={new / before:.6f} "
            f"pred_new={rec['pred_new_s']:.6f} gain={measured_gain:.6f} pred_gain={pred_gain:.6f}"
        )
        # The prediction error is MEASURED, not asserted (the human's decision, 2026-10-09):
        # gain error relative to the predicted gain, and measured/predicted absolute step.
        print(
            f"MEASURED_BIAS before={before / rec['pred_cur_s']:.6f} "
            f"new={new / rec['pred_new_s']:.6f}"
        )
        print(f"MEASURED_PRED_ERROR gain_rel={(measured_gain - pred_gain) / pred_gain:.6f}")
        assert new <= 0.75 * before, (before, new)
    finally:
        c.close()


def test_balanced_fleet_is_never_replanned(tmp_path, port_base):
    """The same 1-2-1 fleet with no slowed peer and a restart cost of 0.5 s:
    the rule is evaluated on every new common wave and never fires."""
    # covers: CONTROL-I21
    # WOULD-FAIL-IF: the rule ignores the 10% threshold or the restart cost, or the CPU noise of
    # two equal peers' telemetry triggers a rebalance of a balanced fleet
    c = LocalCluster(tmp_path, port_base, probe=False, replan_restart_cost_s=0.5)
    try:
        for i in range(4):
            c.add_worker(f"b{i}")
        c.wait_registered(4)
        recipe = _recipe(stages=3, stage_widths=[1, 2, 1], wire_codec="raw", steps=16)
        res = c.run(Job(spec=recipe))
        _assert_bitwise(res, recipe, tmp_path)
        assert res.replans == [], res.replans
        assert (res.attempts, res.restarts, res.heals) == (1, 0, []), res.events[-5:]
    finally:
        c.close()


# ---------------------------------------------------------------- M2b: wide plans, shard pieces

# A toy plan the planner places wide on four local cpu workers: at 64 sequences
# per microbatch two stages of two peers simulate faster than four stages of one.
PIECES_JOB = {"model": "toy", "steps": 3, "device": "cpu", "mb_size": 64, "microbatches": 8}


def _pieces_recipe(**kw):
    return _recipe(shard="pieces", wire="native", stream_grads=True, wire_codec="raw", **kw)


def _plan_cluster(tmp_path, port_base, n=4, declare=None):
    c = LocalCluster(
        tmp_path, port_base, probe=True, probe_bulk_bytes=1 << 20, measure_bulk_bytes=1 << 20
    )
    for i in range(n):
        c.add_worker(f"p{i}", declare=declare)
    c.wait_registered(n)
    return c


def _run_spec(res):
    return JobSpec.load(os.path.join(res.run_dir, "spec.json"))


def test_plan_wide_zero1_oracle(tmp_path, port_base):
    """A plan job on four cpu workers whose fleet makes a wide stage faster,
    restricted to shard 'zero1': the coordinator plans and launches a zero1
    wide stage (each peer its own lane order from the bridge), and the run's
    losses and final state equal the oracle."""
    # covers: CONTROL-I4, CONTROL-I6
    # WOULD-FAIL-IF: plan mode drops the planner's widths (one node per stage), groups the
    # slots other than the placement, mis-pins the endpoints of a wide stage, or the wide
    # zero1 plan moves one bit of the losses or the state
    # Equal declared speeds: the probes' CPU speeds vary with the machine's load, and
    # with them which depth wins; with four equal peers S 2 w 2 simulates at 0.62-0.87 of
    # S 4 w 1 over host DRAM 1-34 GB/s and NIC 8-30 Gbit/s (zero1 admits declared inputs).
    c = _plan_cluster(tmp_path, port_base, declare={"tflops": 1.0})
    try:
        res = c.run(Job(plan=dict(PIECES_JOB, shard="zero1"), devices=4))
        assert res.ok, res.reason
        spec = _run_spec(res)
        rat = res.rationale
        assert spec.shard == "zero1" and max(spec.stage_widths) > 1, rat.get("plan_candidates")
        assert rat["stage_widths"] == list(spec.stage_widths)
        assert rat["schedule"]["source"] == "profiled", rat["schedule"]
        for r, w in enumerate(spec.stage_widths):
            assert len(spec.schedule_order[r]) == len(spec.endpoints[r]) == w
        recipe = JobSpec.from_json(spec.to_json())
        _assert_bitwise(res, recipe, tmp_path)
        _c1_assert_exact(c, res, recipe, tmp_path)
    finally:
        c.close()


def test_plan_wide_spec_fields_round_trip_through_json(tmp_path):
    """On the integrated tree, build_spec(fleet, allow_pieces=True) on a fleet
    I16 admits returns a spec_json that carries shard, wire, emission_order
    and piece_bounds; JobSpec.from_json of it round-trips them exactly and
    passes finalize (the planner set them by attribute, I-8)."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the planner's M2b fields are lost or renamed in to_json/from_json, or the
    # real finalize refuses a combination the planner emits
    from wantrain.coordinator.coordinator import plan_fleet_nodes
    from wantrain.scheduler import admission, fleetplan

    infos = [
        NodeInfo(
            node_id=f"n{i}",
            name=f"w{i}",
            host="127.0.0.1",
            data_port=7000 + 8 * i,
            device="cpu",
            gpu_name="cpu",
            vram_total_bytes=16 << 30,
            vram_free_bytes=16 << 30,
            tflops_fp32=1.0,
            hostname="mac",
            host_ram_bytes=16 << 30,
            host_dram_gbps=34.0,
            device_class="cpu:arm64:DEFAULT",
        )
        for i in range(4)
    ]
    nodes = plan_fleet_nodes(infos, {n.node_id: (16_000.0, 16_000.0) for n in infos})
    fleet = {
        "run_id": "fields",
        "nodes": nodes,
        "link": {"mbps": 16_000.0, "rtt_ms": 0.1, "duplex": "per_direction"},
        "k_passes": {"value": admission.K_HOST_DRAM_COUNTED, "source": "counted"},
        "job": dict(PIECES_JOB, shard="pieces"),
    }
    out = fleetplan.build_spec(fleet, allow_pieces=True)
    d = json.loads(out["spec_json"])
    assert (d["shard"], d["wire"]) == ("pieces", "native")
    assert d["emission_order"] == out["emission_order"] and len(d["emission_order"]) == d["stages"]
    assert set(d["emission_order"]) <= {"store", "forward"}
    assert "piece_bounds" in d and d["piece_bounds"] is None
    back = JobSpec.from_json(out["spec_json"])
    for k in ("shard", "wire", "emission_order", "piece_bounds", "stream_grads", "stage_widths"):
        assert getattr(back, k) == d[k] == getattr(out["spec"], k), k
    assert back.finalize().to_json() == out["spec_json"]


def test_pieces_plan_oracle(tmp_path, port_base):
    """A plan job with shard 'pieces' on four local cpu workers: I16 admits it
    on probed inputs only (no input missing or declared; PCIe absent on cpu
    nodes), the plan has a pieces wide stage, and the losses and final state
    equal the oracle."""
    # covers: CONTROL-I4, SCHEDULER-I16
    # WOULD-FAIL-IF: plan mode admits pieces on a declared or missing input, launches a
    # pieces plan without arming or the agent's launcher flag, or the pieces run moves one
    # bit of the losses or the state
    c = _plan_cluster(tmp_path, port_base)
    try:
        res = c.run(Job(plan=dict(PIECES_JOB, shard="pieces"), devices=4))
        assert res.ok, (res.reason, [e[2] for e in res.events if "registered" in e[2]])
        spec = _run_spec(res)
        assert (spec.shard, spec.wire) == ("pieces", "native") and max(spec.stage_widths) > 1
        for node in res.rationale["inputs"]["nodes"]:
            for k in ("host_ram_bytes", "host_dram_gbps", "nic_up_gbps", "nic_down_gbps"):
                assert node[k]["source"] == "measured", (k, node)
            for k in ("pcie_h2d_gbps", "pcie_d2h_gbps"):
                assert node[k]["source"] == "absent", (k, node)
            for k in ("device", "gpu_name"):
                assert node[k]["source"] == "measured", (k, node)
        assert res.rationale["inputs"]["link"]["source"] == "measured"
        recipe = JobSpec.from_json(spec.to_json())
        _assert_bitwise(res, recipe, tmp_path)
        _c1_assert_exact(c, res, recipe, tmp_path)
    finally:
        c.close()


def _wait_common_wave(c, step, timeout=300):
    deadline = time.time() + timeout
    while True:
        run = c.coord.run
        if run is not None and run.slots and (run.store.common_step(list(run.slots)) or -1) >= step:
            return
        assert time.time() < deadline, f"the coordinator never held a common wave >= {step}"
        time.sleep(0.05)


def test_pieces_heal_run(tmp_path, port_base, monkeypatch):
    """A pieces stage of width 2 loses a peer mid-run. With an identical-model
    spare the slot is healed and the run ends equal to the oracle (CONTROL-I10
    under pieces); with only a spare whose class is not the identical probed
    model, the run fails cleanly naming the class it needed."""
    # covers: CONTROL-I9, CONTROL-I10
    # WOULD-FAIL-IF: a pieces slot is healed onto a declared or different device class, the
    # heal of a pieces stage changes a bit of the state, or a missing identical spare ends
    # in anything but a clean failure naming the class
    monkeypatch.setenv("WANTRAIN_SIM_EXIT_DELAY_S", "10")
    recipe = _pieces_recipe(stages=2, stage_widths=[2, 1], steps=8, ckpt_every=2)
    c = LocalCluster(tmp_path / "same", port_base, probe=False)
    try:
        for i in range(4):  # 3 slots + 1 identical spare
            c.add_worker(f"s{i}")
        c.wait_registered(4)
        c.run_async(Job(spec=recipe))
        _wait_common_wave(c, 3)
        victim = c.name_of((0, 1))
        c.kill_worker(victim)
        res = c.join()
        assert res.ok, res.reason
        assert len(res.heals) == 1 and res.heals[0]["slot"] == [0, 1], res.heals
        _assert_bitwise(res, recipe, tmp_path / "same")
        _c1_assert_exact(c, res, recipe, tmp_path / "same")
    finally:
        c.close()

    c = LocalCluster(tmp_path / "other", port_base + 32, probe=False)  # the block's upper half
    try:
        for i in range(3):
            c.add_worker(f"o{i}")
        c.wait_registered(3)
        c.run_async(Job(spec=recipe))
        _wait_common_wave(c, 1)
        c.add_worker("spare", declare={"gpu_name": "sim-other"})  # only after the slots are taken
        c.wait_registered(1)
        c.kill_worker(c.name_of((0, 0)))
        res = c.join()
        assert not res.ok and res.heals == []
        assert "no registered spare" in res.reason, res.reason
        assert "identical probed device model 'cpu' on 'cpu'" in res.reason, res.reason
    finally:
        c.close()


def test_pieces_rebalance_applies_and_stays_exact(tmp_path, port_base, monkeypatch):
    """A toy 1-2-1 fleet whose pieces stage has one peer 2x slower from step
    6: M2's rebalance rule fires, the I16 re-check of the new ownership passes,
    the rebalance is applied as its own attempt, and the final state equals
    the oracle by sha256 (BA-44)."""
    # covers: SCHEDULER-I16, CONTROL-I21
    # WOULD-FAIL-IF: the pieces re-check refuses an admissible rebalance, the rebalance of a
    # pieces stage changes its layout or a bit of the losses or the state
    monkeypatch.setenv("WANTRAIN_SIM_SLOW", "1:1:2:6")
    # A 2x peer saves ~0.02 s of a ~0.09 s toy step: over the ~25 steps left that
    # pays 2 x R only for R below ~0.3 s (the configured R before one is measured).
    c = LocalCluster(tmp_path, port_base, probe=False, replan_restart_cost_s=0.1)
    try:
        for i in range(4):
            c.add_worker(f"r{i}")
        c.wait_registered(4)
        recipe = _pieces_recipe(stages=3, stage_widths=[1, 2, 1], steps=40)
        res = c.run(Job(spec=recipe))
        assert res.ok, res.reason
        assert len(res.replans) == 1, res.replans
        rec = res.replans[0]
        assert rec.get("applied", True) is True and rec["owner_changed"] is True, rec
        assert (res.attempts, res.restarts, res.heals) == (2, 0, []), res.events[-5:]
        _assert_bitwise(res, recipe, tmp_path)
        _c1_assert_exact(c, res, recipe, tmp_path)
    finally:
        c.close()


# ---------------------------------------------------------------- M4: arch 'hf' and MoE, armed (arm-m4)
#
# Every HF model here is a tiny, randomly initialized config of its pinned transformers config
# class (hfm.tiny_config); no network, no hub, no from_pretrained. The agents inherit the
# offline environment.

M4_HF = {
    "llama": ("llama", {"tie_word_embeddings": False}),
    "gemma2": (
        "gemma2",
        {
            "tie_word_embeddings": True,
            "sliding_window": 8,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
        },
    ),
}
M4_MOE_MODEL = dict(
    vocab_size=32,
    n_layer=2,
    n_head=2,
    d_model=16,
    seq_len=6,
    norm="rmsnorm",
    mlp="swiglu",
    bias=False,
    pos="rope",
    moe_experts=6,
    moe_top_k=2,
    moe_d_ff=8,
    moe_select="score_bias",
    moe_bias_rate=0.01,
    moe_lbl_coef=0.01,
    moe_z_coef=0.001,
)
M4_BUILT = "hf: record built once"
M4_MODELING_LLAMA = "transformers/models/llama/modeling_llama.py"


def _m4_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _m4_hf_config(fam: str) -> dict:
    from wantrain.model import hfm

    mt, over = M4_HF[fam]
    # The run asserts config.use_cache False (SPEC-model §2), as the siblings' arms do.
    return hfm.tiny_config(mt, use_cache=False, **over)


def _m4_hf_recipe(fam: str, **kw) -> JobSpec:
    from wantrain.model import hfm
    from wantrain.model.gpt import GPTConfig

    hc = _m4_hf_config(fam)
    model = GPTConfig(arch="hf", hf_config=hc, seq_len=16, **hfm.config_fields(hc))
    base = dict(model=model, stages=2, steps=6, microbatches=4, mb_size=2, ckpt_every=2)
    base.update(kw)
    spec = JobSpec(**base)
    # A little emulated WAN latency keeps a run long enough to fault mid-way.
    spec.links = [
        dict(latency_ms=15.0, bw_mbps=0.0, jitter_ms=0.0, jitter_seed=0) for _ in range(spec.stages)
    ]
    return spec


def _m4_moe_recipe(**kw) -> JobSpec:
    from wantrain.model.gpt import GPTConfig

    base = dict(
        model=GPTConfig(**M4_MOE_MODEL),
        stages=2,
        stage_widths=[2, 1],
        steps=4,
        microbatches=4,
        mb_size=2,
    )
    base.update(kw)
    return JobSpec(**base)


def _m4_launched(res) -> JobSpec:
    return JobSpec.load(os.path.join(res.run_dir, "spec.json"))


def _m4_exact(c, res, tmp_path) -> dict:
    """The run's losses (hex) and state manifest against the oracle of the spec it
    launched (pins and device classes included): equal, bit for bit."""
    assert res.ok, res.reason
    spec = _m4_launched(res)
    ref = ensure_reference(JobSpec.from_json(spec.to_json()), str(tmp_path / "ref"))
    r = compare(ref, os.path.join(res.run_dir, "losses.json"))
    assert r["ok"] and r["compared"] == spec.steps * spec.microbatches, r
    state = fleet_manifest(c, res, tmp_path / "collected")
    reference = reference_manifest_for(spec, tmp_path / "ref")
    assert diff_manifests(state, reference) == []
    assert state == reference
    return state


def _m4_builds(res) -> int:
    return sum(M4_BUILT in e[2] for e in res.events)


def _m4_platform_refused(c, res) -> bool:
    """BA-463: where hfm.platform_admitted() is False the run is refused before
    placement with FP-TORCH-RECORDS naming the platform tag, and no slot is ever
    launched (strict, never a skip). -> True when it was (the arm ends there)."""
    from wantrain.model import hfm

    if hfm.platform_admitted():
        return False
    assert not res.ok, "a run on a platform without a TORCH_RECORDS entry was admitted"
    with open(os.path.join(res.run_dir, "hf_admission.json")) as f:
        tag = json.load(f)["builder"]["torch"]["platform_tag"]
    assert "FP-TORCH-RECORDS" in res.reason and tag in res.reason, res.reason
    assert not any("launching attempt" in e[2] for e in res.events), res.events[-10:]
    return True


def _m4_run_record(c, res) -> dict:
    """The run record of an hf run: the builder's RECORD text kept verbatim, the
    builder's fingerprint, one fingerprint and gate report per gated node (every
    worker of the cluster: slots and spares), each slot's class, and exactly one
    record builder child."""
    import hashlib

    from wantrain.model import hfm

    with open(os.path.join(res.run_dir, "hf_record.txt")) as f:
        kept = f.read()
    assert kept == hfm.record_text()
    with open(os.path.join(res.run_dir, "hf_admission.json")) as f:
        rec = json.load(f)
    sha = hashlib.sha256(kept.encode()).hexdigest()
    assert rec["hf_code"]["transformers_record_sha256"] == sha
    assert rec["builder"]["transformers"]["record_sha256"] == sha
    assert rec["builder"]["refusals"] == []
    names = {c.coord.workers[n].info.name for n in rec["fingerprints"]}
    assert names == set(c.workdirs), (names, sorted(c.workdirs))
    assert set(rec["reports"]) == set(rec["fingerprints"])
    for n, fp in rec["fingerprints"].items():
        assert fp["refusals"] == [] and fp["transformers"]["record_sha256"] == sha, n
        assert all(r["ok"] is True for r in rec["reports"][n]), rec["reports"][n]
    spec = _m4_launched(res)
    assert spec.model.hf_code == rec["hf_code"]
    want = [f"{r},{p}" for r, w in enumerate(spec.stage_widths) for p in range(w)]
    assert sorted(rec["slots"]) == sorted(want)
    classes = [
        [rec["slots"][f"{r},{p}"]["device_class"] for p in range(w)]
        for r, w in enumerate(spec.stage_widths)
    ]
    assert spec.device_classes == classes
    assert _m4_builds(res) == 1, [e[2] for e in res.events if "hf:" in e[2]]
    return rec


@pytest.mark.parametrize("fam", ["llama", "gemma2"])
def test_m4_coordinator_hf_spec_matches_the_oracle(tmp_path, port_base, monkeypatch, fam):
    """A tiny llama and a tiny gemma2 (tied embeddings, split across the 2
    stages) go through the record child, GATE on every node (the spare
    included), admission, placement, PROFILE and launch; the losses (hex) and
    the state manifest equal the oracle's."""
    # covers: CONTROL-I24, RUNTIME-I19
    # WOULD-FAIL-IF: an armed hf spec is refused or diverges from its oracle by one bit (losses
    # or state, the tied table split across stages included); the run record misses the RECORD
    # text, the builder's fingerprint or a gated node's; the buffer record is not pinned in the
    # launched spec; or the record builder child runs more than once
    _m4_offline(monkeypatch)
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(3):  # 2 slots + 1 spare, gated too
            c.add_worker(f"{fam[0]}{i}")
        c.wait_registered(3)
        recipe = _m4_hf_recipe(fam)
        res = c.run(Job(spec=recipe))
        if _m4_platform_refused(c, res):
            return
        assert res.ok, res.reason
        spec = _m4_launched(res)
        assert spec.model.hf_buffers and spec.model.hf_census, "the buffer record is not pinned"
        _m4_run_record(c, res)
        _m4_exact(c, res, tmp_path)
    finally:
        c.close()


def test_m4_coordinator_moe_spec_zero1_width2_matches_the_oracle(tmp_path, port_base):
    """A native MoE spec whose first stage is a ZeRO-1 stage of width 2 runs
    through the coordinator and equals its oracle: losses, state and the aux
    section (every (step, mb, layer) aux hex)."""
    # covers: CONTROL-I24, RUNTIME-I18
    # WOULD-FAIL-IF: an armed MoE spec is refused, or a ZeRO-1 width-2 MoE run moves a loss bit,
    # a state byte, a selection bias or one aux hex, or a slot's aux rows are lost
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(3):
            c.add_worker(f"m{i}")
        c.wait_registered(3)
        recipe = _m4_moe_recipe()
        res = c.run(Job(spec=recipe))
        assert res.ok, res.reason
        spec = _m4_launched(res)
        assert spec.stage_widths == [2, 1] and spec.shard == "zero1"
        state = _m4_exact(c, res, tmp_path)
        assert state.get("aux"), "the manifest has no aux section"
        assert any(k.endswith("sel_bias") for k in state["weights"])
    finally:
        c.close()


def test_m4_coordinator_moe_plan_keeps_every_moe_key_and_matches_the_oracle(tmp_path, port_base):
    """Plan mode: a plan dict with moe_experts > 0. The planned spec keeps every
    moe_* key the plan gives (BA-461) and the run equals its oracle with the aux
    hex."""
    # covers: CONTROL-I24, RUNTIME-I18
    # WOULD-FAIL-IF: the planner drops a moe_* key (a dense run of an MoE plan), or the planned
    # MoE run differs from its oracle in a loss bit, a state byte or an aux hex
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(3):
            c.add_worker(f"q{i}")
        c.wait_registered(3)
        plan = dict(M4_MOE_MODEL, model="toy", steps=4, mb_size=2, device="cpu")
        res = c.run(Job(plan=plan, stages=2, devices=3))
        assert res.ok, res.reason
        model = _m4_launched(res).model
        for k, v in M4_MOE_MODEL.items():
            assert getattr(model, k) == v, (k, getattr(model, k), v)
        state = _m4_exact(c, res, tmp_path)
        assert state.get("aux"), "the manifest has no aux section"
    finally:
        c.close()


def test_m4_coordinator_hf_llama_plan_mode_pins_the_builders_record(
    tmp_path, port_base, monkeypatch
):
    """Plan mode: a tiny llama plan goes through the record child (once), GATE,
    model_config and build_spec(hf_pins=...): the planned spec carries the
    builder's record, census and hf_code, and the run equals its oracle."""
    # covers: CONTROL-I24, RUNTIME-I19
    # WOULD-FAIL-IF: a plan-mode hf job is refused once armed, the planned spec does not pin
    # the builder's buffer record and hf_code (or pins another), the builder runs twice, or the
    # planned run differs from its oracle by one bit
    _m4_offline(monkeypatch)
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(2):
            c.add_worker(f"l{i}")
        c.wait_registered(2)
        hc = _m4_hf_config("llama")
        plan = {
            "model": "toy",
            "arch": "hf",
            "hf_config": hc,
            "seq_len": 16,
            "steps": 4,
            "mb_size": 2,
            "device": "cpu",
        }
        res = c.run(Job(plan=plan, stages=2))
        if _m4_platform_refused(c, res):
            return
        assert res.ok, res.reason
        rec = _m4_run_record(c, res)
        model = _m4_launched(res).model
        assert model.arch == "hf" and model.hf_config == json.loads(json.dumps(hc))  # JSON keys
        assert model.hf_code == rec["hf_code"] and model.hf_buffers and model.hf_census
        _m4_exact(c, res, tmp_path)
    finally:
        c.close()


def test_m4_coordinator_hf_plan_mode_checkpoint_with_a_stock_config(
    tmp_path, port_base, monkeypatch
):
    """Plan mode, fine-tune: the plan names only hf_model, a checkpoint directory whose
    config.json is a stock one (use_cache true, as every hub config says). The
    coordinator's record builder and gate read the config fleetplan adopts, the stages
    start, and the run equals its oracle."""
    # covers: CONTROL-I24, RUNTIME-I19
    # WOULD-FAIL-IF: the coordinator reads hf_config only from the plan dict (a checkpoint
    # plan is refused with "no hf_config dict"), the record is built for another config than
    # the one fleetplan adopts, the build keeps the stock use_cache true (every stage refuses
    # HF-ASSERT use_cache), or the run differs from its oracle by one bit
    from wantrain.model import hfm
    from wantrain.model.gpt import GPTConfig

    _m4_offline(monkeypatch)
    hc = hfm.tiny_config("llama")
    assert hc["use_cache"] is True
    units = hfm.build(GPTConfig(arch="hf", hf_config=hc, hf_buffers=hfm.buffer_record(hc)), 7)
    ckpt = tmp_path / "ckpt"
    hfm.write_safetensors(
        {n: p.detach().clone() for n, p in units.root.named_parameters()},
        str(ckpt),
        shard_bytes=64 * 1024,
    )
    (ckpt / "config.json").write_text(json.dumps(hc))
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(2):
            c.add_worker(f"k{i}")
        c.wait_registered(2)
        plan = {
            "model": "toy",
            "arch": "hf",
            "hf_model": str(ckpt),
            "seq_len": 16,
            "steps": 3,
            "mb_size": 2,
            "device": "cpu",
        }
        res = c.run(Job(plan=plan, stages=2))
        if _m4_platform_refused(c, res):
            return
        assert res.ok, res.reason
        model = _m4_launched(res).model
        assert model.hf_config == json.loads(json.dumps(hc)) and model.hf_model == str(ckpt)
        assert _m4_builds(res) == 1
        _m4_exact(c, res, tmp_path)
    finally:
        c.close()


def test_m4_coordinator_hf_llama_heal_onto_a_gated_spare_matches_the_oracle(
    tmp_path, port_base, monkeypatch
):
    """A worker of an hf llama run is killed mid-run: the slot heals onto the
    gated spare of the same device class from the common wave, without a second
    record build, and the healed run's manifest equals the oracle's."""
    # covers: CONTROL-I24, RUNTIME-I19
    # WOULD-FAIL-IF: an hf heal rebuilds the record, takes a spare whose gate did not report the
    # slot's recorded device class, restarts from step 0 instead of the common wave, or the
    # healed run differs from its oracle by one bit
    _m4_offline(monkeypatch)
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(3):  # 2 slots + 1 spare
            c.add_worker(f"h{i}")
        c.wait_registered(3)
        recipe = _m4_hf_recipe("llama", steps=12)
        c.run_async(Job(spec=recipe))
        if not _m4_wait_wave_or_refused(c, 3):
            return
        victim = c.name_of((0, 0))
        recorded = c.coord.run.hf["slots"][(0, 0)]["device_class"]
        c.kill_worker(victim)
        res = c.join()
        assert res.ok, res.reason
        assert len(res.heals) == 1 and res.heals[0]["from"] == victim, res.heals
        assert res.heals[0]["resumed_from_step"] is not None, res.heals
        assert res.heals[0]["resumed_from_step"] >= 3, res.heals
        rec = _m4_run_record(c, res)  # one builder child, the heal included
        assert rec["slots"]["0,0"]["device_class"] == recorded
        healed = c.coord.workers[rec["slots"]["0,0"]["node"]]
        assert healed.info.name == res.heals[0]["to"] != victim
        assert recorded in [r["device_class"] for r in rec["reports"][healed.node_id]]
        _m4_exact(c, res, tmp_path)
    finally:
        c.close()


def _m4_wait_wave_or_refused(c, step) -> bool:
    """Wait for a common wave >= step; on a platform without a TORCH_RECORDS
    entry the run ends refused before placement instead (asserted). -> True
    when the wave was reached."""
    from wantrain.model import hfm

    if not hfm.platform_admitted():
        res = c.join()
        assert _m4_platform_refused(c, res)
        return False
    _wait_common_wave(c, step)
    return True


@pytest.mark.parametrize("edited", [False, True], ids=["unedited", "edited"])
def test_m4_coordinator_hf_llama_resume_under_the_kept_record(
    tmp_path, port_base, monkeypatch, edited
):
    """A stalled hf llama run is resumed in place. Unedited, the resume is
    admitted and the run equals its oracle. With the kept RECORD text edited in
    one line (standing for another transformers build on the nodes), the resume
    is refused before any slot launches, naming that path."""
    # covers: CONTROL-I24, RUNTIME-I19
    # WOULD-FAIL-IF: a resume is admitted under another transformers RECORD (the readmission
    # compares only the sha256, or reads a reference other than the kept text), the refusal does
    # not name the differing path, a slot is relaunched after the refusal, or an unedited resume
    # is refused or moves a bit
    _m4_offline(monkeypatch)
    c = LocalCluster(tmp_path, port_base, probe=False, stall_floor_s=5.0)
    try:
        for i in range(2):
            c.add_worker(f"r{i}")
        c.wait_registered(2)
        recipe = _m4_hf_recipe("llama", steps=12)
        c.run_async(Job(spec=recipe))
        if not _m4_wait_wave_or_refused(c, 3):
            return
        run = c.coord.run
        kept = os.path.join(run.run_dir, "hf_record.txt")
        if edited:
            with open(kept) as f:
                lines = f.read().split("\n")
            (i,) = [k for k, line in enumerate(lines) if line.startswith(M4_MODELING_LLAMA + ",")]
            path, digest, size = lines[i].split(",")
            lines[i] = ",".join([path, digest[:-1] + ("A" if digest[-1] != "A" else "B"), size])
            with open(kept, "w") as f:
                f.write("\n".join(lines))
        c.freeze_stage(c.name_of((1, 0)))
        res = c.join()
        assert res.restarts == 1 and res.heals == [], (res.reason, res.restarts, res.heals)
        assert _m4_builds(res) == 1
        if edited:
            assert not res.ok
            assert "admission refused (CONTROL-I24)" in res.reason, res.reason
            assert M4_MODELING_LLAMA in res.reason, res.reason
            assert res.attempts == 1, res.attempts
            assert not any("launching attempt 2" in e[2] for e in res.events)
        else:
            assert res.ok, res.reason
            assert res.attempts == 2, res.attempts
            _m4_run_record(c, res)
            _m4_exact(c, res, tmp_path)
    finally:
        c.close()


# ---------------- RUNTIME-I26 / CONTROL-I27: re-cut and slow stages ----------------


import torch  # noqa: E402  (the re-cut tests build real waves)


def _rc_coordinator(tmp_path, spare=False, **cfg):
    """The _rb_coordinator FAKE boundary on a width-1 3-stage toy chain: a real Coordinator, fake requests
    (every one answered {"ok": True}: profiling fails over to the analytic schedule), no fit check, re-cuts
    worth any cost."""
    cfg.setdefault("replan_restart_cost_s", 0.1)
    cfg.setdefault("fit_check", False)
    cfg.setdefault("recut_cost_s", 0.0)
    coord = Coordinator(
        config=CoordinatorConfig(state_dir=str(tmp_path / "coord"), poll_s=0.01, **cfg)
    )
    sent = []

    def fake_request(w, msg, body=b"", timeout=None, file=None, reply_kind=None):
        sent.append((w.node_id, msg["kind"], dict(msg), body))
        return {"ok": True}

    coord._request = fake_request
    coord._sync_all_connections = lambda run: None
    os.makedirs(tmp_path / "run")
    run = _Run("run-rc", str(tmp_path / "run"))
    spec = _recipe(stages=3, wire_codec="raw", steps=24, optimizer="adamw-host")
    spec.finalize()
    run.spec, run.spec_json = spec, spec.to_json()
    slots = [(0, 0), (1, 0), (2, 0)]
    for i, slot in enumerate(slots + ([None] if spare else [])):
        info = _rb_node(i)
        w = Worker(info, _RbChannel(), f"session{i}", None, peer_id=f"{i:032x}")
        coord.workers[info.node_id] = w
        if slot is not None:
            w.run_id = run.run_id
            run.slots[slot] = info.node_id
    coord.run = run
    coord._launch(run)
    return coord, run, sent


def _rc_waves(run, spec, step):
    """Commit a real wave of `step` for every slot: the stage's toy chunk and its adamw-host state after one
    step, through the store's staging path."""
    import hashlib
    import io

    from wantrain.model.gpt import build_layers
    from wantrain.runtime import optim

    for r, (a, b) in enumerate(spec.partition):
        chunk = torch.nn.ModuleList(build_layers(spec.model, spec.init_seed, span=(a, b))[a:b])
        params = [p for p in chunk.parameters() if p.requires_grad]
        opt = optim.build(spec.optimizer, params, spec.lr)
        for p in params:
            p.grad = torch.ones_like(p)
        opt.step()
        payload = {
            "step": step,
            "chunks": {r: chunk.state_dict()},
            "opt": opt.state_dict(),
            "rng": torch.get_rng_state(),
            "losses": {f"{step}:0": 1.0} if r == spec.stages - 1 else {},
            "pins": spec.pins(),
        }
        buf = io.BytesIO()
        torch.save(payload, buf)
        data = buf.getvalue()
        body = os.path.join(run.run_dir, f"rc-r{r}-{step}.part")
        with open(body, "wb") as f:
            f.write(data)
        sha = hashlib.sha256(data).hexdigest()
        run.store.stage_part((r, 0), run.attempt, step, "state.pt", sha, body)
        m = {
            "v": 1,
            "step": step,
            "files": [{"name": "state.pt", "sha256": sha, "bytes": len(data)}],
        }
        assert run.store.commit_staged((r, 0), run.attempt, step, m) is True


def _rc_telem(coord, run, steps, slow=None):
    """One TELEM_STEP per slot and step; slow: {slot: factor}."""
    spec = coord._owned_spec(run)
    for step in steps:
        for slot in run.slots:
            summary = _rb_summary(spec, slot, step, (slow or {}).get(slot, 1.0))
            summary["hops"] = {}
            msg = {
                "kind": "TELEM_STEP",
                "run_id": run.run_id,
                "attempt": run.attempt,
                "summary": summary,
            }
            coord._dispatch(coord.workers[run.slots[slot]], msg, b"")


def test_a_slow_stage_is_re_cut_from_the_measured_costs(tmp_path):
    """RUNTIME-I26 / CONTROL-I21: a window where stage 1 computes 4x slower re-cuts the chain at the common
    wave: stage 1 loses entries, every slot's wave is converted in the store under the new cut, and the relaunch
    restores and resumes it under a new tag; with recut off nothing is re-cut."""
    # covers: RUNTIME-I26
    # WOULD-FAIL-IF: the re-cut kept the old cut, moved entries onto the slow stage, left the store's waves in the
    # old layout or with the old pins, relaunched under the old tag (a worker holds a different wave of the same
    # step under it), or fired with recut off
    coord, run, sent = _rc_coordinator(tmp_path / "a")
    old = coord._owned_spec(run)
    _rc_waves(run, old, 3)
    _rc_telem(coord, run, range(4), slow={(1, 0): 4.0})
    assert coord._maybe_rebalance(run) is True
    rec = run.replans[-1]
    assert rec["recut"] is True and rec["from"] == old.partition and rec["step"] == 3
    new = rec["to"]
    assert new[1][1] - new[1][0] < old.partition[1][1] - old.partition[1][0]
    assert rec["tag"].endswith(".cut1") and any(e[2].startswith("RECUT {") for e in coord.events)
    for r in range(3):
        st = torch.load(os.path.join(run.store.wave_path((r, 0), 3), "state.pt"), weights_only=True)
        assert st["pins"]["partition"] == new and set(st["chunks"]) == {r}
        a, b = new[r]
        assert {k.split(".", 1)[0] for k in st["chunks"][r]} == {str(j) for j in range(b - a)}
    restores = [m for _n, kind, m, _b in sent if kind == "RESTORE"]
    assert restores and all(m["tag"] == rec["tag"] and m["step"] == 3 for m in restores)
    assigns = _rb_assigns(sent, run.attempt)
    assert len(assigns) == 3
    for _m, body in assigns:
        got = json.loads(body)
        assert got["partition"] == new and got["tag"] == rec["tag"] and got["resume_step"] == 3

    coord2, run2, _ = _rc_coordinator(tmp_path / "b", recut=False)
    _rc_waves(run2, coord2._owned_spec(run2), 3)
    _rc_telem(coord2, run2, range(4), slow={(1, 0): 4.0})
    coord2._maybe_rebalance(run2)
    assert not any(r.get("recut") for r in run2.replans)


def test_a_stage_that_slows_down_is_replaced_by_a_spare_or_re_cut_away_from(tmp_path):
    # covers: CONTROL-I27
    """CONTROL-I27: a slot whose compute per microbatch stays above 1.3x its own baseline for three summaries
    is healed onto an idle spare of the run's class; with no spare the chain is re-cut away from it. A slot slow
    from its first summary is its own baseline and stays."""
    # covers: RUNTIME-I26
    # WOULD-FAIL-IF: a stage that was always slower than its peers were flagged, the baseline counted the
    # warm-up summaries, a slowdown with a spare re-cut instead of healing, or one without a spare did nothing
    coord, run, _ = _rc_coordinator(tmp_path / "a", spare=True)
    _rc_waves(run, coord._owned_spec(run), 3)
    _rc_telem(coord, run, range(2), slow={(1, 0): 5.0})  # warm-up: ignored
    _rc_telem(
        coord, run, range(2, 6), slow={(2, 0): 3.0}
    )  # stage 2 slower from the start: its own baseline
    assert coord._watch_slow(run) is False
    _rc_telem(coord, run, range(6, 9), slow={(1, 0): 2.0, (2, 0): 3.0})
    assert coord._watch_slow(run) is True
    assert len(run.heals) == 1 and run.slots[(1, 0)] == "n3"  # the spare

    coord2, run2, _ = _rc_coordinator(tmp_path / "b")
    _rc_waves(run2, coord2._owned_spec(run2), 3)
    _rc_telem(coord2, run2, range(6))
    _rc_telem(coord2, run2, range(6, 9), slow={(1, 0): 4.0})
    assert coord2._watch_slow(run2) is True
    assert run2.replans and run2.replans[-1]["recut"] is True
    assert run2.heals == []
