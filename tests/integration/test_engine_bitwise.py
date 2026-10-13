"""Zero-bubble pipeline parallelism, run as real multi-process pipelines
over TCP, must leave the per-microbatch loss sequence bit-identical to the
single-device reference (max abs diff == 0): across chain lengths, WAN
latency and jitter, recomputation, every wire codec, adaptive microbatching,
parallel streams, wide ZeRO-1 stages with weighted microbatch ownership,
streamed shard gradients, any solved schedule, and a checkpoint resume.

WOULD-FAIL-IF: the schedule, deferred dW, ownership or transport changed
the order or the bits of a gradient sum, a wire codec were lossy, or a
shutdown race killed a slower rank.
"""

import json
import os
import queue
import socket
import statistics
import subprocess
import sys
import threading
import time

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from helpers.exact import reference_manifest_for

import wantrain
from wantrain.launch.local import (
    ensure_reference,
    jitter_seed_from_env,
    links_for,
    run_and_compare,
    run_local,
)
from wantrain.model.gpt import make_config
from wantrain.runtime.spec import JobSpec
from wantrain.validation.compare import compare
from wantrain.validation.state import diff_manifests, distributed_manifest

pytestmark = pytest.mark.slow

CONFIGS = {
    "chain2": (dict(stages=2), {}),
    "chain3_jitter": (dict(stages=3), dict(latency_ms=20, jitter_ms=40)),
    "chain4_recompute": (dict(stages=4, recompute=True), dict(latency_ms=10, jitter_ms=25)),
    "fp8_wire": (dict(stages=2, wire_codec="fp8_planes", fp8=True), {}),
    "planes_wire": (
        dict(stages=3, recompute=True, wire_codec="planes"),
        dict(latency_ms=25, jitter_ms=10),
    ),
    "adaptive_mb": (dict(stages=2, adapt_budget_ms=8), dict(bw_mbps=40)),
    "multistream": (dict(stages=2, p2p_streams=3, wire_codec="planes"), dict(latency_ms=10)),
    "wide_2_1": (dict(stages=2, stage_widths=[2, 1]), {}),
    "wide_1_2": (dict(stages=2, stage_widths=[1, 2]), {}),
    "wide_2_2_jitter": (dict(stages=2, stage_widths=[2, 2]), dict(latency_ms=10, jitter_ms=20)),
    "wide_1_2_1_recompute": (dict(stages=3, stage_widths=[1, 2, 1], recompute=True), {}),
    "wide_iwrr_weighted": (
        dict(stages=2, stage_widths=[3, 1], peer_weights=[[1.0, 0.5, 0.25], [1.0]]),
        dict(latency_ms=5),
    ),
    "stream_grads": (dict(stages=2, stage_widths=[2, 1], stream_grads=True), {}),
    "strict_order": (dict(stages=3, schedule_fill=False), dict(latency_ms=15, jitter_ms=20)),
    # RUNTIME-I25: a stage that keeps F's graph instead of recomputing gives the same bits
    "recompute_last_keeps_graph": (
        dict(stages=3, recompute=True, recompute_stages=[True, True, False]),
        dict(latency_ms=5),
    ),
    "recompute_mixed": (
        dict(
            stages=3, recompute=True, recompute_stages=[False, True, False], optimizer="adamw-host"
        ),
        dict(latency_ms=10, jitter_ms=20),
    ),
    # RUNTIME-I20: the moments in host memory; the reference builds the same optimizer
    "host_adamw": (
        dict(stages=3, recompute=True, optimizer="adamw-host"),
        dict(latency_ms=10, jitter_ms=20),
    ),
}


def _spec(tmp_path, name, port_base, kw, link, steps=5, microbatches=8):
    kw = dict(kw)
    fp8 = kw.pop("fp8", False)
    spec = JobSpec(
        model=make_config("toy", fp8_boundaries=fp8),
        steps=steps,
        microbatches=microbatches,
        out_dir=str(tmp_path),
        tag=name,
        port_base=port_base,
        dial_timeout_s=60,
        **kw,
    )
    spec.links = links_for(spec, **link)
    return spec


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_mechanism_is_bitwise_identical_to_the_reference(tmp_path, port_base, name):
    # covers: RUNTIME-I1, RUNTIME-I7
    kw, link = CONFIGS[name]
    spec = _spec(tmp_path, name, port_base, kw, link)
    res = run_and_compare(spec, timeout=300)
    assert res.get("ok"), json.dumps(res)
    assert res["compared"] == spec.steps * spec.microbatches
    assert res["max_abs_diff"] == 0.0


def test_checkpoint_resume_continues_the_sequence_bit_for_bit(tmp_path, port_base):
    """Train 3 of 6 steps, stop, resume from the shards: the full loss sequence
    equals an uninterrupted 6-step run (the basis of every heal)."""
    # covers: RUNTIME-I3
    kw = dict(stages=2, recompute=True)
    full = _spec(tmp_path, "full", port_base, kw, {}, steps=6)
    assert run_local(full, timeout=300)["ok"]
    part_dir = tmp_path / "part"
    part = _spec(part_dir, "part", port_base + 16, kw, {}, steps=3)
    part.ckpt_every = 1
    assert run_local(part, timeout=300)["ok"]
    resumed = _spec(part_dir, "part", port_base + 32, kw, {}, steps=6)
    resumed.resume = True
    assert run_local(resumed, timeout=300)["ok"]
    r = compare(
        os.path.join(str(tmp_path), "full_losses.json"),
        os.path.join(str(part_dir), "part_losses.json"),
    )
    assert r["ok"] and r["compared"] == 6 * 8, r


def test_a_boundary_spread_over_every_flow_is_bitwise(tmp_path, port_base, monkeypatch):
    """TRANSPORT-I12: each boundary cut into one fragment per flow (4 flows, a 1 KiB floor) under latency and jitter,
    reassembled by index: the losses equal the single-device reference."""
    # covers: TRANSPORT-I12
    # WOULD-FAIL-IF: fragments of one message were reassembled out of index order, one were dropped or doubled across
    # flows, or a message of another step or microbatch took a fragment
    monkeypatch.setenv("WANTRAIN_TEST_MIN_FRAG", "1024")
    kw = dict(stages=3, p2p_streams=4, wire_codec="planes")
    spec = _spec(tmp_path, "spread", port_base, kw, dict(latency_ms=10, jitter_ms=15))
    res = run_and_compare(spec, timeout=300)
    assert res.get("ok"), json.dumps(res)
    assert res["compared"] == spec.steps * spec.microbatches and res["max_abs_diff"] == 0.0
    frags = set()
    for f in os.listdir(tmp_path):
        if f.endswith("telemetry.jsonl"):
            for line in open(os.path.join(tmp_path, f)):
                x = json.loads(line)
                if x.get("ev") == "encode":
                    frags.add(x.get("frags"))
    assert max(frags) >= 4, frags  # each boundary did cross several flows


def test_a_w_split_into_sub_blocks_while_filling_waits_is_bitwise(tmp_path, port_base, monkeypatch):
    """RUNTIME-I27: under latency and jitter the stages fill their waits with W run one gradient at a time (a 0 s
    slice), the inbox read between; the losses equal the single-device reference, and the W were split."""
    # covers: RUNTIME-I27
    # WOULD-FAIL-IF: a stopped W resumed at the wrong gradient, folded one twice or not at all, or a partially run W
    # left the order (the step ends with W pending or a parameter without its gradient)
    monkeypatch.setenv("WANTRAIN_TEST_W_SLICE_S", "0")
    kw = dict(stages=3, recompute=True, optimizer="adamw-host")
    spec = _spec(tmp_path, "wslice", port_base, kw, dict(latency_ms=15, jitter_ms=25))
    res = run_and_compare(spec, timeout=300)
    assert res.get("ok"), json.dumps(res)
    assert res["compared"] == spec.steps * spec.microbatches and res["max_abs_diff"] == 0.0
    slices = 0
    for f in os.listdir(tmp_path):
        if f.endswith(".log"):
            for line in open(os.path.join(tmp_path, f), errors="replace"):
                if line.startswith("WT_WFILL "):
                    slices += int(line.split("slices=")[1])
    assert slices > 0, "no W was split: the arm did not exercise RUNTIME-I27"


@pytest.mark.parametrize("optimizer", ["adamw", "adamw-host"])
def test_recut_resume_continues_the_sequence_bit_for_bit(tmp_path, port_base, optimizer):
    """RUNTIME-I26: 3 of 6 steps under one cut, the waves re-cut offline to another (layers moved both ways), the
    rest resumed under the new cut: the full loss sequence equals an uninterrupted 6-step run."""
    # covers: RUNTIME-I26
    from wantrain.fault_tolerance.checkpoint import WAVE_STATE, commit_wave, wave_dir
    from wantrain.runtime.recut import entry_param_names, recut_states

    kw = dict(stages=3, recompute=True, optimizer=optimizer)
    full = _spec(tmp_path, "full", port_base, kw, {}, steps=6)
    assert run_local(full, timeout=300)["ok"]
    part_dir = tmp_path / "part"
    part = _spec(part_dir, "part", port_base + 16, kw, {}, steps=3)
    part.ckpt_every = 1
    assert run_local(part, timeout=300)["ok"]
    part.finalize()
    new_cut = [[0, 2], [2, 6], [6, 8]]
    assert part.partition == [[0, 3], [3, 5], [5, 8]]
    states = [
        torch.load(
            os.path.join(wave_dir(str(part_dir), "part", r, 0, 2), WAVE_STATE),
            map_location="cpu",
            weights_only=True,
        )
        for r in range(3)
    ]
    for r, payload in enumerate(
        recut_states(states, part.partition, new_cut, entry_param_names(part.model))
    ):
        commit_wave(str(part_dir), "part.cut", r, 0, 2, lambda path, x=payload: torch.save(x, path))
    resumed = _spec(part_dir, "part.cut", port_base + 32, dict(kw, partition=new_cut), {}, steps=6)
    resumed.resume = True
    assert run_local(resumed, timeout=300)["ok"]
    r = compare(
        os.path.join(str(tmp_path), "full_losses.json"),
        os.path.join(str(part_dir), "part.cut_losses.json"),
    )
    assert r["ok"] and r["compared"] == 6 * 8, r


def test_host_adamw_resume_continues_the_sequence_bit_for_bit(tmp_path, port_base):
    """RUNTIME-I20: a resume reloads host-resident moments from the shards and
    continues the uninterrupted run's loss sequence exactly; RUNTIME-I23: with
    the optimizer overlapped with the next step, joined before each checkpoint."""
    # covers: RUNTIME-I3, RUNTIME-I20, RUNTIME-I23
    kw = dict(stages=2, recompute=True, optimizer="adamw-host")
    full = _spec(tmp_path, "full", port_base, kw, {}, steps=6)
    assert run_local(full, timeout=300)["ok"]
    part_dir = tmp_path / "part"
    part = _spec(part_dir, "part", port_base + 16, kw, {}, steps=3)
    part.ckpt_every = 1
    assert run_local(part, timeout=300)["ok"]
    resumed = _spec(part_dir, "part", port_base + 32, kw, {}, steps=6)
    resumed.resume = True
    assert run_local(resumed, timeout=300)["ok"]
    r = compare(
        os.path.join(str(tmp_path), "full_losses.json"),
        os.path.join(str(part_dir), "part_losses.json"),
    )
    assert r["ok"] and r["compared"] == 6 * 8, r
    # RUNTIME-I23: the step's AdamW ran beside the next step on every stage, not before it
    for rank in range(2):
        log = open(os.path.join(str(tmp_path), f"full_r{rank}p0.log"), errors="replace").read()
        assert "WT_OPT overlap" in log, f"rank {rank} stepped without the overlap"


def test_resume_with_new_ownership_is_byte_identical(tmp_path, port_base):
    """A 1-2-1 run checkpointed at step 3 under equal peer weights, resumed
    under weights [1, 3] on the wide stage (a rebalance: new microbatch owners
    and new orders from reschedule) and extended to 6 steps, leaves the loss
    sequence and the final training state of the 6-step reference."""
    # covers: RUNTIME-I15
    # WOULD-FAIL-IF: a resume under new ownership changes the fold order or a ZeRO-1 shard's
    # owner, the pins refuse a rebalance or an extended run, or the resumed slots load
    # another slot's optimizer state or RNG, or a slot skips its shard and retrains from step 0.
    kw = dict(stages=3, stage_widths=[1, 2, 1])
    part = _spec(tmp_path, "own", port_base, kw, {}, steps=4).finalize()
    assert run_local(part, timeout=300)["ok"]
    resumed = JobSpec.from_json(part.to_json())
    resumed.steps = 6
    resumed.resume = True
    resumed.port_base = port_base + 32
    resumed.peer_weights = [[1.0], [1.0, 3.0], [1.0]]
    resumed.reschedule()
    assert resumed.mb_owner[1] != part.mb_owner[1]
    assert resumed.schedule_order[1] != part.schedule_order[1]
    assert resumed.pins() == part.pins()
    res = run_and_compare(resumed, timeout=300)
    assert res.get("ok"), json.dumps(res)
    assert res["compared"] == 6 * 8 and res["max_abs_diff"] == 0.0, res
    # A run that ignored its shards and retrained from step 0 would match the
    # reference too: every slot must have loaded its step-3 shard.
    for r, w in enumerate(resumed.stage_widths):
        for p in range(w):
            with open(tmp_path / f"own_r{r}p{p}.log", errors="replace") as f:
                assert f"WT_RESUME rank={r} start_step=4 " in f.read(), (r, p)
    _assert_state_is_the_reference(tmp_path, resumed)


@pytest.mark.parametrize("kind", ["paper_2p", "profiled", "heterogeneous"])
def test_solved_schedules_are_bitwise_identical(tmp_path, port_base, kind):
    """Whatever the scheduler solves (uniform costs at ZB-2p memory, costs
    profiled on this machine, or a deliberately lopsided heterogeneous cost
    model), the bits do not move: only the time does."""
    # covers: RUNTIME-I2
    from wantrain.resources.profile import profile_stage
    from wantrain.scheduler.zero_bubble import ZBConfig, schedule_for_spec, zero_bubble

    kw = dict(stages=3, recompute=True, wire_codec="planes")
    spec = _spec(tmp_path, kind, port_base, kw, dict(latency_ms=15, jitter_ms=20)).finalize()
    p, m = spec.stages, spec.microbatches
    if kind == "profiled":
        profs = [[profile_stage(JobSpec.from_json(spec.to_json()), r, reps=1)] for r in range(p)]
        schedule_for_spec(spec, profs)
    else:
        if kind == "paper_2p":
            cfg = ZBConfig(p, m, [1.0] * p, [1.0] * p, [1.0] * p, comm=0.5)
        else:
            cfg = ZBConfig(p, m, [0.2, 3.0, 1.0], [0.3, 4.0, 1.2], [0.1, 2.5, 0.7], comm=[2.0, 0.1])
        spec.schedule_order = [[[[t, mb] for t, mb in o]] for o in zero_bubble(cfg, "2p").orders]
    res = run_and_compare(spec, timeout=300)
    assert res.get("ok"), json.dumps(res)
    assert res["max_abs_diff"] == 0.0


def _telem_summaries(spec, r, p):
    """The WT_TELEM summaries slot (r, p) printed, from its stage log."""
    out = []
    with open(os.path.join(spec.out_dir, f"{spec.tag}_r{r}p{p}.log")) as f:
        for line in f:
            if line.startswith("WT_TELEM "):
                out.append(json.loads(line[len("WT_TELEM ") :]))
    return out


def _telem_events(spec, r, p):
    """The slot's event log: every event a summary took, plus the shutdown drain."""
    with open(os.path.join(spec.out_dir, f"{spec.tag}_rank{r}p{p}_telemetry.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _hop_link(rank, hop):
    """Index into spec.links of the link a hop's sends are paced on: next
    from stage r crosses links[r], prev crosses links[r - 1]; intra is unpaced."""
    if hop.startswith("next"):
        return rank
    if hop.startswith("prev"):
        return rank - 1
    return None


def _check_slot_telemetry(spec, r, p, rates, streams):
    """Ops complete and per-hop frame accounting for one slot. Appends
    (link index, Mb/s) of every paced hop sample to rates; returns the
    summaries and the event log."""
    sums = _telem_summaries(spec, r, p)
    assert [s["step"] for s in sums] == list(range(spec.steps)), (r, p, len(sums))
    # What the slot's schedule runs per step, from the spec, never the events.
    want = {op: sum(1 for o, _ in spec.schedule_order[r][p] if o == op) for op in "FBW"}
    assert want["F"] == len(spec.owned(r, p)), (r, p, want)
    for s in sums:
        assert (s["v"], s["rank"], s["peer"]) == (1, r, p), s
        for op in "FBW":
            assert s["ops"][op]["n"] == want[op], (r, p, op, s)
        assert s["ops"]["OPT"]["n"] == 1, (r, p, s)
        assert all(s["ops"][op]["med_ms"] > 0 for op in ("F", "B", "W", "OPT")), (r, p, s)
        for hop, h in s["hops"].items():
            link = _hop_link(r, hop)
            if link is None or h["busy_ms"] <= 0:
                continue
            bw = spec.links[link]["bw_mbps"]
            rate = h["bytes"] * 8 / (h["busy_ms"] * 1e3)
            assert rate <= 1.1 * streams * bw, (r, p, hop, rate, bw)
            rates.append((link, rate))
    events = _telem_events(spec, r, p)
    sends = [e for e in events if e["ev"] == "send"]
    assert sends and all("hop" in e for e in sends), (r, p)
    for hop in {e["hop"] for e in sends} | {h for s in sums for h in s["hops"]}:
        early = sum(1 for e in sends if e["hop"] == hop and e["step"] <= spec.steps - 2)
        framed = sum(s["hops"].get(hop, {}).get("frames", 0) for s in sums)
        total = sum(1 for e in sends if e["hop"] == hop)
        assert early <= framed <= total, (r, p, hop, early, framed, total)
    return sums, events


def _rate_ratio(rates):
    """Median wire rate on links[1] over the median on links[0]."""
    slow = [x for link, x in rates if link == 0]
    fast = [x for link, x in rates if link == 1]
    assert slow and fast, rates
    return statistics.median(fast) / statistics.median(slow)


def _assert_state_is_the_reference(tmp_path, spec):
    state = distributed_manifest(spec)
    reference = reference_manifest_for(spec, tmp_path / "ref")
    assert diff_manifests(state, reference) == []
    assert state == reference


def test_step_telemetry_measures_ops_and_hops(tmp_path, port_base, monkeypatch):
    """Every slot prints one WT_TELEM per step whose op counts match its
    schedule and whose per-hop wire rates follow the emulated links, while the
    training state stays the reference's. Mesh arm: wide stage 1 of two peers,
    peer 1 slowed 4x by the seam. Multi-stream arm: a chain with two parallel
    streams per hop, where fragment senders, not the encoder, time the wire."""
    # covers: RUNTIME-I9
    # WOULD-FAIL-IF: the multi-stream encoder still emits "send" around encode + enqueue (its
    # rate ratio is ~1 and the send count differs from the fragment count), a sender loses its
    # hop label, summarize_step maps an event name wrongly (n 0 or med_ms 0), sends that
    # complete after their step's summary are dropped, the SIM_SLOW seam misses its slot, or
    # telemetry moves one bit of the training state.
    checked = 0

    monkeypatch.setenv("WANTRAIN_SIM_SLOW", "1:1:4")
    mesh = _spec(
        tmp_path / "mesh",
        "mesh",
        port_base,
        dict(stages=3, stage_widths=[1, 2, 1], wire_codec="raw"),
        dict(bw_mbps=40),
        steps=4,
    )
    mesh.links[1]["bw_mbps"] = 160.0
    res = run_local(mesh, timeout=300)
    assert res["ok"], res
    mesh.finalize()
    rates, f_med = [], {}
    for r, w in enumerate(mesh.stage_widths):
        for p in range(w):
            sums, _ = _check_slot_telemetry(mesh, r, p, rates, 1)
            f_med[(r, p)] = statistics.median(s["ops"]["F"]["med_ms"] for s in sums)
            checked += len(sums)
    _assert_state_is_the_reference(tmp_path, mesh)
    seam_ratio = f_med[(1, 1)] / f_med[(1, 0)]
    hop_ratio = _rate_ratio(rates)
    print(f"SEAM_RATIO={seam_ratio:.6f}")
    print(f"HOP_RATE_RATIO={hop_ratio:.6f}")
    assert seam_ratio >= 2, f_med
    assert hop_ratio >= 2.5, rates

    monkeypatch.delenv("WANTRAIN_SIM_SLOW")
    multi = _spec(
        tmp_path / "multi",
        "multi",
        port_base + 32,
        dict(stages=3, p2p_streams=2, wire_codec="raw"),
        dict(bw_mbps=40),
        steps=4,
    )
    multi.links[1]["bw_mbps"] = 160.0
    res = run_local(multi, timeout=300)
    assert res["ok"], res
    multi.finalize()
    rates = []
    for r in range(multi.stages):
        sums, events = _check_slot_telemetry(multi, r, 0, rates, multi.p2p_streams)
        frags = sum(e["frags"] for e in events if e["ev"] == "encode")
        sends = sum(1 for e in events if e["ev"] == "send")
        assert sends == frags, (r, sends, frags)
        checked += len(sums)
    _assert_state_is_the_reference(tmp_path, multi)
    multi_ratio = _rate_ratio(rates)
    print(f"HOP_RATE_RATIO_MULTI={multi_ratio:.6f}")
    print(f"OPS_CHECKED={checked:.6f}")
    # ADR-0005-m2-p1-measurement-bounds: >= 2.0, not 2.5. The emulation's per-frame cost dilutes the 4x link ratio
    # (measured 2.42-2.48); an encode-timed `send` would give ~1, so 2.0 still fails that mistake.
    assert multi_ratio >= 2.0, rates


# ---------------- RUNTIME-I16: every data-plane socket bound and sealed ----------------


def _key(path):
    """The peer id of the key at path (created on first use)."""
    from wantrain.transport.identity import load_or_create_key, peer_id

    k = load_or_create_key(str(path))
    return peer_id(
        k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )


class _Fleet:
    """Every slot of a pinned local-mode spec as its own process, each proving
    its own key (WANTRAIN_PEER_KEY), started in an order the test chooses, so
    an intruder can reach a listener before the real peer does."""

    def __init__(self, tmp_path, spec):
        self.spec = spec
        self.tmp_path = tmp_path
        self.keys = {}
        spec.finalize()
        for r, w in enumerate(spec.stage_widths):
            for p in range(w):
                self.keys[(r, p)] = str(tmp_path / "keys" / f"r{r}p{p}")
        self.ids = {slot: _key(path) for slot, path in self.keys.items()}
        spec.slot_peers = [
            [self.ids[(r, p)] for p in range(w)] for r, w in enumerate(spec.stage_widths)
        ]
        spec.trusted_peers = sorted(set(self.ids.values()))
        spec.finalize()
        os.makedirs(spec.out_dir, exist_ok=True)
        self.spec_path = os.path.join(spec.out_dir, f"{spec.tag}_spec.json")
        with open(self.spec_path, "w") as f:
            f.write(spec.to_json())
        self.procs = {}

    def port(self, r, p):
        return self.spec.port_base + r * 8 + p

    def log_path(self, r, p):
        return os.path.join(self.spec.out_dir, f"{self.spec.tag}_r{r}p{p}.log")

    def start(self, *slots):
        src = os.path.dirname(os.path.dirname(os.path.abspath(wantrain.__file__)))
        for r, p in slots:
            env = dict(os.environ, PYTHONPATH=src, PYTHONUNBUFFERED="1")
            env["WANTRAIN_PEER_KEY"] = self.keys[(r, p)]
            with open(self.log_path(r, p), "wb") as log:
                self.procs[(r, p)] = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "wantrain.worker.stage_main",
                        "--spec",
                        self.spec_path,
                        "--rank",
                        str(r),
                        "--peer",
                        str(p),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

    def start_rest(self):
        self.start(*[slot for slot in self.keys if slot not in self.procs])

    def lines(self, r, p, prefix):
        try:
            with open(self.log_path(r, p), errors="replace") as f:
                return [ln.rstrip("\n") for ln in f if ln.startswith(prefix)]
        except FileNotFoundError:
            return []

    def wait_line(self, r, p, prefix, timeout=60.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            got = self.lines(r, p, prefix)
            if got:
                return got
            time.sleep(0.05)
        raise AssertionError(f"slot {(r, p)} printed no {prefix!r} within {timeout}s")

    def finish(self, timeout=300.0):
        """Wait for every slot; the run must end cleanly, and its losses and
        final state must be the oracle's, byte for byte."""
        deadline = time.monotonic() + timeout
        try:
            for slot, proc in self.procs.items():
                rc = proc.wait(timeout=max(1.0, deadline - time.monotonic()))
                assert rc == 0, (slot, rc, self.lines(*slot, ""))
        finally:
            self.kill()
        ref = ensure_reference(self.spec, self.spec.out_dir)
        cmp = compare(ref, os.path.join(self.spec.out_dir, f"{self.spec.tag}_losses.json"))
        assert cmp["ok"] and cmp["max_abs_diff"] == 0.0, cmp
        assert cmp["compared"] == self.spec.steps * self.spec.microbatches, cmp
        _assert_state_is_the_reference(self.tmp_path / self.spec.tag, self.spec)

    def kill(self):
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def events(self, r, p, ev):
        return [e for e in _telem_events(self.spec, r, p) if e["ev"] == ev]


def _pinned(tmp_path, name, port_base, **kw):
    kw.setdefault("steps", 3)
    kw.setdefault("microbatches", 4)
    steps, mbs = kw.pop("steps"), kw.pop("microbatches")
    return _spec(tmp_path / name, name, port_base, kw, {}, steps=steps, microbatches=mbs)


def _connect(port, timeout=60.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=timeout)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def _closed(read):
    """Read until the acceptor closes: 'eof' (clean close) or 'reset'."""
    try:
        while True:
            if not read():
                return "eof"
    except TimeoutError:
        # Never closed: an acceptor that logs a refusal but keeps the socket
        # open must fail the caller's close check, not pass it as a reset.
        return "timeout"
    except (ConnectionError, OSError):
        return "reset"


def _intrude(port, key_path=None, hello=None, garbage=False):
    """One rogue connection to a listener. garbage: a malformed handshake.
    Otherwise a full handshake under key_path (an id of the intruder's
    choosing), then, with hello, a sealed HELLO claiming that slot. Returns
    how the acceptor ended the connection ('eof' or 'reset')."""
    from wantrain.transport import framing as fr
    from wantrain.transport.identity import authenticate
    from wantrain.transport.secure import SecureSocket

    sock = _connect(port)
    try:
        if garbage:
            sock.sendall(b"\x05\x00\x00\x00{oops")
            return _closed(lambda: sock.recv(1 << 16))
        auth = authenticate(sock, key_path)
        ss = SecureSocket(
            sock,
            auth["cipher"],
            auth["send_key"],
            auth["recv_key"],
            auth["send_prefix"],
            auth["recv_prefix"],
        )
        if hello is not None:
            try:
                fr.send_frame(ss, fr.HELLO, 0, 0, hello)
            except OSError:
                return "reset"
        return _closed(lambda: ss.recv(1 << 16))
    finally:
        sock.close()


def _refusals(fleet, r, p):
    out = []
    for ln in fleet.lines(r, p, "WT_P2P_REFUSED "):
        out.append(dict(kv.split("=", 1) for kv in ln.split()[1:]))
    return out


def _assert_refused(fleet, r, p, reason, peer="-", st="-", pe="-", how=None):
    """The intruder's connection was accepted and refused: its WT_P2P_REFUSED
    line exists (fails when no intruder engaged) and it saw the close."""
    want = dict(reason=reason, st=str(st), pe=str(pe), peer=peer)
    deadline = time.monotonic() + 60
    while want not in _refusals(fleet, r, p) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert want in _refusals(fleet, r, p), (want, _refusals(fleet, r, p))
    if how is not None:
        assert how in ("eof", "reset"), how


def _assert_bound_once(fleet, r, p, peers):
    """Exactly one p2p_auth and one p2p_connected per legitimate socket, each
    naming the neighbour's pinned id, the mode and the cipher."""
    for ev in ("p2p_auth", "p2p_connected"):
        got = fleet.events(r, p, ev)
        assert sorted(e["peer"] for e in got) == sorted(peers), (r, p, ev, got)
        for e in got:
            assert e["mode"] == "ed25519", e
            assert e["cipher"] in ("aes-256-gcm", "chacha20-poly1305"), e


def _raised(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 -- the test inspects it
        return e
    return None


def test_unpinned_data_plane_is_refused(tmp_path, port_base):
    """A spec with endpoints and no slot_peers is refused at connect, before
    any socket opens; local mode (endpoints None) needs no pins."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: the stage connects a coordinator-run spec whose slots are not pinned (any
    # trusted key could then take any slot), or refuses unpinned local mode.
    from wantrain.runtime.stage import StageRuntime

    spec = _pinned(tmp_path, "unpinned", port_base)
    spec.endpoints = [[["127.0.0.1", port_base, port_base]], [["127.0.0.1", port_base + 8]]]
    st = StageRuntime(spec, 1, 0)
    # On a thread with a bound: a stage that wrongly connects waits on its
    # listener forever, and that must fail here, not hang the suite.
    err = []
    t = threading.Thread(target=lambda: err.append(_raised(st._connect)), daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "an unpinned spec with endpoints reached its accept loop"
    assert isinstance(err[0], ConnectionError) and "unpinned data plane" in str(err[0]), err
    with socket.socket() as s:  # nothing was bound: the refusal came first
        s.bind(("127.0.0.1", port_base))
    single = JobSpec(model=make_config("toy"), stages=1, out_dir=str(tmp_path), tag="one")
    StageRuntime(single, 0, 0)._connect()  # local mode, no pins: nothing to refuse


class _AbortOnce:
    """Fake at the OS boundary: a listener whose first accept() fails the way
    Linux reports a connection reset before accept (ECONNABORTED)."""

    def __init__(self, inner):
        self.inner, self.done = inner, False

    def accept(self):
        if not self.done:
            self.done = True
            raise ConnectionAbortedError(53, "Software caused connection abort")
        return self.inner.accept()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_acceptor_survives_resets_and_a_hostile_handshake(tmp_path):
    """Keyless hosts cannot end a stage's accept loop: a connection reset before
    accept, one reset right after it (TCP_NODELAY then fails with EINVAL on
    macOS) and a handshake nested so deep that json.loads raises RecursionError
    are each refused and closed, and the real peer still binds."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: an accept() error or a failing setsockopt on a reset socket escapes the accept
    # loop (one RST from any host ends the stage), or a handshake error other than
    # ConnectionError/OSError/ValueError kills its handler unlogged and leaves the socket open.
    import struct

    from wantrain.runtime.stage import StageRuntime, _Refused

    spec = JobSpec(model=make_config("toy"), stages=2, out_dir=str(tmp_path), tag="hostile")
    st = StageRuntime(spec, 1, 0)
    st._keyp, st._aes_hw = str(tmp_path / "key"), False
    lst = socket.socket()
    lst.bind(("127.0.0.1", 0))
    lst.listen(16)
    addr = lst.getsockname()
    rst = socket.create_connection(addr, timeout=10)
    rst.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    rst.close()
    deep = socket.create_connection(addr, timeout=10)
    body = b"[" * 60000
    deep.sendall(struct.pack("<I", len(body)) + body)
    good = socket.create_connection(addr, timeout=10)
    good.sendall(b"!")

    def refused():
        with st._tel_lock:
            return [e["reason"] for e in st.telemetry if e["ev"] == "p2p_refused"]

    def admit(sk):
        try:
            first = sk.recv(1, socket.MSG_PEEK)
        except OSError:
            raise _Refused("auth") from None
        if first == b"!":
            # The real peer binds only after both hostile sockets were refused,
            # so neither can be closed as merely "unbound".
            deadline = time.monotonic() + 20
            while len(refused()) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            return None, (sk, dict(peer_id="good", mode="ed25519", cipher="none")), 0, 0, "good"
        st._auth_accepted(sk)
        raise AssertionError("a hostile handshake authenticated")

    out = {}

    def run():
        try:
            out["bound"] = st._accept_until_bound(
                _AbortOnce(lst), [0], admit, [], lambda k: "prev0"
            )
        except BaseException as e:  # noqa: BLE001 -- the test inspects it
            out["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(30)
    try:
        assert not t.is_alive(), "the accept loop never bound the real peer"
        assert "err" not in out, out
        assert list(out["bound"]) == [0]
        assert sorted(refused()) == ["auth", "auth"], refused()
        deep.settimeout(10)
        assert _closed(lambda: deep.recv(1 << 16)) in ("eof", "reset")
    finally:
        for s in (deep, good, lst):
            s.close()
        for sk, _ in out.get("bound", {}).values():
            sk.close()


def test_chain_intruder_is_refused_and_the_run_completes(tmp_path, port_base):
    """Three intruders reach rank 1's listener before rank 0 boots: a broken
    handshake, an unknown key and a key the table pins to another slot. Each
    is refused, logged and closed while accepting goes on; the real peer then
    binds and the run is the oracle's."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: one failed handshake brings the chain down (the old errors[0] raise), the
    # acceptor binds any trusted key instead of slot_peers[rank-1][0], a refused socket takes the
    # stream slot, or a refusal is not logged.
    fleet = _Fleet(tmp_path, _pinned(tmp_path, "chain_rogue", port_base, stages=2))
    try:
        fleet.start((1, 0))
        port = fleet.port(1, 0)
        how = _intrude(port, garbage=True)
        _assert_refused(fleet, 1, 0, "auth", how=how)
        rogue = str(tmp_path / "rogue_key")
        how = _intrude(port, rogue)
        _assert_refused(fleet, 1, 0, "peer", _key(rogue), 0, 0, how)
        how = _intrude(port, fleet.keys[(1, 0)])
        _assert_refused(fleet, 1, 0, "peer", fleet.ids[(1, 0)], 0, 0, how)
        fleet.start_rest()
        fleet.finish()
    finally:
        fleet.kill()
    _assert_bound_once(fleet, 1, 0, [fleet.ids[(0, 0)]])
    assert len(_refusals(fleet, 1, 0)) == 3
    assert [e["reason"] for e in fleet.events(1, 0, "p2p_refused")] == ["auth", "peer", "peer"]


def test_chain_intruder_is_refused_with_two_streams(tmp_path, port_base):
    """With p2p_streams=2 an intruder's socket never takes one of the two
    stream slots: both go to rank 0's two sealed streams."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: the acceptor counts accepted sockets instead of bound ones (the intruder
    # takes stream 0 and the run hangs or fails), or a stream is wired unsealed.
    spec = _pinned(tmp_path, "chain_rogue2", port_base, stages=2, p2p_streams=2)
    fleet = _Fleet(tmp_path, spec)
    try:
        fleet.start((1, 0))
        rogue = str(tmp_path / "rogue_key")
        how = _intrude(fleet.port(1, 0), rogue)
        _assert_refused(fleet, 1, 0, "peer", _key(rogue), 0, 0, how)
        fleet.start_rest()
        fleet.finish()
    finally:
        fleet.kill()
    _assert_bound_once(fleet, 1, 0, [fleet.ids[(0, 0)]] * 2)
    _assert_bound_once(fleet, 0, 0, [fleet.ids[(1, 0)]] * 2)


def test_chain_silent_socket_does_not_hold_a_real_peers_slot(tmp_path, port_base):
    """A socket that connects first and sends nothing holds no stream: the
    real peer binds past it, and the silent one is closed as soon as every
    stream is bound, long before the 30 s handshake timeout."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: accepts are serial (the silent socket blocks the real one for the timeout or
    # takes its slot), or pending sockets are left open after every stream is bound.
    from wantrain.runtime.stage import HANDSHAKE_TIMEOUT_S

    fleet = _Fleet(tmp_path, _pinned(tmp_path, "chain_silent", port_base, stages=2))
    try:
        fleet.start((1, 0))
        silent = _connect(fleet.port(1, 0))
        t0 = time.monotonic()
        fleet.start_rest()
        silent.settimeout(HANDSHAKE_TIMEOUT_S * 2)
        how = _closed(lambda: silent.recv(1 << 16))
        waited = time.monotonic() - t0
        silent.close()
        fleet.finish()
    finally:
        fleet.kill()
    _assert_refused(fleet, 1, 0, "unbound", how=how)
    assert waited < HANDSHAKE_TIMEOUT_S, waited
    _assert_bound_once(fleet, 1, 0, [fleet.ids[(0, 0)]])


def _mesh(tmp_path, name, port_base):
    return _Fleet(tmp_path, _pinned(tmp_path, name, port_base, stages=3, stage_widths=[1, 2, 1]))


def test_mesh_intruder_with_an_unknown_key_is_refused(tmp_path, port_base):
    """An unknown key claiming slot (0, 0) at stage 1 peer 0 is refused
    before its HELLO is read; the real (0, 0) binds afterwards."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: the mesh accepts without authentication or trusts the HELLO (the old
    # _connect_mesh), or a refused socket consumes one of the n_in accepts.
    fleet = _mesh(tmp_path, "mesh_rogue", port_base)
    try:
        fleet.start((2, 0), (1, 0))
        rogue = str(tmp_path / "rogue_key")
        how = _intrude(fleet.port(1, 0), rogue, hello=(0, 0))
        _assert_refused(fleet, 1, 0, "peer", _key(rogue), how=how)
        fleet.start_rest()
        fleet.finish()
    finally:
        fleet.kill()
    _assert_bound_once(fleet, 1, 0, [fleet.ids[(0, 0)], fleet.ids[(1, 1)], fleet.ids[(2, 0)]])


def test_mesh_peer_claiming_another_slot_is_refused(tmp_path, port_base):
    """Pinned keys of the fleet claiming a slot that is not theirs: (1, 1)'s
    key claiming (0, 0) is refused at its HELLO, (2, 0)'s key (expected on
    no inbound slot of stage 1 peer 0) before it."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: the acceptor checks the id against trusted_peers (any fleet key) instead of
    # slot_peers[st][pe], so one slot's key could take another slot.
    fleet = _mesh(tmp_path, "mesh_claim", port_base)
    try:
        fleet.start((2, 0), (1, 0))
        port = fleet.port(1, 0)
        how = _intrude(port, fleet.keys[(1, 1)], hello=(0, 0))
        _assert_refused(fleet, 1, 0, "peer", fleet.ids[(1, 1)], 0, 0, how)
        how = _intrude(port, fleet.keys[(2, 0)], hello=(0, 0))
        _assert_refused(fleet, 1, 0, "peer", fleet.ids[(2, 0)], how=how)
        fleet.start_rest()
        fleet.finish()
    finally:
        fleet.kill()
    _assert_bound_once(fleet, 1, 0, [fleet.ids[(0, 0)], fleet.ids[(1, 1)], fleet.ids[(2, 0)]])


def test_mesh_out_of_range_and_duplicate_hello_are_refused(tmp_path, port_base):
    """(1, 1)'s own key (so only the claim is wrong): a stage outside {r-1, r},
    a peer out of range, an intra peer not above ours, a malformed HELLO, and
    after the real (1, 1) bound, its own slot again (a duplicate)."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: a HELLO's stage, range, intra order or duplicate is not checked (a second
    # socket rebinds a live slot or a phantom slot is wired), or a malformed HELLO kills the stage.
    fleet = _mesh(tmp_path, "mesh_range", port_base)
    key, pid = fleet.keys[(1, 1)], fleet.ids[(1, 1)]
    try:
        fleet.start((2, 0), (1, 0))
        port = fleet.port(1, 0)
        for claim, reason in (((2, 0), "stage"), ((0, 3), "range"), ((1, 0), "range")):
            how = _intrude(port, key, hello=claim)
            _assert_refused(fleet, 1, 0, reason, pid, *claim, how=how)
        how = _intrude(port, key, hello=("x", 1))
        _assert_refused(fleet, 1, 0, "hello", pid, how=how)
        fleet.start((1, 1))
        fleet.wait_line(1, 0, "WT_P2P_BOUND hop=intra1 ")
        how = _intrude(port, key, hello=(1, 1))
        _assert_refused(fleet, 1, 0, "duplicate", pid, 1, 1, how)
        fleet.start_rest()
        fleet.finish()
    finally:
        fleet.kill()
    _assert_bound_once(fleet, 1, 0, [fleet.ids[(0, 0)], pid, fleet.ids[(2, 0)]])
    assert len(_refusals(fleet, 1, 0)) == 5


def test_silent_socket_does_not_hold_a_real_peers_slot(tmp_path, port_base):
    """Mesh: a socket that connects to stage 1 peer 0 first and sends nothing
    never holds (0, 0)'s or (1, 1)'s slot, and is closed once both bind."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: the mesh reads each accepted socket's HELLO inline (the silent socket blocks
    # the loop for the timeout or consumes an accept), or pending sockets survive the binding.
    from wantrain.runtime.stage import HANDSHAKE_TIMEOUT_S

    fleet = _mesh(tmp_path, "mesh_silent", port_base)
    try:
        fleet.start((2, 0), (1, 0))
        silent = _connect(fleet.port(1, 0))
        t0 = time.monotonic()
        fleet.start_rest()
        silent.settimeout(HANDSHAKE_TIMEOUT_S * 2)
        how = _closed(lambda: silent.recv(1 << 16))
        waited = time.monotonic() - t0
        silent.close()
        fleet.finish()
    finally:
        fleet.kill()
    _assert_refused(fleet, 1, 0, "unbound", how=how)
    assert waited < HANDSHAKE_TIMEOUT_S, waited


def test_p2p_auth_counts_match_the_topology(tmp_path, port_base):
    """Exactly one p2p_auth and one p2p_connected per socket of the topology,
    naming the pinned neighbour: a 3-stage chain with 2 streams per hop, and
    the 1-2-1 mesh (every next-stage peer and every intra pair once)."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: one event per hop instead of per socket (the old chain logged results[0]
    # only), a mesh socket goes unauthenticated, or an event names the wrong slot's key.
    chain = _Fleet(tmp_path, _pinned(tmp_path, "count_chain", port_base, stages=3, p2p_streams=2))
    try:
        chain.start_rest()
        chain.finish()
    finally:
        chain.kill()
    ids = chain.ids
    _assert_bound_once(chain, 0, 0, [ids[(1, 0)]] * 2)
    _assert_bound_once(chain, 1, 0, [ids[(0, 0)]] * 2 + [ids[(2, 0)]] * 2)
    _assert_bound_once(chain, 2, 0, [ids[(1, 0)]] * 2)

    mesh = _mesh(tmp_path, "count_mesh", port_base + 32)
    try:
        mesh.start_rest()
        mesh.finish()
    finally:
        mesh.kill()
    ids = mesh.ids
    _assert_bound_once(mesh, 0, 0, [ids[(1, 0)], ids[(1, 1)]])
    _assert_bound_once(mesh, 1, 0, [ids[(0, 0)], ids[(1, 1)], ids[(2, 0)]])
    _assert_bound_once(mesh, 1, 1, [ids[(0, 0)], ids[(1, 0)], ids[(2, 0)]])
    _assert_bound_once(mesh, 2, 0, [ids[(1, 0)], ids[(1, 1)]])
    for fleet in (chain, mesh):
        for r, p in fleet.keys:
            assert fleet.events(r, p, "p2p_refused") == [], (r, p)


def test_every_slot_reports_its_aead_cost(tmp_path):
    """aead_cost runs the sealed toy chain and the 1-2-1 wide run: every slot
    prints one WT_AEAD line with real seal/open time and bytes, the runs are
    the oracle's, and a missing line or a cost at the bound fails the check."""
    # covers: RUNTIME-I16
    # WOULD-FAIL-IF: a slot prints no WT_AEAD (or one per socket), the counters miss a socket
    # (zero bytes on a hop), sealing changes the payload bytes, or a missing slot passes.
    from wantrain.validation import aead_cost

    out = tmp_path / "aead"
    assert (
        aead_cost.main(
            [
                "--steps",
                "3",
                "--max-frac",
                "1.0",
                "--pairs",
                str(aead_cost.MIN_PAIRS),
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    rows, slots = [], 0
    for name in aead_cost.CASES:
        spec = JobSpec.load(str(out / name / f"{name}_spec.json"))
        ref = ensure_reference(spec, spec.out_dir)
        cmp = compare(ref, os.path.join(spec.out_dir, f"{name}_losses.json"))
        assert cmp["ok"] and cmp["max_abs_diff"] == 0.0, cmp
        for r, w in enumerate(spec.stage_widths):
            for p in range(w):
                slots += 1
                with open(out / name / f"{name}_r{r}p{p}.log", errors="replace") as f:
                    lines = [ln for ln in f if ln.startswith("WT_AEAD ")]
                assert len(lines) == 1, (name, r, p, lines)
                rep = aead_cost.slot_report(str(out / name / f"{name}_r{r}p{p}.log"))
                assert (rep["rank"], rep["peer"], rep["steps"]) == (r, p, 3), rep
                assert rep["seal_ns"] > 0 and rep["open_ns"] > 0, rep
                assert rep["bytes_out"] > 0 and rep["bytes_in"] > 0, rep
                assert rep["steps_wall_ns"] > 0, rep
                rows.append(dict(rank=r, peer=p, serialized_cost_frac=0.5))
    assert slots == 6

    # The gated value is the overlap-aware cost (HANDOUT-M2 §2c): the median of the pair ratios.
    def cfg(v):
        return [{"run": "chain2", "ratios": [v] * aead_cost.MIN_PAIRS}]

    assert aead_cost.summarize(cfg(0.5), rows, [], [], {"aes-256-gcm"}, True, 0.6)["ok"] is True
    assert aead_cost.summarize(cfg(0.5), rows, [], [], {"aes-256-gcm"}, True, 0.5)["ok"] is False
    missing = [{"run": "chain2", "rank": 1, "peer": 0}]
    assert (
        aead_cost.summarize(cfg(0.5), rows, missing, [], {"aes-256-gcm"}, True, 1.0)["ok"] is False
    )
    assert aead_cost.slot_report(str(tmp_path / "absent.log")) is None


# ---------------- M2b: exact sharded stages (ADR-0006, RUNTIME-I10..I13) ----------------
#
# Every arm below runs real processes on localhost with latency 10 ms and jitter 15 ms whose
# jitter seed comes from launch.local.jitter_seed_from_env (WANTRAIN_E1_SEED, else
# os.urandom; printed as E1_SEED=<n>), so repeated runs see distinct arrival streams and a
# failure is reproducible from its printed seed. "Equal" always means the canonical state
# manifest (weights, AdamW m, v and step, hyperparameters, RNG, every loss hex) equals the
# single-device reference's, compared by sha256 per tensor.


M2B_LINK = dict(latency_ms=10, jitter_ms=15)
RECIPES = ("fp32", "bf16", "fp16")


@pytest.fixture(scope="module")
def ref_dir(tmp_path_factory):
    """One reference cache for the module: references are keyed by recipe digest."""
    return tmp_path_factory.mktemp("m2b_reference")


def _m2b_spec(out_dir, tag, port_base, compute="bf16", steps=4, microbatches=8, seed=None, **kw):
    """A toy-GPT spec on the M2b links. Non-fp32 recipes carry their activations with the
    exact planes codec (the raw codec frames only fp32 and int64 tensors)."""
    model_kw = kw.pop("model_kw", {})
    link = kw.pop("link", {})
    model = make_config("toy", compute_dtype=compute)
    for k, v in model_kw.items():
        setattr(model, k, v)
    kw.setdefault("wire_codec", "raw" if compute == "fp32" else "planes")
    spec = JobSpec(
        model=model,
        steps=steps,
        microbatches=microbatches,
        out_dir=str(out_dir),
        tag=tag,
        port_base=port_base,
        dial_timeout_s=60,
        **kw,
    )
    if seed is None:
        seed = jitter_seed_from_env()
    spec.links = links_for(spec, jitter_seed=seed, **{**M2B_LINK, **link})
    spec.e1_seed = seed  # not a field: never serialized
    return spec


def _log_tails(spec, n=12):
    out = []
    spec.finalize()
    for r, w in enumerate(spec.stage_widths):
        for p in range(w):
            path = os.path.join(spec.out_dir, f"{spec.tag}_r{r}p{p}.log")
            try:
                with open(path, errors="replace") as f:
                    lines = f.readlines()[-n:]
            except FileNotFoundError:
                lines = ["<no log>"]
            out.append(f"--- r{r}p{p}\n" + "".join(lines))
    return "\n".join(out)


def _run_ok(spec, timeout=240):
    res = run_local(spec, timeout=timeout)
    assert res["ok"], f"{res}\n{_log_tails(spec)}"
    # RUNTIME-I13 in every arm: no exchange copy ever ran on a compute thread.
    for slot, lines in _slot_lines(spec, "WT_COPY ").items():
        for line in lines:
            f = dict(kv.split("=", 1) for kv in line.split()[1:])
            assert int(f["compute_thread"]) not in json.loads(f["engine_threads"]), (slot, line)
    return res


def _assert_exact(spec, ref_dir):
    """The run's canonical state equals the reference's: every tensor's sha256, the RNG and
    every loss hex. -> the state manifest."""
    state = distributed_manifest(spec)
    ref = reference_manifest_for(spec, ref_dir)
    assert diff_manifests(state, ref) == [], diff_manifests(state, ref)
    assert state == ref
    return state


def _slot_lines(spec, prefix):
    spec.finalize()
    out = {}
    for r, w in enumerate(spec.stage_widths):
        for p in range(w):
            with open(
                os.path.join(spec.out_dir, f"{spec.tag}_r{r}p{p}.log"), errors="replace"
            ) as f:
                out[(r, p)] = [ln.rstrip("\n") for ln in f if ln.startswith(prefix)]
    return out


def _manifest_sha(state):
    return __import__("hashlib").sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


E1_NATIVE_TOPO = {
    "w2_1": dict(stages=2, stage_widths=[2, 1]),
    "w1_2_1": dict(stages=3, stage_widths=[1, 2, 1]),
    "w1_3": dict(stages=2, stage_widths=[1, 3]),
    "w3_1_weighted": dict(stages=2, stage_widths=[3, 1], peer_weights=[[1.0, 0.5, 0.25], [1.0]]),
}
E1_NATIVE_ARMS = [
    (recipe, topo, order, fill)
    for recipe in RECIPES
    for topo in E1_NATIVE_TOPO
    for order in ("store", "forward")
    for fill in (True, False)
]


@pytest.mark.parametrize(
    "recipe,topo,order,fill",
    E1_NATIVE_ARMS,
    ids=[f"{r}-{t}-{o}-{'fill' if f else 'strict'}" for r, t, o, f in E1_NATIVE_ARMS],
)
def test_e1_native_identity(tmp_path, port_base, ref_dir, recipe, topo, order, fill):
    """Step 1 (wire "native", shard "zero1"): W streams every gradient in its production
    dtype through need-first queues, the owner widens before the ordered fold; the state is
    the reference's at every topology, recipe, emission order and fill."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: the owner folds a native (bf16/fp16) gradient without widening it, a
    # yielded gradient is shipped to the wrong stride owner or twice, flush_iter's emission
    # order changes a bit, the need-first queue loses or reorders a fold, or a GRADS frame of
    # a valid run is refused.
    t = E1_NATIVE_TOPO[topo]
    spec = _m2b_spec(
        tmp_path,
        "nat",
        port_base,
        compute=recipe,
        steps=3,
        wire="native",
        stream_grads=True,
        schedule_fill=fill,
        emission_order=[order] * t["stages"],
        **t,
    )
    _run_ok(spec)
    _assert_exact(spec, ref_dir)
    for slot, lines in _slot_lines(spec, "WT_COPY ").items():
        w = spec.stage_widths[slot[0]]
        assert len(lines) == (1 if w > 1 else 0), (slot, lines)


def _toy_step_grads(compute, M, n_layer=2, seed=11):
    """Real per-microbatch gradients of a toy GPT through the engine's B/W split (the store
    closures' native dW/db, autograd for the rest), as one stage computes them."""
    from wantrain.model import wgrad
    from wantrain.model.data import make_batch
    from wantrain.model.gpt import GPTConfig, build_layers, loss_fn

    cfg = GPTConfig(n_layer=n_layer, compute_dtype=compute, seq_len=32)
    layers = build_layers(cfg, seed)
    named = [(n, p) for n, p in layers.named_parameters() if p.requires_grad]
    owned = wgrad.owned_params(layers)
    others = [p for _, p in named if id(p) not in owned]
    per_mb = []
    for mb in range(M):
        x, tgt = make_batch(5, 0, mb, 2, cfg.seq_len, cfg.vocab_size)
        with wgrad.capture() as store:
            h = x
            for layer in layers:
                h = layer(h)
            gs = torch.autograd.grad(loss_fn(h, tgt), others)
        native = {id(p): g for p, g in zip(others, gs)}
        for module, fn in store:
            dw, db = fn()
            native[id(module.weight)] = dw
            if module.bias is not None:
                native[id(module.bias)] = db
        per_mb.append({n: native[id(p)] for n, p in named})
    return cfg, layers, named, per_mb


def _digest(t):
    from wantrain.runtime.logical_ckpt import digest

    return digest(t)


_GLOO_WORKER = """
import sys, torch, torch.distributed as dist
rank, world, port, src, dst = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
x = torch.load(src)[rank]
chunks = list(x.chunk(world))
out = torch.empty_like(chunks[rank])
dist.reduce_scatter(out, chunks)
torch.save(out, dst)
dist.destroy_process_group()
"""


def _gloo_reduce_scatter(tmp_path, port, flat_per_rank):
    src = tmp_path / "rs_in.pt"
    torch.save(flat_per_rank, src)
    world = len(flat_per_rank)
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _GLOO_WORKER,
                str(r),
                str(world),
                str(port),
                str(src),
                str(tmp_path / f"rs{r}.pt"),
            ]
        )
        for r in range(world)
    ]
    try:
        for p in procs:
            assert p.wait(timeout=120) == 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.wait()
    return torch.cat([torch.load(tmp_path / f"rs{r}.pt") for r in range(world)])


@pytest.mark.parametrize("recipe", ["bf16", "fp32"])
def test_e1_native_must_fail_controls(tmp_path, port_base, recipe):
    """The E1 must-fail controls, on real gradient pieces of the toy GPT (never through a
    product seam): each alternative a sharded stage could have built gives another digest
    than the reference's left fold, so the identity arms would catch it."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: one of the alternatives (LayerNorm gathered in bf16, a bf16 accumulator
    # clone, an arrival-order fold, gloo reduce_scatter, a division by 3, a bf16 working copy in
    # the fp32 recipe, reordered multi-use contributions) matched the reference bits, i.e. the
    # E1 identity comparison could not tell it from the exact path.
    from wantrain.runtime import pieces
    from wantrain.runtime.accumulate import OrderedAccumulator

    M = 8
    cfg, layers, named, per_mb = _toy_step_grads(recipe, M)
    names = [n for n, _ in named]
    ref = OrderedAccumulator(range(M))
    for mb in range(M):
        ref.add(mb, [per_mb[mb][n].to(torch.float32) for n in names])
    want = dict(zip(names, ref.result()))
    differs = {}
    if recipe == "bf16":
        # The narrow-dtype controls, on bf16 GEMM gradients and a trained LayerNorm.
        big = next(
            n for n in names if per_mb[0][n].dtype == torch.bfloat16 and per_mb[0][n].dim() == 2
        )
        ln = next(n for n in names if "ln" in n and n.endswith("weight"))
        p = torch.nn.Parameter(dict(named)[ln].detach().clone())
        opt = pieces.adamw([p], 1e-3)
        pieces.step_unit(opt, [p], [want[ln]])
        master = p.detach()  # after one real AdamW step: not a bf16 value
        differs["ln_gathered_bf16"] = _digest(master.to(torch.bfloat16).float()) != _digest(master)
        acc = per_mb[0][big].clone()
        for mb in range(1, M):
            acc.add_(per_mb[mb][big])
        differs["bf16_clone"] = _digest(acc.float()) != _digest(want[big])
    else:
        # The order and arithmetic controls need full-precision fp32 gradients (sums of a few
        # bf16 GEMM outputs are exact in fp32 in any order, so there they cannot differ).
        big = "0.tok.weight"
        w = dict(named)["1.qkv.weight"].detach()
        differs["bf16_working_copy_fp32_recipe"] = _digest(
            pieces.cast_piece(w, torch.bfloat16).float()
        ) != _digest(w)
        triple = [per_mb[mb][big] for mb in range(3)]
        differs["multi_use_reordered"] = _digest((triple[0] + triple[1]) + triple[2]) != _digest(
            (triple[2] + triple[1]) + triple[0]
        )
        order = [5, 3, 7, 1, 0, 6, 2, 4]
        arrival = OrderedAccumulator(order)
        for mb in order:
            arrival.add(mb, [per_mb[mb][big]])
        differs["arrival_order"] = _digest(arrival.result()[0]) != _digest(want[big])
        differs["div3"] = _digest((want[big] / 3) * 3) != _digest(want[big])
        flat = [per_mb[mb][big].reshape(-1) for mb in range(4)]
        four = OrderedAccumulator(range(4))
        for mb in range(4):
            four.add(mb, [flat[mb]])
        rs = _gloo_reduce_scatter(tmp_path, port_base + 60, flat)
        differs["gloo_reduce_scatter"] = _digest(rs) != _digest(four.result()[0])
    print(f"E1_CONTROLS recipe={recipe} {json.dumps(differs)}")
    assert differs and all(differs.values()), differs
    # and the exact path itself matches: the piece fold equals the reference slice
    lay = pieces.piece_layout({big: tuple(want[big].shape)}, 3)[big]
    for k in range(3):
        got = pieces.fold(pieces.piece(per_mb[mb][big], lay, k) for mb in range(M))
        assert _digest(got) == _digest(pieces.piece(want[big], lay, k))


_SPY = """
import atexit, os
import wantrain.model.wgrad as _w

_calls = [0]
_orig = _w.flush_iter


def _spy(*a, **k):
    _calls[0] += 1
    return _orig(*a, **k)


_w.flush_iter = _spy


def _dump():
    with open(os.path.join({out!r}, f"flush_iter_{{os.getpid()}}"), "w") as f:
        f.write(str(_calls[0]))


atexit.register(_dump)
"""


def _spy_flush_iter(tmp_path, monkeypatch):
    """A sitecustomize in every stage process counting wgrad.flush_iter calls."""
    d = tmp_path / "spy"
    out = tmp_path / "spy_out"
    d.mkdir()
    out.mkdir()
    (d / "sitecustomize.py").write_text(_SPY.format(out=str(out)))
    monkeypatch.setenv("PYTHONPATH", str(d))
    return out


@pytest.mark.parametrize("widths", [[1, 1], [1, 1, 1]], ids=["chain2", "chain3"])
def test_native_width1_keeps_flush(tmp_path, port_base, ref_dir, monkeypatch, widths):
    """BA-46: a job whose every stage has width 1 under wire "native" takes exactly the fp32
    path: W calls flush, never flush_iter, and the state is the reference's."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: a width-1 stage under wire "native" streams through flush_iter (and folds
    # unwidened native gradients), or the spy does not reach the stage processes (the control
    # run with a wide stage must count calls).
    out = _spy_flush_iter(tmp_path, monkeypatch)
    spec = _m2b_spec(
        tmp_path / "w1",
        "w1",
        port_base,
        compute="bf16",
        steps=3,
        wire="native",
        stages=len(widths),
        stage_widths=widths,
    )
    _run_ok(spec)
    # counted before _assert_exact: the reference process loads the spy too
    counts = [int(p.read_text()) for p in out.iterdir()]
    assert len(counts) == len(widths) and sum(counts) == 0, counts
    _assert_exact(spec, ref_dir)
    # control: the spy sees a wide native stage's W
    for p in out.iterdir():
        p.unlink()
    wide = _m2b_spec(
        tmp_path / "wide",
        "wide",
        port_base + 32,
        compute="bf16",
        steps=1,
        wire="native",
        stream_grads=True,
        stages=2,
        stage_widths=[1, 2],
    )
    _run_ok(wide)
    assert sum(int(p.read_text()) for p in out.iterdir()) > 0
    # control: folding a width-1 stage's bf16 GEMM gradient from a bf16 clone differs
    from wantrain.runtime.accumulate import OrderedAccumulator

    _, _, named, per_mb = _toy_step_grads("bf16", 4)
    n = next(n for n, p in named if per_mb[0][n].dtype == torch.bfloat16)
    ref = OrderedAccumulator(range(4))
    clone = OrderedAccumulator(range(4))
    for mb in range(4):
        ref.add(mb, [per_mb[mb][n].float()])
        clone.add(mb, [per_mb[mb][n]])
    assert _digest(ref.result()[0]) != _digest(clone.result()[0].float())


E1_CORE = {
    "w1_2": dict(stages=2, stage_widths=[1, 2]),
    "w1_3": dict(stages=2, stage_widths=[1, 3]),
    "w1_4": dict(stages=2, stage_widths=[1, 4]),
    "w2_1": dict(stages=2, stage_widths=[2, 1]),
}


def _pieces(**kw):
    return dict(wire="native", shard="pieces", stream_grads=True, **kw)


@pytest.mark.parametrize("topo", sorted(E1_CORE))
def test_e1_core_pieces_identity(tmp_path, port_base, ref_dir, topo):
    """Step 2 (shard "pieces"): peer k owns piece k of every tensor, units update in the
    first pump after their fold, gathers go need-first; bf16, 4 steps, M 8: the state is the
    reference's and no stall occurs within the bounded run timeout."""
    # covers: RUNTIME-I10, RUNTIME-I13
    # WOULD-FAIL-IF: a piece is folded, updated, cast or gathered wrongly (any byte), a unit's
    # F reads a copy of another step, an update waits for a barrier that never comes (stall),
    # or the identity manifest assembles the pieces wrongly.
    spec = _m2b_spec(
        tmp_path, "core", port_base, compute="bf16", steps=4, **_pieces(**E1_CORE[topo])
    )
    _run_ok(spec, timeout=120)
    _assert_exact(spec, ref_dir)
    _aead_ok(spec)


def test_e1_core_tied(tmp_path, port_base, ref_dir):
    """BA-42: tie_embeddings with the Embed and the Head on one wide stage (width 2): the tied
    table is one parameter in the Embed's unit, shipped when W starts with both contributions
    summed, and the Head reads the Embed unit's working copy."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: the tied parameter is listed in two units (its gradient shipped and its
    # piece updated twice), the Head reads a copy the Embed unit's gather did not refresh, or
    # the identity manifest drops the tied alias.
    from wantrain.model.gpt import build_layers
    from wantrain.runtime import pieces

    mk = dict(tie_embeddings=True, init_scheme="gpt2")
    spec = _m2b_spec(
        tmp_path,
        "tied",
        port_base,
        compute="bf16",
        steps=4,
        model_kw=mk,
        **_pieces(stages=1, stage_widths=[2]),
    )
    spec.finalize()
    units = pieces.exchange_units(build_layers(spec.model, spec.init_seed))
    tok = [u for u in units if any(n == "0.tok.weight" for n, _ in u)]
    assert len(tok) == 1 and units.index(tok[0]) == 0
    assert not any(n.endswith("out.weight") for u in units for n, _ in u)
    _run_ok(spec, timeout=120)
    _assert_exact(spec, ref_dir)


def test_e1_core_updates_between_ops_deadlock(tmp_path, port_base, monkeypatch):
    """Liveness control (F8): with the timing-only seam WANTRAIN_TEST_UPDATES=between_ops
    (updates only in the pumps between ops, never inside a blocked wait) the width-4 arm does
    not complete and the bounded run timeout reports it; without the seam the same arm
    completes (test_e1_core_pieces_identity[w1_4])."""
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: updates already ran only between ops (the arm would deadlock in
    # production), or the seam did not reach the stages (the arm would complete).
    monkeypatch.setenv("WANTRAIN_TEST_UPDATES", "between_ops")
    spec = _m2b_spec(
        tmp_path, "dead", port_base, compute="bf16", steps=4, **_pieces(**E1_CORE["w1_4"])
    )
    t0 = time.monotonic()
    res = run_local(spec, timeout=60)
    assert not res["ok"] and res["rc"] == 2, res
    assert time.monotonic() - t0 < 120


def test_copy_engine_stage(tmp_path, port_base, ref_dir):
    """RUNTIME-I13 on CPU: in a multi-process pieces run every exchange D2H/H2D copy runs on
    the copy engine's thread and none on the stage's compute thread (thread identity, WT_COPY);
    the CUDA copy-stream path is real-hardware pending (F7)."""
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: an exchange copy runs on the compute thread (it would be listed among the
    # engine's threads), the engine is bypassed (zero copies), or the run differs.
    spec = _m2b_spec(
        tmp_path,
        "copy",
        port_base,
        compute="bf16",
        steps=2,
        **_pieces(stages=2, stage_widths=[1, 3]),
    )
    _run_ok(spec)
    _assert_exact(spec, ref_dir)
    lines = _slot_lines(spec, "WT_COPY ")
    for p in range(3):
        (line,) = lines[(1, p)]
        f = dict(kv.split("=", 1) for kv in line.split()[1:])
        engine = json.loads(f["engine_threads"])
        assert int(f["copies"]) > 0 and len(engine) == 1, line
        assert int(f["compute_thread"]) not in engine, line
    assert lines[(0, 0)] == []  # a width-1 stage exchanges nothing


def test_pieces_telem_counts_each_step_once(tmp_path, port_base, ref_dir, monkeypatch):
    """BA-45, RUNTIME-I9 under pieces: with one peer 2x slow, step t+1's ops may run before
    WT_TELEM t; every WT_TELEM t still counts F, B and W as the microbatches the slot owns, OPT
    as its step-t unit updates, and no event is logged twice. A control replaying today's
    take-every-queued-event summary on the same log miscounts at least one step."""
    # covers: RUNTIME-I9, RUNTIME-I13
    # WOULD-FAIL-IF: the summary of step t takes step t+1's op events (they would be lost to
    # both summaries), an event is appended to the log twice, an update is not counted as OPT,
    # or the telemetry moves a bit of the state.
    from wantrain.runtime.telemetry import summarize_step

    # Stage 0 (the Embed) is wide: a fast peer's F of step t+1 needs no input, so it starts
    # while its folds still wait for the slow peer's W of step t.
    monkeypatch.setenv("WANTRAIN_SIM_SLOW", "0:1:2")
    spec = _m2b_spec(
        tmp_path,
        "tel",
        port_base,
        compute="bf16",
        steps=5,
        **_pieces(stages=2, stage_widths=[2, 1]),
    )
    _run_ok(spec)
    _assert_exact(spec, ref_dir)
    from wantrain.model.gpt import build_layers
    from wantrain.runtime import pieces

    a, b = spec.partition[0]
    n_units = len(
        pieces.exchange_units(
            torch.nn.ModuleList(list(build_layers(spec.model, spec.init_seed))[a:b])
        )
    )
    control_miscounts = 0
    for p in range(2):
        sums = _telem_summaries(spec, 0, p)
        assert [s["step"] for s in sums] == list(range(spec.steps))
        owned = len(spec.owned(0, p))
        for s in sums:
            for op in "FBW":
                assert s["ops"][op]["n"] == owned, (p, op, s)
            assert s["ops"]["OPT"]["n"] == n_units, (p, s)
        events = _telem_events(spec, 0, p)
        keyed = [
            (e["ev"], e["step"], e["mb"], e.get("unit"), e["t0"])
            for e in events
            if e["ev"] in ("fwd", "bwd_b", "bwd_w", "step_opt")
        ]
        assert len(keyed) == len(set(keyed)), "an event was logged twice"
        marks = sorted((e["t0"], e["step"]) for e in events if e["ev"] == "summary")
        assert [s for _, s in marks] == list(range(spec.steps))
        taken = set()
        for t_mark, step in marks:
            # today's code: the summary of `step` takes every event queued before it
            batch = [
                e
                for e in events
                if e["ev"] != "summary" and e["t1"] <= t_mark and id(e) not in taken
            ]
            taken |= {id(e) for e in batch}
            today = summarize_step(batch, step, 1, p, 0.0, 0.0)
            if any(today["ops"][op]["n"] != owned for op in "FBW"):
                control_miscounts += 1
    print(f"PIECES_TELEM_CONTROL_MISCOUNTS={control_miscounts}")
    assert control_miscounts >= 1, "the control did not discriminate"


def test_pieces_local_class(tmp_path, port_base, monkeypatch):
    """BA-52: run_local refuses shard "pieces" when two slots of one stage resolve to devices
    whose torch device names differ (CUDA get_device_name, real-hardware pending; patched
    here), naming the stage, before any process starts; equal names launch."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: run_local launches a pieces stage over two device models, refuses one
    # whose models agree, or checks a zero1 job. (One device type per map, CONTROL-I25.)
    from wantrain.launch import local

    names = {"cuda:0": "NVIDIA H100", "cuda:1": "NVIDIA A100"}
    monkeypatch.setattr(local, "device_name", lambda d: names.get(d, "cpu"))
    spec = _m2b_spec(
        tmp_path / "a",
        "cls",
        port_base,
        compute="bf16",
        steps=1,
        **_pieces(stages=2, stage_widths=[1, 2]),
    )
    spec.devices = [["cuda:0"], ["cuda:0", "cuda:1"]]
    with pytest.raises(ValueError, match="stage 1 has slots on different device models"):
        run_local(spec, timeout=60)
    assert not os.path.exists(os.path.join(spec.out_dir, "cls_r0p0.log"))
    zero1 = _m2b_spec(
        tmp_path / "z", "z", port_base, compute="fp32", steps=1, stages=2, stage_widths=[1, 2]
    )
    zero1.devices = [["cuda:0"], ["cuda:0", "cuda:1"]]
    local.check_pieces_class(zero1.finalize())  # zero1 is not checked
    monkeypatch.setattr(local, "device_name", lambda d: "same-model")
    ok = _m2b_spec(
        tmp_path / "b",
        "cls",
        port_base,
        compute="bf16",
        steps=1,
        **_pieces(stages=2, stage_widths=[1, 2]),
    )
    _run_ok(ok)
    with open(os.path.join(ok.out_dir, "cls_spec.json")) as f:
        assert json.load(f)["shard"] == "pieces"


def test_pieces_stale_tag(tmp_path):
    """RUNTIME-I10: a working copy carries its step per piece; an op that reads a unit copy
    whose piece is tagged with another step raises (an injected stale tag), and the same op
    with the right tags runs."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: the forward reads a unit whose piece serves another step without a
    # check, or the check fires on a valid copy.
    import random as _random

    spec = _m2b_spec(
        tmp_path,
        "tag",
        29000,
        compute="bf16",
        steps=3,
        seed=1,
        **_pieces(stages=1, stage_widths=[1]),
    )
    rng, py = torch.get_rng_state(), _random.getstate()
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        from wantrain.runtime.stage import StageRuntime

        rt = StageRuntime(spec, 0, 0)
    finally:
        torch.set_rng_state(rng)
        _random.setstate(py)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(det, warn_only=warn)
    rt._init_copies(1)
    rt._pump = lambda block, top=False: None
    rt._forward(1, 0, 0)  # every piece serves step 1
    rt.piece_tag[2][0] = 0  # inject: a block's unit still holds its step-0 piece
    with pytest.raises(RuntimeError, match="stale working copy: an op of step 1 reads unit 2"):
        rt._forward(1, 1, 0)


def test_pieces_checkpoint_round_trips_frozen_params_and_buffers(tmp_path):
    """BA-18: a pieces slot file holds the stage's non-trainable state (frozen parameters and
    buffers) beside its pieces and RNG; a stage resumed from it holds every one of them bit
    for bit. A LoRA stage freezes its base weights; a buffer is registered on a block."""
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: _pieces_payload leaves out a frozen parameter or a buffer, or the resume
    # does not copy them back (the fresh stage keeps its seed values), or a master piece, its
    # moments or the RNG state change on the round trip.
    import random as _random

    kw = dict(compute="bf16", steps=3, seed=1, model_kw={"lora_rank": 4}, ckpt_every=1)
    spec = _m2b_spec(tmp_path, "b18", 29000, **kw, **_pieces(stages=1, stage_widths=[1]))
    rt = _bare_stage(spec, 0, 0)
    g = torch.Generator().manual_seed(5)
    rt.chunks[0][1].register_buffer("probe", torch.randn(7, generator=g))
    frozen = [n for n, p in rt.chunks[0].named_parameters() if not p.requires_grad]
    assert frozen, "the LoRA stage holds no frozen parameter"
    with torch.no_grad():
        for p in rt.chunks[0].parameters():
            if not p.requires_grad:
                p.copy_(torch.randn(p.shape, generator=g))
        for n in rt.pnames:
            rt.master[n].copy_(torch.randn(rt.master[n].shape, generator=g))
    rt.popt.state.clear()
    pc_grads = [torch.randn(rt.master[n].shape, generator=g) for n in rt.pnames]
    from wantrain.runtime import pieces as pc

    pc.step_unit(rt.popt, [rt.master[n] for n in rt.pnames], pc_grads)
    # the final stage's wave must hold every (step, mb) loss up to its step
    rt.losses = {(0, mb): 3.0 + mb / 7 for mb in range(spec.microbatches)}
    rng, py = torch.get_rng_state(), _random.getstate()
    try:
        torch.manual_seed(1234)  # an RNG state the fresh stage does not hold
        want_rng = torch.get_rng_state()
        rt._checkpoint(0)
        rt._await_wave()
        want = {n: t.detach().clone() for n, t in rt._nontrainable().items()}
        spec2 = _m2b_spec(
            tmp_path,
            "b18",
            29000,
            **kw,
            resume=True,
            resume_step=0,
            **_pieces(stages=1, stage_widths=[1]),
        )
        fresh = _bare_stage(spec2, 0, 0)
        fresh.chunks[0][1].register_buffer("probe", torch.zeros(7))
        # control: before the resume the fresh stage differs on the buffer and a frozen param
        got = fresh._nontrainable()
        assert _digest(got["1.probe"]) != _digest(want["1.probe"])
        assert _digest(got[frozen[0]]) != _digest(want[frozen[0]])
        assert fresh._resume() == 1
        got_rng = torch.get_rng_state()
    finally:
        torch.set_rng_state(rng)
        _random.setstate(py)
    got = fresh._nontrainable()
    assert set(got) == set(want) and "1.probe" in want
    for n, t in want.items():
        assert t.dtype == got[n].dtype and _digest(t) == _digest(got[n]), n
    assert _digest(got_rng) == _digest(want_rng)
    assert fresh.losses == rt.losses
    for n in rt.pnames:
        assert _digest(fresh.master[n]) == _digest(rt.master[n]), n
        for f in ("exp_avg", "exp_avg_sq", "step"):
            a, b = rt.popt.state[rt.master[n]][f], fresh.popt.state[fresh.master[n]][f]
            assert _digest(a) == _digest(b), (n, f)


def test_pieces_ready_update_never_waits_on_the_inbox(tmp_path):
    """RUNTIME-I13: a unit whose fold completed inside an op (an own last contribution, every
    fold of a width-1 stage) updates in the next pump at once; a blocking pump (the barrier's,
    a layer's wait) does not first wait out the inbox's 1 s timeout. Without the seam's
    exemption the barrier of a width-1 pieces stage idled 1 s per checkpoint step."""
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: _pump blocks on the inbox while self._ready holds completed folds, or
    # the update does not run in that pump.
    spec = _m2b_spec(
        tmp_path,
        "rdy",
        29000,
        compute="bf16",
        steps=3,
        seed=1,
        **_pieces(stages=2, stage_widths=[1, 2]),
    )
    rt = _bare_stage(spec, 0, 0)
    rt._init_copies(0)
    gets, ran = [], []

    class _Inbox(queue.Queue):
        def get(self, block=True, timeout=None):
            gets.append(block)
            return super().get(block=block, timeout=timeout)

    rt.inbox = _Inbox()
    rt._ready = [(0, 0)]
    rt._update_unit = lambda step, u: ran.append((step, u))
    rt._pump(block=True, top=True)
    assert gets == [False] and ran == [(0, 0)], (gets, ran)
    # control: with nothing ready the same pump blocks on the inbox (and times out empty)
    gets.clear()
    rt._pump(block=True)
    assert gets == [True] and ran == [(0, 0)], (gets, ran)


def test_pieces_budget_files(tmp_path, port_base, ref_dir):
    """BA-55: pacing "shared" creates one budget file per peer egress, per peer ingress and
    per site pair direction, fresh under out_dir (a stale one is replaced), every stage paces
    through them, and the state is the reference's; pacing "link" creates none."""
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: run_local reuses a stale budget file (a clock from an older run), misses
    # a peer's or a site pair's file, creates files under pacing "link", or shared pacing
    # changes a bit.
    from wantrain.transport.shim import SharedBudget

    shared = dict(
        pacing="shared",
        peer_egress_mbps=[[400.0], [400.0, 400.0]],
        peer_ingress_mbps=[[400.0], [400.0, 400.0]],
        sites=[["a"], ["a", "b"]],
        uplink_mbps={"a->b": 300.0, "b->a": 300.0},
    )
    spec = _m2b_spec(
        tmp_path / "s",
        "bud",
        port_base,
        compute="bf16",
        steps=2,
        **_pieces(stages=2, stage_widths=[1, 2], **shared),
    )
    spec.finalize()
    paths = spec.budget_paths()
    assert len(paths) == 3 + 3 + 2
    stale = paths[("egress", 1, 1)]
    os.makedirs(os.path.dirname(stale), exist_ok=True)
    with open(stale, "wb") as f:
        f.write(b"\x00" * 16)  # a budget of 0 Mbit/s: opening it at 400 would be refused
    _run_ok(spec)
    _assert_exact(spec, ref_dir)
    for key, path in paths.items():
        SharedBudget(400.0 if key[0] != "uplink" else 300.0, path).close()  # fresh, right rate
    link = _m2b_spec(
        tmp_path / "l",
        "bud",
        port_base + 32,
        compute="bf16",
        steps=1,
        **_pieces(stages=2, stage_widths=[1, 2]),
    )
    _run_ok(link)
    assert not os.path.exists(os.path.join(link.out_dir, "budgets"))


def _bare_stage(spec, rank, peer):
    """A real StageRuntime that never connects; the process-global torch and random state its
    constructor sets is put back."""
    import random as _random

    from wantrain.runtime.stage import StageRuntime

    rng, py = torch.get_rng_state(), _random.getstate()
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        return StageRuntime(spec, rank, peer)
    finally:
        torch.set_rng_state(rng)
        _random.setstate(py)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(det, warn_only=warn)


def _frame(ftype, step, mb, body):
    from wantrain.transport import framing as fr
    from wantrain.transport import serial

    return fr.Frame(ftype, fr.OBJ, step, mb, 1, serial.dumps(body), ())


@pytest.mark.parametrize(
    "kind,cls", [("GRADS", "dtype"), ("GRADS", "mb"), ("PARAMS", "dtype"), ("PARAMS", "step")]
)
def test_pieces_frame_malformed_is_fatal(tmp_path, kind, cls):
    """BA-12: every GRADS and PARAMS frame passes exchange.check_frame in _pump before it is
    used, and a malformed one is fatal naming its class: a bad GRADS frame, a PARAMS gather
    in fp32 where the plan's table says bf16, a PARAMS gather of step t+2. A control that
    validates only GRADS frames lets the bad PARAMS gather through, and installs it."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: _pump uses a GRADS or PARAMS frame before check_frame (a wrong-dtype
    # gather is installed into the working copy, a gather of step t+2 or an out-of-range
    # microbatch is folded), or the refusal does not name its class.
    from wantrain.runtime import exchange as xc
    from wantrain.transport import framing as fr

    spec = _m2b_spec(
        tmp_path,
        "frm",
        29000,
        compute="bf16",
        steps=4,
        seed=1,
        **_pieces(stages=2, stage_widths=[1, 2]),
    )
    rt = _bare_stage(spec, 1, 0)
    rt._init_copies(1)
    sender, u = 1, 2  # block 0's qkv unit: a split linear gathered in bf16
    names = rt.units[u]
    if kind == "PARAMS":
        body = [torch.zeros(rt.layout[n][2] - rt.layout[n][1], dtype=rt.gdt[n]) for n in names]
        assert body[0].dtype == torch.bfloat16
        if cls == "dtype":
            body = [t.float() for t in body]
        step = 2 if cls == "dtype" else 3  # the copy for step 2 is valid; step 3 is t+2
        ((env, view),) = xc.chunk_object(u, body)
        f = _frame(fr.PARAMS, step, 0, ("x", env, list(view)))
    else:
        body = [torch.zeros(rt.layout[n][1] - rt.layout[n][0], dtype=rt.gdt[n]) for n in names]
        if cls == "dtype":
            body = [t.float() for t in body]
        ((env, view),) = xc.chunk_object(u, body)
        f = _frame(fr.GRADS, 1, 8 if cls == "mb" else 0, ("x", env, list(view)))
    item = ("net_obj", f, f.obj(), ("intra", sender))
    rt.inbox.put(item)
    with pytest.raises(RuntimeError, match=f"refused {kind} frame from \\('intra', 1\\): {cls}:"):
        rt._pump(block=False)
    if kind == "PARAMS":
        # control: a validator that checks only GRADS lets the bad gather in
        real = xc.check_frame
        rt2 = _bare_stage(spec, 1, 0)
        rt2._init_copies(1)
        before = rt2.copy_got.get((u, step), 0)
        import wantrain.runtime.exchange as ex

        try:
            ex.check_frame = lambda k, *a: real(k, *a) if k == "GRADS" else None
            rt2.inbox.put(("net_obj", f, f.obj(), ("intra", sender)))
            rt2._pump(block=False)
        finally:
            ex.check_frame = real
        assert rt2.copy_got.get((u, step), 0) == before + 1, "the control did not install it"


def test_pieces_codec_exact(tmp_path, port_base, ref_dir):
    """BA-56: GRADS and PARAMS chunks are OBJ bodies (the exact byte image), never a tensor
    body a wire codec could narrow: with wire_codec "planes" and a bf16 boundary the fp32
    Embed and LayerNorm pieces and the fp32 gathers arrive bit-identical (the state is the
    reference's). Control: one captured fp32 piece routed through the planes encoder, as
    _encode_body would route a tensor body, differs."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: an exchange chunk is queued as a tensor body (the planes branch casts it to
    # the bf16 boundary dtype and the fp32 pieces lose their low bits).
    from wantrain.codec import planes

    spec = _m2b_spec(
        tmp_path,
        "codec",
        port_base,
        compute="fp32",
        steps=3,
        wire_codec="planes",
        model_kw=dict(boundary_dtype="bf16"),
        **_pieces(stages=3, stage_widths=[1, 3, 1]),
    )
    _run_ok(spec)
    _assert_exact(spec, ref_dir)
    _, _, named, per_mb = _toy_step_grads("fp32", 1)
    piece = per_mb[0]["1.ln1.weight"]
    payload, shape = planes.encode(piece.to(torch.bfloat16))
    back = planes.decode(payload, shape).to(torch.float32)
    assert _digest(back) != _digest(piece), "the planes route kept the fp32 bits"


FEATURES_ON_PIECES = dict(
    stream_grads=True,
    recompute=True,
    wire_codec="planes",
    p2p_streams=2,
    adapt_budget_ms=8,
    wire="native",
    shard="pieces",
    stages=3,
    stage_widths=[1, 3, 1],
    peer_weights=[[1.0], [1.0, 0.5, 0.25], [1.0]],
    emission_order=["store", "forward", "store"],
)


def _aead_ok(spec):
    for slot, lines in _slot_lines(spec, "WT_AEAD ").items():
        (line,) = lines
        f = dict(kv.split("=", 1) for kv in line.split()[1:])
        assert int(f["seal_ns"]) > 0 and int(f["bytes_out"]) > 0 and int(f["bytes_in"]) > 0, slot


def test_e1_features_on(tmp_path, port_base, ref_dir):
    """Every transport mechanism of FEATURES_ON (streamed gradients, recompute, the planes
    codec, two streams per chain hop, adaptive microbatching, links of 10 ms + 15 ms jitter at
    80 Mbps) with wire "native", shard "pieces" and the topology arm-pieces puts in
    FEATURES_ON ([1, 3, 1], stage 1 weighted [1, 0.5, 0.25], forward emission on stage 1),
    straight and resumed from a checkpoint after 2 of 4 steps, on the sealed data plane."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: a mechanism of the every-feature-on topology interacts with pieces or the
    # native wire (a codec narrows an exchange chunk, a resume drops a piece's state, fragments
    # of two units mix), or an intra hop bypasses the sealed sockets.
    straight = _m2b_spec(
        tmp_path / "s",
        "fon",
        port_base,
        compute="fp32",
        steps=4,
        link=dict(bw_mbps=80),
        **FEATURES_ON_PIECES,
    )
    _run_ok(straight)
    _assert_exact(straight, ref_dir)
    _aead_ok(straight)
    part = _m2b_spec(
        tmp_path / "r",
        "fon",
        port_base + 32,
        compute="fp32",
        steps=2,
        ckpt_every=1,
        link=dict(bw_mbps=80),
        **FEATURES_ON_PIECES,
    )
    _run_ok(part)
    resumed = _m2b_spec(
        tmp_path / "r",
        "fon",
        port_base,
        compute="fp32",
        steps=4,
        resume=True,
        link=dict(bw_mbps=80),
        **FEATURES_ON_PIECES,
    )
    _run_ok(resumed)
    _assert_exact(resumed, ref_dir)
    _aead_ok(resumed)
    for r, w in enumerate(resumed.stage_widths):
        for p in range(w):
            with open(os.path.join(resumed.out_dir, f"fon_r{r}p{p}.log")) as f:
                assert f"WT_RESUME rank={r} start_step=2 " in f.read(), (r, p)


WEIGHTS = [1.0, 0.5, 0.25, 0.25]


def _arm_spec(out, tag, port, arm, widths, steps, **kw):
    """One E1/F1 arm: stage widths, recipe, emission order, weighted IWRR, recompute, fill."""
    weights = [WEIGHTS[:w] if arm.get("weighted") and w > 1 else [1.0] * w for w in widths]
    return _m2b_spec(
        out,
        tag,
        port,
        compute=arm["recipe"],
        steps=steps,
        recompute=arm.get("recompute", False),
        schedule_fill=arm.get("fill", True),
        emission_order=[arm["order"]] * len(widths),
        peer_weights=weights,
        **_pieces(stages=len(widths), stage_widths=widths),
        **kw,
    )


def _run_arm(tmp_path, port_base, ref_dir, arm, steps=3):
    """-> (state manifest, the jitter seeds the arm used). "chain432": checkpoint every step,
    resume at stage-1 width 4 -> 3 -> 2 in one out_dir; "widen24": 2 -> 4."""
    topo = arm["topo"]
    out = tmp_path / topo
    seeds = []
    if topo in ("chain432", "widen24"):
        seq = [(4, 1), (3, 2), (2, steps)] if topo == "chain432" else [(2, 1), (4, steps)]
        for i, (w, n) in enumerate(seq):
            spec = _arm_spec(
                out, "arm", port_base + 16 * (i % 3), arm, [1, w], n, ckpt_every=1, resume=i > 0
            )
            seeds.append(spec.e1_seed)
            _run_ok(spec)
    else:
        k = int(topo[-1])
        widths = [1, k] if topo.startswith("s1") else [k, 1]
        spec = _arm_spec(out, "arm", port_base, arm, widths, steps)
        seeds.append(spec.e1_seed)
        _run_ok(spec)
    return _assert_exact(spec, ref_dir), seeds


# A pairwise covering set (every pair of axis values in at least one arm) over the sharded
# topology, recompute, W fill, emission order and IWRR weights, plus a 4 -> 3 -> 2 resume.
E1_MATRIX = [
    ("s1w1", 0, 0, "store", 0), ("s1w1", 1, 1, "forward", 1),
    ("s1w2", 0, 1, "forward", 0), ("s1w2", 1, 0, "store", 1),
    ("s1w3", 0, 0, "forward", 1), ("s1w3", 1, 1, "store", 0),
    ("s1w4", 0, 1, "store", 1), ("s1w4", 1, 0, "forward", 0),
    ("s0w2", 0, 0, "store", 1), ("s0w2", 1, 1, "forward", 0),
    ("s0w4", 0, 1, "forward", 1), ("s0w4", 1, 0, "store", 0),
    ("chain432", 0, 1, "store", 0),
]  # fmt: skip
E1_MATRIX_IDS = [
    f"{t}-{'rc' if r else 'norc'}-{'fill' if f else 'strict'}-{o}-{'wt' if w else 'eq'}"
    for t, r, f, o, w in E1_MATRIX
]


def test_e1_matrix_is_pairwise_covering():
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: the E1 matrix leaves a pair of axis values untested.
    import itertools as it

    grid = [a for a in E1_MATRIX if a[0] != "chain432"]  # the resume arm is added on top
    axes = list(zip(*grid))
    for i, j in it.combinations(range(5), 2):
        want = set(it.product(set(axes[i]), set(axes[j])))
        assert want <= {(a[i], a[j]) for a in grid}, (i, j)
    assert len(E1_MATRIX) <= 30


def _e1_matrix(tmp_path, port_base, ref_dir, recipe, case):
    t, r, f, o, w = case
    arm = dict(topo=t, recipe=recipe, recompute=bool(r), fill=bool(f), order=o, weighted=bool(w))
    _run_arm(tmp_path, port_base, ref_dir, arm)


@pytest.mark.parametrize("case", E1_MATRIX, ids=E1_MATRIX_IDS)
def test_e1_matrix_fp32(tmp_path, port_base, ref_dir, case):
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: any combination of width, owner of the Embed, recompute, fill, emission
    # order and IWRR weights (or a 4 -> 3 -> 2 resume) moves a bit of the fp32 state.
    _e1_matrix(tmp_path, port_base, ref_dir, "fp32", case)


@pytest.mark.parametrize("case", E1_MATRIX, ids=E1_MATRIX_IDS)
def test_e1_matrix_bf16(tmp_path, port_base, ref_dir, case):
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: any combination of the matrix axes moves a bit of the bf16 state (a working
    # copy cast or a native gradient widened at the wrong place).
    _e1_matrix(tmp_path, port_base, ref_dir, "bf16", case)


@pytest.mark.parametrize("case", E1_MATRIX, ids=E1_MATRIX_IDS)
def test_e1_matrix_fp16(tmp_path, port_base, ref_dir, case):
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: any combination of the matrix axes moves a bit of the fp16 state.
    _e1_matrix(tmp_path, port_base, ref_dir, "fp16", case)


# F1: a fixed rotation; run i takes arm i mod 10, so 20 runs cover every axis value twice.
F1_ROTATION = [
    dict(topo="s1w1", recipe="fp32", order="store", weighted=False),
    dict(topo="s1w2", recipe="bf16", order="forward", weighted=True),
    dict(topo="s1w3", recipe="fp16", order="store", weighted=True),
    dict(topo="s1w4", recipe="fp32", order="forward", weighted=False),
    dict(topo="s0w2", recipe="bf16", order="store", weighted=False),
    dict(topo="s0w4", recipe="fp16", order="forward", weighted=True),
    dict(topo="chain432", recipe="bf16", order="store", weighted=False),
    dict(topo="widen24", recipe="fp32", order="forward", weighted=True),
    dict(topo="s1w2", recipe="fp16", order="store", weighted=False),
    dict(topo="s1w4", recipe="bf16", order="forward", weighted=True),
]


def _f1_run(tmp_path, port_base, ref_dir, run):
    arm = F1_ROTATION[run % len(F1_ROTATION)]
    name = f"{arm['topo']}-{arm['recipe']}-{arm['order']}-{'wt' if arm['weighted'] else 'eq'}"
    t0 = time.monotonic()
    state, seeds = _run_arm(tmp_path, port_base, ref_dir, arm)
    rec = {"run": run, "arm": name, "seed": seeds, "sha256": _manifest_sha(state)}
    print("F1_RUN " + json.dumps(rec))
    assert time.monotonic() - t0 < 600  # F8: completes inside the bounded run timeouts


@pytest.mark.parametrize("run", range(10))
def test_f1_rotation_a(tmp_path, port_base, ref_dir, run):
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: any arm of the rotation (widths 1-4 on either stage, every recipe and
    # emission order, weighted IWRR, resume 4 -> 3 -> 2 or 2 -> 4) differs from the reference
    # under its jitter stream (F1) or stalls (F8).
    _f1_run(tmp_path, port_base, ref_dir, run)


@pytest.mark.parametrize("run", range(10, 20))
def test_f1_rotation_b(tmp_path, port_base, ref_dir, run):
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: the second pass over the rotation, under fresh jitter streams, differs
    # from the reference (F1) or stalls (F8).
    _f1_run(tmp_path, port_base, ref_dir, run)


# ---------------- E5: logical checkpoints and resharding (RUNTIME-I12) ----------------


def _wave(out, r, p, step, tag="e5"):
    from wantrain.fault_tolerance.checkpoint import wave_dir

    return wave_dir(str(out), tag, r, p, step)


def _reseal(d, step=None):
    """Rewrite a wave's wave.json for its current state.pt (a test plants or edits waves; a
    resume must then refuse them on their content, not on a stale sha256)."""
    from wantrain.fault_tolerance.checkpoint import sha256_file

    with open(os.path.join(d, "wave.json")) as f:
        m = json.load(f)
    if step is not None:
        m["step"] = step
    path = os.path.join(d, "state.pt")
    m["files"] = [{"name": "state.pt", "sha256": sha256_file(path), "bytes": os.path.getsize(path)}]
    with open(os.path.join(d, "wave.json"), "w") as f:
        json.dump(m, f)


def _rewrite(d, fn):
    path = os.path.join(d, "state.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fn(payload)
    torch.save(payload, path)
    _reseal(d)


def _replace_wave(dst, src):
    import shutil

    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def _e5_kw(widths, **kw):
    return dict(compute="bf16", **_pieces(stages=2, stage_widths=widths), **kw)


@pytest.fixture(scope="module")
def e5_waves(tmp_path_factory):
    """Real committed waves (steps 0 and 1, every slot) of four short runs under one tag:
    pieces at width 2, pieces at width 3, pieces at width 2 with moved boundaries, and a
    whole-tensor zero1 run at width 2."""
    from helpers.ports import reserve_port_block

    block = reserve_port_block()
    root = tmp_path_factory.mktemp("e5_waves")
    try:
        runs = {
            "p2": _e5_kw([1, 2]),
            "p3": _e5_kw([1, 3]),
            "z2": dict(compute="bf16", stages=2, stage_widths=[1, 2], stream_grads=True),
        }
        out = {}
        for i, (name, kw) in enumerate(runs.items()):
            spec = _m2b_spec(
                root / name, "e5", block.base + 16 * (i % 3), steps=2, ckpt_every=1, **kw
            )
            _run_ok(spec)
            out[name] = root / name
        moved = _m2b_spec(root / "pb", "e5", block.base, steps=2, ckpt_every=1, **_e5_kw([1, 2]))
        moved.piece_bounds = [None, _moved_bounds(moved, 1)]
        _run_ok(moved)
        out["pb"] = root / "pb"
        return out
    finally:
        block.release()


def _moved_bounds(spec, r):
    """Explicit bounds for stage r that differ from the equal split: every inner boundary one
    quantum later where the tensor has room."""
    from wantrain.runtime import pieces

    spec.finalize()
    shapes = spec.stage_param_shapes()[r]
    eq = pieces.piece_layout(shapes, spec.stage_widths[r])
    out = {}
    for n, b in eq.items():
        q, numel = pieces.quantum(shapes[n]), b[-1]
        inner = [min(numel, x + q) for x in b[1:-1]]
        out[n] = [0, *inner, numel]
    assert any(out[n] != list(eq[n]) for n in out)
    return out


def _resume_refused(tmp_path, src, plant, match, widths=(1, 2), shard="pieces", resume_step=1):
    """Copy a wave set, plant a defect, resume stage 1's slot p0 in-process through _resume:
    it must refuse naming the defect before any state is applied."""
    import shutil

    out = tmp_path / "w"
    shutil.copytree(src, out)
    plant(out)
    kw = (
        _e5_kw(list(widths))
        if shard == "pieces"
        else dict(compute="bf16", stages=2, stage_widths=list(widths), stream_grads=True)
    )
    spec = _m2b_spec(out, "e5", 29000, steps=4, seed=1, resume=True, resume_step=resume_step, **kw)
    rt = _bare_stage(spec, 1, 0)
    before = {
        n: t.clone()
        for n, t in (rt.master.items() if shard == "pieces" else rt.chunks[1].state_dict().items())
    }
    rng = torch.get_rng_state()
    with pytest.raises(RuntimeError, match=match):
        rt._resume()
    after = rt.master if shard == "pieces" else rt.chunks[1].state_dict()
    assert all(torch.equal(before[n], after[n]) for n in before), "state applied before refusal"
    assert torch.equal(rng, torch.get_rng_state())


def test_e5_refuse_width(tmp_path, e5_waves):
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a slot inside p0..p(w_old-1) whose manifest names another width is read.
    def plant(out):
        _replace_wave(_wave(out, 1, 1, 1), _wave(e5_waves["p3"], 1, 1, 1))

    _resume_refused(tmp_path, e5_waves["p2"], plant, r"slot p1: its manifest names width 3.*r1p1")


def test_e5_refuse_layout(tmp_path, e5_waves):
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a slot whose manifest names another layout than slot p0's is read.
    def plant(out):
        _replace_wave(_wave(out, 1, 1, 1), _wave(e5_waves["pb"], 1, 1, 1))

    _resume_refused(tmp_path, e5_waves["p2"], plant, r"slot p1: .*another layout.*r1p1")


def test_e5_refuse_step(tmp_path, e5_waves):
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a slot file whose payload (and manifest) names another step than its wave
    # (its sidecar) is read as the resume step's state.
    def plant(out):
        d = _wave(out, 1, 1, 1)
        _replace_wave(d, _wave(out, 1, 1, 0))
        _reseal(d, step=1)

    _resume_refused(
        tmp_path, e5_waves["p2"], plant, r"r1p1/step-00000001/ holds step 0, its wave names step 1"
    )


def test_e5_refuse_peer_index(tmp_path, e5_waves):
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a slot file whose manifest names another peer index than its file name
    # is read (two copies of one peer's pieces, the other's missing).
    def plant(out):
        _replace_wave(_wave(out, 1, 1, 1), _wave(out, 1, 0, 1))

    _resume_refused(
        tmp_path, e5_waves["p2"], plant, r"slot p1: its manifest names peer index 0.*r1p1"
    )


def test_e5_refuse_extra_index(tmp_path, e5_waves):
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a slot p(w_old) with a wave at the resume step whose manifest names that
    # step and width w_old is ignored instead of refused as an extra index.
    def plant(out):
        _replace_wave(_wave(out, 1, 2, 1), _wave(out, 1, 1, 1))

    _resume_refused(
        tmp_path, e5_waves["p2"], plant, r"slot p2: an extra index.*r1p2", widths=(1, 3)
    )


def test_e5_refuse_mixed_kind(tmp_path, e5_waves):
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a wave mixing a pieces slot and a whole-tensor zero1 slot is assembled.
    def plant(out):
        _replace_wave(_wave(out, 1, 1, 1), _wave(e5_waves["z2"], 1, 1, 1))

    _resume_refused(tmp_path, e5_waves["p2"], plant, r"r1p1/step-00000001/: a wave mixing kinds")

    def plant_z(out):
        _replace_wave(_wave(out, 1, 1, 1), _wave(e5_waves["p2"], 1, 1, 1))

    _resume_refused(tmp_path / "z", e5_waves["z2"], plant_z, r"a wave mixing kinds")


def test_e5_refuse_missing_slot(tmp_path, e5_waves):
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a missing piece file of the wave is tolerated (a tensor's piece would be
    # zeros or stale).
    import shutil

    def plant(out):
        shutil.rmtree(_wave(out, 1, 1, 1))

    _resume_refused(
        tmp_path, e5_waves["p2"], plant, r"slot p1 of the width-2 wave at step 1 is missing"
    )


@pytest.mark.parametrize(
    "case,match",
    [
        ("sha", "chunk sha256 mismatch"),
        ("manifest", "pieces payload without its manifest"),
        ("manifest_step", "manifest names step 0, the payload holds step 1"),
        ("two_values", r"two values for \(step, mb\)"),
        ("missing_loss", r"no loss for \(step, mb\) \(1, 0\)"),
        ("rng", "RNG state differs"),
        ("nontrainable", "non-trainable state differs"),
    ],
)
def test_e5_refuse_payload(tmp_path, e5_waves, case, match):
    """logical_ckpt's refusals, end to end through _resume on real waves: a chunk hash
    mismatch, a pieces payload without its manifest, a manifest step other than the payload's,
    two values for one (step, mb), a missing (step, mb) on the final stage, and slots whose
    RNG or non-trainable state differ."""

    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: _resume applies a wave whose piece bytes do not hash to its manifest, that
    # lost its manifest, whose manifest names another step, whose loss histories conflict or
    # miss a microbatch, or whose slots disagree on the RNG or non-trainable state.
    def edit(p):
        if case == "sha":
            n = next(n for n, v in p["pieces"].items() if v["master"].numel())
            p["pieces"][n]["master"][0] += 1
        elif case == "manifest":
            del p["pieces_manifest"]
        elif case == "manifest_step":
            p["pieces_manifest"]["step"] = 0
        elif case == "two_values":
            p["losses"]["0:0"] = 123.0
        elif case == "rng":
            p["rng"][0] ^= 1
        else:
            p["nontrainable"]["planted"] = torch.ones(2)

    def plant(out):
        if case == "missing_loss":
            for k in range(2):
                _rewrite(_wave(out, 1, k, 1), lambda p: p["losses"].pop("1:0", None))
            return
        _rewrite(_wave(out, 1, 1, 1), edit)

    _resume_refused(tmp_path, e5_waves["p2"], plant, match)


def test_e5_refuse_zero1_width(tmp_path, e5_waves):
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a whole-tensor zero1 shard is resumed under shard "zero1" at another width
    # (its optimizer state is params[k::w_old], not params[k::w]).
    _resume_refused(
        tmp_path,
        e5_waves["z2"],
        lambda out: None,
        r"pin 'zero1_widths' differs",
        widths=(1, 3),
        shard="zero1",
    )


def _chain(tmp_path, port_base, seq, final_steps, tag="e5", **extra):
    """Runs of one tag in ONE out_dir, no cleanup: seq = [(stage-1 width, steps, kw)]."""
    out = tmp_path / "chain"
    spec = None
    for i, (w, steps, kw) in enumerate(seq):
        spec = _m2b_spec(
            out,
            tag,
            port_base + 16 * (i % 3),
            steps=steps,
            resume=i > 0,
            **{**_e5_kw([1, w]), **extra, **kw},
        )
        _run_ok(spec)
    return spec


def test_e5_chain_4_3_2(tmp_path, port_base, ref_dir):
    """A pieces run checkpointed every step, resumed at stage-1 width 4 -> 3 -> 2 in one
    out_dir with no cleanup (the width-4 wave's r1p3 stays beside the width-3 wave): the
    resumed losses hold every (step, mb) and the final state is the reference's. Control: the
    state assembled with one boundary shifted by one element differs."""
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a resume at another width reads the stale higher-index slot, drops another
    # slot's waves, loses a loss of an old slot, takes w_old from its own slot, or reassembles a
    # piece at the wrong offset.
    spec = _chain(
        tmp_path, port_base, [(4, 2, dict(ckpt_every=1)), (3, 4, dict(ckpt_every=1)), (2, 6, {})], 6
    )
    state = _assert_exact(spec, ref_dir)
    assert os.path.isdir(os.path.join(spec.out_dir, "waves", "e5", "r1p3"))
    for p, start in ((0, 4), (1, 4)):
        with open(os.path.join(spec.out_dir, f"e5_r1p{p}.log")) as f:
            assert f"WT_RESUME rank=1 start_step={start} " in f.read()
    # control: one boundary shifted by one element
    from wantrain.validation.state import tensor_record

    a = [
        torch.load(os.path.join(spec.out_dir, f"e5_ckpt_rank1p{k}.pt"), weights_only=True)
        for k in range(2)
    ]
    name = next(
        n
        for n in a[0]["pieces"]
        if a[0]["pieces"][n]["master"].numel() > 1 and a[1]["pieces"][n]["master"].numel() > 1
    )
    b = a[0]["pieces_manifest"]["layout"][name]
    full = torch.zeros(b[-1])
    full[: b[1]] = a[0]["pieces"][name]["master"]
    full[b[1] + 1 :] = a[1]["pieces"][name]["master"][:-1]
    li, rest = name.split(".", 1)
    key = f"layers.{spec.partition[1][0] + int(li)}.{rest}"
    shifted = tensor_record(full.reshape(state["weights"][key]["shape"]))
    assert shifted != state["weights"][key]


def test_e5_moved_boundaries(tmp_path, port_base, ref_dir):
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a resume with explicit boundaries that differ from the written layout cuts
    # the logical state wrongly (or keeps the old layout).
    spec = _chain(tmp_path, port_base, [(3, 2, dict(ckpt_every=1)), (3, 4, {})], 4)
    spec.finalize()
    spec2 = _m2b_spec(spec.out_dir, "e5", port_base + 32, steps=5, resume=True, **_e5_kw([1, 3]))
    spec2.piece_bounds = [None, _moved_bounds(spec2, 1)]
    _run_ok(spec2)
    _assert_exact(spec2, ref_dir)


def test_e5_widen(tmp_path, port_base, ref_dir):
    """Resume at a RAISED width (pieces 2 -> 4) with resume_step None: the new slots p2 and p3
    hold no wave of their own and take the resume step from old slot p0's newest complete
    wave (BA-51)."""
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a slot without a wave of its own refuses (M2's "holds no wave"), takes
    # another resume step than its peers, or cuts the logical state wrongly at width 4.
    spec = _chain(tmp_path, port_base, [(2, 2, dict(ckpt_every=1)), (4, 4, {})], 4)
    _assert_exact(spec, ref_dir)
    for p in range(4):
        with open(os.path.join(spec.out_dir, f"e5_r1p{p}.log")) as f:
            assert "WT_RESUME rank=1 start_step=2 " in f.read(), p


@pytest.mark.parametrize("width", [2, 1, 3], ids=["same", "lower", "raised"])
def test_e5_zero1_to_pieces(tmp_path, port_base, ref_dir, width):
    """BA-57: shard "pieces" resumes a whole-tensor zero1 wave (w_old from p0's pinned
    zero1_widths, every old slot read through validate_wave, from_zero1, then pieces at the new
    width), at the same, a lower and a raised width (zero1 w2 -> pieces w3)."""
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: the zero1 wave is read as one slot's shard, names[k::w_old] are mapped
    # wrongly, the zero1_widths pin refuses the logical read, or a raised-width slot refuses.
    out = tmp_path / "z"
    z = _m2b_spec(
        out,
        "e5",
        port_base,
        compute="bf16",
        steps=2,
        ckpt_every=1,
        stages=2,
        stage_widths=[1, 2],
        stream_grads=True,
    )
    _run_ok(z)
    spec = _m2b_spec(out, "e5", port_base + 32, steps=4, resume=True, **_e5_kw([1, width]))
    _run_ok(spec)
    _assert_exact(spec, ref_dir)


def test_e5_pieces_to_zero1(tmp_path, port_base, ref_dir):
    """Shard "zero1" resumes a logical-layout wave as names[k::w] of the logical tensors,
    byte-identically, so the shard rolls back."""
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a zero1 slot reads a pieces wave as its own shard (or refuses it on the
    # zero1_widths pin), or maps the logical state to names[k::w] wrongly.
    out = tmp_path / "p"
    p = _m2b_spec(out, "e5", port_base, steps=2, ckpt_every=1, **_e5_kw([1, 3]))
    _run_ok(p)
    z = _m2b_spec(
        out,
        "e5",
        port_base + 32,
        compute="bf16",
        steps=4,
        resume=True,
        stages=2,
        stage_widths=[1, 3],
        stream_grads=True,
        wire="native",
    )
    _run_ok(z)
    _assert_exact(z, ref_dir)


def test_e5_rebalance_owner(tmp_path, port_base, ref_dir):
    """BA-44: a pieces run resumed at the same width with only peer_weights, mb_owner and
    schedule_order changed (what M2's rebalance writes): weighted [1, 0.5, 0.25] -> equal at
    step 2, byte-identical."""
    # covers: RUNTIME-I12, RUNTIME-I15
    # WOULD-FAIL-IF: new owners change a fold's order or a piece's owner, or a rebalance resume
    # is refused on a pin.
    out = tmp_path / "rb"
    a = _m2b_spec(
        out,
        "e5",
        port_base,
        steps=2,
        ckpt_every=1,
        peer_weights=[[1.0], [1.0, 0.5, 0.25]],
        **_e5_kw([1, 3]),
    ).finalize()
    _run_ok(a)
    b = JobSpec.from_json(a.to_json())
    b.steps, b.resume, b.port_base = 4, True, port_base + 32
    b.peer_weights = [[1.0], [1.0, 1.0, 1.0]]
    b.reschedule()
    assert b.mb_owner[1] != a.mb_owner[1] and b.pins() == a.pins()
    _run_ok(b)
    _assert_exact(b, ref_dir)


def test_e5_waves(tmp_path, port_base, ref_dir):
    """A resume at another width reads the old peers' waves at the resume step through
    validate_wave before _drop_abandoned_waves runs, which stays per slot: resuming width 4 ->
    3 from step 1 while every slot holds step 2, the new slots drop their own step 2, the old
    slot r1p3 keeps all of its waves, and the state is the reference's."""
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a slot drops another slot's waves, reads a wave after dropping it, or a
    # resume from an explicit older step reads a newer wave.
    from wantrain.fault_tolerance.checkpoint import slot_waves

    out = tmp_path / "wv"
    a = _m2b_spec(out, "e5", port_base, steps=3, ckpt_every=1, **_e5_kw([1, 4]))
    _run_ok(a)
    b = _m2b_spec(
        out,
        "e5",
        port_base + 32,
        steps=4,
        resume=True,
        resume_step=1,
        ckpt_every=10,
        **_e5_kw([1, 3]),
    )
    _run_ok(b)
    _assert_exact(b, ref_dir)
    assert [s for s, _ in slot_waves(str(out), "e5", 1, 3)] == [0, 1, 2]
    for p in range(3):
        assert [s for s, _ in slot_waves(str(out), "e5", 1, p)] == [0, 1, 3], p


# ---------------- E2 (bytes, H2) and E3 (holds, H4): SIMULATED on this CPU ----------------


def _stage_terms(spec, r):
    """Per tensor of stage r: (numel, gradient bytes per element, gather bytes per element),
    from the plan's dtype table (wgrad.gather_dtypes: compute dtype for split linears under
    autocast, the parameter's own dtype otherwise)."""
    from wantrain.model import wgrad
    from wantrain.model.gpt import build_layers

    spec.finalize()
    a, b = spec.partition[r]
    chunk = torch.nn.ModuleList(list(build_layers(spec.model, spec.init_seed))[a:b])
    m = spec.model
    gd = wgrad.gather_dtypes(chunk, m.compute_dtype, m.autocast_enabled())
    out = {}
    for n, p in chunk.named_parameters():
        if p.requires_grad:
            esz = torch.empty(0, dtype=gd[id(p)]).element_size()
            out[n] = (int(p.numel()), esz, esz)
    return out


def _exchange_bytes(spec, r, p):
    """{step: tensor bytes this slot sent on intra hops}: GRADS of step s and the gathers of
    update s (tagged s + 1, the step that reads them), from the send events."""
    per: dict = {}
    for e in _telem_events(spec, r, p):
        if e["ev"] != "send" or not e.get("hop", "").startswith("intra") or "tb" not in e:
            continue
        s = e["step"] if e["ft"] == 3 else e["step"] - 1  # framing.GRADS == 3, PARAMS == 4
        per[s] = per.get(s, 0) + e["tb"]
    return per


@pytest.mark.parametrize("w", [2, 4])
def test_e2_pieces_bytes_match_h2(tmp_path, port_base, ref_dir, w):
    """E2 (F2): GRADS + PARAMS tensor bytes per peer per step from the intra hops' send events
    are within 2% of H2 = P_s (w-1)/w [M_k b_g + b_p] (envelopes and frame headers counted
    apart), SIMULATED; the planes-codec ratio and the changed fraction of the working copy per
    step are printed (MEASURE, no bound)."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: a gradient piece is shipped in fp32 instead of its production dtype, a
    # piece goes to a peer twice or to the owner itself, gathers are sent more than once per
    # step and peer, or exchange chunks lose their tensor-byte accounting.
    from wantrain.codec import planes

    spec = _m2b_spec(
        tmp_path,
        "e2",
        port_base,
        compute="bf16",
        steps=3,
        ckpt_every=1,
        **_pieces(stages=2, stage_widths=[1, w]),
    )
    _run_ok(spec)
    _assert_exact(spec, ref_dir)
    terms = _stage_terms(spec, 1)
    bgP = sum(n * g for n, g, _ in terms.values())
    bpP = sum(n * b for n, _, b in terms.values())
    for p in range(w):
        mk = len(spec.owned(1, p))
        h2 = (w - 1) / w * (mk * bgP + bpP)
        sent = _exchange_bytes(spec, 1, p)
        assert sorted(sent) == list(range(spec.steps)), sent
        for s, got in sent.items():
            print(
                f"E2 w={w} peer={p} step={s} bytes={got} h2={h2:.0f} ratio={got / h2:.5f} SIMULATED"
            )
            assert abs(got - h2) <= 0.02 * h2, (p, s, got, h2)
    # MEASURE: lossless planes coding of real gradient pieces, and how much of the working copy
    # changes per step (both inputs of ADR-0006's round-5 follow-up; no bound).
    _, _, named, per_mb = _toy_step_grads("bf16", 1)
    raw = enc = 0
    for n, _ in named:
        g = per_mb[0][n].reshape(-1)
        payload, _shape = planes.encode(g)
        raw, enc = raw + g.numel() * g.element_size(), enc + len(payload)
    print(f"E2_PLANES_RATIO={raw / enc:.4f} MEASURE")
    from wantrain.fault_tolerance.checkpoint import wave_dir

    def copy_at(step):
        st = torch.load(
            os.path.join(wave_dir(spec.out_dir, "e2", 1, 0, step), "state.pt"), weights_only=True
        )
        return {n: v["master"].to(torch.bfloat16) for n, v in st["pieces"].items()}

    a, b = copy_at(1), copy_at(2)
    changed = sum(int((a[n] != b[n]).sum()) for n in a)
    total = sum(a[n].numel() for n in a)
    print(f"E2_CHANGED_COPY_FRACTION={changed / total:.4f} MEASURE")


def test_e2_native_zero1_bytes_are_the_stride_slices(tmp_path, port_base, ref_dir):
    """E2 for step 1 alone (wire "native", shard "zero1"): each peer ships, per microbatch it
    owns, every stride tensor it does not own, in its production dtype: the GRADS tensor bytes
    per step are that volume (within 2%), SIMULATED."""
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: native gradients are widened before shipping, an owned stride tensor is
    # shipped, or a non-owned one is skipped.
    spec = _m2b_spec(
        tmp_path,
        "e2z",
        port_base,
        compute="bf16",
        steps=3,
        wire="native",
        stream_grads=True,
        stages=2,
        stage_widths=[1, 2],
    )
    _run_ok(spec)
    _assert_exact(spec, ref_dir)
    sizes = list(_stage_terms(spec, 1).values())
    for p in range(2):
        mk = len(spec.owned(1, p))
        want = mk * sum(n * g for i, (n, g, _) in enumerate(sizes) if i % 2 != p)
        for s, got in _exchange_bytes(spec, 1, p).items():
            print(f"E2_ZERO1 peer={p} step={s} bytes={got} want={want} SIMULATED")
            assert abs(got - want) <= 0.02 * want, (p, s, got, want)


def _need_first_inversions(spec, r, p):
    """From slot (r, p)'s send events: per intra destination, the exchange items sent while an
    item with a higher need-first priority (a smaller key) for the same destination was
    already queued. Push and pop times are stamped under the queue's lock, so "already
    queued" is exact."""
    by_hop: dict = {}
    for e in _telem_events(spec, r, p):
        if e["ev"] == "send" and "key" in e and e.get("hop", "").startswith("intra"):
            by_hop.setdefault(e["hop"], []).append(e)
    bad = 0
    for sends in by_hop.values():
        sends.sort(key=lambda e: e["tp"])
        for i, a in enumerate(sends):
            for b in sends[i + 1 :]:
                if b["tq"] < a["tp"] and b["key"] < a["key"]:
                    bad += 1
    return bad


def _hold_maxima(spec, r, p):
    """Per step, the WT_HOLD maxima of slot (r, p)."""
    out = {}
    with open(os.path.join(spec.out_dir, f"{spec.tag}_r{r}p{p}.log")) as f:
        for line in f:
            if line.startswith("WT_HOLD "):
                parts = line.split()
                out[int(parts[1])] = {k: int(v) for k, v in (kv.split("=") for kv in parts[2:])}
    return out


E3_ARMS = {
    "w2_m8": dict(w=2, M=8),
    "w4_m8": dict(w=4, M=8),
    "w2_m32": dict(w=2, M=32),
    "w4_m32": dict(w=4, M=32),
    "w4_slow_peer": dict(w=4, M=8, slow="1:1:2"),
    "w4_slow_ingress": dict(w=4, M=8, ingress=True),
    "w4_m32_owner12": dict(w=4, M=32, weights=[3.0, 2.0, 2.0, 1.0]),
}


@pytest.mark.parametrize("arm", sorted(E3_ARMS))
def test_e3_holds_stay_within_one_step(tmp_path, port_base, ref_dir, monkeypatch, arm):
    """E3 (F4, F8): per peer the host bytes held for the exchange (gradient pieces held or
    queued, gathers out and in) never exceed b_g P_s [M_k (w-1)/w + (M-M_k)/w] + 2 b_p P_s
    (w-1)/w, the own pieces waiting on the device never exceed sum_u b_g,u M_k P_u / w, summed
    per host likewise (every peer of these arms shares this host), at width 2 and 4, M 8 and
    32, with one peer 2x slow, a slow destination ingress and one peer owning 12 of 32; every
    step completes and the state is the reference's. SIMULATED (CPU, gpu false)."""
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: gradient pieces of two steps are held at once, gathers of a unit pile up
    # over steps, an own piece is held after its prefix folded, a counter misses a hold or
    # double-counts it, or a straggler stalls the step loop.
    a = E3_ARMS[arm]
    w, M = a["w"], a["M"]
    kw = {}
    if a.get("slow"):
        monkeypatch.setenv("WANTRAIN_SIM_SLOW", a["slow"])
    if a.get("weights"):
        kw["peer_weights"] = [[1.0], a["weights"]]
    if a.get("ingress"):
        kw.update(
            pacing="shared",
            peer_egress_mbps=[[2000.0], [2000.0] * w],
            peer_ingress_mbps=[[2000.0], [2000.0, 2000.0, 40.0, 2000.0]],
            sites=[["a"], ["a"] * w],
            uplink_mbps={},
        )
    spec = _m2b_spec(
        tmp_path,
        "e3",
        port_base,
        compute="bf16",
        steps=3,
        microbatches=M,
        **_pieces(stages=2, stage_widths=[1, w]),
        **kw,
    )
    _run_ok(spec, timeout=600)
    _assert_exact(spec, ref_dir)
    if a.get("weights"):
        assert len(spec.owned(1, 0)) == 12
    terms = _stage_terms(spec, 1)
    bgP = sum(n * g for n, g, _ in terms.values())
    bpP = sum(n * b for n, _, b in terms.values())
    host_sum = bound_sum = 0
    for p in range(w):
        mk = len(spec.owned(1, p))
        host_bound = bgP * (mk * (w - 1) / w + (M - mk) / w) + 2 * bpP * (w - 1) / w
        dev_bound = bgP * mk / w
        holds = _hold_maxima(spec, 1, p)
        assert sorted(holds) == list(range(spec.steps)), holds
        host = max(h["host"] for h in holds.values())
        dev = max(h["device_own"] for h in holds.values())
        print(
            f"E3 arm={arm} peer={p} Mk={mk} host={host} host_bound={host_bound:.0f} "
            f"ratio={host / host_bound:.4f} device_own={dev} device_bound={dev_bound:.0f} "
            f"ratio={dev / dev_bound:.4f} SIMULATED"
        )
        assert host <= host_bound, (p, host, host_bound)
        assert dev <= dev_bound, (p, dev, dev_bound)
        assert _need_first_inversions(spec, 1, p) == 0, p
        host_sum += host
        bound_sum += host_bound
    assert host_sum <= bound_sum


@pytest.mark.parametrize("seam", ["fifo", "barrier"])
def test_holds_seams_leave_the_state_unchanged(tmp_path, port_base, ref_dir, monkeypatch, seam):
    """The timing-only seams E4's controls use (WANTRAIN_TEST_QUEUE=fifo: send queues FIFO;
    WANTRAIN_TEST_BARRIER=always: the full step barrier every step) apply to pieces and change
    timing only: the state is the reference's under each."""
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: a seam changes a byte (a FIFO queue reorders a fold, the barrier drops or
    # repeats an update), or the seam does not reach the pieces path.
    if seam == "fifo":
        monkeypatch.setenv("WANTRAIN_TEST_QUEUE", "fifo")
    else:
        monkeypatch.setenv("WANTRAIN_TEST_BARRIER", "always")
    spec = _m2b_spec(
        tmp_path,
        "seam",
        port_base,
        compute="bf16",
        steps=3,
        **_pieces(stages=2, stage_widths=[1, 3]),
    )
    _run_ok(spec)
    _assert_exact(spec, ref_dir)
    inversions = sum(_need_first_inversions(spec, 1, p) for p in range(3))
    print(f"HOLDS_SEAM={seam} NEED_FIRST_INVERSIONS={inversions}")
    if seam == "barrier":
        assert inversions == 0


def test_e5_export_wave(tmp_path, port_base, ref_dir):
    """A pieces wave exported by CheckpointStore.export_wave (M2's whole-wave export: state.pt
    and wave.json, every hash checked on both sides) into a fresh out_dir resumes at the same
    width, byte-identically: the coordinator's ship and restore path needs no change."""
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a pieces slot file needs anything beyond its wave's state.pt and wave.json
    # (another slot's alias, a separate manifest file), or the export changes a byte.
    from wantrain.fault_tolerance.checkpoint import CheckpointStore

    src = tmp_path / "src"
    a = _m2b_spec(src, "e5", port_base, steps=2, ckpt_every=1, **_e5_kw([1, 3]))
    _run_ok(a)
    store = CheckpointStore(os.path.join(str(src), "waves", "e5"))
    dst = tmp_path / "dst"
    slots = [(0, 0)] + [(1, p) for p in range(3)]
    store.export_wave(1, slots, str(dst), "e5")
    b = _m2b_spec(dst, "e5", port_base + 32, steps=4, resume=True, resume_step=1, **_e5_kw([1, 3]))
    _run_ok(b)
    _assert_exact(b, ref_dir)


# ----------------------------------------------- E4 (ADR-0006 F3, F5, F6): exposure at B'
# exposure is imported inside each e4_ test, never at module level (PLAN-M2b e4-exposure).


def test_e4_configuration_guard_and_kept_ratios():
    """The harness reads the granted PLAN, refuses a run off its scaled
    configuration naming the field, and finds every ratio of round5.md
    section 6's table within the PLAN's 0.5% at B' (h' 256, V' 2000, seq'
    256, T_mb' 256) and the pinned F'."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the harness ran off the PLAN's configuration (a shape,
    # w or M/w, the chunk, a paced fold, an op split > 25% off) without
    # refusing, or B' stopped keeping a ratio round 5 relies on.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    sc = plan["scaled"]
    run = {k: sc[k] for k in ("h", "V", "seq", "T_mb", "n_head", "intra_op_threads")}
    run.update(chunk_bytes=sc["chunk_bytes"], fold_paced=False)
    run["stages"] = {s: {"w": sc["w"][s], "M_over_w": sc["M_over_w"][s]} for s in sc["w"]}
    exposure.check_configuration(plan, run)
    for field, bad in (("h", 255), ("chunk_bytes", 1 << 20), ("fold_paced", True)):
        with pytest.raises(ValueError, match=field):
            exposure.check_configuration(plan, {**run, field: bad})
    off = {**run, "stages": {**run["stages"], "interior_w8": {"w": 8, "M_over_w": 8}}}
    with pytest.raises(ValueError, match="interior_w8"):
        exposure.check_configuration(plan, off)
    for w in (4, 8):
        split = sc["op_split_F_B_W"][f"w{w}"]
        exposure.check_split(plan, w, split)
        with pytest.raises(ValueError, match="op split W"):
            exposure.check_split(plan, w, (split[0], split[1], split[2] * 1.3))
    for w in ("w4", "w8"):
        devs = exposure.check_ratios(plan, sc["F_prime_flops"][w])
        assert set(devs) == set(plan["ratios"]["full_B"])
        print(json.dumps({"e4": "ratios", "F_prime": w, "rel": devs}, sort_keys=True))
    shrunk = {**plan, "scaled": {**sc, "V": 1800}}
    with pytest.raises(ValueError, match="E4 ratio"):
        exposure.check_ratios(shrunk, sc["F_prime_flops"]["w4"])


def test_e4_precheck_simulation_reproduces_the_pinned_tables():
    """Before any arm is measured, the as-built SCHEDULER-I17 simulation
    reproduces round5.md section 6's percentages within 5% at round 5's
    configuration (F' 50 GFLOP/s, op split 1:2:1, HBM') and every arm of the
    granted m2b-e4 PLAN's pinned table within 5% at the E4 configuration (the
    measured F', op split and fold rate). Otherwise E4 stops with a spec
    conflict: a simulation that does not reproduce its anchor cannot judge F3."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the simulation and the PLAN's model diverged on an arm
    # (e.g. SC-4: the hot owner's gathers wait on its ingress in I17 but overlap
    # it in derivation section 2's V/cap), an arm were silently skipped, or
    # the harness laid an arm out differently from the PLAN.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    records = exposure.precheck(plan)
    for rec in records:
        print(json.dumps({"e4": "precheck", **rec}, sort_keys=True))
    assert {r["arm"] for r in records if r["config"] == "e4"} == {a["arm"] for a in plan["arms"]}
    assert len([r for r in records if r["config"] == "round5"]) == len(exposure.ROUND5_PCT)
    bad = [
        (r["config"], r["arm"], r["pinned_pct"], r["simulated_pct"]) for r in records if not r["ok"]
    ]
    assert not bad, f"E4 pre-check: the I17 simulation does not reproduce its anchors: {bad}"


def test_e4_sweep_threshold_rule_reproduces_the_plan_fit():
    """The S1 threshold rule (PLAN scaled.sweep.threshold_rule: a least-squares
    line E% = a + b/cap' through the 3 points below S1 and one through the 4
    above, meeting at the threshold) applied to the as-built simulation's
    sweep reproduces the PLAN's pinned fit and its threshold 6.4609 MB/s; the
    same function judges the measured sweep."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the rule fit E% against cap' instead of 1/cap', split the
    # points at the wrong side of S1, fit one line through all of them, or
    # accepted a sweep with fewer than 3 points on a side.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    sw = plan["scaled"]["sweep"]
    rows = [a for a in plan["arms"] if a["arm"].startswith("sweep.")]
    assert len(rows) == len(sw["util_points"]) == 7
    pts = [
        (a["nic_util"], a["cap_MBps"], exposure.simulate_plan_arm(plan, a["arm"]).pct) for a in rows
    ]
    got = exposure.sweep_threshold(pts)
    print(json.dumps({"e4": "sweep_rule", **got}, sort_keys=True))
    assert abs(got["cap_MBps"] / sw["predicted_threshold_cap_MBps"] - 1) < 1e-4, got
    assert abs(got["below"][1] / sw["fit_below"]["b_pct_MBps"] - 1) < 1e-4, got
    assert abs(got["above"][1] / sw["fit_above"]["b_pct_MBps"] - 1) < 1e-4, got
    assert abs(got["below"][0] - sw["fit_below"]["a_pct"]) < 0.01, got
    assert abs(got["above"][0] - sw["fit_above"]["a_pct"]) < 0.01, got
    with pytest.raises(ValueError, match=">= 3 points"):
        exposure.sweep_threshold(pts[1:])


def _e4_ev(ev, step, t0, t1, **kw):
    return {"ev": ev, "step": step, "mb": 0, "vs": 1, "t0": t0, "t1": t1, **kw}


def test_e4_measured_e_reads_the_telemetry_as_i17_defines_it():
    """The training arms' measured E (stage_late over peer_timing): per step
    t >= 1, the latest peer's first forward plus its gather waits
    (_await_layer's stall events, wait "gather") against step t - 1's stage
    end, the latest peer end; a faster peer's wait for a slower one is the
    lane bubble, never exposure. Other stalls (the step loop waiting for an
    upstream activation) are not gather waits, and a forward's op time is net
    of the gather waits inside it. late_parts splits E, on the peer that sets
    it, into the first forward's late start and its gather waits.
    Hand-computed timelines of two peers."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: E were measured against each peer's own end (1.5 s at step
    # 1, not 0.5), dropped the gather waits (0.5 at step 2, not 0.8), counted a
    # non-gather stall as one (1.0 at step 2), or left the gather wait inside F.
    from wantrain.validation import exposure

    p0 = [
        *(_e4_ev(e, 0, a, b) for e, a, b in (("fwd", 0, 1), ("bwd_b", 1, 2), ("bwd_w", 2, 3))),
        _e4_ev("fwd", 1, 3.9, 5.0),
        _e4_ev("stall", 1, 3.9, 4.5, wait="gather"),
        _e4_ev("bwd_b", 1, 5.0, 6.0),
        _e4_ev("bwd_w", 1, 6.0, 7.0),
        _e4_ev("stall", 2, 7.8, 8.0),
        _e4_ev("fwd", 2, 8.0, 9.2),
        _e4_ev("stall", 2, 8.0, 8.3, wait="gather"),
        _e4_ev("bwd_b", 2, 9.2, 10.2),
        _e4_ev("bwd_w", 2, 10.2, 11.0),
    ]
    p1 = [
        *(_e4_ev(e, 0, a, b) for e, a, b in (("fwd", 0, 1), ("bwd_b", 1, 2.5), ("bwd_w", 2.5, 4))),
        *(_e4_ev(e, 1, a, b) for e, a, b in (("fwd", 4.2, 5.2), ("bwd_b", 5.2, 6.5))),
        _e4_ev("bwd_w", 1, 6.5, 7.5),
        *(_e4_ev(e, 2, a, b) for e, a, b in (("fwd", 7.6, 8.6), ("bwd_b", 8.6, 9.6))),
        _e4_ev("bwd_w", 2, 9.6, 10.6),
    ]
    peers = [exposure.peer_timing(p, 3) for p in (p0, p1)]
    assert peers[0]["gather"] == pytest.approx([0.0, 0.6, 0.3])
    assert peers[0]["ops"]["F"] == pytest.approx([0.9])
    for op in "FBW":
        assert peers[1]["ops"][op] == pytest.approx([1.0]), op
    assert exposure.stage_late(peers) == pytest.approx([0.0, 0.5, 0.8])
    # the parts on the peer that sets E: (first F's start against the stage end, gather waits)
    parts = exposure.late_parts(peers)
    assert [list(p) for p in parts] == [
        [0.0, 0.0],
        pytest.approx([-0.1, 0.6]),
        pytest.approx([0.5, 0.3]),
    ]
    with pytest.raises(ValueError, match="no forward"):
        exposure.peer_timing([e for e in p1 if not (e["ev"] == "fwd" and e["step"] == 2)], 3)


def test_e4_summary_line_counts_violations_and_marks_off_configuration():
    """PLAN section 5's summary from synthetic result lines: each F6 fraction
    is the measured margin over twice the PLAN's threshold, F3 violations
    count only F3-judged arms out of band (plus a threshold more than 25%
    off), and every arm that ran off the PLAN's configuration is named
    (M2B-E4-W8-OFFCONFIG); arms not measured are refused by name."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: a fraction were taken against the threshold instead of
    # twice it, a recorded arm or an in-band F3 arm counted as a violation, the
    # sweep threshold were ignored, or an off-configuration arm went unmarked.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    with pytest.raises(ValueError, match="sweep.threshold"):
        exposure.summary_line(plan, {})
    lines = {n: {"arm": n, "measured": 10.0} for n in exposure.missing_arms(plan, {})}
    rows = {r["arm"]: r for r in plan["arms"]}
    for alt, frac in (
        ("stage0.step1", 0.5),
        ("stage0.lpt", 1.0),
        ("interior_w4.step1", 1.5),
        ("interior_w8.lpt", 2.0),
    ):
        lines[alt]["measured"] = 10.0 + frac * 2 * rows[alt]["f6_min_margin_pct"]
    lines["sweep.util_0.5"].update(f3=True, in_f3_band=False)
    lines["sweep.util_0.7"].update(f3=True, in_f3_band=True)
    lines["stage0.barrier_pieces_fwd"].update(f3=False, in_f3_band=False)
    lines["sweep.threshold"]["within_25pct"] = False
    lines["interior_w4.tree"]["fired"] = False
    lines["interior_w4.emission.rtt_2cw"]["agrees"] = True
    lines["interior_w8.lpt"]["off_configuration"] = "E4 configuration: op split off"
    got = exposure.summary_line(plan, lines)
    assert got["f6_margin_fraction_min"] == pytest.approx(0.5)
    assert got["f6_interior_w8_vs_lpt_fraction"] == pytest.approx(2.0)
    assert got["f3_violations"] == 2
    assert got["f5_violations"] == 0 and got["emission_order_violations"] == 0
    assert got["off_configuration"] == ["interior_w8.lpt"]


# The training arms run the runtime (stage.py) over the shared-pacer shim at the PLAN's
# scaled configuration, through the timing-only seams WANTRAIN_TEST_CHUNK_BYTES,
# WANTRAIN_TEST_INTRA_LATENCY_MS and WANTRAIN_TEST_SLOT_PAD (tests/unit/test_timing_seams.py).
# PLAN section 7 sets cap' from the in-situ F', and the in-situ F' moves with the exchange
# load (the exchange threads share each peer's CPU), so `_e4_measure` solves the fixed
# point: up to 3 runs, each set from the F' the previous run of that layout measured.
# Each arm is judged against the I17 simulation at its own in-situ op times and prints
# one SIMULATED JSON line; once every arm PLAN section 5 reads is measured, the summary
# line follows. Verify lines run in separate sessions: with WANTRAIN_E4_RESULTS naming a
# JSONL file, every line is also kept there under this tree's fingerprint, and the
# summary and the cross-arm judgements (sweep threshold, emission order) read them all.
_E4_FPRIME: dict = {}  # (stage, layout, fp32 wire) -> the latest in-situ F' (this session)
_E4_RUNS: dict = {}  # arm -> (measurement, result line) (this session)
_E4_LINES: dict = {}  # arm -> result line (this session)
E4_TRIES = 3


# E4 measures wall time (about 2 h), so its arms run only in their OWN whole-machine
# run: PLAN-M2b's e4-exposure verify lines and GATES-M2b's pinned entry
# e4-exposure-sweep (machine-lock without --slot exports FACTORY_MACHINE_LOCK_HELD
# with that run's label). They are skipped, with the reason below, in a shared test
# slot (no lock: a measured E says nothing beside other suites) and inside a ladder
# rung, which also holds the whole machine in a verdict gate but under the label
# "ladder:<rung>" and a per-step timeout sized for its own suite (run wf_4c4dfd30-00d:
# the arms ran inside rung integration's 3600 s step). The gate cannot pass on a
# skip: e4-exposure-sweep accepts only "N passed" with nothing skipped, and runs
# with --runxfail.
def _e4_own_whole_machine_run(held: str | None) -> bool:
    """Only E4's own whole-machine run: GATES-M2b's pinned entry e4-exposure-sweep or
    a machine-lock label of E4's verify lines (e4..., m2b-e4...). Every other holder of
    the whole machine (a ladder rung, another pinned entry such as f1-matrix-x20) is
    refused: run wf_4c4dfd30-00d ran the arms inside rung integration, and its r0 gate
    ran them inside pinned:f1-matrix-x20."""
    if not held:
        return False
    return held == "pinned:e4-exposure-sweep" or held.startswith(("e4", "m2b-e4"))


_E4_WHOLE_MACHINE = pytest.mark.skipif(
    not _e4_own_whole_machine_run(os.environ.get("FACTORY_MACHINE_LOCK_HELD")),
    reason=(
        "E4 measures wall time: it runs only in its own whole-machine run "
        "(GATES-M2b pinned e4-exposure-sweep or the PLAN's e4 verify lines), "
        "never in a test slot or inside a ladder rung (FACTORY_MACHINE_LOCK_HELD=ladder:*)"
    ),
)


def _e4_lines(exposure):
    """Every result line of this tree: this session's, plus WANTRAIN_E4_RESULTS's."""
    fp = exposure.tree_fingerprint()
    lines = {}
    path = os.environ.get("WANTRAIN_E4_RESULTS")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for raw in f:
                kept = json.loads(raw)
                if kept.get("fingerprint") == fp:
                    lines[kept["arm"]] = kept
    lines.update(_E4_LINES)
    return lines


def _e4_record(exposure, plan, line):
    """Print one result line, keep it, and print PLAN section 5's summary
    line once every arm it reads is measured on this tree."""
    line = {**line, "fingerprint": exposure.tree_fingerprint()}
    print(json.dumps(line))
    _E4_LINES[line["arm"]] = line
    path = os.environ.get("WANTRAIN_E4_RESULTS")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    lines = _e4_lines(exposure)
    if not exposure.missing_arms(plan, lines):
        print(json.dumps({"e4": "summary", **exposure.summary_line(plan, lines)}))


def _e4_measure(exposure, plan, name, out, port, ref_dir):
    """One training arm, its network set from the latest in-situ F' of its
    stage and layout (first the PLAN's). A run whose in-situ F' lands more
    than the PLAN's tolerance off the F' its network was set from is off the
    configuration and runs again, set from the F' it measured (E4_TRIES runs
    at most). A rerun never hides a state that differs from the reference's."""
    a = exposure.arm_of(name, plan)
    key = (a.stage, a.layout, a.wire_fp32)
    for attempt in range(1, E4_TRIES + 1):
        m = exposure.measure_training(
            plan, name, out / f"try{attempt}", port, ref_dir, fprime_net=_E4_FPRIME.get(key)
        )
        _E4_FPRIME[key] = m["fprime_insitu"]
        off = m["off_configuration"] or ""
        if attempt == E4_TRIES or not m["identity"] or not off.startswith("E4 configuration: cap'"):
            return m
        print(
            json.dumps({"e4": "rerun", "arm": name, "why": off, "label": exposure.LABEL_TRAINING})
        )
    raise AssertionError("unreachable")


def _e4_arm(exposure, plan, name, tmp_path, port_base, ref_dir, crit, f3):
    """Measure one training arm (once per session) and record its line. ->
    (measurement, line)."""
    if name not in _E4_RUNS:
        m = _e4_measure(exposure, plan, name, tmp_path / name, port_base, ref_dir)
        pred = m["predicted_insitu_pct"]
        line = exposure.arm_line(
            name,
            pred,
            m["pct"],
            crit,
            ratio=round(m["pct"] / pred, 4),
            in_f3_band=exposure.in_band(m["pct"], pred),
            f3=f3,
            pinned_pct=m["pinned_pct"],
            per_step_pct=m["per_step_pct"],
            per_step_start_gather_pct=m["per_step_start_gather_pct"],
            c_s_ms=m["c_s_ms"],
            c_s_set_ms=m["c_s_set_ms"],
            split_insitu=m["split_insitu"],
            cap_MBps=round(m["cap_MBps"], 4),
            fprime_net=m["fprime_net"],
            identity=m["identity"],
            manifest_sha256=m["manifest_sha256"],
            off_configuration=m["off_configuration"],
            f3_band=exposure.f3_band_mode(_e4_gpu()),
            label=exposure.LABEL_TRAINING,
        )
        _e4_record(exposure, plan, line)
        _E4_RUNS[name] = (m, line)
    return _E4_RUNS[name]


def _e4_gpu():
    """The human's hardware switch, read as the suite reads it (tests/conftest.py:
    anything but a literal "gpu": true is false)."""
    from conftest import gpu_allowed

    return gpu_allowed()


def _e4_gate(exposure, check, ok, why):
    """E4's gate on one check (exposure.e4_gate_fails, HANDOUT-M2b D-3). An
    enforced check fails on every machine. F3's band and an off-configuration
    arm fail only with gpu true; with gpu false a miss prints a "not
    evidence" / "measured-not-enforced" line and the test goes on to its
    enforced checks. Only the hardware switch decides: never a mask."""
    gpu = _e4_gpu()
    if exposure.e4_gate_fails(check, ok, gpu):
        raise AssertionError(why)
    if not ok:
        verdict = "not evidence" if check == "off_configuration" else "measured-not-enforced"
        print(json.dumps({"e4": verdict, "check": check, "gpu": gpu, "why": why}))


def _e4_valid(exposure, *runs):
    """A measured arm is evidence only if its line is a valid measurement and
    its state is the reference's (else a correctness bug), both enforced on
    every machine, and it ran on the PLAN's configuration (section 7), which
    fails only with gpu true."""
    bad = []
    for m, line in runs:
        bad += [f"E4 arm {line['arm']}: {p}" for p in exposure.measurement_problems(line)]
        if not m["identity"]:
            bad.append(
                f"E4 arm {line['arm']}: the state is not the reference's {m['manifest_diff']}"
            )
    assert not bad, bad
    for m, line in runs:
        _e4_gate(
            exposure,
            "off_configuration",
            m["off_configuration"] is None,
            f"E4 arm {line['arm']}: {m['off_configuration']}",
        )


def _e4_in_band(exposure, line, what):
    """F3's band on a valid measurement: fails with gpu true, measured with gpu false."""
    bad = exposure.measurement_problems(line)
    assert not bad, f"E4 arm {line.get('arm')}: not a valid measurement {bad}: {line}"
    _e4_gate(
        exposure,
        "f3_band",
        line["in_f3_band"],
        f"F3 FIRED ({what} outside [0.8, 1.2]x the simulation): {line}",
    )


E4_F6 = [("stage0", "step1"), ("stage0", "lpt"), ("interior_w4", "step1"), ("interior_w8", "lpt")]


@pytest.mark.parametrize("stage,alt", E4_F6, ids=[f"{s}-{a}" for s, a in E4_F6])
@_E4_WHOLE_MACHINE
def test_e4_f6_pieces_beat_the_alternatives(tmp_path, port_base, ref_dir, stage, alt):
    """F6 (ADR-0006, primary): per-tensor pieces (forward emission, need-first
    queues, per-unit updates, no step barrier) expose less of the exchange
    than step 1 alone (stage 0, interior w 4) and than LPT whole tensors
    (stage 0, interior w 8), in % of C, by at least the PLAN's threshold (half
    its predicted margin); both arms' states are the reference's. Training
    arms on the runtime, SIMULATED (gpu false).
    Gate (HANDOUT-M2b D-3): this criterion fails with gpu false and gpu
    true alike; only F3's band and off-configuration arms are measured, not
    failing, under gpu false."""
    # covers: SCHEDULER-I17, RUNTIME-I13
    # WOULD-FAIL-IF: pieces lost their margin (e.g. a step barrier kept, gathers queued
    # behind gradients, whole-unit updates), an alternative were run as pieces, or any
    # training arm's state differed from the reference by one bit.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    row = {a["arm"]: a for a in plan["arms"]}[f"{stage}.{alt}"]
    pieces = _e4_arm(
        exposure, plan, f"{stage}.pieces_fwd", tmp_path, port_base, ref_dir, "F6 (pieces)", False
    )
    other = _e4_arm(
        exposure,
        plan,
        row["arm"],
        tmp_path,
        port_base + 32,
        ref_dir,
        f"F6 (margin vs {row['f6_vs']})",
        False,
    )
    margin = other[0]["pct"] - pieces[0]["pct"]
    verdict = {
        "e4": "F6",
        "arm": row["arm"],
        "vs": row["f6_vs"],
        "margin_pct": round(margin, 4),
        "threshold_pct": row["f6_min_margin_pct"],
        "fraction": round(margin / (2 * row["f6_min_margin_pct"]), 4),
        "fired": margin < row["f6_min_margin_pct"],
        "label": exposure.LABEL_TRAINING,
    }
    # The human's decision (2026-10-10 ~19:45, HANDOUT-M2b D-3): with gpu false the
    # interior w8 arms are not evidence for ANY timing criterion, F6 included (this Mac
    # cannot put w8 on the PLAN's op split, M2B-E4-W8-OFFCONFIG): F6 is printed and
    # gated as "off_configuration" (fails only with gpu true). Byte identity is still
    # enforced on every arm by _e4_valid.
    w8_off = stage == "interior_w8" and not _e4_gpu()
    verdict["evidence"] = "not evidence (interior w8, gpu false)" if w8_off else "evidence"
    print(json.dumps(verdict))
    _e4_valid(exposure, pieces, other)
    _e4_gate(
        exposure,
        "off_configuration" if w8_off else "f6",
        not verdict["fired"],
        f"F6 FIRED (ADR-0006 refuted on this arm): {verdict}",
    )


E4_CONTROLS = [
    (s, k) for s in ("stage0", "interior_w4", "interior_w8") for k in ("barrier", "fifo", "today")
]


@pytest.mark.parametrize("stage,kind", E4_CONTROLS, ids=[f"{s}-{k}" for s, k in E4_CONTROLS])
@_E4_WHOLE_MACHINE
def test_e4_controls_fifo_barrier_today(tmp_path, port_base, ref_dir, stage, kind):
    """F3's control: pieces with WANTRAIN_TEST_BARRIER=always within [0.8,
    1.2]x the I17 simulation with its barrier flag on, else the shim, not H3,
    is wrong. Recorded: the FIFO control (WANTRAIN_TEST_QUEUE=fifo) and
    today's whole-stage flush (fp32 wire, ZeRO-1). Every state is the
    reference's. Training arms on the runtime, SIMULATED (gpu false).
    Gate (HANDOUT-M2b D-3): with gpu true a band miss fails; with gpu false
    it prints its measured line and does not fail. An invalid measurement,
    a state off the reference and an off-configuration arm under gpu true
    fail on every run."""
    # covers: SCHEDULER-I17, RUNTIME-I13
    # WOULD-FAIL-IF: the barrier seam did not hold every unit's gathers to the step
    # barrier (or the simulation's barrier mode disagreed with the runtime's), the shim
    # mis-paced the exchange, or a seam changed a byte of the state.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    name = {"barrier": "barrier_pieces_fwd", "fifo": "fifo_pieces_fwd", "today": "today"}[kind]
    crit = "F3 (barrier control)" if kind == "barrier" else "recorded"
    run = _e4_arm(
        exposure, plan, f"{stage}.{name}", tmp_path, port_base, ref_dir, crit, kind == "barrier"
    )
    _e4_valid(exposure, run)
    if kind == "barrier":
        _e4_in_band(exposure, run[1], "the barrier control: the shim is wrong")


E4_SWEEP = ("0.5", "0.7", "0.9", "1.1", "1.3", "1.6", "2.0")


@pytest.mark.parametrize("util", E4_SWEEP)
@_E4_WHOLE_MACHINE
def test_e4_f3_sweep_point(tmp_path, port_base, ref_dir, util):
    """F3: interior w 4 pieces at one of the PLAN's 7 utilizations (cap'
    varied, the hop latency held): measured E within [0.8, 1.2]x the I17
    simulation, 3 points below S1 and 4 above. Training arm on the runtime,
    SIMULATED (gpu false).
    Gate (HANDOUT-M2b D-3): with gpu true a band miss fails; with gpu false
    it prints its measured line and does not fail. An invalid measurement,
    a state off the reference and an off-configuration arm under gpu true
    fail on every run."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the simulation's link model (shared egress and ingress, chunks of f',
    # need-first queues, the hop latency) mispredicted exposure on this side of S1.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    run = _e4_arm(exposure, plan, f"sweep.util_{util}", tmp_path, port_base, ref_dir, "F3", True)
    _e4_valid(exposure, run)
    _e4_in_band(exposure, run[1], "a sweep point")


@_E4_WHOLE_MACHINE
def test_e4_f3_sweep_threshold():
    """F3: the PLAN's rule (a line E% = a + b/cap' through the 3 points below
    S1 and one through the 4 above) finds a threshold in the measured sweep
    within 25% of the one it finds in the simulated points (each point's
    cap' in the PLAN's units). Reads the 7 sweep points' lines measured on
    this tree (test_e4_f3_sweep_point). SIMULATED (gpu false).
    Gate (HANDOUT-M2b D-3): with gpu true a band miss fails; with gpu false
    it prints its measured line and does not fail. An invalid measurement,
    a state off the reference and an off-configuration arm under gpu true
    fail on every run."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the measured knee moved more than 25% from the simulated one, or the
    # threshold were judged on points of another tree or with points missing.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    rows = [a for a in plan["arms"] if a["arm"].startswith("sweep.")]
    lines = _e4_lines(exposure)
    missing = [a["arm"] for a in rows if a["arm"] not in lines]
    assert not missing, f"E4 sweep threshold: points not measured on this tree: {missing}"
    pinned = plan["scaled"]["F_prime_flops"]["w4"]
    pts = [(a, lines[a["arm"]]) for a in rows]
    caps = [ln["cap_MBps"] * pinned / ln["fprime_net"] for _a, ln in pts]
    meas = [(a["nic_util"], c, ln["measured"]) for (a, ln), c in zip(pts, caps)]
    sim = [(a["nic_util"], c, ln["predicted"]) for (a, ln), c in zip(pts, caps)]
    m_cap = exposure.sweep_threshold(meas)["cap_MBps"]
    s_cap = exposure.sweep_threshold(sim)["cap_MBps"]
    within = abs(m_cap / s_cap - 1) <= 0.25
    _e4_record(
        exposure,
        plan,
        {
            "arm": "sweep.threshold",
            "predicted": round(s_cap, 4),
            "measured": round(m_cap, 4),
            "unit": "cap' MB/s at the PLAN's F'",
            "criterion": "F3 (threshold within 25%)",
            "pinned": plan["scaled"]["sweep"]["predicted_threshold_cap_MBps"],
            "within_25pct": within,
            "f3_band": exposure.f3_band_mode(_e4_gpu()),
            "label": exposure.LABEL_TRAINING,
        },
    )
    _e4_gate(
        exposure,
        "f3_band",
        within,
        f"F3 FIRED: measured S1 threshold {m_cap:.4f} vs simulated {s_cap:.4f} MB/s",
    )


@pytest.mark.parametrize("order", ["fwd", "bwd"])
@_E4_WHOLE_MACHINE
def test_e4_f3_shared_uplink(tmp_path, port_base, ref_dir, order):
    """F3 on a shared uplink: interior w 4 pieces with peers {0, 1} and {2, 3}
    on two sites, one uplink per direction at cap' shared by every
    cross-site flow, within [0.8, 1.2]x the I17 simulation. Training arm on
    the runtime, SIMULATED (gpu false).
    Gate (HANDOUT-M2b D-3): with gpu true a band miss fails; with gpu false
    it prints its measured line and does not fail. An invalid measurement,
    a state off the reference and an off-configuration arm under gpu true
    fail on every run."""
    # covers: SCHEDULER-I17, TRANSPORT-I10
    # WOULD-FAIL-IF: the shim gave each cross-site connection its own uplink (no
    # sharing), or the simulation's uplink resource disagreed with the shim's.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    name = f"interior_w4.uplink_pieces_{order}"
    run = _e4_arm(exposure, plan, name, tmp_path, port_base, ref_dir, "F3", True)
    _e4_valid(exposure, run)
    _e4_in_band(exposure, run[1], "an uplink arm")


E4_EMISSION = [
    (s, o, r)
    for s in ("interior_w4", "stage0")
    for r in ("rtt0", "rtt_2cw")
    for o in ("fwd", "bwd")
]


@pytest.mark.parametrize(
    "stage,order,rtt", E4_EMISSION, ids=[f"{s}-{o}-{r}" for s, o, r in E4_EMISSION]
)
@_E4_WHOLE_MACHINE
def test_e4_emission_rtt_arm(tmp_path, port_base, ref_dir, stage, order, rtt):
    """BA-43 (ADR-0006 E4 arms): pieces under emission 'forward' or 'store'
    at RTT 0 or RTT_hi = 2 c_W (one-way RTT/2 + 2f'/PCIe' on the intra hops),
    within [0.8, 1.2]x the I17 simulation. Training arm on the runtime,
    SIMULATED (gpu false).
    Gate (HANDOUT-M2b D-3): with gpu true a band miss fails; with gpu false
    it prints its measured line and does not fail. An invalid measurement,
    a state off the reference and an off-configuration arm under gpu true
    fail on every run."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the runtime's emission order did not reach W's flush, the RTT were
    # not on the intra hops, or the simulation mispredicted the arm.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    name = f"{stage}.emission_{order}.{rtt}"
    run = _e4_arm(exposure, plan, name, tmp_path, port_base, ref_dir, "F3", True)
    _e4_valid(exposure, run)
    _e4_in_band(exposure, run[1], "an emission arm")


@pytest.mark.parametrize("stage", ["interior_w4", "stage0"])
@_E4_WHOLE_MACHINE
def test_e4_emission_order_at_rtt_hi(stage):
    """BA-43: at RTT_hi the simulation's faster emission order (at each arm's
    in-situ configuration) must be the measured faster one at interior w 4
    (the planner relies on it); stage 0 is recorded (predicted equal). Reads
    the two RTT_hi lines measured on this tree (test_e4_emission_rtt_arm).
    SIMULATED (gpu false).
    Gate (HANDOUT-M2b D-3): this criterion fails with gpu false and gpu
    true alike; only F3's band and off-configuration arms are measured, not
    failing, under gpu false."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the simulation picked the order the runtime measures slower at
    # RTT_hi, or the verdict were drawn from lines of another tree.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    lines = _e4_lines(exposure)
    names = [f"{stage}.emission_{o}.rtt_2cw" for o in ("fwd", "bwd")]
    missing = [n for n in names if n not in lines]
    assert not missing, f"E4 emission order: arms not measured on this tree: {missing}"
    fwd, bwd = (lines[n] for n in names)
    chosen = "forward" if fwd["predicted"] <= bwd["predicted"] else "store"
    faster = "forward" if fwd["measured"] <= bwd["measured"] else "store"
    decides = stage == "interior_w4"
    verdict = {
        "arm": f"{stage}.emission.rtt_2cw",
        "predicted": chosen,
        "measured": faster,
        "unit": "faster emission order",
        "criterion": "BA-43 (decides)" if decides else "BA-43 (recorded)",
        "predicted_pct": [fwd["predicted"], bwd["predicted"]],
        "measured_pct": [fwd["measured"], bwd["measured"]],
        "agrees": chosen == faster,
        "label": exposure.LABEL_TRAINING,
    }
    _e4_record(exposure, plan, verdict)
    if decides:
        _e4_gate(exposure, "emission_order", verdict["agrees"], f"BA-43 FIRED: {verdict}")


@_E4_WHOLE_MACHINE
def test_e4_tree_exposes_more_than_the_left_fold():
    """F5: the standalone exchange harness (w processes moving the stage's
    exchange bytes through the shared-pacer shim, never training state)
    measures the streaming left fold and the residue-group tree at the same
    configuration; at interior w 4, where S1 holds, the tree must expose more
    than the left fold (stage 0 recorded). Each measurement must also sit in
    F3's [0.8, 1.2] band of its I17 simulation, or the harness, not F5, is
    wrong (that band is F3's: measured, not failing, under gpu false).
    SIMULATED (gpu false).
    Gate (HANDOUT-M2b D-3): this criterion fails with gpu false and gpu
    true alike; only F3's band and off-configuration arms are measured, not
    failing, under gpu false."""
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: the tree released its partials before its last W (a scratch
    # mutant releasing them at the first W measured 0.0% against 11.841%:
    # out of the band, and F5 fires), or the harness measured nothing.
    from wantrain.validation import exposure

    plan = exposure.load_plan()
    rows = {a["arm"]: a for a in plan["arms"]}
    assert rows["interior_w4.tree"]["s1_holds"] is True
    fired = []
    for stage, decides in (("interior_w4", True), ("stage0", False)):
        crit = "F5 (decides)" if decides else "F5 (recorded)"
        got, lines = {}, {}
        for tree, name in ((False, f"{stage}.left_fold"), (True, f"{stage}.tree")):
            m = got[tree] = exposure.measure_exchange(plan, stage, tree)
            ratio = m["pct"] / m["predicted_pct"]
            lines[tree] = exposure.arm_line(
                name,
                m["predicted_pct"],
                m["pct"],
                crit,
                ratio=round(ratio, 4),
                in_f3_band=0.8 <= ratio <= 1.2,
                f3_band=exposure.f3_band_mode(_e4_gpu()),
                per_step_pct=m["per_step_pct"],
            )
        tree_pct, left_pct = got[True]["pct"], got[False]["pct"]
        _e4_record(exposure, plan, lines[False])
        _e4_record(exposure, plan, {**lines[True], "fired": tree_pct <= left_pct})
        for line in lines.values():
            _e4_in_band(exposure, line, "the F5 harness off its I17 simulation")
        verdict = {
            "e4": "F5",
            "arm": stage,
            "decides": decides,
            "tree_pct": round(tree_pct, 4),
            "left_fold_pct": round(left_pct, 4),
            "fired": tree_pct <= left_pct,
            "label": exposure.LABEL,
        }
        print(json.dumps(verdict))
        if decides and tree_pct <= left_pct:
            fired.append(verdict)
    _e4_gate(exposure, "f5", not fired, f"F5 FIRED (ADR-0006 refuted on this arm): {fired}")


# ---------------------------------------------------------------- M4 oracle (oracle-m4, wave 1)
# Self-contained: pure helpers of validation/reference.py, validation/state.py and
# launch/local.py, plus dense goldens recorded from the pre-change tree (3b0d2f8
# + M2b snapshot, journal .factory/runs/M4/tasks/oracle-m4.md) before any edit.
# The HF and MoE end-to-end oracle arms are identity-m4's (wave 2).

_M4_MODE_CASES = [
    # (id, device_classes, hosts, expected mode, or the texts its refusal names)
    ("one-class", [["cpu:arm64:DEFAULT"], ["cpu:arm64:DEFAULT"]], None, "vanilla"),
    ("one-class-wide", [["mps:Apple M4", "mps:Apple M4"], ["mps:Apple M4"]], None, "vanilla"),
    ("mixed-one-host", [["cpu:arm64:DEFAULT"], ["mps:Apple M4"]], None, "chain"),
    (
        "mixed-one-host-endpoints",
        [["cpu:arm64:DEFAULT"], ["mps:Apple M4"]],
        [["10.0.0.1"], ["10.0.0.1"]],
        "chain",
    ),
    (
        "mixed-two-hosts",
        [["cpu:arm64:DEFAULT"], ["mps:Apple M4"]],
        [["10.0.0.1"], ["10.0.0.2"]],
        ["across hosts", "cpu:arm64:DEFAULT", "mps:Apple M4"],
    ),
    (
        "two-cpu-classes",
        [["cpu:x86_64:AVX2"], ["cpu:x86_64:AVX512"]],
        None,
        ["CPU classes", "cpu:x86_64:AVX2", "cpu:x86_64:AVX512"],
    ),
    (
        "stage-mixes",
        [["cpu:arm64:DEFAULT", "mps:Apple M4"], ["mps:Apple M4"]],
        None,
        ["stage 0 mixes", "cpu:arm64:DEFAULT", "mps:Apple M4"],
    ),
    ("no-classes", None, None, ["no device classes"]),
]


def _m4_mode_failures(fn) -> list:
    """The case ids on which `fn` disagrees with the table: a wrong mode, a
    refusal where a mode is due, or a refusal that does not name its texts."""
    from wantrain.validation.reference import OracleRefused

    bad = []
    for cid, classes, hosts, want in _M4_MODE_CASES:
        try:
            got = fn(classes, hosts)
        except OracleRefused as e:
            if isinstance(want, str) or not all(t in str(e) for t in want):
                bad.append(cid)
            continue
        except Exception:  # noqa: BLE001 - any other failure is a disagreement
            bad.append(cid)
            continue
        if got != want:
            bad.append(cid)
    return bad


def test_m4_oracle_mode_on_crafted_class_maps():
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: oracle_mode runs the vanilla model on a mixed map, runs a unit chain
    # on a map two hosts hold or with two CPU classes or a stage mixing classes, or
    # refuses without naming the classes (BA-420).
    from wantrain.validation.reference import oracle_mode

    assert _m4_mode_failures(oracle_mode) == []

    # The table discriminates: a mode that looks only at the number of classes
    # (the deliberately wrong implementation) disagrees with it.
    def by_count(classes, hosts=None):
        return "vanilla" if len({c for row in classes for c in row}) == 1 else "chain"

    assert set(_m4_mode_failures(by_count)) >= {"mixed-two-hosts", "two-cpu-classes", "stage-mixes"}


def _m4_manifest(aux=None, bias=(0.0, 0.0)):
    from wantrain.validation.state import build_manifest

    w = {
        "layers.0.tok.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "layers.1.mlp.sel_bias": torch.tensor(list(bias), dtype=torch.float32),
    }
    opt = {"layers.0.tok.weight": {"step": torch.tensor(1.0), "exp_avg": torch.ones(2, 3)}}
    hp = {"lr": 1e-3, "betas": (0.9, 0.95)}
    return build_manifest(w, opt, hp, torch.zeros(4, dtype=torch.uint8), [(0, 0, "0x1p+0")], aux)


def test_m4_oracle_aux_section_and_diff_on_crafted_manifests():
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: the aux section is unsorted, a (step, mb, layer) recorded twice or
    # missing compares equal, an aux hex difference or a selection-bias difference
    # passes diff_manifests, a dense manifest gains an aux key (BA-457), or the oracle's
    # aux text differs from the stage's moe_aux_hex or fails on arch's hex values (I-12).
    from wantrain.runtime.stage import moe_aux_hex
    from wantrain.validation.state import aux_hex, diff_manifests

    h1 = aux_hex({"z": torch.tensor(0.75), "lbl": torch.tensor(-0.0)})
    h2 = aux_hex({"lbl": torch.tensor(2.0**-140), "z": 3.5})
    assert h1 == '{"lbl":"-0x0.0p+0","z":"0x1.8000000000000p-1"}'
    assert h2 == json.dumps({"lbl": (2.0**-140).hex(), "z": "0x1.c000000000000p+1"}).replace(
        " ", ""
    )
    # The oracle's text is the stage's (the rows a slot's checkpoint carries), and
    # arch.moe_forward's hex-string values give the same text as the numbers.
    assert h1 == moe_aux_hex({"z": 0.75, "lbl": -0.0})
    assert aux_hex({"lbl": (-0.0).hex(), "z": (0.75).hex()}) == h1
    rows = [(1, 0, 3, h2), (0, 1, 1, h1), (0, 1, 0, h2), (0, 0, 2, h1)]
    ok = _m4_manifest(rows)
    assert ok["aux"] == [[0, 0, 2, h1], [0, 1, 0, h2], [0, 1, 1, h1], [1, 0, 3, h2]]
    assert "aux" not in _m4_manifest(None), "a dense manifest has no aux key"
    assert diff_manifests(ok, _m4_manifest(list(reversed(rows)))) == []

    dup = _m4_manifest(rows + [(0, 1, 1, h1)])
    d = diff_manifests(dup, ok)
    assert any("(0, 1, 1): recorded 2 times" in x for x in d), d
    # Recorded twice on both sides still fails, though the sections are equal.
    assert any("recorded 2 times" in x for x in diff_manifests(dup, dup))
    # The deliberately wrong comparison, by (step, mb, layer) -> hex, sees no difference.
    assert {(s, m, l): h for s, m, l, h in dup["aux"]} == {(s, m, l): h for s, m, l, h in ok["aux"]}

    missing = diff_manifests(_m4_manifest(rows[1:]), ok)
    assert any("(1, 0, 3): missing in a" in x for x in missing), missing
    moved = diff_manifests(_m4_manifest([(1, 0, 3, h1)] + rows[1:]), ok)
    assert any(x.startswith("aux (1, 0, 3):") for x in moved), moved
    absent = diff_manifests(_m4_manifest(None), ok)
    assert any(x.startswith("aux: absent != 4") for x in absent), absent

    bias = diff_manifests(_m4_manifest(rows, bias=(0.001, 0.0)), ok)
    assert any(x.startswith("weights.layers.1.mlp.sel_bias:") for x in bias), bias


def test_m4_oracle_aux_gathered_from_every_slot_checkpoint(tmp_path):
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: distributed_manifest reads moe_aux from one slot only, drops it, or
    # dedups a (step, mb, layer) two slots both recorded (I-12, BA-457).
    from torch import nn

    from wantrain.model.gpt import GPTConfig, build_layers
    from wantrain.validation.state import diff_manifests, distributed_manifest

    spec = JobSpec(
        model=GPTConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, seq_len=16),
        stages=2,
        microbatches=2,
        out_dir=str(tmp_path),
        tag="aux",
    ).finalize()
    layers = build_layers(spec.model, spec.init_seed)
    rows = {0: [[0, 1, 0, "lbl=0x1.8p-2"], [0, 0, 0, "lbl=0x1p-1"]], 1: [[0, 0, 1, "z=0x1.4p+1"]]}

    def write(slot_rows):
        for r, (a, b) in enumerate(spec.partition):
            chunk = nn.ModuleList(list(layers)[a:b])
            opt = torch.optim.AdamW(list(chunk.parameters()), lr=spec.lr)
            ck = {
                "chunks": {r: chunk.state_dict()},
                "opt": opt.state_dict(),
                "rng": torch.get_rng_state(),
                "moe_aux": slot_rows[r],
            }
            torch.save(ck, os.path.join(spec.out_dir, f"aux_ckpt_rank{r}p0.pt"))
        with open(os.path.join(spec.out_dir, "aux_losses.json"), "w") as f:
            json.dump({"losses": [{"step": 0, "mb": 0, "loss_hex": "0x1p+0"}]}, f)
        return distributed_manifest(JobSpec.from_json(spec.to_json()))

    ok = write(rows)
    assert ok["aux"] == [
        [0, 0, 0, "lbl=0x1p-1"],
        [0, 0, 1, "z=0x1.4p+1"],
        [0, 1, 0, "lbl=0x1.8p-2"],
    ]
    dup = write({0: rows[0], 1: rows[1] + [[0, 0, 0, "lbl=0x1p-1"]]})
    assert any("(0, 0, 0): recorded 2 times" in x for x in diff_manifests(dup, ok))
    lost = write({0: rows[0], 1: []})
    assert any("(0, 0, 1): missing in a" in x for x in diff_manifests(lost, ok))
    dense = write({0: [], 1: []})
    assert "aux" not in dense


def test_m4_oracle_child_prefix_is_fresh_and_removed(tmp_path, monkeypatch):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: an HF child starts without -X pycache_prefix, reuses a prefix or a
    # non-empty one, or its prefix survives the child's exit, its failure or its
    # timeout (BA-423, BA-442).
    import subprocess as sp

    from wantrain.launch import local

    root = tmp_path / "pyc"
    probe = (
        "import os, sys; print(sys.pycache_prefix); print(sorted(os.listdir(sys.pycache_prefix)))"
    )
    env = dict(os.environ)
    seen = []
    for code in (0, 3):
        res = local.run_child(
            ["-c", f"{probe}; sys.exit({code})"], env, str(root), 60, capture_output=True, text=True
        )
        assert res.returncode == code, res.stderr
        prefix, listing = res.stdout.splitlines()[:2]
        assert os.path.dirname(prefix) == str(root) and listing == "[]", res.stdout
        assert not os.path.exists(prefix), "the prefix survived its child"
        seen.append(prefix)
    assert seen[0] != seen[1], "each child gets a fresh prefix"
    with pytest.raises(sp.TimeoutExpired):
        local.run_child(["-c", "import time; time.sleep(30)"], env, str(root), 0.5)
    assert os.listdir(root) == [], "a timed-out child's prefix survived"

    # The check discriminates: with removal disabled (the deliberately wrong
    # runner) the prefix is still there after the child exits.
    monkeypatch.setattr(local, "remove_prefix", lambda p: None)
    res = local.run_child(["-c", probe], env, str(root), 60, capture_output=True, text=True)
    assert os.path.exists(res.stdout.splitlines()[0])


def _m4_cfg_cls():
    """A GPTConfig carrying the M4 fields (I-1) whether or not model-core's
    GPTConfig has them yet: the digest and pin rules read them by name."""
    from dataclasses import dataclass

    from wantrain.model.gpt import GPTConfig

    @dataclass
    class M4Config(GPTConfig):
        hf_config: dict | None = None
        hf_buffers: dict | None = None
        hf_census: list | None = None
        hf_code: dict | None = None
        hf_impl: str = "modules"
        moe_experts: int = 0
        moe_top_k: int = 0
        moe_gamma: float = 0.0

    return M4Config


def _m4_golden_specs():
    from wantrain.model.gpt import GPTConfig, make_config

    builtin = JobSpec(
        model=make_config("toy"),
        init_seed=11,
        data_seed=22,
        lr=2e-3,
        steps=3,
        microbatches=4,
        mb_size=2,
    )
    general = JobSpec(
        model=GPTConfig(
            vocab_size=64,
            n_layer=2,
            n_head=4,
            d_model=32,
            seq_len=16,
            norm="rmsnorm",
            mlp="swiglu",
            pos="rope",
            bias=False,
            tie_embeddings=True,
            init_scheme="gpt2",
            n_kv_head=2,
        ),
        init_seed=7,
        data_seed=9,
        steps=5,
        microbatches=2,
        mb_size=3,
    )
    return builtin, general


# Recorded from the pre-change tree before any edit (journal, "Goldens").
_M4_GOLDEN_DIGEST = {"builtin": "6753397c2be4", "general": "6f2a552df22d"}
_M4_GOLDEN_MANIFEST = "b6c30d8b992ad0e5fd04e24c845c81610ce4cb9ae2b7c140fb9ea91badd20029"
_M4_GOLDEN_LOSSES = [
    "0x1.13d7e00000000p+2",
    "0x1.1bc8360000000p+2",
    "0x1.121eda0000000p+2",
    "0x1.1960600000000p+2",
]
_M4_GOLDEN_STATE = "647b159cfa7e74ab387fc00f25148a8e8b7b6563196e8cbd8fd527f7fcc893e3"


def test_m4_oracle_dense_golden_recipe_digest():
    # covers: RUNTIME-I18, RUNTIME-I19
    # WOULD-FAIL-IF: a dense digest, manifest or baseline changes: a new hf_* or moe_*
    # GPTConfig field enters a dense model's recipe_digest, or the hf digest stops
    # covering the device map and hf_code (BA-422, BA-465).
    import hashlib
    from dataclasses import asdict

    from wantrain.launch.local import recipe_digest

    builtin, general = _m4_golden_specs()
    assert recipe_digest(builtin) == _M4_GOLDEN_DIGEST["builtin"]
    assert recipe_digest(general) == _M4_GOLDEN_DIGEST["general"]

    # The M4 fields at their dense defaults leave both digests unchanged.
    m4 = _m4_cfg_cls()
    for name, spec in (("builtin", builtin), ("general", general)):
        wide = JobSpec.from_json(spec.to_json())
        wide.model = m4(**asdict(spec.model))
        assert recipe_digest(wide) == _M4_GOLDEN_DIGEST[name]
        # The deliberately wrong digest (asdict of the whole model) moves.
        leaky = hashlib.sha256(
            json.dumps(
                {
                    "model": asdict(wide.model),
                    **{k: getattr(wide, k) for k in ("init_seed", "data_seed", "lr", "device")},
                    **{k: getattr(wide, k) for k in ("steps", "microbatches", "mb_size")},
                    **{k: getattr(wide, k) for k in ("dataset_path", "data_shards")},
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        assert leaky != _M4_GOLDEN_DIGEST[name]
    # An MoE model's digest carries its moe_* fields; a dense one ignores them.
    moe = JobSpec.from_json(builtin.to_json())
    moe.model = m4(**{**asdict(builtin.model), "moe_experts": 4, "moe_top_k": 2})
    moe2 = JobSpec.from_json(builtin.to_json())
    moe2.model = m4(**{**asdict(builtin.model), "moe_experts": 4, "moe_top_k": 1})
    assert len({recipe_digest(moe), recipe_digest(moe2), _M4_GOLDEN_DIGEST["builtin"]}) == 3
    dense_k = JobSpec.from_json(builtin.to_json())
    dense_k.model = m4(**{**asdict(builtin.model), "moe_top_k": 2})
    assert recipe_digest(dense_k) == _M4_GOLDEN_DIGEST["builtin"]

    # arch "hf": the device map, its classes and hf_code all enter the digest,
    # and a dense spec's map does not.
    def hf(devices=None, classes=None, code="a"):
        s = JobSpec(model=m4(arch="hf", hf_code={"transformers": code}), devices=devices)
        s.device_classes = classes
        return recipe_digest(s)

    base = hf([["cpu"], ["cpu"]], [["cpu:arm64:DEFAULT"], ["cpu:arm64:DEFAULT"]])
    others = {
        hf([["cpu"], ["mps"]], [["cpu:arm64:DEFAULT"], ["cpu:arm64:DEFAULT"]]),
        hf([["cpu"], ["cpu"]], [["cpu:arm64:DEFAULT"], ["mps:Apple M4"]]),
        hf([["cpu"], ["cpu"]], [["cpu:arm64:DEFAULT"], ["cpu:arm64:DEFAULT"]], code="b"),
    }
    assert base not in others and len(others) == 3
    mapped = JobSpec.from_json(builtin.to_json())
    mapped.devices = [["cpu"], ["mps"]]
    assert recipe_digest(mapped) == _M4_GOLDEN_DIGEST["builtin"]


def test_m4_oracle_dense_golden_manifest_bytes():
    # covers: RUNTIME-I18, RUNTIME-I19
    # WOULD-FAIL-IF: a dense digest, manifest or baseline changes: build_manifest's
    # bytes for a dense state move (a new section, key or encoding; BA-457, BA-465).
    import hashlib

    from wantrain.validation.state import build_manifest

    w = {
        "layers.0.tok.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7,
        "layers.1.ln.bias": torch.tensor([-0.0, 1.5, -2.25]),
        "layers.1.mask": torch.tensor([[1, 0], [0, 1]], dtype=torch.int64),
    }
    opt = {
        "layers.0.tok.weight": {
            "exp_avg": torch.full((3, 4), 0.125),
            "exp_avg_sq": torch.full((3, 4), 3.0e-5),
            "step": torch.tensor(2.0),
        },
        "layers.1.ln.bias": {
            "exp_avg": torch.tensor([1.0, -1.0, 0.5]),
            "exp_avg_sq": torch.tensor([1e-3, 2e-3, 3e-3]),
            "step": torch.tensor(2.0),
        },
    }
    hp = {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False}
    rng = torch.arange(16, dtype=torch.uint8)
    losses = [(1, 0, (5.5).hex()), (0, 1, (5.25).hex()), (0, 0, (5.75).hex())]

    def sha(m):
        return hashlib.sha256((json.dumps(m, indent=1, sort_keys=True) + "\n").encode()).hexdigest()

    assert sha(build_manifest(w, opt, hp, rng, losses)) == _M4_GOLDEN_MANIFEST
    # The deliberately wrong manifest (an empty aux section on a dense state) moves.
    assert sha(build_manifest(w, opt, hp, rng, losses, [])) != _M4_GOLDEN_MANIFEST


def test_m4_oracle_dense_golden_baseline(tmp_path, monkeypatch):
    # covers: RUNTIME-I18, RUNTIME-I19
    # WOULD-FAIL-IF: a dense digest, manifest or baseline changes: the in-process oracle
    # of the tiny dense config moves a loss bit or a state byte (the hf and MoE branches
    # leaking into the dense path, BA-407, BA-411).
    import hashlib

    from wantrain.model.gpt import GPTConfig
    from wantrain.validation.reference import run_baseline

    for name in ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)
    saved = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
    )
    spec = JobSpec(
        model=GPTConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, seq_len=16),
        steps=2,
        microbatches=2,
        mb_size=2,
        out_dir=str(tmp_path),
    )
    try:
        run_baseline(spec, "golden")
    finally:
        torch.set_num_threads(saved[0])
        torch.use_deterministic_algorithms(saved[1], warn_only=saved[2])
        torch.set_float32_matmul_precision(saved[3])
    with open(tmp_path / "golden_losses.json") as f:
        assert [r["loss_hex"] for r in json.load(f)["losses"]] == _M4_GOLDEN_LOSSES
    state = (tmp_path / "golden_state.json").read_bytes()
    assert hashlib.sha256(state).hexdigest() == _M4_GOLDEN_STATE
    assert b'"aux"' not in state


def test_m4_oracle_moe_recompute_replay_is_checked(tmp_path, monkeypatch):
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: the MoE oracle's recompute replay goes unchecked (torch's checkpoint
    # early stop ends it before moe_forward records, and MOE-REPLAY passes on nothing),
    # recompute moves a loss bit, an aux hex or a state byte, the aux section misses a
    # (step, mb, layer), or the selection biases are not moved after AdamW (BA-453).
    import contextlib

    from wantrain.model.gpt import GPTConfig
    from wantrain.validation.reference import run_baseline
    from wantrain.validation.state import tensor_record

    for name in ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)
    saved = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
    )
    spec = JobSpec(
        model=GPTConfig(
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
        ),
        steps=2,
        microbatches=2,
        mb_size=2,
        out_dir=str(tmp_path),
    )

    def run(tag, recompute):
        try:
            run_baseline(spec, tag, recompute=recompute)
        finally:
            torch.set_num_threads(saved[0])
            torch.use_deterministic_algorithms(saved[1], warn_only=saved[2])
            torch.set_float32_matmul_precision(saved[3])
        with open(tmp_path / f"{tag}_state.json") as f:
            return json.load(f)

    plain, rec = run("plain", False), run("rec", True)
    assert rec == plain, "recompute moved a loss, an aux hex or a state byte"
    triples = [(s, m, layer) for s, m, layer, _ in plain["aux"]]
    layers = sorted({t[2] for t in triples})
    assert len(layers) == 2 and len(set(triples)) == len(triples)
    assert triples == [(s, m, layer) for s in range(2) for m in range(2) for layer in layers]
    biases = [k for k in plain["weights"] if k.endswith("sel_bias")]
    assert len(biases) == 2
    zero = tensor_record(torch.zeros(6))
    assert all(plain["weights"][k] != zero for k in biases), "no selection bias moved"

    # The deliberately wrong replay (torch's default early stop) records nothing in
    # the backward; the oracle refuses it instead of passing on an empty check.
    monkeypatch.setattr(
        torch.utils.checkpoint, "set_checkpoint_early_stop", lambda enable: contextlib.nullcontext()
    )
    with pytest.raises(RuntimeError, match="MOE-REPLAY"):
        run("early", True)


def test_m4_oracle_hf_pins_and_gate_reports():
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: run_local's pinning of the builder child's record, census and hf_code
    # accepts a pre-filled pin or mirrored field that differs, leaves a mirrored field
    # unset, or the gate step admits a failed report or maps a slot to another device's
    # class (BA-424, BA-448, MODEL-I10).
    from wantrain.launch.local import gate_classes, gate_devices, pin_hf_record, slot_classes

    m4 = _m4_cfg_cls()
    out = {
        "record": {"v": 1, "tensors": {"model.rotary_emb.inv_freq": {"sha256": "ab"}}},
        "census": [{"unit": 0, "kind": "prologue"}],
        "hf_code": {"transformers": "5.19.0", "torch": "2.14.1"},
        "fields": {
            "n_layer": 2,
            "d_model": 48,
            "n_head": 3,
            "vocab_size": 96,
            "tie_embeddings": True,
        },
    }
    m = m4(arch="hf", hf_config={"model_type": "llama"})
    pin_hf_record(m, out)
    assert (m.n_layer, m.d_model, m.n_head, m.vocab_size, m.tie_embeddings) == (2, 48, 3, 96, True)
    assert (m.hf_buffers, m.hf_census, m.hf_code) == (out["record"], out["census"], out["hf_code"])
    pin_hf_record(m, json.loads(json.dumps(out)))  # an equal pre-filled pin is kept
    with pytest.raises(ValueError, match="n_layer=3"):
        pin_hf_record(m4(arch="hf", n_layer=3), out)
    with pytest.raises(ValueError, match="hf_code"):
        pin_hf_record(m4(arch="hf", hf_code={"transformers": "5.18.0"}), out)

    # CONTROL-I25: a CPU + MPS map is refused at finalize; one class per map runs.
    mixed = JobSpec(stages=2, stage_widths=[2, 1], devices=[["cpu", "mps"], ["cpu:0"]])
    with pytest.raises(ValueError, match="^MIXED-FLEET: the device map spans 2 device classes"):
        mixed.finalize()
    spec = JobSpec(stages=2, stage_widths=[2, 1], devices=[["mps", "mps"], ["mps"]]).finalize()
    assert gate_devices(spec) == ["cpu", "mps"]
    assert gate_devices(JobSpec(device="cpu")) == ["cpu"]
    classes = gate_classes(
        {
            "cpu": {"ok": True, "device_class": "cpu:arm64:DEFAULT"},
            "mps": {"ok": True, "device_class": "mps:Apple M4"},
        }
    )
    assert slot_classes(spec, classes) == [["mps:Apple M4", "mps:Apple M4"], ["mps:Apple M4"]]
    cpus = JobSpec(stages=2, stage_widths=[2, 1], devices=[["cpu", "cpu:0"], ["cpu"]]).finalize()
    assert slot_classes(cpus, classes) == [
        ["cpu:arm64:DEFAULT", "cpu:arm64:DEFAULT"],
        ["cpu:arm64:DEFAULT"],
    ]
    with pytest.raises(ValueError, match="failed on mps: GATE-DIFF: logits"):
        gate_classes(
            {
                "cpu": {"ok": True, "device_class": "cpu:arm64:DEFAULT"},
                "mps": {"ok": False, "cause": "GATE-DIFF", "detail": "logits"},
            }
        )


# ---------------------------------------------------------------- M4 identity (identity-m4, wave 2)
# HF families and native MoE against the single-device oracle, as real multi-process
# localhost CPU pipelines: the sha256 of weights, AdamW m, v and step, the RNG and the
# selection biases, and every loss and aux hex (RUNTIME-I18, RUNTIME-I19). Every HF model
# is a tiny config of its pinned transformers config class (hfm.tiny_config) or a
# checkpoint this repository's writer wrote; nothing is downloaded and the hub is never
# called (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE; the stages run in hfm.stage_env).

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_GEMMA2 = {"sliding_window": 8, "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0}
# id -> (model_type, overrides). The four families model-core proves first, then the
# rest of the 12-family sweep (model-families proves their gate).
_HF_FAMILIES = {
    "llama": ("llama", {"tie_word_embeddings": False}),
    "qwen3": ("qwen3", {"tie_word_embeddings": False}),
    "granite": (
        "granite",
        {
            "embedding_multiplier": 12.0,
            "logits_scaling": 8.0,
            "residual_multiplier": 0.2,
            "attention_multiplier": 0.25,
        },
    ),
    "gemma2-tied": ("gemma2", {**_GEMMA2, "tie_word_embeddings": True}),
    "mistral": ("mistral", {}),
    "qwen2": ("qwen2", {}),
    "gemma": ("gemma", {}),
    "gemma3-text": ("gemma3_text", {"sliding_window": 8}),
    "phi3": ("phi3", {}),
    "olmo2": ("olmo2", {}),
    "gpt-neox": ("gpt_neox", {"classifier_dropout": 0.0}),
    "starcoder2": ("starcoder2", {}),
}
_HF_CORE = ("llama", "qwen3", "granite", "gemma2-tied")
_WAN_40MS_1GBPS = dict(latency_ms=40, bw_mbps=1000)


def _hfm():
    from wantrain.model import hfm

    return hfm


def _hf_config(family: str, **over) -> dict:
    mt, base = _HF_FAMILIES[family]
    return _hfm().tiny_config(mt, use_cache=False, **{**base, **over})


def _hf_spec(
    out_dir, tag, port_base, family, compute="fp32", steps=3, microbatches=4, link=None, **kw
):
    """A 2-row-microbatch spec of a tiny HF model; prepare_hf pins the record, census,
    hf_code and the mirrored fields at its first launch."""
    from wantrain.model.gpt import GPTConfig

    hf_over = kw.pop("hf_over", {})
    model_kw = kw.pop("model_kw", {})
    model = GPTConfig(
        arch="hf",
        hf_config=_hf_config(family, **hf_over),
        seq_len=16,
        compute_dtype=compute,
        **model_kw,
    )
    spec = JobSpec(
        model=model,
        steps=steps,
        microbatches=microbatches,
        mb_size=2,
        out_dir=str(out_dir),
        tag=tag,
        port_base=port_base,
        dial_timeout_s=60,
        **kw,
    )
    spec.links = links_for(spec, **(link or {}))
    return spec


def _platform_tag() -> str:
    import importlib.metadata as md

    return _hfm()._platform_tag(md.distribution("torch"))


def _hf_refused_off_platform(fn) -> bool:
    """BA-463, HANDOUT D-9: where this platform has no TORCH_RECORDS entry every arm
    through run_local is refused FP-TORCH-RECORDS, naming the platform tag (strict,
    never a skip). -> True when the arm ended there."""
    if _hfm().platform_admitted():
        return False
    with pytest.raises(ValueError, match="FP-TORCH-RECORDS") as e:
        fn()
    assert _platform_tag() in str(e.value), str(e.value)
    return True


def _no_prefix_left(out_dir) -> None:
    """BA-423: no -X pycache_prefix directory outlives its process."""
    left = sorted(d for d in os.listdir(out_dir) if d.startswith("pyc-"))
    assert left == [], f"prefix directories survived the run: {left}"


def _hf_exact(spec, timeout=300):
    """run_and_compare of an hf spec, then its canonical state against the oracle's.
    -> the state manifest, or None when the platform refused the arm (BA-463)."""
    if _hf_refused_off_platform(lambda: run_and_compare(spec, timeout=timeout)):
        return None
    res = run_and_compare(spec, timeout=timeout)
    assert res.get("ok"), f"{res}\n{_log_tails(spec)}"
    assert res["compared"] == spec.steps * spec.microbatches and res["max_abs_diff"] == 0.0, res
    _no_prefix_left(spec.out_dir)
    return _assert_exact(spec, spec.out_dir)


def _hf_oracle_child(spec, tag, pre: str = "") -> subprocess.CompletedProcess:
    """The oracle as its own process (python -X pycache_prefix=<fresh> in the stage's
    environment, as ensure_reference runs it), with `pre` (test code: a patch of the
    run, never of the product files) executed first; its output captured."""
    from wantrain.launch import local

    os.makedirs(spec.out_dir, exist_ok=True)
    path = os.path.join(spec.out_dir, f"{tag}_spec.json")
    with open(path, "w") as f:
        f.write(spec.to_json())
    code = (
        "import sys\n"
        "from wantrain.validation import reference as R\n"
        f"{pre}\n"
        f"sys.exit(R.main(['--spec', {path!r}, '--tag', {tag!r}, '--out-dir', {spec.out_dir!r}]))\n"
    )
    return local.run_child(
        ["-c", code], local._hf_env(), spec.out_dir, 600, capture_output=True, text=True
    )


def _oracle_state(out_dir, tag) -> dict:
    with open(os.path.join(str(out_dir), f"{tag}_state.json")) as f:
        return json.load(f)


# A spy on the mode each oracle run's forward took (printed at exit), so neither arm of
# the vanilla-vs-chain comparison can silently run the other's path. _CHAIN_PRE forces the
# unit chain in one process (oracle mode "chain"), on this host's one device: the patch
# only picks the mode; the chain is the product's own code (_hf_build, _hf_loss).
_MODE_SPY = """
import json, os
_modes = []
_loss = R._hf_loss
def _spy(spec, units, devices, mode, *a, **k):
    _modes.append(mode)
    return _loss(spec, units, devices, mode, *a, **k)
R._hf_loss = _spy
import atexit
atexit.register(lambda: print("ORACLE_MODES " + json.dumps(sorted(set(_modes)))))
"""
_CHAIN_PRE = _MODE_SPY + 'R.oracle_mode = lambda classes, hosts=None: "chain"\n'


@pytest.mark.parametrize("compute", ["fp32", "bf16"])
def test_hf_oracle_vanilla_equals_the_unit_chain(tmp_path, compute):
    """On a tiny llama, HF's own ForCausalLM.forward (the vanilla oracle of a one-class
    map) and the run's units chained in one process give the same losses and the same
    state manifest, bit for bit, in fp32 and under bf16 autocast (MODEL-I9, BA-420)."""
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the vanilla oracle and the unit chain differ by a loss bit or a state
    # byte (a unit window that is not HF's forward, a tied table counted twice, autocast
    # applied on one path only), or the chain arm silently ran the vanilla path.
    from wantrain.launch.local import prepare_hf

    spec = _hf_spec(tmp_path, "orc", 0, "llama", compute=compute, stages=2)
    if _hf_refused_off_platform(lambda: prepare_hf(spec)):
        return
    prepare_hf(spec)
    vanilla = _hf_oracle_child(spec, "vanilla", _MODE_SPY)
    assert vanilla.returncode == 0, vanilla.stderr[-3000:]
    assert 'ORACLE_MODES ["vanilla"]' in vanilla.stdout, vanilla.stdout[-500:]
    chain = _hf_oracle_child(spec, "chain", _CHAIN_PRE)
    assert chain.returncode == 0, chain.stderr[-3000:]
    assert 'ORACLE_MODES ["chain"]' in chain.stdout, chain.stdout[-500:]
    a, b = _oracle_state(tmp_path, "vanilla"), _oracle_state(tmp_path, "chain")
    assert diff_manifests(a, b) == [] and a == b
    assert len(a["losses"]) == spec.steps * spec.microbatches
    # The comparison discriminates: the same oracle on another data seed moves the losses.
    other = JobSpec.from_json(spec.to_json())
    other.data_seed += 1
    moved = _hf_oracle_child(other, "moved")
    assert moved.returncode == 0, moved.stderr[-3000:]
    assert _oracle_state(tmp_path, "moved")["losses"] != a["losses"]


def test_hf_oracle_gate_refusal_stops_before_any_stage(tmp_path, port_base):
    """A recipe the launch refuses (attention dropout 0.1) makes run_local refuse before
    any stage process starts, naming the cause (MODEL-I10, BA-424)."""
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: run_local starts a stage (a log, a spec file) for a recipe the gate or
    # the record child refuses, or the refusal names no cause the dropout makes (DROPOUT, or
    # TRACE-RNG: the record child's trace sees the forward draw from the generator).
    spec = _hf_spec(
        tmp_path, "drop", port_base, "llama", stages=2, hf_over={"attention_dropout": 0.1}
    )
    if not _hf_refused_off_platform(lambda: run_local(spec, timeout=120)):
        # On an admitted platform the refusal must be one the dropout causes, not the
        # platform's (FP-TORCH-RECORDS would refuse any recipe and prove nothing here).
        with pytest.raises((ValueError, RuntimeError), match="DROPOUT|TRACE-RNG"):
            run_local(spec, timeout=120)
    left = sorted(f for f in os.listdir(tmp_path) if f.endswith((".log", "_spec.json")))
    assert left == [], f"a stage started: {left}"
    _no_prefix_left(tmp_path)


def _edit_one_buffer(record: dict) -> str:
    """Flip one byte of one tensor of a buffer record, keeping its sha256. -> its name."""
    import base64

    name = sorted(record["tensors"])[0]
    raw = bytearray(base64.b64decode(record["tensors"][name]["b64"]))
    raw[0] ^= 0x01
    record["tensors"][name]["b64"] = base64.b64encode(bytes(raw)).decode()
    return name


def test_hf_oracle_edited_buffer_record_is_refused_naming_it(tmp_path, port_base):
    """A spec whose hf_buffers has one tensor's bytes edited is refused, naming the
    tensor: by the oracle at its start and by run_local before any stage (RUNTIME-I19)."""
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: an edited buffer in the pinned record loads (the oracle or a launch
    # trains on bytes the record's sha256 does not cover), or the refusal does not name
    # the tensor.
    from wantrain.launch.local import prepare_hf

    spec = _hf_spec(tmp_path, "buf", port_base, "llama", stages=2)
    if _hf_refused_off_platform(lambda: prepare_hf(spec)):
        return
    prepare_hf(spec)
    good = _hf_oracle_child(spec, "good")
    assert good.returncode == 0, good.stderr[-3000:]
    bad = JobSpec.from_json(spec.to_json())
    name = _edit_one_buffer(bad.model.hf_buffers)
    res = _hf_oracle_child(bad, "bad")
    assert res.returncode != 0
    assert "BUFFER-PIN" in res.stderr and name in res.stderr, res.stderr[-2000:]
    launch = JobSpec.from_json(bad.to_json())
    launch.tag, launch.device_classes = "buf2", None
    with pytest.raises(ValueError, match=f"hf_buffers.*{name}"):
        run_local(launch, timeout=120)
    assert not [f for f in os.listdir(tmp_path) if f.startswith("buf2_r")], "a stage started"


def test_hf_oracle_resume_refuses_a_kept_record_that_differs(tmp_path):
    """A local resume whose kept transformers RECORD text differs from the installed one
    in one line is refused, naming the path (BA-448); the unedited text is admitted."""
    # covers: RUNTIME-I19, RUNTIME-I15
    # WOULD-FAIL-IF: a resume rebuilds the pins instead of checking them, ignores the kept
    # RECORD, or refuses without naming the differing path.
    from wantrain.launch.local import prepare_hf, record_path

    spec = _hf_spec(tmp_path, "rec", 0, "llama", stages=2)
    if _hf_refused_off_platform(lambda: prepare_hf(spec)):
        return
    prepare_hf(spec)
    resumed = JobSpec.from_json(spec.to_json())
    resumed.resume = True
    prepare_hf(resumed)  # the kept text equals the installed RECORD
    with open(record_path(spec)) as f:
        lines = f.read().splitlines(keepends=True)
    i = next(i for i, ln in enumerate(lines) if ln.startswith("transformers/") and ",sha256=" in ln)
    path = lines[i].split(",")[0]
    lines[i] = lines[i].replace(",sha256=", ",sha256=A", 1)
    with open(record_path(spec), "w") as f:
        f.write("".join(lines))
    with pytest.raises(ValueError, match=f"RECORD at {path}$"):
        prepare_hf(JobSpec.from_json(resumed.to_json()))


def test_hf_oracle_recipe_digest_covers_the_map_and_the_code(tmp_path):
    """recipe_digest of a pinned hf spec moves with its device map, its device classes,
    its hf_code and its HF config, and not with a field that decides no number (BA-422)."""
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: a cached hf reference is reused across another map, another class,
    # another transformers or another config (a stale oracle), or the tag enters the key.
    from wantrain.launch.local import prepare_hf, recipe_digest

    spec = _hf_spec(tmp_path, "dig", 0, "llama", stages=2)
    if _hf_refused_off_platform(lambda: prepare_hf(spec)):
        return
    prepare_hf(spec)
    base = recipe_digest(spec)

    def moved(edit) -> str:
        s = JobSpec.from_json(spec.to_json())
        edit(s)
        return recipe_digest(s)

    assert moved(lambda s: setattr(s, "tag", "other")) == base
    others = {
        "devices": moved(lambda s: setattr(s, "devices", [["cpu"], ["cpu:0"]])),
        "classes": moved(lambda s: setattr(s, "device_classes", [["cpu:x"], ["cpu:x"]])),
        "hf_code": moved(lambda s: s.model.hf_code.update(transformers_record_sha256="0" * 64)),
        "hf_config": moved(lambda s: s.model.hf_config.update(rms_norm_eps=1e-5)),
    }
    assert base not in others.values() and len(set(others.values())) == 4, others


_START_PATCH = """
import torch
from torch import nn
_build = R.build_layers
def _patched(cfg, seed, span=None):
    units = _build(cfg, seed) if span is None else _build(cfg, seed, span)
    root = units.root
    {edit}
    return units
R.build_layers = _patched
"""
_START_EDITS = {
    "_attn_implementation": "root.config._attn_implementation = 'sdpa'",
    "dropout": "root.model.layers[0].self_attn.extra_dropout = nn.Dropout(0.1)",
    "training mode": "root.model.layers[1].mlp.eval()",
    "hf_code": None,
}


@pytest.mark.parametrize("flag", sorted(_START_EDITS))
def test_hf_oracle_start_asserts_are_fatal_naming_the_flag(tmp_path, flag):
    """BA-406: the oracle's start checks after the build: sdpa attention, a dropout 0.1
    module, one module in eval mode and an hf_code other than the installed one are each
    fatal, naming the flag; the unedited oracle starts and finishes."""
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the oracle trains with sdpa, an active dropout, an eval-mode module
    # or under another transformers than the pin, or refuses without naming the flag.
    from wantrain.launch.local import prepare_hf

    spec = _hf_spec(tmp_path, "start", 0, "llama", stages=2, steps=1, microbatches=1)
    if _hf_refused_off_platform(lambda: prepare_hf(spec)):
        return
    prepare_hf(spec)
    edit = _START_EDITS[flag]
    if edit is None:
        spec.model.hf_code = dict(spec.model.hf_code, transformers_record_sha256="0" * 64)
        pre = ""
    else:
        pre = _START_PATCH.format(edit=edit)
    res = _hf_oracle_child(spec, "asserts", pre)
    assert res.returncode != 0, res.stdout[-500:]
    assert f"oracle refused at start: {flag}:" in res.stderr, res.stderr[-2000:]
    assert not os.path.exists(tmp_path / "asserts_state.json")


# id -> (family, JobSpec fields, link profile). Chains 2-4, WAN emulation (40 ms one-way,
# 1 Gbps), recompute, W fill, bf16 autocast, a ZeRO-1 width-2 middle stage (untied only,
# BA-447) and M2b's pieces (wire native) on the HF units.
_HF_ARMS = {
    **{
        f"chain2-{fam}": (fam, dict(stages=2, schedule_fill=False), {})
        for fam in ("llama", "qwen3", "granite", "gemma2-tied")
    },
    "chain3-wan": ("llama", dict(stages=3), _WAN_40MS_1GBPS),
    "chain4-wan": ("llama", dict(stages=4), _WAN_40MS_1GBPS),
    "gemma2-tied-chain3": ("gemma2-tied", dict(stages=3), {}),
    "recompute": ("llama", dict(stages=3, recompute=True), {}),
    "wfill": ("llama", dict(stages=3, microbatches=8, schedule_fill=True), {}),
    # bf16 activations travel in the exact planes codec (raw frames only fp32 and int64).
    "bf16": ("llama", dict(stages=2, compute="bf16", wire_codec="planes"), {}),
    "zero1-mid-llama": ("llama", dict(stages=3, stage_widths=[1, 2, 1]), {}),
    "zero1-mid-granite": ("granite", dict(stages=3, stage_widths=[1, 2, 1]), {}),
    "pieces-native": (
        "llama",
        dict(stages=3, stage_widths=[1, 2, 1], wire="native", shard="pieces", stream_grads=True),
        {},
    ),
}


def _tied_clean(spec) -> None:
    """F30: every slot ended its attempt with no TIED frame refused, missing or left
    in flight at END (a frame past the last step is refused by the sequence check)."""
    for slot, lines in _slot_lines(spec, "").items():
        bad = [ln for ln in lines if "TIED" in ln]
        assert bad == [], (slot, bad)


@pytest.mark.parametrize("arm", sorted(_HF_ARMS))
def test_hf_identity_chain(tmp_path, port_base, arm):
    """HF units as a real multi-process CPU pipeline: losses hex and the state manifest
    (weights, AdamW m, v, step, RNG) equal the single-device oracle's, 3 steps, M 4 (8
    for W fill), fp32 unless the arm says bf16; gemma2's tied table is split between
    the embedding's stage and the Head's (and relayed through a middle stage in chain3)."""
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: an HF unit window, the tied relay (TIED rows, the Head's row fold,
    # the middle stage's forwarding), ZeRO-1 or pieces over HF units, recompute, W fill or
    # WAN pacing moves a loss bit or a state byte, or a pycache prefix outlives its stage.
    family, kw, link = _HF_ARMS[arm]
    spec = _hf_spec(tmp_path, "hfid", port_base, family, link=link, **kw)
    state = _hf_exact(spec)
    if state is None:
        return
    if family == "gemma2-tied":
        assert spec.tied_relay(), "the tied table is not split across stages"
        _tied_clean(spec)
    if arm == "chain2-llama":
        # The identity discriminates: the oracle of another data seed differs.
        other = JobSpec.from_json(spec.to_json())
        other.data_seed += 1
        res = _hf_oracle_child(other, "other_seed")
        assert res.returncode == 0, res.stderr[-2000:]
        assert diff_manifests(state, _oracle_state(tmp_path, "other_seed")) != []


@pytest.mark.parametrize(
    "family", sorted(_HF_FAMILIES), ids=[f"sweep-{f}" for f in sorted(_HF_FAMILIES)]
)
def test_hf_identity_family(tmp_path, port_base, family):
    """The 12-family sweep (GATES-M4 hf-identity-x2): each family's tiny model as a
    2-stage pipeline equals its oracle by sha256 and every loss hex."""
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: one family's units, buffers or tied table move a bit between the
    # pipeline and the vanilla oracle.
    spec = _hf_spec(tmp_path, "sweep", port_base, family, stages=2)
    _hf_exact(spec)


def _hf_part_then_resume(tmp_path, port_base, family, tag, **kw):
    """2 of 3 steps with a wave per step (a checkpoint-and-stop at step 1), then a
    resume from the step-1 waves to step 3. -> (part spec, resumed spec, part state)."""
    part = _hf_spec(tmp_path, tag, port_base, family, steps=2, ckpt_every=1, **kw)
    if _hf_refused_off_platform(lambda: run_local(part, timeout=300)):
        return None
    res = run_local(part, timeout=300)
    assert res["ok"], f"{res}\n{_log_tails(part)}"
    _no_prefix_left(tmp_path)
    with open(tmp_path / f"{tag}_spec.json") as f:
        resumed = JobSpec.from_json(f.read())
    part_state = distributed_manifest(JobSpec.from_json(resumed.to_json()))
    resumed.steps = 3
    resumed.resume = True
    resumed.port_base = port_base + 32
    return part, resumed, part_state


@pytest.mark.parametrize("arm", ["resume"])
def test_hf_identity_resume(tmp_path, port_base, arm):
    """A resume from a committed wave (step 1) to step 3 equals the 3-step oracle; the
    resumed launch checks its pins and its kept RECORD, never rebuilding them."""
    # covers: RUNTIME-I19, RUNTIME-I15
    # WOULD-FAIL-IF: a resumed hf slot rebuilds its units from the init seed instead of
    # loading its wave, re-pins the record, or loads another slot's state.
    got = _hf_part_then_resume(tmp_path, port_base, "llama", "res", stages=2)
    if got is None:
        return
    _, resumed, part_state = got
    state = _hf_exact(resumed)
    for r in range(resumed.stages):
        with open(tmp_path / f"res_r{r}p0.log", errors="replace") as f:
            assert f"WT_RESUME rank={r} start_step=2 " in f.read(), r
    # A run that ignored its waves (the 2-step state) is not the 3-step oracle's.
    assert diff_manifests(part_state, state) != []


@pytest.mark.parametrize("arm", ["tied-end"])
def test_hf_identity_tied_end(tmp_path, port_base, arm):
    """F30 on gemma2's split table: the attempt cut at a checkpoint-and-stop step and the
    run's last step both end with no TIED frame in flight and a clean END drain; the
    resumed run equals the oracle."""
    # covers: RUNTIME-I19, RUNTIME-I15
    # WOULD-FAIL-IF: the Head sends TIED(t+1) after an attempt's last step (a frame in
    # flight at END, refused or missing), or a resumed stage 0 lacks the rows R_k of its
    # first step.
    got = _hf_part_then_resume(tmp_path, port_base, "gemma2-tied", "tend", stages=2)
    if got is None:
        return
    part, resumed, _ = got
    _tied_clean(part)
    _hf_exact(resumed)
    _tied_clean(resumed)


@pytest.mark.parametrize("arm", ["buffers-edited"])
def test_hf_identity_buffers_edited(tmp_path, port_base, arm):
    """A wave whose hf_buffers.json has one tensor edited (its wave.json entry rewritten
    to match, so only the buffer check can see it) is refused at resume, naming the
    tensor, before any state is applied (RUNTIME-I19, BA-417)."""
    # covers: RUNTIME-I19, RUNTIME-I15
    # WOULD-FAIL-IF: a resume loads a wave whose buffer record differs from the pin, or
    # refuses without naming the tensor.
    import base64
    import hashlib

    got = _hf_part_then_resume(tmp_path, port_base, "llama", "bufw", stages=2)
    if got is None:
        return
    _, resumed, _ = got
    wave = tmp_path / "waves" / "bufw" / "r1p0" / "step-00000001"
    rec = json.loads((wave / "hf_buffers.json").read_bytes())
    name = sorted(rec["tensors"])[0]
    raw = bytearray(base64.b64decode(rec["tensors"][name]["b64"]))
    raw[-1] ^= 0x01
    rec["tensors"][name]["b64"] = base64.b64encode(bytes(raw)).decode()
    data = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()
    (wave / "hf_buffers.json").write_bytes(data)
    man = json.loads((wave / "wave.json").read_bytes())
    for ent in man["files"]:
        if ent["name"] == "hf_buffers.json":
            ent.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    (wave / "wave.json").write_text(json.dumps(man))
    res = run_local(resumed, timeout=180)
    assert not res["ok"]
    with open(tmp_path / "bufw_r1p0.log", errors="replace") as f:
        log = f.read()
    assert f"BUFFER-PIN {name}" in log, log[-2000:]
    assert "WT_RESUME rank=1" not in log, "state was applied before the refusal"


@pytest.mark.parametrize("arm", ["ckpt-span"])
def test_hf_identity_ckpt_span(tmp_path, port_base, arm):
    """BA-449: a sharded tiny gemma2 checkpoint written by hfm.write_safetensors, its tied
    table split across 2 stages, each stage loading only its units (the rest on meta),
    and the oracle loading the same checkpoint: the run equals the oracle, and the
    loaded weights are not the init seed's."""
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: a stage loads another span's units or misses one, the split tied
    # table is loaded on one side only, or the oracle and the run read different bytes.
    from wantrain.model import hf
    from wantrain.model.gpt import GPTConfig

    hfm = _hfm()
    hc = _hf_config("gemma2-tied")
    rec = hfm.buffer_record(hc)
    units = hfm.build(GPTConfig(arch="hf", hf_config=hc, hf_buffers=rec, seq_len=16), 99)
    tensors = {n: p.detach().clone() for n, p in units.root.named_parameters()}
    ckpt = tmp_path / "ckpt"
    hfm.write_safetensors(tensors, str(ckpt), shard_bytes=64 * 1024)
    shards = sorted(f for f in os.listdir(ckpt) if f.endswith(".safetensors"))
    assert len(shards) > 1, shards
    spec = _hf_spec(
        tmp_path,
        "span",
        port_base,
        "gemma2-tied",
        stages=2,
        model_kw=dict(hf_model=str(ckpt), hf_model_sha256=hf.checkpoint_digest(str(ckpt))),
    )
    state = _hf_exact(spec)
    if state is None:
        return
    assert spec.tied_relay()
    # The checkpoint decided the weights: the same recipe from the init seed differs.
    seeded = JobSpec.from_json(spec.to_json())
    seeded.model.hf_model = seeded.model.hf_model_sha256 = None
    res = _hf_oracle_child(seeded, "seeded")
    assert res.returncode == 0, res.stderr[-2000:]
    assert _oracle_state(tmp_path, "seeded")["losses"] != state["losses"]


# ---- native MoE (ADR-0008 E2, E3, E4, E6, F7 on arch.MoEMLP)
# E1's sizes: h 64, 4 heads, 4 MoE blocks, E 8, f 96, V 256, T 32 (2 rows of 16), M 4,
# 3 steps; one recipe per row of derivation section 3.
_MOE_BASE = dict(
    vocab_size=256,
    n_layer=4,
    n_head=4,
    d_model=64,
    seq_len=16,
    norm="rmsnorm",
    mlp="swiglu",
    bias=False,
    pos="rope",
    moe_experts=8,
    moe_top_k=2,
    moe_d_ff=96,
)
_MOE_ROWS = {
    "softmax": dict(moe_lbl_coef=0.01),
    "deepseek": dict(
        moe_score="sigmoid",
        moe_select="score_bias",
        moe_groups=4,
        moe_topk_groups=2,
        moe_norm=True,
        moe_scale=2.5,
        moe_bias_rate=0.001,
        moe_shared_d_ff=64,
        moe_shared_gate="none",
    ),
    "gptoss": dict(moe_score="softmax_k", moe_router_bias=True, moe_expert="swiglu_clamped"),
    "llama4": dict(
        moe_top_k=1,
        moe_apply="input",
        moe_score="sigmoid",
        moe_shared_d_ff=96,
        moe_shared_gate="none",
    ),
    "olmoe": dict(moe_lbl_coef=0.01, moe_z_coef=0.001),
    "qwen2": dict(moe_shared_d_ff=96, moe_shared_gate="sigmoid", moe_lbl_coef=0.001),
}


def _moe_model(row="softmax", compute="fp32", **over):
    from wantrain.model.gpt import GPTConfig

    return GPTConfig(**{**_MOE_BASE, **_MOE_ROWS[row], **over}, compute_dtype=compute)


def _moe_spec(
    out_dir, tag, port_base, row="softmax", compute="fp32", steps=3, microbatches=4, **kw
):
    model_kw = kw.pop("model_kw", {})
    link = kw.pop("link", {})
    kw.setdefault("wire_codec", "raw" if compute == "fp32" else "planes")
    spec = JobSpec(
        model=_moe_model(row, compute, **model_kw),
        steps=steps,
        microbatches=microbatches,
        mb_size=2,
        out_dir=str(out_dir),
        tag=tag,
        port_base=port_base,
        dial_timeout_s=60,
        **kw,
    )
    spec.links = links_for(spec, **link)
    return spec


def _moe_exact(spec, timeout=300):
    """run_and_compare, then the state with selection biases and every aux hex per
    (step, mb, layer) against the oracle's. -> the state manifest."""
    res = run_and_compare(spec, timeout=timeout)
    assert res.get("ok"), f"{res}\n{_log_tails(spec)}"
    assert res["compared"] == spec.steps * spec.microbatches and res["max_abs_diff"] == 0.0, res
    state = _assert_exact(spec, spec.out_dir)
    layers = sorted({layer for _, _, layer, _ in state["aux"]})
    assert len(layers) == spec.model.n_layer
    want = [
        (s, m, layer)
        for s in range(spec.steps)
        for m in range(spec.microbatches)
        for layer in layers
    ]
    assert [tuple(r[:3]) for r in state["aux"]] == want, "aux rows missing or recorded twice"
    assert sum(k.endswith("sel_bias") for k in state["weights"]) == spec.model.n_layer
    return state


# Test-only patches of the stage and oracle processes (a sitecustomize on PYTHONPATH,
# enabled by name through WT_TEST_PATCHES): router initializations for the skew arms
# (applied to the oracle and the run alike), trigger records, and ADR-0008 E4's
# must-fail alternatives (applied to the stages only, after the oracle ran clean). Each
# trigger is written to WT_TEST_TRIG as it happens, so a run that dies keeps it.
_MOE_SITE = r"""
import json, os
_names = set(filter(None, os.environ.get("WT_TEST_PATCHES", "").split(",")))
if _names:
    import inspect, textwrap
    import torch
    from wantrain.model import arch as A

    _trig = {}

    def _hit(key, n=1):
        if not n:
            return
        _trig[key] = _trig.get(key, 0) + int(n)
        d = os.environ.get("WT_TEST_TRIG")
        if d:
            with open(os.path.join(d, f"trig_{os.getpid()}.json"), "w") as f:
                json.dump(_trig, f)

    def _patched(fn, old, new):
        src = textwrap.dedent(inspect.getsource(fn))
        assert old in src, (fn.__name__, old)
        ns = dict(fn.__globals__)
        exec(compile(src.replace(old, new), f"<patched {fn.__name__}>", "exec"), ns)
        return ns[fn.__name__]

    def _init(edit):
        build = A.build_layers

        def bl(cfg, seed, *a, **k):
            layers = build(cfg, seed, *a, **k)
            with torch.no_grad():
                for m in layers.modules():
                    if isinstance(m, A.Router):
                        edit(m.weight)
                        if m.bias is not None:
                            m.bias.zero_()
            return layers

        A.build_layers = bl

    if "init-tie" in _names:  # an all-tie router: every key equal at step 0
        _init(lambda w: w.zero_())
    if "init-hot" in _names:  # a hot expert
        _init(lambda w: w[0].mul_(40.0))
    if "init-triple" in _names:  # experts 0, 3 and 6 tie on every token at step 0
        _init(lambda w: (w[3].copy_(w[0]), w[6].copy_(w[0])))
    if "init-groups" in _names:  # groups {0,1}, {2,3} and {4,5} tie on every token at step 0
        _init(lambda w: (w[2:4].copy_(w[0:2]), w[4:6].copy_(w[0:2])))

    if "record-skew" in _names:
        _sel0, _fwd0 = A.moe_select, A.moe_forward

        def _rec_select(key, k, groups=1, topk_groups=1):
            if k < key.shape[1]:
                s = torch.sort(key, dim=-1, descending=True, stable=True).values
                _hit("ties", int((s[:, k - 1] == s[:, k]).sum()))
            return _sel0(key, k, groups, topk_groups)

        def _rec_forward(cfg, view, x):
            out = _fwd0(cfg, view, x)
            _hit("unrouted", int(bool((view.moe_records[-1].counts == 0).any())))
            return out

        A.moe_select, A.moe_forward = _rec_select, _rec_forward
    _select, _forward = A.moe_select, A.moe_forward

    if "topk-stage" in _names:
        def _topk(key, k, groups=1, topk_groups=1):
            exact = _select(key, k, groups, topk_groups)
            got = torch.sort(torch.topk(key, k, dim=-1).indices, dim=-1).values
            _hit("trigger", int((got != exact).any(-1).sum()))
            return got

        A.moe_select = _topk
    if "group-topk" in _names:
        _wrong = _patched(
            A.moe_select,
            "chosen = _stable_desc(gkey)[:, :topk_groups]",
            "chosen = torch.topk(gkey, topk_groups, dim=-1).indices",
        )

        def _gsel(key, k, groups=1, topk_groups=1):
            exact = _select(key, k, groups, topk_groups)
            got = _wrong(key, k, groups, topk_groups)
            _hit("trigger", int((got != exact).any(-1).sum()))
            return got

        A.moe_select = _gsel
    if "skip-unrouted" in _names:
        def _skip(self, xd, counts):
            _hit("trigger", sum(1 for n in counts if n == 0))
            return torch.cat([e(r) for e, r, n in zip(self, torch.split(xd, counts), counts) if n])

        A.ExpertList.forward = _skip
    if "reverse-combine" in _names:
        _rev = _patched(A._Combine.forward, "for j in range(k):", "for j in reversed(range(k)):")

        def _rc(ctx, Y, w, pos, perm):
            _hit("trigger", pos.shape[0] if pos.shape[1] >= 3 else 0)
            return _rev(ctx, Y, w, pos, perm)

        A._Combine.forward = staticmethod(_rc)
    if "descending-rows" in _names:
        _desc = _patched(_forward, "flat_e * T + flat_t", "flat_e * T + (T - 1 - flat_t)")

        def _df(cfg, view, x):
            out = _desc(cfg, view, x)
            _hit("trigger", int(bool((view.moe_records[-1].counts >= 2).any())))
            return out

        A.moe_forward = _df
    if "aux-alpha-m" in _names:
        _M = float(os.environ["WT_TEST_M"])
        _bwd = A._AuxInject.backward

        def _alpha_m(ctx, g):
            _hit("trigger", len(ctx.coefs))
            ctx.coefs = tuple(c / _M for c in ctx.coefs)
            return _bwd(ctx, g)

        A._AuxInject.backward = staticmethod(_alpha_m)
    if "bias-float-mean" in _names:
        _qm = {}
        _fwd2 = A.moe_forward

        def _qf(cfg, view, x):
            out = _fwd2(cfg, view, x)
            with torch.no_grad():
                lg = view.router(x.detach().reshape(-1, x.shape[-1]).float()).float()
                _qm.setdefault(view.moe_layer, []).append(torch.softmax(lg, -1).mean(0))
            return out

        def _float_bias(layers, counts, global_tokens):
            for layer in layers:
                for m in layer.modules():
                    if getattr(m, "moe_records", None) is None or m.moe_layer not in counts:
                        continue
                    r = A.moe_recipe(m.moe_cfg)
                    E, k, gamma = r.moe_experts, r.moe_top_k, float(r.moe_bias_rate)
                    c = counts[m.moe_layer]
                    exact = torch.sign(global_tokens * k - E * c)
                    q = torch.stack(_qm.pop(m.moe_layer)).mean(0)
                    wrong = torch.sign(torch.tensor(k / E, dtype=torch.float32) - q).to(torch.int64)
                    _hit("trigger", int((wrong != exact).sum()))
                    with torch.no_grad():
                        moved = m.sel_bias + wrong.to(torch.float32) * gamma
                        m.sel_bias.copy_(torch.where(wrong == 0, m.sel_bias, moved))

        A.moe_forward, A.moe_bias_update = _qf, _float_bias
    if "bias-one-peer" in _names:
        from wantrain.runtime import stage as S

        _bias_step = S.StageRuntime._moe_bias_step

        def _one_peer(self, step, chunk):
            if not (self._my_bias and self.wide and self.w > 1):
                return _bias_step(self, step, chunk)
            full = self._count_fold(step).result()
            self._cfold.pop(step)
            mine = self._moe_counts.pop(step, {})
            t_own = len(mine) * self.spec.mb_size * self.spec.model.seq_len
            for m in [self._bias_mods[j] for j in self._my_bias]:
                r = A.moe_recipe(m.moe_cfg)
                E, k, gamma = r.moe_experts, r.moe_top_k, float(r.moe_bias_rate)
                layer = int(m.moe_layer)
                own = sum((rec[layer] for rec in mine.values()), torch.zeros(E, dtype=torch.int64))
                t_g = self.M * self.spec.mb_size * self.spec.model.seq_len
                exact = torch.sign(t_g * k - E * full[layer])
                wrong = torch.sign(t_own * k - E * own)
                _hit("trigger", int((wrong != exact).sum()))
                with torch.no_grad():
                    moved = m.sel_bias + wrong.to(torch.float32) * gamma
                    m.sel_bias.copy_(torch.where(wrong == 0, m.sel_bias, moved))

        S.StageRuntime._moe_bias_step = _one_peer
    if "pieces-off-qt" in _names:
        from wantrain.runtime import pieces as P

        _q = P.quantum

        def _off(shape):
            q = P.row_length(shape)
            _hit("trigger", int(q != _q(shape)))
            return q

        P.quantum = _off
    if "count-dropped" in _names or "count-doubled" in _names:
        from wantrain.runtime import exchange as X
        from wantrain.runtime import stage as S

        _send = S.StageRuntime._send_exchange
        _twice = "count-doubled" in _names

        def _count_lane(self, ftype, kind, dest, unit, step, mb, tensors):
            if kind != X.COUNT:
                return _send(self, ftype, kind, dest, unit, step, mb, tensors)
            _hit("trigger")
            if _twice:
                _send(self, ftype, kind, dest, unit, step, mb, tensors)
                return _send(self, ftype, kind, dest, unit, step, mb, tensors)
            return None

        S.StageRuntime._send_exchange = _count_lane
"""


def _moe_site(tmp_path, monkeypatch, patches=""):
    """Install the patch sitecustomize for every child process; -> the trigger dir."""
    site = tmp_path / "site"
    trig = tmp_path / "trig"
    site.mkdir(exist_ok=True)
    trig.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(_MOE_SITE)
    monkeypatch.setenv("PYTHONPATH", str(site))
    monkeypatch.setenv("WT_TEST_TRIG", str(trig))
    monkeypatch.setenv("WT_TEST_PATCHES", patches)
    return trig


def _triggers(trig) -> dict:
    out: dict = {}
    for name in os.listdir(trig):
        with open(os.path.join(trig, name)) as f:
            for k, v in json.load(f).items():
                out[k] = out.get(k, 0) + v
    return out


_MOE_ARMS = {
    **{f"depth{d}-{row}": (row, dict(stages=d)) for d in (1, 2, 4) for row in sorted(_MOE_ROWS)},
    **{
        f"wfill-{row}": (row, dict(stages=4, microbatches=8, schedule_fill=True))
        for row in ("softmax", "deepseek")
    },
    **{
        f"recompute-{row}": (row, dict(stages=2, recompute=True)) for row in ("softmax", "deepseek")
    },
    # ZeRO-1 width 2 on the middle stage, whose peer 1 runs no microbatch yet owns a
    # selection bias (its counts all arrive from peer 0), on each gradient path (BA-454).
    "zero1-batch": ("deepseek", dict(stages=3, stage_widths=[1, 2, 1])),
    "zero1-stream": ("deepseek", dict(stages=3, stage_widths=[1, 2, 1], stream_grads=True)),
    "zero1-native": (
        "deepseek",
        dict(stages=3, stage_widths=[1, 2, 1], wire="native", stream_grads=True),
    ),
    "zero1-native-w1": ("deepseek", dict(stages=3, wire="native", stream_grads=True)),
    **{
        f"pieces-w{w}": ("deepseek", dict(stages=2, stage_widths=[1, w], **_pieces()))
        for w in (1, 2, 3)
    },
    "pieces-w2-bf16": (
        "deepseek",
        dict(stages=2, stage_widths=[1, 2], compute="bf16", **_pieces()),
    ),
    **{f"bf16-{row}": (row, dict(stages=2, compute="bf16")) for row in ("softmax", "deepseek")},
    # RUNTIME-I20/I21: host-resident moments and per-tensor streamed folds under MoE
    "host-deepseek-bf16": (
        "deepseek",
        dict(stages=3, recompute=True, compute="bf16", optimizer="adamw-host"),
    ),
    "host-softmax-wfill": (
        "softmax",
        dict(stages=4, microbatches=8, schedule_fill=True, optimizer="adamw-host"),
    ),
}
_IDLE_OWNER = [[1.0], [1.0, 0.01], [1.0]]


@pytest.mark.parametrize("arm", sorted(_MOE_ARMS))
def test_moe_identity_arm(tmp_path, port_base, arm):
    """ADR-0008 E2: native MoEMLP as a real multi-process pipeline at depths 1, 2 and 4,
    zero-bubble orders with W fill, recompute, ZeRO-1 widths 1 and 2 on the batch,
    stream_grads and wire-native paths, M2b pieces widths 1-3, fp32 and bf16: the state
    (selection biases included), every CE hex and every aux hex per (step, mb, layer)
    equal the oracle's."""
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: a pipeline cut, an order, recompute, ZeRO-1 ownership, a count lane,
    # pieces or bf16 moves a bit of the MoE state, an aux row is lost or recorded twice,
    # or the bias update reads other counts than the step's int64 fold.
    row, kw = _MOE_ARMS[arm]
    kw = dict(kw)
    compute = kw.pop("compute", "fp32")
    if arm.startswith("zero1-") and arm != "zero1-native-w1":
        kw["peer_weights"] = _IDLE_OWNER
    spec = _moe_spec(tmp_path, "moe", port_base, row, compute, **kw)
    if "peer_weights" in kw:
        fin = JobSpec.from_json(spec.to_json()).finalize()
        assert set(fin.mb_owner[1]) == {0}, fin.mb_owner
    state = _moe_exact(spec)
    if _MOE_ROWS[row].get("moe_bias_rate"):
        zero = __import__("wantrain.validation.state", fromlist=["x"]).tensor_record(torch.zeros(8))
        moved = [k for k, v in state["weights"].items() if k.endswith("sel_bias") and v != zero]
        assert len(moved) == spec.model.n_layer, "a selection bias never moved"


@pytest.mark.parametrize("arm", ["skew-tie", "skew-hot"])
def test_moe_identity_skew(tmp_path, port_base, monkeypatch, arm):
    """ADR-0008 E3: an all-tie router (every key equal at step 0) and a hot expert,
    applied to the oracle and the run alike; the ties and the unrouted (layer, mb)
    are recorded first, then the state and every aux hex equal the oracle's at depth 2
    and on a ZeRO-1 width-2 stage."""
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: a tie is broken by anything but the lowest index on one path, an
    # unrouted expert gets no +0 gradient on one path, or the arm never produced a tie or
    # an unrouted expert (it would prove nothing).
    trig = _moe_site(tmp_path, monkeypatch, f"init-{arm.split('-')[1]},record-skew")
    for widths in ([1, 1], [1, 2]):
        out = tmp_path / f"w{widths[1]}"
        spec = _moe_spec(out, "skew", port_base, "deepseek", stages=2, stage_widths=widths)
        state = _moe_exact(spec)
        assert state["aux"]
    got = _triggers(trig)
    print(f"MOE_SKEW {arm} {json.dumps(got)}")
    assert got.get("unrouted", 0) > 0, got
    if arm == "skew-tie":
        assert got.get("ties", 0) > 0, got


# id -> (patch, router init, row, JobSpec fields, model fields). Pipeline must-fails:
# the alternative runs on the stages only; the oracle ran first, unpatched.
_MOE_MUSTFAIL = {
    # Three equal keys at the top-k boundary: topk's tie order is not the lowest index.
    "topk-stage": ("topk-stage", "init-triple", "softmax", dict(stages=2), {}),
    # An all-tie router leaves experts 2..7 unrouted at step 0.
    "skip-unrouted": ("skip-unrouted", "init-tie", "softmax", dict(stages=2), {}),
    "reverse-combine": ("reverse-combine", "", "softmax", dict(stages=2), {"moe_top_k": 3}),
    "descending-rows": ("descending-rows", "", "softmax", dict(stages=2), {}),
    "aux-alpha-m": ("aux-alpha-m", "", "olmoe", dict(stages=2), {}),
    "bias-float-mean": ("bias-float-mean", "", "deepseek", dict(stages=2), {}),
    "bias-one-peer": ("bias-one-peer", "", "deepseek", dict(stages=3, stage_widths=[1, 2, 1]), {}),
    "group-topk": ("group-topk", "init-groups", "deepseek", dict(stages=2), {}),
    "pieces-off-qt": (
        "pieces-off-qt",
        "",
        "deepseek",
        dict(stages=2, stage_widths=[1, 3], **_pieces()),
        {},
    ),
    "count-dropped": (
        "count-dropped",
        "",
        "deepseek",
        dict(stages=3, stage_widths=[1, 2, 1], wire="native", stream_grads=True),
        {},
    ),
    "count-doubled": (
        "count-doubled",
        "",
        "deepseek",
        dict(stages=3, stage_widths=[1, 2, 1], wire="native", stream_grads=True),
        {},
    ),
}


@pytest.mark.parametrize(
    "name", sorted(_MOE_MUSTFAIL), ids=[f"mustfail-{n}" for n in sorted(_MOE_MUSTFAIL)]
)
def test_moe_identity_mustfail(tmp_path, port_base, monkeypatch, name):
    """ADR-0008 E4 (F2): each alternative an MoE stage could have built, run on the
    stages only, records its trigger first and then the run diverges from the oracle
    (a differing state or aux hex) or is refused (a fatal, a hang cut by the timeout, a
    manifest that cannot be assembled)."""
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: the identity comparison cannot tell one of these alternatives from
    # the exact path (its trigger fired and the state still equals the oracle's), or the
    # control never fired (it proves nothing and must be resized, F2).
    patch, init, row, kw, model_kw = _MOE_MUSTFAIL[name]
    trig = _moe_site(tmp_path, monkeypatch, init)
    spec = _moe_spec(tmp_path, "mf", port_base, row, model_kw=model_kw, **kw)
    ref = reference_manifest_for(spec, tmp_path / "ref")
    for f in os.listdir(trig):
        os.remove(os.path.join(trig, f))
    monkeypatch.setenv("WT_TEST_PATCHES", ",".join(p for p in (init, patch) if p))
    monkeypatch.setenv("WT_TEST_M", str(spec.microbatches))
    res = run_local(spec, timeout=120)
    fired = _triggers(trig).get("trigger", 0)
    outcome = "refused"
    if res["ok"]:
        try:
            diff = diff_manifests(distributed_manifest(spec), ref)
        except (ValueError, RuntimeError, KeyError) as e:
            diff = [f"manifest refused: {e}"]
        outcome = "diverged" if diff else "EQUAL"
    print(f"MOE_MUSTFAIL {name} trigger={fired} rc={res['rc']} outcome={outcome}")
    assert fired > 0, f"{name}: the trigger never fired; resize the control (ADR-0008 F2)"
    assert outcome != "EQUAL", f"{name}: fired {fired} times and the state equals the oracle's"


def _moe_block(compute="fp32", init=None, up_stores_x=False, **over):
    """One converted MoEMLP block at E1's sizes (SplitLinear experts, wgrad capture)."""
    from wantrain.model import arch, wgrad

    cfg = _moe_model("softmax", compute, **over)
    torch.manual_seed(5)
    blk = arch.MoEMLP(cfg, 1)
    wgrad.convert(blk)
    if init is not None:
        with torch.no_grad():
            init(blk.router.weight)
    if up_stores_x:
        # The deliberately wrong expert: up records its own copy of X.
        for e in blk.experts:
            gate, up, down = e.gate, e.up, e.down
            e.forward = lambda x, g=gate, u=up, d=down: d(
                torch.nn.functional.silu(g(x)) * u(x.clone())
            )
    return cfg, blk


def _block_store(blk, mb, T=32):
    """B of one microbatch of T tokens through the block under wgrad.capture:
    -> (the deferred W entries, the block's record)."""
    from wantrain.model import arch, wgrad

    g = torch.Generator().manual_seed(100 + mb)
    x = torch.randn(1, T, blk.router.in_features, generator=g, requires_grad=True)
    with wgrad.capture() as store:
        y = blk(x)
        torch.autograd.grad(y.float().pow(2).sum(), [x])
    (rec,) = arch.take_moe_records([blk])
    return list(store), rec


def _pending_bytes(store) -> int:
    """F3's routed pending-W bytes: every tensor a deferred W holds, storages
    deduplicated by data_ptr."""
    seen = {}
    for _, fn in store:
        for t in fn.__defaults__:
            s = t.untyped_storage()
            seen[s.data_ptr()] = s.nbytes()
    return sum(seen.values())


def _mf_up_stores_x():
    """F3: native routed pending W is T*k*(2h+3f)*4 bytes for any skew; up storing its
    own X makes it T*k*(3h+3f)*4 (416 -> 480 values per routed row, +15.4%)."""
    h, f, k, T = _MOE_BASE["d_model"], _MOE_BASE["moe_d_ff"], _MOE_BASE["moe_top_k"], 32
    want = T * k * (2 * h + 3 * f) * 4
    assert 2 * h + 3 * f == 416
    for init in (None, lambda w: w.zero_(), lambda w: w[0].mul_(40.0)):
        _, blk = _moe_block(init=init)
        store, rec = _block_store(blk, 0, T)
        assert _pending_bytes(store) == want, (_pending_bytes(store), want)
    _, bad = _moe_block(up_stores_x=True)
    store, rec = _block_store(bad, 0, T)
    routed = int(rec.counts.sum())
    got = _pending_bytes(store)
    assert got == T * k * (3 * h + 3 * f) * 4, got
    return routed, got != want


def _mf_merged_w_gemm():
    """W of microbatches m and m+1 as one GEMM over their concatenated rows, against
    the exact per-microbatch W folded in microbatch order."""
    from wantrain.runtime.logical_ckpt import digest

    _, blk = _moe_block()
    (s0, r0), (s1, r1) = _block_store(blk, 0), _block_store(blk, 1)
    both = int(((r0.counts > 0) & (r1.counts > 0)).sum())
    differs = 0
    for (m0, f0), (m1, f1) in zip(s0, s1):
        assert m0 is m1
        x0, g0 = f0.__defaults__
        x1, g1 = f1.__defaults__
        if not (x0.shape[0] and x1.shape[0]):
            continue
        exact = f0()[0] + f1()[0]
        merged = torch.cat([g0, g1]).t().mm(torch.cat([x0, x1]))
        differs += digest(exact) != digest(merged)
    return both, differs > 0


def _mf_fold_neg_zero():
    """A fold started from zeros against the exact left fold from the first
    microbatch's gradient, on real per-microbatch dW with a constructed -0 (an
    element every microbatch gives -0)."""
    from wantrain.runtime import pieces
    from wantrain.runtime.logical_ckpt import digest

    _, blk = _moe_block()
    gs = []
    for mb in range(4):
        store, _ = _block_store(blk, mb)
        gs.append(store[0][1]()[0].clone())
    for g in gs:
        g[0, 0] = -0.0
    exact = pieces.fold(gs)
    zeros = torch.zeros_like(gs[0])
    for g in gs:
        zeros = zeros + g
    trigger = int(((exact == 0) & torch.signbit(exact)).sum())
    return trigger, digest(exact) != digest(zeros)


_MOE_MUSTFAIL_FN = {
    "merged-w-gemm": _mf_merged_w_gemm,
    "fold-neg-zero": _mf_fold_neg_zero,
    "up-stores-x": _mf_up_stores_x,
}


@pytest.mark.parametrize(
    "name", sorted(_MOE_MUSTFAIL_FN), ids=[f"mustfail-{n}" for n in sorted(_MOE_MUSTFAIL_FN)]
)
def test_moe_identity_mustfail_fn(name):
    """ADR-0008 E4's must-fails that live below the state manifest, on real MoEMLP W
    entries: one W GEMM over two microbatches, a fold from zeros with a constructed -0,
    and up storing its own X (which must fire F3's pending-W byte check); each records
    its trigger first and then differs from the exact path's bits or bytes."""
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: the exact path and the alternative give the same bits (the control
    # cannot catch the alternative), F3's check accepts up storing X, or the trigger
    # never fired.
    trigger, differs = _MOE_MUSTFAIL_FN[name]()
    print(f"MOE_MUSTFAIL {name} trigger={trigger} differs={differs}")
    assert trigger > 0, f"{name}: the trigger never fired; resize the control (ADR-0008 F2)"
    assert differs, f"{name}: fired {trigger} times and the bits equal the exact path's"


def _moe_restore_flags():
    saved = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
        torch.get_rng_state(),
    )

    def restore():
        torch.set_num_threads(saved[0])
        torch.use_deterministic_algorithms(saved[1], warn_only=saved[2])
        torch.set_float32_matmul_precision(saved[3])
        torch.set_rng_state(saved[4])

    return restore


def test_moe_oracle_hand_fold_one_layer(tmp_path, monkeypatch):
    """The MoE oracle on a 1-layer recipe (single process): every (step, mb, layer) aux
    hex is recorded once, and the selection bias moves after the step's AdamW from the
    int64 counts of every microbatch, equal to a fold written by hand here."""
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: the oracle records an aux row twice or misses one, folds the counts
    # in float or from one microbatch, moves the bias by another rule or before the
    # counts are complete, or the hand fold and the oracle disagree on any byte.
    from wantrain.model.arch import take_moe_records
    from wantrain.model.data import make_batch
    from wantrain.model.gpt import build_layers, loss_fn
    from wantrain.runtime.determinism import setup_determinism
    from wantrain.validation.reference import run_baseline
    from wantrain.validation.state import aux_hex, tensor_record

    for name in ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)
    spec = JobSpec(
        model=_moe_model("deepseek", n_layer=1, moe_bias_rate=0.01),
        steps=3,
        microbatches=4,
        mb_size=2,
        out_dir=str(tmp_path),
    )
    m = spec.model
    restore = _moe_restore_flags()
    try:
        run_baseline(spec, "orc")
        setup_determinism(spec.init_seed)
        layers = build_layers(m, spec.init_seed)
        params = [p for p in layers.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=spec.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        blk = next(x for x in layers.modules() if hasattr(x, "sel_bias"))
        rows, wrong_bias = [], blk.sel_bias.detach().clone()
        T_g, E, k = spec.microbatches * spec.mb_size * m.seq_len, m.moe_experts, m.moe_top_k
        for step in range(spec.steps):
            acc, counts, last = None, torch.zeros(E, dtype=torch.int64), None
            for mb in range(spec.microbatches):
                x, tgt = make_batch(spec.data_seed, step, mb, spec.mb_size, m.seq_len, m.vocab_size)
                h = x
                for layer in layers:
                    h = layer(h)
                gs = torch.autograd.grad(loss_fn(h, tgt), params)
                (rec,) = take_moe_records(layers)
                rows.append([step, mb, rec.layer, aux_hex(rec.aux)])
                counts = counts + rec.counts
                last = rec.counts
                acc = (
                    [g.clone() for g in gs] if acc is None else [a.add_(g) for a, g in zip(acc, gs)]
                )
            for p, g in zip(params, acc):
                p.grad = g
            opt.step()
            opt.zero_grad(set_to_none=True)
            sgn = torch.sign(T_g * k - E * counts)
            with torch.no_grad():
                blk.sel_bias.copy_(
                    torch.where(
                        sgn == 0, blk.sel_bias, blk.sel_bias + sgn.float() * m.moe_bias_rate
                    )
                )
            # The deliberately wrong rule: the last microbatch's counts scaled to the step.
            ws = torch.sign(T_g * k - E * last * spec.microbatches)
            wrong_bias = torch.where(ws == 0, wrong_bias, wrong_bias + ws.float() * m.moe_bias_rate)
    finally:
        restore()
    state = _oracle_state(tmp_path, "orc")
    assert state["aux"] == sorted(rows) and len({tuple(r[:3]) for r in rows}) == len(rows)
    (bias_key,) = [n for n in state["weights"] if n.endswith("sel_bias")]
    assert state["weights"][bias_key] == tensor_record(blk.sel_bias)
    assert blk.sel_bias.abs().sum() > 0, "the bias never moved"
    for n, t in layers.state_dict().items():
        i, rest = n.split(".", 1)
        assert state["weights"][f"layers.{i}.{rest}"] == tensor_record(t), n
    assert tensor_record(wrong_bias) != tensor_record(blk.sel_bias), "the control cannot tell"


def _send_events(spec) -> list:
    """Every exchange frame a slot sent: (rank, peer, step, mb, unit, kind, tensor bytes)."""
    out = []
    for r, w in enumerate(spec.stage_widths):
        for p in range(w):
            path = os.path.join(spec.out_dir, f"{spec.tag}_rank{r}p{p}_telemetry.jsonl")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    ev = json.loads(line)
                    if ev.get("ev") == "send" and "key" in ev:
                        unit, kind, step, mb = ev["key"][:4]
                        out.append((r, p, step, mb, unit, kind, int(ev["tb"])))
    return out


def _frame_tables(spec, moe: bool) -> tuple:
    """F4 per sender peer and step of the wide stage 1 under wire native, ZeRO-1, from
    first principles: per microbatch it runs, one GRAD frame of numel*4 bytes per
    trainable parameter the other peer owns (stride i % w), and one COUNT frame of E*8
    bytes per selection bias it owns ((n + j) % w); the dense model has no COUNT lane.
    -> (expected, measured) {(peer, step, kind): (frames, bytes)}."""
    from torch import nn

    from wantrain.model.gpt import build_layers
    from wantrain.runtime import exchange as xc

    fin = JobSpec.from_json(spec.to_json()).finalize()
    a, b = fin.partition[1]
    chunk = nn.ModuleList(list(build_layers(fin.model, fin.init_seed))[a:b])
    params = [p for p in chunk.parameters() if p.requires_grad]
    biases = [m.sel_bias for m in chunk.modules() if hasattr(m, "sel_bias")] if moe else []
    assert bool(biases) == moe
    w, n = fin.stage_widths[1], len(params)
    want: dict = {}
    for peer in range(w):
        mbs = sum(1 for o in fin.mb_owner[1] if o == peer)
        for step in range(fin.steps):
            g = [p.numel() * 4 for i, p in enumerate(params) if i % w != peer]
            c = [t.numel() * 8 for j, t in enumerate(biases) if (n + j) % w != peer]
            want[(peer, step, xc.GRAD)] = (mbs * len(g), mbs * sum(g))
            if c:
                want[(peer, step, xc.COUNT)] = (mbs * len(c), mbs * sum(c))
    got: dict = {}
    for r, p, step, _, _, kind, nb in _send_events(fin):
        if r == 1 and kind in (xc.GRAD, xc.COUNT):
            fr_, by = got.get((p, step, kind), (0, 0))
            got[(p, step, kind)] = (fr_ + 1, by + nb)
    return want, got


def _routed_unit_frames(spec) -> tuple:
    """F4 under shard pieces: the routed-expert units of each MoE layer (one per
    projection over the E experts, in expert order: 3) and their GRAD frames per
    (sender, step, microbatch). -> (units per layer, {frames per (peer, step, mb)})."""
    from torch import nn

    from wantrain.model.gpt import build_layers
    from wantrain.runtime import exchange as xc
    from wantrain.runtime import pieces

    fin = JobSpec.from_json(spec.to_json()).finalize()
    a, b = fin.partition[1]
    chunk = nn.ModuleList(list(build_layers(fin.model, fin.init_seed))[a:b])
    units = pieces.exchange_units(chunk)
    routed = [u for u, names in enumerate(units) if all(".experts." in n for n, _ in names)]
    layers = {n.split(".experts.")[0] for u in routed for n, _ in units[u]}
    frames: dict = {}
    for r, p, step, mb, unit, kind, _ in _send_events(fin):
        if r == 1 and kind == xc.GRAD and unit in routed:
            frames[(p, step, mb)] = frames.get((p, step, mb), 0) + 1
    return len(routed) / max(1, len(layers)), len(layers), frames


@pytest.mark.parametrize("arm", ["e6-frames", "dense-tables"])
def test_moe_identity_frames(tmp_path, port_base, arm):
    """ADR-0008 E6 (F4, with M2b): on native MoE every frame class per peer and step
    equals its formula (gradient frames and bytes of the other peer's stride, a count
    lane of E*8 bytes per microbatch per owned bias; under pieces 3 routed-expert unit
    frames per destination and microbatch per MoE layer), and a dense model's tables
    are the formula's with no count lane (F4's last clause); every arm stays exact."""
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: a count lane is dropped, doubled or sent for a dense model, the
    # expert projections ship as 3E units instead of 3, a gradient frame goes to the
    # wrong stride owner, or the frame bytes move with MoE on a dense model.
    moe = arm == "e6-frames"
    kw = dict(stages=2, stage_widths=[1, 2], wire="native", stream_grads=True)
    if moe:
        spec = _moe_spec(tmp_path / "z", "e6", port_base, "deepseek", **kw)
        _moe_exact(spec)
    else:
        spec = _m2b_spec(tmp_path / "z", "e6", port_base, compute="fp32", steps=3, **kw)
        _run_ok(spec)
        _assert_exact(spec, tmp_path / "ref")
    want, got = _frame_tables(spec, moe)
    print(f"E6_FRAMES {arm} want={sorted(want.items())} got={sorted(got.items())}")
    assert got == want
    if moe:
        pspec = _moe_spec(
            tmp_path / "p",
            "e6p",
            port_base + 32,
            "deepseek",
            stages=2,
            stage_widths=[1, 2],
            **_pieces(),
        )
        _moe_exact(pspec)
        per_layer, n_layers, frames = _routed_unit_frames(pspec)
        assert per_layer == 3 and n_layers >= 1, (per_layer, n_layers)
        fin = JobSpec.from_json(pspec.to_json()).finalize()
        expect = {
            (p, s, mb): 3 * n_layers
            for s in range(fin.steps)
            for mb, p in enumerate(fin.mb_owner[1])
        }
        assert frames == expect, (frames, expect)


def _telem(spec) -> dict:
    """(rank, peer) -> the WT_TELEM summaries of the slot, in step order."""
    return {
        slot: [json.loads(ln.split(" ", 1)[1]) for ln in lines]
        for slot, lines in _slot_lines(spec, "WT_TELEM ").items()
    }


def _cpu_flops() -> float:
    """MEASURED here (one thread, fp32 GEMM of the recipe's shapes): FLOP/s."""
    restore = _moe_restore_flags()
    try:
        torch.set_num_threads(1)
        a, b = torch.randn(32, 64), torch.randn(64, 96)
        for _ in range(50):
            a.mm(b)
        n, t0 = 2000, time.perf_counter()
        for _ in range(n):
            a.mm(b)
        return 2 * 32 * 64 * 96 * n / (time.perf_counter() - t0)
    finally:
        restore()


@pytest.mark.parametrize("arm", ["f7-softmax"])
def test_moe_identity_f7(tmp_path, port_base, arm):
    """ADR-0008 F7: a 2-stage native MoE run's per-microbatch compute (WT_TELEM F + B + W
    medians) per stage over its analytic c_s = 8 P^F_s T_mb / F (scheduler-m4's
    gemm_params_active over moe_layer_costs; F measured on this CPU), written to the
    JSON report labelled SIMULATED; the >= 0.9 bound is asserted only under
    FACTORY_E7_BOUNDS=1 (GATES-M4, a quiet machine). The run stays exact."""
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: the analytic cost overstates an MoE stage's compute (ratio below 0.9
    # on a quiet machine), a stage emits no telemetry, or the run moves a bit.
    from wantrain.scheduler.planner import gemm_params_active

    spec = _moe_spec(tmp_path, "f7", port_base, "softmax", stages=2)
    _moe_exact(spec)
    fin = JobSpec.from_json(spec.to_json()).finalize()
    rate = _cpu_flops()
    tokens = fin.mb_size * fin.model.seq_len
    rows = []
    for (r, p), summaries in sorted(_telem(fin).items()):
        steady = summaries[1:] or summaries
        per_mb = statistics.median(
            sum(s["ops"][op]["med_ms"] for op in ("F", "B", "W")) / 1e3 for s in steady
        )
        a, b = fin.partition[r]
        c_s = 8 * gemm_params_active(fin.model, a, b) * tokens / rate
        rows.append(dict(stage=r, peer=p, compute_s=per_mb, c_s=c_s, ratio=per_mb / c_s))
    report = {"label": "SIMULATED", "arm": arm, "flops_measured": rate, "stages": rows}
    with open(tmp_path / "moe_f7_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("MOE_F7 " + json.dumps(report))
    assert len(rows) == fin.stages
    if os.environ.get("FACTORY_E7_BOUNDS") == "1":
        low = [x for x in rows if x["ratio"] < 0.9]
        assert not low, f"F7: compute below 0.9x c_s: {low}"


# HF-MoE adapter arms (ADR-0008 E2-HF): Mixtral and Qwen3-MoE built by transformers 5.19.0,
# their sparse blocks swapped to the exact layer by model-families' rows; the run loads a
# hub-format checkpoint (per-expert names) exported from a seeded build, so the loader's
# hub-to-fused map round-trips under the oracle and every stage alike.
_HF_MOE = {"hf-mixtral": "mixtral", "hf-qwen3-moe": "qwen3_moe"}


def _hub_keys(units, model_type: str) -> list:
    """The hub (per-expert) names of a built model's tensors."""
    hfm = _hfm()
    keys = []
    for name, p in units.root.named_parameters():
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            prefix = name.rsplit(".experts.", 1)[0]
            parts = ("down",) if name.endswith("down_proj") else ("gate", "up")
            keys += [
                hfm._hub_name(model_type, prefix, j, part)
                for j in range(p.shape[0])
                for part in parts
            ]
        else:
            ren = hfm._HUB_RENAME.get(model_type)
            keys.append(name.replace(ren[1], ren[0]) if ren and ".mlp.gate." in name else name)
    return keys


@pytest.mark.parametrize("arm", sorted(_HF_MOE))
def test_moe_identity_hf(tmp_path, port_base, arm):
    """A 2-stage HF-MoE pipeline (the swapped blocks, fp32) from a hub-format checkpoint
    equals the oracle loading the same checkpoint: state, selection biases, every CE and
    aux hex; the checkpoint's per-expert names map onto the fused banks and back."""
    # covers: RUNTIME-I18, RUNTIME-I19
    # WOULD-FAIL-IF: the swap rows, the D8 injection or the hub-to-fused loader differ
    # between the oracle and the stages, or an expert's bytes land in another bank slot.
    from wantrain.model import hf
    from wantrain.model.gpt import GPTConfig

    hfm = _hfm()
    mt = _HF_MOE[arm]
    hc = hfm.tiny_config(mt, use_cache=False, num_hidden_layers=2)
    units = hfm.build(
        GPTConfig(arch="hf", hf_config=hc, hf_buffers=hfm.buffer_record(hc), seq_len=16), 99
    )
    keys = _hub_keys(units, mt)
    assert any(".experts.0." in k for k in keys), keys[:8]
    ckpt = tmp_path / "hub"
    hfm.export(units, str(ckpt), keys)
    with open(ckpt / hf.INDEX) as f:
        assert sorted(json.load(f)["weight_map"]) == sorted(keys)
    spec = JobSpec(
        model=GPTConfig(
            arch="hf",
            hf_config=hc,
            seq_len=16,
            hf_model=str(ckpt),
            hf_model_sha256=hf.checkpoint_digest(str(ckpt)),
        ),
        stages=2,
        steps=3,
        microbatches=4,
        mb_size=2,
        out_dir=str(tmp_path),
        tag="hfmoe",
        port_base=port_base,
        dial_timeout_s=60,
    )
    spec.links = links_for(spec)
    state = _hf_exact(spec)
    if state is not None:
        assert state.get("aux"), "the swapped blocks recorded no aux"


# ADR-0009 E4 (round4.md section 1): the tied table split between stage 0 and the Head,
# arms A1-A4 over two vocabulary regimes; each against an untied baseline with the same
# V, S, M, B, L and seed. -> (V, S, M, one-way L in ms).
_E4_TIED = {
    "a1": (8192, 4, 8, 20),
    "a2": (8192, 3, 16, 0),
    "a3": (64, 4, 8, 0),
    "a4": (64, 4, 8, 40),
}
_E4_TIED_MBPS = 50.0  # a/B of a few ms at T_mb 128, h 64 (round4.md: a/B near t_s)


def _host_mem_bw() -> float:
    """MEASURED here: bytes/s of one large host copy (read + write)."""
    src = torch.empty(16 << 20, dtype=torch.uint8)
    dst = torch.empty_like(src)
    dst.copy_(src)
    t0 = time.perf_counter()
    for _ in range(8):
        dst.copy_(src)
    return 2 * 8 * src.numel() / (time.perf_counter() - t0)


def _steady_step_ms(spec) -> float:
    last = spec.stages - 1
    summaries = _telem(spec)[(last, 0)]
    return statistics.median(s["step_ms"] for s in summaries[1:])


@pytest.mark.parametrize("arm", sorted(_E4_TIED))
def test_e4_tied_arm(tmp_path, port_base, arm):
    """ADR-0009 E4 / F4: the split tied table's sha256 (and the whole state) equals the
    oracle's in every arm; the arm's exposed tied time (tied step minus the untied
    baseline's) and Delta T_tied from its measured t_s, t_F, t_B, |T_i| and |R_k| are
    written to the report labelled SIMULATED; the max(20%, 2 ms) bound is asserted
    only under FACTORY_E7_BOUNDS=1 (GATES-M4, a quiet machine)."""
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the relayed table differs from the oracle's by any bit in an arm, a
    # TIED path hangs or fails, or (quiet machine, bounds on) the exposed time is off
    # Delta T_tied by more than max(20%, 2 ms).
    from wantrain.model.data import make_batch
    from wantrain.scheduler.autoconfig import delta_t_tied

    V, S, M, L = _E4_TIED[arm]
    link = dict(latency_ms=L, bw_mbps=_E4_TIED_MBPS)
    common = dict(stages=S, steps=4, microbatches=M, link=link)
    over = dict(vocab_size=V)
    tied = _hf_spec(tmp_path / "tied", "e4t", port_base, "gemma2-tied", hf_over=over, **common)
    tied.model.seq_len = 64
    state = _hf_exact(tied, timeout=600)
    if state is None:
        return
    assert tied.tied_relay()
    _tied_clean(tied)
    untied = _hf_spec(
        tmp_path / "untied",
        "e4u",
        port_base + 32,
        "gemma2-tied",
        hf_over={**over, "tie_word_embeddings": False},
        **common,
    )
    untied.model.seq_len = 64
    _hf_exact(untied, timeout=600)
    exposed = (_steady_step_ms(tied) - _steady_step_ms(untied)) / 1e3
    s0 = _telem(tied)[(0, 0)][1:]
    t_F = statistics.median(s["ops"]["F"]["med_ms"] for s in s0) / 1e3
    t_B = statistics.median(s["ops"]["B"]["med_ms"] for s in s0) / 1e3
    t_W = statistics.median(s["ops"]["W"]["med_ms"] for s in s0) / 1e3
    rows_T, rows_R, seen = [], [], set()
    for mb in range(M):
        x, _ = make_batch(tied.data_seed, 1, mb, tied.mb_size, 64, V)
        ids = set(x.reshape(-1).tolist())
        rows_T.append(len(ids))
        rows_R.append(len(ids - seen))
        seen |= ids
    h = tied.model.d_model
    terms = delta_t_tied(
        t_F + t_B + t_W,
        t_F,
        t_B,
        rows_T,
        rows_R,
        S - 1,
        L / 1e3,
        _E4_TIED_MBPS * 1e6 / 8,
        d_model=h,
        vocab_size=V,
        tokens_mb=tied.mb_size * 64,
        mem_bw_Bps=_host_mem_bw(),
    )
    report = {
        "label": "SIMULATED",
        "arm": arm,
        "V": V,
        "S": S,
        "M": M,
        "L_ms": L,
        "exposed_s": exposed,
        "delta_t_tied": terms,
        "rows_T": rows_T,
        "rows_R": rows_R,
        "t_F": t_F,
        "t_B": t_B,
        "t_W": t_W,
    }
    with open(tmp_path / f"e4_tied_{arm}_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("E4_TIED " + json.dumps(report))
    if os.environ.get("FACTORY_E7_BOUNDS") == "1":
        bound = max(0.2 * terms["total"], 0.002)
        assert abs(exposed - terms["total"]) <= bound, report
