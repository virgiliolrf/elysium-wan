"""E4's timing-only test seams (m2b e4-exposure): the exchange chunk size
(WANTRAIN_TEST_CHUNK_BYTES), a one-way delay on intra-stage hops
(WANTRAIN_TEST_INTRA_LATENCY_MS) and per-slot op padding
(WANTRAIN_TEST_SLOT_PAD). Each changes when bytes move or ops run, never which
bytes or the order they fold in: unset, the parsers return today's values;
set, a run's state is the reference's (and the unset run's) bit for bit, and
the timing effect shows in the slots' telemetry."""

from __future__ import annotations

import json
import os

import pytest
from helpers.exact import reference_manifest_for

from wantrain.launch.local import links_for, run_local
from wantrain.model.gpt import make_config
from wantrain.runtime import exchange as xc
from wantrain.runtime.spec import JobSpec
from wantrain.runtime.stage import intra_latency_seam, slot_pad_seam
from wantrain.validation.state import diff_manifests, distributed_manifest

SEAMS = ("WANTRAIN_TEST_CHUNK_BYTES", "WANTRAIN_TEST_INTRA_LATENCY_MS", "WANTRAIN_TEST_SLOT_PAD")
WIDTH = 3  # stage 0: Embed and blocks in pieces over 3 peers; stage 1 at width 1
M = 6  # two microbatches per stage-0 peer


def test_seam_parsers_unset_is_today_and_malformed_is_refused():
    """Unset or empty, each seam is today's value (16 MiB chunks, no intra
    delay, no padding); a set value is parsed exactly; a malformed one is
    refused naming the seam, never read as a default."""
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: an unset seam changed today's chunk size, intra delay or op timing,
    # or a malformed value (negative, NaN, a chunk above 16 MiB or below one element,
    # padding without room for B and W) were accepted.
    for unset in (None, ""):
        assert xc.chunk_bytes_seam(unset) == xc.CHUNK_BYTES == 16 << 20
        assert intra_latency_seam(unset) == 0.0
        assert slot_pad_seam(unset, 0) is None
    assert xc.chunk_bytes_seam("65536") == 65536
    assert intra_latency_seam("7.842") == 7.842
    assert slot_pad_seam("0:318.2:157:16", 0) == pytest.approx((0.3182, 0.157, 0.016))
    assert slot_pad_seam("0:318.2:157:16", 1) is None
    for bad in ("7", "x", str((16 << 20) + 1), "1.5"):
        with pytest.raises(ValueError, match="WANTRAIN_TEST_CHUNK_BYTES"):
            xc.chunk_bytes_seam(bad)
    for bad in ("-1", "nan", "inf", "x"):
        with pytest.raises(ValueError, match="WANTRAIN_TEST_INTRA_LATENCY_MS"):
            intra_latency_seam(bad)
    for bad in ("0:300:100", "a:300:100:50", "-1:300:1:1", "0:300:200:100", "0:300:-1:1"):
        with pytest.raises(ValueError, match="WANTRAIN_TEST_SLOT_PAD"):
            slot_pad_seam(bad, 0)


def _spec(out_dir, port_base):
    spec = JobSpec(
        model=make_config("toy", compute_dtype="bf16"),
        steps=3,
        microbatches=M,
        stages=2,
        stage_widths=[WIDTH, 1],
        wire="native",
        shard="pieces",
        stream_grads=True,
        wire_codec="planes",
        out_dir=str(out_dir),
        tag="seam",
        port_base=port_base,
        dial_timeout_s=60,
    )
    spec.links = links_for(spec)
    spec.finalize()
    # Stage 0 runs F, B, W per owned microbatch: the order WANTRAIN_TEST_SLOT_PAD lays out.
    spec.schedule_order[0] = [
        [[op, mb] for mb in spec.owned(0, p) for op in ("F", "B", "W")] for p in range(WIDTH)
    ]
    return spec.finalize()


def _run(tmp_path, port_base, monkeypatch, env):
    for k in SEAMS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = _spec(tmp_path, port_base)
    res = run_local(spec, timeout=240)
    assert res["ok"], res
    return spec, distributed_manifest(spec)


def _events(spec, peer):
    path = os.path.join(spec.out_dir, f"{spec.tag}_rank0p{peer}_telemetry.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """The reference state and an unset run's telemetry (every seam unset)."""
    from helpers.ports import reserve_port_block

    out = tmp_path_factory.mktemp("seam_baseline")
    mp = pytest.MonkeyPatch()
    block = reserve_port_block()
    try:
        spec, state = _run(out / "run", block.base, mp, {})
    finally:
        block.release()
        mp.undo()
    ref = reference_manifest_for(spec, out / "ref")
    assert diff_manifests(state, ref) == [] and state == ref
    return {"spec": spec, "state": state, "ref": ref}


def _same_state(state, baseline):
    assert diff_manifests(state, baseline["ref"]) == [], diff_manifests(state, baseline["ref"])
    assert state == baseline["ref"] == baseline["state"]


def _exchange_chunks(spec):
    return [
        e["tb"]
        for p in range(WIDTH)
        for e in _events(spec, p)
        if e["ev"] == "send" and "tb" in e and e["hop"].startswith("intra")
    ]


def test_chunk_seam_cuts_smaller_chunks_and_keeps_the_state(
    tmp_path, port_base, monkeypatch, baseline
):
    """WANTRAIN_TEST_CHUNK_BYTES=4096: every exchange chunk carries at most
    4096 tensor bytes (unset: pieces go whole, above 4096), so more chunks
    cross the intra hops, and the state is the reference's bit for bit."""
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: the seam did not reach stage._send_exchange (chunks stay whole),
    # cut a piece so that it reassembled or folded differently (one bit), or chunked
    # beyond the bound.
    spec, state = _run(tmp_path, port_base, monkeypatch, {"WANTRAIN_TEST_CHUNK_BYTES": "4096"})
    _same_state(state, baseline)
    base, cut = _exchange_chunks(baseline["spec"]), _exchange_chunks(spec)
    print(
        f"SEAM chunk: unset {len(base)} chunks max {max(base)} B, 4096: {len(cut)} max {max(cut)}"
    )
    assert max(base) > 4096
    assert max(cut) <= 4096
    assert len(cut) > 2 * len(base)


def _gather_wait(spec, steps):
    """Per step, the longest gather wait over stage 0's peers (seconds)."""
    out = [0.0] * steps
    for p in range(WIDTH):
        per = [0.0] * steps
        for e in _events(spec, p):
            if e["ev"] == "stall" and e.get("wait") == "gather":
                per[e["step"]] += e["t1"] - e["t0"]
        out = [max(a, b) for a, b in zip(out, per)]
    return out


def test_intra_latency_seam_delays_the_exchange_and_keeps_the_state(
    tmp_path, port_base, monkeypatch, baseline
):
    """WANTRAIN_TEST_INTRA_LATENCY_MS=200: every intra frame arrives 200 ms
    late, so a unit's gathers come back at least a gradient hop plus a gather
    hop (400 ms) after the last remote W: step t >= 1's forwards wait well
    over 200 ms for them (unset: far less); the state is the reference's."""
    # covers: RUNTIME-I13, TRANSPORT-I10
    # WOULD-FAIL-IF: the delay missed the intra connections (on both sides; a scratch
    # mutant delaying only the dialled side still passes), went to the pipeline hops
    # instead, or reordered a fold (one bit of the state).
    spec, state = _run(tmp_path, port_base, monkeypatch, {"WANTRAIN_TEST_INTRA_LATENCY_MS": "200"})
    _same_state(state, baseline)
    base, slow = _gather_wait(baseline["spec"], 3), _gather_wait(spec, 3)
    print(f"SEAM intra latency: gather wait per step unset {base}, 200 ms {slow}")
    for t in (1, 2):
        assert slow[t] >= 0.2, (t, slow)
        assert slow[t] > base[t] + 0.15, (t, base, slow)


def test_slot_pad_seam_holds_each_op_to_its_slot_and_keeps_the_state(
    tmp_path, port_base, monkeypatch, baseline
):
    """WANTRAIN_TEST_SLOT_PAD=0:1000:300:150 on stage 0: on every peer and
    step, microbatch i's F starts no earlier than o + 1000i ms, its B than
    o + 1000(i + 1) - 450 ms and its W than o + 1000(i + 1) - 150 ms (o: the
    step's first F plus its gather waits), so a step spans at least
    M/w x 1000 - 150 ms (unset: less); the state is the reference's."""
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: the padding were computed off the slot (an op runs early), held the
    # wrong op, never applied, or changed an op's inputs or the fold order (one bit).
    tau, b, w = 1.0, 0.3, 0.15
    spec, state = _run(
        tmp_path, port_base, monkeypatch, {"WANTRAIN_TEST_SLOT_PAD": "0:1000:300:150"}
    )
    _same_state(state, baseline)
    names = {"fwd": "F", "bwd_b": "B", "bwd_w": "W"}
    for p in range(WIDTH):
        slot = {mb: i for i, mb in enumerate(spec.owned(0, p))}
        for which, run in (("padded", spec), ("unset", baseline["spec"])):
            ev = _events(run, p)
            for t in range(spec.steps):
                ops = {
                    (names[e["ev"]], e["mb"]): e for e in ev if e["ev"] in names and e["step"] == t
                }
                f0 = ops[("F", spec.owned(0, p)[0])]
                waits = sum(
                    e["t1"] - e["t0"]
                    for e in ev
                    if e["ev"] == "stall" and e.get("wait") == "gather" and e["step"] == t
                )
                o = f0["t0"] + waits
                span = max(e["t1"] for e in ops.values()) - f0["t0"]
                if which == "unset":
                    # No bound on the unpadded run: on a slow machine (a CI runner) compute alone fills the
                    # slots, and the unpadded step came out 2.706 s against the padded 2.687 s. The invariant is
                    # the padded side: no op starts before its slot.
                    continue
                assert span >= len(slot) * tau - w, (p, t, span)
                for (op, mb), e in ops.items():
                    i = slot[mb]
                    if (op, i) == ("F", 0):
                        continue  # the slot origin itself
                    due = {"F": i * tau, "B": (i + 1) * tau - w - b, "W": (i + 1) * tau - w}[op]
                    assert e["t0"] >= o + due, (p, t, op, mb, e["t0"] - o, due)
