"""JobSpec.finalize() is the gate between a solved schedule and the runtime:
an order that misses, repeats or invents work for a slot must be refused
before any process starts, because a slot running it would wait forever for
an input nobody sends or fold a gradient twice."""

# WOULD-FAIL-IF: finalize() accepts a schedule whose F, B or W does not cover exactly the slot's owned microbatches, or an unknown op

import contextlib
import copy
import json
import math
import os
import queue as queue_mod
import random
import re
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from wantrain.model.gpt import make_config
from wantrain.runtime import telemetry
from wantrain.runtime.spec import JobSpec
from wantrain.runtime.stage import StageRuntime


def _spec(**kw):
    return JobSpec(model=make_config("toy"), **kw).finalize()


def _refinalize(spec):
    return JobSpec.from_json(spec.to_json()).finalize()


def test_finalize_refuses_an_order_missing_a_weight_pass():
    # covers: RUNTIME-I5
    spec = _spec(stages=2, microbatches=4)
    spec.schedule_order[1][0] = [o for o in spec.schedule_order[1][0] if o != ["W", 3]]
    with pytest.raises(AssertionError, match="must cover exactly"):
        _refinalize(spec)


def test_finalize_refuses_a_duplicated_forward():
    # covers: RUNTIME-I5
    spec = _spec(stages=2, microbatches=4)
    spec.schedule_order[0][0] = [["F", 0]] + spec.schedule_order[0][0]
    with pytest.raises(AssertionError, match="must cover exactly"):
        _refinalize(spec)


def test_finalize_refuses_work_on_a_microbatch_the_peer_does_not_own():
    # covers: RUNTIME-I5
    spec = _spec(stages=2, microbatches=4, stage_widths=[2, 1])
    foreign = next(mb for mb, p in enumerate(spec.mb_owner[0]) if p == 1)
    spec.schedule_order[0][0] = spec.schedule_order[0][0] + [
        ["F", foreign],
        ["B", foreign],
        ["W", foreign],
    ]
    with pytest.raises(AssertionError, match="must cover exactly"):
        _refinalize(spec)


def test_finalize_refuses_an_unknown_operation():
    # covers: RUNTIME-I5
    spec = _spec(stages=2, microbatches=2)
    spec.schedule_order[0][0] = spec.schedule_order[0][0] + [["X", 0]]
    with pytest.raises(AssertionError, match="unknown op"):
        _refinalize(spec)


def test_a_finalized_default_order_is_accepted_unchanged():
    # covers: RUNTIME-I5
    spec = _spec(stages=3, microbatches=6, stage_widths=[1, 2, 1])
    assert _refinalize(spec).schedule_order == spec.schedule_order


# One stamp unit, 1/1024 s: every duration below is then exact in binary, so
# the summary's milliseconds compare with ==, not within a tolerance.
_TICK = 1 / 1024


def _tel_event(ev, step, start, length, **kw):
    """A stage event of `length` ticks starting at tick `start`."""
    t0 = 100.0 + start * _TICK
    return dict(ev=ev, step=step, mb=0, vs=0, t0=t0, t1=t0 + length * _TICK, **kw)


def test_step_summary_is_a_timing_only_median():
    # covers: RUNTIME-I9
    # WOULD-FAIL-IF: summarize_step reports a mean instead of a median, sums overlapping send
    # intervals instead of taking their union, maps a stage event name wrongly, counts F/B/W
    # or stall events of another step, drops sends of an earlier step, or draws from an RNG.
    ms = 1000 * _TICK  # one tick in ms, exact
    events = [
        _tel_event("p2p_auth", 0, 0, 1),
        _tel_event("fwd", 3, 0, 1),
        _tel_event("fwd", 3, 1, 2),
        _tel_event("fwd", 3, 3, 9),
        _tel_event("fwd", 2, 0, 100),  # an earlier step: not this summary's op
        _tel_event("bwd_b", 3, 0, 3),
        _tel_event("bwd_b", 3, 3, 20),
        _tel_event("bwd_b", 3, 23, 4),
        _tel_event("bwd_w", 3, 0, 5),
        _tel_event("bwd_w", 3, 5, 30),
        _tel_event("bwd_w", 3, 35, 6),
        _tel_event("bwd_w", 3, 41, 7),
        _tel_event("step_opt", 3, 50, 8),
        _tel_event("stall", 3, 0, 3),
        _tel_event("stall", 3, 9, 5),
        _tel_event("stall", 2, 0, 64),
        # Overlapping streams of one hop: union 20 ticks, sum 30. The step-2
        # send completed after step 2's summary and still counts here.
        _tel_event("send", 2, 0, 10, hop="next0", nbytes=100),
        _tel_event("send", 3, 5, 10, hop="next0", nbytes=200),
        _tel_event("send", 3, 20, 5, hop="next0", nbytes=300),
        _tel_event("recv", 2, 0, 4, hop="prev0", nbytes=50, kind="fwd"),
        _tel_event("recv", 3, 10, 2, hop="prev0", nbytes=50, kind="fwd"),
        _tel_event("send", 3, 0, 4, hop="intra1", nbytes=7),
        _tel_event("encode", 3, 0, 50, nbytes=999, frags=1),
    ]
    rng, py = torch.get_rng_state(), random.getstate()
    s = telemetry.summarize_step(events, 3, 1, 0, 12.5, 40.0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == py
    assert set(s) == {
        "v",
        "step",
        "rank",
        "peer",
        "step_ms",
        "emit_us",
        "ops",
        "stall_ms",
        "hops",
        "recv",
    }
    assert (s["v"], s["step"], s["rank"], s["peer"]) == (1, 3, 1, 0)
    assert (s["step_ms"], s["emit_us"]) == (12.5, 40.0)
    # Medians 2, 4, 6.5 ticks; the means would be 4, 9, 12.
    assert s["ops"] == {
        "F": {"n": 3, "med_ms": 2 * ms},
        "B": {"n": 3, "med_ms": 4 * ms},
        "W": {"n": 4, "med_ms": 6.5 * ms},
        "OPT": {"n": 1, "med_ms": 8 * ms},
    }
    assert s["stall_ms"] == 8 * ms
    assert s["hops"] == {
        "intra1": {"frames": 1, "bytes": 7, "busy_ms": 4 * ms},
        "next0": {"frames": 3, "bytes": 600, "busy_ms": 20 * ms},
    }
    # A boundary's spread (first fragment to whole), every step's, like sends: median 3 ticks, max 4.
    assert s["recv"] == {"prev0": {"n": 2, "med_ms": 3 * ms, "max_ms": 4 * ms}}
    assert telemetry.summarize_step([], 0, 0, 0, 1.0, 0.0)["ops"]["F"] == {"n": 0, "med_ms": 0.0}

    assert telemetry.parse_sim_slow(None) is None
    assert telemetry.parse_sim_slow("") is None
    assert telemetry.parse_sim_slow("1:1:4") == (1, 1, 4.0, 0)
    assert telemetry.parse_sim_slow("2:0:3.5:7") == (2, 0, 3.5, 7)
    malformed = ["1:1", "1:1:4:2:9", "a:1:4", "1:1:x", "1:1:0.5", "1:1:nan"]
    malformed += ["1:1:inf", "-1:0:2", "1:1:4:-1", "1:1:4:1.5"]
    for bad in malformed:
        with pytest.raises(ValueError):
            telemetry.parse_sim_slow(bad)


def _stage_runtime_without_sockets():
    """A real StageRuntime that never connects (run() is not called). Its
    constructor sets process-global torch and random state for the stage
    process; that state is put back so later tests in this session see none
    of it."""
    rng, py = torch.get_rng_state(), random.getstate()
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        return StageRuntime(_spec(stages=2, microbatches=2), 0, 0)
    finally:
        torch.set_rng_state(rng)
        random.setstate(py)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(det, warn_only=warn)


def test_telemetry_events_are_neither_lost_nor_duplicated_across_threads(monkeypatch):
    # covers: RUNTIME-I9
    # WOULD-FAIL-IF: _ev or _take_events does not hold rt._tel_lock (the deterministic part
    # fails on every run, whatever the GIL schedule), or the take iterates the shared list and
    # then clear()s it, losing appends that land in between (the stress part).
    monkeypatch.delenv("WANTRAIN_SIM_SLOW", raising=False)
    rt = _stage_runtime_without_sockets()
    rt._take_events()

    # Deterministic: both entry points wait for the lock the test holds.
    takes = []
    appender = threading.Thread(target=rt._ev, args=("probe", 0, -1, -1, 0.0), kwargs={"uid": -1})
    taker = threading.Thread(target=lambda: takes.append(rt._take_events()))
    with rt._tel_lock:
        appender.start()
        taker.start()
        time.sleep(0.2)
        blocked = (appender.is_alive(), taker.is_alive())
    appender.join(5)
    taker.join(5)
    assert blocked == (True, True), f"_ev / _take_events did not wait for _tel_lock: {blocked}"
    assert not appender.is_alive() and not taker.is_alive()
    takes.append(rt._take_events())
    assert [e["uid"] for t in takes for e in t] == [-1]

    # Stress: 4 appenders racing a taking loop under a 1 us switch interval.
    takes = []
    stop = threading.Event()

    def append_many(base):
        for i in range(2000):
            rt._ev("probe", 0, -1, -1, 0.0, uid=base + i)

    def take_loop():
        while not stop.is_set():
            got = rt._take_events()
            if got:
                takes.append(got)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        loop = threading.Thread(target=take_loop)
        loop.start()
        workers = [threading.Thread(target=append_many, args=(k * 2000,)) for k in range(4)]
        for t in workers:
            t.start()
        for t in workers:
            t.join()
        stop.set()
        loop.join()
    finally:
        sys.setswitchinterval(old)
    takes.append(rt._take_events())
    assert sorted(e["uid"] for t in takes for e in t) == list(range(8000))


def test_marker_lines_from_two_threads_are_never_glued(monkeypatch):
    # covers: RUNTIME-I9
    # WOULD-FAIL-IF: run() leaves sys.stdout unwrapped, or LineAtomicStdout passes a partial line
    # through, so the loss tail's WT_METRIC text and its newline straddle this loop's WT_STEP, the
    # agent reads 'WT_METRIC {...}WT_STEP 3' and drops both (a lost final WT_STEP keeps the
    # coordinator out of shutdown and S4 restarts a finished run)
    import io
    import os

    from wantrain.runtime.stage import LineAtomicStdout

    # Deterministic: the tail thread has written its text but not its newline when the main
    # thread prints a whole marker; the wrapped stream sees two whole lines, in that order.
    writes = []
    out = LineAtomicStdout(SimpleNamespace(write=writes.append, flush=lambda: None))
    half, go = threading.Event(), threading.Event()

    def tail():
        print('WT_METRIC {"step": 3}', end="", file=out)
        half.set()
        go.wait(5)
        print(file=out, flush=True)

    t = threading.Thread(target=tail)
    t.start()
    assert half.wait(5)
    print("WT_STEP 3", file=out, flush=True)
    go.set()
    t.join(5)
    assert "".join(writes) == 'WT_STEP 3\nWT_METRIC {"step": 3}\n', writes
    assert all(w.endswith("\n") for w in writes), writes

    # Stress: two threads print 2000 markers each into a real pipe (each write releases the
    # GIL) under a 1 us switch interval; every line read back is one whole marker.
    rfd, wfd = os.pipe()
    out = LineAtomicStdout(io.TextIOWrapper(io.FileIO(wfd, "w"), write_through=True))
    data = []

    def read_all():
        with io.FileIO(rfd, "r") as f:
            data.append(f.readall())

    reader = threading.Thread(target=read_all)
    reader.start()

    def emit(name):
        for i in range(2000):
            print(f"{name} {i}", file=out, flush=True)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        writers = [threading.Thread(target=emit, args=(n,)) for n in ("WT_STEP", "WT_METRIC")]
        for w in writers:
            w.start()
        for w in writers:
            w.join()
    finally:
        sys.setswitchinterval(old)
    out.close()
    reader.join(10)
    want = [f"{n} {i}" for n in ("WT_STEP", "WT_METRIC") for i in range(2000)]
    assert sorted(data[0].decode().splitlines()) == sorted(want)

    # run() installs the wrapper before anything else, so the tail's prints go through it too.
    class _Stop(Exception):
        pass

    def no_connect():
        raise _Stop

    rt = _stage_runtime_without_sockets()
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(write=writes.append, flush=lambda: None))
    monkeypatch.setattr(rt, "_connect", no_connect)
    with pytest.raises(_Stop):
        rt.run()
    assert isinstance(sys.stdout, LineAtomicStdout)


def _commit(rt, step):
    """One checkpoint, waited for: _checkpoint returns once the payload is
    serialized, and the commit (hash, renames, alias, WT_WAVE) runs off the
    step until _await_wave."""
    rt._checkpoint(step)
    rt._await_wave()


def _tiny_stage(tmp_path, fill, **spec_kw):
    """A StageRuntime holding only what _checkpoint and _resume read: one
    nn.Linear chunk with distinct values (from `fill`) and its AdamW after one
    step, so the optimizer state is not empty. No global RNG draw. Its spec is
    a real finalized JobSpec (spec_kw over the toy defaults), whose pins() the
    checkpoint records and the resume checks."""
    chunk = torch.nn.ModuleList([torch.nn.Linear(3, 2)])
    params = list(chunk.parameters())
    with torch.no_grad():
        for i, p in enumerate(params):
            p.copy_(torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape) * fill + i)
    opt = torch.optim.AdamW(params, lr=0.1)
    for p in params:
        p.grad = torch.full_like(p, fill)
    opt.step()
    rt = StageRuntime.__new__(StageRuntime)
    rt.spec = _spec(out_dir=str(tmp_path), tag="t", **spec_kw)
    rt.rank, rt.peer = 0, 0
    rt.my_vs = [0]
    rt.chunks = {0: chunk}
    rt.opt = opt
    rt.losses = {}
    rt.conns = {}  # no peer sockets: a failing run() closes none
    return rt


def _wave5(tmp_path, payload=None, **spec_kw):
    """A real committed wave of step 5 (and its alias) from a stage whose values
    (3.0) differ from the one that will resume; returns the wave directory.
    `payload` replaces the checkpoint's state.pt (a payload _checkpoint would
    not write)."""
    from wantrain.fault_tolerance.checkpoint import commit_wave, link_legacy, wave_dir

    src = _tiny_stage(tmp_path, 3.0, **spec_kw)
    src.losses = {(5, 0): 2.5}
    if payload is None:
        _commit(src, 5)
    else:
        commit_wave(str(tmp_path), "t", 0, 0, 5, lambda tmp: torch.save(payload, tmp))
        link_legacy(wave_dir(str(tmp_path), "t", 0, 0, 5), str(tmp_path), "t", 0, 0)
    return wave_dir(str(tmp_path), "t", 0, 0, 5)


def _rewrite_manifest(d, **changes):
    with open(d + "wave.json") as f:
        m = json.load(f)
    m.update(changes)
    with open(d + "wave.json", "w") as f:
        json.dump(m, f)


def _resume_error(rt):
    before = [p.detach().clone() for p in rt.chunks[0].parameters()]
    opt_before = rt.opt.state_dict()["state"]
    opt_before = {k: {n: t.clone() for n, t in v.items()} for k, v in opt_before.items()}
    rng = torch.get_rng_state()
    err = None
    try:
        rt._resume()
    except RuntimeError as e:
        err = e
    unchanged = all(torch.equal(a, b) for a, b in zip(before, rt.chunks[0].parameters()))
    opt_after = rt.opt.state_dict()["state"]
    opt_same = opt_after.keys() == opt_before.keys() and all(
        torch.equal(opt_before[k][n], opt_after[k][n]) for k in opt_before for n in opt_before[k]
    )
    return err, unchanged, opt_same, torch.equal(rng, torch.get_rng_state())


def test_resume_refuses_a_wave_whose_manifest_names_another_step(tmp_path):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: _resume takes the start step from the directory name or state["step"]
    # alone (or checks wave.json only after loading weights, optimizer or RNG state): it would
    # return 6 from a step-5 wave whose wave.json names step 3, and a healed slot would start
    # at a step its manifest does not prove (the resume-trust mutant, ported from the sidecar).
    d = _wave5(tmp_path)
    _rewrite_manifest(d, step=3)
    rt = _tiny_stage(tmp_path, 7.0, resume=True, resume_step=5)
    err, unchanged, opt_same, rng_same = _resume_error(rt)
    assert err is not None, "resume accepted a step-5 wave whose wave.json names step 3"
    assert d in str(err) and "names step 3" in str(err) and "step 5" in str(err), err
    assert unchanged and opt_same and rng_same and rt.losses == {}


def test_resume_refuses_a_wave_without_a_manifest(tmp_path):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: _resume treats a missing wave.json as "trust the wave" (a directory
    # renamed into place by something other than a commit, or a restore that never wrote one,
    # would resume from an unproven wave).
    d = _wave5(tmp_path)
    os.unlink(d + "wave.json")
    rt = _tiny_stage(tmp_path, 7.0, resume=True, resume_step=5)
    err, unchanged, opt_same, rng_same = _resume_error(rt)
    assert err is not None, "resume accepted a wave with no wave.json"
    assert d in str(err) and "wave.json is missing" in str(err), err
    assert unchanged and opt_same and rng_same and rt.losses == {}


def test_resume_refuses_a_wave_whose_state_is_not_a_checkpoint(tmp_path):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: _resume calls state.get() on whatever state.pt holds: a wave whose manifest
    # proves a list then dies with AttributeError instead of the "resume refused" the run
    # reports, or, worse, a payload with a get() is read as a checkpoint
    d = _wave5(tmp_path, payload=[5, "not a checkpoint"])
    rt = _tiny_stage(tmp_path, 7.0, resume=True, resume_step=5)
    err, unchanged, opt_same, rng_same = _resume_error(rt)
    assert err is not None, "resume accepted a state.pt that holds a list"
    assert "resume refused" in str(err) and d in str(err) and "list" in str(err), err
    assert unchanged and opt_same and rng_same and rt.losses == {}


def _plant_extra(d):
    with open(d + "extra.bin", "wb") as f:
        f.write(b"x")
    return "extra.bin is present but not listed"


def _flip_a_byte(d):
    # In place: the alias is the same inode, so it changes too; the manifest does not.
    with open(d + "state.pt", "r+b") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 1]))
    return "has sha256"


def _truncate(d):
    with open(d + "state.pt", "r+b") as f:
        f.truncate(os.path.getsize(d + "state.pt") - 1)
    return "bytes, manifest lists"


def _remove_listed(d):
    os.unlink(d + "state.pt")
    return "listed file state.pt is missing"


@pytest.mark.parametrize("damage", [_plant_extra, _flip_a_byte, _truncate, _remove_listed])
def test_resume_refuses_an_unlisted_or_mismatching_file(tmp_path, damage):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: _resume skips validate_wave (or runs it after applying state), so a wave
    # holding an unlisted file, a listed file with another sha256 or size, or missing a listed
    # file is loaded (or torch.load fails with an error that does not name the wave).
    d = _wave5(tmp_path)
    reason = damage(d)
    rt = _tiny_stage(tmp_path, 7.0, resume=True, resume_step=5)
    err, unchanged, opt_same, rng_same = _resume_error(rt)
    assert err is not None, f"resume accepted a damaged wave ({damage.__name__})"
    assert d in str(err) and reason in str(err), err
    assert unchanged and opt_same and rng_same and rt.losses == {}


def test_resume_reads_the_named_wave_not_the_alias(tmp_path):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: _resume reads the alias (<tag>_ckpt_rank<r>p<p>.pt + .step.json) instead
    # of the wave of resume_step: with the alias on wave 5 a resume_step=3 launch would start
    # at step 6 with step 5's weights; or, with wave 5 gone and its alias left behind, it would
    # resume from the alias instead of refusing like a missing shard; or local mode (no
    # resume_step) takes another wave than the newest.
    from wantrain.fault_tolerance.checkpoint import wave_dir

    src = _tiny_stage(tmp_path, 3.0)
    for step in (3, 5):
        src.losses[(step, 0)] = float(step)
        with torch.no_grad():
            for p in src.chunks[0].parameters():
                p.add_(1.0)
        _commit(src, step)
    wave3 = torch.load(
        wave_dir(str(tmp_path), "t", 0, 0, 3) + "state.pt", map_location="cpu", weights_only=True
    )
    with open(src._ckpt_path()[:-3] + ".step.json") as f:
        assert json.load(f) == {"step": 5}, "the alias should name step 5"

    rt = _tiny_stage(tmp_path, 7.0, resume=True, resume_step=3)
    assert rt._resume() == 4
    want = wave3["chunks"][0]
    assert all(torch.equal(rt.chunks[0].state_dict()[k], want[k]) for k in want)
    assert rt.losses == {(3, 0): 3.0}

    local = _tiny_stage(tmp_path, 7.0, resume=True)
    assert local._resume() == 6
    assert local.losses == {(3, 0): 3.0, (5, 0): 5.0}

    gone = wave_dir(str(tmp_path), "t", 0, 0, 5)
    for name in os.listdir(gone):
        os.unlink(gone + name)
    os.rmdir(gone)
    assert os.path.exists(src._ckpt_path()), "the alias of step 5 should still be there"
    rt = _tiny_stage(tmp_path, 7.0, resume=True, resume_step=5)
    err, unchanged, opt_same, rng_same = _resume_error(rt)
    assert err is not None, "resume read the alias of a missing wave"
    assert gone in str(err) and "does not exist" in str(err), err
    assert unchanged and opt_same and rng_same and rt.losses == {}

    alias_only = tmp_path / "alias_only"
    alias_only.mkdir()
    for name in ("t_ckpt_rank0p0.pt", "t_ckpt_rank0p0.step.json"):
        with open(tmp_path / name, "rb") as f, open(alias_only / name, "wb") as g:
            g.write(f.read())
    rt = _tiny_stage(alias_only, 7.0, resume=True)
    err, unchanged, opt_same, rng_same = _resume_error(rt)
    assert err is not None, "local mode resumed from an alias with no wave"
    assert "holds no wave" in str(err), err
    assert unchanged and opt_same and rng_same and rt.losses == {}


def test_checkpoint_writes_through_the_seqlock(tmp_path, monkeypatch):
    # covers: RUNTIME-I3, CONTROL-I10
    # WOULD-FAIL-IF: _checkpoint replaces the shard while the old sidecar still names the old
    # step (today's writer, or the stage-bypass mutant that makes checkpoint.invalidate_sidecar
    # a no-op), writes the sidecar in place instead of by rename, or changes the payload's keys,
    # their order, its step or its loss filter.
    rt = _tiny_stage(tmp_path, 7.0)
    rt.losses = {(4, 0): 1.5, (5, 1): 2.5, (6, 0): 3.5}
    path = rt._ckpt_path()
    side = path[:-3] + ".step.json"
    torch.save({"step": 3, "wave": torch.arange(5)}, path)
    with open(side, "w") as f:
        json.dump({"step": 3}, f)

    real = os.replace
    dests = []
    side_at_shard_replace = []

    def recording_replace(src, dst, *a, **kw):
        dests.append(str(dst))
        if str(dst) == path:
            side_at_shard_replace.append(os.path.exists(side))
        return real(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", recording_replace)
    rt._checkpoint(5)
    rt._await_wave()
    monkeypatch.setattr(os, "replace", real)

    assert dests == [path, side], dests
    assert side_at_shard_replace == [False], "the old sidecar still named step 3 at the replace"
    with open(side) as f:
        assert json.load(f) == {"step": 5}
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert list(state) == ["step", "chunks", "opt", "rng", "losses", "pins"]
    assert state["pins"] == rt.spec.pins()
    assert state["step"] == 5
    assert state["losses"] == {"4:0": 1.5, "5:1": 2.5}
    want = rt.chunks[0].state_dict()
    assert all(torch.equal(state["chunks"][0][k], want[k]) for k in want)
    assert not os.path.exists(path + ".tmp") and not os.path.exists(side + ".tmp")


# --- E6: piece kernels against the whole-tensor reference (RUNTIME-I11) ---------------------

_INT_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}


def _same_bits(a, b):
    """Bitwise equality: -0.0 != +0.0 and NaN payloads compare as integers."""
    return a.dtype == b.dtype and torch.equal(
        a.contiguous().view(_INT_VIEW[a.dtype]), b.contiguous().view(_INT_VIEW[b.dtype])
    )


def test_e6_piece_layout_cuts_dim0_on_lcm_quanta():
    # covers: RUNTIME-I11
    # WOULD-FAIL-IF: piece_layout cuts on plain 128-element (or row) multiples instead of
    # lcm(row length, 128), deals quanta unequally (sizes differ by more than one quantum),
    # gives w == 1 more than one piece, or fails to leave empty pieces for a tensor with fewer
    # quanta than peers.
    from wantrain.runtime import pieces

    shapes = {
        "mat": (300, 384),  # Q 384
        "vec": (1000,),  # Q 128
        "tiny": (3,),  # one partial quantum
        "odd": (512, 96),  # row 96 -> Q 384 (not a multiple of 128 rows)
        "ragged": (7, 5),  # row 5 -> Q 640, numel 35
        "scalar": (),
    }
    assert [pieces.quantum(s) for s in shapes.values()] == [384, 128, 128, 384, 640, 128]
    # Hand-computed: 8 quanta of "vec" over 3 peers is 3, 3, 2; "odd" has 128 quanta of 384.
    lay3 = pieces.piece_layout(shapes, 3)
    assert lay3["vec"] == (0, 384, 768, 1000)
    assert lay3["odd"] == (0, 384 * 43, 384 * 86, 49152)
    assert lay3["tiny"] == (0, 3, 3, 3)
    assert lay3["scalar"] == (0, 1, 1, 1)
    assert pieces.piece_layout(shapes, 1) == {
        n: (0, max(1, int(torch.Size(s).numel()))) for n, s in shapes.items()
    }
    assert pieces.piece_layout({"tiny": (3,)}, 4)["tiny"] == (0, 3, 3, 3, 3)
    for w in range(1, 9):
        lay = pieces.piece_layout(shapes, w)
        assert list(lay) == list(shapes)
        for name, shape in shapes.items():
            b, q = lay[name], pieces.quantum(shape)
            numel = int(torch.Size(shape).numel())
            row = pieces.row_length(shape)
            assert len(b) == w + 1 and b[0] == 0 and b[-1] == numel
            assert all(lo <= hi for lo, hi in zip(b, b[1:]))
            assert all(x % q == 0 or x == numel for x in b), (name, w, b)
            quanta = [-(-(hi - lo) // q) for lo, hi in zip(b, b[1:])]
            assert max(quanta) - min(quanta) <= 1, (name, w, quanta)
            n_q = -(-numel // q)
            assert sum(1 for x in quanta if x == 0) == max(0, w - n_q)
            if len(shape) == 2:  # a piece is whole rows: the dim-0 slice
                t = torch.arange(numel, dtype=torch.float32).reshape(shape)
                for k in range(w):
                    want = t[b[k] // row : b[k + 1] // row].reshape(-1)
                    assert torch.equal(pieces.piece(t, b, k), want)


def test_e6_piece_layout_refuses_bounds_that_are_unaligned_unordered_or_not_tiling():
    # covers: RUNTIME-I11
    # WOULD-FAIL-IF: explicit bounds are accepted when an inner boundary is off the
    # lcm(row length, 128) quantum, when they decrease, when they do not start at 0 or end at
    # numel, or when their count is not w + 1; or when the error does not name the tensor.
    from wantrain.runtime import pieces

    shapes = {"blocks.0.fc1.weight": (512, 96), "bias": (1000,)}
    ok = pieces.piece_layout(shapes, 3, {"bias": (0, 0, 896, 1000)})
    assert ok["bias"] == (0, 0, 896, 1000)
    assert ok["blocks.0.fc1.weight"] == pieces.piece_layout(shapes, 3)["blocks.0.fc1.weight"]
    assert pieces.piece_layout({"t": (3,)}, 2, {"t": (0, 3, 3)})["t"] == (0, 3, 3)
    bad = {
        "unaligned": (0, 96 * 3, 96 * 300, 49152),  # row multiple, not a 384 quantum
        "unaligned128": (0, 128, 768, 49152),  # 128 multiple, not a row multiple
        "decreasing": (0, 768, 384, 49152),
        "start": (384, 768, 1152, 49152),
        "end": (0, 384, 768, 49152 - 384),
        "past_end": (0, 384, 49536, 49152),
        "count": (0, 384, 49152),
    }
    for why, b in bad.items():
        with pytest.raises(ValueError, match="blocks.0.fc1.weight"):
            pieces.piece_layout(shapes, 3, {"blocks.0.fc1.weight": b})
            pytest.fail(f"accepted {why} bounds {b}")
    with pytest.raises(ValueError, match="ghost"):
        pieces.piece_layout(shapes, 3, {"ghost": (0, 0, 0, 0)})
    for w in (0, -1, 1.0):
        with pytest.raises(ValueError, match="width"):
            pieces.piece_layout(shapes, w)


def test_e6_exchange_units_are_forward_ordered_and_cover_every_parameter_once():
    # covers: RUNTIME-I11
    # WOULD-FAIL-IF: exchange_units puts the Embed's tables anywhere but first or apart, splits
    # a SplitLinear's weight from its bias, leaves the Head's norm outside the Head linear's unit,
    # spreads a block's norms over several units, lists a parameter twice or not at all (tied
    # embeddings), or depends on anything but the module structure.
    from wantrain.model.gpt import GPTConfig, build_layers
    from wantrain.runtime import pieces

    cfg = GPTConfig(n_layer=2)
    layers = build_layers(cfg, 3)
    units = pieces.exchange_units(layers)
    names = [[n for n, _ in u] for u in units]
    want = [["0.tok.weight", "0.pos.weight"]]
    for li in (1, 2):
        want.append([f"{li}.ln1.weight", f"{li}.ln1.bias", f"{li}.ln2.weight", f"{li}.ln2.bias"])
        want += [[f"{li}.{m}.weight", f"{li}.{m}.bias"] for m in ("qkv", "proj", "fc1", "fc2")]
    want.append(["3.ln_f.weight", "3.ln_f.bias", "3.out.weight"])
    assert names == want
    by_name = dict(layers.named_parameters())
    for u in units:
        for n, p in u:
            li, rest = n.split(".", 1)
            assert p is by_name[f"{li}.{rest}"]
    assert names == [[n for n, _ in u] for u in pieces.exchange_units(build_layers(cfg, 9))]
    # A mid-pipeline chunk names its parameters by position in the chunk.
    mid = torch.nn.ModuleList(list(layers)[2:])
    assert [[n for n, _ in u] for u in pieces.exchange_units(mid)] == [
        [n.replace("2.", "0.", 1).replace("3.", "1.", 1) for n in u] for u in want[6:]
    ]
    # General architecture with tied embeddings: every trainable tensor exactly once.
    gcfg = GPTConfig(
        n_layer=2, norm="rmsnorm", mlp="swiglu", tie_embeddings=True, init_scheme="gpt2"
    )
    glayers = build_layers(gcfg, 3)
    gunits = pieces.exchange_units(glayers)
    got = [id(p) for u in gunits for _, p in u]
    trainable = {id(p) for p in glayers.parameters() if p.requires_grad}
    assert len(got) == len(set(got)) and set(got) == trainable
    # The tied output weight is the Embed's `tok`: it stays in the tables' unit, and the Head
    # (whose linear is a plain nn.Linear, never split) keeps only its norm.
    assert [n for n, _ in gunits[0]] == ["0.tok.weight", "0.pos.weight"]
    assert [n for n, _ in gunits[1]] == ["1.ln1.weight", "1.ln2.weight"]
    assert [n for n, _ in gunits[-1]] == ["3.ln_f.weight"]


def _toy_grads(compute, M):
    """Per-microbatch gradients of a 2-block toy GPT as the B/W split produces them: split
    linears' dW and db in their native (autocast) dtype from the wgrad store, every other
    parameter in fp32 from autograd. -> (params, [{id(p): native grad}] per mb)."""
    from wantrain.model import wgrad
    from wantrain.model.data import make_batch
    from wantrain.model.gpt import GPTConfig, build_layers, loss_fn

    cfg = GPTConfig(n_layer=2, compute_dtype=compute, seq_len=32)
    layers = build_layers(cfg, 11)
    params = [p for p in layers.parameters() if p.requires_grad]
    owned = wgrad.owned_params(layers)
    others = [p for p in params if id(p) not in owned]
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
        assert set(native) == {id(p) for p in params}
        per_mb.append(native)
    return params, per_mb


@pytest.mark.parametrize("compute", ["fp32", "bf16", "fp16"])
def test_e6_piece_fold_equals_the_whole_tensor_ordered_fold(compute):
    # covers: RUNTIME-I11
    # WOULD-FAIL-IF: the piece fold starts from a clone in the wire (autocast) dtype, from
    # zeros, folds in anything but microbatch-index order, or does not widen to fp32 before
    # adding; or a piece is not the exact slice of the whole tensor at some width 1..8.
    from wantrain.runtime import pieces
    from wantrain.runtime.accumulate import OrderedAccumulator

    M = 4
    params, per_mb = _toy_grads(compute, M)
    if compute != "fp32":
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[compute]
        assert any(g.dtype == dt for g in per_mb[0].values()), "no native-dtype gradient"
    # Whole-tensor reference: today's fold over flush-widened gradients, arriving out of order.
    ref = OrderedAccumulator(range(M))
    for mb in (2, 0, 3, 1):
        ref.add(mb, [per_mb[mb][id(p)].to(p.dtype) for p in params])
    whole = ref.result()
    shapes = {str(i): tuple(p.shape) for i, p in enumerate(params)}
    bf16_clone_differs = False
    for w in range(1, 9):
        lay = pieces.piece_layout(shapes, w)
        for i, p in enumerate(params):
            b = lay[str(i)]
            for k in range(w):
                got = pieces.fold(pieces.piece(per_mb[mb][id(p)], b, k) for mb in range(M))
                want = pieces.piece(whole[i], b, k)
                assert got.dtype == torch.float32
                assert _same_bits(got, want), (compute, w, i, k)
                g0 = pieces.piece(per_mb[0][id(p)], b, k)
                if compute == "bf16" and g0.dtype == torch.bfloat16 and g0.numel():
                    # control: an accumulator cloned in the wire dtype
                    acc = g0.clone()
                    for mb in range(1, M):
                        acc.add_(pieces.piece(per_mb[mb][id(p)], b, k))
                    bf16_clone_differs |= not torch.equal(acc.float(), want)
    assert bf16_clone_differs or compute != "bf16", "the bf16-clone control never differed"


def test_e6_piece_fold_keeps_the_sign_of_zero_a_zeros_start_loses():
    # covers: RUNTIME-I11
    # WOULD-FAIL-IF: the fold accumulator starts from zeros (+0 + -0 = +0 drops the sign the
    # reference keeps) or from the input tensor itself (aliasing microbatch 0's buffer).
    from wantrain.runtime import pieces
    from wantrain.runtime.accumulate import OrderedAccumulator

    gs = [torch.full((256,), -0.0, dtype=torch.bfloat16) for _ in range(3)]
    gs[1][:128] = 0.5
    ref = OrderedAccumulator(range(3))
    for mb in range(3):
        ref.add(mb, [gs[mb].float()])
    want = ref.result()[0]
    b = (0, 128, 256)
    for k in range(2):
        got = pieces.fold(pieces.piece(g, b, k) for g in gs)
        assert _same_bits(got, pieces.piece(want, b, k))
    assert torch.signbit(pieces.fold(pieces.piece(g, b, 1) for g in gs)).all()
    zeros = torch.zeros(128)
    for g in gs:
        zeros.add_(pieces.piece(g, b, 1).float())
    assert not _same_bits(zeros, pieces.piece(want, b, 1)), "the zeros control did not differ"
    acc = pieces.fold_start(gs[0])
    pieces.fold_add(acc, gs[1])
    assert torch.signbit(gs[0]).all() and gs[0].dtype == torch.bfloat16  # input untouched
    with pytest.raises(ValueError):
        pieces.fold([])


def test_e6_piece_cast_equals_the_whole_cast_on_all_2_32_fp32_patterns():
    # covers: RUNTIME-I11
    # WOULD-FAIL-IF: a piece-local cast to bf16 or fp16 differs from the slice of the
    # whole-tensor cast for any fp32 bit pattern (a NaN payload, a subnormal, a tie, an
    # overflow), e.g. a cast path that depends on the piece's offset or length; or the check
    # skips patterns (the count is asserted to be exactly 2^32).
    from wantrain.runtime import pieces

    row = 256  # Q 256; 2^26 patterns per chunk are 2^18 rows
    chunk = 1 << 26
    base = torch.arange(chunk, dtype=torch.int32)
    rng = random.Random(6)
    checked = 0
    for c in range(1 << 6):
        off = c * chunk
        off = off - (1 << 32) if off >= 1 << 31 else off  # signed: no int32 overflow below
        x = (base + off).view(torch.float32)
        numel = x.numel()
        q = pieces.quantum((numel // row, row))
        w = rng.randint(1, 8)
        inner = sorted(rng.randrange(0, numel // q + 1) * q for _ in range(w - 1))
        b = pieces.piece_layout({"t": (numel // row, row)}, w, {"t": [0, *inner, numel]})["t"]
        for dt in (torch.bfloat16, torch.float16):
            whole = x.to(dt)
            out = torch.empty_like(whole)
            for k in range(w):
                pieces.piece(out, b, k).copy_(pieces.cast_piece(pieces.piece(x, b, k), dt))
            assert _same_bits(out, whole), (c, dt, b)
        if c == 0:
            # control: truncation instead of round-to-nearest-even must be caught
            trunc = (x.view(torch.int32) >> 16).to(torch.int16)
            assert not torch.equal(trunc, x.to(torch.bfloat16).view(torch.int16))
        checked += numel
    assert checked == 1 << 32
    master = torch.ones(4)
    copy = pieces.cast_piece(master, torch.float32)
    copy.add_(1)
    assert torch.equal(master, torch.ones(4)), "the fp32 working copy aliases the master"


def _adamw_tensors():
    g = torch.Generator().manual_seed(61)
    shapes = {
        "tok": (256, 128),
        "qkv.weight": (384, 128),
        "qkv.bias": (384,),
        "ln.weight": (128,),
        "tiny": (3,),  # one quantum: empty pieces at every w > 1
        "odd": (7, 5),
        "big": (1000, 384),  # above the CPU kernels' parallel grain
    }
    init = {n: torch.randn(s, generator=g) for n, s in shapes.items()}
    grads = [{n: torch.randn(s, generator=g) for n, s in shapes.items()} for _ in range(3)]
    return shapes, init, grads


def _whole_adamw(init, grads, lr, betas=(0.9, 0.95)):
    ps = {n: torch.nn.Parameter(t.clone()) for n, t in init.items()}
    opt = torch.optim.AdamW(list(ps.values()), lr=lr, betas=betas, eps=1e-8, weight_decay=0.0)
    for step_grads in grads:
        for n, p in ps.items():
            p.grad = step_grads[n].clone()
        opt.step()
    return ps, opt


def _ragged_bounds(shapes, w, rng):
    from wantrain.runtime import pieces

    out = {}
    for n, s in shapes.items():
        numel, q = int(torch.Size(s).numel()), pieces.quantum(s)
        inner = sorted(min(numel, rng.randrange(0, numel // q + 2) * q) for _ in range(w - 1))
        out[n] = [0, *inner, numel]
    return out


def test_e6_piece_adamw_equals_slices_of_whole_tensor_adamw():
    # covers: RUNTIME-I11
    # WOULD-FAIL-IF: the piece optimizer is not the reference's constructor (torch's default
    # betas (0.9, 0.999) is the control below), its update depends on the piece's offset or
    # length, an empty piece breaks the step, or a per-unit step touches another unit's state
    # or step counter (e.g. stale gradients left set double-step a unit).
    from wantrain.runtime import pieces

    lr = 1e-3
    shapes, init, grads = _adamw_tensors()
    ps, opt = _whole_adamw(init, grads, lr)
    names = list(shapes)
    units = [names[:3], names[3:]]
    rng = random.Random(8)
    for w in range(1, 9):
        layouts = {
            "equal": pieces.piece_layout(shapes, w),
            "ragged": pieces.piece_layout(shapes, w, _ragged_bounds(shapes, w, rng)),
        }
        for (how, lay), per_unit in [(x, u) for x in layouts.items() for u in (False, True)]:
            for k in range(w):
                mine = {n: pieces.piece(init[n], lay[n], k).clone() for n in names}
                popt = pieces.adamw(list(mine.values()), lr)
                for step_grads in grads:
                    gk = {n: pieces.piece(step_grads[n], lay[n], k).clone() for n in names}
                    if per_unit:
                        for ui, u in enumerate(units):
                            pieces.step_unit(popt, [mine[n] for n in u], [gk[n] for n in u])
                            if ui == 0 and step_grads is grads[0]:
                                assert all(mine[n] not in popt.state for n in units[1])
                    else:
                        pieces.step_unit(popt, [mine[n] for n in names], [gk[n] for n in names])
                for n in names:
                    whole_state, st = opt.state[ps[n]], popt.state[mine[n]]
                    where = (w, how, per_unit, k, n)
                    assert _same_bits(mine[n], pieces.piece(ps[n].detach(), lay[n], k)), where
                    for key in ("exp_avg", "exp_avg_sq"):
                        want = pieces.piece(whole_state[key], lay[n], k)
                        assert _same_bits(st[key], want), (*where, key)
                    assert torch.equal(st["step"], whole_state["step"]), where
                    assert mine[n].grad is None
    # control: torch's default betas over the same pieces must not match
    ps_default, _ = _whole_adamw(init, grads, lr, betas=(0.9, 0.999))
    assert not _same_bits(ps_default["big"].detach(), ps["big"].detach())
    with pytest.raises(ValueError):
        pieces.step_unit(popt, [torch.zeros(1)], [torch.zeros(1)])


def test_e6_kernels_never_draw_from_the_torch_rng():
    # covers: RUNTIME-I11
    # WOULD-FAIL-IF: a piece kernel (layout, units, cast, fold, AdamW, unit step) draws from
    # the global torch generator, which would shift every later draw of the training run.
    from wantrain.model.gpt import GPTConfig, build_layers
    from wantrain.runtime import pieces

    layers = build_layers(GPTConfig(n_layer=1), 1)
    t = torch.arange(4096, dtype=torch.float32).reshape(32, 128)
    calls = {
        "layout": lambda: pieces.piece_layout({"t": (32, 128)}, 3),
        "units": lambda: pieces.exchange_units(layers),
        "cast": lambda: pieces.cast_piece(pieces.piece(t, (0, 1024, 4096), 1), torch.bfloat16),
        "fold": lambda: pieces.fold([t, t.bfloat16(), t]),
    }
    p = t.clone()
    opt = pieces.adamw([p], 1e-3)
    calls["adamw"] = lambda: pieces.step_unit(opt, [p], [torch.ones_like(p)])
    for name, call in calls.items():
        before = torch.get_rng_state()
        call()
        assert torch.equal(before, torch.get_rng_state()), name
    before = torch.get_rng_state()
    torch.rand(1)  # control: a draw does change the state
    assert not torch.equal(before, torch.get_rng_state())


# ------------------------------------------------------------- RUNTIME-I15

_PID = "0123456789abcdef0123456789abcdef"


def test_slot_peers_and_resume_step_are_checked():
    # covers: RUNTIME-I15
    # WOULD-FAIL-IF: finalize() accepts slot_peers of another shape than stage_widths or an id
    # that is not 32 lowercase hex, accepts a negative, bool or non-int resume_step or one
    # without resume, or either field is lost on a JSON round trip.
    ok = _spec(
        stages=3,
        stage_widths=[1, 2, 1],
        slot_peers=[[_PID], [_PID, "f" * 32], ["a" * 32]],
        resume=True,
        resume_step=3,
    )
    back = _refinalize(ok)
    assert back.slot_peers == [[_PID], [_PID, "f" * 32], ["a" * 32]]
    assert back.resume_step == 3
    assert _refinalize(_spec(resume=True, resume_step=0)).resume_step == 0
    assert _refinalize(_spec()).slot_peers is None and _refinalize(_spec()).resume_step is None
    bad_peers = [
        [[_PID], [_PID]],  # wide stage 1 lists one peer
        [[_PID], [_PID, _PID], [_PID]],  # one stage too many
        [[_PID], [_PID, _PID.upper()]],
        [[_PID], [_PID, _PID[:-1]]],
        [[_PID], [_PID, _PID[:-1] + "g"]],
        [[_PID], [_PID, None]],
    ]
    for peers in bad_peers:
        with pytest.raises(AssertionError, match="slot_peers"):
            _spec(stages=2, stage_widths=[1, 2], slot_peers=peers)
    for step, resume in [(-1, True), (True, True), (1.0, True), ("3", True), (2, False)]:
        with pytest.raises(AssertionError, match="resume_step"):
            _spec(resume=resume, resume_step=step)


def _pin_variants(base):
    """One spec per recipe pin, each differing from base in that pin only."""
    from dataclasses import replace

    a, b = base.partition[0]
    moved = [[a, b + 1], [b + 1, base.partition[1][1]]]
    return {
        "model.seq_len": dict(model=replace(base.model, seq_len=base.model.seq_len * 2)),
        "model.compute_dtype": dict(model=replace(base.model, compute_dtype="bf16")),
        "init_seed": dict(init_seed=base.init_seed + 1),
        "data_seed": dict(data_seed=base.data_seed + 1),
        "dataset_path": dict(dataset_path="/data/tokens.bin"),
        "dataset_sha256": dict(dataset_sha256="ab" * 32),
        "data_shards": dict(data_shards=2),
        "microbatches": dict(microbatches=base.microbatches * 2),
        "mb_size": dict(mb_size=base.mb_size + 1),
        "lr": dict(lr=base.lr * 2),
        "wire_codec": dict(wire_codec="planes"),
        "recompute": dict(recompute=not base.recompute),
        "partition": dict(partition=moved),
        # one device type (CONTROL-I25), moved: the slots' type differs from base's cpu
        "device_types": dict(devices=[["mps"], ["mps"]]),
        "zero1_widths": dict(stage_widths=[1, 2]),
    }


def test_pins_cover_the_recipe_and_nothing_else():
    # covers: RUNTIME-I15
    # WOULD-FAIL-IF: pins() leaves out a recipe field (a resume under another seed, partition,
    # codec or width would load), includes an ownership/placement field or steps (a rebalance
    # or an extended run would be refused), first_pin_difference names the wrong field, or a
    # slot moved to another device type (cpu -> mps) is not a pin difference.
    from wantrain.runtime.spec import ADAMW, first_pin_difference

    base = _spec(stages=2)
    pins = base.pins()
    assert pins["adamw"] == {"betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.0}
    assert pins["adamw"] == json.loads(json.dumps(ADAMW))
    assert pins["zero1_widths"] == [1, 1] and pins["partition"] == base.partition
    assert pins["device_types"] == [["cpu"], ["cpu"]]
    wide = _spec(stages=2, stage_widths=[1, 2], devices=[["cuda:1"], ["cuda:0", "cuda:1"]])
    assert wide.pins()["device_types"] == [["cuda"], ["cuda"]]
    assert json.loads(json.dumps(pins)) == pins
    for field, kw in _pin_variants(base).items():
        other = JobSpec(**{"model": make_config("toy"), "stages": 2, **kw}).finalize()
        assert first_pin_difference(pins, other.pins()) == field, field
    same = _spec(
        stages=2,
        steps=base.steps * 3,
        peer_weights=[[1.0], [1.0]],
        endpoints=[[["10.0.0.1", 1]], [["10.0.0.2", 2]]],
        devices=[["cpu"], ["cpu:0"]],
        slot_peers=[[_PID], ["e" * 32]],
        trusted_peers=[_PID],
        resume=True,
        resume_step=4,
        out_dir="/elsewhere",
        tag="other",
    )
    same.schedule_order = [list(reversed(o)) for o in same.schedule_order]
    assert first_pin_difference(pins, same.pins()) is None
    assert same.pins() == pins
    assert first_pin_difference({"a": 1}, {"a": 1, "b": 2}) == "b"


def _checkpointed(tmp_path, **spec_kw):
    """A real step-5 shard (sidecar included) from _checkpoint, values 3.0."""
    src = _tiny_stage(tmp_path, 3.0, **spec_kw)
    src.losses = {(5, 0): 2.5}
    _commit(src, 5)
    return src._ckpt_path()


@pytest.mark.parametrize("field", ["microbatches", "partition", "zero1_widths", "device_types"])
def test_resume_refuses_changed_pins(tmp_path, field):
    # covers: RUNTIME-I15
    # WOULD-FAIL-IF: _resume loads a shard written under other microbatches, another
    # partition, another ZeRO-1 width or with a slot moved to another device type (or checks
    # the pins only after it applied the weights, the optimizer state or the RNG), or the
    # refusal does not name the field.
    _checkpointed(tmp_path)
    kw = _pin_variants(_spec(stages=2))[field]
    rt = _tiny_stage(tmp_path, 7.0, **{"stages": 2, **kw})
    err, unchanged, opt_same, rng_same = _resume_error(rt)
    assert err is not None, f"resume accepted a shard whose {field} differs"
    assert f"pin '{field}' differs" in str(err), err
    assert unchanged and opt_same and rng_same and rt.losses == {}


def test_resume_refuses_a_shard_without_pins(tmp_path):
    # covers: RUNTIME-I15
    # WOULD-FAIL-IF: _resume treats a shard with no pins (written before RUNTIME-I15, or
    # stripped) as matching, or reads it only after applying state.
    other = _tiny_stage(tmp_path, 3.0)
    _wave5(
        tmp_path,
        payload={
            "step": 5,
            "chunks": {0: other.chunks[0].state_dict()},
            "opt": other.opt.state_dict(),
            "rng": torch.Generator().manual_seed(1234).get_state(),
            "losses": {"5:0": 2.5},
        },
    )
    rt = _tiny_stage(tmp_path, 7.0)
    err, unchanged, opt_same, rng_same = _resume_error(rt)
    assert err is not None, "resume accepted a shard without pins"
    assert "no recipe pins" in str(err), err
    assert unchanged and opt_same and rng_same and rt.losses == {}


def test_resume_refuses_steps_below_the_checkpoint(tmp_path):
    # covers: RUNTIME-I15
    # WOULD-FAIL-IF: a resume with steps <= the checkpoint step runs no step and its final
    # checkpoint relabels the step-5 state as an earlier step.
    _checkpointed(tmp_path)
    rt = _tiny_stage(tmp_path, 7.0, steps=5)
    err, unchanged, opt_same, rng_same = _resume_error(rt)
    assert err is not None, "resume accepted steps=5 from a step-5 shard"
    assert "steps=5" in str(err) and "(6)" in str(err), err
    assert unchanged and opt_same and rng_same and rt.losses == {}


def test_resume_accepts_new_ownership_and_more_steps(tmp_path):
    # covers: RUNTIME-I15
    # WOULD-FAIL-IF: _resume refuses a shard whose only differences are peer weights,
    # ownership, schedule, addresses, devices of the same type (cpu:0 for cpu), slot keys,
    # resume_step or a larger steps (every rebalance and heal would stop), or it loads
    # something other than the shard.
    _checkpointed(tmp_path)
    src = torch.load(tmp_path / "t_ckpt_rank0p0.pt", map_location="cpu", weights_only=True)
    rt = _tiny_stage(
        tmp_path,
        7.0,
        steps=6,
        peer_weights=[[2.0], [1.0]],
        devices=[["cpu:0"], ["cpu"]],
        slot_peers=[[_PID], [_PID]],
        resume=True,
        resume_step=5,
    )
    assert rt._resume() == 6
    want = src["chunks"][0]
    assert all(torch.equal(rt.chunks[0].state_dict()[k], want[k]) for k in want)
    assert rt.losses == {(5, 0): 2.5}


# ------------------------------------------------------------- RUNTIME-I17


def _run_without_compute(rt, monkeypatch, at_first_step=None, at_finish=None, at_step=None):
    """rt.run() on a _tiny_stage with the network and the compute taken out
    (no sockets, an empty step, no optimizer update): what runs for real is
    the resume, the wave drop, every checkpoint and every announcement.
    Returns the lines printed; at_first_step() is called when the first step
    begins, before anything of that step is done, at_step(step) when each
    step begins, and at_finish(lines) when run() reaches _finish."""
    lines = []
    started = []

    def run_step(step):
        if not started and at_first_step is not None:
            at_first_step()
        started.append(step)
        if at_step is not None:
            at_step(step)

    monkeypatch.setattr(
        sys, "stdout", SimpleNamespace(write=lambda s: lines.append(s), flush=lambda: None)
    )
    rt.device = torch.device("cpu")
    rt.deferred_w = []
    monkeypatch.setattr(rt, "_connect", lambda: None)
    monkeypatch.setattr(rt, "_run_step", run_step)
    monkeypatch.setattr(rt, "_optimizer_step", lambda step: None)
    monkeypatch.setattr(rt, "_emit_step_telemetry", lambda step, ms: None)
    monkeypatch.setattr(rt, "_finish", lambda: at_finish and at_finish("".join(lines).splitlines()))
    rng = torch.get_rng_state()
    try:
        rt.run()
    finally:
        torch.set_rng_state(rng)
    return "".join(lines).splitlines(), started


def _waves(tmp_path, rank=0, peer=0, tag="t"):
    from wantrain.fault_tolerance.checkpoint import committed_waves

    return [s for s, _ in committed_waves(str(tmp_path), tag, rank, peer)]


def test_checkpoint_commits_a_wave_before_announcing_it(tmp_path, monkeypatch):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: _checkpoint prints WT_WAVE before the wave's wave.json is renamed into
    # place or before the alias and its sidecar name the step, announces a step without a
    # committed wave (the old save_shard-only writer), or the wave's state.pt is not the
    # payload the alias holds.
    from wantrain.fault_tolerance.checkpoint import validate_wave, wave_dir

    rt = _tiny_stage(tmp_path, 7.0)
    rt.losses = {(4, 0): 1.5, (5, 1): 2.5}
    path = rt._ckpt_path()
    seen = []

    def write(s):
        for line in s.splitlines():
            if line.startswith("WT_WAVE "):
                step = int(line.split()[1])
                d = wave_dir(str(tmp_path), "t", 0, 0, step)
                with open(path[:-3] + ".step.json") as f:
                    named = json.load(f)["step"]
                seen.append(
                    (
                        step,
                        validate_wave(d, step)["step"],
                        named,
                        os.path.samefile(path, d + "state.pt"),
                    )
                )

    monkeypatch.setattr(sys, "stdout", SimpleNamespace(write=write, flush=lambda: None))
    _commit(rt, 5)
    _commit(rt, 7)
    monkeypatch.undo()
    assert seen == [(5, 5, 5, True), (7, 7, 7, True)], seen
    assert _waves(tmp_path) == [5, 7]
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["step"] == 7 and state["pins"] == rt.spec.pins()


def test_last_step_is_announced_once(tmp_path, monkeypatch):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: the final checkpoint of a run whose steps is a multiple of ckpt_every
    # commits or announces the last step a second time (BA-4), the final checkpoint stops
    # going through the wave (an unannounced last step), a periodic checkpoint is skipped, or
    # run() reaches _finish before the last wave's background commit is announced.
    rt = _tiny_stage(tmp_path, 7.0, steps=6, ckpt_every=2)
    at_finish = []
    lines, started = _run_without_compute(
        rt, monkeypatch, at_finish=lambda ls: at_finish.append((ls, _waves(tmp_path)))
    )
    assert started == list(range(6))
    assert [ln for ln in lines if ln.startswith("WT_WAVE")] == [
        "WT_WAVE 1",
        "WT_WAVE 3",
        "WT_WAVE 5",
    ]
    # The commit runs off the step, so WT_WAVE 5 may follow WT_STEP 5, but never _finish.
    assert len(at_finish) == 1 and "WT_WAVE 5" in at_finish[0][0], at_finish
    assert at_finish[0][1] == [1, 3, 5]
    assert _waves(tmp_path) == [1, 3, 5]
    rt2 = _tiny_stage(tmp_path / "odd", 7.0, steps=5, ckpt_every=2)
    lines2, _ = _run_without_compute(rt2, monkeypatch)
    assert [ln for ln in lines2 if ln.startswith("WT_WAVE")] == [
        "WT_WAVE 1",
        "WT_WAVE 3",
        "WT_WAVE 4",
    ]


def test_saved_waves_are_byte_equal_to_the_alias(tmp_path):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: the alias is written separately from the wave (two torch.save calls whose
    # bytes may differ), a later checkpoint writes into an earlier wave through the alias (an
    # in-place writer of a hard link), or the alias is left on an older wave.
    from wantrain.fault_tolerance.checkpoint import wave_dir

    rt = _tiny_stage(tmp_path, 7.0)
    path = rt._ckpt_path()
    at_commit = {}
    for step in (1, 3, 5, 7):
        rt.losses[(step, 0)] = float(step) / 4
        with torch.no_grad():
            for p in rt.chunks[0].parameters():
                p.add_(1.0)
        _commit(rt, step)
        with open(path, "rb") as f:
            at_commit[step] = f.read()
        with open(path[:-3] + ".step.json") as f:
            assert json.load(f) == {"step": step}
    assert len(set(at_commit.values())) == 4, "two checkpoints wrote the same bytes"
    assert _waves(tmp_path) == [3, 5, 7]
    for step in (3, 5, 7):
        with open(wave_dir(str(tmp_path), "t", 0, 0, step) + "state.pt", "rb") as f:
            assert f.read() == at_commit[step], step
    assert os.path.samefile(path, wave_dir(str(tmp_path), "t", 0, 0, 7) + "state.pt")


def test_resume_drops_waves_of_the_abandoned_timeline(tmp_path, monkeypatch):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: a resume from step 3 keeps the slot's waves 4 and 7 of the abandoned
    # timeline (a new checkpoint of a held step would commit and announce nothing), keeps the
    # first step after it (off by one), drops them only after the first step ran, drops the
    # resume's own wave, does not log them, or a refused resume drops anything.
    from wantrain.fault_tolerance.checkpoint import link_legacy, wave_dir

    src = _tiny_stage(tmp_path, 3.0)
    for step in (3, 4, 7):
        src.losses[(step, 0)] = float(step)
        _commit(src, step)
    # RESTORE of the common step 3: the alias goes back to wave 3, and the relaunch names it.
    link_legacy(wave_dir(str(tmp_path), "t", 0, 0, 3), str(tmp_path), "t", 0, 0)
    refused = _tiny_stage(tmp_path, 7.0, steps=3, resume=True, resume_step=3)
    with pytest.raises(RuntimeError, match="steps=3"):
        _run_without_compute(refused, monkeypatch)
    assert _waves(tmp_path) == [3, 4, 7], "a refused resume dropped waves"
    held = []
    rt = _tiny_stage(tmp_path, 7.0, steps=6, ckpt_every=2, resume=True, resume_step=3)
    lines, started = _run_without_compute(
        rt, monkeypatch, at_first_step=lambda: held.append(_waves(tmp_path))
    )
    assert started == [4, 5]
    assert held == [[3]], held
    assert "WT_DROPPED_WAVES rank=0 peer=0 after=3 steps=[4, 7]" in lines, lines
    assert [ln for ln in lines if ln.startswith("WT_WAVE")] == ["WT_WAVE 5"]
    assert _waves(tmp_path) == [3, 5]
    assert torch.load(rt._ckpt_path(), map_location="cpu", weights_only=True)["losses"] == {
        "3:0": 3.0
    }


def test_fresh_launch_drops_every_old_wave(tmp_path, monkeypatch):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: a launch that does not resume keeps the slot's waves of an earlier run
    # under its tag (an old wave of a step the new run checkpoints would be held, so it commits
    # and announces nothing), keeps the wave of step 0 (off by one), drops them only after the
    # first step, does not log them, or drops another slot's or another tag's waves.
    old = _tiny_stage(tmp_path, 3.0)
    _commit(old, 0)
    _commit(old, 3)
    other_slot = _tiny_stage(tmp_path, 3.0)
    other_slot.peer = 1
    _commit(other_slot, 1)
    other_tag = _tiny_stage(tmp_path, 3.0)
    other_tag.spec.tag = "u"
    _commit(other_tag, 1)
    with open(old._ckpt_path(), "rb") as f:
        old_bytes = f.read()
    held = []
    rt = _tiny_stage(tmp_path, 7.0, steps=2, ckpt_every=2)
    lines, started = _run_without_compute(
        rt, monkeypatch, at_first_step=lambda: held.append(_waves(tmp_path))
    )
    assert started == [0, 1]
    assert held == [[]], held
    assert "WT_DROPPED_WAVES rank=0 peer=0 after=-1 steps=[0, 3]" in lines, lines
    assert [ln for ln in lines if ln.startswith("WT_WAVE")] == ["WT_WAVE 1"]
    assert _waves(tmp_path) == [1]
    with open(rt._ckpt_path(), "rb") as f:
        assert f.read() != old_bytes
    assert _waves(tmp_path, peer=1) == [1] and _waves(tmp_path, tag="u") == [1]


def test_launch_sweeps_the_temps_of_a_killed_writer(tmp_path, monkeypatch):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: a stage killed (SIGTERM/SIGKILL, no cleanup runs) inside commit_wave, inside
    # a wave drop or between a staged state.pt and its commit leaves its private temp directory, a whole state.pt, in the slot, and
    # the next launch of the slot keeps it, so every kill during a checkpoint leaks one slot
    # state of disk without bound; or the sweep reaches another slot's temps.
    from wantrain.fault_tolerance import checkpoint as ck

    old = _tiny_stage(tmp_path, 3.0)
    _commit(old, 2)
    _commit(old, 5)
    other = _tiny_stage(tmp_path, 3.0)
    other.peer = 1

    def killed(tmp):
        torch.save({"partial": True}, tmp)
        raise KeyboardInterrupt  # stands for the kill; its cleanup is taken out below

    with monkeypatch.context() as m:
        # A killed process runs no cleanup: the real commit_wave and drop_waves_after leave
        # their real temp directories behind.
        m.setattr(ck.shutil, "rmtree", lambda *a, **k: None)
        for stage in (old, other):
            with pytest.raises(KeyboardInterrupt):
                ck.commit_wave(str(tmp_path), "t", 0, stage.peer, 9, killed)
        ck.drop_waves_after(str(tmp_path), "t", 0, 0, 2)
    slot = os.path.dirname(os.path.normpath(ck.wave_dir(str(tmp_path), "t", 0, 0, 0)))
    # A kill between _checkpoint's serialization and its background commit.
    torch.save({"partial": True}, os.path.join(slot, ".staged-step-00000009.pt"))
    other_slot = os.path.dirname(os.path.normpath(ck.wave_dir(str(tmp_path), "t", 0, 1, 0)))
    left = sorted(n for n in os.listdir(slot) if n.startswith("."))
    assert [n[1:].split("-")[0] for n in left] == ["drop", "staged", "tmp"], left
    assert len(os.listdir(other_slot)) == 1
    # RESTORE of step 2: the alias goes back to wave 2.
    ck.link_legacy(ck.wave_dir(str(tmp_path), "t", 0, 0, 2), str(tmp_path), "t", 0, 0)
    rt = _tiny_stage(tmp_path, 7.0, steps=4, ckpt_every=2, resume=True)
    lines, started = _run_without_compute(rt, monkeypatch)
    assert started == [3]
    assert sorted(os.listdir(slot)) == ["step-00000002", "step-00000003"], os.listdir(slot)
    assert f"WT_DROPPED_WAVE_TEMPS rank=0 peer=0 names={left}" in lines, lines
    assert len(os.listdir(other_slot)) == 1, "the sweep reached another slot"


def test_checkpoint_hashes_off_the_step(tmp_path, monkeypatch):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: _checkpoint computes wave.json's sha256 (or fsyncs, renames, links the
    # alias or prunes) on the step loop instead of on its background commit (HANDOUT-M2 "I/O
    # per wave": hashing "off the step"; THREAT_MODEL §3: "~0 on the step"), serializes the
    # payload on the background thread (a later step's update would leak into the wave), or
    # announces WT_WAVE before the commit finished.
    from wantrain.fault_tolerance import checkpoint as ck

    caller = threading.get_ident()
    release = threading.Event()
    hashed_on = []
    real_hash = ck.sha256_file

    def gated_hash(path, *a, **kw):
        hashed_on.append(threading.get_ident())
        if threading.get_ident() != caller:
            assert release.wait(30), "the test never released the hash"
        return real_hash(path, *a, **kw)

    lines = []
    monkeypatch.setattr(ck, "sha256_file", gated_hash)
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(write=lines.append, flush=lambda: None))
    rt = _tiny_stage(tmp_path, 7.0)
    want = {k: v.clone() for k, v in rt.chunks[0].state_dict().items()}
    rt._checkpoint(5)
    # _checkpoint returned while the hash is held: nothing committed or announced yet.
    assert caller not in hashed_on, hashed_on
    assert _waves(tmp_path) == [] and not any("WT_WAVE" in ln for ln in lines)
    with torch.no_grad():
        for p in rt.chunks[0].parameters():
            p.add_(100.0)  # the next step's update, while the commit is in flight
    release.set()
    rt._await_wave()
    monkeypatch.undo()
    assert hashed_on and caller not in hashed_on, hashed_on
    assert "".join(lines).splitlines() == ["WT_WAVE 5"]
    assert _waves(tmp_path) == [5]
    state = torch.load(rt._ckpt_path(), map_location="cpu", weights_only=True)
    assert all(torch.equal(state["chunks"][0][k], want[k]) for k in want)
    slot = os.path.dirname(os.path.normpath(ck.wave_dir(str(tmp_path), "t", 0, 0, 0)))
    assert sorted(os.listdir(slot)) == ["step-00000005"]


def test_a_failed_wave_commit_fails_the_run(tmp_path, monkeypatch):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: an error of the background commit (ENOSPC, the wave cap) is swallowed by
    # its thread, so the run ends "complete" with an uncommitted, unannounced checkpoint, or
    # its staged state.pt is left in the slot.
    import errno

    from wantrain.runtime import stage as stage_mod

    def full(out_dir, tag, rank, peer, step, save_fn, *a, **kw):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(stage_mod, "commit_wave", full)
    rt = _tiny_stage(tmp_path, 7.0, steps=3, ckpt_every=2)
    finished = []
    with pytest.raises(OSError, match="No space left"):
        _run_without_compute(rt, monkeypatch, at_finish=finished.append)
    assert finished == [], "the run reached _finish over a failed checkpoint"
    assert _waves(tmp_path) == []
    slot = os.path.dirname(os.path.normpath(stage_mod.wave_dir(str(tmp_path), "t", 0, 0, 0)))
    assert os.listdir(slot) == [], os.listdir(slot)


def test_a_failing_run_still_commits_its_last_checkpoint(tmp_path, monkeypatch):
    # covers: RUNTIME-I17
    # WOULD-FAIL-IF: a run that fails (a peer lost) while the background commit of its last
    # checkpoint is still in flight raises without waiting for it, so the process exit stops
    # the commit's daemon thread and a step checkpointed before the failure is never a
    # committed, announced wave (with the commit on the step, it was); or the commit's wait
    # replaces the run's own error.
    from wantrain.fault_tolerance import checkpoint as ck

    release = threading.Event()
    real_hash = ck.sha256_file

    def gated_hash(path, *a, **kw):
        assert release.wait(30), "the test never released the hash"
        return real_hash(path, *a, **kw)

    class ReleasedOnWait:
        """The commit in flight; its hash is held until run() waits for it."""

        def __init__(self, inner):
            self.inner = inner

        def get(self):
            release.set()
            return self.inner.get()

    pending = []

    def fail_at_2(step):
        if step == 2:
            # The commit of step 1 is still held: only a wait for it releases it.
            assert _waves(tmp_path) == []
            pending.append(rt._wave_commit)
            rt._wave_commit = ReleasedOnWait(rt._wave_commit)
            raise ConnectionError("reader: peer connection lost")

    monkeypatch.setattr(ck, "sha256_file", gated_hash)
    rt = _tiny_stage(tmp_path, 7.0, steps=6, ckpt_every=2)
    lines = []
    try:
        with pytest.raises(ConnectionError, match="peer connection lost"):
            _run_without_compute(
                rt, monkeypatch, at_step=fail_at_2, at_finish=lambda ls: lines.extend(ls)
            )
        committed = _waves(tmp_path)
    finally:
        release.set()
        for p in pending:
            with contextlib.suppress(BaseException):
                p.get()
    assert committed == [1], committed
    assert lines == [], "a failed run reached _finish"
    assert ck.read_sidecar(ck.sidecar_of(rt._ckpt_path())) == 1


# ------------------------------------------------- M2b JobSpec fields (RUNTIME-I5, ADR-0006)

M2B_DEFAULTS = dict(
    wire="fp32",
    shard="zero1",
    emission_order=None,
    piece_bounds=None,
    pacing="link",
    peer_egress_mbps=None,
    peer_ingress_mbps=None,
    sites=None,
    uplink_mbps=None,
)


def _m2b(**kw):
    return JobSpec(model=make_config("toy"), **kw)


def _shared(widths, **kw):
    sites = [[f"s{p % 2}" for p in range(w)] for w in widths]
    return dict(
        pacing="shared",
        peer_egress_mbps=[[100.0] * w for w in widths],
        peer_ingress_mbps=[[80.0] * w for w in widths],
        sites=sites,
        uplink_mbps={"s0->s1": 50.0, "s1->s0": 40.0},
        **kw,
    )


def test_spec_m2b_fields_round_trip_and_default():
    # covers: RUNTIME-I5
    # WOULD-FAIL-IF: a new field is lost or changed by to_json/from_json, a spec JSON written
    # before M2b (without the fields) does not load with the defaults, or finalize does not fill
    # emission_order with "store" per stage.
    old = json.loads(_m2b(stages=2).to_json())
    for k in M2B_DEFAULTS:
        old.pop(k)
    loaded = JobSpec.from_json(json.dumps(old))
    assert {k: getattr(loaded, k) for k in M2B_DEFAULTS} == M2B_DEFAULTS
    assert loaded.finalize().emission_order == ["store", "store"]
    spec = _m2b(
        stages=2,
        stage_widths=[1, 2],
        stream_grads=True,
        wire="native",
        shard="pieces",
        emission_order=["forward", "store"],
        piece_bounds=[None, {"0.ln1.weight": [0, 0, 128]}],
        **_shared([1, 2]),
    ).finalize()
    back = JobSpec.from_json(spec.to_json())
    for k in M2B_DEFAULTS:
        assert getattr(back, k) == getattr(spec, k), k
    assert back.uplink_mbps == {"s0->s1": 50.0, "s1->s0": 40.0}
    assert back.piece_bounds == [None, {"0.ln1.weight": [0, 0, 128]}]
    assert back.finalize().to_json() == spec.to_json()


def test_spec_m2b_budget_files_refuse_a_tag_that_is_not_one_name(tmp_path):
    # covers: RUNTIME-I5
    # WOULD-FAIL-IF: the local launcher's fresh-budget step (it removes out_dir/budgets/<tag>
    # before creating it) accepts a tag of "", ".", ".." or a path, aiming the removal at
    # out_dir, its parent or another run's budgets.
    from wantrain.launch.local import make_budget_files

    keep = tmp_path / "keep.txt"
    keep.write_text("x")
    (tmp_path / "budgets").mkdir()
    (tmp_path / "budgets" / "other").write_text("another run")
    for tag in ("", ".", "..", "a/b", "../x"):
        spec = _m2b(
            stages=2, stage_widths=[1, 2], out_dir=str(tmp_path), tag=tag, **_shared([1, 2])
        )
        with pytest.raises(ValueError, match="plain tag"):
            make_budget_files(spec.finalize())
    assert keep.read_text() == "x" and (tmp_path / "budgets" / "other").exists()
    spec = _m2b(stages=2, stage_widths=[1, 2], out_dir=str(tmp_path), tag="r1", **_shared([1, 2]))
    made = make_budget_files(spec.finalize())
    assert made and all(os.path.exists(p) for p in made)
    assert keep.exists() and (tmp_path / "budgets" / "other").exists()


def _refused(match, **kw):
    with pytest.raises(ValueError, match=match):
        _m2b(**kw).finalize()


def test_spec_m2b_finalize_refuses_with_the_fields_named():
    # covers: RUNTIME-I5
    # WOULD-FAIL-IF: finalize accepts {wire fp32, shard pieces}, wire native with a wide stage
    # and stream_grads False (BA-6), an unknown wire/shard/pacing/emission order, piece_bounds
    # piece_layout rejects, shared pacing with a missing budget, site or uplink, pieces with the
    # tied Embed and Head on different stages (BA-42), or pieces over mixed device types in one
    # stage (BA-52); or a refusal does not name the field.
    wide = dict(stages=2, stage_widths=[1, 2])
    _refused("shard='pieces' needs wire='native'", shard="pieces", stream_grads=True, **wide)
    _refused("stream_grads=False", wire="native", **wide)
    _refused("wire=", wire="fp16", **wide)
    _refused("shard=", shard="fsdp", **wide)
    _refused("pacing=", pacing="fast", **wide)
    _refused("emission_order\\[1\\]", emission_order=["store", "backward"], **wide)
    _refused("emission_order must list", emission_order=["store"], **wide)
    native = dict(wire="native", shard="pieces", stream_grads=True, **wide)
    _refused(
        "piece_bounds\\[1\\].*0.qkv.weight",
        piece_bounds=[None, {"0.qkv.weight": [0, 100, 49152]}],
        **native,
    )
    _refused("piece_bounds\\[1\\].*ghost", piece_bounds=[None, {"ghost": [0, 0, 1]}], **native)
    _refused("piece_bounds must list", piece_bounds=[None], **native)
    shared = _shared([1, 2])
    for field in ("peer_egress_mbps", "peer_ingress_mbps", "sites"):
        _refused(field, **{**shared, field: None}, **wide)
    _refused(
        "peer_egress_mbps\\[1\\]\\[1\\]",
        **{**shared, "peer_egress_mbps": [[1.0], [1.0, 0.0]]},
        **wide,
    )
    _refused("uplink_mbps\\['s1->s0'\\]", **{**shared, "uplink_mbps": {"s0->s1": 5.0}}, **wide)
    tied = make_config("toy")
    tied.tie_embeddings, tied.init_scheme = True, "gpt2"
    with pytest.raises(ValueError, match="tied parameter layers.0.tok.weight"):
        JobSpec(model=tied, **native).finalize()
    # CONTROL-I25: a map mixing device types is refused for pieces and zero1 alike,
    # before the per-stage pieces class check.
    for kw in (native, wide):
        with pytest.raises(ValueError, match="^MIXED-FLEET: the device map spans 2 device classes"):
            _m2b(devices=[["cpu"], ["cpu", "mps"]], **kw).finalize()
    _m2b(devices=[["cpu"], ["cpu", "cpu:0"]], **wide).finalize()


def test_spec_m2b_fields_are_neither_pins_nor_recipe():
    # covers: RUNTIME-I5
    # WOULD-FAIL-IF: one of the M2b fields enters pins() (a resume that only changes the wire,
    # the shard or the pacing would be refused) or recipe_digest (the oracle cache would split
    # per mechanism, BA-36, BA-53).
    from wantrain.launch.local import recipe_digest

    base = _m2b(stages=2, stage_widths=[1, 2], stream_grads=True).finalize()
    other = _m2b(
        stages=2,
        stage_widths=[1, 2],
        stream_grads=True,
        wire="native",
        shard="pieces",
        emission_order=["forward", "forward"],
        piece_bounds=[None, None],
        **_shared([1, 2]),
    ).finalize()
    assert other.pins() == base.pins()
    assert recipe_digest(other) == recipe_digest(base)
    assert "zero1_widths" in base.pins()  # pins() itself is unchanged
    # control: a recipe field does move the digest
    assert recipe_digest(_m2b(stages=2, lr=2e-3).finalize()) != recipe_digest(base)


def test_spec_m2b_planner_contract():
    # covers: RUNTIME-I5
    # WOULD-FAIL-IF: finalize refuses a combination the frozen I-7 contract lists as planner
    # output (pieces with native and stream_grads, native with stream_grads, piece_bounds None
    # or keyed by exchange_units names on quanta, one device type per pieces stage, a tied Embed
    # and Head on one stage), or the bounds' names differ from exchange_units' names.
    from wantrain.runtime import pieces

    combos = []
    for widths in ([1, 2], [2, 1], [1, 3, 1], [2, 2], [1, 4]):
        n = len(widths)
        for eo in (["store"] * n, ["forward"] * n):
            combos.append(
                dict(
                    stages=n,
                    stage_widths=widths,
                    wire="native",
                    shard="pieces",
                    stream_grads=True,
                    emission_order=eo,
                )
            )
            combos.append(
                dict(
                    stages=n,
                    stage_widths=widths,
                    wire="native",
                    stream_grads=True,
                    emission_order=eo,
                )
            )
    combos.append(dict(stages=2, wire="native"))  # every width 1: native needs no streaming
    combos.append(
        dict(
            stages=2,
            stage_widths=[1, 2],
            wire="native",
            shard="pieces",
            stream_grads=True,
            devices=[["cpu"], ["cpu:0", "cpu:1"]],
        )
    )
    tied = make_config("toy")
    tied.tie_embeddings, tied.init_scheme = True, "gpt2"
    for c in combos:
        spec = _m2b(**c).finalize()
        assert JobSpec.from_json(spec.to_json()).finalize().to_json() == spec.to_json()
    one = JobSpec(
        model=tied, stages=1, stage_widths=[2], wire="native", shard="pieces", stream_grads=True
    ).finalize()
    assert one.shard == "pieces"
    # Explicit bounds keyed by exchange_units' names, on quanta of each shape.
    spec = _m2b(
        stages=2, stage_widths=[1, 3], wire="native", shard="pieces", stream_grads=True
    ).finalize()
    shapes = spec.stage_param_shapes()[1]
    from wantrain.model.gpt import build_layers

    chunk = torch.nn.ModuleList(
        list(build_layers(spec.model, spec.init_seed))[slice(*spec.partition[1])]
    )
    assert sorted(n for u in pieces.exchange_units(chunk) for n, _ in u) == sorted(shapes)
    bounds = {}
    for name, shape in shapes.items():
        q, numel = pieces.quantum(shape), int(torch.Size(shape).numel())
        bounds[name] = [0, min(numel, q), min(numel, 2 * q), numel]
    spec.piece_bounds = [None, bounds]
    assert spec.finalize().piece_bounds[1] == bounds


# ------------------------------------------------- exchange primitives (RUNTIME-I10, I13)


def test_need_first_pops_the_most_needed_item_first():
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: the queue serves FIFO, sorts gradients before gathers of the same unit,
    # orders by step before the forward unit, lets a LOSS or the shutdown sentinel (key None)
    # overtake an exchange item, or reorders keyless items among themselves.
    from wantrain.runtime.exchange import GATHER, GRAD, NeedFirstQueue, NeedKey

    q = NeedFirstQueue()
    items = [
        (NeedKey(3, GRAD, 0, 2, 0), "g3"),
        (None, "loss"),
        (NeedKey(1, GRAD, 1, 0, 1), "g1s1"),
        (NeedKey(1, GATHER, 1, 0, 2), "p1"),
        (NeedKey(0, GRAD, 1, 5, 3), "g0mb5"),
        (None, None),
        (NeedKey(0, GRAD, 1, 4, 4), "g0mb4"),
        (NeedKey(1, GRAD, 0, 7, 5), "g1s0"),
    ]
    for k, v in items:
        q.push(k, v)
    assert q.qsize() == len(items)
    got = [q.pop(timeout=1) for _ in items]
    assert got == ["g0mb4", "g0mb5", "p1", "g1s0", "g1s1", "g3", "loss", None]
    with pytest.raises(queue_mod.Empty):
        q.pop(timeout=0.01)
    fifo = NeedFirstQueue(fifo=True)
    for k, v in items:
        fifo.push(k, v)
    assert [fifo.pop() for _ in items] == [v for _, v in items]


def test_need_first_is_thread_safe_under_concurrent_pushes():
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: concurrent pushes lose or duplicate an item, or a pop blocked on an empty
    # queue misses a push (no wakeup), or the popped order is not the key order.
    from wantrain.runtime.exchange import GATHER, GRAD, NeedFirstQueue, NeedKey

    q = NeedFirstQueue()
    rng = random.Random(5)
    plans = [
        [
            (rng.randrange(8), rng.choice((GATHER, GRAD)), rng.randrange(3), rng.randrange(8))
            for _ in range(300)
        ]
        for _ in range(4)
    ]
    out, start = [], threading.Barrier(5)

    def pusher(t, plan):
        start.wait()
        for i, (u, kind, s, mb) in enumerate(plan):
            q.push(NeedKey(u, kind, s, mb, t * 1000 + i), (u, kind, s, mb, t, i))

    def popper():
        start.wait()
        for _ in range(sum(len(p) for p in plans)):
            out.append(q.pop(timeout=10))

    ts = [threading.Thread(target=pusher, args=(t, p)) for t, p in enumerate(plans)]
    pop = threading.Thread(target=popper)
    for t in [*ts, pop]:
        t.start()
    for t in [*ts, pop]:
        t.join(timeout=30)
    assert len(out) == 1200 and len(set(out)) == 1200
    q2 = NeedFirstQueue()
    for t, plan in enumerate(plans):
        for i, (u, kind, s, mb) in enumerate(plan):
            q2.push(NeedKey(u, kind, s, mb, t * 1000 + i), (u, kind, s, mb, t, i))
    assert sorted(out) == sorted(q2.pop() for _ in range(1200))
    # a blocked pop wakes on a push from another thread
    late = NeedFirstQueue()
    threading.Timer(0.05, lambda: late.push(None, "x")).start()
    assert late.pop(timeout=5) == "x"


def _unit_body():
    g = torch.Generator().manual_seed(3)
    return [
        torch.randn(1000, generator=g).to(torch.bfloat16),
        torch.empty(0, dtype=torch.float32),
        torch.randn(37, generator=g),
        torch.randn(2050, generator=g).to(torch.float16),
    ]


def test_chunk_object_cuts_a_unit_into_bounded_self_contained_chunks():
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: a chunk carries more than max_bytes of tensor data, the envelope lacks the
    # unit, offset or length, chunks overlap or leave a gap, a chunk copies instead of viewing,
    # or reassembly changes one bit or one dtype.
    from wantrain.runtime import exchange

    body = _unit_body()
    for max_bytes in (8, 100, 1000, 4096, 1 << 20):
        parts = exchange.chunk_object(7, body, max_bytes)
        pos = 0
        for env, view in parts:
            assert env["u"] == 7 and env["off"] == pos and env["n"] == [1000, 0, 37, 2050]
            assert sum(t.numel() * t.element_size() for t in view) <= max_bytes
            assert sum(t.numel() for t in view) == env["len"]
            pos += env["len"]
        assert pos == 3087
        rng = random.Random(max_bytes)
        shuffled = list(parts)
        rng.shuffle(shuffled)
        back = exchange.reassemble(shuffled)
        assert [t.dtype for t in back] == [t.dtype for t in body]
        assert all(_same_bits(a, b) if a.numel() else b.numel() == 0 for a, b in zip(back, body))
        assert exchange.reassemble(shuffled[:-1]) is None or len(parts) == 1
    one = exchange.chunk_object(0, body)
    assert len(one) == 1 and one[0][1][0].data_ptr() == body[0].data_ptr()  # views, no copy
    empty = exchange.chunk_object(2, [torch.empty(0)])
    assert len(empty) == 1 and empty[0][0]["len"] == 0
    assert exchange.reassemble(empty)[0].numel() == 0


def test_chunk_reassembly_never_mixes_two_units_of_one_microbatch():
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: chunks of two units of the same (step, mb, sender) share a reassembly key
    # (BA-13), or overlapping chunks are accepted instead of refused.
    from wantrain.runtime import exchange

    a = [torch.arange(3000, dtype=torch.float32)]
    b = [torch.arange(3000, dtype=torch.float32) * -1]
    groups: dict = {}
    for unit, body in ((0, a), (1, b)):
        for env, view in exchange.chunk_object(unit, body, 4000):
            groups.setdefault(("GRADS", 0, 3, env["u"], 1), []).append((env, view))
    assert torch.equal(exchange.reassemble(groups[("GRADS", 0, 3, 0, 1)])[0], a[0])
    assert torch.equal(exchange.reassemble(groups[("GRADS", 0, 3, 1, 1)])[0], b[0])
    parts = exchange.chunk_object(0, a, 4000)
    env, view = parts[1]
    bad = dict(env, off=env["off"] - 1)
    with pytest.raises(exchange.FrameError, match="shape"):
        exchange.reassemble([parts[0], (bad, view), *parts[2:]])


def _expect(**kw):
    table = {
        5: [(torch.bfloat16, 1000), (torch.float32, 0), (torch.float32, 37), (torch.float16, 2050)]
    }
    return dict(units=table, M=8, step=4, seen=set(), **kw)


def _frame_env(env, **kw):
    return dict(env, mb=kw.pop("mb", 3), step=kw.pop("step", 4), sender=kw.pop("sender", 1), **kw)


def test_frame_check_accepts_every_chunk_of_a_valid_unit():
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: a valid chunk (current or next step, any offset of the unit) is refused, or
    # the validator runs a tensor op (it may only read dtype and length).
    from torch.utils._python_dispatch import TorchDispatchMode

    from wantrain.runtime import exchange

    class NoOps(TorchDispatchMode):
        calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            NoOps.calls += 1
            return func(*args, **(kwargs or {}))

    exp = _expect()
    parts = exchange.chunk_object(5, _unit_body(), 1000)
    with NoOps():
        for step in (4, 5):
            for env, view in parts:
                exchange.check_frame("GRADS", _frame_env(env, step=step), list(view), exp)
    assert NoOps.calls == 0


@pytest.mark.parametrize(
    "cls",
    ["unit", "mb", "step", "dtype", "shape", "duplicate", "envelope"],
)
def test_frame_check_refuses_each_malformed_class(cls):
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: a GRADS/PARAMS chunk with an unknown unit, a microbatch outside [0, M), a
    # step neither current nor next, a dtype not in the plan's table (fp32 where bf16 is
    # planned), a length that is not the piece, a repeated (step, mb, unit, sender) or a broken
    # envelope is accepted, or the refusal does not name its class.
    from wantrain.runtime import exchange

    body = _unit_body()
    (env, view), *_ = exchange.chunk_object(5, body, 1 << 20)
    exp = _expect()
    fenv, tensors = _frame_env(env), list(view)
    if cls == "unit":
        fenv["u"] = 6
    elif cls == "mb":
        fenv["mb"] = 8
    elif cls == "step":
        fenv["step"] = 6
    elif cls == "dtype":
        tensors[0] = tensors[0].float()
    elif cls == "shape":
        tensors[2] = tensors[2][:36]
    elif cls == "duplicate":
        exchange.check_frame("PARAMS", dict(fenv), list(view), exp)
    else:
        del fenv["off"]
    with pytest.raises(exchange.FrameError) as err:
        exchange.check_frame("PARAMS", fenv, tensors, exp)
    assert err.value.cls == cls and str(err.value).startswith(cls)
    for bad in ({**_frame_env(env), "step": 3}, {**_frame_env(env), "mb": -1}):
        with pytest.raises(exchange.FrameError):
            exchange.check_frame("GRADS", bad, list(view), _expect())


def test_frame_check_refuses_a_chunk_that_lists_one_tensor_twice():
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: a chunk whose envelope lists a tensor index twice (so the slice lengths and
    # their sum still match while one tensor takes another's elements) passes check_frame, or
    # reassemble returns a tensor of another length than the plan's (a size-1 tensor would
    # broadcast silently into the fold).
    from wantrain.runtime import exchange

    # Tensors 2 (fp32, 37) and the first 37 elements of 3 (fp16) as two fp32 slices of tensor 2:
    # every length and dtype check of the envelope holds; only the index list lies.
    env = {"u": 5, "off": 1000, "len": 74, "n": [1000, 0, 37, 2050], "t": [2, 2]}
    forged = [torch.zeros(37), torch.zeros(37)]
    with pytest.raises(exchange.FrameError) as err:
        exchange.check_frame("GRADS", _frame_env(env), forged, _expect())
    assert err.value.cls == "shape"
    # control: the honest chunk over the same elements passes
    honest = {**env, "t": [1, 2, 3]}
    exchange.check_frame(
        "GRADS",
        _frame_env(honest),
        [torch.zeros(0), torch.zeros(37), torch.zeros(37, dtype=torch.float16)],
        _expect(),
    )
    # reassemble refuses a tensor that does not come back at its planned length
    body = [torch.arange(6.0), torch.arange(4.0)]
    bad = [({"u": 0, "off": 0, "len": 10, "n": [6, 4], "t": [0, 1]}, [body[0][:3], body[1]])]
    bad[0][1][1] = torch.arange(7.0)
    with pytest.raises(exchange.FrameError, match="shape"):
        exchange.reassemble(bad)
    assert [t.numel() for t in exchange.reassemble(exchange.chunk_object(0, body, 16))] == [6, 4]


def test_copy_engine_runs_every_copy_off_the_callers_thread():
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: submit, d2h or h2d runs its copy on the caller's (compute) thread, a copy
    # changes a bit, or an error inside the engine is lost instead of raised by result().
    from wantrain.runtime.exchange import CopyEngine

    eng = CopyEngine("cpu")
    try:
        me = threading.get_ident()
        seen = [eng.submit(threading.get_ident).result(timeout=5) for _ in range(5)]
        assert set(seen) == {eng.thread_ident} and me != eng.thread_ident
        t = torch.randn(64).to(torch.bfloat16)
        assert _same_bits(eng.d2h([t]).result(timeout=5)[0], t)
        assert _same_bits(eng.h2d([t]).result(timeout=5)[0], t)
        assert eng.copies == 2

        def boom():
            raise OSError("copy failed")

        with pytest.raises(OSError, match="copy failed"):
            eng.submit(boom).result(timeout=5)
    finally:
        eng.close()


def test_holds_count_per_step_maxima_and_never_go_negative():
    # covers: RUNTIME-I13
    # WOULD-FAIL-IF: a per-step maximum is lost or carried over wrongly, `host` sums maxima of
    # different instants instead of one instant, the WT_HOLD line misses a term, concurrent
    # updates lose bytes, a release below zero passes silently, or counting draws from the RNG.
    from wantrain.runtime.exchange import HoldCounters

    rng = torch.get_rng_state()
    h = HoldCounters()
    h.add("host_grads", 100)
    h.add("gathers_out", 30)
    h.sub("host_grads", 100)
    h.add("gathers_in", 50)  # host now 80; its maximum was 130
    h.add("device_own", 7)
    assert h.end_step(0) == {
        "host_grads": 100,
        "gathers_out": 30,
        "gathers_in": 50,
        "device_own": 7,
        "host": 130,
    }
    assert (
        h.line(0) == "WT_HOLD 0 host_grads=100 gathers_out=30 gathers_in=50 device_own=7 host=130"
    )
    h.sub("gathers_out", 30)
    assert h.end_step(1) == {
        "host_grads": 0,
        "gathers_out": 30,
        "gathers_in": 50,
        "device_own": 7,
        "host": 80,
    }
    with pytest.raises(RuntimeError, match="below zero"):
        h.sub("host_grads", 1)

    def worker():
        for _ in range(2000):
            h.add("host_grads", 3)
            h.sub("host_grads", 3)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert h.end_step(2)["host_grads"] <= 12 and h._cur["host_grads"] == 0
    assert torch.equal(rng, torch.get_rng_state())


# ------------------------------------------------- logical checkpoints (RUNTIME-I12)


def _logical(seed=4, step=6, M=2):
    """A logical stage state: odd shapes, one tensor with fewer quanta than peers."""
    g = torch.Generator().manual_seed(seed)
    shapes = {
        "0.tok.weight": (40, 128),
        "0.pos.weight": (16, 96),
        "1.ln.weight": (3,),
        "1.fc.bias": (300,),
    }
    tensors = {}
    for n, s in shapes.items():
        tensors[n] = {
            "master": torch.randn(s, generator=g),
            "exp_avg": torch.randn(s, generator=g),
            "exp_avg_sq": torch.rand(s, generator=g),
            "step": torch.tensor(float(step + 1)),
        }
    losses = {f"{s}:{m}": float(s * 10 + m) / 7 for s in range(step + 1) for m in range(M)}
    return shapes, {
        "step": step,
        "tensors": tensors,
        "losses": losses,
        "rng": torch.arange(16, dtype=torch.uint8),
        "nontrainable": {"2.buf": torch.ones(4)},
        "hparams": {"lr": 1e-3, "betas": (0.9, 0.95)},
    }


def _payloads(logical, shapes, w, bounds=None, losses_split=True):
    from wantrain.runtime import logical_ckpt as lc
    from wantrain.runtime import pieces

    layout = pieces.piece_layout(shapes, w, bounds)
    out = []
    keys = sorted(logical["losses"])
    for k in range(w):
        mine = lc.owned_pieces(logical, layout, k)
        payload = {
            "step": logical["step"],
            "pieces": mine,
            "hparams": dict(logical["hparams"]),
            "rng": logical["rng"].clone(),
            "losses": {x: logical["losses"][x] for x in (keys[k::w] if losses_split else keys)},
            "nontrainable": {n: t.clone() for n, t in logical["nontrainable"].items()},
            "pins": {},
        }
        payload["pieces_manifest"] = lc.build_manifest(logical["step"], w, k, layout, mine, shapes)
        out.append((f"r1p{k}", payload))
    return layout, out


def _same_logical(a, b):
    assert list(a["tensors"]) == list(b["tensors"])
    for n in a["tensors"]:
        for f in ("master", "exp_avg", "exp_avg_sq", "step"):
            assert a["tensors"][n][f].shape == b["tensors"][n][f].shape, (n, f)
            assert _same_bits(a["tensors"][n][f].reshape(-1), b["tensors"][n][f].reshape(-1)), (
                n,
                f,
            )


def test_logical_ckpt_reshards_any_width_and_round_trips_through_zero1():
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: assembling the pieces of a width-w wave and cutting them at another width
    # (4 -> 3 -> 2 -> zero1 -> pieces), or reading a zero1 wave as names[k::w], changes one bit
    # of a master, m, v or step counter, loses a loss, or a reader that places one boundary one
    # element off is not caught by the bitwise comparison.
    from wantrain.runtime import logical_ckpt as lc

    shapes, logical = _logical()
    cur = logical
    for w in (4, 3, 2):
        layout, payloads = _payloads(cur, shapes, w)
        back = lc.assemble(payloads, microbatches=2)
        _same_logical(back, logical)
        assert back["losses"] == logical["losses"] and back["width"] == w
        cur = back
    names = list(shapes)
    zero1 = []
    for k in range(2):
        shard = lc.as_zero1(cur, names, 2, k)
        sd = {n: cur["tensors"][n]["master"].clone() for n in names}
        sd["2.buf"] = torch.ones(4)
        zero1.append(
            {
                "step": cur["step"],
                "chunks": {1: sd},
                "opt": {
                    "state": {j: st for j, (_, _, st) in enumerate(shard)},
                    "param_groups": [{**cur["hparams"], "params": list(range(len(shard)))}],
                },
                "rng": cur["rng"].clone(),
                "losses": dict(cur["losses"]),
            }
        )
        assert [n for n, _, _ in shard] == names[k::2]
    via = lc.from_zero1(zero1, names, 2, microbatches=2)
    _same_logical(via, logical)
    assert via["nontrainable"].keys() == {"2.buf"}
    _, payloads = _payloads(via, shapes, 3)
    _same_logical(lc.assemble(payloads), logical)
    # control: a reader that places piece 1 one element late differs
    _, payloads = _payloads(logical, shapes, 2)
    name, b = "1.fc.bias", payloads[0][1]["pieces_manifest"]["layout"]["1.fc.bias"]
    shifted = torch.zeros(300)
    shifted[: b[1]] = payloads[0][1]["pieces"][name]["master"]
    shifted[b[1] + 1 :] = payloads[1][1]["pieces"][name]["master"][:-1]
    assert not _same_bits(shifted, logical["tensors"][name]["master"])


def test_logical_ckpt_select_wave_reads_exactly_the_waves_slots():
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: select_wave takes the width or layout from anywhere but slot p0, reads a
    # slot whose manifest names another width, layout, step or peer index, accepts a missing slot,
    # accepts an extra index (>= w_old, same step and width), or reads an older wave's slot.
    from wantrain.runtime import logical_ckpt as lc

    shapes, logical = _logical()
    layout, payloads = _payloads(logical, shapes, 3)
    mans = {k: p["pieces_manifest"] for k, (_, p) in enumerate(payloads)}
    w_old, lay = lc.select_wave(mans.__getitem__, [0, 1, 2])
    assert w_old == 3 and lay == {n: list(b) for n, b in layout.items()}
    _, older = _payloads(logical, shapes, 4)
    stale = dict(older[3][1]["pieces_manifest"])
    assert (
        lc.select_wave({**mans, 3: stale}.__getitem__, [0, 1, 2, 3])[0] == 3
    )  # older width: ignored
    extra = dict(mans[2], peer=3)
    with pytest.raises(ValueError, match="slot p3: an extra index"):
        lc.select_wave({**mans, 3: extra}.__getitem__, [0, 1, 2, 3])
    with pytest.raises(ValueError, match="slot p2 .* is missing"):
        lc.select_wave(mans.__getitem__, [0, 1])
    with pytest.raises(ValueError, match="missing peer p0"):
        lc.select_wave(mans.__getitem__, [1, 2])
    for field, value in (
        ("width", 4),
        ("step", 5),
        ("layout", dict(older[0][1]["pieces_manifest"]["layout"])),
        ("peer", 2),
    ):
        bad = {**mans, 1: dict(mans[1], **{field: value})}
        with pytest.raises(ValueError, match="slot p1"):
            lc.select_wave(bad.__getitem__, [0, 1, 2])


def _flip(payloads, k, fn):
    out = [(w, dict(p)) for w, p in payloads]
    fn(out[k][1])
    return out


@pytest.mark.parametrize(
    "case",
    [
        "gap",
        "overlap",
        "sha",
        "manifest",
        "manifest_step",
        "peer",
        "two_values",
        "missing_loss",
        "rng",
        "nontrainable",
    ],
)
def test_logical_ckpt_assemble_refuses_before_applying_anything(case):
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: assemble accepts pieces that leave a gap or overlap, a chunk whose sha256
    # mismatches, a pieces payload without its manifest, a manifest step other than the
    # payload's, a missing peer, two values for one (step, mb), a missing (step, mb) on the
    # final stage, or slots whose RNG or non-trainable state differ; or the refusal does not name
    # its reason.
    import copy

    from wantrain.runtime import logical_ckpt as lc

    shapes, logical = _logical()
    _, payloads = _payloads(logical, shapes, 3)
    payloads = copy.deepcopy(payloads)
    p1 = payloads[1][1]
    want = {
        "gap": "gap or overlap",
        "overlap": "gap or overlap",
        "sha": "sha256 mismatch",
        "manifest": "without its manifest",
        "manifest_step": "manifest names step 5",
        "peer": "missing peer p2",
        "two_values": "two values for \\(step, mb\\) 0:0",
        "missing_loss": "no loss for \\(step, mb\\) \\(6, 1\\)",
        "rng": "RNG state differs",
        "nontrainable": "non-trainable state differs",
    }[case]
    if case in ("gap", "overlap"):
        # a consistent manifest whose pieces do not tile [0, numel)
        man = p1["pieces_manifest"]
        b = list(man["layout"]["0.tok.weight"])
        b[2] += 128 if case == "gap" else -128
        lay = dict(man["layout"], **{"0.tok.weight": b})
        t = logical["tensors"]["0.tok.weight"]
        p1["pieces"]["0.tok.weight"] = {
            f: t[f].reshape(-1)[b[1] : b[2]].clone() for f in ("master", "exp_avg", "exp_avg_sq")
        } | {"step": t["step"].clone()}
        p1["pieces_manifest"] = lc.build_manifest(6, 3, 1, lay, p1["pieces"], shapes)
        for _, p in payloads:
            p["pieces_manifest"]["layout"] = lay if p is p1 else p["pieces_manifest"]["layout"]
        with pytest.raises(ValueError, match=want):
            lc.assemble(payloads)
        return
    if case == "sha":
        p1["pieces"]["1.fc.bias"]["master"][0] += 1
    elif case == "manifest":
        del p1["pieces_manifest"]
    elif case == "manifest_step":
        p1["pieces_manifest"]["step"] = 5
    elif case == "peer":
        payloads = payloads[:2]
    elif case == "two_values":
        key = sorted(payloads[0][1]["losses"])[0]
        assert key not in p1["losses"]
        p1["losses"][key] = 99.0
    elif case == "missing_loss":
        for _, p in payloads:
            p["losses"].pop("6:1", None)
    elif case == "rng":
        p1["rng"][0] += 1
    else:
        p1["nontrainable"]["2.buf"][1] = 2.0
    with pytest.raises(ValueError, match=want):
        lc.assemble(payloads, microbatches=2)


def test_logical_ckpt_and_exchange_never_draw_from_the_torch_rng():
    # covers: RUNTIME-I12
    # WOULD-FAIL-IF: a manifest, assembly, reshard, queue, chunk, frame check or copy draws from
    # the global torch generator, which would shift the run's RNG state against the reference.
    from wantrain.runtime import exchange
    from wantrain.runtime import logical_ckpt as lc

    shapes, logical = _logical()
    before = torch.get_rng_state()
    _, payloads = _payloads(logical, shapes, 3)
    back = lc.assemble(payloads, microbatches=2)
    lc.as_zero1(back, list(shapes), 2, 1)
    q = exchange.NeedFirstQueue()
    q.push(None, 1)
    q.pop()
    parts = exchange.chunk_object(5, _unit_body(), 999)
    exp = _expect()
    for env, view in parts:
        exchange.check_frame("GRADS", _frame_env(env), list(view), exp)
    exchange.reassemble(parts)
    eng = exchange.CopyEngine("cpu")
    eng.d2h([torch.ones(3)]).result(timeout=5)
    eng.close()
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(1)  # control
    assert not torch.equal(before, torch.get_rng_state())


# ------------------------------------------------- BA-58: the bare stage entry point


def test_pieces_stage_main_refused(tmp_path, monkeypatch):
    # covers: RUNTIME-I10
    # WOULD-FAIL-IF: `python -m wantrain.worker.stage_main` runs a shard "pieces" spec without
    # --launcher (a one-slot launch that checked no peer's class), refuses it with a launcher, or
    # refuses a zero1 spec with or without the flag.
    from wantrain.worker import stage_main

    reached = []
    monkeypatch.setattr(
        stage_main, "run_stage", lambda spec, r, p: reached.append((spec, r, p)) or 0
    )
    pieces_path = tmp_path / "pieces.json"
    pieces_path.write_text(
        _m2b(
            stages=2, stage_widths=[1, 2], wire="native", shard="pieces", stream_grads=True
        ).to_json()
    )
    zero1_path = tmp_path / "zero1.json"
    zero1_path.write_text(_m2b(stages=2).to_json())
    out = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a))))
    assert stage_main.main(["--spec", str(pieces_path), "--rank", "1", "--peer", "0"]) == 2
    assert reached == [] and any("one-slot launch cannot verify" in x for x in out)
    for launcher in ("local", "coordinator"):
        assert (
            stage_main.main(["--spec", str(pieces_path), "--rank", "1", "--launcher", launcher])
            == 0
        )
    assert stage_main.main(["--spec", str(zero1_path), "--rank", "0"]) == 0
    assert stage_main.main(["--spec", str(zero1_path), "--rank", "0", "--launcher", "local"]) == 0
    assert [r[0] for r in reached] == [str(pieces_path)] * 2 + [str(zero1_path)] * 2
    with pytest.raises(SystemExit):
        stage_main.main(["--spec", str(zero1_path), "--rank", "0", "--launcher", "agent"])


def test_unit_fold_holds_only_the_own_slice_of_an_out_of_order_contribution():
    """An own piece that arrives before its fold prefix is held as its own
    slice (RUNTIME-I13: b_g·P_s/w per held microbatch), never as a view that
    keeps the whole unit gradient (all w pieces) resident; the fold's bits
    are the in-order left fold's."""
    # covers: RUNTIME-I13, RUNTIME-I11
    # WOULD-FAIL-IF: _UnitFold.add held the own piece as a view of the unit
    # gradient (the held storage is w times the piece), or holding changed the
    # fold's order or bits.
    from wantrain.runtime import exchange as xc
    from wantrain.runtime import pieces as pc
    from wantrain.runtime.stage import _UnitFold

    w, n = 4, 32
    grads = [torch.randn(n, generator=torch.Generator().manual_seed(s)) for s in range(3)]
    holds = xc.HoldCounters()
    fold = _UnitFold(3, holds, lambda ts: ts)
    lo, hi = n // w, 2 * n // w  # this peer's piece of each gradient

    assert not fold.add(2, [grads[2][lo:hi]], own=True)
    assert not fold.add(1, [grads[1][lo:hi]], own=True)
    for mb in (1, 2):
        held = fold.hold[mb][0][0]
        assert held.untyped_storage().nbytes() == (hi - lo) * 4
    assert holds._cur["device_own"] == 2 * (hi - lo) * 4
    assert fold.add(0, [grads[0][lo:hi]], own=True)
    want = pc.fold([g[lo:hi] for g in grads])
    assert torch.equal(fold.acc[0].view(torch.int32), want.view(torch.int32))
    assert holds._cur["device_own"] == 0


# ------------------------------------------------------------- M4: stage-m4 (wave 1)
# Self-contained: these tests reach only stage-m4's own files and code already on the tree
# (HANDOUT-M4, Wave order). Behaviour on real HF units and MoE blocks is stage-wire's.

_GOLDEN_DENSE_PINS = (
    '{"model": {"vocab_size": 256, "n_layer": 6, "n_head": 4, "d_model": 128, "seq_len": 64, '
    '"compute_dtype": "fp32", "boundary_dtype": "fp32", "attn_impl": "naive", "sdpa_backend": '
    '"math", "arch": "builtin", "hf_model": null, "hf_model_sha256": null, "user_model": null, '
    '"user_model_sha256": null, "lora_rank": 0, "lora_alpha": 16.0, "lora_targets": "attn", '
    '"n_kv_head": null, "d_ff": null, "norm": "layernorm", "mlp": "gelu", "pos": "learned", '
    '"bias": true, "tie_embeddings": false, "rope_theta": 10000.0, "init_scheme": '
    '"torch_default", "init_std": 0.02, "fp8_boundaries": false}, "init_seed": 1234, '
    '"data_seed": 5678, "dataset_path": null, "data_shards": 1, "microbatches": 8, "mb_size": 4, '
    '"lr": 0.001, "adamw": {"betas": [0.9, 0.95], "eps": 1e-08, "weight_decay": 0.0}, '
    '"wire_codec": "raw", "recompute": false, "partition": [[0, 4], [4, 8]], "device_types": '
    '[["cpu"], ["cpu"]], "zero1_widths": [1, 1]}'
)
_GOLDEN_TIED_PINS = (
    '{"model": {"vocab_size": 256, "n_layer": 2, "n_head": 2, "d_model": 32, "seq_len": 64, '
    '"compute_dtype": "fp32", "boundary_dtype": "fp32", "attn_impl": "naive", "sdpa_backend": '
    '"math", "arch": "builtin", "hf_model": null, "hf_model_sha256": null, "user_model": null, '
    '"user_model_sha256": null, "lora_rank": 0, "lora_alpha": 16.0, "lora_targets": "attn", '
    '"n_kv_head": null, "d_ff": null, "norm": "rmsnorm", "mlp": "swiglu", "pos": "rope", '
    '"bias": false, "tie_embeddings": true, "rope_theta": 10000.0, "init_scheme": "gpt2", '
    '"init_std": 0.02, "fp8_boundaries": false}, "init_seed": 1234, "data_seed": 5678, '
    '"dataset_path": null, "data_shards": 1, "microbatches": 2, "mb_size": 4, "lr": 0.001, '
    '"adamw": {"betas": [0.9, 0.95], "eps": 1e-08, "weight_decay": 0.0}, "wire_codec": "raw", '
    '"recompute": false, "partition": [[0, 4]], "device_types": [["cpu"]], "zero1_widths": [1]}'
)


def _m4_config_cls():
    """A GPTConfig with the M4 fields (I-1) declared, so asdict() carries them as it will
    once model-core lands; a test-local subclass, not a stub of behaviour."""
    import dataclasses

    from wantrain.model.gpt import GPTConfig

    @dataclasses.dataclass
    class M4Config(GPTConfig):
        hf_config: dict | None = None
        hf_buffers: dict | None = None
        hf_census: list | None = None
        hf_code: dict | None = None
        hf_impl: str = "modules"
        moe_experts: int = 0
        moe_top_k: int = 0
        moe_bias_rate: float = 0.0

    return M4Config


def test_m4_stage_dense_pins_golden():
    # covers: RUNTIME-I15
    # WOULD-FAIL-IF: a new GPTConfig field (hf_*, moe_*) or device_classes leaks into dense
    # pins (BA-465), so a pre-M4 dense wave no longer resumes; or an hf model's pins drop
    # device_classes or its hf_* fields, or an MoE model's pins drop its moe_* fields.
    from wantrain.model.gpt import GPTConfig

    dense = JobSpec().finalize()
    assert json.dumps(dense.pins()) == _GOLDEN_DENSE_PINS
    tied = JobSpec(
        model=GPTConfig(
            n_layer=2,
            d_model=32,
            n_head=2,
            norm="rmsnorm",
            mlp="swiglu",
            pos="rope",
            bias=False,
            tie_embeddings=True,
            init_scheme="gpt2",
        ),
        stages=1,
        microbatches=2,
    ).finalize()
    assert json.dumps(tied.pins()) == _GOLDEN_TIED_PINS
    # The same golden with the M4 fields declared on the config (as after model-core).
    M4Config = _m4_config_cls()
    m4_dense = JobSpec(model=M4Config(), device_classes=[["cpu:x"], ["cpu:x"]]).finalize()
    assert json.dumps(m4_dense.pins()) == _GOLDEN_DENSE_PINS
    moe = JobSpec(model=M4Config(moe_experts=4, moe_top_k=2)).finalize()
    mp = moe.pins()["model"]
    assert (mp["moe_experts"], mp["moe_top_k"], mp["moe_bias_rate"]) == (4, 2, 0.0)
    assert not any(k.startswith("hf_c") or k == "hf_impl" for k in mp)
    # The schedule is given (the dense plan's, same partition): the default one prices HF
    # units from a census (scheduler-m4), which this pins check does not need.
    order = JobSpec(partition=[[0, 4], [4, 8]]).finalize().schedule_order
    hf = JobSpec(
        model=M4Config(arch="hf", hf_code={"transformers": "5.19.0"}),
        partition=[[0, 4], [4, 8]],
        device_classes=[["cpu:arm64:a"], ["cpu:arm64:a"]],
        schedule_order=order,
    ).finalize()
    hp = hf.pins()
    assert hp["device_classes"] == [["cpu:arm64:a"], ["cpu:arm64:a"]]
    assert hp["model"]["hf_code"] == {"transformers": "5.19.0"}
    assert hp["model"]["hf_impl"] == "modules" and "moe_top_k" not in hp["model"]
    from wantrain.runtime.spec import first_pin_difference

    moved = copy.deepcopy(hf)
    moved.device_classes = [["cuda:H100"], ["cuda:H100"]]  # one class (CONTROL-I25), moved
    assert first_pin_difference(hf.pins(), moved.finalize().pins()) == "device_classes"


def test_m4_stage_finalize_refuses_a_non_hf_tied_split():
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: finalize admits a dense GPTConfig with tie_embeddings whose Embed and
    # Head lie on different stages (each stage would train its own copy, BA-446), refuses
    # the same tie on one stage, or admits an hf split tied table through a wide stage
    # (BA-447) or with a non-fp32 boundary.
    from wantrain.model.gpt import GPTConfig

    def tied(**kw):
        return GPTConfig(n_layer=2, tie_embeddings=True, init_scheme="gpt2", **kw)

    with pytest.raises(ValueError, match=r"layers\.0\.tok\.weight.*stages 0 and 1"):
        JobSpec(model=tied(), stages=2, microbatches=2).finalize()
    with pytest.raises(ValueError, match="layers.0.tok.weight"):
        JobSpec(model=tied(), stages=2, partition=[[0, 3], [3, 4]], microbatches=2).finalize()
    one = JobSpec(model=tied(), stages=1, microbatches=2).finalize()
    assert one.partition == [[0, 4]] and not one.tied_relay()
    untied = JobSpec(model=GPTConfig(n_layer=2), stages=2, microbatches=2).finalize()
    assert not untied.tied_relay()
    # The schedule is given (an untied dense plan's of the same shape): the default one
    # prices HF units from a census (scheduler-m4), which these refusals do not need.
    part = [[0, 1], [1, 3], [3, 4]]

    def hf(widths, **kw):
        order = JobSpec(
            model=GPTConfig(n_layer=2),
            stages=3,
            stage_widths=widths,
            microbatches=2,
            partition=part,
        ).finalize()
        return JobSpec(
            model=tied(arch="hf", **kw),
            stages=3,
            stage_widths=widths,
            microbatches=2,
            partition=part,
            schedule_order=order.schedule_order,
        )

    with pytest.raises(ValueError, match="stage 1 has width 2"):
        hf([1, 2, 1]).finalize()
    with pytest.raises(ValueError, match="stage 0 has width 2"):
        hf([2, 1, 1]).finalize()
    with pytest.raises(ValueError, match="boundary_dtype 'fp32'"):
        hf([1, 1, 1], compute_dtype="bf16", boundary_dtype="bf16").finalize()


def test_m4_stage_param_shapes_of_hf_come_from_the_census(tmp_path):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: stage_param_shapes for arch 'hf' builds the model (finalize would import
    # transformers or wantrain.model.hfm in the coordinator or the planner, BA-419, BA-441),
    # or its names or shapes are not the census's param_shapes per unit of each stage.
    import subprocess

    census = [
        {"unit": 0, "param_shapes": [["embed_tokens.weight", [256, 64]]]},
        {
            "unit": 1,
            "param_shapes": [
                ["input_layernorm.weight", [64]],
                ["self_attn.q_proj.weight", [256, 64]],
            ],
        },
        {"unit": 2, "param_shapes": [["self_attn.q_proj.weight", [256, 64]]]},
        {"unit": 3, "param_shapes": [["norm.weight", [64]], ["lm_head.weight", [256, 64]]]},
    ]
    want = [
        {"0.embed_tokens.weight": [256, 64]},
        {
            "0.input_layernorm.weight": [64],
            "0.self_attn.q_proj.weight": [256, 64],
            "1.self_attn.q_proj.weight": [256, 64],
        },
        {"0.norm.weight": [64], "0.lm_head.weight": [256, 64]},
    ]
    code = f"""
import importlib.abc, json, sys
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "transformers" or name.startswith("transformers.") or name == "wantrain.model.hfm":
            raise ImportError("refused import of " + name)
        return None
sys.meta_path.insert(0, Refuse())
from wantrain.model.gpt import GPTConfig
from wantrain.runtime.spec import JobSpec
m = GPTConfig(arch="hf", d_model=64, vocab_size=256, n_layer=2)
m.hf_census = {json.dumps(census)}
shape = dict(stages=3, stage_widths=[1, 2, 1], partition=[[0, 1], [1, 3], [3, 4]], microbatches=2)
# the schedule is given (a dense plan's of the same shape): the default one prices HF
# units from the planner's census fields (scheduler-m4), which this crafted census lacks
order = JobSpec(model=GPTConfig(d_model=64, vocab_size=256, n_layer=2), **shape).finalize()
spec = JobSpec(model=m, **shape, schedule_order=order.schedule_order,
               wire="native", shard="pieces", stream_grads=True,
               piece_bounds=[None, {{"0.self_attn.q_proj.weight": [0, 8192, 16384]}}, None])
spec.finalize()
print(json.dumps([{{k: list(v) for k, v in s.items()}} for s in spec.stage_param_shapes()]))
print("hfm" in sys.modules, "transformers" in sys.modules)
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=tmp_path
    )
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert json.loads(lines[0]) == want
    assert lines[1] == "False False"
    # The finder is live: a spec that would build the model imports refused names.
    bad = code.replace('GPTConfig(arch="hf"', 'GPTConfig(arch="builtin"').replace(
        "from wantrain.model.gpt import GPTConfig", "import transformers  # must be refused"
    )
    out = subprocess.run([sys.executable, "-c", bad], capture_output=True, text=True, timeout=300)
    assert out.returncode != 0 and "refused import of transformers" in out.stderr


def test_m4_stage_main_refuses_an_hf_spec_without_a_launcher(tmp_path, monkeypatch):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: `python -m wantrain.worker.stage_main` starts an arch 'hf' slot without
    # --launcher (no gate, no admission, BA-445), reads the arch only after JobSpec.load, or
    # refuses an hf spec under a launcher or a dense spec without one.
    from wantrain.worker import stage_main

    reached = []
    monkeypatch.setattr(stage_main, "run_stage", lambda spec, r, p: reached.append(spec) or 0)
    out = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a))))
    hf_path = tmp_path / "hf.json"
    # Not a loadable JobSpec on purpose: the refusal must come before JobSpec.load.
    hf_path.write_text(json.dumps({"model": {"arch": "hf", "unknown_field": 1}}))
    dense_path = tmp_path / "dense.json"
    dense_path.write_text(_m2b(stages=2).to_json())
    assert stage_main.main(["--spec", str(hf_path), "--rank", "0"]) == 2
    assert reached == []
    assert any(x.startswith("WT_REFUSED arch 'hf' refused") for x in out)
    assert out[-1] == "WT_DONE 2"
    for launcher in ("local", "coordinator"):
        assert stage_main.main(["--spec", str(hf_path), "--rank", "0", "--launcher", launcher]) == 0
    assert stage_main.main(["--spec", str(dense_path), "--rank", "1"]) == 0
    assert reached == [str(hf_path), str(hf_path), str(dense_path)]


def _stock_tied(hs, es, rows):
    """The stock fold of a tied table: g_i = h_i + e_i (e_i dense, +0 outside T_i), summed
    in microbatch-index order by OrderedAccumulator."""
    from wantrain.runtime.accumulate import OrderedAccumulator

    acc = OrderedAccumulator(range(len(hs)))
    for i, (h, e, r) in enumerate(zip(hs, es, rows)):
        dense = torch.zeros_like(h)
        dense[r] = e
        acc.add(i, [h + dense])
    return acc.result()[0]


def _bits(t):
    return t.contiguous().view(torch.int32)


def test_m4_stage_tied_row_fold_equals_the_stock_fold():
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: TiedRowFold folds a held row before an earlier microbatch's embedding
    # rows, folds the Head use without + 0.0 on untouched rows (a -0 survives), assigns
    # instead of adding (or the reverse) at microbatch 0, mishandles repeated ids or an empty
    # T_i, depends on the arrival order of h and e, or equals a per-use fold.
    from wantrain.runtime.accumulate import TiedRowFold

    rng = random.Random(4)
    gen = torch.Generator().manual_seed(4)
    n_rows, d = 24, 5
    cases = 0
    for M in (1, 2, 3, 6):
        for kind in ("distinct", "repeated", "empty"):
            for _ in range(3):
                hs, es, rows = [], [], []
                for i in range(M):
                    if kind == "empty" and i == M // 2:
                        ids = torch.empty(0, dtype=torch.int64)
                    elif kind == "repeated":
                        ids = torch.tensor([rng.randrange(6) for _ in range(9)])
                    else:
                        ids = torch.tensor(rng.sample(range(n_rows), 5))
                    r = torch.unique(ids, sorted=True)
                    h = torch.randn(n_rows, d, generator=gen) * 10 ** rng.randint(-3, 3)
                    h[rng.randrange(n_rows), rng.randrange(d)] = -0.0
                    e = torch.randn(r.numel(), d, generator=gen)
                    if r.numel():
                        e[0, 0] = -0.0
                    hs.append(h)
                    es.append(e)
                    rows.append(r)
                want = _stock_tied(hs, es, rows)
                fold = TiedRowFold(n_rows, range(M))
                events = [("h", i) for i in range(M)] + [("e", i) for i in range(M)]
                rng.shuffle(events)
                for what, i in events:
                    assert not fold.complete()
                    if what == "h":
                        fold.add_head(i, hs[i], rows[i])
                    else:
                        fold.add_embed(i, es[i])
                assert fold.complete()
                assert torch.equal(_bits(fold.result()), _bits(want)), (M, kind)
                cases += 1
    assert cases == 36
    # Untouched rows come out as acc + (h_i + 0.0): a -0 in h_0 becomes +0, as in the stock
    # fold (h + e with e = +0); a fold that took h_0 as is would keep the sign bit.
    h = torch.zeros(4, 2)
    h[3, 1] = -0.0
    fold = TiedRowFold(4, [0])
    fold.add_head(0, h, torch.tensor([0]))
    fold.add_embed(0, torch.ones(1, 2))
    got = fold.result()
    assert _bits(got)[3, 1] == 0 and _bits(h)[3, 1] != 0
    assert torch.equal(_bits(got), _bits(_stock_tied([h], [torch.ones(1, 2)], [torch.tensor([0])])))
    # Control: the per-use fold (sum of h, then of e, added) is another recipe.
    gen = torch.Generator().manual_seed(9)
    hs = [torch.randn(16, 3, generator=gen) for _ in range(6)]
    rows = [torch.tensor([1, 4, 9]) for _ in range(6)]
    es = [torch.randn(3, 3, generator=gen) * 1e3 for _ in range(6)]
    per_h, per_e = hs[0].clone(), torch.zeros(16, 3)
    for i in range(6):
        if i:
            per_h += hs[i]
        per_e[rows[i]] += es[i]
    stock = _stock_tied(hs, es, rows)
    assert not torch.equal(_bits(per_h + per_e), _bits(stock))
    fold = TiedRowFold(16, range(6))
    for i in range(6):
        fold.add_head(i, hs[i], rows[i])
        fold.add_embed(i, es[i])
    assert torch.equal(_bits(fold.result()), _bits(stock))
    with pytest.raises(RuntimeError, match="second Head gradient for microbatch 2"):
        fold.add_head(2, hs[2], rows[2])
    short = TiedRowFold(16, range(2))
    short.add_head(0, hs[0], rows[0])
    with pytest.raises(RuntimeError, match="incomplete"):
        short.result()


def test_m4_stage_count_fold_is_order_free_and_refuses_bad_vectors():
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: the owner's integer count fold depends on the arrival order, accepts a
    # vector whose sum is not T_mb*k (zero1), a float, misshapen or negative vector, a
    # (mb, layer) twice, or completes with a (mb, layer) missing, or a refusal does not name
    # (step, mb, layer, peer).
    from wantrain.runtime.accumulate import CountFold

    rng = random.Random(7)
    t_mb, k, E, M = 12, 2, 4, 5

    def vec():
        cuts = sorted(rng.randrange(t_mb * k + 1) for _ in range(E - 1))
        parts = [b - a for a, b in zip([0, *cuts], [*cuts, t_mb * k])]
        return torch.tensor(parts, dtype=torch.int64)

    vecs = {(mb, layer): vec() for mb in range(M) for layer in (3, 7)}
    want = {
        layer: sum((vecs[(mb, layer)] for mb in range(M)), torch.zeros(E, dtype=torch.int64))
        for layer in (3, 7)
    }
    assert not torch.equal(want[3], want[7])  # distinct expected values
    results = []
    for _ in range(4):
        keys = list(vecs)
        rng.shuffle(keys)
        f = CountFold(9, M, [3, 7], {3: E, 7: E}, {3: t_mb * k, 7: t_mb * k})
        for mb, layer in keys:
            f.add(mb, layer, vecs[(mb, layer)], peer=1)
        got = f.result()
        assert all(torch.equal(got[layer], want[layer]) for layer in (3, 7))
        results.append(got)

    def fresh():
        return CountFold(9, M, [3], {3: E}, {3: t_mb * k})

    short = vecs[(0, 3)].clone()
    short[0] -= 1
    with pytest.raises(RuntimeError, match=r"\(step 9, mb 0, layer 3, peer 1\).*T_mb\*k = 24"):
        fresh().add(0, 3, short, peer=1)
    neg = vecs[(0, 3)].clone()
    neg[0], neg[1] = -1, neg[1] + neg[0] + 1
    with pytest.raises(RuntimeError, match=r"mb 0, layer 3, peer 2\): a negative count"):
        fresh().add(0, 3, neg, peer=2)
    with pytest.raises(RuntimeError, match="not an int64 vector"):
        fresh().add(1, 3, vecs[(1, 3)].float(), peer=1)
    with pytest.raises(RuntimeError, match=r"shape \(5,\)"):
        fresh().add(1, 3, torch.cat([vecs[(1, 3)], torch.zeros(1, dtype=torch.int64)]), peer=1)
    with pytest.raises(RuntimeError, match="no selection bias of that layer"):
        fresh().add(1, 7, vecs[(1, 7)], peer=1)
    twice = fresh()
    twice.add(2, 3, vecs[(2, 3)], peer=1)
    with pytest.raises(RuntimeError, match=r"mb 2, layer 3, peer 0\): count vector given twice"):
        twice.add(2, 3, vecs[(2, 3)], peer=0)
    missing = fresh()
    for mb in (0, 1, 3, 4):
        missing.add(mb, 3, vecs[(mb, 3)], peer=1)
    assert missing.missing() == [(2, 3)] and not missing.complete()
    with pytest.raises(RuntimeError, match=r"mb 2, layer 3.*missing at the owner"):
        missing.result()
    # A piece lane (total None) is not held to the whole vector's sum.
    piece = CountFold(9, 1, [3], {3: 2}, None)
    piece.add(0, 3, torch.tensor([5, 0]), peer=0)
    assert torch.equal(piece.result()[3], torch.tensor([5, 0]))


def test_m4_stage_check_frame_count_lanes():
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: exchange.check_frame accepts a count lane that is not int64, misshapen,
    # holds a negative count, is missing from its unit's frame, or arrives twice, or refuses a
    # valid one (BA-455), or a dense table's frames are checked differently.
    from wantrain.runtime import exchange as xc

    lane = torch.tensor([3, 0, 5], dtype=torch.int64)
    norm = torch.arange(4, dtype=torch.float32)
    table = {0: [(torch.float32, 4), (torch.int64, 3, xc.COUNT_LANE)]}

    def expect():
        return {"units": table, "M": 4, "step": 2, "seen": set()}

    def frame(tensors, mb=1, unit=0):
        (env, view), *_ = xc.chunk_object(unit, tensors)
        return dict(env, mb=mb, step=2, sender=1), list(view)

    env, view = frame([norm, lane])
    ex = expect()
    xc.check_frame("GRADS", env, view, ex)  # valid
    with pytest.raises(xc.FrameError, match="duplicate"):
        xc.check_frame("GRADS", env, view, ex)
    for tensors, cls in (
        ([norm, lane.float()], "dtype"),
        ([norm, lane.to(torch.int32)], "dtype"),
        ([norm, torch.tensor([3, 0, 5, 1])], "shape"),
        ([norm], "shape"),
        ([norm, torch.tensor([3, -1, 5])], "count"),
    ):
        env, view = frame(tensors)
        with pytest.raises(xc.FrameError) as e:
            xc.check_frame("GRADS", env, view, expect())
        assert e.value.cls == cls, (tensors, e.value)
    env, view = frame([norm, torch.tensor([3, -1, 5])])
    with pytest.raises(xc.FrameError, match=r"step 2, mb 1, sender 1\) count lane 1"):
        xc.check_frame("GRADS", env, view, expect())
    # A count lane cut across two chunks is checked chunk by chunk.
    parts = xc.chunk_object(0, [norm, lane], max_bytes=24)
    assert len(parts) == 2
    ex = expect()
    for env, view in parts:
        xc.check_frame("GRADS", dict(env, mb=0, step=3, sender=1), list(view), ex)
    # A dense table (no count lane) accepts a negative float as before.
    dense = {"units": {0: [(torch.float32, 4)]}, "M": 4, "step": 2, "seen": set()}
    env, view = frame([-norm])
    xc.check_frame("GRADS", env, view, dense)


class _FakeMoE(torch.nn.Module):
    """A module that calls moe_forward, as far as the stage can see (I-11): moe_layer, a
    persistent fp32 sel_bias of E experts and its moe_cfg (a recipe dict, as hfm's swap
    sets it; arch.MoEMLP sets its GPTConfig), read through arch.moe_recipe. A crafted
    fixture for the count transport; the real block's records are stage-wire's."""

    def __init__(self, layer, experts, top_k=2):
        super().__init__()
        self.moe_layer = layer
        self.moe_cfg = {"moe_experts": experts, "moe_top_k": top_k, "moe_bias_rate": 0.5}
        self.register_buffer("sel_bias", torch.zeros(experts))


def _zero1_owner(native=False, stream=False):
    """Peer 0 of a width-2 ZeRO-1 stage with two trainable tensors and two MoE layers
    (biases at ownership indices 2 and 3: layer 5 owned by peer 0, layer 8 by peer 1),
    set up by StageRuntime's own _init_moe and _init_zero1_tables."""
    from wantrain.runtime import exchange as xc

    spec = _m2b(
        stages=2,
        stage_widths=[1, 2],
        microbatches=4,
        mb_size=2,
        wire="native" if native else "fp32",
        stream_grads=stream or native,
    ).finalize()
    spec.model.seq_len = 3
    chunk = torch.nn.ModuleList(
        [
            torch.nn.Linear(2, 2, bias=False),
            _FakeMoE(5, 4),
            torch.nn.Linear(2, 1, bias=False),
            _FakeMoE(8, 3),
        ]
    )
    rt = StageRuntime.__new__(StageRuntime)
    rt.spec, rt.rank, rt.peer, rt.M = spec, 1, 0, 4
    rt.widths, rt.w, rt.wide, rt.my_vs = [1, 2], 2, True, [0]
    rt.chunks = {0: chunk}
    rt.device = torch.device("cpu")
    rt.params = [p for p in chunk.parameters()]
    rt.shard_params = rt.params[0::2]
    rt.native, rt.pieces_mode = native, False
    rt._init_moe()
    rt._init_zero1_tables()
    rt._seen, rt._exec_step, rt._parts = set(), 6, {}
    rt.shard_acc, rt._sync_grads, rt.holds = {}, {}, xc.HoldCounters()
    return rt


def _t_vec(E, total, seed):
    g = random.Random(seed)
    cuts = sorted(g.randrange(total + 1) for _ in range(E - 1))
    return torch.tensor([b - a for a, b in zip([0, *cuts], [*cuts, total])], dtype=torch.int64)


def test_m4_stage_zero1_count_payloads_validate_at_the_owner():
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: the ZeRO-1 ownership list does not append the biases after the trainable
    # parameters, a count vector of any of the three gradient paths (batch OBJ dict, stream
    # ('mb', ...) tuple, native COUNT frame) is not folded at its owner, or a malformed,
    # duplicated or missing vector is accepted, or a refusal does not name (step, mb, layer, peer).
    from wantrain.runtime import exchange as xc
    from wantrain.transport import framing as fr

    rt = _zero1_owner()
    assert rt._own[:2] == rt.params and rt._own[2:] == rt._biases
    assert [int(rt._bias_mods[j].moe_layer) for j in rt._my_bias] == [5]
    total = 2 * 3 * 2  # T_mb * k
    good = {mb: _t_vec(4, total, mb) for mb in range(4)}
    want = sum(good.values(), torch.zeros(4, dtype=torch.int64))
    grads = {mb: [torch.full((2, 2), float(mb))] for mb in range(4)}

    def grads_frame(mb=0):
        return SimpleNamespace(type=fr.GRADS, step=6, mb=mb)

    # batch: {mb: tensors, "counts": {mb: {layer: int64 (E,)}}}
    msg = {1: grads[1], 3: grads[3], "counts": {1: {5: good[1]}, 3: {5: good[3]}}}
    rt._on_legacy("GRADS", grads_frame(), msg, 1)
    for mb in (0, 2):
        rt._count_add(6, mb, {5: good[mb]}, 0)
    assert torch.equal(rt._count_fold(6).result()[5], want)
    assert list(rt._sync_grads[0]) == [1, 3]  # the gradients, without the counts key
    for bad, match in (
        ({1: grads[1]}, r"step 6, peer 1\): GRADS without its counts"),
        ({1: grads[1], "counts": {2: {5: good[2]}}}, r"mb 2, peer 1\): counts of a microbatch"),
        ({1: grads[1], "counts": {}}, r"mb 1, peer 1\): count vectors missing"),
        ({1: grads[1], "counts": {1: {5: -good[1]}}}, r"mb 1, layer 5, peer 1\): a negative"),
    ):
        fresh = _zero1_owner()
        with pytest.raises(RuntimeError, match=match):
            fresh._on_legacy("GRADS", grads_frame(), bad, 1)
    # stream: ("mb", mb, tensors, {layer: counts}) per microbatch
    rt = _zero1_owner(stream=True)
    for mb in (1, 3):
        rt._on_legacy("GRADS", grads_frame(mb), ("mb", mb, grads[mb], {5: good[mb]}), 1)
    for mb in (0, 2):
        rt._count_add(6, mb, {5: good[mb]}, 0)
    assert torch.equal(rt._count_fold(6).result()[5], want)
    rt = _zero1_owner(stream=True)
    short = good[1].clone()
    short[0] += 1
    with pytest.raises(RuntimeError, match=r"mb 1, layer 5, peer 1\): counts sum to 13"):
        rt._on_legacy("GRADS", grads_frame(1), ("mb", 1, grads[1], {5: short}), 1)
    with pytest.raises(xc.FrameError, match="not a shard frame"):
        rt._on_legacy("GRADS", grads_frame(2), ("mb", 2, grads[2]), 1)  # the counts dropped
    rt._on_legacy("GRADS", grads_frame(3), ("mb", 3, grads[3], {}), 1)
    with pytest.raises(RuntimeError, match=r"mb 0, layer 5.*missing at the owner"):
        rt._count_fold(6).result()
    # wire native: one COUNT exchange frame per (mb, bias index), RAW_I64 (E,)
    rt = _zero1_owner(native=True)
    assert rt._native_table[2] == [(torch.int64, 4, xc.COUNT_LANE)] and 3 not in rt._native_table

    def count_frame(mb, vec, unit=2):
        (env, view), *_ = xc.chunk_object(unit, [vec])
        f = SimpleNamespace(type=fr.GRADS, step=6, mb=mb)
        return f, ("x", env, list(view))

    for mb in (1, 3):
        rt._on_exchange(*count_frame(mb, good[mb]), ("intra", 1))
    for mb in (0, 2):
        rt._count_add(6, mb, {5: good[mb]}, 0)
    assert torch.equal(rt._count_fold(6).result()[5], want)
    rt = _zero1_owner(native=True)
    rt._on_exchange(*count_frame(1, good[1]), ("intra", 1))
    with pytest.raises(xc.FrameError, match="duplicate"):
        rt._on_exchange(*count_frame(1, good[1]), ("intra", 1))
    with pytest.raises(xc.FrameError, match="dtype"):
        rt._on_exchange(*count_frame(2, good[2].float()), ("intra", 1))
    with pytest.raises(xc.FrameError, match="unknown unit 3"):
        rt._on_exchange(*count_frame(2, good[2], unit=3), ("intra", 1))  # peer 1's bias
    with pytest.raises(xc.FrameError, match="count lane 0 holds a negative"):
        rt._on_exchange(*count_frame(2, -good[2]), ("intra", 1))
    # dense payloads are unchanged: a dense stage's ownership list is its parameters
    dense = _zero1_owner()
    dense.chunks = {0: torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])}
    dense.params = list(dense.chunks[0].parameters())
    dense._init_moe()
    assert not dense.moe and dense._own == dense.params and dense._my_bias == []


def test_m4_stage_commit_wave_lists_hf_buffers_only_for_hf(tmp_path, monkeypatch):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: an hf slot's wave does not list hf_buffers.json (canonical JSON of the
    # pinned record) through commit_wave(..., extra_files) (I-15, BA-460), or a dense slot's
    # commit_wave call changes (an extra_files keyword, even empty).
    from wantrain.runtime import stage as stage_mod

    calls = []

    def recording_commit_wave(*args, **kwargs):  # the module-level name, recorded
        calls.append((args, kwargs))
        return None, False

    monkeypatch.setattr(stage_mod, "commit_wave", recording_commit_wave)
    rt = _tiny_stage(tmp_path, 1.0)
    rt._commit_wave(3, "save", None)
    assert calls == [((str(tmp_path), "t", 0, 0, 3, "save"), {})]
    record = {"v": 1, "tensors": {"b.inv_freq": {"sha256": "ab", "b64": "AA=="}}, "model_type": "x"}
    rt.hf = True
    rt.spec.model.hf_buffers = record
    rt._commit_wave(4, "save", None)
    args, kwargs = calls[1]
    assert args == (str(tmp_path), "t", 0, 0, 4, "save")
    want = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    assert kwargs == {"extra_files": {"hf_buffers.json": want}}
    assert want.startswith(b'{"model_type":"x","tensors"')


def test_m4_stage_hf_buffer_pin_and_device_classes_on_resume(tmp_path):
    # covers: RUNTIME-I19, RUNTIME-I15
    # WOULD-FAIL-IF: a wave whose hf_buffers.json holds a tensor whose bytes differ from the
    # pin resumes (or the refusal does not name the tensor), a missing record file resumes,
    # or a device_classes change is not named by its slot.
    import base64
    import hashlib

    from wantrain.runtime.stage import check_hf_buffers, device_class_change

    raw = bytes(range(8))
    pin = {
        "v": 1,
        "tensors": {
            "model.rotary_emb.inv_freq": {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "b64": base64.b64encode(raw).decode(),
            }
        },
    }
    (tmp_path / "hf_buffers.json").write_text(json.dumps(pin))
    check_hf_buffers(str(tmp_path), pin)
    edited = copy.deepcopy(pin)
    edited["tensors"]["model.rotary_emb.inv_freq"]["b64"] = base64.b64encode(raw[::-1]).decode()
    (tmp_path / "hf_buffers.json").write_text(json.dumps(edited))
    with pytest.raises(RuntimeError, match="BUFFER-PIN model.rotary_emb.inv_freq"):
        check_hf_buffers(str(tmp_path), pin)
    (tmp_path / "hf_buffers.json").unlink()
    with pytest.raises(RuntimeError, match="hf_buffers.json"):
        check_hf_buffers(str(tmp_path), pin)
    a = [["cpu:arm64:x"], ["cpu:arm64:x", "cpu:arm64:y"]]
    assert device_class_change(a, copy.deepcopy(a)) is None
    b = [["cpu:arm64:x"], ["cpu:arm64:x", "cuda:H100"]]
    assert device_class_change(a, b).startswith("r1p1 ('cuda:H100' -> 'cpu:arm64:y')")


def test_m4_stage_moe_replay_names_step_mb_and_layer():
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: B's rebuild may record aux values or counts that differ from F's without
    # MOE-REPLAY naming (step, mb, layer), a layer recorded twice or not at all in one forward
    # passes, or the aux hex is not exact (a 1-ulp change goes unseen).
    from wantrain.runtime.stage import moe_aux_hex, moe_records, moe_replay_check

    def rec(layer, lbl, counts):
        return SimpleNamespace(
            layer=layer, aux={"z": 0.25, "lbl": lbl}, counts=torch.tensor(counts)
        )

    f = moe_records([rec(1, 0.1, [2, 2]), rec(4, 0.3, [3, 1])], 7, 2, [1, 4])
    assert f[1][0] == '{"lbl":"0x1.999999999999ap-4","z":"0x1.0000000000000p-2"}'
    assert f[4][0] != f[1][0] and f[1][1].dtype == torch.int64
    moe_replay_check(7, 2, f, moe_records([rec(1, 0.1, [2, 2]), rec(4, 0.3, [3, 1])], 7, 2, [1, 4]))
    ulp = math.nextafter(0.3, 1.0)
    for replay, layer in (
        ([rec(1, 0.1, [2, 2]), rec(4, ulp, [3, 1])], 4),
        ([rec(1, 0.1, [1, 3]), rec(4, 0.3, [3, 1])], 1),
    ):
        got = moe_records(replay, 7, 2, [1, 4])
        with pytest.raises(RuntimeError, match=rf"MOE-REPLAY \(step 7, mb 2, layer {layer}\)"):
            moe_replay_check(7, 2, f, got)
    with pytest.raises(RuntimeError, match=r"MOE-REPLAY \(step 7, mb 2, layer 4\)"):
        moe_replay_check(7, 2, f, {1: f[1]})
    with pytest.raises(RuntimeError, match=r"MOE-RECORD \(step 3, mb 0, layer 1\): recorded twice"):
        moe_records([rec(1, 0.1, [2, 2]), rec(1, 0.1, [2, 2])], 3, 0, [1])
    with pytest.raises(RuntimeError, match=r"MOE-RECORD \(step 3, mb 0, layer 4\): not recorded"):
        moe_records([rec(1, 0.1, [2, 2])], 3, 0, [1, 4])
    assert moe_aux_hex({"b": 1.5, "a": -0.0}) == '{"a":"-0x0.0p+0","b":"0x1.8000000000000p+0"}'
    # arch.moe_forward records each aux value as a float hex string: the same text as the
    # number, and the oracle's (validation.state.aux_hex) for the same record.
    from wantrain.validation.state import aux_hex

    hexed = {"lbl": (0.1).hex(), "z": (0.25).hex()}
    assert moe_aux_hex(hexed) == moe_aux_hex({"lbl": 0.1, "z": 0.25}) == aux_hex(hexed)
    assert moe_records([rec(1, (0.1).hex(), [2, 2])], 7, 2, [1])[1][0] == f[1][0]


def test_m4_stage_dispatch_raises_on_an_unhandled_frame_type():
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the stage's main dispatch drops a frame of a type it does not handle
    # (BA-416), or accepts a TIED frame under a plan with no tied relay instead of refusing it
    # before its payload is used.
    from wantrain.transport import framing as fr

    rt = StageRuntime.__new__(StageRuntime)
    rt.my_vs, rt.rank = [0], 0
    rt.inbox = queue_mod.Queue()
    for ftype in (99, getattr(fr, "TIED", 7)):
        f = fr.Frame(ftype, fr.RAW_F32, 1, 0, 0, b"\0" * 8, (1, 2))
        rt.inbox.put(("net", f, None, ("prev", 0)))
        with pytest.raises(RuntimeError, match=rf"refused (frame type {ftype}|TIED frame)"):
            rt._pump_inbox(block=False)
    # A handled type still lands where it did (control: the raise is not unconditional).
    rt.wide, rt.fwd_ready = False, {}
    x = torch.arange(2.0)
    rt.inbox.put(("net_dec", fr.Frame(fr.FWD, fr.RAW_F32, 1, 3, 0, b"", (2,)), x, ("prev", 0)))
    rt._pump_inbox(block=False)
    assert rt.fwd_ready == {(1, 3, 0): x}


def test_m4_stage_tied_send_predicate_and_sequence():
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the Head sends TIED(t+1, k) after the attempt's last step (the run's
    # final step, or the step a checkpoint-and-stop attempt ends on), so END finds a TIED
    # frame in flight; or the TIED receipt sequence admits an out-of-order, duplicated or
    # extra frame or fragment.
    from wantrain.runtime.stage import _TiedSeq, tied_sends_after

    spec = _spec(stages=2, steps=6)
    last = spec.steps - 1
    assert [tied_sends_after(s, last) for s in range(spec.steps)] == [True] * 5 + [False]
    # A checkpoint-and-stop attempt (ckpt every 2, cut after step 3): steps = 4.
    cut = _spec(stages=2, steps=4, ckpt_every=2)
    assert [tied_sends_after(s, cut.steps - 1) for s in range(cut.steps)] == [
        True,
        True,
        True,
        False,
    ]
    seq = _TiedSeq([(1, 0), (1, 1), (2, 0)])
    seq.push(1, 0)
    with pytest.raises(RuntimeError, match=r"\(step 1, mb 0\) duplicated"):
        seq.push(1, 0)
    with pytest.raises(RuntimeError, match=r"\(step 2, mb 0\) out of sequence, expected \(1, 1\)"):
        seq.push(2, 0)  # step 2's rows before step 1's last frame: no relay sends that
    with pytest.raises(RuntimeError, match=r"\(step 1, mb 5\) is no frame of this attempt"):
        seq.push(1, 5)
    seq = _TiedSeq([(1, 0), (1, 1), (2, 0)])
    seq.push(1, 0, 1, 2)
    with pytest.raises(RuntimeError, match="fragment 1/2 refused"):
        seq.push(1, 0, 1, 2)
    with pytest.raises(RuntimeError, match="fragment 0/3 refused"):
        seq.push(1, 0, 0, 3)
    # A multi-stream link interleaves the fragments of one step's frames across its
    # sockets: (1, 1) may complete before (1, 0) (control: the check is not plain order).
    seq.push(1, 1, 0, 2)
    seq.push(1, 1, 1, 2)
    assert seq.left() == (1, 0) and not seq.done()
    seq.push(1, 0, 0, 2)
    assert seq.left() == (2, 0)
    with pytest.raises(RuntimeError, match=r"\(step 1, mb 1\) duplicated"):
        seq.push(1, 1)
    seq.push(2, 0)
    assert seq.done() and seq.left() is None
    with pytest.raises(RuntimeError, match="beyond the attempt's last frame"):
        seq.push(2, 0)


def test_m4_stage_tied_sparse_refuses_any_bit_outside_the_ids():
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: TIED-SPARSE admits an embedding gradient with a nonzero or -0 element
    # on a row outside T_i (a NaN, a denormal, -0.0, in fp32 or bf16), or refuses one whose
    # nonzero elements all lie on rows of T_i.
    from wantrain.runtime.stage import check_tied_sparse

    for dtype in (torch.float32, torch.bfloat16):
        e = torch.zeros(6, 4, dtype=dtype)
        e[1] = torch.tensor([1.0, -2.0, 0.5, -0.0], dtype=dtype)
        e[4, 2] = 3.0
        outside = torch.tensor([True, False, True, True, False, True])
        check_tied_sparse(3, 1, e, outside)  # every nonzero on rows 1 and 4 (in T_i)
        for bad in (-0.0, float("nan"), 1e-40, -7.0):
            f = e.clone()
            f[5, 3] = bad
            with pytest.raises(RuntimeError, match=r"TIED-SPARSE \(step 3, mb 1\)"):
                check_tied_sparse(3, 1, f, outside)
        g = e.clone()
        g[0, 0] = -0.0  # control: the same -0 on a row of T_i passes
        inside0 = outside.clone()
        inside0[0] = False
        check_tied_sparse(3, 1, g, inside0)
        with pytest.raises(RuntimeError, match="TIED-SPARSE"):
            check_tied_sparse(3, 1, g, outside)
    check_tied_sparse(0, 0, torch.zeros(0, 4), torch.zeros(0, dtype=torch.bool))


# ======================================================= stage-wire (M4, wave 2)
#
# The stage behaviour stage-m4 coded against its siblings' interfaces, proven on real
# hfm units (tiny configs of the pinned transformers classes, hfm.tiny_config, with the
# buffer record built by hfm.buffer_record in this process), real native MoE blocks
# (arch.MoEMLP) and real framing.TIED frames. HF tests run on the transformers overlay.

_WIRE_FAMS = {
    "llama": ("llama", {"tie_word_embeddings": False}),
    "llama_tied": ("llama", {"tie_word_embeddings": True}),
    "gemma2": (
        "gemma2",
        {"sliding_window": 8, "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0},
    ),
}
_WIRE_CACHE: dict = {}


def _wire_hf(fam):
    """(hf_config, buffer record, census, hf_code) of a tiny family config, built once per
    process (the record hashes the installed RECORDs once, seconds)."""
    if fam not in _WIRE_CACHE:
        from wantrain.model import hfm

        mt, over = _WIRE_FAMS[fam]
        hc = hfm.tiny_config(mt, use_cache=False, **over)
        _WIRE_CACHE[fam] = (hc, hfm.buffer_record(hc), hfm.unit_census(hc), hfm.hf_code())
    return copy.deepcopy(_WIRE_CACHE[fam])


def _wire_cfg(fam, **kw):
    from wantrain.model import hfm
    from wantrain.model.gpt import GPTConfig

    hc, record, census, code = _wire_hf(fam)
    fields = hfm.config_fields(hc)
    fields.update(kw)
    return GPTConfig(
        arch="hf",
        hf_config=hc,
        hf_buffers=record,
        hf_census=census,
        hf_code=code,
        seq_len=16,
        **fields,
    )


def _wire_spec(fam, partition, model_kw=None, **kw):
    kw.setdefault("microbatches", 2)
    kw.setdefault("mb_size", 2)
    kw.setdefault("steps", 3)
    return JobSpec(
        model=_wire_cfg(fam, **(model_kw or {})),
        stages=len(partition),
        partition=partition,
        **kw,
    )


@pytest.fixture
def hf_env(monkeypatch):
    """SPEC-model §2's environment for an hf stage built in this process: HF_ENV set, and
    hfm.guard_environment replaced by a call recorder. A FAKE at the process boundary,
    named as such: the real guard refuses every process started without
    -X pycache_prefix (pytest is), and its refusals are model-core's tests (test_hfm_*)."""
    from wantrain.model import hfm

    for k, v in hfm.HF_ENV.items():
        monkeypatch.setenv(k, v)
    calls: list = []
    monkeypatch.setattr(hfm, "guard_environment", lambda: calls.append("guard"))
    return calls


def _frame_bytes(ftype, step, mb, payload, **kw) -> bytes:
    """The bytes framing.send_frame writes for one frame."""
    from wantrain.transport import framing as fr

    out = bytearray()
    fr.send_frame(SimpleNamespace(sendall=out.extend), ftype, step, mb, payload, **kw)
    return bytes(out)


def _tied_bytes(step, mb, rows, cuts=(), shape=None) -> list:
    """framing.TIED frames of `rows` (n, h), fp32, cut at the byte offsets `cuts`."""
    from wantrain.transport import framing as fr

    payload = rows.float().contiguous().numpy().tobytes()
    edges = [0, *cuts, len(payload)]
    n = len(edges) - 1
    return [
        _frame_bytes(
            fr.TIED,
            step,
            mb,
            payload[a:b],
            codec=fr.RAW_F32,
            shape=tuple(shape or rows.shape),
            frag_idx=i,
            frag_cnt=n,
        )
        for i, (a, b) in enumerate(zip(edges, edges[1:]))
    ]


def _parse(data: bytes):
    """One framing.Frame read back from its bytes by recv_frame."""
    import io

    from wantrain.transport import framing as fr

    return fr.recv_frame(SimpleNamespace(recv=io.BytesIO(data).read))


def _rows(rt, n, seed):
    d = int(rt.spec.model.d_model)
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _restore_determinism():
    from wantrain.runtime.determinism import check_determinism

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    check_determinism()


def test_m4_wire_start_asserts_name_each_flag(hf_env):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: an hf stage starts with deterministic algorithms off, an fp32 matmul
    # precision below 'highest', sdpa attention, a dropout above 0 (a config key or an
    # nn.Dropout module), HF gradient checkpointing on, use_cache True, a module of its chunk
    # in eval mode, or an installed hf_code other than the pin; or a refusal does not name the
    # flag; or StageRuntime.__init__ skips the asserts.
    from torch import nn

    from wantrain.model.gpt import build_layers

    spec = _wire_spec("llama", [[0, 3], [3, 6]])
    rt = StageRuntime(spec, 0)
    assert rt.hf and hf_env == ["guard"]
    pinned = copy.deepcopy(rt.spec.model.hf_code)

    def fresh():
        full = build_layers(rt.spec.model, rt.spec.init_seed)
        rt.chunks = {0: nn.ModuleList(full[0:3])}
        return full

    rt._hf_start_asserts(fresh())  # control: real units under the pins pass
    cases = [
        (
            r"torch\.are_deterministic_algorithms_enabled\(\) is False",
            lambda f: torch.use_deterministic_algorithms(False),
        ),
        (
            r"torch\.get_float32_matmul_precision\(\) is 'high'",
            lambda f: torch.set_float32_matmul_precision("high"),
        ),
        (
            r"HF-ASSERT _attn_implementation: is 'sdpa'",
            lambda f: setattr(f.root.config, "_attn_implementation", "sdpa"),
        ),
        (
            r"HF-ASSERT attention_dropout: is 0\.1",
            lambda f: setattr(f.root.config, "attention_dropout", 0.1),
        ),
        (
            r"HF-ASSERT dropout: module model\.layers\.0\.self_attn\.drop has p=0\.1",
            lambda f: f.root.model.layers[0].self_attn.add_module("drop", nn.Dropout(0.1)),
        ),
        (
            r"HF-ASSERT gradient_checkpointing: is on in module model",
            lambda f: f.root.gradient_checkpointing_enable(),
        ),
        (r"HF-ASSERT use_cache: is True", lambda f: setattr(f.root.config, "use_cache", True)),
        (
            r"HF-ASSERT training: module \S*model\.layers\.1\.mlp is in eval mode",
            lambda f: f.root.model.layers[1].mlp.eval(),
        ),
        (
            r"HF-ASSERT hf_code: the installed code .* differs from the pin",
            lambda f: rt.spec.model.hf_code.update(torch="0.0.0"),
        ),
    ]
    try:
        for pattern, force in cases:
            full = fresh()
            force(full)
            with pytest.raises((RuntimeError, ValueError), match=pattern):
                rt._hf_start_asserts(full)
            _restore_determinism()
            rt.spec.model.hf_code = copy.deepcopy(pinned)
        rt._hf_start_asserts(fresh())  # every case undone: passes again
    finally:
        _restore_determinism()
    # The stage's start runs them: an hf_code pin the installed code does not hold.
    other = copy.deepcopy(spec)
    other.model.hf_code = dict(pinned, transformers_record_sha256="0" * 64)
    with pytest.raises(RuntimeError, match=r"HF-ASSERT hf_code"):
        StageRuntime(other, 1)


def test_m4_wire_stock_config_use_cache_is_pinned_by_the_build(hf_env):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the HF build keeps a checkpoint config.json's use_cache true (every
    # stock config.json says true, and fleetplan adopts it unchanged), so the stage start
    # refuses every real-checkpoint run with HF-ASSERT use_cache; or the build pins it
    # only on a model the stage does not assert on.
    from wantrain.model import hfm
    from wantrain.model.gpt import GPTConfig

    hc = hfm.tiny_config("llama")
    assert hc["use_cache"] is True  # the stock default, as a checkpoint's config.json
    model = GPTConfig(
        arch="hf",
        hf_config=hc,
        hf_buffers=hfm.buffer_record(hc),
        hf_census=hfm.unit_census(hc),
        hf_code=hfm.hf_code(),
        seq_len=16,
        **hfm.config_fields(hc),
    )
    spec = JobSpec(model=model, stages=2, partition=[[0, 3], [3, 6]], microbatches=2, mb_size=2)
    rt = StageRuntime(spec, 0)  # runs the build and the start asserts
    assert rt.hf and hf_env == ["guard"]
    assert spec.model.hf_config["use_cache"] is True  # the run's pinned config is not edited


def _tied_head(fam="gemma2", rank=1, partition=((0, 3), (3, 6))):
    rt = StageRuntime(_wire_spec(fam, [list(p) for p in partition]), rank)
    rt._tied_plan(0)
    return rt


def _read_through(rt, src, chunks):
    """The stage's real reader thread on a socketpair, fed `chunks` (frame bytes) then END;
    returns when the reader has returned."""
    import socket

    from wantrain.transport import framing as fr
    from wantrain.transport.shim import DelayedDelivery, LinkSpec

    a, b = socket.socketpair()
    t = threading.Thread(
        target=rt._reader, args=(a, DelayedDelivery(LinkSpec(), rt.inbox.put), src), daemon=True
    )
    t.start()
    try:
        for c in chunks:
            b.sendall(c)
        b.sendall(_frame_bytes(fr.END, 0, 0, b"", codec=fr.OBJ))
        t.join(30)
        assert not t.is_alive()
    finally:
        a.close()
        b.close()


def test_m4_wire_tied_receipt_is_validated_before_use(hf_env):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: a TIED frame (framing.TIED) out of sequence, misshapen, duplicated, from
    # the wrong side or missing at END is used or passes at an end of the relay, a stage of a
    # plan with no tied relay (untied, or the tie on one stage) accepts one, a refused frame
    # reaches the row fold, or a valid one does not.
    from wantrain.transport import framing as fr

    prev, nxt = ("prev", 0), ("next", 0)
    head = _tied_head()
    assert head._tied_role == "head" and head.spec.tied_relay()
    order = head._tseq["fwd"].expected
    (s0, i0), (_, i1) = order[:2]
    assert s0 == 0 and len(order) == 6 and {i0, i1} == {0, 1}

    def frame(rt, step, mb, n_extra=0, seed=0, fwd=True):
        ids = rt._tied_ids(step, mb) if fwd else rt._tied_R(step)[mb]
        rows = _rows(rt, ids.numel() + n_extra, seed)
        return rows, _parse(_tied_bytes(step, mb, rows)[0])

    good, f = frame(head, 0, i0, seed=1)
    head._on_tied(f, prev)
    assert torch.equal(head._tfold[0].embeds[i0], good)  # held for the row fold
    with pytest.raises(RuntimeError, match=rf"TIED \(step 0, mb {i0}\) duplicated"):
        head._on_tied(frame(head, 0, i0, seed=1)[1], prev)
    with pytest.raises(RuntimeError, match=rf"TIED \(step 1, mb {i0}\) out of sequence"):
        head._on_tied(frame(head, 1, i0)[1], prev)
    n1 = head._tied_ids(0, i1).numel()
    with pytest.raises(
        RuntimeError, match=rf"\(step 0, mb {i1}\): codec 0 shape \({n1 + 1}, 64\), want RAW_F32"
    ):
        head._on_tied(frame(head, 0, i1, n_extra=1)[1], prev)
    assert i1 not in head._tfold[0].embeds  # refused before use
    with pytest.raises(
        RuntimeError, match=r"refused TIED frame \(step 0, mb \d\) from \('next', 0\)"
    ):
        _tied_head()._on_tied(frame(head, 0, i0)[1], nxt)
    # Missing: one fragmented frame of six, then END, through the stage's reader thread.
    head = _tied_head()
    rows, _ = frame(head, 0, i0, seed=2)
    _read_through(head, prev, _tied_bytes(0, i0, rows, cuts=(100, 1000)))
    with pytest.raises(RuntimeError, match=r"TIED frames missing from \('prev', 0\) at its END"):
        head._pump_inbox(block=True, timeout=10)
    assert torch.equal(head._tfold[0].embeds[i0], rows)  # the whole frame came first, exact
    # Stage 0: the backward rows R_k start at step 1 of a fresh launch.
    emb = _tied_head(rank=0)
    assert emb._tied_role == "embed"
    with pytest.raises(RuntimeError, match=r"TIED \(step 0, mb 0\) is no frame of this attempt"):
        emb._on_tied(frame(emb, 0, 0, fwd=False)[1], nxt)
    r0, f = frame(emb, 1, 0, seed=3, fwd=False)
    emb._on_tied(f, nxt)
    ids, held = emb._tied_rows[1][0]
    assert torch.equal(ids, emb._tied_R(1)[0]) and torch.equal(held, r0)
    with pytest.raises(RuntimeError, match=r"\(step 1, mb 1\): codec 0 shape"):
        emb._on_tied(frame(emb, 1, 1, n_extra=2, fwd=False)[1], nxt)
    assert 1 not in emb._tied_rows[1]
    # No relay: an untied hf plan, and a tie on one stage, refuse any TIED at receipt.
    for fam, part, rank in (("llama", ((0, 3), (3, 6)), 1), ("gemma2", ((0, 6),), 0)):
        rt = StageRuntime(_wire_spec(fam, [list(p) for p in part]), rank)
        assert not rt.spec.tied_relay() and not rt._relay
        _read_through(rt, prev, _tied_bytes(0, 0, _rows(rt, 3, 4), cuts=(64,)))
        with pytest.raises(RuntimeError, match=rf"no tied relay at stage {rank}"):
            rt._pump_inbox(block=True, timeout=10)
    assert fr.TIED == 7


def test_m4_wire_tied_sparse_on_a_crafted_embedding_gradient(hf_env):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: stage 0 sends TIED(t, i) for an embedding gradient e_i holding a -0 (or
    # any bit) on a row outside T_i, or sends rows other than e_i[T_i] in ascending id, or a
    # refused e_i still leaves a frame on the next queue.
    from wantrain.transport import framing as fr

    emb = _tied_head(rank=0)
    q = queue_mod.Queue()
    emb.send_qs = {("next", 0): q}
    ids = emb._tied_ids(0, 1)
    V, d = emb._tied.shape
    e = torch.zeros(V, d)
    e[ids] = torch.randn(ids.numel(), d, generator=torch.Generator().manual_seed(5))
    outside = next(r for r in range(V) if r not in set(ids.tolist()))
    bad = e.clone()
    bad[outside, 3] = -0.0
    with pytest.raises(RuntimeError, match=r"TIED-SPARSE \(step 0, mb 1\)"):
        emb._send_tied_embed(0, 1, bad)
    assert q.empty()
    emb._send_tied_embed(0, 1, e)
    ftype, step, mb, chunk, rows, codec = q.get_nowait()
    assert (ftype, step, mb, chunk, codec) == (fr.TIED, 0, 1, 0, "raw")
    assert torch.equal(rows, e[ids]) and not torch.equal(rows, e[ids.flip(0)])


def test_m4_wire_rollback_refuses_a_split_tied_plan(hf_env, monkeypatch):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: a rolled-back relay (runtime-7a, runtime-7b or scheduler-6 in
    # rollback.ROLLED_BACK) still finalizes a split tied hf plan, reports tied_relay(), or a
    # stage sends or accepts TIED; or the rollback is read at import instead of at call
    # time; or it touches an untied hf plan or a tie on one stage; or, not rolled back, the
    # split plan does not finalize.
    from wantrain.model import rollback

    split = [[0, 3], [3, 6]]
    assert _wire_spec("gemma2", split).finalize().tied_relay()  # ROLLED_BACK empty
    head = StageRuntime(_wire_spec("gemma2", split), 1)
    head._tied_plan(0)
    ids = head._tied_ids(0, 0)
    data = _tied_bytes(0, 0, _rows(head, ids.numel(), 6))[0]
    for piece in ("runtime-7a", "runtime-7b", "scheduler-6"):
        monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset({piece}))
        with pytest.raises(
            ValueError, match=rf"piece '{piece}' is rolled back to 'tied-split-refused'"
        ):
            _wire_spec("gemma2", split).finalize()
        assert not _wire_spec("gemma2", split).tied_relay()
        for spec in (_wire_spec("llama", split), _wire_spec("gemma2", [[0, 6]])):
            assert not spec.finalize().tied_relay()  # unaffected
        with pytest.raises(RuntimeError, match=rf"refused TIED frame \(step 0, mb 0\).*'{piece}'"):
            head._on_tied(_parse(data), ("prev", 0))
    monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset())
    head._on_tied(_parse(data), ("prev", 0))  # live again: accepted
    assert 0 in head._tfold[0].embeds


def test_m4_wire_tied_forward_cut_through_at_a_middle_stage(hf_env):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: a middle stage reassembles, reorders or drops a TIED fragment, waits for
    # its own F (or any main-loop work) before forwarding one, changes a byte of a fragment's
    # header or payload, or forwards an out-of-sequence or misshapen fragment instead of
    # failing at the middle stage naming (step, mb).
    import select
    import socket

    from wantrain.runtime.stage import _RawFrag
    from wantrain.transport import framing as fr
    from wantrain.transport.shim import LinkSpec

    mid = StageRuntime(_wire_spec("gemma2", [[0, 1], [1, 5], [5, 6]]), 1)
    assert mid._tied_role == "mid" and mid._relay and _RawFrag
    mid._tied_plan(0)
    up_a, up_b = socket.socketpair()  # stage 0 -> mid: the mid stage reads up_a
    dn_a, dn_b = socket.socketpair()  # mid -> Head: the mid stage writes dn_a
    for s in (up_b, dn_b):
        s.settimeout(30)
    mid._wire(("prev", 0), up_a, LinkSpec())
    mid._wire(("next", 0), dn_a, LinkSpec())
    rng = random.Random(8)

    def fragments(step, mb, fwd, seed):
        ids = mid._tied_ids(step, mb) if fwd else mid._tied_R(step)[mb]
        rows = _rows(mid, ids.numel(), seed)
        size = rows.numel() * 4
        cuts = sorted(rng.sample(range(1, size), rng.randrange(1, 6)))
        return _tied_bytes(step, mb, rows, cuts=cuts)

    def recv_exact(sock, n):
        buf = bytearray()
        while len(buf) < n:
            got = sock.recv(n - len(buf))
            assert got, "the middle stage closed the link"
            buf += got
        return bytes(buf)

    def idle(sock):
        return not select.select([sock], [], [], 0.3)[0]

    try:
        # Forward: the fragments of step 0's two microbatches, interleaved at random, with
        # FWD of microbatch 0 ahead of them, which no F ever consumes in this test.
        (_, i0), (_, i1) = mid._tseq["fwd"].expected[:2]
        frags = fragments(0, i0, True, 1) + fragments(0, i1, True, 2)
        rng.shuffle(frags)
        x = torch.randn(2, 16, 64, generator=torch.Generator().manual_seed(3))
        up_b.sendall(_frame_bytes(fr.FWD, 0, i0, x) + b"".join(frags))
        assert recv_exact(dn_b, sum(map(len, frags))) == b"".join(frags)
        assert idle(dn_b) and mid.fwd_ready == {}  # the main loop never ran
        kinds = [mid.inbox.get_nowait()[1].type for _ in range(mid.inbox.qsize())]
        assert kinds == [fr.FWD]  # no TIED fragment reached the main loop
        # Backward: R_k of step 1 from the Head, toward stage 0.
        back = fragments(1, 0, False, 4) + fragments(1, 1, False, 5)
        dn_b.sendall(b"".join(back))
        assert recv_exact(up_b, sum(map(len, back))) == b"".join(back)
        assert idle(up_b) and mid.inbox.empty()
        # Out of sequence: a fragment of step 2 while step 1's forward frames are due.
        late = fragments(2, i0, True, 6)[0]
        up_b.sendall(late)
        with pytest.raises(
            RuntimeError,
            match=rf"refused TIED \(step 2, mb {i0}\) from \('prev', 0\).*out of sequence",
        ):
            mid._pump_inbox(block=True, timeout=10)
        assert idle(dn_b)  # refused: not forwarded
        # Misshapen: R_0 of step 2 with one row too many (framing accepts it, as a fragment).
        n = mid._tied_R(2)[0].numel()
        bad = _tied_bytes(2, 0, _rows(mid, n + 1, 7), cuts=(16,))[0]
        dn_b.sendall(bad)
        with pytest.raises(
            RuntimeError,
            match=rf"refused TIED \(step 2, mb 0\) from \('next', 0\).*shape \({n + 1}, 64\)",
        ):
            mid._pump_inbox(block=True, timeout=10)
        assert idle(up_b)
    finally:
        for key in (("prev", 0), ("next", 0)):
            mid.send_qs[key].put(None)
        for t in mid._send_threads:
            t.join(10)
        for s in (up_a, up_b, dn_a, dn_b):
            s.close()


def test_m4_wire_rollback_at_a_middle_stage(hf_env, monkeypatch):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: a middle stage of a rolled-back relay (runtime-7b) still forwards a TIED
    # fragment, or reads the rollback once at start instead of at receipt.
    from wantrain.model import rollback
    from wantrain.runtime.stage import _RawFrag

    mid = StageRuntime(_wire_spec("gemma2", [[0, 1], [1, 5], [5, 6]]), 1)
    mid._tied_plan(0)
    q = queue_mod.Queue()
    mid.send_qs = {("next", 0): q}
    (_, i0) = mid._tseq["fwd"].expected[0]
    f = _parse(_tied_bytes(0, i0, _rows(mid, mid._tied_ids(0, i0).numel(), 9))[0])
    monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset({"runtime-7b"}))
    assert mid._forward_tied(f, ("prev", 0)) is False and q.empty()
    kind, msg = mid.inbox.get_nowait()
    assert kind == "fatal" and re.search(rf"refused TIED \(step 0, mb {i0}\).*'runtime-7b'", msg)
    monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset())
    assert mid._forward_tied(f, ("prev", 0)) is True
    item = q.get_nowait()
    assert isinstance(item[4], _RawFrag) and item[4].frame is f


def _resume_spec(spec, **kw):
    s = copy.deepcopy(spec)
    s.resume, s.resume_step = True, 2
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _edit_buffer(d) -> str:
    """Flip one bit of one tensor in a wave's hf_buffers.json, keep the file self-consistent
    (its sha256 entry follows the bytes) and re-list it in wave.json: only the pin can
    refuse it. -> the tensor's name."""
    import base64
    import hashlib

    from wantrain.fault_tolerance.checkpoint import sha256_file

    with open(d + "hf_buffers.json") as f:
        rec = json.load(f)
    name = sorted(rec["tensors"])[-1]
    ent = rec["tensors"][name]
    raw = bytearray(base64.b64decode(ent["b64"]))
    raw[0] ^= 1
    ent["b64"] = base64.b64encode(bytes(raw)).decode()
    ent["sha256"] = hashlib.sha256(raw).hexdigest()
    with open(d + "hf_buffers.json", "w") as f:
        f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")))
    with open(d + "wave.json") as f:
        m = json.load(f)
    for e in m["files"]:
        if e["name"] == "hf_buffers.json":
            e["sha256"] = sha256_file(d + "hf_buffers.json")
            e["bytes"] = os.path.getsize(d + "hf_buffers.json")
    with open(d + "wave.json", "w") as f:
        json.dump(m, f)
    return name


@pytest.mark.parametrize("kind", ["whole", "logical"])
def test_m4_wire_buffer_pin_refuses_an_edited_tensor(hf_env, tmp_path, kind):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: an hf wave committed through commit_wave(..., extra_files) does not list
    # hf_buffers.json (the pinned record's canonical bytes) after state.pt, or a resume from a
    # whole-tensor or logical wave whose record holds one edited tensor applies any state
    # instead of refusing naming the tensor, or an unedited wave is refused.
    from wantrain.fault_tolerance.checkpoint import validate_wave, wave_dir
    from wantrain.runtime.stage import hf_buffers_bytes

    extra = dict(wire="native", shard="pieces") if kind == "logical" else {}
    spec = _wire_spec("llama", [[0, 6]], out_dir=str(tmp_path), tag="t", steps=4, **extra)
    rt = StageRuntime(spec, 0)
    if kind == "logical":  # the AdamW state a step leaves, so the logical payload exists
        for n in rt.pnames:
            rt.master[n].grad = torch.zeros_like(rt.master[n])
        rt.popt.step()
    rt.losses = {(s, mb): 1.0 + s + mb / 4 for s in range(3) for mb in range(2)}
    _commit(rt, 2)
    d = wave_dir(str(tmp_path), "t", 0, 0, 2)
    assert [e["name"] for e in validate_wave(d, 2)["files"]] == ["state.pt", "hf_buffers.json"]
    with open(d + "hf_buffers.json", "rb") as f:
        assert f.read() == hf_buffers_bytes(spec.model)
    state = torch.load(d + "state.pt", map_location="cpu", weights_only=True)
    assert ("pieces" in state) == (kind == "logical")
    assert StageRuntime(_resume_spec(spec), 0)._resume() == 3  # control: the wave as written
    name = _edit_buffer(d)
    rt = StageRuntime(_resume_spec(spec), 0)
    before = [p.detach().clone() for p in rt.chunks[0].parameters()]
    with pytest.raises(
        RuntimeError, match=rf"BUFFER-PIN {re.escape(name)}: its bytes in .* differ"
    ):
        rt._resume()
    assert all(torch.equal(a, b) for a, b in zip(before, rt.chunks[0].parameters()))
    # The same edit without re-listing it: the wave itself is refused, naming the file.
    with open(d + "hf_buffers.json", "a") as f:
        f.write(" ")
    with pytest.raises(RuntimeError, match=r"file hf_buffers\.json has"):
        StageRuntime(_resume_spec(spec), 0)._resume()


def test_m4_wire_resume_refuses_another_device_class_or_hf_code(hf_env, tmp_path):
    # covers: RUNTIME-I15, RUNTIME-I19
    # WOULD-FAIL-IF: an hf resume onto a device_classes map where a slot moved class is not
    # refused naming that slot (its own or another's), the same map is refused, or a wave
    # written under another hf_code (transformers or torch code identity) resumes.
    from wantrain.model import hfm

    here = hfm.device_class("cpu")
    classes = [[here], [here]]
    spec = _wire_spec(
        "llama", [[0, 3], [3, 6]], out_dir=str(tmp_path), tag="t", steps=4, device_classes=classes
    )
    _commit(StageRuntime(spec, 1), 2)
    assert StageRuntime(_resume_spec(spec), 1)._resume() == 3
    # CONTROL-I25: a map of two classes neither launches nor resumes.
    with pytest.raises(ValueError, match="^MIXED-FLEET: the device map spans 2 device classes"):
        StageRuntime(_resume_spec(spec, device_classes=[[here], ["cuda:NVIDIA H100"]]), 1)
    for cls in ("cuda:NVIDIA H100", "mps:Apple M9"):
        moved = [[cls], [cls]]  # the whole one-class map moved: the first slot is named
        rt = StageRuntime(_resume_spec(spec, device_classes=moved), 1)
        with pytest.raises(
            RuntimeError, match=rf"slot r0p0 \('{re.escape(here)}' -> '{cls}'\) moved class"
        ):
            rt._resume()
    # A wave written under another code identity (the pin of an older transformers).
    code = copy.deepcopy(spec.model.hf_code)
    old = _wire_spec(
        "llama", [[0, 3], [3, 6]], out_dir=str(tmp_path), tag="c", steps=4, device_classes=classes
    )
    rt = StageRuntime(copy.deepcopy(old), 1)
    rt.spec.model.hf_code = dict(code, transformers="5.18.0")
    _commit(rt, 2)
    with pytest.raises(RuntimeError, match=r"pin 'model\.hf_code\.transformers' differs"):
        StageRuntime(_resume_spec(old), 1)._resume()


def test_m4_wire_jobspec_hf_relay_widths_and_pins():
    # covers: RUNTIME-I19, RUNTIME-I15
    # WOULD-FAIL-IF: finalize admits a split tied hf table with stage 0, the Head or a middle
    # stage at width > 1 (or does not name that stage), refuses a wide stage outside the
    # relay; tied_relay() is true for anything but a split tied hf plan; an hf model's pins
    # lack device_classes or its hf_* fields; or a dense model's pins (device_classes set or
    # not) leave the pre-M4 bytes.
    from wantrain.model.gpt import GPTConfig

    part3 = [[0, 1], [1, 5], [5, 6]]
    for widths, stage in (([2, 1, 1], 0), ([1, 2, 1], 1), ([1, 1, 2], 2)):
        with pytest.raises(
            ValueError, match=rf"split across stages 0 and 2 .*: stage {stage} has width 2"
        ):
            _wire_spec("gemma2", part3, stage_widths=widths, microbatches=4).finalize()
    assert _wire_spec("llama", part3, stage_widths=[2, 2, 2], microbatches=4).finalize()
    assert _wire_spec("gemma2", part3).finalize().tied_relay()
    assert _wire_spec("llama_tied", [[0, 3], [3, 6]]).finalize().tied_relay()
    assert not _wire_spec("gemma2", [[0, 6]]).finalize().tied_relay()  # the tie on one stage
    assert not _wire_spec("llama", part3).finalize().tied_relay()  # untied
    dense = GPTConfig(n_layer=2, tie_embeddings=True, init_scheme="gpt2")
    assert not JobSpec(model=dense, stages=1, microbatches=2).finalize().tied_relay()
    # Pins: hf adds device_classes and carries every hf_* field; dense keeps its bytes.
    assert json.dumps(JobSpec().finalize().pins()) == _GOLDEN_DENSE_PINS
    with pytest.raises(ValueError, match="^MIXED-FLEET: the device map spans 2 device classes"):
        JobSpec(device_classes=[["cpu:arm64:a"], ["cpu:arm64:b"]]).finalize()  # CONTROL-I25
    classes = [["cpu:arm64:a"], ["cpu:arm64:a"]]
    assert json.dumps(JobSpec(device_classes=classes).finalize().pins()) == _GOLDEN_DENSE_PINS
    hf = _wire_spec("llama", [[0, 3], [3, 6]], device_classes=classes).finalize()
    p = hf.pins()
    assert list(p) == [*json.loads(_GOLDEN_DENSE_PINS), "device_classes"]
    assert p["device_classes"] == classes and p["model"]["hf_code"] == hf.model.hf_code
    assert {"hf_config", "hf_buffers", "hf_census", "hf_code", "hf_impl"} <= set(p["model"])
    assert not any(k.startswith("moe_") for k in p["model"])
    other = _wire_spec("llama", [[0, 3], [3, 6]], device_classes=classes).finalize()
    other.model.hf_code = dict(hf.model.hf_code, torch="0.0.0")
    from wantrain.runtime.spec import first_pin_difference

    assert first_pin_difference(p, other.pins()) == "model.hf_code.torch"


def _unit_shapes(fam, partition):
    """Per stage, {"<unit in chunk>.<HF name>": shape} of the skeleton's units, a tensor
    shared by two units at its first (the census's rule), and whether that equals the
    stage chunk's own named_parameters (the names a pieces stage uses)."""
    from wantrain.model import hfm

    hc = _wire_hf(fam)[0]
    us = hfm.units(hfm.skeleton(hc), hfm.unit_map(hc))
    seen: set = set()
    out, own = [], []
    for a, b in partition:
        shapes = {}
        for li, u in enumerate(us[a:b]):
            for k, p in u.named_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    shapes[f"{li}.{k}"] = tuple(p.shape)
        out.append(shapes)
        own.append({n: tuple(p.shape) for n, p in torch.nn.ModuleList(us[a:b]).named_parameters()})
    return out, own, len(us)


@pytest.mark.parametrize("fam", ["llama", "llama_tied", "gemma2"])
def test_m4_wire_stage_param_shapes_of_hf_equal_the_units(fam):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: stage_param_shapes for arch 'hf' (read from the census's param_shapes)
    # names or shapes a stage's tensors otherwise than the skeleton's units do, counts a tied
    # table twice, or (without a split tie) differs from the stage chunk's named_parameters.
    n = len(_wire_hf(fam)[2])
    for part in ([[0, n]], [[0, 1], [1, n - 1], [n - 1, n]], [[0, 2], [2, n]]):
        want, own, n_units = _unit_shapes(fam, part)
        assert n_units == n
        got = _wire_spec(fam, part).stage_param_shapes()
        assert got == want, (fam, part)
        split_tie = fam != "llama" and len(part) > 1
        if not split_tie:
            assert got == own
        else:  # the table once, at the Embed's stage
            assert sum("embed_tokens" in k or "lm_head" in k for s in got for k in s) == 1


def test_m4_wire_hf_finalize_with_piece_bounds_imports_no_transformers():
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: finalizing an hf spec with piece_bounds (default schedule from the
    # census, the bounds checked against stage_param_shapes) imports transformers or
    # wantrain.model.hfm (BA-441, BA-419), or its shapes differ from the skeleton's units.
    import dataclasses
    import subprocess

    from wantrain.runtime import pieces

    part = [[0, 3], [3, 6]]
    want, _, _ = _unit_shapes("llama", part)
    name = "0.model.layers.2.self_attn.q_proj.weight"
    shape = want[1][name]
    q, numel = pieces.quantum(shape), math.prod(shape)
    bounds = {name: [0, q * (numel // q // 2), numel]}
    code = """
import importlib.abc, json, sys
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "transformers" or name.startswith("transformers.") or name == "wantrain.model.hfm":
            raise ImportError("refused import of " + name)
        return None
sys.meta_path.insert(0, Refuse())
from wantrain.model.gpt import GPTConfig
from wantrain.runtime.spec import JobSpec
args = json.loads(sys.stdin.read())
spec = JobSpec(model=GPTConfig(**args["model"]), stages=2, stage_widths=[1, 2],
               partition=args["part"], microbatches=4, mb_size=2, wire="native",
               shard="pieces", stream_grads=True, piece_bounds=[None, args["bounds"]])
spec.finalize()
print(json.dumps([{k: list(v) for k, v in s.items()} for s in spec.stage_param_shapes()]))
print(json.dumps(sorted(n for n in sys.modules if n.split(".")[0] == "transformers" or n == "wantrain.model.hfm")))
"""
    args = {"model": dataclasses.asdict(_wire_cfg("llama")), "part": part, "bounds": bounds}
    out = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(args),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    shapes, loaded = out.stdout.strip().splitlines()
    assert json.loads(shapes) == [{k: list(v) for k, v in s.items()} for s in want]
    assert json.loads(loaded) == []
    # The bounds are really checked there: off the quantum is refused naming the stage.
    bad = dict(args, bounds={name: [0, 1, numel]})
    out = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(bad),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode != 0 and "piece_bounds[1]" in out.stderr


def _sha_tensor(t):
    import hashlib

    return hashlib.sha256(
        t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def test_m4_wire_autocast_and_the_checkpoint_span_build(hf_env, tmp_path):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: the engine leaves an hf model with autocast without its autocast region
    # (or opens one for fp32 or for a built-in model), or a stage built from an HF checkpoint
    # materializes a unit outside its span, leaves one of its own on meta, or loads bytes
    # other than a full load's.
    from wantrain.model import hf, hfm
    from wantrain.model.gpt import GPTConfig, build_layers
    from wantrain.runtime.stage import compute_autocast, engine_supplies_autocast

    bf16 = _wire_cfg("llama", compute_dtype="bf16")
    assert engine_supplies_autocast(bf16)
    assert isinstance(compute_autocast(bf16, "cpu"), torch.autocast)
    assert not engine_supplies_autocast(_wire_cfg("llama"))  # fp32
    assert not engine_supplies_autocast(GPTConfig(compute_dtype="bf16"))  # its own forward
    hc = _wire_hf("llama")[0]
    root = hfm._full_build(hc, 7, swap=False)
    src = {k: p.detach().clone() for k, p in root.named_parameters()}
    path = str(tmp_path / "ckpt")
    hfm.write_safetensors(src, path, 40_000)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(hc, f)
    kw = dict(hf_model=path, hf_model_sha256=hf.checkpoint_digest(path))
    full = build_layers(_wire_cfg("llama", **kw), 0)  # the full load (no span)
    full_sha = {k: _sha_tensor(p) for u in full for k, p in u.named_parameters()}
    assert full_sha == {k: _sha_tensor(t) for k, t in src.items()}
    part = [[0, 2], [2, 4], [4, 6]]
    for r in range(3):
        rt = StageRuntime(_wire_spec("llama", part, model_kw=kw), r)
        mine = {k: p for u in rt.chunks[r] for k, p in u.named_parameters()}
        assert mine and all(p.device.type == "cpu" for p in mine.values())
        assert {k: _sha_tensor(p) for k, p in mine.items()} == {k: full_sha[k] for k in mine}
        tree = rt.chunks[r][0]._rt.root
        assert {k for k, p in tree.named_parameters() if p.device.type != "meta"} == set(mine)


# ------------------------------------------------- MoE on real blocks (RUNTIME-I18)


def _moe_model(**kw):
    from wantrain.model.gpt import GPTConfig

    base = dict(
        vocab_size=32,
        n_layer=3,
        n_head=2,
        d_model=16,
        seq_len=6,
        norm="rmsnorm",
        mlp="swiglu",
        bias=False,
        pos="rope",
        moe_experts=4,
        moe_top_k=2,
        moe_d_ff=8,
        moe_bias_rate=0.01,
        moe_select="score_bias",
    )
    base.update(kw)
    return GPTConfig(**base)


def test_m4_wire_moe_records_once_and_a_perturbed_rebuild_raises(tmp_path):
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: a stage's F records an MoE layer's aux hex and counts other than once
    # per (step, mb, layer) of the microbatches it owns, a recompute rebuild of real MoE
    # blocks differs from F without MOE-REPLAY naming (step, mb, layer), or an unchanged
    # rebuild is refused.
    spec = JobSpec(
        model=_moe_model(),
        stages=1,
        microbatches=2,
        mb_size=2,
        recompute=True,
        out_dir=str(tmp_path),
    ).finalize()
    rt = StageRuntime(spec, 0)
    assert rt.moe and rt._moe_layers == [1, 2, 3]
    for mb in rt.my_mbs:
        rt._forward(0, mb, 0)
    assert [r[:3] for r in rt.moe_aux] == [[0, mb, layer] for mb in (0, 1) for layer in (1, 2, 3)]
    assert len({r[3] for r in rt.moe_aux}) == 6  # distinct hex per record
    for mb in (0, 1):
        for layer, vec in rt._moe_counts[0][mb].items():
            assert vec.dtype == torch.int64 and int(vec.sum()) == 2 * 6 * 2, (mb, layer)
    rt._rebuild(0, 0, 0)  # control: the replay equals F
    last = rt._bias_mods[-1]
    with torch.no_grad():
        last.router.weight.mul_(-1.0)
    with pytest.raises(
        RuntimeError, match=rf"MOE-REPLAY \(step 0, mb 1, layer {int(last.moe_layer)}\)"
    ):
        rt._rebuild(0, 1, 0)
    again = StageRuntime(spec, 0)
    again._forward(0, 0, 0)
    with pytest.raises(RuntimeError, match=r"MOE-RECORD \(step 0, mb 0\): recorded twice"):
        again._forward(0, 0, 0)


def test_m4_wire_moe_exchange_units_one_per_projection():
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: pieces.exchange_units cuts a native MoE chunk otherwise than one unit
    # per expert projection (every expert, in expert order), puts the selection bias outside
    # its block's small-tensor unit, or changes a dense chunk's units.
    from wantrain.model.gpt import build_layers
    from wantrain.runtime import pieces as pc

    layers = build_layers(_moe_model(), 0)
    units = pc.exchange_units(torch.nn.ModuleList(list(layers)[1:3]))
    names = [[n for n, _ in u] for u in units]
    for li in (0, 1):
        for proj in ("gate", "up", "down"):
            want = [f"{li}.mlp.experts.{e}.{proj}.weight" for e in range(4)]
            assert names.count(want) == 1, (li, proj)
        small = next(u for u in names if f"{li}.ln1.weight" in u)
        assert small[-1] == f"{li}.mlp.sel_bias" and f"{li}.mlp.router.weight" in small
    assert sum(1 for u in names if any(".experts." in n for n in u)) == 6
    dense = build_layers(_moe_model(moe_experts=0, moe_bias_rate=0.0, moe_select="score"), 0)
    dn = [[n for n, _ in u] for u in pc.exchange_units(torch.nn.ModuleList(list(dense)[1:3]))]
    assert not any("sel_bias" in n or "experts" in n for u in dn for n in u)
    assert len(dn) == 2 * 6  # small tensors, qkv, proj, then the dense MLP's three linears


def _moe_wide(peer, tmp_path, **kw):
    """Peer `peer` of a width-2 ZeRO-1 stage of real MoE blocks (stage 1 of [1, 2], the
    final stage) after real F, B and W of the microbatches it owns, its sends captured."""
    from wantrain.runtime import exchange as xc

    spec = JobSpec(
        model=_moe_model(),
        stages=2,
        stage_widths=[1, 2],
        microbatches=4,
        mb_size=2,
        partition=[[0, 2], [2, 5]],
        out_dir=str(tmp_path),
        **kw,
    ).finalize()
    rt = StageRuntime(spec, 1, peer)
    rt._exec_step = 0
    native = spec.wire == "native"
    rt.send_qs = {
        ("intra", 1 - peer): xc.NeedFirstQueue() if native else queue_mod.Queue(),
        ("prev", 0): queue_mod.Queue(),
    }
    g = torch.Generator().manual_seed(11)
    xs = [torch.randn(2, 6, 16, generator=g) for _ in range(4)]
    for mb in rt.my_mbs:  # the same boundary inputs on both peers
        rt.fwd_ready[(0, mb, 1)] = xs[mb]
        rt._forward(0, mb, 1)
    for mb in rt.my_mbs:
        rt._backward_final(0, mb, 1)
    for *_, w in rt.deferred_w:
        w()
    rt.deferred_w = []
    return rt


def _drain(q) -> list:
    out = []
    while True:
        try:
            out.append(q.pop(timeout=0.01) if hasattr(q, "pop") else q.get_nowait())
        except queue_mod.Empty:
            return out


@pytest.mark.parametrize("path", ["batch", "stream", "native"])
def test_m4_wire_moe_zero1_counts_from_real_blocks(tmp_path, path):
    # covers: RUNTIME-I18
    # WOULD-FAIL-IF: a ZeRO-1 MoE stage's ownership list is not its trainable parameters
    # then its selection biases in layer order, the count payload a gradient path builds from
    # a real MoE chunk's records (batch dict, stream ('mb', ...) tuple, wire-native COUNT
    # frame) does not fold at the bias owner into the counts F recorded, or a vector dropped
    # or sent twice is not fatal.
    from wantrain.runtime import exchange as xc
    from wantrain.transport import framing as fr
    from wantrain.transport import serial

    kw = {
        "batch": {},
        "stream": {"stream_grads": True},
        "native": {"stream_grads": True, "wire": "native"},
    }[path]
    owner, sender = _moe_wide(0, tmp_path, **kw), _moe_wide(1, tmp_path, **kw)
    n = len(owner.params)
    assert owner._own[:n] == owner.params and owner._own[n:] == owner._biases
    assert [b is m.sel_bias for b, m in zip(owner._biases, owner._bias_mods)] == [True, True]
    assert owner._my_bias == [j for j in (0, 1) if (n + j) % 2 == 0] != sender._my_bias
    layer = int(owner._bias_mods[owner._my_bias[0]].moe_layer)
    recorded = {**owner._moe_counts[0], **sender._moe_counts[0]}
    assert sorted(recorded) == [0, 1, 2, 3]
    want = sum((recorded[mb][layer] for mb in range(4)), torch.zeros(4, dtype=torch.int64))
    assert int(want.sum()) == 4 * 2 * 6 * 2 and len(set(want.tolist())) > 1

    def frame(mb):
        return SimpleNamespace(type=fr.GRADS, step=0, mb=mb)

    def wire(msg):
        return serial.loads(serial.dumps(msg))  # the OBJ codec of the GRADS frame

    if path == "batch":
        # the message _optimizer_step_wide ships to owner 0 at the step barrier
        msg = {
            mb: [t.detach().cpu() for t in sender.grads[0][(mb, 1)][0::2]] for mb in sender.my_mbs
        }
        msg["counts"] = {mb: sender._counts_for(0, mb, 0) for mb in sender.my_mbs}
        sends = [lambda o: o._on_legacy("GRADS", frame(0), wire(msg), 1)]
        mb0 = sender.my_mbs[0]
        short = dict(msg, counts={mb0: msg["counts"][mb0]})
        dropped = [lambda o: o._on_legacy("GRADS", frame(0), wire(short), 1)]
        twice = sends[0]
    else:
        items = [it for it in _drain(sender.send_qs[("intra", 0)]) if it[0] == fr.GRADS]
        if path == "stream":
            assert [(it[4][0], len(it[4])) for it in items] == [("mb", 4)] * 2
            sends = [
                lambda o, it=it: o._on_legacy("GRADS", frame(it[2]), wire(it[4]), 1) for it in items
            ]
            dropped, twice = sends[:-1], sends[-1]
        else:

            def native(it):
                return wire(("x", it[4].env, it[4].fut.result()))

            sends = [
                lambda o, it=it: o._on_exchange(frame(it[2]), native(it), ("intra", 1))
                for it in items
            ]
            is_count = [it[4].env["u"] >= n for it in items]
            assert sum(is_count) == 2  # one COUNT frame per (mb, bias) shipped to peer 0
            last = max(i for i, c in enumerate(is_count) if c)
            dropped, twice = sends[:last] + sends[last + 1 :], sends[last]

    def fold(deliveries):
        o = _moe_wide(0, tmp_path, **kw)
        if path == "batch":  # the owner's own vectors, as _optimizer_step_wide adds them
            for mb in o.my_mbs:
                o._count_add(0, mb, o._counts_for(0, mb, 0), 0)
        for d in deliveries:
            d(o)
        return o

    got = fold(sends)._count_fold(0).result()
    assert list(got) == [layer] and torch.equal(got[layer], want)
    with pytest.raises(RuntimeError, match=r"count vectors missing|missing at the owner"):
        fold(dropped)._count_fold(0).result()
    with pytest.raises((RuntimeError, xc.FrameError), match="twice|duplicate"):
        fold([*sends, twice])


def test_m4_wire_embed_only_stage_runs_a_step_without_an_optimizer(hf_env, tmp_path):
    # covers: RUNTIME-I19
    # WOULD-FAIL-IF: stage 0 of a split tied hf plan holding only the Embed unit (its one
    # parameter the table the Head owns) builds an AdamW over no parameter (a crash at start),
    # its real HF embedding backward yields a gradient TIED-SPARSE refuses, its TIED(t, i)
    # rows are not e_i[T_i], its step updates anything, or its checkpoint names an
    # optimizer state.
    from wantrain.transport import framing as fr

    spec = _wire_spec("gemma2", [[0, 1], [1, 5], [5, 6]], out_dir=str(tmp_path))
    rt = StageRuntime(spec, 0)
    assert rt._tied_role == "embed" and rt.params == [] and rt.opt is None
    rt._tied_plan(0)
    rt.send_qs = {("next", 0): queue_mod.Queue()}
    table = rt._tied.detach().clone()
    g = torch.Generator().manual_seed(12)
    for mb in rt.my_mbs:
        rt._forward(0, mb, 0)
        _, _, _, _, out, _ = rt.send_qs[("next", 0)].get_nowait()  # FWD of mb
        rt.bwd_ready[(0, mb, 0)] = torch.randn(out.shape, generator=g)
        rt._backward_mid(0, mb, 0)
        ftype, step, got_mb, _, rows, _ = rt.send_qs[("next", 0)].get_nowait()
        assert (ftype, step, got_mb) == (fr.TIED, 0, mb)
        ids = rt._tied_ids(0, mb)
        assert rows.shape == (ids.numel(), table.shape[1]) and torch.isfinite(rows).all()
        assert rows.abs().sum() > 0
    for *_, w in rt.deferred_w:
        w()
    rt._optimizer_step_local(0)
    live = ~torch.isnan(rt._tied.detach())
    assert torch.equal(rt._tied.detach()[live], table[live])  # nothing updated here
    assert rt._opt_state()["state"] == {} and "embed_tokens" not in str(rt._chunk_state(0))
