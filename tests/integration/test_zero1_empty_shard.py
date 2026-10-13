"""M2B-ZERO1-EMPTY-SHARD: a whole-tensor ZeRO-1 stage wider than its trainable
tensors. The toy GPT's Head holds 3 (ln_f.weight, ln_f.bias, out.weight); at
width 4 peer 3 owns params[3::4] = []. It builds no optimizer (AdamW refuses an
empty parameter list) and takes part only in the exchanges: it ships its
gradient slices, receives every owner's PARAMS broadcast and writes a
checkpoint whose optimizer state is empty under the same hyperparameters as
its peers'. The training state, every loss included, equals the reference's
bit for bit, also after a resume from that checkpoint."""

from __future__ import annotations

import random

import pytest
import torch
from helpers.exact import make_spec, reference_manifest_for

from wantrain.launch.local import run_local
from wantrain.validation.state import diff_manifests, distributed_manifest

# The Head alone on stage 1, at width 4 over its 3 trainable tensors.
HEAD_W4 = dict(stages=2, partition=[[0, 7], [7, 8]], stage_widths=[1, 4])
ARMS = [
    ("fp32", False),  # batch exchange at the step barrier
    ("fp32", True),  # streamed shard gradients
    ("native", True),  # per-tensor native-dtype gradients, need-first queues
]


@pytest.fixture(scope="module")
def ref_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("zero1_empty_reference")


def _spec(out, tag, port, wire, stream, **kw):
    return make_spec(
        out, tag, port, {**HEAD_W4, "shard": "zero1", "wire": wire, "stream_grads": stream}, **kw
    )


def _bare(spec, rank, peer):
    """A StageRuntime that never connects; the process-global state its constructor sets is
    put back."""
    from wantrain.runtime.stage import StageRuntime

    rng, py, threads = torch.get_rng_state(), random.getstate(), torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        return StageRuntime(spec, rank, peer)
    finally:
        torch.set_rng_state(rng)
        random.setstate(py)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(det, warn_only=warn)


def _run_exact(spec, ref_dir):
    res = run_local(spec, timeout=240)
    assert res["ok"], res
    state = distributed_manifest(spec)
    ref = reference_manifest_for(spec, ref_dir)
    assert diff_manifests(state, ref) == [], diff_manifests(state, ref)
    assert state == ref
    return state


@pytest.mark.parametrize(
    "wire,stream", ARMS, ids=[f"{w}-{'stream' if s else 'batch'}" for w, s in ARMS]
)
def test_empty_owner_builds_no_optimizer(tmp_path, wire, stream):
    """Peer 3 of the width-4 Head owns no tensor: it constructs (the old code raised
    "optimizer got an empty parameter list") with no optimizer, while peers 0..2 each own one
    tensor under their own AdamW."""
    # covers: RUNTIME-I1
    # WOULD-FAIL-IF: an empty-owner peer builds an optimizer (AdamW over [] raises), or an
    # owning peer lost its optimizer or its stride.
    spec = _spec(tmp_path, "c", 29000, wire, stream).finalize()
    empty = _bare(spec, 1, 3)
    assert empty.shard_params == [] and empty.opt is None
    ck = empty._opt_state()
    assert ck["state"] == {} and ck["param_groups"][0]["params"] == []
    empty._load_opt(ck, "its own checkpoint")
    for bad in ({"state": {0: {}}, "param_groups": [{"params": [0]}]}, {"state": {}}, None):
        with pytest.raises(RuntimeError, match="owns no trainable tensor"):
            empty._load_opt(bad, "a planted wave")
    for k in range(3):
        rt = _bare(spec, 1, k)
        assert len(rt.shard_params) == 1 and rt.opt is not None
        assert rt.shard_params[0] is rt.params[k]


@pytest.mark.parametrize(
    "wire,stream", ARMS, ids=[f"{w}-{'stream' if s else 'batch'}" for w, s in ARMS]
)
def test_empty_owner_run_is_the_reference(tmp_path, port_base, ref_dir, wire, stream):
    """The whole run with an empty owner: the final manifest (every weight, every AdamW
    moment, the hyperparameters, the RNG) and every loss hex equal the reference's."""
    # covers: RUNTIME-I1
    # WOULD-FAIL-IF: the empty owner skips a PARAMS broadcast or a gradient exchange (the run
    # hangs or its replicas disagree), steps an optimizer, or its checkpoint carries no or
    # other hyperparameters (the manifest's hparams disagree).
    spec = _spec(tmp_path, "z", port_base, wire, stream)
    state = _run_exact(spec, ref_dir)
    assert len(state["losses"]) == spec.steps * spec.microbatches
    ck = torch.load(tmp_path / "z_ckpt_rank1p3.pt", map_location="cpu", weights_only=True)
    assert ck["opt"]["state"] == {}
    assert ck["opt"]["param_groups"][0]["params"] == []


def test_empty_owner_resume_is_the_reference(tmp_path, port_base, ref_dir):
    """Checkpoint waves with the empty owner, then a resume from the newest one: the state
    equals the uninterrupted reference's, so the empty slot's wave is well-defined."""
    # covers: RUNTIME-I3, RUNTIME-I17
    # WOULD-FAIL-IF: the empty owner writes no wave or an ill-formed one, its resume loads an
    # optimizer it does not have, or it accepts a wave holding optimizer state it cannot own.
    first = _spec(tmp_path, "r", port_base, "fp32", True, steps=2, ckpt_every=1)
    res = run_local(first, timeout=240)
    assert res["ok"], res
    spec = _spec(tmp_path, "r", port_base + 32, "fp32", True, resume=True)
    _run_exact(spec, ref_dir)
    log = (tmp_path / "r_r1p3.log").read_text(errors="replace")
    assert "WT_RESUME rank=1 start_step=2" in log, log[-2000:]
