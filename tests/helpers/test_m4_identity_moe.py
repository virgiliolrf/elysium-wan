"""The byte-exact rule over native MoE (ADR-0008, BA-468): a tiny decoder whose
every block is arch.MoEMLP (score_bias selection with a moving selection bias,
a shared expert, load-balance and z aux terms) leaves state byte-identical to
the single-device reference: every weight and buffer (the selection biases
among them), every AdamW moment, the RNG state, every CE loss hex and every
aux hex of every (step, mb, layer).

The protocol is tests/oracle/test_state_identity.py's. The mode comes from the
environment, as the identity gate (agents-architetura/tools/verify/identity.mjs)
runs it:

  FACTORY_REFERENCE=1   the single-device synchronous reference
  FACTORY_FEATURES=off  the distributed path with every optional mechanism off
  (neither)             the production path, ZeRO-1 width 2 on the last stage
                        with every mechanism on, under emulated WAN links

Each test asserts its state equals the reference's and, when
FACTORY_IDENTITY_OUT is set, writes it to $FACTORY_IDENTITY_OUT/state/<arm>.json.
Nothing here imports transformers.
"""

import os

import torch

from helpers.exact import RUN_PORTS, reference_manifest_for
from wantrain.launch.local import links_for, run_local
from wantrain.model import arch
from wantrain.model.gpt import GPTConfig
from wantrain.runtime.spec import JobSpec
from wantrain.validation.state import (
    diff_manifests,
    distributed_manifest,
    tensor_record,
    write_manifest,
)

if os.environ.get("FACTORY_REFERENCE") == "1":
    MODE = "reference"
elif os.environ.get("FACTORY_FEATURES") == "off":
    MODE = "off"
else:
    MODE = "on"

LINKS_ON = dict(latency_ms=10, jitter_ms=15, bw_mbps=80)
RECIPE = dict(steps=3, microbatches=4)
FEATURES_OFF = dict(stages=2)
# The last stage is a ZeRO-1 stage of two peers, so its selection-bias counts
# reach their owners through the streamed native-wire path, with weighted
# microbatch ownership, recomputation (and its MOE-REPLAY check), the planes
# codec, two streams per hop, adaptive microbatching and W in forward order.
# Pieces over MoE chunks is identity-m4's sweep (BA-434), not this arm's.
FEATURES_ON = dict(
    stages=2,
    stage_widths=[1, 2],
    peer_weights=[[1.0], [1.0, 0.5]],
    stream_grads=True,
    recompute=True,
    wire_codec="planes",
    p2p_streams=2,
    adapt_budget_ms=8,
    wire="native",
    shard="zero1",
    emission_order=["store", "forward"],
)
N_LAYER = 3
N_EXPERTS = 4
MODEL = dict(
    vocab_size=64,
    n_layer=N_LAYER,
    n_head=2,
    d_model=32,
    seq_len=16,
    norm="rmsnorm",
    mlp="swiglu",
    bias=False,
    pos="rope",
    moe_experts=N_EXPERTS,
    moe_top_k=2,
    moe_d_ff=32,
    moe_shared_d_ff=16,
    moe_select="score_bias",
    moe_bias_rate=0.01,
    moe_lbl_coef=0.01,
    moe_z_coef=0.001,
)


def _spec(out_dir, tag, port_base, features, links=None, **extra) -> JobSpec:
    spec = JobSpec(
        model=GPTConfig(**MODEL),
        out_dir=str(out_dir),
        tag=tag,
        port_base=port_base,
        dial_timeout_s=60,
        **{**RECIPE, **features, **extra},
    )
    spec.links = links_for(spec, **(links or {}))
    return spec


def _run(spec: JobSpec) -> dict:
    res = run_local(spec, timeout=300)
    assert res["ok"], res
    return distributed_manifest(spec)


def _identity(tmp_path, port_base, arm: str, resumed=False) -> dict:
    features, links = (FEATURES_ON, LINKS_ON) if MODE == "on" else (FEATURES_OFF, {})
    extra = dict(steps=2, ckpt_every=1) if resumed else {}
    spec = _spec(tmp_path / MODE, arm, port_base, features, links, **extra)
    spec.finalize()
    if MODE == "on":
        # ZeRO-1 width 2 holds MoE layers, so its count path is exercised.
        a, b = spec.partition[-1]
        assert spec.stage_widths[-1] == 2
        assert any(arch.is_moe_layer(spec.model, i) for i in range(a, b)), spec.partition
    steps = 4 if resumed else RECIPE["steps"]
    ref = JobSpec.from_json(spec.to_json())
    ref.steps = steps
    reference = reference_manifest_for(ref, tmp_path / "ref")
    if MODE == "reference":
        state = reference
    elif resumed:
        # Stop after 2 of 4 steps, then resume from the waves.
        assert run_local(spec, timeout=300)["ok"]
        state = _run(
            _spec(
                tmp_path / MODE, arm, port_base + RUN_PORTS, features, links, steps=4, resume=True
            )
        )
    else:
        state = _run(spec)
    assert diff_manifests(state, reference) == []
    assert state == reference
    # The state carries what RUNTIME-I18 compares: every CE loss, one aux row
    # per (step, mb, layer), and selection biases the counts moved.
    M, n_layer = RECIPE["microbatches"], N_LAYER
    assert len(state["losses"]) == steps * M
    rows = [(s, m, layer) for s, m, layer, _ in state.get("aux", [])]
    assert len(rows) == len(set(rows)) == steps * M * n_layer
    biases = {k: v for k, v in state["weights"].items() if k.endswith("sel_bias")}
    assert len(biases) == n_layer, sorted(biases)
    zero = tensor_record(torch.zeros(N_EXPERTS, dtype=torch.float32))
    assert any(v != zero for v in biases.values()), biases
    out = os.environ.get("FACTORY_IDENTITY_OUT")
    if out:
        write_manifest(os.path.join(out, "state", f"{arm}.json"), state)
    return state


# covers: RUNTIME-I18
# WOULD-FAIL-IF: ZeRO-1 over MoE chunks (count vectors at the bias owner, streamed
# native gradients), recompute and its replay, or the emulated WAN links change one bit
# of a weight, a selection bias, an AdamW moment, the RNG state, a CE loss or an aux
# hex, or drop or duplicate an aux row of some (step, mb, layer).
def test_m4_identity_moe_native(tmp_path, port_base):
    _identity(tmp_path, port_base, "moe_native")


# covers: RUNTIME-I18, RUNTIME-I15
# WOULD-FAIL-IF: a resume from the waves drops or changes part of the MoE state (the
# selection biases, the aux rows of the steps before the stop, an AdamW moment or step,
# the RNG state), so the resumed run differs by one bit from the uninterrupted reference.
def test_m4_identity_moe_native_resumed(tmp_path, port_base):
    _identity(tmp_path, port_base, "moe_native_resumed", resumed=True)
