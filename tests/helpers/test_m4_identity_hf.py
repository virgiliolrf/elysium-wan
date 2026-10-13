"""The byte-exact rule over HF units (BA-468, SPEC-runtime task 7): tiny
llama, qwen3, gemma2 (tied, split across the stages), granite and Mixtral (the
HF-MoE class swap), each trained through the pinned transformers' own modules,
leave state byte-identical to the single-device reference.

The protocol is tests/oracle/test_state_identity.py's. The mode comes from the
environment, as the identity gate (agents-architetura/tools/verify/identity.mjs)
runs it:

  FACTORY_REFERENCE=1   the single-device synchronous reference
  FACTORY_FEATURES=off  the distributed path with every optional mechanism off
  (neither)             the production path with every mechanism the arm's
                        plan admits on, under emulated WAN links

Each test asserts its state equals the reference's and, when
FACTORY_IDENTITY_OUT is set, writes it to $FACTORY_IDENTITY_OUT/state/<arm>.json.

Every arm goes through run_local, which builds and pins the buffer record,
census and hf_code in a child, runs the launch gate and fills the device
classes. A platform without a TORCH_RECORDS entry (BA-463, HANDOUT D-9) is
refused there with FP-TORCH-RECORDS naming its tag: every arm then asserts
that refusal strictly and writes state/<arm>.refused.json in every mode, so the
three modes still compare equal. Nothing is skipped.

No network: every model is a tiny config of its pinned config class
(hfm.tiny_config), and the tests run with HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1.
"""

import copy
import importlib.metadata
import os

import pytest

from helpers.exact import RUN_PORTS, reference_manifest_for
from wantrain.launch.local import links_for, prepare_hf, run_local
from wantrain.model import hfm
from wantrain.model.gpt import GPTConfig
from wantrain.runtime.spec import JobSpec
from wantrain.validation.state import diff_manifests, distributed_manifest, write_manifest

if os.environ.get("FACTORY_REFERENCE") == "1":
    MODE = "reference"
elif os.environ.get("FACTORY_FEATURES") == "off":
    MODE = "off"
else:
    MODE = "on"

LINKS_ON = dict(latency_ms=10, jitter_ms=15, bw_mbps=80)
RECIPE = dict(steps=3, microbatches=4)
SEQ_LEN = 16
FEATURES_OFF = dict(stages=2)
# Every optional mechanism an untied HF plan admits in spec mode (BA-434): the
# last stage is a wide stage of two peers cut into per-tensor pieces (HF banks
# included), with weighted microbatch ownership, streamed native-dtype
# gradients, recomputation, the planes codec, two streams per hop, adaptive
# microbatching and W emitting in forward order.
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
    shard="pieces",
    emission_order=["store", "forward"],
)
# A tied table split across the stages relays its rows in TIED frames along a
# width-1 chain. Its plan refuses the features below; each is listed with what
# refuses it and asserted refused, never silently dropped.
FEATURES_ON_TIED = {
    k: v for k, v in FEATURES_ON.items() if k not in ("stage_widths", "peer_weights", "shard")
}
REFUSED_TIED = [
    # BA-447: a relay exists only when every stage from the Embed to the Head has width 1.
    (
        "BA-447",
        dict(stage_widths=[1, 2], peer_weights=[[1.0], [1.0, 0.5]]),
        "needs width 1 on every",
    ),
    # HANDOUT-M2b I-7 (spec._check_pieces_tied): pieces needs the tied Embed and Head on one stage.
    ("HANDOUT-M2b I-7", dict(shard="pieces"), "Embed and the Head on one stage"),
]

# arm -> (model_type, whether its table is tied)
DENSE_ARMS = {
    "hf_llama": ("llama", False),
    "hf_qwen3": ("qwen3", False),
    "hf_gemma2_tied": ("gemma2", True),
    "hf_granite": ("granite", False),
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _model(model_type: str) -> GPTConfig:
    return GPTConfig(
        arch="hf", hf_config=hfm.tiny_config(model_type, use_cache=False), seq_len=SEQ_LEN
    )


def _spec(out_dir, tag, port_base, model, features, links=None, **extra) -> JobSpec:
    spec = JobSpec(
        model=copy.deepcopy(model),
        out_dir=str(out_dir),
        tag=tag,
        port_base=port_base,
        dial_timeout_s=60,
        **{**RECIPE, **features, **extra},
    )
    spec.links = links_for(spec, **(links or {}))
    return spec


def _emit(name: str, manifest: dict) -> None:
    out = os.environ.get("FACTORY_IDENTITY_OUT")
    if out:
        write_manifest(os.path.join(out, "state", name), manifest)


def _refused(arm: str, spec: JobSpec) -> None:
    """BA-463: this platform has no TORCH_RECORDS entry, so the launch gate
    refuses the run, naming the platform tag."""
    tag = hfm._platform_tag(importlib.metadata.distribution("torch"))
    with pytest.raises(ValueError) as e:
        prepare_hf(spec)
    msg = str(e.value)
    assert "FP-TORCH-RECORDS" in msg and tag in msg, msg
    _emit(f"{arm}.refused.json", {"cause": "FP-TORCH-RECORDS", "platform_tag": tag})


def _run(spec: JobSpec, prepared: bool) -> dict:
    res = run_local(spec, timeout=300, prepared=prepared)
    assert res["ok"], res
    return distributed_manifest(spec)


def _identity(tmp_path, port_base, arm: str, model: GPTConfig, on: dict, resumed=False):
    """The arm's state in this MODE, checked against the reference and written.
    -> (state, the prepared spec), or None when the platform is refused."""
    features, links = (on, LINKS_ON) if MODE == "on" else (FEATURES_OFF, {})
    extra = dict(steps=2, ckpt_every=1) if resumed else {}
    spec = _spec(tmp_path / MODE, arm, port_base, model, features, links, **extra)
    if not hfm.platform_admitted():
        _refused(arm, spec)
        return None
    prepare_hf(spec)
    ref = JobSpec.from_json(spec.to_json())
    if resumed:
        ref.steps = 4
    reference = reference_manifest_for(ref, tmp_path / "ref")
    if MODE == "reference":
        state = reference
    elif resumed:
        # Stop after 2 of 4 steps, then resume from the waves with the pins
        # the first launch made (a resume never rebuilds them).
        assert run_local(spec, timeout=300, prepared=True)["ok"]
        rest = _spec(
            tmp_path / MODE,
            arm,
            port_base + RUN_PORTS,
            spec.model,
            features,
            links,
            steps=4,
            resume=True,
        )
        state = _run(rest, prepared=False)
    else:
        state = _run(spec, prepared=True)
    assert diff_manifests(state, reference) == []
    assert state == reference
    steps = 4 if resumed else RECIPE["steps"]
    assert len(state["losses"]) == steps * RECIPE["microbatches"]
    _emit(f"{arm}.json", state)
    return state, spec


# covers: RUNTIME-I19
# WOULD-FAIL-IF: a mechanism on the production path (pieces over HF units, streamed
# native gradients, recompute, planes, parallel streams, adaptive microbatching) or the
# emulated WAN links, or the TIED relay of gemma2's split table, changes one bit of a
# weight, a buffer, an AdamW moment, the RNG state or a loss against the reference.
@pytest.mark.parametrize("arm", sorted(DENSE_ARMS))
def test_m4_identity_hf_dense(tmp_path, port_base, arm):
    model_type, tied = DENSE_ARMS[arm]
    model = _model(model_type)
    assert hfm.config_fields(model.hf_config)["tie_embeddings"] is tied
    got = _identity(tmp_path, port_base, arm, model, FEATURES_ON_TIED if tied else FEATURES_ON)
    if got is None or not tied:
        return
    # The arm proves the relay only while its table really is split, and the
    # features its plan refuses really are refused (with the pinned model).
    spec = got[1]
    assert spec.tied_relay(), spec.partition
    for _why, refused, text in REFUSED_TIED:
        bad = _spec(tmp_path / "plan", arm, port_base, spec.model, {**FEATURES_ON_TIED, **refused})
        with pytest.raises(ValueError, match=text):
            bad.finalize()


# covers: RUNTIME-I18, RUNTIME-I19
# WOULD-FAIL-IF: under the HF-MoE class swap a production mechanism (pieces over the
# expert banks, streamed gradients, recompute and its replay) or the emulated WAN links
# changes one bit of a weight, an AdamW moment, the RNG state, a loss or an aux hex, or
# drops or duplicates an aux row of some (step, mb, layer).
def test_m4_identity_hf_mixtral(tmp_path, port_base):
    model = _model("mixtral")
    got = _identity(tmp_path, port_base, "hf_mixtral", model, FEATURES_ON)
    if got is None:
        return
    state = got[0]
    n_layer = hfm.config_fields(model.hf_config)["n_layer"]
    rows = [(s, m, layer) for s, m, layer, _ in state.get("aux", [])]
    assert len(rows) == len(set(rows)) == RECIPE["steps"] * RECIPE["microbatches"] * n_layer


# covers: RUNTIME-I19, RUNTIME-I15
# WOULD-FAIL-IF: a resume from the waves (buffer record, pins, kept RECORD text) drops
# or changes part of the state, so a weight, an AdamW moment or step, the RNG state or a
# loss of the resumed run differs by one bit from the uninterrupted reference.
def test_m4_identity_hf_llama_resumed(tmp_path, port_base):
    _identity(tmp_path, port_base, "hf_llama_resumed", _model("llama"), FEATURES_ON, resumed=True)
