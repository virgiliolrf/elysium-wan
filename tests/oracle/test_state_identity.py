"""The byte-exact rule as a gate: the final training state of the production
path equals the single-device reference's, byte for byte.

State is the canonical manifest of validation/state.py: every weight and
buffer, the AdamW state of every trainable parameter, the optimizer
hyperparameters, the torch RNG state and every per-microbatch loss, each by
the sha256 of its raw bytes. The mode comes from the environment, as the
identity gate (agents-architetura/tools/verify/identity.mjs) runs it:

  FACTORY_REFERENCE=1   the single-device synchronous reference
  FACTORY_FEATURES=off  the distributed path with every optional mechanism off
  (neither)             the production path with every mechanism on

Each run asserts its own state equals the reference's and, when
FACTORY_IDENTITY_OUT is set, writes it to $FACTORY_IDENTITY_OUT/state/, where
the gate compares the three modes' files by sha256 independently.

WOULD-FAIL-IF: any engine mechanism (wide ZeRO-1 stages, streamed gradients,
recomputation, the planes wire codec, parallel streams, adaptive
microbatching, WAN latency and jitter, a checkpoint resume) changed one bit of
a weight, an AdamW moment or step, the RNG state or a loss, or a checkpoint
dropped part of the state a resume needs.
"""

import os

import pytest
from helpers.exact import FEATURES_OFF, FEATURES_ON, RUN_PORTS, make_spec, reference_manifest

from wantrain.launch.local import run_local
from wantrain.validation.state import diff_manifests, distributed_manifest, write_manifest

if os.environ.get("FACTORY_REFERENCE") == "1":
    MODE = "reference"
elif os.environ.get("FACTORY_FEATURES") == "off":
    MODE = "off"
else:
    MODE = "on"

LINKS_ON = dict(latency_ms=10, jitter_ms=15, bw_mbps=80)


def _run(spec, timeout=300):
    res = run_local(spec, timeout=timeout)
    assert res["ok"], res
    return distributed_manifest(spec)


def _straight(tmp_path, port_base):
    if MODE == "off":
        return _run(make_spec(tmp_path, "straight", port_base, FEATURES_OFF))
    return _run(make_spec(tmp_path, "straight", port_base, FEATURES_ON, LINKS_ON))


def _resumed(tmp_path, port_base):
    """Stop after 2 of 4 steps and resume from the checkpoints."""
    features, links = (FEATURES_OFF, {}) if MODE == "off" else (FEATURES_ON, LINKS_ON)
    part = make_spec(tmp_path, "resumed", port_base, features, links, steps=2, ckpt_every=1)
    assert run_local(part, timeout=300)["ok"]
    rest = make_spec(tmp_path, "resumed", port_base + RUN_PORTS, features, links, resume=True)
    return _run(rest)


CASES = {"straight": _straight, "resumed": _resumed}


@pytest.mark.parametrize("case", sorted(CASES))
def test_final_state_is_byte_identical_to_the_reference(tmp_path, port_base, case):
    reference = reference_manifest(tmp_path / "ref")
    state = reference if MODE == "reference" else CASES[case](tmp_path / MODE, port_base)
    assert diff_manifests(state, reference) == []
    assert state == reference
    out = os.environ.get("FACTORY_IDENTITY_OUT")
    if out:
        write_manifest(os.path.join(out, "state", f"{case}.json"), state)
