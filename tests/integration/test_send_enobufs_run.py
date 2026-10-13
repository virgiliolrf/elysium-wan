"""A run whose data plane is refused with ENOBUFS on every 7th send attempt of
every SecureSocket (WANTRAIN_TEST_ENOBUFS, timing only) completes, and its
training state equals the reference's bit for bit: every weight, every
optimizer moment, the RNG state and every loss hex. This is the M2b verdict
gate's failure (pieces-heal-run-loaded, "send failed ([Errno 55] No buffer
space available)") made deterministic, with every mechanism on."""

from __future__ import annotations

import pytest
from helpers.exact import FEATURES_ON, make_spec, reference_manifest_for

from wantrain.launch.local import run_local
from wantrain.validation.state import diff_manifests, distributed_manifest


@pytest.fixture(scope="module")
def ref_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("enobufs_reference")


def test_enobufs_on_the_data_plane_is_retried_and_the_run_is_the_reference(
    tmp_path, port_base, ref_dir, monkeypatch
):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: a transient ENOBUFS is treated as a lost peer (a stage dies with
    # "No buffer space available" and the run is not ok), or the retry duplicates, drops or
    # reorders a record (the receiver's tag check fails and the run dies, or the state or a
    # loss differs from the reference).
    monkeypatch.setenv("WANTRAIN_TEST_ENOBUFS", "7")
    spec = make_spec(tmp_path, "enobufs", port_base, FEATURES_ON)
    res = run_local(spec, timeout=240)
    assert res["ok"], res
    state = distributed_manifest(spec)
    monkeypatch.delenv("WANTRAIN_TEST_ENOBUFS")
    ref = reference_manifest_for(spec, ref_dir)
    assert diff_manifests(state, ref) == [], diff_manifests(state, ref)
    assert len(state["losses"]) == spec.steps * spec.microbatches
    assert state == ref
