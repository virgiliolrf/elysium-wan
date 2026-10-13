"""E4's gate mode (HANDOUT-M2b D-3, the human's decision 2026-10-10): with the
hardware switch at gpu false, F3's band and an arm off the PLAN's configuration
are measured and do not fail; a missing or invalid measurement, byte identity,
F6, F5 and the emission order at RTT_hi fail on every machine. With gpu true
every check fails as before. Pure: exposure.e4_gate_fails and its helpers."""

import math

import pytest

from wantrain.validation import exposure

ENFORCED = ("measurement", "identity", "f6", "f5", "emission_order")
MEASURED = ("f3_band", "off_configuration")


def test_gpu_false_measures_f3_and_off_configuration_and_fails_the_rest():
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: gpu false let a state off the reference, a fired F6 or F5, a
    # wrong emission order or an invalid measurement pass, or still failed on an
    # F3 band miss or an off-configuration arm.
    for check in MEASURED:
        assert exposure.e4_gate_fails(check, ok=False, gpu=False) is False, check
    for check in ENFORCED:
        assert exposure.e4_gate_fails(check, ok=False, gpu=False) is True, check
    for check in ENFORCED + MEASURED:
        assert exposure.e4_gate_fails(check, ok=True, gpu=False) is False, check
    assert "not-enforced" in exposure.f3_band_mode(False)


def test_gpu_true_fails_f3_and_w8_again():
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: gpu true kept F3's band or an off-configuration arm (interior
    # w8) measured-only, or failed a check that passed.
    for check in ENFORCED + MEASURED:
        assert exposure.e4_gate_fails(check, ok=False, gpu=True) is True, check
        assert exposure.e4_gate_fails(check, ok=True, gpu=True) is False, check
    assert exposure.f3_band_mode(True).startswith("enforced")


def test_only_a_literal_true_enforces_and_unknown_checks_are_refused():
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: a truthy non-boolean switch value enforced F3 (or a falsy one
    # relaxed an enforced check), or a misspelt check were silently treated as
    # measured-only.
    for gpu in (1, "true", None):
        assert exposure.e4_gate_fails("f3_band", ok=False, gpu=gpu) is False, gpu
        assert exposure.e4_gate_fails("identity", ok=False, gpu=gpu) is True, gpu
    with pytest.raises(ValueError, match="unknown check"):
        exposure.e4_gate_fails("f3", ok=False, gpu=False)
    assert set(exposure.E4_ENFORCED) == set(ENFORCED)
    assert set(exposure.E4_MEASURED_WITHOUT_GPU) == set(MEASURED)


def test_a_measurement_must_exist_and_be_valid():
    # covers: SCHEDULER-I17
    # WOULD-FAIL-IF: a line without its predicted / measured / ratio, or with a
    # NaN, infinite, negative, boolean or zero-prediction value, counted as a
    # measurement (gpu false would then pass an arm that measured nothing).
    good = exposure.arm_line("interior_w4.pieces_fwd", 12.0, 13.5, "F3", ratio=1.125)
    assert exposure.measurement_problems(good) == []
    assert exposure.measurement_problems({**good, "measured": 0.0, "ratio": 0.0}) == []
    for key in ("predicted", "measured", "ratio", "label"):
        line = dict(good)
        del line[key]
        assert exposure.measurement_problems(line) == [f"no {key}"], key
    for key, bad in (
        ("measured", math.nan),
        ("ratio", math.inf),
        ("measured", -0.1),
        ("predicted", 0.0),
        ("predicted", True),
        ("measured", "13.5"),
    ):
        assert exposure.measurement_problems({**good, key: bad}), (key, bad)


def test_e4_arms_run_only_in_their_own_whole_machine_run():
    """covers: SCHEDULER-I17
    WOULD-FAIL-IF: the E4 wall-clock arms ran inside a ladder rung of a verdict
    gate (FACTORY_MACHINE_LOCK_HELD=ladder:<rung>, a per-step timeout sized for
    the suite; run wf_4c4dfd30-00d), inside another pinned entry (pinned:f1-matrix-x20,
    its r0 gate) or in a test slot, or were skipped in their own pinned entry."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "integration" / "test_engine_bitwise.py"
    src = path.read_text()
    start = src.index("def _e4_own_whole_machine_run")
    ns: dict = {}
    exec(src[start : src.index("\n\n\n", start)], ns)
    own = ns["_e4_own_whole_machine_run"]
    assert own("pinned:e4-exposure-sweep") and own("m2b-e4") and own("e4-exposure")
    assert not own(None) and not own("")
    assert not own("ladder:integration") and not own("ladder:full")
    assert not own("pinned:f1-matrix-x20") and not own("pinned:e3-holds-liveness-x3")
