"""The hardware switch decides which tests may use a GPU: with "gpu": false (the default)
every GPU test is not applicable, so a local Apple GPU or a stray CUDA device never decides
a gate that runs in CPU simulation."""

# WOULD-FAIL-IF: a missing, damaged or non-boolean switch lets GPU tests run, or gpu: true is lost

import json

from conftest import gpu_allowed


def test_only_a_literal_true_allows_gpu_tests(tmp_path):
    path = tmp_path / "hardware.json"
    assert gpu_allowed(str(path)) is False  # missing
    for body in ["{", "null", "[]", "{}", '{"gpu": "true"}', '{"gpu": 1}', '{"gpu": false}']:
        path.write_text(body)
        assert gpu_allowed(str(path)) is False, body
    path.write_text(json.dumps({"gpu": True}))
    assert gpu_allowed(str(path)) is True


def test_the_committed_switch_is_read():
    from conftest import HARDWARE

    assert HARDWARE.endswith("agents-architetura/tools/factory/hardware.json")
