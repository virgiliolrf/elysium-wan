"""Shared helpers for the wantrain test suite."""

from __future__ import annotations

import json
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
TESTS = os.path.dirname(os.path.abspath(__file__))
# Peer identities for the data plane live in a throwaway directory, not ~.
os.environ.setdefault(
    "WANTRAIN_PEER_KEY",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wantrain-test-{os.getpid()}", "peer_key"),
)
for p in (SRC, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

# Hypothesis profiles for tests/fuzz, here because a second conftest.py would
# shadow this module's name: "ci" (the fuzz rung) explores more and never fails
# on timing; "dev", the default, keeps local runs quick.
try:
    from hypothesis import HealthCheck, settings
except ImportError:  # only the fuzz suite needs it
    pass
else:
    settings.register_profile(
        "ci", max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    settings.register_profile("dev", max_examples=60, deadline=None)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def port_base():
    """A reserved block of 64 consecutive free ports (tests/helpers/ports.py)."""
    from helpers.ports import reserve_port_block

    block = reserve_port_block()
    try:
        yield block.base
    finally:
        block.release()


# The human's hardware switch (agents-architetura/tools/factory/hardware.json). With "gpu": false,
# the default, the suite runs on the CPU only: every GPU test is not applicable, whatever this
# machine has, so a local Apple GPU or a stray CUDA device never decides a gate the factory runs
# in CPU simulation. Anything but a literal "gpu": true reads as false.
HARDWARE = os.path.join(
    os.path.dirname(TESTS), "agents-architetura", "tools", "factory", "hardware.json"
)
GPU_OFF = "the hardware switch is gpu: false (agents-architetura/tools/factory/hardware.json)"


def gpu_allowed(path: str = HARDWARE) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("gpu") is True
    except (OSError, ValueError, AttributeError):
        return False


def mps_available() -> bool:
    if not gpu_allowed():
        return False
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def cuda_available() -> bool:
    if not gpu_allowed():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False
