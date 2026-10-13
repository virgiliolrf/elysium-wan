"""Determinism configuration shared by every process in every experiment.

Bitwise reproducibility requires: fixed seeds, single-threaded intra-op math
(thread count changes reduction order), deterministic algorithms, and no TF32.
Every entry point calls setup_determinism() before touching torch state.
"""

import os
import platform
import random

_X86 = ("x86_64", "amd64", "i386", "i686")
_PINS = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
# Cross-host CPU bitwise: every x86 node runs the same ISA dispatch, so a Zen3
# (AVX2) and a Zen4 (AVX512) box run identical reduction orders. ARM has no AVX
# dispatch, and the knobs would only warn there.
_X86_PINS = {"ATEN_CPU_CAPABILITY": "avx2", "MKL_ENABLE_INSTRUCTIONS": "AVX2"}


class DeterminismError(ValueError):
    """A determinism pin or flag differs from what bitwise identity requires."""


def _is_x86() -> bool:
    return platform.machine().lower() in _X86


def _pins() -> dict[str, str]:
    return {**_PINS, **_X86_PINS} if _is_x86() else dict(_PINS)


# The pins go into this process's environment at import, before this module
# imports torch: ATen computes its CPU capability once, at its first dispatch,
# and MKL reads its pin at its first call, so a pin assigned later (by
# setup_determinism, after other code already ran a kernel) is refused by
# check_determinism. Importing this module first is how an in-process caller
# gets the pins in time (the test session loads it as pyproject's `-p`). A
# differing pre-set pin is kept, never overwritten: setup_determinism refuses it.
for _name, _pin in _pins().items():
    os.environ.setdefault(_name, _pin)

import torch  # noqa: E402  (after the pins, see above)


def pinned_env(env: dict) -> dict:
    """A child's environment with the numeric pins in place before its
    interpreter starts, so ATen and MKL read them at their first dispatch
    (setup_determinism in the child can only assign them after `import torch`,
    and check_determinism refuses an ATen dispatch cached before that). A pin
    the parent already holds is kept as it is: a differing one is refused by
    the child's setup_determinism, never overwritten here."""
    out = dict(env)
    for name, pin in _pins().items():
        out.setdefault(name, pin)
    return out


def setup_determinism(seed: int) -> None:
    """Assign the numeric pins, set every determinism flag, then read them back.

    A pin already in the environment with a different value is refused before any
    torch flag changes: keeping it would silently run another ISA or workspace.

    PYTORCH_CUDA_ALLOC_CONF stays a default the operator may override: it shapes
    the CUDA allocator only and moves no bit, so it is not a pin.
    """
    pins = _pins()
    for name, pin in pins.items():
        have = os.environ.get(name)
        if have is not None and have != pin:
            raise DeterminismError(f"{name} is pre-set to {have!r}; the determinism pin is {pin!r}")
    os.environ.update(pins)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    # allow_tf32 and the matmul precision are coupled in torch: set the flags
    # first, then the precision, and read both back.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    check_determinism()


def check_determinism() -> None:
    """Read back every pin and flag setup_determinism sets; raise naming what differs.

    On x86 the ATen dispatch is read back too (get_cpu_capability): a pin assigned
    after ATen cached its capability does not take effect. MKL has no read-back API
    in torch, so MKL_ENABLE_INSTRUCTIONS is checked only in the environment; an MKL
    initialized before the pin keeps its own dispatch (HANDOUT-M4 R-9).
    """
    for name, pin in _pins().items():
        have = os.environ.get(name)
        if have != pin:
            raise DeterminismError(f"{name} is {have!r}; the determinism pin is {pin!r}")
    flags = (
        ("torch.get_num_threads()", torch.get_num_threads(), 1),
        (
            "torch.are_deterministic_algorithms_enabled()",
            torch.are_deterministic_algorithms_enabled(),
            True,
        ),
        (
            "torch.is_deterministic_algorithms_warn_only_enabled()",
            torch.is_deterministic_algorithms_warn_only_enabled(),
            False,
        ),
        ("torch.get_float32_matmul_precision()", torch.get_float32_matmul_precision(), "highest"),
        ("torch.backends.cuda.matmul.allow_tf32", torch.backends.cuda.matmul.allow_tf32, False),
        ("torch.backends.cudnn.allow_tf32", torch.backends.cudnn.allow_tf32, False),
        ("torch.backends.cudnn.deterministic", torch.backends.cudnn.deterministic, True),
        ("torch.backends.cudnn.benchmark", torch.backends.cudnn.benchmark, False),
    )
    # The TF32 flags and the matmul precision are one state in torch (forcing
    # either moves the other), so every differing flag is named, the first first.
    bad = [
        f"{name} is {have!r}; determinism requires {want!r}"
        for name, have, want in flags
        if have != want
    ]
    if bad:
        raise DeterminismError("; ".join(bad))
    if _is_x86():
        pin = _X86_PINS["ATEN_CPU_CAPABILITY"]
        cap = torch.backends.cpu.get_cpu_capability()
        if cap.lower() != pin.lower():
            raise DeterminismError(
                f"ATEN_CPU_CAPABILITY is pinned to {pin!r} but ATen reports {cap!r} "
                "(the pin was assigned after ATen cached its dispatch)"
            )


def float_hex(x: float) -> str:
    """Exact hex representation for bitwise comparison of recorded losses."""
    return float(x).hex()
