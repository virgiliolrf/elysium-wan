"""Declared numeric precision as a first-class, user-chosen axis.

Precision is a property of the recipe, never of the transport. The engineer
picks a compute dtype and a boundary dtype; this module reports which claim the
pair earns and the admission gate prices it in bandwidth. boundary == compute
is native: the wire carries the tensor's whole information content and the run
is lossless against a reference recipe a single-device baseline would plausibly
use on its own (standard bf16 mixed precision passes; an e4m3 cast applied only
where a tensor meets a socket does not). Anything narrower is wire-quantized.
"""

import torch

# name -> torch_dtype, wire bytes/elem, stored mantissa bits, min_normal (for the
# underflow check), needs_scale (gradients underflow without a scale), codec, and
# ratio/ceiling: lossless ratio measured through codec/planes.py on d_model=1600
# boundary tensors and the exact entropy ceiling on the same tensors (or None).
RECIPES = {
    "fp32": dict(
        torch_dtype=torch.float32,
        bytes=4,
        mantissa=23,
        min_normal=1.18e-38,
        needs_scale=False,
        codec="planes",
        ratio=1.175,
        ceiling=1.201,
    ),
    "bf16": dict(
        torch_dtype=torch.bfloat16,
        bytes=2,
        mantissa=7,
        min_normal=1.18e-38,
        needs_scale=False,
        codec="planes",
        ratio=1.426,
        ceiling=1.498,
    ),
    # fp16 codes worse than bf16 despite equal width: its 5-bit exponent leaves
    # the high byte mixing exponent and mantissa, so no plane is pure exponent.
    "fp16": dict(
        torch_dtype=torch.float16,
        bytes=2,
        mantissa=10,
        min_normal=6.10e-5,
        needs_scale=True,
        codec="planes",
        ratio=1.165,
        ceiling=None,
    ),
    "fp8_e4m3": dict(
        torch_dtype=torch.float8_e4m3fn,
        bytes=1,
        mantissa=3,
        min_normal=1.95e-3,
        needs_scale=True,
        codec="raw",
        ratio=1.256,
        ceiling=1.238,
    ),
    # e5m2 codes better than e4m3: fewer mantissa bits means less noise per byte.
    "fp8_e5m2": dict(
        torch_dtype=torch.float8_e5m2,
        bytes=1,
        mantissa=2,
        min_normal=1.53e-5,
        needs_scale=True,
        codec="raw",
        ratio=1.397,
        ceiling=None,
    ),
}

# Precision order, widest first. Used only to decide whether a boundary choice
# narrows the tensor relative to compute.
_WIDTH = {"fp32": 3, "bf16": 2, "fp16": 2, "fp8_e5m2": 1, "fp8_e4m3": 1}

NATIVE, WIRE_QUANTIZED = "native", "wire-quantized"


# Dtypes this engine can run the block math in (autocast paths implemented and
# validated). fp8 compute would need a full fine-grained-scaling recipe of the
# DeepSeek-V3 kind; fp8 is implemented only as a boundary format here.
CAN_BE_COMPUTE = {"fp32", "bf16", "fp16"}

# Minimum CUDA compute capability that runs block math in this dtype at a rate
# worth having. Gates compute only: boundary casts work lower down, and a
# lossless claim needs compute == boundary anyway. Keeps the planner from
# declaring bf16 for a pre-Ampere card, where it is emulated or unimplemented.
MIN_COMPUTE_CAP = {
    "fp32": (3, 5),
    "fp16": (5, 3),  # half arithmetic; tensor cores only from sm_70
    "bf16": (8, 0),  # Ampere
    "fp8_e4m3": (8, 9),  # Ada/Hopper
    "fp8_e5m2": (8, 9),
}


def compute_runnable(name: str, cap) -> bool:
    """Can a device of this compute capability run block math in `name`?

    `cap` is (major, minor), or None when the fleet's capability is unknown --
    unknown returns True rather than guessing, so a missing measurement widens
    the search instead of silently narrowing the recipe.
    """
    if cap is None:
        return True
    return tuple(cap) >= MIN_COMPUTE_CAP.get(name, (0, 0))


def spec(name):
    if name not in RECIPES:
        raise ValueError(f"unknown precision {name!r}; choose from {sorted(RECIPES)}")
    return RECIPES[name]


def bytes_per_elem(name):
    return spec(name)["bytes"]


def classify(compute: str, boundary: str) -> dict:
    """Which exactness claim does this (compute, boundary) pair earn?"""
    c, b = spec(compute), spec(boundary)
    narrowed = _WIDTH[boundary] < _WIDTH[compute] or (
        _WIDTH[boundary] == _WIDTH[compute] and b["mantissa"] < c["mantissa"]
    )
    if not narrowed:
        return dict(
            level=NATIVE,
            lossless_vs_reference=True,
            claim=f"boundary is native {boundary}: the wire carries its full "
            f"information content, so the run is numerically lossless "
            f"with respect to the declared {compute} reference recipe",
        )
    return dict(
        level=WIRE_QUANTIZED,
        lossless_vs_reference=False,
        claim=f"boundary narrows {compute} -> {boundary} ({c['bytes']} -> "
        f"{b['bytes']} bytes/elem, {c['mantissa']} -> {b['mantissa']} "
        f"mantissa bits). This is a network-motivated approximation: the "
        f"transport is exact and the run is bit-reproducible, but it is "
        f"NOT numerically lossless against the {compute} reference",
    )


def describe(compute: str, boundary: str, d_model: int) -> dict:
    """Full recipe fingerprint: what was chosen, and what it costs per token."""
    b = spec(boundary)
    cls = classify(compute, boundary)
    per_token_raw = b["bytes"] * d_model
    return dict(
        compute=compute,
        boundary=boundary,
        bytes_per_elem=b["bytes"],
        wire_bytes_per_token_raw=per_token_raw,
        wire_bytes_per_token_coded=round(per_token_raw / b["ratio"], 1),
        codec=b["codec"],
        codec_ratio=b["ratio"],
        entropy_ceiling=b["ceiling"],
        needs_scale=b["needs_scale"],
        **cls,
    )
