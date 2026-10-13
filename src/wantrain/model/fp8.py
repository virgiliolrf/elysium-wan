"""Recipe-declared fp8 boundary quantization with per-token power-of-two scaling.

An unscaled e4m3 cast cannot carry activation gradients: its smallest subnormal
is 2^-9 while dL/dx at a block input is ~1e-5, so everything upstream of the
output head gets exactly zero. The scale is one power of two per token (last
dim = hidden dim): multiplying by it is exact, so the round trip adds nothing
beyond fp8 rounding, and it is recoverable from the dequantized tensor alone
(recover_exp), so the wire carries the bare fp8 image. Per token, not per
tensor: one amax would waste the 8-bit range on rows of very different scale.
"""

import torch

# fmt -> (torch dtype, e such that 2^e > max finite value of the format)
#   e4m3 max finite 448   < 2^9   -> shifting by e-8 leaves headroom for rounding
#   e5m2 max finite 57344 < 2^16  -> likewise with e-15
FORMATS = {
    "e4m3": (torch.float8_e4m3fn, 8),
    "e5m2": (torch.float8_e5m2, 15),
}

# Candidate offsets tried when recovering a scale from a tensor. Rounding at the
# top of the range can carry amax into the next octave, so the exponent implied
# by amax alone can be off by one either way.
_DELTAS = (0, -1, 1, -2, 2)


def scale_exp(x, fmt="e4m3"):
    """Per-row power-of-two exponent k such that x * 2^-k fits the fp8 range.

    Returns an int32 tensor shaped like x with the last dim collapsed to 1, so
    it broadcasts against x. All-zero rows get k = 0.
    """
    _, ebias = FORMATS[fmt]
    amax = x.detach().abs().amax(dim=-1, keepdim=True)
    _, e = torch.frexp(amax)  # amax = m * 2^e with m in [0.5, 1)
    k = (e - ebias).to(torch.int32)
    return torch.where(amax > 0, k, torch.zeros_like(k))


def _pow2(k, dtype):
    return torch.exp2(k.to(dtype))


def quantize(x, fmt="e4m3"):
    """x -> (fp8 tensor, per-row int32 exponent). Exact given the returned k."""
    dt, _ = FORMATS[fmt]
    k = scale_exp(x, fmt)
    return (x * _pow2(-k, x.dtype)).to(dt), k


def dequantize(q, k, dtype=torch.float32):
    return q.to(dtype) * _pow2(k, dtype)


def roundtrip(x, fmt="e4m3"):
    """The declared recipe at a boundary: scaled fp8 quantize, then back.

    This is the reference numerics when enabled, not a wire approximation, so
    the single-device baseline runs the identical cast at the identical points.
    """
    q, k = quantize(x, fmt)
    return dequantize(q, k, x.dtype)


def recover_exp(t, fmt="e4m3"):
    """Per-row power-of-two exponent k making t * 2^-k exactly fp8-representable.

    The codec needs the scale but is handed only the dequantized tensor. Any k
    that makes the row exactly representable is correct, because decode
    multiplies the same k back -- it need not be the k the model used. The
    exponent implied by amax is right to within one octave, so a tiny fixed
    candidate set settles it deterministically.
    """
    dt, ebias = FORMATS[fmt]
    amax = t.abs().amax(dim=-1, keepdim=True)
    _, e = torch.frexp(amax)
    base = torch.where(
        amax > 0, (e - ebias).to(torch.int32), torch.zeros_like(e, dtype=torch.int32)
    )
    best = base
    done = torch.zeros_like(amax, dtype=torch.bool)
    for delta in _DELTAS:
        cand = base + delta
        q = (t * _pow2(-cand, t.dtype)).to(dt)
        ok = (q.to(t.dtype) * _pow2(cand, t.dtype) == t).all(dim=-1, keepdim=True)
        best = torch.where(done, best, cand)
        done = done | ok
        if bool(done.all()):
            break
    if not bool(done.all()):
        raise ValueError("boundary tensor is not scaled-FP8 representable: recipe mismatch")
    return best
