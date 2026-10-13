"""Lossless raw-fp8 wire codec with nibble-plane entropy coding.

Valid only when the boundary tensor is exactly representable in float8_e4m3fn:
the fp8 byte image is then the exact wire format and nothing is quantized for
the network. encode() asserts representability rather than going lossy. Each
byte is split into a sign+exponent nibble (low entropy in trained activations)
and a mantissa nibble, each packed plane is compressed separately, and decode
reverses exactly.
"""

import struct

import numpy as np
import torch

from wantrain.codec.planes import _compress, _decompress


def encode(t: torch.Tensor):
    t = t.detach().cpu().contiguous()
    f8 = t.to(torch.float8_e4m3fn)
    back = f8.to(t.dtype)
    assert torch.equal(back, t), "boundary is not FP8-representable: recipe mismatch"
    raw = f8.view(torch.uint8).numpy().reshape(-1)
    hi = (raw >> 4).astype(np.uint8)  # sign + exponent (+ top mantissa bit)
    lo = (raw & 0x0F).astype(np.uint8)
    hi_p = np.packbits(np.unpackbits(hi.reshape(-1, 1), axis=1, count=8)[:, 4:])
    lo_p = np.packbits(np.unpackbits(lo.reshape(-1, 1), axis=1, count=8)[:, 4:])
    c_hi = _compress(hi_p.tobytes())
    c_lo = _compress(lo_p.tobytes())
    payload = struct.pack("<QII", raw.size, len(c_hi), len(c_lo)) + c_hi + c_lo
    return payload, tuple(t.shape)


def decode(payload: bytes, shape) -> torch.Tensor:
    n, nhi, nlo = struct.unpack_from("<QII", payload, 0)
    off = struct.calcsize("<QII")
    hi_p = np.frombuffer(_decompress(payload[off : off + nhi]), dtype=np.uint8)
    lo_p = np.frombuffer(_decompress(payload[off + nhi : off + nhi + nlo]), dtype=np.uint8)
    hi = np.packbits(
        np.concatenate(
            [np.zeros((n, 4), dtype=np.uint8), np.unpackbits(hi_p)[: n * 4].reshape(n, 4)], axis=1
        )
    )
    lo = np.packbits(
        np.concatenate(
            [np.zeros((n, 4), dtype=np.uint8), np.unpackbits(lo_p)[: n * 4].reshape(n, 4)], axis=1
        )
    )
    raw = ((hi << 4) | lo).astype(np.uint8)
    f8 = torch.from_numpy(raw).view(torch.float8_e4m3fn).reshape(shape)
    return f8.to(torch.float32)
