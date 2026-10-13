"""Lossless fp32 wire codec: byte-plane split plus entropy coding.

Gradients of trained layers concentrate in a narrow exponent band, so splitting
each float's 4 bytes into planes (all byte-0s, then byte-1s, ...) groups the
low-entropy exponent bytes and lets the coder recover a lossless factor. Decode
restores the exact original byte string, so bit-exactness holds by construction.
"""

import struct

import numpy as np
import torch

from wantrain.codec.planes import _compress, _decompress


def encode(t: torch.Tensor):
    t = t.detach().cpu().contiguous()
    assert t.dtype == torch.float32
    # Flatten first: numpy cannot reinterpret a 0-d array's dtype, and a
    # scalar parameter's gradient is a 0-d tensor.
    raw = t.reshape(-1).numpy().view(np.uint8).reshape(-1, 4)
    planes = raw.T.copy().tobytes()  # plane-major: exponent bytes cluster
    comp = _compress(planes)
    payload = struct.pack("<Q", raw.shape[0]) + comp
    return payload, tuple(t.shape)


def decode(payload: bytes, shape) -> torch.Tensor:
    (n,) = struct.unpack_from("<Q", payload, 0)
    planes = np.frombuffer(_decompress(payload[8:]), dtype=np.uint8)
    raw = planes.reshape(4, n).T.copy()
    return torch.from_numpy(raw.view(np.float32).reshape(shape).copy())
