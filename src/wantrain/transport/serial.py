"""Safe, exact object encoding for control frames and control-plane messages.

The source engine used pickle for every non-tensor frame (ZeRO-1 gradient and
parameter exchange, loss reports) and for the
master/worker protocol. Unpickling bytes read from a socket executes whatever
the sender wrote, which on a WAN fleet means any host that can reach a port can
run code on a worker. This codec carries the same object shapes with no code
path from bytes to execution:

    header u32 | JSON structure | tensor blobs

The JSON structure is a tagged tree: tuples, non-string dict keys and tensors
are wrapped in single-key objects so they decode back to the same Python types
(the runtime dispatches on `isinstance(msg, tuple)`). Tensors travel as their
exact byte image with dtype and shape, so a gradient decodes to the identical
bits it was encoded from. Floats survive because json emits repr(), which
round-trips every finite float exactly, and NaN/Infinity are allowed.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np
import torch

_TENSOR = "__t"
_TUPLE = "__tuple"
_DICT = "__dict"
_BYTES = "__b"

# Element types a control frame may carry. Append only.
_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
_NAMES = {v: k for k, v in _DTYPES.items()}

# Integer view of matching width, so dtypes numpy lacks (bf16, fp8) still have
# an exact byte image.
_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}

_HDR = struct.Struct("<I")


class SerialError(ValueError):
    """Malformed or unsupported payload."""


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bool:
        return t.numpy().tobytes()
    return t.view(_VIEW[t.element_size()]).numpy().tobytes()


def _tensor_from(buf: bytes, dtype_name: str, shape: list[int]) -> torch.Tensor:
    if dtype_name not in _DTYPES:
        raise SerialError(f"unsupported tensor dtype {dtype_name!r}")
    dt = _DTYPES[dtype_name]
    if dt == torch.bool:
        arr = np.frombuffer(buf, dtype=np.bool_).copy()
        return torch.from_numpy(arr).reshape(shape)
    esz = torch.empty(0, dtype=dt).element_size()
    view = _VIEW[esz]
    np_view = torch.empty(0, dtype=view).numpy().dtype
    arr = np.frombuffer(buf, dtype=np_view).copy()
    return torch.from_numpy(arr).view(dt).reshape(shape)


def _enc(obj: Any, blobs: list[bytes]) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, torch.Tensor):
        if obj.dtype not in _NAMES:
            raise SerialError(f"cannot encode tensor of dtype {obj.dtype}")
        blobs.append(_tensor_bytes(obj))
        return {_TENSOR: [len(blobs) - 1, _NAMES[obj.dtype], list(obj.shape)]}
    if isinstance(obj, (bytes, bytearray)):
        blobs.append(bytes(obj))
        return {_BYTES: len(blobs) - 1}
    if isinstance(obj, tuple):
        return {_TUPLE: [_enc(x, blobs) for x in obj]}
    if isinstance(obj, list):
        return [_enc(x, blobs) for x in obj]
    if isinstance(obj, dict):
        if all(isinstance(k, str) and not k.startswith("__") for k in obj):
            return {k: _enc(v, blobs) for k, v in obj.items()}
        return {_DICT: [[_enc(k, blobs), _enc(v, blobs)] for k, v in obj.items()]}
    raise SerialError(f"cannot encode {type(obj).__name__}")


def _dec(node: Any, blobs: list[bytes]) -> Any:
    if isinstance(node, list):
        return [_dec(x, blobs) for x in node]
    if not isinstance(node, dict):
        return node
    if len(node) == 1:
        (key,) = node
        val = node[key]
        if key == _TENSOR:
            idx, dtype_name, shape = val
            return _tensor_from(blobs[idx], dtype_name, shape)
        if key == _BYTES:
            return blobs[val]
        if key == _TUPLE:
            return tuple(_dec(x, blobs) for x in val)
        if key == _DICT:
            out = {}
            for k, v in val:
                k = _dec(k, blobs)
                if isinstance(k, list):
                    k = tuple(k)
                out[k] = _dec(v, blobs)
            return out
    return {k: _dec(v, blobs) for k, v in node.items()}


def dumps(obj: Any) -> bytes:
    """Encode `obj` (None/bool/int/float/str/bytes/tensor/list/tuple/dict)."""
    blobs: list[bytes] = []
    tree = json.dumps(_enc(obj, blobs), separators=(",", ":")).encode()
    parts = [_HDR.pack(len(tree)), tree, _HDR.pack(len(blobs))]
    for b in blobs:
        parts.append(struct.pack("<Q", len(b)))
        parts.append(b)
    return b"".join(parts)


def loads(data: bytes) -> Any:
    """Inverse of dumps. Raises SerialError on anything malformed."""
    try:
        mv = memoryview(data)
        (n,) = _HDR.unpack_from(mv, 0)
        off = _HDR.size
        tree = json.loads(bytes(mv[off : off + n]))
        off += n
        (nb,) = _HDR.unpack_from(mv, off)
        off += _HDR.size
        blobs = []
        for _ in range(nb):
            (ln,) = struct.unpack_from("<Q", mv, off)
            off += 8
            if off + ln > len(mv):
                raise SerialError("truncated blob")
            blobs.append(bytes(mv[off : off + ln]))
            off += ln
        if off != len(mv):
            raise SerialError("trailing bytes after payload")
        return _dec(tree, blobs)
    except SerialError:
        raise
    except (struct.error, ValueError, KeyError, IndexError, TypeError) as e:
        raise SerialError(f"malformed payload: {e}") from e
