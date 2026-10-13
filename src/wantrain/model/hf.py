"""Checkpoint digest and the safetensors reader.

The native HF loader (config_from_hf, load_into, the _LLAMA_* maps) is gone
(SPEC-model task 4a, ADR-0009 D11): arch="hf" runs transformers' own modules
through model/hfm.py, which loads checkpoints by HF's own names. What stays is
the digest every stage compares, so stages holding different weights are
refused rather than trained, and the reader both use. Only safetensors is
read: a .bin checkpoint is a pickle, and unpickling a downloaded file executes
whatever it contains.
"""

import hashlib
import json
import os
import struct

import torch

# safetensors dtype codes <-> torch dtypes.
_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
_CODES = {v: k for k, v in _DTYPES.items()}
INDEX = "model.safetensors.index.json"
# A header past this is not a checkpoint this repository wrote or reads.
_MAX_HEADER = 100 * 2**20


def _read_header(file: str) -> tuple[dict, int]:
    """-> ({name: {"dtype", "shape", "data_offsets"}}, offset of the data).

    The file is untrusted input (a downloaded checkpoint): every entry is
    validated here, before any tensor byte is read, as the safetensors format
    requires (non-negative integer dims, 0 <= start <= end within the data,
    end - start equal to the shape's byte size)."""
    with open(file, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{file}: not a safetensors file (no header length)")
        (n,) = struct.unpack("<Q", raw)
        if n > _MAX_HEADER:
            raise ValueError(f"{file}: safetensors header of {n} bytes refused")
        header = json.loads(fh.read(n))
        data_len = os.fstat(fh.fileno()).st_size - 8 - n
    if not isinstance(header, dict) or data_len < 0:
        raise ValueError(f"{file}: not a safetensors file (bad header)")
    header.pop("__metadata__", None)
    for name, info in header.items():
        if not isinstance(info, dict) or info.get("dtype") not in _DTYPES:
            dt = info.get("dtype") if isinstance(info, dict) else info
            raise ValueError(f"{file}: tensor {name} has unsupported dtype {dt!r}")
        shape, offs = info.get("shape"), info.get("data_offsets")
        if not isinstance(shape, list) or not all(type(d) is int and d >= 0 for d in shape):
            raise ValueError(f"{file}: tensor {name} has an invalid shape {shape!r}")
        if (
            not isinstance(offs, list)
            or len(offs) != 2
            or not all(type(o) is int for o in offs)
            or not 0 <= offs[0] <= offs[1] <= data_len
        ):
            raise ValueError(f"{file}: tensor {name} has invalid data_offsets {offs!r}")
        numel = 1
        for d in shape:
            numel *= d
        if (
            offs[1] - offs[0]
            != numel * torch.empty((), dtype=_DTYPES[info["dtype"]]).element_size()
        ):
            raise ValueError(
                f"{file}: tensor {name}'s data_offsets do not match its shape and dtype"
            )
    return header, 8 + n


def _read_tensor(file: str, info: dict, base: int) -> torch.Tensor:
    """One tensor of a safetensors file, its bytes as written."""
    dtype = _DTYPES[info["dtype"]]
    start, end = info["data_offsets"]
    shape = tuple(info["shape"])
    numel = 1
    for d in shape:
        numel *= d
    size = torch.empty((), dtype=dtype).element_size()
    if end - start != numel * size:
        raise ValueError(f"{file}: a tensor's data_offsets do not match its shape and dtype")
    with open(file, "rb") as fh:
        fh.seek(base + start)
        raw = bytearray(fh.read(end - start))
    if len(raw) != end - start:
        raise ValueError(f"{file}: truncated tensor data")
    if numel == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(shape).clone()


def _shard_files(path: str) -> list[str]:
    if not os.path.isdir(path):
        return [path]
    idx = os.path.join(path, INDEX)
    if os.path.exists(idx):
        with open(idx) as fh:
            names = set(json.load(fh)["weight_map"].values())
        files = []
        for v in sorted(names):
            if os.path.basename(v) != v:
                raise ValueError(f"{idx}: shard name {v!r} is not a file of the checkpoint")
            files.append(os.path.join(path, v))
        return files
    return sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors"))


def _index(path: str) -> dict:
    """{name: (file, info, data offset)} over every shard of a checkpoint
    directory (its index when present) or one file. A name in two shards is
    refused."""
    files = _shard_files(path)
    if not files:
        raise FileNotFoundError(
            f"no .safetensors found under {path}. Only safetensors is read: "
            f"a .bin checkpoint is a pickle, and unpickling a file downloaded "
            f"from the internet executes whatever it contains."
        )
    out: dict = {}
    for f in files:
        header, base = _read_header(f)
        for name, info in header.items():
            if name in out:
                raise ValueError(f"tensor {name} appears in two shards of {path}")
            out[name] = (f, info, base)
    return out


def _iter_shards(path: str):
    """Yield (name, tensor) from a checkpoint directory or single file."""
    for name, (f, info, base) in _index(path).items():
        yield name, _read_tensor(f, info, base)


def checkpoint_digest(path: str) -> str:
    """Digest over the checkpoint's tensor names, shapes and bytes.

    Every stage computes this from what it downloaded and the orchestrator
    compares them; stages holding different weights would still train and
    still exchange correctly shaped activations, so nothing else would notice.
    """
    h = hashlib.sha256()
    for name, t in sorted(_iter_shards(path), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(
            t.contiguous().view(torch.uint8).numpy().tobytes()
            if t.dtype == torch.uint8
            else t.contiguous().float().numpy().tobytes()
        )
    return h.hexdigest()
