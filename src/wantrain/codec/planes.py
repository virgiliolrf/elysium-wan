"""Lossless wire codec for boundary tensors of any declared precision.

No quantization happens here: the tensor arrives already in the dtype the
recipe declared (model/precision.py) and its exact bytes are shipped; decode
returns a bit-identical tensor. Multi-byte elements are split into byte planes
so the coder sees a whole plane of low-entropy exponent bytes (the mantissa is
near-uniform noise). Measured on d_model=1600 boundary tensors, planes beat
interleaved bytes for 2- and 4-byte elements, while at one byte per element a
nibble split cost 5%, so single-byte formats go interleaved.
"""

import struct
import threading
import zlib

import numpy as np
import torch

# Compression backend: zstd level 1, with zlib as the fallback. On a 6.3 MB
# bf16 boundary zlib-6 ran at ~7 MB/s, slower than the microbatch's own
# compute window, while zstd-1 gave a better ratio at ~500 MB/s. Decode sniffs
# the container magic per blob, so payloads from either backend decode.
try:
    import zstandard as _zstd
except ImportError:  # boxes without the wheel still work
    _zstd = None  # type: ignore[assignment]
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

# Per thread, never shared: a ZstdCompressor holds internal state and is not
# safe to call from two threads at once, and the engine compresses from both
# the sender and the mux-encoder thread while every socket reader decompresses.
# Building a level-1 context costs microseconds against compressing megabytes.
_local = threading.local()


def _ctx():
    c = getattr(_local, "zc", None)
    if c is None:
        c = _local.zc = _zstd.ZstdCompressor(level=1)
        _local.zd = _zstd.ZstdDecompressor()
    return c, _local.zd


def _compress(b: bytes) -> bytes:
    if _zstd is not None:
        return _ctx()[0].compress(b)
    # zlib level 1, not 6: on these payload sizes level 6 buys 2.5% ratio for
    # 6x the CPU, and CPU here is compute-window stolen from the GPU.
    return zlib.compress(b, 1)


def _decompress(b: bytes) -> bytes:
    if b[:4] == _ZSTD_MAGIC:
        if _zstd is None:
            raise RuntimeError(
                "payload is zstd-compressed but the zstandard module is not "
                "installed on this box; the bootstrap must install it"
            )
        return _ctx()[1].decompress(b)
    return zlib.decompress(b)


# wire dtype ids; append only, never renumber (old payloads must stay readable)
_DT = {
    0: torch.float32,
    1: torch.bfloat16,
    2: torch.float16,
    3: torch.float8_e4m3fn,
    4: torch.float8_e5m2,
    5: torch.int64,
}
_ID = {v: k for k, v in _DT.items()}

# signed integer view of matching width (portable across torch/numpy)
_IVIEW = {
    1: (torch.int8, np.int8, np.uint8),
    2: (torch.int16, np.int16, np.uint16),
    4: (torch.int32, np.int32, np.uint32),
    8: (torch.int64, np.int64, np.uint64),
}

_HDR = struct.Struct("<BBQ")  # dtype id, plane count, element count


def encode_scaled_fp8(t: torch.Tensor, fmt: str = "e4m3"):
    """Wire format for a per-token power-of-two scaled fp8 boundary.

    The tensor arrives as float q * 2^k with a per-row exponent k (an unscaled
    fp8 cast zeroes activation gradients), so it is fp8-representable only after
    dividing by its row's scale. Payload is the fp8 byte image plus one int8
    exponent per row (0.06% overhead at d_model=1600); k is recovered from the
    tensor itself, so the codec interface stays uniform and the assertion local.
    """
    from wantrain.model import fp8 as fp8mod

    t = t.detach().cpu().contiguous()
    flat = t.reshape(-1, t.shape[-1]) if t.dim() > 1 else t.reshape(1, -1)
    k = fp8mod.recover_exp(flat, fmt)  # (rows, 1) int32
    dt = fp8mod.FORMATS[fmt][0]
    q = (flat * torch.exp2(-k.to(flat.dtype))).to(dt)

    # Prove the round trip before anything leaves this process: a silent mismatch
    # here would be a lossy wire disguised as a lossless one.
    back = q.to(flat.dtype) * torch.exp2(k.to(flat.dtype))
    if not torch.equal(back, flat):
        raise ValueError(
            "boundary is not scaled-FP8 representable: recipe mismatch. The tensor "
            "reaching the wire must be the recipe's own fp8 round trip."
        )

    raw = q.view(torch.int8).numpy().view(np.uint8).reshape(-1)
    exps = k.reshape(-1).to(torch.int8).numpy().view(np.uint8)
    c_v = _compress(raw.tobytes())
    c_k = _compress(exps.tobytes())
    head = struct.pack("<BBQ", _SCALED_ID[fmt], 2, raw.size)
    head += struct.pack("<I", len(c_v)) + struct.pack("<I", len(c_k))
    return head + c_v + c_k, tuple(t.shape)


def _decode_scaled_fp8(payload: bytes, shape, did: int, sizes, off: int):
    fmt = _SCALED_FMT[did]
    from wantrain.model import fp8 as fp8mod

    dt = fp8mod.FORMATS[fmt][0]
    raw = np.frombuffer(_decompress(payload[off : off + sizes[0]]), dtype=np.uint8)
    off += sizes[0]
    exps = np.frombuffer(_decompress(payload[off : off + sizes[1]]), dtype=np.uint8)
    q = torch.from_numpy(raw.view(np.int8).copy()).view(dt)
    k = torch.from_numpy(exps.view(np.int8).copy()).to(torch.int32)
    rows = k.numel()
    q = q.reshape(rows, -1)
    out = q.to(torch.float32) * torch.exp2(k.reshape(rows, 1).to(torch.float32))
    return out.reshape(shape)


# Scaled-fp8 wire ids live above the plain dtype ids; append only.
_SCALED_ID = {"e4m3": 32, "e5m2": 33}
_SCALED_FMT = {v: k for k, v in _SCALED_ID.items()}


def encode(t: torch.Tensor):
    """Tensor in its declared boundary dtype -> (payload, shape). Lossless."""
    t = t.detach().cpu().contiguous()
    if t.dtype not in _ID:
        raise ValueError(f"no wire encoding for dtype {t.dtype}")
    esz = t.element_size()
    _, snp, unp = _IVIEW[esz]
    u = t.view(_IVIEW[esz][0]).numpy().view(unp).reshape(-1)
    n = u.size

    if esz == 1:
        blobs = [_compress(u.tobytes())]
    else:
        blobs = [
            _compress(((u >> sh) & 0xFF).astype(np.uint8).tobytes())
            for sh in range(8 * esz - 8, -1, -8)
        ]

    head = _HDR.pack(_ID[t.dtype], len(blobs), n)
    head += b"".join(struct.pack("<I", len(b)) for b in blobs)
    return head + b"".join(blobs), tuple(t.shape)


def decode(payload: bytes, shape) -> torch.Tensor:
    did, nplanes, n = _HDR.unpack_from(payload, 0)
    off = _HDR.size
    sizes = []
    for _ in range(nplanes):
        sizes.append(struct.unpack_from("<I", payload, off)[0])
        off += 4

    if did in _SCALED_FMT:
        return _decode_scaled_fp8(payload, shape, did, sizes, off)

    dtype = _DT[did]
    esz = torch.empty(0, dtype=dtype).element_size()
    tview, snp, unp = _IVIEW[esz]

    if nplanes == 1:
        u = np.frombuffer(_decompress(payload[off : off + sizes[0]]), dtype=unp)
    else:
        acc = np.zeros(n, dtype=unp)
        for i, sz in enumerate(sizes):
            plane = np.frombuffer(_decompress(payload[off : off + sz]), dtype=np.uint8)
            acc |= plane.astype(unp) << unp(8 * (esz - 1 - i))
            off += sz
        u = acc

    out = torch.from_numpy(np.ascontiguousarray(u.view(snp)).copy())
    return out.view(dtype).reshape(shape)
