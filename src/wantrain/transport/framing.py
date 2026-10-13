"""TCP framing for boundary tensors and data-plane control messages.

Frame (little-endian): magic u16=0xE17A | type u8 | codec u8 | step u32 |
mb u32 | chunk u16 | ndim u8 | frag_idx u16 | frag_cnt u16 | shape u32*ndim |
paylen u64 | payload. chunk is the sending peer in a wide stage, else 0. Fragments are byte ranges of one
payload reassembled by concatenation, so adaptive microbatching is exact by
construction. RAW codecs are exact byte images; OBJ (transport/serial.py) is
for control objects (ZeRO-1 shards, losses) and never executes
code on decode.

END is the graceful-shutdown marker: a sender emits it after draining its
queue, so the reader on the other side knows the peer finished on purpose and
a following EOF is not a failure.

TIED carries rows of a tied table between stage 0 and the Head (ADR-0009,
I-5): codec RAW_F32, ndim 2, shape (rows, h), and a whole payload of exactly
rows*h*4 bytes. A fragment carries the whole shape and a byte range of that
payload; the reassembled payload must have the exact length.

Decode is typed: a type outside FRAME_TYPES is refused before its payload is
read, so an unassigned id (a rolled-back TIED included) is never dispatched.
"""

from __future__ import annotations

import os
import socket
import struct
import time
from dataclasses import dataclass

import torch

from wantrain.transport import serial

MAGIC = 0xE17A

FWD, BWD, END, GRADS, PARAMS, LOSS, HELLO, TIED = range(8)
FRAME_TYPES = frozenset({FWD, BWD, END, GRADS, PARAMS, LOSS, HELLO, TIED})
RAW_F32, RAW_I64, FP8_PLANES, OBJ, F32_PLANES, PLANES = range(6)

_HDR = struct.Struct("<HBBIIHBHH")

_DTYPES = {RAW_F32: torch.float32, RAW_I64: torch.int64}

# A peer that claims a multi-terabyte payload is either broken or hostile;
# either way the reader must not try to allocate it.
MAX_PAYLOAD = int(os.environ.get("WANTRAIN_MAX_FRAME_BYTES", str(4 << 30)))


class FrameError(ConnectionError):
    """The byte stream is not a valid frame sequence."""


def _tied_bytes(codec: int, shape: tuple) -> int:
    """The exact payload length of a TIED frame, or FrameError."""
    if codec != RAW_F32 or len(shape) != 2:
        raise FrameError(f"TIED frame needs codec RAW_F32 and ndim 2, got {codec}, {shape}")
    return shape[0] * shape[1] * 4


@dataclass
class Frame:
    type: int
    codec: int
    step: int
    mb: int
    chunk: int
    payload: bytes
    shape: tuple
    frag_idx: int = 0
    frag_cnt: int = 1

    def tensor(self) -> torch.Tensor:
        if self.codec == OBJ:
            raise ValueError("control frame has no tensor")
        if self.codec == FP8_PLANES:
            from wantrain.codec.fp8_planes import decode

            return decode(self.payload, self.shape)
        if self.codec == F32_PLANES:
            from wantrain.codec.f32_planes import decode

            return decode(self.payload, self.shape)
        if self.codec == PLANES:
            # returns the tensor in its declared wire dtype; the receiving stage
            # promotes it to the compute dtype, which it alone knows
            from wantrain.codec.planes import decode

            return decode(self.payload, self.shape)
        if self.codec not in _DTYPES:
            raise FrameError(f"unknown tensor codec {self.codec}")
        if not self.payload:
            # torch.frombuffer rejects an empty buffer; an empty tensor is valid
            return torch.empty(self.shape, dtype=_DTYPES[self.codec])
        t = torch.frombuffer(bytearray(self.payload), dtype=_DTYPES[self.codec])
        return t.reshape(self.shape).clone()

    def obj(self):
        if self.codec != OBJ:
            raise FrameError(f"frame codec {self.codec} carries no object")
        return serial.loads(self.payload)


def tensor_payload(t: torch.Tensor):
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.float32:
        codec = RAW_F32
    elif t.dtype == torch.int64:
        codec = RAW_I64
    else:
        raise ValueError(f"unsupported dtype {t.dtype}")
    return codec, t.numpy().tobytes(), tuple(t.shape)


def send_frame(
    sock: socket.socket,
    ftype: int,
    step: int,
    mb: int,
    payload,
    chunk: int = 0,
    codec: int | None = None,
    shape: tuple = (),
    frag_idx: int = 0,
    frag_cnt: int = 1,
):
    if ftype not in FRAME_TYPES:
        raise ValueError(f"unknown frame type {ftype}")
    if isinstance(payload, torch.Tensor):
        codec, payload, shape = tensor_payload(payload)
    elif codec is None:
        payload, codec = serial.dumps(payload), OBJ
    hdr = _HDR.pack(MAGIC, ftype, codec, step, mb, chunk, len(shape), frag_idx, frag_cnt)
    dims = struct.pack(f"<{len(shape)}I", *shape) if shape else b""
    sock.sendall(hdr + dims + struct.pack("<Q", len(payload)) + payload)


def send_end(sock: socket.socket) -> None:
    """Tell the reader on the other side that this direction is done."""
    send_frame(sock, END, 0, 0, b"", codec=OBJ)


class Reassembler:
    """Collects payload fragments; emits a whole Frame when complete.
    Concatenation in frag_idx order reproduces the original byte string, so
    regranularization can never alter numerics."""

    def __init__(self):
        self._parts: dict = {}
        self._first: dict = {}  # key -> monotonic time its first fragment arrived
        self.spread_s = (
            0.0  # the last whole frame's first fragment to its last (0 for an unfragmented one)
        )

    def push(self, f: Frame):
        if f.frag_cnt <= 1:
            if f.type == TIED and len(f.payload) != _tied_bytes(f.codec, f.shape):
                raise FrameError(f"TIED payload {len(f.payload)} does not match shape {f.shape}")
            self.spread_s = 0.0
            return f
        if f.frag_idx >= f.frag_cnt:
            raise FrameError(f"fragment {f.frag_idx} of {f.frag_cnt}")
        key = (f.type, f.step, f.mb, f.chunk)
        parts = self._parts.setdefault(key, {})
        if not parts:
            self._first[key] = time.monotonic()
        parts[f.frag_idx] = f
        if len(parts) < f.frag_cnt:
            return None
        del self._parts[key]
        self.spread_s = time.monotonic() - self._first.pop(key)
        payload = b"".join(parts[i].payload for i in range(f.frag_cnt))
        first = parts[0]
        if first.type == TIED and len(payload) != _tied_bytes(first.codec, first.shape):
            raise FrameError(f"TIED payload {len(payload)} does not match shape {first.shape}")
        return Frame(
            first.type, first.codec, first.step, first.mb, first.chunk, payload, first.shape
        )


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 22))
        if not chunk:
            raise ConnectionError("peer closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> Frame:
    magic, ftype, codec, step, mb, chunk, ndim, fi, fc = _HDR.unpack(_recv_exact(sock, _HDR.size))
    if magic != MAGIC:
        raise FrameError("bad frame magic")
    if ftype not in FRAME_TYPES:
        raise FrameError(f"unknown frame type {ftype}")
    shape = struct.unpack(f"<{ndim}I", _recv_exact(sock, 4 * ndim)) if ndim else ()
    (paylen,) = struct.unpack("<Q", _recv_exact(sock, 8))
    if paylen > MAX_PAYLOAD:
        raise FrameError(f"frame payload {paylen} exceeds limit {MAX_PAYLOAD}")
    if ftype == TIED:
        whole = _tied_bytes(codec, shape)
        # a fragment carries a byte range of the whole payload; the
        # Reassembler checks the total
        if paylen > whole or (fc <= 1 and paylen != whole):
            raise FrameError(f"TIED payload {paylen} does not match shape {shape}")
    return Frame(ftype, codec, step, mb, chunk, _recv_exact(sock, paylen), shape, fi, fc)
