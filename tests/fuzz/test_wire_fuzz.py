"""Property fuzzing of everything that carries exact bytes over the wire.

Codecs are fed arbitrary bit patterns of every wire dtype (NaN payloads,
infinities, -0.0 and subnormals included), so "lossless" is checked on the
bits, not on values. Frames and fragmented payloads go through a real socket
pair in arbitrary pieces. The control-object decoder and the frame parser are
fed arbitrary bytes and may only reject them with their documented errors.

WOULD-FAIL-IF: a wire codec changed one bit of any tensor of a supported
dtype, fragment reassembly depended on arrival order or fragment sizes, a
frame did not survive the socket intact, or malformed input reached the
engine as anything but SerialError / FrameError / ConnectionError.
"""

import socket
import threading

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from wantrain.codec import f32_planes, fp8_planes, planes
from wantrain.transport import framing, serial

WIRE_DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int64,
]
shapes = st.lists(st.integers(0, 9), min_size=0, max_size=3).map(tuple)


def _bits(t: torch.Tensor) -> bytes:
    flat = t.contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""


@st.composite
def raw_tensors(draw, dtypes=WIRE_DTYPES):
    dtype = draw(st.sampled_from(dtypes))
    shape = draw(shapes)
    n = int(np.prod(shape)) if shape else 1
    esz = torch.empty(0, dtype=dtype).element_size()
    raw = draw(st.binary(min_size=n * esz, max_size=n * esz))
    u8 = (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        if raw
        else torch.empty(0, dtype=torch.uint8)
    )
    return u8.view(dtype).reshape(shape)


@given(raw_tensors())
def test_planes_codec_is_bit_exact_for_every_bit_pattern(t):
    payload, shape = planes.encode(t)
    back = planes.decode(payload, shape)
    assert back.dtype == t.dtype and tuple(back.shape) == tuple(t.shape)
    assert _bits(back) == _bits(t)


@given(raw_tensors([torch.float32]))
def test_f32_planes_codec_is_bit_exact_for_every_bit_pattern(t):
    payload, shape = f32_planes.encode(t)
    back = f32_planes.decode(payload, shape)
    assert _bits(back) == _bits(t)


@given(raw_tensors([torch.float8_e4m3fn]), st.sampled_from([torch.float32, torch.bfloat16]))
def test_fp8_planes_codec_is_bit_exact_on_representable_tensors(f8, wide):
    f8 = torch.where(f8.float().isnan(), torch.zeros_like(f8.float()), f8.float()).to(
        torch.float8_e4m3fn
    )
    t = f8.to(wide)
    payload, shape = fp8_planes.encode(t)
    back = fp8_planes.decode(payload, shape).to(wide)
    assert _bits(back) == _bits(t)


objects = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(2**63), 2**63 - 1)
    | st.floats(allow_nan=False)
    | st.text()
    | st.binary(),
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=4)
    ),
    max_leaves=12,
)


@given(objects)
def test_control_objects_round_trip(obj):
    assert serial.loads(serial.dumps(obj)) == obj


@given(st.binary(max_size=512))
def test_malformed_control_objects_raise_only_serial_error(data):
    try:
        serial.loads(data)
    except serial.SerialError:
        pass


def _through_socket(data: bytes, pieces: list[int]):
    a, b = socket.socketpair()

    def write():
        i = k = 0
        while i < len(data):
            n = pieces[k % len(pieces)]
            a.sendall(data[i : i + n])
            i += n
            k += 1
        a.shutdown(socket.SHUT_WR)

    t = threading.Thread(target=write, daemon=True)
    t.start()
    frames = []
    try:
        while True:
            frames.append(framing.recv_frame(b))
    except ConnectionError as e:  # FrameError is a ConnectionError
        end = e
    finally:
        t.join(10)
        a.close()
        b.close()
    return frames, end


@given(
    raw_tensors(),
    st.booleans(),
    st.lists(st.integers(1, 97), min_size=1, max_size=8),
    st.integers(0, 2**31),
    st.integers(0, 2**31),
)
def test_frames_survive_arbitrary_segmentation(t, raw, pieces, step, mb):
    buf = bytearray()

    class Sink:
        def sendall(self, d):
            buf.extend(d)

    if raw and t.dtype in (torch.float32, torch.int64):
        framing.send_frame(Sink(), framing.FWD, step, mb, t)
    else:
        payload, shape = planes.encode(t)
        framing.send_frame(
            Sink(), framing.FWD, step, mb, payload, codec=framing.PLANES, shape=shape
        )
    framing.send_end(Sink())
    frames, end = _through_socket(bytes(buf), pieces)
    assert len(frames) == 2 and frames[1].type == framing.END, (frames, end)
    f = frames[0]
    assert (f.type, f.step, f.mb) == (framing.FWD, step, mb)
    back = f.tensor()
    assert back.dtype == t.dtype and tuple(back.shape) == tuple(t.shape)
    assert _bits(back) == _bits(t)


@given(st.binary(max_size=256), st.lists(st.integers(1, 64), min_size=1, max_size=4))
def test_garbage_streams_end_only_in_frame_or_connection_errors(data, pieces):
    frames, end = _through_socket(data, pieces)
    assert isinstance(end, ConnectionError)


@given(st.binary(min_size=0, max_size=2000), st.integers(1, 12), st.randoms(use_true_random=False))
def test_fragment_reassembly_is_order_and_size_independent(payload, cnt, rnd):
    cuts = sorted(rnd.sample(range(len(payload) + 1), min(cnt - 1, len(payload) + 1)))
    bounds = [0, *cuts, len(payload)]
    parts = [payload[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    frags = [
        framing.Frame(framing.GRADS, framing.RAW_F32, 3, 5, 1, p, (len(payload),), i, len(parts))
        for i, p in enumerate(parts)
    ]
    rnd.shuffle(frags)
    r = framing.Reassembler()
    out = [x for x in (r.push(f) for f in frags) if x is not None]
    assert len(out) == 1 and out[0].payload == payload
