"""Wire layer: safe object codec, framing, fragments, END handshake, limits."""

import hashlib
import hmac
import json
import os
import pickle
import random
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

import pytest
import torch
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

from wantrain.transport import control, identity, secure, serial, shim
from wantrain.transport import framing as fr
from wantrain.transport.control import ChannelClosed
from wantrain.transport.shim import DelayedDelivery, LinkSpec, SenderPacer, SharedBudget


def _bits(t):
    return t.contiguous().view(
        torch.uint8
        if t.element_size() == 1
        else {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.int64,
    ],
)
def test_serial_tensor_roundtrip_is_bit_exact(dtype):
    # covers: TRANSPORT-I1
    g = torch.Generator().manual_seed(3)
    t = (torch.randn(3, 5, generator=g) * 7).to(dtype)
    back = serial.loads(serial.dumps(t))
    assert back.dtype == dtype and back.shape == t.shape
    assert torch.equal(_bits(back), _bits(t))


def test_serial_preserves_the_shapes_the_runtime_dispatches_on():
    # WOULD-FAIL-IF: tuples decode as lists (stage.py checks isinstance(msg, tuple))
    # covers: TRANSPORT-I1
    msg = ("mb", 3, [torch.ones(2), torch.zeros(1, dtype=torch.int64)])
    back = serial.loads(serial.dumps(msg))
    assert isinstance(back, tuple) and back[0] == "mb" and back[1] == 3
    assert torch.equal(back[2][0], torch.ones(2))
    # dict with int keys (batch GRADS exchange) and float payloads exact
    d = {0: [torch.full((2,), 1.5)], 7: [torch.full((1,), -2.0)]}
    back = serial.loads(serial.dumps(d))
    assert set(back) == {0, 7} and torch.equal(back[7][0], torch.full((1,), -2.0))
    losses = {"0:1": 0.1 + 0.2, "1:0": float("inf")}
    assert serial.loads(serial.dumps(losses)) == losses


def test_serial_rejects_garbage_and_never_unpickles():
    # covers: TRANSPORT-I1
    import pickle

    with pytest.raises(serial.SerialError):
        serial.loads(pickle.dumps({"a": 1}))
    with pytest.raises(serial.SerialError):
        serial.loads(b"\x00\x00\x00\x05abcde")
    good = serial.dumps([1, 2])
    with pytest.raises(serial.SerialError):
        serial.loads(good + b"x")
    with pytest.raises(serial.SerialError):
        serial.dumps(object())


def _pair():
    return _track(*socket.socketpair())


def test_frame_roundtrip_tensor_and_object():
    # covers: TRANSPORT-I2
    a, b = _pair()
    t = torch.randn(4, 3)
    fr.send_frame(a, fr.FWD, 5, 2, t, chunk=1)
    fr.send_frame(a, fr.LOSS, 5, 0, ("loss", None))
    f1 = fr.recv_frame(b)
    assert (f1.type, f1.step, f1.mb, f1.chunk) == (fr.FWD, 5, 2, 1)
    assert torch.equal(f1.tensor(), t)
    f2 = fr.recv_frame(b)
    assert f2.codec == fr.OBJ and f2.obj() == ("loss", None)


def test_fragments_reassemble_to_the_exact_payload_in_any_order():
    # covers: TRANSPORT-I2
    payload = bytes(range(256)) * 37
    parts = [payload[i : i + 1000] for i in range(0, len(payload), 1000)]
    frames = [fr.Frame(fr.BWD, fr.OBJ, 1, 2, 0, p, (), i, len(parts)) for i, p in enumerate(parts)]
    r = fr.Reassembler()
    out = None
    for f in reversed(frames):
        out = r.push(f) or out
    assert out is not None and out.payload == payload


def test_end_frame_and_payload_limit():
    # covers: TRANSPORT-I3
    a, b = _pair()
    fr.send_end(a)
    assert fr.recv_frame(b).type == fr.END
    # a hostile length is refused before allocation
    hdr = struct.Struct("<HBBIIHBHH").pack(fr.MAGIC, fr.FWD, fr.RAW_F32, 0, 0, 0, 0, 0, 1)
    a.sendall(hdr + struct.pack("<Q", fr.MAX_PAYLOAD + 1))
    with pytest.raises(fr.FrameError):
        fr.recv_frame(b)
    c, d = _pair()
    c.sendall(b"\x00" * 64)
    with pytest.raises(fr.FrameError):
        fr.recv_frame(d)


_FRAME_HDR = struct.Struct("<HBBIIHBHH")


def _raw_frame(ftype, codec, shape, payload, paylen=None, fi=0, fc=1):
    """Frame bytes built by hand, so a test can put on the wire what send_frame refuses."""
    hdr = _FRAME_HDR.pack(fr.MAGIC, ftype, codec, 3, 4, 0, len(shape), fi, fc)
    dims = struct.pack(f"<{len(shape)}I", *shape) if shape else b""
    n = len(payload) if paylen is None else paylen
    return hdr + dims + struct.pack("<Q", n) + payload


def test_tied_frame_ids_and_fwd_header_bytes_are_unchanged():
    # covers: TRANSPORT-I2
    # WOULD-FAIL-IF: TIED takes an existing type id, an existing id moves, FRAME_TYPES misses a
    # type or admits an unassigned one, or the FWD header layout changes.
    ids = (fr.FWD, fr.BWD, fr.END, fr.GRADS, fr.PARAMS, fr.LOSS, fr.HELLO, fr.TIED)
    assert ids == (0, 1, 2, 3, 4, 5, 6, 7)
    assert fr.FRAME_TYPES == frozenset(ids)
    buf = bytearray()

    class Sink:
        def sendall(self, d):
            buf.extend(d)

    fr.send_frame(
        Sink(),
        fr.FWD,
        0x01020304,
        0x0A0B0C0D,
        torch.zeros(2, 3),
        chunk=0x0102,
        frag_idx=1,
        frag_cnt=3,
    )
    # golden bytes of the whole 35-byte FWD header (19-byte fixed part, 2 dims, u64 paylen 24),
    # recorded from the framing module before TIED was added
    golden = "7ae10000040302010d0c0b0a0201020100030002000000030000001800000000000000"
    assert bytes(buf[:35]).hex() == golden
    assert bytes(buf[35:]) == bytes(24)


def test_tied_frame_round_trips_rows_exactly():
    # covers: TRANSPORT-I2
    # WOULD-FAIL-IF: a well-formed TIED frame is refused, or its fields or row bytes change in transit.
    a, b = _pair()
    rows = torch.arange(15, dtype=torch.float32).reshape(5, 3) * 0.25 - 1.0
    fr.send_frame(a, fr.TIED, 9, 2, rows)
    f = fr.recv_frame(b)
    assert (f.type, f.codec, f.step, f.mb, f.chunk, f.shape) == (
        fr.TIED,
        fr.RAW_F32,
        9,
        2,
        0,
        (5, 3),
    )
    assert f.payload == rows.numpy().tobytes()
    assert torch.equal(fr.Reassembler().push(f).tensor(), rows)


@pytest.mark.parametrize(
    "codec, shape, paylen",
    [
        (fr.OBJ, (2, 3), 24),  # another codec, right length
        (fr.RAW_I64, (2, 3), 24),
        (fr.PLANES, (2, 3), 24),
        (fr.RAW_F32, (6,), 24),  # ndim 1
        (fr.RAW_F32, (1, 2, 3), 24),  # ndim 3
        (fr.RAW_F32, (2, 3), 20),  # short payload
        (fr.RAW_F32, (2, 3), 28),  # long payload
        (fr.RAW_F32, (2, 3), 0),
    ],
)
def test_tied_frame_with_another_codec_ndim_or_length_is_refused(codec, shape, paylen):
    # covers: TRANSPORT-I2
    # WOULD-FAIL-IF: a TIED frame with a codec other than RAW_F32, an ndim other than 2, or a
    # paylen other than rows*h*4 is accepted at receipt.
    a, b = _pair()
    b.settimeout(10)
    a.sendall(_raw_frame(fr.TIED, codec, shape, b"\x01" * paylen))
    with pytest.raises(fr.FrameError):
        fr.recv_frame(b)


def test_tied_frame_fragments_reassemble_exactly_in_any_order():
    # covers: TRANSPORT-I2
    # WOULD-FAIL-IF: TIED fragments are refused at receipt for carrying a byte range of the
    # payload, or the arrival order of fragments changes the reassembled bytes.
    rows = torch.randn(7, 5, generator=torch.Generator().manual_seed(11))
    payload = rows.numpy().tobytes()
    parts = [payload[i : i + 32] for i in range(0, len(payload), 32)]
    a, b = _pair()
    b.settimeout(10)
    for i, p in enumerate(parts):
        a.sendall(_raw_frame(fr.TIED, fr.RAW_F32, (7, 5), p, fi=i, fc=len(parts)))
    frames = [fr.recv_frame(b) for _ in parts]
    rng = random.Random(5)
    for order in (list(range(len(parts))), list(reversed(range(len(parts)))), None):
        if order is None:
            order = list(range(len(parts)))
            rng.shuffle(order)
        r = fr.Reassembler()
        outs = [r.push(frames[i]) for i in order]
        assert all(o is None for o in outs[:-1])
        out = outs[-1]
        assert (out.type, out.shape) == (fr.TIED, (7, 5))
        assert out.payload == payload
        assert torch.equal(out.tensor(), rows)


def test_tied_frame_fragments_with_a_wrong_total_are_refused():
    # covers: TRANSPORT-I2
    # WOULD-FAIL-IF: TIED fragments whose total is not rows*h*4 reassemble into a frame, or one
    # fragment longer than the whole payload is accepted at receipt.
    a, b = _pair()
    b.settimeout(10)
    a.sendall(_raw_frame(fr.TIED, fr.RAW_F32, (2, 3), b"\x00" * 28, fi=0, fc=2))
    with pytest.raises(fr.FrameError):
        fr.recv_frame(b)
    r = fr.Reassembler()
    short = [fr.Frame(fr.TIED, fr.RAW_F32, 1, 0, 0, b"\x00" * 8, (2, 3), i, 2) for i in range(2)]
    assert r.push(short[1]) is None
    with pytest.raises(fr.FrameError):
        r.push(short[0])
    with pytest.raises(fr.FrameError):
        fr.Reassembler().push(fr.Frame(fr.TIED, fr.RAW_F32, 1, 0, 0, b"\x00" * 20, (2, 3)))


def test_tied_frame_over_max_payload_is_refused_before_reading_it():
    # covers: TRANSPORT-I3
    # WOULD-FAIL-IF: MAX_PAYLOAD stops applying to TIED, so a TIED frame whose rows*h*4 is above
    # it is read (or allocated) instead of refused.
    rows = (
        fr.MAX_PAYLOAD // 4 + 1
    )  # rows x 1 x 4 bytes > MAX_PAYLOAD, and consistent with its shape
    a, b = _pair()
    b.settimeout(10)
    a.sendall(_raw_frame(fr.TIED, fr.RAW_F32, (rows, 1), b"", paylen=rows * 4))
    with pytest.raises(fr.FrameError, match="exceeds limit"):
        fr.recv_frame(b)


@pytest.mark.parametrize("ftype", [8, 9, 0x7F, 0xFF])
def test_unknown_frame_type_is_refused_before_its_payload(ftype):
    # covers: TRANSPORT-I2
    # WOULD-FAIL-IF: recv_frame accepts a type outside FRAME_TYPES, or reads its shape, length or
    # payload (a hostile length would be allocated) before refusing it.
    a, b = _pair()
    b.settimeout(10)
    # the bytes after the bad header are a whole valid frame: refusing before reading anything
    # past the header leaves it intact on the stream
    hdr = _FRAME_HDR.pack(fr.MAGIC, ftype, fr.RAW_F32, 0, 0, 0, 0, 0, 1)
    a.sendall(hdr)
    fr.send_frame(a, fr.LOSS, 4, 1, ("after", ftype))
    with pytest.raises(fr.FrameError, match="unknown frame type"):
        fr.recv_frame(b)
    f = fr.recv_frame(b)
    assert (f.type, f.step, f.mb, f.obj()) == (fr.LOSS, 4, 1, ("after", ftype))


@pytest.mark.parametrize("ftype", [8, 200, -1])
def test_unknown_frame_type_is_refused_by_send_frame(ftype):
    # covers: TRANSPORT-I2
    # WOULD-FAIL-IF: send_frame puts a type outside FRAME_TYPES on the wire.
    sent = []

    class Sink:
        def sendall(self, d):
            sent.append(d)

    with pytest.raises(ValueError, match="unknown frame type"):
        fr.send_frame(Sink(), ftype, 0, 0, torch.zeros(2))
    assert sent == []


def test_shim_delays_delivery_by_the_link_latency():
    # covers: TRANSPORT-I6
    import time

    got = []
    ev = threading.Event()
    dd = DelayedDelivery(
        LinkSpec(latency_ms=80), lambda x: (got.append((x, time.monotonic())), ev.set())
    )
    t0 = time.monotonic()
    dd.push("frame")
    assert ev.wait(2)
    assert got[0][0] == "frame"
    assert got[0][1] - t0 >= 0.075


# Shared pacers (TRANSPORT-I10). Wall-clock rate tests: run alone (HANDOUT-M2b note 9).
_RATE_MBPS = 80.0  # 10 MB/s
_CHUNK = 256 * 1024  # 26.2 ms of the budget per chunk
_FLOW_S = 2.0


def _run_flows(pace_fns, seconds=_FLOW_S, chunk=_CHUNK):
    """One thread per pace function, each pacing chunk-sized chunks until the
    deadline; returns the aggregate rate in Mbit/s over [start, last chunk paced]."""
    sent = [0] * len(pace_fns)
    last = [0.0] * len(pace_fns)
    go = threading.Event()
    t0 = 0.0

    def flow(i, pace):
        go.wait()
        while time.monotonic() < t0 + seconds:
            pace(chunk)
            sent[i] += chunk
        last[i] = time.monotonic()

    threads = [threading.Thread(target=flow, args=(i, f)) for i, f in enumerate(pace_fns)]
    for t in threads:
        t.start()
    t0 = time.monotonic()
    go.set()
    for t in threads:
        t.join()
    return sum(sent) * 8 / 1e6 / (max(last) - t0)


@pytest.mark.parametrize("flows", [1, 2, 4, 8])
def test_shared_pacer_egress_aggregate_stays_within_5pct_of_the_budget(flows):
    # covers: TRANSPORT-I10
    # WOULD-FAIL-IF: SenderPacer paces shared-mode links per connection (each flow gets the whole budget),
    # or the bucket loses time to late wake-ups (1 flow falls below the budget).
    egress = SharedBudget(_RATE_MBPS)
    try:
        pacers = [SenderPacer(LinkSpec(egress=egress)) for _ in range(flows)]
        rate = _run_flows([p.pace for p in pacers])
    finally:
        egress.close()
    assert abs(rate - _RATE_MBPS) <= 0.05 * _RATE_MBPS, (flows, rate)


def test_shared_pacer_per_connection_control_exceeds_the_budget():
    # covers: TRANSPORT-I10
    # WOULD-FAIL-IF: the rate tests cannot discriminate (per-connection pacing would also look like one budget).
    # Today's per-connection pacer loses each sleep's late wake-up (measured ~3 ms on macOS), so the control
    # uses 1 MiB chunks (105 ms each) to keep that loss small next to the 2x it must show.
    pacers = [SenderPacer(LinkSpec(bw_mbps=_RATE_MBPS)) for _ in range(2)]
    rate = _run_flows([p.pace for p in pacers], chunk=4 * _CHUNK)
    assert rate >= 1.8 * _RATE_MBPS, rate


def test_shared_pacer_chunk_waits_for_the_slowest_budget_it_crosses():
    # covers: TRANSPORT-I10
    # WOULD-FAIL-IF: SenderPacer ignores the uplink, or pays egress and uplink one after the other
    # (one flow would run at 160 Mbit/s or at 1/(1/160 + 1/80) = 53 Mbit/s instead of 80).
    egress = SharedBudget(2 * _RATE_MBPS)
    uplink = SharedBudget(_RATE_MBPS)
    try:
        rate = _run_flows([SenderPacer(LinkSpec(egress=egress, uplink=uplink)).pace])
        # two peers on one site, each with its own egress, share the uplink
        egress2 = SharedBudget(2 * _RATE_MBPS)
        try:
            pair = [SenderPacer(LinkSpec(egress=e, uplink=uplink)).pace for e in (egress, egress2)]
            pair_rate = _run_flows(pair)
        finally:
            egress2.close()
    finally:
        egress.close()
        uplink.close()
    assert abs(rate - _RATE_MBPS) <= 0.05 * _RATE_MBPS, rate
    assert abs(pair_rate - _RATE_MBPS) <= 0.05 * _RATE_MBPS, pair_rate


_CHILD = """
import sys, time
from wantrain.transport.shim import LinkSpec, SenderPacer, SharedBudget
mbps, path, chunk, seconds = float(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
pacer = SenderPacer(LinkSpec(uplink=SharedBudget(mbps, path)))
print("ready", flush=True)
t0 = float(sys.stdin.readline())
time.sleep(max(0.0, t0 - time.monotonic()))
sent = 0
while time.monotonic() < t0 + seconds:
    pacer.pace(chunk)
    sent += chunk
print(sent, time.monotonic(), flush=True)
"""


@pytest.mark.parametrize("procs", [2, 4])
def test_shared_pacer_uplink_is_shared_across_processes(procs):
    # covers: TRANSPORT-I10
    # WOULD-FAIL-IF: the budget's state is per process (a copy per worker): 2 processes would carry 2x the uplink.
    uplink = SharedBudget(_RATE_MBPS)
    children = []
    try:
        for _ in range(procs):
            args = [str(_RATE_MBPS), uplink.path, str(_CHUNK), str(_FLOW_S)]
            children.append(
                subprocess.Popen(
                    [sys.executable, "-c", _CHILD, *args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            )
        for c in children:
            assert c.stdout is not None and c.stdout.readline().strip() == "ready"
        t0 = time.monotonic() + 0.2
        for c in children:
            assert c.stdin is not None
            c.stdin.write(f"{t0!r}\n")
            c.stdin.flush()
        results = []
        for c in children:
            out, _ = c.communicate(timeout=60)
            assert c.returncode == 0
            sent, last = out.split()
            results.append((int(sent), float(last)))
    finally:
        for c in children:
            with c:  # closes the pipes and reaps the child
                if c.poll() is None:
                    c.kill()
        uplink.close()
    rate = sum(s for s, _ in results) * 8 / 1e6 / (max(t for _, t in results) - t0)
    assert all(s > 0 for s, _ in results)
    assert abs(rate - _RATE_MBPS) <= 0.05 * _RATE_MBPS, (procs, rate)


@pytest.mark.parametrize("flows", [1, 2, 4, 8])
def test_shared_pacer_ingress_paces_before_delivery(flows):
    # covers: TRANSPORT-I10
    # WOULD-FAIL-IF: DelayedDelivery ignores the ingress budget (each connection delivers at full speed).
    ingress = SharedBudget(_RATE_MBPS)
    delivered = []
    try:
        links = [DelayedDelivery(LinkSpec(ingress=ingress), delivered.append) for _ in range(flows)]
        rate = _run_flows([lambda n, d=d: d.push(b"", n) for d in links])
        with pytest.raises(ValueError):
            links[0].push(b"x")  # an ingress budget never runs unpaced
    finally:
        ingress.close()
    assert delivered and abs(rate - _RATE_MBPS) <= 0.05 * _RATE_MBPS, (flows, rate)


def test_shared_pacer_delivers_the_exact_bytes():
    # covers: TRANSPORT-I10
    # WOULD-FAIL-IF: shared-mode pacing or delivery drops, reorders or alters a payload.
    rng = random.Random(7)
    payloads = [rng.randbytes(rng.randrange(1, 64 * 1024)) for _ in range(40)]
    egress, uplink, ingress = (SharedBudget(400.0) for _ in range(3))
    a, b = socket.socketpair()
    got = []
    done = threading.Event()
    try:
        out_link = LinkSpec(egress=egress, uplink=uplink)
        in_link = LinkSpec(latency_ms=5, ingress=ingress)
        dd = DelayedDelivery(
            in_link, lambda p: (got.append(p), len(got) == len(payloads) and done.set())
        )

        def send():
            pacer = SenderPacer(out_link)
            for p in payloads:
                pacer.pace(len(p))
                a.sendall(struct.pack("<I", len(p)) + p)

        sender = threading.Thread(target=send)
        sender.start()
        for _ in payloads:
            n = struct.unpack("<I", b.recv(4, socket.MSG_WAITALL))[0]
            buf = bytearray()
            while len(buf) < n:
                buf += b.recv(n - len(buf))
            dd.push(bytes(buf), n)
        sender.join()
        assert done.wait(5)
    finally:
        a.close()
        b.close()
        for x in (egress, uplink, ingress):
            x.close()
    assert [hashlib.sha256(p).hexdigest() for p in got] == [
        hashlib.sha256(p).hexdigest() for p in payloads
    ]


def test_shared_pacer_budget_file_pickle_and_boundary_checks(tmp_path):
    # covers: TRANSPORT-I10
    # WOULD-FAIL-IF: a pickled budget opens a fresh bucket, a rate mismatch or a foreign file is accepted,
    # a stale clock is waited on instead of refused, a negative reservation rewinds the clock,
    # a closed budget silently reopens, a bool passes as a rate, or a symlink at the path is
    # followed.
    owned = SharedBudget(_RATE_MBPS)
    copy = pickle.loads(pickle.dumps(owned))
    try:
        end = owned.reserve(_CHUNK)
        assert copy.reserve(_CHUNK) == pytest.approx(
            end + _CHUNK * 8 / (_RATE_MBPS * 1e6), abs=1e-9
        )
        with pytest.raises(ValueError):
            SharedBudget(2 * _RATE_MBPS, owned.path)
    finally:
        copy.close()
        owned.close()
    assert not os.path.exists(owned.path)  # the creator removes its temporary file
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"\x00" * 3)
    with pytest.raises(ValueError):
        SharedBudget(_RATE_MBPS, foreign)
    for bad in (0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            SharedBudget(bad)
    with pytest.raises(ValueError):
        SharedBudget(_RATE_MBPS, burst_ms=True)
    # a symlink at the budget's path is never followed (the mapped file would be its target)
    target = tmp_path / "target"
    link = tmp_path / "link"
    os.symlink(target, link)
    with pytest.raises(OSError):
        SharedBudget(_RATE_MBPS, link)
    assert not target.exists()
    stale = tmp_path / "stale"
    stale.write_bytes(struct.pack("<dd", time.monotonic() + 7200.0, _RATE_MBPS))
    stale_budget = SharedBudget(_RATE_MBPS, stale)
    try:
        with pytest.raises(RuntimeError):
            stale_budget.reserve(1)
    finally:
        stale_budget.close()
    fresh = SharedBudget(_RATE_MBPS)
    try:
        with pytest.raises(ValueError):
            fresh.reserve(-1)  # a negative reservation would rewind the shared clock
    finally:
        fresh.close()
    with pytest.raises(RuntimeError):
        fresh.reserve(1)  # a closed budget never reopens a fresh bucket


def test_shared_pacer_defaults_and_jitter_rng_are_unchanged():
    # covers: TRANSPORT-I10
    # WOULD-FAIL-IF: shared mode becomes the default, or the jitter RNG stops being the seeded random.Random
    # (or the shim starts drawing from torch's RNG).
    spec = LinkSpec()
    assert (spec.egress, spec.uplink, spec.ingress) == (None, None, None)
    assert "torch" not in vars(shim)
    state = torch.random.get_rng_state()
    d1 = DelayedDelivery(LinkSpec(jitter_ms=3, jitter_seed=11), lambda _: None)
    d2 = DelayedDelivery(LinkSpec(jitter_ms=3, jitter_seed=11), lambda _: None)
    assert type(d1._rng) is random.Random
    assert [d1._rng.random() for _ in range(5)] == [d2._rng.random() for _ in range(5)]
    assert torch.equal(torch.random.get_rng_state(), state)


# --- AEAD record layer (TRANSPORT-I7) ----------------------------------------


_K_AB, _K_BA = bytes(range(32)), bytes(range(100, 132))
_P_AB, _P_BA = b"\x01\x02\x03\x04", b"\x0a\x0b\x0c\x0d"
_CIPHERS = [secure.AES_256_GCM, secure.CHACHA20_POLY1305]


# Ends the helpers below hand out; a test that drops one mid-way (a refused
# stream is abandoned, not closed) still has it closed at teardown.
_OPEN_ENDS: list = []


def _track(*ends):
    _OPEN_ENDS.extend(ends)
    return ends


@pytest.fixture(autouse=True)
def _close_ends():
    yield
    while _OPEN_ENDS:
        _OPEN_ENDS.pop().close()


def _secure_pair(cipher, **kw):
    a, b = socket.socketpair()
    sa = secure.SecureSocket(a, cipher, _K_AB, _K_BA, _P_AB, _P_BA, **kw)
    sb = secure.SecureSocket(b, cipher, _K_BA, _K_AB, _P_BA, _P_AB, **kw)
    return _track(sa, sb)


def _raw_records(cipher, payloads, **kw):
    """Seal `payloads` (one sendall each) and return the raw wire records."""
    a, b = socket.socketpair()
    sa = secure.SecureSocket(a, cipher, _K_AB, _K_BA, _P_AB, _P_BA, **kw)
    out = []
    for p in payloads:
        sa.sendall(p)
        (n,) = struct.unpack(">I", fr._recv_exact(b, 4))
        out.append(struct.pack(">I", n) + fr._recv_exact(b, n))
    sa.close()
    b.close()
    return out


def _reader(cipher, **kw):
    """A SecureSocket that reads what the test writes raw into `w`."""
    w, r = socket.socketpair()
    return _track(w, secure.SecureSocket(r, cipher, _K_BA, _K_AB, _P_BA, _P_AB, **kw))


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_secure_socket_round_trips_frames_exactly(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: a record boundary drops, duplicates or reorders bytes (payloads cross the
    # 1 MiB record limit), END is lost, or half-close stops the other direction.
    sa, sb = _secure_pair(cipher)
    g = torch.Generator().manual_seed(11)
    fwd = torch.randn(700, 1024, generator=g)  # 2.8 MB: three records
    bwd = torch.randn(5, 7, generator=g)
    grads = {0: [torch.randn(3, generator=g)], 9: [torch.full((2,), -0.5)]}
    got = []

    def read():
        while True:
            f = fr.recv_frame(sb)
            got.append(f)
            if f.type == fr.END:
                return

    t = threading.Thread(target=read)
    t.start()
    fr.send_frame(sa, fr.FWD, 4, 1, fwd, chunk=2)
    fr.send_frame(sa, fr.BWD, 4, 1, bwd)
    fr.send_frame(sa, fr.GRADS, 4, 0, grads)
    fr.send_end(sa)
    t.join(10)
    assert not t.is_alive()
    assert [f.type for f in got] == [fr.FWD, fr.BWD, fr.GRADS, fr.END]
    assert (got[0].step, got[0].mb, got[0].chunk) == (4, 1, 2)
    assert got[0].payload == fwd.numpy().tobytes()
    assert torch.equal(_bits(got[0].tensor()), _bits(fwd))
    assert got[1].payload == bwd.numpy().tobytes()
    back = got[2].obj()
    assert set(back) == {0, 9} and torch.equal(back[0][0], grads[0][0])
    # half-close: a stops writing; b sees EOF and can still answer.
    sa.shutdown(socket.SHUT_WR)
    assert sb.recv(1) == b""
    reply = bytes(range(256)) * 5000  # 1.28 MB, two records

    def answer():
        sb.sendall(reply)
        sb.shutdown(socket.SHUT_WR)

    t = threading.Thread(target=answer)
    t.start()
    buf = bytearray(len(reply) + 10)
    n = 0
    while True:
        k = sa.recv_into(memoryview(buf)[n:])
        if k == 0:
            break
        n += k
    t.join(10)
    assert bytes(buf[:n]) == reply
    assert sa.stats()["records_in"] == 2
    sa.close()
    sb.close()


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_tampered_record_is_refused(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the tag is not checked, the length header is not authenticated, a short
    # read at EOF is accepted as a record, or a hostile length is read before it is bounded.
    rec = _raw_records(cipher, [b"gradient bytes" * 100])[0]
    for pos in (3, 4, len(rec) // 2, len(rec) - 1):  # length, body, middle, tag
        bad = bytearray(rec)
        bad[pos] ^= 0x01
        w, r = _reader(cipher)
        w.sendall(bytes(bad))
        w.shutdown(socket.SHUT_WR)
        with pytest.raises(ConnectionError):
            r.recv(1 << 20)
        # the stream stays failed
        with pytest.raises(ConnectionError):
            r.recv(1)
    # truncated record
    w, r = _reader(cipher)
    w.sendall(rec[:-5])
    w.shutdown(socket.SHUT_WR)
    with pytest.raises(ConnectionError, match="truncated"):
        r.recv(10)
    # a length above the record limit is refused from the header alone
    for n in (secure.MAX_RECORD + secure.TAG + 1, 0xFFFFFFFF, secure.TAG):
        w, r = _reader(cipher)
        w.sendall(struct.pack(">I", n))
        with pytest.raises(ConnectionError, match="length"):
            r.recv(10)


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_replayed_or_reordered_record_is_refused(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the nonce does not follow the receive counter (a replayed or swapped record
    # opens), or the stream resynchronizes after a failure and accepts the next valid record.
    r0, r1, r2 = _raw_records(cipher, [b"zero" * 10, b"one" * 10, b"two" * 10])
    # in order: accepted
    w, r = _reader(cipher)
    w.sendall(r0 + r1 + r2)
    assert fr._recv_exact(r, 100) == b"zero" * 10 + b"one" * 10 + b"two" * 10
    # replay
    w, r = _reader(cipher)
    w.sendall(r0 + r0)
    assert fr._recv_exact(r, 40) == b"zero" * 10
    with pytest.raises(ConnectionError):
        r.recv(40)
    # reorder, then the right record: never resynchronizes
    w, r = _reader(cipher)
    w.sendall(r1 + r0)
    with pytest.raises(ConnectionError):
        r.recv(40)
    with pytest.raises(ConnectionError):
        r.recv(40)
    # a dropped record
    w, r = _reader(cipher)
    w.sendall(r0 + r2)
    assert r.recv(40) == b"zero" * 10
    with pytest.raises(ConnectionError):
        r.recv(40)


def _rekey(key, epoch):
    info = b"wantrain rekey" + epoch.to_bytes(8, "big")
    return HKDFExpand(algorithm=hashes.SHA256(), length=32, info=info).derive(key)


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_nonces_never_repeat_across_rekeys(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the counter resets or repeats, the rekey is skipped or not deterministic
    # (HKDF-Expand of the current key with 'wantrain rekey' || epoch), the byte threshold is
    # ignored, or a send past the counter limit is sealed.
    aead = {secure.AES_256_GCM: AESGCM, secure.CHACHA20_POLY1305: ChaCha20Poly1305}[cipher]
    # 3 records per key, or 250 bytes: sizes force one rekey by bytes (epoch 1 -> 2).
    sizes = [10, 10, 10, 200, 60, 5, 5, 5, 5]
    payloads = [bytes([i]) * s for i, s in enumerate(sizes)]
    recs = _raw_records(cipher, payloads, rekey_records=3, rekey_bytes=250)
    keys = [_K_AB]
    for e in (1, 2, 3):
        keys.append(_rekey(keys[-1], e))
    # epoch 0: records 0-2 (3 records); epoch 1: 3-4 (210 bytes, then 270 >= 250);
    # epoch 2: 5-7 (3 records); epoch 3: 8.
    epochs = [0, 0, 0, 1, 1, 2, 2, 2, 3]
    seen = set()
    for i, (rec, p, e) in enumerate(zip(recs, payloads, epochs, strict=True)):
        nonce = _P_AB + i.to_bytes(8, "big")
        assert aead(keys[e]).decrypt(nonce, rec[4:], rec[:4]) == p
        for other in set(range(4)) - {e}:
            with pytest.raises(Exception):  # noqa: B017
                aead(keys[other]).decrypt(nonce, rec[4:], rec[:4])
        seen.add(nonce)
    assert len(seen) == len(recs) and len(set(keys)) == 4
    # the receiving end follows the same schedule
    w, r = _reader(cipher, rekey_records=3, rekey_bytes=250)
    w.sendall(b"".join(recs))
    assert fr._recv_exact(r, sum(sizes)) == b"".join(payloads)
    # the counter's limit: refuse to send, and refuse to open, past it
    sa, sb = _secure_pair(cipher, counter_limit=2)
    sa.sendall(b"a")
    sa.sendall(b"b")
    with pytest.raises(ConnectionError, match="exhausted"):
        sa.sendall(b"c")
    with pytest.raises(ConnectionError):
        sa.sendall(b"d")
    assert fr._recv_exact(sb, 2) == b"ab"
    w, r = _reader(cipher, counter_limit=2)
    w.sendall(b"".join(_raw_records(cipher, [b"x", b"y", b"z"])))
    assert fr._recv_exact(r, 2) == b"xy"
    with pytest.raises(ConnectionError, match="exhausted"):
        r.recv(1)


def test_no_plaintext_mode():
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: a null, empty or unknown cipher, a short key, a shared key for both
    # directions, or a bad prefix is accepted, or payload bytes appear on the wire.
    a, b = socket.socketpair()
    for bad in (None, "", "none", "null", "plaintext", "aes-128-gcm"):
        with pytest.raises(ValueError):
            secure.SecureSocket(a, bad, _K_AB, _K_BA, _P_AB, _P_BA)
    for kw in (
        dict(send_key=b"", recv_key=_K_BA),
        dict(send_key=_K_AB[:16], recv_key=_K_BA),
        dict(send_key=_K_AB, recv_key=_K_AB),
    ):
        with pytest.raises(ValueError):
            secure.SecureSocket(a, secure.AES_256_GCM, send_prefix=_P_AB, recv_prefix=_P_BA, **kw)
    with pytest.raises(ValueError):
        secure.SecureSocket(a, secure.AES_256_GCM, _K_AB, _K_BA, b"\x00" * 3, _P_BA)
    assert not hasattr(secure.SecureSocket, "send")  # no unsealed write path
    for cipher in _CIPHERS:
        secret = b"PLAINTEXT-ACTIVATION-" * 50
        wire = b"".join(_raw_records(cipher, [secret]))
        assert b"PLAINTEXT" not in wire and len(wire) == len(secret) + 4 + secure.TAG
    a.close()
    b.close()


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_stats_count_every_record(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: a record (the last partial one of a sendall, or one opened during a
    # partial recv) is not counted, or seal/open time is not measured.
    sa, sb = _secure_pair(cipher)
    zero = {k: 0 for k in ("records_out", "records_in", "bytes_out", "bytes_in")}
    assert {k: sa.stats()[k] for k in zero} == zero
    sizes = [1, 1000, secure.MAX_RECORD, secure.MAX_RECORD + 7]  # 1 + 1 + 1 + 2 records
    total = sum(sizes)
    t = threading.Thread(target=lambda: [sa.sendall(b"\x5a" * s) for s in sizes])
    t.start()
    assert fr._recv_exact(sb, total) == b"\x5a" * total
    t.join(10)
    out, inn = sa.stats(), sb.stats()
    assert (out["records_out"], out["bytes_out"]) == (5, total)
    assert (inn["records_in"], inn["bytes_in"]) == (5, total)
    assert out["records_in"] == out["bytes_in"] == inn["records_out"] == inn["bytes_out"] == 0
    assert out["seal_ns"] > 0 and inn["open_ns"] > 0
    assert out["open_ns"] == 0 and inn["seal_ns"] == 0
    assert set(out) == {"records_out", "records_in", "bytes_out", "bytes_in", "seal_ns", "open_ns"}


class _Wire:
    """Test boundary fake for the kernel socket: sendmsg takes at most `send_cap` bytes per
    call (a full send buffer); recv_into hands out the written bytes in pieces whose sizes
    cycle through `recv_caps` (a peer whose segments arrive split)."""

    def __init__(self, send_cap, recv_caps):
        self.data, self.pos, self.calls = bytearray(), 0, []
        self.send_cap, self.recv_caps, self.reads = send_cap, recv_caps, 0

    def sendmsg(self, parts):
        self.calls.append([len(p) for p in parts])
        # A sender that stops advancing would loop forever and grow `data` without bound:
        # fail fast instead (no test here writes more than 8 MiB).
        assert len(self.data) < 8 << 20, "sender wrote past 8 MiB: not advancing after a short send"
        chunk = b"".join(bytes(p[: self.send_cap]) for p in parts)[: self.send_cap]
        self.data += chunk
        return len(chunk)

    def recv_into(self, view):
        cap = self.recv_caps[self.reads % len(self.recv_caps)]
        self.reads += 1
        n = min(len(view), cap, len(self.data) - self.pos)
        view[:n] = self.data[self.pos : self.pos + n]
        self.pos += n
        return n


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_records_survive_short_sends_and_fragmented_reads(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the gather send drops or repeats bytes after a short sendmsg, the header
    # and ciphertext are concatenated into one buffer again, or the in-place receive buffer
    # loses, repeats or misaligns bytes when it compacts an unread tail or a record arrives
    # in many small pieces.
    wire = _ClosableWire(4093, [1 << 30, 1 << 30, 5, 1 << 30, 70001, 3])
    sa, sb = _track(
        secure.SecureSocket(wire, cipher, _K_AB, _K_BA, _P_AB, _P_BA),
        secure.SecureSocket(wire, cipher, _K_BA, _K_AB, _P_BA, _P_AB),
    )
    g = torch.Generator().manual_seed(5)

    def rand(n):
        return bytes(torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8).tolist())

    # ~100 KB of small records moves the read position past the 64 KiB read-ahead, so the
    # large record behind them is read only after the unread tail is compacted.
    payloads = [rand(97) for _ in range(900)]
    payloads += [rand(s) for s in (secure.MAX_RECORD, 3, secure.MAX_RECORD + 11, 50)]
    for p in payloads:
        sa.sendall(p)
    assert all(c[0] == 4 and len(c) == 2 for c in wire.calls if len(c) == 2)
    assert wire.calls[0] == [4, 97 + secure.TAG]  # gathered, never concatenated
    want = b"".join(payloads)
    got = bytearray()
    while len(got) < len(want):
        piece = sb.recv(7919)
        assert piece
        got += piece
    assert bytes(got) == want
    assert sb.recv(1) == b""  # clean EOF on a record boundary
    assert sb.stats()["records_in"] == sa.stats()["records_out"] == len(payloads) + 1


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_partial_kernel_writes_reach_the_peer_intact(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: after a partial sendmsg on a real socket (timeout set, so CPython sends
    # non-blocking; small send buffer; slow reader) the sender re-sends or skips bytes of the
    # record instead of continuing where the kernel stopped.
    a, b = socket.socketpair()
    try:
        a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 14)
        b.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 14)
        sa, sb = _track(
            secure.SecureSocket(a, cipher, _K_AB, _K_BA, _P_AB, _P_BA),
            secure.SecureSocket(b, cipher, _K_BA, _K_AB, _P_BA, _P_AB),
        )
        sa.settimeout(10)
        sb.settimeout(10)
        g = torch.Generator().manual_seed(11)
        want = bytes(torch.randint(0, 256, (3 << 20,), generator=g, dtype=torch.uint8).tolist())
        err: list[BaseException] = []

        def send():
            try:
                sa.sendall(want)
                sa.shutdown(socket.SHUT_WR)
            except BaseException as e:  # surfaced below
                err.append(e)

        t = threading.Thread(target=send)
        t.start()
        got = bytearray()
        while True:
            piece = sb.recv(4096)  # small reads keep the sender's buffer full
            if not piece:
                break
            got += piece
        t.join(10)
        assert not t.is_alive() and not err, err
        assert bytes(got) == want
        assert sb.stats()["records_in"] == sa.stats()["records_out"] == 3
    finally:
        a.close()
        b.close()


_M = secure.MAX_RECORD


class _ClosableWire(_Wire):
    """_Wire that a SecureSocket can close (its close() shuts the pool down first)."""

    def close(self):
        pass


def _wire_bytes(cipher, payloads, threads, **kw):
    """Everything one SecureSocket with `threads` puts on the wire for `payloads`
    (one sendall each), read raw on the other end."""
    a, b = socket.socketpair()
    sa = secure.SecureSocket(a, cipher, _K_AB, _K_BA, _P_AB, _P_BA, threads=threads, **kw)
    got = bytearray()

    def drain():
        while chunk := b.recv(1 << 20):
            got.extend(chunk)

    t = threading.Thread(target=drain)
    t.start()
    for p in payloads:
        sa.sendall(p)
    used_pool = sa._pool is not None
    sa.shutdown(socket.SHUT_WR)
    t.join(30)
    assert not t.is_alive()
    sa.close()
    b.close()
    return bytes(got), used_pool


# Messages of one record, of several records past the 4-record window, an exact multiple of
# the record size, and short ones in between; with 3 records or 2.5 MiB per key the rekeys
# fall inside the multi-record messages (records 3, 6, 9, ... and by bytes).
_PAR_SIZES = [5, _M + 1, 3 * _M + 5, 7, 9 * _M, 2 * _M, _M, 1000, 4 * _M + 3]
_PAR_REKEY = dict(rekey_records=3, rekey_bytes=(5 << 20) // 2)


def _par_payloads():
    g = torch.Generator().manual_seed(23)
    return [
        torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8).numpy().tobytes()
        for n in _PAR_SIZES
    ]


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_parallel_sealing_puts_the_serial_bytes_on_the_wire(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the pooled send reserves nonces out of record order, writes sealed records
    # in completion order instead of counter order, rekeys at another record or byte point
    # (a stale key captured across a rekey), splits records differently, or changes framing;
    # or the pooled receive opens or delivers them in another order.
    payloads = _par_payloads()
    serial, pooled_ser = _wire_bytes(cipher, payloads, 1, **_PAR_REKEY)
    assert not pooled_ser
    for threads in (2, 4):
        wire, pooled = _wire_bytes(cipher, payloads, threads, **_PAR_REKEY)
        assert pooled  # the multi-record messages went through the pool
        assert hashlib.sha256(wire).hexdigest() == hashlib.sha256(serial).hexdigest()
    # the expected record count: one per started MiB of each message
    assert len(serial) == sum(n + 4 * -(-n // _M) + secure.TAG * -(-n // _M) for n in _PAR_SIZES)
    # pooled and serial receivers both read the serial bytes back exactly
    for threads in (1, 4):
        wire = _ClosableWire(1 << 30, [1 << 30])
        wire.data += serial
        r = secure.SecureSocket(
            wire, cipher, _K_BA, _K_AB, _P_BA, _P_AB, threads=threads, **_PAR_REKEY
        )
        got = bytearray()
        while piece := r.recv(3 << 20):
            got += piece
        assert bytes(got) == b"".join(payloads)
        assert (r._pool is not None) == (threads > 1)
        assert r.stats()["records_in"] == sum(-(-n // _M) for n in _PAR_SIZES)
        r.close()


def _full_records(cipher, k):
    """k full-size records of one message, sealed serially, and their plaintexts."""
    g = torch.Generator().manual_seed(29)
    msg = torch.randint(0, 256, (k * _M,), generator=g, dtype=torch.uint8).numpy().tobytes()
    raw, _ = _wire_bytes(cipher, [msg], 1)
    step = 4 + _M + secure.TAG
    recs = [raw[i : i + step] for i in range(0, len(raw), step)]
    assert len(recs) == k
    return recs, [msg[i * _M : (i + 1) * _M] for i in range(k)]


def _pooled_reader(cipher, raw, **kw):
    wire = _ClosableWire(1 << 30, [1 << 30])  # every byte at once: records pipeline
    wire.data += raw
    (r,) = _track(secure.SecureSocket(wire, cipher, _K_BA, _K_AB, _P_BA, _P_AB, threads=4, **kw))
    return r


def _read_until_refused(r, want_before, pooled=True):
    """Read until the stream refuses; the bytes delivered first must be `want_before`."""
    got = bytearray()
    with pytest.raises(ConnectionError) as e:
        while True:
            piece = r.recv(_M)
            assert piece, "EOF where a refusal was due"
            got += piece
    assert bytes(got) == want_before
    with pytest.raises(ConnectionError):  # and it stays failed
        r.recv(1)
    # past the first full-size record the refusal comes from the pooled open; at the first,
    # from the serial one, which grows nothing (no buffer, pool or wake pair for a forgery)
    assert (r._pool is not None) == pooled
    if not pooled:
        assert r._wake_r is None and len(r._buf) < 2 * _M
    r.close()
    return str(e.value)


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_parallel_open_refuses_tamper_replay_reorder_and_truncation(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: a batched open delivers a record whose tag fails, delivers records after
    # the refused one, resynchronizes, assigns receive nonces out of order (a replayed or
    # swapped record opens), withholds the good records before the bad one, or accepts a
    # truncated tail or a counter past its limit.
    recs, plain = _full_records(cipher, 5)
    # in order: all five, through the pool
    r = _pooled_reader(cipher, b"".join(recs))
    got = bytearray()
    while piece := r.recv(3 << 20):
        got += piece
    assert bytes(got) == b"".join(plain) and r._pool is not None
    for bad_at in (0, 2, 4):  # first of a batch, inside one, the last record
        for pos in (1, 4, 4 + _M // 2, len(recs[0]) - 1):  # length, body, middle, tag
            bad = bytearray(recs[bad_at])
            bad[pos] ^= 0x01
            raw = b"".join(recs[:bad_at]) + bytes(bad) + b"".join(recs[bad_at + 1 :])
            if bad_at == 0 and pos == 1:
                # a length past the limit is refused from the header alone, before any open
                r = _pooled_reader(cipher, raw)
                with pytest.raises(ConnectionError, match="length"):
                    r.recv(1)
                continue
            why = _read_until_refused(
                _pooled_reader(cipher, raw), b"".join(plain[:bad_at]), pooled=bad_at > 0
            )
            assert "tag" in why or "length" in why or "truncated" in why
    # replay, reorder, drop: the counter decides, never arrival
    cases = {
        "replay": (recs[:3] + [recs[2]] + recs[3:], 3),
        "reorder": ([recs[0], recs[2], recs[1]] + recs[3:], 1),
        "drop": (recs[:2] + recs[3:], 2),
    }
    for _name, (seq, ok) in cases.items():
        _read_until_refused(_pooled_reader(cipher, b"".join(seq)), b"".join(plain[:ok]))
    # truncated tail: the complete records arrive, then the refusal
    why = _read_until_refused(_pooled_reader(cipher, b"".join(recs)[:-7]), b"".join(plain[:4]))
    assert "truncated" in why
    # over a real socket (the reader waits on the socket and the pool together): a tampered
    # fourth record still arrives after the first three, and nothing after it
    a, b = socket.socketpair()
    bad = bytearray(recs[3])
    bad[len(bad) // 2] ^= 0x80

    def write():
        try:
            a.sendall(b"".join(recs[:3]) + bytes(bad) + recs[4])
        except OSError:
            pass  # the reader refused and closed: what is left cannot be delivered
        a.close()

    t = threading.Thread(target=write)
    t.start()
    r = secure.SecureSocket(b, cipher, _K_BA, _K_AB, _P_BA, _P_AB, threads=4)
    why = _read_until_refused(r, b"".join(plain[:3]))
    assert "tag" in why
    t.join(30)
    # the counter's limit inside the pipeline: two records open, the third is refused
    why = _read_until_refused(
        _pooled_reader(cipher, b"".join(recs), counter_limit=2), b"".join(plain[:2])
    )
    assert "exhausted" in why


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_parallel_receive_survives_fragmented_reads_and_compaction(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the grown receive buffer loses, repeats or misaligns bytes when it grows
    # under an unread tail, compacts a tail behind greedy reads, or batches a record that has
    # not fully arrived.
    wire = _ClosableWire(1 << 30, [1 << 30, 5, 70001, 3, 1 << 20, 4099])
    sa = secure.SecureSocket(wire, cipher, _K_AB, _K_BA, _P_AB, _P_BA, threads=4)
    sb = secure.SecureSocket(wire, cipher, _K_BA, _K_AB, _P_BA, _P_AB, threads=4)
    g = torch.Generator().manual_seed(31)

    def rand(n):
        return torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8).numpy().tobytes()

    # ~7.1 MiB in all (the fake wire holds 8): more than the grown 4-record buffer, so it
    # compacts at least once with records read ahead behind the one being opened.
    payloads = [rand(97) for _ in range(300)]
    payloads += [rand(s) for s in (2 * _M + 9, 3, _M, 11, 3 * _M - 1, 50)]
    payloads += [rand(97) for _ in range(300)] + [rand(_M + 5)]
    for p in payloads:
        sa.sendall(p)
    want = b"".join(payloads)
    got = bytearray()
    while len(got) < len(want):
        piece = sb.recv(7919)
        assert piece
        got += piece
    assert bytes(got) == want
    assert sb.recv(1) == b""
    assert sb.stats()["records_in"] == sa.stats()["records_out"]
    assert sb._pool is not None and len(sb._buf) > 4 * _M  # grown, and pooled
    sa.close()
    sb.close()


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_parallel_receive_works_on_a_descriptor_past_fd_setsize(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the pooled reader waits on the socket with select(), which refuses any
    # descriptor >= FD_SETSIZE (1024): every large message of a process with many open files
    # would fail its stream as "closed".
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    high = 1500
    if soft <= high:
        resource.setrlimit(resource.RLIMIT_NOFILE, (high + 64, hard))
    a, b = socket.socketpair()
    try:
        fd = os.dup2(b.fileno(), high)
        b.close()
        b = socket.socket(fileno=fd)
        sa = secure.SecureSocket(a, cipher, _K_AB, _K_BA, _P_AB, _P_BA, threads=4)
        sb = secure.SecureSocket(b, cipher, _K_BA, _K_AB, _P_BA, _P_AB, threads=4)
        g = torch.Generator().manual_seed(37)
        msgs = [
            torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8).numpy().tobytes()
            for n in (16 * _M + 3, 9 * _M)
        ]

        def write():
            for m in msgs:
                sa.sendall(m)
            sa.shutdown(socket.SHUT_WR)

        t = threading.Thread(target=write)
        t.start()
        got = bytearray()
        while piece := sb.recv(3 << 20):
            got += piece
        t.join(30)
        assert not t.is_alive()
        assert sb.fileno() == high and sb._pool is not None
        assert hashlib.sha256(got).hexdigest() == hashlib.sha256(b"".join(msgs)).hexdigest()
        sa.close()
        sb.close()
    finally:
        a.close()
        b.close()
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def _reference_wire(cipher, payloads, rekey_records, rekey_bytes):
    """The records TRANSPORT-I7 prescribes, built with the one-shot AEAD calls (independent of
    the record layer's reused buffers): MAX_RECORD chunks, nonce = prefix || u64be(counter),
    the length header as AAD, rekey before the record that would pass either limit."""
    aead = {secure.AES_256_GCM: AESGCM, secure.CHACHA20_POLY1305: ChaCha20Poly1305}[cipher]
    key, epoch, counter, recs, nbytes = _K_AB, 0, 0, 0, 0
    out = bytearray()
    for p in payloads:
        for off in range(0, len(p), _M):
            chunk = p[off : off + _M]
            if recs >= rekey_records or nbytes >= rekey_bytes:
                epoch += 1
                key, recs, nbytes = _rekey(key, epoch), 0, 0
            hdr = struct.pack(">I", len(chunk) + secure.TAG)
            out += hdr + aead(key).encrypt(_P_AB + counter.to_bytes(8, "big"), chunk, hdr)
            counter, recs, nbytes = counter + 1, recs + 1, nbytes + len(chunk)
    return bytes(out)


# Large, then small, then large again: every reused slot is first filled by a bigger record
# than the one that takes it next. Rekeys fall inside the multi-record messages.
_REUSE_SIZES = [3 * _M + 5, 7, _M, 2 * _M + 1, 100, 5 * _M + 3, 1]


def _reuse_payloads():
    g = torch.Generator().manual_seed(41)
    return [
        torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8).numpy().tobytes()
        for n in _REUSE_SIZES
    ]


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_sealing_into_reused_buffers_puts_the_reference_records_on_the_wire(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: a reused send slot puts bytes of an earlier, larger record on the wire
    # (the slot is not cut to len + TAG), a pooled slot is sealed into again before its record
    # was written, or a record is sealed under another nonce, key or AAD than the one-shot
    # reference (a rekey point that moves when sealing goes into buffers).
    payloads = _reuse_payloads()
    want = _reference_wire(cipher, payloads, **_PAR_REKEY)
    for threads in (1, 2, 4):
        wire, pooled = _wire_bytes(cipher, payloads, threads, **_PAR_REKEY)
        assert pooled == (threads > 1)
        assert hashlib.sha256(wire).hexdigest() == hashlib.sha256(want).hexdigest()


def _rx_memory(r):
    """Every buffer a SecureSocket holds, as bytes."""
    found = []
    for v in vars(r).values():
        for b in v if isinstance(v, list) else [v]:
            if isinstance(b, (bytearray, memoryview)):
                found.append(bytes(b))
    return found


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_reused_receive_buffers_never_change_delivered_bytes(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: recv or recv_into hands the caller a view of a reused plaintext slot
    # (a later record changes bytes already delivered), a pooled record opens into a slot
    # another in-flight or undelivered record still uses, or a short record delivers the
    # stale tail of a larger one opened before it in the same slot.
    payloads = _reuse_payloads()
    want = b"".join(payloads)
    raw = _reference_wire(cipher, payloads, **_PAR_REKEY)
    for threads in (1, 4):
        wire = _ClosableWire(1 << 30, [1 << 30, 70001, 5])
        wire.data += raw
        r = secure.SecureSocket(
            wire, cipher, _K_BA, _K_AB, _P_BA, _P_AB, threads=threads, **_PAR_REKEY
        )
        pieces, into, n = [], bytearray(len(want)), 0
        while n < len(want):
            if len(pieces) % 2:
                k = r.recv_into(memoryview(into)[n:], 300007)
                assert k
                pieces.append((n, memoryview(into)[n : n + k]))
            else:
                p = r.recv(7919 if len(pieces) % 4 else 1 << 20)
                assert p
                pieces.append((n, p))
                k = len(p)
            n += k
        assert r.recv(1) == b""
        assert (r._pool is not None) == (threads > 1)
        # every piece, read long ago and then held while later records reused the slots
        for off, p in pieces:
            assert bytes(p) == want[off : off + len(p)], (threads, off)
        r.close()


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_a_refused_record_leaves_no_plaintext_behind(cipher):
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the plaintext decrypt_into writes before it checks the tag (a refused
    # record's bytes, or those of the records opened behind it on the pool and never
    # delivered) stays in the socket's buffers after the refusal, or is delivered, or the
    # records delivered before the refusal change.
    secret = b"UNAUTHENTICATED-SECRET-"
    # serial: one good record, then a smaller one whose tag fails, so it opens into the slot
    # the first one grew (its ciphertext intact: the unauthenticated plaintext decrypt_into
    # writes is the whole secret)
    good, bad = _raw_records(cipher, [b"first" * 400, secret * 40])
    bad = bad[:-1] + bytes([bad[-1] ^ 1])
    w, r = _reader(cipher)
    w.sendall(good + bad)
    first = r.recv(1 << 20)
    with pytest.raises(ConnectionError, match="tag"):
        r.recv(1 << 20)
    assert first == b"first" * 400
    assert not any(secret in b for b in _rx_memory(r))
    w.close()
    r.close()
    # a forged full-size record as the first one: refused, and no plaintext buffer was grown
    # for it (growth only after a record of that size authenticates)
    r = _pooled_reader(cipher, struct.pack(">I", _M + secure.TAG) + os.urandom(_M + secure.TAG))
    with pytest.raises(ConnectionError, match="tag"):
        r.recv(1)
    assert not r._rx_one and not r._rx_ring and r._pool is None
    r.close()
    # pooled: records 0-2 are delivered, 3 fails its tag, 4 may have opened behind it
    g = torch.Generator().manual_seed(43)
    msg = bytearray(torch.randint(0, 256, (5 * _M,), generator=g, dtype=torch.uint8).numpy())
    for i in range(5):
        msg[i * _M : i * _M + len(secret) + 1] = secret + bytes([i])
    raw, _ = _wire_bytes(cipher, [bytes(msg)], 1)
    step = 4 + _M + secure.TAG
    recs = [raw[i : i + step] for i in range(0, len(raw), step)]
    recs[3] = recs[3][:-1] + bytes([recs[3][-1] ^ 1])
    r = _pooled_reader(cipher, b"".join(recs))
    _read_until_refused_open(r, bytes(msg[: 3 * _M]))
    for i in (3, 4):
        assert not any(secret + bytes([i]) in b for b in _rx_memory(r)), i
    r.close()


def _read_until_refused_open(r, want_before):
    got = bytearray()
    with pytest.raises(ConnectionError, match="tag"):
        while True:
            piece = r.recv(_M)
            assert piece, "EOF where a refusal was due"
            got += piece
    assert bytes(got) == want_before and r._pool is not None


def test_close_shuts_the_pool_down_and_refuses_further_work():
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: close() leaves pool threads running, or a send after close seals
    # anything (it must fail the direction instead of using a dead pool silently).
    sa, sb = _secure_pair(secure.AES_256_GCM, threads=4)
    got = bytearray()

    def drain():
        while piece := sb.recv(1 << 20):
            got.extend(piece)

    t = threading.Thread(target=drain)
    t.start()
    msg = os.urandom(3 * _M + 1)
    sa.sendall(msg)
    pool = sa._pool
    assert pool is not None
    workers = list(pool._threads)
    assert workers
    sa.shutdown(socket.SHUT_WR)
    t.join(30)
    assert bytes(got) == msg
    sa.close()
    assert sa._pool is None and all(not w.is_alive() for w in workers)
    with pytest.raises(ConnectionError, match="closed"):
        sa.sendall(msg)
    assert sa._pool is None  # refused before any pool was made again
    for bad in (0, 65):
        with pytest.raises(ValueError, match="threads"):
            secure.SecureSocket(sb, secure.AES_256_GCM, _K_AB, _K_BA, _P_AB, _P_BA, threads=bad)
    sb.close()


def test_hardware_aes_reads_cpu_features_and_is_false_when_unknown():
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: hardware_aes guesses True without evidence or on an unreadable source.
    def cpuinfo(text):
        return lambda _p: text

    def sysctl(val):
        return lambda _n: val

    def boom(*_a):
        raise OSError("no")

    d = secure._detect_aes
    assert d("Linux", "x86_64", cpuinfo("flags\t: fpu sse2 aes avx\n"))
    assert not d("Linux", "x86_64", cpuinfo("flags\t: fpu sse2 avx\n"))
    assert d("Linux", "aarch64", cpuinfo("Features\t: fp asimd aes pmull\n"))
    assert not d("Linux", "aarch64", cpuinfo("Features\t: fp asimd\n"))
    assert not d("Linux", "x86_64", boom)
    assert d("Darwin", "arm64", sysctl=sysctl("1"))
    assert not d("Darwin", "arm64", sysctl=sysctl("0"))
    assert not d("Darwin", "arm64", sysctl=boom)
    assert d("Darwin", "x86_64", sysctl=sysctl("FPU SSE AES AVX1.0"))
    assert not d("Windows", "AMD64")
    assert not d("Linux", "riscv64", cpuinfo("isa : rv64 aes"))
    assert isinstance(secure.hardware_aes(), bool)


# --- sealed control channel (TRANSPORT-I9) -----------------------------------


def _control_pair(tmp_path, coordinator_id=None, coord_key="coord.key", worker_key="worker.key"):
    """control_accept on one end of a socketpair, control_dial on the other.
    Returns {"c": (Channel, info) | exception, "w": ...}."""
    a, b = _track(*socket.socketpair())
    out = {}

    def run(name, fn):
        try:
            out[name] = fn()
        except Exception as e:  # recorded and asserted by the caller
            out[name] = e
            (a if name == "c" else b).close()

    t = threading.Thread(
        target=run, args=("c", lambda: control.control_accept(a, str(tmp_path / coord_key)))
    )
    t.start()
    run(
        "w",
        lambda: control.control_dial(b, str(tmp_path / worker_key), coordinator_id=coordinator_id),
    )
    t.join(10)
    assert not t.is_alive()
    return out


def _peer_id_of(path):
    from cryptography.hazmat.primitives import serialization

    pub = (
        identity.load_or_create_key(str(path))
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    return identity.peer_id(pub)


def test_token_and_session_macs_never_cross():
    # covers: TRANSPORT-I9
    # WOULD-FAIL-IF: token_mac and session_mac share a domain (a session MAC passes as a token
    # MAC or back), a domain string drifts from the spec, or a token-derived coordinator proof
    # (any other MAC function) comes back.
    t = bytes(range(32))
    tok = control.token_mac("s3cret", t)
    ses = control.session_mac("s3cret", t)
    # independent computation of the spec's formulas
    assert (
        tok == hmac.new(b"s3cret", b"wantrain-control-worker|v3|" + t, hashlib.sha256).hexdigest()
    )
    assert (
        ses == hmac.new(b"s3cret", b"wantrain-control-session|v3|" + t, hashlib.sha256).hexdigest()
    )
    assert tok != ses
    assert control.check_mac("s3cret", t, tok)
    assert control.check_session_mac("s3cret", t, ses)
    assert not control.check_mac("s3cret", t, ses)
    assert not control.check_session_mac("s3cret", t, tok)
    public_macs = {
        n
        for n, v in vars(control).items()
        if "mac" in n.lower()
        and not n.startswith("_")
        and callable(v)
        and not isinstance(v, type)
        and getattr(v, "__module__", None) == control.__name__
    }
    assert public_macs == {"token_mac", "session_mac", "check_mac", "check_session_mac"}


def test_mac_is_bound_to_the_transcript(tmp_path):
    # covers: TRANSPORT-I9
    # WOULD-FAIL-IF: the MACs ignore the transcript (or use a per-call nonce that is not the
    # handshake's), so a MAC from one control handshake is accepted on another.
    one, two = _control_pair(tmp_path), _control_pair(tmp_path)
    for out in (one, two):
        assert not isinstance(out["c"], Exception), out
        assert not isinstance(out["w"], Exception), out
        assert out["c"][1]["transcript"] == out["w"][1]["transcript"]
        assert len(out["c"][1]["transcript"]) == 32
    t1, t2 = one["w"][1]["transcript"], two["w"][1]["transcript"]
    assert t1 != t2
    mac1 = control.token_mac("s3cret", t1)
    assert control.check_mac("s3cret", one["c"][1]["transcript"], mac1)
    assert not control.check_mac("s3cret", t2, mac1)
    assert not control.check_mac("other", t1, mac1)
    smac1 = control.session_mac("ab" * 16, t1)
    assert control.check_session_mac("ab" * 16, t1, smac1)
    assert not control.check_session_mac("ab" * 16, t2, smac1)
    for out in (one, two):
        out["c"][0].close()
        out["w"][0].close()


def test_macs_validate_their_input():
    # covers: TRANSPORT-I9
    # WOULD-FAIL-IF: check_mac / check_session_mac accept a MAC that is not exactly 64 lowercase
    # hex (case-folded, padded, truncated, non-str), or a MAC is computed under an empty secret
    # or over an empty transcript.
    t = b"\x07" * 32
    good = control.token_mac("tok", t)
    sgood = control.session_mac("tok", t)
    assert len(good) == 64 and good == good.lower()
    for check, mac in ((control.check_mac, good), (control.check_session_mac, sgood)):
        assert check("tok", t, mac)
        for bad in (
            mac.upper(),
            mac[:-1],
            mac + "0",
            " " + mac[:-1],
            mac[:-1] + "\n",
            mac[:-2] + "zz",
            "",
            None,
            mac.encode(),
            int(mac, 16),
        ):
            assert not check("tok", t, bad), bad
        assert not check("", t, mac)
        assert not check(None, t, mac)
    for fn in (control.token_mac, control.session_mac):
        with pytest.raises(ValueError):
            fn("", t)
        with pytest.raises(ValueError):
            fn("tok", b"")


def test_control_dial_refuses_another_coordinator_key(tmp_path):
    # covers: TRANSPORT-I9
    # WOULD-FAIL-IF: control_dial does not pass coordinator_id as expect_peer, runs the data
    # plane's domain, or does not take the worker role, so a coordinator with another key (or an
    # admitted worker posing as it) completes the handshake.
    real = _peer_id_of(tmp_path / "real.key")
    out = _control_pair(tmp_path, coordinator_id=real, coord_key="impostor.key")
    assert isinstance(out["w"], ConnectionError), out
    assert not isinstance(out["w"], ValueError)
    # the honest arm: pinned to the key the coordinator holds
    out = _control_pair(tmp_path, coordinator_id=real, coord_key="real.key")
    assert not isinstance(out["w"], Exception), out
    ch_w, info_w = out["w"]
    ch_c, info_c = out["c"]
    assert info_w["peer_id"] == real
    assert info_c["peer_id"] == _peer_id_of(tmp_path / "worker.key")
    ch_w.close()
    ch_c.close()
    # the control domain never completes against a data-plane handshake
    a, b = socket.socketpair()
    res = {}

    def p2p():
        try:
            res["p"] = identity.authenticate(a, str(tmp_path / "real.key"), role="coordinator")
        except Exception as e:
            res["p"] = e
            a.close()

    t = threading.Thread(target=p2p)
    t.start()
    with pytest.raises(ConnectionError):
        control.control_dial(b, str(tmp_path / "worker.key"), coordinator_id=real)
    b.close()
    t.join(10)
    assert isinstance(res["p"], ConnectionError)


def test_sealed_channel_round_trips_messages_and_files(tmp_path):
    # covers: TRANSPORT-I9
    # WOULD-FAIL-IF: control_dial / control_accept return an unsealed Channel or seal it with
    # mismatched keys, send_file bypasses the record layer, a file larger than one record loses
    # bytes, or a message travels in clear on the wire.
    out = _control_pair(tmp_path)
    ch_w, _ = out["w"]
    ch_c, _ = out["c"]
    assert ch_w.sealed and ch_c.sealed
    assert isinstance(ch_w.sock, secure.SecureSocket)
    assert isinstance(ch_c.sock, secure.SecureSocket)
    data = os.urandom(2_500_000)  # three records
    src = tmp_path / "shard.bin"
    src.write_bytes(data)
    got = {}

    def read():
        got["m1"] = ch_c.recv()
        got["m2"] = ch_c.recv(str(tmp_path / "in"))

    t = threading.Thread(target=read)
    t.start()
    ch_w.send({"kind": "HELLO", "n": 3}, b"body bytes")
    digest = ch_w.send_file({"kind": "CKPT"}, str(src))
    t.join(10)
    assert not t.is_alive()
    assert got["m1"] == ({"kind": "HELLO", "n": 3}, b"body bytes")
    head, path = got["m2"]
    assert head["sha256"] == digest == hashlib.sha256(data).hexdigest()
    with open(path, "rb") as f:
        assert f.read() == data
    ch_c.send({"kind": "WELCOME", "session": "cd" * 16})
    assert ch_w.recv() == ({"kind": "WELCOME", "session": "cd" * 16}, b"")
    assert ch_w.sock.stats()["records_in"] == 1
    assert ch_c.sock.stats()["records_in"] >= 4
    ch_w.close()
    ch_c.close()
    # on the wire: nothing of the message is readable
    a, b = socket.socketpair()
    ch = control.Channel(a)
    ch.seal(secure.CHACHA20_POLY1305, _K_AB, _K_BA, _P_AB, _P_BA)
    ch.send({"kind": "WELCOME", "session": "cd" * 16})
    raw = b.recv(1 << 16)
    assert b"WELCOME" not in raw and b"cd" * 16 not in raw
    a.close()
    b.close()


@pytest.mark.parametrize("cipher", _CIPHERS)
def test_sealed_channel_refuses_a_flipped_byte(cipher, tmp_path):
    # covers: TRANSPORT-I9
    # WOULD-FAIL-IF: a sealed Channel accepts a tampered record, lets the record layer's error
    # escape as something other than ChannelClosed, keeps a partial file of a tampered shard, or
    # can be sealed after it spoke in clear or sealed twice.
    def captured(send):
        a, b = socket.socketpair()
        ch = control.Channel(a)
        ch.seal(cipher, _K_AB, _K_BA, _P_AB, _P_BA)
        t = threading.Thread(target=lambda: (send(ch), a.shutdown(socket.SHUT_WR)))
        t.start()
        raw = bytearray()
        b.settimeout(10)
        while chunk := b.recv(1 << 20):
            raw.extend(chunk)
        t.join(10)
        assert not t.is_alive()
        a.close()
        b.close()
        return bytes(raw)

    opened = []

    def receiver(raw):
        w, r = socket.socketpair()
        ch = control.Channel(r)
        ch.seal(cipher, _K_BA, _K_AB, _P_BA, _P_AB)
        t = threading.Thread(target=lambda: (w.sendall(raw), w.shutdown(socket.SHUT_WR)))
        t.start()
        opened.append((t, w, r))
        return ch

    def close_all():
        for t, w, r in opened:
            r.close()  # a refused reader stops reading; closing it unblocks the writer
            t.join(10)
            assert not t.is_alive()
            w.close()

    msg = captured(lambda ch: ch.send({"kind": "ASSIGN", "slot": 1}, b"x" * 300))
    assert receiver(msg).recv() == ({"kind": "ASSIGN", "slot": 1}, b"x" * 300)
    for pos in (0, 3, 4, len(msg) // 2, len(msg) - 1):
        bad = bytearray(msg)
        bad[pos] ^= 0x01
        with pytest.raises(ChannelClosed):
            receiver(bytes(bad)).recv()
    src = tmp_path / "shard.bin"
    src.write_bytes(os.urandom(1_500_000))  # two records; flip one in the second
    shard = captured(lambda ch: ch.send_file({"kind": "CKPT"}, str(src)))
    bad = bytearray(shard)
    bad[-100] ^= 0x01
    inbox = tmp_path / "in"
    with pytest.raises(ChannelClosed):
        receiver(bytes(bad)).recv(str(inbox))
    assert os.listdir(inbox) == []
    close_all()
    # sealed once, before the first message
    a, b = socket.socketpair()
    ch = control.Channel(a)
    ch.seal(cipher, _K_AB, _K_BA, _P_AB, _P_BA)
    with pytest.raises(ValueError):
        ch.seal(cipher, _K_AB, _K_BA, _P_AB, _P_BA)
    clear = control.Channel(b)
    clear.send({"kind": "PING"})
    with pytest.raises(ValueError):
        clear.seal(cipher, _K_BA, _K_AB, _P_BA, _P_AB)
    a.close()
    b.close()


def test_channel_recv_bounds_bodies_before_reading_them(tmp_path, monkeypatch):
    # covers: TRANSPORT-I9
    # WOULD-FAIL-IF: Channel.recv streams a file body to disk on a read that passed no file_dir
    # (the pre-admission HELLO read), ignores max_file or max_inline, or checks a bound only
    # after reading the body (the reader would then wait for bytes that never come).
    cipher = secure.CHACHA20_POLY1305
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    os.makedirs(tmp_path / "tmp")

    def pair():
        a, b = socket.socketpair()
        w, r = control.Channel(a), control.Channel(b)
        w.seal(cipher, _K_AB, _K_BA, _P_AB, _P_BA)
        r.seal(cipher, _K_BA, _K_AB, _P_BA, _P_AB)
        b.settimeout(5)  # a regression blocks on the body and fails with TimeoutError
        return w, r

    def declare(w, msg, blen):
        head = json.dumps(msg).encode()
        w.sock.sendall(struct.pack("<I", len(head)) + head + struct.pack("<Q", blen))

    inbox = tmp_path / "in"
    w, r = pair()
    declare(w, {"kind": "HELLO", "file": True}, 1 << 62)
    with pytest.raises(ChannelClosed, match="takes no file"):
        r.recv()
    assert os.listdir(tmp_path / "tmp") == []
    w.close()
    r.close()

    w, r = pair()
    declare(w, {"kind": "CKPT", "file": True}, (1 << 20) + 1)
    with pytest.raises(ChannelClosed, match="bound"):
        r.recv(str(inbox), max_file=1 << 20)
    assert not inbox.exists()  # refused before the directory or a .part file was made
    w.close()
    r.close()

    w, r = pair()
    declare(w, {"kind": "HELLO"}, (64 << 10) + 1)
    with pytest.raises(ChannelClosed, match="bound"):
        r.recv(max_inline=64 << 10)
    w.close()
    r.close()

    # honest arms: bodies at the bound are accepted unchanged
    src = tmp_path / "shard.bin"
    src.write_bytes(os.urandom(1 << 20))
    w, r = pair()
    t = threading.Thread(target=w.send_file, args=({"kind": "CKPT"}, str(src)))
    t.start()
    msg, path = r.recv(str(inbox), max_file=1 << 20)
    t.join(10)
    with open(path, "rb") as f:
        assert msg["kind"] == "CKPT" and f.read() == src.read_bytes()
    t = threading.Thread(target=w.send, args=({"kind": "HELLO"}, b"h" * (64 << 10)))
    t.start()
    assert r.recv(max_inline=64 << 10) == ({"kind": "HELLO"}, b"h" * (64 << 10))
    t.join(10)
    w.close()
    r.close()


def test_a_whole_frame_reports_its_first_to_last_fragment_spread():
    # WOULD-FAIL-IF: the spread is measured from another message's first fragment, kept across messages, or
    # nonzero for an unfragmented frame (the "recv" telemetry event reads it).
    import time

    r = fr.Reassembler()
    a = [fr.Frame(fr.FWD, fr.OBJ, 1, m, 0, b"x", (), i, 2) for m in (0, 1) for i in (0, 1)]
    assert r.push(a[0]) is None
    time.sleep(0.05)
    assert r.push(a[2]) is None  # mb 1's first fragment, later
    assert r.push(a[1]) is not None and r.spread_s >= 0.05
    assert r.push(a[3]) is not None and r.spread_s < 0.05
    assert r.push(fr.Frame(fr.FWD, fr.OBJ, 1, 2, 0, b"x", ())) is not None and r.spread_s == 0.0


def test_pacing_is_on_by_default_off_on_request_and_never_raises(monkeypatch):
    # covers: TRANSPORT-I13
    # WOULD-FAIL-IF: pacing raised on a kernel without SO_MAX_PACING_RATE, ignored WANTRAIN_TCP_PACE_MBPS=off, or
    # (on Linux) left the cap unset, so the kernel kept bursting each window at line rate.
    from wantrain.transport import tuning

    s = socket.socket()
    try:
        monkeypatch.setenv("WANTRAIN_TCP_PACE_MBPS", "off")
        assert tuning.pace(s) == "off"
        monkeypatch.delenv("WANTRAIN_TCP_PACE_MBPS")
        got = tuning.pace(s)
        if tuning.SO_MAX_PACING_RATE is None:
            assert got == "unsupported"
        else:
            assert got == int(tuning.DEFAULT_PACE_MBPS * 1e6 / 8)
        monkeypatch.setattr(tuning, "SO_MAX_PACING_RATE", None)
        assert tuning.pace(s) == "unsupported"
        assert tuning.tune(s)["pace"] == "unsupported"
    finally:
        s.close()
