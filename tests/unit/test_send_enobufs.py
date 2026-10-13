"""A transient send refusal is not a lost peer (TRANSPORT-I7 record layer).

On macOS six workers at once exhaust the kernel's socket buffers and sendmsg
fails with ENOBUFS (55; 105 on Linux); a non-blocking socket may also return
EAGAIN/EWOULDBLOCK. A failed sendmsg writes nothing, so `secure.sendmsg_all`
offers exactly the remaining bytes again after a capped exponential backoff,
for at most SEND_RETRY_BUDGET_S without progress; then, and on any other
error at once, the error surfaces as before (Channel -> ChannelClosed, which
the coordinator turns into WorkerLost, coordinator.py `_request`).

`_FlakyKernel` is a fake at a true external boundary: the kernel's answers to
sendmsg. It refuses or truncates attempts on a script and passes every other
attempt to a real socketpair end, so the bytes the receiver authenticates are
the bytes really written.
"""

from __future__ import annotations

import errno
import os
import socket
import threading
import time

import pytest

from wantrain.transport import control, secure
from wantrain.transport.control import ChannelClosed

_K_AB, _K_BA = bytes(range(32)), bytes(range(100, 132))
_P_AB, _P_BA = b"\x01\x02\x03\x04", b"\x0a\x0b\x0c\x0d"


class _FlakyKernel:
    """sendmsg answers from `script` (one entry per attempt, then real sends):
    an errno raises OSError having written nothing; ("partial", k) writes only
    the first k bytes; None is a real send."""

    def __init__(self, sock: socket.socket, script):
        self._sock = sock
        self.script = list(script)
        self.attempts = 0
        self.wire = bytearray()

    def sendmsg(self, parts):
        self.attempts += 1
        step = self.script.pop(0) if self.script else None
        if isinstance(step, int):
            raise OSError(step, os.strerror(step))
        data = b"".join(bytes(p) for p in parts)
        if step is not None:
            data = data[: step[1]]
        self._sock.sendall(data)
        self.wire += data
        return len(data)

    def __getattr__(self, name):
        return getattr(self._sock, name)


def _pair(script, threads=1):
    a, b = socket.socketpair()
    kernel = _FlakyKernel(a, script)
    tx = secure.SecureSocket(
        kernel, secure.AES_256_GCM, _K_AB, _K_BA, _P_AB, _P_BA, threads=threads
    )
    rx = secure.SecureSocket(b, secure.AES_256_GCM, _K_BA, _K_AB, _P_BA, _P_AB, threads=threads)
    return kernel, tx, rx


def _reference_wire(messages, threads=1):
    """The wire bytes of the same messages over a kernel that never refuses."""
    kernel, tx, rx = _pair([], threads)
    got = []
    reader = threading.Thread(
        target=lambda: got.append(_read(rx, sum(map(len, messages)))), daemon=True
    )
    reader.start()
    for m in messages:
        tx.sendall(m)
    reader.join(30)
    tx.close()
    rx.close()
    return bytes(kernel.wire)


def _read(rx, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = rx.recv(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


@pytest.mark.parametrize("errno_", [errno.ENOBUFS, errno.EAGAIN, errno.EWOULDBLOCK])
@pytest.mark.parametrize("threads", [1, 4])
def test_transient_refusals_then_partial_sends_deliver_every_record_once_in_order(errno_, threads):
    """Refusals and partial sends in the middle of records: the receiver opens every
    record (its implicit counter authenticates order and uniqueness) and gets exactly
    the sender's bytes; the wire is byte-identical to a kernel that never refused."""
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: a transient errno is treated as a lost peer (OSError raised), a retry
    # resends bytes already written or skips some (tag mismatch / short read), or a partial
    # send after a refusal loses its offset (the wire differs from the reference wire).
    messages = [
        b"head-" + bytes(range(256)) * 3,
        os.urandom(secure.MAX_RECORD * 2 + 12345),  # several records (pool when threads > 1)
        b"tail",
    ]
    script = [errno_, errno_, ("partial", 3), errno_, None, ("partial", 70000), errno_] * 3
    kernel, tx, rx = _pair(script, threads)
    total = sum(map(len, messages))
    got: list[bytes] = []
    reader = threading.Thread(target=lambda: got.append(_read(rx, total)), daemon=True)
    reader.start()
    for m in messages:
        tx.sendall(m)
    reader.join(60)
    assert not reader.is_alive()
    assert got == [b"".join(messages)]
    assert kernel.script == [], "every scripted refusal was exercised"
    assert bytes(kernel.wire) == _reference_wire(messages, threads)
    assert tx.stats()["records_out"] == rx.stats()["records_in"]
    tx.close()
    rx.close()


def test_a_kernel_that_keeps_refusing_past_the_budget_loses_the_peer(monkeypatch):
    """ENOBUFS without progress for SEND_RETRY_BUDGET_S: the send fails as before, the
    Channel reports ChannelClosed (the coordinator's WorkerLost) and the direction stays
    failed."""
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the retry is unbounded (the send never returns), or the budget is not
    # enforced (it returns early or swallows the error and reports success).
    monkeypatch.setattr(secure, "SEND_RETRY_BUDGET_S", 0.3)
    a, b = socket.socketpair()
    kernel = _FlakyKernel(a, [errno.ENOBUFS] * 10_000)
    ch = control.Channel(kernel)
    ch.seal(secure.AES_256_GCM, _K_AB, _K_BA, _P_AB, _P_BA)
    t0 = time.monotonic()
    with pytest.raises(ChannelClosed, match="No buffer space|55|105"):
        ch.send({"kind": "PING"})
    took = time.monotonic() - t0
    assert 0.3 <= took < 3.0, took
    assert 3 <= kernel.attempts < 200, kernel.attempts  # backoff, not a spin
    assert kernel.wire == b""
    with pytest.raises(ChannelClosed):
        ch.send({"kind": "PING"})
    ch.sock.close()
    b.close()


@pytest.mark.parametrize("errno_", [errno.EPIPE, errno.ECONNRESET])
def test_a_non_transient_error_loses_the_peer_at_once(errno_):
    """EPIPE or ECONNRESET: one attempt, no backoff, ChannelClosed."""
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the retry set includes errors that mean the peer is gone (a second
    # attempt is made, or the send waits out the budget).
    a, b = socket.socketpair()
    kernel = _FlakyKernel(a, [errno_] * 100)
    ch = control.Channel(kernel)
    ch.seal(secure.AES_256_GCM, _K_AB, _K_BA, _P_AB, _P_BA)
    t0 = time.monotonic()
    with pytest.raises(ChannelClosed):
        ch.send({"kind": "PING"})
    assert kernel.attempts == 1
    assert time.monotonic() - t0 < 0.5
    ch.sock.close()
    b.close()


def test_the_enobufs_seam_refuses_every_nth_attempt_and_changes_no_byte(monkeypatch):
    """WANTRAIN_TEST_ENOBUFS=N: every N-th attempt is refused before the kernel; the wire
    equals the seam-off wire. Malformed values are refused."""
    # covers: TRANSPORT-I7
    # WOULD-FAIL-IF: the seam writes, drops or reorders bytes (wire differs), never fires
    # (attempt count equals the seam-off count), or accepts a value that refuses every send.
    messages = [os.urandom(5000), os.urandom(secure.MAX_RECORD + 1), b"x"]
    plain = _reference_wire(messages)
    monkeypatch.setenv("WANTRAIN_TEST_ENOBUFS", "2")
    kernel, tx, rx = _pair([])
    total = sum(map(len, messages))
    got: list[bytes] = []
    reader = threading.Thread(target=lambda: got.append(_read(rx, total)), daemon=True)
    reader.start()
    for m in messages:
        tx.sendall(m)
    reader.join(30)
    assert got == [b"".join(messages)]
    assert bytes(kernel.wire) == plain
    assert tx._send_attempts >= 2 * kernel.attempts - 1
    tx.close()
    rx.close()
    for bad in ("1", "0", "-3", "x", "2.5"):
        with pytest.raises(ValueError, match="WANTRAIN_TEST_ENOBUFS"):
            secure.enobufs_seam(bad)
    assert secure.enobufs_seam(None) == 0 and secure.enobufs_seam("") == 0
