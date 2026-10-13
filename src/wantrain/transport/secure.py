"""AEAD record layer for the data plane (TRANSPORT-I7).

Every byte written through a `SecureSocket` is sealed in a record:

    u32 big-endian length L | ciphertext || 16-byte tag   (L = len(ciphertext) + 16)

with at most `MAX_RECORD` plaintext bytes per record. The length header is the
associated data. The nonce is the direction's 4-byte prefix followed by its
64-bit big-endian record counter; the counter is implicit (never on the wire),
starts at 0 and only increases, so a replayed, reordered or dropped record
fails its tag. The counter is never reset, not even by a rekey, so a nonce
never repeats in a direction at all.

Each direction rekeys deterministically before sealing (or opening) the record
that would pass `rekey_records` records or `rekey_bytes` plaintext bytes under
one key: key[e+1] = HKDF-Expand-SHA256(key[e], "wantrain rekey" || u64be(e+1)).
Both ends apply the same rule to the same record sequence, so no message is
exchanged. Defaults sit inside RFC 8446 section 5.5's AES-GCM bound.

Any failure on the receive side (bad tag, truncated record, oversized length,
counter past its limit) raises ConnectionError and leaves the stream failed:
it never resynchronizes. There is no plaintext or null-cipher mode.

Large messages are sealed and opened on a small per-socket thread pool (the
AEAD calls release the GIL). The sender reserves each record's nonce (and any
rekey) in record order before handing it to the pool and writes the sealed
records in that same order. The receiver, from a full-size record on, hands
each complete record to the pool in counter order, keeps reading the socket
while they open, and delivers them in order, refusing at the first bad one
exactly where the serial path would. Either way the bytes on the wire are
those of the serial path. A message of one record, and every record outside
a large message, stays on the calling thread. At most `threads` records are
in flight per direction.

Records are sealed and opened into reused buffers (`encrypt_into` /
`decrypt_into`), never into fresh objects: on send, into a ring of `threads`
slots, a slot taken again only after its record was written; on receive,
into one buffer, or a ring of `threads` once the pool is on, a slot taken
again only after its plaintext was delivered and fully consumed (recv and
recv_into copy out of it, so no caller ever holds a view of it). A receive
buffer grows only after a record of the new size has authenticated (a record
larger than every earlier one opens one-shot), so a forged record leaves
nothing allocated behind. A failed
open leaves unauthenticated plaintext in its slot (cryptography writes it
before checking the tag), so every refusal wipes the receive plaintext
buffers once no pool job writes them. Never in place: the binding's input
and output buffers must not overlap.

`python -m wantrain.transport.secure --bench --link-gbps G` measures one
core's seal and open throughput on 1 MiB records.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import errno
import functools
import json
import os
import platform
import select
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from concurrent.futures import wait as _wait_all

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

AES_256_GCM = "aes-256-gcm"
CHACHA20_POLY1305 = "chacha20-poly1305"
_CIPHERS: dict[str, Callable[[bytes], AESGCM | ChaCha20Poly1305]] = {
    AES_256_GCM: AESGCM,
    CHACHA20_POLY1305: ChaCha20Poly1305,
}

MAX_RECORD = 1 << 20
TAG = 16
KEY_LEN = 32
PREFIX_LEN = 4
REKEY_RECORDS = 1 << 24
REKEY_BYTES = 1 << 36
COUNTER_LIMIT = 1 << 64
# Pool threads per SecureSocket for messages of more than one record; 1 seals and
# opens every record on the calling thread.
THREADS = max(1, min(4, os.cpu_count() or 1))
_LEN = struct.Struct(">I")
# A send the kernel refuses for lack of buffer space (ENOBUFS: six workers on one
# Mac exhaust its mbufs) or would block on (EAGAIN/EWOULDBLOCK) sent nothing, so
# the remaining bytes are offered again after a backoff that starts at
# SEND_RETRY_FIRST_S and doubles up to SEND_RETRY_MAX_S. A send that makes no
# progress for SEND_RETRY_BUDGET_S raises its last error: the peer is lost as
# before. Any other error raises at once.
SEND_RETRY_BUDGET_S = 5.0
SEND_RETRY_FIRST_S = 0.001
SEND_RETRY_MAX_S = 0.1
_TRANSIENT_SEND = frozenset({errno.ENOBUFS, errno.EAGAIN, errno.EWOULDBLOCK})
_REKEY_INFO = b"wantrain rekey"
_NOTHING = memoryview(b"")


def enobufs_seam(value) -> int:
    """WANTRAIN_TEST_ENOBUFS="N" (N >= 2): every N-th send attempt of each
    SecureSocket is refused with ENOBUFS before it reaches the kernel, so
    nothing is written and the attempt is retried. Timing only: the same bytes
    leave in the same order, later. Unset or empty -> 0 (off)."""
    if value is None or value == "":
        return 0
    if not value.isdigit() or int(value) < 2:
        raise ValueError(f"WANTRAIN_TEST_ENOBUFS: {value!r} is not an integer >= 2")
    return int(value)


def sendmsg_all(
    sock: socket.socket, parts: list[memoryview], before: Callable[[], None] | None = None
) -> None:
    """Write every byte of `parts`, in order, exactly once. A failed sendmsg
    wrote nothing (the syscall returns -1), so a transient one is retried from
    the same offset; a partial send drops what left and offers the rest.
    `before` runs ahead of every attempt (the ENOBUFS test seam)."""
    parts = list(parts)
    delay, stalled_since = SEND_RETRY_FIRST_S, None
    while parts:
        try:
            if before is not None:
                before()
            sent = sock.sendmsg(parts)
        except OSError as e:
            if e.errno not in _TRANSIENT_SEND:
                raise
            now = time.monotonic()
            if stalled_since is None:
                stalled_since = now
            elif now - stalled_since >= SEND_RETRY_BUDGET_S:
                raise
            time.sleep(delay)
            delay = min(2 * delay, SEND_RETRY_MAX_S)
            continue
        delay, stalled_since = SEND_RETRY_FIRST_S, None
        while parts and sent >= len(parts[0]):
            sent -= len(parts[0])
            parts.pop(0)
        if parts and sent:
            parts[0] = parts[0][sent:]


def _read_text(path: str) -> str:
    with open(path, encoding="ascii", errors="replace") as f:
        return f.read()


def _sysctl(name: str) -> str:
    out = subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True
    )
    return out.stdout.strip()


def _detect_aes(
    system: str,
    machine: str,
    read_text: Callable[[str], str] = _read_text,
    sysctl: Callable[[str], str] = _sysctl,
) -> bool:
    """True only on positive evidence of AES instructions; False when it cannot tell."""
    try:
        machine = machine.lower()
        x86 = machine in ("x86_64", "amd64", "i386", "i686")
        arm = machine in ("aarch64", "arm64")
        if system == "Linux" and (x86 or arm):
            key = "flags" if x86 else "features"
            for line in read_text("/proc/cpuinfo").splitlines():
                name, _, value = line.partition(":")
                if name.strip().lower() == key:
                    return "aes" in value.split()
            return False
        if system == "Darwin" and arm:
            return sysctl("hw.optional.arm.FEAT_AES") == "1"
        if system == "Darwin" and x86:
            return "AES" in sysctl("machdep.cpu.features").upper().split()
    except Exception:
        return False
    return False


@functools.lru_cache(maxsize=1)
def hardware_aes() -> bool:
    """AES-NI (x86 'aes' flag) or ARMv8 AES (Linux 'aes' feature, macOS FEAT_AES)."""
    return _detect_aes(platform.system(), platform.machine())


def _next_key(key: bytes, epoch: int) -> bytes:
    info = _REKEY_INFO + epoch.to_bytes(8, "big")
    return HKDFExpand(algorithm=hashes.SHA256(), length=KEY_LEN, info=info).derive(key)


class _Direction:
    """One direction's key schedule: key, epoch, counter and the rekey rule."""

    def __init__(
        self,
        cipher: str,
        key: bytes,
        prefix: bytes,
        rekey_records: int,
        rekey_bytes: int,
        counter_limit: int,
    ):
        self._make = _CIPHERS[cipher]
        self.key = bytes(key)
        self.aead = self._make(self.key)
        self.prefix = bytes(prefix)
        self.epoch = 0
        self.counter = 0
        self.epoch_records = 0
        self.epoch_bytes = 0
        self.rekey_records = rekey_records
        self.rekey_bytes = rekey_bytes
        self.counter_limit = counter_limit

    def next_nonce(self, nbytes: int) -> bytes:
        if self.counter >= self.counter_limit:
            raise ConnectionError("record counter exhausted; the connection must be renewed")
        if self.epoch_records >= self.rekey_records or self.epoch_bytes >= self.rekey_bytes:
            self.epoch += 1
            self.key = _next_key(self.key, self.epoch)
            self.aead = self._make(self.key)
            self.epoch_records = 0
            self.epoch_bytes = 0
        nonce = self.prefix + self.counter.to_bytes(8, "big")
        self.counter += 1
        self.epoch_records += 1
        self.epoch_bytes += nbytes
        return nonce

    def reserve(self, nbytes: int) -> tuple[AESGCM | ChaCha20Poly1305, bytes]:
        """The next record's key and nonce, in record order (rekeying first if due)."""
        nonce = self.next_nonce(nbytes)  # may rekey: read self.aead only after it
        return self.aead, nonce

    def seal_into(self, chunk, aad: bytes, out: memoryview) -> None:
        """Seal `chunk` into `out` (exactly len(chunk) + TAG bytes)."""
        aead, nonce = self.reserve(len(chunk))
        aead.encrypt_into(nonce, chunk, aad, out)

    def open(self, body, aad: bytes) -> bytes:
        aead, nonce = self.reserve(len(body) - TAG)
        return aead.decrypt(nonce, body, aad)

    def open_into(self, body, aad: bytes, out: memoryview) -> None:
        """Open `body` into `out` (exactly len(body) - TAG bytes). On a bad tag, `out`
        holds unauthenticated plaintext: the caller must never deliver it."""
        aead, nonce = self.reserve(len(body) - TAG)
        aead.decrypt_into(nonce, body, aad, out)


def _drain(futures) -> None:
    """Cancel what has not started and wait for the rest: no pool thread may still
    read a caller's buffer or the receive buffer once the caller moves on."""
    futures = list(futures)
    for f in futures:
        f.cancel()
    _wait_all(futures)


def _slot(ring: list[bytearray], i: int, size: int) -> memoryview:
    """The first `size` bytes of ring slot `i` (slots are taken in order 0, 1, ...),
    growing the slot to `size` first. The caller guarantees nothing still uses it."""
    if i == len(ring):
        ring.append(bytearray(size))
    elif len(ring[i]) < size:
        ring[i] = bytearray(size)
    return memoryview(ring[i])[:size]


def _wipe(ring: list[bytearray]) -> None:
    for b in ring:
        b[:] = bytes(len(b))


def _poke(wake: socket.socket, _fut: Future) -> None:
    with contextlib.suppress(OSError):  # full (a wake is pending anyway) or closed
        wake.send(b"\0")


def _check_args(cipher: str, key: bytes, prefix: bytes, rekey_records: int, rekey_bytes: int):
    if cipher not in _CIPHERS:
        raise ValueError(f"cipher must be one of {sorted(_CIPHERS)}, got {cipher!r}")
    if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_LEN:
        raise ValueError(f"keys must be {KEY_LEN} bytes")
    if not isinstance(prefix, (bytes, bytearray)) or len(prefix) != PREFIX_LEN:
        raise ValueError(f"nonce prefixes must be {PREFIX_LEN} bytes")
    if rekey_records < 1 or rekey_bytes < 1:
        raise ValueError("rekey thresholds must be positive")


class SecureSocket:
    """A connected stream socket whose every byte is sealed in AEAD records.

    Sending and receiving keep separate state, so one sender thread and one
    reader thread may use the same SecureSocket concurrently. `threads` > 1
    seals and opens multi-record work on a pool of that many threads (created
    on first use, shut down by close()); the wire bytes do not depend on it.
    """

    def __init__(
        self,
        sock: socket.socket,
        cipher: str,
        send_key: bytes,
        recv_key: bytes,
        send_prefix: bytes,
        recv_prefix: bytes,
        *,
        rekey_records: int = REKEY_RECORDS,
        rekey_bytes: int = REKEY_BYTES,
        counter_limit: int = COUNTER_LIMIT,
        threads: int = THREADS,
    ):
        _check_args(cipher, send_key, send_prefix, rekey_records, rekey_bytes)
        _check_args(cipher, recv_key, recv_prefix, rekey_records, rekey_bytes)
        if bytes(send_key) == bytes(recv_key):
            # One key for both directions lets an on-path attacker reflect records.
            raise ValueError("send and receive keys must differ")
        if not 1 <= counter_limit <= COUNTER_LIMIT:
            raise ValueError(f"counter_limit must be in [1, 2^64], got {counter_limit}")
        if not 1 <= threads <= 64:
            raise ValueError(f"threads must be in [1, 64], got {threads}")
        self.cipher = cipher
        self._sock = sock
        self._tx = _Direction(
            cipher, send_key, send_prefix, rekey_records, rekey_bytes, counter_limit
        )
        self._rx = _Direction(
            cipher, recv_key, recv_prefix, rekey_records, rekey_bytes, counter_limit
        )
        self._threads = threads
        self._pool: ThreadPoolExecutor | None = None
        self._refuse_every = enobufs_seam(os.environ.get("WANTRAIN_TEST_ENOBUFS"))
        self._send_attempts = 0
        self._pool_lock = threading.Lock()
        self._closed = False
        # Raw records land in place here and are opened from it; [_start, _end) is
        # received but not yet opened, and survives a socket timeout. Reads look at
        # most 64 KiB ahead until a full-size record has opened; with a pool, the
        # buffer then grows to hold `threads` full records and reads take whatever
        # the kernel has, so the records of a large message open while more arrive.
        self._buf = bytearray(_LEN.size + MAX_RECORD + TAG + (1 << 16))
        self._readahead = 1 << 16
        self._view = memoryview(self._buf)
        self._start = self._end = 0
        # Sealed records: slot i % threads of a message's record i (slot 0 serially).
        self._tx_ring: list[bytearray] = []
        # Opened plaintext: one slot serially, grown only to the size of a record that
        # has authenticated; the pool's `threads` slots only after _grow.
        self._rx_one: list[bytearray] = []
        self._rx_ring: list[bytearray] = []
        self._handed = (
            0  # records handed to the pool: the next one opens into slot _handed % threads
        )
        self._plain = _NOTHING  # opened, not yet returned: a view of an rx slot
        self._plain_off = 0
        # Records handed to the pool, in counter order: (end offset, plaintext slot
        # view, job). Their bytes sit in [_start, _scan) and stay in place until
        # they are delivered; [_scan, _end) is received and not yet handed over.
        self._inflight: collections.deque[tuple[int, memoryview, Future]] = collections.deque()
        self._scan = 0
        self._wire_eof = False  # EOF seen while reading ahead of the pool
        # Wakes the reader when a job finishes while it waits for more bytes.
        self._wake_r: socket.socket | None = None
        self._wake_w: socket.socket | None = None
        self._eof = False
        self._rx_failed: str | None = None
        self._tx_failed: str | None = None
        self._records_out = 0
        self._records_in = 0
        self._bytes_out = 0
        self._bytes_in = 0
        self._seal_ns = 0
        self._open_ns = 0

    def _submit(self, fn: Callable[..., int], *args) -> Future:
        # Under the lock, so close() cannot shut the pool down between the check and
        # the submit; what close() cancels later surfaces as CancelledError.
        with self._pool_lock:
            if self._closed:
                raise ConnectionError("secure socket closed")
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self._threads, thread_name_prefix="wt-aead")
            return self._pool.submit(fn, *args)

    # -- send -------------------------------------------------------------

    def _send_record(self, chunk) -> None:
        t0 = time.perf_counter_ns()
        n = len(chunk)
        hdr = _LEN.pack(n + TAG)
        sealed = _slot(self._tx_ring, 0, n + TAG)
        self._tx.seal_into(chunk, hdr, sealed)
        self._seal_ns += time.perf_counter_ns() - t0
        self._write(hdr, sealed, n)

    def _send_parallel(self, view: memoryview) -> None:
        """Seal the records of one message on the pool, `threads` in flight; nonces
        are reserved here in record order and records leave in that order. Record i
        seals into slot i % threads: when it is reserved, at most threads - 1 records
        are in flight (i - threads + 1 .. i - 1), so record i - threads, the slot's
        previous user, has been written."""
        inflight: collections.deque[tuple[bytes, memoryview, int, Future]] = collections.deque()
        try:
            for i, off in enumerate(range(0, len(view), MAX_RECORD)):
                chunk = view[off : off + MAX_RECORD]
                n = len(chunk)
                hdr = _LEN.pack(n + TAG)
                t0 = time.perf_counter_ns()
                sealed = _slot(self._tx_ring, i % self._threads, n + TAG)
                aead, nonce = self._tx.reserve(n)
                fut = self._submit(aead.encrypt_into, nonce, chunk, hdr, sealed)
                inflight.append((hdr, sealed, n, fut))
                self._seal_ns += time.perf_counter_ns() - t0
                if len(inflight) >= self._threads:
                    self._emit(*inflight[0])
                    inflight.popleft()
            while inflight:
                self._emit(*inflight[0])
                inflight.popleft()
        finally:
            # No pool job may still read the caller's data or write a slot.
            _drain(f for _, _, _, f in inflight)

    def _emit(self, hdr: bytes, sealed: memoryview, n: int, fut: Future) -> None:
        t0 = time.perf_counter_ns()
        try:
            fut.result()  # seal_ns: the time this thread waits for sealing
        except CancelledError:
            raise ConnectionError("secure socket closed") from None
        self._seal_ns += time.perf_counter_ns() - t0
        self._write(hdr, sealed, n)

    def _write(self, hdr: bytes, sealed: memoryview, n: int) -> None:
        # Gather write: header and ciphertext leave in one syscall without being
        # concatenated (no extra pass over the record).
        sendmsg_all(
            self._sock, [memoryview(hdr), sealed], self._refuse if self._refuse_every else None
        )
        self._records_out += 1
        self._bytes_out += n

    def _refuse(self) -> None:
        self._send_attempts += 1
        if self._send_attempts % self._refuse_every == 0:
            raise OSError(errno.ENOBUFS, os.strerror(errno.ENOBUFS))

    def sendall(self, data) -> None:
        if self._tx_failed:
            raise ConnectionError(f"secure send failed earlier: {self._tx_failed}")
        view = memoryview(data).cast("B")
        try:
            if self._threads > 1 and len(view) > MAX_RECORD:
                self._send_parallel(view)
            else:
                for off in range(0, len(view), MAX_RECORD):
                    self._send_record(view[off : off + MAX_RECORD])
        except BaseException as e:
            # A partial record on the wire cannot be resumed: fail the direction.
            self._tx_failed = repr(e)
            raise

    # -- receive ----------------------------------------------------------

    def _fail(self, why: str) -> ConnectionError:
        self._rx_failed = why
        return ConnectionError(f"secure record refused: {why}")

    def _fill(self, n: int) -> bool:
        """Read raw bytes until `n` are buffered; False on EOF before any byte of a record."""
        while self._end - self._start < n:
            if self._start + n > len(self._buf):
                # Compact: only the unread tail moves, and it is short (see `want`).
                have = self._end - self._start
                self._view[:have] = self._view[self._start : self._end]
                self._start, self._end = 0, have
            # Serially, read ahead at most 64 KiB, so many small records cost few
            # syscalls while a large record is read exactly into place (compaction
            # never moves a big tail). With a pool, take all the kernel has: the tail
            # moved is still under one record (start + n > len and end <= len).
            want = max(n - (self._end - self._start), self._readahead)
            stop = min(self._end + want, len(self._buf))
            got = self._sock.recv_into(self._view[self._end : stop])
            if not got:
                if self._end > self._start:
                    have = self._end - self._start
                    raise self._fail(f"truncated record ({have} of {n} bytes)")
                return False
            self._end += got
        return True

    def _next_record(self) -> bool:
        if self._inflight:
            return self._pipeline()
        if not self._fill(_LEN.size):
            return False
        (length,) = _LEN.unpack_from(self._buf, self._start)
        if length <= TAG or length > MAX_RECORD + TAG:
            raise self._fail(f"record length {length} outside ({TAG}, {MAX_RECORD + TAG}]")
        full = self._threads > 1 and length == MAX_RECORD + TAG
        self._fill(_LEN.size + length)
        if full and self._readahead == len(self._buf):
            # A full-size record: the rest of a large message is likely behind it.
            self._scan = self._start
            return self._pipeline()
        body = self._start + _LEN.size
        hdr = bytes(self._view[self._start : body])
        n = length - TAG
        t0 = time.perf_counter_ns()
        record = self._view[body : body + length]
        try:
            if self._rx_one and len(self._rx_one[0]) >= n:
                plain = memoryview(self._rx_one[0])[:n]
                self._rx.open_into(record, hdr, plain)
            else:
                # Larger than every record authenticated so far: opened one-shot, and
                # the slot grows only once this record has authenticated.
                plain = memoryview(self._rx.open(record, hdr))
                self._rx_one = [bytearray(n)]
        except InvalidTag:
            _wipe(self._rx_one)
            raise self._fail("authentication tag mismatch") from None
        except ConnectionError as e:
            raise self._fail(str(e)) from None
        finally:
            self._open_ns += time.perf_counter_ns() - t0
        self._start = body + length
        if self._start == self._end:
            self._start = self._end = 0
        self._records_in += 1
        self._bytes_in += n
        self._plain = plain
        self._plain_off = 0
        if full:
            # Only an authenticated full-size record buys the larger buffer, the pool
            # and the wake pair: a forged header costs the peer nothing but the stream.
            self._grow()
        return True

    def _grow(self) -> None:
        """Make room for `threads` full records; the unread bytes move to the front.
        Called between records, when no pool job reads the buffer."""
        if self._readahead == len(self._buf):
            return
        size = self._threads * (_LEN.size + MAX_RECORD + TAG) + (1 << 16)
        have = self._end - self._start
        buf = bytearray(size)
        buf[:have] = self._view[self._start : self._end]
        self._buf, self._view = buf, memoryview(buf)
        self._start, self._end = 0, have
        self._readahead = size
        self._rx_ring = [bytearray(MAX_RECORD) for _ in range(self._threads)]

    def _pipeline(self) -> bool:
        """Deliver the oldest pooled record, keeping up to `threads` complete records
        opening on the pool and reading the socket while they do.

        Nonces are reserved in counter order as records are handed over, and
        records are delivered in that order, so a refusal surfaces exactly after
        the records before it, as on the serial path. Any error here leaves the
        stream failed: the reserved nonces are spent."""
        try:
            self._hand_over()
            end, plain, fut = self._inflight[0]
            while not fut.done():
                self._read_ahead(fut)
                self._hand_over()
            self._inflight.popleft()
            try:
                fut.result()
            except InvalidTag:
                raise self._fail("authentication tag mismatch") from None
            except CancelledError:
                raise self._fail("secure socket closed") from None
        except BaseException as e:
            _drain(f for _, _, f in self._inflight)
            self._inflight.clear()
            _wipe(self._rx_ring)  # after the drain: no job writes a slot any more
            if not self._rx_failed:
                self._rx_failed = repr(e)
            raise
        n = len(plain)
        self._start = end
        self._records_in += 1
        self._bytes_in += n
        if not self._inflight:
            self._scan = self._start
            if self._start == self._end:
                self._start = self._end = self._scan = 0
        self._plain = plain
        self._plain_off = 0
        return True

    def _hand_over(self) -> None:
        """Submit the complete, well-formed records at _scan, in order, while fewer
        than `threads` are in flight. A malformed or incomplete one waits: once the
        records before it are delivered, the serial path refuses or reads it."""
        while len(self._inflight) < self._threads and self._scan + _LEN.size <= self._end:
            pos = self._scan
            (ln,) = _LEN.unpack_from(self._buf, pos)
            body = pos + _LEN.size
            if ln <= TAG or ln > MAX_RECORD + TAG or body + ln > self._end:
                return
            # Slot h % threads for record h: at most threads - 1 records are in flight
            # (h - threads + 1 .. h - 1) and the one delivered before them is fully
            # consumed (hand-over runs only inside _next_record), so nothing reads it.
            plain = memoryview(self._rx_ring[self._handed % self._threads])[: ln - TAG]
            try:
                aead, nonce = self._rx.reserve(ln - TAG)
                if self._wake_w is None:
                    self._wake_r, self._wake_w = socket.socketpair()
                    self._wake_r.setblocking(False)
                    self._wake_w.setblocking(False)
                fut = self._submit(
                    aead.decrypt_into,
                    nonce,
                    self._view[body : body + ln],
                    self._view[pos:body].tobytes(),
                    plain,
                )
            except ConnectionError as e:  # counter exhausted, or closed under the reader
                if self._inflight:
                    return  # delivered first; the next attempt refuses with nothing in flight
                raise self._fail(str(e)) from None
            fut.add_done_callback(functools.partial(_poke, self._wake_w))
            self._inflight.append((body + ln, plain, fut))
            self._handed += 1
            self._scan = body + ln

    def _read_ahead(self, fut: Future) -> None:
        """Wait until `fut` is done or the socket has bytes, and take those bytes.
        Never blocks on the socket alone: the job's completion wakes the wait.
        open_ns counts only waits that brought no bytes (blocked on opening)."""
        fileno = getattr(self._sock, "fileno", None)
        fd = fileno() if fileno is not None else -1
        t0 = time.perf_counter_ns()
        if fd < 0 or self._wire_eof or self._end >= len(self._buf) or self._wake_r is None:
            _wait_all([fut])
            self._open_ns += time.perf_counter_ns() - t0
            return
        # poll, not select: select refuses any descriptor at or past FD_SETSIZE (1024),
        # which a process with many open files hands out.
        poller = select.poll()
        try:
            poller.register(fd, select.POLLIN)
            poller.register(self._wake_r, select.POLLIN)
            ready = {f for f, _ in poller.poll()}
        except ValueError:  # a descriptor closed under the reader
            raise ConnectionError("secure socket closed") from None
        if fd not in ready:
            self._open_ns += time.perf_counter_ns() - t0
        if self._wake_r.fileno() in ready:
            with contextlib.suppress(BlockingIOError):
                self._wake_r.recv(4096)
        if fd in ready:
            got = self._sock.recv_into(self._view[self._end :])
            if got:
                self._end += got
            else:
                self._wire_eof = True  # the blocking path reads the EOF again

    def _ready(self) -> int:
        if self._rx_failed:
            raise ConnectionError(f"secure stream failed earlier: {self._rx_failed}")
        while self._plain_off >= len(self._plain) and not self._eof:
            # Consumed: its slot may be written again, so drop the view first.
            self._plain, self._plain_off = _NOTHING, 0
            if not self._next_record():
                self._eof = True
        return len(self._plain) - self._plain_off

    def recv(self, bufsize: int) -> bytes:
        if bufsize <= 0:
            return b""
        n = min(bufsize, self._ready())
        out = bytes(self._plain[self._plain_off : self._plain_off + n])
        self._plain_off += n
        return out

    def recv_into(self, buffer, nbytes: int = 0) -> int:
        view = memoryview(buffer).cast("B")
        want = nbytes or len(view)
        if want <= 0:
            return 0
        n = min(want, self._ready())
        view[:n] = self._plain[self._plain_off : self._plain_off + n]
        self._plain_off += n
        return n

    # -- passthrough ------------------------------------------------------

    def shutdown(self, how: int) -> None:
        self._sock.shutdown(how)

    def close(self) -> None:
        with self._pool_lock:
            self._closed = True
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for w in (self._wake_r, self._wake_w):
            if w is not None:
                w.close()
        self._sock.close()

    def settimeout(self, value: float | None) -> None:
        self._sock.settimeout(value)

    def setsockopt(self, *args) -> None:
        self._sock.setsockopt(*args)

    def fileno(self) -> int:
        return self._sock.fileno()

    def stats(self) -> dict[str, int]:
        return {
            "records_out": self._records_out,
            "records_in": self._records_in,
            "bytes_out": self._bytes_out,
            "bytes_in": self._bytes_in,
            "seal_ns": self._seal_ns,
            "open_ns": self._open_ns,
        }


def bench(cipher: str | None = None, seconds: float = 0.5) -> dict:
    """Seal and open 1 MiB records on this thread (one core) for about `seconds` each."""
    hw = hardware_aes()
    cipher = cipher or (AES_256_GCM if hw else CHACHA20_POLY1305)
    payload = os.urandom(MAX_RECORD)
    tx = _Direction(
        cipher,
        os.urandom(KEY_LEN),
        os.urandom(PREFIX_LEN),
        REKEY_RECORDS,
        REKEY_BYTES,
        COUNTER_LIMIT,
    )
    hdr = _LEN.pack(MAX_RECORD + TAG)
    budget = int(seconds * 1e9)

    # Into reused buffers, as SecureSocket seals and opens.
    record = memoryview(bytearray(MAX_RECORD + TAG))
    plain = memoryview(bytearray(MAX_RECORD))

    def timed(op: Callable[[], object]) -> tuple[int, int]:
        n = elapsed = 0
        while elapsed < budget or n < 4:
            t0 = time.perf_counter_ns()
            op()
            elapsed += time.perf_counter_ns() - t0
            n += 1
        return n, elapsed

    nonce = tx.next_nonce(MAX_RECORD)
    tx.aead.encrypt_into(nonce, payload, hdr, record)
    sealed = bytes(record)
    n_seal, seal_ns = timed(lambda: tx.seal_into(payload, hdr, record))
    # Opening the same record under its own nonce costs what opening a fresh one does.
    n_open, open_ns = timed(lambda: tx.aead.decrypt_into(nonce, sealed, hdr, plain))
    if plain != payload:
        raise AssertionError("bench round trip is not byte-identical")
    bits = MAX_RECORD * 8
    return {
        "cipher": cipher,
        "hw_aes": hw,
        "seal_gbps": round(n_seal * bits / seal_ns, 3),
        "open_gbps": round(n_open * bits / open_ns, 3),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wantrain.transport.secure")
    ap.add_argument("--bench", action="store_true", required=True)
    ap.add_argument("--link-gbps", type=float, required=True)
    ap.add_argument("--cipher", choices=sorted(_CIPHERS), default=None)
    ap.add_argument("--seconds", type=float, default=0.5)
    args = ap.parse_args(argv)
    if args.link_gbps <= 0 or args.seconds <= 0:
        ap.error("--link-gbps and --seconds must be positive")
    out = bench(args.cipher, args.seconds)
    out["link_gbps"] = args.link_gbps
    out["ok"] = out["seal_gbps"] >= args.link_gbps and out["open_gbps"] >= args.link_gbps
    out["label"] = "SIMULATED"
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
