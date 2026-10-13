"""Injectable WAN emulation, applied per link inside the transport layer.

macOS has no tc/netem, so emulation lives here and behaves identically on
every platform. Per link: latency_ms delays delivery of a received frame to
the engine until arrival + latency on the receiver's clock; bw_mbps paces the
sender's writes at size/BW per frame; jitter_ms adds uniform random delay per
frame from a dedicated RNG that never touches torch's seed, so network timing
cannot perturb numerics. The same knobs drive the bandwidth sweeps of the
overlap-frontier experiment.

Shared-pacer mode (TRANSPORT-I10, opt-in): a LinkSpec may also carry
SharedBudget objects. Every connection of a peer holds the same egress and
ingress budget, and every connection between two sites the same uplink budget
per direction, so concurrent flows share one bandwidth instead of each getting
its own. Pacing only delays bytes; it never touches them.
"""

import fcntl
import heapq
import math
import mmap
import os
import random
import stat
import struct
import tempfile
import threading
import time
from dataclasses import dataclass

_STATE = struct.Struct("<dd")  # virtual clock (time.monotonic seconds), rate (Mbit/s)
_MAX_BACKLOG_S = 3600.0


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class SharedBudget:
    """A token bucket shared by every thread and every local process that opens it.

    The state lives in a 16-byte host file (mode 0600): the bucket's virtual
    clock, i.e. the time.monotonic() at which its last reserved byte has been
    sent, and its rate. time.monotonic() is host-wide (CLOCK_MONOTONIC /
    CLOCK_UPTIME_RAW), so processes of one host agree on it. A thread lock plus
    flock on the file serialize reservations; each process uses its own
    descriptor (a fork child reopens), so flock excludes processes and the lock
    excludes threads. The bucket holds burst_ms of tokens, so a sleep that wakes
    late is made up instead of lost; over a window T it carries at most
    rate * (T + burst_ms).

    path=None creates a private temporary file that close() removes; pickling
    carries the path, so a copy in another process shares the same bucket. A
    named path must be fresh per run: a clock more than an hour ahead (a file
    from before a reboot) is refused, never waited on.
    """

    def __init__(self, mbps: float, path: str | os.PathLike | None = None, burst_ms: float = 10.0):
        if not (_number(mbps) and math.isfinite(mbps) and mbps > 0):
            raise ValueError(f"SharedBudget rate must be a finite positive Mbit/s, got {mbps!r}")
        if not (_number(burst_ms) and math.isfinite(burst_ms) and burst_ms >= 0):
            raise ValueError(f"SharedBudget burst_ms must be finite and >= 0, got {burst_ms!r}")
        self.mbps = float(mbps)
        self.burst_ms = float(burst_ms)
        self._owner_pid: int | None = None
        if path is None:
            fd, path = tempfile.mkstemp(prefix="wantrain-budget-")
            os.close(fd)
            self._owner_pid = os.getpid()
        self.path = os.fspath(path)
        self._lock = threading.Lock()
        self._fd = -1
        self._mm: mmap.mmap | None = None
        self._pid = -1
        self._closed = False
        self._open()

    def _open(self):
        # Never through a symlink, and only a file this user owns: the budget
        # is mapped and written, so a planted path must not redirect it.
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
                raise ValueError(f"{self.path} is not a regular file owned by this user")
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                size = os.fstat(fd).st_size
                if size == 0:
                    os.ftruncate(fd, _STATE.size)
                    os.pwrite(fd, _STATE.pack(0.0, self.mbps), 0)
                elif size != _STATE.size:
                    raise ValueError(f"{self.path} is not a SharedBudget file ({size} bytes)")
                mm = mmap.mmap(fd, _STATE.size)
                _, rate = _STATE.unpack_from(mm, 0)
                if rate != self.mbps:
                    mm.close()
                    raise ValueError(f"{self.path} holds a {rate} Mbit/s budget, not {self.mbps}")
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        except BaseException:
            os.close(fd)
            raise
        self._fd, self._mm, self._pid = fd, mm, os.getpid()

    def reserve(self, nbytes: int) -> float:
        """Reserve nbytes on the bucket; returns the monotonic time they are sent by."""
        if nbytes < 0:
            raise ValueError(f"cannot reserve {nbytes} bytes")  # would rewind the shared clock
        cost = nbytes * 8 / (self.mbps * 1e6)
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self.path}: SharedBudget is closed")
            if self._pid != os.getpid():
                self._close_handles()
                self._open()
            mm = self._mm
            if mm is None:
                raise RuntimeError(f"{self.path}: SharedBudget has no mapping")
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            try:
                now = time.monotonic()
                clock, _ = _STATE.unpack_from(mm, 0)
                if not math.isfinite(clock) or clock - now > _MAX_BACKLOG_S:
                    raise RuntimeError(f"{self.path}: budget clock {clock!r} is stale or corrupt")
                end = max(clock, now - self.burst_ms / 1e3) + cost
                _STATE.pack_into(mm, 0, end, self.mbps)
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        return end

    def pace(self, nbytes: int):
        wait = self.reserve(nbytes) - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _close_handles(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def close(self):
        with self._lock:
            self._closed = True
            self._close_handles()
            if self._owner_pid == os.getpid():
                self._owner_pid = None
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass

    def __getstate__(self):
        return {"mbps": self.mbps, "path": self.path, "burst_ms": self.burst_ms}

    def __setstate__(self, state):
        self.__init__(state["mbps"], state["path"], state["burst_ms"])


@dataclass
class LinkSpec:
    latency_ms: float = 0.0
    bw_mbps: float = 0.0  # 0 = unlimited
    jitter_ms: float = 0.0
    jitter_seed: int = 0
    # Shared-pacer mode (TRANSPORT-I10): None keeps per-connection pacing only.
    egress: SharedBudget | None = None  # the sending peer's NIC, shared by its connections
    uplink: SharedBudget | None = None  # the site pair's uplink in this direction
    ingress: SharedBudget | None = None  # the receiving peer's NIC, shared by its connections


class SenderPacer:
    def __init__(self, spec: LinkSpec):
        self.spec = spec

    def pace(self, nbytes: int):
        spec = self.spec
        shared = [b for b in (spec.egress, spec.uplink) if b is not None]
        if not shared:
            if spec.bw_mbps > 0:
                time.sleep(nbytes * 8 / (spec.bw_mbps * 1e6))
            return
        # The chunk crosses every budget at once: it is sent when the slowest is done.
        end = time.monotonic()
        if spec.bw_mbps > 0:
            end += nbytes * 8 / (spec.bw_mbps * 1e6)
        for b in shared:
            end = max(end, b.reserve(nbytes))
        wait = end - time.monotonic()
        if wait > 0:
            time.sleep(wait)


class DelayedDelivery:
    """Receiver-side delay queue: frames become visible to the engine only
    after the configured latency (+ jitter) has elapsed."""

    def __init__(self, spec: LinkSpec, deliver):
        self.spec = spec
        self.deliver = deliver  # callback into the engine inbox
        self._rng = random.Random(spec.jitter_seed)
        self._heap = []
        self._cv = threading.Condition()
        self._stop = False
        self._seq = 0
        self._thread: threading.Thread | None
        if spec.latency_ms > 0 or spec.jitter_ms > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            self._thread = None

    def push(self, item, nbytes: int | None = None):
        """Queue item for delivery. With an ingress budget, the caller (the
        connection's reader) is first paced through it for nbytes, so a busy
        ingress back-pressures every sender of that peer."""
        if self.spec.ingress is not None:
            if nbytes is None:
                raise ValueError("a link with an ingress budget needs push(item, nbytes)")
            self.spec.ingress.pace(nbytes)
        if self._thread is None:
            self.deliver(item)
            return
        delay = self.spec.latency_ms + self._rng.uniform(0, self.spec.jitter_ms)
        due = time.monotonic() + delay / 1e3
        with self._cv:
            self._seq += 1
            heapq.heappush(self._heap, (due, self._seq, item))
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while not self._heap and not self._stop:
                    self._cv.wait()
                if self._stop and not self._heap:
                    return
                due, _, item = self._heap[0]
                now = time.monotonic()
                if due > now:
                    self._cv.wait(timeout=due - now)
                    continue
                heapq.heappop(self._heap)
            self.deliver(item)
