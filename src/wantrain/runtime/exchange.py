"""Exchange primitives of a sharded stage (ADR-0006, RUNTIME-I10, RUNTIME-I13).

- NeedFirstQueue: one per (sender, destination). The item the receiver needs
  first leaves first: forward unit index, then gathers before gradients, then
  step and microbatch. Items without a key (LOSS, the shutdown sentinel) sort
  after every exchange item, FIFO among themselves, so shutdown still drains.
- chunk_object / reassemble: one unit's frame payload cut into self-contained
  chunks of at most 16 MiB (or WANTRAIN_TEST_CHUNK_BYTES, chunk_bytes_seam);
  the unit, offset and length travel in the OBJ envelope, so the frame header
  is unchanged (BA-13, BA-40).
- check_frame: the receiver's boundary validator for GRADS and PARAMS chunks.
  It reads only the envelope and the tensors' dtype and length: one dict
  lookup and a few integer comparisons per frame, no tensor op, except one
  sign check over an int64 count lane (E values; RUNTIME-I18).
- CopyEngine: every exchange D2H/H2D copy on its own thread (CUDA: a copy
  stream, pinned chunks of at most 16 MiB, ordered by events; real-hardware
  pending, F7), never on the compute thread.
- HoldCounters: monotonic byte counters of what a peer holds for the
  exchange, with per-step maxima printed as one WT_HOLD line.

Nothing here draws from the torch RNG or changes a byte it carries.
"""

from __future__ import annotations

import heapq
import itertools
import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from typing import Any, NamedTuple

import torch

GATHER = 0  # PARAMS: the owner's updated piece, needed by the next forward
GRAD = 1  # GRADS: a gradient piece, needed by the owner's fold
# GRADS under wire "native" (ZeRO-1): one microbatch's int64 selection counts of
# one MoE bias (RUNTIME-I18, BA-454), needed by the bias owner's update.
COUNT = 2
# Third element of a plan-table entry (dtype, numel, COUNT_LANE): the tensor is
# an int64 count lane (an MoE selection bias's counts, or its piece), RAW_I64.
COUNT_LANE = "count"
CHUNK_BYTES = 16 << 20


def chunk_bytes_seam(value: str | None) -> int:
    """WANTRAIN_TEST_CHUNK_BYTES: the exchange chunk size, a timing-only test
    seam (E4 runs at the scaled f'). It moves where a unit's payload is cut,
    so when its chunks leave; never a byte: the receiver reassembles a piece
    by offset and folds whole pieces in canonical microbatch order. Unset or
    empty -> CHUNK_BYTES; else an integer in [8, CHUNK_BYTES]."""
    if value is None or value == "":
        return CHUNK_BYTES
    try:
        n = int(value)
    except ValueError:
        raise ValueError(f"WANTRAIN_TEST_CHUNK_BYTES={value!r}: not an integer") from None
    if not 8 <= n <= CHUNK_BYTES:
        raise ValueError(f"WANTRAIN_TEST_CHUNK_BYTES={value!r}: outside [8, {CHUNK_BYTES}]")
    return n


class NeedKey(NamedTuple):
    fwd_unit: int
    kind: int  # GATHER or GRAD
    step: int
    mb: int
    seq: int


class NeedFirstQueue:
    """A thread-safe priority queue: pop returns the item with the smallest
    key; a None key sorts after every keyed item, FIFO among such items.
    With fifo=True (the WANTRAIN_TEST_QUEUE=fifo seam, timing only) every
    item leaves in push order. put/get make it a drop-in for the sender
    threads' queue.Queue."""

    def __init__(self, fifo: bool = False):
        self._heap: list = []
        self._seq = itertools.count()
        self._cv = threading.Condition()
        self._fifo = fifo

    def push(self, key: NeedKey | None, item: Any) -> None:
        n = next(self._seq)
        if self._fifo:
            rank: tuple = (0, (), n)
        elif key is None:
            rank = (1, (), n)
        else:
            rank = (0, tuple(key), n)
        with self._cv:
            heapq.heappush(self._heap, (rank, time.monotonic(), item))
            self._cv.notify()

    def pop(self, timeout: float | None = None) -> Any:
        return self.pop_with_times(timeout)[0]

    def pop_with_times(self, timeout: float | None = None) -> tuple:
        """-> (item, pushed_at, popped_at), both time.monotonic() stamped under
        the queue's lock, so "queued before this pop" is exact (the E3 check
        that no item leaves while a more needed one waits)."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._heap, timeout=timeout):
                raise queue.Empty
            _, pushed, item = heapq.heappop(self._heap)
            return item, pushed, time.monotonic()

    def put(self, item: Any) -> None:
        self.push(None, item)

    def get(self) -> Any:
        return self.pop()

    def qsize(self) -> int:
        with self._cv:
            return len(self._heap)


def _spans(numels: Sequence[int]) -> list[int]:
    starts, pos = [], 0
    for n in numels:
        starts.append(pos)
        pos += n
    return starts


def chunk_object(unit, body: Sequence[torch.Tensor], max_bytes: int = CHUNK_BYTES) -> list:
    """Cut one unit's payload (1-D tensors, in unit order) into chunks of at
    most max_bytes of tensor data. -> [(env, view)]: env = {"u": unit, "off":
    first element, "len": elements, "n": every tensor's numel, "t": the tensor
    index of each slice in view}; view = the 1-D slices (no copy) covering
    elements [off, off + len) of the unit's concatenated tensors. A payload of
    empty tensors is one chunk of length 0."""
    if max_bytes < 8:
        raise ValueError(f"a chunk must hold at least one element, max_bytes={max_bytes}")
    numels = [int(t.numel()) for t in body]
    for t in body:
        if t.dim() != 1:
            raise ValueError("chunk_object takes 1-D tensors")
    out: list = []
    idx: list[int] = []
    view: list[torch.Tensor] = []
    off = pos = room = 0
    room = max_bytes

    def close():
        nonlocal idx, view, off, room
        ln = pos - off
        env = {"u": unit, "off": off, "len": ln, "n": list(numels), "t": idx}
        out.append((env, view))
        idx, view, off, room = [], [], pos, max_bytes

    for i, t in enumerate(body):
        esz = t.element_size()
        lo = 0
        if numels[i] == 0:
            idx.append(i)
            view.append(t)
            continue
        while lo < numels[i]:
            take = min(numels[i] - lo, room // esz)
            if take == 0:
                close()
                continue
            idx.append(i)
            view.append(t[lo : lo + take])
            lo += take
            pos += take
            room -= take * esz
    if view or not out:
        close()
    return out


class FrameError(ValueError):
    """A malformed GRADS or PARAMS frame; cls names its class."""

    def __init__(self, cls: str, msg: str):
        super().__init__(f"{cls}: {msg}")
        self.cls = cls


def _slice_lengths(env: dict) -> list[int]:
    """The length each slice of a chunk must have, from its envelope."""
    starts = _spans(env["n"])
    lo, hi = env["off"], env["off"] + env["len"]
    return [min(hi, starts[i] + env["n"][i]) - max(lo, starts[i]) for i in env["t"]]


def _is_int(x) -> bool:
    return type(x) is int


def check_frame(kind: str, env: Any, tensors: Any, expect: dict) -> None:
    """Validate one received GRADS or PARAMS chunk before any use; raise
    FrameError naming its class: "envelope" (not a chunk envelope), "unit"
    (unknown unit), "mb" (outside [0, M)), "step" (neither the receiver's
    current step nor its next), "dtype" (not the plan's table), "shape" (not
    the piece or slice the plan names, or offsets that do not fit it),
    "duplicate" (a repeated (step, mb, unit, sender, offset)), "count" (a
    count lane holding a negative value).

    env: the chunk envelope plus the frame's "mb", "step" and "sender".
    expect: {"units": {unit: [(dtype, numel) or (int64, numel, COUNT_LANE),
    ...]}, "M": int, "step": int, "seen": set} (seen is updated). A count
    lane's dtype and length are checked like any tensor's; its sum (zero1's
    T_mb*k) is the owner's fold's check (accumulate.CountFold), since a lane
    may be cut across chunks."""
    keys = ("u", "off", "len", "n", "t", "mb", "step", "sender")
    if not isinstance(env, dict) or any(k not in env for k in keys):
        raise FrameError("envelope", f"{kind} chunk without a full envelope")
    if not isinstance(tensors, list) or not isinstance(env["t"], list):
        raise FrameError("envelope", f"{kind} chunk body is not a list of tensors")
    try:
        table = expect["units"].get(env["u"])
    except TypeError:
        table = None
    if table is None:
        raise FrameError("unit", f"{kind} names unknown unit {env['u']!r}")
    mb, step = env["mb"], env["step"]
    if not _is_int(mb) or not 0 <= mb < expect["M"]:
        raise FrameError(
            "mb", f"{kind} unit {env['u']} microbatch {mb!r} outside [0, {expect['M']})"
        )
    cur = expect["step"]
    if not _is_int(step) or step not in (cur, cur + 1):
        raise FrameError("step", f"{kind} unit {env['u']} of step {step!r}; this peer is at {cur}")
    n, idx = env["n"], env["t"]
    if (
        not isinstance(n, list)
        or len(n) != len(table)
        or any(not _is_int(x) for x in n)
        or len(idx) != len(tensors)
        or any(not _is_int(i) or not 0 <= i < len(table) for i in idx)
        # chunk_object lists each tensor once, in unit order: a repeated index
        # would pass the length sum while giving one tensor another's elements
        or any(a >= b for a, b in zip(idx, idx[1:]))
        or any(not isinstance(t, torch.Tensor) for t in tensors)
    ):
        raise FrameError("shape", f"{kind} unit {env['u']}: envelope does not fit the plan")
    for i, t in zip(idx, tensors):
        if t.dtype != table[i][0]:
            raise FrameError(
                "dtype",
                f"{kind} unit {env['u']} tensor {i} is {t.dtype}, the plan says {table[i][0]}",
            )
    total = sum(x[1] for x in table)
    off, ln = env["off"], env["len"]
    if (
        [x[1] for x in table] != n
        or not _is_int(off)
        or not _is_int(ln)
        or off < 0
        or ln < 0
        or off + ln > total
        or any(t.dim() != 1 for t in tensors)
        or [int(t.numel()) for t in tensors] != _slice_lengths(env)
        or sum(int(t.numel()) for t in tensors) != ln
    ):
        raise FrameError("shape", f"{kind} unit {env['u']} chunk is not a slice of its piece")
    for i, t in zip(idx, tensors):
        if len(table[i]) > 2 and table[i][2] == COUNT_LANE and t.numel() and bool((t < 0).any()):
            raise FrameError(
                "count",
                f"{kind} unit {env['u']} (step {step}, mb {mb}, sender {env['sender']}) "
                f"count lane {i} holds a negative count",
            )
    key = (kind, step, mb, env["u"], env["sender"], off)
    if key in expect["seen"]:
        raise FrameError("duplicate", f"{kind} repeats (step {step}, mb {mb}, unit {env['u']})")
    expect["seen"].add(key)


def reassemble(parts: Sequence[tuple]) -> list[torch.Tensor] | None:
    """[(env, tensors)] of one (kind, step, mb, unit, sender), already checked
    by check_frame -> the unit's 1-D tensors once the chunks tile every element
    exactly, None while some are missing. Overlapping chunks raise
    FrameError("shape")."""
    if not parts:
        return None
    n = parts[0][0]["n"]
    total = sum(n)
    got = sum(env["len"] for env, _ in parts)
    if got < total or (total == 0 and not parts):
        return None
    ordered = sorted(parts, key=lambda p: p[0]["off"])
    pos = 0
    pieces: list[list[torch.Tensor]] = [[] for _ in n]
    for env, tensors in ordered:
        if env["n"] != n or env["off"] != pos:
            raise FrameError("shape", f"unit {env['u']} chunks overlap or leave a gap")
        pos += env["len"]
        for i, t in zip(env["t"], tensors):
            pieces[i].append(t)
    if pos != total:
        raise FrameError("shape", "chunks overlap or leave a gap")
    out = []
    for i, ts in enumerate(pieces):
        if len(ts) == 1:
            out.append(ts[0])
        elif ts:
            out.append(torch.cat(ts))
        else:
            raise FrameError("shape", f"tensor {i} of {n[i]} elements never arrived")
        if out[-1].numel() != n[i]:
            raise FrameError("shape", f"tensor {i} reassembled to {out[-1].numel()} of {n[i]}")
    return out


class CopyEngine:
    """Runs every exchange copy on its own thread. submit(fn) -> Future.
    On CUDA it owns a copy stream: d2h waits on an event the compute stream
    recorded, copies into pinned host memory in chunks of at most 16 MiB and
    synchronizes only its own stream; h2d copies on the same stream and
    returns the event the consumer waits on (real-hardware pending, F7).
    `copies` counts the copies it ran; each ran on thread_ident."""

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self.copies = 0
        self.ran_on: set[int] = set()  # thread idents every copy ran on
        self._stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self._t = threading.Thread(target=self._run, name="copy-engine", daemon=True)
        self._t.start()
        self.thread_ident = self._t.ident

    def _run(self):
        while True:
            job = self._q.get()
            if job is None:
                return
            fn, fut = job
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        fut.set_result(fn())
                else:
                    fut.set_result(fn())
            except BaseException as e:  # noqa: BLE001 -- surfaced by Future.result()
                fut.set_exception(e)

    def submit(self, fn: Callable[[], Any]) -> Future:
        fut: Future = Future()
        self._q.put((fn, fut))
        return fut

    def d2h(self, tensors: Sequence[torch.Tensor]) -> Future:
        """Host copies of `tensors` (detached; CPU tensors are already host
        memory and are only detached). The producer's tensors must not be
        written again until the future resolves."""
        event = None
        if self._stream is not None:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        ts = list(tensors)

        def run():
            self.copies += 1
            self.ran_on.add(threading.get_ident())
            if self._stream is None:
                return [t.detach() for t in ts]
            assert event is not None
            self._stream.wait_event(event)
            out = []
            for t in ts:
                host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                flat, src = host.reshape(-1), t.detach().reshape(-1)
                step = max(1, CHUNK_BYTES // max(1, t.element_size()))
                for lo in range(0, flat.numel(), step):
                    flat[lo : lo + step].copy_(src[lo : lo + step], non_blocking=True)
                out.append(host)
            self._stream.synchronize()
            return out

        return self.submit(run)

    def h2d(self, tensors: Sequence[torch.Tensor]) -> Future:
        """Device copies of host `tensors` (on CPU the tensors themselves)."""
        ts = list(tensors)

        def run():
            self.copies += 1
            self.ran_on.add(threading.get_ident())
            if self._stream is None:
                return [t.to(self.device) for t in ts]
            out = [t.to(self.device, non_blocking=True) for t in ts]
            self._stream.synchronize()
            return out

        return self.submit(run)

    def close(self) -> None:
        self._q.put(None)
        self._t.join(timeout=5.0)


HOLD_KINDS = ("host_grads", "gathers_out", "gathers_in", "device_own")


class HoldCounters:
    """Bytes a peer holds for the exchange, per kind:
    host_grads (gradient pieces queued to send or received and not yet
    folded), gathers_out (gathers queued to send), gathers_in (gathers received
    and not yet installed), device_own (own pieces waiting for their fold
    prefix). add/sub are integer updates under a lock; end_step(step) records
    the maxima since the previous end_step (and `host`, the maximum of the
    first three summed at one instant); line(step) renders them. No tensor op,
    no device sync, no RNG draw."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cur = dict.fromkeys(HOLD_KINDS, 0)
        self._max = dict.fromkeys((*HOLD_KINDS, "host"), 0)
        self.steps: dict[int, dict[str, int]] = {}

    def _host(self) -> int:
        return self._cur["host_grads"] + self._cur["gathers_out"] + self._cur["gathers_in"]

    def add(self, kind: str, nbytes: int) -> None:
        with self._lock:
            self._cur[kind] += nbytes
            self._max[kind] = max(self._max[kind], self._cur[kind])
            self._max["host"] = max(self._max["host"], self._host())

    def sub(self, kind: str, nbytes: int) -> None:
        with self._lock:
            if nbytes > self._cur[kind]:
                raise RuntimeError(f"hold counter {kind} would go below zero")
            self._cur[kind] -= nbytes

    def end_step(self, step: int) -> dict[str, int]:
        with self._lock:
            self.steps[step] = dict(self._max)
            self._max = dict(self._cur)
            self._max["host"] = self._host()
            return self.steps[step]

    def line(self, step: int) -> str:
        m = self.steps[step]
        return f"WT_HOLD {step} " + " ".join(f"{k}={m[k]}" for k in (*HOLD_KINDS, "host"))
