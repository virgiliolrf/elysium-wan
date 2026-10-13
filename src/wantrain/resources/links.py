"""Network resource discovery: pairwise link measurement between workers.

Each worker runs a small echo server. A probe from worker i to worker j
measures round-trip time (median of small round trips) and bulk bandwidth in
the forward direction; the full measurement of a chosen hop adds the reverse
direction and a simultaneous bidirectional transfer, from which the duplex
model is classified (does forward traffic compete with backward traffic on
this path?). The coordinator collects the matrix and feeds the chain ordering
and the planner; values never touch numerics.

WAN emulation (for experiments on hardware without a real WAN, as the source
did with its transport shim, since macOS has no tc/netem): a worker may carry
an emulation profile mapping (its region, peer region) to a rate and RTT. The
measuring side then paces its sends, delays its round trips and throttles its
reads to that profile, so the numbers the scheduler sees come from real
transfers shaped like the declared WAN.
"""

from __future__ import annotations

import os
import socket
import statistics
import struct
import threading
import time
from dataclasses import dataclass

from wantrain.transport import tuning

PROBE_BULK_BYTES = 4 * 1024 * 1024
_OP = struct.Struct("<BQ")
OP_UPLOAD, OP_DOWNLOAD = 0, 1
_ACK = b"\x06" * 8
_CHUNK = 64 * 1024


@dataclass
class LinkProfile:
    mbps: float = 0.0  # 0 = not shaped
    rtt_ms: float = 0.0


class Emulation:
    """Region-pair link model: {"intra": {...}, "inter": {...},
    "pairs": {"A|B": {...}}} with {"mbps": float, "rtt_ms": float} entries."""

    def __init__(self, region: str | None, profile: dict | None):
        self.region = region
        self.profile = profile or {}

    @property
    def enabled(self) -> bool:
        return bool(self.profile)

    def link_to(self, other_region: str | None) -> LinkProfile:
        if not self.profile:
            return LinkProfile()
        a, b = self.region or "", other_region or ""
        pairs = self.profile.get("pairs") or {}
        rec = pairs.get("|".join(sorted((a, b))))
        if rec is None:
            rec = self.profile.get("intra" if a == b else "inter") or {}
        return LinkProfile(float(rec.get("mbps", 0.0)), float(rec.get("rtt_ms", 0.0)))

    def to_dict(self) -> dict:
        return {"region": self.region, "profile": self.profile}


def _pace(nbytes: int, t0: float, mbps: float) -> None:
    if mbps > 0:
        due = t0 + nbytes * 8 / (mbps * 1e6)
        now = time.perf_counter()
        if due > now:
            time.sleep(due - now)


def serve_echo(
    port: int,
    stop: threading.Event,
    # peers on other machines probe this
    host: str = "0.0.0.0",  # nosec B104
) -> int:
    """Start the probe server; returns the bound port (port=0 picks one)."""
    lst = socket.socket()
    lst.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tuning.tune(lst)  # TRANSPORT-I10: the probe measures the path as the data plane will use it
    lst.bind((host, port))
    lst.listen(128)  # p2p_streams parallel probe flows connect at once (TRANSPORT-I11)
    lst.settimeout(0.5)
    bound = lst.getsockname()[1]

    def _conn(conn):
        try:
            while True:
                hdr = conn.recv(_OP.size, socket.MSG_WAITALL)
                if len(hdr) < _OP.size:
                    break
                op, n = _OP.unpack(hdr)
                if n > 1 << 32:
                    break
                if op == OP_UPLOAD:
                    got = 0
                    while got < n:
                        chunk = conn.recv(min(1 << 20, n - got))
                        if not chunk:
                            return
                        got += len(chunk)
                    conn.sendall(_ACK)
                elif op == OP_DOWNLOAD:
                    blob = b"\0" * min(n, 1 << 20)
                    left = n
                    while left:
                        part = blob[: min(left, len(blob))]
                        conn.sendall(part)
                        left -= len(part)
                else:
                    break
        except OSError:
            pass
        finally:
            conn.close()

    def _serve():
        while not stop.is_set():
            try:
                conn, _ = lst.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            tuning.pace(conn)  # TRANSPORT-I13: measured as the data plane will send
            threading.Thread(target=_conn, args=(conn,), daemon=True).start()
        lst.close()

    threading.Thread(target=_serve, daemon=True).start()
    return bound


def _upload(c: socket.socket, n: int, link: LinkProfile) -> float:
    t0 = time.perf_counter()
    c.sendall(_OP.pack(OP_UPLOAD, n))
    sent = 0
    blob = b"\0" * _CHUNK
    while sent < n:
        part = blob[: min(_CHUNK, n - sent)]
        c.sendall(part)
        sent += len(part)
        _pace(sent, t0, link.mbps)
    c.recv(8, socket.MSG_WAITALL)
    if link.rtt_ms:
        time.sleep(link.rtt_ms / 2000.0)  # the ack's one-way trip
    return time.perf_counter() - t0


def _download(c: socket.socket, n: int, link: LinkProfile) -> float:
    t0 = time.perf_counter()
    c.sendall(_OP.pack(OP_DOWNLOAD, n))
    if link.rtt_ms:
        time.sleep(link.rtt_ms / 2000.0)  # the request's one-way trip
    got = 0
    while got < n:
        chunk = c.recv(min(_CHUNK, n - got))
        if not chunk:
            raise ConnectionError("probe server closed mid-download")
        got += len(chunk)
        _pace(got, t0, link.mbps)
    return time.perf_counter() - t0


def _connect(host: str, port: int) -> socket.socket:
    c, _applied = tuning.connect(host, port, timeout=10)  # TRANSPORT-I10
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    c.settimeout(60)
    return c


def _rtt_ms(c: socket.socket, link: LinkProfile, rounds: int = 5) -> float:
    rtts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        c.sendall(_OP.pack(OP_UPLOAD, 1) + b"x")
        c.recv(8, socket.MSG_WAITALL)
        if link.rtt_ms:
            time.sleep(link.rtt_ms / 1000.0)
        rtts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(rtts)


TIMED_S, TIMED_WARM_S = (
    4.0,
    1.0,
)  # TRANSPORT-I11: a multi-flow measurement's length and its discarded ramp
_ENDLESS = (
    1 << 32
) - 1  # "until the client closes": the probe server accepts at most this much per op


def _timed(
    host: str, port: int, n: int, upload: bool, link: LinkProfile, secs: float, warm: float
) -> float:
    """Aggregate Mbps of `n` flows sending (upload) or receiving for `secs`, counting only after `warm`: the
    rate a hop holds under the data plane's continuous load. A fixed bulk per flow measures TCP's ramp instead:
    16 flows of 8 MB across 141 ms read 131 Mbps where the same path held 745-1,032 continuously (2026-10-13)."""
    each = LinkProfile(link.mbps / n if link.mbps else 0.0, link.rtt_ms)
    conns = [_connect(host, port) for _ in range(n)]
    counted = [0] * n
    t0 = time.perf_counter()
    start, stop = t0 + warm, t0 + secs

    def run(i, c):
        blob = b"\0" * _CHUNK
        moved = 0
        try:
            c.sendall(_OP.pack(OP_UPLOAD if upload else OP_DOWNLOAD, _ENDLESS))
            while (now := time.perf_counter()) < stop:
                if upload:
                    c.sendall(blob)
                    got = len(blob)
                else:
                    got = len(c.recv(_CHUNK))
                    if not got:
                        break
                moved += got
                if now >= start:
                    counted[i] += got
                _pace(moved, t0, each.mbps)
        except OSError:
            pass

    ts = [threading.Thread(target=run, args=(i, c), daemon=True) for i, c in enumerate(conns)]
    try:
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        for c in conns:
            c.close()
    return sum(counted) * 8 / max(1e-9, secs - warm) / 1e6


def _streamed(host: str, port: int, n: int, op, nbytes: int, link: LinkProfile) -> float:
    """Aggregate Mbps of `op` (_upload or _download) over `n` parallel connections, `nbytes` each: the rate the
    data plane gets with p2p_streams flows (TRANSPORT-I10). A shaped (emulated) link's rate is shared by the n
    flows, so the aggregate still models the link."""
    if n <= 1:
        c = _connect(host, port)
        try:
            return nbytes * 8 / op(c, nbytes, link) / 1e6
        finally:
            c.close()
    if os.environ.get("WANTRAIN_PROBE_TIMED", "on") != "off":
        return _timed(host, port, n, op is _upload, link, TIMED_S, TIMED_WARM_S)
    each = LinkProfile(link.mbps / n if link.mbps else 0.0, link.rtt_ms)
    conns = [_connect(host, port) for _ in range(n)]
    errs: list = []

    def run(c):
        try:
            op(c, nbytes, each)
        except OSError as e:
            errs.append(e)

    try:
        t0 = time.perf_counter()
        ts = [threading.Thread(target=run, args=(c,), daemon=True) for c in conns]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        dt = time.perf_counter() - t0
    finally:
        for c in conns:
            c.close()
    if errs:
        raise errs[0]
    return n * nbytes * 8 / dt / 1e6


def measure(
    host: str,
    port: int,
    link: LinkProfile | None = None,
    bulk_bytes: int = PROBE_BULK_BYTES,
    streams: int = 1,
) -> dict:
    """Quick probe: RTT and forward bandwidth over `streams` parallel flows (bulk_bytes each). Unreachable -> inf /
    0."""
    link = link or LinkProfile()
    try:
        c = _connect(host, port)
        try:
            rtt = _rtt_ms(c, link)
        finally:
            c.close()
        return {"rtt_ms": rtt, "bw_mbps": _streamed(host, port, streams, _upload, bulk_bytes, link)}
    except OSError:
        return {"rtt_ms": float("inf"), "bw_mbps": 0.0}


def measure_row(
    targets,
    self_index: int,
    emu: Emulation | None = None,
    bulk_bytes: int = PROBE_BULK_BYTES,
    streams: int = 1,
) -> dict:
    """One row of the matrix. targets: [[host, port, region?], ...]."""
    rtt, bw = [], []
    for i, t in enumerate(targets):
        if i == self_index:
            rtt.append(0.0)
            bw.append(float("inf"))
            continue
        host, port = t[0], int(t[1])
        region = t[2] if len(t) > 2 else None
        link = emu.link_to(region) if emu else LinkProfile()
        m = measure(host, port, link, bulk_bytes, streams)
        rtt.append(m["rtt_ms"])
        bw.append(m["bw_mbps"])
    return {"rtt_ms": rtt, "bw_mbps": bw}


def classify_duplex(forward: float, reverse: float, bidir_fwd: float, bidir_rev: float):
    """-> (duplex, note). per_direction: the simultaneous total exceeds the
    best one-way rate, which an aggregate cap cannot produce; aggregate: the
    two directions share one cap; unknown: the bidirectional test is missing
    or near zero on a path that just moved real traffic (a broken test, not a
    slow link, so it must not halve the budget)."""
    total = bidir_fwd + bidir_rev
    best = max(forward, reverse)
    if best > 0 and total < best * 0.05:
        return "unknown", (
            f"bidirectional test returned ~{total:.1f} Mbps against a one-way "
            f"{best:.1f} Mbps: a broken test, not a measurement"
        )
    if total > 0 and best > 0:
        if total > best * 1.15:
            return "per_direction", (
                f"simultaneous total {total:.1f} Mbps exceeds the best one-way rate {best:.1f} Mbps"
            )
        return "aggregate", (
            f"simultaneous total {total:.1f} Mbps does not exceed the best "
            f"one-way rate {best:.1f} Mbps: the directions share one cap"
        )
    return "unknown", "the bidirectional test did not return a usable result"


def measure_pair(
    host: str,
    port: int,
    link: LinkProfile | None = None,
    bulk_bytes: int = PROBE_BULK_BYTES,
    streams: int = 1,
) -> dict:
    """Full hop measurement the planner budgets against, over `streams` parallel flows (bulk_bytes each; the data
    plane's p2p_streams, TRANSPORT-I10): one flow across 100-160 ms measured 38-118 Mbps where 32 flows held
    600-1,100 (2026-10-12), so a one-flow figure priced bubbles that were not there. Raises ConnectionError when
    the path carries no traffic: a published port with nothing behind it would otherwise bind, connect and hang
    the pipeline without an error."""
    link = link or LinkProfile()
    try:
        c = _connect(host, port)
    except OSError as e:
        raise ConnectionError(f"{host}:{port} does not carry traffic ({e})") from e
    try:
        rtt = _rtt_ms(c, link)
    finally:
        c.close()
    fwd = _streamed(host, port, streams, _upload, bulk_bytes, link)
    rev = _streamed(host, port, streams, _download, bulk_bytes, link)

    res = {}

    def _up():
        res["up"] = _streamed(host, port, streams, _upload, bulk_bytes, link)

    def _down():
        res["down"] = _streamed(host, port, streams, _download, bulk_bytes, link)

    ts = [threading.Thread(target=_up), threading.Thread(target=_down)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    bf, br = res.get("up", 0.0), res.get("down", 0.0)
    duplex, note = classify_duplex(fwd, rev, bf, br)
    # The budget is per direction, and the forward direction is what boundary
    # activations get; the planner applies the duplex haircut.
    return {
        "mbps": round(min(fwd, rev), 2),
        "forward_mbps": round(fwd, 2),
        "reverse_mbps": round(rev, 2),
        "bidir_forward_mbps": round(bf, 2),
        "bidir_reverse_mbps": round(br, 2),
        "rtt_ms": round(rtt, 2),
        "duplex": duplex,
        "note": note,
    }
