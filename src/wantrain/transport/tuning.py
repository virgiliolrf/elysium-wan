"""Socket tuning for long, fat WAN paths (TRANSPORT-I10).

The pipeline hides a hop's latency with microbatches in flight; what it cannot hide is a hop that delivers too few
bytes per second. Over 130 ms one default TCP flow (CUBIC, kernel-autotuned buffers) moved 40-50 Mbps between
datacenters that each advertise Gbps (2026-10-12, Denver to Czechia): any loss halves CUBIC's window, which then
regrows one segment per round trip, and the autotuning ceiling (tcp_wmem/tcp_rmem max, 4-6 MB by default) caps a
flow at about ceiling / RTT. Two knobs move that, neither touching a byte of the payload:

- congestion control: BBR paces at the measured bottleneck rate instead of backing off on every loss. Set per socket
  (TCP_CONGESTION), so no root and no sysctl; applied only when the kernel offers it.
- buffers: SO_SNDBUF / SO_RCVBUF sized for the path's bandwidth-delay product (1 Gbps x 130 ms = 16 MB). Setting one
  turns the kernel's autotuning off for it, so a buffer is set only when the kernel allows more than autotuning
  would reach (net.core.[rw]mem_max above tcp_[rw]mem max); otherwise autotuning keeps it.

WANTRAIN_TCP_CC (default "bbr"; "off" keeps the system's) and WANTRAIN_TCP_BUF (bytes, default 32 MiB) override.
Call tune() on a socket before connect() or listen(): accepted sockets inherit the listener's settings.

TRANSPORT-I13, pacing: where BBR is not offered (a container gets the host's reno/cubic), a flow sends each window as
a line-rate burst. Thirty-two flows bursting one boundary at once overflow a shallow bottleneck queue: 0.35% of the
packets Virginia sent to Czechia were retransmitted (2026-10-13), nearly every 7.9 MB boundary lost some, and it
completed only when its slowest flow recovered (median 729 ms where the path's floor is ~120). SO_MAX_PACING_RATE
turns on the kernel's own TCP pacing (each window spread over the round trip, at a rate the congestion control
computes); its value is only a cap. Measured on that path, pacing on with a cap no flow reaches: median 132 ms, p90
220 ms. Unprivileged, per socket, no payload byte changes. WANTRAIN_TCP_PACE_MBPS (per flow, default 10000) caps it;
"off" leaves the kernel's default.
"""

from __future__ import annotations

import os
import socket
import struct

DEFAULT_CC = "bbr"
DEFAULT_BUF = 32 << 20
DEFAULT_PACE_MBPS = 10000.0  # per flow: a cap no WAN flow reaches, so pacing is on and only the kernel's rate counts
SO_MAX_PACING_RATE = getattr(
    socket, "SO_MAX_PACING_RATE", 47 if os.uname().sysname == "Linux" else None
)
_PROC = "/proc/sys/net"


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _int(path: str) -> int | None:
    v = _read(path)
    try:
        return int(v.split()[-1]) if v else None
    except ValueError:
        return None


def available_cc() -> set:
    v = _read(f"{_PROC}/ipv4/tcp_available_congestion_control")
    return set(v.split()) if v else set()


def plan(want_buf: int, caps: dict) -> dict:
    """Which buffers to set, from the kernel's limits (pure, for tests). caps: {"wmem_max", "rmem_max", "tcp_wmem",
    "tcp_rmem"} in bytes (None when unknown). -> {"sndbuf": bytes or None, "rcvbuf": bytes or None}; None keeps
    autotuning: set only when the hard cap admits `want_buf` and `want_buf` beats the autotuning ceiling."""
    out = {}
    for key, hard, auto in (("sndbuf", "wmem_max", "tcp_wmem"), ("rcvbuf", "rmem_max", "tcp_rmem")):
        cap, ceiling = caps.get(hard), caps.get(auto)
        out[key] = (
            want_buf
            if cap and cap >= want_buf and (ceiling is None or want_buf > ceiling)
            else None
        )
    return out


def pace(sock: socket.socket):
    """Turn on the kernel's TCP pacing for `sock` (TRANSPORT-I13). -> the cap in bytes/s, "off", or "unsupported"
    (no SO_MAX_PACING_RATE: not Linux, or refused). Never raises."""
    want = os.environ.get("WANTRAIN_TCP_PACE_MBPS", str(DEFAULT_PACE_MBPS))
    if want == "off":
        return "off"
    if SO_MAX_PACING_RATE is None:
        return "unsupported"
    try:
        rate = int(float(want) * 1e6 / 8)
        sock.setsockopt(socket.SOL_SOCKET, SO_MAX_PACING_RATE, struct.pack("Q", rate))
        return struct.unpack("Q", sock.getsockopt(socket.SOL_SOCKET, SO_MAX_PACING_RATE, 8))[0]
    except (OSError, ValueError, struct.error):
        return "unsupported"


def tune(sock: socket.socket) -> dict:
    """Apply congestion control, buffers and pacing to `sock` (before connect/listen). -> what holds after, for the
    run log: {"cc": name or "unsupported", "sndbuf": bytes or "auto", "rcvbuf": bytes or "auto", "pace": bytes/s or
    "off" or "unsupported" (pace())}. Never raises: a knob the kernel refuses keeps its default."""
    out: dict = {}
    want_cc = os.environ.get("WANTRAIN_TCP_CC", DEFAULT_CC)
    if not hasattr(socket, "TCP_CONGESTION"):
        out["cc"] = "unsupported"
    else:
        if want_cc and want_cc != "off" and want_cc in available_cc():
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, want_cc.encode())
            except OSError:
                pass
        try:
            raw = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, 16)
            out["cc"] = raw.split(b"\0", 1)[0].decode()
        except OSError:
            out["cc"] = "unsupported"
    want_buf = int(os.environ.get("WANTRAIN_TCP_BUF", DEFAULT_BUF))
    caps = {
        "wmem_max": _int(f"{_PROC}/core/wmem_max"),
        "rmem_max": _int(f"{_PROC}/core/rmem_max"),
        "tcp_wmem": _int(f"{_PROC}/ipv4/tcp_wmem"),
        "tcp_rmem": _int(f"{_PROC}/ipv4/tcp_rmem"),
    }
    for key, opt in (("sndbuf", socket.SO_SNDBUF), ("rcvbuf", socket.SO_RCVBUF)):
        size = plan(want_buf, caps)[key]
        if size is None:
            out[key] = "auto"
            continue
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, size)
            out[key] = sock.getsockopt(socket.SOL_SOCKET, opt)
        except OSError:
            out[key] = "auto"
    out["pace"] = pace(sock)
    return out


def connect(host: str, port: int, timeout: float) -> tuple[socket.socket, dict]:
    """socket.create_connection with tune() applied before the handshake (the window scale is fixed then).
    -> (connected socket, what tune applied)."""
    err: OSError | None = None
    for af, kind, proto, _name, addr in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        s = socket.socket(af, kind, proto)
        try:
            applied = tune(s)
            s.settimeout(timeout)
            s.connect(addr)
            return s, applied
        except OSError as e:
            err = e
            s.close()
    raise err or OSError(f"no address for {host}:{port}")
