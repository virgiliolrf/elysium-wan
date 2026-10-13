"""Per-step timing summary of one slot (RUNTIME-I9).

Pure on purpose: no torch, no clock, no RNG. It reads the monotonic t0/t1
stamps the stage runtime already recorded, so summarizing can neither move a
bit of training state nor draw from a random stream the reference also draws
from.
"""

from __future__ import annotations

import math

# Stage event name -> summary op.
OPS = {"fwd": "F", "bwd_b": "B", "bwd_w": "W", "step_opt": "OPT"}


def _median(xs) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2


def _union_len(intervals) -> float:
    """Length of the union of [a, b] intervals: parallel streams of one hop
    overlap, and their busy time is wall time, not the sum."""
    total, end = 0.0, -math.inf
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def summarize_step(events, step, rank, peer, step_ms, emit_us) -> dict:
    """events: the stage events taken since the previous summary. F/B/W/OPT
    and stall count only for `step`; every send counts, because senders run
    asynchronously and a frame may complete after its own step's summary."""
    durs = {op: [] for op in OPS.values()}
    stall_ms = 0.0
    sends: dict = {}
    recvs: dict = {}  # hop -> each whole boundary's first-to-last fragment spread, ms
    for e in events:
        name = e.get("ev")
        ms = (e["t1"] - e["t0"]) * 1e3 if name in OPS or name in ("stall", "send") else 0.0
        if name in OPS and e.get("step") == step:
            durs[OPS[name]].append(ms)
        elif name == "stall" and e.get("step") == step:
            stall_ms += ms
        elif name == "send" and "hop" in e:
            h = sends.setdefault(e["hop"], {"frames": 0, "bytes": 0, "iv": []})
            h["frames"] += 1
            h["bytes"] += int(e.get("nbytes", 0))
            h["iv"].append((e["t0"], e["t1"]))
        elif name == "recv" and "hop" in e:
            recvs.setdefault(e["hop"], []).append((e["t1"] - e["t0"]) * 1e3)
    return {
        "v": 1,
        "step": step,
        "rank": rank,
        "peer": peer,
        "step_ms": step_ms,
        "emit_us": emit_us,
        "ops": {op: {"n": len(d), "med_ms": _median(d)} for op, d in durs.items()},
        "stall_ms": stall_ms,
        "hops": {
            hop: {"frames": h["frames"], "bytes": h["bytes"], "busy_ms": _union_len(h["iv"]) * 1e3}
            for hop, h in sorted(sends.items())
        },
        "recv": {
            hop: {"n": len(d), "med_ms": _median(d), "max_ms": max(d)}
            for hop, d in sorted(recvs.items())
        },
    }


def parse_sim_slow(value):
    """WANTRAIN_SIM_SLOW="rank:peer:factor[:from_step]" -> (rank, peer,
    factor, from_step), or None when unset or empty. Timing-only test seam."""
    if value is None or value == "":
        return None
    parts = value.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"WANTRAIN_SIM_SLOW={value!r}: expected rank:peer:factor[:from_step]")
    try:
        rank, peer = int(parts[0]), int(parts[1])
        factor = float(parts[2])
        from_step = int(parts[3]) if len(parts) == 4 else 0
    except ValueError:
        raise ValueError(f"WANTRAIN_SIM_SLOW={value!r}: malformed number") from None
    if rank < 0 or peer < 0 or from_step < 0 or not math.isfinite(factor) or factor < 1.0:
        raise ValueError(
            f"WANTRAIN_SIM_SLOW={value!r}: rank, peer, from_step >= 0 and a finite factor >= 1"
        )
    return rank, peer, factor, from_step
