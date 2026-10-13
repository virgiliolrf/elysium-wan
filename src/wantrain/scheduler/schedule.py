"""Solve the schedule: how many microbatches, and how many in flight.

Scheduling is solved: microbatches, in-flight depth, microbatch size, layer
partition, boundary precision, data shards. None of these change what model
comes out, so the system owes the operator the best value it can compute.
Semantics are declared: seq_len, vocab, depth, width, learning rate, steps.
Changing any of them changes the model; the planner may report that a longer
sequence would lower the bandwidth floor, but never quietly delivers a
different model than was asked for. Both solvers return their reasoning.
"""

import math


def fbw_costs(d_model: int, seq_len: int) -> tuple:
    """Relative per-layer costs of the three passes: F, B (dL/dX), W (dL/dW).

    From the transformer FLOP counts in Qi et al. 2023 (zero bubble, Table 1):
    per layer and token, F costs sbh(24h+4s), B sbh(24h+8s) and W sbh(24h);
    only the ratios matter, so sbh is dropped. Note t_W < t_F < t_B and
    t_B + t_W = 2 t_F. The GPT 4h FFN is assumed; other MLP shapes move the
    ratios by a few percent, not enough to change a decision here.
    """
    h, s = d_model, seq_len
    return (24 * h + 4 * s, 24 * h + 8 * s, 24 * h)


def solve_microbatches(
    n_stages: int,
    bubble_target: float = 0.05,
    tokens_per_step_cap: int | None = None,
    mb_size: int = 1,
    seq_len: int = 1024,
    floor: int | None = None,
    fbw: tuple | None = None,
    comm_ratio: float = 0.0,
    cap: int = 1024,
) -> dict:
    """Microbatches per optimizer step, from the pipeline bubble you accept.

    The bubble is the one the zero-bubble schedule actually leaves: the
    smallest M whose solved schedule (scheduler/zero_bubble.py, ZB-2p memory)
    idles at most `bubble_target` of the pipeline, with per-microbatch costs
    in the ratio fbw = (t_F, t_B, t_W) and each hop's transfer comm_ratio x
    t_F. Memory does not bound M (the ordered accumulator is O(1) in M); the
    global batch does, so tokens_per_step_cap wins over the bubble target.
    """
    from wantrain.scheduler.zero_bubble import ZBConfig, zero_bubble

    if n_stages < 2:
        return dict(
            microbatches=max(1, floor or 1),
            bubble=0.0,
            family="none",
            reason="a single stage has no pipeline to fill",
        )
    tf, tb, tw = fbw or (1.0, 1.0, 1.0)
    unit = float(tf)
    tf, tb, tw = tf / unit, tb / unit, tw / unit
    comm = [comm_ratio] * (n_stages - 1)

    def bubble(m):
        cfg = ZBConfig(n_stages, m, [tf] * n_stages, [tb] * n_stages, [tw] * n_stages, comm=comm)
        per = abs(cfg.mem_f[0]) + max(0.0, cfg.mem_b[0])
        cfg.max_mem = [2 * n_stages * per] * n_stages
        return zero_bubble(cfg, "given").bubble_rate

    max_m = cap
    if tokens_per_step_cap:
        max_m = max(1, min(cap, tokens_per_step_cap // max(1, mb_size * seq_len)))
    lo = max(1, floor or 1)
    m, b = lo, bubble(lo)
    # bubble falls with M (the fill and drain amortise over more work): walk
    # up geometrically, then bisect to the smallest M meeting the target
    hi = lo
    while b > bubble_target and hi < max_m:
        hi = min(max_m, hi * 2)
        b = bubble(hi)
    if b <= bubble_target and hi > lo:
        left, right = max(lo, hi // 2), hi
        while left < right:
            mid = (left + right) // 2
            if bubble(mid) <= bubble_target:
                right = mid
            else:
                left = mid + 1
        hi = left
    m = hi
    b = bubble(m)
    reason = (
        f"{m} microbatches leave a {b:.1%} bubble in the zero-bubble schedule "
        f"across {n_stages} stages (target {bubble_target:.0%}, F:B:W = "
        f"1:{tb:.2f}:{tw:.2f}, hop transfer {comm_ratio:.2f} x F)"
    )
    if b > bubble_target and m == max_m and tokens_per_step_cap:
        reason = (
            f"{m} microbatches is the most that fits the {tokens_per_step_cap:,}-token "
            f"step budget at mb_size={mb_size}, seq_len={seq_len}; the bubble stays "
            f"at {b:.1%} rather than {bubble_target:.0%}, because a global batch past "
            f"the critical size costs convergence and a bubble only costs speed"
        )
    return dict(microbatches=m, bubble=b, family="zb", reason=reason)


def solve_fleet_ceiling(
    batch_tokens: float,
    mb_tokens: int,
    util_target: float = 0.5,
    fbw: tuple = None,
    w_fill_depth: int = None,
    n_stages: int = None,
    intra_split: int = 1,
    n_params: float = None,
    rtt_s: float = None,
    flops_per_gpu: float = None,
) -> dict:
    """The largest fleet one synchronous run can keep busy, from the batch.

    Batch side: a chain pays its fill/drain once per step, so utilisation is
    M_chain/(M_chain + span_eff * w), and the batch caps M_total because past
    the critical size (McCandlish et al. 2018) extra tokens per step stop
    buying learning. The topology cancels, leaving
        fleet_batch ~= intra_split * (S/span_eff) * (B / mb_tokens) * (1-u)/u
    where S/span_eff is the zero-bubble dividend (banked W fills the drain)
    and intra_split is the number of devices sharing one microbatch (tensor or
    sequence parallel inside a box).

    Latency side, when n_params, rtt_s and flops_per_gpu are given: an update
    costs 6 * N * B FLOPs and cannot finish faster than the fill/drain chain
    S * RTT, so fleet_latency = 6 N B / (F * S * RTT * u). The ceiling is the
    smaller of the two (analysis/SCALING_CEILING.md).
    """
    tf, tb, tw = fbw or fbw_costs(4096, 2048)
    unit = tf + tb + tw
    span_frac = (
        unit
        - 2.0
        * tw
        * (
            1.0
            if w_fill_depth is None or (n_stages or 2) <= 1 or w_fill_depth >= (n_stages or 2) - 1
            else w_fill_depth / ((n_stages or 2) - 1)
        )
    ) / unit
    zb_dividend = 1.0 / max(1e-9, span_frac)
    u = util_target
    fleet_batch = (
        max(1, intra_split) * zb_dividend * (batch_tokens / max(1, mb_tokens)) * (1 - u) / u
    )
    fleet_latency = None
    if n_params and rtt_s and flops_per_gpu:
        fleet_latency = (
            6.0 * n_params * batch_tokens / (flops_per_gpu * max(1, n_stages or 1) * rtt_s * u)
        )
    fleet = fleet_batch if fleet_latency is None else min(fleet_batch, fleet_latency)
    binding = "batch" if fleet_latency is None or fleet_batch <= fleet_latency else "latency"
    reason = (
        f"{int(fleet):,} GPUs is the most a {batch_tokens / 1e6:.0f}M-token "
        f"batch keeps {u:.0%} busy at {mb_tokens} tokens/microbatch"
        + (f" with {intra_split} devices per microbatch" if intra_split > 1 else "")
        + f": utilisation is M/(M+span) per chain, the batch caps total M, "
        f"and topology cancels out; the {zb_dividend:.1f}x factor is the "
        f"drain filled by banked W. Past this, rent buys idle silicon -- "
        f"grow the batch (loss progress raises the noise scale), shrink "
        f"the microbatch, split it across more devices, or train a second model"
    )
    if fleet_latency is not None:
        reason += (
            f". The latency chain S*RTT alone allows {int(fleet_latency):,} GPUs; "
            f"the {binding} side binds"
        )
    return dict(
        fleet_max=int(fleet),
        fleet_batch_side=int(fleet_batch),
        fleet_latency_side=None if fleet_latency is None else int(fleet_latency),
        binding=binding,
        zb_dividend=round(zb_dividend, 2),
        reason=reason,
    )


def stage_live_graphs(n_stages: int, rank: int, k_inflight: int) -> int:
    """Concurrent live activation graphs a rank holds, from lifespans.

    Qi et al. 2024 (controllable memory): peak = ceil(lifespan / interval) per
    stage. A microbatch's graph is born at its F and dies at its B, so on rank
    r the lifespan spans the walk to the last stage and back, (n_stages - rank)
    intervals, and the first stage holds the most. On a WAN the k-1 extra
    microbatches admitted to hide the round trip each add one interval. This
    is why rank 0 is the memory-hungriest slot.
    """
    return max(1, n_stages - rank) + max(0, (k_inflight or 1) - 1)


def solve_k_inflight(
    window_s: float,
    rtt_ms: float,
    boundary_bytes: float,
    link_mbps: float,
    mem_slack_bytes: float = None,
    per_inflight_bytes: float = None,
    cap: int = 16,
) -> dict:
    """How many microbatches to keep in flight, from the measured latency.

    Bandwidth is hidden by the compute window carrying the transfer (the
    admission floor, computed at k=1); latency is what k hides: while
    microbatch i's boundary is in the air the stage works on i+1 .. i+k, so
        k >= (rtt/2 + boundary_bytes / link) / window_s + 1
    Above that each extra in-flight microbatch pins a boundary buffer and a
    live graph for nothing, so k is also capped by the memory left over.
    """
    one_way_s = max(0.0, rtt_ms) / 2000.0
    xfer_s = (boundary_bytes * 8) / (max(1e-9, link_mbps) * 1e6)
    need_s = one_way_s + xfer_s
    if window_s <= 0:
        return dict(k=1, reason="no compute window estimate; keeping k=1")

    k = max(1, math.ceil(need_s / window_s) + 1)
    reason = (
        f"k={k} covers {need_s * 1000:.0f} ms of latency "
        f"({one_way_s * 1000:.0f} ms one-way RTT + {xfer_s * 1000:.0f} ms "
        f"transfer) with a {window_s * 1000:.0f} ms compute window"
    )

    if k > cap:
        k = cap
        reason = (
            f"k capped at {cap}; the link needs {need_s * 1000:.0f} ms "
            f"covered by a {window_s * 1000:.0f} ms window, so the stage "
            f"will still stall — this fleet is latency-bound"
        )

    if mem_slack_bytes is not None and per_inflight_bytes:
        fits = max(1, int(mem_slack_bytes // per_inflight_bytes))
        if fits < k:
            reason = (
                f"k={fits} is all the memory allows ({per_inflight_bytes / 2**20:.0f} "
                f"MB per in-flight microbatch against "
                f"{mem_slack_bytes / 2**30:.1f} GB spare); latency wanted "
                f"k={k}, so expect residual stall"
            )
            k = fits

    return dict(k=k, reason=reason, latency_s=need_s, window_s=window_s)
