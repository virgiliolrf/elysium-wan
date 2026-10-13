"""Partitioning and chain ordering: the static half of the scheduler.

A pipeline is a chain, and which machine holds which rank is a free variable.
The WAN makes it the variable that matters: rank r ships boundary tensors to
rank r+1 only, so the chain should route around slow or dead pairs.

order_stages          stage order as a bottleneck-path problem over a measured
                      link matrix: maximise the minimum bandwidth over adjacent
                      hops, then minimise total RTT (lexicographic). Exact for
                      chains up to EXHAUSTIVE_MAX nodes, greedy beyond.
best_chain            the same max-min Hamiltonian path from an edge list,
                      used when only one-direction probes exist.
assign_layers         contiguous layer -> stage split for simultaneous finish:
                      exact DP on the min-max stage time, with optional
                      per-stage floors (bandwidth) and ceilings (memory).
proportional_partition  the DP applied to measured node speeds, counting the
                      embedding and the head as layers.
"""

from __future__ import annotations

import math

EXHAUSTIVE_MAX = 8


def _chain_key(order, bw, rtt):
    hops = list(zip(order, order[1:]))
    if not hops:
        return (math.inf, 0.0)
    min_bw = min(bw[a][b] for a, b in hops)
    tot_rtt = sum(rtt[a][b] for a, b in hops) if rtt is not None else 0.0
    return (min_bw, -tot_rtt)


def order_stages(rtt_ms, bw_mbps):
    """Return the worker order (list of indices) for the chain.

    Exhaustive over orderings with branch-and-bound up to EXHAUSTIVE_MAX nodes
    (a chain and its reverse carry the same hops, so one direction is
    searched); greedy best-next-hop from every start node beyond that.
    """
    n = len(bw_mbps)
    if n <= 1:
        return list(range(n))
    if n > EXHAUSTIVE_MAX:
        return _greedy_chain(n, bw_mbps, rtt_ms)

    best, best_key = list(range(n)), _chain_key(list(range(n)), bw_mbps, rtt_ms)
    cur: list[int] = []
    used = [False] * n

    def rec():
        nonlocal best, best_key
        if len(cur) == n:
            if cur[0] > cur[-1]:
                return
            key = _chain_key(cur, bw_mbps, rtt_ms)
            if key > best_key:
                best, best_key = list(cur), key
            return
        for k in range(n):
            if used[k]:
                continue
            # A partial path whose newest hop is already below the best
            # bottleneck cannot beat it.
            if cur and bw_mbps[cur[-1]][k] < best_key[0]:
                continue
            used[k] = True
            cur.append(k)
            rec()
            cur.pop()
            used[k] = False

    rec()
    return best


def _greedy_chain(n, bw, rtt):
    best, best_key = None, None
    for start in range(n):
        order = [start]
        used = [False] * n
        used[start] = True
        for _ in range(1, n):
            last = order[-1]
            pick = max(
                (k for k in range(n) if not used[k]),
                key=lambda k: (bw[last][k], -(rtt[last][k] if rtt else 0.0)),
            )
            used[pick] = True
            order.append(pick)
        key = _chain_key(order, bw, rtt)
        if best_key is None or key > best_key:
            best, best_key = order, key
    return best


def best_chain(n: int, edges) -> tuple[list[int], float]:
    """Max-min Hamiltonian path from undirected edges [(i, j, mbps), ...].

    Unprobed or unreachable pairs weigh 0, so they are routed around when any
    ordering avoids them. Returns (order, bottleneck_mbps); a bottleneck of 0
    means no ordering connects every node.
    """
    w = [[0.0] * n for _ in range(n)]
    for i, j, mbps in edges:
        w[i][j] = w[j][i] = float(mbps)
    order = order_stages(None, w)
    key = _chain_key(order, w, None)
    return order, (key[0] if n > 1 else math.inf)


def assign_layers(n_layers, speeds, floors=None, ceilings=None, layer_costs=None, fits=None):
    """Contiguous partition of n_layers across len(speeds) stages minimizing
    the slowest stage, sum(layer_costs[a:b]) / speeds[s], subject to optional
    per-stage floors/ceilings (layer counts) and an optional feasibility
    predicate fits(stage, a, b) (e.g. "this range fits this node's memory").
    Exact O(n^2 * S) DP; n is the layer count, small. layer_costs defaults to
    1 per layer, which is the source's uniform model."""
    S = len(speeds)
    if any(sp <= 0 for sp in speeds):
        raise ValueError(f"every stage needs a positive speed, got {speeds}")
    floors = floors or [1] * S
    ceilings = ceilings or [n_layers] * S
    costs = list(layer_costs) if layer_costs is not None else [1.0] * n_layers
    if len(costs) != n_layers:
        raise ValueError(f"layer_costs has {len(costs)} entries for {n_layers} layers")
    prefix = [0.0]
    for c in costs:
        prefix.append(prefix[-1] + float(c))
    INF = float("inf")
    # dp[s][i] = minimal max-time assigning first i layers to first s stages
    dp = [[INF] * (n_layers + 1) for _ in range(S + 1)]
    cut = [[0] * (n_layers + 1) for _ in range(S + 1)]
    dp[0][0] = 0.0
    for s in range(1, S + 1):
        lo, hi = floors[s - 1], ceilings[s - 1]
        for i in range(1, n_layers + 1):
            for take in range(lo, min(hi, i) + 1):
                prev = dp[s - 1][i - take]
                if prev == INF:
                    continue
                if fits is not None and not fits(s - 1, i - take, i):
                    continue
                t = max(prev, (prefix[i] - prefix[i - take]) / speeds[s - 1])
                if t < dp[s][i]:
                    dp[s][i] = t
                    cut[s][i] = take
    if dp[S][n_layers] == INF:
        raise ValueError(
            "no admissible partition: the floors, ceilings or memory limits "
            "leave no way to cut the model across these stages"
        )
    bounds, i = [], n_layers
    for s in range(S, 0, -1):
        take = cut[s][i]
        bounds.append([i - take, i])
        i -= take
    bounds.reverse()
    return bounds, dp[S][n_layers]


def layer_costs(model_cfg) -> list:
    """Relative compute of each entry of the flat layer list [Embed, Block*n,
    Head], forward + backward per token. A block costs h(72h + 12s) (F, B and
    W from the zero-bubble paper's table: h(24h+4s) + h(24h+8s) + 24h^2);
    the head's logits matmul costs 6hV, which at a 32k-50k vocabulary is
    several blocks; the embedding is a lookup (~3h). Counting the head as one
    block, as a uniform model does, overloads the last stage.

    (M4) An arch "hf" model is costed from the builder's census
    (autoconfig.hf_unit_costs: 6·macs + 12·H·d·S' per unit); an MoE block
    costs 6·P^F + 12·h·s, P^F its attention, router, k experts and shared
    expert (SCHEDULER-I21)."""
    from wantrain.scheduler import autoconfig

    h, s = model_cfg.d_model, model_cfg.seq_len
    if autoconfig.is_hf(model_cfg):
        census = autoconfig.hf_census(model_cfg)
        units = autoconfig.hf_unit_costs(
            census, s, 1, model_cfg.compute_dtype, top_k=autoconfig.hf_top_k(model_cfg)
        )
        return [float(u["cost"]) for u in units]
    v = model_cfg.vocab_size
    block = h * (72 * h + 12 * s)
    moe = autoconfig.moe_fields(model_cfg)
    blocks = [float(block)] * model_cfg.n_layer
    if moe is not None:
        active = (
            autoconfig.builtin_attn_params(model_cfg)
            + autoconfig.moe_layer_costs(moe, h, s, 1)["params_active"]
        )
        for j in range(model_cfg.n_layer):
            if autoconfig.is_moe_block(model_cfg, j):
                blocks[j] = float(6 * active + 12 * h * s)
    return [3.0 * h] + blocks + [6.0 * h * v]


def proportional_partition(model_cfg, speeds, floors=None, ceilings=None, costs=None, fits=None):
    """Contiguous layer bounds for measured stage speeds via the exact
    min-max DP over per-layer costs. Layer count = n_layer + 2 (Embed and
    Head are list entries, priced by `layer_costs` unless `costs` is given).

    Speeds come from a short matmul burn on the actual device, not spec
    sheets: thermals, power caps and host contention all move real
    throughput, and measurements go stale, which is why the coordinator
    re-profiles F/B/W on each node and refines this cut before launch."""
    n_layers = model_cfg.n_layer + 2
    c = costs if costs is not None else layer_costs(model_cfg)
    bounds, _t_max = assign_layers(n_layers, list(speeds), floors, ceilings, c, fits)
    return bounds
