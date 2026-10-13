"""Zero-bubble pipeline schedules (Qi et al., "Zero Bubble Pipeline
Parallelism", ICLR 2024, arXiv:2401.10241).

The backward pass is split into B (gradient w.r.t. the stage input, on the
critical path) and W (gradient w.r.t. the weights, off it). Because W has no
cross-stage dependency it can be placed anywhere after its B, which is what
fills the pipeline bubbles 1F1B leaves.

This module is a port of the paper's automatic scheduling heuristic as
implemented in sail-sg/zero-bubble-pipeline-parallelism
(megatron/core/pipeline_parallel/zerobubble/scheduler/zb.py, `manual_order` +
`initial_solution`): warm up with as many F passes as memory allows until the
first B can arrive, then alternate F and B, insert W into a gap only where it
fits before the next input arrives (or where the stage's bubble is already
below the worst stage's), recycle W under memory pressure, keep stage i at
least one F ahead of stage i+1, flush the remaining W at the tail; grid search
over the three binary choices and keep the shortest makespan.

Two generalisations for heterogeneous hardware over a WAN, both strict
supersets of the reference (equal inputs give the reference's schedule):
  - per-stage costs: cost_f/b/w[stage] come from profiling each stage on its
    own device (the reference profiles per stage but then takes the median);
  - per-hop communication: comm[j] is the boundary transfer time between
    stage j and j+1 (one-way latency + transfer at the measured bandwidth);
    the reference has one cost_comm for every hop.

Memory follows the reference's convention: F allocates mem_f, B releases part
of it (mem_b), W releases the rest (mem_w), mem_f + mem_b + mem_w == 0; a
stage never exceeds max_mem. ZB-1p keeps the 1F1B activation budget
(p x mem_f), ZB-2p doubles it.
"""

from __future__ import annotations

import copy
import math
import statistics
from dataclasses import asdict, dataclass, field

F, B, W = "F", "B", "W"


@dataclass
class ZBConfig:
    n_stages: int
    n_micro: int
    cost_f: list
    cost_b: list
    cost_w: list
    comm: list = field(default_factory=list)  # per hop j -> j+1; len n_stages - 1
    mem_f: list = field(default_factory=list)
    mem_b: list = field(default_factory=list)
    mem_w: list = field(default_factory=list)
    max_mem: list | None = None

    def __post_init__(self):
        p = self.n_stages
        for name in ("cost_f", "cost_b", "cost_w"):
            v = getattr(self, name)
            if len(v) != p:
                raise ValueError(f"{name} needs one entry per stage ({p}), got {len(v)}")
        if isinstance(self.comm, (int, float)):
            self.comm = [float(self.comm)] * max(0, p - 1)
        elif not self.comm:
            self.comm = [0.0] * max(0, p - 1)
        if len(self.comm) != max(0, p - 1):
            raise ValueError(f"comm needs one entry per hop ({p - 1}), got {len(self.comm)}")
        if not self.mem_f:
            self.mem_f = [1.0] * p
            self.mem_b = [-0.5] * p
            self.mem_w = [-0.5] * p
        for f_, b_, w_ in zip(self.mem_f, self.mem_b, self.mem_w):
            if abs(f_ + b_ + w_) > 1e-9:
                raise ValueError("mem_f + mem_b + mem_w must be 0 (F allocates what B and W free)")

    def cost(self, typ: str, stage: int) -> float:
        if typ == F:
            return self.cost_f[stage]
        if typ == B:
            return self.cost_b[stage]
        return self.cost_w[stage]

    def work(self, stage: int) -> float:
        return self.n_micro * (self.cost_f[stage] + self.cost_b[stage] + self.cost_w[stage])


@dataclass
class Schedule:
    kind: str
    orders: list  # per stage: [(type, mb), ...]
    makespan: float
    bubble_rate: float
    params: dict = field(default_factory=dict)

    def as_spec_order(self) -> list:
        return [[[t, mb] for t, mb in order] for order in self.orders]


def bubble_rate(cfg: ZBConfig, makespan: float) -> float:
    """1 - useful work / (stages x makespan): the idle fraction of the pipeline."""
    if makespan <= 0:
        return 0.0
    useful = sum(cfg.work(s) for s in range(cfg.n_stages))
    return max(0.0, 1.0 - useful / (cfg.n_stages * makespan))


# --------------------------------------------------------------------- simulate


def simulate(cfg: ZBConfig, orders) -> tuple[dict, float]:
    """Execute per-stage op orders under the dependency graph and per-hop
    communication delays. -> ({(type, stage, mb): end_time}, makespan).

    F(s, m) needs F(s-1, m) + comm[s-1]; B(s, m) needs B(s+1, m) + comm[s]
    (the last stage's B needs its own F); W(s, m) needs B(s, m). Each stage
    runs its order sequentially. Raises on an order that deadlocks."""
    return simulate_slots(
        cfg,
        [[o] for o in orders],
        [[0] * cfg.n_micro for _ in range(cfg.n_stages)],
        [[1.0] for _ in range(cfg.n_stages)],
    )


def simulate_slots(cfg: ZBConfig, slot_orders, mb_owner, peer_weights) -> tuple[dict, float]:
    """The same for stages of several peers: slot_orders[stage][peer] is that
    peer's order and a peer of weight w runs an op in (stage cost x total
    weight / w): the stage's aggregate cost scaled to that peer's share of the
    stage's throughput. mb_owner is accepted for symmetry with the runtime;
    ownership is already encoded in which order holds which microbatch."""
    p = cfg.n_stages
    end: dict = {}
    slots = [(s, q) for s in range(p) for q in range(len(slot_orders[s]))]
    pos = dict.fromkeys(slots, 0)
    free = dict.fromkeys(slots, 0.0)
    first: dict = {}
    remaining = sum(len(slot_orders[s][q]) for s, q in slots)

    def dep_time(typ, s, mb):
        if typ == F:
            if s == 0:
                return 0.0
            k = (F, s - 1, mb)
            return end[k] + cfg.comm[s - 1] if k in end else None
        if typ == B:
            if s == p - 1:
                return end.get((F, s, mb))
            k = (B, s + 1, mb)
            return end[k] + cfg.comm[s] if k in end else None
        return end.get((B, s, mb))

    while remaining:
        progressed = False
        for s, q in slots:
            order = slot_orders[s][q]
            scale = sum(peer_weights[s]) / peer_weights[s][q]
            while pos[(s, q)] < len(order):
                typ, mb = order[pos[(s, q)]]
                ready = dep_time(typ, s, mb)
                if ready is None:
                    break
                start = max(free[(s, q)], ready)
                first.setdefault((s, q), start)
                fin = start + cfg.cost(typ, s) * scale
                end[(typ, s, mb)] = fin
                free[(s, q)] = fin
                pos[(s, q)] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("schedule deadlocks: an op waits on one that is ordered after it")
    makespan = max(free[k] - first.get(k, 0.0) for k in slots)
    return end, makespan


# --------------------------------------------------------------------- heuristic


def _manual_order(
    cfg: ZBConfig, allow_bubble_before_first_b: bool, prioritize_b: bool, no_bubble_greedy: bool
):
    """Port of Graph.manual_order (sail-sg zb.py) with per-stage costs and
    per-hop communication. Returns (orders, completion-time dict, makespan)."""
    p, nmb = cfg.n_stages, cfg.n_micro
    f = [0] * p
    b = [0] * p
    w = [0] * p
    m = [0.0] * p
    e = [0.0] * p
    t: dict = {}
    orders: list = [[] for _ in range(p)]
    max_mem = cfg.max_mem or [cfg.mem_f[s] * nmb * 3 for s in range(p)]
    stage_bubble = [0.0] * p
    eps = 1e-9

    def comm_in_f(j):  # F(j) waits for F(j-1) across hop j-1
        return cfg.comm[j - 1]

    def comm_in_b(j):  # B(j) waits for B(j+1) across hop j
        return cfg.comm[j]

    def mem_of(typ, j):
        return {F: cfg.mem_f, B: cfg.mem_b, W: cfg.mem_w}[typ][j]

    def get_max_bubble():
        return max(stage_bubble) if stage_bubble else 0.0

    def put(j, typ):
        i = {F: f, B: b, W: w}[typ][j]
        cost = cfg.cost(typ, j)
        tmp = e[j] + cost
        no_bubble = tmp
        if j > 0 and typ == F:
            tmp = max(tmp, t[(F, j - 1, i)] + comm_in_f(j) + cost)
        if j < p - 1 and typ == B:
            tmp = max(tmp, t[(B, j + 1, i)] + comm_in_b(j) + cost)
        if f[j] > 0:
            stage_bubble[j] += tmp - no_bubble
        e[j] = tmp
        t[(typ, j, i)] = tmp
        m[j] += mem_of(typ, j)
        orders[j].append((typ, i))
        if typ == F:
            f[j] += 1
        elif typ == B:
            b[j] += 1
        else:
            w[j] += 1

    for i in range(nmb):
        if i == 0:
            for j in range(p):
                put(j, F)
            f_required = [0] * p
            last_t = 0.0
            for j in range(p - 1, -1, -1):
                if j == p - 1:
                    last_t = t[(F, j, i)] + cfg.cost_b[j]
                    continue
                mem = m[j]
                cost = e[j]
                while True:
                    nxt = f[j] + f_required[j]
                    if nxt < nmb and mem + cfg.mem_f[j] <= max_mem[j] + eps:
                        if allow_bubble_before_first_b:
                            if cost + cfg.cost_f[j] > last_t + comm_in_b(j):
                                break
                        elif cost >= last_t + comm_in_b(j):
                            break
                        mem += cfg.mem_f[j]
                        cost += cfg.cost_f[j]
                        f_required[j] += 1
                    else:
                        break
                last_t = max(cost, last_t + comm_in_b(j)) + cfg.cost_b[j]
            for j in range(p):
                while (
                    j > 0
                    and f_required[j] > 0
                    and f_required[j] >= f_required[j - 1]
                    and f[j] + f_required[j] < nmb
                ):
                    f_required[j] -= 1
            # Heterogeneous stages: a downstream stage with cheap F can want more
            # warm-up F than its upstream neighbour has produced. Never let a
            # stage run ahead of the one feeding it (inactive when costs are
            # uniform, which is all the reference implementation handles).
            for j in range(1, p):
                f_required[j] = min(f_required[j], f_required[j - 1])
            for j in range(p):
                for _ in range(f_required[j]):
                    put(j, F)
            for j in range(p - 1, -1, -1):
                put(j, B)
            continue

        f_required = [0] * p
        for j in range(p):
            if f[j] >= nmb:
                continue
            if j + 1 < p and f[j] >= f[j + 1] + 2 and prioritize_b:
                next_plus_fw = e[j + 1] + cfg.cost_f[j + 1] + cfg.cost_b[j + 1] + comm_in_b(j)
                if e[j] >= next_plus_fw:
                    continue
                f_mem = cfg.mem_f[j]
                w_cost, w_cnt = 0.0, 0
                mem_with_w = m[j] + f_mem
                while mem_with_w > max_mem[j] + eps and w[j] + w_cnt < b[j]:
                    w_cost += cfg.cost_w[j]
                    mem_with_w += cfg.mem_w[j]
                    w_cnt += 1
                if e[j] + cfg.cost_f[j] + w_cost <= next_plus_fw:
                    f_required[j] = 1
                    continue
                w_cost = 0.0
                if next_plus_fw - (e[j] + w_cost) <= get_max_bubble() - stage_bubble[j]:
                    continue
            f_required[j] = 1
        for j in range(p - 2, -1, -1):
            f_required[j] = min(f_required[j], f_required[j + 1])
        for j in range(p):
            if f_required[j] == 0:
                continue
            while m[j] + cfg.mem_f[j] > max_mem[j] + eps:
                if w[j] >= b[j]:
                    raise ValueError("cannot fit memory")
                put(j, W)
            if j > 0:
                arrive = t[(F, j - 1, f[j])] + comm_in_f(j)
                while w[j] < b[j] and e[j] + cfg.cost_w[j] <= arrive:
                    put(j, W)
                if w[j] < b[j] and e[j] < arrive:
                    if arrive - e[j] <= get_max_bubble() - stage_bubble[j]:
                        if no_bubble_greedy:
                            put(j, W)
                    else:
                        put(j, W)
            put(j, F)
        for j in range(p - 1, -1, -1):
            assert b[j] == i
            while m[j] + cfg.mem_b[j] > max_mem[j] + eps:
                if w[j] >= b[j]:
                    raise ValueError("cannot fit memory")
                put(j, W)
            if j + 1 < p:
                arrive = t[(B, j + 1, i)] + comm_in_b(j)
                while w[j] < b[j] and e[j] + cfg.cost_w[j] <= arrive:
                    put(j, W)
                if w[j] < b[j] and e[j] < arrive:
                    if arrive - e[j] <= get_max_bubble() - stage_bubble[j]:
                        if no_bubble_greedy:
                            put(j, W)
                    else:
                        put(j, W)
            if j == 0 and f[j] == nmb:
                while w[j] < b[j]:
                    put(j, W)
            put(j, B)

    for j in range(p):
        while w[j] < nmb:
            put(j, W)

    makespan = max(t[(W, j, nmb - 1)] - t[(F, j, 0)] + cfg.cost_f[j] for j in range(p))
    return orders, t, makespan


def _mem_limit(cfg: ZBConfig) -> list:
    """The per-stage budget the heuristic works against (its default when
    cfg.max_mem is unset)."""
    return cfg.max_mem or [cfg.mem_f[s] * cfg.n_micro * 3 for s in range(cfg.n_stages)]


def _stage_peak(cfg: ZBConfig, s: int, order) -> float:
    mem = {F: cfg.mem_f[s], B: cfg.mem_b[s], W: cfg.mem_w[s]}
    m = peak = 0.0
    for typ, _mb in order:
        m += mem[typ]
        peak = max(peak, m)
    return peak


def peak_memory(cfg: ZBConfig, orders) -> list:
    """Per stage, the peak of the memory model replayed over its order."""
    return [_stage_peak(cfg, s, order) for s, order in enumerate(orders)]


def _fits(cfg: ZBConfig, orders, limit) -> bool:
    return all(pk <= lim + 1e-9 for pk, lim in zip(peak_memory(cfg, orders), limit))


def _warmup_order(n_micro: int, k: int) -> list:
    """k forwards, then B, W and the next F per microbatch: at most k
    microbatches hold activations and W frees each right after its B."""
    o = [(F, mb) for mb in range(k)]
    for i in range(n_micro):
        o += [(B, i), (W, i)]
        if k + i < n_micro:
            o.append((F, k + i))
    return o


def _degraded(cfg: ZBConfig, limit, mem_mode: str) -> Schedule:
    """SCHEDULER-I13: the best warm-up schedule that fits every budget.

    Stage j runs k_j forwards before its first B; k_j is at least one,
    non-increasing down the pipeline (so every B's inputs upstream of it are
    already ordered: no deadlock) and at most the largest warm-up whose memory
    replay fits stage j's budget. Equal warm-ups would make every microbatch
    wait a hop's round trip, so the gap between neighbours is what hides the
    latency: candidates are k_j = min(largest_j, 1 + g (p - 1 - j)) for a
    geometric range of gaps g (g = 1 is 1F1B's warm-up, the largest g the
    largest warm-ups); `simulate` picks the shortest makespan."""
    p, n = cfg.n_stages, cfg.n_micro
    largest: list = []
    cap = n
    for s in range(p):
        lo, hi = 1, cap  # k = 1 fits: _check_one_microbatch ran first
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _stage_peak(cfg, s, _warmup_order(n, mid)) <= limit[s] + 1e-9:
                lo = mid
            else:
                hi = mid - 1
        largest.append(lo)
        cap = lo
    g_max = max(1, -(-(largest[0] - 1) // max(1, p - 1)))
    gaps = sorted(
        {g for e in range(g_max.bit_length() + 1) for g in (2**e, 3 * 2**e // 2)} | {g_max}
    )
    best = None
    for g in (x for x in gaps if 1 <= x <= g_max):
        warm = [min(largest[j], 1 + g * (p - 1 - j)) for j in range(p)]
        orders = [_warmup_order(n, k) for k in warm]
        _end, ms = simulate(cfg, orders)
        if best is None or ms < best[1] - 1e-12:
            best = (orders, ms, warm)
    assert best is not None
    orders, ms, warm = best
    if not _fits(cfg, orders, limit):
        raise ValueError(f"degraded schedule does not fit the memory limit: warm-ups {warm}")
    bubble = bubble_rate(cfg, ms)
    params = dict(degraded=True, warmups=warm, bubble=bubble, mem_mode=mem_mode)
    return Schedule("zb", orders, ms, bubble, params)


def _check_one_microbatch(cfg: ZBConfig, limit) -> None:
    """Refuse a stage whose budget cannot hold one microbatch: its F plus
    what B keeps pending for W."""
    for s in range(cfg.n_stages):
        need = max(0.0, cfg.mem_f[s], cfg.mem_f[s] + cfg.mem_b[s])
        if need > limit[s] + 1e-9:
            raise ValueError(
                f"stage {s} cannot fit one microbatch: needs {need:.0f} bytes "
                f"(F {cfg.mem_f[s]:.0f} + pending W {max(0.0, cfg.mem_b[s]):.0f}), "
                f"budget {limit[s]:.0f} bytes"
            )


def zero_bubble(cfg: ZBConfig, mem_mode: str = "1p") -> Schedule:
    """The paper's automatic schedule. mem_mode "1p" keeps the 1F1B
    activation budget (ZB-1p), "2p" allows twice that (ZB-2p), "given" uses
    cfg.max_mem as is. Grid search over the heuristic's three binary choices,
    shortest makespan wins; every candidate is checked by `simulate` and by a
    memory replay against the budget. The warm-up family of `_degraded`
    (params.degraded) is returned when no grid candidate fits or when it
    beats the grid's best; the solver refuses only a budget below one
    microbatch."""
    if cfg.n_stages == 1:
        limit = _mem_limit(cfg)
        _check_one_microbatch(cfg, limit)
        o = [
            [(F, mb) for mb in range(cfg.n_micro)]
            + [(B, mb) for mb in range(cfg.n_micro)]
            + [(W, mb) for mb in range(cfg.n_micro)]
        ]
        if not _fits(cfg, o, limit):
            return _degraded(cfg, limit, mem_mode)
        _, ms = simulate(cfg, o)
        return Schedule(
            "zb", o, ms, bubble_rate(cfg, ms), {"mem_mode": mem_mode, "degraded": False}
        )
    if mem_mode in ("1p", "2p"):
        k = 1 if mem_mode == "1p" else 2
        cfg = ZBConfig(
            cfg.n_stages,
            cfg.n_micro,
            cfg.cost_f,
            cfg.cost_b,
            cfg.cost_w,
            cfg.comm,
            cfg.mem_f,
            cfg.mem_b,
            cfg.mem_w,
            max_mem=[k * cfg.n_stages * max(cfg.mem_f)] * cfg.n_stages,
        )
    limit = _mem_limit(cfg)
    _check_one_microbatch(cfg, limit)
    best = None
    for allow in (True, False):
        for prio in (True, False):
            for greedy in (True, False):
                try:
                    orders, _t, _ms = _manual_order(cfg, allow, prio, greedy)
                except ValueError:
                    continue
                if not _fits(cfg, orders, limit):
                    continue
                _end, ms = simulate(cfg, orders)
                if best is None or ms < best[1] - 1e-12:
                    best = (
                        orders,
                        ms,
                        dict(
                            allow_bubble_before_first_b=allow,
                            prioritize_b=prio,
                            no_bubble_greedy=greedy,
                        ),
                    )
    if best is None:
        return _degraded(cfg, limit, mem_mode)
    orders, ms, choice = best
    # the best that fits over both families: a grid candidate that barely
    # fits can lose to the warm-up family; ties keep the grid's schedule
    deg = _degraded(cfg, limit, mem_mode)
    if deg.makespan < ms - 1e-12:
        return deg
    params: dict = dict(choice, mem_mode=mem_mode, degraded=False)
    return Schedule("zb", orders, ms, bubble_rate(cfg, ms), params)


def comm_seconds(boundary_bytes: float, mbps: float, rtt_ms: float) -> float:
    """One boundary transfer across a hop: one-way latency + bytes at rate."""
    rate = boundary_bytes * 8 / (mbps * 1e6) if mbps and math.isfinite(mbps) and mbps > 0 else 0.0
    return max(0.0, rtt_ms) / 2000.0 + rate


# --------------------------------------------------------------------- engine bridge


def iwrr(n: int, weights) -> list:
    """Interleaved weighted round robin: owner of each of n microbatches, a
    peer of weight w getting a share w / sum(weights), interleaved so no peer
    receives a run of consecutive microbatches (SWARM's routing rule, static).
    Equal weights reproduce plain round robin (mb % width)."""
    w = [float(x) for x in weights]
    if any(x <= 0 for x in w):
        raise ValueError(f"peer weights must be positive, got {weights}")
    count = [0] * len(w)
    out = []
    for _ in range(n):
        # the peer whose next microbatch would finish first if each ran at its
        # relative speed; ties to the lower index, so the result is stable
        q = min(range(len(w)), key=lambda i: ((count[i] + 1) / w[i], i))
        count[q] += 1
        out.append(q)
    return out


def stage_memory_model(spec, layers: int, span=None, recompute=None) -> tuple[float, float, float]:
    """(mem_f, mem_b, mem_w) bytes of one microbatch on a stage of `layers`
    layers, in the runtime's terms. F holds the boundary (recompute) or the
    graph; B frees the graph and keeps what W needs, each split linear's X and
    dY: 16h values per token per GPT block (qkv 1+3, proj 1+1, fc1 1+4, fc2
    4+1), the paper's M_W = 32sbh bytes at 2 bytes per value.

    (M4) `span` (a, b) names the stage's entries of [Embed, Block*n, Head].
    An MoE block's pending W is its attention's values plus SPEC-model §2's
    MoE count (SCHEDULER-I21). An arch "hf" stage takes the census: its cut
    at HF's residual dtype, each layer's pending W, eager attention's scores
    in the graph, and each 3-D bank's gradient b_m·E·3hf held from B's end to
    W's start (charged in mem_b, freed in mem_w, in both branches). Without
    `span` the stage's layers are priced at the costliest block."""
    from wantrain.model import precision
    from wantrain.scheduler import autoconfig
    from wantrain.scheduler.autoconfig import activation_graph_bytes

    m = spec.model
    tokens = spec.mb_size * m.seq_len
    graph = activation_graph_bytes(spec.mb_size, m.seq_len, m.d_model, max(1, layers))
    bank = 0.0
    if autoconfig.is_hf(m):
        units = autoconfig.hf_unit_costs(
            autoconfig.hf_census(m),
            m.seq_len,
            spec.mb_size,
            m.compute_dtype,
            top_k=autoconfig.hf_top_k(m),
        )
        bnd = tokens * m.d_model * precision.spec(autoconfig.hf_boundary_dtype(m))["bytes"]
        if span is None:
            worst = max(units[1:-1], key=lambda u: u["pending_w_bytes"] + u["eager_bytes"])
            picked = [worst] * layers
        else:
            picked = units[span[0] : span[1]]
        w_pin = sum(u["pending_w_bytes"] for u in picked)
        graph += sum(u["eager_bytes"] for u in picked)
        bank = 4.0 * sum(u["bank_elems"] for u in picked)
    else:
        bnd = tokens * m.d_model * precision.spec(m.boundary_dtype)["bytes"]
        w_pin = layers * tokens * 16 * m.d_model * 4
        moe = autoconfig.moe_fields(m)
        if moe is not None:
            per = (
                autoconfig.builtin_attn_pending(m)
                + autoconfig.moe_layer_costs(moe, m.d_model, m.seq_len, spec.mb_size)[
                    "pending_w_per_token"
                ]
            )
            if span is None:
                w_pin = layers * tokens * max(16 * m.d_model, per) * 4
            else:
                a, b = span
                blocks = range(max(a, 1), min(b, m.n_layer + 1))
                w_pin = (
                    tokens
                    * 4
                    * sum(
                        per if autoconfig.is_moe_block(m, j - 1) else 16 * m.d_model for j in blocks
                    )
                )
    if spec.recompute if recompute is None else recompute:
        return float(bnd), float(w_pin + bank), -float(bnd + w_pin + bank)
    return float(graph), -float(graph - w_pin - bank), -float(w_pin + bank)


def selective_recompute(spec, force: set | frozenset = frozenset()) -> list | None:
    """RUNTIME-I25: per stage, recompute only where its pinned order holds more than one microbatch between an F and
    its B (on any of its slots). A stage at one in flight (always the last) keeps F's graph: B's rebuild would hold
    that same graph for that one microbatch, so keeping it costs no memory and saves a forward (about a quarter of
    the stage's compute at d 6144). `force`: stages a measured fit sent back to recomputing. None when nothing
    changes: the spec does not recompute, is arch "hf", or every stage needs it."""
    if not spec.recompute or spec.model.arch == "hf" or spec.schedule_order is None:
        return None
    out = []
    for r, slots in enumerate(spec.schedule_order):
        peak = 0
        for order in slots:
            live = 0
            for op, _mb in order:
                live += 1 if op == "F" else (-1 if op == "B" else 0)
                peak = max(peak, live)
        out.append(peak > 1 or r in force)
    return None if all(out) else out


def _stage_recomputes(spec, rank: int) -> bool:
    """spec.recomputes(rank) for a JobSpec; a planner namespace without one follows its `recompute`."""
    f = getattr(spec, "recomputes", None)
    return bool(f(rank)) if callable(f) else bool(spec.recompute)


def _boundary_bytes(spec) -> float:
    from wantrain.model import precision

    m = spec.model
    ps = precision.spec(m.boundary_dtype)
    coded = spec.mb_size * m.seq_len * m.d_model * ps["bytes"]
    if spec.wire_codec in ("planes", "fp8_planes"):
        coded /= ps["ratio"]
    return float(coded)


def _link_terms(spec) -> list:
    """Per hop (one-way latency s, transfer s) of the spec's WAN shim settings
    (emulated links); (0, 0) for a hop without settings."""
    coded = _boundary_bytes(spec)
    out = []
    for j in range(spec.stages - 1):
        lk = spec.links[j] if spec.links and j < len(spec.links) else {}
        bw = float(lk.get("bw_mbps", 0.0))
        out.append(
            (float(lk.get("latency_ms", 0.0)) / 1000.0, coded * 8 / (bw * 1e6) if bw > 0 else 0.0)
        )
    return out


def hop_comm_seconds(spec, hops=None) -> list:
    """Per-hop boundary transfer time, one-way latency included. hops:
    measured, in chain order, either [{mbps, rtt_ms}] or the sender's seconds
    per transfer (hops_from_telemetry), to which the link's one-way latency is
    added: a send's busy time ends when the bytes leave, before propagation;
    otherwise the spec's WAN shim settings (emulated links); otherwise 0."""
    coded = _boundary_bytes(spec)
    out = []
    for j, (lat, xfer) in enumerate(_link_terms(spec)):
        h = hops[j] if hops and j < len(hops) else None
        if isinstance(h, (int, float)) and not isinstance(h, bool):
            out.append(lat + float(h))
        elif isinstance(h, dict) and h:
            out.append(comm_seconds(coded, float(h["mbps"]), float(h["rtt_ms"])))
        else:
            out.append(lat + xfer)
    return out


def analytic_costs(spec) -> tuple[list, list, list]:
    """Per-stage F/B/W cost in relative units from the layer FLOP model (the
    paper's Table 1: F = h(24h+4s), B = h(24h+8s), W = 24h^2 per block and
    token; the head's three logits GEMMs 2hV each; the embedding a lookup).
    Recomputation adds one forward to B. Used when nothing was measured.

    (M4) An arch "hf" model takes each census unit's F/B/W
    (autoconfig.hf_unit_costs); an MoE block its attention plus the MoE
    MLP's active compute (autoconfig.moe_layer_costs, SCHEDULER-I21)."""
    from wantrain.scheduler import autoconfig

    m = spec.model
    h, s, v = m.d_model, m.seq_len, m.vocab_size
    if autoconfig.is_hf(m):
        units = autoconfig.hf_unit_costs(
            autoconfig.hf_census(m), s, 1, m.compute_dtype, top_k=autoconfig.hf_top_k(m)
        )
        per = [(u["f"], u["b"], u["w"]) for u in units]
    else:
        dense = (h * (24 * h + 4 * s), h * (24 * h + 8 * s), 24.0 * h * h)
        blocks = [dense] * m.n_layer
        moe = autoconfig.moe_fields(m)
        if moe is not None:
            c = autoconfig.moe_layer_costs(moe, h, s, 1)
            att = 2.0 * autoconfig.builtin_attn_params(m)
            mb = (att + 4 * h * s + c["f"], att + 8 * h * s + c["b"], att + c["w"])
            blocks = [mb if autoconfig.is_moe_block(m, j) else dense for j in range(m.n_layer)]
        per = [(h, h, 0.0)] + blocks + [(2.0 * h * v, 2.0 * h * v, 2.0 * h * v)]
    cf, cb, cw = [], [], []
    for a, b in spec.partition:
        f = sum(x[0] for x in per[a:b])
        cf.append(float(f))
        rc = _stage_recomputes(spec, len(cb))
        cb.append(float(sum(x[1] for x in per[a:b]) + (f if rc else 0.0)))
        cw.append(float(sum(x[2] for x in per[a:b])))
    return cf, cb, cw


def slot_orders(stage_orders, mb_owner, widths) -> list:
    """Split each stage's order into its peers' orders by microbatch owner.
    Each peer keeps the stage order's relative sequence, so the union of
    precedence relations is a subset of the stage-level one, which is acyclic:
    filtering cannot introduce a deadlock."""
    out = []
    for s, order in enumerate(stage_orders):
        per = [[] for _ in range(widths[s])]
        for t, mb in order:
            per[mb_owner[s][mb]].append([t, int(mb)])
        out.append(per)
    return out


def _budget_total(b) -> float:
    """A stage's activation budget: a number, or the sum of its per-peer list."""
    return float(sum(b)) if isinstance(b, (list, tuple)) else float(b)


def _peak_memory(cfg: ZBConfig, s: int, order) -> float:
    """Peak pending activation memory of one order on stage s (cfg's model)."""
    step = {F: cfg.mem_f[s], B: cfg.mem_b[s], W: cfg.mem_w[s]}
    m = peak = 0.0
    for t, _mb in order:
        m += step[t]
        peak = max(peak, m)
    return peak


def _lane_orders(cfg: ZBConfig, mb_owner, widths, peer_weights, mem_mode, per_peer, view):
    """Lane-consistent orders for the wide stages: {stage: [order per peer]}.

    Each peer runs whole microbatches, so a lane is a pipeline of its own: one
    zero-bubble schedule for the microbatches the peer owns, every stage at
    that peer's op time (aggregate cost x total weight / its weight, what
    simulate_slots charges it), mapped back onto those microbatches. The lane
    gets its own peer's activation budget, never the stage's sum: under "auto"
    with budgets a wide stage needs a per-peer list (a scalar keeps the
    stage-level split); the other stages of the lane model take view[r], a
    per-peer view of theirs (see bridge_orders).
    A stage is kept at stage level if any of its lanes does not fit."""
    p = cfg.n_stages
    k = 1 if mem_mode == "1p" else 2
    cache: dict = {}
    out = {}
    for s in range(p):
        if widths[s] == 1 or (per_peer is not None and per_peer[s] is None):
            continue
        total = sum(peer_weights[s])
        per = []
        for q in range(widths[s]):
            own = [mb for mb, o in enumerate(mb_owner[s]) if o == q]
            if not own:
                per.append([])
                continue
            scale = total / peer_weights[s][q]
            lane = ZBConfig(
                p,
                len(own),
                [c * scale for c in cfg.cost_f],
                [c * scale for c in cfg.cost_b],
                [c * scale for c in cfg.cost_w],
                cfg.comm,
                cfg.mem_f,
                cfg.mem_b,
                cfg.mem_w,
            )
            mode = "given"
            if per_peer is not None:
                lane.max_mem = [float(per_peer[s][q]) if r == s else view[r] for r in range(p)]
            elif mem_mode in ("auto", "2p", "1p"):
                lane = _with_pending_budget(lane, k)
            else:
                mode = mem_mode
            key = (len(own), scale, mode, tuple(lane.max_mem) if lane.max_mem else None)
            if key not in cache:
                try:
                    cache[key] = zero_bubble(lane, mode).orders
                except ValueError:
                    cache[key] = None
            orders = cache[key]
            if orders is None:
                break
            order = [[t, own[i]] for t, i in orders[s]]
            if lane.max_mem and _peak_memory(cfg, s, order) > lane.max_mem[s] + 1e-9:
                break
            per.append(order)
        else:
            out[s] = per
    return out


def _lanes_fit(cfg: ZBConfig, orders, per_peer: list) -> bool:
    """Every slot of a stage with per-peer budgets peaks within its own."""
    return all(
        _peak_memory(cfg, s, order) <= caps[q] + 1e-9
        for s, caps in enumerate(per_peer)
        if caps is not None
        for q, order in enumerate(orders[s])
    )


def bridge_orders(cfg: ZBConfig, mb_owner, widths, peer_weights, mem_mode="auto", budgets=None):
    """Slot orders for a pipeline of (possibly wide) stages from per-stage
    aggregate costs. -> (slot orders, stage Schedule, slot-level makespan).

    The stage-level solve (one server per stage, split per peer by owner) is
    the baseline; wide stages also get lane-consistent orders (_lane_orders),
    and those are returned when simulate_slots finds them faster, or when only
    they keep every per-peer budget. Width-1 pipelines get the stage-level
    orders unchanged. budgets[stage] is a number or one number per peer; under
    "auto" the stage-level solve uses their sum."""
    p = cfg.n_stages
    per_peer: list | None = None
    view: list = []
    if mem_mode == "auto" and budgets and any(_budget_total(b) for b in budgets):
        # each measured budget as given (a tight one degrades, SCHEDULER-I13);
        # only an unmeasured stage (0) takes its ZB-2p default
        default = _with_pending_budget(cfg, 2).max_mem or []
        cfg.max_mem = [_budget_total(b) or d for b, d in zip(budgets, default)]
        per_peer = [
            list(b) if isinstance(b, (list, tuple)) and _budget_total(b) else None for b in budgets
        ]
        # one lane's model of another stage: its smallest peer budget, or an
        # even share of the stage's budget
        view = [
            float(min(b)) if per_peer[r] is not None else cfg.max_mem[r] / widths[r]
            for r, b in enumerate(budgets)
        ]
        sched = zero_bubble(cfg, "given")
    elif mem_mode in ("auto", "2p"):
        sched = zero_bubble(_with_pending_budget(cfg, 2), "given")
    elif mem_mode == "1p":
        sched = zero_bubble(_with_pending_budget(cfg, 1), "given")
    else:
        sched = zero_bubble(cfg, mem_mode)
    orders = slot_orders(sched.orders, mb_owner, widths)
    _end, ms = simulate_slots(cfg, orders, mb_owner, peer_weights)
    lanes = _lane_orders(cfg, mb_owner, widths, peer_weights, mem_mode, per_peer, view)
    if not lanes:
        return orders, sched, ms
    mixed = [lanes.get(s, orders[s]) for s in range(p)]
    try:
        _end, lane_ms = simulate_slots(cfg, mixed, mb_owner, peer_weights)
    except ValueError:  # a lane order that waits on a stage-level one: keep the split
        return orders, sched, ms
    if lane_ms < ms or (
        per_peer is not None
        and _lanes_fit(cfg, mixed, per_peer)
        and not _lanes_fit(cfg, orders, per_peer)
    ):
        sched.params["lane_stages"] = sorted(lanes)
        return mixed, sched, lane_ms
    return orders, sched, ms


def solve(spec, cost_f, cost_b, cost_w, comm, mem_budgets=None, mem_mode: str = "auto"):
    """Zero-bubble orders for every slot of `spec` from per-stage aggregate
    costs (bridge_orders). -> (slot orders, stage Schedule, ZBConfig,
    slot-level makespan).

    mem_budgets: measured per-stage bytes, used as given; an entry may be a
    list with one budget per peer of a wide stage (its lanes are solved
    against those). A 0 entry means "not measured" (the coordinator's encoding
    for a device without VRAM figures) and gives that stage alone its ZB-2p
    default; all 0 is ZB-2p."""
    p = spec.stages
    mems = [
        stage_memory_model(spec, b - a, (a, b), _stage_recomputes(spec, r))
        for r, (a, b) in enumerate(spec.partition)
    ]
    cfg = ZBConfig(
        p,
        spec.microbatches,
        list(cost_f),
        list(cost_b),
        list(cost_w),
        comm=comm,
        mem_f=[x[0] for x in mems],
        mem_b=[x[1] for x in mems],
        mem_w=[x[2] for x in mems],
    )
    if mem_mode == "auto" and mem_budgets and any(_budget_total(b) for b in mem_budgets):
        if len(mem_budgets) != p:
            raise ValueError(f"{len(mem_budgets)} memory budgets for {p} stages")
    orders, sched, ms = bridge_orders(
        cfg, spec.mb_owner, spec.stage_widths, spec.peer_weights, mem_mode, mem_budgets
    )
    return orders, sched, cfg, ms


def default_schedule(spec) -> list:
    """The schedule a spec gets when it arrives without one: the paper's
    heuristic on the analytic cost model, ZB-2p memory, no communication
    term (relative units carry no seconds). Deterministic, so every rank that
    finalizes the same spec derives the same orders."""
    cf, cb, cw = analytic_costs(spec)
    agg = [sum(w) for w in spec.peer_weights]
    cf = [c / a for c, a in zip(cf, agg)]
    cb = [c / a for c, a in zip(cb, agg)]
    cw = [c / a for c, a in zip(cw, agg)]
    orders, _s, _c, _ms = solve(spec, cf, cb, cw, [0.0] * (spec.stages - 1), mem_mode="2p")
    return orders


def schedule_for_spec(
    spec, profiles, hops=None, mem_budgets=None, mem_mode: str = "auto", every_peer: bool = False
):
    """Solve the schedule for a placed spec from measured costs.

    profiles[stage][peer] = {t_f, t_b, t_w} seconds on that peer's own device.
    A stage's aggregate cost is the parallel-service cost 1 / sum(1 / c_peer),
    each peer's weight its throughput 1 / (t_f + t_b + t_w): microbatches are
    owned in proportion to it (iwrr), so a twice-as-fast peer computes twice as
    many. hops: measured links in chain order (else the spec's shim settings).
    mem_budgets[stage]: activation bytes the stage may hold, or a list with
    one budget per peer (a wide stage's lanes are solved only against those).
    every_peer: no peer is left without a microbatch (_every_peer_owns), so a
    re-plan keeps every slot measured.

    Mutates spec.peer_weights, spec.mb_owner and spec.schedule_order and
    returns {schedule, config, slot_makespan, peer_weights}."""
    spec.finalize()
    p = spec.stages
    spec.peer_weights, agg = _measured(profiles)
    spec.mb_owner = [iwrr(spec.microbatches, w) for w in spec.peer_weights]
    if every_peer:
        spec.mb_owner = [
            _every_peer_owns(o, len(w)) for o, w in zip(spec.mb_owner, spec.peer_weights)
        ]
    orders, sched, cfg, ms = solve(
        spec,
        [a[0] for a in agg],
        [a[1] for a in agg],
        [a[2] for a in agg],
        hop_comm_seconds(spec, hops),
        mem_budgets,
        mem_mode,
    )
    spec.schedule_order = orders
    spec.finalize()
    assert p == cfg.n_stages
    return {
        "schedule": sched,
        "config": cfg,
        "slot_makespan": ms,
        "peer_weights": spec.peer_weights,
    }


def _every_peer_owns(owner: list, width: int) -> list:
    """owner with every peer owning at least one microbatch when there are as
    many microbatches as peers: each peer without one (lowest index first)
    takes the last microbatch of the peer owning the most (ties to the lower
    index). iwrr starves a peer slower than about n_micro x the fastest; a
    starved slot runs no op, so telemetry could never measure it again."""
    out = list(owner)
    if len(out) < width:
        return out
    for q in range(width):
        if q not in out:
            counts = [out.count(i) for i in range(width)]
            donor = max(range(width), key=lambda i: (counts[i], -i))
            out[max(k for k, o in enumerate(out) if o == donor)] = q
    return out


def _measured(profiles) -> tuple[list, list]:
    """(peer weights, per-stage aggregate (t_f, t_b, t_w)) of profiles: each
    peer's weight its throughput 1 / (t_f + t_b + t_w), normalized to the
    stage's fastest; a stage's cost the parallel-service 1 / sum(1 / c_peer)."""
    weights = [
        [1.0 / max(1e-12, pr["t_f"] + pr["t_b"] + pr["t_w"]) for pr in row] for row in profiles
    ]
    agg = [
        tuple(
            1.0 / sum(1.0 / max(1e-12, float(pr[k])) for pr in row) for k in ("t_f", "t_b", "t_w")
        )
        for row in profiles
    ]
    return [[w / max(row) for w in row] for row in weights], agg


# --------------------------------------------------------------------- re-plan (SCHEDULER-I15)


def _window(windows, r: int, q: int) -> list:
    win = list(windows.get((r, q)) or [])
    if not win:
        raise ValueError(f"no telemetry for slot ({r}, {q})")
    return win


def profiles_from_telemetry(spec, windows) -> list:
    """profiles[stage][peer] = {t_f, t_b, t_w} seconds: per op, the median of
    the validated summaries' med_ms over the slot's window (summaries of a
    step that ran no such op are skipped). windows: {(rank, peer): [summary,
    ...]}, the coordinator's per-slot window (rank == stage). A slot without a
    sample of every op is refused, naming it."""
    out = []
    for r in range(spec.stages):
        row = []
        for q in range(spec.stage_widths[r]):
            win = _window(windows, r, q)
            pr = {}
            for op in (F, B, W):
                xs = [float(s["ops"][op]["med_ms"]) for s in win if s["ops"][op]["n"] > 0]
                if not xs:
                    raise ValueError(f"no {op} sample in the telemetry of slot ({r}, {q})")
                if not all(math.isfinite(x) and x > 0 for x in xs):
                    raise ValueError(f"{op} med_ms of slot ({r}, {q}) must be finite and > 0")
                pr["t_" + op.lower()] = statistics.median(xs) / 1e3
            row.append(pr)
        out.append(row)
    return out


def hops_from_telemetry(spec, windows) -> list:
    """SCHEDULER-I15, per hop j -> j+1 in chain order, the sender's seconds
    per transfer (one boundary tensor): the median of busy_ms / 1e3 / n over
    the window's sends across it (stage j's next<q> and stage j+1's prev<q>
    connections). n = frames with one stream (one send event per transfer);
    with p2p_streams > 1 each event is one fragment, so n = bytes /
    _boundary_bytes(spec). Propagation is not in it; hop_comm_seconds adds
    the link's one-way latency. A hop nobody sent on keeps the spec's
    modelled transfer term."""
    model = [xfer for _lat, xfer in _link_terms(spec)]
    per_xfer = None if spec.p2p_streams == 1 else _boundary_bytes(spec)
    out = []
    for j in range(spec.stages - 1):
        xs = []
        for r, head in ((j, "next"), (j + 1, "prev")):
            for q in range(spec.stage_widths[r]):
                for s in _window(windows, r, q):
                    for label, h in sorted(s["hops"].items()):
                        if label.rstrip("0123456789") != head or h["frames"] <= 0:
                            continue
                        n = h["frames"] if per_xfer is None else h["bytes"] / per_xfer
                        if n > 0:
                            xs.append(h["busy_ms"] / 1e3 / n)
        out.append(statistics.median(xs) if xs else model[j])
    return out


def predict_step(spec, profiles, hops) -> float:
    """Slot-level step time of spec's current orders and owners under the
    measured costs (simulate_slots): each peer runs at its measured speed,
    whatever weights its ownership was derived from."""
    weights, agg = _measured(profiles)
    cfg = ZBConfig(
        spec.stages,
        spec.microbatches,
        [a[0] for a in agg],
        [a[1] for a in agg],
        [a[2] for a in agg],
        comm=hop_comm_seconds(spec, hops),
    )
    _end, ms = simulate_slots(cfg, spec.schedule_order, spec.mb_owner, weights)
    return ms


_REPLANNED = ("peer_weights", "mb_owner", "schedule_order")


def replan(spec, windows, min_gain: float = 0.10, mem_budgets=None, mem_mode: str = "auto"):
    """SCHEDULER-I15: re-solve ownership and order from one window of runtime
    telemetry. Pure: spec is not modified. Both predictions simulate the same
    measured costs, so the profiler's bias cancels. decision is true only when
    the re-solved step is at least min_gain shorter and the ownership or an
    order differs; only peer_weights, mb_owner and schedule_order may change
    (the fold order is the microbatch index, so the bytes do not). Every
    peer keeps at least one microbatch, so the next window measures it again.
    mem_budgets / mem_mode go to schedule_for_spec as at launch."""
    if not 0.0 <= min_gain < 1.0:
        raise ValueError(f"min_gain must be in [0, 1), got {min_gain}")
    cur = copy.deepcopy(spec).finalize()
    profiles = profiles_from_telemetry(cur, windows)
    hops = hops_from_telemetry(cur, windows)
    pred_cur = predict_step(cur, profiles, hops)
    new = copy.deepcopy(cur)
    schedule_for_spec(new, profiles, hops, mem_budgets, mem_mode, every_peer=True)
    a, b = asdict(cur), asdict(new)
    moved = sorted(k for k in a if k not in _REPLANNED and a[k] != b[k])
    if moved:
        raise AssertionError(f"re-plan changed fields outside ownership and order: {moved}")
    pred_new = predict_step(new, profiles, hops)
    owner_changed = new.mb_owner != cur.mb_owner
    order_changed = new.schedule_order != cur.schedule_order
    gain_ok = pred_new <= (1.0 - min_gain) * pred_cur
    decision = gain_ok and (owner_changed or order_changed)
    if decision:
        why = f"predicted {pred_cur:.6g} s -> {pred_new:.6g} s per step"
    elif not (owner_changed or order_changed):
        why = "the re-solve gives the current ownership and order"
    else:
        why = (
            f"predicted {pred_cur:.6g} s -> {pred_new:.6g} s per step, "
            f"below the {min_gain:.0%} threshold"
        )
    return {
        "decision": decision,
        "pred_cur_s": pred_cur,
        "pred_new_s": pred_new,
        "peer_weights": [[float(w) for w in row] for row in new.peer_weights],
        "mb_owner": [[int(q) for q in row] for row in new.mb_owner],
        "schedule_order": [
            [[[str(t), int(mb)] for t, mb in o] for o in per] for per in new.schedule_order
        ],
        "owner_changed": owner_changed,
        "why": why,
    }


def _with_pending_budget(cfg: ZBConfig, k: int) -> ZBConfig:
    """Budget for k x p microbatches pending backward (the repository's
    --zero-bubble-max-pending-backward) in this memory model: each pending
    microbatch costs its F allocation plus, when B allocates (recompute), what
    B keeps for W."""
    per = [abs(f) + max(0.0, b) for f, b in zip(cfg.mem_f, cfg.mem_b)]
    return ZBConfig(
        cfg.n_stages,
        cfg.n_micro,
        cfg.cost_f,
        cfg.cost_b,
        cfg.cost_w,
        cfg.comm,
        cfg.mem_f,
        cfg.mem_b,
        cfg.mem_w,
        max_mem=[k * cfg.n_stages * x for x in per],
    )
