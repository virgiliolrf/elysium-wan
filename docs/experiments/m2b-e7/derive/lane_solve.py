# m2b-e7: the lane solve (SCHEDULER-I13 / I19), written from the SPEC text and first principles, independent of
# src/wantrain.
#
# The human's rule (2026-10-10): the derivation scripts are the oracle and do not copy the implementation; they never
# import src/wantrain nor load its source. This module replaces e7_spec.py's former load of
# src/wantrain/scheduler/zero_bubble.py by path (sha256 a93a8bed..., the as-built lane solve).
#
# What a lane is (SCHEDULER-I19): each peer of a wide stage runs its own microbatches (n = M / w) as a pipeline of its
# own, every stage at that peer's op time, under that peer's activation budget. For a width-1 plan the lane is the
# pipeline itself.
#
# The model (SPEC-scheduler section 2.1, "Inputs" and "Simulators"), restated:
#   F(s, mb) after F(s-1, mb) + comm[s-1]; B(s, mb) after B(s+1, mb) + comm[s] (the last stage's B after its own F);
#   W(s, mb) after B(s, mb); one op at a time per stage; makespan = max over stages of (last end - first start).
#   Memory per stage: F allocates mem_f (the boundary tensor), B allocates mem_b (the weight-gradient inputs W needs,
#   recompute on), W frees both (mem_w = -(mem_f + mem_b)); the budget is p x (|mem_f| + max(0, mem_b)): p microbatches
#   pending backward (--zero-bubble-max-pending-backward 1, as the granted derivation solves the lanes).
#
# The order (solve): a list scheduler that applies the zero-bubble rules the SPEC states (section 2.1 "Heuristic" 1-6,
# Qi et al. ICLR 2024 section 3): whenever a stage is free at time t it takes, in this priority,
#   1. B of its next microbatch if its input has arrived and the budget allows (B is on the critical path);
#   2. a W if the budget forbids the next F or B (W frees memory);
#   3. F of its next microbatch if its input has arrived, the budget allows, and the F surely ends before the next B's
#      input arrives (an F never delays the backward chain; warm-up and 1F1B alternation follow from 1 and 3);
#   4. a W if it surely ends before the next F or B input arrives (W fills bubbles, never delays an input);
#   5. otherwise it waits for the next arrival; after the last F and B it drains its W.
# "Surely" uses the arrival when the producing op has started, else a lower bound (the producer's free time + its op
# cost + the hop). Events are processed in time order, stage index breaking ties, so the result is deterministic.
#
# The closed form (bound): for these lanes the bottleneck is the last stage L (head + the last block). Any order
# satisfies, for every stage j,
#   span_j >= (T_L - T_j) + n (F_L + B_L) + (n - K) W_L + sum_{i=j}^{L-1} (B_i + comm_i) + W_j,
#   T_j = sum_{i<j} (F_i + comm_i),   K = floor(budget_L / (mem_f_L + mem_b_L)) = p,
# because the last stage's first F cannot start before its input arrives, its last B follows its n F, its n B and at
# least n - K of its W (at most K microbatches may be pending W when that B ends: memory), the backward chain then
# crosses every hop down to j, and j's last W follows its last B; and span_j >= n (F_j + B_j + W_j). P* = max_j of
# both is a lower bound on the makespan of every memory-feasible order. solve() attains it on all of E7's lanes (the
# lanes section of e7_spec.out prints both), so the period is fixed by the model, not by a heuristic's choices.
F, B, W = "F", "B", "W"
INF = float("inf")
_IX = {F: 0, B: 1, W: 2}


def execute(cost, comm, orders):
    """cost[s] = (F, B, W) seconds; comm[s] = hop s -> s+1 seconds; orders[s] = [(type, mb)] -> ({(type, s, mb): end},
    makespan). Every op starts as soon as its stage is free and its input has arrived (the SPEC's simulate)."""
    p = len(orders)
    end = {}
    pos = [0] * p
    free = [0.0] * p
    first = [None] * p
    left = sum(len(o) for o in orders)
    while left:
        moved = False
        for s in range(p):
            o = orders[s]
            while pos[s] < len(o):
                t, mb = o[pos[s]]
                r = _ready(end, comm, p, t, s, mb)
                if r is None:
                    break
                st = max(free[s], r)
                if first[s] is None:
                    first[s] = st
                free[s] = st + cost[s][_IX[t]]
                end[(t, s, mb)] = free[s]
                pos[s] += 1
                left -= 1
                moved = True
        if not moved:
            raise ValueError("order deadlocks")
    return end, max(free[s] - first[s] for s in range(p))


def _ready(end, comm, p, t, s, mb):
    if t == F:
        if s == 0:
            return 0.0
        k = (F, s - 1, mb)
        return end[k] + comm[s - 1] if k in end else None
    if t == B:
        if s == p - 1:
            return end.get((F, s, mb))
        k = (B, s + 1, mb)
        return end[k] + comm[s] if k in end else None
    return end.get((B, s, mb))


def budgets(mem_f, mem_b):
    p = len(mem_f)
    return [p * (abs(a) + max(0.0, b)) for a, b in zip(mem_f, mem_b)]


def peak(order, mf, mb_):
    step = {F: mf, B: mb_, W: -(mf + mb_)}
    m = pk = 0.0
    for t, _ in order:
        m += step[t]
        pk = max(pk, m)
    return pk


def solve(cost, comm, n, mem_f, mem_b):
    """-> (orders, makespan). The list scheduler of the header; its orders are re-executed by execute() (the SPEC's
    dependency semantics) and must give the same times; every stage's memory replay must fit its budget."""
    p = len(cost)
    lim = budgets(mem_f, mem_b)
    end = {}
    nf, nb, nw = [0] * p, [0] * p, [0] * p
    mem, free = [0.0] * p, [0.0] * p
    orders = [[] for _ in range(p)]
    eps = 1e-9

    def arrival(t, s, mb):
        """(time, known): the input's arrival, or a lower bound on it while its producer has not started."""
        r = _ready(end, comm, p, t, s, mb)
        if r is not None:
            return r, True
        if t == F:
            return free[s - 1] + cost[s - 1][0] + comm[s - 1], False
        return free[s + 1] + cost[s + 1][1] + comm[s], False

    def put(s, t, mb, st):
        e = st + cost[s][_IX[t]]
        end[(t, s, mb)] = e
        free[s] = e
        orders[s].append((t, mb))
        mem[s] += {F: mem_f[s], B: mem_b[s], W: -(mem_f[s] + mem_b[s])}[t]
        if t == F:
            nf[s] += 1
        elif t == B:
            nb[s] += 1
        else:
            nw[s] += 1

    def act(s, t):
        b_pending = nb[s] < nf[s]
        f_pending = nf[s] < n
        b_mem = mem[s] + mem_b[s] <= lim[s] + eps
        f_mem = mem[s] + mem_f[s] <= lim[s] + eps
        if b_pending and b_mem:
            a, known = arrival(B, s, nb[s])
            if known and a <= t:
                put(s, B, nb[s], t)
                return True
        if ((b_pending and not b_mem) or (f_pending and not f_mem)) and nw[s] < nb[s]:
            put(s, W, nw[s], t)
            return True
        if f_pending and f_mem:
            a, known = arrival(F, s, nf[s])
            if known and a <= t:
                ok = True
                if b_pending and s < p - 1:
                    ab, _ = arrival(B, s, nb[s])
                    ok = t + cost[s][0] <= ab
                if ok:
                    put(s, F, nf[s], t)
                    return True
        if nw[s] < nb[s]:
            nxt = INF
            if b_pending:
                nxt = min(nxt, arrival(B, s, nb[s])[0])
            if f_pending:
                nxt = min(nxt, arrival(F, s, nf[s])[0])
            if t + cost[s][2] <= nxt:
                put(s, W, nw[s], t)
                return True
        return False

    while True:
        live = [s for s in range(p) if nw[s] < n]
        if not live:
            break
        t = min(free[s] for s in live)
        if any(free[s] <= t and act(s, t) for s in live):
            continue
        nxt = INF
        for s in live:
            if free[s] > t:
                nxt = min(nxt, free[s])
            for typ, mb, cond in ((B, nb[s], nb[s] < nf[s]), (F, nf[s], nf[s] < n)):
                if cond:
                    a, known = arrival(typ, s, mb)
                    if known and a > t:
                        nxt = min(nxt, a)
        if nxt == INF:                         # nothing will arrive: drain a W (rule 5)
            s = next(s for s in live if free[s] <= t and nw[s] < nb[s])
            put(s, W, nw[s], t)
            continue
        for s in live:
            if free[s] <= t:
                free[s] = nxt
    end2, ms = execute(cost, comm, orders)
    assert all(abs(end2[k] - end[k]) <= 1e-12 * max(1.0, end[k]) for k in end), "list schedule != its execution"
    for s in range(p):
        assert peak(orders[s], mem_f[s], mem_b[s]) <= lim[s] + eps, ("memory", s)
    return orders, ms, end2


def lower_bound(cost, comm, n, mem_f, mem_b):
    """P* of the header: a lower bound on the makespan of every memory-feasible order of this lane."""
    p = len(cost)
    L = p - 1
    K = int(budgets(mem_f, mem_b)[L] // (abs(mem_f[L]) + max(0.0, mem_b[L])))
    T = [0.0]
    for j in range(L):
        T.append(T[-1] + cost[j][0] + comm[j])
    lb = max(n * sum(c) for c in cost)
    for j in range(p):
        v = (T[L] - T[j]) + n * (cost[L][0] + cost[L][1]) + max(0, n - K) * cost[L][2]
        v += sum(cost[i][1] + comm[i] for i in range(j, L)) + cost[j][2]
        lb = max(lb, v)
    return lb, K


def lanes(cost, comm, n, mem_f, mem_b):
    """-> ({stage: [(type, start, end, mb)]}, makespan, lower bound P*)."""
    orders, ms, end = solve(cost, comm, n, mem_f, mem_b)
    per = {s: [] for s in range(len(cost))}
    for (t, s, mb), e in end.items():
        per[s].append((t, e - cost[s][_IX[t]], e, mb))
    for s in per:
        per[s].sort(key=lambda o: (o[1], o[2]))
    return per, ms, lower_bound(cost, comm, n, mem_f, mem_b)[0]
