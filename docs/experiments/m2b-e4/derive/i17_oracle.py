# SCHEDULER-I17 oracle (SC-4): an independent model of a sharded stage's exchange, written from the text of
# docs/design/SPEC-scheduler.md SCHEDULER-I17 (and I16's S2 for how E and window_s are read), not from the
# implementation. It imports nothing from src/wantrain and reads no product source.
#
# What I17 says, and where each clause lives below:
#   units ............ a SplitLinear module, or the Embed's tables as ONE fp32 unit released when W(mb) starts and
#                      needed first by F (Unit.flops == 0)                                           -> Unit, _release
#   queues ........... one send queue per (sender, destination), served need-first (forward unit, gathers before
#                      gradients), one chunk of at most `chunk` bytes at a time                        -> Fluid
#   resources ........ each peer's egress and ingress at min(NIC, PCIe); each host's DRAM at k passes per byte per
#                      side; each site uplink per direction; all shared max-min fairly among the chunks in flight
#                                                                                                       -> Fluid._rates
#   hop latency ...... 2*f/PCIe per hop (one copy per side), plus RTT/2                               -> _lat
#   gradient release . a W releases a unit's pieces when it computes the unit, in the stage's emission order
#   gather release ... after the owner's LAST fold of the unit and its AdamW, at their HBM op times
#                      ((b_g + 16) B per folded element for a 2-byte wire, 12 for fp32; 30 B per updated element)
#   next forward ..... step t+1's F on a unit waits for that unit's gather                            -> _late
#   fold device time . M*(P_s/w) elements per peer per step (T_ovh)                                  -> overhead
#   one simulation ... E, window_s and the step all come from ONE simulation of every microbatch's traffic.
#
# Reading of E and window_s from the one simulation (I16 S2, ADR-0006 T_step = C(1+beta) + E - window_s + T_ovh):
#   T_end(d)    = end of peer d's last op in step t; slot(d) = period + start of d's first F (step t+1's slot).
#   window_s    = min_d (slot(d) - T_end(d)): the idle time the stage has before step t+1's forward is due.
#   E_s         = max_d D(d), D accumulated over the units in forward order with F(t+1) of peer d needed from
#                 T_end(d) (derivation.md section 2: need = base + D + sum of earlier units' F time).
#   exposed_s   = max(0, E_s - window_s); step = period + max_s exposed_s + T_ovh.
# When the links bind, this one simulation gives E = max(X_s, V/cap + RTT + t_gap - C) (I17's last sentence):
# the tail and the link-bound term are not two runs here, they are what the single run produces.
#
# Modes beyond I17 (stated so a coder can implement the same thing; see e4_spec.py's header for the full text):
#   flush="stage"  whole-stage flush (E4's "today" arms): every unit's pieces of W(mb) released at W(mb)'s END.
#   barrier=True   full step barrier: every gather released at Tb = max over (unit, owner) of (last fold + AdamW);
#                  every unit of step t+1 also waits for Tb.
#   tree=True      F5's residue-group fold (test-only): each peer sends one fp32 partial per unit piece, released
#                  when its LAST W produces the unit; owners fold the w partials (fp32 add) then AdamW, then gather.
#   policy="fifo"  E4's control: queues served by release time.
import heapq
import math

INF = math.inf
ADAMW_B = 30            # B per updated element (I17)


def fold_bytes(b_g):
    """HBM bytes per folded element (I17): widen + add for a 2-byte wire (b_g + 16), 12 for fp32."""
    return b_g + 16 if b_g < 4 else 12


class Unit:
    __slots__ = ("name", "params", "b_g", "b_p", "fwd", "flops")

    def __init__(self, name, params, b_g, b_p, fwd, flops):
        self.name, self.params, self.b_g, self.b_p, self.fwd, self.flops = name, params, b_g, b_p, fwd, flops


class Peer:
    __slots__ = ("name", "nic", "pcie", "host", "site")

    def __init__(self, name, nic, pcie, host, site="s0"):
        self.name, self.nic, self.pcie, self.host, self.site = name, nic, pcie, host, site


class Lane:
    """One peer's op timeline for step t: ops = [(typ, start, end, global_mb)], typ in 'F', 'B', 'W'."""
    __slots__ = ("ops", "first_f", "end", "cost_f")

    def __init__(self, ops):
        self.ops = sorted(ops, key=lambda o: (o[2], o[1]))
        fs = [o for o in self.ops if o[0] == "F"]
        self.first_f = min(o[1] for o in fs)
        self.end = max(o[2] for o in self.ops)
        self.cost_f = fs[0][2] - fs[0][1]


class Fabric:
    """host_dram: host -> B/s (both directions together), INF = unconstrained; k: passes per byte per side;
    uplinks: (site_a, site_b) -> B/s per direction; rtt inside a site; site_rtt between sites; chunk in bytes."""

    def __init__(self, host_dram, k=10.0, uplinks=None, rtt=0.0, site_rtt=0.0, chunk=16 * 2**20):
        self.host_dram, self.k, self.uplinks = host_dram, k, (uplinks or {})
        self.rtt, self.site_rtt, self.chunk = rtt, site_rtt, chunk


class Rates:
    """Device op rates in B/s for the fold of a 2-byte wire, the fold of an fp32 wire, and AdamW."""

    def __init__(self, fold2, fold4, adam):
        self.fold2, self.fold4, self.adam = fold2, fold4, adam

    def fold_t(self, n, b_g):
        return n * fold_bytes(b_g) / (self.fold2 if b_g < 4 else self.fold4)

    def adam_t(self, n):
        return n * ADAMW_B / self.adam


class StageIn:
    """A sharded stage: units, owners[u][q] (fractions summing to 1), peers, lanes (one per peer), the period of the
    step, the boundary bytes per microbatch per direction (load only: released at every F end, both directions),
    device op rates, and the modes."""

    def __init__(self, units, owners, peers, lanes, period, rates, boundary=0.0, barrier=False, flush="module",
                 tree=False, M=None):
        self.units, self.owners, self.peers, self.lanes = units, owners, peers, lanes
        self.period, self.rates, self.boundary = period, rates, boundary
        self.barrier, self.flush, self.tree = barrier, flush, tree
        self.M = M if M is not None else sum(1 for ln in lanes for o in ln.ops if o[0] == "W")


# ------------------------------------------------------------------------------------------------- fluid engine
class _Item:
    __slots__ = ("rel", "q", "prio", "left_bytes", "chunks", "nxt", "lat", "done", "seq")


class Fluid:
    """Chunks in flight, one per queue; rates max-min fair over every finite resource a queue crosses, each with a
    coefficient (bytes of the resource per byte sent: 1 on a port or uplink, k per side on host DRAM)."""

    def __init__(self, fabric, policy):
        self.fab, self.policy = fabric, policy
        self.cap = {}
        self.qres = {}
        self.pending = []
        self.waiting = {}
        self.seq = 0
        self.t = 0.0

    def queue(self, q, res):
        if q in self.qres:
            return
        merged = {}
        for r, (cap, coef) in res:
            if cap == INF:
                continue
            self.cap[r] = cap
            merged[r] = merged.get(r, 0.0) + coef
        if not merged:
            raise ValueError(f"queue {q} crosses no finite resource")
        self.qres[q] = tuple(sorted(merged.items()))
        self.waiting[q] = []

    def add(self, rel, q, prio, nbytes, lat, done):
        it = _Item()
        it.rel, it.q, it.prio, it.left_bytes = rel, q, prio, float(nbytes)
        it.chunks = max(1, math.ceil(nbytes / self.fab.chunk))
        it.nxt, it.lat, it.done, it.seq = 0, lat, done, self.seq
        self.seq += 1
        heapq.heappush(self.pending, (rel, it.seq, it))

    def _key(self, it):
        if self.policy == "need" and it.prio is not None:
            return (0,) + it.prio + (it.nxt, it.seq)
        return (1, it.rel, it.seq, it.nxt)

    def _rates(self, flows):
        users = {}
        for q in flows:
            for r, a in self.qres[q]:
                users.setdefault(r, []).append((q, a))
        left = {r: self.cap[r] for r in users}
        wsum = {r: sum(a for _, a in us) for r, us in users.items()}
        rate = {}
        unfixed = set(flows)
        while unfixed:
            best = None
            for r, us in users.items():
                if wsum[r] > 1e-300:
                    s = max(0.0, left[r]) / wsum[r]
                    if best is None or s < best[0] or (s == best[0] and r < best[1]):
                        best = (s, r)
            s, r = best
            for q, _a in users[r]:
                if q in unfixed:
                    rate[q] = s
                    unfixed.discard(q)
                    for r2, a2 in self.qres[q]:
                        left[r2] -= a2 * s
                        wsum[r2] -= a2
        return rate

    def run(self):
        active = {}
        t = self.t
        order = sorted(self.qres)
        while True:
            while self.pending and self.pending[0][0] <= t + 1e-12:
                _r, _s, it = heapq.heappop(self.pending)
                heapq.heappush(self.waiting[it.q], (self._key(it), it.seq, it))
            for q in order:
                if q not in active and self.waiting[q]:
                    _k, _s, it = heapq.heappop(self.waiting[q])
                    size = min(float(self.fab.chunk), it.left_bytes)
                    it.left_bytes -= size
                    it.nxt += 1
                    active[q] = [it, size]
            if not active:
                if not self.pending:
                    self.t = t
                    return
                t = self.pending[0][0]
                continue
            flows = sorted(active)
            rate = self._rates(flows)
            dt = min(active[q][1] / rate[q] if rate[q] > 0 else INF for q in flows)
            if self.pending:
                dt = min(dt, self.pending[0][0] - t)
            if dt == INF:
                raise ValueError("stalled: a chunk has no rate")
            t += dt
            for q in flows:
                active[q][1] -= rate[q] * dt
            for q in flows:
                it, rem = active[q]
                if rem > 1e-6 * max(1.0, self.fab.chunk / 2**20):
                    continue
                del active[q]
                it.chunks -= 1
                if it.chunks > 0:
                    heapq.heappush(self.waiting[q], (self._key(it), it.seq, it))
                elif it.done is not None:
                    it.done(t + it.lat)


# ------------------------------------------------------------------------------------------------- stage logic
def _release(st, start, end):
    """Release time of each unit's pieces from one W op [start, end] (module flush, emission order) or at W end
    (whole-stage flush). Units without W compute (the Embed) are released at W start (I17)."""
    nu = len(st.units)
    if st.flush == "stage":
        return [end] * nu
    pfl = sum(u.flops for u in st.units)
    seq = sorted(range(nu), key=lambda i: st.units[i].fwd, reverse=(st.emission == "backward"))
    rel = [start] * nu
    t = start
    for i in seq:
        if st.units[i].flops > 0:
            t += (end - start) * st.units[i].flops / pfl
            rel[i] = t
    return rel


class _Stage:
    def __init__(self, st, fluid, fab):
        self.st, self.fl, self.fab = st, fluid, fab
        w = len(st.peers)
        nu = len(st.units)
        self.w, self.nu = w, nu
        self.contrib = {}                         # (u, d) -> list of (key, ready)
        self.need = {}
        self.done_at = {}                         # (u, d) -> last fold end + AdamW end
        self.ready = {}                           # (u, d) -> latest gather arrival at d
        self.owners = [(u, d) for u in range(nu) for d in range(w) if st.owners[u][d] > 0]
        for key in self.owners:
            self.contrib[key] = []
            self.need[key] = 0
        self.pending_barrier = []
        for q in range(w):
            for r in range(w):
                if q != r:
                    fluid.queue(("x", st.peers[q].name, st.peers[r].name), self._res(q, r))
        for q, lane in enumerate(st.lanes):
            ws = [o for o in lane.ops if o[0] == "W"]
            if st.tree:
                ws = ws[-1:]
            for (_t, start, end, mb) in ws:
                rel = _release(st, start, end)
                for u in range(nu):
                    unit = st.units[u]
                    for d in range(w):
                        own = st.owners[u][d]
                        if own <= 0:
                            continue
                        key = q if st.tree else mb
                        self.need[(u, d)] += 1
                        if d == q:
                            self.contrib[(u, d)].append((key, rel[u]))
                            continue
                        b_g = 4 if st.tree else unit.b_g
                        fluid.add(rel[u], ("x", st.peers[q].name, st.peers[d].name), (unit.fwd, 1, key),
                                  b_g * unit.params * own, self._lat(q, d), self._on_grad(u, d, key))
        if st.boundary > 0:
            for q, p in enumerate(st.peers):
                dram = (fab.host_dram.get(p.host, INF), fab.k)
                port = min(p.nic, p.pcie)
                fluid.queue(("bo", p.name), ((("e", p.name), (port, 1.0)), (("dram", p.host), dram)))
                fluid.queue(("bi", p.name), ((("i", p.name), (port, 1.0)), (("dram", p.host), dram)))
                for o in st.lanes[q].ops:
                    if o[0] == "F":
                        fluid.add(o[2], ("bo", p.name), None, st.boundary, 0.0, None)
                        fluid.add(o[2], ("bi", p.name), None, st.boundary, 0.0, None)
        self.complete_initial()

    def complete_initial(self):
        for key in self.owners:
            if self.need[key] == len(self.contrib[key]):
                self._complete(*key)

    def _res(self, q, r):
        a, b = self.st.peers[q], self.st.peers[r]
        fab = self.fab
        res = [(("e", a.name), (min(a.nic, a.pcie), 1.0)), (("i", b.name), (min(b.nic, b.pcie), 1.0))]
        res.append((("dram", a.host), (fab.host_dram.get(a.host, INF), fab.k)))
        res.append((("dram", b.host), (fab.host_dram.get(b.host, INF), fab.k)))
        if a.site != b.site:
            res.append((("up", a.site, b.site), (fab.uplinks[(a.site, b.site)], 1.0)))
        return res

    def _lat(self, q, r):
        a, b = self.st.peers[q], self.st.peers[r]
        rtt = self.fab.site_rtt if a.site != b.site else self.fab.rtt
        return rtt / 2 + self.fab.chunk / a.pcie + self.fab.chunk / b.pcie

    def _on_grad(self, u, d, key):
        def done(arr):
            self.contrib[(u, d)].append((key, arr))
            if len(self.contrib[(u, d)]) == self.need[(u, d)]:
                self._complete(u, d)
        return done

    def _chain_end(self, u, d):
        """The owner's left fold of every contribution in canonical order (microbatch index, or peer index for the
        tree), each fold starting when the previous one ends and its contribution is there; then AdamW."""
        st = self.st
        unit = st.units[u]
        n = unit.params * st.owners[u][d]
        b_g = 4 if st.tree else unit.b_g
        t = -INF
        for _k, rdy in sorted(self.contrib[(u, d)]):
            t = max(t, rdy) + st.rates.fold_t(n, b_g)
        return t + st.rates.adam_t(n)

    def _complete(self, u, d):
        end = self._chain_end(u, d)
        self.done_at[(u, d)] = end
        if not self.st.barrier:
            self._gathers(u, d, end)
            return
        self.pending_barrier.append((u, d))
        if len(self.pending_barrier) == len(self.owners):
            self.tb = max(self.done_at.values())
            for (uu, oo) in self.pending_barrier:
                self._gathers(uu, oo, self.tb)

    def _gathers(self, u, o, t):
        unit = self.st.units[u]
        for d in range(self.w):
            if d != o:
                self.fl.add(t, ("x", self.st.peers[o].name, self.st.peers[d].name), (unit.fwd, 0, -1),
                            unit.b_p * unit.params * self.st.owners[u][o], self._lat(o, d), self._on_gather(u, d))

    def _on_gather(self, u, d):
        def done(arr):
            self.ready[(u, d)] = max(self.ready.get((u, d), -INF), arr)
        return done

    def _late(self, d, base):
        st = self.st
        pfl = sum(un.flops for un in st.units)
        order = sorted(range(self.nu), key=lambda i: st.units[i].fwd)
        tb = getattr(self, "tb", -INF) if st.barrier else -INF
        lane = st.lanes[d]
        D = 0.0
        fcum = 0.0
        for u in order:
            rdy = self.ready.get((u, d), -INF)
            if st.owners[u][d] > 0:
                rdy = max(rdy, self.done_at[(u, d)])
            rdy = max(rdy, tb)
            need = base + D + fcum
            if rdy > need:
                D += rdy - need
            if pfl > 0:
                fcum += lane.cost_f * st.units[u].flops / pfl
        return D

    def result(self):
        st = self.st
        E = max(self._late(d, ln.end) for d, ln in enumerate(st.lanes))
        slot = max(self._late(d, st.period + ln.first_f) for d, ln in enumerate(st.lanes))
        window = min(st.period + ln.first_f - ln.end for ln in st.lanes)
        return dict(E=E, window=window, exposed=max(0.0, E - window), slot_delay=slot, overhead=overhead(st))


def overhead(st):
    """T_ovh: the slowest peer's device time for M folds of its owned elements plus their AdamW (I17)."""
    worst = 0.0
    for d in range(len(st.peers)):
        t = 0.0
        for u, unit in enumerate(st.units):
            n = unit.params * st.owners[u][d]
            if n > 0:
                t += st.M * st.rates.fold_t(n, unit.b_g) + st.rates.adam_t(n)
        worst = max(worst, t)
    return worst


def groups(stages, fab):
    """Stages that share a finite resource (a host's DRAM, an uplink) simulate together; the rest apart (max-min
    sharing decomposes exactly over components with disjoint resources)."""
    parent = list(range(len(stages)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen = {}
    for i, st in enumerate(stages):
        keys = set()
        for p in st.peers:
            if fab.host_dram.get(p.host, INF) != INF:
                keys.add(("h", p.host))
        sites = sorted({p.site for p in st.peers})
        keys |= {("u", a, b) for a in sites for b in sites if a != b}
        for key in sorted(keys):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    out = {}
    for i in range(len(stages)):
        out.setdefault(find(i), []).append(i)
    return [out[r] for r in sorted(out)]


def simulate(stages, fab, emission="forward", policy="need"):
    """One simulation of every sharded stage in `stages` (StageIn list) -> list of per-stage result dicts."""
    res = [None] * len(stages)
    for grp in groups(stages, fab):
        fl = Fluid(fab, policy)
        sims = {}
        for i in grp:
            stages[i].emission = emission
            sims[i] = _Stage(stages[i], fl, fab)
        fl.run()
        for i in grp:
            res[i] = sims[i].result()
    return res
