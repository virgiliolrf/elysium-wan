"""Slot-level exchange simulation of sharded stages (SCHEDULER-I17, ADR-0006).

The peers of a wide stage send each unit's gradient pieces to its owners and
receive the owners' gathers back. This module predicts what that costs per
step, modelled the way the runtime sends it (RUNTIME-I13):

- one send queue per (sender, destination). The queue is served need-first:
  the unit the next forward needs first, a gather before a gradient of the
  same unit, an earlier microbatch first. FIFO is E4's control. A queue sends
  one chunk of at most `Fabric.chunk` bytes at a time;
- every peer's egress and ingress runs at min(NIC, PCIe). Every host's DRAM
  costs k passes per byte per side, and every site uplink is counted per
  direction. All of them are shared max-min fairly among the chunks in flight.
  A transfer arrives 2·f/PCIe (one copy per side) plus RTT/2 after its last
  byte leaves;
- a W releases its microbatch's pieces unit by unit, in the emission order, at
  the op times `simulate_slots` gives each peer. A unit with no W compute (the
  Embed's tables) is released when W starts. Under the whole-stage flush
  (`flush="stage"`, today's runtime) every unit of W(mb) is released at W's end;
- an owner left-folds a unit's contributions in canonical microbatch order,
  each fold starting at max(the previous fold's end, the contribution's
  arrival), then runs the unit's AdamW, at their HBM op times; the unit's
  gathers leave at the end of that chain. Under the step barrier every gather
  leaves at Tb, the latest chain end of the stage, and step t+1's F waits for
  Tb. Step t+1's F on a unit waits for that unit's gather.

Two runs of the same model give the two terms of ADR-0006's exposure
(derivation §2):
- `tail`, X_s: only each peer's last W, with the next forward needed as soon
  as the stage's compute ends (its last peer's end). A faster peer waiting for
  a slower one is the lane bubble β, already in the period (ADR-0006 T_step =
  C·(1 + β) + E − window_s + T_ovh), never exchange exposure; `window_s` is
  measured from that same end;
- `delay`: every microbatch's traffic, boundary bytes included, against the
  period. This is how late step t+1's first F starts beyond its slot. When the
  links bind it is V/cap + RTT + t_gap − C.

E_s is the larger of the two. The predicted step time is the period, plus the
largest delay, plus the slowest peer's fold and AdamW device time.

Pure: no I/O and no randomness. The same inputs give the same floats.
"""

from __future__ import annotations

import bisect
import heapq
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from wantrain.scheduler.zero_bubble import F, W, ZBConfig, simulate_slots

CHUNK_BYTES = 16 * 2**20
ADAMW_BYTES = 30  # read p, g, m, v; write p, m, v; write the cast copy
EMISSIONS = ("forward", "backward")
POLICIES = ("need", "fifo")
LAYOUTS = ("pieces", "zero1", "bounds")
FLUSHES = ("module", "stage")
_T_EPS = 1e-12  # seconds: a release this close to now is released now
_B_EPS = 1e-6  # bytes: a chunk with less left has finished


def fold_bytes(b_g: int) -> int:
    """HBM bytes per folded element: widen to fp32 (read b_g, write 4) unless
    the wire is fp32, then add into the fp32 sum (read 4 + 4, write 4)."""
    return b_g + 16 if b_g < 4 else 12


@dataclass(frozen=True)
class Unit:
    """One exchange unit: a `SplitLinear` module, or the Embed's tables.

    `flops` weights the unit's share of F and W compute; 0 means the unit is
    released when W(mb) starts and costs no forward time (the Embed).
    `lane` is gradient-only bytes per microbatch riding the unit's GRADS
    frame: an MoE block's int64 count lane, E·8 bytes (RUNTIME-I18,
    SCHEDULER-I21); 0 for every dense unit."""

    params: int
    b_g: int
    b_p: int
    fwd_index: int
    flops: float
    lane: int = 0

    def grad_bytes(self) -> float:
        """One microbatch's gradient bytes of the whole unit: b_g·P_u + lane."""
        return float(self.b_g * self.params + self.lane)


@dataclass(frozen=True)
class Peer:
    """One GPU of a stage. Rates are bytes per second, per direction."""

    name: str
    nic: float
    pcie: float
    hbm: float
    host: str
    site: str


@dataclass(frozen=True)
class StageLayout:
    """A sharded stage: its units, which fraction of each unit each peer owns
    (owners[u][q], peer q = slot q of the stage), and the peers.

    `barrier` defaults to on for zero1 and off for pieces and explicit bounds.
    `boundary_bytes` are the pipeline-boundary bytes per microbatch per
    direction. They load the links but do not gate ops: their timing is
    already in the slot timeline. `flush` is "module" (each unit released as
    W computes it, SCHEDULER-I17) or "stage" (every unit at W's end, today's
    whole-stage flush, where the emission order is irrelevant)."""

    units: tuple[Unit, ...]
    owners: tuple[tuple[float, ...], ...]
    peers: tuple[Peer, ...]
    layout: str = "pieces"
    barrier: bool | None = None
    boundary_bytes: float = 0.0
    flush: str = "module"

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {self.layout!r}")
        if self.flush not in FLUSHES:
            raise ValueError(f"flush must be one of {FLUSHES}, got {self.flush!r}")
        if not self.units:
            raise ValueError("a stage needs at least one unit")
        if len(self.owners) != len(self.units):
            raise ValueError("owners needs one row per unit")
        width = len(self.peers)
        if width < 1 or len({p.name for p in self.peers}) != width:
            raise ValueError("a stage needs at least one peer, with distinct names")
        if len({u.fwd_index for u in self.units}) != len(self.units):
            raise ValueError("forward indices must be distinct")
        for u, row in zip(self.units, self.owners, strict=True):
            if u.params <= 0 or u.b_g <= 0 or u.b_p <= 0 or u.flops < 0 or u.lane < 0:
                raise ValueError(
                    f"unit {u.fwd_index}: params and bytes must be > 0, flops and lane >= 0"
                )
            if len(row) != width or min(row) < 0 or abs(sum(row) - 1.0) > 1e-9:
                raise ValueError(
                    f"unit {u.fwd_index}: owner fractions must be >= 0, one per peer, summing to 1"
                )
        if self.boundary_bytes < 0:
            raise ValueError("boundary_bytes must be >= 0")
        for p in self.peers:
            if min(p.nic, p.pcie, p.hbm) <= 0:
                raise ValueError(f"peer {p.name}: rates must be > 0")

    @property
    def step_barrier(self) -> bool:
        return self.layout == "zero1" if self.barrier is None else self.barrier


@dataclass(frozen=True)
class Fabric:
    """What the stages share. `host_dram` has every host's DRAM rate (bytes/s,
    both directions together), `uplinks` every (site, site) direction a
    transfer crosses. `math.inf` means unconstrained. `rtt` is the round trip
    inside a site; `site_rtt` the round trip between two sites, where known."""

    host_dram: Mapping[str, float]
    uplinks: Mapping[tuple[str, str], float] = field(default_factory=dict)
    k_passes: float = 10.0
    rtt: float = 0.0
    chunk: int = CHUNK_BYTES
    site_rtt: Mapping[tuple[str, str], float] = field(default_factory=dict)


@dataclass(frozen=True)
class StagePrediction:
    exposure: float  # E_s = max(tail, delay)
    tail: float  # X_s
    delay: float  # step t+1's first F late beyond its slot in the period
    window: float  # idle time between the stage's compute end and step t+1's F
    overhead: float  # fold of every microbatch + AdamW, slowest peer


@dataclass(frozen=True)
class ExchangePrediction:
    emission: str
    period: float
    step_time: float
    stages: Mapping[int, StagePrediction]


@dataclass(frozen=True)
class EmissionChoice:
    forward: ExchangePrediction
    backward: ExchangePrediction
    best: str

    @property
    def chosen(self) -> ExchangePrediction:
        return self.forward if self.best == "forward" else self.backward


def pieces_owners(n_units: int, width: int) -> tuple[tuple[float, ...], ...]:
    """Every unit split in `width` equal pieces."""
    return tuple((1.0 / width,) * width for _ in range(n_units))


def stride_owners(
    tensors: Sequence[Sequence[tuple[int, int]]], width: int
) -> tuple[tuple[float, ...], ...]:
    """Today's stride ownership of units made of several whole tensors
    (runtime/stage.py `params[peer::w]`): each unit's tensors are given as
    (elements, index in the stage's parameter list); tensor i sits whole on
    peer i % width, so a unit's row is the share of its elements per peer
    (the Embed: tok, index 0, and pos, index 1, on different peers)."""
    out = []
    for unit in tensors:
        total = sum(n for n, _i in unit)
        row = [0.0] * width
        for n, i in unit:
            row[i % width] += n / total
        out.append(tuple(row))
    return tuple(out)


def bounds_owners(params: Sequence[int], edges: Sequence[int]) -> tuple[tuple[float, ...], ...]:
    """Explicit element bounds over the units concatenated in forward order:
    peer q owns elements [edges[q], edges[q + 1])."""
    total = sum(params)
    if (
        edges[0] != 0
        or edges[-1] != total
        or any(b < a for a, b in zip(edges, edges[1:], strict=False))
    ):
        raise ValueError("edges must rise from 0 to the stage's element count")
    out = []
    lo = 0
    for p in params:
        hi = lo + p
        out.append(
            tuple(
                max(0, min(hi, edges[q + 1]) - max(lo, edges[q])) / p for q in range(len(edges) - 1)
            )
        )
        lo = hi
    return tuple(out)


# --------------------------------------------------------------------- engine


class _Item:
    __slots__ = ("rel", "queue", "prio", "unsent", "chunks_left", "next_chunk", "lat", "done")

    def __init__(self, rel, queue, prio, nbytes, nchunks, lat, done):
        self.rel = rel
        self.queue = queue
        self.prio = prio
        self.unsent = nbytes
        self.chunks_left = nchunks
        self.next_chunk = 0
        self.lat = lat
        self.done = done


class _Links:
    """Fluid model of chunks in flight: one chunk per queue, max-min fair rates
    over every resource a queue's chunks cross (resource coefficients count
    bytes of the resource per byte sent)."""

    def __init__(self, fabric: Fabric, policy: str):
        self.fabric = fabric
        self.policy = policy
        self.pending: list = []  # (rel, seq, item)
        self.waiting: dict = {}  # queue -> heap of (key, seq, item)
        self.qres: dict = {}  # queue -> ((resource, coefficient), ...)
        self.cap: dict = {}
        self.order: list = []  # every queue, sorted
        self.seq = 0

    def queue(self, qkey: tuple, res: Mapping[tuple, tuple[float, float]]) -> None:
        """Declare a queue: res maps resource -> (capacity, bytes per byte sent)."""
        if qkey in self.qres:
            return
        merged: dict = {}
        for r, (cap, coef) in res.items():
            if math.isinf(cap):
                continue
            self.cap[r] = cap
            merged[r] = merged.get(r, 0.0) + coef
        self.qres[qkey] = tuple(merged.items())
        self.waiting[qkey] = []
        bisect.insort(self.order, qkey)

    def add(
        self, rel: float, qkey: tuple, prio: tuple, nbytes: float, lat: float, done: Callable | None
    ) -> None:
        chunk = self.fabric.chunk
        n = max(1, math.ceil(nbytes / chunk))
        item = _Item(rel, qkey, prio, float(nbytes), n, lat, done)
        heapq.heappush(self.pending, (rel, self.seq, item))
        self.seq += 1

    def _key(self, item: _Item, seq: int) -> tuple:
        if self.policy == "need" and item.prio:
            return (*item.prio, item.next_chunk)
        return (item.rel, seq, item.next_chunk)

    def _rates(self, flows: list) -> dict:
        users: dict = {}
        for q in flows:
            for r, a in self.qres[q]:
                users.setdefault(r, []).append((q, a))
        left = {r: self.cap[r] for r in users}
        rate: dict = {}
        unfixed = set(flows)
        for q in flows:
            if not self.qres[q]:
                raise ValueError(f"queue {q} crosses no finite resource")
        while unfixed:
            best = None
            for r, us in users.items():
                wsum = sum(a for q, a in us if q in unfixed)
                if wsum > 0:
                    share = max(0.0, left[r]) / wsum
                    if best is None or share < best[0]:
                        best = (share, r)
            assert best is not None
            share, r = best
            for q, _a in users[r]:
                if q in unfixed:
                    rate[q] = share
                    unfixed.discard(q)
                    for r2, a2 in self.qres[q]:
                        left[r2] -= a2 * share
        return rate

    def run(self) -> None:
        t = 0.0
        active: dict = {}  # queue -> [item, bytes left of the chunk]
        while True:
            while self.pending and self.pending[0][0] <= t + _T_EPS:
                _rel, seq, item = heapq.heappop(self.pending)
                heapq.heappush(self.waiting[item.queue], (self._key(item, seq), seq, item))
            for q in self.order:
                if q not in active and self.waiting[q]:
                    _k, seq, item = heapq.heappop(self.waiting[q])
                    size = min(float(self.fabric.chunk), item.unsent)
                    item.unsent -= size
                    item.next_chunk += 1
                    active[q] = [item, size, seq]
            if not active:
                if not self.pending:
                    return
                t = self.pending[0][0]
                continue
            flows = sorted(active)
            rate = self._rates(flows)
            dt = min(active[q][1] / rate[q] if rate[q] > 0 else math.inf for q in flows)
            if self.pending:
                dt = min(dt, self.pending[0][0] - t)
            if math.isinf(dt):
                raise ValueError("exchange stalled: a chunk has no rate")
            t += dt
            for q in flows:
                active[q][1] -= rate[q] * dt
            for q in flows:
                item, rem, seq = active[q]
                if rem > _B_EPS:
                    continue
                del active[q]
                item.chunks_left -= 1
                if item.chunks_left > 0:
                    heapq.heappush(self.waiting[q], (self._key(item, seq), seq, item))
                elif item.done is not None:
                    item.done(t + self.fabric.rtt / 2 + item.lat)


# --------------------------------------------------------------------- stages


@dataclass
class _Lane:
    first_f: float  # start of the peer's first F
    end: float  # end of the peer's last op
    cost_f: float  # one F on this peer
    ws: list  # [(start, end, mb)] of the peer's W ops, by end time
    fends: list  # end of every F op: when its boundary bytes move


def _timeline(cfg: ZBConfig, slot_orders, peer_weights, stages) -> tuple[dict, float]:
    ends, _makespan = simulate_slots(
        cfg, slot_orders, [[] for _ in range(cfg.n_stages)], peer_weights
    )
    lanes: dict = {}
    last = 0.0
    for s in range(cfg.n_stages):
        for q, order in enumerate(slot_orders[s]):
            scale = sum(peer_weights[s]) / peer_weights[s][q]
            ops = []
            for typ, mb in order:
                fin = ends[(typ, s, mb)]
                ops.append((fin - cfg.cost(typ, s) * scale, fin, typ, mb))
            if ops:
                last = max(last, max(o[1] for o in ops))
            if s in stages:
                fs = [o[0] for o in ops if o[2] == F]
                lanes[(s, q)] = _Lane(
                    first_f=min(fs) if fs else 0.0,
                    end=max((o[1] for o in ops), default=0.0),
                    cost_f=cfg.cost(F, s) * scale,
                    ws=sorted(
                        ((o[0], o[1], o[3]) for o in ops if o[2] == W), key=lambda x: (x[1], x[2])
                    ),
                    fends=sorted(o[1] for o in ops if o[2] == F),
                )
    return lanes, last


def _releases(st: StageLayout, start: float, end: float, emission: str) -> list[float]:
    """Release time of every unit's pieces from one W op [start, end]."""
    if st.flush == "stage":
        return [end] * len(st.units)
    pfl = sum(u.flops for u in st.units)
    seq = sorted(
        range(len(st.units)), key=lambda i: st.units[i].fwd_index, reverse=emission == "backward"
    )
    rel = [start] * len(st.units)
    t = start
    for i in seq:
        if st.units[i].flops > 0:
            t += (end - start) * st.units[i].flops / pfl
            rel[i] = t
    return rel


class _Stage:
    """One stage's grads, folds, gathers and the next forward's waits."""

    def __init__(
        self,
        s: int,
        st: StageLayout,
        lanes: dict,
        links: _Links,
        fabric: Fabric,
        tail: bool,
        emission,
    ):
        self.st = st
        self.links = links
        self.fabric = fabric
        self.barrier = st.step_barrier
        w = len(st.peers)
        nu = len(st.units)
        self.lanes = [lanes[(s, q)] for q in range(w)]
        self.contrib: dict = {}  # (unit, owner) -> [(mb, arrival)] of every contribution
        self.need: dict = {}
        self.done_at: dict = {}  # (unit, owner) -> end of its left fold and AdamW
        self.tb = 0.0  # under the barrier: the latest chain end of the stage
        self.ready: dict = {}
        self.owners = [(u, d) for u in range(nu) for d in range(w) if st.owners[u][d] > 0]
        for key in self.owners:
            self.contrib[key] = []
            self.need[key] = 0
        self.done_owners: list = []
        for q in range(w):
            for r in range(w):
                if q != r:
                    links.queue(("x", st.peers[q].name, st.peers[r].name), self._res(q, r))
        for q, lane in enumerate(self.lanes):
            ws = lane.ws[-1:] if tail else lane.ws
            for start, end, mb in ws:
                rel = _releases(st, start, end, emission)
                for u in range(nu):
                    unit = st.units[u]
                    for d in range(w):
                        own = st.owners[u][d]
                        if own <= 0:
                            continue
                        self.need[(u, d)] += 1
                        if d == q:
                            self.contrib[(u, d)].append((mb, rel[u]))
                            continue
                        links.add(
                            rel[u],
                            ("x", st.peers[q].name, st.peers[d].name),
                            (unit.fwd_index, 1, mb),
                            unit.grad_bytes() * own,
                            self._lat(q, d),
                            self._on_grad(u, d, mb),
                        )
        if not tail and st.boundary_bytes > 0:
            self._boundary()
        for key in self.owners:
            if len(self.contrib[key]) == self.need[key]:
                self._complete(*key)

    def _res(self, q: int, r: int) -> dict:
        a, b = self.st.peers[q], self.st.peers[r]
        fab = self.fabric
        res: dict = {
            ("e", a.name): (min(a.nic, a.pcie), 1.0),
            ("i", b.name): (min(b.nic, b.pcie), 1.0),
        }
        for host in (a.host, b.host):
            if host not in fab.host_dram:
                raise ValueError(f"host {host!r} has no DRAM rate in the fabric")
            cap, coef = res.get(("dram", host), (fab.host_dram[host], 0.0))
            res[("dram", host)] = (cap, coef + fab.k_passes)
        if a.site != b.site:
            if (a.site, b.site) not in fab.uplinks:
                raise ValueError(f"no uplink {a.site!r} -> {b.site!r} in the fabric")
            res[("up", a.site, b.site)] = (fab.uplinks[(a.site, b.site)], 1.0)
        return res

    def _lat(self, q: int, r: int) -> float:
        a, b = self.st.peers[q], self.st.peers[r]
        fab = self.fabric
        extra = (
            (fab.site_rtt[(a.site, b.site)] - fab.rtt) / 2
            if (a.site, b.site) in fab.site_rtt
            else 0.0
        )
        return fab.chunk / a.pcie + fab.chunk / b.pcie + extra

    def _boundary(self) -> None:
        fab = self.fabric
        for q, p in enumerate(self.st.peers):
            if p.host not in fab.host_dram:
                raise ValueError(f"host {p.host!r} has no DRAM rate in the fabric")
            port = min(p.nic, p.pcie)
            dram = (fab.host_dram[p.host], fab.k_passes)
            self.links.queue(
                ("bo", p.name, ""), {("e", p.name): (port, 1.0), ("dram", p.host): dram}
            )
            self.links.queue(
                ("bi", p.name, ""), {("i", p.name): (port, 1.0), ("dram", p.host): dram}
            )
            for t in self.lanes[q].fends:
                self.links.add(t, ("bo", p.name, ""), (), self.st.boundary_bytes, 0.0, None)
                self.links.add(t, ("bi", p.name, ""), (), self.st.boundary_bytes, 0.0, None)

    def _on_grad(self, u: int, d: int, mb: int) -> Callable[[float], None]:
        def done(arr: float) -> None:
            self.contrib[(u, d)].append((mb, arr))
            if len(self.contrib[(u, d)]) == self.need[(u, d)]:
                self._complete(u, d)

        return done

    def _chain_end(self, u: int, o: int) -> float:
        """The owner's left fold of every contribution in canonical microbatch
        order, each fold starting when the previous one ends and its
        contribution is there, then the unit's AdamW (SCHEDULER-I17)."""
        unit = self.st.units[u]
        n = unit.params * self.st.owners[u][o]
        hbm = self.st.peers[o].hbm
        fold = n * fold_bytes(unit.b_g) / hbm
        t = 0.0
        for _mb, arr in sorted(self.contrib[(u, o)]):
            t = max(t, arr) + fold
        return t + n * ADAMW_BYTES / hbm

    def _complete(self, u: int, o: int) -> None:
        end = self._chain_end(u, o)
        self.done_at[(u, o)] = end
        if not self.barrier:
            self._gathers(u, o, end)
            return
        self.done_owners.append((u, o))
        if len(self.done_owners) == len(self.owners):
            self.tb = max(self.done_at.values())
            for uu, oo in self.done_owners:
                self._gathers(uu, oo, self.tb)

    def _gathers(self, u: int, o: int, t: float) -> None:
        unit = self.st.units[u]
        for d in range(len(self.st.peers)):
            if d != o:
                self.links.add(
                    t,
                    ("x", self.st.peers[o].name, self.st.peers[d].name),
                    (unit.fwd_index, 0, -1),
                    unit.b_p * unit.params * self.st.owners[u][o],
                    self._lat(o, d),
                    self._on_gather(u, d),
                )

    def _on_gather(self, u: int, d: int) -> Callable[[float], None]:
        def done(arr: float) -> None:
            self.ready[(u, d)] = max(self.ready.get((u, d), 0.0), arr)

        return done

    def late(self, base: Callable[[_Lane], float]) -> float:
        """How late the next forward's units are on the slowest peer, against
        the forward starting at base(lane) (D accumulates, derivation §2)."""
        st = self.st
        pfl = sum(u.flops for u in st.units)
        order = sorted(range(len(st.units)), key=lambda i: st.units[i].fwd_index)
        top = self.tb if self.barrier else 0.0
        worst = 0.0
        for d, lane in enumerate(self.lanes):
            delay = 0.0
            fcum = 0.0
            for u in order:
                rdy = self.ready.get((u, d), 0.0)
                if st.owners[u][d] > 0:
                    rdy = max(rdy, self.done_at[(u, d)])
                rdy = max(rdy, top)
                need = base(lane) + delay + fcum
                if rdy > need:
                    delay += rdy - need
                if pfl > 0:
                    fcum += lane.cost_f * st.units[u].flops / pfl
            worst = max(worst, delay)
        return worst

    def overhead(self) -> float:
        st = self.st
        n_mb = sum(len(lane.ws) for lane in self.lanes)
        worst = 0.0
        for d, p in enumerate(st.peers):
            fold = sum(
                u.params * st.owners[i][d] * n_mb * fold_bytes(u.b_g)
                for i, u in enumerate(st.units)
            )
            adam = sum(u.params * st.owners[i][d] * ADAMW_BYTES for i, u in enumerate(st.units))
            worst = max(worst, (fold + adam) / p.hbm)
        return worst


# --------------------------------------------------------------------- entry points


def _groups(stages: Mapping[int, StageLayout]) -> list[list[int]]:
    """The stages that share a resource (a host's DRAM, or an uplink between
    the same sites), transitively. Max-min sharing decomposes exactly over
    groups with disjoint resources, so each runs on its own links: the same
    model, without one fluid engine over every chunk of the pipeline."""
    parent = {s: s for s in stages}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen: dict = {}
    for s, st in sorted(stages.items()):
        keys: set = {("h", p.host) for p in st.peers}
        sites = sorted({p.site for p in st.peers})
        keys |= {("u", a, b) for a in sites for b in sites if a != b}
        for key in sorted(keys):
            if key in seen:
                parent[find(s)] = find(seen[key])
            else:
                seen[key] = s
    out: dict = {}
    for s in sorted(stages):
        out.setdefault(find(s), []).append(s)
    return [out[r] for r in sorted(out)]


def simulate_exchange(
    cfg: ZBConfig,
    slot_orders: Sequence[Sequence[Sequence[tuple[str, int]]]],
    peer_weights: Sequence[Sequence[float]],
    stages: Mapping[int, StageLayout],
    fabric: Fabric,
    *,
    emission: str = "forward",
    policy: str = "need",
    period: float | None = None,
) -> ExchangePrediction:
    """Predict one step of the pipeline given by (cfg, slot_orders,
    peer_weights) with the sharded stages in `stages`, every W emitting its
    units in `emission` order. `period` pads the step: step t+1 starts at
    max(period, the last op's end). Stages not in `stages` exchange nothing."""
    if emission not in EMISSIONS:
        raise ValueError(f"emission must be one of {EMISSIONS}, got {emission!r}")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if fabric.chunk <= 0 or fabric.k_passes < 0 or fabric.rtt < 0:
        raise ValueError("chunk must be > 0, k_passes and rtt >= 0")
    if any(v < fabric.rtt for v in fabric.site_rtt.values()):
        # _lat adds (site_rtt - rtt) / 2: a site pair closer than one site would
        # subtract latency from the intra-site transfer it extends
        raise ValueError("site_rtt must be >= rtt (and >= 0)")
    for s, st in stages.items():
        if not 0 <= s < cfg.n_stages or len(slot_orders[s]) != len(st.peers):
            raise ValueError(f"stage {s}: one peer per slot order is needed")
    names = [p.name for st in stages.values() for p in st.peers]
    if len(set(names)) != len(names):
        raise ValueError("a peer belongs to one stage")
    lanes, last = _timeline(cfg, slot_orders, peer_weights, stages)
    step = max(last, period or 0.0)

    groups = _groups(stages)

    def run(tail: bool) -> dict:
        sims = {}
        for group in groups:  # stages sharing no resource never interact
            links = _Links(fabric, policy)
            for s in group:
                sims[s] = _Stage(s, stages[s], lanes, links, fabric, tail, emission)
            links.run()
        return sims

    tails = run(tail=True)
    fulls = run(tail=False)
    out = {}
    for s in sorted(stages):
        stage_end = max(lane.end for lane in tails[s].lanes)
        x = tails[s].late(lambda lane, end=stage_end: end)
        late = fulls[s].late(lambda lane: step + lane.first_f)
        window = min(step + lane.first_f - lane.end for lane in fulls[s].lanes)
        out[s] = StagePrediction(
            exposure=max(x, late), tail=x, delay=late, window=window, overhead=fulls[s].overhead()
        )
    worst_delay = max((p.delay for p in out.values()), default=0.0)
    worst_ovh = max((p.overhead for p in out.values()), default=0.0)
    return ExchangePrediction(
        emission=emission, period=step, step_time=step + worst_delay + worst_ovh, stages=out
    )


def predict_exchange(
    cfg: ZBConfig,
    slot_orders: Sequence[Sequence[Sequence[tuple[str, int]]]],
    peer_weights: Sequence[Sequence[float]],
    stages: Mapping[int, StageLayout],
    fabric: Fabric,
    *,
    policy: str = "need",
    period: float | None = None,
) -> EmissionChoice:
    """Both emission orders, and the one with the smaller predicted step time
    (forward on a tie)."""
    fwd, bwd = (
        simulate_exchange(
            cfg, slot_orders, peer_weights, stages, fabric, emission=e, policy=policy, period=period
        )
        for e in EMISSIONS
    )
    return EmissionChoice(
        forward=fwd, backward=bwd, best="backward" if bwd.step_time < fwd.step_time else "forward"
    )
