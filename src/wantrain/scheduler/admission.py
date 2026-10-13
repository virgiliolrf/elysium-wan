"""Admission of wide and sharded stages (SCHEDULER-I16, ADR-0006).

A stage of width w may run `shard: "pieces"` only where five conditions hold,
each on measured inputs: one probed device class, S1 on every link, S2 on the
exposure the exchange simulation predicts (SCHEDULER-I17), device memory per
peer and host memory per host (MEM-HOST). Every refusal is a `Reason` that
names the link, number, class, peer or host that refused.

Units: rates are bytes per second per direction, sizes bytes, times seconds.
Pure: no I/O, no randomness; the planner and the coordinator's re-checks
(BA-44, BA-47, BA-50) call the same functions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from wantrain.scheduler.exchange_sim import (
    ADAMW_BYTES,
    Fabric,
    Peer,
    StageLayout,
    StagePrediction,
    Unit,
    fold_bytes,
    simulate_exchange,
)
from wantrain.scheduler.zero_bubble import ZBConfig

# Host-DRAM passes per sent byte per side on the data plane (SCHEDULER-I16:
# "15 today on the sealed M2 data plane"). Counted, never measured: SC-5 counted
# 12 passes on the sender and 18 on the receiver on the worst default path, and
# S1's factor 2 sums both sides, so k is their mean
# (docs/research/adr-0006/sc5-passes.md; 10 before sealing).
K_HOST_DRAM_COUNTED = 15
# With adaptive microbatching (JobSpec.adapt_budget_ms > 0) exchange frames are
# fragmented: +2 passes on each side (SC-5: sender 14, receiver 20), so k is 17.
K_HOST_DRAM_COUNTED_ADAPTIVE = 17


def counted_k(adapt_budget_ms: float | None) -> int:
    """The transport's counted k for a job (SC-5): 17 when adaptive
    microbatching fragments the exchange frames, 15 otherwise."""
    return K_HOST_DRAM_COUNTED_ADAPTIVE if (adapt_budget_ms or 0) > 0 else K_HOST_DRAM_COUNTED


EPSILON = 0.03  # S2's bound on E - window_s, a fraction of C (ADR-0006 question 1)
OWNED_STATE_BYTES = 16  # fp32 master, m, v and the fold accumulator per owned element
LOGITS_BYTES = 4  # fp32 logits and their gradient on the last stage
# Sources a capacity may carry (BA-48). Pieces admits only probed figures:
# a declared or missing one refuses, naming the field.
SOURCES = ("measured", "declared", "missing", "absent", "counted", "scenario")
PIECES_SOURCES = ("measured", "absent")
K_SOURCES = ("counted", "scenario")
# Devices that are their own host: no PCIe hop (its capacity is 'absent') and
# no pinned copies. 'absent' is a valid source for nothing else.
HOST_DEVICES = ("cpu", "mps")


def is_host_device(device: str) -> bool:
    return device.split(":")[0] in HOST_DEVICES


@dataclass(frozen=True)
class Gpu:
    """One device of the fleet as admission sees it. Capacities are bytes/s
    per direction (None: not measured); `sources` maps each capacity field
    (nic_up, nic_down, pcie_h2d, pcie_d2h, hbm) to where it came from."""

    name: str
    host: str
    site: str
    device: str
    gpu_name: str
    tflops: float
    vram_bytes: int
    nic_up: float | None = None
    nic_down: float | None = None
    pcie_h2d: float | None = None
    pcie_d2h: float | None = None
    hbm: float | None = None
    declared: frozenset = frozenset()
    sources: Mapping[str, str] = field(default_factory=dict)

    def source(self, name: str) -> str:
        return _source(getattr(self, name), self.sources.get(name))


@dataclass(frozen=True)
class Host:
    """One machine: one RAM and one DRAM figure (bytes, bytes/s both
    directions together) whatever the number of its devices (BA-48)."""

    name: str
    ram_bytes: int | None = None
    dram: float | None = None
    sources: Mapping[str, str] = field(default_factory=dict)

    def source(self, name: str) -> str:
        return _source(getattr(self, name), self.sources.get(name))


@dataclass(frozen=True)
class Stage:
    """A stage as placed: its exchange units (forward order), which fraction
    of each unit each peer owns, its peers (slot order), and what one
    microbatch costs it: `flops_mb` (F + B + W, the planner's costs),
    `boundary_mb` (A, pipeline-boundary bytes per direction), `tokens_mb`
    (T_mb), `boundary_bytes` (b_boundary per value), `logits_bytes` (the
    head's fp32 logits and gradient, last stage only). (M4) `pending_w_mb`
    replaces the builtin block's pending-W bytes per microbatch where the
    model knows its own (MoE blocks, SCHEDULER-I21); `bank_elems` is
    P_bank,s, the stage's 3-D expert elements (HF banks), whose gradients at
    `bank_bytes` (b_m) per element are held from B's end to W's start."""

    index: int
    units: tuple[Unit, ...]
    owners: tuple[tuple[float, ...], ...]
    peers: tuple[Gpu, ...]
    flops_mb: float
    boundary_mb: float
    tokens_mb: int
    d_model: int
    n_blocks: int
    boundary_bytes: int = 2
    logits_bytes: float = 0.0
    layout: str = "pieces"
    pending_w_mb: float | None = None
    bank_elems: int = 0
    bank_bytes: int = 4

    @property
    def width(self) -> int:
        return len(self.peers)

    def c(self, q: int) -> float:
        """Seconds one microbatch takes peer q (c_s at that peer's rate)."""
        return self.flops_mb / (self.peers[q].tflops * 1e12)

    def grad_bytes(self) -> float:
        """b_g·P_s = Σ_u b_g,u·P_u (plus each unit's count lane)."""
        return float(sum(u.grad_bytes() for u in self.units))

    def param_bytes(self) -> float:
        """b_p·P_s = Σ_u b_p,u·P_u."""
        return float(sum(u.b_p * u.params for u in self.units))

    def params(self) -> int:
        return sum(u.params for u in self.units)


@dataclass(frozen=True)
class Reason:
    """Why a candidate is refused: `kind` (S1 NIC, S1 PCIe, S1 host DRAM,
    S1 uplink, S2, class, declared, missing, MEM, MEM-HOST, tied, not armed,
    ...), `where` (stage, peer, host, link), the number and its bound."""

    kind: str
    where: str
    value: float | str = ""
    bound: float | str = ""
    unit: str = ""
    stage: int | None = None
    host: str | None = None

    def text(self) -> str:
        if isinstance(self.value, float) and isinstance(self.bound, float):
            return f"{self.kind}: {self.where} {self.value:.4g} > {self.bound:.4g} {self.unit}".rstrip()
        tail = f" ({self.value} vs {self.bound})" if self.value != "" else ""
        return f"{self.kind}: {self.where}{tail}"

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "where": self.where,
            "value": self.value,
            "bound": self.bound,
            "unit": self.unit,
            "stage": self.stage,
            "host": self.host,
            "text": self.text(),
        }


def _source(value, given: str | None) -> str:
    if given == "absent":
        return "absent"
    if value is None:
        return "missing"
    if given is None:
        return "declared"  # a figure nobody vouches for is a claim, never a measurement
    if given not in SOURCES:
        raise ValueError(f"unknown capacity source {given!r}")
    return given


def host_reserve(ram_bytes: int) -> int:
    """Host memory no plan may count on: the OS, the interpreters and their
    libraries. 10% with a 4 GiB floor (HYPOTHESIS, like vram_reserve)."""
    return max(int(ram_bytes * 0.10), 4 << 30)


def vram_reserve(vram: int) -> int:
    """The planner's device reserve: 12%, at least 1 GiB (planner.vram_reserve)."""
    return max(int(vram * 0.12), 1024 * 2**20)


# --------------------------------------------------------------------- class


def check_class(peers: Sequence[Gpu], probed: bool = True) -> list[Reason]:
    """Every peer of one stage shares the device type and model (BA-31). With
    `probed` (pieces), a declared device or gpu_name is refused naming the
    field: the class comes from the probe, never from a claim."""
    out = []
    if probed:
        for g in peers:
            for fld in ("device", "gpu_name"):
                if fld in g.declared:
                    out.append(Reason("declared", f"{g.name} {fld}", "declared", "probed"))
    classes = sorted({(g.device, g.gpu_name) for g in peers})
    if len(classes) > 1:
        out.append(
            Reason(
                "class",
                "peers " + ", ".join(g.name for g in peers),
                " + ".join(f"{d}:{n}" for d, n in classes),
                "one class",
            )
        )
    return out


# --------------------------------------------------------------------- inputs


def missing_inputs(
    stages: Sequence[Stage],
    hosts: Mapping[str, Host],
    k_source: str,
    pieces: bool,
    sites: int = 1,
) -> list[Reason]:
    """Inputs a wide candidate is priced on, refused when missing (any wide
    layout) or when not probed (pieces): NIC and PCIe per direction (PCIe
    'absent' drops out, on a cpu or mps device only), HBM, the host's RAM and
    DRAM, and k. 'absent' anywhere else, and a figure that is not > 0, are
    refused naming the field: neither is a measurement."""
    out = []
    ok = PIECES_SOURCES if pieces else (*PIECES_SOURCES, "declared")
    for st in stages:
        for g in st.peers:
            for fld in ("nic_up", "nic_down", "pcie_h2d", "pcie_d2h", "hbm"):
                src = g.source(fld)
                if src == "absent":
                    if not (fld.startswith("pcie_") and is_host_device(g.device)):
                        out.append(Reason(src, f"{g.name} {fld}", stage=st.index))
                elif src not in ok:
                    out.append(Reason(src, f"{g.name} {fld}", stage=st.index))
                elif not getattr(g, fld) > 0:
                    out.append(
                        Reason("invalid", f"{g.name} {fld}", getattr(g, fld), "> 0", stage=st.index)
                    )
        for h in sorted({g.host for g in st.peers}):
            host = hosts.get(h) or Host(h)
            for fld in ("ram_bytes", "dram"):
                src = host.source(fld)
                if src not in ok or src == "absent":
                    out.append(Reason(src, f"host {h} {fld}", stage=st.index, host=h))
                elif not getattr(host, fld) > 0:
                    out.append(
                        Reason(
                            "invalid",
                            f"host {h} {fld}",
                            getattr(host, fld),
                            "> 0",
                            stage=st.index,
                            host=h,
                        )
                    )
    if k_source not in K_SOURCES:
        out.append(Reason(k_source, "k_passes (host-DRAM passes per side)"))
    return _dedupe(out)


def _dedupe(reasons: list[Reason]) -> list[Reason]:
    seen, out = set(), []
    for r in reasons:
        key = (r.kind, r.where)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# --------------------------------------------------------------------- S1


@dataclass(frozen=True)
class LinkLoad:
    kind: str  # S1 NIC | S1 PCIe | S1 host DRAM | S1 uplink
    where: str
    load: float  # bytes/s the stages put on it
    cap: float
    stage: int | None = None
    host: str | None = None


def s1_loads(
    stages: Sequence[Stage],
    hosts: Mapping[str, Host],
    k: float,
    uplinks: Mapping[tuple[str, str], float] | None = None,
) -> list[LinkLoad]:
    """S1 per link, summed over every (stage, GPU) placed on it, each at its
    own c_s: Σ k·(b_g·P_s·share + A)/c_s. share (w−1)/w on a NIC or PCIe path
    per direction (k 1), 2(w−1)/w on host DRAM (both directions, A both
    ways, k passes per side), n_a·n_b/w on an uplink per direction (k 1; the
    pipeline-boundary crossing is priced as a hop of the step, not here)."""
    out: list[LinkLoad] = []
    dram: dict[str, float] = {}
    up: dict[tuple[str, str], float] = {}
    for st in stages:
        w = st.width
        share = st.grad_bytes() * (w - 1) / w
        for q, g in enumerate(st.peers):
            per_dir = (share + st.boundary_mb) / st.c(q)
            for fld, kind in (
                ("nic_up", "S1 NIC"),
                ("nic_down", "S1 NIC"),
                ("pcie_d2h", "S1 PCIe"),
                ("pcie_h2d", "S1 PCIe"),
            ):
                cap = getattr(g, fld)
                if g.source(fld) == "absent" or cap is None:
                    continue
                out.append(
                    LinkLoad(kind, f"stage {st.index} {g.name} {fld}", per_dir, cap, st.index)
                )
            dram[g.host] = dram.get(g.host, 0.0) + k * 2 * per_dir
        if w > 1:
            rate = sum(1.0 / st.c(q) for q in range(w)) / w  # microbatches/s per peer
            sites = sorted({g.site for g in st.peers})
            for a in sites:
                for b in sites:
                    if a == b:
                        continue
                    n_a = sum(g.site == a for g in st.peers)
                    n_b = sum(g.site == b for g in st.peers)
                    up[(a, b)] = up.get((a, b), 0.0) + st.grad_bytes() * n_a * n_b / w * rate
    for h, load in sorted(dram.items()):
        cap = (hosts.get(h) or Host(h)).dram
        if cap is not None:
            stg = sorted({st.index for st in stages for g in st.peers if g.host == h})
            out.append(LinkLoad("S1 host DRAM", f"host {h} (stages {stg})", load, cap, stg[0], h))
    for (a, b), load in sorted(up.items()):
        cap = (uplinks or {}).get((a, b))
        out.append(LinkLoad("S1 uplink", f"{a}->{b}", load, math.inf if cap is None else cap))
    return out


def s1(
    stages: Sequence[Stage],
    hosts: Mapping[str, Host],
    k: float,
    uplinks: Mapping[tuple[str, str], float] | None = None,
) -> list[Reason]:
    """The S1 refusals: every link whose load exceeds its capacity."""
    return [
        Reason(x.kind, x.where, x.load, x.cap, "B/s", x.stage, x.host)
        for x in s1_loads(stages, hosts, k, uplinks)
        if x.load > x.cap
    ]


# --------------------------------------------------------------------- S2


def pipeline_compute(stages: Sequence[Stage], microbatches: int) -> float:
    """C = (M/w)·max_s c_s: the pipeline's compute per step at its slowest
    stage (each stage at the aggregate rate of its peers)."""
    return microbatches * max(
        st.flops_mb / (sum(g.tflops for g in st.peers) * 1e12) for st in stages
    )


def exposed_beyond_window(p: StagePrediction) -> float:
    """A stage's exposure beyond its idle window: max(0, X_s − window_s, delay)."""
    return max(0.0, p.tail - p.window, p.delay)


def s2(
    predictions: Mapping[int, StagePrediction], compute: float, epsilon: float = EPSILON
) -> list[Reason]:
    """E − window_s ≤ ε·C per stage, E and window_s from SCHEDULER-I17.

    E = max(tail, delay); the delay of step t+1's forward is already measured
    against its slot, so the window comes off the tail only (as the planner's
    `_exposed`): subtracting it from the delay too would admit a delay-bound
    stage whose forward is late."""
    out = []
    for s, p in sorted(predictions.items()):
        exposed = exposed_beyond_window(p)
        if exposed > epsilon * compute:
            out.append(Reason("S2", f"stage {s}", exposed, epsilon * compute, "s", stage=s))
    return out


# --------------------------------------------------------------------- memory


def m_bw_from_order(order) -> int:
    """m_BW: the most microbatches of one order (a peer's [(op, mb), ...])
    between their B's end and their W's start (SCHEDULER-I21)."""
    held: set = set()
    most = 0
    for op, mb in order:
        if op == "B":
            held.add(mb)
            most = max(most, len(held))
        elif op == "W":
            held.discard(mb)
    return most


def device_terms(
    stage: Stage, q: int, m_k: int, n_stages: int, microbatches: int, m_bw: int | None = None
) -> dict:
    """SCHEDULER-I16's device memory of peer q owning m_k microbatches, term
    by term (bytes): state b_p·P_s + 16·P_s/w, the own-piece hold
    b_g·M_k·P_s/w (RUNTIME-I13), one module's dW b_g·P_module,max, in-flight
    boundaries min(S, M/w)·T_mb·h·b_boundary, pending W L_s·T_mb·16·h·4 (or
    the stage's own figure) and, on the last stage, the logits. A stage with
    HF banks adds `hf_bank` = m_BW,k·b_m·P_bank,s, m_BW,k the most of the
    peer's microbatches between B's end and W's start in the emitted order
    (`m_bw`; M_k before an order is solved; SCHEDULER-I21)."""
    w = stage.width
    own = [row[q] for row in stage.owners]
    modules = [u.b_g * u.params for u in stage.units if u.flops > 0]
    pending = (
        stage.n_blocks * stage.tokens_mb * 16 * stage.d_model * 4
        if stage.pending_w_mb is None
        else stage.pending_w_mb
    )
    terms = {
        "working": stage.param_bytes(),
        "owned_state": OWNED_STATE_BYTES * sum(u.params * f for u, f in zip(stage.units, own)),
        "own_hold": m_k * sum(u.grad_bytes() * f for u, f in zip(stage.units, own)),
        "module_dw": float(max(modules, default=0)),
        "boundaries": min(n_stages, microbatches // w)
        * stage.tokens_mb
        * stage.d_model
        * stage.boundary_bytes,
        "pending_w": pending,
        "logits": stage.logits_bytes,
    }
    if stage.bank_elems:
        held = m_k if m_bw is None else m_bw
        terms["hf_bank"] = float(held * stage.bank_bytes * stage.bank_elems)
    return terms


def device_memory(
    stages: Sequence[Stage],
    mb_owned: Mapping[int, Sequence[int]],
    n_stages: int,
    microbatches: int,
    m_bw: Mapping[int, Sequence[int]] | None = None,
) -> list[Reason]:
    """Refuse every peer whose device terms exceed its device less the
    reserve. `m_bw[stage][peer]` is m_BW,k from the emitted order
    (`m_bw_from_order`); without it the bank term holds M_k."""
    out = []
    for st in stages:
        for q, g in enumerate(st.peers):
            held = None if m_bw is None else m_bw[st.index][q]
            need = sum(
                device_terms(st, q, mb_owned[st.index][q], n_stages, microbatches, held).values()
            )
            cap = g.vram_bytes - vram_reserve(g.vram_bytes)
            if need > cap:
                out.append(
                    Reason(
                        "MEM", f"stage {st.index} peer {g.name}", need, float(cap), "B", st.index
                    )
                )
    return out


def host_hold(stage: Stage, q: int, m_k: int, microbatches: int) -> float:
    """One step of exchange bytes peer q stages in host memory (RUNTIME-I13):
    the gradients it receives for what it owns and sends for what it does not,
    Σ_u b_g,u·P_u·[(1 − o)·M_k + o·(M − M_k)], plus one step of gathers out
    and in, Σ_u b_p,u·P_u·[o·(w − 1) + (1 − o)]. With pieces (o = 1/w) this is
    b_g·P_s·[M_k(w−1)/w + (M−M_k)/w] + 2·b_p·P_s·(w−1)/w."""
    w = stage.width
    g = p = 0.0
    for u, row in zip(stage.units, stage.owners):
        o = row[q]
        g += u.grad_bytes() * ((1 - o) * m_k + o * (microbatches - m_k))
        p += u.b_p * u.params * (o * (w - 1) + (1 - o))
    return g + p


def mem_host(
    stages: Sequence[Stage],
    hosts: Mapping[str, Host],
    mb_owned: Mapping[int, Sequence[int]],
    microbatches: int,
    pinned_bytes_per_peer: float | None = None,
) -> list[Reason]:
    """MEM-HOST: per host, the sum over every peer placed on it (any stage)
    of its one-step bound plus its pinned copy chunks in flight (0 on cpu and
    mps, which make no pinned copies; the caller's figure on CUDA), against
    the host's RAM less its reserve. The refusal names the host."""
    need: dict[str, float] = {}
    for st in stages:
        for q, g in enumerate(st.peers):
            pinned = 0.0 if is_host_device(g.device) else float(pinned_bytes_per_peer or 0.0)
            hold = host_hold(st, q, mb_owned[st.index][q], microbatches)
            need[g.host] = need.get(g.host, 0.0) + hold + pinned
    out = []
    for h, total in sorted(need.items()):
        ram = (hosts.get(h) or Host(h)).ram_bytes
        if ram is None:
            continue  # missing_inputs names it
        cap = float(ram - host_reserve(ram))
        if total > cap:
            stg = sorted({st.index for st in stages for g in st.peers if g.host == h})
            out.append(Reason("MEM-HOST", f"host {h} (stages {stg})", total, cap, "B", stg[0], h))
    return out


# --------------------------------------------------------------------- re-check


def exchange_layout(stage: Stage, barrier: bool | None = None) -> StageLayout:
    """The stage as SCHEDULER-I17 simulates it: each peer's port at
    min(NIC, PCIe) per its slower direction, an absent PCIe path unlimited."""
    peers = []
    for g in stage.peers:
        pcie = (
            math.inf
            if g.source("pcie_h2d") == "absent" or g.source("pcie_d2h") == "absent"
            else min(g.pcie_h2d or 0.0, g.pcie_d2h or 0.0)
        )
        peers.append(
            Peer(
                g.name,
                min(g.nic_up or 0.0, g.nic_down or 0.0),
                pcie,
                g.hbm or 0.0,
                g.host,
                g.site,
            )
        )
    return StageLayout(
        stage.units,
        stage.owners,
        tuple(peers),
        layout="zero1" if stage.layout == "zero1" else "pieces",
        barrier=barrier,
        boundary_bytes=stage.boundary_mb,
    )


def fold_overhead(stage: Stage, q: int, n_mb: int) -> float:
    """Device time of peer q's folds of n_mb microbatches and its AdamW."""
    g = stage.peers[q]
    own = [row[q] for row in stage.owners]
    fold = sum(u.params * f * n_mb * fold_bytes(u.b_g) for u, f in zip(stage.units, own))
    adam = sum(u.params * f * ADAMW_BYTES for u, f in zip(stage.units, own))
    return (fold + adam) / (g.hbm or math.inf)


def mb_owned_from(mb_owner: Sequence[Sequence[int]], stages: Sequence[Stage]) -> dict:
    return {st.index: [list(mb_owner[st.index]).count(q) for q in range(st.width)] for st in stages}


def recheck(
    stages: Sequence[Stage],
    hosts: Mapping[str, Host],
    cfg: ZBConfig,
    slot_orders,
    peer_weights,
    mb_owner: Sequence[Sequence[int]],
    fabric: Fabric,
    *,
    emission: str = "forward",
    epsilon: float = EPSILON,
    pinned_bytes_per_peer: float | None = None,
) -> list[Reason]:
    """Re-check a placed pieces plan after its ownership or orders changed
    (re-weighting, a rebalance; BA-44, BA-47): device memory and MEM-HOST
    with each peer's M_k from mb_owner, and S2 from the exchange simulation
    of the given orders, C from cfg's per-stage costs (seconds per microbatch
    at each stage's aggregate rate). Class: check_class. An empty list admits."""
    owned = mb_owned_from(mb_owner, stages)
    out = device_memory(stages, owned, cfg.n_stages, cfg.n_micro)
    out += mem_host(stages, hosts, owned, cfg.n_micro, pinned_bytes_per_peer)
    layouts = {st.index: exchange_layout(st) for st in stages}
    pred = simulate_exchange(cfg, slot_orders, peer_weights, layouts, fabric, emission=emission)
    # C over the whole pipeline (every stage of cfg, not only the re-checked ones)
    compute = cfg.n_micro * max(
        cfg.cost_f[s] + cfg.cost_b[s] + cfg.cost_w[s] for s in range(cfg.n_stages)
    )
    out += s2(pred.stages, compute, epsilon)
    return out
