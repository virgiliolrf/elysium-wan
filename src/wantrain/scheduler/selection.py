"""Choosing which registered workers a job runs on (WAN-aware node selection).

Every fleet built here is inside one region (unless told otherwise), on
distinct addresses, above an uplink floor, inside the compute-capability window
the runtime can execute, and ranked by what earlier runs *measured* between
these machines before anything a host claims about itself: the fastest nodes
on offer are routinely a hemisphere apart, and that fleet does not train.

Every fleet holds one device class (NodeInfo.device_class, CONTROL-I25) until
the heterogeneous track returns: a pool no single class can field is refused
MIXED-FLEET. Within the class, speed may differ (group_by="any"): the
partitioner cuts layers in proportion to each node's measured speed.
group_by="model" keeps one exact GPU model; "series" keeps one silicon
generation. Within a feasible group, faster nodes are preferred, since a
proportional partition's throughput is the sum of its stages' speeds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from wantrain.resources.node import MIXED_FLEET_TAIL, NodeInfo, class_counts
from wantrain.scheduler.linkmemory import PENALTY, LinkMemory, shares_subnet, verdict

# Advertised uplink below this is not worth a pipeline stage: the planner
# budgets against the slower direction. Applied only while it leaves enough.
MIN_UPLINK_MBPS = 200.0


def gpu_series(name: str) -> str | None:
    """The consumer GPU series a device name belongs to ("RTX 40-series"), or
    None for datacenter, workstation and non-GPU devices."""
    m = re.search(r"\b(RTX|GTX)\s?(\d{2})\d{2}(?!\d)", name or "", re.I)
    return f"{m.group(1).upper()} {m.group(2)}-series" if m else None


def node_key(n: NodeInfo) -> str:
    """Identity used by the dead-host memory: the same worker coming back
    under the same name on the same address is the same suspect machine."""
    return f"{n.name}@{n.host}"


@dataclass
class SelectionPolicy:
    group_by: str = "any"  # any | model | series
    same_region: bool = True
    distinct_hosts: bool = True
    min_vram_bytes: int = 0
    min_compute_cap: int = 0  # 800 = sm_80; only CUDA nodes have one
    max_compute_cap: int = 0  # newest the runtime has kernels for; 0 = no cap
    min_uplink_mbps: float = MIN_UPLINK_MBPS
    gpu: str | None = None  # substring filter on the device name
    devices: tuple | None = None  # allowed device types, e.g. ("cuda",)


@dataclass
class Candidate:
    region: str
    group: str
    nodes: list

    def total_speed(self) -> float:
        return sum(n.speed() for n in self.nodes)

    def all_bf16(self) -> bool:
        return all(n.has_bf16() for n in self.nodes)

    def min_memory(self) -> int:
        return min(n.memory_bytes() for n in self.nodes)


@dataclass
class Selection:
    ok: bool
    chosen: list = field(default_factory=list)
    spares: list = field(default_factory=list)
    reason: str | None = None
    note: str | None = None
    alternatives: list = field(default_factory=list)


def _hop_verdict(a: NodeInfo, b: NodeInfo, memory: LinkMemory | None):
    if memory is None:
        return None
    return verdict(memory.path(a.host, b.host))


def _path_penalty(fleet, memory) -> int:
    worst = 0
    for a, b in zip(fleet, fleet[1:]):
        v = _hop_verdict(a, b, memory)
        if v:
            worst = max(worst, PENALTY[v])
    return worst


def _rank_key(n: NodeInfo):
    return (-n.speed(), -n.memory_bytes(), n.node_id)


def distinct_addresses(nodes, n: int, distinct_hosts: bool, memory: LinkMemory | None):
    """Fastest `n` nodes with no two sharing an address, avoiding pairings a
    previous run measured and found bad; falls back to ignoring the path
    memory when nothing better can be assembled (a fleet with a known-bad hop
    is still a fleet)."""
    ordered = sorted(nodes, key=_rank_key)

    def greedy(avoid_slow: bool):
        seen, out = set(), []
        for o in ordered:
            if distinct_hosts and o.host in seen:
                continue
            if (
                avoid_slow
                and memory is not None
                and any((_hop_verdict(p, o, memory) or "good") != "good" for p in out)
            ):
                continue
            seen.add(o.host)
            out.append(o)
            if len(out) == n:
                break
        return out

    clean = greedy(True)
    return clean if len(clean) == n else greedy(False)


def _group_key(n: NodeInfo, group_by: str) -> str:
    if group_by == "model":
        return n.gpu_name
    if group_by == "series":
        return gpu_series(n.gpu_name) or n.gpu_name
    return "*"


def candidate_fleets(
    pool,
    n: int,
    policy: SelectionPolicy | None = None,
    memory: LinkMemory | None = None,
    dead: set | None = None,
):
    """-> (candidates best-first, reason-if-none). Shared by selection and by
    anything that wants to show the alternatives, so nothing can offer a fleet
    the selector would refuse."""
    pol = policy or SelectionPolicy()
    if n < 1:
        return [], "a fleet needs at least one node"
    dead = set(dead or ()) | (memory.dead_keys() if memory else set())
    nodes = [x for x in pool if node_key(x) not in dead]
    total = len(nodes)

    if pol.devices:
        nodes = [x for x in nodes if x.device in pol.devices]
    if pol.gpu:
        want = pol.gpu.lower()
        nodes = [x for x in nodes if want in (x.gpu_name or "").lower()]
        if not nodes:
            return [], f'no registered node matches "{pol.gpu}" ({total} seen)'

    fat = [
        x
        for x in nodes
        if (x.uplink_mbps is None or x.uplink_mbps >= pol.min_uplink_mbps)
        and (x.downlink_mbps is None or x.downlink_mbps >= pol.min_uplink_mbps)
    ]
    if len(fat) >= n:
        nodes = fat

    if pol.min_compute_cap:
        capable = [x for x in nodes if x.cap_int() >= pol.min_compute_cap]
        if len(capable) < n:
            best = max([x.cap_int() for x in nodes] or [0])
            return [], (
                f"the declared precision needs compute capability "
                f"sm_{pol.min_compute_cap // 10} or better on {n} nodes; the best "
                f"registered is sm_{best // 10}"
            )
        nodes = capable
    if pol.max_compute_cap:
        runnable = [x for x in nodes if x.cap_int() <= pol.max_compute_cap]
        if len(runnable) < n:
            return [], (
                f"fewer than {n} nodes are old enough for this runtime's kernels "
                f"(max sm_{pol.max_compute_cap // 10}); a card newer than the "
                f"runtime does not run slowly, it does not run"
            )
        nodes = runnable
    if pol.min_vram_bytes:
        big = [x for x in nodes if x.memory_bytes() >= pol.min_vram_bytes]
        if len(big) < n:
            best = max([x.memory_bytes() for x in nodes] or [0])
            return [], (
                f"fewer than {n} nodes have {pol.min_vram_bytes / 2**30:.1f} GB of "
                f"device memory; the largest has {best / 2**30:.1f} GB"
            )
        nodes = big

    # CONTROL-I25: a fleet is one device class; a node reporting none is in no fleet.
    classes: dict[str, list] = {}
    for x in nodes:
        if x.device_class:
            classes.setdefault(x.device_class, []).append(x)
    mixed = len(classes) > 1 or any(not x.device_class for x in nodes)
    if mixed and not any(
        len(distinct_addresses(g, n, pol.distinct_hosts, memory)) == n for g in classes.values()
    ):
        return [], (
            f"MIXED-FLEET: no single device class can field {n} nodes on distinct addresses "
            f"(pool: {class_counts(nodes)}); " + MIXED_FLEET_TAIL
        )
    by_region: dict[str, list] = {}
    for x in nodes:
        r = (x.region or "??") if pol.same_region else "*"
        by_region.setdefault(r, []).append(x)
    cands = []
    for region, group in by_region.items():
        by_key: dict[str, list] = {}
        for x in group:
            if x.device_class:
                key = f"{x.device_class}|{_group_key(x, pol.group_by)}"
                by_key.setdefault(key, []).append(x)
        for key, same in by_key.items():
            picked = distinct_addresses(same, n, pol.distinct_hosts, memory)
            if len(picked) == n:
                cands.append(Candidate(region, key, picked))

    if not cands:
        counts = " ".join(
            f"{r}:{len(distinct_addresses(g, n, pol.distinct_hosts, memory))}"
            for r, g in sorted(by_region.items())
        )
        where = "single region " if pol.same_region else ""
        return [], (
            f"no {where}can field {n} nodes on distinct addresses "
            f"({len(nodes)} eligible across {len(by_region)} region(s): {counts}). "
            f"Spreading a pipeline across regions is what makes the planner refuse "
            f"it once the link is measured"
        )

    cands.sort(
        key=lambda c: (
            _path_penalty(c.nodes, memory),
            0 if c.all_bf16() else 1,
            -c.total_speed(),
            -c.min_memory(),
            c.region,
            c.group,
        )
    )
    return cands, None


def _path_notes(fleet, memory) -> list[str]:
    out = []
    worst = None
    for a, b in zip(fleet, fleet[1:]):
        p = memory.path(a.host, b.host) if memory else None
        if p and (worst is None or p.worst_mbps < worst.worst_mbps):
            worst = p
    if worst:
        v = verdict(worst)
        spread = (
            f"{worst.worst_mbps:.0f}-{worst.best_mbps:.0f} Mbps"
            if worst.best_mbps != worst.worst_mbps
            else f"{worst.mbps:.0f} Mbps"
        )
        out.append(
            f"WARNING: a previous run measured a hop of this fleet at {spread}, RTT "
            f"{worst.rtt_ms:.0f} ms"
            if v == "slow"
            else f"WARNING: a hop of this fleet is UNRELIABLE, measured at {spread} across runs"
            if v == "variable"
            else f"previously measured between these nodes at {spread}"
        )
    net = shares_subnet([x.host for x in fleet])
    if net and not worst:
        out.append(
            f"these nodes share {net}.0/24 on different addresses: one LAN or one "
            f"gateway hairpin; only a measurement separates the two"
        )
    return out


def select_nodes(
    pool,
    n: int,
    policy: SelectionPolicy | None = None,
    memory: LinkMemory | None = None,
    dead: set | None = None,
) -> Selection:
    """Pick `n` workers that can actually form a pipeline; the rest are spares
    (the replacement pool fault tolerance draws from)."""
    cands, reason = candidate_fleets(pool, n, policy, memory, dead)
    if not cands:
        return Selection(ok=False, reason=reason, spares=list(pool))
    win = cands[0]
    chosen = list(win.nodes)
    ids = {x.node_id for x in chosen}
    spares = sorted([x for x in pool if x.node_id not in ids], key=_rank_key)
    models = sorted({x.gpu_name for x in chosen})
    notes = [
        f"{n} node(s) in {win.region}: "
        + ", ".join(
            f"{x.name}({x.gpu_name}, {x.speed():.2f} TFLOPs, {x.memory_bytes() / 2**30:.1f} GB)"
            for x in chosen
        )
    ]
    notes.append(
        "one device model, so the run stays bit-comparable against itself"
        if len(models) == 1
        else f"heterogeneous fleet ({' + '.join(models)}): layers are cut in proportion "
        f"to measured speed; lossless, but not bit-comparable to a different mix"
    )
    if any(x.device == "cuda" for x in chosen) and not win.all_bf16():
        notes.append(
            "no bf16 on every node: the boundary falls back to fp32 and needs "
            "roughly 2.4x the link for the same plan"
        )
    notes += _path_notes(chosen, memory)
    alts = [
        dict(
            region=c.region,
            group=c.group,
            nodes=[x.name for x in c.nodes],
            total_tflops=round(c.total_speed(), 3),
        )
        for c in cands[1:4]
    ]
    return Selection(
        ok=True, chosen=chosen, spares=spares, note="; ".join(notes), alternatives=alts
    )
