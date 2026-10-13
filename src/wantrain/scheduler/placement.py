"""Placement: bind a recipe (JobSpec) to concrete, measured nodes.

The recipe-given path of the scheduler (the planner path is fleetplan.py):
the operator fixes the model, steps, microbatching and engine flags; the
scheduler decides who holds which slot, where each slot listens, which device
it runs on, and how many layers each stage gets.

- rank order: by the measured link matrix when there is one (chain routing in
  partition.py); otherwise by memory need, because rank 0 holds the deepest
  stack of live graphs and the last rank carries the logits, so the largest
  need gets the largest card;
- layer cut: proportional to measured stage speed (exact min-max DP), so a
  slow node gets fewer layers and every stage finishes together;
- a wide stage's speed is its peers' sum: microbatches are owned in
  proportion to each peer's speed (interleaved weighted round robin), so a
  twice-as-fast peer computes twice as many.
"""

from __future__ import annotations

from dataclasses import asdict

from wantrain.resources.node import mixed_fleet
from wantrain.runtime.spec import JobSpec, LinkCfg
from wantrain.scheduler.autoconfig import BYTES_PER_PARAM_FP32_ADAM, activation_graph_bytes
from wantrain.scheduler.partition import proportional_partition
from wantrain.scheduler.schedule import stage_live_graphs


def rank_memory_weights(spec: JobSpec, k_inflight: int = 1) -> list:
    """Relative memory need per rank, before layers are cut: the stack of live
    graphs from lifespans (deepest at rank 0), the embedding plus its optimizer
    state on rank 0, and the fp32 logits on the last rank, all in bytes."""
    m = spec.model
    S = spec.stages
    layers = max(1, m.n_layer // S)
    graph = activation_graph_bytes(spec.mb_size, m.seq_len, m.d_model, layers)
    k = max(1, k_inflight)
    emb_state = m.vocab_size * m.d_model * BYTES_PER_PARAM_FP32_ADAM
    head = 2 * spec.mb_size * m.seq_len * m.vocab_size * 4
    weights = []
    for r in range(S):
        w = stage_live_graphs(S, r, k) * graph
        if r == 0:
            w += emb_state
        if r == S - 1:
            w += head + (0 if m.tie_embeddings else emb_state)
        weights.append(w)
    return weights


def memory_order(spec: JobSpec, nodes, k_inflight: int = 1) -> list | None:
    """Node order (indices) giving the biggest memory to the biggest need, or
    None when any node cannot price its memory (guessing is worse)."""
    if any(not n.memory_bytes() for n in nodes) or len(nodes) != spec.stages:
        return None
    weights = rank_memory_weights(spec, k_inflight)
    need = sorted(range(spec.stages), key=lambda r: -weights[r])
    have = sorted(range(len(nodes)), key=lambda i: -nodes[i].memory_bytes())
    slot_of = dict(zip(need, have))
    return [slot_of[r] for r in range(spec.stages)]


def stage_speeds(spec: JobSpec, groups) -> list:
    """groups[r] = the nodes of stage r."""
    return [sum(n.speed() for n in g) for g in groups]


def emulated_links(groups, emulations) -> list | None:
    """Per-hop transport shim settings when every node declared a WAN
    emulation profile (experiments without a real WAN). Hop r joins stage r
    and stage r+1 (the last entry is the ring wrap)."""
    if not emulations or any(e is None or not e.enabled for e in emulations.values()):
        return None
    S = len(groups)
    out = []
    for r in range(S):
        a, b = groups[r][0], groups[(r + 1) % S][0]
        link = emulations[a.node_id].link_to(b.region)
        out.append(asdict(LinkCfg(latency_ms=link.rtt_ms / 2.0, bw_mbps=link.mbps)))
    return out


def place(spec: JobSpec, groups, emulations=None, keep_partition: bool = False) -> dict:
    """Fill endpoints, devices, partition and (emulated) links in place.
    groups[r][p] is the NodeInfo holding slot (r, p). Returns the rationale.
    A map spanning more than one device class is refused first (CONTROL-I25)."""
    mixed = mixed_fleet((n.name, n.device_class) for g in groups for n in g)
    if mixed is not None:
        raise ValueError(mixed)
    had_partition = spec.partition is not None
    spec.finalize()
    spec.endpoints = [[[n.host, n.data_port, n.bind_port] for n in g] for g in groups]
    devices = [[n.device for n in g] for g in groups]
    spec.devices = devices if any(d != spec.device for row in devices for d in row) else None
    if spec.devices is None and devices:
        spec.device = devices[0][0]
    speeds = stage_speeds(spec, groups)
    # Ownership follows speed; the schedule is re-solved for the new owners.
    spec.peer_weights = [[n.speed() / max(m.speed() for m in g) for n in g] for g in groups]
    if not keep_partition and (not had_partition or len(set(speeds)) > 1):
        spec.partition = proportional_partition(spec.model, speeds)
    links = emulated_links(groups, emulations)
    if links is not None:
        spec.links = links
    spec.reschedule()
    return {
        "placement": [[f"{n.name}({n.device}/{n.gpu_name})" for n in g] for g in groups],
        "stage_tflops": [round(s, 4) for s in speeds],
        "partition": spec.partition,
        "layers_per_stage": [b - a for a, b in spec.partition],
        "devices": spec.devices or spec.device,
        "emulated_links": links,
    }
