"""Which spare takes a dead slot's place.

Ported from the source's replacement rule, with the marketplace removed: the
pool is the coordinator's registered spares. Requirements: not the dead node
(nor any node the dead-host memory holds), an address distinct from every
survivor, at least the dead node's device memory (the pinned partition must
still fit, because a heal never re-plans), an acceptable uplink when known,
and a device that can run the spec's compute dtype. Preference: architecture
above location above speed -- the identical model keeps the healed run
bit-comparable to an uninterrupted one; the same series keeps the dtype ladder.

A slot of a sharded ('pieces') stage is healed only by the identical device
model, and only a probed one (CONTROL-I9, BA-31): a declared gpu_name or device
is the operator's word, not the silicon, so it never proves the class.

A slot of an arch 'hf' run is healed only by a spare whose launch gate reports
the slot's recorded device class (CONTROL-I24, ADR-0009 decision 8): the run's
device map, and so its oracle, never change.
"""

from __future__ import annotations

from wantrain.model import precision
from wantrain.resources.node import NodeInfo
from wantrain.scheduler.selection import MIN_UPLINK_MBPS, gpu_series, node_key


def find_replacement(
    spares,
    dead: NodeInfo,
    survivors,
    compute_dtype: str = "fp32",
    match: str = "series",
    dead_keys=None,
    distinct_hosts: bool = True,
    min_uplink_mbps: float = MIN_UPLINK_MBPS,
    probed_class: bool = False,
    device_class: str | None = None,
    gate_classes: dict | None = None,
):
    """-> (NodeInfo, note) or None.

    match: "model" (identical device model only), "series" (identical model or
    same consumer series; the source default), "device" (same device type),
    "any" (anything that fits and can run the recipe).
    probed_class: the class (gpu_name, device) of the dead node and of the spare
    must both come from the probe; a declared one is never a match.
    device_class: the slot's recorded device class (arch 'hf'); a spare is
    usable only when gate_classes (node_id -> the device classes its passed
    gate reports list) holds it for that spare.
    """
    dead_keys = set(dead_keys or ())
    survivor_hosts = {s.host for s in survivors}
    survivor_regions = {s.region for s in survivors}
    dead_series = gpu_series(dead.gpu_name)
    if probed_class and _declared_class(dead):
        return None

    def ok(o: NodeInfo) -> bool:
        if o.node_id == dead.node_id or node_key(o) in dead_keys:
            return False
        if node_key(o) == node_key(dead):
            return False
        if distinct_hosts and o.host in survivor_hosts:
            return False
        if o.memory_bytes() < dead.memory_bytes():
            return False
        if o.uplink_mbps is not None and o.uplink_mbps < min_uplink_mbps:
            return False
        if probed_class and _declared_class(o):
            return False
        if device_class is not None and device_class not in (gate_classes or {}).get(o.node_id, ()):
            return False
        if o.device == "cuda" and not precision.compute_runnable(compute_dtype, o.compute_cap):
            return False
        same_model = o.gpu_name == dead.gpu_name and o.device == dead.device
        same_series = dead_series is not None and gpu_series(o.gpu_name) == dead_series
        if match == "model":
            return same_model
        if match == "series":
            return same_model or same_series
        if match == "device":
            return o.device == dead.device
        return True

    usable = [o for o in spares if ok(o)]
    if not usable:
        return None

    def tier(o):
        return 0 if (o.gpu_name == dead.gpu_name and o.device == dead.device) else 1

    def place(o):
        return 0 if o.region == dead.region else (1 if o.region in survivor_regions else 2)

    best = sorted(usable, key=lambda o: (tier(o), place(o), -o.speed(), o.node_id))[0]
    where = best.region or "unknown region"
    place_word = (
        "the dead node's region"
        if place(best) == 0
        else "a surviving node's region"
        if place(best) == 1
        else f"{where}, nothing viable closer"
    )
    if tier(best) == 0:
        note = (
            f"identical device ({best.gpu_name}) in {place_word}: the healed run stays "
            f"bit-comparable to an uninterrupted one"
        )
    else:
        note = (
            f"{best.gpu_name} on {best.device} replaces {dead.gpu_name} on {dead.device} "
            f"in {place_word}: still lossless, no longer bit-comparable to the "
            f"uninterrupted run"
        )
    return best, note


def _declared_class(o: NodeInfo) -> bool:
    return "gpu_name" in o.declared or "device" in o.declared
