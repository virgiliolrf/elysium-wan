"""Realistic WAN conditions on the production path: the final training state
stays byte-identical to the single-device reference, only the time moves.

Latency, jitter and bandwidth come from the engine's per-link shim
(transport/shim.py): long-haul and satellite-like profiles, and an asymmetric
chain where every hop has a different path. Loss comes from a TCP proxy on
every data-plane connection (tests/helpers/wanproxy.py): the byte stream
arrives torn into random pieces and stalls the way TCP retransmission stalls
it. Every test also proves the condition was applied (a wall-clock floor set
by the latency, or the proxy's counters), so a shim or proxy that silently
did nothing cannot pass.

WOULD-FAIL-IF: latency, jitter-induced reordering, pacing, an asymmetric
chain, torn reads or retransmission-like stalls changed one bit of a weight,
an AdamW moment, the RNG state or a loss; or the link emulation stopped being
applied.
"""

import pytest
from helpers.exact import FEATURES_ON, make_spec, reference_manifest
from helpers.wanproxy import LossyLink

from wantrain.launch.local import run_local
from wantrain.runtime.spec import LinkCfg
from wantrain.validation.state import diff_manifests, distributed_manifest

STEPS = 3


def _host_id():
    """The peer id every run_local stage presents: the inherited host key."""
    from cryptography.hazmat.primitives import serialization

    from wantrain.transport.identity import default_key_path, load_or_create_key, peer_id

    pub = load_or_create_key(default_key_path()).public_key()
    return peer_id(pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))


def _link(latency_ms, jitter_ms, bw_mbps, seed):
    return dict(
        LinkCfg(
            latency_ms=latency_ms, jitter_ms=jitter_ms, bw_mbps=bw_mbps, jitter_seed=seed
        ).__dict__
    )


# (latency ms, jitter ms, bandwidth Mbps) per hop of the 3-stage chain.
PROFILES = {
    "intercontinental": [(150, 50, 50)] * 3,
    "satellite": [(300, 100, 20)] * 3,
    "asymmetric": [(5, 2, 500), (200, 80, 20), (60, 20, 100)],
}


def _assert_exact(tmp_path, spec):
    res = run_local(spec, timeout=600)
    assert res["ok"], res
    state = distributed_manifest(spec)
    reference = reference_manifest(tmp_path / "ref", steps=STEPS)
    assert diff_manifests(state, reference) == []
    assert state == reference
    return res


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_wan_profile_leaves_the_state_byte_exact(tmp_path, port_base, profile):
    hops = PROFILES[profile]
    links = [_link(*hop, seed=7 + i) for i, hop in enumerate(hops)]
    spec = make_spec(tmp_path / "run", profile, port_base, FEATURES_ON, links, steps=STEPS)
    res = _assert_exact(tmp_path, spec)
    # Every step crosses the chain forward and back at least once: a run
    # faster than that never saw the latency.
    floor_s = STEPS * 2 * sum(lat for lat, _, _ in hops[: spec.stages - 1]) / 1e3
    assert res["wall_seconds"] >= floor_s, (res, floor_s)


def test_torn_and_stalled_streams_leave_the_state_byte_exact(tmp_path, port_base):
    widths = FEATURES_ON["stage_widths"]
    # Workers listen on base + 8r + p; peers dial a proxy on base + 32 + 8r + p.
    endpoints = [
        [["127.0.0.1", port_base + 32 + 8 * r + p, port_base + 8 * r + p] for p in range(w)]
        for r, w in enumerate(widths)
    ]
    proxies = [
        LossyLink(dial, listen, seed=100 + i, stall_p=0.02, stall_ms=(20, 200)).start()
        for i, (_, dial, listen) in enumerate(ep for row in endpoints for ep in row)
    ]
    try:
        spec = make_spec(
            tmp_path / "run",
            "lossy",
            port_base,
            FEATURES_ON,
            dict(latency_ms=20, jitter_ms=10),
            steps=STEPS,
        )
        spec.endpoints = endpoints
        # A spec with endpoints must pin every slot (RUNTIME-I16). run_local's
        # stages all prove this process's one host key (WANTRAIN_PEER_KEY).
        host = _host_id()
        spec.slot_peers = [[host] * w for w in widths]
        _assert_exact(tmp_path, spec)
    finally:
        for px in proxies:
            px.close()
    total = {k: sum(px.stats[k] for px in proxies) for k in proxies[0].stats}
    assert total["connections"] >= sum(widths) - 1, total
    assert total["bytes"] > 0 and total["pieces"] > total["connections"], total
    assert total["stalls"] > 0, total
