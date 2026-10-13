"""Scheduler: partition DP, chain routing, selection, placement, planner."""

import itertools
import math

import pytest

from wantrain.model.gpt import GPTConfig, make_config
from wantrain.resources.node import NodeInfo
from wantrain.runtime.spec import JobSpec
from wantrain.scheduler import fleetplan, placement
from wantrain.scheduler.linkmemory import LinkMemory, verdict
from wantrain.scheduler.partition import (
    assign_layers,
    best_chain,
    order_stages,
    proportional_partition,
)
from wantrain.scheduler.planner import Fleet, Node, plan_job
from wantrain.scheduler.selection import SelectionPolicy, gpu_series, select_nodes

# CONTROL-I25: every fleet here is one device class (speeds may differ).
ONE_CLASS = "cpu:x86_64:AVX2"


def _brute_partition(n, speeds):
    best = math.inf
    S = len(speeds)
    for cuts in itertools.combinations(range(1, n), S - 1):
        b = (0,) + cuts + (n,)
        t = max((b[i + 1] - b[i]) / speeds[i] for i in range(S))
        best = min(best, t)
    return best


@pytest.mark.parametrize(
    "n,speeds",
    [(14, [1.0, 1.0]), (14, [3.0, 1.0]), (26, [8.0, 2.0, 1.0]), (10, [1.0, 5.0, 1.0, 2.0])],
)
def test_assign_layers_is_the_exact_min_max(n, speeds):
    # covers: SCHEDULER-I9
    bounds, t = assign_layers(n, speeds)
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    assert all(a < b for a, b in bounds)
    assert t == pytest.approx(_brute_partition(n, speeds))


def test_heterogeneous_speeds_get_proportional_layers():
    # WOULD-FAIL-IF: the partition ignored measured speed (equal split)
    cfg = make_config("toy")  # 6 blocks + embed + head = 8 layers
    fast_slow = proportional_partition(cfg, [3.0, 1.0])
    slow_fast = proportional_partition(cfg, [1.0, 3.0])
    assert fast_slow[0][1] - fast_slow[0][0] > fast_slow[1][1] - fast_slow[1][0]
    assert slow_fast[0][1] - slow_fast[0][0] < slow_fast[1][1] - slow_fast[1][0]


def _key(order, bw, rtt):
    hops = list(zip(order, order[1:]))
    return (min(bw[a][b] for a, b in hops), -sum(rtt[a][b] for a, b in hops))


def test_order_stages_matches_brute_force_on_a_two_region_mesh():
    # nodes 0,1 in region X; 2,3 in region Y: fast intra, slow inter
    # covers: SCHEDULER-I10
    intra, inter = (500.0, 2.0), (50.0, 80.0)
    region = [0, 0, 1, 1]
    n = 4
    bw = [
        [math.inf if i == j else (intra if region[i] == region[j] else inter)[0] for j in range(n)]
        for i in range(n)
    ]
    rtt = [
        [0.0 if i == j else (intra if region[i] == region[j] else inter)[1] for j in range(n)]
        for i in range(n)
    ]
    order = order_stages(rtt, bw)
    best = max(itertools.permutations(range(n)), key=lambda p: _key(p, bw, rtt))
    assert _key(order, bw, rtt) == _key(best, bw, rtt)
    crossings = sum(region[a] != region[b] for a, b in zip(order, order[1:]))
    assert crossings == 1  # one WAN hop, not three


def test_best_chain_routes_around_a_dead_pair():
    # covers: SCHEDULER-I10
    edges = [(0, 1, 0.0), (0, 2, 300.0), (1, 2, 200.0)]
    order, bottleneck = best_chain(3, edges)
    assert bottleneck == 200.0
    hops = {frozenset(p) for p in zip(order, order[1:])}
    assert frozenset((0, 1)) not in hops


def test_greedy_ordering_beyond_exhaustive_size():
    # covers: SCHEDULER-I10
    n = 10
    bw = [
        [math.inf if i == j else (1000.0 if abs(i - j) == 1 else 10.0) for j in range(n)]
        for i in range(n)
    ]
    order = order_stages(None, bw)
    assert sorted(order) == list(range(n))
    assert min(bw[a][b] for a, b in zip(order, order[1:])) == 1000.0


def node(i, gpu="cpu", device="cpu", tflops=1.0, gb=8, host=None, region=None, cap=None, up=None):
    # CONTROL-I25: the HELLO's class as probe.device_class reports it (a cuda
    # card's name; every cpu node here one x86 capability).
    cls = {"cuda": f"cuda:{gpu}", "mps": "mps:Apple M4"}.get(device, "cpu:x86_64:AVX2")
    return NodeInfo(
        node_id=f"n{i}",
        name=f"w{i}",
        host=host or f"10.0.0.{i}",
        data_port=29000 + i,
        device=device,
        gpu_name=gpu,
        compute_cap=cap,
        vram_total_bytes=gb << 30,
        vram_free_bytes=gb << 30,
        tflops_fp32=tflops,
        region=region,
        uplink_mbps=up,
        device_class=cls,
    )


def test_selection_prefers_one_region_and_fast_nodes_and_keeps_spares():
    # covers: SCHEDULER-I11
    pool = [
        node(0, region="eu", tflops=10),
        node(1, region="eu", tflops=1),
        node(2, region="us", tflops=50),
        node(3, region="eu", tflops=5),
    ]
    sel = select_nodes(pool, 2, SelectionPolicy())
    assert sel.ok
    assert {n.node_id for n in sel.chosen} == {"n0", "n3"}  # one region, fastest two
    assert {n.node_id for n in sel.spares} == {"n1", "n2"}


def test_selection_refuses_cross_region_when_no_region_can_field_the_fleet():
    # covers: SCHEDULER-I11
    pool = [node(0, region="eu"), node(1, region="us")]
    assert not select_nodes(pool, 2, SelectionPolicy()).ok
    assert select_nodes(pool, 2, SelectionPolicy(same_region=False)).ok


def test_selection_hardware_filters_and_distinct_hosts():
    # covers: SCHEDULER-I11
    pool = [
        node(0, "RTX 3090", "cuda", 30, 24, cap=(8, 6)),
        node(1, "RTX 3080", "cuda", 25, 10, cap=(8, 6)),
        node(2, "GTX 1080", "cuda", 9, 8, cap=(6, 1)),
        node(3, "RTX 3090", "cuda", 30, 24, cap=(8, 6), host="10.0.0.0"),
    ]
    sel = select_nodes(pool, 2, SelectionPolicy(min_compute_cap=800))
    # CONTROL-I25: past the capability floor only the RTX 3090 class (one address)
    # and the RTX 3080 class remain, so no single class fields 2 nodes.
    assert not sel.ok and sel.reason.startswith("MIXED-FLEET: no single device class can field 2")
    assert "cuda:RTX 3090 x2, cuda:RTX 3080 x1" in sel.reason.replace(
        "cuda:RTX 3080 x1, cuda:RTX 3090 x2", "cuda:RTX 3090 x2, cuda:RTX 3080 x1"
    )
    assert "GTX 1080" not in sel.reason
    more = [*pool, node(4, "RTX 3090", "cuda", 30, 24, cap=(8, 6))]
    sel = select_nodes(more, 2, SelectionPolicy(min_compute_cap=800))
    assert sel.ok and "n2" not in {n.node_id for n in sel.chosen}
    assert {n.gpu_name for n in sel.chosen} == {"RTX 3090"}
    assert len({n.host for n in sel.chosen}) == 2  # n0 and n3 share an address
    sel = select_nodes(pool, 2, SelectionPolicy(min_vram_bytes=20 << 30))
    assert not sel.ok  # only one 24 GB address
    sel = select_nodes(
        pool, 2, SelectionPolicy(group_by="model", distinct_hosts=False, min_vram_bytes=20 << 30)
    )
    assert sel.ok and {n.gpu_name for n in sel.chosen} == {"RTX 3090"}
    assert gpu_series("RTX 3090") == gpu_series("RTX 3080") == "RTX 30-series"


def test_selection_ranks_by_measured_paths_before_speed(tmp_path):
    # covers: SCHEDULER-I11
    mem = LinkMemory(str(tmp_path))
    pool = [
        node(0, region="a", tflops=50),
        node(1, region="a", tflops=50),
        node(2, region="b", tflops=5),
        node(3, region="b", tflops=5),
    ]
    mem.record_path("10.0.0.0", "10.0.0.1", 20.0, 150.0, "r1")  # the fast pair is slow
    assert verdict(mem.path("10.0.0.0", "10.0.0.1")) == "slow"
    sel = select_nodes(pool, 2, SelectionPolicy(), mem)
    assert {n.region for n in sel.chosen} == {"b"}
    mem.record_dead("w2@10.0.0.2", "sshd died", "r1", "10.0.0.2")
    sel = select_nodes(pool, 2, SelectionPolicy(), mem)
    assert {n.region for n in sel.chosen} == {"a"}  # b cannot field 2 any more
    # persisted
    again = LinkMemory(str(tmp_path))
    assert again.path("10.0.0.1", "10.0.0.0").samples == 1 and again.dead_keys()


def test_linkmemory_counts_runs_not_readings(tmp_path):
    mem = LinkMemory(str(tmp_path))
    mem.record_path("a", "b", 300, 10, "r1")
    mem.record_path("b", "a", 30, 10, "r1")  # same run: ignored
    rec = mem.record_path("a", "b", 40, 10, "r2")
    assert rec.samples == 2 and rec.worst_mbps == 40 and rec.best_mbps == 300
    assert verdict(rec) == "variable"


def test_placement_cuts_layers_by_speed_and_maps_devices():
    # covers: CONTROL-I25
    # WOULD-FAIL-IF: place admits a CPU + MPS map (two device classes), or no longer
    # cuts one class at two speeds by speed and maps its device
    spec = JobSpec(model=make_config("toy"), stages=2, steps=2)
    mixed = [[node(0, tflops=1.0)], [node(1, "apple-mps-arm64", "mps", tflops=3.0)]]
    with pytest.raises(ValueError) as e:
        placement.place(spec, mixed)
    assert str(e.value) == (
        "MIXED-FLEET: the device map spans 2 device classes (cpu:x86_64:AVX2: w0; "
        "mps:Apple M4: w1); only one device class is admitted until the heterogeneous "
        "track returns"
    )
    assert spec.endpoints is None or spec.endpoints == []
    groups = [
        [node(0, "apple-mps-arm64", "mps", tflops=1.0)],
        [node(1, "apple-mps-arm64", "mps", tflops=3.0)],
    ]
    r = placement.place(spec, groups)
    assert spec.devices == [["mps"], ["mps"]]
    a, b = spec.partition
    assert (b[1] - b[0]) > (a[1] - a[0])
    assert spec.endpoints[1][0][:2] == ["10.0.0.1", 29001]
    assert r["stage_tflops"] == [1.0, 3.0]


def test_memory_order_gives_the_biggest_card_the_biggest_need():
    spec = JobSpec(
        model=GPTConfig(n_layer=12, d_model=256, seq_len=256, vocab_size=50257),
        stages=3,
        mb_size=2,
    )
    nodes = [node(0, gb=8), node(1, gb=24), node(2, gb=12)]
    order = placement.memory_order(spec, nodes, k_inflight=2)
    w = placement.rank_memory_weights(spec, k_inflight=2)
    heaviest = max(range(3), key=lambda r: w[r])
    assert nodes[order[heaviest]].name == "w1"


def _fleet(nodes, mbps=1000.0, rtt=5.0):
    return Fleet(nodes=nodes, link_mbps=mbps, rtt_ms=rtt, duplex="per_direction")


def test_planner_produces_a_production_spec_for_a_heterogeneous_fleet():
    cfg = make_config("gpt124m")
    nodes = [
        Node("a", tflops=80.0, vram_bytes=24 << 30, compute_cap=(8, 9), device_class=ONE_CLASS),
        Node("b", tflops=20.0, vram_bytes=24 << 30, compute_cap=(8, 6), device_class=ONE_CLASS),
    ]
    out = plan_job(cfg, _fleet(nodes), steps=10, device="cuda")
    spec = out["spec"]
    assert spec is not None
    a, b = spec.partition
    assert (a[1] - a[0]) > (b[1] - b[0])  # faster node, more layers
    assert spec.recompute and spec.wire_codec == "planes"
    assert spec.schedule_order and all(
        op in ("F", "B", "W") for row in spec.schedule_order for order in row for op, _ in order
    )
    assert spec.microbatches >= 1 and spec.mb_size >= 1
    assert out["rationale"]["k_inflight"] >= 1


def test_planner_refuses_a_link_nothing_fits():
    # covers: SCHEDULER-I12
    cfg = make_config("gpt124m")
    nodes = [
        Node("a", tflops=80.0, vram_bytes=24 << 30, device_class=ONE_CLASS),
        Node("b", tflops=80.0, vram_bytes=24 << 30, device_class=ONE_CLASS),
    ]
    out = plan_job(cfg, _fleet(nodes, mbps=0.01), steps=10, policy="lossless", require_margin=True)
    assert out["spec"] is None and "no numerically-lossless" in out["rationale"]


def test_fleetplan_builds_endpoints_and_devices():
    fleet = {
        "run_id": "r",
        "nodes": [
            dict(
                rank=0,
                host="h0",
                dial_port=1,
                listen_port=2,
                tflops=4.0,
                vram_bytes=8 << 30,
                device="cpu",
                device_class=ONE_CLASS,
            ),
            dict(
                rank=1,
                host="h1",
                dial_port=3,
                listen_port=4,
                tflops=1.0,
                vram_bytes=8 << 30,
                device="mps",
                device_class="mps:Apple M4",
            ),
        ],
        "link": {"mbps": 1000, "rtt_ms": 1, "duplex": "per_direction"},
        "job": {"model": "toy", "steps": 3, "device": "cpu", "out_dir": "/tmp/x"},
    }
    # CONTROL-I25: a CPU + MPS fleet is two device classes, refused by plan_job.
    with pytest.raises(ValueError, match="^MIXED-FLEET: the device map spans 2 device classes"):
        fleetplan.build_spec(fleet)
    fleet["nodes"][0].update(device="mps", device_class="mps:Apple M4")
    r = fleetplan.build_spec(fleet)
    spec = r["spec"]
    assert spec.endpoints == [[["h0", 1, 2]], [["h1", 3, 4]]]
    assert spec.devices == [["mps"], ["mps"]]
    assert spec.tag == "r"
    with pytest.raises(fleetplan.PlanRefused):
        fleetplan.build_spec(dict(fleet, nodes=fleet["nodes"][:1]))


def test_layer_costs_price_the_head_and_move_blocks_off_the_last_stage():
    # WOULD-FAIL-IF: the head were priced as one block (uniform layer cost)
    from wantrain.scheduler.partition import layer_costs

    cfg = make_config("gpt124m")  # 32k vocab, h=768
    c = layer_costs(cfg)
    assert c[-1] > 2.5 * c[1] and c[0] < 0.01 * c[1]
    weighted = proportional_partition(cfg, [1.0, 1.0])
    uniform_cut, _ = assign_layers(cfg.n_layer + 2, [1.0, 1.0])
    assert (weighted[1][1] - weighted[1][0]) < (uniform_cut[1][1] - uniform_cut[1][0])


def test_partition_respects_a_memory_predicate_and_is_exact():
    # covers: SCHEDULER-I9
    costs = [1.0, 3.0, 3.0, 3.0, 3.0, 2.0]
    bounds, t = assign_layers(6, [1.0, 1.0], layer_costs=costs)
    assert t == pytest.approx(min(max(sum(costs[:k]), sum(costs[k:])) for k in range(1, 6)))
    # stage 0 may hold at most 2 layers (a small card)
    bounds, _ = assign_layers(
        6, [1.0, 1.0], layer_costs=costs, fits=lambda s, a, b: s != 0 or b - a <= 2
    )
    assert bounds[0][1] - bounds[0][0] <= 2
    with pytest.raises(ValueError):
        assign_layers(6, [1.0, 1.0], fits=lambda s, a, b: False)


def test_planner_refuses_cuts_a_small_card_cannot_hold():
    # covers: SCHEDULER-I12
    from wantrain.scheduler.planner import stage_fits

    cfg = make_config("gpt350m")
    big, small = 80 << 30, 6 << 30
    nodes = [
        Node("a", tflops=50.0, vram_bytes=small, device_class=ONE_CLASS),
        Node("b", tflops=10.0, vram_bytes=big, device_class=ONE_CLASS),
    ]
    out = plan_job(cfg, _fleet(nodes), steps=1, device="cuda", require_margin=False)
    a, b = out["rationale"]["partition"][0]
    assert stage_fits(cfg, a, b, small)  # the fast but small card got what fits


# ============================================================ M2b: SCHEDULER-I16 admission

_E7 = "docs/experiments/m2b-e7/PLAN.json"
_MEASURED = {
    k: "measured"
    for k in (
        "host_ram_bytes",
        "host_dram_gbps",
        "pcie_h2d_gbps",
        "pcie_d2h_gbps",
        "nic_up_gbps",
        "nic_down_gbps",
        "hbm_gbps",
    )
}


def _gpt7b(**kw):
    """ADR-0006's 7B: h 4096, 32 blocks, 32k vocabulary, seq 4096, bf16."""
    return GPTConfig(
        vocab_size=32000,
        n_layer=32,
        n_head=32,
        d_model=4096,
        seq_len=4096,
        compute_dtype="bf16",
        boundary_dtype="bf16",
        attn_impl="sdpa",
        sdpa_backend="flash",
        **kw,
    )


def _h100(g, gph, link_gbps=400.0, dram=600.0, ram=None, site="s0", gpu="NVIDIA H100", **kw):
    return Node(
        host=f"10.{g // gph}.0.{g % gph}",
        tflops=400.0,
        vram_bytes=80 * 10**9,
        compute_cap=(9, 0),
        device="cuda",
        hostname=f"h{g // gph:03d}",
        site=site,
        gpu_name=gpu,
        host_ram_bytes=int(ram if ram is not None else 2 * 10**12),
        host_dram_gbps=dram,
        pcie_h2d_gbps=50.0,
        pcie_d2h_gbps=50.0,
        nic_up_gbps=link_gbps,
        nic_down_gbps=link_gbps,
        hbm_gbps=3350.0,
        sources=dict(_MEASURED),
        device_class=f"cuda:{gpu}",
        **kw,
    )


def _e7_fleet(x):
    """One configuration of m2b-e7/PLAN.json as a planner fleet: H100s at
    400 TFLOP/s, PCIe 50 GB/s, HBM 3.35 TB/s, stage-contiguous host packing;
    host RAM after the reserve as pinned (host_ram_avail_gb), else ample."""
    from wantrain.scheduler.admission import host_reserve

    n = int(x["fleet"].split()[0])
    gph, sites = x["gpus_per_host"], x["sites"]
    avail = x["host_ram_avail_gb"]
    ram = None
    if avail is not None:
        ram = int(avail * 1e9) + (4 << 30)
        assert ram - host_reserve(ram) == int(avail * 1e9)
    nodes = [
        _h100(
            g,
            gph,
            float(x["link_gbps"]),
            float(x["host_dram_gbps"]),
            ram,
            site=f"site{g // (n // sites)}" if sites > 1 else "s0",
            gpu="NVIDIA H100 PCIe"
            if x["id"] == "32xGPU-mixed-classes" and g >= 17
            else "NVIDIA H100",
        )
        for g in range(n)
    ]
    up = x.get("uplink_gbps")
    return Fleet(
        nodes=nodes,
        link_mbps=float(up or x["link_gbps"]) * 1000.0,
        rtt_ms=float(x.get("uplink_rtt_ms") or 0.0),
        duplex="per_direction",
        k_passes=(x["k"], "scenario"),
    )


def _plan7b(fleet, allow_pieces=True, shard=None, cfg=None):
    from wantrain.scheduler.planner import Overrides

    n = len(fleet.nodes)
    ov = Overrides(
        mb_size=1, microbatches=n, recompute=True, boundary_dtype="bf16", compute_dtype="bf16"
    )
    return plan_job(
        cfg or _gpt7b(),
        fleet,
        steps=1,
        device="cuda",
        overrides=ov,
        require_margin=False,
        allow_pieces=allow_pieces,
        shard=shard,
    )


def _e7_config(cid):
    import json

    with open(_E7) as f:
        return next(x for x in json.load(f)["configs"] if x["id"] == cid)


def _i16_stage(index, a, b, w, hosts=None, k_layout="pieces", tflops=400.0, device="cuda"):
    """A 7B stage of layers [a, b) at width w for admission (T_mb 4096)."""
    import types

    from wantrain.scheduler import admission
    from wantrain.scheduler.exchange_sim import pieces_owners
    from wantrain.scheduler.planner import _stage_units
    from wantrain.scheduler.zero_bubble import analytic_costs

    cfg = _gpt7b()
    units, _idx, blocks = _stage_units(cfg, a, b, 2, 2)
    f, bb, ww = analytic_costs(types.SimpleNamespace(model=cfg, recompute=True, partition=[(a, b)]))
    srcs = {k: "measured" for k in ("nic_up", "nic_down", "pcie_h2d", "pcie_d2h", "hbm")}
    peers = tuple(
        admission.Gpu(
            f"s{index}p{q}",
            (hosts or [f"h{index}.{q}" for q in range(w)])[q],
            "s0",
            device,
            "H100",
            tflops,
            80 * 10**9,
            50e9,
            50e9,
            50e9,
            50e9,
            3.35e12,
            sources=srcs,
        )
        for q in range(w)
    )
    ends = a == 0 or b == cfg.n_layer + 2
    return admission.Stage(
        index,
        units,
        pieces_owners(len(units), w),
        peers,
        (f[0] + bb[0] + ww[0]) * 4096,
        (1 if ends else 2) * 4096 * 4096 * 2,
        4096,
        4096,
        blocks,
        2,
        2 * 4096 * 32000 * 4.0 if b == cfg.n_layer + 2 else 0.0,
        k_layout,
    )


def _i16_hosts(stages, dram=600e9, ram=2 * 10**12):
    from wantrain.scheduler import admission

    return {
        g.host: admission.Host(g.host, ram, dram, {"ram_bytes": "measured", "dram": "measured"})
        for st in stages
        for g in st.peers
    }


def test_k_counted_equals_the_spec_figure():
    """admission.K_HOST_DRAM_COUNTED is SCHEDULER-I16's counted k (the
    transport's passes per side), and a fleet without k_passes is priced on
    it, marked counted."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the product's k drifted from the figure the SPEC of record
    # states, or the planner assumed another k when the fleet names none.
    import re

    from wantrain.scheduler import admission
    from wantrain.scheduler.planner import _Ctx

    with open("docs/design/SPEC-scheduler.md") as f:
        text = f.read()
    i16 = text[text.index("**SCHEDULER-I16.**") : text.index("**SCHEDULER-I17.**")]
    m = re.search(r"counted passes per side[^(]*\((\d+) today", " ".join(i16.split()))
    assert m, "SCHEDULER-I16 states no counted k"
    assert admission.K_HOST_DRAM_COUNTED == int(m.group(1))
    base = {"spec": JobSpec(microbatches=4, mb_size=1), "rationale": {"k_inflight": 1}}
    ctx = _Ctx(
        make_config("toy"),
        _fleet([Node("a", 1.0, 1 << 30, device_class=ONE_CLASS)] * 2),
        base,
        "cpu",
    )
    assert (ctx.k, ctx.k_source) == (float(admission.K_HOST_DRAM_COUNTED), "counted")


def test_k_counted_refuse_a_counted_claim_of_another_value():
    """'counted' names the transport's own figure (I-8: k_passes is
    {K_HOST_DRAM_COUNTED, counted}): a fleet that calls another k 'counted'
    makes a claim, and every wide candidate is refused naming k_passes; the
    constant itself, or an explicit scenario k, is accepted."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: a fleet could lower host-DRAM S1 (admitting pieces the real
    # transport cannot carry) by labelling a smaller k as counted.
    from wantrain.scheduler import admission
    from wantrain.scheduler.planner import _Ctx

    base = {"spec": JobSpec(microbatches=4, mb_size=1), "rationale": {"k_inflight": 1}}
    nodes = [Node("a", 1.0, 1 << 30, device_class=ONE_CLASS)] * 2
    for k, src, want in (
        (admission.K_HOST_DRAM_COUNTED, "counted", "counted"),
        (2, "counted", "declared"),
        (2, "scenario", "scenario"),
    ):
        fleet = Fleet(nodes=nodes, link_mbps=1000.0, rtt_ms=5.0, k_passes=(k, src))
        ctx = _Ctx(make_config("toy"), fleet, base, "cpu")
        assert (ctx.k, ctx.k_source) == (float(k), want)
        refused = admission.missing_inputs([], {}, ctx.k_source, pieces=False)
        assert bool(refused) == (want == "declared")


def test_i16_s1_anchors_at_point_b():
    """S1 on host DRAM per GPU at point B (7B, S 32, w 4, T_mb 4096, the
    planner's attention-inclusive c_s): stage 0 809.8 GB/s at k 10 and 162.0
    at k 2, an interior stage 383.6 and 76.7 (m2b-e7 PLAN.md, within 0.5%);
    a 600 GB/s host admits no stage-0 GPU at k 10 and 3 at k 2."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: S1 dropped the boundary bytes or the factor 2 for both
    # directions on host DRAM, used another share than 2(w-1)/w, priced c_s
    # without attention or recompute, or the Embed tables at 2 bytes.
    from wantrain.scheduler import admission

    s0, mid = _i16_stage(0, 0, 2, 4), _i16_stage(1, 2, 3, 4)
    for k, want0, want_mid in ((10, 809.8, 383.6), (2, 162.0, 76.7)):
        for st, want in ((s0, want0), (mid, want_mid)):
            dram = [x for x in admission.s1_loads([st], _i16_hosts([st]), k) if "DRAM" in x.kind]
            assert len(dram) == 4  # one 1-GPU host per peer
            assert dram[0].load / 1e9 == pytest.approx(want, rel=0.005)
    for k, fits in ((10, 0), (2, 3)):
        for n in (fits, fits + 1):
            if n == 0:
                continue
            # n of the stage's four GPUs on one 600 GB/s host, the rest elsewhere
            st = _i16_stage(0, 0, 2, 4, hosts=["shared"] * n + [f"own{q}" for q in range(4 - n)])
            refused = admission.s1([st], _i16_hosts([st]), k)
            hot = [r.host for r in refused if "DRAM" in r.kind]
            assert ("shared" in hot) == (n > fits)


def test_i16_absent_pcie_drops_out_of_s1_on_a_cpu_node():
    """A cpu node has no PCIe hop: its PCIe capacity is 'absent', S1 has no
    PCIe link for it and the inputs check never refuses it; a CUDA node whose
    PCIe was not measured is refused naming the field."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: an absent PCIe path were treated as missing (refusing every
    # cpu fleet) or as zero capacity, or a missing CUDA PCIe figure were assumed.
    from wantrain.scheduler import admission
    from wantrain.scheduler.planner import _gpu

    cpu = Node(
        "127.0.0.1",
        tflops=0.1,
        vram_bytes=8 << 30,
        device="cpu",
        hostname="mac",
        site="s0",
        gpu_name="cpu",
        host_ram_bytes=16 << 30,
        host_dram_gbps=32.0,
        nic_up_gbps=10.0,
        nic_down_gbps=10.0,
        sources={"host_ram_bytes": "measured", "host_dram_gbps": "measured"}
        | {"nic_up_gbps": "measured", "nic_down_gbps": "measured"},
        device_class=ONE_CLASS,
    )
    g = _gpu(cpu, 0, "cpu")
    assert g.source("pcie_h2d") == g.source("pcie_d2h") == "absent"
    assert g.hbm == 32e9 and g.source("hbm") == "measured"  # the device is the host
    st = _i16_stage(0, 2, 3, 2)
    st = admission.Stage(**{**st.__dict__, "peers": (g, dataclasses_replace(g, name="b"))})
    hosts = {
        "mac": admission.Host("mac", 16 << 30, 32e9, {"ram_bytes": "measured", "dram": "measured"})
    }
    kinds = {x.kind for x in admission.s1_loads([st], hosts, 10)}
    assert "S1 PCIe" not in kinds and "S1 NIC" in kinds
    assert admission.missing_inputs([st], hosts, "counted", pieces=True) == []
    cuda = _gpu(dataclasses_replace(cpu, device="cuda", gpu_name="H100"), 0, "cuda")
    st = admission.Stage(**{**st.__dict__, "peers": (cuda, dataclasses_replace(cuda, name="b"))})
    missing = admission.missing_inputs([st], hosts, "counted", pieces=True)
    assert {r.where.split()[-1] for r in missing} == {"pcie_h2d", "pcie_d2h", "hbm"}
    assert all(r.kind == "missing" for r in missing)


def test_i16_absent_refuse_outside_host_pcie_and_invalid_figures():
    """'absent' is a source only for the PCIe hop of a cpu or mps device: on a
    CUDA peer an 'absent' NIC, HBM or PCIe figure is refused naming the field
    (for pieces and zero1 alike), and a measured figure that is not > 0 is
    refused as invalid, never priced."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: a fleet could drop an unmeasured CUDA link from S1 (or price
    # an infinite PCIe, a zero HBM) by calling it 'absent', or a zero capacity
    # reached the exchange simulation.
    from wantrain.scheduler import admission

    st = _i16_stage(1, 2, 3, 2)
    hosts = _i16_hosts([st])
    claimed = dict(st.peers[0].sources, nic_up="absent", hbm="absent", pcie_h2d="absent")
    g = dataclasses_replace(st.peers[0], nic_up=None, hbm=None, sources=claimed)
    st = admission.Stage(**{**st.__dict__, "peers": (g, st.peers[1])})
    for pieces in (True, False):
        got = admission.missing_inputs([st], hosts, "counted", pieces=pieces)
        assert [(r.kind, r.where) for r in got] == [
            ("absent", "s1p0 nic_up"),
            ("absent", "s1p0 pcie_h2d"),
            ("absent", "s1p0 hbm"),
        ]
    host_pcie = dict(claimed, nic_up="measured", hbm="measured")  # only the PCIe hop absent
    cpu = tuple(
        dataclasses_replace(p, device="cpu", nic_up=5e9, hbm=4e10, sources=host_pcie)
        for p in st.peers
    )
    st_cpu = admission.Stage(**{**st.__dict__, "peers": cpu})
    assert admission.missing_inputs([st_cpu], hosts, "counted", pieces=True) == []
    zero = dataclasses_replace(cpu[1], nic_down=0.0)
    st_zero = admission.Stage(**{**st.__dict__, "peers": (cpu[0], zero)})
    got = admission.missing_inputs([st_zero], hosts, "counted", pieces=True)
    assert [(r.kind, r.where) for r in got] == [("invalid", "s1p1 nic_down")]


def dataclasses_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


def test_i16_declared_or_missing_capacity_refuse_pieces_not_zero1():
    """Every capacity carries its source: a declared NIC refuses pieces naming
    it (zero1 is priced on it), a missing one refuses every wide layout, and
    a declared k is refused; a counted or scenario k is accepted."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: a declared figure were admitted for pieces, a missing one
    # assumed, or a figure without a source treated as measured.
    from wantrain.scheduler import admission

    st = _i16_stage(0, 2, 3, 2)
    hosts = _i16_hosts([st])
    srcs = dict(st.peers[0].sources, nic_up="declared")
    st = admission.Stage(
        **{**st.__dict__, "peers": (dataclasses_replace(st.peers[0], sources=srcs), st.peers[1])}
    )
    pieces = admission.missing_inputs([st], hosts, "counted", pieces=True)
    assert [(r.kind, r.where) for r in pieces] == [("declared", "s0p0 nic_up")]
    assert admission.missing_inputs([st], hosts, "counted", pieces=False) == []
    gone = dataclasses_replace(st.peers[1], nic_down=None)
    st = admission.Stage(**{**st.__dict__, "peers": (st.peers[0], gone)})
    zero1 = admission.missing_inputs([st], hosts, "scenario", pieces=False)
    assert [(r.kind, r.where) for r in zero1] == [("missing", "s0p1 nic_down")]
    unsourced = dataclasses_replace(st.peers[1], nic_down=5e10, sources={})
    assert unsourced.source("nic_down") == "declared"
    refused = admission.missing_inputs([st], hosts, "declared", pieces=False)
    assert ("declared", "k_passes (host-DRAM passes per side)") in [
        (r.kind, r.where) for r in refused
    ]


def _i16_pipeline(S, w, M, k, dram=600e9):
    """The 7B pipeline at (S, w), M microbatches, every stage a pieces stage on
    1-GPU hosts: (stages, hosts, ZB config, lane orders, weights, owner, fabric)."""
    import types

    from wantrain.scheduler.exchange_sim import Fabric
    from wantrain.scheduler.zero_bubble import (
        ZBConfig,
        analytic_costs,
        bridge_orders,
        iwrr,
        stage_memory_model,
    )

    cfg = _gpt7b()
    part = proportional_partition(cfg, [1.0] * S)
    stages = [_i16_stage(s, a, b, w) for s, (a, b) in enumerate(part)]
    ns = types.SimpleNamespace(model=cfg, recompute=True, partition=part, mb_size=1)
    f, b, ww = analytic_costs(ns)
    agg = w * 400e12 / 4096
    mems = [stage_memory_model(ns, y - x) for x, y in part]
    zcfg = ZBConfig(
        S,
        M,
        [x / agg for x in f],
        [x / agg for x in b],
        [x / agg for x in ww],
        mem_f=[x[0] for x in mems],
        mem_b=[x[1] for x in mems],
        mem_w=[x[2] for x in mems],
    )
    weights = [[1.0] * w for _ in range(S)]
    owner = [iwrr(M, [1.0] * w) for _ in range(S)]
    orders, _sched, _ms = bridge_orders(zcfg, owner, [w] * S, weights, "1p")
    hosts = _i16_hosts(stages, dram=dram)
    fabric = Fabric(host_dram={h: x.dram for h, x in hosts.items()}, k_passes=k)
    return stages, hosts, zcfg, orders, weights, owner, fabric


def test_i16_s2_admits_point_b_stage0_and_refuses_r7():
    """S2 with E and window_s from the exchange simulation (ε 3%): point B
    stage 0 under pieces at k 2 exposes 24.344 ms against ε·C 28.78 ms and is
    admitted (m2b-e7 derive/e7_spec.json, 128xH100-400G-1gph-k2); (8, 4) and
    (4, 8) at 400 Gbps, k 10, are refused at stage 0 (77.80 > 21.05 and 154.18
    > 19.76 ms, m2b-e7 PLAN.md, 32xH100-400G-1gph-k10, ADR-0006 R7), each at
    the faster emission order as the planner takes it. Host DRAM binds none of
    them (the PLAN refuses them for S2 only)."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: S2 compared E with C instead of ε·C, ignored window_s, took
    # C at another stage than the slowest, or E from anything but SCHEDULER-I17.
    from wantrain.scheduler import admission
    from wantrain.scheduler.exchange_sim import predict_exchange

    for (S, w, M, k), want_e, want_bound, admitted in (
        ((32, 4, 128, 2), 24.344, 28.78, True),
        ((8, 4, 32, 10), 77.798, 21.05, False),
        ((4, 8, 32, 10), 154.184, 19.76, False),
    ):
        stages, _h, zcfg, orders, weights, _o, fabric = _i16_pipeline(S, w, M, k)
        layouts = {0: admission.exchange_layout(stages[0])}
        pred = predict_exchange(zcfg, orders, weights, layouts, fabric).chosen.stages
        c = admission.pipeline_compute(stages, M)
        assert pred[0].exposure * 1e3 == pytest.approx(want_e, rel=0.005)
        assert 0.03 * c * 1e3 == pytest.approx(want_bound, rel=0.005)
        refused = admission.s2(pred, c)
        assert (not refused) == admitted
        assert all(r.kind == "S2" and r.stage == 0 for r in refused)


def test_i16_device_bound_term_by_term():
    """SCHEDULER-I16's device memory per peer, each term against a hand
    computation, and DESIGN §9's figures: state b_p + 16/w + b_g·P_mod/P_s =
    6.17 B/param at w 4 and 4.17 at w 8 (7B interior, 4 blocks), own-piece
    hold b_g·M_k/w = 4 B/param at S 8, w 4, M 32 and 32 B/param on the fp32
    Embed tables at S 32, w 4, M 128."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: a term were dropped or mispriced: the owned state at other
    # than 16/w, the own hold without M_k or at the stage's b_g instead of each
    # unit's, the module dW at the stage size, the boundaries without min(S, M/w).
    from wantrain.scheduler import admission

    h, t = 4096, 4096
    mid = _i16_stage(1, 1, 5, 4)
    p = 4 * 12 * h * h
    assert mid.params() == p
    terms = admission.device_terms(mid, 0, 8, 8, 32)
    assert terms == {
        "working": 2 * p,
        "owned_state": 16 * p / 4,
        "own_hold": 8 * 2 * p / 4,
        "module_dw": 2 * 4 * h * h,
        "boundaries": min(8, 32 // 4) * t * h * 2,
        "pending_w": 4 * t * 16 * h * 4,
        "logits": 0.0,
    }
    state = (terms["working"] + terms["owned_state"] + terms["module_dw"]) / p
    assert state == pytest.approx(6.17, rel=0.005)
    assert terms["own_hold"] / p == pytest.approx(4.0, rel=0.005)
    w8 = admission.device_terms(_i16_stage(1, 1, 5, 8), 0, 4, 8, 32)
    assert (w8["working"] + w8["owned_state"] + w8["module_dw"]) / p == pytest.approx(
        4.17, rel=0.005
    )
    s0 = _i16_stage(0, 0, 2, 4)
    t0 = admission.device_terms(s0, 1, 32, 32, 128)
    emb = s0.units[0]
    assert emb.b_g == 4 and emb.params == (32000 + 4096) * h
    assert t0["own_hold"] == 32 * (4 * emb.params + 2 * 12 * h * h) / 4
    assert 32 * 4 * emb.params / 4 / emb.params == pytest.approx(32.0)
    assert t0["boundaries"] == min(32, 128 // 4) * t * h * 2
    last = admission.device_terms(_i16_stage(31, 32, 34, 4), 0, 32, 32, 128)
    assert last["logits"] == 2 * t * 32000 * 4


def test_i16_device_refuse_names_the_peer():
    """A fleet that passes MEM-HOST and fails device memory is refused naming
    the peer; the control that drops the own-piece hold b_g·M_k·P_s/w admits
    the same fleet (the hold is what refuses)."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the own-piece hold were left out of the device term, or a
    # device refusal named no peer.
    from wantrain.scheduler import admission

    st = _i16_stage(0, 0, 2, 4)  # point B stage 0: 32 B/param of fp32 hold on the tables
    hosts = _i16_hosts([st])
    owned = {0: [32] * 4}
    terms = admission.device_terms(st, 0, 32, 32, 128)
    need = sum(terms.values())
    vram = int((need - terms["own_hold"] / 2) / 0.88) + 1  # fits without the hold, not with it
    small = admission.Stage(
        **{**st.__dict__, "peers": tuple(dataclasses_replace(g, vram_bytes=vram) for g in st.peers)}
    )
    assert admission.mem_host([small], hosts, owned, 128) == []
    refused = admission.device_memory([small], owned, 32, 128)
    assert [r.where for r in refused] == [f"stage 0 peer s0p{q}" for q in range(4)]
    assert all(r.kind == "MEM" and r.value > r.bound for r in refused)

    real = admission.device_terms

    def no_hold(*a):
        out = dict(real(*a))
        out["own_hold"] = 0.0
        return out

    admission.device_terms = no_hold
    try:
        assert admission.device_memory([small], owned, 32, 128) == []
    finally:
        admission.device_terms = real


def test_i16_mem_host_reproduces_round5_and_sums_per_host():
    """The one-step host bound (round5.md §4, within 0.5%): 21.74 GB at an
    interior stage of S 8, w 4, M 32 (M_k 8), 24.96 at M_k 12, 19.93 at an
    interior stage of point B, 49.21 on stage 0 and 32.91 on the last; peers
    of every stage on one host are summed against its one RAM figure less the
    reserve, and the refusal names the host."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the host bound used the stage's b_g for the fp32 tables,
    # dropped the gathers, charged M_k on the wrong side, or checked hosts per
    # stage instead of summing every peer placed on them.
    from wantrain.scheduler import admission

    cases = (
        (_i16_stage(1, 1, 5, 4), 8, 32, 21.74),
        (_i16_stage(1, 1, 5, 4), 12, 32, 24.96),
        (_i16_stage(1, 2, 3, 4), 32, 128, 19.93),
        (_i16_stage(0, 0, 2, 4), 32, 128, 49.21),
        (_i16_stage(31, 32, 34, 4), 32, 128, 32.91),
    )
    for st, m_k, m, want in cases:
        assert admission.host_hold(st, 0, m_k, m) / 1e9 == pytest.approx(want, rel=0.005)
    s0 = _i16_stage(0, 0, 2, 4, hosts=["H"] * 4)
    s1 = _i16_stage(1, 2, 3, 4, hosts=["H"] * 4)
    total = 4 * admission.host_hold(s0, 0, 32, 128) + 4 * admission.host_hold(s1, 0, 32, 128)
    owned = {0: [32] * 4, 1: [32] * 4}
    for ram, refused in ((int(total) + (40 << 30), False), (int(total * 1.05), True)):
        hosts = {
            "H": admission.Host("H", ram, 600e9, {"ram_bytes": "measured", "dram": "measured"})
        }
        got = admission.mem_host([s0, s1], hosts, owned, 128)
        assert bool(got) == refused
        if refused:
            assert got[0].kind == "MEM-HOST" and got[0].host == "H" and got[0].value == total


def test_i16_mem_host_pinned_chunks_count_on_cuda_only():
    """MEM-HOST adds each peer's pinned copy bytes in flight on CUDA: a host
    one pinned budget short of the bound is refused naming the host, and the
    same host as cpu nodes (no pinned copies) is admitted."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the pinned bytes were ignored on CUDA or charged on cpu.
    from wantrain.scheduler import admission

    st = _i16_stage(1, 2, 3, 4, hosts=["H"] * 4)
    hold = 4 * admission.host_hold(st, 0, 32, 128)
    pinned = 512 << 20
    ram = int(hold + 3 * pinned) + (8 << 30)
    cap = ram - admission.host_reserve(ram)
    assert hold <= cap < hold + 4 * pinned
    hosts = {"H": admission.Host("H", ram, 600e9, {"ram_bytes": "measured", "dram": "measured"})}
    owned = {1: [32] * 4}
    got = admission.mem_host([st], hosts, owned, 128, pinned_bytes_per_peer=pinned)
    assert [(r.kind, r.host) for r in got] == [("MEM-HOST", "H")]
    cpu = admission.Stage(
        **{**st.__dict__, "peers": tuple(dataclasses_replace(g, device="cpu") for g in st.peers)}
    )
    assert admission.mem_host([cpu], hosts, owned, 128, pinned_bytes_per_peer=pinned) == []


def test_i16_class_refuse_mixed_models_and_declared_fields():
    """The probed-class rule (BA-31): peers of different device type or model
    are refused naming both classes; for pieces a declared device or gpu_name
    is refused naming the field; zero1 accepts a declared but uniform class."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the class came from the node label or a declared field, or
    # a stage mixing device models were admitted.
    from wantrain.scheduler import admission

    g = _i16_stage(0, 2, 3, 2).peers
    assert admission.check_class(g) == []
    mixed = (g[0], dataclasses_replace(g[1], gpu_name="H100 PCIe"))
    (r,) = admission.check_class(mixed)
    assert r.kind == "class" and "cuda:H100" in r.value and "cuda:H100 PCIe" in r.value
    declared = (g[0], dataclasses_replace(g[1], declared=frozenset({"gpu_name"})))
    assert [(r.kind, r.where) for r in admission.check_class(declared)] == [
        ("declared", "s0p1 gpu_name")
    ]
    assert admission.check_class(declared, probed=False) == []


def test_i16_recheck_after_a_rebalance():
    """admission.recheck re-runs device memory, MEM-HOST and S2 for a given
    mb_owner and orders (BA-44, BA-47): point B stage 0 is admitted with
    equal ownership, and refused naming the peer when a rebalance hands one
    peer most microbatches (its own-piece hold grows with M_k)."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the re-check read M_k from anything but the given mb_owner,
    # or skipped device memory after an ownership change.
    from wantrain.scheduler import admission
    from wantrain.scheduler.zero_bubble import bridge_orders

    stages, hosts, zcfg, orders, weights, owner, fabric = _i16_pipeline(32, 4, 128, 2)
    st = stages[0]
    s0 = [  # 32 GB devices: ~13 GB at M_k 32, ~31 GB at M_k 112
        admission.Stage(
            **{
                **st.__dict__,
                "peers": tuple(dataclasses_replace(g, vram_bytes=32 * 10**9) for g in st.peers),
            }
        )
    ]
    assert admission.recheck(s0, hosts, zcfg, orders, weights, owner, fabric) == []
    skew = [list(o) for o in owner]
    skew[0] = [0 if mb % 8 else 1 for mb in range(128)]  # peer 0 owns 112 of 128
    skew[0][2], skew[0][3] = 2, 3
    skew_orders, _s, _m = bridge_orders(zcfg, skew, [4] * 32, weights, "1p")
    refused = admission.recheck(s0, hosts, zcfg, skew_orders, weights, skew, fabric)
    assert any(r.kind == "MEM" and r.where == "stage 0 peer s0p0" for r in refused)


# ============================================================ M2b: plan mode for wide stages


def _contract(c):
    """The frozen finalize contract (I-7) for one candidate the planner can emit."""
    assert c["piece_bounds"] is None
    if c["w"] == 1:
        assert (c["shard"], c["wire"], c["stream_grads"]) == ("zero1", "fp32", False)
        return
    if c["shard"] == "pieces":
        assert c["wire"] == "native" and c["stream_grads"] is True
        if c.get("admitted", True):  # what the planner can emit
            assert all(len({tuple(x) for x in row}) == 1 for row in c["classes"])
            assert not c["tied_cross_stage"]
    if c["wire"] == "native":
        assert c["stream_grads"] is True


def test_plan_wide_chooses_pieces_where_i16_admits():
    """7B on 32 H100 (8-GPU hosts, k 2, 400 Gbps): the planner places
    S 16, w 2, pieces + native (m2b-e7) and returns the M2b fields on its plan
    dict and, by attribute, on the spec: widths, [stage][peer] placement,
    shard, wire, emission order ('store' is the simulation's 'backward'),
    piece_bounds None, stream_grads."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the planner kept one node per stage, picked anything but
    # the smallest simulated admitted step, or returned the fields only in the
    # rationale (or with the simulation's emission names), or its decision text
    # still described the width-1 cut.
    from wantrain.scheduler.planner import EMISSION_SPEC

    out = _plan7b(_e7_fleet(_e7_config("32xH100-400G-8gph-k2")))
    spec = out["spec"]
    assert out["rationale"]["chosen"] == "S 16 w 2 pieces + native"
    assert (out["shard"], out["wire"], out["stream_grads"]) == ("pieces", "native", True)
    assert out["stage_widths"] == spec.stage_widths == [2] * 16
    assert [len(row) for row in out["placement"]] == [2] * 16
    assert [p["node"] for row in out["placement"] for p in row] == list(range(32))
    assert (spec.shard, spec.wire, spec.stream_grads, spec.piece_bounds) == (
        "pieces",
        "native",
        True,
        None,
    )
    assert out["emission_order"] == spec.emission_order
    assert set(spec.emission_order) <= set(EMISSION_SPEC.values()) == {"store", "forward"}
    assert EMISSION_SPEC == {"backward": "store", "forward": "forward"}
    assert spec.endpoints is None and spec.devices == [["cuda", "cuda"]] * 16
    admitted = [c for c in out["candidates"] if c["admitted"]]
    best = min(c["step_s"] for c in admitted)
    chosen = next(c for c in admitted if c["name"] == out["rationale"]["chosen"])
    assert chosen["step_s"] == best  # F9's second clause holds by construction
    # the decision the coordinator logs names the plan returned, not the width-1 cut
    assert out["rationale"]["decision"].startswith("S 16 w 2 pieces + native: ")


def test_plan_wide_unarmed_refuse_pieces_and_native():
    """allow_pieces False (BA-49): every pieces or native candidate is refused
    'not armed' and listed in the rationale; the best remaining candidate is
    chosen (fp32 zero1 or narrower); the default is False."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: a pieces or native plan reached a caller that did not arm
    # it, or the refusal were silent.
    import inspect

    from wantrain.scheduler.planner import NOT_ARMED

    assert inspect.signature(plan_job).parameters["allow_pieces"].default is False
    assert inspect.signature(fleetplan.build_spec).parameters["allow_pieces"].default is False
    out = _plan7b(_e7_fleet(_e7_config("32xH100-400G-8gph-k2")), allow_pieces=False)
    assert out["shard"] == "zero1" and out["wire"] == "fp32"
    assert (out["spec"].shard, out["spec"].wire) == ("zero1", "fp32")
    for c in out["candidates"]:
        unarmed = c["w"] > 1 and (c["shard"] == "pieces" or c["wire"] == "native")
        assert unarmed == any(r["kind"] == NOT_ARMED for r in c["reasons"])
        if unarmed:
            assert not c["admitted"] and any(
                NOT_ARMED in t for t in out["rationale"]["refused"][c["name"]]
            )
    admitted = [c for c in out["candidates"] if c["admitted"]]
    assert {c["shard"] for c in admitted} == {"zero1"} and {c["wire"] for c in admitted} == {"fp32"}
    assert out["rationale"]["chosen"] == min(admitted, key=lambda c: c["step_s"])["name"]


def test_plan_wide_shard_refuse_when_no_candidate_of_that_layout_is_admitted():
    """job shard restricts the candidates: 'pieces' on 4-GPU hosts at k 10
    (S1 host DRAM refuses every width) refuses the job naming every reason;
    'zero1' never yields pieces."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the shard restriction fell back to another layout silently,
    # or the refusal named no link.
    x = _e7_config("32xH100-400G-4gph-k10")
    out = _plan7b(_e7_fleet(x), shard="pieces")
    assert out["spec"] is None
    assert "no candidate is admitted" in out["rationale"]["decision"]
    for name in ("S 16 w 2 pieces + native", "S 8 w 4 pieces + native", "S 4 w 8 pieces + native"):
        assert any("S1 host DRAM" in t for t in out["rationale"]["refused"][name])
    assert any("shard" in t for t in out["rationale"]["refused"]["S 32 w 1"])
    z = _plan7b(_e7_fleet(x), shard="zero1")
    assert z["spec"] is not None and z["shard"] == "zero1"
    assert all(not c["admitted"] for c in z["candidates"] if c["shard"] == "pieces")


def test_plan_wide_finalize_contract():
    """The frozen finalize contract (I-7) over every candidate the planner can
    emit, on fleets that admit pieces, mix classes and tie embeddings: pieces
    only with native and stream_grads, native on a wide plan only with
    stream_grads, piece_bounds None, pieces only on one probed class per stage
    and never with the tied Embed and Head on different stages."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the planner could emit a combination stage-pieces' finalize
    # refuses (pieces + fp32, native batch exchange, bounds, mixed or tied pieces).
    # CONTROL-I25 (the human, 2026-10-11): a fleet mixing classes emits no
    # candidate at all; it is refused before any cost is computed.
    with pytest.raises(ValueError, match="^MIXED-FLEET: the device map spans 2 device classes"):
        _plan7b(_e7_fleet(_e7_config("32xGPU-mixed-classes")))
    outs = [
        _plan7b(_e7_fleet(_e7_config("32xH100-400G-8gph-k2"))),
        _plan7b(
            _e7_fleet(_e7_config("32xH100-400G-8gph-k2")),
            cfg=_gpt7b(tie_embeddings=True, init_scheme="gpt2"),
        ),
    ]
    for out in outs:
        assert out["candidates"]
        for c in out["candidates"]:
            _contract(c)
    tied = outs[1]
    assert tied["shard"] != "pieces"
    assert all(
        any(r["kind"] == "tied" for r in c["reasons"])
        for c in tied["candidates"]
        if c["shard"] == "pieces" and c["w"] > 1
    )
    for out in outs:
        _contract(
            dict(
                w=max(out["stage_widths"]),
                shard=out["shard"],
                wire=out["wire"],
                stream_grads=out["stream_grads"],
                piece_bounds=out["piece_bounds"],
                classes=[[[p["device"], p["gpu_name"]] for p in row] for row in out["placement"]],
                tied_cross_stage=False,
            )
        )


def test_plan_wide_width1_fleets_plan_as_today():
    """Fleets without plan-mode inputs plan exactly as today: two and three
    nodes (no wide candidate), and four nodes, whose wide candidates are
    refused naming the missing measurements (width 1 needs none of them)."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: plan mode changed a width-1 plan, or priced a wide
    # candidate on an assumed capacity.
    from wantrain.scheduler.planner import _plan_depth

    cfg = make_config("gpt124m")
    for n in (2, 3, 4):
        nodes = [
            Node(
                f"n{i}",
                tflops=80.0 - 10 * i,
                vram_bytes=24 << 30,
                compute_cap=(8, 9),
                device_class=ONE_CLASS,
            )
            for i in range(n)
        ]
        out = plan_job(make_config("gpt124m"), _fleet(nodes), steps=10, device="cuda")
        ref = _plan_depth(cfg, _fleet(nodes), steps=10, device="cuda")
        assert out["spec"].to_json() == ref["spec"].to_json()
        assert out["stage_widths"] == [1] * n and out["shard"] == "zero1"
        wide = [c for c in out["candidates"] if c["w"] > 1]
        assert bool(wide) == (n >= 4)
        for c in wide:
            assert not c["admitted"] and {r["kind"] for r in c["reasons"]} <= {
                "missing",
                NOT_ARMED_KIND,
            }
    armed = plan_job(
        make_config("gpt124m"), _fleet(nodes), steps=10, device="cuda", allow_pieces=True
    )
    assert armed["spec"].to_json() == ref["spec"].to_json()
    assert all(
        any(r["kind"] == "missing" for r in c["reasons"]) for c in armed["candidates"] if c["w"] > 1
    )


NOT_ARMED_KIND = "not armed"


def test_plan_wide_override_refuse_a_pinned_partition():
    """Overrides win: a job that pins its partition (a width-1 cut) gets that
    cut, and every wide candidate is refused naming the override; the control
    without it plans a wide zero1 stage on the same fleet."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: a wide plan silently replaced the partition the job pinned.
    from wantrain.scheduler.planner import Overrides

    srcs = {k: "measured" for k in ("host_ram_bytes", "host_dram_gbps")}
    srcs |= {"nic_up_gbps": "measured", "nic_down_gbps": "measured"}
    nodes = [
        Node(
            f"127.0.0.{i}",
            tflops=0.05,
            vram_bytes=8 << 30,
            device="cpu",
            hostname=f"m{i}",
            site="s0",
            gpu_name="cpu",
            host_ram_bytes=64 << 30,
            host_dram_gbps=40.0,
            nic_up_gbps=10.0,
            nic_down_gbps=10.0,
            sources=dict(srcs),
            device_class=ONE_CLASS,
        )
        for i in range(4)
    ]
    pinned = [(0, 2), (2, 4), (4, 6), (6, 8)]
    for ov, wide in (
        (Overrides(microbatches=4), True),
        (Overrides(microbatches=4, partition=pinned), False),
    ):
        out = plan_job(make_config("toy"), _fleet(nodes), steps=1, device="cpu", overrides=ov)
        assert (max(out["stage_widths"]) > 1) == wide, out["rationale"]["chosen"]
        kinds = [{r["kind"] for r in c["reasons"]} for c in out["candidates"] if c["w"] > 1]
        assert any("override" in k for k in kinds) == (not wide)
        if not wide:  # unarmed layouts stay 'not armed'; every other one names the override
            assert all(k in ({NOT_ARMED_KIND}, {"override"}) for k in kinds)
        if not wide:
            assert [tuple(x) for x in out["spec"].partition] == pinned


def test_plan_wide_more_devices_than_cut_points():
    """More devices than the layer list has entries: the width-1 candidate is
    the deepest cut (one stage per block) on fewer devices, the fastest ones."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: plan_job refused a fleet larger than the model, or cut
    # deeper than the model allows.
    cfg = make_config("toy")  # 6 blocks: 8 entries
    nodes = [
        Node(
            f"n{i}", tflops=1.0 + (i % 3), vram_bytes=8 << 30, device="cpu", device_class=ONE_CLASS
        )
        for i in range(12)
    ]
    out = plan_job(cfg, _fleet(nodes), steps=1, device="cpu")
    assert out["spec"] is not None and out["spec"].stages == cfg.n_layer == 6
    assert out["stage_widths"] == [1] * 6
    chosen = sorted(p["node"] for row in out["placement"] for p in row)
    assert chosen == sorted(sorted(range(12), key=lambda i: -nodes[i].tflops)[:6])


def test_plan_wide_class_refuse_groups_by_probed_model_not_label():
    """Grouping is by (device, probed gpu_name), never by node label (closes
    M2-R5-FIT-2): 17 + 15 GPUs of two models refuse every wide candidate
    naming the class of the stage that holds GPUs 16 and 17; relabelling the
    nodes changes nothing, and one model under many labels is admitted."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: classes were grouped by node label, or a fleet mixing
    # device models were admitted for any layout (CONTROL-I25).
    # CONTROL-I25 (the human, 2026-10-11): the 17 + 15 fleet spans two device
    # classes and is refused MIXED-FLEET before any candidate, whatever the labels.
    x = _e7_config("32xGPU-mixed-classes")
    fleet = _e7_fleet(x)
    for i, n in enumerate(fleet.nodes):
        n.label = "same" if i % 2 else f"label{i}"
    with pytest.raises(ValueError) as e:
        _plan7b(fleet)
    assert str(e.value).startswith("MIXED-FLEET: the device map spans 2 device classes ")
    assert "cuda:NVIDIA H100: " in str(e.value) and "cuda:NVIDIA H100 PCIe: " in str(e.value)
    same = _e7_fleet(_e7_config("32xH100-400G-8gph-k2"))
    for i, n in enumerate(same.nodes):
        n.label = f"label{i % 3}"
    assert not any(r["kind"] == "class" for c in _plan7b(same)["candidates"] for r in c["reasons"])


def test_plan_wide_mem_host_refuse_names_the_host():
    """20 GB of host RAM after the reserve on 1-GPU hosts (m2b-e7
    memhost20): every wide pieces candidate is refused MEM-HOST naming a host
    of a stage-0 peer; zero1 keeps today's admission (SCHEDULER-I16, SPEC
    task 4), so the PLAN's choice, S 16 w 2 zero1 + native, is planned."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: MEM-HOST were checked per stage instead of per host, named
    # no host, skipped pieces, or were charged on zero1 too.
    x = _e7_config("32xH100-400G-1gph-k10-memhost20")
    out = _plan7b(_e7_fleet(x))
    chosen = next(c for c in out["candidates"] if c["name"] == out["rationale"]["chosen"])
    assert _e7_match(x["expected_choice"], chosen)
    assert out["rationale"]["chosen"] == "S 16 w 2 zero1 + native"
    wide = [c for c in out["candidates"] if c["w"] > 1 and c["wire"] == "native"]
    assert any(c["shard"] == "zero1" for c in wide)
    for c in wide:
        mh = [r for r in c["reasons"] if r["kind"] == "MEM-HOST"]
        if c["shard"] == "zero1":
            assert not mh
            continue
        assert mh and mh[0]["stage"] == 0 and mh[0]["host"] and mh[0]["value"] > mh[0]["bound"]


def test_plan_wide_uplink_refuse_straddling_stages():
    """Two sites of four H100 joined by one 10 Gbps uplink: a stage whose
    peers straddle the uplink is refused under pieces naming it (S1 uplink),
    the stage-contiguous placement is planned."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the uplink were not a link of S1, or straddling stages
    # were priced at the NIC rate.
    nodes = [_h100(g, 1, site=f"site{g // 4}") for g in range(8)]
    fleet = Fleet(
        nodes=nodes,
        link_mbps=10_000.0,
        rtt_ms=50.0,
        duplex="per_direction",
        k_passes=(2, "scenario"),
    )
    out = _plan7b(fleet)
    names = {c["name"]: c for c in out["candidates"]}
    straddle = names["S 4 w 2 pieces + native, straddling"]
    up = [r for r in straddle["reasons"] if r["kind"] == "S1 uplink"]
    assert up and all(r["value"] > r["bound"] == 1.25e9 for r in up)
    assert {r["where"] for r in up} == {"site0->site1", "site1->site0"}
    assert not any(r["kind"] == "S1 uplink" for r in names["S 4 w 2 pieces + native"]["reasons"])
    assert "straddling" not in out["rationale"]["chosen"]


def test_plan_wide_build_spec_reads_the_plan_node_keys():
    """fleetplan.build_spec reads the M2b node keys (hostname, site, probed
    class, capacities with their sources) and k_passes, passes allow_pieces
    and job shard through, and lays endpoints and devices out [stage][peer]."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: build_spec dropped a capacity key or its source, ignored
    # allow_pieces or job shard, or mapped nodes to stages one to one.
    x = _e7_config("32xH100-400G-8gph-k2")
    nodes = []
    for r, n in enumerate(_e7_fleet(x).nodes):
        d = {
            k: getattr(n, k)
            for k in (
                "host",
                "tflops",
                "vram_bytes",
                "device",
                "device_class",
                "hostname",
                "site",
                "gpu_name",
            )
        }
        d.update(
            {
                k: getattr(n, k)
                for k in ("host_ram_bytes", "host_dram_gbps", "pcie_h2d_gbps", "pcie_d2h_gbps")
            }
        )
        d.update(nic_up_gbps=n.nic_up_gbps, nic_down_gbps=n.nic_down_gbps, hbm_gbps=n.hbm_gbps)
        d.update(
            rank=r,
            compute_cap=[9, 0],
            dial_port=7000 + r,
            listen_port=8000 + r,
            sources=dict(n.sources),
        )
        nodes.append(d)
    job = dict(
        n_layer=32,
        n_head=32,
        d_model=4096,
        seq_len=4096,
        vocab_size=32000,
        compute_dtype="bf16",
        boundary_dtype="bf16",
        attn_impl="sdpa",
        sdpa_backend="flash",
        mb_size=1,
        microbatches=32,
        recompute=True,
        steps=1,
        device="cuda",
    )
    fleet = {
        "run_id": "e7",
        "nodes": nodes,
        "link": {"mbps": 400_000, "rtt_ms": 0, "duplex": "per_direction"},
        "k_passes": {"value": 2, "source": "scenario"},
        "job": job,
    }
    armed = fleetplan.build_spec(fleet, allow_pieces=True)
    spec = armed["spec"]
    assert (armed["shard"], spec.shard, spec.stage_widths) == ("pieces", "pieces", [2] * 16)
    assert spec.endpoints[0] == [["10.0.0.0", 7000, 8000], ["10.0.0.1", 7001, 8001]]
    assert spec.devices == [["cuda", "cuda"]] * 16
    plain = fleetplan.build_spec(fleet)
    assert (plain["shard"], plain["wire"]) == ("zero1", "fp32")
    zero1 = fleetplan.build_spec(dict(fleet, job=dict(job, shard="zero1")), allow_pieces=True)
    assert zero1["shard"] == "zero1"
    del nodes[0]["sources"]["nic_up_gbps"]  # a figure nobody vouches for: declared
    claimed = fleetplan.build_spec(fleet, allow_pieces=True)
    assert claimed["shard"] != "pieces"
    for c in (c for c in claimed["candidates"] if c["shard"] == "pieces" and c["w"] > 1):
        assert any(r["kind"] == "declared" and r["where"].endswith("nic_up") for r in c["reasons"])


# ============================================================ M2b: E7 (F9)


def _e7_match(want, c):
    """Does planner candidate c have the pinned (S, w, shard, wire, gpus_used)?"""
    if want["w"] == 1:
        return (c["S"], c["w"], c["gpus_used"]) == (want["S"], 1, want["gpus_used"])
    return (c["S"], c["w"], c["shard"], c["wire"], c["gpus_used"], c["placement"]) == (
        want["S"],
        want["w"],
        want["shard"],
        want["wire"],
        want["gpus_used"],
        "contiguous",
    )


def _e7_refusal_matches(x, pinned, cands):
    """A pinned refused candidate is refused in the planner's rationale with
    every pinned reason's kind at the pinned place, its value on the refusing
    side of its bound."""
    import re

    name = pinned["candidate"]
    m = re.match(r"S (\d+) w (\d+) (pieces|zero1 \+ native)(, straddling)?$", name)
    S, w = int(m.group(1)), int(m.group(2))
    shard = "pieces" if m.group(3) == "pieces" else "zero1"
    place = "straddling" if m.group(4) else "contiguous"
    c = next(
        (
            c
            for c in cands
            if (c["S"], c["w"], c["shard"], c["wire"], c["placement"])
            == (S, w, shard, "native", place)
        ),
        None,
    )
    if c is None or c["admitted"]:
        return False
    kinds_of = {"first": [0], "last": [S - 1], "mid": list(range(1, S - 1))}
    for want in pinned["reasons"]:
        got = [r for r in c["reasons"] if r["kind"] == want["kind"]]
        where = want["where"]
        if want["kind"] in ("S1 NIC", "MEM-HOST", "S2", "class"):
            kind = re.search(r"(first|mid|last)", where)
            idx = re.search(r"stage (\d+)", where)
            stages = [int(idx.group(1))] if idx else kinds_of[kind.group(1)]
            got = [
                r
                for r in got
                if r["stage"] in stages
                or (
                    want["kind"] == "MEM-HOST"
                    and any(
                        s in stages
                        for s in map(int, re.findall(r"\d+", r["where"].split("stages")[-1]))
                    )
                )
            ]
        elif want["kind"] == "S1 host DRAM":
            stg = re.search(r"stages \[([\d, ]+)\]", where).group(1)
            got = [r for r in got if r["where"].endswith(f"(stages [{stg}])")]
        elif want["kind"] == "S1 uplink":
            got = [r for r in got if r["where"] == "site0->site1"]
        ok = [r for r in got if not isinstance(r["value"], float) or r["value"] > r["bound"]]
        if not ok:
            return False
    return True


def test_e7_planner_choices_match_the_pinned_plan():
    """E7 (ADR-0006 F9; m2b-e7/PLAN.json, granted): on each of the 17 pinned
    fleets the planner chooses the pinned plan, or under the PLAN's tie rule
    its runner-up, refuses every pinned refused candidate for the pinned
    reasons, and never chooses a plan more than 1% above the best admitted
    candidate's simulated step. Prints the metrics line PLAN.json reads."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: S1, S2, the class rule or MEM-HOST were dropped or
    # mispriced, the planner chose by bytes instead of the simulated step, or the
    # recipe ran without recompute (2 of 14 choices flip).
    import json

    with open(_E7) as f:
        plan = json.load(f)
    tie = plan["tieRule"]
    choices = refusals = f9 = 0
    misses = []
    suspended = (0, 0)
    for x in plan["configs"]:
        if x["id"] == "32xGPU-mixed-classes":
            # CONTROL-I25 (the human, 2026-10-11) supersedes this pinned fleet: two
            # device classes are refused before any candidate is priced.
            with pytest.raises(ValueError, match="^MIXED-FLEET: the device map spans 2"):
                _plan7b(_e7_fleet(x))
            suspended = (1, len(x["refused"]))
            continue
        out = _plan7b(_e7_fleet(x))
        cands = out["candidates"]
        admitted = [c for c in cands if c["admitted"]]
        chosen = next(c for c in cands if c["name"] == out["rationale"]["chosen"])
        best = (
            min(c["step_s"] for c in admitted if c["step_s"] is not None)
            if any(c["step_s"] is not None for c in admitted)
            else None
        )
        if best is not None and chosen["step_s"] > best * (1 + tie["within_best_pct"] / 100):
            f9 += 1
        ok = _e7_match(x["expected_choice"], chosen)
        if not ok and x["tie"] and x["margin_pct"] <= tie["margin_pct_max"]:
            ok = _e7_match(x["runner_up"], chosen)
        choices += ok
        if not ok:
            misses.append((x["id"], chosen["name"]))
        for r in x["refused"]:
            hit = _e7_refusal_matches(x, r, cands)
            refusals += hit
            if not hit:
                misses.append((x["id"], "refusal: " + r["candidate"]))
    print(
        json.dumps(
            {"choices_matching": choices, "refusals_matching": refusals, "f9_violations": f9}
        )
    )
    expected = plan["expected"]
    assert f9 == expected["f9_violations"]
    assert suspended != (0, 0)
    assert (choices + suspended[0], refusals + suspended[1]) == (
        expected["choices_matching"],
        expected["refusals_matching"],
    ), misses


def _toy_cpu_plan_fleet(tflops, shard="pieces"):
    """Four probed cpu workers of one host (the LocalCluster of
    pieces_plan_oracle): every capacity measured, PCIe absent."""
    measured = ("host_ram_bytes", "host_dram_gbps", "nic_up_gbps", "nic_down_gbps")
    nodes = [
        dict(
            rank=i,
            host="127.0.0.1",
            tflops=t,
            vram_bytes=16 << 30,
            dial_port=7000 + i,
            listen_port=8000 + i,
            device="cpu",
            hostname="mac",
            site="",
            gpu_name="cpu",
            host_ram_bytes=16 << 30,
            host_dram_gbps=20.0,
            nic_up_gbps=20.0,
            nic_down_gbps=20.0,
            sources={k: "measured" for k in (*measured, "device", "gpu_name")}
            | {"pcie_h2d_gbps": "absent", "pcie_d2h_gbps": "absent"},
            device_class=ONE_CLASS,
        )
        for i, t in enumerate(tflops)
    ]
    job = {"model": "toy", "steps": 3, "device": "cpu", "mb_size": 64, "microbatches": 8}
    return {
        "run_id": "toy",
        "nodes": nodes,
        "link": {"mbps": 20000, "rtt_ms": 0.2, "duplex": "per_direction"},
        "job": dict(job, shard=shard),
    }


@pytest.mark.parametrize("skew", [0.0, 0.1, 0.2, 0.4])
def test_plan_pieces_toy_cpu_fleet_admits_probed_speed_skew(skew):
    """The toy pieces plan on four probed cpu workers is admitted by S2
    whatever the spread of their probed speeds (a loaded Mac spreads them
    20-40%): the faster peer's wait for the slower one is the lane bubble,
    priced in the period, never exposed exchange (M2b round-1 review: S2
    refused this plan at >= ~10% skew, so pieces_plan_oracle passed or failed
    with the machine's load)."""
    # covers: SCHEDULER-I16, SCHEDULER-I17
    # WOULD-FAIL-IF: X_s were measured from each peer's own compute end, so a
    # 10% speed skew put ~30 ms of lane bubble into E against eps*C = 14 ms.
    t = [0.2, 0.2 * (1 - skew), 0.2, 0.2 * (1 - skew)]
    out = fleetplan.build_spec(_toy_cpu_plan_fleet(t), allow_pieces=True)
    assert (out["shard"], out["stage_widths"]) == ("pieces", [2, 2])
    wide = [c for c in out["candidates"] if c["shard"] == "pieces" and c["w"] == 2]
    assert wide and all(c["admitted"] for c in wide)


def test_plan_refused_names_every_candidate_refusal():
    """When every candidate is refused, PlanRefused names why each was
    (SCHEDULER-I16: the number, link or class), not only the width-1
    precision ladder."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: build_spec raised PlanRefused with the report alone,
    # dropping the per-candidate refusals the rationale holds.
    with pytest.raises(fleetplan.PlanRefused) as e:
        fleetplan.build_spec(_toy_cpu_plan_fleet([0.2] * 4), allow_pieces=False)
    msg = str(e.value)
    assert "no candidate is admitted" in msg
    assert "S 2 w 2 pieces + native: not armed" in msg


def test_wide_placements_leave_out_the_slowest_devices():
    """SPEC-scheduler task 4: a wide candidate on fewer devices than the fleet
    has leaves out the slowest, as the width-1 cut does, whatever their site
    and host order; ties keep that order."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the contiguous or straddling packing dropped the devices
    # last in site and host order (here the fastest) instead of the slowest.
    import types

    from wantrain.scheduler import planner

    tf = {0: 1.0, 1: 1.0, 2: 4.0, 3: 4.0, 4: 1.0, 5: 4.0, 6: 4.0}
    site = {0: "a", 1: "a", 2: "a", 3: "a", 4: "b", 5: "b", 6: "b"}
    ctx = types.SimpleNamespace(
        order=list(range(7)),
        sites=["a", "b"],
        fleet=types.SimpleNamespace(
            nodes={i: types.SimpleNamespace(tflops=tf[i], site=site[i]) for i in tf}
        ),
    )
    got = dict(planner._placements(ctx, 2, 2))
    assert got["contiguous"] == [[2, 3], [5, 6]]
    assert got["straddling"] == [[2, 5], [3, 6]]
    tied = types.SimpleNamespace(**dict(vars(ctx), sites=["a"]))
    tied.fleet = types.SimpleNamespace(
        nodes={i: types.SimpleNamespace(tflops=1.0, site="a") for i in tf}
    )
    assert dict(planner._placements(tied, 2, 2))["contiguous"] == [[0, 1], [2, 3]]


# ============================================================ M4: HF plans (SCHEDULER-I22)
# A census fixture as the builder child writes it (HANDOUT-M4 I-10), no
# transformers: a tiny llama-like decoder, V 96, h 32, 4 heads (2 kv, d 8),
# SwiGLU f 80, two layers. Per layer: q 32x32, k and v 16x32, o 32x32, gate
# and up 80x32, down 32x80, two norms: 10,816 parameters, 10,752 linear MACs
# per token, 560 pending-W values per token (in + out of each 2-D weight).
_HF_V, _HF_H, _HF_F = 96, 32, 80


def _hf_census(tied=False, softcap=None):
    h, f, v = _HF_H, _HF_F, _HF_V
    shapes = [
        ["self_attn.q_proj.weight", [h, h]],
        ["self_attn.k_proj.weight", [16, h]],
        ["self_attn.v_proj.weight", [16, h]],
        ["self_attn.o_proj.weight", [h, h]],
        ["mlp.gate_proj.weight", [f, h]],
        ["mlp.up_proj.weight", [f, h]],
        ["mlp.down_proj.weight", [h, f]],
        ["input_layernorm.weight", [h]],
        ["post_attention_layernorm.weight", [h]],
    ]
    layer = dict(
        kind="layer",
        params=10816,
        linear_macs_per_token=10752,
        attn_heads=4,
        attn_kv_heads=2,
        head_dim=8,
        softcap=None,
        window=None,
        tied=False,
        param_shapes=shapes,
    )
    head_shapes = [["model.norm.weight", [h]]] + ([] if tied else [["lm_head.weight", [v, h]]])
    return [
        dict(
            unit=0,
            kind="prologue",
            params=v * h,
            linear_macs_per_token=0,
            attn_heads=0,
            attn_kv_heads=0,
            head_dim=0,
            softcap=None,
            window=None,
            tied=False,
            param_shapes=[["model.embed_tokens.weight", [v, h]]],
        ),
        dict(layer, unit=1),
        dict(layer, unit=2, softcap=softcap),
        dict(
            unit=3,
            kind="epilogue",
            params=h + (0 if tied else v * h),
            linear_macs_per_token=v * h,
            attn_heads=0,
            attn_kv_heads=0,
            head_dim=0,
            softcap=None,
            window=None,
            tied="model.embed_tokens.weight" if tied else False,
            param_shapes=head_shapes,
        ),
    ]


def _hf_cfg(census, compute="fp32", boundary="fp32"):
    """A GPTConfig with the census's mirrored fields, arch 'hf'. The planner
    reads only these and hf_census (I-13); GPTConfig's hf validation is
    model-core's (I-1), so the planner functions here never call it."""
    cfg = GPTConfig(
        vocab_size=_HF_V,
        n_layer=2,
        n_head=4,
        d_model=_HF_H,
        seq_len=16,
        compute_dtype=compute,
        boundary_dtype=boundary,
    )
    cfg.arch = "hf"
    cfg.hf_census = census
    return cfg


def _hf_base(n=4, tie=False):
    """A width-1 base plan (today's planner) on the census's mirrored dense
    dims: the recipe _choose prices every candidate on (mb, M, recompute)."""
    from wantrain.scheduler.planner import _plan_depth

    kw = dict(tie_embeddings=True, init_scheme="gpt2") if tie else {}
    dense = GPTConfig(vocab_size=_HF_V, n_layer=2, n_head=4, d_model=_HF_H, seq_len=16, **kw)
    nodes = [
        Node(f"n{i}", tflops=80.0, vram_bytes=24 << 30, compute_cap=(8, 9), device_class=ONE_CLASS)
        for i in range(n)
    ]
    # a toy block at 80 TFLOP/s needs ~138 Gbps for the 2x margin: on a slower
    # link the precision ladder admits nothing and there is no recipe to price
    fleet = _fleet(nodes, mbps=200_000.0)
    base = _plan_depth(dense, fleet, steps=10, device="cuda")
    assert base["spec"] is not None, base["rationale"]
    return fleet, base


def test_m4_plan_hf_fp32_cut_bytes():
    """Every cut of an hf stage is HF's residual dtype, 4 B per element,
    whatever the compute or a declared boundary (fp32 parameters, autocast);
    a dense recipe keeps its declared cut (the control)."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: an hf cut were priced at the compute or the declared
    # boundary dtype (2 B per element under bf16).
    import types

    from wantrain.scheduler import autoconfig
    from wantrain.scheduler.zero_bubble import stage_memory_model

    for compute, declared in (("bf16", "fp32"), ("bf16", "bf16"), ("fp32", "fp32")):
        cfg = _hf_cfg(_hf_census(), compute, declared)
        ns = types.SimpleNamespace(model=cfg, mb_size=2, recompute=True)
        assert stage_memory_model(ns, 2, (1, 3))[0] == 2 * 16 * 32 * 4 == 4096
        assert autoconfig.hf_boundary_dtype(cfg) == "fp32"
    dense = GPTConfig(d_model=32, seq_len=16, compute_dtype="bf16", boundary_dtype="bf16")
    ns = types.SimpleNamespace(model=dense, mb_size=2, recompute=True)
    assert stage_memory_model(ns, 2)[0] == 2 * 16 * 32 * 2 == 2048


def test_m4_plan_hf_bf16_boundary_refused():
    """A declared boundary narrower than HF's residual dtype is refused,
    naming it: by the precision admission and before any planning; the
    native rung passes and other archs are untouched."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: a bf16 or fp8 boundary were accepted for an hf model.
    from wantrain.scheduler import autoconfig

    cfg = _hf_cfg(_hf_census(), "bf16")
    for bad in ("bf16", "fp8_e4m3"):
        with pytest.raises(ValueError, match=f"boundary_dtype='{bad}' declared for arch 'hf'"):
            autoconfig.plan(cfg, 1000, 2, 1000.0, 1.0, 1e12, boundary_override=bad)
    ok = autoconfig.plan(cfg, 1000, 2, 1000.0, 1.0, 1e12, boundary_override="fp32")
    assert ok["chosen"]["boundary"] == "fp32"
    autoconfig.hf_boundary_check(make_config("toy"), "bf16")  # dense: no refusal
    nodes = [
        Node(f"n{i}", tflops=80.0, vram_bytes=24 << 30, device_class=ONE_CLASS) for i in range(2)
    ]
    from wantrain.scheduler.planner import Overrides

    with pytest.raises(ValueError, match="boundary_dtype='bf16'"):
        plan_job(cfg, _fleet(nodes), overrides=Overrides(boundary_dtype="bf16"))
    with pytest.raises(ValueError, match="boundary_dtype='bf16'"):
        plan_job(_hf_cfg(_hf_census(), "bf16", "bf16"), _fleet(nodes))


def test_m4_plan_hf_ladder_one_rung():
    """SPEC-scheduler §2.3: an hf config's precision admission prices exactly
    one rung, its residual dtype; a dense config prices all five."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: bf16 or an fp8 rung is priced for an hf model.
    from wantrain.scheduler import autoconfig

    hf = autoconfig.evaluate(_hf_cfg(_hf_census(), "bf16"), 1000, 2, 1, 1, 1000.0, 1.0, 1e12)
    assert [r["boundary"] for r in hf] == ["fp32"]
    assert hf[0]["lossless"] is True
    dense = autoconfig.evaluate(make_config("toy"), 1000, 2, 1, 1, 1000.0, 1.0, 1e12)
    assert [r["boundary"] for r in dense] == ["fp32", "bf16", "fp16", "fp8_e5m2", "fp8_e4m3"]


def test_m4_plan_hf_no_census_refused():
    """An hf config without the builder's census is refused, naming
    hf_census, by every cost function and by the planner; a census of the
    wrong length is refused naming the count."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: an hf model were costed from its mirrored dense fields.
    import types

    from wantrain.scheduler import autoconfig
    from wantrain.scheduler.partition import layer_costs
    from wantrain.scheduler.planner import stage_param_split
    from wantrain.scheduler.zero_bubble import analytic_costs

    cfg = _hf_cfg(None)
    ns = types.SimpleNamespace(model=cfg, recompute=True, partition=[(0, 4)])
    for call in (
        lambda: layer_costs(cfg),
        lambda: analytic_costs(ns),
        lambda: stage_param_split(cfg, 0, 2),
        lambda: plan_job(
            cfg, _fleet([Node(f"n{i}", 80.0, 24 << 30, device_class=ONE_CLASS) for i in range(2)])
        ),
    ):
        with pytest.raises(ValueError, match="hf_census"):
            call()
    with pytest.raises(ValueError, match="hf_census has 3 units, but n_layer \\+ 2 = 4"):
        autoconfig.hf_census(_hf_cfg(_hf_census()[:3]))


def test_m4_plan_hf_costs_from_census():
    """layer_costs and analytic_costs of an hf model come from the census:
    6·macs + 12·H·d·S per unit (the prologue a 3h lookup), F/B/W at
    2·macs + 4/8·H·d·S and 2·macs; stage parameters are the census's."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: an hf model were costed as the builtin block of its
    # mirrored fields (h(72h + 12s) per layer).
    import types

    from wantrain.scheduler.partition import layer_costs
    from wantrain.scheduler.planner import stage_param_split
    from wantrain.scheduler.zero_bubble import analytic_costs

    cfg = _hf_cfg(_hf_census())
    att = 4 * 8 * 16  # H·d·S
    assert layer_costs(cfg) == [
        3.0 * 32,
        6 * 10752 + 12 * att,
        6 * 10752 + 12 * att,
        6 * 96 * 32,
    ]
    ns = types.SimpleNamespace(model=cfg, recompute=False, partition=[(0, 2), (2, 4)])
    f, b, w = analytic_costs(ns)
    assert f == [32 + 2 * 10752 + 4 * att, 2 * 10752 + 4 * att + 2 * 3072]
    assert b == [32 + 2 * 10752 + 8 * att, 2 * 10752 + 8 * att + 2 * 3072]
    assert w == [2 * 10752, 2 * 10752 + 2 * 3072]
    assert stage_param_split(cfg, 0, 2) == (3072 + 10816, 0)
    assert stage_param_split(cfg, 2, 4) == (10816 + 32 + 3072, 0)


def test_m4_plan_hf_eager_term():
    """Eager attention holds b·H·S²·s bytes per layer: s 4 in fp32, 6 under
    autocast, plus 4 (fp32) or 2 (autocast) with a softcap; a non-recomputed
    stage's graph carries it."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: the eager term were left out, charged at the flash
    # kernel's 0, or with one s for every recipe.
    import types

    from wantrain.scheduler import autoconfig
    from wantrain.scheduler.zero_bubble import stage_memory_model

    census = _hf_census(softcap=50.0)
    per = 2 * 4 * 16 * 16  # b·H·S² at mb 2
    fp32 = autoconfig.hf_unit_costs(census, 16, 2, "fp32")
    bf16 = autoconfig.hf_unit_costs(census, 16, 2, "bf16")
    assert [u["eager_bytes"] for u in fp32] == [0, per * 4, per * 8, 0]
    assert [u["eager_bytes"] for u in bf16] == [0, per * 6, per * 8, 0]
    ns = types.SimpleNamespace(model=_hf_cfg(census, "bf16"), mb_size=2, recompute=False)
    graph = autoconfig.activation_graph_bytes(2, 16, 32, 2)
    assert stage_memory_model(ns, 2, (1, 3))[0] == graph + per * 6 + per * 8


def test_m4_plan_hf_pending_w_from_shapes():
    """An hf layer's pending W is X and dY of each 2-D weight (560 values per
    token here, against the builtin block's 16h = 512)."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: hf stages were charged the builtin block's 16h.
    import types

    from wantrain.scheduler.zero_bubble import stage_memory_model

    ns = types.SimpleNamespace(model=_hf_cfg(_hf_census()), mb_size=2, recompute=True)
    f, b, w = stage_memory_model(ns, 2, (1, 3))
    assert b == 2 * (2 * 16 * 560 * 4)
    assert f + b + w == 0


def test_m4_plan_hf_tied_delta_t_round5():
    """ΔT_tied's five terms against round4.md §1's table (the formula of
    record, round5.md §8): Gemma2-2B (V 256,000, h 2304, 2.0e9 non-embedding,
    T_mb 4096) at S 4, M 8, L 20 ms, 10 Gbps, 400 TFLOP/s, 2 TB/s:
    165/51/189/9/0 = 414 ms worst case, 32/0/155/9/0 = 197 ms for text rows;
    and the tied bytes per hop."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: a term were dropped or mispriced (the round trip counted
    # once, the warmup over every microbatch, the burst never offset by the
    # idle window, the scan without M).
    from wantrain.scheduler import autoconfig

    V, h, T, S, M, L, Bw = 256000, 2304, 4096, 4, 8, 0.020, 10e9 / 8
    t_s = 6 * (2.0e9 + V * h) * T / S / 4e14
    t_F, t_B = t_s / 3, 2 * t_s / 3
    a = T * h * 4 / Bw  # 30.2 ms
    adamw = V * h * 28 / 2e12
    for frac_T, frac_R, want in ((1.0, 1.0, 414), (0.45, 0.15, 197)):
        rT, rR = int(frac_T * T), int(frac_R * T)
        got = autoconfig.delta_t_tied(
            t_s,
            t_F,
            t_B,
            [rT] * M,
            [rR] * M,
            S - 1,
            L,
            Bw,
            d_model=h,
            vocab_size=V,
            tokens_mb=T,
            mem_bw_Bps=2e12,
        )
        e, r = rT * h * 4 / Bw, rR * h * 4 / Bw
        hand = dict(
            throughput=M * max(0.0, a + e - max(t_s, a)),
            warmup=(S - 1) * max(0.0, r - t_F),
            round_trip=2 * (S - 1) * L + 2 * e + adamw,
            scan=M * V * h * 4 / 2e12,
            back_link=max(0.0, (M - 1) * r - (2 * (S - 1) * L + S * t_F + (S - 1) * a + t_B)),
        )
        for k, v in hand.items():
            assert got[k] == pytest.approx(v, rel=1e-12, abs=1e-15), k
        assert got["total"] == pytest.approx(sum(hand.values()), rel=1e-12)
        assert round(got["total"] * 1e3) == want
    worst = autoconfig.delta_t_tied(
        t_s,
        t_F,
        t_B,
        [T] * M,
        [T] * M,
        3,
        L,
        Bw,
        d_model=h,
        vocab_size=V,
        tokens_mb=T,
        mem_bw_Bps=2e12,
    )
    assert [round(worst[k] * 1e3) for k in ("throughput", "warmup", "round_trip", "scan")] == [
        165,
        51,
        189,
        9,
    ]
    # the burst row of the table: S 2, M 8, L 0 -> back link 75 ms of 157
    t2 = 6 * (2.0e9 + V * h) * T / 2 / 4e14
    burst = autoconfig.delta_t_tied(
        t2,
        t2 / 3,
        2 * t2 / 3,
        [T] * M,
        [T] * M,
        1,
        0.0,
        Bw,
        d_model=h,
        vocab_size=V,
        tokens_mb=T,
        mem_bw_Bps=2e12,
    )
    assert (round(burst["back_link"] * 1e3), round(burst["total"] * 1e3)) == (75, 157)
    assert autoconfig.tied_hop_bytes(4096, 1000, 2304) == (4096 * 2304 * 4, 1000 * 2304 * 4)


def test_m4_plan_hf_tied_memory():
    """A split tied table: the Head stage holds the table (with its state:
    trainable parameters) and n_pend·min(V, n_pend·T_mb)·h·4 B of held rows;
    stage 0 only its working copy and one transient gradient, 2·V·h·4 (no
    optimizer state: SPEC-scheduler §2); one stage holding both counts the
    table once and nothing more."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: the once-counted table stayed (the Head charged nothing),
    # stage 0 kept the table's optimizer state, or a term of stage 0 or the
    # Head were dropped.
    from wantrain.scheduler.planner import _hf_extras, _stage_units, stage_param_split

    cfg = _hf_cfg(_hf_census(tied=True))
    V, h = _HF_V, _HF_H
    assert stage_param_split(cfg, 3, 4) == (32 + V * h, 0)
    assert stage_param_split(cfg, 0, 1) == (0, 0)
    assert stage_param_split(cfg, 0, 2) == (10816, 0)
    assert stage_param_split(cfg, 2, 4) == (10816 + 32 + V * h, 0)
    assert stage_param_split(cfg, 0, 4) == (V * h + 2 * 10816 + 32, 0)
    assert [u.params for u in _stage_units(cfg, 0, 2, 4, 4)[0]] == [10816]
    assert [u.params for u in _stage_units(cfg, 2, 4, 4, 4)[0]] == [10816, 32 + V * h]
    T = 1 * 16
    eager = 1 * 4 * 16 * 16 * 4
    assert _hf_extras(cfg, (3, 4), 1, 3) == (0, 3 * min(V, 3 * T) * h * 4)
    assert _hf_extras(cfg, (0, 2), 1, 3) == (eager, 2 * V * h * 4)
    assert _hf_extras(cfg, (0, 4), 1, 3) == (eager, 0.0)
    untied = _hf_cfg(_hf_census())
    assert stage_param_split(untied, 3, 4) == (32 + V * h, 0)
    assert _hf_extras(untied, (3, 4), 1, 3) == (0, 0.0)


def test_m4_plan_hf_width_refusals():
    """BA-447: under a split tied table every stage from stage 0 to the Head
    at width > 1 is refused, naming the stage and the table: stage 0, the
    Head and a middle stage each alone; a width-1 chain passes."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: a wide stage on the relay chain were admitted, or the
    # refusal named the wrong stage.
    from wantrain.scheduler.planner import tied_width_refusals

    t = "model.embed_tokens.weight"
    for widths, stage in (([2, 1, 1], 0), ([1, 1, 2], 2), ([1, 2, 1], 1)):
        got = tied_width_refusals(widths, t)
        assert [(r.kind, r.stage) for r in got] == [("tied", stage)]
        assert t in got[0].where and f"stage {stage} width 2" in got[0].where
    assert tied_width_refusals([1, 1, 1], t) == []
    assert [r.stage for r in tied_width_refusals([2, 2, 2], t)] == [0, 1, 2]


def test_m4_plan_hf_ba446_w1_dense_tied_split_refused():
    """BA-446: a non-hf tie_embeddings with the Embed and the Head on
    different stages is refused naming tok.weight, at width 1 too (the check
    runs ahead of the width-1 return): every candidate is marked not admitted
    with that reason and the plan carries no spec; never raised. Two nodes
    (no candidate list) are refused the same way."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: the width-1 return still skips the tied check.
    cfg = make_config("gpt124m")
    cfg.tie_embeddings, cfg.init_scheme = True, "gpt2"
    nodes = [
        Node(f"n{i}", tflops=80.0, vram_bytes=24 << 30, compute_cap=(8, 9), device_class=ONE_CLASS)
        for i in range(4)
    ]
    out = plan_job(cfg, _fleet(nodes), steps=10, device="cuda")
    assert out["spec"] is None and "draft" not in out
    w1 = next(c for c in out["candidates"] if c["w"] == 1)
    assert w1["S"] == 4 and not w1["admitted"]
    assert [r["kind"] for r in w1["reasons"]] == ["tied"]
    assert "tok.weight" in w1["reasons"][0]["text"]
    for c in out["candidates"]:
        assert not c["admitted"] and any(r["kind"] == "tied" for r in c["reasons"])
    assert "tok.weight" in out["rationale"]["decision"]
    cfg2 = make_config("gpt124m")
    cfg2.tie_embeddings, cfg2.init_scheme = True, "gpt2"
    two = plan_job(cfg2, _fleet(nodes[:2]), steps=10, device="cuda")
    assert two["spec"] is None and "tok.weight" in two["rationale"]["decision"]
    untied = plan_job(make_config("gpt124m"), _fleet(nodes[:2]), steps=10, device="cuda")
    assert untied["spec"] is not None
    assert untied["rationale"]["boundary_dtype"] == untied["spec"].model.boundary_dtype


def test_m4_plan_hf_wide_unpriced_hf_plans_width_one():
    """BA-466: every wide candidate of an hf config is refused 'unpriced'
    (no exchange-unit model of HF units, D-10), so its plan is width 1."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: a wide hf candidate is priced with the builtin block's units.
    from wantrain.scheduler.planner import _choose

    fleet, base = _hf_base()
    out = _choose(_hf_cfg(_hf_census()), fleet, base, [0, 1, 2, 3], "cuda", True, None)
    wide = [c for c in out["candidates"] if c["w"] > 1]
    assert wide
    for c in wide:
        assert not c["admitted"] and any(r["kind"] == "unpriced" for r in c["reasons"])
        assert any("arch='hf'" in r["text"] for r in c["reasons"])
    assert out["rationale"]["chosen"] == "S 4 w 1"


def test_m4_plan_hf_tied_refused_before_s1_and_step_carries_delta_t(monkeypatch):
    """A split hf tied table: every sharded candidate is refused 'tied'
    (BA-447) before S1 is ever evaluated, so S1 never carries tied bytes; the
    width-1 plan is priced with ΔT_tied's five terms in its step and the tied
    hops and bytes in its detail."""
    # covers: SCHEDULER-I22
    # WOULD-FAIL-IF: a sharded tied plan reached S1, or the width-1 step left
    # ΔT_tied out.
    from wantrain.scheduler import admission
    from wantrain.scheduler.planner import _choose

    def no_s1(*a, **k):
        raise AssertionError("S1 evaluated for a tied hf plan")

    monkeypatch.setattr(admission, "s1", no_s1)
    fleet, base = _hf_base()
    out = _choose(_hf_cfg(_hf_census(tied=True)), fleet, base, [0, 1, 2, 3], "cuda", True, None)
    for c in out["candidates"]:
        if c["w"] > 1:
            tied = [r for r in c["reasons"] if r["kind"] == "tied"]
            assert [r["stage"] for r in tied] == list(range(c["S"]))
            assert all("model.embed_tokens.weight" in r["text"] for r in tied)
    w1 = next(c for c in out["candidates"] if c["w"] == 1)
    assert w1["admitted"] and out["rationale"]["chosen"] == w1["name"]
    detail = w1["detail"]["tied"]
    d = detail["delta_t_tied"]
    five = ("throughput", "warmup", "round_trip", "scan", "back_link")
    assert d["total"] == pytest.approx(sum(d[k] for k in five)) and d["total"] > 0
    assert w1["step_s"] >= d["total"]
    assert detail["hops"] == [[0, 1], [1, 2], [2, 3]]
    tokens = base["spec"].mb_size * 16
    assert detail["fwd_bytes_per_mb"] == min(_HF_V, tokens) * _HF_H * 4
