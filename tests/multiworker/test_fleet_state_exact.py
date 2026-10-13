"""A coordinator-run fleet, the way production runs: heterogeneous workers in
two emulated regions, placed and scheduled by the coordinator, healed when one
dies. The final training state (every weight, AdamW moment and step, the RNG
state and every loss) equals the single-device reference byte for byte.

WOULD-FAIL-IF: heterogeneous placement, the capacity-weighted layer cut,
cross-region routing, the profiled schedule or a heal from checkpoints changed
one bit of the final weights, the optimizer state, the RNG state or a loss, or
a healed slot resumed without part of its state.
"""

from helpers.cluster import LocalCluster
from helpers.exact import fleet_manifest, reference_manifest_for

from wantrain.coordinator.coordinator import Job
from wantrain.model.gpt import make_config
from wantrain.runtime.spec import JobSpec
from wantrain.validation.state import diff_manifests

WAN = {"intra": {"mbps": 400, "rtt_ms": 2}, "inter": {"mbps": 60, "rtt_ms": 60}}


def _recipe(**kw):
    spec = JobSpec(
        model=make_config("toy"),
        **{"steps": 6, "microbatches": 8, "recompute": True, "wire_codec": "planes", **kw},
    )
    spec.model.n_layer = 12
    return spec


def _assert_exact(c, res, recipe, tmp_path):
    assert res.ok, res.reason
    state = fleet_manifest(c, res, tmp_path / "collected")
    reference = reference_manifest_for(recipe, tmp_path / "ref")
    assert diff_manifests(state, reference) == []
    assert state == reference


def test_heterogeneous_two_region_fleet_is_byte_exact(tmp_path, port_base):
    c = LocalCluster(
        tmp_path, port_base, probe=True, probe_bulk_bytes=1 << 20, measure_bulk_bytes=1 << 20
    )
    try:
        fleet = {
            "eu-fast": ("eu", {"tflops": 4.0, "vram_bytes": 4 << 30, "gpu_name": "sim-A"}),
            "eu-slow": ("eu", {"tflops": 1.0, "vram_bytes": 8 << 30, "gpu_name": "sim-C"}),
            "us-mid": ("us", {"tflops": 2.0, "vram_bytes": 16 << 30, "gpu_name": "sim-B"}),
        }
        for name, (region, declare) in fleet.items():
            c.add_worker(name, region=region, wan_profile=WAN, declare=declare)
        c.wait_registered(len(fleet))
        c.coord.cfg.selection.same_region = False
        recipe = _recipe(stages=3)
        res = c.run(Job(spec=recipe))
        _assert_exact(c, res, recipe, tmp_path)
        assert "heterogeneous fleet" in res.rationale["selection"], res.rationale
    finally:
        c.close()


def test_a_healed_fleet_is_byte_exact(tmp_path, port_base):
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(3):  # 2 slots + 1 spare
            c.add_worker(f"w{i}")
        c.wait_registered(3)
        recipe = _recipe(stages=2, steps=12, ckpt_every=2)
        recipe.links = [dict(latency_ms=15.0, bw_mbps=0.0, jitter_ms=0.0, jitter_seed=0)] * 2
        c.run_async(Job(spec=recipe))
        c.wait_step(5)
        victim = c.name_of((1, 0))
        c.kill_worker(victim)
        res = c.join()
        assert len(res.heals) == 1 and res.heals[0]["from"] == victim, res.heals
        _assert_exact(c, res, recipe, tmp_path)
    finally:
        c.close()
