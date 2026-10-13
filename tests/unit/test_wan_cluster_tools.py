"""benchmarks/wan/cluster.py's fleet checks: the rate a hop must sustain, the verdict on a sustained two-way run,
the nodes to replace, and a stage that has slowed against its own baseline.

WOULD-FAIL-IF: need_mbps priced the boundary without the codec ratio or the margin; a run whose median held but
one window dipped passed as ok; a NIC-capped node (failing toward every peer) was not named first; the TCP ramp
window counted; a stage slow for one step only, or slow without a baseline, was reported.
"""

import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "wan", "cluster.py")
_spec = importlib.util.spec_from_file_location("wan_cluster", _PATH)
cluster = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cluster)


def test_need_mbps_is_one_coded_boundary_per_microbatch_with_margin():
    # d 6144, seq 1024, bf16, codec 0.69: 8.68 MB per microbatch; in 250 ms that is 278 Mbps, x1.3 = 361
    got = cluster.need_mbps(6144, 1024, mb_ms=250.0)
    assert got == pytest.approx(1024 * 6144 * 2 * 0.69 * 8 / 0.25 / 1e6 * 1.3)
    assert 355 < got < 365
    assert cluster.need_mbps(6144, 1024, mb_ms=500.0) == pytest.approx(got / 2)


def test_window_stats_drop_the_ramp_window():
    st = cluster.window_stats([5.0, 900.0, 300.0, 880.0, 910.0])
    assert st["min"] == 300.0 and st["median"] == 900.0


def test_verdict_separates_a_dip_from_a_cap():
    need = 361.0
    assert cluster.verdict({"min": 400, "p10": 450, "median": 900}, need) == "ok"
    assert cluster.verdict({"min": 250, "p10": 300, "median": 900}, need) == "weak"
    assert cluster.verdict({"min": 200, "p10": 250, "median": 330}, need) == "fail"


def test_suspects_names_the_node_that_fails_toward_every_peer_first():
    # Nuremberg-like: de's port caps both directions toward everyone; one weak dip elsewhere
    results = {
        ("de", "cz1"): "fail",
        ("cz1", "de"): "fail",
        ("de", "mt"): "fail",
        ("mt", "de"): "weak",
        ("mt", "cz1"): "ok",
        ("cz1", "mt"): "weak",
    }
    assert cluster.suspects(results) == [
        "de",
        "cz1",
        "mt",
    ]  # by failing directions: de 3, cz1 2, mt 1
    assert cluster.suspects({("a", "b"): "ok", ("b", "a"): "ok"}) == []
    assert cluster.suspects({("a", "b"): "weak", ("b", "a"): "ok"}) == []


def test_slow_stages_need_a_baseline_and_two_slow_steps_in_a_row():
    base = {"cz1": 250.0, "cz2": 250.0}
    hist = {"cz1": [250, 260, 400, 420], "cz2": [250, 400, 255, 251]}
    assert cluster.slow_stages(hist, base, slow=1.3) == [("cz1", 1.68)]
    assert cluster.slow_stages({"x": [900, 900]}, {}, slow=1.3) == []


def test_stage_ms_sums_the_median_ops():
    tel = {
        "ops": {
            "F": {"med_ms": 66.0},
            "B": {"med_ms": 141.0},
            "W": {"med_ms": 7.0},
            "OPT": {"med_ms": 2.0},
        }
    }
    assert cluster.stage_ms(tel) == 214.0


def test_microbatches_follow_the_slowest_directions_headroom():
    assert cluster.recommend_microbatches([2.4, 3.0, 2.0]) == 128
    assert cluster.recommend_microbatches([2.4, 1.1]) == 64
    assert cluster.recommend_microbatches([0.9, 5.0]) == 32
    assert cluster.recommend_microbatches([]) == 32


def test_auto_microbatches_need_a_clean_sustain_report():
    job = {"plan": {"microbatches": "auto", "d_model": 8}, "stages": 2}
    assert (
        cluster.resolve_job(job, {"replace": [], "microbatches": 128})["plan"]["microbatches"]
        == 128
    )
    with pytest.raises(SystemExit):
        cluster.resolve_job(job, None)
    with pytest.raises(SystemExit):
        cluster.resolve_job(job, {"replace": ["de"], "microbatches": 32})
    fixed = {"plan": {"microbatches": 64}}
    assert cluster.resolve_job(fixed, None) is fixed


def test_the_compute_per_microbatch_follows_the_model_and_the_gpu():
    plan = {"d_model": 1536, "n_layer": 24, "seq_len": 1024, "recompute": True}
    ms_4090 = cluster.estimate_mb_ms(plan, 2, "NVIDIA GeForce RTX 4090")
    assert (
        75 < ms_4090 < 95
    )  # 0.42B per stage, 8 x 0.42e9 x 1024 FLOPs at 40 TFLOPs; 85 ms measured 2026-10-13
    assert cluster.estimate_mb_ms(plan, 2, "NVIDIA H100 80GB HBM3") < ms_4090
    assert cluster.estimate_mb_ms(plan, 4, "NVIDIA GeForce RTX 4090") < ms_4090
    big = {"d_model": 6144, "n_layer": 54, "seq_len": 1024, "recompute": True}
    assert (
        200 < cluster.estimate_mb_ms(big, 4, "NVIDIA H100 80GB HBM3") < 320
    )  # ~250 measured on 2026-10-12
