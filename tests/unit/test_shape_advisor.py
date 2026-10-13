"""The shape advisor (SCHEDULER-I23) prices the WAN bubble the way the 2026-10-12 H100 run measured it, and picks
standard shapes (width/depth 20-200) that fit memory and hide the link.

WOULD-FAIL-IF: the priced bubble of the measured run were far from the measured ~75%, advice ignored memory or the
width/depth range, or a faster link did not buy a larger model.
"""

from wantrain.scheduler.shape import advise, needed_mbps, price

GIB = 2**30


def test_the_priced_bubble_matches_the_measured_run():
    # covers: SCHEDULER-I23
    # 2.8B dense, 2 stages of 16 layers, d 2560, 16 microbatches, 45 Mbps: 115 ms of compute (99 effective TFLOPs)
    # against 485 ms on the wire; the stages waited 72-79% of each step.
    s = price(2560, 16, 2, tflops=99, link_mbps=45, microbatches=16)
    assert 0.70 <= s.bubble <= 0.82
    assert abs(s.compute_s - 0.115) < 0.01 and abs(s.transfer_s - 0.485) < 0.02


def test_advice_fits_memory_keeps_standard_shapes_and_grows_with_the_link():
    # covers: SCHEDULER-I23
    slow = advise(2, 79 * GIB, tflops=99, link_mbps=45, bubble_target=0.2)
    fast = advise(2, 79 * GIB, tflops=99, link_mbps=500, bubble_target=0.2)
    for r in (slow, fast):
        assert r["met_target"] and r["best"].bubble <= 0.2
        assert all(
            s.stage_bytes <= 79 * GIB and 20 <= s.d_model / s.n_layer <= 200 for s in r["fits"]
        )
    # a slow hop wants depth at a narrow width; a fast one buys width and more parameters
    assert slow["best"].d_model < fast["best"].d_model
    assert fast["best"].params > slow["best"].params
    assert needed_mbps(2560, 64, tflops=99) < needed_mbps(2560, 16, tflops=99)
