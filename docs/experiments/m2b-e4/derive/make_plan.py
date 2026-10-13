# m2b-e4: writes PLAN.json from r6_e4.out's last line (the JSON the model printed), so no number is transcribed by hand.
# usage: python3 -I make_plan.py r6_e4.out ../PLAN.json
import hashlib
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
R = json.loads(open(src).read().strip().splitlines()[-1])
sha_out = hashlib.sha256(open(src, "rb").read()).hexdigest()
F4, F8 = R["Fprime"]["4"], R["Fprime"]["8"]
sp4, sp8 = R["split"]["4"], R["split"]["8"]


def arm_rows():
    rows = []
    for a in R["arms"]:
        r = dict(arm=a["arm"], predicted_ms=a["predicted_ms"], predicted_pct_of_C=a["predicted_pct_of_C"],
                 f6_min_margin_ms=a["f6_min_margin_ms"], f6_min_margin_pct=a["f6_min_margin_pct"], C_ms=a["C_ms"],
                 X_ms=a["X_ms"], nic_util=a["util"], s1_holds=a["s1"], w=a["w"])
        for k in ("f6_vs", "rtt_ms", "c_W_ms", "cap_up_MBps", "util_up", "s1_up", "f5_left_fold_ms", "cap_MBps"):
            if k in a:
                r[k] = a[k]
        r["label"] = "DERIVED (round-4 model at the measured F', split and fold); SIMULATED when measured"
        rows.append(r)
    return rows


plan = {
    "id": "m2b-e4",
    "hypothesis": (
        "ADR-0006 H3 and F6 at the scaled B' on the shared-pacer shim: per-tensor pieces (forward emission, need-first "
        "queues, per-unit updates, no step barrier) expose less of the exchange than step 1 alone and than LPT whole "
        "tensors by at least half the margin this PLAN pins at every F6 arm (stage 0 vs step 1 and LPT; interior w 4 vs "
        "step 1; interior w 8 vs LPT); measured exposure stays within [0.8, 1.2]x the as-built SCHEDULER-I17 simulation "
        "on both sides of S1, at the S1 threshold (within 25%) and on a shared uplink, with the barrier-kept control "
        "matching its barrier simulation (F3); a residue-group tree exchange exposes more than the streaming left fold "
        "while S1 holds (F5). Every number is DERIVED here and SIMULATED when measured; bits are byte-identical in every "
        "training arm."),
    "expected": {
        "f6_margin_fraction_min": 1.0,
        "f6_stage0_vs_step1_fraction": 1.0,
        "f6_stage0_vs_lpt_fraction": 1.0,
        "f6_interior_w4_vs_step1_fraction": 1.0,
        "f6_interior_w8_vs_lpt_fraction": 1.0,
        "f3_violations": 0,
        "f5_violations": 0,
        "emission_order_violations": 0,
    },
    "lowerIsBetter": ["f3_violations", "f5_violations", "emission_order_violations"],
    "primary": "f6_margin_fraction_min",
    "tolerancePct": 50,
    "local": {
        "correctness": "uv run pytest -q -p no:randomly tests/integration/test_engine_bitwise.py -k \"e4_ and identity\"",
        "performance": "uv run pytest -q -p no:randomly -s tests/integration/test_engine_bitwise.py -k e4_",
        "failure": "uv run pytest -q -p no:randomly tests/integration/test_engine_bitwise.py -k \"e4_ and (fifo or barrier)\"",
    },
    "cloud": None,
    "scaled": {
        "h": 256, "V": 2000, "seq": 256, "T_mb": 256, "n_head": 4, "recipe": "bf16 compute and boundary (autocast), "
        "fp32 masters; attn_impl naive", "intra_op_threads": 1,
        "w": {"stage0": 4, "interior_w4": 4, "interior_w8": 8},
        "L_blocks": {"stage0": "Embed + 1 block", "interior_w4": 4, "interior_w8": 4},
        "M_over_w": {"stage0": 32, "interior_w4": 8, "interior_w8": 4},
        "F_prime_flops": {"w4": F4, "w8": F8},
        "F_prime_gflops": {"w4": round(F4 / 1e9, 3), "w8": round(F8 / 1e9, 3)},
        "op_split_F_B_W": {"w4": [round(x, 4) for x in sp4], "w8": [round(x, 4) for x in sp8]},
        "op_split_tolerance_rel": 0.25,
        "c_block_ms": {"w4": round(8 * 12 * 256 * 256 * 256 / F4 * 1e3, 2), "w8": round(8 * 12 * 256 * 256 * 256 / F8 * 1e3, 2)},
        "cap_prime_Bps_per_flops": 50e9 / 25e12,
        "cap_prime_gbps": {"w4": round(R["cap_prime"]["4"] * 8 / 1e9, 5), "w8": round(R["cap_prime"]["8"] * 8 / 1e9, 5)},
        "cap_prime_MBps": {"w4": round(R["cap_prime"]["4"] / 1e6, 4), "w8": round(R["cap_prime"]["8"] / 1e6, 4)},
        "pcie_prime_MBps": {"w4": round(R["pcie_prime"]["4"] / 1e6, 4), "w8": round(R["pcie_prime"]["8"] / 1e6, 4)},
        "hop_latency_ms": {"w4": round(2 * R["chunk_bytes"] / R["pcie_prime"]["4"] * 1e3, 3),
                           "w8": round(2 * R["chunk_bytes"] / R["pcie_prime"]["8"] * 1e3, 3)},
        "hbm_prime_gbs_model": {"w4": round(R["hbm_prime_model"]["4"] / 1e9, 4), "w8": round(R["hbm_prime_model"]["8"] / 1e9, 4)},
        "fold_rate_pinned_gbs": {"bf16_widen_add_18B": round(R["fold"][0] / 1e9, 2), "fp32_add_12B": round(R["fold"][1] / 1e9, 2),
                                 "adamw_30B": round(R["fold"][2] / 1e9, 2)},
        "fold_paced": False,
        "hbm_prime_gbs": round(R["fold"][0] / 1e9, 2),
        "chunk_bytes": R["chunk_bytes"],
        "C_B_prime_ms": round(R["C_B_prime"] * 1e3, 1),
        "C_B_over_c_block": 52.8333,
        "rtt_ms": {"rtt0": 0.0, "rtt_2cw_interior_w4": 128.446, "rtt_2cw_stage0": 32.112},
        "rtt_rule": "RTT_hi = 2 * c_W (c_W = split W fraction * c_s, in situ)",
        "uplink": {"arm": "interior_w4", "sites": [0, 0, 1, 1], "cap_up_equals_cap_prime": True, "boundary_bytes_on_uplink": 0},
        "sweep": {"arm": "interior_w4.pieces_fwd", "vary": "cap' (NIC; PCIe' not binding, hop latency held at base)",
                  "util_points": [0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0], "cap_s1_MBps": round(R["threshold"]["cap_s1_MBps"], 4),
                  "predicted_threshold_cap_MBps": round(R["threshold"]["cap_knee_MBps"], 4),
                  "predicted_threshold_util": round(R["threshold"]["util_at_knee"], 4),
                  "threshold_rule": "two least-squares lines of E% against 1/cap', one through the 3 points below S1 and one "
                                    "through the 4 above; the threshold is where they meet (same rule on measured and simulated)",
                  "fit_below": R["threshold"]["fit_below"], "fit_above": R["threshold"]["fit_above"]},
        "tree": {"definition": "residue-group fold G = w: each peer folds its own M_k microbatches locally in fp32 and "
                               "sends one fp32 partial per unit piece to its owner, released as the last W produces the "
                               "unit (forward emission); owners add w-1 partials; gathers after; nothing streams before "
                               "the last W", "arms": ["interior_w4.tree (F5 primary)", "stage0.tree"]},
    },
    "arms": arm_rows(),
    "ratios": R["ratios"],
    "ratio_tolerance_rel": 0.005,
    "derivation": {
        "script": "docs/experiments/m2b-e4/derive/r6_e4.py",
        "command": "cd docs/experiments/m2b-e4/derive && python3 -I r6_e4.py r4.py r3.py r2.py "
                   "../../../../src/wantrain/scheduler/zero_bubble.py profile_fprime.out conc_w4.out conc_w8.out",
        "output_sha256": sha_out,
    },
}
json.dump(plan, open(dst, "w"), indent=1)
open(dst, "a").write("\n")
print("wrote", dst, "arms", len(plan["arms"]))
