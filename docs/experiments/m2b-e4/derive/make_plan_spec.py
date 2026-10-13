# Builds the SC-4 amended m2b-e4 PLAN.json from the granted PLAN.json and e4_spec.json (no hand-copied numbers), and
# writes the PLAN.md tables (old vs new) as a Markdown fragment.
# usage: python3 -I -B make_plan_spec.py granted_PLAN.json e4_spec.json e4_spec.py e4_spec.out i17_oracle.py \
#            ../PLAN.json e4_tables.md gen_plan_md.py
import copy
import hashlib
import json
import sys

BASE, SPECJ, SCRIPT, OUTTXT, ORACLE, OUT, MD, MDGEN = sys.argv[1:9]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


base = json.load(open(BASE))
assert not any("SC-4" in json.dumps(a) for a in base.get("amendments", [])), "base already carries SC-4"
spec = json.load(open(SPECJ))
new = copy.deepcopy(base)
LABEL = "DERIVED (SCHEDULER-I17 oracle e4_spec.py at the measured F', split and fold); SIMULATED when measured"

old_arms = {a["arm"]: a for a in base["arms"]}
new_arms = []
superseded = {}
for a in spec["arms"]:
    o = old_arms[a["arm"]]
    row = {k: v for k, v in o.items() if k not in ("X_ms",)}
    row.update(predicted_ms=a["predicted_ms"], predicted_pct_of_C=a["predicted_pct_of_C"], C_ms=a["C_ms"],
               window_ms=a["window_ms"], exposed_ms=a["exposed_ms"], nic_util=a["nic_util"], s1_holds=a["s1_holds"],
               f6_min_margin_ms=a["f6_min_margin_ms"], f6_min_margin_pct=a["f6_min_margin_pct"], label=LABEL)
    for k in ("f6_vs", "rtt_ms", "c_W_ms", "cap_up_MBps", "util_up", "s1_up", "f5_left_fold_ms", "cap_MBps"):
        if k in a:
            row[k] = a[k]
    new_arms.append(row)
    for k in ("predicted_ms", "predicted_pct_of_C", "f6_min_margin_ms", "f6_min_margin_pct", "f5_left_fold_ms"):
        if o.get(k) != row.get(k) and k in o:
            superseded[f"arms[{a['arm']}].{k}"] = o[k]
    superseded[f"arms[{a['arm']}].X_ms"] = o.get("X_ms")
assert [a["arm"] for a in new_arms] == [a["arm"] for a in base["arms"]], "arm set or order changed"
new["arms"] = new_arms

sw = new["scaled"]["sweep"]
thr = spec["threshold"]
for k_new, k_old in (("cap_s1_MBps", "cap_s1_MBps"), ("predicted_threshold_cap_MBps", "predicted_threshold_cap_MBps"),
                     ("predicted_threshold_util", "predicted_threshold_util"), ("fit_below", "fit_below"),
                     ("fit_above", "fit_above")):
    if sw.get(k_old) != thr[k_new]:
        superseded[f"scaled.sweep.{k_old}"] = sw.get(k_old)
    sw[k_old] = thr[k_new]
sc = new["scaled"]
sc["stage0_padding"] = ("per peer q, its i-th owned microbatch (global index i*w + q, i = 0 .. M/w - 1) occupies the slot "
                        "[i*tau, (i+1)*tau), tau = C_B'/(M/w); with cF, cB, cW = c_s times the pinned op split: F runs "
                        "[i*tau, i*tau + cF], B runs [(i+1)*tau - cW - cB, (i+1)*tau - cW], W runs [(i+1)*tau - cW, "
                        "(i+1)*tau] (the idle between F and B is the wait for the downstream backward); no F runs ahead "
                        "of its slot. So the last W ends at C_B', the period is C_B', window_s = 0, and step t+1's first "
                        "F is due at C_B'. A harness may realize it with a per-peer wait of tau - c_s between each F and "
                        "its B (one stand-in per peer, never one shared by the peers, comm 0, no exchange), as long as "
                        "the op times above result.")
sc["exposure_definition"] = ("E = how late step t+1's forward is, units in forward order (D accumulates), against the "
                             "peer's compute end, from ONE simulation of every microbatch's traffic (SCHEDULER-I17); "
                             "every E4 arm has window_s = 0, so E is the exposure beyond the slot")
sc["host_dram"] = ("a resource in the simulation at k 10 per side (I17), unconstrained here: every peer runs on one Mac "
                   "whose DRAM (~100 GB/s) carries <= 2.7 GB/s of exchange passes; the shim does not pace it")
sc["embed_unit"] = ("one fp32 unit (tok + pos, I17), released when W(mb) starts; stride and LPT own its two tensors "
                    "whole (tok and pos may sit on different peers)")
sc["fold_rule"] = ("left fold of every microbatch's contribution in canonical order, each at the measured rate "
                   "((b_g + 16) B/element for a 2-byte wire, 12 B for fp32), then AdamW at 30 B/element; the unit's "
                   "gathers leave after the last fold and the AdamW (I17)")
sc["today_mode"] = ("whole-stage flush: stride ZeRO-1 of whole tensors (params[k::w], stage 0 base 2, 12 parameters per "
                    "block, modules at +2, +4, +8, +10; tok and pos are tensors 0 and 1), fp32 wire (b_g = b_p = 4 on "
                    "every unit, so folds at 12 B/element), every unit's gradient pieces of W(mb) released at W(mb)'s "
                    "END (the Embed too, not at W start); the emission order is therefore irrelevant; queues, chunks, "
                    "resources, latencies and the left fold as I17; full step barrier: every gather released at "
                    "Tb = max over (unit, owner) of (its last fold + AdamW), and step t+1's F on every unit waits for "
                    "Tb and for that unit's gather; boundary bytes as every other arm")
new["round5_anchor"] = {
    "config": "F' 50 GFLOP/s, op split 1:2:1, HBM' = 3.35 TB/s x s_rate for fold and AdamW, B' shapes",
    "tolerance_rel": 0.05,
    "arms": {k: v["spec_pct"] for k, v in spec["round5_anchor"].items()},
    "granted_round5_md": {k: v["pinned_pct"] for k, v in spec["round5_anchor"].items()},
}
new["derivation"] = {
    "script": "docs/experiments/m2b-e4/derive/e4_spec.py",
    "oracle": "docs/experiments/m2b-e4/derive/i17_oracle.py",
    "command": ("cd docs/experiments/m2b-e4/derive && python3 -I -B e4_spec.py i17_oracle.py profile_fprime.out "
                "conc_w4.out conc_w8.out e4_spec.json > e4_spec.out && python3 -I -B make_plan_spec.py granted_PLAN.json "
                "e4_spec.json e4_spec.py e4_spec.out i17_oracle.py ../PLAN.json e4_tables.md gen_plan_md.py && python3 -I "
                "-B gen_plan_md.py granted_PLAN.md ../PLAN.json e4_tables.md ../PLAN.md granted_PLAN.json e4_spec.out"),
    "independence": "nothing from src/wantrain is imported or read (the human's rule, 2026-10-10)",
    "script_sha256": sha(SCRIPT),
    "oracle_sha256": sha(ORACLE),
    "output_sha256": sha(OUTTXT),
    "json_sha256": sha(SPECJ),
    "builder_sha256": sha(sys.argv[0]),
    "md_generator_sha256": sha(MDGEN),
    "closes": "SC-4 (HANDOUT-M2b): the I17 anchor re-pinned from the SPEC model; F6 thresholds re-derived",
    "superseded_derivation": base["derivation"],
}
new["hypothesis"] = base["hypothesis"].replace(
    "Every number is DERIVED here", "Every number is DERIVED here by the SCHEDULER-I17 oracle (SC-4)")
amend = {
    "id": "SC-4 / SPEC SCHEDULER-I17",
    "date": "2026-10-10",
    "reason": ("The granted derivation (r6_e4.py on round 4's exch4) let a hot owner's gather egress overlap its own "
               "gradient ingress (derivation.md section 2's V_max/cap term) and modelled no host DRAM, no left fold of "
               "every contribution, and no whole-stage flush. SCHEDULER-I17, the SPEC of record, releases a unit's "
               "gathers only after the owner's last fold and AdamW, puts host DRAM in the simulation, and reads E from "
               "one simulation of every microbatch. Re-derived by an independent oracle written from the SPEC text; "
               "every input the granted PLAN pins is kept. The stage-0 padding is placed per microbatch (window_s 0) "
               "and the 'today' arms get a whole-stage-flush mode (scaled.stage0_padding, scaled.today_mode)."),
    "script": "docs/experiments/m2b-e4/derive/e4_spec.py",
    "script_sha256": sha(SCRIPT),
    "oracle_sha256": sha(ORACLE),
    "output_sha256": sha(OUTTXT),
    "json_sha256": sha(SPECJ),
    "builder_sha256": sha(sys.argv[0]),
    "md_generator_sha256": sha(MDGEN),
    "base_plan_sha256": sha(BASE),
    "superseded": superseded,
}
new["amendments"] = list(base.get("amendments", [])) + [amend]
json.dump(new, open(OUT, "w"), indent=1)
open(OUT, "a").write("\n")

# ---------------------------------------------------------------- PLAN.md fragments
L = []
L.append("| arm | granted % of C | SC-4 % of C | SC-4 E ms | C ms | NIC util | F6: pieces must beat it by (granted → SC-4) |")
L.append("|---|---|---|---|---|---|---|")
for a in new_arms:
    o = old_arms[a["arm"]]
    f6 = "—" if a["f6_min_margin_pct"] is None else (
        f"{o['f6_min_margin_pct']:.3f} → **{a['f6_min_margin_pct']:.3f} pts / {a['f6_min_margin_ms']:.1f} ms** (vs {a['f6_vs']})")
    L.append(f"| `{a['arm']}` | {o['predicted_pct_of_C']:.3f} | {a['predicted_pct_of_C']:.3f} | {a['predicted_ms']:.2f} | "
             f"{a['C_ms']:.1f} | {a['nic_util']:.3f} | {f6} |")
L.append("")
L.append("| round 5 §6 arm (F′ 50, 1:2:1, HBM′) | round5.md | SC-4 | relative |")
L.append("|---|---|---|---|")
for k, v in spec["round5_anchor"].items():
    L.append(f"| `{k}` | {v['pinned_pct']:.2f} | {v['spec_pct']:.3f} | {100*v['rel']:+.2f}% |")
L.append("")
L.append("| F6 arm | alternative | granted threshold (pts) | SC-4 threshold (pts) | SC-4 threshold (ms) |")
L.append("|---|---|---|---|---|")
for a in new_arms:
    if a["f6_min_margin_pct"] is not None:
        o = old_arms[a["arm"]]
        L.append(f"| {a['f6_vs'].split('.')[0]} | {a['arm'].split('.')[1]} | {o['f6_min_margin_pct']:.3f} | "
                 f"{a['f6_min_margin_pct']:.3f} | {a['f6_min_margin_ms']:.2f} |")
L.append("")
osw = base["scaled"]["sweep"]
L.append("| sweep quantity | granted | SC-4 |")
L.append("|---|---|---|")
L.append(f"| S1 at cap′ (MB/s) | {osw['cap_s1_MBps']} | {thr['cap_s1_MBps']} |")
L.append(f"| fit below S1 | {osw['fit_below']['a_pct']:.3f} + {osw['fit_below']['b_pct_MBps']:.3f}/cap′ | "
         f"{thr['fit_below']['a_pct']:.3f} + {thr['fit_below']['b_pct_MBps']:.3f}/cap′ |")
L.append(f"| fit above S1 | {osw['fit_above']['a_pct']:.3f} + {osw['fit_above']['b_pct_MBps']:.3f}/cap′ | "
         f"{thr['fit_above']['a_pct']:.3f} + {thr['fit_above']['b_pct_MBps']:.3f}/cap′ |")
L.append(f"| predicted threshold cap′ (MB/s), utilization | {osw['predicted_threshold_cap_MBps']}, "
         f"{osw['predicted_threshold_util']} | {thr['predicted_threshold_cap_MBps']}, {thr['predicted_threshold_util']} |")
L.append("")
L.append("| diagnostic (not a prediction), % of C beyond the slot | SC-4 | stage 0 packed | one fold after the last arrival |")
L.append("|---|---|---|---|")
for k, v in spec["diagnostics"].items():
    L.append(f"| {k} | {v['spec']:.3f} | {v.get('packed_stage0', '—') if 'packed_stage0' not in v else format(v['packed_stage0'], '.3f')} | "
             f"{v['single_fold']:.3f} |")
open(MD, "w").write("\n".join(L) + "\n")
print("PLAN.json written:", OUT, "sha256", sha(OUT))
print("superseded fields:", len(superseded))
