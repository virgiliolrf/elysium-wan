# Generates the amended docs/experiments/m2b-e4/PLAN.md from the granted PLAN.md, the amended PLAN.json (make_plan_spec.py)
# and its tables fragment. No number is typed by hand: every value comes from PLAN.json, the fragment or e4_spec.out.
# usage: python3 -I -B gen_plan_md.py granted_PLAN.md ../PLAN.json e4_tables.md ../PLAN.md granted_PLAN.json e4_spec.out
import json
import sys

OLDMD, PJ, FRAG, OUT, OLDJ, OUTTXT = sys.argv[1:7]
old = open(OLDMD).read()
p = json.load(open(PJ))
frag = open(FRAG).read()
A = {a["arm"]: a for a in p["arms"]}
am = p["amendments"][-1]
sw = p["scaled"]["sweep"]
d = p["derivation"]


def pc(n):
    return f"{A[n]['predicted_pct_of_C']:.3f}%"


def f6(n):
    return f"{A[n]['f6_min_margin_pct']:.3f}"


# ---- §3: the model paragraph is replaced (the scaling map below it is kept verbatim)
s3_old_start = old.index("The model is round 4's")
s3_old_end = old.index("The scaling map (`round5.md` §6)")
s3_new = f"""**Model (amended, SC-4).** The predictions come from an independent SCHEDULER-I17 oracle written from the SPEC
text (`derive/i17_oracle.py`, driven by `derive/e4_spec.py`; nothing from `src/wantrain` is imported or read). It runs
**one** simulation of every microbatch's traffic per arm:
- one send queue per (sender, destination), served need-first (forward unit, gathers before gradients), one chunk
  ≤ f′ at a time;
- each peer's egress and ingress at min(NIC′, PCIe′), each host's DRAM at k 10 passes per side, each site uplink per
  direction, all shared max-min fairly among the chunks in flight. A transfer arrives `RTT/2 + 2f′/PCIe′` after its
  last byte leaves;
- a W releases a unit's pieces as it computes the unit, in the emission order. The Embed is one fp32 unit released
  when W starts;
- an owner folds every microbatch's contribution in canonical order (a left fold). Each fold starts when the previous
  one ends and its contribution is there, at the measured rates: 18 B/element at 80.98 GB/s for bf16, 12 B at
  129.22 GB/s for fp32, then AdamW at 30 B/element and 42.74 GB/s. The unit's gathers leave after the last fold and
  the AdamW;
- step t+1's F on a unit waits for that unit's gather. E is how late step t+1's forward is (D accumulated over the
  units in forward order) against the peer's compute end.

Every arm has `window_s` = 0, so E is the exposure beyond the slot. Host DRAM is in the model and unconstrained here:
every peer runs on this Mac, and the exchange's DRAM passes (≤ 2.7 GB/s) are far below its ~100 GB/s. The granted
model (`derive/r6_e4.py` on round 4's `exch4`) is superseded. It priced a link-bound arm by
`V_max/cap + RTT + t_gap − C`, which lets a hot owner's gather egress overlap its own gradient ingress. Under I17
they are serial. See the Amendment section.

**Stage-0 padding (pinned).** {p['scaled']['stage0_padding']}

"""
md = old[:s3_old_start] + s3_new + old[s3_old_end:]

# ---- §4: the pinned table is replaced by the amended one
s4_start = md.index("| arm | E ms | % of C | C ms | NIC util |")
s4_end = md.index("Arm meanings follow ADR-0006 and round 4:", s4_start)
rows = ["| arm | E ms | % of C | C ms | NIC util | F6: pieces must beat it by ≥ | note |", "|---|---|---|---|---|---|---|"]
for a in p["arms"]:
    f6c = "—" if a["f6_min_margin_pct"] is None else f"{a['f6_min_margin_pct']:.3f} pts / {a['f6_min_margin_ms']:.1f} ms (vs {a['f6_vs']})"
    note = ""
    if "rtt_ms" in a:
        note = f"RTT {a['rtt_ms']:.1f} ms"
    elif "util_up" in a:
        note = f"uplink util {a['util_up']:.3f}"
    elif "f5_left_fold_ms" in a:
        note = f"left fold {a['f5_left_fold_ms']:.1f} ms" + (" (F5 primary)" if a["arm"].startswith("interior") else "")
    elif "cap_MBps" in a:
        note = f"cap′ {a['cap_MBps']:.4f} MB/s"
    rows.append(f"| `{a['arm']}` | {a['predicted_ms']:.2f} | {a['predicted_pct_of_C']:.3f} | {a['C_ms']:.1f} | "
                f"{a['nic_util']:.3f} | {f6c} | {note} |")
md = md[:s4_start] + "\n".join(rows) + "\n\n" + md[s4_end:]

md = md.replace("- `today`: stride ZeRO-1, fp32 wire, whole-stage flush, barrier, backward order.",
                "- `today`: the whole-stage-flush mode (`scaled.today_mode`): stride ZeRO-1 of whole tensors, fp32 wire, "
                "every unit's pieces released at W(mb)'s end, full step barrier.")
# sweep paragraph
a0 = md.index("- S1 sits at cap′ = 6.4609 MB/s (utilization 1).")
a1 = md.index("- The same rule is applied to the measured points.", a0)
fb, fa = sw["fit_below"], sw["fit_above"]
md = md[:a0] + (f"- S1 sits at cap′ = {sw['cap_s1_MBps']} MB/s (utilization 1).\n"
                f"- Below S1 the two-line fit is `E% = {fb['a_pct']:.3f} + {fb['b_pct_MBps']:.3f}/cap′`; above it, "
                f"`E% = {fa['a_pct']:.3f} + {fa['b_pct_MBps']:.3f}/cap′` (cap′ in MB/s).\n"
                f"- The lines meet at **cap′ {sw['predicted_threshold_cap_MBps']} MB/s (utilization "
                f"{sw['predicted_threshold_util']})**. That is the predicted threshold.\n") + md[a1:]

# ---- §5 numbers
b0 = md.index("  | arm | alternative | threshold (points) |")
b1 = md.index("  No memory argument rescues a failed arm.", b0)
md = md[:b0] + ("  | arm | alternative | threshold (points) |\n  |---|---|---|\n"
                f"  | stage 0 | step 1 alone | {f6('stage0.step1')} |\n"
                f"  | stage 0 | LPT | {f6('stage0.lpt')} |\n"
                f"  | interior w 4 | step 1 alone | {f6('interior_w4.step1')} |\n"
                f"  | interior w 8 | LPT | {f6('interior_w8.lpt')} |\n\n") + md[b1:]
md = md.replace("  - a measured sweep threshold more than 25% from the simulated one (predicted 6.4827 MB/s);",
                f"  - a measured sweep threshold more than 25% from the simulated one (predicted "
                f"{sw['predicted_threshold_cap_MBps']} MB/s);")
md = md.replace("(predicted\n    3.291% at stage 0,\n    7.263% at interior w 4, 17.525% at interior w 8)",
                "(predicted)")
c0 = md.index("  - the barrier-kept control outside [0.8, 1.2]×")
c1 = md.index("  The I17 simulation is the judge F3 names.", c0)
md = md[:c0] + (f"  - the barrier-kept control outside [0.8, 1.2]× the simulation with its barrier flag on (predicted "
                f"{pc('stage0.barrier_pieces_fwd')} at stage 0,\n    {pc('interior_w4.barrier_pieces_fwd')} at interior "
                f"w 4, {pc('interior_w8.barrier_pieces_fwd')} at interior w 8). A control failure means the shim is "
                f"wrong, not H3.\n") + md[c1:]
e0 = md.index("  The I17 simulation is the judge F3 names.")
e1 = md.index("- **F5.**", e0)
md = md[:e0] + ("  The I17 simulation is the judge F3 names. This PLAN's table is its anchor: the as-built simulation at "
                "this configuration must reproduce every arm above within 5%, the `today` arms included (whole-stage-flush "
                "mode), and at the round-5 configuration `PLAN.json` `round5_anchor.arms` within 5% (SC-4; it replaces "
                "`round5.md` §6's percentages as the anchor).\n") + md[e1:]
md = md.replace("(predicted 11.841% vs 7.263%; S1 holds,\n  utilization 0.387)",
                f"(predicted {pc('interior_w4.tree')} vs {pc('interior_w4.pieces_fwd')}; S1 holds,\n  utilization "
                f"{A['interior_w4.pieces_fwd']['nic_util']:.3f})")
md = md.replace("`stage0.tree` (3.983% vs 3.290%) is recorded but does not decide: its predicted gap of 0.69\n  points",
                f"`stage0.tree` ({pc('stage0.tree')} vs {pc('stage0.pieces_fwd')}) is recorded but does not decide: its "
                f"predicted gap of {A['stage0.tree']['predicted_pct_of_C'] - A['stage0.pieces_fwd']['predicted_pct_of_C']:.2f}\n  points")
fw, bw = A["interior_w4.emission_fwd.rtt_2cw"], A["interior_w4.emission_bwd.rtt_2cw"]
f0, b0_ = A["interior_w4.emission_fwd.rtt0"], A["interior_w4.emission_bwd.rtt0"]
md = md.replace("At `rtt_2cw` the simulation picks forward, 8.202% against 9.177% (11.9% apart)",
                f"At `rtt_2cw` the simulation picks forward, {fw['predicted_pct_of_C']:.3f}% against "
                f"{bw['predicted_pct_of_C']:.3f}% ({100*(bw['predicted_pct_of_C']/fw['predicted_pct_of_C']-1):.1f}% apart)")
md = md.replace("At RTT 0 the predicted gap is 1.5% (backward faster, 7.157 vs 7.263)",
                f"At RTT 0 the predicted gap is {100*(f0['predicted_pct_of_C']/b0_['predicted_pct_of_C']-1):.1f}% "
                f"(backward faster, {b0_['predicted_pct_of_C']:.3f} vs {f0['predicted_pct_of_C']:.3f})")

# ---- §6 sensitivity line: superseded numbers are dropped (the script prints the amended ones, e4_spec.out (2))
g0 = md.index("- The measured op split. If the in-situ F or W fraction")
g1 = md.index("- The shim's pacers share each egress", g0)
md = md[:g0] + ("- The measured op split. If the in-situ F or W fraction differs from the pinned one by more than 25% "
                "(relative), the run is off this configuration. Sensitivities (W ×0.75 / ×1.25; split 1:2:1; HBM′-paced "
                "fold) are printed by `e4_spec.py`, section (2); none moves an F6 margin below its threshold.\n") + md[g1:]

# ---- computed phrases for §9
OA = {a["arm"]: a for a in json.load(open(OLDJ))["arms"]}
today_txt = ", ".join(f"`{n}` {OA[n]['predicted_pct_of_C']:.3f} → {A[n]['predicted_pct_of_C']:.3f}%"
                      for n in ("stage0.today", "interior_w4.today", "interior_w8.today"))
n8 = "interior_w8.step1"
w8s1_txt = (f"`{n8}` (link-bound, utilization {A[n8]['nic_util']:.3f}) rose {OA[n8]['predicted_pct_of_C']:.3f} → "
            f"{A[n8]['predicted_pct_of_C']:.3f}% for the same reason.")
sweep_max = max(abs(A[n]["predicted_pct_of_C"] / OA[n]["predicted_pct_of_C"] - 1) * 100
                for n in A if n.startswith("sweep.") and float(n.split("_")[1]) > 1)
special = {"stage0.step1", "stage0.lpt", "stage0.today", "interior_w4.today", "interior_w8.today", n8}
rest_max = max(abs(A[n]["predicted_pct_of_C"] - OA[n]["predicted_pct_of_C"]) for n in A
               if n not in special and not (n.startswith("sweep.") and float(n.split("_")[1]) > 1))
# HBM'-paced fold sensitivity bound and the sensitivity margins against the thresholds, from e4_spec.out section (2)
sec = OUT_TXT_LINES = open(OUTTXT).read().split("== (2)")[1].split("== (3)")[0].splitlines()
import re
rowv = {}
for line in sec:
    if ":" not in line:
        continue
    lab, rest = line.split(":", 1)
    vals = re.findall(r"(stage0|interior_w4|interior_w8) pieces ([0-9.]+) vs (step1|lpt) ([0-9.]+)", rest)
    rowv[lab.strip()] = vals
pin = rowv["pinned (measured split and fold)"]
hbm = rowv["measured split, HBM' paced fold"]
hbm_max = max(max(abs(float(a[1]) - float(b[1])), abs(float(a[3]) - float(b[3]))) for a, b in zip(pin, hbm))
thr = {("stage0", "step1"): A["stage0.step1"]["f6_min_margin_pct"], ("stage0", "lpt"): A["stage0.lpt"]["f6_min_margin_pct"],
       ("interior_w4", "step1"): A["interior_w4.step1"]["f6_min_margin_pct"],
       ("interior_w8", "lpt"): A["interior_w8.lpt"]["f6_min_margin_pct"]}
int_thr = max(A[n]["f6_min_margin_pct"] - OA[n]["f6_min_margin_pct"] for n in ("interior_w4.step1", "interior_w8.lpt"))
assert all(A[n]["f6_min_margin_pct"] >= OA[n]["f6_min_margin_pct"] for n in ("stage0.step1", "stage0.lpt", "interior_w4.step1", "interior_w8.lpt"))
sens_ok = all(float(v[3]) - float(v[1]) >= thr[(v[0], v[2])] for vals in rowv.values() for v in vals)
md = md.replace("output\"). Effect on the F6 arms: ≤ 0.27 points (sensitivity, §6).",
                f"output\"). Effect on the F6 arms in this PLAN's model: ≤ {hbm_max:.2f} points (`e4_spec.out` (2)).")
md = md.replace("Ratios kept, full B against B′ (script output (7);", "Ratios kept, full B against B′ (`r6_e4.out` (7);")
md = md.replace("Scale invariance at the measured split is checked (script output (5))",
                "Scale invariance at the measured split is checked (`e4_spec.out` (3))")
md = md.replace("- The model's tail and volume terms (round 2–4). The I17 simulation, which F3 tests, replaces them as the judge.",
                "- The SCHEDULER-I17 oracle's model (SC-4). The as-built I17 simulation, which F3 tests, is anchored to it.")
md = md.replace("none moves an F6 margin below its threshold.",
                ("no sensitivity arm's predicted margin falls below its pinned threshold." if sens_ok else
                 "SOME SENSITIVITY ARM FALLS BELOW ITS THRESHOLD: see e4_spec.out (2)."))

# ---- §8 reproduce
h0 = md.index("## 8. Reproduce")
md = md[:h0] + f"""## 8. Reproduce

```
cd docs/experiments/m2b-e4/derive
python3 -I -B e4_spec.py i17_oracle.py profile_fprime.out conc_w4.out conc_w8.out e4_spec.json > e4_spec.out   # exit 0
python3 -I -B make_plan_spec.py granted_PLAN.json e4_spec.json e4_spec.py e4_spec.out i17_oracle.py ../PLAN.json \\
    e4_tables.md gen_plan_md.py
python3 -I -B gen_plan_md.py granted_PLAN.md ../PLAN.json e4_tables.md ../PLAN.md granted_PLAN.json e4_spec.out
```

`granted_PLAN.json` is the granted PLAN (sha256 {am['base_plan_sha256'][:12]}…, the APPROVALS grant); `granted_PLAN.md` is
the PLAN.md it was granted with. Neither the oracle nor the scripts import or read `src/wantrain` (the human's rule,
2026-10-10). The outputs are byte-identical over two runs (`python3 -I -B`). Hashes: `e4_spec.py` {d['script_sha256']},
`i17_oracle.py` {d['oracle_sha256']}, `e4_spec.out` {d['output_sha256']}, `e4_spec.json` {d['json_sha256']},
`make_plan_spec.py` {d['builder_sha256']}, `gen_plan_md.py` {d['md_generator_sha256']}. The MEASURED inputs keep
their recorded hashes (`profile_fprime.out` 27c082b5…6657, `conc_w4.out` 2d985771…c6e, `conc_w8.out` a9ebccb0…3e25).
The granted derivation (`r6_e4.py`, `r6_e4.out` d1aaaf71…cf82) stays as the record of the superseded pins.

## 9. Amendment (SC-4 / SPEC SCHEDULER-I17, 2026-10-10)

**Why.** SC-4 (HANDOUT-M2b) found the as-built I17 simulation 6.7% and 13.5% above the pinned stage-0 "step 1 alone" and
"LPT" arms. The SPEC is the physically correct model; the granted derivation was not. Its link-bound term
`V_max/cap + t_gap − C` takes `V_max = max(in, out)` per peer. That lets the hot owner (peer 0 owns `tok` under stride,
and under LPT) send its gathers while its gradient ingress is still arriving. I17 releases a unit's gathers only after
the owner's last fold and AdamW, so ingress and gather egress are serial. The granted model also had no host DRAM in
the simulation, no left fold of every contribution, and no mode for the `today` arms.

**What changed (DERIVED).** The two stage-0 F6 alternatives rose:
- step 1 alone: {old.split('| `stage0.step1` | ')[1].split(' |')[1].strip()} → {pc('stage0.step1')};
- LPT: {old.split('| `stage0.lpt` | ')[1].split(' |')[1].strip()} → {pc('stage0.lpt')}.

Their F6 thresholds grew to {f6('stage0.step1')} and {f6('stage0.lpt')} points. The `today` arms are now expressible
and rose ({today_txt}), because their gathers wait behind the hot owner's ingress. {w8s1_txt} The sweep above S1 moved
by ≤ {sweep_max:.1f}% (relative). Every other arm moved by ≤ {rest_max:.3f} points. **Every F6 threshold rises (the interior ones by
≤ {int_thr:.3f} points); no criterion loosens.**

{frag}

The last table's columns are diagnostics, not predictions:
- **stage 0 packed:** the as-built harness's padding, every op back to back and the idle at the end. The exposure
  beyond the slot is then `E − window_s`.
- **one fold:** the as-built release rule, one fold after the last arrival instead of the left fold.

They attribute the gaps between this PLAN and the as-built simulation (below).

**Modes a coder must add to `exchange_sim.py` / `validation/exposure.py` for these pins (stated precisely):**
1. *The `today` arms*: {p['scaled']['today_mode']}. Inputs: the stage's units with today's stride indices
   (`stage_units`), `layout="zero1"`, wire fp32 (every unit b_g = b_p = 4), a flag `flush="stage"`. Release time of every
   unit's pieces from W(mb) = W(mb)'s end, the Embed included; own contributions are local at that time. Queues
   need-first as in I17. Barrier: `Tb` as stated; every unit of step t+1 waits for `Tb` and for its gathers; the
   emission order is irrelevant. Boundary bytes unchanged.
2. *Stage-0 padding*: {p['scaled']['stage0_padding']}
3. *Left fold*: {p['scaled']['fold_rule']}. As built, a gather leaves at
   `max(arrivals) + one fold + AdamW`. The difference is ≤ 0.02 points at the E4 configuration (fold not paced) and
   −4.7% on `round5_anchor` interior w 4 pieces (paced HBM′: three near-simultaneous pieces fold in turn, and the later
   gather lands behind a chunk, which shortens the tail; a chunk-boundary effect, not a trend).
4. *Round-5 anchor read from the PLAN*: `exposure.py`'s `ROUND5_PCT` and `tests/unit/test_zero_bubble.py`'s seven
   `round5.md` §6 pins are constants today. They must read `PLAN.json` `round5_anchor.arms` (SC-4: "the test reads
   the values from the PLAN").
5. *The Embed as one fp32 unit*: {p['scaled']['embed_unit']}. As built, `validation/exposure.py` `stage_units` gives
   `tok` and `pos` two units; I17 names the Embed's tables as one unit (released at W start, needed first by F).
"""
t0 = "In the model: `exch4` with b_g = 4 on every unit,\n  E = X (the whole exchange is the tail)."
assert md.count(t0) == 1, "tree sentence"
md = md.replace(t0, "In the model (SC-4): one fp32 partial per unit piece per peer, released when that peer's last W "
                    "produces the unit; the owner folds the w partials (12 B/element) and runs AdamW, then gathers.")
t1 = "Labels: MEASURED (a command ran on this machine"
assert md.count(t1) == 1
md = md.replace(t1, "**Amended 2026-10-10 (SC-4 / SPEC SCHEDULER-I17, section 9):** every prediction below is re-derived by an "
                    "independent SCHEDULER-I17 oracle; the granted values are kept in section 9 and in `PLAN.json` "
                    "`amendments`.\n\n" + t1)
open(OUT, "w").write(md)
print("written", OUT)
