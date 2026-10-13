# m2b-e7: the planner's choice per fleet (ADR-0006 E7, F9), re-pinned at the planner's costs

ADR: `docs/design/adr/ADR-0006-exact-fsdp-placement.md` (granted). Handout precondition P3 (`docs/design/HANDOUT-M2b.md`),
ADR R16: "E7's predictions use 8 FLOPs per param; the planner's `layer_costs` counts attention (Head/block 0.558 vs
0.651), so E7's PLAN re-pins them with the planner's costs before it runs." Task `plan-wide-e7`
(`docs/design/PLAN-M2b.json`) reads every configuration's expected choice from `PLAN.json` beside this file. Hardware
switch: gpu false. Cloud: none (`"cloud": null`). Every number below is DERIVED by `derive/e7.py` on this machine's CPU;
every input is HYPOTHESIS, as in rounds 3–5.

**Amended 2026-10-10 (SC-4 / SPEC SCHEDULER-I17, section 9):** the expected choices, steps, margins, ties and refusals below are re-derived by an independent SCHEDULER-I17 oracle with host DRAM inside the exchange simulation and gathers after the owner's last fold and AdamW. 4 choices changed (`32xH100-400G-1gph-k10`, `32xH100-400G-2gph-k10`, `32xH100-2sites-10G-uplink`, `32xH100-400G-1gph-k10-memhost20`); the granted values are kept in section 9 and in `PLAN.json` `amendments`.

## 1. Hypothesis

On each pinned fleet, `plan_job` (SCHEDULER task 4) chooses the plan pinned in `PLAN.json`. It applies SCHEDULER-I16 as
written: refuse pieces on S1 (NIC, host DRAM, uplink), S2, class or MEM-HOST, with ε 3% (ADR question 1, assumed) and
k counted (R10 and R13 deferred). The rationale names every refused candidate with its link, number, class or
MEM-HOST. The planner never picks a plan whose simulated step is more than 1% above the best admitted candidate's.

## 2. What changed against round 5: the costs

FACTS (code read on this tree):
- `partition.layer_costs` (`partition.py:164-175`) prices `[Embed 3h, Block h(72h + 12s) × n, Head 6hV]` per token.
  This is F + B + W from the zero-bubble table, with attention included.
- `zero_bubble.analytic_costs` (`zero_bubble.py:602-618` on today's tree; restated in `e7_spec.py`, not imported) gives per-layer F, B, W from the same table. Recompute adds
  one F to B.
- `planner.plan_job` cuts layers with `proportional_partition` (the exact min-max DP over `layer_costs`,
  `planner.py:264`).

Round 5 (`round5.md` §8, via `r4.py`) uses its own cost: 8 FLOPs per GEMM parameter (F 2, B 4 with recompute, W 2),
with a c/4, c/2, c/4 split and the Head at 8·V·h.

The derivation calls the repository functions and does not re-implement them:
- **Cut:** `proportional_partition(cfg, [1.0]·S)`. It produces the same cuts as round 5 at S 4, 8, 16, 32: first =
  Embed + blocks, last = blocks + Head.
- **F, B, W per stage:** `analytic_costs` with recompute on, in FLOPs per token, times `T_mb/F`. Per block this is
  `96h² + 16hs` = 112 h² at s = h, against round 5's 96 h² (×1.167). Per Head it is `8hV`, unchanged; per Embed `3h`
  (negligible). It equals (4/3)·`layer_costs` per block and per Head, so the Head/block ratio is 0.558 (`layer_costs`)
  against round 5's 0.651.
- **Everything moves with c_s:** `c_s = (F + B + W)·T_mb/F`, `C = (M/w)·max_s c_s`, the lane solve (the repository's
  `zero_bubble`) on those F, B, W. The exposure tail uses the planner's `c_F`, `c_W` and `t_gap = c_B + c_W/U`. S1 on
  NIC and host DRAM and S2 are all computed on this `c_s`.
- **Unchanged from round 5:** `T_ovh`, bytes, ownership, host packing (stage-contiguous), 600 GB/s host DRAM, PCIe
  50 GB/s, F 400 TFLOP/s, `T_mb` 4096, M = 32 (32 GPUs) or 128 (128 GPUs).

Control: the script's `r5` mode reproduces round 5 §3 exactly (11 pinned cells, asserted) and round 5's output hash
(`5ac4f609…50b4`) on the extracted `r5.py`.

Per stage (400 Gbps; ms, GB/s per GPU on host DRAM):

| stage | round 5 `c_s` | planner `c_s` | NIC S1 util (r5 → planner) | host DRAM k 2 / 6 / 10 (planner) | k a 1-GPU 600 GB/s host admits (r5 → planner) |
|---|---|---|---|---|---|
| B stage 0 (Embed + Block 0), S 32, w 4 | 16.49 | 19.24 | 0.945 → 0.810 | 162.0 / 485.9 / 809.8 | 6.35 → 7.41 |
| B interior | 16.49 | 19.24 | 0.448 → 0.384 | 76.7 / 230.2 / 383.6 | 13.41 → 15.64 |
| B last (Block 31 + Head) | 27.23 | 29.98 | 0.391 → 0.355 | 71.0 / 213.0 / 355.0 | 15.35 → 16.90 |
| (16, 2) stage 0 (Embed + 2 blocks) | 32.99 | 38.48 | 0.444 → 0.380 | 76.1 / 228.2 / 380.4 | 13.52 → 15.77 |
| (16, 2) interior | 32.99 | 38.48 | 0.285 → 0.244 | 48.8 / 146.5 / 244.1 | 21.07 → 24.58 |
| (16, 2) last | 43.72 | 49.22 | 0.259 → 0.231 | 46.1 / 138.3 / 230.5 | 23.12 → 26.03 |

The pipeline period and S2:
- **Point B:** C goes 871.4 → 959.3 ms and the lane makespan 1041.1 → 1214.3 ms. At stage 0, E is 24.07 ms against
  ε·C 28.78, so S2 holds by 16% (it was 6% in round 5).
- **(16, 2):** C goes 699.6 → 787.5 ms and the lane makespan 904.0 → 1074.4 ms. At stage 0, E is 12.6 ms against
  ε·C 23.6.

## 3. Expected choices (pinned in `PLAN.json`), against ADR-0006's a-priori choices

Each candidate's step is `max(lane makespan + worst-stage E + T_ovh, busiest host's DRAM time)` (round 5 §3). Pieces
are admitted only where S1 holds on every NIC and host and S2 holds. The runner-up is the second-best **admitted**
candidate, and the margin is its step over the choice's.

| config (`id`) | expected choice | step ms | runner-up, ms | margin % | tie | refused (kinds) | emission | granted choice, ms | changed |
|---|---|---|---|---|---|---|---|---|---|
| 32xH100-400G-1gph-k10 | S 16, w 2, zero1 + native | 1232.1 | S 32, w 1, 1316.0 | 6.81 | no | 3 (S2) | forward | S 16, w 2, pieces + native, 1136.3 | **yes** |
| 32xH100-400G-2gph-k10 | S 32, w 1 | 1316.0 | S 16, w 2, zero1 + native, 1394.3 | 5.95 | no | 3 (S1 host DRAM; S2) | — | S 16, w 2, zero1 + native, 1163.7 | **yes** |
| 32xH100-400G-4gph-k10 | S 32, w 1 | 1316.0 | S 16, w 2, zero1 + native, 1830.4 | 39.08 | no | 3 (S1 host DRAM; S2) | — | S 32, w 1, 1275.4 | no |
| 32xH100-400G-8gph-k10 | S 32, w 1 | 1316.0 | S 16, w 2, zero1 + native, 2700.5 | 105.2 | no | 3 (S1 host DRAM; S2) | — | S 32, w 1, 1275.4 | no |
| 32xH100-400G-8gph-k2 | S 16, w 2, pieces + native | 1155.4 | S 16, w 2, zero1 + native, 1230.3 | 6.48 | no | 2 (S1 host DRAM; S2) | forward | S 16, w 2, pieces + native, 1136.3 | no |
| 32xH100-100G-1gph-k10 | S 32, w 1 | 1440.8 | S 16, w 2, zero1 + native, 2471.5 | 71.53 | no | 3 (S1 NIC; S2) | — | S 32, w 1, 1275.4 | no |
| 32xH100-100G-8gph-k10 | S 32, w 1 | 1440.8 | S 16, w 2, zero1 + native, 3199.5 | 122.06 | no | 3 (S1 NIC; S1 host DRAM; S2) | — | S 32, w 1, 1275.4 | no |
| 128xH100-400G-1gph-k10 | S 32, w 4, zero1 + native | 2270.2 | S 32, w 2, pieces + native on 64 GPUs, 2358.7 | 3.9 | no | 2 (S1 host DRAM; S2) | backward | S 32, w 4, zero1 + native, 1808.6 | no |
| 128xH100-400G-1gph-k6 | S 32, w 4, pieces + native | 1338.1 | S 32, w 4, zero1 + native, 2209.6 | 65.12 | no | 1 (S2) | forward | S 32, w 4, pieces + native, 1297.2 | no |
| 128xH100-400G-1gph-k2 | S 32, w 4, pieces + native | 1338.1 | S 32, w 4, zero1 + native, 2209.6 | 65.12 | no | 1 (S2) | forward | S 32, w 4, pieces + native, 1297.2 | no |
| 128xH100-400G-8gph-k10 | S 32, w 1 on 32 GPUs | 4365.4 | S 32, w 4, zero1 + native, 5305.6 | 21.54 | no | 3 (S1 host DRAM; S2) | — | S 32, w 1 on 32 GPUs, 4324.8 | no |
| 128xH100-400G-8gph-k2 | S 32, w 4, zero1 + native | 2245.9 | S 32, w 2, pieces + native on 64 GPUs, 2344.5 | 4.39 | no | 2 (S1 host DRAM; S2) | forward | S 32, w 4, zero1 + native, 1808.6 | no |
| 128xH100-100G-1gph-k10 | S 32, w 1 on 32 GPUs | 4490.3 | S 32, w 2, zero1 + native on 64 GPUs, 5709.0 | 27.14 | no | 3 (S1 NIC; S2; S1 host DRAM) | — | S 32, w 1 on 32 GPUs, 4324.8 | no |
| 128xH100-100G-8gph-k10 | S 32, w 1 on 32 GPUs | 4490.3 | S 32, w 2, zero1 + native on 64 GPUs, 7601.9 | 69.3 | no | 3 (S1 NIC; S1 host DRAM; S2) | — | S 32, w 1 on 32 GPUs, 4324.8 | no |
| 32xH100-2sites-10G-uplink | S 16, w 2, zero1 + native | 1388.2 | S 32, w 1, 1418.4 | 2.18 | no | 6 (S2; S1 uplink) | forward | S 16, w 2, pieces + native, 1293.7 | **yes** |
| 32xGPU-mixed-classes | S 32, w 1 | 1316.0 | — | — | no | 6 (class) | — | S 32, w 1, 1275.4 | no |
| 32xH100-400G-1gph-k10-memhost20 | S 16, w 2, zero1 + native | 1232.1 | S 32, w 1, 1316.0 | 6.81 | no | 3 (S2; MEM-HOST) | forward | S 32, w 1, 1275.4 | **yes** |

**4 choices changed against the granted pins** (DERIVED, section 9 gives the mechanism):
- `32xH100-400G-1gph-k10`: pieces at (16, 2) is now refused by S2. Stage 0's E − window_s comes out of the I17 simulation with host DRAM at k 10 shared by its exchange and boundary bytes, and it exceeds ε·C (the refusal table gives the numbers). S 16, w 2, zero1 + native is chosen (1232.1 ms; granted S 16, w 2, pieces + native, 1136.3 ms).
- `32xH100-400G-2gph-k10`: two GPUs share one host's DRAM, so the wide candidates are slower than width 1. S 32, w 1 is chosen (1316.0 ms; granted S 16, w 2, zero1 + native, 1163.7 ms).
- `32xH100-2sites-10G-uplink`: as at 1-GPU hosts, contiguous pieces fails S2. S 16, w 2, zero1 + native is chosen (1388.2 ms; granted S 16, w 2, pieces + native, 1293.7 ms).
- `32xH100-400G-1gph-k10-memhost20`: MEM-HOST is charged on pieces only, as the SPEC states it (SCHEDULER-I16; task 4 keeps `zero1` at today's device admission); every pieces candidate is refused by MEM-HOST and `zero1` is priced as on the 1-GPU, k 10 fleet. S 16, w 2, zero1 + native is chosen (1232.1 ms; granted S 32, w 1, 1275.4 ms).
- The 128-GPU fleets gain the candidate S 32, w 2 on 64 GPUs (SCHEDULER task 4's deepest cut on N/w devices), which the granted list lacked.
- No pinned margin is ≤ 2.0%, so the tie rule applies to none. The nearest is `32xH100-2sites-10G-uplink` (2.18%); with the granted derivation's zero hop comm (sensitivity comm0) it is 0.64%.
- Steps moved by -3.4% to +25.5% against the granted ones. The lane solve now carries the SPEC's per-hop comm (one boundary tensor at the link rate), and the exposure includes host DRAM.

The refusal configurations (inputs HYPOTHESIS):
- **Uplink:** two sites of 16 H100 (1-GPU hosts, 400 Gbps NICs) joined by one 10 Gbps uplink, RTT 50 ms (derivation
  §6). Straddling placements (half of each wide stage per site) are refused naming the uplink. With site-major
  stage-contiguous packing no stage straddles. One pipeline hop crosses the uplink, priced as the delay
  `w·A/cap_up + RTT/2` on that hop, with the step floored at `M·A/cap_up + RTT` = 909.0 ms. The job is planned, not
  refused. Straddling `zero1` is priced at ≥ 93 s (uplink-bound) and is never a contender.
- **Mixed classes:** 17 GPUs of class A, then 15 of class B, at equal rates, so only the class predicate differs from
  the homogeneous row. At every width 2, 4, 8 some stage spans both classes. Every wide candidate is refused naming the
  class, and width 1 is chosen.
- **MEM-HOST:** the 1-GPU, k 10 fleet with 20 GB of host RAM available per host after the reserve. The one-step bound
  (round 5 §4, computed with each candidate's own ownership) per peer, worst stage:

  | (S, w) | pieces | `zero1` |
  |---|---|---|
  | (16, 2) | 23.74 | 23.74 |
  | (8, 4) | 29.73 | 46.17 |
  | (4, 8) | 33.36 | 62.98 |

  Cross-check: (8, 4) interior at 21.74 equals round 5 §4's point A. Every pieces candidate is refused by MEM-HOST (SC-4: the `zero1` column is not charged, SCHEDULER task 4), and S 16, w 2, zero1 + native is chosen.

## 4. F9, restated for this PLAN

F9 fires (the hypothesis is REFUTED) in either case:
- `plan_job` chooses, on any of the 17 configurations, a plan whose `(S, w, shard, wire, gpus_used)` differs from
  `expected_choice`;
- the chosen plan's simulated step is more than 1% above the best admitted candidate's in the planner's own
  simulation.

A refused candidate must appear in the rationale with each pinned reason's `kind` (S1 NIC, S1 host DRAM, S1 uplink,
S2, class, MEM-HOST) and `where` (stage or host). Its `value` must lie on the refusing side of its `bound`. The numbers
are this model's (DERIVED) and are not matched to a tolerance. The as-built SCHEDULER-I17 simulation decides step
times, and E4 checks it.

A mismatch is reported as F9, a specConflict, and is never fixed by editing this file (task `plan-wide-e7`).
**Tie rule (the human's ruling, 2026-10-09; `tieRule` in PLAN.json):** where a config's pinned margin to its runner-up
is at most 2% (`tie: true`; after SC-4 no configuration: the nearest is `32xH100-2sites-10G-uplink` at 2.18%), the runner-up also counts as matching if its
simulated step is within 1% of the best admitted candidate's (F9's second clause). Metrics:
- `choices_matching` (primary): expected 17 (the amended pins).
- `refusals_matching`: expected 50 (SC-4; granted 48).
- `f9_violations`: expected 0, lower is better.
- `tolerancePct`: 1, which is below one count at these totals, so the counts are exact.

## 5. Sensitivity (DERIVED by `e7_spec.py`; `PLAN.json` pins mode `spec`, the others are in `PLAN.json` `sensitivity`)

- **comm0** (the lane solve without the SPEC's per-hop comm, as the granted derivation ran it): 0 of the 17 choices differ.
- **nodram** (host DRAM out of the simulation, S1 kept: the granted model's reading of I17): 6 of the 17 choices differ (`32xH100-400G-1gph-k10`, `32xH100-400G-2gph-k10`, `32xH100-400G-4gph-k10`, `32xH100-400G-8gph-k10`, `128xH100-400G-8gph-k10`, `32xH100-2sites-10G-uplink`). This is the term SC-4 adds.
- **embed_p0** (a diagnostic, not a prediction): `zero1` gives the whole Embed unit to peer 0, as the as-built planner lays it out (`planner._stage_units` gives the Embed one stride index, 0). The runtime's stride (`runtime/stage.py:183`, `params[peer::w]`) puts `pos`, parameter 1, on peer 1. This layout changes 1 choice(s): `128xH100-400G-1gph-k10` → S 32 w 2 pieces + native (+0.88% to its runner-up).
- Recompute on stays load-bearing, as the granted section 5 found. The S1 window and the ADR's c/4 : c/2 : c/4 split are superseded by R16 and were not re-run.

## 6. Contradictions with ADR-0006 and its tasks (reported, not changed)

1. **Two S1 windows.** ADR-0006's model and task `i16-admission` pin S1 at `c_s = 8·P^F_s·T_mb/F`, reproducing
   `round5.md` §2: 944.8 GB/s at B stage 0, k 10. R16 puts the attention-inclusive cost in the planner. If both land,
   S1 and S2/C read different windows, which is what round 5 §1 removed. E7's choices do not depend on it (14 / 14), so
   the PLAN pins kinds, places and sides, not S1 values. RULED by the human (2026-10-09): one window, the planner's `layer_costs` in S1, S2
   and C (SCHEDULER-I16 and task `i16-admission` amended). A 1-GPU host admits B stage 0 up to k 7.41.
2. **The ADR text's numbers are superseded, its choices are not.** DERIVED here:

   | quantity | ADR-0006 | this PLAN |
   |---|---|---|
   | 32 H100: pieces, S 32 w 1 | 966.8, 1102.2 ms | 1136.3, 1275.4 |
   | 128 H100: pieces, `zero1` + native, S 32 w 1 | 1124.5, 1721.3, 3887.8 ms | 1297.2, 1808.6, 4324.8 |
   | margin at (16, 2) | 2.8% | 2.41% |
   | C at point B | 871.4 ms | 959.3 |
   | S2 margin at B stage 0 | 6% | 16% |
   | R7: (8, 4), (4, 8) | 902.5, 867.8 ms | 1064.9, 1016.5 |
   | R13 at 100 Gbps: pieces vs S 32 w 1 | 2291.7 vs 3887.8 ms | 2379.0 vs 4324.8 |

3. **Near-tie at 2-GPU hosts, k 10 (1.06%).** F9's first clause fires if the as-built simulation moves `zero1`
   (16, 2) against `zero1` (8, 4) by more than about 1% relative. (8, 4)'s step is its lane time, 1176.0 ms, beside a
   host-DRAM bound of 1137.8 ms that R16 marks HYPOTHESIS. Either the human accepts this pin as written, or F9 gains a
   tie rule. RULED by the human (2026-10-09): the tie rule of §4 applies.
4. **The uplink.** ADR E7's "H100 over one 10 Gbps uplink … refused with the link named" holds for straddling
   candidates only. Under the stage-contiguous packing the ADR assumes, nothing straddles and the job is planned at
   1293.7 ms. The boundary crossing is priced by a fluid hop delay (HYPOTHESIS; the ADR's model has no uplink hop).
5. **MEM-HOST scope. SETTLED by the SPEC (SC-4).** SCHEDULER-I16 admits `shard: "pieces"` under its five conditions, host memory among them; SCHEDULER task 4 keeps `zero1` wide stages at today's admission (device memory), and HANDOUT-M2b's rulings say "zero1 stages keep today's FIT check". MEM-HOST is therefore charged on pieces only: the memhost20 choice is S 16, w 2, zero1 + native at 1232.1 ms. The as-built planner charges it on `zero1` too (`planner._price`, `admission.mem_host` outside the pieces branch): a code change, not a PLAN question.
6. **Local commands.** Task `plan-wide-e7` defines no byte-exact check, so `-k "e7_ and identity"` would collect no
   test (pytest exit 5). Correctness uses E1's `e1_features_on`, from dependency `pieces-identity-e1-e2`. The planner's
   choice moves no bit; E1 is what makes the chosen plan exact. The performance command needs the `e7_` test to print
   one JSON line before it asserts; that requirement is not in `plan-wide-e7`'s acceptance today. The failure command
   needs `plan_wide` tests whose names contain `refuse`.

## 7. Local commands (`PLAN.json` `local`; none exists yet, task `plan-wide-e7` writes them)

- **Correctness:** `uv run pytest -q -p no:randomly tests/integration/test_engine_bitwise.py -k e1_features_on`. It
  checks sha256 identity against `FACTORY_REFERENCE=1`, with every feature on.
- **Performance:**
  `uv run pytest -q -p no:randomly -s tests/unit/test_scheduler.py -k e7_ | grep -E '^\{"choices_matching"' | tail -n 1`.
  Its last line is `{"choices_matching", "refusals_matching", "f9_violations"}`.
- **Failure:** `uv run pytest -q -p no:randomly tests/unit/test_scheduler.py -k "plan_wide and refuse"`. It covers the
  uplink, class, MEM-HOST and "not armed" refusals.

## 8. Derivation (reproducible)

```
cd docs/experiments/m2b-e7/derive
python3 -I -B e7_spec.py ../../m2b-e4/derive/i17_oracle.py lane_solve.py asbuilt_lanes.json e7_spec.json \
    > e7_spec.out                                                     # exit 0, ~9 min
python3 -I -B build_plan_spec.py granted_PLAN.json e7_spec.json e7_spec.py e7_spec.out \
    ../../m2b-e4/derive/i17_oracle.py lane_solve.py asbuilt_lanes.json ../PLAN.json e7_tables.md gen_plan_md.py
python3 -I -B gen_plan_md.py granted_PLAN.md granted_PLAN.json ../PLAN.json e7_tables.md ../PLAN.md
```

Nothing from `src/wantrain` is imported or read (the human's rule, 2026-10-10). `granted_PLAN.json` is the granted PLAN
(sha256 afa2355d645d…, the APPROVALS grant); `granted_PLAN.md` the PLAN.md it was granted with. The outputs
are byte-identical over two runs (`python3 -I -B`). Hashes:
- `e7_spec.py` a1511627e8c90ce44a39237e6a8cf65b19e5230d34f8abfda894235324eca551.
- `i17_oracle.py` d01edde9a98972805034c98ada32bc813f3a4e4d02faddd0466a0232984472d7: the I17 model, written from the SPEC text.
- `lane_solve.py` 7f7bf688380e0039da5c283aa7d6ae4217d1ac18348db6aed3a7fc335e4044b6: the lane solve, written from the SPEC's zero-bubble rules (a list scheduler)
  with a closed-form lower bound P* (the last stage's n·(F + B) + (n − K)·W behind the forward fill, then the backward
  chain; K = p microbatches may hold a pending W). It attains P* on every lane, and its periods equal the as-built
  lane solve's on all 644 priced candidates of the four modes (`e7_spec.out`, lanes section).
- `asbuilt_lanes.json` 59405aeb2c07533cdb660e52c5151e3e22636a268aefb7be0e6fd3421ed18747: the as-built lane solve's numbers (`zero_bubble.py` a93a8bed…,
  extracted by `extract_asbuilt.py` from the earlier `e7_spec.json` 4af7911b…), compared, never used to price.
- `e7_spec.out` b37343df63b1f7ecdad96f49193ebcd2c6503f9f06ba7507617bd04d40802aec; `e7_spec.json` 3822dfda45d264eae5d29ddd7149b4e6bcf2dfea090c8cb11a708587b488a09b.
- `build_plan_spec.py` 434de9f9cb2dbbeb94ae401ca758696e8aa451b9b47f63c749fd1b7a0b992020; `gen_plan_md.py` 022b996a55c7ce3a107c96d889b53acb8310eb23046675eb2ffe3d80a0724849.
- `layer_costs` and `analytic_costs` are restated in the script from the SPEC's formulas, not imported.

The granted derivation (`e7.py`, `e7.out` cdc4e496…e920, `e7.json` cb8a7a8a…8557) stays as the record of the superseded
pins.

## 9. Amendment (SC-4 / SPEC SCHEDULER-I17, 2026-10-10)

**Why.** The granted derivation (`e7.py` on round 4's `exch4`) did not model two things I17 requires:
1. Each host's DRAM, at k passes per byte per side, is a resource shared max-min fairly inside the exchange
   simulation. At k 10 a 600 GB/s host moves 30 GB/s per direction, below one 400 Gbit/s NIC's 50 GB/s, so the
   end-of-step bursts are DRAM-bound and E grows.
2. A unit's gathers leave only after the owner's last fold of that unit and its AdamW, so a hot owner's ingress and
   its gather egress are serial.

The SPEC is the physically correct model; the PLAN is amended. Re-derived by an independent oracle, with every input
the granted PLAN pins kept: fleets, k, 600 GB/s hosts, stage-contiguous packing, ε 3%, recompute on, `layer_costs`.
Three things the SPEC states, which the granted derivation did not, are now in:
- the candidate set of SCHEDULER task 4 (S 32, w 2 on 64 of 128 GPUs);
- the per-hop comm of section 2.1;
- MEM-HOST on pieces only (SCHEDULER-I16 and task 4; the granted open question 5).

| config | granted choice, step ms | SC-4 choice, step ms | SC-4 runner-up, ms | margin % (granted → SC-4) | tie | refused (granted → SC-4) | choice changed |
|---|---|---|---|---|---|---|---|
| `32xH100-400G-1gph-k10` | S 16, w 2, pieces + native, 1136.3 | **S 16, w 2, zero1 + native**, 1232.1 | S 32, w 1 (32 GPUs), 1316.0 | 2.41 → 6.81 | False → False | 2 → 3 | **yes** |
| `32xH100-400G-2gph-k10` | S 16, w 2, zero1 + native, 1163.7 | **S 32, w 1 (32 GPUs)**, 1316.0 | S 16, w 2, zero1 + native, 1394.3 | 1.06 → 5.95 | True → False | 3 → 3 | **yes** |
| `32xH100-400G-4gph-k10` | S 32, w 1 (32 GPUs), 1275.4 | **S 32, w 1 (32 GPUs)**, 1316.0 | S 16, w 2, zero1 + native, 1830.4 | 6.26 → 39.08 | False → False | 3 → 3 | no |
| `32xH100-400G-8gph-k10` | S 32, w 1 (32 GPUs), 1275.4 | **S 32, w 1 (32 GPUs)**, 1316.0 | S 16, w 2, zero1 + native, 2700.5 | 89.04 → 105.2 | False → False | 3 → 3 | no |
| `32xH100-400G-8gph-k2` | S 16, w 2, pieces + native, 1136.3 | **S 16, w 2, pieces + native**, 1155.4 | S 16, w 2, zero1 + native, 1230.3 | 2.41 → 6.48 | False → False | 2 → 2 | no |
| `32xH100-100G-1gph-k10` | S 32, w 1 (32 GPUs), 1275.4 | **S 32, w 1 (32 GPUs)**, 1440.8 | S 16, w 2, zero1 + native, 2471.5 | 65.27 → 71.53 | False → False | 3 → 3 | no |
| `32xH100-100G-8gph-k10` | S 32, w 1 (32 GPUs), 1275.4 | **S 32, w 1 (32 GPUs)**, 1440.8 | S 16, w 2, zero1 + native, 3199.5 | 89.04 → 122.06 | False → False | 3 → 3 | no |
| `128xH100-400G-1gph-k10` | S 32, w 4, zero1 + native, 1808.6 | **S 32, w 4, zero1 + native**, 2270.2 | S 32, w 2, pieces + native, 2358.7 | 15.0 → 3.9 | False → False | 2 → 2 | no |
| `128xH100-400G-1gph-k6` | S 32, w 4, pieces + native, 1297.2 | **S 32, w 4, pieces + native**, 1338.1 | S 32, w 4, zero1 + native, 2209.6 | 39.42 → 65.12 | False → False | 1 → 1 | no |
| `128xH100-400G-1gph-k2` | S 32, w 4, pieces + native, 1297.2 | **S 32, w 4, pieces + native**, 1338.1 | S 32, w 4, zero1 + native, 2209.6 | 39.42 → 65.12 | False → False | 1 → 1 | no |
| `128xH100-400G-8gph-k10` | S 32, w 1 (32 GPUs), 4324.8 | **S 32, w 1 (32 GPUs)**, 4365.4 | S 32, w 4, zero1 + native, 5305.6 | 16.5 → 21.54 | False → False | 2 → 3 | no |
| `128xH100-400G-8gph-k2` | S 32, w 4, zero1 + native, 1808.6 | **S 32, w 4, zero1 + native**, 2245.9 | S 32, w 2, pieces + native, 2344.5 | 15.0 → 4.39 | False → False | 2 → 2 | no |
| `128xH100-100G-1gph-k10` | S 32, w 1 (32 GPUs), 4324.8 | **S 32, w 1 (32 GPUs)**, 4490.3 | S 32, w 2, zero1 + native, 5709.0 | 44.74 → 27.14 | False → False | 2 → 3 | no |
| `128xH100-100G-8gph-k10` | S 32, w 1 (32 GPUs), 4324.8 | **S 32, w 1 (32 GPUs)**, 4490.3 | S 32, w 2, zero1 + native, 7601.9 | 44.74 → 69.3 | False → False | 2 → 3 | no |
| `32xH100-2sites-10G-uplink` | S 16, w 2, pieces + native, 1293.7 | **S 16, w 2, zero1 + native**, 1388.2 | S 32, w 1 (32 GPUs), 1418.4 | 2.12 → 2.18 | False → False | 5 → 6 | **yes** |
| `32xGPU-mixed-classes` | S 32, w 1 (32 GPUs), 1275.4 | **S 32, w 1 (32 GPUs)**, 1316.0 | — | None → None | False → False | 6 → 6 | no |
| `32xH100-400G-1gph-k10-memhost20` | S 32, w 1 (32 GPUs), 1275.4 | **S 16, w 2, zero1 + native**, 1232.1 | S 32, w 1 (32 GPUs), 1316.0 | None → 6.81 | False → False | 6 → 3 | **yes** |

Refusals pinned: 48 → 50.

| config | refused candidate | reasons (kind, where, value vs bound) |
|---|---|---|
| `32xH100-400G-1gph-k10` | S 16 w 2 pieces | S2 at stage 0 (first): 32.18 vs 23.63 ms |
| `32xH100-400G-1gph-k10` | S 8 w 4 pieces | S2 at stage 0 (first): 77.8 vs 21.05 ms |
| `32xH100-400G-1gph-k10` | S 4 w 8 pieces | S2 at stage 0 (first): 154.18 vs 19.76 ms |
| `32xH100-400G-2gph-k10` | S 16 w 2 pieces | S1 host DRAM at host 0 (stages [0]): 760.8 vs 600.0 GB/s; S2 at stage 0 (first): 202.68 vs 23.63 ms |
| `32xH100-400G-2gph-k10` | S 8 w 4 pieces | S1 host DRAM at host 0 (stages [0]): 875.7 vs 600.0 GB/s; S2 at stage 0 (first): 478.9 vs 21.05 ms |
| `32xH100-400G-2gph-k10` | S 4 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 875.6 vs 600.0 GB/s; S2 at stage 0 (first): 656.92 vs 19.76 ms |
| `32xH100-400G-4gph-k10` | S 16 w 2 pieces | S1 host DRAM at host 0 (stages [0, 1]): 1249.0 vs 600.0 GB/s; S2 at stage 0 (first): 638.76 vs 23.63 ms |
| `32xH100-400G-4gph-k10` | S 8 w 4 pieces | S1 host DRAM at host 0 (stages [0]): 1751.5 vs 600.0 GB/s; S2 at stage 0 (first): 1469.76 vs 21.05 ms |
| `32xH100-400G-4gph-k10` | S 4 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 1751.2 vs 600.0 GB/s; S2 at stage 0 (first): 1768.93 vs 19.76 ms |
| `32xH100-400G-8gph-k10` | S 16 w 2 pieces | S1 host DRAM at host 0 (stages [0, 1, 2, 3]): 2225.6 vs 600.0 GB/s; S2 at stage 0 (first): 1508.87 vs 23.63 ms |
| `32xH100-400G-8gph-k10` | S 8 w 4 pieces | S1 host DRAM at host 0 (stages [0, 1]): 3076.8 vs 600.0 GB/s; S2 at stage 0 (first): 2878.44 vs 21.05 ms |
| `32xH100-400G-8gph-k10` | S 4 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 3502.4 vs 600.0 GB/s; S2 at stage 0 (first): 3992.93 vs 19.76 ms |
| `32xH100-400G-8gph-k2` | S 8 w 4 pieces | S1 host DRAM at host 0 (stages [0, 1]): 615.4 vs 600.0 GB/s; S2 at stage 0 (first): 142.07 vs 21.05 ms |
| `32xH100-400G-8gph-k2` | S 4 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 700.5 vs 600.0 GB/s; S2 at stage 0 (first): 434.51 vs 19.76 ms |
| `32xH100-100G-1gph-k10` | S 16 w 2 pieces | S1 NIC at first stage: 1.522 vs 1.0 x cap*c_s; S2 at stage 0 (first): 361.15 vs 23.63 ms |
| `32xH100-100G-1gph-k10` | S 8 w 4 pieces | S1 NIC at first stage: 1.751 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.325 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.312 vs 1.0 x cap*c_s; S2 at stage 0 (first): 677.12 vs 21.05 ms |
| `32xH100-100G-1gph-k10` | S 4 w 8 pieces | S1 NIC at first stage: 1.751 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.5 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.497 vs 1.0 x cap*c_s; S2 at stage 0 (first): 879.33 vs 19.76 ms |
| `32xH100-100G-8gph-k10` | S 16 w 2 pieces | S1 NIC at first stage: 1.522 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0, 1, 2, 3]): 2225.6 vs 600.0 GB/s; S2 at stage 0 (first): 1575.4 vs 23.63 ms |
| `32xH100-100G-8gph-k10` | S 8 w 4 pieces | S1 NIC at first stage: 1.751 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.325 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.312 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0, 1]): 3076.8 vs 600.0 GB/s; S2 at stage 0 (first): 2876.48 vs 21.05 ms |
| `32xH100-100G-8gph-k10` | S 4 w 8 pieces | S1 NIC at first stage: 1.751 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.5 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.497 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0]): 3502.4 vs 600.0 GB/s; S2 at stage 0 (first): 3992.96 vs 19.76 ms |
| `128xH100-400G-1gph-k10` | S 32 w 4 pieces | S1 host DRAM at host 0 (stages [0]): 809.8 vs 600.0 GB/s; S2 at stage 0 (first): 96.28 vs 28.78 ms |
| `128xH100-400G-1gph-k10` | S 16 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 652.6 vs 600.0 GB/s; S2 at stage 0 (first): 103.7 vs 23.63 ms |
| `128xH100-400G-1gph-k6` | S 16 w 8 pieces | S2 at stage 0 (first): 34.4 vs 23.63 ms |
| `128xH100-400G-1gph-k2` | S 16 w 8 pieces | S2 at stage 0 (first): 34.4 vs 23.63 ms |
| `128xH100-400G-8gph-k10` | S 32 w 2 pieces | S1 host DRAM at host 0 (stages [0, 1, 2, 3]): 2777.1 vs 600.0 GB/s; S2 at stage 0 (first): 3563.98 vs 57.56 ms |
| `128xH100-400G-8gph-k10` | S 32 w 4 pieces | S1 host DRAM at host 0 (stages [0, 1]): 4773.8 vs 600.0 GB/s; S2 at stage 0 (first): 3875.29 vs 28.78 ms |
| `128xH100-400G-8gph-k10` | S 16 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 5220.7 vs 600.0 GB/s; S2 at stage 0 (first): 4951.38 vs 23.63 ms |
| `128xH100-400G-8gph-k2` | S 32 w 4 pieces | S1 host DRAM at host 0 (stages [0, 1]): 954.8 vs 600.0 GB/s; S2 at stage 0 (first): 189.95 vs 28.78 ms |
| `128xH100-400G-8gph-k2` | S 16 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 1044.1 vs 600.0 GB/s; S2 at stage 0 (first): 519.21 vs 23.63 ms |
| `128xH100-100G-1gph-k10` | S 32 w 2 pieces | S1 NIC at first stage: 2.206 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.116 vs 1.0 x cap*c_s; S2 at stage 0 (first): 901.57 vs 57.56 ms |
| `128xH100-100G-1gph-k10` | S 32 w 4 pieces | S1 NIC at first stage: 3.239 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.535 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.42 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0]): 809.8 vs 600.0 GB/s; S2 at stage 0 (first): 1244.43 vs 28.78 ms |
| `128xH100-100G-1gph-k10` | S 16 w 8 pieces | S1 NIC at first stage: 2.61 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.604 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.573 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0]): 652.6 vs 600.0 GB/s; S2 at stage 0 (first): 1073.25 vs 23.63 ms |
| `128xH100-100G-8gph-k10` | S 32 w 2 pieces | S1 NIC at first stage: 2.206 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.116 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0, 1, 2, 3]): 2777.1 vs 600.0 GB/s; S2 at stage 0 (first): 3696.72 vs 57.56 ms |
| `128xH100-100G-8gph-k10` | S 32 w 4 pieces | S1 NIC at first stage: 3.239 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.535 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.42 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0, 1]): 4773.8 vs 600.0 GB/s; S2 at stage 0 (first): 3873.33 vs 28.78 ms |
| `128xH100-100G-8gph-k10` | S 16 w 8 pieces | S1 NIC at first stage: 2.61 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.604 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.573 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0]): 5220.7 vs 600.0 GB/s; S2 at stage 0 (first): 4951.41 vs 23.63 ms |
| `32xH100-2sites-10G-uplink` | S 16 w 2 pieces | S2 at stage 0 (first): 32.18 vs 23.63 ms |
| `32xH100-2sites-10G-uplink` | S 16 w 2 pieces, straddling | S1 uplink at site0-site1: 175.5 vs 1.25 GB/s |
| `32xH100-2sites-10G-uplink` | S 8 w 4 pieces | S2 at stage 0 (first): 77.8 vs 21.05 ms |
| `32xH100-2sites-10G-uplink` | S 8 w 4 pieces, straddling | S1 uplink at site0-site1: 175.5 vs 1.25 GB/s |
| `32xH100-2sites-10G-uplink` | S 4 w 8 pieces | S2 at stage 0 (first): 154.18 vs 19.76 ms |
| `32xH100-2sites-10G-uplink` | S 4 w 8 pieces, straddling | S1 uplink at site0-site1: 175.5 vs 1.25 GB/s |
| `32xGPU-mixed-classes` | S 16 w 2 pieces | class at stage 8: A+B vs one class  |
| `32xGPU-mixed-classes` | S 16 w 2 zero1 + native | class at stage 8: A+B vs one class  |
| `32xGPU-mixed-classes` | S 8 w 4 pieces | class at stage 4: A+B vs one class  |
| `32xGPU-mixed-classes` | S 8 w 4 zero1 + native | class at stage 4: A+B vs one class  |
| `32xGPU-mixed-classes` | S 4 w 8 pieces | class at stage 2: A+B vs one class  |
| `32xGPU-mixed-classes` | S 4 w 8 zero1 + native | class at stage 2: A+B vs one class  |
| `32xH100-400G-1gph-k10-memhost20` | S 16 w 2 pieces | S2 at stage 0 (first): 32.18 vs 23.63 ms; MEM-HOST at host of a first (2 blocks) peer: 23.74 vs 20.0 GB |
| `32xH100-400G-1gph-k10-memhost20` | S 8 w 4 pieces | S2 at stage 0 (first): 77.8 vs 21.05 ms; MEM-HOST at host of a first (4 blocks) peer: 29.73 vs 20.0 GB |
| `32xH100-400G-1gph-k10-memhost20` | S 4 w 8 pieces | S2 at stage 0 (first): 154.18 vs 19.76 ms; MEM-HOST at host of a first (8 blocks) peer: 33.36 vs 20.0 GB |

| config | SC-4 (spec) | comm0 | nodram | embed_p0 (diagnostic) |
|---|---|---|---|---|
| `32xH100-400G-1gph-k10` | S 16 w 2 zero1 + native 1232.1 | S 16 w 2 zero1 + native 1212.0 (+5.15%) | S 16 w 2 pieces + native 1155.4 (+6.48%) | S 16 w 2 zero1 + native 1233.0 (+6.73%) |
| `32xH100-400G-2gph-k10` | S 32 w 1 1316.0 | S 32 w 1 1274.4 (+7.83%) | S 16 w 2 zero1 + native 1230.3 (+6.96%) | S 32 w 1 1316.0 (+5.95%) |
| `32xH100-400G-4gph-k10` | S 32 w 1 1316.0 | S 32 w 1 1274.4 (+42.1%) | S 16 w 2 zero1 + native 1230.3 (+6.96%) | S 32 w 1 1316.0 (+39.08%) |
| `32xH100-400G-8gph-k10` | S 32 w 1 1316.0 | S 32 w 1 1274.4 (+110.48%) | S 16 w 2 zero1 + native 1230.3 (+6.96%) | S 32 w 1 1316.0 (+105.2%) |
| `32xH100-400G-8gph-k2` | S 16 w 2 pieces + native 1155.4 | S 16 w 2 pieces + native 1135.3 (+6.6%) | S 16 w 2 pieces + native 1155.4 (+6.48%) | S 16 w 2 pieces + native 1155.4 (+6.71%) |
| `32xH100-100G-1gph-k10` | S 32 w 1 1440.8 | S 32 w 1 1274.4 (+87.61%) | S 32 w 1 1440.8 (+71.53%) | S 32 w 1 1440.8 (+77.86%) |
| `32xH100-100G-8gph-k10` | S 32 w 1 1440.8 | S 32 w 1 1274.4 (+145.08%) | S 32 w 1 1440.8 (+71.53%) | S 32 w 1 1440.8 (+125.91%) |
| `128xH100-400G-1gph-k10` | S 32 w 4 zero1 + native 2270.2 | S 32 w 4 zero1 + native 2228.6 (+3.97%) | S 32 w 4 zero1 + native 2209.6 (+6.11%) | S 32 w 2 pieces + native 2358.7 (+0.88%) |
| `128xH100-400G-1gph-k6` | S 32 w 4 pieces + native 1338.1 | S 32 w 4 pieces + native 1296.5 (+67.21%) | S 32 w 4 pieces + native 1338.1 (+65.12%) | S 32 w 4 pieces + native 1338.1 (+75.05%) |
| `128xH100-400G-1gph-k2` | S 32 w 4 pieces + native 1338.1 | S 32 w 4 pieces + native 1296.5 (+67.21%) | S 32 w 4 pieces + native 1338.1 (+65.12%) | S 32 w 4 pieces + native 1338.1 (+75.05%) |
| `128xH100-400G-8gph-k10` | S 32 w 1 4365.4 | S 32 w 1 4323.8 (+21.76%) | S 32 w 4 zero1 + native 2209.6 (+12.19%) | S 32 w 1 4365.4 (+21.54%) |
| `128xH100-400G-8gph-k2` | S 32 w 4 zero1 + native 2245.9 | S 32 w 4 zero1 + native 2213.2 (+4.05%) | S 32 w 4 zero1 + native 2209.6 (+6.11%) | S 32 w 4 zero1 + native 2344.0 (+0.02%) |
| `128xH100-100G-1gph-k10` | S 32 w 1 4490.3 | S 32 w 1 4323.8 (+28.19%) | S 32 w 1 4490.3 (+27.14%) | S 32 w 1 4490.3 (+34.91%) |
| `128xH100-100G-8gph-k10` | S 32 w 1 4490.3 | S 32 w 1 4323.8 (+75.77%) | S 32 w 1 4490.3 (+27.14%) | S 32 w 1 4490.3 (+73.88%) |
| `32xH100-2sites-10G-uplink` | S 16 w 2 zero1 + native 1388.2 | S 16 w 2 zero1 + native 1369.4 (+0.64%) | S 16 w 2 pieces + native 1311.5 (+5.71%) | S 16 w 2 zero1 + native 1389.1 (+2.11%) |
| `32xGPU-mixed-classes` | S 32 w 1 1316.0 | S 32 w 1 1274.4 (+None%) | S 32 w 1 1316.0 (+None%) | S 32 w 1 1316.0 (+None%) |
| `32xH100-400G-1gph-k10-memhost20` | S 16 w 2 zero1 + native 1232.1 | S 16 w 2 zero1 + native 1212.0 (+5.15%) | S 16 w 2 zero1 + native 1230.3 (+6.96%) | S 16 w 2 zero1 + native 1233.0 (+6.73%) |

**Where the as-built code departs from these pins** (read from the code, file:line; the numbers are this PLAN's
diagnostics, not measurements of the as-built planner):
- `planner._stage_units` gives the Embed one stride index (0), so under `zero1` the whole Embed unit sits on peer 0. The
  runtime shards whole tensors `params[peer::w]` (`runtime/stage.py:183`): `tok` on peer 0, `pos` on peer 1. The
  `embed_p0` diagnostic above prices that layout.
- `planner._price` charges `admission.mem_host` on `zero1` candidates; the SPEC charges MEM-HOST on pieces only (task 4:
  today's device admission for `zero1`). The class check on `zero1` (`admission.check_class(..., probed=False)`) is kept
  in this PLAN as granted; whether task 4's "today's admission" includes it is a question for the SPEC's owner.
- `exchange_sim.simulate_exchange` returns `exposure = max(tail, late)`, where `late` is already measured against the
  slot (`step + first_f`); `planner._price` then takes `exposure − window`, so for a delay-bound stage with
  `window_s` > 0 the window is subtracted twice.
- `exchange_sim._Stage._complete` releases a unit's gathers at `max(arrivals) + one fold + AdamW`; I17 releases them
  after the owner's left fold of every contribution in canonical order, then AdamW.
