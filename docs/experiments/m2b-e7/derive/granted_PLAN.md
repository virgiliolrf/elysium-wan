# m2b-e7: the planner's choice per fleet (ADR-0006 E7, F9), re-pinned at the planner's costs

ADR: `docs/design/adr/ADR-0006-exact-fsdp-placement.md` (granted). Handout precondition P3 (`docs/design/HANDOUT-M2b.md`),
ADR R16: "E7's predictions use 8 FLOPs per param; the planner's `layer_costs` counts attention (Head/block 0.558 vs
0.651), so E7's PLAN re-pins them with the planner's costs before it runs." Task `plan-wide-e7`
(`docs/design/PLAN-M2b.json`) reads every configuration's expected choice from `PLAN.json` beside this file. Hardware
switch: gpu false. Cloud: none (`"cloud": null`). Every number below is DERIVED by `derive/e7.py` on this machine's CPU;
every input is HYPOTHESIS, as in rounds 3–5.

## 1. Hypothesis

On each pinned fleet, `plan_job` (SCHEDULER task 4) chooses the plan pinned in `PLAN.json`. It applies SCHEDULER-I16 as
written: refuse pieces on S1 (NIC, host DRAM, uplink), S2, class or MEM-HOST, with ε 3% (ADR question 1, assumed) and
k counted (R10 and R13 deferred). The rationale names every refused candidate with its link, number, class or
MEM-HOST. The planner never picks a plan whose simulated step is more than 1% above the best admitted candidate's.

## 2. What changed against round 5: the costs

FACTS (code read on this tree):
- `partition.layer_costs` (`partition.py:164-175`) prices `[Embed 3h, Block h(72h + 12s) × n, Head 6hV]` per token.
  This is F + B + W from the zero-bubble table, with attention included.
- `zero_bubble.analytic_costs` (`zero_bubble.py:468-483`) gives per-layer F, B, W from the same table. Recompute adds
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

| config (`id`) | expected choice | step ms | runner-up, ms | margin % | refused (kinds) | ADR choice, round-5 ms | changed |
|---|---|---|---|---|---|---|---|
| 32xH100-400G-1gph-k10 | S 16, w 2, pieces + native | 1136.3 | S 16, w 2, zero1 + native, 1163.7 | 2.41 | 2 (S2) | same, 966.8 | no |
| 32xH100-400G-2gph-k10 | S 16, w 2, zero1 + native | 1163.7 | **S 8, w 4, zero1 + native, 1176.0** | **1.06** | 3 (S1 host DRAM; S2) | same, 994.4 | no |
| 32xH100-400G-4gph-k10 | S 32, w 1 | 1275.4 | S 16, w 2, zero1 + native, 1355.2 | 6.26 | 3 (S1 host DRAM; S2) | same, 1102.2 | no |
| 32xH100-400G-8gph-k10 | S 32, w 1 | 1275.4 | S 16, w 2, zero1 + native, 2411.0 | 89.04 | 3 (S1 host DRAM; S2) | same, 1102.2 | no |
| 32xH100-400G-8gph-k2 | S 16, w 2, pieces + native | 1136.3 | S 16, w 2, zero1 + native, 1163.7 | 2.41 | 2 (S1 host DRAM; S2) | same, 966.8 | no |
| 32xH100-100G-1gph-k10 | S 32, w 1 | 1275.4 | S 16, w 2, zero1 + native, 2107.8 | 65.27 | 3 (S1 NIC; S2) | same, 1102.2 | no |
| 32xH100-100G-8gph-k10 | S 32, w 1 | 1275.4 | S 16, w 2, zero1 + native, 2411.0 | 89.04 | 3 (S1 NIC; S1 host DRAM; S2) | same, 1102.2 | no |
| 128xH100-400G-1gph-k10 | S 32, w 4, zero1 + native | 1808.6 | S 16, w 8, zero1 + native, 2079.9 | 15.0 | 2 (S1 host DRAM 809.8 GB/s at stage 0; S2) | same, 1721.3 | no |
| 128xH100-400G-1gph-k6 | S 32, w 4, pieces + native | 1297.2 | S 32, w 4, zero1 + native, 1808.6 | 39.42 | 1 (S2: S 16, w 8) | same, 1124.5 | no |
| 128xH100-400G-1gph-k2 | S 32, w 4, pieces + native | 1297.2 | S 32, w 4, zero1 + native, 1808.6 | 39.42 | 1 (S2) | same (k ≤ 6), 1124.5 | no |
| 128xH100-400G-8gph-k10 | S 32, w 1 on 32 GPUs | 4324.8 | S 32, w 4, zero1 + native, 5038.6 | 16.5 | 2 (S1 host DRAM; S2) | same, 3887.8 | no |
| 128xH100-400G-8gph-k2 | S 32, w 4, zero1 + native | 1808.6 | S 16, w 8, zero1 + native, 2079.9 | 15.0 | 2 (S1 host DRAM 954.8 GB/s, stages 0–1; S2) | same, 1721.3 | no |
| 128xH100-100G-1gph-k10 | S 32, w 1 on 32 GPUs | 4324.8 | S 32, w 4, zero1 + native, 6259.9 | 44.74 | 2 (S1 NIC; S1 host DRAM; S2) | same, 3887.8 | no |
| 128xH100-100G-8gph-k10 | S 32, w 1 on 32 GPUs | 4324.8 | S 32, w 4, zero1 + native, 6259.9 | 44.74 | 2 (S1 NIC; S1 host DRAM; S2) | same, 3887.8 | no |
| 32xH100-2sites-10G-uplink | S 16, w 2, pieces + native (no stage straddles) | 1293.7 | S 16, w 2, zero1 + native, 1321.1 | 2.12 | 5 (S1 uplink 175.5 GB/s > 1.25, ×140, every straddling width; S2) | refusal only | no |
| 32xGPU-mixed-classes | S 32, w 1 | 1275.4 | — | — | 6 (class: the stage holding GPUs 16 and 17) | refusal only | no |
| 32xH100-400G-1gph-k10-memhost20 | S 32, w 1 | 1275.4 | — | — | 6 (MEM-HOST: ≥ 23.74 GB > 20 GB; S2) | refusal only | no |

**No choice changed.** What changed are the step times (+5% to +18%; `zero1` + native at point B +5%, the rest +11% to +18%), two margins and one runner-up:
- **(16, 2) margin:** pieces now beat `zero1` + native by 2.41% (round 5: 2.85%; ADR text: 2.8%), and depth by 12.2%.
- **2-GPU hosts at k 10:** the runner-up is now `zero1` + native at (8, 4), 1.06% behind. In round 5 it was S 32,
  w 1, 10.8% behind.
- **Pieces at point B:** a 1-GPU host now admits stage 0 up to k ≤ 7.41 (round 5: k ≤ 6.35). The ADR's k 6 pin holds
  either way.

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

  Cross-check: (8, 4) interior at 21.74 equals round 5 §4's point A. Every wide candidate is refused, and S 32, w 1 is
  chosen.

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
is at most 2% (`tie: true`; today only `32xH100-400G-2gph-k10`, 1.06%), the runner-up also counts as matching if its
simulated step is within 1% of the best admitted candidate's (F9's second clause). Metrics:
- `choices_matching` (primary): expected 17.
- `refusals_matching`: expected 48.
- `f9_violations`: expected 0, lower is better.
- `tolerancePct`: 1, which is below one count at these totals, so the counts are exact.

## 5. Sensitivity (DERIVED; same script, `PLAN.json` pins the primary mode)

| arm | choices equal to the pinned ones (of the 14 homogeneous cells) | note |
|---|---|---|
| S1 window kept at `8·P^F·T_mb/F` (as task `i16-admission` pins it) | 14 / 14 | Only S1's numbers move (B stage 0 at k 10: 944.8 against 809.8 GB/s) |
| ADR split `c/4, c/2, c/4` with the planner's `c_s` | 14 / 14 | Steps 42.6–43.2 ms shorter; the 2-GPU margin is 1.49% |
| No recompute (`c_s = layer_costs·T_mb/F`) | 12 / 14 | 128 H100, 1-GPU hosts at k 6 and k 2 flip to `zero1` + native: S1 NIC at B stage 0 rises 0.81 → 1.08, and S2 fails (24.1 > 21.6 ms) |

So recompute on is a load-bearing input: the `e7_` test's recipe must set it, as the ADR's model does.

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
5. **MEM-HOST scope.** I16 states MEM-HOST for pieces. This PLAN charges it on every wide candidate, since `zero1` +
   native also stages exchange bytes in pinned host memory. If the spec exempts `zero1`, the memhost20 choice becomes
   S 16, w 2, `zero1` + native at 1163.7 ms. This is a question for the human.
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

`python3 -I docs/experiments/m2b-e7/derive/e7.py derive/r5.py derive/r4.py derive/r3.py derive/r2.py
src/wantrain/scheduler/zero_bubble.py src/wantrain/scheduler/partition.py derive/e7.json` (paths under
`docs/experiments/m2b-e7/`), then `derive/build_plan.py derive/e7.json PLAN.json <sha e7.py> <sha e7.out>`.
Python 3.14.4, exit 0, about 30 s, output identical over two runs.
- **Inputs extracted** as round 4 §9 does, verified against the recorded hashes:
  - `r2.py` `f43ab4f6…b701eb`
  - `r3.py` `acf1cb86…995ed6`
  - `r4.py` `d4838233…dd0c0`
  - `r5.py` `cc425fde…e37d8`
  - `zero_bubble.py` `22988afc…751ac8d`
  - `partition.py` `0279f6b2f9e190d9e506fd700091231c2fa9f06c902786d883b8a4859b9656d3`
- **Script:** `e7.py` sha256 `7e2b7729eb3d19256830c479d570d571a0f45e3727b35d93bdcfc381b25abbe5`.
- **Output:** `e7.out` sha256 `cdc4e496a96a7656fce285e238c1848f26a13b8e1de8f87286ac273cd317e920`; `e7.json`
  `cb8a7a8a83fb5ca3647483ff48e06fb3bce07d1bd52a790cff05decd8bd98557`.
