# m2b-e4: E4, exposed exchange at the scaled B′ (ADR-0006 F3, F5, F6)

Experiment plan for `docs/design/adr/ADR-0006-exact-fsdp-placement.md` (granted), handout precondition P3
(`docs/design/HANDOUT-M2b.md`). Hardware switch: gpu false. Local only; no cloud run (`"cloud": null`). The predictions
are pinned here, before the code that is judged against them exists: task `e4-exposure` (`docs/design/PLAN-M2b.json`)
writes the harness (`src/wantrain/validation/exposure.py`) and the `e4_` tests in
`tests/integration/test_engine_bitwise.py`. **None of the three local commands below exists yet**; they are what that
task must make pass. Its harness reads this file's `PLAN.json` and refuses to run off the configuration pinned here.

Labels: MEASURED (a command ran on this machine, output cited), DERIVED (the model's arithmetic at measured inputs),
SIMULATED (what E4 will report: wall time on localhost processes and a pacing shim), HYPOTHESIS (not yet tested).

## 1. Hypothesis

At the scaled B′, on the shim with shared per-peer pacers (SPEC-transport task 3):
- **F6 (primary).** Per-tensor pieces (forward emission, need-first queues, per-unit updates, no step barrier) expose
  less exchange than step 1 alone and than LPT whole tensors, at every F6 arm, by at least half of the margin pinned in
  §4: stage 0 against step 1 alone and LPT, interior w 4 against step 1 alone, interior w 8 against LPT.
- **F3.** Measured exposure lies within [0.8, 1.2]× the as-built SCHEDULER-I17 simulation at ≥ 3 points on each side of
  S1, at the S1 threshold (within 25%) and on the shared-uplink arm. Control: the barrier-kept pieces arm matches the
  simulation with its barrier flag on; otherwise the shim, not H3, is wrong.
- **F5.** A test-only residue-group tree exchange exposes more than the streaming left fold while S1 holds.
- Every training arm's state is byte-identical (sha256) to `FACTORY_REFERENCE=1` (F1's oracle; never a tolerance).

## 2. Profiling evidence (MEASURED, Apple M4, arm64, 10 cores, 16 GB; Python 3.12.13 under uv, torch 2.14.1)

The rate is that of the recipe's deterministic kernels at the scaled shapes. `build_layers` builds `[Embed, Block, Head]`
with `SplitLinear` (`wgrad.convert`); `setup_determinism` sets 1 intra-op thread and deterministic algorithms; the
deferred W runs through `wgrad.capture`/`flush`. That is the stage's own code path. Shapes: h′ 256, V′ 2000, seq′ 256,
`T_mb′` 256 (1 × 256 tokens), 4 heads. Medians of 9 runs after 3 warmups (7 per peer in the concurrent runs).

```
uv run python docs/experiments/m2b-e4/derive/profile_fprime.py 9   > derive/profile_fprime.out   (exit 0)
  bf16 block: F 16.80, B 138.33 (incl. its forward), W 13.53 ms -> c 168.66 ms, F'_eff 9.549 GFLOP/s; pure GEMMs 8.842
  fp32 block: F 1.17, B 2.90, W 0.32 ms -> c 4.39 ms, F'_eff 366.6 GFLOP/s; pure GEMMs 1450.9
  Head (bf16) F+B+W 93.89 ms (0.557 x the block; the model's V/(12h) is 0.651); Embed F+B 0.61 ms (0.4% of c)
  fold (one w-4 piece of stage 0, 340,992 elements): bf16 widen + fp32 add_ 81.0 GB/s (18 B/elem convention),
  fp32 add_ 129.2 GB/s (12 B/elem), AdamW step 42.7 GB/s (30 B/elem)
cd derive && uv run --project <repo> python profile_concurrent.py <w> 7 > conc_w<w>.out   (exit 0 each; bf16 block)
  w 1: c 167.6 ms; w 4: slowest peer c 192.73 ms (skew 1.02) -> F' 8.357 GFLOP/s, split F:B:W 0.1022:0.8145:0.0833
  w 8: slowest peer c 269.49 ms (skew 1.05) -> F' 5.976 GFLOP/s, split F:B:W 0.1065:0.8122:0.0813
```

E4 runs the bf16 recipe, because the kept ratio "fp32 share of stage-0 gradient bytes 0.59" needs bf16 GEMM gradients.
F′ is therefore the bf16 rate under w concurrent peers: **F′ = 8.357 GFLOP/s for w-4 arms and 5.976 for w-8 arms**
(`F′ = 8·P^F·T_mb′/c`, `P^F` = 12h′² = 786,432). MEASURED, and the cause of a finding below: the CPU's bf16 split is
B-heavy because the dX GEMM (`g.matmul(weight)`, `wgrad.py`) takes `aten::mm` 28.6 ms per call against 2.6 ms for the
forward `addmm` of the same shape (torch profiler, 3 backward passes). The ADR's model assumes c_F : c_B : c_W = 1:2:1.

## 3. The model and its scaled inputs (DERIVED)

The model is round 4's (`round4.md` §9), loaded exactly as `round5.md` §8 loads it: `exch4` and `stage4` (per-unit
need-first queues, ≤ f chunks, egress and ingress shared max-min fairly, gathers after the owner's fold and AdamW,
`E = max(X_s, V/cap + RTT + t_gap − C)`, `t_gap = c_B + c_W/U`). `derive/r6_e4.py` changes only what was measured:
(a) the op split, from 1:2:1 to the measured fractions; (b) the fold and AdamW rates, from HBM′ to the CPU's measured
rates; (c) two switches round 4 lacked: a shared inter-site uplink resource inside `exch4` (round 3's `exch3` resource
model) and the F5 tree arm; (d) the FIFO policy passed through to `exch4`. Step (0) of the script asserts that, with
the 1:2:1 split and HBM′ fold, it reproduces `round5.md` §6 exactly at full B and at B′ (F′ 50): 2.82 / 71.31 / 20.09,
3.66 / 16.00, 10.03 / 25.10 (reproduced).

The scaling map (`round5.md` §6), at w 4 (w 8 in brackets), with `s_rate = (F′/F)·(T_mb/T_mb′)`:
- `c_block′ = 8 · 786,432 · 256 / 8.357e9` = **192.73 ms** [269.49]; stage 0 (Embed + 1 block) has the same c_s.
- `cap′ = 50 GB/s · s_rate = 50e9 · (8.357e9 / 400e12) · 16` = **16.714 MB/s = 133.7 Mbit/s** [11.953 MB/s].
- `PCIe′` = cap′ (Gen5 = 400 Gbps at full B), so `cap = min(NIC′, PCIe′)` = cap′. Hop latency `2f′/PCIe′` =
  2 · 65,536 / 16.714e6 = **7.84 ms** [10.97]. The shim adds it to every message's one-way delay: `RTT/2 + 2f′/PCIe′`.
- Chunk `f′ = 16 MiB · (256/4096)²` = **65,536 B**.
- HBM′ (model) = 3.35 TB/s · s_rate = 1.120 GB/s [0.801]. The CPU folds at 81.0 GB/s, 72× faster. Pacing it down would
  need a fold-pacing seam in `stage.py`, which `e4-exposure` does not own, so the fold is **not paced**. The model is
  pinned at the measured rates instead (round 5 §6: "the PLAN re-runs this model at the measured HBM′ and pins that
  output"). Effect on the F6 arms: ≤ 0.27 points (sensitivity, §6).
- Stage-0 period: `C_B′ = 32 · (1 + V′/(12h′)) · c_block′` = 32 · 1.65104 · 192.73 = **10,182.3 ms**. The stage-0 arm
  pads each step to it, which keeps `C_B/c_s0` = 52.83 as at B.
- RTT: 0, and `RTT_hi = 2·c_W` (> c_W/4): **128.4 ms** at interior w 4 (`c_W` = 0.0833 · 4 · 192.73 = 64.2 ms), 32.1 ms at
  stage 0.
- Shared uplink: interior w 4, sites {0, 1} × 2 peers each, one uplink per direction at cap′, shared max-min by every
  cross-site flow (each peer's NIC carries 3 flows per direction, the uplink 4). No boundary bytes on the uplink.
  Uplink utilization 0.488 (S1 holds there).
- Tree (F5): a residue-group fold, G = w. Each peer folds its own M_k microbatches locally in fp32. It then sends one
  fp32 partial per unit piece to that piece's owner, released as the last W produces the unit (forward emission). Owners
  add w−1 partials, then gather. Nothing streams before the last W. In the model: `exch4` with b_g = 4 on every unit,
  E = X (the whole exchange is the tail). It is test-only. It moves bytes between w processes and never touches
  training state (a different oracle).

Ratios kept, full B against B′ (script output (7); tolerance ±0.5%): V/h 7.8125, seq/h 1, T_mb/h 1 (exact); Embed
share of stage 0 0.4234 → 0.4224 (−0.23%); tok 0.3753 → 0.3745 (−0.23%); fp32 share of stage-0 gradient bytes
0.5949 → 0.5939 (−0.16%; this count includes the block's 13h biases and norms, so it reads 0.0010 above round 5's
0.5929); NIC utilization at c_s 0.9448 → 0.9448; M/w 32; chunk time / c 0.0203; hop latency / c 0.0407; C_B / c_s0
52.83. Two ratios are **not** kept: the op split (measured, §2) and the fold time / c, which is 0.0284 at B and 0.0004
here (unpaced). Scale invariance at the measured split is checked (script output (5)): full B and B′ give identical
percentages for every arm.

## 4. Pinned predictions (DERIVED; SIMULATED when measured)

The exposure is in ms at the pinned F′ and in % of each arm's C. The percentages are scale-free: a harness whose in-situ
F′ differs sets cap′, PCIe′ and C_B′ from its own F′ by the map in §3, and the percentages stand. The F6 threshold is half
the predicted margin; it refutes pieces if their measured margin falls below it.

| arm | E ms | % of C | C ms | NIC util | F6: pieces must beat it by ≥ | note |
|---|---|---|---|---|---|---|
| `stage0.today` | 12625.05 | 123.990 | 10182.3 | 3.657 | — |  |
| `stage0.step1` | 7316.01 | 71.850 | 10182.3 | 2.803 | 34.280 pts / 3490.5 ms (vs stage0.pieces_fwd) |  |
| `stage0.lpt` | 2101.08 | 20.635 | 10182.3 | 1.948 | 8.673 pts / 883.0 ms (vs stage0.pieces_fwd) |  |
| `stage0.pieces_fwd` | 335.04 | 3.290 | 10182.3 | 0.945 | — |  |
| `stage0.pieces_bwd` | 335.04 | 3.290 | 10182.3 | 0.945 | — |  |
| `stage0.fifo_pieces_fwd` | 327.14 | 3.213 | 10182.3 | 0.945 | — |  |
| `stage0.fifo_pieces_bwd` | 340.27 | 3.342 | 10182.3 | 0.945 | — |  |
| `stage0.barrier_pieces_fwd` | 335.09 | 3.291 | 10182.3 | 0.945 | — |  |
| `interior_w4.today` | 5443.52 | 88.266 | 6167.2 | 1.729 | — |  |
| `interior_w4.step1` | 1200.97 | 19.473 | 6167.2 | 0.875 | 6.105 pts / 376.5 ms (vs interior_w4.pieces_fwd) |  |
| `interior_w4.lpt` | 573.97 | 9.307 | 6167.2 | 0.387 | — |  |
| `interior_w4.pieces_fwd` | 447.95 | 7.263 | 6167.2 | 0.387 | — |  |
| `interior_w4.pieces_bwd` | 441.39 | 7.157 | 6167.2 | 0.387 | — |  |
| `interior_w4.fifo_pieces_fwd` | 440.05 | 7.135 | 6167.2 | 0.387 | — |  |
| `interior_w4.fifo_pieces_bwd` | 513.60 | 8.328 | 6167.2 | 0.387 | — |  |
| `interior_w4.barrier_pieces_fwd` | 447.95 | 7.263 | 6167.2 | 0.387 | — |  |
| `interior_w8.today` | 5999.54 | 139.140 | 4311.9 | 2.014 | — |  |
| `interior_w8.step1` | 2034.62 | 47.187 | 4311.9 | 1.017 | — |  |
| `interior_w8.lpt` | 1155.29 | 26.793 | 4311.9 | 0.519 | 4.761 pts / 205.3 ms (vs interior_w8.pieces_fwd) |  |
| `interior_w8.pieces_fwd` | 744.64 | 17.270 | 4311.9 | 0.448 | — |  |
| `interior_w8.pieces_bwd` | 746.47 | 17.312 | 4311.9 | 0.448 | — |  |
| `interior_w8.fifo_pieces_fwd` | 744.64 | 17.270 | 4311.9 | 0.448 | — |  |
| `interior_w8.fifo_pieces_bwd` | 851.75 | 19.754 | 4311.9 | 0.448 | — |  |
| `interior_w8.barrier_pieces_fwd` | 755.64 | 17.525 | 4311.9 | 0.448 | — |  |
| `interior_w4.emission_fwd.rtt0` | 447.95 | 7.263 | 6167.2 | 0.387 | — | RTT 0.0 ms |
| `interior_w4.emission_bwd.rtt0` | 441.39 | 7.157 | 6167.2 | 0.387 | — | RTT 0.0 ms |
| `interior_w4.emission_fwd.rtt_2cw` | 505.82 | 8.202 | 6167.2 | 0.387 | — | RTT 128.4 ms |
| `interior_w4.emission_bwd.rtt_2cw` | 565.98 | 9.177 | 6167.2 | 0.387 | — | RTT 128.4 ms |
| `stage0.emission_fwd.rtt0` | 335.04 | 3.290 | 10182.3 | 0.945 | — | RTT 0.0 ms |
| `stage0.emission_bwd.rtt0` | 335.04 | 3.290 | 10182.3 | 0.945 | — | RTT 0.0 ms |
| `stage0.emission_fwd.rtt_2cw` | 343.63 | 3.375 | 10182.3 | 0.945 | — | RTT 32.1 ms |
| `stage0.emission_bwd.rtt_2cw` | 343.63 | 3.375 | 10182.3 | 0.945 | — | RTT 32.1 ms |
| `interior_w4.uplink_pieces_fwd` | 636.16 | 10.315 | 6167.2 | 0.387 | — | uplink util 0.488 |
| `interior_w4.uplink_pieces_bwd` | 629.60 | 10.209 | 6167.2 | 0.387 | — | uplink util 0.488 |
| `interior_w4.tree` | 730.26 | 11.841 | 6167.2 | — | — | left fold 447.9 ms (F5 primary) |
| `stage0.tree` | 405.61 | 3.983 | 10182.3 | — | — | left fold 335.0 ms |
| `sweep.util_0.5` | 613.66 | 9.950 | 6167.2 | 0.500 | — | cap′ 12.9219 MB/s |
| `sweep.util_0.7` | 905.79 | 14.687 | 6167.2 | 0.700 | — | cap′ 9.2299 MB/s |
| `sweep.util_0.9` | 1197.92 | 19.424 | 6167.2 | 0.900 | — | cap′ 7.1788 MB/s |
| `sweep.util_1.1` | 2052.00 | 33.273 | 6167.2 | 1.100 | — | cap′ 5.8736 MB/s |
| `sweep.util_1.3` | 3431.51 | 55.641 | 6167.2 | 1.300 | — | cap′ 4.9700 MB/s |
| `sweep.util_1.6` | 5500.77 | 89.194 | 6167.2 | 1.600 | — | cap′ 4.0381 MB/s |
| `sweep.util_2.0` | 8259.78 | 133.931 | 6167.2 | 2.000 | — | cap′ 3.2305 MB/s |

Arm meanings follow ADR-0006 and round 4:
- `today`: stride ZeRO-1, fp32 wire, whole-stage flush, barrier, backward order.
- `step1`: stride, native wire, per-module flush, barrier.
- `lpt`: whole tensors placed by LPT as explicit bounds, per-unit updates.
- `pieces_*`: forward or backward emission.
- `fifo_*`: `WANTRAIN_TEST_QUEUE=fifo`.
- `barrier_*`: `WANTRAIN_TEST_BARRIER=always`.

Stage 0 is Embed + 1 block, w 4, M/w 32, padded to C_B′. The interior arms have 4 blocks, w 4 at M/w 8 and w 8 at M/w 4.

The sweep runs on `interior_w4.pieces_fwd` with cap′ varied (PCIe′ not binding, the hop latency held at 7.84 ms).
- S1 sits at cap′ = 6.4609 MB/s (utilization 1).
- Below S1 the two-line fit is `E% = −1.892 + 153.022/cap′`; above it, `E% = −89.754 + 722.605/cap′` (cap′ in MB/s).
- The lines meet at **cap′ 6.4827 MB/s (utilization 0.9966)**. That is the predicted threshold.
- The same rule is applied to the measured points.

## 5. Falsification criteria, restated with these numbers

- **F6 (primary; refutes pieces on time).** Measured margins are computed in % of C, as E(alternative) − E(pieces_fwd).
  F6 fires if any arm's margin is below its threshold:

  | arm | alternative | threshold (points) |
  |---|---|---|
  | stage 0 | step 1 alone | 34.280 |
  | stage 0 | LPT | 8.673 |
  | interior w 4 | step 1 alone | 6.105 |
  | interior w 8 | LPT | 4.761 |

  No memory argument rescues a failed arm. A run off this configuration is not evidence (§7).
- **F3.** F3 fires on any of:
  - an arm outside [0.8, 1.2]× the I17 simulation at the in-situ configuration. This covers the 3 sweep points below S1,
    the 4 above, both uplink arms and the emission arms;
  - a measured sweep threshold more than 25% from the simulated one (predicted 6.4827 MB/s);
  - the barrier-kept control outside [0.8, 1.2]× the simulation with its barrier flag on (predicted 3.291% at stage 0,
    7.263% at interior w 4, 17.525% at interior w 8). A control failure means the shim is wrong, not H3.
  The I17 simulation is the judge F3 names. This PLAN's table is its anchor: the simulation at this configuration must
  reproduce every arm above within 5%, and at the round-5 configuration `round5.md` §6's percentages within 5%.
- **F5.** F5 fires if `interior_w4.tree` measures ≤ `interior_w4.pieces_fwd` (predicted 11.841% vs 7.263%; S1 holds,
  utilization 0.387). `stage0.tree` (3.983% vs 3.290%) is recorded but does not decide: its predicted gap of 0.69
  points is inside F3's band.
- **Emission order (BA-43).** At `rtt_2cw` the simulation picks forward, 8.202% against 9.177% (11.9% apart); the
  measured faster order must agree. At RTT 0 the predicted gap is 1.5% (backward faster, 7.157 vs 7.263), below what
  the band resolves; both orders must sit within F3's band. At stage 0 the orders are predicted equal (round 5 §5).
- **Correctness.** Any sha256 difference from `FACTORY_REFERENCE=1` in a training arm is a bug (F1), never a tolerance.

`PLAN.json` encodes these as classification metrics. The harness's last JSON line must carry exactly these keys:
- `f6_margin_fraction_min` (primary): the minimum, over the four F6 arms, of measured margin ÷ predicted margin.
  Expected 1.0; with `tolerancePct` 50 it is met iff ≥ 0.5, which is exactly F6.
- the four per-arm F6 fractions.
- `f3_violations`, `f5_violations`, `emission_order_violations`: counts with expected 0. Their slack is 0 at any
  tolerance.

The 50% tolerance is F6's "half the predicted margin". It applies to no exposure value: F3's [0.8, 1.2] band is
enforced by the tests and counted in `f3_violations`.

## 6. What the numbers rest on (HYPOTHESIS until E4 runs)

- The model's tail and volume terms (round 2–4). The I17 simulation, which F3 tests, replaces them as the judge.
- The measured op split. If the in-situ F or W fraction differs from the pinned one by more than 25% (relative), the
  run is off this configuration. Sensitivity at W ×0.75 / ×1.25: interior w 4 pieces 7.47 / 7.07%, w 8 17.67 / 16.89%,
  stage 0 3.32 / 3.26%.
- Other sensitivities:
  - with the round-5 split 1:2:1: stage 0 2.79, interior w 4 3.62 vs 15.83, interior w 8 10.03 vs 25.79;
  - with the fold paced to HBM′: ≤ 0.27 points;
  - with no hop latency: −0.15 to −0.51 points.
- The shim's pacers share each egress, ingress and uplink max-min fairly among chunks in flight (SPEC-transport task 3),
  as the model does. A kernel-TCP incast arm is a follow-up (round 5 §7).
- Wall time (HYPOTHESIS): about 25–40 minutes for every arm at 1 warm-up and 3 measured steps, inside the GATES
  entry's 7200 s.

## 7. Local commands (written by task `e4-exposure`; none exists yet)

- correctness: `uv run pytest -q -p no:randomly tests/integration/test_engine_bitwise.py -k "e4_ and identity"`. Exit 0
  only if every training arm's manifest sha256 equals the reference.
- performance: `uv run pytest -q -p no:randomly -s tests/integration/test_engine_bitwise.py -k e4_`. The last line is
  one JSON object with the keys in §5, labelled SIMULATED. It is preceded by one JSON line per arm: arm, predicted,
  measured, unit, criterion.
- failure: `uv run pytest -q -p no:randomly tests/integration/test_engine_bitwise.py -k "e4_ and (fifo or barrier)"`.
  These are the FIFO and barrier-kept controls.

**Configuration guard.** The harness refuses to run, naming the field, unless all of the following hold:
- shapes h′ 256, V′ 2000, seq′ 256, `T_mb′` 256, 4 heads, the bf16 recipe, 1 intra-op thread;
- w and M/w per arm, as in §4;
- every §3 ratio within ±0.5%;
- the in-situ split within 25% of `PLAN.json` `scaled.op_split_F_B_W`;
- cap′ = PCIe′ = 2.0e−3 B/s per FLOP/s × the in-situ F′;
- f′ 65,536 B, one-way delay `RTT/2 + 2f′/PCIe′`, `C_B′ = 52.8333 × c_block`, `RTT_hi = 2 c_W`, cap_up′ = cap′;
- the fold not paced.

The verify fails until a human grants this PLAN (`approve.mjs check docs/experiments/m2b-e4/PLAN.json`).

## 8. Reproduce

```
cd docs/experiments/m2b-e4/derive
python3 -I r6_e4.py r4.py r3.py r2.py ../../../../src/wantrain/scheduler/zero_bubble.py \
    profile_fprime.out conc_w4.out conc_w8.out > r6_e4.out      # exit 0, 0.6 s, Python 3.14.4
python3 -I make_plan.py r6_e4.out ../PLAN.json
```

`r2.py`, `r3.py`, `r4.py` and `r5.py` are extracted with the commands `round4.md` §9 and `round3.md` §7 give. Their
sha256 match the recorded ones: f43ab4f6…b701eb, acf1cb86…995ed6, d4838233…dd0c0 and cc425fde…e37d8; zero_bubble.py is
22988afc…751ac8d. `r5.py` re-run gives 5ac4f609…250b4, round 5's output. The hashes of the new files are in the
caller's report and are recomputed by `shasum -a 256 derive/*`.
