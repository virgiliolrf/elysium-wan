# m2b-e4: E4, exposed exchange at the scaled B′ (ADR-0006 F3, F5, F6)

Experiment plan for `docs/design/adr/ADR-0006-exact-fsdp-placement.md` (granted), handout precondition P3
(`docs/design/HANDOUT-M2b.md`). Hardware switch: gpu false. Local only; no cloud run (`"cloud": null`). The predictions
are pinned here, before the code that is judged against them exists: task `e4-exposure` (`docs/design/PLAN-M2b.json`)
writes the harness (`src/wantrain/validation/exposure.py`) and the `e4_` tests in
`tests/integration/test_engine_bitwise.py`. **None of the three local commands below exists yet**; they are what that
task must make pass. Its harness reads this file's `PLAN.json` and refuses to run off the configuration pinned here.

**Amended 2026-10-10 (SC-4 / SPEC SCHEDULER-I17, section 9):** every prediction below is re-derived by an independent SCHEDULER-I17 oracle; the granted values are kept in section 9 and in `PLAN.json` `amendments`.

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

**Model (amended, SC-4).** The predictions come from an independent SCHEDULER-I17 oracle written from the SPEC
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

**Stage-0 padding (pinned).** per peer q, its i-th owned microbatch (global index i*w + q, i = 0 .. M/w - 1) occupies the slot [i*tau, (i+1)*tau), tau = C_B'/(M/w); with cF, cB, cW = c_s times the pinned op split: F runs [i*tau, i*tau + cF], B runs [(i+1)*tau - cW - cB, (i+1)*tau - cW], W runs [(i+1)*tau - cW, (i+1)*tau] (the idle between F and B is the wait for the downstream backward); no F runs ahead of its slot. So the last W ends at C_B', the period is C_B', window_s = 0, and step t+1's first F is due at C_B'. A harness may realize it with a per-peer wait of tau - c_s between each F and its B (one stand-in per peer, never one shared by the peers, comm 0, no exchange), as long as the op times above result.

The scaling map (`round5.md` §6), at w 4 (w 8 in brackets), with `s_rate = (F′/F)·(T_mb/T_mb′)`:
- `c_block′ = 8 · 786,432 · 256 / 8.357e9` = **192.73 ms** [269.49]; stage 0 (Embed + 1 block) has the same c_s.
- `cap′ = 50 GB/s · s_rate = 50e9 · (8.357e9 / 400e12) · 16` = **16.714 MB/s = 133.7 Mbit/s** [11.953 MB/s].
- `PCIe′` = cap′ (Gen5 = 400 Gbps at full B), so `cap = min(NIC′, PCIe′)` = cap′. Hop latency `2f′/PCIe′` =
  2 · 65,536 / 16.714e6 = **7.84 ms** [10.97]. The shim adds it to every message's one-way delay: `RTT/2 + 2f′/PCIe′`.
- Chunk `f′ = 16 MiB · (256/4096)²` = **65,536 B**.
- HBM′ (model) = 3.35 TB/s · s_rate = 1.120 GB/s [0.801]. The CPU folds at 81.0 GB/s, 72× faster. Pacing it down would
  need a fold-pacing seam in `stage.py`, which `e4-exposure` does not own, so the fold is **not paced**. The model is
  pinned at the measured rates instead (round 5 §6: "the PLAN re-runs this model at the measured HBM′ and pins that
  output"). Effect on the F6 arms in this PLAN's model: ≤ 0.98 points (`e4_spec.out` (2)).
- Stage-0 period: `C_B′ = 32 · (1 + V′/(12h′)) · c_block′` = 32 · 1.65104 · 192.73 = **10,182.3 ms**. The stage-0 arm
  pads each step to it, which keeps `C_B/c_s0` = 52.83 as at B.
- RTT: 0, and `RTT_hi = 2·c_W` (> c_W/4): **128.4 ms** at interior w 4 (`c_W` = 0.0833 · 4 · 192.73 = 64.2 ms), 32.1 ms at
  stage 0.
- Shared uplink: interior w 4, sites {0, 1} × 2 peers each, one uplink per direction at cap′, shared max-min by every
  cross-site flow (each peer's NIC carries 3 flows per direction, the uplink 4). No boundary bytes on the uplink.
  Uplink utilization 0.488 (S1 holds there).
- Tree (F5): a residue-group fold, G = w. Each peer folds its own M_k microbatches locally in fp32. It then sends one
  fp32 partial per unit piece to that piece's owner, released as the last W produces the unit (forward emission). Owners
  add w−1 partials, then gather. Nothing streams before the last W. In the model (SC-4): one fp32 partial per unit piece per peer, released when that peer's last W produces the unit; the owner folds the w partials (12 B/element) and runs AdamW, then gathers. It is test-only. It moves bytes between w processes and never touches
  training state (a different oracle).

Ratios kept, full B against B′ (`r6_e4.out` (7); tolerance ±0.5%): V/h 7.8125, seq/h 1, T_mb/h 1 (exact); Embed
share of stage 0 0.4234 → 0.4224 (−0.23%); tok 0.3753 → 0.3745 (−0.23%); fp32 share of stage-0 gradient bytes
0.5949 → 0.5939 (−0.16%; this count includes the block's 13h biases and norms, so it reads 0.0010 above round 5's
0.5929); NIC utilization at c_s 0.9448 → 0.9448; M/w 32; chunk time / c 0.0203; hop latency / c 0.0407; C_B / c_s0
52.83. Two ratios are **not** kept: the op split (measured, §2) and the fold time / c, which is 0.0284 at B and 0.0004
here (unpaced). Scale invariance at the measured split is checked (`e4_spec.out` (3)): full B and B′ give identical
percentages for every arm.

## 4. Pinned predictions (DERIVED; SIMULATED when measured)

The exposure is in ms at the pinned F′ and in % of each arm's C. The percentages are scale-free: a harness whose in-situ
F′ differs sets cap′, PCIe′ and C_B′ from its own F′ by the map in §3, and the percentages stand. The F6 threshold is half
the predicted margin; it refutes pieces if their measured margin falls below it.

| arm | E ms | % of C | C ms | NIC util | F6: pieces must beat it by ≥ | note |
|---|---|---|---|---|---|---|
| `stage0.today` | 13381.42 | 131.418 | 10182.3 | 3.657 | — |  |
| `stage0.step1` | 7930.96 | 77.890 | 10182.3 | 2.803 | 37.300 pts / 3797.9 ms (vs stage0.pieces_fwd) |  |
| `stage0.lpt` | 2556.68 | 25.109 | 10182.3 | 1.948 | 10.909 pts / 1110.8 ms (vs stage0.pieces_fwd) |  |
| `stage0.pieces_fwd` | 335.08 | 3.291 | 10182.3 | 0.945 | — |  |
| `stage0.pieces_bwd` | 335.08 | 3.291 | 10182.3 | 0.945 | — |  |
| `stage0.fifo_pieces_fwd` | 327.14 | 3.213 | 10182.3 | 0.945 | — |  |
| `stage0.fifo_pieces_bwd` | 340.27 | 3.342 | 10182.3 | 0.945 | — |  |
| `stage0.barrier_pieces_fwd` | 335.08 | 3.291 | 10182.3 | 0.945 | — |  |
| `interior_w4.today` | 6514.35 | 105.629 | 6167.2 | 1.729 | — |  |
| `interior_w4.step1` | 1201.14 | 19.476 | 6167.2 | 0.875 | 6.106 pts / 376.6 ms (vs interior_w4.pieces_fwd) |  |
| `interior_w4.lpt` | 574.15 | 9.310 | 6167.2 | 0.387 | — |  |
| `interior_w4.pieces_fwd` | 448.00 | 7.264 | 6167.2 | 0.387 | — |  |
| `interior_w4.pieces_bwd` | 441.39 | 7.157 | 6167.2 | 0.387 | — |  |
| `interior_w4.fifo_pieces_fwd` | 440.05 | 7.135 | 6167.2 | 0.387 | — |  |
| `interior_w4.fifo_pieces_bwd` | 513.60 | 8.328 | 6167.2 | 0.387 | — |  |
| `interior_w4.barrier_pieces_fwd` | 448.00 | 7.264 | 6167.2 | 0.387 | — |  |
| `interior_w8.today` | 7523.96 | 174.494 | 4311.9 | 2.014 | — |  |
| `interior_w8.step1` | 2090.82 | 48.490 | 4311.9 | 1.017 | — |  |
| `interior_w8.lpt` | 1155.60 | 26.800 | 4311.9 | 0.519 | 4.765 pts / 205.5 ms (vs interior_w8.pieces_fwd) |  |
| `interior_w8.pieces_fwd` | 744.64 | 17.270 | 4311.9 | 0.448 | — |  |
| `interior_w8.pieces_bwd` | 746.47 | 17.312 | 4311.9 | 0.448 | — |  |
| `interior_w8.fifo_pieces_fwd` | 744.64 | 17.270 | 4311.9 | 0.448 | — |  |
| `interior_w8.fifo_pieces_bwd` | 851.75 | 19.754 | 4311.9 | 0.448 | — |  |
| `interior_w8.barrier_pieces_fwd` | 755.69 | 17.526 | 4311.9 | 0.448 | — |  |
| `interior_w4.emission_fwd.rtt0` | 448.00 | 7.264 | 6167.2 | 0.387 | — | RTT 0.0 ms |
| `interior_w4.emission_bwd.rtt0` | 441.39 | 7.157 | 6167.2 | 0.387 | — | RTT 0.0 ms |
| `interior_w4.emission_fwd.rtt_2cw` | 505.87 | 8.203 | 6167.2 | 0.387 | — | RTT 128.4 ms |
| `interior_w4.emission_bwd.rtt_2cw` | 566.02 | 9.178 | 6167.2 | 0.387 | — | RTT 128.4 ms |
| `stage0.emission_fwd.rtt0` | 335.08 | 3.291 | 10182.3 | 0.945 | — | RTT 0.0 ms |
| `stage0.emission_bwd.rtt0` | 335.08 | 3.291 | 10182.3 | 0.945 | — | RTT 0.0 ms |
| `stage0.emission_fwd.rtt_2cw` | 343.67 | 3.375 | 10182.3 | 0.945 | — | RTT 32.1 ms |
| `stage0.emission_bwd.rtt_2cw` | 343.67 | 3.375 | 10182.3 | 0.945 | — | RTT 32.1 ms |
| `interior_w4.uplink_pieces_fwd` | 636.21 | 10.316 | 6167.2 | 0.387 | — | uplink util 0.488 |
| `interior_w4.uplink_pieces_bwd` | 629.60 | 10.209 | 6167.2 | 0.387 | — | uplink util 0.488 |
| `interior_w4.tree` | 730.28 | 11.841 | 6167.2 | 0.387 | — | left fold 448.0 ms (F5 primary) |
| `stage0.tree` | 405.63 | 3.984 | 10182.3 | 0.945 | — | left fold 335.1 ms |
| `sweep.util_0.5` | 613.70 | 9.951 | 6167.2 | 0.500 | — | cap′ 12.9219 MB/s |
| `sweep.util_0.7` | 905.83 | 14.688 | 6167.2 | 0.700 | — | cap′ 9.2299 MB/s |
| `sweep.util_0.9` | 1197.96 | 19.425 | 6167.2 | 0.900 | — | cap′ 7.1788 MB/s |
| `sweep.util_1.1` | 2029.72 | 32.912 | 6167.2 | 1.100 | — | cap′ 5.8736 MB/s |
| `sweep.util_1.3` | 3401.11 | 55.148 | 6167.2 | 1.300 | — | cap′ 4.9700 MB/s |
| `sweep.util_1.6` | 5458.20 | 88.504 | 6167.2 | 1.600 | — | cap′ 4.0381 MB/s |
| `sweep.util_2.0` | 8200.98 | 132.977 | 6167.2 | 2.000 | — | cap′ 3.2305 MB/s |

Arm meanings follow ADR-0006 and round 4:
- `today`: the whole-stage-flush mode (`scaled.today_mode`): stride ZeRO-1 of whole tensors, fp32 wire, every unit's pieces released at W(mb)'s end, full step barrier.
- `step1`: stride, native wire, per-module flush, barrier.
- `lpt`: whole tensors placed by LPT as explicit bounds, per-unit updates.
- `pieces_*`: forward or backward emission.
- `fifo_*`: `WANTRAIN_TEST_QUEUE=fifo`.
- `barrier_*`: `WANTRAIN_TEST_BARRIER=always`.

Stage 0 is Embed + 1 block, w 4, M/w 32, padded to C_B′. The interior arms have 4 blocks, w 4 at M/w 8 and w 8 at M/w 4.

The sweep runs on `interior_w4.pieces_fwd` with cap′ varied (PCIe′ not binding, the hop latency held at 7.84 ms).
- S1 sits at cap′ = 6.4609 MB/s (utilization 1).
- Below S1 the two-line fit is `E% = -1.891 + 153.022/cap′`; above it, `E% = -89.391 + 718.354/cap′` (cap′ in MB/s).
- The lines meet at **cap′ 6.4609 MB/s (utilization 1.0)**. That is the predicted threshold.
- The same rule is applied to the measured points.

## 5. Falsification criteria, restated with these numbers

- **F6 (primary; refutes pieces on time).** Measured margins are computed in % of C, as E(alternative) − E(pieces_fwd).
  F6 fires if any arm's margin is below its threshold:

  | arm | alternative | threshold (points) |
  |---|---|---|
  | stage 0 | step 1 alone | 37.300 |
  | stage 0 | LPT | 10.909 |
  | interior w 4 | step 1 alone | 6.106 |
  | interior w 8 | LPT | 4.765 |

  No memory argument rescues a failed arm. A run off this configuration is not evidence (§7).
- **F3.** F3 fires on any of:
  - an arm outside [0.8, 1.2]× the I17 simulation at the in-situ configuration. This covers the 3 sweep points below S1,
    the 4 above, both uplink arms and the emission arms;
  - a measured sweep threshold more than 25% from the simulated one (predicted 6.4609 MB/s);
  - the barrier-kept control outside [0.8, 1.2]× the simulation with its barrier flag on (predicted 3.291% at stage 0,
    7.264% at interior w 4, 17.526% at interior w 8). A control failure means the shim is wrong, not H3.
  The I17 simulation is the judge F3 names. This PLAN's table is its anchor: the as-built simulation at this configuration must reproduce every arm above within 5%, the `today` arms included (whole-stage-flush mode), and at the round-5 configuration `PLAN.json` `round5_anchor.arms` within 5% (SC-4; it replaces `round5.md` §6's percentages as the anchor).
- **F5.** F5 fires if `interior_w4.tree` measures ≤ `interior_w4.pieces_fwd` (predicted 11.841% vs 7.264%; S1 holds,
  utilization 0.387). `stage0.tree` (3.984% vs 3.291%) is recorded but does not decide: its predicted gap of 0.69
  points is inside F3's band.
- **Emission order (BA-43).** At `rtt_2cw` the simulation picks forward, 8.203% against 9.178% (11.9% apart); the
  measured faster order must agree. At RTT 0 the predicted gap is 1.5% (backward faster, 7.157 vs 7.264), below what
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

- The SCHEDULER-I17 oracle's model (SC-4). The as-built I17 simulation, which F3 tests, is anchored to it.
- The measured op split. If the in-situ F or W fraction differs from the pinned one by more than 25% (relative), the run is off this configuration. Sensitivities (W ×0.75 / ×1.25; split 1:2:1; HBM′-paced fold) are printed by `e4_spec.py`, section (2); no sensitivity arm's predicted margin falls below its pinned threshold.
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
python3 -I -B e4_spec.py i17_oracle.py profile_fprime.out conc_w4.out conc_w8.out e4_spec.json > e4_spec.out   # exit 0
python3 -I -B make_plan_spec.py granted_PLAN.json e4_spec.json e4_spec.py e4_spec.out i17_oracle.py ../PLAN.json \
    e4_tables.md gen_plan_md.py
python3 -I -B gen_plan_md.py granted_PLAN.md ../PLAN.json e4_tables.md ../PLAN.md granted_PLAN.json e4_spec.out
```

`granted_PLAN.json` is the granted PLAN (sha256 7fb6e9ff4cb2…, the APPROVALS grant); `granted_PLAN.md` is
the PLAN.md it was granted with. Neither the oracle nor the scripts import or read `src/wantrain` (the human's rule,
2026-10-10). The outputs are byte-identical over two runs (`python3 -I -B`). Hashes: `e4_spec.py` 20d1f130ed78639617dbfbc4b6c56864e6f04e59c4748447b0a40f602c2998f4,
`i17_oracle.py` d01edde9a98972805034c98ada32bc813f3a4e4d02faddd0466a0232984472d7, `e4_spec.out` df7e5633677dc28595fb1fc5d479a1d5b99dc4321e2d872aea321b8a6c64e929, `e4_spec.json` 708f7c31030209bffa5276379fcda283df392a13b9539662d08855da2fb64197,
`make_plan_spec.py` 98772fd0137214dc74d92dabe25e4dac3c781e9265f46a5337fabcbfb0118d06, `gen_plan_md.py` 7fbe7925d75b2f8aa39e0416fb94088063495f457043d73fe42b9c63f5494ed1. The MEASURED inputs keep
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
- step 1 alone: 71.850 → 77.890%;
- LPT: 20.635 → 25.109%.

Their F6 thresholds grew to 37.300 and 10.909 points. The `today` arms are now expressible
and rose (`stage0.today` 123.990 → 131.418%, `interior_w4.today` 88.266 → 105.629%, `interior_w8.today` 139.140 → 174.494%), because their gathers wait behind the hot owner's ingress. `interior_w8.step1` (link-bound, utilization 1.017) rose 47.187 → 48.490% for the same reason. The sweep above S1 moved
by ≤ 1.1% (relative). Every other arm moved by ≤ 0.007 points. **Every F6 threshold rises (the interior ones by
≤ 0.004 points); no criterion loosens.**

| arm | granted % of C | SC-4 % of C | SC-4 E ms | C ms | NIC util | F6: pieces must beat it by (granted → SC-4) |
|---|---|---|---|---|---|---|
| `stage0.today` | 123.990 | 131.418 | 13381.42 | 10182.3 | 3.657 | — |
| `stage0.step1` | 71.850 | 77.890 | 7930.96 | 10182.3 | 2.803 | 34.280 → **37.300 pts / 3797.9 ms** (vs stage0.pieces_fwd) |
| `stage0.lpt` | 20.635 | 25.109 | 2556.68 | 10182.3 | 1.948 | 8.673 → **10.909 pts / 1110.8 ms** (vs stage0.pieces_fwd) |
| `stage0.pieces_fwd` | 3.290 | 3.291 | 335.08 | 10182.3 | 0.945 | — |
| `stage0.pieces_bwd` | 3.290 | 3.291 | 335.08 | 10182.3 | 0.945 | — |
| `stage0.fifo_pieces_fwd` | 3.213 | 3.213 | 327.14 | 10182.3 | 0.945 | — |
| `stage0.fifo_pieces_bwd` | 3.342 | 3.342 | 340.27 | 10182.3 | 0.945 | — |
| `stage0.barrier_pieces_fwd` | 3.291 | 3.291 | 335.08 | 10182.3 | 0.945 | — |
| `interior_w4.today` | 88.266 | 105.629 | 6514.35 | 6167.2 | 1.729 | — |
| `interior_w4.step1` | 19.473 | 19.476 | 1201.14 | 6167.2 | 0.875 | 6.105 → **6.106 pts / 376.6 ms** (vs interior_w4.pieces_fwd) |
| `interior_w4.lpt` | 9.307 | 9.310 | 574.15 | 6167.2 | 0.387 | — |
| `interior_w4.pieces_fwd` | 7.263 | 7.264 | 448.00 | 6167.2 | 0.387 | — |
| `interior_w4.pieces_bwd` | 7.157 | 7.157 | 441.39 | 6167.2 | 0.387 | — |
| `interior_w4.fifo_pieces_fwd` | 7.135 | 7.135 | 440.05 | 6167.2 | 0.387 | — |
| `interior_w4.fifo_pieces_bwd` | 8.328 | 8.328 | 513.60 | 6167.2 | 0.387 | — |
| `interior_w4.barrier_pieces_fwd` | 7.263 | 7.264 | 448.00 | 6167.2 | 0.387 | — |
| `interior_w8.today` | 139.140 | 174.494 | 7523.96 | 4311.9 | 2.014 | — |
| `interior_w8.step1` | 47.187 | 48.490 | 2090.82 | 4311.9 | 1.017 | — |
| `interior_w8.lpt` | 26.793 | 26.800 | 1155.60 | 4311.9 | 0.519 | 4.761 → **4.765 pts / 205.5 ms** (vs interior_w8.pieces_fwd) |
| `interior_w8.pieces_fwd` | 17.270 | 17.270 | 744.64 | 4311.9 | 0.448 | — |
| `interior_w8.pieces_bwd` | 17.312 | 17.312 | 746.47 | 4311.9 | 0.448 | — |
| `interior_w8.fifo_pieces_fwd` | 17.270 | 17.270 | 744.64 | 4311.9 | 0.448 | — |
| `interior_w8.fifo_pieces_bwd` | 19.754 | 19.754 | 851.75 | 4311.9 | 0.448 | — |
| `interior_w8.barrier_pieces_fwd` | 17.525 | 17.526 | 755.69 | 4311.9 | 0.448 | — |
| `interior_w4.emission_fwd.rtt0` | 7.263 | 7.264 | 448.00 | 6167.2 | 0.387 | — |
| `interior_w4.emission_bwd.rtt0` | 7.157 | 7.157 | 441.39 | 6167.2 | 0.387 | — |
| `interior_w4.emission_fwd.rtt_2cw` | 8.202 | 8.203 | 505.87 | 6167.2 | 0.387 | — |
| `interior_w4.emission_bwd.rtt_2cw` | 9.177 | 9.178 | 566.02 | 6167.2 | 0.387 | — |
| `stage0.emission_fwd.rtt0` | 3.290 | 3.291 | 335.08 | 10182.3 | 0.945 | — |
| `stage0.emission_bwd.rtt0` | 3.290 | 3.291 | 335.08 | 10182.3 | 0.945 | — |
| `stage0.emission_fwd.rtt_2cw` | 3.375 | 3.375 | 343.67 | 10182.3 | 0.945 | — |
| `stage0.emission_bwd.rtt_2cw` | 3.375 | 3.375 | 343.67 | 10182.3 | 0.945 | — |
| `interior_w4.uplink_pieces_fwd` | 10.315 | 10.316 | 636.21 | 6167.2 | 0.387 | — |
| `interior_w4.uplink_pieces_bwd` | 10.209 | 10.209 | 629.60 | 6167.2 | 0.387 | — |
| `interior_w4.tree` | 11.841 | 11.841 | 730.28 | 6167.2 | 0.387 | — |
| `stage0.tree` | 3.983 | 3.984 | 405.63 | 10182.3 | 0.945 | — |
| `sweep.util_0.5` | 9.950 | 9.951 | 613.70 | 6167.2 | 0.500 | — |
| `sweep.util_0.7` | 14.687 | 14.688 | 905.83 | 6167.2 | 0.700 | — |
| `sweep.util_0.9` | 19.424 | 19.425 | 1197.96 | 6167.2 | 0.900 | — |
| `sweep.util_1.1` | 33.273 | 32.912 | 2029.72 | 6167.2 | 1.100 | — |
| `sweep.util_1.3` | 55.641 | 55.148 | 3401.11 | 6167.2 | 1.300 | — |
| `sweep.util_1.6` | 89.194 | 88.504 | 5458.20 | 6167.2 | 1.600 | — |
| `sweep.util_2.0` | 133.931 | 132.977 | 8200.98 | 6167.2 | 2.000 | — |

| round 5 §6 arm (F′ 50, 1:2:1, HBM′) | round5.md | SC-4 | relative |
|---|---|---|---|
| `interior_w4.pieces_fwd` | 3.66 | 3.488 | -4.70% |
| `interior_w4.step1` | 16.00 | 16.210 | +1.31% |
| `interior_w8.lpt` | 25.10 | 24.637 | -1.84% |
| `interior_w8.pieces_fwd` | 10.03 | 10.028 | -0.02% |
| `stage0.lpt` | 20.09 | 24.978 | +24.33% |
| `stage0.pieces_fwd` | 2.82 | 2.846 | +0.94% |
| `stage0.step1` | 71.31 | 77.577 | +8.79% |

| F6 arm | alternative | granted threshold (pts) | SC-4 threshold (pts) | SC-4 threshold (ms) |
|---|---|---|---|---|
| stage0 | step1 | 34.280 | 37.300 | 3797.94 |
| stage0 | lpt | 8.673 | 10.909 | 1110.80 |
| interior_w4 | step1 | 6.105 | 6.106 | 376.57 |
| interior_w8 | lpt | 4.761 | 4.765 | 205.48 |

| sweep quantity | granted | SC-4 |
|---|---|---|
| S1 at cap′ (MB/s) | 6.4609 | 6.4609 |
| fit below S1 | -1.892 + 153.022/cap′ | -1.891 + 153.022/cap′ |
| fit above S1 | -89.754 + 722.605/cap′ | -89.391 + 718.354/cap′ |
| predicted threshold cap′ (MB/s), utilization | 6.4827, 0.9966 | 6.4609, 1.0 |

| diagnostic (not a prediction), % of C beyond the slot | SC-4 | stage 0 packed | one fold after the last arrival |
|---|---|---|---|
| E4:interior_w4.pieces_fwd | 7.264 | — | 7.263 |
| E4:interior_w8.lpt | 26.800 | — | 26.793 |
| E4:stage0.lpt | 25.109 | 23.877 | 25.109 |
| E4:stage0.pieces_fwd | 3.291 | 0.000 | 3.290 |
| E4:stage0.step1 | 77.890 | 76.657 | 77.888 |
| round5:interior_w4.pieces_fwd | 3.488 | — | 3.661 |
| round5:interior_w8.lpt | 24.637 | — | 25.104 |
| round5:stage0.lpt | 24.978 | 23.746 | 24.978 |
| round5:stage0.pieces_fwd | 2.846 | 0.000 | 2.815 |
| round5:stage0.step1 | 77.577 | 76.345 | 77.494 |


The last table's columns are diagnostics, not predictions:
- **stage 0 packed:** the as-built harness's padding, every op back to back and the idle at the end. The exposure
  beyond the slot is then `E − window_s`.
- **one fold:** the as-built release rule, one fold after the last arrival instead of the left fold.

They attribute the gaps between this PLAN and the as-built simulation (below).

**Modes a coder must add to `exchange_sim.py` / `validation/exposure.py` for these pins (stated precisely):**
1. *The `today` arms*: whole-stage flush: stride ZeRO-1 of whole tensors (params[k::w], stage 0 base 2, 12 parameters per block, modules at +2, +4, +8, +10; tok and pos are tensors 0 and 1), fp32 wire (b_g = b_p = 4 on every unit, so folds at 12 B/element), every unit's gradient pieces of W(mb) released at W(mb)'s END (the Embed too, not at W start); the emission order is therefore irrelevant; queues, chunks, resources, latencies and the left fold as I17; full step barrier: every gather released at Tb = max over (unit, owner) of (its last fold + AdamW), and step t+1's F on every unit waits for Tb and for that unit's gather; boundary bytes as every other arm. Inputs: the stage's units with today's stride indices
   (`stage_units`), `layout="zero1"`, wire fp32 (every unit b_g = b_p = 4), a flag `flush="stage"`. Release time of every
   unit's pieces from W(mb) = W(mb)'s end, the Embed included; own contributions are local at that time. Queues
   need-first as in I17. Barrier: `Tb` as stated; every unit of step t+1 waits for `Tb` and for its gathers; the
   emission order is irrelevant. Boundary bytes unchanged.
2. *Stage-0 padding*: per peer q, its i-th owned microbatch (global index i*w + q, i = 0 .. M/w - 1) occupies the slot [i*tau, (i+1)*tau), tau = C_B'/(M/w); with cF, cB, cW = c_s times the pinned op split: F runs [i*tau, i*tau + cF], B runs [(i+1)*tau - cW - cB, (i+1)*tau - cW], W runs [(i+1)*tau - cW, (i+1)*tau] (the idle between F and B is the wait for the downstream backward); no F runs ahead of its slot. So the last W ends at C_B', the period is C_B', window_s = 0, and step t+1's first F is due at C_B'. A harness may realize it with a per-peer wait of tau - c_s between each F and its B (one stand-in per peer, never one shared by the peers, comm 0, no exchange), as long as the op times above result.
3. *Left fold*: left fold of every microbatch's contribution in canonical order, each at the measured rate ((b_g + 16) B/element for a 2-byte wire, 12 B for fp32), then AdamW at 30 B/element; the unit's gathers leave after the last fold and the AdamW (I17). As built, a gather leaves at
   `max(arrivals) + one fold + AdamW`. The difference is ≤ 0.02 points at the E4 configuration (fold not paced) and
   −4.7% on `round5_anchor` interior w 4 pieces (paced HBM′: three near-simultaneous pieces fold in turn, and the later
   gather lands behind a chunk, which shortens the tail; a chunk-boundary effect, not a trend).
4. *Round-5 anchor read from the PLAN*: `exposure.py`'s `ROUND5_PCT` and `tests/unit/test_zero_bubble.py`'s seven
   `round5.md` §6 pins are constants today. They must read `PLAN.json` `round5_anchor.arms` (SC-4: "the test reads
   the values from the PLAN").
5. *The Embed as one fp32 unit*: one fp32 unit (tok + pos, I17), released when W(mb) starts; stride and LPT own its two tensors whole (tok and pos may sit on different peers). As built, `validation/exposure.py` `stage_units` gives
   `tok` and `pos` two units; I17 names the Embed's tables as one unit (released at W start, needed first by F).
