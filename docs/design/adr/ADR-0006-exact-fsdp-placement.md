# ADR-0006: Exact sharded stages: per-tensor pieces under the recipe's left fold, admitted per link and per unit

<!-- Approval lives in docs/design/APPROVALS.json (agents-architetura/tools/factory/approve.mjs), bound to this exact text. Do not write a
     Status line claiming approval. Mark the ADR "Status: REJECTED", "Status: SUPERSEDED" or "Status: WITHDRAWN" only
     when that is the human's decision. -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE DECISION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: with today's cut points (whole layers of `[Embed, Block × n, Head]`; ASSUMPTION A1) some stages must be wide:
more devices than cut points (7B has 34 layers; 128 devices force w 4 at S 32). Today's wide stage (ZeRO-1) keeps a
full fp32 replica, and its exchange is slow and skewed: whole tensors on owners, the fp32 `tok` table on one peer. The
human asks for byte-exact FSDP / ZeRO-3, placed by the planner inside a site or across ~10 Gbps links.

Current bottleneck: the recipe's reduction is ONE left fold of all M microbatch gradients in index order
(`reference.py:81-91`), so every microbatch's gradient crosses to its owner. Today adds to it: stride layout
(`stage.py:154`), fp32 bytes, whole-microbatch `flush`, a step barrier, synchronous copies.

Proposed architecture (one decision: how a wide stage is sharded exactly, and the predicate that admits it):
1. *Per-tensor pieces.* Every trainable tensor of a width-w stage is cut along dim 0 into w pieces whose boundaries are
   multiples of `Q_t = lcm(row_len_t, 128)` elements. Peer k owns piece k of every tensor: fp32 master, AdamW m and v,
   fp32 fold accumulator. The layout records explicit boundaries (equal by default).
2. *A step-resident working copy in the consumer dtype* (the recipe's compute dtype for `wgrad.owned_params` under
   autocast, else the master dtype). Each unit is updated as soon as its fold completes; the owner casts its piece once
   and sends it (a byte-copy all-gather, unit by unit). The step barrier goes, except at checkpoint, re-plan and last
   steps; step t+1's F on a unit waits for that unit's copy.
3. *Gradients in their production dtype, streamed per module,* in an emission order the plan fixes. Copies between
   device and host are asynchronous, on a copy stream, in pinned chunks of ≤ 16 MiB; each per-destination send queue
   serves need-first (forward unit, gathers first). Remote early arrivals wait in host memory, the owner's own pieces
   on the device; all M fold in microbatch-index order on the device. No collective library, no division.
4. *Admission* per stage and width: device class; S1 on every link (NIC, PCIe, host DRAM and site uplink, summed over
   every stage and GPU that loads it, each at its own `c_s`); S2 from the per-unit simulation; device memory (state,
   own-piece hold, in-flight boundaries, pending W); host memory at its structural bound of one step, summed over the
   peers on each host. The planner takes the smallest simulated step, deeper cuts and today's `zero1` stages included.
5. *Logical-layout checkpoints,* and heal replaces a sharded-stage peer only with the same device model.

First-principles reasoning: every term except the fold is a byte copy or an elementwise op, and the fold is the
reference's own fold on a slice of elements, so the bits are the oracle's at every width. The step-boundary tail hides
behind the last W and the first F only if every unit is spread over all w links, which only per-tensor pieces
guarantee; the largest single tensor (the 524 MB fp32 `tok` gradient at 7B) shows it most.

Mathematical justification: S1 per link ℓ: `Σ_{(s,g) on ℓ} k_ℓ·(b_g·P_s·share + A)/c_s ≤ cap_ℓ`. Exposure
`E = max(X_s, V_ℓ/cap_ℓ + RTT + t_gap − C)`, C from the slowest stage; S2: `E − window_s ≤ ε·C`. Device
`b_p + 16/w + b_g·M_k/w` B/param plus one module; host ≤ `b_g·P_s·[M_k·(w−1)/w + (M−M_k)/w] + 2·b_p·P_s·(w−1)/w`.

Expected improvement (DERIVED, `docs/research/adr-0006/round4.md`, `round5.md`; HYPOTHESIS for real hardware): forced
width (7B, 128 H100, S 32, w 4, 400 Gbps, C 871.4 ms): step 2163.3 → 1124.5 ms with pieces only where S1 admits
stage 0 on host DRAM: 1-GPU 600 GB/s hosts at k ≤ 6 passes per byte. At today's k 10 S1 refuses stage 0 on every host
(≤ 0.64 GPUs per host; 3.18 at k 2; interior 1.34, 6.70); the planner takes step 1 (`wire: native`), 1721.3 ms. 32 H100,
400 Gbps: S 16, w 2 with pieces, 966.8 vs 1102.2 ms, on 1-GPU hosts at k 10 or 8-GPU at k 2; else `zero1` or depth.

Alternatives: do nothing; step 1 alone; LPT; flat ranges; deeper pipeline (block or sub-block cuts); chained folds;
recipe fold tree; per-op ZeRO-3; owner-side W; exact accumulators; TP; stock FSDP (baselines only).
Risks: a new wait (E1); host DRAM binds even 1-GPU hosts at stage 0 at today's k (OPEN, R10); I16 refuses link-bound
pieces that beat every admitted plan (OPEN, R13); the lane-order house fix must land first (R14); copy cost (F7).

Falsification criteria: one sha256 difference (E1, E5); bytes off > 2% (E2); a memory term off > 20% or a hold above
its bound (E3); exposure outside [0.8, 1.2]× the as-built simulation (E4); pieces not beating step 1 alone and LPT by
half the predicted margin at any F6 arm, pinned at the scaled B′ (`round5.md` §6) (F6); E7 off its predicted choice
for its host configuration (F9); a deadlock or stall (F8).

Local experiment: E1–E7, CPU only (gpu false), localhost clusters, WAN shim with shared pacers; timing SIMULATED.
Estimated Vast.ai cost: none: local only (real-hardware validation pending).
Recommendation: PROCEED (local experiment first) inside a site; at today's k the forced-width gain of pieces needs the
transport ADR (R10; R10, R13 deferred, R14 first). Across ~10 Gbps no useful 7B placement is admitted.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Awaiting explicit human approval: node agents-architetura/tools/factory/approve.mjs grant docs/design/adr/ADR-0006-exact-fsdp-placement.md docs/design/DESIGN.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Problem

The request (docs/research/context-fsdp/README.md, 2026-10-09) is stated above. Hard constraints: the product rule;
fp32 masters, optimizer state and outputs (bf16, fp16 and fp32 compute recipes run, `precision.py:84`); step time within
ε of homogeneous PP; one decision per ADR, ≤ 400 lines. Not taken here: a recipe tree, chained folds, sub-block cuts,
transport copy reduction, TP. ASSUMPTION A1: cut points are whole layers of `[Embed, Block × n, Head]` (`arch.py:228-231`,
`partition.py:186`; finer cuts are a model change, R11). ASSUMPTION A2: `seq_len` 4096 at 7B (`pos` is 4096·h).

## Current Architecture

FACTS (code read on this tree):
- Reference (`validation/reference.py:81-91`): one `autograd.grad` per microbatch; `acc = g0.clone(); acc.add_(g_i)` in
  index order; no division; one AdamW (wd 0) step.
- ZeRO-1 (`stage.py:152-158`): `params[peer::w]`, whole tensors; at w 4 peers 1 and 3 get only biases and norms
  (`gpt.py:255-265`). W (`stage.py:915-924`): `wgrad.flush` widens every dW to fp32 (`wgrad.py:57-70`); `_store_grads`
  ships non-owned slices by a synchronous `.cpu()` through FIFO per-destination queues (`stage.py:938-950`, `942`) and
  folds the own slice on the device (`stage.py:940`, `969`), held until its prefix is folded (`accumulate.py:34`, `43`).
- Embed's `tok` (V×h) and `pos` are `nn.Embedding`s: fp32 parameters and gradients, produced in B (`gpt.py:239-249`).
  `Head.out` is an untied `nn.Linear` (`gpt.py:317`, `:97`), converted to a `SplitLinear` (`wgrad.py:139`).
- Step loop: `_optimizer_step_wide` blocks until all w−1 PARAMS broadcasts arrive (`stage.py:1042-1047`, `1065-1070`).
- Transport: 10 host-memory passes per sent byte per side (`serial.py:65`, `79`, `143`; `framing.py:117`, `144`;
  kernel and DMA copies; `round3.md` §2), exchange and boundary frames alike.
- Checkpoints are per slot (`stage.py:708-755`); the identity manifest assumes `names[p::w]` (`validation/state.py:98-125`);
  `find_replacement` defaults to `series` (`replacement.py:20-60`); the engine bridge solves a wide stage as one server.

## Current Bottleneck

`T_step = C·(1 + β) + E − window_s + T_ovh`: β the lane bubble, E the exposed exchange, `T_ovh` device and
compute-thread overhead. At the forced-width point B (7B: h 4096, V 32k, 32 blocks; 128 H100 at 400 TFLOP/s,
HYPOTHESIS; S 32, w 4, M 128, `T_mb` 4096) the min-max cut is `[Embed + Block 0] … [Block 31 + Head]`; the Head stage
sets c (27.23 ms against 16.49): C 871.4 ms, lane makespan 1041.1 ms. Today stage 0 exposes 1075.7 ms and the last
1080.4 (S1 fails on the owners of `tok` and the Head): a 2163.3 ms step. Synchronous copies add 419 ms per step at PCIe
Gen5 (838 at Gen4) at layout point A (an interior stage: S 8, w 4, M 32).

## Evidence

- DERIVED (no measurement): `adr6_model_r2.py` (sha256 `f43ab4f6…b701eb`) and `depth.py` (`4cdc2dec…ef5621`) in
  `derivation.md` §9; `r3.py` (`acf1cb86…995ed6`) in `round3.md` §7; `r4.py` (`d4838233…dd0c0`, runs the repository's
  `zero_bubble.py`, `22988afc…751ac8d`) in `round4.md` §9; `r5.py` (`cc425fde…e37d8`, loads `r4.py`) in `round5.md`
  §8; `python3 -I`, exit 0 (round 5 reproduced round 4's output hash `7ea94c7f…ecdbe`).
- MEASURED (round 0): `(2⁻²⁴, 2⁻²⁴)` and `(2⁻²³, 0)` have equal fp32 sums; folded onto `s = 1` they give `0x1.000000p+0`
  and `0x1.000002p+0`. MEASURED by research angle 1 (scratch probe, torch 2.14.1, arm64; E1 and E6 reproduce it):
  fp32→bf16 equals integer RNE on 4,718,592 patterns; `SplitLinear` dW under autocast is bf16-exact, LayerNorm
  gradients are not; a ring reduce-scatter differs from the left fold in 48,284 of 65,536 elements at w 4; ÷3 changes
  51,959; single, foreach and fused AdamW are bit-equal on whole tensors and on 3 range shards (CPU).
- Cited, not re-measured: FSDP/FSDP2; ZeRO, ZeRO++ hpZ; Zero Bubble PP; P3; ByteScheduler; Megatron DistOpt; LPT;
  DCP/UCP; veScale-FSDP RaggedShard (arXiv 2602.22437).

## Hypothesis

- **H1, exactness.** With items 1–3, the sha256 of the logical training state and every loss hex equal the oracle
  (`FACTORY_REFERENCE=1`) at every width, piece split, compute dtype (fp32, bf16, fp16), equal or weighted IWRR
  ownership, zero-bubble order, W fill and emission order, with per-unit updates, and through resume at another layout.
- **H2, volume.** Egress per peer per step is `P_s·(w−1)/w·[(M_k)·b̄_g + b̄_p]` plus frame headers. **H3, exposure.**
  The as-built per-unit simulation (SCHEDULER-I17) predicts each stage's exposed exchange `E`.
- **H4, memory.** Device: copy `b_p·P_s`, owned `16·P_s/w`, own-piece hold `≤ b_g·M_k·P_s/w`, `b_g`·largest module;
  host: gradient hold and backlog plus gathers out and in never above
  `b_g·P_s·[M_k·(w−1)/w + (M−M_k)/w] + 2·b_p·P_s·(w−1)/w` (`b_g·P_s` meaning `Σ_u b_g,u·P_u`).
- **H5, liveness.** The per-unit step loop completes every step under any valid order, fill, jitter and straggler.

## First-Principles Derivation

**Exactness, term by term.** (a) *Working copy:* a `SplitLinear` consumes weight and bias only through
`w.to(autocast dtype)`, or the master when autocast is off; the owner applies that cast once per step, so the
consumer's cast is a no-op; LayerNorm, embeddings (`gpt.py:239-251`) and tied weights stay fp32. (b) *Forward and
backward* run the same kernels on the same bytes (MODEL-I1, RUNTIME-I6). (c) *Gradients:* `_weight_grads` returns the
autocast dtype, whose widening is exactly the fp32 the reference adds; a multi-use parameter keeps today's fp32 sum in
store order. (d) *Emission order:* each store closure computes `dYᵀX` from its own tensors, so order changes no bit.
(e) *Fold:* fp32 `add_` is correctly rounded per element, so the piece fold performs the reference's adds in its order,
from an fp32 copy of microbatch 0's widened piece (never a bf16 clone, never zeros: `+0 + −0 = +0`); where a piece waits
changes no bit. (f) *Update:* AdamW is elementwise; quantum-aligned pieces take the same vector-or-tail path per element
(RUNTIME-I11). (g) *Device class:* gradient bits depend on the device, so every peer is in the reference's class, at
admission, re-plan and heal.

**The left fold cannot be pre-summed.** `s_M = fl(…fl(fl(s_0+g_1)+g_2)…)` depends on each `g_i`, not on a summary of a
subset that does not start at index 0 (counterexample above). Any exact scheme that keeps the fold where it is moves
each microbatch's owned pieces: `M_k·b_g·P_s·(w−1)/w` per peer per step. Moving the fold (a chained fold) is a second
fold placement (derivation §8).

**The tail (derivation §2–§3, `round3.md` §4, `round4.md` §1).** Step t+1's F of unit u needs unit u's update, which
needs every microbatch's unit-u gradient. Backward emission produces unit 1 last, forward emission first (the Embed's
tables, produced in B, ship when W(mb) starts); on a volume-bound link both end near `G + P_g + RTT − (c_W + c_F)`, and
forward hides the RTT (H100, `T_mb` 16k, RTT 10 ms: 14.4 → 0 ms). The plan picks the order by simulating the runtime
as specified: need-first send queues, ≤ 16 MiB chunks, links shared max-min fairly, gathers after the owner's AdamW.

**Links, PCIe and host DRAM (derivation §1, `round3.md` §2, `round4.md` §5, `round5.md` §2).** A microbatch puts
`P_s·share_ℓ` bytes per byte-per-param on link ℓ: `(w−1)/w` on NIC and PCIe, `n_a·n_b/w` on an inter-site uplink.
PCIe is in series with the NIC (`min(NIC, PCIe)`, plus `2·f/PCIe` = 0.67 ms per hop at Gen5); host DRAM carries `k`
passes per byte per side, exchange and boundary bytes alike, for every GPU of the host (k 10 today, 2 at the floor).
Copies run on transport threads; the compute thread records one event per yielded tensor (HYPOTHESIS ≈ 1 ms per step).

**Memory (derivation §4, `round3.md` §1, `round4.md` §7, `round5.md` §4).** Device: copy `b_p`, owned `16/w`, own
pieces waiting for their prefix (`≤ b_g·M_k/w`), one module's dW, boundaries, pending W. Host: one step of gradients
(a step-(t+1) piece follows F(t+1), hence every step-t gather and fold) plus one step of gathers per unit, out and in;
the planner sums it over each host's peers. Width buys no usable memory plan at whole-layer cuts: a 1T layer admits
`M/w ≤ 4` (w 4) or 14 (w 8), a lane bubble of 17.1 or 4.9 C at S 138; a 405B layer fits at w 1 (62.3 of 70.4 GB).

**Step loop and liveness (derivation §7).** A unit's update runs at the first pump after its fold completes. Step-t
updates depend only on step-t W pieces, which every peer emits before any step-(t+1) op, so no slot waits on a
step-(t+1) copy while holding an update another needs; by induction every step completes (barrier kept at checkpoint,
re-plan and last steps; the step loop and `_optimizer_step_wide` change only under `pieces`).

**Composition.** *Pipeline:* stages are sharded independently; `window_s` and the step come from one slot-level
simulation (SCHEDULER-I17) assuming the lane-order house fix (R14). *M2 phase 2:* re-planned microbatch ownership moves
no bit and changes `M_k` (re-charged); re-planned pieces move master, m and v at a barrier step. *Checkpoints:* a
`(name, offset, length)` manifest, per-chunk sha256. *Heal:* same device model (CONTROL-I9). *Security:* the audit
unit stays one microbatch (CONTROL-I13); frames are checked at the boundary (dtype, piece shape, mb, step, one per key).

## Mathematical Model

This ADR moves `E`, `T_ovh` and per-peer memory; β is the planner's (lane solve). Variables: `P_s`, w, S, U units,
`T_mb`, `M_k` (peer k's microbatches; `M/w` equal weights), F (deterministic-kernel rate), `cap_ℓ`, `share_ℓ` (on host
DRAM `2·(w−1)/w`, both directions), `k_ℓ` (passes per byte per side: 1 on a NIC, PCIe or uplink; the transport's count
on host DRAM), `A_ℓ` (boundary bytes per microbatch round on ℓ), `b_g`/`b_p` per unit (`b_g·P_s` = `Σ_u b_g,u·P_u`), ε.

```
c_s = 8·P^F_s·T_mb/F (P^F: GEMM params) ;  c_F = c_W = c/4 ;  c_B = c/2 ;  C = (M/w)·max_s c_s ;  β: lane solve
S1:  Σ_{(s,g) on ℓ} k_ℓ·(b_g·P_s·share_{s,ℓ} + A_{s,ℓ})/c_s ≤ cap_ℓ     ℓ: NIC, PCIe per direction; host DRAM; uplinks
G = b_g·P_s·share/cap ;  P_g = b_p·P_s·share/cap ;  lat = 2·f/cap_PCIe per hop
X_s = as-built simulated tail; volume-bound limit ≈ G + P_g + RTT − (c_W + c_F)
E   = max( X_s , max_ℓ V_ℓ/cap_ℓ + RTT + t_gap − C ),  V_ℓ = M_k·(b_g·P_s·share_ℓ + A_ℓ) + b_p·P_s·share_ℓ
S2:  E − window_s ≤ ε·C          (window_s = 0 on stage 0)
T_ovh = M·(P_s/w)·B_f/BW_HBM (fold; B_f 18 for a 2-byte wire, 12 fp32) + (P_s/w)·30/BW_HBM (AdamW) + events (≈ 1 ms)  HYPOTHESIS
device_k = b_p·P_s + 16·P_s/w + b_g·M_k·P_s/w + b_g·P_mod + min(S, M/w)·T_mb·h·2 + L_s·T_mb·16·h·4
host_k  ≤ b_g·P_s·[ M_k·(w−1)/w + (M − M_k)/w ] + 2·b_p·P_s·(w−1)/w ;  Σ_{k on host} host_k ≤ host memory
```

Point B (`round4.md` §1, `round5.md` §2): `T_ovh` 53.4, 36.1, 58.9 ms (stage 0, interior, last; the fold is `M·P/N`
elements per GPU at every placement). S2 at stage 0: 24.5 ≤ ε·C = 26.1 ms; its NIC at 94%. Host DRAM per GPU, S1 as
written: stage 0 944.8 GB/s at k 10, 189.0 at k 2; interior 447.6 and 89.5. S1 uses each stage's own `c_s`, so it is
conservative with unequal stages (R10); with every parameter in the window, stage 0's c would be 1.73× too long.
Across one 10 Gbps uplink (derivation §6) straddling stages need `T_mb ≥ S·40k` (H100, w 2), which memory refuses.

**What moves and when the gain disappears.** Bytes ×0.5 (gone for an fp32 recipe); E (gone when S1 fails on any link,
host DRAM included); synchronous copies (only with async copies); device state `b_p + 16/w` against 11.0 (gone at
w = 1), own hold growing with M/w. **Placement** (`round4.md` §3, `round5.md` §3): with the Head counted, deep
whole-layer cuts pay its imbalance (slowest stage 1.65× at S 32, 1.08× at S 4), so at 400 Gbps width shortens the step
where host DRAM carries it; pieces are 2.8% faster than `zero1` at (16, 2) (1-GPU hosts at k 10, 8-GPU at k 2) and are
refused by S2 at (8, 4) and (4, 8); on 4- and 8-GPU hosts at k 10 depth wins. At 100 Gbps S1 refuses every width.

## Expected Improvement

DERIVED (`round4.md` §1–§4, `round5.md` §2–§3); SIMULATED in E4, E7; real hardware pending. As presented above (B:
LPT 1275.1 ms; 100 Gbps fails S1 everywhere). Interior A (w 4): step 1 531.7 → 84.5, pieces 18.8. Device state
(B/param): w 4 11.0 → 6.17 (6.67 at `L_s` 1), w 8 9.5 → 4.17; own hold up to 4 at A, 16 at B (32 for the fp32 tables;
today's fp32 hold up to 31 and 127). Memory: none. Bits unchanged.

## Alternatives

Priced at interior point A (S 8, w 4, 400 Gbps; w 8 where stated) and at point B's binding stage 0 (`round4.md`):

| Option | Exposed (% of C); device B/param (state) | Why not chosen |
|---|---|---|
| Do nothing (stride ZeRO-1, fp32, barrier) | A 531.7 ms (100.7%); 11.0. B: step 2163.3 ms | S1 fails: skew, 2× bytes; sync copies |
| Step 1 alone (`wire: native`, stride, barrier) | A 84.5 (16.0%), w 8 111.5 (42.3%); B stage 0 621.3 (71.3%), step 1721.3; 11.0 | `tok` and whole tensors on few owners; barrier; first migration step |
| LPT whole tensors, per-unit | A 35.3 (6.7%), w 8 66.2 (25.1%); B stage 0 175.1 (20.1%), step 1275.1; A 6.17, w 8 4.50, B 8.00 / 10.35 | Units land on one owner; S1 fails at B stage 0 |
| Flat element ranges | A 40.5 (7.7%), w 8 61.0 (23.1%); 6.17 | Unit 1 on `max(1, w/U)` owners; cross-tensor ranges |
| Per-tensor pieces (chosen) | A 18.8 (3.6%), w 8 26.5 (10.0%); B stage 0 24.5 (2.8%), step 1124.5; 6.17, w 8 4.17, B 6.67 / 8.35 | — |
| Deeper pipeline, block cuts, w = 1 | 32 H100, 400 Gbps: 1102.2 ms vs 966.8 (S 16, w 2) | Not exclusive: priced every plan; kept at 100 Gbps and on 4–8-GPU hosts at k 10 |
| Sub-block cuts, w = 1 | B: makespan 1427.6 ms (per linear), 1176.5 (halves) vs 1124.5; 8.6–23.6 GB per GPU per direction through host DRAM | Head not cuttable (2.55× imbalance); same host link; a model change (A1, R11) |
| Chained fold, site or peer | Cross-site 6 B/param per step, no M/w factor; 64 ms per serialized hop at A | Second fold placement; own ADR (only path to cross-site width) |
| Recipe fold tree; exact or binned accumulators | per-step volume 6–34 B/param | New oracle bytes: recipe change, own ADR |
| Per-op ZeRO-3 regather (FSDP RAF) | 586 Gbps at `T_mb` 4096 on H100; 1T w 4 `M/w ≤ 7` (bubble 9.8 C), w 8 ≤ 22 (3.1 C) | No usable 1T plan either; 586 Gbps at 7B |
| Owner-side W (ship X, dY) | bytes ×`16·T_mb/(12h)` = 1.33 | Wins only at `T_mb` < 0.75h; needs whole-tensor owners |
| Column-parallel TP | One layer > a device | Not needed below h ≈ 50k |
| Stock FSDP/HSDP, Megatron DistOpt, NCCL, RaggedShard | Speed ceiling | Ring order varies with w; world-size division: baselines only |

## Trade-offs

- Volume and the own-piece hold scale with `M_k`; slow links need a large `T_mb`, which memory caps; host DRAM carries
  every sent byte k times per side; one device class per stage, also at heal; host memory holds one step per peer. The
  working copy keeps `b_p` per param; frames per microbatch grow from `w−1` to `U·(w−1)`; F waits on a unit's copy.

## Risks

- R1: the plan-ordered iterator must not change the multi-use path (MODEL-I1, MODEL-I7); E1's controls catch it.
  R2: a stale working copy is a correctness bug: copies are step-tagged per unit and every op asserts the tag.
  R3: host memory is charged at its one-step bound, gathers included, summed per host (A 21.7 GB, +15% owning 12 of
  32; B interior 19.9, stage 0 49.2, last 32.9; an 8-GPU host with B's stages 0 and 1: 276.5 GB; `round5.md` §4).
  R4: liveness rests on updates in every pump (E1, E3, F8). R5: async copies are unobservable on CPU (F7). R6: init
  peak until MODEL-I6. R7: ε is the human's (3%); S2 refuses (8, 4), (4, 8) at 400 Gbps though faster (902.5, 867.8
  ms). R8: `cap_ℓ` measured under load. R9: own pieces wait on the device (charged; E3). Security: frame checks, < 0.1%.
- R10 (DEFERRED by the human, 2026-10-09; reopened after E4/E7 validate the simulation and M1 measures k on real
  hardware): host DRAM binds, stage 0 first: per 600 GB/s host at B it admits ≤ 0.64 GPUs at k 10 (none) and 3.18 at k
  2; interior 1.34, 6.70; last 1.54, 7.68. At today's k the point-B pieces plan is refused on every host; the planner
  takes `zero1` + native (1721.3 ms, 1-GPU hosts) or S 32, w 1 on 32 GPUs (3887.8, 8-GPU hosts; `zero1` 5038.6). A k ≤
  6 transport, an S1 on the pipeline's period, or host packing lifts it: each its own decision.
- R11 (OPEN, for the human): sub-block cuts do not tie point B (≥ 52 ms slower in halves, 303 per linear), load host
  DRAM alike, and fit a 1T layer per linear at w 1 (48.3 GB): the only usable 1T path. R12 (OPEN): on open fleets a
  quarantined microbatch (CONTROL-I13) holds its fold; S2 is optimistic there.
- R13 (DEFERRED by the human, 2026-10-09, reopened with R10): I16 refuses pieces whenever S1 or S2 fails, even where
  they beat every admitted plan (100 Gbps, 128 H100: S 32, w 1 3887.8 ms vs pieces 2291.7); a predicate change.
- R14 (prerequisite; M2b's first task, by the human's decision): the engine bridge solves a wide stage as one server
  (109.4% vs the lane's 43.8% at S 8, w 4); every lane makespan here assumes its fix lands before SCHEDULER task 4.
  House: `find_replacement` defaults to `series`; the TCP checksum. R15 (OPEN): the send key orders unit before step,
  so a step-(t+1) gradient can pass a step-t gather (F3 bounds it; R3's bound does not depend on it).
- R16 (OPEN): E7 assumes stage-contiguous host packing (`round5.md` §3); the host-DRAM time of `zero1` candidates is a
  bound (busiest host), HYPOTHESIS until I17 shares host DRAM as built. S2 at B stage 0 holds by 6% (24.5 vs
  26.1 ms), inside F3's ±20%. E7's predictions use 8 FLOPs per param; the planner's `layer_costs` counts attention
  (Head/block 0.558 vs 0.651), so E7's PLAN re-pins them with the planner's costs before it runs.
- Refuted: "credits move the hold" (r1); "forward emission lowers `X_s`" (r2, only where RTT dominates); "the fold
  flips depth against width" (r3, `M·P/N` per GPU at every placement); "host copies belong in `T_ovh`" (r3, a link);
  "one Embed unit moves S2 at stage 0" (r5: X 24.53 ms either way; stride keeps tok and pos apart, `round5.md` §5).

## Falsification Criteria

- F1 (H1): any sha256 difference (weights, m, v, step, RNG, any loss hex) against `FACTORY_REFERENCE=1` in 20 jittered
  runs at w ∈ {1, 2, 3, 4}, in bf16, fp16 and fp32, either emission order, weighted IWRR, or across resume w 4 → 3 → 2.
- F2 (H2): measured GRADS+PARAMS bytes per peer per step differ from H2 by more than 2% plus frame headers.
- F3 (H3): on the shim with shared pacers, measured exposure outside [0.8, 1.2]× the as-built simulation at ≥ 3 points
  on each side of S1, or the S1 threshold off by > 25%, or the shared-uplink arm missing by the same bounds. Control:
  with the barrier kept, the measurement matches the barrier simulation; otherwise the shim, not H3, is wrong.
- F4 (H4): any memory term exceeds its prediction by > 20%, or the host bytes (gradients and gathers, per peer and
  summed per host) or the device own-piece hold ever exceed their one-step bound (a bug, not a tolerance), including
  with one peer 2× slow, a slow destination ingress, several peers per host and weighted IWRR.
- F5 (streaming beats the tree): refuted if a test-only tree arm exposes ≤ the left fold's time while S1 holds.
- F6 (the decision): pieces are refuted on time if, at ANY one arm, their measured exposure does not beat an
  alternative by at least half the predicted margin, at the scaled B′ whose ratios (E4) and predictions `round5.md` §6
  pins (the PLAN re-pins them at the profiled F′ before E4 runs): stage 0 (Embed + 1 block, w 4, M/w 32, period C_B′)
  against step 1 alone (71.31% vs 2.82% of C_B′; 1213.6 vs 47.9 ms at F′ 50 GFLOP/s) and LPT (20.09%; 341.9 ms);
  interior w 4 against step 1 alone (16.00% vs 3.66% of C); interior w 8 against LPT (25.10% vs 10.03%). No memory
  argument rescues a failed arm; a run off the PLAN's scaled configuration is not evidence.
- F7 (real hardware, pending): the compute thread stalls on exchange copies or the GIL for more than 1% of C, a hop's
  copy latency exceeds `2·f/PCIe` by 2×, or measured host DRAM traffic per sent byte exceeds the counted k by 20%.
- F8 (H5): any deadlock, stall verdict or timeout in an E1/E3 arm.
- F9 (planner): E7 chooses a plan other than the one predicted for that host configuration (below), or one whose
  simulated step time is > 1% above the best admitted candidate's.

## Local Experiment

CPU, gpu false, localhost clusters with the LinkSpec shim; `WANTRAIN_SIM_SLOW` sets F. Every number is SIMULATED.
- E1, identity and liveness. 2 stages, stage 1 sharded at w ∈ {2, 3, 4} (and a sharded stage 0 holding Embed), bf16,
  fp16 and fp32, recompute, W fill, both emission orders, per-unit updates, equal and weighted IWRR, a checkpoint and a
  re-plan step, 20 jittered runs, by sha256. Controls that must FAIL: LayerNorm gathered in bf16; a bf16 accumulator
  clone; arrival-order fold; gloo `reduce_scatter`; ÷3; a bf16 working copy in the fp32 recipe; multi-use contributions
  reordered. Liveness control that must DEADLOCK (timeout-bounded): updates run only between ops.
- E2, bytes, against H2 (F2), plus the planes codec and changed-copy fraction. E3, memory: byte counters per term at
  w 2 and 4 against H4, then one peer 2× slow, one destination with a slow ingress pacer, weighted IWRR with one peer
  owning 12 of 32, and two to four peers per process host summed (F4, F8).
- E4, exposure. One sharded stage at w 4, w 8 on the shim with per-peer shared pacers (SPEC-transport task 3) at the
  scaled B′, which keeps (`round5.md` §6, PLAN-checked, ±0.5%): the Embed's share of stage 0 (tables 0.42, `tok` 0.375
  of params), the fp32 share of its gradient bytes (0.59), NIC utilization at `c_s` (0.94), M/w, fold time and chunk
  over `c_s`, RTT over `c_W`. Arms: today, step 1, LPT, pieces (both orders), stage 0 (Embed + 1 block, w 4, M/w 32,
  padded to C_B′), FIFO control; `T_mb`, B through S1; RTT 0, > c_W/4; a shared-uplink arm (F3, F6); a tree arm (F5).
- E5, resharding: resume at w 3, w 2, moved boundaries, as `zero1`, all sha256 equal; a one-element shift FAILS. E6,
  kernels: cast and AdamW on pieces against the whole tensor, plus an exhaustive 2³² cast check (F1).
- E7, planner, predicted choices (`round5.md` §3; 600 GB/s hosts, stage-contiguous packing): 7B on 32 H100 at
  400 Gbps → S 16, w 2, pieces on 1-GPU hosts at k 10 and 8-GPU hosts at k 2; `zero1` + native at (16, 2) on 2-GPU
  hosts at k 10 (stage 0's host DRAM named); S 32, w 1 on 4- and 8-GPU hosts at k 10; at 100 Gbps → S 32, w 1.
  7B on 128 H100 at 400 Gbps: 1-GPU hosts at k 10 → S 32, w 4, `zero1` + native (pieces refused, stage 0's host DRAM
  named); at k ≤ 6 → pieces; 8-GPU hosts at k 10 → S 32, w 1 on 32 GPUs; at k 2 → `zero1` + native; at 100 Gbps →
  S 32, w 1 on 32 GPUs. H100 over one 10 Gbps uplink and mixed classes are refused with the link or class named; host
  memory below the per-host sum of one-step bounds refuses (MEM-HOST).

## Vast.ai Experiment

None while gpu is false; with GPUs on, a PLAN.json (2 GPUs on one host; two hosts) measures S1 incl. host DRAM, S2, F7.

## Migration Impact

- No recipe and no reference change; existing runs keep their bytes. Two `JobSpec` fields, defaults `fp32` and `zero1`
  until E1–E6 pass; JobSpec refuses `{wire: fp32, shard: pieces}`:
  - `wire: "fp32" | "native"` (step 1): native dtype, plan-ordered per-module iterator, asynchronous copies,
    need-first send queues; valid for ZeRO-1; the step barrier stays; no checkpoint change.
  - `shard: "zero1" | "pieces"` (step 2): layout, working copy, per-unit updates and gathers without the barrier
    (kept at checkpoint, re-plan and last steps), holds, logical checkpoints, which read into `zero1` as `names[k::w]`
    of the logical tensors (RUNTIME-I12), so `shard` rolls back byte-identically; old checkpoints read as `names[k::w]`.
- Plans: a job planned today keeps a plan: where I16 refuses pieces the planner prices `zero1` wide stages and narrower
  plans in the same simulation, takes the smallest step, and names every refused width with its link, number or class.
- The identity manifest changes only under `pieces`: the logical state in canonical name order and, in place of the
  replica check, each peer's final working-copy digest per tensor; under `zero1` it keeps `names[p::w]`.
- Workstreams: model (`flush_iter`), runtime (pieces, working copy, queues, updates, copies, holds, checkpoint,
  manifest), scheduler (S1, simulation, S2, memory, class, candidates), control (class-preserving heal), transport
  (shared pacers for E4). The lane-order fix is a scheduler house item (R14).

## Validation Plan

RUNTIME-I10–I13, MODEL-I7, SCHEDULER-I16–I17, CONTROL-I9 (amended) and TRANSPORT-I8 on the unit, integration, identity,
chaos and wan rungs; the identity rung gains w 3, fp16, fp32, emission-order, weighted-IWRR and other-width arms.

## Ideas considered

| Idea (source) | Verdict | Reason and number |
|---|---|---|
| Rounds 0–2: native wire dtype, async copies, forward-need priority (P3, ByteScheduler), topological links, summed S1, pieces, hpZ copy, logical checkpoints, class-preserving heal, forward-order W, lane pricing, host bound, deterministic F; credits, element ranges, host-side fold, RDMA, weighted pieces, relay, multipath, spill, chained folds, accumulators, tree kernels, split | ADOPTED; WITHDRAWN / REJECTED / DEFERRED / own ADRs | 2× bytes; sync copies 419–838 ms at A; A 84.5 → 18.8 ms; RTT 10 ms 14.4 → 0; blocked W stalls; ranges 40.5 vs 18.8 ms; host-side fold +64 GB per GPU on host DRAM; recipe change |
| Round 3: need-first queues; lossless, delta and exponent coding; devmem TCP, NVLink, zero-copy; contiguous ownership, fused fold, own pieces via host, epoch, S1 rate form, audit latency | ADOPTED (queues: 18.8 vs 33.9 ms FIFO) / MEASURE in E2 / FOLLOW-UP | Coding ratio unmeasured (r = 1); host passes 10 → 0–2; own hold 16 → ≈ 0.5 B/param; fold 34.6 → 23.1 ms |
| Round 4 (13 ideas; rows in `round5.md` §7): barrier-kept LPT, lower k, thread-free optimizer, RaggedShard, several TCP flows, step-first key, bounded D2H, sparse touched-rows `tok`, incast arm, contiguous ownership; host-DRAM share | FOLLOW-UP / CITED; share APPLIED as text | LPT with barrier ≥ 175.1 ms at B stage 0 (6.7× ε·C); lower k: pieces at B need k ≤ 6 (`round5.md` §3); sparse `tok` 524 → ≤ 67 MB per mb |
| Round 5: admit pieces by simulated step time, S1 and S2 as reports (alternatives) | FOLLOW-UP (predicate change; R13, question 6) | Recovers 1124.5 vs 1721.3 ms at B on 8-GPU hosts at k 2 and 2291.7 vs 3887.8 at 100 Gbps; no bit changes |
| Round 5: a stage's peers on one host or NVLink domain, exchange by CUDA IPC or NVLink copies (alternatives, frontier) | FOLLOW-UP (transport ADR, question 5) | Byte copies, exactness untouched; drops the exchange's `k·2·(w−1)/w` host term: B's busiest 8-GPU host 5038.6 ms → its boundary bytes, ≈ 430 ms at k 10 (HYPOTHESIS) |
| Round 5: host-packing-aware placement, stage 0 beside light stages (consistency, first-principles) | FOLLOW-UP (planner; R10, R16) | S1 is summed per host already; at k 2 a 600 GB/s host fits stage 0 and 4 interior GPUs (547 GB/s) where it fits 3 stage-0 GPUs; at k 10 stage 0 alone exceeds a host (944.8) |
| Round 5: a tables-only S1 term; owner-side W for the Embed, shipping token ids and dX rows (consistency, first-principles) | FOLLOW-UP (predicate change; new mechanism, E6 must prove the row-restricted kernel's order) | The window ambiguity is removed by text (one `c_s`); owner-side Embed W: `tok` 393 → ≤ 67 MB per mb, stage 0 NIC 0.94 → ≈ 0.45, host S1 at k 10 1.57× → ≈ 0.75× (reviewer's figures, HYPOTHESIS) |
| Round 5: attention-inclusive `layer_costs` (`h(72h + 12s)`) in `c_s` and C (alternatives) | FOLLOW-UP (scheduler; R16) | Conservative for S2 (C grows); moves the Head/block ratio 0.651 → 0.558 and E7's 2.8% margin at (16, 2) |
| Round 5: lossless exponent coding fused before D2H (ZipCCL, UCCL-Zip, DFloat11; frontier) | MEASURE in E2 on gradient pieces first; FOLLOW-UP | Lossless, so allowed; ratio unmeasured (planes codec 1.426 on bf16 boundaries, `precision.py:36`); at 1.3–1.4 interior k 2 → 8.7–9.4 GPUs per host, stage 0 3.18 → 4.1–4.5 (HYPOTHESIS) |
| Round 5: OPEN-1B (arXiv 2609.17380) as prior art and F6 datapoint; Tree-Based Invariant Kernels (arXiv 2511.17826) for the recipe-tree ADR (frontier) | CITED by the reviewer, not read or re-measured here; FOLLOW-UP | Same class (all-to-all, then a fixed-order owner sum); its overhead against FSDP2 would bound the stock-FSDP row; tree kernels change the oracle's fold: a recipe ADR |
| Round 5: pin the AdamW implementation (foreach, for-loop, fused) as a recipe field in the fingerprint (frontier) | FOLLOW-UP (RUNTIME-I11 owner) | Bit-equal on CPU (round 0 probe); a CUDA default change is unobservable while gpu is false |

## Changelog

- Rounds 0–2 (2026-10-09): the mechanism (items 1–5), the predicate, F6–F9. Round 3: mechanism frozen; own pieces on
  the device, weighted-IWRR bounds, host DRAM in S1, fold in `T_ovh`, exposure as built, A1. Round 4 (`round4.md`):
  Embed and Head priced (C 871.4 ms, stage 0 binds), F6 per arm, depth against width, boundary bytes at k, sub-block
  cuts and the memory pillar priced, the `zero1` fallback, R13–R15, 13 ideas.
- Round 5 (2026-10-09): mechanism and predicate frozen; accounting corrected (`round5.md`). S1's window is the stage's
  `c_s` (GEMM params) in I16 and DESIGN §9, as in the model; host-DRAM S1 per stage and per host (stage 0 binds:
  0.64 GPUs per host at k 10), so point-B pieces need k ≤ 6 and E7 is restated per host configuration; the host bound
  gains one step of gathers and is summed per host (MEM-HOST); own hold 32 B/param for the fp32 tables; E4's scaling
  map and scaled F6 predictions; R10 restated, R16 added; the Embed-unit finding refuted; 12 ideas recorded.

## Amendments (written with this ADR; texts in DESIGN.md and the specs)

- DESIGN.md §0, §8 (M2, M2b), §9, §10; SPEC-scheduler I16 (S1 at `c_s`, per link and host; MEM-HOST), I17, task 4;
  SPEC-runtime I10–I13 (I12's manifest is the step-bound sidecar of M2 phase 2), task 5; SPEC-model I7, task 2;
  SPEC-control I9 (heal by device model); SPEC-transport I8, task 3. Each change is in its -history.md.

## Recommendation

PROCEED with exact sharded stages (per-tensor pieces under the existing left fold) inside a site, local experiment
E1–E7 first, migrating in two flags (`wire`, then `shard`). Do not commission the recipe tree; the memory case is
withdrawn. At today's k = 10 step 1 (`wire`) carries the forced-width gain (2163.3 → 1721.3 ms at B) and pieces win
only on 1-GPU hosts at (16, 2); pieces' 1721.3 → 1124.5 ms needs k ≤ 6 (R10). The human deferred R10 and R13
(reopened after E4/E7 and M1) and made R14 M2b's first task.

Questions for the human: (1) ε (3% assumed); (2) is a bf16- or fp16-produced gradient shipped as its exact 2-byte image
acceptable under "fp32 on the wire"? No value is rounded; (3) one ADR with two flags, or split? (4) commission the
chained-fold ADR? (5) commission a transport ADR (R10: lifts pieces, `zero1` and sub-block depth alike) before a
sub-block-cut ADR (R11: the only usable 1T path found)? (6) R13: deferred by the human with R10.

## Human Approval
Granted only by `node agents-architetura/tools/factory/approve.mjs grant` in a human's terminal; recorded in docs/design/APPROVALS.json.
