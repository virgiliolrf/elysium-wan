# ADR-0002: The recipe fold tree and element-range ZeRO-1

Status: PROPOSED

<!-- Approval lives in docs/design/APPROVALS.json (agents-architetura/tools/factory/approve.mjs), bound to this exact text. Do not write a
     Status line claiming approval. Mark the ADR "Status: REJECTED", "Status: SUPERSEDED" or "Status: WITHDRAWN" only
     when that is the human's decision. -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE DECISION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: the reduction order is part of the recipe (CLAUDE.md). Today's order, one left fold of all M microbatch
gradients in index order, makes a wide stage ship every microbatch's gradient: `(M/w)·P_s·4·(w−1)/w` bytes per peer
per step, M/w times a native reduce-scatter. Whole-tensor ZeRO-1 (`params[k::w]`) also sends every microbatch gradient
of the embedding or head to one owner: at 1B, w = 8, M = 32, 100 Gbps, 28 × 1.05 GB = 29.4 GB, 2.35 s.

Current bottleneck: the wide stage's exchange **where `τ > c`** (τ: one partial's transfer per peer; c: one
microbatch's F+B+W on that peer), exposed after the stage's last W: 379 ms against 75 ms of stage compute at 1B,
w = 8, 100 Gbps (DERIVED). Where `τ ≤ c` (7B at 400 Gbps) today's shipping is hidden and nothing here moves the step.

Proposed architecture (one decision: **the recipe's reduction order for wide stages and the sharding it allows**):
1. *Element-range ZeRO-1* (OUT). A stage's flat list of recipe-fixed chunks (ADR-0001 D3) is split into w contiguous
   ranges of nearly equal element count; peer k owns range k's optimizer state (flat fp32 buffers), `sumsq` and `opt`.
2. *Residue radix tree* (IN), **per class block**. Two recipe constants per class block, set before planning and never
   derived from a width or a peer speed: `fold_groups` G and `fold_radix` `[r_0, …, r_{L−1}]`, `Π r_i = G`. The root's
   children are the classes `mb mod r_0`; each splits by `mb mod r_0·r_1`, down to the G groups `mb mod G`. A group is
   the left fold of its microbatches in index order; every node is the left fold of its children in index order. A
   block without them: `G = M`, `[G]`, today's bytes and one accumulator. `fold_mode` (IN) is reserved; only `tree`.
3. *Ownership and exchange* (OUT). At an aligned width `w = r_0·…·r_{d−1}` peer k owns `mb ≡ k (mod w)`, one subtree,
   and ships one partial. Owners pull every final leaf from all senders at once, fragment by fragment, and compute a
   node once its children are present. Unequal speeds: the cover with the smallest predicted exposure.

First-principles reasoning: microbatches arrive in index order and residue ownership deals them round robin, so every
peer folds on arrival; the tree is a function of the recipe alone, so width, ownership and speed decide only who
computes which node, never a bit. A tree that one wide width evaluates natively must keep `w` partials open on a
width-1 stage (bound below): the class-block scope confines that cost to blocks the author plans wide.

Mathematical justification: per peer per step `k_p` leaves cost `k_p·P_s·4·(w−1)/w` plus the fp32 gather; exposure
from each leaf's closing time (§Mathematical Model); accumulators `(G/w)·P_s·4` on a tree block of width w.

Expected improvement (DERIVED; SIMULATED by E18e): `G = 8`, `[2, 2, 2]`, 100 Gbps: 1B, w = 8 379 → 105 ms; 1B, w = 2
721 → 58 ms ([1, 0.7]: 68.5); 7B, w = 8 2510 → 794 ms; 0 where `τ ≤ c`. Element-range alone: incast 2.35 s → 440 ms.

Alternatives (all priced): do nothing; element-range only; fewer, larger microbatches (recipe M); left fold over
groups; contiguous groups; `G = k·w` (withdrawn); weighted IWRR partition (Q7); exact fold of native fp32 dW; exact
integer dW; binned; ring and chained orders; recursive halving.

Risks: a tree block on a width-1 stage holds G accumulators: +11.0 GB at 1B, +86.8 GB at 7B (G = 8), refused by the
planner; grain 1/G; the audit unit becomes a leaf partial (CONTROL-I13); legacy checkpoints stay refused (ADR-0001 D11).

Falsification criteria: one sha256 difference in N ≥ 20 runs among widths 1, 2, 3, 4 of one recipe; `G = M` differing
from today's oracle; per-peer bytes off by more than one chunk (≤ 3.1% of the mean) plus 5%; exposure off the per-leaf
prediction, or drain off `drain(F)` at a pinned RTT, by > 5% of the step; peak memory off the accumulator count.

Local experiment: CPU only (gpu false): E12, E18a–i on ADR-0001's CPU classes; timing SIMULATED.
Estimated Vast.ai cost: none: local only.

Recommendation: PROCEED with the class-block residue radix tree, local experiment first, and a ruling on Q7.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Awaiting explicit human approval: node agents-architetura/tools/factory/approve.mjs grant docs/design/adr/ADR-0002-recipe-fold-tree-element-range-zero1.md docs/design/DESIGN.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Problem

"Every step reduces the gradients of the whole global batch in a canonical order that the recipe defines" (CLAUDE.md).
Today's order is correct but forces a wide stage to move M/w times the native ZeRO-1 volume, and whole-tensor sharding
concentrates the largest tensors on one owner. Question: which order, fixed by the recipe alone, can a wide stage
evaluate at native volume without moving a bit relative to its reference on any fleet, and at what cost to the stages
that are not wide? Second of four one-decision ADRs split from ADR-0001 round 3 (`docs/research/hetero-fold-tree.md`).

## Current Architecture

FACTS (code read):
- With `stream_grads`, each microbatch's non-owned shard slices go to their owners as soon as its W ends
  (`stage.py:834-857`); owners fold all M in index order (`OrderedAccumulator(range(self.M))`, `stage.py:868-873`).
- Whole-tensor ZeRO-1: `self.shard_params = self.params[self.peer :: self.w]` (`stage.py:116-120`); checkpoints save
  `self.opt.state_dict()` per slot (`stage.py:633-641`); the identity manifest assumes it: `mine = names[p::w]`
  (`validation/state.py:125-127`). Parameters are fp32 masters; ownership is IWRR (`zero_bubble.py:408-424`).
- `OrderedAccumulator` quarantines an untrusted contribution at its index (`runtime/accumulate.py:13-22`; CONTROL-I13).
- Recipe text this ADR builds on: chunks of `C` = `sumsq_chunk` elements (IN, default 2^20, DESIGN §11.3); class
  blocks (layer range → class key) are IN (§11.6); the planner gives each class one width (§11.7); ADR-0003's memory
  predicate already charges `4·G/w·P_s` of accumulators; RUNTIME-I22 refuses every pre-ADR-0001 `.pt` checkpoint.

## Current Bottleneck

Per peer of a wide stage (DERIVED): `τ = P_s·4·(w−1)/w / B_intra`, `c` one microbatch's F+B+W, `t_W = c/4`. Today a
peer ships M/w per-microbatch partials, exposed for `E(M/w)` with `E(k) = max(0, τ − t_W) + (k−1)·max(0, τ − c)` (k
partials closing at successive microbatches). Shipping hides iff `T_mb ≥ T* = 4·((w−1)/w)·F_eff/(9.33·B_intra)`
tokens. ADR-0003's points (H100, 400 TFLOP/s; 1B: T_mb 2048, `P_s` 393M; 7B: T_mb 4096, `P_s` 3.1B; M = 32), ms:

| point | w | Gbps | τ | c | today `E(M/w)` | `[8]`: `E(G/w)` | `[2,2,2]` aligned | `[2,2,2]`, [1, 0.7] | stage `M·c/w` |
|---|---|---|---|---|---|---|---|---|---|
| 1B | 2 | 100 | 63 | 18.8 | 721 | 191 | 58 | 68.5 | 300 |
| 1B | 2 | 400 | 16 | 18.8 | 11 | 11 | 11 | 11 | 300 |
| 1B | 8 | 100 | 110 | 18.8 | 379 | 105 | 105 | — | 75 |
| 1B | 8 | 400 | 28 | 18.8 | 49 | 23 | 23 | — | 75 |
| 7B | 2 | 100 | 496 | 296 | 3422 | 1022 | 422 | — | 4740 |
| 7B | 8 | 100 | 868 | 296 | 2510 | 794 | 794 | — | 1185 |
| 7B | 8 | 400 | 217 | 296 | 143 | 143 | 143 | — | 1185 |

Incast, 1B, w = 8, 100 Gbps, head 262.7M elements: whole-tensor 29.4 GB, 2.35 s; element-range 5.50 GB per link,
440 ms (110 ms with one leaf).

## Evidence

- MEASURED script output (round 3, `docs/research/hetero-scripts/spread.py`, sha256 `4c22a457…6342`): stage time with
  whole-group ownership against `M/Σv`, M = 32, `G/w` = 1, 2, 4 and `G = M`: [1, 0.7] +21.4/+21.4/+6.2/+0.9%;
  [1, 0.9] +5.6/+5.6/+5.6/+0.9%; [1,1,1,0.9] +8.3% throughout.
- MEASURED script output (round 2, scratchpad): `fill.py`, contiguous group ownership against round robin adds
  `(G/w − 1)(w−1)c/w` (1.5c, 0.75c, 0.5c at three points); `ek.py`: the table above.
- MEASURED script output (this round, scratchpad `leafexp.py`, sha256 `54d7c3be…0708`, not in the repo: the owned
  paths exclude `docs/research/`): a stage fed in index order at `Σv/c`, each peer running its microbatches in index
  order, each leaf's slices streamed from its close − `t_W` and serialized on the peer's egress; all covers enumerated.
  1B, 100 Gbps, `[2,2,2]`: w = 2 equal E 58.2 ms; [1, 0.7] n = (5, 3), k = (2, 2): **E 68.5 ms with the cover rule,
  102.3 ms for the worst equal-k cover**, stage 387.1 ms; w = 3 equal n = (3,3,2), k = 2 each: E 125.4 (worst 144.2),
  stage 238.1. Exact fold, 8 B partials, IWRR at 1/M: w = 2 E 121.1 (stage 310.2 equal, 376.0 at [1, 0.7]); w = 3
  163.0 (213.1); w = 8 215.4. Today: w = 2 719.4, w = 8 379.1.
- HUMAN INPUT (recorded in `docs/research/hetero-exact-accumulation-track.md` §2.3): E-a on CPU, exact accumulation
  equal by sha256 under any order; 4.7–9.3 int8 passes per bf16 GEMM, 2.9–4.3 per fp8. Not reproduced by the factory.
- Related work, verified this round or by reviewers on 2026-10-08 (verdicts in §Ideas considered): H2, arXiv
  2505.17548 (full text read this round: "1,024 chips of four distinct architectures", 100B model; precision checked
  only as loss "MRE below 1.5%" against A100, worst chip 1.215%, 20B, 300 iterations; no bitwise claim); OPEN-1B,
  arXiv 2609.17380 (1.61B, 400B tokens, 48 H100; "ascending sum over shards", "TreeFold, a binary-carry fold" pairing
  replicas as recursive doubling does; dQ "in 64-bit fixed point"; auditor replays "in mesh order"); Cooper et al.
  2609.25624 (inference, Ampere/Ada/Hopper, 1.17–3.1×); HAPT/Harp 2509.24859 (1.3–1.6×); HETHUB 2405.16256; AccPar
  (HPCA 2020, simulated); Srivastava, Arora & Boneh 2403.09603; Verde/RepOps 2502.19405; RepDL 2510.09180 (no
  affiliation on arXiv); DASH 2601.21824 (≤ 37.9% loss, ≤ 1.28×); Ozaki 2306.11975, 2504.08009; FP4 limbs 2608.06812
  (authors' claim); Jetfire 2403.12422 (1.42×); GeoMesh 2609.18388 (sign-compressed ≈ 32×, lossy).
  Knowledge-verified: Thakur, Rabenseifner & Gropp, IJHPCA 2005; Knuth TAOCP 2 §4.2.2 (TwoSum); Ogita, Rump & Oishi,
  SISC 2005; Neal 1505.05571; Demmel & Nguyen 2015; Graham 1969; Rizzo, CCR 1997; Zero Bubble PP 2024; Megatron-LM.
  **Not verified, left out:** HAP5 (ICDCS 2026), Tessera (OSDI 2026).

## Hypothesis

- **H-tree.** With the radix tree and element-range ZeRO-1, every run (any width `w ≤ G`, any lane ownership, any
  chunk-range split, any zero-bubble order, W fill) equals R by sha256 (ADR-0001 H1). **H-fleet:** one recipe gives one
  sha256 at widths 1, 2, 3, 4 and any peer speeds.
- **H-legacy.** A recipe without `fold_groups` gives today's oracle bytes on every class where P-slice holds.
- **H-volume.** Peer p's egress is `k_p·P_s·4·(w−1)/w + P_s·4·(w−1)/w`; largest link − mean ≤ C elements.
- **H-exposure.** Exposure after a wide stage's last W follows the per-leaf `E` within 5% of the step.
- **H-drain.** The last-W-to-last-fold time follows `drain(F)` for F = F_min and F = 1 at a pinned RTT.
- **H-memory.** A peer holds `(G_b/w)·P_b·4` accumulators per tree block b (w = 1 included), plus `(w−1)·F·f` fragments
  and the slices of leaves final before their siblings.

## First-Principles Derivation

- **Product-rule conformance.** CLAUDE.md forbids an order "chosen by topology, arrival order or worker speed"; C4
  admits only the stage-to-class map. G and the tree are fixed per class block before any plan; widths, ownership and
  speeds decide only who computes which node (OUT). Q7 asks whether more may join. **Why a tree:** one left fold over
  all M forces per-microbatch shipping (the owner adds mb before mb + 1).
- **Width-1 bound (finding 1, accepted).** A wide stage at width w deals microbatches round robin (required: bursts
  cost `(G/w − 1)(w−1)c/w`, `fill.py`), so a tree it evaluates with one partial per peer has peer k's partial = the fold
  of `mb ≡ k (mod w)`. Upstream emits in index order through every stage, so a width-1 stage of the same block sees
  all w residues interleaved until the last round: it holds ≥ w accumulators, G for the residue tree. Any tree native at
  width G pays `(G−1)·P_s·4` on width 1: **11.0 GB at 1B (P_s 393M), 86.8 GB at 7B (P_s 3.1B), G = 8**; 99.2 GB of
  accumulators exceed an 80 GB H100 before weights. Reordering W on the width-1 stage instead buffers stashed W inputs
  across rounds. So the cost cannot be removed, only confined: G is scoped **per class block**, the scope at which the
  planner chooses width (one width per class, DESIGN §11.7); a block without G keeps one accumulator.
- **Shape.** A node's value is fixed by its operands and their order. Residue ordering (digit-reversed) makes each
  aligned subtree a residue class `mb ≡ k (mod w)`: round-robin dealing, today's IWRR at equal speeds. Aligned widths
  are the prefix products `{1, r_0, r_0·r_1, …, G}`; the author orders the radix by planned widths.
- **Ownership (finding 2, accepted).** Peer k owns `n_k` groups, `Σ n_k = G`, n minimizing `max_k n_k/v_k`, then the
  fewest leaves `max k_p`, then the smallest predicted exposure `E` (below), then the lexicographically first
  assignment: all OUT, none moves a bit. Without the exposure rule equal-k covers differ by 34 ms at [1, 0.7], w = 2
  (68.5 vs 102.3 ms, `leafexp.py`): the extra leaf that closes earliest ships behind the peer's remaining compute.
- **Exchange (D3).** A sender serves a leaf's chunk once its last W for that chunk is done; an owner pulls every final
  leaf's fragments of its range from all senders at once, F fragments in flight per sender, and computes a node for a
  fragment once its children are present. Per flow (S = `P_s·4/w` bytes, `n_f = ⌈S/f⌉`, `s = (w−1)·f/B` per fragment)
  a window of F completes every `max(F·s, RTT + s)`: `drain(F) = (n_f/F)·max(F·s, RTT + s) + RTT`, which is `τ + RTT`
  (+ s) for `F ≥ F_min = 1 + ⌈RTT·B/((w−1)·f)⌉` and `τ + (n_f + 1)·RTT` at F = 1 (finding 3: round 2's "`(w−1)·τ`" was
  round 1's serialized credit order, not a small window). 1B, w = 8, 100 Gbps: F_min 2 (RTT 1 ms, 56 MiB per owner)
  and 6 (RTT 10 ms, 168 MiB); F = 1: 158 and 590 ms against τ = 110. A leaf final before its siblings is pulled and held
  at the owner (≤ `P_s·4/w`); its sender frees it. Senders never wait on a pull to compute: no circular wait.
- **Tail bound.** ≤ `k_max·τ` + drain slack; ≥ τ for any exact scheme (the last leaf leaves its peer once final).
- **Element-range ZeRO-1** (replaces `params[k::w]`). Fold, `sumsq` and `opt` are per chunk, so a range boundary never
  changes a summation. Owner state is flat fp32 range buffers of chunk views (torch AdamW cannot own a slice of an
  `nn.Parameter`): `exp_avg`, `exp_avg_sq` per chunk, one `step`; the AdamW implementation is pinned (IN) and passes
  RUNTIME-I11 per chunk. Largest link − mean ≤ one chunk (0.34 ms at 100 Gbps). **Gather** in the master dtype (4 B).
- **R (finding 10, accepted).** R evaluates the same trees at width 1. Order is free for bits; it sets memory. In the
  tree's leaf order (groups digit-reversed, each in index order: `[2,2,2]` → 0, 4, 2, 6, 1, 5, 3, 7) a stack holds
  `L + 1` accumulators per element (4 at G = 8); natural group order holds `G/2 + 1` (5); round 1's "two" held only for
  the left fold over groups. When class blocks differ in tree, R runs in index order and holds `G_b` per tree block;
  R_paged pages accumulators like optimizer state.
- **SCHEDULER-I18.** On one peer, a group's W units run in ascending microbatch order; `finalize()` refuses others.

## Mathematical Model

`T_step = T_compute + T_communication + T_bubble + T_sync + T_overhead`. This ADR moves the wide stage's exposed
exchange (`T_communication`), its balance (`T_compute`), and memory on every stage of a tree block:

```
τ        = P_s·4·(w−1)/w / B_intra                     one leaf's non-owned slices
t_ℓ      = end of the last W of leaf ℓ on its peer p     (close time; t_W = c/(4·v_p))
φ_p      = fold over p's leaves by t_ℓ: φ ← max(φ, t_ℓ − t_W) + τ        (serialized egress)
E        = max_p φ_p − max_p T_lastW,p                  (= E(k) when k leaves close at successive microbatches)
drain(F) = (n_f/F)·max(F·s, RTT + s) + RTT,  s = (w−1)·f/B,  n_f = ⌈P_s·4/(w·f)⌉
work_s   = max_k n_k·(M/G)·λ_s/v_k                      integer n, Σ n_k = G
memory   = Σ_b a_b·P_b·4 + (w−1)·F·f (+ early leaves ≤ P_s·4/w each)   a_b = G_b/w on a tree block, 1 otherwise
```

Which term moves: `E(M/w) − E(1) = (M/w − 1)·max(0, τ − c)`: 274 ms at 1B, w = 8, 100 Gbps; 663 ms at 1B, w = 2;
1716 ms at 7B, w = 8. Memory: a tree block costs `(G/w − 1)·P_s·4` over today (4.7 GB at 1B, w = 2; 11.0 GB at w = 1).
**The gain disappears** when `τ ≤ c` (`T_mb ≥ T*`), at width 1 (cost only), with `G = M`, or in a pure data-parallel
recipe (one stage) whose activation memory admits `M = w` (§Alternatives). At non-aligned widths or unequal speeds it
falls to the per-leaf E (68.5 ms at [1, 0.7], 125.4 ms at w = 3; 1B, 100 Gbps).

## Expected Improvement

Exposure: tables above (DERIVED; E18e SIMULATED; `T_step` in ADR-0003's E11). Volume: one leaf per peer at aligned
widths, `k_p` otherwise, M/w today. Incast 2.35 s → 440 ms → 110 ms. Bytes unchanged for recipes without `fold_groups`.

## Alternatives

Priced at 1B, 100 Gbps, M = 32 (`leafexp.py`, `ek.py`); "E" is exposure after the last W; "w1 mem" the extra
accumulator bytes on a width-1 stage of the block.

| Option | Volume | E: w = 2 / [1, 0.7] / w = 3 / w = 8 (ms) | w1 mem | Grain | Verdict |
|---|---|---|---|---|---|
| Do nothing (`G = M`, `params[k::w]`) | M/w | 719 / — / — / 379 + 2.35 s incast | 0 | 1/M | REJECTED |
| Element-range only (`G = M`) | M/w | 719 / — / — / 379 | 0 | 1/M | ADOPTED as D1 |
| **Residue radix tree `[2,2,2]` per class block, D1, D3** | 1 aligned, `k_p` else | **58 / 68.5 / 125 / 105** | +11.0 GB | 1/G | **PROCEED** |
| Fewer, larger microbatches: M = 8, T_mb 8192, today's fold | M/w | 44 / — / — / 91 | 0 | 1/M | REJECTED: +42 ms fill per upstream stage |
| Exact fold of native fp32 dW (fp64 + TwoSum flag, expansion fallback) | 2 (8 B) | 121 / 121 / 163 / 215 | +1.57 GB | 1/M | REJECTED here; ADR-0005 candidate |
| Left fold over groups (`[G]`, round 1) | G/w | 191 / — / — / 105 | +11.0 GB | 1/G | REJECTED: dominated |
| Natural-order tree, contiguous groups | 1 aligned | 58 + 28 fill | +11.0 GB | 1/G | REJECTED: dominated |
| Planner-chosen `G = k·w` | k | — | — | 1/(k·w) | WITHDRAWN: prohibited order |
| Weighted IWRR partition, `G = w` per wide stage | 1 | 58 / 58 / — / 105 | 0 | 1/M | Q7 variant |
| Exact integer dW (`fold_mode = exact`) | ≥ 2 | — | 8 B | 1/M | ADR-0005: W ×1.34–1.91 |
| Binned (ReproBLAS); chained; per-chunk ring | 3; 1; ≥ 1.75 | — | +8 B; M/w | 1/M | REJECTED (§Ideas) |

- **Fewer, larger microbatches** (finding 6). Same global batch, M = 8: per-microbatch time `c′ = 4c` = 75 ms at 1B.
  w = 8: `E(1) = τ − c′/4` = 91 ms, 14 ms better than the tree's 105; w = 2: `E(4)` = 44 ms (τ = 63 < c′). Cost: each
  upstream stage adds `3λ/4·(M/M′ − 1)` = 42 ms of fill (ADR-0003's `Σ_{q<s} 3λ_q/4`); activation memory per microbatch
  ×4 (in-flight count per ADR-0003's `infl_s`, HYPOTHESIS). With one upstream stage: 91 + 42 = 133 vs 105 ms (w = 8),
  86 vs 58 (w = 2); 7B, w = 8: 572 + 666 = 1238 vs 794 ms. It wins only for a one-stage (pure data-parallel) recipe
  whose memory admits it (91 vs 105 ms; 572 vs 794 ms at 7B). The tree composes with any M, so the author keeps it.
- **Exact fold of native fp32 dW** (finding 5). Each microbatch's dW is the class-native fp32 GEMM output; the fold is
  exact: fp64 accumulation is exact iff `24 + d + log2 M ≤ 53` (d the element's exponent spread, ≤ 24 at M = 32), and
  TwoSum's error term flags the elements that are not, which continue as exact expansions; one rounding to fp32. Bits
  are independent of order, grouping, width and speed. Total stage + E (`leafexp.py`): w = 2 equal 431 vs tree 368 ms;
  [1, 0.7] 497 vs 456; w = 3 376 vs 364; w = 8 equal 307 vs 197. The tree wins every priced point because 8 B partials
  double τ; the exact fold wins width-1 memory (+1.57 vs +11.0 GB at 1B; +12.4 vs +86.8 GB at 7B), grain, and needs no
  radix, I18 or block scope. Its fold reads and writes 8 B more per element per microbatch (0.94 ms, 5.0% of c at 1B,
  3.35 TB/s HBM; HYPOTHESIS). PROCEED is not conditional on E-a's d histogram: the 8 B best case is priced above and
  loses on exposure; the histogram can only add expansion traffic. It is a `fold_mode` candidate for ADR-0005.

## Exact-accumulation track (ADR-0005; not decided here)

- Human direction (`.factory/adr-split-args.json`): per-class mode stays the default; an exact mode where one reference
  GPU and every vendor compute the same bits; recipe precisions bf16/fp8 inputs, fp32 masters; int8 slices only as an
  internal exact accumulation. E-a ran (§Evidence); E-b, E-c are N/A under gpu false, pending a granted plan.
- `fold_mode = exact` (reserved, defined by ADR-0005): the fold is order-independent; `fold_groups`, `fold_radix`,
  SCHEDULER-I18 and D3's node order are then void, ownership is IWRR at 1/M, and Q7 is moot. E-a gains one arm for
  ADR-0005 (local, runs now): the histogram of `d` per dW element across M microbatches.

## Trade-offs

- Fleet independence costs `(G/w − 1)·P_s·4` on stages of a tree block narrower than G, width 1 included; the author
  scopes G to the class blocks planned wide (guide: G the largest planned width of the block, radix ordered by planned
  widths, `G | M`). Weighting grain is 1/G; unequal weights cost extra leaves. D3 holds 16–168 MiB per owner.

## Risks

- **R-narrow (finding 1).** A tree block planned at width 1 holds G accumulators: +11.0 GB at 1B, +86.8 GB at 7B
  (G = 8). ADR-0003's memory predicate (`4·G/w·P_s`) refuses the stage; the planner reports the refusal naming the
  block, and the author removes G from it or the planner widens the class. Planned at w = 2: +4.7 GB at 1B.
- **R-weight, R-heal, R-slice.** [1, 0.7], w = 2: +21.4% at `G = 2`, +6.2% at `G = 8`, +0.9% at 1/M. A heal runs at
  `(G/w)/⌈G/(w−1)⌉` of the rate until a spare (ADR-0004); any orphan rule combines group sums by the tree (RUNTIME-I4).
  Where P-slice fails on a class (E12), legacy bytes change.
- **R-audit (accepted round 2).** With `G < M` an owner never receives a per-microbatch tensor: CONTROL-I13's unit
  becomes the leaf partial. Leaf digests hash `k_p·P_s·4` B per peer per step, `M/(w·k_p)` times fewer bytes than
  per-microbatch digests; a dispute recomputes `n_p·M/G` microbatches. The hash (1.57 GB at 1B, w = 8, ≈ 0.8 core-s)
  exceeds 1% of the step unless parallel or offloaded: M4's ADR prices it (rule 10). No protection is removed.
- **R-migration-dep (finding 7, accepted in part).** In-flight legacy runs cannot resume on MC code whatever this ADR
  does: RUNTIME-I22 (ADR-0001 D11) refuses every pre-ADR `.pt` with `LegacyCheckpointError`, because it records no
  `recipe_fp` or numerics in force. Consequence: a legacy run finishes on pre-MC code or restarts from step 0. Rollback
  from MC code loses every step trained since the upgrade unless the operator restarts from the last pre-upgrade `.pt`
  (kept by policy). The lossless layout converter is real (below) but not the binding constraint; refuted: "element
  range alone breaks continuity".

## Falsification Criteria

Identity criteria pass only at N/N with N ≥ 20 plus 10 amplified reruns; one difference refutes. Experiment recipes
for E18c, E18e and E18e-8 set C = 4096 elements and f = C, and size each wide stage so `P_s ≥ 32·w·C` (E18e-8: d 256,
2 layers per stage, 1.6M elements per stage): every range spans ≥ 32 chunks, one chunk is ≤ 3.1% of the mean range,
and each flow carries ≥ 32 fragments (finding 4: a 2-layer d-64 model is one chunk per tensor and passes vacuously).

| Hypothesis | Experiment | Result that proves it WRONG |
|---|---|---|
| H-tree, H-fleet | E18b | `G = 8`, `[2,2,2]`, widths 1, 2, 3, 4, equal and [1, 0.7] ownership, two chunk splits, one tree block and one block without G: any sha256 differs from the one R |
| H-legacy | E18a | `G = M` differs from today's oracle bytes on today's integration recipes |
| H-volume | E18c | egress off `k_p·P_s·4·(w−1)/w + P_s·4·(w−1)/w`, or largest link off the mean, by > C elements + 5% |
| SCHEDULER-I18 | E18d | an out-of-order W within a group is not refused by `finalize()` |
| H-exposure | E18e | w = 2, `τ/c` = 3.35 and 0.42, `[2,2,2]`, `[8]`, `G = M`, equal and [1, 0.7] (cover rule and the worst equal-k cover): measured exposure off the per-leaf E from measured close times by > 5% of the step |
| H-drain | E18e-8 | w = 8, shim links (`transport/shim.py`) with `latency_ms` 0.5 and 5 (RTT 1, 10 ms), B set for `τ/c` = 5.86, F = F_min and F = 1: last-W-to-last-fold off `drain(F)` by > 5% of τ. The arm is void unless F_min ≥ 3 at the pinned RTT (else RTT is raised until it is) |
| H-memory | E18f | width 3, `G = 6`, `[3,2]` and `[2,3]`, one peer at 0.5; and a width-1 stage of a `G = 8` block beside a block without G: peak RSS above `memory` + resident model + 5%, or the block without G holding more than one accumulator |
| Fill model | E18g | w = 2, `G = 8`: residue against contiguous ownership off `(G/w − 1)(w−1)c/w` by > 5% of the stage |
| Element-range | E12 | AdamW single/foreach/fused, width 1 vs 3, `G = 3` and `G = M`, two splits: optimizer bytes differ |
| Manifest | E18h | `validation/state.py`'s distributed manifest under element-range, a range boundary inside a tensor: differs from `reference_manifest`; today's `names[p::w]` code passes on it (WOULD-FAIL-IF) |
| R order | E18i | R in leaf order and in index order: sha256 differs, or leaf-order peak accumulators exceed `L + 1` |
| Weighting | E11 (ADR-0003) | w = 2, one peer at 0.7, `G = 2` and `8`: O_het off the prediction by > 2 points (SIMULATED) |

## Local Experiment

CPU only (gpu false), under `experiments/hetero-native-per-class/`: E12; E18a pinned before MC; E18b–d, f, h, i at the
integration rung; E18e (3 processes) and E18e-8 (9 processes) on the transport shim's paced, delayed links, 20 steps
each, per-peer timestamps of every leaf close and last fold, SIMULATED next to the per-leaf E and `drain(F)`; E18g on
the scheduler simulator and the localhost stage. ADR-0003's E11 carries the `T_step` arms.

## Vast.ai Experiment

None: everything is tested on CPU classes and emulated links; real-link timing stays pending (gpu false).

## Migration Impact

| Change | Effect | Mitigation |
|---|---|---|
| `fold_mode`, `fold_groups`, `fold_radix` per class block (IN) | absent: `tree`, `G = M`, `[G]`: today's bytes; canonical IN omits defaults, so a legacy `recipe_fp` is unchanged | G < M is a new recipe with its own R |
| Element-range ZeRO-1 | flat fp32 range buffers; per-chunk `opt`; bytes unchanged where P-slice holds (E12) | where it fails, recorded per class |
| Optimizer state, checkpoint | a slot's checkpoint holds its range keyed by (tensor, chunk) | the canonical image (ADR-0004) stays per tensor |
| `validation/state.py` manifest | `names[p::w]` (lines 125-127) mis-keys range state | reassemble range slices into full-tensor AdamW state keyed by name, one `step` per tensor; E18h |
| Ownership | IWRR when `G = M`; residue lane sets otherwise (SCHEDULER-I18, I20) | `finalize()` refuses a split group |
| Exchange | push per microbatch becomes concurrent pull per leaf (D3) on tree blocks | `G = M` keeps today's push |
| Audit (CONTROL-I13, M4) | unit: a microbatch at `G = M`, a leaf partial at `G < M` | amended text in SPEC-control |
| R | leaf order; index order when blocks differ | E18i |

**Resume across versions and rollback.** MC checkpoints are `*.f1.pt` with `recipe_format`; pre-MC code refuses both
(RUNTIME-I22). MC code refuses every pre-MC `.pt` (`LegacyCheckpointError`, ADR-0001 D11) before any layout question,
so no checkpoint is ever loaded under the other layout. Rollback = revert the commits and restart from a checkpoint the
old code wrote (R-migration-dep prices the loss). Round 2's "conversion is ADR-0004's offline canonical-image path" is
withdrawn: ADR-0004 follows RUNTIME-I22 and offers no conversion.

**Text in other ADRs this round invalidates (not owned; reported):** ADR-0003 lines 24 and 176 "`G_c = k·w_c`" and
`Gth_s` at 2 B/param → "`G` and `fold_radix` per class block are the recipe's; `w ≤ G`; `Gth_s` at `b_m` B"; ADR-0004
line 123 "the default `G = w`" → "½ at an aligned `w = G`"; ADR-0004 lines 328-329 "resumed only for a recipe whose
bytes ADR-0001 leaves unchanged" → "refused (`LegacyCheckpointError`, RUNTIME-I22)"; DESIGN §11.6 heal step 2
(ADR-0004's text) "folded per chunk range at the range owners in index order" → "… by the `fold_radix` tree".

## Validation Plan

- **Invariants:** INV-4 (RUNTIME-I4, I18; SCHEDULER-I18), INV-13 (RUNTIME-I10, I11, I20); SCHEDULER-I6, I20; CONTROL-I6,
  I13; each test states `covers:` and `WOULD-FAIL-IF:`. **Rungs:** unit (lane assignment, leaf cover and exposure
  tie-break vs brute force; radix validation; `finalize()` refusals; node computation under scripted late and
  out-of-order fragments; orphan combine by the tree); integration (E12, E18b, c, f, h, i); E18a pinned before MC;
  identity (N ≥ 20 plus amplified reruns); bench (E18e, E18e-8, E18g, SIMULATED).

## Ideas considered

| Idea (source) | Verdict | Reason and number |
|---|---|---|
| Recipe tree over group sums, residue order (round 2; Higham §4) | **ADOPTED** | E(1) at every aligned width: 58 vs 191 ms (1B, w = 2); natural order adds a 28 ms fill |
| Scope G per layer block (round 3, first-principles lens; Megatron-LM, FSDP) | **ADOPTED at class-block scope** | removes +11.0 GB (1B) / +86.8 GB (7B) from width-1 blocks; the planner sets one width per class, so a finer scope buys nothing and a class block is already IN |
| Cover tie-break by earliest-closing leaf (round 3; Graham 1969) | **ADOPTED** as "smallest predicted E" | [1, 0.7], w = 2: 68.5 vs 102.3 ms; w = 3: 125.4 vs 144.2 ms |
| Exact fold of native fp32 dW, fp64 + TwoSum + expansion (round 3; Knuth; Ogita, Rump & Oishi 2005; Neal; Demmel & Nguyen) | REJECTED here, ADR-0005 candidate | loses stage + E at every priced point (368 vs 431, 456 vs 497, 364 vs 376, 197 vs 307 ms); wins width-1 memory (1.57 vs 11.0 GB) |
| Recipe M / T_mb as alternative (round 3; Zero Bubble PP) | REJECTED as default, recorded | +42 ms fill per upstream stage (1B) beats its 14 ms gain at w = 8; wins only for one-stage recipes |
| Lossless whole-tensor → range checkpoint converter (round 3; `state.py:125-127`; Megatron dist_checkpointing) | DEFERRED to ADR-0001 | lossless (AdamW elementwise, one shared `step`, no arithmetic) but RUNTIME-I22 refuses the legacy `.pt` first; the reverse direction would defeat D11's rollback refusal |
| Recursive-halving reduce-scatter for above-peer nodes (round 3; Thakur, Rabenseifner & Gropp 2005; OPEN-1B TreeFold) | REJECTED | same bytes; in-flight memory is the bandwidth-delay product either way (124 vs 168 MiB, w = 8, RTT 10 ms); adds 2·RTT (20 ms) at w = 8; only power-of-two binary radices and k = 1, so D3 stays the one mechanism. Related work: the tree is TreeFold's pairing |
| Real delay emulator, dummynet/netem (round 3; Rizzo 1997) | REJECTED, intent ADOPTED | the transport shim already delays and paces per link on macOS and Linux without root; E18e-8 pins its `latency_ms` |
| H2 MRE as positioning (round 3) | **ADOPTED** | verified in full text this round (MRE < 1.5%, worst 1.215%, 20B, 300 iterations; 1,024 chips, 4 architectures): the tolerance standard the product rule rejects |
| OPEN-1B 64-bit fixed-point fold for small bounded tensors (round 3) | REJECTED | needs a recipe scale per tensor; norms and biases are < 0.1% of `P_s`, so the gain is < 0.1% of bytes |
| OPEN-1B step-level many-auditor replay (round 3) | DEFERRED to ADR-0001 D5 / M4 | identity-rung cost, not the fold order; compatible with leaf-partial audit units |
| Ozaki-II pass count in E-a (round 3; 2504.08009) | DEFERRED to ADR-0005 | exact-mode GEMM cost, not this decision |
| HAP5, Tessera as `/wan-scalebench` baselines (round 3) | not placed | not verified |
| Reference corrections (round 3, both lenses) | **ADOPTED** | AccPar, HETHUB, HAPT moved to verified; RepDL affiliation removed; H2 numbers from full text |
| Rounds 1–2: D3 window (ADOPTED, re-derived); leaf-partial digests (ADOPTED, CONTROL-I13); interleaved `[G]`, rotated order, `G = k·w`, chained, binned, consumer-dtype gather (REJECTED); integer-limb dW, limb compression (ADR-0005); Cooper, FP4, DASH, Jetfire, GeoMesh (ADR-0001/0005, positioning) | unchanged | numbers above and in `docs/research/hetero-review-record.md` |

## Changelog

- **Before.** Unsplit ADR-0001 rounds 2–3 (`docs/research/hetero-review-record.md`). Round 1: `G = k·w` withdrawn, G a
  recipe constant, `E(k)`, gather at 4 B. Round 2: `fold_radix`, residue tree, D3 pull, CONTROL-I13 leaf audit.
- **Round 3.** Finding 1: width-1 bound derived, G scoped per class block (DESIGN §11.3, §11.6; SPEC-runtime Fold),
  E18f width-1 arm. Finding 2: per-leaf E; exposure tie-break (SCHEDULER-I20); [1, 0.7] 102 → 68.5 ms. Finding 3:
  `drain(F)` re-derived; pinned RTT; validity condition F_min ≥ 3. Finding 4: experiment C, f and model size fixed.
  Findings 5, 6: exact-fold and recipe-M rows priced. Finding 7: R-migration-dep; the false ADR-0004 conversion claim
  withdrawn. Blocker (orphan sums "in group order"): ADR-0004 round 3 removed that sentence (reroute deferred) before this
  edit; RUNTIME-I4 now binds any orphan rule to the tree. INV-4/RUNTIME-I4 and R's "two accumulators" fixed; R leaf order, E18i. References re-verified; H2 numbers restored from full text.

## Recommendation

**PROCEED** with the class-block residue radix tree and element-range ZeRO-1, local experiment first. It removes the
incast, reaches native volume and `E(1)` at every aligned width, keeps every legacy recipe's bytes, beats every priced
alternative on stage + exposure at 1B, 100 Gbps, and computes the same bits on every fleet. Its price is memory on
narrow stages of a tree block, confined by the class-block scope. Do not build Q7's variant or `fold_mode = exact`
without their rulings.

**Open question for the human.**
- **Q7 (ruling).** May a wide stage's width and planned peer weights join C4's stage-to-class map? **No** (proposed):
  G and `fold_radix` are recipe constants per class block; a tree block narrower than G pays `(G/w − 1)·P_s·4`
  (4.7 GB at 1B, w = 2; 11.0 GB at w = 1, refused at 7B) and grain 1/G ([1, 0.7]: +6.2%, E 68.5 vs 58 ms).
  **Yes:** each wide stage uses the weighted IWRR partition at `G = w`, with exact text: DESIGN §11.6 IN
  "`fold_partition` per wide stage (`iwrr(M, planned weights)`, written by the planner)"; §11.3 and SPEC-runtime Fold
  "microbatch `mb` is in group `fold_partition[s][mb]`"; SCHEDULER-I20 "the planner writes `fold_partition` from
  measured speeds"; CLAUDE.md reference list "the stage widths and planned peer weights of wide stages (ADR-0002)".
  Each fleet then has its own R (OPEN-1B's mesh-defined tree is this shape). **Third branch (later):** a `fold_mode`
  from ADR-0005 (exact fold of native dW at 2× volume, or exact integer dW at ×1.34–1.91) makes the question moot.

## Human Approval
Granted only by `node agents-architetura/tools/factory/approve.mjs grant` in a human's terminal; recorded in docs/design/APPROVALS.json.
