# The recipe fold tree and element-range ZeRO-1 (research note)

The heterogeneous track is paused, and the `hetero-*` files this note cites are not part of this repository.

Cited by: ADR-0002. Extracted verbatim from the unsplit ADR-0001 round 3 (§D3 fold tree and element-range bullet,
the `spread.py` analysis, and the rejected alternatives with their numbers) on 2026-10-08, when the human asked for
the ADR to be split.

Section references inside the extracted text (D1–D10, §Mathematical Model, §Migration Impact, App. A, E1–E19,
Q0–Q7, "round N") refer to the unsplit ADR-0001 round-3 text, archived byte-for-byte at
`hetero-adr-0001-round3-unsplit.md` (sha256
`6a67784175b4813fdde751b9a9afb182595b7c64c38659e16b0b0eabc1bac2ba`). The split map in
`hetero-review-record.md` says which ADR now decides each item. This note decides nothing.

## Fold tree (unsplit §D3)

**The fold tree (round 2, replaces "fold in microbatch-index order").** The cause of round-2 findings 1 and 6 is that
U3 as written (one left fold over all M) forces a wide stage to ship every microbatch's gradient to the tensor's owner.
That is `(M/w)·P_s·4·(w−1)/w` bytes per peer per step, M/w times a native reduce-scatter. The reduction order is part of
the recipe (CLAUDE.md), so the recipe may choose a tree that a wide stage can evaluate locally:
- **Groups are residue classes**, `g = mb mod G`. The upstream stage emits microbatches in index order, so each group's
  microbatches reach a peer in increasing index. A peer folds them as they arrive and keeps one accumulator per group
  it owns. Contiguous groups would serialize the stage, or force buffering whole gradients, so residue groups are the
  only choice.
- **Ownership is by whole groups.** Peer k of a width-w stage owns `n_k` whole groups, `Σ n_k = G`. Throughput weights
  act on group counts: the planner picks `n` to minimize `max_k n_k/v_k` for measured peer speeds `v_k`. With `G = w`
  every peer owns one group, which is today's equal-weight IWRR; finer weighting needs `G = k·w` (round 3, below).
- **Traffic.** Each peer ships one partial per owned group, sliced to the chunk-range owners: `(G/w)·P_s·4·(w−1)/w`
  bytes. Add the bf16 parameter gather `P_s·2·(w−1)/w`. At `G = w` this equals native ZeRO-1, and with element-range
  ownership (below) every link carries that volume within one chunk. Owners combine the incoming partials in group
  order, streaming by fragment, so no whole partial is buffered.
- **Memory.** Each peer holds `G/w` accumulators (`4·P_s` bytes each). At `G = w` that is one, the same as today.
- **R** evaluates the same tree at width 1 in group-major order (all of group 0, then group 1, …). Order is free by U2,
  so R needs two accumulators.
- **Legacy.** `G = M` (one microbatch per group) and `G = 1` both reduce to today's left fold over all M. A recipe without
  `fold_groups` is read as `G = M` and keeps today's bytes and today's per-microbatch shipping.
- **Default and choice (round 3).** The planner sets `G_c = k_c·w_c` with `k_c ∈ {1, 2, 4}` (and `G ≤ M`), the k
  that minimizes the predicted `T_step`: k = 1 gives native exchange volume and equal weighting; larger k buys finer
  weighting and gentler width-elastic heals (D8) at k partials per peer. With equal peers k = 1. Each choice is a
  recipe with its own R.
- **What weighting is worth** (`spread.py`, App. A, MEASURED script output, M = 32): two peers at speeds [1, 0.9] lose
  5.6% at `G = w`, 0.9% at `G = M`; [1, 0.7] lose 21.4% at `G = w`, 6.2% at `G = 4w`, 0.9% at `G = M`. Four peers at
  [1, 1, 1, 0.9] lose 8.3% at every G ≤ M, today's IWRR included: one microbatch is 1/8 of a peer's work, coarser
  than a 10% spread. So group weighting matters only when the spread exceeds about `w/M`, and below that today's
  IWRR gains nothing either.
- **Cost against a homogeneous run:** 0 at `G = w` against an equally wide homogeneous ZeRO-1 stage. The exchange
  itself is priced in T_step on both sides (§Mathematical Model), so the planner widens a stage only where it pays.
  Predicted effect of the tree against today's shipping on the round-2 searched plans (round-2 model): G = M cost +0
  to +125% of the step. The volume ratio (M/w partials against one) is model-independent; the step-time figure came
  from the withdrawn fill model and is re-measured by E11's `G = M` arm.
- **Scheduler constraint (SCHEDULER-I18).** Within one peer, the W units of one fold group run in ascending microbatch
  order. Today `OrderedAccumulator` buffers out-of-order contributions. Under the tree, an out-of-order W would hold a
  whole gradient until its predecessors arrive. The solver therefore emits them in order, and `finalize()` refuses an
  order that does not.

## Element-range ZeRO-1 (unsplit §D3, "What each piece costs")

- *Element-range ZeRO-1 (round 3, replaces `params[k::w]`).* Round 2 kept today's whole-tensor sharding
  (`stage.py:116`, `self.params[self.peer :: self.w]`) while pricing an even `P_s/w` split. Under whole-tensor
  ownership the owner of the embedding or head (`V·h` = 262.7M parameters at 1B) receives `(w−1)` full partials of it
  and holds its AdamW state alone: at 1B, w = 8, 100 Gbps that is 7.36 GB in, 589 ms, against a modelled 110 ms.
  Now the stage's flat chunk list is split into w contiguous ranges of nearly equal element count (within one chunk),
  and peer k owns range k: its optimizer state and its `sumsq` and `opt` units. Fold, `sumsq` and `opt` are per chunk,
  so where a range boundary falls never changes a summation, and ranges are fingerprint OUT. The largest per-link
  volume exceeds the mean by at most one chunk (4 MiB fp32, 0.34 ms at 100 Gbps), so the exchange and memory terms
  of §Mathematical Model now describe the mechanism. Single, `foreach` and `fused` AdamW are elementwise, so the
  implementation is pinned (IN) and must pass RUNTIME-I11 per chunk.

## Within-class speed spread against fold groups (`spread.py`)

Script: `hetero-scripts/spread.py`, sha256
`4c22a457fb62d0c708a0300ba98ab575412dd4396de2e26afb60a392e25c6342` (equal to the hash the unsplit ADR recorded;
checked with `shasum -a 256` during the split). Run: `python3 -I docs/research/hetero-scripts/spread.py` (exit 0 in
round 3; not re-run in the split). Output (round 3, MEASURED script output, an analytic model):

```
[1.0, 0.95] G= 2:  +2.6% | G= 4:  +2.6% | G= 8:  +2.6% | G=32:  +2.6%
[1.0, 0.9] G= 2:  +5.6% | G= 4:  +5.6% | G= 8:  +5.6% | G=32:  +0.9%
[1.0, 0.7] G= 2: +21.4% | G= 4: +21.4% | G= 8:  +6.2% | G=32:  +0.9%
[1.0, 1.0, 1.0, 0.95] G= 4:  +3.9% | G= 8:  +3.9% | G=16:  +3.9% | G=32:  +3.9%
[1.0, 1.0, 1.0, 0.9] G= 4:  +8.3% | G= 8:  +8.3% | G=16:  +8.3% | G=32:  +8.3%
[1.0, 0.97, 0.94, 0.9] G= 4:  +5.8% | G= 8:  +5.8% | G=16:  +5.8% | G=32:  +5.8%
```

## Alternatives to the tree, with their numbers (unsplit §Ideas considered, rounds 2–3)

| Idea (source) | Verdict | Reason and number |
|---|---|---|
| Recipe-declared fixed fold tree (alternatives lens; frontier lens; Demmel–Nguyen; Higham §4) | **ADOPTED (D3)**, as residue groups `mb mod G` rather than contiguous blocks | contiguous groups would serialize a wide stage (microbatches arrive in index order) or force buffering whole gradients; residue groups arrive in order and give today's IWRR at `G = w`. Exchange = native ZeRO-1 at `G = w`; saves 0–56% of the step vs `G = M` (App. A) |
| Chained ordered fold (first-principles lens; Patarasuk & Yuan) | REJECTED | bytes would be right, but peer k must hold its `M/w − 1` later gradients until the running sum arrives: `(M/w − 1)·P_s·4` B (37 GB at the 7B, w = 8, M = 32 point), and the chain puts `(w−1)·P_s·4/B` on the critical path. The fold tree reaches the same volume with one accumulator |
| Order-independent binned fold (ReproBLAS K-fold; consistency lens) | REJECTED | any tree, but k ≈ 3 accumulators per element (+8 B/param, 25 GB on a 3.1B-param stage) and ×k exchange volume; the fold tree at `G = w` has native volume and one accumulator. Gain over the tree: only free ownership granularity, which `G = k·w` buys more cheaply |
| `G = k·w` per class from measured peer spread (alternatives lens) | **ADOPTED (D3, D9; SCHEDULER-I20)** | `spread.py`: [1, 0.7] loses 21.4% at `G = w`, 6.2% at `G = 4w`; [1, 0.9] loses 5.6% at `G = w`. Cost: k partials per peer, priced in `R_s` |
| Element-granular ZeRO-1 over recipe-fixed chunks, per-chunk `sumsq` in canonical order (frontier lens; Megatron distributed optimizer) | **ADOPTED (D3, D4.2; finding 2)** | makes the even-split exchange and memory terms true; the head owner's incast at 1B, w = 8, 100 Gbps falls from 589 ms to the modelled 110 ms ± one chunk (0.34 ms) |
| Fleet-independent `G` constant, e.g. 12 or 24 (consistency lens) | REJECTED | at w = 2, `G = 24` ships 12 partials per peer, 12× native volume; round 2 priced `G = M` (16× at w = 2, M = 32) at up to +125% of the step. Its weighting gain is bounded by the spread table above, and the planner's `G = k·w` buys it where it pays. The "planner writes an IN field" concern is not circular: class blocks are planner-chosen and IN too, and the fingerprint is computed after planning |
