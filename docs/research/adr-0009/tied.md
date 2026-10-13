# ADR-0009 D6 derivation: a tied table across pipeline stages, in the stock order

Notation: table `W` (V×h), owned by the Head stage (the stage holding the epilogue unit); stage 0 holds the prologue
unit. Microbatch i gives the Head use `h_i = ∂ℓ_i/∂W` through the output projection (dense, every row) and the
embedding use `e_i` through the lookup (rows of the ids of microbatch i, `T_i`, unique, `|T_i| ≤ T_mb`).

## 1. The order to keep (FACT, code and measurement)

The oracle (`validation/reference.py:77-88`) and every stock single-device run take one `autograd.grad` per
microbatch over the tied Parameter, so autograd adds the two uses (`g_i = h_i + e_i`, one IEEE add, commutative), then
fold in index order: `G = (((g_0) + g_1) + …) + g_{M−1}`. Holding `h_i` until `e_i` arrives keeps this order but costs
one dense table per microbatch in flight (`n_pend·V·h·4` B; 5.6 GB per table at Gemma3-27B in fp32). The per-use fold
`(Σ e_i) + (Σ h_i)` (round 0) is cheaper but is another function: 107,254 of 163,840 elements differ
(`out/split2-rerun-r1.txt`), 4,592–4,731 of 262,144 at V 4096 (`out/tied_rowsparse-r1.txt`).

## 2. Row-sparse pending fold (exact by construction)

The dense embedding backward writes `+0` to every row outside `T_i` (MEASURED: every case of
`out/tied_rowsparse-r1.txt`, sign bit clear). For such a row `g_i[r] = h_i[r] + (+0)`, which the Head computes
without `e_i` (the add maps `−0` to `+0` exactly as the oracle's add does). Elementwise, the fold of row `r` only
needs every earlier contribution to `r` applied first. So:

- `P` = the union of the rows stored by microbatches whose `e` has not arrived.
- On `h_i`: rows outside `P ∪ T_i` fold now (`acc[r] = acc[r] + (h_i[r] + 0.0)`, or `h_i[r] + 0.0` for i = 0);
  rows of `W_i = P ∪ T_i` are stored as `h_i[W_i]`.
- On `e_j` (frames in microbatch order), resolve the oldest pending microbatches in order: `acc[R] = acc[R] +
  (h_j[R] + e_j[R])` (for i = 0 the assignment), where `e_j[R]` is `+0` outside `T_j`.

Every element then receives exactly the oracle's adds in the oracle's order: MEASURED equal (sha256-level
`torch.equal`) to the stock fold at n_pend 1, 2, 3 and M = 6, for gemma2 (softcaps), llama and qwen3, V 128 and
4096, fp32 and bf16 (`scripts/tied_rowsparse.py`). Stage 0 checks that `e_i` is `+0` (sign bit clear) outside `T_i`
before it ships `e_i[T_i]`; a violation is fatal (`TIED-SPARSE`), so a module whose backward writes other rows (a
tied table used as a dense matrix on the input side) is refused rather than folded wrong.

Pending memory on the Head: `Σ_pending |W_k|·h·4 ≤ n_pend·min(V, n_pend·T_mb)·h·4` B, plus the dense accumulator and
one transient dense `h_i`. Worst case at Gemma3-27B (n_pend 8, T_mb 4096): 5.6 GB, one table's size, against 45 GB
for dense pending tables. Measured peak at V 4096, T_mb 48, n_pend 6: 636 rows (bound 1,728).

## 3. Placement and bytes (one direction per microbatch, one burst per step)

Stage 0 holds a working copy of the table, NaN-filled except the rows `U_t` of step t's ids (all known from
`make_batch(data_seed, step, mb)` on every stage, so no index crosses the wire). It runs HF's own embedding module
(a gather: byte copies, the oracle's bits) and the module's own backward (dense, transient), and sends `e_i[T_i]`,
`|T_i|·h·4` B, to the Head. After its AdamW the Head sends the rows `U_{t+1}`, ordered by first use (`T_0` first),
`|U_{t+1}|·h·4` B. Alternatives: the Head computing the embedding output per microbatch (round 0) moves `T_mb·h·4` B
in both directions per microbatch (`≥` this scheme's bytes, since `|T_i| ≤ T_mb` and `|U| ≤ M·T_mb`); the Head
shipping the summed table per step moves `2·V·h·4` B (3.8 s at 10 Gbps for Gemma2-2B).

## 4. Path and exposed time (chain only; DESIGN §10 unchanged)

The frames ride the existing chain sockets, forwarded unchanged by every stage between stage 0 and the Head
(RUNTIME-I16 untouched: no socket outside `{rank−1, rank}`). Per hop, the forward direction carries up to
`T_mb·h·4` B more per microbatch (the activation's own size under the fp32 cut, ×2 worst case) and the backward
direction `|U|·h·4` B more per step.

The step's critical path gains one round trip between stage 0 and the Head: stage 0's last B → `e_{M−1}` relayed
forward → the Head's fold and AdamW of the table → rows `T_0` relayed back → stage 0's first F:
`T_tied ≈ 2(S−1)·L + (|T_{M−1}| + |T_0|)·h·4/B + t_AdamW(table)` (chunked forwarding; store-and-forward adds
`(S−2)·chunk/B`). Gemma2-2B (h 2304, fp32 table, `T_mb` 4096, worst `|T| = T_mb`), L 20 ms, B 10 Gbps:
S 4: 120 + 60 + 8 ≈ 188 ms; S 8: 280 + 60 + 8 ≈ 348 ms (HYPOTHESIS until E4 measures it). A direct Head↔stage-0 link
would make it `2L + …` (108 ms) and is a topology decision outside this ADR; a plan that puts the prologue and
epilogue units on one stage moves no frame. The planner charges `T_tied` and the per-hop bytes (SCHEDULER-I22), so it
trades stages against this term like any other.

Gain disappears: with `M` small and `L` large the round trip dominates the step; the planner then prefers fewer
stages, and refusing tied models at S > 1 remains the human's alternative.
