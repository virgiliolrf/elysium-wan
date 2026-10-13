# ADR-0008: Exact Mixture-of-Experts layer: an exactness skeleton and an enumerated routing recipe

<!-- Approval lives in docs/design/APPROVALS.json (agents-architetura/tools/factory/approve.mjs), bound to this exact text. Do not write a
     Status line claiming approval. Mark the ADR "Status: REJECTED", "Status: SUPERSEDED" or "Status: WITHDRAWN" only
     when that is the human's decision. -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE DECISION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: the human asks that MoE models train under FSDP (ADR-0006) and every existing mechanism, tested, in M4 "any
LLM, dense and MoE" (decision 5), also under ADR-0009's HF-modules adapter (in this tree, awaiting its grant). No MoE
model runs today: an MoE layer puts a data-dependent choice inside every microbatch's gradient.
Current bottleneck: correctness, not time. On one CPU device map even HF's eager block is exact between oracle and
stage (40/40, MEASURED on a62e4eb). What breaks it: an expert GEMM whose rows a path splits or joins (P3); a `None`
gradient for an unrouted expert, which both guards refuse and wide stages cannot carry; a global aux scope; RNG;
a selection undefined on ties (`topk`'s tie sets on identical keys: CPU vs MPS 436/20,000; stable sort 0);
`index_add_` refused on MPS. Then ADR-0006's S1: bytes scale with total parameters, compute with active (ρ 3.68–10.95).
Proposed architecture (one decision: what an MoE layer computes, so that existing mechanisms carry it):
1. *Exactness skeleton* (D0, D1, D5–D7, D9): one expert call per (microbatch, expert) on exactly its rows in token
   order, also on 0 rows (+0 gradient; AdamW steps every expert); every sum outside one kernel call has a written
   left-fold order; every scatter is a byte move; no atomics, no RNG.
2. *Enumerated routing recipe* (D2–D4): score, key, groups, renormalization, scaling, weight placement, biases,
   clamped SwiGLU; the family's router dtype; ties by the stable descending sort. It expresses Mixtral, Qwen2/3-MoE,
   OLMoE, DeepSeek-V3/R1, Kimi K2, GPT-OSS and Llama 4; in M4 only Mixtral and Qwen2/3-MoE have a checkpoint path
   (under 6a through ADR-0009 D7's hub-to-fused map, decision 10 (i)); the other rows serve from-scratch recipes.
3. *Aux terms per layer and microbatch* (D8); *selection-bias update* (D10) from int64 counts in a lane of the
   existing gradient frames, the bias in the identity state.
4. *Parallelism inherits it* (fold, deferred W, ZeRO-1, pieces for elementwise optimizers); one exchange unit per
   projection (or per 3-D expert tensor). Expert parallelism is not decided here.
5. *HF boundary and loader* (decision 6): under ADR-0009 HF's objects and names stay; each sparse-MoE block and its
   router and experts children are class-swapped to subclasses computing this layer, by a table keyed on transformers
   5.19.0, under the HF-MoE seam rule (decision 11, CLOSED); no native HF loader. Else: the allowlist loader.
First-principles reasoning: a float result is fixed by its operands, operations and order; the same code on the same
microbatch gives the same bits on one device map (P8, P9). The layer pins what code leaves to a kernel, the engine,
the backend or the topology, and makes each microbatch's gradient dense, as the fold, ZB, ZeRO-1 and pieces need.
Mathematical justification: dropless routing gives Σ_e n_e = T·k (expert FLOPs skew-independent); egress per peer
`b_g·P_s·(w−1)/w` (total), compute `c_s = 8·P^F_s·T_mb/F` (active): S1 needs T_mb ≥ ρ_s × 1,500 at 400 Gbps, w 4,
conservative (`c_s` omits launches). D10's lane ≤ 184 KB per microbatch per peer. HF v5: dW in B (t_B ×1.40–1.89),
fp32 bank gradients held B→W (5.25 GiB per Mixtral layer and microbatch, × the order's in-flight count `m_BW`), S1
×1.91–1.97, +3E launches (Qwen3 1.92 ms).
Expected improvement: from "no MoE model runs" to seven families' MoE layers byte-identical to the oracle at every
depth, zero-bubble order, recompute, ZeRO-1 and piece width; HF configs and hub checkpoints under ADR-0009.
Alternatives: refuse MoE; HF's block as-is, tuned, or v5 hook; dense-masked; capacity factor; grouped GEMM; EP; 3 rows.
Risks: GPU facts (R1); launches (R5); the ADR-0009 seam (R9; R13, R14 CLOSED by decisions 10, 11); rows (R10).
Falsification criteria: a sha256, CE or aux hex difference in a true arm (F1); a firing must-fail that does not
diverge (F2); pending-W, held, exchange, unit or count-lane bytes off (F3, F4); a 6b map missed (F5); the layer > 1.15×
HF's block on CPU (F6); a stage faster than its `c_s` (F7); HF block ≠ `MoEMLP` (F8). Local experiment: CPU, tiny
models, one arm per row; depths 1/2/4, ZeRO-1 1/2, pieces 1/2/3, fp32/bf16, recompute, skew, must-fails, HF (E1–E7).
Estimated Vast.ai cost: none: local only. GPU kernel facts (R1) and the grouped-GEMM question (R5) wait for M1.
Recommendation: PROCEED (local experiment first); decisions 1–11 decided by autonomy, 2026-10-11 (§Recommendation).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Awaiting explicit human approval: node agents-architetura/tools/factory/approve.mjs grant docs/design/adr/ADR-0008-moe-exact.md docs/design/DESIGN.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Problem

Train MoE decoders (from scratch or a checkpoint) under PP, zero-bubble and sharded stages, byte-identical to the
reference, natively and under ADR-0009's HF modules; EP is separate (A5). Native: MLA, sinks, banded attention, NoPE,
MuonClip, MXFP4 refused by name (derivation §3); HF: attention is ADR-0009's.

## Current Architecture

FACTS (a62e4eb; M2b db3f4e0, merged into `factory/m4` as the snapshot 8dee0b9, `runtime/pieces.py:106`; lines in derivation §18): `wgrad.convert` makes every
unshared `nn.Linear` a `SplitLinear` (`(X, dY)` per call, `flush` in store order); the oracle's `autograd.grad` has no
`allow_unused` and `stage.py:1291` raises on a missing gradient; ZeRO-1 peer `k` owns `params[k::w]`; M2b's units are
one per `SplitLinear`, the Embed's fp32 tables shipped when W(mb) starts. ADR-0009 (in this tree): D2 trace,
D5 own-forward guard, D7 loader (hub-to-fused map), D8 this ADR's seam, D9 gate, pin 5.19.0. At 5.19.0 each block
returns one tensor and calls `gate` then `experts` (`modeling_mixtral.py:126-127`); `*TopKRouter` holds `weight`;
`*Experts` holds `gate_up_proj` (E, 2f, h) and `down_proj`.

## Current Bottleneck

Correctness, not time: MoE cannot be expressed (derivation §12); once exact, S1 limits sharding (bytes count all E).

## Evidence

MEASURED (Apple M4, torch 2.14.1, deterministic algorithms; scripts, outputs and sha256 in derivation §1–§2).
- On 5f2c815 (HYPOTHESIS on a62e4eb until E1; `wgrad.py` is blob 86c0404 on both): P1 a 0-row expert call gives an
  all-+0 dW on both paths; P2 AdamW with `None` skips, with zeros steps; P3 row count and dW split change GEMM bits
  (462/512; 115,955/131,072); P4 bf16 router, E 64, k 8: 4.25% boundary ties; P5 a reversed k-fold differs on 100% of
  tokens at k ≥ 3; P6 39/39 pipeline arms identical; P7 pieces at `Q_t` 16/16 identical, off `Q_t` 2/2 diverge.
- On a62e4eb (`probe3.py`, CPU and MPS): P8 `topk` on both paths 14/14 identical; P9 an HF-style eager block on both
  paths 40/40 identical; P10 `topk` CPU vs MPS 436/20,000 sets differ, stable sort 0; P11 MPS `index_add_` (unique
  indices too) and `scatter_add_` raise under deterministic algorithms; `torch._grouped_mm` = per-segment mm.

## Hypothesis

H1: under the layer below, every recipe row's per-microbatch gradient is a dense function of (microbatch, parameters),
so the fold, deferred W, ZeRO-1 and pieces reproduce the oracle bit for bit at every depth, order and width.
H2, per path. Native, against the v4-style eager loop (`probe3.py`): no byte, no host sync, no expert GEMM (only
O(T·k·h) elementwise work). HF v5, against transformers 5.19.0's own block: no byte, no sync, but two f-wide GEMMs for
its one 2f-wide add E launches in F, 2E in B, E in F′ (derivation §20). Grouped GEMM: 12(E−1) fewer (A4, M1).
H3: S1/S2 with the analytic `c_s` admit MoE pieces only at `T_mb ≥ ρ_s`× dense and stay conservative for MoE.

## First-Principles Derivation

The oracle and every stage run the same module code on the same microbatch, so a difference can come only from a
choice the code leaves to a kernel, the autograd engine, the backend or the topology. The layer removes each.

**Skeleton (normative, every recipe).** Input `x̂ = ln2(x)`, (b, t, h) flattened row-major: T tokens, token = row.
- D0 Reductions. Inside one kernel call on a fixed shape the order is the kernel's, the same on every path of one
  device map (as every GEMM's; R1). Every sum assembled outside one kernel call (autograd accumulation, a scatter,
  microbatches, peers, slots) has a written order; atomics are never used (P11).
- D1 Fan-out. A tensor with ≥ 3 autograd consumers passes through one function whose backward is the left fold of
  their gradients in a written order, the recipe's, not the engine's (CPU structural, P6; CUDA threads HYPOTHESIS).
- D5 Dispatch. Slots sorted by (expert, token) ascending; one gather gives each expert its rows in ascending token
  index. Backward: a token's k slot gradients gathered and folded in ascending expert index.
- D6 Experts. One call per (microbatch, expert) on exactly its `n_e` rows, every expert called (0 rows when
  `n_e = 0`, P1); rows never concatenated across microbatches, padded, capped or reordered (P3).
- D7 Combine. `o_t = (…(w_t1·Y_t1) + …) + w_tk·Y_tk` in fp32, ascending expert index, plus the shared expert's output
  in fp32; one cast to the residual dtype. Backward: a gather through the inverse slot permutation; `dw_tj = Σ_h
  g_t·Y_tj` (one kernel, D0).
- D9 Unrouted experts contribute +0 (D6); AdamW steps every expert. HF's skip is exact on a pipeline (P9) but needs
  both guards weakened and an absent marker on wide stages (A7). No RNG anywhere.

**Routing recipe (normative, enumerated; per-family values in derivation §3).**
- D2 Router. `logits = x̂·W_rᵀ (+ b_r)`; `router_dtype` ∈ {`compute` (under autocast, as HF's gate), `fp32`
  (autocast off, fp32 operands)}, default the family's own: compute for Mixtral, Qwen2/3-MoE, OLMoE, GPT-OSS,
  Llama 4 and from-scratch recipes, fp32 for DeepSeek-V3/R1 and Kimi K2. `W_r` is an fp32 master in an `nn.Linear`
  subclass (`wgrad.convert` leaves it; dW in B). `score` ∈ {`softmax` over E, `sigmoid`, `softmax_k`}, fp32.
- D3 Selection. Key = `logit`, `score` or `score + bias_l` (fp32; `bias_l` a per-layer E-vector of state, never
  differentiated). Groups (G > 1 only if G divides E, E/G ≥ 2, 1 ≤ G_top ≤ G and k ≤ G_top·E/G; else refused): group
  key = sum of the group's top-2 keys, groups chosen by the stable descending sort. Then the first k indices of the
  stable descending sort of the key (larger value, then lower index; NaN fatal), re-sorted ascending. The reason is
  definedness, not identity (per device map `topk` is exact, P8): selection is a written function of the keys, ties
  and ±0 included; `topk`'s tie order is unspecified (on identical keys CPU and MPS differ, P10). It does not make
  selection equal across backends, whose keys already differ (R1). Cost: a sort over E.
- D4 Weights. `p_tj` = the score at `e_tj` (`softmax_k`: of the k selected logits; the bias never enters). With
  `norm`: `w = p / ((…(p_t1 + p_t2) + …) + p_tk + norm_eps)`; then `w ← w·scale`. `apply` = `output` (D7) or `input`
  (k = 1). Expert: SwiGLU, or clamped SwiGLU (`limit`, `alpha`, `(up + 1)`) with biases (`db` over D6's rows). Shared
  expert: `f_sh` (0 = none), gate `sigmoid` or none.
- D8 Aux terms, per microbatch and layer, gradients injected as their coefficients by an identity function on the
  layer output in the reference and the stages alike: `lbl` (scope `layer` or `sequence`) and `z` (derivation §5),
  recorded as hex. `router_aux_loss_coef` (a pooled form, k/L_m as strong per layer) is never mapped;
  `output_router_logits = true` is refused unless the recipe sets aux fields.
- D10 Selection-bias update (`bias_rate` γ > 0). After the step's AdamW: `c_i` = int64 counts over the global batch,
  `bias_l,i += γ·sign(T_g·k − E·c_i)` (sign in integers, 0 on equality). Each microbatch's counts ride to the bias's
  owner in an int64 lane of the frames already carrying that microbatch's gradients (zero1: owned after the trainable
  names; pieces: in the block's small-tensor unit), folded by integer add; the owner updates and the existing
  broadcast or gather returns it before step t+1's F (derivation §6). In checkpoints, manifests and `check_frame`
  (int64, shape, step, mb; zero1 `Σ c = T_mb·k`).

**What parallelism needs.**
- Zero-bubble: native, B records `(X_e, dY_e)` per expert call and W runs that GEMM (MODEL-I1); routed pending W
  `T·k·(2h + 3f)` values per layer per microbatch, any skew. HF v5: the banks' dW runs in B (routed pending W 0), so
  B leaves each bank's dense gradient, `b_m·E·3hf` bytes per MoE layer (`b_m` 4 today), live from B(mb)'s end until
  W(mb) starts; a peer holds it `m_BW` times, its microbatches between B and W in the order (I21, derivation §23.1).
  Recompute re-runs D2–D8 on the same bits; the stage compares rebuilt aux values and counts with F's (a tripwire).
- ZeRO-1 and pieces: every expert tensor has a dense per-microbatch gradient (D9), so ADR-0006's argument holds for
  elementwise optimizers (P7); Muon needs whole-matrix units (derivation §7).
- Exchange units: native, one unit per projection over the E experts in expert order (3, not 3E); HF v5, each 3-D
  bank one unit (2), cut on dim 0 at `Q_t = lcm(trailing product, 128)` (DERIVED; P7 measured 2-D only), its fp32
  gradient made whole in B, held (`b_m·E·3hf` per layer per microbatch in flight) and shipped when W(mb) starts;
  shared-expert `SplitLinear`s one unit each; router, bias small.
- Planner: bytes and memory at total parameters; HF v5: `mem_b` adds `b_m·Σ E·3hf` and `mem_w` frees it, so the
  solve's replay counts the hold `m_BW` times, and I16's memory condition adds `m_BW,k·b_m·P_bank,s` (I21). Compute
  at active parameters in the analytic `layer_costs` and `c_s`, which omit launches and syncs (derivation §15).

**HF boundary (ADR-0009's D8 seam; derivation §14, §23.2).** HF's module objects, names and layouts stay; classes are
swapped (`module.__class__ = Sub`, as `wgrad.convert` swaps `SplitLinear`) to registered subclasses whose own
`forward` is this layer, never an instance attribute, so ADR-0009 D5's "forward is its class's own" holds and D9's
code-object check exempts them (this repository's code, MODEL-I10). Per call the experts subclass splits
`gate_up_proj` and `down_proj` by one `unbind(0)` each and expert j's (2f, h) view by one `chunk(2, dim=0)` into gate
and up rows (backwards: a stack and a cat, byte moves; never two index slices, whose zero-padded sum turns a −0 into
+0); each runs one f-wide `F.linear` on exactly its rows, the shapes and dtypes of `MoEMLP`'s `SplitLinear`s, and
`down_proj[j]` one h-wide. The shared expert's Linears go through `wgrad.convert`; the banks' dW runs in B (equal to
the deferred one by MODEL-I1; deferring it is a follow-up). HF's MoE forward never runs: `topk` across backends (P10),
`index_add_` on MPS (P11), pooled aux (D8). **Loader (6b only, §13):** allowlist, orientation table, `hf_unplaced`.

**HF-MoE seam rule** (ADR-0008 decision 11 and R14 = ADR-0009 D8 and R15; CLOSED, decided by autonomy, 2026-10-11;
the same text in both ADRs; derivation: ADR-0008 `derivation.md` §23.2). (1) `SUBSTITUTE[(class name, transformers
version)]` has rows at 5.19.0 for the blocks `MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`,
`Qwen3MoeSparseMoeBlock`, the routers `MixtralTopKRouter`, `Qwen2MoeTopKRouter`, `Qwen3MoeTopKRouter` and the experts
`MixtralExperts`, `Qwen2MoeExperts`, `Qwen3MoeExperts`; each row is a class swap to a registered subclass of this
repository. (2) The swapped block calls its `gate` once (the router GEMM) and its `experts` once (dispatch and the
per-expert f-wide GEMMs) as modules, with the GEMMs F8 checks, so ADR-0009's D2 trace sees a pre-hook call of every
parameter-holding module. (3) ADR-0009's D8 classifier finds the block by routing structure, and its name test is an
exact lookup of the block's class; a router or experts row is admitted only under a block with a row. (4) The launch
gate's reference (MODEL-I9, I10) and the `arch="hf"` oracle are the vanilla `ForCausalLM` with the same SUBSTITUTE
swap and D8 injection applied, never HF's own block. (5) The substituted block itself is proved by ADR-0008's F8
(MODEL-I8) against its own oracle, `MoEMLP`, at the pin and in every census. Cost: two module calls per MoE layer
and pass, 0 bytes, 0 waits.

## Mathematical Model

`T_step = M·(t_F + t_B + t_W)_max + T_bubble + T_opt + T_exposed` (DESIGN §9). Per MoE layer and microbatch:
`t_F ≈ 2T·(P_attn + r·P_r + k·3hf + P_sh)/F + β·T·k·h/BW + 3E·t_launch + t_sync` (r: 1 in the compute dtype,
~14.8 for an fp32 router on H100, HYPOTHESIS; β ≈ 10 B per routed element; t_sync one count read, ~10–50 µs,
HYPOTHESIS). Exchange per peer `V_s = Σ_u b_u·P_u·(w−1)/w`, `P_s = Σ(P_attn + P_r + E·3hf + P_sh)`;
`c_s = 8·P^F_s·T_mb/F`, `P^F_s = Σ(P_attn + P_r + k·3hf + P_sh)`; `ρ_s = P_s/P^F_s`. DERIVED (`numbers.py`; w 4,
b_g = b_p = 2; F 400 TFLOP/s and 5 µs launches HYPOTHESIS):

| Model (E, k) | ρ per layer | S1 min T_mb at 400 / 100 / 10 Gbps | device state S 8, w 4 (dense of equal active) | E[unrouted] T 64 / 4096, uniform routing (a lower bound) |
|---|---|---|---|---|
| dense | 1 | 1,500 / 6,000 / 60,000 | — | — |
| Mixtral-8x7B (8, 2) | 3.68 | 5,521 / 22,084 / 220,838 | 34.8 GB (9.5) | 8e-8 / 0 |
| Qwen1.5-MoE-A2.7B (60, 4, shared) | 6.63 | 9,939 / 39,756 / 397,558 | 10.3 GB (1.5) | 0.73 / 1e-121 |
| Qwen3-30B-A3B (128, 8) | 10.95 | 16,431 / 65,724 / 657,235 | 22.4 GB (2.0) | 2.06 / 2e-113 |

| Term per MoE layer per microbatch, T 4096 | Mixtral | Qwen1.5-MoE | Qwen3-30B-A3B |
|---|---|---|---|
| GEMM time `c` (recompute, 8 FLOP/param/token) | 32.3 ms | 7.05 ms | 4.66 ms |
| per-expert launches F, F′, B, W at 5 µs (grouped: 12 launches) | 0.48 ms | 3.60 ms | 7.68 ms |
| host count reads (F, F′) at ≤ 50 µs | ≤ 0.1 ms | ≤ 0.1 ms | ≤ 0.1 ms |
| native pending W, routed `T·k·(2h+3f)·2 B` (dense MLP of width f) | 800 MiB (400) | 260 MiB (65) | 400 MiB (50) |
| fp32 router opt-in / layer GEMM work (compute-dtype router) | 0.12% (0.01%) | 2.12% (0.14%) | 6.85% (0.46%) |
| routing byte moves / forward expert GEMM time | 1.4% | 14.1% | 25.9% |
| HF v5 (ADR-0009; no recompute, F:B:W 2:2:2, derivation §17): routed dW Δ moved from W into B | 7.22 ms | 0.71 ms | 0.77 ms |
| t_B native → HF (t_W native → HF) | 8.08 → 15.29 ms (→ 0.86) | 1.76 → 2.47 ms (→ 1.05) | 1.17 → 1.94 ms (→ 0.39) |
| HF v5 fp32 bank gradient held B(mb) → W(mb), `4·E·3hf` per microbatch in flight (× native pending W) | 5,376 MiB (×6.72) | 1,980 MiB (×7.62) | 2,304 MiB (×5.76) |
| ZB-H1 bubble `(S−1)(t_F+t_B−t_W)` per (S−1), native → HF (1F1B, Qi et al. 2023) | 8.08 → 22.51 ms (24.23) | 1.76 → 3.18 ms (5.29) | 1.17 → 2.71 ms (3.50) |
| share of zero-bubble's reduction kept, HF v5 | 10.6% | 59.8% | 33.6% |
| S1 min T_mb at 400 Gbps (expert gradients fp32, b 4) | 10,882 (×1.97) | 18,981 (×1.91) | 32,357 (×1.97) |
| HF v5 launches over HF's own block, +3E (+4E with F′), derivation §20 | 0.12 (0.16) ms | 0.90 (1.20) ms | 1.92 (2.56) ms |

Which term moves: no exactness rule adds a gradient byte or a wait over HF's per-expert loop (H2); the HF v5 f-wide
split adds 3E launches (with F′ 2.56 ms, 55% of Qwen3's 4.66 ms, HYPOTHESIS); an fp32 router is an opt-in (≤ 6.85%).
D10 adds `L_m·E·8` B per microbatch per peer (118,784 B DeepSeek-V3, 184,320 B Kimi K2) on existing frames and
waits. S1 moves against MoE by ρ: pieces pay only at `T_mb ≥ ρ·1,500` at 400 Gbps, never across 10 Gbps. Launches,
not exactness, decide fine-grained MoE on GPU (A4 at M1); `c_s` omits them, so S1 and S2 under-admit MoE stages (F7).
HF v5: the bubble grows by `2Δ·(S−1)` per MoE layer under zero-bubble and S1's threshold ×1.91–1.97 (both vanish at
S 1, w 1 or with the deferred bank dW), and memory grows, not shrinks: a peer holds `m_BW·4·Σ E·3hf`, `m_BW` its
microbatches between B and W in the zero-bubble order (a 4-layer Mixtral stage: 22.5 / 45.1 / 90.2 GB at `m_BW`
1 / 2 / 4, outside the device-state column), charged by SCHEDULER-I21 in `mem_b`/`mem_w` and I16's memory condition
(derivation §22, §23.1). A global-batch aux would add a bubble (FOLLOW-UP).

## Expected Improvement

Seven families' MoE layers byte-identical under every mechanism (P6–P9 softmax rows; others DERIVED until E2); dense
bits unchanged under 6a, HF loads change under 6b (R11); hub checkpoints through ADR-0009 D7's map (R13); no time claim.

## Alternatives

| Option | Evidence that would make it win | Why not (number) |
|---|---|---|
| A0 Do nothing: refuse MoE | The human drops MoE | The human asked for it |
| A1 HF's block as-is (`topk`, loop over hit experts, `index_add_`), natively or under an HF-modules adapter | Wide stages needing no dense gradient and one backend only | Exact on one CPU map (P9 40/40), but: `None` gradients need both guards weakened and an absent marker on wide stages (A7); `index_add_` refused on MPS under deterministic algorithms (P11); pooled aux (D8); `topk`'s tie order undefined (436/20,000 across CPU/MPS on identical keys, P10), not an identity defect per device map |
| A1t, strongest form: transformers 5.19.0's own block with only `topk` replaced by D3's stable descending sort (derivation §23.3) | GPU launches dominate and HF-block fidelity is wanted (fine-grained MoE: Qwen3) | Exact per device map on CPU (P9 40/40), 8E launches vs 12E: −0.12/−0.90/−1.92 ms (−0.16/−1.20/−2.56 with F′) at 5 µs, i.e. 0.4% / 12.8% / 41.2% of the layer's GEMM time (HYPOTHESIS). What still differs: its `index_add_` combine cannot run on MPS (P11; CUDA R1), seeds `+0` (a −0 sign under the fp32 recipe, k roundings under bf16); its `.sum` renormalization leaves the order to the kernel (equal at k 2, may differ at k ≥ 3: Qwen 4, 8; P5); x̂'s gradient is the engine's sum (D1: CUDA HYPOTHESIS); its 2f-wide GEMM is another function than `MoEMLP` (F8 fails); aux injection and the count lane must be added anyway. The swap still wins on exactness (written orders, one function); A1t-strong wins on time until a fused gate_up GEMM in both paths (FOLLOW-UP) removes the gap; under the seam rule an HF-fidelity row needs no ADR-0009 change |
| A1′ transformers v5 experts hook with an exact experts function | Router and aux pinned too | Covers the experts only (router as A1); superseded by replacing the whole block's forward on HF's objects (§HF boundary) |
| A2 Dense-masked (every expert on every token, weight 0 off top-k) | Static shapes worth E/k× compute | 4× Mixtral, 16× Qwen3 compute; S1 unchanged |
| A3 Capacity factor with drops | A recipe that wants static shapes | Changes the loaded models' function; per-rank capacity is topology-dependent (forbidden) |
| A4 3-D bank as the parameter, grouped GEMM per (mb, projection) | M1 measures per-expert launches > 25% of the layer's time | Exact by D0 alone when both paths run the same grouped call (no segment independence needed); needs a grouped deferred-dW record, CUDA determinism (R1), a private API; unmeasurable at gpu false (derivation §8) |
| A5 Expert parallelism / owner-side W | T_mb below the bytes breakeven inside a site | Gradient bytes ×1/10.5 (Mixtral, T 4096) but 4 all-to-alls per layer per microbatch on the critical path (160 ms across a 40 ms link against 32.3 ms of compute) and new process responsibilities; its own ADR, inheriting D5–D7 |
| A6 Three families only (softmax over E) | The human narrows M4 | In M4 both options give checkpoint paths to the same families (3 under 6b; 0 under 6a until R13 closes); A6 also refuses the four other rows for from-scratch recipes, and saves only elementwise ops, two sorts and D10's lane |
| A7 Lossless "absent" flag; data-dependent gather (A8 per-op regather: ≈ 2× gradient volume, FOLLOW-UP) | > 5% unrouted (peer, step, expert) | A new frame type or a critical-path wait. E[unrouted] ≈ 0 at T ≥ 256 holds for uniform routing only; a skewed router leaves more (E3 records the share) |
| A9 D10 by replicating the bias and sending counts to every peer | No owner exists for the bias | A new frame type and w−1 sends per peer for the same bytes (derivation §6) |

## Trade-offs

Bits differ from HF's block: the defined sort over E (D3), the fp32 combine (D7), D1's fold and two f-wide GEMMs
(F8), traded against HF fidelity for one MoE function (A1t). Unrouted experts drift by momentum; per-microbatch balancing (R4);
launches (R5); no LoRA with MoE; on the HF v5 path dW in B, fp32 bank gradients held B→W and shipped, 3E launches.

## Risks

- R1 (HYPOTHESIS, M1): on CUDA a 0-row GEMM returns +0, `torch.sort(stable=True)` and each fixed-shape kernel are
  run-to-run deterministic. MPS stable sort: 1 output in 20 runs (P11). Identity per device map (DESIGN §3).
- R2: M2b is merged here as a snapshot (8dee0b9); units and the count lane change its tables for MoE only (E6). R3: S1 needs large `T_mb`.
- R4: fidelity, not exactness: per-layer or per-sequence aux vs HF's pooled form and Qwen3's global batch (Qiu et al.
  2025); global, grouped (G fixed by the recipe) and HF-pooled scopes are FOLLOW-UPs (derivation §5).
- R5: per-expert launches dominate fine-grained MoE on GPU (7.68 vs 4.66 ms per Qwen3 layer, HYPOTHESIS); A4 at M1.
  HF v5 path: 12E per layer per microbatch against HF's own 8E (+2.56 ms for Qwen3 with F′).
  R6: skew changes kernel efficiency, never bits. R7 (6b only): the allowlist stops loads that pass silently today.
- R9: under 6a this ADR builds no native HF loader or dense-loader field (ADR-0009 decision 5, D11); under 6b it
  alone owns them (MODEL-I4, SPEC-model task 3). R10: rows other than softmax-over-E are DERIVED until E2's arms.
- R11 (6b only, FACT): `rope_style = "half"` changes every dense HF RoPE load's bits (ADR-0009 Evidence 1: 7.86e-3
  max|Δlogit| llama, 0 with `rotate_half`). R12: one transformers, 5.19.0 (product pin under 6a, test-only under 6b).
- R13 (CLOSED by decision 10 (i), autonomy, 2026-10-11): hub tensors of the three families are per expert, the
  skeleton's fused (FACT, `conversion_mapping.py:1068-1108`); ADR-0009 D7 now owns the hub-to-fused byte map with a
  round-trip test (derivation §23.2), so their hub checkpoints load under 6a.
- R14 (CLOSED by the seam rule, decision 11, autonomy, 2026-10-11): ADR-0009's (a) own-forward guard, (b) trace,
  (c) classifier and (d) vanilla gate each refused the HF path as written; the class swap removes (a), the router
  and experts rows called as modules remove (b), the routing classifier with an exact block lookup removes (c), and
  the gate's reference carrying the same swap and D8 injection removes (d) (§HF boundary). The block is F8's.
- House defects (reported): `config_from_hf` ignores granite's multipliers (closed by 6b's allowlist or 6a's deletion).

## Falsification Criteria

- F1 (H1): any sha256 of weights, AdamW m/v/step, RNG or selection biases, any CE or aux hex, differs from the oracle
  in a true arm of E2, E2-HF or E3, for any recipe row.
- F2: a must-fail control of E4 whose trigger count is > 0 does not diverge: resized before any claim rests on it.
- F3: native, the routed pending-W bytes (storages deduplicated by `data_ptr`) ≠ `T·k·(2h+3f)·b` per layer exactly,
  for any skew (E3); HF v5, any routed-expert pending-W entry (must be 0), or the bank-gradient bytes held between
  B(mb) and W(mb) ≠ `b_m·E·3hf` per MoE layer times `m_BW` (E2-HF). E4's duplicated-X control must fire it.
- F4 (with M2b): per peer per step, any frame class's payload bytes ≠ its formula exactly: gradient units
  `Σ_u b_u·P_u·M_k·(w−1)/w` (b 4 for HF v5 expert units), gathers, count lane `L_m·E·8` per microbatch per owner (E4:
  a dropped or doubled lane must fire it); routed-expert unit frames per (destination, microbatch) ≠ 3 (native) or ≠ 2
  (HF v5); shared-expert frames ≠ one per `SplitLinear`; or a dense model's tables changed (E6).
- F5 (6b only): a real config key mapped unlike E5a's table, an unknown key accepted, a loaded byte unlike HF's
  per-expert view (E5c), an unplaced parameter loaded under `refuse`, or E5b's bound missed (or a must-fail within it).
- F6 (H2, CPU, MEASURED local): native, the layer's F + B + W wall time > 1.15× the v4-style block's (E7a); HF v5,
  the replaced forward's F + B > 1.15× transformers 5.19.0's own block's plus 3E times a measured 0-row `F.linear`
  call (E7b). F7 (H3): an MoE stage's compute (WT_TELEM, SIMULATED) below 0.9× its analytic `c_s` (E7).
- F8: the HF-object block at 5.19.0 differs from `MoEMLP` on byte-mapped parameters in any output, aux value or
  gradient bit (−0 included), or its HF decoder layer fails on the one-tensor contract (E2-HF).

## Local Experiment

gpu false: CPU, multi-process localhost. h 64, 4 heads, 4 MoE blocks, E 8, f 96, V 256, T 32, M 4, 3 steps, fp32 and
bf16; one arm per recipe row of derivation §3 (DeepSeek with groups 4/2, scaling, bias and D10; GPT-OSS with
`softmax_k`, biases and clamp; Llama 4 with k 1 and input weighting; OLMoE with z; Qwen2 with a shared gate).
- E1 (unit): P1–P5, P8–P10 as committed tests; D3 on crafted keys (ties, ±0, subnormals, ±inf, group ties); D3's
  group refusals (E 8 with G 8, G 3, k > G_top·E/G); D1, D5, D7 backward equal their folds; 0-row gradients +0 on
  both paths; D10's sign at equality; MPS rows non-gating.
- E2 (identity): `FACTORY_REFERENCE=1` vs depths 1, 2, 4, zero-bubble orders with W fill, recompute, ZeRO-1 widths
  1, 2 (a bias whose owner runs no microbatch), M2b piece widths 1, 2, 3; sha256 of the state; every CE and aux hex.
- E2-HF (6a, required M4 gate entry): tiny Mixtral, Qwen2-MoE and Qwen3-MoE built by transformers 5.19.0, class
  swapped: the decoder layer runs; output, aux and every gradient (sign of zero included) equal `MoEMLP`'s bit for
  bit, fp32 and bf16, 0, 1 and many rows per expert (F8); oracle vs a 2-stage capture path over the HF layer list;
  bank-gradient peak bytes, `m_BW` 1 and 2 (F3); must-fail: gate and up by two index slices with a constructed
  −0; then the identity arm through ADR-0009's adapter, its gate reference and oracle carrying the same swap and D8
  injection (seam rule), and its loader's hub-to-fused round trip (decision 10 (i)).
- E3 (skew): an all-tie router and a hot expert (ties and unrouted (layer, mb) > 0 recorded first); E2's arms; F3.
- E4 (must-fail, each records its trigger): `topk` on the stage path only; skipped unrouted experts; reverse combine at
  k 3; descending token rows; one W GEMM over microbatches m, m+1; aux as α/M; bias from float means or one peer's
  counts; group ties by `topk`; pieces off `Q_t`; count lane dropped or sent twice (F4); fold from zeros with a
  constructed −0; up storing its own X (F3: 416 → 480 values per routed row at E1's sizes, +15.4%).
- E5 (6b only, derivation §13): real config.json files vs a field table (E5a), fidelity vs transformers (E5b), E5c.
  E6 (with M2b): F4's frame classes per peer and step, native and (6a) HF v5 (2 expert units, fp32, `Q_t` per §HF).
- E7 (cost): median of 20, CPU 1 thread, (h 256, f 384, E 16, k 2) and (E 32, k 8), T 1024 (F6, MEASURED local).
  E7a: `MoEMLP` vs `probe3.py`'s v4-style block, F + B + W. E7b (6a): the replaced forward vs transformers 5.19.0's
  own `MixtralSparseMoeBlock`, F + B, with the 0-row call measured. Analytic `c_s` vs WT_TELEM (F7, SIMULATED).

## Vast.ai Experiment

None (gpu false). At M1: E2 on one GPU class (R1), launch share and A4's grouped call (R5); plan and budget to the human.

## Migration Impact

New `moe_*` fields default to dense (`moe_experts = 0`); manifest `aux`, `moe_bias` and M2b's table changes for MoE
only; transformers per R12. Rollback: under 6a `moe_experts = 0` leaves dense bits unchanged; under 6b only a revert
restores dense HF loads (R7, R11). MoE checkpoints, count-lane frames and in-flight MoE runs do not survive a revert.
Workstreams: model, runtime, scheduler.

## Validation Plan

MODEL-I8, RUNTIME-I18, SCHEDULER-I21 (MODEL-I4 under 6b) on unit, integration, identity; E1–E4, E7, E2-HF or E5 in M4's GATES; E6 on M2b.

## Ideas considered

Rounds 0–2: derivation §16; round 3: §19; round 4: §21; rounds 5–6: §23.4 (moved verbatim in round 7, verdicts
unchanged but one). Round 7 (correction pass under autonomy, 2026-10-11):

| Idea (source; verified?) | Verdict | Reason and number |
|---|---|---|
| Gate reference = vanilla model with the same class swap and D8 injection; F8 proves the block (round-5 row; ADR-0009 R15) | ADOPTED (decision 11, seam rule) | Removes R14 (d), the refusal of every HF MoE run; 0 per step, the gate's 1–2 s unchanged |
| Router and experts rows called as modules (round-4 row, decision 11 (i)) | ADOPTED (seam rule (1)–(3)) | Removes R14 (b), (c) with the same GEMMs; two module calls per layer and pass (µs) |
| Hub-to-fused byte map in ADR-0009 D7 with a round trip (decision 10 (i)) | ADOPTED (ADR-0009 D7) | Closes R13; one load-time strided copy per expert tensor; derivation §23.2 |
| A1t in its strongest form: HF's block with only `topk` replaced (finding) | Restated (Alternatives); swap kept | −3E/−4E launches (41–55% of Qwen3's GEMM time, HYPOTHESIS) against MPS-refused `index_add_`, kernel and engine orders, F8; derivation §23.3 |
| The bank-gradient hold × the zero-bubble in-flight count (finding) | CORRECTION (text) | `m_BW·4·Σ E·3hf`: 22.5 / 45.1 / 90.2 GB at `m_BW` 1 / 2 / 4 (Mixtral, 4 layers); derivation §23.1, SPEC-scheduler §2.1, I21 |

## Changelog

- Rounds 0–4: P1–P11, skeleton, recipe, D0, D8, D10's lane, allowlist; (frozen from 3) loader on ADR-0009's grant, HF table, F3/F4/F8, E2-HF, E6, R13, R14; H2 per path, E7b/F6, A1t, class swap, decision 11, group refusals. Round 5 (frozen; text, numbers, amendments): HF v5 bank-gradient hold costed and charged (table, §17/§22, F3, SPEC-model §2, SCHEDULER-I21); R14 (d), decision 11 (d), code-object part refuted; A1t restated; D3's reason is definedness; F4 per frame class; Llama 4 source; ideas §21.
- Round 6 (references and ideas only, decision unchanged): the human's seven model references and four facts checked and placed; reviewer ideas ruled (none adopted); MXFP4 range correction to derivation §9 recorded; torchtitan #1974's numbers re-fetched.
- Round 7 (correction pass, autonomy, 2026-10-11): decisions 1–11 recorded as decided by autonomy; decision 10 (i)
  and 11 closed as the HF-MoE seam rule, written identically in ADR-0009 D8 (R13, R14 CLOSED); the bank-gradient
  hold charged per in-flight microbatch (`m_BW`); A1t compared in its strongest form; round 5–6 ideas moved to §23.4.

## Amendments (texts in DESIGN.md and the specs, each in its -history.md and `freeze.mjs amend --by ADR-0008`)

DESIGN.md §0, §3, §4 (INV-1), §8 (M4 row), §9, §10 (row, question 6); SPEC-model §2, §3, MODEL-I4, MODEL-I8, task 3, §6; SPEC-runtime §2 (oracle), RUNTIME-I18, tasks 5, 6; SPEC-scheduler §2.3, SCHEDULER-I21, task 5. Rounds 3–4 re-amend DESIGN §0, §8–§10, SPEC-model §2, §3, MODEL-I4, I8, task 3, SPEC-runtime task 5, SCHEDULER-I21; round 5 DESIGN §3 (MoE row), §8, §9, §10 (row, question 6), SPEC-model §2 (bank hold), §3 (gate), SPEC-scheduler §2.1, I21; round 7 DESIGN §0, §8 (M4 row), §9, §10 question 6, SPEC-model §2 (hold × in flight), §3 (seam rule), MODEL-I8, SPEC-scheduler §2.1, I21, SPEC-runtime §2 (oracle swap).

## Recommendation

PROCEED with D0–D10, the HF boundary at the pin, the loader per decision 6 and the planner's split; E1–E4, E2-HF or E5,
E7 first, E6 on M2b. Not built now: EP, grouped GEMM, global or pooled aux, regather, MXFP4, deferred bank dW, more rows.

Decisions that are the human's (each decided by autonomy, 2026-10-11: `autonomy.json` `humanGrants` false; the
recommended option is taken, recorded at the end of each item):
1. Router precision: each family's own (recommended), fp32 an opt-in (≤ 6.85%). 2. Aux: per-layer or per-sequence
   lbl and z-loss, injected locally (recommended). 3. Identity record: CE per (step, mb), aux per (step, mb, layer)
   hex, selection biases in the sha256 state; or CE only. 4. Unrouted experts: +0 gradient, AdamW steps them
   (recommended). 5. Milestone: M4 (DESIGN §8's M4 row), on M2b's merged code.
   Decided by autonomy, 2026-10-11: 1 family's own; 2 per layer or sequence, injected; 3 CE, aux hex and biases; 4 +0; 5 M4.
6. Loader, keyed on ADR-0009's grant. 6a, granted (recommended): no native HF loader, no dense-loader field, no
   allowlist; transformers is ADR-0009's pin. 6b: the allowlist (`head_dim`, `norm_eps`, `rope_style` per R11,
   `qkv_bias`, `qk_norm`), `hf_unplaced`, the orientation table, transformers 5.19.0 test-only for E5b, E5c. Decided by autonomy, 2026-10-11: 6a.
7. Routing scope: the enumerated recipe (recommended) adds from-scratch recipes of the four other rows at elementwise
   ops and two sorts; the 3 softmax rows refuse them. Checkpoint families in M4 are the same either way (A6).
   Decided by autonomy, 2026-10-11: the enumerated recipe.
8. D10 now (recommended) or later: no M4 checkpoint exercises it; same cost whenever built (an int64 lane ≤ 184,320 B
   per microbatch per peer, 256 B in E2; zero1's list; `check_frame`), on M2b's merged code. Decided by autonomy, 2026-10-11: now.
9. HF boundary: HF objects at the pin, class swapped to the exact layer (recommended); it computes this recipe's
   bits, not HF's block's (A1t: f-wide GEMMs, fp32 combine, D1's fold; an HF-fidelity row is a follow-up); E2-HF's adapter
   arm is a required M4 gate entry; under 6b no HF-modules path exists and the table and E2-HF are not built.
    Decided by autonomy, 2026-10-11: the class swap.
10. MoE hub checkpoints under 6a (R13): (i) ADR-0009 D7 owns the hub-to-fused map with a round-trip test
    (recommended; 22 model types alias `qwen2_moe`'s entry at 5.19.0); (ii) this ADR's orientation table serves it;
    (iii) M4 trains MoE from scratch and HF configs only. Decided by autonomy, 2026-10-11: (i), written into ADR-0009 D7.
11. ADR-0009's trace, classifier and gate (R14 (b)–(d)): (i) router and experts rows called as modules, an exact
    block lookup, and the gate's reference carrying the same swap and D8 injection, F8 proving the block
    (recommended); (ii) ADR-0009 counts a substituted subtree as covered; (iii) no HF MoE path in M4. Decided by
    autonomy, 2026-10-11: (i), the HF-MoE seam rule (§HF boundary), the same text in ADR-0009 D8; CLOSED.

## Human Approval
Granted only by `node agents-architetura/tools/factory/approve.mjs grant` in a human's terminal; recorded in docs/design/APPROVALS.json.
