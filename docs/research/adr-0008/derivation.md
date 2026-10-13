# ADR-0008 derivations (exact Mixture-of-Experts layer)

Long derivations behind `docs/design/adr/ADR-0008-moe-exact.md`, moved here so the ADR stays one decision within
400 lines. Labels: MEASURED (a command ran, with the tree it ran on), DERIVED (arithmetic on stated inputs),
HYPOTHESIS (unmeasured or recalled), FACT (code read on a62e4eb unless stated).

## 1. Scripts and provenance

`scripts/` holds the round-0 scratch scripts byte for byte (copied in round 1 from the session scratchpad that ran
them; full sha256 below, which match the 16-hex prefixes the round-0 ADR cited).

| File | sha256 | What it measures |
|---|---|---|
| `probe.py` | `0ca97fdb3a407b1676ec58585890dea3d59940753c5610efe4a7a5423103c548` | P1–P5 |
| `proto.py` | `0b37f806e6ba51146a06215a5e889257b823cef6b80e7783e85cab311353730a` | P6 (39 identity arms, controls) |
| `proto.out.txt` | `9fc88713706e9099ef4fda9ffe6b8823713086dc6dbc178ce6eb6b73b612781c` | P6's stored output |
| `pieces.py` | `744f55fe41a678c324fcaf02b1eb5b7fc2670edce9ad141daa8811c1d62f224d` | P7 (16 piece arms, controls) |
| `numbers.py` | `f6155c981431bf85a81f76f5d30ef4a78fcee76babb6d6355b1c031c221eaa9b` | the ADR's DERIVED tables |
| `model.py` | `52e83bc57a9b60f24822ea40fe403681a7d0dec3c87c58a8b02ce96bc24cb6c8` | early bytes-vs-EP arithmetic |
| `probe2.py` | `491dff880abb2df78594ebdcfb0b119fa9ee39f09d805dd6d7a26aaf0a63cc60` | P3's row-count sweep |
| `probe3.py` | `e4773a4926ec0c380fb56d7e1540d1a4b523c35ca75c3ab7f61dc3db8ad6709d` | P8–P11 (round 2: both-paths arms, cross-backend, MPS accumulate-scatter, grouped GEMM) |
| `probe3.out.txt` | `a3dec519a0b9bc4df010c11fdb08f8237b78c02ec5099733733eda420d41c61e` | P8–P11's stored output |

- Tree: they ran on 5f2c815 (worktree `edt-sec`), Apple M4, torch 2.14.1 CPU, 1 thread,
  `use_deterministic_algorithms(True)`, `.venv/bin/python -I <script>` (pieces.py from the scripts' directory, since
  it imports proto). `probe.py` and `proto.py` put the repository's `src` first on `sys.path` (run them from the repository root).
- The only product module they import is `wantrain.model.wgrad`. FACT: its blob is `86c0404` on both 5f2c815 and
  a62e4eb (`git rev-parse <tree>:src/wantrain/model/wgrad.py`), and `git diff 5f2c815 a62e4eb -- src/wantrain/model
  src/wantrain/validation/reference.py src/wantrain/runtime/accumulate.py` is empty. The scripts reimplement the
  stage path rather than import `stage.py` (which changed: 553+/121−).
- Only `proto.py`'s output was stored (`proto.out.txt`). P1–P5 and P7 numbers are as the round-0 ADR recorded them.
- `probe3.py` (round 2) ran on a62e4eb (working tree with documentation changes only; `src/` as committed, wgrad blob
  `86c0404`), Apple M4, torch 2.14.1, CPU 1 thread and MPS, through a machine test slot:
  `node agents-architetura/tools/verify/machine-lock.mjs run --slot --label adr-0008-probe3 --timeout 600 --
  .venv/bin/python -I docs/research/adr-0008/scripts/probe3.py <checkout>/src` (exit 0). It imports `proto.py` with
  this checkout's `wgrad` loaded first, so its P8–P11 are MEASURED on a62e4eb. Run twice (the second run added Q7);
  every number of Q1–Q6 repeated exactly; the stored output is the second run's.
- Status of P1–P7 on a62e4eb: HYPOTHESIS until E1 re-runs them on this tree as committed tests (ADR §Local Experiment).
  Not re-run in round 1 (the round prompt forbids test runs by a parallel agent). A re-run on this tree needs the
  `sys.path.insert` line pointed at this checkout's `src` (which changes the file's sha256), then
  `.venv/bin/python -I proto.py` from `scripts/`; E1 replaces the scripts with committed tests.

## 2. The hazards (P1–P7 MEASURED on 5f2c815; P8–P11 MEASURED on a62e4eb)

- P1. An expert `SplitLinear` called on 0 rows: the oracle's `autograd.grad` without `allow_unused` returns an all-+0
  dW (no sign bit); under `capture()` the store holds the entry and `flush` yields all +0.
- P2. AdamW given `grad=None` at step 2 skips the parameter (step count 1); given zeros it steps (step 2); the
  weights differ.
- P3. fp32: the first row of `x[:1]@Wᵀ` differs from the same row of the 2048-row GEMM in 462 of 512 elements; one
  dW GEMM over 300 rows vs two over 150 plus an add: 115,955 of 131,072 elements differ.
- P4. bf16-rounded N(0,1) router scores, E 64, k 8, 20,000 tokens: 4.25% of tokens tie at the k-th boundary (fp32
  scores: 0 of 20,000). `torch.topk` selects another set than the stable descending sort on 420 tokens. The stable sort
  equals an integer composite key (value, then lower index) on 20,000/20,000, and CPU equals MPS on 20,000/20,000.
- P5. A k-term fold of 64-wide rows in reversed order differs on 100% of tokens at k 3 and k 8, 0% at k 2 (IEEE
  addition is commutative). `sum(dim)` matched the left fold on this CPU kernel: kernel-dependent, not relied upon.
- P6 (`proto.py`, stored output). The round-0 block in a 4-layer model (Embed, 2 MoE blocks, Head; h 16, f 24, V 32,
  T_mb 6, M 3, 2 steps, router rows duplicated so tokens tie), oracle (one graph, immediate dW) vs a 2-stage path
  (graph cut, `capture()`, W in reverse microbatch order): weights, AdamW m, v, step, every CE and aux hex identical
  in 39/39 arms: fp32 and bf16 × (E 8 k 2, E 8 k 3, E 6 k 4) × shared expert on/off × 3 cuts, an all-tie router (69
  boundary ties, 44 empty (layer, microbatch) experts), and recompute in fp32 and bf16. Controls (oracle unchanged,
  one mutation on the stage path): `torch.topk` 2/2 diverge; skipping unrouted experts 2/2; reversed combine at k 3
  2/2; rows in descending token order 1/2 (bf16 with ≤ 6 rows per expert did not diverge); no fan-out on the stage
  path 2/2. No fan-out on both paths: 6/6 identical (autograd's order is structural on this engine, but it is not the
  fan-out's order).
- P7 (`pieces.py`). ADR-0006-style pieces over the same model (each piece folded in microbatch order from its slice of
  every microbatch's gradient, its own AdamW, gathered back by byte copy), boundaries at multiples of `Q_t`: 16/16
  identical (fp32 and bf16, E 8 k 2/3, w 2/3, uniform and all-tie routers). Row-aligned pieces not at `Q_t`: 2/2
  diverge. A fold from zeros with no −0 present: identical, so that control needs a constructed −0.
- P8 (round 2, `probe3.py` Q1). `torch.topk` on BOTH the oracle and the 2-stage path (all other code as P6), fp32 and
  bf16, the duplicated-row and the all-tie routers, cuts 1–3, recompute: 14/14 identical. P6's `topk` control mutated
  the stage path only; it showed that `topk` and the stable sort pick different sets, not that `topk` separates oracle
  from stage. On one device map it does not (D0: one kernel call on the same shape).
- P9 (Q2, Q3). An HF-style block (v4 `MixtralSparseMoeBlock`'s eager loop: compute-dtype gate, fp32 softmax, `topk`,
  renormalization by `.sum`, weights cast to the hidden dtype, a loop over HIT experts with `index_add_` into zeros,
  no fan-out function; Qwen2-MoE's sigmoid-gated shared expert) on BOTH paths, the oracle with `allow_unused` and
  None-skipping AdamW like the stage path: 40/40 identical (fp32 and bf16 × 3 (E, k) × shared on/off × 3 cuts, the
  all-tie router with 38–39 empty (layer, microbatch) experts, recompute). Looping over EVERY expert (0-row calls, no
  None): 40/40 identical. On one CPU device map HF's block is exact oracle-vs-stage; its differences from this ADR's
  layer are definedness, cross-backend behaviour and the distributed paths (§12), not P6's one-sided controls.
- P10 (Q4). P4's 20,000 bf16-rounded rows (E 64, k 8): CPU `topk` vs CPU stable sort 436 sets differ; MPS `topk` vs
  MPS stable sort 0; `topk` CPU vs MPS 436 sets and 2,427 index orders differ; stable sort CPU vs MPS 0 ranks differ;
  MPS `topk` 1 distinct output in 20 runs (fp32 and bf16). The tie hazard of `topk` is cross-backend, not
  oracle-vs-stage.
- P11 (Q5, Q6). MPS, 200,000 rows into 512 targets: `index_add_` and `scatter_add_` with duplicate indices raise under
  `use_deterministic_algorithms(True)` (`index_put_with_accumulate_mps`, `scatter_reduce_mps` have no deterministic
  implementation) and give 20 distinct results in 20 runs without it, none equal to the CPU's; `index_add_` with
  unique indices (HF's per-expert call, Q7) raises the same way. MPS stable sort: 1
  distinct output in 20 runs. `torch._grouped_mm` exists on this build on CPU and MPS and equals a per-segment `mm` on
  one bf16 case with segments (0, 5, 37, 128) rows (local observation, not a CUDA fact).
- Limits: the prototype router is softmax-over-all with renormalization only; the round-1 recipe family (§3) adds
  sigmoid scoring, logit-space selection, a selection bias, group-limited selection, scaling, input-side weighting,
  expert biases, a clamped activation and a z-loss, none of which P6 exercised. E1/E2 carry them.

## 3. The routing recipe family

Every field is a recipe field (part of the identity record's recipe). The exactness skeleton (ADR D0, D1, D5–D7, D9)
is the same for every row. Values from public config.json files and HF modeling code, recalled: HYPOTHESIS until E5
reads the real configs.

| Family | score | select key | groups | weights | apply | expert | shared | aux | layers |
|---|---|---|---|---|---|---|---|---|---|
| Mixtral (HF) | softmax over E | score | — | renorm | output | SwiGLU | — | lbl | all |
| Qwen2-MoE (Qwen1.5-MoE) | softmax over E | score | — | `norm_topk_prob` (false) | output | SwiGLU | sigmoid-gated | lbl | `decoder_sparse_step`, `mlp_only_layers` |
| Qwen3-MoE | softmax over E | score | — | `norm_topk_prob` (true) | output | SwiGLU | — | lbl (global in training) | as Qwen2 |
| OLMoE | softmax over E | score | — | none | output | SwiGLU | — | lbl + z | all |
| DeepSeek-V3/R1, Kimi K2 | sigmoid | score + bias | n_group 8, topk_group 4 (DS-V3) | renorm + `routed_scaling_factor` | output | SwiGLU | ungated | seq-wise lbl + bias update | `first_k_dense_replace` |
| GPT-OSS | softmax over the k selected logits | logit (router has a bias) | — | that softmax | output | clamped SwiGLU (limit, α), biases, interleaved gate/up | — | lbl | all |
| Llama 4 | sigmoid of the selected logit | logit | — | that sigmoid | input (k 1) | SwiGLU | ungated | — | `interleave_moe_layer_step` |

Router dtype (round 2, recipe field `moe_router_dtype`, per family): the dtype the family's own code runs its router
in. FACT from HF code the round-2 reviewers fetched: Mixtral's gate is an `nn.Linear` in the model dtype, softmax in
fp32 (`modeling_mixtral.py`, main); DeepSeek-V3 computes `F.linear(hidden.float(), weight.float())`, masks excluded
groups with `-inf` and normalizes with a `+1e-20` denominator (`modeling_deepseek_v3.py`, main). HYPOTHESIS (recalled,
same pattern as Mixtral): Qwen2/3-MoE, OLMoE, GPT-OSS and Llama 4 run the gate in the model dtype; Kimi K2 uses the
DeepSeek-V3 architecture (fp32). So `compute` for Mixtral, Qwen2/3-MoE, OLMoE, GPT-OSS, Llama 4 and from-scratch
recipes; `fp32` for DeepSeek-V3/R1 and Kimi K2. In `compute` the router GEMM runs under autocast (compute-dtype
output, as HF's logits), then the score is taken in fp32; ties are resolved by the stable sort under either dtype.

Config values FACT from the published config.json files fetched by the round-2 reviewers (2026-10-10):
Qwen3-30B-A3B hidden 2048, 32 heads, `head_dim` 128 (so q is 4096 wide), `rms_norm_eps` 1e-6, 128 experts, 8 active,
`moe_intermediate_size` 768; Mixtral-8x7B-v0.1 `rms_norm_eps` 1e-5; Qwen1.5-MoE-A2.7B `use_sliding_window` false;
Kimi-K2-Instruct `n_group` 1, `topk_group` 1, `routed_scaling_factor` 2.827, `norm_topk_prob` true, 61 layers with
`first_k_dense_replace` 1 (60 MoE layers: §6's 184,320 B of counts and 92,160 B of fp32 bias hold); gpt-oss-20b 32
experts, top-4, `swiglu_limit` 7.0, `sliding_window` 128, `router_aux_loss_coef` 0.9 (never mapped, D8).
Round 3 (configs fetched by the reviewers, 2026-10-10): DeepSeek-V3 61 layers, `first_k_dense_replace` 3 (58 MoE
layers), 256 routed + 1 shared, top-8, `n_group` 8, `topk_group` 4, `routed_scaling_factor` 2.5, sigmoid,
`topk_method` noaux_tc; Kimi-K2-Instruct `aux_loss_alpha` 0.001, `seq_aux` true (recipe defaults for that row are a
FOLLOW-UP with its HF table row). FACT from transformers 5.19.0 (uv cache, read in round 3): GPT-OSS's router takes
the softmax over the top-k values in the logits' dtype (`modeling_gpt_oss.py:128`); Llama 4 takes the sigmoid in fp32
and casts it back to the logits' dtype (`modeling_llama4.py:150`); DeepSeek-V3's `e_score_correction_bias` is an fp32
buffer (`_keep_in_fp32_modules_strict`) with no update code upstream. Secondary sources calling GPT-OSS's score a
sigmoid (NVIDIA NIM, SemiAnalysis) are not verified and are not used. The other table values stay HYPOTHESIS until E5a.

Outside this ADR (the model is refused, naming the piece, until its own decision lands): DeepSeek-V3/Kimi K2 MLA
attention and the MTP module; Kimi K2's MuonClip (not elementwise, §7); GPT-OSS attention sinks, alternating banded
attention and MXFP4 experts (§9); Llama 4 chunked attention and NoPE layers; OLMoE's q/k norm shape (HYPOTHESIS:
over the whole projection, not per head); expert-choice routing (Zhou et al. 2022) and BASE layers (Lewis et al. 2021),
whose assignment of a token depends on the other tokens of the batch; zero-computation experts (LongCat-Flash), whose
active compute depends on the routing; DeepSeek-V4's hash-routed layers (a token-id table: the stage would need the
token ids in its FWD frame). These refusals are the native path's (ADR decision 6b): on the native path three families
load, limited to the configs the allowlist of §13 fully maps: Mixtral, Qwen2-MoE, Qwen3-MoE. Under ADR-0009 (6a) the
attention runs as HF's own code and a model is refused only if its sparse-MoE class has no row in §14's table: at
5.19.0 `DeepseekV3MoE` (also Kimi K2), `OlmoeSparseMoeBlock`, `GptOssMLP` and `Llama4TextMoe` have none (FOLLOW-UP).

Mixtral: the paper writes `Softmax(TopK(x·W_g))`; HF computes softmax over all E, top-k, then renormalizes. Equal in
exact arithmetic, other bits; the loader follows HF (the checkpoints' code). Logit-space and fp32-probability selection
can differ: distinct logits may round to equal probabilities, so `select key` is part of the recipe, never inferred.

## 4. New reductions and their orders

Rule (ADR D0): a reduction inside one kernel call on a fixed shape is the kernel's, identical on every path of one
device map (as every GEMM's internal order already is; R1); a sum whose terms are assembled outside one kernel call
(autograd accumulation, a scatter, microbatches, peers, slots) or by atomics has a written order.

| Quantity | Terms | Order |
|---|---|---|
| softmax / sigmoid / logsumexp per token | E or k values, one kernel on (T, E) or (T, k) | kernel (D0) |
| group key (DeepSeek) | the group's top-2 keys | 2 terms commute: exact in any order |
| group selection | `n_group` group keys | stable descending sort, then lower group index |
| renormalization `d_t` | k selected weights | left fold, ascending expert index, then `+ norm_eps` once |
| scaling | one multiply after renorm | — |
| combine | k weighted rows, then shared | left fold, ascending expert index (D7) |
| dispatch backward | a token's k slot gradients | left fold, ascending expert index (D5) |
| `S_i`, z-loss and lbl sums over tokens | T values per expert, one kernel | kernel (D0) |
| seq-wise lbl | b per-sequence values | kernel mean over b (D0) |
| counts `c_i` | integers | any order (exact) |
| bias-update counts over the global batch | integers from every microbatch and peer | any order (exact) |
| a tensor's ≥ 3 autograd consumers | their gradients | the fan-out's left fold (D1) |

## 5. Load-balancing forms

Notation per MoE layer and microbatch: T tokens, counts `c_i` (integers, `Σ c_i = T·k`), `S_i = Σ_t q_ti` with `q` the
router's normalized probabilities (softmax over E; `s/Σ_j s_j` for sigmoid scoring).

- Per layer (the ADR's default, Switch/Megatron): `aux_l = (E/(T²·k))·Σ_i c_i·S_i`; uniform routing gives 1.
- HF `load_balancing_loss_func` (Mixtral, Qwen2/3-MoE, OLMoE; FACT from the HF code the reviewers fetched, recalled):
  softmax over the concatenation of all L_m layers' logits (L_m·T rows), top-k, one-hot (L_m·T, k, E), mean over rows,
  sum over k and E, times E: `aux_HF = (E/(L_m²·T²))·Σ_i C_i·S_i^pool` with `C_i = Σ_l c_il`, `S_i^pool = Σ_l S_il`;
  uniform routing gives k. The trainer adds `router_aux_loss_coef·aux_HF` once.
- Ratio. Uniform routing: the per-layer form summed over layers gives L_m, HF's gives k. Same coefficient α, per-layer
  injection is L_m/k times stronger: 16 for Mixtral (L 32, k 2), 6 for Qwen1.5-MoE (24, 4) and Qwen3-30B-A3B (48, 8).
  Mapping `router_aux_loss_coef` onto the per-layer form is therefore not faithful; the loader never does it (ADR D8).
- HF's pooled form is exact and zero-bubble-compatible (its gradient on layer l is `α·E·C_i/(L_m²·T²)` per token
  probability, and C is known once the microbatch's forward is done, before any of its backward), but `C` crosses
  stages: the FWD frame carries the running integer vector `Σ_{l<cut} c_il` (8E bytes) and the BWD frame carries `C`
  back. 8E bytes per hop per microbatch is 1,024 B for Qwen3 against 16 MiB of bf16 boundary at T 4096, h 2048
  (6·10⁻⁵). It is a runtime frame field and buys fidelity to HF's fine-tuning trainer only: FOLLOW-UP.
- Global batch (Qwen3's training): microbatch m's aux gradient needs counts from every microbatch of the step. The
  input gradient of the MoE layer depends on them, so it cannot be deferred to W: every forward of the step precedes
  every MoE backward. A GPipe-ordered step costs up to `(S−1)/(M+S−1)` of bubble (15.8% at S 4, M 16; 17.9% at S 8,
  M 32; DERIVED upper bound, the solver would compute it) and holds every microbatch's boundary until its B. A group
  of G consecutive microbatches (Qiu et al. 2025) interpolates (G = 1 is per-microbatch, G = M global) with the same
  barrier per group. FOLLOW-UP (a scheduler constraint, its own decision).
- Round 2. Qiu et al.'s "DP-global" scope (counts all-reduced over the micro-batches that the data-parallel ranks run
  at the same time) is, here, the set of microbatches a wide stage's peers happen to run together: it depends on w
  and on IWRR ownership, so it is a topology-chosen reduction (forbidden by the product rule). The admissible form is
  a group of G consecutive microbatch indices fixed by the recipe, never by w; it is the FOLLOW-UP above, its bubble
  between 0 (G = 1) and the GPipe bound (G = M), computed by the solver, not estimated here.
- Round 2, an aux-only second backward (CE backward first, then a backward of the global-batch aux over the router
  path, its parameter gradients added after the CE ones in a written order): exact, and it lifts the barrier off the
  CE backward, but the aux gradient of layer l reaches every upstream layer through input gradients, so the second
  pass is a second input-gradient sweep of the model: about one more B per microbatch, ≈ +33% of F + B + W at
  F:B:W 1:1:1 (HYPOTHESIS), and it still needs every forward's counts before it starts. REJECTED: above the ≤ 17.9%
  barrier bubble it would replace.
- Lagged global counts (the previous step's): exact, but not a recipe any checkpoint was trained with; step 0 needs a
  defined value; the counts become checkpointed state. REJECTED (no fidelity gain over per-microbatch for any named
  family, one more state).
- DeepSeek-V3 sequence-wise: per sequence u of t tokens, `(E/(k·t²))·Σ_i c_iu·S_iu` with `q = s/Σ_j s_j`, mean over the
  b sequences of the microbatch (the paper's batch reduction is not published: HYPOTHESIS, a recipe field). Sequences
  never cross microbatches, so it is local.
- Router z-loss (ST-MoE, OLMoE): `z_l = (1/T)·Σ_t logsumexp(logits_t)²`, injected as its own coefficient.

## 6. Selection-bias update (auxiliary-loss-free balancing)

DeepSeek-V3 §2.1.2 (Wang et al. 2024): a per-layer, per-expert bias `b_l` is added to the selection key only (never to
the weights, never differentiated), and after each step `b_i ← b_i + γ·sign(c̄ − c_i)`.

- Exact form: `c_i` = integer count of slots on expert i over the step's global batch (every microbatch, every wide
  peer); `c̄ = T_g·k/E`; the sign is taken in integers, `sign(T_g·k − E·c_i)` (no float mean, ties give 0); then one
  fp32 add of `±γ` or 0. The counts are integers, so their sum is order-free; the update is elementwise.
- Placement: at the step end, after AdamW; step t+1's forward of that layer waits for it, which is the existing wait
  for a unit's working copy (ADR-0006) or for zero1's broadcast. No new barrier.
- Transport (round 2; round 1 named it for pieces only and wrongly placed it in a zero1 "small-tensor unit", which
  zero1 does not have). `sel_bias` is a stage tensor updated by its own rule instead of AdamW: its per-microbatch
  "gradient" is that microbatch's int64 count vector `c_l,m` (E values, `Σ_i c_l,m,i = T_mb·k`), and its owner applies
  the sign update. It travels wherever a trainable tensor's gradient and new value travel:
  - width 1: local;
  - `zero1`: the stage's ownership list is its trainable names in order, then its selection biases in layer order (a
    dense stage's list is unchanged); peer `k` owns `names[k::w]`; each peer ships, with a microbatch's shard
    gradients to `k`, the count vectors of the biases `k` owns; `k` sums them (integers: exact in any order; folded in
    microbatch order like every shard), updates after AdamW and broadcasts the bias with its new shard; the shard
    checkpoint and the replica-agreement check cover it;
  - `pieces`: the bias is a tensor of its block's small-tensor unit, cut like any tensor of it (`Q_t` = 128 elements);
    the count vector is cut at the same boundaries and rides the unit's GRADS frame in an int64 lane; the piece owner
    folds, updates elementwise and gathers the piece with the unit's working copy; RUNTIME-I12's logical checkpoint
    and manifest include it;
  - `check_frame`: a count lane that is not int64, whose shape differs from the piece's (or E under zero1), whose
    (step, mb, sender) breaks the GRADS rules, that holds a negative count, or (zero1, whole vector) whose sum differs
    from `T_mb·k` is fatal; a missing count vector at the owner is fatal like a missing gradient.
  The alternative the round-2 reviewer proposed, replicating the bias on every peer and sending each peer's counts to
  all others, needs a new frame type and `w−1` sends per peer for the same bytes; rejected for the owner path, which
  adds one lane to frames that already flow and reuses the existing gathers, checkpoints and replica checks.
- Bytes: counts `L_m·E·8` per microbatch per peer to the owners and the bias `L_m·E·4` in the gathers, per stage
  slice: 118,784 B per microbatch for all of DeepSeek-V3 (58 MoE layers, E 256; FACT since round 3, §3),
  184,320 B for Kimi K2 (60, 384; FACT, §3), against megabytes of boundary and gigabytes of gradient: ~0.
- State: `b_l` (fp32, E per layer) joins the weights in the checkpoint, the manifest and the sha256 identity state;
  a checkpoint's `e_score_correction_bias` loads into it byte for byte.

## 7. Optimizers that are not elementwise

ADR-0006's piece argument uses "AdamW is elementwise". Muon (Kimi K2's MuonClip) orthogonalizes whole matrices, so a
dim-0 piece of a matrix cannot be updated alone. Each expert's projection is an independent matrix, so whole-expert
units would be the natural pieces. Not this ADR's decision (the optimizer is the recipe's); recorded as a limit of
"parallelism inherits it unchanged": it holds for elementwise optimizers.

## 8. Fused expert layouts and the grouped GEMM

- Orientation table (round 2; round 1 assumed one layout). A fused tensor loads only through a row of this table,
  keyed on (model_type, tensor); a fused tensor of any other key is refused even when its shapes fit, because a
  square expert (gpt-oss: h = f = 2880) or h = 2f lets a transposed map pass every shape check. The first row is FACT
  since round 3 (transformers 5.19.0 `MixtralExperts`, `Qwen2MoeExperts`, `Qwen3MoeExperts` read locally); the others
  are HYPOTHESIS until E5c reads HF's own objects. The table applies to the native loader (ADR decision 6b); read the
  other way it is the hub-per-expert to v5-fused byte map that ADR-0009 D7 needs (ADR R13, decision 10):

| model_type | `gate_up_proj` | `down_proj` | per-expert map |
|---|---|---|---|
| mixtral, qwen2_moe, qwen3_moe (v5 fused) | (E, 2f, h), gate rows first | (E, h, f) | `gate=G[j,:f]`, `up=G[j,f:]`, `down=D[j]` |
| gpt_oss (refused model; row for the map) | (E, h, 2f), gate/up interleaved | (E, f, h) | `gate=G[j,:,0::2]ᵀ`, `up=G[j,:,1::2]ᵀ`, `down=D[j]ᵀ` |
| llama4, qwen3_vl_moe (refused models) | (E, h, 2f), chunked | (E, f, h) | `gate=G[j,:,:f]ᵀ`, `up=G[j,:,f:]ᵀ`, `down=D[j]ᵀ` |
| qwen3_5_moe (not mapped) | (E, 2f, h) | (E, h, f) | refused until its config map exists |

- Each map is a strided view copied element for element; no arithmetic. E5c checks semantics, not bijectivity (a
  transposed map is a bijection too): the fused source is taken from HF's own v5 objects (pinned transformers), and
  the loaded per-expert bytes must equal HF's per-expert view of the same model, at h = f and h = 2f.
- A4 (3-D bank as the parameter, a grouped GEMM) re-ranked in round 2. Under D0 a grouped GEMM per (microbatch,
  projection) is one kernel call on a fixed shape (the per-expert offsets are part of the shape, and they are a
  function of the microbatch), so the oracle and every stage running the same grouped call get the same bits without
  any per-segment independence; independence matters only for equality with the per-expert loop (another recipe) or
  for EP, which regroups segments. What A4 still needs: (a) a deferred-dW record per (microbatch, projection) that
  reruns the same grouped dW call (a `wgrad` extension, MODEL-I1 by construction); (b) the CUDA kernel's run-to-run
  determinism (R1); (c) dim-0 pieces of a 3-D tensor at `Q_t = lcm(row, 128)` with the row the trailing product.
  `torch._grouped_mm` exists on CPU and MPS here and equals a per-segment mm on one case (P11), with device-side
  offsets (no host sync); it is a private API. Adopt at M1 if per-expert launches are measured > 25% of the layer's
  time; ScatterMoE (Tan et al. 2024) and MegaBlocks are the other M1 candidates under the same rule.
## 9. MXFP4 experts (GPT-OSS)

An MXFP4 value is an E2M1 element (magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6) times an E8M0 block scale 2^(s−127). The
product has at most 2 significant bits, so it is exact in bf16 and fp32 whenever it is inside their range: exact for
s ≤ 252 (6·2^125 < 3.4·10³⁸), and 2^−128 (the smallest nonzero, s 0) is a bf16 and fp32 subnormal. A decode that
re-encodes to the source bytes is a byte-exact load; s ≥ 253 with a magnitude ≥ 6 overflows and is refused. FOLLOW-UP
with GPT-OSS's attention pieces.

## 10. What the dense loader drops today (`hf.py`, `rep.unused`)

FACT: `load_into` sends to `rep.unused` (a) a block tensor whose suffix is not in `_LLAMA_BLOCK` (`hf.py:272-275`),
(b) every tensor of a layer `≥ cfg.n_layer` (`hf.py:254-256`), (c) a top-level tensor not in `_LLAMA_TOP`
(`hf.py:279-281`), (d) an `assign` target that does not exist (`hf.py:233-235`). Per accepted model_type
(HYPOTHESIS from HF modeling code, recalled; E5 lists them from real index.json files where present):

| model_type | Reaches `unused` | Class |
|---|---|---|
| llama, mistral | `self_attn.rotary_emb.inv_freq` in older conversions | buffer, recomputed from `rope_theta` |
| qwen2 | `self_attn.{q,k,v}_proj.bias` | dropped parameters (silent today) |
| qwen3 | `self_attn.{q,k}_norm.weight` | dropped parameters (silent today) |
| phi3 | `qkv_proj`, `gate_up_proj` | unplaced, but q/k/v and gate/up are then missing: refused today |
| starcoder2, olmo, gemma, gemma2 | names or activation the map lacks | refused today (missing tensors or `hidden_act`) |
| granite | nothing | config multipliers ignored: a house defect, not a tensor |
| any | layers ≥ `n_layer` when a recipe sets `n_layer` below the checkpoint's | explicit truncation |

So the refusal of unplaced parameters changes the outcome only for qwen2 and qwen3 dense (whose drops are MODEL-I4
violations today) and is lifted for them by `qkv_bias`/`qk_norm`. Buffers and explicit truncation are reported, not
refused.

## 11. References (verified by the round-1 and round-2 reviewers; verdicts in the ADR)

Mixtral (arXiv:2401.04088): 8 experts, top-2. Qwen3 technical report (arXiv:2505.09388): Qwen3-30B-A3B 128 experts, 8
active, no shared expert, global-batch load balancing. DeepSeek-V3 (arXiv:2412.19437): 1 shared + 256 routed, top-8,
sigmoid, auxiliary-loss-free bias, group-limited routing, sequence-wise aux, MTP. Wang et al. 2024
(arXiv:2408.15664): loss-free balancing. OLMoE (arXiv:2409.02060): 64 experts, top-8, load balancing + z-loss. ST-MoE
(arXiv:2202.08906): router z-loss. Kimi K2 (arXiv:2507.20534): 384 experts, top-8, MuonClip. gpt-oss model card
(OpenAI 2025): 128/32 experts, top-4, attention sinks, banded attention, MXFP4. Llama 4 (Meta 2025): Scout 16 /
Maverick 128 experts, interleaved MoE, shared expert, sigmoid top-1. Qiu et al. 2025 (arXiv:2501.11873): micro-batch
vs global-batch balancing. MegaBlocks (arXiv:2211.15841). He / Thinking Machines 2025, batch-invariant kernels.
Round 2 (fetched by the reviewers, 2026-10-10): HF transformers main `modeling_mixtral.py` (`torch.topk`, a loop over
hit experts, `index_add_`, 3-D `gate_up_proj` experts); transformers v4.46.0 `modeling_mixtral.py`
(`MixtralSparseMoeBlock.forward` returns `(final_hidden_states, router_logits)`, `MixtralDecoderLayer` unpacks it);
HF main `modeling_deepseek_v3.py` (fp32 router, `-inf` group masking, `+1e-20`); the config.json files of
Qwen3-30B-A3B, Mixtral-8x7B-v0.1, Kimi-K2-Instruct, gpt-oss-20b and Qwen1.5-MoE-A2.7B (§3); pytorch/torchtitan PR
#1974 (scatter_add removed; output spread 0.0297 → 0 over 100 CUDA bf16 runs). Cited and not verified (left out of
every claim): Megatron-Core's router options and `get_updated_expert_bias`, GLM-4.5 (arXiv:2508.06471) and MiniMax-M2
configs, DeepSeek-V4's routing (NVIDIA Megatron-Bridge docs; top-6 of 384 from secondary sources), LongCat-Flash
(arXiv:2509.01322), MoE++ (arXiv:2410.07348), expert choice (arXiv:2202.09368), BASE layers (arXiv:2103.16716),
ScatterMoE (arXiv:2403.08245), VeOmni's v5 loading notes.
Round 3 (verified and placed in the ADR's Ideas considered): transformers 5.19.0 `modeling_mixtral.py`,
`modeling_qwen2_moe.py`, `modeling_qwen3_moe.py`, `modeling_deepseek_v3.py`, `modeling_gpt_oss.py`, `modeling_llama4.py`
and `conversion_mapping.py` (read locally from the uv cache); the DeepSeek-V3, Kimi-K2-Instruct and gpt-oss-20b
config.json files; DeepSeek-V4 (arXiv:2606.19348); NVIDIA Megatron-LM `megatron/core/transformer/moe/moe_utils.py`.

## 12. What each hazard separates (round 2)

Round 1 called six hazards "places where the pipeline path can compute other bits than the oracle". P8–P11 show
that only some are. Columns: oracle vs stage on one device map; across backends (a heterogeneous mix; DESIGN §3 makes
the device map part of the spec); definedness (the bits depend on an unwritten engine or kernel choice).

| Hazard | Oracle vs stage, one map | Across backends | Rule |
|---|---|---|---|
| `topk` tie order | identical on both paths (P8 14/14) | CPU vs MPS 436/20,000 sets differ; stable sort 0 (P10) | D3 (definedness, cross-backend) |
| ≥ 3-term sum order (renorm, combine, dispatch backward) | identical when both paths run the same code (P9); differs when one path folds otherwise (P5, P6 controls) | kernel-dependent (P5: `sum` matched the fold on CPU only) | D4, D5, D7 |
| autograd fan-out of ≥ 3 consumers | identical, structural on the CPU engine (P6 6/6, P9 40/40) | engine order; CUDA device threads HYPOTHESIS | D1 (definedness) |
| expert GEMM row set and split | differs when one path splits or joins rows (P3; E4's two-microbatch W) | — | D6 |
| unrouted expert: `None` | oracle raises without `allow_unused`; with it both skip alike (P9) | — | D9: wide stages need dense gradients (or a new absent marker, A7); the guards stay |
| aux entry point and scope | same on both paths when both inject alike; a global scope needs every microbatch's counts | — | D8 |
| `index_add_` / `scatter_add_` | CPU deterministic | MPS: refused under deterministic algorithms (also `index_add_` with unique indices), else 20/20 distinct with duplicates (P11) | D0 (no atomics) |
| RNG (router jitter) | differs unless the generator is replayed | — | refused |

So HF's eager block is exact between oracle and stage on one CPU map (P9). Against it this ADR's layer buys: dense
gradients for the wide-stage paths (D9), selection identical across CPU and MPS (D3), no accumulate-scatter (it
cannot run on MPS under deterministic algorithms, P11), aux terms injected per layer (D8), and written orders that do
not depend on the engine. That, not P6's one-sided controls, is why HF's forward is not run (§14).

## 13. The loader's config map is an allowlist (round 2; round 3: built only under ADR decision 6b)

Round 3: this section applies only if ADR-0009 is not granted. Under ADR-0009 (6a) the native HF path is deleted
(ADR-0009 D11) and none of the fields below is built; the RoPE delta below is MEASURED by ADR-0009 (Evidence 1:
llama max|Δlogit| 7.86e-3 against transformers, 0 with `rotate_half`). Where this section says "transformers
(pinned, test-only)", the pin is 5.19.0 (ADR R12).

FACT (a62e4eb): `config_from_hf` reads no `rms_norm_eps` or `head_dim` (`hf.py:24-120`); `RMSNorm` has `eps=1e-6`
(`arch.py:22`); `Block` sets `hd = d // nh` (`arch.py:119`); `apply_rope` rotates adjacent pairs (`arch.py:82-86`)
and `load_into` copies `q_proj`/`k_proj` without a permutation (`hf.py:258-270`). HYPOTHESIS (HF code recalled; E5b
measures it): HF Llama-family attention rotates halves (`rotate_half`), so every RoPE family the loader accepts today
computes another function than its checkpoint (a house defect older than this ADR), and Mixtral (eps 1e-5) would too.
Qwen3-30B-A3B (head_dim 128, h/nh 64) cannot be built at all.

Rule: every key of config.json is (a) mapped, (b) inert (it changes no computed bit: token ids, `use_cache`,
`torch_dtype`, `transformers_version`, `architectures`, `model_type`, `initializer_range`, `max_position_embeddings`
(a bound), `router_aux_loss_coef` (never mapped, D8), `output_router_logits` false, `_name_or_path`, `tie_word_embeddings`
(mapped)), or (c) accepted only at a listed value: `rope_scaling` null, `attention_bias` false (Qwen2's q/k/v biases are
mapped by `qkv_bias`), `mlp_bias` false, `attention_dropout` 0, `router_jitter_noise` 0, `partial_rotary_factor`
absent or 1, `pretraining_tp` 1, `use_sliding_window` false (Qwen) or `sliding_window` null or ≥ `seq_len`. Any other
key or value is refused, naming it. Newly mapped: `head_dim` (q and o widths `n_head·head_dim`, k and v
`n_kv·head_dim`; default `d_model/n_head`, so today's bits are unchanged), `rms_norm_eps` → `norm_eps` (default 1e-6,
today's), and the RoPE convention → `rope_style` (`"pairs"`, today's default; `"half"` for every HF RoPE family).

Names (unchanged from round 1): Mixtral `block_sparse_moe.gate` → `mlp.router`, `block_sparse_moe.experts.j.w1/w3/w2`
→ `mlp.experts.j.gate/up/down`; Qwen2/3-MoE `mlp.gate` → `mlp.router`, `mlp.experts.j.{gate,up,down}_proj` →
`mlp.experts.j.{gate,up,down}`, `mlp.shared_expert.*` → `mlp.shared_expert.*`, `mlp.shared_expert_gate` →
`mlp.shared_expert_gate`; Qwen2 `self_attn.{q,k,v}_proj.bias` → the `qkv_bias` tensors; Qwen3 `self_attn.{q,k}_norm` →
the per-head `qk_norm` weights. MoE layers from `decoder_sparse_step` and `mlp_only_layers` (Qwen), every layer
(Mixtral).

Checks (ADR E5): E5a, the real config.json files as fixtures (with URL and sha256) map to a hand-written table of
expected fields, and the meta skeleton's shapes equal each checkpoint's safetensors header shapes (recorded once);
E5b, function fidelity in fp32 against transformers (pinned, test-only): max |Δlogit| / max |logit| ≤ 1e-4 for tiny
models carrying each family's real function-changing keys, while `rope_style "pairs"`, a transposed `down`, swapped
gate/up and a wrong `head_dim` each give ≥ 1e-1 (bounds HYPOTHESIS until E5b's first run records both sides). E5b is
fidelity to HF, never identity evidence (identity stays the exact oracle).

## 14. The HF-object forward contract (round 2; round 3 corrections first)

Round 3 corrections (they override the round-2 text below where the two differ). ADR-0009 is in the tree; its D8 is
the seam. The table is keyed on (HF class, pinned transformers version) and has rows for 5.19.0 only:
`MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`, `Qwen3MoeSparseMoeBlock`. FACT (5.19.0, read locally): each
block's `forward` returns one tensor; `gate` is a `*TopKRouter` holding a raw `weight` Parameter (no `nn.Linear`), its
logits recorded by `_can_record_outputs` (refused anyway: `output_router_logits`); experts are `*Experts` modules with
`gate_up_proj` (E, 2f, h), gate rows first, and `down_proj` (E, h, f). The v4 rows and the v4 arms of E2-HF and
MODEL-I8 are dropped (no product path runs 4.x, ADR R12). The replaced forward per call:
- `G = gate_up_proj.unbind(0)`, `D = down_proj.unbind(0)` (one node each; backward a stack, a byte move);
- per expert j: `g, u = G[j].chunk(2, dim=0)` (one node; backward a cat, a byte move), then `F.linear(x_j, g)`,
  `F.linear(x_j, u)` (two f-wide GEMMs, the shapes, strides and dtypes of `MoEMLP`'s gate and up `SplitLinear`s) and
  `F.linear(a_j, D[j])`. Never `G[j][:f]` and `G[j][f:]`: two slice backwards are summed by autograd into one
  zero-filled tensor, and `+0 + (−0) = +0`, so a −0 element of a dW would lose its sign against `MoEMLP`'s (DERIVED
  from IEEE 754; E2-HF's must-fail arm constructs it). HF's own single 2f-wide GEMM is another shape: not used;
- every expert is called (D6), so every unbind output has a gradient and no zero fill enters the stack;
- the banks' dW runs in B (autograd's `mm`, equal to `SplitLinear`'s deferred GEMM by MODEL-I1); its cost is §17.
Exchange units on this path: `gate_up_proj` and `down_proj` (2 per MoE layer), each cut along dim 0 at
`Q_t = lcm(trailing product, 128)`, i.e. on whole experts (ADR-0006's dim-0 rule with the row being one dim-0
index; P7 measured 2-D tensors only, so E6 runs an HF v5 arm); their fp32 gradients are produced in B and ship when
W(mb) starts, like the Embed's tables. The trace hazard of ADR R14 is open.

Round 2 text. Under an HF-modules path (ADR-0009, absent from this tree in round 2) round 1 substituted `MoEMLP` for HF's block. That breaks
two things the reviewers verified: the v4 block returns `(hidden, router_logits)` and the decoder layer unpacks it;
and `MoEMLP`'s names (`experts.j.gate`) are not HF's (`experts.j.w1`, or v5's 3-D `experts.gate_up_proj`), which
ADR-0009's loader, ADR-0007's skeleton and init record and the checkpoints key on. Round 2 keeps HF's module objects
and parameters and replaces only the forward:

- A table keyed on (HF class, transformers version) names, per block: where the router, experts, shared expert and
  gate live, the expert layout (v4 per-expert `nn.Linear`; v5 3-D tensors, oriented by §8's table), the fields of
  derivation §3 it implies, and the call contract it returns (v4: `(hidden, router_logits)` with the fp32 logits;
  v5: as the table records it, HYPOTHESIS until E2-HF reads the class).
- The forward is the same exact function as `MoEMLP`'s (D0–D10), reading the block's parameters through a view: v4
  per-expert Linears are converted by `wgrad.convert` (deferred W as on the native path); v5 3-D tensors are split by
  one `unbind(0)` per call (one autograd node, whose backward stacks the per-expert dW: a byte move), each expert's
  GEMM the same `F.linear` on its rows, dW computed in B (exact; zero-bubble deferral of a 3-D bank is A4's (a)).
- Aux values and counts are recorded inside the function; the D8 injection never depends on the HF decoder layer.
  `output_router_logits = true`, a class or version outside the table, and a block attribute the function does not
  read (jitter, an unmapped bias) are refused.
- Names stay HF's: the skeleton, the init record, the exchange units (an MoE layer's expert tensors of one projection,
  or a v5 3-D tensor, form one unit) and the checkpoints see HF's state_dict; the native byte map does not apply.
- Check (ADR E2-HF): for tiny v4 and v5 models of the three families, the replaced block's output, aux values and every
  gradient equal `MoEMLP`'s on byte-mapped parameters, bit for bit, fp32 and bf16; oracle vs a 2-stage capture path
  over the HF layer list; then the identity arm through ADR-0009's adapter.

## 15. Cost terms round 1 left out (round 2)

- Launches. D6 runs 3 GEMMs per expert per phase: 12E launches per MoE layer per microbatch over F, F′ (recompute), B
  and W. At 5 µs (HYPOTHESIS, H100 launch latency): 0.48 ms (Mixtral), 3.60 ms (Qwen1.5-MoE), 7.68 ms (Qwen3-30B-A3B)
  against 32.3, 7.05 and 4.66 ms of GEMM time at 400 TFLOP/s: fine-grained MoE is launch-bound on GPU.
- Host syncs. Slicing `n_e` rows per expert needs the counts on the host: one device-to-host read per MoE layer in F
  and in F′ (B and W reuse the saved sizes). HF's eager loop needs at least as many (`torch.where` on a mask is a
  nonzero, one per hit expert). Exposed cost per sync ≈ one launch latency plus host issue time, ~10–50 µs
  (HYPOTHESIS): ≤ 0.1 ms per layer per microbatch, ≤ 1.3% of Qwen3's launch term.
- H2 restated. Against HF's eager per-expert loop the rules add no byte (the same dense gradient tensors), no host
  sync (1 vs ≥ 1 per F) and no expert GEMM; they add O(T·k·h) elementwise work (fp32 combine, dispatch gather) and a
  sort over E instead of `topk`. Against a grouped GEMM with device-side offsets they cost 12(E − 1) extra launches and
  2 syncs per layer per microbatch (Qwen3: up to 7.68 ms against 4.66 ms, HYPOTHESIS): A4's question, measured at M1.
- H3 restated. S1 and S2 use the analytic `c_s = 8·P^F_s·T_mb/F` (SCHEDULER-I16), which omits launches and syncs, so
  for an MoE stage it is below the real compute time: S1 (bytes/c_s ≤ cap) and S2 (`E − window ≤ ε·C`) are then
  conservative, refusing some feasible plans and never admitting an infeasible one. They would admit wrongly only if
  an MoE stage ran faster than its FLOP count at the profiled rate (ADR F7). The solve already uses profiled per-op
  costs (`schedule_for_spec`), and the runtime re-plan (WT_TELEM) corrects a cut the analytic `layer_costs`
  misbalanced; both are performance, never bits.

## 16. Ideas considered in rounds 0–2 (rounds 0–1 moved from the ADR in round 2, round 2 in round 3)

Round 0: ADOPTED stable-sort ties, fp32 router default (round 2: per family), token-ordered gather and per-(mb,
expert) GEMM, no atomics (torchtitan #1974: spread 0.0297 → 0), +0 for unrouted experts, fan-out fold, dropless,
per-microbatch aux. REJECTED: integer key + `topk`, router jitter, per-expert units, empty-piece flag, data-dependent
gather, dense-masked. FOLLOW-UP: per-op regather, owner-side W. SEPARATE ADR: EP.

| Round-1 idea (source, verified) | Verdict | Reason and number |
|---|---|---|
| Enumerated routing family (Mixtral 2401.04088, DeepSeek-V3 2412.19437, gpt-oss card, Llama 4) | ADOPTED, D2–D4 | 7 of 7 named families' MoE layers instead of 3; no new mechanism |
| Weight at the expert input (HF Llama 4) | ADOPTED, D4 `apply` | Otherwise a Llama 4 fine-tune computes another function |
| Loss-free bias from integer counts (2412.19437 §2.1.2; 2408.15664; Kimi K2 2507.20534) | ADOPTED, D10 | Exact (integers), elementwise, on an existing wait |
| Router z-loss per layer (ST-MoE 2202.08906, OLMoE 2409.02060) | ADOPTED, D8 | Same injection as lbl |
| Fused 3-D experts loaded by byte slicing | ADOPTED, Loader (round 2: orientation table) | A load-time copy |
| MuonClip is not elementwise (Kimi K2) | ADOPTED as a limit | §7 |
| v5 experts hook with an exact experts function | REJECTED, A1′ (round 2: re-ranked, §14) | Leaves HF's router |
| Grouped GEMM / 3-D bank (MegaBlocks 2211.15841, `torch._grouped_mm`, CUTLASS) | FOLLOW-UP at M1, A4 (round 2: §8) | Launch share > 25% |
| Global-batch or grouped LBL (Qwen3 2505.09388; Qiu et al. 2501.11873) | FOLLOW-UP | ≤ 17.9% bubble at S 8, M 32 |
| Lagged global counts | REJECTED | No named family trained with it; one more state |
| Batch-invariant kernels (Thinking Machines 2025) | REJECTED now | D6 removes P3's hazard at zero cost |
| MXFP4 experts decoded losslessly | FOLLOW-UP with GPT-OSS attention | Exact for scale exponents ≤ 252 (§9) |
| HF pooled aux with a frame field | FOLLOW-UP | 8E B per hop per microbatch; HF-trainer fidelity only |

Round 2 (reviewers' ideas; moved here from the ADR in round 3, verdicts unchanged; the Megatron-Core row is
superseded by the round-3 verified entry in the ADR's Ideas considered):

| Idea (source; verified?) | Verdict | Reason and number |
|---|---|---|
| Rely on v5's fused 3-D experts under an HF-modules path (dense gradient without substitution) (HF main `MixtralExperts`, verified) | ADOPTED in part, §14 | HF's 3-D tensors stay, viewed per expert by `unbind`; HF's forward is not run (P10, P11) |
| Recipe rows confirmed; DeepSeek-V3 fp32 router, `-inf` masking, `+1e-20` (HF main, verified) | ADOPTED, D2, D3 | Router dtype per family; DeepSeek `norm_eps` 1e-20 |
| Cross-backend tie test for every selection primitive (probe.py; product rule) | ADOPTED, P10, E1 | `topk` CPU vs MPS 436/20,000; stable sort 0: D3's actual reason |
| Qiu et al.'s DP-global LBL across wide peers (2501.11873, verified) | REJECTED as stated | The group would be the microbatches peers run together: chosen by w and IWRR (forbidden); a recipe-fixed G is the FOLLOW-UP |
| Megatron-Core's router as prior art; `get_updated_expert_bias` float mean (not verified) | Not placed (unverified); superseded in round 3 | E4's float-mean control stands without it |
| MPS as a non-gating observation of atomics and sort determinism (torchtitan #1974, verified) | ADOPTED, P11, E1 | MPS duplicate scatter: refused or 20/20 distinct; stable sort 1/20 |
| Device-side counts with a grouped GEMM (`torch._grouped_mm`, MegaBlocks); ScatterMoE (2403.08245, not verified) | FOLLOW-UP at M1, A4 re-ranked | Exact by D0 when both paths run the same call; P11 shows it here; private API, CUDA facts unmeasurable |
| Aux-only second backward for global-batch LBL | REJECTED | ≈ +33% compute (one more input-gradient sweep, HYPOTHESIS) vs ≤ 17.9% barrier bubble |
| Keep HF module objects, replace only the forward (v4.46 tuple contract, verified) | ADOPTED, §14 | Removes the name, layout and contract mismatch of substitution (round 3: rows at the 5.19.0 pin only) |
| Recipe values from real configs now (Kimi K2, gpt-oss-20b, Qwen1.5-MoE configs, fetched) | ADOPTED, §3 | FACT: Kimi K2 60 MoE layers, `use_sliding_window` false, gpt-oss aux coef 0.9 (never mapped) |
| GLM-4.5 / MiniMax-M2 as a real checkpoint for the sigmoid row (configs recalled, not verified) | FOLLOW-UP | Needs partial RoPE and its config read first; one more map, no mechanism |
| DeepSeek-V4 sqrtsoftplus and hash routing (secondary sources) | FOLLOW-UP, refused now | A score row is elementwise; hash routing needs token ids in the FWD frame (round 3: primary source verified, verdict unchanged) |
| Zero-computation experts (LongCat-Flash, MoE++); expert choice, BASE layers (not verified) | ADOPTED as refusals | Data-dependent compute, or a token's routing set by the batch; no named family uses them |

Verified by the round-2 reviewers and placed (§11): HF `modeling_mixtral.py` (main, v4.46.0), `modeling_deepseek_v3.py`,
five config.json files, torchtitan #1974, Qiu et al. 2025. Not verified in round 2, left out of every claim:
GLM-4.5, MiniMax-M2, LongCat-Flash, MoE++, expert choice, BASE, ScatterMoE, VeOmni's notes.

## 17. Cost of the HF v5 path (round 3)

Under ADR-0009 the routed experts are HF's 3-D banks; their dW is computed by autograd in B (§14), not deferred to W.
Inputs as the ADR's tables (T 4096, F 400 TFLOP/s HYPOTHESIS, configs of §3), no recompute, F:B:W = 2:2:2 FLOP per
active parameter and token (each linear's forward, input gradient and weight gradient), attention's S² terms and
launches omitted. DERIVED by this script (`python3 -I`, run in round 3; it reproduces the ADR's c column 32.30, 7.05,
4.66 ms, so its parameter counts match `numbers.py`'s):

```python
F=400e12; T=4096
models={  # attn, router, routed active (k·3hf), shared, per expert (3hf), E, native S1 T_mb at 400 Gbps
 'Mixtral':(4096*(4096+1024+1024+4096), 4096*8, 2*3*4096*14336, 0, 3*4096*14336, 8, 5521),
 'Qwen1.5-MoE':(4*2048*2048, 2048*60, 4*3*2048*1408, 3*2048*5632+2048, 3*2048*1408, 60, 9939),
 'Qwen3-30B-A3B':(2048*4096*2+2048*512*2, 2048*128, 8*3*2048*768, 0, 3*2048*768, 128, 16431)}
for n,(a,r,ra,sh,pe,E,s1) in models.items():
    PF=a+r+ra+sh; Ps=a+r+E*pe+sh
    t=2*PF*T/F*1e3; d=2*ra*T/F*1e3          # ms: one of F, B, W; the routed dW
    tB, tW = t+d, t-d
    zb_n, zb_h, f1 = t, t+tB-tW, 3*t        # ZB-H1 bubble per (S-1): native, HF; 1F1B
    share=E*pe/Ps                           # expert share of P_s, shipped at 4 B instead of 2
    print(n, round(d,3), round(tB,3), round(tW,3), round(zb_h,2), round((f1-zb_h)/(f1-zb_n),3), round(s1*(1+share)))
```

Output: Mixtral Δ 7.216 ms, t_B 15.291, t_W 0.860, ZB-H1 22.51 ms per (S−1) against 8.08 native and 24.23 for
1F1B (10.6% of zero-bubble's reduction kept), S1 10,882; Qwen1.5-MoE 0.709, 2.472, 1.055, 3.18 (5.29; 59.8%), 18,981;
Qwen3-30B-A3B 0.773, 1.938, 0.392, 2.71 (3.50; 33.6%), 32,357.

- ZB-H1's bubble `(S−1)(t_F + t_B − t_W)` and 1F1B's `(S−1)(t_F + t_B + t_W)` are Qi et al. 2023's (Zero Bubble
  Pipeline Parallelism); the planner's ZB-2p solve (SPEC-scheduler §2.2) fills W into bubbles, so the bubble grows by
  up to `2Δ·(S−1)` per MoE layer when W runs short. The solve measures it; these numbers bound its direction.
- Bytes: autograd leaves an fp32 gradient on an fp32 master under autocast (the cast's backward), shipped as produced
  (SPEC-runtime task 5, like the Embed's tables), so the expert units cost `b = 4` against the native path's 2: S1's
  threshold scales by `1 + P_experts/P_s` (×1.97, ×1.91, ×1.97). DERIVED; E6's HF arm measures the frames.
- Memory (corrected in round 5, §22; in-flight count made explicit in round 7, §23.1): the routed pending W is 0 on
  this path (X_e and dY_e are freed at B's end), but B leaves each bank's gradient dense in the master dtype,
  `b_m·E·3hf` bytes per MoE layer, held from the end of B(mb) until W(mb) starts, when it ships (pieces) or is
  assembled (width 1). Per stage and peer the live hold is `H = m_BW·b_m·Σ_{MoE layers} E·3hf`, where `m_BW` is the
  number of that peer's microbatches whose B has ended and whose W has not started (the zero-bubble order's in-flight
  count, ≥ 1 whenever W is deferred past the next F or B). Per layer and microbatch: 5,376 / 1,980 / 2,304 MiB at
  `b_m` 4, against the native path's 800 / 260 / 400 MiB of pending W (×6.72 / ×7.62 / ×5.76); a 4-layer Mixtral
  stage holds 22.5 GB at `m_BW` 1, 45.1 GB at 2, 90.2 GB at 4. It is a cost of this path, not a saving, and the
  planner charges it (SCHEDULER-I21, §22).
- What removes both costs: deferring the bank's dW to W with a per-(microbatch, projection) record (ADR Ideas
  considered, FOLLOW-UP), which also yields the gradient in the GEMM's dtype. At S 1 and w 1 neither cost exists.

## 18. Current architecture facts (moved from the ADR in round 4, unchanged)

FACTS (files read on a62e4eb; M2b code read on `factory/m2b` db3f4e0, built, not merged):
- `arch.py` dense MLPs, `Block` with `hd = d // nh`, `RMSNorm(eps=1e-6)`, `apply_rope` on adjacent pairs; `gpt.py:335-357`
  then `wgrad.convert` (every unshared `nn.Linear` → `SplitLinear`, which records `(X, dY)` per call under `capture()`;
  `flush` computes `dW = dYᵀX` per record, summing a parameter's records in store order, `wgrad.py:57-108`).
- `reference.py:77-88`: `autograd.grad` per microbatch, no `allow_unused`; fold in microbatch order. `stage.py:1291`
  raises when a trainable parameter received no gradient in B/W. `state.py`'s manifest: weights, AdamW, RNG, loss hex.
- ZeRO-1 peer `k` owns `params[k::w]` (SPEC-runtime §2). M2b `pieces.py:106-140` `exchange_units`: a block's small
  tensors, then one unit per `SplitLinear`; `check_frame` validates frames against the unit table; the Embed's tables
  are one unit whose fp32 gradients, produced in B, ship when W(mb) starts (SPEC-runtime task 5).
- `hf.py:24-120` reads no `rms_norm_eps`, `head_dim` or RoPE convention; `pyproject.toml` has no `transformers`.
- ADR-0009 (round 0, awaiting its grant) is in this tree, its amendments in SPEC-model §1–§3, MODEL-I4, I6, I9, I10,
  task 4, §6. D7 refuses names needing a v5 conversion entry until a byte map passes the round trip; D8 is this ADR's
  seam; D11 deletes `hf.load_into`; decision 5: this ADR's native loader is not built, the pin 5.19.0 is a product one.
- transformers 5.19.0 (uv cache, read 2026-10-10): `MixtralSparseMoeBlock.forward` returns one tensor; `gate` is a
  `MixtralTopKRouter` holding a raw `weight` Parameter; `MixtralExperts.gate_up_proj` (E, 2f, h), gate rows first (HF
  runs one 2f-wide GEMM, then `chunk(2, dim=-1)`), `down_proj` (E, h, f); `Qwen2Moe*`, `Qwen3Moe*` the same.
  `conversion_mapping.py` merges hub per-expert `w1`/`w3` (`mixtral`) and `gate_proj`/`up_proj` (`qwen2_moe`, aliased
  by `qwen3_moe`, `olmoe`, `deepseek_v3`) into `gate_up_proj` by `MergeModulelist(dim=0)`, `Concatenate(dim=1)`.

## 19. Ideas considered in round 3 (moved from the ADR in round 4, verdicts unchanged)

| Idea (source; verified?) | Verdict | Reason and number |
|---|---|---|
| Register `DeepseekV3MoE` (Kimi K2 is `deepseek_v3`), `OlmoeSparseMoeBlock`, `GptOssMLP`, `Llama4TextMoe` in the HF table (classes confirmed at 5.19.0) | FOLLOW-UP | One row and one E2-HF arm each, but each class needs reading first: GPT-OSS's interleaved gate/up and clamp, `Llama4Router(nn.Linear)` with input weighting, DeepSeek's fp32 `e_score_correction_bias` buffer as D10's state. Until then 4 of 7 rows are from-scratch only (decision 7) |
| Kimi K2's `aux_loss_alpha` 0.001, `seq_aux` true as recipe defaults (config fetched) | FOLLOW-UP with the row above | Fidelity only; no M4 Kimi path. Values recorded as FACT in derivation §3 |
| Reference values checked (Mixtral, Qwen3, DeepSeek-V3 61 layers with `first_k_dense_replace` 3, gpt-oss-20b, OLMoE, Kimi K2) | CORRECTION (labels) | All matched; derivation §6's 118,784 B (58 MoE layers) is now FACT |
| GPT-OSS score from HF code, not secondary sources | FOLLOW-UP with row 1 | FACT at 5.19.0: softmax over the top-k in the logits' dtype (`modeling_gpt_oss.py:128`); "sigmoid" sources left out |
| Score dtypes per row (GPT-OSS bf16 softmax; Llama 4 fp32 sigmoid cast back, input multiply in the compute dtype) | FOLLOW-UP with row 1 | FACT at 5.19.0 (`modeling_llama4.py:150`); fidelity of rows refused on the HF path in M4 |
| Defer the v5 bank's dW to W (a per-(mb, projection) record, or `_SplitLinear` over the `unbind` views) | FOLLOW-UP | Returns 7.22 / 0.71 / 0.77 ms per layer per microbatch to W and halves expert gradient bytes (S1 ×1.97 → ×1); a new `wgrad` entry kind, so not in a frozen round |
| ADR-0009's MEASURED native-load deltas for R11 | CORRECTION (label) | R11 FACT under 6b, moot under 6a |
| DeepSeek-V4 (arXiv:2606.19348, fetched) as prior art | Placed (related work) | Token order within a rank (D5/D6), a deterministic sum instead of atomics (D0), no split-k; its Sqrt(Softplus) score and hash routing stay refused rows |
| Megatron-LM `moe_utils.py` (fetched) in place of round 2's unverified entry | Placed (related work) | `get_updated_expert_bias`: `sign(total − c·E)`, D10's form; `MoEAuxLossAutoScaler`: D8's injection; its count all-reduce over TP+DP+CP is topology-scoped, D10's is recipe-scoped; E4's float-mean control stays a must-fail, not Megatron's behaviour |
| transformers v5 contract (one tensor, raw router weight, gate rows first; DeepSeek's bias an fp32 buffer, no upstream update) | CORRECTION (FACT, read locally at 5.19.0) | Derivation §8's v5 row and §14's contract are FACT |
| One bidirectional byte map per (model_type, tensor), hub per-expert ↔ v5 fused | FOLLOW-UP, decision 10 (ii) | Closes R13 with one load-time copy; whether it lives here or in ADR-0009's D7 is the human's call |

Verified and placed: transformers 5.19.0 modeling files of the six families and `conversion_mapping.py` (uv cache);
the DeepSeek-V3, Kimi-K2-Instruct, gpt-oss-20b configs; DeepSeek-V4; Megatron-LM `moe_utils.py`. Not verified, left
out: NVIDIA NIM and SemiAnalysis pages on GPT-OSS's router; Llama 4 Maverick's layer alternation (not re-fetched).

## 20. Round 4: H2 per path and the HF v5 launch count

FACT (transformers 5.19.0, `modeling_mixtral.py:55-92`, uv cache, read 2026-10-11): `MixtralExperts` is decorated by
`@use_experts_implementation` (`integrations/moe.py:542`, which replaces the class's `forward` by a dispatcher on
`config._experts_implementation`, falling back to the original); the original runs, per hit expert, one
`F.linear(x, gate_up_proj[j])` (2f wide) then `chunk(2, dim=-1)`, and one `F.linear(a, down_proj[j])`, and combines by
`index_add_` into zeros in the hidden dtype (each product `Y·w` taken in fp32, cast, then added: k roundings).

GEMM calls per MoE layer per microbatch, every expert hit (realistic T; HF skips unhit experts):

| | F | F′ (recompute) | B | W | total |
|---|---|---|---|---|---|
| native `MoEMLP` (deferred W) | 3E | 3E | 3E (dX) | 3E (dW) | 12E |
| HF v5, replaced forward (banks' dW in B) | 3E | 3E | 6E | 0 | 12E |
| HF v5, transformers 5.19.0's own block | 2E | 2E | 4E | 0 | 8E |

Excess of the replaced forward over HF's own block: +3E without recompute, +4E with. At 5 µs per launch (HYPOTHESIS,
the ADR's H100 figure): Mixtral (E 8) 0.12 / 0.16 ms, Qwen1.5-MoE (E 60) 0.90 / 1.20 ms, Qwen3-30B-A3B (E 128) 1.92 /
2.56 ms, against 32.3 / 7.05 / 4.66 ms of GEMM time per layer per microbatch. The FLOPs are equal (two f-wide GEMMs
do the work of one 2f-wide GEMM). The split exists because F8 requires the HF block's bits to equal `MoEMLP`'s, whose
gate and up are separate `SplitLinear`s; one 2f-wide GEMM is another shape, whose bits need not equal two f-wide ones.
Removing it means one fused gate_up GEMM on both paths (a native `MoEMLP` change: an idea recorded in the ADR,
FOLLOW-UP). E7b measures the CPU excess against 3E times a 0-row `F.linear` call; E7a keeps the v4-style baseline
(`probe3.py`) for the native path.

HF's tuned block (ADR A1t). FACT (P9's every-expert arm): calling every expert with `topk`, the `.sum`
renormalization and a hidden-dtype combine is exact between oracle and stage on one CPU map (40/40). Its remaining
departures from this layer: `topk` (banned by the product rule's kernel clause: CPU vs MPS tie sets differ, P10;
`index_add_` cannot run on MPS under deterministic algorithms, P11, and a gathered add of unique indices gives its
bits on CPU, DERIVED: one add per element per expert), the hidden-dtype combine (D7's fp32 combine is a precision choice) and the engine's accumulation order
(D1's fold is a written order guarding a CUDA HYPOTHESIS). The last two are recipe choices, not identity needs per
device map.

lbl scale (FACT, transformers 5.19.0 `load_balancing_loss_func`, `modeling_mixtral.py:539-540`): `tokens_per_expert`
divides counts by rows (tokens), not by `T·k`, and the result is `Σ(f_i·P_i)·E` with no 1/k: uniform routing gives k.
The recipe's per-layer form gives 1, so matching HF at L_m = 1 needs ×k, besides §5's L_m/k pooling factor (fidelity of
from-scratch defaults only, R4).

## 21. Ideas considered in round 4 (moved from the ADR in round 5, verdicts unchanged)

Two cells corrected in round 5 and marked so: the HF-fidelity row's reason (A1t restated with the stable sort) and the
Llama 4 reference (the withdrawn arXiv:2601.11659 removed).

| Idea (source; verified?) | Verdict | Reason and number |
|---|---|---|
| HF-fidelity recipe row: `topk`, `.sum` renormalization, hidden-dtype ascending combine, every expert (P9; ADR-0009 Trade-offs) | FOLLOW-UP, decision 9 | Exact per device map (P9 40/40); round 5: with D3's stable sort in place of `topk` the product rule allows it, and the reason it is not the decision is F8 and D1 (ADR A1t); a new row and an E2 arm |
| Swap the router and experts child classes too, the block calling them (ADR-0009 D2, D5, D8) | Open as decision 11 (i) | Fixes R14 (b) and the `*Experts` part of (c) at no step cost; a seam change, so not in a frozen round |
| Class swap to a per-row subclass (`wgrad.py:139-140` precedent; SPEC-model MODEL-I10 fingerprint) | CORRECTION (text) | States how the frozen "forward replaced" is done; fixes R14 (a) at zero cost |
| Narrow the v5 banks' fp32 gradients to bf16 (each element the exact widening of one bf16 dW under autocast) | FOLLOW-UP | DERIVED lossless with a low-16-bit check; S1 10,882 → 5,521 (Mixtral); a new wire rule, not in a frozen round |
| One 2f-wide gate_up GEMM per expert on both paths (transformers 5.19.0 `MixtralExperts`) | FOLLOW-UP | −E F and −2E B launches (Qwen3 −1.92 ms at 5 µs); changes `MoEMLP` and F8 |
| `torch.nn.functional.grouped_mm` (public, PyTorch 2.11–2.14 docs; reviewer-verified, not re-fetched) for A4 and the deferred bank dW | FOLLOW-UP, M1 | Removes A4's private-API objection; 12(E−1) fewer launches; CUDA determinism (R1) unmeasurable at gpu false |
| `@use_experts_implementation` on `*Experts` (FACT, `integrations/moe.py:542`, local) | FOLLOW-UP (ADR-0009's fingerprint) | The class swap overrides the dispatcher; its source belongs in ADR-0009's fingerprint list |
| Mixtral's paper gate Softmax(TopK(x·W_g)) (arXiv:2401.04088 §2.1) vs HF's softmax over E, topk, renormalize | Placed (related work) | Equal in exact arithmetic, not in bits; the Mixtral row follows HF's code, the paper's form is the `softmax_k` row |
| lbl per layer is 1/k of HF's `load_balancing_loss_func` (FACT, `modeling_mixtral.py:539-540`) | CORRECTION (R4) | Matching HF at L_m = 1 needs ×k besides derivation §5's L_m/k; fidelity of defaults only (derivation §20) |

References the reviewers verified, placed (all match derivation §3; none changes the decision): Mixtral
(arXiv:2401.04088), Qwen3 (arXiv:2505.09388: 128 experts, 8 active, no shared, global-batch LBL), DeepSeek-V3
(arXiv:2412.19437; HF router re-fetched: D2–D4), OLMoE (arXiv:2409.02060), Kimi K2 (arXiv:2507.20534), gpt-oss-120b
config (128 experts, top-4, `swiglu_limit` 7.0), Llama 4 (Scout 16, Maverick 128 + 1 shared, top-1; round 5: the
dense/MoE alternation from transformers 5.19.0 `configuration_llama4.py:85,161,186`, `interleave_moe_layer_step`
default 1, not from the withdrawn arXiv:2601.11659), transformers 5.19.0 (PyPI). DeepSeek-V4 (arXiv:2606.19348, fetched): Sqrt(Softplus)
and hash routing §2.1; token order per rank, per-SM buffers for atomics, no split-k §3.3. Not verified, left out:
Maverick's `interleave_moe_layer_step` value (gated config).

## 22. Round 5: the HF v5 bank-gradient hold

FACT (SPEC-runtime §2 and task 5): W(mb) assembles its result "with B's other gradients and stores [it] for the fold";
on the HF v5 path each 3-D bank's gradient is produced in B in the master dtype and ships (pieces) or is assembled
(width 1) when W(mb) starts. So from the end of B(mb) to the start of W(mb) every MoE layer of the stage holds
`b_m·E·3hf` bytes per microbatch, the whole bank (not `1/w` of it) also on a wide stage.

```python
for n,E,h,f,nat in [('Mixtral',8,4096,14336,800),('Qwen1.5',60,2048,1408,260),('Qwen3',128,2048,768,400)]:
    b=E*3*h*f*4; print(n, b, b/2**20, round(b/2**20/nat,2))
print(4*8*3*4096*14336*4/1e9)   # Mixtral stage of 4 MoE layers (S 8), one outstanding microbatch
```

Output: Mixtral 5,637,144,576 B = 5,376 MiB (×6.72 the native pending W); Qwen1.5-MoE 2,076,180,480 B = 1,980 MiB
(×7.62); Qwen3-30B-A3B 2,415,919,104 B = 2,304 MiB (×5.76); a Mixtral stage of 4 MoE layers 22.5 GB per
outstanding microbatch, outside the ADR's device-state column (34.8 GB at S 8, w 4). Under a zero-bubble order with
`m` microbatches between B and W the hold is `m·22.5 GB`; at `m` 2, 34.8 + 45.1 = 79.9 GB before activations and the
reserve, beyond an 80 GB device.

Accounting (round 5 amendments, text only): SPEC-model §2 states the hold; SCHEDULER-I21 adds it to
`stage_memory_model`'s `mem_b` and releases it in `mem_w` (`mem_w = −(mem_f + mem_b)` keeps the sum 0), so the ZB-2p
solve sees it per outstanding microbatch, and adds `m_BW,k·b_m·P_bank,s` to SCHEDULER-I16's device-memory condition,
with `m_BW,k` the most of peer k's microbatches between B's end and W's start in the emitted order (`M_k` before an
order is solved: conservative). F3 and E2-HF measure it exactly. What would remove it: folding each bank gradient at
B's end (width 1), or the deferred bank dW (both FOLLOW-UPs in the ADR).


## 23. Round 7 (correction pass, 2026-10-11): hold × in-flight, the HF-MoE seam rule, A1t's strongest form

No new mechanism beyond what the ADR's decisions 10 and 11 recommended; under the autonomy switch
(`agents-architetura/tools/factory/autonomy.json`, `humanGrants` false) each takes its recommended option.

### 23.1 The bank-gradient hold under a zero-bubble order (finding a)

DERIVED (§17, §22 numbers): per stage s and peer k, `H_s,k = m_BW,k·b_m·Σ_{MoE layers of s} E·3hf`. In the ZB-2p
solve (SPEC-scheduler §2.1) `mem_b` adds `b_m·Σ E·3hf` and `mem_w` subtracts it, so the solver's memory replay counts
it once per microbatch between B and W, i.e. `m_BW` times, in both memory branches (with recompute, beside the
pending W; without, beside the share B keeps). SCHEDULER-I16's device-memory condition adds `m_BW,k·b_m·P_bank,s`
(`M_k` before an order exists: conservative). Mixtral, 4 MoE layers per stage (S 8): 22.5 / 45.1 / 90.2 GB at
`m_BW` 1 / 2 / 4, against the device-state column's 34.8 GB at w 4. F3 and E2-HF measure it per microbatch.

### 23.2 The HF-MoE seam rule (findings b; decisions 10 (i) and 11)

The rule (written identically in ADR-0008 §HF boundary and ADR-0009 D8):
(1) `SUBSTITUTE[(class name, transformers version)]` rows at 5.19.0: blocks `MixtralSparseMoeBlock`,
`Qwen2MoeSparseMoeBlock`, `Qwen3MoeSparseMoeBlock`; routers `MixtralTopKRouter`, `Qwen2MoeTopKRouter`,
`Qwen3MoeTopKRouter`; experts `MixtralExperts`, `Qwen2MoeExperts`, `Qwen3MoeExperts` (FACT: `modeling_mixtral.py:56,95,113`,
`modeling_qwen2_moe.py:271,310,331`, `modeling_qwen3_moe.py:209,248,269`, read locally). Each row is a class swap to a
registered subclass of this repository. (2) The swapped block calls `self.gate(x̂)` once (the router GEMM, D2) and
`self.experts(x̂, slots)` once (D5 dispatch and the per-expert `unbind`/`chunk`/f-wide `F.linear` of §14), as module
calls, plus the shared expert and its gate as HF's block already does (`modeling_qwen2_moe.py:342-346`). The GEMMs, their
shapes and their order are those F8 checks; only the Python call boundary moves. (3) ADR-0009's classifier finds the
block by routing structure and looks up the block's class by exact name; a router or experts row is admitted only
under a block with a row. (4) The launch gate's reference and the `arch="hf"` oracle are the vanilla `ForCausalLM`
with the same swap and the same D8 aux injection. (5) The substituted block is proved by F8 (MODEL-I8) against
`MoEMLP`, at the pin and in each census.
What (4) gives up, stated: the gate no longer compares the swapped block with HF's own block. It never could pass:
the block differs by design (decision 9), and with `moe_lbl_coef` or `moe_z_coef` > 0 the aux gradients exist only
on the swapped side. Cost: two `nn.Module.__call__` per MoE layer per F and F′ (µs of host, HYPOTHESIS; < 0.2% of
Qwen3's 4.66 ms GEMM time per layer), 0 bytes, 0 waits; the gate's 1–2 s per device unchanged.

Hub-to-fused map (decision 10 (i), owned by ADR-0009 D7). FACT (`conversion_mapping.py:1068-1108` and the alias table
`:44-68`, 5.19.0): `mixtral` renames `.block_sparse_moe.` → `.mlp.` and builds `experts.gate_up_proj` by
`MergeModulelist(dim=0)` over `experts.*.w1.weight`, `experts.*.w3.weight` then `Concatenate(dim=1)`, and
`experts.down_proj` from `experts.*.w2.weight`; `qwen2_moe` (aliased by `qwen3_moe`, `deepseek_v3` and 20 others)
does the same from `mlp.experts.*.{gate_proj,up_proj,down_proj}.weight`. As a byte map, per layer and expert j:
`gate_up_proj[j, 0:f, :] ← gate_j` (f, h), `gate_up_proj[j, f:2f, :] ← up_j` (f, h), `down_proj[j] ← down_j` (h, f),
each a strided element copy, every source and destination element written exactly once (counted). Export inverts it.
Round-trip test: a hub-layout safetensors file of distinct byte patterns (−0 and NaN payloads included) loads, exports
and re-loads with every tensor's sha256 equal, and the load equals transformers 5.19.0's own conversion bit for bit
(test-only call). DeepSeek-V3's entry exists for the test; no run uses it until a SUBSTITUTE row exists.

### 23.3 A1t in its strongest form (finding c)

A1t-strong: transformers 5.19.0's own block (`modeling_mixtral.py:68-127`) with only `torch.topk` replaced by D3's
stable descending sort. What still differs from the chosen class swap, per source:

| Source | A1t-strong | Class swap | Exactness per device map | Bits vs `MoEMLP` |
|---|---|---|---|---|
| combine | `index_add_` into `zeros_like` per hit expert | gathered fp32 left fold (D7) | `index_add_` raises on MPS under deterministic algorithms (P11); CUDA: R1 HYPOTHESIS | fp32 recipe: order equal, the `+0` seed loses a −0 sign; bf16 hidden: k roundings |
| renormalization | `.sum(dim=-1)` over k, one kernel | written left fold (D4) | per map yes (D0) | equal at k ≤ 2 (Mixtral); may differ at k ≥ 3 (Qwen1.5 k 4, Qwen3 k 8; P5) |
| x̂'s gradient | engine sums router, per-expert gathers, shared expert | D1 fold | CPU structural (P6); CUDA HYPOTHESIS | may differ |
| gate/up | one 2f-wide GEMM, `chunk(2, dim=-1)` | two f-wide GEMMs | per map yes | differ (another shape, §20) |
| aux, D10 lane | none (pooled `router_aux_loss_coef`) | D8 injection, count lane | — | A1t-strong needs both added |
| unrouted experts | skipped; the bank's gradient is still dense (+0 rows from the indexing backward, DERIVED) | called on 0 rows | both dense | +0 either way |
| launches per MoE layer per mb | 8E (F 2E, F′ 2E, B 4E) | 12E | — | −3E / −4E: 0.12/0.16, 0.90/1.20, 1.92/2.56 ms at 5 µs (HYPOTHESIS) |

Launch share of GEMM time per layer (DERIVED): Mixtral 0.4% / 0.5%, Qwen1.5-MoE 12.8% / 17.0%, Qwen3-30B-A3B 41.2% /
54.9% (without / with recompute). Verdict: the class swap still wins on exactness, not on time. A1t-strong cannot run
on MPS without a second replacement (the combine), it needs the aux injection and count lane added anyway, and it
leaves three orders (renormalization at k ≥ 3, x̂'s gradient, the combine on CUDA) to the kernel or engine where the
skeleton writes them; it is also another function than the native layer (F8). It wins on time: for fine-grained MoE
on GPU its 3E–4E fewer launches are 41–55% of Qwen3's GEMM time (HYPOTHESIS until M1). That gap is not intrinsic to
the swap: one 2f-wide gate_up GEMM in `MoEMLP` and the swap alike removes it with F8 intact (FOLLOW-UP, §21). With
the seam rule an HF-fidelity SUBSTITUTE row (A1t-strong plus a gathered combine) needs no ADR-0009 change (FOLLOW-UP).

### 23.4 Ideas considered in rounds 5–6 (moved verbatim from the ADR in round 7)

Verdicts unchanged except one: the round-5 row "Gate reference = vanilla model with the same class swap and D8
injection" (OPEN, decision 11 (d)) is ADOPTED in round 7 as the seam rule (§23.2), decided by autonomy, 2026-10-11.

Rounds 0–2: derivation §16; round 3: §19; round 4: §21 (moved in round 5, verdicts unchanged). Round 5, mechanism frozen, no idea adopted:

| Idea (source; verified?) | Verdict | Reason and number |
|---|---|---|
| Gate reference = vanilla model with the same class swap and D8 injection; F8 proves the block (ADR-0009 D9, MODEL-I9/I10 vs decision 9) | OPEN, decision 11 (d) | Removes R14 (d), the refusal of every HF MoE run; ≈ 1–2 s per launch (ADR-0009's gate figure); a change to ADR-0009's gate, so not adopted here |
| HF-fidelity row: stable sort, HF's 2f-wide gate_up GEMM and hidden-dtype combine (derivation §20) | FOLLOW-UP (A1t restated) | Allowed by the product rule; −3E/−4E launches (Qwen3 −1.92/−2.56 ms at 5 µs, HYPOTHESIS); needs a fused `MoEMLP` for F8; passes ADR-0009's vanilla gate only at aux 0 and where `topk` and the sort agree on ties (P4: 4.25% boundary ties, bf16 E 64) |
| Fold each bank's fp32 dW at B's end instead of holding it to W(mb) (two lenses; SPEC-runtime §2 fold, I17) | FOLLOW-UP | Bounds the hold to one fp32 accumulator per layer (−5,376 MiB per Mixtral layer per extra outstanding mb); width 1 a release-time change, pieces need I17's release at B(mb): a scheduler change |
| Batch-invariant expert GEMMs (no split-k) as a recipe kernel class (Thinking Machines 2025; DeepSeek-V4 §3.3, reviewer-fetched) | FOLLOW-UP, EP ADR and M1 (R5) | Would make P3 vanish and reopen A4/A5 as exact; cost ≈ 1.6× (reviewer's vLLM figure, not re-fetched); unmeasurable at gpu false |
| Routing replay in F′: store T·k indices and weights from F (Rollout Routing Replay, arXiv:2510.11370: not verified) | FOLLOW-UP | Saves the router GEMM and sort in F′ (≤ 6.85% with an fp32 router, Qwen3) for T·k·4 B (128 KiB, Qwen3, T 4096); changes recompute; the tripwire already compares them |
| D10 against Loss-Free Balancing (arXiv:2408.15664, derivation §11) | Placed (related work) | `sign(T_g·k − E·c_i) = sign(mean − c_i)`: the published rule over the global batch; no change |
| DeepSeek-V4 rows: Sqrt(Softplus) score, hash routing (arXiv:2606.19348 §2.1, reviewer-fetched) | FOLLOW-UP | `sqrt_softplus` is one elementwise row at sigmoid's cost; hash routing needs token ids in the FWD frame, refused (derivation §3) |
| `group_key ∈ {top2_sum, max}` for DeepSeek-V2 (DeepSeek-V3 inference `model.py` Gate, reviewer-fetched) | FOLLOW-UP | One recipe field, a max over E/G in one kernel call (D0); adds V2/V2-Lite; not an M4 family |
| Llama 4 alternation from a primary source (finding; HF `Llama4TextConfig`) | CORRECTION (reference) | FACT, transformers 5.19.0 `configuration_llama4.py:85,161,186` (read locally): `interleave_moe_layer_step` default 1 gives `moe_layers`; Maverick's value unverified |

Round 5 verified, placed: transformers 5.19.0 `configuration_llama4.py` (local), HF main `modeling_deepseek_v3.py` (matches D2–D4), transformers 5.19.0 on PyPI, arXiv:2606.19348, Thinking Machines 2025, DeepSeek-V3 `model.py`. Removed: arXiv:2601.11659 (withdrawn). Not verified, left out: arXiv:2510.11370, GSPO, MaxText docs. Round 6, mechanism frozen, no idea adopted, no decision changed (most ideas restate earlier verdicts, cited):

| Idea (source; verified?) | Verdict | Reason and number |
|---|---|---|
| Lagged global-batch LBL from the previous step's integer counts (Qwen3 arXiv:2505.09388; Qiu et al. arXiv:2501.11873; verified); or prefix counts over microbatches 0..m (first-principles lens) | REJECTED (lagged: round 1, derivation §5) | Exact, but neither is Qwen3's function (its f_i are the current step's whole batch), so R4's gap stays; lagged adds checkpointed counts and a step-0 value; prefix makes microbatch m's aux depend on its index and, on a wide stage, makes B(m) wait for counts of microbatches other peers ran: a new critical-path wait |
| Grouped (recipe-fixed G) or global-batch LBL, bubble computed by the solver (Qiu et al.; Qwen3) | FOLLOW-UP (R4, derivation §5) | Barrier bubble up to 15.8% (S 4, M 16) and 17.9% (S 8, M 32), DERIVED bounds; a scheduler constraint, its own decision |
| Already in the decision: DeepSeek-V3 loss-free bias from integer counts (arXiv:2412.19437 §2.1.2, 2408.15664, Kimi K2 2507.20534; three lenses); the enumerated score/selection family incl. groups, `softmax_k`, sigmoid top-1 + shared; z-loss as a written fold (ST-MoE 2202.08906, OLMoE 2409.02060); weight at the expert input (HF `modeling_llama4.py`, fetched); fused banks read per expert by byte views; MuonClip's whole-matrix pieces | Placed (D10; D2–D4; D8 + D0; D4 `apply`; §HF boundary `unbind`/`chunk`, derivation §8; derivation §7) | No change. The lens's "5 of 7 families refused" reads an older draft: all seven rows exist (derivation §3), and the `logit` vs `score` key covers topk-on-logits vs topk-on-fp32-softmax tie sets |
| v5 experts hook / kernel-hub seam (transformers `integrations/moe.py:542`, local; VeOmni notes, not verified); fixed-config grouped GEMM as the recipe kernel, or one equal to the per-expert loop (MegaBlocks 2211.15841, `torch._grouped_mm`, CUTLASS); batch-invariant forward and dX GEMMs (Thinking Machines 2025) | FOLLOW-UP (A1′ round 4; A4 at M1; round 5) | The hook covers the experts only, the class swap the router and aux too; R5's 7.68 vs 4.66 ms (HYPOTHESIS) argues for A4 at M1, whose CUDA determinism (R1) is unmeasurable at gpu false (P11: equal on one CPU case); D6 already removes P3 at zero cost |
| MXFP4 experts decode losslessly (OCP MX v1.0; gpt-oss card) | FOLLOW-UP with GPT-OSS attention; CORRECTION (range) | Exact only in range: `m·2^(s−127)` (m ∈ ±{0,…,6}) fits bf16 and fp32 for every s ≤ 252; at s 253 \|m\| ≥ 4 and at s 254 \|m\| ≥ 2 reach 2^128 and overflow; s 255 is NaN. Derivation §9's "s ≥ 253 with a magnitude ≥ 6" understates the refusal set (reported for its owner) |
| Unsort by index assignment, not `scatter_add` (pytorch/torchtitan PR #1974, re-fetched 2026-10-11) | Placed (H2 support) | FACT, third party, one CUDA bf16 config: output std 0.0297 → 0, F + B 19.71 → 19.03 ms (−3.4%), peak active 5.93 → 5.05 GiB (−14.8%). D5/D7 already gather; exactness was not slower there |
| Measure R1's kernel facts and the router dtype ratio on this Mac's GPU (CLAUDE.md hardware switch) | FOLLOW-UP (E1's non-gating MPS rows) | P10/P11 already measure MPS; a 0-row MPS GEMM sign and an fp32/bf16 router ratio would be MEASURED (local Mac), never R1's CUDA fact or the H100 14.8 (no gate on the Mac's GPU) |

Human references (round 6), verified by the reviewers and placed: Mixtral (arXiv:2401.04088; its "softmax over the top-k logits" is the paper's form, the `softmax_k` row; HF's code is softmax over E then renormalize, the Mixtral row: equal in exact math, not in bits), Qwen3-MoE (128/8, no shared, `norm_topk_prob`, global-batch LBL, `decoder_sparse_step`, `mlp_only_layers`), DeepSeek-V3/R1 (1 + 256, top-8, sigmoid, bias, groups, `first_k_dense_replace`, sequence-wise aux; MLA and MTP refused by name), GPT-OSS (128/32, top-4, clamped SwiGLU, biases; sinks, banded attention, MXFP4 refused), Llama 4 (16/128, interleaved, 1 shared + sigmoid top-1, input weighting; NoPE, chunked attention refused), OLMoE (64/8, lbl + z), Kimi K2 (384/8, 1 shared, MLA, MuonClip: §7 limit): all match derivation §3 and §11. Facts checked against this text: ADR-0006's pieces are relied on as M2b (`factory/m2b`, unmerged), not re-decided; invariants are RUNTIME-I18 and SCHEDULER-I21 (no I15, I19 or I20 here); the milestone is M4; ADR-0009's FROZEN amendments are its own, and this block is stated for both paths (§HF boundary, R14, decision 11). Not verified, left out: VeOmni's v5 loading notes, vLLM's transformers MoE backend docs.
