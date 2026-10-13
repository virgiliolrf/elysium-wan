# SPEC-model: layers, precision, data and deferred weight gradients

**Workstream:** `model`. **Owned paths:** `src/wantrain/model/`, `tests/unit/test_finetune.py`.

## 1. Scope

The model workstream builds deterministic flat layer lists (`[Embed, Block × n, Head]`): the built-in GPT,
Llama-family pieces, HuggingFace checkpoints (M4, ADR-0009: transformers' own modules, windowed per unit), or a user
`build(cfg)`. It also owns:
- the declared precision recipes and the fp8 boundary scaling;
- LoRA;
- deterministic batches and corpora;
- the deferred weight-gradient machinery that makes the zero-bubble B/W split real.

## 2. Domain model (normative)

- **Flat list.** A stage is a contiguous slice `layers[a:b]`. Layers are stateless across microbatches. The boundary
  between slices is one tensor `(mb, seq, d_model)` in the recipe's boundary dtype.
- **Determinism.** `build_layers(cfg, seed)` produces identical parameters on every host whose sampler computes the
  reference's bits. A host is admitted to init only when its sampler fingerprint, a necessary condition, equals the
  init record's, and its bytes are proved by the record's per-tensor digests (MODEL-I6; ADR-0007: torch's CPU samplers
  may differ by ISA and libm, a HYPOTHESIS until its experiment E-a). `build_slice(cfg, seed, a, b, home, record)` produces exactly layers `[a, b)` of that build and leaves
  the global generator where the full build leaves it (MODEL-I6). LoRA adapters are seeded independently of what was built before them. `make_batch(data_seed, step, mb, ...)` depends only on its arguments and
  the declared dataset shard mapping.
- **Deferred dW** (`wgrad.py`, the reference implementation's WeightGradStore):
  - `convert(module)` replaces every `nn.Linear` whose weight is not shared or tied by a `SplitLinear` with the same
    parameters. (M4, ADR-0009) Under HF modules it takes an exclusion set from the unit trace: a Linear whose weight or
    bias an op outside its own forward reads stays `nn.Linear`, its dW in B (a converted one would lose that use's
    gradient: mamba's `dt_proj`).
  - Inside `with capture() as store:`, `SplitLinear.backward` returns only `dX = dY·W`. It appends
    `(module, deferred (X, dY))` to the store; both `dW` and `db` are deferred.
  - `flush(store)` returns `{id(param): grad}` for every weight and bias. It computes `dW = dYᵀX` and
    `db = Σ dY` over the flattened batch dims.
  - Outside `capture()`, `SplitLinear` behaves exactly like `nn.Linear`.
  - The store is **process-wide** (lock-protected), because autograd runs CUDA/MPS backward on device threads.
  - Autocast casts happen outside the Function, so the captured tensors are exactly what stock autograd would use.
- **Pending-W memory.** One pending microbatch holds `16·h` values per token per GPT block (qkv 1+3, proj 1+1, fc1 1+4,
  fc2 4+1). The scheduler's memory model uses it (`stage_memory_model`). (M4, ADR-0008) An MoE block holds its
  attention's values plus `k·(2h + 3f)` per token for the routed experts (X shared by gate and up; dY of gate, up and
  down; X of down) plus `2h + 3f_sh` for a shared expert, whatever the routing: the routed rows are `T·k` exactly.
  Under the HF-object forward (§3) the routed experts' dW runs in B, so their pending W is 0; instead B leaves each
  3-D bank's dense gradient in the master dtype, `b_m·E·3hf` bytes per MoE layer (`b_m` 4 under today's recipe), live
  from B(mb)'s end until W(mb) starts; a peer holds it once per microbatch between B and W in the emitted order
  (`m_BW`), and the scheduler charges `m_BW·b_m·Σ E·3hf` (SCHEDULER-I21).
- **MoE block** (M4, ADR-0008; normative). `MoEMLP` replaces a block's MLP where the config marks the layer MoE.
  Fields: `moe_experts` E (0 = dense), `moe_top_k` k (1 ≤ k ≤ E), `moe_d_ff` f, `moe_shared_d_ff` (0 = none),
  `moe_shared_gate` (`"sigmoid"` or `"none"`), `moe_layers` (None = every block), `moe_router_dtype` (`"compute"`,
  the default, or `"fp32"`; the HF-object table, or under ADR-0008 decision 6b the loader, sets each family's own, ADR-0008 derivation §3), `moe_router_bias`, `moe_score` (`"softmax"` over E, `"softmax_k"` over the k selected logits,
  `"sigmoid"`), `moe_select` (`"score"`, `"logit"`, `"score_bias"`), `moe_groups` G and `moe_topk_groups` (1 = none; with G > 1
  the config is refused unless G divides E, E/G ≥ 2, 1 ≤ `moe_topk_groups` ≤ G and k ≤ `moe_topk_groups`·E/G),
  `moe_norm` with `moe_norm_eps`, `moe_scale`, `moe_apply` (`"output"`, or `"input"` with k = 1), `moe_expert`
  (`"swiglu"`, or `"swiglu_clamped"` with `moe_limit`, `moe_alpha` and expert biases), `moe_lbl_coef` with
  `moe_lbl_scope` (`"layer"` or `"sequence"`), `moe_z_coef`, `moe_bias_rate` γ (0 = no selection bias update).
  Attributes in construction order: `router` (an `nn.Linear(h, E)` subclass: initialized as a Linear, never converted
  to `SplitLinear`, fp32 master), `experts` (E SwiGLU MLPs of `SplitLinear`s), `shared_expert`, `shared_expert_gate`,
  and the buffer-like state `sel_bias` (fp32, E; trained by the update rule, never by a gradient). Forward on `x̂`
  (b, t, h), flattened row-major to T tokens (token index = row):
  - reductions: one kernel call on a fixed shape keeps the kernel's order (as every GEMM does); every sum assembled
    outside one kernel call has the written order below; no scatter-add, `index_add_` or atomics; a scatter is a byte
    move with unique targets; nothing draws from any generator;
  - one fan-out function hands each tensor with ≥ 3 autograd consumers (`x̂`; the router probabilities when several
    terms read them) to its consumers; its backward is the left fold of their gradients in a written order;
  - router: under `"compute"` the GEMM runs under the recipe's autocast (compute-dtype logits, as HF's gate); under
    `"fp32"` autocast is off and the operands are fp32; the score per `moe_score` is taken in fp32 from the logits;
  - selection: the key per `moe_select` (fp32); with groups, the group key is the sum of the group's top-2 keys and the
    `moe_topk_groups` groups are the first of `torch.sort(group keys, descending=True, stable=True)`, the other groups'
    experts excluded; then the first k indices of `torch.sort(key, descending=True, stable=True)`, re-sorted
    ascending; a NaN key is fatal; no `topk`;
  - weights: the score at the selected experts (never the bias); with `moe_norm`, divided by (their left-fold sum in
    ascending expert index) + `moe_norm_eps`; times `moe_scale`; fp32;
  - dispatch: one gather in (expert, token) ascending order; backward gathers a token's k slot gradients and folds
    them in ascending expert index;
  - experts: expert e is called exactly once per call of the block, on exactly its routed rows in ascending token
    index (scaled by its weight first when `moe_apply = "input"`), also on 0 rows; rows are never concatenated across
    calls, padded, capped or reordered;
  - combine: the left fold of `w·Y` in fp32 in ascending expert index (the expert's row when `moe_apply = "input"`),
    then plus the shared expert's output (times `σ(gate_sh(x̂))` when gated) in fp32, one cast to the residual dtype;
    backward: a gather through the inverse slot permutation and `dw = Σ_h g·Y` in fp32;
  - aux terms: integer counts `c_i`; `S_i = Σ_t q_ti` (q: softmax over E, or `s/Σ_j s_j` for sigmoid scores);
    lbl `(E/(T²·k))·Σ_i c_i·S_i` per layer, or the same per sequence and averaged over the b sequences; z
    `(1/T)·Σ_t logsumexp(logits_t)²`; an identity function on the block output returns each term's coefficient as its
    gradient; the block exposes, per call, every term's value and the counts;
  - selection-bias update (γ > 0), run by the bias's owner after AdamW: with `c_i` the int64 counts of the whole
    global batch, `sel_bias_i += γ·sign(T_g·k − E·c_i)` (sign in integers, 0 on equality). Each call's count vector
    (int64, E; its sum is `T·k`) is the bias's per-microbatch input; SPEC-runtime RUNTIME-I18 carries it to the owner.
- **Precision.** A recipe declares `compute_dtype` and `boundary_dtype`. A boundary is lossless if the compute is
  declared at that precision (standard mixed precision) or the wire codec is exact. Narrowing fp32 math at the socket is
  never lossless.
- **HF modules** (M4, ADR-0009; normative). Every `arch="hf"` model is the pinned transformers' module tree.
  - Environment: the worker starts each stage and buffer-builder process with `-X pycache_prefix=` a fresh empty
    per-launch directory on its command line (in force at interpreter start, before any import; `sys.pycache_prefix`
    is never assigned in-process), `USE_HUB_KERNELS=0`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
    `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, and no `LD_PRELOAD`, `LD_LIBRARY_PATH` or `DYLD_*`. The guard checks, never
    sets, so it may run after torch's import (`hf.py:17`): it asserts each, and that every loaded torch, transformers
    and `wantrain` module's `__cached__` lies under the prefix (a process started without the flag fails both). Refused, naming the cause: a name probed by any
    `is_*_available`/`_is_package_available` call of the pinned transformers (`hfm.PROBED`, built at the pin) that is
    importable (`find_spec` or the probe) unless the `hf` extra locks it (then RECORD-fingerprinted, MODEL-I10), so
    `mambapy`; `unsloth`; a config with `auto_map`; a
    `model_type` outside the pinned `MODEL_FOR_CAUSAL_LM_MAPPING`; `is_encoder_decoder`; any dropout probability > 0
    unless the recipe sets it to 0 (recorded); a `rope_parameters` entry (per layer type) whose `rope_type` contains
    `"dynamic"`, or is `"longrope"` with `original_max_position_embeddings` below the recipe's S; a module whose `forward` is not
    its class's own; a recipe whose `boundary_dtype` is not HF's residual dtype under it; `lora_rank > 0`.
  - Skeleton: the whole `ForCausalLM` built from the config with every parameter on meta and buffers on the CPU;
    building it draws nothing (asserted by state equality); `config._attn_implementation = "eager"` is asserted. Every
    buffer is then overwritten from the run's buffer record (RUNTIME-I19). Sparse-MoE blocks are found by routing
    structure on one CPU forward of the gate's tiny model before any swap: with E the config's `num_local_experts`,
    `num_experts`, `n_routed_experts` or `moe_num_experts` (none: no MoE), a `topk`, `sort`, `argsort`, `argmax`,
    `max` or `multinomial` over a dimension of size E tags its indices (and every tensor computed from them); a module
    is sparse-MoE when a tagged value meets a parameter leading with E in one op (an index, a gather, a grouped GEMM,
    or routing scores weighting every expert's rows: llama4_text's dense dispatch) or chooses which members of a
    `ModuleList` of E run; the block is the innermost such module. Its class is looked up by exact name in ADR-0008's (HF class,
    transformers version) table (§3); no row refuses the model. `wgrad.convert` runs on the skeleton; `Conv1D` and
    3-D expert tensors keep dW in B.
  - Unit trace (on the gate's tiny model after its SUBSTITUTE swap, D8 having classified it unswapped; one forward with pre-hooks and a function-mode spy over the module-call
    stack, on ids drawn independently of the gate's microbatches at a batch size equal to no config dimension): the
    layer list is the unique `ModuleList` of length `num_hidden_layers` whose elements are called in order, equal to
    the class's `base_model_pp_plan` layer path where it declares one; each parameterized module is prologue (called
    before layer 0), layer i, or epilogue (called after layer n−1). It records each layer's output structure, each
    prologue module's output dtype and trailing size, and the Linears whose parameters an op outside their own forward
    reads (`convert`'s exclusion set). Refused, naming the module: a parameter in no unit; one Parameter in two units
    other than the (prologue embedding, epilogue head) tie; a parameterized module called between layers or from two
    layers; a parameterized module of a layer its forward never calls (`TRACE-UNPLACED`) unless an op elsewhere reads its
    parameters and its class is not a SUBSTITUTE class; a layer called twice; a non-floating tensor in a layer's output; a forward that changes any generator's
    state. Aliases within a unit are re-tied from the skeleton's sharing map.
  - Units: the flat list is `[prologue, layer 0, …, layer n−1, epilogue]` (n+2 unit modules over one shared tree; each
    registers only its own modules). Unit u's entry is layer u−1 (1 ≤ u ≤ n) or the first epilogue module (u = n+1).
    Calling unit u runs `ForCausalLM.forward(input_ids=ids, use_cache=False)` on the unit's device with: for u ≥ 1, a pre-hook on its entry
    that replaces the residual argument (positional 0 or `hidden_states`) by the unit's input; for u ≤ n, a pre-hook on
    unit u+1's entry that stops the call and returns its residual argument; every layer before u replaced for the
    call by a pass stub returning stride-0 NaN views (`torch.full((), nan).expand(shape)`) in the layer's traced
    output structure (the residual's shape; other tensors at their traced shape with the call's batch and length;
    non-tensors as traced); on u ≠ 0 every prologue module replaced by a stub returning a fresh NaN tensor in its traced dtype and
    trailing size, shaped from its first argument. A stub never returns a recorded value. Unit n+1 returns the logits;
    `loss_fn(logits.float(), targets)`. Every unit takes the microbatch's `input_ids` from `make_batch`; only the
    residual crosses a cut; a stub read through a NaN-absorbing op is not caught (ADR-0009 R13, open).
  - Loading: `hfm.load` maps checkpoint names 1:1 onto the skeleton's `state_dict()` names (parameters and persistent
    buffers) for the units it is asked for, materializes them on `home`, copies buffers from the CPU skeleton, and keeps
    `checkpoint_digest`. Export writes the same names and the source's tied-key set as sharded safetensors with an
    index. `from_pretrained`, `save_pretrained`, tokenizers and weight converters are never called on the training path.
  - Pin: the `hf` extra, `transformers==5.19.0` and `safetensors>=0.4`, locked in `uv.lock` with hashes; a pin bump is a census PR (every model type of the pinned mapping
    through the gate, and its controls).

## 3. Interfaces

**Provides**
- `GPTConfig`, `make_config(preset, ...)`, `build_layers(cfg, seed) -> nn.ModuleList`, `loss_fn(logits, targets)`.
- (M4, ADR-0008) `GPTConfig.moe_*` fields; `arch.MoEMLP(cfg)`, whose call returns the block output and records every
  aux value and the counts for the caller; `moe_bias_update(layers, counts, global_tokens)`. Decision 6b's fields are not built (ADR-0009 granted; text in the history, 2026-10-11, M4).
- (M4, ADR-0008) `arch.moe_forward(cfg, view, x)`: the exact function of §2 over a parameter view, the one `MoEMLP`
  calls. Under HF modules (ADR-0009) HF's module objects, parameter names and layouts stay and the forward of each HF
  sparse-MoE block is replaced by `moe_forward` through a table keyed on (HF class, pinned transformers version) that
  names the block's router, experts, shared expert and gate and its expert layout. The replacement is a class swap
  (`module.__class__` set to a registered subclass whose own `forward` calls `moe_forward`, as `wgrad.convert` swaps
  `SplitLinear`), never an instance attribute, so the module's `forward` is its class's own. Rows at 5.19.0:
  `MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`, `Qwen3MoeSparseMoeBlock`, each returning one tensor (the hidden
  states). Per call `gate_up_proj` (E, 2f, h) and `down_proj` (E, h, f) are split by one `unbind(0)` each and expert
  j's (2f, h) view by one `chunk(2, dim=0)` into its gate rows (first) and up rows; each runs one `F.linear` on
  exactly its routed rows, the shapes and dtypes of `MoEMLP`'s gate, up and down `SplitLinear`s; never two index
  slices of one view (their backwards are summed, turning a −0 into +0). The banks' dW is computed in B; the shared
  expert's Linears are converted by `wgrad.convert`. Aux values and counts are recorded inside `moe_forward`. A class
  or version outside the table, `output_router_logits = true`, or a block attribute `moe_forward` does not read is
  refused. HF's own MoE forward never runs; the names the skeleton, the init record, the exchange units and the
  checkpoints see are HF's. HF-MoE seam rule (ADR-0008 decision 11 = ADR-0009 D8): the table also has rows for
  `MixtralTopKRouter`, `Qwen2MoeTopKRouter`, `Qwen3MoeTopKRouter`, `MixtralExperts`, `Qwen2MoeExperts`,
  `Qwen3MoeExperts`, admitted only under a block with a row; the swapped block calls `gate` once (the router GEMM)
  and `experts` once (dispatch and the per-expert GEMMs) as modules, so the unit trace sees every parameter-holding
  module; MODEL-I9/I10's reference carries the same swap and aux injection.
- (M4, ADR-0007) `init_phases(cfg)` (each layer's phases in the recipe's order and their write sets, as data); `skeleton(cfg) -> nn.ModuleList`
  on the meta device for `arch` builtin, general and hf (names, shapes, sharing, trainable flags and the `wgrad.convert`
  classes of the full model; no storage, no RNG draw); `init_record(cfg, seed) -> dict`; `sampler_fingerprint() -> str`;
  `build_slice(cfg, seed, a, b, home, record) -> nn.ModuleList` (layers `[a, b)` on device `home`);
  `lora.apply(..., first_index=0)`; `hf.load_into(..., span=None)` only if ADR-0009 is not granted.
- `wgrad.capture()` (context manager → store), `wgrad.flush(store) -> dict`, `wgrad.convert(module)`,
  `wgrad.owned_params(module) -> set`, `SplitLinear`.
- `precision.spec(name)`, `classify(compute, boundary)`, `bytes_per_elem`, `compute_runnable(name, cap)`.
- `fp8.quantize/dequantize/roundtrip`.
- `lora.apply(layers, rank, alpha, targets, seed)`, `lora.split_params`.
- `hf.checkpoint_digest`; only if ADR-0009 is not granted: `hf.config_from_hf`, `hf.load_into(layers, cfg, path,
  strict=True) -> LoadReport`.
- (M4, ADR-0009) `hfm.guard_environment()`; `hfm.skeleton(hf_config, dtype) -> PreTrainedModel`;
  `hfm.trace(model) -> UnitMap` (layer list path, prologue and epilogue modules, entries, output structures and
  dtypes); `hfm.units(model, unit_map) -> nn.ModuleList` (n+2 units); `hfm.load(model, path, units=None) -> LoadReport`;
  `hfm.export(model, path, source_keys)`; `hfm.fingerprint(model) -> dict`; `hfm.gate(hf_config, recipe, device) ->
  GateReport`. `build_layers` and `build_slice` for `arch="hf"` return `hfm.units`. Removed: `hf.load_into` and
  `_LLAMA_*`; ADR-0008's `moe_forward` attaches through its HF-object table (ADR-0009 decision 5).
- `make_batch(...)`, `dataset.prepare/shard_view/check_compatible`.
- `provision.needed_shards(spec, rank, peer)`, `provision.provision(dataset, shards, n_shards, pinned, source)`,
  `provision.provision_slot(spec, rank, peer)`, `provision.manifest_digest(dataset)`, `dataset.shard_digests(...)`.

**Consumes:** nothing from other workstreams.

## 4. Invariants

- **MODEL-I1.** Deferred dW is bit-identical to stock autograd (fp32 and bf16), and `SplitLinear` matches `nn.Linear`
  numerically.
- **MODEL-I2.** Backward running on an autograd device thread sees the store, so no dW is dropped.
- **MODEL-I3.** Tied or shared weights are never split; autograd keeps them.
- **MODEL-I4.** A pretrained checkpoint lands completely, or loading is refused. Missing tensors, shape disagreement,
  an unsupported architecture or a sequence length past training are all refused with the reason. Nothing is left at
  random init. (M4, ADR-0008 decision 6b's allowlist paragraph is not built: decision 6a, by autonomy, 2026-10-11; its text is
  in SPEC-model-history.md, 2026-10-11, round 7.)
  (M4, ADR-0009) Under HF modules every checkpoint tensor maps 1:1 onto a skeleton name of the units loaded: a missing
  name, an unused tensor (except recomputed buffers and an explicit recipe truncation, reported), a shape or dtype-class
  disagreement, or a name that differs from the module tree (a v5 conversion entry without a byte map that passed the
  round trip) is refused, naming it; nothing is left at meta or random values. The one byte map (ADR-0008 decision
  10 (i)): hub per-expert tensors of `mixtral` (`.block_sparse_moe.` → `.mlp.`; `w1`, `w3`, `w2`) and `qwen2_moe`,
  `qwen3_moe`, `deepseek_v3` (`gate_proj`, `up_proj`, `down_proj`) load as `gate_up_proj[j, :f]`, `gate_up_proj[j,
  f:]`, `down_proj[j]`, every element written once, inverted on export; a load–export–load round trip keeps every
  sha256, and the load equals transformers 5.19.0's own conversion bit for bit.
- **MODEL-I5.** LoRA leaves the model numerically untouched at step 0, freezes the base, and trains only the adapters.
  Its init does not depend on build history, and the frozen base is not charged optimizer state.
- **MODEL-I7.** (DEFERRED-TO: M2b) `wgrad.flush_iter(store, order)` runs the deferred GEMMs one store entry at a time
  and yields `(param, grad)` for a parameter as soon as its last contribution is computed. `order` is fixed by the plan:
  `"store"` (backward unit order, today's) or `"forward"` (forward unit order). Every entry of one module runs
  consecutively, in store order, so a parameter's contributions keep their relative order.
  - Each yielded gradient is bit-identical under both orders: each entry computes `dYᵀX` from its own resident tensors.
  - A single-contribution `SplitLinear` gradient is yielded in its GEMM's output dtype; widening it to the master dtype
    equals `flush(store)[id(param)]` bit for bit.
  - A parameter with several contributions is widened and summed in fp32 exactly as `flush(store)` does.
  - At most one entry's GEMM outputs plus the not-yet-complete multi-contribution sums are resident at once.
- **MODEL-I8.** (DEFERRED-TO: M4; ADR-0008) The MoE block is one function of (microbatch, parameters) on every path,
  for every recipe row of ADR-0008 (derivation §3).
  - Its selection (and group selection) equals the lowest-index rule on crafted ties, ±0, subnormals and ±inf, a NaN
    key is fatal, and `torch.topk` is not used; a group configuration outside §2's rule is refused at construction.
  - Its dispatch, combine and fan-out backward equal their written left folds bit for bit.
  - An expert routed no token gets an all-+0 gradient (no sign bit) on the oracle path and from the deferred store;
    `flush` yields it; no trainable tensor of the block is ever absent from a gradient.
  - Under `capture()`, each expert linear records one `(X, dY)` per call with exactly its routed rows, and its deferred
    dW equals the oracle's immediate dW bit for bit (MODEL-I1), for 0, 1 and many rows.
  - The block draws nothing from any generator; recomputing it on the same input gives the same output, aux values and
    counts.
  - The selection-bias update is a function of the integer counts alone (the same result for any order of their
    summation) and moves each bias by exactly `+γ`, `−γ` or 0.
  - Under the HF-object forward (§3, ADR-0009 granted), a transformers 5.19.0 block of each class in the table returns
    one tensor inside its HF decoder layer, calls its router and experts children once each as modules, and its
    output, aux values and every gradient (sign of zero included) equal `MoEMLP`'s on byte-mapped parameters bit for
    bit, fp32 and bf16, for 0, 1 and many rows per expert: the block's own proof (ADR-0008 F8).
- **MODEL-I9.** (DEFERRED-TO: M4; ADR-0009) HF units compose to the vanilla HF model bit for bit.
  - For every config the gate admits, the units `0..n+1`, cut after any unit and split across any processes, give the
    logits, the loss hex and every parameter gradient of the vanilla `ForCausalLM` (same pinned code, same buffers,
    eager attention, no surgery but §3's SUBSTITUTE swap and its aux injection, applied alike) with `wgrad.convert` (its exclusion set applied) and deferred W under `capture()`,
    with recompute, in fp32 and in the recipe's parameter dtype, for M = 1 and M ≥ 2 under the stock fold (a tied
    table through RUNTIME-I19's row-sparse fold), on each device. The reference is SPEC-runtime §2's oracle;
    a map with none is refused (CONTROL-I24).
  - The skeleton draws nothing, its buffers equal the run's buffer record, and a unit holds no parameter outside its
    own modules (a tied table held by the Head unit only, RUNTIME-I19). A stride-0 stub written in place refuses.
  - Each refusal of §2's environment, trace and loading rules names its cause; none trains.
- **MODEL-I10.** (DEFERRED-TO: M4; ADR-0009) No `arch="hf"` run starts without passing its launch gate on every node.
  - `hfm.gate` builds a tiny config of the run's config class (dimension keys shrunk by a table; every flag, rope
    type, `layer_types` value, window, chunk, softcap, sink, NoPE list and MoE structure kept; `S` above every window
    and chunk), the vanilla model and the run's unit chain from the same parameters, and compares MODEL-I9's outputs at
    every cut, fp32 and the recipe's dtype, M 1 and 2, bitwise, on the CPU and on the slot's device when it is not the
    CPU. Its microbatches are drawn independently of the unit trace's ids. Causality probe: two id rows equal except at
    the last position give bitwise equal vanilla logits at every earlier position, or the config is refused
    (`GATE-CAUSAL`). During the probe only, each substituted expert runs one GEMM per routed row
    (`hfm._rowwise_experts`): a CPU GEMM's row bits depend on its row count (MEASURED, torch 2.14.1: 6 and 7 of 39
    counts) and a later token's routing moves each expert's count; training and the bitwise comparison never run it.
  - It records trigger counts (a window or chunk shorter than `S`, a NoPE layer, a nonzero softcap, a sink, a tie, a
    substituted MoE module) and refuses a trigger the config declares that does not fire, a build over 64 MB or 30 s,
    and any difference, naming the first differing tensor.
  - `hfm.fingerprint`: every file of the installed `transformers`, `torch` and each distribution torch requires in
    `uv.lock` (`nvidia-*` on CUDA wheels) verified against its RECORD sha256 (a mismatch, or a file outside RECORD
    other than `__pycache__/*.pyc`, refuses, naming it); `sys.pycache_prefix` is the launch's directory and every loaded
    torch, transformers or `wantrain` module's `__cached__` lies under it; transformers' RECORD sha256 and version; torch's version,
    `git_version`, platform tag and those RECORD sha256s, which must equal that platform's entry in `hfm.TORCH_RECORDS` (built by the pin-bump PR from the wheels in `uv.lock`); and every function of each
    loaded `transformers` module and class has a code object equal by value to the one compiled from its source (a monkeypatch refuses; ADR-0008's subclasses are ours);
    the probe vector of `hfm.PROBED` (name → importable, version), equal on every node; and, after the gate ran,
    every loaded native object (`/proc/self/maps` file-backed executable mappings, or the dyld image list): a file
    of a fingerprinted RECORD, tied by path and inode to the file its check hashed (a RECORD basename loaded from
    another path refuses), else an OS or driver object recorded by sha256 (a dyld shared-cache image by cache UUID
    and OS build), equal across nodes of one (platform tag, device class). gpu false: the macOS and Linux-CPU arms
    are checked; the `nvidia-*` and `libcuda` arm waits for M1. Returned with the gate report (CONTROL-I24).
- **MODEL-I6.** (DEFERRED-TO: M4; ADR-0007) A stage builds only its own slice `[a, b)`, identical to slicing the full
  build, without changing the recipe's init.
  - One init program: `init_phases(cfg)` is data. For each layer in order it lists the phases (construction;
    `gpt2_init` exactly when the build reaches `arch.build_layers`, that is `_uses_general_arch(cfg)` (or `arch="hf"`
    only if ADR-0009 is not granted),
    and `init_scheme == "gpt2"`, as today: a builtin-shaped config that declares `gpt2` has no `gpt2_init` phase; LoRA
    with the global layer index when declared: per layer, `lora.apply`'s wrapping primitive (freeze, wrap the matching
    Linears, seed from the global index) with no empty-match error, while the "matched no Linear layer" refusal and
    `WT_LORA`'s trainable and frozen counts are whole-program checks computed from `skeleton(cfg)`), each with its write set,
    the canonical names of the tensors it writes. A phase after construction writes its whole write set and reads no
    tensor. The last writer of a tensor is the last (phase, layer) whose write set names it; the tied table's is the
    Head's `gpt2_init`. `build_layers` runs the program phase-major over every layer. Today's `build_layers` code is
    kept only as a test fixture, equal to it for every preset. The program id is the sha256 of the canonical JSON of
    `init_phases(cfg)`; the code id is the sha256 of the init-path sources `src/wantrain/model/{gpt,arch,lora,hf,
    wgrad}.py`, `torch.__version__` and `torch.version.git_version` (for `arch="hf"` also `src/wantrain/model/hfm.py`
    and the sha256 of transformers' RECORD, ADR-0009); the record's binding is the sha256 of the
    canonical JSON of (cfg, including `hf_model_sha256` and the LoRA fields; seed; program id; code id).
  - `init_record(cfg, seed)` runs in a fresh process that calls `setup_determinism(seed)` before its first kernel
    dispatch, in the slot's environment. It makes one phase-major walk and holds at most one working layer.
    Construction runs the real constructor; a tie target is the skeleton tensor `to_empty`, filled with NaN. Every
    later phase runs on the layer's skeleton as it stands before that phase, `to_empty` on the CPU and filled with NaN.
    After each phase, every tensor in its write set holds no NaN and every other tensor of the working layer is still
    all NaN, else `INIT-PROGRAM` names the (phase, layer, tensor). Each tensor is hashed right after its last writer,
    and the working layer is then dropped. `get_rng_state()` is recorded before every (phase, layer) and after the last
    (the ledger). Under `arch="hf"` a checkpoint tensor's digest is that of its loaded bytes (MODEL-I4).
  - The record holds the binding, the ledger, the sha256 of every initial tensor by canonical name, the sha256 of the
    end state, `sampler_fingerprint()` and the diagnostic class (platform, libc, torch version, ATen CPU capability). It
    is JSON: every state is exactly 5056 bytes, base64-encoded, and every digest is 64 hex characters under its
    canonical name.
  - `sampler_fingerprint()` is the sha256 of `torch.backends.cpu.get_cpu_capability()` and of a fixed probe over every
    sampler path the init calls: `normal_` and `uniform_` at 15, 16, 17 and 2^16+3 elements, one `nn.Linear` and one
    `nn.Embedding` construction, each from a fixed seed, and the state after. It is valid only in a process that called
    `setup_determinism` before its first kernel dispatch. It restores the global generator's state and runs in
    milliseconds.
  - `build_slice` first refuses a record whose binding differs from that of (cfg, seed, program id, code id), with
    `INIT-RECORD`. It refuses a first ledger entry that differs from `get_rng_state()` right after
    `torch.manual_seed(seed)`, with `INIT-LEDGER`. The home tensors are those an owned layer references in
    `skeleton(cfg)`; a tied table is home on the Embed's and on the Head's stage. A writer is any layer whose write set
    names a home tensor, owned or not. For each writer in layer order, and for each of its phases:
    - `set_rng_state(ledger[phase][layer])`;
    - the phase runs in the writer's CPU working layer: the layer its constructor built, as in `build_layers`, with a
      home tensor an earlier writer wrote taken from the CPU;
    - `get_rng_state()` must equal the next ledger state, else `INIT-LEDGER` names the (phase, layer).
  - Each home tensor is finalized once, right after its last writer: hashed on the CPU, compared with the record's
    digest (else `INIT-DIGEST` names the tensor), trimmed to the slot's pieces and stored to `home`. A tied table
    stays on the CPU until the Head's last phase. Finally `set_rng_state(ledger end)`. No init call ever runs on a
    device.
  - Every owned tensor's bytes equal the full build's under the same name, and `torch.get_rng_state()` after the build
    equals the full build's. The slots of a run own every layer between them and each checks the anchor, so a launched
    run has checked every ledger transition from the seed to the end state, on every arch, hf included (below).
  - `wgrad.convert`'s classes, the trainable flags, LoRA's report and the names come from `skeleton(cfg)`, so a slice
    converts exactly the modules the full build converts (MODEL-I3).
  - Materializing a slice of `skeleton(cfg)` (`to_empty`, which gives each module referencing a shared meta Parameter
    its own new Parameter: MEASURED, torch 2.14.1) re-ties it from the skeleton's sharing map, taken before `to_empty`,
    before any load: every group of names sharing one skeleton tensor holds one Parameter, else `INIT-PROGRAM` names
    the group. This holds for the hf build, the resume (RUNTIME-I14), `PROFILE` and `FIT` (CONTROL-I19).
  - Only if ADR-0009 is not granted (it supersedes this bullet): `arch="hf"` keeps no random value: the slice goes
    `to_empty`, every owned tensor loads from the checkpoint (MODEL-I4 over `[a, b)`, strict; other tensors unread
    except a home tensor's source), the LoRA phase runs from its ledger entry. A home tensor's writers include the load
    of its source, even outside `[a, b)`: a stage holding a tied Head and not the Embed reads `model.embed_tokens.weight`
    alone into its `out.weight`, which the span load counts as required. Every home tensor is finalized against the
    record's digest. Before the load the slot runs its writers' random phases (construction, `gpt2_init`) in one reused
    scratch layer from their ledger states, checks each next state (`INIT-LEDGER`) and discards the values (ADR-0007
    R10, closed by decision 6).
  - (ADR-0009) Under HF modules the build of a checkpoint draws nothing: the program has no random phase (parameters
    on meta, then loaded; buffers from the buffer record), so the ledger is the anchor alone, each slot checks it
    (`INIT-LEDGER`), and the record digests every loaded parameter; buffers are checked against the buffer record's pin
    (RUNTIME-I19). LoRA is refused under `arch="hf"`. An HF run without a checkpoint keeps the full build per slot, as
    `arch="user"`, until HF's init program is expressed as phases.
  - `arch="user"` has no slice build, no record and no skeleton (a user module may hold non-persistent buffers or
    tensor attributes outside `state_dict()`, or not construct on the meta device): it keeps the full build, and its
    resume, `PROFILE` and `FIT` keep the full build too, then load.
  - Host memory during the build: `4·(P_layer,max + P_held)` bytes over the interpreter's baseline, where `P_held` is
    `vocab_size·d_model` on a stage that holds one end of a tied table and 0 otherwise; `home` holds the slice, so a
    CPU `home` adds `4·P_s` host bytes.

- **MODEL-I11.** A data-reading slot (first or last stage) holds, verified, every shard its owned microbatches draw
  (`mb % data_shards`) before it connects, or it refuses at startup naming the shards and the fix. Its manifest's
  sha256 equals the one the job was planned with (`dataset_sha256`, a recipe pin); each shard's bytes, from its shard
  file or that range of the whole file, equal the manifest's per-shard sha256 (the whole file's sha256 when the
  recipe's shard count differs from the prepared one). Missing or wrong bytes are fetched from `dataset_source` (a
  directory or an http(s) URL) and verified before use; a corrupt shard file is never served. A change of ownership
  (heal, rebalance) changes the set and is provisioned the same way at the slot's start (`model/provision.py`).
  The planner's default `data_shards` is the corpus's prepared shard count when it was prepared in shards (else the
  node count); it is a recipe pin, and the plan says so in its warnings. A corpus without a manifest is not pinned by
  content, and the plan warns.

## 5. Ordered task list

1. **M4 — slice-only build (MODEL-I6, ADR-0007).** `init_phases(cfg)` with `build_layers` as its phase-major walk
   (the legacy code moves to a test fixture); `skeleton(cfg)`; `init_record(cfg, seed)`; `sampler_fingerprint()`;
   `build_slice(cfg, seed, lo, hi, home, record)` by ledger replay, each home tensor finalized after its last writer;
   `lora.apply(first_index)`; `hf.load_into(span)` only if ADR-0009 is not granted;
   `wgrad.convert` driven by the skeleton's sharing map.
   - The init bits do not change: no recipe version.
   - A per-parameter generator (a recipe version) is not built; it is reopened only by the init-portability decision
     (ADR-0007 R1).

2. **M2b — streamed native-dtype dW (MODEL-I7, ADR-0006).**
   - `flush(store)` keeps today's behaviour. `flush_iter(store, order)` is the per-entry iterator of MODEL-I7; the plan
     picks `order` per stage from the simulated exposure (SCHEDULER-I17).
   - The gather-dtype table of sharded stages is derived from `owned_params` and the recipe: the compute dtype
     (`precision.spec(compute)`) for owned parameters when the recipe autocasts, the master dtype otherwise.
   - Large sharded stages also need task 1 (MODEL-I6) to bound the init peak.

3. **M4 — exact MoE (MODEL-I8, ADR-0008).** `MoEMLP` in `arch.py` (with `_uses_general_arch` true when
   `moe_experts > 0`), the `moe_*` fields and their validation (LoRA with MoE refused), the selection-bias update.
   With ADR-0009 granted (decision 6a): `moe_forward` and the HF-object table of §3 at transformers 5.19.0 (ADR-0009's
   pin), with MODEL-I8's equality test; no native HF loader. Decision 6b is not built (history, 2026-10-11, M4).

4. **M4 — any HF decoder (MODEL-I9, MODEL-I10, MODEL-I4, MODEL-I6, ADR-0009).** 4a, its own commit, kept by a
   rollback: `arch="hf"` refused at config validation; `hf.load_into` and the `_LLAMA_*` maps deleted. A rollback
   reverts by task id, each piece to its own target (ADR-0009 `round6.md` §4): scheduler 6 and runtime 7a, 7b (tied
   relay) to tied HF at S > 1 refused; `TIED` to an unassigned type; runtime 7c, 7d, 4b and control 6 to 4a; task 3's
   HF objects (6a) to MoE from scratch. 4b lifts it:
   `hfm.py` (the environment guard, skeleton, unit trace, units, `load`/`export`, fingerprint, gate with its shrink
   table); `build_layers` and `build_slice` for `arch="hf"` through it; `arch="hf"` with `lora_rank > 0` refused. Local experiments E1, E2, E5–E8 of ADR-0009 first; the census in the PR.

## 6. Decisions

- **WeightGradStore over the two-call split** (one `autograd.grad` for inputs, then one for weights). The second call
  re-walks the whole activation chain, so W costs a full backward. Rejected for that cost.
- **Slices replay the recipe's init program from a recorded ledger (ADR-0007)** over a per-parameter generator. The
  replay is exact with no recipe change: 84 slices and 168 ledger builds of four configurations, RNG state included,
  MEASURED. At 70B (S 8) both take 94 s per launch with the digest check; the record takes 811 s once per recipe
  against 101 s for a per-parameter generator's digests on 8 cores (DERIVED, ADR-0007 round 3). A per-parameter
  generator would cost a new recipe identity and a second init path, and on torch's
  samplers it is no more portable. Rejected: a per-parameter generator on torch's samplers; a declared class key as
  the init gate (neither necessary nor sufficient for equal bits); DTensor's CUDA offset RNG (its bits depend on the
  GPU's SM count). Deferred to the init-portability decision: a portable counter-based sampler.
- **Split every linear except tied weights.** Rejected: splitting tied embeddings and heads, where autograd already
  accumulates across both uses.
- **The MoE block fixes every row set, order and reduction, and its routing is an enumerated recipe (ADR-0008).**
  Each microbatch's gradient is then a dense function of that microbatch alone (39/39 pipeline and 16/16 piece arms,
  MEASURED). Rejected (ADR-0008 Alternatives): HF's block as-is or tuned (A1, A1t), `MoEMLP` in HF's place, v5's
  experts hook, a grouped GEMM (M1), a capacity factor, mapping `router_aux_loss_coef`.
- **Any HF decoder runs transformers' own modules, windowed per unit, behind an exact gate (ADR-0009).** 102 of 178
  causal-LM types are bit-identical across a cut after every unit, none differs (CPU census, MEASURED). Rejected
  (ADR-0009 Alternatives): native per family, an own loop, `inputs_embeds` re-entry, two tiers (deferred), export.
