# SPEC-model history

- 2026-10-09 — ADR-0006 (exact sharded stages; awaiting the human grant). Task 2 added (M2b). New invariant MODEL-I7
  (DEFERRED-TO: M2b): `flush(store, native=True)` returns single-contribution dW in its GEMM output dtype.
- 2026-10-09 — ADR-0006 round 1 (review findings; awaiting the human grant). MODEL-I7 rewritten from
  `flush(store, native=True)` to the per-entry iterator `flush_iter(store)`, which bounds the W transient to one entry.
  Task 2: the gather-dtype table is derived from `owned_params` and the recipe's compute dtype, never bf16 by literal.
- 2026-10-09 — ADR-0006 round 2 (review findings; awaiting the human grant). MODEL-I7: `flush_iter(store, order)`
  with a plan-fixed `order` (`"store"` or `"forward"`) replaces "in store order"; a module's entries run consecutively in
  store order, and each yielded gradient is bit-identical under both orders. Task 2: the plan picks the order.
- 2026-10-09 — ADR-0007 (slice-only build; awaiting the human grant). §2 determinism: identical on every host of one
  init class, and `build_slice` reproduces a slice and the post-init RNG state. §3: `skeleton`, `build_slice`,
  `lora.apply(first_index)`, `hf.load_into(span)`. MODEL-I6 rewritten: replay of the unchanged init stream, layer by
  layer, owned layers at their home, structure from the meta skeleton, hf over the span, user models keep the full
  build. Task 1: no recipe version; the per-parameter generator is not built (reopened by ADR-0007 R1 or R4). §6: the
  decision and its rejected alternatives.
- 2026-10-09 — ADR-0007 round 1 (review findings; awaiting the human grant). §2 determinism: identical on every host of
  the record's sampler fingerprint; `build_slice` takes the record. §3: `init_phases`, `init_record`,
  `sampler_fingerprint`. MODEL-I6 rewritten: one init program (`build_layers` is the phase-major walk of `init_phases`,
  the legacy code a fixture); the init record (ledger, per-tensor sha256, fingerprint, diagnostic class); writers of a
  home tensor, owned or not, load and store it (the tied table's last write is the Head's); ledger replay with
  `INIT-LEDGER` and `INIT-DIGEST`; host memory one working layer. Task 1 and §6 follow.
- 2026-10-09 — ADR-0007 round 2 (review findings; awaiting the human grant). MODEL-I6: `init_phases` is data with a write
  set per phase and a program id; the record's binding (cfg, seed, program id) and anchor (`manual_seed(seed)`);
  `init_record` runs pinned in a fresh process, makes one phase-major walk on NaN-filled skeleton layers (`INIT-PROGRAM`
  on a write-set violation) and hashes each tensor after its last writer, so the second pass is gone; the fingerprint
  includes the ATen CPU capability; `build_slice` finalizes each home tensor after its last writer (the tied table is
  held on the CPU until the Head's phase); hf replays no random phase; host memory `4·(P_layer,max + P_held)`. Task 1
  and §6: the per-launch tie with recipe v2, the portable sampler deferred with its rule.
- 2026-10-09 — ADR-0007 round 3 (review findings; awaiting the human grant; no mechanism changed). MODEL-I6: the
  `gpt2_init` phase is in `init_phases(cfg)` exactly when the build reaches `arch.build_layers` (`arch="hf"` or
  `_uses_general_arch(cfg)`) and `init_scheme == "gpt2"`, as today; a builtin-shaped config that declares `gpt2` has
  no such phase (the declared field alone would have changed its init bits). The build's host memory adds `4·P_s`
  for a CPU `home`. §6 decision: 88 → 94 s per launch, record 708 → 811 s, the per-parameter generator's record
  88 → 101 s on 8 cores (the record walk's memory passes and the slot's trim copy, `cost_r3.py`).
- 2026-10-09 — ADR-0007 round 4 (review findings; awaiting the human grant; no mechanism added). §2: identical bits
  follow from a sampler that computes the reference's bits; the fingerprint is stated as a necessary condition and the
  record's digests as the proof (round 3 had made a fingerprint match imply equal bits, which the ADR does not claim).
  §3: `skeleton(cfg)` exists for builtin, general and hf. MODEL-I6: materializing a skeleton slice re-ties shared
  parameters from the skeleton's sharing map (`to_empty` unties them, MEASURED by `docs/research/adr-0007/probe_r4.py`);
  under hf a home tensor's writers include the load of its checkpoint source, so a tied Head-only stage reads
  `embed_tokens` into `out.weight`, and every hf home tensor is checked against its digest; `arch="user"` has no
  skeleton and keeps the full build for build, resume, PROFILE and FIT. Task 1: the 1% reopening trigger and the
  portable-sampler design sentence are removed (the ADR does not decide them).
- 2026-10-09 — ADR-0007 round 5 (review findings; awaiting the human grant; no mechanism changed). MODEL-I6: the
  per-layer LoRA phase is `lora.apply`'s wrapping primitive without the empty-match error (a per-layer call on the
  Embed, or the Head under `attn`, would have raised, `lora.py:113-119`); the "matched no Linear layer" refusal and
  `WT_LORA`'s counts are whole-program, from the skeleton. The record's encoding is fixed (base64 states, hex digests).
  The hf bullet states the consequence of replaying no random phase: the ledger states after the anchor up to the first
  LoRA entry, and the end state without LoRA, are trusted from the bound record (ADR-0007 R10, open).
- 2026-10-09 — ADR-0007 human decision 6 (awaiting the human grant). MODEL-I6: the binding adds the code id (sha256 of
  `src/wantrain/model/{gpt,arch,lora,hf,wgrad}.py`, `torch.__version__`, `torch.version.git_version`); under hf the slot
  replays its writers' random phases in a scratch working layer, checks each state against the ledger (`INIT-LEDGER`)
  and discards the values before the load, so every transition is checked on every arch (R10 closed).
- 2026-10-09 — ADR-0008 round 0 (awaiting the human grant). §2: pending W of an MoE block; the normative MoE block
  (fan-out fold, fp32-output router, stable-sort selection, left-fold renormalization, (expert, token) dispatch with a
  folded backward, one call per expert on exactly its rows also when empty, fp32 left-fold combine, per-layer aux
  injected at the block output, no scatter-add or atomics, no RNG). §3: `moe_*` fields, `qkv_bias`, `qk_norm`,
  `MoEMLP`, the MoE config map. MODEL-I4: unplaceable tensors, router jitter, a short sliding window and stacked
  experts are refused. MODEL-I8 (new). Task 3 (M2c). §6: a decision bullet.
- 2026-10-10 — ADR-0008 round 1 (awaiting the human grant). M2c → M4 throughout. §2 MoE block: the routing recipe
  fields (score, selection key, groups, norm eps, scale, weight placement, router and expert biases, clamped SwiGLU,
  aux scope, z-loss, bias rate), the kernel-vs-written-order reduction rule, the fan-out for any tensor with ≥ 3
  consumers, group selection by the stable sort, the selection-bias update. §3: `hf_unplaced`, `moe_bias_update`,
  refusal of other MoE model_types, the HF-modules substitution rule. MODEL-I4: tensor classification (buffers and
  explicit truncation reported; unplaced parameters refused, `hf_unplaced = "report"` as rollback), refusal of
  `output_router_logits` without recipe aux fields, the fused-layout byte map (the stacked-layout refusal removed).
  MODEL-I8: every recipe row, group ties, NaN keys, the bias update. Task 3 and §6 updated.
- 2026-10-10 — ADR-0008 round 2 (awaiting the human grant). §2: `moe_router_dtype` defaults to `"compute"` and the
  loader sets each family's own; `"compute"` runs the router GEMM under autocast (compute-dtype logits, as HF's gate),
  `"fp32"` with autocast off; the selection-bias update runs at the bias's owner from int64 count vectors (RUNTIME-I18
  carries them). §3: `head_dim`, `norm_eps`, `rope_style`; `arch.moe_forward` over a parameter view; the HF-modules
  rule now keeps HF's objects, names and layouts and replaces each sparse-MoE block's forward through a (class,
  transformers version) table with its call contract (the `MoEMLP` substitution removed: other names, other call
  contract). MODEL-I4: `config_from_hf` is an allowlist (mapped, inert, accepted only at a listed value, else refused);
  `head_dim`, `rms_norm_eps` and the RoPE convention mapped; fused tensors only through the orientation table, an
  unknown fused key refused even when shapes fit. MODEL-I8: the HF-object block equals `MoEMLP` bit for bit. Task 3:
  the new fields, the allowlist, real-config fixtures, `transformers` as a pinned test-only dependency, `moe_forward`.
  §6: HF's block re-ranked on the measured reasons (exact on one CPU map; `None` gradients, MPS `index_add_`, `topk`
  across backends); the grouped GEMM exact under the same call on both paths.
- 2026-10-10 — ADR-0009 round 0 (awaiting the human grant). §1: HF checkpoints run transformers' own modules. §2: a
  normative "HF modules" bullet (environment guard, skeleton with parameters on meta and CPU buffers, MoE forward
  replacement or refusal, unit trace, unit windows, 1:1 loading and export, the pin). §3: the `hfm` interface; the
  native `hf.load_into` and `_LLAMA_*` removed for HF checkpoints. MODEL-I4: 1:1 HF names, unused and renamed tensors
  refused. MODEL-I6: the hf code id includes `hfm.py` and the fingerprint; the HF-modules build draws nothing (ledger =
  anchor); from-config HF keeps the full build. New MODEL-I9 (units compose to vanilla HF bit for bit) and MODEL-I10
  (launch gate). Task 4. §6: the decision.
- 2026-10-10 — ADR-0008 round 3 (awaiting the human grant; mechanism frozen). ADR-0009 is in the tree, so the
  native loader is keyed on its grant. §2: the pending-W note for the HF-object forward (routed pending W 0, dW in B);
  `moe_router_dtype` is set by the HF-object table, or by the loader under decision 6b. §3: `qkv_bias`, `qk_norm`,
  `head_dim`, `norm_eps`, `rope_style`, `hf_unplaced` and `config_from_hf`'s MoE types only if ADR-0009 is not
  granted; the HF-object table keyed on the pinned transformers with 5.19.0 rows only (one-tensor contract), its v5
  forward written out (`unbind(0)`, `chunk(2, dim=0)`, f-wide GEMMs of `MoEMLP`'s shapes, never two index slices);
  the 4.x tuple contract and per-expert `nn.Linear` rows removed. MODEL-I4: ADR-0008's paragraph applies only if
  ADR-0009 is not granted. MODEL-I8: the HF bullet at 5.19.0, sign of zero included. Task 3: split by decision 6
  (6a: `moe_forward` and the table; 6b: the loader fields, allowlist and fidelity checks at transformers 5.19.0).
- 2026-10-11 — ADR-0008 round 4 (awaiting the human grant; mechanism frozen). §2: `moe_groups` G > 1 is refused
  unless G divides E, E/G ≥ 2, 1 ≤ `moe_topk_groups` ≤ G and k ≤ `moe_topk_groups`·E/G (the group key "sum of the
  top-2 keys" was undefined for groups of one). §3: the HF-object replacement is a class swap to a registered
  subclass whose own `forward` calls `moe_forward` (never an instance attribute); the note that ADR-0009's unit trace
  and classifier refuse the block as written, pending ADR-0008 decision 11. MODEL-I8: the group refusal.
- 2026-10-11 — ADR-0009 round 1 (awaiting the human grant). §1 Deferred dW: under HF modules `convert` takes the
  trace's exclusion set (a Linear read outside its own forward stays `nn.Linear`, dW in B: mamba's `dt_proj`). §2 HF
  modules: `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, `unsloth` refused, a `boundary_dtype` other than HF's residual dtype
  refused; the unit trace on independent ids at a distinct batch size, with a function-mode spy, the
  `base_model_pp_plan` cross-check, re-tied aliases, refusals for a Parameter in two units (other than the tie) and a
  non-floating layer output; stubs return fresh NaN tensors, never recorded values. §3: `hfm.gate(..., device)`.
  MODEL-I9: M ≥ 2 under the stock fold (the per-use fold removed). MODEL-I10: the gate also on the slot's device, its
  microbatches independent of the trace's; the fingerprint by RECORD plus code objects. Task 4: E1, E2, E5–E8. §6: 102
  of 178 types, 8.18e-3, the rejected alternatives updated.
- 2026-10-11 — ADR-0008 round 5 (awaiting the human grant; mechanism frozen). §2 pending-W note: under the HF-object
  forward B leaves each 3-D bank's gradient, `b_m·E·3hf` bytes per MoE layer, held until W(mb) starts and charged by
  SCHEDULER-I21 (it was stated as pending W 0 only, which under-charged memory). §3: MODEL-I10's launch gate also
  refuses the swapped block as written (its vanilla reference runs HF's own block and has no aux gradients), pending
  ADR-0008 decision 11.
- 2026-10-11 — ADR-0009 round 2 (awaiting the human grant). §2 HF modules: `PYTHONPYCACHEPREFIX` in the environment;
  buffers overwritten from the run's buffer record; the sparse-MoE rule narrowed to `(?i)(sparse|moe$|experts?$)` and
  admitted only inside a module with a table row (round 1's `(?i)moe|expert|sparse` matched norms and attention of MoE
  families); units run on the unit's device; layer stubs are stride-0 NaN views, prologue stubs fresh NaN tensors.
  MODEL-I6: the code id binds transformers' RECORD sha256 (not a per-platform fingerprint). MODEL-I9: per device, and
  the device-map reference is the units in one process on that map; buffers equal the record; an in-place write into
  a stride-0 stub refuses. MODEL-I10: the causality probe; the fingerprint per platform (`hfm.TORCH_RECORDS`), the
  `__pycache__` exemption stated, code objects compared by value with the compiled source. §6: 60 / 30 / 12.
- 2026-10-11 — ADR-0009 round 3 (awaiting the human grant; mechanism frozen, text corrected). §2 HF modules: the
  buffer record is built once at the first launch under the environment guard, its bytes in the JobSpec and every wave
  (RUNTIME-I19); `lora_rank > 0` refused under `arch="hf"` (today's target sets name native leaves: `attn` matches
  nothing on Llama-named HF modules, `attn_mlp` only `fc1`/`fc2` on phi-style ones); the NaN claim narrowed to
  arithmetic leaks (R13). §3: `hf.load_into` (with or without `span`) and `hf.config_from_hf` only if ADR-0009 is not
  granted (round 2 listed them as provided and removed at once). MODEL-I6: `gpt2_init` for `arch="hf"` and the hf
  ledger-replay bullet only if ADR-0009 is not granted (compressed); LoRA phases on hf replaced by the refusal.
  MODEL-I9: the per-map reference exists only where one host holds the map's device classes (R11). Tasks 1 and 4 match;
  §6's ADR-0009 decision compressed (same facts) to stay under the 40 KB cap.
- 2026-10-11 — ADR-0009 round 4 (awaiting the human grant; mechanism frozen, text corrected). §2 HF modules: the
  bytecode prefix is `sys.pycache_prefix`, set in-process (`PYTHONPYCACHEPREFIX` is read only at interpreter start;
  set in-process it leaves `sys.pycache_prefix` None, MEASURED); the dynamic-RoPE refusal keyed on
  `rope_parameters` per layer type (`"dynamic" in rope_type`, as the pinned wrapper tests), plus `"longrope"` whose
  `original_max_position_embeddings` is below the recipe's S (both rebuild `inv_freq` inside forward,
  `modeling_rope_utils.py:47-127`; round 3 keyed the v4 `rope_scaling.rope_type == "dynamic"`). MODEL-I6: the hf
  checkpoint record digests the loaded parameters only; buffers are checked against the buffer record's pin alone
  (round 3 had the init record digest buffers, which it makes on another host before the buffer record exists:
  `INIT-DIGEST` on every slot of a mixed fleet). MODEL-I9: the reference is vanilla HF in one process on a map of one
  device class (the units were the oracle of themselves), the units in one process on a mixed map one node holds, any
  other map refused (decision 8's recommendation, CONTROL-I24). Task 4: 4a (the `arch="hf"` refusal and the loader's
  deletion) lands alone first as the rollback target, then 4b. §6's ADR-0009 entry compressed (same decision) to stay
  under the 40 KB cap.
- 2026-10-11 — ADR-0009 round 5 (awaiting the human grant; mechanism frozen, text corrected; `round5.md`). §2 HF
  modules: (a) the environment is set by the worker on the process's command line (`-X pycache_prefix`), in force
  before torch's import; the in-process assignment missed 726 of 726 torch modules (MEASURED). Loader variables
  (`LD_PRELOAD`, `LD_LIBRARY_PATH`, `DYLD_*`) are refused. (b) The importable-package deny-list becomes: a distribution
  named by a pinned `_is_package_available` call (`hfm.PROBED`) installed outside the `hf` extra's lock is refused, so
  `mambapy` (which switches mamba's scan) is refused with the old seven; `unsloth` stays refused by name. (c) The
  sparse-MoE rank rule counts a rank ≥ 3 parameter only when its leading dimension is the config's expert count (one
  of four keys); it flagged `Conv1d` weights (C, 1, k) and rwkv's (1, 1, h) and refused 8 of the 60 dense in-scope
  types, mamba included. (d) `hfm.fingerprint` also checks the distributions torch requires in `uv.lock` (`nvidia-*`)
  against their RECORDs, and asserts every loaded torch or transformers module's `__cached__` lies under the prefix.
  §3, MODEL-I6, MODEL-I9: compressed (same meaning; the oracle's per-map definition is SPEC-runtime §2's) to fit the
  40 KB cap. Task 4: a rollback keeps 4a and reverts 4b with runtime task 7, control task 6, scheduler task 6 and the
  `TIED` type by task id, and withdraws task 3's HF objects (ADR-0008 decision 6a) until 6b is built. §6: two tiers
  deferred to the precision ADR (they win the WAN term; a native family is another function).
- 2026-10-11 — ADR-0008 round 7 and ADR-0009 round 6 (correction pass under the autonomy switch; decisions taken as
  recommended, "decided by autonomy, 2026-10-11"). By ADR-0008: (a) §2 pending W: the HF v5 bank gradient is live
  from B(mb)'s end until W(mb) starts and a peer holds it `m_BW` times (its microbatches between B and W in the
  emitted order); the scheduler charges `m_BW·b_m·Σ E·3hf`. (b) §3: the HF-MoE seam rule replaces "as written ...
  refuse": router and experts rows (`*TopKRouter`, `*Experts` at 5.19.0) admitted under a block row, called once each
  as modules; MODEL-I8's HF bullet requires those calls and names F8 as the block's own proof. MODEL-I4: the
  hub-to-fused byte map (decision 10 (i)) with its round trip. MODEL-I4's decision-6b paragraph is not built
  (decision 6a) and is moved here verbatim to fit the 40 KB cap:
  > (M4, ADR-0008 decision 6b: only if ADR-0009 is not granted; with ADR-0009 the paragraph after this one
  >   governs and this one is not built) Every checkpoint tensor is classified: placed; a recomputed buffer
  >   (`*.rotary_emb.inv_freq`), reported; a layer `≥ n_layer` under an explicit recipe truncation, reported; otherwise an
  >   unplaced parameter, refused naming it under `hf_unplaced = "refuse"` (the default) and reported under `"report"`
  >   (recorded in the run fingerprint). `config_from_hf` is an allowlist per `model_type`: every config.json key is
  >   mapped, inert (it changes no computed bit), or accepted only at a listed value (`rope_scaling` null, `attention_bias`
  >   false unless `qkv_bias` maps it, `mlp_bias` false, `attention_dropout` 0, `router_jitter_noise` 0,
  >   `partial_rotary_factor` absent or 1, `pretraining_tp` 1, `use_sliding_window` false or `sliding_window` null or
  >   ≥ `seq_len`, `output_router_logits` false unless the recipe sets its own aux fields); any other key or value is
  >   refused naming it (ADR-0008 derivation §13). `head_dim`, `rms_norm_eps` and the RoPE convention (`rope_style =
  >   "half"` for every HF RoPE family) are mapped; `router_aux_loss_coef` is inert and never mapped. Fused expert tensors
  >   load only through the (model_type, tensor) orientation table of ADR-0008 derivation §8 (per expert a strided view
  >   copied element for element, every source element exactly once); a fused tensor whose key is not in the table is
  >   refused even when its shapes fit.
  By ADR-0009: (b) MODEL-I9's reference is the vanilla model with §3's SUBSTITUTE swap and aux injection applied
  alike. (d) §2: sparse-MoE blocks are classified by routing structure (a selection over the expert dimension whose
  indices index an expert-leading parameter or choose `ModuleList` members), not by parameter rank or class-name
  regex; `Conv1d` and rwkv's rank-3 vectors are dense by construction. (e) §2: every name a pinned `is_*_available`
  probes that is importable (`find_spec` or the probe) refuses unless the `hf` extra locks it; MODEL-I10 adds the
  probe vector. (i) §2: the prefix is an interpreter-start fact (`-X`), never assigned in-process; the guard checks,
  so it may run after `hf.py:17`'s `import torch`, and covers `wantrain` modules too. (j) MODEL-I10: loaded native
  objects tied to RECORD files or recorded by sha256 per (platform, device class); gpu false checks the macOS and
  Linux-CPU arms, `nvidia-*`/`libcuda` at M1. (g) Task 4: rollback per piece, each to its own target. §6: the two
  rejection lists shortened to pointers to the ADRs' Alternatives (same verdicts).
- 2026-10-11 — reconcile check of the ADR-0008 round 7 / ADR-0009 round 6 correction pass (independent reviewer,
  autonomy; item i, ADR-0009 D5/D9, `round6.md` §6). MODEL-I10's bytecode check now names `wantrain` modules beside
  torch and transformers, as §2 and ADR-0009 D5 already did; §2 states that a process started without
  `-X pycache_prefix` fails the check (no prefix; in-tree `__cached__`) and is refused. No mechanism change.
- 2026-10-11 — M4 implement deviations recorded (model-families, hf-pin; amended `--by ADR-0008` after the
  coordinator's verdict request: the probe reading checked against ADR-0008 decision 11 / ADR-0009 D8 and MODEL-I8,
  I9, I10). Verified against `src/wantrain/model/hfm.py` before writing.
  (1) MODEL-I10's causality probe: during the probe only, each substituted expert runs one GEMM per routed row
  (`hfm._rowwise_experts`, `hfm.py:681-696`, used only at `hfm.py:2610` under `torch.no_grad()`; `_SwapExperts.forward`
  `hfm.py:663-673`). Why: the probe (ADR-0009 `round2.md` §4) exists to refuse a model whose earlier positions read
  later tokens (bidirectional encoders); its premise "GEMM order does not depend on values" is false for an exact MoE
  block, whose per-expert row counts move with a later token's routing, and a CPU `F.linear` row's bits depend on the
  row count (MEASURED by model-families, torch 2.14.1, Apple M-series CPU: 6 of 39 counts at 32x64, 7 of 39 at 64x32;
  qwen2_moe and qwen3_moe were refused GATE-CAUSAL, Mixtral passed by chance). One row per GEMM removes only the
  row-count term; routing, selection, combine, attention and every other op still run on the whole row, so a real
  cross-position dependence still fails the probe. Training and the bitwise comparison run the batched GEMMs, the
  same function as the oracle (MODEL-I8: one function of (microbatch, parameters) on every path).
  (2) §2 D8: "a tagged value indexes a parameter leading with E" widened to "meets a parameter leading with E in one
  op" (`hfm.py:956-968`), so llama4_text's score-weighted dense dispatch is classified and refused MOE-NO-ROW.
  (3) §2 unit trace: it runs on the swapped model after D8 classified the unswapped one (`hfm._traced`,
  `hfm.py:1113-1130`), and refuses TRACE-UNPLACED a parameterized module of a layer the forward never calls unless an
  op elsewhere reads its parameters and its class is not a SUBSTITUTE class (`hfm.py:1021-1035`).
  (4) §2 pin: the `hf` extra is `transformers==5.19.0` and `safetensors>=0.4` (`pyproject.toml:21`), locked with
  hashes (`uv.lock`: transformers 5.19.0, safetensors 0.8.0).
  To stay under the 40 KB cap, two superseded decision-6b passages (ADR-0009 granted, decision 6a) moved here verbatim:
  §3 Interfaces, after `moe_bias_update(layers, counts, global_tokens)`.:
  > Only if ADR-0009 is not
  > granted (ADR-0008 decision 6b; with ADR-0009 none of these is built): `qkv_bias`, `qk_norm` (per-head RMSNorm on q
  > and k before RoPE), `head_dim` (q and o widths `n_head·head_dim`, k and v `n_kv_head·head_dim`; default
  > `d_model // n_head`), `norm_eps` (default 1e-6), `rope_style` (`"pairs"`, the default, or `"half"`, HF's rotation of
  > halves), `hf_unplaced` (`"refuse"` default, `"report"`), and `hf.config_from_hf` for `mixtral`, `qwen2_moe`,
  > `qwen3_moe`, refusing every other MoE `model_type` naming the missing piece.
  §5 task 3, after "no native HF loader.":
  > Only if ADR-0009 is not granted (decision 6b, MODEL-I4's
  > ADR-0008 paragraph): `qkv_bias`, `qk_norm`, `head_dim`, `norm_eps`, `rope_style`, the allowlist loader for Mixtral
  > and Qwen2/3-MoE per-expert and fused names (the orientation table) with its classification and refusals, checked
  > on the real config.json files (fixtures with URL and sha256) and for function fidelity against transformers 5.19.0
  > (a test-only dependency group); no HF-object table.
  Tests this requires (MODEL-I10): `test_moe_hf_gate_admits_the_swapped_model` parametrized over `mixtral`,
  `qwen2_moe` and `qwen3_moe` (today only mixtral and qwen3_moe: `tests/unit/test_finetune.py:2387`), each passing
  with no GATE-CAUSAL; a tiny swapped qwen3_moe made non-causal (a module mixing a later position into an earlier
  one, e.g. attention without its causal mask) refused GATE-CAUSAL with the row-by-row probe on (covers: MODEL-I10,
  WOULD-FAIL-IF: the row-by-row probe hides a real cross-position dependence); `hfm._ROWWISE[0]` is false after
  `hfm.gate` returns or raises (WOULD-FAIL-IF: the probe's mode leaks into training or the bitwise comparison); bert
  stays refused GATE-CAUSAL (`test_hfm_gate_refuses_bert_and_names_the_cause`).
