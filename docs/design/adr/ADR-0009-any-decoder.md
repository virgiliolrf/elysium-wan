# ADR-0009: Any HuggingFace decoder through transformers' own modules, windowed per pipeline unit, behind an exact launch gate

<!-- Approval lives in docs/design/APPROVALS.json (agents-architetura/tools/factory/approve.mjs), bound to this exact text. Do not write a
     Status line claiming approval. Mark the ADR "Status: REJECTED", "Status: SUPERSEDED" or "Status: WITHDRAWN" only
     when that is the human's decision. -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE DECISION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: M4 asks for any HuggingFace decoder-only LLM without a rewrite per family. Today `model/hf.py` maps 14
families into native blocks and nothing compares a logit. Bottleneck: correctness coverage (T_support), not step time;
MEASURED: tiny llama, llama-yarn, qwen3 differ from transformers by max|Δlogit| 8.18e-3, 6.26e-3, 0.296.

Proposed architecture (one decision: what a pipeline unit runs for `arch="hf"`):
1. *Units are windows of HF's own forward*: the flat list stays `[prologue, layer 0..n−1, epilogue]`; unit u runs
   `ForCausalLM.forward` of one shared module tree on its device, input injected at u's entry, stopped at the next
   unit's entry, every stubbed value a stride-0 NaN, so glue runs where HF runs it and only the residual crosses a cut.
2. *Skeleton*: parameters on meta, loaded 1:1 by HF names; buffers are recorded bytes of the run, built once.
3. *Exactness policy*: eager attention, strict determinism, dropout 0, no RNG, no unlocked optional package, no remote
   code, offline, verified bytecode; `SplitLinear` only where no other op reads W. 4. *Tied tables keep the stock order*.
5. *A launch gate*: unit chain vs vanilla HF on a tiny config of the run's class, every cut, bitwise, a causality
   probe, on the CPU and the slot's device; a code and native-library fingerprint. 6. *ADR-0008's seam* (D8).

First-principles reasoning: in `logits = E(L_{n−1}(…L_0(P(ids))))` every side input of a layer (cos/sin, masks,
layer type) is a function of (ids, positions, config, buffers), not of hidden values. The same code on the same
operands on the same device gives the same bytes, so a cut carrying only the residual is exact if every glue op runs
once, in its place, and every gradient reaches its parameter. The identity reference is the datacenter's run (D2;
a map no host holds whole is refused, R11, decision 8).

Mathematical justification (`docs/research/adr-0009/round2.md`, `round4.md`, `round5.md`): `T_support` falls from
N_families·(port + test) to a gate per pin bump; the unit window costs 2.7–5.5% of an 80-layer last stage
(HYPOTHESIS); a 4 B cut vs native 2 B is ×2.43 wire bytes (decision 3); eager saves 6 B per `b·H·S²` per layer; a tied
table across stages adds ≈ 197–414 ms against a 438 ms step (Gemma2-2B, S 4, M 8, 20 ms, 10 Gbps; DERIVED).

Expected improvement: from 0 families computing HF's function to 60 dense in-scope types of the census's 102 EXACT
(none DIFFER), bit-identical at every cut, no family code, as far as the gate sees (R13); of 30 sparse-MoE types,
Mixtral and Qwen2/3-MoE train through the seam rule (R15 CLOSED), the rest are refused until a row exists; 12
encoders refused (by class names, HYPOTHESIS until E7's census).

Alternatives: A0, A0′, A0″ (do little or tune what exists), A1–A5, on one cost model (`round6.md` §3): A0/A4 win the
WAN term below ≈ 7 Gbps (×2.43 at 1 Gbps), A4 deferred to decision 3. Risks: R1–R18 (R11 refused at every
admission; R15 CLOSED; R13, R14, R16–R18 open, R18 narrowed).

Falsification criteria: a differing bit in a true arm of E1–E4 or E8; a fired must-fail control that does not diverge
(E2); an identity mismatch against the vanilla oracle (E3, H1 at full size); an E4 arm off `ΔT_tied` by more than
max(20%, 2 ms); a drift E6 does not refuse; E7's window cost, saved bytes or census out of bounds.

Local experiment: E1–E8 on this Mac (CPU; E8 on MPS), no downloads. Vast.ai cost: none now (gpu false); H100 at M1.
Recommendation: PROCEED; decisions 1–8 decided by autonomy, 2026-10-11 (each its recommendation); R13, R14, R16–R18 open.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Awaiting explicit human approval: node agents-architetura/tools/factory/approve.mjs grant docs/design/adr/ADR-0009-any-decoder.md docs/design/DESIGN.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Problem

M4 is "any LLM, dense and MoE": a decoder family released next month trains byte-identically to its reference with no
new code here, or is refused with the reason. Encoder-decoders, encoders and multimodal prefix encoders are out of scope.

## Current Architecture

- `src/wantrain/model/hf.py:27-70` allowlists 14 `model_type`s; `:124-138` maps Llama names only; `load_into`
  (`:220-327`) drops unmapped tensors; `:249-256` copies q/k unpermuted while `arch.py:82-84` rotates adjacent pairs.
- `arch.py:108,180`: `boundary_dtype` at every block exit; `wgrad.py:127-143` swaps every untied `nn.Linear` for
  `SplitLinear`; `lora.py:68-72,101`: LoRA on native leaf names; `pyproject.toml`: no transformers.
- `validation/reference.py:27-94`: one device, one `autograd.grad` per microbatch, fold in index order (a tied table's
  two uses summed by autograd); no tied table synchronized across stages in `runtime/` (grep); the CPU+MPS fleet test
  checks that the run trains, not its bits (`test_coordinator.py:188-216`). DESIGN §10: a chain of duplex hops.

## Current Bottleneck

`T_support`: a port and test per family, still wrong (Evidence 1), ~17 new families per 4–5 months (Evidence 11).

## Evidence

MEASURED on the local Mac (Apple M4, CPU, torch 2.14.1, transformers 5.19.0, one thread, no network) at a62e4eb by
the committed scripts in `docs/research/adr-0009/` (README: script, command, output); "reported" = not re-run; full
wording in `round6.md` §11. 1. `native_vs_hf.py`: max|Δlogit| llama 8.18e-3, llama-yarn 6.26e-3, qwen3 0.296.
2. Stub and inject: 70 arms exact; a 1-ulp boundary changes the logits; eager ≠ sdpa; sdpa ignores Gemma2's softcap; a
query-chunked eager changes dK, dV. 3. `proto.py`: 22/22 unit-window arms exact (11 families, 6 stages, M 2, deferred
W, tied stock order). 4. Granite re-entered through `inputs_embeds` diverges. 5. Census (`census3.py`, the 178 types of
`MODEL_FOR_CAUSAL_LM_MAPPING`, shrunk, fp32, M 2, a cut after every unit): 102 EXACT, 0 DIFFER, 61 refused or errors
(13 RNG; none silent), 15 not built; by class names (`round2.md` §4) 60 dense in scope, 30 MoE, 12 encoders. 6.
`skel.py`: no draw; buffers equal a CPU build; `to_empty` leaves every one wrong. 7. `overhead.py`: HF 477, native 480
µs per layer F+B (one contended sample each). 8. `SplitLinear` = `nn.Linear` bitwise. 9. Autocast + fp32 parameters:
fp32 cut; bf16 params: bf16. 10. Import 1.19–2.34 s (reported); RECORD 2,725 files. 11. Churn (reported): ~17 new
causal-LM families per 4–5 months. 12. `tied_rowsparse.py`: row-sparse fold = stock at n_pend 1, 2, 3, 6; untouched
rows +0; per-use fold ≠ stock (≈4,600/262,144). 13. `ctrl_thread.py`: NaN stubs caught; no NaN-absorbing leak tested
(R13). 14. `fp_probe.py`: RECORD transformers 0.31 s, torch 12,863 files 2.87 s. 15. Export: 11 families exact. 16.
Eager's fp32 softmax then PV. 17. `libm_margin.py` (R17); `PYTHONPYCACHEPREFIX` exported in-process leaves
`sys.pycache_prefix` None. 18. `round5.md`: rank-3 dense parameters (`Conv1d`, rwkv); mamba's `mambapy` branch; a
prefix assigned after `import torch` leaves 726/726 torch modules on in-tree `.pyc`, `-X pycache_prefix` 0/726.

## Hypothesis

H1: windowing HF's forward per unit computes at every cut, on each device, that device's HF function bit for bit for
every family looping over a layer list with value-independent side inputs, no RNG, converted Linears read only inside. H2: every other family whose stubs reach the logits by arithmetic fails the gate,
the probe or the trace; a stub read only through a NaN-absorbing op (comparison, `where`, index, cast, `.item()`) whose
outcome coincides on the gate's ids is not detected (R13). H3: the row-sparse fold keeps the stock order.

## First-Principles Derivation

Normative texts: SPEC-model §2 "HF modules", MODEL-I9/I10, RUNTIME-I19, CONTROL-I24; D1–D11 give their reasons.

**D1. Where glue lives.** HF code sits at three levels: modules (attention, MLP, norms, rotary), `Model.forward`
(embedding glue, masks and rotary per layer type, the loop, the final norm) and `ForCausalLM.forward` (head, softcap,
scaling). An own loop re-derives the last two per family (A1); re-entering `Model.forward` with `inputs_embeds`
reapplies its glue per stage (A2, Evidence 4). Running HF's whole forward and choosing where to enter and leave puts
every glue op where HF puts it, once.

**D2. Units.** A trace of the gate's tiny model (ids independent of the gate's, a batch size equal to no config
dimension) classifies every parameterized module: prologue (unit 0), layer i (unit i+1, the unique `ModuleList` of
`num_hidden_layers` called in order, checked against `base_model_pp_plan`), epilogue (unit n+1); it refuses a parameter
in no unit or two (but the tie), a module called between layers or by two, a layer called twice, a draw, a non-floating
layer output. Unit u runs HF's forward on its slot's device with a pre-hook injecting its input at its entry, one
raising at unit u+1's entry, and every earlier layer and (u ≠ 0) prologue module replaced by a NaN stub in its traced
structure: a layer stub (the `n²/S` term) is `torch.full((), nan, dtype, device).expand(shape)` (stride 0, no byte
written, an in-place write raises: a refusal); a prologue stub (one per unit) is a fresh NaN tensor, which HF may scale
in place. A stub value carried to the logits by arithmetic is NaN and fails the gate (Evidence 13); a comparison,
`where`, index, cast or `.item()` absorbs NaN, so such a leak is caught only where its outcome differs on the gate's ids
(R13). No stub replays a recorded value. Side inputs are computed on the unit's device, as HF does. The oracle is
SPEC-runtime §2's (`round4.md` §2; one-class maps: vanilla HF at full size; R11, R16). The flat-list contract
(`layers[a:b]`, one tensor per cut) is unchanged for the planner, profiler, recompute, deferred W, ADR-0006 and 0007.

**D3. Skeleton, buffers and slices.** Every node builds the full `ForCausalLM`, parameters on meta, buffers on the CPU
(no draw, Evidence 6). Buffers are bytes of the run (`pow` differs by CPU ISA, `round2.md` §1): built once at the first
launch in a coordinator child under D5's guard and an admitted D9 fingerprint, pinned by sha256 per tensor in the
JobSpec, state file and every wave, never rebuilt (RUNTIME-I19; constructor floats: R17). Owned parameters load on the
slot's device; a checkpoint's init draws nothing (`round4.md` §5); from-config keeps the full build.

**D4. Deferred W.** `wgrad.convert` skips tied Linears and every Linear whose parameter an op outside its own forward
reads (the trace's function-mode spy): those keep dW in B, as `Conv1D` and 3-D banks do (mamba's `dt_proj`).

**D5. Kernels and environment.** Only the family's own eager attention is admitted (asserted at the gate and every
unit): sdpa differs in bits and drops Gemma2's softcap, FLASH/EFFICIENT/cuDNN are nondeterministic or
architecture-specific (Evidence 2). Stage start asserts deterministic algorithms
(`warn_only` false), fp32 matmul `highest`, dropout 0, `use_cache=False`, HF checkpointing off, and no `rope_parameters`
entry (per layer type) whose `rope_type` contains `"dynamic"` or is `"longrope"` with `original_max_position_embeddings`
below S (both rebuild `inv_freq` in forward, `round4.md` §4). The worker starts every stage and buffer-builder process
in SPEC-model §2's environment: `-X pycache_prefix=<fresh dir>` on its command line, an interpreter-start fact in force
before any import (never assigned in-process: after `import torch` it misses 726/726 torch modules, Evidence 18), hub
kernels off, offline, no backend autoload, no `LD_PRELOAD`, `LD_LIBRARY_PATH` or `DYLD_*`. The guard checks, never
sets, so it may run after `hf.py:17`'s `import torch`: every loaded torch, transformers and `wantrain` module's
`__cached__` lies under the prefix, else refused (`round6.md` §6). Refused: any name a pinned `is_*_available` /
`_is_package_available` call probes (`hfm.PROBED`) that is importable (`find_spec` or the probe itself) unless the
`hf` extra locks it, then RECORD-fingerprinted; the probe vector is fingerprinted (`kernels`, `flash_attn`, `fla`,
`mamba_ssm`, `causal_conv1d`, `liger_kernel`, `mambapy`: each switches an algorithm, `round6.md` §2); `unsloth`;
`auto_map`; `is_encoder_decoder`; an unpinned `model_type`; a foreign `forward`.

**D6. Tied tables** (`tied.md`, `round2.md` §3; normative text RUNTIME-I19). The oracle folds `g_i = h_i + e_i` in index
order. The embedding backward leaves rows outside microbatch i's ids `T_i` at +0, so the Head (owner of the table,
its AdamW state and accumulator) folds those rows at once and holds the rest until `e_j[T_j]` resolves them in order
(Evidence 12). Stage 0 sends `TIED(t, i)` = `e_i[T_i]` forward when `B_i` ends (untouched rows +0, else fatal); the
Head sends `TIED(t+1, k)` = `R_k` backward after AdamW; intermediate stages cut-through. `TIED` is its own frame type
(`e_i` exists only after `FWD_i` left); no index crosses (`make_batch`); width 1 at both ends (R14).

**D7. Loader, export, tokenizer.** Names map 1:1 onto the skeleton's `state_dict()`; missing, unused (ADR-0008's
exceptions reported) and v5-conversion names refused, except the hub-to-fused expert map (ADR-0008 decision 10 (i),
its derivation §23.2): for `mixtral` (`.block_sparse_moe.` → `.mlp.`; `w1`, `w3`, `w2`) and `qwen2_moe`, `qwen3_moe`,
`deepseek_v3` (`gate_proj`, `up_proj`, `down_proj`), per layer and expert j `gate_up_proj[j, :f] ← gate_j`,
`gate_up_proj[j, f:] ← up_j`, `down_proj[j] ← down_j`, strided copies writing each element once, inverted on export;
a round-trip test (distinct bytes, −0 and NaN payloads) and equality with 5.19.0's own conversion gate it. Export
writes the same names and tied keys; tokenizer sha256.

**D8. ADR-0008's seam.** Classification by routing structure (`round6.md` §1), on one CPU forward of the gate's tiny
model before any swap: with E the config's expert count (`num_local_experts`, `num_experts`, `n_routed_experts`,
`moe_num_experts`; none: no MoE), a selection (`topk`, `sort`, `argmax`, …) over a dimension of size E tags its
indices; a module is sparse-MoE when a tagged value indexes a parameter leading with E, or chooses which of a
`ModuleList` of E members run; the block is the innermost such module. No rank test: `Conv1d` (mamba, falcon_h1) and
rwkv's vectors select nothing and are dense. A block without a SUBSTITUTE row refuses the model.

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

**D9. The launch gate and the pin.** `transformers==5.19.0` with hashes in `uv.lock` (the `hf` extra). Every node, after
D5's environment: (1) fingerprints the code: every transformers file against its RECORD (one `py3-none-any` wheel), the
RECORD's sha256 equal on every node; every file of torch and of each distribution its wheel requires in `uv.lock`
(`nvidia-*` on CUDA wheels) against its RECORD, the sha256s equal to the entry for (version, platform tag) in
`hfm.TORCH_RECORDS` (built by the pin-bump PR from `uv.lock`'s wheels), version and `git_version` equal on every node; a
file outside RECORD is refused except `__pycache__/*.pyc`, which the prefix keeps CPython from reading (D5's check);
each loaded `transformers` function's code object equals by value the one compiled from its verified source (D8's
subclasses are ours); the probe vector (D5); after the gate ran, every loaded native object (`/proc/self/maps`, or the
dyld image list): a file of a fingerprinted RECORD, tied by path and inode to the file its check hashed (a RECORD
basename loaded from elsewhere refuses: `LD_LIBRARY_PATH` outranks `RUNPATH`), or else an OS or driver object
recorded by sha256 (a dyld shared-cache image by cache UUID and OS build), equal across nodes of one (platform, device
class) (`round6.md` §7; gpu false: the Mac dyld and Linux CPU arms now, `nvidia-*` and `libcuda` at M1). The run
records the fingerprints (the buffer builder's too), each slot's torch RECORD hash, native set and device class, and
every admission re-checks them and decision 8's map (CONTROL-I24). (2) Runs the gate on its CPU and
the slot's device: a tiny config of the run's class (dimensions shrunk, every flag kept, `S` above every window and
chunk, every `layer_types` value present), vanilla (seam rule (4)) vs unit chain at every cut, fp32 and the recipe's dtype, M 1 and 2,
deferred W, recompute; logits, loss hex, every gradient bitwise; trigger counts; a causality probe (two id rows equal
but for the last position give bitwise equal vanilla logits before it). A difference, a probe failure, a declared
trigger that does not fire, or a build over 64 MB or 30 s refuses the run, naming it. A pin bump is a census PR.

**D10. Precision.** HF fixes the cut's dtype: fp32 under today's recipe (fp32 parameters, autocast), bf16 only with bf16
parameters (decision 3's ADR). A declared `boundary_dtype` other than HF's residual dtype is refused: narrowing only at
the cut changes HF's function; a per-layer narrowing (oracle too) is decision 3's. **D11. What stays native.**
`arch.py` keeps the presets; `hf.load_into` and `_LLAMA_*` are deleted; LoRA on `arch="hf"` is refused (Migration).

## Mathematical Model

`T_step = M·(t_F + t_B + t_W)_max + T_bubble + T_opt + T_exposed` (DESIGN §9); `T_support = Σ_fam T_port + Σ_bump
T_audit`: native N·(port + test), N +17 per 4–5 months; here one census per bump. Derivations: `round2.md`, `round4.md`.
- Host per stage and microbatch: vanilla plus `Σ_u (c_glue + n_before(u)·c_pass) + Σ_u b·S²·4/BW_mem` (stubs write 0 B):
  80 layers, last of 8 stages (825 stubs, 11 units) 4.7–9.6 ms host + 0.35 ms masks vs ≈ 176 ms of GEMMs, 2.7–5.5%
  (HYPOTHESIS), ×2 with recompute, hidden while the host runs ahead; PROFILE times units with windows; grows as `n²/S`.
- Attention (HYPOTHESIS until E7): eager saves `b·H·S²·s` B per layer, `s` = 4 (fp32), 6 (autocast), softcap +2 / +4:
  3.22 GB at H 32, S 4096. `T_comm` per hop: 4 B per element (3.40 after the codec) vs 2 B native (1.40): ×2.43; on a
  bandwidth-bound hop (7B, T_mb 2048, 1 Gbps) 228 vs 94 ms per microbatch, the step ×2.43 there (`round5.md` §4).
- Tied table across stages: `ΔT_tied` (throughput, warmup, round trip, +0 scan, row burst; `round5.md` §8). Gemma2-2B,
  S 4, M 8, L 20 ms, 10 Gbps: ≈ 414 ms worst, ≈ 197 ms text, base step 438 ms; Gemma3-27B, S 8, M 16: ≈ 495 ms of
  4.8 s (DERIVED). Charged by the planner; 0 under a V placement (`2a` per microbatch per direction).
- `T_overhead` per launch: import 1.2–2.3 s, RECORD 3.2 s, compiling 1–4 s, native set < 0.5 s, gate 1–2 s per device
  (HYPOTHESIS, E6); the buffer record (KB) per wave, one sha256 per load: 0% of step time.

## Expected Improvement

As the summary's; costs: the window (2.7–5.5%), ×2.43 boundary bytes, eager memory, `ΔT_tied`.

## Alternatives

| Option | Wins if | Rejected because (number) |
|---|---|---|
| A0 Fix the native path family by family | Bandwidth-bound hops: 2 B cut (1.40 after codec) vs 4 B (3.40), ×2.43: 11.7 vs 28.5 MB per hop per microbatch at 7B, T_mb 2048; busy-GPU 2.9 vs 7.0 Gbps per direction on 8 H100; 94 vs 228 ms at 1 Gbps, 33 vs 57 ms at 4 Gbps, equal ≥ 7 Gbps (`round6.md` §3) | Another function than HF's (per-block bf16 narrowing, D10; Evidence 1), so a port, oracle and gate per family (`T_support`, ~17 new per 4–5 months); its exact attention (`math` SDPA) is S² too (cap ≈ 2.9k vs eager's 2.4k at 7B, HYPOTHESIS); host 480 vs 477 µs decides nothing; the gate needs transformers anyway |
| A0′ Do nothing beyond a refusal gate on the native path (the rollback target) | Cheapest | The gate refuses every RoPE family (Evidence 1); 0 of 14 allowlisted families would train |
| A0″ Tune what exists: `arch="user"` with HF modules wrapped by the user into a flat list | No new mechanism; MODEL-I6 keeps the full build | The user writes the glue per family (A1's cost, `T_support = N·port`); the gate and fingerprint still needed for HF code; it measures the trace's share of the gain, so E7 times one wrapped llama against its windows |
| A1 Own loop over HF modules + per-family glue table + compat shim | Stable HF APIs | Consumes the APIs that churn (Evidence 11) |
| A2 HF `Model.forward` per stage, `inputs_embeds` re-entry (torchtitan, NeMo) | No glue outside modules | Glue reapplied per stage (granite diverges, Evidence 4) |
| A3 One stub-and-inject call per stage | Fewer wrapper calls | Same bits (Evidence 2); breaks the per-unit flat list of the runtime and ADR-0006/0007; kept as a fused fast path |
| A4 Two tiers (HF + native for hot families) | A0's numbers on the hot families' hops (×2.43 bytes, 2.9 vs 7.0 Gbps busy, S cap ≈ 2.9k vs 2.4k), the chosen design's elsewhere (`round6.md` §3) | DEFERRED to decision 3: each hot family computes two functions (HF checkpoint vs native) and pays a port, oracle and gate per bump; a bf16 HF residual brings every family to 11.7 MB per hop at once |
| A5 Export and split (`torch.export`, pipelining `SplitPoint`) | Static data-flow evidence | Exact whole-graph (Evidence 15) but side inputs become stage edges (mask `b·S²·4` B per hop) unless a graph pass recomputes them; one export per shape; deferred W and per-unit modules rebuilt as graph passes |
| Tied: per-use fold (Megatron); dense pending tables; the summed table per step; a direct Head↔stage-0 link | Simpler state; long chains | A recipe change (Evidence 12); 45 GB pending at Gemma3-27B vs ≤ 5.6 GB; 3.8 s per step at 10 Gbps for Gemma2-2B; DESIGN §10 rejects a wrap link (saves 80 ms at S 4), a topology ADR |
| Tied: V placement (ZB-V, M3) | Latency-bound chains | Not on M4's linear chain; `2a` per microbatch per direction: 165 ms each way at Gemma2-2B vs 414 ms relay worst, 197 ms typical; a planner choice once SCHEDULER-I14 lands |
| Tied: stage 0 folds and updates `U_t` itself | Large L, small `|U|` | The Head needs updated `w[U_t]` before its first F: ≤ 242 ms at M 8 on the critical path, above the 60 ms it removes |

## Trade-offs

Fidelity, no family code; against a pinned dependency, ≈ 6–9 s per launch, eager only, an fp32 cut, `ΔT_tied`.

## Risks

- R1 (HYPOTHESIS, M1): eager attention, masks and rotary run-to-run deterministic on CUDA under strict mode; bits per
  device map (DESIGN §3; a CPU class includes its ATen capability); a CPU GEMM's thread-count dependence: E8.
- R2: a shrunk config may hide size-triggered paths (D9's named triggers; E5). R3: no HF wrapper may swallow the exit's
  exception (pinned, gated). R4: a pin bump can refuse families that passed (intended). R5: eager memory (`s` 6) caps
  S near 2.4k at 7B on 80 GB (8-layer stages, 4 microbatches in flight); recompute (25.8 GB at S 4096) or decision 4's
  ADR lifts it. R6: ×2.43 WAN bytes until decision 3. R7: real configs may hit refused paths (never wrong). R8: ids
  shared with ADR-0008 renumbered by the later. R9: provenance as in Evidence; `saved_bytes.py` has not run (exit 75).
  R10: `ΔT_tied` (197–414 vs 438 ms) may favor fewer stages.
- R11 (refused by default, decision 8): a reference exists only where one host holds every device class of the map
  (CPU+MPS on a Mac; one GPU model). A map no node of the pool holds whole (x86 and ARM CPU slots, two GPU models) is
  refused for `arch="hf"`, re-checked at every admission (launch, heal, restart, rebalance, resume) from the current
  pool's gate reports, with each replacement of its slot's recorded class; gate, fingerprint and torch RECORD alone
  never admit (CONTROL-I24, `round6.md` §5). R12: `TIED`, a new frame type, is bounded by typed
  decode and `MAX_PAYLOAD`; a missing, duplicated, misshapen, out-of-order or unexpected (no tied relay) one is fatal.
- R13 (open): a stub read only through a NaN-absorbing op (`where(isinf(x).any(), a, b)`) whose outcome equals
  vanilla's on the gate's ids passes the gate (the census's 0 DIFFER is no evidence); the vanilla oracle catches it at
  full size on one-class maps (E3), never on a mixed map (R16). Closing it is a mechanism (poisoning, taint).
- R14 (open): `TIED` routing is specified at width 1 on stage 0 and the Head; a wider plan is refused (SCHEDULER-I22).
- R15 (CLOSED by the HF-MoE seam rule, D8; ADR-0008 decision 11, autonomy, 2026-10-11): the trace sees the router
  and experts as module calls, the classifier finds the block by routing and looks it up exactly, and the gate's
  reference carries the same swap and D8 injection; the block is proved by ADR-0008's F8.
- R16 (open): on a mixed map the oracle is the run's own unit chain, so a size-triggered window defect the shrunk gate
  misses (R2) passes identity there; one-class maps compare with vanilla HF at full size (E3).
- R17 (open): constructor floats that are not buffers (`head_dim**-0.5`, `attention_scaling`) come from each node's
  libm; for 1,099 of them every double a < 1-ULP libm may return rounds to one fp32 (MEASURED, `libm_margin.py`); another
  expression or a worse libm stays open. Closing it is recording them with the buffers (a mechanism).
- R18 (open, narrowed in round 6, `round6.md` §7): every loaded native object is now tied to a RECORD-verified file
  or recorded and compared across nodes of its class (D9), but OS and driver objects are compared, not verified
  against a source (equal tampering agrees), and other locked distributions are not fingerprinted. Closing it is an
  import-closure fingerprint or a cross-node recompute audit.
- Refuted (round 4, `round4.md` §1): a 119 ms backward-link excess (one pacing charged twice). House findings (not
  fixed here): `hf.py:117-119` resolves `auto` to flash on CUDA; `model_contract.py:332` uses `warn_only=True`;
  `wgrad.convert` on any arch loses the dW of a Linear read outside its forward; `runtime/determinism.py:16,22,23,25`
  pins by `os.environ.setdefault` (a pre-set value wins) and never reads `allow_tf32` back (`:30-32`): a planned M4
  task with SPEC-runtime task 7, its own test (`covers: RUNTIME-I19`, `WOULD-FAIL-IF:` a pre-set
  `ATEN_CPU_CAPABILITY` survives or `allow_tf32` is True after setup; `round6.md` §8).

## Falsification Criteria

- F1: in E1, a differing logit, loss hex or gradient bit at any cut, dtype or M in a true arm, or a causal family
  failing the probe. F2: in E2, a fired must-fail control that does not diverge or is not refused. F3: in E3, any
  sha256 of weights, AdamW state or RNG, or any loss hex, differs from the oracle, before or after resume.
- F4: in E4, the tied table differs from the stock oracle by any bit, a `TIED` path hangs (exit 124), or an arm's
  exposed time is off the full `ΔT_tied` (from its measured `t_s`, `t_F`, `t_B`, `|T_i|`, `|R_k|`) by > max(20%, 2 ms).
- F5: in E5, the skeleton draws, a buffer differs from the record, a tampered wave buffer loads, or a slice exceeds
  ADR-0007's bound. F6: in E6, a drifted node or record builder trains, a planted `.pyc` runs, an importable probed
  package, a loader variable or a shadowing native library is admitted, or a node matching its entries is refused.
- F7: in E7 (Mac CPU, uncontended): `c_glue` > 1 ms, `c_pass` > 10 µs, a stub allocating, HF host > 1.5× native, saved
  bytes per `b·H·S²` ≠ 4 / 6 / +2; the census with D8 < 55 in-scope EXACT (60 predicted), any DIFFER, an encoder admitted.
- F8: in E8, the MPS gate passes where the CPU gate does, yet a 2-stage MPS or CPU+MPS run differs from the
  one-process chain on the same device map.

## Local Experiment

All localhost, no downloads; tiny models written as safetensors by our own writer.
- E1 gate: llama (default, llama3, yarn, short longrope), mistral, qwen2, qwen3, gemma, gemma2, gemma3_text, phi3,
  olmo2, starcoder2, granite, cohere, gpt2, gpt_neox, minicpm3, falcon_h1, mamba; tied on and off; every cut, fp32 and
  bf16, M 1 and 2, deferred W, recompute; the causality probe.
- E2 must fail or be refused: `to_empty` buffers; granite through `inputs_embeds`; tied use dropped; exit one unit late;
  `layer_types` indexed locally; dropout 0.1; sdpa on gemma2; `kernels`; dynamic rope (per layer type); longrope past
  its length; zero prologue stubs; threaded state; mamba's read-elsewhere Linear converted; phimoe's RNG router; a
  split alias; bert; llama4_text, gpt_oss, deepseek_v3, jamba (D8, no row); an in-place stub write;
  LoRA on hf; a map of two CPU capabilities no node holds (R11, refused before placement) and a heal onto another
  class at every admission (refused); D8's classifier: mamba, falcon_h1 and rwkv admitted, Mixtral with its block
  row removed refused, a swapped block that skips its `experts` module call refused by the trace.
  Recorded, not judged (R13): `where(isinf(x).any(), a, b)` on the prologue output, gate verdict and run divergence.
- E3 identity: 3-, 4-stage CPU clusters (vanilla oracle), WAN 40 ms, 1 Gbps; llama, gemma2, granite, qwen3; a heal.
- E4 tied (`round4.md` §1, arms A1–A4: two vocabulary regimes, S 3–4, M 8–16, L 0–40 ms): exposed time vs `ΔT_tied`
  from measured terms; sha256 vs the stock oracle; a non-+0 untouched row fatal; a resume.
- E5 slices of an hf checkpoint vs the full build (buffers vs the record, RNG, peak RSS); real configs shrunk; a resume
  from a wave whose buffer file was edited (refused); an init record made on another host than the coordinator.
- E6 drift: edited `modeling_llama.py`, `activations.py`, `utils/generic.py`; an added file; a planted `__pycache__`
  `.pyc` with the source's mtime and size, under transformers and under torch (`nn/functional`); a sourceless `.pyc`; a
  local `mambapy` stub with `use_mambapy` (importable, no metadata); `LD_PRELOAD`; a torch native library loaded from
  a copy elsewhere; two nodes whose OS-object sets differ; a monkeypatch; `auto_map`; `kernels`; two test
  `TORCH_RECORDS` entries (admitted), a RECORD outside them and a drifted record builder (refused); the compile time.
- E7 performance: `saved_bytes.py`; `c_glue`, `c_pass`, unit-window F+B per layer at 4 and 32 layers against the
  vanilla forward and a user-wrapped list (A0″); allocator bytes during stubs; the census in three columns.
- E8 (MPS, gates nothing): E1's device gate on MPS; 2-stage MPS and CPU+MPS runs vs the one-process chain on the same
  map, CPU slot at 1 and 4 threads; eager bytes and time, S 2048/4096.

## Vast.ai Experiment

None (gpu false). M1: eager attention on H100, a 2-stage HF run's bits, the CUDA embedding backward (must fail), the
tied round trip across regions, an `nvidia-*` file edited and `libcuda` in the native set (E6's CUDA arm).

## Migration Impact

`arch="hf"` changes meaning (`hf_impl: "modules"`, fingerprinted): old hf runs do not resume. Lost: hf LoRA until
targets need no family code (`lora.py:68-72`); a declared `boundary_dtype` for HF (D10); hf on a map no node holds whole
(R11). Superseded from ADR-0007 task 1 (kept only if this ADR is not granted): `hf.load_into(span)` and MODEL-I6's hf
ledger replay.
Rollback, per piece, each with its own target (`round6.md` §4; reverted in this order, named by task id in the run
ledger): SCHEDULER-I22 (task 6) and runtime 7a (row-sparse tied fold) and 7b (`TIED` relay) → tied HF models at
S > 1 refused at planning, any `TIED` frame fatal; transport's `TIED` type → id unassigned, refused by typed decode;
runtime 7c (buffer record in JobSpec, state, waves) and 7d (HF oracle) → `arch="hf"` refused, a wave with a buffer
record refused on resume; model 4b → 4a; control task 6 (builder, `GATE`, CONTROL-I24) → 4a, never reverted while
4b is live; ADR-0008's HF pieces → MoE from scratch only. 4a stays (A0′); reverting it restores today's loader
(drops qwen3's q/k norms, `hf.py:273-275`; ignores rope scaling): never a target.

## Validation Plan

E1, E2 → `unit` (`covers: MODEL-I9`/`I10`); E3, E4 → `integration`, `identity` (RUNTIME-I19); E5 → ADR-0007's; E6 → `unit` (CONTROL-I24); census per pin bump.

## Ideas considered

Rounds 0–5: `round6.md` §9 (moved verbatim, verdicts unchanged but one). Round 6 (`round6.md` §10): ADOPTED the
gate reference with the same swap and the router and experts module calls (D8 seam rule, R15 CLOSED), D7's
hub-to-fused map, D8's routing classifier, and the probe vector and native objects in the fingerprint (D5, D9).

## Changelog

- Rounds 0–2: mechanism, evidence, census. 3 (frozen): R11, R13, R14; hf LoRA refused; rollback to A0′. 4: `t_s`, burst
  term; E4 arms; one-class vanilla oracle; task 4a; R15–R17. 5 (`round5.md`): rank rule keyed on E; probed packages;
  the prefix on the command line; native wheels, R18; heals keep the class; A0/A4 on the WAN term; rollback.
- Round 6 (correction pass, autonomy, 2026-10-11; `round6.md`): D8 by routing and the seam rule (R15 CLOSED); D7's map;
  probes by importability, probe vector and native objects fingerprinted (R18 narrowed); prefix ordering; decision 8 at
  every admission; A0/A4 on one cost model; rollback per piece; determinism house item; decisions 1–8 by autonomy.

## Amendments (texts in DESIGN.md and the specs, each in its -history.md and `freeze.mjs amend --by ADR-0009`)

Rounds 0–5: DESIGN §0, §2–§5, §8–§10, question 7; SPEC-model §1–§3, MODEL-I4, I6, I9, I10, tasks 1, 4, §6;
SPEC-runtime §2, I14, I15, I19, task 7, §6; SPEC-scheduler §2.3, I22, task 6; SPEC-control §1, §2.1, §2.3, I24, task 6;
SPEC-transport §2. Round 6: DESIGN §3 HF row, §5 Model code row, §10 question 7; SPEC-model §2, §3, MODEL-I4, I9, I10, task 4;
SPEC-runtime §2, I15, task 7; SPEC-control §2.3, I24, task 6; SPEC-scheduler task 6; SPEC-transport §2.

## Recommendation

PROCEED with D1–D11 and E1–E8 first. Decisions that are the human's (each decided by autonomy, 2026-10-11:
`autonomy.json` `humanGrants` false, the recommended option taken):
1. HF modules windowed per unit for every `arch="hf"` run; the native loader deleted; hf LoRA refused (recommended). Decided by autonomy, 2026-10-11.
2. `transformers==5.19.0` pinned with hashes, bumped only by a census PR that regenerates `TORCH_RECORDS` (recommended). Decided by autonomy, 2026-10-11.
3. Precision: the fp32 cut (×2.43 bytes) in M4, HF's `boundary_dtype` fixed; the precision ADR next (recommended). Decided by autonomy, 2026-10-11.
4. Attention: eager only, RUNTIME-I6's recompute past ≈ 2.4k at 7B (recommended); an attention ADR for long sequences.
   Decided by autonomy, 2026-10-11.
5. With ADR-0008: its pin is this one; its MoE function attaches through D8 under the HF-MoE seam rule (R15 CLOSED). Decided by autonomy, 2026-10-11.
6. Tied tables: the stock order by the row-sparse fold on the Head, rows in `TIED` frames on the chain, `ΔT_tied`
   charged (197–414 vs 438 ms, Gemma2-2B), width 1 at both ends until R14 (recommended); or refused at S > 1. Decided by autonomy, 2026-10-11: the stock order.
7. From-config HF runs keep the full build per slot until HF's init is expressed as ADR-0007 phases (recommended). Decided by autonomy, 2026-10-11.
8. Spec conflict: CLAUDE.md asks every hardware mix for the same bits, DESIGN §3 per device map; keep §3, remove the
   buffer part (D3), refuse `arch="hf"` on a map no node of the pool holds whole at every admission (R11); R13's and
   R18's mechanisms as own ADRs after M4's build. Decided by the human, 2026-10-11: homogeneous fleets only (CONTROL-I25).

## Human Approval

Granted only by `node agents-architetura/tools/factory/approve.mjs grant` in a human's terminal; recorded in
docs/design/APPROVALS.json.
