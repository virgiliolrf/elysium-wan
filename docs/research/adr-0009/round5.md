# ADR-0009 round 5: derivations behind the text corrections

Round 5 changes no mechanism (frozen since round 3). It corrects text, numbers and amendments for eight review
findings. Facts below are labelled CODE (read in transformers 5.19.0 from the uv cache, `scripts/paths.txt`), MEASURED
(command run on this Mac, Apple M4, CPython 3.12.13, torch 2.14.1, on 2026-10-11) or DERIVED (arithmetic).

## 1. The sparse-MoE classifier (D8, SPEC-model §2)

- CODE: `models/mamba/modeling_mamba.py:302-309` holds `nn.Conv1d(..., groups=intermediate_size)`, a weight of shape
  (C, 1, k), rank 3. `nn.Conv1d` also appears in `lfm2`, `mamba2`, `falcon_h1`, `falcon_mamba`, `recurrent_gemma`,
  `qwen3_5` and `olmo_hybrid` (`grep -l "nn.Conv1d" models/<type>/modeling_*.py`).
- CODE: `models/rwkv/modeling_rwkv.py:237-239`: `time_mix_key`, `time_mix_value`, `time_mix_receptance` are
  `nn.Parameter(torch.empty(1, 1, hidden_size))`, rank 3.
- So round 4's rule "a parameter of rank ≥ 3" flags mamba, falcon_mamba, falcon_h1, lfm2, recurrent_gemma,
  olmo_hybrid, qwen3_5_text and rwkv, all eight in the 60 dense in-scope types (`round2.md:117`), and refuses them
  (no ancestor has a SUBSTITUTE row). E1's mamba and falcon_h1 arms would fail.
- CODE: the expert count of every 5.19.0 config is one of four keys: `num_experts` (26 config files), `n_routed_experts`
  (21), `num_local_experts` (15), `moe_num_experts` (4) (`grep -rhoE "^\s+(num_local_experts|num_experts|
  n_routed_experts|moe_num_experts|n_experts|num_routed_experts)\s*[:=]" models/*/configuration_*.py | sort | uniq -c`).
  A 3-D expert bank leads with that count: `qwen3_next` `gate_up_proj` (E, 2f, h), `down_proj` (E, h, f)
  (`modeling_qwen3_next.py:787-788`).
- Corrected rule: a parameter counts only when its rank is ≥ 3 and its leading dimension equals the config's expert
  count under one of those four keys. (C, 1, k) and (1, 1, h) never match (a config with none of the keys has no
  count). The name and `ModuleList` rules are unchanged. A misclassified MoE that slips through still runs HF's own
  block in both the chain and the vanilla comparator, so it is gated like any dense module, never silently different.
- The 60 stays a HYPOTHESIS until E7's census runs with D8 (round 2 §4 counted by class names).

## 2. Optional imports switch algorithms (D5)

- CODE: `modeling_mamba.py:195-198` imports `mambapy.pscan` when `is_mambapy_available()`; `:217-218` uses the
  parallel scan when `use_mambapy` and it imported, else the sequential loop: another reduction order.
- CODE: `utils/import_utils.py:73` `_is_package_available(pkg_name)`; 141 `def is_*_available` functions; `kernels`,
  `flash_attn`, `fla`, `mamba_ssm`, `causal_conv1d`, `liger_kernel` and `mambapy` are each probed through it by name
  (`grep -c '"<name>"' import_utils.py` ≥ 1 for each); `unsloth` is not probed (0).
- The per-node gate cannot see this: vanilla and chain take the same branch on one node. Two nodes with and without
  the package compute two functions.
- Corrected refusal: a distribution that one of the pinned transformers' `_is_package_available` calls names, installed
  but absent from the `hf` extra's locked set, refuses the node; the names are listed from the pinned source by the
  pin-bump PR (`hfm.PROBED`, as `hfm.TORCH_RECORDS` is). `unsloth` stays refused by name (it patches, unprobed).
- Not covered (R18, open): a locked distribution at another version (only transformers and torch are fingerprinted),
  and an optional import that bypasses the probes. The general fix is an import-closure fingerprint (a follow-up).

## 3. The bytecode prefix must hold before torch's first import (D5, D9)

- CODE (this repo): `src/wantrain/model/hf.py:17` imports torch at module top; the worker starts a stage with
  `subprocess.Popen(cmd, env=env, ...)` (`src/wantrain/worker/agent.py:92`), so it controls argv and environment.
- MEASURED: `.venv/bin/python -I -X pycache_prefix=<fresh> -c "import sys, types, torch; ..."`: `sys.pycache_prefix`
  is the fresh directory; 726 torch modules loaded, 0 with `__cached__` outside it (`-X` holds under `-I`, which
  ignores `PYTHON*` variables).
- MEASURED: the same with `import torch` first and `sys.pycache_prefix` assigned after: 726 of 726 torch modules have
  `__cached__` in the tree (`torch/_strobelight/__pycache__/__init__.cpython-312.pyc`). Round 4's in-process
  assignment therefore left torch's in-tree `.pyc` live, unverified by RECORD (exempt) and by the code-object check
  (transformers only).
- Corrected text: the worker passes `-X pycache_prefix=<fresh per-launch directory>` on the stage and buffer-builder
  command line; the guard asserts `sys.pycache_prefix` is it and every loaded torch or transformers module's
  `__cached__` lies under it. Cost: 0 per step; the compile time per launch already counted (1–4 s, HYPOTHESIS, E6).

## 4. A0 and A4 on the binding term (Alternatives)

DERIVED from DESIGN §9 (7B, h 4096, T_mb 2048, 8 H100 stages, F+B+W ≈ 33 ms per microbatch):
- HF fp32 cut: 2048·4096·3.40 B = 28.5 MB per hop per microbatch; at 1 Gbps (125 MB/s) 228 ms.
- Native bf16 cut: 2048·4096·1.40 B = 11.7 MB; 94 ms. Ratio ×2.43.
- Both exceed 33 ms, so the hop paces the step: ≈ ×2.43 step time on that hop. Busy-GPU bandwidth per direction:
  7.0 Gbps (HF) vs ≈ 2.9 Gbps (native). Between ≈ 2.9 and 7 Gbps the native tier keeps the GPU busy and HF does not.
- So A0/A4 win on the WAN term. They are rejected (A0) and deferred (A4) for another reason: a native family computes
  another function than HF's (per-block bf16 narrowing, D10; Evidence 1: max|Δlogit| up to 0.296), so each needs its
  own port, oracle and gate per bump (`T_support`). Decision 3's precision ADR (HF's residual bf16 under bf16
  parameters) removes the ×2.43 for every family at once; a native tier is reopened only if it still pays then.
- The host-time numbers (477 vs 480 µs) are one contended sample each (`out/overhead-rerun-r1-contended.txt`) and
  did not reproduce round 0's 858 vs 624 µs: they do not discriminate and decide nothing.

## 5. Rollback by commit set (Migration Impact)

| Workstream | Pieces (ADR-0009) | Reachable only through | Rollback |
|---|---|---|---|
| model | 4a: `arch="hf"` refused, `hf.load_into` and `_LLAMA_*` deleted | — | kept (A0′) |
| model | 4b: `hfm` (guard, skeleton, trace, units, load/export, fingerprint, gate) | `arch="hf"` | revert |
| runtime | task 7: row-sparse tied fold, `TIED` relay, buffer record in JobSpec/state/waves, HF oracle | `arch="hf"` | revert |
| transport | the `TIED` frame type | a plan with a tied relay | revert |
| control | task 6: buffer-record builder, CONTROL-I24 admission | `arch="hf"` | revert |
| scheduler | task 6: SCHEDULER-I22's `ΔT_tied` charge and width refusal | `arch="hf"` with a tied table | revert |

The set is named by task id in the run ledger (`.factory/runs/M4/`), not by a point in history, so unrelated M4 work
landed after 4a stays. Between the revert of one piece and the next, an unexpected `TIED` frame (a plan with no tied
relay) is fatal at receipt (RUNTIME-I19). Rolling back also withdraws ADR-0008's HF-checkpoint MoE path, built only
under ADR-0008 decision 6a; Mixtral and Qwen2/3-MoE checkpoints have no path until 6b's allowlist loader is built.

## 6. Heals keep the device class (decision 8)

CONTROL-I24 refused a map no node holds "before placement" only. A heal admitted a replacement on gate, fingerprint
and torch RECORD; `find_replacement` falls back from the identical model to the region (SPEC-control §2.3), and
RUNTIME-I15 admits a same-type move between GPU models. So an H100 slot healed onto an A100 with the same wheel gave
{H100, A100}, a map with no oracle. Corrected text: the run records each slot's device class (type, model, a CPU's
ATen capability, as its gate reported); for `arch="hf"` `find_replacement` admits only that class (`match="model"`
extended to the class), and without one the heal fails cleanly with the reason. Cost: 0 per step; availability:
a heal needs a spare of the slot's class.

## 7. Native code outside the RECORDs (D9, R18)

- On Linux CUDA wheels, cuBLAS, cuDNN and NCCL ship in separate `nvidia-*` distributions, outside torch's RECORD; a
  loader variable (`LD_PRELOAD`, `LD_LIBRARY_PATH`, `DYLD_*`) can substitute any shared object.
- Corrected text (same RECORD check, wider set): the fingerprint covers every distribution torch's wheel requires in
  `uv.lock`, each against its RECORD, the set's sha256s in `hfm.TORCH_RECORDS` per (version, platform tag); the
  worker starts the process without those loader variables and the guard refuses them.
- Open (R18): OS and driver libraries (libm, Accelerate, glibc, the CUDA driver) are in no RECORD, and no check ties a
  loaded shared object to a verified file. A class's identity there rests on the per-node gate (tiny config) and, on a
  one-class map, the vanilla oracle. Closing it is a mechanism: a shared-object fingerprint from the loader's list, or
  a sampled cross-node recompute audit.

## 8. `ΔT_tied`, the formula of record (moved verbatim from the ADR's Mathematical Model, round 5)

`a = T_mb·h·4`, `e_i = |T_i|·h·4`, `t_s` the per-microbatch stage time (the tied head GEMM included):

`ΔT_tied = M·max(0, (a+ē)/B − max(t_s, a/B)) + Σ_{k=1}^{S−1} max(0, |R_k|·h·4/B − t_F) + 2(S−1)L
+ (|T_{M−1}|+|T_0|)·h·4/B + t_AdamW(table) + M·V·h·4/BW_mem + max(0, Σ_{k≥1}|R_k|·h·4/B − W_idle)`

Terms: throughput, warmup, round trip (`2(S−1)L`, the edge rows `T_{M−1}` and `T_0`, the table's AdamW), the +0 scan,
the row burst past the backward link's idle window `W_idle ≈ 2(S−1)L + S·t_F + (S−1)·a/B + t_B` (`round4.md` §1).
F4 compares each E4 arm's exposed time with this formula evaluated from the arm's measured `t_s`, `t_F`, `t_B`,
`|T_i|` and `|R_k|`.
