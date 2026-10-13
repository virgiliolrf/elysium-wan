# HANDOUT M4: any dense HF decoder and exact MoE, in one run

This plan builds any dense HF decoder through transformers' own modules, with its parity gate. It also builds exact MoE,
native and HF class-swapped, proves pipeline identity for both, and makes the determinism fix.

Machine plan: [PLAN-M4.json](PLAN-M4.json): 18 tasks in 2 waves, binding audit BA-401..BA-468, tripwire `m4-unarmed`.
- Design of record: [DESIGN.md](DESIGN.md) §3, §5, §8 (M4 row).
- Decisions: [ADR-0009](adr/ADR-0009-any-decoder.md) (any decoder) and [ADR-0008](adr/ADR-0008-moe-exact.md) (exact
  MoE), as reconciled on 2026-10-11 (commit 3b0d2f8). Their decisions were taken by autonomy.
- Specs:
  - SPEC-model tasks 3 (MoE) and 4 (4a, 4b);
  - SPEC-runtime tasks 6 (MoE) and 7;
  - SPEC-scheduler tasks 5 and 6;
  - SPEC-control task 6;
  - SPEC-transport §2 (`TIED`).

Revision 2 (after review round 1): MoE is planned in this DAG, because the plan does not pass the 24-task slicing bar.
- Wave-1 tasks test only their own code.
- Cross-task proofs moved to wave-2 integration tasks.
- Seven binding items were added.

Revision 3 (after review round 2):
- Wave-1 tasks no longer reach a sibling's symbol on a dense path (BA-407) or in a test (`hf_admission` takes its hfm
  inputs as arguments).
- Every wave-1 file has a named wave-2 fix owner, and no wave-2 fast verify depends on a fix a concurrent sibling may
  make.
- No wave-2 task waits on `hf-pin`.
- New task `wave-files` lets a wave list `hf_buffers.json` (BA-460).
- Plan-mode jobs keep their hf and MoE keys (BA-461).
- MoE records under the HF swap are specified (BA-462), and Linux has a stated disposition (BA-463, D-9).
- E7 has writers, the missing E2/E4/E6 controls are in, and the ZeRO-1 count transport is ruled per gradient path.

Revision 4 (final check of every round-1 and round-2 finding against both files):
- `fingerprint` and `gate` take SPEC-model §3's signatures (I-2).
- An hf plan's piece shapes come from the census, so `finalize` never imports transformers in the coordinator or
  planner (I-10, BA-419).
- Dense pins and digests omit the new fields (BA-465).
- The oracle runs the stage's start order (BA-406), and the hfm CLI children run under `setup_determinism` (BA-432).
- The HF-MoE rows cover all nine classes (BA-438).
- `wantrain local` and `wantrain stage` sit under the tripwire (BA-464, I-16).
- GATE runs off the agent's reader thread.
- No `TIED` is sent past the attempt's last step.
- The import, lock and dense checks fail whatever the environment holds.

Revision 5 (the final check's 11 open findings, each re-read against the current files; dispositions at the end):
- The HF and MoE arms reach the `oracle` and `identity` rungs (BA-468): `identity-corpus` writes them under
  `tests/helpers/` with the three-mode protocol, and `corpus-wiring` adds them to `identity.json` and the oracle rung.
- Every rollback piece has its own revertible unit, the `model/rollback.py` table (BA-467, D-3).
- The planner's width-1 early return and its `unpriced` refusal are ruled (BA-446, BA-466, D-10).
- SCHEDULER-I21's bank term, its `c_s` rationale and the hf LADDER clause have owners and tests.
- Each task's coverage check reads only its own tests (D-5).
- identity-m4's multi-process items are GATES entries whose collect checks require every arm id. stage-wire tests
  the middle stage's `TIED` forwarding.
- transport-tied runs the fuzz targets; hf-pin checks the hash of every locked file.
- The one-owner-per-file-per-wave reading is recorded honestly (D-11).

The review notes at the end give each finding's disposition.

## Goal

An operator gives a HuggingFace decoder config (or a checkpoint) of any in-scope family, dense or one of the MoE classes
with a class-swap row. The run trains it through transformers' own modules: each pipeline unit is a window of HF's
`ForCausalLM.forward`, and each sparse-MoE block is swapped to ADR-0008's exact layer. The run's bytes are identical to
the single-device oracle, or the run is refused before it starts, naming the cause. A native MoE recipe (`moe_experts >
0`) is byte-identical too, under every existing mechanism, M2b's pieces included. No family-specific code is written.

| Invariant | What it says (short) | Delivered by |
|---|---|---|
| MODEL-I4 (HF clause) | an HF checkpoint lands 1:1 by HF names, or loading is refused naming the tensor; the hub-to-fused expert map round-trips | `model-core` |
| MODEL-I8 | the MoE block is one exact function on every path (selection, folds, +0 for unrouted experts, no RNG, bias rule); the HF swap equals it (F8) | `model-core` (native), `model-families` (HF rows) |
| MODEL-I9 | HF units compose to the vanilla HF model bit for bit | `model-core` (llama, qwen3, gemma2, granite), `model-families` (the 12) |
| MODEL-I10 | no `arch="hf"` run starts without its launch gate and fingerprint | `model-core`, `model-families` |
| RUNTIME-I18 | MoE byte-identical: state, CE hex, aux hex; bias update by counts under width 1, ZeRO-1 and pieces | `stage-m4` (code), `stage-wire`, `identity-m4`, `identity-corpus` (proof) |
| RUNTIME-I19 | HF units and tied tables byte-identical; buffer record in every wave; determinism asserts | `determinism-pins` (house item), `stage-m4`, `stage-wire`, `identity-m4`, `identity-corpus` |
| RUNTIME-I15 (hf clause) | a resume onto another device class is refused, naming the slot | `stage-m4`, `stage-wire` |
| SCHEDULER-I21 | MoE stages: compute at active parameters, bytes and memory at total | `scheduler-m4`, `plan-wire` |
| SCHEDULER-I22 | HF plans: the fp32 cut, the tied bytes and `ΔT_tied`, the tied memory, eager attention | `scheduler-m4`, `plan-wire` |
| CONTROL-I24 | an hf run trains only on gated, fingerprint-admitted nodes; decision 8 and the RECORD check at every admission | `control-m4`, `agent-hf`, `agent-wire`, `arm-m4` |

Every task that writes tests runs an own-test coverage check in its verify. The check parses the task's own test file
with `ast` and selects the tests whose names match the task's prefix (`test_m4_stage_`, `test_hfm_`, ...). It exits 1
unless each id the task delivers has a `covers:` line in one of those tests, and every one of them has a
`WOULD-FAIL-IF:` line. It reads nothing else, so it fails on the task's own omission, whatever other files cover. The
global `invariants.mjs --json` read counts any `covers:` anywhere, so it is kept only as GATES-M4's milestone-level
check (D-5).

**The evidence for the human's four items:**

1. **The HF adapter and the MANDATORY per-family parity gate** (logits and gradients vs vanilla transformers; the run is
   refused on divergence).
   - `model-core` builds the mechanism and proves llama, qwen3, gemma2 and granite.
   - `model-families` proves the 12 families: llama, mistral, qwen2, qwen3, gemma, gemma2, gemma3 (`gemma3_text`),
     phi3, olmo2, granite, gpt-neox, starcoder2. It also adds E1's extras, the E2 controls and the E6 drift arms.
   - Every model is a tiny, randomly initialized config of its transformers config class. There is no `from_pretrained`,
     no network and no download.
   - ADR-0009 E7 (`saved_bytes`, `c_glue`, `c_pass`, stub allocator bytes, the A0″ comparison) is `model-families`'
     `test_hf_e7_*`; its F7 bounds are a GATES entry.
2. **Exact MoE** (ADR-0008).
   - `model-core`: the native block and E1.
   - `model-families`: the HF class-swap rows for Mixtral, Qwen2-MoE and Qwen3-MoE, with F8 parity against the
     transformers 5.19.0 blocks, the 2-span capture path over the HF layer list, and E7a/E7b (`test_moe_e7_*`).
   - The tiny DeepSeek-V3: its native recipe row is exact against its oracle, and its router's selected sets equal
     transformers' on tie-free keys. This is fidelity, not identity (ADR-0008 decision 9). `arch="hf"` `deepseek_v3`
     stays refused (MLA, no row).
3. **Pipeline byte-exactness against the single-device reference.** `identity-m4` compares the sha256 of weights,
   AdamW state, RNG and selection biases, plus every CE and aux hex.
   - HF: llama, qwen3, gemma2 (tied, split) and granite in the fast arms; all 12 families and the HF-MoE adapter arms
     in GATES-M4.
   - The `oracle` and `identity` rungs (SPEC-runtime task 7): `identity-corpus` runs tiny llama, qwen3, gemma2 tied,
     granite, Mixtral and native MoE in three modes: reference, features off, and production under emulated WAN.
     The identity rung compares them by sha256 (BA-468).
   - MoE: depths 1, 2 and 4, ZeRO-1 widths 1 and 2, M2b pieces widths 1, 2 and 3, skew, and the must-fails.
   - The coordinator path is `arm-m4`.
4. **The house fix of `runtime/determinism.py`** (SPEC-runtime task 7's item, `covers: RUNTIME-I19`). It is its own task,
   `determinism-pins`, with its own tests.

## Scope: one DAG, no slices

The human's rule slices the work only if the plan passes 24 tasks. This plan has 18, so dense and MoE are one DAG run
by one `wan-implement`, and there are no later slices.

The 2-wave rule holds, and within each wave every file has exactly one owner. That is achieved by merging the HF and
MoE pieces that write the same file into one task:
- `model-core`: `gpt.py`, `arch.py`, `wgrad.py`, `hfm.py`, `rollback.py`;
- `stage-m4`: the stage files;
- `oracle-m4`: the oracle files;
- `scheduler-m4`: the planner files.

The human's "one task per file" is NOT met as written: 35 files have one wave-1 owner and one wave-2 owner. D-11
records this and lists them. The scarcity of test files forces the same merges (next section).

**Not in this run.** These M4 items are outside the human's scope and stay `DEFERRED-TO: M4` in their specs:
- the ADR-0007 slice-only build (MODEL-I6, RUNTIME-I14, SCHEDULER-I18, CONTROL-I19), except ADR-0009 D3's
  checkpoint-backed HF span build (BA-449);
- audit wiring (CONTROL-I13), coordinator standby (CONTROL-I14) and shard replication (CONTROL-I15).

From-config HF runs keep the full build per slot (ADR-0009 decision 7).

## Plan-rule conflict (recorded; a factory "house" finding, not worked around)

The PLAN RULES ask that every task put its tests in its OWN new test file. The machine check requires every scope path to
lie inside the workstream's owned paths (DESIGN §7), and those name a fixed set of existing test files:
- model: 1 file;
- runtime: 3 files;
- scheduler: 3 files;
- transport: 2 files;
- control: 6 files (`test_fault_tolerance.py` among them, `wave-files`' test file).

The machine check enforces only the second rule. So each task's tests live in one owned test file that no concurrent task
touches, under a name prefix its verify selects with `-k`. A file has a second owner only in a wave-2 task that depends
on the first. The model workstream's single test file is why `model-core` holds both hfm and the MoE block: two wave-1
model tasks would share `test_finetune.py`. The factory fix is to own `tests/unit/test_<task-id>.py` per workstream; it
goes to the factory backlog.

## Starting point (read on this tree, 2026-10-11)

- **Code base.** `factory/m4` at 3b0d2f8. **M2b is merged into `factory/m4` as a SNAPSHOT of `origin/factory/m2b`
  8dee0b9** (commit e7d4585 "m4: merge-m2b"), as M2b did with M2. M2b's closeout verdict is still pending in another
  session. If M2b's closeout changes code, `factory/m4` merges that change before M4's closeout, and this plan's
  `stage.py`, `pieces.py`, `exchange.py` and `logical_ckpt.py` hunks are re-checked against it. Every BA `where` cites
  this snapshot.
- **None of ADR-0008 or ADR-0009 is built.**
  - There is no `model/hfm.py`, and nothing under `src/` mentions MoE.
  - `model/hf.py` is still the native loader. Its only users are `scheduler/fleetplan.py:88-97` (a local import inside
    the `arch == "hf"` branch) and `tests/unit/test_finetune.py`. No scheduler test exercises that fleetplan branch
    (grep), so deleting `config_from_hf` in wave 1 breaks no test before `plan-wire` lands.
- **transformers is not installed** in this checkout's `.venv` (`ls site-packages`: absent). `pyproject.toml`'s `hf`
  extra holds `safetensors>=0.4` only.
  - MEASURED on this Mac (uv 0.11.14, 2026-10-11): `uv run --no-sync --offline --with transformers==5.19.0` builds an
    overlay from the uv cache. It installs 27 packages in 160 ms and imports transformers 5.19.0 with torch 2.14.1.
  - The overlay leaves `.venv` untouched. No probed package is importable in it.
  - The packages it shadows have the project's versions: numpy 2.5.3, safetensors 0.8.0 and filelock 4.0.12 checked.
  - That is the **overlay command** every HF test uses (see Wave order).
- **transformers' RECORD** is 292,932 B and 2,726 lines (125 KB gzip). MEASURED from the uv cache's
  `transformers-5.19.0.dist-info/RECORD`. This sets I-3's RECORD text size.
- **Research prototypes** (`docs/research/adr-0009/`, `docs/research/adr-0008/`; MEASURED on a62e4eb):
  - unit windows: 22/22 arms exact;
  - census: 102 EXACT, 0 DIFFER;
  - MoE: 39/39 pipeline and 16/16 piece arms.

  The product code is new. These numbers set expectations; they are not evidence for the new code.
- **Disk:** 11 GB free (`df -g .`, 2026-10-11).

## Preconditions (checked before wave 1)

| Id | Precondition | Check |
|---|---|---|
| P-1 | DESIGN.md, ADR-0008, ADR-0009, and this HANDOUT, PLAN and GATES are approved. With `autonomy.json` `humanGrants: false`, `approve.mjs check` reports them `waived`. | `node agents-architetura/tools/factory/approve.mjs check docs/design/adr/ADR-0009-any-decoder.md` |
| P-2 | transformers 5.19.0 and its dependencies are in the uv cache (the overlay resolved them offline, MEASURED above). | `uv run --no-sync --offline --with transformers==5.19.0 python -c "import transformers"` |
| P-3 | At least 5 GB of free disk before wave 1. | `df -g . \| tail -1` |
| P-4 | The heavy M2b gate shares this 16 GB Mac. Every test command takes a `machine-lock.mjs run --slot` (timeout 1200 s), and no task runs a whole-machine command. | the verify lines of PLAN-M4.json |

## What this run moves in the cost model

The step time is `T_step = M·(t_F + t_B + t_W)_max + T_bubble + T_opt + T_exposed`, plus a launch overhead.

| Term | Change | Label |
|---|---|---|
| `T_comm` per hop (HF) | The cut is HF's residual dtype: 4 B per element (3.40 B after the lossless codec), against 2 B (1.40 B) natively. That is ×2.43 on a bandwidth-bound hop: 228 vs 94 ms per microbatch at 7B, T_mb 2048, 1 Gbps. The two are equal at ≥ 7 Gbps. | DERIVED (ADR-0009 `round6.md` §3) |
| host time per HF unit | `c_glue + n_before(u)·c_pass`: 2.7–5.5% of an 80-layer last stage, ×2 with recompute | HYPOTHESIS until E7 (GATES) |
| `T_exposed`, tied table | `ΔT_tied`, five terms (`round5.md` §8): 197–414 ms against a 438 ms step for Gemma2-2B (S 4, M 8, 20 ms, 10 Gbps); 0 when both units share a stage | DERIVED; E4 arms SIMULATED |
| activation memory (HF) | eager attention holds `b·H·S²·s` bytes per layer (s = 4 / 6 / +2 or +4 with a softcap) | HYPOTHESIS |
| MoE compute vs bytes | compute at active parameters (`c_s = 8·P^F_s·T_mb/F`), bytes and memory at total (ρ 3.68–10.95). S1 needs `T_mb ≥ ρ_s × 1,500` at 400 Gbps, w 4. | DERIVED (ADR-0008) |
| MoE count lane | `L_m·E·8` B per microbatch per peer: ≤ 184,320 B at the largest row, 256 B in E2 | DERIVED |
| HF MoE banks | dW in B (`t_B` ×1.40–1.89); bank gradients `b_m·E·3hf` held B→W, × `m_BW`: 5.25 GiB per Mixtral layer and microbatch | DERIVED; measured in `model-families` (F3) |
| launch overhead (HF) | import 1.2–2.3 s, RECORD check 3.2 s, fresh-prefix compile 1–4 s, gate 1–2 s per device: 0% of step time, but the dominant cost of every hf test process on this Mac | partly MEASURED |
| determinism fix | none per step | DERIVED |

**When the gain vanishes.** The ×2.43 boundary cost is the price of computing HF's exact function. It goes away only with
a bf16-parameter recipe, which is decision 3's precision ADR and not part of M4. Nothing in this run trades a bit for
speed.

## Tasks

| Id | Workstream | Wave | Files (scope) | Delivers | Risk |
|---|---|---|---|---|---|
| `hf-pin` | integrate | 1 | `pyproject.toml`, `uv.lock` | the transformers pin with hashes; additive install (BA-440) | high |
| `corpus-wiring` | integrate | 1 | `agents-architetura/tools/verify/identity.json`, `agents-architetura/tools/gates/rungs.json` (HOT_FILES) | the identity corpus glob and the oracle rung's `tests/helpers` step (BA-468) | medium |
| `determinism-pins` | runtime | 1 | `runtime/determinism.py`, `tests/unit/test_user_model_precision.py` | the house fix | high |
| `model-core` | model | 1 | `model/hfm.py` (new), `rollback.py` (new), `hf.py`, `gpt.py`, `wgrad.py`, `arch.py`, `tests/unit/test_finetune.py` | hfm D1–D9, 4a + 4b; native MoE block (MODEL-I8); the rollback table (BA-467) | high |
| `stage-m4` | runtime | 1 | `runtime/stage.py`, `accumulate.py`, `spec.py`, `pieces.py`, `exchange.py`, `logical_ckpt.py`, `worker/stage_main.py`, `tests/unit/test_runtime_spec.py` | HF stage, tied relay, buffer waves; MoE aux replay, bias update, count transport; one-slot hf refusal | high |
| `oracle-m4` | runtime | 1 | `validation/reference.py`, `validation/state.py`, `launch/local.py`, `tests/integration/test_engine_bitwise.py` | HF and MoE oracle, manifest sections, local launcher | high |
| `transport-tied` | transport | 1 | `transport/framing.py`, `tests/unit/test_transport.py` | `TIED`, typed decode | high |
| `scheduler-m4` | scheduler | 1 | `scheduler/planner.py`, `autoconfig.py`, `zero_bubble.py`, `partition.py`, `admission.py`, `exchange_sim.py`, `tests/unit/test_scheduler.py`, `tests/unit/test_zero_bubble.py` | SCHEDULER-I21, I22 as pure cost functions | medium |
| `agent-hf` | control | 1 | `worker/agent.py`, `resources/profile.py`, `resources/fit.py`, `cli.py`, `tests/unit/test_resources.py` | GATE in the agent; HF children; `wantrain stage` hf refusal | high |
| `control-m4` | control | 1 | `coordinator/coordinator.py`, `fault_tolerance/replacement.py`, `tests/unit/test_control_admission.py` | CONTROL-I24 admission; plan-mode pins and `plan_model_mismatch`; first leg of `m4-unarmed` | high |
| `wave-files` | control | 1 | `fault_tolerance/checkpoint.py`, `tests/unit/test_fault_tolerance.py` | `commit_wave(..., extra_files)` (I-15, BA-460) | high |
| `model-families` | model | 2 | `model/hfm.py`, `rollback.py`, `arch.py`, `gpt.py`, `wgrad.py`, `hf.py`, `tests/unit/test_finetune.py` | the 12-family gate, E2, E6, E7, census; HF-MoE swap rows, recipes, records, F8, E7a/b, DeepSeek-V3; rollback of the HF rows | high |
| `stage-wire` | runtime | 2 | `runtime/stage.py`, `spec.py`, `accumulate.py`, `pieces.py`, `exchange.py`, `logical_ckpt.py`, `determinism.py`, `worker/stage_main.py`, `tests/unit/test_runtime_spec.py` | stage behaviour proven on real units, blocks and frames; middle-stage `TIED` forwarding; the tied-split rollback target | high |
| `identity-m4` | runtime | 2 | `validation/reference.py`, `validation/state.py`, `launch/local.py`, `tests/integration/test_engine_bitwise.py` | HF and MoE pipeline identity by sha256 | high |
| `identity-corpus` | control | 2 | `tests/helpers/test_m4_identity_hf.py`, `tests/helpers/test_m4_identity_moe.py` (new) | the HF and MoE arms of the `oracle` and `identity` rungs (BA-468) | high |
| `plan-wire` | scheduler | 2 | `scheduler/fleetplan.py`, `planner.py`, `autoconfig.py`, `zero_bubble.py`, `partition.py`, `admission.py`, `exchange_sim.py`, `tests/unit/test_schedule.py` | GPTConfig hf and MoE plans; plan-mode keys and pins (I-14); MoE units equal the runtime's; `c_s` rationale; the planner's rollback target | medium |
| `agent-wire` | control | 2 | `worker/agent.py`, `resources/profile.py`, `resources/fit.py`, `cli.py`, `tests/unit/test_resources.py` | GATE end to end; hf and MoE PROFILE/FIT | high |
| `arm-m4` | control | 2 | `coordinator/coordinator.py`, `fault_tolerance/replacement.py`, `fault_tolerance/checkpoint.py`, `tests/unit/test_control_admission.py`, `tests/integration/test_coordinator.py` | arms `m4-unarmed`; coordinator end to end (spec and plan mode), heal, resume | high |

`model-core` (about 2.5k lines) and `stage-m4` are the largest tasks. Each is one task because one test file serves its
workstream in wave 1, and the one-task-per-file rule allows a big task but not a chain. Their acceptance is itemized per
ADR decision.

## Wave order and the isolation rule

**Wave 1 (11 tasks, in parallel):** `hf-pin`, `corpus-wiring`, `determinism-pins`, `model-core`, `stage-m4`,
`oracle-m4`, `transport-tied`, `scheduler-m4`, `agent-hf`, `control-m4`, `wave-files`.
- **Isolation.** A wave-1 task's tests exercise only its own files, code already on the tree, and transformers through
  the overlay command. A sibling's new symbol (Interfaces below) is imported only inside the arch-'hf' or MoE code path,
  at call time. So every module imports, and every existing test passes, before any sibling lands. Code that calls a
  sibling's interface is proven by the wave-2 task named for it, never by a stub, a skip or an xfail.
  - Dense, builtin, general, user and native-MoE call sites keep `layer(x)` byte for byte. Only the arch-`"hf"` branch
    calls `gpt.call_layer`, imported at call time (BA-407). A dense test never reaches `model-core`'s new symbol.
- **Pure functions carry the wave-1 tests.** Those that need data from a sibling take it as explicit arguments:
  - `scheduler-m4`'s cost functions over a census fixture and MoE dims;
  - `control-m4`'s `hf_admission` over crafted reports and RECORD texts, with `torch_records` and `record_diff` passed as
    arguments (I-9), and `plan_model_mismatch` over crafted plan dicts;
  - `agent-hf`'s `validate_gate_request` and `run_child`;
  - `oracle-m4`'s `oracle_mode` and its manifest-section builders;
  - `stage-m4`'s fold, count-lane, count-fold and replay comparators.
- **The overlay command.** Every HF test command, in both waves, is `uv run --no-sync --offline --with
  transformers==5.19.0 pytest ...`. Every other test command is `uv run --no-sync --offline pytest ...`.
  - `--no-sync` means no test command ever re-syncs `.venv` while siblings import from it.
  - The overlay makes no task's tests wait for `hf-pin`.
  - `hf-pin`'s own `uv sync --inexact` is the only writer of `.venv` in the run. It is additive: no locked package moves,
    which is its acceptance. A concurrent reader therefore sees either the old set of packages or the old set plus the
    new ones.
  - The ladder after integration runs on the locked `.venv`.

**Wave 2 (7 tasks, in parallel, each depending only on wave-1 tasks):**

| Task | Depends on | Proves |
|---|---|---|
| `model-families` | `model-core` | the 12 families, E2, E6, E7, census; HF-MoE swap rows (F8), recipes and records over unit spans, E7a/b, DeepSeek-V3 |
| `stage-wire` | `stage-m4`, `model-core`, `transport-tied`, `determinism-pins`, `wave-files` | start asserts, TIED receipt, buffer and code pins, hf JobSpec rules, span build, MoE replay and the three ZeRO-1 count paths on real blocks |
| `identity-m4` | `model-core`, `stage-m4`, `oracle-m4`, `transport-tied`, `scheduler-m4`, `determinism-pins`, `wave-files` | HF and MoE oracle (fast); HF and MoE pipeline identity, checkpoint arm, resume, E2/E3/E4/E6, F7 (post-wave ladder and GATES) |
| `identity-corpus` | `model-core`, `stage-m4`, `oracle-m4`, `transport-tied`, `determinism-pins`, `wave-files` | the HF and MoE identity-corpus arms (fast: collection and the own-test check; the arms run in the post-wave `oracle` and `identity` rungs and in GATES-M4) |
| `plan-wire` | `scheduler-m4`, `model-core`, `stage-m4` | plans of GPTConfig hf and MoE jobs; plan-mode keys and pins (I-14); no transformers in the planner; the planner's MoE units equal `pieces.exchange_units` (BA-466) |
| `agent-wire` | `agent-hf`, `model-core`, `stage-m4`, `determinism-pins` | the HF environment, GATE end to end, hf and MoE PROFILE/FIT |
| `arm-m4` | every wave-1 task but `hf-pin` | lifts `m4-unarmed` (fast: the flag and its tests); coordinator end to end in spec and plan mode, heal, resume under another RECORD (post-wave ladder and GATES) |

No wave-2 task depends on `hf-pin`. Every wave-2 test command runs on the overlay, and only the ladder after the waves
needs the locked `.venv`. If `hf-pin` STOPS (R-4), wave 2 still runs, and the stop is a recorded conflict for the
integrate step. `model-families` does not depend on `determinism-pins`: its gate children call the existing
`setup_determinism`, which takes the fix whenever it lands.

**Who fixes what in wave 2.** Each wave-1 file has exactly one wave-2 fix owner. That owner edits the file only to fix
a defect a test exposes, and keeps the frozen interfaces.

| Wave-1 files | Wave-2 fix owner |
|---|---|
| `model/hfm.py`, `rollback.py`, `arch.py`, `gpt.py`, `wgrad.py`, `hf.py` | `model-families` |
| `runtime/stage.py`, `spec.py`, `accumulate.py`, `pieces.py`, `exchange.py`, `logical_ckpt.py`, `determinism.py`, `worker/stage_main.py` | `stage-wire` |
| `validation/reference.py`, `validation/state.py`, `launch/local.py` | `identity-m4` |
| `scheduler/planner.py`, `autoconfig.py`, `zero_bubble.py`, `partition.py`, `admission.py`, `exchange_sim.py` (and `fleetplan.py`) | `plan-wire` |
| `worker/agent.py`, `resources/profile.py`, `resources/fit.py`, `cli.py` | `agent-wire` |
| `coordinator/coordinator.py`, `fault_tolerance/replacement.py`, `fault_tolerance/checkpoint.py` | `arm-m4` |
| `transport/framing.py` | none in wave 2. Its wave-1 tests are self-contained framing tests. A later defect is fixed by the post-wave whole-change review or the fail-fast ladder, whose fixers may edit any file. |
| `pyproject.toml`, `uv.lock`, `identity.json`, `rungs.json` | the integrate step |
| `tests/helpers/test_m4_identity_*.py` (wave 2 only) | `identity-corpus` is their only owner; a defect found after the wave goes to the post-wave fixers |

A defect in a file another task owns is recorded as a spec conflict naming that task, never worked around. The post-wave
review and the fail-fast ladder then fix it: wan-implement's whole-change reviewers and the `wan-fix` agent edit any
file.

**Within wave 2: what a fast verify may cite.** A wave-2 fast verify cites only tests whose code paths run through the
task's own files and through its dependencies' files. It never cites a test that would need a fix a concurrent wave-2
task may make.
- **In fast verifies:**
  - `stage-wire`: in-process, its own files plus `hfm` on the families `model-core` proved;
  - `agent-wire`: the GATE child on llama;
  - `plan-wire`;
  - `model-families`;
  - the single-process oracle arms of `identity-m4`, and the collection of its arms;
  - `arm-m4`'s flag tests;
  - `identity-corpus`: collection and the own-test check only.
- **Written in wave 2, run after it:** every multi-process arm runs through `stage.py` and `spec.py` (fix owner
  `stage-wire`), and the coordinator arms also through `agent.py`, `profile.py` and `fit.py` (`agent-wire`) and
  `fleetplan.py` (`plan-wire`). This covers every `hf_identity`, `moe_identity`, E4, `m4_coordinator` and
  `m4_identity` (corpus) arm.
  - The coder runs one of them once as its last command and records the result in its journal. That is evidence for its
    tree only, naming the owner of any failing file.
  - The ladder's `integration` rung runs `tests/integration` whole (`rungs.json`: no marker deselection). The `oracle`
    and `identity` rungs run the corpus files once `corpus-wiring` has landed. So every arm runs in the post-wave
    fail-fast ladder (at most 8 rounds), whose fixer owns cross-file fixes.
  - GATES-M4 runs them again with repetitions. The collect checks of `hf-identity-x2` and `moe-identity-x2` require
    every parametrization id `identity-m4` names, so an item missing from a fast verify (finding 9) is a gate failure
    if the arm is absent, never a silent gap. The items are: the 3- and 4-stage WAN chains, the gemma2 tied chain3,
    the HF ZeRO-1 middle stage, the HF pieces arm, bf16, E4, and the MoE must-fails.
- **Arming.** `arm-m4` sets `M4_ARMED = True` in wave 2, beside the proofs, so the flag never waits for a third wave.
  What justifies it is the final tree's evidence: the post-wave ladder and GATES-M4 run every end-to-end arm. The PR
  leaves draft only on a VERIFIED closeout, and only the human merges. No unverified tree reaches `main` armed.
- `model-families` edits `hfm.py` and `arch.py` while others import them. Every edit keeps both importable and never
  changes interfaces I-1..I-4, I-10, I-11, I-14 or I-17.

After the waves, wan-implement's integrate, build, review and fail-fast ladder run, then `GATES-M4.json` at closeout.

## Interfaces (frozen for wave 1; a change is a recorded spec conflict, never made)

**I-1 `model/gpt.py`** (owner `model-core`)
```python
@dataclass
class GPTConfig:                      # new fields; every existing field and default unchanged
    hf_config: dict | None = None     # config.json content of the pinned config class ("model_type" selects it)
    hf_buffers: dict | None = None    # I-4, pinned at the first launch
    hf_census: list | None = None     # I-10, from the builder child
    hf_code: dict | None = None       # {"transformers", "transformers_record_sha256", "torch", "torch_git_version"}, pinned at the first launch
    hf_impl: str = "modules"          # the only accepted value for arch "hf"
    moe_experts: int = 0              # ... and every moe_* field of SPEC-model §2, with its default (dense)
def call_layer(layer: nn.Module, x: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor
    # layer(x, ids=ids) when getattr(layer, "takes_ids", False), else layer(x)
    # called only from arch-"hf" branches, imported there at call time (BA-407); dense call sites keep layer(x)
def build_layers(cfg, seed, span: tuple[int, int] | None = None) -> nn.ModuleList
    # arch "hf": hfm.build(cfg, seed, span), then wgrad.convert(..., exclude=units.exclusions); other archs ignore span
```
- `validate()` for arch `"hf"` refuses, naming the field, each of:
  - `hf_config` missing, or without `model_type`;
  - `hf_impl` other than `"modules"`;
  - `lora_rank > 0`;
  - `boundary_dtype` other than `"fp32"`, or fp8 boundaries;
  - `compute_dtype` other than fp32 or bf16;
  - mirrored fields that differ from `hf_census`.
- For MoE it refuses, naming the field:
  - a group configuration outside SPEC-model §2's rule;
  - LoRA with MoE.
- `validate()` never imports transformers.

**I-2 `model/hfm.py`** (owner `model-core`; `model-families` adds SUBSTITUTE rows and fixes in wave 2). transformers is
imported inside functions only.
```python
class HFRefused(ValueError): cause: str
HF_ENV: dict[str, str]  # USE_HUB_KERNELS=0, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, TORCH_DEVICE_BACKEND_AUTOLOAD=0
PROBED: tuple[str, ...]  # kernels, flash_attn, fla, mamba_ssm, causal_conv1d, liger_kernel, mambapy (+ unsloth refused)
SUBSTITUTE: dict[tuple[str, str], type] = {}   # (HF class, "5.19.0") -> swap subclass; rows by model-families
    # a block row's subclass has classmethod recipe(hf_config: dict) -> dict (SPEC-model §2 moe_* fields, gamma 0);
    # model-core builds the swap mechanism (class swap, moe_layer from the unit map, MOE-NO-ROW) with the table empty
def platform_admitted() -> bool                # this platform tag and torch version have a TORCH_RECORDS entry (BA-463)
def stage_env(base: dict) -> dict              # base + HF_ENV, minus LD_PRELOAD, LD_LIBRARY_PATH and every DYLD_*
def interpreter_args(prefix_dir: str) -> list[str]   # ["-X", f"pycache_prefix={prefix_dir}"]
def guard_environment() -> None                # checks, never sets; HFRefused on any violation
def check_config(hf_config: dict, cfg) -> None
def config_fields(hf_config: dict) -> dict     # n_layer, d_model, n_head, vocab_size, tie_embeddings
def tiny_config(model_type: str, **overrides) -> dict
def skeleton(hf_config: dict, dtype=torch.float32) -> "PreTrainedModel"   # params on meta, buffers on CPU, eager, train()
def trace(model) -> "UnitMap"
def units(model, unit_map) -> "HFUnits"
def build(cfg, seed: int, span: tuple[int, int] | None = None) -> "HFUnits"
    # from-config: seeded full build (span ignored); cfg.hf_model: skeleton + strict load of the units in span, rest on meta
def buffer_record(hf_config: dict) -> dict     # I-4
def apply_buffer_record(model, record: dict) -> None   # HFRefused("BUFFER-PIN") naming the tensor
def unit_census(hf_config: dict) -> list[dict] # I-10
def load(model, path: str, units=None) -> "LoadReport"
def export(model, path: str, source_keys) -> None
def write_safetensors(tensors: dict, path: str, shard_bytes: int) -> None
def fingerprint(model=None) -> dict            # I-3; SPEC-model §3's fingerprint(model); None: the environment only
def hf_code() -> dict                          # I-1's hf_code of the installed transformers and torch
def record_text() -> str                       # the installed transformers RECORD
def record_diff(recorded: str, other: str) -> str | None   # the first differing RECORD path, or None
TORCH_RECORDS: dict[tuple[str, str], dict[str, str]]
def torch_records_entry(wheel_path: str) -> dict
def device_class(device: str) -> str           # "cpu:<machine>:<ATen capability>", "mps:<chip>", "cuda:<name>"
def gate(hf_config: dict, recipe: dict, device: str) -> "GateReport"   # SPEC-model §3; recipe = I-9's recipe dict
class HFUnits(nn.ModuleList):                  # n+2 units over one shared tree, root in training mode
    root; unit_map; exclusions: frozenset; tied: tuple[int, int] | None
    def vanilla(self, ids) -> torch.Tensor
# CLI: python -m wantrain.model.hfm {gate|record|census|torch-records} --in IN.json --out OUT.json
```
Cause tokens (the message names the module, tensor, field or file):
- **environment:** `ENV-PREFIX`, `ENV-VAR`, `ENV-LOADER`, `PROBED-PKG`, `UNSLOTH`;
- **config:** `AUTO-MAP`, `ENCODER-DECODER`, `MODEL-TYPE`, `DROPOUT`, `ROPE-DYNAMIC`, `ROPE-LONGROPE`,
  `FOREIGN-FORWARD`, `LORA`, `BOUNDARY-DTYPE`, `ATTN-IMPL`, `MIRROR`, `EVAL-MODE`;
- **build:** `SKELETON-DRAW`, `BUFFER-RECORD`, `BUFFER-PIN`;
- **trace:** `TRACE-UNPLACED`, `TRACE-SHARED`, `TRACE-BETWEEN`, `TRACE-TWICE`, `TRACE-NONFLOAT`, `TRACE-RNG`,
  `MOE-NO-ROW`, `STUB-WRITE`;
- **load:** `LOAD-MISSING`, `LOAD-UNUSED`, `LOAD-SHAPE`, `LOAD-NAME`;
- **gate:** `GATE-DIFF`, `GATE-CAUSAL`, `GATE-TRIGGER`, `GATE-BUDGET`;
- **fingerprint:** `FP-RECORD`, `FP-PYC`, `FP-CODE`, `FP-PROBE`, `FP-NATIVE`, `FP-TORCH-RECORDS`, `FP-RUN-RECORD`.

**I-3 GateReport and fingerprint** (JSON dicts)
- `GateReport`: `{"ok", "cause", "detail", "device", "device_class", "cuts", "dtypes", "ms", "triggers",
  "build_bytes", "build_s", "first_difference", "fingerprint"}`.
- `fingerprint(model=None)`:
  - `transformers`: `{"version", "record_sha256", "record_gz_b64"}`. The RECORD text is about 166 KB in base64, so a
    coordinator can name the differing file.
  - `torch`: `{"version", "git_version", "platform_tag", "records"}`.
  - `probe_vector`, `native`, `python`, `prefix_ok`, `code_ok`.
- `TORCH_RECORDS`: `model-core` computes the macOS arm64 entry from the installed wheel files (MEASURED). See Q-1 and
  D-9 for Linux.
- The gate computes its comparison (`cuts`, `first_difference`, `triggers`) on every platform. A fingerprint refusal
  (`FP-*`) is reported in `cause` without skipping the comparison, so parity evidence exists on Linux too (BA-463).

**I-4 Buffer record.** `{"v": 1, "model_type", "hf_config_sha256", "transformers": "5.19.0",
"builder_fingerprint_sha256", "tensors": {name: {"dtype", "shape", "sha256", "b64"}}}`.
- Names: the full names of `root.named_buffers()`.
- Pinned as `cfg.hf_buffers`.
- Each wave holds it as `hf_buffers.json` (canonical JSON), listed in `wave.json`.

**I-5 `TIED` frame** (owner `transport-tied`)
- Type 7, codec `RAW_F32`, `ndim` 2, shape `(rows, h)` (the whole payload's, on every fragment), `chunk` 0. The whole
  payload is `rows·h·4` bytes: an unfragmented frame (`frag_cnt ≤ 1`) needs `paylen == rows·h·4` at receipt, a
  fragment `paylen ≤ rows·h·4`, and the reassembled payload exactly `rows·h·4`; each check is a `FrameError`. A frame of
  0 rows is one empty fragment (amended 2026-10-11, M4 implement deviation, SPEC-transport §2).
- Forward direction (stage 0 → Head, on `next` sockets): `step = t`, `mb = i`, payload `e_i[T_i]`.
- Backward direction (Head → stage 0, on `prev` sockets): `step = t+1`, `mb = k`, payload `R_k`.
- Rows are in ascending id. The ids come from `make_batch` on both ends.
- A relay exists only when every stage from stage 0 to the Head has width 1 (BA-447). Any wider stage on the way is
  refused at finalize and in the planner.

**I-6 `runtime/accumulate.py`** (owner `stage-m4`)
```python
class TiedRowFold:
    def __init__(self, n_rows: int, order: Sequence[int]) -> None
    def add_head(self, mb: int, h: torch.Tensor, rows: torch.Tensor) -> None
    def add_embed(self, mb: int, e_rows: torch.Tensor) -> None
    def complete(self) -> bool
    def result(self) -> torch.Tensor
```
The semantics are RUNTIME-I19's:
- Rows outside `P ∪ T_i` fold as `acc + (h_i + 0.0)`.
- Held rows resolve in index order as `acc + (h_j + e_j)`.

**I-7 `runtime/spec.py`** (owner `stage-m4`)
- `JobSpec.device_classes: list[list[str]] | None = None`.
- `JobSpec.tied_relay() -> bool`.
- `pins()` gains `"device_classes"` for arch `"hf"` only. `hf_code` rides `asdict(model)`.
- `pins()["model"]` (and `recipe_digest`'s model dict in `local.py`) omits every `hf_*` field of a non-hf model and
  every `moe_*` field of a model with `moe_experts == 0` (BA-465). So dense pins and digests are byte-identical to
  3b0d2f8's, and a pre-M4 dense wave still resumes. Golden tests in `stage-m4` and `oracle-m4` check this.
- `stage_param_shapes()` for arch `"hf"` reads `cfg.hf_census`'s `param_shapes` (I-10). It never imports hfm or
  transformers, because `finalize` runs in the coordinator and planner processes (BA-419, BA-441).
- `finalize()` refuses, naming the parameter and the stage:
  - a split tied hf table with any stage from stage 0 to the Head at width > 1;
  - a non-hf `tie_embeddings` split across stages (BA-446).
- `worker/stage_main.py` refuses an hf spec without `--launcher` (BA-445).

**I-8 `runtime/determinism.py`** (owner `determinism-pins`)
- `class DeterminismError(ValueError)`.
- `setup_determinism(seed)` assigns the pins, refuses a pre-set differing pin, sets every flag and reads it back.
- `check_determinism()` does the read-back only.

**I-9 GATE on the control channel** (owners `agent-hf`, `control-m4`)
- Request: `{"kind": "GATE", "run_id", "hf_config" (≤ 256 KiB), "recipe": {"compute_dtype", "seq_len",
  "mb_size"}, "devices": [str]}`.
- Reply: `{"kind": "GATE_REPORT", "run_id", "reports": [GateReport], "fingerprint", "device_classes"}`, at most 1 MiB.
  The child has a 300 s timeout.
- The pure admission function is `hf_admission(reports, fingerprints, builder, slot_map, pool, *, torch_records,
  record_diff, recorded=None) -> list[str]`, where `recorded = {"hf_code", "record_text"}` of the run.
  - The coordinator passes `hfm.TORCH_RECORDS` and `hfm.record_diff`, imported at call time in the hf branch.
  - The wave-1 tests pass crafted ones. `record_diff` itself is `model-core`'s test.

**I-10 Builder child and census** (owner `model-core`; callers `control-m4`, `oracle-m4`)
- Command: `python -X pycache_prefix=<fresh> -m wantrain.model.hfm record --in {"hf_config", "seed"} --out OUT.json`, in
  `stage_env`.
- Output: `{"record": I-4, "census": [unit dicts], "fields": {...}, "hf_code": I-1, "record_text_gz_b64", "fingerprint":
  I-3}`.
- Each unit dict: `{"unit", "kind", "params", "linear_macs_per_token", "attn_heads", "attn_kv_heads", "head_dim",
  "softcap", "window", "tied", "param_shapes"}`.
  - `param_shapes` is `[[name, [dims]], ...]`: the unit's trainable parameters, in its `named_parameters` order.
  - A tensor shared by two units is listed once, at its first unit.
- The coordinator and the planner never import transformers.

**I-16 Tripwire decision** (owner `control-m4`; callers `coordinator.unarmed_reason` and `cli.py`, `agent-hf`)
```python
def m4_unarmed_reason(model: dict) -> str | None   # in coordinator/coordinator.py; reads M4_ARMED
```
- It returns the refusal for `arch == "hf"` or `moe_experts > 0` while `M4_ARMED` is False, and None otherwise.
- `cli.py` reads the spec JSON's model before `JobSpec.load`. Only for hf or MoE, it imports this function at call time.

**I-11 MoE in `model/arch.py`** (owner `model-core`; SPEC-model §2 and §3 are normative)
```python
class MoEMLP(nn.Module): ...      # router (nn.Linear subclass, never SplitLinear), experts (SplitLinear SwiGLU),
                                  # shared_expert, shared_expert_gate, sel_bias (persistent fp32 buffer, E)
def moe_forward(cfg, view, x) -> torch.Tensor      # the exact function; records into the calling module
def take_moe_records(layers) -> list["MoERecord"]  # walks each layer's modules(); pops, sorted by (moe_layer, call)
MoERecord = NamedTuple("MoERecord", [("layer", int), ("aux", dict), ("counts", torch.Tensor)])  # counts int64 (E,)
def moe_bias_update(layers, counts: dict[int, torch.Tensor], global_tokens: int) -> None
```
The aux gradients are injected locally per layer and microbatch (decision 2). The CE loss never carries them.
- **Records** (BA-462). Every module that calls `moe_forward` carries `moe_layer: int` and holds its own records. That
  module is a `MoEMLP` or a SUBSTITUTE subclass.
  - Native: `moe_layer` is the index in `build_layers`.
  - HF: `moe_layer` is the decoder-layer index from the unit map, set by hfm when it applies the swap. So the records of
    `HFUnits` windows over one shared tree are keyed by the HF layer.
- **HF recipe.** An HF row's `cfg` is the row's `recipe(hf_config)`: E, k, router dtype, norm, the shared expert and
  its gate, gamma 0. The lbl and z coefficients come from the run's GPTConfig `moe_*` fields. `router_aux_loss_coef`
  is never mapped (ADR-0008 D8).
  - `output_router_logits = true` without aux fields is refused, naming the field.
  - So is a block attribute `moe_forward` does not read, such as `router_jitter_noise > 0`.

**I-12 MoE state on the wire and in checkpoints** (owners `stage-m4`, `oracle-m4`)
- Stage checkpoints gain `"moe_aux": [[step, mb, layer, hex], ...]`.
- `sel_bias` is a persistent buffer named `<layer>.mlp.sel_bias` in `state_dict()`, and a weight of the manifest.
- ZeRO-1 adds the biases to the ownership list after the trainable parameters (index `n_params + j`, owner `index %
  w`). Each microbatch's count vectors ride to the owner on the frames that already carry that microbatch's gradients
  (ADR-0008 D10). There is one ruling per gradient path (BA-454):
  - batch (`_optimizer_step_wide`, one OBJ `GRADS` per step): `{mb: tensors}` gains a `"counts": {layer: int64 (E,)}`
    entry per mb;
  - `stream_grads` (one OBJ `GRADS` per microbatch): `("mb", mb, tensors)` gains the same dict;
  - wire native (`_w_native`): one exchange frame per (mb, bias index), of kind `exchange.COUNT`, codec `RAW_I64`, shape
    `(E,)`, validated by `exchange.check_frame`.

  On every path the owner folds the vectors by integer add in microbatch order. A vector whose sum is not `T_mb·k`,
  or that is negative, misshapen, duplicated or missing at the barrier, is fatal and named. The bias returns in the
  existing `PARAMS` broadcast. Dense payloads are byte-identical.
- Pieces put the bias in its block's small-tensor unit. Its count lane is `RAW_I64`, cut at the unit's boundaries, on the
  same GRADS frame, and validated by `exchange.check_frame`.

**I-13 Scheduler cost functions** (owner `scheduler-m4`; caller `plan-wire`)
- `hf_unit_costs(census, seq_len, mb_size, compute_dtype)`.
- `delta_t_tied(t_s, t_F, t_B, rows_T, rows_R, hops, latency_s, bandwidth_Bps, *, d_model, vocab_size, tokens_mb,
  mem_bw_Bps, adamw_bytes_per_elem=28) -> dict` (its five terms and their sum). The keyword-only arguments carry h, V,
  T_mb and BW_mem, which the formula needs and the 8 positional ones cannot give; `mem_bw_Bps = math.inf` (unmeasured)
  makes the scan and AdamW terms 0, a lower bound (amended 2026-10-11, M4 implement deviation, SPEC-scheduler §2.3).
- `moe_layer_costs(moe: dict, d_model, seq_len, mb_size) -> dict`.
- All three are pure, over explicit arguments.

**I-14 `scheduler/fleetplan.py` plan-mode pins** (owner `plan-wire`; caller `control-m4`)
```python
MODEL_KEYS  # + "hf_config", "hf_impl" and every moe_* field of SPEC-model §2
def model_config(job: dict, hf_pins: dict | None = None) -> tuple[GPTConfig, list[str]]
def build_spec(fleet: dict, allow_pieces: bool = False, hf_pins: dict | None = None) -> dict
# hf_pins = the builder child's {"hf_buffers", "hf_census", "hf_code", "fields"} (I-10)
```
- A plan dict carrying `hf_buffers`, `hf_census` or `hf_code` is refused, naming the key. These are never trusted from
  the caller.
- An hf plan without `hf_pins` is refused, naming the census.
- `control-m4` passes the keyword only for hf jobs. After every plan-mode `build_spec` it runs
  `plan_model_mismatch(plan_job, model) -> list[str]` over arch, `hf_config` (canonical sha256), `hf_impl` and every
  `moe_*` key, and refuses on any difference (BA-461).

**I-15 `fault_tolerance/checkpoint.py`** (owner `wave-files`; caller `stage-m4`)
```python
def commit_wave(out_dir, tag, rank, peer, step, save_fn, keep=WAVE_KEEP,
                extra_files: dict[str, bytes] | None = None) -> tuple
```
- Extras are written before `wave.json` and listed after `state.pt` as `{name, sha256, bytes}`. Their names are
  validated first.
- With `None` or `{}` the manifest is byte-identical to today's.
- A held wave of the same step whose extras differ is refused, naming the file.
- Readers are unchanged: they already verify every listed file.

**I-17 `model/rollback.py`** (owner `model-core`; readers `gpt.validate`, `hfm`, `spec.finalize`, the stage's `TIED`
receipt and `planner`, each at call time; BA-467)
```python
ROLLED_BACK: frozenset[str] = frozenset()
PIECES: dict[str, str]   # piece id -> target ('model-4a' | 'tied-split-refused' | 'moe-from-scratch'); see D-3
def target(piece: str) -> str | None   # None unless rolled back; KeyError naming an unknown piece
```
The module imports nothing. Rolling a piece back is one edit, adding its id to `ROLLED_BACK`.

## Binding audit (PLAN-M4.json `binding`, BA-401..BA-468)

The ids start at 401 because the M2 and M2b code comments already cite BA-1..BA-60. Every instance of the hazard this run
touches is listed with its ruling. Every ruling is delivered by some task's `binding` list, and the machine check
verifies that.

- **One build path** (BA-401..BA-404, BA-449).
  - arch `"hf"` builds only through `hfm.build`, and the native loader is deleted.
  - A checkpoint build loads only the stage's units; the rest stay on meta.
  - `wgrad.convert` takes the exclusion set and never converts a router.
- **RNG and mode** (BA-405, BA-406, BA-444).
  - Only the from-config HF init draws.
  - Trace, gate and census run under `fork_rng`, and the MoE block draws nothing.
  - Every HF module runs in training mode, set explicitly by hfm and asserted by stage, oracle, profile and fit. The
    gate therefore checks the code path the stages run.
- **Call sites, ids, autocast and loss** (BA-407..BA-410). Only arch-`"hf"` branches call `call_layer`. Dense call sites
  are unchanged.
- **Tied table split across stages** (BA-411..BA-414, BA-416, BA-447).
  - The Head owns the table and folds it with `TiedRowFold`.
  - Stage 0's copy is out of every list.
  - `TIED` is relayed cut-through only along a width-1 chain.
  - The stage dispatch raises on an unhandled frame type; today it drops one silently (`stage.py:2413-2431`).
- **Non-hf tied split** (BA-446). This is silently wrong today: each stage trains its own copy, and only pieces refuses
  it (`planner.py:940`). It is now refused for every arch but `hf`.
  - In the planner the check moves ahead of `_refusals_before_pricing`'s width-1 early return (`planner.py:928-931`).
    Today that return admits every width-1 candidate before any tied check, which is the common case.
- **The planner's `unpriced` refusal** (BA-466, `planner.py:722-740`, `:946-947`). Today every wide candidate of an
  arch other than the builtin block is refused.
  - arch `hf`: kept, so plan mode emits hf plans at width 1. ZeRO-1 and pieces over HF units (BA-434) are reachable
    in spec mode only (D-10).
  - MoE: priced by SCHEDULER-I17's MoE unit model, never by the dense block's units. `plan-wire` proves the planner's
    units equal `pieces.exchange_units`.
- **Rollback per piece** (BA-467, I-17, D-3).
- **Identity corpus** (BA-468). `tests/oracle/` is a harness directory, so no task may own it. The arms live in
  `tests/helpers/test_m4_identity_*.py` (control owns `tests/helpers/`), and `identity.json` and the oracle rung name
  them.
- **Typed decode** (BA-415).
- **Buffers and code identity are bytes of the run** (BA-417, BA-418, BA-424, BA-433, BA-436, BA-448, BA-460).
  - The record is built once and pinned.
  - `hf_buffers.json` is listed in `wave.json` through `commit_wave(..., extra_files)`. Today `commit_wave` writes a
    one-entry manifest and `_check_files` refuses an unlisted file (`checkpoint.py:433-436`, `:210-211`), so a
    "carried unchanged" ruling would have failed every hf resume.
  - `hf_code` is pinned.
  - The RECORD text is kept, and every admission and local resume compares against it, naming the first differing path
    (CONTROL-I24).
- **Oracle and manifest** (BA-420..BA-422, BA-457).
- **Processes** (BA-423, BA-442, BA-445, BA-464).
  - Every HF child runs with `-X pycache_prefix=<fresh>` in `stage_env`, and its prefix is deleted at exit.
  - `wantrain stage` and `stage_main` without `--launcher` refuse an hf spec permanently, as M2b refuses pieces.
  - `wantrain local` and `wantrain stage` refuse an hf or MoE spec until armed (BA-464, I-16).
- **Admission** (BA-425..BA-429, BA-441). GATE runs before placement; decision 8 is checked at every admission; the
  coordinator never imports transformers.
- **Determinism** (BA-431, BA-432). The hfm CLI children call `setup_determinism` before any build.
- **Dense pins and digests** (BA-465). The new `GPTConfig` fields never enter a dense model's pins or digest.
- **MoE** (BA-450..BA-459, BA-462):
  - the selection without `topk`;
  - written folds and +0 for unrouted experts;
  - aux recorded and replayed (`MOE-REPLAY`);
  - the bias update from int64 counts after AdamW;
  - ZeRO-1's ownership list and count vectors, ruled for each of the three gradient paths (batch, `stream_grads`, wire
    native);
  - records and recipes under the HF class swap (BA-462);
  - all nine SUBSTITUTE rows of SPEC-model §3: three blocks, three routers and three experts classes (BA-438), so HF's
    own experts forward never runs;
  - pieces' count lane and the small-tensor unit;
  - expert exchange units;
  - the manifest sections;
  - the planner charges;
  - HF banks with dW in B, and their held bytes.
- **Plan mode** (BA-461). `MODEL_KEYS` gains `hf_config`, `hf_impl` and `moe_*`. The builder's pins reach the
  planner, and a dropped key is refused.
  - Today `fleetplan.model_config` copies only `MODEL_KEYS` (`fleetplan.py:36-77`). A plan-mode MoE job would therefore
    train dense.
- **Platforms without a `TORCH_RECORDS` entry** (BA-463). Each test asserts the refusal there and never skips.
- **Carried unchanged, proven by arms:** ZeRO-1 and pieces over HF and MoE chunks (BA-434; HF in spec mode only, BA-466), and `TIED` on the sealed and
  paced sockets (BA-435). Waves through the control plane (BA-436) are carried by `export_wave`, which already copies
  every listed file.

## Tripwire `m4-unarmed`

- **Flag.** `coordinator.M4_ARMED = False`. `unarmed_reason(job)` makes `Coordinator.run_job` refuse every job whose
  model arch is `"hf"` or whose `moe_experts > 0`, before any agent request.
  - The check is independent of `PIECES_ARMED`. Today's early return at `coordinator.py:1773` must not skip it.
  - The same decision (`m4_unarmed_reason`, I-16) refuses the operator entries `wantrain local` and `wantrain stage`
    for an hf or MoE spec (BA-464).
  - The `run_local` and `run_baseline` Python APIs, which the tests use, and hfm's gate stay possible, each behind
    its own guard and gate.
  - `wantrain stage` refuses hf permanently.
- **Why.** With only some legs in, a production run would be silently wrong. Examples:
  - a split tied table without the fold and relay;
  - units launched without the gate;
  - buffers rebuilt per node;
  - a heal onto another class;
  - a plan that charges a 2-byte cut;
  - an MoE bias updated without its counts;
  - `wantrain local` of an MoE spec with neither the stage's nor the oracle's bias update in, which `--compare` would
    pass.
- **Legs.** `control-m4` (the first leg: it adds the refusal and the test), `model-core`, `stage-m4`, `transport-tied`,
  `scheduler-m4`, `agent-hf`.
  - Every leg's verify includes the tripwire test.
  - Exit 5 (no test collected) means the first leg has not landed. It is recorded as NOT RUN, never cited as passed.
- **Arm.** `arm-m4` depends on every wave-1 task but `hf-pin`, which is not a leg. It sets `M4_ARMED = True` and keeps
  the refusal tests passing by monkeypatching the flag to False.
  - Its fast verify is the flag and its tests, including `test_m4_unarmed_cli_*` (the two CLI entries refused when
    the flag is False).
  - The end-to-end arms that justify arming run in the post-wave ladder and in GATES-M4 (see "Within wave 2").
- **Test.** `node agents-architetura/tools/verify/machine-lock.mjs run --slot --label m4-unarmed --wait 3600 -- uv run
  --no-sync --offline pytest -q -p no:randomly tests/unit/test_control_admission.py -k m4_unarmed`.

## Hardware mode: gpu false

- No GPU, no second host and no Vast.ai. Every arm is a localhost CPU cluster with emulated WAN links.
- **N/A, with the reason recorded and never faked:**
  - the `nvidia-*`/`libcuda` fingerprint arm;
  - eager attention on H100;
  - the CUDA embedding backward;
  - the tied round trip across real regions;
  - MoE GPU kernel facts (ADR-0008 R1) and the grouped GEMM question (R5).

  All of these wait for M1.
- **E8 (MPS) gates nothing.** Its tests carry the `mps` marker.
- Simulated-fleet numbers are SIMULATED. Mac CPU timings (E7 for both ADRs) are MEASURED (local Mac: Apple M4, with the
  command). Real-hardware validation stays pending (M1).

## Bar to freeze in GATES-M4.json (input for the gate step)

**Rungs.** DESIGN §8's M4 row: the M0 rungs plus `identity`, `fuzz` and `redteam`, green twice in a row, with `full` in
the last pass only. A rung with no inputs is N/A with its reason.

**Pinned commands.** Each runs through `pinned.mjs` with the whole machine. HF commands use the locked `.venv` after
integration.

| Command | Bar |
|---|---|
| `pytest tests/unit/test_finetune.py -k "hf_family or hf_e2 or hf_e6"` | the 12-family parity gate, E1 extras, E2, E6 |
| `pytest tests/unit/test_finetune.py -k "moe"` | MODEL-I8: native E1, HF swap F8 (Mixtral, Qwen2-MoE, Qwen3-MoE), DeepSeek-V3 row and fidelity |
| `pytest tests/integration/test_engine_bitwise.py -k "hf_identity or hf_oracle"` | every mandatory family's pipeline identity (all 12), checkpoint arm, resume, E3, E4 by sha256; a collect check requires every arm id `identity-m4` names; 2 repetitions |
| `pytest tests/integration/test_engine_bitwise.py -k "moe_identity or moe_oracle"` | ADR-0008 E2 (ZeRO-1 on the batch, `stream_grads` and wire-native paths), E3, every E4 must-fail (the up-stores-X control firing F3), E6 native and HF v5 with dense tables unchanged, the HF-MoE adapter arms (Mixtral, Qwen3-MoE, E2-HF); a collect check requires every arm id, each must-fail by name; 2 repetitions |
| `pytest tests/helpers/test_m4_identity_hf.py tests/helpers/test_m4_identity_moe.py` with `FACTORY_IDENTITY_OUT` | every corpus arm writes its state, and none is a refusal on this Mac (BA-468); the `identity` rung (pinned, twice) compares the three modes by sha256; 2 repetitions |
| the guarded-harness check | the harness hashes, plus the exact planned content of `identity.json` and of the oracle and rest rungs |
| `pytest tests/integration/test_coordinator.py -k m4_coordinator` | coordinator end to end (hf and MoE, spec and plan mode), heal, resume refusal |
| `pytest tests/unit/test_runtime_spec.py -k "m4_stage or m4_wire"`, `tests/unit/test_user_model_precision.py -k determinism` and `tests/unit/test_fault_tolerance.py -k wave_extra` | the stage behaviour, the house fix (ISA read-back included) and the wave extras |
| the tripwire test | as above |
| the invariant-coverage check for MODEL-I4, I8, I9, I10, RUNTIME-I15, I18, I19, SCHEDULER-I21, I22, CONTROL-I24 | each has at least one `covers:` test (D-5) |
| `python -m wantrain.model.hfm census --out <json>` (10 types per subprocess, SKIP above 20M parameters) | ADR-0009 F7: no DIFFER; ≥ 55 of the ~60 in-scope dense types EXACT; no encoder admitted |
| `FACTORY_E7_BOUNDS=1 pytest tests/unit/test_finetune.py -k "hf_e7 or moe_e7"` on a quiet machine | ADR-0009 F7 (`test_hf_e7_*`: `c_glue` ≤ 1 ms, `c_pass` ≤ 10 µs, HF host ≤ 1.5× native, no stub allocation, saved bytes 4/6/+2 per `b·H·S²`) and ADR-0008 F6 (`test_moe_e7_*`: `MoEMLP` F+B+W ≤ 1.15× the v4-style block; the swap's F+B ≤ 1.15× transformers' block + 3E 0-row calls), MEASURED local Mac. Without the variable the same tests measure and assert only the byte counts. |
| `FACTORY_E7_BOUNDS=1 pytest tests/integration/test_engine_bitwise.py -k "e4_tied or moe_identity_f7"` on a quiet machine | ADR-0009 F4: each tied arm's exposed time within max(20%, 2 ms) of `ΔT_tied` from its measured terms (SIMULATED); ADR-0008 F7 (`test_moe_identity_f7_*`): an MoE stage's WT_TELEM compute ≥ 0.9× its `c_s` (SIMULATED) |

**Campaigns.** The existing `fuzz` rung covers the typed-decode change and the count-lane `check_frame`.

**N/A:** every CUDA arm, E8 (MPS) and `multinode`.

## Risks, deviations and questions (facts separated from hypotheses)

- **Q-1 (for the human; Linux TORCH_RECORDS).**
  - FACT: only the macOS arm64 entry can be computed offline. Until a Linux entry exists, a Linux node's fingerprint is
    refused (`FP-TORCH-RECORDS`).
  - Each test's disposition is fixed by BA-463 and D-9.
  - Options: (a) one human-allowed fetch of the Linux torch wheels named in `uv.lock` (0.9–3 GB of transient disk,
    HYPOTHESIS; no model download); (b) Linux hf runs stay refused until M1.
  - Recommendation: (a), at M1 or earlier.
- **R-1.** `model-core` (about 2.5k lines) and `stage-m4` are large single tasks. Mitigations:
  - acceptance itemized per ADR decision;
  - frozen interfaces I-1..I-17;
  - the 12-family sweep and the HF-MoE rows moved to `model-families`.
- **R-2 (HYPOTHESIS).** An HF test process pays 3–6 s for import and a fresh-prefix compile. The 12-family identity
  sweep takes 6–10 min, and the MoE identity sweep is about that size too. Fast verifies run 2–4 arms; the sweeps are
  GATES entries.
- **R-3 (HYPOTHESIS).** A fresh pycache prefix holds 50–100 MB, so a 4-stage run plus the oracle holds about 0.5 GB
  transiently. Prefix deletion at exit is acceptance in `agent-hf`, `oracle-m4` and `control-m4`.
- **R-4.** If `hf-pin`'s offline lock moves a locked package or lacks a dependency, it STOPS with a recorded conflict.
  No other task depends on it: every test command runs on the overlay. Only the ladder after the waves needs the lock,
  so a stop blocks the ladder, not the waves.
- **R-5.** E6 drift arms run on a symlinked copy of the site directory under `tmp_path`, never on `.venv`.
- **R-6.** `gemma3`'s causal-LM text model is `gemma3_text`. `gemma3` maps to the multimodal class, which is out of
  scope.
- **R-8 (HYPOTHESIS).** The ladder's `integration` rung runs `tests/integration` whole within 3600 s. The M4 arms
  (the 12-family and MoE sweeps included) add an estimated 15–25 min to M2b's suite.
  - Falsified if the rung times out on the post-wave ladder.
  - Then the sweeps move to their own pinned entries and skip under `ladder:*`, as M2b's E4 wall-clock arms do (LESSONS
    2026-10-10).
- **R-9 (HYPOTHESIS; residual of the house fix).** The ISA pin is read back for ATen (`get_cpu_capability`). MKL has no
  read-back API in torch, so an MKL initialized before `setup_determinism` would keep its own dispatch.
  - This is not reachable on arm64, which is this Mac and every gpu-false arm.
  - On x86 it is caught only by the cross-host identity arms (M1).
- **R-7 (open in ADR-0009, unchanged here).**
  - R13: a stub read through a NaN-absorbing op is not caught by the gate; it is recorded, not judged.
  - R16: the oracle of a mixed map is the unit chain.
  - R17: constructor floats come from libm.
  - R18: OS objects are compared, not verified.
- **D-1 (deviation, recorded).** ADR-0009 asks for its local experiments E1–E8 first, and ADR-0008 for E1–E7. Here they
  are the build's own tests (E1–E4, E6, E8). E7's measurements are also tests: `model-families` writes `test_hf_e7_*`
  and `test_moe_e7_*`, and `identity-m4` writes `test_moe_identity_f7_*`. Their timing bounds and the census are GATES
  entries. The research prototypes already
  ran the E1-like arms (MEASURED), so F1–F8 become gate bars instead of a separate wan-experiment run. ADR-0009's E5 is
  ADR-0007's slice build, deferred. Two of its arms are built here: the tampered-buffer wave and the checkpoint span
  build.
- **D-3 (resolved in revision 5; rollback granularity).** SPEC-model task 4 wants 4a "its own commit, kept by a
  rollback". SPEC-runtime task 7 wants 7a/7b/7c/7d to roll back by task id, each to its own target (ADR-0009
  `round6.md` §4).
  - FACT: a file-level revert by task id is not possible for the fused pieces. Under the 2-wave and
    one-owner-per-file-per-wave rules, 4a, 4b and 6a share `model-core` and `model-families` (`gpt.py`, `hf.py`,
    `test_finetune.py`). 7a, 7b and 7c share `stage-m4` and `stage-wire` (`stage.py`, `spec.py`, the one runtime test
    file). Splitting them needs a third wave or test files no workstream owns.
  - Ruling (BA-467, I-17): every piece gets its own revertible unit, an entry of `model/rollback.py`'s `ROLLED_BACK`
    table. Its target is enforced at one place and tested on its own, so each piece reverts independently:

    | Piece (round6 §4 row) | Plan tasks | Unit | Target and where it is enforced | Test |
    |---|---|---|---|---|
    | scheduler 6 (#1) | `scheduler-m4`, `plan-wire` | `'scheduler-6'` | tied HF at S > 1 refused at planning (planner) and at finalize | `test_m4_plan_wire_rollback_*`, `test_m4_wire_rollback_*` |
    | runtime 7a (#2) | `stage-m4`, `stage-wire` | `'runtime-7a'` | the same; `tied_relay()` False | `test_m4_wire_rollback_*`, `test_m4_plan_wire_rollback_*` |
    | runtime 7b (#3) | `stage-m4`, `stage-wire` | `'runtime-7b'` | the same; any `TIED` received fatal | `test_m4_wire_rollback_*` |
    | transport §2 (#4) | `transport-tied` | the task's own two files | revert them: the type id is unassigned, and typed decode refuses it | its tests go with it |
    | runtime 7c, 7d (#5) | `stage-m4`, `oracle-m4`, `identity-m4` | `'runtime-7c'`, `'runtime-7d'` | 4a: `GPTConfig.validate()` refuses arch `hf`, so no spec, plan, oracle or resume of an hf wave starts | `test_hfm_rollback_*` |
    | model 4b (#6) | `model-core` | `'model-4b'` | 4a, as above | `test_hfm_rollback_*` |
    | control 6 (#7) | `control-m4`, `agent-hf`, `agent-wire`, `arm-m4` | `'control-6'` | 4a, as above (hf refused before placement) | `test_hfm_rollback_*` |
    | model 4a (#8) | `model-core` | none | kept; never a target | — |
    | ADR-0008 6a (#9) | `model-core`, `model-families` | `'model-3-6a'` | MoE from scratch: hfm ignores the SUBSTITUTE rows (MOE-NO-ROW); native MoE runs | `test_hfm_rollback_*`, `test_moe_hf_rollback_*` |

  - The pieces' code stays in the tree while it is rolled back. What reverts is reachability, which is what round6's
    targets define. Reverting the source itself stays possible per file only for `transport-tied`.
- **D-4 (resolved in revision 5; the identity corpus).** SPEC-runtime task 7 puts tiny HF families (llama, gemma2 tied,
  granite, qwen3) in the `oracle` and `identity` rungs. ADR-0008's validation plan asks the same for MoE.
  - FACT: those rungs ran `tests/oracle/**` only (`rungs.json`, `identity.json`). `tests/oracle/` is in HARNESS_DIRS
    (`agents-architetura/tools/wf/lib/project.js:120`), so the plan validator refuses any task that owns a file there.
  - Ruling (BA-468):
    - `identity-corpus` (control, which owns `tests/helpers/`) writes `tests/helpers/test_m4_identity_hf.py` and
      `test_m4_identity_moe.py`, with `test_state_identity.py`'s three-mode protocol.
    - `corpus-wiring` (integrate; both files are HOT_FILES) adds them to `identity.json`'s corpus and to the oracle rung.
    - The identity rung therefore compares, by sha256, the reference, feature-off and production runs of tiny llama,
      qwen3, gemma2 tied, granite, Mixtral and native MoE.
    - GATES-M4 pins the config content and a run that must write every arm's state.
  - The wide sweeps stay in `tests/integration` (`hf-identity-x2`, `moe-identity-x2`).
- **D-5 (conflict; the deferral lines).** MODEL-I8, I9, I10, RUNTIME-I18, I19, SCHEDULER-I21, I22 and CONTROL-I24 still
  say `DEFERRED-TO: M4` in their specs, and `invariants.mjs` exempts a deferred id from coverage.
  - No task may edit the specs (docs/design is no workstream's).
  - Correction (revision 5): the earlier task-level check read the global `invariants.mjs --json` report. That report
    counts a `covers:` line anywhere, so it could not fail on one task's omission.
  - Each task now runs the own-test check described under Goal: its own test file, its own test-name prefix, every id it
    delivers, and a WOULD-FAIL-IF line in each test.
  - GATES-M4 keeps the global check for all ten ids as the milestone-level bar.
  - Lifting the deferral lines is a `freeze.mjs amend` at closeout, for the researcher.
- **D-6 (deviation, for the human; TIED through a wide middle stage).** RUNTIME-I19 says intermediate stages forward
  `TIED` "under the chain's (step, mb) rules and sockets". With a middle stage of width > 1, consecutive microbatches
  take two paths, and the Head's in-order receipt rule would fire spuriously. This plan refuses such a plan (BA-447),
  which is narrower than the spec's text. The ZeRO-1 middle-stage arms use untied families. Specifying a fixed relay
  peer and a reordering buffer is a follow-up.
- **D-7 (recorded).** BA-446 refuses a non-hf tied table split across stages. This configuration is silently wrong today,
  and no spec line covers it. The refusal changes no admitted run's bits.
- **D-9 (deviation, for the human; Linux CI carries no HF pipeline evidence).**
  - FACT: SPEC-model §3 makes the fingerprint require the platform's `TORCH_RECORDS` entry, and only macOS arm64 has
    one (Q-1). So on Linux every HF path through the gate's verdict, `run_local`, the agent's GATE or the coordinator is
    refused (`FP-TORCH-RECORDS`).
  - Ruling (BA-463):
    - These tests assert the comparison fields on every platform. Where `hfm.platform_admitted()` is False they assert
      the refusal, strictly, naming the platform tag. Nothing is skipped or xfailed.
    - The unit-level parity tests (unit chain vs vanilla, the MoE block, F8) do not depend on the fingerprint and pass on
      Linux.
  - Consequence: the HF pipeline identity evidence comes from this Mac's closeout (GATES-M4). Linux CI proves only the
    refusal.
  - The researcher opens a findings.json entry for it at closeout; this plan owns no factory file.
  - Recommendation: Q-1 (a).
- **D-10 (deviation, recorded; plan-mode width of hf jobs).** The planner keeps refusing wide hf candidates as
  `unpriced` (`planner.py:946-947`, BA-466). M4 builds no exchange-unit model of HF units for the planner.
  - Consequences:
    - plan mode emits hf plans at width 1;
    - ZeRO-1 and pieces over HF units are reachable in spec mode only, where the runtime carries them and
      `identity-m4` proves them (BA-434);
    - SCHEDULER-I21's HF-bank term of SCHEDULER-I16's device-memory condition is built and tested in `admission.py`,
      but the planner reaches it only once wide hf candidates are priced.
  - SCHEDULER-I22's "and in S1 for sharded plans" was already unreachable under a split tied table (BA-447).
  - The refusal is conservative. It may refuse a feasible plan; it never admits a mispriced one, and it changes no bit.
    Native MoE wide candidates are priced (SCHEDULER-I17's MoE units).
  - Follow-up: an HF unit model built from the census's `param_shapes`.
- **D-11 (deviation, recorded; "one task per file").** The human asked for one task per file. This plan meets the
  weaker reading the 2-wave validator enforces: one owner per file per wave. A wave-1 task builds a file, and one
  wave-2 task is its only fix owner (Who fixes what in wave 2). 35 files have one wave-1 and one wave-2 owner:
  - `model-core` → `model-families`: `model/hfm.py`, `model/rollback.py`, `model/hf.py`, `model/gpt.py`,
    `model/wgrad.py`, `model/arch.py`, `tests/unit/test_finetune.py`;
  - `stage-m4` → `stage-wire`: `runtime/stage.py`, `runtime/accumulate.py`, `runtime/spec.py`, `runtime/pieces.py`,
    `runtime/exchange.py`, `runtime/logical_ckpt.py`, `worker/stage_main.py`, `tests/unit/test_runtime_spec.py`;
  - `determinism-pins` → `stage-wire`: `runtime/determinism.py`;
  - `oracle-m4` → `identity-m4`: `validation/reference.py`, `validation/state.py`, `launch/local.py`,
    `tests/integration/test_engine_bitwise.py`;
  - `scheduler-m4` → `plan-wire`: `scheduler/planner.py`, `autoconfig.py`, `zero_bubble.py`, `partition.py`,
    `admission.py`, `exchange_sim.py`;
  - `agent-hf` → `agent-wire`: `worker/agent.py`, `resources/profile.py`, `resources/fit.py`, `cli.py`,
    `tests/unit/test_resources.py`;
  - `control-m4` → `arm-m4`: `coordinator/coordinator.py`, `fault_tolerance/replacement.py`,
    `tests/unit/test_control_admission.py`;
  - `wave-files` → `arm-m4`: `fault_tolerance/checkpoint.py`.

  Why: each workstream owns a fixed, small set of test files (model 1, runtime 3), and a wave-2 proof must sit beside
  the code it proves. Strict one-task-per-file would merge each pair into one wave-1 task. That task would then
  depend on its siblings and need a third wave. The human decides whether this reading stands. The factory fix
  (per-task test files, `tests/unit/test_<task-id>.py`) is in the plan-rule conflict above.
- **D-8 (recorded).** Two strengthenings beyond the spec text, both refusals, neither changing a bit:
  - `GPTConfig.hf_code` and the RECORD text are pinned (I-1, I-10), to deliver CONTROL-I24's "resume under another
    transformers RECORD is refused, naming the differing file".
  - The training-mode assert (BA-444).

## Review notes (round 1 findings and their disposition)

No round-1 finding was refuted. Each was verified against the code and specs and fixed, or recorded for the human where
fixing it needs a document no task owns:

1. oracle-hf had no deps (blocker): fixed. `oracle-m4`'s wave-1 tests are self-contained, and its end-to-end tests moved
   to `identity-m4` (wave 2, depends on all of their inputs).
2. hf-pin raced the shared `.venv` (blocker): fixed.
   - The overlay command (MEASURED) makes no test wait for the pin.
   - `--no-sync` means no test command writes `.venv`.
   - The pin's install is `uv sync --inexact`, which is additive.
3. stage-hf had no deps (blocker): fixed. `stage-m4`'s wave-1 tests are self-contained, and the cross-task tests are
   `stage-wire`'s.
4. agent-hf had no deps (blocker): fixed. The GATE end to end and hf PROFILE moved to `agent-wire`.
5. scheduler-hf and coordinator-hf had no deps (major): fixed.
   - Pure functions over explicit inputs carry the wave-1 tests (I-13, I-9).
   - The GPTConfig wiring is tested in `plan-wire` and `arm-m4`.
6. hf-identity depended on hf-families in the same wave (major): fixed. The fast identity arms use the families
   `model-core` proves, and the 12-family sweep is a GATES entry.
7. MoE was dropped (blocker): fixed. 15 tasks is not past 24, so MoE is in this DAG.
8. The oracle/identity rungs never ran the HF tests (blocker): fixed by citing only `integration` and pinning the arms
   in GATES. The missing `tests/oracle` owner is recorded as D-4.
9. DEFERRED-TO hid missing coverage (major): fixed. Every delivering task runs a coverage check, and GATES pins it. The
   spec amendment is D-5.
10. A resume under another RECORD was not refused (major): fixed (BA-448). `hf_code` is pinned, the RECORD text is kept,
    `record_diff` names the path, and the check runs in `control-m4`, `oracle-m4` and `arm-m4`.
11. There was no meta-device slice build and no checkpoint arm (major): fixed. BA-449 adds `span` to `build`/`build_layers`
    and the checkpoint arm to `identity-m4`.
12. Per-piece rollback was lost (major): recorded as D-3. Splitting needs more waves than the human's rule allows.
13. The one-slot launch path was unruled (blocker): fixed. BA-445 refuses hf in `cli.py` (`agent-hf`) and `stage_main.py`
    (`stage-m4`).
14. The non-hf tied split was unruled (major): fixed. BA-446 refuses it in `finalize` and the planner (D-7).
15. TIED through a wide middle stage was unruled (major): fixed by refusal (BA-447), recorded as D-6.
16. Module mode was unruled (major): fixed. BA-444 sets training mode explicitly in hfm, asserts it in the stage, the
    oracle, profile and fit, and adds an eval-mode control.

## Review notes (round 2 findings and their disposition)

Every round-2 finding was checked against the code and the ADRs and found real. None is refuted. Each is fixed in both
files:

1. Wave-1 consumers reached `model-core`'s symbols (blocker): fixed.
   - `hf_admission` takes `torch_records` and `record_diff` as arguments (I-9).
   - BA-407 now keeps `layer(x)` on every dense call site, and only arch-`"hf"` branches call `call_layer`.
2. Wave-1 files had no wave-2 fix owner (major): fixed.
   - `stage-wire` owns every runtime file of `stage-m4`.
   - `model-families` owns `gpt.py`, `wgrad.py` and `hf.py`.
   - `plan-wire` owns the scheduler files.
   - `agent-wire` owns `cli.py`.
   - `arm-m4` owns `replacement.py` and `checkpoint.py`.
   - `framing.py` goes to the post-wave fixers ("Who fixes what in wave 2").
3. `arm-m4` and `identity-m4` ran end to end over concurrent siblings' files (major): fixed. Their fast verifies cite
   only single-process or flag tests. The multi-process arms run in the post-wave ladder (`integration` runs
   `tests/integration` whole) and in GATES-M4.
4. Plan-mode jobs lost the hf and MoE keys (major): fixed by BA-461 and I-14 (`plan-wire`). `plan_model_mismatch`
   (`control-m4`) makes a dropped key a refusal, and `arm-m4` adds a plan-mode MoE arm.
5. Wave-2 tasks depended on `hf-pin` without using it (major): fixed. `hf-pin` is removed from five wave-2 deps, and
   `determinism-pins` from `model-families`.
6. E7 had no writer (blocker): fixed. `test_hf_e7_*` and `test_moe_e7_*` are in `model-families`, and
   `test_moe_identity_f7_*` in `identity-m4`. The GATES rows name them and `FACTORY_E7_BOUNDS=1`.
7. Missing falsification controls (major): fixed.
   - Every ADR-0008 E4 must-fail is in, the up-stores-X control firing F3 included.
   - The E6 HF v5 arm and the dense-tables check are in.
   - ADR-0009 E2's threaded state and swapped-block-skips-experts controls are in.
   - `model-core`'s MOE-NO-ROW test monkeypatches SUBSTITUTE, and its verify no longer asserts that SUBSTITUTE is empty.
8. The HF-MoE path through stage and oracle was unplanned (major): fixed by BA-462 and I-11. It covers `moe_layer` from
   the unit map, the row's `recipe(hf_config)`, and the `output_router_logits` and unread-attribute refusals. A
   single-process 2-span HF-Mixtral arm in `model-families` runs the path before closeout.
9. Linux disposition was unplanned (major): fixed by BA-463 and D-9 (strict refusal asserts, never skips).
10. The ISA pin was not read back (major): fixed. `check_determinism` reads `get_cpu_capability` on x86, with a
    monkeypatched-AVX512 test. MKL's missing read-back is R-9.
11. `arm-m4` armed before the proofs (major): fixed as in 3. The arming's evidence is the post-wave ladder plus GATES,
    and nothing leaves draft unless VERIFIED.
12. `commit_wave` could not list `hf_buffers.json` (blocker): fixed. The new task `wave-files` (I-15, BA-460) does it,
    BA-436 is corrected, and `stage-wire`, `identity-m4` and `arm-m4` depend on it.
13. ZeRO-1 count transport was ruled for one of three gradient paths (major; the finding text was cut off and was
    reconstructed from `stage.py:2056-2178`): fixed. BA-454 and I-12 rule batch, `stream_grads` and wire native (a
    `RAW_I64` COUNT exchange frame). `stage-m4`, `stage-wire` and `identity-m4` test all three.

## Final check (every round-1 and round-2 finding against both files, revision 4)

Most findings were already fixed in revisions 2 and 3. The final check found these still open, or reopened by a
later fix, and fixed them here. The numbers are the final check's F1..F55 (round 1 is F1..F38, round 2 is
F39..F55):
- F17: no task made the hfm CLI children call `setup_determinism`. `model-core` now does (BA-432).
- F18: the oracle lacked the stage's start asserts. `oracle-m4` now has them, and `identity-m4` tests them (BA-406).
- F22: `oracle-m4`'s fast verify could not fail on a dense regression. Dense golden tests now run in it.
- F23: `stage_param_shapes` for hf built the skeleton, so `finalize` in the coordinator imported transformers. The
  census now carries `param_shapes`.
- F28: `gate` and `fingerprint` differed from SPEC-model §3. Their signatures are aligned.
- F29: these were added:
  - the tokenizer sha256 on export;
  - E6's OS-object pair, the drifted builder and the compile time;
  - the S1 clause, ruled unreachable under BA-447.
- F30: the Head sent `TIED` past the attempt's last step. It no longer does.
- F33: the new fields leaked into dense pins and digests (BA-465).
- F35: the no-transformers tests passed vacuously. They now use a raising meta-path finder.
- F36: nothing checked "no locked version moves". `hf-pin`'s verify now pins the 63 baseline pairs.
- F38: GATE ran on the agent's reader thread. It now runs off it.
- F52: the router and experts rows were missing. All nine rows are in.
- F54: the CLI entries were outside the tripwire (BA-464).

## Final check 2 (revision 5: the 11 findings left open, each re-read against the current files)

Round-1 findings that named old task ids were mapped to the current owner first: hf-identity → `identity-m4`;
oracle-hf → `oracle-m4`; stage-hf → `stage-m4`/`stage-wire`; hfm → `model-core`/`model-families`.
1. HF arms absent from the `oracle`/`identity` rungs (blocker): fixed. BA-468 adds `identity-corpus` and
   `corpus-wiring`, and GATES adds `m4-identity-corpus-x2` and the config pins in guarded-harness-unchanged. The plan
   validator refuses any task that owns `tests/oracle/` (HARNESS_DIRS), so the arms live in `tests/helpers/` (D-4).
2. Rollback targets fused (major): fixed by BA-467 and I-17. Each piece has its own entry in `ROLLED_BACK` and its
   own test. The mapping is D-3's table.
3. `call_layer` on dense paths in wave 1 (major): no longer holds. BA-407 and the acceptance of `model-core`,
   `stage-m4` and `oracle-m4` keep `layer(x)` on every dense call site. Only arch-`hf` branches call `call_layer`,
   imported at call time (revision 3).
4. New `GPTConfig` fields in dense pins and digests (major): no longer holds. BA-465 omits the new fields. Two
   goldens check it: `stage-m4`'s `test_m4_stage_*` on a dense `pins()` literal recorded before the edit, and
   `oracle-m4`'s `test_m4_oracle_dense_golden_*` on `recipe_digest`, manifest and baseline. Both run in each task's
   fast verify. BA-422's text was aligned with BA-465. `JobSpec.to_json` (`spec.py:442-445`) also serializes the new
   fields, but it is no pin and no digest, and `from_json` reads a pre-M4 spec through the defaults.
5. SCHEDULER-I21 items without a task (minor): fixed in `scheduler-m4`: the I16 bank term
   (`test_m4_plan_moe_i16_bank_*`) and §2.3's single hf LADDER rung (`test_m4_plan_hf_ladder_*`). The `c_s` rationale
   is `plan-wire`'s (`test_m4_plan_wire_moe_rationale_*`). Each has its own `-k` verify line.
6. and 8. Global coverage lines (minor): fixed. Each task's own-test ast check replaces them, and D-5 is corrected.
7. The planner's width-1 early return and the `unpriced` refusal (minor): ruled in BA-446 (the check moves ahead of
   the return), BA-466 and D-10, with tests in `scheduler-m4` and `plan-wire`.
9. identity-m4's fast verify skips multi-process items; nothing selects the middle-stage `TIED` forwarding (minor):
   fixed.
   - The items are named parametrization ids that GATES' collect checks require.
   - `identity-m4` adds the gemma2 tied chain3 arm.
   - `stage-wire` adds an in-process forwarding test with its own verify line.
   - The multi-process arms stay out of fast verifies for the "Within wave 2" reason.
10. Fuzz targets and the lock check unverified (minor): fixed. `transport-tied` runs the fuzz targets through a test
    slot; `tests/fuzz` stays the harness's, and a failure there is a recorded conflict for wan-fuzz. The "no locked
    version moves" check already existed (revision 4's 63-pair baseline line). `hf-pin` now also checks the sha256 of
    every wheel and sdist.
11. "One task per file" claimed (minor): fixed by stating it honestly. D-11 lists the 35 files with one wave-1 and one
    wave-2 owner. The finding counted 17, but the validator's own listing gives 35.

Open for the human: D-10 (hf plans are width 1 in plan mode) and D-11 (the one-owner-per-file-per-wave reading). D-3
and D-4 are resolved by BA-467 and BA-468; the human may still prefer two sequential runs to the rollback table.

## Later slices

None. Dense and MoE are one DAG of 18 tasks (the slicing bar is 24).
