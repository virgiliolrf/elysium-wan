# HANDOUT M2b: exact sharded stages (ADR-0006, as granted), replanned in 2 waves

Machine plan: [PLAN-M2b.json](PLAN-M2b.json). Design of record: [DESIGN.md](DESIGN.md) §8 (M2b row);
decision: [ADR-0006](adr/ADR-0006-exact-fsdp-placement.md); specs: SPEC-runtime task 5, SPEC-model task 2,
SPEC-scheduler task 4, SPEC-transport task 3, SPEC-control CONTROL-I9.

## Goal

A stage that has to be wide runs byte-exactly with its training state cut into per-tensor pieces under the
recipe's own left fold. The planner places such stages, inside a site or across a link, only where
SCHEDULER-I16 admits them on measured inputs. Every experiment is falsified locally (E1-E7, CPU, gpu false).

The milestone ends with these true:
- RUNTIME-I10-I13, MODEL-I7, SCHEDULER-I16, SCHEDULER-I17, SCHEDULER-I19, CONTROL-I9 (amended) and TRANSPORT-I8;
- wire `native` and shard `pieces` are in the every-feature-on identity topology.

## Scope

**The human's scope (2026-10-09), unchanged:**
1. R14 first (done in the stopped run; re-verified here).
2. E1-E7, whose falsification criteria gate what follows.
3. Step 1, `wire: native`.
4. Step 2, `shard: pieces`, with:
   - S1/S2 admission;
   - plan mode for wide stages;
   - class-preserving heal;
   - logical checkpoints.

Deferred, not in this plan: R10 (measuring host-DRAM passes per byte) and R13 (admission by simulated step time).
k on host DRAM stays a **counted** figure, never measured. The count must include M2's sealing (SC-5).

**This replan (the human's rule #45, 2026-10-09).** The previous plan ran 19 tasks in 13 waves, because
`test_runtime_spec.py` and `test_engine_bitwise.py` had 7 owners each and `stage.py` had 5. This plan has the same
work in **8 tasks and 2 waves**:
- One task per file. Each workstream's wave-1 work is ONE task. The runtime's is `stage-pieces`; the scheduler's is
  `scheduler-pieces-plan`, which absorbs the done r14.
- Wave 1 builds every piece in parallel. Every cross-workstream symbol a wave-1 task imports is already on the tree
  at 4201c65, and its owner may finish it in the wave but not change it: the frozen interfaces I-2, I-3, I-4, I-7 (with
  the finalize contract) and I-10 (§Interfaces). The owner of a file that a sibling imports keeps it importable after
  every edit.
  - FACT (review round 2): the shared working tree still lets a wave-1 test import a sibling's half-edited file. So a
    wave-1 verify line runs only tests of the task's own files, plus tests that reach a sibling's file only through a
    frozen interface.
  - Evidence that needs the integrated tree runs in `arm-pieces`' verify in wave 2. That covers stage processes
    launched by the coordinator, RUNTIME-I2 under the scheduler's orders, and the deferred-dW tests in
    `test_zero_bubble.py` (D-10).
- Wave 2 only integrates: coordinator plan mode with the arm, and E4.

Nothing is dropped. Every old task, invariant, binding item and the tripwire is delivered by a new task (§Old to new).
- Review round 1 added BA-56 and BA-57 and rewrote BA-16, BA-44, BA-48 and BA-53.
- Review round 2 added BA-58 and the resume-step rule of BA-51 and BA-54. It widened the tripwire to wire `native`
  (§Binding audit, §Tripwire).

**Code base: the M2 that P-M2 merges.** M2 (PR #33) enters main before this build. The focus said to plan against
`origin/factory/m2`. That ref (3ebf48d) is older than the M2 being built (`~/edt-factory-run`, factory/m2 d070a12
plus work in progress). The M2 being built adds four things:
- the sealed, authenticated data plane (RUNTIME-I16, TRANSPORT-I7);
- the rebalance loop (CONTROL-I21, SCHEDULER-I15, RUNTIME-I15);
- immutable waves (RUNTIME-I17, CONTROL-I23);
- pins.

The plan follows the M2 that merges, not the stale ref (D-8). Every BA `where` cites that M2 by **symbol**, not by line.
Line numbers move with every M2 commit, and each coder reads the merged tree.

## Starting point (work done, not to redo)

| Code | State | Journal (`.factory/runs/M2b/tasks/`) | Goes to |
|---|---|---|---|
| `zero_bubble.py` bridge_orders, lane_order tests (r14) | **done**, reviewed | `r14-lane-order.md` | `scheduler-pieces-plan` (re-verified, not rewritten) |
| `runtime/pieces.py`, 11 `e6_` tests | partial: the e6_ tests pass | `e6-piece-kernels.md` | `stage-pieces` |
| `model/wgrad.py` flush_iter, gather_dtypes, 15 tests | partial: its tests pass | `model-flush-iter.md` | `model-flush-iter` |
| `transport/shim.py` SharedBudget, 15 tests | partial: its tests pass | `transport-shared-pacers.md` | `transport-shared-pacers` |
| `resources/node.py`, `probe.py`, capacities tests | partial: reviewed once, amend applied | `fleet-capacities.md` | `fleet-capacities` |
| `scheduler/exchange_sim.py`, 16 `exchange_sim` tests in `test_zero_bubble.py` | partial: 14 pass, 2 red (SC-4) | `i17-exchange-sim.md` | `scheduler-pieces-plan` |

All of it is saved in commit 4201c65. Only r14 and the fleet amend have been reviewed. Each task's coder starts from
that code and its journal, and finishes and reviews it. The ledger lists `r14-lane-order` as done. That id is no longer
a task, so a carry takes nothing from it, and the merged task re-runs r14's tests.

## Preconditions before wave 1 (wan-implement does not start until each is closed)

| Id | What | State | Check |
|---|---|---|---|
| P-M2 | M2 is merged into main, and main is merged into `factory/m2b`. The merged specs carry RUNTIME-I15, RUNTIME-I16, CONTROL-I21 and the sealed TRANSPORT-I7. | Open: the orchestrator's (the build waits for M2 green) | `grep -q "RUNTIME-I16\." docs/design/SPEC-runtime.md && grep -q "CONTROL-I21\." docs/design/SPEC-control.md && grep -q "class SecureSocket" src/wantrain/transport/secure.py && grep -q "def _maybe_rebalance" src/wantrain/coordinator/coordinator.py` |
| SC-1 | RUNTIME-I12 amended: the manifest is embedded in every piece file. | Applied | `stage-pieces` verify greps it |
| SC-2 | SCHEDULER-I19 defined (renumbered from I18, D-1) | Applied | `scheduler-pieces-plan` verify greps it |
| SC-3 | E1's integration arms replace the ADR's fp16/fp32 identity arms | Ruled by the human | GATES entry `sc3-identity-recipe-arms-e1` |
| P3 | `docs/experiments/m2b-e4/PLAN.json` and `m2b-e7/PLAN.json` re-pinned and granted | Granted 2026-10-09 | `approve.mjs check` in `scheduler-pieces-plan` and `e4-exposure` |
| **SC-4** | **I17 anchor (from the stopped run).** For the stage-0 "step 1 alone" and "LPT" arms, the simulation gives 76.34% and 23.28% of C. The pinned values are 71.31% and 20.09%, which is +7.05% and +15.9% relative, outside the 5% bound. The other five arms match within 0.2%. Cause: derivation §2's analytic `V_max/cap + t_gap − C` lets the hot owner's gather egress overlap its gradient ingress. I17 and the runtime release a unit's gathers only after the owner's last fold. With that serial order, the reference model gives 76.00% and 22.62%. This is timing only: no bit moves. | **Open: the researcher's, before wave 1** | The researcher amends derivation §2 for owners whose gathers wait on their ingress. The researcher re-pins the two arms and F6's thresholds in `m2b-e4/PLAN.json` and records "SC-4" in its `derivation`. Autonomy covers local plans. Check: `python3 -c "import json;d=json.load(open('docs/experiments/m2b-e4/PLAN.json'));assert 'SC-4' in json.dumps(d['derivation'])"` and `node agents-architetura/tools/factory/approve.mjs check docs/experiments/m2b-e4/PLAN.json`. The test reads the values from the PLAN and is never edited. F6's margins only grow. |
| **SC-5** | **k on the sealed data plane (new, review round 1).** SCHEDULER-I16 defines k on host DRAM as "the transport's counted passes per side (10 today)". M2 seals every data-plane socket. HANDOUT-M2 says sealing adds about 2 host-DRAM passes per byte per side ("M2b's S1 must count it"), so 10 is no longer the transport's count. Admission only: no bit moves. | **Open: the researcher's, before wave 1** | The researcher counts the passes per side on the merged tree, sealing included, and amends SCHEDULER-I16's k clause through `freeze.mjs`, with a `SPEC-scheduler-history.md` entry "SC-5" giving the number and how it was counted. Check: `grep -q "SC-5" docs/design/SPEC-scheduler-history.md`. The E7 scenarios and S1 anchors keep their explicit k (they are inputs). The product's k is `admission.K_HOST_DRAM_COUNTED`, and a test asserts it equals the amended figure. |

## What the mechanism moves (ADR-0006 §Mathematical Model)

`T_step = C·(1 + β) + E − window_s + T_ovh`.

| Mechanism | Term it moves | Effect at the ADR's points | When the gain vanishes |
|---|---|---|---|
| Step 1 | E: native bytes, streamed per module, copies off the compute thread | DERIVED: B 2163.3 → 1721.3 ms; interior A 531.7 → 84.5 ms | fp32 recipe (no byte saving) |
| Step 2 | E and per-peer state: pieces spread every unit over all w links | DERIVED: A 84.5 → 18.8 ms; B stage 0 needs k ≤ 6 (R10, deferred), so at k 10 B gets step 1 | When S1 fails on any link, host DRAM included. With sealing, k rises toward 12 (SC-5), and S1 refuses more hosts |
| R14 | β only: lane-consistent orders | DERIVED: 109.4% → 43.8% bubble at S 8, w 4, M 32 | — |

No row changes a bit. Every number is DERIVED (`docs/research/adr-0006/`). On this machine, measured numbers are
SIMULATED, and real hardware is pending.

The replan moves none of these terms. It changes the run's wall time: 13 serial waves become 2. The critical path is
`stage-pieces` (the largest wave-1 task) plus `arm-pieces`.

## Tasks

| # | id | Wave | Workstream | Risk | Delivers | Experiment / falsifier |
|---|---|---|---|---|---|---|
| 1 | `model-flush-iter` | 1 | model | high | `wgrad.flush_iter`, `gather_dtypes` (MODEL-I7) | Bit-equality against `flush`, both orders |
| 2 | `transport-shared-pacers` | 1 | transport | high | Shared egress, ingress and uplink budgets (TRANSPORT-I8) | Aggregate within 5%; bytes unchanged |
| 3 | `fleet-capacities` | 1 | control | high | Probed host RAM, host-DRAM and PCIe, each with its source; validated at HELLO | A missing or declared value refuses, never assumed |
| 4 | `stage-pieces` | 1 | runtime | high | Everything in the runtime: piece kernels; JobSpec fields and refusals; `exchange.py`; `logical_ckpt.py`; steps 1 and 2 in `stage.py`; logical resume (zero1 ↔ pieces, rebalance owners); exact OBJ exchange bodies; telemetry and holds; `run_local` class check, budget files, jitter seed; identity manifest (RUNTIME-I10..I13) | **E6**, **E1**, **E2**, **E3**, **E5**; the F1/F8 tests |
| 5 | `scheduler-pieces-plan` | 1 | scheduler | medium | r14 kept green; I17 simulation; I16 admission with counted k; plan mode for wide and sharded stages with `allow_pieces` (SCHEDULER-I16, I17, I19) | **E7** (F9); I17 anchor (after SC-4) |
| 6 | `heal-class-tripwire` | 1 | control | high | Heal only by an identical probed model (CONTROL-I9). Pieces and wire `native` are refused at the coordinator, and pieces is refused on `wantrain stage` | First leg of the tripwire |
| 7 | `e4-exposure` | 2 | runtime | medium | `validation/exposure.py`: E4 at the scaled B′ against the granted PLAN; both emission orders; RTT arms; a tree harness | **E4** (F3, F5, F6), SIMULATED |
| 8 | `arm-pieces` | 2 | control | high | Coordinator plan mode on probed inputs: per-peer budgets; I16 re-check after re-weighting and before every rebalance; spec-mode pieces checks; the planner's spec fields round trip. Then, gated by F1/F8, it lifts the refusal, and FEATURES_ON gains native, pieces and w 3 | **F1**, **F8** gate; identity, wan, chaos, pieces heal, pieces rebalance |

Eight tasks fit one wan-implement run, so there are no later slices.

## Wave order

| Wave | Tasks (parallel, disjoint paths) | Why |
|---|---|---|
| 1 | `model-flush-iter`, `transport-shared-pacers`, `fleet-capacities`, `stage-pieces`, `scheduler-pieces-plan`, `heal-class-tripwire` | Each owns its files alone. Each imports from other workstreams only frozen interfaces on the tree (§Interfaces). Its verify runs only its own files' tests (D-10) |
| 2 | `arm-pieces` (deps: every wave-1 task), `e4-exposure` (deps: `stage-pieces`, `scheduler-pieces-plan`, `transport-shared-pacers`) | Integration. `arm-pieces` is the second owner of `coordinator.py` and `test_control_admission.py`, after `heal-class-tripwire`. `e4-exposure` is the second owner of `test_engine_bitwise.py`, after `stage-pieces` |

The machine check (`checkMilestonePlan`, `maxWaves` 2) passes with these two waves, and no two concurrent tasks share a
path. FACT: the planner ran the check locally on this file, and it exited 0 with exactly these waves.

## Old to new (no work lost)

| Old task (13-wave plan) | New task | What moved |
|---|---|---|
| `r14-lane-order` | `scheduler-pieces-plan` | Done code, not rewritten. Its lane_order tests are re-run after P-M2, and its acceptance is re-done only if the M2 merge turned them red. It is merged because a separate r14 could not own `test_zero_bubble.py` beside the i17 tests, and `scheduler-pieces-plan` consumes its orders (review round 1) |
| `e6-piece-kernels`, `spec-fields` | `stage-pieces` | Whole. `consumer_dtype` folds into `gather_dtypes` |
| `wire-native`, `pieces-core`, `pieces-logical-resume`, `pieces-identity-e1-e2`, `pieces-holds-e3` | `stage-pieces` | Whole. The queue, chunks, validator, copy engine, hold counters, manifest and resharding, and their unit tests, are part of the same task (review round 1: a separate library task in the same wave was a dependency the DAG could not express) |
| `falsify-f1-f8` | `stage-pieces` writes `f1_rotation_a/b`; `arm-pieces` runs F1 and E3 ×3 as its gate | F1/F8 are still decided before the refusal is lifted |
| `i17-exchange-sim`, `i16-admission`, `plan-wide-e7` | `scheduler-pieces-plan` | Whole. Its JSON round trip of the spec fields moves to `arm-pieces` (D-9) |
| `heal-class-tripwire` | `heal-class-tripwire` | The shard is read without the spec field (I-9) |
| `plan-wide-coordinator`, `arm-pieces` | `arm-pieces` | Whole. `plan_wide_unarmed` stays, with the flag patched off (D-6) |
| `e4-exposure` | `e4-exposure` | Unchanged, wave 2 (D-3) |
| `model-flush-iter`, `transport-shared-pacers`, `fleet-capacities` | same ids | Unchanged. Their interfaces are now frozen for same-wave readers |

## Interfaces

**Between wave-1 tasks: only symbols already on the tree.** A wave-1 coder imports these as they are at 4201c65. The
owning task finishes them in the same wave, but it never changes a signature, a key or the bits they produce. A change
that would need one is recorded as a spec conflict that names both sides; no adapter is ever added. Each reader's
first verify line greps the symbols, so it holds from the start of the wave.

**I-2 model** (`model-flush-iter`, read by `stage-pieces`; `model/wgrad.py`):
- `FLUSH_ORDERS = ("store", "forward")`.
- `flush_iter(store, order) -> Iterator[(param, grad)]` takes the store over.
  - A single-contribution grad comes in its GEMM dtype.
  - A multi-contribution grad is widened and summed exactly as `flush` does.
- `gather_dtypes(module, compute_dtype: str, autocast: bool) -> dict[id(param), torch.dtype]`.

**I-3 shim** (`transport-shared-pacers`, read by `stage-pieces` and, in wave 2, by `e4-exposure`; `transport/shim.py`):
- `SharedBudget(mbps, path=None, burst_ms=10.0)` with `reserve(nbytes) -> float`, `pace(nbytes)` and `close()`.
  - It is picklable by path.
  - A named path must be fresh per run.
- `LinkSpec.egress`, `.uplink` and `.ingress` are `SharedBudget | None`, default None.
- `DelayedDelivery.push(item, nbytes=None)` requires nbytes when an ingress budget is set.

**I-4 capacities** (`fleet-capacities`, read by `scheduler-pieces-plan` and, in wave 2, by `arm-pieces`;
`resources/node.py`):
- NodeInfo gains `host_ram_bytes: int | None` and `host_dram_gbps`, `pcie_h2d_gbps`, `pcie_d2h_gbps: float | None`.
- The probe keys are `hostRamBytes`, `hostDramGBps`, `pcieH2dGBps` and `pcieD2hGBps`.
- `capacity_source(info, name) -> "measured" | "declared" | "missing" | "absent"`.

**I-9 refusal** (`heal-class-tripwire`, read by `arm-pieces` in wave 2):
- `coordinator.PIECES_ARMED = False`.
- `run_job` refuses shard `pieces` and wire `native` (review round 2). It reads them as follows, without waiting for the
  JobSpec fields:
  - spec mode: `getattr(job.spec, "shard", "zero1")` and `getattr(job.spec, "wire", "fp32")`;
  - plan mode: `(job.plan or {}).get("shard")` and `.get("wire")`.
- Its reason contains `not armed`.
- `wantrain stage` reads `"shard"` from the spec JSON before `JobSpec.load`. So does `stage_main.main` (BA-58).

**I-10 zero_bubble** (`scheduler-pieces-plan`, called by `stage-pieces`' `finalize` and `stage.py` in the same wave;
new in review round 2):
- The signatures of `solve`, `schedule_for_spec`, `default_schedule`, `analytic_costs` and `simulate_slots` are those
  of the merged tree.
- A scalar `mem_budgets` and every width-1 pipeline get byte-identical orders to today's.
- Outputs are deterministic.
- An order moves timing, never bits (RUNTIME-I2). The proof under the new orders is `arm-pieces`' `-k "wide or
  solved_schedules"` line in wave 2.

**I-7 JobSpec fields, read by the planner without waiting for them** (`stage-pieces` writes `runtime/spec.py`):

| Field | Values and default |
|---|---|
| `wire` | `"fp32"` (default) or `"native"` |
| `shard` | `"zero1"` (default) or `"pieces"`, job-wide |
| `emission_order` | per stage, `"store"` (default) or `"forward"`. `"store"` is exchange_sim's `"backward"` |
| `piece_bounds` | None, or per stage None or `{name: [b_0..b_w]}`. `name` is the stage chunk's `named_parameters()` name, the same `"<layer index in chunk>.<param name>"` that `exchange_units` uses. The bounds lie on multiples of `pieces.quantum(shape)` (`lcm(row length, ALIGN = 128)`), with `b_0 = 0` and `b_w = numel` |
| `pacing` | `"link"` (default) or `"shared"` |
| `peer_egress_mbps`, `peer_ingress_mbps` | `[stage][peer]` |
| `sites` | `[stage][peer]` |
| `uplink_mbps` | `{"<site>-><site>": mbps}` |

None of these fields is a pin or enters `recipe_digest` (BA-53).

**The finalize contract (frozen; review round 2).** `build_spec` and `plan_job` call `finalize()`, and `stage-pieces`
rewrites `finalize` in the same wave. `finalize` must therefore accept every planner output below, and the planner emits
nothing else:
- shard `pieces` only with wire `native` and `stream_grads: true`;
- wire `native` on a plan with any wide stage only with `stream_grads: true`;
- `piece_bounds` always None in M2b. The planner prices the equal layout. Explicit bounds are a runtime and E4 input
  (LPT, E5), never a planner output;
- pieces only on a stage whose peers share the device type and the probed model.
- never pieces on a `tie_embeddings` recipe whose Embed and Head lie on different stages (BA-42; final check).

Each side tests its half of the contract:
- `stage-pieces`: `spec_m2b_planner_contract` finalizes one spec per combination above.
- `scheduler-pieces-plan`: `plan_wide_finalize_contract` asserts the planner emits only these combinations. Its other
  M2b assertions read `plan_job`'s plan dict and the spec attributes, so they hold whichever `finalize` is on disk.
- `arm-pieces`: `plan_wide_spec_fields`, in wave 2, runs `build_spec(allow_pieces=True)` through the real `finalize`
  and checks the JSON round trip (D-9).

The planner sets the fields by attribute assignment, as `build_spec` already sets `out_dir` and `tag`, never as
constructor keywords. A change to the contract is a recorded conflict naming both tasks.

**I-8 planner** (`scheduler-pieces-plan`, read by `arm-pieces` in wave 2):
- `plan_job(..., allow_pieces=False, shard=None)` and `build_spec(fleet, allow_pieces=False)`.
- Fleet node keys: `hostname`, `site`, `device`, `gpu_name`, `declared`, `host_ram_bytes`, `host_dram_gbps`,
  `pcie_h2d_gbps`, `pcie_d2h_gbps`, `nic_up_gbps`, `nic_down_gbps`, and `sources: {key: source}`.
- Fleet keys:
  - `k_passes: {"value": admission.K_HOST_DRAM_COUNTED, "source": "counted"}` (SC-5);
  - `job.shard`.
- The result adds:
  - `stage_widths`, the [stage][peer] placement, `shard`, `wire`, `emission_order`, `piece_bounds` (None) and
    `stream_grads`;
  - a rationale that lists every refused candidate and its reason. With `allow_pieces` False, every pieces or native
    candidate is listed as `not armed`.
- `admission.recheck(..., pinned_bytes_per_peer=None)` re-checks device memory, MEM-HOST and S2 for a given mb_owner and
  orders.
  - The pinned copy bytes are an input: 0 on cpu and mps.
  - On CUDA, `arm-pieces` passes the copy engine's bound, so admission never imports the runtime written in wave 1.
- `admission.check_class(peers)` applies the probed-class rule only.
- `admission.K_HOST_DRAM_COUNTED` is the counted k.

**Contracts inside `stage-pieces`** (one task, so no other task depends on them; written down for E4 and review):
- **I-1 kernels (`runtime/pieces.py`):**
  - `ALIGN = 128`; `quantum(shape) = lcm(row_length(shape), 128)`.
  - `piece_layout(shapes, w, bounds=None)` returns flat offsets, with `b_0 = 0` and `b_w = numel`.
  - `piece(t, b, k)` is a flat view.
  - `exchange_units(chunk)` lists units in forward order. Names are `"<layer index in chunk>.<param name>"`. A tied
    parameter (tie_embeddings) is in the Embed's unit only (BA-42).
  - `cast_piece`, `fold_start`, `fold_add`, `fold`; `adamw(params, lr)` with betas (0.9, 0.95), eps 1e-8 and
    weight_decay 0; `step_unit`.
  - `consumer_dtype` is removed.
- **I-5 (`runtime/exchange.py`):**
  - `NeedKey(fwd_unit, kind, step, mb, seq)`, with kind GATHER 0 and GRAD 1.
  - `NeedFirstQueue.push(key | None, item)`, `pop(timeout=None)` and `qsize()`. A None key sorts last, FIFO among
    such keys.
  - `chunk_object(unit, body, max_bytes=16 << 20) -> [(env, view)]`; `reassemble(parts)`.
  - `FrameError(ValueError).cls`; `check_frame(kind, env, tensors, expect)`.
  - `CopyEngine(device).submit(fn) -> Future`, `thread_ident`, `close()`.
  - `HoldCounters.add/sub/end_step/line`, which prints `WT_HOLD <step> host_grads=… gathers_out=… gathers_in=…
    device_own=…`.
- **I-6 (`runtime/logical_ckpt.py`):**
  - The pieces payload is the wave's `state.pt`, with keys `step`, `pieces: {name: {master, exp_avg, exp_avg_sq,
    step}}`, `hparams`, `rng`, `losses`, `nontrainable`, `pins` and `pieces_manifest`.
  - `build_manifest`, `check_manifest`, `select_wave(read_manifest, peers_present) -> (w_old, layout)`, `assemble(payloads)`,
    `owned_pieces`, `as_zero1`, `from_zero1(payloads, names, w)`.
  - `select_wave(read_manifest, peers_present)`: `peers_present` lists the stage's peer indices with a wave at the
    resume step, so an extra index (>= w_old, naming that step and width w_old) is refused (RUNTIME-I12; final check).
- **Seams and lines (read by `e4-exposure` and `arm-pieces` in wave 2):**
  - `launch.local.jitter_seed_from_env() -> int` prints `E1_SEED=<n>`.
  - `WANTRAIN_TEST_QUEUE=fifo`, `WANTRAIN_TEST_BARRIER=always` and `WANTRAIN_TEST_UPDATES=between_ops` change timing
    only.
  - One `WT_HOLD` line per step.

## Binding audit (PLAN-M2b.json `binding`, BA-1..BA-58)

The rulings of BA-1..BA-55 stand, except these:
- the four that review round 1 rewrote, listed below;
- review round 2 rewrote BA-49 and added the resume step to BA-51 and BA-54.

Every `where` cites the M2 that P-M2 merges by symbol. The rulings that shape the design:
- **Fold and RNG.**
  - The oracle's fold (`reference.py:85-88`) and `OrderedAccumulator` are never changed.
  - Under `wire: native` on a wide stage, every fold input is widened to fp32 before `add`.
  - A width-1 stage keeps `wgrad.flush`. Under `pieces`, every stage folds through the piece kernel (BA-7, BA-8,
    BA-23, BA-46).
  - No path draws from the torch RNG (BA-22).
  - **Tied weights (BA-42, final check).** Under `tie_embeddings` the Head's `out.weight` is the Embed's `tok.weight`.
    It is one parameter in the Embed's unit; autograd sums both contributions in B, so it ships when W(mb) starts.
    `finalize` refuses pieces when the tied Embed and Head lie on different stages. Test `e1_core_tied`.
- **Wire framing and encoding.**
  - Unit, offset and length travel in the OBJ envelope, and the frame header is unchanged (BA-40).
  - Each chunk of at most 16 MiB is self-contained (BA-13).
  - LOSS frames and the sentinel sort after every exchange item (BA-14).
  - **BA-56 (new):** GRADS and PARAMS chunks are OBJ bodies only, the exact byte image through `serial.dumps`. They
    never reach `_encode_body`'s `planes` or `fp8_planes` branch, which would narrow an fp32 piece to the boundary
    dtype. Test: `pieces_codec_exact`, with a must-fail control.
- **Sealed data plane (BA-16, rewritten).** M2 authenticates and seals every chain and mesh socket. Pieces adds no
  listener and no socket.
  - A 16 MiB chunk is cut into ≤ 1 MiB AEAD records below the frame layer, so BA-13 and BA-40 hold unchanged.
  - Shared budgets pace the frame bytes handed to the sealed socket, at the point where M2's per-connection pacer does.
  - The record overhead is not paced separately. HYPOTHESIS: about 20 B per MiB, 0.002%.
  - The seal/open passes are part of k (SC-5).
  - M2-P1-MESH-AUTH is closed by M2.
- **Checkpoint (BA-18, BA-19, BA-51, BA-54, BA-57).**
  - The pieces payload is the wave's `state.pt`, written by M2's `commit_wave`. The embedded manifest names the step,
    the width, the peer and the layout, and `wave.json`'s sha256 binds it (SC-1).
  - A resume at another width reads p0's manifest, then exactly p0..p(w_old − 1) at the resume step, through
    `validate_wave`. It does so before the slot's own `_drop_abandoned_waves`.
  - **Extra index (final check, RUNTIME-I12).** A slot with index >= w_old whose wave at the resume step names that
    step and width w_old is refused as an extra index (`e5_refuse_extra_index`). One naming another width belongs to an
    older wave and is never read as state.
  - It merges loss histories, and it refuses a gap, a conflict or differing replicas (BA-19).
  - **Resume step (review round 2).** Every slot of a pieces stage takes the same resume step, from one of two sources:
    - `spec.resume_step`;
    - otherwise, old slot p0's newest complete wave.

    It never uses its own `slot_waves`. M2's `_resume` raises on an empty `slot_waves`, so a resume that raises the
    width (2 → 4: slots p2 and p3 have no wave) would otherwise fail. Tests: `e5_widen`, plus the raised-width arm of
    `e5_zero1_to_pieces` (zero1 w2 → pieces w3). The F1 rotation gains a 2 → 4 resume.
  - **BA-57 (new): zero1 → pieces.**
    - The payload kind is decided by the `pieces` key. A pieces payload without its manifest is refused.
    - w_old comes from p0's pinned `zero1_widths`.
    - Every old slot is read through `validate_wave`, and a wave that mixes kinds is refused.
    - `from_zero1`, then `owned_pieces` at any width.
- **Pins (BA-53, rewritten).** `pins()` is unchanged, and the M2b fields are not pins. `_check_resume_pins` skips only
  `zero1_widths`, and only in two cases: for a pieces payload, and for a zero1 wave read logically by a pieces resume.
  A zero1 shard is never read as a shard at another width. The must-fail control is `e5_refuse_zero1_width`.
  `stage-pieces`' acceptance item now states both cases; review round 2 found that it stated only the first.
- **Frame validation (BA-12).** `check_frame` guards both the GRADS and the PARAMS branch of `_pump`. `pieces_frame`
  injects a bad GRADS frame, a PARAMS gather of the wrong dtype and a PARAMS gather of step t+2. Its control validates
  GRADS only and lets the bad gather through.
- **Re-plan (BA-44, rewritten).** M2 has a between-step re-plan: the rebalance loop changes `mb_owner` and
  `peer_weights` at a wave by a restart in place. For a run with a pieces stage:
  - before applying a rebalance, the coordinator runs `admission.check_class` and `admission.recheck` (device memory,
    MEM-HOST, S2) with the new ownership;
  - a failure skips the rebalance, with the reason recorded, and never relaunches;
  - the resume keeps the layout and is byte-identical (`e5_rebalance_owner`, `pieces_rebalance`).

  Width and split change only at a checkpoint resume (RUNTIME-I12).
- **Budget files (BA-55).** Only `run_local` creates budget files: fresh per run, under `out_dir`, for
  `pacing: shared`.
- **Telemetry under pieces (BA-45).** The events of step t+1 stay queued for their own summary (RUNTIME-I9).
- **Admission inputs and re-checks (BA-47, BA-48, BA-50).**
  - Inputs are probed capacities, NIC from the link probes, one RAM and one DRAM figure per host, and k =
    `K_HOST_DRAM_COUNTED`, counted (BA-48, rewritten for SC-5).
  - A missing input refuses the wide candidate that needs it.
  - Pieces stages are re-checked after the profiled re-weighting, and a spec-mode pieces job gets the class and memory
    checks.
- **Lane budgets (BA-24, BA-29).** A lane solve gets its own peer's budget, never the stage's sum.
- **Device class (BA-31, BA-52, BA-58).**
  - `JobSpec.finalize` refuses mixed device types, and `run_local` refuses mixed device models.
  - `wantrain stage` refuses pieces.
  - **BA-58 (new, review round 2).** `python -m wantrain.worker.stage_main` is a third one-slot entry point. `run_local`
    and the agent launch every slot through it, so the refusal cannot live in `run_stage`.
    - `stage_main.main` refuses pieces unless it gets `--launcher local` or `--launcher coordinator`. `run_local`
      passes `local` after its class check. The agent passes `coordinator`; `arm-pieces` adds that flag.
    - The flag is not an authentication: the operator of a host is trusted (THREAT_MODEL). It removes the accidental
      bypass. Cost: one JSON read per launch, nothing per step.
  - A declared class is refused for pieces.
- **Batch exchange refused (BA-6).** `wire: native` with `stream_grads: false` on a wide stage is refused.
- **Liveness (BA-10, BA-41).** `WT_STEP` stays strictly increasing.

## Tripwire `pieces-unarmed`

A pieces run with only some legs in is silently wrong. A heal onto another device model differs in bits, and an
old-format checkpoint resumes wrong. Wire `native` is the same (review round 2). Its bits depend on three tasks
together: `flush_iter` (`model-flush-iter`), the W pass and fp32 widening (`stage-pieces`), and the planner's native
candidates (`scheduler-pieces-plan`). If the run stops after wave 1, a spec-mode zero1 + native job must not reach
production.
- **Refusal.** While `PIECES_ARMED` is False:
  - `Coordinator.run_job` refuses `shard: "pieces"` and `wire: "native"`;
  - the planner refuses pieces and native candidates as `not armed` unless `allow_pieces` is set.

  `heal-class-tripwire` adds the refusal.
- **Legs:** `heal-class-tripwire` (the first leg), `model-flush-iter`, `stage-pieces` and `scheduler-pieces-plan`. All
  four are in wave 1.
- **Arm:** `arm-pieces`, which depends on every leg. It lifts the refusal only after the following pass on the tree
  where every leg has landed:
  - F1 and F8;
  - `e1_native_` (per recipe) and `native_width1`.
- **Test:** `uv run pytest -q -p no:randomly tests/unit/test_control_admission.py -k pieces_unarmed`. It is the last line
  of every leg's verify, which the machine check requires (`plan.js` checkTripwires).
- **Wave-1 rule (D-7, rewritten in review round 1).** The line counts as evidence only when it exits 0.
  - Exit 5 (not collected) means the first leg has not landed yet. The coder runs the line again as its last command.
    If it still exits 5, the journal records "NOT RUN: first leg not landed".
  - That line is never cited as passed, and never as "pending".
  - The proof that the refusal holds over every leg comes from two places: `arm-pieces`' verify, before it arms, and
    the ladder at the end of the run.
- **After arming.** `pieces_unarmed` stays, with `PIECES_ARMED` patched off. The tripwire's selector therefore keeps
  collecting and proving the refusal path; it never exits 5 after the arm.
- **Local runs.** `run_local` stays open for the experiments, with the class rule at its boundary (BA-52).
  `wantrain stage` and a bare `stage_main` refuse pieces permanently (BA-58).

## Deviations (recorded; the human sees them with the plan)

- **D-1, invariant id (merge of main 50d34cd).** ADR-0007 took SCHEDULER-I18. M2b's lane-order invariant (SC-2) is
  SCHEDULER-I19, with the same text. The tests' `covers:` lines and this plan follow. No M2b amendment contradicts
  ADR-0007's.
- **D-2, experiments gate the arm, not the writing.** With 2 waves, all code is written in wave 1. What is gated is
  production use:
  - E6, E1, E2, E3 and E5 (`stage-pieces`) and E7 (`scheduler-pieces-plan`) are acceptance of their tasks;
  - F1 and F8 run again on the integrated tree as `arm-pieces`' gate, before the refusal is lifted.

  A refuted experiment can now waste code, but it can never arm a wrong tree.
- **D-3, E4 beside the arm.** `e4-exposure` and `arm-pieces` are both wave-2 tasks, so the arm cannot depend on E4.
  - E4 checks the timing model behind S2. That is performance admission, and it moves no bit.
  - A fired F3, F5 or F6 is recorded as a spec conflict. GATES entry `e4-exposure-sweep` keeps closeout NOT VERIFIED
    until it is resolved.
  - **E4 on the Mac: measure, do not block (the human's decision, 2026-10-10 ~19:45, `--by HANDOUT-M2b`; as the AEAD
    cost and the rebalance prediction in M2).** While `hardware.json` says gpu false (SIMULATED), F3's band
    [0.8, 1.2]x is MEASURED: every arm prints predicted / measured / ratio, the gate requires those lines, and a ratio
    outside the band does not fail. Arms this Mac cannot put on the PLAN's configuration (interior w8,
    M2B-E4-W8-OFFCONFIG) print "not evidence" and do not fail on the Mac. Still enforced, and failing: every arm's
    manifest equal to the reference (byte identity), F6 (pieces beat the alternatives), F5 (tree vs left fold) and
    the emission order at RTT_hi. No skip can turn the gate green: it still accepts only "N passed". With gpu true
    (M1) F3's band and the w8 configuration fail again as before. The E4 PLAN's predictions and criteria are
    unchanged and the falsification stays recorded: M2B-E4-F3 (interior w4 barrier control 1.296x, earlier trees
    1.273x and 1.265x; interior w4 backward emission at RTT_hi 1.232x), M2B-E4-W8-OFFCONFIG (F share 0.04-0.076
    against 0.1065 +-25%) and M2B-E4-TODAY-STAGE0 (12.4% against 131.4% simulated) stay open, deferred to M1 (GPU).
- **D-6, campaigns (the human's decision, 2026-10-10 ~15:40, `--by HANDOUT-M2b`; as in M2).** The M2b closeout runs
  without the `amplify-chaos` and `load-chaos` campaigns; `py-devmode` stays. Why: on this 16 GB Mac, shared with the
  M2 build, those two campaigns do not fit the closeout (load-chaos still needs factory PR #48 for its per-round
  timeout). Both stay pending, to run before the M1 sign-off together with M2's.
- **D-7, audit (the orchestrating session's decision, 2026-10-11 ~09:45, delegated by the human on 2026-10-10 ~23:40
  "decida as coisas", to be confirmed by the human; `--by HANDOUT-M2b`; as D-6).** The M2b closeout runs without the
  `audit` entry. Why: `docs/security/AUDIT-RESULTS.json` has never existed; the mechanism-class audit (`/wan-audit`,
  malicious-worker) is planned before any rented machine joins, so the entry could only fail on a missing file, not on
  M2b's code. It stays pending, to run before the M1 sign-off together with D-6's campaigns.
- **D-4, test files.** The rule asks for a new test file per task. The DESIGN's frozen workstream ownership lists only
  named test files, so a new file would fail the machine check.
  - Instead, each test file has exactly one owner per wave, which is the purpose of the rule.
  - **Recommendation:** amend the workstreams block so each workstream owns a test-file prefix.
- **D-5, r14 merged (rewritten in review round 1).** `r14-lane-order` is no longer a task of its own; the focus allowed
  this. A separate r14 could not own `test_zero_bubble.py`, which holds both its tests and the i17 tests.
  `scheduler-pieces-plan` also consumes its orders, and an edge between the two would have made 3 waves. The done
  code is the starting point and is re-verified, not rewritten.
- **D-6, plan mode merged into the arm.** `plan_wide_unarmed` is kept, with `PIECES_ARMED` patched off.
  - It and `plan_wide_armed` are direct calls in `tests/unit/test_control_admission.py`, not in `test_coordinator.py`
    (review round 2).
  - GATES-M2b entry `pieces-armed` ends with `! grep -n 'not armed' tests/integration/test_coordinator.py`. That grep
    stays valid without hiding the literal; `arm-pieces`' verify runs the same grep.
- **D-7, tripwire legs in one wave.** See §Tripwire.
- **D-8, base tree (new).** The focus named `origin/factory/m2` (3ebf48d). The M2 that will merge has more: the sealed
  data plane, the rebalance loop, and functions at other lines. This plan follows the M2 that merges (P-M2's check),
  cites it by symbol, and rules the sealed plane (BA-16) and the rebalance (BA-44) as they will exist.
- **D-9, where the spec-field round trip is tested (new).** The planner sets the JobSpec fields by attribute in wave 1.
  The JSON round trip of its output (`plan_wide_spec_fields`) is asserted by `arm-pieces` in wave 2. No SPEC text
  changes: `build_spec` still emits the fields.
  - Review round 2 added the frozen finalize contract (I-7), which both wave-1 sides test.
  - It also fixed one point: the planner emits `piece_bounds` None in M2b.
- **D-10, integrated-tree evidence moves to wave 2 (new, review round 2).** These checks used to run in wave 1; they
  now run in `arm-pieces`' verify:
  - RUNTIME-I2 under the scheduler's orders;
  - the MODEL-I1..I3 deferred-dW tests in `test_zero_bubble.py`;
  - the coordinator integration and chaos selections of `heal-class-tripwire`.

  Each one ran a file a sibling edits in the same wave. Their owners keep the acceptance; only where it is evidenced
  moves.
- **D-11, the tripwire covers wire `native` (new, review round 2).** No SPEC text changes. Wire `native` reaches
  production only through the arm, like pieces.
  - `model-flush-iter` becomes a leg.
  - E7's native choices are priced with `allow_pieces` True.
- **D-12, timing lines take the whole machine (new, review round 2).** E3, E4, the shared-pacer tests,
  `pieces_rebalance`, F1 and `holds_seams` run under `machine-lock.mjs run` without `--slot`, with `--timeout 1200`, and
  the command quoted as one string. That is the "alone" the risks list promised; it is now written in the verify
  lines. Every other line runs in a slot, as the factory requires.

## Hardware mode: gpu false

Everything runs on the CPU and localhost. Timing is SIMULATED next to the model's prediction.
- The CUDA copy-stream path and the PCIe probe are built but not exercised. F7 (compute-thread stalls, copy latency,
  host-DRAM passes) is real-hardware pending, never passed.
- On CPU, the RUNTIME-I13 copy bullet is proven by thread identity.
- No Vast.ai.

## Bar to freeze in GATES-M2b.json (input for the gate step)

The current GATES-M2b.json stays valid with this plan, because every selector below is still produced:
1. The base rungs twice in a row, plus `oracle`, `identity`, `chaos` and `wan`.
2. `f1-matrix-x20`: `test_engine_bitwise.py -k "e1_ or e5_ or native"`, ×20, alone under the machine lock, fresh
   printed seeds (`stage-pieces`). This now also selects `e5_zero1_to_pieces` and `e5_rebalance_owner`.
3. `e3-holds-liveness-x3`: `-k e3_` ×3 (`stage-pieces`).
4. `e4-exposure-sweep`: `-k e4_`, alone (`e4-exposure`).
5. `e6-exhaustive-cast`: `test_runtime_spec.py -k e6_` (`stage-pieces`).
6. `pieces-armed`: `test_control_admission.py -k pieces_armed` and `test_coordinator.py -k "pieces or
   plan_wide_armed"` (`arm-pieces`). This also selects `pieces_rebalance`.
7. `no-invariant-deferred-to-m2b`.
8. `control-i9-pieces-heal`: `test_fault_tolerance.py -k pieces_heal` (`heal-class-tripwire`).
9. `sc3-identity-recipe-arms-e1`: `-k "e1_ and fp16"` and `-k "e1_ and fp32"` (`stage-pieces`).
10. The amplified and loaded concurrency entries:
    - `test_runtime_spec.py -k "need_first or copy_engine or holds"` (`stage-pieces`);
    - `test_transport.py -k shared_pacer`;
    - `test_fault_tolerance.py -k pieces_heal`;
    - E3 loaded.

Suggested addition for the gate step: `pieces_codec_exact` ×20 under load, beside entry 2. Today only the base
`integration` rung selects it.

Notes for the gate step (review round 2; the planner does not own GATES-M2b.json):
- **`bench` rung: N/A, with the reason.** DESIGN §8 lists `bench` for M2b. `bench/suite.json` does not exist, and
  the hardware switch is gpu false. The SIMULATED E3 and E4 arms stand in, labelled SIMULATED, and real-hardware
  validation stays pending. The rung is recorded N/A; it is not silently dropped.
- **Entry 6, `pieces-armed`.** It stays valid: no test in `test_coordinator.py` contains `not armed` (D-6).
- **Entry 2, `f1-matrix-x20`.** It runs under the exclusive lock (no slot, so no 1200 s ceiling) with `timeoutSec`
  7200. Its selection now also takes `e5_widen` and the `e1_matrix_<recipe>_*` arms.
  - HYPOTHESIS: about 140 arms at about 20 s each come to about 47 min per repetition.
  - The gate step should either split it by recipe or raise `timeoutSec` per repetition. It must never shrink the
    selection.

At closeout the researcher removes the `DEFERRED-TO: M2b` markers through `freeze.mjs`.

## Risks, gaps and questions (facts separated from hypotheses)

1. **Size of `stage-pieces` (FACT: 85 acceptance items, 45 verify commands, each sized under the 1200 s ceiling).** It holds the work of the old seven
   runtime tasks, because no split of the runtime files avoids a sibling dependency within the wave.
   - Mitigations:
     - the task's first acceptance item fixes an order of work;
     - the journal records each stage as it passes, and a re-attempt resumes from it;
     - the ledger is written per wave.
   - HYPOTHESIS: one coder context holds it. If it does not, the run stops with the task failed and its journal
     current. That is a re-attempt, not a re-plan.
2. **Prompt truncation (FACT, factory).** wan-implement fences a task's acceptance at 3000 characters and its verify
   at 2000 (`wan-implement.js` taskPrompt). Four tasks are longer. Each now opens with a READ FIRST item that points
   the coder to the whole entry in PLAN-M2b.json. This is a factory finding (house scope) for the backlog. It is not
   fixed here.
3. **Same-wave readers of on-tree code (FACT + HYPOTHESIS).**
   - FACT: `flush_iter`, `gather_dtypes`, `SharedBudget`, `LinkSpec.egress` and `capacity_source` exist at 4201c65,
     and their tests pass per their journals.
   - HYPOTHESIS: their owners' remaining work does not change those contracts. A change would be a recorded conflict,
     caught by the reader's tests and the ladder.
4. **The M2 merge (FACT, P-M2).**
   - Both sides change `zero_bubble.py` (`solve`, `schedule_for_spec`, two `_fits`) and four shared test files.
   - r14's lane_order tests are re-run by `scheduler-pieces-plan`, and its acceptance is re-done only if they turned
     red.
5. **SC-4 and SC-5 (FACT).** Both are the researcher's, and both close before wave 1. Neither moves a bit. SC-5 may
   shrink the set of hosts S1 admits, which is the intended effect of counting sealing.
6. **E7 margin (HYPOTHESIS, R16).** The attention-inclusive `layer_costs` may flip the (16, 2) choice; the margin is
   2.8%. The granted PLAN pins the choice, with the 2% tie rule. A mismatch is F9, never an edited expectation.
7. **NIC capacity under load (HYPOTHESIS).** The link probe's bidirectional figure equals the per-peer budget on
   loopback. Whether it does on a real NIC with w − 1 destinations is real-hardware pending.
8. **CPU fleet admitted for pieces (HYPOTHESIS).** The toy model on 4 localhost workers passes S1 and S2 at the SC-5
   k. If it does not, that is a conflict, and the bound is never loosened.
9. **Timing under load (FACT).** E3, E4, the shared-pacer tests and `pieces_rebalance` measure wall time. Their verify
   lines take the whole machine (D-12). A flaky timing pass is not a pass.
10. **Liveness control (HYPOTHESIS).** The `between_ops` seam must deadlock. If it completes, F8's control is reported.
11. **Security.**
    - The frame validator costs one dict lookup and a few integer comparisons per frame. HYPOTHESIS: well under 0.1% of
      the step; E4's telemetry measures it.
    - The HELLO capacity fields are validated at the boundary.
    - Budget files are 0600 under `out_dir`.
    - Pieces traffic rides M2's sealed sockets: no new listener, no new trust boundary.
12. **Process note (FACT).** The planner ran read-only git commands: `rev-parse`, `log` and `status`, plus `git show
    origin/factory/m2:<path>`, which the human's focus asked for. Nothing was staged or committed.

## Rulings at the plan grant (the human, 2026-10-09), carried

- **SC-1, SC-2:** applied through `freeze.mjs`, with history entries.
- **SC-3:** E1's integration arms replace the ADR's fp16 and fp32 identity arms (GATES entry 9).
- **P3:** both experiment PLANs granted.
- **E4 op split:** E4 pins this CPU's measured F:B:W split (about 0.10 : 0.81 : 0.08), not 1:2:1. The order is
  asserted only at RTT 2·c_W.
- **S1 window:** one `c_s`, the planner's `layer_costs`, used in S1, S2 and C.
- **F9 tie rule:** where E7's pinned margin is at most 2%, the runner-up also matches if its simulated step is within
  1% of the best admitted candidate's.
- **`shard` is job-wide (ruling to confirm).** The planner chooses pieces only if every wide stage is admitted.

## Review notes (2-wave replan, round 1)

| # | Finding | Verdict | Where it is fixed |
|---|---|---|---|
| 1 | `stage-pieces` in wave 1 runs code a sibling writes in the same wave (blocker) | Valid | `pieces-lib` is merged into `stage-pieces`. Remaining cross-workstream imports are on the tree (§Interfaces) |
| 2 | `scheduler-pieces-plan` needs the JobSpec fields (blocker) | Valid | Attribute assignment in wave 1; the JSON round trip moves to `arm-pieces` (D-9, I-7) |
| 3 | A re-run of r14 cannot own its tests, and the scheduler consumes r14 with no edge | Valid | r14 merged into `scheduler-pieces-plan` (D-5) |
| 4 | The tripwire test in non-producing legs exits 5, reported as "pending" | Partly valid | The machine check requires the line in every leg (`plan.js` checkTripwires), so it stays. "Pending" is removed: exit 5 is NOT RUN, never cited, and the proof is the arm's verify and the ladder (D-7) |
| 5 | BA-44 ignores M2's rebalance; I16 requires a re-check at every re-plan (blocker) | Valid | BA-44 rewritten; `arm-pieces`: `pieces_rebalance`, `pieces_rebalance_refused`; `stage-pieces`: `e5_rebalance_owner` |
| 6 | k "counted 10" ignores M2's sealing (+~2 passes per side) | Valid | Precondition SC-5; `K_HOST_DRAM_COUNTED`; test `k_counted`; BA-48 |
| 7 | SC-4 leaves two anchor arms red with no task to fix them | Valid | SC-4 is a precondition with a check, closed before wave 1 |
| 8 | `_encode_body` would narrow fp32 pieces under `planes` | Valid | BA-56; `pieces_codec_exact` with a must-fail control |
| 9 | BA-16 written for an unauthenticated M2; line citations are stale | Valid | BA-16 rewritten for the sealed plane; every `where` by symbol (D-8) |
| 10 | No ruling for pieces resuming a zero1 wave | Valid | BA-57; BA-53 rewritten; `e5_zero1_to_pieces`, `e5_refuse_mixed_kind` |
| 11, 12 | `stage-pieces` deps [] while it imports `exchange` / `logical_ckpt` | Valid (same as 1) | Same fix as 1 |

## Review notes (2-wave replan, round 2)

Every finding was checked against the code (`fleetplan.py:197` and `planner.py:471` call `finalize()`;
`stage_main.py:95-101`; `local.py:106` and `agent.py` launch through `stage_main`; M2's `_resume` raises on an empty
`slot_waves`; `machine-lock.mjs` `SLOT_TIMEOUT_SEC = 1200`). All 14 are valid. None changes the training math.

| # | Finding | Verdict | Where it is fixed |
|---|---|---|---|
| 1 | The scheduler's tests run through a `finalize` that `stage-pieces` rewrites in the same wave | Valid | A frozen finalize contract (I-7), tested on both sides (`spec_m2b_planner_contract`, `plan_wide_finalize_contract`). M2b assertions read the plan dict. The real `finalize` path runs in `arm-pieces` |
| 2 | The `piece_bounds` key format and alignment are unfixed | Valid | I-7 fixes the key (the `named_parameters()` name, as `exchange_units` uses it) and the quantum. The planner emits None in M2b |
| 3 | `zero_bubble.py` changes under `stage-pieces`; the scheduler's RUNTIME-I2 is evidenced by a sibling | Valid | I-10 freezes the entry points. RUNTIME-I2 runs in `arm-pieces`' verify (D-10). `lane_order_interface` is the scheduler's own test |
| 4 | Timing lines run in a slot beside heavy lines | Valid | Whole-machine lines with `--timeout 1200` for E3, E4, the shared pacer, F1, `holds_seams` and `pieces_rebalance` (D-12) |
| 5 | `_check_resume_pins` item contradicts BA-53 | Valid | Item rewritten with both cases, plus the control `e5_refuse_zero1_width` |
| 6 | Only GRADS frames are required to pass `check_frame` | Valid | GRADS and PARAMS. `pieces_frame` covers both kinds, with a GRADS-only control |
| 7 | The GATES grep `not armed` fails on `plan_wide_unarmed` in `test_coordinator.py` | Valid | Both `plan_wide_unarmed` and `plan_wide_armed` move to `test_control_admission.py`. The grep is also in `arm-pieces`' verify (D-6) |
| 8 | No discriminating test for I16's device memory; the pinned-chunk term is missing | Valid | `i16_device_bound` (DESIGN §9: 6.17 and 4.17 B/param; holds of 4 and 32 B/param), `i16_device_refuse` with a control that drops the own-piece term, and `i16_mem_host_pinned` |
| 9 | `heal-class-tripwire`'s verify launches a stage process from a half-edited `stage.py` | Valid | Its verify is unit-only. The selections move to `arm-pieces` (D-10). `model-flush-iter`'s `test_zero_bubble.py` line moves with them (same hazard) |
| 10 | `python -m wantrain.worker.stage_main` bypasses the pieces refusal (blocker) | Valid | BA-58: `--launcher` on `stage_main.main`; `run_local` passes `local`, the agent passes `coordinator` (`arm-pieces` owns `agent.py`); test `pieces_stage_main_refused` |
| 11 | Wire `native` reaches production with only some legs in | Valid | The tripwire refuses native too, and `model-flush-iter` is a leg. `arm-pieces` reruns `e1_native_` and `native_width1` before arming (D-11) |
| 12 | No ruling for the resume step when the width is raised | Valid | BA-51 and BA-54: `resume_step`, else p0's newest complete wave. `e5_widen`, zero1 w2 → pieces w3, and a 2 → 4 run in the F1 rotation |
| 13 | The whole-file verify of `test_engine_bitwise.py` exceeds 1200 s | Valid | Whole-file line dropped. The M2 tests run as one exclusion selection; `e1_native_` and the E1 matrix run per recipe (pairwise covering set, at most 30 arms each); there is a per-command time budget |
| 14 | `arm-pieces` acceptance items have no named test and no selecting verify line | Valid | Every item names its test (`plan_wide_lane_budgets`, `plan_wide_recheck`, `pieces_spec_mode_checks`, `plan_wide_zero1_oracle`, `plan_wide_missing_dram`, `pieces_plan_oracle`, `pieces_heal_run`). Each has its own verify line, never in an OR. `stage-pieces`' OR line (`pieces_local_class`, a round-1 minor) is split the same way |

Round-1 minors that this round also closes:
- the tripwire selector collecting nothing after the arm: `pieces_unarmed` is kept, patched off;
- MEM-HOST pinned chunks: finding 8;
- `bench` dropped silently: §Bar records it N/A with the reason;
- E4 importing an unfinished module: `e4-exposure` imports `exposure` inside its test functions.

The review notes of the 13-wave plan (rounds 1 and 2) are in the history of this file (commit 45bf898). Every finding
they validated is still delivered, by the task the §Old to new table names.

## Final check (after review round 2)

Every finding of both rounds was re-checked against these two files as they now stand. Four were still open and are
fixed here, in both files:
- `stage-pieces`' verify still selected required tests through OR'd `-k` expressions (`need_first or chunk`, four
  `e5_` names in one line), and no line selected `copy_engine_stage`. Each now has its own line, so a missing test
  exits 5.
- Two `arm-pieces` items had no named test: the plan fleet inputs (BA-48) and the rationale. They are now
  `plan_wide_fleet_inputs` and `plan_wide_rationale`, each with its own verify line.
- RUNTIME-I12 refuses an extra index; BA-51 ignored it. It is now `e5_refuse_extra_index`, through
  `select_wave(read_manifest, peers_present)`.
- Tied weights (BA-42) had no ruling: the Embed's unit, a cross-stage refusal in `finalize`, and `e1_core_tied`.

None of these changes the training math: a unit assignment and three refusals.
