export const meta = {
  name: 'wan-x-m2-p1-liveness-telemetry',
  description: 'M2 phase 1 of 3: a sound liveness judge across restart in place and shutdown (C1, CONTROL-I17) and one validated per-step F/B/W/OPT and per-hop telemetry summary per slot delivered to the coordinator (R1 RUNTIME-I9, C2 CONTROL-I18), with the training state equal to the reference by sha256 and nothing consuming the telemetry yet.',
  whenToUse: 'After a human grants docs/design/NEXT-m2-p1-liveness-telemetry.md, docs/design/GATES-m2-p1-liveness-telemetry.json and this built workflow, and after the spec amendments of the NEXT document are applied through /wan-adr. Rerun wan-next for phase 2 (re-plan behind replan = False).',
  phases: [
    { title: 'Gate', detail: 'Refuse without the human grant of the NEXT document, its pinned gates and this built workflow, then of DESIGN.md and the m2-p1-spec ADR; require the spec amendments; save the pre-phase C1 files, test files and test list, checked against the digests E32/E33/E34, inside a scope audit' },
    { title: 'Implement', detail: 'Wave 1: R1 (runtime) and C1 (control) in parallel on disjoint paths; wave 2: C2 (control) on top of both; nobody runs a command; every wave scope-audited' },
    { title: 'Review', detail: 'Six lenses written for this objective (objective, liveness, byte-exact, tests, boundary by the security architect, fidelity) over every owned path, so a rerun reviews what an earlier run changed, until zero blocker/major; every fix round scope-audited; a review that does not converge stops the phase for a human, with its open findings' },
    { title: 'Verify', detail: 'Solo, inside a scope audit that allows no tracked change: pinned.mjs runs GATES-m2-p1-liveness-telemetry.json and decides; the script accepts the relayed JSON only for this gate id, every entry and the audited tree' },
    { title: 'Stabilize', detail: 'One item at a time, every proposal, vote and apply carrying the phase facts: proposal -> 3 votes (3/3 for anything under tests/, else a majority with the spec veto) -> scoped apply, each scope-audited; then Verify again; <= 4 rounds; a round that changed no file ends the loop and a VERIFIED on a tree that already failed is intermittent (NOT VERIFIED); code Stabilize changed is reviewed again whether or not the last verify passed' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project
// @include gates
// @include triage
// @include fix
// @include scope

const SLUG = 'm2-p1-liveness-telemetry'
const NEXT = 'docs/design/NEXT-m2-p1-liveness-telemetry.md'
const GATES = 'docs/design/GATES-m2-p1-liveness-telemetry.json'
const PINNED = 'node agents-architetura/tools/gates/pinned.mjs m2-p1-liveness-telemetry --json'
const WORKFLOW = '.claude/workflows/wan-x-m2-p1-liveness-telemetry.js'
// Verbatim from the NEXT document (sections Objective .. Exit gate and Risks). Two edits, both forced by the house lint
// (no agent prompt may name the approvals or frozen records): row E31 is replaced by a pointer and "frozen (... absent)"
// names no file. The Phases section and the "Alternatives not picked" list are not copied: the script implements the
// first, and the second is a decision record, not a fact a writer acts on.
const FACTS = `
FACTS OF THIS PHASE (verbatim from ${NEXT}; re-check each against its source before relying on it):
# NEXT: m2-p1-liveness-telemetry

Phase workflow: \`wan-x-m2-p1-liveness-telemetry\`. Pinned verify: \`docs/design/GATES-m2-p1-liveness-telemetry.json\`.
Milestone M2 (adaptivity), phase 1 of 3. Hardware: \`{"gpu": false}\`. Everything below runs on this machine's CPU;
every performance number is SIMULATED and real-hardware validation stays pending.

Split (LESSONS.md:10-13, dependency order): **phase 1 (this)** sound liveness accounting for restart in place (C1) and
per-step F/B/W plus per-hop telemetry delivered to the coordinator (R1, C2), with no consumer; **phase 2** re-solve from
telemetry, resume under a new ownership, the re-plan loop behind \`replan = False\` (SCHEDULER-I15); **phase 3** plan mode
for wide stages (M2-R5-FIT-2, FIT-5) and SCHEDULER-I13. This phase has 3 work packages and 14 pinned entries (one above
the lesson's ~12: \`would-fail-mutants\`, see "Exit gate", is the mechanical E13 check and is not traded away).

# Objective

Make the coordinator's liveness judge sound across restarts and shutdown (replayed loss lines never feed the stall
clock; quiet after every slot finished its last optimizer step is a bounded shutdown, not a stall), and make every slot
stream one validated per-step summary of measured F/B/W/optimizer times and per-hop wire throughput to the
coordinator, with the training state still equal to the reference by sha256 and nothing consuming the telemetry yet.

# Evidence

Facts only, each with its source. "Read" means read for this document on the current tree.

| # | Fact | Source |
|---|---|---|
| E1 | The ladder is all green; \`missingRequired\` is empty. \`full\` and \`mutation\` are on demand; \`frozen\`, \`component\`, \`multinode\`, \`matrix\`, \`rest\`, \`bench\`, \`redteam\` are N/A. | state snapshot, \`ladder\` |
| E2 | \`invariants: 43/52 covered\`; SCHEDULER-I13 and SCHEDULER-I15 are DEFERRED to M2. | state snapshot, \`invariants\` |
| E3 | The hardware switch is \`"gpu": false\`. | agents-architetura/tools/factory/hardware.json:2 (read) |
| E4 | \`gpu_allowed()\` is True only for a literal \`"gpu": true\`; \`mps_available()\`/\`cuda_available()\` return False otherwise. The MPS test is \`@pytest.mark.mps\` + \`skipif(not mps_available(), reason=... GPU_OFF)\`. | tests/conftest.py:59-86; tests/integration/test_coordinator.py:181-185 (read) |
| E5 | findings.json holds 15 findings, all open, all from "wan-next m2-adaptive-ownership review round 5 (2026-10-08)". | agents-architetura/docs/findings.json:1-139 |
| E6 | M2-MPS-INTERMITTENT (major): the CPU+MPS fleet test stalls intermittently ("no optimizer step fleet-wide for 8s (limit 8s) while every worker is alive"), restart budget spent; N/A under gpu false; "may share a cause with the restart-in-place liveness accounting (M2 C1)". | findings.json:3-11 |
| E7 | The prior snapshot's failing log: "restarting in place from step 7" twice, then "the restart budget (2) is spent", recipe \`steps=8\`. | .factory/prior/NEXT-m2-adaptive-ownership.md:21-22 |
| E8 | M2-R5-FIT-1 (major): the MPS test failed in \`integration\` and passed in \`full\` in one ladder run; recorded fix: resolved for gates by \`gpu_allowed\`. | findings.json:13-20 |
| E9 | M2-R5-FIT-3 (minor): "Send events do not record the sending host (stage.py:369-371), and there is no step_opt event." | findings.json:31-38 |
| E10 | M2-R5-SAFE-1/2/3/4 (stabilize writes skip review; stabilize review lacks the tests lens and passes 2/3; specConflict ignored outside Implement; marker scan misses spellings and \`mark.mps\`). | findings.json:58-92 |
| E11 | M2-R5-VER-1 (major): suite gates check only pytest's last line; no frozen node list. VER-3: grep exit 2 passes. VER-4: a causal timeout below its inner worst case. VER-5: telemetry cost measured on the summary only. | findings.json:94-137 |
| E12 | M2-R5-FIT-2 (plan-mode grouping label, coordinator.py:723) and FIT-5 (plan-mode fixture) concern plan mode. | findings.json:22-29, 49-56 |
| E13 | LESSONS: keep a phase to ~3 work packages and ~12 gate entries, split along the dependency order; use distinct expected values and check against a deliberately wrong implementation. | agents-architetura/docs/LESSONS.md:7-13 |
| E14 | S4: "no optimizer step fleet-wide in max(floor, 6 x median step)"; the limit is \`max(stall_floor_s, stall_steps * step_s)\`, \`first_step_timeout_s\` while \`median_step_s\` is None. Nothing in the judge knows the final step. | src/wantrain/fault_tolerance/liveness.py:14-15, 217-228, 259-265 (read) |
| E15 | \`median_step_s\` takes gaps between all slots' \`(step, t)\` points with \`s1 > s0 and t1 > t0\`, needs ≥ 3 gaps, median of the last 12. | liveness.py:73-82 (read) |
| E16 | Progress is stamped with the coordinator's arrival time, deduplicated only against the previous step. | src/wantrain/coordinator/coordinator.py:385-388 (read) |
| E17 | The agent turns both \`WT_STEP\` and \`WT_METRIC\` lines into the same \`PROGRESS\` message (no source field; \`WT_STEP\` gives \`"loss": None\`). | src/wantrain/worker/agent.py:463-491 (read) |
| E18 | On resume, \`run_stage\` keeps the loss stream and \`stream_losses\` tails it from offset 0, re-printing one \`WT_METRIC\` per past step. | src/wantrain/worker/stage_main.py:29-55, 63-65 (read) |
| E19 | Every slot prints \`WT_STEP step\` after its optimizer step and checkpoint; after the loop the final checkpoint is unconditional, then \`_finish\` (loss gather at final peer 0, sender drain and reader joins bounded by \`SHUTDOWN_GRACE_S = 120\`). | src/wantrain/runtime/stage.py:66, 959-990, 1095-1160 (read) |
| E20 | After the stage process exits, the agent ships the final wave over the control channel, then sends \`EXIT\`; heartbeats report \`stage_alive\` until the exit is reported. | agent.py:280-296, 493-518, 551-592 (read) |
| E21 | \`default_ckpt_every = 1\`, \`stall_floor_s = 90.0\`, \`max_restarts = 2\`; the test helper sets \`stall_floor_s = 8.0\`. | coordinator.py:72, 77, 90; tests/helpers/cluster.py:24-33 (read) |
| E22 | \`_launch\` creates fresh \`SlotState\`s per attempt; \`_slot_for\` drops messages of another attempt. \`_restart_in_place\` relaunches from \`common_step\` whatever it is. | coordinator.py:363-372, 914-931, 1195-1205 (read) |
| E23 | \`_conn_loop\` catches only \`(ChannelClosed, OSError, ValueError)\`; the \`TELEMETRY\` branch indexes \`msg['rank']\`, \`msg['peer']\`, \`msg['attempt']\` and calls \`os.replace(body, dst)\` with \`dst\` built from those message fields. | coordinator.py:297-310, 352-356 (read) |
| E24 | \`step_opt\` events exist (local and wide). \`send\` events carry no hop key. In the multi-stream path the \`send\` event is emitted by the encoder after enqueueing fragments, so it times encode + enqueue, not the wire; fragment senders emit no event. | stage.py:358-371, 373-396, 548-578, 894, 955 (read) |
| E25 | \`self.telemetry\` is appended by sender threads and the main thread through \`_ev\`; \`_flush_telemetry\` iterates it, then \`clear()\`s it, without a lock. | stage.py:133, 595-607, 681-683 (read) |
| E26 | Senders are created per connection without the connection key: \`_wire_shared\`, \`_wire\`, \`_wire_multi\`. Paced hops use \`LinkSpec(**links[r])\`; intra hops use \`LinkSpec()\` (unpaced). \`SenderPacer.pace\` sleeps \`nbytes*8/(bw_mbps*1e6)\`. | stage.py:175-187, 271-291, 445-455, 472-516; src/wantrain/transport/shim.py:27-33 (read) |
| E27 | No code reads telemetry event names except the agent, which ships the per-slot telemetry file at exit. | grep over src/ tests/ (read) |
| E28 | \`patch_spec_for_resume\` lives at fault_tolerance/checkpoint.py:143-153 (not spec.py:52-56, M2-R5-FIT-4). | src/wantrain/fault_tolerance/checkpoint.py:143-153 (read) |
| E29 | \`invariants.mjs\` fails on a test claiming an id no spec defines. | agents-architetura/tools/gates/invariants.mjs:3, 47-48 (read) |
| E30 | SPEC-control §2.2 step 5 states S4 without any final-step exception; §2.1 monitor inputs and §2.4 agent name \`PROGRESS (from WT_STEP)\` and \`TELEMETRY\`. SPEC-runtime task 1 (M2) asks for per-op F/B/W and per-hop transfer times "in TELEMETRY (median per step)". | docs/design/SPEC-control.md:41-44, 62-68, 91; docs/design/SPEC-runtime.md:113-115 (read) |
| E31 | [row omitted from agent prompts: it names the human approvals record, which no agent prompt may carry; the Gate step checked the grants] | NEXT document, E31 |
| E32 | Pre-phase sha256: liveness.py \`4132edab…467a6\`, coordinator.py \`a02f7687…ffe226e\` (equal to the prior draft's pins); the \`causal.mjs save\` manifest for exactly these two files in this order: \`ac243d33…952f07\`. | \`shasum -a 256\` (run); node JSON.stringify digest (run) |
| E33 | Pre-phase test functions: 153 (no test classes). Canonical list sha256 \`21e0c82f276af6b090806045212de1087e2b18def459eedf4eb84aa566a20f38\` (command in "Exit gate"). Pre-phase skip/marker scan (widened, "Exit gate"): 3 lines, normalized sha256 \`9841c3222d276dc6eb87ad45fe0b418fd6d7d580a6f3299e119ca8b2117dbb6d\`. No test says "awaiting implementation". | grep runs (run) |
| E34 | Pre-phase sha256 of the four owned test files: test_fault_tolerance.py \`70dab584…685030\`, test_coordinator.py \`2d42e69c…1382b1\`, test_runtime_spec.py \`af63f4c7…d7a8a3\`, test_engine_bitwise.py \`0b1a8873…3da99a\` (full digests in the Gate step and \`pre-tests-unchanged\`). Existing tests build \`SlotState\`/\`LivenessJudge\` directly and assign \`progress\` lists; none sends a \`PROGRESS\` message, so C1 needs no edit of an existing test. | \`shasum -a 256\` (run); tests/unit/test_fault_tolerance.py:21-52, 122 (read) |
| E35 | With \`steps = 4\` the pre-phase judge can reach the end of a run with \`median_step_s\` None: 3 step transitions give at most 3 gap candidates after sorting by \`(step, t)\`, ≥ 3 are needed, and a \`WT_METRIC s\` that arrives after \`WT_STEP s+1\` (the loss tail polls every 0.5 s; dedup is only against the previous step) is appended as \`(s, t_late)\` and voids one, so the S4 limit is \`first_step_timeout_s\` (90 s in \`FAST_FT\`), not the 8 s floor; a 12 s delay would then never trip S4. Arithmetic for the slow-shutdown recipe (\`first_step_timeout_s = 30\`, exit delay 45 s): the pre-phase limit is 30 s when the median is None and max(8, 6 × median) otherwise, so ≤ 30 s while the median step is ≤ 5 s (HYPOTHESIS for this CPU toy step, 0.1–1 s per the cost model); the quiet after the last \`WT_STEP\` is ≥ 45 s (\`EXIT\` follows the delay), so S4 trips either way, and a relaunched pre-phase attempt has the same ≤ 30 s limit and the same 45 s quiet until the budget is spent. No pre-phase run of this recipe is recorded: arm A of \`causal-slow-shutdown\` is that run. | liveness.py:73-82, 222-228; tests/helpers/cluster.py:24-33; coordinator.py:385-388; stage_main.py:29 (read) |

# Why now

Precedence: no red rung (E1). Open majors next (E5): the ones this phase can act on are M2-MPS-INTERMITTENT (its
CPU-reproducible mechanism, below), VER-1 (suite gates), SAFE-1/2 (the phase workflow's own stabilize loop). C1 is also
the mechanism every later M2 phase relies on: a re-plan is applied through restart in place (SPEC-control task 2), and
the re-solve needs measured costs (SPEC-runtime task 1). Lesson 2 says split first along that dependency.

**The mechanism under M2-MPS-INTERMITTENT, derived from the code (facts E14-E22, conclusion HYPOTHESIS):**
1. After \`WT_STEP steps−1\`, every slot still owes: the final \`torch.save\` (E19), \`_finish\` (loss gather, drain ≤ 120 s,
   E19), and the agent's final-wave ship before \`EXIT\` (E20). S4 sees no progress during all of it (E14).
2. When that quiet exceeds \`max(stall_floor_s, 6 × median)\`, the verdict is \`stalled\` and \`_restart_in_place\` relaunches
   from wave \`steps−1\` (E22, E7 "from step 7" with \`steps=8\`).
3. The relaunched attempt runs no step, but \`stream_losses\` replays every past \`WT_METRIC\` in one burst (E18), each
   stamped at arrival (E16, E17): \`median_step_s\` becomes ~0 and the limit collapses to the floor (E15, E14). The same
   shutdown quiet repeats, the budget is spent (E7).
4. Mid-run the same replay defeats \`first_step_timeout_s\` after any resume: progress is non-empty, so a resumed attempt
   whose connect + load + first step exceeds the floor is restarted again.

Cost model of the quiet (the term C1 changes), \`T_quiet = T_save + T_finish + S_shard / B_ctrl\`. Arithmetic, not a
measurement: a 7B model in fp32 with Adam is ~16 B/param; over 4 stages a shard is ~28 GB; at B_ctrl = 1 Gb/s the ship
alone is ~224 s, already above the default 90 s floor. So S4 misfires at the end of a real-size run, on any hardware.
On this CPU the toy model's quiet is ~seconds, which is why only the slower MPS pair showed it. HYPOTHESIS, falsified by
\`causal-slow-shutdown\`: an injected post-exit delay of 45 s, above every limit the pre-phase S4 can use in that recipe
(E35: floor 8 s, \`first_step_timeout_s\` 30 s, 6 × median while the median step is ≤ 5 s), makes the pre-phase
coordinator fail for the stall and the phase's coordinator pass.

**Findings, ruled for this phase:**
| Finding | Ruling |
|---|---|
| M2-MPS-INTERMITTENT | C1 fixes the CPU-reproducible mechanism; the finding stays open (needs gpu true + \`/wan-bughunt\`). Closeout records the causal result next to it. |
| M2-R5-FIT-1 | Confirmed resolved by E4: the \`suite-nodes\` gate allows that skip only with the GPU_OFF reason. Closeout may close it, citing that gate's run. |
| M2-R5-FIT-3 | Half refuted: \`step_opt\` exists (E24). Half confirmed: no hop key; the multi-stream event times encode, not the wire. Fixed by R1. |
| M2-R5-FIT-4 | Cited correctly here (E28); this phase does not touch \`patch_spec_for_resume\`. |
| M2-R5-FIT-2, FIT-5 | Out of scope (plan mode, phase 3). |
| M2-R5-SAFE-1/2/3 | Requirements on this phase's workflow (see Phases, Verify). |
| M2-R5-SAFE-4 | \`no-new-skip-markers\` widened (spacing, \`_pytest\`, \`getattr(pytest\`, \`mark.mps\`, collection hooks) and pinned by digest. |
| M2-R5-VER-1 | \`suite-nodes\`: every pre-phase test function and every new node must appear as \`PASSED\`; exactly the 2 GPU_OFF skips. |
| M2-R5-VER-2 | Out of scope (re-plan arm, phase 2). |
| M2-R5-VER-3 | \`no-pending-markers\` passes only on grep exit 1. |
| M2-R5-VER-4 | \`causal-slow-shutdown\`: outer \`timeoutSec\` 4200 > 2 arms × 2 runs × 900 s. |
| M2-R5-VER-5 | The stage measures its whole per-step telemetry block (\`emit_us\`); the coordinator measures its intake (\`handle_us_max\`). |

Why no ADR: no training, synchronization, protocol-layer or recovery-semantics change. C1 restores the judge's stated
intent (S4 judges a fleet that still owes optimizer steps); recovery paths are unchanged. Telemetry is timing-only and
has no consumer. Reversal: if any lens finds a change to a byte of state or to when a recovery fires other than the
shutdown/replay rules below, STOP and \`/wan-adr\`.

# Scope

Two workstreams, because the mechanism crosses the stage process (runtime: measures) and the coordinator/agent
(control: judges and receives); either half alone has no observable effect.

| Writer | Workstream | Owned paths |
|---|---|---|
| R | runtime | \`src/wantrain/runtime/stage.py\`, \`src/wantrain/runtime/telemetry.py\` (new), \`tests/unit/test_runtime_spec.py\`, \`tests/integration/test_engine_bitwise.py\` |
| C | control | \`src/wantrain/fault_tolerance/liveness.py\`, \`src/wantrain/coordinator/coordinator.py\`, \`src/wantrain/worker/agent.py\`, \`tests/unit/test_fault_tolerance.py\`, \`tests/integration/test_coordinator.py\` |

Not touched: \`stage_main.py\` (the replay is neutralized at the coordinator boundary), \`tests/helpers/\` (\`LocalCluster\`
already inherits \`os.environ\`, cluster.py:92), the scheduler, the transport, \`tests/oracle/\`, \`tests/chaos/\`.

## Spec amendments (precondition; applied by the sanctioned writer before writers start)

No phase agent writes \`docs/design/\` and E29 forbids covering undefined ids. Route: \`/wan-adr {"problem": "apply
docs/design/NEXT-m2-p1-liveness-telemetry.md §Spec amendments verbatim to SPEC-control.md and SPEC-runtime.md, with
history entries", "id": "m2-p1-spec"}\`, then the human grants it. wan-adr writes these texts in its draft, before any grant, so the greps alone prove
nothing: the Gate refuses to start until exactly one \`docs/design/adr/ADR-NNNN-m2-p1-spec.md\` exists and a human grant
covers it and \`docs/design/DESIGN.md\`; \`spec-amended\` checks the same grant, then greps for the result.
- SPEC-control §2.2 step 5, appended: "S4 judges only while some launched slot still owes an optimizer step: it reads
  only progress parsed from \`WT_STEP\` whose step lies in [the attempt's start step, \`steps − 1\`]; replayed \`WT_METRIC\`
  lines never feed it. Once every launched slot exited 0 or reported step \`steps − 1\` (or the attempt started at
  \`steps\`), the attempt is in shutdown: quiet is bounded by \`shutdown_limit_s\` (default 600), past which the verdict is
  failed with 'shutdown exceeded … after the final step'. S1–S3 and strikes still apply in shutdown."
- SPEC-control §2.1 monitor inputs and §2.4 agent: "\`PROGRESS\` carries \`src\` (\`step\` from \`WT_STEP\`, \`metric\` from
  \`WT_METRIC\`). \`TELEM_STEP\`: the live per-step summary parsed from \`WT_TELEM\`, validated at the coordinator
  (CONTROL-I18). \`TELEMETRY\` stays the telemetry file shipped at exit."
- SPEC-runtime task 1: "in TELEMETRY" becomes "in one \`WT_TELEM\` line per step (RUNTIME-I9)".
- New ids, texts in Invariants: CONTROL-I17, CONTROL-I18, RUNTIME-I9.

## Mechanism (the contract writers implement)

**C1 — liveness accounting (control, CONTROL-I17).**
- \`liveness.py\`: \`SHUTDOWN_LIMIT_S = 600.0\`. \`SlotState.start_step: int = 0\`. \`SlotState.add_progress(step: int, t: float,
  final_step: int | None) -> bool\` appends \`(step, t)\` and returns True iff \`start_step <= step\`, \`final_step is None or
  step <= final_step\`, and \`step\` differs from the last recorded step.
  \`LivenessJudge(..., final_step: int | None = None, shutdown_limit_s: float = SHUTDOWN_LIMIT_S)\`. Shutdown mode holds
  when \`final_step is not None\` and every launched slot has \`exit_code == 0\`, or \`last_step() == final_step\`, or
  \`start_step > final_step\`. In shutdown mode, after the done / S1 / S2 / grace / crashed checks and the S3 peer-lost
  check, the S4 block is replaced by: quiet = now − max(last progress, started_at); \`quiet > shutdown_limit_s\` →
  \`Verdict("failed", None, "shutdown exceeded {limit:.0f}s after the final step …")\`; otherwise heartbeat staleness
  still earns strikes, else \`ok\`. \`final_step=None\` keeps today's behavior exactly (existing tests unchanged).
- \`coordinator.py\`: \`CoordinatorConfig.shutdown_limit_s: float = 600.0\`. \`_judge\` passes \`final_step = spec.steps − 1\`
  of the run and \`shutdown_limit_s\`. \`_launch\` sets every new \`SlotState.start_step\` to \`wave + 1\` when relaunched from
  a wave (\`_restore_and_relaunch\`'s \`step\`), else 0. \`_slot_update(progress=True)\` records progress only when
  \`msg.get("src") == "step"\`, through \`add_progress\` with an integer step (a non-integer step is dropped and logged).
- \`agent.py\`: \`PROGRESS\` gains \`"src": "step"\` (from \`WT_STEP\`) or \`"metric"\` (from \`WT_METRIC\`).
- Restart from the final wave stays reachable through S2/S3/convict (a shutdown-time peer loss): it reruns \`_finish\`
  from wave \`steps − 1\` with zero optimizer steps, which RUNTIME-I3 makes exact. It is no longer reachable from S4
  quiet. Default 600 s covers \`T_quiet\` up to ~(600 − 120 − T_save) s × B_ctrl, e.g. ~55 GB per shard at 1 Gb/s;
  larger shards raise it (Risk R1).
- Test seam (control, timing only): \`WANTRAIN_SIM_EXIT_DELAY_S\` read once at agent start (float in [0, 3600], malformed
  raises \`ValueError\`); in \`_supervise\`, after \`stage.wait()\` returns 0 and the post-exit \`_ship_checkpoint\`, before \`EXIT\`, sleep
  that long (the coordinator holds the final wave during the quiet, as after a slow final ship). Absent means no effect.

**R1 — per-step telemetry (runtime, RUNTIME-I9).**
- \`_ev\` appends under \`self._tel_lock\`; \`_take_events()\` swaps the list under the lock (no lost or duplicated event).
- Every sender knows its hop: \`_sender(sock, q, pacer, hop)\` and \`_frag_sender(sock, frag_q, pacer, hop)\` with hop
  labels \`next<q>\`, \`prev<q>\`, \`intra<k>\` from the connection key. Chain path: the \`send\` event gains \`hop\`. Multi-stream
  path: the encoder emits \`encode\`; each fragment sender emits \`send\` with \`hop\`, \`nbytes\`, spanning \`pace\` +
  \`send_frame\`.
- \`runtime/telemetry.py\`, pure (no torch import, no clock, no RNG): \`summarize_step(events, step, rank, peer, step_ms,
  emit_us) -> dict\` returning \`{"v": 1, "step", "rank", "peer", "step_ms", "emit_us", "ops": {"F"|"B"|"W"|"OPT":
  {"n": int, "med_ms": float}}, "stall_ms": float, "hops": {<hop>: {"frames": int, "bytes": int, "busy_ms": float}}}\`.
  \`med_ms\` is the median of the op's event durations; \`busy_ms\` is the length of the union of the hop's send intervals
  (parallel streams overlap). F/B/W/OPT and stall events are those of this step; send events are those completed since
  the previous summary (senders run asynchronously). \`parse_sim_slow(value) -> (rank, peer, factor, from_step) | None\`
  raises \`ValueError\` on a malformed value.
- In \`run()\`, after \`_optimizer_step(step)\`: one block, timed with \`time.perf_counter\`, takes the events, appends them
  to the existing telemetry file, summarizes, prints \`WT_TELEM <json>\` (≤ 4 KiB), before \`_checkpoint\` and \`WT_STEP\`.
  Its duration is the next summary's \`emit_us\` (the first summary's is 0). \`step_ms\` spans \`_run_step\` start to the end of
  \`_optimizer_step\`.
- Test seam (timing only): \`WANTRAIN_SIM_SLOW="rank:peer:factor[:from_step]"\`, read once in \`StageRuntime.__init__\`; for
  the matching slot at global step ≥ \`from_step\`, each F/B/W sleeps \`(factor − 1) × elapsed\` inside its timed span,
  before its \`_ev\`.

**C2 — telemetry intake (control, CONTROL-I18).** Depends on R1's line format and C1's files.
- Agent: a \`WT_TELEM\` line of ≤ 4 KiB that parses as a JSON object becomes \`{"kind": "TELEM_STEP", "run_id", "attempt",
  "summary"}\`; anything else is dropped. It never reuses \`TELEMETRY\`.
- Coordinator: \`_conn_loop\` wraps each \`_dispatch\` so that any exception from one message is logged as \`warn malformed
  <kind> from <worker>: <error>\`, the message's file body (if any) unlinked, and reading continues; only channel
  errors end the loop. (Today a \`KeyError\` ends the reader silently, E23, and the unread heartbeats would later convict
  a live worker.)
- \`TELEMETRY\` branch: rank, peer, attempt come from \`_slot_for\` (the connection's slot and the current attempt), never
  from the message; no slot or a stale attempt unlinks the body and drops it; a missing or non-path body is dropped.
- \`TELEM_STEP\` branch: slot from \`_slot_for\`; accepted only if the summary is a dict, ≤ 4096 bytes re-serialized, has
  exactly the known keys, \`v == 1\`, its \`rank\`/\`peer\` equal the slot, its \`step\` is an int in [start step, steps − 1],
  every number finite with times in [0, 1e6] ms and counts/bytes in [0, 1e12]. Accepted summaries go to a per-slot
  in-memory window (\`deque(maxlen=64)\`) and are appended to \`<run_dir>/telem_steps.jsonl\` with \`attempt\`, until that
  file reaches \`telem_log_max_bytes\` (default 64 MiB, then one \`warn\` and no more writes). \`run.rationale["telemetry"] =
  {"accepted": n, "dropped": d, "handle_us_max": x}\` (perf_counter around the branch). Nothing reads the window yet.

**Cost model of telemetry (HYPOTHESIS, measured by the gates).** Per slot per step, E ≈ 3·M/w op events + sends + stalls
(M = 8 → ~30-40; M = 64, w = 1 → ~250). Summarize O(E log E) in Python ≈ 1 µs/event, JSON + print ≈ 50 µs: predicted
emit_us ≤ 300 µs vs a CPU step of 0.1–1 s (≤ 0.3%) and seconds on real models (< 0.03%). Coordinator intake ≈ 50 µs per
message; control bytes ≈ 1 KB/slot/step (64 slots at 1 step/s = 64 KB/s), negligible next to per-step wave shipping. The
gain vanishes (cost > 1%) only for steps < ~30 ms with M ≥ 64, which \`TELEM_FRAC\` would expose.

# Invariants

Delivered (spec text for the amendments):
- **CONTROL-I17.** The stall clock reads only \`WT_STEP\` progress inside [the attempt's start step, \`steps − 1\`], so a
  replayed loss stream never moves it and a resumed attempt is judged by \`first_step_timeout_s\` until its first real
  step. Once every launched slot finished the final step, quiet is shutdown, bounded by \`shutdown_limit_s\`, never a
  stall.
- **CONTROL-I18.** A worker's message never stops the coordinator from reading that worker: a malformed or stale
  message is logged and dropped and its file body unlinked. \`TELEMETRY\` and \`TELEM_STEP\` take rank, peer and attempt from
  the connection's slot, never from the message; a \`TELEM_STEP\` summary is accepted only within the size, key, range and
  identity rules of C2.
- **RUNTIME-I9.** Each slot prints one \`WT_TELEM\` per optimizer step, before \`WT_STEP\`: per-op F/B/W/OPT count and median
  ms, stall ms, per-hop frames, bytes and wire-busy ms, and its own emit cost, computed from monotonic timestamps only
  (no tensor op, no device sync, no RNG draw). The training state stays byte-identical to the reference.

Kept: RUNTIME-I1..I7 (I3 resume bit-for-bit, I4 fold by index, I7 normal shutdown), CONTROL-I6, I7, I8, I10.
CONTROL-I5 stays N/A under gpu false. SCHEDULER-I13/I15 stay DEFERRED (phases 2-3).

# Binding audit

No reduction, fold order or RNG draw is changed by this phase; every row below is state the change reads or rewrites.

| BA | What | Where (file:line) | Ruling |
|---|---|---|---|
| BA-1 | Index-order folds \`OrderedAccumulator(range(self.M))\` (chain, shard, batch) | stage.py:864, 872, 926 | Untouched. The SIM_SLOW sleep is inside op spans, after compute, before \`_ev\`: no data path. Proven by manifests equal to the reference in both integration tests. |
| BA-2 | Global RNG saved/restored in waves | stage.py:639, 669-670 | Telemetry and seams draw nothing (\`telemetry.py\` imports no torch, no random). Unit test asserts \`torch.get_rng_state()\` unchanged around \`summarize_step\`; manifests compare the RNG state. |
| BA-3 | Batch generation \`make_batch(data_seed, step, mb)\` | stage.py:720, 772 | Untouched. |
| BA-4 | Telemetry list shared by sender threads and the main loop | stage.py:133, 595-607, 681-683 | Lock in \`_ev\`, swap in \`_take_events\`; unit test: both block while the lock is held (deterministic), and concurrent appenders under a 1 µs switch interval lose and duplicate nothing. |
| BA-5 | Multi-stream \`send\` event times encode + enqueue | stage.py:358-371, 373-396 | Encoder emits \`encode\`; fragment senders emit \`send\` with hop around pace + write. Proven by the multi-stream arm of \`test_step_telemetry_measures_ops_and_hops\` (HOP_RATE_RATIO_MULTI, send count == fragment count). |
| BA-6 | \`send\` events lack hop; senders built without key | stage.py:175-187, 271-291, 445-455, 548-578 | Hop label passed to every sender; summary groups by hop. |
| BA-7 | \`step_opt\` events | stage.py:894, 955 | Exist; summarized as \`OPT\`. |
| BA-8 | Existing per-slot telemetry file and its exit shipment | stage.py:595-607, 1099; agent.py:534-547 | File keeps receiving every event (now from the swapped list); format gains \`hop\`/\`encode\`; no consumer (E27). |
| BA-9 | Step loop order: optimizer → telemetry → checkpoint → \`WT_STEP\` | stage.py:967-990 | \`WT_TELEM\` inserted before \`_checkpoint\`; \`WT_STEP\` stays last per step, so progress still means "optimizer step done and wave written". |
| BA-10 | Loss stream replay on resume | stage_main.py:29-55, 63-65 | Not edited; neutralized at the coordinator (BA-12). |
| BA-11 | \`PROGRESS\` source | agent.py:463-491 | Adds \`src\`. |
| BA-12 | Progress stamping at arrival | coordinator.py:374-388 | Only \`src == "step"\`, via \`add_progress\` (range [start_step, steps−1], int). |
| BA-13 | \`median_step_s\`, \`last_progress_at\`, S4 limit | liveness.py:73-90, 217-228 | Formulas unchanged; their inputs no longer contain replay. |
| BA-14 | S4 after the final step | liveness.py:217-265 | Replaced by the shutdown bound in shutdown mode; S3 unchanged. |
| BA-15 | Per-attempt \`SlotState\` creation | coordinator.py:914-931, 1087-1118 | Sets \`start_step\` from the restored wave. |
| BA-16 | Restart / heal from wave \`steps − 1\` | coordinator.py:1142-1205; stage.py:651-679, 959-990 | Kept for S2/S3/convict (reruns \`_finish\`, exact by RUNTIME-I3); unreachable from S4 quiet. |
| BA-17 | Resume spec | fault_tolerance/checkpoint.py:143-153 | Untouched (FIT-4 citation corrected). |
| BA-18 | Reader thread death on a message exception | coordinator.py:297-310, 324-361 | Per-message catch, log, unlink body, continue. |
| BA-19 | \`TELEMETRY\` destination from message fields | coordinator.py:352-356 | From \`_slot_for\` + \`run.attempt\` only. |
| BA-20 | Agent post-exit path (seam point) | agent.py:493-518 | Delay only when \`WANTRAIN_SIM_EXIT_DELAY_S\` is set and rc == 0; heartbeats keep \`stage_alive\` true (agent.py:284-290). |
| BA-21 | Stage stdout pipe drained by the agent's pump, which calls \`send\` synchronously | agent.py:59-70, 213-221 | \`WT_TELEM\` adds ~1 KB/step to an existing exposure (a coordinator that stops reading back-pressures the stage). Not changed; Risk R4. |

# Tripwire

None. Reason: nothing consumes \`TELEM_STEP\` or the window in this phase, so telemetry without its consumer is inert,
and the consumer (re-plan) arrives in phase 2 behind its own switch (\`CoordinatorConfig.replan = False\`, armed by no
phase until M1 real-hardware evidence). Partial landings here are loud, not silent: a coordinator that requires
\`src == "step"\` without the agent's \`src\` sees no progress and fails at \`first_step_timeout_s\` in every integration
test; telemetry without intake is ignored by today's \`_dispatch\`. C1 and C2's agent edits land in one writer's tasks.

# Exit gate

Rungs, run twice in a row: factory, build, syntax, fmt, lint, types, discipline, invariants (CONTROL-I17/I18, RUNTIME-I9
covered), claims, unit, integration, multiworker, oracle, identity, chaos, wan, full, static, security, fuzz. N/A with
reason: frozen (its record is absent), component, multinode, matrix, rest, redteam (absent), bench (bench/ absent;
replaced by the measured telemetry numbers below).

| Number | Bound | Model | Measured by |
|---|---|---|---|
| SEAM_RATIO, slowed/unslowed F \`med_ms\` of the two peers of the wide stage, SIM_SLOW factor 4 | ≥ 2 | 4 | \`tests/integration/test_engine_bitwise.py::test_step_telemetry_measures_ops_and_hops\` (\`telemetry-engine\`) |
| HOP_RATE_RATIO, median wire rate on links[1] (160 Mb/s) / links[0] (40 Mb/s), rate = bytes·8/busy_ms; mesh path (wide stages: \`_connect_mesh\` → \`_wire\` → \`_sender\`) | ≥ 2.5 | 4 | same |
| HOP_RATE_RATIO_MULTI, the same ratio on the multi-stream path (chain, \`p2p_streams = 2\`: \`_wire_multi\` → \`_mux_encoder\` → \`_frag_sender\`) | ≥ 2.5 | 4 | same test, second arm |
| TELEM_US_MAX, max \`emit_us\` over every summary of the run | ≤ 1000 | ≤ 300 | \`tests/integration/test_coordinator.py::test_live_telemetry_reaches_the_coordinator_byte_exact\` (\`telemetry-intake\`) |
| TELEM_FRAC, median(\`emit_us\`)/1000 / median(\`step_ms\`) | ≤ 0.01 | ≤ 0.003 | same |
| TELEM_INTAKE_US_MAX, \`rationale["telemetry"]["handle_us_max"]\` | ≤ 2000 | ~50 | same |
| QUIET_S, quiet between the fleet's last \`WT_STEP\` and the first slot's exit report, with delay 45 s, \`first_step_timeout_s\` 30 s, floor 8 s | > 30 (asserted in test) | 45 | \`test_slow_shutdown_after_the_last_step_completes\` |
| Restarts in the slow-shutdown run | == 0 (asserted) | 0 | same |

Discriminating arms: the only performance number is the mechanism's own cost, measured in-band on its own block (one
factor). An on/off A/B of step time would sit below CPU step-time noise (σ ≫ 1 ms) and attribute nothing; not pinned.
SEAM_RATIO, HOP_RATE_RATIO and HOP_RATE_RATIO_MULTI each vary one factor (compute of one peer; bandwidth of one link)
with distinct expected values (E13). A constant implementation fails all three. An encode-only \`send\` fails only
HOP_RATE_RATIO_MULTI: on the mesh path the pre-phase \`send\` event already spans encode + pace + \`send_frame\`
(stage.py:548-578), so HOP_RATE_RATIO is ~4 there whatever the multi-stream path does; only the second arm, a chain with
\`p2p_streams = 2\`, runs \`_mux_encoder\` and \`_frag_sender\` (stage.py:256-291), where BA-5 sits.

Test contracts (every test states \`covers:\` and \`WOULD-FAIL-IF:\`):
- C1 unit (\`tests/unit/test_fault_tolerance.py\`, covers CONTROL-I17, injected clock):
  \`test_replayed_metric_progress_never_feeds_the_stall_clock\` (a burst of \`src="metric"\` and of out-of-range steps is
  not recorded; an S4 limit computed after it equals the limit without it; a coordinator that stamps metric progress
  fails); \`test_resumed_attempt_is_judged_by_the_first_step_timeout\` (two parts. Judge: start_step = 5, no real progress, quiet
  of floor + 1 s is \`ok\`, quiet of \`first_step_timeout_s\` + 1 is \`stalled\`. Relaunch: a \`Coordinator\` on \`tmp_path\` and a
  \`_Run\` of 2 slots with a 4-step spec, whose worker requests are replaced inside the test by a recorder (a fake at the
  worker-network boundary, named as such: \`_request\` returns \`{"ok": True}\`, \`_sync_all_connections\` does nothing); the
  real \`_restore_and_relaunch(run, 1, spec_json)\` gives every new \`SlotState\` \`start_step == 2\` (distinct from 0,
  \`steps − 1 = 3\` and \`steps = 4\`); a \`src="step"\` progress of step 1 is not recorded and one of step 2 is; judged with
  \`final_step = 3\` and no real progress, quiet of \`first_step_timeout_s\` + 1 is \`stalled\` (not shutdown); then
  \`_restore_and_relaunch(run, None, spec_json)\` gives \`start_step == 0\`. WOULD-FAIL-IF \`_launch\` sets \`start_step\` to
  \`steps\` or \`final_step + 1\` (shutdown from the start, every real \`WT_STEP\` dropped: S4 off for a resumed attempt) or
  leaves it at 0 on a mid-run relaunch; \`final-wave-relaunch\` cannot tell these apart, since there \`wave + 1 = steps\`); \`test_quiet_after_the_final_step_is_shutdown_not_a_stall\`
  (quiet > floor before the final step is \`stalled\`; after every slot reported \`steps − 1\` it is \`ok\`; past
  \`shutdown_limit_s\` it is \`failed\` with "shutdown exceeded"; one slot at \`steps − 1\` and another at \`steps − 2\`, quiet
  > floor, is \`stalled\`; one slot exited 0 and another still owes a step, quiet > floor, is \`stalled\`; a judge that only
  raises the floor fails case 1 or 3, and a judge that enters shutdown when any slot (not every slot) finished fails
  the two lagging-slot cases).
- C2 unit (covers CONTROL-I18): \`test_malformed_messages_never_stop_the_reader\` (a kindless message, a bodiless
  \`TELEMETRY\`, one with a non-int rank: each logged, the next \`HEARTBEAT\` still updates \`last_heartbeat\`, no file
  moved outside \`run_dir\`, stale bodies unlinked. Identity part, BA-19: a run at \`attempt\` 1 with two slots whose rank
  and peer both differ (e.g. (0,0) and (1,1)), slot (0,0) already holding \`run_dir/telemetry_r0p0_a1.jsonl\`; from the
  connection bound to (1,1), a \`TELEMETRY\` whose body is a real file and whose \`rank\` 0, \`peer\` 0 are ints naming the
  other slot, \`attempt\` 1: the body's bytes land at \`run_dir/telemetry_r1p1_a1.jsonl\`, (0,0)'s file is byte-unchanged
  and no other \`telemetry_*\` file appears; then from the same connection a \`TELEMETRY\` with \`attempt\` 0 (stale) and a
  real body: the body is unlinked, \`telemetry_r1p1_a0.jsonl\` does not exist and (1,1)'s file is unchanged. WOULD-FAIL-IF
  the branch builds the destination from the message's \`rank\`, \`peer\` or \`attempt\` (slot (0,0)'s file is overwritten)
  or moves a stale attempt's body); \`test_telem_step_identity_and_bounds_are_checked_at_the_boundary\`
  (every \`TELEM_STEP\` sent through the real \`_dispatch\` from the connection bound to its slot; a run with two slots
  whose rank and peer both differ, e.g. (0,0) and (1,1); one accepted summary; each of: wrong identity, step out of
  range, NaN, 5 KiB, unknown key, stale attempt → dropped and counted; distinct accepted/dropped counts. The wrong
  identity case names the other existing slot: from the connection bound to (1,1), a summary with \`rank\` 0, \`peer\` 0
  (ints) is dropped, and (0,0)'s window and the accepted count stay unchanged. WOULD-FAIL-IF the branch indexes the
  window by the summary's \`rank\`/\`peer\` instead of \`_slot_for\` (the summary lands in (0,0)'s window)).
- R1 unit (\`tests/unit/test_runtime_spec.py\`, covers RUNTIME-I9): \`test_step_summary_is_a_timing_only_median\`
  (synthetic events with distinct durations where mean ≠ median; overlapping send intervals where union ≠ sum; RNG
  state unchanged; \`parse_sim_slow\` rejects malformed values); \`test_telemetry_events_are_neither_lost_nor_duplicated_across_threads\`
  (on a \`StageRuntime\` built without sockets; two parts. Deterministic: while the test itself holds \`rt._tel_lock\`, an
  \`_ev\` call in one thread and a \`_take_events()\` call in another are both still blocked after 0.2 s; after the release
  both finish, and the event appears exactly once across the two takes. Stress: \`sys.setswitchinterval(1e-6)\` (restored
  after), 4 appender threads × 2000 \`_ev\` with distinct ids racing a loop of \`_take_events()\`; the union of all takes
  holds every id exactly once. WOULD-FAIL-IF \`_ev\` or \`_take_events\` does not hold \`_tel_lock\` (the deterministic part
  fails on every run, whatever the GIL schedule), or the take iterates and then \`clear()\`s (the stress part)).
- Integration: \`test_step_telemetry_measures_ops_and_hops\` (runtime; local run, widths [1,2,1], M = 8, \`wire_codec="raw"\`,
  links [40, 160] Mb/s, \`WANTRAIN_SIM_SLOW=1:1:4\`; one \`WT_TELEM\` per slot per step parsed from the stage logs, F \`n\`
  per slot equals its owned microbatches, ops complete (below), every paced hop's rate ≤ 1.1 × its link, manifests \`diff_manifests(...) ==
  []\` and \`==\` against \`reference_manifest_for\`; prints \`SEAM_RATIO=\`, \`HOP_RATE_RATIO=\`; covers RUNTIME-I9). Second
  arm in the same test, the multi-stream path: a chain of 3 stages (no wide stage), \`p2p_streams = 2\`, M = 8,
  \`wire_codec="raw"\`, links [40, 160] Mb/s, no SIM_SLOW; in each slot's \`<tag>_rank<r>p<p>_telemetry.jsonl\` every
  \`send\` event carries a \`hop\`, and the number of \`send\` events equals the sum of \`frags\` over its \`encode\` events (the
  fragment senders emit \`send\`, the encoder emits \`encode\`); every paced hop's rate ≤ 1.1 × \`p2p_streams\` × its link;
  manifests equal the reference as in the first arm; ops complete (below) in both arms; prints \`HOP_RATE_RATIO_MULTI=\` and
  \`OPS_CHECKED=\`, the number of summaries whose ops and hop frames it checked over both arms. WOULD-FAIL-IF the encoder still
  emits \`send\` around encode + enqueue (the ratio is ~1 and the counts differ).
  *Ops complete* (both integration telemetry tests, every summary they read): \`ops\` F, B and W each have \`n\` equal to
  the number of that op the slot's schedule runs in that step, derived from the spec (its owned microbatches × its
  virtual stages), never from the events; OPT \`n\` == 1; every op's \`med_ms\` > 0. Per slot and per hop \`h\`, with the
  slot's telemetry file (the event log the same take appends to): count of its \`send\` events with \`hop == h\` and
  \`step ≤ steps − 2\` ≤ Σ over the slot's summaries of \`hops[h].frames\` ≤ count of all its \`send\` events with \`hop == h\`.
  The upper bound holds exactly (every taken event is appended to the file). The lower bound holds by the step order:
  a frame of step ≤ \`steps − 2\` is received before any slot can finish step \`steps − 1\`, so its \`send\` completed
  before the slot's final summary; only the final step's sends (and the shutdown drain) may complete after it, and
  they reach the file only. Exact equality is not required for that reason. WOULD-FAIL-IF \`summarize_step\` maps an
  event name wrongly (stage events are \`fwd\`, \`bwd_b\`, \`bwd_w\`, \`step_opt\`, \`stall\`, \`send\`, \`encode\`: a wrong name
  gives \`n\` 0 or \`med_ms\` 0) or drops the sends that complete after their own step's summary.
  \`test_slow_shutdown_after_the_last_step_completes\` (control; \`LocalCluster(..., first_step_timeout_s=30)\` with only
  today's config fields, \`monkeypatch.setenv("WANTRAIN_SIM_EXIT_DELAY_S", "45")\`, 2 stages, 4 steps (E35: the pre-phase
  limit is then ≤ 30 s whether or not the median exists); assertions in this order and exactly these messages:
  \`assert res.ok, f"RUN_FAILED: {res.reason}"\`, \`assert res.restarts == 0, f"RESTARTED: {res.restarts} restart(s)"\`,
  \`assert quiet_s > first_step_s, f"QUIET_TOO_SHORT: {quiet_s:.1f}s"\` (\`first_step_s = 30\`), then the manifest checks; no
  \`pytest.fail\`, no multi-line message; it uses no name, config field or message field this phase adds; covers
  CONTROL-I17). \`test_relaunch_from_the_final_wave_runs_no_step_and_is_not_restarted\`
  (control; \`LocalCluster(..., first_step_timeout_s=20)\` (an existing field), 3 workers for 2 stages (one spare), 4
  steps, \`monkeypatch.setenv("WANTRAIN_SIM_EXIT_DELAY_S", "30")\`; waits until every slot reported step 3 and
  \`run.store.common_step(list(run.slots)) == 3\`, then kills one assigned worker's agent process; asserts \`res.ok\`,
  exactly one heal with \`resumed_from_step == 3\`, \`res.restarts == 0\`, manifests equal the reference; covers
  CONTROL-I17; WOULD-FAIL-IF \`_launch\` leaves \`start_step\` at 0: the relaunched attempt runs no step, its 30 s quiet
  exceeds \`first_step_timeout_s\` and it is restarted until the budget is spent).
  \`test_live_telemetry_reaches_the_coordinator_byte_exact\` (control; widths
  [2,1]; \`telem_steps.jsonl\` holds exactly one summary per slot per step, rank/peer equal the slot, F \`n\` equals the
  slot's owned count, ops complete (above; the slot's telemetry file is the one shipped at exit,
  \`run_dir/telemetry_r<r>p<p>_a<attempt>.jsonl\`), \`dropped == 0\`, manifests equal the reference; prints \`TELEM_US_MAX=\`,
  \`TELEM_FRAC=\`, \`TELEM_INTAKE_US_MAX=\`, \`OPS_CHECKED=\`, each fixed-point \`:.6f\`; covers CONTROL-I18, RUNTIME-I9).
- Pre-phase tests: every top-level statement of the four owned test files at E34 stays AST-identical; new tests, imports
  and helpers are added as new statements, never by editing an existing one (\`pre-tests-unchanged\`). A new top-level
  statement is only an import, a function definition or an assignment to plain names; it never rebinds a name the
  pre-phase file binds at top level (a later \`def test_<same name>\` would replace the old body at import), never
  declares a fixture with \`autouse\` and never defines a \`pytest_*\` hook. It declares no fixture at all (new tests use
  \`tmp_path\`, \`monkeypatch\`, \`port_base\` and plain helpers): a fixture named, by function name or \`name=\`, like one a
  pre-phase test requests would change how that test runs. A new def carries no decorator other than
  \`pytest.mark.<name>\` with call-free arguments and has no call in its defaults or annotations (both run at import); a
  new assignment has no call anywhere in it (only constants, names, attribute reads, subscripts, operators, tuples,
  lists, dicts, sets and call-free comprehensions: a computed value belongs in a def called inside the test, since a
  call at import, e.g. \`_lv.__dict__.update(...)\`, would change what every later test of the session sees) and binds
  no fixture name a pre-phase function or \`tests/conftest.py\` uses. The pre-phase statements keep their relative order
  (new statements may sit between them, e.g. an import in its sorted place).
- Mutants (\`would-fail-mutants\`, E13 made mechanical for the four claims no other gate runs against their regression):
  a pytest plugin defined in the GATES entry, loaded with \`-p\`, breaks one thing at run time and the named node must
  fail with an \`AssertionError\` (exit 1, \`1 failed\`, no \`TypeError\`/\`AttributeError\`/\`KeyError\`-class error, and the
  plugin recorded that its mutation was reached). \`lock\`: \`StageRuntime._tel_lock\` becomes a lock that never blocks →
  \`test_telemetry_events_are_neither_lost_nor_duplicated_across_threads\` fails. \`launch-zero\` / \`launch-late\`: after
  \`Coordinator._restore_and_relaunch(run, step, ...)\` with a wave, every new \`SlotState.start_step\` is set to 0 / to
  its value + 2 (\`steps\` in the 4-step spec) → \`test_resumed_attempt_is_judged_by_the_first_step_timeout\` fails.
  \`identity\`: \`Coordinator._slot_for\` returns the slot the \`TELEMETRY\` message's \`rank\`/\`peer\` name →
  \`test_malformed_messages_never_stop_the_reader\` fails. \`identity-telem\`: \`Coordinator._slot_for\` returns the slot a
  \`TELEM_STEP\` message's \`summary\` \`rank\`/\`peer\` name → \`test_telem_step_identity_and_bounds_are_checked_at_the_boundary\`
  fails. So these tests must drive the real \`_restore_and_relaunch\`, send \`TELEMETRY\` and \`TELEM_STEP\` through
  \`_dispatch\` (whose branches take the slot from \`_slot_for\`), and hold \`rt._tel_lock\`, the
  attribute \`_ev\` and \`_take_events\` lock. The mutants are fixed by this granted text, not written by the test's
  author; no \`.would-fail.patch\` is used because the phase owns no path for one.
- \`causal-slow-shutdown\` runs \`tests/integration/test_coordinator.py\` against the pre-phase \`liveness.py\` and
  \`coordinator.py\`: at module level that file imports only names that exist in them; a name this phase adds is imported
  inside the test function that needs it (an \`ImportError\` at collection is \`CAUSE=other\`).

Pinned entries (\`GATES-m2-p1-liveness-telemetry.json\`, 14): \`spec-amended\` (exactly one
\`docs/design/adr/ADR-NNNN-m2-p1-spec.md\`, the human grant of it and of \`docs/design/DESIGN.md\`, then the greps),
\`no-pending-markers\` (grep exit 1 only), \`no-new-skip-markers\` (widened scan, digest E33), \`pre-tests-unchanged\` (the
Gate's saved copies at digests E34; every pre-phase top-level statement of the four owned test files is still present,
AST-equal and in their pre-phase relative order; every new top-level statement is an import, a def or a plain-name
assignment that rebinds no pre-phase top-level name or requested fixture name, with no \`autouse\`, no \`pytest_*\` hook,
no fixture, no decorator but \`pytest.mark.<name>\` with call-free arguments, no call in a new def's signature and no
call anywhere in a new assignment), \`liveness-unit\` ×20, \`intake-unit\` ×5, \`telemetry-unit\` ×20, \`would-fail-mutants\`
(5 mutants, each caught),
\`slow-shutdown-cpu\` ×3 under 2× load,
\`final-wave-relaunch\` ×3, \`causal-slow-shutdown\` (digests E32, arm A must fail with \`CAUSE=stall\`, arm B must pass,
2 runs, 900 s per run, 4200 s outer; \`CAUSE=stall\` needs at least one \`fleet stalled: no optimizer step fleet-wide\`
coordinator line, no \`fleet stalled:\` line of another kind, a \`RUN_FAILED:\` naming that S4 reason and the restart budget
or a \`RESTARTED:\` assertion, and no conviction, no thread exception, no \`KeyError\`/\`TypeError\`-class error; any other
failure of arm A is \`CAUSE=other\`), \`telemetry-engine\` ×3 (SEAM_RATIO ≥ 2, HOP_RATE_RATIO ≥ 2.5,
HOP_RATE_RATIO_MULTI ≥ 2.5 and OPS_CHECKED ≥ 7: 4 slots + 3 slots, at least one step each), \`telemetry-intake\` ×3
(OPS_CHECKED ≥ 3: 3 slots), \`suite-nodes\` (frozen list digest E33; the canonical command
without \`-u\` prints no duplicate node, so no test function is shadowed by a later one of the same name; every
pre-phase function and the 11 new nodes \`PASSED\`; exactly two \`SKIPPED\` lines, both GPU_OFF; no
\`FAILED\`/\`ERROR\`/\`XFAIL\`/\`XPASS\`). Every node-naming entry requires pytest exit 0 and the exact \`<n> passed\` summary.
Every \`KEY=\` value a test prints for a gate is fixed-point, \`print(f"KEY={value:.6f}")\` (never \`str\`/\`repr\` of a
float, which prints \`5e-05\` below 1e-4); the gates' awk and metric patterns also read exponents (\`[0-9.eE+-]+\`).
The phase workflow accepts the relayed verdict only for this gate id (\`milestone\`), with every one of these entries,
the 20 rungs and \`consecutive\` present and passing, on the tree it audited before and after the verify.
Canonical list command (Gate step and E33):
\`grep -rnE '^(    )?(async )?def test_[A-Za-z0-9_]+' tests --include='test_*.py' | sed -E 's/^([^:]+):[0-9]+:[[:space:]]*(async )?def (test_[A-Za-z0-9_]+).*/\\1::\\3/' | LC_ALL=C sort -u\`.

# Risks

- **R1 — \`shutdown_limit_s\` too low for large shards.** T_quiet grows with S_shard/B_ctrl; past 600 s a completed run is
  reported failed (weights stay on worker disks and the last shipped wave). Configurable; the failure text names the
  knob. A size-aware bound is a later decision.
- **R2 — CPU timing noise** in SEAM_RATIO and HOP_RATE_RATIO: bounds sit at half (2/4) and 0.6× (2.5/4) of the model,
  each run ×3. A failure is evidence against the model, never a reason to widen a bound.
- **R3 — the MPS stall may have a second cause.** M2-MPS-INTERMITTENT stays open until a gpu-true \`/wan-bughunt\`.
- **R4 — stdout back-pressure** (BA-21): pre-existing; telemetry adds ~1 KB/step. Recorded for phase 2's review.
- **R5 — spec-amendment precondition** needs a human grant before the phase starts.
- **R6 — real hardware:** all numbers SIMULATED; device-sync effects on GPU op timing (asynchronous kernels make CPU
  timestamps measure launch, not execution) are unverified and matter for phase 2's re-solve. M1 work.
`
const MAX_STABILIZE = 4
const PINNED_RESULT = {
  type: 'object',
  required: ['pinned', 'items'],
  properties: {
    pinned: { type: 'object', description: `the JSON printed by ${PINNED}, verbatim` },
    items: TRIAGE.properties.items,
  },
}

// Two writers (NEXT, Scope): R owns the runtime half, C the control half. C delivers C1 in wave 1 and C2 in wave 2.
const WRITERS = {
  runtime: ['src/wantrain/runtime/stage.py', 'src/wantrain/runtime/telemetry.py', 'tests/unit/test_runtime_spec.py', 'tests/integration/test_engine_bitwise.py'],
  control: ['src/wantrain/fault_tolerance/liveness.py', 'src/wantrain/coordinator/coordinator.py', 'src/wantrain/worker/agent.py', 'tests/unit/test_fault_tolerance.py', 'tests/integration/test_coordinator.py'],
}
const OWNED = Object.values(WRITERS).flat()
// Journals live in the gitignored run state: a later fix of a writer's files starts from its author's reasoning.
const JOURNAL = { runtime: journalPath(SLUG, 'runtime'), control: journalPath(SLUG, 'control') }
const WORK = [
  {
    key: 'r1-telemetry', writer: 'runtime', wave: 1,
    binding: ['BA-1', 'BA-2', 'BA-3', 'BA-4', 'BA-5', 'BA-6', 'BA-7', 'BA-8', 'BA-9'],
    task: `R1 — per-step telemetry (RUNTIME-I9), exactly as "Mechanism, R1" in the facts: _ev under self._tel_lock and
_take_events() swapping the list under it; hop labels next<q>/prev<q>/intra<k> passed to _sender and _frag_sender; in the
multi-stream path the encoder emits "encode" and each fragment sender emits "send" with hop and nbytes around pace +
send_frame; new pure src/wantrain/runtime/telemetry.py (summarize_step, parse_sim_slow; no torch, no clock, no RNG);
in run(), after _optimizer_step(step), one perf_counter-timed block that takes the events, appends them to the existing
telemetry file, summarizes and prints WT_TELEM <json> (<= 4 KiB) before _checkpoint and WT_STEP, its duration being the
next summary's emit_us (first is 0); step_ms from _run_step start to the end of _optimizer_step; the WANTRAIN_SIM_SLOW
seam read once in StageRuntime.__init__, sleeping (factor - 1) x elapsed inside each F/B/W span before its _ev.
Tests, per "Test contracts": tests/unit/test_runtime_spec.py::test_step_summary_is_a_timing_only_median and
::test_telemetry_events_are_neither_lost_nor_duplicated_across_threads (its deterministic part: while the test holds
rt._tel_lock, an _ev and a _take_events() in other threads are both still blocked after 0.2 s, so an unlocked
implementation fails on every run whatever the GIL schedule; its stress part under sys.setswitchinterval(1e-6));
tests/integration/test_engine_bitwise.py::test_step_telemetry_measures_ops_and_hops with BOTH arms: the mesh arm (widths
[1,2,1], M = 8, wire_codec="raw", links [40, 160] Mb/s, WANTRAIN_SIM_SLOW=1:1:4, prints SEAM_RATIO= and HOP_RATE_RATIO=)
and the multi-stream arm (a 3-stage chain, p2p_streams = 2, links [40, 160] Mb/s: the only arm that runs _mux_encoder and
_frag_sender, where BA-5 sits; send count == sum of encode frags in each slot's telemetry file, prints
HOP_RATE_RATIO_MULTI=), manifests equal to the reference in both, and "ops complete" in both arms as the facts define it
(every summary: F, B and W n equal to the count the slot's schedule runs that step, derived from the spec, never from the
events; OPT n == 1; every med_ms > 0; per slot and hop, count of file send events with step <= steps - 2 <= sum of
hops[h].frames <= count of all file send events of that hop), printing OPS_CHECKED= (summaries checked, both arms,
f"OPS_CHECKED={value:.6f}"; the gate needs >= 7). summarize_step must map the stage's real event names (fwd, bwd_b,
bwd_w, step_opt, stall, send, encode) and count every send taken since the previous summary. Distinct expected values (mean != median, union != sum)
so a mean, a sum or an encode-timed hop fails a named assertion. The lock mutant replaces StageRuntime._tel_lock with a
lock that never blocks: _ev and _take_events must take exactly self._tel_lock (no other lock object), and the
deterministic part must fail with an assertion when they do not block.`,
  },
  {
    key: 'c1-liveness', writer: 'control', wave: 1,
    binding: ['BA-10', 'BA-11', 'BA-12', 'BA-13', 'BA-14', 'BA-15', 'BA-16', 'BA-17', 'BA-20'],
    task: `C1 — liveness accounting (CONTROL-I17), exactly as "Mechanism, C1" in the facts: SHUTDOWN_LIMIT_S = 600.0,
SlotState.start_step, SlotState.add_progress(step, t, final_step) -> bool, LivenessJudge(..., final_step=None,
shutdown_limit_s=SHUTDOWN_LIMIT_S) with the shutdown mode replacing only the S4 block (S1-S3, grace, crashed and strikes
unchanged; final_step=None keeps today's behavior exactly); CoordinatorConfig.shutdown_limit_s = 600.0, _judge passing
final_step = spec.steps - 1, _launch setting start_step = wave + 1 on a relaunch from a wave, _slot_update recording
progress only for src == "step" with an integer step through add_progress; agent.py PROGRESS gains "src" ("step" from
WT_STEP, "metric" from WT_METRIC) and the WANTRAIN_SIM_EXIT_DELAY_S seam (float in [0, 3600], malformed raises
ValueError, read once at agent start, sleep after stage.wait() returns 0 and the post-exit _ship_checkpoint, before EXIT).
stage_main.py is NOT yours and stays untouched (BA-10: the replay is neutralized at the coordinator).
Tests, per "Test contracts": tests/unit/test_fault_tolerance.py::test_replayed_metric_progress_never_feeds_the_stall_clock,
::test_resumed_attempt_is_judged_by_the_first_step_timeout (both parts: the judge part, and the relaunch part that drives
the real _restore_and_relaunch -> _launch from wave 1 of a 4-step spec with only the worker requests faked, asserting
start_step == 2 on every new SlotState, a step-1 WT_STEP not recorded and a step-2 one recorded, quiet of
first_step_timeout_s + 1 judged stalled with final_step = 3, and start_step == 0 without a wave: the only check that tells
wave + 1 from steps or final_step + 1, since final-wave-relaunch relaunches from steps - 1 where they coincide),
::test_quiet_after_the_final_step_is_shutdown_not_a_stall
(injected clock; the shutdown test includes the two lagging-slot cases — one slot at steps-1 and another at steps-2, and
one slot exited 0 while another still owes a step, each with quiet > floor -> stalled — so shutdown entry by "any slot"
instead of "every slot" fails), and tests/integration/test_coordinator.py::test_slow_shutdown_after_the_last_step_completes with
exactly the recipe (LocalCluster(..., first_step_timeout_s=30), WANTRAIN_SIM_EXIT_DELAY_S=45, 2 stages, 4 steps: E35 shows
why the pre-phase limit is then at most 30 s whether or not the median exists) and exactly the assertion order and
messages the contract gives (RUN_FAILED:, RESTARTED:, QUIET_TOO_SHORT:; no pytest.fail, no multi-line message): the
causal-slow-shutdown gate classifies the pre-phase failure by those messages and by the coordinator's "fleet stalled: no
optimizer step fleet-wide" lines. causal-slow-shutdown runs tests/integration/test_coordinator.py against the PRE-PHASE
liveness.py and coordinator.py: at module level that file imports only names that exist in them (import a name this
phase adds inside the test function that needs it), and the slow-shutdown test uses no name, config field or message
field this phase adds, or arm A fails at collection as CAUSE=other. Also
tests/integration/test_coordinator.py::test_relaunch_from_the_final_wave_runs_no_step_and_is_not_restarted exactly as its
contract says (first_step_timeout_s=20, exit delay 30 s, one spare, kill after wave 3 is held for every slot; one heal
resumed from step 3, 0 restarts, manifests equal the reference): it is the only gate that runs _launch's start_step on a
real relaunch. The launch-zero and launch-late mutants set every new SlotState.start_step to 0 / to its value + 2 right
after the real _restore_and_relaunch(run, 1, spec_json) returns: the relaunch part must call that method (never build
the SlotStates by hand) and assert start_step == 2 and the step-1/step-2 recording on what it produced.`,
  },
  {
    key: 'c2-intake', writer: 'control', wave: 2,
    binding: ['BA-18', 'BA-19', 'BA-21'],
    task: `C2 — telemetry intake (CONTROL-I18), exactly as "Mechanism, C2" in the facts, on top of R1's WT_TELEM line
format and your own C1 (read both in the current files, not only the summaries below): the agent forwards a WT_TELEM
line of <= 4 KiB that parses as a JSON object as {"kind": "TELEM_STEP", "run_id", "attempt", "summary"} and drops anything
else (never reusing TELEMETRY); _conn_loop wraps each _dispatch so one message's exception is logged as
"warn malformed <kind> from <worker>: <error>", its file body unlinked, and reading continues (only channel errors end the
loop); the TELEMETRY branch takes rank, peer and attempt from _slot_for and run.attempt, never from the message, and drops
(unlinking the body) a missing slot, a stale attempt or a missing/non-path body; the TELEM_STEP branch accepts only a dict
of <= 4096 bytes re-serialized, exactly the known keys, v == 1, rank/peer equal to the slot, an int step in
[start step, steps - 1], every number finite with times in [0, 1e6] ms and counts/bytes in [0, 1e12]; accepted summaries
go to a per-slot deque(maxlen=64) and to <run_dir>/telem_steps.jsonl with attempt until telem_log_max_bytes (64 MiB, then
one warn and no more writes); run.rationale["telemetry"] = {"accepted", "dropped", "handle_us_max"} (perf_counter around
the branch). Nothing reads the window. BA-21 (stdout back-pressure) is recorded, not changed.
Tests, per "Test contracts": tests/unit/test_fault_tolerance.py::test_malformed_messages_never_stop_the_reader (including
its BA-19 identity part: a TELEMETRY naming another slot's rank and peer lands at the connection's own slot and
run.attempt, the other slot's file byte-unchanged; a stale attempt's body is unlinked, not moved) and
::test_telem_step_identity_and_bounds_are_checked_at_the_boundary (distinct accepted/dropped counts; every TELEM_STEP
sent through the real _dispatch from the connection bound to its slot; its wrong-identity case is a summary with rank 0,
peer 0 sent from the connection bound to (1,1), which must be dropped with (0,0)'s window and the accepted count
unchanged), and
tests/integration/test_coordinator.py::test_live_telemetry_reaches_the_coordinator_byte_exact (widths [2,1], prints
TELEM_US_MAX=, TELEM_FRAC=, TELEM_INTAKE_US_MAX=, OPS_CHECKED= (the gate needs >= 3), each printed fixed-point as
f"KEY={value:.6f}", "ops complete" as the facts define it for every summary in telem_steps.jsonl against the slot's
telemetry file shipped at exit, manifests equal to the reference). causal-slow-shutdown runs
tests/integration/test_coordinator.py against the PRE-PHASE liveness.py and coordinator.py: add no module-level import of
a name this phase creates there (import it inside the test function that needs it), or arm A fails at collection.
The identity mutant makes Coordinator._slot_for return the slot a TELEMETRY message's rank/peer name: the TELEMETRY
branch takes its slot from _slot_for (as the contract says), and the identity part sends its TELEMETRY through the real
_dispatch from the connection bound to (1,1), so that mutant overwrites (0,0)'s file and a named assertion fails. The
identity-telem mutant makes _slot_for return the slot a TELEM_STEP summary's rank/peer name: the TELEM_STEP branch takes
its slot from _slot_for, so under that mutant the rank 0, peer 0 summary from (1,1) lands in (0,0)'s window and a named
assertion fails.`,
  },
]

// Who wrote a file: a fix inside one writer's paths starts from that writer's journal.
const ownerOf = scope => {
  const w = Object.keys(WRITERS).find(k => scope.every(p => WRITERS[k].includes(p)))
  return w ? { ownerTask: w, ownerJournal: JOURNAL[w] } : {}
}
const inOwned = p => OWNED.includes(p)

phase('Gate')
const grant = await checkApprovals([NEXT, GATES, WORKFLOW], 'gate:approval', 'Gate')
if (!approvedAll(grant, [NEXT, GATES, WORKFLOW])) throw new Error(`${NEXT}, ${GATES} and ${WORKFLOW} are not all approved by a human — run: node agents-architetura/tools/factory/approve.mjs grant ${NEXT} ${GATES} ${WORKFLOW}`)

// The phase builds on spec text the m2-p1-spec ADR writes. wan-adr writes that text in its draft, before any grant, so the
// greps below prove nothing alone: the human grant of DESIGN.md and of that ADR is checked here. The ADR's path is built
// by the script from a validated integer, never taken as text from an agent.
const ADR_ROUTE = `run /wan-adr {"problem": "apply ${NEXT} §Spec amendments verbatim to SPEC-control.md and SPEC-runtime.md, with history entries", "id": "m2-p1-spec"}; after the human grant of the ADR and ${DESIGN}, rerun this phase`
const ADR_FOUND = {
  type: 'object',
  required: ['found', 'unique', 'number'],
  properties: {
    found: { type: 'boolean', description: 'true only if docs/design/adr/ holds a file named ADR-<4 digits>-m2-p1-spec.md' },
    unique: { type: 'boolean', description: 'true only if exactly one such file exists' },
    number: { type: 'integer', minimum: 1, maximum: 9999, description: 'its 4-digit number as an integer (any value in range when found is false)' },
  },
}
const adrFound = await agent(common('READ-ONLY.') + `
List docs/design/adr/ (it may not exist). Report whether a file named exactly ADR-<4 digits>-m2-p1-spec.md exists, whether it
is the only one, and its number. Change nothing.`,
  { label: 'gate:adr', phase: 'Gate', schema: ADR_FOUND, agentType: ROLE.readonly })
const adrNumber = adrFound?.number
if (!adrFound?.found || !adrFound.unique || !Number.isInteger(adrNumber) || adrNumber < 1 || adrNumber > 9999) {
  return { ok: false, stoppedAt: 'gate', reason: 'no single docs/design/adr/ADR-NNNN-m2-p1-spec.md (the spec amendments of the NEXT document)', next: ADR_ROUTE }
}
const ADR = `docs/design/adr/ADR-${String(adrNumber).padStart(4, '0')}-m2-p1-spec.md`
const specGrant = await checkApprovals([DESIGN, ADR], 'gate:approval-spec', 'Gate')
if (!approvedAll(specGrant, [DESIGN, ADR])) {
  const why = clean((specGrant?.docs ?? []).filter(x => !x?.approved).map(x => `${x?.doc}: ${x?.reason ?? 'not approved'}`).join('; ') || 'no approval record', 400)
  return { ok: false, stoppedAt: 'gate', reason: `${DESIGN} and ${ADR} are not both granted by a human (${why})`, next: `a human reviews and grants ${ADR} and ${DESIGN} (/wan-approve), then reruns this phase` }
}

// NEXT, Phases 1: the spec amendments are a precondition (the invariants gate fails on ids no spec defines); exactly one
// m2-p1-spec ADR exists and it is the one granted above; the four owned test files are saved for pre-tests-unchanged only
// while they match E34 and the C1 files for causal-slow-shutdown only while they match E32 (an existing save is kept only
// if its digests match); the pre-phase test list must hash to E33. Any mismatch stops for a human: nothing is
// re-baselined here.
const PRE_CMD = String.raw`set -u
d=.factory/runs/m2-p1-liveness-telemetry; c=$d/causal-c1; n=$d/nodes-pre.txt; t=$d/tests-pre
L=src/wantrain/fault_tolerance/liveness.py; K=src/wantrain/coordinator/coordinator.py
T1=tests/unit/test_fault_tolerance.py; T2=tests/integration/test_coordinator.py; T3=tests/unit/test_runtime_spec.py; T4=tests/integration/test_engine_bitwise.py
fail() { echo "$1"; echo PRE_FAIL; exit "$2"; }
k=$(ls docs/design/adr 2>/dev/null | grep -cE '^ADR-[0-9]{4}-m2-p1-spec\.md$')
[ "$k" -eq 1 ] && [ -f '` + ADR + String.raw`' ] || fail "SPEC_AMENDED=no ADR_MATCHES=$k" 3
grep -q 'CONTROL-I17' docs/design/SPEC-control.md && grep -q 'CONTROL-I18' docs/design/SPEC-control.md && grep -q 'RUNTIME-I9' docs/design/SPEC-runtime.md && grep -q 'shutdown_limit_s' docs/design/SPEC-control.md && grep -q 'TELEM_STEP' docs/design/SPEC-control.md && grep -q 'WT_TELEM' docs/design/SPEC-runtime.md || fail SPEC_AMENDED=no 3
echo SPEC_AMENDED=yes
saved() { printf '%s  %s\n' ac243d335750dfdc79c4f2d90751a159805aac6f770ed4e9d51591850d952f07 "$c/manifest.json" 4132edaba91a38efcfd104cdc02cb69bda5f32cf061c40e962a84de1344467a6 "$c/files/$L" a02f76871ccb4d4b16e43495136093a87aa4e12e4b6c740ee0466aa20ffe226e "$c/files/$K" | shasum -a 256 -c - >/dev/null 2>&1; }
if [ -e "$c" ]; then
  saved || fail CAUSAL=existing-save-mismatch 4
  echo CAUSAL=kept
else
  printf '%s  %s\n' 4132edaba91a38efcfd104cdc02cb69bda5f32cf061c40e962a84de1344467a6 "$L" a02f76871ccb4d4b16e43495136093a87aa4e12e4b6c740ee0466aa20ffe226e "$K" | shasum -a 256 -c - >/dev/null 2>&1 || fail CAUSAL=live-files-changed 4
  node agents-architetura/tools/verify/causal.mjs save --out "$c" -- "$L" "$K" >/dev/null || fail CAUSAL=save-failed 4
  saved || fail CAUSAL=save-mismatch 4
  echo CAUSAL=saved
fi
listed() { printf '%s  %s\n' 21e0c82f276af6b090806045212de1087e2b18def459eedf4eb84aa566a20f38 "$n" | shasum -a 256 -c - >/dev/null 2>&1; }
if [ -e "$n" ]; then
  listed || fail NODES=existing-list-mismatch 5
  echo NODES=kept
else
  mkdir -p "$d" || fail NODES=mkdir-failed 5
  grep -rnE '^(    )?(async )?def test_[A-Za-z0-9_]+' tests --include='test_*.py' | sed -E 's/^([^:]+):[0-9]+:[[:space:]]*(async )?def (test_[A-Za-z0-9_]+).*/\1::\3/' | LC_ALL=C sort -u >"$n"
  listed || { rm -f "$n"; fail NODES=list-mismatch 5; }
  echo NODES=written
fi
tsums() { printf '%s  %s\n' 70dab584baff9a8d6dc7b97868ae539225fbca3ff5a04573f996607744685030 "$1/$T1" 2d42e69ca339b12f5eb5aeb8ddbaac706a12258dbe17b7aca3a577ab5d1382b1 "$1/$T2" af63f4c7f8f218dc105e976146deb05ca7273eeb9c71480b84e092a398d7a8a3 "$1/$T3" 0b1a8873a4befedafc10ec128f17766edb46f1af5e4628e3756525c4073da99a "$1/$T4" | shasum -a 256 -c - >/dev/null 2>&1; }
if [ -e "$t" ]; then
  tsums "$t" || fail TESTS_PRE=existing-save-mismatch 6
  echo TESTS_PRE=kept
else
  tsums . || fail TESTS_PRE=live-files-changed 6
  for f in "$T1" "$T2" "$T3" "$T4"; do mkdir -p "$t/$(dirname "$f")" && cp "$f" "$t/$f" || fail TESTS_PRE=copy-failed 6; done
  tsums "$t" || { rm -rf "$t"; fail TESTS_PRE=save-mismatch 6; }
  echo TESTS_PRE=saved
fi
echo PRE_OK`
const PRE_RESULT = {
  type: 'object',
  required: ['verdict', 'exitCode', 'output'],
  properties: {
    verdict: { type: 'string', enum: ['PRE_OK', 'PRE_FAIL'], description: 'PRE_OK only if the exit code is 0 and the last line printed is PRE_OK' },
    exitCode: { type: 'number', description: 'the real exit code of the command' },
    output: { type: 'string', description: 'everything the command printed, verbatim' },
  },
}
// Fail closed on the audit's own report: the review lenses and the stabilize review are driven by `changed`, which the
// AUDIT schema does not require. A missing list, a bad tree id, or a tree change without changed files (or the reverse)
// stops the phase instead of silently skipping a review.
const audited = async (label, phaseTitle, since, scopes) => {
  const a = await scopeCheck(label, phaseTitle, since, scopes)
  if (!Array.isArray(a.changed) || !a.changed.every(f => typeof f === 'string') || !TREE_ID_RE.test(a.tree ?? '')) {
    throw new Error(`scope audit (${label}) did not report its changed files and its tree — stopping rather than skipping a review`)
  }
  if ((a.tree === since) !== (a.changed.length === 0)) {
    throw new Error(`scope audit (${label}) is inconsistent: tree ${a.tree === since ? 'unchanged' : 'changed'} with ${a.changed.length} changed file(s)`)
  }
  return a
}

// The precondition agent runs a shell: no tracked file may change while it runs (its saves land in the gitignored .factory/).
const preSince = await scopeSnapshot('precondition', 'Gate')
const pre = await agent(common('You run ALONE — you MAY run this one command. Change nothing else, fix nothing.') + `
Run exactly this bash script from the repo root, once (for example: bash -c '<script>' or a here-document into bash):
${PRE_CMD}
Return its real exit code, everything it printed verbatim, and the verdict (PRE_OK only if the exit code is 0 and the last
line is PRE_OK). Never edit a spec, a digest or a saved file to make it pass.`,
  { label: 'gate:precondition', phase: 'Gate', schema: PRE_RESULT, agentType: ROLE.verifier })
await audited('precondition', 'Gate', preSince, [])
if (pre?.verdict !== 'PRE_OK' || pre?.exitCode !== 0) {
  const output = clean(pre?.output ?? 'the precondition agent returned nothing', 1500)
  if (/SPEC_AMENDED=no/.test(output)) return { ok: false, stoppedAt: 'gate', reason: output, next: `the spec amendments are missing or ambiguous: ${ADR_ROUTE}` }
  return { ok: false, stoppedAt: 'gate', reason: output, next: `a human decides: the pre-phase files, test files or test list no longer match the digests E32/E33/E34 of ${NEXT} (rerun wan-next from the current state)` }
}

// A writer that finds the spec or design wrong stops the phase: the conflict becomes an ADR for the human, never a workaround.
const conflictStop = (where, results) => {
  const c = results.find(r => r?.specConflict)?.specConflict
  return c ? { ok: false, stoppedAt: where, specConflict: c, next: 'run /wan-adr with this conflict as the problem; the phase resumes after the human grant' } : null
}

const writerPrompt = (w, extra) => common(`OWNED PATHS: ${WRITERS[w.writer].join(', ')} (only).`) + `${FACTS}
TASK (${w.key}, writer ${w.writer}, wave ${w.wave}): ${w.task}
Binding audit items whose ruling you deliver (see the facts): ${w.binding.join(', ')}. Tripwire: none in this phase (see
the facts) — add no consumer of TELEM_STEP or of the window and no re-plan switch.
${extra}
- You run no command: no tests, no builds, no git (a solo verifier runs the pinned gates). "evidence" is [] or only reads
  (command "read <file:lines>", exitCode 0, output what you saw).
- docs/design/ is not yours: the Gate checked the spec amendments. Every new test states covers: <id> and WOULD-FAIL-IF:.
- Tests that existed before this phase are frozen: every top-level statement of your pre-phase test files must stay
  AST-identical (pre-tests-unchanged compares them with the Gate's saved copies). Add new tests, imports and helpers as
  new statements (imports, defs, plain-name assignments only); never edit, move into a class or delete an existing one,
  never reuse a pre-phase top-level name, and add no fixture at all (use tmp_path, monkeypatch, port_base and plain
  helper functions), no pytest_* hook and no patch at import. A new def carries no decorator other than pytest.mark.<name>
  with call-free arguments (write literal lists), and no call in its defaults or annotations; a new assignment has no
  call anywhere in it (a computed value goes in a def called inside the test) and never binds a fixture name a pre-phase
  test or tests/conftest.py uses; the pre-phase statements keep their relative order (an import may go in its sorted
  place between them).
- would-fail-mutants (see "Mutants" in the facts) runs your tests against fixed regressions; write them so each mutant
  reaches the code it breaks and fails a named assertion, not an AttributeError/TypeError/KeyError.
  Add no skip, skipif, xfail or mps marker and no "awaiting implementation" text under tests/ (no-new-skip-markers pins the
  marker scan by digest, no-pending-markers refuses the phrase, suite-nodes needs every listed node PASSED).
- Write your journal to ${JOURNAL[w.writer]} (append if it exists): what you changed and why, decisions, assumptions, doubts.
If the spec or design is wrong for this task, set specConflict and stop. Return every file you created or modified.`

// One wave: its writers in parallel, then the mechanical scope audit over the wave's owned paths. Parallel writers share
// one tree, so the audit bounds the wave and each writer's own report is checked against its own paths.
const changedInPhase = new Set()
const runWave = async (wave, since, extra = '') => {
  const work = WORK.filter(w => w.wave === wave)
  const results = await parallel(work.map(w => () => agent(writerPrompt(w, extra),
    { label: `impl:${w.key}`, phase: 'Implement', schema: TASK_RESULT, agentType: ROLE.coder })))
  const audit = await audited(`implement-wave${wave}`, 'Implement', since, [...new Set(work.flatMap(w => WRITERS[w.writer]))])
  for (const f of audit.changed) changedInPhase.add(f)
  const missing = work.filter((w, i) => !results[i]).map(w => w.key)
  if (missing.length) throw new Error(`wave ${wave}: ${missing.join(', ')} produced nothing — resume this run`)
  const stop = conflictStop(`implement-wave${wave}`, results)
  if (stop) return { stop }
  work.forEach((w, i) => {
    // Writers may report absolute paths; an owned path (or the writer's own journal, which its prompt asks for) matches
    // itself or any path ending in '/<path>'.
    const owns = f => [...WRITERS[w.writer], JOURNAL[w.writer]].some(p => f === p || f.endsWith(`/${p}`))
    const outside = (results[i].files ?? []).filter(f => !owns(f))
    if (outside.length) throw new Error(`${w.key} reports files outside its owned paths: ${clean(outside.join(', '), 300)}`)
  })
  return { results }
}

phase('Implement')
const before = await scopeSnapshot('implement', 'Implement')
const wave1 = await runWave(1, before)
if (wave1.stop) return wave1.stop
const since2 = await scopeSnapshot('implement-wave2', 'Implement')
const wave2 = await runWave(2, since2, `What wave 1 reported (R1 and C1; data, re-read the files themselves):
${fence('wave1_results', wave1.results.map(r => ({ summary: r.summary, files: r.files, risks: r.risks ?? [] })), 6000)}`)
if (wave2.stop) return wave2.stop

// NEXT, Phases 4 (lenses a-e) plus the objective with its numbers. Lens d is a trust boundary (worker -> coordinator
// messages), so the security architect runs it.
const LENSES = [
  ['objective', 'Does the change achieve the objective with the exit gate\'s numbers: SEAM_RATIO >= 2 (model 4), HOP_RATE_RATIO >= 2.5 (model 4, mesh path), HOP_RATE_RATIO_MULTI >= 2.5 (model 4, multi-stream path), TELEM_US_MAX <= 1000 (model <= 300), TELEM_FRAC <= 0.01 (model <= 0.003), TELEM_INTAKE_US_MAX <= 2000 (model ~50), QUIET_S > 30 (delay 45 s, first_step_timeout_s 30 s) with 0 restarts? Do both integration telemetry tests check "ops complete" (F/B/W n from the spec, OPT n == 1, med_ms > 0, send frames bounded by the slot\'s telemetry file) and print OPS_CHECKED= (>= 7 engine, >= 3 intake)? Does each of the 11 new test nodes exist under the exact name the pinned gates run, print the exact KEY= line the gate parses, and end in exactly "<n> passed"? Is every BA-1..BA-21 ruling delivered as written?'],
  ['liveness', '(a) Liveness soundness: any path that still stamps metric, replayed or out-of-range progress (grep every caller of the progress stamp); shutdown-mode entry when one slot lags (must not enter); S3 and heal during shutdown (still apply); first-step timeout after a resume (start_step); the final-wave relaunch still exact (RUNTIME-I3); final_step=None keeps today\'s behavior exactly.'],
  ['byte-exact', '(b) Byte-exact (the product rule; one differing bit is a blocker): walk BA-1..BA-9. Can a seam sleep, the telemetry lock, the event swap or the summary touch a tensor, an RNG draw (torch, numpy, random), a fold order or a device sync? Is WT_STEP still last per step, after the wave is written? Do both integration tests compare manifests to the reference with ==?'],
  ['tests', '(c) Tests that fail on a wrong implementation: for a judge that only raises the floor, a summary using mean or sum, an encode-timed hop, a coordinator trusting message rank/peer (TELEMETRY destination, and a TELEM_STEP window indexed by the summary\'s rank/peer: the wrong-identity case must name the other existing slot, rank 0 peer 0 from (1,1)), a summarize_step that maps an event name wrongly (OPT/B/W n 0 or med_ms 0) or drops sends completing after their step\'s summary (ops complete), a stall clock fed by src="metric", a lost or duplicated event at the swap, a _launch that leaves start_step at 0 or sets it to steps or final_step + 1 on a mid-run relaunch (the relaunch part of test_resumed_attempt_is_judged_by_the_first_step_timeout, from wave 1 of 4), a shutdown mode entered when any slot (not every slot) finished, an encoder that still emits send on the multi-stream path — name the assertion that fails for each, or file it. No pre-phase test statement was edited (only additions), and no new top-level statement rebinds a pre-phase name (a shadowing def), declares any fixture (a fixture named like one a pre-phase test requests, by function name or name=, would change how that test runs), carries a decorator other than pytest.mark.<name> with call-free arguments, has a call in its defaults or annotations, defines a pytest_* hook or patches at import. tests/integration/test_coordinator.py imports at module level only names that exist in the pre-phase coordinator.py and liveness.py, and the slow-shutdown test uses no name this phase adds (causal-slow-shutdown runs that module against the pre-phase files; an ImportError there is CAUSE=other). The slow-shutdown recipe is first_step_timeout_s=30 with a 45 s exit delay (E35), so arm A stalls whether or not the pre-phase median exists. Every test states covers: <id> and WOULD-FAIL-IF:; expected values are distinct; no skip/xfail/mps marker or "awaiting implementation" text was added. No new top-level assignment contains a call, and the pre-phase statements keep their order (pre-tests-unchanged). For each would-fail-mutants mutant (lock, launch-zero, launch-late, identity, identity-telem: see "Mutants" in the facts), does the named test reach the mutated code and fail an assertion, not an AttributeError/TypeError/KeyError?'],
  ['boundary', '(d) Boundary and cost: identity (rank, peer, attempt) only from the connection\'s slot; size, key, range and finiteness checks of TELEM_STEP before any use; TELEMETRY destination never built from message fields (path traversal, symlinks, stale attempt); bodies unlinked on drop; the telem_steps.jsonl cap; a malformed message never ends the reader; WANTRAIN_SIM_* seams inert when unset and validated when set. State each mechanism\'s cost against the step time, MEASURED by a named gate or HYPOTHESIS.'],
  ['fidelity', '(e) Telemetry fidelity: step attribution of asynchronous sends (completed since the previous summary; none lost: the "ops complete" frame bounds), every op counted under the stage\'s real event names (fwd, bwd_b, bwd_w, step_opt), busy_ms as the union of intervals not their sum, no lost or duplicated event at the swap, emit_us covering the whole block (take, file append, summarize, print — VER-5), step_ms spanning _run_step start to the end of _optimizer_step, one WT_TELEM per slot per optimizer step and none on a resumed attempt\'s replay.'],
]
const reviewerFor = name => (name === 'boundary' ? ROLE.security : ROLE.reviewer)
// Granted after the phase grant (the review stopped on a spec conflict): only the review and fix prompts carry it, so a
// resumed run reuses the cached waves.
const AMENDED = `
AMENDED AFTER THE PHASE GRANT by docs/design/adr/ADR-0003-m2-p1-progress-monotone.md (human-granted; where it differs from
the facts above, it wins): (a) C1 / CONTROL-I17: SlotState.add_progress records a step only if it is STRICTLY GREATER than
the last recorded step (a src="step" stream 2, 3, 2, 3, 2 records 2 and 3 once, and a later quiet past the limit reaches
S4); the C1 unit test test_replayed_metric_progress_never_feeds_the_stall_clock covers that stream. (b) CKPT and RESULT in
coordinator._dispatch taking rank/peer from the message, and CKPT skipping the attempt check, are DEFERRED to a later phase
by that ADR (finding M2-P1-CKPT-IDENTITY, THREAT_MODEL T7): not a defect of this phase and not a spec conflict; do not file
or fix it here.
AMENDED AFTER THE PHASE GRANT by docs/design/adr/ADR-0004-m2-p1-mesh-auth-deferred.md (human-granted): the wide-stage
mesh path in stage.py _connect_mesh accepting a HELLO and trusting its (st, pe) without authenticate/trusted_peers predates
this phase and is DEFERRED to a later phase (finding M2-P1-MESH-AUTH, THREAT_MODEL T3): not a defect of this phase and not a
spec conflict; do not file or fix it here.`
const reviewPrompt = (name, lens, round, what, files) => common('READ-ONLY.') + `${FACTS}${AMENDED}${name === 'boundary' ? SECURITY_COST : ''}
LENS (${name}, round ${round}): ${lens}
${what}: ${fence('files', files, 4000)}`

// Every fix round is bracketed by its own scope audit; a spec conflict stops the phase at once (M2-R5-SAFE-3).
const fixRound = (stage, phaseTitle, changed, fixes) => async (findings, round) => {
  fixes.rounds++
  const since = await scopeSnapshot(`${stage}-fix-r${round}`, phaseTitle)
  const r = await agent(common(`OWNED PATHS: ${OWNED.join(', ')} (only).`) + `${FACTS}${AMENDED}
Fix these validated findings at their root cause (verify each against the code first). You run no command. Append what you
changed and why to ${JOURNAL.runtime} (runtime files) or ${JOURNAL.control} (control files). A finding whose fix needs a
path outside yours or a spec change is a specConflict: set it and stop.
${fence('findings', findings, 12000)}`,
    { label: `${stage}-fix:r${round}`, phase: phaseTitle, schema: TASK_RESULT, agentType: ROLE.coder })
  const audit = await audited(`${stage}-fix-r${round}`, phaseTitle, since, OWNED)
  for (const f of audit.changed) changed.add(f)
  const stop = conflictStop(stage, [r])
  if (stop) throw Object.assign(new Error(`${stage}: spec conflict`), { stop })
  return r
}
// Reviewers see the files the fixes changed too: the fix rounds add to the same set.
// A review that does not converge hands the human the last round's serious findings, not just "no".
const reviewLoop = async (stage, phaseTitle, what, files) => {
  const fixes = { rounds: 0 }
  const serious = new Map()
  try {
    const r = await reviewUntilClean({
      label: stage,
      lenses: LENSES,
      review: (name, lens, round) => agent(reviewPrompt(name, lens, round, what, [...files]),
        { label: `${stage}:${name}:r${round}`, phase: phaseTitle, schema: FINDINGS, agentType: reviewerFor(name) }).then(x => {
        const s = (x?.findings ?? []).filter(f => f?.severity === 'blocker' || f?.severity === 'major')
        if (s.length) serious.set(round, [...(serious.get(round) ?? []), ...s.map(f => ({ lens: name, severity: f.severity, location: clean(f.location, 200), problem: clean(f.problem, 600) }))])
        return x
      }),
      fix: fixRound(stage, phaseTitle, files, fixes),
    })
    return { ...r, fixRounds: fixes.rounds, ...(r.converged ? {} : { open: (serious.get(r.rounds) ?? []).slice(0, 30) }) }
  } catch (e) {
    if (e?.stop) return { converged: false, fixRounds: fixes.rounds, stop: e.stop }
    throw e
  }
}

phase('Review')
// Every owned path, not only this run's changes: a rerun keeps the Gate's saves and may change little, while an earlier
// run's stabilize or fix edits may never have been reviewed. The fix rounds add to the same set.
const reviewSet = new Set([...OWNED, ...changedInPhase])
const review = await reviewLoop('review', 'Review', 'Every owned path of this phase, reviewed whole (an earlier run of this phase may have changed any of them, unreviewed; a path that does not exist yet is a finding only if the facts require it)', reviewSet)
if (review.stop) return review.stop
// Never verify or stabilize on top of blocker/major findings nobody resolved: the human decides.
if (!review.converged) {
  return { ok: false, stoppedAt: 'review', review, open: review.open ?? [], next: `the review did not converge in ${review.rounds} rounds; its open findings are in "open". A human decides: grant with those findings recorded, one more review (a rerun of this phase reviews every owned path again), split the phase, or stop` }
}

// The script decides: VERIFIED only when the JSON pinned.mjs printed says so with nothing failing, for this gate id, with
// every entry of the GATES file present and passing, on the tree audited before and after the run. The verifier runs a
// shell, so the run is bracketed by a scope audit that allows no tracked change: an edit it made would reach no review.
const GATE_IDS = [
  ...['factory', 'build', 'syntax', 'fmt', 'lint', 'types', 'discipline', 'invariants', 'claims', 'unit', 'integration', 'multiworker', 'oracle', 'identity', 'chaos', 'wan', 'full', 'static', 'security', 'fuzz'].map(r => `rung:${r}`),
  'consecutive',
  ...['spec-amended', 'no-pending-markers', 'no-new-skip-markers', 'pre-tests-unchanged', 'liveness-unit', 'intake-unit', 'telemetry-unit', 'would-fail-mutants', 'slow-shutdown-cpu', 'final-wave-relaunch', 'causal-slow-shutdown', 'telemetry-engine', 'telemetry-intake', 'suite-nodes'].map(p => `pinned:${p}`),
]
const verifyOnce = async round => {
  const since = await scopeSnapshot(`verify-r${round}`, 'Verify')
  const v = await agent(common('You run ALONE — you MAY build and run anything (no git). Fix nothing.') + `${FACTS}
Round ${round}.${round ? ' Fixes were applied since the last verify: re-establish ground truth yourself.' : ''}
1. Run exactly: ${PINNED}
   Put the JSON it prints into "pinned" EXACTLY as printed. Do not edit it, summarize it or rerun a gate for a better result.
2. For every failing gate, one INDEPENDENT item: id, the gate id as "rung", the symptom, exact evidence (error text, failing
   test ids, the repro command), the suspected cause, its root-cause "mechanism" (the same words for failures that share a
   cause), and the minimal scope inside ${OWNED.join(', ')}. Merge failures that share a root-cause file.
Edit no file: the run is scope-audited and any tracked change stops the phase.`,
    { label: `verify:r${round}`, phase: 'Verify', schema: PINNED_RESULT, agentType: ROLE.verifier })
  const audit = await audited(`verify-r${round}`, 'Verify', since, [])
  const out = v ? { ...v, since, tree: audit.tree } : null
  history.push({ round, tree: audit.tree, verified: verified(out), failing: failingIds(out) })
  return out
}
// Every verify, kept for the verdict (rule 8: a flaky pass is not a pass): a VERIFIED on a tree that already produced a
// failing verify is intermittent, never a pass.
const history = []
const failingIds = v => {
  const p = v?.pinned
  const ids = [...(Array.isArray(p?.failing) ? p.failing : []), ...(Array.isArray(p?.gates) ? p.gates.filter(g => g?.ok !== true).map(g => g?.id) : [])]
  return [...new Set(ids.map(x => clean(String(x), 80)))].slice(0, 40)
}
const verified = v => {
  const p = v?.pinned
  if (p?.verdict !== 'VERIFIED' || p.milestone !== SLUG || !Array.isArray(p.failing) || p.failing.length) return false
  if (!TREE_ID_RE.test(p.tree ?? '') || p.tree !== v.since || p.tree !== v.tree) return false
  const gates = Array.isArray(p.gates) ? p.gates : []
  return gates.every(g => g?.ok === true) && GATE_IDS.every(id => gates.some(g => g?.id === id))
}

// Every stabilize item, not only those under tests/, is proposed, voted and applied with the phase's facts: the generic
// runFixPipeline carries none, and a src/ fix voted without the C1/C2/R1 contract, the BA rulings or the causal
// classifier's messages can turn a gate green by breaking them. M2-R5-SAFE-2: an item that edits anything under tests/ is
// applied only on 3 of 3 approvals; any other on a majority of all three voters, with the spec voter's veto (as in
// runFixPipeline).
const touchesTests = it => it.scope.some(p => p.startsWith('tests/'))
const CONTRACT_KEPT = `The fix keeps the contract in the facts above: the C1, R1 and C2 mechanism, every BA ruling, the tripwire
(none: no consumer of TELEM_STEP or of the window), the assertion order and messages of the slow-shutdown test and the
coordinator's "fleet stalled: no optimizer step fleet-wide for " line (causal-slow-shutdown classifies by them), the rules
for pre-phase tests, the five would-fail-mutants (a test must still fail under each), "ops complete" in both integration telemetry tests and stage_main.py untouched. A change
that breaks any of them is wrong even if it turns the gate green.`
const stabilizeFix = async it => {
  const unanimous = touchesTests(it)
  const rule = unanimous
    ? 'It edits tests/, so it is applied only on 3 of 3 approvals: fix a test only where the test is wrong, never loosen it.'
    : 'It is applied only on a majority of the three voters, and a reject by the spec voter is a veto.'
  const prop = await agent(common('READ-ONLY: propose a fix, do not apply it.') + ownerContext(it) + `${FACTS}
Fix item ${it.id} (source: a failing gate of ${GATES}; rung ${it.rung}). ${rule}
${CONTRACT_KEPT}
Symptom: ${fence('symptom', it.symptom, 2000)}
Evidence: ${fence('failure_evidence', it.evidence, 6000)}
Suspected cause: ${fence('suspected_cause', it.suspectedCause ?? '', 1000)}
Allowed scope: ${JSON.stringify(it.scope)}
${(it.priorRejections ?? []).length ? `Earlier proposals for this scope were REJECTED — satisfy their objections:\n${fence('prior_rejections', it.priorRejections, 6000)}` : ''}
Propose the minimal correct fix as exact old->new snippets strictly inside the scope. If the true cause is outside the
scope, set rootCauseOutsideScope and propose nothing else.`,
    { label: `propose:${it.id}`, phase: 'Stabilize', schema: PROPOSAL, agentType: ROLE.proposer })
  if (!prop) return { it, outcome: 'no-proposal' }
  if (prop.rootCauseOutsideScope) return { it, prop, outcome: 'outside-scope' }
  const raw = await parallel(FIX_VOTERS.map(([name, lens]) => () => agent(common('READ-ONLY adversarial voter.') + `${FACTS}
Vote on a PROPOSED (not applied) fix for item ${it.id} (rung ${it.rung}); ${unanimous ? 'it edits tests/ and needs all three approvals' : 'it needs a majority, and a reject by the spec voter is a veto'}.
Assume it is wrong until the code proves otherwise. LENS ${name}: ${lens}
${CONTRACT_KEPT} Reject such a change.
Evidence: ${fence('failure_evidence', it.evidence, 4000)}
Proposal: ${fence('proposal_from_another_agent', prop, 8000)}
Allowed scope: ${JSON.stringify(it.scope)}
Read the actual files. approve=true only if you would ship it; an amendment must be an exact snippet.`,
    { label: `vote:${it.id}:${name}`, phase: 'Stabilize', schema: VOTE, agentType: ROLE.reviewer })))
  const votes = raw.map((x, i) => (x ? { ...x, lens: FIX_VOTERS[i][0] } : null)).filter(Boolean)
  const tally = tallyVotes(raw, FIX_VOTERS.length)
  if (unanimous ? !tally.unanimous : !tally.majority) return { it, prop, votes, tally, outcome: 'rejected' }
  // fail closed: a missing spec vote is not an approval
  if (!unanimous && votes.find(x => x.lens === 'spec')?.approve !== true) return { it, prop, votes, tally, outcome: 'vetoed' }
  const applied = await agent(common(`You APPLY one reviewed fix. OWNED PATHS: ${it.scope.join(', ')} (only).`) + ownerContext(it) + `${FACTS}
Before writing, verify each target is a regular file (or a new file) inside the repo — not a symlink. Apply the proposal
with the approving voters' amendments (where they conflict, follow the majority). ${CONTRACT_KEPT}
If the spec itself is wrong, set specConflict and stop. Do not run builds; you may run only the tests that cover your
change and must report them as evidence.${it.ownerJournal ? ` Append what you changed and why to ${clean(it.ownerJournal, 200)}.` : ''}
Proposal: ${fence('proposal_from_another_agent', prop, 8000)}
Votes (${tally.approvals}/${tally.expected} approve): ${fence('votes', votes.map(x => ({ approve: x.approve, reasons: x.reasons, amendment: x.amendment })), 8000)}`,
    { label: `apply:${it.id}`, phase: 'Stabilize', schema: TASK_RESULT, agentType: ROLE.coder })
  if (applied?.specConflict) return { it, prop, votes, tally, outcome: 'spec-conflict', applied }
  return { it, prop, votes, tally, outcome: applied ? 'applied' : 'apply-failed', applied }
}

phase('Verify')
let check = await verifyOnce(0)
const rejected = []
const changedInStabilize = new Set()
let round = 0
while (check && !verified(check) && round < MAX_STABILIZE) {
  round++
  phase('Stabilize')
  const normalized = normalizeItems(check.items, { knownRungs: RUNG_IDS })
  const dominant = dominantMechanism(normalized.items)
  if (dominant.focus) log(`Stabilize ${round}: one mechanism explains ${dominant.explained}/${dominant.of} failures — fixing only it this round`)
  const items = dominant.items
  const scoped = items.filter(it => it.scope.every(inOwned))
  for (const d of [...normalized.dropped, ...items.filter(it => !scoped.includes(it))]) log(`Stabilize ${round}: item ${d.id} dropped (${d.why ?? 'outside this phase\'s owned paths'})`)
  if (!scoped.length) { log(`Stabilize ${round}: the gate is red and no item is inside this phase's owned paths — a human decides`); break }
  // One item at a time (no two appliers run tests together), each bracketed by a scope audit on its own scope.
  const outcomes = []
  const roundStart = check.tree
  let roundTree = roundStart
  // the owner journal comes from this phase's writers, never from the verifier's text
  for (const it of withPriorRejections(scoped, rejected).map(x => ({ ...x, ownerTask: undefined, ownerJournal: undefined, ...ownerOf(x.scope) }))) {
    const since = await scopeSnapshot(`stabilize-r${round}-${it.id}`, 'Stabilize')
    const outcome = await stabilizeFix(it)
    const audit = await audited(`stabilize-r${round}-${it.id}`, 'Stabilize', since, it.scope)
    // M2-R5-SAFE-1: whatever the applier's outcome, a changed file means the round stabilized and gets reviewed
    for (const f of audit.changed) changedInStabilize.add(f)
    roundTree = audit.tree
    outcomes.push(outcome)
    const stop = conflictStop('stabilize', [outcome.applied])
    if (stop) return stop
  }
  rejected.push(...rejectionsOf(outcomes, round))
  // Rule 8: a round that left the tree as the failing verify saw it is not re-verified — a rerun could only turn a flaky
  // failure into a pass.
  if (roundTree === roundStart) { log(`Stabilize ${round}: no file changed (${outcomes.map(o => o.outcome).join(', ')}) — not re-verifying the same tree; a human decides`); break }
  phase('Verify')
  check = await verifyOnce(round)
}
await audited('phase', 'Verify', before, OWNED)
// Code changed by Stabilize is reviewed with all six lenses (tests and liveness included, M2-R5-SAFE-2), whether or not the
// last verify passed (M2-R5-SAFE-1): a NOT VERIFIED run's edits stay in the tree a rerun starts from. If that review fixes
// anything, the pinned gate runs once more on the final tree.
let finalReview = { converged: true, rounds: 0 }
if (changedInStabilize.size) {
  finalReview = await reviewLoop('stabilize-review', 'Stabilize', 'These files changed during Stabilize, after the first review', changedInStabilize)
  if (finalReview.stop) return finalReview.stop
  if (finalReview.fixRounds) {
    phase('Verify')
    check = await verifyOnce(round + 1)
  }
}
// The verdict stands only for the tree it ran on: nothing may have changed since that verify.
const finalTree = verified(check) ? await scopeSnapshot('final', 'Verify') : null
const earlierFailing = history.filter(h => !h.verified).map(({ round: r, tree, failing }) => ({ round: r, tree, failing }))
const intermittent = verified(check) ? earlierFailing.filter(h => h.tree === check.tree) : []
if (intermittent.length) log(`${meta.name}: VERIFIED on tree ${check.tree}, which already failed ${intermittent.map(h => `in round ${h.round} (${h.failing.join(', ') || 'not VERIFIED'})`).join('; ')} — intermittent, NOT VERIFIED`)
const ok = verified(check) && !intermittent.length && finalTree === check.tree && review.converged && finalReview.converged
log(`${meta.name}: ${ok ? 'VERIFIED' : 'NOT VERIFIED'} after ${round} stabilize round(s)`)
const failing = failingIds(check)
const next = ok
  ? 'rerun wan-next to write phase 2 (re-plan behind replan = False) from the new state'
  : intermittent.length
    ? `NOT VERIFIED: run /wan-bughunt on the intermittent gate(s) ${[...new Set(intermittent.flatMap(h => h.failing))].join(', ') || '(see earlierFailing)'}: they failed and passed on the same tree`
    : !check
      ? 'NOT VERIFIED: the verifier returned nothing — rerun this phase'
      : !verified(check)
        ? `NOT VERIFIED: failing gate(s) ${failing.join(', ') || `none listed — the relayed JSON is not a VERIFIED for ${SLUG} on the audited tree (see pinned)`} after ${round} stabilize round(s); a human decides (/wan-bughunt for a failure that survived them, or a rerun of this phase)`
        : !finalReview.converged
          ? 'NOT VERIFIED: the post-stabilize review did not converge; its open findings are in finalReview.open. A human decides: grant with those findings recorded, one more review, or stop'
          : 'NOT VERIFIED: the tree changed after the verify — rerun this phase'
// Stabilize edits the post-stabilize review did not clear: a rerun reviews every owned path, these included.
const unreviewed = finalReview.converged ? [] : [...changedInStabilize]
return { ok, review, finalReview, unreviewed, failing, pinned: check?.pinned ?? null, tree: check?.tree ?? null, finalTree, stabilizeRounds: round, rejected, earlierFailing, intermittent, next }
