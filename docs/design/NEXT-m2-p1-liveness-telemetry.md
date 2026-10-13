# NEXT: m2-p1-liveness-telemetry

Phase workflow: `wan-x-m2-p1-liveness-telemetry`. Pinned verify: `docs/design/GATES-m2-p1-liveness-telemetry.json`.
Milestone M2 (adaptivity), phase 1 of 3. Hardware: `{"gpu": false}`. Everything below runs on this machine's CPU;
every performance number is SIMULATED and real-hardware validation stays pending.

Split (LESSONS.md:10-13, dependency order): **phase 1 (this)** sound liveness accounting for restart in place (C1) and
per-step F/B/W plus per-hop telemetry delivered to the coordinator (R1, C2), with no consumer; **phase 2** re-solve from
telemetry, resume under a new ownership, the re-plan loop behind `replan = False` (SCHEDULER-I15); **phase 3** plan mode
for wide stages (M2-R5-FIT-2, FIT-5) and SCHEDULER-I13. This phase has 3 work packages and 14 pinned entries (one above
the lesson's ~12: `would-fail-mutants`, see "Exit gate", is the mechanical E13 check and is not traded away).

# Objective

Make the coordinator's liveness judge sound across restarts and shutdown (replayed loss lines never feed the stall
clock; quiet after every slot finished its last optimizer step is a bounded shutdown, not a stall), and make every slot
stream one validated per-step summary of measured F/B/W/optimizer times and per-hop wire throughput to the
coordinator, with the training state still equal to the reference by sha256 and nothing consuming the telemetry yet.

# Evidence

Facts only, each with its source. "Read" means read for this document on the current tree.

| # | Fact | Source |
|---|---|---|
| E1 | The ladder is all green; `missingRequired` is empty. `full` and `mutation` are on demand; `frozen`, `component`, `multinode`, `matrix`, `rest`, `bench`, `redteam` are N/A. | state snapshot, `ladder` |
| E2 | `invariants: 43/52 covered`; SCHEDULER-I13 and SCHEDULER-I15 are DEFERRED to M2. | state snapshot, `invariants` |
| E3 | The hardware switch is `"gpu": false`. | agents-architetura/tools/factory/hardware.json:2 (read) |
| E4 | `gpu_allowed()` is True only for a literal `"gpu": true`; `mps_available()`/`cuda_available()` return False otherwise. The MPS test is `@pytest.mark.mps` + `skipif(not mps_available(), reason=... GPU_OFF)`. | tests/conftest.py:59-86; tests/integration/test_coordinator.py:181-185 (read) |
| E5 | findings.json holds 15 findings, all open, all from "wan-next m2-adaptive-ownership review round 5 (2026-10-08)". | agents-architetura/docs/findings.json:1-139 |
| E6 | M2-MPS-INTERMITTENT (major): the CPU+MPS fleet test stalls intermittently ("no optimizer step fleet-wide for 8s (limit 8s) while every worker is alive"), restart budget spent; N/A under gpu false; "may share a cause with the restart-in-place liveness accounting (M2 C1)". | findings.json:3-11 |
| E7 | The prior snapshot's failing log: "restarting in place from step 7" twice, then "the restart budget (2) is spent", recipe `steps=8`. | .factory/prior/NEXT-m2-adaptive-ownership.md:21-22 |
| E8 | M2-R5-FIT-1 (major): the MPS test failed in `integration` and passed in `full` in one ladder run; recorded fix: resolved for gates by `gpu_allowed`. | findings.json:13-20 |
| E9 | M2-R5-FIT-3 (minor): "Send events do not record the sending host (stage.py:369-371), and there is no step_opt event." | findings.json:31-38 |
| E10 | M2-R5-SAFE-1/2/3/4 (stabilize writes skip review; stabilize review lacks the tests lens and passes 2/3; specConflict ignored outside Implement; marker scan misses spellings and `mark.mps`). | findings.json:58-92 |
| E11 | M2-R5-VER-1 (major): suite gates check only pytest's last line; no frozen node list. VER-3: grep exit 2 passes. VER-4: a causal timeout below its inner worst case. VER-5: telemetry cost measured on the summary only. | findings.json:94-137 |
| E12 | M2-R5-FIT-2 (plan-mode grouping label, coordinator.py:723) and FIT-5 (plan-mode fixture) concern plan mode. | findings.json:22-29, 49-56 |
| E13 | LESSONS: keep a phase to ~3 work packages and ~12 gate entries, split along the dependency order; use distinct expected values and check against a deliberately wrong implementation. | agents-architetura/docs/LESSONS.md:7-13 |
| E14 | S4: "no optimizer step fleet-wide in max(floor, 6 x median step)"; the limit is `max(stall_floor_s, stall_steps * step_s)`, `first_step_timeout_s` while `median_step_s` is None. Nothing in the judge knows the final step. | src/wantrain/fault_tolerance/liveness.py:14-15, 217-228, 259-265 (read) |
| E15 | `median_step_s` takes gaps between all slots' `(step, t)` points with `s1 > s0 and t1 > t0`, needs ≥ 3 gaps, median of the last 12. | liveness.py:73-82 (read) |
| E16 | Progress is stamped with the coordinator's arrival time, deduplicated only against the previous step. | src/wantrain/coordinator/coordinator.py:385-388 (read) |
| E17 | The agent turns both `WT_STEP` and `WT_METRIC` lines into the same `PROGRESS` message (no source field; `WT_STEP` gives `"loss": None`). | src/wantrain/worker/agent.py:463-491 (read) |
| E18 | On resume, `run_stage` keeps the loss stream and `stream_losses` tails it from offset 0, re-printing one `WT_METRIC` per past step. | src/wantrain/worker/stage_main.py:29-55, 63-65 (read) |
| E19 | Every slot prints `WT_STEP step` after its optimizer step and checkpoint; after the loop the final checkpoint is unconditional, then `_finish` (loss gather at final peer 0, sender drain and reader joins bounded by `SHUTDOWN_GRACE_S = 120`). | src/wantrain/runtime/stage.py:66, 959-990, 1095-1160 (read) |
| E20 | After the stage process exits, the agent ships the final wave over the control channel, then sends `EXIT`; heartbeats report `stage_alive` until the exit is reported. | agent.py:280-296, 493-518, 551-592 (read) |
| E21 | `default_ckpt_every = 1`, `stall_floor_s = 90.0`, `max_restarts = 2`; the test helper sets `stall_floor_s = 8.0`. | coordinator.py:72, 77, 90; tests/helpers/cluster.py:24-33 (read) |
| E22 | `_launch` creates fresh `SlotState`s per attempt; `_slot_for` drops messages of another attempt. `_restart_in_place` relaunches from `common_step` whatever it is. | coordinator.py:363-372, 914-931, 1195-1205 (read) |
| E23 | `_conn_loop` catches only `(ChannelClosed, OSError, ValueError)`; the `TELEMETRY` branch indexes `msg['rank']`, `msg['peer']`, `msg['attempt']` and calls `os.replace(body, dst)` with `dst` built from those message fields. | coordinator.py:297-310, 352-356 (read) |
| E24 | `step_opt` events exist (local and wide). `send` events carry no hop key. In the multi-stream path the `send` event is emitted by the encoder after enqueueing fragments, so it times encode + enqueue, not the wire; fragment senders emit no event. | stage.py:358-371, 373-396, 548-578, 894, 955 (read) |
| E25 | `self.telemetry` is appended by sender threads and the main thread through `_ev`; `_flush_telemetry` iterates it, then `clear()`s it, without a lock. | stage.py:133, 595-607, 681-683 (read) |
| E26 | Senders are created per connection without the connection key: `_wire_shared`, `_wire`, `_wire_multi`. Paced hops use `LinkSpec(**links[r])`; intra hops use `LinkSpec()` (unpaced). `SenderPacer.pace` sleeps `nbytes*8/(bw_mbps*1e6)`. | stage.py:175-187, 271-291, 445-455, 472-516; src/wantrain/transport/shim.py:27-33 (read) |
| E27 | No code reads telemetry event names except the agent, which ships the per-slot telemetry file at exit. | grep over src/ tests/ (read) |
| E28 | `patch_spec_for_resume` lives at fault_tolerance/checkpoint.py:143-153 (not spec.py:52-56, M2-R5-FIT-4). | src/wantrain/fault_tolerance/checkpoint.py:143-153 (read) |
| E29 | `invariants.mjs` fails on a test claiming an id no spec defines. | agents-architetura/tools/gates/invariants.mjs:3, 47-48 (read) |
| E30 | SPEC-control §2.2 step 5 states S4 without any final-step exception; §2.1 monitor inputs and §2.4 agent name `PROGRESS (from WT_STEP)` and `TELEMETRY`. SPEC-runtime task 1 (M2) asks for per-op F/B/W and per-hop transfer times "in TELEMETRY (median per step)". | docs/design/SPEC-control.md:41-44, 62-68, 91; docs/design/SPEC-runtime.md:113-115 (read) |
| E31 | APPROVALS.json grants `docs/design/DESIGN.md` at sha256 `053fbb2f…caf4`, which equals the current file's sha256. FROZEN.json is absent. | docs/design/APPROVALS.json:4-7; `shasum -a 256 docs/design/DESIGN.md` (run) |
| E32 | Pre-phase sha256: liveness.py `4132edab…467a6`, coordinator.py `a02f7687…ffe226e` (equal to the prior draft's pins); the `causal.mjs save` manifest for exactly these two files in this order: `ac243d33…952f07`. | `shasum -a 256` (run); node JSON.stringify digest (run) |
| E33 | Pre-phase test functions: 153 (no test classes). Canonical list sha256 `21e0c82f276af6b090806045212de1087e2b18def459eedf4eb84aa566a20f38` (command in "Exit gate"). Pre-phase skip/marker scan (widened, "Exit gate"): 3 lines, normalized sha256 `9841c3222d276dc6eb87ad45fe0b418fd6d7d580a6f3299e119ca8b2117dbb6d`. No test says "awaiting implementation". | grep runs (run) |
| E34 | Pre-phase sha256 of the four owned test files: test_fault_tolerance.py `70dab584…685030`, test_coordinator.py `2d42e69c…1382b1`, test_runtime_spec.py `af63f4c7…d7a8a3`, test_engine_bitwise.py `0b1a8873…3da99a` (full digests in the Gate step and `pre-tests-unchanged`). Existing tests build `SlotState`/`LivenessJudge` directly and assign `progress` lists; none sends a `PROGRESS` message, so C1 needs no edit of an existing test. | `shasum -a 256` (run); tests/unit/test_fault_tolerance.py:21-52, 122 (read) |
| E35 | With `steps = 4` the pre-phase judge can reach the end of a run with `median_step_s` None: 3 step transitions give at most 3 gap candidates after sorting by `(step, t)`, ≥ 3 are needed, and a `WT_METRIC s` that arrives after `WT_STEP s+1` (the loss tail polls every 0.5 s; dedup is only against the previous step) is appended as `(s, t_late)` and voids one, so the S4 limit is `first_step_timeout_s` (90 s in `FAST_FT`), not the 8 s floor; a 12 s delay would then never trip S4. Arithmetic for the slow-shutdown recipe (`first_step_timeout_s = 30`, exit delay 45 s): the pre-phase limit is 30 s when the median is None and max(8, 6 × median) otherwise, so ≤ 30 s while the median step is ≤ 5 s (HYPOTHESIS for this CPU toy step, 0.1–1 s per the cost model); the quiet after the last `WT_STEP` is ≥ 45 s (`EXIT` follows the delay), so S4 trips either way, and a relaunched pre-phase attempt has the same ≤ 30 s limit and the same 45 s quiet until the budget is spent. No pre-phase run of this recipe is recorded: arm A of `causal-slow-shutdown` is that run. | liveness.py:73-82, 222-228; tests/helpers/cluster.py:24-33; coordinator.py:385-388; stage_main.py:29 (read) |

# Why now

Precedence: no red rung (E1). Open majors next (E5): the ones this phase can act on are M2-MPS-INTERMITTENT (its
CPU-reproducible mechanism, below), VER-1 (suite gates), SAFE-1/2 (the phase workflow's own stabilize loop). C1 is also
the mechanism every later M2 phase relies on: a re-plan is applied through restart in place (SPEC-control task 2), and
the re-solve needs measured costs (SPEC-runtime task 1). Lesson 2 says split first along that dependency.

**The mechanism under M2-MPS-INTERMITTENT, derived from the code (facts E14-E22, conclusion HYPOTHESIS):**
1. After `WT_STEP steps−1`, every slot still owes: the final `torch.save` (E19), `_finish` (loss gather, drain ≤ 120 s,
   E19), and the agent's final-wave ship before `EXIT` (E20). S4 sees no progress during all of it (E14).
2. When that quiet exceeds `max(stall_floor_s, 6 × median)`, the verdict is `stalled` and `_restart_in_place` relaunches
   from wave `steps−1` (E22, E7 "from step 7" with `steps=8`).
3. The relaunched attempt runs no step, but `stream_losses` replays every past `WT_METRIC` in one burst (E18), each
   stamped at arrival (E16, E17): `median_step_s` becomes ~0 and the limit collapses to the floor (E15, E14). The same
   shutdown quiet repeats, the budget is spent (E7).
4. Mid-run the same replay defeats `first_step_timeout_s` after any resume: progress is non-empty, so a resumed attempt
   whose connect + load + first step exceeds the floor is restarted again.

Cost model of the quiet (the term C1 changes), `T_quiet = T_save + T_finish + S_shard / B_ctrl`. Arithmetic, not a
measurement: a 7B model in fp32 with Adam is ~16 B/param; over 4 stages a shard is ~28 GB; at B_ctrl = 1 Gb/s the ship
alone is ~224 s, already above the default 90 s floor. So S4 misfires at the end of a real-size run, on any hardware.
On this CPU the toy model's quiet is ~seconds, which is why only the slower MPS pair showed it. HYPOTHESIS, falsified by
`causal-slow-shutdown`: an injected post-exit delay of 45 s, above every limit the pre-phase S4 can use in that recipe
(E35: floor 8 s, `first_step_timeout_s` 30 s, 6 × median while the median step is ≤ 5 s), makes the pre-phase
coordinator fail for the stall and the phase's coordinator pass.

**Alternatives not picked:**
- `/wan-bughunt` on the MPS test: needs the Apple GPU; E3 makes it N/A. Its CPU mechanism is C1.
- SCHEDULER-I15 re-plan first: needs both telemetry and a sound restart path; the prior attempt bundled them and did not
  converge (E13).
- SCHEDULER-I13 (degrade instead of refuse): deferred M2 work, but unblocks nothing else in M2; phase 3.
- Plan mode / FIT-2 / FIT-5: phase 3; independent of liveness and telemetry.
- M1 real-hardware evidence: needs GPUs (E3).

**Findings, ruled for this phase:**
| Finding | Ruling |
|---|---|
| M2-MPS-INTERMITTENT | C1 fixes the CPU-reproducible mechanism; the finding stays open (needs gpu true + `/wan-bughunt`). Closeout records the causal result next to it. |
| M2-R5-FIT-1 | Confirmed resolved by E4: the `suite-nodes` gate allows that skip only with the GPU_OFF reason. Closeout may close it, citing that gate's run. |
| M2-R5-FIT-3 | Half refuted: `step_opt` exists (E24). Half confirmed: no hop key; the multi-stream event times encode, not the wire. Fixed by R1. |
| M2-R5-FIT-4 | Cited correctly here (E28); this phase does not touch `patch_spec_for_resume`. |
| M2-R5-FIT-2, FIT-5 | Out of scope (plan mode, phase 3). |
| M2-R5-SAFE-1/2/3 | Requirements on this phase's workflow (see Phases, Verify). |
| M2-R5-SAFE-4 | `no-new-skip-markers` widened (spacing, `_pytest`, `getattr(pytest`, `mark.mps`, collection hooks) and pinned by digest. |
| M2-R5-VER-1 | `suite-nodes`: every pre-phase test function and every new node must appear as `PASSED`; exactly the 2 GPU_OFF skips. |
| M2-R5-VER-2 | Out of scope (re-plan arm, phase 2). |
| M2-R5-VER-3 | `no-pending-markers` passes only on grep exit 1. |
| M2-R5-VER-4 | `causal-slow-shutdown`: outer `timeoutSec` 4200 > 2 arms × 2 runs × 900 s. |
| M2-R5-VER-5 | The stage measures its whole per-step telemetry block (`emit_us`); the coordinator measures its intake (`handle_us_max`). |

Why no ADR: no training, synchronization, protocol-layer or recovery-semantics change. C1 restores the judge's stated
intent (S4 judges a fleet that still owes optimizer steps); recovery paths are unchanged. Telemetry is timing-only and
has no consumer. Reversal: if any lens finds a change to a byte of state or to when a recovery fires other than the
shutdown/replay rules below, STOP and `/wan-adr`.

# Scope

Two workstreams, because the mechanism crosses the stage process (runtime: measures) and the coordinator/agent
(control: judges and receives); either half alone has no observable effect.

| Writer | Workstream | Owned paths |
|---|---|---|
| R | runtime | `src/wantrain/runtime/stage.py`, `src/wantrain/runtime/telemetry.py` (new), `tests/unit/test_runtime_spec.py`, `tests/integration/test_engine_bitwise.py` |
| C | control | `src/wantrain/fault_tolerance/liveness.py`, `src/wantrain/coordinator/coordinator.py`, `src/wantrain/worker/agent.py`, `tests/unit/test_fault_tolerance.py`, `tests/integration/test_coordinator.py` |

Not touched: `stage_main.py` (the replay is neutralized at the coordinator boundary), `tests/helpers/` (`LocalCluster`
already inherits `os.environ`, cluster.py:92), the scheduler, the transport, `tests/oracle/`, `tests/chaos/`.

## Spec amendments (precondition; applied by the sanctioned writer before writers start)

No phase agent writes `docs/design/` and E29 forbids covering undefined ids. Route: `/wan-adr {"problem": "apply
docs/design/NEXT-m2-p1-liveness-telemetry.md §Spec amendments verbatim to SPEC-control.md and SPEC-runtime.md, with
history entries", "id": "m2-p1-spec"}`, then the human grants it. wan-adr writes these texts in its draft, before any grant, so the greps alone prove
nothing: the Gate refuses to start until exactly one `docs/design/adr/ADR-NNNN-m2-p1-spec.md` exists and a human grant
covers it and `docs/design/DESIGN.md`; `spec-amended` checks the same grant, then greps for the result.
- SPEC-control §2.2 step 5, appended: "S4 judges only while some launched slot still owes an optimizer step: it reads
  only progress parsed from `WT_STEP` whose step lies in [the attempt's start step, `steps − 1`]; replayed `WT_METRIC`
  lines never feed it. Once every launched slot exited 0 or reported step `steps − 1` (or the attempt started at
  `steps`), the attempt is in shutdown: quiet is bounded by `shutdown_limit_s` (default 600), past which the verdict is
  failed with 'shutdown exceeded … after the final step'. S1–S3 and strikes still apply in shutdown."
- SPEC-control §2.1 monitor inputs and §2.4 agent: "`PROGRESS` carries `src` (`step` from `WT_STEP`, `metric` from
  `WT_METRIC`). `TELEM_STEP`: the live per-step summary parsed from `WT_TELEM`, validated at the coordinator
  (CONTROL-I18). `TELEMETRY` stays the telemetry file shipped at exit."
- SPEC-runtime task 1: "in TELEMETRY" becomes "in one `WT_TELEM` line per step (RUNTIME-I9)".
- New ids, texts in Invariants: CONTROL-I17, CONTROL-I18, RUNTIME-I9.

## Mechanism (the contract writers implement)

**C1 — liveness accounting (control, CONTROL-I17).**
- `liveness.py`: `SHUTDOWN_LIMIT_S = 600.0`. `SlotState.start_step: int = 0`. `SlotState.add_progress(step: int, t: float,
  final_step: int | None) -> bool` appends `(step, t)` and returns True iff `start_step <= step`, `final_step is None or
  step <= final_step`, and `step` is strictly greater than the last recorded step (ADR-0003: a stream such as 2, 3, 2, 3
  records 2 and 3 once, so `progress` holds at most `steps − start_step` entries per attempt).
  `LivenessJudge(..., final_step: int | None = None, shutdown_limit_s: float = SHUTDOWN_LIMIT_S)`. Shutdown mode holds
  when `final_step is not None` and every launched slot has `exit_code == 0`, or `last_step() == final_step`, or
  `start_step > final_step`. In shutdown mode, after the done / S1 / S2 / grace / crashed checks and the S3 peer-lost
  check, the S4 block is replaced by: quiet = now − max(last progress, started_at); `quiet > shutdown_limit_s` →
  `Verdict("failed", None, "shutdown exceeded {limit:.0f}s after the final step …")`; otherwise heartbeat staleness
  still earns strikes, else `ok`. `final_step=None` keeps today's behavior exactly (existing tests unchanged).
- `coordinator.py`: `CoordinatorConfig.shutdown_limit_s: float = 600.0`. `_judge` passes `final_step = spec.steps − 1`
  of the run and `shutdown_limit_s`. `_launch` sets every new `SlotState.start_step` to `wave + 1` when relaunched from
  a wave (`_restore_and_relaunch`'s `step`), else 0. `_slot_update(progress=True)` records progress only when
  `msg.get("src") == "step"`, through `add_progress` with an integer step (a non-integer step is dropped and logged).
- `agent.py`: `PROGRESS` gains `"src": "step"` (from `WT_STEP`) or `"metric"` (from `WT_METRIC`).
- Restart from the final wave stays reachable through S2/S3/convict (a shutdown-time peer loss): it reruns `_finish`
  from wave `steps − 1` with zero optimizer steps, which RUNTIME-I3 makes exact. It is no longer reachable from S4
  quiet. Default 600 s covers `T_quiet` up to ~(600 − 120 − T_save) s × B_ctrl, e.g. ~55 GB per shard at 1 Gb/s;
  larger shards raise it (Risk R1).
- Test seam (control, timing only): `WANTRAIN_SIM_EXIT_DELAY_S` read once at agent start (float in [0, 3600], malformed
  raises `ValueError`); in `_supervise`, after `stage.wait()` returns 0 and the post-exit `_ship_checkpoint`, before `EXIT`, sleep
  that long (the coordinator holds the final wave during the quiet, as after a slow final ship). Absent means no effect.

**R1 — per-step telemetry (runtime, RUNTIME-I9).**
- `_ev` appends under `self._tel_lock`; `_take_events()` swaps the list under the lock (no lost or duplicated event).
- Every sender knows its hop: `_sender(sock, q, pacer, hop)` and `_frag_sender(sock, frag_q, pacer, hop)` with hop
  labels `next<q>`, `prev<q>`, `intra<k>` from the connection key. Chain path: the `send` event gains `hop`. Multi-stream
  path: the encoder emits `encode`; each fragment sender emits `send` with `hop`, `nbytes`, spanning `pace` +
  `send_frame`.
- `runtime/telemetry.py`, pure (no torch import, no clock, no RNG): `summarize_step(events, step, rank, peer, step_ms,
  emit_us) -> dict` returning `{"v": 1, "step", "rank", "peer", "step_ms", "emit_us", "ops": {"F"|"B"|"W"|"OPT":
  {"n": int, "med_ms": float}}, "stall_ms": float, "hops": {<hop>: {"frames": int, "bytes": int, "busy_ms": float}}}`.
  `med_ms` is the median of the op's event durations; `busy_ms` is the length of the union of the hop's send intervals
  (parallel streams overlap). F/B/W/OPT and stall events are those of this step; send events are those completed since
  the previous summary (senders run asynchronously). Two pure module-level helpers carry those two statistics:
  `_median(xs) -> float` (the median of a non-empty sequence, `statistics.median` semantics) and
  `_union_len(intervals) -> float` (the total length of the union of `(start, end)` pairs, in their own unit);
  `summarize_step` computes every `med_ms` through `_median` and every `busy_ms` through `_union_len`, looked up as
  module globals at call time (no alias, default argument or inlined copy), so `would-fail-mutants` can replace them.
  `parse_sim_slow(value) -> (rank, peer, factor, from_step) | None`
  raises `ValueError` on a malformed value.
- In `run()`, after `_optimizer_step(step)`: one block, timed with `time.perf_counter`, takes the events, appends them
  to the existing telemetry file, summarizes, prints `WT_TELEM <json>` (≤ 4 KiB), before `_checkpoint` and `WT_STEP`.
  Its duration is the next summary's `emit_us` (the first summary's is 0). `step_ms` spans `_run_step` start to the end of
  `_optimizer_step`.
- Test seam (timing only): `WANTRAIN_SIM_SLOW="rank:peer:factor[:from_step]"`, read once in `StageRuntime.__init__`; for
  the matching slot at global step ≥ `from_step`, each F/B/W sleeps `(factor − 1) × elapsed` inside its timed span,
  before its `_ev`.

**C2 — telemetry intake (control, CONTROL-I18).** Depends on R1's line format and C1's files.
- Agent: a `WT_TELEM` line of ≤ 4 KiB that parses as a JSON object becomes `{"kind": "TELEM_STEP", "run_id", "attempt",
  "summary"}`; anything else is dropped. It never reuses `TELEMETRY`.
- Coordinator: `_conn_loop` wraps each `_dispatch` so that any exception from one message is logged as `warn malformed
  <kind> from <worker>: <error>`, the message's file body (if any) unlinked, and reading continues; only channel
  errors end the loop. (Today a `KeyError` ends the reader silently, E23, and the unread heartbeats would later convict
  a live worker.)
- `TELEMETRY` branch: rank, peer, attempt come from `_slot_for` (the connection's slot and the current attempt), never
  from the message; no slot or a stale attempt unlinks the body and drops it; a missing or non-path body is dropped.
- `TELEM_STEP` branch: slot from `_slot_for`; accepted only if the summary is a dict, ≤ 4096 bytes re-serialized, has
  exactly the known keys, `v == 1`, its `rank`/`peer` equal the slot, its `step` is an int in [start step, steps − 1],
  every number finite with times in [0, 1e6] ms and counts/bytes in [0, 1e12]. Accepted summaries go to a per-slot
  in-memory window (`deque(maxlen=64)`) and are appended to `<run_dir>/telem_steps.jsonl` with `attempt`, until that
  file reaches `telem_log_max_bytes` (default 64 MiB, then one `warn` and no more writes). `run.rationale["telemetry"] =
  {"accepted": n, "dropped": d, "handle_us_max": x}` (perf_counter around the branch). Nothing reads the window yet.

**Cost model of telemetry (HYPOTHESIS, measured by the gates).** Per slot per step, E ≈ 3·M/w op events + sends + stalls
(M = 8 → ~30-40; M = 64, w = 1 → ~250). Summarize O(E log E) in Python ≈ 1 µs/event, JSON + print ≈ 50 µs: predicted
emit_us ≤ 300 µs vs a CPU step of 0.1–1 s (≤ 0.3%) and seconds on real models (< 0.03%). Coordinator intake ≈ 50 µs per
message; control bytes ≈ 1 KB/slot/step (64 slots at 1 step/s = 64 KB/s), negligible next to per-step wave shipping. The
gain vanishes (cost > 1%) only for steps < ~30 ms with M ≥ 64, which `TELEM_FRAC` would expose.

# Invariants

Delivered (spec text for the amendments):
- **CONTROL-I17.** The stall clock reads only `WT_STEP` progress inside [the attempt's start step, `steps − 1`], so a
  replayed loss stream never moves it and a resumed attempt is judged by `first_step_timeout_s` until its first real
  step. Once every launched slot finished the final step, quiet is shutdown, bounded by `shutdown_limit_s`, never a
  stall.
- **CONTROL-I18.** A worker's message never stops the coordinator from reading that worker: a malformed or stale
  message is logged and dropped and its file body unlinked. `TELEMETRY` and `TELEM_STEP` take rank, peer and attempt from
  the connection's slot, never from the message; a `TELEM_STEP` summary is accepted only within the size, key, range and
  identity rules of C2.
- **RUNTIME-I9.** Each slot prints one `WT_TELEM` per optimizer step, before `WT_STEP`: per-op F/B/W/OPT count and median
  ms, stall ms, per-hop frames, bytes and wire-busy ms, and its own emit cost, computed from monotonic timestamps only
  (no tensor op, no device sync, no RNG draw). The training state stays byte-identical to the reference.

Kept: RUNTIME-I1..I7 (I3 resume bit-for-bit, I4 fold by index, I7 normal shutdown), CONTROL-I6, I7, I8, I10.
CONTROL-I5 stays N/A under gpu false. SCHEDULER-I13/I15 stay DEFERRED (phases 2-3).

# Binding audit

No reduction, fold order or RNG draw is changed by this phase; every row below is state the change reads or rewrites.

| BA | What | Where (file:line) | Ruling |
|---|---|---|---|
| BA-1 | Index-order folds `OrderedAccumulator(range(self.M))` (chain, shard, batch) | stage.py:864, 872, 926 | Untouched. The SIM_SLOW sleep is inside op spans, after compute, before `_ev`: no data path. Proven by manifests equal to the reference in both integration tests. |
| BA-2 | Global RNG saved/restored in waves | stage.py:639, 669-670 | Telemetry and seams draw nothing (`telemetry.py` imports no torch, no random). Unit test asserts `torch.get_rng_state()` unchanged around `summarize_step`; manifests compare the RNG state. |
| BA-3 | Batch generation `make_batch(data_seed, step, mb)` | stage.py:720, 772 | Untouched. |
| BA-4 | Telemetry list shared by sender threads and the main loop | stage.py:133, 595-607, 681-683 | Lock in `_ev`, swap in `_take_events`; unit test: both block while the lock is held (deterministic), and concurrent appenders under a 1 µs switch interval lose and duplicate nothing. |
| BA-5 | Multi-stream `send` event times encode + enqueue | stage.py:358-371, 373-396 | Encoder emits `encode`; fragment senders emit `send` with hop around pace + write. Proven by the multi-stream arm of `test_step_telemetry_measures_ops_and_hops` (HOP_RATE_RATIO_MULTI, send count == fragment count). |
| BA-6 | `send` events lack hop; senders built without key | stage.py:175-187, 271-291, 445-455, 548-578 | Hop label passed to every sender; summary groups by hop. |
| BA-7 | `step_opt` events | stage.py:894, 955 | Exist; summarized as `OPT`. |
| BA-8 | Existing per-slot telemetry file and its exit shipment | stage.py:595-607, 1099; agent.py:534-547 | File keeps receiving every event (now from the swapped list); format gains `hop`/`encode`; no consumer (E27). |
| BA-9 | Step loop order: optimizer → telemetry → checkpoint → `WT_STEP` | stage.py:967-990 | `WT_TELEM` inserted before `_checkpoint`; `WT_STEP` stays last per step, so progress still means "optimizer step done and wave written". |
| BA-10 | Loss stream replay on resume | stage_main.py:29-55, 63-65 | Not edited; neutralized at the coordinator (BA-12). |
| BA-11 | `PROGRESS` source | agent.py:463-491 | Adds `src`. |
| BA-12 | Progress stamping at arrival | coordinator.py:374-388 | Only `src == "step"`, via `add_progress` (range [start_step, steps−1], int, strictly increasing: ADR-0003). |
| BA-13 | `median_step_s`, `last_progress_at`, S4 limit | liveness.py:73-90, 217-228 | Formulas unchanged; their inputs no longer contain replay. |
| BA-14 | S4 after the final step | liveness.py:217-265 | Replaced by the shutdown bound in shutdown mode; S3 unchanged. |
| BA-15 | Per-attempt `SlotState` creation | coordinator.py:914-931, 1087-1118 | Sets `start_step` from the restored wave. |
| BA-16 | Restart / heal from wave `steps − 1` | coordinator.py:1142-1205; stage.py:651-679, 959-990 | Kept for S2/S3/convict (reruns `_finish`, exact by RUNTIME-I3); unreachable from S4 quiet. |
| BA-17 | Resume spec | fault_tolerance/checkpoint.py:143-153 | Untouched (FIT-4 citation corrected). |
| BA-18 | Reader thread death on a message exception | coordinator.py:297-310, 324-361 | Per-message catch, log, unlink body, continue. |
| BA-19 | `TELEMETRY` destination from message fields | coordinator.py:352-356 | From `_slot_for` + `run.attempt` only. |
| BA-20 | Agent post-exit path (seam point) | agent.py:493-518 | Delay only when `WANTRAIN_SIM_EXIT_DELAY_S` is set and rc == 0; heartbeats keep `stage_alive` true (agent.py:284-290). |
| BA-21 | Stage stdout pipe drained by the agent's pump, which calls `send` synchronously | agent.py:59-70, 213-221 | `WT_TELEM` adds ~1 KB/step to an existing exposure (a coordinator that stops reading back-pressures the stage). Not changed; Risk R4. |

# Tripwire

None. Reason: nothing consumes `TELEM_STEP` or the window in this phase, so telemetry without its consumer is inert,
and the consumer (re-plan) arrives in phase 2 behind its own switch (`CoordinatorConfig.replan = False`, armed by no
phase until M1 real-hardware evidence). Partial landings here are loud, not silent: a coordinator that requires
`src == "step"` without the agent's `src` sees no progress and fails at `first_step_timeout_s` in every integration
test; telemetry without intake is ignored by today's `_dispatch`. C1 and C2's agent edits land in one writer's tasks.

# Exit gate

Rungs, run twice in a row: factory, build, syntax, fmt, lint, types, discipline, invariants (CONTROL-I17/I18, RUNTIME-I9
covered), claims, unit, integration, multiworker, oracle, identity, chaos, wan, full, static, security, fuzz. N/A with
reason: frozen (FROZEN.json absent), component, multinode, matrix, rest, redteam (absent), bench (bench/ absent;
replaced by the measured telemetry numbers below).

| Number | Bound | Model | Measured by |
|---|---|---|---|
| SEAM_RATIO, slowed/unslowed F `med_ms` of the two peers of the wide stage, SIM_SLOW factor 4 | ≥ 2 | 4 | `tests/integration/test_engine_bitwise.py::test_step_telemetry_measures_ops_and_hops` (`telemetry-engine`) |
| HOP_RATE_RATIO, median wire rate on links[1] (160 Mb/s) / links[0] (40 Mb/s), rate = bytes·8/busy_ms; mesh path (wide stages: `_connect_mesh` → `_wire` → `_sender`) | ≥ 2.5 | 4 | same |
| HOP_RATE_RATIO_MULTI, the same ratio on the multi-stream path (chain, `p2p_streams = 2`: `_wire_multi` → `_mux_encoder` → `_frag_sender`) | ≥ 2.0 (ADR-0005: measured 2.42-2.48 under the emulation's per-frame cost; an encode-timed `send` gives ~1) | 4 | same test, second arm |
| TELEM_US_MAX, max `emit_us` over every summary of the run | reported, not bounded (ADR-0005: an absolute µs bound measures the host's load; TELEM_FRAC bounds the cost against the step) | ≤ 300 | `tests/integration/test_coordinator.py::test_live_telemetry_reaches_the_coordinator_byte_exact` (`telemetry-intake`) |
| TELEM_FRAC, median(`emit_us`)/1000 / median(`step_ms`) | ≤ 0.01 | ≤ 0.003 | same |
| TELEM_INTAKE_US_MAX, `rationale["telemetry"]["handle_us_max"]` | ≤ 2000 | ~50 | same |
| QUIET_S, quiet between the fleet's last `WT_STEP` and the first slot's exit report, with delay 45 s, `first_step_timeout_s` 30 s, floor 8 s | > 30 (asserted in test) | 45 | `test_slow_shutdown_after_the_last_step_completes` |
| Restarts in the slow-shutdown run | == 0 (asserted) | 0 | same |

Discriminating arms: the only performance number is the mechanism's own cost, measured in-band on its own block (one
factor). An on/off A/B of step time would sit below CPU step-time noise (σ ≫ 1 ms) and attribute nothing; not pinned.
SEAM_RATIO, HOP_RATE_RATIO and HOP_RATE_RATIO_MULTI each vary one factor (compute of one peer; bandwidth of one link)
with distinct expected values (E13). A constant implementation fails all three. An encode-only `send` fails only
HOP_RATE_RATIO_MULTI: on the mesh path the pre-phase `send` event already spans encode + pace + `send_frame`
(stage.py:548-578), so HOP_RATE_RATIO is ~4 there whatever the multi-stream path does; only the second arm, a chain with
`p2p_streams = 2`, runs `_mux_encoder` and `_frag_sender` (stage.py:256-291), where BA-5 sits.

Test contracts (every test states `covers:` and `WOULD-FAIL-IF:`):
- C1 unit (`tests/unit/test_fault_tolerance.py`, covers CONTROL-I17, injected clock):
  `test_replayed_metric_progress_never_feeds_the_stall_clock` (a burst of `src="metric"` and of out-of-range steps is
  not recorded; a `src="step"` stream 2, 3, 2, 3, 2 records 2 and 3 once and a later quiet still reaches S4 (ADR-0003); an S4 limit computed after it equals the limit without it; a coordinator that stamps metric progress
  fails. Coordinator part: a `Coordinator` on `tmp_path` with a `_Run` as in the relaunch part below, a `src="metric"`
  burst of `PROGRESS` messages carrying the run's `run_id` and `attempt`, steps inside [start_step, `steps − 1`] and each
  different from the slot's last recorded step, sent through the real `Coordinator._dispatch` from the connection bound
  to its slot: the slot's `progress` is unchanged after the burst; then one `src="step"` `PROGRESS` of an in-range new
  step through the same path is recorded. WOULD-FAIL-IF `_slot_update` records progress without checking `src`); `test_resumed_attempt_is_judged_by_the_first_step_timeout` (two parts. Judge: start_step = 5, no real progress, quiet
  of floor + 1 s is `ok`, quiet of `first_step_timeout_s` + 1 is `stalled`. Relaunch: a `Coordinator` on `tmp_path` and a
  `_Run` of 2 slots with a 4-step spec, whose worker requests are replaced inside the test by a recorder (a fake at the
  worker-network boundary, named as such: `_request` returns `{"ok": True}`, `_sync_all_connections` does nothing); the
  real `_restore_and_relaunch(run, 1, spec_json)` gives every new `SlotState` `start_step == 2` (distinct from 0,
  `steps − 1 = 3` and `steps = 4`); a `src="step"` progress of step 1 is not recorded and one of step 2 is; judged with
  `final_step = 3` and no real progress, quiet of `first_step_timeout_s` + 1 is `stalled` (not shutdown); then
  `_restore_and_relaunch(run, None, spec_json)` gives `start_step == 0`. WOULD-FAIL-IF `_launch` sets `start_step` to
  `steps` or `final_step + 1` (shutdown from the start, every real `WT_STEP` dropped: S4 off for a resumed attempt) or
  leaves it at 0 on a mid-run relaunch; `final-wave-relaunch` cannot tell these apart, since there `wave + 1 = steps`); `test_quiet_after_the_final_step_is_shutdown_not_a_stall`
  (quiet > floor before the final step is `stalled`; after every slot reported `steps − 1` it is `ok`; past
  `shutdown_limit_s` it is `failed` with "shutdown exceeded"; one slot at `steps − 1` and another at `steps − 2`, quiet
  > floor, is `stalled`; one slot exited 0 and another still owes a step, quiet > floor, is `stalled`; a judge that only
  raises the floor fails case 1 or 3, and a judge that enters shutdown when any slot (not every slot) finished fails
  the two lagging-slot cases).
- C2 unit (covers CONTROL-I18): `test_malformed_messages_never_stop_the_reader` (a kindless message, a bodiless
  `TELEMETRY`, one with a non-int rank: each logged, the next `HEARTBEAT` still updates `last_heartbeat`, no file
  moved outside `run_dir`, stale bodies unlinked. Identity part, BA-19: a run at `attempt` 1 with two slots whose rank
  and peer both differ (e.g. (0,0) and (1,1)), slot (0,0) already holding `run_dir/telemetry_r0p0_a1.jsonl`; from the
  connection bound to (1,1), a `TELEMETRY` whose body is a real file and whose `rank` 0, `peer` 0 are ints naming the
  other slot, `attempt` 1: the body's bytes land at `run_dir/telemetry_r1p1_a1.jsonl`, (0,0)'s file is byte-unchanged
  and no other `telemetry_*` file appears; then from the same connection a `TELEMETRY` with `attempt` 0 (stale) and a
  real body: the body is unlinked, `telemetry_r1p1_a0.jsonl` does not exist and (1,1)'s file is unchanged. WOULD-FAIL-IF
  the branch builds the destination from the message's `rank`, `peer` or `attempt` (slot (0,0)'s file is overwritten)
  or moves a stale attempt's body); `test_telem_step_identity_and_bounds_are_checked_at_the_boundary`
  (every `TELEM_STEP` sent through the real `_dispatch` from the connection bound to its slot; a run with two slots
  whose rank and peer both differ, e.g. (0,0) and (1,1); one accepted summary; each of: wrong identity, step out of
  range, NaN, 5 KiB, unknown key, stale attempt → dropped and counted; distinct accepted/dropped counts. The wrong
  identity case names the other existing slot: from the connection bound to (1,1), a summary with `rank` 0, `peer` 0
  (ints) is dropped, and (0,0)'s window and the accepted count stay unchanged. WOULD-FAIL-IF the branch indexes the
  window by the summary's `rank`/`peer` instead of `_slot_for` (the summary lands in (0,0)'s window)).
- R1 unit (`tests/unit/test_runtime_spec.py`, covers RUNTIME-I9): `test_step_summary_is_a_timing_only_median`
  (synthetic events with distinct durations where mean ≠ median; overlapping send intervals where union ≠ sum; through
  `summarize_step`, asserts each op's `med_ms` equals the median (not the mean) and the hop's `busy_ms` equals the union
  length (not the sum); RNG state unchanged; `parse_sim_slow` rejects malformed values. WOULD-FAIL-IF `summarize_step`
  uses a mean or a sum of intervals); `test_telemetry_events_are_neither_lost_nor_duplicated_across_threads`
  (on a `StageRuntime` built without sockets; two parts. Deterministic: while the test itself holds `rt._tel_lock`, an
  `_ev` call in one thread and a `_take_events()` call in another are both still blocked after 0.2 s; after the release
  both finish, and the event appears exactly once across the two takes. Stress: `sys.setswitchinterval(1e-6)` (restored
  after), 4 appender threads × 2000 `_ev` with distinct ids racing a loop of `_take_events()`; the union of all takes
  holds every id exactly once. WOULD-FAIL-IF `_ev` or `_take_events` does not hold `_tel_lock` (the deterministic part
  fails on every run, whatever the GIL schedule), or the take iterates and then `clear()`s (the stress part)).
- Integration: `test_step_telemetry_measures_ops_and_hops` (runtime; local run, widths [1,2,1], M = 8, `wire_codec="raw"`,
  links [40, 160] Mb/s, `WANTRAIN_SIM_SLOW=1:1:4`; one `WT_TELEM` per slot per step parsed from the stage logs, F `n`
  per slot equals its owned microbatches, ops complete (below), every paced hop's rate ≤ 1.1 × its link, manifests `diff_manifests(...) ==
  []` and `==` against `reference_manifest_for`; prints `SEAM_RATIO=`, `HOP_RATE_RATIO=`; covers RUNTIME-I9). Second
  arm in the same test, the multi-stream path: a chain of 3 stages (no wide stage), `p2p_streams = 2`, M = 8,
  `wire_codec="raw"`, links [40, 160] Mb/s, no SIM_SLOW; in each slot's `<tag>_rank<r>p<p>_telemetry.jsonl` every
  `send` event carries a `hop`, and the number of `send` events equals the sum of `frags` over its `encode` events (the
  fragment senders emit `send`, the encoder emits `encode`); every paced hop's rate ≤ 1.1 × `p2p_streams` × its link;
  manifests equal the reference as in the first arm; ops complete (below) in both arms; prints `HOP_RATE_RATIO_MULTI=` and
  `OPS_CHECKED=`, the number of summaries whose ops and hop frames it checked over both arms. WOULD-FAIL-IF the encoder still
  emits `send` around encode + enqueue (the ratio is ~1 and the counts differ).
  *Ops complete* (both integration telemetry tests, every summary they read): `ops` F, B and W each have `n` equal to
  the number of that op the slot's schedule runs in that step, derived from the spec (its owned microbatches × its
  virtual stages), never from the events; OPT `n` == 1; every op's `med_ms` > 0. Per slot and per hop `h`, with the
  slot's telemetry file (the event log the same take appends to): count of its `send` events with `hop == h` and
  `step ≤ steps − 2` ≤ Σ over the slot's summaries of `hops[h].frames` ≤ count of all its `send` events with `hop == h`.
  The upper bound holds exactly (every taken event is appended to the file). The lower bound holds by the step order:
  a frame of step ≤ `steps − 2` is received before any slot can finish step `steps − 1`, so its `send` completed
  before the slot's final summary; only the final step's sends (and the shutdown drain) may complete after it, and
  they reach the file only. Exact equality is not required for that reason. WOULD-FAIL-IF `summarize_step` maps an
  event name wrongly (stage events are `fwd`, `bwd_b`, `bwd_w`, `step_opt`, `stall`, `send`, `encode`: a wrong name
  gives `n` 0 or `med_ms` 0) or drops the sends that complete after their own step's summary.
  `test_slow_shutdown_after_the_last_step_completes` (control; `LocalCluster(..., first_step_timeout_s=30)` with only
  today's config fields, `monkeypatch.setenv("WANTRAIN_SIM_EXIT_DELAY_S", "45")`, 2 stages, 4 steps (E35: the pre-phase
  limit is then ≤ 30 s whether or not the median exists); assertions in this order and exactly these messages:
  `assert res.ok, f"RUN_FAILED: {res.reason}"`, `assert res.restarts == 0, f"RESTARTED: {res.restarts} restart(s)"`,
  `assert quiet_s > first_step_s, f"QUIET_TOO_SHORT: {quiet_s:.1f}s"` (`first_step_s = 30`), then the manifest checks; no
  `pytest.fail`, no multi-line message; it uses no name, config field or message field this phase adds; covers
  CONTROL-I17). `test_relaunch_from_the_final_wave_runs_no_step_and_is_not_restarted`
  (control; `LocalCluster(..., first_step_timeout_s=20)` (an existing field), 3 workers for 2 stages (one spare), 4
  steps, `monkeypatch.setenv("WANTRAIN_SIM_EXIT_DELAY_S", "30")`; waits until every slot reported step 3 and
  `run.store.common_step(list(run.slots)) == 3`, then kills one assigned worker's agent process; asserts `res.ok`,
  exactly one heal with `resumed_from_step == 3`, `res.restarts == 0`, manifests equal the reference; covers
  CONTROL-I17; WOULD-FAIL-IF `_launch` leaves `start_step` at 0: the relaunched attempt runs no step, its 30 s quiet
  exceeds `first_step_timeout_s` and it is restarted until the budget is spent).
  `test_live_telemetry_reaches_the_coordinator_byte_exact` (control; widths
  [2,1]; `telem_steps.jsonl` holds exactly one summary per slot per step, rank/peer equal the slot, F `n` equals the
  slot's owned count, ops complete (above; the slot's telemetry file is the one shipped at exit,
  `run_dir/telemetry_r<r>p<p>_a<attempt>.jsonl`), `dropped == 0`, manifests equal the reference; prints `TELEM_US_MAX=`,
  `TELEM_FRAC=`, `TELEM_INTAKE_US_MAX=`, `OPS_CHECKED=`, each fixed-point `:.6f`; covers CONTROL-I18, RUNTIME-I9).
- Pre-phase tests: every top-level statement of the four owned test files at E34 stays AST-identical; new tests, imports
  and helpers are added as new statements, never by editing an existing one (`pre-tests-unchanged`). A new top-level
  statement is only an import, a function definition or an assignment to plain names; it never rebinds a name the
  pre-phase file binds at top level (a later `def test_<same name>` would replace the old body at import), never
  declares a fixture with `autouse` and never defines a `pytest_*` hook. It declares no fixture at all (new tests use
  `tmp_path`, `monkeypatch`, `port_base` and plain helpers): a fixture named, by function name or `name=`, like one a
  pre-phase test requests would change how that test runs. A new def carries no decorator other than
  `pytest.mark.<name>` with call-free arguments and has no call in its defaults or annotations (both run at import); a
  new assignment has no call anywhere in it (only constants, names, attribute reads, subscripts, operators, tuples,
  lists, dicts, sets and call-free comprehensions: a computed value belongs in a def called inside the test, since a
  call at import, e.g. `_lv.__dict__.update(...)`, would change what every later test of the session sees) and binds
  no fixture name a pre-phase function or `tests/conftest.py` uses. The pre-phase statements keep their relative order
  (new statements may sit between them, e.g. an import in its sorted place).
- Mutants (`would-fail-mutants`, E13 made mechanical for the four claims no other gate runs against their regression):
  a pytest plugin defined in the GATES entry, loaded with `-p`, breaks one thing at run time and the named node must
  fail with an `AssertionError` (exit 1, `1 failed`, no `TypeError`/`AttributeError`/`KeyError`-class error, and the
  plugin recorded that its mutation was reached). `lock`: `StageRuntime._tel_lock` becomes a lock that never blocks →
  `test_telemetry_events_are_neither_lost_nor_duplicated_across_threads` fails. `launch-zero` / `launch-late`: after
  `Coordinator._restore_and_relaunch(run, step, ...)` with a wave, every new `SlotState.start_step` is set to 0 / to
  its value + 2 (`steps` in the 4-step spec) → `test_resumed_attempt_is_judged_by_the_first_step_timeout` fails.
  `identity`: `Coordinator._slot_for` returns the slot the `TELEMETRY` message's `rank`/`peer` name →
  `test_malformed_messages_never_stop_the_reader` fails. `identity-telem`: `Coordinator._slot_for` returns the slot a
  `TELEM_STEP` message's `summary` `rank`/`peer` name → `test_telem_step_identity_and_bounds_are_checked_at_the_boundary`
  fails. So these tests must drive the real `_restore_and_relaunch`, send `TELEMETRY` and `TELEM_STEP` through
  `_dispatch` (whose branches take the slot from `_slot_for`), and hold `rt._tel_lock`, the
  attribute `_ev` and `_take_events` lock. The mutants are fixed by this granted text, not written by the test's
  author; no `.would-fail.patch` is used because the phase owns no path for one.
- `causal-slow-shutdown` runs `tests/integration/test_coordinator.py` against the pre-phase `liveness.py` and
  `coordinator.py`: at module level that file imports only names that exist in them; a name this phase adds is imported
  inside the test function that needs it (an `ImportError` at collection is `CAUSE=other`).

Pinned entries (`GATES-m2-p1-liveness-telemetry.json`, 14): `spec-amended` (exactly one
`docs/design/adr/ADR-NNNN-m2-p1-spec.md`, the human grant of it and of `docs/design/DESIGN.md`, then the greps),
`no-pending-markers` (grep exit 1 only), `no-new-skip-markers` (widened scan, digest E33), `pre-tests-unchanged` (the
Gate's saved copies at digests E34; every pre-phase top-level statement of the four owned test files is still present,
AST-equal and in their pre-phase relative order; every new top-level statement is an import, a def or a plain-name
assignment that rebinds no pre-phase top-level name or requested fixture name, with no `autouse`, no `pytest_*` hook,
no fixture, no decorator but `pytest.mark.<name>` with call-free arguments, no call in a new def's signature and no
call anywhere in a new assignment), `liveness-unit` ×20, `intake-unit` ×5, `telemetry-unit` ×20, `would-fail-mutants`
(5 mutants, each caught),
`slow-shutdown-cpu` ×3 under 2× load,
`final-wave-relaunch` ×3, `causal-slow-shutdown` (digests E32, arm A must fail with `CAUSE=stall`, arm B must pass,
2 runs, 900 s per run, 4200 s outer; `CAUSE=stall` needs at least one `fleet stalled: no optimizer step fleet-wide`
coordinator line, no `fleet stalled:` line of another kind, a `RUN_FAILED:` naming that S4 reason and the restart budget
or a `RESTARTED:` assertion, and no conviction, no thread exception, no `KeyError`/`TypeError`-class error; any other
failure of arm A is `CAUSE=other`), `telemetry-engine` ×3 (SEAM_RATIO ≥ 2, HOP_RATE_RATIO ≥ 2.5,
HOP_RATE_RATIO_MULTI ≥ 2.5 and OPS_CHECKED ≥ 7: 4 slots + 3 slots, at least one step each), `telemetry-intake` ×3
(OPS_CHECKED ≥ 3: 3 slots), `suite-nodes` (frozen list digest E33; the canonical command
without `-u` prints no duplicate node, so no test function is shadowed by a later one of the same name; every
pre-phase function and the 11 new nodes `PASSED`; exactly two `SKIPPED` lines, both GPU_OFF; no
`FAILED`/`ERROR`/`XFAIL`/`XPASS`). Every node-naming entry requires pytest exit 0 and the exact `<n> passed` summary.
Every `KEY=` value a test prints for a gate is fixed-point, `print(f"KEY={value:.6f}")` (never `str`/`repr` of a
float, which prints `5e-05` below 1e-4); the gates' awk and metric patterns also read exponents (`[0-9.eE+-]+`).
The phase workflow accepts the relayed verdict only for this gate id (`milestone`), with every one of these entries,
the 20 rungs and `consecutive` present and passing, on the tree it audited before and after the verify.
Canonical list command (Gate step and E33):
`grep -rnE '^(    )?(async )?def test_[A-Za-z0-9_]+' tests --include='test_*.py' | sed -E 's/^([^:]+):[0-9]+:[[:space:]]*(async )?def (test_[A-Za-z0-9_]+).*/\1::\3/' | LC_ALL=C sort -u`.

# Phases

1. **Gate.** `checkApprovals` on this NEXT, the GATES file and the built workflow, and nothing before it. Then a
   read-only agent reports the number of the one `docs/design/adr/ADR-NNNN-m2-p1-spec.md` (the script builds the path
   from a validated integer) and `checkApprovals` covers `docs/design/DESIGN.md` and that ADR; a missing, ambiguous or
   ungranted ADR stops the phase and names the `/wan-adr` route. Precondition (bracketed by a scope audit that allows no
   tracked change): exactly one such ADR file, then the `spec-amended` greps. Save the four owned test files to
   `.factory/runs/m2-p1-liveness-telemetry/tests-pre/` only if they match E34 (keep an existing save whose four digests
   match; otherwise stop for a human). Save C1's pre-phase files:
   `node agents-architetura/tools/verify/causal.mjs save --out .factory/runs/m2-p1-liveness-telemetry/causal-c1 --
   src/wantrain/fault_tolerance/liveness.py src/wantrain/coordinator/coordinator.py` only if the live files match E32
   (keep an existing save whose three digests match; otherwise stop for a human). Write
   `.factory/runs/m2-p1-liveness-telemetry/nodes-pre.txt` with the canonical command; stop unless its sha256 is E33's.
2. **Wave 1 (parallel, disjoint).** R1 (writer R). C1 with its 3 unit tests, the slow-shutdown and the final-wave
   relaunch integration tests (writer C). Each wave, review fix and stabilize item is bracketed by `scope-audit.mjs snapshot` / `check --scopes`
   with that item's own owned paths.
3. **Wave 2.** C2 (writer C): agent forwarding, per-message catch, `TELEMETRY` hardening, `TELEM_STEP` intake, its 2 unit
   tests and the intake integration test. Depends on R1 and C1.
4. **Review (reviewUntilClean, ≤ 3 rounds), lenses written for this objective, over every owned path (not only the
   files this run changed: a rerun of the phase must review what an earlier run's stabilize or fix rounds changed,
   since the Gate keeps the earlier saves and a rerun may change little):**
   (a) *liveness soundness:* any path that still stamps metric/replayed/out-of-range progress; shutdown-mode entry when
   one slot lags; S3 and heal during shutdown; first-step timeout after resume; the final-wave relaunch still exact.
   (b) *byte-exact:* walk BA-1..BA-9; can a seam, lock or summary touch a tensor, an RNG draw or a fold?
   (c) *tests that fail on a wrong implementation:* a judge that raises the floor, a summary using mean or sum, an
   encode-timed hop, a coordinator trusting message rank/peer, a `_launch` whose `start_step` is 0, `steps` or
   `final_step + 1` on a mid-run relaunch: does a named assertion fail for each? No new fixture or non-mark decorator;
   `test_coordinator.py` imports at module level only pre-phase names.
   (d) *boundary and cost:* identity from the connection, size/range checks, file paths, the log cap; each cost against
   the step time (MEASURED by the gates or HYPOTHESIS).
   (e) *telemetry fidelity:* step attribution of async sends, union vs sum, lost events at swap, `emit_us` covering the
   whole block (VER-5).
   A review that does not converge stops the phase before Verify (`stoppedAt: "review"`, with the open findings): a
   human decides (grant with the findings recorded, one more review, a split, or stop). Stabilize never applies code
   on top of unresolved blocker or major findings.
5. **Verify and stabilize.** A solo verifier runs `node agents-architetura/tools/gates/pinned.mjs m2-p1-liveness-telemetry
   --json` and relays it. ≤ 4 stabilize rounds, one item at a time (no two appliers run tests together), each
   proposal → 3 votes → scoped apply → the same command. Rules this phase's workflow must implement:
   - M2-R5-SAFE-1: a round counts as stabilized, and its review runs (whether or not the last verify passed; files
     it could not clear are returned as `unreviewed`), whenever its scope audit shows changed files,
     whatever the applier's outcome;
   - M2-R5-SAFE-2: the stabilize review includes the tests lens (c) and lens (a); an item that edits any file under
     `tests/` needs 3 of 3 votes, any other a majority with the spec voter's veto; every stabilize proposer, voter and
     applier carries this document's facts (mechanism, binding audit, exit gate), not only the generic fix contract;
   - M2-R5-SAFE-3: a `specConflict` from any writer (wave, review fix, stabilize applier) goes to the ADR stop;
   - every verify is bracketed by a scope audit that allows no tracked change (the verifier runs a shell; an edit it
     made would reach no review);
   - the relayed JSON counts only if its `milestone` is this slug, it lists every rung, `consecutive` and pinned entry
     of the GATES file as passing, and its `tree` equals the tree audited before and after that verify;
   - `ok` only on VERIFIED, a converged review, a clean stabilize review when anything was applied, and a final
     snapshot equal to the verified tree; the result's `next` follows from it: phase 2 only when `ok`; otherwise the
     failing gate ids, the intermittent gates, or the unconverged stabilize review, whichever holds;
   - every scope audit must report its `changed` list and its tree, consistently (a changed tree with no changed file,
     or the reverse, stops the phase): the stabilize review is driven by that list, the first review by every owned path;
   - a flaky pass is not a pass: a stabilize round that changed no file ends the loop (the same tree is never verified
     again), and a VERIFIED on a tree that already produced a failing verify is NOT VERIFIED, routed to `/wan-bughunt`
     as intermittent; every earlier failing gate id is carried in the result.

# Risks

- **R1 — `shutdown_limit_s` too low for large shards.** T_quiet grows with S_shard/B_ctrl; past 600 s a completed run is
  reported failed (weights stay on worker disks and the last shipped wave). Configurable; the failure text names the
  knob. A size-aware bound is a later decision.
- **R2 — CPU timing noise** in SEAM_RATIO and HOP_RATE_RATIO: bounds sit at half (2/4) and 0.6× (2.5/4) of the model,
  each run ×3. A failure is evidence against the model, never a reason to widen a bound.
- **R3 — the MPS stall may have a second cause.** M2-MPS-INTERMITTENT stays open until a gpu-true `/wan-bughunt`.
- **R4 — stdout back-pressure** (BA-21): pre-existing; telemetry adds ~1 KB/step. Recorded for phase 2's review.
- **R5 — spec-amendment precondition** needs a human grant before the phase starts.
- **R6 — real hardware:** all numbers SIMULATED; device-sync effects on GPU op timing (asynchronous kernels make CPU
  timestamps measure launch, not execution) are unverified and matter for phase 2's re-solve. M1 work.
- Deferred by ADR-0003: `CKPT` and `RESULT` in `coordinator._dispatch` take rank and peer from the message, and `CKPT`
  does not check the attempt, so an admitted worker can ship a wave or a final result under another slot. This phase does
  not change them; the finding M2-P1-CKPT-IDENTITY records it for a later phase and THREAT_MODEL T7 names the residual.
