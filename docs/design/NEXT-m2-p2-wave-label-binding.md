# NEXT: m2-p2-wave-label-binding

Phase workflow: `wan-x-m2-p2-wave-label-binding`. Pinned verify: `docs/design/GATES-m2-p2-wave-label-binding.json`.
Milestone M2 (adaptivity), phase 2. Hardware: `{"gpu": false}`. Everything below runs on this machine's CPU. The one
performance number is SIMULATED, and real-hardware validation stays pending.

Re-split of M2 (LESSONS.md:10-13). The red `chaos` rung comes before deferred work, so it takes phase 2. The phases
that follow are: **phase 3** re-solve from telemetry and resume under a new ownership, behind `replan = False`
(SCHEDULER-I15, M2-R5-VER-2), the work NEXT-m2-p1 had planned as phase 2; **phase 4** plan mode for wide stages
(M2-R5-FIT-2, FIT-5) and SCHEDULER-I13. This phase has 2 work packages and 9 pinned entries.

# Objective

Every checkpoint wave that a worker ships, the coordinator stores and restores, and a stage resumes must hold the step
its label names. The fix makes the stage's shard/sidecar write a seqlock, makes the agent's restore and ship keep that
contract, and makes a resume refuse a shard whose step differs from its sidecar. The goal is that
`test_dead_worker_is_replaced_and_the_run_stays_exact` (red `chaos` rung) heals exactly and never ends in a stall.

# Evidence

Facts only, each with its source. "Read" means read for this document on the current tree. The failing run's files are
in `/private/var/folders/vd/lgy0gn091fq4zbt0fnvp7xkc0000gn/T/pytest-of-virgiliofilho/pytest-331/test_dead_worker_is_replaced_a0/`
(below: `$D`, run `run-20261009-102607-d67f50` = `$R`). That directory is pytest's temporary output and will be rotated.
The facts copied here are the record.

| # | Fact | Source |
|---|---|---|
| E1 | Ladder on tree `1cbc5459…`: every applicable rung passes except `chaos`. `chaos` fails on `tests/chaos/test_recovery.py::test_dead_worker_is_replaced_and_the_run_stays_exact`: "fleet cannot progress (no optimizer step fleet-wide for 90s (limit 90s) while every worker is alive) and the restart budget (2) is spent", 1 failed, 5 passed in 353 s. | state snapshot `ladder`; `.factory/ladder/last.json` (read) |
| E2 | Coordinator log of that run: w1 is convicted, w2 replaces slot (1, 0), "every slot resumes after step 3", "wave 3 restored to every slot (sha256 verified)". Then attempts 2, 3 and 4 each end in "fleet stalled", with restarts in place from step 3. | state snapshot `ladder` detail; `$D/coord/runs/$R/result.json` (read) |
| E3 | Stage logs. Rank 0 (w0) prints `WT_RESUME rank=0 start_step=4` in attempts 2, 3 and 4. Rank 1 on the replacement w2 prints `WT_RESUME rank=1 start_step=6` in all three. The two ranks resumed from different waves (3 and 5). | `$D/w-w0/runs/$R/stage-r0p0.log`, `$D/w-w2/runs/$R/stage-r1p0.log` (read) |
| E4 | The coordinator's store holds `slot-r0p0-step-3.pt` (sha256 `fc3f3cc1…`), `slot-r0p0-step-5.pt` (`c8c7c79a…`) and `slot-r1p0-step-3.pt` (`1c39599d…`). `1c39599d…` is also the sha256 of the victim w1's last local shard, whose sidecar reads `{"step": 5}`, and of the shard restored to w2. **The wave the store labels step 3 for slot (1, 0) holds the bytes of step 5.** | `shasum -a 256` over `$D/coord/runs/$R/ckpt/*` and `$D/w-w*/runs/$R/out/*.pt`; `$D/w-w1/runs/$R/out/*step.json` (run for this document) |
| E5 | After the restore, w0's shard is wave 3 (sha256 `fc3f3cc1…` = store step 3), but w0's sidecar still reads `{"step": 5}`. w2's out dir has a shard and no sidecar. **A restore leaves a stale sidecar, or none.** | `$D/w-w0/runs/$R/out/` and `$D/w-w2/runs/$R/out/` listings and sha256 (run for this document) |
| E6 | Writer order in `StageRuntime._checkpoint`: `torch.save(…, tmp)`, then `os.replace(tmp, path)`, then the sidecar is written in place with `open(…, "w")`. Nothing invalidates the old sidecar before the new shard replaces the old one. | src/wantrain/runtime/stage.py:708-736 (read) |
| E7 | Reader `WorkerAgent._ship_checkpoint` reads the sidecar (`before`), copies the shard into `ship-r{r}p{p}.pt`, reads the sidecar again, and ships with `"step": before` if the two reads are equal. Between E6's `os.replace` and its sidecar write, the sidecar still names the old step while the shard is the new one. A copy that opens the shard in that window passes the check and labels the new bytes with the old step. | src/wantrain/worker/agent.py:598-639 (read) |
| E8 | `_ship_checkpoint` runs both from `_supervise` (its own thread, every 0.5 s and after exit) and from `_on_abort` (the reader thread). Both write the same snapshot path `ship-r{r}p{p}.pt`, and no lock covers the copy or the send. `Channel.send_file` reads the file twice: once to hash it, then again to send it. | src/wantrain/worker/agent.py:538-543, 613, 641-652; src/wantrain/worker/agent.py:280-292 (handlers run on the reader thread); src/wantrain/transport/control.py:66-90 (read) |
| E9 | `_on_restore` verifies the sha256 and runs `os.replace(path, dst)`. It neither writes nor removes the sidecar. | src/wantrain/worker/agent.py:654-672 (read) |
| E10 | `_resume` takes the start step from the shard's own `state["step"]` and never compares it with any label. | src/wantrain/runtime/stage.py:738-766 (read) |
| E11 | The coordinator files a `CKPT` under the step the message names. If that (slot, step) is already held, a later wave carrying the same label is unlinked. `CheckpointStore.add` checks only the sha256 the sender computed. `common_step` picks the newest step that every slot holds. | src/wantrain/coordinator/coordinator.py:388-397; src/wantrain/fault_tolerance/checkpoint.py:89-113 (read) |
| E12 | The contract the code claims: "a sidecar naming step N proves the shard for step N is complete" (checkpoint.py docstring; stage.py comment at the sidecar write). SPEC-runtime §Checkpoint says "plus a sidecar with the step". CONTROL-I8 says "A corrupt shard (sha256 mismatch) is refused". CONTROL-I10 says every recovered run stays bit-identical, a dead worker included. | src/wantrain/fault_tolerance/checkpoint.py:3-6; src/wantrain/runtime/stage.py:732-734; docs/design/SPEC-runtime.md:53-55; docs/design/SPEC-control.md:156-166 (read) |
| E13 | Phase 1 closed VERIFIED on tree `885310c5…`, with the `chaos` and `full` rungs passing in 2 of 2 consecutive runs. The same test now fails once, so the failure is intermittent. | `.factory/gates/m2-p1-liveness-telemetry.json` (read) |
| E14 | A stage exception exits nonzero with `WT_DONE`. When the log tail lacks "peer connection lost", the judge returns `failed` ("this is the run, not the hardware") immediately, without a stall or a conviction. | src/wantrain/worker/stage_main.py:80-91; src/wantrain/fault_tolerance/liveness.py:233-248 (read) |
| E15 | Every resume path: (a) coordinator heal and restart in place, via `RESTORE` and then `resume=true`; (b) the oracle's `resumed` case and (c) the integration test `test_checkpoint_resume_continues_the_sequence_bit_for_bit` (`integration` rung), each resuming from the first run's own shard and sidecar in the same `out_dir`. `CheckpointStore.export_wave` copies shards without sidecars and has no caller. `tests/helpers/exact.py` `fleet_manifest` opens every shard's sidecar unconditionally and picks the newest step. | src/wantrain/coordinator/coordinator.py:1212-1244, 1298, 1329; tests/oracle/test_state_identity.py:55-61; tests/integration/test_engine_bitwise.py:80-98; src/wantrain/fault_tolerance/checkpoint.py:132-140; `grep -rn export_wave src tests` (one hit); tests/helpers/exact.py:72-85 (read) |
| E16 | Open finding M2-MPS-INTERMITTENT (major): a CPU+MPS fleet "stalls … restart budget spent", with restarts in place from the same step. Under gpu false it is N/A. | agents-architetura/docs/findings.json:3-11 |
| E17 | Open finding M2-P1-CKPT-IDENTITY (major): `CKPT`/`RESULT` take rank and peer from the message, and `CKPT` skips the attempt check. ADR-0003 deferred it, and THREAT_MODEL T7 records it as a residual. | agents-architetura/docs/findings.json:138-146; docs/design/THREAT_MODEL.md:36 |
| E18 | The hardware switch is `{"gpu": false}`. | agents-architetura/tools/factory/hardware.json:2 (read) |
| E19 | LESSONS: keep a phase to about 3 work packages and 12 gate entries; use distinct expected values and a deliberately wrong implementation; a phase is configuration for runPhase. | agents-architetura/docs/LESSONS.md:7-17 |
| E20 | The skip-marker digest on this tree is `MARKERS_SHA=9841c322…bb6d`, the same as phase 1's pinned value. sha256 of `tests/chaos/test_recovery.py` = `e7ffb96f…28b1`; of `tests/helpers/cluster.py` = `92e20b25…1245`; of `tests/multiworker/test_fleet_state_exact.py` = `051822a5…7b2a`. sha256 of `tests/helpers/exact.py` with the `fleet_manifest` function's lines removed (python `ast`) = `EXACT_REST_SHA=606039e9…53d7`: C1 owns that file for BA-13 only, and `RECIPE`, `FEATURES_ON`, `FEATURES_OFF`, `RUN_PORTS`, `make_spec` and `reference_manifest*` decide what the oracle and identity rungs compare. | phase-1 `no-new-skip-markers` command, `shasum -a 256` and the `guarded-tests-unchanged` command (run for this document) |
| E22 | `tests/multiworker/test_fleet_state_exact.py::test_a_healed_fleet_is_byte_exact` kills slot (1, 0)'s worker at step 5 of a 2-stage, 12-step, `ckpt_every = 2` run and asserts `fleet_manifest(...) == reference_manifest_for(...)`: the sha256 of every weight, AdamW moment and step, the RNG state and every loss. The red chaos test compares only the loss sequence (`_bitwise`, test_recovery.py:50-53). The identity rung's corpus is `tests/oracle/**` and runs no heal. | tests/multiworker/test_fleet_state_exact.py:33-38, 61-79; tests/chaos/test_recovery.py:50-53; agents-architetura/tools/verify/identity.json:2 (read) |
| E21 | NEXT-m2-p1 planned phase 2 as the re-plan loop (SCHEDULER-I15) and phase 3 as plan mode plus SCHEDULER-I13. | docs/design/NEXT-m2-p1-liveness-telemetry.md:7-11 |

**Mechanism of the failure (facts E4-E9, the ordering derived from code; the timing is HYPOTHESIS H1).** A ship labels
step N with bytes from step N+k only if two things both happen. First, the agent's `open(pt)` comes after the stage's
`os.replace` for N+k. Second, the agent's second sidecar read comes before the stage writes the sidecar for N+k. So the
stage has to sit between `os.replace` and its sidecar write for at least one copy of the 7.6 MB shard (a few ms).
*H1:* this happens when the stage's main thread waits for the GIL behind its own reader and sender threads (switch
interval 5 ms) or is preempted on a loaded host. H1 stays unverified in this phase. The `writer-order` mutant against
test 1 shows only that the logical window exists and that test 1 catches it: test 1 forces the window with Events, so
the mutant is caught whatever the real cause of E4 was. It cannot show that the window occurs in real runs or that it
is the only cause. That is why the heal repeats are a regression check and a residual goes to `/wan-bughunt` (R1).

Once one wave is mislabeled, the failure is deterministic. `common_step` returns 3, and every restart in place restores
the same mislabeled wave. Rank 0 starts at step 4 and rank 1 at step 6, step-tagged frames never match, and the fleet
waits out three 90 s stall limits until the restart budget is spent (E2, E3).

# Why now

- **Chosen:** the red `chaos` rung. It comes first in precedence, and it breaks CONTROL-I10, the product rule's
  "through every failure, restart and recovery". In the worst interleaving, a mislabeled wave that still lines up
  would resume from the wrong state without anyone noticing. The rung can be fixed with a lossless file protocol that
  needs no design change.
- Re-plan from telemetry (SCHEDULER-I15, the human's hint, E21): a deferred invariant. That ranks below a red rung and
  an open recovery bug. A re-plan resumes under a new ownership through this same restore path, so it builds on the
  defect this phase removes.
- M2-P1-CKPT-IDENTITY (E17): an open major on the same `CKPT` message, deferred by ADR-0003. Fixing it changes the
  THREAT_MODEL T7 residual text, which lies outside the workstream paths, and it adds a third work package. It is
  recorded for phase 3 or a security phase.
- M2-MPS-INTERMITTENT (E16): its symptom matches (HYPOTHESIS: same cause). Under gpu false it is N/A. This phase
  removes the candidate cause, and the finding stays open until a gpu-true `/wan-bughunt`.
- M2-R5-SAFE-1/2, VER-* (factory findings): house machinery, kept out of product phases (LESSONS lesson 3).

**Why no ADR.** The phase changes how bytes move and are labeled. It does not change what is computed. The training
architecture, checkpoint schedule, recovery semantics (`common_step`, heal, restart in place) and wire protocol stay
as they are. The code is made to meet the contract that SPEC-runtime §Checkpoint and checkpoint.py already state (E12).

**Spec amendment (required before this phase runs).** The resume refusal is a new fatal condition. On the tree this
document was written on, SPEC-runtime §Checkpoint makes only a missing shard fatal at resume (SPEC-runtime.md:55), and
CONTROL-I8 is about the store's sha256 check (SPEC-control.md:156), not a label/content mismatch. Required text for the
Resume bullet of SPEC-runtime §Checkpoint: "Resume loads the slot's own shard. A missing shard, or a shard whose sidecar
is missing or names another step, is fatal." This is reported to the researcher as a spec conflict. The researcher
writes it into SPEC-runtime.md through the conflict protocol (recorded with `freeze.mjs` if the spec is frozen by then;
`docs/design/FROZEN.json` does not exist on this tree), and the human grants the amended `docs/design/SPEC-runtime.md`:
the phase lists it in `requires`, so the Gate refuses without a grant of its exact text. No writer implements the refusal
ahead of the spec: the precondition (Phases §1) fails closed with `GUARD=SPEC_NOT_AMENDED` while SPEC-runtime.md
(whitespace-normalized) lacks the required sentence above verbatim. It is not an ADR: it changes no
architecture, only which inconsistent input a resume refuses.

# Scope

Two workstreams, because the protocol has two ends. The writer (`StageRuntime._checkpoint` and `_resume`) is
**runtime**. The reader, restorer and protocol helpers (agent and `checkpoint.py`) are **control**. Neither end alone
can make a label true (proof below).

| Package | Writer | Owned paths |
|---|---|---|
| C1 (protocol, ship, restore) | control | `src/wantrain/fault_tolerance/checkpoint.py`, `src/wantrain/worker/agent.py`, `tests/unit/test_fault_tolerance.py`, `tests/helpers/exact.py` |
| R1 (writer, resume check) | runtime | `src/wantrain/runtime/stage.py`, `tests/unit/test_runtime_spec.py` |

Not owned and not to change: `tests/chaos/test_recovery.py`, `tests/helpers/cluster.py`,
`tests/multiworker/test_fleet_state_exact.py` (pinned by sha256, E20), every line of `tests/helpers/exact.py` outside
`fleet_manifest` (pinned by sha256, E20), the
coordinator, liveness, transport, THREAT_MODEL and the specs.

## Mechanism (the contract writers implement)

New functions in `src/wantrain/fault_tolerance/checkpoint.py` (C1). They import only the stdlib, and every call below
goes through these module-level names:

```python
def sidecar_of(shard_path: str) -> str            # shard_path[:-3] + ".step.json"  (same file as today)
def invalidate_sidecar(side: str) -> None         # os.unlink; FileNotFoundError ignored
def write_sidecar(side: str, step: int) -> None   # write side + ".tmp" with {"step": step}, then os.replace onto side
def save_shard(path: str, step: int, save_fn) -> None
    # save_fn(path + ".tmp"); invalidate_sidecar(sidecar_of(path)); os.replace(path + ".tmp", path);
    # write_sidecar(sidecar_of(path), step)   -- in exactly this order
def check_resume_step(side: str, shard_step: int) -> None
    # read_sidecar(side) is None -> RuntimeError naming the missing sidecar;
    # != shard_step -> RuntimeError "... sidecar names step X, the shard holds step Y: mislabeled wave"
```

- **R1 writer.** `StageRuntime._checkpoint` builds the same dict as today and calls
  `save_shard(path, step, lambda tmp: torch.save(payload, tmp))`. The `.pt` bytes do not change. It does not inline its
  own order: test 6 and the `stage-bypass` mutant check that the stage's write reaches `checkpoint.invalidate_sidecar`.
- **R1 resume.** Right after `torch.load`, before any `load_state_dict`, `set_rng_state` or loss restore, `_resume`
  calls `check_resume_step(sidecar_of(path), int(state["step"]))` through the name the stage module imports. A missing
  or disagreeing sidecar is fatal, like a missing shard.
- **C1 restore.** `_on_restore` parses `step = int(msg["step"])` first, so a malformed step raises before any file
  changes, and verifies the sha256 (unchanged). Then it runs `invalidate_sidecar`, then `os.replace(path, dst)`, then
  `write_sidecar(sidecar_of(dst), step)`. It calls the last one as `checkpoint.write_sidecar` or as a name imported into
  `agent`; the mutant patches both.
- **C1 ship.** The whole body of `_ship_checkpoint` runs under `self._ship_lock` (a `threading.Lock` set in
  `__init__`), so the snapshot file and `_shipped` have one user at a time. The before/after sidecar check stays.
- **C1 helpers.** `export_wave` writes each copied shard's sidecar with `write_sidecar`. `fleet_manifest` skips a shard
  without a sidecar: a stage killed between `invalidate_sidecar` and `write_sidecar` leaves one on the victim's disk.

**Proof that a shipped label is true.** Writer W(N) runs: save tmp; unlink sidecar; replace pt with N; write sidecar N.
Steps strictly increase within an attempt. A restore runs only while no stage runs, and it follows the same order.
Suppose the reader reads `b` at t1 and `b` again at t3, and opens pt at t2 (t1 < t2 < t3). Sidecar = `b` at t1 means
`write(b)` completed and no unlink has followed, so pt = wave `b` at t1. If pt changed by t2, some replace happened in
(t1, t2), and an unlink came before it in that same interval. At t3 the sidecar is then absent or names a step
greater than `b`. Either way it differs from `b`, and the ship is dropped. An open descriptor keeps the inode it
opened, so later replaces cannot change the bytes being copied. The lock serializes copy and send, so the bytes sent
are the bytes copied. A restore of wave W writes sidecar W next to the wave-W bytes, so the next ship's label is true
too. ∎

**Cost model.** T_step is unchanged. Only per-checkpoint metadata changes. Before: 1 `rename` plus 1 in-place 13-byte
write. After: 1 `unlink` plus 1 write to a temp file plus 2 `rename`s. HYPOTHESIS H2: the added cost is ≤ 1000 µs per
checkpoint. Against the toy's predicted 83 ms step with `ckpt_every = 2` (SIMULATED), that is ≤ 0.6 % of the step;
for real shards, `torch.save` dominates. Measured by `protocol-cost`. HYPOTHESIS: the ship lock delays an
`ABORT_ACK` by at most one shard copy plus one sha256 pass (a few ms), far below the coordinator's 60 s ABORT timeout
(coordinator.py:1202). `Channel.send_file` already holds `_send_lock` for the send (control.py:76-86), so an ack
already waited behind an in-flight send; the copy into the snapshot and `send_file`'s hashing read (control.py:69-73)
run outside `_send_lock` and now sit under `_ship_lock`. Resume adds one sidecar read. Recovery after a mislabel goes from about 3 × 90
s of stalls followed by a lost run to zero mislabels by construction. A residual mislabel now fails the run at once,
with the cause named (E14).

# Invariants

- **Delivers:** CONTROL-I10 (a dead worker heals bit-identically: the red test for the losses, `heal-state-exact-loaded`
  for the sha256 of weights, optimizer state, RNG state and losses, E22) and RUNTIME-I3 (checkpoint then resume
  continues bit for bit). The resume refusal is the required SPEC-runtime amendment above, not CONTROL-I8.
- **Keeps:** CONTROL-I8 (unchanged), RUNTIME-I1, RUNTIME-I2 and RUNTIME-I4 (no tensor, RNG draw or fold changes; the `.pt` dict is unchanged).
  It also keeps CONTROL-I17 and CONTROL-I18 (liveness and intake untouched) and every oracle and identity case,
  including `resumed`.

# Binding audit

Every piece of state the change reads or rewrites, and every RNG draw or reduction near it.

| BA | What | Where | Ruling |
|---|---|---|---|
| BA-1 | Shard replace happens before any sidecar change | stage.py:731 | **Fix** (R1): `save_shard` unlinks the sidecar before the replace |
| BA-2 | Sidecar written in place (`open "w"`); a reader can see an empty or partial file | stage.py:735-736 | **Fix** (R1 via C1 `write_sidecar`): temp file plus `os.replace` |
| BA-3 | Ship's before/after sidecar check | agent.py:607-623 | **Keep**: sound once BA-1 is fixed (proof) |
| BA-4 | Two shippers (`_supervise`, `_on_abort`) share `ship-r{r}p{p}.pt` without a lock | agent.py:538-543, 613, 641-652 | **Fix** (C1): `_ship_lock` around the whole body |
| BA-5 | `send_file` hashes, then rereads, the file | control.py:66-90 | **Keep**: the snapshot is private under BA-4's lock, and a size change already raises |
| BA-6 | `_shipped` key `(run, rank, peer, step)` has no attempt | agent.py:181, 610-612, 637 | **Keep**: a replayed attempt recomputes wave N with the same bytes (product rule), and the store keeps the first wave whose label is true |
| BA-7 | Restore leaves a stale sidecar, or none (E5) | agent.py:654-672 | **Fix** (C1): invalidate, replace, write sidecar = restored step |
| BA-8 | Coordinator files `CKPT` under the message's label; a duplicate label is unlinked | coordinator.py:388-397 | **Keep**: correct once labels are true. Message identity is M2-P1-CKPT-IDENTITY, deferred (Why now) |
| BA-9 | Store verifies the sender's sha256 only | checkpoint.py:89-104 | **Keep**: the sha binds the transport; the label is bound by BA-1, BA-4, BA-7 and BA-11 |
| BA-10 | `common_step` resume point | checkpoint.py:106-113; coordinator.py:1298, 1329 | **Keep** |
| BA-11 | Resume trusts `state["step"]` alone | stage.py:752-761 | **Fix** (R1): `check_resume_step` before any state is applied |
| BA-12 | `export_wave` copies shards without sidecars (no caller) | checkpoint.py:132-140 | **Fix** (C1): writes each copied shard's sidecar naming the exported step, so a future caller passes BA-11; test 7 is its evidence |
| BA-13 | `fleet_manifest` opens every sidecar unconditionally | tests/helpers/exact.py:72-85 | **Fix** (C1): skip a shard with no sidecar; the "every slot reached the last step" assertion stays. A skip cannot hide a slot: `distributed_manifest` opens every (rank, peer) shard of the spec from the fresh `out_dir` and fails loudly on a missing one (src/wantrain/validation/state.py:111-117). The skip branch has no deterministic test in this phase (recorded); its happy path runs in `heal-state-exact-loaded` and `rung:multiworker` |
| BA-14 | RNG state saved and restored in the shard | stage.py:726, 756-757 | **Keep**, byte-identical: the payload dict and its order do not change; the check runs before `set_rng_state` |
| BA-15 | Loss history `s <= step` in the shard, restored on resume | stage.py:727, 758-760 | **Keep**, unchanged |
| BA-16 | Final unconditional checkpoint and the post-exit ship | stage.py:1088-1090; agent.py:542-543 | **Keep**: same protocol through `save_shard` and the lock |
| BA-17 | A mislabeled resume showed up as an S4 stall | liveness.py:233-276 | **Keep** the judge: a mismatch is now a stage exception, so it takes the `failed` path (E14) |
| BA-18 | Oracle `resumed` case and the integration resume test resume from their own shard and sidecar | tests/oracle/test_state_identity.py:55-61; tests/integration/test_engine_bitwise.py:80-98 | **Keep**: the part run writes the sidecar through `save_shard`, so the check passes (gates: `rung:oracle`, `rung:identity`, `rung:integration`) |

No reduction order changes, and no fold or RNG draw moves. The only bytes written differently are sidecar files and
the ship snapshot.

# Tripwire

`check_resume_step` is the tripwire, and R1 arms it in this phase. With only part of the change in, the failure is
loud:
- R1 without C1's restore: every heal fails at resume with "no sidecar".
- C1 without R1's writer: a race mislabel fails the run at resume with "mislabeled wave" instead of stalling.

`resume-binding-unit` proves the tripwire fires. The `resume-trust` mutant proves the test catches a bypass.

# Exit gate

Ladder rungs ×2 consecutive: factory, build, syntax, fmt, lint, types, discipline, invariants, claims, unit,
integration, multiworker, oracle, identity, chaos, wan, full, static, security, fuzz. Pinned entries:

| Id | What | Bound |
|---|---|---|
| `guarded-tests-unchanged` | sha256 of `tests/chaos/test_recovery.py`, `tests/helpers/cluster.py` and `tests/multiworker/test_fleet_state_exact.py`; sha256 of `tests/helpers/exact.py` outside `fleet_manifest` = `606039e9…` (E20); skip-marker digest = `9841c322…` | exact |
| `ship-protocol-unit` | 4 C1 tests (1-3 and 7 below) | 20/20 passed |
| `resume-binding-unit` | 3 R1 tests (4-6 below) | 20/20 passed |
| `would-fail-mutants` | `writer-order` (vs test 1), `ship-unlocked` (vs 2), `restore-stale` (vs 3), `resume-trust` (vs 4), `stage-bypass` (vs 6) | `MUTANTS_SURVIVING == 0` |
| `chaos-dead-worker-loaded` | the red test, under 2 CPU-burning loaders | 20/20 passed |
| `chaos-dead-worker-plain` | the red test, alone | 10/10 passed |
| `heal-state-exact-loaded` | `test_a_healed_fleet_is_byte_exact` (E22), under 2 CPU-burning loaders | 10/10 passed |
| `protocol-cost` | arms: old write order vs `save_shard`, same 1 MiB body, 300 interleaved runs, medians | `PROTO_ADDED_US <= 1000` (SIMULATED) |
| `h1-never-seen` | records the previous verify's H1-incomplete failures (from the pinned report `.factory/gates/m2-p2-wave-label-binding.json`, when newer than the precondition's stamp) into the H1 marker `.factory/runs/m2-p2-wave-label-binding/h1-seen.log`, then requires that the marker does not exist (Phases §5) | `H1_SEEN=0` |

Tests the writers add (each with `covers:` and `WOULD-FAIL-IF:`). Use plain `assert` only, no `pytest.raises`, so a
surviving mutant fails with an AssertionError. The skip-marker digest of `guarded-tests-unchanged` covers every line of
`tests/`: add no line its grep matches. No `from unittest ...`, `from pytest import ...`, `from _pytest ...`,
`import pytest as` or `import unittest as`, and no skip, xfail, exit or marker; use the `monkeypatch` and `tmp_path`
fixtures, `threading.Event` and a plain `import pytest` only:
1. `tests/unit/test_fault_tolerance.py::test_ship_never_labels_a_newer_shard_with_an_older_step` (covers
   CONTROL-I10). The writer is the production `save_shard` in a thread, with `checkpoint.write_sidecar` held on an
   Event. Wave 3 is in place, with distinct bytes per wave. The agent's real `_ship_checkpoint` reads `before`; the
   writer then runs to the held point (pt replaced by wave 5) and the copy proceeds. The test asserts that every
   captured `CKPT` body has the bytes of the wave its `step` names. A fake channel reads the file at send time.
2. `…::test_concurrent_ships_never_cross_waves` (covers CONTROL-I10). Ship 1 (wave 3) blocks inside the fake
   `send_file` on an Event. `save_shard` saves wave 5, and ship 2 starts in a second thread. The test joins ship 2 with
   a 2 s timeout, then releases ship 1. It asserts the same label-equals-bytes property for every send.
3. `…::test_restore_rewrites_the_sidecar` (covers CONTROL-I10). Start with a stale sidecar naming 5, then
   `_on_restore` of wave 3. Assert that the sidecar reads 3 and the shard has the bytes of wave 3.
4. `tests/unit/test_runtime_spec.py::test_resume_refuses_a_shard_its_sidecar_does_not_name` (covers RUNTIME-I3). A
   real loadable shard (step 5, from a tiny `nn.Linear` chunk and its AdamW), a sidecar naming 3, and
   `StageRuntime._resume` on a `StageRuntime.__new__` object with only the attributes `_resume` reads. Assert that
   `RuntimeError` names the sidecar and that no parameter changed. Without the check, `_resume` returns 6 and the
   assertion fails.
   Tests 1 and 2 must also catch an agent that never ships, where the property "every send is labeled truly" holds
   because nothing was sent (LESSONS lesson 1). Test 1 releases the writer, ships again and asserts that exactly one
   send is labeled 5 with the bytes of wave 5. Test 2 asserts that both labels 3 and 5 were sent, each with its own
   bytes.
5. `…::test_resume_refuses_a_shard_without_a_sidecar` (covers RUNTIME-I3). Same setup with no sidecar file.
6. `…::test_checkpoint_writes_through_the_seqlock` (covers RUNTIME-I3, CONTROL-I10). `StageRuntime._checkpoint(5)` on a
   `StageRuntime.__new__` object with only the attributes `_checkpoint` reads (a tiny `nn.Linear` chunk and its AdamW),
   over a wave-3 shard and a sidecar naming 3. `os.replace` is wrapped on the `os` module (monkeypatch) to record each
   destination and, at the shard's replace, whether the sidecar file exists. Assert that the destinations are exactly
   [shard, sidecar] in that order, that the sidecar did not exist at the shard's replace, that the sidecar then reads 5,
   and that the loaded shard has step 5 and keys exactly `["step", "chunks", "opt", "rng", "losses"]` in that order.
   Today's writer fails it (the sidecar exists at the replace and is written in place), and so does the `stage-bypass`
   mutant (`checkpoint.invalidate_sidecar` a no-op), so the production writer, not only the helper, is under a
   deterministic gate.
7. `tests/unit/test_fault_tolerance.py::test_export_wave_writes_each_sidecar` (covers CONTROL-I10). A real
   `CheckpointStore` holding wave 5 for two slots, with distinct bytes per slot (and a wave 3 with other bytes). After
   `export_wave(5, ...)`, assert that each returned path has the bytes of its slot's wave 5, that its sidecar reads 5
   and that `check_resume_step(sidecar_of(path), 5)` returns. Today's `export_wave` fails it (no sidecar), and so does
   one that writes another step.

# Phases

1. **Gate.** `checkApprovals` on this NEXT, the GATES file, the built workflow, `docs/design/DESIGN.md`,
   `docs/design/adr/ADR-0003-m2-p1-progress-monotone.md` (the ADR that defers M2-P1-CKPT-IDENTITY, which BA-8 and
   the restore-boundary lens rely on) and the amended `docs/design/SPEC-runtime.md` (§Mechanism, Spec amendment). Then the precondition: it first requires the SPEC-runtime amendment (§Mechanism)
   and stops with `GUARD=SPEC_NOT_AMENDED` without it, before it writes anything; then it removes the H1 marker, writes
   the stamp `.factory/runs/m2-p2-wave-label-binding/phase-start` (Phases §5) and runs `guarded-tests-unchanged`. It
   stops for a human on a mismatch or on a check that could not run (each case tagged `GUARD=MISMATCH`,
   `GUARD=GREP_ERROR` or `GUARD=GUARD_ERROR`).
2. **Wave 1 — C1 (writer C).** The functions in §Mechanism, the ship lock, restore, `export_wave`, `fleet_manifest`, and
   tests 1-3 and 7.
3. **Wave 2 — R1 (writer R), depends on C1.** `_checkpoint` through `save_shard`; `check_resume_step` in `_resume`;
   tests 4-6.
4. **Review (runPhase `reviewUntilClean`, ≤ 3 rounds) over every owned path, with the lenses written for this
   objective:**
   (a) *protocol proof.* Walk every writer and reader of a shard or sidecar (BA-1..BA-18). Can any path label bytes
   with a step they do not hold, including kill points between the unlink and the write, restore during a live ship,
   the final wave and restart in place?
   (b) *byte-exact.* Are the `.pt` payload and order unchanged? Does the check come before any state is applied? Does
   the oracle `resumed` case still pass? Are `RECIPE`, `FEATURES_ON`, `FEATURES_OFF`, `RUN_PORTS`, `make_spec` and
   `reference_manifest*` in `tests/helpers/exact.py` byte-unchanged?
   (c) *tests that fail on a wrong implementation.* Is each of the five mutants caught by a named assertion? Are
   expected values distinct per wave (LESSONS lesson 1)? Is there no sleep-only synchronization?
   (d) *loud failure.* Does a mislabel or a missing sidecar end the run at once with its cause, never as a stall? Does
   a victim's half-saved shard break no helper?
   (e) *cost.* What do the metadata operations cost against the step? Is the wait the lock adds before an
   `ABORT_ACK` bounded by one shard copy plus one sha256 pass beyond the send wait `_send_lock` already imposed, far
   below the 60 s ABORT timeout, with nothing else held under it?
5. **Verify and stabilize (runPhase).** `node agents-architetura/tools/gates/pinned.mjs m2-p2-wave-label-binding --json`,
   ≤ 4 stabilize rounds. Stabilize fixes failures of the deterministic entries and rungs. The two heal tests
   (`test_dead_worker_is_replaced_and_the_run_stays_exact` and `test_a_healed_fleet_is_byte_exact`) also run inside
   `rung:chaos`, `rung:multiworker` and `rung:full`, each twice in a row, and the last failure was intermittent (E13).
   Every other test in `tests/chaos/test_recovery.py` and `tests/multiworker/test_fleet_state_exact.py` (killed stage,
   restart in place, wide-stage peer death, control-plane blip) goes through the same ship, store, `RESTORE` and resume
   path, so the rule covers them too. "mislabeled wave" is the tripwire's text (`check_resume_step`, owned code): it
   says a mislabel was refused, not why, so it marks H1 only when it comes back after a passing repetition. A failure
   is **H1 incomplete** when:
   - `chaos-dead-worker-loaded`, `chaos-dead-worker-plain` or `heal-state-exact-loaded` shows the stall signature
     ("fleet cannot progress" or "fleet stalled", a mislabel that went unrefused), is killed by its timeout (outcome
     `hang`), or fails after at least one repetition passed (intermittent on the same tree);
   - `rung:chaos`, `rung:multiworker` or `rung:full` fails (status `fail`; a rung skipped behind a failing blocking
     rung, `n/a` or not run ran no heal test and is never H1) and shows the stall signature or a timeout; or, when no heal entry of
     the same verify failed deterministically, `rung:chaos` or `rung:multiworker` fails at all (each directory holds
     only its heal file), or `rung:full` fails naming a heal file, or `rung:full` fails and neither its kept output
     (cut to 600 characters, pinned.mjs:112) shows a complete failure list naming only other files nor another test
     rung (unit, integration, oracle, identity, wan) of the same verify failed. The rung rule is conservative and
     prefers a false NOT VERIFIED to a quiet pass.

   A heal entry fails **deterministically** when it fails on its first repetition (`0/N passed, then fail`) without
   the stall signature, such as a traceback inside owned code (`fleet_manifest`, the agent's ship or restore path, the
   stage's `_checkpoint` or `_resume`) or the tripwire's "mislabeled wave" (the entry prints `TRIPWIRE=mislabeled
   wave`). That failure, and the heal-file rung failures of the same verify, are normal items scoped to the owned
   file the traceback points to: the tripwire catching our own defect (Tripwire) is fixed in this phase, never sent to
   `/wan-bughunt`, and `h1-never-seen` never records it in the marker. If the race is still there after the fix, it comes back as a stall, a hang or an intermittent
   failure and is H1 then.

   The stop holds across rounds and trees. runPhase alone would not keep it: its intermittent check compares rounds on
   the same tree only, so a stabilize fix of another item would let a lucky rerun pass. The record is the marker
   `.factory/runs/m2-p2-wave-label-binding/h1-seen.log`, which is gitignored and so outside the tree id and every scope
   audit. It is written mechanically; no agent writes it, and the verifier stays read-only (its role and runPhase's
   verify prompt forbid edits). The three heal entries append to it themselves when their output carries the stall signature
   (they print `H1_SIGNATURE=...`). Every other case (a rung failure, a heal entry killed by its timeout, which never
   reaches its own grep, an intermittent heal entry) is recorded by `h1-never-seen` of the next verify: it reads the
   report pinned.mjs wrote at the end of the previous verify (`.factory/gates/m2-p2-wave-label-binding.json`, pinned.mjs:192-193),
   used only when newer than the precondition's stamp, and appends one line per H1-incomplete failure. The round
   where the failure happened is already red, so the one-round lag loses nothing. `h1-never-seen`, the last pinned
   entry, fails while the marker exists, so no later round, on any tree, can be VERIFIED. The precondition removes the
   marker and writes the stamp once, at the start of each phase run. The verifier classifies each failure by the rule
   above itself (the marker may not exist yet in the round the failure happens), files each deterministic heal failure
   as a normal item with its owned scope, and files all H1-incomplete failures,
   `h1-never-seen`'s included, as one item whose mechanism reads "H1 incomplete: route to /wan-bughunt" with an EMPTY
   scope. runPhase drops an item with an empty scope (`normalizeItems`), so Stabilize changes no file for it. Other
   items are still fixed, and the phase then ends NOT VERIFIED, with `pinned:h1-never-seen` failing and `/wan-bughunt`
   in `next`; the hunt starts from this document's E-table. Do not retune timeouts. House item, recorded and not in
   this phase's paths: a runPhase option for failures that stick across trees (keyed by test id), with `history` in its
   result, would replace the marker; pinned.mjs keeping a rung's full failing-test list would make the rung rule exact.

# Risks

- **R1.** A cause besides this race also mislabels or stalls the run. Mitigation: 30 chaos repeats (20 loaded and 10
  plain) plus 10 loaded `heal-state-exact-loaded` runs; any H1-incomplete failure (stall, hang or intermittent) in any round sticks through
  `h1-never-seen` and goes to `/wan-bughunt` (Phase 5). **The repeats are a regression check, not a statistical
  closure.** The pre-fix rate is unknown: the test passed 2 of 2 at the phase-1 closeout and then failed once (E13),
  roughly 1 in 3 to 1 in 5. By the rule of three, 30 clean chaos runs bound the post-fix rate only below about 10 %
  at 95 % confidence, about 2x below that rate, not the 10x `/wan-bughunt` demands. A residual cause at 5 % passes all
  30 runs with probability 0.95^30 ≈ 21 %. The causal evidence is deterministic: the `writer-order` mutant against
  test 1 for the protocol, and the `stage-bypass` mutant against test 6 for the stage that had the defect (E6). These
  show the tests catch this window; they do not show it is the only cause. No causal A/B arm (pre-fix files restored,
  the failure must come back) is pinned: at a pre-fix rate of 20-33 %, 10 arm-A runs would miss the bug 2-11 % of the
  time, so the arm would itself be flaky. If a residual cause shows up later, `/wan-bughunt` closes it statistically
  and causally.
- **R2.** The strict missing-sidecar rule breaks an unlisted resume path. Every place that turns resume on was found
  by `grep -rnE 'resume *= *True|"resume": *True|\.resume\b *=' src tests` plus `patch_spec_for_resume`'s callers:
  the coordinator (checkpoint.py:152 via coordinator.py:1266), tests/oracle/test_state_identity.py:60 and
  tests/integration/test_engine_bitwise.py:92, all in E15. The oracle, identity and integration rungs run the last two.
- **R3.** A stage killed between `invalidate_sidecar` and `write_sidecar` leaves its newest wave unshippable. That wave
  was not yet in the store, so the resume point is unchanged.
- **R4.** M2-MPS-INTERMITTENT may share this cause (HYPOTHESIS). This cannot be verified under gpu false, and the
  finding stays open.
- **R5.** M2-P1-CKPT-IDENTITY stays open: a malicious admitted worker can still ship another slot's wave (THREAT_MODEL
  T7 residual).
- **R6.** The temporary-dir evidence (`$D`) will be rotated by pytest. E3-E5 copy the decisive hashes and lines.
