# SPEC-control: coordinator, agents, discovery and fault tolerance

**Workstream:** `control`. **Owned paths:**
- `src/wantrain/coordinator/`, `src/wantrain/worker/agent.py`, `src/wantrain/fault_tolerance/`,
  `src/wantrain/resources/`, `src/wantrain/cli.py`, `src/wantrain/__main__.py`;
- `tests/integration/test_coordinator.py`, `tests/unit/test_fault_tolerance.py`, `tests/unit/test_resources.py`,
  `tests/unit/test_audit.py`, `tests/unit/test_control_admission.py`;
- `tests/helpers/`, `examples/`.

The chaos suite (`tests/chaos/`) is harness-owned, but it verifies this spec's recovery invariants.

## 1. Scope

The control plane runs a job from a fleet of registered machines to a completed, verified loss sequence. It never
touches numerics, with one exception: for `arch="hf"` it builds the run's buffer record once, at the first launch
(§2.1 step 4, ADR-0009).

## 2. Domain model (normative)

### 2.1 Run lifecycle (`Coordinator.run_job`)

1. **Discovery.** Agents connect, run the control handshake against the pinned coordinator id, answer `CHALLENGE`
   inside the sealed channel with `token_mac(token, transcript)`, and send `HELLO` with measured capabilities
   (`resources/probe.measure`: matmul TFLOPs, memory, device, compute capability, and (CONTROL-I25) `device_class`).
   The declared region and uplink are marked as declared.
2. **Selection** (`scheduler.select_nodes`). The rest of the pool are spares. Every arch: the selected nodes hold one
   device class, else nothing launches (`MIXED-FLEET`, CONTROL-I25).
3. **Routing.**
   1. Probe matrix (`PROBE`).
   2. `best_chain`.
   3. Full bidirectional measurement per hop (`MEASURE`, duplex class).
   4. Re-route if the measurement disagrees with the probe.
4. **Planning.** (M4, ADR-0009) For `arch="hf"`, first `GATE`: every selected node and spare runs `hfm.gate` on its
   CPU and on each of its slots' devices that is not the CPU, and returns the reports and `hfm.fingerprint`; the run is refused before placement unless every report passed
   and every fingerprint is admitted (CONTROL-I24). At the first launch only, the coordinator builds the run's buffer
   record in a child process on its own CPU, under SPEC-model §2's environment guard and with an `hfm.fingerprint`
   admitted as a node's (else nothing launches), records that fingerprint with the record, pins the record in the
   JobSpec (RUNTIME-I19) and sends it with `GATE`. A resume, heal, restart or rebalance never rebuilds it.
   - Recipe mode: `placement.place`.
   - Plan mode: `planner.plan_job`, then an on-device `FIT` (measured peak under the slot's schedule), with a re-plan on
     failure.
5. **Profiling.** `PROFILE`: every slot times F, B and W on its own device (`resources/profile.profile_stage`, median of
   `profile_reps`).
6. **Cut refinement** (`_refine_partition`). Re-cut by measured stage throughput when the predicted slowest stage
   improves by more than 5%.
7. **Zero-bubble solve** (`scheduler.schedule_for_spec`): from the profiles, the measured hops and each stage's
   activation budget. The rationale is logged: costs, comm, predicted step time, bubble, peer weights.
8. **Launch.** `ASSIGN` the same spec to every slot, last rank first.
9. **Monitor.**
   - Inputs: `HEARTBEAT`, `PROGRESS` (from `WT_STEP`), `TELEMETRY`, `CKPT` waves (sha256-verified into
     `CheckpointStore`), `EXIT`.
   - `PROGRESS` carries `src` (`step` from `WT_STEP`, `metric` from `WT_METRIC`). `TELEM_STEP`: the live per-step
     summary parsed from `WT_TELEM`, validated at the coordinator (CONTROL-I18). `TELEMETRY` stays the telemetry file
     shipped at exit.
   - The liveness judge runs every `poll_s`.
10. **Complete.** Collect the last stage's `RESULT` (losses in hex) and write `losses.json`, then `RELEASE` the workers.

### 2.2 Liveness judge (`LivenessJudge.judge`)

Verdicts are: `ok`, `suspect`, `convict(slot)`, `stalled`, `failed` and `done`.

The judge checks, in this order:

1. **Done** when every launched slot exited 0.
2. **Hard failures**, which are facts, not suspicions:
   - S2: the stage process is gone without its done marker;
   - S1: the control connection has been lost for longer than `reconnect_grace_s`.

   Exactly one is a **convict**. More than one at once is **failed**: a network or coordinator problem. Replacing
   hardware would not fix it.
3. **Suspect** while any slot is disconnected inside the grace. Nothing else is judged then.
4. **Failed** when a slot exited non-zero *with* its done marker and without "peer lost". The run raised, so a
   replacement would hit it again.
5. **Look closer** when a slot lost its peer (S3: the silent slots are suspects) or there has been no optimizer step
   fleet-wide for `max(stall_floor_s, stall_steps × median step)` (S4: every slot is a suspect). The limit before the
   first step is `first_step_timeout_s`.
   - Among the suspects, exactly one unreachable slot earns a strike; a second strike `strike_gap_s` later convicts it.
   - Several unreachable slots are a suspect: wait.
   - If every suspect is alive and heartbeating, the run is **stalled**: the data plane is broken, so restart in place.

   S4 judges only while some launched slot still owes an optimizer step: it reads only progress parsed from `WT_STEP`
   whose step lies in [the attempt's start step, `steps − 1`] and is greater than the last step it recorded for that
   slot; replayed `WT_METRIC` lines never feed it. Once every
   launched slot exited 0 or reported step `steps − 1` (or the attempt started at `steps`), the attempt is in shutdown:
   quiet is bounded by `shutdown_limit_s` (default 600), past which the verdict is failed with 'shutdown exceeded …
   after the final step'. S1–S3 and strikes still apply in shutdown.
6. Ordinary heartbeat staleness earns strikes even without a stall.

Liveness race fix: an agent reports `stage_alive` until its `EXIT` is sent, and the judge allows
`exit_report_grace_s`.

### 2.3 Recovery

- **Heal** (on convict, within `max_heals`):
  1. `ABORT` the survivors; each ships its last wave.
  2. `find_replacement`: identical model first, then the dead node's region; respect the recipe's compute dtype and the
     dead list; distinct hosts. (M2b, ADR-0006) For a slot of a sharded stage (`shard: "pieces"`), only the identical
     device model (`match="model"`); without one there is no spare and the heal fails cleanly with the reason.
     (M4, ADR-0009) For `arch="hf"`, only a node whose gate reports the slot's recorded device class, and only if the
     map the heal builds passes decision 8's re-check (CONTROL-I24), with the same failure when there is none.
     (CONTROL-I25) For every arch, only a node of the run's device class; with none the heal fails cleanly
     (`MIXED-FLEET`) and nothing relaunches.
  3. Take the newest step that every slot holds a wave for (`common_step`).
  4. `RESTORE` the wave (files and manifest, each hash checked) to every slot.
  5. Relaunch with `resume=true` under the original spec, with `resume_step` set to that wave (`patch_spec_for_resume`
     changes only endpoints, devices, slot keys and `resume_step`, and for a rebalance the ownership).
- **Restart in place** (on stalled, within `max_restarts`): the same resume on the same nodes.
- **Lost work bound:** at most `ckpt_every` steps plus the restore time.
- **Wave retention (`commit_wave`, `commit_received_wave`; agent slots and the store alike).** After a commit of step n,
  the slot keeps the newest `keep` committed waves whose step is ≤ n and removes the older ones of those. Waves newer
  than n are never removed by a commit: they belong to an abandoned timeline (a RESTORE or a replay below the slot's
  newest wave) and only `drop_waves_after` removes them (BA-6, at the relaunch's resume, before its first step). So the
  wave just committed is never removed, and in the ascending case (n is the newest) this is exactly "keep the newest
  `keep`". Example: RESTORE of step 2 into a slot holding 4, 5, 6 leaves 2, 4, 5, 6; the resume from 2 then leaves 2.
  A commit of a step below the slot's newest wave is not refused for being below it. A step already committed with an
  equal manifest is kept untouched, and one with another manifest raises, as before.
- **Wave cap (`MAX_WAVES_AHEAD` = 4; both commit functions).** A slot never holds more than `keep + MAX_WAVES_AHEAD`
  step directories. Before writing anything for a step the slot does not already hold, the commit raises `ValueError`
  naming the slot, the step and the count when the slot holds `MAX_WAVES_AHEAD` or more step directories above the
  step, or `keep + MAX_WAVES_AHEAD` or more in total (step directories counted by name, as the pruning lists them).
  The check comes after the held-step check, so a re-commit of a held step is never refused by the cap. A refusal
  removes nothing and leaves the slot as it was. Why: the step of a received wave comes from the sending peer's
  manifest, and workers may be adversarial (THREAT_MODEL); without the cap a peer shipping descending steps gets every
  one committed and kept on the coordinator. Honest flows stay inside it: the ascending case holds at most `keep`, and
  a RESTORE below the slot's newest waves adds one wave under at most `keep` newer ones (2 into 4, 5, 6). The caller
  handles a refusal like any malformed input: the coordinator logs and drops the CKPT and unlinks its staged parts
  (CONTROL-I18, BA-43); the agent answers RESTORE_ACK ok=false naming the reason. An honest flow that meets the cap is
  a spec conflict to report, never a reason to raise the cap.

### 2.4 Agent (`WorkerAgent`)

- Registers, then sends heartbeats every few seconds.
- Serves `PROBE`/`MEASURE` with echo servers, `PROFILE`/`FIT` as subprocesses, `ASSIGN` by launching `stage_main` under
  supervision, and `ABORT`, `RESTORE`, `RELEASE` and `SHUTDOWN`.
- Parses `WT_*` markers into `PROGRESS`/`TELEMETRY`, and ships every committed wave in ascending step order, woken by
  `WT_WAVE` and by a backstop scan, as its files (`CKPT_PART`) followed by its manifest (`CKPT`); it never reads the
  alias.
  `PROGRESS` carries `src` (`step` from `WT_STEP`, `metric` from `WT_METRIC`). `TELEM_STEP`: the live per-step summary
  parsed from `WT_TELEM`, validated at the coordinator (CONTROL-I18). `TELEMETRY` stays the telemetry file shipped at
  exit.
- Reconnects with backoff, keeping its stage alive within the grace.

### 2.5 Audit policy (library; `fault_tolerance/audit.py`)

Verify-before-fold against malicious workers:
- newcomers are always replicated;
- reputation decays to a floor, never to zero;
- one mismatch is a ban;
- the sortition is seeded, so the audited set cannot be predicted without the seed;
- the evidence is recorded.

It is not wired into the runtime yet.

## 3. Interfaces

**Provides**
- `Coordinator(cfg).start() / stop() / wait_for_workers(n) / run_job(job) -> RunResult`, `CoordinatorConfig`, `Job`,
  `RunResult{ok, reason, losses, run_dir, attempts, heals, restarts, events}`.
- `WorkerAgent(...).run()`.
- `LivenessJudge(...).judge(states) -> Verdict`, `CheckpointStore`, `find_replacement`, `AuditPolicy`.
- `patch_spec_for_resume(spec_json, endpoints, devices, slot_peers=None, resume_step=None, ownership=None)`.
- `probe.measure`, `links.measure_pair/measure_row/classify_duplex`, `profile.profile_stage`, `fit.measure`, `NodeInfo`.
- CLI: `wantrain coordinator | worker | stage | local | plan | compare | validate-model | prepare-data`.

**Consumes**
- scheduler: `select_nodes`, `best_chain`, `place`, `plan_job`, `schedule_for_spec`, `LinkMemory`.
- runtime: `JobSpec`, `stage_main`.
- transport: `control_dial`, `control_accept`, `Channel`, `token_mac`, `session_mac`.

## 4. Invariants

- **CONTROL-I1.** Node lifecycle: register → available → assigned → run → progress and exit → complete → idle again. A
  coordinated two-stage run's losses equal the oracle bit for bit.
- **CONTROL-I2.** Layers are cut in proportion to declared heterogeneous capacity.
- **CONTROL-I3.** WAN-aware routing crosses the slow inter-region link exactly once. The schedule is solved from
  profiled costs and measured hop communication.
- **CONTROL-I4.** Plan mode solves a complete job from the measured fleet.
- **CONTROL-I5.** (SUSPENDED by CONTROL-I25, the human, 2026-10-11, until the heterogeneous track returns: such a
  fleet is refused `MIXED-FLEET`, and this invariant's test asserts that refusal meanwhile.) A mixed CPU + GPU (MPS)
  fleet trains losslessly: bit-identical to a run on the same device map.
  Verified only on hosts with an Apple MPS device (the test is skipped in Linux CI). A CUDA + CPU variant on GPU CI is
  M1 work.
- **CONTROL-I6.** A wide stage with throughput-weighted ownership, launched by the coordinator, is bit-identical to
  the oracle.
- **CONTROL-I7.** Liveness:
  - a healthy fleet is ok, and all clean exits are done;
  - a process killed from outside is convicted at once;
  - an engine exception fails the run without convicting a machine;
  - a lost connection waits for the reconnect grace;
  - two simultaneous losses are not two dead machines;
  - heartbeat silence needs two strikes;
  - a peer-lost report points at the silent slot;
  - alive but not progressing is a stall, not a conviction.
- **CONTROL-I8.** The checkpoint store finds the common step and prunes. A corrupt shard (sha256 mismatch) is refused.
  A resume spec pins everything except addresses, devices, slot keys and `resume_step` (and, for a rebalance only,
  ownership). A wave whose manifest names another step, or any of whose files mismatches its hash, is refused.
- **CONTROL-I9.** Replacement prefers an identical model, then the region. It respects the recipe's compute dtype and
  the dead list. (DEFERRED-TO: M2b) A slot of a sharded stage is replaced only by the identical device model.
- **CONTROL-I10.** Every recovered run stays bit-identical to the oracle:
  - a dead worker is healed;
  - a killed stage is healed;
  - a frozen stage is restarted;
  - a wide-stage peer death is healed.

  A control-plane blip reconnects without recovery. No spare fails cleanly, with the reason.
- **CONTROL-I11.** Probes measure this node. Fields a worker only *declares* (region, uplink, capability) are marked
  declared, never treated as measured. Link measurement works over loopback with emulated WAN. Duplex is
  classified. An unreachable hop raises instead of hanging.
- **CONTROL-I12.** The audit policy:
  - always replicates newcomers;
  - decays reputation to a floor, not zero;
  - bans on one mismatch;
  - needs the seed for its sortition;
  - records evidence.

  A rejected poisoned contribution leaves the honest bits.
- **CONTROL-I16.** The coordinator holds its own ed25519 key and prints its id. Every control connection first runs the
  control-domain handshake (TRANSPORT-I5) and is sealed (TRANSPORT-I9) before any message. A worker refuses a
  coordinator whose key is not its pinned `--coordinator-id`, which is required off loopback.

  With a token, a worker is admitted only with `token_mac(token, transcript)`. A wrong or missing MAC is refused and
  never registers. Outside a loopback bind, the coordinator refuses to start without a token.

  A worker holding a token refuses `auth=false`. A reconnect sends `session_mac(session, transcript)`, never the
  session, and is accepted only from the key the node registered with.
- **CONTROL-I17.** The stall clock reads only strictly increasing `WT_STEP` progress inside [the attempt's start step,
  `steps − 1`], so a
  replayed loss stream never moves it and a resumed attempt is judged by `first_step_timeout_s` until its first real
  step. Once every launched slot finished the final step, quiet is shutdown, bounded by `shutdown_limit_s`, never a
  stall.
- **CONTROL-I18.** A worker's message never stops the coordinator from reading that worker: a malformed or stale
  message is logged and dropped and its file body unlinked. `TELEMETRY` and `TELEM_STEP` take rank, peer and attempt from
  the connection's slot, never from the message; a `TELEM_STEP` summary is accepted only within the size, key, range and
  identity rules of C2 (ADR-0002-m2-p1-spec, from `docs/design/NEXT-m2-p1-liveness-telemetry.md` §Mechanism).
- **CONTROL-I13.** (DEFERRED-TO: M4) The audit policy is wired: replicated microbatches are compared by digest before the fold, and a
  mismatch quarantines and ejects the worker without changing the result.
- **CONTROL-I14.** (DEFERRED-TO: M4) Coordinator loss does not lose the run: a standby takes over from the replicated registry and the
  checkpoint store.
- **CONTROL-I15.** (DEFERRED-TO: M4) A wide-stage peer loss heals from replicated ZeRO-1 shards without rolling back to a checkpoint.
- **CONTROL-I19.** (DEFERRED-TO: M4; ADR-0007) The init record and the init class.
  - `HELLO` carries the node's `sampler_fingerprint()` and diagnostic class, computed once per agent start in a fresh
    child process that calls `setup_determinism` before its first kernel dispatch, in the environment `stage_main`
    gets; never in the agent process, whose `probe.measure` dispatches kernels unpinned (MODEL-I6).
  - `JobSpec.init_record` is the recipe's (MODEL-I6), made once by `init_record` on a host the operator chooses; the
    coordinator copies it and never derives it from the selection. A first launch with `build: "slice"` and no record,
    or a record whose binding is not the job's, is refused with `INIT-RECORD`, naming the command that makes one.
  - One exclusion rule: a node whose fingerprint differs from the record's, or whose slot reports `INIT-CLASS`, an
    `INIT-LEDGER` at a transition or `INIT-DIGEST`, is excluded from init for that record. Selection (SCHEDULER-I18)
    and `find_replacement` apply it from `HELLO` before a launch, and a slot's report adds the node and reruns the
    selection. A slot's `INIT-RECORD`, or its `INIT-LEDGER` at the anchor, is the record's fault and equal on every
    node: the coordinator refuses the job once with it, naming the record, and excludes no node.
  - A heal or restart before the run's first wave relaunches from step 0, so its replacement builds its slice under
    RUNTIME-I14 and the exclusion rule applies; without an admitted spare the heal fails cleanly with the reason. After
    the first wave a replacement resumes and draws nothing, so init excludes no node. A slot whose device is the CPU
    still computes with its CPU class, which this rule does not decide (ADR-0007 R9): in a run with a record,
    `find_replacement` admits for it only a node whose fingerprint equals the record's (a necessary condition, not a
    proof of equal step bits), and fails cleanly with the reason otherwise. A slot whose device is not the CPU is
    healed under CONTROL-I9 alone.
  - `PROFILE` and `FIT` take the slot's slice of `skeleton(cfg)`, `to_empty`, re-tied (MODEL-I6), filled uniformly
    within `±init_std` from a private `torch.Generator` seeded 0: they never draw from the global generator, need no
    record and hold the slice (`4·P_s` plus their gradients and activations), never the full model. Under
    `arch="user"`, which has no skeleton, they keep today's full build.
- **CONTROL-I20.** CKPT, CKPT_PART, RESULT and request replies take the slot from the connection, never from the
  message. A CKPT or CKPT_PART of another attempt, or a reply from a worker other than the one asked, is dropped and
  logged. Staged parts are bounded: at most 2 uncommitted steps per slot, each part no larger than its manifest entry;
  anything else is dropped and unlinked.
- **CONTROL-I21.** A rebalance runs only under the rule of task 2, on the thread that runs heal and restart, and it
  reads only the current attempt's telemetry. After the abort it restores the newest common wave. It is recorded,
  consumes neither restarts nor heals, and the run stays byte-identical to the oracle.
- **CONTROL-I22.** Each agent's key lives in its workdir. It authenticates the agent's control connection and is passed
  to its stage. Every launched, healed, restarted or rebalanced spec pins `slot_peers[stage][peer]` to the peer id the
  assigned node proved in its control handshake, and `trusted_peers` to their set.
- **CONTROL-I23.** Every wave a slot commits before a kill is shipped, in step order, unless it was pruned unshipped,
  which is logged as a `ship gap`. The store holds and restores waves as units. RESTORE:
  - is refused while the slot's stage is alive;
  - runs under the ship lock;
  - unlinks every body on every path;
  - refuses a local wave of the same step whose manifest differs;
  - marks the restored wave shipped.
- **CONTROL-I24.** (DEFERRED-TO: M4; ADR-0009) An `arch="hf"` run trains only on nodes whose launch gate passed
  (MODEL-I10) and whose `hfm.fingerprint` is admitted.
  - Admitted: transformers' RECORD sha256 and version equal on every node; torch's version and `git_version` equal on
    every node, and each node's torch RECORD sha256 equal to `hfm.TORCH_RECORDS`' entry for its platform tag. Nodes of
    different platforms are admitted together only within one device class (CONTROL-I25). A device map whose classes (device type
    and model; a CPU's ATen capability) no single node of the pool holds among its CPU and slot devices, as the gate
    reports list them, has no oracle and is refused before placement, naming the classes (MODEL-I9; ADR-0009 R11).
    The refusal is re-checked at every admission (first launch, heal, restart, rebalance, resume) on the map that
    admission would build, from the gate reports of the nodes then in the pool: the run records each slot's device
    class (device type and model; a CPU's ATen capability, as its gate reported), admits a replacement only of that
    class, and refuses unless some node of the current pool holds every class of the map; so the recorded map, and
    its oracle, never change (ADR-0009 decision 8). A heal that fails it fails cleanly; nothing relaunches.
  - Each node's native set (MODEL-I10: OS and driver objects by sha256) equals that of every node of the same
    (platform tag, device class).
  - The coordinator records the fingerprint (its record builder's included), each slot's torch RECORD sha256 and the
    buffer record (bytes and a sha256 per tensor, pinned in the JobSpec; every slot lists it in each wave, RUNTIME-I19)
    in the run; a heal, restart, rebalance or resume admits a node for a slot only after its
    gate passed, its fingerprint is admitted, its torch RECORD sha256 and native set equal the slot's recorded ones,
    its device class equals the slot's and decision 8's re-check passes (gate, fingerprint and RECORD alone never
    admit); a resume under
    another transformers RECORD is refused, naming the differing file.
  - A failed gate, a refused environment (SPEC-model §2) or an unequal fingerprint names the node and the cause; no
    slot is launched.
- **CONTROL-I25.** (The human's decision, 2026-10-11; it settles ADR-0009 decision 8's recorded conflict and DESIGN §10
  question 7: the oracle holds per device map, ADR-0005) Until the heterogeneous track returns, a run trains only on a
  homogeneous fleet: every slot of its device map has one device class, for every arch (builtin, general, user, hf).
  - A slot's class is its node's `NodeInfo.device_class`, sent in `HELLO`: `"cuda:<device name>"`, `"mps:<chip>"` or
    `"cpu:<machine>:<ATen CPU capability>"`, the string `hfm.device_class` returns, computed by one torch-only function
    both use. An operator may set `WANTRAIN_DEVICE_CLASS=arch` on every worker: CUDA classes are then
    `"cuda:sm_<major><minor>"`, one class per architecture (an RTX 3060 and an RTX 3070 Laptop GPU are both sm_86), for
    a run whose oracle holds per device map only (ADR-0005; the human, 2026-10-12: "same architecture, balance the
    load"). The grouping holds at admission only: a heal of such a run takes a spare of the dead slot's GPU model
    (`NodeInfo.gpu_name`), or fails naming it, so recovery never changes the device map the run's bits belong to. A node with an empty class is refused, naming it; for `arch="hf"`, a gate-reported class other than the
    node's `device_class` is refused, naming both.
  - Checked before anything launches, on the map each admission would build: selection and the first launch's
    admission before placement, `planner.plan_job` (plan mode) and `placement.place` (recipe mode), heal, restart in
    place, rebalance and resume; `JobSpec.finalize` refuses a `devices` map of more than one device type (RUNTIME-I15).
    Peers of one class at different speeds (a slowed peer, a SIMULATED heterogeneous fleet) are admitted. Spares of
    another class stay registered and are never assigned; a heal with no spare of the run's class fails cleanly and
    nothing relaunches.
  - Refusal reasons, exact form (`MIXED-FLEET:` is the stable prefix tests match):
    `MIXED-FLEET: the device map spans <k> device classes (<class>: <node>, ...; <class>: <node>, ...); only one device
    class is admitted until the heterogeneous track returns`; at a heal `MIXED-FLEET: no spare of the run's device class
    <class> (pool: <class> x<n>, ...)`; for a missing class `MIXED-FLEET: node <name> reports no device class`.
  - CONTROL-I24's decision-8 re-check stays; on a one-class map it refuses nothing this invariant admits.
  - Test (`tests/unit/test_control_admission.py`; covers: CONTROL-I25; WOULD-FAIL-IF: a two-class device map is
    admitted for a builtin/general/hf arch at launch or heal): for each of `arch` builtin, general and hf, a pool of two
    classes (`cpu:x86_64:AVX2` and `cpu:x86_64:AVX512`; `cuda:A` and `cuda:B`) is refused at launch admission with
    `MIXED-FLEET` naming both classes and nothing launched; `plan_job` and `place` refuse it; a heal whose only spare
    is of another class fails with `MIXED-FLEET` and relaunches nothing; control: a one-class pool at two speeds is
    admitted.
- **CONTROL-I26.** A measured fit overshoot is fed back before the microbatch is halved. An overshoot is a slot's
  `peakReservedBytes` above its budget (`(1 - FIT_HEADROOM)` of the node's memory), or an out-of-memory fit, counted
  as a peak of the whole card. A node's first overshoot takes `MEM_FEEDBACK` (1.25) times the excess off its
  activation budget for the zero-bubble solve (`_cut_budgets`): fewer microbatches in flight there, same cut. A
  repeat, when the operator fixed no partition, lowers that node's memory as the planner sees it by the same amount
  (`shrink_for_overshoot`), so the cut moves layers to nodes with room; fit budgets stay the nodes' real memory. Only
  without an overshoot, or with the operator's partition, is the microbatch halved; all share `FIT_ATTEMPTS`.
- **CONTROL-I27.** The cut follows the run's measured costs. At a fresh common wave, before re-planning ownership
  (CONTROL-I21), the coordinator re-cuts a chain RUNTIME-I26 admits (`_maybe_recut`). The new cut is proportional to
  each stage's throughput in the telemetry window (`refined_partition`, under the planner's memory fit). It applies
  when it shortens the slowest stage by `recut_min_gain` (0.10) and saves more than `recut_cost_s` (600 s) over the
  remaining steps. The new cut is re-profiled, fit-checked (a refusal relaunches the current cut) and resumed from
  the converted waves; it spends one of `max_replans`. A slot whose compute per microbatch (F + B + W medians) stays
  above `slow_ratio` (1.3) times its own baseline for `slow_runs` (3) summaries in a row is a slow stage
  (`_watch_slow`). Its baseline is the median of its first `slow_runs` summaries of the attempt, after the warm-up
  and pinning ones. A slow stage is healed onto an idle spare of the run's class or, with none, re-cut away from on
  its recent costs; nothing happens before a common wave exists. `recut` false or WANTRAIN_RECUT=off disables both
  re-cut paths.

## 5. Ordered task list

1. **M1 — live smoke.** `/wan-smoke`, then a 2–4 GPU run across two regions under an approved experiment plan. This
   validates the probe numbers, the profiler and the predicted step time against measurements.
2. **M2 — re-plan loop.** Re-solve (SPEC-scheduler task 2) when `simulate_slots` on the current attempt's telemetry
   window predicts `T_new ≤ 0.90·T_cur` and `(T_cur − T_new)·steps_remaining > 2·R`, at most 2 times per run, at a
   fresh common wave, and apply it at a checkpoint boundary through restart in place (CONTROL-I21).
3. **M4 — audit wiring (CONTROL-I13)**, with SPEC-runtime's quarantine path (`OrderedAccumulator.release/reject`).
4. **M4 — coordinator standby (CONTROL-I14).** Replicate the registry and the waves. Agents fail over to a standby
   endpoint.
5. **M4 — shard replication (CONTROL-I15).** Each ZeRO-1 shard is also held by one other peer.
6. **M4 — HF launch gate (CONTROL-I24, ADR-0009).** The `GATE` request in the agent and the coordinator before
   placement; the buffer record built once under the guard, pinned in the JobSpec; the fingerprints (the builder's
   included) in the run record; their check, and decision 8's re-check, at heal, restart, rebalance and resume.
   Rollback (ADR-0009 `round6.md` §4): to SPEC-model 4a (hf refused before placement), never while 4b is live.
7. **M4 — homogeneous fleets (CONTROL-I25, the human, 2026-10-11).** `NodeInfo.device_class` in `HELLO`; the
   `MIXED-FLEET` refusal in selection, launch admission, `plan_job`, `place`, heal, restart, rebalance, resume and
   `JobSpec.finalize`; CONTROL-I5's test turned into the refusal; the CONTROL-I25 test.

## 6. Decisions

- **A long-lived coordinator with spares**, instead of the source's one-shot master.
- **Heartbeats and exit reports from agents, instead of SSH probes.** The control plane must not depend on shell
  access.
- **Simultaneous losses are a network failure.** Rejected: convicting several machines at once, which empties a fleet
  on one partition.
