# HANDOUT M2: rest of M2 plus the M4 security work brought forward, as one plan

Machine plan: [PLAN-M2.json](PLAN-M2.json). It has 16 tasks in 7 waves, 47 binding items and no tripwires (the reason
is in §6). Replanned on 2026-10-09 by the human's decision (§2a): the experiment task and its gates are removed. The scope is the human's focus of 2026-10-09 (`.factory/runs/M2/focus.txt`), which overrides DESIGN §8 where
the two differ. It includes the clause added after ADR-0006 (6bff9f5): the wave redesign stays compatible with
RUNTIME-I12. The base tree is `factory/m2` fast-forwarded to `origin/main` at 6bff9f5. Hardware: `gpu: false`.
Everything validates on this machine's CPU, and every performance number is labelled SIMULATED. Real-hardware
validation is pending (M1).

## 1. Goal and scope

| Focus item | Delivered by | Invariant |
|---|---|---|
| (1) M2-P2-NO-COMMON-WAVE: the causal fix (the experiment's exploratory data is recorded in §2a), then the M2-P2 minors in the files touched; compatible with RUNTIME-I12 | `restore-hardening` (minors) → `wave-layout` → `wave-stage-write` → `wave-ship-restore` → `wave-resume` | RUNTIME-I17, CONTROL-I23 |
| (2) Rebalancing between steps from phase-1 telemetry, with the fold order unchanged | `replan-solver`, `resume-ownership`, `rebalance-loop` | SCHEDULER-I15, RUNTIME-I15, CONTROL-I21 |
| (3) Degrade instead of refusing under tight memory | `degrade-memory` | SCHEDULER-I13 |
| (4) M2-P1-MESH-AUTH, M2-P1-CKPT-IDENTITY, coordinator token and proof, slot key pinning | `ctrl-msg-identity`, `ctrl-mutual-auth`, `slot-key-pinning`, `stage-p2p-secure`; CKPT_PART identity in `wave-ship-restore` | CONTROL-I20, CONTROL-I16 (amended), CONTROL-I22, RUNTIME-I16 |
| (5) TRANSPORT-I7 with the human's cipher decision, cost measured | `identity-handshake`, `aead-records`, `control-auth-primitives`, `stage-p2p-secure` (`aead_cost`) | TRANSPORT-I5 (amended), TRANSPORT-I7, TRANSPORT-I9 |

**Not in this plan:**
- Plan mode for wide stages: moved to M2b by ADR-0006, as the focus says.
- `M2-MPS-INTERMITTENT`: N/A. It needs an Apple MPS device, and `gpu` is false.
- `M2-R5-*`: factory or phase-workflow findings, except FIT-3, a telemetry minor that is not one of the M2-P2 minors.
- Fuzzing of `serial` and frames, audit wiring, coordinator standby and the slice-only build stay in M4.

**Facts (a)–(h) from the security session, checked against the code on this tree (6bff9f5):**

- (a) True. `stage.py:525-559` accepts `n_in` sockets and trusts each HELLO's `(st, pe)`.
- (b) True. `stage.py:289-290` raises on the first failed authentication. The chain has a second defect the fact does
  not name: `stage.py:259-264` accepts exactly `n` sockets with no timeout before authenticating any, so a silent
  connector holds a stream slot and blocks the join forever. Both are fixed (BA-26).
- (c) True. `trusted_peers` is read at `stage.py:280` and filled only by tests. The key is per host (`identity.py:57-61`).
- (d) True. See `identity.py:33-45`.
- (e) True. See `identity.py:146-166`.
- (f) True for the control plane: `control.py:138-145` has a single domain. The data plane has the same defect:
  `identity.py:132-136` signs `wantrain-auth|` plus the peer's challenge, a symmetric message that a second connection
  can reflect. Both are fixed (BA-32, BA-33). The coordinator stops proving itself with the token at all: it proves its
  own key (BA-42), so no token-derived proof is left to reflect.
- (g) True. `stage.py:552-556` sends the HELLO only after every accept.
- (h) Taken into the acceptance criteria and §7. No task's scope contains `tests/helpers/exact.py`, and every task that
  runs a fleet checks its sha256 `9fcb6cb57fec968b9f348ae867a01567b20a095c1744ba4b43b9fb2464255e7d`.

**A spec/code divergence that bears on (1).** SPEC-control §2.4 says the agent "ships checkpoint files after `WT_*`
checkpoint lines". The code prints no such line. It samples the shard every 0.5 s (`agent.py:542-548`). The fix
restores what the spec describes. The experiment decides whether that is the cause.

## 2. Preconditions (before `/wan-implement M2`)

- **P1. Spec amendments on the record.** The invariants rung fails on a test that claims an id no spec defines, and
  this plan's tests claim the ids below. Planners cannot write specs. After the human grants this handout, the
  researcher records every text in §8, including the replacements of passages that would otherwise contradict the new
  invariants, with `node agents-architetura/tools/factory/freeze.mjs amend <spec> --reason "..." --by HANDOUT-M2`.
  This includes the SPEC-transport cipher amendment that the focus asks for.
- **P2, P3: removed by the replan (§2a).** No base-tree export is needed, and no verdict gates `wave-layout`.

## 2b. Invariant ids renumbered after merging main (deviation, 2026-10-09)

Main's ADR-0007 (granted) took CONTROL-I19 (the init record and the init class) and RUNTIME-I14 (a slot materializes
only its slice). By the human's rule (main wins where the two collide), this plan's wave invariants are renumbered:
the wave store and shipping, formerly CONTROL-I19, is **CONTROL-I23**, and the immutable wave and its resume, formerly
RUNTIME-I14, is **RUNTIME-I17**. Every reference in this handout, the PLAN, the code comments and the tests'
`covers:` lines was renamed; the texts of §8 are recorded under the new ids. CONTROL-I20..I22 and RUNTIME-I15..I16
keep their ids.

## 2c. Amendments after the build (the human's decisions, 2026-10-09, `--by HANDOUT-M2`)

- **BA-44 and GATES `aead-step-cost-simulated`.** The 1% bound stays; what is measured changes. Reason: the toy moves
  ~10 MB per step in a ~41 ms step, so the serialized sum `seal_ns + open_ns` measures the cipher, not the cost on
  training (measured 3.6% on the 1-2-1 wide middle slots, recorded deviation of `stage-p2p-secure`). The gate now bounds
  the overlap-aware cost (sealed step wall time minus unsealed, interleaved pairs on the same seed, k ≥ 5, median of
  the ratio, `OVERLAP_COST_FRAC < 0.01`); the serialized sum is reported as MEASURED with no bound. No production flag
  disables sealing: the unsealed arm is a swap inside `validation/aead_cost.py`'s measurement process. If the overlap
  cost is ≥ 1% on the toy, the gate stays red and the closeout reports it; the bound is not loosened.
- **AEAD on the toy: accept now, ADR later (the human's decision, 2026-10-10 ~09:15, `--by HANDOUT-M2`).** GATES
  `aead-step-cost-simulated` becomes MEASURED with no bound on the toy: it prints the overlap-aware cost per
  configuration (median and distribution-free CI) and the serialized sum, requires those lines, and fails only on an
  invalid measurement. Measured (SIMULATED toy, this Mac, shared with the FSDP build): wide_1_2_1 ~2.6–4.5%
  (0.0420 [0.0371, 0.0487] and 0.0455 [0.0395, 0.0496] on the final zero-copy tree; 2.56% and 3.24% [2.66, 3.75] in
  earlier runs), chain2 0.77% [0.44, 1.04]. Cause, from the profile: the AES-GCM pass itself on the slots'
  communication threads competes with the Python compute, ~1.07 ms per step on a wide slot (~10 MB sealed and opened
  per step); copies, allocations and GIL handoffs cost 2–4 µs per frame. Tried: a thread pool for large messages
  (the cipher releases the GIL, 3.8× on 4 threads) and a zero-copy record layer (`encrypt_into`/`decrypt_into`); neither
  moved the toy's step cost, because most records are whole 131 KB messages. **Follow-up ADR:** take the cipher off the
  critical path (sealing outside the GIL: a dedicated process with shared memory, or a native extension that releases
  the GIL; several cores per large message, needed for a 100 Gbit/s NIC above the ~7 GB/s of one core).
- **GATES `rebalance-arms`: measured speedup.** `test_slowed_peer_is_rebalanced_and_the_state_stays_exact` also measures
  the real step (mean wall time per step over the last steps of attempt 1 after the slowdown and before the replan,
  against attempt 2) and asserts `measured_new ≤ 0.75 × measured_before`; it prints `MEASURED_SPEEDUP`, which the gate
  requires. The bound was fixed before any run; a red is a finding, never a reason to move it.
- **Rebalance prediction: measure, do not block (the human's decision, 2026-10-09 ~22:45, `--by HANDOUT-M2`).** The
  planner's prediction error (the gain `pred_cur_s − pred_new_s` against the measured gain, and the absolute step
  `pred_new_s` against the measured one) is reported as MEASURED, with no bound: the test prints `MEASURED_BIAS` and
  `MEASURED_PRED_ERROR`, which the gate requires. Measured on the toy (SIMULATED, this Mac): measured/predicted
  1.28–2.3, red in 2 of 3 against the former ±35%. The speedup bound (≤ 0.75×) stays a gate. **M3 item:** calibrate
  `predict_step` for CPU contention (a fixed per-step overhead the cost model leaves out, HYPOTHESIS).

## 2a. Replan of 2026-10-09 (the human's decision)

`/wan-implement M2` stopped on its first task: the frozen verdict rule of `wave-cause-experiment` could not tell the two
hypotheses apart (step 1 is overwritten within one 0.5 s supervise period and the killed slot's step 5 precedes the
kill, so "unseen" appeared in every symptomatic run and SEQLOCK was unreachable). The human decided: no experiment as a
gatekeeper. The task `wave-cause-experiment` and the gates `wave-cause-verdict` and `no-common-wave-closure` are
removed; only the wave tasks depend on each other, and every other root starts in wave 1.

**Evidence kept (exploratory, `.factory/runs/M2/wave-exp-tree.json`, tree 6bff9f5 plus the experiment's log lines):**
18 unloaded runs of `test_dead_worker_is_replaced_and_the_run_stays_exact`, 3 symptomatic (runs 7, 14, 17; p0 ≈ 0.17).
Both kinds were observed: `unseen` (a saved step the sampler never read) and `race` (a seqlock race-skip, run 17).
Runs 7 and 14 had no race-skip on any slot: the symptom occurred with the seqlock never firing.

**Why the decision does not need the verdict.** The redesign (immutable waves with the step in the file name, every
committed wave shipped in ascending order) removes both observed causes: nothing is sampled, so nothing goes unseen,
and nothing shipped can be rewritten, so nothing races. Its effect is proven by the causal closure (§7), which compares
the tree immediately before `wave-ship-restore` (snapshot left by `wave-stage-write`'s verify at
`.factory/runs/M2/wave-prefix.tree`) with the fixed tree, both without load.

## 3. First-principles cost models (the terms each mechanism moves)

`T_step = T_compute + T_comm + T_sync + T_overhead`. Recovery adds `R` (restore plus relaunch) and the lost work.

### The experiment (measurement, not mechanism)

**Why alone.** The symptom needs `ckpt_every · T_step < 0.5 s` on unloaded toy steps: it vanished 20/20 under 2× load
(findings.json). Any concurrent task's tests load the CPU, lengthen `T_step`, lower `p0` and push the verdict toward
INCONCLUSIVE. Wave 1 therefore holds only the experiment.

**Why a snapshot.** Each tree arm copies `src/`, `tests/` and `pyproject.toml` before its first run and runs every fleet
from the copy. The evidence then names a tree that never changes, `--check` stays valid after later tasks edit `src/`,
and the copy is the causal closure's pre-fix arm.

**Length.** Up to 60 fleet runs at 30–60 s each (HYPOTHESIS), so up to an hour. No single command may outlast an agent
shell's 10-minute limit (LESSONS), so the script appends a row per run, stops starting runs after `--budget-s`
(default 480 s), exits 2 ("call again"), and resumes from its rows. The rough prior is one symptomatic run in six
unloaded runs (findings.json), so 60 runs expect about 10 symptomatic runs; `K = 3` symptoms are required for any
verdict but INCONCLUSIVE (HYPOTHESIS: the rate is not measured yet).

### Waves: every one shipped, as manifest-bound units (RUNTIME-I17, CONTROL-I23, compatible with RUNTIME-I12)

**Hazard.** A slot saves a wave every `ckpt_every · T_step`. The sampler sees one wave per 0.5 s poll. When
`ckpt_every · T_step < 0.5 s` (toy: `2 × ~0.1 s`, HYPOTHESIS from the finding), two slots can ship disjoint steps.
`common_step` is then None, and the heal restarts at step 0. That costs time only; the bytes stay exact.

**Mechanism.** Write once, announce, ship in order. **The wave is the unit everywhere:**
- The stage commits `waves/<tag>/r<r>p<p>/step-<n>/` with `wave.json` written last. `wave.json` lists every file with
  its sha256.
- The agent ships the files and then the manifest (`CKPT_PART`…, `CKPT`).
- The store holds the wave. RESTORE writes the wave back.
- Resume validates the wave of `resume_step` before it applies any state.

`wave.json` is the only step binding, which is what RUNTIME-I12 calls "the manifest is the wave's sidecar". M2b adds
piece entries `(tensor, offset, length)` and per-chunk hashes to the same file list. CKPT, the store, RESTORE and resume
then need no redesign.

The old pair `<tag>_ckpt_rank<r>p<p>.pt` + `.step.json` survives only as an **alias derived from the wave**, for the
oracle readers (`validation/state.py`, `tests/helpers/exact.py`), which may not change. The alias is a hard link, so
every writer replaces the directory entry and none writes into it in place (BA-46).

**I/O per wave.** The stage's I/O is unchanged: today it writes a temp file and renames it. Now it writes a temp
directory and renames that, and the alias is a hard link costing O(1). The agent no longer copies a snapshot before
sending, which saves one read and one write of `B_s` per wave. Hashing adds one read of `B_s` at the stage, off the
step: HYPOTHESIS ≈ `B_s / 1 GB/s`, which is 10 s for a 10.5 GB slot, inside `ckpt_every · T_step` whenever the control
link is the limit below.

**Disk per slot.** 3 waves (`3·B_s`), against `2·B_s` today (the shard plus the agent's snapshot).

**Control-link requirement, unchanged per wave.** `B_s / BW_ctl ≤ ckpt_every · T_step`.
- Example: 7B on 8 stages gives `P_slot ≈ 0.875e9`. At 12 B/param (fp32 weights + AdamW m, v), `B_s ≈ 10.5 GB`. At
  1 Gbps that is about 84 s per wave, so at `T_step = 2 s`, `ckpt_every ≥ 42`.
- A sustained violation now appears as a counted `ship gap` once the local queue passes 3 waves. It is never a silent
  sample.

**When the gain vanishes.** When `ckpt_every · T_step ≫ 0.5 s`, sampling and shipping every wave behave the same. That
is the loaded case, which already passed 20/20.

### Rebalancing (SCHEDULER-I15, CONTROL-I21)

**Gain.** `G = (T_cur − T_new) · steps_remaining`. Both terms come from `simulate_slots` on the **same measured
costs**: the median of `med_ms` over the window, and the per-hop `busy_ms / frames`. Comparing one prediction with
another cancels the profiler's ~30% CPU bias. The window holds only the current attempt (BA-23), so after a heal or a
rebalance no summary of a dead node or an old owner enters the prediction.

**Cost.** `R` (abort + restore + relaunch to the first step). The lost work is at most the steps since the common wave
the abort re-reads (BA-24), which with every wave shipped is at most `ckpt_every` steps.

**Where it runs.** On the `_monitor` thread only, which already owns heal and restart (BA-45). A CKPT is handled on its
worker's connection thread, which is also the only thread that can deliver that worker's ABORT_ACK; applying there
would wait 60 s for a reply it blocks itself. One thread also means a rebalance and a heal can never both start an
attempt.

**Rule (the "X%" the specs leave to this handout).** Apply when all three hold:
- `T_new ≤ 0.90 · T_cur`. Telemetry medians on CPU vary by a few percent (HYPOTHESIS, measured by the decision tests),
  and 10% stays above that noise.
- `G > 2R`, a margin for error in the estimate of R.
- Fewer than 2 rebalances so far in the run.

**Example.** A width-2 stage has one peer 4× slower:
- equal IWRR gives a stage time of `max(M/2·4c, M/2·c) = 2Mc`;
- weights 1 : 0.25 give `0.8Mc`, which is −60% on the stage term;
- with `R ≈ 10 s` (HYPOTHESIS), the rebalance pays when `1.2Mc · steps_remaining > 20 s`.

**Exactness.**
- The fold is in microbatch-index order (RUNTIME-I4).
- ZeRO-1 state is `params[k::w]`, independent of `mb_owner`.
- No training op draws the global RNG.
- The data RNG is keyed by `(seed, step, mb)`.

So moving ownership moves no bit. BA-18 to BA-21 rule each of these, and `resume-ownership` proves it by sha256.

### Degrade under tight memory (SCHEDULER-I13)

**Today.** `solve()` silently raises the measured budget to the 1F1B floor (`zero_bubble.py:516-520`). A tight device
over-commits and OOMs at run time.

**New.** Use the largest per-stage warm-up `k_j ≥ 1`, non-increasing down the pipeline, whose memory replay fits.
- Peak memory is `≤ max_mem` by construction.
- The bubble grows as `k_j` shrinks, and it is reported.
- The only refusal left is a budget below one microbatch, and it names the bytes.

### Data-plane AEAD and handshake (TRANSPORT-I5, I7, RUNTIME-I16)

**Handshake.** Per connection: one X25519 exchange, one ed25519 signature and one verification, `< 1 ms`
(HYPOTHESIS). There are `2(S−1)·streams` connections per attempt on a chain, and
`Σ_s w_s·w_{s+1} + Σ_s w_s(w_s−1)/2` on a mesh. The handshake costs nothing per step. An intruder costs the acceptor
one thread for at most the 30 s handshake timeout and never a stream or mesh slot.

**Records.** Each socket and direction seals in the sender thread and opens in the reader thread, overlapped with
compute. Two numbers are **measured**:
- **Per-core throughput `X`** (`secure --bench`). HYPOTHESIS: AES-256-GCM with AES-NI or ARMv8 at 2–8 GB/s through
  OpenSSL, ChaCha20-Poly1305 at 1–2 GB/s. The requirement is `X ≥` the link rate: at 10 Gbps, 1.25 GB/s.
- **The cost on the step, measured directly** (`validation/aead_cost.py`, BA-44). Each SecureSocket times its own seal
  and open calls, and each slot reports `(seal_ns + open_ns) / steps_wall_ns`. This is the serialized upper bound,
  because it counts the time as if none of it overlapped; it is reported as MEASURED, with no bound (amended, §2c).
  The gated cost is the overlap-aware one: the wall time of the sealed step minus that of the unsealed step, in
  interleaved pairs on the same seed (k ≥ 5 pairs, median of the ratio), `< 1%` on the toy chain and on the 1-2-1 wide
  run (SIMULATED). The unsealed arm exists only inside the validation tool's measurement process.

**Where it fails.** In the datacenter regime of large models (16 MB per microbatch per hop), the serialized bound
fails, and only the overlap holds the cost under 1%. The real-hardware claim stays pending (M1).

**Second-order effect (R4).** Sealing adds about 2 host-DRAM passes per byte per side, which raises ADR-0006's `k_ℓ`
from 10 toward 12. M2b's S1 must count it.

**Rekey.** A deterministic HKDF ratchet per direction after `2^24` records or `2^36` bytes, inside RFC 8446 §5.5's
AES-GCM bound. Counter nonces never repeat under a key.

### Control channel (CONTROL-I16 amended, TRANSPORT-I9)

**Threat.** THREAT_MODEL says workers may be adversarial, and every admitted worker holds the token. A proof derived
from the token therefore cannot show that the peer is the coordinator. Any admitted worker on the path could compute it
and rewrite ASSIGN, including the pinned slot keys.

**Mechanism.**
- The coordinator has its own ed25519 key (`<state_dir>/coord_key`), and workers pin its id out of band
  (`--coordinator-id`, printed at start).
- The control connection first runs the same transcript-bound handshake as the data plane, under its own domain, and
  is sealed before any message.
- Inside the sealed channel, the worker proves the token with `HMAC(token, worker-domain ‖ transcript)`. This binds the
  proof to this session, so a relay that terminates the handshake gets a MAC over another transcript and is refused.
- HELLO's fields (endpoints, capabilities) and WELCOME's session travel only sealed.
- The worker's control key is its data-plane key, so the coordinator learns each slot's peer id from a verified
  handshake, never from a message field.

**Cost.** Control traffic is small, apart from waves. A 10.5 GB wave sealed at about 2 GB/s takes about 5 s of the
agent's CPU per wave, off the stage's critical path (HYPOTHESIS). On loopback without a token, the channel is still
sealed, and workers accept any coordinator key there (local mode).

## 4. Tasks

| # | Task | WS | Risk | Deps | Delivers |
|---|---|---|---|---|---|
| 1 | ~~`wave-cause-experiment`~~ | — | — | — | removed by the replan (§2a) |
| 2 | `degrade-memory` | scheduler | medium | — | SCHEDULER-I13 |
| 3 | `resume-ownership` | runtime | high | — | RUNTIME-I15 (recipe pins, `zero1_widths` only for whole-tensor shards), `slot_peers` and `resume_step` fields |
| 4 | `identity-handshake` | transport | high | — | TRANSPORT-I5 amended (per-plane domains, roles, transcript signatures, X25519 keys, SHA-256 ids, no HMAC) |
| 5 | `aead-records` | transport | high | — | TRANSPORT-I7 (records, counter nonces, rekey, seal/open stats, bench) |
| 6 | `ctrl-msg-identity` | control | high | — | CONTROL-I20 for CKPT, RESULT and replies (M2-P1-CKPT-IDENTITY) |
| 7 | `restore-hardening` | control | high | — | RESTORE-LIVE-STAGE, -BODY-LEAK, -SHIP-SPANS-RESTORE, -RESHIP, SHIP-TEST-TIMING |
| 8 | `replan-solver` | scheduler | medium | 2 | SCHEDULER-I15 |
| 9 | `control-auth-primitives` | transport | high | 4, 5 | TRANSPORT-I9 (control handshake, transcript-bound token and session MACs, sealed `Channel`) |
| 10 | `wave-layout` | control | high | 7 | immutable waves, hashed manifest, `validate_wave`, alias by replace only, commit once |
| 11 | `wave-stage-write` | runtime | high | 3, 10 | `WT_WAVE`, waves committed and announced, abandoned timelines dropped; resume still on the alias; leaves the pre-fix snapshot `.factory/runs/M2/wave-prefix.tree` |
| 12 | `wave-ship-restore` | control | high | 6, 7, 11 | CONTROL-I23; in-order shipping; wave store with bounded CKPT_PART staging (CONTROL-I20); wave RESTORE; `resume_step` on heal and restart |
| 13 | `wave-resume` | runtime | high | 12 | RUNTIME-I17 completed: resume from the validated wave of `resume_step`, never the alias |
| 14 | `ctrl-mutual-auth` | control | high | 9, 12 | CONTROL-I16 amended (coordinator key pinned, token off loopback, sealed before any message, `session_mac`) |
| 15 | `slot-key-pinning` | control | high | 3, 14 | CONTROL-I22 (stage gets its agent's key; slot → peer id pinned at launch, heal, restart and rebalance) |
| 16 | `stage-p2p-secure` | runtime | high | 3, 4, 5, 13, 15 | RUNTIME-I16 (M2-P1-MESH-AUTH: chain and mesh bind, refuse and keep accepting, seal every socket); `aead_cost` |
| 17 | `rebalance-loop` | control | high | 3, 8, 15 | CONTROL-I21 (decided and applied on the monitor thread, window per attempt, common wave re-read after the abort) |

## 5. Wave order

1. `degrade-memory`, `resume-ownership`, `identity-handshake`, `aead-records`, `ctrl-msg-identity`, `restore-hardening`
2. `replan-solver`, `control-auth-primitives`, `wave-layout`
3. `wave-stage-write`
4. `wave-ship-restore`
5. `wave-resume`, `ctrl-mutual-auth`
6. `slot-key-pinning`
7. `stage-p2p-secure`, `rebalance-loop`

**Why this order.**
- **No experiment gate (replan, §2a).** Only the wave tasks depend on each other; every other root starts in wave 1.
- **The causal fix before every chaos-gated task.** `ctrl-mutual-auth`, `slot-key-pinning`, `stage-p2p-secure` and
  `rebalance-loop` run the chaos rung, which holds the open major's test. They come after `wave-ship-restore`, which
  removes the cause, so none of them can be blocked by the flake it fixes. No pre-fix task runs the chaos rung.
- **Rebalancing is written once, against the wave store.** `rebalance-loop` depends (through `slot-key-pinning`) on
  `wave-ship-restore`, so it restores waves and sets `resume_step` with the final interfaces.
- **File order.** `agent.py`: 7 → 12 → 14 → 15. `coordinator.py`: 6 → 12 → 14 → 15 → 17. `checkpoint.py`:
  10 → 12 → 15 → 17. `stage.py`: 3 → 11 → 13 → 16. `test_fault_tolerance.py`: 7 → 10 → 12 → 15 → 17. No two tasks that
  may run concurrently share a path.

**Intermediate states (each is consistent on its own; none is silently wrong):**
- **After `wave-stage-write`, before `wave-ship-restore`.** The stage commits waves and derives the alias; the old agent
  samples and restores the alias; resume reads the alias. Heal and restart behave exactly as on the base tree, so the
  integration rung holds, including `test_relaunch_from_the_final_wave_runs_no_step_and_is_not_restarted`.
- **After `wave-ship-restore`, before `wave-resume`.** RESTORE writes the wave and its alias, and the spec names
  `resume_step`. The stage still reads the alias, which is that same restored wave, so every slot resumes from the
  common step.
- **After `slot-key-pinning`.** The chain authenticates against the filled `trusted_peers`. The mesh stays as it is
  today until task 16.
- **After `identity-handshake`.** Both ends of every chain run the same v2 handshake, because it is one tree.

## 6. Binding audit and tripwires

PLAN-M2.json holds 47 binding items. Each has a ruling and a task that delivers it. They cover:
- every piece of state that a change of wave layout, ownership or restart reads or writes (BA-1 to BA-15, BA-18 to
  BA-24, BA-43, BA-46, BA-47), including the double commit of the last step (BA-4), the waves of an abandoned or
  restarted timeline (BA-6), the alias as a hard link (BA-46) and the telemetry window across attempts (BA-23);
- the RNG draws and the reduction order whose bytes must not move (BA-19, BA-20);
- the solver's memory paths (BA-16, BA-17);
- where an attempt-changing action runs (BA-24, BA-45);
- every socket, handshake, message field and measurement the security work touches (BA-25 to BA-42, BA-44), including
  HELLO's identity and endpoints, WELCOME's session and the coordinator's key (BA-40 to BA-42).

**No tripwire is declared.** No multi-leg change here is silently wrong while only some of its legs are in:
- **Waves.** The resume leg was split so that each intermediate tree resumes from the bytes it restored (§5). The
  stage never reads a wave the agent did not restore, and the alias is always derived from a committed wave.
- **Rebalancing.** The one entry point is `rebalance-loop`, which depends on its guard (`resume-ownership`) and on the
  solver.
- **Slot keys, AEAD and the control handshake.** A partial state fails loudly, as a refused connection or a failed tag.

## 7. Proposed GATES-M2 (frozen by wan-handout after the grant; closeout runs it)

- **Rungs, twice in a row:** build, syntax, fmt, lint, types, discipline, frozen, invariants, claims, unit, integration,
  multiworker, oracle, identity, chaos, wan, rest, full, static, security, fuzz.
- **N/A, with the reason:**
  - multinode: no second host;
  - the MPS test: `gpu` false;
  - redteam: `tests/redteam/` does not exist on this tree. Its rung runs only when the folder exists, so a red-team
    pass that adds it is pinned automatically;
  - bench: there is no `bench/suite.json`. DESIGN §8's "M2 rungs + bench" is met by the pinned SIMULATED commands below,
    each with its bound.
- **M2-P2-NO-COMMON-WAVE.** The statistical closure gate (`no-common-wave-closure`) is removed by the replan (§2a); the loaded arm is kept: 20/20 under 2× load.
- **Causal closure, on whole trees.** Restoring single files cannot work, because the security tasks change the
  interfaces those files depend on. The closure compares two whole trees:
  - the bug is present on the pre-fix snapshot, the tree immediately before `wave-ship-restore` (left by
    `wave-stage-write`'s verify), without load:
    `wave_experiment.py --base-tree .factory/runs/M2/wave-prefix.tree --until-symptoms 1 --max-runs 60 --expect-symptom`;
  - the bug is absent on the fixed tree, without load: `wave_experiment.py --runs 30 --expect-fixed`.
- **Rogue arms** (chain: `intruder`, `intruder_with_two_streams`, `chain_silent_socket`; mesh: `intruder`,
  `claiming_another_slot`, `out_of_range`, `silent_socket`): 5/5 each. Every run shows the intruder engaged and
  refused, `p2p_auth`/`p2p_connected` counts equal to the topology on chain and mesh, and the run byte-identical.
- **Control arms.** `test_admitted_worker_cannot_pose_as_the_coordinator`, `test_relayed_hello_cannot_swap_the_worker_key`
  and `test_session_secret_never_travels_in_clear`: 5/5 each.
- **Rebalance arms:** `test_slowed_peer_is_rebalanced_and_the_state_stays_exact` 5/5, with `replans == 1`,
  `owner_changed`, losses equal to the oracle and `fleet_manifest` equal to the reference by sha256;
  `test_rebalance_restores_the_common_wave_after_the_abort` and `test_rebalance_and_a_heal_in_one_window_make_one_attempt`
  5/5 each.
- **AEAD (SIMULATED):**
  - `python -m wantrain.transport.secure --bench --link-gbps 10`: 3/3;
  - `python -m wantrain.validation.aead_cost --steps 20 --max-frac 0.01`: 3/3, with its JSON kept next to the
    performance model's prediction.
- **Oracle integrity.** The sha256 of `tests/helpers/exact.py` is unchanged.

## 8. Spec amendment texts (P1; recorded by the researcher, `--by HANDOUT-M2`)

### A1 — SPEC-runtime §2 Checkpoint

**Replace** "A missing shard, or a shard whose sidecar is missing or names another step, is fatal" **with:**

> **RUNTIME-I17.** Every checkpoint of step n is committed as an immutable wave
> `out_dir/waves/<tag>/r<rank>p<peer>/step-<n:08d>/`:
> - its files are written first;
> - `wave.json` (`{"v": 1, "step": n, "files": [{"name", "sha256", "bytes"}]}`) is written last;
> - the directory is renamed into place.
>
> A committed wave is never written again, and a second commit of the same step writes nothing. After a new commit, the
> slot derives the alias `<tag>_ckpt_rank<r>p<p>.pt` + `.step.json` from the wave (read only by the oracle readers;
> every writer replaces it, none writes into it), prints `WT_WAVE <n>`, and keeps its newest 3 waves.
>
> Resume reads only the wave of `resume_step` (the newest, in local mode). Before any state is applied, resume refuses:
> - a missing wave or `wave.json`;
> - a `wave.json` naming another step;
> - a listed file that is missing or has another sha256 or size;
> - an unlisted file.
>
> A resume from step n then removes the slot's waves newer than n; a launch that does not resume removes every wave of
> the slot.
>
> `wave.json` is the wave's sidecar (RUNTIME-I12): M2b lists piece files in it.

### A2 — SPEC-runtime §2 Checkpoint, last bullet; §2 Ownership; §3

**Replace** "The partition, `mb_size`, microbatches and schedule must equal those of the writing run" **with:**

> The state file records the recipe pins: model, seeds, data, microbatches, `mb_size`, `seq_len`, optimizer and
> hyperparameters, precision, wire codec, recompute, steps and partition. For whole-tensor ZeRO-1 shards it also
> records `zero1_widths` (the stage widths), because `names[k::w]` depends on w. A logical-layout checkpoint
> (RUNTIME-I12) is width-free and is checked through its manifest.
>
> Resume refuses a state file whose pins differ or that has none. `peer_weights`, `mb_owner` and `schedule_order` may
> change at a resume.

**§2 Ownership, replace** "The owner is static per run (IWRR over `peer_weights`)." **with:**

> The owner is static per attempt (IWRR over `peer_weights`); it changes only at a resume (RUNTIME-I15).

**§3 Consumes, transport line, replace** "`authenticate`" **with** "`authenticate`, `SecureSocket`, `hardware_aes`".

**Add:**

> **RUNTIME-I15.** A resume whose spec changes only `peer_weights`, `mb_owner`, `schedule_order`, `endpoints`,
> `devices`, `slot_peers` or `resume_step` continues byte-identically to the oracle: losses, weights, optimizer state
> and RNG. A resume whose pins differ is refused before any state is applied, naming the field.

### A3 — SPEC-runtime

**Add:**

> **RUNTIME-I16.** Every data-plane socket, chain or mesh, is authenticated (TRANSPORT-I5) before any frame and then
> sealed (TRANSPORT-I7).
>
> An accepted socket is refused, logged (`WT_P2P_REFUSED`) and closed, while accepting continues, when:
> - its key fails, or it sends nothing within the handshake timeout;
> - its id is not `slot_peers[st][pe]` (on the chain, `slot_peers[rank−1][0]`);
> - `st ∉ {rank−1, rank}`;
> - `pe` is out of range, or for intra `pe ≤` own peer;
> - `(st, pe)` is already bound.
>
> A refused socket never consumes a legitimate peer's slot or stream.
>
> A coordinator-run spec (endpoints set) without `slot_peers` is refused. Each legitimate socket yields exactly one
> `p2p_auth` and one `p2p_connected` event. Each slot reports its seal and open time (`WT_AEAD`).

### A4 — SPEC-control §2.4 and §2.3

**§2.4, replace** "ships checkpoint files after `WT_*` checkpoint lines" **with:**

> ships every committed wave in ascending step order, woken by `WT_WAVE` and by a backstop scan, as its files
> (`CKPT_PART`) followed by its manifest (`CKPT`); it never reads the alias.

**§2.3 step 4, replace** "`RESTORE` it to every slot (sha256-checked)" **with:**

> `RESTORE` the wave (files and manifest, each hash checked) to every slot.

**§2.3 step 5, replace** "(`patch_spec_for_resume` changes only endpoints and devices)" **with:**

> with `resume_step` set to that wave (`patch_spec_for_resume` changes only endpoints, devices, slot keys and
> `resume_step`, and for a rebalance the ownership).

**Add:**

> **CONTROL-I23.** Every wave a slot commits before a kill is shipped, in step order, unless it was pruned unshipped,
> which is logged as a `ship gap`. The store holds and restores waves as units. RESTORE:
> - is refused while the slot's stage is alive;
> - runs under the ship lock;
> - unlinks every body on every path;
> - refuses a local wave of the same step whose manifest differs;
> - marks the restored wave shipped.

**CONTROL-I8, append:**

> A wave whose manifest names another step, or any of whose files mismatches its hash, is refused.

### A5 — SPEC-control

**Add:**

> **CONTROL-I20.** CKPT, CKPT_PART, RESULT and request replies take the slot from the connection, never from the
> message. A CKPT or CKPT_PART of another attempt, or a reply from a worker other than the one asked, is dropped and
> logged. Staged parts are bounded: at most 2 uncommitted steps per slot, each part no larger than its manifest entry;
> anything else is dropped and unlinked.

### A6 — SPEC-control CONTROL-I16, §2.1 step 1 and §3

**CONTROL-I16, replace with:**

> The coordinator holds its own ed25519 key and prints its id. Every control connection first runs the
> control-domain handshake (TRANSPORT-I5) and is sealed (TRANSPORT-I9) before any message. A worker refuses a
> coordinator whose key is not its pinned `--coordinator-id`, which is required off loopback.
>
> With a token, a worker is admitted only with `token_mac(token, transcript)`. A wrong or missing MAC is refused and
> never registers. Outside a loopback bind, the coordinator refuses to start without a token.
>
> A worker holding a token refuses `auth=false`. A reconnect sends `session_mac(session, transcript)`, never the
> session, and is accepted only from the key the node registered with.

**§2.1 step 1, replace** "Agents connect, answer `CHALLENGE` with `token_mac`, and send `HELLO`" **with:**

> Agents connect, run the control handshake against the pinned coordinator id, answer `CHALLENGE` inside the sealed
> channel with `token_mac(token, transcript)`, and send `HELLO`

**§3 Consumes, transport line, replace** "`Channel`, `token_mac`" **with** "`control_dial`, `control_accept`, `Channel`,
`token_mac`, `session_mac`".

**§3 Provides:** `patch_spec_for_resume(spec_json, endpoints, devices, slot_peers=None, resume_step=None,
ownership=None)`.

### A7 — SPEC-control task 2

**Replace** "more than X% over the prediction" **with:**

> when `simulate_slots` on the current attempt's telemetry window predicts `T_new ≤ 0.90·T_cur` and
> `(T_cur − T_new)·steps_remaining > 2·R`, at most 2 times per run, at a fresh common wave.

**Add:**

> **CONTROL-I21.** A rebalance runs only under that rule, on the thread that runs heal and restart, and it reads only
> the current attempt's telemetry. After the abort it restores the newest common wave. It is recorded, consumes neither
> restarts nor heals, and the run stays byte-identical to the oracle.

### A8 — SPEC-control

**Add:**

> **CONTROL-I22.** Each agent's key lives in its workdir. It authenticates the agent's control connection and is passed
> to its stage. Every launched, healed, restarted or rebalanced spec pins `slot_peers[stage][peer]` to the peer id the
> assigned node proved in its control handshake, and `trusted_peers` to their set.

**CONTROL-I8, replace** "except addresses and devices" **with:**

> except addresses, devices, slot keys and `resume_step` (and, for a rebalance only, ownership).

### A9 — SPEC-transport

**§1, replace** "**Peer identity:** ed25519, with an HMAC fallback." **with:**

> **Peer identity and record protection:** ed25519 identities, X25519 session keys, AEAD records.

**§2 Identity, replace the bullet** ("an ed25519 key per host … There is no default token.") **with:**

> - **Identity** (`authenticate`): an ed25519 key per agent (atomic create, mode 0600) and `peer_id = sha256(pub)[:32]`.
>   - The handshake binds the plane's domain, both identities, both challenges and both X25519 ephemerals.
>   - `trusted_peers` is an allowlist and `expect_peer` a single expected id. Any other peer is refused.
>   - There is no shared-token mode.
> - **Records** (`SecureSocket`): AEAD records under the handshake's per-direction keys (TRANSPORT-I7).

**§2 Control channel, replace** "Admission is `token_mac(token, nonce)` against a challenge." **with:**

> It runs the control-domain handshake and is sealed before any message (TRANSPORT-I9). Admission is
> `token_mac(token, transcript)` inside the sealed channel.

**§3 Provides, replace** "`authenticate(sock, key_path, shared_token, trusted_peers)`" **with**
"`authenticate(sock, key_path, trusted_peers, expect_peer, aes_hw, domain, role)`, `SecureSocket`, `hardware_aes`,
`control_dial`, `control_accept`", **and** "`token_mac`, `check_mac`" **with** "`token_mac`, `session_mac`, `check_mac`,
`check_session_mac`".

**TRANSPORT-I5, replace with:**

> The handshake signs, per role, the hash of the plane's domain, both keys, both challenges and both X25519
> ephemerals. A reflected challenge, a signature under another domain, and non-complementary declared roles are
> refused. The allowlist or the expected peer refuses any other id.
>
> `peer_id` is `sha256(pub)[:32]`. There is no shared-token mode.

**TRANSPORT-I7, drop `(DEFERRED-TO: M4)` and replace with:**

> The data plane is sealed after the handshake:
> - with AES-256-GCM when both ends have hardware AES (AES-NI or ARMv8), else ChaCha20-Poly1305;
> - under per-direction keys from X25519 signed by the ed25519 identities (HKDF-SHA256 over the transcript);
> - with 4-byte prefix plus 64-bit counter nonces that never repeat under a key;
> - with a deterministic rekey after `2^24` records or `2^36` bytes.
>
> Tampered, truncated, replayed or reordered records are fatal. The payload bytes delivered to the runtime are
> unchanged. Per-core throughput is at least the link rate. The measured seal-plus-open time is under 1% of the step
> (SIMULATED under `gpu: false`).

**Add:**

> **TRANSPORT-I9.** The control channel runs the control-domain handshake with the coordinator in the coordinator role,
> and is sealed with its keys before any message. The worker's token MAC and session MAC use distinct domains over the
> transcript and are validated as 64 lowercase hex. No token-derived value proves the coordinator.

**Tasks 1–2:** replace them with the above (task 2 moved into M2; see Q1).

### A10 — SPEC-scheduler

- SCHEDULER-I13 and I15: drop `(DEFERRED-TO: M2)`.
- Task 1 **appends:**

  > `solve()` uses the measured budget as given (no 1F1B floor); refuse only below one microbatch, naming the bytes.

- Task 2 **appends:**

  > `replan(spec, windows, min_gain=0.10)`; the threshold is A7's rule.

### Ported tests (the mechanism they guard is replaced; each keeps its assertions on the new input)

| Today (phase 2) | After | Task | Property kept |
|---|---|---|---|
| `test_resume_refuses_a_shard_its_sidecar_does_not_name` | `test_resume_refuses_a_wave_whose_manifest_names_another_step` | `wave-resume` | refused before any state is applied; names the step |
| `test_resume_refuses_a_shard_without_a_sidecar` | `test_resume_refuses_a_wave_without_a_manifest` | `wave-resume` | same |
| `test_ship_never_labels_a_newer_shard_with_an_older_step` | `test_ship_labels_each_wave_from_its_manifest` | `wave-ship-restore` | a shipped label always names the shipped bytes |
| `test_concurrent_ships_never_cross_waves` | same name, Event-synchronized (`restore-hardening`), then on waves | both | two ships never cross (M2-P2-SHIP-TEST-TIMING) |
| `test_restore_rewrites_the_sidecar` | `test_restore_writes_the_wave_and_its_alias` | `wave-ship-restore` | the restored label names the restored bytes |
| `test_export_wave_writes_each_sidecar` | `test_export_wave_writes_each_wave_and_alias` | `wave-ship-restore` | same |
| `test_checkpoint_store_common_step_and_prune`, `test_corrupt_shard_is_refused` | same names, on wave directories | `wave-ship-restore` | every assertion kept |
| `test_checkpoint_writes_through_the_seqlock` | kept for the alias | `wave-stage-write` | seqlock-order assertion unchanged |

### Follow-ups for the human's re-grant (the tasks do not need them)

- **DESIGN §5 and §8:** the security items move from M4 to M2. **DESIGN §7:** the workstream provides JSON (the
  `authenticate`, `token_mac` and `patch_spec_for_resume` signatures). Editing DESIGN voids its grant.
- **THREAT_MODEL:**
  - T3: residual closed by RUNTIME-I16 and CONTROL-I22.
  - T4: AEAD.
  - T5: token required off loopback, and coordinator key pinned.
  - T7: CKPT identity closed.
  - T12: the coordinator proves its own key. Coordinator loss stays M4.

## 9. Risks

1. **Obsolete after the replan (§2a):** the experiment no longer gates the fix.
2. **The causal closure's length.** The pre-fix arm runs up to 60 unloaded fleet runs until one symptom (p0 ≈ 0.17 on
   the exploratory data, HYPOTHESIS for the tree before `wave-ship-restore`), the fixed arm 30, in resumable slices.
   If the pre-fix tree shows no symptom in 60 runs, `causal-pre-fix-has-symptom` fails loudly rather than passing.
3. **Ids before amendments.** If P1 is not recorded first, the invariants rung goes red at the first new `covers:` id.
4. **AEAD costs.** About +2 host-DRAM passes per byte per side, which M2b's S1 must count (ADR-0006 R10). GIL
   contention between the sender and reader threads and the compute thread is measured only on the CPU toy
   (SIMULATED). Real hardware is pending (M1).
5. **Disk per slot** rises to 3 waves. A control link slower than the save cadence becomes counted `ship gap`s, never
   silent ones.
6. **Rebalance estimates.** R is estimated until the run has measured one restore. A spurious rebalance costs time,
   never bits, and there are at most 2 per run.
7. **Compatibility breaks, all intended:**
   - the HMAC fallback is removed (`cryptography` is already required);
   - the key is per agent, not per host;
   - a non-loopback coordinator without a token refuses to start;
   - a worker needs `--coordinator-id` off loopback;
   - the checkpoint layout moves to `out/waves/`, with the old pair kept as a derived alias.
8. **Degrade changes schedules** only where today's floor over-committed memory.

## 10. Questions for the human (answer at the grant)

- **Q1.** The coordinator gets its own key, and workers pin its id (`--coordinator-id`). The focus asks for "the
  coordinator proving itself to workers". A token-derived proof would not hold against an admitted worker, so this
  plan derives the pinned key. Keep it, or accept the residual in T12?
- **Q2.** Obsolete: no 2e51b5b export is needed after the replan (§2a).
- **Q3.** Rebalance rule: is 10% predicted gain, more than 2× the restart cost, and at most 2 per run acceptable?
- **Q4.** The trust unit is the agent (a per-agent key) rather than the host. Confirm.
- **Q5.** Keep 3 local waves per slot?
- **Q6.** Obsolete: no verdict gates the redesign after the replan (§2a).

No ADR is opened. Each protocol and checkpoint change here implements the human's decisions of 2026-10-09 recorded in
the focus, and ADR-0006's granted RUNTIME-I12. The exception is Q1, which is put to you at this grant. If you
want an ADR on record for the wave unit or the control handshake, run `/wan-adr` before `/wan-implement`.

## Later slices

None. The whole open scope fits one DAG of 17 tasks (≤ 24).

## Review notes (round 1)

Every finding below was checked against the code on 6bff9f5 and fixed in both files. None was refuted.

- **DAG soundness: verdict-gated chain in the middle** (filed twice). Fixed in round 1 by moving the chain to the
  leaves. Round 2 reverses that order for stronger reasons (below): the chain now runs right after the experiment, and
  P3 states what a stop blocks.
- **The `shipped` line is not a contract.** Fixed. `SHIP_RE` and `GAP_RE` are module constants in
  `wave_experiment.py`, required verbatim by the experiment and emitted by `wave-ship-restore`, which tests them in
  `test_ship_logs_each_wave_in_the_experiment_format`.
- **AEAD `< 1%` of the step had no measurement** (blocker). Fixed. SecureSocket `stats()`, `WT_AEAD`, and
  `validation/aead_cost.py`, which is pinned in GATES. The bench rung is N/A with its reason (§7).
- **RUNTIME-I12 compatibility** (filed twice, once as a blocker). Fixed. The wave is the unit for CKPT, the store,
  RESTORE and resume. `wave.json` is the only step binding, the old pair is a derived alias, and resume refuses a
  missing manifest, one naming another step, or a mismatching or unlisted file (A1, A4, BA-3/5/9/14).
- **The A2 width pin contradicts RUNTIME-I12.** Fixed. The pin is `zero1_widths`, scoped to whole-tensor shards.
- **The verdict is vacuous when the symptom is rare.** Fixed. Each releasing verdict needs 3 or more symptomatic runs,
  else it is INCONCLUSIVE.
- **The token-derived proof is forgeable by an admitted worker.** Fixed. The coordinator has its own key, pinned by the
  workers (A6, BA-42, Q1).
- **Rung set incomplete.** Fixed. full, discipline, fmt, syntax, build and rest are added. redteam is N/A because
  `tests/redteam/` is absent.
- **HELLO `peer_pub` and WELCOME session were unprotected.** Fixed. The channel is sealed before any message, the peer
  id comes from the handshake, and the token MAC is bound to the transcript (BA-37, BA-40, BA-41).
- **The last step is committed twice.** Fixed (BA-4, `test_final_step_is_committed_once`,
  `test_last_step_is_announced_once`).
- **File-level causal closure is unrunnable.** Fixed. The closure compares whole trees.
- **The telemetry window spans attempts.** Fixed. `_launch` clears it (BA-23).

## Review notes (round 2)

Every finding below was checked against the code on 6bff9f5 and fixed in both files. None was refuted.

- **Blocker: `wave-stage-write` broke resume until `wave-ship-restore`.** Confirmed: `_restart_in_place` and `_heal`
  both restore through `_restore_and_relaunch` (`coordinator.py:1212-1244`), which writes only the legacy pair. Fixed
  by splitting the resume leg. `wave-stage-write` keeps resume on the alias; the new `wave-resume` task, after
  `wave-ship-restore`, switches resume to the wave of `resume_step` (BA-5, §5).
- **`--check` compared the live tree.** Fixed. Each arm snapshots the tree it measures and runs from the copy, and
  `--check` validates against the snapshot. The verdict gate is `--require-verdict SAMPLING,MIXED`, `wave-layout`'s
  first verify command, which stays true after its own edit.
- **The smoke criterion failed exactly under SAMPLING.** Fixed. It now requires one `seen` line and one `SHIP_RE` line
  per slot per run; per-wave completeness stays in `--expect-fixed`.
- **Chaos-gated tasks ran before the fix of the flake they hold.** Fixed by reordering. No task before
  `wave-ship-restore` runs the chaos rung or the dead-worker test (§5).
- **The experiment's deps were unused and confounded the measurement.** Fixed. It has no deps, runs alone in wave 1 on
  6bff9f5, and the roots of the other workstreams depend on it only for CPU isolation (§3).
- **`rebalance-loop` came before the wave interface it consumes.** Fixed. It follows `wave-ship-restore` through
  `slot-key-pinning`, and restores waves with `resume_step`.
- **The experiment ran late, on sealed trees.** Fixed with the reordering above.
- **The chain had no silent-socket or multi-stream protection.** Confirmed at `stage.py:259-264`. Fixed. The chain
  acceptor follows the mesh rule (BA-26), with `test_chain_silent_socket_does_not_hold_a_real_peers_slot`,
  `test_chain_intruder_is_refused_with_two_streams` and the matching GATES arms.
- **CKPT_PART identity and staging bounds had no test.** Fixed. `wave-ship-restore` carries four tests that would fail
  on the message-trusting or unbounded mutant (BA-47, A5).
- **Amended specs would contradict themselves.** Fixed. §8 now replaces SPEC-transport §1, §2 Identity, §2 Control
  channel and §3; SPEC-control §2.1 step 1, §2.3 step 5 and §3; SPEC-runtime §2 Ownership and §3. DESIGN §7 is added to
  the follow-ups.
- **The M2-P2 restore minors were verdict-gated.** Fixed. The new `restore-hardening` task delivers all four, plus
  SHIP-TEST-TIMING, on the legacy pair before the gate. `wave-ship-restore` keeps every property on waves.
- **The rebalance restored a wave the abort had pruned.** Confirmed: ABORT ships newer waves and the CKPT intake prunes
  older ones (`coordinator.py:394-395`). Fixed. BA-24 re-reads `common_step` after `_abort_all`, as `_heal` does, and
  `test_rebalance_restores_the_common_wave_after_the_abort` covers it.
- **The rebalance had no execution context.** Confirmed: one reader thread per connection (`coordinator.py:330-336`).
  Fixed by BA-45. The rule is evaluated and applied only on `_monitor`, and
  `test_rebalance_and_a_heal_in_one_window_make_one_attempt` covers it.
- **Found while fixing these.** A launch from step 0 would keep the old timeline's waves, because a repeated commit
  writes nothing (BA-6 now drops them). The alias is a hard link, so an in-place writer would corrupt a wave (BA-46).
  A step lost after both slots shipped it fit no verdict; it is now OTHER, which stops the chain. The experiment's
  hour-long run broke the 10-minute shell limit in LESSONS; it now runs in resumable slices.
- The findings list for this revision was cut off inside the execution-context finding. Any later finding was not seen
  and remains for the next review.
