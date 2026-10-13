# SPEC-control history

- 2026-10-08 — ADR-0002-m2-p1-spec (M2 phase 1, liveness and telemetry). §2.1 monitor inputs and §2.4 agent: `PROGRESS`
  carries `src`; `TELEM_STEP` is the live per-step summary parsed from `WT_TELEM`, validated at the coordinator;
  `TELEMETRY` stays the file shipped at exit. §2.2 step 5: S4 judges only `WT_STEP` progress inside the attempt's step
  range, and quiet after every slot finished the final step is shutdown, bounded by `shutdown_limit_s`. New invariants
  CONTROL-I17 and CONTROL-I18. Text copied from docs/design/NEXT-m2-p1-liveness-telemetry.md §Spec amendments and
  §Invariants.
- 2026-10-08 — ADR-0003-m2-p1-progress-monotone. §2.2 step 5 and CONTROL-I17: S4 reads only strictly increasing
  `WT_STEP` progress (a stream such as 2, 3, 2, 3 no longer grows `progress` or keeps the stall clock alive).
- 2026-10-09 — ADR-0006 round 2 (exact sharded stages; awaiting the human grant). §2.3 heal step 2 and CONTROL-I9
  (DEFERRED-TO: M2b): a slot of a sharded stage is replaced only by the identical device model (`match="model"`), else
  no spare and a clean failure, because a sharded stage's bits require one device class (ADR-0006 (g)).
- 2026-10-09 — ADR-0007 (slice-only build; awaiting the human grant). New CONTROL-I19 (DEFERRED-TO: M4): `HELLO`
  carries the init class, the coordinator writes it into `JobSpec.init_class` at a first launch, a heal before the first
  wave takes only a replacement of that class, and `PROFILE` and `FIT` build slices.
- 2026-10-09 — ADR-0007 round 1 (review findings; awaiting the human grant). CONTROL-I19 rewritten: `HELLO` carries the
  sampler fingerprint; `JobSpec.init_record` is the recipe's, never derived from the selection, and its absence refuses a
  slice launch (`INIT-RECORD`); a slot failing a class, ledger or digest check is excluded and selection reruns; heal
  before the first wave within the record's fingerprint; `PROFILE` and `FIT` from the skeleton with a private-generator
  fill, no record.
- 2026-10-09 — ADR-0007 round 2 (review findings; awaiting the human grant). CONTROL-I19: the `HELLO` fingerprint comes from a fresh
  child process pinned by `setup_determinism`, never from the agent's unpinned `probe.measure`; a record whose binding
  is not the job's is refused (`INIT-RECORD`); one exclusion rule, applied by selection and `find_replacement` from
  `HELLO` and extended by slot reports, replaces the special heal rule before the first wave.
- 2026-10-09 — ADR-0007 round 4 (review findings; awaiting the human grant; no mechanism added). CONTROL-I19: "after
  the first wave no fingerprint is compared" overclaimed. Init excludes no node after the first wave, but a CPU-device
  slot computes with its CPU class (ADR-0007 R9, open), so in a run with a record its replacement must have the
  record's fingerprint, the same comparison the exclusion rule makes; a device slot heals under CONTROL-I9 alone.
  `PROFILE` and `FIT`: re-tied skeleton slice, their host memory stated, `arch="user"` keeps the full build.
- 2026-10-09 — ADR-0007 round 5 (review findings; awaiting the human grant; no mechanism changed). CONTROL-I19: the
  exclusion rule applies to `INIT-CLASS`, an `INIT-LEDGER` at a transition and `INIT-DIGEST`; an `INIT-RECORD` or an
  anchor `INIT-LEDGER` is the record's fault, equal on every node, and refuses the job once instead of excluding nodes
  until the pool is exhausted and the refusal names the wrong cause.
- 2026-10-09 — conflict wave-layout (M2). §2.3 adds **wave retention**. The plan said a commit "removes committed waves
  beyond the newest keep", but it did not say what happens when a commit is older than the slot's newest wave. RESTORE
  makes such commits: wave-ship-restore commits the common wave into a slot that may still hold waves of the abandoned
  timeline, and BA-6 drops those only at the resume. Read literally, the rule deletes the restored wave (restore 2 into
  4, 5, 6 leaves 4, 5, 6). The code exempted only the step just committed, so a later commit could prune the restored
  wave. Ruling: keep is counted over the waves at or below the committed step; newer waves are left to
  `drop_waves_after`. The implementer's alternative, which refuses an older commit and drops at RESTORE, was rejected:
  the coordinator store uses the same `commit_received_wave`, and after a heal from common step c a survivor's replay
  legitimately ships steps c+1.. that are older than the store's newest wave for that slot. The amendment changes no
  bytes and no protocol; in the ascending case it is the plan's rule unchanged.
- 2026-10-09 — conflict wave-layout (M2), second ruling. §2.3 adds the **wave cap**. The retention rule of the first
  ruling never removes waves above the committed step, and `commit_received_wave` takes the step from the sender's
  manifest, so a worker that ships descending steps (N, N-1, ...) grows a coordinator slot by one shard per message
  without bound (THREAT_MODEL: workers may be adversarial; the coordinator is the single point of failure, T12). The
  old `CheckpointStore.prune` had `MAX_WAVES_AHEAD`; the wave path lost it. Ruling: a commit of a step the slot does
  not hold is refused, before anything is written, when the slot holds `MAX_WAVES_AHEAD` (4) or more step directories
  above the step or `keep + MAX_WAVES_AHEAD` or more in total; a re-commit of a held step is never refused by the cap.
  Rejected: evicting the abandoned-timeline waves on a lower commit (contradicts BA-6, which removes them only at the
  resume, and could delete waves an aborted RESTORE still needs); a per-connection monotone-step rule at the
  coordinator only (leaves the agent path and the library unbounded, and the store legitimately sees replays below its
  newest wave, first ruling). The cap removes nothing and changes no bytes of any committed wave; it bounds a slot to
  `keep + MAX_WAVES_AHEAD` shards. Cost: one listdir of at most 7 entries per commit, once per checkpoint wave, off the
  per-step path (HYPOTHESIS: under 0.01% of the step time; a checkpoint save already costs a full shard write).
  HYPOTHESIS, to be falsified by the tests: no honest flow (ascending saves, one RESTORE below the newest waves, a
  survivor's replay at the store) reaches the cap; if one does, it is reported as a conflict.
- 2026-10-09 — HANDOUT-M2 §8 (P1) recorded on top of ADR-0007; M2 ids CONTROL-I19->CONTROL-I23, RUNTIME-I14->RUNTIME-I17 (collision with ADR-0007). The human's rule: main (ADR-0007, approved) wins, the M2 side yields and the renumbering is a
  deviation; ADR-0007's CONTROL-I19 ("The init record and the init class") is untouched. Applied: **A4** — §2.4 ships
  every committed wave in ascending order (`WT_WAVE` + backstop scan, `CKPT_PART` then `CKPT`, never the alias); §2.3
  steps 4 (RESTORE the wave, each hash checked) and 5 (`resume_step` set to that wave; `patch_spec_for_resume` changes
  only endpoints, devices, slot keys, `resume_step`, and ownership for a rebalance); **CONTROL-I23** (HANDOUT's
  CONTROL-I19 text, renumbered); CONTROL-I8 append (wave of another step or mismatching file refused). **A5** —
  **CONTROL-I20**. **A6** — CONTROL-I16 replaced (coordinator ed25519 key, control-domain handshake, sealed channel,
  pinned `--coordinator-id`, transcript-bound `token_mac`, token required off loopback, `session_mac` reconnect);
  §2.1 step 1; §3 Consumes (`control_dial`, `control_accept`, `session_mac`); §3 Provides
  `patch_spec_for_resume(spec_json, endpoints, devices, slot_peers=None, resume_step=None, ownership=None)`. **A7** —
  task 2's "more than X% over the prediction" replaced by the rule (`T_new ≤ 0.90·T_cur`, gain `> 2·R`, at most 2 per
  run, at a fresh common wave); the sentence was reordered to stay grammatical ("Re-solve ... when ..."); **CONTROL-I21**,
  whose "under that rule" reads "under the rule of task 2" because the invariant is not next to the rule. **A8** —
  **CONTROL-I22**; CONTROL-I8 "except addresses and devices" replaced (slot keys, `resume_step`, ownership for a
  rebalance only). No A-item text contradicts an ADR-0006/0007 sentence; ADR-0007's CONTROL-I19 rule that a heal before
  the first wave relaunches from step 0 is the no-wave case of step 5 and is kept as written.
- 2026-10-10 — ADR-0009 round 0 (awaiting the human grant). §2.1 step 4: the `GATE` request and fingerprint comparison
  before placement for `arch="hf"`, and the tied link's probe. New CONTROL-I24 (gate passed and fingerprint equal on
  every node, recorded in the run and wave sidecars, checked at heal, restart, rebalance and resume). Task 6.
- 2026-10-11 — ADR-0009 round 1 (awaiting the human grant). §2.1 step 4: the gate also runs on each slot's device
  that is not the CPU; the tied link's probe removed (tied lanes ride the chain). Task 6: the probe removed.
- 2026-10-11 — ADR-0009 round 2 (awaiting the human grant). §2.1 step 4: fingerprints "admitted", the coordinator
  builds and sends the buffer record. CONTROL-I24: admission per package (transformers' RECORD equal everywhere; torch's
  version and `git_version` equal and each node's RECORD equal to its platform's `hfm.TORCH_RECORDS` entry), so
  mixed-platform fleets (CONTROL-I5) are admitted; each slot's torch RECORD and the buffer record recorded, and a heal,
  restart, rebalance or resume needs the slot's torch RECORD.
- 2026-10-11 — ADR-0009 round 3 (awaiting the human grant; mechanism frozen, text corrected). §1: the control plane
  never touches numerics except the HF buffer record, built once at the first launch. §2.1 step 4: the record is built
  in a child process under SPEC-model §2's environment guard with an admitted `hfm.fingerprint` (recorded), pinned in
  the JobSpec, never rebuilt by a resume, heal, restart or rebalance (round 2 left the builder unguarded and cited
  MODEL-I9, which defines no construction). CONTROL-I24: the builder's fingerprint recorded; the record's bytes in the
  JobSpec and each wave (RUNTIME-I19), not a coordinator-written sidecar; a map whose device classes no host holds is
  unverifiable (R11, open).
  Task 6: the fingerprints go into the run record, not the wave sidecar (RUNTIME-I17's `wave.json` format is fixed);
  the buffer record is built once under the guard and pinned in the JobSpec.
- 2026-10-11 — ADR-0009 round 4 (awaiting the human grant; mechanism frozen, text corrected). CONTROL-I24: a device
  map whose classes (device type and model, a CPU's ATen capability) no single node of the pool holds, as the gate
  reports list them, is refused before placement, naming the classes. Round 3 trained it with no reference; ADR-0009
  now recommends the refusal (decision 8), which the autonomy switch takes.
- 2026-10-11 — ADR-0009 round 5 (awaiting the human grant; mechanism frozen, text corrected). CONTROL-I24: the run
  records each slot's device class, and a heal, restart, rebalance or resume admits a replacement only of that class,
  so decision 8's refusal of a map with no oracle holds after placement too. Round 4 refused such a map only before
  placement, while `find_replacement` falls back from the identical model to the region, so an H100 slot could heal
  onto an A100 and leave the run on a map no node can reproduce. §2.3 step 2 states the class filter for `arch="hf"`.
- 2026-10-11 — ADR-0009 round 6 (correction pass, autonomy). CONTROL-I24: decision 8's refusal is re-checked at every
  admission (first launch, heal, restart, rebalance, resume) on the map that admission builds, from the current
  pool's gate reports; a replacement is admitted only after its gate, fingerprint, torch RECORD, native set, device
  class and that re-check pass (gate, fingerprint and RECORD alone never admit). Each node's native set (MODEL-I10)
  equals that of every node of the same (platform tag, device class). §2.3 step 2 states the re-check for heals.
  Task 6: the re-check and its rollback target (SPEC-model 4a; never reverted while 4b is live).
- 2026-10-11 — the human's decision (~10:35, confirmed ~19:35, `.factory/pending-orchestrator-notes.txt`; CLAUDE.md on
  main, PR #54): the oracle holds per device map (ADR-0005), and for now only homogeneous fleets (one device class)
  are admitted; every mixed fleet is refused with a clear reason, for every arch, at launch admission, in the planner
  and at every heal, restart, rebalance and resume, until the heterogeneous track returns. This settles ADR-0009
  decision 8's recorded conflict and DESIGN §10 question 7. New CONTROL-I25 (next free id; CONTROL-I24 was the
  highest) with the `MIXED-FLEET:` reasons and its test; §2.1 steps 1-2 and §2.3 heal step 2 point to it;
  CONTROL-I5 SUSPENDED (its text kept); CONTROL-I24's "Nodes of different platforms (a CPU+MPS fleet, CONTROL-I5) are
  admitted together." now reads "Nodes of different platforms are admitted together only within one device class
  (CONTROL-I25)."; task 7 added. Code state at 92261ca: no `MIXED-FLEET` refusal exists (`grep -rn MIXED src/wantrain`
  finds none) and `NodeInfo` (`resources/node.py:46-72`) has no `device_class` field: this is work for the code.
