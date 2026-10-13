# SPEC-runtime history

- 2026-10-08 — ADR-0002-m2-p1-spec (M2 phase 1, liveness and telemetry). Task 1: telemetry for re-planning is one
  `WT_TELEM` line per step instead of the `TELEMETRY` file. New invariant RUNTIME-I9. Text copied from
  docs/design/NEXT-m2-p1-liveness-telemetry.md §Spec amendments and §Invariants.
- 2026-10-09 — M2 phase 2 (NEXT-m2-p2-wave-label-binding §Mechanism, Spec amendment; spec conflict, not an ADR).
  §Checkpoint, Resume: a shard whose sidecar is missing or names another step is fatal, like a missing shard. Text
  copied verbatim from that document.
- 2026-10-09 — ADR-0006 (exact sharded stages; awaiting the human grant). Task 5 added (M2b). New invariants
  RUNTIME-I10, I11 and I12 (DEFERRED-TO: M2b). Text copied from ADR-0006 §First-Principles Derivation and §Validation Plan.
- 2026-10-09 — ADR-0006 round 1 (review findings; awaiting the human grant). RUNTIME-I10: fp16 and fp32 recipes, frame
  index and duplicate checks, per-unit step tags. RUNTIME-I11: pieces at `lcm(row length, 128)` boundaries replace
  element ranges; the accumulator is an fp32 copy of the widened first piece. RUNTIME-I12: pieces. Task 5 rewritten:
  two steps (`wire: native`, `shard: pieces`), working-copy dtype from the recipe, per-unit exchange, no credits (W
  never waits on the network, §2 unchanged), host-memory holds with `HostHoldExceeded`.
- 2026-10-09 — ADR-0006 round 2 (review findings; awaiting the human grant). RUNTIME-I10: both emission orders,
  per-unit updates, frames of a step other than the receiver's current or next are fatal. New RUNTIME-I13
  (DEFERRED-TO: M2b): updates run in the first pump after a fold completes; F waits only on its unit's copy; barrier
  kept at checkpoint, re-plan and last steps; host hold plus backlog bounded by one step; no blocking host↔device copy
  on the compute thread. Task 5: `flush_iter(store, order)`, asynchronous chunked copies on a copy stream, the PARAMS
  barrier removed, `HostHoldExceeded` and its heal-resume removed (the one-step bound is charged at admission).
  §2 Dependencies: a sharded stage's F of step t+1 also needs its unit's step-t working copy.
- 2026-10-09 — ADR-0006 round 3 (review findings; mechanism frozen; awaiting the human grant). RUNTIME-I10: weighted
  IWRR. RUNTIME-I12: a logical checkpoint reads into `shard: zero1` as `names[k::w]`, so `shard` rolls back. RUNTIME-I13:
  host bound per peer `b_g·P_s·[M_k·(w−1)/w + (M−M_k)/w]`; own pieces wait on the device, at most `b_g·M_k·P_s/w`;
  per-destination send queues served need-first in ≤ 16 MiB chunks (the round-1 priority, now written). Task 5: step 1
  sends through the need-first queues; own pieces stay on the device; `{wire: fp32, shard: pieces}` refused; the
  identity-manifest change and its working-copy digest check apply only under `pieces`.
- 2026-10-09 — ADR-0006 round 4 (review findings; mechanism frozen; awaiting the human grant). Task 5: the Embed's
  tables are one exchange unit, first in forward order, whose fp32 gradients (produced in B) ship when W(mb) starts;
  the Head's norm rides with the Head's linear. Previously unspecified; the round-4 cost model depends on it.
- 2026-10-09 — ADR-0006 round 5 (review findings; mechanism frozen; awaiting the human grant). RUNTIME-I13: the host
  bound covers gathers too (queued to send, or received and awaiting H2D): `+ 2·b_p·P_s·(w−1)/w`, one step per unit
  each way; the round-3 proof covered gradient pieces only, and a step-t gather to a slow destination can coexist with
  step-(t+1) gradient pieces. `b_g·P_s` and `b_p·P_s` are per-unit sums (the Embed's fp32 tables at 4 bytes); the
  own-piece device hold is `Σ_u b_g,u·M_k·P_u/w`. Derivation: ADR-0006 `round5.md` §4.
- 2026-10-09 — ADR-0006, merge with M2 phase 2 (main ec39441; awaiting the human grant). RUNTIME-I12: the logical
  manifest is the wave's sidecar and binds every piece file to its step; a resume at any width refuses a piece file
  whose manifest is missing or names another step, as §Checkpoint now does for a slot's shard. Reconciles ADR-0006
  item 5 with NEXT-m2-p2-wave-label-binding's spec amendment; no other text changed.
- 2026-10-09 — M2b handout SC-1 (HANDOUT-M2b.md; awaiting the human grant with the plan). RUNTIME-I12: the manifest
  is embedded in every piece file and names the step, the wave's width, the slot's peer index and the layout; the
  wave's sidecar names the same step and the file's sha256 binds both; a resume reads exactly the files of that step
  and width, peer indices 0..w−1. Was: "the manifest is the wave's sidecar"; the agent's RESTORE rewrites the sidecar
  with the step only (`agent.py:663-686`), which would destroy a manifest kept there.
- 2026-10-09 — ADR-0007 (slice-only build; awaiting the human grant). New RUNTIME-I14 (DEFERRED-TO: M4): a slot builds
  `build_slice` with its device as home within one working layer of host memory, a sharded-stage peer trims to its
  pieces and casts its step-0 working copy locally, a resume never draws (skeleton, `to_empty`, load), and a slot
  refuses a foreign init class before its first draw. Task 4 rewritten: `JobSpec.build` and `init_class`, the manifest's
  names from the skeleton, the reference unchanged.
- 2026-10-09 — ADR-0007 round 1 (review findings; awaiting the human grant). RUNTIME-I14 rewritten: the slice build
  takes `JobSpec.init_record`, trims each owned layer at its store, checks the sampler fingerprint before its first draw
  (`INIT-CLASS`), refuses a missing record (`INIT-RECORD`), validates the record at the boundary; `run_local` makes one
  in-process; the reference checks its build against the record; a resume refuses a wave without the RNG state or a
  tensor. Task 4: `init_record` replaces `init_class`; the manifest records the record's sha256 and fingerprint.
- 2026-10-09 — ADR-0007 round 2 (review findings; awaiting the human grant). RUNTIME-I14: the host bound adds the held tied table;
  pieces are trimmed when each home tensor is finalized; `setup_determinism` before the first kernel dispatch; a record
  whose binding is not the spec's raises `INIT-RECORD`; the manifest records the binding; `run_local` makes the record
  in a fresh process; the reference checks the ledger states at every boundary as well as the digests. Task 4 follows.
- 2026-10-09 — ADR-0007 round 3 (review findings; awaiting the human grant; no mechanism changed). RUNTIME-I14: the
  build's host bound adds the slice, `4·P_s`, when the slot's device is the CPU; a resume's host memory is stated (the
  wave as `torch.load` brings it, `12·P_s`, plus the slice on a CPU device) and charged by SCHEDULER-I18; `build:
  "full"` is stated not to roll back the draw-free resume or the refusal of an RNG-less wave (ADR-0007 R8).
- 2026-10-09 — ADR-0007 round 4 (review findings; awaiting the human grant; no mechanism added). RUNTIME-I14: the
  resume's skeleton slice is re-tied from the skeleton's sharing map after `to_empty` (which unties shared parameters,
  MEASURED); `arch="user"` resumes as today, full build then the wave's tensors and RNG state, because a user module may
  hold state outside `state_dict()`; the slot compares no fingerprint on resume, and which node may heal is CONTROL-I19's.
  Task 4 follows.
- 2026-10-09 — ADR-0007 round 5 (review findings; awaiting the human grant; no mechanism changed). RUNTIME-I14: the
  constant 4 MiB record cap is replaced by a cap derived from the config's ledger length and tensor names (a 405B LoRA
  record in hex came within 3% of 4 MiB); states are base64; a slot's `INIT-LEDGER` report says whether it is at the
  anchor (the record's fault) or at a transition. Task 4: the identity manifest takes names from today's full build
  under `arch="user"`, which has no skeleton.
- 2026-10-09 — ADR-0007 human decisions 5 and 6 (awaiting the human grant). RUNTIME-I14: `INIT-RECORD` compares the
  binding of (cfg, seed, program id, code id); the draw-free resume and the RNG-less refusal roll back by code revert,
  with no `resume` field.
- 2026-10-09 — Spec conflict `resume-ownership` (M2; amend-spec, not an ADR). HANDOUT-M2 §8 A2 applied with one
  correction: `steps` is removed from the recipe pins. Evidence: `steps` is consumed only as the loop bound
  (stage.py:1064, 1089; reference.py:44; coordinator.py:1111 for liveness); no LR schedule or other update reads the
  total step count (grep for lr_sched/cosine/LambdaLR/total_steps in src/wantrain: none); the data RNG is keyed by
  (seed, step, mb) (data.py:31-32). So a run cut at step k is a prefix of the same recipe with more steps, and RUNTIME-I3's
  existing tests (test_engine_bitwise.py:88-93 resumes steps 3 -> 6; tests/oracle/test_state_identity.py:58-61 resumes
  2 -> 4) rely on extending `steps`; pinning it contradicted the task's own "existing tests pass unmodified". The guard
  `steps` would have given is kept where it matters: resume refuses `steps` below checkpoint step + 1, because the
  unconditional final checkpoint (stage.py:1089) would otherwise write the resumed state under an earlier step label.
  A future field that couples an update to the total step count (an LR schedule horizon) becomes a pin when introduced.
  RUNTIME-I15 states the upward `steps` change; the oracle is that of the resumed spec's recipe.
- 2026-10-09 — Spec conflict `resume-ownership` #2 (M2; amend-spec, not an ADR). RUNTIME-I15 listed `devices` as
  freely changeable at a resume, which contradicts DESIGN.md §3 ("bit-comparable to the oracle on the same device map";
  mixed backends are not cross-backend identical), spec.py:46-49 and gpt.py:39-44 (`sdpa_backend: auto` resolves to
  math off CUDA, efficient/flash on CUDA). A CPU slot healed onto CUDA cannot stay byte-identical. Amendment: the state
  file pins `device_types` (per stage, sorted distinct torch device types of its slots, from `devices` or `device`),
  so a resume may change `devices` only within those types and is refused otherwise, naming the field; I15's identity
  claim is scoped to runs whose every slot has the oracle's (`device`'s) type. Per-stage sorted sets, not per slot,
  so the pin survives M2b's width-free logical checkpoints. Rejected parts of the implementer's proposal: a separate
  resolved-`sdpa_backend` pin (redundant: `auto` is a pure function of device type and `compute_dtype`, both pinned)
  and a GPU-name pin (it would refuse heals onto a different card model, a fault-tolerance semantics change that needs
  an ADR; the product rule already bans architecture-dependent kernels, and cross-model identity is real-hardware
  validation, N/A while gpu is false). Not an ADR: DESIGN.md already makes the device map part of the run's identity
  ("Every peer is the reference's device model, also after a heal"); this aligns the invariant with it. The pins()
  docstring must drop "never moves a bit" for `devices`. Follow-up (control, not this task): `find_replacement` should
  prefer spares of the dead slot's device type so a heal is not refused at resume.
- 2026-10-09 — HANDOUT-M2 §8 (P1) recorded on top of ADR-0007; M2 ids CONTROL-I19->CONTROL-I23, RUNTIME-I14->RUNTIME-I17 (collision with ADR-0007). The human's rule: main (ADR-0007, approved) wins, the M2 side yields and the renumbering is a
  deviation; ADR-0007's RUNTIME-I14 ("A slot materializes only its slice") is untouched. Applied: **A1** as
  **RUNTIME-I17** (immutable waves `step-<n:08d>/` with `wave.json` last, alias, `WT_WAVE`, keep 3; resume reads only
  the wave of `resume_step` and refuses a missing/foreign/mismatching/unlisted file; drops newer waves), replacing in §2
  Checkpoint "A missing shard, or a shard whose sidecar is missing or names another step, is fatal"; "Resume loads the
  slot's own shard." is kept, and §4 gains a pointer line for RUNTIME-I17. **A3** as **RUNTIME-I16** (every data-plane
  socket authenticated and sealed; refusal rules, `WT_P2P_REFUSED`, `WT_AEAD`). Already present, not re-applied:
  **A2** (pins, ownership per attempt, Consumes `SecureSocket`/`hardware_aes`, RUNTIME-I15), recorded earlier by the
  `resume-ownership` conflicts with their two corrections (`steps` not a pin; `device_types` pin). Verbatim
  otherwise.
- 2026-10-09 — ADR-0008 round 0 (awaiting the human grant). §2 Oracle: aux hex per (step, mb, layer) recorded, the
  gradient from the block's injection. RUNTIME-I18 (new): MoE identity at every depth, order, recompute, ZeRO-1 and
  piece width and skew; the missing-gradient guard stays fatal; aux and counts recorded once from F, `MOE-REPLAY` on a
  rebuild mismatch; the manifest's `aux` section. Task 5: an MoE block's experts of one projection form one exchange
  unit. Task 6 (M2c).
- 2026-10-10 — ADR-0008 round 1 (awaiting the human grant). M2c → M4. §2 Oracle: the reference applies the
  selection-bias update after each AdamW step. RUNTIME-I18: every recipe row; the selection biases in the sha256
  state; the bias update's counts (every microbatch, every peer), its place after AdamW and before step t+1's F, a
  missing count vector fatal. Task 5: the count vector rides in the layer's small-tensor unit; the unit change is to
  M2b's built code (db3f4e0), MoE layers only. Task 6: the bias update and the `moe_bias` manifest section.
- 2026-10-10 — ADR-0008 round 2 (awaiting the human grant). §2 Optimizer: under ZeRO-1 a stage's ownership list is
  its trainable names, then its selection biases in layer order (dense lists unchanged). RUNTIME-I18: the selection
  bias is a stage tensor updated by its own rule at its owner; each microbatch's int64 count vector travels where a
  gradient travels (width 1 local; zero1 with the shard gradients to the owner, the bias in the owner's broadcast;
  pieces in an int64 lane of the small-tensor unit's GRADS frame, the bias in the unit's gather and RUNTIME-I12's
  logical checkpoint); frame checks for the lane (int64, shape, step/mb/sender, no negative count, zero1 sum
  `T_mb·k`). Round 1's "rides in its small-tensor unit" had no zero1 path. Task 5: the count lane, a 3-D expert tensor
  as its projection's unit under an HF-modules path. Task 6: the count transport.
- 2026-10-10 — ADR-0009 round 0 (awaiting the human grant). §2 Oracle: the HF oracle (vanilla `ForCausalLM.forward`)
  and the per-use fold of a parameter used by several modules, for every arch. New RUNTIME-I19 (HF units and tied
  tables byte-identical; determinism asserts at stage start; tied table hosted by the Head stage with `FWD`/`BWD`
  frames on the tied link; frozen tied tables not hosted). Task 7. §6: the decision.

- 2026-10-10 — ADR-0008 round 3 (awaiting the human grant; mechanism frozen). Task 5: under HF modules each 3-D
  expert tensor is one unit (`gate_up_proj`, `down_proj`: two per MoE layer), cut along dim 0 at
  `Q_t = lcm(trailing product, 128)`, its gradient produced in B in the master dtype and shipped when W(mb) starts,
  as the Embed's tables; a shared expert's `SplitLinear`s one unit each and the router weight in the small-tensor
  unit on both paths.
- 2026-10-11 — ADR-0009 round 1 (awaiting the human grant). §2 Oracle: the per-use fold removed; the HF oracle folds
  as every arch does (one `autograd.grad` per microbatch, index order). RUNTIME-I19: a tied table across stages keeps
  the stock order by the row-sparse pending fold on the Head stage; stage 0 holds the step's rows, checks the
  untouched rows of its embedding gradient are +0 (`TIED-SPARSE`), and sends the touched rows in a tied lane of its
  `FWD` frames; the Head sends the next step's rows in a tied lane of its `BWD` frames; intermediate stages forward the
  lanes on the chain's sockets (no tied link, no probe; RUNTIME-I16 unchanged). Task 7 and §6 rewritten to match.
- 2026-10-11 — ADR-0009 round 2 (awaiting the human grant). RUNTIME-I19: tied rows travel in `TIED` frames, not in
  lanes of `FWD`/`BWD` frames (`e_i` exists only after `FWD_i` left, and the next step's rows inside step-(t+1) `BWD`
  frames would wait for the F they feed: a deadlock); `TIED(t, i)` sent when stage 0's `B_i` ends, `TIED(t+1, k)` with
  `R_k` after the Head's AdamW, stage 0's `F_k` waiting for `R_0..R_k`, a resume sending them first, stages between
  forwarding each fragment as it arrives; the fingerprint "admitted" (CONTROL-I24); the oracle runs the units on the
  run's device map with its buffer record. Task 7 and the §6 tied decision match.
- 2026-10-11 — ADR-0009 round 3 (awaiting the human grant; mechanism frozen, text corrected). §2 Oracle: for
  `arch="hf"` the oracle runs the run's units in one process on the device map with the recorded buffers (round 2 left
  the round-1 text, vanilla HF with CPU-skeleton buffers, contradicting RUNTIME-I19 and task 7); vanilla HF is the
  gate's comparator only. RUNTIME-I19: piece width listed; the buffer record's carrier (JobSpec, state file, a file in
  every wave under RUNTIME-I17's unchanged format), its check at every load and no rebuild after the first launch;
  `TIED` routing specified at width 1 only, a plan with stage 0 or the Head wider refused (R14, open); frozen tied
  tables note that hf LoRA is refused. RUNTIME-I14: a resume also loads the hf buffers against the pin. Task 7 matches.
- 2026-10-11 — ADR-0009 round 4 (awaiting the human grant; mechanism frozen, text corrected). §2 Oracle: for
  `arch="hf"` on a map of one device class the oracle is the vanilla `ForCausalLM.forward` in one process (today's
  `reference.py` path), so the identity rung compares the units with HF's function at full size; the units in one
  process remain the oracle only on a mixed map one node holds (ADR-0009 R16), and any other map is refused. Round 3
  made the units the oracle everywhere, which put the mechanism under test on both sides of the comparison.
- 2026-10-11 — ADR-0009 round 5 (awaiting the human grant; mechanism frozen, text corrected). RUNTIME-I15: for
  `arch="hf"` a move to another device class (GPU model or CPU ATen capability) is refused (CONTROL-I24), since the
  oracle is per device map. RUNTIME-I19: a `TIED` frame received under a plan with no tied relay is fatal at receipt
  (validation at the boundary; before, only missing, duplicated, misshapen or out-of-order frames were). Task 7: every
  piece is reachable only through `arch="hf"`, and a rollback to SPEC-model task 4a reverts this task by id.
- 2026-10-11 — ADR-0008 round 7 (correction pass, autonomy). §2 Oracle: an HF MoE model's oracle carries the same
  SUBSTITUTE class swap and aux injection as the run (the HF-MoE seam rule), never HF's own block; MODEL-I8 proves the
  block. ADR-0009 round 6: RUNTIME-I15: every resume of an `arch="hf"` run re-checks decision 8 on the map it builds
  (CONTROL-I24). Task 7: each piece has its own rollback target (7a/7b to tied HF at S > 1 refused at planning; 7c/7d
  to SPEC-model 4a); the house defect of `runtime/determinism.py` (`os.environ.setdefault` lets a pre-set value win;
  `allow_tf32` never read back) becomes part of this task with its own test.
- 2026-10-11 — reconcile check of the ADR-0009 round 6 correction pass (independent reviewer, autonomy; item k).
  Task 7's determinism test: `WOULD-FAIL-IF` also names `allow_tf32` True after setup, matching ADR-0009's house item
  and `round6.md` §8 (before, only the pre-set `ATEN_CPU_CAPABILITY` clause was stated). No mechanism change.
- 2026-10-11 — M4 implement deviation (oracle-m4, BA-465). §2 state-file pins: BA-465 said "omit every `hf_*` field of a
  non-hf model" and "dense digests byte-identical to 3b0d2f8's", which cannot both hold: `hf_model` and
  `hf_model_sha256` were GPTConfig fields at 3b0d2f8 (`git show 3b0d2f8:src/wantrain/model/gpt.py`, lines 76-77). As
  implemented (`launch/local.py:31` `_PRE_M4_HF_FIELDS`, `:48-57` `recipe_model`; `runtime/spec.py:35` `_HF_FIELDS`,
  `:577-588` `_model_pins`): only the M4 `hf_*` fields are omitted for a non-hf model, and every `moe_*` field when
  `moe_experts == 0`; `hf_model`/`hf_model_sha256` stay.
- 2026-10-11 — the human's decision (~10:35, confirmed ~19:35): homogeneous fleets only until the heterogeneous track
  returns (CONTROL-I25). RUNTIME-I15's scope bullet replaced; its former text, verbatim:
  > type. A stage whose slots mix device types is lossless but not oracle-identical before or after a resume (DESIGN
  > §3); a resume there is allowed under the same pins and its result is claimed only against a run of the same map
  > and ownership.
  New text: `JobSpec.finalize` refuses a `devices` map spanning more than one device type (`MIXED-FLEET`), so a mixed
  map neither launches nor resumes; the class-level check (GPU model, CPU capability) is CONTROL-I25's.
