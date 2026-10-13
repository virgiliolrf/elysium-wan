# ADR-0002: Spec amendments for M2 phase 1 (liveness across restarts and shutdown, live per-step telemetry)

<!-- Approval lives in docs/design/APPROVALS.json (agents-architetura/tools/factory/approve.mjs), bound to this exact text. Do not write a
     Status line claiming approval. Mark the ADR "Status: REJECTED", "Status: SUPERSEDED" or "Status: WITHDRAWN" only
     when that is the human's decision. -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE DECISION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: M2 (microbatch adaptivity) re-plans ownership from measured per-step costs. Today the coordinator has no live
per-step costs (telemetry arrives only as a file at exit), and its stall clock is fed by replayed loss lines after a
resume and treats a slow but normal shutdown as a stall. Phase 1 fixes both; its invariants need spec text first.
Current bottleneck: correctness of the control plane, not speed: a restarted attempt can be judged stalled or alive by
replayed lines, and a re-planner would have no runtime numbers to act on.
Proposed architecture: the amendments below, copied verbatim from docs/design/NEXT-m2-p1-liveness-telemetry.md
§Spec amendments and §Invariants (reviewed with that document). No new mechanism beyond that phase's §Mechanism.
First-principles reasoning: a liveness verdict must depend only on facts of the current attempt (its own WT_STEP inside
its step range); telemetry that steers a re-plan must come from the slot's monotonic clock and never touch training
state, so the run stays byte-identical to the reference.
Mathematical justification: the telemetry block costs TELEM_FRAC = median(emit_us) / median(step_ms) <= 0.01 of the step
(model <= 0.003), measured by the phase's pinned gates.
Expected improvement: no false stall after a resume or during shutdown; per-step F/B/W/OPT and per-hop costs at the
coordinator, the input phase 2 re-plans from.
Alternatives: (1) do nothing — phase 2 would re-plan from profiler numbers that underestimate by about 30% on CPU
(SPEC-runtime §5 task 1); (2) keep telemetry in the exit file — no live input for between-step re-planning.
Risks: telemetry on the step path (bounded by the gate above); coordinator intake of worker-supplied data (validated at
the boundary, identity from the connection: CONTROL-I18).
Falsification criteria: the phase's pinned gates (docs/design/GATES-m2-p1-liveness-telemetry.json): the causal
slow-shutdown arm must stall before the phase and pass after it; manifests must equal the reference; TELEM_FRAC above
0.01 or any differing bit refutes the amendment as written.
Local experiment: the phase itself, on CPU under gpu false (localhost clusters, emulated WAN).
Estimated Vast.ai cost: none: local only.
Recommendation: PROCEED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Awaiting explicit human approval: node agents-architetura/tools/factory/approve.mjs grant docs/design/adr/ADR-0002-m2-p1-spec.md docs/design/DESIGN.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Problem
See the summary above and docs/design/NEXT-m2-p1-liveness-telemetry.md §Objective and §Evidence.

## Amendments (applied to the specs, with history entries)
- SPEC-control §2.2 step 5, appended: "S4 judges only while some launched slot still owes an optimizer step: it reads
  only progress parsed from `WT_STEP` whose step lies in [the attempt's start step, `steps − 1`]; replayed `WT_METRIC`
  lines never feed it. Once every launched slot exited 0 or reported step `steps − 1` (or the attempt started at
  `steps`), the attempt is in shutdown: quiet is bounded by `shutdown_limit_s` (default 600), past which the verdict is
  failed with 'shutdown exceeded … after the final step'. S1–S3 and strikes still apply in shutdown."
- SPEC-control §2.1 monitor inputs and §2.4 agent: "`PROGRESS` carries `src` (`step` from `WT_STEP`, `metric` from
  `WT_METRIC`). `TELEM_STEP`: the live per-step summary parsed from `WT_TELEM`, validated at the coordinator
  (CONTROL-I18). `TELEMETRY` stays the telemetry file shipped at exit."
- SPEC-runtime §5 task 1: "in `TELEMETRY`" becomes "in one `WT_TELEM` line per step (RUNTIME-I9)".
- New invariants: CONTROL-I17, CONTROL-I18 (SPEC-control §4) and RUNTIME-I9 (SPEC-runtime §4), with the texts of the NEXT
  document's §Invariants. CONTROL-I18 names this ADR and the NEXT document's §Mechanism for the C2 intake rules.

## Migration Impact
Additive. `final_step=None` keeps today's liveness behavior exactly; a worker that sends no `WT_TELEM` changes nothing at
the coordinator. Roll back by reverting the phase and these spec lines.

## Validation Plan
docs/design/GATES-m2-p1-liveness-telemetry.json: rungs oracle, identity and full, plus the phase's pinned entries.

## Human Approval
Granted only by `node agents-architetura/tools/factory/approve.mjs grant` in a human's terminal; recorded in docs/design/APPROVALS.json.
