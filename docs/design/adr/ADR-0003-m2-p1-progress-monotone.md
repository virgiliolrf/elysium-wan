# ADR-0003: Liveness progress is strictly increasing; message identity for CKPT and RESULT is deferred

<!-- Approval lives in docs/design/APPROVALS.json (agents-architetura/tools/factory/approve.mjs), bound to this exact text. Do not write a
     Status line claiming approval. Mark the ADR "Status: REJECTED", "Status: SUPERSEDED" or "Status: WITHDRAWN" only
     when that is the human's decision. -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE DECISION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: the review of M2 phase 1 (wan-x-m2-p1-liveness-telemetry, run wf_8f18b5aa-df4) stopped on a spec conflict with
two validated findings against the granted contract (NEXT-m2-p1-liveness-telemetry, ADR-0002).
  (a) C1 / CONTROL-I17: `add_progress` accepts any step that differs from the last one. A buggy or hostile worker that
      sends 2, 3, 2, 3 grows `progress` without bound; each poll sorts it under `run.lock`, and `last_progress_at` keeps
      renewing, so S4 never fires.
  (b) `CKPT` and `RESULT` in `coordinator._dispatch` take rank and peer from the message (coordinator.py:372, 397-398)
      and `CKPT` does not check the attempt: an admitted worker can ship another slot's wave or forge the final RESULT.
Current bottleneck: correctness of the liveness judge (a) and message identity at the coordinator (b); not speed.
Proposed architecture:
  (a) ADOPT. `add_progress` records a step only if it is strictly greater than the last recorded one. An honest stream
      is strictly increasing within an attempt (optimizer steps are sequential), so nothing honest is lost, and
      `progress` holds at most `steps − start_step` entries per attempt.
  (b) DEFER to a later phase, recorded as finding M2-P1-CKPT-IDENTITY and as a residual of THREAT_MODEL T7. DESIGN
      already accepts that a malicious worker can corrupt a run undetected until M4 (audit wiring, CONTROL-I13); phase 1
      stays the size it was granted ("divide always").
First-principles reasoning: the judge must depend only on facts of the current attempt, and a fact that can be replayed
in a loop is not progress. A strictly increasing sequence bounded by `steps − 1` is finite, so `progress` is bounded and
quiet after the last real step is measured from that step.
Mathematical justification: with steps in [start_step, steps − 1] and strict increase, |progress| <= steps − start_step;
the per-message cost is one integer compare.
Expected improvement: S4 cannot be suppressed by a non-increasing stream; per-poll sort cost bounded.
Alternatives: (1) keep "differs from the last" — rejected, the 2, 3, 2, 3 stream above; (2) a set of seen steps — same
effect for honest input, more memory, and it still accepts an out-of-order step as progress; (3) fix (b) in this phase —
rejected for scope: it adds CKPT/RESULT paths, tests and gates to a phase already at the size limit.
Risks: (b) stays open until its phase: an admitted worker can inject a wave or a RESULT under another slot.
Falsification criteria: the C1 unit test sends 2, 3, 2, 3, 2 as `src="step"`: 2 and 3 are recorded once, and a later
quiet past the limit reaches S4. A judge that records a repeat or a lower step fails it.
Local experiment: the phase's own gates (CPU, gpu false).
Estimated Vast.ai cost: none: local only.
Recommendation: PROCEED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Awaiting explicit human approval: node agents-architetura/tools/factory/approve.mjs grant docs/design/adr/ADR-0003-m2-p1-progress-monotone.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Amendments (applied with this ADR)
- NEXT-m2-p1-liveness-telemetry §Mechanism C1: "`step` differs from the last recorded step" becomes "`step` is strictly
  greater than the last recorded step"; BA-12's ruling adds "strictly increasing"; the C1 test contract adds the
  2, 3, 2, 3, 2 stream; §Risks records the deferral of (b).
- SPEC-control §2.2 step 5 and CONTROL-I17: S4 reads only strictly increasing `WT_STEP` progress (history entry).
- THREAT_MODEL T7 residual: CKPT and RESULT identity (finding M2-P1-CKPT-IDENTITY).
- agents-architetura/docs/findings.json: M2-P1-CKPT-IDENTITY, open, major.
- The phase workflow's review and fix prompts name this ADR, so the resumed run reuses the cached waves while its review
  and fixes apply (a) and do not re-file (b).

## Migration Impact
None outside this phase: its code is not merged yet.

## Human Approval
Granted only by `node agents-architetura/tools/factory/approve.mjs grant` in a human's terminal; recorded in docs/design/APPROVALS.json.
