# ADR-0005: M2 phase 1 measurement bounds: HOP_RATE_RATIO_MULTI >= 2.0; TELEM_US_MAX reported, TELEM_FRAC bounds the cost

<!-- Approval lives in docs/design/APPROVALS.json (agents-architetura/tools/factory/approve.mjs), bound to this exact text. Do not write a
     Status line claiming approval. Mark the ADR "Status: REJECTED", "Status: SUPERSEDED" or "Status: WITHDRAWN" only
     when that is the human's decision. -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE DECISION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: M2 phase 1 (run wf_8f18b5aa-df4, published 8d3c101) ended NOT VERIFIED on two measurement bounds only. Byte
identity (oracle, identity), liveness, and the telemetry's function all passed.
  (a) HOP_RATE_RATIO_MULTI = 2.42-2.48 against >= 2.5. The multi-stream arm emulates links of 40 and 160 Mb/s (ratio 4).
      The emulation's fixed per-frame cost, outside this phase's owned paths, dilutes the measured ratio. Stabilize
      found no fix inside scope, and the voters rejected (3 to 0) a change to what the metric measures.
  (b) TELEM_US_MAX = 1374-1841 us against <= 1000 us, on a host running other work. TELEM_FRAC (the block's median cost
      over the median step) passed its <= 0.01 bound.
Current bottleneck: the bounds, not the mechanism.
Proposed architecture:
  (a) HOP_RATE_RATIO_MULTI >= 2.0. The arm exists to catch an encode-timed `send` (BA-5). Such a `send` measures encode
      time, which does not depend on the link, so its ratio is ~1. A correct one measures 2.4-4. A bound of 2.0 keeps
      that discrimination and leaves margin for the emulation's per-frame cost.
  (b) TELEM_US_MAX is reported, not bounded. An absolute microsecond bound measures the host's load. TELEM_FRAC <= 0.01
      bounds what matters: the cost relative to the step. It becomes the `telemetry-intake` gate's metric.
First-principles reasoning: a gate must fail on a wrong implementation and pass on a correct one on the machine that
runs it. (a) at 2.5 fails a correct implementation under this emulation. (b) at 1000 us fails a correct one on a loaded
host while its relative cost is in bounds.
Mathematical justification: with a fixed per-frame cost c and frame time b*8/B, the measured rate ratio is
(b*8/B_slow + c)/(b*8/B_fast + c), which is 4 at c = 0 and falls toward 1 as c grows. The measured 2.42-2.48 means c is
comparable to the fast link's frame time. An encode-timed `send` has a ratio ~1.
Expected improvement: the gate passes the correct implementation and still fails the two mistakes it was written for.
Alternatives: (1) keep 2.5 and change the link emulation: outside this phase's owned paths and a larger change;
(2) /wan-bughunt on TELEM_US_MAX: hours for a bound that measures the host; (3) open the telemetry log once instead of
per step: a real but separate improvement, recorded as a follow-up, not needed for this decision.
Risks: (a) a partially encode-timed `send` could measure between 1 and 2; the send-count == fragment-count check in the
same arm still catches a missing per-fragment `send`. (b) a cost regression shows in TELEM_FRAC, not in absolute us.
Falsification criteria: an encode-timed `send` (the BA-5 mistake) that passes HOP_RATE_RATIO_MULTI >= 2.0; a telemetry
block whose TELEM_FRAC exceeds 0.01 without failing `telemetry-intake`.
Local experiment: the phase's pinned gate, rerun on the current tree.
Estimated Vast.ai cost: none: local only.
Recommendation: PROCEED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Awaiting explicit human approval: node agents-architetura/tools/factory/approve.mjs grant docs/design/adr/ADR-0005-m2-p1-measurement-bounds.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Amendments (applied with this ADR)
- NEXT-m2-p1-liveness-telemetry §Exit gate: HOP_RATE_RATIO_MULTI bound >= 2.0; TELEM_US_MAX reported, not bounded.
- GATES-m2-p1-liveness-telemetry.json: `telemetry-engine` checks HOP_RATE_RATIO_MULTI >= 2.0; `telemetry-intake`'s
  metric is TELEM_FRAC <= 0.01 (its TELEM_INTAKE_US_MAX and OPS_CHECKED checks unchanged).
- tests/integration/test_engine_bitwise.py: `assert multi_ratio >= 2.0`, with a comment naming this ADR.
- Follow-up (not in this ADR): open the telemetry log once per run instead of per step (stage.py `_append_telemetry`).

## Human Approval
Granted only by `node agents-architetura/tools/factory/approve.mjs grant` in a human's terminal; recorded in docs/design/APPROVALS.json.
