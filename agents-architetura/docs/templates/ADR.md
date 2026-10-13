# ADR-NNNN: <decision title>

<!-- Approval lives in docs/design/APPROVALS.json (agents-architetura/tools/factory/approve.mjs), bound to this exact text. Do not write a
     Status line claiming approval. Mark the ADR "Status: REJECTED", "Status: SUPERSEDED" or "Status: WITHDRAWN" only
     when that is the human's decision. -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE DECISION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: …
Current bottleneck: …
Proposed architecture: …
First-principles reasoning: …
Mathematical justification: …
Expected improvement: …
Alternatives: …
Risks: …
Falsification criteria: …
Local experiment: …
Estimated Vast.ai cost: … (or "none: local only")
Recommendation: PROCEED / DO NOT PROCEED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Awaiting explicit human approval: node agents-architetura/tools/factory/approve.mjs grant <this file> [docs/design/DESIGN.md]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Problem

## Current Architecture
What exists today (docs/design/CURRENT_ARCHITECTURE.md), with file references.

## Current Bottleneck
Which term of the cost model dominates, and the measurement or derivation that shows it.

## Evidence
Measurements (command, hardware, network, result) and cited sources. Label anything unmeasured as a hypothesis.

## Hypothesis

## First-Principles Derivation
Derive the mechanism. "Commonly used", "a paper says it works" and "should scale" are not arguments.

## Mathematical Model
Start from the decomposition the proposal actually changes, for example
`T_total = T_compute + T_communication + T_synchronization + T_scheduler + T_overhead`,
with only the variables that matter here (N workers, B bandwidth, L latency, P parameter size, G gradient size,
C compute time, S synchronization time, W worker throughput, R recovery time, …). Say which term the proposal changes,
by how much, and under which conditions the change disappears. No decorative mathematics.

## Expected Improvement
Numbers with units and the conditions they hold under.

## Alternatives
At least two genuinely different options, including "do nothing", each with the evidence that would make it win.

## Trade-offs

## Risks

## Falsification Criteria
The experiment and the observed result that would prove this idea WRONG. Mandatory.

## Local Experiment
The smallest local prototype, the correctness check against the reference, the performance measurement, and the
failure injection that must pass before any paid compute.

## Vast.ai Experiment
Only after the local experiment validates: hardware, network profile, duration, estimated cost, maximum budget, and
the expected-vs-observed table that will classify the result (VALIDATED / PARTIALLY VALIDATED / REFUTED / INCONCLUSIVE).

## Migration Impact
What changes for existing code, data, checkpoints and users; how to roll back.

## Validation Plan
Tests, rungs and gates that prove the change once implemented.

## Recommendation

## Human Approval
Granted only by `node agents-architetura/tools/factory/approve.mjs grant` in a human's terminal; recorded in docs/design/APPROVALS.json.
