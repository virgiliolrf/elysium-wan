export const meta = {
  name: 'wan-x-example',
  description: 'TEMPLATE for the phase workflows wan-next writes (agents-architetura/tools/wf/src/wan-x-<slug>.js). One objective from docs/design/NEXT-example.md, attacked with the facts of the current state: the files, numbers and invariant ids the generic workflows do not know.',
  whenToUse: 'After a human grants docs/design/NEXT-example.md, docs/design/GATES-example.json and this built workflow. Written by wan-next from the state it records; rerun wan-next for the phase after this one.',
  phases: [
    { title: 'Gate', detail: 'Refuse to run without the human grant of the NEXT document, its pinned gates and this built workflow' },
    { title: 'Implement', detail: 'Writers in waves, parallel on disjoint owned paths; nobody runs a command; every wave scope-audited' },
    { title: 'Review', detail: 'Adversarial lenses written for this objective, looped until zero blocker/major, <= 5 rounds' },
    { title: 'Verify', detail: 'Solo: pinned.mjs runs this phase\'s V0..Vn (GATES-example.json) and decides; the verifier relays it and files items' },
    { title: 'Stabilize', detail: 'Each failing gate: proposal -> 3 votes -> scoped apply, then Verify again; <= 4 rounds' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project
// @include gates
// @include triage
// @include fix
// @include scope
// @include phase

// A phase workflow is its facts, its writers, its lenses and its gate ids: the machinery (grant, scope audits, review,
// pinned verify, stabilize, post-stabilize review, intermittent and tree checks) is runPhase in lib/phase.js, written and
// reviewed once. wf-lint refuses a phase that calls agents or the loops itself, or that grows past its size budget.
//
// What makes a phase workflow better than a generic one (the reference PR's thread-*.js): its prompts carry the exact
// state. Copy the facts VERBATIM from docs/design/NEXT-example.md — paths, measured numbers with their source, invariant
// ids, the failing test ids — never a paraphrase. Keep them to what the prompts need: every agent can read the NEXT
// document itself.
const FACTS = `
FACTS OF THIS PHASE (from docs/design/NEXT-example.md; re-check each against its source before relying on it):
- objective: <one sentence>
- evidence: <fact> (source: <command output or file:line>)
- invariants: <ID-In, ...>
- binding audit: BA-1 <what> @ <file:line> -> <ruling>   (every item of the NEXT document's table)
- tripwire: <switch that raises until armed; its test; who arms it> | none: <reason>
- exit gate: docs/design/GATES-example.json — rungs <...>; pinned <id: what it proves, e.g. "step: step_time <= 1.05 x predicted, 20 runs">`

return await runPhase({
  slug: 'example',
  facts: FACTS,
  // One writer per disjoint owned path in a wave; waves run in order. `binding` names the BA items whose ruling it delivers.
  work: [
    { key: 'part-a', owns: ['src/example/a/'], task: '<the change, in terms of the facts above>', binding: ['BA-1'] },
  ],
  lenses: [
    ['objective', 'Does the change achieve the objective as the facts define it, with the numbers the exit gate names?'],
    ['soundness', 'Races, ordering, partial failure, resource lifetimes; the product rule (a differing bit is a blocker). Is every binding item\'s ruling delivered (grep for instances the audit missed), and does the tripwire still refuse a partial change?'],
    ['tests', 'Would the tests fail on a wrong implementation? Each test states covers: <id> and WOULD-FAIL-IF.'],
  ],
  // Every gate id pinned.mjs reports for docs/design/GATES-example.json; wf-lint checks this list against the file.
  gateIds: ['rung:oracle', 'rung:identity', 'rung:full', 'pinned:step'],
})
