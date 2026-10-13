export const meta = {
  name: 'wan-redteam',
  description: 'Assume the system is wrong and prove it: enumerate failure-mechanism classes against the design, write one susceptibility test suite per class (tests/redteam/<class>/, never product code), run them solo, have 2 refuters verify each failure is real, record findings, and optionally hand confirmed ones to wan-bughunt. Tests stay as a permanent "redteam" ladder rung.',
  whenToUse: 'After wan-implement, before closeout, and periodically. args: {"autofix": true} to run wan-bughunt on up to 3 confirmed findings.',
  phases: [
    { title: 'Gate', detail: 'Design approved' },
    { title: 'Classes', detail: 'Researcher maps mechanism classes onto the design and code; CLASSES.md' },
    { title: 'Tests', detail: 'One red-team writer per class, disjoint dirs, inside a scope audit (production code untouchable), no runs' },
    { title: 'Run', detail: 'Solo: run every class suite; susceptible / resistant / broken-test' },
    { title: 'Triage', detail: '2 refuters per susceptible class; FINDINGS.md; optional wan-bughunt' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project
// @include gates
// @include ledger
// @include campaign

const OUT = 'agents-architetura/docs/redteam'
const MAX_CLASSES = 16
const MAX_AUTOFIX = 3
const SEED_CLASSES = `
divergence from the oracle under message reordering or duplication; nondeterministic reduction order; numeric overflow,
underflow or NaN propagation through compression or aggregation; stale parameters or optimizer state after rejoin;
lost or doubled work across checkpoint/restore; deadlock or livelock on worker churn; unbounded queues/backpressure
collapse on a slow link; partial or torn writes on the wire; version skew between processes; malformed or adversarial
updates (if the trust model admits them); resource exhaustion (memory, file descriptors, threads); timeouts that turn
into silent drops`

phase('Gate')
await requireApprovedDesign('Gate')

phase('Classes')
const classesOut = await agent(common(`OWNED PATHS: ${OUT}/CLASSES.md (only).`) + `
You may use web search for public post-mortems and known failure classes of distributed training systems.
Map failure-MECHANISM classes onto THIS system: read ${DESIGN}, the specs and the code. Seed list (extend, merge, drop what
cannot apply, and justify drops): ${SEED_CLASSES}
For each class: id (short slug), the mechanism, where in the design/code it could bite (sections, file paths), and the
concrete test that would demonstrate susceptibility (what to inject, what correct behavior to assert). Write CLASSES.md.
At most ${MAX_CLASSES} classes, most dangerous first.`,
  { label: 'classes', phase: 'Classes', schema: CLASSES, agentType: ROLE.researcher })
const seen = new Set()
const classes = (classesOut?.classes ?? [])
  .map(c => ({ ...c, id: token(c.id, 40) }))
  .filter(c => c.id !== 'item' && !seen.has(c.id) && seen.add(c.id))
  .slice(0, MAX_CLASSES)
if (!classes.length) throw new Error('no failure classes produced — nothing to test')
const scopeOf = c => [`tests/redteam/${c.id}/`]
const disjoint = validatePlan(classes.map(c => ({ id: c.id, deps: [], scope: scopeOf(c) })))
if (!disjoint.ok) throw new Error(`class ids do not map to disjoint test directories: ${disjoint.errors.join('; ')}`)
log(`Classes: ${classes.map(c => c.id).join(', ')}`)

phase('Tests')
const TREE_RE = /^[0-9a-f]{40,64}$/
const solo = 'You run ALONE — no other agent is active; you MAY run commands.'
const snap = await agent(common(solo) + `
Run exactly: node agents-architetura/tools/wf/scope-audit.mjs snapshot — report the "tree" it prints verbatim. Do nothing else.`,
  { label: 'rt-audit:snapshot', phase: 'Tests', schema: AUDIT, agentType: ROLE.verifier })
if (!TREE_RE.test(snap?.tree ?? '')) throw new Error('scope-audit snapshot returned no tree id')
const written = await parallel(classes.map(c => () =>
  agent(common(`OWNED PATHS: ${scopeOf(c)[0]} (only). Product code is READ-ONLY for you. Write code only — no runs.`) + `
Write the susceptibility test suite for failure class "${c.id}": ${fence('failure_class', c, 4000)}
Each test injects the condition and asserts the CORRECT behavior the design promises (so the test FAILS if the system is
susceptible). Deterministic and seeded; reuse tests/chaos fixtures where they exist; follow the project's test conventions.
No expected-failure markers: a susceptible system must turn the "redteam" ladder rung red.`,
    { label: `rt-test:${c.id}`, phase: 'Tests', schema: RESULT, agentType: ROLE.redteam })))
const rtScopes = classes.flatMap(scopeOf)
const audit = await agent(common(solo) + `
Run exactly: node agents-architetura/tools/wf/scope-audit.mjs check --since ${snap.tree} --scopes '${JSON.stringify(rtScopes)}'
Exit 0 = no violations, 1 = violations listed. Return its JSON verbatim. Do nothing else.`,
  { label: 'rt-audit:check', phase: 'Tests', schema: AUDIT, agentType: ROLE.verifier })
if (!audit || !Array.isArray(audit.violations)) throw new Error('red-team scope audit produced no result')
if (audit.violations.length) throw new Error(`red team modified files outside tests/redteam/: ${audit.violations.join(', ')} — production code must not change`)
const testedClasses = classes.filter((c, i) => written[i])
for (const c of classes.filter((c, i) => !written[i])) log(`Tests: writer for ${c.id} produced nothing — class skipped`)
if (!testedClasses.length) throw new Error('no susceptibility suites were written')

phase('Run')
const run = await agent(common(`You run ALONE — you MAY build and run anything (no git).`) + `
Run each class suite separately with the project's test runner (one directory per class under tests/redteam/), and report per
class: susceptible (tests fail because the system misbehaves), resistant (all pass), or broken-test (the test itself
errors, e.g. a fixture bug). Paste the exact failing output as evidence and give a runnable reproCommand per class.
Classes: ${JSON.stringify(testedClasses.map(c => c.id))}`,
  { label: 'rt-run', phase: 'Run', schema: REDTEAM_RUN, agentType: ROLE.verifier })
const resultOf = id => (run?.results ?? []).find(r => token(r.id, 40) === id)
const outcomes = testedClasses.map(c => ({ c, r: resultOf(c.id) }))
const unreported = outcomes.filter(o => !o.r).map(o => o.c.id)
const resistant = outcomes.filter(o => o.r?.outcome === 'resistant').map(o => o.c.id)
const broken = outcomes.filter(o => o.r?.outcome === 'broken-test').map(o => o.c.id)
const susceptible = outcomes.filter(o => o.r?.outcome === 'susceptible')

phase('Triage')
const judged = await parallel(susceptible.map(({ c, r }) => () =>
  parallel([0, 1].map(n => () =>
    agent(common('READ-ONLY refuter.') + `
A susceptibility test for class "${c.id}" failed. Try to REFUTE that this is a real defect: is the test sound, does it assert
behavior the design actually promises, is the failure caused by the system rather than the test or the environment?
Read the test (tests/redteam/${c.id}/), the code, and ${DESIGN}.
Class: ${fence('failure_class', c, 3000)}
Failure: ${fence('failure_output', r.evidence, 6000)}`,
      { label: `rt-refute:${c.id}:${n}`, phase: 'Triage', schema: VERDICT, agentType: ROLE.reviewer })))
    .then(vs => ({ c, r, verdicts: vs.filter(Boolean) }))))
// Triage: both refuters fail to refute -> confirmed (fix); both refute -> false positive (rejected with their evidence);
// anything else -> uncertain: a human decides (recorded as a needs-human finding, which blocks VERIFIED).
const confirmed = judged.filter(j => j && j.verdicts.length === 2 && j.verdicts.every(v => v.verdict !== 'refuted'))
const falsePositives = judged.filter(j => j && j.verdicts.length === 2 && j.verdicts.every(v => v.verdict === 'refuted'))
const uncertain = judged.filter(j => j && !confirmed.includes(j) && !falsePositives.includes(j))
const disputed = falsePositives.map(j => j.c.id)
const needsHuman = uncertain.map(j => j.c.id)
if (confirmed.length || uncertain.length) {
  await agent(common('You run ALONE — run this one command and report its output verbatim in "summary"; change nothing else.') + `
${recordCommand('findings', { findings: [
    ...confirmed.map(j => ({ id: `RT-${j.c.id}`.slice(0, 64), status: 'open', title: clean(j.c.mechanism, 200), source: 'wan-redteam' })),
    ...uncertain.map(j => ({ id: `RT-${j.c.id}`.slice(0, 64), status: 'needs-human', title: clean(j.c.mechanism, 200), source: 'wan-redteam', why: clean(j.verdicts.map(v => `${v.verdict}: ${v.argument}`).join(' | '), 600) })),
  ] })}`,
    { label: 'rt-record', phase: 'Triage', schema: RESULT, agentType: ROLE.verifier })
}

await agent(common(`OWNED PATHS: ${OUT}/FINDINGS.md (only).`) + `
Write ${OUT}/FINDINGS.md: for each confirmed finding the class, mechanism, failing test, reproCommand and evidence; then the findings
that need a HUMAN decision (the refuters disagreed — give both arguments); then the false positives with the refuters' evidence, and the
broken-test, unreported and resistant classes (one line each). No recommendations beyond the evidence.
${fence('redteam_results', {
    confirmed: confirmed.map(j => ({ id: j.c.id, mechanism: j.c.mechanism, repro: j.r.reproCommand, evidence: clean(j.r.evidence, 1500) })),
    needsHuman: uncertain.map(j => ({ id: j.c.id, verdicts: j.verdicts.map(v => ({ verdict: v.verdict, argument: clean(v.argument, 600) })) })),
    disputed, broken, unreported, resistant,
  }, 20000)}`,
  { label: 'rt-findings', phase: 'Triage', schema: RESULT, agentType: ROLE.researcher })

const bughunts = []
if (args?.autofix === true) {
  for (const j of confirmed.slice(0, MAX_AUTOFIX)) {
    const r = await workflow('wan-bughunt', { bug: `redteam class ${j.c.id}: ${clean(j.c.mechanism, 600)}`, id: `rt-${j.c.id}`, repro: clean(j.r.reproCommand, 500) })
    bughunts.push({ id: j.c.id, solved: r?.solved === true })
  }
}

log(`Redteam: ${confirmed.length} confirmed, ${needsHuman.length} need a human, ${disputed.length} false positives, ${broken.length} broken tests, ${resistant.length} resistant, ${unreported.length} unreported`)
return {
  ok: !confirmed.length && !uncertain.length && !broken.length && !unreported.length && bughunts.every(b => b.solved),
  confirmed: confirmed.map(j => j.c.id),
  needsHuman,
  disputed,
  broken,
  unreported,
  resistant,
  bughunts,
  findings: `${OUT}/FINDINGS.md`,
}
