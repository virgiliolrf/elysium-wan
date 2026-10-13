import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, finding, planTask, assertRoles, approved, servePlan } from './helpers.mjs'

const script = built('wan-handout')
const ok = () => ({ summary: 'ok', files: [] })
const WS = [{ key: 'transport', title: 't', scope: 's', ownedPaths: ['src/transport/', 'tests/unit/transport/'] }]
const task = planTask
const GOOD = { milestone: 'M0', tasks: [task('wire'), task('peer', { deps: ['wire'] })] }
const BAD = { milestone: 'M0', tasks: [task('wire'), task('peer', { verify: [] })] }
const GATES = { milestone: 'M0', consecutive: 2, rungs: ['factory', 'build', 'syntax', 'discipline', 'invariants', 'claims', 'full', 'unit'], pinned: [{ id: 'smoke', cmd: 'uv run pytest -q tests/unit', repeat: 20 }], campaigns: [], fuzz: [] }
const base = {
  'gate:approval': () => approved(),
  'extract:workstreams': () => ({ workstreams: WS }),
  'draft': ok, 'repair:': ok, 'revise:': ok,
  'extract:': servePlan(GOOD),
  'review:': () => ({ findings: [] }),
  'final-check': () => ({ items: [] }),
  'gates': ok,
  'extract:gates:': () => GATES,
  'validate:gates:': () => ({ json: { ok: true, errors: [] }, exitCode: 0 }),
}

test('requires a well-formed milestone arg', async () => {
  for (const a of [undefined, {}, { milestone: 'm0' }, { milestone: 'M0; rm' }])
    await assert.rejects(simulate(script, { args: a, respond: byLabel(base) }), /args.milestone is required/)
})

test('valid plan: waves returned, plan file paths are milestone-scoped', async () => {
  const { result, calls } = await simulate(script, { args: { milestone: 'M0' }, respond: byLabel(base) })
  assert.equal(result.ok, true)
  assert.deepEqual(result.waves, [['wire'], ['peer']])
  assert.equal(result.plan, 'docs/design/PLAN-M0.json')
  assert.ok(calls.find(c => c.opts.label === 'draft').prompt.includes('"workstream":"transport"'))
})

test('a plan the script rejects (task without verify commands) is repaired before review', async () => {
  const { result, calls } = await simulate(script, {
    args: { milestone: 'M0' },
    respond: byLabel({ ...base, 'extract:': servePlan((p, o) => (o.label.startsWith('extract:draft:a1') ? BAD : GOOD)) }),
  })
  assert.equal(result.ok, true)
  const repair = calls.find(c => c.opts.label === 'repair:draft:a1')
  assert.ok(repair.prompt.includes('peer: no verification commands'))
  assertRoles(assert, calls, { 'draft': 'wan-researcher', 'repair:': 'wan-researcher', 'review:': 'wan-reviewer', 'extract:': 'wan-readonly' })
})

test('a plan that never validates aborts', async () => {
  await assert.rejects(
    simulate(script, { args: { milestone: 'M0' }, respond: byLabel({ ...base, 'extract:': servePlan(BAD) }) }),
    /plan still invalid after 3 repairs/,
  )
})

test('a review revision is re-validated; breaking the plan aborts', async () => {
  let reviews = 0
  const r = byLabel({
    ...base,
    'review:': () => (reviews++ < 3 ? { findings: [finding('blocker')] } : { findings: [] }),
    'extract:': servePlan((p, o) => (o.label.startsWith('extract:r1') ? BAD : GOOD)),
  })
  await assert.rejects(simulate(script, { args: { milestone: 'M0' }, respond: r }), /plan invalid after review round 1/)
})

test('the plan is reviewed twice at most and every validated finding is fixed, minors included', async () => {
  const minors = Array.from({ length: 10 }, (_, i) => finding('minor', { location: `docs/design/PLAN-M0.json:${i}` }))
  const { result, calls } = await simulate(script, {
    args: { milestone: 'M0' },
    respond: byLabel({ ...base, 'review:': () => ({ findings: [finding('major'), ...minors] }) }),
  })
  assert.equal(result.ok, true, 'two rounds that fix everything do not block the grant')
  assert.deepEqual([result.review.converged, result.review.rounds], [false, 2])
  assert.equal(calls.filter(c => c.opts.label.startsWith('review:dag-soundness:')).length, 2)
  const revisions = calls.filter(c => c.opts.label.startsWith('revise:'))
  assert.equal(revisions.length, 2, 'the second round\'s findings are fixed too')
  assert.ok(revisions[0].prompt.includes('docs/design/PLAN-M0.json:9') && revisions[0].prompt.includes('fix EVERY real'))
  assert.match(result.next, /approve\.mjs grant docs\/design\/HANDOUT-M0\.md docs\/design\/PLAN-M0\.json docs\/design\/GATES-M0\.json/)
})

test('one plan for one run: every open phase in one DAG; past 24 tasks the planner is told to slice', async () => {
  const big = { milestone: 'M0', tasks: Array.from({ length: 25 }, (_, i) => task(`t${i}`)) }
  let n = 0
  const { result, calls } = await simulate(script, {
    args: { milestone: 'M0', focus: 'the rest of M2: rebalancing, degrade, plan mode' },
    respond: byLabel({ ...base, 'extract:': servePlan((p, o) => (o.label.startsWith('extract:draft:a1') ? big : GOOD)) }),
  })
  assert.equal(result.ok, true)
  const draft = calls.find(c => c.opts.label === 'draft').prompt
  assert.ok(draft.includes('as ONE DAG') && draft.includes('the rest of M2: rebalancing, degrade, plan mode'))
  assert.ok(calls.find(c => c.opts.label === 'repair:draft:a1').prompt.includes('25 tasks: more than one run takes (24)'))
})

test('gates: the milestone bar is frozen, shape-checked by the script and validated by pinned.mjs', async () => {
  const { result, calls } = await simulate(script, { args: { milestone: 'M0' }, respond: byLabel(base) })
  assert.equal(result.gates, 'docs/design/GATES-M0.json')
  assert.deepEqual(result.gateErrors, [])
  assert.ok(calls.find(c => c.opts.label === 'validate:gates:a1').prompt.includes('node agents-architetura/tools/gates/pinned.mjs M0 --validate'))
  assert.ok(calls.find(c => c.opts.label === 'gates').prompt.includes('consecutive >= 2'))
  assertRoles(assert, calls, { 'gates': 'wan-researcher', 'extract:gates:': 'wan-readonly', 'validate:gates:': 'wan-verifier' })
})

test('gates: a bar missing a rung the plan names is repaired; still invalid after 3 repairs fails the handout', async () => {
  const weak = { ...GATES, consecutive: 1, rungs: ['factory'] }
  const repaired = await simulate(script, { args: { milestone: 'M0' }, respond: byLabel({ ...base, 'extract:gates:': (p, o, i, n) => (n === 0 ? weak : GATES) }) })
  assert.equal(repaired.result.ok, true)
  const fix = repaired.calls.find(c => c.opts.label === 'repair:gates:a1')
  assert.ok(fix.prompt.includes('rung unit is required'))
  assert.ok(fix.prompt.includes('consecutive must be 2..10'))
  const never = await simulate(script, { args: { milestone: 'M0' }, respond: byLabel({ ...base, 'validate:gates:': () => ({ json: { ok: false, errors: ['unknown campaign "ghost"'] }, exitCode: 2 }) }) })
  assert.equal(never.result.ok, false)
  assert.deepEqual(never.result.gateErrors, ['unknown campaign "ghost"'])
})

test('the human\'s scope reaches the planner, the reviewers and the revisions whole; a focus over 8000 characters is refused', async () => {
  // M2's single plan (2026-10-09): a 2117-character focus was cut to 600 and the reviewers never saw it
  const focus = 'SCOPE-START ' + 'x'.repeat(2100) + ' (5) TRANSPORT-I7 AES-256-GCM SCOPE-END'
  const { calls } = await simulate(script, {
    args: { milestone: 'M0', focus },
    respond: byLabel({ ...base, 'review:': (p, o, i, n) => ({ findings: n === 0 ? [finding('major')] : [] }) }),
  })
  for (const label of ['draft', 'review:spec-coverage:r1', 'revise:r1']) {
    const prompt = calls.find(c => c.opts.label === label).prompt
    assert.ok(prompt.includes('SCOPE-START') && prompt.includes('SCOPE-END'), `${label} sees the whole scope`)
  }
  await assert.rejects(simulate(script, { args: { milestone: 'M0', focus: 'y'.repeat(8001) }, respond: byLabel(base) }), /args\.focus is 8001 characters; at most 8000/)
})

test('a milestone split off by a granted ADR carries a letter (M2b): its plan, handout and gates are named after it', async () => {
  const { result, calls } = await simulate(script, {
    args: { milestone: 'M2b' },
    respond: byLabel({ ...base, 'extract:gates:': () => ({ ...GATES, milestone: 'M2b' }) }),
  })
  assert.equal(result.ok, true, JSON.stringify(result.gateErrors))
  assert.deepEqual([result.handout, result.plan, result.gates], ['docs/design/HANDOUT-M2b.md', 'docs/design/PLAN-M2b.json', 'docs/design/GATES-M2b.json'])
  assert.ok(calls.find(c => c.opts.label === 'draft').prompt.includes('milestone M2b'))
  for (const bad of ['M2bb', 'M2B', 'm2b']) await assert.rejects(simulate(script, { args: { milestone: bad }, respond: byLabel(base) }), /must look like "M0" or "M2b"/)
})

test('fixes see the earlier rounds; after the last fix one agent checks and fixes every finding, then the script validates', async () => {
  // M2 (2026-10-09): round 1 condemned a vacuous verdict rule and security coupled to the wave chain; the unreviewed fix after
  // round 2 re-coupled everything to the experiment, and the build stopped on its first task
  const f1 = finding('major', { location: 'docs/design/PLAN-M0.json:305', problem: 'the verdict rule passes whatever the truth' })
  const f2 = finding('major', { location: 'docs/design/PLAN-M0.json:12', problem: 'security sits behind the wave chain' })
  const { result, calls } = await simulate(script, {
    args: { milestone: 'M0' },
    respond: byLabel({ ...base, 'review:dag-soundness:': (p, o) => ({ findings: [o.label.endsWith('r1') ? f1 : f2] }), 'final-check': () => ({ items: [{ n: 1, status: 'fixed-now', evidence: 'PLAN:305' }, { n: 2, status: 'resolved', evidence: 'PLAN:12' }] }) }),
  })
  assert.equal(result.ok, true)
  const r2 = calls.find(c => c.opts.label === 'revise:r2').prompt
  assert.ok(r2.includes('must not bring any of them back') && r2.includes('the verdict rule passes whatever the truth'), 'the round-2 fix sees round 1')
  const fc = calls.find(c => c.opts.label === 'final-check')
  assert.ok(fc.prompt.includes('the verdict rule passes whatever the truth') && fc.prompt.includes('security sits behind the wave chain'), 'the final check sees every round')
  assert.equal(fc.opts.agentType, 'wan-researcher')
  const order = calls.map(c => c.opts.label)
  assert.ok(order.indexOf('final-check') > order.indexOf('revise:r2') && order.some((l, i) => i > order.indexOf('final-check') && l.startsWith('extract:final')), 'validated after the final check')
  assert.deepEqual([result.review.finalCheck, result.review.open], [{ resolved: 1, 'fixed-now': 1, refuted: 0, 'still-open': 0 }, []])
  // a finding the check does not account for stays open
  const silent = await simulate(script, { args: { milestone: 'M0' }, respond: byLabel({ ...base, 'review:dag-soundness:': () => ({ findings: [f1] }), 'final-check': () => ({ items: [{ n: 1, status: 'resolved', evidence: 'x' }] }) }) })
  assert.equal(silent.result.review.finalCheck['still-open'], 1)
  assert.equal(silent.result.review.open.length, 1)
})

test('at most 2 waves: a chained plan is sent back to the planner with the rule and the files that chain it', async () => {
  const DEEP = { milestone: 'M0', tasks: [task('wire'), task('peer', { deps: ['wire'] }), task('loop', { deps: ['peer'] })] }
  const { result, calls } = await simulate(script, {
    args: { milestone: 'M0' },
    respond: byLabel({ ...base, 'extract:': servePlan((p, o) => (o.label.startsWith('extract:draft:a1') ? DEEP : GOOD)) }),
  })
  assert.equal(result.ok, true)
  assert.ok(calls.find(c => c.opts.label === 'draft').prompt.includes('AT MOST 2 WAVES'), 'the planner is told up front')
  assert.ok(calls.find(c => c.opts.label === 'repair:draft:a1').prompt.includes('the plan runs in 3 waves (at most 2): wave 3: loop'))
})
