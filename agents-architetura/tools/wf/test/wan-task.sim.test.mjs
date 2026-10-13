import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, finding, approved, notApproved, assertRoles, publisher, payloadOf, publishLabels } from './helpers.mjs'

const script = built('wan-task')
const TASK = 'Train a 1B-parameter model across 8 consumer GPUs on 3 continents without losing accuracy.'
const ok = () => ({ summary: 'ok', files: [] })
const absent = () => notApproved('docs/design/DESIGN.md', 'document does not exist')
const base = {
  'state:approval': absent,
  'spec': ok, 'spec-fix:': ok, 'report': ok,
  'spec-review:': () => ({ findings: [] }),
  'spec:freeze': () => ({ json: {}, exitCode: 0 }),
  'extract:milestones': () => ({ milestones: ['M0', 'M1'] }),
  'publish': publisher(),
}
const flows = (over = {}) => {
  const order = []
  const wf = {
    'wan-survey': () => ({ ok: true, specialists: [] }),
    'wan-design': () => ({ ok: true, adr: 'docs/design/adr/ADR-0001-initial-architecture.md' }),
    'wan-prep': () => ({ ok: true }),
    'wan-cycle': a => ({ ok: true, verdict: 'VERIFIED', milestone: a.milestone }),
    'wan-scalebench': () => ({ ok: true, applicable: false, reason: 'no GPU' }),
    ...over,
  }
  return { order, workflows: Object.fromEntries(Object.entries(wf).map(([k, f]) => [k, a => { order.push([k, a]); return f(a) }])) }
}
const run = (extra = {}, args = { task: TASK }, f = flows()) => simulate(script, { args, workflows: f.workflows, respond: byLabel({ ...base, ...extra }) }).then(r => ({ ...r, order: f.order }))

test('fresh: survey -> specified, reviewed and frozen request -> design -> STOP for the human grant', async () => {
  const { result, calls, order } = await run({}, { task: TASK, reference: 'github:virgiliolrf/Elysium-Old' })
  assert.equal(result.ok, false)
  assert.equal(result.stoppedAt, 'approve')
  assert.equal(result.awaitingHuman, true)
  assert.equal(result.command, 'node agents-architetura/tools/factory/approve.mjs grant docs/design/DESIGN.md docs/design/adr/ADR-0001-initial-architecture.md')
  assert.deepEqual(order, [['wan-survey', { reference: 'github:virgiliolrf/Elysium-Old' }], ['wan-design', {}]])
  assert.ok(calls.find(c => c.opts.label === 'spec').prompt.includes('# Acceptance Criteria'))
  assert.match(calls.find(c => c.opts.label === 'spec:freeze').prompt, /freeze\.mjs freeze docs\/design\/BRIEF\.md/)
  assertRoles(assert, calls, { 'state:': 'wan-verifier', 'spec-review:': 'wan-reviewer', 'spec:freeze': 'wan-verifier', 'spec': 'wan-researcher' })
})

test('resume after the grant: prep -> every milestone VERIFIED -> scalability study -> implementation report', async () => {
  const { result, calls, order } = await run({ 'state:approval': () => approved() })
  assert.equal(result.ok, true)
  assert.deepEqual(order.map(o => o[0]), ['wan-prep', 'wan-cycle', 'wan-cycle', 'wan-scalebench'])
  const report = calls.find(c => c.opts.label === 'report').prompt
  for (const section of ['# Distributed Training Validation', '"SYSTEM WORKED"', '# Vast.ai Results', '# Acceptance Criteria']) assert.ok(report.includes(section), section)
})

test('a design that exists but is not approved stops before anything runs — silence is not approval', async () => {
  const { result, order } = await run({ 'state:approval': () => notApproved('docs/design/DESIGN.md', 'the document changed after its last approval — a human must approve the new text') })
  assert.equal(result.awaitingHuman, true)
  assert.match(result.why, /changed after its last approval/)
  assert.deepEqual(order, [])
  const re = await run({ 'state:approval': () => notApproved() }, { task: TASK, redesign: true })
  assert.deepEqual(re.order.map(o => o[0]), ['wan-survey', 'wan-design'])
})

test('a request spec that never converges or cannot be frozen stops before the design', async () => {
  const never = await run({ 'spec-review:': () => ({ findings: [finding('blocker')] }) })
  assert.equal(never.result.stoppedAt, 'spec')
  assert.ok(!never.order.some(o => o[0] === 'wan-design'))
  const unfrozen = await run({ 'spec:freeze': () => ({ json: {}, exitCode: 2 }) })
  assert.equal(unfrozen.result.stoppedAt, 'spec')
})

test('stops at the first failing stage; an invalid scalability study stops before the report', async () => {
  const survey = await run({}, { task: TASK }, flows({ 'wan-survey': () => ({ ok: false }) }))
  assert.equal(survey.result.stoppedAt, 'survey')
  const cycle = await run({ 'state:approval': () => approved() }, { task: TASK }, flows({ 'wan-cycle': a => ({ ok: a.milestone !== 'M0', stoppedAt: 'closeout' }) }))
  assert.equal(cycle.result.stoppedAt, 'cycle M0')
  const scale = await run({ 'state:approval': () => approved() }, { task: TASK }, flows({ 'wan-scalebench': () => ({ ok: false, applicable: true, valid: false }) }))
  assert.equal(scale.result.stoppedAt, 'scalebench')
  await assert.rejects(simulate(script, { args: { task: 'short' }, respond: () => null }), /args.task/)
})

test('publishing: one branch per task, a commit per stage (message from the result), every stop recorded, the timer stopped', async () => {
  const { calls, logs } = await run()
  assert.deepEqual(publishLabels(calls), ['publish:start', 'publish:watch', 'publish:survey', 'publish:spec', 'publish:design', 'publish:stopped-at-approve', 'publish:stop-watch'])
  const start = calls.find(c => c.opts.label === 'publish:start')
  assert.equal(Buffer.from(/--text-b64 (\S+)/.exec(start.prompt)[1], 'base64').toString('utf8'), TASK)
  assert.deepEqual(payloadOf(calls.find(c => c.opts.label === 'publish:survey').prompt), { ok: true })
  assert.deepEqual(payloadOf(calls.find(c => c.opts.label === 'publish:stopped-at-approve').prompt).awaitingHuman, true)
  assertRoles(assert, calls, { 'publish:': 'wan-verifier' })
  assert.ok(calls.filter(c => c.opts.label.startsWith('publish:')).every(c => /Never run git or gh yourself/.test(c.prompt)))
  assert.ok(logs.some(l => /publishing on factory\/demo/.test(l)))
  const resumed = await run({ 'state:approval': () => approved() }, { task: TASK, id: 'gpt-wan' })
  assert.match(resumed.calls.find(c => c.opts.label === 'publish:start').prompt, /publish\.mjs start --task gpt-wan$/m)
  assert.deepEqual(publishLabels(resumed.calls).slice(-4), ['publish:prep', 'publish:scalebench', 'publish:report', 'publish:stop-watch'])
})

test('publishing: a refusal (secret, dirty base, identity) stops the run; a push failure is logged and retried; publish:false is silent', async () => {
  await assert.rejects(run({ 'publish:survey': publisher(3) }), /publish refused at survey: refused to commit: src\/a\.py:1 \(GitHub token\)/)
  await assert.rejects(run({ 'publish:start': publisher(3) }), /publish refused at start/)
  const flaky = await run({ 'publish:survey': publisher(1) })
  assert.equal(flaky.result.stoppedAt, 'approve')
  assert.ok(flaky.logs.some(l => /publish: survey NOT published \(push failed/.test(l)))
  const off = await run({}, { task: TASK, publish: false })
  assert.equal(off.result.stoppedAt, 'approve')
  assert.deepEqual(publishLabels(off.calls), [])
  const offResumed = await run({ 'state:approval': () => approved() }, { task: TASK, publish: false })
  assert.deepEqual(offResumed.order.filter(o => o[0] === 'wan-cycle').map(o => o[1]), [{ milestone: 'M0', publish: false }, { milestone: 'M1', publish: false }])
  assert.deepEqual(publishLabels(offResumed.calls), [])
})
