import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, planTask, assertRoles, approved, servePlan } from './helpers.mjs'

const script = built('wan-closeout')
const ok = () => ({ summary: 'ok', files: [] })
const TREE = 'f'.repeat(40)
const WS = [{ key: 'transport', title: 't', scope: 's', ownedPaths: ['src/transport/', 'tests/unit/transport/'] }]
const PLAN = { milestone: 'M0', tasks: [planTask('wire', { invariants: ['TRANSPORT-I1'] }), planTask('peer', { deps: ['wire'], rungs: ['oracle', 'chaos'] })] }
const GATES = {
  milestone: 'M0', consecutive: 2, rungs: ['factory', 'build', 'syntax', 'discipline', 'invariants', 'claims', 'full', 'unit', 'oracle', 'chaos'],
  pinned: [{ id: 'smoke', cmd: 'uv run pytest -q tests/chaos/test_smoke.py', repeat: 20 }], campaigns: [{ id: 'rust-tsan' }], fuzz: [{ id: 'fuzz-wire', minCpuHours: 4 }], audit: { bar: 'no-open-tier-a' }, mutation: { changedSince: 'auto' },
}
const gate = (id, kind, okv = true, extra = {}) => ({ id, kind, ok: okv, detail: okv ? 'ok' : 'boom', ...extra })
const verified = () => ({ pinned: { milestone: 'M0', verdict: 'VERIFIED', failing: [], gates: [gate('rung:unit', 'rung')], tree: TREE }, tree: TREE, pendingMarkers: [], benchRecorded: true })
const failed = gates => ({ pinned: { milestone: 'M0', verdict: 'NOT VERIFIED', failing: gates.map(g => g.id), gates, tree: TREE }, tree: TREE, pendingMarkers: [] })
const base = {
  'gate:approval': () => approved(),
  'extract:workstreams': () => ({ workstreams: WS }),
  'extract:plan': servePlan(PLAN),
  'extract:gates': () => GATES,
  'verify:': verified,
  'verify-wait:': () => ({ status: 'finished' }),
  'report': ok,
}
const run = (over = {}, args = { milestone: 'M0', review: { converged: true, rounds: 2, findingsPerRound: [2, 0] } }) => {
  const flows = []
  const wf = name => a => { flows.push([name, a]); return { ok: true } }
  return simulate(script, { args, respond: byLabel({ ...base, ...over }), workflows: { 'wan-fix': wf('wan-fix'), 'wan-campaign': wf('wan-campaign'), 'wan-fuzz': wf('wan-fuzz'), 'wan-audit': wf('wan-audit'), 'wan-bughunt': wf('wan-bughunt') } })
    .then(r => ({ ...r, flows }))
}

test('VERIFIED: the frozen gates run by pinned.mjs on the current tree; the report carries the verdict', async () => {
  const { result, calls } = await run()
  assert.equal(result.ok, true)
  assert.equal(result.verdict, 'VERIFIED')
  assert.ok(calls.find(c => c.opts.label === 'verify:r0').prompt.includes('node agents-architetura/tools/gates/pinned.mjs M0 --json'))
  assert.ok(calls.find(c => c.opts.label === 'report').prompt.includes('Status: VERIFIED'))
  assertRoles(assert, calls, { 'extract:gates': 'wan-readonly', 'verify:': 'wan-verifier', 'report': 'wan-researcher' })
})

test('stale evidence (pinned ran on another tree) is NOT VERIFIED even with a VERIFIED verdict', async () => {
  const stale = () => ({ ...verified(), tree: 'a'.repeat(40) })
  const { result } = await run({ 'verify:': stale })
  assert.equal(result.verdict, 'NOT VERIFIED')
  assert.match(result.why.join(), /stale evidence/)
})

test('each failing gate goes to the workflow that owns it, then the gate re-runs', async () => {
  let n = 0
  const gates = [gate('rung:chaos', 'rung', false), gate('campaign:rust-tsan', 'campaign', false), gate('fuzz:fuzz-wire', 'fuzz', false), gate('audit', 'audit', false), gate('mutation', 'mutation', false), gate('pinned:smoke', 'pinned', false, { cmd: 'uv run pytest -q tests/chaos/test_smoke.py' })]
  const { result, flows } = await run({ 'verify:': () => (n++ === 0 ? failed(gates) : verified()) })
  assert.equal(result.ok, true)
  assert.deepEqual(flows.map(f => f[0]), ['wan-fix', 'wan-campaign', 'wan-fuzz', 'wan-audit', 'wan-bughunt'])
  assert.deepEqual(flows[0][1], { rungs: ['chaos', 'mutation'] })
  assert.deepEqual(flows[1][1], { campaign: 'rust-tsan' })
  assert.deepEqual(flows[2][1], { campaign: 'fuzz-wire' })
  assert.deepEqual(flows[3][1], { close: true })
  assert.equal(flows[4][1].repro, 'uv run pytest -q tests/chaos/test_smoke.py')
})

test('a red second consecutive pass re-fixes every gate rung', async () => {
  let n = 0
  const { flows } = await run({ 'verify:': () => (n++ === 0 ? failed([gate('consecutive', 'rung', false)]) : verified()) })
  assert.deepEqual(flows[0][1].rungs, ['factory', 'build', 'syntax', 'discipline', 'invariants', 'claims', 'full', 'unit', 'oracle', 'chaos'])
})

test('missing or weak gates refuse to close; pending markers stop without a fix loop; non-converged review is NOT VERIFIED', async () => {
  await assert.rejects(run({ 'extract:gates': () => ({ milestone: '', consecutive: 0, rungs: [], pinned: [], campaigns: [], fuzz: [] }) }), /GATES-M0\.json is missing or invalid/)
  await assert.rejects(run({ 'extract:gates': () => ({ ...GATES, rungs: GATES.rungs.filter(r => r !== 'oracle') }) }), /rung oracle is required/)
  const pending = await run({ 'verify:': () => ({ ...verified(), pendingMarkers: ['tests/unit/transport/test_wire.py:12'] }) })
  assert.equal(pending.result.ok, false)
  assert.deepEqual(pending.flows, [])
  const review = await run({}, { milestone: 'M0', review: { converged: false, rounds: 4 } })
  assert.equal(review.result.verdict, 'NOT VERIFIED')
  assert.ok(review.calls.find(c => c.opts.label === 'report').prompt.includes('adversarial review did not converge'))
})

test('still failing after 4 stabilize rounds: NOT VERIFIED report', async () => {
  const { result, flows, calls } = await run({ 'verify:': () => failed([gate('rung:chaos', 'rung', false)]) })
  assert.equal(result.ok, false)
  assert.equal(flows.length, 4)
  assert.equal(calls.filter(c => /^verify:r\d+$/.test(c.opts.label)).length, 5)
  assert.ok(calls.find(c => c.opts.label === 'report').prompt.includes('Status: NOT VERIFIED (pinned verdict NOT VERIFIED (rung:chaos))'))
})
