import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { finding, built, ladder, assertRoles, approved, notApproved } from './helpers.mjs'

const script = built('wan-prep')
const ok = () => ({ summary: 'ok', files: [] })
const WS = [
  { key: 'transport', title: 't', scope: 's', ownedPaths: ['src/transport/', 'tests/unit/transport/'] },
  { key: 'train', title: 't', scope: 's', ownedPaths: ['src/train/'] },
]
const green = () => ({ allGreen: true, ladder: ladder(true), items: [] })
const base = {
  'gate:approval': () => approved(),
  'extract:workstreams': () => ({ workstreams: WS }),
  'spec:': ok, 'sizegate:': ok, 'spec-review:': () => ({ findings: [] }), 'spec-fix:': ok,
  'system-review:': () => ({ findings: [] }), 'system-fix:': ok,
  'harness:': ok, 'skeleton': ok, 'tests:': ok, 'verify': green,
  'freeze': () => ({ json: {}, exitCode: 0 }),
}

test('happy path: specs per workstream, system review, 4 harness writers, tests only where a tests/ path is owned', async () => {
  const { result, calls, logs } = await simulate(script, { respond: byLabel(base) })
  const labels = calls.map(c => c.opts.label)
  assert.equal(result.ok, true)
  assert.deepEqual(labels.filter(l => l.startsWith('spec:')), ['spec:transport', 'spec:train'])
  assert.deepEqual(labels.filter(l => l.startsWith('harness:')), ['harness:oracle', 'harness:chaos', 'harness:fuzz', 'harness:distributed', 'harness:bench'])
  assert.deepEqual(labels.filter(l => l.startsWith('tests:')), ['tests:transport'])
  assert.ok(logs.some(l => l.includes('workstream train owns no tests/ path')))
  assertRoles(assert, calls, { 'gate:': 'wan-verifier', 'extract:': 'wan-readonly', 'spec:': 'wan-researcher', 'spec-review:': 'wan-reviewer', 'system-review:': 'wan-reviewer', 'harness:': 'wan-coder', 'skeleton': 'wan-coder', 'tests:': 'wan-coder', 'verify': 'wan-verifier' })
})

test('refuses an unapproved design before any agent writes', async () => {
  const labels = []
  const respond = byLabel({ ...base, 'gate:approval': () => notApproved() })
  await assert.rejects(simulate(script, { respond: (p, o, i) => { labels.push(o.label); return respond(p, o, i) } }), /not approved/)
  assert.deepEqual(labels, ['gate:approval'])
})

test('workstreams that claim a harness directory are rejected at the gate', async () => {
  const bad = [{ key: 'transport', title: 't', scope: 's', ownedPaths: ['tests/'] }]
  await assert.rejects(simulate(script, { respond: byLabel({ ...base, 'extract:workstreams': () => ({ workstreams: bad }) }) }), /collides with the harness/)
})

test('a spec finding is fixed and re-reviewed; non-convergence makes the run not ok', async () => {
  const once = await simulate(script, {
    respond: byLabel({ ...base, 'spec-review:': (p, o) => ({ findings: o.label === 'spec-review:train:contracts:r1' ? [finding('major')] : [] }) }),
  })
  assert.equal(once.result.ok, true)
  assert.ok(once.calls.some(c => c.opts.label === 'spec-fix:train:r1'))
  assert.ok(once.calls.some(c => c.opts.label === 'sizegate:train:r1'))

  const never = await simulate(script, { respond: byLabel({ ...base, 'system-review:': () => ({ findings: [finding('blocker')] }) }) })
  assert.equal(never.result.ok, false)
  assert.equal(never.result.system.converged, false)
  assert.equal(never.calls.filter(c => c.opts.label.startsWith('system-fix:')).length, 2, 'stalled after three rounds')
})

test('a failed spec draft stops the run before the system review', async () => {
  await assert.rejects(
    simulate(script, { respond: byLabel({ ...base, 'spec:': (p, o) => (o.label === 'spec:train' ? null : ok()) }) }),
    /spec pipeline failed for: train/,
  )
})

test('red verify hands off to wan-fix with the failing rungs', async () => {
  const red = () => ({ allGreen: false, ladder: ladder(false, ['lint']), items: [] })
  const seen = []
  const { result } = await simulate(script, {
    respond: byLabel({ ...base, verify: red }),
    workflows: { 'wan-fix': a => { seen.push(a); return { ok: true, green: true } } },
  })
  assert.deepEqual(seen[0].rungs, ['lint'])
  assert.equal(result.verifyGreen, true)
})
