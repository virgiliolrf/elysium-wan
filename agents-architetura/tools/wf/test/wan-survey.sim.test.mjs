import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, finding, assertRoles } from './helpers.mjs'

const script = built('wan-survey')
const ok = () => ({ summary: 'ok', files: [] })
const INV = { files: 12, byTopDir: { '(root)': 4, 'pipeline/': 6, 'deploy/': 2, 'docs/': 1 }, imports: [] }
const comp = (component, paths) => ({ component, paths, responsibility: 'r', dependencies: [], interfaces: ['i'], dataFlow: 'd', controlFlow: 'c', failureBoundary: 'f', securityBoundary: 's', resourceBoundary: 'res' })
const GOOD = { components: [comp('pipeline', ['pipeline/']), comp('deploy', ['deploy/'])] }
const SCHED = { id: 'distributed-training', paths: ['pipeline/'], evidence: ['pipeline/stage.py:10'], focus: 'f', reviewLens: 'l' }
const base = {
  'inventory': () => ({ json: INV, exitCode: 0 }),
  'map': ok, 'repair:': ok, 'fix:': ok, 'specialists:write': ok,
  'extract:components:': () => GOOD,
  'check-paths:': () => ({ json: { ok: true, missing: [], invalid: [] }, exitCode: 0 }),
  'review:': () => ({ findings: [] }),
  'specialists:decide': () => ({ specialists: [SCHED, { ...SCHED, id: 'scheduler', paths: ['scheduler/'] }, { ...SCHED, id: 'wizard', paths: ['x/'] }], rejected: [{ id: 'observability', why: 'no metrics code' }] }),
  'reference:inventory': () => ({ json: { files: 30, byTopDir: {} }, exitCode: 0 }),
  'reference:map': ok,
}
const run = (extra = {}, args = {}) => simulate(script, { args, respond: byLabel({ ...base, ...extra }) })

test('maps what exists from script facts; checks paths and coverage; specialists only with real evidence', async () => {
  const { result, calls } = await run({ 'check-paths:specialists': () => ({ json: { ok: false, missing: ['scheduler/'] }, exitCode: 1 }) })
  assert.equal(result.ok, true)
  assert.deepEqual(result.specialists, ['distributed-training'])
  assert.deepEqual(result.rejected, ['observability'])
  assert.ok(calls.find(c => c.opts.label === 'map').prompt.includes('["pipeline/","deploy/"]'))
  assert.ok(calls.find(c => c.opts.label === 'specialists:write').prompt.includes('"id":"distributed-training"'))
  assert.ok(!calls.find(c => c.opts.label === 'specialists:write').prompt.includes('wizard'))
  assertRoles(assert, calls, { 'inventory': 'wan-verifier', 'map': 'wan-researcher', 'extract:': 'wan-readonly', 'review:': 'wan-reviewer', 'specialists:decide': 'wan-researcher' })
})

test('missing paths, uncovered source dirs and empty fields are repaired; review findings fixed', async () => {
  const partial = { components: [{ ...comp('pipeline', ['pipeline/', 'scheduler/']), failureBoundary: '' }] }
  const r = await run({
    'extract:components:': (p, o, i, n) => (n === 0 ? partial : GOOD),
    'check-paths:a1': () => ({ json: { ok: false, missing: ['scheduler/'] }, exitCode: 1 }),
    'review:fidelity:r1': () => ({ findings: [finding('major')] }),
  })
  const repair = r.calls.find(c => c.opts.label === 'repair:a1').prompt
  assert.ok(repair.includes('path scheduler/ does not exist'))
  assert.ok(repair.includes('source directory deploy/ belongs to no component'))
  assert.ok(repair.includes('field failureBoundary is empty'))
  assert.ok(r.calls.some(c => c.opts.label === 'fix:r1'))
  assert.equal(r.result.ok, true)
})

test('a reference repo is cloned read-only and mapped; a bad reference is ignored', async () => {
  const { result, calls } = await run({}, { reference: 'github:virgiliolrf/Elysium-Old' })
  assert.equal(result.reference, 'docs/design/reference/Elysium-Old.md')
  assert.match(calls.find(c => c.opts.label === 'reference:inventory').prompt, /gh repo clone virgiliolrf\/Elysium-Old \.factory\/reference\/Elysium-Old -- --depth 1/)
  const bad = await run({}, { reference: 'github:a/b; rm -rf ~' })
  assert.equal(bad.result.reference, null)
  assert.ok(!bad.calls.some(c => c.opts.label.startsWith('reference:')))
})
