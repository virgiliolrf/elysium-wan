import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, ladder, approved, notApproved, assertRoles } from './helpers.mjs'

const script = built('wan-experiment')
const ADR = 'docs/design/adr/ADR-0003-outer-sync.md'
const TREE = 'a'.repeat(40)
const ok = () => ({ summary: 'ok', files: [] })
const PLAN = { id: 'e1', hypothesis: 'outer-step sync every 50 steps cuts step time 4x over 80 ms RTT', expected: { speedup: 4, stepTimeSec: 2.5 }, lowerIsBetter: ['stepTimeSec'], primary: 'speedup', tolerancePct: 10, local: { correctness: 'uv run pytest -q experiments/e1/test_ref.py', performance: 'uv run python experiments/e1/bench.py', failure: 'uv run pytest -q experiments/e1/test_churn.py' } }
const localRun = (metrics = { speedup: 3.9, stepTimeSec: 2.6 }, extra = {}) => ({ correctness: { exitCode: 0 }, performance: { exitCode: 0, metrics }, failure: { exitCode: 0 }, ladder: ladder(true), falsified: false, ...extra })
const base = {
  'gate:approval': () => approved(),
  'gate:adr': () => approved([ADR]),
  'gate:plan': () => approved(['docs/experiments/e1/PLAN.json']),
  'gate:hardware': () => ({ json: { gpu: true }, exitCode: 0 }),
  'plan': ok, 'prototype': () => ({ summary: 'ok', files: [], evidence: [{ claim: 'ran', command: 'x', exitCode: 0, output: '' }] }),
  'extract:plan': () => PLAN,
  'audit:snapshot:': () => ({ tree: TREE }),
  'audit:check:': () => ({ tree: TREE, changed: [], violations: [] }),
  'local': () => localRun(),
  'record:': ok,
  'cloud': () => ({ outcome: 'completed', observed: { speedup: 4.1, stepTimeSec: 2.4 }, spentUsd: 6.2, liveInstances: 0 }),
  'report': ok,
}
const run = (extra = {}, args = { adr: ADR, id: 'e1' }) => simulate(script, { args, respond: byLabel({ ...base, ...extra }) })
const decode = p => JSON.parse(Buffer.from(p.match(/--b64 (\S+)/)[1], 'base64').toString('utf8'))

test('local only: the script classifies expected vs observed and records the promotion checklist', async () => {
  const { result, calls } = await run()
  assert.deepEqual([result.ok, result.local, result.cloud], [true, 'VALIDATED', null])
  const rec = decode(calls.find(c => c.opts.label === 'record:local').prompt)
  assert.equal(rec.classification, 'VALIDATED')
  assert.ok(Object.values(rec.checklist).every(Boolean))
  assert.ok(!calls.some(c => c.opts.label === 'cloud'))
  assertRoles(assert, calls, { 'plan': 'wan-researcher', 'prototype': 'wan-coder', 'local': 'wan-verifier', 'report': 'wan-researcher' })
})

test('a refuted hypothesis is reported as refuted, never promoted, and becomes a lesson', async () => {
  const { result, calls } = await run({ 'local': () => localRun({ speedup: 1.3, stepTimeSec: 7 }) }, { adr: ADR, id: 'e1', cloud: true })
  assert.deepEqual([result.ok, result.classification], [false, 'REFUTED'])
  assert.match(result.stoppedFor, /improvementDemonstrated/)
  assert.ok(!calls.some(c => c.opts.label === 'cloud'))
  assert.ok(calls.find(c => c.opts.label === 'report').prompt.includes('Do not rationalize it'))
})

test('cloud: needs the ADR and the exact plan+budget granted by a human; then runs only through vast.mjs and classifies again', async () => {
  const noAdr = await run({ 'gate:adr': () => notApproved(ADR) }, { adr: ADR, id: 'e1', cloud: true })
  assert.match(noAdr.result.stoppedFor, /architectureApproved/)
  const noPlan = await run({ 'gate:plan': () => notApproved('docs/experiments/e1/PLAN.json') }, { adr: ADR, id: 'e1', cloud: true })
  assert.equal(noPlan.result.awaitingHuman, true)
  assert.match(noPlan.result.stoppedFor, /approve\.mjs grant docs\/experiments\/e1\/PLAN\.json/)
  const { result, calls } = await run({}, { adr: ADR, id: 'e1', cloud: true })
  assert.deepEqual([result.ok, result.cloud], [true, 'VALIDATED'])
  const c = calls.find(x => x.opts.label === 'cloud')
  assert.equal(c.opts.agentType, 'wan-vastai')
  assert.ok(c.prompt.includes('never the vastai CLI directly') && c.prompt.includes('vast.mjs monitor'))
})

test('a paused or emergency-stopped cloud run is not a result', async () => {
  const paused = await run({ 'cloud': () => ({ outcome: 'paused', observed: {}, liveInstances: 0, detail: 'budget warning 82%' }) }, { adr: ADR, id: 'e1', cloud: true })
  assert.equal(paused.result.ok, false)
  assert.match(paused.result.stoppedFor, /cloud run paused/)
  await assert.rejects(run({}, { adr: 'docs/x.md', id: 'e1' }), /args.adr/)
  await assert.rejects(run({ 'gate:adr': () => notApproved(ADR, 'document does not exist') }), /does not exist/)
})

test('cloud with the hardware switch off: the local result stands, nothing reaches Vast.ai, and a human is asked', async () => {
  for (const off of [{ json: { gpu: false, reason: 'gpu is false' }, exitCode: 0 }, { json: {}, exitCode: 1 }, null]) {
    const { result, calls } = await run({ 'gate:hardware': () => off }, { adr: ADR, id: 'e1', cloud: true })
    assert.equal(result.ok, false)
    assert.equal(result.awaitingHuman, true)
    assert.equal(result.local, 'VALIDATED')
    assert.match(result.stoppedFor, /gpu=false.*\/wan-gpu/)
    assert.ok(!calls.some(c => c.opts.label === 'cloud'))
    assert.ok(calls.find(c => c.opts.label === 'report').prompt.includes('gpu=false'))
  }
})
