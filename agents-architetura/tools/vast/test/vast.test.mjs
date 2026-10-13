import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync, existsSync, readdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { validatePlan, estimate, launchGate, launch, preflight, start, monitorOnce, stop, hardwareRefusal } from '../vast.mjs'
import { fakeClient } from '../client.mjs'
import { recordGrant } from '../../factory/approve.mjs'

const PLAN = {
  id: 'e1-outer-sync', hypothesis: 'outer-step sync every 50 steps cuts step time 4x at 4 workers over 80 ms RTT',
  expected: { stepTimeSec: 2.5, speedup: 4 }, maxBudgetUsd: 10, warnAtPct: 80, expectedHours: 2, expectedTokens: 50e6,
  instances: [{ role: 'worker', count: 2, maxDph: 0.5, query: 'gpu_name=RTX_4090 num_gpus=1' }],
  image: 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime', diskGb: 40, setup: 'bash deploy/setup.sh', run: 'python -m wan.launch', metrics: 'tail -n 20 /workspace/metrics.jsonl',
  startupCheck: 'python -m wan.selfcheck', stop: { maxLoss: 20, maxRestarts: 3, stallMinutes: 15 },
}
const PLAN_PATH = 'docs/experiments/e1-outer-sync/PLAN.json'
const OFFERS = [{ id: 1, gpu_name: 'RTX_4090', dph_total: 0.45 }, { id: 2, gpu_name: 'RTX_4090', dph_total: 0.30 }, { id: 3, gpu_name: 'RTX_4090', dph_total: 0.9 }]
const LOCAL = { classification: 'VALIDATED', checklist: { hypothesisDefined: true, architectureApproved: true, localPrototypeWorks: true, localCorrectness: true, testsPass: true, failureCasesTested: true, improvementDemonstrated: true } }

function setup({ plan = PLAN, approve = true, local = LOCAL, exec = {}, hardware = { gpu: true } } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'vast-'))
  const write = (f, b) => { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), typeof b === 'string' ? b : JSON.stringify(b)) }
  write(PLAN_PATH, plan)
  if (local) write('docs/experiments/e1-outer-sync/LOCAL.json', local)
  if (approve) recordGrant(root, [PLAN_PATH], { approvedBy: 'human' })
  if (hardware) write('agents-architetura/tools/factory/hardware.json', hardware)
  const state = join(root, 'fake-vast.json')
  writeFileSync(state, JSON.stringify({ offers: OFFERS, instances: [], exec: { '*': exec } }))
  return { root, client: fakeClient(state), state }
}

test('plan validation', () => {
  assert.deepEqual(validatePlan(PLAN), [])
  const e = validatePlan({ id: 'X', instances: [{ role: 'w', count: 0 }], diskGb: 1 })
  for (const want of ['id must match', 'hypothesis is required', 'expected metrics', 'maxBudgetUsd', 'instances[0].count', 'instances[0].maxDph', 'image is required', 'diskGb']) assert.ok(e.some(x => x.includes(want)), want)
})

test('estimate rents the cheapest offers under the cap and refuses over budget', () => {
  const { client } = setup()
  const e = estimate(PLAN, client)
  assert.deepEqual(e.picks.map(p => p.offerId), [2, 1])
  assert.equal(e.estimatedUsd, 1.5)
  assert.equal(e.costPerMillionTokensUsd, 0.03)
  assert.equal(e.ok, true)
  const over = estimate({ ...PLAN, expectedHours: 20, maxBudgetUsd: 5 }, client)
  assert.match(over.problems[0], /exceeds the approved budget \$5/)
  assert.match(estimate({ ...PLAN, instances: [{ ...PLAN.instances[0], count: 3 }] }, client).problems[0], /2 offers under \$0.5\/h, need 3/)
})

test('launch gate: human-approved exact plan, a promotable local result, a complete checklist', () => {
  assert.match(launchGate(setup({ approve: false }).root, PLAN_PATH, PLAN)[0], /not approved by a human/)
  assert.match(launchGate(setup({ local: null }).root, PLAN_PATH, PLAN)[0], /local before paid cloud/)
  assert.match(launchGate(setup({ local: { ...LOCAL, classification: 'REFUTED' } }).root, PLAN_PATH, PLAN)[0], /only VALIDATED or PARTIALLY VALIDATED/)
  assert.match(launchGate(setup({ local: { ...LOCAL, checklist: { ...LOCAL.checklist, failureCasesTested: false } } }).root, PLAN_PATH, PLAN)[0], /checklist incomplete: failureCasesTested/)
  const { root } = setup()
  writeFileSync(join(root, PLAN_PATH), JSON.stringify({ ...PLAN, maxBudgetUsd: 1000 }))
  assert.match(launchGate(root, PLAN_PATH, PLAN)[0], /changed after its last approval/)
})

test('launch creates and records every instance; a second launch is refused while they live', () => {
  const { root, client } = setup()
  const r = launch(root, PLAN_PATH, PLAN, client, 0)
  assert.equal(r.ok, true)
  assert.equal(r.instances.length, 2)
  assert.equal(client.listInstances().length, 2)
  assert.match(launch(root, PLAN_PATH, PLAN, client, 0).refused.join(), /still recorded as live/)
})

test('a critical pre-flight failure means DO NOT START TRAINING', () => {
  const { root, client } = setup({ exec: { 'python -c "import torch,sys': { code: 1, out: 'CUDA unavailable' } } })
  launch(root, PLAN_PATH, PLAN, client, 0)
  const p = preflight(root, PLAN, client)
  assert.equal(p.verdict, 'DO NOT START TRAINING')
  assert.equal(p.failed.length, 2)
  assert.match(start(root, PLAN, client).refused[0], /DO NOT START TRAINING/)
})

test('monitor: budget warning pauses; divergence, a vanished instance or the budget cap emergency-stop; stop preserves logs then destroys', () => {
  const H = 3600000
  const healthy = { 'tail -n 20 /workspace/metrics.jsonl': { code: 0, out: `{"step": 10, "loss": 3.2, "ts": "${new Date(2 * H).toISOString()}"}` } }
  const a = setup({ exec: healthy })
  launch(a.root, PLAN_PATH, PLAN, a.client, 0)
  preflight(a.root, PLAN, a.client); start(a.root, PLAN, a.client)
  assert.equal(monitorOnce(a.root, PLAN, a.client, 2 * H).action, 'continue') // 2 x (0.30 + 0.45) = $1.5
  assert.equal(monitorOnce(a.root, { ...PLAN, maxBudgetUsd: 1.7 }, a.client, 2 * H).action, 'pause') // 88% of 1.7
  const cap = monitorOnce(a.root, { ...PLAN, maxBudgetUsd: 1.4 }, a.client, 2 * H)
  assert.equal(cap.action, 'emergency-stop')
  assert.match(cap.stops.join(), /reached the approved budget/)

  const nan = setup({ exec: { 'tail -n 20 /workspace/metrics.jsonl': { code: 0, out: '{"step": 11, "loss": "NaN", "ts": "1970-01-01T00:00:00Z"}' } } })
  launch(nan.root, PLAN_PATH, PLAN, nan.client, 0)
  assert.match(monitorOnce(nan.root, PLAN, nan.client, 1000).stops.join(), /divergence/)

  const gone = setup({ exec: healthy })
  launch(gone.root, PLAN_PATH, PLAN, gone.client, 0)
  const s = JSON.parse(readFileSync(gone.state, 'utf8')); s.instances = s.instances.slice(1); writeFileSync(gone.state, JSON.stringify(s))
  assert.match(monitorOnce(gone.root, PLAN, gone.client, 1000).stops.join(), /vanished/)

  const r = stop(a.root, PLAN, a.client, 'emergency stop: test')
  assert.equal(r.ok, true)
  assert.equal(a.client.listInstances().length, 0)
  assert.ok(readdirSync(r.logs).some(f => f.endsWith('run.log')))
  assert.ok(existsSync(join(a.root, '.factory/vast/e1-outer-sync/ledger.json')))
})

test('gpu=false: launch refuses before anything is rented, even with an approved plan and a validated local run', () => {
  for (const hardware of [{ gpu: false }, { gpu: 'true' }, {}, null]) {
    const { root, client, state } = setup({ hardware })
    const r = launch(root, PLAN_PATH, PLAN, client, 0)
    assert.equal(r.ok, false)
    assert.match(r.refused.join(), /gpu=false/)
    assert.deepEqual(JSON.parse(readFileSync(state, 'utf8')).instances, [])
  }
  const { root, client } = setup()
  assert.equal(launch(root, PLAN_PATH, PLAN, client, 0).ok, true)
})

test('gpu=false refuses every account action but never stopping or monitoring', () => {
  const off = { gpu: false, reason: 'gpu is false' }
  for (const cmd of ['doctor', 'estimate', 'launch', 'preflight', 'start']) assert.match(hardwareRefusal(cmd, off), /gpu=false.*\/wan-gpu/)
  for (const cmd of ['validate', 'status', 'monitor', 'stop']) assert.equal(hardwareRefusal(cmd, off), null)
  for (const cmd of ['doctor', 'estimate', 'launch', 'preflight', 'start']) assert.equal(hardwareRefusal(cmd, { gpu: true }), null)
})
