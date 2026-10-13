import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, ladder, assertRoles } from './helpers.mjs'

const script = built('wan-bughunt')
const hyp = id => ({ id, mechanism: `m-${id}`, interleaving: 'i', confirmIf: 'c', refuteIf: 'r' })
const approve = () => ({ approve: true, reasons: 'ok' })
const applied = () => ({ summary: 'applied', files: ['src/transport/peer.py'], evidence: [{ claim: 'repro ran', command: 'r', exitCode: 0, output: '' }] })
const base = {
  'evidence': () => ({ reproduced: true, reproCommand: 'uv run pytest tests/x.py', rate: '7/200', summary: 's' }),
  'find:ordering': () => ({ hypotheses: [hyp('race')] }),
  'find:': () => ({ hypotheses: [] }),
  'refute:': () => ({ verdict: 'survives', argument: 'cannot refute' }),
  'experiment:': () => ({ results: [{ hypothesisId: 'ordering-race', outcome: 'confirms', detail: 'seen' }] }),
  'fixplan:': () => ({ hypothesisId: 'ordering-race', fix: 'f', happensBefore: 'hb', files: ['src/transport/peer.py'] }),
  'fixvote:': approve,
  'apply:': applied,
  'verify:': () => verified(),
  'revert:': () => ({ summary: 'reverted', files: [] }),
  'causal:save:': () => ({ summary: 'saved', files: [] }),
  'closure': () => ({ summary: 'written', files: [] }),
}
// 7/200 observed -> closure needs ceil(30 / 0.035) = 858 clean runs: 429 amplified + 6 x 72 under load.
// The causal A/B: the pre-fix files bring the failure back (reverted > 0), the fixed tree stays clean.
const CAUSAL = { causal: true, reverted: { runs: 2000, failures: 5 }, fixed: { runs: 2000, failures: 0 } }
function verified({ ampFindings = 0, loadFailed = 0, ampTotal = 429, loadTotal = 432, green = true, causal = CAUSAL } = {}) {
  return { fixed: true, runs: ampTotal + loadTotal, failures: 0, ladder: ladder(green), amplify: { total: ampTotal, findings: Array.from({ length: ampFindings }, (_, i) => ({ seed: i })) }, load: { total: loadTotal, failed: loadFailed }, causal }
}
const run = (over = {}, a = { bug: 'loss differs from baseline after step 3', id: 'loss-drift' }) =>
  simulate(script, { args: a, respond: byLabel({ ...base, ...over }) })

test('requires a bug description', async () => {
  await assert.rejects(simulate(script, { args: {}, respond: () => null }), /args.bug/)
})

test('happy path: author applies, an independent verifier proves it, every step on its role', async () => {
  const { result, calls } = await run()
  assert.equal(result.solved, true)
  assert.deepEqual(result.fixedBy, { hypothesisId: 'ordering-race', files: ['src/transport/peer.py'] })
  assertRoles(assert, calls, { 'evidence': 'wan-researcher', 'find:': 'wan-reviewer', 'refute:': 'wan-reviewer', 'experiment:': 'wan-researcher', 'fixplan:': 'wan-proposer', 'fixvote:': 'wan-reviewer', 'apply:': 'wan-coder', 'verify:': 'wan-verifier' })
  assert.ok(!calls.some(c => c.opts.label.startsWith('revert:')))
})

test('not reproduced: stops without hunting', async () => {
  const { result, calls } = await run({ 'evidence': () => ({ reproduced: false, summary: 'could not' }) })
  assert.equal(result.reason, 'not reproduced')
  assert.equal(calls.length, 1)
})

test('a refuted hypothesis is never fixed and is fed to the next round', async () => {
  const { calls } = await run({ 'refute:': (p, o) => (o.label.endsWith(':0:r1') ? { verdict: 'refuted', argument: 'path unreachable' } : { verdict: 'survives', argument: 'x' }) })
  assert.ok(!calls.some(c => c.opts.label === 'fixplan:r1'))
  assert.ok(calls.find(c => c.opts.label === 'find:ordering:r2').prompt.includes('path unreachable'))
})

test('a fix needs BOTH reviewers; a split vote goes back to the hunt', async () => {
  const { result, calls } = await run({ 'fixvote:breaks-nothing': () => ({ approve: false, reasons: 'breaks oracle' }) })
  assert.equal(result.solved, false)
  assert.ok(!calls.some(c => c.opts.label.startsWith('apply:')))
  assert.ok(calls.find(c => c.opts.label === 'find:ordering:r2').prompt.includes('breaks oracle'))
})

test('the verification bar is statistical: N = 30 / observed rate, split amplified and under load', async () => {
  const { result, calls } = await run()
  assert.equal(result.closureRunsNeeded, 858)
  const v = calls.find(c => c.opts.label === 'verify:r1').prompt
  assert.match(v, /amplify\.mjs --runs 429 --seed-base 1 --periods 16,64,256 --json -- uv run pytest tests\/x\.py/)
  assert.match(v, /load\.mjs --workers 6 --runs-per-worker 72 --seed-base 100000 --json -- uv run pytest tests\/x\.py/)
  assert.deepEqual(result.verified, { runs: 861, failures: 0, needed: 858, ladderGreen: true, causal: true, reverted: CAUSAL.reverted, fixedArm: CAUSAL.fixed })
  assert.ok(calls.some(c => c.opts.label === 'closure' && c.prompt.includes('CLOSED')))
})

test('the counts come from the drivers JSON, not from the agent claim', async () => {
  const lying = () => ({ ...verified({ loadFailed: 2 }), failures: 0 })
  const { result } = await run({ 'verify:': lying })
  assert.equal(result.solved, false)
})

test('clean but below the bound when the bound exceeds the cap: INCONCLUSIVE, the fix stays', async () => {
  const { result, calls } = await run({ 'evidence': () => ({ reproduced: true, reproCommand: 'r', rate: '1/2000', summary: 's' }), 'verify:': () => verified({ ampTotal: 50, loadTotal: 54 }) }, { bug: 'rare', id: 'rare', maxRuns: 100 })
  assert.deepEqual([result.solved, result.inconclusive], [false, true])
  assert.ok(!calls.some(c => c.opts.label.startsWith('revert:')))
  assert.ok(calls.find(c => c.opts.label === 'closure').prompt.includes('INCONCLUSIVE'))
})

test('independent verification must show 0 failures over N runs AND a green ladder; otherwise the author reverts', async () => {
  for (const v of [
    verified({ ampFindings: 1 }),
    verified({ ampTotal: 10, loadTotal: 0 }),
    verified({ green: false }),
  ]) {
    const { result, calls } = await run({ 'verify:': () => v })
    assert.equal(result.solved, false, JSON.stringify(v))
    assert.equal(calls.find(c => c.opts.label === 'revert:r1').opts.agentType, 'wan-coder')
  }
})

test('unsafe file lists in a fix proposal are rejected without applying', async () => {
  const { result, calls } = await run({ 'fixplan:': () => ({ hypothesisId: 'x', fix: 'f', happensBefore: 'h', files: ['../../etc/passwd'] }) })
  assert.equal(result.solved, false)
  assert.ok(!calls.some(c => c.opts.label.startsWith('fixvote:')))
})

test('closure is causal too: the pre-fix files must bring the failure back, or the fix stays INCONCLUSIVE', async () => {
  const { result, calls } = await run()
  const save = calls.find(c => c.opts.label === 'causal:save:r1')
  assert.ok(save.prompt.includes('causal.mjs save --out agents-architetura/docs/bughunt/loss-drift/prefix-r1 -- src/transport/peer.py'))
  assert.ok(calls.indexOf(save) < calls.findIndex(c => c.opts.label === 'apply:r1'), 'saved before the fix is applied')
  // 7/200 observed: expect ~3 failures in ceil(3 / 0.035) = 86 reverted runs
  assert.ok(calls.find(c => c.opts.label === 'verify:r1').prompt.includes('causal.mjs ab --prefix agents-architetura/docs/bughunt/loss-drift/prefix-r1 --runs 86 --json -- uv run pytest tests/x.py'))
  assert.equal(result.verified.causal, true)
  for (const causal of [{ reverted: { runs: 86, failures: 0 }, fixed: { runs: 86, failures: 0 } }, { reverted: { runs: 86, failures: 4 }, fixed: { runs: 10, failures: 0 } }, null]) {
    const { result: r, logs } = await run({ 'verify:': () => verified({ causal }) })
    assert.deepEqual([r.solved, r.inconclusive], [false, true], JSON.stringify(causal))
    assert.ok(logs.some(l => l.includes('the causal A/B is not proven')))
  }
})
