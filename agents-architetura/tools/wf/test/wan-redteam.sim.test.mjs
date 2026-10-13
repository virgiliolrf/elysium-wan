import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, assertRoles, approved } from './helpers.mjs'

const script = built('wan-redteam')
const ok = () => ({ summary: 'ok', files: [] })
const TREE = 'b'.repeat(40)
const cls = id => ({ id, mechanism: `m-${id}`, whereInDesign: 'w', susceptibilityTest: 't' })
const base = {
  'gate:approval': () => approved(),
  'classes': () => ({ classes: [cls('reorder'), cls('nan'), cls('churn'), cls('reorder')] }),
  'rt-audit:snapshot': () => ({ tree: TREE }),
  'rt-audit:check': () => ({ tree: TREE, changed: ['tests/redteam/nan/test_nan.py'], violations: [] }),
  'rt-test:': ok,
  'rt-run': () => ({ results: [
    { id: 'reorder', outcome: 'susceptible', evidence: 'diverged at step 4', reproCommand: 'uv run pytest tests/redteam/reorder' },
    { id: 'nan', outcome: 'resistant', evidence: 'pass' },
    { id: 'churn', outcome: 'broken-test', evidence: 'fixture error' },
  ] }),
  'rt-refute:': () => ({ verdict: 'survives', argument: 'real' }),
  'rt-record': () => ({ summary: 'written', files: [] }),
  'rt-findings': ok,
}
const run = (over = {}, args = {}) => {
  const hunts = []
  return simulate(script, { args, respond: byLabel({ ...base, ...over }), workflows: { 'wan-bughunt': a => { hunts.push(a); return { solved: true } } } })
    .then(r => ({ ...r, hunts }))
}

test('classes deduped, one red-team writer per test dir inside a scope audit, outcomes triaged by role', async () => {
  const { result, calls } = await run()
  assert.deepEqual(calls.filter(c => c.opts.label.startsWith('rt-test:')).map(c => c.opts.label), ['rt-test:reorder', 'rt-test:nan', 'rt-test:churn'])
  assert.ok(calls.find(c => c.opts.label === 'rt-test:nan').prompt.includes('OWNED PATHS: tests/redteam/nan/ (only)'))
  assert.ok(calls.find(c => c.opts.label === 'rt-audit:check').prompt.includes(`--since ${TREE} --scopes '["tests/redteam/reorder/","tests/redteam/nan/","tests/redteam/churn/"]'`))
  assert.deepEqual([result.confirmed, result.broken, result.resistant], [['reorder'], ['churn'], ['nan']])
  assert.equal(result.ok, false)
  assertRoles(assert, calls, { 'classes': 'wan-researcher', 'rt-audit:': 'wan-verifier', 'rt-test:': 'wan-redteam', 'rt-run': 'wan-verifier', 'rt-refute:': 'wan-reviewer', 'rt-findings': 'wan-researcher' })
})

test('the red team may not touch production code: an out-of-scope write stops the run', async () => {
  await assert.rejects(run({ 'rt-audit:check': () => ({ tree: TREE, changed: ['src/transport/peer.py'], violations: ['src/transport/peer.py'] }) }),
    /red team modified files outside tests\/redteam\/: src\/transport\/peer.py/)
})

test('triage: refuters split -> a human decides (needs-human finding); both refute -> false positive rejected with evidence', async () => {
  const split = await run({ 'rt-refute:': (p, o) => (o.label.endsWith(':1') ? { verdict: 'refuted', argument: 'test asserts an unpromised order' } : { verdict: 'survives', argument: 'x' }) })
  assert.deepEqual([split.result.confirmed, split.result.needsHuman, split.result.disputed], [[], ['reorder'], []])
  assert.equal(split.result.ok, false)
  const rec = JSON.parse(Buffer.from(split.calls.find(c => c.opts.label === 'rt-record').prompt.match(/--b64 (\S+)/)[1], 'base64').toString('utf8'))
  assert.deepEqual(rec.findings.map(f => [f.id, f.status]), [['RT-reorder', 'needs-human']])
  const both = await run({ 'rt-refute:': () => ({ verdict: 'refuted', argument: 'fixture bug' }) })
  assert.deepEqual([both.result.confirmed, both.result.needsHuman, both.result.disputed], [[], [], ['reorder']])
})

test('classes the runner did not report are surfaced, never counted as resistant', async () => {
  const { result } = await run({ 'rt-run': () => ({ results: [{ id: 'nan', outcome: 'resistant', evidence: 'pass' }] }) })
  assert.deepEqual(result.unreported, ['reorder', 'churn'])
  assert.equal(result.ok, false)
})

test('autofix hands confirmed findings to wan-bughunt; off by default', async () => {
  assert.deepEqual((await run()).hunts, [])
  const { hunts, result } = await run({}, { autofix: true })
  assert.deepEqual(hunts.map(h => [h.id, h.repro]), [['rt-reorder', 'uv run pytest tests/redteam/reorder']])
  assert.deepEqual(result.bughunts, [{ id: 'reorder', solved: true }])
})

test('all resistant: ok; no classes or no suites written stops the run', async () => {
  const { result } = await run({ 'classes': () => ({ classes: [cls('nan')] }), 'rt-run': () => ({ results: [{ id: 'nan', outcome: 'resistant', evidence: 'pass' }] }) })
  assert.equal(result.ok, true)
  await assert.rejects(run({ 'classes': () => ({ classes: [] }) }), /no failure classes/)
  await assert.rejects(run({ 'rt-test:': () => null }), /no susceptibility suites/)
})
