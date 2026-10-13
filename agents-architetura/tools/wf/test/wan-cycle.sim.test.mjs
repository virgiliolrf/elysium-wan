import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, ladder, publisher, payloadOf, publishLabels, approved } from './helpers.mjs'

const script = built('wan-cycle')
const REVIEW = { converged: true, rounds: 2 }
const GATES = { milestone: 'M2', consecutive: 2, rungs: ['factory'], pinned: [], campaigns: [{ id: 'rust-tsan' }, { id: 'amplify-chaos' }], fuzz: [{ id: 'fuzz-wire', minCpuHours: 4 }] }
const NAMES = ['wan-survey', 'wan-handout', 'wan-implement', 'wan-corpus', 'wan-redteam', 'wan-audit', 'wan-campaign', 'wan-fuzz', 'wan-scanners', 'wan-fix', 'wan-closeout']
const TREE = 'a'.repeat(40)
const flows = (failAt = null) => {
  const order = []
  const mk = name => a => { order.push([name, a]); return name === 'wan-implement' ? { ok: name !== failAt, review: REVIEW } : name === 'wan-closeout' ? { ok: name !== failAt, verdict: name !== failAt ? 'VERIFIED' : 'NOT VERIFIED', tree: TREE } : { ok: name !== failAt } }
  return { order, workflows: Object.fromEntries(NAMES.map(n => [n, mk(n)])) }
}
const PLAN_DOCS = ['docs/design/HANDOUT-M2.md', 'docs/design/PLAN-M2.json', 'docs/design/GATES-M2.json']
const respond = (green = true, gates = GATES, extra = {}) => byLabel({
  'publish': publisher(),
  'gate:approval:plan': p => { const m = /HANDOUT-(M\d+)\.md/.exec(p)[1]; return approved(PLAN_DOCS.map(d => d.replace('M2', m))) },
  ...extra,
  'extract:gates': () => gates,
  'benchmark': () => ({ allGreen: green, ladder: green ? ladder(true) : ladder(false, ['bench']), items: [] }),
})

test('SURVEY -> PLAN -> IMPLEMENT -> CORPUS -> AUDIT -> CAMPAIGNS -> FUZZ -> SCANNERS -> BENCHMARK -> RED TEAM -> VERIFY -> CLOSEOUT', async () => {
  const f = flows()
  const { result, calls } = await simulate(script, { args: { milestone: 'M2' }, workflows: f.workflows, respond: respond() })
  assert.equal(result.ok, true)
  assert.equal(result.verdict, 'VERIFIED')
  assert.deepEqual(f.order, [
    ['wan-survey', {}],
    ['wan-handout', { milestone: 'M2' }],
    ['wan-implement', { milestone: 'M2' }],
    ['wan-corpus', {}],
    ['wan-audit', { close: true }],
    ['wan-campaign', { campaign: 'rust-tsan' }],
    ['wan-campaign', { campaign: 'amplify-chaos' }],
    ['wan-fuzz', { campaign: 'fuzz-wire' }],
    ['wan-scanners', {}],
    ['wan-redteam', { autofix: true }],
    ['wan-fix', {}],
    ['wan-closeout', { milestone: 'M2', review: REVIEW }],
  ])
  assert.equal(calls.find(c => c.opts.label === 'benchmark').opts.agentType, 'wan-verifier')
})

test('quick depth shortens campaigns and fuzzing; gate ids are validated', async () => {
  const f = flows()
  await simulate(script, { args: { milestone: 'M2', depth: 'quick' }, workflows: f.workflows, respond: respond(true, { ...GATES, campaigns: [{ id: 'rust-tsan' }, { id: '../evil' }] }) })
  assert.deepEqual(f.order.filter(o => o[0] === 'wan-campaign').map(o => o[1]), [{ campaign: 'rust-tsan', maxWaves: 2 }])
  assert.deepEqual(f.order.find(o => o[0] === 'wan-fuzz')[1], { campaign: 'fuzz-wire', minutes: 10 })
})

test('stops at the first stage that is not ok', async () => {
  for (const [failAt, stoppedAt] of [['wan-survey', 'survey'], ['wan-implement', 'implement'], ['wan-audit', 'audit'], ['wan-campaign', 'campaign rust-tsan'], ['wan-fuzz', 'fuzz fuzz-wire'], ['wan-scanners', 'scanners']]) {
    const f = flows(failAt)
    const { result } = await simulate(script, { args: { milestone: 'M0' }, workflows: f.workflows, respond: respond() })
    assert.equal(result.stoppedAt, stoppedAt)
    assert.ok(!f.order.some(o => o[0] === 'wan-closeout'))
  }
})

test('a benchmark regression stops before the red team and closeout; requires a milestone', async () => {
  const f = flows()
  const { result } = await simulate(script, { args: { milestone: 'M0' }, workflows: f.workflows, respond: respond(false) })
  assert.equal(result.stoppedAt, 'benchmark')
  assert.ok(!f.order.some(o => o[0] === 'wan-redteam'))
  await assert.rejects(simulate(script, { args: {}, workflows: flows().workflows }), /args.milestone/)
})

test('publishing: draft at the start, a commit per stage (red or green), ready only for the tree closeout verified', async () => {
  const { result, calls } = await simulate(script, { args: { milestone: 'M2' }, workflows: flows().workflows, respond: respond() })
  assert.deepEqual(publishLabels(calls), ['publish:draft', 'publish:M2-survey', 'publish:M2-plan', 'publish:M2-implement', 'publish:M2-corpus', 'publish:M2-audit',
    'publish:M2-campaign-rust-tsan', 'publish:M2-campaign-amplify-chaos', 'publish:M2-fuzz-fuzz-wire', 'publish:M2-scanners', 'publish:M2-benchmark', 'publish:M2-redteam',
    'publish:M2-verify', 'publish:M2-closeout', 'publish:ready'])
  assert.match(calls.find(c => c.opts.label === 'publish:ready').prompt, new RegExp(`publish\\.mjs ready --tree ${TREE}$`, 'm'))
  assert.deepEqual(payloadOf(calls.find(c => c.opts.label === 'publish:M2-closeout').prompt), { verdict: 'VERIFIED', ok: true, tree: TREE })
  assert.equal(result.prReady, true)
  // A failing stage is published with its NOT OK result before the cycle stops; no ready.
  const failed = await simulate(script, { args: { milestone: 'M0' }, workflows: flows('wan-closeout').workflows, respond: respond() })
  assert.equal(failed.result.stoppedAt, 'closeout')
  assert.deepEqual(payloadOf(failed.calls.find(c => c.opts.label === 'publish:M0-closeout').prompt).verdict, 'NOT VERIFIED')
  assert.ok(!publishLabels(failed.calls).includes('publish:ready'))
  // Off a factory branch (standalone run): every publish is skipped with a log line, the cycle still runs.
  const off = await simulate(script, { args: { milestone: 'M2' }, workflows: flows().workflows, respond: respond(true, GATES, { 'publish': publisher(4, {}) }) })
  assert.equal(off.result.ok, true)
  assert.equal(off.result.prReady, false)
  assert.ok(off.logs.some(l => /publish: M2-survey not published/.test(l)))
})

test('publishing: a refused commit stops the cycle; publish:false makes no publish calls', async () => {
  await assert.rejects(simulate(script, { args: { milestone: 'M2' }, workflows: flows().workflows, respond: respond(true, GATES, { 'publish:M2-implement': publisher(3) }) }), /publish refused at M2-implement/)
  const { result, calls } = await simulate(script, { args: { milestone: 'M2', publish: false }, workflows: flows().workflows, respond: respond() })
  assert.equal(result.ok, true)
  assert.deepEqual(publishLabels(calls), [])
})

test('factory v2: the cycle stops for the human\'s one grant of the plan before implementing anything', async () => {
  const f = flows()
  const { result } = await simulate(script, { args: { milestone: 'M2' }, workflows: f.workflows, respond: respond(true, GATES, { 'gate:approval:plan': () => approved(PLAN_DOCS.slice(1)) }) })
  assert.deepEqual([result.ok, result.stoppedAt], [false, 'grant'])
  assert.match(result.next, /A human grants docs\/design\/HANDOUT-M2\.md/)
  assert.deepEqual(f.order.map(([n]) => n), ['wan-survey', 'wan-handout'])
})
