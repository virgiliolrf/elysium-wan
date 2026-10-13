import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, finding, approved, notApproved, assertRoles } from './helpers.mjs'

const script = built('wan-adr')
const ok = () => ({ summary: 'ok', files: [] })
const base = {
  'gate:approval': () => approved(),
  'adr:next': () => ({ next: 'ADR-0003' }),
  'research': () => ({ bottleneck: 'all-reduce over 80 ms RTT dominates (T_comm = 2(N-1)/N * G/B)', options: [{ idea: 'outer-step sync every H steps', source: 'DiLoCo (arXiv 2311.08105)', mechanism: 'communicate every H steps', effectOnBottleneck: 'T_comm / H', tradeoffs: 'staleness', experiment: 'H in {1,10,100} on the local simulator' }] }),
  'draft': ok, 'fix:': ok, 'ideas:': ok, 'references:': ok,
  'review:': () => ({ findings: [] }),
  'present': () => ({ block: 'ARCHITECTURE DECISION REQUIRED\nProblem: ...\nRecommendation: PROCEED' }),
  // doc-lines: every ADR the prompt names, within budget unless a test says otherwise
  'size:': lines(() => 120),
  'gate:approval-base': () => approved(),
}
function lines(of) {
  return prompt => ({ files: (/doc-lines\.mjs (.+?) --json/.exec(prompt)?.[1] ?? '').split(' ').map(path => ({ path, exists: of(path) > 0, lines: of(path) })) })
}
const run = (extra = {}, args = { problem: 'gradient all-reduce over the WAN dominates step time at 8 workers', id: 'outer-sync' }) => simulate(script, { args, respond: byLabel({ ...base, ...extra }) })

test('research -> first-principles ADR with amendments on the record -> review -> STOP for the human', async () => {
  const { result, calls, logs } = await run()
  assert.deepEqual([result.ok, result.awaitingHuman, result.adr], [true, true, 'docs/design/adr/ADR-0003-outer-sync.md'])
  assert.equal(result.command, 'node agents-architetura/tools/factory/approve.mjs grant docs/design/adr/ADR-0003-outer-sync.md docs/design/DESIGN.md')
  const draft = calls.find(c => c.opts.label === 'draft').prompt
  assert.ok(draft.includes('freeze.mjs amend <spec>') && draft.includes('--by ADR-0003'))
  assert.ok(draft.includes('Falsification Criteria'))
  assert.ok(logs.some(l => l.includes('ARCHITECTURE DECISION REQUIRED')))
  assertRoles(assert, calls, { 'gate:': 'wan-verifier', 'adr:next': 'wan-readonly', 'research:': 'wan-researcher', 'draft': 'wan-architect', 'size:': 'wan-verifier', 'review:': 'wan-reviewer', 'present': 'wan-readonly' })
  assert.ok(!calls.some(c => c.opts.label.startsWith('split:')), 'an ADR within budget is not split')
})

test('an unapproved design, a bad ADR number or no research options stop it; review findings are fixed', async () => {
  await assert.rejects(run({ 'gate:approval': () => notApproved() }), /not approved by a human/)
  await assert.rejects(run({ 'adr:next': () => ({ next: 'ADR-3; rm' }) }), /next ADR number/)
  await assert.rejects(run({ 'research': () => ({ bottleneck: 'x', options: [] }) }), /no options/)
  const fixed = await run({ 'review:ADR-0003:first-principles:r1': () => ({ findings: [finding('blocker')] }) })
  assert.ok(fixed.calls.some(c => c.opts.label === 'fix:ADR-0003:r1'))
  await assert.rejects(run({ 'size:': lines(() => 0) }), /ADR-0003-outer-sync\.md does not exist/)
  await assert.rejects(simulate(script, { args: { problem: 'short' }, respond: () => null }), /args.problem/)
})

test('thesis-style: researchers in parallel per angle, the architect drafts from all of them, reviewers with the web bring ideas the architect must adopt or reject', async () => {
  const idea = { idea: 'per-stage homogeneous device classes', source: 'derivation', why: 'native speed, exact per device map' }
  const { result, calls, logs } = await run({
    'review:ADR-0003:frontier:r1': () => ({ findings: [finding('major')], ideas: [idea] }),
    'review:ADR-0003:frontier:r2': () => ({ findings: [], ideas: [{ idea: 'later idea', source: 's', why: 'w' }] }),
  })
  assert.equal(result.ok, true)
  const research = calls.filter(c => c.opts.label.startsWith('research:'))
  assert.deepEqual(research.map(c => c.opts.label), ['research:numerics', 'research:systems', 'research:hardware', 'research:performance', 'research:contrarian'])
  assert.ok(research.every(c => c.prompt.includes('You MAY use web search')))
  assert.ok(logs.some(l => l.includes('Research: 5/5 angles reported')))
  const draft = calls.find(c => c.opts.label === 'draft')
  assert.equal(draft.opts.agentType, 'wan-architect')
  assert.ok(draft.prompt.includes('<untrusted_research>') && draft.prompt.includes('"angle":"systems"'))
  const reviews = calls.filter(c => c.opts.label.startsWith('review:'))
  assert.ok(reviews.every(c => c.prompt.includes('You MAY use web search') && c.prompt.includes('"ideas"')))
  assert.ok(reviews.some(c => c.opts.label === 'review:ADR-0003:frontier:r1'))
  const fix = calls.find(c => c.opts.label === 'fix:ADR-0003:r1')
  assert.equal(fix.opts.agentType, 'wan-architect')
  assert.ok(fix.prompt.includes('per-stage homogeneous device classes') && fix.prompt.includes('Ideas considered'))
  const record = calls.find(c => c.opts.label === 'ideas:record:ADR-0003')
  assert.ok(record && record.prompt.includes('later idea'), 'ideas from the converged round are recorded')
  const custom = await run({}, { problem: 'gradient all-reduce over the WAN dominates step time at 8 workers', id: 'x', angles: ['a first angle', 'a second angle'] })
  assert.deepEqual(custom.calls.filter(c => c.opts.label.startsWith('research:')).map(c => c.opts.label), ['research:angle-1', 'research:angle-2'])
})

test('revise mode: no research, no draft; the reviewers verify the human references on the web and the architect places them', async () => {
  const refs = ['H2 hyper-heterogeneous training over 1,024 chips of four architectures: DiTorch, DiComm, HeteroPP', 'OPEN-1B, a fully auditable training run: bitwise replay on heterogeneous commodity hardware']
  const { result, calls, logs } = await run({ 'review:ADR-0001:frontier:r1': () => ({ findings: [finding('major')], ideas: [] }) }, { revise: 'docs/design/adr/ADR-0001-hetero-native-per-class.md', references: refs })
  assert.deepEqual([result.ok, result.adr, result.revised, result.references], [true, 'docs/design/adr/ADR-0001-hetero-native-per-class.md', true, 2])
  for (const skipped of ['adr:next', 'research:', 'draft']) assert.ok(!calls.some(c => c.opts.label.startsWith(skipped)), `${skipped} must not run`)
  const review = calls.find(c => c.opts.label === 'review:ADR-0001:consistency:r1').prompt
  assert.ok(review.includes('VERIFY each one on the web') && review.includes('<untrusted_references>') && review.includes('DiComm'))
  const fix = calls.find(c => c.opts.label === 'fix:ADR-0001:r1')
  assert.equal(fix.opts.agentType, 'wan-architect')
  assert.ok(fix.prompt.includes('Place every reference the') && fix.prompt.includes('--by ADR-0001'))
  assert.ok(logs.some(l => l.includes('revising with 2 reference(s)')))
  await assert.rejects(run({}, { revise: 'docs/design/DESIGN.md', references: refs }), /args\.revise must be an existing ADR path/)
  // a review that converges at once, or files the references only as ideas, still places them
  const clean = await run({ 'review:ADR-0001:frontier:r1': () => ({ findings: [], ideas: [{ idea: 'cite H2', source: 'arXiv', why: 'closest system' }] }) }, { revise: 'docs/design/adr/ADR-0001-hetero-native-per-class.md', references: refs })
  assert.ok(!clean.calls.some(c => c.opts.label.startsWith('fix:')))
  const place = clean.calls.find(c => c.opts.label === 'references:place:ADR-0001')
  assert.ok(place && place.prompt.includes('DiComm') && place.prompt.includes('cite H2') && place.opts.agentType === 'wan-architect')
})

test('revise and split start from an ADR in progress: the committed design must be granted, not its working copy', async () => {
  const args = { revise: 'docs/design/adr/ADR-0001-hetero-native-per-class.md' }
  const { calls } = await run({ 'gate:approval': () => notApproved() }, args)
  assert.ok(calls.some(c => c.opts.label === 'gate:approval-base' && c.prompt.includes('approved-base.mjs docs/design/DESIGN.md --json')))
  assert.ok(!calls.some(c => c.opts.label === 'gate:approval'), 'the working copy carries the pending amendments')
  await assert.rejects(run({ 'gate:approval-base': () => notApproved() }, args), /the committed docs\/design\/DESIGN\.md is not granted by a human/)
  await assert.rejects(run({}, { revise: args.revise, split: args.revise }), /exclusive/)
})

const SPLIT_INTO = ['docs/design/adr/ADR-0001-hetero-native-per-class.md', 'docs/design/adr/ADR-0004-exact-accumulation.md']
const splitTo = (parts = SPLIT_INTO) => () => ({ adrs: parts.map(path => ({ path, decision: `decide ${path}` })), notes: ['docs/research/hetero.md'] })
// a split that keeps the ADR it was given first, and adds one new number per call
let nextNumber = 10
const splitEach = () => prompt => {
  const path = /\n(docs\/design\/adr\/ADR-\d{4}-[a-z0-9-]+\.md) is /.exec(prompt)[1]
  return splitTo([path, `docs/design/adr/ADR-00${nextNumber++}-part.md`])()
}

test('split mode: one-decision ADRs within budget, each reviewed on its own, all granted together', async () => {
  const refs = ['OPEN-1B: bitwise replay on heterogeneous hardware']
  const { result, calls, logs } = await run({ 'split:': splitTo() }, { split: SPLIT_INTO[0], references: refs })
  assert.equal(result.ok, true)
  assert.deepEqual(result.adrs.map(a => a.adr), SPLIT_INTO)
  assert.equal(result.command, `node agents-architetura/tools/factory/approve.mjs grant ${SPLIT_INTO.join(' ')} docs/design/DESIGN.md`)
  const split = calls.find(c => c.opts.label === 'split:ADR-0001')
  assert.equal(split.opts.agentType, 'wan-architect')
  assert.ok(split.prompt.includes('ONE decision') && split.prompt.includes('at most\n400 lines') && split.prompt.includes('docs/research/<slug>.md'))
  for (const id of ['ADR-0001', 'ADR-0004']) {
    assert.ok(calls.some(c => c.opts.label === `review:${id}:first-principles:r1`), id)
    assert.ok(calls.some(c => c.opts.label === `references:place:${id}`), `${id}: the references are placed in each part`)
  }
  assert.ok(logs.some(l => l.includes('Split docs/design/adr/ADR-0001-hetero-native-per-class.md (to be split, as the human asked)')))
  for (const bad of [[SPLIT_INTO[0]], [SPLIT_INTO[1], SPLIT_INTO[0]], [SPLIT_INTO[0], 'docs/design/adr/ADR-0001-dup.md'], [SPLIT_INTO[0], 'docs/design/x.md']]) {
    await assert.rejects(run({ 'split:': splitTo(bad) }, { split: SPLIT_INTO[0] }), /2-4 ADR paths, the first/)
  }
  await assert.rejects(run({ 'split:': splitTo(), 'size:': lines(p => (p.includes('0004') ? 0 : 50)) }, { split: SPLIT_INTO[0] }), /did not write docs\/design\/adr\/ADR-0004/)
})

test('a draft over the size budget is split before any review; a review that stalls splits the ADR and reviews the parts', async () => {
  const over = await run({ 'size:ADR-0003': lines(() => 2006), 'split:': splitTo(['docs/design/adr/ADR-0003-outer-sync.md', 'docs/design/adr/ADR-0005-outer-sync-research.md']) })
  const order = over.calls.map(c => c.opts.label)
  assert.ok(order.indexOf('split:ADR-0003') < order.findIndex(l => l.startsWith('review:')), 'split before review')
  assert.ok(over.logs.some(l => l.includes('2006 lines, over the 400-line budget')))
  assert.deepEqual(over.result.adrs.map(a => a.adr), ['docs/design/adr/ADR-0003-outer-sync.md', 'docs/design/adr/ADR-0005-outer-sync-research.md'])
  // 17 -> 17 -> 19 majors: stalled, split, the parts reviewed (and converging)
  const counts = [17, 17, 19]
  const stalled = await run({
    'review:ADR-0003:first-principles:': (p, o, i, n) => ({ findings: n < 3 ? Array.from({ length: counts[n] }, () => finding('major')) : [] }),
    'split:': splitTo(['docs/design/adr/ADR-0003-outer-sync.md', 'docs/design/adr/ADR-0006-outer-sync-b.md']),
  })
  assert.ok(stalled.calls.some(c => c.opts.label === 'split:ADR-0003' && c.prompt.includes('stalled under review (17 -> 17 -> 19')))
  assert.ok(stalled.calls.some(c => c.opts.label === 'review:ADR-0006:frontier:r1'))
  assert.equal(stalled.result.ok, true)
  // a part that stalls again at the depth limit goes to the human, not into an endless split
  const always = await run({ 'review:': () => ({ findings: [finding('major')] }), 'split:': splitEach() })
  assert.equal(always.result.ok, false)
  assert.ok(always.result.adrs.every(a => a.stalled && !a.failed), JSON.stringify(always.result.adrs))
  assert.equal(always.result.adrs.length, 4)
  // a split that breaks its contract is reported as failed, not as a decision
  const broken = await run({ 'review:': () => ({ findings: [finding('major')] }), 'split:': (p, o, i, n) => (n === 0 ? splitTo(['docs/design/adr/ADR-0003-outer-sync.md', 'docs/design/adr/ADR-0008-y.md'])() : { adrs: [] }) })
  assert.ok(broken.result.adrs.some(a => a.failed) && broken.logs.some(l => l.includes('its review or split failed')))
  assert.equal(always.calls.filter(c => c.opts.label.startsWith('split:')).length, 3, 'the draft, then each of its two parts once')
})

test('from round 3 the decision is frozen: fixes correct without redesigning, and ideas become follow-ups', async () => {
  const falling = [5, 4, 3, 1]
  const { calls } = await run({ 'review:ADR-0003:first-principles:': (p, o, i, n) => ({ findings: Array.from({ length: falling[n] ?? 0 }, () => finding('major')), ideas: [{ idea: 'a radix tree', source: 's', why: 'w' }] }) })
  const fix = r => calls.find(c => c.opts.label === `fix:ADR-0003:r${r}`).prompt
  assert.ok(fix(2).includes('adopt it and\nredo the derivation') && !fix(2).includes('frozen'))
  assert.ok(fix(3).includes('the decision and\nits mechanism are frozen') && fix(3).includes('without adopting it') && !fix(3).includes('adopt it and\nredo'))
  const review = r => calls.find(c => c.opts.label === `review:ADR-0003:frontier:r${r}`).prompt
  assert.ok(review(1).includes('Think beyond the draft') && review(3).includes('never "a better mechanism exists"'))
})
