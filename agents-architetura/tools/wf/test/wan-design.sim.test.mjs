import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, finding, assertRoles } from './helpers.mjs'

const script = built('wan-design')
const ok = () => ({ summary: 'ok', files: [] })
const ws = (paths = [['transport', ['src/transport/']], ['train', ['src/train/']]]) =>
  ({ workstreams: paths.map(([key, ownedPaths]) => ({ key, title: key, scope: key, ownedPaths })) })
const base = {
  'brief:check': () => ({ present: true, bytes: 1200 }),
  'survey:': ok, 'draft': ok, 'sizegate:': ok, 'revise:': ok, 'ownership-fix:': ok, 'finalize': ok,
  'threats': () => ({ boundaries: ['client -> control plane'], threats: [
    { id: 'T1', layer: 'worker', threat: 'impersonated worker', scenario: 'joins without a credential', impact: 'poisoned update', mitigation: 'mTLS (design §5)', gap: false, test: 'tests/redteam/auth' },
    { id: 'T2', layer: 'runtime', threat: 'pickle in checkpoints', scenario: 'peer sends a crafted checkpoint', impact: 'RCE', mitigation: 'none', gap: true, test: 'tests/fuzz/ckpt' },
  ] }),
  'threat-model': ok, 'adr': ok, 'adr-fix': ok,
  'adr-review': () => ({ findings: [] }),
  'review:': () => ({ findings: [] }),
  'extract:': () => ws(),
}

test('happy path: surveys, draft, threat model, clean review, disjoint ownership, ADR-0001, finalize — approval left to a human', async () => {
  const { result, calls } = await simulate(script, { respond: byLabel(base) })
  assert.deepEqual(calls.filter(c => c.opts.label.startsWith('survey:')).map(c => c.opts.label), ['survey:brief', 'survey:prior-art', 'survey:repo'])
  assert.equal(result.review.converged, true)
  assert.deepEqual(result.ownership, { ok: true, errors: [], keys: ['transport', 'train'] })
  assertRoles(assert, calls, { 'brief:': 'wan-readonly', 'survey:': 'wan-researcher', 'draft': 'wan-researcher', 'threats': 'wan-security', 'threat-model': 'wan-researcher', 'review:': 'wan-reviewer', 'extract:': 'wan-readonly', 'adr-review': 'wan-reviewer', 'adr': 'wan-researcher', 'finalize': 'wan-researcher' })
  assert.equal(result.ok, true)
  assert.deepEqual([result.threats, result.threatGaps], [2, 1])
  assert.ok(calls.find(c => c.opts.label === 'threat-model').prompt.includes('For each of the 1 gap(s)'))
  assert.ok(calls.find(c => c.opts.label === 'draft').prompt.includes('T_total = T_compute + T_communication'))
  assert.ok(calls.find(c => c.opts.label === 'adr').prompt.includes('ARCHITECTURE DECISION REQUIRED'))
  assert.ok(!calls.find(c => c.opts.label === 'finalize').prompt.includes('Status: DRAFT'))
  assert.match(result.next, /approve\.mjs grant docs\/design\/DESIGN\.md docs\/design\/adr\/ADR-0001-initial-architecture\.md/)
})

test('no threat model stops the design; ADR review findings are fixed once', async () => {
  await assert.rejects(simulate(script, { respond: byLabel({ ...base, 'threats': () => ({ boundaries: [], threats: [] }) }) }), /no threat model/)
  const { calls } = await simulate(script, { respond: byLabel({ ...base, 'adr-review': () => ({ findings: [finding('major')] }) }) })
  assert.ok(calls.some(c => c.opts.label === 'adr-fix'))
})

test('a missing or empty brief stops the workflow before any survey', async () => {
  for (const status of [{ present: false, bytes: 0 }, { present: true, bytes: 0 }, null]) {
    const calls = []
    await assert.rejects(
      simulate(script, { respond: byLabel({ ...base, 'brief:check': () => status, 'survey:': (p, o) => { calls.push(o.label); return ok() } }) }),
      /BRIEF.md is missing or empty/,
    )
    assert.deepEqual(calls, [])
  }
})

test('a failed surveyor aborts before drafting', async () => {
  await assert.rejects(
    simulate(script, { respond: byLabel({ ...base, 'survey:': (p, o) => (o.label === 'survey:prior-art' ? null : ok()) }) }),
    /survey incomplete \(2\/3\)/,
  )
})

test('review findings trigger revise + size gate, then a fresh round', async () => {
  const { result, calls } = await simulate(script, {
    respond: byLabel({ ...base, 'review:': (p, o) => ({ findings: o.label === 'review:buildability:r1' ? [finding('major')] : [] }) }),
  })
  assert.deepEqual(result.review.findingsPerRound, [1, 0])
  assert.ok(calls.some(c => c.opts.label === 'revise:r1'))
  assert.ok(calls.some(c => c.opts.label === 'sizegate:r1'))
})

test('overlapping ownership is repaired, and reported if it never converges', async () => {
  const bad = ws([['transport', ['src/']], ['train', ['src/train/']]])
  const fixed = await simulate(script, { respond: byLabel({ ...base, 'extract:': (p, o, i, n) => (n === 0 ? bad : ws()) }) })
  assert.equal(fixed.result.ownership.ok, true)
  assert.ok(fixed.calls.some(c => c.opts.label === 'ownership-fix:a1'))

  const never = await simulate(script, { respond: byLabel({ ...base, 'extract:': () => bad }) })
  assert.equal(never.result.ownership.ok, false)
  assert.match(never.result.ownership.errors[0], /scope collision/)
  assert.ok(never.calls.find(c => c.opts.label === 'finalize').prompt.includes('ownership is still invalid'))
})

test('non-convergence is surfaced to the finalizer as open questions', async () => {
  const { result, calls } = await simulate(script, {
    respond: byLabel({ ...base, 'review:': () => ({ findings: [finding('blocker')] }) }),
  })
  assert.equal(result.review.converged, false)
  // a count that never falls stalls after three rounds instead of spending the whole budget
  assert.deepEqual([result.review.rounds, result.review.stalled], [3, true])
  assert.ok(calls.find(c => c.opts.label === 'finalize').prompt.includes('did NOT converge'))
})

test('a workstream claiming a harness directory is caught by the ownership check', async () => {
  const bad = ws([['transport', ['src/transport/', 'tests/fuzz/']], ['train', ['src/train/']]])
  const { result, calls } = await simulate(script, { respond: byLabel({ ...base, 'extract:': (p, o, i, n) => (n === 0 ? bad : ws()) }) })
  assert.match(calls.find(c => c.opts.label === 'ownership-fix:a1').prompt, /scope collision: harness owns tests\/fuzz\//)
  assert.equal(result.ownership.ok, true)
})

test('a workstream claiming the agent factory is caught by the ownership check', async () => {
  const bad = ws([['transport', ['src/transport/', 'agents-architetura/tools/verify/']], ['train', ['src/train/']]])
  const { result, calls } = await simulate(script, { respond: byLabel({ ...base, 'extract:': (p, o, i, n) => (n === 0 ? bad : ws()) }) })
  assert.match(calls.find(c => c.opts.label === 'ownership-fix:a1').prompt, /scope collision: factory owns agents-architetura\//)
  assert.equal(result.ownership.ok, true)
})
