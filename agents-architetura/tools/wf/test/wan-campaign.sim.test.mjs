import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, ladder, finding, assertRoles, approved } from './helpers.mjs'

const script = built('wan-campaign')
const TREE = 'b'.repeat(40)
const fam = (id, count, extra = {}) => ({ id, kind: 'tsan-data-race', count, signature: `tsan-data-race|${id}`, locations: [`src/${id}.rs:1`], message: 'data race', ...extra })
const round = (families, extra = {}) => ({ applicable: true, campaign: 'rust-tsan', round: 0, tree: TREE, broken: false, unsuppressedReports: families.reduce((n, f) => n + f.count, 0), families, suppressed: [], suppressionViolations: [], delta: { new: [], fixed: [], persisting: [] }, ...extra })
const rulingFor = (f, ruling = 'protocol-fix', files = [`src/${f.id}.rs`]) => ({ familyId: f.id, ruling, fixShape: 'use the concurrent accessor', files })
const R0 = [fam('stats', 30), fam('peer', 10)]
const base = (rounds, extra = {}) => ({
  'gate:approval': () => approved(),
  'round:r': (p, o) => rounds[Number(o.label.match(/round:r(\d+)/)[1])],
  'rule:': p => ({ rulings: R0.concat(rounds.flatMap(r => r.families)).map(f => rulingFor(f, f.id === 'peer' ? 'real-bug' : 'protocol-fix')) }),
  'audit:snapshot:': () => ({ tree: TREE }),
  'audit:check:': () => ({ tree: TREE, changed: [], violations: [] }),
  'fix:': () => ({ summary: 'fixed; happens-before: publish after init', files: ['x'] }),
  'build:': () => ({ allGreen: true, ladder: ladder(true), items: [] }),
  'review:': () => ({ findings: [] }),
  'amend:': () => ({ summary: 'ok', files: [] }),
  'final': () => ({ allGreen: true, ladder: ladder(true), items: [] }),
  'record': () => ({ summary: 'written', files: [] }),
  'report': () => ({ summary: 'ok', files: [] }),
  ...extra,
})
const run = (rounds, extra = {}, opts = {}) => {
  const called = []
  return simulate(script, {
    args: { campaign: 'rust-tsan', ...(opts.args ?? {}) },
    respond: byLabel(base(rounds, extra)),
    workflows: { 'wan-bughunt': a => { called.push(['bughunt', a]); return { solved: true } }, 'wan-fix': a => { called.push(['fix', a]); return { ok: opts.fixOk ?? true } }, ...(opts.workflows ?? {}) },
  }).then(r => ({ ...r, called }))
}

test('40 -> 12 -> 0: waves of disjoint fixers, one re-run per wave, final gate, results recorded', async () => {
  const rounds = [round(R0), round([fam('stats', 12)], { round: 1 }), round([], { round: 2 })]
  const { result, calls } = await run(rounds)
  assert.equal(result.ok, true)
  assert.deepEqual(result.rounds.map(r => r.unsuppressed), [40, 12, 0])
  assert.equal(result.waves, 2)
  const order = calls.map(c => c.opts.label).filter(l => /^(round|fix|build|audit|final)/.test(l))
  assert.deepEqual(order, ['round:r0', 'audit:snapshot:w1', 'fix:peer:w1', 'fix:stats:w1', 'audit:check:campaign-w1', 'build:w1', 'round:r1',
    'audit:snapshot:w2', 'fix:stats:w2', 'audit:check:campaign-w2', 'build:w2', 'round:r2', 'final'])
  assert.ok(calls.find(c => c.opts.label === 'fix:peer:w1').prompt.includes('REAL BUG'))
  assert.ok(calls.find(c => c.opts.label === 'record').prompt.includes('node agents-architetura/tools/campaign/record.mjs campaign rust-tsan --b64 '))
  assertRoles(assert, calls, { 'round:': 'wan-verifier', 'rule:': 'wan-researcher', 'fix:': 'wan-coder', 'build:': 'wan-verifier', 'review:': 'wan-reviewer', 'final': 'wan-verifier', 'report': 'wan-researcher' })
})

test('not applicable is honest n/a; a broken snapshot is never zero', async () => {
  const na = await run([{ applicable: false, reason: 'Cargo.toml absent' }])
  assert.deepEqual([na.result.ok, na.result.applicable], [true, false])
  const broken = await run([round([], { broken: true, brokenReason: 'could not compile' })])
  assert.equal(broken.result.ok, false)
  assert.equal(broken.result.stoppedAt, 'snapshot')
})

test('a stalled count sends the top family to wan-bughunt; stalling again after that stops for a human', async () => {
  const stuck = round([fam('stats', 12)])
  const rounds = [round(R0), stuck, stuck, stuck, stuck, stuck]
  const { result, called } = await run(rounds, {}, { args: { maxWaves: 6 } })
  assert.equal(result.ok, false)
  assert.deepEqual(called.filter(c => c[0] === 'bughunt').map(c => c[1].id), ['rust-tsan-stats'])
  assert.match(result.stopReason, /stalled at 12 even after wan-bughunt on stats/)
})

test('a fixer that finds the ruling wrong changes nothing and the family is re-ruled with its evidence', async () => {
  const rounds = [round([fam('stats', 5)]), round([fam('stats', 5)], { round: 1 }), round([], { round: 2 })]
  const { result, calls } = await run(rounds, { 'fix:stats:w1': () => ({ summary: 'stopped', files: [], misruled: 'this counter gates a free; not benign' }) })
  assert.ok(calls.some(c => c.opts.label === 'rule:r1' && c.prompt.includes('this counter gates a free')))
  assert.equal(result.ok, true)
})

test('benign-by-spec rulings are applied by re-reading the stored round with the new suppressions', async () => {
  const benign = { 'rule:': () => ({ rulings: [rulingFor(fam('stats', 3), 'benign-by-spec', [])] }), 'reparse:': () => round([]) }
  const { result, calls } = await run([round([fam('stats', 3)])], benign)
  assert.equal(result.ok, true)
  assert.ok(calls.some(c => c.opts.label.startsWith('reparse:r0') && c.prompt.includes('--round 0 --json') && !c.prompt.includes('--run')))
  assert.ok(!calls.some(c => c.opts.label.startsWith('fix:')))
})

test('review findings trigger an amend; a red build goes to wan-fix; an unrepaired build stops the campaign', async () => {
  const rounds = [round(R0), round([], { round: 1 })]
  const amended = await run(rounds, { 'review:benign-or-bug:w1': () => ({ findings: [finding('blocker')] }) })
  assert.ok(amended.calls.some(c => c.opts.label === 'amend:w1'))
  const red = await run(rounds, { 'build:': () => ({ allGreen: false, ladder: ladder(false, ['build']), items: [] }) }, { fixOk: false })
  assert.equal(red.result.ok, false)
  assert.match(red.result.stopReason, /broke the build/)
})

test('zero reports but a suppressions-audit violation in the final gate is not ok; scope violations abort', async () => {
  const r = await run([round([])], { 'final': () => ({ allGreen: false, ladder: ladder(true), items: [{ id: 'sup', rung: 'discipline', symptom: 'unjustified', evidence: 'x', scope: ['agents-architetura/tools/campaign/suppressions/rust-tsan.txt'] }] }) })
  assert.equal(r.result.ok, false)
  assert.match(r.result.stopReason, /suppressions audit is red/)
  await assert.rejects(run([round(R0)], { 'audit:check:': () => ({ tree: TREE, changed: ['src/other.rs'], violations: ['src/other.rs'] }) }), /wrote outside its owned paths: src\/other\.rs/)
  await assert.rejects(simulate(script, { args: { campaign: '../x' }, respond: () => null }), /args.campaign must be/)
})
