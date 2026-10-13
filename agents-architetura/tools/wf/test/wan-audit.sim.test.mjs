import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, assertRoles, approved } from './helpers.mjs'

const script = built('wan-audit')
const TREE = 'c'.repeat(40)
const CLASSES = [
  { id: 'MC-DESER', failingInvariant: 'untrusted bytes never reach a code-executing decoder', exemplars: ['CVE-2025-32434'], surface: 'checkpoint loader', fixShape: 'safetensors + signature check' },
  { id: 'MC-BYZ', failingInvariant: 'one peer cannot move the aggregate beyond the robust bound', exemplars: ['Krum break (Baruch 2019)'], surface: 'aggregator', fixShape: 'bounded robust aggregation' },
]
const resistant = [{ classId: 'MC-DESER', surfaceId: 'S1', outcome: 'resistant', evidence: '3 passed' }, { classId: 'MC-BYZ', surfaceId: 'S1', outcome: 'resistant', evidence: '2 passed' }]
const base = {
  'gate:approval': () => approved(),
  'extract:classes': () => ({ classes: [] }),
  'research:': () => ({ summary: 'ok', files: [] }),
  'catalog': () => ({ classes: CLASSES }),
  'map:MC-DESER': () => ({ surfaces: [{ id: 'S1', location: 'src/ckpt/load.py:10', verdict: 'needs-test', argument: 'loads peer bytes', test: 'pickle payload' }, { id: 'S2', location: 'src/ckpt/save.py:3', verdict: 'immune', argument: 'writes only' }] }),
  'map:MC-BYZ': () => ({ surfaces: [{ id: 'S1', location: 'src/agg/mean.py:5', verdict: 'needs-test', argument: 'plain mean', test: 'one huge update' }] }),
  'audit:snapshot:': () => ({ tree: TREE }),
  'audit:check:': () => ({ tree: TREE, changed: [], violations: [] }),
  'cve-test:': () => ({ summary: 'ok', files: [] }),
  'run:': () => ({ results: resistant }),
  'refute:': () => ({ verdict: 'survives', argument: 'real' }),
  'record:': () => ({ summary: 'written', files: [] }),
  'results-doc': () => ({ summary: 'ok', files: [] }),
}
const decode = (prompt, target) => JSON.parse(Buffer.from(prompt.match(new RegExp(`record\\.mjs ${target} --b64 (\\S+)`))[1], 'base64').toString('utf8'))
const run = (extra = {}, args = {}, workflows = {}) => simulate(script, { args, respond: byLabel({ ...base, ...extra }), workflows: { 'wan-fix': () => ({ ok: true }), ...workflows } })

test('research -> catalog -> maps -> tests -> run: all resistant means no open tier-A and nothing unverdicted', async () => {
  const { result, calls } = await run()
  assert.equal(result.ok, true)
  assert.deepEqual([result.classes, result.surfaces, result.openTierA, result.unverdicted], [2, 3, 0, 0])
  const rec = calls.find(c => c.opts.label === 'record:r1')
  assert.deepEqual(decode(rec.prompt, 'audit-results').verdicts, { immune: 3, SUSCEPTIBLE: 0, INCONCLUSIVE: 0 })
  assert.deepEqual(decode(rec.prompt, 'findings').findings.map(f => [f.id, f.status]), [['SEC-MC-DESER-S1', 'closed'], ['SEC-MC-BYZ-S1', 'closed']])
  assert.ok(calls.find(c => c.opts.label === 'cve-test:MC-BYZ').prompt.includes('covers: MC-BYZ-<surface id>'))
  assertRoles(assert, calls, { 'research:': 'wan-researcher', 'catalog': 'wan-researcher', 'map:': 'wan-researcher', 'cve-test:': 'wan-redteam', 'run:': 'wan-verifier', 'record:': 'wan-verifier' })
})

test('a confirmed susceptible surface is open tier-A and an open finding; a refuted one is INCONCLUSIVE', async () => {
  const sus = [resistant[0], { classId: 'MC-BYZ', surfaceId: 'S1', outcome: 'susceptible', evidence: 'aggregate moved 1e9', reproCommand: 'uv run pytest tests/redteam/cve/mc-byz' }]
  const confirmed = await run({ 'run:': () => ({ results: sus }) })
  assert.equal(confirmed.result.ok, false)
  assert.equal(confirmed.result.openTierA, 1)
  assert.deepEqual(decode(confirmed.calls.find(c => c.opts.label === 'record:r1').prompt, 'findings').findings.find(f => f.id === 'SEC-MC-BYZ-S1').status, 'open')
  const refuted = await run({ 'run:': () => ({ results: sus }), 'refute:': (p, o) => ({ verdict: o.label.endsWith(':0:r1') ? 'refuted' : 'survives', argument: 'the test asserts more than the design promises' }) })
  assert.deepEqual([refuted.result.openTierA, refuted.result.unverdicted], [0, 1])
  const split = decode(refuted.calls.find(c => c.opts.label === 'record:r1').prompt, 'findings').findings.find(f => f.id === 'SEC-MC-BYZ-S1')
  assert.equal(split.status, 'needs-human')
})

test('close: SUSCEPTIBLE surfaces go through wan-fix on the redteam rung and are re-run', async () => {
  let runs = 0
  const sus = [resistant[0], { classId: 'MC-BYZ', surfaceId: 'S1', outcome: 'susceptible', evidence: 'moved', reproCommand: 'x' }]
  const fixes = []
  const { result, calls } = await run({ 'run:': () => ({ results: runs++ === 0 ? sus : [resistant[1]] }) }, { close: true }, { 'wan-fix': a => { fixes.push(a); return { ok: true } } })
  assert.deepEqual(fixes, [{ rungs: ['redteam'] }])
  assert.ok(calls.some(c => c.opts.label === 'run:close' && c.prompt.includes('tests/redteam/cve/mc-byz/')))
  assert.equal(result.ok, true)
})

test('an existing catalog skips research on pass 1; pass 2 re-researches and keeps the taxonomy', async () => {
  const cached = await run({ 'extract:classes': () => ({ classes: CLASSES }) })
  assert.ok(!cached.calls.some(c => c.opts.label.startsWith('research:')))
  const second = await run({ 'extract:classes': () => ({ classes: CLASSES }) }, { pass: 2 })
  assert.equal(second.calls.filter(c => c.opts.label.startsWith('research:')).length, 3)
  assert.ok(second.calls.find(c => c.opts.label === 'catalog').prompt.includes('KEEP the existing taxonomy'))
})

test('unmapped classes are unverdicted; tests written outside tests/redteam/cve abort', async () => {
  const unmapped = await run({ 'map:MC-BYZ': () => null })
  assert.equal(unmapped.result.unverdicted, 1)
  assert.deepEqual(unmapped.result.unmappedClasses, ['MC-BYZ'])
  await assert.rejects(run({ 'audit:check:': () => ({ tree: TREE, changed: ['src/agg/mean.py'], violations: ['src/agg/mean.py'] }) }), /wrote outside its owned paths/)
})
