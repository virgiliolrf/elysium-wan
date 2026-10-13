import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, finding, assertRoles, approved } from './helpers.mjs'

const script = built('wan-corpus')
const TREE = 'd'.repeat(40)
const good = t => ({ test: t, claim: 'catches', repeatRuns: 20, repeatPassed: 20, amplifyClean: true, seconds: 4 })
const base = {
  'gate:approval': () => approved(),
  'audit:snapshot:': () => ({ tree: TREE }),
  'audit:check:': () => ({ tree: TREE, changed: [], violations: [] }),
  'author:': () => ({ summary: 'ok', files: [] }),
  'review:': () => ({ findings: [] }),
  'revise:': () => ({ summary: 'ok', files: [] }),
  'validate:chaos': () => ({ results: [good('tests/chaos/staging/test_partition.py'), { ...good('tests/chaos/staging/test_vacuous.py'), claim: 'vacuous' }] }),
  'validate:semantics': () => ({ results: [{ ...good('tests/semantics/staging/test_flaky.py'), repeatPassed: 19 }, { ...good('tests/semantics/staging/test_slow.py'), seconds: 95 }, { ...good('tests/semantics/staging/test_amp.py'), amplifyClean: false }, good('tests/unit/test_escape.py')] }),
  'promote': () => ({ summary: 'ok', files: [] }),
  'report': () => ({ summary: 'ok', files: [] }),
}
const run = (extra = {}, args = { suites: ['chaos', 'semantics'] }) => simulate(script, { args, respond: byLabel({ ...base, ...extra }) })

test('only tests whose own regression makes them fail, 20/20 clean, amplifier-clean and fast are promoted', async () => {
  const { result, calls } = await run()
  assert.equal(result.accepted, 1)
  assert.deepEqual(result.rejected.map(r => [r.test, r.why.split(' (')[0].split(' —')[0]]), [
    ['tests/chaos/staging/test_vacuous.py', 'claim vacuous'],
    ['tests/semantics/staging/test_flaky.py', '19/20 clean runs'],
    ['tests/semantics/staging/test_slow.py', '95s exceeds the 30s budget'],
    ['tests/semantics/staging/test_amp.py', 'fails under the amplifier'],
    ['tests/unit/test_escape.py', 'not a staged test path'],
  ])
  assert.ok(calls.find(c => c.opts.label === 'promote').prompt.includes('tests/chaos/staging/test_partition.py'))
  assert.ok(calls.find(c => c.opts.label === 'author:chaos').prompt.includes('.would-fail.patch'))
  assert.equal(result.ok, true)
  const none = await run({ 'validate:chaos': () => ({ results: [] }), 'validate:semantics': () => ({ results: [] }) })
  assert.equal(none.result.ok, false)
  assertRoles(assert, calls, { 'author:': 'wan-coder', 'review:': 'wan-reviewer', 'validate:': 'wan-verifier', 'promote': 'wan-coder', 'report': 'wan-researcher' })
})

test('review findings loop through the reviser; all accepted and converged is ok; bad suite names refused', async () => {
  const { result, calls } = await run({ 'review:chaos:would-it-catch:r1': () => ({ findings: [finding('blocker')] }), 'validate:semantics': () => ({ results: [good('tests/semantics/staging/test_a.py')] }), 'validate:chaos': () => ({ results: [good('tests/chaos/staging/test_b.py')] }) })
  assert.ok(calls.some(c => c.opts.label === 'revise:chaos:r1'))
  assert.equal(result.ok, true)
  await assert.rejects(run({}, { suites: ['../x', 'unit'] }), /no valid suite names/)
})
