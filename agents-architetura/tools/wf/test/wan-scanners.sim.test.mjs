import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, assertRoles, approved } from './helpers.mjs'

const script = built('wan-scanners')
const PICKLE = { id: 'scan-semgrep-1', count: 2, signature: 'scan|semgrep|pickle|src/ckpt.py|x', locations: ['src/ckpt.py:12'], message: 'pickle on untrusted data' }
const ASSERT = { id: 'scan-bandit-2', count: 1, signature: 'scan|bandit|B101|src/a.py|assert', locations: ['src/a.py:3'], message: 'assert used' }
const battery = { applicable: true, ok: false, findings: 3, unavailable: ['codeql'], scanners: [
  { id: 'semgrep', status: 'ran', findings: 2, families: [PICKLE], suppressionViolations: [] },
  { id: 'bandit', status: 'ran', findings: 1, families: [ASSERT], suppressionViolations: [] },
  { id: 'codeql', status: 'skipped-unavailable', reason: 'codeql not installed' },
] }
const base = {
  'gate:approval': () => approved(),
  'battery': () => ({ json: battery, exitCode: 1 }),
  'triage:': () => ({ verdicts: [
    { familyId: PICKLE.id, verdict: 'real', argument: 'load.py:12 unpickles peer bytes', scope: ['src/ckpt.py'] },
    { familyId: ASSERT.id, verdict: 'false-positive', argument: 'assert is in a debug-only invariant check that never guards input' },
  ] }),
  'propose:': () => ({ fix: 'use safetensors', rationale: 'no code execution' }),
  'vote:': () => ({ approve: true, reasons: 'ok' }),
  'apply:': () => ({ summary: 'ok', files: ['src/ckpt.py'], evidence: [{ claim: 't', command: 'x', exitCode: 0, output: '' }] }),
  'suppress': () => ({ summary: 'ok', files: ['agents-architetura/tools/campaign/suppressions/scan-bandit.txt'] }),
  'rescan': () => ({ json: { applicable: true, ok: true, scanners: [{ id: 'semgrep', status: 'ran', families: [] }, { id: 'bandit', status: 'ran', families: [] }] }, exitCode: 0 }),
  'report': () => ({ summary: 'ok', files: [] }),
  'record:uncertain': () => ({ summary: 'written', files: [] }),
}
const run = extra => simulate(script, { args: {}, respond: byLabel({ ...base, ...extra }) })

test('real findings are fixed by vote, false positives get justified suppressions, unavailable scanners are loud', async () => {
  const { result, calls } = await run()
  assert.equal(result.ok, true)
  assert.deepEqual(result.unavailable, [{ id: 'codeql', reason: 'codeql not installed' }])
  assert.deepEqual([result.real, result.suppressed, result.applied], [1, 1, 1])
  assert.match(calls.find(c => c.opts.label === 'rescan').prompt, /scan\.mjs --only semgrep,bandit --json/)
  assert.ok(calls.find(c => c.opts.label === 'suppress').prompt.includes('# RULING: false-positive'))
  assertRoles(assert, calls, { 'battery': 'wan-verifier', 'triage:': 'wan-reviewer', 'propose:': 'wan-proposer', 'apply:': 'wan-coder', 'suppress': 'wan-researcher', 'report': 'wan-researcher' })
})

test('residual findings, broken scanners or unjudged families keep it red; no applicable scanner is n/a', async () => {
  const residual = await run({ 'rescan': () => ({ json: { scanners: [{ id: 'semgrep', status: 'ran', families: [PICKLE] }] }, exitCode: 1 }) })
  assert.equal(residual.result.ok, false)
  assert.equal(residual.result.residual.length, 1)
  const unjudged = await run({ 'triage:': () => ({ verdicts: [] }) })
  assert.deepEqual(unjudged.result.unjudged, [PICKLE.id, ASSERT.id])
  assert.equal(unjudged.result.ok, false)
  const broken = await run({ 'battery': () => ({ json: { scanners: [{ id: 'semgrep', status: 'broken', reason: 'exit 2' }] }, exitCode: 1 }) })
  assert.equal(broken.result.ok, false)
  const na = await run({ 'battery': () => ({ json: {}, exitCode: 5 }) })
  assert.deepEqual([na.result.ok, na.result.applicable], [true, false])
})

test('an uncertain family goes to a human as a needs-human finding and keeps the battery red', async () => {
  const { result, calls } = await run({ 'triage:': () => ({ verdicts: [
    { familyId: PICKLE.id, verdict: 'uncertain', argument: 'depends on whether peers are trusted, which the threat model leaves open' },
    { familyId: ASSERT.id, verdict: 'false-positive', argument: 'debug-only invariant check that never guards input' },
  ] }) })
  assert.deepEqual(result.needsHuman, [PICKLE.id])
  assert.equal(result.ok, false)
  assert.ok(calls.find(c => c.opts.label === 'record:uncertain').prompt.includes('record.mjs findings --b64'))
})
