import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { amplify } from '../amplify.mjs'
import { classify, normalizeOutput } from '../exec.mjs'

const CLI = join(dirname(fileURLToPath(import.meta.url)), '..', 'amplify.mjs')
const node = js => `${JSON.stringify(process.execPath)} -e ${JSON.stringify(js)}`
const seed = 'const s = Number(process.env.FACTORY_SEED);'

test('classification: signal and 128+n are crashes, timeouts are hangs', () => {
  assert.equal(classify({ status: 0 }), 'pass')
  assert.equal(classify({ status: 1 }), 'fail')
  assert.equal(classify({ status: null, signal: 'SIGSEGV' }), 'crash')
  assert.equal(classify({ status: 139 }), 'crash')
  assert.equal(classify({ status: null, timedOut: true }), 'hang')
})

test('normalization hides durations and addresses but not values', () => {
  assert.equal(normalizeOutput('1 passed in 0.12s at 0xdeadBEEF'), '1 passed in <dur> at <addr>')
  assert.equal(normalizeOutput('2 passed in 93.45s (0:01:33)'), normalizeOutput('2 passed in 12.30s'))
  assert.notEqual(normalizeOutput('loss=0.5'), normalizeOutput('loss=0.6'))
  assert.equal(normalizeOutput('a\nnoise 123\nb', ['^noise']), 'a\nb')
})

test('a failure on one seed is found, named and replayable', () => {
  const out = mkdtempSync(join(tmpdir(), 'amp-'))
  const r = amplify(node(`${seed} if (s === 1007) process.exit(3); console.log('ok')`), { runs: 10, seedBase: 1000, out, timeoutSec: 20 })
  assert.equal(r.clean, false)
  assert.deepEqual(r.findings.map(f => [f.seed, f.outcome]), [[1007, 'FAIL']])
  assert.match(r.findings[0].replay, /^FACTORY_SEED=1007 FACTORY_PERTURB_PERIOD=64 FACTORY_PERTURB_MAX_US=100 /)
  assert.ok(existsSync(r.findings[0].log))
  assert.equal(r.counts.pass, 9)
})

test('hang, crash and divergence are classified', () => {
  const r = amplify(node(`${seed} if (s === 3) setTimeout(() => {}, 60000); else if (s === 4) process.kill(process.pid, 'SIGSEGV'); else console.log(s === 5 ? 'value=2' : 'value=1')`), { runs: 6, seedBase: 0, timeoutSec: 2 })
  assert.deepEqual(r.findings.map(f => [f.seed, f.outcome]), [[3, 'HANG'], [4, 'CRASH'], [5, 'DIVERGENCE']])
  assert.equal(r.referenceSeed, 1)
})

test('periods multiply runs; output that depends on the perturbation period is a divergence', () => {
  const r = amplify(node('console.log(process.env.FACTORY_PERTURB_PERIOD)'), { runs: 2, seedBase: 500, periods: [16, 256] })
  assert.equal(r.total, 4)
  assert.deepEqual(r.findings.map(f => [f.seed, f.period, f.outcome]), [[503, 256, 'DIVERGENCE'], [504, 256, 'DIVERGENCE']])
})

test('CLI: exit 0 when clean, 1 with findings, 2 on usage; summary written', () => {
  const out = mkdtempSync(join(tmpdir(), 'ampcli-'))
  const clean = spawnSync(process.execPath, [CLI, '--runs', '3', '--seed-base', '10', '--out', out, '--json', '--', node("console.log('same')")], { encoding: 'utf8' })
  assert.equal(clean.status, 0)
  assert.equal(JSON.parse(clean.stdout).clean, true)
  assert.equal(JSON.parse(readFileSync(join(out, 'summary.json'), 'utf8')).total, 3)
  const bad = spawnSync(process.execPath, [CLI, '--runs', '3', '--seed-base', '10', '--out', out, '--', node(`${seed} process.exit(s === 12 ? 1 : 0)`)], { encoding: 'utf8' })
  assert.equal(bad.status, 1)
  assert.match(bad.stdout, /FAIL seed=12/)
  assert.equal(spawnSync(process.execPath, [CLI, '--runs', '0', '--', 'true'], { encoding: 'utf8' }).status, 2)
  assert.equal(spawnSync(process.execPath, [CLI, '--bogus', '--', 'true'], { encoding: 'utf8' }).status, 2)
})

test('a misconfigured campaign stops before amplifying: a clean run that hangs, fails or nearly fills --timeout', () => {
  // the shape of M2's amplify-chaos: an 88 s suite under a 60 s timeout made every run a false HANG
  const slow = node('setTimeout(() => console.log("ok"), 1500)')
  assert.throws(() => amplify(slow, { runs: 50, seedBase: 0, timeoutSec: 1 }), /clean run \(perturbation off\) exceeded --timeout 1s/)
  assert.throws(() => amplify(slow, { runs: 50, seedBase: 0, timeoutSec: 2 }), /more than half of --timeout 2s/)
  assert.throws(() => amplify(node('process.exit(4)'), { runs: 50, seedBase: 0, timeoutSec: 20 }), /clean run \(perturbation off\) does not pass \(fail, exit 4\)/)
  let calls = 0
  const counting = (cmd, o) => { calls++; return { status: null, signal: null, timedOut: true, stdout: '', stderr: '', ms: 1000, outcome: 'hang' } }
  assert.throws(() => amplify('x', { runs: 200, periods: [16, 64, 256], seedBase: 0, timeoutSec: 1, run: counting }))
  assert.equal(calls, 1, 'one calibration run, not 600 false hangs')
  const ok = amplify(node('console.log(process.env.FACTORY_PERTURB_MAX_US)'), { runs: 1, seedBase: 0, timeoutSec: 20 })
  assert.equal(typeof ok.cleanRunMs, 'number')
})

test('CLI: a misconfigured campaign exits 2 (usage), not 1 (findings)', () => {
  const r = spawnSync(process.execPath, [CLI, '--runs', '5', '--seed-base', '1', '--timeout', '1', '--', node('setTimeout(() => {}, 1500)')], { encoding: 'utf8' })
  assert.equal(r.status, 2, r.stderr)
  assert.match(r.stderr, /exceeded --timeout 1s/)
})
