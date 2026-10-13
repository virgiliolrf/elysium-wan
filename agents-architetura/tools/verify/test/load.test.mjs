import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, existsSync, readdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { load } from '../load.mjs'

const CLI = join(dirname(fileURLToPath(import.meta.url)), '..', 'load.mjs')
const node = js => `${JSON.stringify(process.execPath)} -e ${JSON.stringify(js)}`

test('workers run concurrently with rotating seeds; failing logs kept, passing logs not', async () => {
  const out = mkdtempSync(join(tmpdir(), 'load-'))
  const r = await load(node("const s = Number(process.env.FACTORY_SEED); setTimeout(() => process.exit(s === 1001 || s === 1010 ? 1 : 0), 150)"), { workers: 3, runsPerWorker: 4, seedBase: 1000, out })
  assert.equal(r.total, 12)
  assert.equal(r.failed, 2)
  assert.deepEqual(r.failures.map(f => f.seed), [1001, 1010])
  assert.ok(r.peakConcurrency >= 2, `peak concurrency ${r.peakConcurrency}`)
  assert.equal(readdirSync(out).length, 2)
  assert.ok(r.failures.every(f => existsSync(f.log)))
})

test('pass marker: exit 0 without PASS is a failure when required', async () => {
  const r = await load(node("console.log('done')"), { workers: 1, runsPerWorker: 2, requirePassMarker: true })
  assert.equal(r.failed, 2)
  assert.equal(r.failures[0].outcome, 'no-pass-marker')
  const ok = await load(node("console.log('PASS')"), { workers: 1, runsPerWorker: 2, requirePassMarker: true })
  assert.equal(ok.clean, true)
})

test('CLI exit code is the failure count', () => {
  const out = mkdtempSync(join(tmpdir(), 'loadcli-'))
  const p = spawnSync(process.execPath, [CLI, '--workers', '2', '--runs-per-worker', '3', '--seed-base', '0', '--out', out, '--', node('process.exit(Number(process.env.FACTORY_SEED) % 2)')], { encoding: 'utf8' })
  assert.equal(p.status, 3)
  assert.match(p.stdout, /3\/6 passed/)
  assert.equal(spawnSync(process.execPath, [CLI, '--workers', '0', '--', 'true'], { encoding: 'utf8' }).status, 126)
})
