import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fuzzOnce, replay, barStatus, loadStats } from '../fuzz-run.mjs'

const NODE = JSON.stringify(process.execPath)
const noLock = (_, fn) => fn()
// A fake fuzzer: drops two crash inputs with the same bug and one with a different bug.
const FUZZER = `${NODE} -e "const fs=require('fs'),p=require('path'),d=process.argv[1];fs.writeFileSync(p.join(d,'crash-a'),'BAD:len');fs.writeFileSync(p.join(d,'crash-b'),'BAD:len');fs.writeFileSync(p.join(d,'crash-c'),'BAD:type');fs.writeFileSync(p.join(d,'crash-d'),'flaky')" {crashes}`
const REPRO = `${NODE} -e "const s=require('fs').readFileSync(process.argv[1],'utf8');if(s.startsWith('BAD:')){console.error('Traceback (most recent call last):\\n  File \\"/r/src/wire/decode.py\\", line 9, in decode_'+s.slice(4)+'\\nValueError: '+s.slice(4));process.exit(1)}" {input}`
const campaign = extra => ({ id: 'fuzz-wire', kind: 'fuzz', cmd: FUZZER, repro: REPRO, ...extra })

test('new crashes are replayed and deduped by signature; non-reproducing crashes are kept as flaky', () => {
  const root = mkdtempSync(join(tmpdir(), 'fuzz-'))
  const r = fuzzOnce(root, campaign(), { minutes: 0.01, lock: noLock })
  assert.equal(r.ok, false)
  assert.equal(r.newCrashFiles, 4)
  assert.equal(r.newUnique.length, 3)
  assert.deepEqual(r.newUnique.map(n => n.flaky).sort(), [false, false, true])
  const sigs = Object.keys(loadStats(root, 'fuzz-wire').signatures)
  assert.ok(sigs.some(s => s.startsWith('py-ValueError|decode_len@src/wire/decode.py')))
  const again = fuzzOnce(root, campaign({ cmd: 'true' }), { minutes: 0.01, lock: noLock })
  assert.equal(again.ok, true)
  assert.equal(again.uniqueTotal, 3)
})

test('a failing self-test stops the campaign before it fuzzes nothing', () => {
  const root = mkdtempSync(join(tmpdir(), 'fuzz-'))
  const r = fuzzOnce(root, campaign({ selfTest: 'echo "coverage of src/wire: 0 edges"; exit 1' }), { minutes: 1, lock: noLock })
  assert.equal(r.selfTestFailed, true)
  assert.match(r.detail, /fuzzing nothing/)
})

test('replay: regressions and stored crashes must no longer crash', () => {
  const root = mkdtempSync(join(tmpdir(), 'fuzz-'))
  mkdirSync(join(root, 'tests/fuzz/regressions/fuzz-wire'), { recursive: true })
  writeFileSync(join(root, 'tests/fuzz/regressions/fuzz-wire/len.bin'), 'ok-now')
  assert.equal(replay(root, campaign({ regressions: 'tests/fuzz/regressions/fuzz-wire' })).ok, true)
  mkdirSync(join(root, 'tests/fuzz/regressions/fuzz-wire/type-sig'), { recursive: true })
  writeFileSync(join(root, 'tests/fuzz/regressions/fuzz-wire/type-sig/README.md'), 'WOULD-FAIL-IF: BAD:type decodes')
  writeFileSync(join(root, 'tests/fuzz/regressions/fuzz-wire/type-sig/min.bin'), 'BAD:type')
  const r = replay(root, campaign({ regressions: 'tests/fuzz/regressions/fuzz-wire' }))
  assert.equal(r.ok, false)
  assert.equal(r.replayed, 2)
  assert.equal(r.stillCrashing.length, 1)
  assert.equal(replay(root, campaign(), undefined, 'tests/fuzz/regressions/fuzz-wire/len.bin').replayed, 1)
})

test('bar: CPU-hours and a quiet final hour', () => {
  assert.equal(barStatus({ cpuSeconds: 5 * 3600, runs: [{ seconds: 7200, lastNewUniqueSec: 1000 }] }).ok, true)
  const loud = barStatus({ cpuSeconds: 5 * 3600, runs: [{ seconds: 7200, lastNewUniqueSec: 7000 }] })
  assert.equal(loud.ok, false)
  assert.match(loud.why[0], /new unique crash appeared 3 min before the end/)
  assert.match(barStatus({ cpuSeconds: 3600, runs: [{ seconds: 3600, lastNewUniqueSec: null }] }).why[0], /1\.00 CPU-hours < 4/)
})
