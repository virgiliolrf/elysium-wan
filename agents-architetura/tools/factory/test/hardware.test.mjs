import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { readHardware, writeHardware, HARDWARE_FILE } from '../hardware.mjs'

const CLI = join(dirname(fileURLToPath(import.meta.url)), '..', 'hardware.mjs')
const root = body => {
  const r = mkdtempSync(join(tmpdir(), 'hw-'))
  mkdirSync(join(r, dirname(HARDWARE_FILE)), { recursive: true })
  if (body !== undefined) writeFileSync(join(r, HARDWARE_FILE), body)
  return r
}

test('anything but a literal "gpu": true reads as gpu=false, so a missing or damaged switch never starts spending', () => {
  for (const body of [undefined, '', '{', 'null', '[]', '{}', '{"gpu": false}', '{"gpu": "true"}', '{"gpu": 1}', '{"GPU": true}']) {
    const hw = readHardware(root(body))
    assert.equal(hw.gpu, false, `body ${JSON.stringify(body)}`)
    assert.ok(hw.reason, `body ${JSON.stringify(body)} gives a reason`)
  }
  assert.deepEqual(readHardware(root('{"gpu": true}')), { gpu: true })
})

test('writeHardware round-trips booleans and refuses anything else', () => {
  const r = root()
  assert.equal(writeHardware(r, true).gpu, true)
  assert.deepEqual(JSON.parse(readFileSync(join(r, HARDWARE_FILE), 'utf8')), { gpu: true })
  assert.equal(writeHardware(r, false).gpu, false)
  for (const bad of ['true', 1, null, undefined]) assert.throws(() => writeHardware(r, bad), /true or false/)
  assert.equal(readHardware(r).gpu, false)
})

test('the CLI refuses an unknown value with a usage error and reads the switch as JSON', () => {
  const bad = spawnSync(process.execPath, [CLI, 'set', 'gpu', 'maybe'], { encoding: 'utf8' })
  assert.equal(bad.status, 2)
  assert.match(bad.stderr, /usage/)
  const shown = spawnSync(process.execPath, [CLI, 'show', '--json'], { encoding: 'utf8' })
  assert.equal(shown.status, 0)
  assert.equal(typeof JSON.parse(shown.stdout).gpu, 'boolean')
})
