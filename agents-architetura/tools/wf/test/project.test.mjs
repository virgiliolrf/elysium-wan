import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadLib } from '../sim.mjs'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..')

test('RUNG_IDS in lib/project.js matches agents-architetura/tools/gates/rungs.json exactly (order included)', () => {
  const { RUNG_IDS } = loadLib(['project'], { agent: () => null })
  const config = JSON.parse(readFileSync(join(ROOT, 'agents-architetura', 'tools', 'gates', 'rungs.json'), 'utf8'))
  assert.deepEqual(RUNG_IDS, config.rungs.map(r => r.id))
})

test('common() carries the hard rules into every prompt', () => {
  const { common } = loadLib(['project'], { agent: () => null })
  const p = common('OWNED PATHS: x')
  for (const rule of ['Do NOT run git', 'WRITE only inside the owned paths', 'fabricated green is not', 'OWNED PATHS: x', 'current working directory']) assert.ok(p.includes(rule), rule)
})
