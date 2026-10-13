import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { inventory, checkPaths } from '../inventory.mjs'

const repo = files => {
  const root = mkdtempSync(join(tmpdir(), 'inv-'))
  for (const [f, b] of Object.entries(files)) { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), b) }
  return root
}

test('inventory: dirs, languages, manifests, entrypoints, tests and the internal import graph', () => {
  const root = repo({
    'pipeline/__init__.py': '',
    'pipeline/stage.py': 'import torch\nfrom pipeline.comm import send\n',
    'pipeline/comm.py': 'import socket\n',
    'start_master.py': 'from pipeline import stage\nif __name__ == "__main__":\n    stage.run()\n',
    'setup_and_launch_worker.sh': '#!/bin/sh\n',
    'requirements.txt': 'torch\n',
    'tests/test_stage.py': 'from pipeline.stage import x\n',
    '.venv/lib/x.py': 'ignored',
    'agents-architetura/tools/gates/ladder.mjs': 'the factory is not the product',
    '.claude/agents/wan-coder.md': 'factory',
  })
  const inv = inventory(root)
  assert.deepEqual(inv.excluded, ['agents-architetura/', '.claude/'])
  assert.deepEqual(inv.byTopDir, { '(root)': 3, 'pipeline/': 3, 'tests/': 1 })
  assert.deepEqual(inv.manifests, ['requirements.txt'])
  assert.deepEqual(inv.entrypoints, ['setup_and_launch_worker.sh', 'start_master.py'])
  assert.deepEqual(inv.tests, ['tests/test_stage.py'])
  assert.ok(inv.imports.includes('pipeline/stage.py -> pipeline/comm.py'))
  assert.ok(inv.imports.includes('start_master.py -> pipeline/__init__.py'))
  assert.ok(inv.imports.includes('tests/test_stage.py -> pipeline/stage.py'))
  assert.ok(!inv.imports.some(e => e.includes('torch') || e.includes('socket')))
})

test('check-paths: missing and unsafe paths are reported', () => {
  const root = repo({ 'pipeline/stage.py': '' })
  assert.deepEqual(checkPaths(root, ['pipeline/', 'pipeline/stage.py']), { ok: true, missing: [], invalid: [] })
  assert.deepEqual(checkPaths(root, ['scheduler/', '../etc']), { ok: false, missing: ['scheduler/'], invalid: ['../etc'] })
})
