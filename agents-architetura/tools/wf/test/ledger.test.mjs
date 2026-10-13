import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { write, owner } from '../ledger.mjs'
import { loadLib } from '../sim.mjs'

const CLI = join(dirname(fileURLToPath(import.meta.url)), '..', 'ledger.mjs')
const { base64, ledgerCommand } = loadLib(['ledger'])
const payload = {
  summary: { waves: 1, counts: { done: 1 }, tasks: [{ id: 'add', status: 'done' }] },
  owners: { add: ['src/mathx/', 'tests/unit/mathx/'] },
  journals: { add: '.factory/runs/M0/tasks/add.md' },
  transitions: [{ wave: 1, id: 'add', from: 'pending', to: 'running' }],
}

test('pure-JS base64 matches Buffer for ASCII, accents, emoji and empty', () => {
  for (const s of ['', 'a', 'ab', 'abc', 'ação → ∞ 🚀', JSON.stringify(payload)]) assert.equal(base64(s), Buffer.from(s, 'utf8').toString('base64'))
})

test('write persists state and transitions; owner finds the task and journal for a path', () => {
  const root = mkdtempSync(join(tmpdir(), 'ledger-'))
  assert.deepEqual(write(root, '.factory/runs/M0', payload), ['.factory/runs/M0/state.json', '.factory/runs/M0/transitions.jsonl'])
  const state = JSON.parse(readFileSync(join(root, '.factory/runs/M0/state.json'), 'utf8'))
  assert.equal(state.summary.tasks[0].status, 'done')
  assert.deepEqual(owner(root, 'src/mathx/__init__.py'), { task: 'add', journal: '.factory/runs/M0/tasks/add.md', run: 'M0' })
  assert.equal(owner(root, 'src/other.py'), null)
})

test('write refuses to escape .factory/', () => {
  const root = mkdtempSync(join(tmpdir(), 'ledger-'))
  for (const d of ['src', '.factory', '.factory/../src', '/tmp/x', '']) assert.throws(() => write(root, d, payload), /inside .factory/)
})

test('CLI round trip through the command the workflow generates', () => {
  const root = mkdtempSync(join(tmpdir(), 'ledger-'))
  const cmd = ledgerCommand('.factory/runs/M1', payload).split(' ')
  const r = spawnSync(process.execPath, [CLI, ...cmd.slice(2), '--root', root], { encoding: 'utf8' })
  assert.equal(r.status, 0, r.stderr)
  assert.deepEqual(JSON.parse(r.stdout).written, ['.factory/runs/M1/state.json', '.factory/runs/M1/transitions.jsonl'])
  const o = spawnSync(process.execPath, [CLI, 'owner', '--path', 'tests/unit/mathx/test_add.py', '--root', root], { encoding: 'utf8' })
  assert.equal(JSON.parse(o.stdout).task, 'add')
})
