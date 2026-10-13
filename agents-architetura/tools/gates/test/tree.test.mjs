import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFileSync } from 'node:child_process'
import { treeId, lockHashes, staleReason, toolchain } from '../tree.mjs'

const gitRepo = () => {
  const root = mkdtempSync(join(tmpdir(), 'tree-'))
  execFileSync('git', ['init', '-q'], { cwd: root })
  writeFileSync(join(root, '.gitignore'), 'ignored/\n')
  writeFileSync(join(root, 'a.py'), 'x = 1\n')
  execFileSync('git', ['add', '-A'], { cwd: root })
  execFileSync('git', ['-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'init'], { cwd: root })
  return root
}

test('tree id: stable, sensitive to tracked and untracked edits, blind to ignored files, real index untouched', () => {
  const root = gitRepo()
  const a = treeId(root)
  assert.match(a, /^[0-9a-f]{40}$/)
  assert.equal(treeId(root), a)
  execFileSync('mkdir', ['-p', join(root, 'ignored')])
  writeFileSync(join(root, 'ignored', 'big.bin'), 'zzz')
  assert.equal(treeId(root), a)
  writeFileSync(join(root, 'new.py'), 'y = 2\n')
  const b = treeId(root)
  assert.notEqual(b, a)
  writeFileSync(join(root, 'a.py'), 'x = 2\n')
  assert.notEqual(treeId(root), b)
  assert.equal(execFileSync('git', ['diff', '--cached', '--name-only'], { cwd: root, encoding: 'utf8' }), '')
})

test('tree id is null outside a git repository', () => {
  assert.equal(treeId(mkdtempSync(join(tmpdir(), 'nogit-'))), null)
})

test('lock hashes and stale-evidence reasons', () => {
  const root = mkdtempSync(join(tmpdir(), 'locks-'))
  writeFileSync(join(root, 'uv.lock'), 'v1')
  const h = lockHashes(root)
  assert.deepEqual(Object.keys(h), ['uv.lock'])
  assert.match(h['uv.lock'], /^[0-9a-f]{64}$/)
  assert.equal(staleReason('a'.repeat(40), 'a'.repeat(40)), null)
  assert.match(staleReason('a'.repeat(40), 'b'.repeat(40)), /stale evidence/)
  assert.match(staleReason(null, 'b'.repeat(40)), /no tree id/)
})

test('toolchain reports n/a for missing tools', () => {
  const t = toolchain({ node: ['node', ['--version']], ghost: ['definitely-not-a-tool-xyz', ['--version']] })
  assert.match(t.node, /^v\d+/)
  assert.equal(t.ghost, 'n/a')
})
