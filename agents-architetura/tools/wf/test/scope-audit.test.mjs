import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, rmSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync, spawnSync } from 'node:child_process'
import { snapshot, check } from '../scope-audit.mjs'

const CLI = join(dirname(fileURLToPath(import.meta.url)), '..', 'scope-audit.mjs')

function repo() {
  const root = mkdtempSync(join(tmpdir(), 'audit-'))
  const g = (...a) => execFileSync('git', a, { cwd: root, encoding: 'utf8' })
  g('init', '-q', '-b', 'main')
  g('config', 'user.email', 't@example.com')
  g('config', 'user.name', 't')
  mkdirSync(join(root, 'src', 'a'), { recursive: true })
  writeFileSync(join(root, 'src', 'a', 'x.py'), 'x = 1\n')
  writeFileSync(join(root, 'README.md'), 'r\n')
  writeFileSync(join(root, '.gitignore'), '.factory/\n')
  g('add', '-A')
  g('commit', '-q', '-m', 'init')
  return { root, g }
}

test('detects edits, new files and deletions; flags paths outside the scopes', () => {
  const { root } = repo()
  const t0 = snapshot(root)
  writeFileSync(join(root, 'src', 'a', 'x.py'), 'x = 2\n')
  writeFileSync(join(root, 'src', 'a', 'new.py'), 'n\n')
  rmSync(join(root, 'README.md'))
  mkdirSync(join(root, 'src', 'b'))
  writeFileSync(join(root, 'src', 'b', 'intruder.py'), 'i\n')
  const r = check(root, t0, ['src/a/'])
  assert.deepEqual(r.changed, ['README.md', 'src/a/new.py', 'src/a/x.py', 'src/b/intruder.py'])
  assert.deepEqual(r.violations, ['README.md', 'src/b/intruder.py'])
})

// Racy git: an edit that keeps the size and lands in the same timestamp as the index write looks clean by stat; git
// re-checks such "racily clean" entries only because their mtime is not older than the index file's. A copied index
// with a fresh mtime hides them, so the snapshot must keep the real index's mtime. Deterministic reproduction: the
// edited file and the real index carry the entry's original mtime (ctime ignored, as on filesystems without it).
test('a same-size edit inside the racy window is still detected (the copied index keeps its mtime)', () => {
  const { root, g } = repo()
  g('config', 'core.trustctime', 'false')
  const file = join(root, 'src', 'a', 'x.py')
  const index = join(root, '.git', 'index')
  // An index whose entries carry the files' true stat (not smudged): refreshed while the index looks newer than them.
  // The entry is dated in the past, so the snapshot runs in a later second than it (git's racy check is per second).
  execFileSync('touch', ['-t', '202001010000', file])
  execFileSync('touch', ['-t', '203001010000', index])
  g('update-index', '-q', '--refresh')
  const stamp = join(root, '.git', 'stamp')
  execFileSync('touch', ['-r', file, stamp]) // full-precision copy of the entry's timestamps
  writeFileSync(file, 'x = 9\n') // same size
  execFileSync('touch', ['-r', stamp, file])
  execFileSync('touch', ['-r', stamp, index]) // the index was written in the same tick as the edit: racy
  const truth = mkdtempSync(join(tmpdir(), 'audit-'))
  const env = { ...process.env, GIT_INDEX_FILE: join(truth, 'index') }
  execFileSync('git', ['add', '-A'], { cwd: root, env })
  const actual = execFileSync('git', ['write-tree'], { cwd: root, env, encoding: 'utf8' }).trim()
  assert.notEqual(actual, g('rev-parse', 'HEAD^{tree}').trim())
  assert.equal(snapshot(root), actual)
})

test('never touches the real index, HEAD or working tree; ignores .gitignored paths', () => {
  const { root, g } = repo()
  writeFileSync(join(root, 'staged.txt'), 's\n')
  g('add', 'staged.txt')
  const before = { status: g('status', '--porcelain'), head: g('rev-parse', 'HEAD') }
  const t0 = snapshot(root)
  mkdirSync(join(root, '.factory'))
  writeFileSync(join(root, '.factory', 'scratch'), 'z\n')
  const r = check(root, t0, ['src/'])
  assert.deepEqual(r.changed, [])
  assert.equal(g('rev-parse', 'HEAD'), before.head)
  assert.equal(g('status', '--porcelain'), before.status)
  assert.equal(readFileSync(join(root, 'staged.txt'), 'utf8'), 's\n')
})

test('rejects malformed inputs', () => {
  const { root } = repo()
  assert.throws(() => check(root, 'HEAD; rm -rf /', ['src/']), /not a tree id/)
  const t0 = snapshot(root)
  assert.throws(() => check(root, t0, ['../etc']), /repo-relative/)
  assert.throws(() => check(root, t0, '/abs'), /repo-relative/)
})

test('CLI: snapshot, check exit codes and JSON', () => {
  const { root } = repo()
  const snap = spawnSync(process.execPath, [CLI, 'snapshot', '--root', root], { encoding: 'utf8' })
  assert.equal(snap.status, 0)
  const { tree } = JSON.parse(snap.stdout)
  writeFileSync(join(root, 'README.md'), 'changed\n')
  const bad = spawnSync(process.execPath, [CLI, 'check', '--root', root, '--since', tree, '--scopes', '["src/"]'], { encoding: 'utf8' })
  assert.equal(bad.status, 1)
  assert.deepEqual(JSON.parse(bad.stdout).violations, ['README.md'])
  const ok = spawnSync(process.execPath, [CLI, 'check', '--root', root, '--since', tree, '--scopes', '["README.md"]'], { encoding: 'utf8' })
  assert.equal(ok.status, 0)
  const usage = spawnSync(process.execPath, [CLI, 'nope', '--root', root], { encoding: 'utf8' })
  assert.equal(usage.status, 2)
})
