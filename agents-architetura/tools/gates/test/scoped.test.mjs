import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { execFileSync } from 'node:child_process'
import { planScope, runScoped, resolveScopeBase } from '../scoped.mjs'

const git = (root, ...a) => execFileSync('git', a, { cwd: root, encoding: 'utf8' })
const write = (root, files) => {
  for (const [f, body] of Object.entries(files)) { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), body) }
}
const commit = (root, msg) => {
  git(root, 'add', '-A')
  git(root, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', msg)
}
const NO_ENV = {}

// main holds another owner's file; the branch adds its own.
function repo() {
  const root = mkdtempSync(join(tmpdir(), 'scoped-'))
  git(root, '-c', 'init.defaultBranch=main', 'init', '-q')
  write(root, { 'pyproject.toml': '[project]\nname = "x"\n', 'theirs/legacy.py': 'x=1\n', 'README.md': '# r\n' })
  commit(root, 'main')
  git(root, 'checkout', '-qb', 'feature')
  return root
}

test('a branch is checked on the files it changed, never on files main already holds', () => {
  const root = repo()
  write(root, { 'src/mine.py': 'y = 2\n', 'docs/notes.md': 'n\n' })
  commit(root, 'feature work')
  write(root, { 'src/untracked.py': 'z = 3\n' })
  const plan = planScope(root, 'ruff-format', NO_ENV)
  assert.equal(plan.mode, 'diff')
  assert.deepEqual(plan.files, ['src/mine.py', 'src/untracked.py'])
})

test('touching the formatter configuration checks the whole tree', () => {
  const root = repo()
  write(root, { 'pyproject.toml': '[project]\nname = "x"\n[tool.ruff]\nline-length = 60\n' })
  commit(root, 'tighten rules')
  const plan = planScope(root, 'ruff-check', NO_ENV)
  assert.equal(plan.mode, 'all')
  assert.match(plan.why, /pyproject\.toml/)
})

test('a nested tool config or a non-ASCII path is not missed', () => {
  const root = repo()
  write(root, { 'src/pkg/ruff.toml': 'line-length = 60\n' })
  commit(root, 'nested rules')
  const nested = planScope(root, 'ruff-format', NO_ENV)
  assert.equal(nested.mode, 'all')
  assert.match(nested.why, /src\/pkg\/ruff\.toml/)

  const other = repo()
  write(other, { 'src/m\u00f3dulo.py': 'x = 1\n' })
  commit(other, 'unicode')
  assert.deepEqual(planScope(other, 'ruff-check', NO_ENV).files, ['src/m\u00f3dulo.py'])
})

test('on main itself, outside git, or without a reachable base, the fallback is the whole tree', () => {
  const root = repo()
  git(root, 'checkout', '-q', 'main')
  assert.equal(planScope(root, 'ruff-format', NO_ENV).mode, 'all')
  const plain = mkdtempSync(join(tmpdir(), 'scoped-plain-'))
  assert.equal(planScope(plain, 'ruff-format', NO_ENV).mode, 'all')
  git(root, 'checkout', '-q', 'feature')
  const missing = planScope(root, 'ruff-format', { FACTORY_BASE: '0123456789abcdef0123456789abcdef01234567' })
  assert.equal(missing.mode, 'all')
  assert.match(missing.why, /not in this clone/)
})

test('FACTORY_BASE pins the base, and a malformed one is refused', () => {
  const root = repo()
  const base = git(root, 'rev-parse', 'HEAD').trim()
  write(root, { 'a.py': 'a = 1\n' })
  commit(root, 'one')
  write(root, { 'b.py': 'b = 1\n' })
  commit(root, 'two')
  assert.equal(resolveScopeBase(root, { FACTORY_BASE: base }).base, base)
  assert.deepEqual(planScope(root, 'ruff-format', { FACTORY_BASE: 'HEAD~1' }).files, ['b.py'])
  assert.throws(() => planScope(root, 'ruff-format', { FACTORY_BASE: 'main; rm -rf /' }), /bad FACTORY_BASE/)
})

test('runScoped hands only the changed files to the tool, keeps configured excludes, and is N/A with nothing to check', () => {
  const root = repo()
  write(root, { 'src/mine.py': 'y = 2\n' })
  commit(root, 'feature work')
  const calls = []
  const status = runScoped(root, 'ruff-format', { env: NO_ENV, run: (cmd, args) => { calls.push([cmd, ...args]); return 0 } })
  assert.equal(status, 0)
  assert.deepEqual(calls, [['uv', 'run', 'ruff', 'format', '--check', '--force-exclude', '--', 'src/mine.py']])

  const docsOnly = repo()
  write(docsOnly, { 'docs/notes.md': 'n\n' })
  commit(docsOnly, 'docs')
  const none = []
  assert.equal(runScoped(docsOnly, 'ruff-check', { env: NO_ENV, run: (...c) => { none.push(c); return 0 } }), 5)
  assert.deepEqual(none, [])
})

test('a failing tool fails the gate; the whole-tree command runs on main', () => {
  const root = repo()
  write(root, { 'src/bad.py': 'x=1\n' })
  commit(root, 'feature work')
  assert.equal(runScoped(root, 'ruff-check', { env: NO_ENV, run: () => 1 }), 1)
  git(root, 'checkout', '-q', 'main')
  const calls = []
  runScoped(root, 'ruff-check', { env: NO_ENV, run: (cmd, args) => { calls.push([cmd, ...args]); return 0 } })
  assert.deepEqual(calls, [['uv', 'run', 'ruff', 'check', '.']])
})
