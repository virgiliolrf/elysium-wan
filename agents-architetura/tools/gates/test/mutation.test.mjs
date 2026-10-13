import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { execFileSync } from 'node:child_process'
import { changedLines, runMutation } from '../mutation.mjs'

// Python reuses a cached .pyc when the source's mtime (whole seconds) and size match. CI writes bytecode; some
// shells set PYTHONDONTWRITEBYTECODE and would hide a same-size mutant or patch running stale code. Test as CI runs.
delete process.env.PYTHONDONTWRITEBYTECODE
delete process.env.PYTHONPYCACHEPREFIX
const PY = process.env.PYTHON ?? 'python3'
const git = (root, ...a) => execFileSync('git', a, { cwd: root, encoding: 'utf8' })
const write = (root, files) => {
  for (const [f, body] of Object.entries(files)) { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), body) }
}
function project(checks) {
  const root = mkdtempSync(join(tmpdir(), 'mut-'))
  write(root, { 'src/calc.py': 'def double(x):\n    return x * 2\n', 'tests/check.py': 'import sys; sys.path.insert(0, "src")\nimport calc\nassert calc.double(3) == 6\n' })
  git(root, 'init', '-q')
  git(root, 'add', '-A')
  git(root, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'base')
  write(root, {
    'src/calc.py': 'def double(x):\n    return x * 2\n\ndef is_adult(age):\n    return age >= 18\n',
    'tests/check.py': `import sys; sys.path.insert(0, "src")\nimport calc\nassert calc.double(3) == 6\n${checks}\n`,
  })
  return root
}
const cfg = { python: { paths: ['src/'], cmd: `${PY} tests/check.py`, minTimeoutSec: 20 } }

test('changed lines: only lines added since the revision, untracked files whole', () => {
  const root = project('')
  write(root, { 'src/new_mod.py': 'a = 1\nb = 2\n' })
  const c = changedLines(root, 'HEAD', f => f.startsWith('src/'))
  assert.deepEqual([...c['src/calc.py']].sort(), [3, 4, 5])
  assert.deepEqual([...c['src/new_mod.py']].sort(), [1, 2, 3])
  assert.throws(() => changedLines(root, 'HEAD; rm -rf /', () => true), /bad revision/)
})

test('a weak test lets boundary mutants survive; a boundary test kills them all; the live tree is untouched', () => {
  const weak = runMutation(project('assert calc.is_adult(30)\nassert not calc.is_adult(5)'), 'HEAD', cfg)
  assert.equal(weak.ok, false)
  assert.deepEqual(weak.survivors.map(s => s.op).sort(), ['18->19', 'GtE->Gt'])
  const strongRoot = project('assert calc.is_adult(18)\nassert not calc.is_adult(17)')
  const strong = runMutation(strongRoot, 'HEAD', cfg)
  assert.equal(strong.ok, true, JSON.stringify(strong.survivors))
  assert.ok(strong.engines.python.killed >= 4)
  assert.match(readFileSync(join(strongRoot, 'src/calc.py'), 'utf8'), /age >= 18/)
})

test('justified equivalent mutants are excluded; unjustified ones are violations', () => {
  const root = project('assert calc.is_adult(30)\nassert not calc.is_adult(5)')
  write(root, { 'agents-architetura/docs/mutants-equivalent.json': JSON.stringify({ equivalent: [
    { file: 'src/calc.py', line: 5, op: 'GtE->Gt', justification: 'ages are recorded in whole years from 0..150 and 18 is never sampled' },
    { file: 'src/calc.py', line: 5, op: '18->19', justification: 'nope' },
  ] }) })
  const r = runMutation(root, 'HEAD', cfg)
  assert.deepEqual(r.survivors.map(s => s.op), ['18->19'])
  assert.match(r.violations[0], /needs a justification/)
})

test('a red baseline is reported, not mistaken for killed mutants; no changes is n/a', () => {
  const root = project('assert False')
  const r = runMutation(root, 'HEAD', cfg)
  assert.equal(r.ok, false)
  assert.match(r.broken[0], /fails before any mutation/)
  const clean = mkdtempSync(join(tmpdir(), 'mut0-'))
  write(clean, { 'src/a.py': 'x = 1\n' })
  git(clean, 'init', '-q'); git(clean, 'add', '-A'); git(clean, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'b')
  assert.equal(runMutation(clean, 'HEAD', cfg).applicable, false)
})
