// Opt-in: runs the shipped rungs.json against a real Python (uv) + Rust (cargo) fixture project, so the
// ladder's commands are proven against the actual toolchains — not only against a fake runner.
// Needs uv, cargo and network for the first `uv lock`.  Run with: FACTORY_TOOLCHAIN_TEST=1 node --test agents-architetura/tools/gates/test/rungs.toolchain.test.mjs
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync } from 'node:child_process'
import { runLadder } from '../ladder.mjs'

const enabled = process.env.FACTORY_TOOLCHAIN_TEST === '1'
const config = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'rungs.json'), 'utf8'))

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'rungs-toolchain-'))
  const w = (p, s) => { mkdirSync(dirname(join(root, p)), { recursive: true }); writeFileSync(join(root, p), s) }
  w('pyproject.toml', `[project]
name = "fixture"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = []

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6", "mypy>=1.10"]

[tool.uv]
package = false

[tool.mypy]
files = ["src"]
strict = true

[tool.pytest.ini_options]
pythonpath = ["src"]
`)
  w('src/fixture/__init__.py', 'def add(a: int, b: int) -> int:\n    return a + b\n')
  w('tests/unit/test_add.py', 'from fixture import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n')
  w('Cargo.toml', '[package]\nname = "fixture"\nversion = "0.0.1"\nedition = "2021"\n\n[dependencies]\n')
  w('src/lib.rs', 'pub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn adds() {\n        assert_eq!(super::add(2, 3), 5);\n    }\n}\n')
  execFileSync('uv', ['lock', '-q'], { cwd: root })
  execFileSync('cargo', ['generate-lockfile', '-q'], { cwd: root })
  return { root, w }
}

test('real toolchains: build, syntax, fmt, lint, types, unit and static pass on a clean fixture; real defects turn the right rung red; absent suites are N/A', { skip: !enabled && 'set FACTORY_TOOLCHAIN_TEST=1', timeout: 900000 }, () => {
  const { root, w } = fixture()
  const steps = ['build', 'syntax', 'fmt', 'lint', 'types', 'unit', 'static']
  const greenRun = runLadder({ root, config, only: steps })
  assert.deepEqual(greenRun.rungs.map(r => [r.rung, r.status]), steps.map(s => [s, 'pass']), JSON.stringify(greenRun.rungs.filter(r => r.status !== 'pass'), null, 2))

  w('tests/unit/test_add.py', 'from fixture import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 6\n')
  const red = runLadder({ root, config, only: ['unit'] })
  assert.equal(red.allGreen, false)
  assert.match(red.rungs[0].detail, /assert 5 == 6/)

  w('src/fixture/__init__.py', 'def add(a, b):\n    return a + b\n')
  assert.equal(runLadder({ root, config, only: ['types'] }).rungs[0].status, 'fail')

  w('src/fixture/broken.py', 'def f(:\n')
  assert.equal(runLadder({ root, config, only: ['syntax'] }).rungs[0].status, 'fail')

  const na = runLadder({ root, config, only: ['oracle', 'fuzz', 'redteam', 'bench'] })
  assert.deepEqual(na.rungs.map(r => r.status), ['n/a', 'n/a', 'n/a', 'n/a'])
  assert.ok(na.rungs.every(r => r.detail.startsWith('not applicable:')))
})
