import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { execFileSync } from 'node:child_process'
import { checkDiscipline } from '../discipline.mjs'
import { checkInvariants } from '../invariants.mjs'
import { lintClaims, verifyClaim } from '../claims.mjs'

const tree = files => {
  const root = mkdtempSync(join(tmpdir(), 'gates-'))
  for (const [f, body] of Object.entries(files)) {
    mkdirSync(join(root, dirname(f)), { recursive: true })
    writeFileSync(join(root, f), body)
  }
  return root
}
const STRICT = '[tool.pytest.ini_options]\nxfail_strict = true\n'
const FINDINGS = JSON.stringify({ findings: [{ id: 'F-7', status: 'open' }, { id: 'F-2', status: 'closed' }] })

test('discipline: strict xfail with ledger reasons passes', () => {
  const r = checkDiscipline(tree({
    'pyproject.toml': STRICT,
    'agents-architetura/docs/findings.json': FINDINGS,
    'tests/test_a.py': '@pytest.mark.xfail(strict=True, reason="LEDGER:F-7 reorder race")\ndef test_a(): pass\n' +
      '@pytest.mark.xfail(strict=True, reason="awaiting implementation: TRANSPORT-I1")\ndef test_b(): pass\n',
    'tests/wire.rs': '#[test]\n#[ignore = "LEDGER:F-7 pinned"]\nfn t() {}\n',
  }))
  assert.equal(r.ok, true, JSON.stringify(r.violations))
  assert.equal(r.pins, 3)
})

test('discipline: missing reasons, unknown or closed ledger ids, strict=False and lax config all fail', () => {
  const r = checkDiscipline(tree({
    'pyproject.toml': '[tool.pytest.ini_options]\n',
    'agents-architetura/docs/findings.json': FINDINGS,
    'tests/test_a.py': [
      '@pytest.mark.xfail\ndef t1(): pass',
      '@pytest.mark.xfail(reason="flaky sometimes")\ndef t2(): pass',
      '@pytest.mark.xfail(reason="LEDGER:F-99 x")\ndef t3(): pass',
      '@pytest.mark.xfail(reason="LEDGER:F-2 x")\ndef t4(): pass',
      '@pytest.mark.xfail(strict=False, reason="LEDGER:F-7 x")\ndef t5(): pass',
      'def t6():\n    pytest.xfail("LEDGER:F-7 dynamic")',
    ].join('\n'),
    'tests/x.rs': '#[ignore]\nfn t() {}\n',
  }))
  const v = r.violations.join('\n')
  assert.match(v, /xfail_strict = true/)
  assert.match(v, /test_a\.py:1: expected-failure pin without a reason/)
  assert.match(v, /must start with "LEDGER:<id>"/)
  assert.match(v, /LEDGER:F-99 is not in/)
  assert.match(v, /LEDGER:F-2 is closed — the pin must be removed/)
  assert.match(v, /strict=False/)
  assert.match(v, /x\.rs:1: expected-failure pin without a reason/)
  assert.equal(r.violations.length, 7)
  assert.equal(checkDiscipline(tree({ 'README.md': '' })).applicable, false)
})

test('invariants: uncovered, deferred and unknown ids', () => {
  const root = tree({
    'docs/design/SPEC-transport.md': '| TRANSPORT-I1 | frames are length-prefixed |\n| TRANSPORT-I2 | retries are idempotent |\n| TRANSPORT-I3 | DEFERRED-TO: M2 zero-copy |\nrefers to SYNC-I1\n',
    'docs/design/SPEC-transport-history.md': 'TRANSPORT-I9 was removed\n',
    'docs/design/SPEC-sync.md': '- SYNC-I1: rounds are monotonic\n',
    'tests/unit/test_wire.py': '# covers: TRANSPORT-I1, SYNC-I1\n# covers: TRANSPORT-I7\n',
  })
  const r = checkInvariants(root)
  assert.equal(r.ok, false)
  assert.equal(r.defined, 4)
  assert.equal(r.covered, 2)
  assert.deepEqual(r.uncovered, ['TRANSPORT-I2 (SPEC-transport.md)'])
  assert.deepEqual(r.deferred, { 'TRANSPORT-I3': 'M2' })
  assert.deepEqual(r.unknown, ['TRANSPORT-I7 claimed by tests/unit/test_wire.py'])
  assert.equal(checkInvariants(tree({ 'docs/design/DESIGN.md': '' })).applicable, false)
})

test('claims lint: every test states its regression', () => {
  const r = lintClaims(tree({
    'tests/unit/test_ok.py': '# WOULD-FAIL-IF: decode() accepts a truncated frame\n',
    'tests/unit/test_none.py': 'def test(): pass\n',
    'tests/unit/test_empty.py': '# WOULD-FAIL-IF: x\n',
    'tests/unit/conftest.py': '',
  }))
  assert.deepEqual(r.missing, ['tests/unit/test_none.py'])
  assert.deepEqual(r.empty, ['tests/unit/test_empty.py'])
  assert.equal(r.tests, 3)
})

const gitTree = files => {
  const root = tree(files)
  execFileSync('git', ['init', '-q'], { cwd: root })
  return root
}
// Python reuses a cached .pyc when the source's mtime (whole seconds) and size match. CI writes bytecode; some
// shells set PYTHONDONTWRITEBYTECODE and would hide a same-size mutant or patch running stale code. Test as CI runs.
delete process.env.PYTHONDONTWRITEBYTECODE
delete process.env.PYTHONPYCACHEPREFIX
const PY = process.env.PYTHON ?? 'python3'
const SRC = 'def is_adult(age):\n    return age >= 18\n'
const PATCH = 'diff --git a/src/calc.py b/src/calc.py\n--- a/src/calc.py\n+++ b/src/calc.py\n@@ -1,2 +1,2 @@\n def is_adult(age):\n-    return age >= 18\n+    return age > 18\n'

test('claims verify: catches, vacuous, and not-a-test-failure — on a snapshot, never the live tree', () => {
  const strong = 'import sys; sys.path.insert(0, "src")\nfrom calc import is_adult\n# WOULD-FAIL-IF: the boundary 18 is excluded\nassert is_adult(18)\n'
  const weak = 'import sys; sys.path.insert(0, "src")\nfrom calc import is_adult\n# WOULD-FAIL-IF: the boundary 18 is excluded\nassert is_adult(30)\n'
  const root = gitTree({ 'src/calc.py': SRC, 'tests/test_strong.py': strong, 'tests/test_strong.py.would-fail.patch': PATCH, 'tests/test_weak.py': weak, 'tests/test_weak.py.would-fail.patch': PATCH })
  const cmd = `${PY} {test}`
  assert.equal(verifyClaim(root, 'tests/test_strong.py', { cmd }).verdict, 'catches')
  assert.equal(verifyClaim(root, 'tests/test_weak.py', { cmd }).verdict, 'vacuous')
  // A same-size regression (18 -> 19) applied right after the clean run must run the NEW code, not a stale .pyc.
  const sameSize = gitTree({ 'src/calc.py': SRC, 'tests/test_b.py': strong, 'tests/test_b.py.would-fail.patch': PATCH.replace('+    return age > 18', '+    return age >= 19') })
  assert.equal(verifyClaim(sameSize, 'tests/test_b.py', { cmd }).verdict, 'catches')
  const syntax = gitTree({ 'src/calc.py': SRC, 'tests/test_s.py': strong, 'tests/test_s.py.would-fail.patch': PATCH.replace('+    return age > 18', '+    return age >') })
  assert.equal(verifyClaim(syntax, 'tests/test_s.py', { cmd, failCodes: [2] }).verdict, 'not-a-test-failure')
  assert.equal(verifyClaim(root, 'tests/test_nopatch.py', { cmd }).verdict, 'no-test')
  assert.throws(() => verifyClaim(root, 'tests/test_strong.py', { cmd: 'x' }), /must contain \{test\}/)
  assert.equal(execFileSync('cat', [join(root, 'src/calc.py')], { encoding: 'utf8' }), SRC)
})
