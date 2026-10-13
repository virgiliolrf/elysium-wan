#!/usr/bin/env node
// Test claims (the PR's WOULD-FAIL-IF discipline, made mechanical).
//
// Lint (the ladder rung): every test file under tests/ states the regression it exists to catch:
//     # WOULD-FAIL-IF: <the precise regression this test catches>
//
// Verify (wan-corpus, closeout): a test may ship the regression itself as <test>.would-fail.patch. On a snapshot
// of the working tree (never the live tree) the test must PASS clean and FAIL with the patch applied — with a
// test-failure exit code, not a crash or a syntax error. A test that still passes with its own regression applied
// is vacuous.
//
//   node agents-architetura/tools/gates/claims.mjs [--root DIR] [--json]
//   node agents-architetura/tools/gates/claims.mjs --verify <test> --cmd "<command with {test}>" [--fail-codes 1,101] [--timeout S]
// Exit: 0 ok · 1 violation / vacuous · 2 usage · 5 not applicable (no tests).
import { existsSync, rmSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { isPyTest, isRustTest, repoFiles, read } from './scan.mjs'
import { materialize } from './snapshot-dir.mjs'
import { runSync, shellQuote } from '../verify/exec.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
export const CLAIM = 'WOULD-FAIL-IF:'
export const patchFor = test => `${test}.would-fail.patch`

export function lintClaims(root) {
  const tests = repoFiles(root).filter(p => p.startsWith('tests/') && (isPyTest(p) || isRustTest(p)))
  if (!tests.length) return { applicable: false, reason: 'no test files under tests/' }
  const missing = []
  const empty = []
  for (const p of tests) {
    const text = read(root, p)
    const m = text.match(/WOULD-FAIL-IF:[ \t]*(.*)/)
    if (!m) missing.push(p)
    else if (m[1].trim().length < 10) empty.push(p)
  }
  const patches = tests.filter(p => existsSync(join(root, patchFor(p)))).length
  return { applicable: true, ok: !missing.length && !empty.length, tests: tests.length, withPatch: patches, missing, empty }
}

export function verifyClaim(root, test, { cmd, failCodes = [1], timeoutSec = 900, run = runSync } = {}) {
  if (!cmd?.includes('{test}')) throw new Error('--cmd must contain {test}')
  const patch = join(root, patchFor(test))
  if (!existsSync(join(root, test))) return { test, verdict: 'no-test' }
  if (!existsSync(patch)) return { test, verdict: 'no-patch' }
  const { dir, tree, env } = materialize(root)
  try {
    const command = cmd.replaceAll('{test}', shellQuote(test))
    const clean = run(command, { cwd: dir, env, timeoutSec })
    if (clean.outcome !== 'pass') return { test, tree, verdict: 'fails-clean', detail: `${clean.stdout}${clean.stderr}`.slice(-2000) }
    const applied = spawnSync('git', ['apply', '--whitespace=nowarn', patch], { cwd: dir, encoding: 'utf8' })
    if (applied.status !== 0) return { test, tree, verdict: 'patch-does-not-apply', detail: applied.stderr.slice(-1000) }
    const broken = run(command, { cwd: dir, env, timeoutSec })
    if (broken.outcome === 'pass') return { test, tree, verdict: 'vacuous', detail: 'the test still passes with its own regression applied' }
    if (broken.outcome !== 'fail' || !failCodes.includes(broken.status)) {
      return { test, tree, verdict: 'not-a-test-failure', detail: `patched run ended with ${broken.outcome} (exit ${broken.status}); expected a test failure exit (${failCodes.join('/')}) — the patch breaks the build, not the behavior` }
    }
    return { test, tree, verdict: 'catches' }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const opt = (name, d) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : d }
  const root = resolve(opt('--root', ROOT))
  try {
    if (argv.includes('--verify')) {
      const r = verifyClaim(root, opt('--verify'), {
        cmd: opt('--cmd'),
        failCodes: opt('--fail-codes', '1').split(',').map(Number),
        timeoutSec: Number(opt('--timeout', '900')),
      })
      console.log(JSON.stringify(r))
      process.exit(r.verdict === 'catches' ? 0 : 1)
    }
    const r = lintClaims(root)
    if (argv.includes('--json')) console.log(JSON.stringify(r))
    else if (!r.applicable) console.log(`claims: n/a (${r.reason})`)
    else {
      for (const p of r.missing) console.log(`MISSING ${p}: no "${CLAIM} <regression>" line`)
      for (const p of r.empty) console.log(`EMPTY   ${p}: the ${CLAIM} line does not name a regression`)
      console.log(`claims: ${r.tests - r.missing.length - r.empty.length}/${r.tests} tests state their regression; ${r.withPatch} ship a would-fail patch`)
    }
    process.exit(!r.applicable ? 5 : r.ok ? 0 : 1)
  } catch (e) {
    console.error(`claims: ${e.message}`)
    process.exit(2)
  }
}
