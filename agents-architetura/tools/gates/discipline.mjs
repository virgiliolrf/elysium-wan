#!/usr/bin/env node
// Expected-failure discipline (the PR's threadsExpectFail rule): a test may be pinned as expected-to-fail only
// strictly and only against a recorded reason, so an unexpected pass (XPASS) fails the suite — the fix landed and
// the pin must go, or the failure went latent and must be investigated, never celebrated.
//   - pyproject.toml sets xfail_strict = true (when Python tests exist); no xfail passes strict=False
//   - every xfail reason starts with "LEDGER:<id>" (an OPEN finding in agents-architetura/docs/findings.json) or is an
//     invariant test still "awaiting implementation: <ID>"
//   - every Rust #[ignore] carries the same kind of reason
//   - a pin on a CLOSED finding is a failure (remove the pin)
//
//   node agents-architetura/tools/gates/discipline.mjs [--root DIR] [--json]
// Exit: 0 ok · 1 violations · 5 not applicable (no tests).
import { existsSync, readFileSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { AWAITING, isPyTest, isRustTest, repoFiles, read, lineOf, callText, loadFindings } from './scan.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const LEDGER_RE = /^LEDGER:([A-Za-z0-9_-]+)/
const XFAIL_RE = /(?:pytest\.mark\.xfail|pytest\.xfail|\bmark\.xfail)\b(\s*\()?/g
const STRING_RE = /(?:^|[(,\s])(?:reason\s*=\s*)?[rRuUbBfF]{0,2}("""|'''|"|')([\s\S]*?)\1/

function checkReason(reason, where, findings, out) {
  if (reason === null) return out.push(`${where}: expected-failure pin without a reason — use "LEDGER:<id> ..." or "${AWAITING}: <ID>"`)
  if (reason.includes(AWAITING)) return
  const m = reason.match(LEDGER_RE)
  if (!m) return out.push(`${where}: reason ${JSON.stringify(reason.slice(0, 80))} must start with "LEDGER:<id>" or be "${AWAITING}: <ID>"`)
  const f = findings?.get(m[1])
  if (!findings) out.push(`${where}: LEDGER:${m[1]} but agents-architetura/docs/findings.json does not exist`)
  else if (!f) out.push(`${where}: LEDGER:${m[1]} is not in agents-architetura/docs/findings.json`)
  else if (f.status !== 'open') out.push(`${where}: LEDGER:${m[1]} is ${f.status} — the pin must be removed`)
}

export function checkDiscipline(root) {
  const files = repoFiles(root)
  const py = files.filter(isPyTest)
  const rs = files.filter(isRustTest)
  if (!py.length && !rs.length) return { applicable: false, reason: 'no test files' }
  const findings = loadFindings(root)
  const violations = []
  let pins = 0
  if (py.length) {
    const pp = join(root, 'pyproject.toml')
    if (!existsSync(pp) || !/^\s*xfail_strict\s*=\s*true\b/m.test(readFileSync(pp, 'utf8'))) violations.push('pyproject.toml: [tool.pytest.ini_options] must set xfail_strict = true (an unexpected pass must fail the suite)')
  }
  for (const p of py) {
    const text = read(root, p)
    for (const m of text.matchAll(XFAIL_RE)) {
      pins++
      const where = `${p}:${lineOf(text, m.index)}`
      if (!m[1]) { checkReason(null, where, findings, violations); continue }
      const call = callText(text, m.index + m[0].length - 1)
      if (/\bstrict\s*=\s*False\b/.test(call)) violations.push(`${where}: strict=False lets an unexpected pass go unnoticed`)
      const s = call.match(/reason\s*=\s*[rRuUbBfF]{0,2}("""|'''|"|')([\s\S]*?)\1/) ?? (m[0].includes('pytest.xfail') ? call.match(STRING_RE) : null)
      checkReason(s ? s[2] : null, where, findings, violations)
    }
  }
  for (const p of rs) {
    const text = read(root, p)
    for (const m of text.matchAll(/#\[ignore(?:\s*=\s*"((?:[^"\\]|\\.)*)")?\]/g)) {
      pins++
      checkReason(m[1] ?? null, `${p}:${lineOf(text, m.index)}`, findings, violations)
    }
  }
  return { applicable: true, ok: violations.length === 0, files: py.length + rs.length, pins, violations }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const i = argv.indexOf('--root')
  const r = checkDiscipline(resolve(i >= 0 ? argv[i + 1] : ROOT))
  if (argv.includes('--json')) console.log(JSON.stringify(r))
  else if (!r.applicable) console.log(`discipline: n/a (${r.reason})`)
  else {
    for (const v of r.violations) console.log(`VIOLATION ${v}`)
    console.log(`discipline: ${r.files} test files, ${r.pins} expected-failure pins, ${r.violations.length} violations`)
  }
  process.exit(!r.applicable ? 5 : r.ok ? 0 : 1)
}
