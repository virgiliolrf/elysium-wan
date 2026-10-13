#!/usr/bin/env node
// Flag-off identity (the PR's V5a): for a deterministic sample of the corpus, the feature-off run must be
// identical to the reference run — exit code, stdout+stderr and every artifact the run writes into
// $FACTORY_IDENTITY_OUT (training state under state/, wire captures under wire/), compared by sha256. With
// --against-base the feature-off run is also compared with the same command on the base revision (a temporary
// git worktree).
// Byte-exact rule (CLAUDE.md, "Product rule"): with "exact", the production path (every feature on) must leave
// training state byte-identical to the reference run's — the artifacts matching exact.state, plus the exit code.
// Wire bytes and logs legitimately differ there, so only the state is compared; a side that writes no state fails.
//
// agents-architetura/tools/verify/identity.json:
//   {"corpus": ["tests/oracle/**/test_*.py"], "every": 1, "max": 40,
//    "a": "FACTORY_FEATURES=off uv run pytest -q {test}", "b": "FACTORY_REFERENCE=1 uv run pytest -q {test}",
//    "exact": {"cmd": "uv run pytest -q {test}", "state": ["state/**"]},
//    "normalize": true, "baseRev": "origin/main", "timeoutSec": 600}
// normalize=true masks durations and addresses only; set false for raw byte identity.
//
//   node agents-architetura/tools/verify/identity.mjs [--config PATH] [--against-base] [--json]
// Exit: 0 identical · 1 mismatch or no "exact" check · 2 usage · 5 not applicable (no config / empty corpus).
import { existsSync, readFileSync, mkdtempSync, rmSync, mkdirSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { tmpdir } from 'node:os'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync } from 'node:child_process'
import { runSync, normalizeOutput, shellQuote, optionReader } from './exec.mjs'
import { glob, globToRegExp, walk } from './glob.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
export const OUT_ENV = 'FACTORY_IDENTITY_OUT'

export function selectSample(files, every = 1, max = 40) {
  return [...files].sort().filter((_, i) => i % every === 0).slice(0, max)
}

const hashDir = dir => Object.fromEntries(walk(dir).sort().map(p => [p, createHash('sha256').update(readFileSync(join(dir, p))).digest('hex')]))

function runSide(cmdTemplate, test, { cwd, scratch, label, timeoutSec, run }) {
  const out = join(scratch, label)
  mkdirSync(out, { recursive: true })
  const r = run(cmdTemplate.replaceAll('{test}', shellQuote(test)), { cwd, env: { [OUT_ENV]: out }, timeoutSec })
  return { status: r.status, signal: r.signal ?? null, outcome: r.outcome, text: `${r.stdout}${r.stderr}`, artifacts: hashDir(out) }
}

function diffSides(x, y, normalize) {
  const why = []
  if (x.status !== y.status || x.signal !== y.signal) why.push(`exit ${x.status}/${y.status}`)
  const tx = normalize ? normalizeOutput(x.text) : x.text
  const ty = normalize ? normalizeOutput(y.text) : y.text
  if (tx !== ty) {
    const lx = tx.split('\n')
    const ly = ty.split('\n')
    const i = lx.findIndex((l, k) => l !== ly[k])
    why.push(`output differs at line ${i + 1}: ${JSON.stringify(lx[i] ?? '')} vs ${JSON.stringify(ly[i] ?? '')}`)
  }
  const keys = [...new Set([...Object.keys(x.artifacts), ...Object.keys(y.artifacts)])].sort()
  for (const k of keys) if (x.artifacts[k] !== y.artifacts[k]) why.push(`artifact ${k} differs`)
  return why
}

// Byte-exact rule: only the training state is compared, and a run that wrote none proves nothing.
function diffState(on, ref, patterns) {
  const res = patterns.map(globToRegExp)
  const pick = side => Object.fromEntries(Object.entries(side.artifacts).filter(([k]) => res.some(re => re.test(k))))
  const x = pick(on)
  const y = pick(ref)
  const why = []
  if (on.status !== ref.status || on.signal !== ref.signal) why.push(`exit ${on.status}/${ref.status}`)
  if (!Object.keys(y).length) why.push(`reference wrote no training state matching ${patterns.join(', ')}`)
  if (!Object.keys(x).length) why.push(`production path wrote no training state matching ${patterns.join(', ')}`)
  for (const k of [...new Set([...Object.keys(x), ...Object.keys(y)])].sort()) {
    if (!(k in x)) why.push(`state ${k} written only by the reference`)
    else if (!(k in y)) why.push(`state ${k} written only by the production path`)
    else if (x[k] !== y[k]) why.push(`state ${k} is not byte-identical to the reference`)
  }
  return why
}

export function runIdentity(cfg, { root = ROOT, againstBase = false, run = runSync } = {}) {
  for (const k of ['a', 'b']) if (typeof cfg?.[k] !== 'string' || !cfg[k].includes('{test}')) throw new Error(`identity: "${k}" must be a command containing {test}`)
  const exact = cfg.exact ?? null
  if (exact !== null) {
    if (typeof exact?.cmd !== 'string' || !exact.cmd.includes('{test}')) throw new Error('identity: "exact.cmd" must be a command containing {test}')
    if (!Array.isArray(exact.state) || !exact.state.length || exact.state.some(p => typeof p !== 'string' || !p)) throw new Error('identity: "exact.state" must be a non-empty list of artifact globs')
  }
  const files = glob(root, cfg.corpus ?? [])
  const sample = selectSample(files, cfg.every ?? 1, cfg.max ?? 40)
  if (!sample.length) return { applicable: false, reason: 'corpus matched no files' }
  const normalize = cfg.normalize !== false
  const timeoutSec = cfg.timeoutSec ?? 600
  const scratch = mkdtempSync(join(tmpdir(), 'identity-'))
  let base = null
  try {
    if (againstBase) {
      if (typeof cfg.baseRev !== 'string' || !/^[\w./~^-]+$/.test(cfg.baseRev)) throw new Error('identity: --against-base needs a valid "baseRev"')
      base = join(scratch, 'base-tree')
      execFileSync('git', ['worktree', 'add', '--detach', '--quiet', base, cfg.baseRev], { cwd: root, stdio: 'pipe' })
    }
    const rows = sample.map((test, i) => {
      const a = runSide(cfg.a, test, { cwd: root, scratch, label: `${i}-a`, timeoutSec, run })
      const b = runSide(cfg.b, test, { cwd: root, scratch, label: `${i}-b`, timeoutSec, run })
      const row = { test, mismatch: diffSides(a, b, normalize) }
      if (exact) row.exactMismatch = diffState(runSide(exact.cmd, test, { cwd: root, scratch, label: `${i}-on`, timeoutSec, run }), b, exact.state)
      if (base) {
        if (!existsSync(join(base, test))) row.base = 'absent at base (new test)'
        else {
          const ab = runSide(cfg.a, test, { cwd: base, scratch, label: `${i}-base`, timeoutSec, run })
          row.baseMismatch = diffSides(a, ab, normalize)
        }
      }
      return row
    })
    const bad = rows.filter(r => r.mismatch.length || (r.baseMismatch ?? []).length || (r.exactMismatch ?? []).length)
    return { applicable: true, ok: bad.length === 0, exact: !!exact, sampled: rows.length, corpus: files.length, mismatches: bad.length, rows }
  } finally {
    if (base) { try { execFileSync('git', ['worktree', 'remove', '--force', base], { cwd: root, stdio: 'pipe' }) } catch { /* reported by git worktree prune */ } }
    rmSync(scratch, { recursive: true, force: true })
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const o = optionReader(process.argv.slice(2))
  let r
  try {
    const path = resolve(o.get('--config', join(ROOT, 'agents-architetura', 'tools', 'verify', 'identity.json')))
    const againstBase = o.flag('--against-base')
    const json = o.flag('--json')
    const root = resolve(o.get('--root', ROOT))
    if (o.unknown().length) throw new Error(`unknown option(s): ${o.unknown().join(' ')}`)
    if (!existsSync(path)) { console.log('identity: n/a (no config)'); process.exit(5) }
    r = runIdentity(JSON.parse(readFileSync(path, 'utf8')), { root, againstBase })
    if (json) console.log(JSON.stringify(r))
    else if (!r.applicable) console.log(`identity: n/a (${r.reason})`)
    else {
      for (const row of r.rows) {
        const problems = [...row.mismatch, ...(row.baseMismatch ?? []).map(m => `vs base: ${m}`), ...(row.exactMismatch ?? []).map(m => `features on vs reference: ${m}`)]
        console.log(`${problems.length ? 'MISMATCH' : 'OK      '} ${row.test}${row.base ? ` (${row.base})` : ''}${problems.length ? `\n  ${problems.join('\n  ')}` : ''}`)
      }
      console.log(`identity: ${r.sampled - r.mismatches}/${r.sampled} identical (corpus ${r.corpus})${r.exact ? '' : '\nidentity: FAIL — no "exact" check configured, so the byte-exact rule (CLAUDE.md, "Product rule") is unverified'}`)
    }
  } catch (e) {
    console.error(`identity: ${e.message}`)
    process.exit(2)
  }
  // A corpus without the byte-exact check is not a pass: the product rule would go unverified.
  process.exit(!r.applicable ? 5 : r.ok && r.exact ? 0 : 1)
}
