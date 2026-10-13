#!/usr/bin/env node
// Mutation testing on the lines a change touched. A mutant the tests do not kill is a behavior change nothing
// checks — a finding, unless agents-architetura/docs/mutants-equivalent.json lists it with a justification (an equivalent
// mutant). Python mutants come from agents-architetura/tools/gates/mutate.py and run on a snapshot of the tree (never the live tree);
// Rust uses cargo-mutants --in-diff when it is installed (otherwise that engine is reported unavailable).
//
// --changed-since auto resolves the base as $FACTORY_BASE, else the merge-base with origin/main (or main), else
// the root commit.
//
// agents-architetura/tools/gates/mutation.json:
//   {"python": {"paths": ["src/"], "cmd": "uv run pytest -q -x -p no:randomly tests", "timeoutFactor": 10, "minTimeoutSec": 60, "maxMutants": 400},
//    "rust": {"cmd": "cargo mutants --in-diff {diff} --no-shuffle --output {out}", "timeoutSec": 14400}}
//
//   node agents-architetura/tools/gates/mutation.mjs --changed-since <rev|auto> [--config PATH] [--root DIR] [--json]
// Exit: 0 no surviving mutant · 1 survivors or invalid equivalents · 2 usage · 5 not applicable (no changed lines).
import { existsSync, readFileSync, writeFileSync, rmSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync, spawnSync } from 'node:child_process'
import { materialize } from './snapshot-dir.mjs'
import { runSync } from '../verify/exec.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..', '..', '..')
const MUTATE_PY = join(HERE, 'mutate.py')
const git = (root, args) => execFileSync('git', args, { cwd: root, encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 })

// {file: Set(lines)} of lines added or changed since rev (tracked diffs + untracked files), restricted by predicate.
export function changedLines(root, rev, keep) {
  if (!/^[\w./~^@{}-]+$/.test(rev)) throw new Error(`bad revision: ${JSON.stringify(rev)}`)
  const out = {}
  let file = null
  for (const line of git(root, ['diff', '-U0', '--no-color', '--no-ext-diff', rev, '--']).split('\n')) {
    if (line.startsWith('+++ ')) file = line.startsWith('+++ b/') ? line.slice(6) : null
    const h = file && keep(file) && line.match(/^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@/)
    if (h) {
      const start = Number(h[1])
      const count = h[2] === undefined ? 1 : Number(h[2])
      for (let i = 0; i < count; i++) (out[file] ??= new Set()).add(start + i)
    }
  }
  for (const f of git(root, ['ls-files', '--others', '--exclude-standard']).split('\n').filter(Boolean).filter(keep)) {
    const n = readFileSync(join(root, f), 'utf8').split('\n').length
    out[f] = new Set(Array.from({ length: n }, (_, i) => i + 1))
  }
  return out
}

export function resolveBase(root, rev) {
  if (rev !== 'auto') return rev
  if (process.env.FACTORY_BASE) return process.env.FACTORY_BASE
  for (const ref of ['origin/main', 'main']) {
    const r = spawnSync('git', ['merge-base', 'HEAD', ref], { cwd: root, encoding: 'utf8' })
    if (r.status === 0 && r.stdout.trim()) return r.stdout.trim()
  }
  const roots = spawnSync('git', ['rev-list', '--max-parents=0', 'HEAD'], { cwd: root, encoding: 'utf8' })
  if (roots.status === 0 && roots.stdout.trim()) return roots.stdout.trim().split('\n').pop()
  throw new Error('cannot resolve a base revision (no commits?) — pass --changed-since <rev>')
}

export function loadEquivalents(root) {
  const p = join(root, 'agents-architetura', 'docs', 'mutants-equivalent.json')
  const list = existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')).equivalent ?? [] : []
  const justified = e => typeof e.justification === 'string' && e.justification.trim().length >= 20
  const violations = list.filter(e => !justified(e)).map(e => `equivalent mutant ${e.file}:${e.line} ${e.op} needs a justification of at least 20 characters`)
  const keys = new Set(list.filter(justified).map(e => `${e.file}:${e.line}:${e.op}`))
  return { isEquivalent: m => keys.has(`${m.file}:${m.line}:${m.op}`), violations }
}

// The changed, non-test Python source lines under cfg.paths since rev: what mutation testing mutates.
export const pythonChanged = (root, rev, cfg) => {
  const paths = cfg.paths ?? ['src/']
  return changedLines(root, rev, f => f.endsWith('.py') && paths.some(p => f.startsWith(p)) && !/(^|\/)(test_[^/]*|[^/]*_test|conftest)\.py$/.test(f))
}

// baseline: a green pythonBaseline of the same tree (the verdict gate's preflight) — its time stands for the clean run.
export function runPython(root, rev, cfg, { python = process.env.PYTHON ?? 'python3', run = runSync, baseline = null } = {}) {
  const changed = pythonChanged(root, rev, cfg)
  const files = Object.keys(changed).sort()
  if (!files.length) return { applicable: false, reason: 'no changed Python source lines' }
  const mutants = []
  for (const f of files) {
    const r = spawnSync(python, [MUTATE_PY, join(root, f), [...changed[f]].sort((a, b) => a - b).join(',')], { encoding: 'utf8', maxBuffer: 512 * 1024 * 1024 })
    if (r.status !== 0) throw new Error(`mutate.py failed on ${f}: ${r.stderr.slice(-500)}`)
    for (const m of JSON.parse(r.stdout)) mutants.push({ file: f, ...m })
  }
  const cap = cfg.maxMutants ?? 400
  const selected = mutants.slice(0, cap)
  const { dir, tree, env } = materialize(root, 'mutation-')
  try {
    const clean = baseline?.ok ? { outcome: 'pass', ms: baseline.ms } : run(cfg.cmd, { cwd: dir, env, timeoutSec: cfg.baselineTimeoutSec ?? 3600 })
    if (clean.outcome !== 'pass') return { applicable: true, ok: false, tree, baseline: 'red', detail: `the test command fails before any mutation: ${`${clean.stdout}${clean.stderr}`.slice(-1500)}` }
    const timeoutSec = Math.max(cfg.minTimeoutSec ?? 60, Math.ceil(((cfg.timeoutFactor ?? 10) * clean.ms) / 1000))
    const survived = []
    let killed = 0
    for (const m of selected) {
      const target = join(dir, m.file)
      const original = readFileSync(target, 'utf8')
      writeFileSync(target, m.source)
      try {
        const r = run(cfg.cmd, { cwd: dir, env, timeoutSec })
        if (r.outcome === 'pass') survived.push({ file: m.file, line: m.line, op: m.op })
        else killed++
      } finally {
        writeFileSync(target, original)
      }
    }
    return { applicable: true, tree, mutants: mutants.length, ran: selected.length, truncated: mutants.length - selected.length, killed, survived }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

export function runRust(root, rev, cfg, { run = runSync } = {}) {
  if (!existsSync(join(root, 'Cargo.toml'))) return { applicable: false, reason: 'no Cargo.toml' }
  const diff = git(root, ['diff', '--no-color', '--no-ext-diff', rev, '--', '*.rs'])
  if (!diff.trim()) return { applicable: false, reason: 'no changed Rust lines' }
  if (run('cargo mutants --version', { cwd: root, timeoutSec: 60 }).outcome !== 'pass') return { applicable: true, available: false, reason: 'cargo-mutants is not installed (cargo install cargo-mutants)' }
  const tmp = mkdtempSync(join(tmpdir(), 'mutants-'))
  try {
    writeFileSync(join(tmp, 'in.diff'), diff)
    const r = run((cfg.cmd ?? 'cargo mutants --in-diff {diff} --no-shuffle --output {out}').replace('{diff}', join(tmp, 'in.diff')).replace('{out}', tmp), { cwd: root, timeoutSec: cfg.timeoutSec ?? 14400 })
    const outcomesPath = join(tmp, 'mutants.out', 'outcomes.json')
    if (!existsSync(outcomesPath)) return { applicable: true, available: true, ok: false, detail: `cargo mutants produced no outcomes: ${`${r.stdout}${r.stderr}`.slice(-1500)}` }
    const outcomes = JSON.parse(readFileSync(outcomesPath, 'utf8')).outcomes ?? []
    const mutantsOnly = outcomes.filter(o => o.scenario?.Mutant)
    const survived = mutantsOnly.filter(o => o.summary === 'MissedMutant').map(o => {
      const m = o.scenario.Mutant
      return { file: m.file ?? m.source_file?.path ?? '?', line: m.span?.start?.line ?? 0, op: `${m.genre ?? 'mutant'}:${m.replacement ?? m.name ?? ''}` }
    })
    return { applicable: true, available: true, mutants: mutantsOnly.length, killed: mutantsOnly.filter(o => o.summary === 'CaughtMutant' || o.summary === 'Timeout').length, survived }
  } finally {
    rmSync(tmp, { recursive: true, force: true })
  }
}

// The Python test command once, on a materialized copy, with no mutant. The verdict gate runs it in its preflight: a
// command that cannot run at all (M2b's closeout: an extra missing in the copy's environment) then fails the gate in
// minutes instead of after its hours of rungs and repetitions.
export function pythonBaseline(root, cfg, { run = runSync } = {}) {
  if (!cfg?.python?.cmd) return { ok: true, detail: 'no python mutation command configured' }
  const { dir, env } = materialize(root, 'mutation-baseline-')
  try {
    const r = run(cfg.python.cmd, { cwd: dir, env, timeoutSec: cfg.python.baselineTimeoutSec ?? 3600 })
    return r.outcome === 'pass' ? { ok: true, ms: r.ms, detail: 'baseline green' } : { ok: false, detail: `the test command fails before any mutation: ${`${r.stdout}${r.stderr}`.slice(-1500)}` }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

export function runMutation(root, rev, cfg, opts = {}) {
  const engines = {
    python: cfg.python ? runPython(root, rev, cfg.python, opts) : { applicable: false, reason: 'not configured' },
    rust: cfg.rust ? runRust(root, rev, cfg.rust, opts) : { applicable: false, reason: 'not configured' },
  }
  const eq = loadEquivalents(root)
  const survivors = Object.values(engines).flatMap(e => e.survived ?? []).filter(m => !eq.isEquivalent(m))
  const broken = Object.entries(engines).filter(([, e]) => e.applicable && (e.baseline === 'red' || e.ok === false)).map(([k, e]) => `${k}: ${e.detail}`)
  const applicable = Object.values(engines).some(e => e.applicable) || eq.violations.length > 0
  return { applicable, ok: !survivors.length && !broken.length && !eq.violations.length, survivors, broken, violations: eq.violations, engines }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const opt = (name, d) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : d }
  let r
  try {
    const root = resolve(opt('--root', ROOT))
    const rev = opt('--changed-since')
    if (!rev) throw new Error('--changed-since <rev> is required')
    const cfgPath = resolve(opt('--config', join(root, 'agents-architetura', 'tools', 'gates', 'mutation.json')))
    if (!existsSync(cfgPath)) { console.log('mutation: n/a (no config)'); process.exit(5) }
    r = runMutation(root, resolveBase(root, rev), JSON.parse(readFileSync(cfgPath, 'utf8')))
    if (argv.includes('--json')) console.log(JSON.stringify(r))
    else {
      for (const [k, e] of Object.entries(r.engines)) console.log(`${k}: ${e.applicable ? (e.available === false ? `UNAVAILABLE — ${e.reason}` : `${e.killed ?? 0}/${e.ran ?? e.mutants ?? 0} killed${e.truncated ? ` (${e.truncated} not run: maxMutants)` : ''}`) : `n/a (${e.reason})`}`)
      for (const s of r.survivors) console.log(`SURVIVED ${s.file}:${s.line} ${s.op}`)
      for (const b of [...r.broken, ...r.violations]) console.log(`ERROR ${b}`)
    }
  } catch (e) {
    console.error(`mutation: ${e.message}`)
    process.exit(2)
  }
  process.exit(!r.applicable ? 5 : r.ok ? 0 : 1)
}
