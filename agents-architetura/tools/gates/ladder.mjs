#!/usr/bin/env node
// Verification ladder: runs the rungs in agents-architetura/tools/gates/rungs.json in order and
// reports {allGreen, rungs:[{rung,status,detail,durationMs}], missingRequired, tree, toolchain, locks}.
// Agents never self-report gate status — they run this and paste its JSON.
//
// A rung's steps run only when all their `when` paths exist (so the ladder is
// meaningful from an empty repo to the full stack); a rung with no applicable
// step is reported as "n/a" with the reason — never silently ignored. A failing
// `blocking` rung marks everything after it "skipped"; with --fail-fast (a fix round's fast ladder, the reference PR's "stop
// adding rungs once one fails") every failing rung does. A required rung counts
// only when it passed: n/a and skipped are both missing.
//
// A step with "repeat": N passes only if all N runs exit 0 — a flaky pass is not a pass. A rung with
// "heavy": true runs under the machine lock. A rung with "onDemand": true (mutation testing: hours) runs only when it is
// named by --only or --require; otherwise it is reported n/a with that reason. --consecutive N runs the whole selection N times in a row and is
// green only if every pass is green. The report names the tree, toolchain and lockfiles it ran on, so stale
// evidence can be rejected.
//
//
// --reuse [--max-age-hours H] (the reference PR's habit: read the last run instead of repeating it): when the last report in
// .factory/ladder/last.json was produced on the same inputs — every file outside the factory, plus every factory file a
// product rung runs (the tools its commands name and everything they import, the configs they read, rungs.json), the same
// selection, toolchain and lockfiles — and is younger than H hours (default 24), it is reused and only the `factory` rung
// runs again (the factory itself may have changed). Anything else, or a red `factory` rung, runs the whole selection. A
// reused report says so ("reused"); pinned gates and closeout never reuse.
//
//   node agents-architetura/tools/gates/ladder.mjs [--json] [--only a,b] [--upto id] [--require a,b] [--consecutive N] [--fail-fast] [--reuse] [--max-age-hours H] [--config p] [--root p]
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync, mkdirSync, writeFileSync, statSync } from 'node:fs'
import { join, dirname, resolve, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { identityOf, treeId } from './tree.mjs'
import { withMachineLock, withTestSlot } from '../verify/machine-lock.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const DEFAULT_ROOT = resolve(HERE, '..', '..', '..')
const TAIL_CHARS = 4000
// The tail alone can hide WHICH check failed (a long test run ends in its summary): failure lines are reported first.
const FAILURE_LINE = /^\s*(not ok \d|✖ |(?:Assertion)?Error\b|error(?:\[|:)|FAIL(?:ED)?\b|E\s{3})|\bpanicked at\b/
// A TAP `error: |-` block carries its message on the following, deeper-indented lines: up to 6 of them come along.
export const failureLines = (output, max = 40) => {
  const lines = String(output).split('\n')
  const hits = []
  lines.forEach((l, i) => {
    if (!FAILURE_LINE.test(l)) return
    hits.push(l)
    if (!/^\s*error:\s*[|>]-?\s*$/.test(l)) return
    const depth = l.search(/\S/)
    for (let j = i + 1; j < lines.length && j <= i + 6 && lines[j].search(/\S/) > depth; j++) hits.push(lines[j])
  })
  return [...new Set(hits.map(l => l.slice(0, 300)))].slice(0, max)
}

export function defaultRun(cmd, { cwd, timeoutSec }) {
  const r = spawnSync('/bin/bash', ['-o', 'pipefail', '-c', cmd], {
    cwd, encoding: 'utf8', timeout: timeoutSec * 1000, maxBuffer: 64 * 1024 * 1024,
  })
  const timedOut = r.error?.code === 'ETIMEDOUT'
  return { code: r.status ?? (timedOut ? 124 : 1), timedOut, output: `${r.stdout ?? ''}${r.stderr ?? ''}${r.error && !timedOut ? String(r.error) : ''}` }
}

export function validateConfig(config) {
  const ids = new Set()
  for (const r of config?.rungs ?? []) {
    if (!/^[a-z][a-z0-9-]*$/.test(r.id ?? '')) throw new Error(`bad rung id: ${JSON.stringify(r.id)}`)
    if (ids.has(r.id)) throw new Error(`duplicate rung id: ${r.id}`)
    ids.add(r.id)
    if (!Array.isArray(r.steps) || !r.steps.length) throw new Error(`rung ${r.id}: needs at least one step`)
    if (r.deterministic !== undefined && typeof r.deterministic !== 'boolean') throw new Error(`rung ${r.id}: deterministic must be true or false`)
    if (r.coveredBy !== undefined && (r.coveredBy !== 'full' || r.id === 'full' || !config.rungs.some(x => x.id === 'full'))) throw new Error(`rung ${r.id}: coveredBy can only name the full rung`)
    for (const s of r.steps) {
      if (typeof s.cmd !== 'string' || !s.cmd) throw new Error(`rung ${r.id}: step without cmd`)
      if (s.repeat !== undefined && !(Number.isInteger(s.repeat) && s.repeat >= 1 && s.repeat <= 1000)) throw new Error(`rung ${r.id}: repeat must be an integer in 1..1000`)
    }
  }
  if (!ids.size) throw new Error('config has no rungs')
  return config
}

// Runs one step `repeat` times (default 1). Returns {kind: 'pass'|'na'|'fail', text}.
function runStep(s, { root, rungTimeout, run }) {
  const times = s.repeat ?? 1
  for (let i = 1; i <= times; i++) {
    const r = run(s.cmd, { cwd: root, timeoutSec: s.timeoutSec ?? rungTimeout ?? 900 })
    if (r.code === 0) continue
    if (!r.timedOut && (s.naExitCodes ?? []).includes(r.code)) {
      // "nothing to check" is a property of the input, not of the run: never repeated, never mixed with passes
      if (i === 1) return { kind: 'na', text: `$ ${s.cmd}: ${s.naReason ?? `exit ${r.code} means not applicable`}` }
    }
    const flaky = times > 1 ? `\n${i - 1}/${times} runs passed before run ${i} failed — a flaky pass is not a pass` : ''
    const lines = failureLines(r.output)
    const named = lines.length ? `\nfailures:\n${lines.join('\n')}\n--- tail ---` : ''
    return { kind: 'fail', text: `$ ${s.cmd}\n(exit ${r.code}${r.timedOut ? ', TIMED OUT' : ''})${flaky}${named}\n${r.output.slice(-TAIL_CHARS)}` }
  }
  return { kind: 'pass', text: times > 1 ? `$ ${s.cmd}: ${times}/${times} runs passed` : '' }
}

// lock: the whole machine, for heavy rungs; slot: one machine-wide test slot, for every other rung (machine-lock.mjs).
export function runLadder({ root = DEFAULT_ROOT, config, only = null, upto = null, require = [], failFast = false, run = defaultRun, now = () => performance.now(), lock = (label, fn) => fn(), slot = (label, fn) => fn() }) {
  validateConfig(config)
  const known = id => config.rungs.some(r => r.id === id)
  for (const id of [...(only ?? []), ...require, ...(upto ? [upto] : [])]) if (!known(id)) throw new Error(`unknown rung: ${id}`)
  let rungs = config.rungs
  if (upto) rungs = rungs.slice(0, rungs.findIndex(r => r.id === upto) + 1)
  if (only) rungs = rungs.filter(r => only.includes(r.id))

  const results = []
  let blockedBy = null
  for (const rung of rungs) {
    if (blockedBy) { results.push({ rung: rung.id, status: 'skipped', detail: `blocked by failing rung ${blockedBy}`, durationMs: 0 }); continue }
    if (rung.onDemand && !(only ?? []).includes(rung.id) && !require.includes(rung.id)) {
      results.push({ rung: rung.id, status: 'n/a', detail: `on demand: run it with --only ${rung.id} or --require ${rung.id} (the milestone gates require it)`, durationMs: 0 })
      continue
    }
    const steps = rung.steps.filter(s => (s.when ?? []).every(p => existsSync(join(root, p))))
    if (!steps.length) {
      const absent = [...new Set(rung.steps.flatMap(st => (st.when ?? []).filter(p => !existsSync(join(root, p)))))]
      results.push({ rung: rung.id, status: 'n/a', detail: `not applicable: ${absent.join(', ') || 'no steps'} absent`, durationMs: 0 })
      continue
    }
    const t0 = now()
    const failures = []
    const notApplicable = []
    const repeated = []
    let passed = 0
    const body = () => {
      for (const s of steps) {
        const r = runStep(s, { root, rungTimeout: rung.timeoutSec, run })
        if (r.kind === 'pass') { passed++; if (r.text) repeated.push(r.text) } else if (r.kind === 'na') notApplicable.push(r.text)
        else failures.push(r.text)
      }
    }
    if (rung.heavy) lock(`ladder:${rung.id}`, body)
    else slot(`ladder:${rung.id}`, body)
    // A step may declare exit codes that mean "nothing to check here" (pytest: 5 = no tests collected).
    const status = failures.length ? 'fail' : passed ? 'pass' : 'n/a'
    const detail = failures.length ? failures.join('\n\n') : status === 'n/a' ? `not applicable: ${notApplicable.join('; ')}` : repeated.join('; ')
    results.push({ rung: rung.id, status, detail, durationMs: Math.round(now() - t0) })
    if (status === 'fail' && (rung.blocking || failFast)) blockedBy = rung.id
  }
  const missingRequired = require.filter(id => !results.some(r => r.rung === id && r.status === 'pass'))
  return { allGreen: results.every(r => r.status !== 'fail') && !missingRequired.length, rungs: results, missingRequired }
}

// What each pass of a consecutive run executes. A later pass reruns only what a rerun can refute:
// - a rung marked "deterministic" in rungs.json (a static check: the same tree gives the same result) runs in pass 1 only;
// - when `full` is in the selection, it runs in the last pass only, and there it stands in for the per-folder rungs it
//   contains ("coveredBy": "full"), which run in every earlier pass. Each test still runs once per pass, not twice.
// M2b's closeout (2 passes, `full` beside unit, integration, multiworker, oracle, chaos, wan and fuzz) ran every test four
// times and every static check twice (2026-10-10).
export function passPlan(config, selected, consecutive) {
  const ids = new Set(selected)
  const withFull = consecutive > 1 && ids.has('full')
  return Array.from({ length: consecutive }, (_, i) => {
    const last = i === consecutive - 1
    const skip = new Map()
    for (const r of config.rungs) {
      if (!ids.has(r.id)) continue
      if (i > 0 && r.deterministic) skip.set(r.id, 'deterministic: pass 1 stands for the same tree')
      else if (withFull && last && r.coveredBy === 'full') skip.set(r.id, `covered by full in pass ${i + 1}`)
      else if (withFull && !last && r.id === 'full') skip.set(r.id, `runs in the last pass (${consecutive}) only, in place of the per-folder rungs`)
    }
    return { run: selected.filter(id => !skip.has(id)), skip }
  })
}

// Runs the same selection `consecutive` times in a row (passPlan says what each pass reruns); green only if every pass is
// green. Stops at the first red pass (the stability claim is already refuted). A rung's merged status is the worst status
// it had in the passes that ran it.
export function runLadderConsecutive({ consecutive = 1, ...opts }) {
  if (!(Number.isInteger(consecutive) && consecutive >= 1 && consecutive <= 10)) throw new Error('consecutive must be an integer in 1..10')
  if (consecutive === 1) return runLadder(opts)
  validateConfig(opts.config)
  let selected = opts.config.rungs.map(r => r.id)
  if (opts.upto) selected = selected.slice(0, selected.indexOf(opts.upto) + 1)
  if (opts.only) selected = selected.filter(id => opts.only.includes(id))
  else selected = selected.filter(id => !opts.config.rungs.find(r => r.id === id).onDemand || (opts.require ?? []).includes(id))
  const plan = passPlan(opts.config, selected, consecutive)
  // `full` standing in for a heavy rung (wan, multiworker: timing-sensitive) holds the machine as that rung would
  const heavyCovered = opts.config.rungs.some(r => r.heavy && r.coveredBy === 'full' && selected.includes(r.id))
  const config = heavyCovered ? { ...opts.config, rungs: opts.config.rungs.map(r => (r.id === 'full' ? { ...r, heavy: true } : r)) } : opts.config
  const passes = []
  for (const p of plan) {
    const r = runLadder({ ...opts, config, only: p.run, upto: null, require: (opts.require ?? []).filter(id => p.run.includes(id)) })
    passes.push(r)
    if (!r.allGreen) break
  }
  const RANK = { fail: 4, skipped: 3, 'n/a': 2, pass: 1 }
  const rungs = selected.map(id => {
    const all = passes.map((p, k) => ({ k, r: p.rungs.find(x => x.rung === id) })).filter(x => x.r)
    if (!all.length) return { rung: id, status: 'skipped', detail: `not run: pass ${passes.length}/${consecutive} was red before its turn`, durationMs: 0 }
    const worst = all.reduce((a, b) => (RANK[b.r.status] > RANK[a.r.status] ? b : a))
    const skipped = plan.slice(0, passes.length).map((p, k) => p.skip.get(id) && `pass ${k + 1}: ${p.skip.get(id)}`).filter(Boolean)
    const detail = worst.r.status === 'fail' ? `pass ${worst.k + 1}/${consecutive}: ${worst.r.detail}` : [worst.r.detail, ...skipped].filter(Boolean).join('; ')
    return { ...worst.r, detail, durationMs: all.reduce((t, x) => t + x.r.durationMs, 0) }
  })
  const red = passes.length < consecutive || passes.some(p => !p.allGreen)
  const missingRequired = (opts.require ?? []).filter(id => !rungs.some(r => r.rung === id && r.status === 'pass'))
  return {
    allGreen: !red && !missingRequired.length,
    rungs,
    missingRequired,
    consecutive: { required: consecutive, ran: passes.length, green: passes.filter(p => p.allGreen).length },
  }
}

// ---- reuse ---------------------------------------------------------------------------------------------------------
const FACTORY_PREFIXES = ['agents-architetura/', '.claude/', '.github/']
const FACTORY_FILES = ['CLAUDE.md']
const REF_RE = /agents-architetura\/[\w./-]+/g
const IMPORT_RE = /(?:from|import)\s*\(?\s*['"](\.{1,2}\/[^'"]+)['"]/g
const LITERAL_RE = /['"]([\w./-]+\.(?:mjs|js|py|json|sh))['"]/g

// The factory files a non-factory rung depends on: what its commands and when-paths name, transitively through imports and
// sibling files named in string literals. Over-inclusion only means less reuse, never stale evidence.
export function ladderInputs(root, config) {
  const needed = new Set(['agents-architetura/tools/gates/rungs.json'])
  // the runner itself decides what pass means: it and everything it imports are inputs too
  const queue = ['agents-architetura/tools/gates/ladder.mjs']
  for (const r of config.rungs) {
    if (r.id === 'factory') continue
    for (const st of r.steps) for (const ref of `${st.cmd} ${(st.when ?? []).join(' ')}`.match(REF_RE) ?? []) queue.push(ref.replace(/\/$/, ''))
  }
  while (queue.length) {
    const rel = queue.pop()
    if (needed.has(rel)) continue
    needed.add(rel)
    const abs = join(root, rel)
    if (!/\.(?:mjs|js)$/.test(rel) || !existsSync(abs) || !statSync(abs).isFile()) continue
    const text = readFileSync(abs, 'utf8')
    for (const m of [...text.matchAll(IMPORT_RE), ...text.matchAll(LITERAL_RE)]) {
      const target = relative(root, resolve(dirname(abs), m[1])).split('\\').join('/')
      if (target.startsWith('agents-architetura/') && existsSync(join(root, target))) queue.push(target)
    }
  }
  return needed
}

const inert = (path, needed) => (FACTORY_FILES.includes(path) || FACTORY_PREFIXES.some(p => path.startsWith(p)))
  && ![...needed].some(n => path === n || path.startsWith(`${n}/`))

// Content fingerprint of everything a non-factory rung can see: the working tree (tracked + untracked) minus inert
// factory files. Two trees that differ only in inert files have the same fingerprint.
export function ladderFingerprint(root, config, { tree = treeId(root) } = {}) {
  if (!tree) return null
  const needed = ladderInputs(root, config)
  const ls = spawnSync('git', ['ls-tree', '-r', '--full-tree', tree], { cwd: root, encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 })
  if (ls.status !== 0) return null
  const kept = ls.stdout.split('\n').filter(Boolean).filter(line => !inert(line.slice(line.indexOf('\t') + 1), needed))
  return createHash('sha256').update(kept.join('\n')).digest('hex')
}

const sameJson = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null)

// Why the last report cannot stand for this run, or null when it can.
export function reuseRefusal(last, { fingerprint, selection, toolchain, locks, now = Date.now(), maxAgeHours = 24 }) {
  if (!last) return 'no previous report'
  if (!fingerprint || last.fingerprint !== fingerprint) return 'the inputs changed (code, specs, tests, configs or the tools a rung runs)'
  if (!sameJson(last.selection, selection)) return 'a different rung selection'
  if (!sameJson(last.toolchain, toolchain) || !sameJson(last.locks, locks)) return 'a different toolchain or lockfile'
  const at = Date.parse(last.evidenceAt ?? last.at ?? '')
  if (!Number.isFinite(at)) return 'the previous report has no timestamp'
  if (now - at > maxAgeHours * 3600 * 1000) return `the previous report is older than ${maxAgeHours} h`
  return null
}

// The reused report: the previous rungs, with `factory` replaced by a fresh run of it.
export function mergeReused(last, fresh) {
  const factory = fresh.rungs.find(r => r.rung === 'factory')
  const rungs = last.rungs.map(r => (r.rung === 'factory' && factory ? factory : r))
  const missingRequired = (last.selection?.require ?? []).filter(id => !rungs.some(r => r.rung === id && r.status === 'pass'))
  return { allGreen: rungs.every(r => r.status !== 'fail') && !missingRequired.length, rungs, missingRequired, ...(last.consecutive ? { consecutive: last.consecutive } : {}) }
}

function parseArgs(argv) {
  const o = { json: false, only: null, upto: null, require: [], consecutive: 1, failFast: false, config: null, root: DEFAULT_ROOT, reuse: false, maxAgeHours: 24 }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    const val = () => { const v = argv[++i]; if (v === undefined) throw new Error(`${a} needs a value`); return v }
    if (a === '--json') o.json = true
    else if (a === '--only') o.only = val().split(',').filter(Boolean)
    else if (a === '--upto') o.upto = val()
    else if (a === '--require') o.require = val().split(',').filter(Boolean)
    else if (a === '--consecutive') o.consecutive = Number(val())
    else if (a === '--fail-fast') o.failFast = true
    else if (a === '--config') o.config = val()
    else if (a === '--root') o.root = resolve(val())
    else if (a === '--reuse') o.reuse = true
    else if (a === '--max-age-hours') { o.maxAgeHours = Number(val()); if (!(o.maxAgeHours > 0 && o.maxAgeHours <= 168)) throw new Error('--max-age-hours must be in (0, 168]') }
    else throw new Error(`unknown argument: ${a}`)
  }
  o.config ??= join(o.root, 'agents-architetura', 'tools', 'gates', 'rungs.json')
  return o
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  let o
  try { o = parseArgs(process.argv.slice(2)) } catch (e) { console.error(`ladder: ${e.message}`); process.exit(2) }
  const config = JSON.parse(readFileSync(o.config, 'utf8'))
  const lock = (label, fn) => withMachineLock(o.root, label, fn, { waitSec: 3600 })
  const slot = (label, fn) => withTestSlot(label, fn, { waitSec: 3600 })
  const lastPath = join(o.root, '.factory', 'ladder', 'last.json')
  const selection = { only: o.only, upto: o.upto, require: o.require, consecutive: o.consecutive, ...(o.failFast ? { failFast: true } : {}) }
  let report
  try {
    const identity = identityOf(o.root)
    const fingerprint = ladderFingerprint(o.root, config, { tree: identity.tree })
    const at = new Date().toISOString()
    let last = null
    try { last = JSON.parse(readFileSync(lastPath, 'utf8')) } catch {}
    const refusal = o.reuse ? reuseRefusal(last, { fingerprint, selection, toolchain: identity.toolchain, locks: identity.locks, maxAgeHours: o.maxAgeHours }) : 'not asked (--reuse)'
    const fresh = refusal ? null : runLadder({ root: o.root, config, only: ['factory'], lock, slot })
    if (fresh && fresh.rungs.every(r => r.status !== 'fail')) {
      const evidenceAt = last.evidenceAt ?? last.at
      report = { ...mergeReused(last, fresh), ...identity, fingerprint, selection, at, evidenceAt, reused: { evidenceAt, fromTree: last.tree, rerun: ['factory'], why: 'every input a product rung reads is unchanged; only the factory rung ran again' } }
    } else {
      if (o.reuse) console.error(`ladder: not reusing the last report: ${refusal ?? 'the factory rung is red now'}`)
      report = { ...runLadderConsecutive({ root: o.root, config, only: o.only, upto: o.upto, require: o.require, consecutive: o.consecutive, failFast: o.failFast, lock, slot }), ...identity, fingerprint, selection, at, evidenceAt: at }
    }
  } catch (e) { console.error(`ladder: ${e.message}`); process.exit(2) }
  mkdirSync(join(o.root, '.factory', 'ladder'), { recursive: true })
  writeFileSync(lastPath, JSON.stringify(report, null, 2))
  if (o.json) console.log(JSON.stringify(report))
  else {
    for (const r of report.rungs) console.log(`${r.status.toUpperCase().padEnd(7)} ${r.rung.padEnd(12)} ${r.status === 'n/a' ? r.detail : `${r.durationMs}ms`}${r.status === 'fail' ? `\n${r.detail}\n` : ''}`)
    if (report.missingRequired.length) console.log(`required but not passing: ${report.missingRequired.join(', ')}`)
    if (report.consecutive) console.log(`consecutive: ${report.consecutive.green}/${report.consecutive.required} green passes`)
    console.log(`tree ${report.tree ?? 'n/a (not a git repository)'}`)
    if (report.reused) console.log(`reused the report of ${report.reused.evidenceAt} (same inputs); factory rung re-run`)
    console.log(report.allGreen ? 'LADDER GREEN' : 'LADDER RED')
  }
  process.exit(report.allGreen ? 0 : 1)
}
