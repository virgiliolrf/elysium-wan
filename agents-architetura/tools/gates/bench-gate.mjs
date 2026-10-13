#!/usr/bin/env node
// Performance gate. bench/suite.json:
//   { "runs": 9, "warmup": 1, "budgetPct": 1, "maxLoadPerCore": 0.25,
//     "benches": [{ "id": "x", "cmd": "...", "higherIsBetter": false }] }
// Each cmd prints a JSON object {"value": <number>} as its LAST stdout line. Benches are run interleaved
// (A, B, A, B, ...) so slow drift hits every bench alike; the first `warmup` rounds are discarded; the median of
// the remaining `runs` is compared with bench/baseline.json ({ "<id>": <number> }). A regression beyond
// budgetPct fails the gate. A machine busier than maxLoadPerCore (1-minute load average per core) cannot
// produce a meaningful measurement: the gate refuses with exit 3 (reported N/A by the ladder), never a pass.
//
//   node agents-architetura/tools/gates/bench-gate.mjs                              compare (exit 1 on regression or missing baseline)
//   node agents-architetura/tools/gates/bench-gate.mjs --record                     (re)write baseline.json from a fresh run
//   node agents-architetura/tools/gates/bench-gate.mjs --bisect <good> <bad> --bench <id>   first commit that regressed <id>
// Exit: 0 ok · 1 regression · 2 usage/bench error · 3 machine too noisy to measure.
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs'
import { join, dirname, resolve } from 'node:path'
import { tmpdir, loadavg, cpus, totalmem, platform, release } from 'node:os'
import { fileURLToPath } from 'node:url'
import { execFileSync, spawnSync } from 'node:child_process'
import { defaultRun } from './ladder.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')

export class NoisyMachine extends Error {}

// Performance evidence names the hardware it was measured on.
export function machineInfo() {
  const gpu = spawnSync('nvidia-smi', ['--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], { encoding: 'utf8', timeout: 10000 })
  return {
    os: `${platform()} ${release()}`,
    cpu: cpus()[0]?.model ?? 'unknown',
    cores: cpus().length,
    memoryGb: Math.round(totalmem() / 2 ** 30),
    gpus: gpu.status === 0 ? gpu.stdout.trim().split('\n').filter(Boolean) : [],
    loadPerCore: Math.round(loadPerCore() * 100) / 100,
  }
}

export const median = xs => {
  const s = [...xs].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

export const loadPerCore = () => loadavg()[0] / Math.max(1, cpus().length)

export function assertQuiet(maxLoadPerCore, load = loadPerCore) {
  const l = load()
  if (l > maxLoadPerCore) throw new NoisyMachine(`machine too noisy to measure: load ${l.toFixed(2)}/core > ${maxLoadPerCore} — rerun on a quiet machine`)
}

function sample(b, { run, cwd }) {
  const r = run(b.cmd, { cwd, timeoutSec: b.timeoutSec ?? 600 })
  if (r.code !== 0) throw new Error(`bench ${b.id}: exit ${r.code}\n${r.output.slice(-2000)}`)
  const last = r.output.trim().split('\n').pop()
  let v
  try { v = JSON.parse(last).value } catch { v = undefined }
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`bench ${b.id}: last line is not {"value": number}: ${last}`)
  return v
}

// Returns {<id>: {median, min, max, samples}}.
export function measureDetailed(suite, { run = defaultRun, cwd = ROOT, load = loadPerCore, allowNoisy = false, only = null } = {}) {
  if (!allowNoisy) assertQuiet(suite.maxLoadPerCore ?? 0.25, load)
  const benches = suite.benches.filter(b => !only || only.includes(b.id))
  const runs = suite.runs ?? 9
  const warmup = suite.warmup ?? 1
  const samples = Object.fromEntries(benches.map(b => [b.id, []]))
  for (let round = 0; round < warmup + runs; round++) {
    for (const b of benches) {
      const v = sample(b, { run, cwd })
      if (round >= warmup) samples[b.id].push(v)
    }
  }
  return Object.fromEntries(Object.entries(samples).map(([id, xs]) => [id, { median: median(xs), min: Math.min(...xs), max: Math.max(...xs), samples: xs }]))
}

export const measure = (suite, opts) => Object.fromEntries(Object.entries(measureDetailed(suite, opts)).map(([id, d]) => [id, d.median]))

const regressionPct = (b, base, cur) => {
  const deltaPct = ((cur - base) / base) * 100
  return { deltaPct, regression: b.higherIsBetter ? -deltaPct : deltaPct }
}

export function compare(suite, baseline, current) {
  const budget = suite.budgetPct ?? 1
  const rows = suite.benches.map(b => {
    const base = baseline[b.id]
    const cur = typeof current[b.id] === 'object' ? current[b.id].median : current[b.id]
    if (typeof base !== 'number') return { id: b.id, ok: false, why: 'no baseline — run with --record' }
    const { deltaPct, regression } = regressionPct(b, base, cur)
    const spread = typeof current[b.id] === 'object' ? { min: current[b.id].min, max: current[b.id].max } : {}
    return { id: b.id, base, cur, deltaPct: Math.round(deltaPct * 100) / 100, ...spread, ok: regression <= budget }
  })
  return { ok: rows.every(r => r.ok), budgetPct: budget, rows }
}

// First commit in (good, bad] whose median for `benchId` regresses beyond budget relative to `good`.
// Each probe runs in a temporary git worktree, so the live tree is never checked out.
export function bisect({ root = ROOT, suite, good, bad, benchId, run = defaultRun, load = loadPerCore, allowNoisy = false }) {
  for (const rev of [good, bad]) if (!/^[\w./~^-]+$/.test(rev)) throw new Error(`bad revision: ${rev}`)
  const b = suite.benches.find(x => x.id === benchId)
  if (!b) throw new Error(`unknown bench: ${benchId}`)
  const one = { ...suite, benches: [b] }
  const scratch = mkdtempSync(join(tmpdir(), 'bisect-'))
  const at = rev => {
    const dir = join(scratch, rev.replace(/[^\w]/g, '_'))
    execFileSync('git', ['worktree', 'add', '--detach', '--quiet', dir, rev], { cwd: root, stdio: 'pipe' })
    try { return measure(one, { run, cwd: dir, load, allowNoisy })[benchId] } finally { execFileSync('git', ['worktree', 'remove', '--force', dir], { cwd: root, stdio: 'pipe' }) }
  }
  try {
    const commits = execFileSync('git', ['rev-list', '--reverse', '--ancestry-path', `${good}..${bad}`], { cwd: root, encoding: 'utf8' }).trim().split('\n').filter(Boolean)
    if (!commits.length) throw new Error(`no commits between ${good} and ${bad}`)
    const ref = at(good)
    const budget = suite.budgetPct ?? 1
    const regressed = rev => regressionPct(b, ref, at(rev)).regression > budget
    let lo = 0
    let hi = commits.length - 1
    if (!regressed(commits[hi])) return { found: false, reference: ref, why: `${bad} is within budget of ${good}` }
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (regressed(commits[mid])) hi = mid
      else lo = mid + 1
    }
    return { found: true, commit: commits[lo], reference: ref }
  } finally {
    rmSync(scratch, { recursive: true, force: true })
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const opt = name => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined }
  const allowNoisy = argv.includes('--allow-noisy')
  try {
    const suite = JSON.parse(readFileSync(join(ROOT, 'bench', 'suite.json'), 'utf8'))
    if (argv.includes('--bisect')) {
      const i = argv.indexOf('--bisect')
      const r = bisect({ suite, good: argv[i + 1], bad: argv[i + 2], benchId: opt('--bench'), allowNoisy })
      console.log(r.found ? `bench-gate: first regressing commit ${r.commit} (reference ${r.reference})` : `bench-gate: ${r.why}`)
      process.exit(0)
    }
    const current = measureDetailed(suite, { allowNoisy })
    if (argv.includes('--record')) {
      writeFileSync(join(ROOT, 'bench', 'baseline.json'), JSON.stringify(Object.fromEntries(Object.entries(current).map(([k, v]) => [k, v.median])), null, 2) + '\n')
      console.log(`bench-gate: baseline recorded for ${Object.keys(current).join(', ')}`)
      process.exit(0)
    }
    const baseline = JSON.parse(readFileSync(join(ROOT, 'bench', 'baseline.json'), 'utf8'))
    const report = compare(suite, baseline, current)
    console.log(`machine: ${JSON.stringify(machineInfo())}`)
    for (const r of report.rows) console.log(`${r.ok ? 'ok  ' : 'FAIL'} ${r.id} ${r.why ?? `${r.base} -> ${r.cur} (${r.deltaPct}%, range ${r.min}..${r.max})`}`)
    console.log(report.ok ? `BENCH GREEN (budget ${report.budgetPct}%)` : `BENCH RED (budget ${report.budgetPct}%)`)
    process.exit(report.ok ? 0 : 1)
  } catch (e) {
    console.error(`bench-gate: ${e.message}`)
    process.exit(e instanceof NoisyMachine ? 3 : 2)
  }
}
