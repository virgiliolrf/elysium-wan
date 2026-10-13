#!/usr/bin/env node
// Fuzz campaigns (the PR's Fuzzilli rig, engine-agnostic). A campaign in agents-architetura/tools/campaign/campaigns.json with
// kind "fuzz" names:
//   cmd         the fuzzer invocation; placeholders {corpus} {crashes} {seconds} {jobs}
//   repro       replays ONE input; placeholder {input}; nonzero exit = still crashing
//   selfTest    optional: proves the generators reach the code before any CPU is spent ("fail fast instead of
//               fuzzing nothing")
//   regressions directory of minimized repros that must never crash again (seeded into the corpus)
// State lives in .factory/fuzz/<id>/{corpus,crashes,stats.json} and survives runs (--resume semantics).
// Every new crash is replayed: its output is parsed into a signature (dedupe key); a crash that does not
// reproduce is kept and marked flaky, never dropped.
//
//   node agents-architetura/tools/campaign/fuzz-run.mjs --campaign <id> --minutes M [--jobs J] [--json]
//   node agents-architetura/tools/campaign/fuzz-run.mjs --campaign <id> --replay [--input PATH] [--json]   regressions + stored crashes (or one input)
//   node agents-architetura/tools/campaign/fuzz-run.mjs --campaign <id> --status [--json]
//   node agents-architetura/tools/campaign/fuzz-run.mjs --replay-all [--json]                   every applicable fuzz campaign (ladder rung)
// Exit: 0 no new unique crash (run) / nothing crashes (replay) · 1 otherwise · 2 usage · 5 not applicable.
import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync, statSync, copyFileSync } from 'node:fs'
import { join, resolve, dirname, basename } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createHash } from 'node:crypto'
import { runSync, shellQuote } from '../verify/exec.mjs'
import { withMachineLock } from '../verify/machine-lock.mjs'
import { parseText } from './parse.mjs'
import { loadCampaigns } from './families.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const HOUR = 3600

export const fuzzDir = (root, id) => join(root, '.factory', 'fuzz', id)
const files = dir => (existsSync(dir) ? readdirSync(dir).filter(f => !f.startsWith('.')).map(f => join(dir, f)).filter(p => statSync(p).isFile()).sort() : [])
// Regression inputs may be grouped in one directory per signature, each with a README.md stating its claim.
const regressionInputs = dir => (existsSync(dir)
  ? readdirSync(dir).filter(f => !f.startsWith('.')).flatMap(f => (statSync(join(dir, f)).isDirectory() ? regressionInputs(join(dir, f)) : [join(dir, f)])).filter(p => !/\.md$/i.test(p)).sort()
  : [])
const sha = p => createHash('sha256').update(readFileSync(p)).digest('hex').slice(0, 12)

export function loadStats(root, id) {
  const p = join(fuzzDir(root, id), 'stats.json')
  return existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : { cpuSeconds: 0, runs: [], signatures: {} }
}

// Replays one input; returns {crashes, signature, flaky?, tail}.
export function triageInput(root, campaign, input, exec = runSync) {
  const r = exec(campaign.repro.replaceAll('{input}', shellQuote(input)), { cwd: root, timeoutSec: campaign.reproTimeoutSec ?? 120 })
  const text = `${r.stdout}${r.stderr}`
  if (r.outcome === 'pass') return { crashes: false, tail: text.slice(-500) }
  const parsed = parseText(text)[0]
  const last = text.trim().split('\n').filter(Boolean).pop() ?? r.outcome
  const signature = parsed?.signature ?? `fuzz-${r.outcome}|${last.replace(/0x[0-9a-f]+/gi, '<addr>').replace(/\b\d+\b/g, 'N').slice(0, 160)}`
  return { crashes: true, signature, location: parsed?.location ?? null, tail: text.slice(-1500) }
}

export function fuzzOnce(root, campaign, { minutes, jobs = 1, exec = runSync, now = () => Date.now() / 1000, lock = (label, fn) => withMachineLock(root, label, fn, { waitSec: 3600 }) }) {
  const dir = fuzzDir(root, campaign.id)
  const corpus = join(dir, 'corpus')
  const crashes = join(dir, 'crashes')
  mkdirSync(corpus, { recursive: true })
  mkdirSync(crashes, { recursive: true })
  if (campaign.regressions) for (const f of regressionInputs(join(root, campaign.regressions))) copyFileSync(f, join(corpus, `regression-${sha(f)}-${basename(f)}`))
  if (campaign.selfTest) {
    const st = exec(campaign.selfTest, { cwd: root, timeoutSec: 600 })
    if (st.outcome !== 'pass') return { ok: false, selfTestFailed: true, detail: `self-test failed — the fuzzer would be fuzzing nothing: ${`${st.stdout}${st.stderr}`.slice(-1500)}` }
  }
  const before = new Set(files(crashes))
  const seconds = Math.round(minutes * 60)
  const start = now()
  const cmd = campaign.cmd.replaceAll('{corpus}', shellQuote(corpus)).replaceAll('{crashes}', shellQuote(crashes)).replaceAll('{seconds}', String(seconds)).replaceAll('{jobs}', String(jobs))
  const r = lock(`fuzz:${campaign.id}`, () => exec(cmd, { cwd: root, timeoutSec: seconds + 600 }))
  const end = now()
  const stats = loadStats(root, campaign.id)
  const fresh = files(crashes).filter(p => !before.has(p))
  const newUnique = []
  for (const p of fresh) {
    const t = triageInput(root, campaign, p, exec)
    const signature = t.crashes ? t.signature : `fuzz-flaky|${sha(p)}`
    const firstSeenSec = Math.max(0, Math.round(statSync(p).mtimeMs / 1000 - start))
    const known = stats.signatures[signature]
    if (known) { known.files.push(basename(p)); continue }
    stats.signatures[signature] = { firstSeen: new Date(end * 1000).toISOString(), files: [basename(p)], flaky: !t.crashes, location: t.location ?? null, tail: t.tail }
    newUnique.push({ signature, file: p, firstSeenSec, flaky: !t.crashes })
  }
  const elapsed = Math.round(end - start)
  stats.cpuSeconds += elapsed * jobs
  const lastNewUniqueSec = newUnique.length ? Math.max(...newUnique.map(n => n.firstSeenSec)) : null
  stats.runs.push({ start: new Date(start * 1000).toISOString(), seconds: elapsed, jobs, exit: r.status, newCrashFiles: fresh.length, newUnique: newUnique.map(n => n.signature), lastNewUniqueSec })
  writeFileSync(join(dir, 'stats.json'), JSON.stringify(stats, null, 2))
  return {
    ok: newUnique.length === 0,
    campaign: campaign.id,
    seconds: elapsed,
    cpuHours: Math.round((stats.cpuSeconds / HOUR) * 100) / 100,
    corpusSize: files(corpus).length,
    newCrashFiles: fresh.length,
    newUnique,
    quietTailMinutes: Math.round((elapsed - (lastNewUniqueSec ?? 0)) / 60),
    uniqueTotal: Object.keys(stats.signatures).length,
  }
}

export function replay(root, campaign, exec = runSync, only = null) {
  const inputs = only ? [join(root, only)] : [...(campaign.regressions ? regressionInputs(join(root, campaign.regressions)) : []), ...files(join(fuzzDir(root, campaign.id), 'crashes'))]
  if (!inputs.length) return { ok: true, replayed: 0, stillCrashing: [] }
  const stillCrashing = inputs.map(p => ({ input: p, ...triageInput(root, campaign, p, exec) })).filter(t => t.crashes).map(t => ({ input: t.input, signature: t.signature }))
  return { ok: stillCrashing.length === 0, replayed: inputs.length, stillCrashing }
}

// The campaign bar: enough CPU time, and no new unique crash in the final hour of the latest run.
export function barStatus(stats, { minCpuHours = 4, quietMinutes = 60 } = {}) {
  const last = stats.runs.at(-1)
  const cpuHours = stats.cpuSeconds / HOUR
  const quiet = last ? (last.seconds - (last.lastNewUniqueSec ?? 0)) / 60 : 0
  const why = []
  if (cpuHours < minCpuHours) why.push(`${cpuHours.toFixed(2)} CPU-hours < ${minCpuHours}`)
  if (!last) why.push('no campaign run recorded')
  else if (quiet < quietMinutes) why.push(`a new unique crash appeared ${Math.round(quiet)} min before the end of the last run (< ${quietMinutes})`)
  return { ok: !why.length, cpuHours: Math.round(cpuHours * 100) / 100, quietTailMinutes: Math.round(quiet), why }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const opt = (name, d) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : d }
  const json = argv.includes('--json')
  let out
  try {
    const root = resolve(opt('--root', ROOT))
    if (argv.includes('--replay-all')) {
      const fuzz = loadCampaigns(root).filter(c => c.kind === 'fuzz' && (c.when ?? []).every(p => existsSync(join(root, p))))
      if (!fuzz.length) { console.log('fuzz: n/a (no applicable fuzz campaign)'); process.exit(5) }
      const results = fuzz.map(c => ({ id: c.id, ...replay(root, c) }))
      out = { ok: results.every(r => r.ok), results }
      console.log(json ? JSON.stringify(out) : JSON.stringify(out, null, 2))
      process.exit(out.ok ? 0 : 1)
    }
    const campaign = loadCampaigns(root).find(c => c.id === opt('--campaign') && c.kind === 'fuzz')
    if (!campaign) throw new Error(`no fuzz campaign ${JSON.stringify(opt('--campaign'))} in agents-architetura/tools/campaign/campaigns.json`)
    if ((campaign.when ?? []).some(p => !existsSync(join(root, p)))) { console.log(`fuzz: n/a (${campaign.when.join(', ')} absent)`); process.exit(5) }
    if (argv.includes('--status')) {
      const stats = loadStats(root, campaign.id)
      out = { ...barStatus(stats, { minCpuHours: Number(opt('--min-cpu-hours', '4')) }), uniqueTotal: Object.keys(stats.signatures).length, runs: stats.runs.length }
    } else if (argv.includes('--replay')) {
      const input = opt('--input')
      if (input && (input.includes('..') || input.startsWith('/') || !existsSync(join(root, input)))) throw new Error(`--input must be an existing repo-relative path: ${input}`)
      out = replay(root, campaign, runSync, input ?? null)
    }
    else {
      const minutes = Number(opt('--minutes'))
      if (!(minutes > 0)) throw new Error('--minutes M is required')
      out = fuzzOnce(root, campaign, { minutes, jobs: Number(opt('--jobs', '1')) })
    }
    console.log(json ? JSON.stringify(out) : JSON.stringify(out, null, 2))
  } catch (e) {
    console.error(`fuzz: ${e.message}`)
    process.exit(2)
  }
  process.exit(out.ok ? 0 : 1)
}
