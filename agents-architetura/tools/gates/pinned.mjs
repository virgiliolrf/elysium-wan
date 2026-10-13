#!/usr/bin/env node
// Pinned final gate (the PR's PINNED_VERIFY + MEGA-RUN): runs EXACTLY the gates frozen in
// docs/design/GATES-<id>.json and prints one verdict, VERIFIED or NOT VERIFIED. <id> is a milestone (M0, frozen by
// wan-handout) or a phase slug (the V0..Vn of one wan-x-<slug> phase, written by wan-next and granted by the human). The script runs every command
// itself — an agent cannot substitute flags, test selections or a lucky rerun — and records the tree it ran on;
// if the tree changes while the gate runs, the verdict is NOT VERIFIED.
//
// GATES-<m>.json:
//   {"milestone": "M0", "consecutive": 2,
//    "rungs": ["factory", "build", "unit", ...],                        ladder --require, run `consecutive` times
//    "pinned": [{"id": "smoke", "cmd": "...", "repeat": 20}],           each must exit 0 on every repetition
//      optional "load": {"cmd": "...", "copies": 6}                     amplified: repetitions run while `copies` loops of
//                                                                       cmd keep the machine busy (stopped afterwards)
//      optional "metric": {"pattern": "step=([0-9.]+)", "op": "<=", "value": 1.05}
//                                                                       the number the pattern captures (last match in
//                                                                       stdout) must hold the bound on every repetition
//    "campaigns": [{"id": "rust-tsan"}],                                one fresh round each: zero unsuppressed
//    "fuzz": [{"id": "fuzz-wire", "minCpuHours": 4}],                   campaign bar + regressions replay clean
//    "audit": {"bar": "no-open-tier-a"},                                docs/security/AUDIT-RESULTS.json
//    "mutation": {"changedSince": "<rev>"},                             no surviving mutant on changed lines
//    "runAll": false}                                                   default: campaigns and mutation (the heavy tail) are
//                                                                       skipped once an earlier gate is red; true runs them
//
// Rungs: ladder.mjs passPlan decides what each consecutive pass reruns (static checks once; `full` in the last pass in
// place of the per-folder rungs it contains).
//
//   node agents-architetura/tools/gates/pinned.mjs <id> [--json] [--root DIR]
//   node agents-architetura/tools/gates/pinned.mjs <id> --validate     check the gates file against rungs.json and campaigns.json only
//   node agents-architetura/tools/gates/pinned.mjs <id> --json --label L --wait SEC [--fresh]
//                                                                     the gate in a detached run; returns within SEC (below)
// Exit: 0 VERIFIED (or valid) · 1 NOT VERIFIED · 2 usage / invalid gates file / the run failed · 3 still running (--wait)
// · 5 no gates file for the milestone.
import { existsSync, readFileSync, mkdirSync, writeFileSync, renameSync, openSync, closeSync, unlinkSync, statSync } from 'node:fs'
import { spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runLadderConsecutive, defaultRun } from './ladder.mjs'
import { identityOf, treeId } from './tree.mjs'
import { runMutation, resolveBase, pythonBaseline, pythonChanged } from './mutation.mjs'
import { runSync } from '../verify/exec.mjs'
import { withMachineLock } from '../verify/machine-lock.mjs'
import { loadCampaigns, runRound, roundClean, campaignApplicable } from '../campaign/families.mjs'
import { loadStats, barStatus, replay } from '../campaign/fuzz-run.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const ID_RE = /^[a-z0-9][a-z0-9-]{0,40}$/
// A milestone (M0) or a wan-x phase slug — the same alphabet wan-next allows for slugs.
export const GATES_ID_RE = /^(?:M\d{1,3}[a-z]?|[a-z0-9][a-z0-9-]{1,38})$/
export const gatesPath = (root, m) => join(root, 'docs', 'design', `GATES-${m}.json`)
const OPS = { '<': (a, b) => a < b, '<=': (a, b) => a <= b, '>': (a, b) => a > b, '>=': (a, b) => a >= b, '==': (a, b) => a === b }
const printable = c => typeof c === 'string' && !!c.trim() && !/[\x00-\x08\x0b-\x1f\x7f]/.test(c)

// The number a metric pattern captures from the LAST match in a run's stdout, or null.
export function metricValue(stdout, pattern) {
  let last = null
  for (const m of String(stdout ?? '').matchAll(new RegExp(pattern, 'g'))) last = m
  const v = last ? Number(last[1]) : NaN
  return Number.isFinite(v) ? v : null
}

// Background load for amplified reruns: `copies` shell loops of cmd, each in its own process group so stopping them
// also stops whatever they spawned. Plain OS processes, so they keep running while the gate blocks in spawnSync.
export function startLoad(root, load) {
  const kids = Array.from({ length: load.copies }, () => spawn('/bin/bash', ['-c', `while :; do ${load.cmd} >/dev/null 2>&1; done`], { cwd: root, detached: true, stdio: 'ignore' }))
  return () => { for (const k of kids) try { process.kill(-k.pid, 'SIGKILL') } catch {} }
}

export function validateGates(gates, { rungIds, campaigns }) {
  const errors = []
  if (!GATES_ID_RE.test(gates?.milestone ?? '')) errors.push('milestone must be a milestone like "M0" or a phase slug')
  if (gates.consecutive !== undefined && !(Number.isInteger(gates.consecutive) && gates.consecutive >= 1 && gates.consecutive <= 10)) errors.push('consecutive must be 1..10')
  for (const r of gates.rungs ?? []) if (!rungIds.includes(r)) errors.push(`unknown rung ${JSON.stringify(r)}`)
  for (const p of gates.pinned ?? []) {
    if (!ID_RE.test(p.id ?? '')) errors.push(`pinned: bad id ${JSON.stringify(p.id)}`)
    if (typeof p.cmd !== 'string' || !p.cmd.trim() || /[\x00-\x08\x0b-\x1f\x7f]/.test(p.cmd)) errors.push(`pinned ${p.id}: cmd must be a printable command`)
    if (p.repeat !== undefined && !(Number.isInteger(p.repeat) && p.repeat >= 1 && p.repeat <= 1000)) errors.push(`pinned ${p.id}: repeat must be 1..1000`)
    if (p.load !== undefined && !(printable(p.load?.cmd) && Number.isInteger(p.load?.copies) && p.load.copies >= 1 && p.load.copies <= 16)) errors.push(`pinned ${p.id}: load must be {cmd, copies 1..16}`)
    if (p.metric !== undefined) {
      let groups = -1
      try { groups = new RegExp(`${p.metric?.pattern}|`).exec('').length - 1 } catch {}
      if (typeof p.metric?.pattern !== 'string' || groups < 1) errors.push(`pinned ${p.id}: metric.pattern must be a regex with a capture group`)
      if (!Object.hasOwn(OPS, p.metric?.op ?? '')) errors.push(`pinned ${p.id}: metric.op must be one of ${Object.keys(OPS).join(' ')}`)
      if (!Number.isFinite(p.metric?.value)) errors.push(`pinned ${p.id}: metric.value must be a number`)
    }
  }
  for (const c of gates.campaigns ?? []) if (!campaigns.some(x => x.id === c.id && x.kind !== 'fuzz')) errors.push(`unknown campaign ${JSON.stringify(c.id)}`)
  for (const f of gates.fuzz ?? []) if (!campaigns.some(x => x.id === f.id && x.kind === 'fuzz')) errors.push(`unknown fuzz campaign ${JSON.stringify(f.id)}`)
  if (gates.audit && gates.audit.bar !== 'no-open-tier-a') errors.push('audit.bar must be "no-open-tier-a"')
  if (gates.mutation && !/^[\w./~^-]+$/.test(gates.mutation.changedSince ?? '')) errors.push('mutation.changedSince must be a revision')
  if (gates.runAll !== undefined && typeof gates.runAll !== 'boolean') errors.push('runAll must be true or false')
  return errors
}

// The gate ids runPinned reports for a GATES file whatever the machine's state: a phase workflow lists them (runPhase's
// gateIds), so a relayed verdict that leaves one out is not VERIFIED. "consecutive", "findings" and "tree-stable" depend on
// the run and are not listed.
export function expectedGateIds(gates) {
  return [
    ...(gates?.rungs ?? []).map(r => `rung:${r}`),
    ...(gates?.pinned ?? []).map(p => `pinned:${p.id}`),
    ...(gates?.campaigns ?? []).map(c => `campaign:${c.id}`),
    ...(gates?.fuzz ?? []).map(f => `fuzz:${f.id}`),
    ...(gates?.audit ? ['audit'] : []),
    ...(gates?.mutation ? ['mutation'] : []),
  ]
}

// docs/security/AUDIT-RESULTS.json against the audit bar: every class verdicted, no open tier-A finding.
function auditResult(root) {
  const p = join(root, 'docs', 'security', 'AUDIT-RESULTS.json')
  const a = existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : null
  const ok = !!a && a.openTierA === 0 && a.unverdicted === 0
  return { ok, detail: a ? `${a.classes} classes, ${a.surfaces} surfaces, ${a.unverdicted} without verdict, ${a.openTierA} open tier-A` : 'docs/security/AUDIT-RESULTS.json is missing — run wan-audit' }
}

export function runPinned(root, gates, { run = defaultRun, exec = runSync, load = startLoad, lock = (label, fn) => withMachineLock(root, label, fn, { waitSec: 3600 }) } = {}) {
  const rungsConfig = JSON.parse(readFileSync(join(root, 'agents-architetura', 'tools', 'gates', 'rungs.json'), 'utf8'))
  const campaigns = loadCampaigns(root)
  const errors = validateGates(gates, { rungIds: rungsConfig.rungs.map(r => r.id), campaigns })
  if (errors.length) throw new Error(`GATES-${gates?.milestone} is invalid: ${errors.join('; ')}`)
  const identity = identityOf(root)
  const results = []
  const add = (id, kind, ok, detail, extra = {}) => results.push({ id, kind, ok, detail, ...extra })
  const finish = () => {
    const findingsPath = join(root, 'agents-architetura', 'docs', 'findings.json')
    if (existsSync(findingsPath)) {
      const waiting = (JSON.parse(readFileSync(findingsPath, 'utf8')).findings ?? []).filter(f => f.status === 'needs-human')
      add('findings', 'findings', !waiting.length, waiting.length ? `${waiting.length} finding(s) await a human decision: ${waiting.slice(0, 5).map(f => f.id).join(', ')}` : 'no finding awaits a human decision')
    }
    const after = treeId(root)
    if (identity.tree && after !== identity.tree) add('tree-stable', 'identity', false, `the tree changed while the gate ran (${identity.tree} -> ${after}) — evidence is not of one tree`)
    const failing = results.filter(r => !r.ok && r.kind !== 'skipped').map(r => r.id)
    const skipped = results.filter(r => r.kind === 'skipped').map(r => r.id)
    return { milestone: gates.milestone, verdict: failing.length || skipped.length ? 'NOT VERIFIED' : 'VERIFIED', failing, ...(skipped.length ? { skipped } : {}), gates: results, ...identity }
  }

  // Preflight (minutes): what needs no product run is checked first, so a gate that cannot pass for a reason found in a
  // minute does not spend its hours of rungs and repetitions before saying so. M2b's closeout (2026-10-11) ran 9 h to
  // learn that the audit results were missing and that the mutation command could not import an extra. The preflight
  // checks the audit results, every campaign's applicability and the mutation baseline. A red preflight stops the gate:
  // its failures are reported under their own ids, every other gate is skipped, and the verdict is NOT VERIFIED.
  // "runAll": true runs the whole gate anyway.
  const mutationCfgPath = join(root, 'agents-architetura', 'tools', 'gates', 'mutation.json')
  const pre = []
  if (gates.audit) { const a = auditResult(root); if (!a.ok) pre.push(['audit', 'audit', a.detail]) }
  for (const c of gates.campaigns ?? []) {
    const r = campaignApplicable(root, campaigns.find(x => x.id === c.id), { exec })
    if (!r.applicable) pre.push([`campaign:${c.id}`, 'campaign', `not applicable: ${r.reason} — a pinned campaign must run`])
  }
  // the baseline runs only when mutation testing applies (changed Python source lines); a green one is reused below
  let mutationBaseline = null
  if (gates.mutation && existsSync(mutationCfgPath)) {
    const cfg = JSON.parse(readFileSync(mutationCfgPath, 'utf8'))
    if (cfg.python?.cmd && Object.keys(pythonChanged(root, resolveBase(root, gates.mutation.changedSince), cfg.python)).length) {
      mutationBaseline = lock('preflight:mutation', () => pythonBaseline(root, cfg, { run: exec }))
      if (!mutationBaseline.ok) pre.push(['mutation', 'mutation', mutationBaseline.detail])
    }
  }
  if (pre.length && !gates.runAll) {
    for (const [id, kind, detail] of pre) add(id, kind, false, `preflight: ${detail}`)
    for (const id of expectedGateIds(gates)) if (!results.some(r => r.id === id)) add(id, 'skipped', false, `not run: the preflight failed (${pre.map(x => x[0]).join(', ')}) — fix it, then the gate runs in full`)
    return finish()
  }

  if ((gates.rungs ?? []).length) {
    // a verdict gate owns the machine for every rung, light ones included: timing-sensitive tests (chaos) must not share it
    const l = runLadderConsecutive({ root, config: rungsConfig, require: gates.rungs, consecutive: gates.consecutive ?? 1, run, lock, slot: lock })
    for (const id of gates.rungs) {
      const r = l.rungs.find(x => x.rung === id)
      add(`rung:${id}`, 'rung', r?.status === 'pass', r ? `${r.status}${r.detail ? `: ${r.detail.slice(0, 600)}` : ''}` : 'not run')
    }
    if (l.consecutive) add('consecutive', 'rung', l.consecutive.green === l.consecutive.required, `${l.consecutive.green}/${l.consecutive.required} consecutive green passes`)
  }
  for (const p of gates.pinned ?? []) {
    const times = p.repeat ?? 1
    const values = []
    let passed = 0
    let detail = null
    // a pinned entry (repetitions, loaded reruns, soaks) owns the machine while it runs: no other build's tests beside it
    lock(`pinned:${p.id}`, () => {
    const stopLoad = p.load ? load(root, p.load) : () => {}
    try {
      for (; passed < times; passed++) {
        const r = exec(p.cmd, { cwd: root, timeoutSec: p.timeoutSec ?? 3600 })
        if (r.outcome !== 'pass') { detail = `${passed}/${times} passed, then ${r.outcome} (exit ${r.status}): ${`${r.stdout}${r.stderr}`.slice(-800)}`; break }
        if (p.metric) {
          const v = metricValue(r.stdout, p.metric.pattern)
          if (v === null) { detail = `${passed}/${times} passed, then the output had no number for /${p.metric.pattern}/: ${r.stdout.slice(-400)}`; break }
          values.push(v)
          if (!OPS[p.metric.op](v, p.metric.value)) { detail = `${passed}/${times} passed, then ${v} broke the bound ${p.metric.op} ${p.metric.value}`; break }
        }
      }
    } finally { stopLoad() }
    })
    const span = values.length ? `; metric ${Math.min(...values)}..${Math.max(...values)} (bound ${p.metric.op} ${p.metric.value})` : ''
    add(`pinned:${p.id}`, 'pinned', passed === times, `${detail ?? `${times}/${times} passed`}${p.load ? ` under ${p.load.copies}x load` : ''}${span}`, { cmd: p.cmd, ...(values.length ? { values } : {}) })
  }
  // The heavy tail runs last and only on a gate still green: a campaign round or mutation testing (hours each) on a tree
  // whose rungs or pinned entries are already red cannot change the verdict, and the fix that follows reruns the whole gate
  // anyway. Skipped gates are listed with kind "skipped" (no stabilize round is sent to them) and keep the verdict NOT
  // VERIFIED. "runAll": true in the GATES file runs them regardless.
  const redAt = () => results.find(r => !r.ok)?.id
  const tail = (id, fn) => {
    const red = gates.runAll ? null : redAt()
    if (red) add(id, 'skipped', false, `not run: the gate is already red at ${red} (set "runAll": true to run the heavy tail anyway)`)
    else fn()
  }
  for (const c of gates.campaigns ?? []) tail(`campaign:${c.id}`, () => {
    const r = runRound(root, campaigns.find(x => x.id === c.id), 999, { run: true, exec, lock })
    add(`campaign:${c.id}`, 'campaign', r.applicable && roundClean(r), !r.applicable ? `not applicable: ${r.reason} — a pinned campaign must run` : r.broken ? r.brokenReason : `${r.unsuppressedReports} unsuppressed reports in ${r.families.length} families${r.suppressionViolations.length ? `; suppression violations: ${r.suppressionViolations.join('; ')}` : ''}`)
  })
  for (const f of gates.fuzz ?? []) {
    const campaign = campaigns.find(x => x.id === f.id)
    const bar = barStatus(loadStats(root, f.id), { minCpuHours: f.minCpuHours ?? 4, quietMinutes: f.quietMinutes ?? 60 })
    const rep = replay(root, campaign, exec)
    add(`fuzz:${f.id}`, 'fuzz', bar.ok && rep.ok, [...bar.why, ...(rep.ok ? [] : [`${rep.stillCrashing.length} stored crash(es) still crash`])].join('; ') || `${bar.cpuHours} CPU-hours, quiet final ${bar.quietTailMinutes} min, ${rep.replayed} regressions replayed clean`)
  }
  if (gates.audit) { const a = auditResult(root); add('audit', 'audit', a.ok, a.detail) }
  if (gates.mutation) tail('mutation', () => {
    const m = runMutation(root, resolveBase(root, gates.mutation.changedSince), JSON.parse(readFileSync(mutationCfgPath, 'utf8')), { run: exec, baseline: mutationBaseline })
    add('mutation', 'mutation', !m.applicable || m.ok, !m.applicable ? 'no changed source lines' : `${m.survivors.length} surviving mutants${m.survivors.length ? `: ${m.survivors.slice(0, 10).map(s => `${s.file}:${s.line} ${s.op}`).join(', ')}` : ''}${m.broken.length ? `; ${m.broken.join('; ')}` : ''}`)
  })
  return finish()
}

// --wait: the gate takes hours and an agent's shell call returns within 10 minutes, so the caller polls. The first call
// starts ONE run in a detached process; later calls (another agent, a resumed workflow, a human) find that run, and a
// finished run's verdict is stored and printed again at once. A run is keyed by (id, label, tree, gates file): a new tree
// or new gates start a new run, the same key never reruns for a better result. --fresh (for a human, e.g. after the
// machine slept through a run) discards a stored verdict or failure; a run whose process died without a record (killed,
// rebooted) is started again.
export const runStatePath = (root, m, label) => join(root, '.factory', 'gates', 'runs', `${m}.${label}.json`)
const sha256 = s => createHash('sha256').update(s).digest('hex')
const readJson = p => { try { return JSON.parse(readFileSync(p, 'utf8')) } catch { return null } }
const writeAtomic = (p, obj) => { const t = `${p}.${process.pid}.tmp`; writeFileSync(t, JSON.stringify(obj)); renameSync(t, p) }
// alive and still this gate's run: a reused pid runs another command
export const runAlive = pid => {
  try { process.kill(pid, 0) } catch (e) { if (e.code !== 'EPERM') return false }
  const ps = spawnSync('ps', ['-o', 'command=', '-p', String(pid)], { encoding: 'utf8' })
  return ps.status === 0 && ps.stdout.includes('pinned.mjs') && ps.stdout.includes('--record')
}
export const runKey = (root, m) => ({ tree: treeId(root), gatesSha: sha256(readFileSync(gatesPath(root, m), 'utf8')) })

// The detached run is the only writer of its state file (running, then done or failed), so a waiter never overwrites a
// verdict; the starter holds the start lock until the run has written its first state.
async function spawnRun(root, m, statePath, key, { sleep, now }) {
  const log = openSync(`${statePath.slice(0, -5)}.log`, 'w')
  let child
  try {
    child = spawn(process.execPath, [fileURLToPath(import.meta.url), m, '--root', root, '--record', statePath], { cwd: root, detached: true, stdio: ['ignore', log, log], env: { ...process.env, PINNED_RUN_KEY: JSON.stringify(key) } })
    child.unref()
  } finally { closeSync(log) }
  const until = now() + 15000
  for (;;) {
    const s = readJson(statePath)
    if (s?.pid === child.pid) return s
    if (now() >= until) return { status: 'running', ...key, pid: child.pid, startedAt: new Date().toISOString() }
    await sleep(100)
  }
}

// Returns the run's state: {status: 'done', result} | {status: 'failed', error} | {status: 'running', ...} after waitSec.
export async function waitPinned(root, m, label, { waitSec, fresh = false, start = spawnRun, alive = runAlive, pollMs = 5000, sleep = ms => new Promise(r => setTimeout(r, ms)), now = Date.now } = {}) {
  const statePath = runStatePath(root, m, label)
  mkdirSync(dirname(statePath), { recursive: true })
  const key = runKey(root, m)
  const usable = s => !!s && s.tree === key.tree && s.gatesSha === key.gatesSha && (s.status === 'running' ? alive(s.pid) : !fresh)
  const deadline = now() + waitSec * 1000
  let s = readJson(statePath)
  while (!usable(s)) {
    // one starter: an exclusive lock file; a lock left by a starter that died is cleared after a minute
    const lock = `${statePath}.lock`
    let fd = null
    try { fd = openSync(lock, 'wx') } catch (e) {
      if (e.code !== 'EEXIST') throw e
      try { if (now() - statSync(lock).mtimeMs > 60000) unlinkSync(lock) } catch {}
    }
    if (fd !== null) {
      try {
        s = readJson(statePath)
        if (!usable(s)) s = await start(root, m, statePath, key, { sleep, now })
      } finally { closeSync(fd); unlinkSync(lock) }
      fresh = false
      break
    }
    if (now() >= deadline) return { status: 'running', ...key, starting: true }
    await sleep(Math.min(pollMs, 1000))
    s = readJson(statePath)
  }
  // short runs (tests, a stored verdict a moment away) return quickly; long ones are polled every pollMs
  let delay = Math.min(200, pollMs)
  while (s.status === 'running' && now() < deadline) {
    await sleep(delay)
    delay = Math.min(delay * 2, pollMs)
    s = readJson(statePath) ?? s
    // a run that finished between the read and the check wrote its state first: read once more before calling it dead
    if (s.status === 'running' && !alive(s.pid)) { s = readJson(statePath) ?? s; if (s.status === 'running') return { ...s, died: true } }
  }
  return s
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const valueOf = flag => { const k = argv.indexOf(flag); return k >= 0 ? argv[k + 1] : undefined }
  const valueAt = new Set(['--root', '--label', '--wait', '--record'].map(f => argv.indexOf(f)).filter(k => k >= 0).map(k => k + 1))
  const root = resolve(valueOf('--root') ?? ROOT)
  const m = argv.find((a, k) => GATES_ID_RE.test(a) && !valueAt.has(k))
  if (!m) { console.error('usage: pinned.mjs <milestone|phase-slug> [--json] [--root DIR] [--label L --wait SEC [--fresh]]'); process.exit(2) }
  if (!existsSync(gatesPath(root, m))) { console.log(`pinned: n/a (no docs/design/GATES-${m}.json)`); process.exit(5) }
  if (argv.includes('--wait')) {
    const label = valueOf('--label')
    const waitSec = Number(valueOf('--wait'))
    if (!ID_RE.test(label ?? '') || !(Number.isInteger(waitSec) && waitSec >= 1 && waitSec <= 590)) { console.error('pinned: --wait needs --label <[a-z0-9-]> and 1..590 seconds'); process.exit(2) }
    const s = await waitPinned(root, m, label, { waitSec, fresh: argv.includes('--fresh') })
    if (s.status === 'done') {
      if (argv.includes('--json')) console.log(JSON.stringify(s.result))
      else console.log(`${m}: ${s.result.verdict}${s.result.failing.length ? ` (${s.result.failing.join(', ')})` : ''} — tree ${s.result.tree ?? 'n/a'}`)
      process.exit(s.result.verdict === 'VERIFIED' ? 0 : 1)
    }
    if (s.status === 'failed') { console.error(`pinned: ${s.error} (rerun with --fresh after fixing it)`); process.exit(2) }
    const elapsedSec = s.startedAt ? Math.round((Date.now() - Date.parse(s.startedAt)) / 1000) : 0
    console.log(JSON.stringify({ status: 'running', milestone: m, label, tree: s.tree, startedAt: s.startedAt ?? null, elapsedSec, log: `${runStatePath(root, m, label).slice(0, -5)}.log` }))
    process.exit(3)
  }
  const record = valueOf('--record')
  if (argv.includes('--validate')) {
    let errors
    try {
      const rungIds = JSON.parse(readFileSync(join(root, 'agents-architetura', 'tools', 'gates', 'rungs.json'), 'utf8')).rungs.map(r => r.id)
      errors = validateGates(JSON.parse(readFileSync(gatesPath(root, m), 'utf8')), { rungIds, campaigns: loadCampaigns(root) })
    } catch (e) { errors = [e.message] }
    console.log(JSON.stringify({ ok: !errors.length, errors }))
    process.exit(errors.length ? 2 : 0)
  }
  // a detached run (--wait) records under the key its starter computed, so a waiter matches it to the tree it asked about
  const started = record ? { ...JSON.parse(process.env.PINNED_RUN_KEY ?? '{}'), pid: process.pid, startedAt: new Date().toISOString() } : null
  if (record) writeAtomic(record, { status: 'running', ...started })
  const done = extra => { if (record) writeAtomic(record, { ...started, ...extra, finishedAt: new Date().toISOString() }) }
  let r
  try {
    r = runPinned(root, JSON.parse(readFileSync(gatesPath(root, m), 'utf8')))
  } catch (e) {
    done({ status: 'failed', error: e.message })
    console.error(`pinned: ${e.message}`)
    process.exit(2)
  }
  mkdirSync(join(root, '.factory', 'gates'), { recursive: true })
  writeFileSync(join(root, '.factory', 'gates', `${m}.json`), JSON.stringify(r, null, 2))
  done({ status: 'done', result: r })
  if (argv.includes('--json')) console.log(JSON.stringify(r))
  else {
    for (const g of r.gates) console.log(`${g.ok ? 'PASS' : g.kind === 'skipped' ? 'SKIP' : 'FAIL'} ${g.id.padEnd(28)} ${g.detail.split('\n')[0]}`)
    console.log(`${m}: ${r.verdict}${r.failing.length ? ` (${r.failing.join(', ')})` : ''} — tree ${r.tree ?? 'n/a'}`)
  }
  process.exit(r.verdict === 'VERIFIED' ? 0 : 1)
}
