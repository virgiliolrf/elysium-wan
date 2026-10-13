#!/usr/bin/env node
// Paid experiments on Vast.ai, with the human gates in code rather than in prompts:
//   - nothing launches unless a human granted the exact experiment plan (agents-architetura/tools/factory/approve.mjs) AND the local
//     promotion record says the hypothesis survived locally (docs/experiments/<id>/LOCAL.json);
//   - nothing launches over the approved budget (offers are priced before creation);
//   - monitor pauses (stops instances, keeps disks) at warnAtPct of the budget and emergency-stops at 100%, on
//     divergence, a dead or stalled worker, repeated crashes or a vanished instance — preserving logs and metrics first;
//   - every instance is recorded in a ledger and `stop` destroys them all;
//   - with the human's hardware switch off (agents-architetura/tools/factory/hardware.json, gpu false — the default), no
//     account action runs at all: only validate, status, monitor and stop, which never rent and only ever reduce spending.
//
// Plan: docs/experiments/<id>/PLAN.json — see agents-architetura/docs/EXPERIMENTS.md for the fields.
//
//   node agents-architetura/tools/vast/vast.mjs validate|estimate|launch|preflight|start|monitor|stop|status --plan <PLAN.json> [--json]
//   node agents-architetura/tools/vast/vast.mjs doctor          is the CLI installed and the account reachable (no spending)
// Exit: 0 ok · 1 refused / failed check / stop triggered · 2 usage · 3 paused at the budget warning (human decision).
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { checkApproval } from '../factory/approve.mjs'
import { readHardware, HARDWARE_FILE } from '../factory/hardware.mjs'
import { clientFromEnv } from './client.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const ID_RE = /^[a-z0-9][a-z0-9-]{1,48}$/
const PROMOTABLE = ['VALIDATED', 'PARTIALLY VALIDATED']
// Subcommands that reach the Vast.ai account or rent compute. Turning the switch off never blocks stopping.
const ACCOUNT = new Set(['doctor', 'estimate', 'launch', 'preflight', 'start'])

export function hardwareRefusal(cmd, hw) {
  if (hw.gpu === true || !ACCOUNT.has(cmd)) return null
  return `hardware mode is gpu=false (${HARDWARE_FILE}: ${hw.reason ?? 'not true'}) — simulation only, no Vast.ai account action; a human turns GPUs on with /wan-gpu`
}

export function validatePlan(plan) {
  const e = []
  if (!ID_RE.test(plan?.id ?? '')) e.push('id must match ^[a-z0-9][a-z0-9-]{1,48}$')
  if (typeof plan?.hypothesis !== 'string' || plan.hypothesis.length < 20) e.push('hypothesis is required')
  if (!plan?.expected || typeof plan.expected !== 'object' || !Object.keys(plan.expected).length) e.push('expected metrics are required (the prediction the run will be classified against)')
  if (!(plan?.maxBudgetUsd > 0)) e.push('maxBudgetUsd must be > 0')
  if (!(plan?.expectedHours > 0)) e.push('expectedHours must be > 0')
  if (!(plan?.warnAtPct > 0 && plan.warnAtPct < 100)) e.push('warnAtPct must be in (0, 100)')
  if (!Array.isArray(plan?.instances) || !plan.instances.length) e.push('instances are required')
  for (const [i, r] of (plan?.instances ?? []).entries()) {
    if (!/^[a-z][a-z0-9-]{0,20}$/.test(r?.role ?? '')) e.push(`instances[${i}].role is invalid`)
    if (!(Number.isInteger(r?.count) && r.count >= 1 && r.count <= 64)) e.push(`instances[${i}].count must be 1..64`)
    if (!(r?.maxDph > 0)) e.push(`instances[${i}].maxDph must be > 0`)
    if (typeof r?.query !== 'string' || !r.query.trim()) e.push(`instances[${i}].query (vastai offer filter) is required`)
  }
  for (const k of ['image', 'run', 'metrics']) if (typeof plan?.[k] !== 'string' || !plan[k].trim()) e.push(`${k} is required`)
  if (!(plan?.diskGb >= 10)) e.push('diskGb must be >= 10')
  return e
}

const ledgerPath = (root, id) => join(root, '.factory', 'vast', id, 'ledger.json')
const loadLedger = (root, id) => (existsSync(ledgerPath(root, id)) ? JSON.parse(readFileSync(ledgerPath(root, id), 'utf8')) : { instances: [], events: [] })
const saveLedger = (root, id, l) => { mkdirSync(dirname(ledgerPath(root, id)), { recursive: true }); writeFileSync(ledgerPath(root, id), JSON.stringify(l, null, 2)) }
const event = (l, type, detail, now) => { l.events.push({ at: new Date(now).toISOString(), type, detail }) }

// Cheapest offers per role that satisfy the price cap; the estimate is what launch will actually rent.
export function estimate(plan, client) {
  const picks = []
  const problems = []
  for (const r of plan.instances) {
    const offers = client.searchOffers(r.query).filter(o => o.dph_total <= r.maxDph).sort((a, b) => a.dph_total - b.dph_total)
    if (offers.length < r.count) problems.push(`role ${r.role}: ${offers.length} offers under $${r.maxDph}/h, need ${r.count}`)
    picks.push(...offers.slice(0, r.count).map(o => ({ role: r.role, offerId: o.id, dph: o.dph_total, gpu: o.gpu_name ?? null })))
  }
  const dph = picks.reduce((s, p) => s + p.dph, 0)
  const total = Math.round(dph * plan.expectedHours * 100) / 100
  if (total > plan.maxBudgetUsd) problems.push(`estimated $${total} exceeds the approved budget $${plan.maxBudgetUsd}`)
  const perMTokens = plan.expectedTokens > 0 ? Math.round((total / (plan.expectedTokens / 1e6)) * 10000) / 10000 : null
  return { ok: !problems.length, picks, dphTotal: Math.round(dph * 1000) / 1000, estimatedUsd: total, budgetUsd: plan.maxBudgetUsd, costPerMillionTokensUsd: perMTokens, problems }
}

// The promotion gate: a human-approved plan, a local result that supports the hypothesis, and no instance already running.
export function launchGate(root, planPath, plan) {
  const why = []
  const off = hardwareRefusal('launch', readHardware(root))
  if (off) why.push(off)
  // strict: money is never covered by the autonomy switch; a paid plan needs a real human grant
  const approval = checkApproval(root, planPath, { strict: true })
  if (!approval.approved) why.push(`the experiment plan is not approved by a human (${approval.reason}) — node agents-architetura/tools/factory/approve.mjs grant ${planPath}`)
  const localPath = join(root, 'docs', 'experiments', plan.id, 'LOCAL.json')
  const local = existsSync(localPath) ? JSON.parse(readFileSync(localPath, 'utf8')) : null
  if (!local) why.push('no local result (docs/experiments/<id>/LOCAL.json): local before paid cloud')
  else {
    if (!PROMOTABLE.includes(local.classification)) why.push(`the local experiment is ${local.classification} — only VALIDATED or PARTIALLY VALIDATED hypotheses go to paid compute`)
    const failed = Object.entries(local.checklist ?? {}).filter(([, v]) => v !== true).map(([k]) => k)
    if (failed.length) why.push(`promotion checklist incomplete: ${failed.join(', ')}`)
  }
  return why
}

export function launch(root, planPath, plan, client, now = Date.now()) {
  const why = launchGate(root, planPath, plan)
  const l = loadLedger(root, plan.id)
  if (l.instances.some(i => !i.destroyed)) why.push('instances from an earlier launch are still recorded as live — stop them first')
  const est = estimate(plan, client)
  why.push(...est.problems)
  if (why.length) return { ok: false, refused: why, estimate: est }
  for (const p of est.picks) {
    const r = client.createInstance(p.offerId, { image: plan.image, diskGb: plan.diskGb, label: `factory-${plan.id}-${p.role}` })
    const id = r?.new_contract ?? r?.id
    if (!id) { event(l, 'create-failed', JSON.stringify(r).slice(0, 300), now); continue }
    l.instances.push({ id, role: p.role, offerId: p.offerId, dph: p.dph, createdAt: now })
  }
  l.plan = { id: plan.id, budgetUsd: plan.maxBudgetUsd, warnAtPct: plan.warnAtPct }
  event(l, 'launched', `${l.instances.length} instances, estimate $${est.estimatedUsd}`, now)
  saveLedger(root, plan.id, l)
  const ok = l.instances.length === est.picks.length
  return { ok, instances: l.instances, estimate: est, ...(ok ? {} : { refused: ['not every instance could be created — run stop'] }) }
}

const CRITICAL = ['gpu', 'cuda', 'torch', 'disk', 'startup']
const PREFLIGHT = [
  ['gpu', "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"],
  ['driver', 'nvidia-smi --query-gpu=driver_version --format=csv,noheader'],
  ['cuda', 'python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"'],
  ['torch', 'python -c "import torch; print(torch.__version__, torch.version.cuda)"'],
  ['cpu-ram', 'nproc && free -g | head -2'],
  ['disk', 'df -BG --output=avail / | tail -1'],
  ['network', 'curl -s -o /dev/null -w "%{time_total}" https://pypi.org/simple/ || exit 1'],
]

export function preflight(root, plan, client) {
  const l = loadLedger(root, plan.id)
  const live = l.instances.filter(i => !i.destroyed)
  const rows = []
  for (const inst of live) {
    const checks = [...PREFLIGHT, ...(plan.startupCheck ? [['startup', plan.startupCheck]] : []), ...(plan.peerCheck ? [['peers', plan.peerCheck]] : [])]
    for (const [name, cmd] of checks) {
      const r = client.exec(inst.id, cmd, 300)
      rows.push({ instance: inst.id, role: inst.role, check: name, ok: r.code === 0, critical: CRITICAL.includes(name) || name === 'peers', out: String(r.out).trim().slice(0, 300) })
    }
  }
  const critical = rows.filter(r => !r.ok && r.critical)
  l.preflight = { ok: !critical.length && live.length > 0, rows }
  saveLedger(root, plan.id, l)
  return { ok: l.preflight.ok, verdict: l.preflight.ok ? 'PREFLIGHT PASSED' : 'DO NOT START TRAINING', failed: critical.map(r => `${r.instance}:${r.check}`), rows }
}

export function start(root, plan, client) {
  const l = loadLedger(root, plan.id)
  if (!l.preflight?.ok) return { ok: false, refused: ['pre-flight has not passed — DO NOT START TRAINING'] }
  const results = l.instances.filter(i => !i.destroyed).map(i => {
    const setup = plan.setup ? client.exec(i.id, plan.setup, 3600) : { code: 0 }
    const run = setup.code === 0 ? client.exec(i.id, `nohup sh -c ${JSON.stringify(plan.run)} > /workspace/run.log 2>&1 &`, 60) : setup
    return { instance: i.id, ok: setup.code === 0 && run.code === 0 }
  })
  l.started = results.every(r => r.ok)
  saveLedger(root, plan.id, l)
  return { ok: l.started, results }
}

// One monitoring pass: cost, instance health and the plan's metrics (JSON lines with step, loss, restarts, ts).
export function monitorOnce(root, plan, client, now = Date.now()) {
  const l = loadLedger(root, plan.id)
  const live = l.instances.filter(i => !i.destroyed)
  const listed = client.listInstances()
  const spent = Math.round(live.reduce((s, i) => s + i.dph * ((now - i.createdAt) / 3600000), 0) * 100) / 100
  const pct = (100 * spent) / plan.maxBudgetUsd
  const stops = []
  for (const i of live) {
    const st = listed.find(x => String(x.id) === String(i.id))
    if (!st) stops.push(`instance ${i.id} vanished (terminated by the host?)`)
    else if (!['running', 'loading'].includes(st.actual_status)) stops.push(`instance ${i.id} is ${st.actual_status}`)
  }
  const rule = plan.stop ?? {}
  const metrics = []
  for (const i of live) {
    const r = client.exec(i.id, plan.metrics, 60)
    for (const line of String(r.out).split('\n')) { try { metrics.push({ instance: i.id, ...JSON.parse(line) }) } catch { /* not a metrics line */ } }
  }
  const last = metrics.at(-1)
  if (metrics.some(m => m.loss !== undefined && !Number.isFinite(Number(m.loss)))) stops.push('training divergence: loss is NaN/Inf')
  if (rule.maxLoss !== undefined && metrics.some(m => Number(m.loss) > rule.maxLoss)) stops.push(`training divergence: loss above ${rule.maxLoss}`)
  if (rule.maxRestarts !== undefined && metrics.some(m => Number(m.restarts) > rule.maxRestarts)) stops.push(`a worker crashed more than ${rule.maxRestarts} times`)
  if (rule.stallMinutes && l.started && last?.ts && (now - Date.parse(last.ts)) / 60000 > rule.stallMinutes) stops.push(`no training progress for more than ${rule.stallMinutes} minutes (stalled or deadlocked)`)
  if (l.started && !metrics.length && live.length && rule.stallMinutes && (now - Math.min(...live.map(i => i.createdAt))) / 60000 > rule.stallMinutes) stops.push('no metrics from any worker')
  if (pct >= 100) stops.push(`spent $${spent} reached the approved budget $${plan.maxBudgetUsd}`)
  const action = stops.length ? 'emergency-stop' : pct >= plan.warnAtPct ? 'pause' : 'continue'
  event(l, `monitor:${action}`, `$${spent} (${Math.round(pct)}%)${stops.length ? `: ${stops.join('; ')}` : ''}`, now)
  saveLedger(root, plan.id, l)
  return { action, spentUsd: spent, budgetPct: Math.round(pct), stops, lastMetrics: last ?? null, instances: live.length }
}

export function pause(root, plan, client, now = Date.now()) {
  const l = loadLedger(root, plan.id)
  for (const i of l.instances.filter(x => !x.destroyed)) client.stopInstance(i.id)
  l.paused = true
  event(l, 'paused', `at the ${plan.warnAtPct}% budget warning — awaiting a human decision`, now)
  saveLedger(root, plan.id, l)
  return { paused: true }
}

// Preserve logs and metrics FIRST, then destroy every instance.
export function stop(root, plan, client, reason, now = Date.now()) {
  const l = loadLedger(root, plan.id)
  const dir = join(root, '.factory', 'vast', plan.id, 'logs')
  mkdirSync(dir, { recursive: true })
  for (const i of l.instances.filter(x => !x.destroyed)) {
    for (const [name, cmd] of [['run.log', 'tail -n 5000 /workspace/run.log'], ['metrics.jsonl', plan.metrics], ['nvidia-smi.txt', 'nvidia-smi']]) {
      const r = client.exec(i.id, cmd, 120)
      writeFileSync(join(dir, `${i.id}-${name}`), String(r.out))
    }
    client.destroyInstance(i.id)
    i.destroyed = now
  }
  event(l, 'stopped', reason, now)
  l.stopReason = reason
  saveLedger(root, plan.id, l)
  const remaining = client.listInstances().filter(x => l.instances.some(i => String(i.id) === String(x.id)))
  return { ok: remaining.length === 0, reason, logs: dir, remaining: remaining.map(x => x.id) }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [cmd, ...rest] = process.argv.slice(2)
  const opt = name => { const i = rest.indexOf(name); return i >= 0 ? rest[i + 1] : undefined }
  const out = (o, code = 0) => { console.log(JSON.stringify(o, null, rest.includes('--json') ? 0 : 2)); process.exit(code) }
  try {
    const off = hardwareRefusal(cmd, readHardware(ROOT))
    if (off) out({ ok: false, refused: [off] }, 1)
    const client = clientFromEnv()
    if (cmd === 'doctor') {
      const offers = client.searchOffers('rentable=true num_gpus=1')
      out({ ok: true, client: client.kind, sampleOffers: offers.length })
    }
    const planPath = opt('--plan')
    if (!planPath || planPath.includes('..') || planPath.startsWith('/')) throw new Error('--plan <repo-relative PLAN.json> is required')
    const plan = JSON.parse(readFileSync(join(ROOT, planPath), 'utf8'))
    const errors = validatePlan(plan)
    if (cmd === 'validate') out({ ok: !errors.length, errors }, errors.length ? 1 : 0)
    if (errors.length) out({ ok: false, errors }, 1)
    if (cmd === 'estimate') { const e = estimate(plan, client); out(e, e.ok ? 0 : 1) }
    if (cmd === 'launch') { const r = launch(ROOT, planPath, plan, client); out(r, r.ok ? 0 : 1) }
    if (cmd === 'preflight') { const r = preflight(ROOT, plan, client); out(r, r.ok ? 0 : 1) }
    if (cmd === 'start') { const r = start(ROOT, plan, client); out(r, r.ok ? 0 : 1) }
    if (cmd === 'monitor') {
      const r = monitorOnce(ROOT, plan, client)
      if (r.action === 'emergency-stop') out({ ...r, stopped: stop(ROOT, plan, client, `emergency stop: ${r.stops.join('; ')}`) }, 1)
      if (r.action === 'pause') out({ ...r, ...pause(ROOT, plan, client) }, 3)
      out(r, 0)
    }
    if (cmd === 'stop') { const r = stop(ROOT, plan, client, opt('--reason') ?? 'stopped by request'); out(r, r.ok ? 0 : 1) }
    if (cmd === 'status') { const l = loadLedger(ROOT, plan.id); const live = l.instances.filter(i => !i.destroyed); out({ live: live.length, instances: l.instances, events: l.events.slice(-20) }, live.length ? 1 : 0) }
    throw new Error('usage: vast.mjs validate|estimate|launch|preflight|start|monitor|stop|status --plan <PLAN.json> | doctor')
  } catch (e) {
    console.error(`vast: ${e.message}`)
    process.exit(2)
  }
}
