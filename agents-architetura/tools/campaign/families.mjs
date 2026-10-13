#!/usr/bin/env node
// Campaign rounds (the PR's TSAN campaign mechanics, for any report-producing tool). One round:
//   1. (--run) run the campaign command ONCE under the machine lock; every report goes to the round directory
//      (the command can write sanitizer logs to $FACTORY_CAMPAIGN_REPORTS; stdout/stderr land in run.log)
//   2. parse every file in the round directory into reports, dedupe by signature into families
//   3. apply the campaign's justified suppressions
//   4. compare with the previous round: new / fixed / persisting families
// A round whose command failed while producing no parsable report is BROKEN, never "zero". A campaign may declare a
// "sentinel" regex its run output must contain (e.g. "GIL False" for the free-threaded lane): a round that does not
// prove its instrumentation was active is BROKEN too — a clean run of an uninstrumented build proves nothing.
//
// agents-architetura/tools/campaign/campaigns.json: {"campaigns": [{"id", "kind", "cmd", "env", "timeoutSec", "suppressions", "when"}]}
//
//   node agents-architetura/tools/campaign/families.mjs --campaign <id> --round <n> [--run] [--json] [--root DIR]
// Exit: 0 zero unsuppressed · 1 reports remain (or broken round) · 2 usage · 5 not applicable (campaign's `when` paths absent).
import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync, statSync, rmSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseFile, shortHash } from './parse.mjs'
import { loadSuppressions } from './suppressions.mjs'
import { runSync } from '../verify/exec.mjs'
import { withMachineLock } from '../verify/machine-lock.mjs'
import { treeId } from '../gates/tree.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const ID_RE = /^[a-z0-9][a-z0-9-]{0,40}$/

export function loadCampaigns(root) {
  const p = join(root, 'agents-architetura', 'tools', 'campaign', 'campaigns.json')
  const list = existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')).campaigns ?? [] : []
  for (const c of list) {
    if (!ID_RE.test(c.id ?? '')) throw new Error(`campaigns.json: bad id ${JSON.stringify(c.id)}`)
    if (!['sanitizer', 'amplify', 'load', 'scanner', 'fuzz'].includes(c.kind)) throw new Error(`campaigns.json: ${c.id} has unknown kind ${JSON.stringify(c.kind)}`)
  }
  return list
}

export const roundDir = (root, id, round) => join(root, '.factory', 'campaigns', id, `r${round}`)

export function groupFamilies(reports) {
  const byKey = new Map()
  for (const r of reports) {
    const f = byKey.get(r.signature) ?? { id: `${r.kind.replace(/[^a-z0-9]+/gi, '-').toLowerCase().slice(0, 32)}-${shortHash(r.signature)}`, signature: r.signature, kind: r.kind, count: 0, locations: new Set(), sample: r }
    f.count++
    if (r.location) f.locations.add(r.location)
    byKey.set(r.signature, f)
  }
  return [...byKey.values()].map(f => ({ ...f, locations: [...f.locations].slice(0, 10) })).sort((a, b) => b.count - a.count || a.id.localeCompare(b.id))
}

export function applySuppressions(families, entries) {
  const kept = []
  const suppressed = []
  for (const f of families) {
    const rule = entries.find(e => f.signature.includes(e.pattern))
    if (rule) suppressed.push({ id: f.id, count: f.count, signature: f.signature, rule: rule.where })
    else kept.push(f)
  }
  return { kept, suppressed }
}

export function diffRounds(prev, cur) {
  const before = new Set((prev ?? []).map(f => f.signature))
  const now = new Set(cur.map(f => f.signature))
  return {
    new: cur.filter(f => !before.has(f.signature)).map(f => f.id),
    fixed: (prev ?? []).filter(f => !now.has(f.signature)).map(f => f.id),
    persisting: cur.filter(f => before.has(f.signature)).map(f => f.id),
  }
}

// Driver campaigns (amplify, load) summarize every finding in summary.json; their per-seed logs repeat it.
const DEFAULT_REPORTS = { amplify: ['summary\\.json$'], load: ['summary\\.json$'] }

function readReports(dir, patterns) {
  if (!existsSync(dir)) return []
  const files = []
  const walk = d => { for (const e of readdirSync(d)) { const p = join(d, e); if (statSync(p).isDirectory()) walk(p); else files.push(p) } }
  walk(dir)
  const res = (patterns ?? []).map(p => new RegExp(p))
  return files.sort().filter(p => !res.length || res.some(re => re.test(p))).flatMap(p => parseFile(p, readFileSync(p, 'utf8')))
}

// Whether a campaign can run here: its when-paths exist and its probe passes. Cheap (a probe is capped at 120 s), so the
// verdict gate checks it before its hours of rungs.
export function campaignApplicable(root, campaign, { exec = runSync } = {}) {
  if ((campaign.when ?? []).some(p => !existsSync(join(root, p)))) return { applicable: false, reason: `campaign ${campaign.id}: ${campaign.when.filter(p => !existsSync(join(root, p))).join(', ')} absent` }
  if (campaign.probe) {
    const p = exec(campaign.probe, { cwd: root, timeoutSec: 120 })
    if (p.outcome !== 'pass') return { applicable: false, reason: `campaign ${campaign.id}: probe failed — tool unavailable on this machine (${`${p.stdout}${p.stderr}`.trim().slice(-200)})` }
  }
  return { applicable: true }
}

export function runRound(root, campaign, round, { run = false, exec = runSync, lock = (label, fn) => withMachineLock(root, label, fn, { waitSec: 3600 }) } = {}) {
  const can = campaignApplicable(root, campaign, { exec })
  if (!can.applicable) return can
  const dir = roundDir(root, campaign.id, round)
  let commandExit = null
  let tail = ''
  if (run) {
    rmSync(dir, { recursive: true, force: true })
    mkdirSync(dir, { recursive: true })
    const r = lock(`campaign:${campaign.id}`, () => exec(campaign.cmd, { cwd: root, env: { ...(campaign.env ?? {}), FACTORY_CAMPAIGN_REPORTS: dir }, timeoutSec: campaign.timeoutSec ?? 14400 }))
    writeFileSync(join(dir, 'run.log'), `${r.stdout}${r.stderr}`)
    commandExit = r.status
    tail = `${r.stdout}${r.stderr}`.slice(-2000)
    if (r.outcome === 'hang') commandExit = 'timeout'
  }
  const reports = readReports(dir, campaign.reports ?? DEFAULT_REPORTS[campaign.kind])
  const { entries, violations } = loadSuppressions(root, campaign.suppressions)
  const { kept, suppressed } = applySuppressions(groupFamilies(reports), entries)
  const prevPath = join(root, '.factory', 'campaigns', campaign.id, `families-r${round - 1}.json`)
  const prev = existsSync(prevPath) ? JSON.parse(readFileSync(prevPath, 'utf8')).families : null
  const sentinelMissing = run && campaign.sentinel && !new RegExp(campaign.sentinel).test(readFileSync(join(dir, 'run.log'), 'utf8'))
  const broken = (run && commandExit !== 0 && reports.length === 0) || !!sentinelMissing
  const result = {
    applicable: true,
    campaign: campaign.id,
    round,
    tree: treeId(root),
    commandExit,
    broken,
    brokenReason: !broken ? null : sentinelMissing
      ? `the run output lacks the sentinel /${campaign.sentinel}/ — the instrumentation was not proven active, so a clean result proves nothing`
      : `the campaign command ended with ${commandExit} and produced no parsable report — fix the run, a broken round is not zero: ${tail.slice(-600)}`,
    unsuppressedReports: kept.reduce((n, f) => n + f.count, 0),
    families: kept.map(f => ({ id: f.id, kind: f.kind, count: f.count, signature: f.signature, locations: f.locations, message: f.sample.message.slice(0, 300), frames: f.sample.frames.slice(0, 6) })),
    suppressed,
    suppressionViolations: violations,
    delta: diffRounds(prev, kept),
  }
  mkdirSync(join(root, '.factory', 'campaigns', campaign.id), { recursive: true })
  writeFileSync(join(root, '.factory', 'campaigns', campaign.id, `families-r${round}.json`), JSON.stringify(result, null, 2))
  return result
}

export const roundClean = r => r.applicable && !r.broken && r.unsuppressedReports === 0 && !r.suppressionViolations.length

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const opt = name => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined }
  let r
  try {
    const root = resolve(opt('--root') ?? ROOT)
    const id = opt('--campaign')
    const round = Number(opt('--round'))
    if (!Number.isInteger(round) || round < 0 || round > 999) throw new Error('--round <n> (0..999) is required')
    const campaign = loadCampaigns(root).find(c => c.id === id)
    if (!campaign) throw new Error(`unknown campaign ${JSON.stringify(id)} (see agents-architetura/tools/campaign/campaigns.json)`)
    r = runRound(root, campaign, round, { run: argv.includes('--run') })
    if (argv.includes('--json')) console.log(JSON.stringify(r))
    else if (!r.applicable) console.log(`campaign: n/a (${r.reason})`)
    else {
      for (const f of r.families) console.log(`${String(f.count).padStart(6)}  ${f.id}  ${f.locations[0] ?? ''}`)
      for (const v of r.suppressionViolations) console.log(`SUPPRESSION ${v}`)
      if (r.broken) console.log(`BROKEN ${r.brokenReason}`)
      console.log(`campaign ${r.campaign} r${r.round}: ${r.unsuppressedReports} unsuppressed reports in ${r.families.length} families (${r.suppressed.length} suppressed families); new ${r.delta.new.length}, fixed ${r.delta.fixed.length}`)
    }
  } catch (e) {
    console.error(`campaign: ${e.message}`)
    process.exit(2)
  }
  process.exit(!r.applicable ? 5 : roundClean(r) ? 0 : 1)
}
