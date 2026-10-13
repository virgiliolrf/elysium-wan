#!/usr/bin/env node
// Scanner battery (the PR's thread-scanners). Each scanner in agents-architetura/tools/campaign/scanners.json runs once, under the
// machine lock, writing its raw report into $FACTORY_SCAN_OUT (.factory/scan/<id>/); reports are parsed (SARIF,
// mypy-style diagnostics, junit, sanitizer/traceback text), deduped into families and filtered through
// agents-architetura/tools/campaign/suppressions/scan-<id>.txt. A scanner that cannot run on this machine is reported
// SKIPPED-UNAVAILABLE with the reason — loudly, never silently dropped. A scanner that fails without producing a
// parsable report is BROKEN.
//
//   node agents-architetura/tools/campaign/scan.mjs [--only id,id] [--json] [--root DIR]
// Exit: 0 no unsuppressed finding · 1 findings or broken scanners · 2 usage · 5 not applicable (no scanner applies).
import { existsSync, readFileSync, mkdirSync, readdirSync, rmSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseFile } from './parse.mjs'
import { groupFamilies, applySuppressions } from './families.mjs'
import { loadSuppressions } from './suppressions.mjs'
import { runSync } from '../verify/exec.mjs'
import { withMachineLock } from '../verify/machine-lock.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')

export function runScanners(root, { only = null, exec = runSync, lock = (label, fn) => withMachineLock(root, label, fn, { waitSec: 3600 }) } = {}) {
  const cfgPath = join(root, 'agents-architetura', 'tools', 'campaign', 'scanners.json')
  const scanners = (existsSync(cfgPath) ? JSON.parse(readFileSync(cfgPath, 'utf8')).scanners ?? [] : []).filter(s => !only || only.includes(s.id))
  const results = []
  for (const s of scanners) {
    if (!/^[a-z0-9][a-z0-9-]{0,40}$/.test(s.id ?? '')) throw new Error(`scanners.json: bad id ${JSON.stringify(s.id)}`)
    const absent = (s.when ?? []).filter(p => !existsSync(join(root, p)))
    if (absent.length) { results.push({ id: s.id, status: 'n/a', reason: `${absent.join(', ')} absent` }); continue }
    if (s.probe) {
      const p = exec(s.probe, { cwd: root, timeoutSec: 120 })
      if (p.outcome !== 'pass') { results.push({ id: s.id, status: 'skipped-unavailable', reason: `probe "${s.probe}" failed: ${`${p.stdout}${p.stderr}`.trim().slice(-200)}` }); continue }
    }
    const out = join(root, '.factory', 'scan', s.id)
    rmSync(out, { recursive: true, force: true })
    mkdirSync(out, { recursive: true })
    const r = lock(`scan:${s.id}`, () => exec(s.cmd, { cwd: root, env: { FACTORY_SCAN_OUT: out }, timeoutSec: s.timeoutSec ?? 7200 }))
    const files = readdirSync(out).sort()
    const reports = files.flatMap(f => parseFile(join(out, f), readFileSync(join(out, f), 'utf8')))
    const { entries, violations } = loadSuppressions(root, `agents-architetura/tools/campaign/suppressions/scan-${s.id}.txt`)
    const { kept, suppressed } = applySuppressions(groupFamilies(reports), entries)
    const broken = r.outcome !== 'pass' && reports.length === 0
    results.push({
      id: s.id,
      status: broken ? 'broken' : 'ran',
      reason: broken ? `exit ${r.status}, no parsable report: ${`${r.stdout}${r.stderr}`.slice(-600)}` : undefined,
      exit: r.status,
      findings: kept.reduce((n, f) => n + f.count, 0),
      families: kept.map(f => ({ id: f.id, count: f.count, signature: f.signature, locations: f.locations, message: f.sample.message.slice(0, 300) })),
      suppressed: suppressed.length,
      suppressionViolations: violations,
    })
  }
  const ran = results.filter(r => r.status === 'ran' || r.status === 'broken')
  const findings = ran.reduce((n, r) => n + (r.findings ?? 0), 0)
  const bad = ran.filter(r => r.status === 'broken' || r.findings || r.suppressionViolations?.length)
  return { applicable: ran.length > 0, ok: bad.length === 0, findings, unavailable: results.filter(r => r.status === 'skipped-unavailable').map(r => r.id), scanners: results }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const opt = name => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined }
  let r
  try {
    r = runScanners(resolve(opt('--root') ?? ROOT), { only: opt('--only')?.split(',').filter(Boolean) ?? null })
  } catch (e) {
    console.error(`scan: ${e.message}`)
    process.exit(2)
  }
  if (argv.includes('--json')) console.log(JSON.stringify(r))
  else {
    for (const s of r.scanners) console.log(`${s.status.toUpperCase().padEnd(20)} ${s.id}${s.status === 'ran' ? `  ${s.findings} findings in ${s.families.length} families` : `  ${s.reason}`}`)
    if (r.unavailable.length) console.log(`UNAVAILABLE ON THIS MACHINE: ${r.unavailable.join(', ')}`)
  }
  process.exit(!r.applicable ? 5 : r.ok ? 0 : 1)
}
