#!/usr/bin/env node
// Validity check for a comparative scalability study (bench/scalebench/results.json). The reference PR invalidated any
// run whose checksum differed across languages or thread counts; here every system and node count must have trained
// on the same data order from the same initial weights, or the comparison is meaningless. Byte-exact rule (CLAUDE.md,
// "Product rule"): every run of our system must also end in the same training state (checksums.finalState) at every node
// count, profile and repetition, equal to the synchronous reference's; baselines need not match.
// Discriminating arms (the reference PR's `flat` arm, which split threading cost from object-model cost): a gap is
// attributed, not only observed. Every arm is a variant of one system that changes exactly ONE factor; the attribution
// says how much of the gap each arm explains, summing to 100% ± 10 with a named residual.
//   results.json: {"ours": system, "reference": system, "runs": [{"system", "nodes", "rep", "arm"?,
//                  "checksums": {"dataOrder", "initWeights", "finalState"}, "metrics": {...}}],
//                  "arms": [{"id", "of": system, "varies": "<one factor>"}],
//                  "attribution": [{"arm": id | "residual", "factor", "sharePct"}]}
//
//   node agents-architetura/tools/gates/scalebench-check.mjs [--results PATH] [--json]
// Exit: 0 valid · 1 invalid · 5 not applicable (no results file).
import { existsSync, readFileSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')

export function checkScalebench(results) {
  const runs = results?.runs ?? []
  const problems = []
  if (!runs.length) problems.push('no runs recorded')
  const systems = new Set(runs.map(r => r.system))
  if (runs.length && systems.size < 2) problems.push('a comparison needs at least two systems (ours and a baseline)')
  for (const key of ['dataOrder', 'initWeights']) {
    const values = new Map()
    for (const r of runs) {
      const v = r?.checksums?.[key]
      if (typeof v !== 'string' || !v) { problems.push(`${r.system}@${r.nodes} rep ${r.rep}: missing checksum ${key}`); continue }
      values.set(v, [...(values.get(v) ?? []), `${r.system}@${r.nodes}#${r.rep}`])
    }
    if (values.size > 1) problems.push(`${key} differs across runs: ${[...values].map(([v, who]) => `${v.slice(0, 12)} (${who.slice(0, 4).join(', ')}${who.length > 4 ? ', ...' : ''})`).join(' vs ')}`)
  }
  const ours = results?.ours
  if (typeof ours !== 'string' || !systems.has(ours)) problems.push(`results.ours must name the system under test among the runs (got ${JSON.stringify(ours ?? null)})`)
  else {
    const reference = results?.reference
    if (reference !== undefined && (typeof reference !== 'string' || !systems.has(reference))) problems.push(`results.reference names no system among the runs (got ${JSON.stringify(reference)})`)
    const exact = new Map()
    for (const r of runs.filter(r => r.system === ours || (r.system === reference && reference !== ours))) {
      const v = r?.checksums?.finalState
      if (typeof v !== 'string' || !v) { problems.push(`${r.system}@${r.nodes} rep ${r.rep}: missing checksum finalState (byte-exact rule)`); continue }
      exact.set(v, [...(exact.get(v) ?? []), `${r.system}@${r.nodes}#${r.rep}`])
    }
    if (exact.size > 1) problems.push(`finalState is not byte-identical across ${ours}${reference ? ` and ${reference}` : ''} runs (byte-exact rule): ${[...exact].map(([v, who]) => `${v.slice(0, 12)} (${who.slice(0, 4).join(', ')}${who.length > 4 ? ', ...' : ''})`).join(' vs ')}`)
  }
  problems.push(...checkArms(results, systems))
  return { ok: !problems.length, runs: runs.length, systems: [...systems], problems }
}

export function checkArms(results, systems = new Set((results?.runs ?? []).map(r => r.system))) {
  const problems = []
  const runs = results?.runs ?? []
  const arms = Array.isArray(results?.arms) ? results.arms : []
  if (!arms.length) problems.push('no discriminating arms: the gap is observed, not attributed (results.arms: variants that change one factor each)')
  const ids = new Set()
  for (const a of arms) {
    if (typeof a?.id !== 'string' || !a.id || ids.has(a.id) || a.id === 'residual') problems.push(`arm ${JSON.stringify(a?.id ?? null)}: missing, reserved or duplicate id`)
    ids.add(a?.id)
    if (typeof a?.varies !== 'string' || !a.varies.trim() || /,|;|\band\b|\+/.test(a.varies)) problems.push(`arm ${a?.id}: "varies" must name exactly one factor (got ${JSON.stringify(a?.varies ?? null)})`)
    if (!systems.has(a?.of)) problems.push(`arm ${a?.id}: "of" names no system among the runs`)
    if (!runs.some(r => r.arm === a?.id)) problems.push(`arm ${a?.id}: no runs`)
  }
  for (const r of runs) if (r.arm !== undefined && !ids.has(r.arm)) problems.push(`${r.system}@${r.nodes} rep ${r.rep}: unknown arm ${JSON.stringify(r.arm)}`)
  if (arms.length) {
    const attribution = Array.isArray(results?.attribution) ? results.attribution : []
    if (!attribution.length) problems.push('no attribution: say how much of the gap each arm explains (results.attribution)')
    for (const a of attribution) {
      if (a?.arm !== 'residual' && !ids.has(a?.arm)) problems.push(`attribution names unknown arm ${JSON.stringify(a?.arm ?? null)}`)
      if (!Number.isFinite(a?.sharePct)) problems.push(`attribution for ${a?.arm}: sharePct must be a number`)
    }
    const total = attribution.reduce((s, a) => s + (Number.isFinite(a?.sharePct) ? a.sharePct : 0), 0)
    if (attribution.length && !(total >= 90 && total <= 110)) problems.push(`attribution shares sum to ${total}%, not 100% ± 10 (name what is left as {"arm": "residual"})`)
  }
  return problems
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const i = argv.indexOf('--results')
  const path = resolve(i >= 0 ? argv[i + 1] : join(ROOT, 'bench', 'scalebench', 'results.json'))
  if (!existsSync(path)) { console.log(argv.includes('--json') ? JSON.stringify({ ok: false, applicable: false }) : 'scalebench: n/a (no results)'); process.exit(5) }
  const r = checkScalebench(JSON.parse(readFileSync(path, 'utf8')))
  console.log(argv.includes('--json') ? JSON.stringify(r) : `${r.ok ? 'VALID' : 'INVALID'}: ${r.runs} runs, systems ${r.systems.join(', ')}${r.problems.length ? `\n  ${r.problems.join('\n  ')}` : ''}`)
  process.exit(r.ok ? 0 : 1)
}
