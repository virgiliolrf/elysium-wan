#!/usr/bin/env node
// Writes machine records that workflow scripts compute (the script cannot write files; an agent that retyped the
// numbers could change them). The payload travels as one base64 token produced by the script; only these targets
// are writable:
//   audit-results   -> docs/security/AUDIT-RESULTS.json (replaced)
//   findings        -> agents-architetura/docs/findings.json        (merged by id: new findings added, status updates applied)
//   campaign <id>   -> agents-architetura/docs/campaigns/<id>/RESULTS.json (replaced)
//   experiment-local <id>    -> docs/experiments/<id>/LOCAL.json (replaced: the promotion record vast.mjs launch checks)
//   experiment-results <id>  -> docs/experiments/<id>/RESULTS.json (replaced)
//
//   node agents-architetura/tools/campaign/record.mjs <target> [<id>] --b64 <payload>
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')

export function targetPath(root, target, id) {
  if (target === 'audit-results') return join(root, 'docs', 'security', 'AUDIT-RESULTS.json')
  if (target === 'findings') return join(root, 'agents-architetura', 'docs', 'findings.json')
  if (target === 'campaign' && /^[a-z0-9][a-z0-9-]{0,40}$/.test(id ?? '')) return join(root, 'agents-architetura', 'docs', 'campaigns', id, 'RESULTS.json')
  if (target === 'experiment-local' && /^[a-z0-9][a-z0-9-]{1,48}$/.test(id ?? '')) return join(root, 'docs', 'experiments', id, 'LOCAL.json')
  if (target === 'experiment-results' && /^[a-z0-9][a-z0-9-]{1,48}$/.test(id ?? '')) return join(root, 'docs', 'experiments', id, 'RESULTS.json')
  throw new Error(`unknown record target ${JSON.stringify(target)}${id ? ` ${JSON.stringify(id)}` : ''}`)
}

export function mergeFindings(existing, incoming) {
  const byId = new Map((existing?.findings ?? []).map(f => [String(f.id), f]))
  for (const f of incoming.findings ?? []) {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(String(f.id ?? ''))) throw new Error(`bad finding id ${JSON.stringify(f.id)}`)
    if (!['open', 'closed', 'needs-human'].includes(f.status)) throw new Error(`finding ${f.id}: status must be open, closed or needs-human`)
    byId.set(String(f.id), { ...(byId.get(String(f.id)) ?? {}), ...f })
  }
  return { findings: [...byId.values()].sort((a, b) => String(a.id).localeCompare(String(b.id))) }
}

export function record(root, target, id, payload) {
  const p = targetPath(root, target, id)
  mkdirSync(dirname(p), { recursive: true })
  const value = target === 'findings' ? mergeFindings(existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : null, payload) : payload
  writeFileSync(p, JSON.stringify(value, null, 2) + '\n')
  return p
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const i = argv.indexOf('--b64')
  try {
    if (i < 0) throw new Error('usage: record.mjs <target> [<id>] --b64 <payload>')
    const target = argv[0]
    const id = ['campaign', 'experiment-local', 'experiment-results'].includes(target) ? argv[1] : undefined
    const payload = JSON.parse(Buffer.from(argv[i + 1], 'base64').toString('utf8'))
    console.log(JSON.stringify({ written: record(resolve(ROOT), target, id, payload) }))
  } catch (e) {
    console.error(`record: ${e.message}`)
    process.exit(2)
  }
}
