#!/usr/bin/env node
// Was the committed version of a document granted by a human? `approve.mjs check` hashes the working copy. An ADR in
// progress (wan-adr revise or split) has already written its pending amendments into DESIGN.md, so its working copy is
// unapproved by construction. This checks the version at a git revision (default HEAD): the base the ADR started from.
// The pending amendments are granted at the end, together with the ADR. Read-only; prints the shape of
// `approve.mjs check --json`.
//
//   node agents-architetura/tools/factory/approved-base.mjs <doc>... [--rev HEAD] [--json]
// Exit: 0 every document approved at that revision · 1 not · 2 usage.
import { spawnSync } from 'node:child_process'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { sha256, loadStore } from './approve.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const DOC_RE = /^[\w./-]+$/
const REV_RE = /^[\w./~^-]{1,100}$/

const gitShow = (root, rev, doc) => {
  const r = spawnSync('git', ['show', `${rev}:${doc}`], { cwd: root, maxBuffer: 64 * 1024 * 1024 })
  return r.status === 0 ? r.stdout : null
}

export function checkApprovedAt(root, docs, rev = 'HEAD', show = gitShow) {
  if (!REV_RE.test(rev) || rev.startsWith('-')) throw new Error(`not a git revision: ${JSON.stringify(rev)}`)
  const store = loadStore(root)
  const out = docs.map(doc => {
    if (!DOC_RE.test(doc) || doc.includes('..') || doc.startsWith('/')) throw new Error(`not a repo-relative document path: ${JSON.stringify(doc)}`)
    const text = show(root, rev, doc)
    if (text === null) return { doc, rev, approved: false, reason: `${doc} does not exist at ${rev}` }
    const sha = sha256(text)
    const match = store.approvals.find(a => a.doc === doc && a.sha256 === sha && !a.revoked)
    return match
      ? { doc, rev, approved: true, sha256: sha, approvedBy: match.approvedBy, at: match.at }
      : { doc, rev, approved: false, sha256: sha, reason: `the version at ${rev} was never granted by a human` }
  })
  return { approved: out.every(d => d.approved), docs: out }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const i = argv.indexOf('--rev')
  const rev = i >= 0 ? argv[i + 1] : 'HEAD'
  const docs = argv.filter((a, k) => !a.startsWith('--') && !(i >= 0 && k === i + 1))
  if (!docs.length) {
    console.error('usage: approved-base.mjs <doc>... [--rev HEAD] [--json]')
    process.exit(2)
  }
  let r
  try { r = checkApprovedAt(ROOT, docs, rev) } catch (e) { console.error(`approved-base: ${e.message}`); process.exit(2) }
  if (argv.includes('--json')) console.log(JSON.stringify(r))
  else for (const d of r.docs) console.log(`${d.approved ? 'APPROVED' : 'NOT APPROVED'} ${d.doc} at ${d.rev}${d.reason ? ` — ${d.reason}` : ''}`)
  process.exit(r.approved ? 0 : 1)
}
