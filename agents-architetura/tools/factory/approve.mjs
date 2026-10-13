#!/usr/bin/env node
// Human approval of architecture decisions. Silence is not approval: a decision document (the design of record,
// an ADR, a paid-experiment plan) counts as approved only while docs/design/APPROVALS.json holds a grant for its
// EXACT current content (sha256). Editing an approved document voids its approval; nothing downstream proceeds
// until a human grants the new text.
//
// Granting is a human decision, taken one of two ways:
// - In the human's own terminal: `grant` needs an interactive terminal and the human types APPROVE. Agents are
//   refused (CLAUDECODE is set in their sessions) and denied the command by .claude/settings.json.
// - Inside Claude Code: the agent runs `confirm <doc...> --sha <sha12,...>`, which writes nothing. It matches an
//   explicit ASK rule, and Claude Code puts ask-rule matches to the human in every permission mode, bypassPermissions
//   included; background agents cannot show the prompt and are denied. Only after the human allows it does the
//   command run, and only then does the PostToolUse hook approval-record.mjs record the grant, for that exact
//   command. Any other shape of the command, or a denial, records nothing: the failure mode is "not approved".
//   --sha binds the grant to the text shown; a document edited since is refused.
// Agents may otherwise only `check`, `pending` and `list`. (Defense in depth: .claude/settings.json denies direct edits
// of APPROVALS.json.)
//
//   node agents-architetura/tools/factory/approve.mjs check <doc...> [--json]     exit 0 approved · 1 not approved · 2 usage
//   node agents-architetura/tools/factory/approve.mjs pending [--json]            decision documents awaiting a grant
//   node agents-architetura/tools/factory/approve.mjs list [--json]
//   node agents-architetura/tools/factory/approve.mjs grant <doc...> [--note "..."]   HUMAN ONLY (terminal)
//   node agents-architetura/tools/factory/approve.mjs confirm <doc...> --sha <sha12,...>   from Claude Code: the ask prompt, then the hook records
// With the autonomy switch off for grants (autonomy.json "humanGrants": false), check waives every grant except a paid
// experiment plan's, and pending lists only what still needs a human.
import { existsSync, readFileSync, writeFileSync, readdirSync, mkdirSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { join, resolve, dirname, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { createInterface } from 'node:readline'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
export const STORE = 'docs/design/APPROVALS.json'
const DOC_RE = /^[\w./-]+$/

export const sha256 = text => createHash('sha256').update(text).digest('hex')
export const loadStore = root => (existsSync(join(root, STORE)) ? JSON.parse(readFileSync(join(root, STORE), 'utf8')) : { approvals: [] })

const checkDocPath = doc => {
  if (!DOC_RE.test(doc) || doc.includes('..') || doc.startsWith('/')) throw new Error(`not a repo-relative document path: ${JSON.stringify(doc)}`)
  return doc
}

// The human's autonomy switch (agents-architetura/tools/factory/autonomy.json; agents are denied edits of it). With
// "humanGrants": false (the human's decision, 2026-10-09, like the reference PR) a document needs no human grant: check
// reports it approved and says it was waived, and nothing is ever written to APPROVALS.json in the human's name. A paid
// experiment plan (cloud spend or a budget) always needs a real human grant, and so does any strict check (vast.mjs).
export const AUTONOMY = 'agents-architetura/tools/factory/autonomy.json'
export function readAutonomy(root) {
  try { const a = JSON.parse(readFileSync(join(root, AUTONOMY), 'utf8')); return { humanGrants: a.humanGrants !== false, at: a.at ?? null } } catch { return { humanGrants: true, at: null } }
}
const spendsMoney = (root, doc) => {
  if (!/^docs\/experiments\/[^/]+\/PLAN\.json$/.test(doc)) return false
  try { const plan = JSON.parse(readFileSync(join(root, doc), 'utf8')); return plan.cloud === true || Number(plan.maxBudgetUsd ?? 0) > 0 } catch { return true }
}

export function checkApproval(root, doc, { strict = false } = {}) {
  checkDocPath(doc)
  const p = join(root, doc)
  if (!existsSync(p)) return { doc, approved: false, reason: 'document does not exist' }
  const current = sha256(readFileSync(p))
  const auto = readAutonomy(root)
  if (!strict && !auto.humanGrants && !spendsMoney(root, doc) && !loadStore(root).approvals.some(a => a.doc === doc && a.sha256 === current && !a.revoked)) {
    return { doc, approved: true, sha256: current, waived: true, approvedBy: `autonomy (the human waived grants on ${auto.at ?? 'record'}; no human grant)`, at: auto.at }
  }
  const grants = loadStore(root).approvals.filter(a => a.doc === doc)
  const match = grants.find(a => a.sha256 === current && !a.revoked)
  if (match) return { doc, approved: true, sha256: current, approvedBy: match.approvedBy, at: match.at }
  return { doc, approved: false, sha256: current, reason: grants.length ? 'the document changed after its last approval — a human must approve the new text' : 'never approved — silence is not approval' }
}

// Decision documents that exist and lack a grant for their current content.
export function pendingDecisions(root) {
  const docs = []
  if (existsSync(join(root, 'docs/design/DESIGN.md'))) docs.push('docs/design/DESIGN.md')
  const adr = join(root, 'docs/design/adr')
  if (existsSync(adr)) for (const f of readdirSync(adr).sort()) if (/^ADR-\d{4}.*\.md$/.test(f) && /^Status:\s*PROPOSED\b/m.test(readFileSync(join(adr, f), 'utf8'))) docs.push(`docs/design/adr/${f}`)
  // A phase written by wan-next runs only with a grant of its NEXT document, its pinned gates (the phase's V0..Vn) AND the
  // built workflow that executes it.
  const design = join(root, 'docs/design')
  if (existsSync(design)) {
    for (const f of readdirSync(design).sort()) {
      const m = f.match(/^NEXT-([a-z0-9][a-z0-9-]*)\.md$/)
      if (!m) continue
      docs.push(`docs/design/${f}`)
      if (existsSync(join(design, `GATES-${m[1]}.json`))) docs.push(`docs/design/GATES-${m[1]}.json`)
      if (existsSync(join(root, '.claude/workflows', `wan-x-${m[1]}.js`))) docs.push(`.claude/workflows/wan-x-${m[1]}.js`)
    }
  }
  const exp = join(root, 'docs/experiments')
  if (existsSync(exp)) for (const d of readdirSync(exp).sort()) if (existsSync(join(exp, d, 'PLAN.json'))) docs.push(`docs/experiments/${d}/PLAN.json`)
  return docs.map(doc => checkApproval(root, doc)).filter(c => !c.approved)
}

// The only writer of grants. Exported for tests; the CLI wraps it in the human-only guards.
export function recordGrant(root, docs, { approvedBy, note = '', at = new Date().toISOString() }) {
  const store = loadStore(root)
  const out = []
  for (const doc of docs) {
    checkDocPath(doc)
    if (!existsSync(join(root, doc))) throw new Error(`${doc} does not exist`)
    const rec = { doc, sha256: sha256(readFileSync(join(root, doc))), approvedBy, at, note }
    store.approvals.push(rec)
    out.push(rec)
  }
  mkdirSync(dirname(join(root, STORE)), { recursive: true })
  writeFileSync(join(root, STORE), JSON.stringify(store, null, 2) + '\n')
  return out
}

export function humanGuard(env = process.env, stdin = process.stdin, stdout = process.stdout) {
  if (env.CLAUDECODE) return 'refused: grants are not taken from an agent session — in Claude Code use `approve.mjs confirm <doc> --sha <sha12>` (Claude Code asks the human, then a hook records it), or run grant in your own terminal'
  if (!stdin.isTTY || !stdout.isTTY) return 'refused: granting needs an interactive terminal'
  return null
}

// Each prefix (at least 12 hex characters, one per document, in order) must match that document's current sha256.
export function verifyShown(root, docs, confirm) {
  const prefixes = String(confirm ?? '').split(',').map(s => s.trim().toLowerCase()).filter(Boolean)
  if (!docs.length) throw new Error('grant needs at least one document')
  if (prefixes.length !== docs.length) throw new Error(`--confirm needs one sha256 prefix per document (${docs.length}), got ${prefixes.length}`)
  docs.forEach((doc, i) => {
    checkDocPath(doc)
    if (!existsSync(join(root, doc))) throw new Error(`${doc} does not exist`)
    if (!/^[0-9a-f]{12,64}$/.test(prefixes[i])) throw new Error(`--confirm for ${doc} must be at least 12 hex characters of its sha256`)
    const current = sha256(readFileSync(join(root, doc)))
    if (!current.startsWith(prefixes[i])) throw new Error(`${doc} changed since it was shown (sha256 ${current.slice(0, 12)}, confirmed ${prefixes[i].slice(0, 12)}) — show it again and ask again`)
  })
  return prefixes
}

// Recorded only by the PostToolUse hook, after the human allowed the exact `confirm` command.
export function confirmedGrant(root, docs, confirm, { approvedBy, note = '', at } = {}) {
  verifyShown(root, docs, confirm)
  return recordGrant(root, docs, { approvedBy, note, ...(at ? { at } : {}) })
}

const ask = q => new Promise(res => { const rl = createInterface({ input: process.stdin, output: process.stdout }); rl.question(q, a => { rl.close(); res(a.trim()) }) })

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [cmd, ...rest] = process.argv.slice(2)
  const json = rest.includes('--json')
  const noteAt = rest.indexOf('--note')
  const shaAt = rest.indexOf('--sha')
  const valueAt = new Set([noteAt, shaAt].filter(i => i >= 0).map(i => i + 1))
  // The value after --note is not a document. Without --note, indexOf is -1: nothing is skipped (noteAt + 1 would be 0,
  // the first document, which made `check <doc>` and `grant <doc>` see no document at all).
  const docs = rest.filter((a, i) => !a.startsWith('--') && !valueAt.has(i))
  try {
    if (cmd === 'check') {
      if (!docs.length) throw new Error('check needs at least one document')
      const results = docs.map(d => checkApproval(ROOT, d))
      const approved = results.every(r => r.approved)
      console.log(json ? JSON.stringify({ approved, docs: results }) : results.map(r => `${r.approved ? 'APPROVED' : 'NOT APPROVED'} ${r.doc}${r.reason ? ` — ${r.reason}` : ` (by ${r.approvedBy}, ${r.at})`}`).join('\n'))
      process.exit(approved ? 0 : 1)
    } else if (cmd === 'pending') {
      const p = pendingDecisions(ROOT)
      console.log(json ? JSON.stringify({ pending: p }) : p.length ? p.map(x => `PENDING ${x.doc} — ${x.reason}`).join('\n') : 'no decision awaiting approval')
    } else if (cmd === 'list') {
      const s = loadStore(ROOT)
      console.log(json ? JSON.stringify(s) : s.approvals.map(a => `${a.at} ${a.approvedBy} ${a.doc} ${a.sha256.slice(0, 12)}${a.note ? ` — ${a.note}` : ''}`).join('\n') || 'no approvals')
    } else if (cmd === 'confirm') {
      // Writes nothing. Its ask rule makes Claude Code put it to the human; approval-record.mjs records it afterwards.
      const prefixes = verifyShown(ROOT, docs, shaAt >= 0 ? rest[shaAt + 1] : '')
      docs.forEach((d, i) => console.log(`CONFIRMED ${d} ${prefixes[i].slice(0, 12)}`))
    } else if (cmd === 'grant') {
      const refused = humanGuard()
      if (refused) { console.error(`approve: ${refused}`); process.exit(3) }
      if (!docs.length) throw new Error('grant needs at least one document')
      for (const d of docs) {
        checkDocPath(d)
        if (!existsSync(join(ROOT, d))) throw new Error(`${d} does not exist`)
        console.log(`\n${d}  sha256 ${sha256(readFileSync(join(ROOT, d))).slice(0, 16)}…`)
      }
      console.log('\nRead the documents above in full before approving. Approval binds to this exact text.')
      const typed = await ask(`Type APPROVE to approve ${docs.length} document(s): `)
      if (typed !== 'APPROVE') { console.log('not approved'); process.exit(1) }
      const who = spawnSync('git', ['config', 'user.name'], { cwd: ROOT, encoding: 'utf8' }).stdout.trim() || 'human'
      const note = noteAt >= 0 ? rest[noteAt + 1] ?? '' : ''
      for (const r of recordGrant(ROOT, docs, { approvedBy: who, note })) console.log(`APPROVED ${r.doc} ${r.sha256.slice(0, 12)} by ${r.approvedBy}`)
    } else {
      throw new Error('usage: approve.mjs check <doc...> | pending | list | grant <doc...> [--note "..."] | confirm <doc...> --sha <sha12,...>')
    }
  } catch (e) {
    console.error(`approve: ${e.message}`)
    process.exit(2)
  }
}
