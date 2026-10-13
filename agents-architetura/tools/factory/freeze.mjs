#!/usr/bin/env node
// Frozen documents (specs, the request specification, milestone gates). Once frozen, a document may change only
// through a recorded amendment — the conflict protocol (STOP -> report -> the researcher re-evaluates -> amend with
// a reason). A frozen document whose content no longer matches its last record changed SILENTLY: the "frozen"
// ladder rung fails until it is reverted or amended on the record.
//
//   node agents-architetura/tools/factory/freeze.mjs freeze <doc...>                      first freeze (refuses an already-frozen doc)
//   node agents-architetura/tools/factory/freeze.mjs amend <doc> --reason "<why>" --by <decision id>
//   node agents-architetura/tools/factory/freeze.mjs check [--json]                       exit 0 ok · 1 silent change · 5 nothing frozen
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
export const STORE = 'docs/design/FROZEN.json'
const DOC_RE = /^[\w./-]+$/
const sha = p => createHash('sha256').update(readFileSync(p)).digest('hex')
const load = root => (existsSync(join(root, STORE)) ? JSON.parse(readFileSync(join(root, STORE), 'utf8')) : { docs: {} })
const save = (root, s) => { mkdirSync(dirname(join(root, STORE)), { recursive: true }); writeFileSync(join(root, STORE), JSON.stringify(s, null, 2) + '\n') }
const valid = doc => DOC_RE.test(doc) && !doc.includes('..') && !doc.startsWith('/')

export function freeze(root, docs, at = new Date().toISOString()) {
  const s = load(root)
  for (const doc of docs) {
    if (!valid(doc)) throw new Error(`not a repo-relative path: ${JSON.stringify(doc)}`)
    if (!existsSync(join(root, doc))) throw new Error(`${doc} does not exist`)
    if (s.docs[doc]) throw new Error(`${doc} is already frozen — change it with "amend" and a reason`)
    s.docs[doc] = { history: [{ sha256: sha(join(root, doc)), at, reason: 'frozen' }] }
  }
  save(root, s)
  return s
}

export function amend(root, doc, { reason, by, at = new Date().toISOString() }) {
  const s = load(root)
  if (!s.docs[doc]) throw new Error(`${doc} is not frozen`)
  if (typeof reason !== 'string' || reason.trim().length < 20) throw new Error('an amendment needs a reason of at least 20 characters')
  if (!/^[\w-]{2,64}$/.test(by ?? '')) throw new Error('an amendment names the decision that authorized it (--by <id>)')
  s.docs[doc].history.push({ sha256: sha(join(root, doc)), at, reason: reason.trim(), by })
  save(root, s)
  return s.docs[doc]
}

export function check(root) {
  const s = load(root)
  const docs = Object.keys(s.docs).sort()
  if (!docs.length) return { applicable: false }
  const violations = []
  for (const doc of docs) {
    if (!existsSync(join(root, doc))) { violations.push(`${doc}: frozen but deleted`); continue }
    const last = s.docs[doc].history.at(-1)
    if (sha(join(root, doc)) !== last.sha256) violations.push(`${doc}: changed without an amendment record (last recorded: ${last.reason}, ${last.at})`)
  }
  return { applicable: true, ok: !violations.length, frozen: docs.length, amendments: docs.reduce((n, d) => n + s.docs[d].history.length - 1, 0), violations }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [cmd, ...rest] = process.argv.slice(2)
  const opt = name => { const i = rest.indexOf(name); return i >= 0 ? rest[i + 1] : undefined }
  const docs = rest.filter((a, i) => !a.startsWith('--') && !['--reason', '--by', '--root'].includes(rest[i - 1]))
  const root = resolve(opt('--root') ?? ROOT)
  try {
    if (cmd === 'freeze') { freeze(root, docs); console.log(`frozen: ${docs.join(', ')}`) } else if (cmd === 'amend') { amend(root, docs[0], { reason: opt('--reason'), by: opt('--by') }); console.log(`amended: ${docs[0]}`) } else if (cmd === 'check') {
      const r = check(root)
      if (rest.includes('--json')) console.log(JSON.stringify(r))
      else if (!r.applicable) console.log('frozen: n/a (nothing frozen yet)')
      else { for (const v of r.violations) console.log(`VIOLATION ${v}`); console.log(`frozen: ${r.frozen} documents, ${r.amendments} recorded amendments, ${r.violations.length} silent changes`) }
      process.exit(!r.applicable ? 5 : r.ok ? 0 : 1)
    } else throw new Error('usage: freeze.mjs freeze <doc...> | amend <doc> --reason "..." --by <id> | check [--json]')
  } catch (e) {
    console.error(`freeze: ${e.message}`)
    process.exit(2)
  }
}
