#!/usr/bin/env node
// Run ledger: persists a workflow run's task state so humans, later workflows and owner-routed fixes can read it.
//
//   node agents-architetura/tools/wf/ledger.mjs write --dir .factory/runs/<m> --b64 <base64 of JSON>
//     JSON: { summary: {waves, counts, tasks}, owners: {taskId: [scope...]}, journals: {taskId: path}, transitions: [...], notes?: {...} }
//     writes <dir>/state.json and <dir>/transitions.jsonl, prints {"written": [...]}
//   node agents-architetura/tools/wf/ledger.mjs owner --path <repo-relative file>   -> {"task", "journal", "run"} of the latest run owning it
//
// The JSON travels as base64 so no shell quoting can corrupt it; --dir must stay under .factory/.
import { mkdirSync, writeFileSync, readFileSync, readdirSync, existsSync, statSync } from 'node:fs'
import { join, resolve, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const covers = (owned, p) => owned === p || (owned.endsWith('/') && p.startsWith(owned))

export function write(root, dir, payload) {
  const abs = resolve(root, dir)
  const rel = relative(resolve(root, '.factory'), abs)
  if (rel.startsWith('..') || rel === '' || resolve(root, '.factory', rel) !== abs) throw new Error('--dir must be inside .factory/')
  if (!payload || typeof payload !== 'object' || !payload.summary) throw new Error('payload must be an object with a summary')
  mkdirSync(abs, { recursive: true })
  const state = { summary: payload.summary, owners: payload.owners ?? {}, journals: payload.journals ?? {}, notes: payload.notes ?? {} }
  writeFileSync(join(abs, 'state.json'), JSON.stringify(state, null, 2) + '\n')
  writeFileSync(join(abs, 'transitions.jsonl'), (payload.transitions ?? []).map(t => JSON.stringify(t)).join('\n') + '\n')
  return [join(dir, 'state.json'), join(dir, 'transitions.jsonl')]
}

export function owner(root, path) {
  const runs = join(root, '.factory', 'runs')
  if (!existsSync(runs)) return null
  const dirs = readdirSync(runs).filter(d => existsSync(join(runs, d, 'state.json')))
    .sort((a, b) => statSync(join(runs, b, 'state.json')).mtimeMs - statSync(join(runs, a, 'state.json')).mtimeMs)
  for (const d of dirs) {
    const s = JSON.parse(readFileSync(join(runs, d, 'state.json'), 'utf8'))
    const task = Object.keys(s.owners ?? {}).find(id => s.owners[id].some(o => covers(o, path)))
    if (task) return { task, journal: s.journals?.[task] ?? null, run: d }
  }
  return null
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [cmd, ...rest] = process.argv.slice(2)
  const opt = n => { const i = rest.indexOf(n); return i >= 0 ? rest[i + 1] : undefined }
  const root = resolve(opt('--root') ?? process.cwd())
  try {
    if (cmd === 'write') {
      const payload = JSON.parse(Buffer.from(opt('--b64') ?? '', 'base64').toString('utf8'))
      console.log(JSON.stringify({ written: write(root, opt('--dir') ?? '', payload) }))
    } else if (cmd === 'owner') {
      console.log(JSON.stringify(owner(root, opt('--path') ?? '')))
    } else {
      throw new Error('usage: ledger.mjs write --dir <.factory/...> --b64 <json> | owner --path <file>')
    }
  } catch (e) {
    console.error(`ledger: ${e.message}`)
    process.exit(2)
  }
}
