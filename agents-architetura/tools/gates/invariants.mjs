#!/usr/bin/env node
// Invariant coverage (the PR's API-I1..I24 coverage grep): every numbered invariant a frozen spec defines
// (SPEC-<key>.md defines <KEY>-I<n>) must be claimed by at least one test through a "covers: <ID>" comment,
// unless its spec line says "DEFERRED-TO: M<n>". A test claiming an id no spec defines is a failure too.
//
//   node agents-architetura/tools/gates/invariants.mjs [--root DIR] [--json]
// Exit: 0 ok · 1 uncovered or unknown ids · 5 not applicable (no spec defines invariants).
import { readdirSync, existsSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { repoFiles, read } from './scan.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const SPEC_RE = /^SPEC-([a-z0-9][a-z0-9-]*)\.md$/
const TEST_ROOTS = ['tests/', 'src/', 'crates/']
const TEST_EXT = /\.(py|rs|ts|js|mjs)$/
const ID_RE = /\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-I\d+\b/g

export function checkInvariants(root) {
  const dir = join(root, 'docs', 'design')
  const specs = existsSync(dir) ? readdirSync(dir).filter(f => SPEC_RE.test(f) && !/-(history|annex)\.md$/.test(f)) : []
  const defined = new Map()
  const deferred = new Map()
  for (const f of specs) {
    const key = f.match(SPEC_RE)[1].toUpperCase()
    const own = new RegExp(`\\b${key.replace(/[-]/g, '\\-')}-I\\d+\\b`, 'g')
    for (const line of read(root, join('docs', 'design', f)).split('\n')) {
      for (const [id] of line.matchAll(own)) {
        if (!defined.has(id)) defined.set(id, f)
        const d = line.match(/DEFERRED-TO:\s*(M\d+[a-z]?)/)
        if (d) deferred.set(id, d[1])
      }
    }
  }
  if (!defined.size) return { applicable: false, reason: specs.length ? 'specs define no numbered invariants' : 'no specs' }
  const covered = new Map()
  for (const p of repoFiles(root).filter(p => TEST_ROOTS.some(r => p.startsWith(r)) && TEST_EXT.test(p))) {
    const text = read(root, p)
    if (!text.includes('covers:')) continue
    for (const line of text.split('\n')) {
      const at = line.indexOf('covers:')
      if (at < 0) continue
      for (const [id] of line.slice(at + 7).matchAll(ID_RE)) covered.set(id, [...(covered.get(id) ?? []), p])
    }
  }
  const uncovered = [...defined.keys()].filter(id => !covered.has(id) && !deferred.has(id)).sort()
  const unknown = [...covered.keys()].filter(id => !defined.has(id)).sort()
  return {
    applicable: true,
    ok: !uncovered.length && !unknown.length,
    defined: defined.size,
    covered: [...defined.keys()].filter(id => covered.has(id)).length,
    deferred: Object.fromEntries([...deferred].filter(([id]) => !covered.has(id))),
    uncovered: uncovered.map(id => `${id} (${defined.get(id)})`),
    unknown: unknown.map(id => `${id} claimed by ${covered.get(id).join(', ')}`),
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const i = argv.indexOf('--root')
  const r = checkInvariants(resolve(i >= 0 ? argv[i + 1] : ROOT))
  if (argv.includes('--json')) console.log(JSON.stringify(r))
  else if (!r.applicable) console.log(`invariants: n/a (${r.reason})`)
  else {
    for (const u of r.uncovered) console.log(`UNCOVERED ${u}`)
    for (const u of r.unknown) console.log(`UNKNOWN   ${u}`)
    for (const [id, m] of Object.entries(r.deferred)) console.log(`DEFERRED  ${id} -> ${m}`)
    console.log(`invariants: ${r.covered}/${r.defined} covered`)
  }
  process.exit(!r.applicable ? 5 : r.ok ? 0 : 1)
}
