// Shared source scanning for the static gates (discipline, invariants, claims).
import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { walk } from '../verify/glob.mjs'

export const AWAITING = 'awaiting implementation'
export const isPyTest = p => /(^|\/)(test_[^/]*|[^/]*_test)\.py$/.test(p)
export const isRustTest = p => p.endsWith('.rs') && (p.startsWith('tests/') || /\/tests\//.test(p))

export const repoFiles = root => walk(root)
export const read = (root, p) => readFileSync(join(root, p), 'utf8')
export const lineOf = (text, index) => text.slice(0, index).split('\n').length

// Text of a call starting at the "(" at `open`, honoring nesting and quoted strings.
export function callText(text, open) {
  let depth = 0
  let quote = null
  for (let i = open; i < text.length; i++) {
    const c = text[i]
    if (quote) {
      if (c === '\\') i++
      else if (text.startsWith(quote, i)) { i += quote.length - 1; quote = null }
      continue
    }
    if (text.startsWith('"""', i) || text.startsWith("'''", i)) { quote = text.slice(i, i + 3); i += 2 } else if (c === '"' || c === "'") quote = c
    else if (c === '#') { const nl = text.indexOf('\n', i); i = nl < 0 ? text.length : nl }
    else if (c === '(') depth++
    else if (c === ')' && --depth === 0) return text.slice(open, i + 1)
  }
  return text.slice(open)
}

export function loadFindings(root) {
  const p = join(root, 'agents-architetura', 'docs', 'findings.json')
  if (!existsSync(p)) return null
  const data = JSON.parse(readFileSync(p, 'utf8'))
  const map = new Map()
  for (const f of data.findings ?? []) map.set(String(f.id), f)
  return map
}
