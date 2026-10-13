#!/usr/bin/env node
// Line counts of documents, for size budgets a workflow decides on (an ADR over budget is split). Read-only.
//   node agents-architetura/tools/factory/doc-lines.mjs <file>... --json
// Prints {"files": [{"path", "exists", "lines"}]}. Exit 0, or 2 on a path outside the repo.
import { existsSync, readFileSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const PATH_RE = /^[\w./-]+$/

export function docLines(root, paths) {
  return {
    files: paths.map(path => {
      if (!PATH_RE.test(path) || path.includes('..') || path.startsWith('/')) throw new Error(`not a repo-relative path: ${JSON.stringify(path)}`)
      const p = join(root, path)
      if (!existsSync(p)) return { path, exists: false, lines: 0 }
      const text = readFileSync(p, 'utf8')
      return { path, exists: true, lines: text ? text.split('\n').length - (text.endsWith('\n') ? 1 : 0) : 0 }
    }),
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const paths = process.argv.slice(2).filter(a => !a.startsWith('--'))
  try {
    console.log(JSON.stringify(docLines(ROOT, paths)))
  } catch (e) {
    console.error(`doc-lines: ${e.message}`)
    process.exit(2)
  }
}
