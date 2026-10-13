#!/usr/bin/env node
// Repository inventory: the facts an architecture map is built from, computed by script so the map starts from what
// the code IS, not from what an agent assumes. Files per top-level directory, languages, build manifests,
// entrypoints, test locations, CI, docs, the largest files, and the internal Python import graph. The agent factory
// itself (agents-architetura/, .claude/) is not the product and is left out; the result names what it excluded.
//
//   node agents-architetura/tools/factory/inventory.mjs [--root DIR] [--json]
//   node agents-architetura/tools/factory/inventory.mjs --check-paths '<json array of repo-relative paths>' [--root DIR]
// Exit: 0 · 1 (check-paths: some missing) · 2 usage.
import { existsSync, readFileSync, statSync } from 'node:fs'
import { join, resolve, dirname, extname, basename } from 'node:path'
import { fileURLToPath } from 'node:url'
import { walk } from '../verify/glob.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const MANIFESTS = /(^|\/)(pyproject\.toml|setup\.py|setup\.cfg|requirements[\w.-]*\.txt|environment\.ya?ml|Cargo\.toml|package\.json|go\.mod|Dockerfile|docker-compose\.ya?ml|Makefile)$/
const LANG = { '.py': 'python', '.rs': 'rust', '.js': 'javascript', '.mjs': 'javascript', '.ts': 'typescript', '.sh': 'shell', '.cu': 'cuda', '.cpp': 'c++', '.c': 'c', '.go': 'go', '.md': 'markdown', '.json': 'json', '.toml': 'toml', '.yaml': 'yaml', '.yml': 'yaml' }

export const FACTORY_PREFIXES = ['agents-architetura/', '.claude/']

export function inventory(root) {
  const files = walk(root).filter(f => !FACTORY_PREFIXES.some(p => f.startsWith(p))).sort()
  const byTopDir = {}
  const languages = {}
  for (const f of files) {
    const top = f.includes('/') ? `${f.split('/')[0]}/` : '(root)'
    byTopDir[top] = (byTopDir[top] ?? 0) + 1
    const lang = LANG[extname(f)] ?? 'other'
    languages[lang] = (languages[lang] ?? 0) + 1
  }
  const text = f => { try { return readFileSync(join(root, f), 'utf8') } catch { return '' } }
  const py = files.filter(f => f.endsWith('.py'))
  const entrypoints = [
    ...py.filter(f => /if\s+__name__\s*==\s*['"]__main__['"]/.test(text(f))),
    ...files.filter(f => f.endsWith('.rs') && /\bfn\s+main\s*\(/.test(text(f))),
    ...files.filter(f => f.endsWith('.sh') && !f.includes('/')),
  ].sort()
  // Internal import graph: module path -> repo file, then edges between repo files.
  const modOf = f => f.replace(/\.py$/, '').replace(/\/__init__$/, '').replace(/\//g, '.')
  const fileByMod = new Map(py.map(f => [modOf(f), f]))
  const resolveMod = m => { for (let x = m; x; x = x.includes('.') ? x.slice(0, x.lastIndexOf('.')) : '') { if (fileByMod.has(x)) return fileByMod.get(x); const tail = [...fileByMod.keys()].find(k => k.endsWith(`.${x}`)); if (tail) return fileByMod.get(tail) } return null }
  const imports = []
  for (const f of py) {
    for (const m of text(f).matchAll(/^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))/gm)) {
      const target = resolveMod(m[1] ?? m[2])
      if (target && target !== f) imports.push([f, target])
    }
  }
  return {
    root: basename(root),
    excluded: FACTORY_PREFIXES,
    files: files.length,
    byTopDir,
    languages,
    manifests: files.filter(f => MANIFESTS.test(f)),
    entrypoints,
    tests: files.filter(f => /(^|\/)(tests?|testing)\//.test(f) || /(^|\/)(test_[^/]*|[^/]*_test)\.(py|rs)$/.test(f)).slice(0, 200),
    ci: files.filter(f => f.startsWith('.github/workflows/')),
    docs: files.filter(f => f.endsWith('.md')).slice(0, 100),
    largest: files.map(f => [f, statSync(join(root, f)).size]).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([f, size]) => ({ file: f, bytes: size })),
    imports: [...new Set(imports.map(e => e.join(' -> ')))].sort().slice(0, 500),
  }
}

export function checkPaths(root, paths) {
  if (!Array.isArray(paths)) throw new Error('--check-paths takes a JSON array')
  const bad = paths.filter(p => typeof p !== 'string' || p.includes('..') || p.startsWith('/'))
  const missing = paths.filter(p => typeof p === 'string' && !bad.includes(p) && !existsSync(join(root, p.replace(/\/$/, ''))))
  return { ok: !bad.length && !missing.length, missing, invalid: bad }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const opt = name => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined }
  const root = resolve(opt('--root') ?? ROOT)
  try {
    if (argv.includes('--check-paths')) {
      const r = checkPaths(root, JSON.parse(opt('--check-paths')))
      console.log(JSON.stringify(r))
      process.exit(r.ok ? 0 : 1)
    }
    const inv = inventory(root)
    console.log(argv.includes('--json') ? JSON.stringify(inv) : JSON.stringify(inv, null, 2))
  } catch (e) {
    console.error(`inventory: ${e.message}`)
    process.exit(2)
  }
}
