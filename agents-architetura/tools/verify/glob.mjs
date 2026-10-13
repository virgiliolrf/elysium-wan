// Minimal, dependency-free glob over the repository: `**` crosses directories, `*` and `?` do not.
// Build/venv/VCS directories are never walked. Results are repo-relative and sorted.
import { readdirSync } from 'node:fs'
import { join, relative } from 'node:path'

const SKIP_DIRS = new Set(['.git', 'node_modules', 'target', '.factory', '__pycache__', '.hypothesis', '.pytest_cache', '.mypy_cache', '.ruff_cache'])
const skipDir = name => SKIP_DIRS.has(name) || name.startsWith('.venv')

export function globToRegExp(glob) {
  let re = ''
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i]
    if (c === '*' && glob[i + 1] === '*') {
      if (glob[i + 2] === '/') { re += '(?:.*/)?'; i += 2 } else { re += '.*'; i++ }
    } else if (c === '*') re += '[^/]*'
    else if (c === '?') re += '[^/]'
    else re += c.replace(/[.+^${}()|[\]\\]/g, '\\$&')
  }
  return new RegExp(`^${re}$`)
}

export function walk(root, dir = root, out = []) {
  let entries
  try { entries = readdirSync(dir, { withFileTypes: true }) } catch { return out }
  for (const e of entries) {
    if (e.isDirectory()) { if (!skipDir(e.name)) walk(root, join(dir, e.name), out) } else if (e.isFile()) out.push(relative(root, join(dir, e.name)))
  }
  return out
}

export function glob(root, patterns) {
  const res = (Array.isArray(patterns) ? patterns : [patterns]).map(globToRegExp)
  return walk(root).filter(p => res.some(re => re.test(p))).sort()
}
