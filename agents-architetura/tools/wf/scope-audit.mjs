#!/usr/bin/env node
// Scope audit: proves that a fan-out of writer agents touched only the paths they own.
//
//   node agents-architetura/tools/wf/scope-audit.mjs snapshot
//       -> {"tree": "<sha>"}   snapshot of the working tree (tracked + untracked, .gitignore respected)
//   node agents-architetura/tools/wf/scope-audit.mjs check --since <sha> --scopes '<json array of owned paths>'
//       -> {"tree": "<sha>", "changed": [...], "violations": [...]}   exit 1 if any violation
//
// Snapshots are written through a temporary index file, so the real index, HEAD and working tree are never
// modified — safe to run between waves while nothing else is running.
import { execFileSync } from 'node:child_process'
import { copyFileSync, existsSync, rmSync, mkdirSync, statSync, utimesSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const git = (root, args, env = {}) =>
  execFileSync('git', args, { cwd: root, encoding: 'utf8', env: { ...process.env, ...env }, maxBuffer: 256 * 1024 * 1024 }).trim()

const covers = (owned, p) => owned === p || (owned.endsWith('/') && p.startsWith(owned))

export function snapshot(root) {
  const gitDir = resolve(root, git(root, ['rev-parse', '--git-dir']))
  const dir = join(gitDir, 'factory')
  mkdirSync(dir, { recursive: true })
  const index = join(dir, `audit-index-${process.pid}`)
  try {
    const real = join(gitDir, 'index')
    if (existsSync(real)) {
      copyFileSync(real, index) // reuse the stat cache: fast on big trees
      // ...but keep the real index's mtime (Date precision rounds it down, which only widens the window). Git re-checks a
      // "racily clean" entry, one whose stat matches but whose mtime is not older than the index file's, by content.
      // A copy with a fresh mtime would hide a same-size edit made in the same second as the last index write.
      const { atime, mtime } = statSync(real)
      utimesSync(index, atime, mtime)
    }
    git(root, ['add', '-A'], { GIT_INDEX_FILE: index })
    return git(root, ['write-tree'], { GIT_INDEX_FILE: index })
  } finally {
    rmSync(index, { force: true })
  }
}

export function changedSince(root, since) {
  if (!/^[0-9a-f]{40,64}$/.test(since)) throw new Error(`not a tree id: ${JSON.stringify(since)}`)
  const now = snapshot(root)
  const out = git(root, ['diff-tree', '-r', '--name-only', '--no-renames', since, now])
  return { tree: now, changed: out ? out.split('\n').sort() : [] }
}

export function check(root, since, scopes) {
  if (!Array.isArray(scopes) || !scopes.every(s => typeof s === 'string' && s && !s.startsWith('/') && !s.includes('..'))) {
    throw new Error('scopes must be an array of repo-relative paths')
  }
  const { tree, changed } = changedSince(root, since)
  return { tree, changed, violations: changed.filter(p => !scopes.some(s => covers(s, p))) }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [cmd, ...rest] = process.argv.slice(2)
  const opt = name => { const i = rest.indexOf(name); return i >= 0 ? rest[i + 1] : undefined }
  const root = resolve(opt('--root') ?? process.cwd())
  try {
    if (cmd === 'snapshot') {
      console.log(JSON.stringify({ tree: snapshot(root) }))
    } else if (cmd === 'check') {
      const r = check(root, opt('--since') ?? '', JSON.parse(opt('--scopes') ?? 'null'))
      console.log(JSON.stringify(r))
      process.exit(r.violations.length ? 1 : 0)
    } else {
      throw new Error('usage: scope-audit.mjs snapshot | check --since <tree> --scopes <json>')
    }
  } catch (e) {
    console.error(`scope-audit: ${e.message}`)
    process.exit(2)
  }
}
