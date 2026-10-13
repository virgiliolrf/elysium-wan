#!/usr/bin/env node
// Formatter and linter scoped to the change. A branch is accountable for the files it touches, not for files another
// owner changed on main: a whole-tree `ruff format --check .` made one workstream's PR fail on another owner's file
// (root README.md) the moment main was merged in. Main itself stays whole-tree clean, because every push to main
// checks everything; so by induction every PR only needs to check what it changed.
//
// Scope:
// - The base is $FACTORY_BASE, else the merge-base with origin/main (or main).
// - Diff mode checks the files added, copied, modified or renamed since the base, plus untracked files.
// - The whole tree is checked when there is no base (no git, no main, a shallow clone without history), when HEAD is the
//   base (a push to main), or when the change touches the tool's own configuration (pyproject.toml, ruff.toml,
//   .ruff.toml): a rule change can make untouched files fail. The fallback is always more checking, never less.
// - Explicit file arguments go through --force-exclude, so the configured excludes still apply.
//
//   node agents-architetura/tools/gates/scoped.mjs ruff-format|ruff-check [--root DIR] [--plan]
// Exit: the tool's exit code · 2 usage · 5 not applicable (diff mode with no changed file of the tool's kind).
import { existsSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')

export const TOOLS = {
  'ruff-format': {
    kind: f => /\.pyi?$/.test(f),
    config: ['pyproject.toml', 'ruff.toml', '.ruff.toml'],
    all: ['uv', ['run', 'ruff', 'format', '--check', '.']],
    files: fs => ['uv', ['run', 'ruff', 'format', '--check', '--force-exclude', '--', ...fs]],
  },
  'ruff-check': {
    kind: f => /\.pyi?$/.test(f),
    config: ['pyproject.toml', 'ruff.toml', '.ruff.toml'],
    all: ['uv', ['run', 'ruff', 'check', '.']],
    files: fs => ['uv', ['run', 'ruff', 'check', '--force-exclude', '--', ...fs]],
  },
}

const SAFE_REV = /^[\w./~^@{}-]+$/
// Above this many files one command line risks the OS argument limit (E2BIG); the whole tree is checked instead.
const MAX_FILES = 2000
const git = (root, args) => spawnSync('git', args, { cwd: root, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 })

export function resolveScopeBase(root, env = process.env) {
  if (git(root, ['rev-parse', '--git-dir']).status !== 0) return { base: null, why: 'not a git checkout' }
  const head = git(root, ['rev-parse', 'HEAD']).stdout.trim()
  const pinned = env.FACTORY_BASE
  if (pinned) {
    if (!SAFE_REV.test(pinned)) throw new Error(`bad FACTORY_BASE: ${JSON.stringify(pinned)}`)
    const r = git(root, ['rev-parse', '--verify', `${pinned}^{commit}`])
    if (r.status !== 0) return { base: null, why: `FACTORY_BASE ${pinned} is not in this clone (shallow?)` }
    return { base: r.stdout.trim(), head, why: 'FACTORY_BASE' }
  }
  for (const ref of ['origin/main', 'main']) {
    const r = git(root, ['merge-base', 'HEAD', ref])
    if (r.status === 0 && r.stdout.trim()) return { base: r.stdout.trim(), head, why: `merge-base with ${ref}` }
  }
  return { base: null, why: 'no merge-base with main' }
}

export function changedFiles(root, base) {
  // -z: raw paths, never C-quoted (core.quotePath would hide a non-ASCII name from existsSync)
  const tracked = git(root, ['diff', '--name-only', '-z', '--diff-filter=ACMR', '--no-renames', base, '--'])
  if (tracked.status !== 0) throw new Error(`git diff against ${base} failed: ${tracked.stderr.trim()}`)
  const untracked = git(root, ['ls-files', '-z', '--others', '--exclude-standard'])
  const all = [...tracked.stdout.split('\0'), ...untracked.stdout.split('\0')].filter(Boolean)
  return [...new Set(all)].filter(f => existsSync(join(root, f))).sort()
}

// {mode: 'all' | 'diff', files, why}. Pure planning, testable without the tool installed.
export function planScope(root, tool, env = process.env) {
  const t = TOOLS[tool]
  if (!t) throw new Error(`unknown tool ${tool}; choose from ${Object.keys(TOOLS).join(', ')}`)
  const { base, head, why } = resolveScopeBase(root, env)
  if (!base) return { mode: 'all', files: [], why: `whole tree: ${why}` }
  if (base === head) return { mode: 'all', files: [], why: 'whole tree: HEAD is the base (main itself)' }
  const changed = changedFiles(root, base)
  // Tool configuration is hierarchical (ruff reads the nearest pyproject.toml/ruff.toml), so match by file name at any depth.
  const config = changed.filter(f => t.config.includes(f.split('/').pop()))
  if (config.length) return { mode: 'all', files: [], why: `whole tree: the change touches ${config.join(', ')}` }
  const files = changed.filter(t.kind)
  if (files.length > MAX_FILES) return { mode: 'all', files: [], why: `whole tree: ${files.length} changed files exceed one command line` }
  return { mode: 'diff', files, why: `${files.length} changed file(s) since ${base.slice(0, 12)} (${why})` }
}

const spawnTool = root => (cmd, args) => {
  const r = spawnSync(cmd, args, { cwd: root, stdio: 'inherit' })
  if (r.error) console.error(`scoped: could not run ${cmd}: ${r.error.message}`)
  return r.status
}

export function runScoped(root, tool, { env = process.env, run = spawnTool(root) } = {}) {
  const plan = planScope(root, tool, env)
  console.log(`scoped ${tool}: ${plan.why}`)
  if (plan.mode === 'diff' && !plan.files.length) return 5
  const [cmd, args] = plan.mode === 'all' ? TOOLS[tool].all : TOOLS[tool].files(plan.files)
  const status = run(cmd, args)
  return status ?? 1
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const tool = argv[0]
  const at = argv.indexOf('--root')
  const root = at >= 0 ? resolve(argv[at + 1]) : ROOT
  if (!TOOLS[tool]) {
    console.error(`usage: scoped.mjs ${Object.keys(TOOLS).join('|')} [--root DIR] [--plan]`)
    process.exit(2)
  }
  if (argv.includes('--plan')) {
    console.log(JSON.stringify(planScope(root, tool), null, 2))
    process.exit(0)
  }
  process.exit(runScoped(root, tool))
}
