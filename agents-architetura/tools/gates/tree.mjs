// Build identity: what tree, toolchain and lockfiles a piece of evidence was produced on.
// Evidence whose tree differs from the current tree is stale and must not be accepted
// (the PR's "binaries of record": `find Source -newer jsc` returned nothing).
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { snapshot } from '../wf/scope-audit.mjs'

// Content hash of the working tree (tracked + untracked, .gitignore respected); null outside a git repository.
export function treeId(root) {
  const inside = spawnSync('git', ['rev-parse', '--is-inside-work-tree'], { cwd: root, encoding: 'utf8' })
  if (inside.status !== 0 || inside.stdout.trim() !== 'true') return null
  return snapshot(root)
}

const VERSION_PROBES = {
  node: ['node', ['--version']],
  python: ['python3', ['--version']],
  uv: ['uv', ['--version']],
  cargo: ['cargo', ['--version']],
  rustc: ['rustc', ['--version']],
}

export function toolchain(probes = VERSION_PROBES) {
  const out = {}
  for (const [name, [cmd, args]] of Object.entries(probes)) {
    const r = spawnSync(cmd, args, { encoding: 'utf8', timeout: 10000 })
    out[name] = r.status === 0 ? `${r.stdout}${r.stderr}`.trim().split('\n')[0] : 'n/a'
  }
  return out
}

export const LOCKFILES = ['uv.lock', 'Cargo.lock', 'package-lock.json']

export function lockHashes(root, files = LOCKFILES) {
  const out = {}
  for (const f of files) {
    if (existsSync(join(root, f))) out[f] = createHash('sha256').update(readFileSync(join(root, f))).digest('hex')
  }
  return out
}

export const identityOf = root => ({ tree: treeId(root), toolchain: toolchain(), locks: lockHashes(root) })

// Evidence is fresh only when it names a tree and that tree is the current one.
export function staleReason(evidenceTree, currentTree) {
  if (!evidenceTree) return 'evidence carries no tree id'
  if (!currentTree) return 'current tree id unavailable'
  return evidenceTree === currentTree ? null : `stale evidence: produced on tree ${evidenceTree}, current tree is ${currentTree}`
}
