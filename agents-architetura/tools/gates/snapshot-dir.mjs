// Materializes the current working tree (tracked + untracked, .gitignore respected) into a temp directory via
// its tree id, so destructive checks (applying regression patches, writing mutants) never touch the live tree
// other agents share.
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFileSync, spawnSync } from 'node:child_process'
import { snapshot } from '../wf/scope-audit.mjs'

// Commands run in the snapshot get `env`: no Python bytecode is read or written there. Python (and pytest's assertion
// rewriter) trust a cached .pyc when the source's mtime, in whole seconds, and size match. A mutant or regression
// patch of the same size, written within the same second as the last run, would silently execute the OLD code. The
// pycache prefix is a path that never exists, so no cache (not even a committed __pycache__) is ever consulted.
export function materialize(root, prefix = 'snap-') {
  const tree = snapshot(root)
  const dir = mkdtempSync(join(tmpdir(), prefix))
  const tar = spawnSync('git', ['archive', '--format=tar', tree], { cwd: root, maxBuffer: 2 * 1024 * 1024 * 1024 })
  if (tar.status !== 0) throw new Error(`git archive failed: ${tar.stderr}`)
  execFileSync('tar', ['-x', '-C', dir], { input: tar.stdout })
  return { dir, tree, env: { PYTHONDONTWRITEBYTECODE: '1', PYTHONPYCACHEPREFIX: `${dir}.no-pycache` } }
}
