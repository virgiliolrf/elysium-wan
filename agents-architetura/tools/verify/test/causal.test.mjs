import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, cpSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { save, ab } from '../causal.mjs'

// A tiny "repo": bug.sh fails when flaky=1 is in config.txt. The fix rewrites config.txt and adds guard.txt.
const repo = () => {
  const root = mkdtempSync(join(tmpdir(), 'causal-'))
  mkdirSync(join(root, 'src'))
  writeFileSync(join(root, 'src/config.txt'), 'flaky=1\n')
  writeFileSync(join(root, 'repro.sh'), 'grep -q "flaky=1" src/config.txt && [ ! -e src/guard.txt ] && exit 1; exit 0\n')
  return root
}
// The snapshot is a plain copy here; the real one materializes the git tree.
const copySnapshot = root => { const dir = mkdtempSync(join(tmpdir(), 'causal-snap-')); cpSync(root, dir, { recursive: true }); return { dir, env: {} } }

test('causal: the pre-fix files bring the failure back, the fixed tree is clean, and the live tree is untouched', () => {
  const root = repo()
  const prefix = join(root, '.prefix')
  const m = save(root, prefix, ['src/config.txt', 'src/guard.txt'])
  assert.deepEqual(m, { files: ['src/config.txt'], absent: ['src/guard.txt'] })
  writeFileSync(join(root, 'src/config.txt'), 'flaky=0\n')
  writeFileSync(join(root, 'src/guard.txt'), 'on\n')
  const r = ab(root, prefix, 'sh repro.sh', { runs: 4, snapshot: copySnapshot })
  assert.equal(r.causal, true, r.why)
  assert.deepEqual([r.reverted.failures, r.fixed.failures], [4, 0])
  assert.equal(readFileSync(join(root, 'src/config.txt'), 'utf8'), 'flaky=0\n')
  assert.ok(existsSync(join(root, 'src/guard.txt')))
})

test('not causal: a fix that changed nothing relevant, or a fixed tree that still fails', () => {
  const root = repo()
  const prefix = join(root, '.prefix')
  save(root, prefix, ['src/config.txt'])
  writeFileSync(join(root, 'src/guard.txt'), 'on\n')
  const lucky = ab(root, join(root, '.prefix'), 'true', { runs: 3, snapshot: copySnapshot })
  assert.equal(lucky.causal, false)
  assert.match(lucky.why, /did not bring the failure back/)
  const still = ab(root, prefix, 'false', { runs: 2, snapshot: copySnapshot })
  assert.equal(still.causal, false)
  assert.match(still.why, /fixed tree still fails \(2\/2\)/)
  assert.throws(() => save(root, prefix, ['../etc/passwd']), /without "\.\."/)
  assert.throws(() => save(root, prefix, []), /repo-relative/)
})
