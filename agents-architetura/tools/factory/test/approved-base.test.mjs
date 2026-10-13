import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { spawnSync } from 'node:child_process'
import { recordGrant } from '../approve.mjs'
import { checkApprovedAt } from '../approved-base.mjs'
import { docLines } from '../doc-lines.mjs'

const git = (root, ...args) => spawnSync('git', args, { cwd: root, encoding: 'utf8' })
const repo = files => {
  const root = mkdtempSync(join(tmpdir(), 'base-'))
  for (const [f, b] of Object.entries(files)) { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), b) }
  git(root, 'init', '-q')
  git(root, 'add', '-A')
  git(root, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'base')
  return root
}

test('the committed version must carry a human grant; pending amendments in the working copy do not void it', () => {
  const root = repo({ 'docs/design/DESIGN.md': 'v1' })
  assert.equal(checkApprovedAt(root, ['docs/design/DESIGN.md']).approved, false, 'never granted')
  recordGrant(root, ['docs/design/DESIGN.md'], { approvedBy: 'human', at: 't' })
  writeFileSync(join(root, 'docs/design/DESIGN.md'), 'v1 + the ADR\'s pending amendment')
  const r = checkApprovedAt(root, ['docs/design/DESIGN.md'])
  assert.deepEqual([r.approved, r.docs[0].approvedBy, r.docs[0].rev], [true, 'human', 'HEAD'])
  // a committed but ungranted change is not a base
  git(root, 'add', '-A')
  git(root, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'unreviewed')
  assert.equal(checkApprovedAt(root, ['docs/design/DESIGN.md']).approved, false)
  assert.match(checkApprovedAt(root, ['docs/design/NOPE.md']).docs[0].reason, /does not exist at HEAD/)
  assert.throws(() => checkApprovedAt(root, ['../x']), /not a repo-relative/)
  assert.throws(() => checkApprovedAt(root, ['docs/design/DESIGN.md'], '--output=x'), /not a git revision/)
})

test('doc-lines counts lines of repo files and refuses paths outside the repo', () => {
  const root = mkdtempSync(join(tmpdir(), 'lines-'))
  writeFileSync(join(root, 'a.md'), 'one\ntwo\nthree\n')
  writeFileSync(join(root, 'b.md'), 'one\ntwo')
  assert.deepEqual(docLines(root, ['a.md', 'b.md', 'c.md']).files.map(f => [f.path, f.exists, f.lines]), [['a.md', true, 3], ['b.md', true, 2], ['c.md', false, 0]])
  assert.throws(() => docLines(root, ['../etc/passwd']), /not a repo-relative/)
})
