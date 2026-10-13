import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { checkApproval, recordGrant, pendingDecisions, humanGuard, confirmedGrant, sha256 } from '../approve.mjs'
import { freeze, amend, check } from '../freeze.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const repo = files => {
  const root = mkdtempSync(join(tmpdir(), 'appr-'))
  for (const [f, b] of Object.entries(files)) { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), b) }
  return root
}

test('approval binds to the exact text: silence is not approval; an edit voids it', () => {
  const root = repo({ 'docs/design/DESIGN.md': 'v1' })
  assert.match(checkApproval(root, 'docs/design/DESIGN.md').reason, /silence is not approval/)
  recordGrant(root, ['docs/design/DESIGN.md'], { approvedBy: 'human', at: 't' })
  assert.equal(checkApproval(root, 'docs/design/DESIGN.md').approved, true)
  writeFileSync(join(root, 'docs/design/DESIGN.md'), 'v2')
  const c = checkApproval(root, 'docs/design/DESIGN.md')
  assert.equal(c.approved, false)
  assert.match(c.reason, /changed after its last approval/)
  assert.throws(() => checkApproval(root, '../etc/passwd'), /not a repo-relative/)
})

test('pending: the design, PROPOSED ADRs and experiment plans without a grant for their current text', () => {
  const root = repo({
    'docs/design/DESIGN.md': 'd',
    'docs/design/adr/ADR-0001-initial.md': 'Status: ACCEPTED\n',
    'docs/design/adr/ADR-0002-sync.md': 'Status: PROPOSED\n',
    'docs/experiments/e1/PLAN.json': '{}',
  })
  assert.deepEqual(pendingDecisions(root).map(p => p.doc), ['docs/design/DESIGN.md', 'docs/design/adr/ADR-0002-sync.md', 'docs/experiments/e1/PLAN.json'])
  recordGrant(root, ['docs/design/DESIGN.md'], { approvedBy: 'human' })
  assert.equal(pendingDecisions(root).length, 2)
})

test('pending lists a phase NEXT document, its pinned gates and the built workflow it authorizes; a later edit to either voids the grant', () => {
  const root = repo({ 'docs/design/NEXT-zbv.md': 'objective', 'docs/design/GATES-zbv.json': '{"milestone":"zbv"}', '.claude/workflows/wan-x-zbv.js': 'code v1', 'docs/design/NEXT-x.md': 'no workflow yet' })
  assert.deepEqual(pendingDecisions(root).map(p => p.doc), ['docs/design/NEXT-x.md', 'docs/design/NEXT-zbv.md', 'docs/design/GATES-zbv.json', '.claude/workflows/wan-x-zbv.js'])
  recordGrant(root, ['docs/design/NEXT-zbv.md', 'docs/design/GATES-zbv.json', '.claude/workflows/wan-x-zbv.js'], { approvedBy: 'human' })
  assert.deepEqual(pendingDecisions(root).map(p => p.doc), ['docs/design/NEXT-x.md'])
  writeFileSync(join(root, 'docs/design/GATES-zbv.json'), '{"milestone":"zbv","pinned":[]}')
  assert.deepEqual(pendingDecisions(root).map(p => p.doc), ['docs/design/NEXT-x.md', 'docs/design/GATES-zbv.json'])
  recordGrant(root, ['docs/design/GATES-zbv.json'], { approvedBy: 'human' })
  writeFileSync(join(root, '.claude/workflows/wan-x-zbv.js'), 'code v2')
  assert.deepEqual(pendingDecisions(root).map(p => p.doc), ['docs/design/NEXT-x.md', '.claude/workflows/wan-x-zbv.js'])
})

test('the CLI reads every document: one doc, several, with or without --note and --json', () => {
  const root = repo({ 'docs/design/DESIGN.md': 'v1', 'docs/design/NEXT-a.md': 'a' })
  recordGrant(root, ['docs/design/DESIGN.md'], { approvedBy: 'human' })
  const cli = (...a) => spawnSync(process.execPath, [join(HERE, '..', 'approve.mjs'), ...a], { encoding: 'utf8', env: { ...process.env, CLAUDECODE: '' }, cwd: root })
  // The CLI resolves the repo from its own location, so exercise it on the real repo for parsing, and on `root` through the library.
  const one = cli('check', 'docs/design/DESIGN.md', '--json')
  assert.notEqual(one.status, 2, one.stderr)
  assert.doesNotMatch(one.stderr, /needs at least one document/)
  assert.deepEqual(JSON.parse(one.stdout).docs.map(d => d.doc), ['docs/design/DESIGN.md'])
  const two = cli('check', 'docs/design/DESIGN.md', 'docs/design/BRIEF.md', '--json')
  assert.deepEqual(JSON.parse(two.stdout).docs.map(d => d.doc), ['docs/design/DESIGN.md', 'docs/design/BRIEF.md'])
  const noted = cli('grant', '--note', 'why', 'docs/design/DESIGN.md')
  assert.equal(noted.status, 3)  // refused by the human guard (no terminal), after parsing
  // --note's value is skipped and every document after it is read
  const checkedWithNote = cli('check', '--note', 'not-a-doc', 'docs/design/DESIGN.md', '--json')
  assert.deepEqual(JSON.parse(checkedWithNote.stdout).docs.map(d => d.doc), ['docs/design/DESIGN.md'])
  assert.equal(checkApproval(root, 'docs/design/NEXT-a.md').approved, false)
})

test('in Claude Code a grant needs --confirm with the hash of the exact text shown; a changed or mismatched document is refused', () => {
  const root = repo({ 'docs/design/DESIGN.md': 'v1', 'docs/design/adr/ADR-0002.md': 'adr' })
  const h = d => sha256(readFileSync(join(root, d))).slice(0, 12)
  assert.throws(() => confirmedGrant(root, ['docs/design/DESIGN.md'], '000000000000', { approvedBy: 'h' }), /changed since it was shown/)
  assert.throws(() => confirmedGrant(root, ['docs/design/DESIGN.md'], 'abc', { approvedBy: 'h' }), /at least 12 hex/)
  assert.throws(() => confirmedGrant(root, ['docs/design/DESIGN.md', 'docs/design/adr/ADR-0002.md'], h('docs/design/DESIGN.md'), { approvedBy: 'h' }), /one sha256 prefix per document/)
  assert.equal(checkApproval(root, 'docs/design/DESIGN.md').approved, false)
  const shown = h('docs/design/DESIGN.md')
  writeFileSync(join(root, 'docs/design/DESIGN.md'), 'v2 edited after it was shown')
  assert.throws(() => confirmedGrant(root, ['docs/design/DESIGN.md'], shown, { approvedBy: 'h' }), /changed since it was shown/)
  const r = confirmedGrant(root, ['docs/design/DESIGN.md', 'docs/design/adr/ADR-0002.md'], `${h('docs/design/DESIGN.md')},${h('docs/design/adr/ADR-0002.md')}`, { approvedBy: 'h (Claude Code prompt)' })
  assert.deepEqual(r.map(x => x.doc), ['docs/design/DESIGN.md', 'docs/design/adr/ADR-0002.md'])
  assert.equal(checkApproval(root, 'docs/design/DESIGN.md').approvedBy, 'h (Claude Code prompt)')
})

test('grant is human-only: an agent session needs the confirmed path, a terminal needs a TTY', () => {
  assert.match(humanGuard({ CLAUDECODE: '1' }, { isTTY: true }, { isTTY: true }), /grants are not taken from an agent session/)
  assert.match(humanGuard({}, { isTTY: false }, { isTTY: true }), /interactive terminal/)
  assert.equal(humanGuard({}, { isTTY: true }, { isTTY: true }), null)
  const cli = spawnSync(process.execPath, [join(HERE, '..', 'approve.mjs'), 'grant', 'docs/design/DESIGN.md'], { encoding: 'utf8', env: { ...process.env, CLAUDECODE: '1' } })
  assert.equal(cli.status, 3)
  const noTty = spawnSync(process.execPath, [join(HERE, '..', 'approve.mjs'), 'grant', 'docs/design/DESIGN.md'], { encoding: 'utf8', env: { ...process.env, CLAUDECODE: '' }, input: 'APPROVE\n' })
  assert.equal(noTty.status, 3)
  assert.match(noTty.stderr, /interactive terminal/)
})

test('freeze: silent changes fail the check; recorded amendments pass; re-freezing is refused', () => {
  const root = repo({ 'docs/design/SPEC-a.md': 'one', 'docs/design/BRIEF.md': 'b' })
  assert.equal(check(root).applicable, false)
  freeze(root, ['docs/design/SPEC-a.md', 'docs/design/BRIEF.md'], 't0')
  assert.equal(check(root).ok, true)
  writeFileSync(join(root, 'docs/design/SPEC-a.md'), 'two')
  assert.match(check(root).violations[0], /changed without an amendment record/)
  assert.throws(() => freeze(root, ['docs/design/SPEC-a.md']), /already frozen/)
  assert.throws(() => amend(root, 'docs/design/SPEC-a.md', { reason: 'short', by: 'c1' }), /at least 20 characters/)
  assert.throws(() => amend(root, 'docs/design/SPEC-a.md', { reason: 'the spec was wrong about frame sizes', by: '' }), /names the decision/)
  amend(root, 'docs/design/SPEC-a.md', { reason: 'conflict wire-a2: the spec was wrong about frame sizes', by: 'conflict-wire-a2' })
  const r = check(root)
  assert.deepEqual([r.ok, r.amendments], [true, 1])
})

test('autonomy: with humanGrants false a grant is waived and said so, never written; a paid plan and a strict check still need a human', () => {
  const root = repo({
    'docs/design/DESIGN.md': 'd',
    'docs/design/adr/ADR-0009-x.md': 'Status: PROPOSED\n',
    'docs/experiments/local/PLAN.json': '{"id":"local","cloud":false}',
    'docs/experiments/paid/PLAN.json': '{"id":"paid","cloud":true,"maxBudgetUsd":40}',
    'agents-architetura/tools/factory/autonomy.json': '{"humanGrants": false, "at": "2026-10-09"}',
  })
  const d = checkApproval(root, 'docs/design/DESIGN.md')
  assert.deepEqual([d.approved, d.waived], [true, true])
  assert.match(d.approvedBy, /autonomy \(the human waived grants on 2026-10-09; no human grant\)/)
  assert.equal(checkApproval(root, 'docs/experiments/local/PLAN.json').approved, true)
  const paid = checkApproval(root, 'docs/experiments/paid/PLAN.json')
  assert.equal(paid.approved, false, 'money always needs a real human grant')
  assert.equal(checkApproval(root, 'docs/design/DESIGN.md', { strict: true }).approved, false, 'a strict check ignores the switch')
  assert.deepEqual(pendingDecisions(root).map(p => p.doc), ['docs/experiments/paid/PLAN.json'])
  assert.equal(existsSync(join(root, 'docs/design/APPROVALS.json')), false, 'nothing is written in the human\'s name')
  writeFileSync(join(root, 'agents-architetura/tools/factory/autonomy.json'), '{"humanGrants": true}')
  assert.equal(checkApproval(root, 'docs/design/DESIGN.md').approved, false, 'switched back on, grants are required again')
})
