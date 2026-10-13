import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loadLib } from '../sim.mjs'

const { validatePlan, scopesOverlap, safeScopePath } = loadLib(['dag'])

test('safeScopePath accepts repo-relative files and dirs, rejects escapes', () => {
  for (const ok of ['src/a.py', 'src/net/', 'tests/unit/net/test_x.py', 'docs/design/SPEC-net.md']) assert.ok(safeScopePath(ok), ok)
  for (const bad of ['/etc/passwd', '../x', 'src/../x', 'src//x', './src', '.git/config', '.claude/workflows/wan-fix.js', 'a b', '', 7])
    assert.ok(!safeScopePath(bad), String(bad))
})

test('scopesOverlap: equal paths and directory containment', () => {
  assert.ok(scopesOverlap('src/a.py', 'src/a.py'))
  assert.ok(scopesOverlap('src/', 'src/a.py'))
  assert.ok(scopesOverlap('src/net/x.py', 'src/net/'))
  assert.ok(!scopesOverlap('src/net/', 'src/network/'))
  assert.ok(!scopesOverlap('src/a.py', 'src/a.pyi'))
})

test('waves are deterministic Kahn levels', () => {
  const r = validatePlan([
    { id: 'tests', deps: ['backend', 'api'], scope: ['tests/'] },
    { id: 'api', deps: ['core'], scope: ['src/api/'] },
    { id: 'backend', deps: ['core'], scope: ['src/backend/'] },
    { id: 'core', scope: ['src/core/'] },
  ])
  assert.deepEqual(r, { ok: true, errors: [], waves: [['core'], ['api', 'backend'], ['tests']], overlaps: [] })
})

test('concurrent tasks with overlapping scopes are rejected', () => {
  const r = validatePlan([
    { id: 'a', scope: ['src/net/'] },
    { id: 'b', scope: ['src/net/peer.py'] },
  ])
  assert.equal(r.ok, false)
  assert.match(r.errors[0], /scope collision: a owns src\/net\/, b owns src\/net\/peer.py/)
})

test('overlap is allowed when one task transitively depends on the other', () => {
  const r = validatePlan([
    { id: 'a', scope: ['src/net/'] },
    { id: 'm', deps: ['a'], scope: ['docs/m.md'] },
    { id: 'b', deps: ['m'], scope: ['src/net/peer.py'] },
  ])
  assert.equal(r.ok, true)
})

test('cycles, unknown deps, bad ids, duplicates and empty scopes are reported', () => {
  assert.match(validatePlan([{ id: 'a', deps: ['b'], scope: ['x'] }, { id: 'b', deps: ['a'], scope: ['y'] }]).errors[0], /cycle.*a, b/)
  assert.match(validatePlan([{ id: 'a', deps: ['zz'], scope: ['x'] }]).errors[0], /unknown dependency "zz"/)
  assert.match(validatePlan([{ id: 'Bad Id', scope: ['x'] }]).errors[0], /bad task id/)
  assert.match(validatePlan([{ id: 'a', scope: ['x'] }, { id: 'a', scope: ['y'] }]).errors[0], /duplicate task id: a/)
  assert.match(validatePlan([{ id: 'a', scope: [] }]).errors[0], /empty scope/)
  assert.deepEqual(validatePlan([]).errors, ['plan has no tasks'])
})

test('allowOverlap reports concurrent overlaps instead of failing (the scheduler serializes them)', () => {
  const r = validatePlan([{ id: 'a', scope: ['src/net/'] }, { id: 'b', scope: ['src/net/peer.py'] }], { allowOverlap: true })
  assert.equal(r.ok, true)
  assert.deepEqual(r.waves, [['a', 'b']])
  assert.match(r.overlaps[0], /scope collision: a owns src\/net\//)
})
