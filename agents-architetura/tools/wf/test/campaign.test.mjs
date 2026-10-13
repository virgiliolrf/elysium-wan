import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loadLib } from '../sim.mjs'

const L = loadLib(['hygiene', 'dag', 'project', 'campaign'], { agent: () => null })
const fam = (id, count = 1) => ({ id, kind: 'k', count, signature: id })
const rule = (familyId, ruling, files) => ({ familyId, ruling, fixShape: 's', files })

test('partitionWave: real bugs solo (max 3), mechanical batched on disjoint files, overlaps deferred', () => {
  const fams = ['r1', 'r2', 'r3', 'r4', 'm1', 'm2', 'm3', 'b1', 'x1', 'u1'].map(id => fam(id))
  const rulings = [
    rule('r1', 'real-bug', ['src/a.py']), rule('r2', 'real-bug', ['src/b.py']), rule('r3', 'real-bug', ['src/c.py']), rule('r4', 'real-bug', ['src/d.py']),
    rule('m1', 'protocol-fix', ['src/a.py']), rule('m2', 'protocol-fix', ['src/e/']), rule('m3', 'protocol-fix', ['src/e/f.py']),
    rule('b1', 'benign-by-spec', []), rule('x1', 'protocol-fix', ['../etc/passwd']),
  ]
  const w = L.partitionWave(fams, rulings)
  assert.deepEqual(w.work.map(x => [x.family.id, x.solo]), [['r1', true], ['r2', true], ['r3', true], ['m2', false]])
  assert.deepEqual(w.deferred.sort(), ['m1', 'm3', 'r4'])
  assert.deepEqual(w.unplaceable, ['x1'])
  assert.deepEqual(w.unruled, ['u1'])
})

test('stalled: no reduction over the window', () => {
  assert.equal(L.stalled([40, 12, 12, 12]), true)
  assert.equal(L.stalled([40, 12, 12]), false)
  assert.equal(L.stalled([40, 12, 13, 11]), false)
  assert.equal(L.stalled([5]), false)
})

test('closure runs: rule of three with a 10x margin, capped honestly', () => {
  assert.deepEqual(L.parseRate('7/200'), { k: 7, n: 200, p: 0.035 })
  assert.equal(L.parseRate('never'), null)
  assert.equal(L.parseRate('0/50'), null)
  assert.deepEqual(L.closureRuns('1/120'), { needed: 3600, cap: 5000, feasible: true, runs: 3600 })
  assert.deepEqual(L.closureRuns('1/1000'), { needed: 30000, cap: 5000, feasible: false, runs: 5000 })
  assert.equal(L.closureRuns('9/10').needed, 50)
  assert.equal(L.closureRuns(null).needed, 50)
})

test('checkGates: base rungs, plan rungs, consecutive >= 2, ids', () => {
  const ok = { milestone: 'M0', consecutive: 2, rungs: ['factory', 'build', 'syntax', 'discipline', 'invariants', 'claims', 'full', 'unit', 'chaos'], pinned: [{ id: 'smoke', cmd: 'x' }], campaigns: [{ id: 'rust-tsan' }], fuzz: [] }
  assert.deepEqual(L.checkGates(ok, { milestone: 'M0', planRungs: ['unit', 'chaos'] }), [])
  const bad = L.checkGates({ ...ok, milestone: 'M1', consecutive: 1, rungs: ['factory', 'ghost'], pinned: [{ id: 'A', cmd: '' }], campaigns: [{ id: 'x y' }] }, { milestone: 'M0', planRungs: ['oracle'] })
  assert.ok(bad.some(e => /milestone must be "M0"/.test(e)))
  assert.ok(bad.some(e => /consecutive must be 2/.test(e)))
  assert.ok(bad.some(e => /unknown rung "ghost"/.test(e)))
  assert.ok(bad.some(e => /rung build is required/.test(e)))
  assert.ok(bad.some(e => /rung oracle is required/.test(e)))
  assert.ok(bad.some(e => /bad id "A"/.test(e)))
  assert.ok(bad.some(e => /bad campaign id/.test(e)))
})
