import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loadLib } from '../sim.mjs'

const { normalizeItems } = loadLib(['hygiene', 'dag', 'triage'])
const knownRungs = ['unit', 'lint']
const item = (id, scope, extra = {}) => ({ id, rung: 'unit', symptom: `s-${id}`, evidence: `e-${id}`, scope, ...extra })

test('unsafe or empty scopes are dropped and reported', () => {
  const r = normalizeItems([item('a', ['../etc/x']), item('b', []), item('c', ['src/c.py'])], { knownRungs })
  assert.deepEqual(r.items.map(i => i.id), ['c'])
  assert.deepEqual(r.dropped.map(d => d.id), ['a', 'b'])
})

test('ids become tokens, unknown rungs become "unknown", scopes are deduped and sorted', () => {
  const r = normalizeItems([item('fix `this`!', ['src/b.py', 'src/a.py', 'src/a.py'], { rung: 'rm -rf' })], { knownRungs })
  assert.deepEqual(r.items[0].id, 'fix-this')
  assert.equal(r.items[0].rung, 'unknown')
  assert.deepEqual(r.items[0].scope, ['src/a.py', 'src/b.py'])
})

test('items with overlapping scopes are merged transitively', () => {
  const r = normalizeItems([
    item('a', ['src/net/']),
    item('b', ['src/net/peer.py', 'src/x.py']),
    item('c', ['src/x.py']),
    item('d', ['src/other.py']),
  ], { knownRungs })
  assert.deepEqual(r.items.map(i => i.id), ['a-b-c', 'd'])
  assert.deepEqual(r.items[0].mergedFrom, ['a', 'b', 'c'])
  assert.deepEqual(r.items[0].scope, ['src/net/', 'src/net/peer.py', 'src/x.py'])
  assert.match(r.items[0].evidence, /\[a\] e-a/)
})

test('duplicate ids are disambiguated and the list is capped', () => {
  const r = normalizeItems([item('x', ['a.py']), item('x', ['b.py']), item('x', ['c.py'])], { knownRungs, cap: 2 })
  assert.deepEqual(r.items.map(i => i.id), ['x', 'x-2'])
  assert.equal(r.overflow, 1)
})

test('one dominant mechanism: a round fixes only it when it explains more than half of the failures', () => {
  const { dominantMechanism } = loadLib(['hygiene', 'dag', 'triage'])
  const exc = (id, scope) => item(id, scope, { mechanism: 'exception scope anchored on the spawning stack' })
  const { items } = normalizeItems([exc('a', ['src/a.py']), exc('b', ['src/b.py']), exc('c', ['src/c.py']), item('d', ['src/d.py'], { mechanism: 'stack limit' })], { knownRungs })
  const d = dominantMechanism(items)
  assert.equal(d.focus, 'exception scope anchored on the spawning stack')
  assert.deepEqual([d.items.map(i => i.id), d.deferred.map(i => i.id), d.explained, d.of], [['a', 'b', 'c'], ['d'], 3, 4])
  // merged items weigh as many failures as they merged; a split vote keeps everything
  const merged = normalizeItems([exc('a', ['src/x/']), exc('b', ['src/x/y.py']), item('c', ['src/c.py'], { mechanism: 'other' }), item('e', ['src/e.py'], { mechanism: 'third' })], { knownRungs }).items
  assert.equal(dominantMechanism(merged).focus, null, '2 of 4 is not more than half')
  assert.equal(dominantMechanism(normalizeItems([exc('a', ['src/a.py']), exc('b', ['src/b.py'])], { knownRungs }).items).focus, null, 'too few failures to call one dominant')
  assert.equal(dominantMechanism(normalizeItems([item('a', ['src/a.py']), item('b', ['src/b.py']), item('c', ['src/c.py'])], { knownRungs }).items).focus, null, 'no mechanism named')
})
