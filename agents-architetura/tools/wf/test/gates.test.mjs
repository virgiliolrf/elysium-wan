import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loadLib } from '../sim.mjs'

const { ladderCommand, ladderGreen, failingRungs, knownRungs } = loadLib(['project', 'gates'], { agent: () => null })

test('ladderCommand only ever interpolates known rung ids', () => {
  assert.equal(ladderCommand(), 'node agents-architetura/tools/gates/ladder.mjs --json')
  assert.equal(ladderCommand({ only: ['unit', 'x; rm -rf ~'], require: ['oracle'], upto: 'evil' }),
    'node agents-architetura/tools/gates/ladder.mjs --json --only unit --require oracle')
  assert.equal(ladderCommand({ upto: 'types' }), 'node agents-architetura/tools/gates/ladder.mjs --json --upto types')
  assert.equal(ladderCommand({ reuse: true }), 'node agents-architetura/tools/gates/ladder.mjs --json --reuse')
  assert.deepEqual(knownRungs('unit'), [])
})

test('ladderGreen requires the claim and the pasted ladder to agree', () => {
  assert.equal(ladderGreen({ allGreen: true, ladder: { allGreen: true } }), true)
  assert.equal(ladderGreen({ allGreen: true, ladder: { allGreen: false } }), false)
  assert.equal(ladderGreen({ allGreen: false, ladder: { allGreen: true } }), false)
  assert.equal(ladderGreen(null), false)
})

test('failingRungs lists failing known rungs only', () => {
  assert.deepEqual(failingRungs({ rungs: [{ rung: 'unit', status: 'fail' }, { rung: 'lint', status: 'pass' }, { rung: 'bogus', status: 'fail' }] }), ['unit'])
  assert.deepEqual(failingRungs(undefined), [])
})
