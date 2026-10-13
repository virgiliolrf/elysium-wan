import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, ladder, assertRoles } from './helpers.mjs'

// The agent that runs the ladder fixes what it finds (the human's rule, 2026-10-09): one solo coder per round, no proposer and
// no voters.
const script = built('wan-fix')
const green = () => ({ allGreen: true, ladder: ladder(true), fixed: [], remaining: [] })
const red = (fixed = [], failing = ['unit']) => ({ allGreen: false, ladder: ladder(false, failing), fixed, remaining: [{ rung: failing[0], symptom: 's', evidence: 'e' }] })
const fix = (mechanism, files = ['src/a.py']) => ({ mechanism, files, evidence: 'uv run pytest -q tests/unit/test_a.py -> 1 passed' })

test('green at the first round: one agent, nothing else', async () => {
  const { result, calls } = await simulate(script, { respond: byLabel({ 'fix:': green }) })
  assert.deepEqual([result.ok, result.rounds], [true, 1])
  assert.deepEqual(calls.map(c => c.opts.label), ['fix:r1'])
})

test('the agent that finds the failures fixes them: a coder, told never to mask a failure, with the author\'s context', async () => {
  const { result, calls } = await simulate(script, { respond: byLabel({ 'fix:': (p, o, i, n) => (n === 0 ? red([fix('lock order')]) : green()) }) })
  assert.deepEqual([result.ok, result.rounds], [true, 2])
  assertRoles(assert, calls, { 'fix:': 'wan-coder' })
  const prompt = calls[0].prompt
  for (const want of ["You find the ladder's failures AND fix them yourself", 'Never weaken, skip, delete or loosen a test', 'ledger.mjs owner --path', 'paste THAT run\'s JSON']) assert.ok(prompt.includes(want), want)
  assert.ok(calls[1].prompt.includes('lock order'), 'the next round starts from what this round fixed and left red')
  assert.deepEqual(result.fixed.map(f => [f.round, f.mechanism]), [[1, 'lock order']])
  assert.ok(!calls.some(c => /^(propose|vote|apply|triage|verify):/.test(c.opts.label)), 'no proposer, no voters')
})

test('an agent that claims green while the pasted ladder is red is not believed', async () => {
  const { result, logs } = await simulate(script, { respond: byLabel({ 'fix:': (p, o, i, n) => (n === 0 ? { ...red(), allGreen: true } : green()) }) })
  assert.equal(result.rounds, 2)
  assert.ok(logs.some(l => l.includes('claimed green but the pasted ladder JSON is red')))
})

test('never green: stops at the round budget and returns the last report; a milestone run asks for 8 fail-fast rounds', async () => {
  const plain = await simulate(script, { respond: byLabel({ 'fix:': () => red() }) })
  assert.deepEqual([plain.result.ok, plain.result.rounds], [false, 4])
  assert.ok(plain.result.lastReport.remaining.length)
  assert.ok(!plain.calls[0].prompt.includes('--fail-fast'))
  const fast = await simulate(script, { args: { rungs: ['unit', 'chaos', 'bogus'], maxRounds: 8, failFast: true }, respond: byLabel({ 'fix:': () => red() }) })
  assert.equal(fast.result.rounds, 8)
  assert.ok(fast.calls[0].prompt.includes('ladder.mjs --json --only unit,chaos --fail-fast'), 'only known rungs, fail-fast')
  const outOfRange = await simulate(script, { args: { maxRounds: 99 }, respond: byLabel({ 'fix:': () => red() }) })
  assert.equal(outOfRange.result.rounds, 4)
})

test('a spec conflict met while fixing does not stop the loop: it is returned for the human', async () => {
  const conflict = { ...red([fix('pin')]), specConflict: { description: 'the handout pins steps', evidence: 'HANDOUT §8', proposedChange: 'the SPEC of record lets a resume extend steps' } }
  const { result } = await simulate(script, { respond: byLabel({ 'fix:': (p, o, i, n) => (n === 0 ? conflict : green()) }) })
  assert.equal(result.ok, true)
  assert.deepEqual(result.deviations.map(d => d.implemented), ['the SPEC of record lets a resume extend steps'])
})

test('an agent that produces nothing stops the loop instead of looping', async () => {
  await assert.rejects(simulate(script, { respond: byLabel({ 'fix:': () => null }) }), /the fix agent produced nothing/)
})
