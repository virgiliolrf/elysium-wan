import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loadLib, makeRuntime } from '../sim.mjs'
import { finding } from './helpers.mjs'

const lensesOf = n => Array.from({ length: n }, (_, i) => [`l${i}`, `lens ${i}`])

function lib() {
  const { runtime, logs } = makeRuntime()
  return { ...loadLib(['loop'], { parallel: runtime.parallel, log: runtime.log }), logs }
}

const MINOR = { lens: 'l0', severity: 'minor', location: 'src/x.py:1', problem: 'p' }

test('converges on the first round with zero serious findings; its validated minors are returned open, not fixed', async () => {
  const { reviewUntilClean } = lib()
  let fixes = 0
  const r = await reviewUntilClean({
    label: 'doc', lenses: lensesOf(3), maxRounds: 4,
    review: async name => ({ findings: name === 'l0' ? [finding('minor'), finding('minor', { validatedAgainstCurrentCode: false })] : [] }),
    fix: async () => { fixes++ },
  })
  assert.deepEqual(r, { converged: true, rounds: 1, findingsPerRound: [0], discarded: 0, minors: [MINOR], house: [] })
  assert.equal(fixes, 0)
})

test('validated minors ride along with the serious findings of a round, after them, so they do not come back as majors', async () => {
  const { reviewUntilClean } = lib()
  const fixed = []
  const r = await reviewUntilClean({
    label: 'doc', lenses: lensesOf(2), maxRounds: 4,
    review: async (name, lens, round) => ({ findings: round > 1 ? [] : name === 'l0' ? [finding('minor', { problem: 'm' })] : [finding('major'), finding('minor', { reproduction: '' })] }),
    fix: async findings => { fixed.push(findings.map(f => [f.severity, f.lens])) },
  })
  assert.equal(r.converged, true)
  assert.deepEqual(fixed, [[['major', 'l1'], ['minor', 'l0']]], 'serious first; an unproven minor is not fixed')
  assert.deepEqual(r.minors, [])
})

test('a "house" finding is the factory backlog only when the caller names its own files and the finding is outside them', async () => {
  const { reviewUntilClean } = lib()
  const house = finding('major', { scope: 'house', location: 'agents-architetura/tools/wf/lib/loop.js:3' })
  const mislabeled = finding('major', { scope: 'house', location: 'src/x.py:9' })
  const run = (findings, ownsFile) => {
    const fixed = []
    return reviewUntilClean({
      label: 'doc', lenses: lensesOf(1), ...(ownsFile ? { ownsFile } : {}),
      review: async (n, l, round) => ({ findings: round === 1 ? findings : [] }),
      fix: async f => { fixed.push(...f) },
    }).then(r => ({ r, fixed }))
  }
  const owns = f => f.startsWith('src/')
  const a = await run([house], owns)
  assert.deepEqual([a.r.converged, a.r.rounds, a.fixed.length], [true, 1, 0])
  assert.deepEqual(a.r.house, [{ lens: 'l0', severity: 'major', location: 'agents-architetura/tools/wf/lib/loop.js:3', problem: 'p' }])
  const b = await run([mislabeled], owns)
  assert.deepEqual([b.r.rounds, b.fixed.length, b.r.house.length], [2, 1, 0], 'a file the change owns is never house')
  const c = await run([house])
  assert.deepEqual([c.r.rounds, c.fixed.length], [2, 1], 'without ownsFile nothing is house')
})

test('a review that does not converge returns the open blocker/major findings of its last round for the human', async () => {
  const { reviewUntilClean } = lib()
  const r = await reviewUntilClean({
    label: 'doc', lenses: lensesOf(1), maxRounds: 5,
    // still falling, never clean: the budget ends it, not the stall rule
    review: async (n, l, round) => ({ findings: Array.from({ length: 6 - round }, (_, i) => finding('major', { problem: `p${round}.${i}` })) }),
    fix: async () => {},
  })
  assert.deepEqual([r.converged, r.stalled, r.rounds], [false, undefined, 5])
  assert.deepEqual(r.open, [{ lens: 'l0', severity: 'major', location: 'src/x.py:1', problem: 'p5.0' }])
})

test('a review whose blocker/major count stops falling for three rounds is stalled: it stops for a split instead of more rounds', async () => {
  const { reviewUntilClean, logs } = lib()
  let fixes = 0
  const counts = [17, 17, 19, 3, 1]
  const r = await reviewUntilClean({
    label: 'adr', lenses: lensesOf(1),
    review: async (n, l, round) => ({ findings: Array.from({ length: counts[round - 1] }, () => finding('major')) }),
    fix: async () => { fixes++ },
  })
  assert.deepEqual([r.converged, r.stalled, r.rounds, r.findingsPerRound, fixes], [false, true, 3, [17, 17, 19], 2])
  assert.equal(r.open.length, 19)
  assert.ok(logs.some(l => l.includes('review STALLED (17 -> 17 -> 19)')))
  // a dip breaks the trend: 7 -> 6 -> 6 -> 1 -> 1 is converging, not stalled
  const m2 = [7, 6, 6, 1, 1]
  const ok = await reviewUntilClean({ label: 'm2', lenses: lensesOf(1), review: async (n, l, round) => ({ findings: Array.from({ length: m2[round - 1] }, () => finding('major')) }), fix: async () => {} })
  assert.deepEqual([ok.stalled, ok.rounds], [undefined, 5])
})

test('fixes between rounds and re-reviews from scratch', async () => {
  const { reviewUntilClean } = lib()
  const fixed = []
  const r = await reviewUntilClean({
    label: 'doc', lenses: lensesOf(2), maxRounds: 4,
    review: async (name, lens, round) => ({ findings: round === 1 && name === 'l0' ? [finding('blocker')] : [] }),
    fix: async (serious, round) => { fixed.push([serious.length, round]) },
  })
  assert.deepEqual(r, { converged: true, rounds: 2, findingsPerRound: [1, 0], discarded: 0, minors: [], house: [] })
  assert.deepEqual(fixed, [[1, 1]])
})

test('a missing reviewer is never a clean pass', async () => {
  const { reviewUntilClean, logs } = lib()
  const r = await reviewUntilClean({
    label: 'doc', lenses: lensesOf(3), maxRounds: 2,
    review: async name => (name === 'l2' ? null : { findings: [] }),
    fix: async () => { throw new Error('must not fix') },
  })
  assert.deepEqual(r, { converged: false, rounds: 2, findingsPerRound: Array(2).fill('missing:1'), discarded: 0, minors: [], house: [], open: [] })
  assert.ok(logs.some(l => l.includes('only 2/3 reviewers reported')))
})

test('a reviewer that throws counts as missing', async () => {
  const { reviewUntilClean } = lib()
  const r = await reviewUntilClean({
    label: 'doc', lenses: lensesOf(2), maxRounds: 1,
    review: async name => { if (name === 'l1') throw new Error('agent error'); return { findings: [] } },
    fix: async () => {},
  })
  assert.equal(r.converged, false)
})

test('tallyVotes: majority, quorum, unanimity', () => {
  const { tallyVotes } = lib()
  assert.deepEqual(tallyVotes([{ approve: true }, { approve: true }, { approve: false }], 3),
    { expected: 3, received: 3, approvals: 2, quorum: true, majority: true, unanimous: false })
  assert.equal(tallyVotes([{ approve: true }, null, null], 3).majority, false)
  assert.equal(tallyVotes([{ approve: true }, { approve: true }], 2).unanimous, true)
})

test('blocker/major findings not validated against the current code are discarded, never fixed', async () => {
  const { reviewUntilClean, logs } = lib()
  let fixes = 0
  const r = await reviewUntilClean({
    label: 'doc', lenses: lensesOf(2), maxRounds: 2,
    review: async () => ({ findings: [finding('blocker', { validatedAgainstCurrentCode: false }), finding('major', { reproduction: '' }), finding('major', { location: ' ' })] }),
    fix: async () => { fixes++ },
  })
  assert.deepEqual(r, { converged: true, rounds: 1, findingsPerRound: [0], discarded: 6, minors: [], house: [] })
  assert.equal(fixes, 0)
  assert.ok(logs.some(l => l.includes('discarded 6 blocker/major finding(s) not validated')))
})

test('the review budget is 5 rounds by default; a milestone run asks for 2, and the last round is fixed too', async () => {
  const { reviewUntilClean } = lib()
  // a falling count that never reaches zero: only the budget ends it
  const falling = async (n, l, round) => ({ findings: Array.from({ length: 10 - round }, () => ({ severity: 'major', location: 'x:1', evidence: 'e', reproduction: 'r', problem: 'p', validatedAgainstCurrentCode: true })) })
  const count = async maxRounds => { let fixes = 0; const r = await reviewUntilClean({ label: 'doc', lenses: lensesOf(1), ...(maxRounds ? { maxRounds } : {}), review: falling, fix: async () => { fixes++ } }); return [r.rounds, fixes, r.open.length] }
  assert.deepEqual(await count(), [5, 5, 5])
  assert.deepEqual(await count(2), [2, 2, 8], 'two reviews, two fix rounds; the second round\'s findings are fixed and listed in open')
  assert.deepEqual(await count(6), [6, 6, 4])
  await assert.rejects(reviewUntilClean({ label: 'doc', lenses: lensesOf(1), maxRounds: 0, review: falling, fix: async () => {} }), /maxRounds must be a positive integer/)
})

test('fixAll: every validated minor rides along, not only the first eight', async () => {
  const { reviewUntilClean } = lib()
  const f = (severity, i) => ({ severity, location: `x:${i}`, evidence: 'e', reproduction: 'r', problem: `p${i}`, validatedAgainstCurrentCode: true })
  const review = async (n, l, round) => ({ findings: round === 1 ? [f('major', 0), ...Array.from({ length: 12 }, (_, i) => f('minor', i + 1))] : [] })
  const sent = []
  await reviewUntilClean({ label: 'plan', lenses: lensesOf(1), maxRounds: 2, fixAll: true, review, fix: async items => { sent.push(items.length) } })
  assert.deepEqual(sent, [13])
  sent.length = 0
  await reviewUntilClean({ label: 'plan', lenses: lensesOf(1), maxRounds: 2, review, fix: async items => { sent.push(items.length) } })
  assert.deepEqual(sent, [9], 'without fixAll at most 8 minors ride along')
})

test('every fix sees the earlier rounds; a final check runs once over every finding when the loop did not converge', async () => {
  const { reviewUntilClean } = lib()
  const f = (sev, i) => ({ severity: sev, location: `x:${i}`, evidence: 'e', reproduction: 'r', problem: `p${i}`, validatedAgainstCurrentCode: true })
  const review = async (n, l, round) => ({ findings: [f('major', round)] })
  const earlier = []
  let checked = null
  const r = await reviewUntilClean({ label: 'plan', lenses: lensesOf(1), maxRounds: 2, review, fix: async (items, round, prev) => { earlier.push(prev.map(x => x.problem)) }, finalCheck: async all => { checked = all; return { items: [{ n: 1, status: 'resolved', evidence: 'a' }, { n: 2, status: 'still-open', evidence: 'b' }] } } })
  assert.deepEqual(earlier, [[], ['p1']])
  assert.deepEqual(checked.map(x => [x.n, x.round, x.problem]), [[1, 1, 'p1'], [2, 2, 'p2']])
  assert.deepEqual(r.finalCheck, { resolved: 1, 'fixed-now': 0, refuted: 0, 'still-open': 1 })
  assert.deepEqual(r.open.map(x => x.problem), ['p2'])
  let called = false
  const clean = await reviewUntilClean({ label: 'plan', lenses: lensesOf(1), maxRounds: 2, review: async () => ({ findings: [] }), fix: async () => {}, finalCheck: async () => { called = true } })
  assert.equal(clean.converged, true)
  assert.equal(called, false, 'a converged review needs no final check')
})
