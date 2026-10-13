import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildScript } from '../build.mjs'
import { simulate, byLabel } from '../sim.mjs'
import { approved, assertRoles, finding } from './helpers.mjs'

// The house template is what every wan-x phase starts from, and runPhase (lib/phase.js) is what it runs: its flow is tested
// like a house workflow, against a simulated repository whose tree changes only when an agent writes.
const TEMPLATE = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'docs', 'templates', 'wan-x-workflow.js'), 'utf8')
const build = src => buildScript(src, { name: 'wan-x-example' }).script
const script = build(TEMPLATE)
const DOCS = ['docs/design/NEXT-example.md', 'docs/design/GATES-example.json', '.claude/workflows/wan-x-example.js']
const GATE_IDS = ['rung:oracle', 'rung:identity', 'rung:full', 'pinned:step']
const FILE = 'src/example/a/x.py'

// Tree ids are 40 hex digits; a write moves the tree, and a check reports what changed since a snapshot and what fell
// outside the scopes it was given, exactly like scope-audit.mjs.
function repo() {
  let n = 0
  const writes = []
  const id = i => String(i).padStart(40, '0')
  return {
    tree: () => id(n),
    write: files => { n++; writes.push({ at: n, files }) },
    snapshot: () => ({ tree: id(n) }),
    check: prompt => {
      const since = parseInt(/--since (\d+)/.exec(prompt)[1], 10)
      const scopes = JSON.parse(/--scopes '(.*)'/.exec(prompt)[1])
      const changed = [...new Set(writes.filter(w => w.at > since).flatMap(w => w.files))].sort()
      return { tree: id(n), changed, violations: changed.filter(p => !scopes.some(s => p === s || (s.endsWith('/') && p.startsWith(s)))) }
    },
  }
}

const pinned = (r, { failing = [], ...extra } = {}) => ({
  milestone: 'example',
  verdict: failing.length ? 'NOT VERIFIED' : 'VERIFIED',
  failing,
  gates: GATE_IDS.map(id => ({ id, ok: !failing.includes(id) })),
  tree: r.tree(),
  ...extra,
})
const item = (id, scope = [FILE], extra = {}) => ({ id, rung: 'pinned:step', symptom: 's', evidence: 'e', scope, ...extra })
const green = r => () => ({ pinned: pinned(r), items: [] })
// red for the first `times` verifies, then green
const redThenGreen = (r, times, it = item('step')) => (p, o, i, n) => (n < times ? { pinned: pinned(r, { failing: ['pinned:step'] }), items: [it] } : { pinned: pinned(r), items: [] })

function run(extra = {}, { src = script, r = repo() } = {}) {
  const writer = files => () => { r.write(files); return { summary: 'ok', files, evidence: [] } }
  const handlers = {
    'gate:approval': () => approved(DOCS),
    'audit:snapshot:': () => r.snapshot(),
    'audit:check:': p => r.check(p),
    'impl:': writer([FILE]),
    'review:': () => ({ findings: [] }),
    'review-fix:': writer([FILE]),
    'propose:': () => ({ fix: 'f', rationale: 'r' }),
    'vote:': () => ({ approve: true, reasons: 'ok' }),
    'apply:': writer([FILE]),
    'stabilize-review:': () => ({ findings: [] }),
    'stabilize-review-fix:': writer([FILE]),
    'verify:': green(r),
    'verify-wait:': () => ({ status: 'finished' }),
    ...(typeof extra === 'function' ? extra(r, writer) : extra),
  }
  return simulate(src, { respond: byLabel(handlers) }).then(out => ({ ...out, r }))
}
const labels = (calls, prefix) => calls.filter(c => c.opts.label.startsWith(prefix)).map(c => c.opts.label)

test('VERIFIED only when the relayed pinned JSON is this phase\'s, complete, green and of the audited tree', async () => {
  const { result, calls } = await run()
  assert.equal(result.ok, true, result.next)
  assert.equal(result.stabilizeRounds, 0)
  assert.ok(calls.find(c => c.opts.label === 'verify:r0').prompt.includes('Run exactly: node agents-architetura/tools/gates/pinned.mjs example --json'))
  assert.ok(calls[0].opts.label === 'gate:approval' && DOCS.every(d => calls[0].prompt.includes(d)))
  for (const [why, lie] of [
    ['a VERIFIED verdict with a failing gate', { failing: [] , gates: GATE_IDS.map(id => ({ id, ok: id !== 'pinned:step' })) }],
    ['a verdict that leaves out one of the gate ids', { gates: GATE_IDS.slice(1).map(id => ({ id, ok: true })) }],
    ['another phase\'s verdict', { milestone: 'other' }],
    ['a verdict of another tree', { tree: 'f'.repeat(40) }],
  ]) {
    const claimed = await run(rr => ({ 'verify:': () => ({ pinned: pinned(rr, lie), items: [] }) }))
    assert.equal(claimed.result.ok, false, why)
  }
})

test('a red gate is stabilized like the reference PR: proposal, 3 votes, scoped apply, verify again', async () => {
  const { result, calls } = await run(r => ({ 'verify:': redThenGreen(r, 2) }))
  assert.equal(result.ok, true, result.next)
  assert.equal(result.stabilizeRounds, 2)
  assert.deepEqual(labels(calls, 'verify:'), ['verify:r0', 'verify:r1', 'verify:r2'])
  assert.equal(labels(calls, 'vote:').length, 6)
  assert.ok(calls.find(c => c.opts.label.startsWith('propose:')).prompt.includes('FACTS OF THIS PHASE'), 'every fix prompt carries the facts')
  assert.ok(calls.find(c => c.opts.label.startsWith('propose:')).prompt.includes('read its journal .factory/runs/example/tasks/part-a.md FIRST'), 'a fix starts from its writer\'s journal')
  assertRoles(assert, calls, { 'verify:': 'wan-verifier', 'propose:': 'wan-proposer', 'vote:': 'wan-reviewer', 'apply:': 'wan-coder', 'impl:': 'wan-coder' })
})

test('a pass on a tree that already failed is intermittent, never VERIFIED', async () => {
  // the stabilize item changes nothing, so verify r1 runs on the same tree as the failing r0
  const { result } = await run(r => ({ 'verify:': redThenGreen(r, 1), 'apply:': () => ({ summary: 'nothing to change', files: [], evidence: [] }) }))
  assert.equal(result.ok, false)
  assert.deepEqual(result.intermittent, [{ round: 0, failing: ['pinned:step'] }])
  assert.match(result.next, /intermittent/)
})

test('at most 4 stabilize rounds; items outside the phase owned paths are never applied; a test fix needs every vote', async () => {
  const red = await run(r => ({ 'verify:': () => ({ pinned: pinned(r, { failing: ['pinned:step'] }), items: [item('step')] }) }))
  assert.equal(red.result.ok, false)
  assert.equal(red.result.stabilizeRounds, 4)
  assert.equal(labels(red.calls, 'verify:').length, 5)
  const outside = await run(r => ({ 'verify:': () => ({ pinned: pinned(r, { failing: ['pinned:step'] }), items: [item('wide', ['src/other/'])] }) }))
  assert.equal(outside.result.ok, false)
  assert.ok(!outside.calls.some(c => c.opts.label.startsWith('apply:') || c.opts.label.startsWith('propose:')))
  assert.ok(outside.logs.some(l => l.includes('no item is inside this phase')))
  // a phase that owns tests/: a fix there with one dissenting vote is rejected
  const withTests = build(TEMPLATE.replace("owns: ['src/example/a/']", "owns: ['src/example/a/', 'tests/unit/test_a.py']"))
  const split = await run(r => ({
    'verify:': redThenGreen(r, 1, item('t', ['tests/unit/test_a.py'])),
    'vote:': (p, o) => ({ approve: !o.label.endsWith(':regression'), reasons: 'r' }),
  }), { src: withTests })
  assert.ok(!split.calls.some(c => c.opts.label.startsWith('apply:')), 'a 2/3 vote does not apply a fix under tests/')
  assert.ok(split.calls.find(c => c.opts.label.startsWith('propose:')).prompt.includes('applied only on 3 of 3 approvals'))
})

test('a write outside the owned paths, a writer reporting another writer\'s files, or a spec conflict stops the phase before any verify', async () => {
  const escaped = run({ 'impl:': (p, o) => o && { summary: 's', files: ['agents-architetura/x.js'], evidence: [] } }, {})
  await assert.rejects(escaped, /writer part-a reports files outside its own paths: agents-architetura\/x\.js/)
  // its own journal, reported relatively or absolutely, is not outside (M2 phase 1 stopped on exactly this)
  const journaled = await run({ 'impl:': (p, o) => ({ summary: 's', files: ['.factory/runs/example/tasks/part-a.md', '/abs/repo/.factory/runs/example/tasks/part-a.md'], evidence: [] }) })
  assert.equal(journaled.result.ok, true, journaled.result.next)
  await assert.rejects(run({ 'impl:': () => ({ summary: 's', files: ['.factory/runs/example/tasks/other.md'], evidence: [] }) }), /reports files outside its own paths: \.factory\/runs\/example\/tasks\/other\.md/)
  const wrote = run((r, writer) => ({ 'impl:': writer(['agents-architetura/x.js']) }))
  await assert.rejects(wrote, /implement-wave1 wrote outside its owned paths: agents-architetura\/x\.js/)
  const conflict = await run({ 'impl:': () => ({ summary: 's', files: [], evidence: [], specConflict: { description: 'SPEC says X', evidence: 'e', proposedChange: 'Y' } }) })
  assert.deepEqual([conflict.result.ok, conflict.result.stoppedAt], [false, 'implement'])
  assert.match(conflict.result.next, /\/wan-adr/)
  assert.ok(!conflict.calls.some(c => c.opts.label.startsWith('verify:')))
})

test('writers run in waves: a later wave starts after the earlier one, is told what it reported, and is audited on its own', async () => {
  const twoWaves = build(TEMPLATE.replace(
    "{ key: 'part-a', owns: ['src/example/a/'], task: '<the change, in terms of the facts above>', binding: ['BA-1'] },",
    "{ key: 'part-a', owns: ['src/example/a/'], task: 'a', binding: ['BA-1'] },\n    { key: 'part-b', owns: ['src/example/a/'], task: 'b', wave: 2, writer: 'part-a' },"))
  const { result, calls } = await run({}, { src: twoWaves })
  assert.equal(result.ok, true, result.next)
  const order = calls.map(c => c.opts.label).filter(l => /^(impl:|audit:check:implement)/.test(l))
  assert.deepEqual(order, ['impl:part-a', 'audit:check:implement-wave1', 'impl:part-b', 'audit:check:implement-wave2'])
  assert.ok(calls.find(c => c.opts.label === 'impl:part-b').prompt.includes('earlier_waves'))
  assert.ok(calls.find(c => c.opts.label === 'impl:part-b').prompt.includes('.factory/runs/example/tasks/part-a.md'), 'items of one writer share its journal')
  const overlap = build(TEMPLATE.replace("binding: ['BA-1'] },", "binding: ['BA-1'] },\n    { key: 'part-b', owns: ['src/example/a/b.py'], task: 'b' },"))
  await assert.rejects(run({}, { src: overlap }), /part-a and part-b run in the same wave on overlapping paths/)
})

test('review fixes are scope-audited, carry the minors, and a review that does not converge still verifies but is not ok', async () => {
  const { result, calls } = await run({ 'review:': (p, o) => ({ findings: o.label.startsWith('review:objective:') ? [finding('major', { location: `${FILE}:1` }), finding('minor', { location: `${FILE}:2` })] : [] }) })
  // one major every round: the count never falls, so the review stalls after three rounds (two fixed)
  assert.deepEqual([result.review.converged, result.review.stalled], [false, true])
  assert.equal(labels(calls, 'audit:check:review-fix-r').length, 2)
  assert.ok(calls.find(c => c.opts.label === 'review-fix:r1').prompt.includes(`${FILE}:2`), 'the minor rides along')
  assert.ok(calls.some(c => c.opts.label === 'verify:r0'), 'Verify still runs: the human decides with the gate\'s evidence')
  assert.equal(result.ok, false)
  assert.match(result.next, /review stalled[\s\S]*split it/)
  assert.equal(result.open.length, 1)
})

test('a defect in shared code is returned as house, never blocks the phase', async () => {
  const house = finding('major', { scope: 'house', location: 'agents-architetura/tools/wf/lib/fix.js:40' })
  const { result, calls } = await run({ 'review:': (p, o, i, n) => ({ findings: n === 0 ? [house] : [] }) })
  assert.equal(result.ok, true, result.next)
  assert.ok(!calls.some(c => c.opts.label.startsWith('review-fix:')))
  assert.equal(result.house.length, 1)
})

test('code changed by Stabilize is reviewed before VERIFIED counts; a fix there runs the pinned gate again', async () => {
  const flow = stabilizeReview => run(r => ({ 'verify:': redThenGreen(r, 1), 'stabilize-review:': stabilizeReview }))
  const clean = await flow(() => ({ findings: [] }))
  assert.equal(clean.result.ok, true, clean.result.next)
  assert.ok(clean.calls.find(c => c.opts.label === 'stabilize-review:tests:r1').prompt.includes(FILE))
  const fixed = await flow((p, o, i, n) => ({ findings: n === 0 ? [finding('major', { location: `${FILE}:1` })] : [] }))
  assert.equal(fixed.result.ok, true, fixed.result.next)
  assert.deepEqual(labels(fixed.calls, 'verify:'), ['verify:r0', 'verify:r1', 'verify:r2'])
  assert.ok(fixed.calls.some(c => c.opts.label === 'audit:check:stabilize-review-fix-r1'))
})

test('fixes by the post-stabilize review stop on a write outside the owned paths and on a spec conflict', async () => {
  const flow = extra => run((r, writer) => ({
    'verify:': redThenGreen(r, 1),
    'stabilize-review:': (p, o, i, n) => ({ findings: n === 0 ? [finding('major', { location: `${FILE}:1` })] : [] }),
    ...extra(r, writer),
  }))
  await assert.rejects(flow((r, writer) => ({ 'stabilize-review-fix:': writer(['tests/oracle/x.py']) })), /stabilize-review-fix-r1 wrote outside its owned paths/)
  const conflict = await flow(() => ({ 'stabilize-review-fix:': () => ({ summary: 's', files: [], evidence: [], specConflict: { description: 'd', evidence: 'e', proposedChange: 'p' } }) }))
  assert.deepEqual([conflict.result.ok, conflict.result.stoppedAt], [false, 'stabilize-review'])
})

test('a precondition that fails stops the phase before any writer; requires adds documents to the grant', async () => {
  const pre = build(TEMPLATE.replace("slug: 'example',", "slug: 'example',\n  requires: ['docs/design/adr/ADR-0007-x.md'],\n  precondition: { cmd: 'test -f docs/design/adr/ADR-0007-x.md', why: 'the ADR is missing: run /wan-adr first' },"))
  const stopped = await run({ 'gate:approval': () => approved([...DOCS, 'docs/design/adr/ADR-0007-x.md']), 'gate:precondition': () => ({ exitCode: 1, output: 'no' }) }, { src: pre })
  assert.deepEqual([stopped.result.ok, stopped.result.stoppedAt, stopped.result.next], [false, 'precondition', 'the ADR is missing: run /wan-adr first'])
  assert.ok(!stopped.calls.some(c => c.opts.label.startsWith('impl:')))
  assert.ok(stopped.calls[0].prompt.includes('docs/design/adr/ADR-0007-x.md'))
  const notGranted = run({ 'gate:precondition': () => ({ exitCode: 0, output: '' }) }, { src: pre })
  await assert.rejects(notGranted, /are not all approved by a human/)
})

test('the script owns the pinned wait: slices until the detached gate finishes, then one verifier relays the stored verdict', async () => {
  // M2 phase 2's verifier started pinned.mjs in the background and ended before it finished: the verdict was lost
  const { result, calls } = await run({ 'verify-wait:': (p, o, i, n) => ({ status: n < 3 ? 'running' : 'finished' }) })
  assert.equal(result.ok, true, result.next)
  assert.deepEqual(labels(calls, 'verify-wait:'), ['verify-wait:r0:s0', 'verify-wait:r0:s1', 'verify-wait:r0:s2', 'verify-wait:r0:s3'])
  const slice = calls.find(c => c.opts.label === 'verify-wait:r0:s0').prompt
  assert.ok(slice.includes('pinned.mjs example --json --label r0 --wait 540') && slice.includes('600000 ms') && slice.includes('never in the background'))
  assert.ok(calls.find(c => c.opts.label === 'verify:r0').prompt.includes('pinned.mjs example --json --label r0 --wait 540'))
  assert.ok(labels(calls, '').indexOf('verify:r0') > labels(calls, '').indexOf('verify-wait:r0:s3'), 'the verifier runs after the gate finished')
  // each verify round is its own run: r1 waits on its own label
  const red = await run(r => ({ 'verify:': redThenGreen(r, 1) }))
  assert.deepEqual(labels(red.calls, 'verify-wait:'), ['verify-wait:r0:s0', 'verify-wait:r1:s0'])
  // a gate that never finishes stops the phase with where its run is, instead of relaying no verdict
  await assert.rejects(run({ 'verify-wait:': () => ({ status: 'running' }) }), /did not finish within 12 hours; it keeps running detached \(\.factory\/gates\/runs\/example\.r0\.log\)/)
})
