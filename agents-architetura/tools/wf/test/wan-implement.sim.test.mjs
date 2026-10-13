import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, ladder, finding, planTask, assertRoles, approved, servePlan } from './helpers.mjs'

const script = built('wan-implement')
const ok = () => ({ summary: 'ok', files: [] })
const done = () => ({ summary: 'ok', files: ['f'], evidence: [{ claim: 'tests pass', command: 'uv run pytest -q x', exitCode: 0, output: '1 passed' }] })
const WS = [{ key: 'transport', title: 't', scope: 's', ownedPaths: ['src/transport/', 'tests/unit/transport/'] }]
const PLAN = { milestone: 'M0', tasks: [planTask('wire'), planTask('codec'), planTask('peer', { deps: ['wire', 'codec'], rungs: ['unit', 'integration'] })] }
const PLAN_DOCS = ['docs/design/HANDOUT-M0.md', 'docs/design/PLAN-M0.json', 'docs/design/GATES-M0.json']
const base = {
  'gate:approval': () => approved(),
  'gate:approval:plan': () => approved(PLAN_DOCS),
  'extract:workstreams': () => ({ workstreams: WS }),
  'extract:plan': servePlan(PLAN),
  'ledger:': () => ({ exitCode: 0 }),
  'audit:ledger': () => ({ exitCode: 0 }),
  'task:': done, 'task-review:': () => ({ findings: [] }),
  'integrate': done,
  'build:': () => ({ ladder: ladder(true), files: [] }),
  'review:': () => ({ findings: [] }), 'review-final-check': () => ({ items: [] }),
  'extract:specialists': () => ({ specialists: [], rejected: [] }),
}
const run = (overrides = {}, opts = {}) => {
  const fixArgs = []
  return simulate(script, {
    args: { milestone: 'M0' },
    respond: byLabel({ ...base, ...overrides }),
    workflows: { 'wan-fix': a => { fixArgs.push(a); return { ok: true, green: true, rounds: 1 } } },
    ...opts,
  }).then(r => ({ ...r, fixArgs }))
}
const decodeLedger = prompt => JSON.parse(Buffer.from(prompt.match(/--b64 (\S+)/)[1], 'base64').toString('utf8'))
const statuses = result => Object.fromEntries(result.tasks.tasks.map(t => [t.id, t.status]))

test('happy path: DAG waves, every stage routed to its role, ledger persisted, wan-fix on the milestone rungs', async () => {
  const { result, calls, fixArgs } = await run()
  assert.equal(result.ok, true)
  assert.deepEqual(result.tasks.tasks.map(t => [t.id, t.status, t.waves]), [['codec', 'done', [1]], ['wire', 'done', [1]], ['peer', 'done', [2]]])
  const order = calls.map(c => c.opts.label).filter(l => /^(audit|ledger|task|integrate|build)/.test(l) && !l.startsWith('task-review'))
  // like the reference PR: no per-wave snapshot, verification or scope audit; everything is checked at the end. The ledger is
  // written after each wave so a stopped run carries what it finished.
  assert.deepEqual(order, ['task:codec:a1', 'task:wire:a1', 'ledger:w1', 'task:peer:a1', 'ledger:w2', 'ledger:integrate', 'integrate', 'build:r1', 'audit:ledger:final'])
  const w1 = decodeLedger(calls.find(c => c.opts.label === 'ledger:w1').prompt)
  assert.deepEqual([w1.notes.stage, w1.summary.counts.done], ['wave 1', 2], 'wave 1 persisted with its two done tasks')
  assert.deepEqual(fixArgs, [{ rungs: ['unit', 'integration'], maxRounds: 8, failFast: true }], 'the fast ladder: fail-fast, up to 8 rounds')
  // the agent that finds an error fixes it: reviewers, the builder and the whole-change lenses all write (coder role)
  assertRoles(assert, calls, { 'task:': 'wan-coder', 'task-review:': 'wan-coder', 'ledger:': 'wan-verifier', 'audit:': 'wan-verifier', 'integrate': 'wan-coder', 'build:': 'wan-coder', 'review:': 'wan-coder' })
  assert.ok(!calls.some(c => /^(amend|review-fix|propose|vote|apply):/.test(c.opts.label)), 'no separate fixer, proposer or voter')
  const final = decodeLedger(calls.find(c => c.opts.label === 'audit:ledger:final').prompt)
  assert.deepEqual(final.summary.counts, { pending: 0, running: 0, done: 3, failed: 0, blocked: 0 })
  assert.deepEqual(final.journals.peer, '.factory/runs/M0/tasks/peer.md')
  assert.ok(final.transitions.length >= 6)
  assert.deepEqual(result.deviations, [])
})

test('ownership: a plan whose independent tasks overlap is refused before any task runs (add a dependency or an integration task)', async () => {
  const plan = { milestone: 'M0', tasks: [planTask('codec', { scope: ['src/transport/'] }), planTask('wire'), planTask('peer', { deps: ['wire', 'codec'] })] }
  await assert.rejects(run({ 'extract:plan': servePlan(plan) }), /PLAN-M0\.json is invalid.*scope collision: codec owns src\/transport\//)
})

test('specialists add their focus and review lens to the tasks touching their paths; high-risk tasks get a security review', async () => {
  const sp = { id: 'wan-networking', paths: ['src/transport/'], evidence: ['src/transport/wire.py:1'], focus: 'retries must be idempotent', reviewLens: 'reconnect storms, backoff' }
  const plan = { milestone: 'M0', tasks: [planTask('wire', { risk: 'high' }), planTask('codec'), planTask('peer', { deps: ['wire', 'codec'] })] }
  const { result, calls } = await run({ 'extract:specialists': () => ({ specialists: [sp], rejected: [] }), 'extract:plan': servePlan(plan) })
  assert.equal(result.ok, true)
  assert.ok(calls.find(c => c.opts.label === 'task:wire:a1').prompt.includes('retries must be idempotent'))
  // one reviewer per task, covering every lens the task gets: the base ones, its specialists and security when high-risk
  const wireReview = calls.filter(c => c.opts.label.startsWith('task-review:wire:'))
  assert.deepEqual(wireReview.map(c => [c.opts.label, c.opts.agentType]), [['task-review:wire:a1', 'wan-coder']])
  for (const lens of ['spec-conformance', 'correctness-and-tests', 'specialist-wan-networking', 'security']) assert.ok(wireReview[0].prompt.includes(lens), lens)
  assert.ok(!calls.find(c => c.opts.label === 'task-review:codec:a1').prompt.includes('"security"'), 'security only for high-risk tasks')
  const ctx = calls.find(c => c.opts.label === 'task:peer:a1').prompt
  for (const step of ['1. SPEC', '2. ARCHITECTURE', '3. TASK', '4. DEPENDENCIES', '5. FILE OWNERSHIP', '6. ACCEPTANCE CRITERIA', '7. VERIFICATION']) assert.ok(ctx.includes(step), step)
  assert.ok(ctx.includes('.factory/runs/M0/tasks/wire.md'))
})

test('no evidence is never success: the task is retried with the failure fed back', async () => {
  const { result, calls } = await run({ 'task:wire:a1': () => ({ summary: 'should work', files: [], evidence: [] }) })
  assert.equal(result.ok, true)
  assert.deepEqual(result.tasks.tasks.find(t => t.id === 'wire').waves, [1, 2])
  assert.ok(calls.find(c => c.opts.label === 'task:wire:a2').prompt.includes('PREVIOUS ATTEMPT FAILED'))
  assert.ok(calls.find(c => c.opts.label === 'task:wire:a2').prompt.includes('no evidence'))
})

test('a spec conflict does not stop the task: the coder records it and continues; the run returns it for the human', async () => {
  // the reference PR's rule; in M2 a stop-on-conflict blocked whole waves twice in one day
  const conflict = { summary: 's', files: ['src/transport/codec.py'], evidence: [{ claim: 'c', command: 'uv run pytest -q x', exitCode: 0, output: '' }], specConflict: { description: 'the handout pins steps; SPEC-runtime lets a resume extend them', evidence: 'HANDOUT §8 vs SPEC-runtime RUNTIME-I3', proposedChange: 'steps is not a recipe pin (the SPEC of record)' } }
  const adrs = []
  const { result, calls } = await simulate(script, {
    args: { milestone: 'M0' },
    respond: byLabel({ ...base, 'task:codec:a1': () => conflict }),
    workflows: { 'wan-adr': a => { adrs.push(a); return {} }, 'wan-fix': () => ({ ok: true }) },
  })
  assert.equal(result.ok, true, result.next)
  assert.deepEqual(statuses(result), { codec: 'done', wire: 'done', peer: 'done' })
  assert.deepEqual(result.deviations.map(d => [d.task, d.implemented]), [['codec', 'steps is not a recipe pin (the SPEC of record)']])
  assert.match(result.next, /Record the 1 recorded spec conflict\(s\) \(deviations\) in the PR/)
  assert.ok(!calls.some(c => c.opts.label.startsWith('conflict:')) && !adrs.length, 'no researcher, no ADR, no stop')
  const prompt = calls.find(c => c.opts.label === 'task:codec:a1').prompt
  assert.ok(prompt.includes('are the authority over this\nprompt; the SPEC is the doc of record on conflict') || prompt.includes('the SPEC is the doc of record on conflict'))
  assert.ok(prompt.includes('record the genuine conflict in specConflict') && prompt.includes('then finish the task'))
})

test('ownership is the instruction, like the reference PR: no scope audit runs; a task is done once written and reviewed', async () => {
  const { result, calls } = await run()
  assert.ok(!calls.some(c => /^audit:(snapshot|check)/.test(c.opts.label)), 'no scope snapshot or check')
  assert.ok(calls.find(c => c.opts.label === 'task:wire:a1').prompt.includes('write ONLY them'))
  assert.equal(result.ok, true)
})

test('one reviewer per task covers every lens and fixes what it finds; what it could not fix goes to the whole-change review', async () => {
  const fixedOne = finding('major', { location: 'src/transport/wire.py:3', problem: 'missing lock on the send path', fixed: true })
  const stuck = finding('blocker', { location: 'src/transport/wire.py:9', problem: 'needs the codec change', fixed: false })
  const { result, calls } = await run({ 'task-review:wire:': () => ({ findings: [fixedOne, stuck] }) })
  assert.equal(result.ok, true, result.next)
  const review = calls.find(c => c.opts.label === 'task-review:wire:a1')
  for (const want of ['and you FIX what you find', 'Go through EVERY lens below', 'through a test', 'Never weaken, skip, delete or loosen a test']) assert.ok(review.prompt.includes(want), want)
  assert.equal(calls.filter(c => c.opts.label.startsWith('task-review:')).length, 3, 'one reviewer for each of the 3 tasks')
  assert.ok(!calls.some(c => /^(apply-fixes|amend):/.test(c.opts.label)))
  assert.ok(calls.find(c => c.opts.label === 'review:r1').prompt.includes('needs the codec change'), 'the whole-change review starts from what the reviewer could not fix')
})

test('the builder fixes the build errors it finds and re-runs; no proposal, no votes', async () => {
  let builds = 0
  const r = await run({ 'build:': () => (builds++ === 0 ? { ladder: ladder(false, ['types']), files: [{ file: 'src/transport/wire.py', errors: ['E1'] }] } : { ladder: ladder(true), files: [] }) })
  assert.equal(r.result.build.green, true)
  assert.deepEqual(r.calls.filter(c => c.opts.label.startsWith('build:')).map(c => c.opts.label), ['build:r1', 'build:r2'])
  const prompt = r.calls.find(c => c.opts.label === 'build:r1').prompt
  assert.ok(prompt.includes('fix every error yourself') && prompt.includes('no casts, ignores or disabled checks'))
  assert.ok(!r.calls.some(c => /^(propose|vote|apply):/.test(c.opts.label)))
  let never = 0
  const red = await run({ 'build:': () => { never++; return { ladder: ladder(false, ['build']), files: [] } } })
  assert.deepEqual([red.result.build.green, never], [false, 5], 'at most 5 build rounds')
})

test('budget: stops releasing waves when the remaining budget is low', async () => {
  const { result, calls } = await run({}, { budgetTotal: 100000 })
  assert.equal(result.ok, false)
  assert.equal(result.stopped, 'token budget low')
  assert.ok(!calls.some(c => c.opts.label.startsWith('task:')))
})

test('an invalid plan is refused before any task runs', async () => {
  const bad = { milestone: 'M0', tasks: [planTask('wire', { scope: ['src/train/x.py'] })] }
  await assert.rejects(run({ 'extract:plan': servePlan(bad) }), /PLAN-M0.json is invalid/)
})

test('factory v2: the plan grant is checked first; coders run fast checks; each whole-change lens fixes what it finds, 2 rounds, then the final check', async () => {
  await assert.rejects(run({ 'gate:approval:plan': () => approved(PLAN_DOCS.slice(0, 2)) }), /are not all approved by a human — run: node agents-architetura\/tools\/factory\/approve\.mjs grant docs\/design\/HANDOUT-M0\.md/)
  const minors = Array.from({ length: 3 }, (_, i) => finding('minor', { location: `src/transport/wire.py:${i}`, fixed: true }))
  const { result, calls, fixArgs } = await run({ 'review:': () => ({ findings: [finding('major', { problem: 'p-major', fixed: true }), ...minors] }), 'review-final-check': () => ({ items: [{ n: 1, status: 'resolved', evidence: 'x' }] }) })
  assert.ok(calls.find(c => c.opts.label === 'task:wire:a1').prompt.includes('uv run ruff format <files>; uv run ruff check --fix <files>; uv run mypy <files>'))
  assert.deepEqual(calls.filter(c => c.opts.label.startsWith('review:')).map(c => c.opts.label), ['review:r1', 'review:r2'], 'one reviewer per round')
  const r1 = calls.find(c => c.opts.label === 'review:r1').prompt
  for (const want of ['REVIEW AND FIX', 'spec-conformance', 'failure-and-concurrency', 'test-honesty']) assert.ok(r1.includes(want), want)
  assert.ok(calls.find(c => c.opts.label === 'review:r2').prompt.includes('p-major'), 'round 2 sees what round 1 found')
  const fc = calls.find(c => c.opts.label === 'review-final-check')
  assert.ok(fc && fc.opts.agentType === 'wan-coder' && fc.prompt.includes('p-major'))
  assert.equal(result.review.finalCheck.resolved, 1)
  assert.equal(fixArgs.length, 1, 'the ladder runs after the review')
  assert.equal(result.ok, true, result.next)
})

test('a plan too big for one agent output is copied in pieces, whole and in file order; a piece short twice stops the run', async () => {
  // M2b (2026-10-09): the 95 KB PLAN-M2b.json came back cut at ~54 KB, the copy fell back to an empty plan, and the run
  // stopped with "plan has no tasks"
  const n = 19
  const tasks = Array.from({ length: n }, (_, i) => planTask(`t${String(i).padStart(2, '0')}`, { scope: [`src/transport/t${i}.py`], deps: i ? [`t${String(i - 1).padStart(2, '0')}`] : [] }))
  const binding = Array.from({ length: 52 }, (_, i) => ({ id: `BA-${i + 1}`, what: 'w', where: 'x.py:1', ruling: 'r' }))
  const big = { milestone: 'M0', tasks, binding: binding.map((b, i) => ({ ...b })), tripwires: [] }
  big.tasks.forEach((t, i) => { t.binding = binding.filter((_, k) => k % n === i).map(b => b.id) })
  const serve = servePlan(big)
  const { result, calls } = await run({ 'extract:plan': serve })
  assert.equal(result.ok, true, result.next)
  const copies = calls.filter(c => c.opts.label.startsWith('extract:plan-M0'))
  assert.deepEqual(copies.map(c => c.opts.label.replace('extract:plan-M0:', '')).sort(), ['index', ...[0, 1, 2, 3, 4].map(b => `tasks:b${b}`), ...[0, 1, 2, 3, 4].map(b => `binding:b${b}`)].sort())
  assert.ok(copies.every(c => c.opts.agentType === 'wan-readonly'))
  assert.deepEqual(result.tasks.tasks.map(t => t.id), tasks.map(t => t.id), 'every task, in file order')
  // one short answer is copied again; a piece short twice stops the run instead of shrinking the plan
  let shorts = 0
  const flaky = (p, o) => (o.label.endsWith(':tasks:b1') && shorts++ === 0 ? { tasks: [] } : serve(p, o))
  const retried = await run({ 'extract:plan': flaky })
  assert.equal(retried.result.ok, true, retried.result.next)
  assert.ok(retried.calls.some(c => c.opts.label === 'extract:plan-M0:tasks:b1:again'))
  const cut = (p, o) => (o.label.includes(':tasks:b4') ? { tasks: [] } : serve(p, o))
  await assert.rejects(run({ 'extract:plan': cut }), /could not copy docs\/design\/PLAN-M0\.json whole: tasks t16, t17, t18 came back incomplete twice/)
})

test('carry: a restarted run takes the tasks its previous run finished as done, and writes only the rest', async () => {
  const { result, calls } = await run({ 'extract:carry': () => ({ done: ['codec', 'wire', 'ghost'] }) }, { args: { milestone: 'M0', carry: true } })
  assert.equal(result.ok, true, result.next)
  assert.deepEqual(statuses(result), { codec: 'done', wire: 'done', peer: 'done' })
  assert.deepEqual(calls.filter(c => c.opts.label.startsWith('task:')).map(c => c.opts.label), ['task:peer:a1'], 'only the unfinished task is written')
  assert.equal(calls.find(c => c.opts.label === 'extract:carry').opts.agentType, 'wan-readonly')
  const fresh = await run()
  assert.ok(!fresh.calls.some(c => c.opts.label === 'extract:carry'), 'without carry nothing is read')
})

test('carry keeps the spec conflicts the previous run recorded for the tasks it carries (they reach the human once)', async () => {
  const dev = (task, description) => ({ task, description, evidence: 'HANDOUT §8', implemented: 'the SPEC reading' })
  const prev = { done: ['codec', 'wire'], deviations: [dev('codec', 'steps pinned'), dev('codec', 'steps pinned'), dev('peer', 'not carried'), dev('ghost', 'not in plan')] }
  const { result, calls } = await run({ 'extract:carry': () => prev }, { args: { milestone: 'M0', carry: true } })
  assert.equal(result.ok, true, result.next)
  assert.deepEqual(result.deviations.map(d => [d.task, d.description]), [['codec', 'steps pinned']], 'deduplicated; only carried tasks')
  const final = decodeLedger(calls.find(c => c.opts.label === 'audit:ledger:final').prompt)
  assert.deepEqual(final.notes.deviations.map(d => d.task), ['codec'], 'the next carry reads them back')
})

test('a run stopped mid-way has persisted every finished wave: the next carry skips them', async () => {
  // the run dies in wave 2 (its session was stopped for a factory update); wave 1 must already be in the ledger
  const ledgers = []
  const respond = byLabel({ ...base, 'ledger:': p => { ledgers.push(decodeLedger(p)); return { exitCode: 0 } }, 'task:peer': () => { throw new Error('session stopped') } })
  await simulate(script, { args: { milestone: 'M0' }, respond }).catch(() => null)
  const w1 = ledgers.find(l => l.notes.stage === 'wave 1')
  assert.ok(w1, 'wave 1 was persisted before wave 2 started')
  assert.deepEqual(w1.summary.tasks.filter(t => t.status === 'done').map(t => t.id).sort(), ['codec', 'wire'])
})
