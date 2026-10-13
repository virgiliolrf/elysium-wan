import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loadLib } from '../sim.mjs'

const { checkMilestonePlan } = loadLib(['dag', 'plan'])
const ctx = {
  workstreams: [{ key: 'transport', ownedPaths: ['src/transport/', 'tests/unit/transport/'] }, { key: 'train', ownedPaths: ['src/train/'] }],
  rungIds: ['build', 'unit', 'oracle'],
  hotFiles: ['pyproject.toml', 'agents-architetura/tools/gates/rungs.json'],
  integrator: 'integrate',
}
const task = (id, extra = {}) => ({ id, title: id, objective: `o-${id}`, workstream: 'transport', deps: [], scope: [`src/transport/${id}.py`], acceptance: ['a'], verify: ['uv run pytest -q'], rungs: ['unit'], risk: 'low', ...extra })

test('a valid plan yields waves', () => {
  const r = checkMilestonePlan({ milestone: 'M0', tasks: [
    task('wire'),
    task('peer', { deps: ['wire'] }),
    task('deps', { workstream: 'integrate', scope: ['pyproject.toml'] }),
    task('loop', { workstream: 'train', scope: ['src/train/loop.py'], deps: ['peer'] }),
  ] }, ctx)
  assert.deepEqual(r, { ok: true, errors: [], waves: [['deps', 'wire'], ['peer'], ['loop']], serialized: [] })
})

test('tasks must stay inside their workstream, and the integrator inside hot files', () => {
  const r = checkMilestonePlan({ milestone: 'M0', tasks: [
    task('a', { scope: ['src/train/x.py'] }),
    task('b', { workstream: 'integrate', scope: ['src/transport/b.py'] }),
    task('c', { workstream: 'nope' }),
  ] }, ctx)
  assert.deepEqual(r.errors, [
    "a: src/train/x.py is outside workstream transport's owned paths",
    'b: the integrator task may only own shared hot files, not src/transport/b.py',
    'c: unknown workstream "nope"',
  ])
})

test('objective, acceptance, verify, rungs and risk are mandatory and rungs must be known', () => {
  const r = checkMilestonePlan({ milestone: 'M0', tasks: [task('a', { objective: ' ', acceptance: [], verify: [], rungs: [], risk: 'huge' }), task('b', { rungs: ['tsan'] })] }, ctx)
  assert.deepEqual(r.errors, ['a: no objective', 'a: no acceptance criteria', 'a: no verification commands', 'a: names no ladder rungs', 'a: risk must be low, medium or high', 'b: unknown rung "tsan"'])
})

test('independent tasks with overlapping scopes are an error: add the dependency or an integration task', () => {
  const r = checkMilestonePlan({ milestone: 'M0', tasks: [task('a', { scope: ['src/transport/'] }), task('b')] }, ctx)
  assert.equal(r.ok, false)
  assert.match(r.errors[0], /scope collision: a owns src\/transport\/, b owns src\/transport\/b\.py/)
  assert.equal(checkMilestonePlan({ milestone: 'M0', tasks: [task('a', { scope: ['src/transport/'] }), task('b', { deps: ['a'] })] }, ctx).ok, true)
})

test('duplicate tasks (same objective and scope) are rejected', () => {
  const r = checkMilestonePlan({ milestone: 'M0', tasks: [task('a', { objective: 'Codec', scope: ['src/transport/c.py'] }), task('b', { objective: 'codec ', scope: ['src/transport/c.py'], deps: ['a'] })] }, ctx)
  assert.ok(r.errors.includes('b: duplicates task a (same objective and scope)'))
})

test('DAG errors from validatePlan surface too', () => {
  const r = checkMilestonePlan({ milestone: 'M0', tasks: [task('a', { deps: ['ghost'] })] }, ctx)
  assert.equal(r.ok, false)
  assert.match(r.errors[0], /unknown dependency "ghost"/)
  assert.deepEqual(checkMilestonePlan(null, ctx).errors, ['plan has no tasks'])
})

test('binding audit: every item has a ruling and some task delivers it; tasks cannot resolve unknown items', () => {
  const binding = [
    { id: 'BA-1', what: 'IWRR ownership table', where: 'src/transport/sched.py:40', ruling: 'rebuilt at the checkpoint boundary' },
    { id: 'BA-2', what: 'dropout RNG stream', where: 'src/train/loop.py:12', ruling: 'carried unchanged' },
  ]
  const ok = checkMilestonePlan({ milestone: 'M2', binding, tasks: [task('wire', { binding: ['BA-1'] }), task('peer', { binding: ['BA-2'] })] }, ctx)
  assert.deepEqual(ok.errors, [])
  const bad = checkMilestonePlan({ milestone: 'M2', binding: [...binding, { id: 'BA-2', what: 'dup', where: 'x', ruling: '' }, { id: 'X', what: 'w', where: 'x', ruling: 'r' }], tasks: [task('wire', { binding: ['BA-1', 'BA-9'] })] }, ctx)
  for (const want of [/binding BA-2: duplicate id/, /binding BA-2: no ruling/, /binding "X": id must look like BA-1/, /wire: resolves unknown binding item "BA-9"/, /binding BA-2: no task delivers its ruling/]) assert.ok(bad.errors.some(e => want.test(e)), `${want}: ${bad.errors.join('; ')}`)
})

test('tripwire: the arm task depends on every leg and every leg proves the refusal still holds', () => {
  const T = 'uv run pytest -q tests/unit/test_rebalance_refused.py'
  const legs = [task('telemetry', { verify: ['uv run pytest -q', T] }), task('resolve', { deps: ['telemetry'], verify: [T] }), task('restart', { verify: [T] })]
  const wire = { id: 'rebalance', flag: 'rebalance=True raises until armed', legs: ['telemetry', 'resolve', 'restart'], arm: 'arm', test: T }
  const ok = checkMilestonePlan({ milestone: 'M2', tripwires: [wire], tasks: [...legs, task('arm', { deps: ['resolve', 'restart'] })] }, ctx)
  assert.deepEqual(ok.errors, [], 'telemetry is reached through resolve')
  const partial = checkMilestonePlan({ milestone: 'M2', tripwires: [wire], tasks: [...legs, task('arm', { deps: ['resolve'] })] }, ctx)
  assert.ok(partial.errors.some(e => /arm task arm does not depend on leg restart/.test(e)), partial.errors.join('; '))
  const unproven = checkMilestonePlan({ milestone: 'M2', tripwires: [wire], tasks: [legs[0], legs[1], task('restart'), task('arm', { deps: ['resolve', 'restart'] })] }, ctx)
  assert.ok(unproven.errors.some(e => /leg restart does not verify that the refusal still holds/.test(e)))
  const thin = checkMilestonePlan({ milestone: 'M2', tripwires: [{ id: 'w', flag: '', legs: ['telemetry'], arm: 'telemetry', test: '' }], tasks: legs }, ctx)
  for (const want of [/no flag/, /no test command/, /at least two legs/, /cannot be one of its legs/]) assert.ok(thin.errors.some(e => want.test(e)), `${want}: ${thin.errors.join('; ')}`)
})

test('maxWaves: a plan deeper than 2 waves is refused, naming the waves past the limit and the files that chain them', () => {
  const chained = { milestone: 'M0', tasks: [
    task('a', { scope: ['src/transport/a.py', 'tests/unit/transport/test_all.py'] }),
    task('b', { deps: ['a'], scope: ['src/transport/b.py', 'tests/unit/transport/test_all.py'] }),
    task('c', { deps: ['b'], scope: ['src/transport/c.py', 'tests/unit/transport/test_all.py'] }),
  ] }
  assert.equal(checkMilestonePlan(chained, ctx).ok, true, 'without maxWaves (wan-implement on an old plan) the depth is not checked')
  const r = checkMilestonePlan(chained, { ...ctx, maxWaves: 2 })
  assert.equal(r.ok, false)
  assert.equal(r.errors.length, 1)
  assert.match(r.errors[0], /runs in 3 waves \(at most 2\): wave 3: c\./)
  assert.match(r.errors[0], /tests\/unit\/transport\/test_all\.py \(a, b, c\)/)
  const flat = { milestone: 'M0', tasks: [
    task('a', { scope: ['src/transport/a.py', 'tests/unit/transport/test_a.py'] }),
    task('b', { scope: ['src/transport/b.py', 'tests/unit/transport/test_b.py'] }),
    task('wire', { deps: ['a', 'b'], scope: ['src/transport/c.py', 'tests/unit/transport/test_wire.py'] }),
  ] }
  assert.deepEqual(checkMilestonePlan(flat, { ...ctx, maxWaves: 2 }).waves, [['a', 'b'], ['wire']], 'one file per task, own test files: 2 waves')
})
