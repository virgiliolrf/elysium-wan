import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loadLib } from '../sim.mjs'

const { createRun, nextWave, markDone, markFailed, runSummary } = loadLib(['dag', 'scheduler'])
const t = (id, deps = [], scope = [`src/${id}.py`]) => ({ id, deps, scope })
const drain = (run, onTask) => {
  const waves = []
  for (;;) {
    const w = nextWave(run)
    if (w.stuck) return { waves, stuck: w.stuckReason }
    if (!w.wave.length) return { waves }
    waves.push({ ids: w.wave, deferred: w.deferred.map(d => d.id) })
    for (const id of w.wave) onTask(id)
  }
}

test('dependencies are resolved in topological order; independent tasks share a wave', () => {
  const run = createRun([t('d', ['b', 'c']), t('b', ['a']), t('c', ['a']), t('a')])
  const r = drain(run, id => markDone(run, id))
  assert.deepEqual(r.waves.map(w => w.ids), [['a'], ['b', 'c'], ['d']])
  assert.deepEqual(runSummary(run).counts, { pending: 0, running: 0, done: 4, failed: 0, blocked: 0 })
})

test('independent tasks with overlapping ownership never run in the same wave', () => {
  const run = createRun([t('x', [], ['src/net/']), t('y', [], ['src/net/peer.py']), t('z')])
  const r = drain(run, id => markDone(run, id))
  assert.deepEqual(r.waves, [{ ids: ['x', 'z'], deferred: ['y'] }, { ids: ['y'], deferred: [] }])
})

test('a failing task is retried, then fails; its dependents are blocked, independents finish', () => {
  const run = createRun([t('a'), t('b', ['a']), t('c', ['b']), t('solo')], { maxAttempts: 2 })
  const tries = []
  drain(run, id => { tries.push(id); if (id === 'a') markFailed(run, id, 'tests red'); else markDone(run, id) })
  assert.deepEqual(tries, ['a', 'solo', 'a'])
  const s = Object.fromEntries(runSummary(run).tasks.map(x => [x.id, [x.status, x.attempts, x.reason]]))
  assert.deepEqual(s.a, ['failed', 2, 'tests red'])
  assert.deepEqual(s.b, ['blocked', 0, 'dependency a is failed'])
  assert.deepEqual(s.c, ['blocked', 0, 'dependency b is blocked'])
  assert.deepEqual(s.solo, ['done', 0, null])
})

test('stuck DAG is detected with the reason instead of looping', () => {
  const run = createRun([t('a', ['ghost'])])
  const r = drain(run, id => markDone(run, id))
  assert.match(r.stuck, /a waits on ghost/)
})

test('a task left running is reported as stuck, not silently dropped', () => {
  const run = createRun([t('a'), t('b', ['a'])])
  nextWave(run) // a -> running, never completed
  const w = nextWave(run)
  assert.equal(w.stuck, true)
  assert.match(w.stuckReason, /b waits on a/)
})

test('every transition is recorded with its wave', () => {
  const run = createRun([t('a'), t('b', ['a'])], { maxAttempts: 2 })
  drain(run, id => (id === 'a' && run.state.get('a').attempts === 0 ? markFailed(run, id, 'x') : markDone(run, id)))
  assert.deepEqual(run.transitions.map(x => `${x.wave}:${x.id}:${x.from}->${x.to}`), [
    '1:a:pending->running', '1:a:running->pending', '2:a:pending->running', '2:a:running->done', '3:b:pending->running', '3:b:running->done',
  ])
})
