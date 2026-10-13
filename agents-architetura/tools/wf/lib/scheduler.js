// DAG task scheduler with explicit task state. Requires lib/dag.js (scopesOverlap).
//
// States: pending -> running -> done
//                            -> pending (retry, attempts < maxAttempts)
//                            -> failed  (attempts exhausted)
//         pending -> blocked (a dependency failed or is blocked)
//
// nextWave() resolves dependencies (only tasks whose deps are all done are ready), keeps a deterministic
// topological order (depth, then id), and enforces file ownership at run time: a ready task whose scope
// overlaps a task already selected for the wave is deferred to a later wave instead of running
// concurrently. When nothing is ready but unfinished tasks remain, the DAG is stuck and the reason is
// reported instead of looping.
const TASK_STATES = ['pending', 'running', 'done', 'failed', 'blocked']

const createRun = (tasks, { maxAttempts = 2 } = {}) => {
  const byId = new Map(tasks.map(t => [t.id, t]))
  const depth = new Map()
  const depthOf = id => {
    if (depth.has(id)) return depth.get(id)
    depth.set(id, 0) // cycle guard; validatePlan rejects cycles before a run is created
    const d = Math.max(0, ...(byId.get(id)?.deps ?? []).filter(x => byId.has(x)).map(x => depthOf(x) + 1))
    depth.set(id, d)
    return d
  }
  const order = [...byId.keys()].sort((a, b) => depthOf(a) - depthOf(b) || (a < b ? -1 : 1))
  const state = new Map(order.map(id => [id, { id, status: 'pending', attempts: 0, waves: [], reason: null, evidence: null }]))
  return { byId, order, state, maxAttempts, wave: 0, transitions: [] }
}

const transition = (run, id, status, reason = null) => {
  const s = run.state.get(id)
  run.transitions.push({ wave: run.wave, id, from: s.status, to: status, reason })
  s.status = status
  s.reason = reason
}

const nextWave = run => {
  // Failure propagation: anything downstream of a failed/blocked task can never run.
  let changed = true
  while (changed) {
    changed = false
    for (const id of run.order) {
      const s = run.state.get(id)
      if (s.status !== 'pending') continue
      const dead = (run.byId.get(id).deps ?? []).find(d => ['failed', 'blocked'].includes(run.state.get(d)?.status))
      if (dead) { transition(run, id, 'blocked', `dependency ${dead} is ${run.state.get(dead).status}`); changed = true }
    }
  }
  const ready = run.order.filter(id => run.state.get(id).status === 'pending' &&
    (run.byId.get(id).deps ?? []).every(d => run.state.get(d)?.status === 'done'))
  const wave = []
  const deferred = []
  for (const id of ready) {
    const clash = wave.find(w => run.byId.get(w).scope.some(a => run.byId.get(id).scope.some(b => scopesOverlap(a, b))))
    if (clash) deferred.push({ id, reason: `ownership overlaps ${clash}` })
    else wave.push(id)
  }
  const unfinished = run.order.filter(id => ['pending', 'running'].includes(run.state.get(id).status))
  const stuck = !wave.length && unfinished.length > 0
  const stuckReason = stuck
    ? unfinished.map(id => `${id} waits on ${(run.byId.get(id).deps ?? []).filter(d => run.state.get(d)?.status !== 'done').join(', ') || '(nothing — scheduler bug)'}`).join('; ')
    : null
  if (wave.length) {
    run.wave++
    for (const id of wave) {
      run.state.get(id).waves.push(run.wave)
      transition(run, id, 'running')
    }
  }
  return { wave, deferred, stuck, stuckReason, done: !unfinished.length }
}

const markDone = (run, id, evidence = null) => {
  run.state.get(id).evidence = evidence
  transition(run, id, 'done')
}

// Returns true when the task will be retried. final=true fails it now (e.g. a re-plan is required).
const markFailed = (run, id, reason, { final = false } = {}) => {
  const s = run.state.get(id)
  s.attempts++
  if (!final && s.attempts < run.maxAttempts) { transition(run, id, 'pending', `retry ${s.attempts}/${run.maxAttempts - 1}: ${reason}`); return true }
  transition(run, id, 'failed', reason)
  return false
}

const runSummary = run => {
  const counts = Object.fromEntries(TASK_STATES.map(st => [st, 0]))
  for (const s of run.state.values()) counts[s.status]++
  return {
    waves: run.wave,
    counts,
    tasks: run.order.map(id => {
      const s = run.state.get(id)
      return { id, status: s.status, attempts: s.attempts, waves: s.waves, reason: s.reason }
    }),
  }
}
