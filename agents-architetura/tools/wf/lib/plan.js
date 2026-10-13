// Milestone plan validation (HANDOUT/PLAN-<milestone>.json). Requires dag (validatePlan).
const covers = (owned, p) => owned === p || (owned.endsWith('/') && p.startsWith(owned))

// plan: { milestone, tasks: [{ id, title, objective, workstream, deps, scope, acceptance, verify, rungs }] }
// Beyond DAG validity: every task has an objective, acceptance criteria, verification commands and known
// rungs, a risk level, and stays inside the owned paths of its workstream — or, for the integrator task, inside the
// shared hot files. Independent tasks whose scopes overlap are an ERROR: the planner must add the dependency or an
// integration task explicitly (the scheduler's run-time serialization is only a safety net). Duplicate tasks (same
// objective and scope) are an error too.
const RISKS = ['low', 'medium', 'high']
// The reference PR's binding audits ("every VM-singular field reachable from a spawned thread, each with a ruling"): every
// instance of the hazard the milestone touches is listed with a ruling BEFORE code is written, and some task delivers each
// ruling — found by reading, not by crashing.
const BINDING_ID = /^BA-\d{1,4}$/
const checkBinding = (plan, tasks, errors) => {
  const ids = new Set()
  for (const b of plan?.binding ?? []) {
    const id = b?.id ?? '?'
    if (!BINDING_ID.test(b?.id ?? '')) errors.push(`binding ${JSON.stringify(id)}: id must look like BA-1`)
    else if (ids.has(id)) errors.push(`binding ${id}: duplicate id`)
    ids.add(id)
    for (const k of ['what', 'where', 'ruling']) if (!String(b?.[k] ?? '').trim()) errors.push(`binding ${id}: no ${k}`)
  }
  const delivered = new Set()
  for (const t of tasks) for (const b of t?.binding ?? []) {
    if (!ids.has(b)) errors.push(`${t?.id}: resolves unknown binding item ${JSON.stringify(b)}`)
    delivered.add(b)
  }
  for (const id of ids) if (!delivered.has(id)) errors.push(`binding ${id}: no task delivers its ruling`)
}
// The reference PR's tripwire ("refused multi-thread entry until every leg landed"): a change that is silently wrong when
// only some of its legs are in stays refused in the product until an arm task, which depends on every leg, lifts the
// refusal. Every leg proves the refusal still holds.
const checkTripwires = (plan, tasks, errors) => {
  const byId = new Map(tasks.map(t => [t?.id, t]))
  const upstream = id => {
    const seen = new Set()
    const stack = [...(byId.get(id)?.deps ?? [])]
    while (stack.length) {
      const d = stack.pop()
      if (seen.has(d)) continue
      seen.add(d)
      stack.push(...(byId.get(d)?.deps ?? []))
    }
    return seen
  }
  for (const w of plan?.tripwires ?? []) {
    const id = w?.id ?? '?'
    const legs = Array.isArray(w?.legs) ? w.legs : []
    if (!String(w?.flag ?? '').trim()) errors.push(`tripwire ${id}: no flag (what stays refused until it is armed)`)
    if (!String(w?.test ?? '').trim()) errors.push(`tripwire ${id}: no test command proving the refusal holds`)
    if (legs.length < 2) errors.push(`tripwire ${id}: needs at least two legs`)
    for (const l of legs) {
      if (!byId.has(l)) errors.push(`tripwire ${id}: unknown leg ${JSON.stringify(l)}`)
      else if (!(byId.get(l).verify ?? []).includes(w?.test)) errors.push(`tripwire ${id}: leg ${l} does not verify that the refusal still holds (${w?.test})`)
    }
    if (!byId.has(w?.arm)) errors.push(`tripwire ${id}: unknown arm task ${JSON.stringify(w?.arm)}`)
    else if (legs.includes(w.arm)) errors.push(`tripwire ${id}: the arm task cannot be one of its legs`)
    else {
      const up = upstream(w.arm)
      for (const l of legs) if (byId.has(l) && !up.has(l)) errors.push(`tripwire ${id}: arm task ${w.arm} does not depend on leg ${l} — it could lift the refusal over a partial change`)
    }
  }
}
// maxWaves (the human's rule, 2026-10-09: "at most 2 waves"): a plan deeper than this is refused at planning time. M2's plan
// ran in 5 waves and M2b's in 13, not for lack of parallel work but because shared files (one test file with 7 owners,
// stage.py with 5) chained every task touching them. The refusal names the files that chain.
const checkDepth = (tasks, waves, maxWaves, errors) => {
  if (!maxWaves || waves.length <= maxWaves) return
  const owners = new Map()
  for (const t of tasks) for (const p of t.scope ?? []) owners.set(p, [...(owners.get(p) ?? []), t.id])
  const shared = [...owners].filter(([, ids]) => ids.length > 1).map(([p, ids]) => `${p} (${ids.join(', ')})`)
  errors.push(`the plan runs in ${waves.length} waves (at most ${maxWaves}): ${waves.slice(maxWaves).map((w, i) => `wave ${i + maxWaves + 1}: ${w.join(', ')}`).join('; ')}. ` +
    `Make the work one task per file: merge the tasks that chain on the same files, give every task its own new test file, and keep wave ${maxWaves} for integration tasks.` +
    (shared.length ? ` Files with more than one owner: ${shared.join('; ')}` : ''))
}
const checkMilestonePlan = (plan, { workstreams, rungIds, hotFiles, integrator, maxWaves }) => {
  const tasks = plan?.tasks ?? []
  const base = validatePlan(tasks)
  const errors = [...base.errors]
  if (base.ok) checkDepth(tasks, base.waves, maxWaves, errors)
  const seen = new Map()
  for (const t of tasks) {
    const key = `${String(t?.objective ?? '').trim().toLowerCase()}|${[...(t?.scope ?? [])].sort().join(',')}`
    if (seen.has(key)) errors.push(`${t?.id}: duplicates task ${seen.get(key)} (same objective and scope)`)
    else seen.set(key, t?.id)
  }
  const owners = new Map((workstreams ?? []).map(w => [w.key, w.ownedPaths ?? []]))
  for (const t of tasks) {
    const id = t?.id ?? '?'
    if (!String(t?.objective ?? '').trim()) errors.push(`${id}: no objective`)
    if (!(t?.acceptance ?? []).length) errors.push(`${id}: no acceptance criteria`)
    if (!(t?.verify ?? []).length) errors.push(`${id}: no verification commands`)
    if (!(t?.rungs ?? []).length) errors.push(`${id}: names no ladder rungs`)
    if (!RISKS.includes(t?.risk)) errors.push(`${id}: risk must be low, medium or high`)
    for (const r of t?.rungs ?? []) if (!rungIds.includes(r)) errors.push(`${id}: unknown rung ${JSON.stringify(r)}`)
    if (t?.workstream === integrator) {
      for (const p of t.scope ?? []) if (!hotFiles.includes(p)) errors.push(`${id}: the integrator task may only own shared hot files, not ${p}`)
    } else if (!owners.has(t?.workstream)) {
      errors.push(`${id}: unknown workstream ${JSON.stringify(t?.workstream)}`)
    } else {
      for (const p of t.scope ?? []) if (!owners.get(t.workstream).some(o => covers(o, p))) errors.push(`${id}: ${p} is outside workstream ${t.workstream}'s owned paths`)
    }
  }
  checkBinding(plan, tasks, errors)
  checkTripwires(plan, tasks, errors)
  const unique = [...new Set(errors)]
  return { ok: !unique.length, errors: unique, waves: unique.length ? [] : base.waves, serialized: [] }
}

// Reads PLAN-<m>.json through a read-only agent and validates it. Requires hygiene, schemas, project
// (common, planPath, RUNG_IDS, HOT_FILES, INTEGRATOR) and the runtime global agent().
// A plan is copied in pieces: one agent's structured output is capped. PLAN-M2b.json (95 KB) came back cut at ~54 KB
// twice, the agent fell back to an empty plan, and wan-implement stopped with "plan has no tasks" (2026-10-09). An index
// call names every entry; parallel calls copy the entries a few at a time; an entry missing twice stops the run, never
// a silently shorter plan.
const PLAN_INDEX = {
  type: 'object',
  required: ['milestone', 'taskIds', 'bindingIds', 'tripwireIds'],
  properties: {
    milestone: { type: 'string' },
    taskIds: { type: 'array', items: { type: 'string' }, description: 'the id of every entry of "tasks", in file order' },
    bindingIds: { type: 'array', items: { type: 'string' }, description: 'the id of every entry of "binding", in file order' },
    tripwireIds: { type: 'array', items: { type: 'string' }, description: 'the id of every entry of "tripwires", in file order' },
  },
}
const PLAN_PIECE = { tasks: 4, binding: 12, tripwires: 6 }
const piecesOf = (xs, n) => Array.from({ length: Math.ceil(xs.length / n) }, (_, i) => xs.slice(i * n, i * n + n))
const extractPlan = async (m, label, phaseTitle) => {
  const PLAN_PART = { type: 'object', properties: { tasks: PLAN.properties.tasks, binding: PLAN.properties.binding, tripwires: PLAN.properties.tripwires } }
  const READ = common('READ-ONLY.') + `
Read ${planPath(m)}. Copy, never fix, reorder, complete or reinterpret anything.`
  const index = await agent(READ + `
Return its "milestone" and the "id" of every entry of "tasks", "binding" and "tripwires", in file order, as taskIds,
bindingIds and tripwireIds (an absent list is []). If the file is missing or not valid JSON, return
{"milestone": "${m}", "taskIds": [], "bindingIds": [], "tripwireIds": []}.`,
    { label: `${label}:index`, phase: phaseTitle, schema: PLAN_INDEX, agentType: ROLE.readonly })
  const idsOf = key => (Array.isArray(index?.[key]) ? index[key] : []).map(x => String(x))
  const jobs = [['tasks', 'taskIds'], ['binding', 'bindingIds'], ['tripwires', 'tripwireIds']]
    .flatMap(([field, key]) => piecesOf(idsOf(key), PLAN_PIECE[field]).map((ids, b) => ({ field, ids, label: `${label}:${field}:b${b}` })))
  const copy = async job => {
    for (let attempt = 0; attempt < 2; attempt++) {
      const r = await agent(READ + `
Return VERBATIM, as {"${job.field}": [...]}, the entries of "${job.field}" whose id is in this list, in this order, every
field exactly as written, and nothing else.
IDS: ${JSON.stringify(job.ids)}`,
        { label: attempt ? `${job.label}:again` : job.label, phase: phaseTitle, schema: PLAN_PART, agentType: ROLE.readonly })
      const got = new Map((Array.isArray(r?.[job.field]) ? r[job.field] : []).filter(x => x && typeof x === 'object').map(x => [String(x.id), x]))
      if (job.ids.every(id => got.has(id))) return job.ids.map(id => got.get(id))
    }
    return null
  }
  const copied = await parallel(jobs.map(job => () => copy(job)))
  const plan = { milestone: String(index?.milestone ?? m), tasks: [], binding: [], tripwires: [] }
  const incomplete = []
  jobs.forEach((job, i) => { if (copied[i]) plan[job.field].push(...copied[i]); else incomplete.push(`${job.field} ${job.ids.join(', ')}`) })
  if (incomplete.length) throw new Error(`could not copy ${planPath(m)} whole: ${incomplete.join('; ')} came back incomplete twice`)
  return plan
}

const loadPlan = async (m, workstreams, phaseTitle) => {
  const plan = await extractPlan(m, `extract:plan-${m}`, phaseTitle)
  const check = checkMilestonePlan(plan, { workstreams, rungIds: RUNG_IDS, hotFiles: HOT_FILES, integrator: INTEGRATOR })
  if (!check.ok) throw new Error(`${planPath(m)} is invalid — rerun wan-handout: ${check.errors.slice(0, 20).join('; ')}`)
  return { plan, waves: check.waves, serialized: check.serialized, byId: new Map(plan.tasks.map(t => [t.id, t])) }
}
