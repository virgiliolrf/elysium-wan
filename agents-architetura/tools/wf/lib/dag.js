// File-ownership + task-DAG validation. Parallel agents share one working tree,
// so two tasks that may run concurrently must never own overlapping paths.
// Scope entries are repo-relative; a trailing "/" means "this directory and
// everything under it".
const TASK_ID_RE = /^[a-z0-9][a-z0-9-]{0,47}$/
const SCOPE_PATH_RE = /^[\w.@+/-]+$/
const FORBIDDEN_SCOPE_PREFIXES = ['.git/', '.claude/workflows/']

const safeScopePath = p => {
  if (typeof p !== 'string' || !SCOPE_PATH_RE.test(p) || p.startsWith('/')) return false
  if (FORBIDDEN_SCOPE_PREFIXES.some(f => p === f.slice(0, -1) || p.startsWith(f))) return false
  const segs = (p.endsWith('/') ? p.slice(0, -1) : p).split('/')
  return segs.every(s => s !== '' && s !== '.' && s !== '..')
}

const scopesOverlap = (a, b) =>
  a === b || (a.endsWith('/') && b.startsWith(a)) || (b.endsWith('/') && a.startsWith(b))

// tasks: [{ id, deps?: string[], scope: string[] }]
// returns { ok, errors: string[], waves: string[][], overlaps: string[] } — waves are Kahn levels, ids sorted
// inside each wave so the schedule is deterministic. Concurrent tasks with overlapping scopes are errors,
// unless allowOverlap is set: then they are reported in `overlaps` and the run-time scheduler serializes them.
const validatePlan = (tasks, { allowOverlap = false } = {}) => {
  if (!Array.isArray(tasks) || !tasks.length) return { ok: false, errors: ['plan has no tasks'], waves: [], overlaps: [] }
  const errors = []
  const byId = new Map()
  for (const t of tasks) {
    if (!TASK_ID_RE.test(t?.id ?? '')) errors.push(`bad task id: ${JSON.stringify(t?.id)}`)
    else if (byId.has(t.id)) errors.push(`duplicate task id: ${t.id}`)
    else byId.set(t.id, t)
  }
  for (const t of byId.values()) {
    for (const d of t.deps ?? []) if (!byId.has(d)) errors.push(`${t.id}: unknown dependency ${JSON.stringify(d)}`)
    if (!(t.scope ?? []).length) errors.push(`${t.id}: empty scope`)
    for (const p of t.scope ?? []) if (!safeScopePath(p)) errors.push(`${t.id}: unsafe scope path ${JSON.stringify(p)}`)
  }
  if (errors.length) return { ok: false, errors, waves: [], overlaps: [] }

  const indeg = new Map([...byId.keys()].map(id => [id, 0]))
  const children = new Map([...byId.keys()].map(id => [id, []]))
  for (const t of byId.values()) for (const d of new Set(t.deps ?? [])) {
    indeg.set(t.id, indeg.get(t.id) + 1)
    children.get(d).push(t.id)
  }
  const waves = []
  let frontier = [...indeg].filter(([, n]) => n === 0).map(([id]) => id).sort()
  let placed = 0
  while (frontier.length) {
    waves.push(frontier)
    placed += frontier.length
    const next = []
    for (const id of frontier) for (const c of children.get(id)) {
      indeg.set(c, indeg.get(c) - 1)
      if (indeg.get(c) === 0) next.push(c)
    }
    frontier = next.sort()
  }
  if (placed !== byId.size) {
    const stuck = [...indeg].filter(([, n]) => n > 0).map(([id]) => id).sort()
    return { ok: false, errors: [`dependency cycle (stuck DAG) among: ${stuck.join(', ')}`], waves: [], overlaps: [] }
  }

  const memo = new Map()
  const ancestors = id => {
    if (memo.has(id)) return memo.get(id)
    const s = new Set()
    for (const d of byId.get(id).deps ?? []) { s.add(d); for (const x of ancestors(d)) s.add(x) }
    memo.set(id, s)
    return s
  }
  const ids = [...byId.keys()].sort()
  const overlaps = []
  for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
    const [a, b] = [ids[i], ids[j]]
    if (ancestors(a).has(b) || ancestors(b).has(a)) continue // ordered, never concurrent
    for (const pa of byId.get(a).scope) for (const pb of byId.get(b).scope)
      if (scopesOverlap(pa, pb)) overlaps.push(`scope collision: ${a} owns ${pa}, ${b} owns ${pb}, and they may run concurrently`)
  }
  if (!allowOverlap) errors.push(...overlaps)
  return errors.length ? { ok: false, errors, waves: [], overlaps } : { ok: true, errors: [], waves, overlaps }
}
