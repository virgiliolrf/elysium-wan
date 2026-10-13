// Normalizes agent-produced fix items before fan-out. Requires lib/hygiene.js
// (clean, token) and lib/dag.js (safeScopePath, scopesOverlap) in the same script.
//
// - ids/rungs are agent-authored and get interpolated into prompts and labels:
//   reduced to inert tokens / a known vocabulary at the source;
// - items with empty or unsafe scopes are dropped (reported, never fanned out);
// - items whose scopes overlap are MERGED, so parallel appliers never write the
//   same file (the upstream fix workflow trusted the triage agent on this);
// - ids are de-duplicated; the list is capped.
const mechanismKey = m => (typeof m === 'string' && m.trim() ? token(m, 48).toLowerCase() : '')

// The reference PR's rounds: "each round had one dominant root cause" (at one point 104 of 110 failures were a single
// mechanism). When one mechanism explains more than half of a round's failures (counted before merging), the round fixes
// only that mechanism; the rest wait for the next verify, which usually shows them gone or changed. Scattered patches on
// symptoms of one cause are how a fix round goes wrong.
const dominantMechanism = (items, { share = 0.5, min = 3 } = {}) => {
  const weight = it => Math.max(1, (it.mergedFrom ?? []).length)
  const total = items.reduce((s, it) => s + weight(it), 0)
  const groups = new Map()
  for (const it of items) {
    const k = mechanismKey(it.mechanism)
    if (k) groups.set(k, [...(groups.get(k) ?? []), it])
  }
  let best = null
  for (const [key, list] of groups) {
    const w = list.reduce((s, it) => s + weight(it), 0)
    if (!best || w > best.weight) best = { key, list, weight: w }
  }
  if (!best || total < min || best.weight / total <= share) return { focus: null, items, deferred: [], share: best ? best.weight / Math.max(total, 1) : 0 }
  return { focus: best.list[0].mechanism, items: best.list, deferred: items.filter(it => !best.list.includes(it)), share: best.weight / total, of: total, explained: best.weight }
}

const normalizeItems = (rawItems, { knownRungs, cap = 20 }) => {
  const items = []
  const dropped = []
  for (const it of rawItems ?? []) {
    const scope = (it?.scope ?? []).filter(p => typeof p === 'string')
    if (!scope.length || !scope.every(safeScopePath)) {
      dropped.push({ id: token(it?.id), why: 'empty or unsafe scope', scope: clean(JSON.stringify(it?.scope ?? []), 300) })
      continue
    }
    items.push({
      ...it,
      id: token(it.id),
      rung: knownRungs.includes(it.rung) ? it.rung : 'unknown',
      scope: [...new Set(scope)].sort(),
    })
  }

  const parent = items.map((_, i) => i)
  const find = i => (parent[i] === i ? i : (parent[i] = find(parent[i])))
  for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++)
    if (items[i].scope.some(a => items[j].scope.some(b => scopesOverlap(a, b)))) parent[find(j)] = find(i)
  const groups = new Map()
  items.forEach((it, i) => {
    const r = find(i)
    if (!groups.has(r)) groups.set(r, [])
    groups.get(r).push(it)
  })
  const merged = [...groups.values()].map(g => g.length === 1 ? g[0] : {
    id: g.map(x => x.id).join('-').slice(0, 64),
    rung: g[0].rung,
    symptom: g.map(x => `[${x.id}] ${x.symptom ?? ''}`).join('\n'),
    evidence: g.map(x => `[${x.id}] ${x.evidence ?? ''}`).join('\n\n'),
    suspectedCause: g.map(x => x.suspectedCause).filter(Boolean).join(' | '),
    scope: [...new Set(g.flatMap(x => x.scope))].sort(),
    mergedFrom: g.map(x => x.id),
    // one mechanism for the merged item only if every member names the same one
    ...(new Set(g.map(x => mechanismKey(x.mechanism))).size === 1 && mechanismKey(g[0].mechanism) ? { mechanism: g[0].mechanism } : {}),
    // one owner for the merged item only if every member agrees; otherwise no single context owns it
    ...(new Set(g.map(x => x.ownerJournal ?? null)).size === 1 && g[0].ownerJournal ? { ownerTask: g[0].ownerTask, ownerJournal: g[0].ownerJournal } : {}),
  })

  const used = new Set()
  for (const it of merged) {
    let id = it.id
    for (let k = 2; used.has(id); k++) id = `${it.id}-${k}`
    it.id = id
    used.add(id)
  }
  return { items: merged.slice(0, cap), dropped, overflow: Math.max(0, merged.length - cap) }
}
