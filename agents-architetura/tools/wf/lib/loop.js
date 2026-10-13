// Adversarial review loop + vote tally. Uses the workflow runtime globals
// parallel() and log().
//
// A round converges only when EVERY lens reported and none filed an actionable
// blocker/major. A missing reviewer (skipped or errored agent) is not a clean
// pass — the PR this is modeled on silently treated an all-null round as clean.
// A finding is actionable only if the reviewer validated it against the
// current code and gave location, evidence and reproduction; the rest are
// discarded (and counted) so hallucinated findings never drive a fix.
const findingProven = f => f.validatedAgainstCurrentCode === true &&
  [f.location, f.evidence, f.reproduction].every(x => typeof x === 'string' && x.trim().length > 0)
const isActionable = f => !!f && f.severity !== 'minor' && findingProven(f)

// Default review budget: up to 5 rounds (the human's call, 2026-10-08, after the M2 phase review fell 12 -> 9 -> 6 serious
// findings and stopped at its old limit of 3 while still converging). A milestone run asks for fewer (the human's call,
// 2026-10-09, factory v2): its plan review and its final code review have 2 rounds each and fix everything they find.
const REVIEW_ROUNDS = 5
// Validated minors ride along with the serious findings of a round (after them, so a capped fence cuts minors first). Only
// blocker/major decide convergence, but a minor left unfixed came back the next round and was often re-filed as a major
// (M2 phase review, 2026-10-08: the same three minors went r4 -> r5 -> r1 and two were promoted).
const MINORS_PER_FIX = 8
// "house": a defect in shared code the reviewed change did not write and does not need changed to be correct (a factory
// lib, the template, another workstream). Honored only when the caller says which files the change owns (ownsFile) and the
// finding's file is outside them; it never blocks the loop and is returned for the factory backlog instead.
// A review whose blocker/major count stops falling is a document that is too big, not one that needs more rounds (the
// human's rule, 2026-10-08: "divide always"; the hetero ADR went 17 -> 17 -> 19 at 2006 lines while each fix round got
// slower). Three counted rounds in a row that never fall end the loop as stalled, and the caller splits the work.
const STALL_ROUNDS = 3
const isStalled = counts => {
  const n = counts.filter(x => typeof x === 'number').slice(-STALL_ROUNDS)
  return n.length === STALL_ROUNDS && n.every((x, i) => i === 0 || x >= n[i - 1])
}
const findingFile = location => String(location ?? '').trim().split(/[:\s]/)[0]
const findingBrief = f => ({
  ...(f.lens ? { lens: f.lens } : {}),
  severity: f.severity,
  location: String(f.location ?? '').slice(0, 200),
  problem: String(f.problem ?? '').slice(0, 600),
})
// The findings of the last round are fixed too, with no review after them: "open" lists them for the record. fixAll: every
// validated minor rides along, not only MINORS_PER_FIX (a milestone run fixes everything its two rounds find).
//
// Every fix gets the findings of the earlier rounds (fix(items, round, earlier)), so it does not undo them. finalCheck,
// when given, runs once after the last round's fix if the loop did not converge: one agent goes through every finding of
// every round against the current state and fixes what is still open or came back (finalCheck(all) -> FINAL_CHECK). The
// M2 single plan (2026-10-09) is why: round 1 condemned a verdict rule that could not discriminate and security work
// coupled to the wave chain; the unreviewed fix after round 2 re-coupled everything to the experiment, the vacuous rule
// survived, and the whole build stopped on its first task.
const FINAL_CHECK = {
  type: 'object',
  required: ['items'],
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        required: ['n', 'status', 'evidence'],
        properties: {
          n: { type: 'integer', description: 'the finding number F<n> as listed' },
          status: { type: 'string', enum: ['resolved', 'fixed-now', 'refuted', 'still-open'], description: 'resolved: the current state already fixes it; fixed-now: you fixed it in this pass (or it had come back); refuted: it was never real; still-open: you could not fix it' },
          evidence: { type: 'string', description: 'file:line where it is resolved or fixed, or why it is refuted or still open' },
        },
      },
    },
    summary: { type: 'string' },
  },
}
const numbered = items => items.map((f, i) => ({ n: i + 1, ...f }))
const reviewUntilClean = async ({ label, lenses, maxRounds = REVIEW_ROUNDS, review, fix, ownsFile = null, fixAll = false, finalCheck = null }) => {
  if (!(Number.isInteger(maxRounds) && maxRounds >= 1)) throw new Error(`reviewUntilClean(${label}): maxRounds must be a positive integer`)
  const findingsPerRound = []
  const house = new Map()
  let discarded = 0
  let open = []
  let minors = []
  const sent = [] // every finding sent to a fix, with its round: what later fixes must not undo
  const isHouse = f => typeof ownsFile === 'function' && f.scope === 'house' && !ownsFile(findingFile(f.location))
  const done = extra => ({ findingsPerRound, discarded, minors: minors.map(findingBrief), house: [...house.values()].map(findingBrief), ...extra })
  for (let round = 1; round <= maxRounds; round++) {
    const results = await parallel(lenses.map(([name, lens]) => () => review(name, lens, round)))
    const reviews = lenses.map(([name], i) => results[i] && { name, r: results[i] }).filter(Boolean)
    if (reviews.length < lenses.length) {
      findingsPerRound.push(`missing:${lenses.length - reviews.length}`)
      log(`${label} round ${round}: only ${reviews.length}/${lenses.length} reviewers reported — not a clean pass, re-reviewing`)
      continue
    }
    const all = reviews.flatMap(({ name, r }) => (r.findings ?? []).filter(Boolean).map(f => ({ ...f, lens: name })))
    for (const f of all.filter(f => isHouse(f) && findingProven(f))) house.set(`${findingFile(f.location)}|${String(f.problem ?? '').slice(0, 120)}`, f)
    const mine = all.filter(f => !isHouse(f))
    const serious = mine.filter(isActionable)
    minors = mine.filter(f => f.severity === 'minor' && findingProven(f))
    const dropped = mine.filter(f => f.severity !== 'minor' && !isActionable(f)).length
    discarded += dropped
    if (dropped) log(`${label} round ${round}: discarded ${dropped} blocker/major finding(s) not validated against the current code`)
    findingsPerRound.push(serious.length)
    if (!serious.length) {
      log(`${label}: clean pass on round ${round} (findings per round: ${findingsPerRound.join(' -> ')})${minors.length ? `; ${minors.length} minor(s) left open` : ''}`)
      return done({ converged: true, rounds: round })
    }
    open = serious
    if (isStalled(findingsPerRound)) {
      log(`${label}: review STALLED (${findingsPerRound.join(' -> ')}): the blocker/major count stopped falling — split the work instead of more rounds`)
      return done({ converged: false, stalled: true, rounds: round, open: open.map(findingBrief).slice(0, 30) })
    }
    const riding = fixAll ? minors : minors.slice(0, MINORS_PER_FIX)
    log(`${label} round ${round}: ${serious.length} blocker/major finding(s)${riding.length ? ` and ${riding.length} minor(s)` : ''} -> fixing`)
    const earlier = sent.slice()
    sent.push(...[...serious, ...riding].map(f => ({ round, ...findingBrief(f) })))
    await fix([...serious, ...riding], round, earlier)
  }
  if (!finalCheck) {
    log(`${label}: did NOT converge in ${maxRounds} rounds (${findingsPerRound.join(' -> ')}) — needs human review`)
    return done({ converged: false, rounds: maxRounds, open: open.map(findingBrief).slice(0, 30) })
  }
  // One pass over every finding of every round: checked against the current state, and fixed when still open or back.
  const all = numbered(sent)
  const check = await finalCheck(all)
  const items = Array.isArray(check?.items) ? check.items : []
  const statusOf = n => items.find(x => x?.n === n)?.status
  const stillOpen = all.filter(f => !['resolved', 'fixed-now', 'refuted'].includes(statusOf(f.n)))
  const finalSummary = Object.fromEntries(['resolved', 'fixed-now', 'refuted'].map(k => [k, all.filter(f => statusOf(f.n) === k).length]))
  finalSummary['still-open'] = stillOpen.length
  log(`${label}: ${maxRounds} round(s) (${findingsPerRound.join(' -> ')}); final check over ${all.length} finding(s): ${Object.entries(finalSummary).map(([k, v]) => `${v} ${k}`).join(', ')}`)
  return done({ converged: false, rounds: maxRounds, finalCheck: finalSummary, open: stillOpen.map(({ n, ...f }) => f).slice(0, 30) })
}

const tallyVotes = (votes, expected) => {
  const got = (votes ?? []).filter(Boolean)
  const approvals = got.filter(v => v.approve === true).length
  return {
    expected,
    received: got.length,
    approvals,
    quorum: got.length === expected,
    majority: approvals * 2 > expected,
    unanimous: got.length === expected && approvals === expected,
  }
}
