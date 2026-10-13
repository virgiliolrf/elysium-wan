// The flow every wan-x phase runs: the house template's machinery, written and reviewed once. A phase workflow
// (agents-architetura/tools/wf/src/wan-x-<slug>.js) is only its facts, its writers, its review lenses and its gate ids, and
// ends with `return await runPhase({...})`; wf-lint refuses one that re-implements this machinery. The M2 phase-1 workflow
// that re-wrote it inline reached 924 lines, and its review did not converge in six rounds because nearly every finding was
// about that wiring rather than the objective. Requires hygiene, schemas, dag, loop, project, gates, triage, fix and scope.
//
// runPhase({ slug, facts, work, lenses, gateIds, requires?, precondition?, unanimous? }):
// - slug: the phase slug. docs/design/NEXT-<slug>.md, docs/design/GATES-<slug>.json, `pinned.mjs <slug> --json` and the built
//   workflow .claude/workflows/wan-x-<slug>.js derive from it; the human grant covers all three.
// - facts: the text every prompt carries, copied from the NEXT document: paths, numbers with their sources, invariant ids,
//   every BA-<n> ruling, the tripwire, the exit gate.
// - work: [{ key, owns: [paths], task, binding?: ['BA-<n>'], wave?: 1, writer?: key }]. Writers of one wave run in parallel
//   on disjoint paths; waves run in order, and a later wave is told what the earlier ones reported. Items with the same
//   `writer` share one journal, which a later fix of their files starts from.
// - lenses: [[name, question], ...] or [name, question, { security: true }] for a trust boundary: the review lenses written
//   for this objective.
// - gateIds: every gate id pinned.mjs reports for GATES-<slug>.json ("rung:<id>", "pinned:<id>", ...). wf-lint checks the
//   list against the file; a relayed verdict that lacks one of them is not VERIFIED.
// - requires: more documents a human must have granted before anything runs (the ADR the phase builds on).
// - precondition: { cmd, why }: one command a solo verifier runs before any writer, with no tracked change allowed; any exit
//   code but 0 stops the phase and hands the human `why`.
// - unanimous: path prefixes whose stabilize fixes need every vote instead of a majority (default tests/).
//
// Decisions, so that a review does not re-file them as defects:
// - A review that does not converge does not stop the phase: Verify still runs, so the human decides with the gate's
//   evidence (the reference PR moved on after its rounds and let the gate decide). ok is false and `next` says why.
// - VERIFIED counts only when the relayed pinned JSON names this slug, lists every gate id, fails none, and ran on the tree
//   the scope audit saw before and after it. A pass on a tree that already failed a verify is intermittent, never VERIFIED,
//   and the tree may not change after the last verify.
// - The pinned gate runs for hours, longer than any agent's shell call: it runs detached (pinned.mjs --wait, one run per
//   verify round and tree) and the script waits in short slices (waitPinned in lib/gates.js) before one verifier relays it.
// - Every write is scope-audited mechanically where it happens: per wave, per review-fix round, per stabilize item, with
//   no audit over the whole run (a resumed run would count human-granted amendments as the phase's writes). A writer's own
//   report is also checked against its own paths. A spec conflict anywhere stops the phase for an ADR.
const PHASE_PINNED_RESULT = {
  type: 'object',
  required: ['pinned', 'items'],
  properties: {
    pinned: { type: 'object', description: 'the JSON pinned.mjs printed, verbatim' },
    items: TRIAGE.properties.items,
  },
}
const PHASE_PRE_RESULT = {
  type: 'object',
  required: ['exitCode', 'output'],
  properties: {
    exitCode: { type: 'integer', description: 'the real exit code of the command' },
    output: { type: 'string', description: 'everything the command printed, verbatim' },
  },
}
// The reference PR's stabilize budget: up to 4 rounds of fix -> verify again.
const PHASE_MAX_STABILIZE = 4
const PHASE_SLUG_RE = /^[a-z0-9][a-z0-9-]{1,38}$/
const PHASE_KEY_RE = /^[a-z0-9][a-z0-9-]{0,39}$/
const phaseCovers = (paths, p) => paths.some(o => p === o || (o.endsWith('/') && p.startsWith(o)))
// Agents sometimes report a file absolutely (/.../<repo>/src/x.py): map it back to the repo-relative path it ends with.
const phaseRel = (f, owned) => {
  const s = String(f ?? '').replace(/^\.\//, '')
  if (!s.startsWith('/')) return s
  for (const o of owned) {
    const i = o.endsWith('/') ? s.indexOf(`/${o}`) : (s.endsWith(`/${o}`) ? s.length - o.length - 1 : -1)
    if (i >= 0) return s.slice(i + 1)
  }
  return s
}

const phaseConfigErrors = ({ slug, facts, work, lenses, gateIds, requires, precondition, unanimous }) => {
  const errors = []
  if (!PHASE_SLUG_RE.test(String(slug ?? ''))) errors.push(`slug ${JSON.stringify(slug)} must match ${PHASE_SLUG_RE}`)
  if (typeof facts !== 'string' || !facts.trim()) errors.push('facts must be the non-empty FACTS text')
  if (!Array.isArray(work) || !work.length) errors.push('work must list at least one writer')
  for (const w of Array.isArray(work) ? work : []) {
    if (!PHASE_KEY_RE.test(String(w?.key ?? ''))) errors.push(`work key ${JSON.stringify(w?.key)} must match ${PHASE_KEY_RE}`)
    if (w?.writer !== undefined && !PHASE_KEY_RE.test(String(w.writer))) errors.push(`work ${w?.key}: writer must match ${PHASE_KEY_RE}`)
    if (!Array.isArray(w?.owns) || !w.owns.length || !w.owns.every(p => typeof p === 'string' && safeScopePath(p))) errors.push(`work ${w?.key}: owns must be a non-empty list of safe repo-relative paths`)
    if (typeof w?.task !== 'string' || !w.task.trim()) errors.push(`work ${w?.key}: task is required`)
    if (w?.wave !== undefined && !(Number.isInteger(w.wave) && w.wave >= 1)) errors.push(`work ${w?.key}: wave must be a positive integer`)
  }
  const list = Array.isArray(work) ? work : []
  if (new Set(list.map(w => w?.key)).size !== list.length) errors.push('work keys must be unique')
  for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
    const a = list[i]
    const b = list[j]
    if ((a?.wave ?? 1) === (b?.wave ?? 1) && (a?.owns ?? []).some(x => (b?.owns ?? []).some(y => scopesOverlap(x, y)))) errors.push(`work ${a?.key} and ${b?.key} run in the same wave on overlapping paths`)
  }
  if (!Array.isArray(lenses) || lenses.length < 3 || !lenses.every(l => Array.isArray(l) && PHASE_KEY_RE.test(String(l[0] ?? '')) && typeof l[1] === 'string' && l[1].trim())) errors.push('lenses must be at least three [name, question] pairs written for this objective')
  if (Array.isArray(lenses) && new Set(lenses.map(l => Array.isArray(l) ? l[0] : l)).size !== lenses.length) errors.push('lens names must be unique')
  if (!Array.isArray(gateIds) || !gateIds.length || !gateIds.every(g => typeof g === 'string' && g)) errors.push('gateIds must list every gate id of the GATES file')
  if (!Array.isArray(requires) || !requires.every(d => typeof d === 'string' && safeScopePath(d))) errors.push('requires must be a list of repo-relative documents')
  if (precondition !== null && (typeof precondition?.cmd !== 'string' || !precondition.cmd.trim() || typeof precondition?.why !== 'string' || !precondition.why.trim())) errors.push('precondition must be { cmd, why }')
  if (!Array.isArray(unanimous) || !unanimous.every(u => typeof u === 'string' && u)) errors.push('unanimous must be a list of path prefixes')
  return errors
}

const runPhase = async ({ slug, facts, work, lenses, gateIds, requires = [], precondition = null, unanimous = ['tests/'] }) => {
  // The configuration is checked before any agent runs; the grant check is still the first agent call.
  const configErrors = phaseConfigErrors({ slug, facts, work, lenses, gateIds, requires, precondition, unanimous })
  if (configErrors.length) throw new Error(`runPhase: ${configErrors.join('; ')}`)
  const NEXT = `docs/design/NEXT-${slug}.md`
  const GATES = `docs/design/GATES-${slug}.json`
  const WORKFLOW = `.claude/workflows/wan-x-${slug}.js`
  const OWNED = [...new Set(work.flatMap(w => w.owns))]
  const writerOf = w => w.writer ?? w.key
  const journal = w => journalPath(slug, writerOf(w))
  const journals = [...new Map(work.map(w => [writerOf(w), { writer: writerOf(w), owns: w.owns, journal: journal(w) }])).values()]
  // the owner journal of a fix comes from this phase's writers, never from an agent's text
  const ownerOf = scope => {
    const w = work.find(x => scope.every(p => phaseCovers(x.owns, p)))
    return w ? { ownerTask: writerOf(w), ownerJournal: journal(w) } : {}
  }
  const lensOpts = Object.fromEntries(lenses.map(([name, , o]) => [name, o ?? {}]))

  phase('Gate')
  const docs = [NEXT, GATES, WORKFLOW, ...requires]
  const grant = await checkApprovals(docs, 'gate:approval', 'Gate')
  if (!approvedAll(grant, docs)) throw new Error(`${docs.join(', ')} are not all approved by a human — run: node agents-architetura/tools/factory/approve.mjs grant ${docs.join(' ')}`)

  // A writer that finds the spec or design wrong stops the phase: the conflict becomes an ADR for the human, never a workaround.
  const conflictStop = (where, results) => {
    const c = results.find(r => r?.specConflict)?.specConflict
    return c ? { ok: false, stoppedAt: where, specConflict: c, next: 'run /wan-adr with this conflict as the problem; the phase resumes after the human grant' } : null
  }
  // Fail closed on the audit's own report: reviews are driven by `changed`, which the AUDIT schema does not require.
  const audited = async (label, phaseTitle, since, scopes) => {
    const a = await scopeCheck(label, phaseTitle, since, scopes)
    if (!Array.isArray(a.changed) || !a.changed.every(f => typeof f === 'string') || !TREE_ID_RE.test(a.tree ?? '')) {
      throw new Error(`scope audit (${label}) did not report its changed files and its tree — stopping rather than skipping a review`)
    }
    if ((a.tree === since) !== (a.changed.length === 0)) {
      throw new Error(`scope audit (${label}) is inconsistent: tree ${a.tree === since ? 'unchanged' : 'changed'} with ${a.changed.length} changed file(s)`)
    }
    return a
  }

  if (precondition) {
    const since = await scopeSnapshot('precondition', 'Gate')
    const pre = await agent(common('You run ALONE — you MAY run this one command. Change nothing else, fix nothing.') + `${facts}
Run exactly this command from the repo root, once (a multi-line script: pipe it into bash with a quoted here-document):
\`\`\`
${precondition.cmd}
\`\`\`
Return its real exit code and everything it printed, verbatim. Never edit a file, a digest or a saved copy to make it pass.`,
      { label: 'gate:precondition', phase: 'Gate', schema: PHASE_PRE_RESULT, agentType: ROLE.verifier })
    await audited('precondition', 'Gate', since, [])
    if (pre?.exitCode !== 0) return { ok: false, stoppedAt: 'precondition', reason: clean(pre?.output ?? 'the precondition agent returned nothing', 1500), next: precondition.why }
  }

  phase('Implement')
  const changedInPhase = new Set()
  const reported = []
  const waves = [...new Set(work.map(w => w.wave ?? 1))].sort((a, b) => a - b)
  for (const wave of waves) {
    const items = work.filter(w => (w.wave ?? 1) === wave)
    const since = await scopeSnapshot(`implement-wave${wave}`, 'Implement')
    const earlier = reported.length ? `\nWhat earlier waves reported (data; re-read the files themselves): ${fence('earlier_waves', reported, 6000)}` : ''
    const results = await parallel(items.map(w => () => agent(common(`OWNED PATHS: ${w.owns.join(', ')} (only).`) + `${facts}
TASK (${w.key}): ${w.task}
Binding audit items whose ruling you deliver (see the facts): ${(w.binding ?? []).join(', ') || 'none'}. A tripwire stays
refused unless this phase arms it. Append your decisions and assumptions to ${journal(w)}: a later fix of your files starts
from it. If the spec or design is wrong for this task, set specConflict and stop.${earlier}
Return every file you created or modified.`,
      { label: `impl:${w.key}`, phase: 'Implement', schema: TASK_RESULT, agentType: ROLE.coder })))
    const missing = items.filter((_, i) => !results[i])
    if (missing.length) throw new Error(`writer(s) ${missing.map(w => w.key).join(', ')} of wave ${wave} produced nothing — resume this run`)
    // the audit comes first: a writer that reports a conflict after writing outside its paths still stops on the write
    const audit = await audited(`implement-wave${wave}`, 'Implement', since, [...new Set(items.flatMap(w => w.owns))])
    const stop = conflictStop('implement', results)
    if (stop) return stop
    // the writer's own journal (gitignored run state its prompt tells it to write) is not a file outside its paths
    items.forEach((w, i) => {
      const outside = (results[i].files ?? []).map(f => phaseRel(f, [...OWNED, journal(w)])).filter(f => !phaseCovers(w.owns, f) && f !== journal(w))
      if (outside.length) throw new Error(`writer ${w.key} reports files outside its own paths: ${clean(outside.join(', '), 400)}`)
    })
    for (const f of audit.changed) changedInPhase.add(f)
    reported.push(...items.map((w, i) => ({ key: w.key, summary: clean(results[i].summary, 1500), files: (results[i].files ?? []).slice(0, 40) })))
  }

  // Each fix round is bracketed by its own scope audit and adds what it changed to the files the next round reviews; a spec
  // conflict ends the loop at once.
  const SPEC_STOP = Symbol('spec-conflict')
  const reviewLoop = async (stage, phaseTitle, what, files) => {
    let fixRounds = 0
    try {
      const r = await reviewUntilClean({
        label: stage,
        lenses: lenses.map(([name, q]) => [name, q]),
        ownsFile: f => phaseCovers(OWNED, phaseRel(f, OWNED)),
        review: (name, lens, round) => agent(common('READ-ONLY.') + `${facts}${lensOpts[name].security ? SECURITY_COST : ''}
LENS (${name}, round ${round}): ${lens}
${what}: ${fence('files', [...files], 4000)}
Scope "house" is only for a defect in code this phase did not write and does not need changed (outside ${OWNED.join(', ')}):
it goes to the factory backlog. Anything this phase wrote, or must change, is "change".`,
          { label: `${stage}:${name}:r${round}`, phase: phaseTitle, schema: FINDINGS, agentType: lensOpts[name].security ? ROLE.security : ROLE.reviewer }),
        fix: async (findings, round) => {
          fixRounds++
          const since = await scopeSnapshot(`${stage}-fix-r${round}`, phaseTitle)
          const r = await agent(common(`OWNED PATHS: ${OWNED.join(', ')} (only).`) + `${facts}
Fix these validated findings at their root cause, blocker and major first (verify each against the code first; a minor is
cheap: fix it too unless it conflicts with a serious fix). You run no command. Append what you changed and why to the journal
of the writer whose files you change: ${fence('journals', journals, 3000)}
A finding whose fix needs a path outside yours or a spec change is a specConflict: set it and stop.
${fence('findings', findings, 12000)}`,
            { label: `${stage}-fix:r${round}`, phase: phaseTitle, schema: TASK_RESULT, agentType: ROLE.coder })
          const a = await audited(`${stage}-fix-r${round}`, phaseTitle, since, OWNED)
          for (const f of a.changed) files.add(f)
          const stop = conflictStop(stage, [r])
          if (stop) throw Object.assign(new Error(`${stage}: spec conflict`), { [SPEC_STOP]: stop })
          return r
        },
      })
      return { ...r, fixRounds }
    } catch (e) {
      if (e?.[SPEC_STOP]) return { converged: false, fixRounds, stop: e[SPEC_STOP] }
      throw e
    }
  }

  phase('Review')
  const review = await reviewLoop('review', 'Review', 'Files changed in this phase (the scope audits of every wave)', changedInPhase)
  if (review.stop) return review.stop
  if (!review.converged) log(`Review did not converge (${review.findingsPerRound.join(' -> ')}): Verify still runs so the human decides with the gate's evidence`)

  // The script decides: VERIFIED only when the relayed pinned JSON is this phase's, complete, green, and of one audited tree.
  const history = []
  const failingOf = v => {
    const p = v?.pinned
    const ids = [...(Array.isArray(p?.failing) ? p.failing : []), ...(Array.isArray(p?.gates) ? p.gates.filter(g => g?.ok !== true).map(g => g?.id) : [])]
    return [...new Set(ids.map(x => clean(String(x), 80)))].slice(0, 40)
  }
  const isVerified = v => {
    const p = v?.pinned
    if (p?.verdict !== 'VERIFIED' || p.milestone !== slug || !Array.isArray(p.failing) || p.failing.length) return false
    if (!TREE_ID_RE.test(p.tree ?? '') || p.tree !== v.since || p.tree !== v.tree) return false
    const gates = Array.isArray(p.gates) ? p.gates : []
    return gates.every(g => g?.ok === true) && gateIds.every(id => gates.some(g => g?.id === id))
  }
  // The verifier runs a shell, so its run is bracketed by a scope audit that allows no tracked change.
  // The gate runs detached for hours (waitPinned); the verifier then reads its stored verdict, which prints at once.
  const verifyOnce = async round => {
    const since = await scopeSnapshot(`verify-r${round}`, 'Verify')
    if (!(await waitPinned(slug, `r${round}`, 'Verify'))) throw new Error(pinnedStillRunning(slug, `r${round}`))
    const v = await agent(common('You run ALONE — you MAY build and run anything (no git). Fix nothing.') + `${facts}
Round ${round}.${round ? ' Fixes were applied since the last verify: re-establish ground truth yourself.' : ''}
1. Run exactly: ${pinnedCommand(slug, `r${round}`)}
   The gate has finished, so it prints its stored verdict at once (exit 2: the run failed; relay what it printed).
   Put the JSON it prints into "pinned" EXACTLY as printed. Do not edit it, summarize it or rerun a gate for a better result.
2. For every failing gate, one INDEPENDENT item: id, the gate id as "rung", the symptom, exact evidence (error text, failing
   test ids, the repro command), the suspected cause, its root-cause "mechanism" (the same words for failures that share a
   cause), and the minimal scope inside ${OWNED.join(', ')}. Merge failures that share a root-cause file.
Edit no file: the run is scope-audited and any tracked change stops the phase.`,
      { label: `verify:r${round}`, phase: 'Verify', schema: PHASE_PINNED_RESULT, agentType: ROLE.verifier })
    const a = await audited(`verify-r${round}`, 'Verify', since, [])
    const out = v ? { ...v, since, tree: a.tree } : null
    history.push({ round, tree: a.tree, verified: isVerified(out), failing: failingOf(out) })
    return out
  }

  phase('Verify')
  let check = await verifyOnce(0)
  const rejected = []
  const changedInStabilize = new Set()
  const needsAll = it => it.scope.some(p => unanimous.some(u => p.startsWith(u)))
  let round = 0
  while (check && !isVerified(check) && round < PHASE_MAX_STABILIZE) {
    round++
    phase('Stabilize')
    const normalized = normalizeItems(check.items, { knownRungs: [...RUNG_IDS, ...gateIds] })
    // The reference PR's rounds: one dominant root cause at a time.
    const dominant = dominantMechanism(normalized.items)
    if (dominant.focus) log(`Stabilize ${round}: one mechanism explains ${dominant.explained}/${dominant.of} failures — fixing only it this round`)
    const scoped = dominant.items.filter(it => it.scope.every(p => phaseCovers(OWNED, p)))
    for (const d of [...normalized.dropped, ...dominant.items.filter(it => !scoped.includes(it))]) log(`Stabilize ${round}: item ${d.id} dropped (${d.why ?? 'outside this phase\'s owned paths'})`)
    if (!scoped.length) { log(`Stabilize ${round}: the gate is red and no item is inside this phase's owned paths — a human decides`); break }
    // One item at a time (no two appliers run tests together), each bracketed by a scope audit on its own scope.
    const outcomes = []
    for (const it of withPriorRejections(scoped, rejected).map(x => ({ ...x, ownerTask: undefined, ownerJournal: undefined, ...ownerOf(x.scope) }))) {
      const since = await scopeSnapshot(`stabilize-r${round}-${it.id}`, 'Stabilize')
      const [o] = await runFixPipeline([it], { phaseTitle: 'Stabilize', source: `a failing gate of ${GATES}`, context: facts, unanimous: needsAll })
      const a = await audited(`stabilize-r${round}-${it.id}`, 'Stabilize', since, it.scope)
      // whatever the applier's outcome, a changed file is reviewed below
      for (const f of a.changed) changedInStabilize.add(f)
      if (o) outcomes.push(o)
      const stop = specConflictStop(outcomes, 'stabilize')
      if (stop) return stop
    }
    rejected.push(...rejectionsOf(outcomes, round))
    phase('Verify')
    check = await verifyOnce(round)
  }

  // The Review above covered the code before Stabilize. Code Stabilize changed is reviewed with the same lenses before a
  // VERIFIED counts; if that review fixes anything, the pinned gate runs once more on the final tree.
  let finalReview = { converged: true, rounds: 0, findingsPerRound: [], discarded: 0, minors: [], house: [] }
  if (changedInStabilize.size && isVerified(check)) {
    finalReview = await reviewLoop('stabilize-review', 'Stabilize', 'These files changed during Stabilize, after the first review', changedInStabilize)
    if (finalReview.stop) return finalReview.stop
    if (finalReview.fixRounds) {
      phase('Verify')
      check = await verifyOnce(round + 1)
    }
  }

  // No audit over the whole run: every write is audited where it happens (per wave, per fix round, per stabilize item),
  // and a run resumed after a human-granted amendment (an ADR, the NEXT document, approvals) would count the human's
  // edits as the phase's (M2 phase 1 crashed on exactly this).
  const green = isVerified(check)
  const finalTree = green ? await scopeSnapshot('final', 'Verify') : null
  const intermittent = green ? history.filter(h => !h.verified && h.tree === check.tree).map(({ round: r, failing }) => ({ round: r, failing })) : []
  const ok = green && !intermittent.length && finalTree === check.tree && review.converged && finalReview.converged
  const failing = failingOf(check)
  const next = ok ? 'rerun wan-next to write the next phase from the new state'
    : intermittent.length ? `a gate failed earlier on the same tree it then passed on (${intermittent.flatMap(h => h.failing).join(', ')}): intermittent, not VERIFIED — run /wan-bughunt on it`
      : green && finalTree !== check.tree ? 'the tree changed after the last verify: rerun this phase so its verdict is of the final tree'
        : !green ? `NOT VERIFIED after ${round} stabilize round(s)${failing.length ? ` (failing: ${failing.join(', ')})` : ''}: a human decides — rerun wan-next from the new state, or /wan-bughunt on the failing gate`
          : review.stalled || finalReview.stalled ? 'VERIFIED, but the review stalled (its blocker/major count stopped falling): the phase is too big — split it (rerun wan-next for a smaller objective); its findings are in "open"'
            : `VERIFIED, but the ${review.converged ? 'post-stabilize review' : 'review'} did not converge: its findings are in "open". A human decides: record them and accept, or fix them and rerun this phase.`
  log(`wan-x-${slug}: ${ok ? 'VERIFIED' : 'NOT VERIFIED'} after ${round} stabilize round(s)`)
  return {
    ok,
    review,
    finalReview,
    pinned: check?.pinned ?? null,
    stabilizeRounds: round,
    rejected,
    intermittent,
    open: [...(review.open ?? []), ...(finalReview.open ?? [])],
    minors: [...(review.minors ?? []), ...(finalReview.minors ?? [])],
    // defects in shared code the reviewers found on the way: the factory backlog, not this phase's
    house: [...(review.house ?? []), ...(finalReview.house ?? [])],
    next,
  }
}
