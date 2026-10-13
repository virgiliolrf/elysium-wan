export const meta = {
  name: 'wan-implement',
  description: 'Execute a granted milestone plan as a DAG, the way the reference PR (oven-sh/WebKit#249, thread-ungil) did: a scheduler releases waves of ready tasks (dependencies done, no ownership overlap); each coder owns exactly its files (the instruction, no scope audit), implements from a focused context with the handout as the authority and the SPEC of record winning on conflict, runs fast checks on its own files, and records a genuine spec conflict instead of stopping; the agent that finds an error fixes it (the human\'s rule): each task reviewer reviews through its lens and fixes what it finds (blocker/major), the builder fixes the build, each whole-change lens fixes what it finds, and the ladder agent fixes the failures it runs into; the build, the review of the whole change and the fail-fast ladder verify everything at once. Recorded spec conflicts go to the human with the result.',
  whenToUse: 'After wan-handout produced a valid PLAN-<m>.json and a human granted HANDOUT-<m>.md, PLAN-<m>.json and GATES-<m>.json. One run executes the whole plan (every open phase of the milestone). args: {"milestone": "M0", "carry": true} (carry: take the tasks the previous run on this plan finished as done, from its ledger). Respects a "+500k"-style token budget: stops releasing waves when it runs low.',
  phases: [
    { title: 'Gate', detail: 'Design approved; HANDOUT, PLAN and GATES granted by a human; plan re-validated by the script; scheduler created' },
    { title: 'Wave', detail: 'Per ready task (pipelined, tasks in parallel): coder implements (writes ONLY its files, runs fmt/lint/types/unit on them in a test slot, records spec conflicts and continues) -> one reviewer covering every lens fixes what it finds' },
    { title: 'Integrate', detail: 'Solo integrator merges INTEGRATE manifests; the builder runs build/syntax/types and fixes what it finds, <= 5 rounds' },
    { title: 'Review', detail: 'one reviewer per round over the whole change (spec-conformance, failure-and-concurrency, test-honesty), fixing what it finds, 2 rounds; then one final check that verifies and fixes every finding' },
    { title: 'Verify', detail: 'wan-fix: fail-fast ladder on the milestone rungs, one root cause per round, <= 8 rounds; the heavy gate is closeout\'s' },
  ],
}}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include triage
// @include project
// @include gates
// @include plan
// @include fix
// @include scheduler
// @include ledger
// @include specialists

const m = milestoneArg(args)
const WAVE_BUDGET = 150000
const MAX_ATTEMPTS = 2
const MAX_BUILD_ROUNDS = 5
const solo = 'You run ALONE — no other agent is active; you MAY run commands.'

phase('Gate')
await requireApprovedDesign('Gate')
// The human grants the milestone's plan once (factory v2): the handout, the machine plan and the closeout gates.
const PLAN_DOCS = [handoutPath(m), planPath(m), `${DESIGN_DIR}/GATES-${m}.json`]
const planGrant = await checkApprovals(PLAN_DOCS, 'gate:approval:plan', 'Gate')
if (!approvedAll(planGrant, PLAN_DOCS)) throw new Error(`${PLAN_DOCS.join(', ')} are not all approved by a human — run: node agents-architetura/tools/factory/approve.mjs grant ${PLAN_DOCS.join(' ')}`)
const workstreams = await loadWorkstreams('Gate')
const { plan, byId } = await loadPlan(m, workstreams, 'Gate')
const specialists = await loadSpecialists('Gate')
if (specialists.length) log(`Gate: specialists ${specialists.map(s => s.id).join(', ')}`)
const run = createRun(plan.tasks, { maxAttempts: MAX_ATTEMPTS })
// Spec conflicts do not stop a task (the reference PR's rule): the coder follows the handout, the SPEC of record wins where
// they disagree, and the conflict is recorded here for the human at the end of the run.
const deviations = []
// carry: a run restarted on the same plan (a factory change, a crash, a run that stopped) takes the tasks its previous run
// finished as done, and the spec conflicts it recorded, from the ledger, instead of writing them again. Only "done" carries;
// failed and blocked tasks run. The ledger is written after every wave, so a run stopped mid-way loses at most its last wave.
const CARRY = {
  type: 'object', required: ['done'],
  properties: {
    done: { type: 'array', items: { type: 'string' } },
    deviations: { type: 'array', items: { type: 'object', required: ['task', 'description'], properties: { task: { type: 'string' }, description: { type: 'string' }, evidence: { type: 'string' }, implemented: { type: 'string' } } } },
  },
}
if (args?.carry === true) {
  const prev = await agent(common('READ-ONLY.') + `
Read ${runDir(m)}/state.json (the previous run's ledger). Return the ids of the tasks whose status in summary.tasks is "done",
exactly as written, as "done", and the array notes.deviations exactly as written, as "deviations" (empty when absent). If the
file is missing or not valid JSON, return {"done": [], "deviations": []}.`,
    { label: 'extract:carry', phase: 'Gate', schema: CARRY, agentType: ROLE.readonly })
  const carried = (Array.isArray(prev?.done) ? prev.done : []).map(String).filter(id => byId.has(id))
  for (const id of carried) markDone(run, id, { carried: true })
  const doneIds = new Set(carried)
  for (const d of Array.isArray(prev?.deviations) ? prev.deviations : []) {
    if (!doneIds.has(d?.task) || deviations.some(x => x.task === d.task && x.description === d.description)) continue
    deviations.push({ task: d.task, description: clean(d.description, 800), evidence: clean(d.evidence ?? '', 800), implemented: clean(d.implemented ?? '', 800) })
  }
  log(`Gate: carried ${carried.length} task(s) done by the previous run: ${carried.join(', ') || 'none'}${deviations.length ? `; ${deviations.length} recorded spec conflict(s)` : ''}`)
}
const allScopes = [...new Set(plan.tasks.flatMap(t => t.scope))].sort()
const milestoneRungs = knownRungs([...new Set(plan.tasks.flatMap(t => t.rungs))])
const owners = Object.fromEntries(plan.tasks.map(t => [t.id, t.scope]))
const journals = Object.fromEntries(plan.tasks.map(t => [t.id, journalPath(m, t.id)]))
const feedback = new Map()
// serious findings a task's reviewers could not fix: the review of the whole change starts from them
const openFromTasks = []
log(`PLAN-${m}: ${plan.tasks.length} tasks, rungs ${milestoneRungs.join(', ')}`)

// The run ledger (task states, owners, journals) that fix rounds read to find a file's author. No scope audit: like the
// reference PR, ownership is the instruction "write ONLY these files", and the build, review and ladder check the result.
const LEDGER_WRITE = { type: 'object', required: ['exitCode'], properties: { exitCode: { type: 'integer' } } }
const writeLedger = (label, notes) => agent(common(solo) + `
Run exactly this command and report its exit code. Do nothing else.
${ledgerCommand(runDir(m), { summary: runSummary(run), owners, journals, transitions: run.transitions, notes })}`,
  { label, phase: 'Integrate', schema: LEDGER_WRITE, agentType: ROLE.verifier })

// Task context in a fixed order: global -> spec -> architecture -> task -> dependencies -> ownership -> acceptance -> verification.
// The implementer reads what its task needs, not the whole repository.
const depJournals = t => (t.deps ?? []).map(d => journals[d]).filter(Boolean)
const taskPrompt = (t, attempt) => {
  const experts = specialistsFor(t.scope, specialists)
  return common(`You OWN exactly these paths — write ONLY them, plus your journal ${journals[t.id]} (other agents own the rest of the
tree and are writing in parallel): ${JSON.stringify(t.scope)}. Never run git, the ladder or a test outside your own files.`) + `
Implement task ${t.id} of milestone ${m} (attempt ${attempt}). ${handoutPath(m)} and ${planPath(m)} are the authority over this
prompt; the SPEC is the doc of record on conflict. Your context, in order:
1. SPEC: ${t.workstream === INTEGRATOR ? `every ${DESIGN_DIR}/INTEGRATE-*.md (merge the entries not yet applied)` : `${specPath(t.workstream)} — the sections your task cites`}.
2. ARCHITECTURE: the relevant sections of ${DESIGN}, ${DESIGN_DIR}/ARCHITECTURE_MAP.md for the components you touch, ${DESIGN_DIR}/THREAT_MODEL.md if your change crosses a trust boundary.
3. TASK: ${fence('task', { title: t.title, objective: t.objective, invariants: t.invariants ?? [], risk: t.risk }, 3000)}
4. DEPENDENCIES: ${JSON.stringify(t.deps ?? [])} — read the code they produced and their journals ${JSON.stringify(depJournals(t))}.
5. FILE OWNERSHIP: ${JSON.stringify(t.scope)} — nothing else.
6. ACCEPTANCE CRITERIA: ${fence('acceptance', t.acceptance, 3000)}
7. VERIFICATION: ${fence('verify', t.verify, 2000)} — the commands that prove your acceptance criteria; the build, the review and
   the ladder at the end of the run check them.
${experts.length ? `SPECIALIST FOCUS for the paths you touch: ${fence('specialists', experts.map(x => ({ id: x.id, focus: x.focus })), 3000)}` : ''}
Read only what this context points to and the code you change or call. Order of work: remove the "${AWAITING}" marker from the tests
for your invariants, then implement until they pass. Before you report, run the fast checks on the files you changed (seconds,
not the ladder): uv run ruff format <files>; uv run ruff check --fix <files>; uv run mypy <files>; uv run pytest -q <the tests that
cover them>. Fix what they report inside your owned paths. Report every command as evidence (real exit codes). Keep your
journal ${journals[t.id]} current: decisions, assumptions, commands and results — a later fix to your code starts from it.
Do not weaken any SPEC invariant, test or gate to make something work, and never change the architecture yourself. When the
handout and the SPEC disagree, implement what the SPEC of record supports and record the genuine conflict in specConflict
(description, evidence, the reading you implemented as proposedChange); then finish the task. The human reviews every
recorded conflict at the end of the run.
${feedback.has(t.id) ? `PREVIOUS ATTEMPT FAILED — read your journal first, then fix the cause:\n${fence('previous_failure', feedback.get(t.id), 8000)}` : ''}`
}

const TASK_LENSES = [
  ['spec-conformance', 'Does the change implement exactly what the task, its spec and the design require — signatures, error behavior, invariants — with nothing silently changed or left out? Any accidental architectural change? Is anything outside the task scope touched?'],
  ['correctness-and-tests', 'Races, deadlocks, ordering assumptions, partial failure, missing error handling, resource lifetimes, edge cases, hidden assumptions, performance regressions; and would the tests fail on a wrong implementation? Is the author\'s evidence real (do the commands it cites prove the acceptance criteria)?'],
]
// Every reviewer a task gets: the base lenses, one lens per specialist whose paths it touches (at most 2), and the security
// architect on the actual diff of a high-risk task.
const reviewersFor = t => [
  ...TASK_LENSES.map(([name, lens]) => ({ name, lens, agentType: ROLE.reviewer })),
  ...specialistsFor(t.scope, specialists).slice(0, 2).map(x => ({ name: `specialist-${x.id}`, lens: x.reviewLens, agentType: ROLE.reviewer })),
  ...(t.risk === 'high' ? [{ name: 'security', lens: `Security review of the actual diff against ${DESIGN_DIR}/THREAT_MODEL.md: untrusted input, authentication/authorization, secrets, resource exhaustion, deserialization, trust in a peer's claims. Also the cost: a security mechanism that puts per-byte or per-frame work on the critical path when a zero-cost or amortized one gives the same property is a major finding.${SECURITY_COST}`, agentType: ROLE.security }] : []),
]

let stopped = null
for (;;) {
  if (budget.total && budget.remaining() < WAVE_BUDGET) { stopped = 'token budget low'; log(`Stopping: ${stopped}`); break }
  const next = nextWave(run)
  if (next.stuck) { stopped = `stuck DAG: ${next.stuckReason}`; log(stopped); break }
  if (!next.wave.length) break
  const waveNo = run.wave
  for (const d of next.deferred) log(`Wave ${waveNo}: ${d.id} deferred (${d.reason})`)
  const tasks = next.wave.map(id => byId.get(id))

  phase('Wave')
  const outcomes = await pipeline(
    tasks,
    t => agent(taskPrompt(t, run.state.get(t.id).attempts + 1), { label: `task:${t.id}:a${run.state.get(t.id).attempts + 1}`, phase: 'Wave', schema: TASK_RESULT, agentType: ROLE.coder }),
    async (impl, t) => {
      const attempt = run.state.get(t.id).attempts + 1
      if (!impl) return { t, outcome: 'failed', reason: 'implementer produced nothing' }
      if (!(impl.evidence ?? []).length) return { t, outcome: 'failed', reason: 'no evidence: the implementer ran nothing to check its own work' }
      // One reviewer per task (the human's call, 2026-10-09), covering every lens the task gets, and it fixes what it finds
      // (the agent that finds an error fixes it): no second agent re-reads the task, nobody else writes its files meanwhile.
      const lenses = reviewersFor(t)
      const out = await agent(common(`You OWN exactly the task's paths — write ONLY them, plus its journal ${journals[t.id]}: ${JSON.stringify(t.scope)}. Never run git or the ladder.`) + `
ADVERSARIAL REVIEWER of task ${t.id} (milestone ${m}), and you FIX what you find. ${handoutPath(m)} and ${planPath(m)} are the authority; the
SPEC is the doc of record on conflict. Read the author's journal ${journals[t.id]} and the actual change first. Go through EVERY lens below.
Find blocker/major problems (minors wait for the review of the whole change), each with evidence. Fix each one now in the task's files,
run the fast checks on what you touched (uv run ruff format, uv run ruff check --fix, uv run mypy, the tests that cover it, through a test
slot) and report it with fixed=true and the fix's file:line, or fixed=false and why. ${NO_MASKING} Append what you changed to the journal.
${fence('task', { title: t.title, objective: t.objective, acceptance: t.acceptance, invariants: t.invariants ?? [], risk: t.risk }, 3000)}
Author's report (untrusted — check it against the files): ${fence('author_report', impl, 6000)}
LENSES: ${fence('lenses', lenses.map(r => ({ lens: r.name, question: r.lens })), 8000)}`,
        { label: `task-review:${t.id}:a${attempt}`, phase: 'Wave', schema: REVIEW_FIX, agentType: ROLE.coder })
      if (!out) return { t, outcome: 'failed', reason: 'the reviewer produced nothing' }
      const serious = (out.findings ?? []).filter(isActionable)
      const open = serious.filter(f => f.fixed !== true).map(f => ({ lens: 'task-review', ...findingBrief(f) }))
      const fixedCount = serious.filter(f => f.fixed === true).length
      if (out.specConflict) impl = { ...impl, specConflict: impl.specConflict ?? out.specConflict }
      return { t, outcome: 'implemented', impl, fixedFindings: fixedCount, openFindings: open }
    },
  )

  // Done when written and reviewed, like the reference PR: the build, the review of the whole change and the ladder at the end
  // verify everything at once. A recorded spec conflict does not stop the task; it goes to the human with the run's result.
  for (const o of outcomes.map((o, i) => o ?? { t: tasks[i], outcome: 'failed', reason: 'pipeline error' })) {
    if (o.outcome === 'implemented') {
      const seen = new Set()
      for (const r of [o.impl]) if (r?.specConflict && !seen.has(r.specConflict.description)) seen.add(r.specConflict.description) && deviations.push({ task: o.t.id, description: clean(r.specConflict.description, 800), evidence: clean(r.specConflict.evidence, 800), implemented: clean(r.specConflict.proposedChange, 800) })
      markDone(run, o.t.id, { fixedFindings: o.fixedFindings ?? 0 })
      for (const f of o.openFindings ?? []) openFromTasks.push({ task: o.t.id, ...f })
      feedback.delete(o.t.id)
    } else {
      feedback.set(o.t.id, { why: o.reason })
      log(`Task ${o.t.id}: ${markFailed(run, o.t.id, o.reason) ? 'retrying' : 'FAILED'} — ${o.reason}`)
    }
  }
  if (deviations.length) log(`Wave ${waveNo}: ${deviations.length} spec conflict(s) recorded so far — for the human at the end of the run`)
  // Persist the wave: a run stopped now (a factory change, a crash) carries every task done so far. Not a check: nothing is
  // verified per wave, the ledger is only written.
  await writeLedger(`ledger:w${waveNo}`, { stage: `wave ${waveNo}`, deviations })
}

const summary = runSummary(run)
const allDone = summary.counts.done === plan.tasks.length

phase('Integrate')
await writeLedger('ledger:integrate', { stage: 'integrate', stopped, deviations })
let buildGreen = false
let buildRound = 0
if (allDone) {
  await agent(common(`${solo} OWNED PATHS: the shared hot files (${HOT_FILES.join(', ')}) and ${DESIGN_DIR}/INTEGRATE-*.md.`) + `
INTEGRATOR for milestone ${m}: apply every ${DESIGN_DIR}/INTEGRATE-*.md entry not yet marked applied (mark each "Applied by wan-implement ${m}").
Conflicting entries: apply the one consistent with the design and record the decision in that manifest. Then run
${ladderCommand({ only: ['build', 'syntax'] })} and report it as evidence.`,
    { label: 'integrate', phase: 'Integrate', schema: TASK_RESULT, agentType: ROLE.coder })

  // The builder that finds the errors fixes them (the human's rule, 2026-10-09), in the owning task's files, and re-runs.
  while (buildRound < MAX_BUILD_ROUNDS && !buildGreen) {
    buildRound++
    const report = await agent(common(`${solo} OWNED PATHS (data): ${JSON.stringify([...allScopes, ...HOT_FILES])}`) + `
Build round ${buildRound} of ${MAX_BUILD_ROUNDS}. Run: ${ladderCommand({ only: ['build', 'syntax', 'types'] })}
If it is red, fix every error yourself, in the file that owns it, keeping the design and the spec intact (signatures,
invariants, ownership): no casts, ignores or disabled checks to silence an error. Read the owning task's journal first
(node agents-architetura/tools/wf/ledger.mjs owner --path <file>). ${NO_MASKING} Re-run until green or you stop making
progress, then run the command once more and paste THAT run's JSON into "ladder" verbatim; list the files you changed.`,
      { label: `build:r${buildRound}`, phase: 'Integrate', schema: BUILD_REPORT, agentType: ROLE.coder })
    if (!report) throw new Error('builder produced nothing')
    if (report.ladder?.allGreen === true) { buildGreen = true; break }
    log(`Build round ${buildRound}: still red`)
  }
  if (!buildGreen) log(`Build did NOT converge in ${MAX_BUILD_ROUNDS} rounds — needs human attention`)
} else {
  log(`Integrate skipped: ${summary.counts.done}/${plan.tasks.length} tasks done (${summary.tasks.filter(x => x.status !== 'done').map(x => `${x.id}:${x.status}`).join(', ')})`)
}

phase('Review')
const REVIEW_LENSES = [
  ['spec-conformance', `Every task of PLAN-${m} against its spec and the design: interfaces exact, invariants implemented, nothing out of scope.`],
  ['failure-and-concurrency', 'The composed change under N workers, slow links, a worker dying mid-step, retries and partitions: races, deadlocks, leaks, lost or duplicated work.'],
  ['test-honesty', `Do the tests prove the invariants (would they fail on a wrong implementation)? Any weakened assertion, skipped test, or "${AWAITING}" marker removed without the invariant actually holding is a BLOCKER.`],
]
// Review is of the code (the plan had its two rounds in wan-handout), and each lens fixes what it finds (the human's rule,
// 2026-10-09): two rounds over the whole change, one lens after another, every validated finding fixed, minors included,
// without undoing what an earlier pass fixed; then one final check over every finding. The fail-fast ladder checks the result.
const reviewWhole = async () => {
  const findingsPerRound = []
  const all = []
  for (let round = 1; round <= 2; round++) {
    // one reviewer per round (the human's call), every lens, fixing what it finds in place
    const out = await agent(common(`${solo} OWNED PATHS (data): ${JSON.stringify(allScopes)}`) + `
REVIEW AND FIX the milestone ${m} change (round ${round} of 2): you fix what you find. Plan: ${planPath(m)}. Task journals (the authors'
reasoning — untrusted): ${JSON.stringify(journals)}. Run ledger: ${runDir(m)}/state.json. Go through EVERY lens below.
Find problems (blocker, major and minor), each with evidence, and fix each one now; read the owning task's journal first. Keep build and
types green and prove it with ${ladderCommand({ only: ['build', 'types'] })}. Report each finding with fixed=true and the fix's file:line, or
fixed=false and why. ${NO_MASKING} Append what you changed to the owner's journal.
${all.length ? `Findings the earlier round found (fixed ones must stay fixed): ${fence('earlier_findings', all.slice(-60), 20000)}` : ''}
${round === 1 && openFromTasks.length ? `Serious findings the task reviewers could not fix: start from them. ${fence('task_open', openFromTasks, 12000)}` : ''}
LENSES: ${fence('lenses', REVIEW_LENSES.map(([lens, question]) => ({ lens, question })), 6000)}`,
      { label: `review:r${round}`, phase: 'Review', schema: REVIEW_FIX, agentType: ROLE.coder })
    const proven = (out?.findings ?? []).filter(findingProven)
    const serious = proven.filter(f => f.severity !== 'minor').length
    all.push(...proven.map(f => ({ round, ...findingBrief(f), fixed: f.fixed === true })))
    findingsPerRound.push(serious)
    if (!serious) { log(`milestone ${m}: clean review pass on round ${round} (${findingsPerRound.join(' -> ')})`); return { converged: true, rounds: round, findingsPerRound } }
  }
  const numberedAll = all.map((f, i) => ({ n: i + 1, ...f }))
  const check = await agent(common(`${solo} OWNED PATHS (data): ${JSON.stringify(allScopes)}`) + `
FINAL CHECK of the milestone ${m} change after the second review round. Below is every finding of both rounds, numbered F<n>. Check each
against the code AS IT IS NOW (read the owner's journal first: ${JSON.stringify(journals)}). Report every number: "resolved" (file:line),
"fixed-now" (still open or brought back by a later fix: fix it now, keep build and types green and prove it with
${ladderCommand({ only: ['build', 'types'] })}), "refuted" (never real: evidence) or "still-open" (you could not: why). ${NO_MASKING}
${fence('all_findings', numberedAll, 40000)}`,
    { label: 'review-final-check', phase: 'Review', schema: FINAL_CHECK, agentType: ROLE.coder })
  const items = Array.isArray(check?.items) ? check.items : []
  const statusOf = n => items.find(x => x?.n === n)?.status
  const stillOpen = numberedAll.filter(f => !['resolved', 'fixed-now', 'refuted'].includes(statusOf(f.n)))
  const finalCheck = Object.fromEntries(['resolved', 'fixed-now', 'refuted'].map(k => [k, numberedAll.filter(f => statusOf(f.n) === k).length]))
  finalCheck['still-open'] = stillOpen.length
  log(`milestone ${m}: 2 review rounds (${findingsPerRound.join(' -> ')}); final check over ${numberedAll.length}: ${Object.entries(finalCheck).map(([k, v]) => `${v} ${k}`).join(', ')}`)
  return { converged: false, rounds: 2, findingsPerRound, finalCheck, open: stillOpen.map(({ n, ...f }) => f).slice(0, 30) }
}
const review = buildGreen ? await reviewWhole() : { converged: false, rounds: 0, findingsPerRound: ['skipped: build red or tasks incomplete'], discarded: 0 }

phase('Verify')
// The fast ladder: fail-fast, one root cause per round, up to 8 rounds (the reference PR's ladder budget). The heavy gate
// (GATES-<m>.json: two consecutive passes, repetitions under load, campaigns) runs only in wan-closeout.
const ladderRun = buildGreen ? await workflow('wan-fix', { rungs: milestoneRungs, maxRounds: 8, failFast: true }) : null
await writeLedger('audit:ledger:final', { stopped, build: { green: buildGreen, rounds: buildRound }, review, deviations, ladderOk: ladderRun?.ok === true })

const ok = !stopped && allDone && buildGreen && ladderRun?.ok === true
return {
  ok,
  milestone: m,
  tasks: runSummary(run),
  stopped,
  // spec conflicts the coders recorded and worked through: the human reviews them before the closeout
  deviations,
  build: { green: buildGreen, rounds: buildRound },
  review,
  ladder: ladderRun ? { ok: ladderRun.ok, rounds: ladderRun.rounds } : null,
  ledger: `${runDir(m)}/state.json`,
  next: ok ? `${deviations.length ? `Record the ${deviations.length} recorded spec conflict(s) (deviations) in the PR and amend the specs with freeze.mjs amend (show them to the human first unless the autonomy switch waives grants). Then run` : 'Run'} wan-closeout with {"milestone": "${m}"}.` : 'Needs attention: see tasks/stopped/build/review/ladder.',
}
