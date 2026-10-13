export const meta = {
  name: 'wan-handout',
  description: 'Turn one milestone of the approved design into an executable task DAG: HANDOUT-<m>.md + PLAN-<m>.json, machine-validated (DAG, ownership inside workstreams, integrator-only hot files, acceptance + known rungs), adversarially reviewed, and re-validated after every revision; then freeze the milestone\'s verification bar in GATES-<m>.json (pinned rungs run twice in a row, pinned commands, campaigns to zero, fuzz bars, audit, mutation) — the exact gate closeout will run.',
  whenToUse: 'After wan-prep, once per milestone (or per large slice of one). args: {"milestone": "M0", "focus": "<optional: the slice of the milestone this run plans>"}. The human grants HANDOUT, PLAN and GATES once; wan-implement then runs the whole DAG.',
  phases: [
    { title: 'Gate', detail: 'Design approved; workstreams loaded' },
    { title: 'Draft', detail: 'Solo planner writes HANDOUT-<m>.md + PLAN-<m>.json: every open phase of the milestone in ONE DAG for one run (split only past 24 tasks)' },
    { title: 'Validate', detail: 'Script validates the extracted plan; repair loop <= 3' },
    { title: 'Review', detail: 'dag-soundness / spec-coverage / binding-audit / scope-minimality: 2 rounds, every validated finding fixed (minors too) without undoing earlier ones; then one final check that verifies and fixes every finding; re-validated after each change' },
    { title: 'Gates', detail: 'Researcher freezes GATES-<m>.json; script shape check + pinned.mjs --validate; repair <= 3' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project
// @include plan
// @include ledger
// @include campaign

const m = milestoneArg(args)
// The human's scope for this run: the slice of the milestone, items moved in from other milestones, decisions and the
// criteria earlier sessions learned. It reaches the planner, the reviewers and every revision whole (M2's single plan,
// 2026-10-09: a 2117-character focus was cut to 600, and the reviewers never saw it).
const FOCUS_MAX = 8000
const FOCUS = clean(args?.focus ?? '', FOCUS_MAX)
if (String(args?.focus ?? '').length > FOCUS_MAX) throw new Error(`args.focus is ${String(args.focus).length} characters; at most ${FOCUS_MAX}: move the detail into a document the focus names`)
const SCOPE = FOCUS ? `
THE HUMAN'S SCOPE FOR THIS RUN (it overrides the milestone section of the design where they differ: items it moves in
from other milestones belong in the plan, items it defers do not):
${fence('focus', FOCUS, FOCUS_MAX + 200)}` : ''
// One plan, one run (the human's rule, 2026-10-09): every phase still open in the milestone goes into ONE task DAG that a
// single wan-implement run executes, so the plan, the review and the integration are paid once. Only a milestone too big
// for one run (more tasks than this) is split: the plan is a self-contained first slice and the rest are listed for later.
const MAX_PLAN_TASKS = 24
// At most 2 waves (the human's rule, 2026-10-09): wave 1 builds the pieces in parallel, wave 2 integrates them. A deeper DAG
// is the plan's shape, not the work's: M2 ran in 5 waves and M2b's plan had 13, chained by files with many owners.
const MAX_PLAN_WAVES = 2
const OWN = `OWNED PATHS: ${handoutPath(m)}, ${planPath(m)} (only files you may write).`
const GATES_PATH = `${DESIGN_DIR}/GATES-${m}.json`

phase('Gate')
await requireApprovedDesign('Gate')
const workstreams = await loadWorkstreams('Gate')
const ownershipTable = workstreams.map(w => ({ workstream: w.key, ownedPaths: w.ownedPaths }))
const PLAN_RULES = `
PLAN RULES (the workflow machine-checks every one):
- ${planPath(m)} is JSON: {"milestone": "${m}", "binding": [...], "tripwires": [...], "tasks": [{"id", "title", "objective", "workstream", "deps", "scope", "acceptance", "verify", "rungs", "invariants", "binding", "risk"}]}.
- binding (the reference PR's binding audit, written BEFORE any task): read the code and list EVERY instance of the hazard
  this milestone touches — every piece of state a change of ownership, schedule or restart reads or writes, every reduction
  and RNG draw whose order decides the bytes, every resource a new path shares — as {"id": "BA-<n>", "what", "where"
  (file:line or symbol), "ruling" (what happens to it: rebuilt at the boundary, carried unchanged, per-peer, refused...)}.
  Every item's ruling is delivered by some task's "binding" list. An instance found later by a crash is a planning failure.
- tripwires: when a change is silently wrong with only some of its legs in (the training result would differ without an
  error), declare {"id", "flag" (the switch or entry point that raises until armed), "legs" (task ids, >= 2), "arm" (the
  task that lifts the refusal; it depends on every leg), "test" (a command proving the refusal holds)}. The first leg adds
  the refusal; every leg lists the test in its verify; only the arm task removes it.
- risk: low | medium | high. high = touches a trust boundary (${DESIGN_DIR}/THREAT_MODEL.md), concurrency, the training math, or
  checkpoint/recovery — high-risk tasks get a security review of their diff.
- objective: one sentence. verify: the exact commands that prove the acceptance criteria (they must exit non-zero when the
  criterion is not met); the coder runs them on its own files, and the build, review and ladder at the end of the run cover them. They are FAST checks (seconds to a few minutes: the task's own
  tests, a targeted integration test): repetitions, amplified or loaded reruns and soaks belong in ${GATES_PATH}, which
  only closeout runs.
- ONE RUN: plan every phase of the milestone that is still open${FOCUS ? ' within the focus' : ''} as ONE DAG, executed by a single
  wan-implement run. Only when that would take more than ${MAX_PLAN_TASKS} tasks, plan the first slice that stands alone (at most
  ${MAX_PLAN_TASKS} tasks, its own acceptance and gates) and list the remaining slices, in order, under "Later slices" in the handout.
- id matches ^[a-z0-9][a-z0-9-]{0,47}$; deps name existing ids; no cycles.
- scope: minimal repo-relative paths (directories end with "/") INSIDE the owned paths of the task's workstream:
  ${JSON.stringify(ownershipTable)}
- Shared hot files (${HOT_FILES.join(', ')}) only in tasks whose workstream is "${INTEGRATOR}".
- AT MOST ${MAX_PLAN_WAVES} WAVES (machine-checked). Wave 1: the tasks that build the pieces, in parallel. Wave 2: the integration
  tasks that wire them in, each depending only on wave-1 tasks. One task per file: the work a chain of tasks would do on one
  file is ONE task (a big task is fine; a chain is not). Every task writes its tests in its OWN new test file
  (tests/unit/test_<task-id>.py, tests/integration/test_<task-id>.py), never appended to a shared one: a shared test file
  chains every task that touches it. The handout fixes the interfaces between wave-1 tasks (signatures, types, wire
  formats), so they code against them in parallel. A file may have a second owner only in a wave-2 task that depends on the
  first.
- Tasks that may run concurrently must NOT share a path (the machine check rejects it). No duplicate tasks.
- Read ${DESIGN_DIR}/ARCHITECTURE_MAP.md first: plan against the code as it is, reuse what exists.
- acceptance: testable statements; rungs: ladder rung ids from ${RUNG_IDS.join(', ')} that must pass.
- invariants: the INV-* / <KEY>-I* ids the task makes true; its scope includes the tests marked "${AWAITING}" for
  them, and the task removes those markers.`

phase('Draft')
const draft = await agent(common(OWN) + PLAN_RULES + `
Solo planner for milestone ${m}. Read ${DESIGN} (its milestone ${m} section is the scope), every ${DESIGN_DIR}/SPEC-*.md
(their ordered task lists), and the tree.${FOCUS ? `${SCOPE}
Work already merged for ${m} is the starting point, not work to redo.` : ''} Write ${handoutPath(m)} (the narrative: goal, task table, wave order, risks)
and ${planPath(m)} (the machine plan). Every invariant the milestone promises must be delivered by exactly one task.`,
  { label: 'draft', phase: 'Draft', schema: RESULT, agentType: ROLE.researcher })
if (!draft) throw new Error('planner produced nothing')

const validateLoop = async tag => {
  let check = { ok: false, errors: ['not checked'], waves: [] }
  let plan = null
  for (let attempt = 1; attempt <= 3; attempt++) {
    plan = await extractPlan(m, `extract:${tag}:a${attempt}`, 'Validate')
    check = checkMilestonePlan(plan, { workstreams, rungIds: RUNG_IDS, hotFiles: HOT_FILES, integrator: INTEGRATOR, maxWaves: MAX_PLAN_WAVES })
    const n = (plan?.tasks ?? []).length
    if (n > MAX_PLAN_TASKS) check = { ...check, ok: false, errors: [...check.errors, `${n} tasks: more than one run takes (${MAX_PLAN_TASKS}) — plan the first self-contained slice and list the rest under "Later slices" in the handout`] }
    if (check.ok) break
    log(`Validate (${tag}) attempt ${attempt}: ${check.errors.length} problem(s) -> repairing`)
    await agent(common(OWN) + PLAN_RULES + `
The machine check of ${planPath(m)} failed.${SCOPE}
Fix every error below without dropping work — re-scope, split or re-order
tasks instead — and keep ${handoutPath(m)} consistent with the plan.
${fence('plan_errors', check.errors, 12000)}`,
      { label: `repair:${tag}:a${attempt}`, phase: 'Validate', schema: RESULT, agentType: ROLE.researcher })
  }
  return { check, plan }
}

phase('Validate')
let { check, plan } = await validateLoop('draft')
if (!check.ok) throw new Error(`plan still invalid after 3 repairs: ${check.errors.slice(0, 10).join('; ')}`)

phase('Review')
const LENSES = [
  ['dag-soundness', `Is the dependency order real? A task that consumes an interface must depend on the task that provides it;
a task must not depend on something it does not use (it serializes the milestone for nothing). Any task whose
acceptance cannot pass with only its deps done is a BLOCKER.`],
  ['spec-coverage', `Walk milestone ${m} in the design and every spec's task list: is every promised behavior and invariant delivered
by exactly one task, with acceptance criteria and rungs that would actually catch it missing? Missing work is a BLOCKER.`],
  ['binding-audit', `Is the binding audit complete? Search the code yourself (grep the state, the reductions, the RNG draws the
milestone touches): an instance without a BA-<n> ruling, or a ruling no task delivers, is a BLOCKER. Is every change that would
be silently wrong when partly landed behind a tripwire whose arm task depends on every leg?`],
  ['scope-minimality', `Is each scope the minimum the task needs? Over-broad scopes serialize the DAG and hide out-of-scope edits;
under-broad scopes force violations. Are the tests carrying "${AWAITING}" markers in the scope of the task that delivers them?
Would each task's verify commands actually fail if its acceptance criteria were not met?`],
]
// The human's rule (2026-10-09, factory v2): the plan is reviewed twice at most and every validated finding is fixed,
// minors included; the code is what gets reviewed after that, and the heavy gate is closeout's.
const review = await reviewUntilClean({
  label: `PLAN-${m}`,
  lenses: LENSES,
  maxRounds: 2,
  fixAll: true,
    review: (name, lens, round) => agent(common('READ-ONLY adversarial reviewer.') + `
Review ${handoutPath(m)} and ${planPath(m)} (round ${round}, lens ${name}) against ${DESIGN} and the specs. Assume the plan is
wrong until the documents prove otherwise.${SCOPE} LENS: ${lens}
Severity: blocker = executing this plan produces a broken milestone; major = forces mid-flight re-planning.`,
    { label: `review:${name}:r${round}`, phase: 'Review', schema: FINDINGS, agentType: ROLE.reviewer }),
  fix: async (serious, round, earlier) => {
    await agent(common(OWN) + PLAN_RULES + `
Review round ${round} (of 2) filed these findings, minors included.${SCOPE} Verify each against the design and specs; fix EVERY real
one in both files; refute false ones in a "Review notes" section of the handout (one line each).
${fence('reviewer_findings', serious, 30000)}${earlier.length ? `
Findings of earlier rounds, already fixed: your changes must not bring any of them back (a fix that satisfies one finding by
undoing another is not a fix).
${fence('earlier_findings', earlier, 20000)}` : ''}`,
      { label: `revise:r${round}`, phase: 'Review', schema: RESULT, agentType: ROLE.researcher })
    const again = await validateLoop(`r${round}`)
    if (!again.check.ok) throw new Error(`plan invalid after review round ${round}: ${again.check.errors.slice(0, 10).join('; ')}`)
    check = again.check
    plan = again.plan
  },
  // After the last fix: one agent checks every finding of both rounds against the plan as it is now, and fixes what is still
  // open or came back; the script validates the result.
  finalCheck: async all => {
    const r = await agent(common(OWN) + PLAN_RULES + `
FINAL CHECK of ${handoutPath(m)} and ${planPath(m)} after the last review round.${SCOPE}
Below is every finding of every review round, numbered F<n>; each was sent to a fix. Check each against the two files AS
THEY ARE NOW. Report every number: "resolved" (cite where), "fixed-now" (it was still open, or a later fix brought it back:
fix it now in both files), "refuted" (it was never real: evidence), or "still-open" (you could not fix it: why). Be
especially hard on fixes that satisfied one finding by undoing another, and on any acceptance criterion, verdict rule or
verify command that would pass whatever the truth is.
${fence('all_findings', all, 40000)}`,
      { label: 'final-check', phase: 'Review', schema: FINAL_CHECK, agentType: ROLE.researcher })
    const again = await validateLoop('final')
    if (!again.check.ok) throw new Error(`plan invalid after the final check: ${again.check.errors.slice(0, 10).join('; ')}`)
    check = again.check
    plan = again.plan
    return r
  },
})

phase('Gates')
const planRungs = [...new Set((plan?.tasks ?? []).flatMap(t => t.rungs ?? []))].filter(r => RUNG_IDS.includes(r))
const GATE_RULES = `
${GATES_PATH} is the milestone's FROZEN verification bar — closeout runs exactly it (node agents-architetura/tools/gates/pinned.mjs ${m}) and nothing else
counts. JSON: {"milestone": "${m}", "consecutive": 2, "rungs": [...], "pinned": [{"id", "cmd", "repeat"}], "campaigns": [{"id"}],
"fuzz": [{"id", "minCpuHours"}], "audit": {"bar": "no-open-tier-a"}, "mutation": {"changedSince": "auto"}}.
- consecutive >= 2: the pinned rungs must be green twice in a row (stability is measured, not assumed).
- rungs: ${BASE_GATE_RUNGS.join(', ')}, every rung a plan task names (${planRungs.join(', ') || 'none'}), and every verification rung the
  milestone's code makes applicable (oracle, chaos, matrix, identity, fuzz, redteam, mutation, bench) — ids from ${RUNG_IDS.join(', ')}.
- pinned: exact commands with bars, e.g. the milestone's entry smoke with "repeat": 20 (a flaky pass is not a pass), and for every
  concurrency or failure test "node agents-architetura/tools/verify/amplify.mjs --runs 200 --seed-base 1000 --periods 16,64,256 -- <cmd>" and
  "node agents-architetura/tools/verify/load.mjs --workers 6 --runs-per-worker 20 -- <cmd>".
- campaigns / fuzz: ids from agents-architetura/tools/campaign/campaigns.json that apply to code this milestone ships (sanitizer, amplifier and load
  campaigns must reach zero; fuzz campaigns need minCpuHours, default 4, with a quiet final hour). Omit none that apply.
- audit: include it when the milestone touches a trust boundary; mutation: include it whenever the milestone changes source.`
const readGates = tag => agent(common('READ-ONLY.') + `
Read ${GATES_PATH} and return its JSON verbatim (do not fix it). If it is missing or not valid JSON, return
{"milestone": "", "consecutive": 0, "rungs": [], "pinned": [], "campaigns": [], "fuzz": []}.`,
  { label: `extract:gates:${tag}`, phase: 'Gates', schema: GATES, agentType: ROLE.readonly })
await agent(common(`OWNED PATHS: ${GATES_PATH} (only).`) + GATE_RULES + `
Write ${GATES_PATH} for milestone ${m} from ${planPath(m)}, ${handoutPath(m)}, ${DESIGN} and agents-architetura/tools/campaign/campaigns.json.`,
  { label: 'gates', phase: 'Gates', schema: RESULT, agentType: ROLE.researcher })
let gateErrors = ['not checked']
for (let attempt = 1; attempt <= 3; attempt++) {
  const gates = await readGates(`a${attempt}`)
  gateErrors = checkGates(gates, { milestone: m, planRungs })
  if (!gateErrors.length) {
    const v = await runTool(`node agents-architetura/tools/gates/pinned.mjs ${m} --validate`, { label: `validate:gates:a${attempt}`, phaseTitle: 'Gates' })
    gateErrors = v?.json?.ok === true ? [] : (v?.json?.errors ?? ['pinned.mjs --validate produced nothing']).map(e => clean(e, 300))
  }
  if (!gateErrors.length) break
  log(`Gates attempt ${attempt}: ${gateErrors.length} problem(s) -> repairing`)
  await agent(common(`OWNED PATHS: ${GATES_PATH} (only).`) + GATE_RULES + `
The check of ${GATES_PATH} failed. Fix every error below without lowering the bar.
${fence('gate_errors', gateErrors, 8000)}`,
    { label: `repair:gates:a${attempt}`, phase: 'Gates', schema: RESULT, agentType: ROLE.researcher })
}
if (gateErrors.length) log(`GATES-${m}: still invalid after 3 repairs — ${gateErrors.slice(0, 5).join('; ')}`)

log(`PLAN-${m}: ${check.waves.flat().length} tasks in ${check.waves.length} waves; review ${review.converged ? 'converged' : `ended after ${review.rounds} rounds, its last findings fixed`}; gates ${gateErrors.length ? 'INVALID' : 'frozen'}`)
const DOCS = [handoutPath(m), planPath(m), GATES_PATH]
return {
  ok: check.ok && !gateErrors.length,
  gates: GATES_PATH,
  gateErrors,
  milestone: m,
  handout: handoutPath(m),
  plan: planPath(m),
  waves: check.waves,
  review,
  next: `A human grants ${DOCS.join(', ')} (node agents-architetura/tools/factory/approve.mjs grant ${DOCS.join(' ')}); then run wan-implement with {"milestone": "${m}"}.`,
}
