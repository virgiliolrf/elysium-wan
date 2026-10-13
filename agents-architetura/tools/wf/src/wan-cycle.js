export const meta = {
  name: 'wan-cycle',
  description: 'One milestone end to end, at the reference PR\'s verification depth: REPOSITORY ANALYSIS refresh (wan-survey) -> PLAN + frozen GATES (wan-handout) -> TASK DAG / PARALLEL IMPLEMENTATION / PER-TASK REVIEW / INTEGRATION / VERIFICATION (wan-implement) -> CORPUS with mechanically verified WOULD-FAIL-IF claims (wan-corpus) -> SECURITY AUDIT by mechanism class (wan-audit) -> CAMPAIGNS to zero for every sanitizer/amplifier/load campaign the gates pin (wan-campaign) -> FUZZ (wan-fuzz) -> SCANNERS (wan-scanners) -> PERFORMANCE BENCHMARK -> RED TEAM (wan-redteam) -> TRIAGE / FIX / VERIFY AGAIN (wan-fix over the whole ladder) -> CLOSEOUT (pinned gates, VERIFIED or NOT VERIFIED). Stops at the first stage that is not ok.',
  whenToUse: 'After wan-prep, per milestone (wan-task runs it for every milestone). args: {"milestone": "M0", "depth": "full"}; depth "quick" shortens campaigns and fuzzing for a rehearsal — closeout always runs the frozen gates. On a factory/* branch (wan-task) the PR goes back to draft at the start, every stage ends in a published commit, and the PR is marked ready only for the tree closeout VERIFIED; args.publish=false turns publishing off.',
  phases: [
    { title: 'Survey', detail: 'wan-survey: the architecture map as it is now, before planning (repository-aware planning)' },
    { title: 'Plan', detail: 'wan-handout (task DAG + frozen GATES-<m>.json)' },
    { title: 'Implement', detail: 'wan-implement (DAG waves, review, integrate, verify)' },
    { title: 'Corpus', detail: 'wan-corpus (stress/semantics tests that fail under their own regression patch)' },
    { title: 'Audit', detail: 'wan-audit with close (mechanism-class security audit)' },
    { title: 'Campaigns', detail: 'wan-campaign for every campaign in the gates, to zero' },
    { title: 'Fuzz', detail: 'wan-fuzz for every fuzz campaign in the gates' },
    { title: 'Scanners', detail: 'wan-scanners (battery + triage)' },
    { title: 'Benchmark', detail: 'Verifier: bench rung (baseline recorded if missing)' },
    { title: 'Redteam', detail: 'wan-redteam with autofix (bughunts for confirmed findings; uncertain ones go to a human)' },
    { title: 'Verify', detail: 'wan-fix over the whole ladder after all of the above' },
    { title: 'Closeout', detail: 'wan-closeout: the frozen gates, VERIFIED or NOT VERIFIED' },
  ],
}

// @include hygiene
// @include schemas
// @include project
// @include gates
// @include ledger
// @include dag
// @include campaign
// @include publish

const m = milestoneArg(args)
const quick = args?.depth === 'quick'
const results = {}
const stopAt = (stage, r) => {
  log(`Cycle ${m}: stopped at ${stage} (not ok) — see its result`)
  return { ok: false, milestone: m, stoppedAt: stage, results }
}
const ID = /^[a-z0-9][a-z0-9-]{0,40}$/
// Every stage is published as it ends, red or green (the commit message is derived from its result).
const done = (stage, r, phaseTitle) => publishStage(`${m} ${stage}`, r, phaseTitle)

phase('Survey')
await publishCall('draft', 'draft', 'Survey') // a new milestone: bots review only VERIFIED states
results.survey = await workflow('wan-survey', {})
await done('survey', results.survey, 'Survey')
if (results.survey?.ok !== true) return stopAt('survey', results.survey)

phase('Plan')
results.plan = await workflow('wan-handout', { milestone: m })
await done('plan', results.plan, 'Plan')
if (results.plan?.ok !== true) return stopAt('plan', results.plan)
const gates = await agent(common('READ-ONLY.') + `
Read ${DESIGN_DIR}/GATES-${m}.json and return its JSON verbatim. If it is missing or invalid, return
{"milestone": "", "consecutive": 0, "rungs": [], "pinned": [], "campaigns": [], "fuzz": []}.`,
  { label: 'extract:gates', phase: 'Plan', schema: GATES, agentType: ROLE.readonly })
const campaignIds = [...new Set((gates?.campaigns ?? []).map(c => c.id).filter(x => ID.test(x ?? '')))]
const fuzzIds = [...new Set((gates?.fuzz ?? []).map(c => c.id).filter(x => ID.test(x ?? '')))]

// Factory v2: the human grants the milestone plan once (HANDOUT, PLAN, GATES) before anything is implemented.
const PLAN_DOCS = [`${DESIGN_DIR}/HANDOUT-${m}.md`, `${DESIGN_DIR}/PLAN-${m}.json`, `${DESIGN_DIR}/GATES-${m}.json`]
const planGrant = await checkApprovals(PLAN_DOCS, 'gate:approval:plan', 'Plan')
if (!approvedAll(planGrant, PLAN_DOCS)) {
  log(`Cycle ${m}: the plan awaits its human grant — run: node agents-architetura/tools/factory/approve.mjs grant ${PLAN_DOCS.join(' ')}; then run wan-cycle again`)
  return { ...stopAt('grant', results.plan), next: `A human grants ${PLAN_DOCS.join(', ')}; then run wan-cycle with {"milestone": "${m}"} again.` }
}

phase('Implement')
results.implement = await workflow('wan-implement', { milestone: m })
await done('implement', results.implement, 'Implement')
if (results.implement?.ok !== true) return stopAt('implement', results.implement)

phase('Corpus')
results.corpus = await workflow('wan-corpus', {})
await done('corpus', results.corpus, 'Corpus')
if (results.corpus?.ok !== true) return stopAt('corpus', results.corpus)

phase('Audit')
results.audit = await workflow('wan-audit', { close: true })
await done('audit', results.audit, 'Audit')
if (results.audit?.ok !== true) return stopAt('audit', results.audit)

phase('Campaigns')
results.campaigns = {}
for (const id of campaignIds) {
  const r = await workflow('wan-campaign', { campaign: id, ...(quick ? { maxWaves: 2 } : {}) })
  results.campaigns[id] = { ok: r?.ok === true, applicable: r?.applicable !== false, unsuppressed: r?.unsuppressed ?? null }
  await done(`campaign ${id}`, results.campaigns[id], 'Campaigns')
  if (r?.ok !== true) return stopAt(`campaign ${id}`, r)
}

phase('Fuzz')
results.fuzz = {}
for (const id of fuzzIds) {
  const r = await workflow('wan-fuzz', { campaign: id, ...(quick ? { minutes: 10 } : {}) })
  results.fuzz[id] = { ok: r?.ok === true, cpuHours: r?.cpuHours ?? null, open: (r?.open ?? []).length }
  await done(`fuzz ${id}`, results.fuzz[id], 'Fuzz')
  if (r?.ok !== true) return stopAt(`fuzz ${id}`, r)
}

phase('Scanners')
results.scanners = await workflow('wan-scanners', {})
await done('scanners', results.scanners, 'Scanners')
if (results.scanners?.ok !== true) return stopAt('scanners', results.scanners)

phase('Benchmark')
const bench = await agent(common('You run ALONE — you MAY run commands. You are the independent verifier; change nothing in the code.') + `
If bench/suite.json exists and bench/baseline.json does not, first run: node agents-architetura/tools/gates/bench-gate.mjs --record
Then run: ${ladderCommand({ only: ['bench'] })} and paste its JSON into "ladder" verbatim. allGreen=true only if the pasted ladder says so.
A bench rung that is n/a (no suite, or a machine too noisy to measure) is reported as n/a, not as a pass.
items: one per regressed benchmark (scope: bench/).`,
  { label: 'benchmark', phase: 'Benchmark', schema: TRIAGE, agentType: ROLE.verifier })
results.benchmark = { ladder: bench?.ladder ?? null }
await done('benchmark', { ok: ladderGreen(bench) }, 'Benchmark')
if (!ladderGreen(bench)) return stopAt('benchmark', results.benchmark)

phase('Redteam')
results.redteam = await workflow('wan-redteam', { autofix: true })
await done('redteam', results.redteam, 'Redteam')
if (results.redteam?.ok !== true) return stopAt('redteam', results.redteam)

phase('Verify')
results.verify = await workflow('wan-fix', {})
await done('verify', results.verify, 'Verify')
if (results.verify?.ok !== true) return stopAt('verify', results.verify)


phase('Closeout')
results.closeout = await workflow('wan-closeout', { milestone: m, review: results.implement.review })
await done('closeout', results.closeout, 'Closeout')
if (results.closeout?.ok !== true) return stopAt('closeout', results.closeout)

// The PR leaves draft only for the tree closeout verified (its report and LESSONS.md may have been written since).
const verifiedTree = /^[0-9a-f]{40,64}$/.test(results.closeout?.tree ?? '') ? results.closeout.tree : null
const readied = verifiedTree ? await publishCall(`ready --tree ${verifiedTree}`, 'ready', 'Closeout') : { ok: false }
if (!verifiedTree) log(`Cycle ${m}: closeout reported no tree id — the PR stays a draft`)
log(`Cycle ${m}: VERIFIED${readied.ok ? ` — PR ready for review${readied.pr?.url ? ` (${clean(readied.pr.url, 200)})` : ''}` : ''}`)
return { ok: true, milestone: m, verdict: 'VERIFIED', results, prReady: readied.ok === true }
