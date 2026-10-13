export const meta = {
  name: 'wan-task',
  description: 'The human gives the task and makes the architecture decisions; agents do the rest. REPOSITORY ANALYSIS (wan-survey: what exists, specialists only where the code needs them) -> REQUEST SPECIFICATION (researcher, adversarially reviewed, frozen) -> DESIGN with first-principles cost model, THREAT MODEL and ADR-0001 (wan-design) -> STOP: a human grants the design and its ADR (agents-architetura/tools/factory/approve.mjs; silence is not approval). Run again after the grant: PREP (frozen specs, verification seams) -> every milestone through wan-cycle (each must close VERIFIED) -> comparative scalability study -> implementation report. Stops at the first stage that is not ok, with its evidence. Publishing (the reference PR\'s git model): the run works on branch factory/<task> with one draft PR; every stage ends in a commit whose message is derived from its result, a timer checkpoints long stages, the PR leaves draft when a milestone closes VERIFIED; merging is human-only.',
  whenToUse: 'Start of any piece of work, and again after approving the design. args: {"task": "<what to build>", "reference": "github:owner/repo", "redesign": false, "skipPrep": false, "maxMilestones": 10, "id": "branch-slug (default: from the task)", "publish": true}. Start from a clean checkout of main (publishing refuses to carry uncommitted changes into the task branch).',
  phases: [
    { title: 'State', detail: 'Is the design of record approved (resume) or not yet written (fresh)?' },
    { title: 'Survey', detail: 'wan-survey: CURRENT_ARCHITECTURE.md, ARCHITECTURE_MAP.md, specialists, optional reference map' },
    { title: 'Spec', detail: 'Researcher: request specification (BRIEF.md) -> 2 reviewers loop -> frozen' },
    { title: 'Design', detail: 'wan-design: surveys, first-principles draft, threat model, review, ownership, ADR-0001' },
    { title: 'Approve', detail: 'STOP for the human grant of DESIGN.md + ADR-0001' },
    { title: 'Prep', detail: 'wan-prep: frozen specs, system review, verification seams, skeleton, invariant tests' },
    { title: 'Milestones', detail: 'wan-cycle for every milestone in order; each must close VERIFIED' },
    { title: 'Scale', detail: 'wan-scalebench: comparative study (honest n/a without the hardware)' },
    { title: 'Report', detail: 'Researcher: agents-architetura/docs/reports/TASK.md — the implementation report' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project
// @include gates
// @include ledger
// @include campaign
// @include publish

if (typeof args?.task !== 'string' || args.task.trim().length < 10) throw new Error('args.task (a description of what to build) is required')
const maxMilestones = Number.isInteger(args.maxMilestones) && args.maxMilestones > 0 && args.maxMilestones <= 20 ? args.maxMilestones : 10
const ADR0 = `${DESIGN_DIR}/adr/ADR-0001-initial-architecture.md`
const MILESTONES = { type: 'object', required: ['milestones'], properties: { milestones: { type: 'array', items: { type: 'string' } } } }
const stages = {}
// Every stop is published too (red is recorded like green), and the checkpoint timer is stopped.
const stop = async (stage, why, extra = {}) => {
  log(`wan-task: stopped at ${stage} — ${why}`)
  await publishStage(`stopped at ${stage}`, { ok: false, why, ...(extra.awaitingHuman ? { awaitingHuman: true } : {}) }, 'Report')
  await publishCall('stop-watch', 'stop-watch', 'Report')
  return { ok: false, stoppedAt: stage, why, stages, ...extra }
}
const awaitHuman = why => stop('approve', why, {
  awaitingHuman: true,
  decision: [DESIGN, ADR0],
  command: `node agents-architetura/tools/factory/approve.mjs grant ${DESIGN} ${ADR0}`,
  next: 'Read the ADR (its ARCHITECTURE DECISION REQUIRED block first), the design and the threat model; approve with /wan-approve in Claude Code (or in your own terminal), then run wan-task again with the same task.',
})

phase('State')
const state = await checkApprovals([DESIGN], 'state:approval', 'State')
const designState = (state?.docs ?? [])[0] ?? {}
const resume = approvedAll(state, [DESIGN])
if (!resume && designState.reason !== 'document does not exist' && args.redesign !== true) {
  return awaitHuman(`the design of record exists but is not approved (${clean(designState.reason ?? 'unknown', 200)})`)
}
// One branch and one draft PR per task (resumed after the human grant); a refusal (dirty base, wrong branch, identity) stops here.
const TASK_ID = typeof args.id === 'string' && /^[a-z0-9][a-z0-9._-]{0,60}$/.test(args.id) ? args.id : null
const started = await publishCall(TASK_ID ? `start --task ${TASK_ID}` : `start --text-b64 ${base64(args.task)}`, 'start', 'State')
if (started.ok) {
  log(`wan-task: publishing on ${clean(started.branch, 80)}${started.resumed ? ' (resumed)' : ''}`)
  await publishCall('watch --detach', 'watch', 'State')
}

if (!resume) {
  phase('Survey')
  stages.survey = await workflow('wan-survey', typeof args.reference === 'string' ? { reference: args.reference } : {})
  await publishStage('survey', stages.survey, 'Survey')
  if (stages.survey?.ok !== true) return stop('survey', 'the architecture map did not pass its checks or its review — fix the map before specifying')

  phase('Spec')
  const SPEC_SECTIONS = ['Goal', 'Existing Architecture', 'Problem', 'Non-goals', 'Proposed Change', 'Components Affected', 'Interfaces', 'Data Flow', 'Control Flow', 'Failure Modes', 'Security Model', 'Performance Requirements', 'Compatibility Requirements', 'Testing Requirements', 'Acceptance Criteria']
  const OWN_SPEC = `OWNED PATHS: ${BRIEF} (only).`
  const SPEC_RULES = `
${BRIEF} is the REQUEST SPECIFICATION — never implement a request before it is specified. Sections, in order:
${SPEC_SECTIONS.map(s => `# ${s}`).join(' · ')}.
- Existing Architecture comes from ${DESIGN_DIR}/CURRENT_ARCHITECTURE.md and ${DESIGN_DIR}/ARCHITECTURE_MAP.md (cite components); Components
  Affected names real components or says "new".
- The human's request is the only source of intent: everything inferred beyond it is marked "ASSUMPTION:" with the reason.
- Acceptance Criteria are measurable: a command or a number, with tolerances only for performance. They always include the
  product rule (CLAUDE.md): the training result is byte-identical to the synchronous datacenter reference in every scenario the
  request touches. Performance Requirements state workload, hardware and network conditions; Failure Modes and the Security Model state the trust assumptions (who runs the hardware, who may be adversarial).`
  await agent(common(OWN_SPEC) + SPEC_RULES + `
Specify the human's request.
${fence('human_task', args.task, 8000)}`,
    { label: 'spec', phase: 'Spec', schema: RESULT, agentType: ROLE.researcher })
  stages.spec = await reviewUntilClean({
    label: 'request specification',
    lenses: [
      ['testable', 'Can every acceptance criterion be checked by a command with a pass/fail outcome? Are non-goals explicit, assumptions marked, failure modes and the security model concrete enough to test? Is anything the request asks for missing?'],
      ['grounded', `Does Existing Architecture match ${DESIGN_DIR}/CURRENT_ARCHITECTURE.md and the code? Are the components affected, interfaces and flows real? Does the spec silently decide architecture that is the design's (or a human's) call?`],
    ],
        review: (name, lens, round) => agent(common('READ-ONLY adversarial reviewer.') + `
Review ${BRIEF} against the human's request (round ${round}, lens ${name}). ${lens}
${fence('human_task', args.task, 8000)}`,
      { label: `spec-review:${name}:r${round}`, phase: 'Spec', schema: FINDINGS, agentType: ROLE.reviewer }),
    fix: (serious, round) => agent(common(OWN_SPEC) + SPEC_RULES + `
Revise ${BRIEF} for these findings (round ${round}); refute false ones in an "Open questions" section with evidence.
${fence('reviewer_findings', serious, 20000)}`,
      { label: `spec-fix:r${round}`, phase: 'Spec', schema: RESULT, agentType: ROLE.researcher }),
  })
  if (!stages.spec.converged) return stop('spec', 'the request specification did not converge under review — a human must clarify the request')
  const frozen = await runTool(`node agents-architetura/tools/factory/freeze.mjs freeze ${BRIEF} || node agents-architetura/tools/factory/freeze.mjs amend ${BRIEF} --reason "request specification rewritten by a new wan-task run" --by wan-task`,
    { label: 'spec:freeze', phaseTitle: 'Spec' })
  if (frozen?.exitCode !== 0) return stop('spec', 'the request specification could not be frozen')
  await publishStage('spec', { ok: true, converged: true, rounds: stages.spec.rounds }, 'Spec')

  phase('Design')
  stages.design = await workflow('wan-design', {})
  await publishStage('design', stages.design, 'Design')
  if (stages.design?.ok !== true) return stop('design', 'adversarial review did not converge, ownership is invalid, or finalize failed — a human must decide')

  phase('Approve')
  return awaitHuman('the design of record and ADR-0001 are written; major architecture decisions need an explicit human grant')
}

log(`wan-task: design approved by ${clean(designState.approvedBy ?? 'a human', 60)} — resuming`)
if (args.skipPrep !== true) {
  phase('Prep')
  stages.prep = await workflow('wan-prep', {})
  await publishStage('prep', stages.prep, 'Prep')
  if (stages.prep?.ok !== true) return stop('prep', 'specs did not converge or the substrate is not green')
}

phase('Milestones')
const ms = await agent(common('READ-ONLY.') + `
Read ${DESIGN}. Return the milestone ids of its milestones section in order (e.g. ["M0", "M1"]), verbatim.`,
  { label: 'extract:milestones', phase: 'Milestones', schema: MILESTONES, agentType: ROLE.readonly })
const milestones = (ms?.milestones ?? []).filter(x => /^M\d{1,3}[a-z]?$/.test(x)).slice(0, maxMilestones)
if (!milestones.length) return stop('milestones', 'the design lists no milestones')
stages.cycles = {}
for (const m of milestones) {
  const r = await workflow('wan-cycle', { milestone: m, ...(publishing() ? {} : { publish: false }) })
  stages.cycles[m] = { ok: r?.ok === true, verdict: r?.verdict ?? null, stoppedAt: r?.stoppedAt ?? null }
  if (r?.ok !== true) return stop(`cycle ${m}`, `stopped at ${r?.stoppedAt ?? 'unknown'}`)
}

phase('Scale')
stages.scalebench = await workflow('wan-scalebench', {})
await publishStage('scalebench', stages.scalebench, 'Scale')
if (stages.scalebench?.ok !== true) return stop('scalebench', 'the scalability study is invalid (checksum mismatch) or its review did not converge')

phase('Report')
await agent(common('OWNED PATHS: agents-architetura/docs/reports/TASK.md (only).') + `
Write agents-architetura/docs/reports/TASK.md, the implementation report, with these sections: # Feature · # Architectural Changes (every ADR
and its human grant) · # Files Changed · # Tasks Completed · # Tests · # Distributed Training Validation (keep "SYSTEM WORKED"
separate from "TRAINING RESULT IS CORRECT", with the oracle evidence) · # Security (threat model coverage, audit verdicts,
red-team findings) · # Performance (before/after, workload, hardware, network, method) · # Failure Recovery · # Benchmarks ·
# Vast.ai Results (expected vs observed, classification — or "none run") · # Known Limitations · # Remaining Risks ·
# Acceptance Criteria (each criterion of ${BRIEF} with its evidence, or NOT MET) · # Recommended Next Work.
Sources: the milestone reports (agents-architetura/docs/reports/M*.md), gate results, docs/security/, bench/scalebench/RESULTS.md,
docs/experiments/. Facts only; cite them. An acceptance criterion without evidence is NOT MET.
${fence('stages', stages, 12000)}`,
  { label: 'report', phase: 'Report', schema: RESULT, agentType: ROLE.researcher })
await publishStage('report', { ok: true, milestones }, 'Report')
await publishCall('stop-watch', 'stop-watch', 'Report')

return { ok: true, milestones, stages }
