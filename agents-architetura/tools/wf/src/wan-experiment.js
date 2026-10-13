export const meta = {
  name: 'wan-experiment',
  description: 'Local before paid cloud: test an architecture hypothesis (an ADR) the cheap way first; the cloud step needs the hardware switch on (gpu true). The researcher writes the experiment plan with predictions derived from the ADR\'s model -> a prototype in experiments/<id>/ (scope-audited) -> local correctness against the reference, local performance and failure injection run by the verifier -> the SCRIPT classifies expected vs observed (VALIDATED / PARTIALLY VALIDATED / REFUTED / INCONCLUSIVE) and records the promotion checklist -> only with cloud=true, every checklist item true, the ADR and the exact plan+budget granted by a human: the Vast.ai engineer runs it through agents-architetura/tools/vast/vast.mjs (pre-flight, cost gate, monitoring, emergency stop, cleanup) -> expected vs observed again -> report. A refuted hypothesis goes back to first principles; it is never rationalized.',
  whenToUse: 'For an ADR whose falsification criteria need a measurement. args: {"adr": "docs/design/adr/ADR-0003-x.md", "id": "e1-outer-sync", "cloud": false}.',
  phases: [
    { title: 'Gate', detail: 'Design approved; the ADR exists; is it approved?' },
    { title: 'Plan', detail: 'Researcher: docs/experiments/<id>/PLAN.md + PLAN.json (predictions, local commands, cloud plan); validated by vast.mjs' },
    { title: 'Prototype', detail: 'Coder: the smallest prototype in experiments/<id>/, scope-audited' },
    { title: 'Local', detail: 'Verifier: correctness vs reference, performance, failure injection, ladder; script classification + LOCAL.json' },
    { title: 'Promote', detail: 'Promotion gate: checklist, ADR grant, plan+budget grant (else STOP for the human)' },
    { title: 'Cloud', detail: 'Vast.ai engineer through vast.mjs; script classification; zero live instances at the end' },
    { title: 'Report', detail: 'Researcher: LOCAL.md / RESULTS.md, expected vs observed; lessons on refutation' },
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
// @include scope
// @include experiment

const adr = String(args?.adr ?? '')
if (!/^docs\/design\/adr\/ADR-\d{4}[\w-]*\.md$/.test(adr)) throw new Error('args.adr must be docs/design/adr/ADR-NNNN-<slug>.md')
const id = String(args?.id ?? '')
if (!/^[a-z0-9][a-z0-9-]{1,48}$/.test(id)) throw new Error('args.id must match ^[a-z0-9][a-z0-9-]{1,48}$')
const DIR = `docs/experiments/${id}`
const PLAN_PATH = `${DIR}/PLAN.json`
const PROTO = `experiments/${id}/`
const EXP_PLAN = {
  type: 'object',
  description: `the content of ${PLAN_PATH}, verbatim`,
  required: ['id', 'hypothesis', 'expected', 'primary', 'tolerancePct', 'local'],
  properties: {
    id: { type: 'string' }, hypothesis: { type: 'string' }, primary: { type: 'string' }, tolerancePct: { type: 'number' },
    expected: { type: 'object', description: 'metric -> predicted value' },
    lowerIsBetter: { type: 'array', items: { type: 'string' } },
    local: { type: 'object', required: ['correctness', 'performance', 'failure'], properties: { correctness: { type: 'string' }, performance: { type: 'string' }, failure: { type: 'string' } } },
  },
}
const LOCAL_RUN = {
  type: 'object',
  required: ['correctness', 'performance', 'failure', 'ladder'],
  properties: {
    correctness: { type: 'object', required: ['exitCode'], properties: { exitCode: { type: 'number' }, output: { type: 'string' } } },
    performance: { type: 'object', required: ['exitCode', 'metrics'], properties: { exitCode: { type: 'number' }, metrics: { type: 'object', description: 'the JSON object the performance command printed last, verbatim' } } },
    failure: { type: 'object', required: ['exitCode'], properties: { exitCode: { type: 'number' }, output: { type: 'string' } } },
    falsified: { type: 'boolean', description: 'true if the observed results meet the ADR falsification criteria' },
    ladder: LADDER_RESULT,
  },
}
const CLOUD_RUN = {
  type: 'object',
  required: ['outcome', 'observed', 'liveInstances'],
  properties: {
    outcome: { type: 'string', enum: ['completed', 'emergency-stopped', 'paused', 'refused', 'preflight-failed'] },
    observed: { type: 'object', description: 'metric -> observed value, from the run\'s results command' },
    spentUsd: { type: 'number' },
    liveInstances: { type: 'number', description: 'from vast.mjs status at the end' },
    detail: { type: 'string' },
  },
}
const judge = (plan, observed, falsified) => {
  const lower = new Set(plan.lowerIsBetter ?? [])
  const metrics = Object.fromEntries(Object.entries(plan.expected ?? {}).map(([k, v]) => [k, { expected: v, observed: observed?.[k] ?? null, higherIsBetter: !lower.has(k) }]))
  return classifyExperiment({ metrics, tolerancePct: plan.tolerancePct ?? 10, primary: plan.primary, falsified: falsified === true })
}

phase('Gate')
await requireApprovedDesign('Gate')
const adrState = await checkApprovals([adr], 'gate:adr', 'Gate')
const adrDoc = (adrState?.docs ?? [])[0] ?? {}
if (adrDoc.reason === 'document does not exist') throw new Error(`${adr} does not exist — write it with wan-adr first`)
const adrApproved = approvedAll(adrState, [adr])
log(`Experiment ${id}: ${adr} is ${adrApproved ? 'approved' : 'NOT yet approved (local experiments may proceed; paid ones may not)'}`)

phase('Plan')
await agent(common(`OWNED PATHS: ${DIR}/PLAN.md, ${PLAN_PATH} (only).`) + `
Write the experiment plan for ${adr}:
- ${DIR}/PLAN.md: the hypothesis, the predictions DERIVED from the ADR's mathematical model (show the arithmetic), the falsification
  criteria, the local prototype (the smallest one in ${PROTO}), the trusted reference it is checked against, the failure injection, and
  — only if the ADR calls for it — the Vast.ai run with hardware, network profile, duration and budget.
- ${PLAN_PATH}: {"id": "${id}", "hypothesis", "expected": {metric: predicted value}, "lowerIsBetter": [metrics], "primary": the metric
  the falsification criterion names, "tolerancePct" (performance metrics only), "local": {"correctness": cmd (exit 0 only if the training
  state is byte-identical, by sha256, to the synchronous reference — CLAUDE.md "Product rule"; never a tolerance), "performance": cmd (prints a JSON object {metric: observed value} as its last line), "failure": cmd (failure injection;
  exit 0 only if the system behaves as specified)}, and for a cloud run the fields of agents-architetura/docs/EXPERIMENTS.md (maxBudgetUsd, warnAtPct,
  expectedHours, expectedTokens, instances, image, diskGb, setup, run, metrics, results, startupCheck, peerCheck, stop)}.`,
  { label: 'plan', phase: 'Plan', schema: RESULT, agentType: ROLE.researcher })
const plan = await agent(common('READ-ONLY.') + `Return the content of ${PLAN_PATH} verbatim.`, { label: 'extract:plan', phase: 'Plan', schema: EXP_PLAN, agentType: ROLE.readonly })
if (!plan || plan.id !== id || !plan.primary || !(plan.primary in (plan.expected ?? {}))) throw new Error(`${PLAN_PATH} is invalid: it needs id "${id}", expected metrics and a primary metric among them`)

phase('Prototype')
const since = await scopeSnapshot(`experiment-${id}`, 'Prototype')
await agent(common(`OWNED PATHS: ${PROTO} (only). Product code is READ-ONLY for you.`) + `
Build the smallest prototype that tests the hypothesis in ${DIR}/PLAN.md, in ${PROTO}, and make the three local commands in ${PLAN_PATH}
work (correctness vs the reference, performance printing the JSON metrics line, failure injection). Run them once yourself and report
the commands as evidence. A prototype is not product code: no shortcuts that would bias the measurement in the hypothesis' favor.`,
  { label: 'prototype', phase: 'Prototype', schema: TASK_RESULT, agentType: ROLE.coder })
await scopeCheck(`experiment-${id}`, 'Prototype', since, [PROTO])

phase('Local')
const local = await agent(common('You run ALONE — you MAY run anything (no git). You did not build this prototype; change nothing.') + `
Run, exactly and in order, and report each:
1. correctness: ${clean(plan.local?.correctness, 600)}
2. performance: ${clean(plan.local?.performance, 600)} — paste the JSON object it printed last into performance.metrics verbatim
3. failure injection: ${clean(plan.local?.failure, 600)}
4. ${ladderCommand({ only: ['unit', 'oracle', 'chaos', 'wan'] })} — paste its JSON into "ladder" verbatim
Then read the falsification criteria in ${adr} and set falsified=true if the observed results meet them.`,
  { label: 'local', phase: 'Local', schema: LOCAL_RUN, agentType: ROLE.verifier })
if (!local) throw new Error('the local experiment produced no result')
const localJudgment = judge(plan, local.performance?.metrics, local.falsified)
const checklist = promotionChecklist({
  hypothesis: plan.hypothesis,
  adrApproved,
  prototypeOk: local.performance?.exitCode === 0,
  correctnessOk: local.correctness?.exitCode === 0,
  ladderGreen: local.ladder?.allGreen === true,
  failureOk: local.failure?.exitCode === 0,
  classification: localJudgment.classification,
})
const localRecord = { adr, classification: localJudgment.classification, why: localJudgment.why, rows: localJudgment.rows, checklist }
await agent(common('You run ALONE — run this one command and report its output verbatim in "summary"; change nothing else.') + `
${recordCommand('experiment-local', localRecord, id)}`,
  { label: 'record:local', phase: 'Local', schema: RESULT, agentType: ROLE.verifier })
log(`Experiment ${id} locally: ${localJudgment.classification} — ${localJudgment.why}`)

let cloud = null
let cloudJudgment = null
let stoppedFor = null
if (args.cloud === true) {
  phase('Promote')
  const missing = Object.entries(checklist).filter(([, v]) => !v).map(([k]) => k)
  const planGrant = await checkApprovals([PLAN_PATH], 'gate:plan', 'Promote')
  // The human's hardware switch: with gpu off nothing reaches Vast.ai (vast.mjs refuses too); the local result stands.
  const hardware = missing.length ? null : await runTool('node agents-architetura/tools/factory/hardware.mjs show --json', { label: 'gate:hardware', phaseTitle: 'Promote' })
  if (missing.length) stoppedFor = `promotion checklist incomplete: ${missing.join(', ')}`
  else if (hardware?.json?.gpu !== true) stoppedFor = 'awaiting human approval of GPU use: the hardware switch is gpu=false (agents-architetura/tools/factory/hardware.json), so this run stays local; a human turns GPUs on with /wan-gpu, then runs this experiment again'
  else if (!approvedAll(planGrant, [PLAN_PATH])) stoppedFor = `awaiting human approval of the experiment plan and budget: node agents-architetura/tools/factory/approve.mjs grant ${PLAN_PATH}`
  if (!stoppedFor) {
    phase('Cloud')
    const V = `node agents-architetura/tools/vast/vast.mjs`
    cloud = await agent(common(`You run ALONE. Every paid action goes through ${V} — never the vastai CLI directly.`) + `
Run the approved experiment ${PLAN_PATH} on Vast.ai:
1. ${V} estimate --plan ${PLAN_PATH} — stop (outcome "refused") if it exits non-zero.
2. ${V} launch --plan ${PLAN_PATH} — stop (outcome "refused") if it exits non-zero.
3. ${V} preflight --plan ${PLAN_PATH} — on "DO NOT START TRAINING": attempt only safe remediation, re-run it once, and if it still fails
   run ${V} stop --plan ${PLAN_PATH} --reason "pre-flight failed" (outcome "preflight-failed").
4. ${V} start --plan ${PLAN_PATH}
5. Every 5 minutes run ${V} monitor --plan ${PLAN_PATH} --json until the run's results command reports completion. Exit 1 means it already
   emergency-stopped (outcome "emergency-stopped"); exit 3 means it PAUSED at the budget warning (outcome "paused": report and stop working
   — only a human decides what happens next).
6. On completion: run the plan's results command on the instances to read the observed metrics, then ${V} stop --plan ${PLAN_PATH} --reason "completed".
7. ${V} status --plan ${PLAN_PATH} — report its live instance count as liveInstances.`,
      { label: 'cloud', phase: 'Cloud', schema: CLOUD_RUN, agentType: ROLE.vastai })
    if ((cloud?.liveInstances ?? 1) !== 0) log(`Experiment ${id}: WARNING — ${cloud?.liveInstances ?? 'unknown'} instance(s) may still be running and billing: run ${V} stop --plan ${PLAN_PATH}`)
    if (cloud?.outcome === 'completed') {
      cloudJudgment = judge(plan, cloud.observed, false)
      await agent(common('You run ALONE — run this one command and report its output verbatim in "summary"; change nothing else.') + `
${recordCommand('experiment-results', { adr, local: localRecord.classification, cloud: cloudJudgment, spentUsd: cloud.spentUsd ?? null }, id)}`,
        { label: 'record:results', phase: 'Cloud', schema: RESULT, agentType: ROLE.verifier })
    } else stoppedFor = `cloud run ${cloud?.outcome ?? 'produced nothing'}: ${clean(cloud?.detail ?? '', 300)}`
  }
}

phase('Report')
const final = cloudJudgment ?? localJudgment
await agent(common(`OWNED PATHS: ${DIR}/LOCAL.md, ${DIR}/RESULTS.md, ${LESSONS} (only).`) + `
Write ${DIR}/LOCAL.md${cloudJudgment ? ` and ${DIR}/RESULTS.md` : ''}: expected vs observed for every metric (throughput, communication overhead, scaling,
cost, speedup — whatever the plan predicted), the conditions (workload, hardware, network, method), the classification below and its
reason — the classification is computed, do not argue with it. ${final.classification === 'REFUTED' ? `The hypothesis is REFUTED: say so plainly, state
what the evidence shows instead, recommend returning to first-principles analysis (a revised or withdrawn ADR), and append one line to
${LESSONS}. Do not rationalize it.` : ''}${stoppedFor ? ` The run stopped: ${clean(stoppedFor, 400)}.` : ''}
${fence('results', { local: localRecord, cloud: cloud ?? null, cloudJudgment }, 16000)}`,
  { label: 'report', phase: 'Report', schema: RESULT, agentType: ROLE.researcher })

return {
  ok: !stoppedFor && ['VALIDATED', 'PARTIALLY VALIDATED'].includes(final.classification),
  id,
  adr,
  local: localJudgment.classification,
  checklist,
  cloud: cloudJudgment ? cloudJudgment.classification : cloud ? cloud.outcome : null,
  classification: final.classification,
  stoppedFor,
  awaitingHuman: /awaiting human approval/.test(stoppedFor ?? ''),
}
