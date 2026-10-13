export const meta = {
  name: 'wan-closeout',
  description: 'Honest final gate for a milestone (the reference PR\'s pinned verify + mega-run): the independent verifier runs `node agents-architetura/tools/gates/pinned.mjs <m>` — exactly the bar frozen in GATES-<m>.json (rungs green twice in a row, pinned commands with repetitions, campaigns re-run to zero, fuzz bars, audit, mutation) — and the script accepts it only if its tree is the current tree, no invariant test still says "awaiting implementation" and adversarial review converged. Failing gates are routed to the workflow that owns them (wan-fix, wan-campaign, wan-fuzz, wan-audit, wan-bughunt), <= 4 rounds. Verdict: VERIFIED or NOT VERIFIED, never "mostly". The researcher writes the report and distills lessons.',
  whenToUse: 'Last step of a milestone (wan-cycle runs it). args: {"milestone": "M0", "review": <wan-implement review result, optional>}.',
  phases: [
    { title: 'Gate', detail: 'Design approved; plan re-validated; GATES-<m>.json present' },
    { title: 'Verify', detail: 'Verifier: pinned.mjs (fresh), current tree id, pending-marker scan, bench baseline' },
    { title: 'Stabilize', detail: 'Each failing gate to its owner workflow, <= 4 rounds (the reference PR\'s budget); pending markers stop the closeout' },
    { title: 'Report', detail: 'Researcher: milestone report with the verdict + lessons appended to agents-architetura/docs/LESSONS.md' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include project
// @include gates
// @include plan
// @include ledger
// @include campaign

const m = milestoneArg(args)
// The reference PR's stabilize budget: up to 4 fix rounds, each followed by the full pinned gate again.
const MAX_STABILIZE = 4
const REPORT = `agents-architetura/docs/reports/${m}.md`
const GATES_PATH = `${DESIGN_DIR}/GATES-${m}.json`
const reviewEvidence = args?.review && typeof args.review === 'object' ? args.review : null
const PINNED_CHECK = {
  type: 'object',
  required: ['pinned', 'tree', 'pendingMarkers'],
  properties: {
    pinned: { type: 'object', description: `the JSON printed by node agents-architetura/tools/gates/pinned.mjs ${m} --json, verbatim` },
    tree: { type: 'string', description: 'the tree id printed by node agents-architetura/tools/wf/scope-audit.mjs snapshot AFTER the gate ran' },
    pendingMarkers: { type: 'array', items: { type: 'string' } },
    benchRecorded: { type: 'boolean' },
  },
}

phase('Gate')
await requireApprovedDesign('Gate')
const workstreams = await loadWorkstreams('Gate')
const { plan } = await loadPlan(m, workstreams, 'Gate')
const invariants = [...new Set(plan.tasks.flatMap(t => t.invariants ?? []).map(i => token(i, 40)))]
const gates = await agent(common('READ-ONLY.') + `
Read ${GATES_PATH} and return its JSON verbatim. If it is missing or invalid, return
{"milestone": "", "consecutive": 0, "rungs": [], "pinned": [], "campaigns": [], "fuzz": []}.`,
  { label: 'extract:gates', phase: 'Gate', schema: GATES, agentType: ROLE.readonly })
const gateErrors = checkGates(gates, { milestone: m, planRungs: plan.tasks.flatMap(t => t.rungs ?? []) })
if (gateErrors.length) throw new Error(`${GATES_PATH} is missing or invalid — rerun wan-handout: ${gateErrors.slice(0, 5).join('; ')}`)
log(`Closeout ${m}: gates ${gates.rungs.length} rungs x${gates.consecutive}, ${gates.pinned.length} pinned, ${gates.campaigns.length} campaigns, ${gates.fuzz.length} fuzz${gates.audit ? ', audit' : ''}${gates.mutation ? ', mutation' : ''}; ${invariants.length} invariants; review ${reviewEvidence ? (reviewEvidence.converged ? 'converged' : 'NOT converged') : 'not provided'}`)

// The gate runs detached for hours (waitPinned); the verifier then reads its stored verdict, which prints at once.
const verifyOnce = async round => {
  if (!(await waitPinned(m, `r${round}`, 'Verify'))) throw new Error(pinnedStillRunning(m, `r${round}`))
  return agent(common('You run ALONE — you MAY build and run anything (no git). Fix nothing.') + `
Closeout verification for milestone ${m}, round ${round}. You are the independent verifier.
1. Run: ${pinnedCommand(m, `r${round}`)}
   The frozen gates have finished, so it prints their stored verdict (one JSON object) at once: paste it into "pinned" verbatim.
   Never paste an older result (.factory/gates/${m}.json) and never rerun a single gate to improve it — a pass on anything other
   than this command is not a pass.
2. Then run: node agents-architetura/tools/wf/scope-audit.mjs snapshot — put the "tree" it prints into "tree".
3. Search the whole repository for test markers containing "${AWAITING}" that reference any of these invariants:
   ${JSON.stringify(invariants)} — list each as file:line in pendingMarkers (empty list if none).
4. If bench/suite.json exists and bench/baseline.json does not: run node agents-architetura/tools/gates/bench-gate.mjs --record on this build and
   set benchRecorded=true.`,
  { label: `verify:r${round}`, phase: 'Verify', schema: PINNED_CHECK, agentType: ROLE.verifier })
}

// The verdict comes from pinned.mjs, and only counts if it is of the tree that exists now.
const judge = c => {
  const p = c?.pinned ?? {}
  const why = []
  if (p.verdict !== 'VERIFIED') why.push(`pinned verdict ${clean(p.verdict ?? 'missing', 40)}${(p.failing ?? []).length ? ` (${p.failing.map(f => clean(f, 60)).join(', ')})` : ''}`)
  if (!p.tree || p.tree !== c?.tree) why.push(`stale evidence: gate ran on ${clean(p.tree ?? 'no tree', 64)}, tree is ${clean(c?.tree ?? '?', 64)}`)
  if ((c?.pendingMarkers ?? []).length) why.push(`${c.pendingMarkers.length} invariant test(s) still "${AWAITING}"`)
  // a skipped gate (the heavy tail on an already red gate) has no owner to fix it: the rerun after the fix runs it
  return { verified: !why.length, why, failing: (p.gates ?? []).filter(g => g && g.ok !== true && g.kind !== 'skipped') }
}

// Every failing gate goes to the workflow that owns its mechanism.
const stabilize = async failing => {
  const rungs = [...new Set(failing.filter(g => g.kind === 'rung' && String(g.id).startsWith('rung:')).map(g => String(g.id).slice(5)))]
  if (failing.some(g => g.id === 'consecutive')) rungs.push(...gates.rungs)
  const fixRungs = knownRungs([...new Set([...rungs, ...(failing.some(g => g.id === 'mutation') ? ['mutation'] : [])])])
  if (fixRungs.length) await workflow('wan-fix', { rungs: fixRungs })
  for (const g of failing.filter(g => g.kind === 'campaign')) await workflow('wan-campaign', { campaign: token(String(g.id).slice(9), 40) })
  for (const g of failing.filter(g => g.kind === 'fuzz')) await workflow('wan-fuzz', { campaign: token(String(g.id).slice(5), 40) })
  if (failing.some(g => g.id === 'audit')) await workflow('wan-audit', { close: true })
  for (const g of failing.filter(g => g.kind === 'pinned')) {
    await workflow('wan-bughunt', { bug: `pinned gate ${clean(g.id, 60)} of ${m} fails: ${clean(g.detail, 800)}`, id: token(`${m}-${String(g.id).slice(7)}`, 40), repro: clean(g.cmd ?? '', 500) })
  }
}

phase('Verify')
let check = await verifyOnce(0)
if (!check) throw new Error('verifier produced nothing')
let verdict = judge(check)
let round = 0
while (!verdict.verified && round < MAX_STABILIZE) {
  if ((check.pendingMarkers ?? []).length) {
    log(`Closeout ${m}: ${check.pendingMarkers.length} test(s) still "${AWAITING}" — the milestone is not implemented; not a fix-loop problem`)
    break
  }
  round++
  phase('Stabilize')
  log(`Stabilize ${round}: ${verdict.why.join('; ')}`)
  await stabilize(verdict.failing)
  phase('Verify')
  check = await verifyOnce(round)
  if (!check) throw new Error('verifier produced nothing')
  verdict = judge(check)
}
const reviewOk = !reviewEvidence || reviewEvidence.converged === true
const verified = verdict.verified && reviewOk
const statusLine = verified ? 'VERIFIED' : `NOT VERIFIED (${[...verdict.why, ...(reviewOk ? [] : ['adversarial review did not converge'])].join('; ')})`

phase('Report')
await agent(common(`OWNED PATHS: ${REPORT}, ${LESSONS} (only).`) + `
1. Write the milestone report ${REPORT} from the plan (${planPath(m)}), the handout (${handoutPath(m)}), the gates (${GATES_PATH}), the run
   ledger (${runDir(m)}/state.json and transitions.jsonl), the task journals, the campaign/fuzz/audit results under docs/, the gate
   result and the review evidence below. Status line at the top, verbatim: "Status: ${clean(statusLine, 600)}". Sections:
   # Feature · # Architectural Changes (ADRs and their human grants; architecture as built vs the design) · # Files Changed ·
   # Tasks Completed (the DAG as executed: waves, retries, failures, conflicts) · # Tests (every gate: PASS / FAIL with exact counts —
   rungs x consecutive passes, pinned commands with repetitions, campaigns, fuzz CPU-hours, mutation survivors — and the tree they ran on) ·
   # Distributed Training Validation (keep SYSTEM WORKED separate from TRAINING RESULT IS CORRECT, with the oracle evidence) · # Security
   (threat-model coverage, audit verdicts, high-risk task reviews) · # Performance (before/after, workload, hardware, network, method) ·
   # Failure Recovery · # Benchmarks · # Vast.ai Results (expected vs observed and classification, or "none run") · # Known Limitations ·
   # Remaining Risks · # Acceptance Criteria (each with its evidence, or NOT MET) · # Recommended Next Work. Every number comes from the
   gate JSON, the ledger, a results file or a file you read — label anything else as an assumption.
2. Append to ${LESSONS} one line per NEW lesson this milestone taught (from retries, spec conflicts, real review findings, campaign
   families, fuzz crashes, audit findings): what went wrong, the rule that prevents it, where it was caught. No duplicates.
${fence('closeout_check', check, 30000)}
${fence('review_evidence', reviewEvidence ?? { note: 'not provided' }, 4000)}`,
  { label: 'report', phase: 'Report', schema: RESULT, agentType: ROLE.researcher })

return {
  ok: verified,
  verdict: verified ? 'VERIFIED' : 'NOT VERIFIED',
  why: verdict.why,
  milestone: m,
  gates: GATES_PATH,
  stabilizeRounds: round,
  failing: verdict.failing.map(g => g.id),
  tree: check.tree ?? null,
  pendingMarkers: check.pendingMarkers ?? [],
  reviewConverged: reviewEvidence ? reviewEvidence.converged === true : null,
  benchRecorded: check.benchRecorded === true,
  report: REPORT,
}
