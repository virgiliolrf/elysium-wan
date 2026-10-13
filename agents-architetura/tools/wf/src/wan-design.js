export const meta = {
  name: 'wan-design',
  description: 'Design of record, the way an architecture team decides: the request specification + prior-art + current-architecture surveys -> a solo first-principles draft that preserves what exists unless evidence says otherwise (quantitative cost model, alternatives, falsifiable hypotheses) -> the security architect\'s threat model (docs/design/THREAT_MODEL.md) -> 4 adversarial lenses looped to convergence -> script-validated workstream ownership -> ADR-0001 with the ARCHITECTURE DECISION REQUIRED presentation -> fresh-eyes finalize. Docs only; nothing is approved here — a human grants the design and its ADR.',
  whenToUse: 'After the request specification (docs/design/BRIEF.md) is frozen, and whenever the human asks for a design revision. A human approves DESIGN.md and ADR-0001 (agents-architetura/tools/factory/approve.mjs grant) before any spec work starts.',
  phases: [
    { title: 'Survey', detail: '3 parallel surveyors (request spec, public prior art, current architecture); each writes only its own SURVEY-*.md' },
    { title: 'Draft', detail: 'Solo: docs/design/DESIGN.md (hard cap) + DESIGN-history.md, first-principles cost model' },
    { title: 'Threats', detail: 'Security architect (read-only) threat-models the draft; researcher writes THREAT_MODEL.md and closes gaps in the design' },
    { title: 'Review', detail: 'Loop <= 6: claims-and-mechanism, failures-and-threats, buildability, verifiability -> reviser -> size gate' },
    { title: 'Ownership', detail: 'Extract the workstreams block; script-validated disjoint ownership; repair loop <= 3' },
    { title: 'Decision', detail: 'Researcher writes ADR-0001 (template) with the decision presentation; one fidelity review' },
    { title: 'Finalize', detail: 'Fresh-eyes pass and executive summary; no Status line — approval is a human grant' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project

const SURVEYS = [
  ['brief', `Read ${BRIEF} (the human's brief — the only source of product intent). Extract: goals, hard constraints,
success criteria, explicit non-goals, trust assumptions (who runs the hardware, who may be adversarial), target
hardware and network conditions. List every ambiguity as a numbered question for the human. Quote the brief for
each extracted item; invent nothing it does not say.`],
  ['prior-art', `Survey PUBLIC prior art for training over slow, heterogeneous, wide-area links (use web search; cite a
URL for every entry): infrequent-synchronization data parallelism (local SGD and its descendants), gradient and
activation compression, pipeline parallelism over slow links and its scheduling, heterogeneity-aware partitioning,
fault tolerance under churn, Byzantine-robust aggregation. For each: the claim, the setting it was measured in, the
measured number, and whether independent work reproduced it. Also say whether it keeps the training result byte-identical
to synchronous datacenter training (CLAUDE.md, "Product rule"); if it does not, it is a baseline to measure against, never
an option. Separate MEASURED from CLAIMED explicitly.`],
  ['repo', `What exists today: read docs/design/CURRENT_ARCHITECTURE.md and docs/design/ARCHITECTURE_MAP.md (written by
wan-survey from the code) and any reference-system maps under docs/design/reference/; spot-check them against the code.
Report the components, interfaces and boundaries a design must preserve or explicitly migrate, the factory contracts
(CLAUDE.md, agents-architetura/tools/gates/rungs.json, agents-architetura/docs/VERIFICATION.md), and what the existing code already measures.`],
]
const OWN_DESIGN = `OWNED PATHS: ${DESIGN}, ${DESIGN_HISTORY} (only files you may write).`
const DESIGN_RULES = `
DESIGN DOC RULES:
- ${DESIGN} is HARD-CAPPED at ${DESIGN_CAP} bytes (wc -c after every save). Overflow goes to ${DESIGN_HISTORY}
  (binding annex: review-resolution logs, refutations, worked examples). Never cut normative content to fit.
- Every quantitative claim is MEASURED (cite the survey entry and its source) or labeled HYPOTHESIS with the
  experiment that would falsify it.
- Numbered invariants INV-1, INV-2, ... — each testable, each mapped to a verification-ladder rung id from:
  ${RUNG_IDS.join(', ')}.
- Decisions are recorded with rationale + rejected alternatives (one line each) so review does not relitigate.
- Production defaults: no "mock", "stub for now" or "phase-1 fake" in shipped components; a component is either
  real or explicitly out of scope.
- Byte-exact, lossless (CLAUDE.md, "Product rule"): every mechanism keeps the training result byte-identical to the
  synchronous datacenter reference. Never choose a lossy, async, local-step or order-changing method. If the brief
  seems to need one, record it as a numbered question for the human.
- Understand before modifying: what exists (docs/design/CURRENT_ARCHITECTURE.md) is preserved unless evidence shows it
  cannot satisfy the requirement. Every significant change records: current design, problem, evidence, alternatives
  considered, chosen design, trade-offs, migration impact, risks, validation plan.
- First principles before complexity: research generates possibilities; the cost model decides what belongs. A
  mechanism that does not change a dominant term of the model does not belong in the design.
- Local before paid cloud: every hypothesis names the local experiment that can refute it cheaply before any
  Vast.ai run.
- The workstreams block (exact format below) is machine-read by later workflows.`
const WORKSTREAMS_FORMAT = `
WORKSTREAMS BLOCK — exactly one, in this form:
<!-- workstreams -->
\`\`\`json
[{"key": "transport", "title": "...", "scope": "...", "ownedPaths": ["src/transport/", "tests/unit/transport/"],
  "provides": ["exact interface names/signatures"], "consumes": ["key:interface"]}]
\`\`\`
Rules: keys match ^[a-z0-9][a-z0-9-]{0,47}$; ownedPaths are repo-relative, directories end with "/", and are
DISJOINT across workstreams (the workflow rejects overlaps), and must not claim the harness directories
${HARNESS_DIRS.join(', ')} (wan-prep and the red team own them) or the agent factory ${FACTORY_DIRS.join(', ')} (not product). Shared hot files — pyproject.toml, Cargo.toml (workspace
root), the root package.json, agents-architetura/tools/gates/rungs.json, CLAUDE.md — belong to NO workstream: changes to them are
written as paste-ready entries in ${DESIGN_DIR}/INTEGRATE-<key>.md and merged by a solo integrator.`
const BRIEF_STATUS = {
  type: 'object',
  required: ['present', 'bytes'],
  properties: { present: { type: 'boolean' }, bytes: { type: 'number' } },
}

phase('Survey')
const brief = await agent(common('READ-ONLY.') + `
Check whether ${BRIEF} exists. If it does, report present=true and its size in bytes (Read it; count roughly).
If it does not exist or is empty, report present=false, bytes=0. Do nothing else.`,
  { label: 'brief:check', phase: 'Survey', schema: BRIEF_STATUS, agentType: ROLE.readonly })
if (!brief?.present || !(brief.bytes > 0)) throw new Error(`${BRIEF} is missing or empty — the human writes the brief first`)

const surveys = (await parallel(SURVEYS.map(([key, task]) => () =>
  agent(common(`OWNED PATHS: ${DESIGN_DIR}/SURVEY-${key}.md only.`) + `
Solo surveyor "${key}". Write ${DESIGN_DIR}/SURVEY-${key}.md (uncapped).
${task}
Cite a source for every load-bearing statement. No opinion without evidence.`,
    { label: `survey:${key}`, phase: 'Survey', schema: RESULT, agentType: ROLE.researcher })
))).filter(Boolean)
if (surveys.length < SURVEYS.length) throw new Error(`survey incomplete (${surveys.length}/${SURVEYS.length}) — resume this run`)

phase('Draft')
const draft = await agent(common(OWN_DESIGN) + DESIGN_RULES + WORKSTREAMS_FORMAT + `
Solo designer. Read ${BRIEF} and ${DESIGN_DIR}/SURVEY-*.md fully; follow their sources where you need detail.
Work first-principles: derive the bottlenecks with numbers, generate at least two genuinely different architectures, compare
them on bandwidth, latency, compute, memory, fault tolerance, cost and complexity, and choose the simplest that meets the
brief. Record the losers with the evidence that would make you reconsider them. Prefer a small falsifiable experiment
(milestone M0) over committing the whole architecture on an untested assumption.
Write ${DESIGN} with these sections:
0. Executive summary (10 lines) · goals · NON-goals (from the brief; anything else is a decision with rationale).
1. Product: who uses it, for what, and the single command/API they touch first.
2. System model: control plane vs data plane, every process, every wire protocol, trust boundaries.
3. Correctness oracle: the synchronous datacenter reference of the same recipe, with a canonical reduction order and
   deterministic, architecture-independent kernels. State how every mechanism keeps the result byte-identical to it
   (CLAUDE.md, "Product rule"): weights, optimizer state, RNG state, the loss at every step and checkpoints, by sha256.
   This holds with every feature on, under every failure and on every hardware mix. There are no lossy paths: a mechanism
   that cannot be byte-exact is not in the design.
4. Invariants INV-* (numbered, testable, each mapped to a ladder rung).
5. Security model per the request's trust assumptions (the security architect threat-models this draft next, into
   docs/design/THREAT_MODEL.md): trust boundaries, authentication and authorization between every pair of processes,
   secrets, workload isolation, and a mechanism per in-scope threat.
6. Failure model: churn, stragglers, partitions, coordinator loss — recovery per failure, with the bound on lost work.
7. Workstreams: table + the machine-read block (format above). Each workstream: owned paths, provides/consumes.
8. Milestones M0..Mn: each yields a runnable artifact and names the ladder rungs that must pass.
9. Performance model from first principles: write T_total = T_compute + T_communication + T_synchronization + T_scheduler +
   T_overhead (or the decomposition this system really has) with its actual variables (N workers, B bandwidth, L latency,
   P parameters, G gradient bytes, C compute time, S sync time, W throughput, R recovery time — only those that matter),
   evaluate it for the target hardware and network profiles, name the dominant term, and show which term each mechanism
   changes and by how much. Answer: network 10x slower? bandwidth- vs latency-bound? a worker disappears? several
   stragglers? GPUs of radically different speed? communication dominating compute? a malicious worker? a bad scheduler
   decision? the coordinator fails? Measured baselines from the prior-art survey; hypotheses with falsification tests;
   cost (GPU-hours, $ per million tokens).
10. Decisions (with rationale + rejected alternatives) · Open questions for the human (include the brief survey's).`,
  { label: 'draft', phase: 'Draft', schema: RESULT, agentType: ROLE.researcher })
if (!draft) throw new Error('draft agent produced nothing')
await sizeGate(DESIGN, DESIGN_HISTORY, DESIGN_CAP, 'sizegate:draft', 'Draft')

phase('Threats')
const THREATS = {
  type: 'object',
  required: ['boundaries', 'threats'],
  properties: {
    boundaries: { type: 'array', items: { type: 'string' }, description: 'trust boundaries along client -> control plane -> scheduler/coordinator -> workers -> runtime -> GPU/dataset/model' },
    threats: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'layer', 'threat', 'scenario', 'impact', 'mitigation', 'cost', 'gap', 'test'],
        properties: {
          id: { type: 'string', description: 'T1, T2, ...' },
          layer: { type: 'string' },
          threat: { type: 'string' },
          scenario: { type: 'string', description: 'how an attacker or failure does it, concretely' },
          impact: { type: 'string' },
          mitigation: { type: 'string', description: 'the design mechanism that stops it, with the design section — or "none"' },
          cost: { type: 'string', description: 'the mitigation\'s cost model (per byte/frame/connection/step) and its expected share of the step time, MEASURED or HYPOTHESIS' },
          gap: { type: 'boolean', description: 'true when the design has no adequate mitigation' },
          test: { type: 'string', description: 'the red-team or audit test that would prove the mitigation works' },
        },
      },
    },
  },
}
const threats = await agent(common('READ-ONLY security architect.') + `
Threat-model the draft ${DESIGN} against ${BRIEF} (its trust assumptions bind you) and ${DESIGN_DIR}/CURRENT_ARCHITECTURE.md.
Walk every hop: client -> control plane -> scheduler/coordinator -> workers -> training runtime -> GPU / dataset / model.
Consider at least: untrusted or impersonated workers, malicious clients, compromised or impersonated coordinators, credential
theft, authentication, authorization, secrets handling, workload isolation, arbitrary code execution (deserialization, job
payloads), model poisoning, dataset poisoning, network attacks (MITM, replay, reordering), resource exhaustion and denial of
service, malicious training jobs, dependency/supply-chain attacks. A threat the request puts out of scope is listed with that
justification, not dropped. gap=true wherever the design has no adequate mitigation.${SECURITY_COST}`,
  { label: 'threats', phase: 'Threats', schema: THREATS, agentType: ROLE.security })
const threatList = (threats?.threats ?? []).map(t => ({ ...t, id: token(t.id, 12) }))
if (!threatList.length) throw new Error('the security architect produced no threat model — resume this run')
const gaps = threatList.filter(t => t.gap)
await agent(common(`OWNED PATHS: ${DESIGN_DIR}/THREAT_MODEL.md, ${DESIGN}, ${DESIGN_HISTORY} (only).`) + DESIGN_RULES + `
1. Write ${DESIGN_DIR}/THREAT_MODEL.md from the security architect's analysis below: trust boundaries, then one row per threat
   (id, layer, threat, scenario, impact, mitigation with design section, residual risk, verifying test), out-of-scope threats with
   their justification, then a cost section: per mitigation its cost model and expected share of the step time (MEASURED or
   HYPOTHESIS), and the mitigations rejected as cost without a new property.
2. For each of the ${gaps.length} gap(s), add the mechanism to ${DESIGN} (or record in Decisions why the risk is accepted, with the
   evidence) — the design and the threat model must agree.
${fence('threat_model', { boundaries: threats.boundaries, threats: threatList }, 30000)}`,
  { label: 'threat-model', phase: 'Threats', schema: RESULT, agentType: ROLE.researcher })
await sizeGate(DESIGN, DESIGN_HISTORY, DESIGN_CAP, 'sizegate:threats', 'Threats')

phase('Review')
const LENSES = [
  ['claims-and-mechanism', `Every number and capability claim: is it measured (open the cited survey entry and check the
number against its source) or labeled a hypothesis with a falsification test? A claim stronger than its source is a
BLOCKER. Is the performance model DERIVED — the dominant term identified with the target numbers, each mechanism tied to
the term it changes, the conditions under which the gain disappears stated — or decorative? A mechanism justified only by
"a paper does it" or "it is common" is a major finding. Does the design preserve the existing architecture, or document the
evidence for each change? Scope no team could build is major. Product rule (CLAUDE.md): any mechanism that can change one bit of
the training result versus the datacenter reference is a BLOCKER. That covers lossy compression; local or infrequent sync;
async or stale updates; dropped or reweighted contributions; a reduction order set by topology, arrival or speed; and kernels
whose bits differ across GPU architectures.`],
  ['failures-and-threats', `Correctness: does every distributed mechanism have a stated oracle and a test that would catch
it diverging? Invariants testable and mapped to rungs? Walk the failure model end to end (a worker vanishing mid-step,
stragglers, a partitioned or failed coordinator, a partial checkpoint, a bad scheduler decision) and ${DESIGN_DIR}/THREAT_MODEL.md
(every in-scope threat has a mechanism in the design and a test). A hand-off or threat with no owner is a BLOCKER.`],
  ['verifiability', `Can the factory verify this at the reference PR's depth (agents-architetura/docs/VERIFICATION.md)? Every network, time
and randomness effect behind an injectable Transport/Clock/Rng so a deterministic simulator can run the whole cluster in one
process; perturb(site) hooks named on the slow paths; every WAN optimization behind a flag whose off-path is byte-identical to
the synchronous reference mode; the exact oracle for every mechanism with every feature on (byte-identical training state,
by sha256: a bounded or statistical oracle is a BLOCKER under the product rule); every untrusted input boundary
listed with its fuzz target and never reaching pickle; numbered invariants. A mechanism no campaign, oracle or fuzzer can
check is a BLOCKER — it would ship unverified.`],
  ['buildability', `Could ~5 agents implement the workstreams concurrently from this document without redesigning?
Hunt: interfaces consumed but provided by nobody (or with a different signature), overlapping ownedPaths, shared hot
files claimed by a workstream, milestones without a runnable artifact or without named rungs, hand-waved steps
("somehow synchronize"). Production defaults: any mock/stub in a shipped component is a major finding.`],
]
const review = await reviewUntilClean({
  label: 'DESIGN.md',
  lenses: LENSES,
  maxRounds: 6,
  review: (name, lens, round) => agent(common('READ-ONLY adversarial reviewer.') + `
Review ${DESIGN} (round ${round}, lens ${name}). Assume it is wrong until the document, the brief and the surveys
prove otherwise. LENS: ${lens}
${round > 1 ? 'The document was revised after earlier findings: review the CURRENT text from scratch; revisions introduce new errors as often as they fix old ones.' : ''}
Re-litigating a recorded decision-with-rationale is not a finding. Severity: blocker = building from this produces
an unsound or unbuildable system; major = forces a mid-flight redesign; minor = everything else. No style nits.`,
    { label: `review:${name}:r${round}`, phase: 'Review', schema: FINDINGS, agentType: ROLE.reviewer }),
  fix: async (serious, round) => {
    await agent(common(OWN_DESIGN) + DESIGN_RULES + `
Review round ${round} filed these findings against ${DESIGN} (blocker/major first; a minor is cheap, fix it unless it conflicts). For each: verify it against the brief and
the surveys; if real, revise the design; if false, add a ONE-LINE refutation note in the Decisions section (full
argument in ${DESIGN_HISTORY}) so the next round does not trip on it. The revised document is re-reviewed from scratch.
${fence('reviewer_findings', serious, 30000)}`,
      { label: `revise:r${round}`, phase: 'Review', schema: RESULT, agentType: ROLE.researcher })
    await sizeGate(DESIGN, DESIGN_HISTORY, DESIGN_CAP, `sizegate:r${round}`, 'Review')
  },
})

phase('Ownership')
let ownership = { ok: false, errors: ['not checked'], keys: [] }
for (let attempt = 1; attempt <= 3; attempt++) {
  const extracted = await agent(common('READ-ONLY.') + `
Read ${DESIGN}. Find the block after "<!-- workstreams -->" and return its JSON array as "workstreams",
verbatim (do not fix, reorder or complete it). If the block is missing or not valid JSON, return an empty array.`,
    { label: `extract:a${attempt}`, phase: 'Ownership', schema: WORKSTREAMS, agentType: ROLE.readonly })
  const ws = extracted?.workstreams ?? []
  const check = ws.length
    ? validatePlan([...ws.map(w => ({ id: w.key, deps: [], scope: w.ownedPaths })), { id: 'harness', deps: [], scope: HARNESS_DIRS }, { id: 'factory', deps: [], scope: FACTORY_DIRS }])
    : { ok: false, errors: ['workstreams block missing, empty or invalid JSON'] }
  ownership = { ok: check.ok, errors: check.errors, keys: ws.map(w => token(w.key)) }
  if (check.ok) { log(`Ownership: ${ws.length} workstreams, disjoint (${ownership.keys.join(', ')})`); break }
  log(`Ownership attempt ${attempt}: ${check.errors.length} problem(s) -> repairing`)
  await agent(common(OWN_DESIGN) + WORKSTREAMS_FORMAT + `
The machine check of the workstreams block failed. Fix the block (and the prose that describes ownership) so every
error below is resolved without dropping any responsibility — re-home it explicitly instead.
${fence('ownership_errors', check.errors, 8000)}`,
    { label: `ownership-fix:a${attempt}`, phase: 'Ownership', schema: RESULT, agentType: ROLE.researcher })
}

phase('Decision')
const ADR = `${DESIGN_DIR}/adr/ADR-0001-initial-architecture.md`
await agent(common(`OWNED PATHS: ${ADR} (only).`) + `
Write ${ADR} from the template agents-architetura/docs/templates/ADR.md — the architecture decision memo for this design, with the
"ARCHITECTURE DECISION REQUIRED" presentation block filled in at the top. Every section of the template is mandatory;
Falsification Criteria names the experiment and the observed result that would prove the architecture wrong; Local
Experiment comes before any Vast.ai cost; Mathematical Model is the design's performance model, not decoration. Take every
fact from ${DESIGN}, ${DESIGN_DIR}/THREAT_MODEL.md and the surveys — invent nothing. Recommendation: PROCEED or DO NOT PROCEED,
with the reason. Do not write a Status line: approval is a human grant recorded by agents-architetura/tools/factory/approve.mjs.
${review.converged ? '' : `Adversarial review did NOT converge (${review.findingsPerRound.join(' -> ')}): list the unresolved findings under Risks.`}`,
  { label: 'adr', phase: 'Decision', schema: RESULT, agentType: ROLE.researcher })
const adrCheck = await agent(common('READ-ONLY adversarial reviewer.') + `
Check ${ADR} against ${DESIGN} and ${DESIGN_DIR}/THREAT_MODEL.md: every template section present and non-empty; the presentation
block matches the body; the falsification criteria are concrete and observable; the cost estimate is derived; nothing in the ADR
contradicts or overstates the design. Findings: blocker/major only.`,
  { label: 'adr-review', phase: 'Decision', schema: FINDINGS, agentType: ROLE.reviewer })
const adrIssues = (adrCheck?.findings ?? []).filter(isActionable)
if (adrIssues.length) {
  await agent(common(`OWNED PATHS: ${ADR} (only).`) + `
Fix these findings in ${ADR} (verify each against the design first; refute false ones in the Risks section with evidence).
${fence('reviewer_findings', adrIssues, 12000)}`,
    { label: 'adr-fix', phase: 'Decision', schema: RESULT, agentType: ROLE.researcher })
}

phase('Finalize')
const final = await agent(common(OWN_DESIGN) + DESIGN_RULES + `
FRESH-EYES finalizer (no stake in prior rounds). Read ${DESIGN} and the history end to end as a skeptical
implementer: where would you have to guess? Fix what you find (record it as the final round in the history),
keep the cap and make sure the executive summary is accurate. Do NOT write a Status line: approval is a human grant of
this exact text and of ${ADR} (agents-architetura/tools/factory/approve.mjs), recorded outside the document.
${review.converged ? '' : `NOTE: adversarial review did NOT converge (${review.findingsPerRound.join(' -> ')}). List the unresolved findings under "Open questions for the human".`}
${ownership.ok ? '' : `NOTE: workstream ownership is still invalid: ${fence('ownership_errors', ownership.errors, 4000)} — list it under "Open questions for the human".`}`,
  { label: 'finalize', phase: 'Finalize', schema: RESULT, agentType: ROLE.researcher })
await sizeGate(DESIGN, DESIGN_HISTORY, DESIGN_CAP, 'sizegate:final', 'Finalize')

return {
  ok: review.converged && ownership.ok && !!final,
  design: DESIGN,
  adr: ADR,
  threatModel: `${DESIGN_DIR}/THREAT_MODEL.md`,
  threats: threatList.length,
  threatGaps: gaps.length,
  surveys: SURVEYS.map(([key]) => `${DESIGN_DIR}/SURVEY-${key}.md`),
  review,
  ownership,
  finalized: !!final,
  next: `Human: read ${ADR}, ${DESIGN} and ${DESIGN_DIR}/THREAT_MODEL.md, then approve with /wan-approve in Claude Code, or in your own terminal: node agents-architetura/tools/factory/approve.mjs grant ${DESIGN} ${ADR}`,
}
