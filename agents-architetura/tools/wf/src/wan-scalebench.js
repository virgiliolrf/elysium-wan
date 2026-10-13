export const meta = {
  name: 'wan-scalebench',
  description: 'Comparative scalability study, modeled on the reference PR\'s scalebench (JS vs Go vs Java under binding fairness rules): the researcher specifies the workload, baselines, network and heterogeneity profiles, fairness rules and the checksums every run must match -> implementers write runner, baselines and network emulation in disjoint directories (scope-audited) -> fairness and measurement-validity reviewers loop -> a solo run on the declared hardware under the machine lock (honest n/a without it) -> the script-checked validity (same data order and initial weights everywhere) -> report.',
  whenToUse: 'Once the system trains end to end (wan-task runs it after the last milestone). args: {}.',
  phases: [
    { title: 'Gate', detail: 'Design approved' },
    { title: 'Design', detail: 'Researcher: bench/scalebench/SPEC.md + hardware probe + run command + implementer split' },
    { title: 'Implement', detail: 'Parallel implementers, disjoint dirs under bench/scalebench/, scope-audited' },
    { title: 'Review', detail: 'fairness + measurement-validity reviewers, <= 5 rounds' },
    { title: 'Run', detail: 'Verifier: hardware probe, full matrix under the machine lock, validity check' },
    { title: 'Report', detail: 'Researcher: bench/scalebench/RESULTS.md' },
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

const DIR = 'bench/scalebench'
const SCALE_SPEC = {
  type: 'object',
  required: ['hardwareProbe', 'runCommand', 'implementers'],
  properties: {
    hardwareProbe: { type: 'string', description: 'command that exits 0 only when the hardware the SPEC requires is present' },
    runCommand: { type: 'string', description: `command that runs the full matrix and writes ${DIR}/results.json` },
    implementers: {
      type: 'array',
      items: { type: 'object', required: ['id', 'dir', 'task'], properties: { id: { type: 'string' }, dir: { type: 'string', description: `a directory under ${DIR}/ ending in "/"` }, task: { type: 'string' } } },
    },
  },
}
const SCALE_RUN = {
  type: 'object',
  required: ['ran', 'reason', 'validity'],
  properties: {
    ran: { type: 'boolean', description: 'false when the hardware probe failed' },
    reason: { type: 'string' },
    validity: { type: 'object', description: `the JSON printed by node agents-architetura/tools/gates/scalebench-check.mjs --json, verbatim ({} if not run)` },
  },
}

phase('Gate')
await requireApprovedDesign('Gate')

phase('Design')
const spec = await agent(common(`OWNED PATHS: ${DIR}/SPEC.md (only). You MAY use web search for baseline systems and their published settings.`) + `
Write ${DIR}/SPEC.md — a scalability study of this system against the strongest public baselines for training over slow,
heterogeneous wide-area links (at least a synchronous data-parallel baseline and a local-SGD/DiLoCo-style baseline; justify any
other). It must fix, as binding rules:
1. WORKLOAD: model sizes, dataset, token budget, target loss; time-to-target-loss is the primary metric (plus tokens/s, bytes on
   the wire per step, scaling efficiency, final eval).
2. FAIRNESS: same model, data order, seeds, optimizer and hyperparameter budget for every system; baselines at their documented
   defaults (any tuning documented and given to every system); no system gets a hand-optimized path the others lack.
3. CHECKSUMS: every run records checksums.dataOrder and checksums.initWeights. They must be identical across systems and node
   counts; a mismatch invalidates the study. Our system's runs also record checksums.finalState, the sha256 of the training state
   at the target step. It must be the same for every node count, network profile and repetition, and equal to the synchronous
   reference's (CLAUDE.md, "Product rule"); baselines need not match. ${DIR}/results.json format: {"ours": system name,
   "reference": system name, "runs": [{"system", "nodes", "rep", "arm"?, "checksums": {...}, "metrics": {...}}], "arms", "attribution"}.
4. MATRIX: nodes {1, 2, 4, 8}; network profiles (datacenter; cross-region ~1 Gbps / 80 ms; consumer ~100 Mbps / 150 ms / 0.5% loss)
   emulated with tc netem (Linux, root) or toxiproxy; heterogeneity profiles; 5 repetitions, median; systems interleaved, warmup
   discarded; a quiet machine.
5. HARDWARE: what the study needs, and a probe command that exits 0 only when it is present.
6. ARMS (the reference PR's discriminating arms, e.g. its "flat" arm that split threading cost from object-model cost): every
   gap the study reports — ours vs the best baseline, measured vs the performance model — is ATTRIBUTED, not only observed.
   Define variants of our system that each change exactly ONE factor (network profile, pipeline schedule, rebalancing off,
   serialization, compute kernel...), run them in the same matrix, and record results.json "arms": [{"id", "of", "varies"}],
   each arm run with "arm": "<id>", and "attribution": [{"arm": "<id>" | "residual", "factor", "sharePct"}] summing to
   100% ± 10. Our arms keep the byte-exact final state. agents-architetura/tools/gates/scalebench-check.mjs enforces this.
Split the implementation into at most 4 implementers with disjoint directories under ${DIR}/ (runner, one per baseline, network).`,
  { label: 'design', phase: 'Design', schema: SCALE_SPEC, agentType: ROLE.researcher })
const implementers = (spec?.implementers ?? []).map(x => ({ ...x, id: token(x.id, 24), dir: String(x.dir ?? '') }))
  .filter(x => x.dir.startsWith(`${DIR}/`) && x.dir.endsWith('/') && safeScopePath(x.dir)).slice(0, 4)
if (!spec || !implementers.length) throw new Error('the scalebench design produced no valid implementer split')
const disjoint = validatePlan(implementers.map(x => ({ id: x.id, deps: [], scope: [x.dir] })))
if (!disjoint.ok) throw new Error(`implementer directories overlap: ${disjoint.errors.join('; ')}`)
for (const k of ['hardwareProbe', 'runCommand']) if (/[\x00-\x1f]/.test(spec[k] ?? '') || !String(spec[k] ?? '').trim()) throw new Error(`${k} must be a printable command`)

phase('Implement')
const since = await scopeSnapshot('scalebench', 'Implement')
await parallel(implementers.map(x => () =>
  agent(common(`OWNED PATHS: ${x.dir} (only). Write code only — no runs.`) + `
Implement your part of the study in ${DIR}/SPEC.md: ${fence('task', x.task, 3000)}
Follow the SPEC's fairness and checksum rules exactly; the runner writes ${DIR}/results.json in the SPEC's format.`,
    { label: `impl:${x.id}`, phase: 'Implement', schema: RESULT, agentType: ROLE.coder })))
await scopeCheck('scalebench', 'Implement', since, implementers.map(x => x.dir))

phase('Review')
const review = await reviewUntilClean({
  label: 'scalebench',
  lenses: [
    ['fairness', 'Does every system run the same algorithmic work at the same level of effort? Hand-tuned paths for ours only, baselines crippled by bad defaults, mismatched data order or seeds, unequal tuning budgets?'],
    ['measurement-validity', 'Is the protocol sound: warmup discarded, systems interleaved, repetitions and medians, quiet-machine check, emulated network actually applied (verify the netem/toxiproxy wiring), checksums recorded and compared, time-to-target measured to the same target?'],
  ],
    review: (name, lens, round) => agent(common('READ-ONLY adversarial reviewer.') + `
Review the scalability study in ${DIR}/ against ${DIR}/SPEC.md (round ${round}, lens ${name}). ${lens}`,
    { label: `review:${name}:r${round}`, phase: 'Review', schema: FINDINGS, agentType: ROLE.reviewer }),
  fix: (serious, round) => agent(common(`OWNED PATHS: ${DIR}/ (only, except SPEC.md). Write code only — no runs.`) + `
Fix these review findings (round ${round}); refute false ones in your summary with file:line evidence.
${fence('reviewer_findings', serious, 20000)}`,
    { label: `fix:r${round}`, phase: 'Review', schema: RESULT, agentType: ROLE.coder }),
})

phase('Run')
const run = await agent(common('You run ALONE — you MAY run anything (no git). Change nothing.') + `
1. Run the hardware probe: ${clean(spec.hardwareProbe, 500)}
   If it fails: ran=false, reason = its output; stop.
2. Run: node agents-architetura/tools/verify/machine-lock.mjs run --label scalebench --wait 3600 -- ${clean(spec.runCommand, 800)}
3. Run: node agents-architetura/tools/gates/scalebench-check.mjs --json — paste its JSON into "validity" verbatim.`,
  { label: 'run', phase: 'Run', schema: SCALE_RUN, agentType: ROLE.verifier })
const ran = run?.ran === true
const valid = ran && run?.validity?.ok === true

phase('Report')
await agent(common(`OWNED PATHS: ${DIR}/RESULTS.md (only).`) + `
Write ${DIR}/RESULTS.md from ${DIR}/results.json: per profile and node count, time-to-target-loss and the other SPEC metrics for every
system (medians with min/max), scaling efficiency, and what the numbers do and do not show. ${ran ? (valid ? '' : 'The validity check FAILED: say so first; the numbers are not a valid comparison.') : `The study did not run (${clean(run?.reason, 300)}): say so; report no numbers.`}
${fence('run', run ?? {}, 8000)}`,
  { label: 'report', phase: 'Report', schema: RESULT, agentType: ROLE.researcher })

log(`Scalebench: ${!ran ? `n/a — ${clean(run?.reason, 200)}` : valid ? 'valid study recorded' : 'INVALID (checksum or system mismatch)'}`)
return { ok: review.converged && (!ran || valid), applicable: ran, valid, review: { converged: review.converged, rounds: review.rounds }, reason: ran ? null : clean(run?.reason, 300) }
