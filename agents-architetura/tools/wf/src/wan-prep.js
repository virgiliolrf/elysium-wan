export const meta = {
  name: 'wan-prep',
  description: 'From an APPROVED design to a verifiable substrate: one frozen spec per workstream (draft -> size gate -> 3-lens adversarial loop), a whole-system review of all specs together, harness writers (oracle, chaos, fuzz, bench) on disjoint paths, a solo skeleton builder that merges every INTEGRATE manifest, invariant tests per workstream, and a solo ladder verification.',
  whenToUse: 'Once per approved design, and again after a design revision. Requires a human grant of docs/design/DESIGN.md (agents-architetura/tools/factory/approve.mjs).',
  phases: [
    { title: 'Gate', detail: 'Design approved; workstream ownership disjoint and clear of the harness directories' },
    { title: 'Specs', detail: 'Per workstream (pipelined): draft SPEC -> size gate -> soundness/implementability/contracts loop <= 5' },
    { title: 'System', detail: 'All specs as one system: cohesion/end-to-end/performance loop <= 5, one solo reviser' },
    { title: 'Freeze', detail: 'Converged specs frozen on the record (agents-architetura/tools/factory/freeze.mjs); later changes only by recorded amendment' },
    { title: 'Harness', detail: 'Parallel writers, disjoint paths, no builds: oracle, chaos, fuzz, distributed matrix, bench' },
    { title: 'Skeleton', detail: 'Solo builder: scaffolding + merge every INTEGRATE manifest; build/fmt/lint/types green' },
    { title: 'Tests', detail: 'Per workstream: executable invariant tests marked "awaiting implementation"' },
    { title: 'Verify', detail: 'Solo ladder up to unit; wan-fix if red' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include triage
// @include project
// @include gates
// @include ledger
// @include campaign

const SEAMS = 'agents-architetura/docs/VERIFICATION.md'
const HARNESS = [
  { id: 'oracle', scope: ['tests/oracle/'], task: `the correctness oracle from the design's oracle section (contract: ${SEAMS}, "Oracle and identity"):
the synchronous reference mode of each training recipe and a byte-exact comparator (CLAUDE.md, "Product rule": one oracle class,
exact — no tolerance, no allclose, no confidence interval). The training state — weights, optimizer state, RNG state, the loss at
every step, checkpoints — must be byte-identical (sha256) to the reference with every feature on. When $FACTORY_IDENTITY_OUT is set,
runs write that state under $FACTORY_IDENTITY_OUT/state/ and wire captures under $FACTORY_IDENTITY_OUT/wire/. They honor
FACTORY_FEATURES=off (feature-off path) and FACTORY_REFERENCE=1 (reference mode). Add an INTEGRATE entry for
agents-architetura/tools/verify/identity.json: corpus = these tests; a/b commands; exact.cmd = the production path with every feature on;
exact.state = ["state/**"]. Collected by the "oracle" and "identity" rungs.` },
  { id: 'chaos', scope: ['tests/chaos/'], task: `the deterministic simulation substrate and fault injection (contract: ${SEAMS}, "Deterministic
simulation" and "Perturbation"): tests/chaos/sim/ — virtual clock, seeded network model (latency distribution, jitter, bandwidth
as serialization delay, loss, duplication, reordering, scheduled partitions, crash/restart, straggler factor), the fault schedule as
data (so failing schedules can be minimized), invariant-checker hooks and a canonical trace hash; same FACTORY_SEED => same trace.
Faults as reusable fixtures: kill/restart a worker mid-step, delay/drop/reorder messages, partition the coordinator, throttle a
straggler. Product hooks (perturb(site) reading FACTORY_SEED / FACTORY_PERTURB_PERIOD / FACTORY_PERTURB_MAX_US, the injectable
Transport/Clock/Rng) go into INTEGRATE entries, as do agents-architetura/tools/verify/matrix.json modes (the design's fault and config matrix) and the
amplify/load/sanitizer campaigns in agents-architetura/tools/campaign/campaigns.json. Collected by the "chaos" and "matrix" rungs.` },
  { id: 'fuzz', scope: ['tests/fuzz/'], task: `fuzzing of every trust boundary the design lists (contract: ${SEAMS}, "Fuzzing"): property-based
tests (Hypothesis, with a fast "ci" profile in tests/fuzz/conftest.py loaded from HYPOTHESIS_PROFILE) and coverage-guided targets
under tests/fuzz/targets/ (atheris for Python decoders, cargo-fuzz for Rust) with a --self-test that proves the target reaches the
decoder, a single-input replay, and tests/fuzz/regressions/<campaign>/ for minimized crashes; plus the scenario fuzzer whose
generators emit cluster-event programs for the simulator. Register each as a "fuzz" campaign in agents-architetura/tools/campaign/campaigns.json
through an INTEGRATE entry; add fuzzing libraries the same way. Collected by the "fuzz" rung.` },
  { id: 'distributed', scope: ['tests/multiworker/', 'tests/wan/', 'tests/multinode/'], task: `the distributed test matrix (contract: ${SEAMS}): tests/multiworker/
(1, 2 and 4 workers; heterogeneous workers; a slow worker; a failed worker; a worker restart; a coordinator failure; a partial
checkpoint — on one machine, real processes), tests/wan/ (high latency, packet loss, bandwidth limitation, network interruption —
through the deterministic simulator and, where the stack allows, a real emulated link such as toxiproxy or tc netem), and
tests/multinode/ (the same scenarios across machines listed in $FACTORY_MULTINODE_HOSTS; reported N/A without them). Every
scenario asserts BOTH that the system worked AND that the training result is byte-identical (sha256 of the training state) to the
synchronous datacenter reference — failures, restarts, slow links and heterogeneous workers included; there is no tolerance.
Collected by the "multiworker", "wan" and "multinode" rungs.` },
  { id: 'bench', scope: ['bench/'], task: `the design's performance budget as bench/suite.json + bench scripts (each prints {"value": n} as
its last stdout line; higherIsBetter where relevant; budgetPct from the design; runs 9, warmup 1) and bench/scaling.json for the
design's scaling claims (weak or strong mode, speedup floors, the serial tax bound of the feature). Do NOT create
bench/baseline.json — closeout records it on a real build.` },
]
const harnessScope = h => [...h.scope, `${DESIGN_DIR}/INTEGRATE-harness-${h.id}.md`]
// HARNESS scopes must stay inside HARNESS_DIRS (lib/project.js), the directories wan-design keeps workstreams out of.

const SPEC_RULES = key => `
SPEC RULES:
- ${specPath(key)} is HARD-CAPPED at ${SPEC_CAP} bytes (wc -c after every save). Overflow — review logs,
  refutations, worked examples — goes to ${specHistoryPath(key)}. Never cut normative content to fit.
- Ground every claim in the design (cite its section) and, where code exists, in the tree (file:line you READ).
- Number the invariants ${key.toUpperCase()}-I1, -I2, ...: each testable, each traced to a design INV-* and to a ladder rung.
- Exact public interface: names, signatures, error behavior, concurrency/async model, wire formats with versioning.
- Deviations from the design go in a "Deviations" section with rationale — never silently.
- Shared hot files (${HOT_FILES.join(', ')}) are never edited directly: write paste-ready entries
  (file, insertion point, exact text) to ${DESIGN_DIR}/INTEGRATE-${key}.md.
- End with an ordered task list, each task sized for one implementation agent, each with: objective, dependencies,
  file scope, acceptance criteria and the exact verification commands that prove them.`
const ownSpec = key => `OWNED PATHS: ${specPath(key)}, ${specHistoryPath(key)}, ${DESIGN_DIR}/INTEGRATE-${key}.md (only files you may write).`
const SPEC_LENSES = [
  ['soundness', `Verify every file/section/symbol citation by reading it. Check the mechanisms against the design: ordering, fences,
idempotence, failure handling. Any deviation from the design not listed under Deviations is a BLOCKER.`],
  ['implementability', `Could ONE agent implement each task without redesigning? Hunt hand-waved steps, missing data layouts,
invariants that are not numbered or not testable, interface entries without exact signatures, a task list that skips
work the scope promises, and edits outside the owned paths without an INTEGRATE entry.`],
  ['contracts', `Read the other docs/design/SPEC-*.md that exist so far (and the design's workstream table). Hunt interfaces this
spec consumes that nobody provides, name/signature drift between provider and consumer, ownership overlaps, and
conflicting configuration names.`],
]
const SYSTEM_LENSES = [
  ['cohesion', `Read ALL specs as one system. Hunt: consumed-but-unprovided interfaces, two specs owning one responsibility, lock or
ordering rules that are fine alone but cyclic when composed, recovery protocols that disagree about who does what,
and INTEGRATE entries that collide.`],
  ['end-to-end', `Walk the design's core scenarios across ALL specs: one full training step with every message; a worker dying
mid-step; a straggler; a coordinator partition; (if in scope) an adversarial update. Every step needs an owner and
the hand-offs must compose. A step with no owner is a BLOCKER.`],
  ['performance', `Audit the composed design against its performance budget: added round trips, serialization on hot paths,
synchronization points that scale with worker count, memory per worker. Cite the spec sections that compose badly.`],
]

phase('Gate')
await requireApprovedDesign('Gate')
const workstreams = await loadWorkstreams('Gate')
const ownership = validatePlan([
  ...workstreams.map(w => ({ id: `ws-${w.key}`, deps: [], scope: w.ownedPaths })),
  ...HARNESS.map(h => ({ id: `harness-${h.id}`, deps: [], scope: harnessScope(h) })),
])
if (!ownership.ok) throw new Error(`workstream ownership collides with the harness directories — revise the design: ${ownership.errors.join('; ')}`)
log(`Gate: design approved; ${workstreams.length} workstreams (${workstreams.map(w => w.key).join(', ')})`)

phase('Specs')
const specs = await pipeline(
  workstreams,
  w => agent(common(ownSpec(w.key)) + SPEC_RULES(w.key) + `
Write the FROZEN implementation spec for workstream "${w.key}" to ${specPath(w.key)}.
Workstream entry from the design (data): ${fence('workstream', w, 4000)}
Read ${DESIGN} fully first. Five implementation agents will later follow the specs concurrently without talking to
each other: everything they must agree on has to be in writing here.`,
    { label: `spec:${w.key}`, phase: 'Specs', schema: RESULT, agentType: ROLE.researcher }),
  async (draft, w) => {
    if (!draft) throw new Error(`spec draft for ${w.key} failed`)
    await sizeGate(specPath(w.key), specHistoryPath(w.key), SPEC_CAP, `sizegate:${w.key}:draft`, 'Specs')
    const review = await reviewUntilClean({
      label: `SPEC-${w.key}`,
      lenses: SPEC_LENSES,
            review: (name, lens, round) => agent(common('READ-ONLY adversarial reviewer.') + `
Review ${specPath(w.key)} — a FROZEN-candidate spec that implementation agents will follow verbatim, concurrently,
without coordinating. Assume it is wrong until the document and the tree prove otherwise. LENS ${name}: ${lens}
${round > 1 ? `ROUND ${round}: it was revised after earlier findings — review the CURRENT text from scratch.` : ''}
Severity: blocker = following it produces broken or unsound code; major = forces a mid-flight redesign;
minor = everything else. No style nits.`,
        { label: `spec-review:${w.key}:${name}:r${round}`, phase: 'Specs', schema: FINDINGS, agentType: ROLE.reviewer }),
      fix: async (serious, round) => {
        await agent(common(ownSpec(w.key)) + SPEC_RULES(w.key) + `
Adversarial round ${round} filed these findings against ${specPath(w.key)} (blocker/major first; a minor is cheap, fix it unless it conflicts). For each: verify it against the
design and the tree; if real, revise; if false, add a ONE-LINE refutation in the Deviations/Notes section (full
argument in the history file). The spec is re-reviewed from scratch.
${fence('reviewer_findings', serious, 30000)}`,
          { label: `spec-fix:${w.key}:r${round}`, phase: 'Specs', schema: RESULT, agentType: ROLE.researcher })
        await sizeGate(specPath(w.key), specHistoryPath(w.key), SPEC_CAP, `sizegate:${w.key}:r${round}`, 'Specs')
      },
    })
    return { key: w.key, ...review }
  },
)
const failedSpecs = workstreams.filter((w, i) => !specs[i]).map(w => w.key)
if (failedSpecs.length) throw new Error(`spec pipeline failed for: ${failedSpecs.join(', ')} — resume this run`)

phase('System')
const allSpecs = workstreams.map(w => specPath(w.key)).join(', ')
const system = await reviewUntilClean({
  label: 'system (all specs)',
  lenses: SYSTEM_LENSES,
    review: (name, lens, round) => agent(common('READ-ONLY adversarial reviewer.') + `
You review the COMPLETE design: ${DESIGN} plus every spec (${allSpecs}) as ONE system. Each spec passed review alone;
your job is the composition — assume it is wrong until the documents prove otherwise. LENS ${name}: ${lens}
Round ${round}${round > 1 ? ' — the specs were revised; review the CURRENT documents from scratch' : ''}.
Tag each finding's title with the spec it belongs to (or "cross-cutting").`,
    { label: `system-review:${name}:r${round}`, phase: 'System', schema: FINDINGS, agentType: ROLE.reviewer }),
  fix: async (serious, round) => {
    await agent(common(`OWNED PATHS: every ${DESIGN_DIR}/SPEC-*.md, SPEC-*-history.md and INTEGRATE-<workstream>.md (you run alone for this revision).`) + `
System review round ${round} filed these findings. For each: verify against the design, the specs and the tree; if real,
revise the affected specs and keep provider and every consumer consistent; if false, refute in one line in the
relevant spec's Deviations/Notes (full argument in its history file). Keep each spec under ${SPEC_CAP} bytes.
${fence('system_findings', serious, 30000)}`,
      { label: `system-fix:r${round}`, phase: 'System', schema: RESULT, agentType: ROLE.researcher })
    await parallel(workstreams.map(w => () => sizeGate(specPath(w.key), specHistoryPath(w.key), SPEC_CAP, `sizegate:${w.key}:system-r${round}`, 'System')))
  },
})

phase('Freeze')
let specsFrozen = false
if (system.converged && specs.every(sp => sp.converged)) {
  const docs = workstreams.map(w => specPath(w.key))
  const frozen = await runTool(`for f in ${docs.join(' ')}; do node agents-architetura/tools/factory/freeze.mjs freeze "$f" 2>/dev/null || node agents-architetura/tools/factory/freeze.mjs amend "$f" --reason "re-specified by wan-prep after a design revision" --by wan-prep || exit 1; done`,
    { label: 'freeze', phaseTitle: 'Freeze' })
  specsFrozen = frozen?.exitCode === 0
  if (!specsFrozen) log('Freeze: the specs could not be frozen — see the freeze output')
} else log('Freeze: skipped — specs or the system review did not converge')

phase('Harness')
const harness = (await parallel(HARNESS.map(h => () =>
  agent(common(`OWNED PATHS: ${harnessScope(h).join(', ')} (only files you may write).`) + `
Write ${h.task}
Follow the stack, layout and test conventions the design and specs prescribe. Product code is read-only for you:
if the harness needs a hook in product code or a change to a shared hot file (dependencies, ladder rungs), write it
as a paste-ready entry in ${DESIGN_DIR}/INTEGRATE-harness-${h.id}.md. Write code only — no builds, no runs.
If the design and specs make this harness genuinely inapplicable (e.g. no network or failure surface for chaos), write
nothing and state why in your summary — the ladder then reports the rung as N/A with that reason; never write a token test.`,
    { label: `harness:${h.id}`, phase: 'Harness', schema: RESULT, agentType: ROLE.coder })))).filter(Boolean)
if (harness.length < HARNESS.length) throw new Error(`harness incomplete (${harness.length}/${HARNESS.length}) — resume this run`)

phase('Skeleton')
const skeleton = await agent(common(`You run ALONE — you MAY build and run anything (no git). OWNED PATHS: the shared hot files
(${HOT_FILES.join(', ')}), package/crate roots inside each workstream's owned paths, and ${DESIGN_DIR}/INTEGRATE-*.md.`) + `
1. Create the project skeleton exactly as the design and specs prescribe: build configuration, one package/crate root per
   workstream inside its owned paths (empty modules — no fake implementations), and test-runner configuration so the
   ladder's rungs collect the test directories that exist. Do NOT create empty harness directories (tests/oracle,
   tests/chaos, tests/fuzz, tests/redteam, bench) — an absent harness is reported as N/A; an empty one is noise.
2. Merge EVERY ${DESIGN_DIR}/INTEGRATE-*.md entry; mark each applied entry with a line "Applied by wan-prep skeleton".
   Conflicting entries: apply the one consistent with the design and record the decision in that manifest.
3. Run ${ladderCommand({ upto: 'types' })} and fix configuration/scaffolding until build, fmt, lint and types pass.
Report honestly what does not pass.`,
  { label: 'skeleton', phase: 'Skeleton', schema: TASK_RESULT, agentType: ROLE.coder })
if (!skeleton) throw new Error('skeleton builder produced nothing')

phase('Tests')
const testRoots = w => w.ownedPaths.filter(p => p.startsWith('tests/'))
const testable = workstreams.filter(w => testRoots(w).length)
for (const w of workstreams.filter(w => !testRoots(w).length)) log(`Tests: workstream ${w.key} owns no tests/ path — its invariants must be tested by its implementation tasks`)
const tests = (await parallel(testable.map(w => () =>
  agent(common(`OWNED PATHS: ${testRoots(w).join(', ')} (only files you may write).`) + `
EXCEPTION to the slow-command rule: you MAY run your OWN test directories with the project's test runner without
syncing or building (e.g. uv run --no-sync pytest -q <dir>). Never the build.
Write executable tests for EVERY numbered invariant in ${specPath(w.key)} (${w.key.toUpperCase()}-I*). Each test carries a
"covers: <invariant id>" comment and a "WOULD-FAIL-IF: <the regression it catches>" comment (the invariants and claims rungs check
both). The implementation does not exist yet, so mark each test as expected-to-fail with the stack's strict
native marker whose reason contains exactly "${AWAITING}: <invariant id>" (pytest: xfail(strict=True, reason=...);
Rust: #[ignore = "..."]). A test must fail only because the implementation is missing — never because the test is
wrong. The task that delivers an invariant removes its marker.`,
    { label: `tests:${w.key}`, phase: 'Tests', schema: RESULT, agentType: ROLE.coder })))).filter(Boolean)

phase('Verify')
const verifyPrompt = common('You run ALONE — you MAY build and run anything (no git).') + `
Run: ${ladderCommand({ upto: 'unit' })}
Put the JSON it prints into "ladder" EXACTLY as printed. For every failing rung, record items (rung, exact evidence,
minimal repo-relative scope). allGreen=true only if the pasted ladder says so. Fix nothing.`
let verify = await agent(verifyPrompt, { label: 'verify', phase: 'Verify', schema: TRIAGE, agentType: ROLE.verifier })
let fixed = null
if (!ladderGreen(verify)) {
  log(`Verify: ladder red (${failingRungs(verify?.ladder).join(', ') || 'unknown'}) -> wan-fix`)
  fixed = await workflow('wan-fix', { rungs: failingRungs(verify?.ladder), report: verify })
}
const green = ladderGreen(verify) || fixed?.ok === true

return {
  ok: green && system.converged && specs.every(s => s.converged) && specsFrozen,
  specsFrozen,
  specs: specs.map(s => ({ key: s.key, converged: s.converged, findingsPerRound: s.findingsPerRound })),
  system,
  harness: harness.length,
  tests: tests.length,
  verifyGreen: green,
  next: 'Run wan-handout with {"milestone": "M0"}.',
}
