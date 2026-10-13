export const meta = {
  name: 'wan-corpus',
  description: 'Grow the stress and semantics corpus the way the reference PR grew its thread corpus, with the would-it-catch check made mechanical: parallel writers per suite in disjoint staging directories (scope-audited) — every test states covers: <INV> and WOULD-FAIL-IF: <regression> and ships that regression as <test>.would-fail.patch -> per suite, would-it-catch and discipline reviewers loop with a reviser -> a solo validator proves each test passes 20/20, survives 50 amplified seeds, runs under the time budget, and FAILS when its own regression patch is applied (claims.mjs --verify) -> accepted tests are promoted, rejected ones are reported and removed.',
  whenToUse: 'After wan-implement of a milestone, before the red team; or when a bug escaped. args: {"suites": ["semantics", "chaos", "scaling", "fault-matrix"], "testCmd": "uv run pytest -q -p no:randomly {test}", "maxSeconds": 30}.',
  phases: [
    { title: 'Gate', detail: 'Design approved' },
    { title: 'Author', detail: 'One writer per suite in tests/<suite>/staging/, scope-audited' },
    { title: 'Review', detail: 'Per suite: would-it-catch + discipline reviewers -> reviser, <= 5 rounds' },
    { title: 'Validate', detail: 'Solo: claim verified by patch, 20/20, 50 amplified seeds, time budget' },
    { title: 'Promote', detail: 'Accepted tests move into the suite; staging removed (scope-audited)' },
    { title: 'Report', detail: 'Researcher: agents-architetura/docs/corpus/REPORT.md' },
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

const SUITE_RE = /^[a-z][a-z0-9-]{0,30}$/
const suites = (Array.isArray(args?.suites) && args.suites.length ? args.suites : ['semantics', 'chaos', 'scaling', 'fault-matrix']).filter(s => SUITE_RE.test(s) && s !== 'unit')
if (!suites.length) throw new Error('no valid suite names')
const testCmd = typeof args?.testCmd === 'string' && args.testCmd.includes('{test}') && !/[\x00-\x1f]/.test(args.testCmd) ? args.testCmd : 'uv run pytest -q -p no:randomly {test}'
const MAX_SECONDS = Number.isInteger(args?.maxSeconds) && args.maxSeconds > 0 && args.maxSeconds <= 600 ? args.maxSeconds : 30
const staging = s => `tests/${s}/staging/`
const VALIDATE = {
  type: 'object',
  required: ['results'],
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        required: ['test', 'claim', 'repeatRuns', 'repeatPassed', 'amplifyClean', 'seconds'],
        properties: {
          test: { type: 'string', description: 'repo-relative path of the staged test file' },
          claim: { type: 'string', description: 'the "verdict" printed by claims.mjs --verify, verbatim' },
          repeatRuns: { type: 'number' },
          repeatPassed: { type: 'number' },
          amplifyClean: { type: 'boolean', description: 'the amplify run printed clean: true' },
          seconds: { type: 'number', description: 'wall time of one clean run' },
          evidence: { type: 'string' },
        },
      },
    },
  },
}

phase('Gate')
await requireApprovedDesign('Gate')

phase('Author')
const since = await scopeSnapshot('corpus-author', 'Author')
await parallel(suites.map(s => () =>
  agent(common(`OWNED PATHS: ${staging(s)} (only). Product code is READ-ONLY for you. Write code only — no runs.`) + `
Write new tests for the "${s}" suite into ${staging(s)}. Read ${DESIGN}, the specs (their numbered invariants) and agents-architetura/docs/LESSONS.md;
aim each test at a behavior a real regression could break, especially concurrency, ordering, failure and recovery paths.
EVERY test file:
- carries "covers: <invariant id>" and "WOULD-FAIL-IF: <the precise regression this test catches>" comments;
- ships that regression as <test file>.would-fail.patch: a unified diff (git apply format) of PRODUCT code that introduces exactly
  the regression — the validator applies it to a copy of the tree and the test must then FAIL as a test failure (not a crash, not
  a syntax error);
- is deterministic (seeded; fixed iteration counts; sorted output; no wall-clock luck), bounded (< ${MAX_SECONDS}s), cleans up what it
  starts, and is amplifier-ready (honors FACTORY_SEED through the project's fixtures where randomness matters).`,
    { label: `author:${s}`, phase: 'Author', schema: RESULT, agentType: ROLE.coder })))
await scopeCheck('corpus-author', 'Author', since, suites.map(staging))

phase('Review')
const LENSES = [
  ['would-it-catch', `For EVERY staged test: does the WOULD-FAIL-IF claim hold? Walk the test logic against its .would-fail.patch and argue
the test actually trips when that regression exists. A race or fault test that passes vacuously when the bad interleaving
happens is the classic failure — check the assertions observe the state the regression corrupts.`],
  ['discipline', `Test discipline: deterministic (seeded, no timing luck, sorted output), bounded runtime, cleanup of every worker or
process it starts, actionable failure messages, covers: ids that exist in the specs, patches that touch product code only.`],
]
const reviews = await parallel(suites.map(s => () =>
  reviewUntilClean({
    label: `corpus:${s}`,
    lenses: LENSES,
        review: (name, lens, round) => agent(common('READ-ONLY adversarial reviewer.') + `
Review the staged "${s}" suite in ${staging(s)} (round ${round}, lens ${name}). ${lens}`,
      { label: `review:${s}:${name}:r${round}`, phase: 'Review', schema: FINDINGS, agentType: ROLE.reviewer }),
    fix: (serious, round) => agent(common(`OWNED PATHS: ${staging(s)} (only). Write code only — no runs.`) + `
Revise the staged "${s}" suite for these review findings (round ${round}). Verify each against the files; fix the real ones,
refute the rest in your summary with file:line evidence.
${fence('reviewer_findings', serious, 20000)}`,
      { label: `revise:${s}:r${round}`, phase: 'Review', schema: RESULT, agentType: ROLE.coder }),
  })))

phase('Validate')
const validated = []
for (const s of suites) {
  const v = await agent(common('You run ALONE — you MAY run commands (no git). Change nothing.') + `
Validate every staged test file in ${staging(s)}, ONE AT A TIME (nice -n 10). Test command: ${testCmd}
Per test file T:
1. node agents-architetura/tools/gates/claims.mjs --verify T --cmd "${testCmd}" — copy its "verdict" into "claim" verbatim.
2. Run the test command 20 times; report repeatRuns and repeatPassed.
3. node agents-architetura/tools/verify/amplify.mjs --runs 50 --seed-base 1 --json -- <the test command for T> — amplifyClean = its "clean".
4. seconds = wall time of one clean run.
Report every staged test file; paste failing output as evidence.`,
    { label: `validate:${s}`, phase: 'Validate', schema: VALIDATE, agentType: ROLE.verifier })
  for (const r of v?.results ?? []) {
    const test = String(r.test ?? '')
    const inStaging = test.startsWith(staging(s)) && safeScopePath(test)
    const why = !inStaging ? 'not a staged test path'
      : r.claim !== 'catches' ? `claim ${clean(r.claim, 40)} (the test does not fail with its own regression applied)`
        : !(r.repeatRuns >= 20 && r.repeatPassed === r.repeatRuns) ? `${r.repeatPassed}/${r.repeatRuns} clean runs — a flaky pass is not a pass`
          : r.amplifyClean !== true ? 'fails under the amplifier'
            : !(r.seconds <= MAX_SECONDS) ? `${r.seconds}s exceeds the ${MAX_SECONDS}s budget` : null
    validated.push({ suite: s, test, accepted: !why, why, evidence: clean(r.evidence, 600) })
  }
}
const accepted = validated.filter(v => v.accepted)
log(`Validate: ${accepted.length}/${validated.length} staged tests accepted`)

phase('Promote')
const promoteScopes = suites.map(s => `tests/${s}/`)
const psince = await scopeSnapshot('corpus-promote', 'Promote')
await agent(common(`You run ALONE. OWNED PATHS: ${promoteScopes.join(', ')} (only).`) + `
Promote the ACCEPTED tests: move each accepted test file and its .would-fail.patch from tests/<suite>/staging/ into tests/<suite>/
(fix relative imports if the move breaks them). Then delete every staging directory entirely — rejected tests are not kept.
${fence('accepted', accepted.map(a => a.test), 8000)}`,
  { label: 'promote', phase: 'Promote', schema: RESULT, agentType: ROLE.coder })
await scopeCheck('corpus-promote', 'Promote', psince, promoteScopes)

phase('Report')
await agent(common('OWNED PATHS: agents-architetura/docs/corpus/REPORT.md (only).') + `
Write agents-architetura/docs/corpus/REPORT.md: per suite, tests accepted (with the regression each catches) and tests rejected with the
reason; review rounds per suite. Rejections are findings about our tests, not noise — list every one.
${fence('corpus', { reviews: suites.map((s, i) => ({ suite: s, converged: reviews[i]?.converged ?? false, rounds: reviews[i]?.rounds ?? null })), validated }, 20000)}`,
  { label: 'report', phase: 'Report', schema: RESULT, agentType: ROLE.researcher })

const converged = reviews.every(r => r?.converged)
// Rejected tests are removed and reported (they are findings about our tests); the corpus grew if anything was accepted.
return { ok: converged && accepted.length > 0, suites, accepted: accepted.length, rejected: validated.filter(v => !v.accepted), reviewsConverged: converged }
