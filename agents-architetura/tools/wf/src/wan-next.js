export const meta = {
  name: 'wan-next',
  description: 'Writes the workflow for the next phase from the real state, the way the reference PR\'s author wrote thread-bughunter2..4 and thread-ab17b..e after reading each run instead of re-running one generic script: a state snapshot (ladder, invariants gate, open findings, lessons, the last run) -> ONE objective with evidence in docs/design/NEXT-<slug>.md and its pinned verify (V0..Vn) in docs/design/GATES-<slug>.json -> a phase workflow agents-architetura/tools/wf/src/wan-x-<slug>.js that carries the exact facts (files, numbers, invariant ids) into its prompts -> wf-lint (static rules, build, simulation) -> adversarial review until clean -> build -> STOP for the human grant of the NEXT document.',
  whenToUse: 'After a milestone, a campaign or a run that stalled, when the next step needs prompts that know the current state. Optional args: {focus: "<one-line hint from the human>"}; {review: "<slug>"} resumes a phase whose review stopped at its round limit (Check, Review and Build only, on the existing files). Running the written workflow needs the human grant of its NEXT document.',
  phases: [
    { title: 'State', detail: 'Solo verifier runs the ladder and the invariants gate; read-only collectors gather findings, lessons, the last run and the deferred work' },
    { title: 'Objective', detail: 'Researcher picks ONE objective from the evidence and writes docs/design/NEXT-<slug>.md and GATES-<slug>.json' },
    { title: 'Author', detail: 'Coder writes the phase workflow from the house template, with the facts embedded verbatim' },
    { title: 'Check', detail: 'Solo: wf-lint (static rules, build, simulation); fix and re-check, at most 3 rounds' },
    { title: 'Review', detail: 'Three lenses (fit to the state, safety, verification strength) until zero blocker/major, <= 5 rounds' },
    { title: 'Build', detail: 'Solo: rebuild .claude/workflows and re-lint the final source' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project
// @include gates

const TEMPLATE = 'agents-architetura/docs/templates/wan-x-workflow.js'
const SLUG_RE = /^[a-z0-9][a-z0-9-]{1,38}$/
const focus = typeof args?.focus === 'string' ? clean(args.focus, 300) : ''
// Resume a phase whose review stopped at its round limit: skip State, Objective and Author, and run Check, Review and Build
// again on the phase's existing NEXT, GATES and workflow source — the reference PR iterated on its own phase workflows.
const REVIEW = typeof args?.review === 'string' ? args.review : null
if (REVIEW !== null && !SLUG_RE.test(REVIEW)) throw new Error(`args.review must be an existing phase slug (got ${JSON.stringify(clean(REVIEW, 60))})`)

const STATE_RUN = {
  type: 'object',
  required: ['ladder', 'invariants'],
  properties: {
    ladder: LADDER_RESULT,
    invariants: { type: 'string', description: 'the exact output of the invariants gate, unmodified' },
  },
}
const FACTS = {
  type: 'object',
  required: ['facts'],
  properties: {
    facts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['fact', 'source'],
        properties: { fact: { type: 'string' }, source: { type: 'string', description: 'file:line, file path or the command whose output shows it' } },
      },
    },
  },
}
const OBJECTIVE = {
  type: 'object',
  required: ['slug', 'objective', 'evidence', 'ownedPaths', 'rungs', 'doc', 'gates'],
  properties: {
    slug: { type: 'string', description: 'lowercase words joined by "-", 2-39 chars; the workflow becomes wan-x-<slug>' },
    objective: { type: 'string' },
    evidence: { type: 'array', items: { type: 'string' }, description: 'the facts (with their sources) that make this the next phase' },
    ownedPaths: { type: 'array', items: { type: 'string' }, description: 'repo-relative paths the phase may write; directories end with "/"' },
    invariants: { type: 'array', items: { type: 'string' } },
    rungs: { type: 'array', items: { type: 'string' }, description: 'ladder rungs the exit gate requires' },
    doc: { type: 'string', description: 'the NEXT document you wrote, repo-relative (docs/design/NEXT-<slug>.md)' },
    gates: { type: 'string', description: 'the pinned gates file you wrote, repo-relative (docs/design/GATES-<slug>.json)' },
  },
}
const LINT = {
  type: 'object',
  description: 'the exact JSON printed by node agents-architetura/tools/wf/lint.mjs <name> --json',
  required: ['ok', 'errors'],
  properties: { ok: { type: 'boolean' }, errors: { type: 'array', items: { type: 'string' } } },
}

phase('State')
await requireApprovedDesign('State')
let ws = null
const pickObjective = async () => {
  const run = await agent(common('You run ALONE — you MAY run these two commands. Change nothing.') + `
1. Run: ${ladderCommand({ reuse: true })}
   Put the JSON it prints into "ladder" EXACTLY as printed. When nothing a product rung reads changed since the last run
   (only factory files did), it reuses that report and reruns only the factory rung; its "reused" field says so.
2. Run: node agents-architetura/tools/gates/invariants.mjs
   Put its output into "invariants" EXACTLY as printed (uncovered, deferred and covered counts).`,
    { label: 'state:run', phase: 'State', schema: STATE_RUN, agentType: ROLE.verifier })
  if (!run) throw new Error('the state run produced nothing — resume this run')
  const COLLECT = [
    ['findings', `Open findings in agents-architetura/docs/findings.json, the latest reports under agents-architetura/docs/
(campaigns, bughunt closures, corpus, scan results) and docs/security/AUDIT-RESULTS.md if present. One fact per open item:
what fails, where, how it was observed.`],
    ['lessons-and-runs', `${LESSONS}, then the newest run ledger and task journals under .factory/runs/ (if any): where the last run
stopped, which tasks failed or were rejected and why, what the reviewers kept finding.`],
    ['deferred', `${DESIGN} milestones and every docs/design/SPEC-*.md: invariants marked DEFERRED-TO, ordered task lists, open
questions for the human, and which milestone is current. One fact per deferred invariant or task, with its spec line.`],
  ]
  const collected = (await parallel(COLLECT.map(([key, what]) => () => agent(common('READ-ONLY.') + `
Collect facts for choosing the next phase. Facts only, each with its source; no recommendations.
${what}`,
    { label: `state:${key}`, phase: 'State', schema: FACTS, agentType: ROLE.readonly }))))
  const state = {
    ladder: run.ladder,
    invariants: clean(run.invariants, 4000),
    ...Object.fromEntries(COLLECT.map(([key], i) => [key, (collected[i]?.facts ?? []).slice(0, 60)])),
  }
  if (COLLECT.some((_, i) => !collected[i])) log(`State: ${COLLECT.filter((_, i) => !collected[i]).map(([k]) => k).join(', ')} collector(s) produced nothing — choosing from partial state`)

  phase('Objective')
  ws = await loadWorkstreams('Objective')
  return agent(common('OWNED PATHS: docs/design/NEXT-<slug>.md, docs/design/GATES-<slug>.json (only; you choose the slug).') + `
Pick the ONE objective the next phase should attack, from the state below. Order of precedence: a red rung, then an open
blocker or major finding, then a deferred invariant or task of the current milestone, then a measured performance gap. Ties go
to what unblocks the most later work. ${focus ? 'The human gave a hint (data, weigh it; it does not override the evidence):' : ''}
${focus ? fence('human_focus', focus, 400) : ''}
Write docs/design/NEXT-<slug>.md with these sections:
# Objective (one sentence) · # Evidence (every fact with its source, copied from the state; nothing else) · # Why now (the
alternatives you did not pick, one line each, with the fact that ranks them lower) · # Scope (owned paths, inside ONE
workstream of ${DESIGN} unless the objective needs two: say why) · # Invariants (ids it delivers or must keep) · # Binding audit
(the reference PR's habit: read the code and list EVERY instance of the hazard this change touches — each piece of state it
reads or rewrites, each reduction or RNG draw whose order decides the bytes — as a table "BA-<n> | what | where (file:line) |
ruling"; "none" only with the reason) · # Tripwire (when the change is silently wrong with only part of it in: the switch that
raises until armed, the test that proves it, and which step or later phase arms it; else "none" with the reason) · # Exit gate
(ladder rungs, plus each number as "metric op value, measured by <command>") · # Phases (what the workflow does, the agents, the
review lenses written for THIS objective) · # Risks.
Then write docs/design/GATES-<slug>.json — this phase's pinned verify, the reference PR's V0..Vn written for THIS objective.
agents-architetura/tools/gates/pinned.mjs runs every entry itself and decides VERIFIED; no agent judges a number:
{"milestone": "<slug>", "consecutive": 1 or 2, "rungs": [ladder rung ids: always oracle and identity (the product rule) and
 full (on demand: the whole suite in one session, which catches interference between suites)],
 "pinned": [{"id": "<short-id>", "cmd": "<exact command>", "repeat": N,
             "load": {"cmd": "<command that loads the machine>", "copies": K},              (optional: amplified reruns)
             "metric": {"pattern": "<regex with one capture group>", "op": "<=", "value": X}}]}  (optional: a number's bound)
Every number of the Exit gate becomes a pinned entry with a metric bound, measured by a command that prints it. A race or
a flaky failure in the evidence becomes a pinned entry with repeat (and load when it shows only under load), so a lucky
pass cannot close the phase. A tripwire this phase does not arm is a pinned entry running its test. A performance number
comes with discriminating arms: pinned entries for variants that each change ONE factor (compute, communication, pipeline
bubbles, the new mechanism's own cost), so the gain or gap is attributed, not just observed. Everything it names must exist
and run on this machine (read the hardware switch).
Workstreams (data): ${fence('workstreams', ws.map(w => ({ key: w.key, ownedPaths: w.ownedPaths })), 6000)}
State (data): ${fence('state', state, 30000)}`,
    { label: 'objective', phase: 'Objective', schema: OBJECTIVE, agentType: ROLE.researcher })
}
const reviewObjective = async () => {
  phase('Objective')
  ws = await loadWorkstreams('Objective')
  log(`wan-next: review mode for ${REVIEW} — State, Objective and Author are skipped; Check, Review and Build run on the existing files`)
  return agent(common('READ-ONLY.') + `
Read docs/design/NEXT-${REVIEW}.md and docs/design/GATES-${REVIEW}.json and return this phase's objective, verbatim from them:
slug "${REVIEW}"; objective (the Objective section); evidence (the Evidence rows, with their sources); ownedPaths (the owned
paths of the Scope section); invariants; rungs (the "rungs" of the GATES file); doc "docs/design/NEXT-${REVIEW}.md"; gates
"docs/design/GATES-${REVIEW}.json". Do not fix, complete or reinterpret anything.`,
    { label: 'extract:objective', phase: 'Objective', schema: OBJECTIVE, agentType: ROLE.readonly })
}
const chosen = REVIEW ? await reviewObjective() : await pickObjective()
if (!chosen) throw new Error('no objective was chosen — resume this run')
if (REVIEW && chosen.slug !== REVIEW) throw new Error(`docs/design/NEXT-${REVIEW}.md names another slug (${JSON.stringify(clean(String(chosen.slug ?? ''), 60))}) — fix the document`)
const slug = String(chosen.slug ?? '')
if (!SLUG_RE.test(slug)) throw new Error(`invalid slug ${JSON.stringify(clean(slug, 60))} — resume this run`)
const name = `wan-x-${slug}`
const nextDoc = `docs/design/NEXT-${slug}.md`
const gatesDoc = `docs/design/GATES-${slug}.json`
const source = `agents-architetura/tools/wf/src/${name}.js`
// Agents sometimes report the path absolutely (/.../<repo>/docs/design/NEXT-x.md). Accept that form for exactly this file and
// nothing else; whether the file really is in this repo is checked by wf-lint, which requires it to exist.
const sameFile = (reported, expected) => {
  const p = typeof reported === 'string' ? reported : ''
  return p === expected || p === `./${expected}` || (p.startsWith('/') && p.endsWith(`/${expected}`) && !p.split('/').includes('..'))
}
if (!sameFile(chosen.doc, nextDoc)) throw new Error(`the objective was written to ${JSON.stringify(clean(chosen.doc, 120))}, not ${nextDoc} — resume this run`)
if (!sameFile(chosen.gates, gatesDoc)) throw new Error(`the pinned gates were written to ${JSON.stringify(clean(chosen.gates, 120))}, not ${gatesDoc} — resume this run`)
// A phase writes product code only: inside a DESIGN workstream, never the factory, the harness or docs/design (the
// approvals record, frozen specs and the design of record live there).
const requested = Array.isArray(chosen.ownedPaths) ? chosen.ownedPaths.map(p => String(p)) : []
const NOT_OWNABLE = [...FACTORY_DIRS, ...HARNESS_DIRS, `${DESIGN_DIR}/`]
const insideWorkstream = p => ws.some(w => w.ownedPaths.some(o => p === o || (o.endsWith('/') && p.startsWith(o))))
const refused = requested.filter(p => !safeScopePath(p) || NOT_OWNABLE.some(d => scopesOverlap(p, d)) || !insideWorkstream(p))
if (!requested.length || refused.length) throw new Error(`the phase must own product paths inside a workstream only (refused: ${JSON.stringify(clean(JSON.stringify(refused.length ? refused : requested), 300))}) — resume this run`)
const owned = requested
const rungs = knownRungs(chosen.rungs)
if (!rungs.length) throw new Error(`the exit gate names no known ladder rung (${JSON.stringify(clean(JSON.stringify(chosen.rungs), 200))}) — a gate that requires nothing cannot fail; resume this run`)
log(`Objective: ${clean(chosen.objective, 200)} -> ${name} (owns ${owned.join(', ')}; gate ${rungs.join(', ') || 'default'})`)

phase('Author')
const AUTHOR_RULES = `
Write ${source} — a phase workflow for exactly this objective. Copy ${TEMPLATE} (read it, and the contract at the top of
agents-architetura/tools/wf/lib/phase.js) and change only its configuration. The machinery — the grant check first, scope
audits, review loop, pinned verify, stabilize with votes, post-stabilize review, intermittent and tree checks — is runPhase,
written and reviewed once. A phase workflow is:
- meta: a pure literal first; meta.name "${name}"; the template's five phases.
- the includes of the template, unchanged.
- FACTS: the facts from ${nextDoc} copied VERBATIM (paths, numbers with sources, invariant ids, failing test ids, every BA-<n>
  binding item with its ruling, the tripwire, the exit gate) — what the prompts need, not the whole document: every agent can
  read ${nextDoc}.
- \`return await runPhase({ slug: '${slug}', facts: FACTS, work, lenses, gateIds })\`, plus requires / precondition / unanimous
  only when the phase needs them:
  - work: writers inside ${JSON.stringify(owned)}, disjoint paths within a wave; a later wave builds on an earlier one; every
    BA-<n> assigned to the writer that delivers its ruling (binding). Writers never run commands.
  - lenses: at least three written for THIS objective (the objective with its numbers, soundness with the product rule, tests
    that would fail on a wrong implementation), plus one with { security: true } when the phase touches a trust boundary.
  - gateIds: exactly the gate ids pinned.mjs reports for ${gatesDoc}: "rung:<id>" for each rung, "pinned:<id>" for each pinned
    entry (and "campaign:<id>", "fuzz:<id>", "audit", "mutation" when the file has them).
  - requires: an ADR or other document the phase builds on, which a human must have granted.
  - precondition: { cmd, why } — one command checked before any writer (a baseline a causal gate needs saved, a spec text that
    must exist); any exit code but 0 stops the phase with why.
- No agent(), parallel(), loop, audit or approval call of your own, and at most 350 lines: wf-lint refuses both. If the phase
  needs something runPhase cannot do, say so in your summary (a human adds it to lib/phase.js); never re-implement it.
- Never: git, merges, force flags, approvals, .claude/ (it is generated).`
const authored = REVIEW ? { summary: 'review mode: the existing source is kept' } : await agent(common(`OWNED PATHS: ${source} (only).`) + AUTHOR_RULES + `
The objective (data): ${fence('objective', chosen, 12000)}`,
  { label: 'author', phase: 'Author', schema: RESULT, agentType: ROLE.coder })
if (!authored) throw new Error('the author produced nothing — resume this run')

phase('Check')
const lintCmd = `node agents-architetura/tools/wf/lint.mjs ${name} --json`
let lint = null
for (let round = 1; round <= 3; round++) {
  lint = await agent(common('You run ALONE — you MAY run this one command. Change nothing.') + `
Run: ${lintCmd}
Return the JSON it prints EXACTLY as printed.`,
    { label: `lint:r${round}`, phase: 'Check', schema: LINT, agentType: ROLE.verifier })
  if (lint?.ok === true && Array.isArray(lint.errors) && !lint.errors.length) break
  if (round === 3) break
  await agent(common(`OWNED PATHS: ${source}, ${gatesDoc} (only).`) + AUTHOR_RULES + `
wf-lint rejected the workflow (or its pinned gates ${gatesDoc}). Fix every violation at its cause (read the lint rules in agents-architetura/tools/wf/lint.mjs):
${fence('lint', lint, 8000)}`,
    { label: `lint-fix:r${round}`, phase: 'Check', schema: RESULT, agentType: ROLE.coder })
}
if (!(lint?.ok === true && !(lint.errors ?? []).length)) {
  log(`Check: ${name} still fails wf-lint after 3 rounds — needs human attention`)
  return { ok: false, stage: 'check', name, nextDoc, lint }
}

phase('Review')
const LENSES = [
  ['fit', `Does ${source} attack the objective of ${nextDoc} with the facts of the state — the exact files, numbers and
invariant ids — or is it a generic workflow with a new name? Would it make progress where the last run stalled (read
${LESSONS} and the run ledger)? Is the exit gate the one the NEXT document names, with its numbers?`],
  ['safety', `runPhase (agents-architetura/tools/wf/lib/phase.js) enforces the grant check first, the scope audits, the integrity
veto, the spec-conflict stop, the post-stabilize review and the intermittent and tree checks, and wf-lint passed: do not re-file
them. Review this phase's configuration: writers only inside ${JSON.stringify(owned)}, disjoint within a wave, waves in the right
order; the precondition (if any) changes no tracked file and stops for the right reason; requires names what the phase builds
on; unanimous covers the test files a fix could weaken; a security lens where a trust boundary is touched.`],
  ['verification', `Read ${gatesDoc}: is it this phase's real V0..Vn? Can it fail on a WRONG implementation of this objective — an
honest mistake such as the wrong ownership, the wrong order, missing telemetry or a fold that is not exact? Is every exit-gate
number of ${nextDoc} a pinned metric bound printed by a command, every race or flake of the evidence a repeated (and, when it
needs it, loaded) entry, and are oracle, identity and full required? Do not enumerate ways a malicious writer could game a regex
or a count: the house template stops writes outside the owned paths (scope audit), test weakening (the integrity veto) and
unreviewed stabilize edits (the post-stabilize review). File a gaming scenario only when those protections do not cover it.`],
]
// A defect in runPhase, the template or another factory file is "house": it goes to the factory backlog and does not block
// this phase, whose own files are the three below.
const PHASE_FILES = [source, nextDoc, gatesDoc]
const review = await reviewUntilClean({
  label: 'review',
  lenses: LENSES,
  ownsFile: f => PHASE_FILES.some(p => sameFile(f, p)),
  review: (lensName, lens, round) => agent(common('READ-ONLY.') + `
Adversarially review ${source} (and ${nextDoc}, ${gatesDoc}). LENS (${lensName}, round ${round}): ${lens}
Scope "house" is only for a defect in factory code this phase did not write (agents-architetura/tools/wf/lib/, the template,
pinned.mjs): it goes to the factory backlog. Anything in the three files above is "change".`,
    { label: `review:${lensName}:r${round}`, phase: 'Review', schema: FINDINGS, agentType: ROLE.reviewer }),
  fix: (findings, round) => agent(common(`OWNED PATHS: ${source}, ${nextDoc}, ${gatesDoc} (only).`) + AUTHOR_RULES + `
Fix these validated findings at their root cause, blocker and major first (verify each first; a minor is cheap: fix it too
unless it conflicts with a serious fix):
${fence('findings', findings, 12000)}`,
    { label: `review-fix:r${round}`, phase: 'Review', schema: RESULT, agentType: ROLE.coder }),
})

phase('Build')
const final = await agent(common('You run ALONE — you MAY run these two commands. Change nothing else.') + `
1. Run: node agents-architetura/tools/wf/build.mjs
2. Run: ${lintCmd}
Return the JSON the second command prints EXACTLY as printed.`,
  { label: 'build', phase: 'Build', schema: LINT, agentType: ROLE.verifier })
const ok = final?.ok === true && !(final.errors ?? []).length && review.converged
return {
  ok,
  name,
  nextDoc,
  gatesDoc,
  source,
  objective: clean(chosen.objective, 300),
  review,
  lint: final,
  // A review that did not converge still ends with the human: the open findings go with the documents to the grant
  // question (/wan-run asks), instead of a bare "no" (the reference PR moved on after 3 rounds and let the gate decide).
  ...(review.converged ? {} : { open: review.open, awaitingHuman: final?.ok === true }),
  // minors the last round left, and defects found in factory code: the human sees both; house goes to a factory PR
  minors: review.minors,
  house: review.house,
  next: ok
    ? `Human: read ${nextDoc}, ${gatesDoc} (the pinned verify that decides the phase) and ${source}, then approve all three with /wan-approve in Claude Code, or in your own terminal: node agents-architetura/tools/factory/approve.mjs grant ${nextDoc} ${gatesDoc} .claude/workflows/${name}.js; then run /${name}. Any later edit to any of them voids the grant. After it finishes, run /wan-next again.`
    : final?.ok === true && review.stalled
      ? `The review stalled (${review.findingsPerRound.join(' -> ')} blocker/major findings): the phase is too big. Split it: rerun wan-next with a focus on the first part of this objective (the mechanism the rest relies on). Its open findings are in "open".`
      : final?.ok === true && !review.converged
      ? `The review did not converge after ${review.rounds} rounds; its open findings are in "open". The human decides: grant ${nextDoc}, ${gatesDoc} and the workflow with those findings recorded, run wan-next {"review": "${slug}"} once more, split the phase, or stop.`
      : 'Not ready: see the review and lint results.',
}
