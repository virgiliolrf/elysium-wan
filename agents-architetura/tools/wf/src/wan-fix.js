export const meta = {
  name: 'wan-fix',
  description: 'Repair loop for a red verification ladder, where the agent that finds an error fixes it (the human\'s rule, 2026-10-09): one solo agent per round runs the ladder, finds the root cause of the first failing rung, fixes it in the product code, re-runs until it passes, and pastes the last ladder run. No proposer and no voters: the no-masking rule, the review of the whole change and the closeout\'s guarded gates hold the line. Up to 4 rounds by default, 8 for a milestone run\'s fail-fast ladder.',
  whenToUse: 'Whenever agents-architetura/tools/gates/ladder.mjs is red. Optional args: {rungs: ["unit", ...]} to restrict the ladder, {report: <ladder or previous round JSON>} as the starting evidence, {maxRounds: 1..8} (default 4), {failFast: true} to stop each ladder run at its first failing rung (a milestone run\'s fast ladder).',
  phases: [
    { title: 'Fix', detail: 'Solo: run the ladder, fix the first failing rung\'s root cause yourself, re-run until it passes, paste the last ladder run; loop while red' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include triage
// @include project
// @include gates
// @include fix

// The agent that runs the ladder is the one that fixes what it finds: it already holds the failure's evidence, so no second
// agent re-reads it, and no vote stands between the finding and the fix. One root cause per round (the reference PR's rule).
const MAX_ROUNDS = Number.isInteger(args?.maxRounds) && args.maxRounds >= 1 && args.maxRounds <= 8 ? args.maxRounds : 4
const rungFilter = knownRungs(args?.rungs)
const ladderCmd = ladderCommand({ only: rungFilter, failFast: args?.failFast === true })

let last = args?.report ?? { note: 'no report passed: establish ground truth yourself' }
const fixed = []
const deviations = []
let green = false
let round = 0
while (round < MAX_ROUNDS && !green) {
  round++
  phase('Fix')
  const r = await agent(common(`You run ALONE — you MAY build, run and edit product code (no git). Round ${round} of ${MAX_ROUNDS}.`) + `
You find the ladder's failures AND fix them yourself.
1. Run: ${ladderCmd}
   If it is green, paste its JSON into "ladder" verbatim, set allGreen, and stop.
2. Otherwise take the first failing rung. Find the root cause of its failures; when one mechanism explains most of them, fix
   that one this round and leave the rest for the next. Read the author's context first: for the failing file run
   node agents-architetura/tools/wf/ledger.mjs owner --path <file> and read the journal it names.
3. Fix it in the product code. ${NO_MASKING}
4. Re-run the failing rung's command until it passes or you stop making progress; then run exactly: ${ladderCmd}
   and paste THAT run's JSON into "ladder" verbatim.
Report each fix (mechanism, files, the command that now passes) and what is still red (rung, symptom, evidence). If the handout
and the SPEC disagree on a fix, follow the SPEC of record and say so in specConflict.
Previous round (evidence, not truth): ${fence('previous_round', last, 12000)}`,
    { label: `fix:r${round}`, phase: 'Fix', schema: FIX_ROUND, agentType: ROLE.coder })
  if (!r) throw new Error('the fix agent produced nothing — cannot continue')
  fixed.push(...(Array.isArray(r.fixed) ? r.fixed : []).map(f => ({ round, mechanism: clean(f.mechanism, 200), files: (f.files ?? []).map(x => clean(x, 200)).slice(0, 20) })))
  if (r.specConflict) deviations.push({ round, description: clean(r.specConflict.description, 800), implemented: clean(r.specConflict.proposedChange, 800) })
  if (ladderGreen(r)) { green = true; log(`Ladder green after ${round} round(s)`); break }
  if (r.allGreen) log(`Round ${round}: the agent claimed green but the pasted ladder JSON is red — treating as red`)
  last = { ladder: r.ladder, remaining: r.remaining, fixed: r.fixed }
  log(`Round ${round}: still red (${failingRungs(r.ladder).join(', ') || 'unknown'}); fixed ${(r.fixed ?? []).length} mechanism(s)`)
}

if (!green) log(`Stopped after ${round} round(s) without a green ladder — needs human attention`)
return { ok: green, green, rounds: round, fixed, deviations, rejected: [], lastReport: green ? null : last }
