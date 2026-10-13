export const meta = {
  name: 'wan-bughunt',
  description: 'Hypothesis-driven hunt for ONE bug that survived scoped fixes (closure is statistical AND causal: zero failures in enough amplified and loaded runs to bound the rate 10x below the observed one, rule of three; and, as in the reference PR, restoring the pre-fix files brings the failure back on demand while the fixed tree stays clean): solo evidence pack -> 6 read-only finders with distinct angles (one skeptic of the test itself) -> 2 refuters per hypothesis -> solo experimenter runs confirm/refute observations -> causal fix proposal -> 2 reviewers who must BOTH approve -> solo implement + repeated-repro verify (reverts on failure). Knowledge accumulates across rounds.',
  whenToUse: 'When a failure survives wan-fix rounds, is intermittent, or is a silent wrong answer. args: {"bug": "<what goes wrong>", "id": "<short-slug>", "repro": "<command, optional>", "runs": 50, "maxRuns": 5000}.',
  phases: [
    { title: 'Evidence', detail: 'Solo: reproduce, minimize, characterize; EVIDENCE.md' },
    { title: 'Hunt', detail: '6 finders -> 2 refuters per hypothesis' },
    { title: 'Experiment', detail: 'Solo: run each survivor\'s confirmIf/refuteIf and the refuters\' requests' },
    { title: 'Fix', detail: 'Causal fix proposal -> 2 reviewers, unanimous approval required' },
    { title: 'Verify', detail: 'Pre-fix files saved; coder applies; verifier: N = max(runs, 30/rate) runs split amplified + under load, + ladder, + causal A/B; coder reverts if not fixed' },
    { title: 'Closure', detail: 'Researcher: CLOSURE.md with root cause, counts, the statistical bound and the causal A/B (or INCONCLUSIVE)' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project
// @include gates
// @include campaign

if (typeof args?.bug !== 'string' || !args.bug.trim()) throw new Error('args.bug (a description of what goes wrong) is required')
const bugId = token(args.id ?? args.bug, 40)
const DIR = `agents-architetura/docs/bughunt/${bugId}`
const RUNS = Number.isInteger(args.runs) && args.runs >= 10 && args.runs <= 2000 ? args.runs : 50
const MAX_RUNS = Number.isInteger(args.maxRuns) && args.maxRuns >= 100 && args.maxRuns <= 100000 ? args.maxRuns : 5000
const MAX_ROUNDS = 4
const MAX_HYPOTHESES = 12
const solo = `You run ALONE — you MAY build, run and instrument anything (no git).`

const ANGLES = [
  ['ordering', 'message ordering and concurrency: reordered, duplicated, delayed or dropped messages; races between steps, epochs or workers'],
  ['numerics', 'numerical behavior: nondeterministic reduction order, precision/accumulation differences, overflow, NaN/Inf propagation'],
  ['state-lifecycle', 'state lifecycle: stale parameters or optimizer state, step/epoch bookkeeping, checkpoint/restore, worker join/leave'],
  ['resources', 'resources and time: deadlocks, backpressure, timeouts, memory pressure, file descriptors, thread/process exhaustion'],
  ['boundaries', 'boundaries: serialization and wire formats, version skew between processes, partial reads/writes, encoding'],
  ['test-or-oracle', 'the skeptic angle: is the TEST or ORACLE itself sound? Is the "wrong" result actually allowed by the design, or does the test assert something the design never promised? Decode the observed values'],
]

phase('Evidence')
const evidence = await agent(common(`${solo} OWNED PATHS: ${DIR}/ (only).`) + `
Build the EVIDENCE PACK for this bug — no fixing, no hypotheses beyond what the evidence forces.
Bug report: ${fence('bug_report', args.bug, 4000)}
Repro hint: ${fence('repro_hint', args.repro ?? '(none)', 1000)}
1. Reproduce it. Measure the failure rate over repeated runs; find the smallest command and input that still fails.
2. Characterize: exact observed vs expected values (decode them), dependence on load, worker count, seed, timing.
3. Record environment and commands so anyone can rerun them. Write all of it to ${DIR}/EVIDENCE.md.
reproduced=true only if you observed the failure yourself; reproCommand must be runnable as-is.`,
  { label: 'evidence', phase: 'Evidence', schema: EVIDENCE, agentType: ROLE.researcher })
if (!evidence?.reproduced) {
  log(`Evidence: bug not reproduced — nothing to hunt (${clean(evidence?.summary, 200)})`)
  return { ok: false, solved: false, reason: 'not reproduced', evidence }
}
const reproCommand = clean(evidence.reproCommand, 500)
const closure = closureRuns(evidence.rate, { min: RUNS, cap: MAX_RUNS })
// Runs per causal arm: enough to expect the reverted arm to fail ~3 times at the observed rate (rule of three, inverted).
const causalRuns = closureRuns(evidence.rate, { factor: 1, min: 15, cap: Math.min(MAX_RUNS, 2000) }).runs
const amplifyRuns = Math.ceil(closure.runs / 2)
const loadPerWorker = Math.ceil((closure.runs - amplifyRuns) / 6)
log(`Evidence: reproduced (${clean(evidence.rate, 40)}): ${reproCommand} — closure needs ${closure.needed} clean runs${closure.feasible ? '' : ` (over the ${MAX_RUNS} cap: a clean result will be INCONCLUSIVE)`}`)

const knowledge = { refuted: [], experiments: [], rejectedFixes: [] }
let solved = false
let inconclusive = false
let round = 0
let fixedBy = null
let verified = null

while (round < MAX_ROUNDS && !solved && !inconclusive) {
  round++
  phase('Hunt')
  const found = (await parallel(ANGLES.map(([key, angle]) => () =>
    agent(common('READ-ONLY finder.') + `
Round ${round}. Read ${DIR}/EVIDENCE.md and the code. Propose the causes of this bug from ONE angle: ${angle}.
Each hypothesis: a file:line-grounded mechanism, the exact sequence of events that produces the observed values,
a runnable confirmIf observation and a runnable refuteIf observation. Quality over quantity: a hypothesis that
cannot explain ALL the evidence (rate, load dependence, exact values) is noise. Zero hypotheses is a valid answer.
Already refuted (do not resubmit; build on them): ${fence('refuted', knowledge.refuted.slice(-20), 6000)}
Experiment results so far: ${fence('experiments', knowledge.experiments.slice(-20), 6000)}
Rejected fixes: ${fence('rejected_fixes', knowledge.rejectedFixes.slice(-10), 4000)}`,
      { label: `find:${key}:r${round}`, phase: 'Hunt', schema: HYPOTHESES, agentType: ROLE.reviewer })
      .then(r => (r?.hypotheses ?? []).map(h => ({ ...h, id: `${key}-${token(h.id, 24)}` })))
  ))).filter(Boolean).flat()
  const seenIds = new Set()
  const hypotheses = found.filter(h => !seenIds.has(h.id) && seenIds.add(h.id)).slice(0, MAX_HYPOTHESES)
  if (!hypotheses.length) { log(`Round ${round}: no hypotheses`); continue }

  const judged = await parallel(hypotheses.map(h => () =>
    parallel([0, 1].map(n => () =>
      agent(common('READ-ONLY refuter.') + `
Try to REFUTE this hypothesis about the bug in ${DIR}/EVIDENCE.md. It survives only if you genuinely cannot refute it.
Hypothesis: ${fence('hypothesis', h, 6000)}
Does it explain every evidence fact (rate, load dependence, exact values)? Is the code path real (read it)? Verdict
refuted (with the proof), survives, or confirmed (only with direct evidence). Request a discriminating experiment if one exists.`,
        { label: `refute:${h.id}:${n}:r${round}`, phase: 'Hunt', schema: VERDICT, agentType: ROLE.reviewer })))
      .then(vs => ({ h, verdicts: vs.filter(Boolean) }))))
  const live = judged.filter(Boolean)
  const survivors = live.filter(j => j.verdicts.length === 2 && j.verdicts.every(v => v.verdict !== 'refuted'))
  for (const j of live.filter(j => j.verdicts.some(v => v.verdict === 'refuted'))) {
    knowledge.refuted.push({ id: j.h.id, mechanism: clean(j.h.mechanism, 200), why: clean(j.verdicts.find(v => v.verdict === 'refuted')?.argument, 300) })
  }
  log(`Round ${round}: ${survivors.length}/${hypotheses.length} hypotheses survive refutation`)
  if (!survivors.length) continue

  phase('Experiment')
  const exp = await agent(common(`${solo} OWNED PATHS: ${DIR}/ and temporary instrumentation you remove before finishing.`) + `
EXPERIMENTER, round ${round}. For each surviving hypothesis, run its confirmIf and refuteIf observations and the refuters'
experiment requests (instrument with logging/assertions as needed — then REMOVE all instrumentation). Report per
hypothesis: confirms, refutes or inconclusive, with the observed output. Append the results to ${DIR}/EVIDENCE.md.
Repro: ${reproCommand}
${fence('survivors', survivors.map(s => ({ h: s.h, requests: s.verdicts.map(v => v.experimentRequest).filter(Boolean) })), 16000)}`,
    { label: `experiment:r${round}`, phase: 'Experiment', schema: EXPERIMENTS, agentType: ROLE.researcher })
  const results = exp?.results ?? []
  knowledge.experiments.push(...results.map(r => ({ round, hypothesisId: token(r.hypothesisId, 64), outcome: r.outcome, detail: clean(r.detail, 400) })))
  const outcomeOf = id => results.filter(r => r.hypothesisId === id).map(r => r.outcome)
  const confirmed = survivors.filter(s => outcomeOf(s.h.id).includes('confirms') && !outcomeOf(s.h.id).includes('refutes'))
  const pool = confirmed.length ? confirmed : survivors.filter(s => !outcomeOf(s.h.id).includes('refutes'))
  for (const s of survivors.filter(s => outcomeOf(s.h.id).includes('refutes'))) knowledge.refuted.push({ id: s.h.id, mechanism: clean(s.h.mechanism, 200), why: 'refuted by experiment' })
  if (!pool.length) { log(`Round ${round}: experiments refuted every survivor`); continue }
  const target = pool[0].h
  log(`Round ${round}: fixing ${target.id} (${confirmed.length ? 'experimentally confirmed' : 'unrefuted'})`)

  phase('Fix')
  const prop = await agent(common('READ-ONLY: propose, do not apply.') + `
Propose the fix for the cause below (${confirmed.length ? 'experimentally confirmed' : 'survived refutation and experiment'}).
Cause: ${fence('cause', target, 6000)}
Give exact old->new snippets, the list of files, and the causal argument (happensBefore) that makes the failure
IMPOSSIBLE, not merely rarer. A timing tweak, retry or sleep that narrows a window is not a fix.`,
    { label: `fixplan:r${round}`, phase: 'Fix', schema: CAUSAL_FIX, agentType: ROLE.proposer })
  const files = (prop?.files ?? []).filter(f => typeof f === 'string')
  if (!prop || !files.length || !files.every(safeScopePath)) {
    knowledge.rejectedFixes.push({ round, hypothesisId: target.id, why: 'no proposal or unsafe file list' })
    continue
  }
  const votes = await parallel(['closes-the-cause', 'breaks-nothing'].map(lens => () =>
    agent(common('READ-ONLY fix reviewer.') + `
Review this proposed fix (not applied) for ${target.id}. LENS ${lens}: ${lens === 'closes-the-cause'
      ? 'Does it make the failure impossible? Demand the happens-before/causal argument and check it against the code.'
      : 'What does it break: other tests, the design invariants, performance-sensitive paths, the oracle?'}
${fence('proposal_from_another_agent', prop, 10000)}`,
      { label: `fixvote:${lens}:r${round}`, phase: 'Fix', schema: VOTE, agentType: ROLE.reviewer })))
  const tally = tallyVotes(votes, 2)
  if (!tally.unanimous) {
    knowledge.rejectedFixes.push({ round, hypothesisId: target.id, fix: clean(prop.fix, 600), objections: votes.filter(v => v && !v.approve).map(v => clean(v.reasons, 300)) })
    log(`Round ${round}: fix rejected (${tally.approvals}/2) — back to the hunt with the objections`)
    continue
  }

  phase('Verify')
  // The causal A/B needs the files as they were before the fix: saved by a tool, not remembered by an agent.
  const prefix = `${DIR}/prefix-r${round}`
  await agent(common(solo) + `
Run exactly this one command and report its output in "summary"; change nothing else:
node agents-architetura/tools/verify/causal.mjs save --out ${prefix} -- ${files.join(' ')}`,
    { label: `causal:save:r${round}`, phase: 'Verify', schema: RESULT, agentType: ROLE.verifier })
  const applied = await agent(common(`OWNED PATHS (data): ${JSON.stringify(files)} and ${DIR}/.`) + `
Apply this approved fix exactly (with the reviewers' amendments). Run the repro once and report it as evidence; record what you
changed and why in ${DIR}/EVIDENCE.md. Do not judge whether the bug is fixed — an independent verifier does that.
${fence('approved_fix', { prop, amendments: votes.filter(Boolean).map(x => x.amendment).filter(Boolean) }, 12000)}`,
    { label: `apply:r${round}`, phase: 'Verify', schema: TASK_RESULT, agentType: ROLE.coder })
  const v = applied ? await agent(common(solo) + `
You are the independent verifier; you did not write this fix. The bar is statistical: ${closure.runs} clean runs in total.
1. node agents-architetura/tools/verify/amplify.mjs --runs ${amplifyRuns} --seed-base 1 --periods 16,64,256 --json -- ${reproCommand}
   Paste its JSON into "amplify" verbatim.
2. node agents-architetura/tools/verify/load.mjs --workers 6 --runs-per-worker ${loadPerWorker} --seed-base 100000 --json -- ${reproCommand}
   Paste its JSON into "load" verbatim.
3. Run ${ladderCommand({ upto: 'unit' })} and paste its JSON into "ladder" verbatim.
4. node agents-architetura/tools/verify/causal.mjs ab --prefix ${prefix} --runs ${causalRuns} --json -- ${reproCommand}
   Paste its JSON into "causal" verbatim. It restores the pre-fix files in a snapshot (the bug must come back) and reruns the
   fixed tree (it must stay clean); it never touches the live files.
runs/failures: the totals of both drivers. fixed=true only if there were 0 failures AND the ladder is green. Change nothing.`,
    { label: `verify:r${round}`, phase: 'Verify', schema: BUG_VERIFY, agentType: ROLE.verifier }) : null
  // The counts come from the drivers' own JSON, not from the agent's summary of it.
  const totalRuns = (v?.amplify?.total ?? 0) + (v?.load?.total ?? 0)
  const totalFailures = (v?.amplify?.findings?.length ?? 0) + (v?.load?.failed ?? 0)
  const clean0 = !!v && v.fixed === true && totalFailures === 0 && totalRuns >= closure.runs && v.ladder?.allGreen === true
  // Causal: decided from the tool's own counts — the reverted arm failed at least once, the fixed arm never did.
  const causal = (v?.causal?.reverted?.failures ?? 0) > 0 && v?.causal?.fixed?.failures === 0 && (v?.causal?.fixed?.runs ?? 0) >= causalRuns
  verified = v ? { runs: totalRuns, failures: totalFailures, needed: closure.needed, ladderGreen: v.ladder?.allGreen === true, causal, reverted: v?.causal?.reverted ?? null, fixedArm: v?.causal?.fixed ?? null } : null
  solved = clean0 && closure.feasible && causal
  inconclusive = clean0 && !solved
  if (clean0) fixedBy = { hypothesisId: target.id, files }
  else {
    knowledge.rejectedFixes.push({ round, hypothesisId: target.id, why: applied ? `verify failed: ${v?.failures ?? '?'}/${v?.runs ?? '?'} failures, ladder ${v?.ladder?.allGreen === true ? 'green' : 'red'}` : 'apply produced nothing' })
    if (applied) {
      await agent(common(`OWNED PATHS (data): ${JSON.stringify(files)} and ${DIR}/.`) + `
The fix you applied for ${target.id} did not survive independent verification. REVERT every change it made (restore the files
exactly as they were before the fix) and append the verifier's observations to ${DIR}/EVIDENCE.md.
${fence('verification', v, 6000)}`,
        { label: `revert:r${round}`, phase: 'Verify', schema: RESULT, agentType: ROLE.coder })
    }
  }
}

if (solved || inconclusive) {
  phase('Closure')
  await agent(common(`OWNED PATHS: ${DIR}/CLOSURE.md (only).`) + `
Write ${DIR}/CLOSURE.md — the auditable closure record: status line "${solved ? 'CLOSED' : 'INCONCLUSIVE'}"; the root cause in one sentence
(from ${DIR}/EVIDENCE.md); how many failures carried the signature; the fix (files, the causal argument); the post-fix counts and
the statistical bound they give (0 failures in N runs bounds the rate below 3/N at 95% confidence; the observed rate was
${clean(evidence.rate, 40)}); the causal A/B (pre-fix files restored: failures/runs; fixed tree: failures/runs) and what it proves;
the exact commands to rerun the verification.${solved ? '' : ` INCONCLUSIVE means ${!closure.feasible ? `the clean runs were not enough to bound the
rate 10x below the observed one (needed ${closure.needed}, cap ${MAX_RUNS})` : 'the fix is not shown to be the cause: restoring the pre-fix files did not bring the failure back'} — say what would discharge it.`}
${fence('closure', { rate: evidence.rate, verified, fixedBy, rounds: round }, 6000)}`,
    { label: 'closure', phase: 'Closure', schema: RESULT, agentType: ROLE.researcher })
}
if (!solved && !inconclusive) log(`Bughunt exhausted ${MAX_ROUNDS} rounds — human review needed; knowledge in ${DIR}/EVIDENCE.md and the result`)
if (inconclusive) log(`Bughunt: clean over ${verified?.runs} runs but ${verified?.causal ? `${closure.needed} are needed for the statistical bound` : `the causal A/B is not proven (pre-fix files restored: ${verified?.reverted?.failures ?? '?'}/${verified?.reverted?.runs ?? '?'} failures; fixed tree: ${verified?.fixedArm?.failures ?? '?'}/${verified?.fixedArm?.runs ?? '?'}, ${causalRuns} needed)`} — INCONCLUSIVE, the fix stays in place`)
return {
  ok: solved,
  solved,
  inconclusive,
  verified,
  closureRunsNeeded: closure.needed,
  bug: bugId,
  rounds: round,
  fixedBy,
  knowledge: { refuted: knowledge.refuted.length, experiments: knowledge.experiments.length, rejectedFixes: knowledge.rejectedFixes },
}
