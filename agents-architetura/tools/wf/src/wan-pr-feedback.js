export const meta = {
  name: 'wan-pr-feedback',
  description: 'Close the loop with the PR review bots: collect CodeRabbit / Claude Code Review comments on a PR (read-only gh calls), have 2 read-only refuters verify each against the current code, run only confirmed findings through the scoped propose -> 3 votes -> apply pipeline, then a solo ladder run (wan-fix if red). Posts nothing to GitHub.',
  whenToUse: 'After the bots reviewed a PR. args: {"pr": 12}. Rerun after each push until no confirmed findings remain.',
  phases: [
    { title: 'Collect', detail: 'Solo: gh api reads of review + issue comments from the bots' },
    { title: 'Verify', detail: '2 read-only refuters per comment; bots hallucinate too' },
    { title: 'Fix', detail: 'Confirmed, file-anchored findings: propose -> 3 votes -> apply in scope' },
    { title: 'Check', detail: 'Solo ladder; wan-fix if red' },
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

const pr = Number.isInteger(args?.pr) && args.pr > 0 && args.pr < 1e7 ? args.pr : null
if (!pr) throw new Error('args.pr (the pull request number) is required')
const BOTS = ['coderabbitai[bot]', 'claude[bot]', 'github-actions[bot]']
const MAX_COMMENTS = 40
const MIN_BODY = 40

phase('Collect')
const collected = await agent(common('You run ALONE — you MAY run read-only gh commands. Never post, edit, resolve or react to anything.') + `
1. repo = output of: gh repo view --json nameWithOwner -q .nameWithOwner
2. Read, with --paginate: gh api repos/<repo>/pulls/${pr}/comments ; gh api repos/<repo>/pulls/${pr}/reviews ;
   gh api repos/<repo>/issues/${pr}/comments
3. Keep only items whose author login is one of ${JSON.stringify(BOTS)}. For each: id (string), author, path and line when
   the comment is anchored to code, outdated=true when GitHub marks it outdated (position null on a review comment),
   and the body verbatim.`,
  { label: 'collect', phase: 'Collect', schema: PR_COMMENTS, agentType: ROLE.verifier })
if (!collected) throw new Error('collector produced nothing')
const seenIds = new Set()
const comments = (collected.comments ?? [])
  .filter(c => BOTS.includes(c.author) && typeof c.body === 'string' && c.body.trim().length >= MIN_BODY && !c.outdated)
  .map(c => ({ ...c, id: token(c.id, 32) }))
  .filter(c => !seenIds.has(c.id) && seenIds.add(c.id))
  .slice(0, MAX_COMMENTS)
log(`Collect: ${comments.length} actionable bot comment(s) on PR #${pr}`)
if (!comments.length) return { ok: true, pr, collected: 0, accepted: [], applied: [], rejected: [], unscoped: [] }

phase('Verify')
const judged = await parallel(comments.map(c => () =>
  parallel([0, 1].map(n => () =>
    agent(common('READ-ONLY refuter.') + `
A review bot (${clean(c.author, 40)}) left this comment on PR #${pr}${c.path ? ` at ${clean(c.path, 200)}:${c.line ?? '?'}` : ''}. Try to REFUTE it against
the CURRENT code: is the claimed defect real, reachable and unfixed? Bots hallucinate — a plausible-sounding claim about
code that does not exist, or a style preference presented as a bug, is refuted.
${fence('bot_comment', c.body, 8000)}`,
      { label: `check:${c.id}:${n}`, phase: 'Verify', schema: VERDICT, agentType: ROLE.reviewer })))
    .then(vs => ({ c, verdicts: vs.filter(Boolean) }))))
const accepted = judged.filter(j => j && j.verdicts.length === 2 && j.verdicts.every(v => v.verdict !== 'refuted'))
const refutedIds = judged.filter(j => j && !accepted.includes(j)).map(j => j.c.id)
log(`Verify: ${accepted.length} confirmed, ${refutedIds.length} refuted or unverified`)

phase('Fix')
const anchored = accepted.filter(j => j.c.path && safeScopePath(j.c.path))
const unscoped = accepted.filter(j => !anchored.includes(j)).map(j => ({ id: j.c.id, author: j.c.author, body: clean(j.c.body, 600) }))
const { items } = normalizeItems(anchored.map(j => ({
  id: `c${j.c.id}`,
  rung: 'unknown',
  symptom: j.c.body,
  evidence: j.verdicts.map(v => v.argument).join('\n---\n'),
  scope: [j.c.path],
})), { knownRungs: RUNG_IDS, cap: 20 })
const outcomes = items.length ? await runFixPipeline(items, { phaseTitle: 'Fix', source: 'a verified PR review-bot comment' }) : []
const conflict = specConflictStop(outcomes, 'pr-feedback')
if (conflict) return conflict
const applied = outcomes.filter(o => o.outcome === 'applied').map(o => o.it.id)
const rejected = rejectionsOf(outcomes, 1)

phase('Check')
let green = true
if (applied.length) {
  const check = await agent(common('You run ALONE — you MAY build and run anything (no git).') + `
Run: ${ladderCommand()}
Put the JSON into "ladder" verbatim; for each failing rung record items (rung, evidence, minimal scope). Fix nothing.`,
    { label: 'check-ladder', phase: 'Check', schema: TRIAGE, agentType: ROLE.verifier })
  green = ladderGreen(check)
  if (!green) green = (await workflow('wan-fix', { rungs: failingRungs(check?.ladder), report: check }))?.ok === true
}

return {
  ok: green && !rejected.length && !unscoped.length,
  pr,
  collected: comments.length,
  accepted: accepted.map(j => j.c.id),
  refuted: refutedIds,
  applied,
  rejected,
  unscoped,
  ladderGreen: green,
}
