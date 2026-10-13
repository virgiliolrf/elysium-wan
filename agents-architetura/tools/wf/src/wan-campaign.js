export const meta = {
  name: 'wan-campaign',
  description: 'Drive one verification campaign (sanitizer, amplifier or load run) to ZERO unsuppressed reports — the mechanics that took the reference PR from 10,515 TSAN reports to 0: one snapshot run -> reports deduped into families by signature -> the researcher rules every family (real-bug / protocol-fix / benign-by-spec / third-party) with the files its fix writes -> waves of parallel write-only fixers on disjoint files inside a scope audit -> one build -> two adversarial reviewers (benign-or-bug, flag-off-identity) -> amend -> ONE re-run per wave -> loop. A stalled count sends the top family to wan-bughunt. Final gate: zero, every suppression justified, unit/oracle/identity green.',
  whenToUse: 'Per campaign in agents-architetura/tools/campaign/campaigns.json (wan-cycle runs the ones the milestone GATES file pins). args: {"campaign": "rust-tsan", "maxWaves": 8}.',
  phases: [
    { title: 'Gate', detail: 'Design approved; campaign id valid' },
    { title: 'Snapshot', detail: 'Verifier: one full campaign run (round 0); families pasted verbatim' },
    { title: 'Rule', detail: 'Researcher: ruling + fix shape + written files per family; suppressions only with a ruling and a justification' },
    { title: 'Waves', detail: 'Disjoint write-only fixers in a scope audit -> build -> 2 reviewers -> amend -> one re-run' },
    { title: 'Final', detail: 'Zero unsuppressed; suppressions audit; unit, oracle and identity green' },
    { title: 'Report', detail: 'Researcher: agents-architetura/docs/campaigns/<id>/RESULTS.md; script: RESULTS.json' },
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

const id = String(args?.campaign ?? '')
if (!/^[a-z0-9][a-z0-9-]{0,40}$/.test(id)) throw new Error('args.campaign must be a campaign id from agents-architetura/tools/campaign/campaigns.json')
const MAX_WAVES = Number.isInteger(args?.maxWaves) && args.maxWaves >= 1 && args.maxWaves <= 20 ? args.maxWaves : 8
const OUT = `agents-architetura/docs/campaigns/${id}`
const SUPPRESSIONS = `agents-architetura/tools/campaign/suppressions/${id}.txt`
const solo = 'You run ALONE — you MAY build and run anything (no git).'

phase('Gate')
await requireApprovedDesign('Gate')

let reparses = 0
const runRound = async (round, phaseTitle, { rerun = true } = {}) => {
  const r = await agent(common(solo) + `
Run exactly: node agents-architetura/tools/campaign/families.mjs --campaign ${id} --round ${round}${rerun ? ' --run' : ''} --json
${rerun ? 'It runs the whole campaign once (it may take hours) and' : 'It re-reads the stored round-' + round + ' reports with the current suppressions and'} prints one JSON object.
Return that JSON EXACTLY as printed. Do not rerun it to get a better number; change nothing.`,
    { label: rerun ? `round:r${round}` : `reparse:r${round}:${++reparses}`, phase: phaseTitle, schema: ROUND, agentType: ROLE.verifier })
  if (!r) throw new Error(`campaign round ${round} produced no result`)
  return r
}

phase('Snapshot')
let current = await runRound(0, 'Snapshot')
if (current.applicable === false) {
  log(`Campaign ${id}: not applicable — ${clean(current.reason, 300)}`)
  return { ok: true, applicable: false, campaign: id, reason: clean(current.reason, 300) }
}
if (current.broken) return { ok: false, campaign: id, stoppedAt: 'snapshot', why: clean(current.brokenReason, 800) }
const counts = [current.unsuppressedReports]
const rounds = [{ round: 0, unsuppressed: current.unsuppressedReports, families: current.families.length }]
log(`Campaign ${id} r0: ${current.unsuppressedReports} unsuppressed reports in ${current.families.length} families`)

const RULE_RULES = `
RULINGS (one per family; read the code at each location and the specs first):
- real-bug: a defect no spec blesses — it gets fixed; the fixer must state the interleaving/ordering it closes.
- protocol-fix: a mechanical fix of known shape (atomic access, the existing concurrent accessor, a lock, an ordering).
- benign-by-spec: ONLY when a spec section explicitly blesses the behavior. Cite it (specRef =
  docs/design/SPEC-<key>.md#<anchor>) AND write the suppression entry in ${SUPPRESSIONS}:
      # RULING: benign-by-spec <specRef>
      # JUSTIFICATION: <why the spec makes this harmless, at least 20 characters>
      match: <a substring of the family signature>
- third-party: the report lies entirely in third-party code we cannot change — same entry with
  "# RULING: third-party <component>"; the match may not name our own source paths.
Suppression is the last resort: a report in code the factory owns that no spec blesses is a bug, never noise.
files: the exact files the fix will WRITE — exhaustive and minimal (disjoint waves are planned from them).`

const ruled = new Map()
const rule = async (families, round, misruled = []) => {
  if (!families.length) return
  const r = await agent(common(`OWNED PATHS: ${SUPPRESSIONS}, ${OUT}/TRIAGE.md (only).`) + RULE_RULES + `
Campaign ${id}, round ${round}. Rule every family below; keep ${OUT}/TRIAGE.md as the table of every family you ever ruled
(family, count per round, ruling, spec citation, fix shape, files).
${misruled.length ? `The agents who tried to fix these families say the ruling is WRONG — re-rule them on the evidence:\n${fence('misruled', misruled, 6000)}` : ''}
${fence('families', families.map(f => ({ id: f.id, kind: f.kind, count: f.count, signature: f.signature, locations: f.locations, message: f.message })), 40000)}`,
    { label: `rule:r${round}`, phase: 'Rule', schema: RULINGS, agentType: ROLE.researcher })
  for (const x of r?.rulings ?? []) ruled.set(token(x.familyId, 64), { ...x, familyId: token(x.familyId, 64) })
}

phase('Rule')
await rule(current.families, 0)
if (current.families.some(f => ['benign-by-spec', 'third-party'].includes(ruled.get(f.id)?.ruling))) current = await runRound(0, 'Rule', { rerun: false })

const LENSES = [
  ['benign-or-bug', `The failure mode of a campaign is a REAL defect annotated into silence. For every family fixed or suppressed this
wave, check the ruling against the spec it cites: "benign" is sound only where a stale or reordered value is harmless by the
spec. Pull the real-bug fixes apart first: demand the happens-before (or message-ordering) argument and check it in the code.`],
  ['flag-off-identity', `Did any fix change the feature-off path or the synchronous reference mode (extra synchronization, changed layout,
changed numerics, new work on the default path)? Did anyone weaken an assertion, a test or a timeout to make a report go away?`],
]

const bughunted = new Set()
const bughunts = []
let wave = 0
let stopReason = null
while (!cleanRound(current) && wave < MAX_WAVES) {
  wave++
  phase('Waves')
  const unruled = current.families.filter(f => !ruled.has(f.id))
  if (unruled.length) await rule(unruled, wave)
  const plan = partitionWave(current.families, [...ruled.values()])
  if (plan.unplaceable.length) log(`Wave ${wave}: ${plan.unplaceable.length} family(ies) ruled fixable but with no safe file list — re-ruled next round: ${plan.unplaceable.join(', ')}`)
  if (!plan.work.length) {
    const suppressible = current.families.filter(f => ['benign-by-spec', 'third-party'].includes(ruled.get(f.id)?.ruling))
    if (suppressible.length) {
      await rule(suppressible, wave, suppressible.map(f => ({ familyId: f.id, evidence: 'ruled benign/third-party but its suppression entry does not match the family signature (or is invalid)' })))
      current = await runRound(wave - 1, 'Waves', { rerun: false })
      continue
    }
    stopReason = `nothing fixable this wave (${plan.unplaceable.length} unplaceable, ${plan.unruled.length} unruled) — a human must rule`
    break
  }
  log(`Wave ${wave}: ${plan.work.length} families (${plan.work.filter(w => w.solo).length} real-bug solo-grade) — ${plan.work.map(w => w.family.id).join(', ')}${plan.deferred.length ? `; deferred ${plan.deferred.length} (file overlap)` : ''}`)

  const since = await scopeSnapshot(`w${wave}`, 'Waves')
  const fixes = await parallel(plan.work.map(w => () =>
    agent(common(`WRITE-ONLY family fixer, wave ${wave}: do NOT build or run anything — one builder compiles the whole wave after you.
OWNED PATHS (data): ${JSON.stringify(w.files.slice(0, 16))}`) + `
Family ${w.family.id} (${w.family.count} reports, ruling ${w.ruling.ruling}).
Fix shape from the ruling: ${fence('fix_shape', w.ruling.fixShape, 1500)}
Family: ${fence('family', { signature: w.family.signature, locations: w.family.locations, message: w.family.message }, 4000)}
Triage table: ${OUT}/TRIAGE.md. Read the spec sections it cites.
${w.solo
    ? 'REAL BUG: state in your summary the interleaving (or message ordering) that produced the reports and the happens-before your fix establishes — the reviewers will check it.'
    : 'Mechanical ruling: apply exactly that fix shape. If reading the code shows the ruling is WRONG (the behavior is not benign, or the shape cannot be right), set "misruled" with the evidence and change nothing.'}
Feature-off and reference-mode behavior must be unchanged. Never weaken an assertion or a test.`,
      { label: `fix:${w.family.id}:w${wave}`, phase: 'Waves', schema: CAMPAIGN_FIX, agentType: ROLE.coder })))
  await scopeCheck(`campaign-w${wave}`, 'Waves', since, [...new Set(plan.work.flatMap(w => w.files))])
  const misruled = plan.work.map((w, i) => ({ w, f: fixes[i] })).filter(x => x.f?.misruled).map(x => ({ familyId: x.w.family.id, evidence: clean(x.f.misruled, 1500) }))

  const build = await agent(common(solo) + `
Wave ${wave} builder: run ${ladderCommand({ only: ['build', 'syntax', 'unit'] })} and paste its JSON into "ladder" verbatim.
For each failing rung, record items (rung, exact evidence, minimal scope). allGreen=true only if the pasted ladder says so.`,
    { label: `build:w${wave}`, phase: 'Waves', schema: TRIAGE, agentType: ROLE.verifier })
  if (!ladderGreen(build)) {
    const fixed = await workflow('wan-fix', { rungs: failingRungs(build?.ladder), report: build })
    if (fixed?.ok !== true) { stopReason = `wave ${wave} broke the build and wan-fix could not repair it`; break }
  }

  const reviews = (await parallel(LENSES.map(([name, lens]) => () =>
    agent(common('READ-ONLY adversarial reviewer.') + `
Campaign ${id}, wave ${wave}, lens ${name}. ${lens}
Families handled this wave: ${fence('wave', plan.work.map(w => ({ id: w.family.id, ruling: w.ruling.ruling, files: w.files, fixer: clean(fixes[plan.work.indexOf(w)]?.summary, 800) })), 12000)}
Suppressions file: ${SUPPRESSIONS}. Findings: blocker/major only, validated against the current code.`,
      { label: `review:${name}:w${wave}`, phase: 'Waves', schema: FINDINGS, agentType: ROLE.reviewer })))).filter(Boolean)
  const serious = reviews.flatMap(r => r.findings ?? []).filter(isActionable)
  if (reviews.length < LENSES.length) log(`Wave ${wave}: only ${reviews.length}/${LENSES.length} reviewers reported`)
  if (serious.length) {
    log(`Wave ${wave}: ${serious.length} blocker/major review finding(s) -> amend`)
    await agent(common(`OWNED PATHS (data): ${JSON.stringify([...new Set([...plan.work.flatMap(w => w.files), SUPPRESSIONS])])}`) + `
Amend wave ${wave} of campaign ${id} for these reviewed findings. Verify each against the code; fix the real ones; refute the
rest in your summary with file:line evidence. Do not build — the re-run measures the result.
${fence('reviewer_findings', serious, 20000)}`,
      { label: `amend:w${wave}`, phase: 'Waves', schema: RESULT, agentType: ROLE.coder })
  }

  current = await runRound(wave, 'Waves')
  if (current.applicable === false || current.broken) { stopReason = `round ${wave} is ${current.broken ? 'broken' : 'not applicable'}: ${clean(current.brokenReason ?? current.reason, 600)}`; break }
  counts.push(current.unsuppressedReports)
  rounds.push({ round: wave, unsuppressed: current.unsuppressedReports, families: current.families.length, new: current.delta?.new?.length ?? 0, fixed: current.delta?.fixed?.length ?? 0 })
  log(`Wave ${wave}: ${counts.at(-2)} -> ${current.unsuppressedReports} unsuppressed`)
  for (const m of misruled) ruled.delete(m.familyId)
  if (misruled.length) await rule(current.families.filter(f => misruled.some(m => m.familyId === f.id)), wave, misruled)

  if (stalled(counts) && current.families.length) {
    const top = current.families[0]
    if (bughunted.has(top.id)) { stopReason = `stalled at ${current.unsuppressedReports} even after wan-bughunt on ${top.id}`; break }
    bughunted.add(top.id)
    log(`Wave ${wave}: no progress for 2 waves -> wan-bughunt on ${top.id}`)
    const r = await workflow('wan-bughunt', { bug: `campaign ${id} family ${top.id} (${top.count} reports): ${clean(top.message, 600)} at ${clean((top.locations ?? []).join(', '), 300)}`, id: `${id}-${top.id}`.slice(0, 40), repro: `node agents-architetura/tools/campaign/families.mjs --campaign ${id} --round 900 --run` })
    bughunts.push({ family: top.id, solved: r?.solved === true })
  }
}

phase('Final')
let final = null
if (cleanRound(current)) {
  final = await agent(common(solo) + `
Final gate for campaign ${id}.
1. Run ${ladderCommand({ only: ['unit', 'oracle', 'identity', 'discipline'] })} and paste its JSON into "ladder" verbatim.
2. Run node agents-architetura/tools/campaign/suppressions-audit.mjs --json; if it reports violations, add one item per violation (rung "discipline").
allGreen=true only if the pasted ladder is green and the suppressions audit has no violations. Fix nothing.`,
    { label: 'final', phase: 'Final', schema: TRIAGE, agentType: ROLE.verifier })
}
const ok = cleanRound(current) && ladderGreen(final) && !(final?.items ?? []).length
if (!ok && !stopReason) stopReason = cleanRound(current) ? 'zero reports, but the final ladder or the suppressions audit is red' : `still ${current.unsuppressedReports} unsuppressed after ${wave} wave(s)`

phase('Report')
const results = { campaign: id, ok, rounds, waves: wave, unsuppressed: current.unsuppressedReports ?? null, stopReason, bughunts, suppressed: (current.suppressed ?? []).length }
await agent(common(solo) + `
Run exactly: ${recordCommand('campaign', results, id)}
It writes ${OUT}/RESULTS.json. Report its output verbatim in "summary"; change nothing else.`,
  { label: 'record', phase: 'Report', schema: RESULT, agentType: ROLE.verifier })
await agent(common(`OWNED PATHS: ${OUT}/RESULTS.md (only).`) + `
Write ${OUT}/RESULTS.md: the round-by-round table (round, unsuppressed reports, families, new, fixed), families fixed by ruling
type, every suppression with its ruling and justification (read ${SUPPRESSIONS}), bughunts, residuals and why. Honest partials
over fake green: if the campaign did not reach zero, say so first.
${fence('campaign_results', results, 12000)}`,
  { label: 'report', phase: 'Report', schema: RESULT, agentType: ROLE.researcher })

log(`Campaign ${id}: ${ok ? 'ZERO — final gate green' : `NOT at zero (${stopReason})`}`)
return results
