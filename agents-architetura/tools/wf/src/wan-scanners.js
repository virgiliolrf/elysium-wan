export const meta = {
  name: 'wan-scanners',
  description: 'Static and dynamic analyzer battery with triage, modeled on the reference PR\'s thread-scanners: every scanner in agents-architetura/tools/campaign/scanners.json runs solo (SARIF / diagnostics / junit parsed and deduped into families; scanners this machine cannot run are reported loudly, never dropped) -> read-only reviewers rule each family real / false-positive / out-of-scope against the current code -> real findings go through propose -> 3 votes -> apply; false positives get a justified suppression entry -> the affected scanners re-run and residuals are reported honestly.',
  whenToUse: 'Per milestone (wan-cycle) and before a release. args: {"maxFamilies": 60}.',
  phases: [
    { title: 'Gate', detail: 'Design approved' },
    { title: 'Battery', detail: 'Verifier: node agents-architetura/tools/campaign/scan.mjs --json, pasted verbatim' },
    { title: 'Triage', detail: 'Reviewers rule every family against the current code' },
    { title: 'Fix', detail: 'Real findings: propose -> 3 votes -> apply; false positives: justified suppressions' },
    { title: 'Verify', detail: 'Affected scanners re-run; residuals' },
    { title: 'Report', detail: 'Researcher: agents-architetura/docs/SCAN-RESULTS.md' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project
// @include gates
// @include triage
// @include fix
// @include ledger
// @include campaign

const MAX_FAMILIES = Number.isInteger(args?.maxFamilies) && args.maxFamilies >= 1 && args.maxFamilies <= 200 ? args.maxFamilies : 60
const BATCH = 10
const SCAN_TRIAGE = {
  type: 'object',
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['familyId', 'verdict', 'argument'],
        properties: {
          familyId: { type: 'string' },
          verdict: { type: 'string', enum: ['real', 'false-positive', 'out-of-scope', 'uncertain'], description: 'uncertain = you cannot establish it either way from the code: a human decides' },
          argument: { type: 'string', description: 'quoted code proving it; for false-positive, why the flagged pattern cannot misbehave here' },
          scope: { type: 'array', items: { type: 'string' }, description: 'real: minimal files the fix must touch' },
        },
      },
    },
  },
}

phase('Gate')
await requireApprovedDesign('Gate')

phase('Battery')
const battery = await runTool('node agents-architetura/tools/campaign/scan.mjs --json', { label: 'battery', phaseTitle: 'Battery', note: 'It runs every applicable scanner (it may take a long time).' })
if (battery?.exitCode === 5) return { ok: true, applicable: false, reason: 'no scanner applies to this tree yet' }
const scanners = battery?.json?.scanners ?? []
if (!scanners.length) throw new Error('the scanner battery produced no result')
const unavailable = scanners.filter(s => s.status === 'skipped-unavailable').map(s => ({ id: s.id, reason: clean(s.reason, 300) }))
const brokenScanners = scanners.filter(s => s.status === 'broken').map(s => ({ id: s.id, reason: clean(s.reason, 600) }))
for (const u of unavailable) log(`Battery: ${u.id} UNAVAILABLE on this machine — ${u.reason}`)
const families = scanners.flatMap(s => (s.families ?? []).map(f => ({ ...f, scanner: s.id }))).slice(0, MAX_FAMILIES)
log(`Battery: ${scanners.filter(s => s.status === 'ran').length} scanners ran, ${families.length} families, ${unavailable.length} unavailable, ${brokenScanners.length} broken`)

phase('Triage')
const batches = []
for (let i = 0; i < families.length; i += BATCH) batches.push(families.slice(i, i + BATCH))
const judged = (await parallel(batches.map((b, i) => () =>
  agent(common('READ-ONLY adversarial reviewer.') + `
Rule each static-analysis finding family below against the CURRENT code (open the file at each location).
real: the flagged pattern can misbehave in this code (give the minimal fix scope). false-positive: it cannot — quote the code
that makes it impossible. out-of-scope: generated, vendored or test-only code the factory does not ship.
${fence('families', b.map(f => ({ id: f.id, scanner: f.scanner, signature: f.signature, locations: f.locations, message: f.message })), 20000)}`,
    { label: `triage:b${i + 1}`, phase: 'Triage', schema: SCAN_TRIAGE, agentType: ROLE.reviewer })))).filter(Boolean)
const verdictOf = new Map(judged.flatMap(j => j.verdicts ?? []).map(v => [token(v.familyId, 64), v]))
const real = families.filter(f => verdictOf.get(f.id)?.verdict === 'real')
const quiet = families.filter(f => ['false-positive', 'out-of-scope'].includes(verdictOf.get(f.id)?.verdict))
const unjudged = families.filter(f => !verdictOf.has(f.id))
const uncertain = families.filter(f => verdictOf.get(f.id)?.verdict === 'uncertain')
if (uncertain.length) {
  await agent(common('You run ALONE — run this one command and report its output verbatim in "summary"; change nothing else.') + `
${recordCommand('findings', { findings: uncertain.map(f => ({ id: `SCAN-${f.id}`.slice(0, 64), status: 'needs-human', title: clean(`${f.scanner}: ${f.message}`, 200), source: 'wan-scanners', why: clean(verdictOf.get(f.id).argument, 600) })) })}`,
    { label: 'record:uncertain', phase: 'Triage', schema: RESULT, agentType: ROLE.verifier })
}

phase('Fix')
const raw = real.map(f => ({ id: f.id, rung: 'static', symptom: `${f.scanner}: ${clean(f.message, 300)}`, evidence: `${clean(f.signature, 300)} at ${clean((f.locations ?? []).join(', '), 300)}\n${clean(verdictOf.get(f.id).argument, 1500)}`, scope: verdictOf.get(f.id).scope ?? [] }))
const { items, dropped } = normalizeItems(raw, { knownRungs: RUNG_IDS, cap: 20 })
for (const d of dropped) log(`Fix: dropped ${d.id} (${d.why})`)
const outcomes = items.length ? await runFixPipeline(items, { phaseTitle: 'Fix', source: 'a static analyzer finding confirmed against the code' }) : []
const conflict = specConflictStop(outcomes, 'scanners')
if (conflict) return conflict
if (quiet.length) {
  const files = [...new Set(quiet.map(f => `agents-architetura/tools/campaign/suppressions/scan-${f.scanner}.txt`))]
  await agent(common(`OWNED PATHS: ${files.join(', ')} (only).`) + `
Add one suppression entry per family below to agents-architetura/tools/campaign/suppressions/scan-<scanner>.txt, in this exact format:
    # RULING: false-positive <rule id>          (or: # RULING: third-party <component> for out-of-scope vendored code)
    # JUSTIFICATION: <the reviewer's argument, condensed, at least 40 characters>
    match: <the family signature, verbatim>
${fence('families', quiet.map(f => ({ scanner: f.scanner, signature: f.signature, verdict: verdictOf.get(f.id).verdict, argument: clean(verdictOf.get(f.id).argument, 1200) })), 20000)}`,
    { label: 'suppress', phase: 'Fix', schema: RESULT, agentType: ROLE.researcher })
}

phase('Verify')
const affected = [...new Set([...real, ...quiet].map(f => f.scanner))]
const after = affected.length ? await runTool(`node agents-architetura/tools/campaign/scan.mjs --only ${affected.join(',')} --json`, { label: 'rescan', phaseTitle: 'Verify' }) : battery
const residual = (after?.json?.scanners ?? []).filter(s => affected.includes(s.id) || !affected.length).flatMap(s => (s.families ?? []).map(f => ({ scanner: s.id, id: f.id, count: f.count, signature: f.signature })))
  .filter(f => !uncertain.some(u => u.id === f.id))
const violations = (after?.json?.scanners ?? []).flatMap(s => s.suppressionViolations ?? [])

phase('Report')
const results = {
  ran: scanners.filter(s => s.status === 'ran').map(s => s.id),
  unavailable,
  broken: brokenScanners,
  families: families.length,
  real: real.length,
  suppressed: quiet.length,
  unjudged: unjudged.map(f => f.id),
  needsHuman: uncertain.map(f => f.id),
  applied: outcomes.filter(o => o.outcome === 'applied').length,
  residual,
  suppressionViolations: violations,
}
await agent(common('OWNED PATHS: agents-architetura/docs/SCAN-RESULTS.md (only).') + `
Write agents-architetura/docs/SCAN-RESULTS.md: which scanners ran, which could NOT run on this machine and why (first, loudly), broken
scanners, findings fixed, findings suppressed with their justification, residual findings and why they remain.
${fence('scan_results', results, 20000)}`,
  { label: 'report', phase: 'Report', schema: RESULT, agentType: ROLE.researcher })

const ok = !residual.length && !brokenScanners.length && !violations.length && !unjudged.length && !uncertain.length
log(`Scanners: ${ok ? 'clean' : `${residual.length} residual famil(ies), ${brokenScanners.length} broken, ${unjudged.length} unjudged`}; ${unavailable.length} unavailable`)
return { ok, ...results }
