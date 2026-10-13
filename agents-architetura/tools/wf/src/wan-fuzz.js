export const meta = {
  name: 'wan-fuzz',
  description: 'Coverage-guided fuzz campaign with triage, modeled on the reference PR\'s Fuzzilli rig: status -> smoke (self-test proves the generators reach the code) -> one solo campaign run under the machine lock -> every NEW unique crash (deduped by signature) is minimized into a committed regression input, diagnosed, fixed through propose -> 3 votes -> apply, and independently verified (minimized repro 20/20, 10/10 under load, unit + fuzz rungs, a 10-minute re-fuzz). Flaky crashes are re-run under the amplifier, never dropped.',
  whenToUse: 'Per fuzz campaign in agents-architetura/tools/campaign/campaigns.json; wan-cycle runs the ones the milestone GATES file pins. args: {"campaign": "fuzz-wire", "minutes": 60, "jobs": 1, "maxTriage": 8}.',
  phases: [
    { title: 'Gate', detail: 'Design approved; campaign id valid' },
    { title: 'Setup', detail: 'Status; a 10-minute smoke with the self-test if the campaign never ran' },
    { title: 'Campaign', detail: 'Verifier: one campaign run, new unique crashes pasted verbatim' },
    { title: 'Triage', detail: 'Per crash: minimize + regression input + diagnosis -> propose -> 3 votes -> apply' },
    { title: 'Verify', detail: 'Repro 20/20, load 10/10, unit + fuzz rungs, 10-minute re-fuzz; flaky crashes amplified' },
    { title: 'Report', detail: 'Researcher: agents-architetura/docs/fuzz/<id>.md' },
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

const id = String(args?.campaign ?? '')
if (!/^[a-z0-9][a-z0-9-]{0,40}$/.test(id)) throw new Error('args.campaign must be a fuzz campaign id from agents-architetura/tools/campaign/campaigns.json')
const intArg = (v, lo, hi, d) => (Number.isInteger(v) && v >= lo && v <= hi ? v : d)
const MINUTES = intArg(args?.minutes, 1, 1440, 60)
const JOBS = intArg(args?.jobs, 1, 64, 1)
const MAX_TRIAGE = intArg(args?.maxTriage, 1, 20, 8)
const REG = `tests/fuzz/regressions/${id}`
const FUZZ = `node agents-architetura/tools/campaign/fuzz-run.mjs --campaign ${id}`

const DIAGNOSIS = {
  type: 'object',
  required: ['minimizedInput', 'reproCommand', 'scope', 'suspectedCause'],
  properties: {
    minimizedInput: { type: 'string', description: `path of the minimized crashing input you saved under ${REG}/` },
    reproCommand: { type: 'string', description: 'command that replays exactly that input; exits non-zero while the bug exists' },
    scope: { type: 'array', items: { type: 'string' }, description: 'minimal product files the fix must touch' },
    suspectedCause: { type: 'string', description: 'file:line-grounded cause; facts and hypotheses separated' },
  },
}
const FUZZ_VERIFY = {
  type: 'object',
  required: ['items', 'ladder', 'refuzz'],
  properties: {
    items: { type: 'array', items: { type: 'object', required: ['id', 'reproRuns', 'reproFailures', 'loadFailed'], properties: { id: { type: 'string' }, reproRuns: { type: 'number' }, reproFailures: { type: 'number' }, loadFailed: { type: 'number' }, evidence: { type: 'string' } } } },
    ladder: LADDER_RESULT,
    refuzz: { type: 'object', description: 'the JSON the 10-minute re-fuzz printed, verbatim' },
  },
}

phase('Gate')
await requireApprovedDesign('Gate')

phase('Setup')
const status = await runTool(`${FUZZ} --status --json`, { label: 'status', phaseTitle: 'Setup' })
if (status?.exitCode === 5) { log(`Fuzz ${id}: not applicable`); return { ok: true, applicable: false, campaign: id } }
if ((status?.json?.runs ?? 0) === 0) {
  const smoke = await runTool(`${FUZZ} --minutes 10 --jobs 1 --json`, { label: 'smoke', phaseTitle: 'Setup', note: 'This is the 10-minute smoke; it runs the self-test first.' })
  if (!smoke || smoke.json?.selfTestFailed) return { ok: false, campaign: id, stoppedAt: 'setup', why: clean(smoke?.json?.detail ?? 'smoke produced nothing', 800) }
}

phase('Campaign')
const camp = await runTool(`${FUZZ} --minutes ${MINUTES} --jobs ${JOBS} --json`, { label: 'campaign', phaseTitle: 'Campaign', note: `It fuzzes for ${MINUTES} minutes under the machine lock.` })
if (!camp?.json || camp.json.selfTestFailed) return { ok: false, campaign: id, stoppedAt: 'campaign', why: clean(camp?.json?.detail ?? 'campaign produced nothing', 800) }
const fresh = (camp.json.newUnique ?? []).filter(n => typeof n?.signature === 'string')
const flaky = fresh.filter(n => n.flaky)
const crashes = fresh.filter(n => !n.flaky).slice(0, MAX_TRIAGE)
log(`Fuzz ${id}: ${camp.json.cpuHours ?? '?'} CPU-hours total, ${fresh.length} new unique (${flaky.length} flaky), triaging ${crashes.length}`)

phase('Triage')
const diagnosed = (await parallel(crashes.map((c, i) => () =>
  agent(common(`OWNED PATHS: ${REG}/${token(c.signature, 40)}/ (only). You MAY run the campaign's repro command on single inputs.`) + `
New unique crash ${i + 1}/${crashes.length} of fuzz campaign ${id}.
${fence('crash', { signature: c.signature, input: c.file }, 3000)}
1. Minimize the input (the engine's minimizer if the campaign has one, otherwise by hand: drop and simplify bytes while it still
   crashes with the same signature). Save the minimized input under ${REG}/${token(c.signature, 40)}/ — it becomes a permanent
   regression input that every later fuzz rung replays.
2. Add a README.md there with one line "WOULD-FAIL-IF: <the regression this input catches>".
3. Diagnose — do not fix: the file:line-grounded cause and the minimal product scope a fix must touch.`,
    { label: `diagnose:${token(c.signature, 24)}`, phase: 'Triage', schema: DIAGNOSIS, agentType: ROLE.coder })
    .then(d => (d ? { c, d } : null))))).filter(Boolean)
const raw = diagnosed.map(({ c, d }) => ({ id: token(c.signature, 40), rung: 'fuzz', symptom: c.signature, evidence: `repro: ${clean(d.reproCommand, 400)}\nminimized input: ${clean(d.minimizedInput, 200)}`, suspectedCause: d.suspectedCause, scope: d.scope, repro: clean(d.reproCommand, 400) }))
const { items, dropped } = normalizeItems(raw, { knownRungs: RUNG_IDS, cap: MAX_TRIAGE })
for (const x of dropped) log(`Triage: dropped ${x.id} (${x.why})`)
const outcomes = items.length ? await runFixPipeline(items, { phaseTitle: 'Triage', source: `a crash found by fuzz campaign ${id}` }) : []
const conflict = specConflictStop(outcomes, 'fuzz')
if (conflict) return conflict
const applied = outcomes.filter(o => o.outcome === 'applied')

phase('Verify')
let verify = null
if (applied.length) {
  verify = await agent(common('You run ALONE — you MAY build and run anything (no git). You did not write these fixes; change nothing.') + `
Independent verification of ${applied.length} fuzz fix(es).
Per item: run its repro command 20 times (count failures) and once under load:
  node agents-architetura/tools/verify/load.mjs --workers 2 --runs-per-worker 5 --json -- <repro>     (report "failed")
Then once: ${ladderCommand({ only: ['unit', 'fuzz'] })} — paste its JSON into "ladder" verbatim (the fuzz rung replays every
regression input). Then: ${FUZZ} --minutes 10 --json — paste its JSON into "refuzz" verbatim.
${fence('items', applied.map(o => ({ id: o.it.id, repro: o.it.repro })), 6000)}`,
    { label: 'verify', phase: 'Verify', schema: FUZZ_VERIFY, agentType: ROLE.verifier })
}
const triaged = new Set(applied.map(o => o.it.symptom))
const reopened = (verify?.refuzz?.newUnique ?? []).filter(n => triaged.has(n?.signature)).map(n => n.signature)
const fixedIds = new Set((verify?.items ?? []).filter(x => x.reproRuns >= 20 && x.reproFailures === 0 && x.loadFailed === 0).map(x => token(x.id, 40)))
const open = [
  ...items.filter(it => !fixedIds.has(it.id)).map(it => ({ id: it.id, signature: it.symptom, why: outcomes.find(o => o.it.id === it.id)?.outcome === 'applied' ? 'still crashes after the fix' : outcomes.find(o => o.it.id === it.id)?.outcome ?? 'no fix' })),
  ...crashes.filter(c => !diagnosed.some(d => d.c === c)).map(c => ({ id: token(c.signature, 40), signature: c.signature, why: 'diagnosis produced nothing' })),
  ...reopened.map(s => ({ id: token(s, 40), signature: s, why: 'the re-fuzz found the same signature again' })),
  ...(fresh.length - flaky.length > crashes.length ? [{ id: 'overflow', signature: '', why: `${fresh.length - flaky.length - crashes.length} unique crash(es) beyond maxTriage — next run` }] : []),
]
const flakyResults = []
for (const f of flaky) {
  const input = String(f.file ?? '')
  if (!safeScopePath(input.replace(/^.*?(\.factory\/)/, '$1'))) { flakyResults.push({ signature: f.signature, reproducedUnderAmplifier: null, why: 'unsafe input path' }); continue }
  const amp = await runTool(`node agents-architetura/tools/verify/amplify.mjs --runs 200 --seed-base 1 --json -- ${FUZZ} --replay --input ${input.replace(/^.*?(\.factory\/)/, '$1')}`, { label: `amplify:${token(f.signature, 24)}`, phaseTitle: 'Verify', note: `Flaky crash ${clean(f.signature, 200)}: amplify the replay of that one input.` })
  flakyResults.push({ signature: f.signature, reproducedUnderAmplifier: (amp?.json?.findings ?? []).length > 0 })
}
const ladderOk = !applied.length || ladderGreen({ allGreen: true, ladder: verify?.ladder })

phase('Report')
const results = { campaign: id, cpuHours: camp.json.cpuHours ?? null, newUnique: fresh.length, fixed: [...fixedIds], open, flaky: flakyResults, ladderGreen: ladderOk }
await agent(common(`OWNED PATHS: agents-architetura/docs/fuzz/${id}.md (only).`) + `
Write agents-architetura/docs/fuzz/${id}.md: campaign totals (CPU-hours, corpus, unique crashes), each crash (signature, minimized input,
cause, fix, verification), open crashes and why, flaky crashes and whether the amplifier reproduced them. Honest partials first.
${fence('fuzz_results', results, 16000)}`,
  { label: 'report', phase: 'Report', schema: RESULT, agentType: ROLE.researcher })

const ok = !open.length && ladderOk && flakyResults.every(f => f.reproducedUnderAmplifier === false)
log(`Fuzz ${id}: ${ok ? 'no open crash' : `${open.length} open crash(es)${flakyResults.some(f => f.reproducedUnderAmplifier) ? ', a flaky crash reproduces under the amplifier' : ''}`}`)
return { ok, ...results }
