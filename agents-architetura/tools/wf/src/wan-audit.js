export const meta = {
  name: 'wan-audit',
  description: 'Mechanism-class security audit, as the reference PR audited its engine against 20 classes distilled from V8/JSC/SpiderMonkey/JVM/CLR/Go CVEs: three parallel research passes (ML frameworks and distributed runtimes; federated/adversarial ML; P2P and WAN systems) -> a catalog of classes deduped by FAILING INVARIANT -> per class, a surface map with a verdict per surface (immune with the protocol cited, or needs-test) -> susceptibility tests in tests/redteam/cve/<class>/ inside a scope audit -> a solo run -> two refuters per susceptible surface -> verdicts SUSCEPTIBLE / immune / INCONCLUSIVE recorded by the script (docs/security/AUDIT-RESULTS.json, open findings in agents-architetura/docs/findings.json). With close=true, SUSCEPTIBLE surfaces go through wan-fix and are re-run.',
  whenToUse: 'Per milestone after implementation (wan-cycle), and when the threat model changes. args: {"pass": 1, "close": false}; pass=2 re-runs the research and folds new findings into the existing classes.',
  phases: [
    { title: 'Gate', detail: 'Design approved; existing catalog loaded' },
    { title: 'Research', detail: '3 researchers, web, cited and confidence-labeled: docs/security/research/<area>.md' },
    { title: 'Catalog', detail: 'Researcher: classes by failing invariant -> docs/security/AUDIT.md' },
    { title: 'Map', detail: 'Per class: surfaces with verdict immune (cited) or needs-test -> docs/security/map-<class>.md' },
    { title: 'Test', detail: 'Red-team writers per class in tests/redteam/cve/<class>/ inside a scope audit' },
    { title: 'Run', detail: 'Verifier runs every class suite; 2 refuters per susceptible surface' },
    { title: 'Results', detail: 'Script verdicts -> AUDIT-RESULTS.json + findings; researcher writes AUDIT-RESULTS.md' },
    { title: 'Close', detail: 'close=true: wan-fix on the redteam rung, then re-run the susceptible classes' },
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

const SEC = 'docs/security'
const PASS = args?.pass === 2 ? 2 : 1
const MAX_CLASSES = 24
const CLASS_RE = /^MC-[A-Z0-9][A-Z0-9-]{1,23}$/
const solo = 'You run ALONE — you MAY build and run anything (no git).'
const AREAS = [
  ['ml-runtimes', `ML frameworks and distributed runtimes: PyTorch distributed and torch.load/pickle deserialization, TorchServe,
Ray (unauthenticated job and dashboard APIs), Horovod, DeepSpeed, NCCL/gloo, gRPC and protobuf (message size, decompression
bombs), MPI — CVEs, advisories and post-mortems`],
  ['adversarial-ml', `federated and collaborative training under adversaries: data and model poisoning, model replacement, Byzantine
updates against robust aggregators (Krum, trimmed mean, median and their known breaks), gradient inversion and membership
leakage, free-riding, sybil participants`],
  ['p2p-wan', `P2P and wide-area systems: DHT Sybil and eclipse attacks (libp2p, Hivemind, Petals), NAT traversal, TLS and
authentication mistakes, replay and reordering, version skew, resource exhaustion and amplification, clock skew`],
]
const CLASSES_SCHEMA = {
  type: 'object',
  required: ['classes'],
  properties: {
    classes: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'failingInvariant', 'exemplars', 'surface', 'fixShape'],
        properties: {
          id: { type: 'string', description: 'MC-<NAME>, e.g. MC-DESER' },
          failingInvariant: { type: 'string', description: 'the invariant every member of the class breaks' },
          exemplars: { type: 'array', items: { type: 'string' }, description: 'CVE ids / papers / post-mortems, each cited in the research files' },
          surface: { type: 'string', description: 'where our design and code expose it' },
          fixShape: { type: 'string', description: 'the known-good fix shape and its runtime cost model (per byte/frame/connection/step)' },
        },
      },
    },
  },
}
const SURFACES = {
  type: 'object',
  required: ['surfaces'],
  properties: {
    surfaces: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'location', 'verdict', 'argument'],
        properties: {
          id: { type: 'string', description: 'S1, S2, ...' },
          location: { type: 'string', description: 'file:line or spec section' },
          verdict: { type: 'string', enum: ['immune', 'needs-test'] },
          argument: { type: 'string', description: 'immune: the protocol that makes it impossible, cited adversarially (file:line / spec §); needs-test: why it might be susceptible' },
          test: { type: 'string', description: 'needs-test: what the susceptibility test injects and the CORRECT behavior it asserts' },
        },
      },
    },
  },
}
const AUDIT_RUN = {
  type: 'object',
  required: ['results'],
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        required: ['classId', 'surfaceId', 'outcome', 'evidence'],
        properties: {
          classId: { type: 'string' },
          surfaceId: { type: 'string' },
          outcome: { type: 'string', enum: ['susceptible', 'resistant', 'broken-test', 'not-run'] },
          evidence: { type: 'string', description: 'exact failing output, or the passing run summary' },
          reproCommand: { type: 'string' },
        },
      },
    },
  },
}

phase('Gate')
await requireApprovedDesign('Gate')
const existing = await agent(common('READ-ONLY.') + `
If ${SEC}/AUDIT.md exists, find the block after "<!-- classes -->" and return its JSON array as "classes", verbatim. Otherwise
(or if the block is missing or invalid) return an empty array. Do nothing else.`,
  { label: 'extract:classes', phase: 'Gate', schema: CLASSES_SCHEMA, agentType: ROLE.readonly })
let classes = (existing?.classes ?? []).filter(c => CLASS_RE.test(c.id ?? ''))

if (!classes.length || PASS === 2) {
  phase('Research')
  const research = (await parallel(AREAS.map(([key, area]) => () =>
    agent(common(`OWNED PATHS: ${SEC}/research/${key}.md (only). You MAY use web search and fetch.`) + `
Research pass ${PASS} for the security audit of this system. Area: ${area}.
Write ${SEC}/research/${key}.md: every entry has a citation (CVE id, advisory, paper or post-mortem URL), a confidence label
(confirmed / reported / speculative) and the MECHANISM in one sentence — what invariant broke, not just what happened.
${PASS === 2 ? 'This is the second pass: add what the first pass missed (newer entries, other systems); keep existing entries.' : ''}
Defensive purpose: we test OUR implementation for these mechanisms; do not write exploit code.`,
      { label: `research:${key}`, phase: 'Research', schema: RESULT, agentType: ROLE.researcher })))).filter(Boolean)
  if (research.length < AREAS.length) throw new Error(`research incomplete (${research.length}/${AREAS.length}) — resume this run`)

  phase('Catalog')
  const cat = await agent(common(`OWNED PATHS: ${SEC}/AUDIT.md (only).`) + `
Merge ${AREAS.map(([k]) => `${SEC}/research/${k}.md`).join(', ')} into the catalog ${SEC}/AUDIT.md.
CLASSES ARE THE UNIT OF TESTING, not individual CVEs: dedupe by FAILING INVARIANT (two CVEs in different systems that break the
same invariant are one class). Per class: id MC-<NAME>, the failing invariant, exemplars (cited), where THIS design and code expose
it (read ${DESIGN}, the specs and the code), and the known-good fix shape with its runtime cost. At most ${MAX_CLASSES} classes, most dangerous first.${SECURITY_COST}
${PASS === 2 && classes.length ? `Second pass: KEEP the existing taxonomy and ids; map every new finding to an existing class, add a class only if no existing failing invariant covers it. Existing: ${fence('existing_classes', classes, 12000)}` : ''}
End the file with "<!-- classes -->" followed by the classes as one JSON array on the next lines.`,
    { label: 'catalog', phase: 'Catalog', schema: CLASSES_SCHEMA, agentType: ROLE.researcher })
  const seen = new Set()
  classes = (cat?.classes ?? []).filter(c => CLASS_RE.test(c.id ?? '') && !seen.has(c.id) && seen.add(c.id)).slice(0, MAX_CLASSES)
}
if (!classes.length) throw new Error('the audit catalog has no valid MC-* classes — nothing to audit')
log(`Audit pass ${PASS}: ${classes.length} classes — ${classes.map(c => c.id).join(', ')}`)

phase('Map')
const maps = await parallel(classes.map(c => () =>
  agent(common(`OWNED PATHS: ${SEC}/map-${c.id}.md (only).`) + `
Surface map for class ${c.id}. ${fence('class', c, 4000)}
Enumerate every place in THIS code and design where the class's failing invariant could break (surfaces S1, S2, ...). For each:
verdict "immune" ONLY with the protocol that makes it impossible, argued adversarially and cited (file:line / spec section) —
assume it is susceptible until the code proves otherwise; otherwise "needs-test" with the test that would demonstrate it.
Write ${SEC}/map-${c.id}.md and return the surfaces.`,
    { label: `map:${c.id}`, phase: 'Map', schema: SURFACES, agentType: ROLE.researcher })))
const surfaces = classes.flatMap((c, i) => (maps[i]?.surfaces ?? []).map(s => ({ classId: c.id, ...s, id: token(s.id, 12) })))
const unmapped = classes.filter((c, i) => !maps[i]?.surfaces?.length).map(c => c.id)
const needsTest = surfaces.filter(s => s.verdict === 'needs-test')
const testDir = cls => `tests/redteam/cve/${cls.toLowerCase()}/`

phase('Test')
const testedClasses = [...new Set(needsTest.map(s => s.classId))]
if (testedClasses.length) {
  const since = await scopeSnapshot('audit-tests', 'Test')
  await parallel(testedClasses.map(cls => () =>
    agent(common(`OWNED PATHS: ${testDir(cls)} (only). Product code is READ-ONLY for you. Write code only — no runs.`) + `
Write the susceptibility tests for class ${cls}, one test per surface below. Each test injects the condition and asserts the
CORRECT behavior the design promises, so it FAILS while the system is susceptible. Name each test after its surface id
(e.g. test_s2_*), put "covers: ${cls}-<surface id>" and "WOULD-FAIL-IF: <the vulnerability>" in it. Deterministic and seeded.
No expected-failure markers. Defensive tests only: assert the system rejects or survives the input.
${fence('surfaces', needsTest.filter(s => s.classId === cls), 12000)}`,
      { label: `cve-test:${cls}`, phase: 'Test', schema: RESULT, agentType: ROLE.redteam })))
  await scopeCheck('audit-tests', 'Test', since, testedClasses.map(testDir))
}

const runSuites = async (classIds, tag) => {
  if (!classIds.length) return []
  const r = await agent(common(solo) + `
Run each class suite separately with the project's test runner (${classIds.map(testDir).join(', ')}). Report one result per
surface (the tests are named after their surface id): susceptible (the test fails because the system misbehaves), resistant
(passes), broken-test (the test itself errors), or not-run. Paste the exact failing output as evidence; give a reproCommand.
Surfaces: ${JSON.stringify(needsTest.filter(s => classIds.includes(s.classId)).map(s => `${s.classId}/${s.id}`))}`,
    { label: `run:${tag}`, phase: 'Run', schema: AUDIT_RUN, agentType: ROLE.verifier })
  return r?.results ?? []
}

const refute = async (results, tag) => {
  const sus = results.filter(r => r.outcome === 'susceptible')
  const judged = await parallel(sus.map(r => () =>
    parallel([0, 1].map(n => () =>
      agent(common('READ-ONLY refuter.') + `
A susceptibility test for ${clean(r.classId, 32)}/${clean(r.surfaceId, 12)} failed. Try to REFUTE that this is a real vulnerability:
is the test sound, does it assert behavior the design actually promises, is the failure caused by the system rather than the
test or the environment? Read the test (${testDir(clean(r.classId, 32))}), the code and ${DESIGN}.
${fence('failure', { evidence: clean(r.evidence, 6000), repro: r.reproCommand }, 8000)}`,
        { label: `refute:${token(r.classId, 24)}-${token(r.surfaceId, 8)}:${n}:${tag}`, phase: 'Run', schema: VERDICT, agentType: ROLE.reviewer })))
      .then(vs => {
        const got = vs.filter(Boolean)
        const refuted = got.filter(v => v.verdict === 'refuted').length
        return { r, confirmed: got.length === 2 && refuted === 0, split: got.length < 2 || (refuted > 0 && refuted < got.length) }
      })))
  return judged.filter(Boolean)
}

phase('Run')
let runResults = await runSuites(testedClasses, 'r1')
let judged = await refute(runResults, 'r1')

const verdicts = () => surfaces.map(s => {
  if (s.verdict === 'immune') return { ...s, final: 'immune', basis: 'protocol cited' }
  const r = runResults.find(x => token(x.classId, 32) === token(s.classId, 32) && token(x.surfaceId, 12) === s.id)
  if (!r || r.outcome === 'not-run') return { ...s, final: 'INCONCLUSIVE', basis: 'test not run' }
  if (r.outcome === 'resistant') return { ...s, final: 'immune', basis: 'test passes' }
  if (r.outcome === 'broken-test') return { ...s, final: 'INCONCLUSIVE', basis: 'test is broken' }
  const j = judged.find(x => x.r === r)
  if (j?.confirmed) return { ...s, final: 'SUSCEPTIBLE', basis: 'test fails; not refuted', repro: clean(r.reproCommand, 300), evidence: clean(r.evidence, 800) }
  return { ...s, final: 'INCONCLUSIVE', basis: j?.split ? 'refuters disagree — a human decides' : 'both refuters say the test, not the system, is wrong', needsHuman: j?.split === true }
})

const recordResults = async (tag, phaseTitle) => {
  const v = verdicts()
  const count = f => v.filter(x => x.final === f).length
  const payload = {
    pass: PASS,
    classes: classes.length,
    surfaces: v.length,
    verdicts: { immune: count('immune'), SUSCEPTIBLE: count('SUSCEPTIBLE'), INCONCLUSIVE: count('INCONCLUSIVE') },
    openTierA: count('SUSCEPTIBLE'),
    unverdicted: count('INCONCLUSIVE') + unmapped.length,
    unmappedClasses: unmapped,
    items: v.filter(x => x.final !== 'immune').map(x => ({ class: x.classId, surface: x.id, verdict: x.final, basis: x.basis, location: clean(x.location, 200), repro: x.repro ?? null })),
  }
  const status = x => (x.final === 'SUSCEPTIBLE' ? 'open' : x.needsHuman ? 'needs-human' : 'closed')
  const findings = { findings: v.filter(x => x.verdict === 'needs-test' && (x.final !== 'INCONCLUSIVE' || x.needsHuman)).map(x => ({ id: `SEC-${x.classId}-${x.id}`.replace(/[^A-Za-z0-9_-]/g, '-'), status: status(x), title: `${x.classId} ${x.id}: ${clean(x.location, 120)}`, source: 'wan-audit' })) }
  await agent(common(solo) + `
Run exactly these two commands, in order, and report their output verbatim in "summary"; change nothing else:
${recordCommand('audit-results', payload)}
${recordCommand('findings', findings)}`,
    { label: `record:${tag}`, phase: phaseTitle, schema: RESULT, agentType: ROLE.verifier })
  return payload
}

phase('Results')
let results = await recordResults('r1', 'Results')
await agent(common(`OWNED PATHS: ${SEC}/AUDIT-RESULTS.md (only).`) + `
Write ${SEC}/AUDIT-RESULTS.md: SUSCEPTIBLE first (class, surface, hole at file:line, evidence, fix shape from ${SEC}/AUDIT.md, the
gating test), then INCONCLUSIVE (what would discharge each), then the immune surfaces (one line each with the protocol or test).
Numbers come from the recorded results below — do not recount.
${fence('audit_results', results, 20000)}`,
  { label: 'results-doc', phase: 'Results', schema: RESULT, agentType: ROLE.researcher })

let closed = null
if (args?.close === true && results.openTierA > 0) {
  phase('Close')
  closed = await workflow('wan-fix', { rungs: ['redteam'] })
  const reclasses = [...new Set(verdicts().filter(v => v.final === 'SUSCEPTIBLE').map(v => v.classId))]
  const rerun = await runSuites(reclasses, 'close')
  runResults = [...runResults.filter(r => !reclasses.includes(r.classId)), ...rerun]
  judged = [...judged.filter(j => !reclasses.includes(j.r.classId)), ...(await refute(rerun, 'close'))]
  results = await recordResults('close', 'Close')
}

log(`Audit: ${results.classes} classes, ${results.surfaces} surfaces — ${results.verdicts.SUSCEPTIBLE} SUSCEPTIBLE, ${results.verdicts.INCONCLUSIVE} INCONCLUSIVE, ${results.verdicts.immune} immune${unmapped.length ? `; ${unmapped.length} class(es) without a surface map` : ''}`)
return { ok: results.openTierA === 0 && results.unverdicted === 0, ...results, closeFix: closed ? { ok: closed.ok === true } : null }
