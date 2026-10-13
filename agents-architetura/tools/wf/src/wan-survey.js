export const meta = {
  name: 'wan-survey',
  description: 'Understand before modifying: map the repository AS IT IS. A script inventory (files, languages, manifests, entrypoints, tests, CI, the internal import graph) grounds the researcher\'s docs/design/CURRENT_ARCHITECTURE.md and ARCHITECTURE_MAP.md (per component: responsibility, dependencies, interfaces, data flow, control flow, failure, security and resource boundaries); the script checks every cited path exists and every source directory belongs to a component; fidelity and boundary reviewers loop; then specialists are created ONLY where the code holds that responsibility. Optionally maps a reference codebase read-only (e.g. the previous Elysium) into docs/design/reference/. Never redesigns anything.',
  whenToUse: 'First step of wan-task, and at the start of every milestone (repository-aware planning). args: {"reference": "github:owner/repo"} optional.',
  phases: [
    { title: 'Inventory', detail: 'Script facts: node agents-architetura/tools/factory/inventory.mjs --json' },
    { title: 'Map', detail: 'Researcher: CURRENT_ARCHITECTURE.md + ARCHITECTURE_MAP.md with a machine-read components block' },
    { title: 'Check', detail: 'Script: 9 fields per component, cited paths exist, every source directory covered; repair <= 2' },
    { title: 'Review', detail: 'fidelity + boundaries reviewers loop <= 5' },
    { title: 'Specialists', detail: 'Only responsibilities the code has, with evidence; validated by script' },
    { title: 'Reference', detail: 'Optional: clone a reference repo read-only into .factory/reference and map it' },
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
// @include specialists

const CURRENT = `${DESIGN_DIR}/CURRENT_ARCHITECTURE.md`
const MAP = `${DESIGN_DIR}/ARCHITECTURE_MAP.md`
const OWN = `OWNED PATHS: ${CURRENT}, ${MAP} (only).`
const FIELDS = ['component', 'paths', 'responsibility', 'dependencies', 'interfaces', 'dataFlow', 'controlFlow', 'failureBoundary', 'securityBoundary', 'resourceBoundary']
const COMPONENTS = {
  type: 'object',
  required: ['components'],
  properties: {
    components: {
      type: 'array',
      items: {
        type: 'object',
        required: FIELDS,
        properties: Object.fromEntries(FIELDS.map(f => [f, f === 'paths' || f === 'dependencies' || f === 'interfaces' ? { type: 'array', items: { type: 'string' } } : { type: 'string' }])),
      },
    },
  },
}
const MAP_RULES = `
${MAP} ends with "<!-- components -->" followed by one JSON array: per component {${FIELDS.map(f => `"${f}"`).join(', ')}} —
paths are repo paths (directories end with "/"); dependencies and interfaces name real modules, functions, endpoints or
wire formats with file references. Describe what EXISTS — never what should exist. No redesign, no recommendations in
these two files (open questions go in a final "Unknowns" section).`
const REF_RE = /^github:([\w.-]+)\/([\w.-]+)$/

phase('Inventory')
const inv = await runTool('node agents-architetura/tools/factory/inventory.mjs --json', { label: 'inventory', phaseTitle: 'Inventory' })
if (!inv?.json?.files && inv?.json?.files !== 0) throw new Error('the inventory tool produced nothing')
const sourceDirs = Object.keys(inv.json.byTopDir ?? {}).filter(d => d !== '(root)' && !/^(docs|\.github|\.claude)\/$/.test(d))

phase('Map')
await agent(common(OWN) + MAP_RULES + `
Map this repository as it stands. Start from the script inventory below (facts), then READ the code: trace how a request,
a training step and a failure move through it end to end. Write ${CURRENT} (what actually exists: purpose, processes,
modules, how they talk, how it is configured, built, tested, deployed and observed, current limitations — every claim with a
file reference) and ${MAP} (a table: component | responsibility | dependencies | interfaces | data flow | control flow | failure
boundaries | security boundaries | resource boundaries, plus the components block). Every source directory belongs to a
component: ${JSON.stringify(sourceDirs)}.
${fence('inventory', inv.json, 30000)}`,
  { label: 'map', phase: 'Map', schema: RESULT, agentType: ROLE.researcher })

phase('Check')
let components = []
let problems = ['not checked']
for (let attempt = 1; attempt <= 3; attempt++) {
  const got = await agent(common('READ-ONLY.') + `
Read ${MAP}. Find the block after "<!-- components -->" and return its JSON array as "components", verbatim. If missing or
invalid, return an empty array.`,
    { label: `extract:components:a${attempt}`, phase: 'Check', schema: COMPONENTS, agentType: ROLE.readonly })
  components = got?.components ?? []
  problems = []
  if (!components.length) problems.push('the components block is missing or empty')
  for (const c of components) {
    for (const f of FIELDS) if (!(Array.isArray(c[f]) ? c[f].length || f === 'dependencies' : String(c[f] ?? '').trim())) problems.push(`${clean(c.component, 40)}: field ${f} is empty`)
    for (const p of c.paths ?? []) if (!safeScopePath(p)) problems.push(`${clean(c.component, 40)}: unsafe path ${clean(p, 80)}`)
  }
  const paths = [...new Set(components.flatMap(c => (c.paths ?? []).filter(p => safeScopePath(p))))]
  if (paths.length) {
    const exist = await runTool(`node agents-architetura/tools/factory/inventory.mjs --check-paths '${JSON.stringify(paths)}'`, { label: `check-paths:a${attempt}`, phaseTitle: 'Check' })
    for (const p of exist?.json?.missing ?? []) problems.push(`path ${clean(p, 80)} does not exist`)
  }
  for (const d of sourceDirs) if (!paths.some(p => scopesOverlap(p, d))) problems.push(`source directory ${d} belongs to no component`)
  if (!problems.length) break
  if (attempt === 3) break
  log(`Check attempt ${attempt}: ${problems.length} problem(s) -> repairing`)
  await agent(common(OWN) + MAP_RULES + `
The script check of ${MAP} failed. Fix every problem below by reading the code — never by inventing components.
${fence('map_problems', problems, 8000)}`,
    { label: `repair:a${attempt}`, phase: 'Check', schema: RESULT, agentType: ROLE.researcher })
}

phase('Review')
const review = await reviewUntilClean({
  label: 'architecture map',
  lenses: [
    ['fidelity', `Is every statement in ${CURRENT} and ${MAP} grounded in the code it cites (open the files)? A claim the code does not
support, an aspirational description, a missing component, or a recommendation smuggled into a description is a finding.`],
    ['boundaries', `Are the control and data flows traced end to end (request -> scheduling -> workers -> training step -> result, and
the failure paths)? Are the failure, security and resource boundaries real and complete (who trusts whom, what one process
can exhaust or crash in another)?`],
  ],
    review: (name, lens, round) => agent(common('READ-ONLY adversarial reviewer.') + `
Review the architecture map (round ${round}, lens ${name}). ${lens}`,
    { label: `review:${name}:r${round}`, phase: 'Review', schema: FINDINGS, agentType: ROLE.reviewer }),
  fix: (serious, round) => agent(common(OWN) + MAP_RULES + `
Fix these findings (round ${round}) by reading the code; refute false ones in the Unknowns section with file references.
${fence('reviewer_findings', serious, 20000)}`,
    { label: `fix:r${round}`, phase: 'Review', schema: RESULT, agentType: ROLE.researcher }),
})

phase('Specialists')
const proposal = await agent(common('READ-ONLY.') + `
Decide which specialists this repository needs NOW. A specialist is justified only when the code already holds that
responsibility — cite file:line evidence and give the paths. Catalog: ${fence('catalog', SPECIALIST_CATALOG, 3000)}
A custom specialist (id custom-<name>) needs a responsibility the catalog does not cover. Everything not justified goes in
"rejected" with the reason ("no scheduler code exists yet" is a fine reason). Fewer, real specialists beat many speculative ones.
Read ${MAP} and the code. Write nothing.`,
  { label: 'specialists:decide', phase: 'Specialists', schema: SPECIALISTS, agentType: ROLE.researcher })
let specialists = validSpecialists(proposal?.specialists)
const specPaths = [...new Set(specialists.flatMap(s => s.paths))]
if (specPaths.length) {
  const exist = await runTool(`node agents-architetura/tools/factory/inventory.mjs --check-paths '${JSON.stringify(specPaths)}'`, { label: 'check-paths:specialists', phaseTitle: 'Specialists' })
  const missing = new Set(exist?.json?.missing ?? [])
  const dropped = specialists.filter(s => s.paths.some(p => missing.has(p))).map(s => s.id)
  if (dropped.length) log(`Specialists: dropped ${dropped.join(', ')} — they cite paths that do not exist`)
  specialists = specialists.filter(s => !dropped.includes(s.id))
}
await agent(common(`OWNED PATHS: ${MAP}, agents-architetura/docs/specialists/ (only).`) + `
Record the validated specialists: one file agents-architetura/docs/specialists/<id>.md each (responsibility, paths, evidence, the focus an
implementer must keep, the lens a reviewer must apply), delete specialist files not in this list, and replace the block after
"<!-- specialists -->" at the end of ${MAP} with exactly this JSON array (create the marker if absent):
${fence('specialists', specialists, 12000)}`,
  { label: 'specialists:write', phase: 'Specialists', schema: RESULT, agentType: ROLE.researcher })

let reference = null
const ref = String(args?.reference ?? '').match(REF_RE)
if (ref) {
  phase('Reference')
  const name = token(ref[2], 60)
  const dir = `.factory/reference/${name}`
  const cloned = await runTool(`test -d ${dir}/.git || gh repo clone ${ref[1]}/${ref[2]} ${dir} -- --depth 1 --quiet; node agents-architetura/tools/factory/inventory.mjs --root ${dir} --json`,
    { label: 'reference:inventory', phaseTitle: 'Reference', note: 'The reference repository is READ-ONLY: never modify, commit or push anything in it.' })
  if (cloned?.json?.files) {
    await agent(common(`OWNED PATHS: ${DESIGN_DIR}/reference/${name}.md (only). ${dir} is a READ-ONLY reference — never modify it.`) + `
Map the reference codebase in ${dir} into ${DESIGN_DIR}/reference/${name}.md: what it does, its processes and modules, how control
and data flow, how it fails, what it measured (experiments, logs, results), and — clearly separated — what a new design should
keep, what it should avoid, and why, each with file references. Start from its inventory:
${fence('reference_inventory', cloned.json, 30000)}`,
      { label: 'reference:map', phase: 'Reference', schema: RESULT, agentType: ROLE.researcher })
    reference = `${DESIGN_DIR}/reference/${name}.md`
  } else log(`Reference: ${ref[1]}/${ref[2]} could not be read — skipped (${clean(JSON.stringify(cloned ?? {}), 200)})`)
}

const ok = !problems.length && review.converged
log(`Survey: ${components.length} components, ${specialists.length} specialists${reference ? `, reference ${reference}` : ''}; map ${ok ? 'checked and reviewed' : 'NOT clean'}`)
return { ok, components: components.length, problems, review: { converged: review.converged, rounds: review.rounds }, specialists: specialists.map(s => s.id), rejected: (proposal?.rejected ?? []).map(r => clean(r.id, 40)), reference }
