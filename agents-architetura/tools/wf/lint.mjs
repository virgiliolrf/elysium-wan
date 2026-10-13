#!/usr/bin/env node
// House-rule lint for the phase workflows wan-next writes (agents-architetura/tools/wf/src/wan-x-<slug>.js). An agent
// wrote the workflow, so nothing in it is trusted until it passes three layers:
//
// 1. Static: the name, the includes, and the shape of a phase: its facts, writers, lenses and gate ids handed to
//    `return await runPhase({...})` (lib/phase.js) for its own slug, with no house machinery re-implemented (no agent,
//    loop, audit or approval call of its own) and within its size budget; none of the forbidden operations (git push,
//    merges, force flags, approvals, generated files). With the repo at hand: the NEXT document exists, the gates file
//    (GATES-<slug>.json, the phase's V0..Vn) is valid and can fail, and gateIds lists exactly its gate ids.
// 2. Build: everything build.mjs enforces (pure meta, declared phases, determinism, size, parses).
// 3. Simulation, offline (sim.mjs):
//    - Without the grant, the workflow refuses after the approval check and runs no other agent.
//    - With the grant and every agent answering a minimal schema-valid "success", it terminates and reaches its pinned
//      gate (pinned.mjs <slug>). Every call carries a label, a declared phase, a schema and the factory's common() preamble.
//    - With every agent failing (null), it stops with a reasoned Error, never a TypeError.
//
//   node agents-architetura/tools/wf/lint.mjs <wan-x-name> [--json]
// Exit: 0 clean · 1 violations · 2 usage.
import { readFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildScript } from './build.mjs'
import { simulate, loadLib } from './sim.mjs'
import { validateGates, expectedGateIds } from '../gates/pinned.mjs'
import { loadCampaigns } from '../campaign/families.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = join(HERE, 'src')
export const AUTHORED = /^wan-x-[a-z0-9][a-z0-9-]{1,38}$/
const COMMON_MARK = 'Repo: the repository root (your current working directory).'
const MAX_CALLS = 400
// Paths no phase writer may own, from the factory's own lists: the factory, the harness, and docs/design (approvals,
// frozen specs, the design of record).
const { FACTORY_DIRS, HARNESS_DIRS } = loadLib(['project'], { agent: null })
export const NOT_OWNABLE = [...FACTORY_DIRS, ...HARNESS_DIRS, 'docs/design/']
const overlaps = (p, d) => p === d || p.startsWith(d) || d.startsWith(p)
export function forbiddenOwnership(paths) {
  return paths.filter(p => !p || p === '.' || p === './' || p.startsWith('/') || p.includes('..') || NOT_OWNABLE.some(d => overlaps(p, d)))
}
const ownedIn = prompt => [...prompt.matchAll(/OWNED PATHS:\s*([^\n]*?)\s*\(only/g)].flatMap(m => m[1].split(/,\s*/).map(s => s.trim()))
const REQUIRED_INCLUDES = ['hygiene', 'schemas', 'dag', 'loop', 'project', 'gates', 'triage', 'fix', 'scope', 'phase']
// A phase workflow is facts, writers, lenses and gate ids; its machinery is runPhase. The M2 phase-1 workflow re-wrote that
// machinery inline (924 lines, a 400-line FACTS block) and its review did not converge in six rounds, nearly every finding
// being about the wiring. The budget keeps the facts to what the prompts need: every agent can read the NEXT document.
export const MAX_PHASE_LINES = 350
// What runPhase owns. A phase that calls one of these itself re-implements reviewed machinery and gets it re-reviewed; a
// capability the phase needs goes into lib/phase.js through a factory PR instead.
// "Divide always" (the human's rule, 2026-10-08), from LESSONS.md: a phase of about 3 work packages and 12 gate entries
// converges under review; the M2 phase with 5 packages and 26 entries did not. Over either cap, wan-next splits the phase.
export const MAX_PHASE_WORK = 3
export const MAX_PHASE_PINNED = 12
const MACHINERY = ['agent', 'parallel', 'pipeline', 'workflow', 'reviewUntilClean', 'runFixPipeline', 'scopeSnapshot', 'scopeCheck', 'checkApprovals', 'approvedAll', 'normalizeItems', 'dominantMechanism']
const FORBIDDEN = [
  [/git\s+push/, 'runs git push (publishing goes through publish.mjs)'],
  [/gh\s+pr\s+merge/, 'merges a pull request (merging is the human\'s)'],
  [/--force(?![\w-])|--force-with-lease|\bpush\s+-f\b/, 'uses a force flag'],
  [/git\s+reset\s+--hard/, 'resets the working tree'],
  [/OWNED PATHS:[^`]*\.claude\//, 'gives an agent write access to .claude/ (generated workflows and agent definitions)'],
]
// Instructions no agent prompt may carry (checked on every prompt the simulation records).
const FORBIDDEN_IN_PROMPTS = [
  [/approve\.mjs\s+grant/, 'tells an agent to grant an approval (human-only)'],
  [/APPROVALS\.json|FROZEN\.json/, 'points an agent at the approvals or frozen records'],
]

const slugOf = name => name.replace(/^wan-x-/, '')

// The source with comments and the text of string and template literals blanked (newlines kept), so a call is found only
// where it is code: a prompt that says "the agent(s)" is not an agent() call, and `${agent(...)}` inside a template still is.
export function codeOnly(src) {
  let out = ''
  const stack = [] // template literals we are inside; each entry is the brace depth of its open ${ ... }
  let i = 0
  const blank = c => (c === '\n' ? '\n' : ' ')
  while (i < src.length) {
    const c = src[i]
    const top = stack.length ? stack[stack.length - 1] : null
    if (top !== null && top.inText) {
      if (c === '\\') { out += blank(c) + blank(src[i + 1] ?? ''); i += 2; continue }
      if (c === '`') { stack.pop(); out += c; i++; continue }
      if (c === '$' && src[i + 1] === '{') { top.inText = false; top.depth = 0; out += '${'; i += 2; continue }
      out += blank(c); i++; continue
    }
    if (c === '/' && src[i + 1] === '/') { while (i < src.length && src[i] !== '\n') { out += ' '; i++ } continue }
    if (c === '/' && src[i + 1] === '*') { const end = src.indexOf('*/', i + 2); const stop = end < 0 ? src.length : end + 2; for (; i < stop; i++) out += blank(src[i]); continue }
    if (c === "'" || c === '"') {
      out += c; i++
      while (i < src.length && src[i] !== c && src[i] !== '\n') { if (src[i] === '\\') { out += ' '; i++ } out += blank(src[i] ?? ''); i++ }
      if (i < src.length) { out += src[i]; i++ }
      continue
    }
    if (c === '`') { stack.push({ inText: true, depth: 0 }); out += c; i++; continue }
    if (top !== null && c === '{') top.depth++
    if (top !== null && c === '}') { if (top.depth === 0) { top.inText = true; out += c; i++; continue } top.depth-- }
    out += c; i++
  }
  return out
}
export const nextDocFor = name => `docs/design/NEXT-${slugOf(name)}.md`
export const gatesFor = name => `docs/design/GATES-${slugOf(name)}.json`
export const pinnedFor = name => `node agents-architetura/tools/gates/pinned.mjs ${slugOf(name)} --json`
export const builtFor = name => `.claude/workflows/${name}.js`

export function lintStatic(src, name) {
  const errors = []
  if (!AUTHORED.test(name)) errors.push(`name ${JSON.stringify(name)} must match ${AUTHORED}`)
  for (const inc of REQUIRED_INCLUDES) if (!new RegExp(`^// @include ${inc}\\s*$`, 'm').test(src)) errors.push(`missing // @include ${inc}`)
  const code = src.replace(/^\s*\/\/.*$/gm, '')
  const calls = codeOnly(src)
  if ((calls.match(/\brunPhase\(/g) ?? []).length !== 1 || !/^return await runPhase\(\{/m.test(calls)) errors.push('must end with `return await runPhase({...})`, called once: the house machinery runs the phase (lib/phase.js)')
  if (!new RegExp(`\\bslug:\\s*'${slugOf(name)}'`).test(code)) errors.push(`runPhase must be given slug: '${slugOf(name)}' (its NEXT document, gates, pinned command and built workflow derive from it)`)
  for (const fn of MACHINERY) if (new RegExp(`(?<![\\w.$])${fn}\\(`).test(calls)) errors.push(`re-implements the house machinery: calls ${fn}() itself (runPhase does; a capability the phase lacks goes into lib/phase.js through a factory PR)`)
  const lines = src.split('\n').length
  if (lines > MAX_PHASE_LINES) errors.push(`is ${lines} lines; the budget is ${MAX_PHASE_LINES}: keep FACTS to what the prompts need (every agent can read ${nextDocFor(name)}) and leave the machinery to runPhase`)
  for (const [re, why] of FORBIDDEN) if (re.test(code)) errors.push(`forbidden: ${why}`)
  // The grant command may only be printed for the human (a thrown reason, a log line, the returned next step), on any path.
  code.split('\n').forEach((line, i) => {
    const outsideHumanText = line.replace(/(throw new Error\(|\blog\(|\bnext:\s*)`[^`]*`/g, '')
    if (/approve\.mjs\s+grant/.test(outsideHumanText)) errors.push(`forbidden: line ${i + 1} puts the grant command where an agent would read it (human-only)`)
    if (/APPROVALS\.json|FROZEN\.json/.test(line)) errors.push(`forbidden: line ${i + 1} names the approvals or frozen records`)
  })
  return errors
}

// A minimal value that satisfies a JSON schema: required properties only, empty arrays, true booleans.
export function synthesize(schema) {
  if (!schema || typeof schema !== 'object') return null
  if (Array.isArray(schema.enum)) return schema.enum[0]
  switch (schema.type) {
    case 'object': return Object.fromEntries((schema.required ?? []).map(k => [k, synthesize(schema.properties?.[k])]))
    case 'array': return schema.minItems > 0 ? [synthesize(schema.items)] : []
    case 'string': return 'x'
    case 'boolean': return true
    case 'number': case 'integer': return schema.minimum ?? 0
    default: return null
  }
}

const grantFor = prompt => {
  const m = prompt.match(/approve\.mjs check (.+?) --json/)
  const docs = m ? m[1].trim().split(/\s+/) : []
  return { approved: true, docs: docs.map(doc => ({ doc, approved: true })) }
}
const isApprovalCall = prompt => /approve\.mjs check /.test(prompt)
const anyWorkflow = new Proxy({}, { get: () => async () => ({ ok: true, green: true }) })
// A scope audit's minimal success is a real-looking tree with no violations (an empty tree id would read as a failure).
const SIM_TREE = '0'.repeat(40)
const succeed = (prompt, opts) => (isApprovalCall(prompt) ? grantFor(prompt)
  : /^audit:(snapshot|check):/.test(opts?.label ?? '') ? { tree: SIM_TREE, changed: [], violations: [] } : synthesize(opts?.schema))
const programmingError = e => e instanceof TypeError || e instanceof ReferenceError || e instanceof SyntaxError || e instanceof RangeError

// Runs the simulation; past MAX_CALLS the responder parks every further call (a never-settling promise), so a loop that
// would spin forever on swallowed failures stops, and the run reports non-termination instead of hanging the lint.
async function run(script, respond) {
  let calls = 0
  let abort
  const aborted = new Promise(resolve => { abort = resolve })
  const capped = (prompt, opts, index) => {
    if (++calls > MAX_CALLS) {
      abort()
      return new Promise(() => {})
    }
    return respond(prompt, opts, index)
  }
  try {
    const out = await Promise.race([
      simulate(script, { respond: capped, workflows: anyWorkflow }),
      aborted.then(() => { throw new Error(`does not terminate: more than ${MAX_CALLS} agent calls`) }),
    ])
    return { result: out.result, error: null }
  } catch (e) {
    return { result: null, error: e }
  }
}

export async function lintDynamic(script, meta) {
  const errors = []
  const declared = new Set((meta.phases ?? []).map(p => p.title))

  // Without the grant: only the approval check runs, then a reasoned refusal.
  const denied = []
  const checked = []
  const refused = await run(script, (prompt, opts) => {
    denied.push(opts?.label)
    if (isApprovalCall(prompt)) checked.push(...grantFor(prompt).docs.map(d => d.doc))
    return isApprovalCall(prompt) ? { approved: false, docs: [] } : succeed(prompt, opts)
  })
  if (!refused.error) errors.push('runs to completion without the human grant')
  else if (programmingError(refused.error)) errors.push(`without the grant it crashes: ${refused.error.message}`)
  if (!denied.length) errors.push('stops before checking its grant: the approval check must be its first agent call')
  for (const doc of [nextDocFor(meta.name), gatesFor(meta.name), builtFor(meta.name)]) if (denied.length && !checked.includes(doc)) errors.push(`its approval check does not cover ${doc}`)
  if (denied.length > 1) errors.push(`without the grant it still ran ${denied.length - 1} agent(s) after the approval check`)

  // With the grant and schema-valid success everywhere: terminates and reaches the pinned gate.
  const calls = []
  const granted = await run(script, (prompt, opts) => {
    calls.push({ prompt, opts })
    return succeed(prompt, opts)
  })
  if (granted.error) errors.push(`under all-success answers it stops early: ${granted.error.message}`)
  if (!calls.some(c => c.prompt.includes(pinnedFor(meta.name)))) errors.push(`under all-success answers it never reaches its pinned gate (${pinnedFor(meta.name)})`)
  calls.forEach((c, i) => {
    const o = c.opts ?? {}
    const where = `call ${i + 1} (${o.label ?? 'no label'})`
    if (!o.label) errors.push(`${where}: no label`)
    if (!o.phase || !declared.has(o.phase)) errors.push(`${where}: phase ${JSON.stringify(o.phase)} is not declared in meta.phases`)
    if (!o.schema || typeof o.schema !== 'object') errors.push(`${where}: no structured-output schema`)
    if (!c.prompt.includes(COMMON_MARK)) errors.push(`${where}: prompt does not start from common() (house rules, lessons, scope)`)
    for (const [re, why] of FORBIDDEN_IN_PROMPTS) if (!isApprovalCall(c.prompt) && re.test(c.prompt)) errors.push(`${where}: ${why}`)
    const bad = forbiddenOwnership(ownedIn(c.prompt))
    if (bad.length) errors.push(`${where}: gives an agent write access to ${bad.join(', ')} (factory, harness, docs/design or unsafe paths)`)
  })
  if (granted.result && typeof granted.result.ok !== 'boolean') errors.push('does not return { ok: boolean, ... }')
  const writers = calls.filter(c => (c.opts?.label ?? '').startsWith('impl:')).length
  if (writers > MAX_PHASE_WORK) errors.push(`runs ${writers} work packages; a phase has at most ${MAX_PHASE_WORK}: split the objective across phases`)

  // Every agent failing: a reasoned stop, never a programming error.
  // every agent fails; the scope audit (a tool, not a judgment) still answers, so the code behind it is exercised
  const failing = await run(script, (prompt, opts) => (isApprovalCall(prompt) || /^audit:(snapshot|check):/.test(opts?.label ?? '') ? succeed(prompt, opts) : null))
  if (failing.error && programmingError(failing.error)) errors.push(`when agents fail it crashes instead of stopping with a reason: ${failing.error.message}`)
  return errors
}

// The reference PR's binding audit and tripwire travel from the NEXT document into the workflow: the document has both
// sections, and every BA-<n> it lists reaches the code that runs (FACTS / WORK[].binding), so no ruling is dropped.
export function nextProblems(nextText, src, name) {
  const errors = []
  for (const h of ['Binding audit', 'Tripwire']) if (!new RegExp(`^#{1,3}\\s*${h}\\b`, 'mi').test(nextText)) errors.push(`${nextDocFor(name)} has no "# ${h}" section`)
  const ids = [...new Set(nextText.match(/\bBA-\d{1,4}\b/g) ?? [])]
  const missing = ids.filter(id => !new RegExp(`\\b${id}\\b`).test(src))
  if (missing.length) errors.push(`binding audit items of ${nextDocFor(name)} never reach the workflow: ${missing.join(', ')}`)
  return errors
}

// Every phase verdict runs the product rule's rungs and the whole suite once (full is on demand everywhere else).
export const PHASE_GATE_RUNGS = ['oracle', 'identity', 'full']

// gateIds: [...] as the phase source writes it (string literals only), or null when it has none.
export function gateIdsOf(src) {
  const m = /\bgateIds:\s*\[([^\]]*)\]/.exec(src ?? '')
  return m ? [...m[1].matchAll(/'([^']*)'|"([^"]*)"/g)].map(x => x[1] ?? x[2]) : null
}

// The phase's V0..Vn: valid against the real rungs and campaigns, named for this phase, and able to fail. With the source,
// its gateIds must be exactly the ids pinned.mjs reports for the file: runPhase refuses a verdict that lacks one.
export function gatesProblems(root, name, src = null) {
  const path = join(root, gatesFor(name))
  if (!existsSync(path)) return [`its pinned gates ${gatesFor(name)} do not exist`]
  let gates
  try { gates = JSON.parse(readFileSync(path, 'utf8')) } catch (e) { return [`its pinned gates ${gatesFor(name)} are not JSON: ${e.message}`] }
  const errors = []
  try {
    const rungIds = JSON.parse(readFileSync(join(root, 'agents-architetura/tools/gates/rungs.json'), 'utf8')).rungs.map(r => r.id)
    errors.push(...validateGates(gates, { rungIds, campaigns: loadCampaigns(root) }).map(e => `${gatesFor(name)}: ${e}`))
  } catch (e) { errors.push(`${gatesFor(name)} could not be validated: ${e.message}`) }
  if (gates?.milestone !== slugOf(name)) errors.push(`${gatesFor(name)}: "milestone" must be the phase slug ${JSON.stringify(slugOf(name))}`)
  if ((gates?.pinned ?? []).length > MAX_PHASE_PINNED) errors.push(`${gatesFor(name)} has ${gates.pinned.length} pinned entries; a phase has at most ${MAX_PHASE_PINNED}: split the objective across phases`)
  if (!(gates?.rungs ?? []).length && !(gates?.pinned ?? []).length) errors.push(`${gatesFor(name)} requires nothing (no rungs, no pinned commands) — a gate that cannot fail`)
  for (const r of PHASE_GATE_RUNGS) if (!(gates?.rungs ?? []).includes(r)) errors.push(`${gatesFor(name)}: "rungs" must include ${r} (${r === 'full' ? 'the one whole-suite session of the phase: everyday ladders skip it' : 'the product rule'})`)
  if (src !== null) {
    const listed = gateIdsOf(src)
    const expected = expectedGateIds(gates)
    if (!listed) errors.push(`gateIds is missing: list every gate id of ${gatesFor(name)} (${expected.join(', ')})`)
    else {
      const missing = expected.filter(id => !listed.includes(id))
      const extra = listed.filter(id => !expected.includes(id))
      if (missing.length) errors.push(`gateIds lacks ${missing.join(', ')} (gates of ${gatesFor(name)})`)
      if (extra.length) errors.push(`gateIds names ${extra.join(', ')}, which ${gatesFor(name)} does not define`)
    }
  }
  return errors
}

export async function lintWorkflow(src, name, { root = null } = {}) {
  const errors = lintStatic(src, name)
  if (root && !existsSync(join(root, nextDocFor(name)))) errors.push(`its NEXT document ${nextDocFor(name)} does not exist`)
  else if (root) errors.push(...nextProblems(readFileSync(join(root, nextDocFor(name)), 'utf8'), src, name))
  if (root) errors.push(...gatesProblems(root, name, src))
  let built
  try {
    built = buildScript(src, { name })
  } catch (e) {
    return { ok: false, errors: [...errors, `build: ${e.message}`] }
  }
  errors.push(...await lintDynamic(built.script, built.meta))
  return { ok: !errors.length, errors }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const name = argv.find(a => !a.startsWith('--'))
  if (!name) {
    console.error('usage: lint.mjs <wan-x-name> [--json]')
    process.exit(2)
  }
  const path = join(SRC, `${name}.js`)
  const r = existsSync(path) ? await lintWorkflow(readFileSync(path, 'utf8'), name, { root: join(HERE, '..', '..', '..') }) : { ok: false, errors: [`no source ${path}`] }
  if (argv.includes('--json')) console.log(JSON.stringify(r, null, 2))
  else for (const e of r.errors) console.error(`wf-lint: ${name}: ${e}`)
  if (!argv.includes('--json')) console.log(`wf-lint: ${name}: ${r.ok ? 'clean' : `${r.errors.length} violation(s)`}`)
  process.exit(r.ok ? 0 : 1)
}
