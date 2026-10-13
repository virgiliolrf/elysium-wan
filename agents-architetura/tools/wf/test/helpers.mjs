import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildScript } from '../build.mjs'

const SRC = join(dirname(fileURLToPath(import.meta.url)), '..', 'src')

// Builds the workflow from source in memory, so tests never depend on a stale .claude/workflows file.
export const built = name => buildScript(readFileSync(join(SRC, `${name}.js`), 'utf8'), { name }).script

export const ladder = (green, failing = []) => ({
  allGreen: green,
  rungs: [{ rung: 'factory', status: 'pass' }, ...failing.map(r => ({ rung: r, status: 'fail', detail: 'boom' }))],
  missingRequired: [],
})

// A finding in the adversarial-review contract: validated against the current code, with location,
// evidence and reproduction. Pass validated=false to model a hallucinated finding.
export const finding = (severity = 'major', extra = {}) => ({
  severity, location: 'src/x.py:1', problem: 'p', evidence: 'quoted code', reproduction: 'call f(0)', recommendedFix: 'fix', validatedAgainstCurrentCode: true, ...extra,
})

export const planTask = (id, extra = {}) => ({
  id, title: id, objective: `deliver ${id}`, workstream: 'transport', deps: [], scope: [`src/transport/${id}.py`],
  acceptance: ['a'], verify: [`uv run pytest -q tests/unit/transport/test_${id}.py`], rungs: ['unit'], invariants: [], risk: 'low', ...extra,
})

// The approval gate's answer when a human granted the design (agents-architetura/tools/factory/approve.mjs check --json).
export const approved = (docs = ['docs/design/DESIGN.md']) => ({ approved: true, docs: docs.map(doc => ({ doc, approved: true })) })
export const notApproved = (doc = 'docs/design/DESIGN.md', reason = 'never approved — silence is not approval') => ({ approved: false, docs: [{ doc, approved: false, reason }] })

// Every call whose label starts with a key must have been routed to exactly that agent role.
export const assertRoles = (assert, calls, roles) => {
  for (const c of calls) {
    const prefix = Object.keys(roles).find(p => c.opts.label.startsWith(p))
    if (prefix) assert.equal(c.opts.agentType, roles[prefix], `${c.opts.label} should run as ${roles[prefix]}`)
  }
  for (const p of Object.keys(roles)) assert.ok(calls.some(c => c.opts.label.startsWith(p)), `no call with label prefix ${p}`)
}

// agents-architetura/tools/factory/publish.mjs as the verifier pastes it: a publisher that succeeds, or exits with `exitCode`.
export const PUBLISHED = { ok: true, branch: 'factory/demo', resumed: false, committed: true, sha: 'abc1234def', pushed: true, pr: { number: 1, url: 'https://github.com/o/r/pull/1' } }
export const publisher = (exitCode = 0, json = PUBLISHED) => () => ({ json: exitCode === 0 ? json : { ok: false, code: exitCode, reason: exitCode === 3 ? 'refused to commit: src/a.py:1 (GitHub token) — files left unstaged' : 'push failed (commit kept locally; the next publish retries)' }, exitCode })
export const payloadOf = prompt => JSON.parse(Buffer.from(/--b64 (\S+)/.exec(prompt)[1], 'base64').toString('utf8'))
export const publishLabels = calls => calls.filter(c => (c.opts.label ?? '').startsWith('publish:')).map(c => c.opts.label)

// Serves a milestone plan to extractPlan's piecewise copy: the index, then the entries each call names. planOf is a plan or
// (prompt, opts) => plan, so a test can serve different plans to different extractions by label.
export const servePlan = planOf => (p, o, ...rest) => {
  const plan = typeof planOf === 'function' ? planOf(p, o, ...rest) : planOf
  const list = f => plan?.[f] ?? []
  if (o.label.endsWith(':index')) return { milestone: plan?.milestone ?? '', taskIds: list('tasks').map(t => t.id), bindingIds: list('binding').map(b => b.id), tripwireIds: list('tripwires').map(t => t.id) }
  const field = ['tasks', 'binding', 'tripwires'].find(f => o.label.includes(`:${f}:`))
  const ids = JSON.parse(/IDS: (\[.*\])/.exec(p)[1])
  return { [field]: ids.map(id => list(field).find(x => x.id === id)).filter(Boolean) }
}
