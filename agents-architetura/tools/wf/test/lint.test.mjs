import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { lintWorkflow, lintStatic, lintDynamic, synthesize, nextDocFor, gatesFor, pinnedFor, gatesProblems, gateIdsOf, MAX_PHASE_LINES } from '../lint.mjs'
import { buildScript } from '../build.mjs'

const TEMPLATE = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'docs', 'templates', 'wan-x-workflow.js'), 'utf8')
const NAME = 'wan-x-example'
const edit = (from, to) => {
  assert.ok(TEMPLATE.includes(from), `template lost: ${from}`)
  return TEMPLATE.replace(from, to)
}
const errorsOf = async src => (await lintWorkflow(src, NAME)).errors

test('the house template is clean: it is what wan-next tells the author to start from', async () => {
  assert.deepEqual(await errorsOf(TEMPLATE), [])
  assert.equal(nextDocFor('wan-x-zbv'), 'docs/design/NEXT-zbv.md')
  assert.equal(gatesFor('wan-x-zbv'), 'docs/design/GATES-zbv.json')
  assert.equal(pinnedFor('wan-x-zbv'), 'node agents-architetura/tools/gates/pinned.mjs zbv --json')
})

// A workflow written without runPhase (the shape wf-lint refuses statically) still meets the dynamic layer: it is how the
// simulation's own checks are exercised.
const ROGUE = `export const meta = { name: 'wan-x-example', description: 'd', whenToUse: 'w', phases: [{ title: 'Gate', detail: 'g' }, { title: 'Verify', detail: 'v' }] }
// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project
// @include gates
phase('Gate')
const docs = ['docs/design/NEXT-example.md', 'docs/design/GATES-example.json', '.claude/workflows/wan-x-example.js']
const grant = await checkApprovals(docs, 'gate:approval', 'Gate')
if (!approvedAll(grant, docs)) throw new Error('not approved')
phase('Verify')
const v = await agent(common('You run ALONE.') + 'Run exactly: node agents-architetura/tools/gates/pinned.mjs example --json', { label: 'verify:r0', phase: 'Verify', schema: RESULT, agentType: ROLE.verifier })
return { ok: v !== null }`
const rogue = (from, to) => {
  assert.ok(ROGUE.includes(from), `rogue lost: ${from}`)
  return ROGUE.replace(from, to)
}
const dynamicErrors = src => { const b = buildScript(src, { name: NAME }); return lintDynamic(b.script, b.meta) }

test('a phase that runs its own machinery instead of runPhase is refused statically, whatever it calls', () => {
  const e = lintStatic(ROGUE, NAME)
  assert.ok(e.some(x => /must end with `return await runPhase/.test(x)), e.join('\n'))
  for (const fn of ['agent', 'checkApprovals', 'approvedAll']) assert.ok(e.some(x => x.includes(`calls ${fn}() itself`)), fn)
  for (const call of ['reviewUntilClean({})', 'runFixPipeline([])', 'scopeSnapshot()', 'parallel([])', 'workflow(\'wan-fix\')']) {
    const src = edit('return await runPhase({', `await ${call}\nreturn await runPhase({`)
    assert.ok(lintStatic(src, NAME).some(x => /re-implements the house machinery/.test(x)), call)
  }
  assert.ok(lintStatic(edit("slug: 'example',", "slug: 'other',"), NAME).some(x => /runPhase must be given slug: 'example'/.test(x)))
  assert.ok(lintStatic(edit('return await runPhase({', 'const r = await runPhase({').replace(/\}\)\s*$/, '})\nreturn r'), NAME).some(x => /must end with `return await runPhase/.test(x)))
})

test('only code counts as a call: prompt text that mentions agent( is not one, a call hidden in ${...} is', async () => {
  const { codeOnly } = await import('../lint.mjs')
  const prose = edit("task: '<the change, in terms of the facts above>'", "task: 'tell the agent(s) which call agent() makes'")
  assert.ok(!lintStatic(prose, NAME).some(x => /calls agent\(\) itself/.test(x)))
  const hidden = edit('- objective: <one sentence>', "- objective: ${await agent('x', {})}")
  assert.ok(lintStatic(hidden, NAME).some(x => /calls agent\(\) itself/.test(x)))
  assert.equal(codeOnly("a('x(') + `t ${b(`u ${c('v')}`)} w` // d(\n"), "a('  ') + `  ${b(`  ${c(' ')}`)}  `      \n")
})

test('a phase with more than 3 work packages or 12 pinned gate entries is refused: the objective is split across phases', async () => {
  const four = edit("{ key: 'part-a', owns: ['src/example/a/'], task: '<the change, in terms of the facts above>', binding: ['BA-1'] },",
    ['a', 'b', 'c', 'd'].map(k => `{ key: 'part-${k}', owns: ['src/example/${k}/'], task: 't', binding: ['BA-1'] },`).join('\n    '))
  assert.ok((await errorsOf(four)).some(x => /runs 4 work packages; a phase has at most 3/.test(x)))
  const root = mkdtempSync(join(tmpdir(), 'lint-cap-'))
  const write = (f, b) => { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), JSON.stringify(b)) }
  write('agents-architetura/tools/gates/rungs.json', { rungs: ['oracle', 'identity', 'full'].map(id => ({ id, steps: [] })) })
  write('agents-architetura/tools/campaign/campaigns.json', { campaigns: [] })
  write('docs/design/GATES-example.json', { milestone: 'example', rungs: ['oracle', 'identity', 'full'], pinned: Array.from({ length: 13 }, (_, i) => ({ id: `p${i}`, cmd: 'true' })) })
  assert.match(gatesProblems(root, NAME).join(), /has 13 pinned entries; a phase has at most 12/)
})

test('a phase over its size budget is refused: the facts are what the prompts need, the machinery is runPhase', () => {
  const long = edit("- objective: <one sentence>", `- objective: <one sentence>\n${'- filler\n'.repeat(MAX_PHASE_LINES)}`)
  assert.ok(lintStatic(long, NAME).some(x => new RegExp(`is \\d+ lines; the budget is ${MAX_PHASE_LINES}`).test(x)))
  assert.ok(TEMPLATE.split('\n').length < MAX_PHASE_LINES / 2)
})

test('a forbidden operation, a wrong name or a missing include is refused statically', () => {
  for (const [bad, why] of [
    ['Run: git push origin HEAD', /git push/],
    ['then gh pr merge 1', /merges a pull request/],
    ['git push --force', /force flag/],
    ['git reset --hard HEAD', /resets the working tree/],
  ]) {
    const src = edit("task: '<the change, in terms of the facts above>'", `task: '${bad}'`)
    assert.ok(lintStatic(src, NAME).some(e => why.test(e)), `${bad} not caught`)
  }
  assert.ok(lintStatic(TEMPLATE, 'thread-fix').some(e => /must match/.test(e)))
  for (const inc of ['gates', 'phase', 'fix']) assert.ok(lintStatic(edit(`// @include ${inc}\n`, ''), NAME).some(e => e === `missing // @include ${inc}`), inc)
})

test('the simulation refuses a workflow that runs without its grant, runs agents before it, or whose grant misses a document', async () => {
  const e1 = await dynamicErrors(rogue("if (!approvedAll(grant, docs)) throw new Error('not approved')\n", ''))
  assert.ok(e1.some(e => /runs to completion without the human grant/.test(e)), e1.join('\n'))
  const e2 = await dynamicErrors(rogue("phase('Gate')\n", "phase('Gate')\nawait agent(common('READ-ONLY.') + 'peek', { label: 'peek', phase: 'Gate', schema: RESULT, agentType: ROLE.readonly })\n"))
  assert.ok(e2.some(e => /still ran 1 agent\(s\) after the approval check/.test(e)), e2.join('\n'))
  const e3 = await dynamicErrors(rogue(", '.claude/workflows/wan-x-example.js']", ']'))
  assert.ok(e3.some(e => /approval check does not cover \.claude\/workflows\/wan-x-example\.js/.test(e)), e3.join('\n'))
  const e4 = await dynamicErrors(rogue("phase('Gate')\n", "phase('Gate')\nthrow new Error('nothing to do')\n"))
  assert.ok(e4.some(e => /stops before checking its grant/.test(e)), e4.join('\n'))
})

test('the simulation catches calls without a schema, outside common(), in an undeclared phase, or telling an agent to grant', async () => {
  assert.ok((await dynamicErrors(rogue('schema: RESULT, agentType: ROLE.verifier', 'agentType: ROLE.verifier'))).some(e => /no structured-output schema/.test(e)))
  assert.ok((await dynamicErrors(rogue("agent(common('You run ALONE.') + 'Run", "agent('Run"))).some(e => /does not start from common\(\)/.test(e)))
  assert.ok((await dynamicErrors(rogue("'Run exactly:", "'Then run approve.mjs grant x. Run exactly:"))).some(e => /tells an agent to grant an approval/.test(e)))
  assert.throws(() => buildScript(rogue("phase: 'Verify', schema", "phase: 'Elsewhere', schema"), { name: NAME }), /phase titles used but not declared in meta\.phases: Elsewhere/)
})

test('a workflow that never reaches its gate, or never terminates, is reported, not waited on', async () => {
  assert.ok((await dynamicErrors(rogue("phase('Verify')\n", "phase('Verify')\nreturn { ok: false }\n"))).some(x => /never reaches its pinned gate \(node agents-architetura\/tools\/gates\/pinned\.mjs example --json\)/.test(x)))
  const forever = rogue("phase('Verify')\n", "phase('Verify')\nfor (;;) await parallel([() => agent(common('READ-ONLY.') + 'spin', { label: 'spin', phase: 'Verify', schema: RESULT, agentType: ROLE.readonly })])\n")
  assert.ok((await dynamicErrors(forever)).some(x => /does not terminate/.test(x)))
})

test('the template runs clean through all three simulations; runPhase refuses a bad configuration before any agent', async () => {
  assert.deepEqual(await errorsOf(TEMPLATE), [])
  const e = await errorsOf(TEMPLATE.replace(/lenses: \[[\s\S]*?\n  \],\n/, "lenses: [['only', 'one']],\n"))
  assert.ok(e.some(x => /lenses must be at least three/.test(x)) && e.some(x => /stops before checking its grant/.test(x)), e.join('\n'))
})

test('write access is checked on the real prompts: owned paths cannot reach the factory, harness or docs/design', async () => {
  for (const owns of ['.claude/agents/', 'agents-architetura/tools/', 'docs/design/', 'tests/chaos/']) {
    const e = await errorsOf(edit("owns: ['src/example/a/']", `owns: ['${owns}']`))
    assert.ok(e.some(x => x.includes(`gives an agent write access to ${owns}`)), `${owns}: ${e.join('\n')}`)
  }
  const e = await errorsOf(edit("owns: ['src/example/a/']", "owns: ['../outside/']"))
  assert.ok(e.some(x => /owns must be a non-empty list of safe repo-relative paths/.test(x)), e.join('\n'))
})

test('the grant command is allowed only inside human-facing text', () => {
  const sneaky = edit("task: '<the change, in terms of the facts above>'", "task: 'Run approve.mjs grant docs/design/NEXT-example.md first'")
  assert.ok(lintStatic(sneaky, NAME).some(e => /grant command where an agent would read it/.test(e)))
  assert.ok(!lintStatic(TEMPLATE, NAME).some(e => /grant command/.test(e)))
})

test('the CLI requires the NEXT document to exist', async () => {
  const missing = await lintWorkflow(TEMPLATE, NAME, { root: '/nonexistent-root' })
  assert.ok(missing.errors.some(x => /NEXT document docs\/design\/NEXT-example\.md does not exist/.test(x)))
})

test('synthesize builds the minimal schema-valid success', () => {
  assert.deepEqual(synthesize({ type: 'object', required: ['a', 'b', 'c'], properties: { a: { type: 'boolean' }, b: { type: 'array', items: { type: 'string' } }, c: { type: 'string', enum: ['pass', 'fail'] } } }), { a: true, b: [], c: 'pass' })
})

test('with the repo at hand, the phase gates must exist, validate against the real rungs, name the phase and be able to fail', () => {
  const root = mkdtempSync(join(tmpdir(), 'lint-gates-'))
  const write = (f, b) => { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), typeof b === 'string' ? b : JSON.stringify(b)) }
  write('agents-architetura/tools/gates/rungs.json', { rungs: [{ id: 'unit', steps: [] }, { id: 'oracle', steps: [] }, { id: 'identity', steps: [] }, { id: 'full', steps: [] }] })
  write('agents-architetura/tools/campaign/campaigns.json', { campaigns: [] })
  assert.match(gatesProblems(root, NAME).join(), /GATES-example\.json do not exist/)
  write('docs/design/GATES-example.json', '{')
  assert.match(gatesProblems(root, NAME).join(), /not JSON/)
  write('docs/design/GATES-example.json', { milestone: 'other', rungs: ['ghost'] })
  const bad = gatesProblems(root, NAME).join('\n')
  assert.match(bad, /unknown rung "ghost"/)
  assert.match(bad, /must be the phase slug "example"/)
  write('docs/design/GATES-example.json', { milestone: 'example' })
  assert.match(gatesProblems(root, NAME).join(), /requires nothing/)
  write('docs/design/GATES-example.json', { milestone: 'example', rungs: ['unit', 'oracle'], pinned: [{ id: 'x', cmd: 'true' }] })
  assert.match(gatesProblems(root, NAME).join('\n'), /"rungs" must include identity[\s\S]*"rungs" must include full \(the one whole-suite session/)
  write('docs/design/GATES-example.json', { milestone: 'example', rungs: ['unit', 'oracle', 'identity', 'full'], pinned: [{ id: 'step', cmd: 'uv run python bench/step.py', repeat: 5, metric: { pattern: 'step_s=([0-9.]+)', op: '<=', value: 1.05 } }] })
  assert.deepEqual(gatesProblems(root, NAME), [])
  // with the source: gateIds must be exactly the gate ids pinned.mjs reports for the file
  assert.deepEqual(gateIdsOf(TEMPLATE), ['rung:oracle', 'rung:identity', 'rung:full', 'pinned:step'])
  const ids = gatesProblems(root, NAME, TEMPLATE).join('\n')
  assert.match(ids, /gateIds lacks rung:unit/)
  write('docs/design/GATES-example.json', { milestone: 'example', rungs: ['oracle', 'identity', 'full'], pinned: [{ id: 'step', cmd: 'true' }] })
  assert.deepEqual(gatesProblems(root, NAME, TEMPLATE), [])
  assert.match(gatesProblems(root, NAME, TEMPLATE.replace("'pinned:step'", "'pinned:step', 'pinned:ghost'")).join(), /gateIds names pinned:ghost, which docs\/design\/GATES-example\.json does not define/)
  assert.match(gatesProblems(root, NAME, TEMPLATE.replace(/gateIds: \[[^\]]*\],?/, '')).join(), /gateIds is missing/)
})

test('the NEXT document carries a binding audit and a tripwire, and every BA item reaches the workflow', async () => {
  const { nextProblems } = await import('../lint.mjs')
  const doc = '# Objective\nx\n# Binding audit\n| BA-1 | owner table | sched.py:40 | rebuilt |\n| BA-2 | RNG | loop.py:9 | carried |\n# Tripwire\nnone: one leg\n'
  assert.deepEqual(nextProblems(doc, TEMPLATE.replace("binding: ['BA-1']", "binding: ['BA-1', 'BA-2']"), NAME), [])
  assert.match(nextProblems(doc, TEMPLATE, NAME).join(), /never reach the workflow: BA-2/)
  const bare = nextProblems('# Objective\nx\n', TEMPLATE, NAME)
  assert.ok(bare.some(e => /no "# Binding audit" section/.test(e)) && bare.some(e => /no "# Tripwire" section/.test(e)), bare.join('; '))
})
