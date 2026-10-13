import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, assertRoles, approved } from './helpers.mjs'

const script = built('wan-scalebench')
const TREE = 'e'.repeat(40)
const SPEC = { hardwareProbe: 'nvidia-smi -L | grep -c GPU', runCommand: 'uv run python bench/scalebench/runner/run.py', implementers: [
  { id: 'runner', dir: 'bench/scalebench/runner/', task: 'matrix runner' },
  { id: 'ddp', dir: 'bench/scalebench/baselines/ddp/', task: 'DDP baseline' },
  { id: 'net', dir: 'bench/scalebench/net/', task: 'netem profiles' },
] }
const base = {
  'gate:approval': () => approved(),
  'design': () => SPEC,
  'audit:snapshot:': () => ({ tree: TREE }),
  'audit:check:': () => ({ tree: TREE, changed: [], violations: [] }),
  'impl:': () => ({ summary: 'ok', files: [] }),
  'review:': () => ({ findings: [] }),
  'run': () => ({ ran: true, reason: '', validity: { ok: true, runs: 40, systems: ['ours', 'ddp'], problems: [] } }),
  'report': () => ({ summary: 'ok', files: [] }),
}
const run = extra => simulate(script, { args: {}, respond: byLabel({ ...base, ...extra }) })

test('valid study: disjoint implementers, reviewed, run under the machine lock, checksums validated by script', async () => {
  const { result, calls } = await run()
  assert.deepEqual([result.ok, result.applicable, result.valid], [true, true, true])
  assert.match(calls.find(c => c.opts.label === 'run').prompt, /machine-lock\.mjs run --label scalebench --wait 3600 -- uv run python bench\/scalebench\/runner\/run\.py/)
  assertRoles(assert, calls, { 'design': 'wan-researcher', 'impl:': 'wan-coder', 'review:': 'wan-reviewer', 'run': 'wan-verifier', 'report': 'wan-researcher' })
})

test('no hardware is an honest n/a; a checksum mismatch is invalid; overlapping or escaping dirs are refused', async () => {
  const na = await run({ 'run': () => ({ ran: false, reason: 'no NVIDIA GPU', validity: {} }) })
  assert.deepEqual([na.result.ok, na.result.applicable], [true, false])
  assert.ok(na.calls.find(c => c.opts.label === 'report').prompt.includes('did not run'))
  const bad = await run({ 'run': () => ({ ran: true, reason: '', validity: { ok: false, problems: ['dataOrder differs'] } }) })
  assert.equal(bad.result.ok, false)
  await assert.rejects(run({ 'design': () => ({ ...SPEC, implementers: [{ id: 'a', dir: 'bench/scalebench/x/', task: 't' }, { id: 'b', dir: 'bench/scalebench/x/y/', task: 't' }] }) }), /overlap/)
  await assert.rejects(run({ 'design': () => ({ ...SPEC, implementers: [{ id: 'a', dir: 'src/', task: 't' }] }) }), /no valid implementer split/)
})
