import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built, ladder, assertRoles, approved } from './helpers.mjs'

const script = built('wan-fuzz')
const SIG = 'py-ValueError|decode_len@src/wire/decode.py'
const crash = { signature: SIG, file: '/repo/.factory/fuzz/fuzz-wire/crashes/crash-a', flaky: false }
const flakyCrash = { signature: 'fuzz-flaky|abc123', file: '/repo/.factory/fuzz/fuzz-wire/crashes/crash-d', flaky: true }
const base = {
  'gate:approval': () => approved(),
  'status': () => ({ json: { ok: true, runs: 2, cpuHours: 3 }, exitCode: 0 }),
  'campaign': () => ({ json: { ok: false, cpuHours: 4.2, newUnique: [crash, flakyCrash] }, exitCode: 1 }),
  'diagnose:': () => ({ minimizedInput: 'tests/fuzz/regressions/fuzz-wire/x/min.bin', reproCommand: 'uv run python tests/fuzz/targets/wire.py tests/fuzz/regressions/fuzz-wire/x/min.bin', scope: ['src/wire/decode.py'], suspectedCause: 'length read before bounds check (decode.py:9)' }),
  'propose:': () => ({ fix: 'check len first', rationale: 'bounds' }),
  'vote:': () => ({ approve: true, reasons: 'ok' }),
  'apply:': () => ({ summary: 'applied', files: ['src/wire/decode.py'], evidence: [{ claim: 'repro', command: 'x', exitCode: 0, output: 'ok' }] }),
  'verify': () => ({ items: [{ id: 'py-ValueError-decode_len-src-wire-decode-py', reproRuns: 20, reproFailures: 0, loadFailed: 0 }], ladder: ladder(true), refuzz: { ok: true, newUnique: [] } }),
  'amplify:': () => ({ json: { clean: true, findings: [] }, exitCode: 0 }),
  'report': () => ({ summary: 'ok', files: [] }),
}
const run = (extra = {}, args = {}) => simulate(script, { args: { campaign: 'fuzz-wire', ...args }, respond: byLabel({ ...base, ...extra }) })

test('a new unique crash is minimized, fixed by vote, verified 20/20 + under load + re-fuzz; flaky crashes are amplified', async () => {
  const { result, calls } = await run()
  assert.equal(result.ok, true, JSON.stringify(result.open))
  assert.deepEqual(result.fixed, ['py-ValueError-decode_len-src-wire-decode'])
  assert.deepEqual(result.flaky, [{ signature: 'fuzz-flaky|abc123', reproducedUnderAmplifier: false }])
  const amp = calls.find(c => c.opts.label.startsWith('amplify:'))
  assert.match(amp.prompt, /--replay --input \.factory\/fuzz\/fuzz-wire\/crashes\/crash-d/)
  assert.ok(calls.find(c => c.opts.label.startsWith('diagnose:')).prompt.includes('WOULD-FAIL-IF'))
  assertRoles(assert, calls, { 'status': 'wan-verifier', 'campaign': 'wan-verifier', 'diagnose:': 'wan-coder', 'propose:': 'wan-proposer', 'vote:': 'wan-reviewer', 'apply:': 'wan-coder', 'verify': 'wan-verifier', 'report': 'wan-researcher' })
})

test('n/a campaign; a campaign that never ran gets a smoke first, and a failing self-test stops it', async () => {
  const na = await run({ 'status': () => ({ json: {}, exitCode: 5 }) })
  assert.deepEqual([na.result.ok, na.result.applicable], [true, false])
  const fresh = await run({ 'status': () => ({ json: { runs: 0 }, exitCode: 1 }), 'smoke': () => ({ json: { ok: true, newUnique: [] }, exitCode: 0 }) })
  assert.ok(fresh.calls.some(c => c.opts.label === 'smoke' && c.prompt.includes('--minutes 10')))
  const dead = await run({ 'status': () => ({ json: { runs: 0 }, exitCode: 1 }), 'smoke': () => ({ json: { selfTestFailed: true, detail: 'fuzzing nothing' }, exitCode: 1 }) })
  assert.deepEqual([dead.result.ok, dead.result.stoppedAt], [false, 'setup'])
})

test('a fix that still crashes, a re-fuzz that finds the same signature, or a flaky crash the amplifier reproduces keeps it open', async () => {
  const still = await run({ 'verify': () => ({ items: [{ id: 'py-ValueError-decode_len-src-wire-decode-py', reproRuns: 20, reproFailures: 3, loadFailed: 0 }], ladder: ladder(true), refuzz: { newUnique: [] } }) })
  assert.equal(still.result.ok, false)
  assert.match(still.result.open[0].why, /still crashes after the fix/)
  const again = await run({ 'verify': () => ({ items: [{ id: 'py-ValueError-decode_len-src-wire-decode-py', reproRuns: 20, reproFailures: 0, loadFailed: 0 }], ladder: ladder(true), refuzz: { newUnique: [{ signature: SIG }] } }) })
  assert.match(again.result.open.map(o => o.why).join(), /re-fuzz found the same signature/)
  const amp = await run({ 'amplify:': () => ({ json: { clean: false, findings: [{ seed: 3 }] }, exitCode: 1 }) })
  assert.equal(amp.result.ok, false)
  assert.equal(amp.result.flaky[0].reproducedUnderAmplifier, true)
})

test('a rejected fix leaves the crash open; bad args are refused', async () => {
  const rejected = await run({ 'vote:': () => ({ approve: false, reasons: 'masks the bug' }) })
  assert.equal(rejected.result.ok, false)
  assert.equal(rejected.result.open[0].why, 'rejected')
  assert.ok(!rejected.calls.some(c => c.opts.label === 'verify'))
  await assert.rejects(simulate(script, { args: { campaign: 'A B' }, respond: () => null }), /args.campaign must be/)
})
