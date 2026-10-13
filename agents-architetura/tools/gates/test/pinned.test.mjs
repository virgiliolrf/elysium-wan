import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync, statSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync, execFileSync } from 'node:child_process'
import { runPinned, validateGates, metricValue, waitPinned, runStatePath, runKey, GATES_ID_RE } from '../pinned.mjs'

const CLI = join(dirname(fileURLToPath(import.meta.url)), '..', 'pinned.mjs')
// The CLI runs take the machine lock. They get their own lock directory, or they would wait (up to an hour) behind the
// real machine's test slots whenever other builds are testing.
process.env.FACTORY_LOCK_DIR = mkdtempSync(join(tmpdir(), 'pinned-locks-'))
delete process.env.FACTORY_MACHINE_LOCK_HELD
delete process.env.FACTORY_TEST_SLOT_HELD
const noLock = (_, fn) => fn()
function project({ rungs, campaigns = [], files = {} }) {
  const root = mkdtempSync(join(tmpdir(), 'pinned-'))
  const write = (f, body) => { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), body) }
  write('agents-architetura/tools/gates/rungs.json', JSON.stringify({ rungs }))
  write('agents-architetura/tools/campaign/campaigns.json', JSON.stringify({ campaigns }))
  write('.gitignore', '.factory/\n.n\n.c\n')
  for (const [f, b] of Object.entries(files)) write(f, b)
  execFileSync('git', ['init', '-q'], { cwd: root })
  return root
}

test('VERIFIED only when every frozen gate passes; the verdict names the tree', () => {
  const root = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'true' }] }, { id: 'chaos', steps: [{ cmd: 'true' }] }], campaigns: [{ id: 'tsan', kind: 'sanitizer', cmd: 'echo quiet' }] })
  const r = runPinned(root, { milestone: 'M0', consecutive: 2, rungs: ['unit', 'chaos'], pinned: [{ id: 'smoke', cmd: 'true', repeat: 5 }], campaigns: [{ id: 'tsan' }] }, { lock: noLock })
  assert.equal(r.verdict, 'VERIFIED', JSON.stringify(r.failing))
  assert.match(r.tree, /^[0-9a-f]{40}$/)
  assert.deepEqual(r.gates.map(g => g.id), ['rung:unit', 'rung:chaos', 'consecutive', 'pinned:smoke', 'campaign:tsan'])
})

test('NOT VERIFIED: a flaky pinned command, a campaign with reports, a missing audit, the second pass red', () => {
  const counter = 'n=$(cat .n 2>/dev/null || echo 0); echo $((n+1)) > .n; test $n -lt 3'
  const root = project({
    rungs: [{ id: 'chaos', steps: [{ cmd: 'c=$(cat .c 2>/dev/null || echo 0); echo $((c+1)) > .c; test $c -lt 1' }] }],
    campaigns: [{ id: 'tsan', kind: 'sanitizer', cmd: 'printf "Traceback (most recent call last):\\n  File \\"src/a.py\\", line 1, in f\\nValueError: x\\n"; exit 1' }],
  })
  const r = runPinned(root, { milestone: 'M1', consecutive: 2, rungs: ['chaos'], pinned: [{ id: 'smoke', cmd: counter, repeat: 20 }], campaigns: [{ id: 'tsan' }], audit: { bar: 'no-open-tier-a' }, runAll: true }, { lock: noLock })
  assert.equal(r.verdict, 'NOT VERIFIED')
  assert.deepEqual(r.failing, ['rung:chaos', 'consecutive', 'pinned:smoke', 'campaign:tsan', 'audit'])
  assert.match(r.gates.find(g => g.id === 'pinned:smoke').detail, /^3\/20 passed, then fail/)
  assert.match(r.gates.find(g => g.id === 'rung:chaos').detail, /pass 2\/2/)
  assert.match(r.gates.find(g => g.id === 'audit').detail, /AUDIT-RESULTS\.json is missing/)
})

test('the heavy tail (campaigns, mutation) is skipped once the gate is red, listed apart and never VERIFIED', () => {
  const root = project({ rungs: [{ id: 'chaos', steps: [{ cmd: 'exit 1' }] }], campaigns: [{ id: 'tsan', kind: 'sanitizer', cmd: 'touch .c' }] })
  const r = runPinned(root, { milestone: 'M1', rungs: ['chaos'], pinned: [{ id: 'smoke', cmd: 'true' }], campaigns: [{ id: 'tsan' }], mutation: { changedSince: 'HEAD' } }, { lock: noLock })
  assert.equal(r.verdict, 'NOT VERIFIED')
  assert.deepEqual(r.failing, ['rung:chaos'], 'pinned entries still run; only the heavy tail is skipped')
  assert.deepEqual(r.skipped, ['campaign:tsan', 'mutation'])
  assert.equal(r.gates.find(g => g.id === 'pinned:smoke').ok, true)
  assert.match(r.gates.find(g => g.id === 'campaign:tsan').detail, /already red at rung:chaos/)
  assert.equal(r.gates.find(g => g.id === 'campaign:tsan').kind, 'skipped')
  assert.equal(existsSync(join(root, '.c')), false, 'the campaign command never ran')
  assert.deepEqual(validateGates({ milestone: 'M1', runAll: 'yes' }, { rungIds: [], campaigns: [] }), ['runAll must be true or false'])
})

test('a gate that edits tracked files while it runs is NOT VERIFIED (evidence must be of one tree)', () => {
  const root = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'echo drift >> src.py' }] }] })
  const r = runPinned(root, { milestone: 'M0', rungs: ['unit'] }, { lock: noLock })
  assert.deepEqual(r.failing, ['tree-stable'])
})

test('a pinned campaign that cannot run is a failure, not an n/a pass', () => {
  const root = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'true' }] }], campaigns: [{ id: 'gpu', kind: 'sanitizer', probe: 'exit 1', cmd: 'true' }] })
  const r = runPinned(root, { milestone: 'M0', campaigns: [{ id: 'gpu' }] }, { lock: noLock })
  assert.equal(r.verdict, 'NOT VERIFIED')
  assert.match(r.gates[0].detail, /a pinned campaign must run/)
})

test('gates file validation and CLI exit codes', () => {
  const ctx = { rungIds: ['unit'], campaigns: [{ id: 'tsan', kind: 'sanitizer' }, { id: 'fz', kind: 'fuzz' }] }
  assert.deepEqual(validateGates({ milestone: 'M0', rungs: ['unit'], campaigns: [{ id: 'tsan' }], fuzz: [{ id: 'fz' }] }, ctx), [])
  const errs = validateGates({ milestone: 'X', rungs: ['nope'], pinned: [{ id: 'A', cmd: 'a\x01' }], campaigns: [{ id: 'fz' }], fuzz: [{ id: 'tsan' }], mutation: { changedSince: 'a;b' } }, ctx)
  assert.equal(errs.length, 7)
  const root = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'true' }] }] })
  assert.equal(spawnSync(process.execPath, [CLI, 'M0', '--root', root], { encoding: 'utf8' }).status, 5)
  writeFileSync(join(root, 'docs-gates.json'), '')
  mkdirSync(join(root, 'docs/design'), { recursive: true })
  writeFileSync(join(root, 'docs/design/GATES-M0.json'), JSON.stringify({ milestone: 'M0', rungs: ['unit'] }))
  const ok = spawnSync(process.execPath, [CLI, 'M0', '--root', root, '--json'], { encoding: 'utf8' })
  assert.equal(ok.status, 0, ok.stderr)
  assert.equal(JSON.parse(ok.stdout).verdict, 'VERIFIED')
  assert.deepEqual(JSON.parse(spawnSync(process.execPath, [CLI, 'M0', '--root', root, '--validate'], { encoding: 'utf8' }).stdout), { ok: true, errors: [] })
  writeFileSync(join(root, 'docs/design/GATES-M0.json'), JSON.stringify({ milestone: 'M0', rungs: ['ghost'] }))
  assert.equal(spawnSync(process.execPath, [CLI, 'M0', '--root', root], { encoding: 'utf8' }).status, 2)
  const invalid = spawnSync(process.execPath, [CLI, 'M0', '--root', root, '--validate'], { encoding: 'utf8' })
  assert.equal(invalid.status, 2)
  assert.match(JSON.parse(invalid.stdout).errors[0], /unknown rung "ghost"/)
})

test('a finding awaiting a human decision blocks VERIFIED', () => {
  const root = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'true' }] }], files: { 'agents-architetura/docs/findings.json': JSON.stringify({ findings: [{ id: 'RT-race', status: 'needs-human' }, { id: 'F-1', status: 'open' }] }) } })
  const r = runPinned(root, { milestone: 'M0', rungs: ['unit'] }, { lock: noLock })
  assert.deepEqual(r.failing, ['findings'])
  assert.match(r.gates.find(g => g.id === 'findings').detail, /1 finding\(s\) await a human decision: RT-race/)
})

test('a phase gate (wan-x slug) runs like a milestone gate; numeric bounds are decided by the script on every repetition', () => {
  const root = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'true' }] }] })
  const gate = metric => ({ milestone: 'microbatch-rebalance', rungs: ['unit'], pinned: [{ id: 'step', cmd: 'echo warmup step=9; echo step=0.95', repeat: 3, metric }] })
  const ok = runPinned(root, gate({ pattern: 'step=([0-9.]+)', op: '<=', value: 1 }), { lock: noLock })
  assert.equal(ok.verdict, 'VERIFIED', JSON.stringify(ok.gates))
  assert.deepEqual(ok.gates.find(g => g.id === 'pinned:step').values, [0.95, 0.95, 0.95])
  const over = runPinned(root, gate({ pattern: 'step=([0-9.]+)', op: '<', value: 0.9 }), { lock: noLock })
  assert.equal(over.verdict, 'NOT VERIFIED')
  assert.match(over.gates.find(g => g.id === 'pinned:step').detail, /0\/3 passed, then 0.95 broke the bound < 0.9/)
  const missing = runPinned(root, gate({ pattern: 'latency=([0-9.]+)', op: '<=', value: 1 }), { lock: noLock })
  assert.match(missing.gates.find(g => g.id === 'pinned:step').detail, /no number/)
  assert.equal(spawnSync(process.execPath, [CLI, '--root', root, 'microbatch-rebalance'], { encoding: 'utf8' }).status, 5)
  writeFileSync(join(root, 'docs-gates.json'), '')
  mkdirSync(join(root, 'docs/design'), { recursive: true })
  writeFileSync(join(root, 'docs/design/GATES-microbatch-rebalance.json'), JSON.stringify(gate({ pattern: 'step=([0-9.]+)', op: '<=', value: 1 })))
  const cli = spawnSync(process.execPath, [CLI, '--root', root, 'microbatch-rebalance', '--json'], { encoding: 'utf8' })
  assert.equal(cli.status, 0, cli.stderr)
  assert.equal(JSON.parse(cli.stdout).milestone, 'microbatch-rebalance')
  assert.equal(metricValue('a=1 a=2.5e-1', 'a=([0-9.e-]+)'), 0.25)
  assert.equal(metricValue('a=x', 'a=(\\w+)'), null)
})

test('amplified reruns: the load runs during the repetitions and is stopped afterwards, pass or fail', () => {
  const root = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'true' }] }] })
  const counter = join(root, '.n')
  writeFileSync(counter, '')
  const gate = cmd => ({ milestone: 'amp', pinned: [{ id: 'race', cmd, repeat: 3, load: { cmd: `echo x >> ${counter}; sleep 0.02`, copies: 2 } }] })
  const r = runPinned(root, gate('sleep 0.2'), { lock: noLock })
  assert.equal(r.verdict, 'VERIFIED', JSON.stringify(r.gates))
  assert.match(r.gates[0].detail, /3\/3 passed under 2x load/)
  const ran = statSync(counter).size
  assert.ok(ran > 0, 'the load ran while the repetitions did')
  spawnSync('sleep', ['0.3'])
  assert.equal(statSync(counter).size, ran, 'the load stopped with the gate')
  const stops = []
  const red = runPinned(root, gate('false'), { lock: noLock, load: () => () => stops.push(1) })
  assert.equal(red.verdict, 'NOT VERIFIED')
  assert.deepEqual(stops, [1])
})

test('load and metric are validated like the rest of the frozen bar', () => {
  const ctx = { rungIds: ['unit'], campaigns: [] }
  assert.deepEqual(validateGates({ milestone: 'x-phase', pinned: [{ id: 'a', cmd: 'true', load: { cmd: 'true', copies: 4 }, metric: { pattern: 'v=(\\d+)', op: '>=', value: 3 } }] }, ctx), [])
  const errs = validateGates({ milestone: 'Bad Slug', pinned: [{ id: 'a', cmd: 'true', load: { cmd: '', copies: 99 }, metric: { pattern: 'no-group', op: '~', value: 'x' } }] }, ctx)
  assert.equal(errs.length, 5, errs.join('; '))
  assert.match(validateGates({ milestone: 'M1', pinned: [{ id: 'a', cmd: 'true', metric: { pattern: '(', op: '<', value: 1 } }] }, ctx).join(), /capture group/)
})

// The gate runs for hours and an agent's shell call returns within 10 minutes: M2 phase 2's verifier started pinned.mjs
// in the background, ended, and the verdict was lost. --wait runs it once, detached, and any later call finds it.
test('--wait: one detached run per (label, tree, gates); a later call prints the stored verdict without rerunning it', () => {
  const root = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'true' }] }] })
  mkdirSync(join(root, 'docs/design'), { recursive: true })
  writeFileSync(join(root, 'docs/design/GATES-m2-p9.json'), JSON.stringify({ milestone: 'm2-p9', rungs: ['unit'], pinned: [{ id: 'count', cmd: 'echo x >> .c' }] }))
  const runs = () => readFileSync(join(root, '.c'), 'utf8').split('\n').filter(Boolean).length
  const call = (...extra) => spawnSync(process.execPath, [CLI, 'm2-p9', '--root', root, '--json', '--label', 'r0', '--wait', '60', ...extra], { encoding: 'utf8' })
  const first = call()
  assert.equal(first.status, 0, first.stderr + first.stdout)
  assert.equal(JSON.parse(first.stdout).verdict, 'VERIFIED')
  const again = call()
  assert.equal(again.status, 0, again.stderr)
  assert.deepEqual(JSON.parse(again.stdout), JSON.parse(first.stdout), 'the stored verdict, printed again')
  assert.equal(runs(), 1, 'the same key never reruns')
  assert.equal(call('--fresh').status, 0)
  assert.equal(runs(), 2, '--fresh discards the stored verdict')
  writeFileSync(join(root, 'tracked.txt'), 'new tree')
  assert.equal(call().status, 0)
  assert.equal(runs(), 3, 'a new tree starts a new run')
  const other = spawnSync(process.execPath, [CLI, 'm2-p9', '--root', root, '--json', '--label', 'r1', '--wait', '60'], { encoding: 'utf8' })
  assert.equal(other.status, 0, other.stderr)
  assert.equal(runs(), 4, 'another label (a later verify round) is another run')
  for (const bad of [['--label', 'R!', '--wait', '5'], ['--label', 'r0', '--wait', '900'], ['--label', 'r0']]) {
    assert.equal(spawnSync(process.execPath, [CLI, 'm2-p9', '--root', root, '--json', ...bad], { encoding: 'utf8' }).status, bad.includes('--wait') ? 2 : 0)
  }
})

test('--wait returns "running" at its deadline, finds a live run, restarts a dead one, and keeps a failure until --fresh', async () => {
  const root = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'true' }] }] })
  mkdirSync(join(root, 'docs/design'), { recursive: true })
  writeFileSync(join(root, 'docs/design/GATES-m2-p9.json'), JSON.stringify({ milestone: 'm2-p9', rungs: ['unit'] }))
  const state = runStatePath(root, 'm2-p9', 'r0')
  let t = 0
  const clock = { now: () => t, sleep: async ms => { t += ms } }
  const live = new Set()
  let starts = 0
  const start = async (r, m, statePath, key) => {
    starts++
    const s = { status: 'running', ...key, pid: 1000 + starts, startedAt: 'x' }
    live.add(s.pid)
    writeFileSync(statePath, JSON.stringify(s))
    return s
  }
  const opts = extra => ({ waitSec: 30, start, alive: pid => live.has(pid), pollMs: 5000, ...clock, ...extra })
  const r1 = await waitPinned(root, 'm2-p9', 'r0', opts())
  assert.deepEqual([r1.status, starts], ['running', 1])
  // the run finishes while a later caller waits
  const finishAt = t + 10000
  const r2 = await waitPinned(root, 'm2-p9', 'r0', opts({ sleep: async ms => { t += ms; if (t >= finishAt) writeFileSync(state, JSON.stringify({ ...JSON.parse(readFileSync(state, 'utf8')), status: 'done', result: { verdict: 'VERIFIED' } })) } }))
  assert.deepEqual([r2.status, r2.result.verdict, starts], ['done', 'VERIFIED', 1])
  // a run whose process died without a record is started again
  writeFileSync(state, JSON.stringify({ status: 'running', ...runKey(root, 'm2-p9'), pid: 999 }))
  const r3 = await waitPinned(root, 'm2-p9', 'r0', opts({ waitSec: 1 }))
  assert.deepEqual([r3.status, starts], ['running', 2])
  live.clear()
  const died = await waitPinned(root, 'm2-p9', 'r0', opts({ waitSec: 10, start: async () => assert.fail('a live-looking state is waited on first') , alive: (() => { let n = 0; return () => n++ === 0 })() }))
  assert.equal(died.died, true, 'a run that dies while waited on is reported, and the next call restarts it')
  writeFileSync(state, JSON.stringify({ status: 'failed', ...runKey(root, 'm2-p9'), pid: 1, error: 'GATES invalid' }))
  const failed = await waitPinned(root, 'm2-p9', 'r0', opts())
  assert.deepEqual([failed.status, starts], ['failed', 2])
  const fresh = await waitPinned(root, 'm2-p9', 'r0', opts({ fresh: true, waitSec: 1 }))
  assert.deepEqual([fresh.status, starts], ['running', 3])
  assert.ok(!statSync(join(root, '.factory/gates/runs')).isFile())
})

test('gate ids: milestones (M0, M2b), phase slugs; anything else is refused', () => {
  for (const ok of ['M0', 'M2', 'M2b', 'M12', 'm2-p2-wave-label-binding']) assert.ok(GATES_ID_RE.test(ok), ok)
  for (const bad of ['M2bb', 'M2B', 'M', 'M1234', '../x', 'M2;rm']) assert.ok(!GATES_ID_RE.test(bad), bad)
})

test('preflight: a missing audit, an unrunnable campaign or a red mutation baseline stops the gate before its rungs', () => {
  const root = project({
    rungs: [{ id: 'unit', steps: [{ cmd: 'mkdir -p .factory && touch .factory/rung-ran' }] }],
    campaigns: [{ id: 'gpu', kind: 'sanitizer', probe: 'exit 1', cmd: 'true' }],
    files: { 'agents-architetura/tools/gates/mutation.json': JSON.stringify({ python: { cmd: 'echo "No module named safetensors"; exit 1' } }) },
  })
  // a committed base and one changed Python source file, so mutation testing applies and its baseline runs
  execFileSync('git', ['-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '-A'], { cwd: root })
  execFileSync('git', ['-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'base'], { cwd: root })
  mkdirSync(join(root, 'src'), { recursive: true })
  writeFileSync(join(root, 'src/m.py'), 'x = 1\n')
  const r = runPinned(root, { milestone: 'M2', rungs: ['unit'], pinned: [{ id: 'smoke', cmd: 'mkdir -p .factory && touch .factory/pinned-ran' }], campaigns: [{ id: 'gpu' }], audit: { bar: 'no-open-tier-a' }, mutation: { changedSince: 'HEAD' } }, { lock: noLock })
  assert.equal(r.verdict, 'NOT VERIFIED')
  assert.deepEqual(r.failing.sort(), ['audit', 'campaign:gpu', 'mutation'])
  assert.deepEqual(r.skipped, ['rung:unit', 'pinned:smoke'])
  assert.equal(existsSync(join(root, '.factory/rung-ran')), false, 'no rung ran')
  assert.equal(existsSync(join(root, '.factory/pinned-ran')), false, 'no pinned entry ran')
  assert.match(r.gates.find(g => g.id === 'mutation').detail, /^preflight: the test command fails before any mutation: .*safetensors/s)
  assert.match(r.gates.find(g => g.id === 'audit').detail, /^preflight: .*AUDIT-RESULTS\.json is missing/)
  // a green preflight runs the gate in full
  const ok = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'mkdir -p .factory && touch .factory/rung-ran' }] }], files: { 'agents-architetura/tools/gates/mutation.json': JSON.stringify({ python: { cmd: 'true' } }) } })
  const v = runPinned(ok, { milestone: 'M2', rungs: ['unit'] }, { lock: noLock })
  assert.equal(v.verdict, 'VERIFIED', JSON.stringify(v.failing))
  assert.equal(existsSync(join(ok, '.factory/rung-ran')), true)
  // no changed Python source line: mutation does not apply, so its (red) baseline never runs and nothing fails on it
  const quiet = project({ rungs: [{ id: 'unit', steps: [{ cmd: 'true' }] }], files: { 'agents-architetura/tools/gates/mutation.json': JSON.stringify({ python: { cmd: 'exit 1' } }) } })
  execFileSync('git', ['-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '-A'], { cwd: quiet })
  execFileSync('git', ['-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'base'], { cwd: quiet })
  const q = runPinned(quiet, { milestone: 'M2', rungs: ['unit'], mutation: { changedSince: 'HEAD' } }, { lock: noLock })
  assert.equal(q.verdict, 'VERIFIED', JSON.stringify(q.gates.filter(g => !g.ok)))
})
