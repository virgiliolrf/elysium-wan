import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync, execFileSync } from 'node:child_process'
import { runLadder, validateConfig, failureLines, ladderFingerprint, reuseRefusal, mergeReused } from '../ladder.mjs'

const LADDER = join(dirname(fileURLToPath(import.meta.url)), '..', 'ladder.mjs')

function repo(files = []) {
  const root = mkdtempSync(join(tmpdir(), 'ladder-'))
  for (const f of files) {
    if (f.endsWith('/')) mkdirSync(join(root, f), { recursive: true })
    else writeFileSync(join(root, f), '')
  }
  return root
}
const cfg = rungs => ({ rungs })

test('steps run only when their when-paths exist; rungs with no applicable step are n/a with a reason', () => {
  const root = repo(['pyproject.toml'])
  const ran = []
  const r = runLadder({
    root,
    config: cfg([
      { id: 'lint', steps: [{ when: ['pyproject.toml'], cmd: 'py-lint' }, { when: ['Cargo.toml'], cmd: 'rs-lint' }] },
      { id: 'oracle', steps: [{ when: ['tests/oracle'], cmd: 'oracle' }] },
    ]),
    run: cmd => { ran.push(cmd); return { code: 0, output: '' } },
  })
  assert.deepEqual(ran, ['py-lint'])
  assert.deepEqual(r.rungs.map(x => [x.rung, x.status]), [['lint', 'pass'], ['oracle', 'n/a']])
  assert.equal(r.rungs[1].detail, 'not applicable: tests/oracle absent')
  assert.equal(r.allGreen, true)
})

test('a failing step fails the rung with the output tail; a blocking failure skips the rest', () => {
  const r = runLadder({
    root: repo(),
    config: cfg([
      { id: 'build', blocking: true, steps: [{ cmd: 'make' }] },
      { id: 'unit', steps: [{ cmd: 'test' }] },
    ]),
    run: () => ({ code: 2, output: 'x'.repeat(5000) + 'BOOM' }),
  })
  assert.equal(r.allGreen, false)
  assert.equal(r.rungs[0].status, 'fail')
  assert.match(r.rungs[0].detail, /\$ make\n\(exit 2\)\n/)
  assert.ok(r.rungs[0].detail.endsWith('BOOM'))
  assert.ok(r.rungs[0].detail.length < 4100)
  assert.deepEqual(r.rungs[1], { rung: 'unit', status: 'skipped', detail: 'blocked by failing rung build', durationMs: 0 })
})

test('--fail-fast: a fix round stops at the first failing rung, blocking or not; without it later rungs still run', () => {
  const ran = []
  const config = cfg([
    { id: 'lint', steps: [{ cmd: 'lint' }] },
    { id: 'unit', steps: [{ cmd: 'unit' }] },
    { id: 'chaos', steps: [{ cmd: 'chaos' }] },
  ])
  const run = cmd => { ran.push(cmd); return { code: cmd === 'unit' ? 1 : 0, output: '' } }
  const fast = runLadder({ root: repo(), config, failFast: true, run })
  assert.deepEqual(fast.rungs.map(r => r.status), ['pass', 'fail', 'skipped'])
  assert.equal(fast.rungs[2].detail, 'blocked by failing rung unit')
  assert.deepEqual(ran, ['lint', 'unit'])
  ran.length = 0
  assert.deepEqual(runLadder({ root: repo(), config, run }).rungs.map(r => r.status), ['pass', 'fail', 'pass'])
  assert.deepEqual(ran, ['lint', 'unit', 'chaos'])
})

test('a failure names what failed even when the tail is only a summary', () => {
  const output = ['# Subtest: a', 'not ok 7 - publish guards refuse secrets', '  error: \'boom\'', ...Array.from({ length: 400 }, (_, i) => `ok ${i} - fine`), '# fail 1']
  const r = runLadder({ root: repo(), config: cfg([{ id: 'unit', steps: [{ cmd: 'test' }] }]), run: () => ({ code: 1, output: output.join('\n') }) })
  assert.match(r.rungs[0].detail, /failures:\nnot ok 7 - publish guards refuse secrets\n  error: 'boom'\n--- tail ---/)
  assert.ok(r.rungs[0].detail.endsWith('# fail 1'))
  assert.deepEqual(failureLines("not ok 61 - mutants\n  error: |-\n    Expected values to be strictly deep-equal:\n    + ['GtE->Lt']\n  code: 'ERR_ASSERTION'\nok 62"), ['not ok 61 - mutants', '  error: |-', '    Expected values to be strictly deep-equal:', "    + ['GtE->Lt']"])
  assert.deepEqual(failureLines('ok 1\n✖ broke (3ms)\nAssertionError [ERR_ASSERTION]: x\nthread main panicked at src/a.rs:1\nE   assert 1 == 2\nok'), ['✖ broke (3ms)', 'AssertionError [ERR_ASSERTION]: x', 'thread main panicked at src/a.rs:1', 'E   assert 1 == 2'])
})

test('--only, --upto and --require', () => {
  const config = cfg([{ id: 'a', steps: [{ cmd: 'a' }] }, { id: 'b', steps: [{ cmd: 'b' }] }, { id: 'c', steps: [{ when: ['nope'], cmd: 'c' }] }])
  const run = () => ({ code: 0, output: '' })
  assert.deepEqual(runLadder({ root: repo(), config, only: ['b'], run }).rungs.map(r => r.rung), ['b'])
  assert.deepEqual(runLadder({ root: repo(), config, upto: 'b', run }).rungs.map(r => r.rung), ['a', 'b'])
  const req = runLadder({ root: repo(), config, require: ['c'], run })
  assert.deepEqual(req.missingRequired, ['c'])
  assert.equal(req.allGreen, false)
  assert.throws(() => runLadder({ root: repo(), config, only: ['zz'], run }), /unknown rung: zz/)
})

test('config validation', () => {
  assert.throws(() => validateConfig(cfg([])), /no rungs/)
  assert.throws(() => validateConfig(cfg([{ id: 'A', steps: [{ cmd: 'x' }] }])), /bad rung id/)
  assert.throws(() => validateConfig(cfg([{ id: 'a', steps: [] }])), /at least one step/)
  assert.throws(() => validateConfig(cfg([{ id: 'a', steps: [{ cmd: 'x' }] }, { id: 'a', steps: [{ cmd: 'y' }] }])), /duplicate/)
})

test('the shipped rungs.json is valid', () => {
  validateConfig(JSON.parse(readFileSync(join(dirname(LADDER), 'rungs.json'), 'utf8')))
})

test('CLI: real shell execution, JSON output, exit code, last.json', () => {
  const root = repo(['marker'])
  mkdirSync(join(root, 'agents-architetura', 'tools', 'gates'), { recursive: true })
  writeFileSync(join(root, 'agents-architetura', 'tools', 'gates', 'rungs.json'), JSON.stringify(cfg([
    { id: 'ok', steps: [{ when: ['marker'], cmd: 'echo fine' }] },
    { id: 'bad', steps: [{ cmd: 'echo nope >&2; exit 3' }] },
  ])))
  const p = spawnSync(process.execPath, [LADDER, '--json', '--root', root], { encoding: 'utf8' })
  assert.equal(p.status, 1)
  const report = JSON.parse(p.stdout)
  assert.deepEqual(report.rungs.map(r => [r.rung, r.status]), [['ok', 'pass'], ['bad', 'fail']])
  assert.match(report.rungs[1].detail, /nope/)
  assert.deepEqual(JSON.parse(readFileSync(join(root, '.factory', 'ladder', 'last.json'), 'utf8')), report)
  const green = spawnSync(process.execPath, [LADDER, '--json', '--root', root, '--only', 'ok'], { encoding: 'utf8' })
  assert.equal(green.status, 0)
})

test('a step exit code declared not-applicable (pytest 5: no tests collected) makes the rung n/a, not fail', () => {
  const config = cfg([
    { id: 'oracle', steps: [{ cmd: 'pytest tests/oracle', naExitCodes: [5], naReason: 'no tests collected' }] },
    { id: 'unit', steps: [{ cmd: 'pytest a', naExitCodes: [5] }, { cmd: 'cargo test' }] },
    { id: 'chaos', steps: [{ cmd: 'pytest tests/chaos', naExitCodes: [5] }] },
  ])
  const codes = { 'pytest tests/oracle': 5, 'pytest a': 5, 'cargo test': 0, 'pytest tests/chaos': 1 }
  const r = runLadder({ root: repo(), config, run: cmd => ({ code: codes[cmd], output: 'x' }) })
  assert.deepEqual(r.rungs.map(x => [x.rung, x.status]), [['oracle', 'n/a'], ['unit', 'pass'], ['chaos', 'fail']])
  assert.match(r.rungs[0].detail, /not applicable: \$ pytest tests\/oracle: no tests collected/)
})

test('repeat: a step passes only if every run passes — a flaky pass is not a pass', () => {
  let n = 0
  const config = cfg([{ id: 'smoke', steps: [{ cmd: 'smoke', repeat: 20 }] }])
  const flaky = runLadder({ root: repo(), config, run: () => ({ code: ++n === 7 ? 1 : 0, output: 'boom' }) })
  assert.equal(flaky.rungs[0].status, 'fail')
  assert.match(flaky.rungs[0].detail, /6\/20 runs passed before run 7 failed — a flaky pass is not a pass/)
  assert.equal(n, 7)
  let m = 0
  const steady = runLadder({ root: repo(), config: cfg([{ id: 'smoke', steps: [{ cmd: 'smoke', repeat: 3 }] }]), run: () => { m++; return { code: 0, output: '' } } })
  assert.equal(steady.rungs[0].status, 'pass')
  assert.equal(steady.rungs[0].detail, '$ smoke: 3/3 runs passed')
  assert.equal(m, 3)
})

test('repeat with a not-applicable exit code: n/a on the first run, not repeated', () => {
  let n = 0
  const r = runLadder({ root: repo(), config: cfg([{ id: 'oracle', steps: [{ cmd: 'pytest', repeat: 5, naExitCodes: [5] }] }]), run: () => { n++; return { code: 5, output: '' } } })
  assert.equal(r.rungs[0].status, 'n/a')
  assert.equal(n, 1)
  assert.throws(() => validateConfig(cfg([{ id: 'a', steps: [{ cmd: 'x', repeat: 0 }] }])), /repeat must be an integer/)
})

test('heavy rungs run under the machine lock; light rungs do not', () => {
  const locked = []
  runLadder({
    root: repo(),
    config: cfg([{ id: 'unit', steps: [{ cmd: 'u' }] }, { id: 'bench', heavy: true, steps: [{ cmd: 'b' }] }]),
    run: () => ({ code: 0, output: '' }),
    lock: (label, fn) => { locked.push(label); return fn() },
  })
  assert.deepEqual(locked, ['ladder:bench'])
})

test('--consecutive: green only if every pass is green; stops at the first red pass', async () => {
  const { runLadderConsecutive } = await import('../ladder.mjs')
  const config = cfg([{ id: 'unit', steps: [{ cmd: 'u' }] }, { id: 'chaos', steps: [{ cmd: 'c' }] }])
  let calls = 0
  const secondRed = runLadderConsecutive({ root: repo(), config, consecutive: 3, run: cmd => { calls++; return { code: cmd === 'c' && calls > 2 ? 1 : 0, output: 'race' } } })
  assert.equal(secondRed.allGreen, false)
  assert.deepEqual(secondRed.consecutive, { required: 3, ran: 2, green: 1 })
  assert.equal(secondRed.rungs.find(r => r.rung === 'chaos').status, 'fail')
  assert.match(secondRed.rungs.find(r => r.rung === 'chaos').detail, /^pass 2\/3:/)
  const green = runLadderConsecutive({ root: repo(), config, consecutive: 2, run: () => ({ code: 0, output: '' }) })
  assert.equal(green.allGreen, true)
  assert.deepEqual(green.consecutive, { required: 2, ran: 2, green: 2 })
  assert.throws(() => runLadderConsecutive({ root: repo(), config, consecutive: 0 }), /consecutive must be/)
})

test('--consecutive reruns only what a rerun can refute: static checks once, full once in the last pass in place of its folders', async () => {
  const { runLadderConsecutive, passPlan } = await import('../ladder.mjs')
  const config = cfg([
    { id: 'lint', deterministic: true, steps: [{ cmd: 'lint' }] },
    { id: 'unit', steps: [{ cmd: 'unit' }] },
    { id: 'integration', coveredBy: 'full', steps: [{ cmd: 'integration' }] },
    { id: 'wan', heavy: true, coveredBy: 'full', steps: [{ cmd: 'wan' }] },
    { id: 'full', onDemand: true, steps: [{ cmd: 'full' }] },
  ])
  const ran = []
  const locked = []
  const r = runLadderConsecutive({ root: repo(), config, consecutive: 2, require: ['lint', 'unit', 'integration', 'wan', 'full'], run: cmd => { ran.push(cmd); return { code: 0, output: '' } }, lock: (label, fn) => { locked.push(label); return fn() } })
  assert.equal(r.allGreen, true, JSON.stringify(r))
  assert.deepEqual(ran, ['lint', 'unit', 'integration', 'wan', 'unit', 'full'], 'each test runs once per pass; the static check once')
  assert.deepEqual(locked, ['ladder:wan', 'ladder:full'], 'full standing in for a heavy rung holds the machine')
  assert.deepEqual(r.rungs.map(x => [x.rung, x.status]), [['lint', 'pass'], ['unit', 'pass'], ['integration', 'pass'], ['wan', 'pass'], ['full', 'pass']])
  assert.match(r.rungs.find(x => x.rung === 'lint').detail, /pass 2: deterministic/)
  assert.match(r.rungs.find(x => x.rung === 'integration').detail, /pass 2: covered by full/)
  assert.deepEqual(r.missingRequired, [])
  // without full in the selection every non-static rung reruns; with one pass nothing changes
  assert.deepEqual(passPlan(config, ['lint', 'unit', 'integration'], 2).map(p => p.run), [['lint', 'unit', 'integration'], ['unit', 'integration']])
  assert.deepEqual(passPlan(config, ['lint', 'integration', 'full'], 1).map(p => p.run), [['lint', 'integration', 'full']])
  // a red first pass stops the run: full never got its turn and is reported skipped, not passed
  const red = runLadderConsecutive({ root: repo(), config, consecutive: 2, require: ['unit', 'integration', 'full'], run: cmd => ({ code: cmd === 'integration' ? 1 : 0, output: 'boom' }) })
  assert.equal(red.allGreen, false)
  assert.equal(red.rungs.find(x => x.rung === 'full').status, 'skipped')
  assert.deepEqual(red.missingRequired.sort(), ['full', 'integration'])
  assert.throws(() => validateConfig(cfg([{ id: 'unit', coveredBy: 'full', steps: [{ cmd: 'u' }] }])), /coveredBy can only name the full rung/)
  assert.throws(() => validateConfig(cfg([{ id: 'unit', deterministic: 'yes', steps: [{ cmd: 'u' }] }])), /deterministic must be/)
})

test('CLI report names the tree it ran on (null outside git) and the toolchain', () => {
  const root = repo(['marker'])
  mkdirSync(join(root, 'agents-architetura', 'tools', 'gates'), { recursive: true })
  writeFileSync(join(root, 'agents-architetura', 'tools', 'gates', 'rungs.json'), JSON.stringify(cfg([{ id: 'ok', steps: [{ cmd: 'true' }] }])))
  const outside = JSON.parse(spawnSync(process.execPath, [LADDER, '--json', '--root', root], { encoding: 'utf8' }).stdout)
  assert.equal(outside.tree, null)
  assert.match(outside.toolchain.node, /^v\d+/)
  spawnSync('git', ['init', '-q'], { cwd: root })
  const inside = JSON.parse(spawnSync(process.execPath, [LADDER, '--json', '--root', root, '--consecutive', '2'], { encoding: 'utf8' }).stdout)
  assert.match(inside.tree, /^[0-9a-f]{40}$/)
  assert.deepEqual(inside.consecutive, { required: 2, ran: 2, green: 2 })
})

test('on-demand rungs run only when named by --only or --require', () => {
  const config = cfg([{ id: 'unit', steps: [{ cmd: 'u' }] }, { id: 'mutation', onDemand: true, steps: [{ cmd: 'm' }] }])
  const ran = []
  const run = cmd => { ran.push(cmd); return { code: 0, output: '' } }
  const full = runLadder({ root: repo(), config, run })
  assert.deepEqual(ran, ['u'])
  assert.match(full.rungs[1].detail, /^on demand/)
  assert.equal(runLadder({ root: repo(), config, run, require: ['mutation'] }).rungs[1].status, 'pass')
  assert.equal(runLadder({ root: repo(), config, run, only: ['mutation'] }).rungs[0].status, 'pass')
})

// A git repo with a product file, a factory tool a product rung runs (and what it imports), and inert factory files.
function factoryRepo() {
  const root = mkdtempSync(join(tmpdir(), 'ladder-reuse-'))
  const put = (f, b) => { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), b) }
  put('.gitignore', '.factory/\n')
  put('src/a.py', 'x = 1\n')
  put('agents-architetura/tools/gates/check.mjs', "import './helper.mjs'\n")
  put('agents-architetura/tools/gates/helper.mjs', 'export {}\n')
  put('agents-architetura/tools/wf/src/wan-x.js', 'v1\n')
  put('CLAUDE.md', 'rules v1\n')
  const config = cfg([
    { id: 'factory', steps: [{ cmd: 'echo f >> .factory/factory-runs' }] },
    { id: 'unit', steps: [{ cmd: 'echo u >> .factory/unit-runs; node agents-architetura/tools/gates/check.mjs' }] },
  ])
  put('agents-architetura/tools/gates/rungs.json', JSON.stringify(config))
  mkdirSync(join(root, '.factory'), { recursive: true })
  execFileSync('git', ['init', '-q'], { cwd: root })
  return { root, put, config }
}

test('reuse fingerprint: factory-only edits keep it; product code, a tool a rung runs (or its imports) or a new file change it', () => {
  const { root, put, config } = factoryRepo()
  const fp = () => ladderFingerprint(root, config)
  const base = fp()
  assert.match(base, /^[0-9a-f]{64}$/)
  put('CLAUDE.md', 'rules v2\n')
  put('agents-architetura/tools/wf/src/wan-x.js', 'v2\n')
  put('agents-architetura/docs/NOTES.md', 'new\n')
  assert.equal(fp(), base, 'inert factory files do not change what a product rung sees')
  put('agents-architetura/tools/gates/helper.mjs', 'export const changed = 1\n')
  const tool = fp()
  assert.notEqual(tool, base, 'a file imported by a tool a rung runs is an input')
  put('src/a.py', 'x = 2\n')
  assert.notEqual(fp(), tool)
  const code = fp()
  put('src/new.py', '\n')
  assert.notEqual(fp(), code, 'an untracked product file is an input')
  put('agents-architetura/tools/gates/ladder.mjs', '// the runner\n')
  assert.notEqual(fp(), code)
})

test('reuse is refused on any difference that could change a result, and merges a fresh factory rung', () => {
  const now = Date.parse('2026-10-08T12:00:00Z')
  const last = { fingerprint: 'f', selection: { only: null }, toolchain: { node: 'v24' }, locks: { 'uv.lock': 'h' }, at: '2026-10-08T10:00:00Z', rungs: [{ rung: 'factory', status: 'pass' }, { rung: 'unit', status: 'pass' }] }
  const same = { fingerprint: 'f', selection: { only: null }, toolchain: { node: 'v24' }, locks: { 'uv.lock': 'h' }, now }
  assert.equal(reuseRefusal(last, same), null)
  assert.match(reuseRefusal(null, same), /no previous report/)
  assert.match(reuseRefusal(last, { ...same, fingerprint: 'g' }), /inputs changed/)
  assert.match(reuseRefusal(last, { ...same, fingerprint: null }), /inputs changed/)
  assert.match(reuseRefusal(last, { ...same, selection: { only: ['unit'] } }), /selection/)
  assert.match(reuseRefusal(last, { ...same, toolchain: { node: 'v25' } }), /toolchain/)
  assert.match(reuseRefusal(last, { ...same, locks: {} }), /lockfile/)
  assert.match(reuseRefusal(last, { ...same, now: now + 23 * 3600e3 }), /older than 24 h/)
  assert.match(reuseRefusal({ ...last, at: undefined }, same), /no timestamp/)
  const merged = mergeReused(last, { rungs: [{ rung: 'factory', status: 'fail', detail: 'boom' }] })
  assert.deepEqual([merged.allGreen, merged.rungs[0].detail], [false, 'boom'])
})

test('CLI --reuse: same inputs reuse the last report and rerun only the factory rung; changed inputs run everything', () => {
  const { root, put } = factoryRepo()
  const run = (...a) => spawnSync(process.execPath, [LADDER, '--json', '--root', root, ...a], { encoding: 'utf8' })
  const count = f => readFileSync(join(root, '.factory', f), 'utf8').split('\n').filter(Boolean).length
  const first = run('--reuse')
  assert.equal(first.status, 0, first.stderr)
  assert.match(first.stderr, /not reusing the last report: no previous report/)
  assert.equal(JSON.parse(first.stdout).reused, undefined)
  put('CLAUDE.md', 'rules v2\n')
  const second = JSON.parse(run('--reuse').stdout)
  assert.equal(second.reused.rerun[0], 'factory')
  assert.equal(second.evidenceAt, JSON.parse(first.stdout).at, 'the evidence keeps its original time')
  assert.deepEqual([count('factory-runs'), count('unit-runs')], [2, 1])
  assert.equal(JSON.parse(run().stdout).reused, undefined, 'without --reuse the ladder always runs')
  assert.deepEqual([count('factory-runs'), count('unit-runs')], [3, 2])
  put('src/a.py', 'x = 3\n')
  const changed = run('--reuse')
  assert.equal(JSON.parse(changed.stdout).reused, undefined)
  assert.match(changed.stderr, /inputs changed/)
  assert.equal(count('unit-runs'), 3)
})

test('rest runs exactly the tests no other rung runs; full runs the whole suite only when a gate requires it', () => {
  const shipped = JSON.parse(readFileSync(join(dirname(LADDER), 'rungs.json'), 'utf8'))
  const folders = new Set(shipped.rungs.filter(r => !['rest', 'full'].includes(r.id)).flatMap(r => r.steps.flatMap(s => s.cmd.match(/\btests\/[\w-]+/g) ?? [])))
  const rest = shipped.rungs.find(r => r.id === 'rest').steps[0].cmd
  const ignored = new Set([...rest.matchAll(/--ignore=(tests\/[\w-]+)/g)].map(m => m[1]))
  assert.deepEqual([...ignored].sort(), [...folders].sort(), 'a folder with its own rung must be ignored by rest, and only those')
  const full = shipped.rungs.find(r => r.id === 'full')
  assert.equal(full.onDemand, true)
  const ran = []
  const run = cmd => { ran.push(cmd); return { code: 0, output: '' } }
  const root = repo(['pyproject.toml', 'tests/'])
  const everyday = runLadder({ root, config: shipped, only: ['rest', 'full'], run })
  assert.deepEqual(everyday.rungs.map(r => [r.rung, r.status]), [['rest', 'pass'], ['full', 'pass']], 'named by --only, it runs')
  ran.length = 0
  const daily = runLadder({ root, config: { rungs: shipped.rungs.filter(r => ['rest', 'full'].includes(r.id)) }, run })
  assert.deepEqual(daily.rungs.map(r => [r.rung, r.status]), [['rest', 'pass'], ['full', 'n/a']])
  assert.ok(ran.every(c => !/^uv run pytest -q tests$/.test(c)), 'an everyday ladder never runs the whole suite')
  const gate = runLadder({ root, config: { rungs: shipped.rungs.filter(r => ['rest', 'full'].includes(r.id)) }, require: ['full'], run })
  assert.deepEqual(gate.rungs.map(r => [r.rung, r.status]), [['rest', 'pass'], ['full', 'pass']])
})
