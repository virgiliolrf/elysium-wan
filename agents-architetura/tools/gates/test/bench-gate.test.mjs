import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFileSync } from 'node:child_process'
import { median, measure, measureDetailed, compare, bisect, NoisyMachine } from '../bench-gate.mjs'
import { runScaling } from '../scaling-gate.mjs'

const quiet = () => 0
test('median', () => {
  assert.equal(median([3, 1, 2]), 2)
  assert.equal(median([4, 1, 3, 2]), 2.5)
})

test('warmup rounds are discarded and benches are interleaved', () => {
  const order = []
  const values = { a: [100, 10, 11, 12], b: [200, 20, 21, 22] }
  const suite = { runs: 3, warmup: 1, benches: [{ id: 'a', cmd: 'a' }, { id: 'b', cmd: 'b' }] }
  const out = measureDetailed(suite, { load: quiet, run: cmd => { order.push(cmd); return { code: 0, output: `{"value": ${values[cmd].shift()}}` } } })
  assert.deepEqual(order, ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'])
  assert.deepEqual(out.a, { median: 11, min: 10, max: 12, samples: [10, 11, 12] })
  assert.equal(out.b.median, 21)
})

test('a noisy machine refuses to measure instead of producing a number', () => {
  const suite = { runs: 1, benches: [{ id: 'a', cmd: 'a' }] }
  assert.throws(() => measure(suite, { load: () => 3.5, run: () => ({ code: 0, output: '{"value": 1}' }) }), NoisyMachine)
  assert.deepEqual(measure(suite, { load: () => 3.5, allowNoisy: true, run: () => ({ code: 0, output: '{"value": 1}' }) }), { a: 1 })
})

test('measure rejects failing or malformed benches', () => {
  const suite = { runs: 1, warmup: 0, benches: [{ id: 'b', cmd: 'x' }] }
  assert.throws(() => measure(suite, { load: quiet, run: () => ({ code: 1, output: 'err' }) }), /bench b: exit 1/)
  assert.throws(() => measure(suite, { load: quiet, run: () => ({ code: 0, output: '{"value": "fast"}' }) }), /not \{"value": number\}/)
})

test('compare respects direction and budget; missing baseline fails', () => {
  const suite = { budgetPct: 1, benches: [{ id: 'lat' }, { id: 'tps', higherIsBetter: true }, { id: 'new' }] }
  const r = compare(suite, { lat: 100, tps: 1000 }, { lat: 100.9, tps: 985, new: 5 })
  assert.deepEqual(r.rows.map(x => [x.id, x.ok]), [['lat', true], ['tps', false], ['new', false]])
  assert.equal(r.ok, false)
  assert.match(r.rows[2].why, /no baseline/)
})

test('bisect finds the first commit over budget, in temporary worktrees', () => {
  const root = mkdtempSync(join(tmpdir(), 'bisect-'))
  const git = (...a) => execFileSync('git', a, { cwd: root, encoding: 'utf8' }).trim()
  git('init', '-q')
  const commit = (v, msg) => { writeFileSync(join(root, 'cost.txt'), String(v)); git('add', '-A'); git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '--allow-empty', '-qm', msg) }
  commit(100, 'c0')
  const good = git('rev-parse', 'HEAD')
  commit(100, 'c1'); commit(100.5, 'c2'); commit(130, 'c3'); commit(131, 'c4')
  const c3 = git('rev-parse', 'HEAD~1')
  const suite = { runs: 1, warmup: 0, budgetPct: 1, benches: [{ id: 'cost', cmd: 'echo "{\\"value\\": $(cat cost.txt)}"' }] }
  const r = bisect({ root, suite, good, bad: 'HEAD', benchId: 'cost', load: quiet })
  assert.equal(r.found, true)
  assert.equal(r.commit, c3)
  assert.equal(git('worktree', 'list').split('\n').length, 1)
})

test('scaling: weak/strong speedup, floors only enforced with --gate, serial tax', () => {
  const t = { 1: 10, 2: 10.5, 4: 12, 8: 20 }
  const run = cmd => {
    const n = Number(cmd.match(/w(\d+)/)?.[1])
    const v = cmd === 'on' ? 10.3 : cmd === 'off' ? 10 : t[n]
    return { code: 0, output: `{"value": ${v}}` }
  }
  const cfg = { workloads: [{ id: 'ar', cmd: 'w{N}', threads: [1, 2, 4, 8], runs: 1, floors: { 4: 2.8, 8: 4.5 } }], serial: { on: 'on', off: 'off', runs: 1, maxTaxPct: 2 } }
  const report = runScaling(cfg, { run, load: quiet })
  assert.equal(report.ok, true)
  assert.deepEqual(report.workloads[0].speedup, { 1: 1, 2: 1.9, 4: 3.33, 8: 4 })
  assert.deepEqual(report.misses, ['ar: speedup(8) = 4 < 4.5', 'serial tax 3% > 2%'])
  const gated = runScaling(cfg, { run, load: quiet, gate: true })
  assert.equal(gated.ok, false)
  const strong = runScaling({ workloads: [{ id: 's', cmd: 'w{N}', threads: [1, 2], runs: 1, mode: 'strong' }] }, { run: cmd => ({ code: 0, output: `{"value": ${cmd === 'w1' ? 10 : 5}}` }), load: quiet })
  assert.deepEqual(strong.workloads[0].speedup, { 1: 1, 2: 2 })
  assert.throws(() => runScaling(cfg, { run, load: () => 9, gate: true }), NoisyMachine)
})

test('scalebench validity: same data order and initial weights everywhere, at least two systems', async () => {
  const { checkScalebench } = await import('../scalebench-check.mjs')
  const run = (system, nodes, d = 'aaa', w = 'bbb', f = 'fff') => ({ system, nodes, rep: 1, checksums: { dataOrder: d, initWeights: w, finalState: f } })
  const arms = { arms: [{ id: 'no-wan', of: 'ours', varies: 'network profile (datacenter)' }], attribution: [{ arm: 'no-wan', factor: 'WAN latency', sharePct: 70 }, { arm: 'residual', factor: 'compute', sharePct: 30 }] }
  const armRun = { ...run('ours', 8), arm: 'no-wan' }
  assert.equal(checkScalebench({ ours: 'ours', runs: [run('ours', 1), run('ours', 8), run('ddp', 8), armRun], ...arms }).ok, true)
  const bad = checkScalebench({ ours: 'ours', runs: [run('ours', 1), run('ddp', 8, 'zzz'), { system: 'x', nodes: 2, rep: 1 }] })
  assert.equal(bad.ok, false)
  assert.ok(bad.problems.some(p => /dataOrder differs/.test(p)))
  assert.ok(bad.problems.some(p => /missing checksum initWeights/.test(p)))
  assert.match(checkScalebench({ ours: 'ours', runs: [run('ours', 1)] }).problems[0], /at least two systems/)
})

test('scalebench byte-exact rule: our final state is identical at every node count and equal to the reference; baselines may differ', async () => {
  const { checkScalebench } = await import('../scalebench-check.mjs')
  const run = (system, nodes, f) => ({ system, nodes, rep: 1, checksums: { dataOrder: 'd', initWeights: 'w', ...(f ? { finalState: f } : {}) } })
  // A lossy baseline (DiLoCo-style) ends elsewhere: allowed, it is not ours.
  const arms = { arms: [{ id: 'one-region', of: 'ours', varies: 'network profile' }], attribution: [{ arm: 'one-region', factor: 'latency', sharePct: 100 }] }
  assert.equal(checkScalebench({ ours: 'ours', reference: 'sync', runs: [run('ours', 1, 'f'), run('ours', 8, 'f'), { ...run('ours', 8, 'f'), arm: 'one-region' }, run('sync', 1, 'f'), run('diloco', 8, 'g')], ...arms }).ok, true)
  // One bit of drift at 8 nodes invalidates the study.
  const drift = checkScalebench({ ours: 'ours', reference: 'sync', runs: [run('ours', 1, 'f'), run('ours', 8, 'f2'), run('sync', 1, 'f')] })
  assert.equal(drift.ok, false)
  assert.ok(drift.problems.some(p => /finalState is not byte-identical/.test(p)))
  // Ours identical to itself but not to the reference: still invalid.
  assert.equal(checkScalebench({ ours: 'ours', reference: 'sync', runs: [run('ours', 1, 'f'), run('sync', 1, 'g')] }).ok, false)
  assert.ok(checkScalebench({ ours: 'ours', runs: [run('ours', 1), run('sync', 1, 'f')] }).problems.some(p => /missing checksum finalState/.test(p)))
  assert.ok(checkScalebench({ runs: [run('ours', 1, 'f'), run('sync', 1, 'f')] }).problems.some(p => /results\.ours must name/.test(p)))
  assert.ok(checkScalebench({ ours: 'ours', reference: 'nope', runs: [run('ours', 1, 'f'), run('sync', 1, 'f')] }).problems.some(p => /results\.reference names no system/.test(p)))
})

test('performance evidence names the machine it was measured on', async () => {
  const { machineInfo } = await import('../bench-gate.mjs')
  const m = machineInfo()
  assert.ok(m.cores >= 1 && m.memoryGb >= 1 && typeof m.cpu === 'string' && Array.isArray(m.gpus))
})

test('scalebench discriminating arms: one factor per arm, runs for every arm, attribution that adds up', async () => {
  const { checkArms } = await import('../scalebench-check.mjs')
  const runs = [{ system: 'ours', nodes: 4, rep: 1 }, { system: 'ours', nodes: 4, rep: 1, arm: 'flat' }]
  const good = { runs, arms: [{ id: 'flat', of: 'ours', varies: 'tensor serialization' }], attribution: [{ arm: 'flat', factor: 'serialization', sharePct: 60 }, { arm: 'residual', factor: 'other', sharePct: 45 }] }
  assert.deepEqual(checkArms(good), [])
  assert.match(checkArms({ runs }).join(), /no discriminating arms/)
  const bad = checkArms({ runs: [...runs, { system: 'ours', nodes: 4, rep: 2, arm: 'ghost' }], arms: [{ id: 'flat', of: 'ours', varies: 'serialization and compression' }, { id: 'idle', of: 'nope', varies: 'x' }], attribution: [{ arm: 'flat', sharePct: 30 }, { arm: 'zzz', sharePct: 'lots' }] })
  for (const want of [/exactly one factor/, /"of" names no system/, /arm idle: no runs/, /unknown arm "ghost"/, /unknown arm "zzz"/, /sharePct must be a number/, /sum to 30%/]) assert.ok(bad.some(p => want.test(p)), `${want}: ${bad.join('; ')}`)
})
