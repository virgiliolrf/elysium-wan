import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync, execFileSync } from 'node:child_process'
import { runMatrix, validateMatrix } from '../matrix.mjs'
import { runIdentity, selectSample } from '../identity.mjs'
import { glob, globToRegExp } from '../glob.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const node = js => `${JSON.stringify(process.execPath)} -e ${JSON.stringify(js)}`

test('glob: ** crosses directories, * does not; build dirs are never walked', () => {
  const root = mkdtempSync(join(tmpdir(), 'glob-'))
  for (const f of ['tests/oracle/test_a.py', 'tests/oracle/deep/test_b.py', 'tests/oracle/helper.py', '.venv/lib/test_x.py', 'node_modules/test_y.py']) {
    mkdirSync(join(root, dirname(f)), { recursive: true })
    writeFileSync(join(root, f), '')
  }
  assert.deepEqual(glob(root, 'tests/oracle/**/test_*.py'), ['tests/oracle/deep/test_b.py', 'tests/oracle/test_a.py'])
  assert.deepEqual(glob(root, 'tests/oracle/test_*.py'), ['tests/oracle/test_a.py'])
  assert.deepEqual(glob(root, '**/test_*.py'), ['tests/oracle/deep/test_b.py', 'tests/oracle/test_a.py'])
  assert.ok(globToRegExp('a.b').test('a.b') && !globToRegExp('a.b').test('axb'))
})

test('matrix: a crashing probe fails, a matching "unsupported" skips, >maxSkipPct skipped is vacuous', () => {
  const cfg = {
    cmd: node("process.exit(process.env.MODE === 'bad' ? 1 : 0)"), maxSkipPct: 50,
    modes: [
      { id: 'ok', env: { MODE: 'ok' } },
      { id: 'bad', env: { MODE: 'bad' } },
      { id: 'nobuild', probe: node("console.error('unsupported mode: lossy'); process.exit(1)"), unsupported: 'unsupported mode' },
      { id: 'crashy', probe: node("process.kill(process.pid, 'SIGSEGV')"), unsupported: 'unsupported mode' },
    ],
  }
  const r = runMatrix(cfg)
  assert.deepEqual(r.modes.map(m => [m.mode, m.status]), [['ok', 'pass'], ['bad', 'fail'], ['nobuild', 'skipped'], ['crashy', 'fail']])
  assert.match(r.modes[3].detail, /probe crash/)
  assert.equal(r.ok, false)
  const vac = runMatrix({ cmd: 'true', modes: [{ id: 'a', probe: 'echo unsupported; exit 1', unsupported: 'unsupported' }, { id: 'b', probe: 'echo unsupported; exit 1', unsupported: 'unsupported' }, { id: 'c' }] })
  assert.equal(vac.vacuous, true)
  assert.equal(vac.ok, false)
  assert.equal(runMatrix({ cmd: 'true', modes: [] }).applicable, false)
  assert.throws(() => validateMatrix({ cmd: 'x', modes: [{ id: 'A' }] }), /bad mode id/)
  assert.throws(() => validateMatrix({ cmd: 'x', modes: [{ id: 'a', env: { lower: '1' } }] }), /UPPER_CASE/)
})

test('matrix CLI: n/a without config, args substituted', () => {
  const dir = mkdtempSync(join(tmpdir(), 'mx-'))
  const cli = join(HERE, '..', 'matrix.mjs')
  assert.equal(spawnSync(process.execPath, [cli, '--config', join(dir, 'none.json')], { encoding: 'utf8' }).status, 5)
  writeFileSync(join(dir, 'm.json'), JSON.stringify({ cmd: 'test "{args}" = "--fast"', modes: [{ id: 'fast', args: '--fast' }] }))
  assert.equal(spawnSync(process.execPath, [cli, '--config', join(dir, 'm.json')], { encoding: 'utf8' }).status, 0)
})

const corpus = files => {
  const root = mkdtempSync(join(tmpdir(), 'ident-'))
  for (const f of files) { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), f) }
  return root
}

test('identity: deterministic sample; identical sides pass; output, exit and artifact differences are named', () => {
  assert.deepEqual(selectSample(['c', 'a', 'b', 'd', 'e'], 2, 2), ['a', 'c'])
  const root = corpus(['tests/oracle/test_1.py', 'tests/oracle/test_2.py'])
  const same = runIdentity({ corpus: ['tests/oracle/test_*.py'], a: 'cat {test}; echo took 0.5s', b: 'cat {test}; echo took 0.7s' }, { root })
  assert.equal(same.ok, true)
  assert.equal(same.sampled, 2)
  const raw = runIdentity({ corpus: ['tests/oracle/test_*.py'], normalize: false, a: 'cat {test}; echo took 0.5s', b: 'cat {test}; echo took 0.7s' }, { root })
  assert.equal(raw.ok, false)
  const art = runIdentity({ corpus: ['tests/oracle/test_1.py'], a: 'test -f {test} && echo w1 > "$FACTORY_IDENTITY_OUT/weights.bin"', b: 'test -f {test} && echo w2 > "$FACTORY_IDENTITY_OUT/weights.bin"' }, { root })
  assert.deepEqual(art.rows[0].mismatch, ['artifact weights.bin differs'])
  const ex = runIdentity({ corpus: ['tests/oracle/test_1.py'], a: 'test -f {test} && exit 1', b: 'test -f {test}' }, { root })
  assert.match(ex.rows[0].mismatch[0], /exit 1\/0/)
  assert.equal(runIdentity({ corpus: ['nothing/*.py'], a: '{test}', b: '{test}' }, { root }).applicable, false)
  assert.throws(() => runIdentity({ corpus: [], a: 'x', b: '{test}' }, { root }), /must be a command containing \{test\}/)
})

test('identity exact (byte-exact rule): features on vs reference compares training state only; any differing or missing state fails', () => {
  const root = corpus(['tests/oracle/test_1.py'])
  // Each side writes state/weights.bin and a wire capture; wire bytes legitimately differ between the WAN path and the reference.
  const side = (weights, wire) => `test -f {test} && mkdir -p "$FACTORY_IDENTITY_OUT/state" "$FACTORY_IDENTITY_OUT/wire" && printf ${weights} > "$FACTORY_IDENTITY_OUT/state/weights.bin" && printf ${wire} > "$FACTORY_IDENTITY_OUT/wire/cap.bin"`
  const cfg = cmd => ({ corpus: ['tests/oracle/test_*.py'], a: side('w', 'ref'), b: side('w', 'ref'), exact: { cmd, state: ['state/**'] } })
  // WAN path: same state, different wire bytes -> byte-exact holds.
  const ok = runIdentity(cfg(side('w', 'compressed')), { root })
  assert.equal(ok.ok, true)
  assert.equal(ok.exact, true)
  assert.deepEqual(ok.rows[0].exactMismatch, [])
  // One differing bit in the state fails, named.
  const bit = runIdentity(cfg(side('x', 'ref')), { root })
  assert.equal(bit.ok, false)
  assert.deepEqual(bit.rows[0].exactMismatch, ['state state/weights.bin is not byte-identical to the reference'])
  // A production path that writes no state proves nothing.
  const none = runIdentity(cfg('test -f {test}'), { root })
  assert.equal(none.ok, false)
  assert.ok(none.rows[0].exactMismatch.some(m => /production path wrote no training state/.test(m)))
  assert.ok(none.rows[0].exactMismatch.some(m => /written only by the reference/.test(m)))
  // A different exit code fails even with identical state.
  assert.match(runIdentity(cfg(`${side('w', 'ref')} && exit 3`), { root }).rows[0].exactMismatch[0], /exit 3\/0/)
  assert.throws(() => runIdentity({ ...cfg('x'), exact: { cmd: 'x', state: ['state/**'] } }, { root }), /exact\.cmd/)
  assert.throws(() => runIdentity({ ...cfg(side('w', 'ref')), exact: { cmd: side('w', 'ref'), state: [] } }, { root }), /exact\.state/)
})

test('identity CLI: a config without the exact check fails the rung (the byte-exact rule would go unverified)', () => {
  const root = corpus(['tests/oracle/test_1.py'])
  const cli = join(HERE, '..', 'identity.mjs')
  const write = cfg => { const p = join(root, 'identity.json'); writeFileSync(p, JSON.stringify(cfg)); return p }
  const base = { corpus: ['tests/oracle/test_*.py'], a: 'cat {test}', b: 'cat {test}' }
  const without = spawnSync(process.execPath, [cli, '--root', root, '--config', write(base)], { encoding: 'utf8' })
  assert.equal(without.status, 1)
  assert.match(without.stdout, /no "exact" check configured/)
  const st = 'mkdir -p "$FACTORY_IDENTITY_OUT/state" && cat {test} > "$FACTORY_IDENTITY_OUT/state/w"'
  const withExact = spawnSync(process.execPath, [cli, '--root', root, '--config', write({ ...base, a: st, b: st, exact: { cmd: st, state: ['state/**'] } })], { encoding: 'utf8' })
  assert.equal(withExact.status, 0, withExact.stdout + withExact.stderr)
})

test('identity --against-base: compares the feature-off run with the base revision in a temp worktree', () => {
  const root = corpus(['tests/oracle/test_1.py', 'value.txt'])
  const git = (...a) => execFileSync('git', a, { cwd: root, encoding: 'utf8' })
  git('init', '-q')
  git('add', '-A')
  git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'base')
  const cfg = { corpus: ['tests/oracle/test_*.py'], a: 'cat {test} value.txt', b: 'cat {test} value.txt', baseRev: 'HEAD' }
  assert.equal(runIdentity(cfg, { root, againstBase: true }).ok, true)
  writeFileSync(join(root, 'value.txt'), 'changed behavior')
  const drift = runIdentity(cfg, { root, againstBase: true })
  assert.equal(drift.ok, false)
  assert.match(drift.rows[0].baseMismatch[0], /output differs/)
  assert.equal(git('worktree', 'list').trim().split('\n').length, 1)
})
