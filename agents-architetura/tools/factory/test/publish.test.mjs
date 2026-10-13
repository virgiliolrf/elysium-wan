import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, existsSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync, spawnSync } from 'node:child_process'
import { start, commit, ready, draft, watchLoop, messageFor, checkpointMessage, slugify, guardStaged, loadState, bodyFor, stageEntry } from '../publish.mjs'
import { lockPath as machineLockPath } from '../../verify/machine-lock.mjs'
import { snapshot } from '../../wf/scope-audit.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const CLI = join(HERE, '..', 'publish.mjs')
const NOREPLY = '1+dev@users.noreply.github.com'
const git = (root, ...a) => execFileSync('git', a, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim()
const put = (root, f, text) => { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), text) }

// A repo on main with one commit, pushed to a local bare "origin".
function repo({ email = NOREPLY } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'publish-'))
  const root = join(dir, 'work')
  const remote = join(dir, 'origin.git')
  execFileSync('git', ['init', '-q', '--bare', '-b', 'main', remote])
  mkdirSync(root)
  git(root, 'init', '-q', '-b', 'main')
  git(root, 'config', 'user.email', email)
  git(root, 'config', 'user.name', 'Dev')
  put(root, '.gitignore', '.factory/\n')
  put(root, 'README.md', 'hello\n')
  git(root, 'add', '-A')
  git(root, 'commit', '-qm', 'init')
  git(root, 'remote', 'add', 'origin', remote)
  git(root, 'push', '-q', '-u', 'origin', 'main')
  return { root, remote }
}

// A fake gh: records calls, keeps PR state.
function fakeGh() {
  const calls = []
  const prs = []
  const gh = (_root, args) => {
    calls.push(args)
    const [a, b] = args
    if (a === 'pr' && b === 'list') return JSON.stringify(prs.filter(p => p.head === args[args.indexOf('--head') + 1]).map(({ number, url, isDraft }) => ({ number, url, isDraft })))
    if (a === 'pr' && b === 'create') { const n = prs.length + 1; prs.push({ number: n, url: `https://github.com/o/r/pull/${n}`, isDraft: args.includes('--draft'), head: args[args.indexOf('--head') + 1] }); return prs[n - 1].url }
    if (a === 'pr' && b === 'ready') { const p = prs.find(x => x.number === Number(args[2])); p.isDraft = args.includes('--undo'); return '' }
    if (a === 'pr' && b === 'edit') { const p = prs.find(x => x.number === Number(args[2])); p.body = args[args.indexOf('--body') + 1]; return '' }
    throw new Error(`unexpected gh ${args.join(' ')}`)
  }
  return { gh, calls, prs }
}
const remoteHas = (remote, ref) => spawnSync('git', ['rev-parse', '--verify', '--quiet', ref], { cwd: remote }).status === 0
const lastMessage = root => git(root, 'log', '-1', '--format=%B')

test('messageFor is a pure function of the stage result: verdict, NOT OK, truncation, no attribution trailer', () => {
  const m = messageFor({ task: 'wan-x', stage: 'closeout M0', result: { ok: false, verdict: 'NOT VERIFIED', why: ['pinned verdict NOT VERIFIED (rung:unit)'], failing: Array.from({ length: 15 }, (_, i) => `g${i}`), tree: 'abc' } })
  assert.match(m, /^wan-x: closeout M0 — NOT VERIFIED\n\n/)
  assert.match(m, /why: pinned verdict NOT VERIFIED \(rung:unit\)/)
  assert.match(m, /failing: g0, g1, .*g11, … \(\+3\)/)
  assert.ok(!/Co-Authored-By/i.test(m))
  assert.equal(messageFor({ task: 't', stage: 's', result: { ok: false } }), 't: s — NOT OK\n')
  assert.match(messageFor({ task: 't', stage: 's', result: { ok: true, rounds: 2 } }), /^t: s — ok\n\nrounds: 2\n/)
  assert.equal(messageFor({ task: 't', stage: 's', result: 'junk' }).split('\n')[0], 't: s — done')
  assert.equal(checkpointMessage('t'), 'Checkpoint t work in progress [skip ci]\n')
  assert.equal(slugify('Treinar GPT-2 em GPUs heterogêneas via WAN!'), 'treinar-gpt-2-em-gpus-heterogeneas-via')
  assert.equal(slugify('x'.repeat(50)), 'x'.repeat(40))
  assert.equal(slugify('!!!'), 'task')
})

test('start: only from a clean base, only with a noreply identity; resume is idempotent', () => {
  const { root } = repo()
  put(root, 'factory.js', 'uncommitted\n')
  assert.throws(() => start({ root, task: 'demo' }), e => e.code === 3 && /1 uncommitted path/.test(e.message))
  git(root, 'add', '-A'); git(root, 'commit', '-qm', 'factory')
  assert.throws(() => start({ root, task: 'Bad Slug' }), e => e.code === 2)
  const s = start({ root, task: 'demo' })
  assert.deepEqual([s.branch, s.resumed], ['factory/demo', false])
  assert.equal(git(root, 'branch', '--show-current'), 'factory/demo')
  put(root, 'docs/design/APPROVALS.json', '{}\n') // a human grant between the two runs
  assert.equal(start({ root, task: 'demo' }).resumed, true)
  assert.throws(() => start({ root, task: 'other' }), e => e.code === 3 && /checkout is on factory\/demo/.test(e.message))
  const personal = repo({ email: 'someone@gmail.com' }).root
  assert.throws(() => start({ root: personal, task: 'demo' }), e => e.code === 3 && /noreply/.test(e.message) && !/gmail/.test(e.message))
})

test('commit: one descriptive commit per stage, pushed to factory/<task>; the draft PR is opened once; main is never pushed', () => {
  const { root, remote } = repo()
  const { gh, calls, prs } = fakeGh()
  const mainBefore = git(remote, 'rev-parse', 'main')
  assert.throws(() => commit({ root, stage: 'survey', gh }), e => e.code === 4) // still on main
  start({ root, task: 'demo' })
  put(root, 'docs/design/CURRENT_ARCHITECTURE.md', '# map\n')
  const a = commit({ root, stage: 'survey', result: { ok: true, components: 3 }, gh })
  assert.equal(a.committed, true)
  assert.equal(a.pushed, true)
  assert.equal(a.pr.number, 1)
  assert.equal(lastMessage(root), 'demo: survey — ok\n\ncomponents: 3')
  assert.ok(remoteHas(remote, 'refs/heads/factory/demo'))
  assert.equal(prs[0].isDraft, true)
  assert.match(calls.find(c => c[1] === 'create').join(' '), /--draft --base main --head factory\/demo/)
  put(root, 'docs/design/BRIEF.md', '# brief\n')
  const b = commit({ root, stage: 'spec', result: { ok: false, why: 'did not converge' }, gh })
  assert.equal(b.pr.number, 1)
  assert.equal(calls.filter(c => c[1] === 'create').length, 1)
  assert.match(lastMessage(root), /^demo: spec — NOT OK\n\nwhy: did not converge/)
  const none = commit({ root, stage: 'noop', gh })
  assert.deepEqual([none.committed, none.pushed], [false, false])
  assert.equal(git(remote, 'rev-parse', 'main'), mainBefore)
  assert.equal(git(remote, 'rev-parse', 'factory/demo'), git(root, 'rev-parse', 'HEAD'))
  assert.equal(loadState(root).pr.number, 1)
})

test('commit guards: a secret, key material or an oversized file refuses the commit, unstaged, the secret never echoed', () => {
  const { root, remote } = repo()
  const { gh } = fakeGh()
  start({ root, task: 'demo' })
  const fake = 'ghp_' + 'A1b2C3d4'.repeat(5)
  put(root, 'src/client.py', `TOKEN = "${fake}"\n`)
  assert.throws(() => commit({ root, stage: 'implement', gh }), e => {
    assert.equal(e.code, 3)
    assert.match(e.message, /src\/client\.py:1 \(GitHub token\)/)
    assert.ok(!e.message.includes(fake))
    return true
  })
  assert.equal(git(root, 'diff', '--cached', '--name-only'), '')
  assert.ok(existsSync(join(root, 'src/client.py')))
  assert.ok(!remoteHas(remote, 'refs/heads/factory/demo'))
  // A reviewed false positive is allowed on its own line.
  put(root, 'src/client.py', `EXAMPLE = "${fake}"  # secret-scan: allow (documented fake token for the parser test)\n`)
  assert.equal(commit({ root, stage: 'implement', gh }).committed, true)
  put(root, 'deploy/server.pem', 'x\n')
  assert.throws(() => commit({ root, stage: 'x', gh }), e => e.code === 3 && /server\.pem \(key material/.test(e.message))
  git(root, 'clean', '-fdq', 'deploy')
  put(root, 'big.bin', 'x'.repeat(64))
  assert.throws(() => commit({ root, stage: 'x', gh, maxBytes: 32 }), e => e.code === 3 && /big\.bin \(larger than/.test(e.message))
  git(root, 'clean', '-fq', 'big.bin')
  git(root, 'add', '-A')
  assert.deepEqual(guardStaged(root, (r, a) => git(r, ...a)), [])
})

test('checkpoint: [skip ci] commit only when something changed; skipped while a heavy job holds the machine lock', () => {
  const { root } = repo()
  const { gh } = fakeGh()
  start({ root, task: 'demo' })
  assert.deepEqual(commit({ root, checkpoint: true, gh }), { ok: true, committed: false, pushed: false })
  put(root, 'src/a.py', 'x = 1\n')
  const c = commit({ root, checkpoint: true, gh })
  assert.equal(c.committed, true)
  assert.match(lastMessage(root), /^Checkpoint demo work in progress \[skip ci\]/)
  put(root, 'src/a.py', 'x = 2\n')
  // the machine lock is machine-wide (machine-lock.mjs), not inside the checkout
  mkdirSync(dirname(machineLockPath()), { recursive: true })
  writeFileSync(machineLockPath(), JSON.stringify({ pid: process.pid, label: 'bench', since: 'now' }))
  try {
    const busy = commit({ root, checkpoint: true, gh })
    assert.match(busy.skipped, /machine lock held by bench/)
    assert.equal(git(root, 'status', '--porcelain'), 'M src/a.py')
  } finally { rmSync(machineLockPath(), { force: true }) }
})

test('ready: only for the tree closeout VERIFIED (plus its own report), committed and pushed; draft puts it back', () => {
  const { root } = repo()
  const { gh, prs } = fakeGh()
  start({ root, task: 'demo' })
  put(root, 'src/a.py', 'x = 1\n')
  commit({ root, stage: 'implement', gh })
  const verified = snapshot(root)
  // closeout writes its report and lessons after the verifier ran: allowed.
  put(root, 'agents-architetura/docs/reports/M0.md', 'Status: VERIFIED\n')
  put(root, 'agents-architetura/docs/LESSONS.md', '- lesson\n')
  assert.throws(() => ready({ root, tree: verified, gh }), e => e.code === 3 && /uncommitted/.test(e.message))
  commit({ root, stage: 'closeout M0', result: { ok: true, verdict: 'VERIFIED' }, gh })
  assert.deepEqual(ready({ root, tree: verified, gh }).ready, true)
  assert.equal(prs[0].isDraft, false)
  assert.equal(draft({ root, gh }).draft, true)
  assert.equal(prs[0].isDraft, true)
  // Code changed after the verdict: refused.
  put(root, 'src/a.py', 'x = 2\n')
  commit({ root, stage: 'late edit', gh })
  assert.throws(() => ready({ root, tree: verified, gh }), e => e.code === 3 && /changed after closeout VERIFIED it: src\/a\.py/.test(e.message))
  assert.equal(prs[0].isDraft, true)
  assert.throws(() => ready({ root, tree: 'nope', gh }), e => e.code === 2)
})

test('watchLoop: checkpoints on a timer, exits after idle ticks or when the checkout leaves the factory branch', async () => {
  let n = 0
  const idle = await watchLoop({ everyMs: 0, idleExit: 3, sleep: async () => {}, onBranch: () => true, tick: () => ({ committed: ++n <= 2 }) })
  assert.deepEqual(idle, { stopped: 'idle for 3 ticks', ticks: 5 })
  let k = 0
  const left = await watchLoop({ everyMs: 0, idleExit: 99, sleep: async () => {}, onBranch: () => ++k < 3, tick: () => ({ committed: true }) })
  assert.deepEqual(left, { stopped: 'left the factory branch', ticks: 2 })
  const errors = await watchLoop({ everyMs: 0, idleExit: 2, sleep: async () => {}, onBranch: () => true, tick: () => { throw new Error('index.lock') } })
  assert.equal(errors.stopped, 'idle for 2 ticks')
})

test('CLI: options are validated before anything runs; exit codes 2 (usage) and 4 (not a factory branch)', () => {
  const { root } = repo()
  const run = (...a) => spawnSync(process.execPath, [CLI, ...a, '--root', root], { encoding: 'utf8' })
  assert.equal(run('frobnicate').status, 2)
  const bad = run('start', '--task', 'demo', '--bogus', 'x')
  assert.equal(bad.status, 2)
  assert.equal(git(root, 'branch', '--show-current'), 'main')
  const off = run('commit', '--stage', 'survey', '--b64', Buffer.from('{"ok":true}').toString('base64'))
  assert.equal(off.status, 4)
  assert.match(JSON.parse(off.stdout).reason, /not on a factory\/\* branch/)
  assert.equal(run('commit', '--stage', 's', '--b64', '%%%').status, 2)
  const st = JSON.parse(run('status').stdout)
  assert.equal(st.branch, 'main')
  assert.equal(st.watching, false)
  assert.equal(readFileSync(join(root, 'README.md'), 'utf8'), 'hello\n')
})

test('the PR description is a living bring-up log derived from stage results: status, every stage, an honest section', () => {
  const { root } = repo()
  const { gh, prs } = fakeGh()
  start({ root, task: 'microbatch', text: 'Speed-proportional microbatch allocation, byte-exact.' })
  put(root, 'docs/a.md', 'a\n')
  const a = commit({ root, stage: 'survey', result: { ok: true, components: 3 }, gh })
  assert.equal(a.description, 'updated')
  assert.match(prs[0].body, /## What this is\n> Speed-proportional microbatch allocation, byte-exact\./)
  assert.match(prs[0].body, /## Status, current\n\*\*survey\*\*: ok \(components: 3\)/)
  put(root, 'docs/b.md', 'b\n')
  commit({ root, stage: 'cycle M2', result: { ok: false, why: 'oracle red: finalState differs at step 40' }, gh })
  commit({ root, stage: 'closeout M2', result: { verdict: 'NOT VERIFIED', failing: ['rung:oracle'] }, gh })
  const body = prs[0].body
  assert.match(body, /1\. `survey`: ok[\s\S]*2\. `cycle M2`: NOT OK \(why: oracle red[\s\S]*3\. `closeout M2`: NOT VERIFIED/)
  assert.match(body, /## The honest section\n- `cycle M2` was red: why: oracle red: finalState differs at step 40\n- `closeout M2` was red/)
  assert.match(body, /No agent wrote this description/)
  start({ root, task: 'microbatch' })
  assert.equal(loadState(root).text, 'Speed-proportional microbatch allocation, byte-exact.', 'a resume keeps the task text and the log')
  assert.equal(loadState(root).log.length, 3)
  assert.equal(stageEntry({ stage: 'x', result: { verdict: 'VERIFIED' } }).red, false)
  assert.match(bodyFor({ task: 't', branch: 'factory/t' }), /No stage has finished yet/)
})
