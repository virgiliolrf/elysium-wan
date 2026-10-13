import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, existsSync, readFileSync, readdirSync, utimesSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { tryAcquire, release, readLock, lockPath, lockDir, withMachineLock, tryAcquireSlot, releaseSlot, withTestSlot, status, slotCount } from '../machine-lock.mjs'

const CLI = join(dirname(fileURLToPath(import.meta.url)), '..', 'machine-lock.mjs')
// these tests own the locks; an ambient holder (e.g. a locked campaign) must not leak in
delete process.env.FACTORY_MACHINE_LOCK_HELD
delete process.env.FACTORY_TEST_SLOT_HELD
// every test gets its own machine-wide lock directory
const fresh = () => { process.env.FACTORY_LOCK_DIR = mkdtempSync(join(tmpdir(), 'locks-')); process.env.FACTORY_TEST_SLOTS = '2'; return mkdtempSync(join(tmpdir(), 'checkout-')) }
const cli = (args, extra = {}) => spawnSync(process.execPath, [CLI, ...args], { encoding: 'utf8', env: { ...process.env, ...extra } })

test('a second acquirer is refused while the holder lives; release frees it', () => {
  const r = fresh()
  assert.equal(tryAcquire(r, 'campaign').ok, true)
  const second = tryAcquire(r, 'bench', process.ppid) // a different, live pid
  assert.equal(second.ok, false)
  assert.equal(second.holder.label, 'campaign')
  release(r)
  assert.equal(readLock(r), null)
  assert.equal(tryAcquire(r, 'bench').ok, true)
  release(r)
})

test('the lock is machine-wide: two checkouts see the same lock (they did not, and the Mac panicked)', () => {
  const a = fresh()
  const b = mkdtempSync(join(tmpdir(), 'checkout-'))
  assert.equal(tryAcquire(a, 'build-in-checkout-a').ok, true)
  const other = tryAcquire(b, 'build-in-checkout-b', process.ppid)
  assert.equal(other.ok, false, 'another checkout waits for the same machine')
  assert.ok(!lockPath(a).startsWith(a), 'the lock does not live inside a checkout')
  release(a)
})

test('a lock held by a dead pid is reclaimed', () => {
  const r = fresh()
  mkdirSync(lockDir(), { recursive: true })
  writeFileSync(lockPath(r), JSON.stringify({ pid: 2 ** 22 + 12345, label: 'ghost', since: 'x' }))
  assert.equal(tryAcquire(r, 'fresh').ok, true)
  assert.equal(readLock(r).label, 'fresh')
  release(r)
})

test('withMachineLock releases on throw; a busy machine fails fast', () => {
  const r = fresh()
  assert.throws(() => withMachineLock(r, 'x', () => { throw new Error('inner') }), /inner/)
  assert.equal(existsSync(lockPath(r)), false)
  mkdirSync(lockDir(), { recursive: true })
  writeFileSync(lockPath(r), JSON.stringify({ pid: process.ppid, label: 'other' }))
  assert.throws(() => withMachineLock(r, 'x', () => 1), /machine is busy/)
})

test('slots: N light runs together; a heavy run waits for them; light runs wait while the machine is held', () => {
  const r = fresh()
  const s1 = tryAcquireSlot('task-a')
  const s2 = tryAcquireSlot('task-b', process.ppid)
  assert.deepEqual([s1.ok, s2.ok], [true, true])
  assert.equal(tryAcquireSlot('task-c', process.ppid).ok, false, 'only 2 slots')
  assert.equal(tryAcquire(r, 'heavy', process.ppid).ok, false, 'a heavy run waits until no slot is held')
  releaseSlot(s1.slot)
  releaseSlot(s2.slot, process.ppid)
  assert.equal(tryAcquire(r, 'heavy').ok, true)
  assert.equal(tryAcquireSlot('task-d', process.ppid).ok, false, 'no light run while the machine is held')
  release(r)
  assert.deepEqual(status(), { exclusive: null, slots: [null, null] })
})

test('a lock file still being written is held, not reclaimed; an old unreadable one is reclaimed; no temp file is left', () => {
  fresh()
  mkdirSync(lockDir(), { recursive: true })
  const slot0 = join(lockDir(), 'slot-0.lock')
  writeFileSync(slot0, '') // another claimer created it and has not written its body yet
  const s = tryAcquireSlot('task-a')
  assert.equal(s.slot, 'slot-1', 'the empty slot-0 belongs to a claimer mid-write: take the next slot')
  assert.equal(readFileSync(slot0, 'utf8'), '', 'and leave its file alone (it was removed as stale, and two runs shared one slot)')
  releaseSlot(s.slot)
  const old = (Date.now() - 60_000) / 1000
  utimesSync(slot0, old, old)
  const t = tryAcquireSlot('task-b')
  assert.equal(t.slot, 'slot-0', 'an unreadable file a minute old is stale')
  assert.equal(JSON.parse(readFileSync(slot0, 'utf8')).label, 'task-b', 'the lock holds its body from the moment it exists')
  releaseSlot(t.slot)
  assert.deepEqual(readdirSync(lockDir()).filter(f => f.endsWith('.tmp')), [])
})

test('withTestSlot releases its slot; a heavy run started inside a slot does not wait on its own slot', () => {
  const r = fresh()
  assert.equal(withTestSlot('t', () => withMachineLock(r, 'heavy-inside', () => 'ok')), 'ok')
  assert.deepEqual(status(), { exclusive: null, slots: [null, null] })
})

test('CLI run: holds the lock during the command, releases after, propagates the exit code; --slot takes a slot', () => {
  const r = fresh()
  const p = cli(['run', '--root', r, '--label', 'job', '--', `cat ${lockPath(r)}; exit 3`])
  assert.equal(p.status, 3)
  assert.match(p.stdout, /"label":"job"/)
  assert.equal(existsSync(lockPath(r)), false)
  const slot = cli(['run', '--slot', '--label', 'unit', '--', `cat ${join(lockDir(), 'slot-0.lock')}`])
  assert.equal(slot.status, 0, slot.stderr)
  assert.match(slot.stdout, /"label":"unit"/)
  mkdirSync(lockDir(), { recursive: true })
  writeFileSync(lockPath(r), JSON.stringify({ pid: process.pid, label: 'held' }))
  assert.equal(cli(['run', '--root', r, '--label', 'job', '--', 'true']).status, 75)
  assert.equal(cli(['run', '--slot', '--label', 'unit', '--', 'true']).status, 75, 'a slot is refused while the machine is held')
})

test('re-entrant: a job started by the holder does not wait on its own parent', () => {
  const r = fresh()
  const inner = `${JSON.stringify(process.execPath)} ${JSON.stringify(CLI)} run --root ${r} --label inner -- 'echo inner-ran'`
  const p = cli(['run', '--root', r, '--label', 'outer', '--', inner])
  assert.equal(p.status, 0, p.stderr)
  assert.match(p.stdout, /inner-ran/)
  assert.equal(existsSync(lockPath(r)), false)
  assert.equal(withMachineLock(r, 'x', () => withMachineLock(r, 'nested', () => 'ok')), 'ok')
  assert.equal(existsSync(lockPath(r)), false)
})

test('a slot run past --timeout is killed with everything it started, exits 124 and frees its slot', () => {
  fresh()
  const t0 = Date.now()
  // the grandchild holds the pipe open like a hung stage; only a process-group kill ends it
  const p = cli(['run', '--slot', '--label', 'hung', '--timeout', '1', '--', 'sleep 30 & sleep 30; wait'])
  assert.equal(p.status, 124, p.stderr)
  assert.match(p.stderr, /ran past --timeout 1s and was killed: a hang/)
  assert.ok(Date.now() - t0 < 15000, 'killed at the timeout, not after the sleeps')
  assert.deepEqual(status().slots, [null, null], 'the slot is free again')
  assert.equal(cli(['run', '--slot', '--label', 'quick', '--timeout', '5', '--', 'exit 7']).status, 7, 'a run inside its timeout keeps its own exit code')
})

test('slot count: FACTORY_TEST_SLOTS wins; else the machine-wide slots file, read on every call; else 2', () => {
  fresh()
  delete process.env.FACTORY_TEST_SLOTS
  assert.equal(slotCount(), 2, 'no env, no file: the default')
  writeFileSync(join(lockDir(), 'slots'), '4\n')
  assert.equal(slotCount(), 4, 'the file widens the slots without a restart')
  const held = [0, 1, 2, 3].map(() => tryAcquireSlot('t'))
  assert.ok(held.every(h => h.ok), 'four slots can be held at once')
  assert.equal(tryAcquireSlot('t').ok, false, 'a fifth waits')
  for (const h of held) releaseSlot(h.slot)
  writeFileSync(join(lockDir(), 'slots'), 'many')
  assert.equal(slotCount(), 2, 'garbage falls back to the default')
  process.env.FACTORY_TEST_SLOTS = '3'
  writeFileSync(join(lockDir(), 'slots'), '4')
  assert.equal(slotCount(), 3, 'the environment wins over the file')
})

test('run: one argument after -- is a shell line; several are an argv whose spaces and quotes survive', () => {
  const r = fresh()
  const argv = cli(['run', '--slot', '--label', 't', '--root', r, '--', 'printf', '%s|', 'a or b', "it's", 'c'])
  assert.equal(argv.status, 0, argv.stderr)
  assert.equal(argv.stdout, "a or b|it's|c|")
  const line = cli(['run', '--slot', '--label', 't', '--root', r, '--', 'printf "%s|" x y | tr x z'])
  assert.equal(line.status, 0, line.stderr)
  assert.equal(line.stdout, 'z|y|')
})
