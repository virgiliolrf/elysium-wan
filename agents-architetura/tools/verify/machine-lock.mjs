#!/usr/bin/env node
// Machine lock: test runs share one machine, whichever checkout or session starts them.
// - Light runs (an agent's own tests, a ladder's light rungs) take one of N slots (FACTORY_TEST_SLOTS, default 2).
// - Heavy runs (sanitizer and fuzz campaigns, load runs, benchmarks, heavy ladder rungs, pinned gate entries) take the whole
//   machine: the exclusive lock, once no slot is held.
// The locks live in one machine-wide directory (FACTORY_LOCK_DIR, default ~/.factory-locks), never inside a checkout. They
// lived in <checkout>/.factory before: two builds in two checkouts never saw each other, ran their heavy tests together, and
// the 16 GB Mac panicked (2026-10-09). Each lock is a file created atomically holding {pid, label, since}; a lock whose pid is dead is
// reclaimed. A holder exports FACTORY_MACHINE_LOCK_HELD (exclusive) or FACTORY_TEST_SLOT_HELD (its slot) to everything it
// runs, so a job started BY the holder re-enters instead of waiting on its own parent.
//
// A run started with --slot is killed (its whole process group) after --timeout SECS, default SLOT_TIMEOUT_SEC, and exits 124:
// a light test that runs that long is hung, not slow. A deadlocked mutant kept pytest at 0% CPU for 20 minutes and held one of
// the two slots (2026-10-09); without the timeout it would have held it forever.
//
//   node agents-architetura/tools/verify/machine-lock.mjs status
//   node agents-architetura/tools/verify/machine-lock.mjs run [--slot] --label <what> [--wait SECS] [--timeout SECS] -- <command...>
import { writeFileSync, linkSync, statSync, readFileSync, rmSync, mkdirSync } from 'node:fs'
import { join, dirname, resolve } from 'node:path'
import { homedir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { spawn } from 'node:child_process'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
export const HELD_ENV = 'FACTORY_MACHINE_LOCK_HELD'
export const SLOT_ENV = 'FACTORY_TEST_SLOT_HELD'
// The root argument is kept for the callers' signatures; the directory is machine-wide on purpose.
export const lockDir = () => process.env.FACTORY_LOCK_DIR || join(homedir(), '.factory-locks')
export const lockPath = () => join(lockDir(), 'exclusive.lock')
const slotPath = i => join(lockDir(), `slot-${i}.lock`)
// The slot count: FACTORY_TEST_SLOTS, else the machine-wide file <lockDir>/slots (one integer), else 2. The file is read on
// every call, so the machine's operator can widen or narrow the test slots while sessions run, without restarting them
// (sessions keep the environment they started with). The human widened the 16 GB Mac to 4 slots on 2026-10-11.
const SLOTS_FILE = () => join(lockDir(), 'slots')
export const slotCount = () => {
  let raw = process.env.FACTORY_TEST_SLOTS
  if (raw === undefined) { try { raw = readFileSync(SLOTS_FILE(), 'utf8').trim() } catch { raw = '2' } }
  const n = Number(raw)
  return Number.isInteger(n) && n >= 1 && n <= 16 ? n : 2
}
export const SLOT_TIMEOUT_SEC = 1200
export const TIMEOUT_EXIT = 124

const alive = pid => {
  try { process.kill(pid, 0); return true } catch (e) { return e.code === 'EPERM' }
}
const readJson = p => { try { return JSON.parse(readFileSync(p, 'utf8')) } catch { return null } }
const liveHolder = p => { const h = readJson(p); return h && Number.isInteger(h.pid) && alive(h.pid) ? h : null }

// Atomic create of one lock file, already holding its {pid, label, since}: the body goes to a private temp file, which is
// hard-linked to the lock path (EEXIST if taken). The lock used to be opened O_EXCL and written after: in between it
// was empty, a second claimer read it as unreadable, removed it as stale and took the same slot. Four slot runs woken
// together on 2026-10-11 left two of them running with no slot file, so a heavy run could start beside them.
// A dead holder is reclaimed once. An unreadable file younger than YOUNG_MS is still being written by an older copy of
// this tool (other checkouts run their own), so it counts as held. Returns {ok} or {ok:false, holder}.
const YOUNG_MS = 10_000
const young = p => { try { return Date.now() - statSync(p).mtimeMs < YOUNG_MS } catch { return false } }
const claim = (p, label, pid) => {
  mkdirSync(dirname(p), { recursive: true })
  const tmp = `${p}.${pid}.${process.hrtime.bigint()}.tmp`
  writeFileSync(tmp, JSON.stringify({ pid, label, since: new Date().toISOString() }))
  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        linkSync(tmp, p)
        return { ok: true }
      } catch (e) {
        if (e.code !== 'EEXIST') throw e
        const holder = liveHolder(p)
        if (holder) return { ok: false, holder }
        if (readJson(p) === null && young(p)) return { ok: false, holder: { label: '(lock being written)' } }
        rmSync(p, { force: true }) // dead holder, or unreadable and old: reclaim once
      }
    }
    return { ok: false, holder: readJson(p) }
  } finally {
    rmSync(tmp, { force: true })
  }
}

export function readLock() {
  return readJson(lockPath())
}

export function status() {
  return { exclusive: liveHolder(lockPath()), slots: Array.from({ length: slotCount() }, (_, i) => liveHolder(slotPath(i))) }
}

// Exclusive: the whole machine. Taken only when no slot is held (a slot this process tree holds does not count).
// Returns {ok:true} or {ok:false, holder}. Never blocks.
export function tryAcquire(root, label, pid = process.pid) {
  if (process.env[HELD_ENV]) return { ok: true, reentrant: true }
  const r = claim(lockPath(), label, pid)
  if (!r.ok) return r
  const mine = process.env[SLOT_ENV]
  for (let i = 0; i < slotCount(); i++) {
    if (`slot-${i}` === mine) continue
    const holder = liveHolder(slotPath(i))
    if (holder) { rmSync(lockPath(), { force: true }); return { ok: false, holder } }
  }
  return { ok: true }
}

export function release(root, pid = process.pid) {
  const holder = readLock()
  if (holder?.pid === pid) rmSync(lockPath(), { force: true })
}

// Slot: one of N light-test slots, refused while the machine is held exclusively. Returns {ok, slot} or {ok:false, holder}.
export function tryAcquireSlot(label, pid = process.pid) {
  if (process.env[SLOT_ENV] || process.env[HELD_ENV]) return { ok: true, reentrant: true }
  const ex = liveHolder(lockPath())
  if (ex) return { ok: false, holder: ex }
  let holder = null
  for (let i = 0; i < slotCount(); i++) {
    const r = claim(slotPath(i), label, pid)
    if (r.ok) {
      // an exclusive taken between the check and the claim wins: give the slot back
      if (liveHolder(lockPath())) { rmSync(slotPath(i), { force: true }); return { ok: false, holder: readLock() } }
      return { ok: true, slot: `slot-${i}` }
    }
    holder = r.holder
  }
  return { ok: false, holder }
}

export function releaseSlot(slot, pid = process.pid) {
  if (!slot) return
  const p = join(lockDir(), `${slot}.lock`)
  if (readJson(p)?.pid === pid) rmSync(p, { force: true })
}

const sleep = ms => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms)

const waitFor = (attempt, { waitSec = 0, pollMs = 500 } = {}) => {
  const deadline = performance.now() + waitSec * 1000
  for (;;) {
    const r = attempt()
    if (r.ok || performance.now() >= deadline) return r
    sleep(pollMs)
  }
}

export function acquire(root, label, opts) {
  return waitFor(() => tryAcquire(root, label), opts)
}

export function acquireSlot(label, opts) {
  return waitFor(() => tryAcquireSlot(label), opts)
}

export function withMachineLock(root, label, fn, opts) {
  const r = acquire(root, label, opts)
  if (!r.ok) throw new Error(`machine is busy: held by ${JSON.stringify(r.holder)}`)
  if (r.reentrant) return fn()
  process.env[HELD_ENV] = label
  try { return fn() } finally { delete process.env[HELD_ENV]; release(root) }
}

export function withTestSlot(label, fn, opts) {
  const r = acquireSlot(label, opts)
  if (!r.ok) throw new Error(`no test slot free: held by ${JSON.stringify(r.holder)}`)
  if (r.reentrant) return fn()
  process.env[SLOT_ENV] = r.slot
  try { return fn() } finally { delete process.env[SLOT_ENV]; releaseSlot(r.slot) }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const sep = argv.indexOf('--')
  const head = sep >= 0 ? argv.slice(0, sep) : argv
  // One argument after -- is a shell command line, run as written. Several are an argv: each is quoted, so an argument
  // with spaces stays one argument. A slot run of `pytest -k "a or b"` used to reach bash as `-k a or b`, and every verify
  // command with such a selection failed before pytest started (c14's M4 coders, 2026-10-11).
  const quote = a => (/^[\w@%+=:,./-]+$/.test(a) ? a : `'${a.replace(/'/g, `'\\''`)}'`)
  const rest = sep >= 0 ? argv.slice(sep + 1) : []
  const cmd = rest.length === 1 ? rest[0] : rest.map(quote).join(' ')
  const opt = (name, d) => { const i = head.indexOf(name); return i >= 0 ? head[i + 1] : d }
  const root = resolve(opt('--root', ROOT))
  if (head[0] === 'status') {
    console.log(JSON.stringify(status()))
  } else if (head[0] === 'run' && cmd) {
    const label = opt('--label', 'job')
    const slot = head.includes('--slot')
    const r = slot ? acquireSlot(label, { waitSec: Number(opt('--wait', 0)) }) : acquire(root, label, { waitSec: Number(opt('--wait', 0)) })
    if (!r.ok) { console.error(`machine-lock: busy, held by ${JSON.stringify(r.holder)}`); process.exit(75) }
    const env = { ...process.env, ...(slot ? (r.slot ? { [SLOT_ENV]: r.slot } : {}) : { [HELD_ENV]: label }) }
    const timeoutSec = Number(opt('--timeout', slot ? SLOT_TIMEOUT_SEC : 0))
    const done = code => { if (!r.reentrant) (slot ? releaseSlot(r.slot) : release(root)); process.exit(code) }
    // its own process group, so a timeout kills the test and everything it started (workers, stages), not only bash
    const child = spawn('/bin/bash', ['-o', 'pipefail', '-c', cmd], { stdio: 'inherit', cwd: root, env, detached: true })
    let timedOut = false
    const killGroup = sig => { try { process.kill(-child.pid, sig) } catch {} }
    const timer = timeoutSec > 0 ? setTimeout(() => {
      timedOut = true
      console.error(`machine-lock: ${label} ran past --timeout ${timeoutSec}s and was killed: a hang (deadlock, missing timeout in the code), not a slow test`)
      killGroup('SIGTERM')
      setTimeout(() => killGroup('SIGKILL'), 5000).unref()
    }, timeoutSec * 1000) : null
    for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => { killGroup(sig); done(130) })
    child.on('error', () => { if (timer) clearTimeout(timer); done(1) })
    child.on('exit', code => { if (timer) clearTimeout(timer); done(timedOut ? TIMEOUT_EXIT : (code ?? 1)) })
  } else {
    console.error('usage: machine-lock.mjs status | run [--slot] --label <what> [--wait SECS] [--timeout SECS] -- <command...>')
    process.exit(2)
  }
}
