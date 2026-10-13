// Shared process runner for the verification drivers (amplify, load, matrix, identity).
// Every run is classified the same way:
//   pass  exit 0
//   fail  nonzero exit
//   crash killed by a signal (or a shell exit code 128+n reporting one)
//   hang  exceeded its timeout (killed with SIGKILL)
import { spawn, spawnSync } from 'node:child_process'

export const SEED_ENV = 'FACTORY_SEED'
export const PERIOD_ENV = 'FACTORY_PERTURB_PERIOD'
export const MAX_US_ENV = 'FACTORY_PERTURB_MAX_US'

const MAX_BUFFER = 64 * 1024 * 1024

export function classify({ status, signal, timedOut }) {
  if (timedOut) return 'hang'
  if (signal) return 'crash'
  if (status === 0) return 'pass'
  if (status > 128 && status < 160) return 'crash'
  return 'fail'
}

export function runSync(cmd, { cwd, env = {}, timeoutSec = 600 } = {}) {
  const t0 = performance.now()
  const r = spawnSync('/bin/bash', ['-o', 'pipefail', '-c', cmd], {
    cwd, env: { ...process.env, ...env }, encoding: 'utf8', timeout: timeoutSec * 1000, killSignal: 'SIGKILL', maxBuffer: MAX_BUFFER,
  })
  const timedOut = r.error?.code === 'ETIMEDOUT'
  const out = { status: r.status, signal: timedOut ? null : r.signal, timedOut, stdout: r.stdout ?? '', stderr: r.stderr ?? '', ms: Math.round(performance.now() - t0) }
  return { ...out, outcome: classify(out) }
}

export function runAsync(cmd, { cwd, env = {}, timeoutSec = 600 } = {}) {
  return new Promise(resolvePromise => {
    const t0 = performance.now()
    const child = spawn('/bin/bash', ['-o', 'pipefail', '-c', cmd], { cwd, env: { ...process.env, ...env } })
    let stdout = ''
    let stderr = ''
    let timedOut = false
    child.stdout.on('data', d => { if (stdout.length < MAX_BUFFER) stdout += d })
    child.stderr.on('data', d => { if (stderr.length < MAX_BUFFER) stderr += d })
    const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL') }, timeoutSec * 1000)
    child.on('close', (status, signal) => {
      clearTimeout(timer)
      const out = { status, signal: timedOut ? null : signal, timedOut, stdout, stderr, start: t0, end: performance.now(), ms: Math.round(performance.now() - t0) }
      resolvePromise({ ...out, outcome: classify(out) })
    })
  })
}

// Output normalization for divergence detection: durations and addresses vary between correct runs.
export function normalizeOutput(text, ignore = []) {
  const res = ignore.map(r => new RegExp(r))
  return String(text)
    .split('\n')
    .filter(line => !res.some(re => re.test(line)))
    .join('\n')
    // pytest appends "(H:MM:SS)" to a duration only past a minute: drop it, or a slower run reads as a mismatch
    .replace(/(\d(?:\.\d+)?\s?(?:ns|us|µs|ms|s|sec|secs|seconds)) \(\d+:\d{2}:\d{2}(?:\.\d+)?\)/g, '$1')
    .replace(/\b\d+(?:\.\d+)?\s?(?:ns|us|µs|ms|s|sec|secs|seconds)\b/g, '<dur>')
    .replace(/0x[0-9a-fA-F]+/g, '<addr>')
}

export const shellQuote = s => `'${String(s).replace(/'/g, `'\\''`)}'`

// "--" separates driver options from the command; the command is everything after it, joined by spaces.
export function splitArgv(argv) {
  const sep = argv.indexOf('--')
  return { opts: sep >= 0 ? argv.slice(0, sep) : argv, cmd: sep >= 0 ? argv.slice(sep + 1).join(' ') : '' }
}

export function optionReader(opts) {
  const used = new Set()
  const get = (name, d) => { const i = opts.indexOf(name); if (i < 0) return d; used.add(i); used.add(i + 1); return opts[i + 1] }
  const flag = name => { const i = opts.indexOf(name); if (i < 0) return false; used.add(i); return true }
  const all = name => opts.flatMap((o, i) => (o === name ? (used.add(i), used.add(i + 1), [opts[i + 1]]) : []))
  const unknown = () => opts.filter((_, i) => !used.has(i))
  return { get, flag, all, unknown }
}

export const intOpt = (v, name, lo, hi) => {
  const n = Number(v)
  if (!Number.isInteger(n) || n < lo || n > hi) throw new Error(`${name} must be an integer in ${lo}..${hi}`)
  return n
}
