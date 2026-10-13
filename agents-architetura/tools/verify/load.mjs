#!/usr/bin/env node
// Load harness: runs one command repeatedly across W concurrent workers with rotating seeds, under the load the
// workers generate themselves — the condition in which timing-dependent bugs surface (the PR's load6.sh and its
// "50/50 standalone + 10/10 under load" closure bars). A run fails on a nonzero exit, a signal, a timeout, or —
// with --require-pass-marker — output without the PASS marker. Failing logs are kept, passing logs are not.
//
//   node agents-architetura/tools/verify/load.mjs --workers W --runs-per-worker R [--seed-base S] [--timeout SECS]
//        [--require-pass-marker] [--out DIR] [--json] -- <command...>
// Exit: number of failed runs (capped at 125; 0 = clean) · 126 usage error.
import { mkdirSync, writeFileSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runAsync, splitArgv, optionReader, intOpt, SEED_ENV } from './exec.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
export const PASS_MARKER = /\bPASS\b/

export async function load(cmd, { workers = 6, runsPerWorker = 20, seedBase = 1000, timeoutSec = 120, requirePassMarker = false, cwd = ROOT, out = null, run = runAsync } = {}) {
  if (!cmd) throw new Error('no command given (put it after --)')
  const failures = []
  const spans = []
  const worker = async w => {
    for (let j = 0; j < runsPerWorker; j++) {
      const seed = seedBase + w * runsPerWorker + j
      const r = await run(cmd, { cwd, env: { [SEED_ENV]: String(seed) }, timeoutSec })
      spans.push([r.start ?? 0, r.end ?? 0])
      const text = `${r.stdout}${r.stderr}`
      const outcome = r.outcome === 'pass' && requirePassMarker && !PASS_MARKER.test(text) ? 'no-pass-marker' : r.outcome
      if (outcome !== 'pass') {
        let log = null
        if (out) {
          mkdirSync(out, { recursive: true })
          log = join(out, `w${w}-seed-${seed}.log`)
          writeFileSync(log, `$ ${SEED_ENV}=${seed} ${cmd}\n# outcome: ${outcome} (exit ${r.status})\n${text}`)
        }
        failures.push({ worker: w, seed, outcome, exit: r.status, log, tail: text.slice(-1500) })
      }
    }
  }
  await Promise.all(Array.from({ length: workers }, (_, w) => worker(w)))
  // Peak concurrency actually achieved (a "load" run that ran serially proves nothing).
  const events = spans.flatMap(([s, e]) => [[s, 1], [e, -1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1])
  let cur = 0
  let peak = 0
  for (const [, d] of events) { cur += d; peak = Math.max(peak, cur) }
  return { cmd, workers, runsPerWorker, total: workers * runsPerWorker, failed: failures.length, peakConcurrency: peak, clean: failures.length === 0, failures: failures.sort((a, b) => a.seed - b.seed) }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { opts, cmd } = splitArgv(process.argv.slice(2))
  const o = optionReader(opts)
  let summary
  try {
    const workers = intOpt(o.get('--workers', '6'), '--workers', 1, 256)
    const runsPerWorker = intOpt(o.get('--runs-per-worker', '20'), '--runs-per-worker', 1, 100000)
    const seedBase = intOpt(o.get('--seed-base', '1000'), '--seed-base', 0, 2 ** 31)
    const timeoutSec = intOpt(o.get('--timeout', '120'), '--timeout', 1, 86400)
    const requirePassMarker = o.flag('--require-pass-marker')
    const json = o.flag('--json')
    const out = resolve(o.get('--out', join(ROOT, '.factory', 'load', new Date().toISOString().replace(/[:.]/g, '-'))))
    if (o.unknown().length) throw new Error(`unknown option(s): ${o.unknown().join(' ')}`)
    summary = await load(cmd, { workers, runsPerWorker, seedBase, timeoutSec, requirePassMarker, out })
    mkdirSync(out, { recursive: true })
    writeFileSync(join(out, 'summary.json'), JSON.stringify(summary, null, 2))
    if (json) console.log(JSON.stringify(summary))
    else {
      console.log(`load: ${summary.total - summary.failed}/${summary.total} passed (${workers} workers, peak concurrency ${summary.peakConcurrency})`)
      for (const f of summary.failures) console.log(`${f.outcome.toUpperCase()} worker=${f.worker} seed=${f.seed} log=${f.log}`)
    }
  } catch (e) {
    console.error(`load: ${e.message}`)
    process.exit(126)
  }
  process.exit(Math.min(summary.failed, 125))
}
