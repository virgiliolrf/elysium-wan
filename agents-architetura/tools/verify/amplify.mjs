#!/usr/bin/env node
// Race amplifier driver. Runs a command M times, each under a distinct seed, and classifies every run:
//   CRASH       killed by a signal
//   HANG        exceeded --timeout (lost wakeup / deadlock smell)
//   FAIL        nonzero exit
//   DIVERGENCE  exit 0 but output differs from the reference (the first clean run), after normalization
// The product's perturbation hook reads FACTORY_SEED / FACTORY_PERTURB_PERIOD / FACTORY_PERTURB_MAX_US (see
// agents-architetura/docs/VERIFICATION.md), so every finding is replayable from its seed. Tests must print deterministic
// output for a correct implementation, or divergence detection degrades to crash/hang/fail detection.
//
//   node agents-architetura/tools/verify/amplify.mjs --runs M [--seed-base S] [--periods 16,64,256] [--max-sleep-us U]
//        [--timeout SECS] [--ignore REGEX]... [--keep-logs] [--out DIR] [--json] -- <command...>
// Exit: 0 all runs clean · 1 any finding · 2 usage error.
//
// Before amplifying, one clean run (perturbation off) calibrates the timeout: a command that hangs, fails or takes more
// than half of --timeout unperturbed is misconfigured, and amplify stops at once (exit 2) instead of reporting every run
// as a HANG. M2's amplify-chaos ran tests/chaos (88 s clean) under a 60 s timeout: 262 false hangs in 4 h 20 min of a
// heavy gate, holding the whole machine (2026-10-10).
import { mkdirSync, writeFileSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { randomInt } from 'node:crypto'
import { runSync, normalizeOutput, splitArgv, optionReader, intOpt, SEED_ENV, PERIOD_ENV, MAX_US_ENV } from './exec.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')

export const replayLine = (cmd, seed, period, maxUs) => `${SEED_ENV}=${seed} ${PERIOD_ENV}=${period} ${MAX_US_ENV}=${maxUs} ${cmd}`

export function amplify(cmd, { runs = 100, seedBase = null, periods = [64], maxSleepUs = 100, timeoutSec = 60, ignore = [], cwd = ROOT, out = null, keepLogs = false, run = runSync } = {}) {
  if (!cmd) throw new Error('no command given (put it after --)')
  const base = seedBase ?? randomInt(1, 2 ** 31 - runs * periods.length - 1)
  const clean = run(cmd, { cwd, env: { [SEED_ENV]: String(base), [PERIOD_ENV]: String(periods[0]), [MAX_US_ENV]: '0' }, timeoutSec })
  if (clean.timedOut) throw new Error(`the clean run (perturbation off) exceeded --timeout ${timeoutSec}s: every perturbed run would look like a HANG; set --timeout to at least 3x the clean run's duration`)
  if (clean.outcome !== 'pass') throw new Error(`the clean run (perturbation off) does not pass (${clean.outcome}, exit ${clean.status}): amplification needs a passing baseline. Tail: ${`${clean.stdout}${clean.stderr}`.slice(-400)}`)
  if (clean.ms * 2 > timeoutSec * 1000) throw new Error(`the clean run took ${(clean.ms / 1000).toFixed(1)}s, more than half of --timeout ${timeoutSec}s: perturbation slows runs down, so a correct run could be reported as a HANG; set --timeout to at least 3x the clean run's duration`)
  const findings = []
  const counts = { pass: 0, fail: 0, crash: 0, hang: 0, divergence: 0 }
  let reference = null
  let i = 0
  for (const period of periods) {
    for (let k = 0; k < runs; k++) {
      i++
      const seed = base + i
      const env = { [SEED_ENV]: String(seed), [PERIOD_ENV]: String(period), [MAX_US_ENV]: String(maxSleepUs) }
      const r = run(cmd, { cwd, env, timeoutSec })
      const text = `${r.stdout}${r.stderr}`
      let outcome = r.outcome
      if (outcome === 'pass') {
        const norm = normalizeOutput(text, ignore)
        if (reference === null) reference = { seed, norm }
        else if (norm !== reference.norm) outcome = 'divergence'
      }
      counts[outcome]++
      const finding = outcome !== 'pass'
      let log = null
      if (out && (finding || keepLogs)) {
        mkdirSync(out, { recursive: true })
        log = join(out, `seed-${seed}.log`)
        writeFileSync(log, `$ ${replayLine(cmd, seed, period, maxSleepUs)}\n# outcome: ${outcome} (exit ${r.status}${r.signal ? `, signal ${r.signal}` : ''}${r.timedOut ? ', timed out' : ''})\n${text}`)
      }
      if (finding) findings.push({ seed, period, outcome: outcome.toUpperCase(), exit: r.status, signal: r.signal ?? null, log, replay: replayLine(cmd, seed, period, maxSleepUs), tail: text.slice(-1500) })
    }
  }
  return { cmd, seedBase: base, periods, runsPerPeriod: runs, total: i, cleanRunMs: clean.ms, counts, referenceSeed: reference?.seed ?? null, clean: findings.length === 0, findings }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { opts, cmd } = splitArgv(process.argv.slice(2))
  const o = optionReader(opts)
  let summary
  try {
    const runs = intOpt(o.get('--runs', '100'), '--runs', 1, 100000)
    const seedBaseRaw = o.get('--seed-base')
    const periods = o.get('--periods', '64').split(',').map(p => intOpt(p, '--periods', 1, 1 << 20))
    const maxSleepUs = intOpt(o.get('--max-sleep-us', '100'), '--max-sleep-us', 0, 10_000_000)
    const timeoutSec = intOpt(o.get('--timeout', '60'), '--timeout', 1, 86400)
    const ignore = o.all('--ignore')
    const keepLogs = o.flag('--keep-logs')
    const json = o.flag('--json')
    const out = resolve(o.get('--out', join(ROOT, '.factory', 'amplify', `${new Date().toISOString().replace(/[:.]/g, '-')}`)))
    if (o.unknown().length) throw new Error(`unknown option(s): ${o.unknown().join(' ')}`)
    summary = amplify(cmd, { runs, seedBase: seedBaseRaw === undefined ? null : intOpt(seedBaseRaw, '--seed-base', 0, 2 ** 31), periods, maxSleepUs, timeoutSec, ignore, out, keepLogs })
    mkdirSync(out, { recursive: true })
    writeFileSync(join(out, 'summary.json'), JSON.stringify(summary, null, 2))
    summary.summaryPath = join(out, 'summary.json')
    if (json) console.log(JSON.stringify(summary))
    else {
      console.log(`amplify: ${summary.total} runs, seeds ${summary.seedBase + 1}..${summary.seedBase + summary.total}, periods ${periods.join(',')}`)
      console.log(`amplify: ${JSON.stringify(summary.counts)}`)
      for (const f of summary.findings) console.log(`${f.outcome} seed=${f.seed} log=${f.log}\n  replay: ${f.replay}`)
      console.log(summary.clean ? 'AMPLIFY CLEAN' : `AMPLIFY FOUND ${summary.findings.length}`)
    }
  } catch (e) {
    console.error(`amplify: ${e.message}`)
    process.exit(2)
  }
  process.exit(summary.clean ? 0 : 1)
}
