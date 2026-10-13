#!/usr/bin/env node
// Scalability gate (the PR's scaling-gate.sh). bench/scaling.json:
//   {"maxLoadPerCore": 0.25,
//    "workloads": [{"id": "allreduce-weak", "cmd": "... --workers {N}", "threads": [1, 2, 4, 8], "runs": 5,
//                   "mode": "weak", "floors": {"4": 2.8, "8": 4.5}}],
//    "serial": {"on": "<cmd, features on>", "off": "<cmd, features off>", "runs": 5, "maxTaxPct": 2}}
// Each cmd prints {"value": <seconds>} as its last stdout line. speedup(N) is N*T(1)/T(N) in weak mode (every
// worker does the same fixed work) and T(1)/T(N) in strong mode (fixed total work). The serial check bounds what
// the feature costs a single worker. REPORT-ONLY unless --gate: numbers from a shared, noisy machine are for trend
// tracking; --gate enforces the floors and refuses (exit 3) on a busy machine.
//
//   node agents-architetura/tools/gates/scaling-gate.mjs [--gate] [--config PATH] [--json]
// Exit: 0 ok or report-only · 1 a floor or the tax bound missed (--gate) · 2 usage · 3 noisy · 5 n/a (no config).
import { existsSync, readFileSync } from 'node:fs'
import { join, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defaultRun } from './ladder.mjs'
import { median, assertQuiet, loadPerCore, NoisyMachine } from './bench-gate.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')

function timeOf(cmd, runs, run, cwd) {
  const xs = []
  for (let i = 0; i < runs; i++) {
    const r = run(cmd, { cwd, timeoutSec: 3600 })
    if (r.code !== 0) throw new Error(`scaling: exit ${r.code} for ${cmd}\n${r.output.slice(-1500)}`)
    const v = JSON.parse(r.output.trim().split('\n').pop()).value
    if (typeof v !== 'number' || !(v > 0)) throw new Error(`scaling: last line is not {"value": seconds>0} for ${cmd}`)
    xs.push(v)
  }
  return median(xs)
}

export function runScaling(cfg, { gate = false, run = defaultRun, cwd = ROOT, load = loadPerCore } = {}) {
  if (gate) assertQuiet(cfg.maxLoadPerCore ?? 0.25, load)
  const workloads = (cfg.workloads ?? []).map(w => {
    const threads = w.threads ?? [1, 2, 4, 8]
    if (threads[0] !== 1) throw new Error(`scaling: workload ${w.id} must start at 1 thread`)
    const t = Object.fromEntries(threads.map(n => [n, timeOf(w.cmd.replaceAll('{N}', String(n)), w.runs ?? 5, run, cwd)]))
    const speedup = Object.fromEntries(threads.map(n => [n, Math.round(((w.mode === 'strong' ? 1 : n) * t[1] / t[n]) * 100) / 100]))
    const misses = Object.entries(w.floors ?? {}).filter(([n, floor]) => speedup[n] === undefined || speedup[n] < floor).map(([n, floor]) => `${w.id}: speedup(${n}) = ${speedup[n] ?? 'not measured'} < ${floor}`)
    return { id: w.id, mode: w.mode ?? 'weak', seconds: t, speedup, misses }
  })
  let serial = null
  if (cfg.serial) {
    const on = timeOf(cfg.serial.on, cfg.serial.runs ?? 5, run, cwd)
    const off = timeOf(cfg.serial.off, cfg.serial.runs ?? 5, run, cwd)
    const taxPct = Math.round(((on - off) / off) * 10000) / 100
    serial = { on, off, taxPct, ok: taxPct <= (cfg.serial.maxTaxPct ?? 2) }
  }
  const misses = [...workloads.flatMap(w => w.misses), ...(serial && !serial.ok ? [`serial tax ${serial.taxPct}% > ${cfg.serial.maxTaxPct ?? 2}%`] : [])]
  return { gate, ok: !gate || !misses.length, misses, workloads, serial }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv.slice(2)
  const i = argv.indexOf('--config')
  const path = resolve(i >= 0 ? argv[i + 1] : join(ROOT, 'bench', 'scaling.json'))
  if (!existsSync(path)) { console.log('scaling: n/a (no bench/scaling.json)'); process.exit(5) }
  try {
    const r = runScaling(JSON.parse(readFileSync(path, 'utf8')), { gate: argv.includes('--gate') })
    if (argv.includes('--json')) console.log(JSON.stringify(r))
    else {
      for (const w of r.workloads) console.log(`${w.id} (${w.mode}): ${Object.entries(w.speedup).map(([n, s]) => `${n}:${s}x`).join(' ')}`)
      if (r.serial) console.log(`serial tax: ${r.serial.taxPct}%`)
      for (const m of r.misses) console.log(`${r.gate ? 'FAIL' : 'WARN'} ${m}`)
    }
    process.exit(r.ok ? 0 : 1)
  } catch (e) {
    console.error(`scaling: ${e.message}`)
    process.exit(e instanceof NoisyMachine ? 3 : 2)
  }
}
