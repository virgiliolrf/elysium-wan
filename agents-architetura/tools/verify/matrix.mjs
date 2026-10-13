#!/usr/bin/env node
// Configuration matrix: runs one command under every named mode (env + args), the PR's per-tier and
// GC-stress-matrix runs generalized. Each mode may declare a probe:
//   - probe output matching the mode's "unsupported" pattern  -> the mode is SKIPPED (the build lacks it)
//   - any other probe failure (crash, timeout, other exit)    -> FAIL: an engine crash never masquerades as
//                                                                "unsupported"
// If more than maxSkipPct of the modes are skipped the run is vacuous and FAILS.
//
// agents-architetura/tools/verify/matrix.json:
//   {"cmd": "uv run pytest -q tests/integration {args}", "maxSkipPct": 50, "timeoutSec": 3600,
//    "modes": [{"id": "lossy", "env": {"FACTORY_NET_LOSS": "0.05"}, "args": "", "probe": "...", "unsupported": "unsupported mode"}]}
//
//   node agents-architetura/tools/verify/matrix.mjs [--config PATH] [--only id,id] [--json]
// Exit: 0 pass · 1 fail · 2 usage · 5 not applicable (no config / no modes).
import { existsSync, readFileSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runSync, optionReader } from './exec.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const ID_RE = /^[a-z0-9][a-z0-9-]{0,40}$/

export function validateMatrix(cfg) {
  if (typeof cfg?.cmd !== 'string' || !cfg.cmd) throw new Error('matrix: "cmd" is required')
  const ids = new Set()
  for (const m of cfg.modes ?? []) {
    if (!ID_RE.test(m.id ?? '')) throw new Error(`matrix: bad mode id ${JSON.stringify(m.id)}`)
    if (ids.has(m.id)) throw new Error(`matrix: duplicate mode ${m.id}`)
    ids.add(m.id)
    if (m.env && (typeof m.env !== 'object' || Object.entries(m.env).some(([k, v]) => !/^[A-Z_][A-Z0-9_]*$/.test(k) || typeof v !== 'string'))) throw new Error(`matrix: mode ${m.id} env must map UPPER_CASE names to strings`)
    if (m.probe !== undefined && typeof m.probe !== 'string') throw new Error(`matrix: mode ${m.id} probe must be a string`)
  }
  return cfg
}

export function runMatrix(cfg, { cwd = ROOT, only = null, run = runSync } = {}) {
  validateMatrix(cfg)
  const modes = (cfg.modes ?? []).filter(m => !only || only.includes(m.id))
  if (!modes.length) return { applicable: false, reason: 'no modes configured', modes: [] }
  const results = []
  for (const m of modes) {
    const env = m.env ?? {}
    if (m.probe) {
      const p = run(m.probe, { cwd, env, timeoutSec: m.probeTimeoutSec ?? 120 })
      if (p.outcome !== 'pass') {
        const text = `${p.stdout}${p.stderr}`
        const unsupported = p.outcome === 'fail' && m.unsupported && new RegExp(m.unsupported).test(text)
        results.push({ mode: m.id, status: unsupported ? 'skipped' : 'fail', detail: unsupported ? `unsupported: ${text.trim().slice(-300)}` : `probe ${p.outcome} (exit ${p.status}${p.signal ? `, ${p.signal}` : ''}): ${text.slice(-1500)}` })
        continue
      }
    }
    const cmd = cfg.cmd.replace('{args}', m.args ?? '')
    const r = run(cmd, { cwd, env, timeoutSec: m.timeoutSec ?? cfg.timeoutSec ?? 3600 })
    results.push({ mode: m.id, status: r.outcome === 'pass' ? 'pass' : 'fail', detail: r.outcome === 'pass' ? '' : `${r.outcome} (exit ${r.status}${r.signal ? `, ${r.signal}` : ''}): ${`${r.stdout}${r.stderr}`.slice(-2000)}` })
  }
  const skipped = results.filter(r => r.status === 'skipped').length
  const skipPct = (100 * skipped) / results.length
  const vacuous = skipPct > (cfg.maxSkipPct ?? 50)
  const ok = !vacuous && results.every(r => r.status !== 'fail')
  return { applicable: true, ok, vacuous, skipPct: Math.round(skipPct), modes: results }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const o = optionReader(process.argv.slice(2))
  let r
  try {
    const path = resolve(o.get('--config', join(ROOT, 'agents-architetura', 'tools', 'verify', 'matrix.json')))
    const only = o.get('--only')?.split(',').filter(Boolean) ?? null
    const json = o.flag('--json')
    if (o.unknown().length) throw new Error(`unknown option(s): ${o.unknown().join(' ')}`)
    if (!existsSync(path)) { console.log(json ? JSON.stringify({ applicable: false, reason: 'no matrix config' }) : 'matrix: n/a (no config)'); process.exit(5) }
    r = runMatrix(JSON.parse(readFileSync(path, 'utf8')), { only })
    if (json) console.log(JSON.stringify(r))
    else if (!r.applicable) console.log(`matrix: n/a (${r.reason})`)
    else {
      for (const m of r.modes) console.log(`${m.status.toUpperCase().padEnd(8)} ${m.mode}${m.detail ? `  ${m.detail.split('\n')[0]}` : ''}`)
      console.log(r.vacuous ? `MATRIX VACUOUS (${r.skipPct}% skipped)` : r.ok ? 'MATRIX GREEN' : 'MATRIX RED')
    }
  } catch (e) {
    console.error(`matrix: ${e.message}`)
    process.exit(2)
  }
  process.exit(!r.applicable ? 5 : r.ok ? 0 : 1)
}
