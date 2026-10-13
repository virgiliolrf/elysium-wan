#!/usr/bin/env node
// Causal closure (the reference PR's controlled experiment: "re-enabling the pre-fix behavior reproduces the corruption on
// demand (4/15) while the fixed path shows zero (0/15)"). A clean run after a fix can be a lucky window; this proves the
// fix is the cause. Before the fix is applied, `save` copies the files it will touch. After, `ab` runs the repro
//   A) on a snapshot of the working tree with those files restored to their pre-fix content — the bug must come back;
//   B) on the live, fixed tree — it must not fail once.
// The script decides: causal = A.failures > 0 && B.failures == 0. Neither arm touches the live tree's files.
//
//   node agents-architetura/tools/verify/causal.mjs save --out DIR -- <file...>
//   node agents-architetura/tools/verify/causal.mjs ab --prefix DIR --runs N [--timeout SECS] [--json] -- <repro command...>
// Exit: 0 saved / causal · 1 not causal · 2 usage.
import { existsSync, mkdirSync, copyFileSync, writeFileSync, readFileSync, rmSync } from 'node:fs'
import { join, resolve, dirname, isAbsolute, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import { materialize } from '../gates/snapshot-dir.mjs'
import { runSync, splitArgv, intOpt, SEED_ENV } from './exec.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const MANIFEST = 'manifest.json'
const safeRel = f => typeof f === 'string' && !!f && !isAbsolute(f) && !normalize(f).split(/[\\/]/).includes('..') && !normalize(f).startsWith('.git')

export function save(root, out, files) {
  const bad = files.filter(f => !safeRel(f))
  if (!files.length || bad.length) throw new Error(`save needs repo-relative files without "..": ${JSON.stringify(bad.length ? bad : files)}`)
  mkdirSync(out, { recursive: true })
  const manifest = { files: [], absent: [] }
  for (const f of files) {
    const src = join(root, f)
    if (!existsSync(src)) { manifest.absent.push(f); continue }
    mkdirSync(dirname(join(out, 'files', f)), { recursive: true })
    copyFileSync(src, join(out, 'files', f))
    manifest.files.push(f)
  }
  writeFileSync(join(out, MANIFEST), JSON.stringify(manifest, null, 2))
  return manifest
}

const arm = (cmd, { cwd, env, runs, timeoutSec, run }) => {
  const failures = []
  for (let i = 0; i < runs; i++) {
    const r = run(cmd, { cwd, env: { ...env, [SEED_ENV]: String(1 + i) }, timeoutSec })
    if (r.outcome !== 'pass') failures.push({ run: i, outcome: r.outcome, status: r.status, tail: `${r.stdout ?? ''}${r.stderr ?? ''}`.slice(-300) })
  }
  return { runs, failures: failures.length, examples: failures.slice(0, 3) }
}

export function ab(root, prefix, cmd, { runs, timeoutSec = 600, run = runSync, snapshot = materialize } = {}) {
  const manifest = JSON.parse(readFileSync(join(prefix, MANIFEST), 'utf8'))
  const snap = snapshot(root, 'causal-')
  try {
    for (const f of manifest.files) {
      mkdirSync(dirname(join(snap.dir, f)), { recursive: true })
      copyFileSync(join(prefix, 'files', f), join(snap.dir, f))
    }
    for (const f of manifest.absent) rmSync(join(snap.dir, f), { force: true, recursive: true })
    const reverted = arm(cmd, { cwd: snap.dir, env: snap.env ?? {}, runs, timeoutSec, run })
    const fixed = arm(cmd, { cwd: root, env: {}, runs, timeoutSec, run })
    const causal = reverted.failures > 0 && fixed.failures === 0
    return {
      causal,
      reverted,
      fixed,
      restored: manifest.files,
      removed: manifest.absent,
      why: causal ? `the pre-fix files bring the failure back (${reverted.failures}/${runs}); the fixed tree is clean (0/${runs})`
        : reverted.failures === 0 ? `restoring the pre-fix files did not bring the failure back in ${runs} runs — the fix is not shown to be the cause`
          : `the fixed tree still fails (${fixed.failures}/${runs})`,
    }
  } finally {
    rmSync(snap.dir, { recursive: true, force: true })
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { opts, cmd } = splitArgv(process.argv.slice(2))
  const sub = opts[0]
  const val = name => { const i = opts.indexOf(name); return i >= 0 ? opts[i + 1] : undefined }
  try {
    if (sub === 'save') {
      const out = val('--out')
      if (!out || !safeRel(out)) throw new Error('--out <repo-relative dir> is required')
      const m = save(ROOT, join(ROOT, out), cmd.split(/\s+/).filter(Boolean))
      console.log(JSON.stringify({ ok: true, ...m }))
      process.exit(0)
    }
    if (sub === 'ab') {
      const prefix = val('--prefix')
      if (!prefix || !safeRel(prefix) || !existsSync(join(ROOT, prefix, MANIFEST))) throw new Error('--prefix <dir written by save> is required')
      if (!cmd.trim()) throw new Error('the repro command goes after --')
      const runs = intOpt(val('--runs'), '--runs', 1, 100000)
      const timeoutSec = val('--timeout') ? intOpt(val('--timeout'), '--timeout', 1, 86400) : 600
      const r = ab(ROOT, join(ROOT, prefix), cmd, { runs, timeoutSec })
      console.log(opts.includes('--json') ? JSON.stringify(r) : `${r.causal ? 'CAUSAL' : 'NOT CAUSAL'}: ${r.why}`)
      process.exit(r.causal ? 0 : 1)
    }
    throw new Error('usage: causal.mjs save --out DIR -- <file...> | ab --prefix DIR --runs N [--timeout S] [--json] -- <command...>')
  } catch (e) {
    console.error(`causal: ${e.message}`)
    process.exit(2)
  }
}
