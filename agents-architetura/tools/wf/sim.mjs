// Offline simulator for workflow scripts and lib fragments. Mirrors the
// Claude Code workflow runtime contract: agent(), parallel() (barrier, failed
// thunks -> null), pipeline() (per-item stage chains, stage receives
// (prev, item, index), a throwing stage drops the item to null), phase(),
// log(), args, budget, workflow(). Lets every workflow's control flow be
// unit-tested with scripted agent responses — no model calls.
import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const LIB = join(dirname(fileURLToPath(import.meta.url)), 'lib')
const AsyncFunction = (async () => {}).constructor
const TOP_LEVEL_DECL = /^(?:const|let|var|async function|function|class)\s+([A-Za-z_$][\w$]*)/gm

export function makeRuntime({ respond = () => null, args, budgetTotal = null, workflows = {} } = {}) {
  const calls = []
  const logs = []
  const phases = []
  const runtime = {
    args,
    log: m => { logs.push(String(m)) },
    phase: t => { phases.push(t) },
    agent: async (prompt, opts = {}) => {
      const index = calls.length
      calls.push({ prompt, opts })
      return respond(prompt, opts, index)
    },
    parallel: thunks => Promise.all(thunks.map(t => Promise.resolve().then(t).catch(() => null))),
    pipeline: (items, ...stages) => Promise.all(items.map(async (item, i) => {
      try {
        let prev = item
        for (const stage of stages) prev = await stage(prev, item, i)
        return prev
      } catch {
        return null
      }
    })),
    budget: {
      total: budgetTotal,
      spent: () => 0,
      remaining: () => (budgetTotal == null ? Infinity : budgetTotal),
    },
    workflow: async (name, a) => {
      if (!workflows[name]) throw new Error(`no simulated workflow: ${name}`)
      return workflows[name](a)
    },
  }
  return { runtime, calls, logs, phases }
}

// Evaluates lib fragments (in order, one shared scope — exactly how a built
// workflow sees them) and returns their top-level declarations.
export function loadLib(names, globals = {}) {
  const code = names.map(n => readFileSync(join(LIB, `${n}.js`), 'utf8')).join('\n')
  const exported = [...code.matchAll(TOP_LEVEL_DECL)].map(m => m[1])
  const keys = Object.keys(globals)
  return new Function(...keys, `${code}\nreturn { ${exported.join(', ')} }`)(...keys.map(k => globals[k]))
}

export async function simulate(script, opts = {}) {
  const { runtime, calls, logs, phases } = makeRuntime(opts)
  const body = script.replace(/^export const meta =/, 'const meta =')
  const keys = Object.keys(runtime)
  const result = await new AsyncFunction(...keys, body)(...keys.map(k => runtime[k]))
  return { result, calls, logs, phases }
}

export const simulateFile = (path, opts) => simulate(readFileSync(path, 'utf8'), opts)

// Routes a call to the handler with the LONGEST label prefix matching opts.label (most specific wins).
// Handlers: { 'triage:': (prompt, opts, index, seen) => value, ... }; `seen`
// counts prior calls with the same prefix (0-based).
export function byLabel(handlers, fallback = () => { throw new Error('unrouted agent call') }) {
  const counts = new Map()
  return (prompt, opts, index) => {
    const label = opts.label ?? ''
    const prefix = Object.keys(handlers).filter(p => label.startsWith(p)).sort((a, b) => b.length - a.length)[0]
    if (!prefix) return fallback(prompt, opts, index)
    const seen = counts.get(prefix) ?? 0
    counts.set(prefix, seen + 1)
    return handlers[prefix](prompt, opts, index, seen)
  }
}
