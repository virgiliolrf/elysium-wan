// Verification-depth helpers shared by wan-campaign, wan-fuzz, wan-audit, wan-scanners, wan-corpus, wan-bughunt,
// wan-handout and wan-closeout. Pure functions (unit-tested via loadLib) plus the schemas for the JSON the
// tools print — agents paste that JSON verbatim; the script never trusts a summary of it.
// Requires hygiene (clean, token), dag (safeScopePath, scopesOverlap) and project (RUNG_IDS).

// Exact JSON printed by `node agents-architetura/tools/campaign/families.mjs --json`.
const ROUND = {
  type: 'object',
  description: 'the exact JSON printed by agents-architetura/tools/campaign/families.mjs --json, unmodified',
  required: ['applicable'],
  properties: {
    applicable: { type: 'boolean' },
    reason: { type: 'string' },
    campaign: { type: 'string' },
    round: { type: 'number' },
    tree: { description: 'tree id (null outside git)' },
    broken: { type: 'boolean' },
    brokenReason: { description: 'why the round is broken (null when it is not)' },
    unsuppressedReports: { type: 'number' },
    families: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'kind', 'count', 'signature'],
        properties: { id: { type: 'string' }, kind: { type: 'string' }, count: { type: 'number' }, signature: { type: 'string' }, locations: { type: 'array', items: { type: 'string' } }, message: { type: 'string' } },
      },
    },
    suppressed: { type: 'array', items: { type: 'object' } },
    suppressionViolations: { type: 'array', items: { type: 'string' } },
  },
}

const RULINGS = {
  type: 'object',
  required: ['rulings'],
  properties: {
    rulings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['familyId', 'ruling', 'fixShape', 'files'],
        properties: {
          familyId: { type: 'string' },
          ruling: { type: 'string', enum: ['real-bug', 'protocol-fix', 'benign-by-spec', 'third-party'] },
          specRef: { type: 'string', description: 'for benign-by-spec: docs/design/SPEC-<key>.md#<anchor> that blesses it' },
          fixShape: { type: 'string', description: '1-3 sentences: exactly what to change, or the suppression entry you wrote' },
          files: { type: 'array', items: { type: 'string' }, description: 'files the fix will WRITE (exhaustive and minimal) — wave partitioning depends on it' },
        },
      },
    },
  },
}

const CAMPAIGN_FIX = {
  type: 'object',
  required: ['summary', 'files'],
  properties: {
    summary: { type: 'string', description: 'for a real bug: the interleaving and the happens-before (or ordering) the fix establishes' },
    files: { type: 'array', items: { type: 'string' } },
    misruled: { type: 'string', description: 'set if reading the code showed the ruling is wrong (e.g. "benign" is a real bug): the evidence; then change nothing' },
  },
}

const PASTED = {
  type: 'object',
  required: ['json'],
  properties: { json: { type: 'object', description: 'the JSON the command printed, verbatim' }, exitCode: { type: 'number' } },
}

// Families to fix this wave: real bugs get solo-grade attention (at most maxReal per wave, never batched with a
// family that shares a file); mechanical families batch on disjoint files. Families without safe files cannot be
// fixed by a write-only agent and are returned as unplaceable.
const partitionWave = (families, rulings, { maxReal = 3, maxBatch = 12 } = {}) => {
  const byId = new Map(rulings.map(r => [r.familyId, r]))
  const fixable = families
    .map(f => ({ f, r: byId.get(f.id) }))
    .filter(x => x.r && (x.r.ruling === 'real-bug' || x.r.ruling === 'protocol-fix'))
    .map(x => ({ ...x, files: (x.r.files ?? []).filter(p => typeof p === 'string' && safeScopePath(p)) }))
  const claimed = []
  const overlaps = files => files.some(a => claimed.some(b => scopesOverlap(a, b)))
  const work = []
  const deferred = []
  const unplaceable = fixable.filter(x => !x.files.length).map(x => x.f.id)
  const real = fixable.filter(x => x.files.length && x.r.ruling === 'real-bug')
  const mech = fixable.filter(x => x.files.length && x.r.ruling !== 'real-bug')
  let realCount = 0
  for (const x of real) {
    if (realCount >= maxReal || overlaps(x.files)) { deferred.push(x.f.id); continue }
    claimed.push(...x.files)
    work.push({ family: x.f, ruling: x.r, files: x.files, solo: true })
    realCount++
  }
  let batch = 0
  for (const x of mech) {
    if (batch >= maxBatch || overlaps(x.files)) { deferred.push(x.f.id); continue }
    claimed.push(...x.files)
    work.push({ family: x.f, ruling: x.r, files: x.files, solo: false })
    batch++
  }
  const unruled = families.filter(f => !byId.has(f.id)).map(f => f.id)
  return { work, deferred, unplaceable, unruled }
}

// Progress across rounds: true when the last `window` rounds did not reduce the count.
const stalled = (counts, window = 2) => counts.length > window && counts.slice(-window - 1).every((c, i, a) => i === 0 || c >= a[i - 1])

// Statistical closure (rule of three): 0 failures in N runs bounds the failure rate below 3/N with 95%
// confidence. To claim a fix cut the observed rate p = k/n at least `factor`-fold, N >= factor * 3 / p.
const parseRate = s => {
  const m = String(s ?? '').match(/(\d+)\s*\/\s*(\d+)/)
  if (!m) return null
  const k = Number(m[1])
  const n = Number(m[2])
  return n > 0 && k > 0 && k <= n ? { k, n, p: k / n } : null
}
const closureRuns = (rate, { factor = 10, min = 50, cap = 5000 } = {}) => {
  const r = typeof rate === 'string' ? parseRate(rate) : rate
  const needed = r ? Math.max(min, Math.ceil((factor * 3) / r.p)) : min
  return { needed, cap, feasible: needed <= cap, runs: Math.min(needed, cap) }
}

// GATES-<m>.json written by wan-handout: shape check (ids, rungs, consecutive); existence of campaigns is
// checked by `node agents-architetura/tools/gates/pinned.mjs --validate`, which knows agents-architetura/tools/campaign/campaigns.json.
// `full` is on demand in everyday ladders (each suite already runs through its own rung); the verdict gate runs the whole
// suite once in one session, which is what catches interference between suites.
const BASE_GATE_RUNGS = ['factory', 'build', 'syntax', 'discipline', 'invariants', 'claims', 'full']
const GATES = {
  type: 'object',
  required: ['milestone', 'consecutive', 'rungs', 'pinned', 'campaigns', 'fuzz'],
  properties: {
    milestone: { type: 'string' },
    consecutive: { type: 'number' },
    rungs: { type: 'array', items: { type: 'string' } },
    pinned: { type: 'array', items: { type: 'object', required: ['id', 'cmd'], properties: { id: { type: 'string' }, cmd: { type: 'string' }, repeat: { type: 'number' } } } },
    campaigns: { type: 'array', items: { type: 'object', required: ['id'], properties: { id: { type: 'string' } } } },
    fuzz: { type: 'array', items: { type: 'object', required: ['id'], properties: { id: { type: 'string' }, minCpuHours: { type: 'number' } } } },
    audit: { type: 'object', properties: { bar: { type: 'string' } } },
    mutation: { type: 'object', properties: { changedSince: { type: 'string' } } },
  },
}
const checkGates = (gates, { milestone, planRungs = [] }) => {
  const errors = []
  if (!gates) return ['no gates file']
  if (gates.milestone !== milestone) errors.push(`milestone must be "${milestone}"`)
  if (!(Number.isInteger(gates.consecutive) && gates.consecutive >= 2 && gates.consecutive <= 10)) errors.push('consecutive must be 2..10 (stability is measured, not assumed)')
  const rungs = gates.rungs ?? []
  for (const r of rungs) if (!RUNG_IDS.includes(r)) errors.push(`unknown rung ${JSON.stringify(r)}`)
  for (const r of [...BASE_GATE_RUNGS, ...planRungs]) if (RUNG_IDS.includes(r) && !rungs.includes(r)) errors.push(`rung ${r} is required (base gate or named by a plan task)`)
  for (const p of gates.pinned ?? []) {
    if (!/^[a-z0-9][a-z0-9-]{0,40}$/.test(p.id ?? '')) errors.push(`pinned: bad id ${JSON.stringify(p.id)}`)
    if (typeof p.cmd !== 'string' || !p.cmd.trim()) errors.push(`pinned ${p.id}: empty cmd`)
  }
  for (const c of [...(gates.campaigns ?? []), ...(gates.fuzz ?? [])]) if (!/^[a-z0-9][a-z0-9-]{0,40}$/.test(c.id ?? '')) errors.push(`bad campaign id ${JSON.stringify(c.id)}`)
  return errors
}

const cleanRound = r => !!r && r.applicable === true && !r.broken && r.unsuppressedReports === 0 && !(r.suppressionViolations ?? []).length

// Machine records (requires ledger: base64). The script computes the payload; an agent only runs the command.
const recordCommand = (target, payload, id = '') => `node agents-architetura/tools/campaign/record.mjs ${target}${id ? ` ${id}` : ''} --b64 ${base64(JSON.stringify(payload))}`

// A verifier runs one factory tool and pastes its JSON verbatim (schema PASTED).
const runTool = (cmd, { label, phaseTitle, note = '' }) => agent(common('You run ALONE — you MAY run commands (no git). Change nothing.') + `
Run exactly: ${cmd}
Return the JSON object it prints as "json", EXACTLY as printed, and its exit code as "exitCode". If it printed no JSON
(e.g. "n/a"), return {} as json with the exit code. Do not rerun it for a better result. ${note}`,
  { label, phase: phaseTitle, schema: PASTED, agentType: ROLE.verifier })
