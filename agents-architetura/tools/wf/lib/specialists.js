// Specialists: domain expertise attached to tasks by the paths they touch — not new agent types. The repository
// survey creates a specialist only when the code has the responsibility (evidence: files that exist); a task whose
// scope overlaps a specialist's paths gets its focus in the implementer's prompt and its lens as an extra reviewer.
// Requires hygiene (clean, token), dag (scopesOverlap, safeScopePath), project (common, ROLE) and agent().
const SPECIALIST_CATALOG = {
  scheduler: 'GPU allocation, heterogeneous hardware, task placement, work stealing, stragglers, fairness, utilization',
  'distributed-training': 'training execution, synchronization, gradient/activation communication, checkpointing, recovery, heterogeneous GPU behavior',
  'wan-networking': 'latency, bandwidth, connection reliability, retries and backoff, transports, P2P, NAT/firewalls, network failures',
  'fault-tolerance': 'worker and coordinator failures, partitions, checkpoint recovery, partial progress, retry and idempotency semantics',
  performance: 'GPU utilization, communication overhead, throughput, step time, network utilization, scheduler overhead, scaling efficiency, cost per step and per million tokens',
  observability: 'metrics, structured logs, traces, distributed debugging, scheduler decisions, worker health, training progress',
  'training-correctness': 'loss, gradients, parameter updates, convergence, checkpoints, reproducibility, byte-exact identity with the synchronous datacenter reference (no numerical tolerance), reduction order, synchronization and asynchronous behavior, recovery correctness',
}
const SPECIALISTS = {
  type: 'object',
  required: ['specialists', 'rejected'],
  properties: {
    specialists: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'paths', 'evidence', 'focus', 'reviewLens'],
        properties: {
          id: { type: 'string', description: `one of ${Object.keys(SPECIALIST_CATALOG).join(', ')}, or custom-<name> with a justification` },
          paths: { type: 'array', items: { type: 'string' }, description: 'repo paths (directories end with "/") holding this responsibility' },
          evidence: { type: 'array', items: { type: 'string' }, description: 'file:line references proving the responsibility exists in the code' },
          focus: { type: 'string', description: 'what an implementer touching these paths must get right' },
          reviewLens: { type: 'string', description: 'what a reviewer of these paths must hunt' },
        },
      },
    },
    rejected: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, why: { type: 'string' } } } },
  },
}
const SPECIALIST_ID_RE = /^([a-z][a-z-]{1,30}|custom-[a-z][a-z0-9-]{1,30})$/

// Validates the survey's choice: known or custom ids, safe paths, evidence present. Path existence is checked by the
// caller (inventory --check-paths).
const validSpecialists = list => (list ?? []).filter(s =>
  SPECIALIST_ID_RE.test(s?.id ?? '') && (SPECIALIST_CATALOG[s.id] || s.id.startsWith('custom-')) &&
  (s.paths ?? []).length && s.paths.every(p => safeScopePath(p)) && (s.evidence ?? []).length && String(s.focus ?? '').trim() && String(s.reviewLens ?? '').trim())

const specialistsFor = (scope, specialists) => (specialists ?? []).filter(s => (scope ?? []).some(a => s.paths.some(b => scopesOverlap(a, b))))

const loadSpecialists = async phaseTitle => {
  const r = await agent(common('READ-ONLY.') + `
Read docs/design/ARCHITECTURE_MAP.md. Find the block after "<!-- specialists -->" and return its JSON array as "specialists"
verbatim (and "rejected": []). If the file or the block is missing, return both as empty arrays. Do nothing else.`,
    { label: 'extract:specialists', phase: phaseTitle, schema: SPECIALISTS, agentType: ROLE.readonly })
  return validSpecialists(r?.specialists).map(s => ({ ...s, id: token(s.id, 40) }))
}
