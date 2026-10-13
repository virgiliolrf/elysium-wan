// Ladder helpers. Requires project (RUNG_IDS).
const knownRungs = rungs => (Array.isArray(rungs) ? rungs : []).filter(r => RUNG_IDS.includes(r))

// reuse: accept the last report when every input a product rung reads is unchanged (ladder.mjs --reuse). Only for reading
// the state (wan-next); gates that decide a verdict always run.
// failFast: stop at the first failing rung (a fix round's fast ladder; the heavy gate is closeout's).
const ladderCommand = ({ only = [], require = [], upto = null, reuse = false, failFast = false } = {}) => {
  const parts = ['node agents-architetura/tools/gates/ladder.mjs --json']
  const o = knownRungs(only)
  const r = knownRungs(require)
  if (o.length) parts.push(`--only ${o.join(',')}`)
  if (r.length) parts.push(`--require ${r.join(',')}`)
  if (upto && RUNG_IDS.includes(upto)) parts.push(`--upto ${upto}`)
  if (reuse) parts.push('--reuse')
  if (failFast) parts.push('--fail-fast')
  return parts.join(' ')
}

// Green only when the agent's claim AND the pasted ladder JSON agree.
const ladderGreen = report => report?.allGreen === true && report?.ladder?.allGreen === true

const failingRungs = ladder => knownRungs((ladder?.rungs ?? []).filter(r => r?.status === 'fail').map(r => r.rung))

// The pinned gate runs for hours; an agent's shell call returns within 10 minutes. M2 phase 2's verifier started
// pinned.mjs in the background and ended before it finished, and the verdict was lost. pinned.mjs --wait runs the gate once
// in a detached process keyed by (id, label, tree, gates file) and returns within PINNED_WAIT_SEC with its verdict or
// {"status":"running"}. The script owns the wait, one short agent per slice: an agent that ends early loses nothing, and a
// resumed workflow finds the run, or its stored verdict, instead of starting the gate again.
const PINNED_WAIT_SEC = 540
const PINNED_MAX_SLICES = 80
const pinnedCommand = (id, label) => `node agents-architetura/tools/gates/pinned.mjs ${id} --json --label ${label} --wait ${PINNED_WAIT_SEC}`
const PINNED_SLICE = {
  type: 'object',
  required: ['status'],
  properties: {
    status: { type: 'string', enum: ['finished', 'running'], description: 'finished: it printed a verdict or a failure; running: it printed {"status":"running",...} or the call timed out' },
    detail: { type: 'string', description: 'the first 300 characters it printed' },
  },
}
// true once the gate has finished (its verdict, or its failure, is stored and prints at once); false after the last slice
const waitPinned = async (id, label, phaseTitle) => {
  for (let s = 0; s < PINNED_MAX_SLICES; s++) {
    const w = await agent(`Repo: the repository root (your current working directory). You run ALONE. Change nothing, no git.
Run exactly this command in the FOREGROUND, with the Bash tool's timeout set to 600000 ms (never in the background):
${pinnedCommand(id, label)}
It starts the pinned gate once in a detached process (or finds that run, or its stored verdict) and returns within ${PINNED_WAIT_SEC / 60}
minutes. Return status "running" if it printed {"status":"running",...} or the call timed out, otherwise "finished". Run it once.`,
      { label: `verify-wait:${label}:s${s}`, phase: phaseTitle, schema: PINNED_SLICE, agentType: ROLE.verifier })
    if (w?.status === 'finished') return true
  }
  return false
}
const pinnedStillRunning = (id, label) => `the pinned gate for ${id} did not finish within ${PINNED_MAX_SLICES * PINNED_WAIT_SEC / 3600} hours; it keeps running detached (.factory/gates/runs/${id}.${label}.log) — rerun this workflow when it ends: its verify picks the stored verdict up`
