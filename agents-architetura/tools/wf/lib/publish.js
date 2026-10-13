// Publishing (the reference PR's git model, made automatic): one factory/<task> branch and one draft PR per wan-task,
// one commit per finished stage whose message agents-architetura/tools/factory/publish.mjs derives from the stage result (red or green),
// a checkpoint timer while the run is long, and the PR leaves draft only when a milestone closes VERIFIED. Agents never
// run git themselves; merging is human-only. args.publish === false turns it off.
// Requires hygiene (clean, token), ledger (base64), project (common, ROLE) and campaign (PASTED).
const PUBLISHER = 'node agents-architetura/tools/factory/publish.mjs'
const publishing = () => args?.publish !== false
const PUBLISH_KEYS = ['verdict', 'ok', 'stoppedAt', 'why', 'failing', 'milestone', 'tree', 'converged', 'rounds', 'applicable', 'valid', 'unsuppressed', 'cpuHours', 'open', 'awaitingHuman', 'awaitingApproval', 'reason', 'report', 'gates', 'stabilizeRounds', 'pendingMarkers']
// The stage result, compacted: only the fields a reader needs, arrays and strings capped.
const publishSummary = r => {
  if (!r || typeof r !== 'object') return { ok: false, reason: 'the stage produced no result' }
  const out = {}
  for (const k of PUBLISH_KEYS) {
    if (!(k in r)) continue
    const v = r[k]
    out[k] = Array.isArray(v) ? v.slice(0, 12).map(x => (typeof x === 'string' ? clean(x, 160) : x)) : typeof v === 'string' ? clean(v, 400) : v !== null && typeof v === 'object' ? clean(JSON.stringify(v), 300) : v
  }
  return out
}
const runPublisher = (cmd, label, phaseTitle) => agent(common('You run ALONE — no other agent is active. You MAY run exactly this one command: it commits and pushes through the factory\'s guarded publisher. Never run git or gh yourself.') + `
Run exactly: ${cmd}
Return the JSON object it prints as "json", EXACTLY as printed, and its exit code as "exitCode". Do not rerun it, and do not
try to fix a refusal — report it.`,
  { label, phase: phaseTitle, schema: PASTED, agentType: ROLE.verifier })

// Exit 3 (refused: a secret, key material, an oversized file, a wrong identity) stops the run: something unsafe was
// written into the tree and a human must look. Any other failure is logged and retried by the next stage's publish.
const publishOutcome = (what, r) => {
  const j = r?.json ?? {}
  if (r?.exitCode === 3) throw new Error(`publish refused at ${what}: ${clean(j.reason ?? 'unsafe change', 500)} — nothing was pushed; a human must look`)
  if (r?.exitCode === 4) { log(`publish: ${what} not published — ${clean(j.reason ?? 'not on a factory/* branch', 200)}`); return { ok: false, skipped: true } }
  if (r?.exitCode !== 0) { log(`publish: ${what} NOT published (${clean(j.reason ?? `exit ${r?.exitCode ?? 'none'}`, 300)}) — the next publish retries`); return { ok: false } }
  return { ok: true, ...j }
}

const publishStage = async (stage, result, phaseTitle) => {
  if (!publishing()) return { ok: false, skipped: true }
  const s = token(stage, 60)
  const r = await runPublisher(`${PUBLISHER} commit --stage ${s} --b64 ${base64(JSON.stringify(publishSummary(result)))}`, `publish:${s}`, phaseTitle)
  const o = publishOutcome(s, r)
  if (o.ok) log(`publish: ${s} — ${o.committed ? `committed ${clean(o.sha ?? '', 12)}` : 'nothing new'}${o.pushed ? ', pushed' : ''}${o.pr?.url ? ` (${clean(o.pr.url, 200)})` : ''}`)
  return o
}
const publishCall = async (cmdArgs, what, phaseTitle) => publishing() ? publishOutcome(what, await runPublisher(`${PUBLISHER} ${cmdArgs}`, `publish:${what}`, phaseTitle)) : { ok: false, skipped: true }
