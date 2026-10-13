// Scope audit around a fan-out of writers (requires schemas AUDIT and project). Returns the snapshot tree; check()
// throws on any write outside `scopes`.
const TREE_ID_RE = /^[0-9a-f]{40,64}$/
const scopeSnapshot = async (label, phaseTitle) => {
  const s = await agent(common('You run ALONE — no other agent is active; you MAY run commands.') + `
Run exactly: node agents-architetura/tools/wf/scope-audit.mjs snapshot — report the "tree" it prints verbatim. Do nothing else.`,
    { label: `audit:snapshot:${label}`, phase: phaseTitle, schema: AUDIT, agentType: ROLE.verifier })
  if (!TREE_ID_RE.test(s?.tree ?? '')) throw new Error(`scope-audit snapshot (${label}) returned no tree id`)
  return s.tree
}
const scopeCheck = async (label, phaseTitle, since, scopes) => {
  const a = await agent(common('You run ALONE — no other agent is active; you MAY run commands.') + `
Run exactly: node agents-architetura/tools/wf/scope-audit.mjs check --since ${since} --scopes '${JSON.stringify(scopes)}'
Exit 0 = no violations, 1 = violations listed. Return its JSON verbatim. Do nothing else.`,
    { label: `audit:check:${label}`, phase: phaseTitle, schema: AUDIT, agentType: ROLE.verifier })
  if (!a || !Array.isArray(a.violations)) throw new Error(`scope audit (${label}) produced no result`)
  if (a.violations.length) throw new Error(`${label} wrote outside its owned paths: ${a.violations.join(', ')}`)
  return a
}
