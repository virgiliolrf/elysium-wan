// Campaign suppressions. agents-architetura/tools/campaign/suppressions/<campaign>.txt starts EMPTY and may only hold justified
// entries — races or reports in code the factory owns are bugs to fix, never to silence:
//
//   # RULING: benign-by-spec docs/design/SPEC-transport.md#TRANSPORT-I4
//   # JUSTIFICATION: the stats counter is read racily by design; a stale read only delays a log line.
//   match: tsan-data-race|record@src/transport/stats.py
//
// Rulings: "benign-by-spec <SPEC file>#<anchor>" (the file must exist and contain the anchor text),
// "third-party <component>" (the match may not name a path in the repository's own source trees), or — for static
// analyzers only (scan-*.txt) — "false-positive <rule>" with a justification of at least 40 characters.
// A "match:" entry suppresses every family whose signature contains its text.
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

const OWN_SOURCE = /(^|[@|;\s])(src|crates|python|tests)\//

export function parseSuppressions(text, file = 'suppressions') {
  const entries = []
  const violations = []
  let ruling = null
  let justification = null
  String(text).split('\n').forEach((raw, i) => {
    const line = raw.trim()
    const where = `${file}:${i + 1}`
    if (!line) { ruling = null; justification = null; return }
    let m
    if ((m = line.match(/^#\s*RULING:\s*(.+)$/))) ruling = { text: m[1].trim(), where }
    else if ((m = line.match(/^#\s*JUSTIFICATION:\s*(.+)$/))) justification = m[1].trim()
    else if (line.startsWith('#')) { /* plain comment */ } else if ((m = line.match(/^match:\s*(.+)$/))) {
      const pattern = m[1].trim()
      if (!ruling) violations.push(`${where}: "match: ${pattern}" has no "# RULING:" line above it`)
      if (!justification || justification.length < 20) violations.push(`${where}: "match: ${pattern}" needs a "# JUSTIFICATION:" of at least 20 characters`)
      entries.push({ pattern, ruling: ruling?.text ?? null, justification, where })
    } else violations.push(`${where}: unrecognized line ${JSON.stringify(line.slice(0, 60))}`)
  })
  return { entries, violations }
}

export function auditEntries(root, entries, { file = '' } = {}) {
  const violations = []
  for (const e of entries) {
    if (!e.ruling) continue
    let m
    if ((m = e.ruling.match(/^benign-by-spec\s+(docs\/design\/SPEC-[\w-]+\.md)#(.+)$/))) {
      const p = join(root, m[1])
      if (!existsSync(p)) violations.push(`${e.where}: cited spec ${m[1]} does not exist`)
      else if (!readFileSync(p, 'utf8').includes(m[2].trim())) violations.push(`${e.where}: cited section "${m[2].trim()}" is not in ${m[1]}`)
    } else if ((m = e.ruling.match(/^third-party\s+(\S.*)$/))) {
      if (OWN_SOURCE.test(e.pattern)) violations.push(`${e.where}: "third-party" ruling on a pattern that names the repository's own source (${e.pattern})`)
    } else if ((m = e.ruling.match(/^false-positive\s+(\S.*)$/))) {
      if (!/(^|\/)scan-[\w-]+\.txt$/.test(file)) violations.push(`${e.where}: "false-positive" rulings are only for static-analyzer findings (scan-*.txt)`)
      else if ((e.justification ?? '').length < 40) violations.push(`${e.where}: a false-positive needs a justification of at least 40 characters`)
    } else violations.push(`${e.where}: unknown ruling "${e.ruling}" (use "benign-by-spec <SPEC>#<anchor>" or "third-party <component>")`)
  }
  return violations
}

export function loadSuppressions(root, path) {
  if (!path || !existsSync(join(root, path))) return { entries: [], violations: [] }
  const { entries, violations } = parseSuppressions(readFileSync(join(root, path), 'utf8'), path)
  return { entries, violations: [...violations, ...auditEntries(root, entries, { file: path })] }
}

export function auditAll(root, dir = 'agents-architetura/tools/campaign/suppressions') {
  const abs = join(root, dir)
  const files = existsSync(abs) ? readdirSync(abs).filter(f => f.endsWith('.txt')) : []
  const out = { files: files.length, entries: 0, violations: [] }
  for (const f of files) {
    const r = loadSuppressions(root, `${dir}/${f}`)
    out.entries += r.entries.length
    out.violations.push(...r.violations)
  }
  return out
}
