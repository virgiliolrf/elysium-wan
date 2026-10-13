import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { parseSarif, parseDiagnostics, parseFile } from '../parse.mjs'
import { runScanners } from '../scan.mjs'
import { record, mergeFindings } from '../record.mjs'
import { auditAll } from '../suppressions.mjs'

const SARIF = JSON.stringify({ runs: [{ tool: { driver: { name: 'Semgrep OSS' } }, results: [
  { ruleId: 'python.lang.security.pickle', level: 'error', message: { text: 'Avoid pickle on untrusted data' }, locations: [{ physicalLocation: { artifactLocation: { uri: 'src/wan/ckpt.py' }, region: { startLine: 12 } } }] },
  { ruleId: 'python.lang.security.pickle', level: 'error', message: { text: 'Avoid pickle on untrusted data' }, locations: [{ physicalLocation: { artifactLocation: { uri: 'src/wan/ckpt.py' }, region: { startLine: 40 } } }] },
] }] })
const repo = files => {
  const root = mkdtempSync(join(tmpdir(), 'scan-'))
  for (const [f, b] of Object.entries(files)) { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), b) }
  return root
}
const noLock = (_, fn) => fn()

test('SARIF and mypy diagnostics: same rule in the same file is one family across lines', () => {
  const r = parseSarif(SARIF)
  assert.equal(r.length, 2)
  assert.equal(r[0].signature, r[1].signature)
  assert.equal(r[0].kind, 'scan-semgrep-oss')
  assert.equal(r[0].location, 'src/wan/ckpt.py:12')
  const d = parseDiagnostics('src/wan/a.py:3: error: Incompatible return value type (got "int", expected "str")  [return-value]\nSuccess: no issues')
  assert.equal(d.length, 1)
  assert.match(d[0].signature, /^scan\|mypy\|return-value\|src\/wan\/a\.py\|/)
})

test('battery: unavailable scanners are loud, broken ones fail, findings are counted and suppressible', () => {
  const root = repo({
    'src/x.py': '',
    'raw.sarif': SARIF,
    'agents-architetura/tools/campaign/scanners.json': JSON.stringify({ scanners: [
      { id: 'semgrep', when: ['src'], cmd: 'cp raw.sarif "$FACTORY_SCAN_OUT/semgrep.sarif"; exit 1' },
      { id: 'codeql', probe: 'exit 1', cmd: 'true' },
      { id: 'crashy', cmd: 'echo boom; exit 3' },
      { id: 'cargo', when: ['Cargo.toml'], cmd: 'true' },
    ] }),
  })
  const r = runScanners(root, { lock: noLock })
  assert.deepEqual(r.scanners.map(s => [s.id, s.status]), [['semgrep', 'ran'], ['codeql', 'skipped-unavailable'], ['crashy', 'broken'], ['cargo', 'n/a']])
  assert.equal(r.findings, 2)
  assert.deepEqual(r.unavailable, ['codeql'])
  assert.equal(r.ok, false)
  mkdirSync(join(root, 'agents-architetura/tools/campaign/suppressions'), { recursive: true })
  writeFileSync(join(root, 'agents-architetura/tools/campaign/suppressions/scan-semgrep.txt'), '# RULING: false-positive python.lang.security.pickle\n# JUSTIFICATION: ckpt.py only unpickles files the coordinator signed and verified two lines earlier.\nmatch: scan|semgrep-oss|python.lang.security.pickle|src/wan/ckpt.py\n')
  const again = runScanners(root, { lock: noLock, only: ['semgrep'] })
  assert.equal(again.findings, 0)
  assert.equal(again.ok, true)
  writeFileSync(join(root, 'agents-architetura/tools/campaign/suppressions/rust-tsan.txt'), '# RULING: false-positive x\n# JUSTIFICATION: this justification is long enough to pass the forty char bar.\nmatch: y\n')
  assert.match(auditAll(root).violations.join('\n'), /only for static-analyzer findings/)
})

test('record: findings merge by id and validate; unknown targets refused', () => {
  const m = mergeFindings({ findings: [{ id: 'F-1', status: 'open', title: 'a' }] }, { findings: [{ id: 'F-1', status: 'closed' }, { id: 'F-2', status: 'open', title: 'b' }] })
  assert.deepEqual(m.findings.map(f => [f.id, f.status, f.title]), [['F-1', 'closed', 'a'], ['F-2', 'open', 'b']])
  assert.throws(() => mergeFindings(null, { findings: [{ id: 'bad id!', status: 'open' }] }), /bad finding id/)
  const root = repo({})
  const p = record(root, 'audit-results', undefined, { classes: 3, openTierA: 0 })
  assert.equal(JSON.parse(readFileSync(p, 'utf8')).classes, 3)
  assert.throws(() => record(root, '../../etc', undefined, {}), /unknown record target/)
  assert.throws(() => record(root, 'campaign', '../x', {}), /unknown record target/)
})

test('dependency auditors: pip-audit JSON and cargo-deny JSON lines become findings', () => {
  const pip = parseFile('/o/pip-audit.json', JSON.stringify({ dependencies: [{ name: 'torch', version: '2.5.0', vulns: [{ id: 'GHSA-53q9-r3pm-6pq6', fix_versions: ['2.6.0'] }] }, { name: 'numpy', version: '2.0', vulns: [] }] }))
  assert.deepEqual(pip.map(r => r.signature), ['scan|pip-audit|GHSA-53q9-r3pm-6pq6|torch'])
  const deny = parseFile('/o/cargo-deny.jsonl', [JSON.stringify({ type: 'diagnostic', fields: { severity: 'error', code: 'vulnerability', message: 'RUSTSEC-2024-0001 in foo 1.2' } }), JSON.stringify({ type: 'summary' }), 'not json'].join('\n'))
  assert.equal(deny.length, 1)
  assert.match(deny[0].signature, /^scan\|cargo-deny\|vulnerability\|/)
})
