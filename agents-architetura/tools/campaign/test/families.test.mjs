import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { groupFamilies, applySuppressions, diffRounds, runRound, roundClean } from '../families.mjs'
import { parseSuppressions, auditAll } from '../suppressions.mjs'
import { parseSanitizer } from '../parse.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const fx = name => readFileSync(join(HERE, 'fixtures', name), 'utf8')
const repo = files => {
  const root = mkdtempSync(join(tmpdir(), 'camp-'))
  for (const [f, body] of Object.entries(files)) { mkdirSync(join(root, dirname(f)), { recursive: true }); writeFileSync(join(root, f), body) }
  return root
}
const noLock = (_, fn) => fn()

test('families: counts per signature, stable ids, deltas across rounds', () => {
  const fams = groupFamilies([...parseSanitizer(fx('tsan.txt')), ...parseSanitizer(fx('asan.txt'))])
  assert.deepEqual(fams.map(f => [f.kind, f.count]), [['tsan-data-race', 2], ['asan-heap-use-after-free', 1], ['ubsan', 1]])
  assert.match(fams[0].id, /^tsan-data-race-[0-9a-f]{8}$/)
  const d = diffRounds([fams[0], fams[1]], [fams[1], fams[2]])
  assert.deepEqual(d, { new: [fams[2].id], fixed: [fams[0].id], persisting: [fams[1].id] })
})

test('suppressions: justified entries suppress; missing ruling, short justification, bad citations are violations', () => {
  const root = repo({ 'docs/design/SPEC-transport.md': '| TRANSPORT-I4 | stats are advisory |\n' })
  const text = [
    '# RULING: benign-by-spec docs/design/SPEC-transport.md#TRANSPORT-I4',
    '# JUSTIFICATION: stats counters are advisory; a stale read only delays a log line.',
    'match: transport::stats::record',
    '',
    'match: codec::decode_frame',
    '',
    '# RULING: benign-by-spec docs/design/SPEC-transport.md#TRANSPORT-I9',
    '# JUSTIFICATION: this anchor does not exist in the spec at all, so it cannot be cited.',
    'match: x',
    '',
    '# RULING: third-party numpy',
    '# JUSTIFICATION: numpy internals race on a private cache we cannot change.',
    'match: tsan|helper@src/wan/x.py',
  ].join('\n')
  writeFileSync(join(root, 'tools.txt'), text)
  mkdirSync(join(root, 'agents-architetura/tools/campaign/suppressions'), { recursive: true })
  writeFileSync(join(root, 'agents-architetura/tools/campaign/suppressions/rust-tsan.txt'), text)
  const { entries } = parseSuppressions(text)
  const fams = groupFamilies(parseSanitizer(fx('tsan.txt')))
  const { kept, suppressed } = applySuppressions(fams, entries)
  assert.equal(kept.length, 0)
  assert.equal(suppressed[0].count, 2)
  const audit = auditAll(root)
  const v = audit.violations.join('\n')
  assert.match(v, /no "# RULING:" line/)
  assert.match(v, /needs a "# JUSTIFICATION:"/)
  assert.match(v, /cited section "TRANSPORT-I9" is not in/)
  assert.match(v, /"third-party" ruling on a pattern that names the repository's own source/)
  assert.equal(audit.violations.length, 4)
})

test('round: runs the command once, parses its reports, diffs with the previous round, persists the table', () => {
  const root = repo({ 'Cargo.toml': '', 'raw/tsan.txt': fx('tsan.txt') })
  const campaign = { id: 'rust-tsan', kind: 'sanitizer', when: ['Cargo.toml'], cmd: 'cp raw/tsan.txt "$FACTORY_CAMPAIGN_REPORTS/tsan.1"; exit 66' }
  const r0 = runRound(root, campaign, 0, { run: true, lock: noLock })
  assert.equal(r0.unsuppressedReports, 2)
  assert.equal(r0.families.length, 1)
  assert.equal(r0.broken, false)
  assert.equal(roundClean(r0), false)
  const r1 = runRound(root, { ...campaign, cmd: 'echo all quiet' }, 1, { run: true, lock: noLock })
  assert.equal(r1.unsuppressedReports, 0)
  assert.deepEqual(r1.delta.fixed, [r0.families[0].id])
  assert.equal(roundClean(r1), true)
  assert.ok(JSON.parse(readFileSync(join(root, '.factory/campaigns/rust-tsan/families-r1.json'), 'utf8')))
})

test('a round whose command fails without any parsable report is BROKEN, never zero; probes gate applicability', () => {
  const root = repo({ 'Cargo.toml': '' })
  const r = runRound(root, { id: 'x', kind: 'sanitizer', cmd: 'echo "error: could not compile"; exit 101' }, 0, { run: true, lock: noLock })
  assert.equal(r.broken, true)
  assert.match(r.brokenReason, /not zero/)
  assert.equal(roundClean(r), false)
  assert.equal(runRound(root, { id: 'y', kind: 'sanitizer', when: ['nope'], cmd: 'true' }, 0, { run: true, lock: noLock }).applicable, false)
  const probe = runRound(root, { id: 'z', kind: 'sanitizer', probe: 'exit 1', cmd: 'true' }, 0, { run: true, lock: noLock })
  assert.equal(probe.applicable, false)
  assert.match(probe.reason, /tool unavailable/)
})

test('CLI: unknown campaign is a usage error; shipped campaigns.json is valid and n/a here', () => {
  const cli = join(HERE, '..', 'families.mjs')
  assert.equal(spawnSync(process.execPath, [cli, '--campaign', 'nope', '--round', '0'], { encoding: 'utf8' }).status, 2)
  const root = repo({})
  mkdirSync(join(root, 'agents-architetura/tools/campaign'), { recursive: true })
  writeFileSync(join(root, 'agents-architetura/tools/campaign/campaigns.json'), readFileSync(join(HERE, '..', 'campaigns.json')))
  for (const id of ['rust-tsan', 'py-devmode', 'amplify-chaos', 'cuda-racecheck']) {
    assert.equal(spawnSync(process.execPath, [cli, '--root', root, '--campaign', id, '--round', '0', '--run'], { encoding: 'utf8' }).status, 5, id)
  }
})

test('a round that does not prove its instrumentation was active is broken, even when clean', () => {
  const root = repo({})
  const quiet = runRound(root, { id: 'ft', kind: 'sanitizer', sentinel: 'GIL False', cmd: 'echo "GIL True"; echo all passed' }, 0, { run: true, lock: noLock })
  assert.equal(quiet.broken, true)
  assert.match(quiet.brokenReason, /not proven active/)
  const ok = runRound(root, { id: 'ft', kind: 'sanitizer', sentinel: 'GIL False', cmd: 'echo "GIL False"; echo all passed' }, 1, { run: true, lock: noLock })
  assert.equal(roundClean(ok), true)
})
