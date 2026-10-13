import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseSanitizer, parsePython, parseRust, parseComputeSanitizer, parseJunit, parseFile } from '../parse.mjs'

const fx = name => readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'fixtures', name), 'utf8')

test('TSAN: two-stack race; stack order, line numbers, addresses and checkout path do not split the family', () => {
  const r = parseSanitizer(fx('tsan.txt'))
  assert.equal(r.length, 2)
  assert.equal(r[0].kind, 'tsan-data-race')
  assert.equal(r[0].signature, r[1].signature)
  assert.match(r[0].signature, /transport::stats::record@src\/transport\/stats\.rs/)
  assert.ok(!r[0].signature.includes('library/std'))
  assert.equal(r[0].location, 'src/transport/stats.rs:41')
})

test('ASAN and UBSAN', () => {
  const r = parseSanitizer(fx('asan.txt'))
  assert.deepEqual(r.map(x => x.kind), ['asan-heap-use-after-free', 'ubsan'])
  assert.match(r[0].signature, /^asan-heap-use-after-free\|codec::decode_frame@src\/codec\/frame\.rs;codec::tests::roundtrip@src\/codec\/tests\.rs$/)
  assert.equal(r[1].signature, 'ubsan|src/codec/len.rs|signed integer overflow: N + N cannot be represented in type \'int\'')
})

test('Python traceback: exception type plus the deepest in-repo frames', () => {
  const [r] = parsePython(fx('python.txt'))
  assert.equal(r.kind, 'py-StaleRoundError')
  assert.equal(r.signature, 'py-StaleRoundError|apply@src/wan/sync/apply.py;run_round@src/wan/sync/round.py;test_reorder@tests/chaos/test_reorder.py')
  assert.equal(r.location, 'src/wan/sync/apply.py:15')
})

test('Rust panic and Miri UB', () => {
  const r = parseRust(fx('rust.txt'))
  assert.deepEqual(r.map(x => x.kind), ['rust-panic', 'miri-ub'])
  assert.equal(r[0].signature, 'rust-panic|src/codec/frame.rs|index out of bounds: the len is N but the index is N')
  assert.match(r[1].signature, /^miri-ub\|src\/ring\/buffer\.rs\|/)
})

test('compute-sanitizer races and invalid accesses', () => {
  const r = parseComputeSanitizer(fx('cuda.txt'))
  assert.equal(r.length, 2)
  assert.equal(r[0].signature, 'cuda-race-reported|reduce_kernel')
  assert.equal(r[1].signature, 'cuda-invalid-__global__|scatter_kernel')
})

test('junit: failures by test id; Hypothesis falsifying examples are property failures; passes ignored', () => {
  const r = parseJunit(fx('junit.xml'))
  assert.deepEqual(r.map(x => x.signature), ['property-failure|tests.unit.test_codec::test_props|AssertionError', 'test-failure|tests.chaos.test_partition::test_heal|TimeoutError'])
})

test('driver summaries: inner reports are re-kinded by outcome; opaque findings still produce a family', () => {
  const summary = JSON.stringify({ cmd: 'pytest tests/chaos', findings: [
    { outcome: 'HANG', tail: 'Timeout' },
    { outcome: 'FAIL', tail: fx('python.txt') },
  ] })
  const r = parseFile('/x/summary.json', summary)
  assert.equal(r.length, 2)
  assert.match(r[0].signature, /^hang\|/)
  assert.match(r[1].signature, /^fail:py-StaleRoundError\|/)
})
