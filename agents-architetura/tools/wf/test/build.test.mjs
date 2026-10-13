import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { buildScript, buildAll, extractMeta, GENERATED_MARK } from '../build.mjs'

const META = `export const meta = {
  name: 'demo',
  description: 'demo workflow',
  phases: [{ title: 'One', detail: 'x' }],
}
`
function fixture() {
  const dir = mkdtempSync(join(tmpdir(), 'wf-build-'))
  for (const d of ['lib', 'src', 'out']) mkdirSync(join(dir, d))
  writeFileSync(join(dir, 'lib', 'greet.js'), "const greet = n => `hi ${n}`\n")
  return dir
}

test('extractMeta parses a pure literal and finds its end', () => {
  const { meta, end } = extractMeta(`${META}phase('One')\n`)
  assert.equal(meta.name, 'demo')
  assert.equal(`${META}phase('One')\n`.slice(end).trim(), "phase('One')")
})

test('meta rules: must start the file, no interpolation, no identifiers, no spreads, no comments', () => {
  assert.throws(() => extractMeta(`// hi\n${META}`), /must begin with/)
  assert.throws(() => extractMeta("export const meta = { name: `x${1}`, description: 'd' }"), /pure literal/)
  assert.throws(() => extractMeta("export const meta = { name: NAME, description: 'd' }"), /pure literal/)
  assert.throws(() => extractMeta("export const meta = { ...base, name: 'x' }"), /no spreads/)
  assert.throws(() => extractMeta("export const meta = { name: 'x', // c\n description: 'd' }"), /comments/)
})

test('includes are spliced, phases checked, banner added', () => {
  const dir = fixture()
  const { script } = buildScript(`${META}// @include greet\nphase('One')\nlog(greet('x'))\n`, { name: 'demo', libDir: join(dir, 'lib') })
  assert.ok(script.startsWith('export const meta = {'))
  assert.match(script, /---- begin lib\/greet.js ----\nconst greet/)
  assert.ok(script.includes(GENERATED_MARK))
})

test('rejections: undeclared phase, name mismatch, nondeterminism, double include, unknown include, syntax', () => {
  const dir = fixture()
  const lib = join(dir, 'lib')
  assert.throws(() => buildScript(`${META}phase('Two')\n`, { name: 'demo', libDir: lib }), /not declared in meta.phases: Two/)
  assert.throws(() => buildScript(`${META}agent('x', { phase: 'Nope' })\n`, { name: 'demo', libDir: lib }), /Nope/)
  assert.throws(() => buildScript(META, { name: 'other', libDir: lib }), /must equal the file name/)
  assert.throws(() => buildScript(`${META}const t = Date.now()\n`, { name: 'demo', libDir: lib }), /deterministic/)
  assert.throws(() => buildScript(`${META}// @include greet\n// @include greet\n`, { name: 'demo', libDir: lib }), /included twice/)
  assert.throws(() => buildScript(`${META}// @include nope\n`, { name: 'demo', libDir: lib }), /unknown include: nope/)
  assert.throws(() => buildScript(`${META}const x = (\n`, { name: 'demo', libDir: lib }), /does not parse/)
})

test('lib files may not import or export', () => {
  const dir = fixture()
  writeFileSync(join(dir, 'lib', 'bad.js'), "import fs from 'node:fs'\n")
  assert.throws(() => buildScript(`${META}// @include bad\n`, { name: 'demo', libDir: join(dir, 'lib') }), /must not import\/export/)
})

test('buildAll writes outputs; --check detects stale and orphaned files', () => {
  const dir = fixture()
  const opts = { srcDir: join(dir, 'src'), outDir: join(dir, 'out'), libDir: join(dir, 'lib') }
  writeFileSync(join(dir, 'src', 'demo.js'), `${META}phase('One')\n`)
  assert.deepEqual(buildAll(opts), { built: ['demo'], problems: [] })
  assert.deepEqual(buildAll({ ...opts, check: true }).problems, [])
  writeFileSync(join(dir, 'src', 'demo.js'), `${META}phase('One')\nlog('changed')\n`)
  assert.match(buildAll({ ...opts, check: true }).problems[0], /demo.js: stale/)
  writeFileSync(join(dir, 'out', 'gone.js'), `${META}\n${GENERATED_MARK}\n`)
  assert.ok(buildAll({ ...opts, check: true }).problems.some(p => /gone.js: .*orphan/.test(p)))
  assert.ok(readFileSync(join(dir, 'out', 'demo.js'), 'utf8').startsWith('export const meta'))
})

test('every real workflow source builds', () => {
  const { problems } = buildAll({ check: false, outDir: mkdtempSync(join(tmpdir(), 'wf-real-')) })
  assert.deepEqual(problems, [])
})
