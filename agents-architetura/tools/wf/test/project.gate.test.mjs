import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loadLib, makeRuntime } from '../sim.mjs'

const libWith = respond => {
  const { runtime, calls } = makeRuntime({ respond })
  return { ...loadLib(['hygiene', 'schemas', 'dag', 'project'], { agent: runtime.agent }), calls }
}

test('requireApprovedDesign: only a human grant of the current DESIGN.md text passes; checked by running the tool', async () => {
  const never = libWith(() => ({ approved: false, docs: [{ doc: 'docs/design/DESIGN.md', approved: false, reason: 'never approved — silence is not approval' }] }))
  await assert.rejects(never.requireApprovedDesign('Gate'), /not approved by a human \(never approved — silence is not approval\) — run \/wan-approve in Claude Code, or in your own terminal: node agents-architetura\/tools\/factory\/approve\.mjs grant docs\/design\/DESIGN\.md/)
  assert.equal(never.calls[0].opts.agentType, 'wan-verifier')
  assert.match(never.calls[0].prompt, /node agents-architetura\/tools\/factory\/approve\.mjs check docs\/design\/DESIGN\.md --json/)
  await libWith(() => ({ approved: true, docs: [{ doc: 'docs/design/DESIGN.md', approved: true }] })).requireApprovedDesign('Gate')
  // an "approved" claim that does not name the design is not approval
  await assert.rejects(libWith(() => ({ approved: true, docs: [] })).requireApprovedDesign('Gate'), /not approved/)
  await assert.rejects(libWith(() => null).requireApprovedDesign('Gate'), /not approved/)
})

test('loadWorkstreams validates raw keys and disjoint ownership', async () => {
  const ws = (k1, p1) => ({ workstreams: [{ key: k1, title: 't', scope: 's', ownedPaths: p1 }, { key: 'train', title: 't', scope: 's', ownedPaths: ['src/train/'] }] })
  const ok = await libWith(() => ws('transport', ['src/transport/'])).loadWorkstreams('Gate')
  assert.deepEqual(ok.map(w => w.key), ['transport', 'train'])
  await assert.rejects(libWith(() => ws('Bad Key!', ['src/x/'])).loadWorkstreams('Gate'), /bad task id/)
  await assert.rejects(libWith(() => ws('transport', ['src/'])).loadWorkstreams('Gate'), /scope collision/)
  await assert.rejects(libWith(() => ws('transport', ['agents-architetura/tools/gates/'])).loadWorkstreams('Gate'), /scope collision: factory owns agents-architetura\//)
  await assert.rejects(libWith(() => ws('transport', ['.claude/agents/'])).loadWorkstreams('Gate'), /scope collision: factory owns \.claude\//)
  await assert.rejects(libWith(() => ({ workstreams: [] })).loadWorkstreams('Gate'), /no valid workstreams block/)
})
