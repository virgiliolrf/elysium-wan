import { test } from 'node:test'
import assert from 'node:assert/strict'
import { simulate, byLabel } from '../sim.mjs'
import { built } from './helpers.mjs'

const script = built('wan-smoke')
const ROLES = ['wan-readonly', 'wan-reviewer', 'wan-proposer', 'wan-security']
const echo = p => ({ echo: p.match(/"([^"]+)"/)[1] })

test('passes when echoes match and no read-only role can write', async () => {
  const { result, calls } = await simulate(script, {
    args: { token: 'abc' },
    respond: byLabel({ 'echo:': echo, 'readonly:attempt': () => ({ wrote: false, tools: ['Read', 'Grep', 'Glob'] }), 'readonly:probe': () => ({ exists: false }) }),
  })
  assert.equal(result.ok, true)
  assert.deepEqual(calls.filter(c => c.opts.label.startsWith('readonly:attempt')).map(c => c.opts.agentType), ROLES)
})

test('fails, naming the role, when one read-only role managed to write', async () => {
  const { result } = await simulate(script, {
    respond: byLabel({
      'echo:': echo,
      'readonly:attempt': () => ({ wrote: false, tools: ['Read'] }),
      'readonly:probe': (p, o) => ({ exists: o.label === 'readonly:probe:wan-proposer' }),
    }),
  })
  assert.equal(result.ok, false)
  assert.deepEqual(result.checks.filter(c => !c.ok).map(c => c.check), ['wan-proposer-cannot-write'])
})

test('a hostile token arg is not interpolated into prompts', async () => {
  const { calls } = await simulate(script, { args: { token: 'x"; rm -rf ~' }, respond: () => null })
  assert.ok(calls.every(c => !c.prompt.includes('rm -rf')))
})
