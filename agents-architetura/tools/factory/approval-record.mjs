#!/usr/bin/env node
// PostToolUse hook (Bash): records a human approval granted by answering Claude Code.
//
// `node agents-architetura/tools/factory/approve.mjs confirm <doc...> --sha <sha12,...>` writes nothing. It matches an
// explicit ASK rule in .claude/settings.json, and Claude Code puts ask-rule matches to the human in every permission
// mode, bypassPermissions included ("Claude Code doesn't auto-approve the following in any mode, including
// bypassPermissions: Tools matched by an explicit ask rule"); background agents cannot show the prompt and are denied.
// PostToolUse fires only after a tool ran (a denial never runs it), so this hook runs after the human said yes, and it
// records the grant only for that exact command shape. Anything else, an obfuscated variant included, matches no ask
// rule but also records nothing. A changed document is refused (approve.mjs verifyShown). Every failure means "not
// approved".
//
// Residual risk, the same as before this path existed: a deliberately malicious agent can call the library
// (recordGrant), feed this script a forged payload, or edit the gate's files from a shell (the Edit denies cover only
// Claude Code's file tools). No control inside the repository stops that in a
// session without permission prompts.
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { confirmedGrant } from './approve.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
export const CONFIRM = /^node (?:\.\/)?agents-architetura\/tools\/factory\/approve\.mjs confirm((?: [\w./-]+)+) --sha ([0-9a-f]{12,64}(?:,[0-9a-f]{12,64})*)$/

export const ASK_RULES = [
  'Bash(node agents-architetura/tools/factory/approve.mjs confirm:*)',
  'Bash(node ./agents-architetura/tools/factory/approve.mjs confirm:*)',
]
const MANAGED = ['/Library/Application Support/ClaudeCode/managed-settings.json', '/etc/claude-code/managed-settings.json']

// A hook matcher is a tool-name pattern. Fail closed: empty, "*", an exact "Bash", a pattern that matches "Bash" as a
// regex, or one that does not compile all count as able to answer a Bash prompt.
export function matcherHitsBash(matcher) {
  const m = String(matcher ?? '')
  if (!m || m === '*' || m === 'Bash') return true
  try {
    return new RegExp(m).test('Bash')
  } catch {
    return true
  }
}

const readJson = p => { try { return JSON.parse(readFileSync(p, 'utf8')) } catch { return null } }
const permissionHooks = settings => {
  const h = settings?.hooks
  return Array.isArray(h?.PermissionRequest) ? h.PermissionRequest : []
}

// A Claude Code process, judged by its executable (ps comm), never by command-line text: a shell whose command line
// merely mentions --settings is not Claude Code. The CLI runs as the native `claude` binary, or under node/bun.
export function isClaudeProcess(comm, args) {
  const exe = String(comm ?? '').split('/').pop()
  if (exe === 'claude') return true
  return (exe === 'node' || exe === 'bun') && /@anthropic-ai\/claude-code\/|\/claude(?:\.js|\.mjs)?(?:\s|$)/.test(String(args ?? ''))
}

// The Claude Code processes this hook runs under, with their command lines (ps), nearest first.
export function claudeAncestors(start = process.ppid, max = 10) {
  const ps = (field, pid) => (spawnSync('ps', ['-ww', '-o', `${field}=`, '-p', String(pid)], { encoding: 'utf8' }).stdout ?? '').trim()
  const out = []
  let pid = start
  for (let i = 0; i < max && pid > 1; i++) {
    const comm = ps('comm', pid)
    if (!comm) break
    const args = ps('args', pid)
    if (isClaudeProcess(comm, args)) out.push(args)
    pid = Number(ps('ppid', pid)) || 0
  }
  return out
}

// The JSON object that follows --settings in a command line (string-aware brace matching), or a settings file path.
function settingsArg(args) {
  const at = args.indexOf('--settings')
  if (at < 0) return { present: false }
  const rest = args.slice(at + '--settings'.length).replace(/^[= ]/, '')
  if (!rest.startsWith('{')) {
    const path = rest.split(' ')[0]
    return { present: true, value: readJson(path), raw: path }
  }
  let depth = 0
  let quote = false
  for (let i = 0; i < rest.length; i++) {
    const c = rest[i]
    if (quote) { if (c === '\\') i++; else if (c === '"') quote = false; continue }
    if (c === '"') quote = true
    else if (c === '{') depth++
    else if (c === '}' && --depth === 0) {
      try { return { present: true, value: JSON.parse(rest.slice(0, i + 1)) } } catch { return { present: true, value: null } }
    }
  }
  return { present: true, value: null }
}

// The prompt is only as good as the configuration that forces it. Record nothing unless the confirm ask rule is in the
// project settings and nothing that Claude Code loads could answer a Bash prompt for the human:
// - no PermissionRequest hook matching Bash in any settings file (project, local, user, managed), in an installed
//   plugin, in a --settings argument of the Claude Code process, or in agent or skill frontmatter;
// - no --permission-prompt-tool other than "stdio" (the desktop app's own UI).
// (Agents are also denied edits to these files. A deliberate shell-level subversion remains the residual risk.)
export function gateProblems(root, home = homedir(), { ancestors = claudeAncestors } = {}) {
  const problems = []
  const ask = readJson(join(root, '.claude/settings.json'))?.permissions?.ask ?? []
  for (const r of ASK_RULES) if (!ask.includes(r)) problems.push(`the ask rule ${r} is missing from .claude/settings.json`)
  const check = (where, settings) => {
    for (const h of permissionHooks(settings)) if (matcherHitsBash(h?.matcher)) problems.push(`${where} has a PermissionRequest hook (matcher ${JSON.stringify(h?.matcher ?? '')}) that could answer the prompt for the human`)
  }
  for (const f of [join(root, '.claude/settings.json'), join(root, '.claude/settings.local.json'), join(home, '.claude/settings.json'), join(home, '.claude/settings.local.json'), ...MANAGED]) check(f, readJson(f))
  const installed = readJson(join(home, '.claude/plugins/installed_plugins.json'))?.plugins ?? {}
  for (const entries of Object.values(installed)) {
    for (const e of Array.isArray(entries) ? entries : []) {
      if (!e?.installPath) continue
      check(join(e.installPath, 'hooks/hooks.json'), readJson(join(e.installPath, 'hooks/hooks.json')))
      check(join(e.installPath, '.claude-plugin/plugin.json'), readJson(join(e.installPath, '.claude-plugin/plugin.json')))
    }
  }
  for (const dir of [join(root, '.claude/agents'), join(root, '.claude/skills'), join(home, '.claude/agents'), join(home, '.claude/skills')]) {
    for (const f of markdownUnder(dir)) {
      const front = (readFileSync(f, 'utf8').match(/^---\n([\s\S]*?)\n---/) ?? [])[1] ?? ''
      if (/PermissionRequest/.test(front)) problems.push(`${f} declares a PermissionRequest hook in its frontmatter`)
    }
  }
  for (const args of ancestors()) {
    const s = settingsArg(args)
    if (s.present && !s.value) problems.push('the Claude Code process has a --settings argument that cannot be read')
    if (s.present && s.value) check('the Claude Code --settings argument', s.value)
    const tool = (args.match(/--permission-prompt-tool[= ](\S+)/) ?? [])[1]
    if (tool && tool !== 'stdio') problems.push(`the Claude Code process answers prompts with --permission-prompt-tool ${tool}, not the human's UI`)
  }
  return problems
}

function markdownUnder(dir, depth = 0) {
  if (depth > 3 || !existsSync(dir)) return []
  const out = []
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name)
    if (e.isDirectory()) out.push(...markdownUnder(p, depth + 1))
    else if (e.name.endsWith('.md')) out.push(p)
  }
  return out
}

// -> null (not ours) | { docs, sha }
export function parseConfirm(command) {
  const m = String(command ?? '').match(CONFIRM)
  return m ? { docs: m[1].trim().split(' '), sha: m[2] } : null
}

export function record(root, payload, { who, home, ancestors } = {}) {
  if (payload?.tool_name !== 'Bash') return null
  const parsed = parseConfirm(payload.tool_input?.command)
  if (!parsed) return null
  const problems = gateProblems(root, home, ancestors ? { ancestors } : undefined)
  if (problems.length) throw new Error(`the approval gate is not intact, nothing recorded: ${problems.join('; ')}`)
  const by = `${who ?? 'human'} (Claude Code prompt)`
  return confirmedGrant(root, parsed.docs, parsed.sha, { approvedBy: by })
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  let payload
  try {
    payload = JSON.parse(readFileSync(0, 'utf8') || '{}')
  } catch {
    process.exit(0)
  }
  try {
    const who = spawnSync('git', ['config', 'user.name'], { cwd: ROOT, encoding: 'utf8' }).stdout.trim() || 'human'
    const done = record(ROOT, payload, { who })
    if (done) {
      const text = done.map(r => `APPROVED ${r.doc} ${r.sha256.slice(0, 12)} by ${r.approvedBy}`).join('\n')
      process.stdout.write(JSON.stringify({ hookSpecificOutput: { hookEventName: 'PostToolUse', additionalContext: `${text}\n(recorded in docs/design/APPROVALS.json by the approval hook)` } }))
    }
    process.exit(0)
  } catch (e) {
    process.stderr.write(`approval not recorded: ${e.message}`)
    process.exit(2)
  }
}
