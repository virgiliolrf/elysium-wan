#!/usr/bin/env node
// Publishing, the reference PR's git model made automatic. In oven-sh/WebKit#249 the workflows never touch git: one
// long-lived branch and one PR carry the work, a timer commits "Checkpoint ... work in progress" every 30 minutes
// (only when something changed), and the orchestrating session writes one descriptive commit per finished stage with
// the measured numbers — red or green. Here the same model is one guarded tool:
//
//   start --task <slug> | --text-b64 <b64 task text> [--base main]
//                                        create or resume branch factory/<slug> (from a clean base only)
//   commit --stage <name> --b64 <json>   commit everything, message derived from the stage result; push; open the
//                                        draft PR on the first push; rewrite the PR description (the living bring-up log)
//   checkpoint                           "Checkpoint <task> work in progress [skip ci]" if anything changed; skipped
//                                        while the machine lock is held (benchmarks must not be perturbed)
//   draft                                put the PR back into draft (a new milestone starts)
//   ready --tree <tree id>               leave draft: only if the tree closeout VERIFIED is the tree on the branch
//                                        (the closeout report and LESSONS.md may have been written since), clean, pushed
//   watch [--every 1800] [--idle-exit 3] [--detach]   checkpoint timer; exits after N idle ticks or off the branch
//   stop-watch · status
//
// Guards (all exit 3, nothing pushed): never a branch other than factory/<slug> — so never main; never force; never
// merge (merging is human-only); the commit identity must be a GitHub noreply address; staged files are refused when
// oversized, when they look like key material, or when an added line matches a secret pattern (a reviewed false
// positive carries "secret-scan: allow" on its line). A refused commit is unstaged and the files stay as they were.
// Exit: 0 ok · 1 git/gh failure (the commit is kept; the next publish retries the push) · 2 usage · 3 refused ·
// 4 not on a factory/* branch (nothing published).
import { readFileSync, writeFileSync, mkdirSync, statSync, openSync, closeSync, rmSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync, spawn } from 'node:child_process'
import { optionReader, intOpt } from '../verify/exec.mjs'
import { readLock as readMachineLock } from '../verify/machine-lock.mjs'
import { snapshot } from '../wf/scope-audit.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const SELF = fileURLToPath(import.meta.url)
export const BRANCH_RE = /^factory\/[a-z0-9][a-z0-9._-]{0,60}$/
const SLUG_RE = /^[a-z0-9][a-z0-9._-]{0,60}$/
const TREE_RE = /^[0-9a-f]{40,64}$/
export const NOREPLY_RE = /@users\.noreply\.github\.com$/
export const MAX_FILE_BYTES = 20 * 1024 * 1024
// What may change between the tree closeout VERIFIED and the tree marked ready: what closeout itself writes afterwards
// (the milestone report, lessons, and the first benchmark baseline, recorded on the verified build).
export const READY_DOC_PATHS = ['agents-architetura/docs/reports/', 'agents-architetura/docs/LESSONS.md', 'bench/baseline.json']
const PR_FOOTER = '🤖 Generated with [Claude Code](https://claude.com/claude-code)'
export const ALLOW_MARK = 'secret-scan: allow'
export const SECRET_PATTERNS = [
  ['private key', /-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----/],
  ['AWS access key', /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/],
  ['GitHub token', /\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b/],
  ['Anthropic key', /\bsk-ant-[A-Za-z0-9_-]{20,}/],
  ['OpenAI-style key', /\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}/],
  ['Slack token', /\bxox[abprs]-[A-Za-z0-9-]{10,}/],
  ['Google API key', /\bAIza[0-9A-Za-z_-]{35}\b/],
  ['assigned secret', /(?:api[_-]?key|secret|token|passw(?:or)?d)["']?\s*[:=]\s*["'][A-Za-z0-9/+_=.-]{24,}["']/i],
]
export const FORBIDDEN_FILES = [/(^|\/)\.env(\.(?!example$)[^/]+)?$/, /\.(pem|key|p12|pfx|keystore|jks)$/i, /(^|\/)id_(rsa|dsa|ecdsa|ed25519)(\.pub)?$/, /(^|\/)\.netrc$/, /(^|\/)credentials(\.json)?$/i]

class Refused extends Error { constructor(msg, extra = {}) { super(msg); this.code = 3; this.extra = extra } }
class NotFactory extends Error { constructor(msg) { super(msg); this.code = 4 } }
class Failed extends Error { constructor(msg) { super(msg); this.code = 1 } }

const realGit = (root, args, input) => execFileSync('git', args, { cwd: root, encoding: 'utf8', input, stdio: ['pipe', 'pipe', 'pipe'], maxBuffer: 256 * 1024 * 1024 }).replace(/\n$/, '')
const realGh = (root, args) => execFileSync('gh', args, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim()
const tryGit = (git, root, args) => { try { return git(root, args) } catch { return null } }
const errText = e => String(e?.stderr || e?.message || e).trim().split('\n').slice(-3).join(' ').slice(0, 400)

const stateDir = root => join(root, '.factory', 'publish')
const statePath = root => join(stateDir(root), 'state.json')
export const loadState = root => { try { return JSON.parse(readFileSync(statePath(root), 'utf8')) } catch { return {} } }
const saveState = (root, patch) => {
  mkdirSync(stateDir(root), { recursive: true })
  const next = { ...loadState(root), ...patch }
  writeFileSync(statePath(root), JSON.stringify(next, null, 2) + '\n')
  return next
}
const alive = pid => { if (!Number.isInteger(pid) || pid <= 0) return false; try { process.kill(pid, 0); return true } catch (e) { return e.code === 'EPERM' } }

// Branch-safe slug of the task text, at most 40 characters, cut at a word boundary.
export const slugify = text => {
  const s = String(text ?? '').toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
  if (s.length <= 40) return s || 'task'
  const cut = s.slice(0, 41)
  return cut.lastIndexOf('-') > 0 ? cut.slice(0, cut.lastIndexOf('-')) : s.slice(0, 40)
}

const currentBranch = (root, git) => tryGit(git, root, ['symbolic-ref', '--short', '-q', 'HEAD']) ?? ''
const requireFactoryBranch = (root, git) => {
  const b = currentBranch(root, git)
  if (!BRANCH_RE.test(b)) throw new NotFactory(`not on a factory/* branch (on ${b || 'a detached HEAD'}) — nothing published`)
  return b
}
const requireIdentity = (root, git) => {
  const email = tryGit(git, root, ['config', 'user.email']) ?? ''
  if (!NOREPLY_RE.test(email)) throw new Refused(`git user.email must be a GitHub noreply address (…@users.noreply.github.com), got ${email ? 'a different address' : 'none'} — set it with git config user.email`)
}

// Exclusive lock around index operations (stage commits vs the checkpoint timer). Stale locks (dead pid) are reclaimed.
const lockFile = root => join(stateDir(root), 'lock')
function withLock(root, fn, { waitMs = 120000, sleep = ms => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms) } = {}) {
  mkdirSync(stateDir(root), { recursive: true })
  const deadline = Date.now() + waitMs
  for (;;) {
    try {
      writeFileSync(lockFile(root), String(process.pid), { flag: 'wx' })
      break
    } catch (e) {
      if (e.code !== 'EEXIST') throw e
      let raw = null
      try { raw = readFileSync(lockFile(root), 'utf8') } catch { continue } // released meanwhile
      // An empty file is a lock being created right now, not a stale one.
      if (raw !== '' && !alive(Number(raw))) { rmSync(lockFile(root), { force: true }); continue }
      if (Date.now() >= deadline) return { busy: true, holder: Number(raw) || null }
      sleep(250)
    }
  }
  try { return fn() } finally { rmSync(lockFile(root), { force: true }) }
}

// ---- message ------------------------------------------------------------------------------------------------------
const oneLine = s => String(s).replace(/\s+/g, ' ').trim()
const render = v => {
  if (v === null || v === undefined) return 'null'
  if (typeof v === 'string') return oneLine(v).slice(0, 300)
  if (typeof v !== 'object') return String(v)
  if (Array.isArray(v)) {
    if (!v.length) return '(none)'
    const shown = v.slice(0, 12).map(x => (x !== null && typeof x === 'object' ? JSON.stringify(x) : oneLine(String(x)))).map(x => x.slice(0, 120))
    return shown.join(', ') + (v.length > 12 ? `, … (+${v.length - 12})` : '')
  }
  return JSON.stringify(v).slice(0, 300)
}
// The message is a pure function of the stage result: no agent writes it, so it cannot be embellished. No
// Co-Authored-By or other attribution trailer (CLAUDE.md, "Commits").
export function messageFor({ task, stage, result }) {
  const r = result && typeof result === 'object' && !Array.isArray(result) ? result : {}
  const status = typeof r.verdict === 'string' && r.verdict ? oneLine(r.verdict) : r.ok === true ? 'ok' : r.ok === false ? 'NOT OK' : 'done'
  const subject = oneLine(`${task}: ${stage} — ${status}`).slice(0, 100)
  const body = Object.entries(r).filter(([k]) => k !== 'verdict' && k !== 'ok').map(([k, v]) => `${oneLine(k).slice(0, 40)}: ${render(v)}`).join('\n').slice(0, 4000)
  return body ? `${subject}\n\n${body}\n` : `${subject}\n`
}
export const checkpointMessage = task => `Checkpoint ${task} work in progress [skip ci]\n`

// The reference PR's description is a living bring-up log: what this is, the status now, what it took to get here (every
// stage, red or green) and an honest section. Here it is a pure function of the recorded stage results — no agent writes
// it, so it cannot be embellished — rewritten on the PR after every stage commit.
export function stageEntry({ stage, result, sha = null, at }) {
  const r = result && typeof result === 'object' && !Array.isArray(result) ? result : {}
  const status = typeof r.verdict === 'string' && r.verdict ? oneLine(r.verdict) : r.ok === true ? 'ok' : r.ok === false ? 'NOT OK' : 'done'
  const detail = Object.entries(r).filter(([k]) => k !== 'verdict' && k !== 'ok').slice(0, 8).map(([k, v]) => `${oneLine(k).slice(0, 40)}: ${render(v)}`).join('; ').slice(0, 500)
  return { stage: oneLine(String(stage)).slice(0, 80), status, detail, sha, at, red: r.ok === false || /NOT VERIFIED/.test(status) }
}
const HOW = `One descriptive commit per finished stage (the message is derived from the stage result), plus checkpoints while a run is long. This PR stays a draft until a milestone closes VERIFIED on the exact tree it carries; merging is the human's.`
export function bodyFor({ task, branch, text = '', log = [] }) {
  const quote = t => oneLine(t).slice(0, 2000)
  const last = log.at(-1)
  const red = log.filter(e => e.red)
  return [
    '## What this is', text ? `> ${quote(text)}` : `Factory run \`${task}\` on \`${branch}\`.`, '',
    '## Status, current', last ? `**${last.stage}**: ${last.status}${last.detail ? ` (${last.detail})` : ''}` : 'No stage has finished yet.', '',
    '## What it took to get here',
    ...(log.length ? log.map((e, i) => `${i + 1}. \`${e.stage}\`: ${e.status}${e.detail ? ` (${e.detail})` : ''}${e.sha ? `, ${e.sha.slice(0, 7)}` : ''}`) : ['Nothing yet.']), '',
    '## The honest section',
    ...(red.length ? red.map(e => `- \`${e.stage}\` was red: ${e.detail || e.status}`) : ['- No stage has been red so far.']),
    '- The publisher derives every line here from a stage result. No agent wrote this description.', '',
    '## How this PR works', HOW,
  ].join('\n').slice(0, 60000)
}

// ---- guards on the staged change ----------------------------------------------------------------------------------
export function guardStaged(root, git, { maxBytes = MAX_FILE_BYTES } = {}) {
  const findings = []
  const files = git(root, ['diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z']).split('\0').filter(Boolean)
  for (const f of files) {
    if (FORBIDDEN_FILES.some(re => re.test(f))) findings.push({ file: f, kind: 'key material or credentials file' })
    try { if (statSync(join(root, f)).size > maxBytes) findings.push({ file: f, kind: `larger than ${Math.round(maxBytes / 1048576 * 10) / 10} MB` }) } catch { /* deleted after staging */ }
  }
  // Added lines only; the secret itself is never echoed.
  let file = null
  let line = 0
  for (const l of git(root, ['diff', '--cached', '-U0', '--no-color', '--no-ext-diff', '--diff-filter=ACMR']).split('\n')) {
    if (l.startsWith('+++ ')) {
      let p = l.slice(4)
      if (p.startsWith('"')) { try { p = JSON.parse(p) } catch { /* keep git's quoting */ } }
      file = p.replace(/^b\//, '')
      continue
    }
    const h = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(l)
    if (h) { line = Number(h[1]); continue }
    if (l.startsWith('+')) {
      if (!l.includes(ALLOW_MARK)) for (const [kind, re] of SECRET_PATTERNS) if (re.test(l)) { findings.push({ file, line, kind }); break }
      line++
    }
  }
  return findings
}

// ---- operations ---------------------------------------------------------------------------------------------------
export function start({ root = ROOT, task, base = 'main', text = '', git = realGit } = {}) {
  if (!SLUG_RE.test(task ?? '')) throw Object.assign(new Error(`--task must match ${SLUG_RE}`), { code: 2 })
  if (!/^[\w./-]+$/.test(base) || base.startsWith('factory/')) throw Object.assign(new Error('--base must be a plain branch name, not a factory branch'), { code: 2 })
  requireIdentity(root, git)
  const branch = `factory/${task}`
  const cur = currentBranch(root, git)
  let resumed = cur === branch
  if (!resumed) {
    if (cur !== base) throw new Refused(`start from ${base}: the checkout is on ${cur || 'a detached HEAD'}`)
    const dirty = git(root, ['status', '--porcelain']).split('\n').filter(Boolean)
    if (dirty.length) throw new Refused(`${base} has ${dirty.length} uncommitted path(s) — commit or stash them first, or they would land in the task's first commit (e.g. ${dirty.slice(0, 3).map(l => l.slice(3)).join(', ')})`)
    const exists = tryGit(git, root, ['rev-parse', '--verify', '--quiet', `refs/heads/${branch}`]) !== null
    git(root, exists ? ['switch', '-q', branch] : ['switch', '-q', '-c', branch])
    resumed = exists
  }
  const prev = loadState(root)
  saveState(root, { task, branch, base, startedAt: prev.branch === branch ? prev.startedAt : new Date().toISOString(), text: text || (prev.branch === branch ? prev.text ?? '' : ''), log: prev.branch === branch ? prev.log ?? [] : [] })
  return { ok: true, branch, base, resumed }
}

const upstreamAhead = (root, git) => {
  const n = tryGit(git, root, ['rev-list', '--count', '@{u}..HEAD'])
  return n === null ? null : Number(n)
}
function push(root, git, branch) {
  const ahead = upstreamAhead(root, git)
  if (ahead === 0) return false
  if (!tryGit(git, root, ['remote', 'get-url', 'origin'])) throw new Failed('no "origin" remote to push to')
  try { git(root, ['push', '-q', '-u', 'origin', `refs/heads/${branch}:refs/heads/${branch}`]) } catch (e) { throw new Failed(`push failed (commit kept locally; the next publish retries): ${errText(e)}`) }
  return true
}
const findPr = (root, gh, branch) => {
  const list = JSON.parse(gh(root, ['pr', 'list', '--head', branch, '--state', 'open', '--json', 'number,url,isDraft']) || '[]')
  return list[0] ?? null
}
function ensurePr(root, gh, { branch, base, task }) {
  let pr = findPr(root, gh, branch)
  if (!pr) {
    const body = `Factory run \`${task}\` (branch \`${branch}\`).\n\nOne descriptive commit per finished stage (the message is derived from the stage result), plus checkpoints while a run is long. This PR stays a draft until a milestone closes VERIFIED; it goes back to draft when the next milestone starts. Merging is human-only.\n\n${PR_FOOTER}`
    gh(root, ['pr', 'create', '--draft', '--base', base, '--head', branch, '--title', `factory: ${task}`, '--body', body])
    pr = findPr(root, gh, branch)
    if (!pr) throw new Failed('gh pr create reported success but no open PR was found')
  }
  saveState(root, { pr: { number: pr.number, url: pr.url } })
  return pr
}

export function commit({ root = ROOT, stage, result = {}, checkpoint = false, git = realGit, gh = realGh, maxBytes, waitMs, now = () => new Date().toISOString() } = {}) {
  if (!checkpoint && (typeof stage !== 'string' || !/^[\w .:/@-]{1,80}$/.test(stage))) throw Object.assign(new Error('--stage must be 1..80 plain characters'), { code: 2 })
  const branch = requireFactoryBranch(root, git)
  requireIdentity(root, git)
  const st = loadState(root)
  const task = st.branch === branch && st.task ? st.task : branch.slice('factory/'.length)
  const base = st.branch === branch && st.base ? st.base : 'main'
  if (checkpoint) {
    const m = readMachineLock(root)
    if (m && alive(m.pid)) return { ok: true, skipped: `machine lock held by ${String(m.label ?? 'a heavy job').slice(0, 60)}`, committed: false }
  }
  const out = withLock(root, () => {
    git(root, ['add', '-A'])
    const staged = git(root, ['diff', '--cached', '--name-only']).split('\n').filter(Boolean)
    let sha = null
    if (staged.length) {
      const findings = guardStaged(root, git, { maxBytes })
      if (findings.length) {
        git(root, ['reset', '-q'])
        throw new Refused(`refused to commit: ${findings.slice(0, 5).map(f => `${f.file}${f.line ? `:${f.line}` : ''} (${f.kind})`).join('; ')}${findings.length > 5 ? ` and ${findings.length - 5} more` : ''} — files left unstaged`, { findings })
      }
      git(root, ['commit', '-q', '-F', '-'], checkpoint ? checkpointMessage(task) : messageFor({ task, stage, result }))
      sha = git(root, ['rev-parse', 'HEAD'])
    }
    return { committed: !!sha, sha, files: staged.length }
  }, { waitMs: checkpoint ? 0 : waitMs })
  if (out.busy) return { ok: true, skipped: `another publish holds the lock (pid ${out.holder})`, committed: false }
  if (checkpoint && !out.committed) return { ok: true, committed: false, pushed: false }
  const pushed = push(root, git, branch)
  let pr = null
  try { pr = ensurePr(root, gh, { branch, base, task }) } catch (e) { throw new Failed(`pushed, but the PR could not be opened: ${errText(e)}`) }
  saveState(root, { lastPublish: { stage: checkpoint ? 'checkpoint' : stage, sha: out.sha, at: now() } })
  let description = null
  if (!checkpoint) {
    const st2 = saveState(root, { log: [...(loadState(root).log ?? []), stageEntry({ stage, result, sha: out.sha, at: now() })].slice(-200) })
    try {
      gh(root, ['pr', 'edit', String(pr.number), '--body', bodyFor({ task, branch, text: st2.text ?? '', log: st2.log })])
      description = 'updated'
    } catch (e) { description = `not updated: ${errText(e)}` }
  }
  return { ok: true, branch, ...out, pushed, pr, ...(description ? { description } : {}) }
}

export function draft({ root = ROOT, git = realGit, gh = realGh } = {}) {
  const branch = requireFactoryBranch(root, git)
  const pr = findPr(root, gh, branch)
  if (!pr) return { ok: true, pr: null, draft: null }
  if (!pr.isDraft) gh(root, ['pr', 'ready', String(pr.number), '--undo'])
  return { ok: true, pr: { number: pr.number, url: pr.url }, draft: true }
}

export function ready({ root = ROOT, tree, git = realGit, gh = realGh, current = () => snapshot(root) } = {}) {
  if (!TREE_RE.test(tree ?? '')) throw Object.assign(new Error('--tree must be the tree id closeout verified'), { code: 2 })
  const branch = requireFactoryBranch(root, git)
  const now = current()
  if (now !== tree) {
    const changed = git(root, ['diff', '--name-only', tree, now]).split('\n').filter(Boolean)
    const outside = changed.filter(p => !READY_DOC_PATHS.some(d => (d.endsWith('/') ? p.startsWith(d) : p === d)))
    if (outside.length) throw new Refused(`the tree changed after closeout VERIFIED it: ${outside.slice(0, 5).join(', ')}${outside.length > 5 ? ` and ${outside.length - 5} more` : ''}`)
  }
  if (git(root, ['status', '--porcelain']).trim()) throw new Refused('uncommitted changes: publish the closeout stage before marking the PR ready')
  if (upstreamAhead(root, git) !== 0) throw new Refused('the branch is not pushed: publish the closeout stage before marking the PR ready')
  const pr = findPr(root, gh, branch)
  if (!pr) throw new Failed('no open PR for this branch')
  if (pr.isDraft) gh(root, ['pr', 'ready', String(pr.number)])
  return { ok: true, ready: true, pr: { number: pr.number, url: pr.url } }
}

// The checkpoint timer. tick() returns the checkpoint result; exits after idleExit ticks without a commit, or when
// the checkout leaves the factory branch.
export async function watchLoop({ tick, sleep, everyMs, idleExit = 3, onBranch }) {
  let idle = 0
  let ticks = 0
  for (;;) {
    await sleep(everyMs)
    if (!onBranch()) return { stopped: 'left the factory branch', ticks }
    ticks++
    let r
    try { r = tick() } catch (e) { r = { ok: false, error: errText(e) } }
    idle = r?.committed ? 0 : idle + 1
    if (idle >= idleExit) return { stopped: `idle for ${idle} ticks`, ticks }
  }
}

function detachWatch(root, everySec, idleExit) {
  const st = loadState(root)
  if (alive(st.watchPid)) return { ok: true, already: true, pid: st.watchPid }
  mkdirSync(stateDir(root), { recursive: true })
  const log = openSync(join(stateDir(root), 'watch.log'), 'a')
  const child = spawn(process.execPath, [SELF, 'watch', '--every', String(everySec), '--idle-exit', String(idleExit), '--root', root], { cwd: root, detached: true, stdio: ['ignore', log, log] })
  child.unref()
  closeSync(log)
  saveState(root, { watchPid: child.pid })
  return { ok: true, started: true, pid: child.pid, everySec, idleExit }
}

export function status({ root = ROOT, git = realGit } = {}) {
  const st = loadState(root)
  return { ok: true, branch: currentBranch(root, git) || null, state: st, watching: alive(st.watchPid) }
}

if (process.argv[1] === SELF) {
  const [cmd, ...rest] = process.argv.slice(2)
  const print = r => console.log(JSON.stringify(r))
  const usage = msg => Object.assign(new Error(msg), { code: 2 })
  try {
    // Every option is read (and validated) before anything runs.
    const o = optionReader(rest)
    const opt = { root: resolve(o.get('--root', ROOT)), task: o.get('--task'), base: o.get('--base', 'main'), stage: o.get('--stage'), b64: o.get('--b64', ''), tree: o.get('--tree'), textB64: o.get('--text-b64', ''), every: o.get('--every', '1800'), idleExit: o.get('--idle-exit', '3'), detach: o.flag('--detach') }
    if (o.unknown().length) throw usage(`unknown option(s): ${o.unknown().join(' ')}`)
    const { root } = opt
    let r
    if (cmd === 'start') {
      const text = opt.textB64 ? Buffer.from(opt.textB64, 'base64').toString('utf8') : ''
      r = start({ root, task: opt.task ?? (text ? slugify(text) : undefined), base: opt.base, text })
    }
    else if (cmd === 'commit') {
      let result = {}
      if (opt.b64) { try { result = JSON.parse(Buffer.from(opt.b64, 'base64').toString('utf8')) } catch { throw usage('--b64 is not base64 JSON') } }
      r = commit({ root, stage: opt.stage, result })
    } else if (cmd === 'checkpoint') r = commit({ root, checkpoint: true })
    else if (cmd === 'draft') r = draft({ root })
    else if (cmd === 'ready') r = ready({ root, tree: opt.tree })
    else if (cmd === 'status') r = status({ root })
    else if (cmd === 'stop-watch') {
      const st = loadState(root)
      const was = alive(st.watchPid)
      if (was) process.kill(st.watchPid, 'SIGTERM')
      saveState(root, { watchPid: null })
      r = { ok: true, stopped: was }
    } else if (cmd === 'watch') {
      let everySec, idleExit
      try { everySec = intOpt(opt.every, '--every', 1, 86400); idleExit = intOpt(opt.idleExit, '--idle-exit', 1, 1000) } catch (e) { throw usage(e.message) }
      if (opt.detach) r = detachWatch(root, everySec, idleExit)
      else {
        r = await watchLoop({
          everyMs: everySec * 1000, idleExit,
          sleep: ms => new Promise(res => setTimeout(res, ms)),
          onBranch: () => BRANCH_RE.test(currentBranch(root, realGit)),
          tick: () => { const t = commit({ root, checkpoint: true }); console.log(`${new Date().toISOString()} ${JSON.stringify(t)}`); return t },
        })
        saveState(root, { watchPid: null })
      }
    } else throw usage('usage: publish.mjs start|commit|checkpoint|draft|ready|watch|stop-watch|status (see the header)')
    print(r)
    process.exit(0)
  } catch (e) {
    const code = [1, 2, 3, 4].includes(e.code) ? e.code : 1
    print({ ok: false, code, reason: e.message, ...(e.extra ?? {}) })
    process.exit(code)
  }
}
