// Report parsers for campaigns. Every parser turns raw tool output into normalized reports
//   {tool, kind, signature, frames: [{fn, file, line}], location, message}
// The signature is the dedupe key: kind + the top in-repo frames (function + file; line numbers and addresses
// dropped, so an unrelated edit above a bug does not split its family). Two-stack reports (data races) sort
// their stacks, so the order in which the two threads were reported does not matter.
import { createHash } from 'node:crypto'

const FOREIGN = /site-packages|dist-packages|\/lib\/python\d|<frozen|\/rustc\/|\.cargo\/registry|\/library\/(std|core|alloc)\/|^\/usr\/|\/usr\/lib|libc\.so|libpthread|libsystem_|tsan_|asan_|ubsan_|sanitizer_common|\/llvm\/|node:internal/
const TOP = 3

export const shortHash = s => createHash('sha256').update(s).digest('hex').slice(0, 8)

// Repo-relative where possible: strip everything up to a known top-level source directory.
export function relPath(p) {
  const s = String(p ?? '').replace(/\\/g, '/')
  const m = s.match(/(?:^|\/)((?:src|tests|crates|python|lib|bench|tools)\/.*)$/)
  return m ? m[1] : s.split('/').slice(-3).join('/')
}

const normMessage = m => String(m ?? '').replace(/0x[0-9a-fA-F]+/g, '<addr>').replace(/\b\d+\b/g, 'N').replace(/\s+/g, ' ').trim().slice(0, 160)
const inRepo = f => f.file && !FOREIGN.test(f.rawFile ?? f.file)

function topFrames(frames) {
  const own = frames.filter(inRepo)
  return (own.length ? own : frames).slice(0, TOP)
}
const frameKey = fs => fs.map(f => `${f.fn}@${f.file}`).join(';')

function report(tool, kind, stacks, message) {
  const tops = stacks.map(topFrames).filter(s => s.length)
  const keys = tops.map(frameKey).sort()
  const signature = `${kind}|${keys.join(' <-> ') || normMessage(message)}`
  const first = tops[0]?.[0]
  return { tool, kind, signature, frames: tops.flat(), location: first ? `${first.file}:${first.line ?? '?'}` : null, message: String(message ?? '').trim().slice(0, 500) }
}

// ---- clang/Rust sanitizers (TSAN, ASAN, LSAN share the frame format) ----
const SAN_FRAME = /^\s*#\d+\s+(?:0x[0-9a-fA-F]+\s+in\s+)?(.+?)\s+((?:\/|[A-Za-z]:|\.{0,2}\/?)[^\s:()]+\.[A-Za-z0-9]+)(?::(\d+))?(?::\d+)?(?:\s+\(.*\))?\s*$/
const SAN_FRAME_NOFILE = /^\s*#\d+\s+(?:0x[0-9a-fA-F]+\s+in\s+)?(\S+)/

function sanFrames(lines) {
  const stacks = []
  let cur = null
  for (const line of lines) {
    if (/^\s*#\d+\s/.test(line)) {
      if (!cur) { cur = []; stacks.push(cur) }
      const m = line.match(SAN_FRAME)
      if (m) cur.push({ fn: m[1].replace(/\(.*\)$/, ''), file: relPath(m[2]), rawFile: m[2], line: m[3] ? Number(m[3]) : null })
      else { const n = line.match(SAN_FRAME_NOFILE); if (n) cur.push({ fn: n[1], file: null, rawFile: line, line: null }) }
    } else if (line.trim()) cur = null
  }
  return stacks
}

export function parseSanitizer(text) {
  const out = []
  const lines = String(text).split('\n')
  for (let i = 0; i < lines.length; i++) {
    let m = lines[i].match(/WARNING: ThreadSanitizer: ([\w -]+?)(?: \(pid=\d+\))?\s*$/)
    if (m) {
      const block = []
      for (let j = i + 1; j < lines.length && !/^SUMMARY: ThreadSanitizer|^={10,}/.test(lines[j]); j++) block.push(lines[j])
      const stacks = sanFrames(block)
      const kind = `tsan-${m[1].trim().replace(/\s+/g, '-')}`
      out.push(report('tsan', kind, m[1].includes('race') ? stacks.slice(0, 2) : stacks.slice(0, 1), m[1]))
      continue
    }
    m = lines[i].match(/ERROR: (AddressSanitizer|LeakSanitizer|MemorySanitizer): ([\w-]+(?: [\w-]+)*)/)
    if (m) {
      const block = []
      for (let j = i + 1; j < lines.length && !/^SUMMARY: /.test(lines[j]); j++) block.push(lines[j])
      const tool = m[1] === 'AddressSanitizer' ? 'asan' : m[1] === 'LeakSanitizer' ? 'lsan' : 'msan'
      const kind = `${tool}-${m[2].split(' ')[0]}`
      out.push(report(tool, kind, sanFrames(block).slice(0, 1), lines[i]))
      continue
    }
    m = lines[i].match(/^(.+?):(\d+):\d+: runtime error: (.+)$/)
    if (m) out.push({ ...report('ubsan', 'ubsan', [[{ fn: '<ub>', file: relPath(m[1]), rawFile: m[1], line: Number(m[2]) }]], m[3]), signature: `ubsan|${relPath(m[1])}|${normMessage(m[3])}` })
  }
  return out
}

// ---- Python tracebacks ----
export function parsePython(text) {
  const out = []
  const lines = String(text).split('\n')
  for (let i = 0; i < lines.length; i++) {
    if (!/^\s*Traceback \(most recent call last\):/.test(lines[i])) continue
    const indent = lines[i].match(/^\s*/)[0].length
    const frames = []
    let j = i + 1
    for (; j < lines.length; j++) {
      const f = lines[j].match(/^\s*File "(.+?)", line (\d+), in (.+)$/)
      if (f) { frames.push({ fn: f[3].trim(), file: relPath(f[1]), rawFile: f[1], line: Number(f[2]) }); continue }
      if (lines[j].match(/^\s*/)[0].length <= indent && lines[j].trim()) break
    }
    const exc = (lines[j] ?? '').trim()
    const type = exc.match(/^([\w.]+)(?::|$)/)?.[1] ?? 'Exception'
    out.push(report('python', `py-${type.split('.').pop()}`, [frames.reverse()], exc))
    i = j
  }
  return out
}

// ---- Rust panics and Miri ----
export function parseRust(text) {
  const out = []
  const t = String(text)
  for (const m of t.matchAll(/thread '([^']*)' panicked at (?:'([^']*)', )?([^\s:]+\.rs):(\d+):\d+:?\n?(.*)/g)) {
    const file = relPath(m[3])
    const msg = m[2] ?? m[5] ?? ''
    out.push({ ...report('rust', 'rust-panic', [[{ fn: '<panic>', file, rawFile: m[3], line: Number(m[4]) }]], msg), signature: `rust-panic|${file}|${normMessage(msg)}` })
  }
  for (const m of t.matchAll(/error: Undefined Behavior: (.+)\n(?:[\s\S]*?)-->\s+([^\s:]+):(\d+):\d+/g)) {
    const file = relPath(m[2])
    out.push({ ...report('miri', 'miri-ub', [[{ fn: '<ub>', file, rawFile: m[2], line: Number(m[3]) }]], m[1]), signature: `miri-ub|${file}|${normMessage(m[1])}` })
  }
  return out
}

// ---- CUDA compute-sanitizer ----
export function parseComputeSanitizer(text) {
  const out = []
  const lines = String(text).split('\n')
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^=+\s+(Race reported.*|Invalid __\w+__ (?:read|write|atomic) of size \d+ bytes|Uninitialized __global__ memory read.*|Barrier error.*|Program hit .+)$/)
    if (!m) continue
    let kernel = null
    for (let j = i + 1; j < Math.min(lines.length, i + 8) && !kernel; j++) kernel = lines[j].match(/^=+\s+at\s+(?:0x[0-9a-f]+\s+in\s+)?([^\s(]+)/)?.[1] ?? null
    const kind = `cuda-${m[1].split(/\s+/).slice(0, 2).join('-').toLowerCase().replace(/[^a-z-_]/g, '')}`
    out.push({ ...report('compute-sanitizer', kind, [], m[1]), signature: `${kind}|${kernel ?? normMessage(m[1])}`, location: kernel })
  }
  return out
}

// ---- pytest junit XML ----
const unxml = s => String(s).replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, '&')

export function parseJunit(xml) {
  const out = []
  for (const tc of String(xml).matchAll(/<testcase\b([^>]*?)(?:\/>|>([\s\S]*?)<\/testcase>)/g)) {
    const attrs = Object.fromEntries([...tc[1].matchAll(/(\w+)="([^"]*)"/g)].map(a => [a[1], unxml(a[2])]))
    const body = tc[2] ?? ''
    const f = body.match(/<(failure|error)\b([^>]*)>([\s\S]*?)<\/\1>|<(failure|error)\b([^>]*)\/>/)
    if (!f) continue
    const fattrs = Object.fromEntries([...(f[2] ?? f[5] ?? '').matchAll(/(\w+)="([^"]*)"/g)].map(a => [a[1], unxml(a[2])]))
    const detail = unxml(f[3] ?? '')
    const id = `${attrs.classname ?? ''}::${attrs.name ?? ''}`
    const type = (fattrs.message ?? detail).match(/^\s*([\w.]+(?:Error|Exception|Failed|Exit))\b/)?.[1] ?? (f[1] ?? f[4])
    const kind = /Falsifying example/.test(detail) ? 'property-failure' : (f[1] ?? f[4]) === 'error' ? 'test-error' : 'test-failure'
    out.push({ tool: 'pytest', kind, signature: `${kind}|${id}|${type}`, frames: [], location: attrs.file ? `${attrs.file}:${attrs.line ?? '?'}` : id, message: (fattrs.message ?? detail).trim().slice(0, 500) })
  }
  return out
}

// ---- static analyzers: SARIF (semgrep, bandit, ruff, codeql, gitleaks, trivy, osv-scanner, clippy-sarif) ----
export function parseSarif(json) {
  let doc
  try { doc = typeof json === 'string' ? JSON.parse(json) : json } catch { return [] }
  const out = []
  for (const run of doc?.runs ?? []) {
    const tool = String(run?.tool?.driver?.name ?? 'sarif').toLowerCase().replace(/[^a-z0-9-]+/g, '-')
    for (const r of run?.results ?? []) {
      const loc = r?.locations?.[0]?.physicalLocation
      const file = relPath(loc?.artifactLocation?.uri ?? '')
      const line = loc?.region?.startLine ?? null
      const rule = String(r?.ruleId ?? 'rule')
      const message = r?.message?.text ?? ''
      out.push({ tool, kind: `scan-${tool}`, signature: `scan|${tool}|${rule}|${file}|${normMessage(message).slice(0, 80)}`, frames: [], location: `${file}:${line ?? '?'}`, message: `[${r?.level ?? 'warning'}] ${rule}: ${message}`.slice(0, 500) })
    }
  }
  return out
}

// ---- mypy / pyright-style text diagnostics: file:line[:col]: error: message [code] ----
export function parseDiagnostics(text, tool = 'mypy') {
  const out = []
  for (const m of String(text).matchAll(/^(.+?\.pyi?):(\d+):(?:\d+:)?\s*error:\s*(.+?)(?:\s+\[([\w-]+)\])?$/gm)) {
    const file = relPath(m[1])
    const code = m[4] ?? 'error'
    out.push({ tool, kind: `scan-${tool}`, signature: `scan|${tool}|${code}|${file}|${normMessage(m[3]).slice(0, 80)}`, frames: [], location: `${file}:${m[2]}`, message: m[3].slice(0, 500) })
  }
  return out
}

// ---- dependency auditors: pip-audit -f json, cargo deny --format json (JSON lines) ----
export function parsePipAudit(json) {
  let doc
  try { doc = JSON.parse(json) } catch { return [] }
  return (doc?.dependencies ?? []).flatMap(d => (d.vulns ?? []).map(v => ({
    tool: 'pip-audit', kind: 'scan-pip-audit', signature: `scan|pip-audit|${v.id}|${d.name}`, frames: [], location: `${d.name}==${d.version}`,
    message: `${v.id} in ${d.name} ${d.version}${(v.fix_versions ?? []).length ? ` (fixed in ${v.fix_versions.join(', ')})` : ''}`.slice(0, 500),
  })))
}
export function parseCargoDeny(text) {
  const out = []
  for (const line of String(text).split('\n')) {
    let d
    try { d = JSON.parse(line) } catch { continue }
    const f = d?.fields
    if (d?.type !== 'diagnostic' || !f || !['error', 'warning'].includes(f.severity)) continue
    out.push({ tool: 'cargo-deny', kind: 'scan-cargo-deny', signature: `scan|cargo-deny|${f.code ?? 'diag'}|${normMessage(f.message).slice(0, 80)}`, frames: [], location: f.code ?? null, message: String(f.message ?? '').slice(0, 500) })
  }
  return out
}

// All text formats at once (a campaign log mixes them freely).
export const parseText = text => [...parseSanitizer(text), ...parsePython(text), ...parseRust(text), ...parseComputeSanitizer(text)]

export function parseFile(name, content) {
  if (/\.xml$/.test(name) && content.includes('<testsuite')) return parseJunit(content)
  if (/\.sarif(\.json)?$/.test(name)) return parseSarif(content)
  if (/pip-audit\.json$/.test(name)) return parsePipAudit(content)
  if (/cargo-deny\.jsonl$/.test(name)) return parseCargoDeny(content)
  if (/\.diag\.txt$/.test(name)) return parseDiagnostics(content, name.split('/').pop().replace(/\.diag\.txt$/, ''))
  if (/summary\.json$/.test(name)) {
    try {
      const s = JSON.parse(content)
      const items = s.findings ?? s.failures ?? []
      return items.flatMap(f => {
        const inner = parseText(f.tail ?? '')
        const outcome = String(f.outcome ?? 'failure').toLowerCase()
        if (inner.length) return inner.map(r => ({ ...r, kind: `${outcome}:${r.kind}`, signature: `${outcome}:${r.signature}` }))
        return [{ tool: 'driver', kind: outcome, signature: `${outcome}|${shortHash(String(s.cmd ?? ''))}|${normMessage(String(f.tail ?? '').split('\n').filter(Boolean).pop())}`, frames: [], location: f.log ?? null, message: String(f.tail ?? '').slice(-500) }]
      })
    } catch { return [] }
  }
  return parseText(content)
}
