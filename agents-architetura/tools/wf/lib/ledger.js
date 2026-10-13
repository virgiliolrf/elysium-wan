// Run-ledger helpers. Requires project (runDir). Pure JS UTF-8 + base64 so the workflow runtime needs no
// Buffer/btoa: the state travels to agents-architetura/tools/wf/ledger.mjs as one shell-safe token.
const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
const utf8Bytes = s => {
  const out = []
  for (const ch of String(s)) {
    const c = ch.codePointAt(0)
    if (c < 0x80) out.push(c)
    else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63))
    else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63))
    else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63))
  }
  return out
}
const base64 = s => {
  const b = utf8Bytes(s)
  let out = ''
  for (let i = 0; i < b.length; i += 3) {
    const n = (b[i] << 16) | ((b[i + 1] ?? 0) << 8) | (b[i + 2] ?? 0)
    out += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + (i + 1 < b.length ? B64[(n >> 6) & 63] : '=') + (i + 2 < b.length ? B64[n & 63] : '=')
  }
  return out
}
const ledgerCommand = (dir, payload) => `node agents-architetura/tools/wf/ledger.mjs write --dir ${dir} --b64 ${base64(JSON.stringify(payload))}`
