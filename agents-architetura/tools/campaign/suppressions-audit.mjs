#!/usr/bin/env node
// Audits every campaign suppressions file: each entry justified, each ruling valid (see suppressions.mjs).
//   node agents-architetura/tools/campaign/suppressions-audit.mjs [--root DIR] [--json]
// Exit: 0 ok · 1 violations · 5 not applicable (no suppressions files).
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { auditAll } from './suppressions.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const argv = process.argv.slice(2)
const i = argv.indexOf('--root')
const r = auditAll(resolve(i >= 0 ? argv[i + 1] : ROOT))
if (argv.includes('--json')) console.log(JSON.stringify(r))
else {
  for (const v of r.violations) console.log(`VIOLATION ${v}`)
  console.log(`suppressions: ${r.entries} entries in ${r.files} files, ${r.violations.length} violations`)
}
process.exit(!r.files ? 5 : r.violations.length ? 1 : 0)
