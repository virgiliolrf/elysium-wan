#!/usr/bin/env node
// The human's hardware switch, in agents-architetura/tools/factory/hardware.json.
//   gpu false (the default): no GPU, no second machine, no Vast.ai. The factory validates everything by local
//     simulation, and vast.mjs refuses every account action except validate, status, monitor and stop.
//   gpu true: GPU experiments on Vast.ai are allowed, still only through vast.mjs and only for an experiment plan and
//     budget a human granted. Claude Code asks the human before this switch is turned on (.claude/settings.json).
// Anything but a literal `"gpu": true` reads as false, so a missing or damaged file can never start spending.
//
//   node agents-architetura/tools/factory/hardware.mjs show [--json]
//   node agents-architetura/tools/factory/hardware.mjs set gpu true|false
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

export const HARDWARE_FILE = 'agents-architetura/tools/factory/hardware.json'
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')

export function readHardware(root) {
  const path = join(root, HARDWARE_FILE)
  if (!existsSync(path)) return { gpu: false, reason: `${HARDWARE_FILE} does not exist` }
  let doc
  try { doc = JSON.parse(readFileSync(path, 'utf8')) } catch { return { gpu: false, reason: `${HARDWARE_FILE} is not valid JSON` } }
  if (doc?.gpu === true) return { gpu: true }
  return { gpu: false, reason: doc?.gpu === false ? 'gpu is false' : `"gpu" is ${JSON.stringify(doc?.gpu ?? null)}, not true` }
}

export function writeHardware(root, gpu) {
  if (typeof gpu !== 'boolean') throw new Error('gpu must be true or false')
  writeFileSync(join(root, HARDWARE_FILE), `${JSON.stringify({ gpu }, null, 2)}\n`)
  return readHardware(root)
}

export const describe = hw => hw.gpu
  ? 'gpu: true — GPU experiments on Vast.ai allowed, only through vast.mjs and only for a plan and budget a human granted'
  : `gpu: false (${hw.reason}) — simulation only: no GPU, no second machine, no Vast.ai`

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [cmd, key, value] = process.argv.slice(2)
  const show = hw => console.log(process.argv.includes('--json') ? JSON.stringify(hw) : describe(hw))
  if (cmd === 'show') show(readHardware(ROOT))
  else if (cmd === 'set' && key === 'gpu' && (value === 'true' || value === 'false')) show(writeHardware(ROOT, value === 'true'))
  else {
    console.error('usage: hardware.mjs show [--json] | set gpu true|false')
    process.exit(2)
  }
}
