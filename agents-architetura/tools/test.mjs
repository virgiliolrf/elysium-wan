#!/usr/bin/env node
// Runs the factory's tests with a private TMPDIR that is removed afterwards, pass or fail. The tests build many
// throwaway git repositories under os.tmpdir(); without this they would accumulate in the system temp directory.
//
//   node agents-architetura/tools/test.mjs [extra node --test args...]
// Exit: node --test's exit code.
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const HERE = dirname(fileURLToPath(import.meta.url))
const scratch = mkdtempSync(join(tmpdir(), 'factory-tests-'))
let status = 1
try {
  const r = spawnSync(process.execPath, ['--test', ...process.argv.slice(2), join(HERE, '**', '*.test.mjs')], {
    stdio: 'inherit',
    // the factory's own tests take test locks in a private directory, never the machine-wide one real builds use
    env: { ...process.env, TMPDIR: scratch, TMP: scratch, TEMP: scratch, FACTORY_LOCK_DIR: join(scratch, 'locks') },
  })
  status = r.status ?? 1
} finally {
  rmSync(scratch, { recursive: true, force: true })
}
process.exit(status)
