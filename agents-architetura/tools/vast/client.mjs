// Vast.ai client adapters. The real adapter shells out to the official `vastai` CLI (pip install vastai; the API key
// lives in the human's environment, never in the repo). The fake adapter replays a JSON state file so every guard in
// vast.mjs is tested without spending money. The real adapter's command lines follow the vastai CLI and have NOT been
// exercised against the live API from this repository yet — run `vast.mjs doctor` before a first paid experiment.
import { spawnSync } from 'node:child_process'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'

const sh = (args, opts = {}) => {
  const r = spawnSync('vastai', args, { encoding: 'utf8', timeout: opts.timeoutSec ? opts.timeoutSec * 1000 : 120000 })
  if (r.error?.code === 'ENOENT') throw new Error('the vastai CLI is not installed (pip install vastai) or not on PATH')
  return { code: r.status ?? 1, out: `${r.stdout ?? ''}${r.stderr ?? ''}` }
}
const json = (args, what) => {
  const r = sh([...args, '--raw'])
  if (r.code !== 0) throw new Error(`vastai ${what} failed: ${r.out.slice(-500)}`)
  try { return JSON.parse(r.out) } catch { throw new Error(`vastai ${what} returned non-JSON: ${r.out.slice(0, 300)}`) }
}

export function realClient() {
  return {
    kind: 'vastai',
    searchOffers: query => json(['search', 'offers', query, '-o', 'dph_total'], 'search offers'),
    createInstance: (offerId, { image, diskGb, label }) => json(['create', 'instance', String(offerId), '--image', image, '--disk', String(diskGb), '--label', label, '--ssh', '--direct'], 'create instance'),
    listInstances: () => json(['show', 'instances'], 'show instances'),
    exec: (id, cmd, timeoutSec = 600) => {
      const url = sh(['ssh-url', String(id)])
      const m = url.out.trim().match(/^ssh:\/\/(\S+?)@([\w.-]+):(\d+)/)
      if (!m) return { code: 255, out: `no ssh url for ${id}: ${url.out.slice(0, 200)}` }
      const r = spawnSync('ssh', ['-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=20', '-p', m[3], `${m[1]}@${m[2]}`, cmd], { encoding: 'utf8', timeout: timeoutSec * 1000 })
      return { code: r.status ?? 255, out: `${r.stdout ?? ''}${r.stderr ?? ''}` }
    },
    stopInstance: id => sh(['stop', 'instance', String(id)]),
    destroyInstance: id => sh(['destroy', 'instance', String(id)]),
  }
}

// Fake: state file {offers: [...], instances: [...], exec: {"<id>": {"<cmd prefix>": {code, out}}}, nextId}
export function fakeClient(path) {
  const load = () => JSON.parse(readFileSync(path, 'utf8'))
  const save = s => writeFileSync(path, JSON.stringify(s, null, 2))
  return {
    kind: 'fake',
    searchOffers: () => load().offers ?? [],
    createInstance: (offerId, { label }) => {
      const s = load()
      const offer = (s.offers ?? []).find(o => String(o.id) === String(offerId))
      const id = s.nextId ?? 1000
      s.nextId = id + 1
      s.instances = [...(s.instances ?? []), { id, label, actual_status: 'running', dph_total: offer?.dph_total ?? 0, gpu_name: offer?.gpu_name }]
      save(s)
      return { success: true, new_contract: id }
    },
    listInstances: () => load().instances ?? [],
    exec: (id, cmd) => {
      const table = load().exec?.[String(id)] ?? load().exec?.['*'] ?? {}
      const key = Object.keys(table).sort((a, b) => b.length - a.length).find(k => cmd.startsWith(k) || cmd.includes(k))
      return key ? table[key] : { code: 0, out: '' }
    },
    stopInstance: id => { const s = load(); s.instances = s.instances.map(i => (String(i.id) === String(id) ? { ...i, actual_status: 'stopped' } : i)); save(s); return { code: 0, out: '' } },
    destroyInstance: id => { const s = load(); s.instances = s.instances.filter(i => String(i.id) !== String(id)); s.destroyed = [...(s.destroyed ?? []), id]; save(s); return { code: 0, out: '' } },
  }
}

export const clientFromEnv = (env = process.env) => (env.FACTORY_VAST_FAKE && existsSync(env.FACTORY_VAST_FAKE) ? fakeClient(env.FACTORY_VAST_FAKE) : realClient())
