// Project context shared by every workflow. Requires the runtime global
// agent() for sizeGate().
const PROJECT = 'efficient distributed training of neural networks over wide-area networks on heterogeneous hardware'
const DESIGN_DIR = 'docs/design'
const BRIEF = `${DESIGN_DIR}/BRIEF.md`
const DESIGN = `${DESIGN_DIR}/DESIGN.md`
const DESIGN_HISTORY = `${DESIGN_DIR}/DESIGN-history.md`
const DESIGN_CAP = 60000

// Must equal the rung ids in agents-architetura/tools/gates/rungs.json (pinned by agents-architetura/tools/wf/test/project.test.mjs).
const RUNG_IDS = ['factory', 'frozen', 'build', 'syntax', 'fmt', 'lint', 'types', 'discipline', 'invariants', 'claims', 'unit', 'component', 'integration', 'multiworker', 'multinode', 'oracle', 'identity', 'chaos', 'wan', 'matrix', 'rest', 'full', 'static', 'security', 'fuzz', 'bench', 'redteam', 'mutation']

// Agent roles (.claude/agents/<type>.md). The researcher decides what and why; the coder implements; the
// proposer designs fixes without editing; the reviewer attacks; the verifier measures; the red team breaks.
const ROLE = {
  researcher: 'wan-researcher',
  architect: 'wan-architect',
  coder: 'wan-coder',
  proposer: 'wan-proposer',
  reviewer: 'wan-reviewer',
  verifier: 'wan-verifier',
  redteam: 'wan-redteam',
  readonly: 'wan-readonly',
  security: 'wan-security',
  vastai: 'wan-vastai',
}
const LESSONS = 'agents-architetura/docs/LESSONS.md'
const runDir = m => `.factory/runs/${m}`
const journalPath = (m, id) => `${runDir(m)}/tasks/${id}.md`

const NO_SLOW = `
HARD RULES (other agents share this working tree):
- Do NOT run git. Do NOT run builds, tests, benchmarks or any slow command unless your prompt says you run ALONE.
- WRITE only inside the owned paths your prompt lists. Read anything in the repo.
- Generated files (.claude/workflows/*.js) are never edited by hand: edit agents-architetura/tools/wf/src or agents-architetura/tools/wf/lib and rebuild.
- Honesty: partial is fine, fabricated green is not. Never weaken a test, invariant or gate to make something pass.`

// Security from first principles at the smallest speed cost (the human's rule, 2026-10-08). Spliced into every
// prompt that threat-models, audits or reviews security, so a mitigation is chosen by risk removed per unit of cost.
const SECURITY_COST = `
SECURITY FROM FIRST PRINCIPLES, AT THE SMALLEST SPEED COST:
- Rank mitigations by risk removed per unit of runtime cost. Cheapest first: remove the dangerous capability (no code path
  from bytes, no pickle/torch.load of untrusted data), validate types and bounds at the boundary, least privilege, constant-time
  secret comparison — these cost ~0 at runtime.
- Amortize: authenticate per connection, not per frame; derive session keys once; sign nothing per frame that an AEAD already
  authenticates.
- Keep per-byte work (encryption, hashing) off the critical path: sender/reader threads overlapped with compute, hardware
  primitives (AES-GCM with AES-NI/ARMv8 crypto; ChaCha20-Poly1305 without them).
- State each mitigation's cost model (per byte, per frame, per connection, per step) and its expected share of the step time from
  ${DESIGN} (its performance model), labeled MEASURED (command, bench rung) or HYPOTHESIS (with the experiment that measures it).
- A mitigation above 1% of the measured step time needs an ADR that weighs it against the threat. Never weaken a security
  property for speed silently, and never add a layer that costs time without adding a property another layer lacks.`
// The human's hardware switch (agents-architetura/tools/factory/hardware.json). Every agent plans and verifies inside it;
// vast.mjs enforces the same switch in code, so a prompt that ignores it still cannot rent anything.
const HARDWARE = `
HARDWARE MODE: read agents-architetura/tools/factory/hardware.json before you plan, measure or promise anything (anything but
"gpu": true means false).
- gpu false (the default): there is no rented GPU, no second machine and no Vast.ai. Validate correctness by local simulation on
  this machine's CPU: multi-process localhost clusters, emulated WAN links (latency, bandwidth, loss, partitions), heterogeneous
  fleets made by slowing peers down, the byte-exact oracle and identity gates (the reference runs on the CPU). Never choose an
  objective, acceptance criterion, exit gate, pinned rung, campaign or experiment step that needs a rented GPU, a second host or
  paid cloud: those are N/A with the reason recorded, never pinned, never faked. Label performance numbers of the simulated fleet
  SIMULATED next to the performance model's prediction; real-hardware validation is recorded as pending, never as passed.
- This Mac's own hardware is measured for experiments (the human's decision, 2026-10-09). Its CPU: AES-GCM and ChaCha20-Poly1305
  throughput per core, copies and passes through host memory, codec and checksum costs. Its GPU, through MLX (uv run --with mlx,
  outside the project lock) or torch MPS: bf16/fp8 matmul and cast throughput, unified-memory bandwidth, the run-to-run
  determinism of its kernels. Label those numbers MEASURED (local Mac: chip, the command) and keep them apart from SIMULATED
  ones. They feed the cost models; they never replace real-hardware validation (M1), and no gate depends on the Mac's GPU (CI
  runs on Linux without one).
- gpu true: GPU work runs only on Vast.ai through agents-architetura/tools/vast/vast.mjs, only for an experiment plan and budget a
  human granted. Never handle the Vast.ai API key and never change the hardware switch — both belong to the human.`
// The agent that finds an error fixes it (the human's rule, 2026-10-09), with no voters in between. What the votes used to
// catch is a rule every fixer carries, and the review at the end and the closeout's guarded gates check it.
const NO_MASKING = `Fix causes, never symptoms. Never weaken, skip, delete or loosen a test, assert, gate, timeout or tolerance;
never retry, sleep or reorder a race away; never touch agents-architetura/ or docs/design/. A test changes only when the test
itself is wrong, and you say why in your report. The product rule holds: one differing bit is a bug.`
// Tests share the machine through machine-wide slots (machine-lock.mjs): two builds in two checkouts ran their tests
// side by side and the 16 GB Mac panicked (2026-10-09). Every agent's test command goes through a slot; the ladder takes
// its own; a heavy run (a pinned entry, a heavy rung, a campaign) takes the whole machine.
const TEST_SLOTS = `Run every test command (pytest, a test script) through a machine-wide test slot:
node agents-architetura/tools/verify/machine-lock.mjs run --slot --label <your task or lens> --wait 3600 -- <command>
(the ladder takes its own slots; never run tests outside a slot). A slot run is killed after 1200 s (--timeout) and exits 124:
that is a hang (a deadlock, a wait with no timeout), never a slow test. Find and fix the hang; do not raise --timeout to pass.
When a mutation or fault-injection run is expected to hang, give it a short --timeout and count exit 124 as detected.`
const common = (extra = '') => `Repo: the repository root (your current working directory).
Project: ${PROJECT}. Human brief: ${BRIEF}. Design of record: ${DESIGN}. Factory rules: ./CLAUDE.md.
Shared lessons (read first, do not repeat them): ${LESSONS}.
${NO_SLOW}
${HARDWARE}
${TEST_SLOTS}
${extra}`

// Single-objective size enforcer. Multi-objective agents (fix findings AND stay
// small) reliably sacrifice the size rule; this one has nothing else to optimize.
const sizeGate = (file, historyFile, cap, label, phaseTitle) => agent(`Repo: the repository root (your current working directory).
Single task, nothing else. Run: wc -c ${file}
If it is <= ${cap} bytes: change NOTHING and report "under cap: <bytes>".
If it is over: compress ${file} to AT MOST ${cap} bytes. You may write ONLY ${file} and ${historyFile}.
- MOVE (never delete) review-resolution logs, refutations, revision history and worked examples to ${historyFile}
  (leave a one-line pointer where they were).
- KEEP every normative requirement intact: layouts, exact signatures, numbered invariants, lock orders,
  owned paths, the workstreams block, ordered task lists, decisions-with-rationale one-liners.
- COMPRESS the rest: tables over prose, dedupe, drop background the surveys already cover.
- Verify with wc -c before finishing; iterate until under cap. No git, no builds.`,
  { label, phase: phaseTitle, schema: RESULT })

const SPEC_CAP = 40000
const specPath = key => `${DESIGN_DIR}/SPEC-${key}.md`
const specHistoryPath = key => `${DESIGN_DIR}/SPEC-${key}-history.md`
// Files no workstream owns; only an "integrate" task (or a solo integrator) edits them.
const HOT_FILES = ['pyproject.toml', 'uv.lock', 'Cargo.toml', 'Cargo.lock', 'package.json', 'package-lock.json', 'agents-architetura/tools/gates/rungs.json', 'CLAUDE.md',
  'agents-architetura/tools/campaign/campaigns.json', 'agents-architetura/tools/campaign/scanners.json', 'agents-architetura/tools/verify/matrix.json', 'agents-architetura/tools/verify/identity.json', 'agents-architetura/tools/gates/mutation.json']
const INTEGRATOR = 'integrate'

// The agent factory itself: never product, never owned by a workstream (the integrator may still apply INTEGRATE entries
// to the factory config files listed in HOT_FILES).
const FACTORY_DIRS = ['agents-architetura/', '.claude/', '.github/']

// Directories owned by the verification harness (wan-prep, wan-redteam); no workstream may claim them.
const HARNESS_DIRS = ['tests/oracle/', 'tests/chaos/', 'tests/wan/', 'tests/multiworker/', 'tests/multinode/', 'tests/fuzz/', 'tests/redteam/', 'tests/semantics/', 'tests/scaling/', 'tests/fault-matrix/', 'bench/']

// Human approval (agents-architetura/tools/factory/approve.mjs): a decision document is approved only while a human grant exists for its
// exact current content. Silence is not approval; an edited document needs a new grant. Requires schemas
// (APPROVAL_CHECK) and hygiene (clean).
const approvalCommand = docs => `node agents-architetura/tools/factory/approve.mjs check ${docs.join(' ')} --json`
const checkApprovals = (docs, label, phaseTitle) => agent(common('You run ALONE — you MAY run this one command. Change nothing.') + `
Run exactly: ${approvalCommand(docs)}
Return the JSON it prints EXACTLY as printed. Never grant, edit or work around an approval — only a human can approve.`,
  { label, phase: phaseTitle, schema: APPROVAL_CHECK, agentType: ROLE.verifier })
const approvedAll = (r, docs) => r?.approved === true && docs.every(d => (r.docs ?? []).some(x => x.doc === d && x.approved === true))

// Every workflow after wan-design starts here.
const requireApprovedDesign = async phaseTitle => {
  const r = await checkApprovals([DESIGN], 'gate:approval', phaseTitle)
  if (!approvedAll(r, [DESIGN])) {
    const why = clean((r?.docs ?? [])[0]?.reason ?? 'no approval record', 200)
    throw new Error(`${DESIGN} is not approved by a human (${why}) — run /wan-approve in Claude Code, or in your own terminal: node agents-architetura/tools/factory/approve.mjs grant ${DESIGN}`)
  }
}

// Reads the design's workstreams block and checks disjoint ownership. Requires dag (validatePlan) and schemas (WORKSTREAMS).
const loadWorkstreams = async phaseTitle => {
  const extracted = await agent(common('READ-ONLY.') + `
Read ${DESIGN}. Find the block after "<!-- workstreams -->" and return its JSON array as "workstreams", verbatim
(do not fix, reorder or complete it). If the block is missing or not valid JSON, return an empty array.`,
    { label: 'extract:workstreams', phase: phaseTitle, schema: WORKSTREAMS, agentType: ROLE.readonly })
  // Keys are validated raw (validatePlan enforces the id grammar) — never normalized into validity.
  const ws = (extracted?.workstreams ?? []).map(w => ({ ...w, ownedPaths: Array.isArray(w.ownedPaths) ? w.ownedPaths : [] }))
  if (!ws.length) throw new Error(`${DESIGN} has no valid workstreams block — rerun wan-design`)
  const check = validatePlan([...ws.map(w => ({ id: w.key, deps: [], scope: w.ownedPaths })), { id: 'factory', deps: [], scope: FACTORY_DIRS }])
  if (!check.ok) throw new Error(`workstream ownership is invalid — rerun wan-design: ${check.errors.join('; ')}`)
  return ws
}

// Tests written before their implementation carry this marker (pytest xfail(strict=True) reason,
// Rust #[ignore = ...]); the task that delivers the invariant removes it; closeout refuses leftovers.
const AWAITING = 'awaiting implementation'

const milestoneArg = a => {
  const m = String(a?.milestone ?? '')
  // A milestone may carry one letter after its number when a granted ADR splits one off (M2b, ADR-0006).
  if (!/^M\d{1,3}[a-z]?$/.test(m)) throw new Error('args.milestone is required and must look like "M0" or "M2b"')
  return m
}
const handoutPath = m => `${DESIGN_DIR}/HANDOUT-${m}.md`
const planPath = m => `${DESIGN_DIR}/PLAN-${m}.json`
