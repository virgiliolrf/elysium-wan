# Factory Plan

> **Status:** implemented. This factory merges three sources:
> - the agent workflow of oven-sh/WebKit#249 and its verification depth (see [VERIFICATION.md](VERIFICATION.md));
> - the Elysium Autonomous Software Engineering System specification: human architecture approval, repository
>   analysis, threat model, ADRs, and local-first experiments with Vast.ai guarded in code
>   ([EXPERIMENTS.md](EXPERIMENTS.md));
> - this repository's own factory.
>
> Test status: 269 passing, 1 opt-in toolchain test skipped. Every workflow has a simulation test, and every gate and
> guard tool is tested against real processes.
>
> **Not yet done:**
> - the end-to-end self-test of the specification's item 35, deferred by the human to the end;
> - PLAN → CLOSEOUT of this version with real agents;
> - the native Claude Code workflow runtime;
> - the live Vast.ai API.
>
> Changing a workflow: adjust its simulation test first, then `node agents-architetura/tools/wf/build.mjs`, then commit.

**Goal:** an evidence-driven, architecture-aware, security-conscious and cost-conscious engineering system for
distributed training over WAN on heterogeneous hardware. The human is the architect, research director and final
decision maker. Agents research, understand, specify, design, implement, review, test, attack, benchmark and debug.
Nothing is done without measured evidence.

**Architecture:** deterministic JavaScript workflow scripts (`.claude/workflows/*.js`, generated from `agents-architetura/tools/wf/src`)
orchestrate a small set of role agents through the Claude Code workflow runtime. The script owns:
- control flow, scheduling and file ownership;
- votes, evidence checks and convergence;
- approvals and budgets.

The agents own judgement and code. Gate status, approvals, freezes, classifications and cloud spending come from
tools, never from an agent's claim.

---

## Agents: the smallest set, separated by capability

| Agent | Role | Can write |
|---|---|---|
| `wan-researcher` | Systems architect and PhD researcher: repository understanding, request specs, design, ADRs with first-principles cost models, experiment design, conflict re-evaluation | documents |
| `wan-coder` | Production engineer (Correctness > Reliability > Simplicity > Maintainability > Performance > Elegance) | its task's owned paths |
| `wan-proposer` | Minimal correct fixes, as snippets | nothing |
| `wan-reviewer` | Adversarial review: validated findings only, never "looks good" | nothing |
| `wan-security` | Security architect: threat model, diff review of high-risk tasks | nothing |
| `wan-verifier` | Runs commands, pastes outputs verbatim; keeps SYSTEM WORKED separate from TRAINING RESULT IS CORRECT | nothing (runs) |
| `wan-redteam` | Proves the implementation wrong, with tests only | `tests/redteam/` |
| `wan-vastai` | Runs approved experiments on Vast.ai, or stops spending | via `agents-architetura/tools/vast/vast.mjs` only |
| `wan-readonly` | Verbatim extraction for scripts | nothing |

The domain roles the specification lists are **specialists**, not agent types: scheduler, distributed training, WAN
networking, fault tolerance, performance, observability and training correctness. `wan-survey` creates a specialist
only when the code holds that responsibility, with file:line evidence and paths that the script checks exist. Tasks
touching those paths get the specialist's focus in their context and its lens as an extra reviewer. Today the
product does not exist, so there are zero specialists. That is the correct answer, not a gap.

## Flow

```
/wan-task  REQUEST → wan-survey (CURRENT_ARCHITECTURE, ARCHITECTURE_MAP, specialists, optional reference map)
           → request SPEC (15 sections, 2-lens review, frozen) → wan-design (surveys, first-principles draft,
           THREAT_MODEL by wan-security, 4-lens review, ownership, ADR-0001 with the decision block)
           → STOP: human grant (agents-architetura/tools/factory/approve.mjs)
/wan-task  (again) → wan-prep (frozen specs, verification seams, distributed matrix) → per milestone wan-cycle:
           survey refresh → handout (DAG with risk, no overlaps, frozen GATES) → implement (waves, focused context,
           per-task review + specialists + security for high risk, independent verification, ADR stop on major change)
           → corpus → audit → campaigns → fuzz → scanners → benchmark → red team → fix → closeout (VERIFIED or
           NOT VERIFIED, implementation report) → wan-scalebench → TASK report
/wan-adr         major change → research → ADR with amendments → review → STOP for the human
/wan-experiment  ADR → plan with predictions → local prototype → correctness/perf/failure → script classification
                 → promotion gate → (human-approved plan + budget) → Vast.ai via vast.mjs → expected vs observed
```

## Principles and how each is enforced

| Principle | Mechanism (script- or tool-enforced unless noted) |
|---|---|
| Understand before modifying | `wan-survey` runs first and at every milestone. The map comes from script facts (`inventory.mjs`). Cited paths must exist and every source directory must belong to a component. Fidelity and boundary reviewers loop over it |
| Research before inventing, first principles before complexity | The design and ADR require a derived cost model (`T_total = …`): which term each mechanism moves, and when the gain vanishes. Review lenses reject decorative math and appeal to authority (prompt + review) |
| Major architecture changes need explicit human approval | `approve.mjs`: grants are bound to the exact text (sha256), and editing voids them. `grant` refuses inside agent sessions and without a TTY. Settings deny agents the command and edits of `APPROVALS.json`. Every workflow checks the design grant first. Conflicts needing a major change open `wan-adr` and stop the run |
| Frozen specifications | `freeze.mjs`: the request spec and workstream specs are frozen; changes only by recorded amendment (`--by` a decision). The `frozen` rung fails on a silent change |
| Evidence before confidence | `TASK_RESULT.evidence` is required. The independent verifier runs every verify command. Gate JSON is pasted verbatim and tied to its tree id |
| Strict scope | Workstream ownership is validated. Overlapping concurrent tasks are a plan error (dependency or integration task required). A git scope audit runs around waves, the integrator, red team, corpus, prototypes and studies |
| Per-task adversarial review | 2 lenses, plus 1–2 specialist lenses, plus a `wan-security` diff review for `risk: high`. Only validated findings count; a missing reviewer fails the task |
| Never trust a finding blindly | Triage: confirmed → fix by vote; both refuters refute → rejected with evidence; refuters split or uncertain → a `needs-human` finding, which blocks VERIFIED |
| Local before paid cloud | `wan-experiment` classifies the local result by script. `vast.mjs launch` refuses without a promotable `LOCAL.json`, a complete checklist, and a human grant of the exact plan and budget |
| Never exceed the budget | Offers are priced before creation. Monitor pauses at `warnAtPct` and emergency-stops at 100%, on divergence, dead or stalled workers, or vanished instances. Stop preserves logs first and then destroys. Agents are denied `vastai create`/`launch` |
| Do not rationalize failed hypotheses | Classification is arithmetic: VALIDATED / PARTIALLY VALIDATED / REFUTED / INCONCLUSIVE. A refutation is recorded in LESSONS and sent back to first principles |
| Never declare success without verification | `pinned.mjs` runs the frozen gates on the current tree. VERIFIED / NOT VERIFIED; no "mostly" |

## What is built

**Workflows (21):**

| Workflow | Stage | Counterpart in oven-sh/WebKit#249 |
|---|---|---|
| `wan-task` | entry | none (a human sequenced the PR's workflows) |
| `wan-survey` | repository analysis | none (the spec's §1–2) |
| `wan-design` | spec, architecture, threat model | `aot-design`, plus the human-written `THREAD.md` |
| `wan-adr` | architecture decision | none (the spec's §7–9) |
| `wan-experiment` | local, then Vast.ai | none (the spec's §21–28) |
| `wan-prep` | specs, verification seams | `thread-prep`, `thread-specs2` |
| `wan-handout` | plan and frozen gates | `thread-ungil-spec` |
| `wan-implement` | implement, review, integrate, verify | `thread-implement`, `thread-ungil` |
| `wan-corpus` | corpus | `thread-corpus2` |
| `wan-audit` | security | `thread-cve-*` |
| `wan-campaign` | campaigns to zero | `thread-tsan` |
| `wan-fuzz` | fuzz | `thread-fuzz*` |
| `wan-scanners` | scanners | `thread-scanners` |
| `wan-redteam` | red team | classes + `thread-bughunter*` |
| `wan-fix` | fix | `thread-fix`, `thread-ab17*` |
| `wan-bughunt` | stubborn bugs | `thread-bughunter[1-4]` |
| `wan-cycle` | per milestone | none |
| `wan-closeout` | closeout | `thread-closeout`, `MEGA-RUN` |
| `wan-scalebench` | benchmark study | `thread-scalebench` |
| `wan-pr-feedback` | review bots | none |
| `wan-smoke` | runtime check | none |

**Tools:**
- `agents-architetura/tools/factory/`: `approve`, `freeze`, `inventory`.
- `agents-architetura/tools/gates/`:
  - the ladder (27 rungs: repeat, consecutive, on-demand, tree identity);
  - `pinned`, `discipline`, `invariants`, `claims`;
  - `mutation` and `mutate.py`;
  - `bench-gate` (with machine info) and `scaling-gate`;
  - `scalebench-check`, `tree`.
- `agents-architetura/tools/verify/`: `amplify`, `load`, `matrix`, `identity`, `machine-lock`.
- `agents-architetura/tools/campaign/`: `families`, `fuzz-run`, `scan`, `suppressions-audit`, `record`, `parse`.
- `agents-architetura/tools/vast/`: `vast`, `client`.
- `agents-architetura/tools/wf/`: `build`, `sim`, `ledger`, `scope-audit`.
- Libs: `hygiene · schemas · dag · loop · triage · project · gates · plan · fix · ledger · campaign · specialists · experiment`.

## How it was verified

| Check | Result |
|---|---|
| Simulation and tool tests | 269 passing, covering:<br>• every workflow's control flow, stops and failure paths, and role routing;<br>• the gate and guard tools against real processes: signals, timeouts, git worktrees, regression patches, real mutants, real report formats;<br>• human-approval guards: refused inside an agent session and without a TTY;<br>• Vast.ai guards against a fake client: approval, promotion, budget, pre-flight, pause, emergency stop, cleanup |
| Ladder vs real toolchains (`FACTORY_TOOLCHAIN_TEST=1`) | Real uv + cargo fixture: real defects turn the right rung red; absent suites are N/A |
| E2E run 1 (Sonnet, earlier version) | `wan-implement M0` plus nested `wan-fix`: ok |
| E2E run 2 (Haiku, earlier version, ~93 agent calls) | brief → design → prep → verify → `wan-fix`; found and fixed 5 factory defects; stopped by the human before PLAN → CLOSEOUT |

**Not yet verified:**
- **The specification's §35 self-test**, deferred by the human to the end. It runs a small feature through the whole
  pipeline with a controlled failure: detection, reviewer finding, triage, fix, regression detection, refusal to
  close, then success.
- PLAN → CLOSEOUT of this version with real agents.
- The native Claude Code runtime: both e2e runs used `agents-architetura/tools/wf/sim.mjs` with real agents, because the CLI login had
  expired.
- Tool-level read-only enforcement: `/wan-smoke` checks it.
- The real `vastai` adapter against the live API: `vast.mjs doctor` before the first paid run.

## Reference audit

The audit compares against oven-sh/WebKit#249 and against the Elysium specification. Ratings:
- **MATCH:** same behavior, demonstrated by a test.
- **IMPROVED:** goes beyond the reference, demonstrated by a test.
- **PARTIAL:** built, but part of it is not yet demonstrated.
- **MISSING:** not demonstrated at all.

| Capability | Rating | Behavior shown by |
|---|---|---|
| Specification (frozen) | IMPROVED | Request spec reviewed and frozen; specs frozen after system review; silent change fails `frozen` (`approve-freeze.test`, `wan-task`/`wan-prep` sims) |
| Repository-aware planning | IMPROVED | Survey before every milestone; map checked against real paths and coverage (`wan-survey` sim, `inventory.test`) |
| DAG, topological order, cycles, stuck DAG, dedupe | MATCH | `dag.test`, `plan.test` (duplicates, cycles), `scheduler.test` (stuck detection) |
| Dependency-aware execution, parallel waves | MATCH | `wan-implement` sim (waves, blocked dependents) |
| File ownership | IMPROVED | Overlapping concurrent tasks are a plan error; scope audit on real git trees (`plan.test`, `scope-audit.test`) |
| Per-task adversarial review | IMPROVED | Base lenses + specialists + security for high risk; validated findings only (`wan-implement` sim) |
| Independent verification | IMPROVED | Verifier runs verify commands; evidence tied to the tree (`wan-implement` sim, `tree.test`) |
| Security (threat model, diff review, audit) | IMPROVED | `wan-design` threat phase, high-risk security review, mechanism-class audit (sims) |
| Fuzzing | MATCH | `fuzz-run.test`, `wan-fuzz` sim; no domain fuzz profile until wire formats exist |
| Benchmarking | MATCH | Interleaved gate with noise guard, bisect, machine info, scaling gate, comparative study (`bench-gate.test`, `wan-scalebench` sim) |
| Red team | IMPROVED | Confirmed / false positive / needs-human triage (`wan-redteam` sim) |
| Fix loop | IMPROVED | Owner context, prior rejections, statistical bug closure (`wan-fix`, `wan-bughunt` sims) |
| Closeout | IMPROVED | Pinned gates on the current tree; failing gates routed to owners; implementation-report sections (`pinned.test`, `wan-closeout` sim) |
| Failure handling | MATCH | Bounded loops end in "needs human", never silent success (all sims) |
| Human architecture approval | IMPROVED | Hash-bound grants, human-only `grant`, run stops (`approve-freeze.test`, `wan-task`/`wan-adr`/`wan-implement` sims) |
| ADR with falsification criteria | MATCH | `wan-design` Decision phase, `wan-adr` sim |
| Local-first experiments, promotion gate | MATCH | `experiment.test`, `wan-experiment` sim |
| Vast.ai pre-flight, cost gate, monitoring, emergency stop | PARTIAL | `vast.test` against a fake client; the real adapter is not yet exercised live |
| Elysium verification ladder (multi-worker, multi-node, WAN, correctness) | PARTIAL | Rungs and harness contract exist; the scenarios are written by `wan-prep` once a product exists |
| Specialist agents only when justified | MATCH | `wan-survey` sim: unevidenced or missing-path specialists are dropped |
| Git publishing: branch, per-stage commits, checkpoints, draft → ready | IMPROVED | The reference ran a timer and wrote commits by hand. Here one guarded tool does it, with secret/identity/size guards and ready only for the VERIFIED tree (`publish.test`, `wan-task`/`wan-cycle` sims) |
| Self-test of the workflow (spec §35) | MISSING | Deferred by the human to the end |

## Activation (human, once)

- [ ] `claude /login`, then enable "Dynamic workflows" in `/config` in this repo.
- [ ] `/wan-smoke {"token": "s1"}` and expect `ok: true`, including `wan-security-cannot-write`.
- [ ] CodeRabbit: install the GitHub app on this repository only.
- [ ] `claude setup-token`, then `gh secret set CLAUDE_CODE_OAUTH_TOKEN` (paste the token yourself).
- [ ] Commit the factory itself to `main`: `/wan-task` refuses to start from a checkout with uncommitted changes.
- [ ] Recommended: branch protection on `main` (a PR is required, and force pushes are blocked), set by you in GitHub settings.
- [ ] Vast.ai, only when needed:
  - `pip install vastai`;
  - set your API key yourself;
  - run `node agents-architetura/tools/vast/vast.mjs doctor`.
- [ ] `/wan-task {"task": "...", "reference": "github:virgiliolrf/Efficient-Distributed-Training-with-Heterogeneous-Hardware-over-Wide-Area-Networks"}`, then grant, then run `/wan-task` again.
