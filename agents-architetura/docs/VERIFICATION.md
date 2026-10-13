# Verification contract

This file defines what the factory verifies, the tools that verify it, and the seams the product must expose so it can
be verified. The depth is that of oven-sh/WebKit#249, translated to distributed training over WAN.

`wan-design` reviews every design against this contract (the `verifiability` lens), and `wan-prep` builds the seams.
The plan that produced it, with the PR's mechanisms and their numbers, is in [VERIFICATION-PLAN.md](VERIFICATION-PLAN.md).

## The rule

A milestone is **VERIFIED** only when `node agents-architetura/tools/gates/pinned.mjs <M>` passes on the current tree. That command runs
exactly the gates frozen in `docs/design/GATES-<M>.json`; anything else is **NOT VERIFIED**. Failures are never made
quiet, which means:

- A flaky pass is not a pass.
- An expected failure is pinned strictly and against an open finding.
- A suppression carries a ruling and a justification.
- Evidence counts only for the tree it ran on.

## Product seams (what the design must provide)

| Seam | Requirement | Used by |
|---|---|---|
| Injectable effects | All network, time and randomness behind `Transport`, `Clock` and `Rng` interfaces, so one process can simulate the whole cluster | deterministic simulation, chaos, scenario fuzzing |
| Perturbation hook | `perturb(site)` on slow paths only (send/recv, reduce-round boundaries, membership changes, checkpoint commit, optimizer barrier), placed *inside* the racy window. It reads `FACTORY_SEED`, `FACTORY_PERTURB_PERIOD` (0 = off) and `FACTORY_PERTURB_MAX_US`. About 3 in 4 firings yield and 1 in 4 sleep for 1..max µs; the re-arm interval is uniform in [1, 2·period). Each task or thread seeds its PRNG from the seed mixed with its creation ordinal (never an OS id), so a seed replays. When off it costs one load and a branch: the bench gate must show under 1%. The site list lives in the spec. | `amplify.mjs`, `load.mjs`, `wan-campaign` |
| Feature flags | Every WAN optimization (lossless encoding, overlap, scheduling, transport) has a flag. `FACTORY_FEATURES=off` selects the off-path, which must be byte-identical to the synchronous reference (`FACTORY_REFERENCE=1`) and to the base revision. With every feature on, the training state must still be byte-identical to the reference (CLAUDE.md, "Product rule") | `identity.mjs` |
| Oracle | One class only: `exact`. Weights, optimizer state, RNG state, the loss at every step and every checkpoint are byte-identical (sha256) to the synchronous datacenter reference, with every feature on, on every network profile, failure and hardware mix. These are not acceptance criteria: "bounded" (an error bound), "statistical" (a CI over seeds), `allclose`, or identity only under degenerate settings (CLAUDE.md, "Product rule") | oracle, identity rungs |
| Deterministic mode | Fixed seeds, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`, no TF32. The reduction order is the recipe's canonical order, never the one topology, arrival or worker speed would give. Kernels produce the same bits on every GPU architecture in the mix. Output is canonical: sorted dumps and weight hashes instead of timings | divergence detection in `amplify.mjs`, identity rung |
| Identity artifacts | When `FACTORY_IDENTITY_OUT` is set, runs write the training state (weights, optimizer state, RNG state, per-step loss, checkpoints) under `state/` and wire captures under `wire/`, compared by sha256. `exact` in `identity.json` compares only `state/` between the production path and the reference, because wire bytes legitimately differ | `identity.mjs` |
| Trust boundaries | Every untrusted input (wire frames, checkpoints, configs, peer metadata) is listed in the design, has a fuzz target, and never reaches `pickle` or `torch.load(weights_only=False)` | `wan-fuzz`, `wan-audit` |
| Distributed matrix | tests/multiworker (1/2/4 workers, heterogeneous, slow, failed, restarted worker, coordinator failure, partial checkpoint), tests/wan (high latency, loss, bandwidth limit, interruption), tests/multinode (`FACTORY_MULTINODE_HOSTS`). Every scenario asserts SYSTEM WORKED and TRAINING RESULT IS CORRECT separately | multiworker, wan, multinode rungs |
| Numbered invariants | `SPEC-<key>.md` defines `<KEY>-I<n>`; a test claims one with `covers: <KEY>-I<n>`; one that is not yet testable says `DEFERRED-TO: M<n>` | invariants rung |

## Test conventions

- Every test file under `tests/` contains `WOULD-FAIL-IF: <the precise regression it catches>`, enforced by the claims rung.
- A corpus test ships `<test>.would-fail.patch`, a unified diff of product code that introduces that regression.
  `node agents-architetura/tools/gates/claims.mjs --verify <test> --cmd "<cmd with {test}>"` applies it to a snapshot of the tree, never
  the live tree. The test must pass clean and **fail as a test failure** with the patch applied; a crash or a syntax
  error does not count.
- Expected failures: pytest `xfail(strict=True, reason="LEDGER:<id> ...")`, where the id is an open finding in
  `agents-architetura/docs/findings.json`, or `"awaiting implementation: <ID>"` for invariant tests written before their
  implementation. `pyproject.toml` sets `xfail_strict = true`, so an unexpected pass fails the suite. In Rust:
  `#[ignore = "LEDGER:<id> ..."]`.

## Tools

| Tool | What it does | Exit |
|---|---|---|
| `agents-architetura/tools/factory/approve.mjs` | Human grants bound to a document's exact text; `check`, `pending`, `list`; `grant` is human-only | 0 approved · 1 not |
| `agents-architetura/tools/factory/freeze.mjs` | Frozen documents and recorded amendments; `check` is the `frozen` rung | 0 · 1 silent change · 5 |
| `agents-architetura/tools/factory/inventory.mjs` | Repository facts and `--check-paths` for the architecture map | 0 · 1 · 2 |
| `agents-architetura/tools/vast/vast.mjs` | Paid experiments behind the approval, promotion and budget gates (see EXPERIMENTS.md) | 0 · 1 · 3 paused |
| `agents-architetura/tools/gates/ladder.mjs` | 28 rungs (Elysium ladder: … unit → component → integration → multiworker → multinode → oracle → identity → chaos → wan → matrix → rest → full (on demand) … bench → redteam → mutation). `repeat: N` per step (N/N or fail); `heavy` rungs run under the machine lock; `onDemand` rungs (mutation) run only when `--only` or `--require` names them; `--consecutive N`; the report names the tree, toolchain and lockfiles. `--reuse` (state reading only, never a verdict) accepts the last report when every input a product rung reads is unchanged — code, specs, tests, configs, the tools its commands run and their imports, the runner, toolchain, lockfiles, selection — and it is under 24 h old, rerunning only the `factory` rung | 0 green · 1 red |
| `agents-architetura/tools/gates/pinned.mjs <M\|slug>` | The frozen final gate of a milestone or a wan-x phase. Pinned commands take `repeat`, `load` {cmd, copies} for amplified reruns and `metric` {pattern, op, value} for a number the script checks on every repetition. `--validate` checks only the gates file | 0 VERIFIED · 1 NOT · 2 invalid · 5 no gates |
| `agents-architetura/tools/gates/discipline.mjs` | Strict xfail and ledger-backed pins | 0 · 1 · 5 |
| `agents-architetura/tools/gates/invariants.mjs` | Every spec invariant covered; no unknown ids | 0 · 1 · 5 |
| `agents-architetura/tools/gates/claims.mjs` | WOULD-FAIL-IF lint; `--verify` applies the regression patch | 0 · 1 · 2 · 5 |
| `agents-architetura/tools/gates/mutation.mjs --changed-since <rev\|auto>` | Mutants on changed lines: its own Python mutator and cargo-mutants for Rust. Survivors are findings unless justified in `agents-architetura/docs/mutants-equivalent.json` | 0 · 1 · 5 |
| `agents-architetura/tools/gates/bench-gate.mjs` | 9 runs, 1 warmup, interleaved, ≤ 1% budget; refuses on a noisy machine; `--bisect <good> <bad> --bench <id>` | 0 · 1 · 3 noisy |
| `agents-architetura/tools/gates/scaling-gate.mjs` | `bench/scaling.json`: speedup floors (weak or strong scaling) and the serial tax of the features; report-only unless `--gate` | 0 · 1 · 3 · 5 |
| `agents-architetura/tools/gates/scalebench-check.mjs` | The comparative study is valid only if every run shares the data-order and initial-weight checksums, and every run of our system ends in the same training state (`finalState`), equal to the reference's (byte-exact rule) | 0 · 1 · 5 |
| `agents-architetura/tools/verify/amplify.mjs` | M seeded runs × periods. Each run is CRASH, HANG, FAIL or DIVERGENCE against the first clean run, with a replay line | 0 clean · 1 findings |
| `agents-architetura/tools/verify/load.mjs` | W concurrent workers × R runs with rotating seeds; reports peak concurrency | number of failures |
| `agents-architetura/tools/verify/matrix.mjs` | Config and fault matrix. A probe matching "unsupported" means SKIP; any other probe failure means FAIL; more than 50% skipped is vacuous | 0 · 1 · 5 |
| `agents-architetura/tools/verify/identity.mjs` | Feature-off vs reference (and `--against-base`): exit code, output and artifact hashes. `exact`: features on vs reference, training state byte-identical; a config without `exact` fails | 0 · 1 · 5 |
| `agents-architetura/tools/verify/machine-lock.mjs` | One heavy job owns the machine at a time; jobs started by the holder re-enter (`FACTORY_MACHINE_LOCK_HELD`) | — |
| `agents-architetura/tools/campaign/families.mjs` | One campaign round: run once, parse, dedupe into families, suppress, diff against the previous round. A failed command with no report is BROKEN, not zero, and so is a run that lacks its `sentinel` (the instrumentation was not proven active, e.g. `GIL False`) | 0 zero · 1 · 5 |
| `agents-architetura/tools/campaign/fuzz-run.mjs` | Fuzz campaigns: self-test, resumable corpus, crash dedupe, flaky crashes kept, `--replay[-all]`, the CPU-hours and quiet-hour bar | 0 · 1 · 5 |
| `agents-architetura/tools/campaign/scan.mjs` | Scanner battery (SARIF and diagnostics); unavailable scanners are reported loudly | 0 · 1 · 5 |
| `agents-architetura/tools/campaign/suppressions-audit.mjs` | Every suppression has a valid ruling and justification | 0 · 1 · 5 |
| `agents-architetura/tools/campaign/record.mjs` | Writes records that workflow scripts computed (audit results, findings, campaign results) | — |

## Configuration (shared hot files: only an integrator edits them)

- `agents-architetura/tools/campaign/campaigns.json`: sanitizer campaigns (Rust TSAN, Rust ASAN, Miri, free-threaded Python, Python dev
  mode, CUDA compute-sanitizer), amplifier and load campaigns, and fuzz campaigns. Each one is N/A until its `when`
  paths exist, or until its `probe` finds the tool.
- `agents-architetura/tools/campaign/scanners.json`:
  - semgrep, bandit, ruff (S and B rules), mypy `--strict`, pip-audit;
  - clippy pedantic, cargo-deny;
  - osv-scanner, gitleaks, trivy;
  - torch paranoia modes;
  - CodeQL, only when the repository is licensed for it.
- `agents-architetura/tools/verify/matrix.json` and `agents-architetura/tools/verify/identity.json`: filled in by `wan-prep` from the design.
- `agents-architetura/tools/gates/mutation.json`: mutation paths and test commands.
- `agents-architetura/tools/campaign/suppressions/<campaign>.txt`: starts empty and stays justified.

## Workflows and what they enforce

| Workflow | PR counterpart | Bar |
|---|---|---|
| `wan-campaign` | `thread-tsan` | Zero unsuppressed reports. The researcher rules each family; fixes run in waves on disjoint files inside a scope audit, then one build, the benign-or-bug and flag-off-identity reviewers, and one re-run per wave. A stalled count goes to `wan-bughunt` |
| `wan-fuzz` | `thread-fuzz`, `thread-fuzz-setup` | Every new unique crash is minimized into a permanent regression input, fixed by vote, and verified 20/20, 10/10 under load, and by a 10-minute re-fuzz. Flaky crashes are amplified |
| `wan-audit` | `thread-cve-research`, `-audit`, `-close` | Classes by failing invariant; a verdict for every surface; no open SUSCEPTIBLE and nothing INCONCLUSIVE |
| `wan-scanners` | `thread-scanners` | Every family judged against the code: real ones fixed, false positives suppressed with a justification, residuals reported |
| `wan-corpus` | `thread-corpus2` | Tests whose own regression patch makes them fail, 20/20 clean, amplifier-clean and within the time budget |
| `wan-bughunt` | `thread-bughunter[1-4]` | Closure is statistical: zero failures in N ≥ 30 / observed rate runs, split between amplified and loaded runs. If that exceeds the cap the result is INCONCLUSIVE. Ends with `CLOSURE.md` |
| `wan-scalebench` | `thread-scalebench` | Fairness rules and matching checksums, including a byte-identical final state for our system; honest N/A without the hardware |
| `wan-handout` → `GATES-<M>.json` | the pinned `PINNED_VERIFY` lists | Frozen bar: base rungs, the plan's rungs, at least 2 consecutive passes, pinned commands, campaigns, fuzz, audit, mutation |
| `wan-closeout` | `thread-closeout`, `MEGA-RUN-RESULTS` | `pinned.mjs` on the current tree. Each failing gate goes to the workflow that owns it. Verdict VERIFIED or NOT VERIFIED |

## What needs a human or special hardware

| Need | Without it |
|---|---|
| Linux with root (`tc netem`) for real-network chaos and scalebench profiles | toxiproxy, or the in-process simulator |
| An NVIDIA GPU for compute-sanitizer and scalebench | those campaigns and the study are N/A, with the reason recorded |
| Nightly Rust for the TSAN, ASAN and Miri campaigns | those campaigns are N/A until it is installed |
| Compute budget: the PR ran 27 TSAN rounds and multi-hour fuzz campaigns | `{"depth": "quick"}` rehearses the cycle, but closeout still runs the frozen gates |
