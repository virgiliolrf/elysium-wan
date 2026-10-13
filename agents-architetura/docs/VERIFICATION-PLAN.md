# Verification Depth Plan — reaching the level of oven-sh/WebKit#249

> **For agentic workers:** execute task by task with superpowers:subagent-driven-development or
> superpowers:executing-plans. Each task follows the factory protocol: write the listed tests first and run them red;
> implement the task; run them green; then run `node agents-architetura/tools/wf/build.mjs` and `node --test 'agents-architetura/tools/**/*.test.mjs'`. Do not
> commit unless the human asks. Steps use checkbox (`- [ ]`) syntax.

**Goal:** give the factory the same verification depth that let PR #249 land shared-memory threads in JavaScriptCore,
translated from first principles to distributed training over WAN. The end state is a factory where "done" means the
code survived campaigns that look for its bugs: amplified, loaded, sanitized, fuzzed, audited and oracle-checked,
with the evidence pinned.

**Architecture:**
- **Layer 1, command-level gate tools** (`agents-architetura/tools/gates`, `agents-architetura/tools/verify`, `agents-architetura/tools/campaign`; Node, language-agnostic,
  product-independent). They can be built and tested now.
- **Layer 2, campaign workflows** (`agents-architetura/tools/wf/src`). They drive those tools with agents, using the same patterns as the
  PR's `thread-tsan`, `thread-fuzz`, `thread-cve-*`, `thread-scanners`, `thread-corpus2`, `thread-scalebench` and
  `thread-closeout`.
- **Layer 3, verifiability seams.** A contract that `wan-design` must impose on the product and that `wan-prep`'s
  Harness phase must build. It covers the deterministic simulator, the perturbation hook, oracle mode and identity
  flags.

**Status (2026-10-07): Phases A and B are implemented and tested** (242 tests; see [VERIFICATION.md](VERIFICATION.md)
for the resulting contract). Generalizations made while implementing:
- the perturbation and seed variables are factory-wide: `FACTORY_SEED`, `FACTORY_PERTURB_PERIOD`, `FACTORY_PERTURB_MAX_US`;
- the identity variables are `FACTORY_FEATURES=off`, `FACTORY_REFERENCE=1` and `FACTORY_IDENTITY_OUT`;
- campaigns of any kind (sanitizer, amplifier, load, fuzz, scanner) share one triage core.

**Scope now: the factory only (Phases A and B).**
- Every task is verified with fake commands and workflow simulations.
- No product code is written.
- Phases C and D describe what the factory will demand from, and run on, the product. They are **deferred** until
  the human starts the product with `/wan-task`.

**Tech stack:** Node ≥ 22 (tools, workflows, simulation tests), Python + uv/pytest/Hypothesis/atheris/mutmut, Rust +
cargo-fuzz/Miri/loom/`-Zsanitizer`, semgrep, compute-sanitizer (GPU lanes), and toxiproxy / `tc netem` for real-network
chaos.

---

## 1. What made the PR's code reliable (facts from the PR tree, `3a14f2a`)

The PR's reliability did not come from its review loops alone. It came from **campaigns** that drove bug counts to zero
and from **pinned gates** that could not be satisfied by a lucky run.

| # | Mechanism | What the PR did (source) |
|---|---|---|
| M1 | **Pinned ladder with repetition** | V0 build · V1 smoke 20/20 ("flaky-pass is NOT a pass") · V2 corpus no-JIT · V3 full JIT, "must hold across 2 consecutive full-corpus runs" · V4 tier-forced · V5 flag-off identity + bench · V6 GIL-on regression · V-TSAN 0 · V-FUNC 50/50 + 10/10 under load. "A pass on anything other than these exact commands is NOT a pass." (`thread-closeout.js`) |
| M2 | **Sanitizer campaign to zero** | TSAN: one snapshot run → all reports to a file → triage into families by deduped stack pair → ruling (relaxed-atomic / concurrent-accessor / lock / real-bug / suppress) with owned files → waves of parallel write-only fixers on disjoint files → one builder → 2 reviewers (`benign-or-bug`, `flag-off-identity`) → one re-run per wave. Unsuppressed reports went 10,515 (r0) → 0 (r16), back up to 3,692 when the corpus grew to 252 files, then 0 again at r27. The suppressions file starts empty and every entry carries a justification. (`thread-tsan.js`, `TSAN.md`, `TSAN-RESULTS.md`) |
| M3 | **Race amplifier** | A `RaceAmplifier::perturb()` hook sits on slow paths only. It injects a seeded random yield or sleep, gets per-thread PRNGs from seed + creation ordinal, and costs one load plus one branch when off, so it ships in release builds. The `amplify.sh` driver takes M runs with seeds S+1..S+M and uses the first clean run's output as the reference. Each run is classified CRASH / HANG / DIVERGENCE, with a one-line replay command. Test output must be deterministic. (`AMPLIFIER.md`, `amplify.sh`) |
| M4 | **Load harness** | `load6.sh`: 6 concurrent workers × 20 runs with rotating seeds, under self-generated load. Closure bars include "50/50 standalone + 10/10 under load" and "240/240 under 24-way load". |
| M5 | **Oracle mode, tiers and stress matrix** | The GIL-on corpus must stay green (V6). The corpus runs per execution tier (no-JIT, full, tier-forced). `gc-stress-matrix.sh` covers scribble, zombie, contgc and eden. An option-set probe produces SKIP only on "invalid option"; any other probe failure is a FAIL. A run that skips more than 50% exits nonzero as vacuous. (`run-tests.sh`, `gc-stress-matrix.sh`) |
| M6 | **Flag-off identity and serial perf** | `v5a-identity.sh` takes every 50th stress test (40 tests) and requires identical rc + stdout + stderr between the flag-off run and the no-flag run. `bench-gate.sh`: 9-run median, at most 1% regression, baseline recorded by the integrator, `bisect-thc.sh` for bisection. `scaling-gate.sh` sets speedup floors plus a serial-identity tax bound. |
| M7 | **Pinned expected failures and invariant coverage** | `threadsExpectFail`: a nonzero exit counts as XFAIL (tracked separately); **XPASS fails the suite**, because the fix either landed or went latent. A coverage grep requires API-I1..I24 each to have a test. |
| M8 | **Corpus with WOULD-FAIL-IF claims** | Every test ends with `// WOULD-FAIL-IF: <the precise regression>`. A `would-it-catch` reviewer walks the logic to check the test actually trips, and a `discipline` reviewer checks for bounded runtime, joins, deterministic output and no wall-clock luck. (`thread-corpus2.js`) |
| M9 | **Fuzz campaign** | Fuzzilli with a custom profile of about a dozen thread generators, rotating stress flags, an ASAN + coverage build in its own build dir, a resumable corpus, and startup self-tests ("fail fast instead of fuzzing nothing"). The 10-minute smoke reached 1.2 M instrumented edges and found 17 crashes. Triage per unique signature: minimize → diagnose → propose → 2 reviewers → fix → re-fuzz, with the minimized repro kept as a regression test. Flaky crashes are kept and re-run under TSAN. (`FUZZ.md`, `thread-fuzz.js`) |
| M10 | **Mechanism-class security audit** | Three research passes (JS engines, JVM, CLR/Go/BEAM) produced a catalog of 20 classes deduped by *failing invariant*. Each class has a surface map and a verdict: SUSCEPTIBLE (file:line + executed test) / immune (protocol cited) / INCONCLUSIVE. The suite holds 65 susceptibility tests, run by the mechanical `run-tests.sh --cve` gate, and a close workflow handles fixes. (`CVE-AUDIT*.md`, `thread-cve-*.js`) |
| M11 | **Scanner battery** | Five scanners: tsan-deep, ubsan, clang-analyzer, codeql/semgrep, and the engine's own validation modes. Each runs solo and saves raw reports; scanners that cannot run are skipped **loudly**. Results go through dedupe → relevance filter → verify → propose → 2 reviewers → fix. (`thread-scanners.js`) |
| M12 | **Bughunt with closure evidence** | Hypotheses carry `confirmIf` / `refuteIf`. Refuted hypotheses are carried forward between rounds, and discriminating experiments decide between survivors. Closure docs give counts: "69/69 abort cores carry exactly this signature". (`thread-bughunter4.js`, `STW-WATCHDOG-CLOSURE.md`) |
| M13 | **Scalability study** | `thread-scalebench`: a big-program workload specified identically in JS/Go/Java under binding fairness rules. Checksums must match across languages and thread counts. Matrix of 1–32 threads, 5 reps with the median taken, languages interleaved, warmup discarded. Two reviewers check fairness and measurement validity. |
| M14 | **Honest final gate** | `MEGA-RUN-RESULTS.md` records the binaries of record ("`find Source -newer jsc` returned nothing"), the exact commands, per-gate counts, and a verdict of **NOT VERIFIED** when one gate fails. |
| M15 | **Machine discipline** | One build dir per configuration (Debug/Release/TSan/Fuzz never share caches), a loadavg check before performance runs, and only one heavy agent "owns the machine" at a time. |

**What the PR did not have** (we add these as IMPROVED):
- mechanized would-it-catch: the regression patch must make the test fail;
- mutation testing on changed lines;
- statistical closure bars (rule of three);
- deterministic simulation of the whole cluster;
- protocol model checking.

## 2. Where the factory is today

| Mechanism | Today | Gap |
|---|---|---|
| M1 pinned ladder | 16 rungs, `--require`, explicit N/A | No repetition, no "2 consecutive", no build identity, and no pinned per-milestone command file |
| M2 campaigns | none | Whole mechanism missing |
| M3 amplifier | none | Driver and product hook missing |
| M4 load | none | Missing |
| M5 oracle, tiers, matrix | `oracle` rung exists (pytest dir) | No oracle contract, no config matrix, no vacuity rule |
| M6 identity and perf | `bench-gate.mjs` (5 runs, 1%) | No identity harness, no noise guard or interleaving, no scaling gate |
| M7 XFAIL and coverage | invariant tests with "awaiting implementation" markers | No strict-XPASS policy, no invariant-ID coverage gate |
| M8 claims | reviewers judge tests | No WOULD-FAIL-IF, no mechanical check |
| M9 fuzz | `fuzz` rung (Hypothesis CI profile) | No campaign, triage, minimization or regression seeds |
| M10 audit | `wan-redteam` classes (one pass, no research) | No research catalog, surface maps, verdict legend or close loop |
| M11 scanners | `static` (bandit) and `security` (pip-audit, cargo audit) rungs | No battery, no triage loop |
| M12 bughunt | `wan-bughunt` has hypotheses, refuters and experiments | No refuted-knowledge carry between rounds, no closure doc, no statistical bar |
| M13 scale | none | No scaling gate, no study against baselines |
| M14 final gate | `wan-closeout` with `--require` | No build identity, no consecutive runs, no campaign bars, no VERIFIED / NOT VERIFIED verdict |
| M15 machine | none | No heavy-run lock, no per-config dirs |

## 3. Translation to distributed training over WAN (first principles)

In WAN training, bugs come from four sources of nondeterminism:
- **message timing, ordering and loss** across the WAN;
- **membership**: peers joining, leaving, crashing and straggling;
- **intra-node concurrency**: communication threads overlapping compute, async I/O, CUDA streams;
- **floating-point reduction order**.

A JS engine's "races" are this system's interleavings of messages and membership events. Its "GIL-on" is synchronous
training. Its "flag-off" is "WAN optimizations off".

| PR concept | Our analog |
|---|---|
| TSAN / ASAN / UBSAN lanes | Rust: `-Zsanitizer=thread\|address` (nightly, own target dir), Miri for `unsafe`, loom for lock-free code. Python: free-threaded lane (`python3.13t`/`3.14t`, `PYTHON_GIL=0`) for threaded code, `PYTHONDEVMODE=1`, `-X faulthandler`, asyncio debug, pytest-timeout (a hang is a finding). GPU: `compute-sanitizer --tool racecheck\|memcheck\|synccheck\|initcheck` for custom kernels. |
| Race amplifier | `perturb(site)` hook on product slow paths (send/recv, reduce-round boundaries, membership changes, checkpoint commit, optimizer barrier), plus **deterministic simulation** of the whole cluster (seeded network model and fault schedule), as in FoundationDB / TigerBeetle VOPR |
| GIL-on oracle (V6) | **Synchronous reference mode**: all-reduce every step, the recipe's precision, canonical reduction order. One oracle class, `exact`: the training state is byte-identical to the reference with every feature on (CLAUDE.md, "Product rule"; bounded and statistical classes are not accepted) |
| JIT tiers (V2–V4) | Config matrix: CPU/GPU, fp32/bf16, gloo/nccl, compression on/off, world size {1, 2, 4, 8}, heterogeneous mix |
| GC stress matrix | Fault matrix: loss {0, 1, 5}%, RTT {1, 80, 300} ms, bandwidth {10 G, 1 G, 100 M}, churn {off, steady, storm}, straggler ×{1, 4} |
| Flag-off identity (V5a) | Every WAN optimization sits behind a flag. Flag off ⇒ loss trace, weight hash, checkpoint bytes and wire bytes are **byte-identical** to the reference path and to the base revision |
| Fuzzilli profile | Two kinds of fuzz target. **Boundary fuzzers** (atheris / cargo-fuzz) for the wire decoder, checkpoint loader and config parser. A **scenario fuzzer** whose generators emit cluster-event programs for the simulator: `PeerJoinLeaveStorm`, `Straggler`, `PartitionDuringReduce`, `DuplicateFrame`, `ReorderChunks`, `TruncatedFrame`, `ClockSkew`, `CheckpointDuringFailure`, `BandwidthCollapse`, and `ByzantineGradient` if the threat model includes it. The oracle is the invariant checkers. |
| CVE classes | Research areas: PyTorch distributed / `torch.load` deserialization, Ray (unauthenticated job API), Horovod / NCCL / gRPC DoS, federated-learning attacks (poisoning, model replacement, gradient inversion), P2P DHT attacks (Sybil, eclipse; Hivemind / Petals), Byzantine-robust aggregation literature. Classes are deduped by failing invariant, as `MC-DESER`, `MC-AUTH`, `MC-BYZ`, `MC-SYBIL`, `MC-DOS`, `MC-REPLAY`, `MC-STALE`, `MC-TOCTOU`, `MC-LEAK`, `MC-SUPPLY` (the researcher finalizes the list). |
| scalebench vs Go/Java | Time-to-target-loss against baselines (PyTorch DDP sync all-reduce, a DiLoCo-style local-SGD baseline, and others the researcher justifies) under the network profiles. Fairness rules are binding, and the data-order hash and initial-weights hash must match across systems. |

## 4. Verifiability seams the product must have

`wan-design` gains a `verifiability` review lens and `wan-prep`'s Harness phase builds the seams. Without them, the PR's
techniques cannot be applied.

- **R1 Injectable effects.** All network, time and randomness go through `Transport`, `Clock` and `Rng`
  interfaces, so the deterministic simulator can run a whole cluster in one process.
- **R2 Perturbation hook.** `perturb(site)` on slow paths only, driven by `FACTORY_PERTURB_PERIOD`, `FACTORY_SEED` and
  `FACTORY_PERTURB_MAX_US`. It costs one global check when off (the bench gate proves under 1%). Each task's or thread's
  PRNG is seeded from seed + creation ordinal. The site list lives in the spec.
- **R3 Feature flags with identical off-paths.** Every WAN optimization has a flag, and its off-path is
  byte-identical to the reference.
- **R4 Synchronous reference mode.** It uses the same API, so the corpus runs in both modes (the analog of V6).
- **R5 Deterministic mode.** Seeds fixed, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG` set,
  fixed reduction order, and canonical output (sorted dumps, weight hashes) so DIVERGENCE detection works.
- **R6 Invariant IDs and claims.** Every spec invariant has an ID `INV-<ws>-<n>`. Every test declares
  `covers: INV-…` and `WOULD-FAIL-IF: …`.
- **R7 Enumerated trust boundaries.** Every untrusted input (wire, checkpoint, config, peer metadata) is listed in
  the design, has a fuzz target, and never reaches `pickle` or `torch.load(weights_only=False)`.

---

## Phase A — Gate tools (product-independent; build now)

All files are new unless marked *Modify*. Tests live in `agents-architetura/tools/{gates,verify,campaign}/test/*.test.mjs` and use fake
commands (small `node -e` scripts) so they run anywhere.

### A1. Ladder repetition and consecutive runs (M1)
**Files:**
- *Modify* `agents-architetura/tools/gates/ladder.mjs` (step field `repeat`, CLI `--consecutive N`)
- *Modify* `agents-architetura/tools/gates/rungs.json`
- *Modify* `agents-architetura/tools/gates/test/ladder.test.mjs`

**Interface:**
- A step with `"repeat": N` passes only if all N runs exit 0. Detail on failure:
  `"19/20 passed (run 7 failed) — a flaky pass is not a pass"`.
- `--consecutive N` runs the selected rungs N times in a row. Status is pass only if every pass is green; the JSON
  gains `passes: [...]`.

**Tests:**
- [ ] `repeat: 20` where run 7 fails ⇒ `fail`, and the detail contains `19/20`.
- [ ] `repeat: 3`, all pass ⇒ `pass`, `runs: 3`.
- [ ] `--consecutive 2` where the second pass fails a rung ⇒ `allGreen: false`, and `passes[1]` names the rung.
- [ ] `repeat` together with `naExitCodes` ⇒ N/A short-circuits on the first run (no repetition of N/A).

### A2. Build identity and stale-evidence detection (M14)
**Files:**
- `agents-architetura/tools/gates/identity.mjs` (tree id)
- *Modify* `agents-architetura/tools/gates/ladder.mjs` (emit `tree`, `toolchain`, `locks`)
- *Modify* `agents-architetura/tools/wf/lib/gates.js` (`sameTree(ladder, fresh)`)
- Tests in `agents-architetura/tools/gates/test/identity.test.mjs`

**Interface:**
- `treeId(root)` computes the content hash of the working tree, untracked files included, via a temp `GIT_INDEX_FILE`
  plus `git add -A` and `git write-tree` (the same technique as `scope-audit.mjs`).
- Every ladder JSON carries `tree`, plus `toolchain: {node, python, uv, cargo, rustc}` (`n/a` when absent) and `locks`
  (sha256 of `uv.lock` / `Cargo.lock`).
- Closeout and wan-fix compare the pasted ladder `tree` with a fresh `treeId`. A mismatch fails with
  `stale evidence: ladder ran on <a>, tree is <b>`.

**Tests:**
- [ ] The id is stable across two calls.
- [ ] The id changes after editing a tracked file.
- [ ] The id changes after adding an untracked file.
- [ ] Ignored files do not change it.
- [ ] The real index is untouched (`git diff --cached` is empty before and after).

### A3. Pinned per-milestone gates (M1, M14)
**Files:**
- `agents-architetura/tools/gates/pinned.mjs`
- *Modify* `agents-architetura/tools/wf/lib/project.js` (`gatesPath(m) = docs/design/GATES-<m>.json`)
- *Modify* `agents-architetura/tools/wf/lib/schemas.js` (`GATES`)
- Tests

**Interface:** `GATES-<m>.json` is written by `wan-handout` and frozen like specs:

```json
{"milestone":"M0","consecutive":2,
 "rungs":["factory","build","syntax","unit","oracle","integration","chaos","fuzz","identity","matrix","invariants","discipline","bench"],
 "pinned":[{"id":"amplify-core","cmd":"node agents-architetura/tools/verify/amplify.mjs --runs 200 --seed-base 1000 -- uv run pytest -q tests/chaos","bar":"exit0"},
           {"id":"load-membership","cmd":"node agents-architetura/tools/verify/load.mjs --workers 6 --runs-per-worker 20 -- uv run pytest -q tests/chaos/test_membership.py","bar":"exit0"}],
 "campaigns":[{"id":"rust-tsan","bar":"zero"},{"id":"fuzz-wire","bar":"no-new-unique","minCpuHours":4}],
 "audit":{"bar":"no-open-tier-a"}}
```

The script runs the commands itself, so agents cannot substitute different commands. It writes
`.factory/gates/<m>.json` with every command, its exit code, a tail of its output, its duration, and the `tree`.

**Tests:**
- [ ] An unknown rung id ⇒ refuse.
- [ ] A `cmd` with control characters ⇒ refuse.
- [ ] A pinned command exiting nonzero ⇒ that gate fails.
- [ ] The consecutive rule is honored.
- [ ] The output carries `tree`.

### A4. Expected-failure discipline (M7)
**Files:**
- `agents-architetura/tools/gates/discipline.mjs`
- `agents-architetura/docs/findings.json` (open-findings ledger: `{id, title, status, opened, source}`)
- New rung `discipline` in `rungs.json`

**Rules:**
- `pyproject.toml` must set `xfail_strict = true`, so an XPASS fails the suite.
- Every `pytest.mark.xfail` / `pytest.xfail` reason starts with `LEDGER:<id>`, and the id must be `open` in
  `findings.json`.
- Every Rust `#[ignore]` reads `#[ignore = "LEDGER:<id> …"]`.
- A `closed` finding still pinned as xfail is a failure.

**Tests:**
- [ ] `xfail` without a ledger id ⇒ fail.
- [ ] An unknown id ⇒ fail.
- [ ] A pin on a closed id ⇒ fail.
- [ ] `xfail_strict` missing ⇒ fail.
- [ ] No tests dir ⇒ `n/a`.

### A5. Invariant coverage gate (M7)
**Files:**
- `agents-architetura/tools/gates/invariants.mjs`
- New rung `invariants`

**Rules:**
- Defined ids are the `INV-[A-Z0-9]+-[0-9]+` ids in the invariant tables of `docs/design/SPEC-*.md`.
- Covered ids are the `covers: INV-…` lines in `tests/**`.
- Each defined id needs at least one test, unless its spec row says `DEFERRED-TO: M<n>` with a reason; deferred ids
  are listed in the output.
- A test referencing an undefined id fails the gate.
- The output is `{defined, covered, uncovered, unknown, deferred}`.

**Tests:**
- [ ] One uncovered id ⇒ fail, naming it.
- [ ] A deferred id ⇒ pass, listed.
- [ ] A typo id in a test ⇒ fail.
- [ ] No specs ⇒ `n/a`.

### A6. Test claims and mutation (M8, IMPROVED)
**Files:**
- `agents-architetura/tools/gates/claims.mjs`
- `agents-architetura/tools/gates/mutation.mjs`
- New rungs `claims` and `mutation`
- `agents-architetura/docs/mutants-equivalent.json`

**Claims:**
- Every test file under `tests/{unit,oracle,integration,chaos,redteam,fuzz}` contains `WOULD-FAIL-IF:` and at least
  one `covers: INV-…`.
- `claims.mjs --verify <test>` applies the sibling `<test>.would-fail.patch` (`git apply` on a temp worktree) and
  requires the test to **fail**. It then requires it to pass on the clean tree. This makes the PR's `would-it-catch`
  review mechanical.

**Mutation:**
- `mutation.mjs --changed-since <rev>` runs mutmut (Python) / `cargo mutants --in-diff` (Rust) on lines changed since
  the milestone base.
- A surviving mutant is a finding unless it is listed in `mutants-equivalent.json` with a justification.

**Tests:**
- [ ] A missing claim ⇒ fail.
- [ ] A patch that does not make the test fail ⇒ `vacuous test` fail.
- [ ] A patch that does not apply ⇒ fail.
- [ ] Equivalent-mutant entries without a justification ⇒ fail.
- [ ] mutmut absent ⇒ `n/a` with reason.

### A7. Amplifier driver (M3)
**Files:**
- `agents-architetura/tools/verify/amplify.mjs`
- `agents-architetura/tools/verify/test/amplify.test.mjs`

**CLI:**
```
node agents-architetura/tools/verify/amplify.mjs --runs M [--seed-base S] [--period N] [--max-sleep-us U] [--timeout SECS] [--keep-logs] -- <cmd...>
```

**Behavior:**
- Per run, set `FACTORY_SEED=S+i`, `FACTORY_PERTURB_PERIOD=N`, `FACTORY_PERTURB_MAX_US=U`.
- The first exit-0 run's normalized output is the reference (strip lines matching `--ignore <regex>`, default timing
  lines).
- Each run is classified CRASH (killed by a signal), FAIL (nonzero), HANG (timeout) or DIVERGENCE (output differs).
- Findings print the seed, the log path, and the replay line `FACTORY_SEED=… <cmd>`.
- The run writes `.factory/amplify/<stamp>/summary.json`.
- Exit codes: 0 clean, 1 any finding, 2 usage.

**Tests:**
- [ ] The fake command fails only on seed 1007 ⇒ exit 1, the summary names 1007, and the replay line contains
  `FACTORY_SEED=1007`.
- [ ] Sleeps on seed 1003 ⇒ HANG.
- [ ] Prints a seed-dependent value ⇒ DIVERGENCE.
- [ ] All clean ⇒ exit 0.
- [ ] `--seed-base` makes the seed list reproducible.

### A8. Load driver (M4)
**Files:**
- `agents-architetura/tools/verify/load.mjs`
- Tests

**CLI:**
```
node agents-architetura/tools/verify/load.mjs --workers W --runs-per-worker R [--seed-base S] [--timeout SECS] [--require-pass-marker] -- <cmd...>
```

**Behavior:**
- Runs W concurrent workers with rotating seeds.
- Keeps failing logs only.
- The exit code is the failure count (capped at 125).
- Writes a summary JSON.

**Tests:**
- [ ] 3 workers × 4 runs with 2 seeded failures ⇒ exit 2 and both logs kept.
- [ ] The concurrency is real: max overlap ≥ 2, measured from timestamps.
- [ ] With the pass-marker flag, a run that exits 0 but lacks `PASS` ⇒ failure.

### A9. Config/fault matrix with vacuity rule (M5)
**Files:**
- `agents-architetura/tools/verify/matrix.mjs`
- `agents-architetura/tools/verify/matrix.json` (empty `modes` until wan-prep fills it)
- New rung `matrix`

**Config:**
```json
{"modes":[{"id":"lossy","env":{"FACTORY_NET_LOSS":"0.05"},
           "probe":"uv run python -m wan.selfcheck",
           "unsupported":"unsupported mode"}],
 "cmd":"uv run pytest -q tests/integration","maxSkipPct":50}
```

**Behavior:**
- A mode is SKIPPED only when the probe output matches `unsupported`. Any other probe failure FAILs every run of
  that mode.
- If skips exceed `maxSkipPct`, the result is `fail: vacuous`.

**Tests:**
- [ ] A probe that crashes with a signal ⇒ FAIL, not SKIP.
- [ ] 3 of 4 modes skipped ⇒ vacuous fail.
- [ ] Per-mode env reaches the command.

### A10. Identity harness (M6)
**Files:**
- `agents-architetura/tools/verify/identity.mjs`
- `agents-architetura/tools/verify/identity.json`
- New rung `identity`

**Config:**
```json
{"corpus":"tests/oracle/**/test_*.py","every":50,"max":40,
 "a":"FACTORY_FEATURES=off uv run pytest -q {test}","b":"uv run pytest -q {test} --reference",
 "artifacts":[".factory/identity/{run}/weights.sha256"],"baseRev":"origin/main"}
```

**Behavior:**
- Selection is deterministic: the sorted list, every Nth file.
- Comparison covers exit code + stdout + stderr bytes + the sha256 of each declared artifact.
- `--against-base` runs side `a` in a temp `git worktree` at `baseRev` as well.
- Mismatches print the first 10 diff lines.

**Tests:**
- [ ] Identical commands ⇒ pass N/N.
- [ ] A command printing pid ⇒ mismatch.
- [ ] Artifact hash differs ⇒ mismatch.
- [ ] Selection is stable across file-order shuffles.

### A11. Campaign triage core (M2, M9, M11)
**Files:**
- `agents-architetura/tools/campaign/parse.mjs`
- `agents-architetura/tools/campaign/families.mjs`
- `agents-architetura/tools/campaign/suppressions-audit.mjs`
- `agents-architetura/tools/campaign/campaigns.json`
- `agents-architetura/tools/campaign/suppressions/` (empty files)
- Fixtures in `agents-architetura/tools/campaign/test/fixtures/`

**Parsers** normalize to `{tool, kind, signature, frames, location, message}`. They cover:
- clang TSAN (`WARNING: ThreadSanitizer: data race`, 2 stacks);
- ASAN / UBSAN / LSAN;
- Rust panic and Miri UB;
- Python traceback;
- pytest junit XML;
- Hypothesis `Falsifying example`;
- compute-sanitizer racecheck / memcheck;
- amplify and load summaries.

**Signature rule:** kind + the top 3 in-repo frames (function + file, line numbers dropped, addresses stripped). For
two-stack reports, the stacks are sorted, so pair order does not matter.

**`families.mjs --round rN --prev rM`:**
- Outputs `{unsuppressed, families:[{id, count, signature, sample}], new, fixed, persisting}`.
- Suppressions are applied first.

**Suppressions audit:**
- Every entry is preceded by `# RULING: benign-by-spec <SPEC §>` or `# RULING: third-party <why>`, plus
  `# JUSTIFICATION: …`.
- A `benign-by-spec` entry must cite a spec section that exists.
- An entry with no ruling fails.

**Tests:** one real-format fixture per parser. Then:
- [ ] Two reports with swapped stack order ⇒ one family.
- [ ] A line-number change ⇒ same family.
- [ ] `new` / `fixed` deltas are correct across rounds.
- [ ] An unjustified suppression ⇒ fail.
- [ ] A cited spec section that does not exist ⇒ fail.

### A12. Bench hardening and scaling gate (M6, M13)
**Files:**
- *Modify* `agents-architetura/tools/gates/bench-gate.mjs`
- `agents-architetura/tools/gates/scaling-gate.mjs`
- `bench/scaling.json` schema

**Bench gate:**
- Default 9 runs and one discarded warmup per bench.
- Benches are interleaved round-robin instead of run in blocks.
- Quiet-machine guard: load average per core above `--max-load` (default 0.25) ⇒ exit 3, which reports as
  `n/a: noisy machine`, never as a pass.
- Reports median, min, max and the delta against the baseline.
- `--bisect <good> <bad> --bench <id>` builds temp worktrees per commit and finds the first commit over budget.

**Scaling gate:**
- `speedup(N) = N·T(1)/T(N)`, with floors per workload.
- Serial-identity tax: T(1) with features on vs off within `serialPct`.
- Report-only unless `--gate`.

**Tests:**
- [ ] The warmup sample is excluded from the median.
- [ ] Interleaving order is A, B, A, B.
- [ ] A noisy load average ⇒ exit 3.
- [ ] A floor miss with `--gate` ⇒ exit 1; report-only ⇒ exit 0 with a warning.
- [ ] Bisect on a synthetic 5-commit history finds commit 3.

### A13. Machine lock and per-config directories (M15)
**Files:**
- `agents-architetura/tools/verify/machine-lock.mjs`
- *Modify* `agents-architetura/tools/gates/ladder.mjs`: a rung with `"heavy": true` takes the lock

**Lock:**
- An `O_EXCL` lock file `.factory/machine.lock` holds `{pid, label, since}`.
- A stale lock (dead pid) is reclaimed.
- `--wait SECS` waits for the lock.

**Per-config dirs:**
- Campaign configs set `CARGO_TARGET_DIR=target/<config>` and `UV_PROJECT_ENVIRONMENT=.venv-<config>`, so sanitizer
  builds never poison the normal ones.
- Add `.venv-*/` to `.gitignore`.

**Tests:**
- [ ] A second acquirer is refused.
- [ ] The lock of a dead pid is reclaimed.
- [ ] The lock is released on exit.
- [ ] Heavy rungs serialize when two ladders run at once.

## Phase B — Campaign workflows (each with a simulation test, as for every workflow today)

### B1. `wan-campaign` — sanitizer or amplifier campaign to zero (from `thread-tsan`)
**Files:**
- `agents-architetura/tools/wf/src/wan-campaign.js`
- `agents-architetura/tools/wf/test/wan-campaign.sim.test.mjs`
- *Modify* `lib/schemas.js` (`FAMILY_RULINGS`)

**Args:** `{"campaign": "<id from agents-architetura/tools/campaign/campaigns.json>", "maxWaves": 8}`

**Phases:**
1. **Snapshot.** The verifier runs alone: it takes the machine lock, runs the campaign cmd, then
   `node agents-architetura/tools/campaign/families.mjs --json`, and pastes the output verbatim. The researcher then rules each family:
   `benign-by-spec` (must cite a SPEC §), `protocol-fix`, `real-bug` or `suppress`, and gives `fixShape` and the
   exact `files` the fix will write.
2. **Waves.**
   - The script partitions families: real-bug families get solo-grade attention (at most 3 per wave), and mechanical
     families are batched on disjoint files (at most 12).
   - Coders write without building, from their task journal.
   - One builder (the verifier) runs `ladder --only build,syntax,unit`.
   - Two reviewers check the wave: `benign-or-bug` ("a real bug annotated into silence") and
     `flag-off-identity / oracle` (did the off-path or the reference mode change?).
   - The owners amend.
   - **One** re-run produces the new family table.
   - The script tracks the count per round and stops at zero.
   - If the count does not drop for 2 waves, it calls `wan-bughunt` on the top family.
   - A family a fixer reports as mis-ruled is re-ruled next wave.
3. **Gate.** Zero unsuppressed; `suppressions-audit` passes; normal-build sanity (`ladder --only unit,oracle`); the
   bench gate; identity.
4. **Report.** `agents-architetura/docs/campaigns/<id>/RESULTS.md`: a round-by-round count table (r0 → rN), families by ruling,
   suppressions with justifications, residuals. "Honest partials over fake green."

**Sim tests:**
- [ ] 40 → 12 → 0 over 2 waves ⇒ ok, with the round table persisted.
- [ ] Wave file sets are disjoint.
- [ ] A real-bug family is never batched with others.
- [ ] A stall (12 → 12 → 12) ⇒ `wan-bughunt` is called with the top family.
- [ ] A suppression without justification ⇒ the gate fails.
- [ ] Missing families JSON ⇒ stop, not success.
- [ ] Role routing: researcher rules, coder fixes, verifier runs, reviewer reviews.

### B2. `wan-fuzz` — coverage-guided campaign with triage (from `thread-fuzz`)
**Files:**
- `agents-architetura/tools/wf/src/wan-fuzz.js`
- `agents-architetura/tools/campaign/fuzz-run.mjs` (runs atheris / cargo-fuzz / Hypothesis targets with `--minutes`, `--jobs` and
  `--resume`, storing everything under `.factory/fuzz/<target>/{corpus,crashes,stats.json}`)
- Sim test

**Phases:**
1. **Setup.** A solo coder builds the targets listed in the design's trust boundaries (R7) and the scenario fuzzer
   (§3). Startup self-tests prove the generators reach the code: coverage on the transport module must be > 0, or the
   phase fails fast instead of "fuzzing nothing". The coder then runs a 10-minute smoke that must show the corpus
   growing.
2. **Campaign.** Solo, under the machine lock, `fuzz-run.mjs` per target. Crashes are deduped by `families.mjs`, and
   flaky crashes are kept.
3. **Triage.** Each unique signature goes through a pipeline:
   - minimize (libFuzzer `-minimize_crash`, Hypothesis shrink, or ddmin over scenario events);
   - propose (proposer);
   - 2 reviewers (correctness, regression);
   - apply (coder, owner context).
   The verifier then checks four things:
   - the minimized repro no longer fails, 20/20, plus 10/10 under `load.mjs`;
   - the ladder passes `unit,fuzz`;
   - a re-fuzz of the target for `--minutes 10` with the seed in the corpus finds nothing;
   - the repro is committed as `tests/fuzz/regressions/<signature>.*` with a `WOULD-FAIL-IF`.
   A flaky crash that will not reproduce is re-run under `amplify.mjs` and the sanitizer lane before it is declared
   unreproducible.

**Bar:** `no-new-unique` — no new unique signature in the last hour of a campaign that lasts at least `minCpuHours`.

**Sim tests:**
- [ ] Two crashes with the same signature ⇒ one triage.
- [ ] A repro that still fails after apply ⇒ revert and a finding stays open.
- [ ] A smoke with zero coverage ⇒ stop.
- [ ] A flaky crash is never dropped silently.

### B3. `wan-audit` — mechanism-class security audit (from `thread-cve-research` / `-audit` / `-close`; supersedes the classes step of `wan-redteam`)
**Files:**
- `agents-architetura/tools/wf/src/wan-audit.js`
- Sim test
- Outputs under `docs/security/`

**Phases:**
1. **Research.** Three researchers in parallel, one per area in §3, with web access. Each writes
   `docs/security/research/<area>.md` with a citation and a confidence label for every entry.
2. **Catalog.** The researcher merges the research by **failing invariant** into `docs/security/AUDIT.md`. Each class
   records its exemplars, its surface in our design, and the known-good fix shape.
3. **Map.** One reviewer per class, in parallel and read-only, writes `docs/security/map-MC-*.md`. Each surface gets a
   verdict: `immune` (protocol cited adversarially) or `needs-test`.
4. **Test.** Red-team writers per class put tests in `tests/redteam/cve/<class>/`, inside a scope audit.
5. **Run.** The verifier runs alone, class by class, and execution promotes each surface to `SUSCEPTIBLE` (file:line
   + executed evidence), `immune`, or `INCONCLUSIVE` (what would discharge it).
6. **Results.** `docs/security/AUDIT-RESULTS.md` is tiered: A for confirmed safety or security holes with executed
   evidence, B for spec obligations pinned with `xfail(strict) LEDGER:`, C for inconclusive surfaces. Each item has
   its hole, fix shape and gating test.
7. **Close** (when `{"close": true}`). Tier A and B go to `wan-fix`, and to `wan-bughunt` when stubborn. The pinned
   verify runs the exact suite; 0 failures are allowed, and an XPASS fails it.

`{"pass": 2}` re-runs Research and folds new findings into the existing classes without changing the taxonomy, as the
PR's second pass did.

**Sim tests:**
- [ ] A class with no surface verdict ⇒ incomplete.
- [ ] A SUSCEPTIBLE surface without executed evidence ⇒ demoted to INCONCLUSIVE.
- [ ] Close routes only tiers A and B.
- [ ] Writers outside `tests/redteam/cve/` ⇒ the scope audit stops the run.

### B4. `wan-scanners` — battery with triage (from `thread-scanners`)
**Files:**
- `agents-architetura/tools/wf/src/wan-scanners.js`
- `agents-architetura/tools/campaign/scanners.json`
- Sim test

**Battery** (each scanner solo, raw reports under `.factory/scan/<id>/`; scanners that cannot run are recorded as
`skipped-unavailable` with the reason and listed in the report):
- semgrep (`p/python`, `p/security-audit`, `p/secrets`);
- bandit `-ll -iii`;
- ruff `S,B` rules;
- mypy `--strict` on `src`;
- pyright;
- cargo clippy pedantic, cargo-deny and cargo-geiger;
- Miri on crates with `unsafe`;
- Rust TSAN / ASAN lanes;
- osv-scanner, gitleaks, and trivy when a Dockerfile exists;
- compute-sanitizer when an NVIDIA GPU exists;
- torch paranoia modes over the oracle corpus (`detect_anomaly`, `TORCH_DISTRIBUTED_DEBUG=DETAIL`,
  `NCCL_ASYNC_ERROR_HANDLING=1`, deterministic algorithms);
- CodeQL only if the repository is licensed for it (GitHub Advanced Security on private repos).

**Triage:** `families.mjs` dedupes. The verifier filters for relevance (in-repo, product scope). Each finding then goes
through verify → propose → 2 reviewers → apply. The affected scanners are re-run, and `agents-architetura/docs/SCAN-RESULTS.md`
lists the residuals.

**Sim tests:**
- [ ] An unavailable scanner is listed, never silently dropped.
- [ ] An out-of-scope finding is filtered with a reason.
- [ ] Re-run residuals are reported.

### B5. `wan-corpus` — stress corpus with mechanized claims (from `thread-corpus2`)
**Files:**
- `agents-architetura/tools/wf/src/wan-corpus.js`
- Sim test

**Phases:**
1. **Author.** Parallel writers per suite (`semantics`, `chaos`, `scaling`, `fault-matrix`) write to disjoint
   `tests/<suite>/staging/` dirs under a scope audit. Each test carries `covers:` and `WOULD-FAIL-IF:` and comes with
   a `.would-fail.patch`.
2. **Review.** Per suite, the `would-it-catch` and `discipline` lenses loop at most 3 rounds.
3. **Validate.** A solo verifier runs one test at a time and checks four things:
   - clean tree ⇒ pass, 20/20;
   - `claims.mjs --verify` ⇒ fails with the patch;
   - `amplify.mjs --runs 50` is clean;
   - runtime is under 30 s.
4. **Promote.** The staging dir moves into the suite (the integrator does this).

**Sim tests:**
- [ ] A test whose patch does not make it fail is rejected as vacuous.
- [ ] A runtime over budget is rejected.
- [ ] Staging dirs are disjoint.

### B6. `wan-scalebench` — comparative study (from `thread-scalebench`)
**Files:**
- `agents-architetura/tools/wf/src/wan-scalebench.js`
- Sim test
- Outputs under `bench/scalebench/`

**Phases:**
1. **Design.** The researcher writes `bench/scalebench/SPEC.md`. It fixes:
   - the workload: model sizes, dataset, token budget, target loss;
   - the baselines;
   - the network and heterogeneity profiles from §3;
   - binding fairness rules: same model, data order, seeds, optimizer and hyperparameters; equal tuning budget;
     defaults documented;
   - checksums: data-order hash and initial-weights hash must match across systems;
   - the matrix: nodes {1, 2, 4, 8}, 5 reps with the median, interleaved, warmup discarded;
   - the metrics: time-to-target-loss, tokens/s, bytes on wire per step, scaling efficiency, final eval.
2. **Implement.** Runner, baseline adapters and the network emulation (`tc netem` on Linux with root; toxiproxy
   elsewhere) are written in parallel in disjoint dirs.
3. **Review.** `fairness` and `measurement-validity` lenses, at most 2 loops.
4. **Run.** Solo, under the machine lock and the quiet check. `n/a` without the declared hardware, never fake numbers.
5. **Report.** `bench/scalebench/RESULTS.md`.

**Sim tests:**
- [ ] A checksum mismatch invalidates the run.
- [ ] Missing hardware ⇒ `n/a` with reason.
- [ ] Reviewer findings loop back to the implementers.

### B7. `wan-bughunt` closure upgrade (M12, IMPROVED)
**Files:**
- *Modify* `agents-architetura/tools/wf/src/wan-bughunt.js`
- *Modify* its sim test

**Knowledge carry:** refuted hypotheses, experiment results and rejected fixes are carried into every finder and
proposer prompt across rounds, as `thread-bughunter4` did.

**Statistical closure (rule of three):**
- From the evidence phase's measured rate p̂ = k/n, closure needs **0 failures in N ≥ ⌈30/p̂⌉ runs**. The 95% upper
  bound 3/N then sits at least 10× below the old rate.
- N is split across `amplify.mjs` (standalone) and `load.mjs` (under load).
- If the cap (`maxRuns`, default 5,000) is hit first, the verdict is `INCONCLUSIVE`, never `fixed`.

**Closure doc:** `agents-architetura/docs/bughunt/<id>/CLOSURE.md` records:
- the root cause in one sentence;
- signature match (k/k failures carried it);
- the fix location;
- the post-fix counts;
- the build identity (`tree`).

**Sim tests:**
- [ ] p̂ = 1/120 ⇒ N = 3,600.
- [ ] One failure in N ⇒ not fixed, and the coder reverts.
- [ ] Cap reached ⇒ INCONCLUSIVE.
- [ ] Refuted hypotheses appear in the next round's prompts.

### B8. `wan-closeout` as the honest final gate (M14)
**Files:**
- *Modify* `agents-architetura/tools/wf/src/wan-closeout.js`
- *Modify* its sim test

**Behavior:**
- The verifier runs `node agents-architetura/tools/gates/pinned.mjs <m> --json`. It covers the GATES file's rungs `--consecutive 2`, the
  pinned commands, the campaign bars (read from `agents-architetura/docs/campaigns/*/RESULTS.json`), the audit bar, the
  invariants and the discipline rung.
- The script checks `tree` against a fresh `treeId`.
- The report `agents-architetura/docs/reports/<m>-GATE.md` has a per-gate table with exact counts and a verdict that is exactly
  `VERIFIED` or `NOT VERIFIED (<gates>)`.
- A NOT VERIFIED milestone stops `wan-cycle`.

**Sim tests:**
- [ ] One campaign not at zero ⇒ NOT VERIFIED.
- [ ] A stale tree ⇒ NOT VERIFIED.
- [ ] The second consecutive pass red ⇒ NOT VERIFIED.
- [ ] All green ⇒ VERIFIED.

### B9. Wiring
**Files:**
- *Modify* `wan-design.js`: `verifiability` lens checks R1–R7; design must list trust boundaries, flags, oracle
  classes, perturb sites.
- *Modify* `wan-prep.js`: Harness builds the Phase C seams and fills `matrix.json`, `identity.json`,
  `campaigns.json`, fuzz targets. Harness writers may declare `oracle` / `chaos` / `fuzz` N/A only if the design has
  no network component, with the researcher's justification.
- *Modify* `wan-handout.js`: writes and freezes `GATES-<m>.json`.
- *Modify* `wan-cycle.js`: Plan → Implement → Corpus → Audit → Campaigns (per GATES) → Fuzz → Scanners → Verify →
  Benchmark → Closeout. `{"depth": "quick"}` scales bars down for CI-sized runs; closeout always uses the GATES bars.
- *Modify* `wan-task.js`, `CLAUDE.md`, `README.md`, `agents-architetura/docs/PLAN.md`.

**Sim tests:**
- [ ] Cycle order is as listed.
- [ ] A campaign failure stops the cycle before closeout.
- [ ] A design missing a seam does not converge.

## Phase C — Verifiability seams (DEFERRED: built by `wan-prep` Harness once the product design exists)

Acceptance tests are fixed here so the harness writers cannot weaken them.

| Task | Deliverable | Acceptance |
|---|---|---|
| C1 Deterministic simulator | `tests/harness/dst/`: discrete-event loop, virtual clock, seeded network model (latency distribution, jitter, bandwidth/serialization delay, loss, duplication, reordering, scheduled partitions, crash/restart, straggler factor), fault schedule as data, invariant-checker hooks, canonical trace hash | Same seed ⇒ identical trace hash over 100 runs. Distinct seeds ⇒ distinct traces. Bandwidth model exact to the formula. ddmin shrinks a synthetic 200-event failing schedule to its 2 causal events. |
| C2 Perturb hook | `perturb(site)` in product slow paths; site list in the spec | Off-cost < 1% by `bench-gate`. The same seed reproduces the same decision stream (per-task ordinal). A planted window bug that never fails in 1,000 plain runs fails within 200 amplified runs. |
| C3 Oracle mode | Synchronous reference trainer; exact comparator | Training state byte-identical (sha256) to the reference with every feature on, across the fault and hardware matrix |
| C4 Flags + identity | Flag per WAN optimization; `identity.json` | 40/40 identical (flag off vs reference; features on vs reference on `state/`), and identical against the base revision |
| C5 Fuzz targets | Boundary fuzzers + scenario fuzzer with the generators from §3 | Startup self-test proves generator reach. A 10-minute smoke grows the corpus. |
| C6 Sanitizer lanes | `campaigns.json` entries for the Rust TSAN/ASAN/Miri/loom, free-threaded Python and compute-sanitizer lanes, each in its own dir | Each lane runs the corpus or is `n/a` with the reason (no Rust / no GPU) |
| C7 Protocol model *(beyond the PR)* | TLA+ spec of the coordination protocol (rounds, membership, reduce, checkpoint commit) checked by TLC in a `model` rung | No invariant violation or deadlock at the declared bounds. Each reachable-state property maps to an `INV-` id. |

## Phase D — Run to the PR's level (DEFERRED: bars on the first real milestone)

`VERIFIED` requires all of the following:
- the pinned ladder green twice in a row, with build identity;
- every sanitizer and amplifier campaign at **0 unsuppressed**, every suppression justified;
- amplifier: chaos and oracle corpus × 200 seeds × periods {16, 64, 256} clean;
- load: 6 × 20 clean for every concurrency test;
- identity 40/40, against both the flag-off path and the base revision;
- the exact oracle holds (byte-identical training state);
- fuzz: each target ≥ 4 CPU-hours with no new unique signature in the last hour, and every crash found has a
  regression test;
- audit: every class has a verdict per surface and no open tier A;
- scanners run, with residuals listed;
- no surviving mutant on changed lines (or a justified equivalent);
- bench ≤ 1%, scaling floors met;
- the scalebench report written (or `n/a` without hardware).

Anything less is `NOT VERIFIED` with the failing gates named.

## Human touchpoints

- **Infrastructure:**
  - a Linux host with root for `tc netem` (GitHub-hosted Ubuntu runners allow `sudo tc`);
  - an NVIDIA GPU for compute-sanitizer and scalebench;
  - nightly Rust for the sanitizers and Miri.
- **Compute budget for campaigns.** The PR ran its campaigns over weeks: 27 TSAN rounds, multi-hour fuzzing.
- **The threat model** (are peers adversarial?). The researcher marks it as `ASSUMPTION:` in the brief; the human may
  override.
- **CodeQL licensing** on a private repository.

## Order of execution

A1 → A2 → A3 → A13 → A7 → A8 → A11 → A4 → A5 → A6 → A9 → A10 → A12, then B1 → B8 → B7 → B2 → B5 → B3 → B4 → B6 → B9.
That is the whole of the current work.

Phase C runs only when the human starts the product with `/wan-task`, and Phase D is that task's closeout. Neither
is part of building the factory.

Each Phase A task is independently testable. Each Phase B workflow ships with its simulation test, and the full
factory suite must stay green after every task.

## Honest limits

No amount of verification proves the absence of bugs. The PR's own record shows this:
- TSAN went back up from 0 to 3,692 reports when the corpus grew;
- its final mega-run was declared NOT VERIFIED when one gate failed.

Depth makes bugs cheap to find and expensive to ship. That is the property this plan buys.
