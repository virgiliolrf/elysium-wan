# Lessons

Shared memory of the factory. Every agent reads this before working. One line per lesson, written when a review,
verification or red-team finding exposed a mistake that could recur — what went wrong, the rule that prevents it,
and where it was caught. Closeout appends; nothing is deleted without a recorded reason.

- A test whose expected values are all the same constant passes for a constant implementation — use independent,
  distinct expected values and check against at least one deliberately wrong implementation. (caught by test-honesty
  review, factory e2e 2026-10-07)
- A phase that is too large does not converge under review. The M2 phase had 5 work packages, 26 gate entries and a
  980-line workflow; its serious findings went 12 -> 9 -> 6 -> 4 -> 5 over 5 rounds, and each round found a new kind of
  defect. Keep a phase to about 3 work packages and 12 gate entries, and split along the dependency order (first the
  mechanism the rest relies on). (caught by wan-next review, M2 phase, 2026-10-08)
- A phase workflow that re-implements the house machinery gets that machinery re-reviewed every round. The M2 phase-1
  workflow (924 lines, a 400-line FACTS block) wrote its own waves, audits, review loop and stabilize; its review went
  7 -> 6 -> 6 -> 1 -> 1 -> 3, nearly every finding about the wiring, and unfixed minors came back as majors. A phase is
  configuration for runPhase (lib/phase.js); validated minors are fixed with the serious findings; a defect in shared
  code is "house" and goes to the factory backlog. (caught by wan-next review, M2 phase 1, 2026-10-08)
- More review rounds do not shrink a document that is too big. The hetero ADR reached 2006 lines; its majors went
  17 -> 17 -> 19 while each fix round got slower (22, 30, 54 minutes). A review whose blocker/major count does not fall
  for three rounds is stalled: split the work into one-decision pieces and review each. (caught by wan-adr review,
  hetero ADR v2, 2026-10-08)
- A review whose fixes may adopt new ideas never converges: each round reopens the design. The hetero ADR-0002 adopted a
  reviewer's mechanism in round 2, and round 3 filed the text it left behind in four documents (7 -> 12 majors). From
  round 3, wan-adr fixes correct without redesigning; new ideas are recorded as follow-ups. (caught by wan-adr split
  review, hetero ADRs, 2026-10-08)
- A phase workflow must normalize the paths writers report (absolute or ./-prefixed) and count a writer's own journal,
  which its prompt tells it to write, as its own; and an audit over the whole run breaks a run resumed after a
  human-granted amendment (ADR, NEXT, approvals), counting the human's edits as the phase's. runPhase does the first and
  audits every write where it happens instead of the second. (caught by M2 phase 1, 2026-10-08)
- An agent cannot wait for a command longer than its shell's 10-minute limit. M2 phase 2's verifier started the
  hours-long pinned gate in the background and ended; the verdict was lost and the gate ran again by hand. A long gate
  runs detached and stores its verdict by tree (pinned.mjs --wait), and the script owns the wait in short agent slices,
  so an agent that ends early or a resumed run loses nothing. (caught by M2 phase 2, 2026-10-09)
- A ceremony paid per small phase costs more than the work. M2 phase 1 (9 files) took ~14 h and phase 2 (6 files) ~6 h
  plus hours of gate: a NEXT, GATES and workflow per phase, 3 grants, up to 5 plan-review rounds, up to 5 code-review
  rounds, and the heavy gate on every stabilize round. ~20 of the first 27 commits on main were factory work. The reference
  PR (oven-sh/WebKit#249) paid its ceremony once per milestone: one handout, one run over the whole DAG, a fail-fast ladder
  per fix round, the heavy gates at the end. Factory v2 does the same. (caught by M2 phases 1-2, 2026-10-09)
- A human's scope must reach every agent that judges the plan, whole. M2's single plan carried a 2117-character focus
  (5 items, a cipher decision, criteria from another session); wan-handout cut it to 600 for the planner and never showed
  it to the reviewers, who would have filed the deferred items as missing. The focus now goes whole (up to 8000
  characters) to the planner, the reviewers and every revision. (caught before the M2 handout drafted, 2026-10-09)
- A milestone id is whatever the granted design names. ADR-0006 split M2b off M2, and wan-handout refused "M2b" because
  four tools hard-coded /^M\d{1,3}$/. The tools now accept one letter after the number (M2b), and refuse anything else.
  (caught by the first /wan-run M2b, 2026-10-09)
- A workflow run by name executes the copy loaded when the session started. After #28 and #30 merged, c13's
  Workflow({name: "wan-handout"}) still ran the old script and refused M2b. Launching from the file
  (scriptPath: ".claude/workflows/wan-handout.js") ran the current one. /wan-run and CLAUDE.md now launch by path.
  (caught by the first /wan-run M2b, 2026-10-09)
- One agent output cannot carry a whole plan. extractPlan asked a read-only agent to return PLAN-M2b.json (95 KB) as
  structured output. Twice the output was cut at ~54 KB (wf_b98aae79-ea6, wf_31c4d57b-b79), the agent fell back to
  {"tasks": []}, and wan-implement stopped with "plan has no tasks". The plan is now copied in pieces: an index of ids,
  then 4 tasks, 12 binding items or 6 tripwires per parallel call. A piece short twice stops the run, instead of the plan
  silently shrinking. (caught by the first wan-implement M2b, 2026-10-09)
- The reviews found the defect, and the fix undid it. In M2's single plan, round 1 condemned a verdict rule that could
  not discriminate, and security coupled to the wave chain. Round 2 asked for the experiment to run first. The unreviewed
  final fix met that by making all 16 other tasks depend on the experiment, and the vacuous rule survived. The build
  stopped on its first task. Now every fix sees the earlier rounds' findings, and after the last fix one agent checks
  every finding against the current state and fixes what is open or came back. (caught by wan-implement M2, 2026-10-09)
- The build was more rigid than the reference PR, and the rigidity is what stopped it. In one day, M2 stopped its whole
  build on a spec conflict (the experiment) and then stopped 3 of 6 wave-1 tasks on spec conflicts. In the reference PR
  (thread-ungil) a coder records a spec conflict and continues, with the handout as the authority and the SPEC of record
  winning on conflict. Ownership there is an instruction, not an audit; reviewers report blocker/major only; and the build
  and the ladder verify everything at the end. wan-implement now works the same way and returns the recorded conflicts to
  the human. The heavy gate at closeout is unchanged. (the human's call after M2's wave 1, 2026-10-09)
- The agent that finds an error fixes it (the human's rule, 2026-10-09). Handing a finding to another agent made it
  re-read everything the finder already held: an author amending after its reviewers, a proposer and 3 voters and an
  applier after the triage, a fixer after the whole-change reviewers. Now each finder fixes what it finds: task
  reviewers in turn, the builder, each whole-change lens and the wan-fix agent. The voters' job, refusing a fix that
  masks a failure, is a rule every fixer carries (NO_MASKING) and the closeout's guarded gates check.
- A machine lock inside a checkout locks nothing across checkouts. machine-lock.mjs kept .factory/machine.lock in each
  checkout, so c8 (~/edt-factory-run) and c13 (~/edt-fsdp) never saw each other. Their builds ran test clusters side by
  side, and the 16 GB Mac panicked at 15:44 (watchdog timeout, jetsam, disk full of swap). The locks now live in one
  machine-wide directory with N test slots, and every agent's test command, every ladder rung and every verdict gate
  goes through them. Many agents can run in parallel; only their test runs queue. (2026-10-09)
- A ledger written only after the last wave loses a stopped run's work. wan-implement wrote .factory/runs/<m>/state.json
  at integrate. When c8's M2 run was stopped mid-way to pick up a factory change, its ledger was the one an older run had
  written, so the carry run wrote resume-ownership, replan-solver and wave-layout a second time. The ledger is now written
  after every wave, with the recorded spec conflicts, and carry brings both back: a stop loses at most the wave in
  flight. (2026-10-09)
- A plan's depth is its files, not its work. M2 ran its last 7 tasks in 5 waves because 4 of them wrote coordinator.py;
  M2b's plan had 13 waves for 19 tasks because tests/unit/test_runtime_spec.py and test_engine_bitwise.py had 7 owners
  each and stage.py 5. Each wave costs a coder and a reviewer in series. Plans now have at most 2 waves (the human's
  rule): one task per file, every task with its own test file, wave 2 only for integration; wan-handout refuses a deeper
  plan and names the files that chain it. (2026-10-09)
- A light test with no timeout can hold a test slot forever. stage-p2p-secure's mutation run applied a mutant that let an
  unpinned stage wait for peers that never came: pytest sat at 0% CPU for 20 minutes holding one of the two slots, with the
  mutant still written in stage.py. Slot runs are now killed with their whole process group after 1200 s and exit 124; the
  ladder's rungs already had their own timeouts. (2026-10-09)
- A campaign's per-run timeout must fit the command it amplifies. M2's amplify-chaos ran the whole tests/chaos suite
  (88 s clean) 200 times per period under amplify's default 60 s timeout: every run was a false HANG, and the heavy gate
  held the whole machine for 4 h 20 min of an 8 h budget before it was stopped. amplify now runs one clean calibration run
  first and stops (exit 2) when it hangs, fails or takes over half of --timeout. On the 16 GB Mac the campaign runs 30
  per period with a 300 s timeout (the human's call, 2026-10-10); the 200-per-period bar runs on the GPU fleet before the
  M1 sign-off. (2026-10-10)
- A test that must own the whole machine checks WHOSE whole-machine run it is in, not only that one exists: in a verdict
  gate every ladder rung also holds the machine (FACTORY_MACHINE_LOCK_HELD=ladder:<rung>), so M2b's E4 wall-clock arms
  (~2 h, their own pinned entry e4-exposure-sweep) ran inside rung integration's 3600 s step and would have failed the
  gate on a timeout, not on code (run wf_4c4dfd30-00d, stopped by the orchestrator). They now skip under `ladder:*`. (2026-10-10) The r0 gate then ran them inside pinned:f1-matrix-x20, which holds the
  machine too: the rule is an allow-list (the test's own labels: pinned:e4-exposure-sweep, e4*, m2b-e4*), never a
  deny-list of the holders seen so far. (2026-10-11)
- A verdict gate's second pass must rerun only what a rerun can refute. M2b's closeout (2 consecutive passes, `full`
  beside unit, integration, multiworker, oracle, chaos, wan and fuzz) ran every test four times, once in its folder's rung
  and once in `full` per pass, and every static check (lint, types, fmt, claims, audit tools) twice on the same tree; and a
  red rung still let hour-long campaigns and mutation testing run on code already known broken. Now a "deterministic"
  rung runs in pass 1 only, `full` runs in the last pass only in place of the rungs it contains ("coveredBy": "full";
  holding the machine when one of them is heavy), so each test runs once per pass, and campaigns and mutation are
  skipped once the gate is red ("runAll": true runs them). (2026-10-10)
- A test that must own the whole machine checks WHOSE whole-machine run it is in, not only that one exists: in a verdict
  gate every ladder rung also holds the machine (FACTORY_MACHINE_LOCK_HELD=ladder:<rung>), so M2b's E4 wall-clock arms
  (~2 h, their own pinned entry e4-exposure-sweep) ran inside rung integration's 3600 s step and would have failed the
  gate on a timeout, not on code (run wf_4c4dfd30-00d, stopped by the orchestrator). They now skip under `ladder:*`. (2026-10-10)
- A campaign that runs pytest gives it a private --basetemp: the default one is shared by every pytest of the same user,
  and a concurrent build's pytest rotated it away mid-run (py-devmode r999 on M2b: 174 of 358 reports were
  FileNotFoundError setup errors, not leaks, and those tests never ran). (2026-10-11)
- Mutation testing runs its command in a materialized copy of the tree, where `uv run` syncs only the default
  dependencies. M2b's closeout mutation stage died before any mutant with "No module named 'safetensors'"
  (tests/unit/test_finetune.py needs the hf extra), and the gate counted it as a failure of the code. The command now runs
  `uv run --all-extras`, the same environment the build rung syncs (`uv sync --all-extras`). (2026-10-11)
- A running session keeps the environment it started with, so FACTORY_TEST_SLOTS could not widen the test slots of
  sessions already working: two factory sessions (M2b stabilize and M4 implement) queued their tests behind 2 slots
  while the Mac sat idle. The slot count now also comes from the machine-wide file ~/.factory-locks/slots, read on every
  call. The human set the 16 GB Mac to 4 slots on 2026-10-11, and the orchestrator narrows it again under memory
  pressure. (2026-10-11)
- A gate whose cheapest checks run last learns its cheapest failures last. M2b's closeout ran 9 h of rungs and
  repetitions before reporting that docs/security/AUDIT-RESULTS.json was missing (a file read) and that the mutation
  command could not import an extra (its baseline). pinned.mjs now runs a preflight first: the audit results, every
  campaign's applicability and, when mutation testing applies, its baseline (reused by the mutation stage). A red
  preflight stops the gate before any rung, with every other gate
  skipped, and the fix then runs the gate in full. Its CLI tests also got their own lock directory: they had waited up to
  an hour behind the real machine's test slots. (2026-10-11)
- `machine-lock run ... -- <command...>` joined its arguments with spaces, so `pytest -k "a or b"` reached bash as
  `-k a or b` and every M4 verify command with such a selection failed before pytest started. One argument after `--` is
  still a shell line run as written; several are now an argv whose elements are quoted. (2026-10-11)
- A lock created first and written after is empty for a moment, and a second claimer read that empty file as stale,
  removed it and took the same slot. Four M4 slot runs woken together ended with two of them running and no slot file,
  so the machine looked free to a heavy run. The lock is now written to a temp file and hard-linked into place, so it
  holds its body from the moment it exists. An unreadable lock younger than 10 s counts as held, because other
  checkouts still run the old copy of the tool. (2026-10-11)
