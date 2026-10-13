---
description: Run a milestone with the factory in one plan and one run, stopping only for the human's grant and merge
argument-hint: <milestone, e.g. M2> [focus: the slice of the milestone, when it is not the whole]
---
Run milestone `$ARGUMENTS` through the factory the way the reference PR's author ran a milestone (oven-sh/WebKit#249,
`thread-ungil-spec` then `thread-ungil`): ONE plan for every phase still open in the milestone, granted once, then ONE run
that implements the whole task DAG, then the heavy gate once. The human only answers your questions: the grant, which
Claude Code asks for, and the merge, which only they can do. Everything else runs without them.

The rules this follows (the human's, 2026-10-09, factory v2):
- **One plan, one run.** wan-handout plans every open phase of the milestone as one DAG. Only a milestone past 24 tasks is
  split: the plan is a self-contained first slice, and the handout lists the later slices.
- **At most 2 waves.** Wave 1 builds the pieces in parallel, wave 2 integrates them. One task per file, each task with its
  own test file: a shared file chains every task that touches it (M2 ran in 5 waves, M2b's first plan had 13).
- **Review is of the code.** The plan gets 2 review rounds that fix every validated finding, minors included. After that,
  each task gets 2 reviewers and an amend, and the whole change gets 2 rounds that fix everything.
- **Fast checks while working, the heavy gate once.** Coders run fmt, lint, types and the unit tests of their own files. Fix
  rounds use a fail-fast ladder, one root cause per round, at most 8 rounds. The milestone gate (`GATES-<M>.json`: two
  consecutive passes, repetitions under load, campaigns) runs only in wan-closeout.

Never run `git push`, `gh pr merge` or `approve.mjs grant`, and never edit `docs/design/APPROVALS.json`. Git goes through
`node agents-architetura/tools/factory/publish.mjs`, which commits, pushes and opens the PR with its own guards. If a step
below fails in a way it does not cover, stop and tell the human what happened. Never work around a gate.

**Launch every workflow from its file, never by name:** `Workflow({scriptPath: ".claude/workflows/<name>.js", args: {...}})`,
with `args` as a JSON object, not a string. A workflow run by name executes the copy loaded when this session started. A
factory change merged since then (a pull, a fast-forward) is silently missing: the first `/wan-run M2b` kept failing on the
old milestone check after main had the fix. To check what a run executed, grep its agent transcripts for text only the new
version has.

## 0. Preconditions
- `git status --porcelain` is empty. If not, show the paths and stop.
- You are on `main` (a new run) or on `factory/<id>` (resuming). Otherwise, stop.
- Resuming: if `node agents-architetura/tools/factory/approve.mjs pending --json` lists `docs/design/HANDOUT-<M>.md`,
  `docs/design/PLAN-<M>.json` and `docs/design/GATES-<M>.json`, the plan is written and awaits its grant: start at step 4.
  If `approve.mjs check` passes for all three, start at step 5.
- Run `node agents-architetura/tools/factory/hardware.mjs show` and tell the human the mode. With `gpu: false`, every
  number is SIMULATED.

## Steps
1. **Branch.** On `main`, run `node agents-architetura/tools/factory/publish.mjs start --task <milestone>` (lowercase, e.g.
   `m2`). On a `factory/` branch, keep it.
2. **Plan.** Run the `wan-handout` workflow with `{"milestone": "<M>"}`, adding `"focus": "<focus>"` when the human gave
   one. Wait for it to finish.
   - If `ok` is false, the plan or the gates are invalid after their repairs: show `gateErrors` or the error and stop.
   - Show the human the plan in one table: task, wave, risk. Say how many binding-audit items and tripwires it has.
   - If the handout lists "Later slices", say so: this run is the first slice.
   - If the review's last round filed findings, they were fixed without another review: list them (`review.open`) so the
     grant covers them.
3. **Publish the plan.** Run `node agents-architetura/tools/factory/publish.mjs commit --stage plan-<milestone>`.
4. **Grant, once, unless autonomy waives it.** First run `node agents-architetura/tools/factory/approve.mjs check
   docs/design/HANDOUT-<M>.md docs/design/PLAN-<M>.json docs/design/GATES-<M>.json`. If it passes (the autonomy switch,
   `agents-architetura/tools/factory/autonomy.json`, waives human grants), do not ask the human: go to step 5.
   Otherwise follow `/wan-approve` (`.claude/commands/wan-approve.md`) for the three documents:
   - HANDOUT and PLAN: the tasks, the waves, the binding audit and the tripwires;
   - GATES: the rungs (oracle, identity and full must be there) and every pinned entry with its bound.

   Give the path, then run exactly `node agents-architetura/tools/factory/approve.mjs confirm <doc> --sha <sha12>`, one
   document per command. Claude Code asks the human. A denial stops the run. Then
   `node agents-architetura/tools/factory/publish.mjs commit --stage grant-<milestone>`.
5. **Implement.** Run the `wan-implement` workflow with `{"milestone": "<M>"}` and wait for it to finish. Publish the stage:
   build the result with
   `node -e "process.stdout.write(Buffer.from(JSON.stringify({ok: <true|false>})).toString('base64'))"`, then run
   `node agents-architetura/tools/factory/publish.mjs commit --stage implement-<milestone> --b64 <that base64>`.
   - If `deviations` lists recorded spec conflicts, put them in the PR description and record each accepted amendment
     with `freeze.mjs amend`. Show them to the human and wait only when the autonomy switch does not waive grants.
   - If `stopped` names an ADR or a human approval, stop and say what is needed.
   - If `ladder.ok` is false after its rounds, show the failing rungs from the result and stop. Offer `/wan-bughunt` on
     the failure, or a rerun of wan-implement after a fix.
6. **Closeout: the heavy gate, once.** Run the `wan-closeout` workflow with
   `{"milestone": "<M>", "review": <wan-implement's review object>}`. The pinned gate runs detached for hours, and the
   workflow waits for it. Publish the stage as in step 5, with `--stage closeout-<milestone>` and
   `{ok, verdict: '<VERIFIED|NOT VERIFIED>'}`.
7. **Verdict.**
   - **VERIFIED:** run `node agents-architetura/tools/factory/publish.mjs ready --tree <the tree in the verdict's pinned JSON>`.
     Ask the human with AskUserQuestion, giving the PR URL: "Merged: next slice", "Merged: the milestone is done", or
     "Stop". Merging is theirs. After "Merged: next slice", run `git switch main && git pull --ff-only` and go to step 1
     with the focus set to the handout's next later slice.
   - **NOT VERIFIED:** show the failing gates from the pinned JSON and the result's `next`, and stop. Offer `/wan-bughunt`
     for a failure that survived the stabilize rounds or was intermittent.

## When you stop
Say which step you stopped at, why, the PR URL, and the exact command that resumes from there.
