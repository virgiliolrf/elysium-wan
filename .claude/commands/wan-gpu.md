---
description: Show or change the factory's hardware switch (gpu true/false); with GPUs on, set up Vast.ai by asking the human at every step
---
The hardware switch is the human's, in `agents-architetura/tools/factory/hardware.json`.
- `gpu: false` (the default): no GPU, no second machine, no Vast.ai. Every workflow validates by local simulation, and
  `vast.mjs` refuses every account action except `validate`, `status`, `monitor` and `stop`.
- `gpu: true`: GPU experiments on Vast.ai are allowed, only through `agents-architetura/tools/vast/vast.mjs` and only for an
  experiment plan and budget the human granted.

Your job is to show the human where things stand and ask before every step. Never decide for them.

1. Run `node agents-architetura/tools/factory/hardware.mjs show` and tell the human the current mode.
2. Ask the human, with the AskUserQuestion tool, whether to keep the mode, turn GPUs on or turn GPUs off.
3. **Turning GPUs off:** run `node agents-architetura/tools/factory/hardware.mjs set gpu false`.
   - For every `docs/experiments/*/PLAN.json`, run `node agents-architetura/tools/vast/vast.mjs status --plan <PLAN.json>`.
   - If any instance is live, ask the human whether to stop it. If they say yes, run
     `node agents-architetura/tools/vast/vast.mjs stop --plan <PLAN.json> --reason "GPUs turned off"`.
4. **Turning GPUs on:** do one step at a time, and ask before each one.
   1. Run exactly `node agents-architetura/tools/factory/hardware.mjs set gpu true`. Claude Code asks the human to allow
      it. If they deny it, the switch stays off: stop here.
   2. Check that the `vastai` CLI is installed (`vastai --help`). If it is missing, ask the human before installing it
      (`uv tool install vastai`).
   3. The account login is the human's. The Vast.ai API key never passes through you. Ask the human to run
      `vastai set api-key <their key>` in their own terminal, not in this chat, and wait until they say it is done.
   4. Run `node agents-architetura/tools/vast/vast.mjs doctor`. Claude Code asks the human first. It only reads the account
      and the GPU market. Report how many offers it saw.
   5. GPUs are rented only by an experiment: `/wan-experiment` with `"cloud": true`. Its `PLAN.json` sets the GPU types,
      count, hours, `maxBudgetUsd` and `warnAtPct`, and the human grants that exact plan with `/wan-approve`. List the
      pending plans with their budgets, and say that granting a plan is the permission to rent within that budget.
5. Finish with `node agents-architetura/tools/factory/hardware.mjs show`.

Never write `hardware.json` by hand. Never ask for, print or store the API key. Never run `vastai create` or
`vastai launch`. Never raise a budget. Never retry a step the human denied.
