# Experiments: local first, then Vast.ai

Architecture hypotheses are tested cheaply before anyone pays for compute. They are judged by arithmetic, not
by how confident an agent sounds. Paid cloud runs happen only behind human gates that are enforced in code.

## Sequence

```
ADR (hypothesis + first-principles model + falsification criteria)    wan-adr; a human grants it
  ↓
Experiment plan with predictions derived from the model                 wan-experiment: Plan
  ↓
Local prototype (experiments/<id>/, scope-audited)                      wan-experiment: Prototype
  ↓
Local correctness vs the reference · performance · failure injection    wan-experiment: Local (verifier)
  ↓
Script classification + promotion checklist → LOCAL.json                VALIDATED / PARTIALLY VALIDATED / REFUTED / INCONCLUSIVE
  ↓
PROMOTION GATE                                                          every checklist item true, ADR granted, plan + budget granted
  ↓
Vast.ai run through agents-architetura/tools/vast/vast.mjs                                 wan-vastai: estimate → launch → preflight → start → monitor → stop
  ↓
Expected vs observed → RESULTS.json + RESULTS.md                         a refuted hypothesis goes back to first principles
```

## Classification (`agents-architetura/tools/wf/lib/experiment.js`)

The plan predicts each metric. Metrics listed in `lowerIsBetter` count as met when they come in at or under the
prediction; the rest count as met when they reach it. `tolerancePct` sets the allowed slack in both cases. It applies only
to performance metrics. Correctness is never a tolerance: the training state must be byte-identical to the reference
(CLAUDE.md, "Product rule"). A correctness failure fails `localCorrectness` and blocks promotion, whatever the metrics say.

| Classification | Meaning |
|---|---|
| VALIDATED | Every metric meets its prediction within tolerance |
| PARTIALLY VALIDATED | The primary metric (the one the falsification criterion names) meets its prediction; at least one other metric does not |
| REFUTED | The primary metric misses, or the ADR's falsification criterion fired |
| INCONCLUSIVE | A predicted metric was not measured, so the run did not measure what it promised |

Failed hypotheses are not rationalized. A REFUTED result is reported as such, recorded in
`agents-architetura/docs/LESSONS.md`, and sent back to first-principles analysis (a revised or withdrawn ADR).

## Promotion checklist (recorded in `docs/experiments/<id>/LOCAL.json`)

The checklist has seven items. The script derives each one from the evidence:
- `hypothesisDefined`
- `architectureApproved`
- `localPrototypeWorks`
- `localCorrectness`
- `testsPass`
- `failureCasesTested`
- `improvementDemonstrated`

A paid run also needs the Vast.ai environment validated (the pre-flight) and the cost estimated (`vast.mjs
estimate`). It needs the exact plan and budget approved by a human: `node agents-architetura/tools/factory/approve.mjs grant
docs/experiments/<id>/PLAN.json`. Editing the plan afterwards voids that approval.

## Plan fields (`docs/experiments/<id>/PLAN.json`)

| Field | Purpose |
|---|---|
| `id`, `hypothesis` | Identity and the hypothesis under test |
| `expected` | Map of metric → predicted value |
| `lowerIsBetter` | Metrics where smaller is better |
| `primary` | The metric the falsification criterion names |
| `tolerancePct` | How far a performance observation may miss and still count (never applied to correctness) |
| `local.correctness` | Command that exits 0 only if the training state is byte-identical (sha256) to the synchronous reference |
| `local.performance` | Command whose last line is JSON `{metric: observed}` |
| `local.failure` | Failure-injection command that exits 0 only if the behavior is as specified |
| `maxBudgetUsd`, `warnAtPct`, `expectedHours`, `expectedTokens` | The cost gate (cost per million tokens is reported) |
| `instances: [{role, count, query, maxDph}]` | `query` is a vastai offer filter; `maxDph` is the price cap per instance-hour |
| `image`, `diskGb`, `setup`, `run` | What runs on the instances |
| `metrics` | Command printing JSON lines: `step`, `loss`, `restarts`, `ts` |
| `results` | Command printing the observed metrics at the end |
| `startupCheck`, `peerCheck` | Critical pre-flight checks: product startup and worker-to-worker connectivity, latency and bandwidth |
| `stop: {maxLoss, maxRestarts, stallMinutes}` | Emergency-stop thresholds |

## Guards in `agents-architetura/tools/vast/vast.mjs`

- **Launch** refuses when:
  - the plan has no current human grant;
  - there is no promotable local result, or the checklist is incomplete;
  - instances from an earlier launch are still alive;
  - the estimate, built from the cheapest offers under the price cap, exceeds the budget.
- **Preflight** checks, per instance:
  - GPU visibility and memory, driver, CUDA, PyTorch;
  - CPU/RAM, disk, network;
  - the product's startup check and peer connectivity.
  
  If a critical check fails, the result is "DO NOT START TRAINING" and `start` refuses.
- **Monitor** computes spend from the ledger (instance price × elapsed time) and reads health and the plan's
  metrics:
  - at `warnAtPct` of the budget it pauses: instances are stopped with their disks kept, and a human decides;
  - it emergency-stops on any of: the budget cap, NaN/Inf loss, loss above `maxLoss`, restarts above
    `maxRestarts`, no progress for `stallMinutes`, an instance vanished or not running, or no metrics at all.
- **Stop** first preserves `run.log`, the metrics and `nvidia-smi` output for every instance to
  `.factory/vast/<id>/logs/`, then destroys every instance. It then confirms that none remains.
- **Agents** run all of this only through the wrapper: `.claude/settings.json` denies `vastai create` and
  `vastai launch`. The API key stays in the human's environment.

**Unverified so far:** the real `vastai` adapter (`agents-architetura/tools/vast/client.mjs`) follows the vastai CLI, but it has not
yet been exercised against the live API from this repository. Every guard above is tested with the fake client.
Run `node agents-architetura/tools/vast/vast.mjs doctor` (it spends nothing) before the first paid experiment.
