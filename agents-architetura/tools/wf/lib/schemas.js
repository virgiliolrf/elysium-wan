// JSON Schemas for agent() structured output. Every factory agent returns one
// of these — the script never parses free text.
const RESULT = {
  type: 'object',
  required: ['summary', 'files'],
  properties: {
    summary: { type: 'string' },
    files: { type: 'array', items: { type: 'string' }, description: 'files created or modified' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

// Adversarial findings. A finding counts only if the reviewer validated it against the CURRENT code
// (validatedAgainstCurrentCode=true) and gave a location, evidence and a reproduction.
const FINDINGS = {
  type: 'object',
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['severity', 'location', 'problem', 'evidence', 'reproduction', 'recommendedFix', 'validatedAgainstCurrentCode'],
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          location: { type: 'string', description: 'file:line (or document section) where the defect is' },
          problem: { type: 'string' },
          evidence: { type: 'string', description: 'the code/text you read that proves it, quoted' },
          reproduction: { type: 'string', description: 'a concrete input, interleaving or command that exhibits the defect' },
          recommendedFix: { type: 'string' },
          validatedAgainstCurrentCode: { type: 'boolean', description: 'true only if you re-read the current file and the defect is still there' },
          scope: { type: 'string', enum: ['change', 'house'], description: '"house" only for a defect in shared code this change did not write and does not need changed to be correct (a factory lib, the template, code another phase owns): it goes to the factory backlog and does not block this review. Anything this change wrote, or must change, is "change" (the default).' },
        },
      },
    },
  },
}

const LADDER_RESULT = {
  type: 'object',
  description: 'the exact JSON printed by `node agents-architetura/tools/gates/ladder.mjs --json`, unmodified',
  required: ['allGreen', 'rungs'],
  properties: {
    allGreen: { type: 'boolean' },
    rungs: {
      type: 'array',
      items: {
        type: 'object',
        required: ['rung', 'status'],
        properties: {
          rung: { type: 'string' },
          status: { type: 'string', enum: ['pass', 'fail', 'n/a', 'skipped'], description: 'n/a = not applicable to this project; skipped = blocked by an earlier failing rung' },
          detail: { type: 'string' },
        },
      },
    },
    missingRequired: { type: 'array', items: { type: 'string' } },
  },
}

const TRIAGE = {
  type: 'object',
  required: ['allGreen', 'ladder', 'items'],
  properties: {
    allGreen: { type: 'boolean', description: 'true only if the ladder run you pasted has allGreen=true and there is nothing to fix' },
    ladder: LADDER_RESULT,
    items: {
      type: 'array',
      description: 'independent fix items with DISJOINT file scopes',
      items: {
        type: 'object',
        required: ['id', 'rung', 'symptom', 'evidence', 'scope'],
        properties: {
          id: { type: 'string' },
          rung: { type: 'string', description: 'ladder rung id that is failing' },
          symptom: { type: 'string' },
          evidence: { type: 'string', description: 'exact error output / failing test ids / repro command' },
          scope: { type: 'array', items: { type: 'string' }, description: 'repo-relative files (or dirs ending in /) the fix may touch' },
          suspectedCause: { type: 'string' },
          mechanism: { type: 'string', description: 'the root-cause mechanism in a few words — the SAME words for every failure that shares a cause' },
          ownerTask: { type: 'string', description: 'task that owns the failing file, from: node agents-architetura/tools/wf/ledger.mjs owner --path <file>' },
          ownerJournal: { type: 'string', description: 'that task\'s journal path, from the same command' },
        },
      },
    },
    note: { type: 'string' },
  },
}

const PROPOSAL = {
  type: 'object',
  required: ['fix', 'rationale'],
  properties: {
    fix: { type: 'string', description: 'exact change as old->new snippets per file, NOT applied' },
    rationale: { type: 'string', description: 'why this resolves the root cause, not the symptom' },
    rootCauseOutsideScope: { type: 'string', description: 'set if the true cause is outside the allowed scope — names the file' },
  },
}

const VOTE = {
  type: 'object',
  required: ['approve', 'reasons'],
  properties: {
    approve: { type: 'boolean' },
    reasons: { type: 'string' },
    amendment: { type: 'string', description: 'approve-with-changes: the exact changed snippet' },
  },
}

const WORKSTREAMS = {
  type: 'object',
  required: ['workstreams'],
  properties: {
    workstreams: {
      type: 'array',
      items: {
        type: 'object',
        required: ['key', 'title', 'scope', 'ownedPaths'],
        properties: {
          key: { type: 'string' },
          title: { type: 'string' },
          scope: { type: 'string' },
          ownedPaths: { type: 'array', items: { type: 'string' } },
          provides: { type: 'array', items: { type: 'string' } },
          consumes: { type: 'array', items: { type: 'string' } },
        },
      },
    },
  },
}

const APPROVAL_CHECK = {
  type: 'object',
  description: 'the exact JSON printed by node agents-architetura/tools/factory/approve.mjs check ... --json',
  required: ['approved', 'docs'],
  properties: {
    approved: { type: 'boolean' },
    docs: { type: 'array', items: { type: 'object', required: ['doc', 'approved'], properties: { doc: { type: 'string' }, approved: { type: 'boolean' }, reason: { type: 'string' } } } },
  },
}


const PLAN = {
  type: 'object',
  required: ['milestone', 'tasks'],
  properties: {
    milestone: { type: 'string' },
    binding: {
      type: 'array',
      description: 'binding audit: every instance of the hazard this milestone touches, each with a ruling, before any code',
      items: {
        type: 'object',
        required: ['id', 'what', 'where', 'ruling'],
        properties: {
          id: { type: 'string', description: 'BA-<n>' },
          what: { type: 'string' },
          where: { type: 'string', description: 'file:line or symbol' },
          ruling: { type: 'string', description: 'what happens to it, e.g. rebuilt at the checkpoint boundary / carried unchanged / refused' },
        },
      },
    },
    tripwires: {
      type: 'array',
      description: 'multi-leg changes that stay refused in the product until an arm task that depends on every leg lifts the refusal',
      items: {
        type: 'object',
        required: ['id', 'flag', 'legs', 'arm', 'test'],
        properties: {
          id: { type: 'string' },
          flag: { type: 'string', description: 'what stays refused (the switch or entry point that raises until armed)' },
          legs: { type: 'array', items: { type: 'string' } },
          arm: { type: 'string', description: 'task id that lifts the refusal; depends on every leg' },
          test: { type: 'string', description: 'command proving the refusal holds; every leg verifies it' },
        },
      },
    },
    tasks: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'title', 'objective', 'workstream', 'deps', 'scope', 'acceptance', 'verify', 'rungs', 'risk'],
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          objective: { type: 'string' },
          verify: { type: 'array', items: { type: 'string' }, description: 'exact commands that prove the acceptance criteria' },
          workstream: { type: 'string', description: 'workstream key, or "integrate" for shared hot files' },
          deps: { type: 'array', items: { type: 'string' } },
          scope: { type: 'array', items: { type: 'string' } },
          acceptance: { type: 'array', items: { type: 'string' } },
          rungs: { type: 'array', items: { type: 'string' } },
          invariants: { type: 'array', items: { type: 'string' }, description: 'INV-* ids this task makes true' },
          binding: { type: 'array', items: { type: 'string' }, description: 'BA-* binding audit items whose ruling this task delivers' },
          risk: { type: 'string', enum: ['low', 'medium', 'high'], description: 'high = touches a trust boundary, concurrency, the training math or recovery; high-risk tasks get a security review of their diff' },
        },
      },
    },
  },
}

const AUDIT = {
  type: 'object',
  required: ['tree'],
  properties: {
    tree: { type: 'string', description: 'tree id printed by scope-audit (snapshot) or the "now" tree (check)' },
    changed: { type: 'array', items: { type: 'string' } },
    violations: { type: 'array', items: { type: 'string' } },
  },
}

const BUILD_REPORT = {
  type: 'object',
  required: ['ladder', 'files'],
  properties: {
    ladder: LADDER_RESULT,
    errorLog: { type: 'string', description: 'where the full raw error log was saved' },
    files: {
      type: 'array',
      description: 'one entry per source file with errors (link errors map to the file owning the symbol)',
      items: {
        type: 'object',
        required: ['file', 'errors'],
        properties: { file: { type: 'string' }, errors: { type: 'array', items: { type: 'string' } } },
      },
    },
  },
}

const EVIDENCE = {
  type: 'object',
  required: ['reproduced', 'summary'],
  properties: {
    reproduced: { type: 'boolean' },
    reproCommand: { type: 'string' },
    rate: { type: 'string', description: 'failures / runs observed, e.g. "7/200"' },
    summary: { type: 'string' },
  },
}

const HYPOTHESES = {
  type: 'object',
  required: ['hypotheses'],
  properties: {
    hypotheses: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'mechanism', 'interleaving', 'confirmIf', 'refuteIf'],
        properties: {
          id: { type: 'string' },
          mechanism: { type: 'string', description: 'file:line-grounded cause' },
          interleaving: { type: 'string', description: 'the concrete sequence of events that produces the evidence' },
          confirmIf: { type: 'string', description: 'a runnable observation that would confirm it' },
          refuteIf: { type: 'string', description: 'a runnable observation that would refute it' },
        },
      },
    },
  },
}

const VERDICT = {
  type: 'object',
  required: ['verdict', 'argument'],
  properties: {
    verdict: { type: 'string', enum: ['refuted', 'survives', 'confirmed'] },
    argument: { type: 'string' },
    experimentRequest: { type: 'string', description: 'optional: a discriminating experiment to run' },
  },
}

const EXPERIMENTS = {
  type: 'object',
  required: ['results'],
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        required: ['hypothesisId', 'outcome', 'detail'],
        properties: {
          hypothesisId: { type: 'string' },
          outcome: { type: 'string', enum: ['confirms', 'refutes', 'inconclusive'] },
          detail: { type: 'string' },
        },
      },
    },
  },
}

const CAUSAL_FIX = {
  type: 'object',
  required: ['hypothesisId', 'fix', 'happensBefore', 'files'],
  properties: {
    hypothesisId: { type: 'string' },
    fix: { type: 'string', description: 'exact old->new snippets, NOT applied' },
    happensBefore: { type: 'string', description: 'the ordering/causal argument that makes the failure impossible, not merely rarer' },
    files: { type: 'array', items: { type: 'string' } },
  },
}

const BUG_VERIFY = {
  type: 'object',
  required: ['fixed', 'runs', 'failures', 'ladder'],
  properties: {
    fixed: { type: 'boolean' },
    runs: { type: 'number' },
    failures: { type: 'number' },
    ladder: LADDER_RESULT,
    amplify: { type: 'object', description: 'the JSON printed by agents-architetura/tools/verify/amplify.mjs --json, verbatim' },
    load: { type: 'object', description: 'the JSON printed by agents-architetura/tools/verify/load.mjs --json, verbatim' },
    causal: { type: 'object', description: 'the JSON printed by agents-architetura/tools/verify/causal.mjs ab --json, verbatim' },
    detail: { type: 'string' },
  },
}

const CLASSES = {
  type: 'object',
  required: ['classes'],
  properties: {
    classes: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'mechanism', 'whereInDesign', 'susceptibilityTest'],
        properties: {
          id: { type: 'string' },
          mechanism: { type: 'string' },
          whereInDesign: { type: 'string', description: 'the design/spec section and code paths exposed to it' },
          susceptibilityTest: { type: 'string', description: 'the test that would demonstrate susceptibility' },
        },
      },
    },
  },
}

const REDTEAM_RUN = {
  type: 'object',
  required: ['results'],
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'outcome', 'evidence'],
        properties: {
          id: { type: 'string' },
          outcome: { type: 'string', enum: ['susceptible', 'resistant', 'broken-test'] },
          evidence: { type: 'string', description: 'exact failing output or the passing run summary' },
          reproCommand: { type: 'string' },
        },
      },
    },
  },
}

const PR_COMMENTS = {
  type: 'object',
  required: ['repo', 'comments'],
  properties: {
    repo: { type: 'string', description: 'owner/name' },
    comments: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'author', 'body'],
        properties: {
          id: { type: 'string' },
          author: { type: 'string' },
          path: { type: 'string' },
          line: { type: 'number' },
          outdated: { type: 'boolean' },
          body: { type: 'string' },
        },
      },
    },
  },
}

const CLOSEOUT_CHECK = {
  type: 'object',
  required: ['ladder', 'pendingMarkers'],
  properties: {
    ladder: LADDER_RESULT,
    pendingMarkers: { type: 'array', items: { type: 'string' }, description: 'file:line of every expected-failure/ignore marker still referencing this milestone' },
    benchRecorded: { type: 'boolean' },
  },
}

// What an implementer returns. "Done" is never a claim: it is the evidence list. specConflict is the
// STOP -> REPORT CONFLICT -> PROPOSE CHANGE half of the conflict protocol.
const TASK_RESULT = {
  type: 'object',
  required: ['summary', 'files', 'evidence'],
  properties: {
    summary: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    evidence: {
      type: 'array',
      description: 'commands you ran to check your own work, with their real exit codes and output',
      items: {
        type: 'object',
        required: ['claim', 'command', 'exitCode', 'output'],
        properties: { claim: { type: 'string' }, command: { type: 'string' }, exitCode: { type: 'number' }, output: { type: 'string' } },
      },
    },
    specConflict: {
      type: 'object',
      description: 'set ONLY if the spec/design is wrong or contradictory for this task; then stop and do not work around it',
      required: ['description', 'evidence', 'proposedChange'],
      properties: { description: { type: 'string' }, evidence: { type: 'string' }, proposedChange: { type: 'string' } },
    },
    journal: { type: 'string', description: 'path of the task journal you wrote' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

// A reviewer or fixer that fixes what it finds (the human's rule, 2026-10-09: the agent that finds an error fixes it).
const REVIEW_FIX = {
  type: 'object',
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        ...FINDINGS.properties.findings.items,
        properties: {
          ...FINDINGS.properties.findings.items.properties,
          fixed: { type: 'boolean', description: 'true if you fixed it in this pass (cite the file:line of the fix in evidence)' },
        },
      },
    },
    evidence: TASK_RESULT.properties.evidence,
    specConflict: { ...TASK_RESULT.properties.specConflict, description: 'set if the handout and the SPEC disagree on what you fixed; you followed the SPEC of record' },
    summary: { type: 'string' },
  },
}

// One round of wan-fix: the agent that runs the ladder fixes what it finds and pastes the last ladder run.
const FIX_ROUND = {
  type: 'object',
  required: ['allGreen', 'ladder', 'fixed', 'remaining'],
  properties: {
    allGreen: { type: 'boolean', description: 'true only if the LAST ladder run you pasted has allGreen=true' },
    ladder: LADDER_RESULT,
    fixed: { type: 'array', items: { type: 'object', required: ['mechanism', 'files', 'evidence'], properties: { mechanism: { type: 'string' }, files: { type: 'array', items: { type: 'string' } }, evidence: { type: 'string', description: 'the command that now passes and its output tail' } } } },
    remaining: { type: 'array', items: { type: 'object', required: ['rung', 'symptom', 'evidence'], properties: { rung: { type: 'string' }, symptom: { type: 'string' }, evidence: { type: 'string' } } } },
    specConflict: TASK_RESULT.properties.specConflict,
  },
}

// Independent per-task verification (run by the verifier, not the author).
const TASK_VERIFY = {
  type: 'object',
  required: ['results'],
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        required: ['taskId', 'passed', 'runs'],
        properties: {
          taskId: { type: 'string' },
          passed: { type: 'boolean', description: 'true only if every verify command exited 0' },
          runs: {
            type: 'array',
            items: {
              type: 'object',
              required: ['command', 'exitCode', 'output'],
              properties: { command: { type: 'string' }, exitCode: { type: 'number' }, output: { type: 'string' } },
            },
          },
        },
      },
    },
  },
}

// RE-EVALUATE half of the conflict protocol (the researcher decides; nobody edits silently).
const CONFLICT_DECISION = {
  type: 'object',
  required: ['decision', 'rationale'],
  properties: {
    decision: { type: 'string', enum: ['amend-spec', 'reject', 'replan', 'adr'], description: 'adr = the fix needs a MAJOR architecture change: write the ADR; implementation stops for human approval' },
    rationale: { type: 'string' },
    adr: { type: 'string', description: 'for adr: path of the ADR you wrote (docs/design/adr/ADR-NNNN-<slug>.md)' },
    amendment: { type: 'string', description: 'for amend-spec: what changed in the spec (already written, with a history entry)' },
  },
}


const WAVE_CHECK = {
  type: 'object',
  required: ['audit', 'results'],
  properties: { audit: AUDIT, results: TASK_VERIFY.properties.results },
}
