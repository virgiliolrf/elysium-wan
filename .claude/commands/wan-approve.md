---
description: Approve pending factory decisions (design, ADRs, experiment plans, phase workflows) by answering Claude Code's question
---
Help the human decide on the factory's pending decisions. Approval is theirs; your job is to show what they approve.
Your summary is a reading aid, not the decision: always give the document's path, and ask the human to read the
document itself, which is what the grant binds by its sha256.

1. Run `node agents-architetura/tools/factory/approve.mjs pending --json`. If nothing is pending, say so and stop.
2. For each pending document, read it and summarize in at most 10 lines what approving it commits the project to:
   - for DESIGN.md, its open questions;
   - for an ADR, its decision;
   - for an experiment PLAN.json, the budget ceiling;
   - for a `wan-x-*` workflow, its owned paths and exit gate.

   Then give the first 12 hex characters of its sha256, from the pending output.
3. For each document the human wants, run exactly
   `node agents-architetura/tools/factory/approve.mjs confirm <doc> --sha <sha12>`, one document per command, with
   nothing before or after it. Claude Code asks the human to allow it. If they allow it, the approval hook records the
   grant and says so. If they deny it, it is not approved.
4. Finish with `node agents-architetura/tools/factory/approve.mjs check <doc...>` and report the result.

Never run `approve.mjs grant` (it is for the human's own terminal), never retry a denied confirm, and never edit
`docs/design/APPROVALS.json`.
