# ADR-0004: Peer authentication on the wide-stage mesh is deferred out of M2 phase 1

<!-- Approval lives in docs/design/APPROVALS.json (agents-architetura/tools/factory/approve.mjs), bound to this exact text. Do not write a
     Status line claiming approval. Mark the ADR "Status: REJECTED", "Status: SUPERSEDED" or "Status: WITHDRAWN" only
     when that is the human's decision. -->

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE DECISION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: the review of M2 phase 1 (wan-x-m2-p1-liveness-telemetry, run wf_8f18b5aa-df4, resumed after ADR-0003)
stopped on a spec conflict with one validated finding that predates the phase. The wide-stage mesh data plane does not
authenticate its peers: `_connect_mesh` accepts a socket, reads the HELLO and trusts its `(st, pe)`
(src/wantrain/runtime/stage.py:541-551), then picks the link with `LinkSpec(**s.links[st])`. The chain path runs
`authenticate(sock, key, trusted_peers=...)` on every socket (stage.py:267-279). THREAT_MODEL T3 claims ed25519 plus
`trusted_peers` for the whole data plane. The same code is on `main`; this phase only derives the hop label from the key.
Current bottleneck: data-plane identity on wide stages; not speed, and not this phase's liveness or telemetry contract.
Proposed architecture: DEFER to a later phase owned by the transport/runtime workstream, recorded as finding
M2-P1-MESH-AUTH (open, major) and as a residual of THREAT_MODEL T3. That phase runs `authenticate` with
`trusted_peers` on every mesh socket before the HELLO, binds the HELLO's `(st, pe)` to the authenticated identity and
the spec's endpoint table, and rejects an `st` outside {rank − 1, rank} or a duplicate key.
First-principles reasoning: the fix is a data-plane protocol change (a handshake on a path that has none), which
NEXT-m2-p1-liveness-telemetry rules out of this phase. Deferring it changes nothing this phase delivers: the gap exists
with or without the phase, and the phase's telemetry adds no new trust in the HELLO beyond a label.
Mathematical justification: none needed for the deferral. For the later fix: one sign and one verify per mesh
connection per attempt, the cost the chain path already pays; ~0 of the step time (HYPOTHESIS).
Expected improvement: the phase can reach its verdict on its granted scope; the gap is recorded instead of silent.
Alternatives: (1) fix it in this phase — rejected for scope: a protocol change with its own tests and gates in a phase
already at the size limit (LESSONS.md, "divide always"); (2) do nothing and leave it unrecorded — rejected: T3 would keep
claiming a mitigation the mesh does not have.
Risks: until its phase, on a wide stage a host that can reach a stage's mesh port can join as a claimed `(st, pe)`
without an identity check, and feed wrong activations or gradients: a corrupt run. DESIGN already accepts that a
malicious participant can corrupt a run undetected until M4 (audit wiring, CONTROL-I13).
Falsification criteria: the later phase's test dials a mesh port with a key outside `trusted_peers`, or claims an `st`
outside {rank − 1, rank}, and must be refused; today that test would fail.
Local experiment: none for the deferral; the later phase's gates (CPU, gpu false).
Estimated Vast.ai cost: none: local only.
Recommendation: PROCEED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Awaiting explicit human approval: node agents-architetura/tools/factory/approve.mjs grant docs/design/adr/ADR-0004-m2-p1-mesh-auth-deferred.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Amendments (applied with this ADR)
- THREAT_MODEL T3 residual: the wide-stage mesh path authenticates no peer (finding M2-P1-MESH-AUTH).
- agents-architetura/docs/findings.json: M2-P1-MESH-AUTH, open, major.
- The phase workflow's review and fix prompts name this ADR, so the resumed run reuses its cached work while its review
  does not re-file the finding.

## Migration Impact
None: no code changes with this ADR.

## Human Approval
Granted only by `node agents-architetura/tools/factory/approve.mjs grant` in a human's terminal, or `/wan-approve`;
recorded in docs/design/APPROVALS.json.
