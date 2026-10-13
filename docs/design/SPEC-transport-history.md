# SPEC-transport history

- 2026-10-09 — ADR-0006 round 2 (exact sharded stages; awaiting the human grant). New invariant TRANSPORT-I8
  (DEFERRED-TO: M2b) and task 3: a shared-pacer mode for the shim (per-peer egress and ingress budgets, per-site-pair
  uplink budgets), needed by ADR-0006 E4 to measure concentrated layouts.
- 2026-10-09 — HANDOUT-M2 §8 A9 (granted by the human with HANDOUT-M2, sha256 d2e1c1f4…), recorded on the conflict
  raised by task `identity-handshake`. Precondition P1 assigned this recording to the researcher and it had not been
  done, so the frozen text still described the HMAC/shared-token fallback while the granted plan, its tests and
  `identity.py` implement the amended invariant. Applied verbatim: §1 peer identity and record protection; §2 Identity
  (domain-, challenge- and ephemeral-bound handshake, `expect_peer`, no shared-token mode) and new Records bullet;
  §2 Control channel (control-domain handshake, sealed, transcript-bound `token_mac`); §3 Provides (new `authenticate`
  signature, `SecureSocket`, `hardware_aes`, `control_dial`, `control_accept`, `session_mac`, `check_session_mac`);
  TRANSPORT-I5 replaced; TRANSPORT-I7 undeferred and replaced (AES-256-GCM with hardware AES else ChaCha20-Poly1305,
  HKDF per-direction keys, counter nonces, deterministic rekey); TRANSPORT-I9 added; tasks 1–2 moved into M2.
- 2026-10-11 — ADR-0009 round 2 (awaiting the human grant). §2 frame types: `TIED` added (fp32 rows of a tied table
  between stage 0 and the Head, ids derived from `make_batch` and never sent, forwarded fragment by fragment).
- 2026-10-11 — ADR-0009 round 5 (awaiting the human grant; mechanism frozen, text corrected). §2 `TIED`: a frame under
  a plan with no tied relay is fatal at receipt (RUNTIME-I19); the type goes with ADR-0009's runtime task on a rollback.
- 2026-10-11 — ADR-0009 round 6 (correction pass, autonomy). §2 `TIED`: its rollback target is stated: the type id
  is left unassigned, so typed decode refuses such a frame as an unknown type.
- 2026-10-11 — M4 implement deviation (transport-tied). §2 `TIED`: HANDOUT I-5's `paylen == rows·h·4` read per frame
  would refuse every fragment, since a fragment carries the whole shape and a byte range of the payload
  (`runtime/stage.py` send loop, `fr.send_frame(..., shape=shape, frag_idx=i, frag_cnt=len(frags))`). Recorded as
  implemented (`transport/framing.py:54-58` `_tied_bytes`, `:152-154` and `:166-167` in `Reassembler.push`,
  `:193-198` in `recv_frame`): an unfragmented frame needs `paylen == rows·h·4`, a fragment `paylen ≤ rows·h·4` (both
  at receipt, before the payload is read), the reassembled payload exactly `rows·h·4`; a 0-row frame is one empty
  fragment (the identity-m4 E4 a3/a4 fix, `stage.py` `frags = [...] or [b""]`). HANDOUT I-5 amended alike.
