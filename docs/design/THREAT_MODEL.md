# Threat model: wantrain

Scope: the engine as merged (M0) and the hardening in milestone M4. Design sections refer to [DESIGN.md](DESIGN.md).

**Security rule (from the human, 2026-10-08): raise security from first principles with the smallest possible speed
cost.** Every mitigation states its cost against the dominant term of the step time (DESIGN §9). Mechanisms that cost
nothing come first: types, bounds, validation at the boundary, no code paths from data. A mitigation that would add
more than 1% to the measured step time in the `bench` rung needs an ADR that weighs it against the threat. Costs not
yet measured are labeled HYPOTHESIS, with the experiment that measures them.

## 1. Trust boundaries

| Boundary | Trusted side | Untrusted side | Crossing |
|---|---|---|---|
| B1 Operator ↔ coordinator | operator | — | local process, config, token in env |
| B2 Coordinator ↔ agents | coordinator | agents (rented machines on the internet) | control channel (TCP, JSON, token HMAC) |
| B3 Slot ↔ slot | each slot | neighbours, links | data plane (TCP frames, ed25519 or HMAC) |
| B4 Network bytes → objects | parsers | any host that reaches a port | `recv_frame`, `serial.loads`, `Channel.recv` |
| B5 Checkpoint waves | coordinator store | the agent that shipped them | files over the channel, sha256 |
| B6 Model and data inputs | engine | user `build(cfg)` code, HF checkpoints, corpora | `model_contract.validate`, strict `hf.load_into`, dataset manifests |

**Assumptions** (BRIEF): the operator runs the coordinator and chooses which workers to admit. Workers may be rented
and may be adversarial. Links are public. A user model's code is trusted by its operator: it runs in the operator's
processes on purpose.

## 2. Threats

| ID | Layer | Threat | Scenario | Impact | Mitigation (design section) | Residual risk | Verifying test |
|---|---|---|---|---|---|---|---|
| T1 | B4 | Code execution from network bytes | A peer sends a crafted pickle | Full worker compromise | No pickle anywhere. `serial` is a tagged tree with no code path (SPEC-transport §2). | Parser bugs. Fuzzing is in M4. | `test_serial_rejects_garbage_and_never_unpickles` (TRANSPORT-I1) |
| T2 | B4 | Memory exhaustion from a declared length | `paylen = 2^60` | OOM kill of the slot | `MAX_PAYLOAD` checked before allocation | A legal 4 GiB payload is still allowed. A tighter bound per frame type is M4. | `test_end_frame_and_payload_limit` (TRANSPORT-I3) |
| T3 | B3 | Unauthenticated peer joins the data plane | A host dials a stage port | Wrong activations: a corrupt run | ed25519 handshake plus `trusted_peers`. HMAC only with an explicit token. | Without an allowlist, any authenticated identity is accepted. The wide-stage mesh path (`_connect_mesh`) runs no handshake and trusts the HELLO's `(st, pe)`: finding M2-P1-MESH-AUTH, deferred by ADR-0004. | `test_allowlist_refuses_an_unknown_peer`, `test_hmac_fallback_has_no_default_token` (TRANSPORT-I5) |
| T4 | B3 | Eavesdropping or tampering on links | An on-path attacker reads or flips activation bytes | Data leak; corruption | Today: none (authenticated handshake, plaintext frames). M4: AEAD per frame (TRANSPORT-I7). | Plaintext until M4 | M4: AEAD round trip and tamper rejection |
| T5 | B2 | Rogue worker registers | No coordinator token set | Joins as a slot or spare | Token HMAC challenge. The coordinator warns loudly when no token is set. | No token = open. M4 requires a token outside local mode. | `test_only_the_worker_holding_the_token_registers`, `test_mac_binds_token_and_nonce_and_refuses_forgeries` (CONTROL-I16) |
| T6 | B3 | Malicious worker sends wrong gradients | An admitted worker computes garbage or a targeted poison | A silently wrong model | `AuditPolicy` replication plus digest compare (library). M4 wiring: CONTROL-I13. | Undetected until M4 | `test_rejected_poison_yields_honest_bits`, `test_one_mismatch_is_a_ban` (CONTROL-I12) |
| T7 | B5 | Corrupt or forged checkpoint shard | A bit flip, a truncated file, or a malicious agent | A resume from a wrong state | sha256 on ship and restore. The common-step wave. | A malicious agent can ship a self-consistent wrong shard; audit (T6) covers it. An admitted agent can also ship a wave, or the final `RESULT`, under another slot's rank and peer: `CKPT` and `RESULT` take identity from the message and `CKPT` skips the attempt check (ADR-0003; finding M2-P1-CKPT-IDENTITY, deferred to a later phase). | `test_corrupt_shard_is_refused` (CONTROL-I8) |
| T8 | B2 | Fake liveness signals | An agent fakes heartbeats while its stage is dead | A hung run | Liveness reads process facts (exit, done marker, progress), not heartbeats alone. The stall rule catches no-progress. | Bounded by `stall_floor_s` | `test_alive_but_not_progressing_is_a_stall_not_a_conviction` (CONTROL-I7) |
| T9 | B2 | Partition used to empty a fleet | Cut the coordinator's network | Mass conviction of healthy machines | Several simultaneous losses → `failed`, never several convictions | The run stops, but no hardware is blamed | `test_two_simultaneous_losses_are_not_two_dead_boxes` (CONTROL-I7) |
| T10 | B1 | Key or token theft from disk | Another local user reads the peer key | Impersonation | Key created atomically with mode 0600; tokens from env, never logged | Root on the host defeats it | `test_ed25519_handshake_binds_identities` |
| T11 | B6 | Half-loaded pretrained model | A checkpoint missing tensors loads partially | Training a broken model | Strict load: all tensors land, or the load is refused | — | `test_an_incomplete_checkpoint_is_refused_rather_than_half_loaded` (MODEL-I4) |
| T12 | B2 | Coordinator compromise or loss | The coordinator host dies or is taken | The run stops or is hijacked | Out of scope for M0. Standby and an encrypted channel in M4. | The single point of failure | M4 |

## 3. Cost of each mitigation (first principles)

| Mitigation | Cost model | Expected cost vs the step | Status |
|---|---|---|---|
| Typed decode, payload caps, no pickle | O(header) checks per frame | ~0 (nanoseconds per frame against milliseconds per transfer) | in place |
| ed25519 handshake | One signature and one verify per connection | ~0 (connections live for the whole run) | in place |
| sha256 on checkpoint waves | ~1–2 GB/s per core, off the critical path (the agent ships after the step) | ~0 on the step | in place |
| AEAD per frame (M4) | ChaCha20-Poly1305 ~1–3 GB/s per core in software; AES-GCM with AES-NI or ARMv8 crypto ~4–10 GB/s per core. The boundary traffic is ≤ the link rate (≤ a few Gbps). Encryption runs in the sender and reader threads, overlapped with compute. | HYPOTHESIS: < 1% of the step at ≤ 10 Gbps per hop. Measured in the M4 `bench` rung. | M4 |
| Audit replication (M4) | Recomputing a fraction `p` of microbatches on a second worker. `p` is high for newcomers and decays to a floor. | HYPOTHESIS: ≈ `p` extra compute (5% at a 5% floor), off the critical path if replicas run on spares | M4; the floor `p` is set by ADR |
| Coordinator token required | One HMAC per connection | 0 | M4 (policy) |
| Fuzzing `serial`, `recv_frame`, `Channel` | CI time only | 0 at runtime | M4 |

**Rejected by this rule:**
- Per-tensor signatures on every frame. AEAD already authenticates them; signatures would add an asymmetric operation
  per frame for no extra property.
- Full replication of every microbatch. It doubles compute; sortition at a reputation floor gives the same deterrence
  at a fraction of the cost.
