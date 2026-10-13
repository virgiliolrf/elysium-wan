# SPEC-transport: frames, identity, control channel, emulation and lossless codecs

**Workstream:** `transport`. **Owned paths:**
- `src/wantrain/transport/`, `src/wantrain/codec/`;
- `tests/unit/test_transport.py`, `tests/unit/test_codecs_identity.py`.

## 1. Scope

Everything that turns objects into bytes on a socket and back. This is the only code that parses bytes from the
network.
- **Data plane:** frames, fragments and the boundary codecs.
- **Object codec:** `serial`, which never executes code.
- **Peer identity and record protection:** ed25519 identities, X25519 session keys, AEAD records.
- **Control channel:** coordinator ↔ agent.
- **WAN emulation shim.**

## 2. Domain model (normative)

- **Frame** (little-endian):
  - header: `magic u16 = 0xE17A | type u8 | codec u8 | step u32 | mb u32 | chunk u16 | ndim u8 | frag_idx u16 | frag_cnt u16 | shape u32·ndim | paylen u64`;
  - then the payload.
  - **Types:** `FWD`, `BWD`, `END`, `GRADS`, `PARAMS`, `LOSS`, `HELLO`, and (M4, ADR-0009) `TIED`: the rows of a tied
    table between stage 0 and the Head, fp32 rows of shape (rows, h) in ascending id (the ids come from `make_batch` on
    both ends and never travel), forwarded fragment by fragment by the stages between (RUNTIME-I19). A `TIED` frame under a plan with no tied relay
    is fatal at receipt (RUNTIME-I19); the type exists only with ADR-0009's runtime task and goes with its rollback,
    whose target leaves the type id unassigned, so typed decode refuses such a frame as an unknown type. `chunk` is the sending peer in a wide stage, else 0.
    A `TIED` frame's codec is `RAW_F32`, `ndim` 2, and every fragment carries the whole payload's shape `(rows, h)`:
    the whole payload is `rows·h·4` bytes, so an unfragmented frame (`frag_cnt ≤ 1`) needs `paylen == rows·h·4` and a
    fragment `paylen ≤ rows·h·4`, both checked at receipt before the payload is read, and the `Reassembler` refuses a
    reassembled payload of any other length; each is a `FrameError`. A `TIED` frame of 0 rows is one empty fragment.
  - **Codecs:** `RAW_F32`, `RAW_I64`, `FP8_PLANES`, `OBJ`, `F32_PLANES`, `PLANES`.
  - `paylen ≤ MAX_PAYLOAD` (env `WANTRAIN_MAX_FRAME_BYTES`, default 4 GiB). Larger is a `FrameError`.
- **Fragments.** A payload is split into byte ranges `frag_idx/frag_cnt`. The `Reassembler` concatenates them in index
  order, so the result is exact whatever the arrival order and however many parallel streams carried it.
- **END.** The graceful-shutdown marker. The sender emits it after draining, then half-closes. A following EOF is not
  a failure.
- **Object codec** (`serial`):
  - Layout: `header u32 | JSON structure | tensor blobs`, as a tagged tree. Tuples, non-string dict keys and tensors
    (dtype, shape, raw bytes) round-trip exactly.
  - Unknown tags and malformed input raise `SerialError`.
  - There is no code path from bytes to execution.
- **Boundary codecs.** Byte-plane split plus zstd. `encode`/`decode` are exact byte images.
  - `planes`: any declared precision.
  - `f32_planes`: fp32.
  - `fp8_planes`, `encode_scaled_fp8`: raw fp8 nibble planes, and recipe-declared scaled fp8 that is exact for the
    recipe's round trip.
- **Identity** (`authenticate`): an ed25519 key per agent (atomic create, mode 0600) and `peer_id = sha256(pub)[:32]`.
  - The handshake binds the plane's domain, both identities, both challenges and both X25519 ephemerals.
  - `trusted_peers` is an allowlist and `expect_peer` a single expected id. Any other peer is refused.
  - There is no shared-token mode.
- **Records** (`SecureSocket`): AEAD records under the handshake's per-direction keys (TRANSPORT-I7).
- **Control channel** (`Channel`): length-prefixed JSON frames, plus `send_file` streaming for checkpoint shards.
  It runs the control-domain handshake and is sealed before any message (TRANSPORT-I9). Admission is
  `token_mac(token, transcript)` inside the sealed channel.
- **Shim:** `SenderPacer` paces at the link bandwidth, and `DelayedDelivery` adds the latency plus seeded jitter, per link.

## 3. Interfaces

**Provides**
- `send_frame(sock, ftype, step, mb, payload, chunk, codec, shape, frag_idx, frag_cnt)`, `recv_frame(sock) -> Frame`,
  `send_end(sock)`, `Reassembler().push(frame)`, `tensor_payload`, `FrameError`, `MAX_PAYLOAD`.
- `serial.dumps(obj) -> bytes`, `serial.loads(bytes)`, `SerialError`.
- `authenticate(sock, key_path, trusted_peers, expect_peer, aes_hw, domain, role)`, `SecureSocket`, `hardware_aes`,
  `control_dial`, `control_accept`, `load_or_create_key`, `peer_id`.
- `Channel(sock).send/send_file/recv/close`, `token_mac`, `session_mac`, `check_mac`, `check_session_mac`, `ChannelClosed`.
- `LinkSpec`, `SenderPacer.pace`, `DelayedDelivery.push`.
- `codec.planes.encode/decode/encode_scaled_fp8`, `codec.f32_planes`, `codec.fp8_planes`.

**Consumes:** model `precision` and `fp8` (scale recovery for the scaled-fp8 codec).

## 4. Invariants

- **TRANSPORT-I1.** `serial` round-trips tensors bit-exactly and preserves the shapes the runtime dispatches on (tuples
  stay tuples). It rejects garbage and never unpickles.
- **TRANSPORT-I2.** Frames round-trip tensors and objects, and fragments reassemble to the exact payload in any order.
- **TRANSPORT-I3.** `END` is delivered as a distinct frame, and a payload above `MAX_PAYLOAD` is refused.
- **TRANSPORT-I4.** Every boundary codec round trip is bit-exact: `planes`, `f32_planes`, and scaled fp8 for the
  recipe's round trip.
- **TRANSPORT-I5.** The handshake signs, per role, the hash of the plane's domain, both keys, both challenges and both
  X25519 ephemerals. A reflected challenge, a signature under another domain, and non-complementary declared roles are
  refused. The allowlist or the expected peer refuses any other id. `peer_id` is `sha256(pub)[:32]`. There is no
  shared-token mode.
- **TRANSPORT-I6.** The shim delays delivery by the configured link latency.
- **TRANSPORT-I8.** (DEFERRED-TO: M2b) In shared-pacer mode the shim paces every connection of a peer through one
  egress budget and one ingress budget, and every connection between two declared sites through one uplink budget per
  direction. The aggregate throughput of concurrent flows through a budget stays within 5% of it, and the delivered
  bytes are unchanged.
- **TRANSPORT-I7.** The data plane is sealed after the handshake:
  - with AES-256-GCM when both ends have hardware AES (AES-NI or ARMv8), else ChaCha20-Poly1305;
  - under per-direction keys from X25519 signed by the ed25519 identities (HKDF-SHA256 over the transcript);
  - with 4-byte prefix plus 64-bit counter nonces that never repeat under a key;
  - with a deterministic rekey after `2^24` records or `2^36` bytes.

  Tampered, truncated, replayed or reordered records are fatal. The payload bytes delivered to the runtime are
  unchanged. Per-core throughput is at least the link rate. The measured seal-plus-open time is under 1% of the step
  (SIMULATED under `gpu: false`).
- **TRANSPORT-I9.** The control channel runs the control-domain handshake with the coordinator in the coordinator role,
  and is sealed with its keys before any message. The worker's token MAC and session MAC use distinct domains over the
  transcript and are validated as 64 lowercase hex. No token-derived value proves the coordinator.

- **TRANSPORT-I10.** Data-plane and probe sockets are tuned for long paths before their handshake
  (`transport/tuning.py`): congestion control BBR per socket when the kernel offers it (`WANTRAIN_TCP_CC`, "off"
  keeps the system's), and send/receive buffers of `WANTRAIN_TCP_BUF` (32 MiB) only where the kernel's hard cap admits
  them and they exceed its autotuning ceiling (a fixed buffer below that ceiling would cut, not raise, the window).
  A refused knob keeps its default and never fails a connection; no payload byte changes. Each stage logs what held
  (`WT_NET`). Production plans use 8 parallel flows per hop (`p2p_streams`).
- **TRANSPORT-I11.** The coordinator probes and measures each hop with the run's own flow count (`p2p_streams`): the
  PROBE and MEASURE messages carry `streams`, and a multi-flow measurement is time-based (`resources/links._timed`:
  every flow sends for 4 s, the first 1 s of TCP's ramp discarded), so the rate the solver prices is the rate the data
  plane can hold, not one flow's nor a ramp's.
- **TRANSPORT-I12.** A message on a hop with N > 1 flows is cut into fragments of ceil(len / N) bytes, clamped to
  [64 KiB, 1 MiB] (`frag_size`), so a boundary spreads over every flow instead of the first few (2026-10-13: a 2.15 MB
  boundary in 1 MiB fragments used 3 of 16 flows). The receiver reassembles by index; the bytes are unchanged
  (`test_a_boundary_spread_over_every_flow_is_bitwise`).
- **TRANSPORT-I13.** Every data-plane and probe socket, dialed, listening or accepted, turns on the kernel's TCP pacing
  (`SO_MAX_PACING_RATE`, `tuning.pace`; cap `WANTRAIN_TCP_PACE_MBPS` per flow, default 10000, "off" keeps the
  kernel's default). Without BBR, CUBIC sends each window as a line-rate burst; 32 flows bursting one boundary lost
  0.35% of their packets Virginia to Czechia and a 7.9 MB boundary took a median 729 ms (p90 969). Paced: median 132
  ms, p90 220 ms on the same path (2026-10-13). No payload byte changes; a kernel without the option keeps bursting.


## 5. Ordered task list

1. **M2 — identity and sealed data plane (TRANSPORT-I5 amended, TRANSPORT-I7).** The domain-, role- and
   ephemeral-bound handshake and `SecureSocket` AEAD records as stated in TRANSPORT-I5 and TRANSPORT-I7, with the
   seal-plus-open cost measured against the step (HANDOUT-M2 tasks `identity-handshake`, `aead-records`,
   `stage-p2p-secure`).
2. **M2 — sealed control channel (TRANSPORT-I9; moved from M4).** The control-domain handshake, transcript-bound token
   and session MACs, and a sealed `Channel` (HANDOUT-M2 task `control-auth-primitives`).
3. **M2b — shared pacers for exchange experiments (TRANSPORT-I8, ADR-0006 E4).** `LinkSpec` gains a shared-pacer mode
   (per-peer egress and ingress budgets, per-site-pair uplink budgets) so that layouts that concentrate traffic on few
   NICs are measured as they would run. Per-connection pacing (today) gives each flow its own bandwidth and cannot
   show the concentration ADR-0006 F3 and F6 test.

## 6. Decisions

- **Exact tagged JSON plus blobs over pickle.** Pickle on a socket is remote code execution for anyone who can reach
  the port.
- **zstd byte planes over zlib.** About 70× faster on boundary planes, and lossless.
- **Lossless codecs only.** A lossy wire changes the result. Rejected by the product rule.
