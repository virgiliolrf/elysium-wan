# SC-5: host-DRAM passes per sent byte on the sealed data plane

Precondition SC-5 of `docs/design/HANDOUT-M2b.md`. Tree: `factory/m2b` after the M2 merge (a62e4eb), with
`src/wantrain/transport/secure.py` sealing every chain and mesh socket. All figures are **COUNTED** from the code path,
by the method of ADR-0006 `round3.md` §2: a pass is one read or one write of every byte in host DRAM; a user-space copy
is 2 (read + write); the kernel's socket copy is 2; a DMA is 1 (D2H or NIC receive writes host memory, NIC send or H2D
reads it). Nothing here is measured, except the two figures in "Toy CPU fleet", which are labelled MEASURED.

## Exchange frame (GRADS/PARAMS, OBJ codec, `_send_exchange` -> `_sender_loop`)

| side | step | file:line | passes |
|---|---|---|---|
| send | D2H DMA into pinned host memory (CopyEngine) | `runtime/exchange.py:358`, `runtime/stage.py:351` | 1 W |
| send | `t.view(...).numpy().tobytes()` | `transport/serial.py:66` | 2 |
| send | `b"".join(parts)` | `transport/serial.py:143` | 2 |
| send | fragment split `payload[i:i+size]` (one fragment: CPython returns the same object) | `runtime/stage.py:1325` | 0 (2 if fragmented) |
| send | `hdr + dims + len + payload` | `transport/framing.py:117` | 2 |
| send | `encrypt_into` (read plaintext, write ciphertext into the ring slot) | `transport/secure.py:197`, `:369` | 2 |
| send | gather `sendmsg([hdr, sealed])`: no user-space concatenation; kernel socket copy | `transport/secure.py:394-396` | 2 |
| send | NIC DMA read | — | 1 R |
| **send** | | | **12** (14 fragmented) |
| recv | NIC DMA write | — | 1 W |
| recv | kernel copy into the record buffer (`recv_into`) | `transport/secure.py:440`, `:613` | 2 |
| recv | `decrypt_into` (or one-shot `decrypt` for a larger record) | `transport/secure.py:207`, `:572`, `:201` | 2 |
| recv | `SecureSocket.recv`: `bytes(self._plain[...])` | `transport/secure.py:633` | 2 |
| recv | `_recv_exact`: `buf.extend(chunk)` | `transport/framing.py:157` | 2 |
| recv | `_recv_exact`: `bytes(buf)` | `transport/framing.py:158` | 2 |
| recv | `Reassembler` join | `transport/framing.py:144` | 0 (2 if fragmented) |
| recv | `serial.loads`: `bytes(mv[off:off+ln])` per blob | `transport/serial.py:162` | 2 |
| recv | `np.frombuffer(...).copy()` | `transport/serial.py:79` | 2 |
| recv | `exchange.reassemble`: `torch.cat` of a tensor spanning more than one 16 MiB chunk | `runtime/exchange.py:287` | 2 (0 if one chunk) |
| recv | H2D DMA read (CopyEngine) | `runtime/exchange.py:374` | 1 R |
| **recv** | | | **18** (16 when every tensor fits one chunk; 20 fragmented) |

## Boundary frame (FWD/BWD, `wire_codec` "raw", `_encode_body` -> `framing.tensor_payload`)

| side | step | file:line | passes |
|---|---|---|---|
| send | D2H DMA (`t.cpu()`) | `transport/framing.py:89` | 1 W |
| send | `t.numpy().tobytes()` | `transport/framing.py:96` | 2 |
| send | fragment split (mux encoder at 1 MiB when `p2p_streams` > 1, or adaptive) | `runtime/stage.py:1026`, `:1325` | 0 (2 if fragmented) |
| send | frame concatenation, seal, kernel copy, NIC DMA (as above) | `framing.py:117`, `secure.py:197/369`, `:394` | 2 + 2 + 2 + 1 |
| **send** | | | **10** (12 fragmented) |
| recv | NIC DMA, kernel copy, open, `SecureSocket.recv`, `extend`, `bytes(buf)` (as above) | `secure.py:440/613`, `:207/572`, `:633`, `framing.py:157-158` | 1 + 2 + 2 + 2 + 2 + 2 |
| recv | `Reassembler` join | `transport/framing.py:144` | 0 (2 if fragmented) |
| recv | `torch.frombuffer(bytearray(self.payload))` | `transport/framing.py:79` | 2 |
| recv | `.clone()` | `transport/framing.py:80` | 2 |
| recv | H2D DMA | — | 1 R |
| **recv** | | | **16** (18 fragmented) |

## k for S1

S1 prices host DRAM as `k·2·per_dir` per GPU (`scheduler/admission.py:322`): each GPU's host carries its sent bytes
at the sender's count and its received bytes at the receiver's count, so the exact k is the mean of the two sides.

| path | send | recv | k = (send + recv)/2 |
|---|---|---|---|
| exchange, unit tensors within one chunk | 12 | 16 | 14 |
| exchange, a tensor spanning chunks (default config, worst) | 12 | 18 | **15** |
| boundary raw, one fragment | 10 | 16 | 13 |
| boundary raw, fragmented (`p2p_streams` > 1) | 12 | 18 | 15 |
| exchange fragmented (`adapt_budget_ms` > 0; off by default, `runtime/spec.py:115`) | 14 | 20 | 17 |

**k = 15**: the worst path of either frame kind in the default configuration (`adapt_budget_ms` 0). Before sealing, the
same table gives 10 on the sender and 12 on the receiver for the exchange path (round3 §2 listed 10 per side; its
receiver list does not name `serial.py:162` or `exchange.py:287`). Sealing adds 2 on the sender (`encrypt_into`) and 4
on the receiver (`decrypt_into`, plus the `bytes()` copy in `SecureSocket.recv` that the kernel copy used to cover),
not the 2 per side HANDOUT-M2 R4 estimated.

Not in the integer (stated so nothing is quiet):
- the record buffer's compaction moves a tail of at most one record per buffer refill: at most 2/`THREADS` passes
  (0.5 at 4 threads) with the pool, about 0.125 serially (`secure.py:429-433`);
- on CUDA, `t.cpu()` and `t.to(device)` from pageable host memory stage through a driver bounce buffer (+2 per side,
  HYPOTHESIS, real-hardware pending, F7); round3 counted these DMAs as 1, and so does this count;
- `wire_codec` "planes" (production): the codec's passes are per raw byte before compression
  (`codec/planes.py:143-161`, `:164-190`), not per sent byte, and are outside k. Open: S1's `A` under a coded boundary.
- `host_dram_gbps` is a copy rate (`resources/probe.py:71-78`: bytes copied per second, each byte read and written),
  while k counts single passes; S1 against the probe is conservative by up to 2x. Open, not changed here.

## Consequences

Host-DRAM load in S1 is linear in k, so k 15 scales every round5 §2 figure by 1.5 (per GPU, 600 GB/s host):

| stage | k 10 | k 15 | 1-GPU host at 15 |
|---|---|---|---|
| B stage 0 | 944.8 GB/s (0.64) | 1417.2 (0.42) | refused (was refused) |
| B interior | 447.6 (1.34) | 671.4 (0.89) | **refused** (admitted at 10; bound k ≤ 13.4) |
| B last | 390.9 (1.54) | 586.4 (1.02) | admitted (bound k ≤ 15.4) |
| (16, 2) stage 0 | 443.8 (1.35) | 665.7 (0.90) | **refused** (admitted at 10; bound k ≤ 13.5) |
| (16, 2) interior | 284.8 (2.11) | 427.2 (1.40) | admitted |
| (16, 2) last | 259.5 (2.31) | 389.3 (1.54) | admitted |

- SCHEDULER-I16's anchor "at 10, a 600 GB/s host admits no GPU of point B's stage 0" still holds and is stronger at
  15: no GPU of stage 0 or of an interior stage.
- DESIGN §9's "At 7B on 32 H100, M 32, 400 Gbps, the planner takes S 16, w 2 with pieces (966.8 ms) on 1-GPU hosts at
  today's 10" no longer holds at the transport's k: (16, 2) stage 0 needs k ≤ 13.5. HYPOTHESIS (scaled, not re-run):
  the planner takes (16, 2) `zero1` + native at 994.4 ms there, whose host time on 1-GPU hosts at 15 is at most
  1.5/2 of its 2-GPU time at 10 (≤ 745.8 ms). Falsifier: round5 §8's script at k 15.
- `zero1` (fp32) at 128 H100 on 1-GPU hosts: host time 1897.4 ms at 10 becomes 2846.1 at 15, above its 2163.3 ms step,
  so it turns host-bound (step = max bound). The `zero1` + native 1721.3 ms row needs round5 §8 re-run at 15.
- E7 and E4 scenarios carry explicit k and are unchanged.

Toy CPU fleet (`tests/integration/test_coordinator.py::test_pieces_plan_oracle`, 4 cpu workers on one host, S 2 w 2
pieces), from `fleetplan.build_spec` with the scenario k (all 4 GPUs on one host):

| inputs | DRAM load per k | k 10 | k 15 | refuses above |
|---|---|---|---|---|
| tflops 1.0, DRAM 34 GB/s (the fields test's figures) | 1.249 GB/s | 12.49 (0.37) | 18.73 (0.55) | k 27 |
| tflops 1.6, DRAM 46 GB/s (MEASURED: this Mac, Apple M4 16 GB, `bench_tflops("cpu")` 1.60–1.66, `bench_host_dram_gbps()` 45.1–48.3, 3 runs, single idle process) | 1.998 GB/s | 19.98 (0.43) | 29.98 (0.65) | k 23 |

At k 15 the toy fleet passes S1 with the host's DRAM above 30.0 GB/s (at the probed 1.6 TFLOPS). HYPOTHESIS: four
workers probing at once on the 16 GB Mac keep the host's DRAM figure (the max of the workers' measurements) above it;
the falsifier is `test_pieces_plan_oracle` refusing with "S1 host DRAM".
