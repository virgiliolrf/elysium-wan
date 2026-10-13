# ADR-0006 derivations (round 2)

Research note behind `docs/design/adr/ADR-0006-exact-fsdp-placement.md`. It holds the derivations the ADR cites, so the
ADR stays one decision within 400 lines. It decides nothing. Every number here is DERIVED by the two scripts in §9
(analytic or simulated on this machine's CPU; no measurement of a GPU or a network). Inputs are HYPOTHESIS: H100 at
400 TFLOP/s effective with the recipe's deterministic kernels, 80 GB, PCIe Gen5 x16 at 50 GB/s per direction; RTX 4090
at 50 TFLOP/s, 24 GB, PCIe Gen4 x16 at 25 GB/s; 7B = h 4096, 32 blocks, 12h² params per block, V 32k. Timing on real
hardware is pending (gpu false).

**Round 3 (`round3.md`) supersedes** this note's §4 device memory (own pieces held on the device), §4 host bound
(per peer under weighted IWRR), §1 link set (host DRAM is a link) and the §5 layout tables (as-built exchange model,
fold and AdamW op times). The round-2 script below is unchanged and is the base the round-3 script loads.

Notation. `P_s` stage params, w peers, `L_s` blocks per stage, U exchange units per stage, `T_mb` tokens per
microbatch, `M/w` microbatches per peer per step, F effective FLOP/s, `b_g`/`b_p` gradient and working-copy bytes per
param, ε the human's bound (3% placeholder). Recompute on: 8 FLOPs per param per token, split F 2, B 4 (recompute F +
dX), W 2. One microbatch on one peer costs `c = 8·P_s·T_mb/F`, `c_F = c_W = c/4`, `c_B = c/2`; `C = (M/w)·c`.

Round 2 changes the model in five places, each from a review finding: (1) the unit of exchange is the `SplitLinear`
module (qkv 3h², proj h², fc1 4h², fc2 4h²), not the block; (2) the tail is a discrete-event simulation with one queue
per peer egress, per peer ingress and per uplink direction, each non-preemptive and serving the unit needed first in
the next forward; (3) the emission order of the trailing W is a plan parameter (forward or backward unit order);
(4) PCIe is a link and its copies are asynchronous; (5) a shared link sums every stage that loads it.

## 1. Links, S1 and PCIe

Every exchange byte crosses physical queues: the sender's egress, the receiver's ingress and, between sites, the
uplink of that direction. Per round (each peer runs one microbatch), a peer's microbatch puts `P_s·share` bytes per
byte-per-param on a queue: `(w−1)/w` on its own NIC per direction (per-tensor pieces make it symmetric), and
`n_a·n_b/w` per direction on an uplink between sites holding `n_a` and `n_b` of the stage's peers.

**A shared link sums every stage that loads it (round 2, self-found while fixing finding 4).** Round 1 wrote S1 per
stage. When several stages load the same link (two sites, every stage holding peers in both), the link carries all of
them in the same period: `S1_ℓ: Σ_s b_g·P_s·share_{s,ℓ} + A_ℓ ≤ cap_ℓ·c` (equal stages share one c). For S equal
stages all straddling one uplink, the threshold is S times round 1's: `T_mb ≥ S·b_g·F·share/(8·cap_up)`.

**PCIe is a link, and the copies must not run on the compute thread (finding 1).** FACT: today the slot's compute
thread copies every exchanged tensor synchronously: `t.detach().cpu()` inside `_store_grads` (`stage.py:948`), called
by `w_pass` (`stage.py:915-926`); `t.to(self.device)` in `_shard_add` (`stage.py:969`) and in the broadcast receive
(`stage.py:1047`). These add to compute, not to link time. Per step per peer at worked point A (below):
D2H 10.07 GB, H2D 10.87 GB, so 838 ms (159% of C) at 25 GB/s and 419 ms (79% of C) at 50 GB/s if they stay synchronous.

Round 2 specifies asynchronous copies: the compute stream records an event after each yielded gradient; a transport
thread issues the D2H on a dedicated copy stream into pinned host buffers in chunks of at most `f = 16 MiB`, after
that event, and ships each chunk when its copy completes; received chunks go H2D on the copy stream into a device
staging buffer, and the fold kernel waits for that copy on the device (a stream wait, never a host wait). The compute
thread's cost is one event record per yielded tensor (HYPOTHESIS: ≈ 5 µs × about 30 tensors × M/w per step, about
1 ms at worked point A, 0.2% of C). PCIe then is a link in series with the NIC carrying the same bytes, so the
binding capacity is `min(NIC, PCIe)` per direction, and each hop adds a store-and-forward latency of `2·f/PCIe`
(0.67 ms at Gen5). Copies also draw HBM bandwidth: 21 GB per step against ≈ 1.7 TB available in C (≈ 1.2%,
HYPOTHESIS). A CPU run cannot observe any of this (`.cpu()` and `.to(device)` are no-ops on CPU and
`WANTRAIN_SIM_SLOW` sleeps), so the CPU experiment checks the structure (no blocking copy on the compute thread,
RUNTIME-I13) and the cost stays a real-hardware falsification item (ADR F7).

Rejected alternatives for this term: folding on the host (64 GB of host-memory traffic per step at worked point A,
≥ 121 GB/s sustained next to the NIC and PCIe DMA; and CPU/GPU denormal handling would need its own proof) and
GPUDirect RDMA (needs RDMA NICs; the WAN transport is TCP).

S1 per NIC at w 8, pipeline traffic included: H100 179 Gbps at `T_mb` 4096 and 51 at 16384; RTX 4090 22.4 and 6.4
(round 1 numbers; unchanged by round 2, PCIe Gen5 is not below these).

## 2. The step-boundary tail (findings 1, 2 and 3)

Step t+1's first forward of unit u needs unit u's step-t update, which needs every microbatch's unit-u gradient. The
deferred-W store holds one independent closure per `SplitLinear` call (`wgrad.py:100`); each computes
`dW = dYᵀX` from tensors already resident. So the order in which W computes units is free: each closure's bits
depend only on its own inputs and kernel, and the fold is keyed by microbatch index at the owner. Only the
contributions of one parameter used several times must keep their relative (store) order, because `flush` sums them in
that order (`wgrad.py:57-70`).

- *Backward emission* (round 1, the store order): unit 1 is produced last and needed first.
- *Forward emission* (round 2 option): the trailing W computes units in forward order, so unit 1 ships, updates and
  gathers while W computes units 2..U. Its slack is `(U−1)/U·c_W`.

Neither order dominates. When the link is fast against W (volume-bound: `G + P_g > c_W + c_F`), both end near
`G + P_g + RTT − (c_W + c_F)`. When W is long against the link and RTT is large, forward emission hides the RTT. The
plan fixes the order per stage from the simulation (SCHEDULER-I17); MODEL-I7 makes it a parameter.

The simulation (script `exch`): lockstep peers; U module units; k trailing W ops; each gradient piece of unit u from W j
is released when produced; a unit's gathers are released when the owner's fold of that unit is complete (with
`barrier`, when every fold of the stage is complete, as today's `_optimizer_step_wide`); a transfer is served on each
queue it crosses independently and finishes when the last finishes (decoupled bottlenecks: optimistic for layouts
that concentrate a unit on few owners, so it favours the alternatives); each queue serves the released transfer whose
unit the next forward needs first; unit u is needed at `k·c_W + D + Σ_{v<u} c_F·P_v/P_s`, and D accumulates.

| emission | worked point A (RTT 0) | H100, S 8, w 4, `T_mb` 16k, 400 Gbps, RTT 10 ms | RTX 4090, S 8, w 4, 16k, 100 Gbps, RTT 50 ms |
|---|---|---|---|
| backward | 18.8 ms | 14.4 ms | 65.2 ms |
| forward | 19.1 ms | 0.0 ms | 0.0 ms |

`T_min` (tokens per peer per step with `E ≤ ε·C`, `window_s` = 0) per NIC, S 8, w 8, RTT 0, forward emission, round
1's backward value in brackets where it differs: H100 100 Gbps 0.229M at `T_mb` 16k, 0.066M at 64k; 400 Gbps 0.057M
at 4k; RTX 4090 10 Gbps 0.344M at 16k, 0.066M at 64k; 100 Gbps 0.004M [0.008M] at 4k. "BW" (S1 fails): H100 at
10 Gbps at every `T_mb` to 64k, H100 100 Gbps at 4k.

When S1 fails the link sets the period: `E = max(X_s, V_ℓ/cap_ℓ + RTT + t_gap − C)` with
`V_ℓ = (M/w)·(b_g·P_s·share_ℓ + A_ℓ) + b_p·P_s·share_ℓ` and `t_gap = c_B + c_W/U` (round 1, unchanged).

## 3. Layouts (finding 3: per-owner loads inside the tail)

Round 1 priced LPT and flat ranges with an additive correction. Round 2 simulates each layout's per-owner loads:
`own[u][k]`, the fraction of unit u that peer k owns, sets the bytes on each queue.
- today's stride `params[k::w]` over the block's 12-entry list (`gpt.py:255-265`, weights at positions 2, 4, 8, 10):
  binding link 2.33× the balanced load at w 4;
- LPT whole tensors (Graham 1969) over the stage's module weights;
- flat contiguous element ranges of the flattened stage;
- per-tensor dim-0 pieces: `own = 1/w` everywhere (imbalance ≤ one quantum `lcm(row, 128)`).

## 4. Memory, charged where it lands (finding 4, findings 5–7 on the host hold)

Device per peer, B per stage param: working copy `b_p` = 2; owned master, m, v, accumulator 16/w; W transient
`b_g`·largest module (per-module iterator); plus the activation model. Today's ZeRO-1, same accounting: replica 4,
owned 12/w, whole-stage fp32 flush dict 4, device hold 4δ. At w 2/4/8/16: today 16.0/14.0/13.0/12.5, proposed
10.17/6.17/4.17/3.17. The largest dense layer as its own stage (`2P + 16P/w + 2·4h²` against today's
`4P + 12P/w + 4P + 4P(w−1)/w`): 405B (h 16384) w 2 34.4 GB against 51.5; 1T (h 24576) w 4 48.3 GB, w 8 33.8 GB,
against 101.5 and 94.2 today. Today cannot place a 1T dense layer on an 80 GB device at any w ≤ 8.

**The deferred-W store and in-flight boundaries cap `T_mb` (finding 4).** The scheduler charges
`mem_b = L_s·T_mb·16·h·4` bytes per pending W (`stage_memory_model`, SPEC-scheduler §2.1). Stage 0 of a lane also
holds `min(S, M/w)` in-flight boundaries of `T_mb·h·2` bytes (1F1B and ZB-1p depth). The cross-site tables in §6 run
every entry through `device = 2·P_s + 16·P_s/w + 2·4h² + min(S, M/w)·T_mb·h·2 + L_s·T_mb·16·h·4` against the device
less the planner's reserve (`vram_reserve`: 12%, ≥ 1 GB): 70.4 GB on H100, 21.1 GB on RTX 4090. The head stage holds
fp32 logits and their gradient, `2·T_mb·V·4` bytes: 16.8 GB at `T_mb` 64k. Round 1's H100 w 4 entry (`T_mb` 160k):
one pending W 42.9 GB at L_s 1 plus 32 boundaries of 1.34 GB: above 80 GB. Refused (MEM), as the reviewer said.

**The host hold has a structural bound of one step (findings 5, 6).** Claim: at any time a peer holds at most one
step's exchange bytes, so hold plus send backlog ≤ `2·b_g·(M/w)·(w−1)/w·P_s`.
Proof. A peer computes W(t+1, m) only after B(t+1, m), hence after F(t+1, m) over every unit of its stage. F(t+1) of
unit u needs unit u's step-t gather from every owner. An owner sends that gather only after its fold of unit u for
step t is complete, which consumed every step-t piece of unit u addressed to it. So when any peer emits a step-(t+1)
piece, every owner has folded every step-t piece of every unit (all units are needed by that F), and every sender's
step-t backlog is empty. Frames are checked at the boundary: one per (step, mb, unit, sender), with step equal to the
receiver's current or next step (RUNTIME-I10), so a peer cannot inject a second step. □

The bound is charged at admission against host memory: 19.3 GB per peer at worked point A (M/w 8, w 4) and at worked
point B (M/w 32, w 4, `P_s` 0.201B); 11.3 GB at M/w 4, w 8; 618.5 GB at M/w 256, w 4, which the planner refuses
(MEM-HOST) unless the host holds it. No runtime failure path remains, so there is no `HostHoldExceeded`, no
fail-and-resume and no crash loop under a persistent straggler (whose cost is time: the step waits for its folds).
Measured δ (the actual hold) is telemetry only. A spill tier on local NVMe (lossless) would lift MEM-HOST refusals at
large M/w; it is deferred because no plan in the derived domain needs it.

## 5. Depth against width (finding: "Deeper pipeline, w = 1")

Round 1 divided the bubble by w. Each peer of a width-w stage runs whole microbatches over the whole stage, so a lane's
fill and drain cost the per-peer op time c, not c/w. Script `depth.py` runs the repository's own solver
(`zero_bubble.py`, sha256 `22988afc…751ac8d`): one ZB schedule for a lane of (S, M/w) at per-peer cost, 1p and 2p
memory. 32 H100, 7B, `T_mb` 4096; bubble as % over C:

| M | S 32, w 1 | S 16, w 2 | S 8, w 4 | S 4, w 8 |
|---|---|---|---|---|
| 32 (ZB-1p = ZB-2p) | 48.4 | 46.9 | 43.8 | 37.5 |
| 128, ZB-1p | 12.1 | 11.7 | 10.9 | 9.4 |
| 128, ZB-2p | 5.9 | 5.5 | 4.7 | 3.1 |

ZB-1p equals `(S−1)·w/(2M)`. ZB-2p helps only when a lane has more microbatches than stages. Adding the exposure of
per-tensor pieces (forward emission, conservative: additive, `window_s` = 0) at M 32:

| NIC | S 32, w 1 | S 16, w 2 | S 8, w 4 | S 4, w 8 |
|---|---|---|---|---|
| 100 Gbps | 48.4 | 70.3 | 123.2 | 167.9 |
| 400 Gbps | 48.4 | 47.6 | 47.4 | 47.1 |

So where the deepest cut is available, width does not shorten the step: a tie within 1.3 points at 400 Gbps, a clear
loss at 100 Gbps. The case for sharded stages rests on the cases depth cannot cover:
- *width forced by the device count:* more devices than cut points (7B has 32 blocks; 128 devices force w 4 at S 32).
  The do-nothing option there is today's wide stage (stride ZeRO-1);
- *memory:* a stage that cannot hold 20 B/param at w = 1 (a 405B or 1T dense layer).

**House finding (engine bridge).** `schedule_for_spec` and `default_schedule` solve a wide stage as one server of cost
`1/Σ(1/c_peer)` and then split the order per peer (`slot_orders`). `simulate_slots` on those orders gives, at M 32,
ZB-2p: S 16, w 2 46.9%; S 8, w 4 109.4%; S 4, w 8 187.5% (lane solve: 46.9, 43.8, 37.5). At M 128, ZB-2p: 40.6, 78.1,
196.9 (lane: 5.5, 4.7, 3.1). The stage-level order interleaves microbatches as if one server ran them, so per peer it
idles. Any width comparison through the current bridge is biased against width by up to 5×. SPEC-scheduler task 4
(M2b) requires lane-consistent orders for equal-weight wide stages; the defect itself predates this ADR and is filed
for the scheduler backlog.

**Worked point B (width forced): 7B on 128 H100, S 32 (one block per stage), w 4, M 128, `T_mb` 4096,** C 527.8 ms,
lane bubble 48.4% (M/w = S):

| layout | 400 Gbps: exposed (% of C) | 100 Gbps |
|---|---|---|
| today: stride, fp32, whole-stage flush, step barrier | 454.6 ms (86.1%), S1 fails | 3373.8 ms (639.3%) |
| step 1 alone: native dtype, per-module flush, barrier kept | 23.7 ms (4.5%) | 1513.6 ms (286.8%) |
| LPT whole tensors, per-unit updates | 17.4 ms (3.3%) | 716.3 ms (135.7%) |
| per-tensor pieces (proposed) | 7.6 ms (1.4%) | 450.6 ms (85.4%), S1 fails |

**Worked point A (layout comparison at fixed S 8):** 7B, S 8, w 4 (M 32, C 527.8 ms) and w 8 (C 263.9 ms), `T_mb`
4096, H100, 400 Gbps:

| layout | w 4 | w 8 |
|---|---|---|
| today: stride, fp32, whole-stage flush, barrier | 531.7 ms (100.7%), S1 fails | 433.2 (164.2%), S1 fails |
| step 1 alone: native dtype, per-module flush, barrier kept | 83.5 (15.8%) | 110.5 (41.9%), S1 fails |
| step 1 + per-unit updates (barrier removed), stride | 48.6 (9.2%) | 104.3 (39.5%) |
| LPT whole tensors, per-unit updates | 33.8 (6.4%) | 52.4 (19.8%) |
| flat element ranges, per-unit updates | 41.3 (7.8%) | 57.3 (21.7%) |
| per-tensor pieces, backward / forward emission | 18.8 / 19.1 (3.6%) | 26.8 / 27.1 (10.3%) |

Attribution at w 4: step 1 (native dtype and per-module flush, barrier kept) 531.7 → 83.5; per-unit updates without
the barrier 83.5 → 48.6; balance (LPT) 48.6 → 33.8; per-tensor pieces 33.8 → 19.1. Round 1's "534.8 → 88.0" credited
step 1 with per-unit gather overlap that step 1 does not deliver while the barrier stays (finding 7).

Owner-side W (ship X and dY to a whole-tensor owner): bytes ratio `16·T_mb/(12·h)`, 1.33 at h 4096, `T_mb` 4096;
it wins only for `T_mb < 0.75·h` and needs whole-tensor ownership (round 1, unchanged).

## 6. Across sites (findings 4 and the summed uplink)

Two equal sites, one 10 Gbps uplink, NICs 100 Gbps, RTT 50 ms, ε 3%, forward emission. Two placements:
- *all:* every stage straddles (data parallel across sites). Lanes stay inside a site, so no pipeline boundary
  crosses; the uplink carries every stage's exchange, `cap_up/S` per stage.
- *one:* a single stage straddles. Each lane still crosses the uplink once at a pipeline boundary
  (`w·2·T_mb·h` bytes per direction per round), the same crossing the plain site cut has, and the stage's exchange
  comes on top.

Every entry is S1 on the uplink, then `T_min` (window 0), then the device predicate of §4 (L_s 1, 2, 4):
- H100, w 2, 4, 8, both placements, `T_mb` 16k–160k: refused (S1) in every entry. For *all*, S1 needs
  `T_mb ≥ S·40k` at w 2; for *one*, the boundary bytes alone take `w·2·h/cap_up` = 81% of the per-token compute
  time at L_s 4, w 2, leaving `T_mb ≥ 215k`, where one pending W is 225 GB.
- RTX 4090, *all*: refused (S1 or MEM) in every entry.
- RTX 4090, *one*: admitted only at w 2: L_s 1, `T_mb` 16k (`T_min` 0.20M, batch 0.4M; device 8 GB) and 64k (0.13M;
  20 GB); L_s 2, 16k (0.11M; 14 GB). Every w 4 and w 8 entry is refused (S1 or MEM).

Conclusion: streamed exact sharding across a ~10 Gbps uplink carries no useful placement for a 7B model on either
fleet. *All* is refused; *one* never removes the boundary crossing, only adds bytes to it, so the planner's
simulation will prefer the plain cut. Round 1's admission domain for H100 (w ≤ 4, `T_mb` 40–160k) is withdrawn. The
exact candidate for cross-site width is a chained fold (§8), which needs its own ADR.

## 7. Step loop and liveness (finding 7)

Today a step is a hard barrier: `_run_step`, then `_optimizer_step_wide` blocks until every PARAMS broadcast arrives
(`stage.py:1042-1043`, `1065-1070`). Per-unit updates remove it:
- a unit's update (AdamW on the owner's piece, then the cast and the gather send) runs at the first `_pump` after its
  fold completes, wherever the slot is: between ops, inside a blocked wait of step t, or inside a wait of step t+1;
- an F of step t+1 on unit u waits for unit u's gathered copy of step t (a new wait type, filled like any network
  wait: `_fill_w` may run a runnable W of the same step; there is none of step t+1 before its B).

Liveness. Step-t ops depend only on step-t inputs (the existing schedule is deadlock-free: INV-3). Every peer runs
all its step-t W before any step-(t+1) op (`_run_step` raises on a pending W). A step-t update depends only on step-t
W pieces, which every peer emits before blocking on any step-(t+1) input, and the transport threads deliver without
the compute thread. The pump performs updates in every wait, so no slot blocks on a step-(t+1) gather while holding
an update another slot needs. By induction on t, every step completes. The condition that must hold, and that E1
tests with W fill and jitter, is: *an update is never deferred to a scheduled point; it runs in the first pump after
its fold completes.*

Boundaries that keep today's full barrier: a checkpoint step (a slot first completes every owned unit's update, a
local wait on step-t pieces only), a between-step re-plan (M2: ownership moves at a quiescent point) and the last
step. ZB-V (M3) applies the same rule per virtual stage and must re-check it.

## 8. Chained folds (ideas from round 2; not adopted here)

The left fold `s_i = fl(s_{i−1} + g_i)` lets a contiguous block of microbatches be folded wherever its gradients are,
if the fp32 prefix of the previous block arrives first. With contiguous block ownership (not IWRR), peer j or site j
continues the fold from its predecessor's prefix. Bytes across a boundary fall from `(M/w)·b_g·share` per step to one
fp32 prefix (4 B per param) plus the gather (2 B), independent of M/w. Costs: the chain is serialized at the step end
(the last block waits for every earlier prefix, pipelined per unit: at worked point A, `4·P_s/cap` = 64 ms per hop, 12%
of C, against 3.6% streamed), each block's gradients are held until the prefix arrives (`(M/w)·b_g·P_s`), and a
second fold placement must be proven equal to the first. It is the only exact path found for cross-site width (§6),
so it is recommended as its own ADR (site-chained fold, round 1: cross-site `T_peer` ≥ 8.0M for H100 and 1.0M for
RTX 4090 at 10 Gbps, ε 3%, no `T_mb` condition).

## 9. Scripts

Both scripts run with `python3 -I` (Python 3.14 on this machine), exit 0, in 11.3 s and 0.6 s. Outputs are quoted in
§1–§7. sha256 of the files as run:
- `adr6_model_r2.py`: `f43ab4f6e3758d663f992cd527799110664ecaba9914c8bfe6eaebed20b701eb`
- `depth.py` (argument: `src/wantrain/scheduler/zero_bubble.py`): `4cdc2dec349698b34beaafb1ab904281a8ded8a934e77a1909f396e3edef5621`

```python
# ADR-0006 round-2 cost model. Analytic; every input is HYPOTHESIS unless noted. Recompute on:
# 8 FLOPs/param/token per stage: F 2, B 4 (recompute F + dX), W 2  => c_F = c/4, c_B = c/2, c_W = c/4.
# Exchange units are SplitLinear modules in forward order: qkv 3h^2, proj h^2, fc1 4h^2, fc2 4h^2 per block
# (LayerNorm and biases ride with the next module; they are < 0.01% of the bytes).
import heapq
G9 = 1e9
DEV = {"H100": (400e12, 80e9, 50e9), "RTX4090": (50e12, 24e9, 25e9)}  # F, device bytes, PCIe B/s per direction
EPS = 0.03
h = 4096; V = 32000; NBLK = 32; P = 12 * h * h * NBLK
MOD = [("qkv", 3), ("proj", 1), ("fc1", 4), ("fc2", 4)]
F_COPY = 16 * 2**20  # copy chunk (bytes)
def reserve(mem): return max(0.12 * mem, 2**30)

def units(L):
    return [(f"b{b}.{n}", s * h * h) for b in range(L) for n, s in MOD]

def own_pieces(us, w): return [[1.0 / w] * w for _ in us]

def own_stride(L, w):
    # today's params[k::w] over the stage's 12-entry-per-block list (gpt.py:255-265); weights at 2, 4, 8, 10
    idx = {"qkv": 2, "proj": 4, "fc1": 8, "fc2": 10}
    out = []
    for b in range(L):
        for n, _ in MOD:
            row = [0.0] * w; row[(12 * b + idx[n]) % w] = 1.0; out.append(row)
    return out

def own_lpt(L, w):
    ts = sorted([(s, i) for i, (_, s) in enumerate([(n, s) for b in range(L) for n, s in MOD])], key=lambda x: (-x[0], x[1]))
    load = [0] * w; out = [[0.0] * w for _ in range(4 * L)]
    for s, i in ts:
        j = load.index(min(load)); load[j] += s; out[i][j] = 1.0
    return out

def own_flat(us, w):
    tot = sum(p for _, p in us); edges = [tot * j / w for j in range(w + 1)]; out = []; a = 0.0
    for _, p in us:
        b = a + p
        out.append([max(0.0, min(b, edges[j + 1]) - max(a, edges[j])) / p for j in range(w)]); a = b
    return out

def exch(us, own, w, cap, c, k=1, rtt=0.0, order="fwd", barrier=False, flush="module", bg=2, bp=2,
         lat=0.0, sites=None, cap_up=None):
    """Exposed step-boundary time D of one sharded stage (lockstep peers), discrete-event.
    Queues: each peer's egress and ingress and, across sites, the uplink of each direction. A transfer is served
    on each queue it crosses independently (decoupled bottlenecks: optimistic for concentrated layouts, so it
    favours the alternatives) and finishes when the last of them finishes. Each queue is non-preemptive and serves
    the released transfer whose unit is needed first in the next forward (P3-style priority; gathers before
    gradients of the same unit). k trailing W ops; a unit's gathers are released when the owner's fold of that unit
    is complete (with barrier: when every fold of the stage is complete)."""
    U = len(us); Ptot = sum(p for _, p in us)
    cW = cF = c / 4
    seq = list(range(U)) if order == "fwd" else list(range(U - 1, -1, -1))
    prod = {}
    for j in range(k):
        t = j * cW
        for u in seq:
            t += cW * us[u][1] / Ptot
            prod[(j, u)] = t if flush == "module" else (j + 1) * cW
    def queues(s, d):
        q = [("e", s), ("i", d)]
        if sites and sites[s] != sites[d]:
            q.append(("up", sites[s], sites[d]))
        return q
    def dur(q, nbytes):
        return nbytes / (cap_up if q[0] == "up" else cap)
    items = []  # [rel, prio, kind, u, s, d, j, bytes, served{q: fin}]
    pend = {}   # queue -> list of item indices not yet served there
    def add(rel, kind, u, s, d, j, nbytes):
        it = [rel, (u, kind, j, (d - s) % w, s), kind, u, s, d, j, nbytes, {}]
        items.append(it); i = len(items) - 1
        for q in queues(s, d):
            pend.setdefault(q, []).append(i)
    for (j, u), t in prod.items():
        for s in range(w):
            for d in range(w):
                if s != d and own[u][d] > 0:
                    add(t, 1, u, s, d, j, bg * us[u][1] * own[u][d])
    need_g = {(u, d): (w - 1 if own[u][d] > 0 else 0) for u in range(U) for d in range(w)}
    got = {}
    upd = {(u, d): prod[(k - 1, u)] for u in range(U) for d in range(w) if own[u][d] > 0}
    ready = {(u, d): 0.0 for u in range(U) for d in range(w)}
    owners_total = sum(1 for x in need_g if need_g[x] > 0)
    done_owners = []
    def gathers(u, o, t):
        for d in range(w):
            if d != o:
                add(t, 0, u, o, d, -1, bp * us[u][1] * own[u][o])
    free = {}
    while True:
        best = None
        for q, lst in pend.items():
            if not lst:
                continue
            rel = min(items[i][0] for i in lst)
            t = max(free.get(q, 0.0), rel)
            if best is None or t < best[0] - 1e-15:
                best = (t, q)
        if best is None:
            break
        t, q = best
        cand = [i for i in pend[q] if items[i][0] <= t + 1e-15]
        i = min(cand, key=lambda x: items[x][1])
        pend[q].remove(i)
        it = items[i]
        fin = t + dur(q, it[7]); free[q] = fin; it[8][q] = fin
        if len(it[8]) < len(queues(it[4], it[5])):
            continue
        arr = max(it[8].values()) + rtt / 2 + lat
        kind, u, s, d, j = it[2], it[3], it[4], it[5], it[6]
        if kind == 0:
            ready[(u, d)] = max(ready[(u, d)], arr)
            continue
        if j != k - 1:
            continue
        upd[(u, d)] = max(upd[(u, d)], arr)
        got[(u, d)] = got.get((u, d), 0) + 1
        if got[(u, d)] == need_g[(u, d)]:
            if not barrier:
                gathers(u, d, upd[(u, d)])
            else:
                done_owners.append((u, d))
                if len(done_owners) == owners_total:
                    tb = max(upd.values())
                    for (uu, oo) in done_owners:
                        gathers(uu, oo, tb)
    end = k * cW
    D = 0.0
    for d in range(w):
        Dd = 0.0; fcum = 0.0
        for u in range(U):
            own_ready = upd[(u, d)] if own[u][d] > 0 else 0.0
            rdy = max(ready[(u, d)], own_ready if own[u][d] > 0 else 0.0)
            if barrier:
                rdy = max(rdy, max(upd.values()))
            need = end + Dd + fcum
            if rdy > need:
                Dd += rdy - need
            fcum += cF * us[u][1] / Ptot
        D = max(D, Dd)
    return D

def lam_of(own, us, w):
    Pt = sum(p for _, p in us)
    ing = max(sum((w - 1) * own[u][j] * us[u][1] for u in range(len(us))) for j in range(w)) / Pt
    eg = max(sum((1 - own[u][j]) * us[u][1] for u in range(len(us))) for j in range(w)) / Pt
    return max(ing, eg) / ((w - 1) / w)

def stage(dev, L, w, Tmb, Mw, cap_nic, lay="pieces", k=1, rtt=0.0, order="fwd", barrier=False, flush="module",
          bg=2, bp=2, sites=None, cap_up=None, A=None, A_up=0.0, pcie=True):
    F, mem, cpcie = DEV[dev]
    us = units(L); Ps = sum(p for _, p in us)
    own = {"pieces": own_pieces(us, w), "stride": own_stride(L, w), "lpt": own_lpt(L, w), "flat": own_flat(us, w)}[lay]
    cap = min(cap_nic, cpcie) if pcie else cap_nic
    lat = 2 * F_COPY / cpcie if pcie else 0.0
    c = 8 * Ps * Tmb / F; C = Mw * c
    X = exch(us, own, w, cap, c, k, rtt, order, barrier, flush, bg, bp, lat, sites, cap_up)
    A = 4 * Tmb * h if A is None else A
    lam = lam_of(own, us, w)
    if sites:  # binding uplink: n_a*n_b/w of every unit per direction
        na = sum(1 for s in sites if s == 0); nb = w - na
        share, cb = na * nb / w, cap_up
        A = A_up
    else:
        share, cb = (w - 1) / w * lam, cap
    G = bg * Ps * share / cb
    s1 = G + A / cb <= c
    Vb = Mw * (bg * Ps * share + A) + bp * Ps * share
    linkb = Vb / cb + rtt + c / 2 + (c / 4) / len(us) - C
    return dict(c=c, C=C, X=X, G=G, s1=s1, linkb=linkb, E=max(X, linkb), Ps=Ps, lam=lam)

def tmin(dev, L, w, Tmb, cap_nic, **kw):
    """Smallest tokens per peer per step with E <= eps*C (window_s = 0, conservative). X does not depend on M/w;
    linkb is linear in M/w, so one simulation and a scan over M/w suffice."""
    r1 = stage(dev, L, w, Tmb, 1, cap_nic, **kw)
    if not r1["s1"]:
        return None, None
    r2 = stage(dev, L, w, Tmb, 2, cap_nic, **kw)
    slope = r2["linkb"] - r1["linkb"]; base = r1["linkb"] - slope
    for Mw in range(1, 10**6):
        if max(r1["X"], base + slope * Mw) <= EPS * Mw * r1["c"]:
            return Mw * Tmb, Mw
    return float("inf"), None

links = {"10G": 10e9 / 8, "100G": 100e9 / 8, "400G": 400e9 / 8}
Ps8 = P / 8
print(f"P={P/G9:.3f}B  P_s(S=8)={Ps8/G9:.4f}B")

print("== worked point A (layout comparison at fixed S 8, w 4): 7B, M 32, T_mb 4096, H100, 400G NIC, PCIe Gen5 50 GB/s")
cap = links["400G"]
rows = [
    ("today: stride, fp32, whole-stage flush, barrier", dict(lay="stride", bg=4, bp=4, flush="stage", barrier=True, order="bwd")),
    ("step 1 alone: stride, native, per-module flush, barrier", dict(lay="stride", barrier=True)),
    ("step 1 + per-unit updates (no barrier), stride", dict(lay="stride")),
    ("LPT whole tensors, native, per-unit", dict(lay="lpt")),
    ("flat element ranges, native, per-unit", dict(lay="flat")),
    ("pieces, backward-order emission (round 1)", dict(lay="pieces", order="bwd")),
    ("pieces, forward-order emission (round 2)", dict(lay="pieces")),
]
for ww in (4, 8):
    Mw = 32 // ww
    for name, kw in rows:
        r = stage("H100", 4 if ww == 4 else 4, ww, 4096, Mw, cap, **kw)
        print(f"w={ww} {name:58s} S1={r['s1']!s:5s} lam={r['lam']:.2f} X={r['X']*1e3:6.1f} linkb={r['linkb']*1e3:7.1f} E={r['E']*1e3:6.1f} ms ({100*r['E']/r['C']:5.1f}% of C={r['C']*1e3:.1f})")
r = stage("H100", 4, 4, 4096, 8, cap, pcie=False, order="bwd")
print(f"check vs round 1 (pieces, bwd, no PCIe, block units differ): X={r['X']*1e3:.1f} ms")
r = stage("H100", 4, 4, 4096, 8, cap, k=2)
print(f"pieces fwd k=2: X={r['X']*1e3:.1f} ms")

print("== synchronous host copies on the compute thread (today's .cpu() / .to(device)), per step, worked point A")
Mw, ww, bg, bp = 8, 4, 2, 2
d2h = Mw * (ww - 1) / ww * bg * Ps8 + bp * Ps8 / ww
h2d = Mw * (ww - 1) / ww * bg * Ps8 + (ww - 1) / ww * bp * Ps8
C = Mw * 8 * Ps8 * 4096 / 4e14
for name, bw in (("PCIe Gen4 25 GB/s", 25e9), ("PCIe Gen5 50 GB/s", 50e9)):
    t = (d2h + h2d) / bw
    print(f"{name}: D2H {d2h/G9:.2f} GB + H2D {h2d/G9:.2f} GB = {t*1e3:.0f} ms per step = {100*t/C:.0f}% of C ({C*1e3:.1f} ms)")
print(f"async chunked copies: added latency per hop 2*f/PCIe = {2*F_COPY/50e9*1e3:.2f} ms (Gen5)")

print("== depth vs width at 32 H100 (bubble from the lane solve, see depth.py) + exposure (pieces, fwd, 400G / 100G)")
bub = {(32, 1): 48.4, (16, 2): 46.9, (8, 4): 43.8, (4, 8): 37.5}
for ln in ("100G", "400G"):
    for (SS, ww), b in bub.items():
        LL = 32 // SS
        if ww == 1:
            print(f"{ln} S={SS:2d} w={ww}: bubble {b:5.1f}% + exposed  0.0% = {b:5.1f}%"); continue
        r = stage("H100", LL, ww, 4096, 32 // ww, links[ln])
        e = 100 * r["E"] / r["C"]
        print(f"{ln} S={SS:2d} w={ww}: bubble {b:5.1f}% + exposed {e:5.1f}% = {b+e:5.1f}%  (S1 {r['s1']})")

print("== worked point B (width forced: 7B on 128 H100, S 32 one block per stage, w 4, M 128, T_mb 4096)")
for ln in ("100G", "400G"):
    for name, kw in rows[:2] + rows[3:4] + rows[6:]:
        r = stage("H100", 1, 4, 4096, 32, links[ln], **kw)
        print(f"{ln} {name:58s} S1={r['s1']!s:5s} X={r['X']*1e3:6.1f} E={r['E']*1e3:7.1f} ms ({100*r['E']/r['C']:5.1f}% of C={r['C']*1e3:.1f})")

print("== memory per peer, device B/param")
def mem_today(w, d): return 4 + 12 / w + 4 + 4 * d
def mem_new(w, L): return 2 + 16 / w + 2 * (4 / 12) / L
for ww in (2, 4, 8, 16):
    print(f"w={ww:2d}: narrow 20.0 | today {mem_today(ww, (ww-1)/ww):5.2f} | proposed {mem_new(ww, 4):5.2f} | host worst (hold+backlog) 2*b_g*(M/w)*(w-1)/w")
print("== largest dense layer (12h^2) as its own stage, device GB = 2P + 16P/w + 2*4h^2 (today: 4P+12P/w+4P+4P(w-1)/w)")
for name, hh in (("405B", 16384), ("1T", 24576)):
    p = 12 * hh * hh
    for ww in (1, 2, 4, 8):
        today = (4 * p + 12 * p / ww + 4 * p + 4 * p * (ww - 1) / ww) if ww > 1 else 20 * p
        print(f"{name} w={ww}: proposed {(2*p + 16*p/ww + 2*4*hh*hh)/G9:6.1f} GB | today {today/G9:6.1f} GB")

print("== per-NIC T_min (tokens/peer/step, w 8, eps 3%, k 1, RTT 0, S 8): fwd order; round-1 bwd in brackets")
for d in DEV:
    for ln, cp in links.items():
        row = []
        for T in (4096, 16384, 65536):
            a, _ = tmin(d, 4, 8, T, cp)
            b, _ = tmin(d, 4, 8, T, cp, order="bwd")
            row.append("BW" if a is None else f"{a/1e6:.3f}M [{b/1e6:.3f}M]")
        print(f"{d:8s} {ln:5s}: " + " | ".join(row))

print("== cross-site: two equal sites, one 10 Gbps uplink, NICs 100G, RTT 50 ms. The uplink is shared by every")
print("   stage that straddles it: 'one' = only this stage straddles; 'all' = every stage straddles (cap_up/S each)")
cap_up = 10e9 / 8
for d, (F, memb, _) in DEV.items():
    avail = memb - reserve(memb)
    for n in (1, 2, 4):
        ww = 2 * n; sites = [0] * n + [1] * n
        for L in (1, 2, 4):
            S = NBLK // L; Ps = 12 * h * h * L
            for mode in ("one", "all"):
                cu = cap_up if mode == "one" else cap_up / S
                Au = ww * 2 * T * h if mode == "one" else 0.0
                row = []
                for T in (16384, 65536, 131072, 163840):
                    Au = ww * 2 * T * h if mode == "one" else 0.0
                    r1 = stage(d, L, ww, T, 1, links["100G"], sites=sites, cap_up=cu, rtt=0.05, A_up=Au)
                    if not r1["s1"]:
                        row.append("S1"); continue
                    tp, Mw = tmin(d, L, ww, T, links["100G"], sites=sites, cap_up=cu, rtt=0.05, A_up=Au)
                    if Mw is None:
                        row.append("E"); continue
                    infl = min(S, Mw)
                    dev_b = 2 * Ps + 16 * Ps / ww + 2 * 4 * h * h + infl * T * h * 2 + L * T * 16 * h * 4
                    host_b = 2 * 2 * Mw * (ww - 1) / ww * Ps
                    tag = "MEM" if dev_b > avail else "ok"
                    row.append(f"{tp/1e6:.2f}M({ww*tp/1e6:.1f}M){tag}:{dev_b/G9:.0f}GB,host{host_b/G9:.0f}")
                print(f"{d:8s} w={ww} L_s={L} S={S:2d} {mode:3s}: T_mb 16k/64k/128k/160k: " + "  ".join(row))
print("head stage logits at V 32k, fp32 value + grad, per microbatch: " + ", ".join(f"T_mb {T//1024}k {2*T*V*4/G9:.1f} GB" for T in (16384, 65536, 131072)))

print("== emission order where RTT matters (X in ms, fwd vs bwd; window_s = 0)")
for d, L, ww, T, cp, rtt, tag in (("H100", 4, 4, 4096, links["400G"], 0.0, "worked point A, RTT 0"),
                                  ("H100", 4, 4, 16384, links["400G"], 0.010, "H100 w4 S8 T_mb 16k, 400G, RTT 10 ms"),
                                  ("RTX4090", 4, 4, 16384, links["100G"], 0.050, "RTX4090 w4 S8 T_mb 16k, 100G, RTT 50 ms")):
    a = stage(d, L, ww, T, 8, cp, rtt=rtt); b = stage(d, L, ww, T, 8, cp, rtt=rtt, order="bwd")
    print(f"{tag}: fwd X={a['X']*1e3:.1f}  bwd X={b['X']*1e3:.1f}  (c_W={a['c']/4*1e3:.1f}, G={a['G']*1e3:.1f})")
sites = [0, 1]
for T in (16384, 65536):
    a = stage("RTX4090", 1, 2, T, 1, links["100G"], sites=sites, cap_up=10e9/8, rtt=0.05, A_up=2*2*T*h)
    b = stage("RTX4090", 1, 2, T, 1, links["100G"], sites=sites, cap_up=10e9/8, rtt=0.05, A_up=2*2*T*h, order="bwd")
    print(f"RTX4090 cross-site one-stage w2 L_s1 T_mb {T//1024}k: fwd X={a['X']*1e3:.1f}  bwd X={b['X']*1e3:.1f} ms (c_W={a['c']/4*1e3:.1f})")
print("== host worst case (owner hold + send backlog) = 2*b_g*(M/w)*(w-1)/w*P_s")
for Mw, ww, Ps in ((8, 4, Ps8), (32, 4, P / 32), (4, 8, Ps8), (256, 4, Ps8)):
    print(f"M/w={Mw} w={ww} P_s={Ps/G9:.3f}B: {2*2*Mw*(ww-1)/ww:.0f} B/param = {2*2*Mw*(ww-1)/ww*Ps/G9:.1f} GB per peer")
```

```python
# Depth vs width through the repository's own solver (zero_bubble.py), per-peer op time.
import importlib.util, sys
spec = importlib.util.spec_from_file_location("zb", sys.argv[1]); zb = importlib.util.module_from_spec(spec); sys.modules["zb"] = zb; spec.loader.exec_module(zb)
h = 4096; P = 12 * h * h * 32; F = 4e14; Tmb = 4096
def run(S, w, M, k):
    Ps = P / S; L = 32 // S
    c = 8 * Ps * Tmb / F
    cf, cb, cw = c / 4, c / 2, c / 4                     # per-peer, recompute on
    bnd = Tmb * h * 2; wpin = L * Tmb * 16 * h * 4         # SPEC-scheduler memory model
    cfg = zb.ZBConfig(S, M, [cf / w] * S, [cb / w] * S, [cw / w] * S, comm=[0.0] * (S - 1),
                      mem_f=[float(bnd)] * S, mem_b=[float(wpin)] * S, mem_w=[-float(bnd + wpin)] * S)
    if S == 1:
        sched = zb.zero_bubble(cfg, "1p")
    else:
        sched = zb.zero_bubble(zb._with_pending_budget(cfg, k), "given")
    owner = [zb.iwrr(M, [1.0] * w) for _ in range(S)]
    orders = zb.slot_orders(sched.orders, owner, [w] * S)
    _e, ms = zb.simulate_slots(cfg, orders, owner, [[1.0] * w for _ in range(S)])
    C = (M / w) * c
    return C, ms, 100 * (ms / C - 1), 100 * (S - 1) * w / (2 * M)
for M in (32, 128):
    for S, w in ((32, 1), (16, 2), (8, 4), (4, 8)):
        for k in (1, 2):
            C, ms, over, closed = run(S, w, M, k)
            print(f"M={M:3d} S={S:2d} w={w} ZB-{k}p: C={C*1e3:7.1f} ms makespan={ms*1e3:7.1f} ms bubble={over:5.1f}% (closed form (S-1)w/2M {closed:5.1f}%)")
print("== lane solve: one ZB schedule for (S, M/w) at per-peer op time, every lane the same")
def lane(S, w, M, k):
    Ps = P / S; L = 32 // S; c = 8 * Ps * Tmb / F; m = M // w
    bnd = Tmb * h * 2; wpin = L * Tmb * 16 * h * 4
    cfg = zb.ZBConfig(S, m, [c / 4] * S, [c / 2] * S, [c / 4] * S, comm=[0.0] * (S - 1),
                      mem_f=[float(bnd)] * S, mem_b=[float(wpin)] * S, mem_w=[-float(bnd + wpin)] * S)
    sched = zb.zero_bubble(cfg, "1p") if S == 1 else zb.zero_bubble(zb._with_pending_budget(cfg, k), "given")
    _e, ms = zb.simulate(cfg, sched.orders)
    C = m * c
    return C, ms, 100 * (ms / C - 1)
for M in (32, 128):
    for S, w in ((32, 1), (16, 2), (8, 4), (4, 8)):
        r = [lane(S, w, M, k) for k in (1, 2)]
        print(f"M={M:3d} S={S:2d} w={w}: lane ZB-1p bubble={r[0][2]:5.1f}%  ZB-2p bubble={r[1][2]:5.1f}%")
```
