# 2026-10-12: three H100 NVL on two continents

## Result

**Bit-exact across the Atlantic.** A 2.8B dense GPT trained in a 2-stage pipeline between Rakovník (Czechia) and
Denver (Colorado) produced the same loss, bit for bit, as the same recipe on one H100. All 320 losses matched: 20
steps × 16 microbatches, compared as fp32 hex (`proof/identity.json`). The reference ran on one of the rented
H100 NVLs from the run's own spec.

| | Pipeline, CZ ↔ Denver | One H100 (reference) |
|---|---|---|
| Model | 2.77B dense (d 2560, 32 layers, vocab 50257, seq 1024) | same |
| Recipe | bf16 compute and boundary, recompute, AdamW moments in host memory (RUNTIME-I20), lossless `planes` wire | same |
| Losses equal to the reference | **320 / 320** | — |
| Loss, step 0 → step 19 | 11.01 → 6.22 | 11.01 → 5.99 (step 19's last microbatch) |
| Tokens/s | 2,047 (median step), 1,726 (wall clock) | 5,304 |
| Step time median / p10 / p90 | 8.0 / 5.3 / 14.8 s | 3.1 s |
| Share of the step waiting on the network | 78.5% (stage 0), 71.5% (stage 1) | — |
| Link | 41 / 49 Mbps (fwd / rev), RTT 134 ms | — |
| GPU peak | 29.6 / 30.7 GiB of 93 | — |

Charts: `proof/plots/` (loss, step time against network wait, tokens/s, NIC traffic, GPU utilization).
Per-stage detail: `proof/REPORT.md` and `proof/metrics.json`.

## What limited the speed: the path, not the engine

The datacenters advertise 8–35 Gbps NICs. Across the Atlantic, one TCP flow carried 40–50 Mbps at 130 ms, and with
both directions loaded at once 5 + 13 Mbps (`moe-attempt/coord.log`). Each microbatch's boundary is 5.2 MB raw
(2.6 MB after the lossless planes codec), so a 33 ms forward and 68 ms backward per stage waited about a second per
microbatch on the link. Virginia → Denver measured 106–118 Mbps at 41 ms.

The 130 ms of latency is not what costs the time: a pipeline hides latency with a few microbatches in flight (about
RTT / compute, here 3). The cost is the bytes a single flow delivers over that round trip. CUBIC halves its window on
a loss and regrows one segment per round trip, and kernel autotuning caps a flow near its ceiling / RTT. TRANSPORT-I10
(commit 4a32a2d, after this run) answers that without touching a payload byte:
- BBR congestion control per socket;
- send/receive buffers sized for the bandwidth-delay product;
- 8 parallel flows per hop in production plans.

## The 19B MoE: planned, not trained

The plan:
- 15 layers, d 3072, 16 SwiGLU experts, top-4;
- 19.0B total and 5.4B active parameters;
- three stages, Virginia → Denver → Czechia.

The fit check measured 91 GiB peaks against a 79 GiB budget on two stages and an out-of-memory on the third, under a
32-microbatch zero-bubble schedule. The coordinator re-solved the schedule with calibrated budgets (CONTROL-I26).
A relaunch with 16 microbatches and 16 flows per hop was stopped by the operator before its checks finished, because
the transatlantic hop (5–50 Mbps) would have left the GPUs idle ~90% of each step.

## Fleet and cost

| Node | Where | Network | GPU | $/h |
|---|---|---|---|---|
| cz | Rakovník, Central Bohemia (ISP Alliance) | 8–9 Gbps NIC | H100 NVL 95.8 GB | 3.60 |
| us | Denver, Colorado (H5 Data Centers) | 0.6–0.7 Gbps NIC | H100 NVL 95.8 GB | 2.34 |
| va | Dulles, Virginia (Microsoft) | 29–38 Gbps NIC | H100 NVL 95.8 GB | 3.05 |

Rejected or replaced before use:
- a Bulgarian NVL whose host went offline;
- a Czech H100 SXM in the same datacenter as cz;
- a Virginia NVL stuck pulling its image for over 20 minutes;
- H100s on CUDA 12 drivers, which cannot run the locked torch 2.14.1+cu130;
- an H100 PCIe, a different chip (114 SMs) that would not be bit-comparable.
