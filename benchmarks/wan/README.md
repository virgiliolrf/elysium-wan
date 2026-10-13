# WAN training benchmark

Runs the engine across real machines over the public internet and turns the run into numbers: tokens/s, step time,
how much of each step waits on the network, link RTT and bandwidth, GPU utilization and memory, and the loss curve.

## Files

| File | Runs on | What it does |
|---|---|---|
| `node_setup.sh` | each node | uv, iperf3, the engine at /workspace/elysium, its locked environment, a small corpus; prints `NODE {json}` |
| `corpus_tinystories.sh` | each node | the same ~105M-token TinyStories slice on every node (the plan pins the manifest digest) |
| `sysmon.sh` | each node | per-second GPU util/memory/power and NIC bytes, to CSV |
| `netprobe.py` | one node | TCP RTT and iperf3 throughput (1 and 4 streams, both directions) to a peer |
| `run_2node.sh` | the operator | coordinator on node A, one worker per node, sysmon on both (edit hosts and ports at the top) |
| `collect.sh` | the operator | pulls stage logs, telemetry, losses and sysmon of one run |
| `analyze.py` | the operator | `metrics.json` and `REPORT.md` for a collected run |
| `plots.py` | the operator | SVG charts (loss, step time vs network wait, tokens/s, NIC traffic, GPU utilization), standard library only |
| `job-2node-us-kr.json` | node A | the plan job used for the first US-Korea run |
| `cluster.py` | the operator | N nodes: `check` (driver, TLS as uv sees it, congestion control, shared addresses, disk), parallel setup, port discovery, `linkcheck` (node-to-node Mbps with parallel flows), `sustain` (every pair loaded both ways at once for 30 s against the job's per-hop need; names the nodes to replace), launch, `watch` (per-step time, tokens/s, idle per stage; a stage slower than its own baseline, stalls, errors), status, stop, collect |
| `advise.py` | the operator | the dense shape the measured links and GPUs run best (SCHEDULER-I23) |
| `job-9stage.json` | node A | nine 24 GB GPUs: a 13B dense GPT, moments in host memory (`optimizer` "adamw-host") |
| `job-9stage-moe.json` | node A | nine 24 GB GPUs of one model: an exact MoE, 15.7B total / 4.8B active (8 experts, top-2) |
| `check_host_adamw.py` | one node | "adamw-host" gives CUDA's bits on the CPU too, and its step time (PCIe round trip) |
| `fitsweep.py` | one node | the largest GPT (dense, or `--moe`) whose stage 0, heaviest middle and last stage fit this GPU |

## Behind a port map (Vast.ai and similar)

Each service binds a container port and advertises the public port mapped to it: the coordinator listens on one
mapped port; each worker passes `--listen-port <container> --data-port <public>` and
`--probe-listen-port <container> --probe-port <public>`. Workers in different regions need `--any-region` on the
coordinator. GPUs of one architecture but different models need `WANTRAIN_DEVICE_CLASS=arch` on every worker
(CONTROL-I25): the run is lossless and reproducible on the same device map, but not byte-exact against a one-GPU
reference (cuBLAS guarantees its bits only for the same architecture and the same SM count). A byte-exactness proof
uses one GPU model throughout.

## Fitting the largest model (RUNTIME-I20, RUNTIME-I21)

Two engine options decide how many parameters a stage holds; neither changes a bit of the result relative to a
reference built with the same recipe, nor the quality of training:

- `"extras": {"optimizer": "adamw-host"}`: AdamW's moments m and v live in host memory and stream through the GPU at
  the step (8 bytes per trained parameter on the GPU instead of 16; host RAM needs 8 bytes per parameter, pinned when
  available). It is the same AdamW in fp32, written in single-rounding operations, so it is a recipe pin: the oracle
  builds the same optimizer.
- Always on for width-1 stages: every gradient folds into the fixed-order running sum the moment it is produced, B's
  at B and W's at W, so a stage never holds its whole gradient set (or a dense embedding gradient per pending
  microbatch) beside the sum.

- RUNTIME-I23: with "adamw-host" the step's AdamW runs beside the next step, each layer released as soon as it holds
  the update, so the PCIe round trip of the moments leaves the critical path; same ops, same bits.

ZeRO-style sharding across hosts is not used over WAN: it moves the parameters every step, which a datacenter fabric
absorbs and an internet link does not.

## What the first run proves, and what it does not

The engine is exact by construction: fp32 master weights, gradients folded in a fixed order, lossless wire. The
US-Korea run was an operator run on a mixed fleet (RTX 3070 Laptop and RTX 3060) and was not compared bit for bit
against an oracle; the oracle of a mixed map is per device map (ADR-0005). It stopped at step 606 of 900 when its
30 minutes ran out.

## Results

| Run | Fleet | Result |
|---|---|---|
| [`2026-10-12-us-kr-607m`](runs/2026-10-12-us-kr-607m/REPORT.md) | 2 consumer GPUs, Washington ↔ Seoul | 607M dense, 30 min, loss 11.0 → 3.1, 1,741 tok/s; latency-bound |
| [`2026-10-12-moe9`](runs/2026-10-12-moe9/NOTES.md) | 10 RTX 4090s, US and Canada | 1.0B dense, 18 steps; 4 of 10 nodes unusable; adamw-host sqrt fixed |
| [`2026-10-12-west`](runs/2026-10-12-west/REPORT.md) | 7-8 RTX 4090s, US | MoE 8-20B planned; fit and memory-model fixes (CONTROL-I26, RUNTIME-I22); no step trained |
| [`2026-10-12-us2`](runs/2026-10-12-us2/REPORT.md) | 2 H100 NVL, Virginia ↔ Denver | 6.96B dense, ~12 steps; 16 flows took the hop from ~100 to ~280 Mbps; 2.9-3.2k tok/s, still ~65% waiting: a per-layer memory overhead capped the stage at 7 layers (fixed in us3) |
| [`2026-10-12-sxm4`](runs/2026-10-12-sxm4/REPORT.md) | 4 H100 SXM, Montana → Czechia ×2 → Nuremberg | **25.1B dense** at ~2,500 tok/s, 64 microbatches: bottleneck stage 29% idle, ~45% across stages; a 340 Mbps NIC (Nuremberg) and a noisy host (one Czech node) set the pace; 256 microbatches made it worse (the NIC saturated); RUNTIME-I24 verified on GPU |
| [`2026-10-12-us3`](runs/2026-10-12-us3/REPORT.md) | 2 H100 NVL, Virginia ↔ Czechia (Denver dropped: its route lost packets) | **13.5B dense**, 25-31 steps per run, ~2,500 tok/s, ~25% waiting (was 63-72%); three memory defects fixed; the four runs' losses bit-identical; the rest of the wait is the host optimizer (RUNTIME-I24) |
| [`2026-10-12-h100`](runs/2026-10-12-h100/REPORT.md) | 3 H100 NVL, Virginia, Denver, Czechia | **2.8B dense across the Atlantic, 320/320 losses bit-identical to one H100**; 2,047 tok/s, 72-79% waiting on a 41-49 Mbps hop |

## Renting for a run

1. `cluster.py check nodes.txt`: replace every FAIL (old driver, TLS interception, slow download, small disk) and any
   pair sharing a public address.
2. `cluster.py setup`, then `discover`, then `linkcheck nodes.json 8`. Hops below ~200 Mbps make the pipeline wait;
   rent in one region or replace the slow node.
3. `advise.py --stages S --mem-gib M --tflops T --mbps <slowest hop>` picks the model shape; then `launch`.

## The bubble over WAN

A pipeline stage waits when moving one microbatch's boundary (`seq × d × 2` bytes, less with the lossless planes codec)
across its slowest hop takes longer than computing that microbatch. Latency itself is hidden by keeping about
RTT / compute microbatches in flight. The levers, none of which changes a bit or the quality:

1. **Bandwidth per hop.** Long paths starve one TCP flow: 40-50 Mbps at 130 ms between Gbps datacenters. TRANSPORT-I10
   tunes every data-plane and probe socket (BBR, buffers sized for the bandwidth-delay product), and production plans
   run 8 flows per hop. Renting nodes in one region or with measured ≥1 Gbps between them matters more than GPU model.
2. **Compute per byte.** Traffic grows with `d`, compute with `d² × layers per stage`: deeper, denser stages (or fewer
   stages, or a larger top-k in an MoE) hide more transfer. MoE models move as many bytes per token as a dense model
   of the same `d` with a fraction of the compute.
3. **Microbatches per step.** More of them shrink pipeline fill and drain; recompute keeps their held inputs small.

Asynchronous methods (DiLoCo, stale weights) hide the link entirely, but they change the result; this engine stays
exact instead.
