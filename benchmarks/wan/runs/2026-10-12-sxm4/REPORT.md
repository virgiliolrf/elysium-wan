# 2026-10-12 (night): 25B dense on four H100 SXM, three countries

This was the largest model trained so far: 25.1B parameters, which no single machine here holds. It ran at about
2,500 tokens/s. Two of the four rented nodes were weak, and they set the pace. Credit ran out during the second run;
the instances were reclaimed at 21:28.

## Fleet

| Node | Where | Pairwise links (32 flows, one way, `linkcheck`) |
|---|---|---|
| mt | Missoula, Montana | to Czechia 3,266–3,306 Mbps; to Nuremberg 409 |
| cz1 | Litvínov, Czechia | to cz2 8,299; to Montana 3,387; to Nuremberg 597 |
| cz2 | Litvínov, Czechia (same /24 as cz1) | to cz1 9,452; to Montana 1,847; to Nuremberg 680 |
| de | Nuremberg, Germany | out 721–871 |

All four are H100 80GB HBM3 (SXM), the only four on offer. The coordinator chained them mt → cz2 → cz1 → de in the
first run and mt → cz1 → cz2 → de in the second (its 1-flow probe orders them, and it is noisy).

The GPU measured 396 TFLOPs on de against 609–625 elsewhere, so the planner gave de 9 layers and the others 15.

## Before the runs

`check_host_adamw.py` on this hardware:

| Arm | Bits equal to the CPU | Seconds per step, 200M parameters |
|---|---|---|
| serial copies | yes | 0.074 |
| staged copies (RUNTIME-I24) | yes | 0.045 |
| staged, half the moments resident | yes | 0.024 |

## Run 1: 25.1B, 64 microbatches (`run25b-64mb/`)

- **Model:** d 6144, 54 layers, 48 heads, seq 1024, bf16, recompute, `adamw-host`, 32 flows per hop.
- **Fit:** passed on the first try (61.8 / 59.7 / 59.6 / 54.7 GiB of a 66.9 GiB budget).
- **Steady state, steps 4–14:** ~26 s per step, ~2,500 tokens/s.
- **Idle per stage:** 28% on the slowest (cz1), 47–50% on mt and cz2, 36–46% on de.

Step 2's per-op timeline, on the bottleneck stage:

| Part of the step | Time | Share |
|---|---|---|
| computing | 18.8 s | 71% |
| step boundary (drain, optimizer, refill) | 4.3 s | 16% |
| waiting for activations mid-step | 3.4 s | 13% |

Causes, measured:
1. **The cut, balanced on profiled costs.** At runtime stage 0's backward took 69 ms against the profiled ~117 ms, so
   mt sat idle while cz1 bottlenecked. Memory kept mt at 15 layers anyway.
2. **cz1's host is noisy.**
   - Its backward swung from 128 to 1,719 ms per microbatch, while cz2's twin stage held about 122 ms.
   - Its CPU codec took 120–150 ms, against cz2's 103–118.
   - There was no GPU throttling: 41 °C, steady power.
   - At step 15 the step doubled to 53.8 s, with cz1 the only stage busy.
3. **The optimizer is no longer the main cost.** RUNTIME-I24 cut the first forward's wait to 1.2–1.4 s, from 2.7 s
   in us3, and `opt_bg` to 0.7–1.7 s.
4. **Moving moments to the GPU (step 3) stalled that step once:** 40.8 s, with ~5.4 s pauses. Steps 4 onward were
   back to ~26 s.

## Run 2: 23.7B, 256 microbatches (`run24b-256mb/`, partial)

- **Plan:** amortize the step boundary over a 4x longer step, on an order-robust cut (15 | 13 | 13 | 10 + head).
- **Result:**
  - step 0: 221 s, 68–74% idle;
  - step 1 (pinning): 264 s;
  - **step 2: 193 s, 1,359 tokens/s, 63–69% idle.**

It was worse because de's port saturated:
- de's 1 MB fragments took a median 399 ms to send, against 4–14 ms on the other hops.
- Stages 0–2 waited together, repeatedly, 10–21 s at a time, for gradients from de.

The host advertises a 340 Mbps NIC. A short one-way `linkcheck` passed it at 600–870 Mbps; with no pause in the
traffic it could not hold the ~280 Mbps per direction a stage needs. Credit ran out at 21:28. The telemetry of mt and
cz2 was collected; only the stage logs of cz1 and de were.

## What changed in the tools

- **`cluster.py sustain nodes.json job.json`** loads every pair both ways at once for 30 s.
  - It reads the rate in 5 s windows and compares it with the job's need (one coded boundary per microbatch,
    ×1.3 margin).
  - It names the nodes to replace: a node capped by its port fails toward every peer.
- **`cluster.py watch nodes.json`** prints one line per step: time, tokens/s, and idle and compute per stage. It also
  reports:
  - each stage's baseline from its first clean steps;
  - a stage that stays 1.3x slower than its baseline (replace it or re-cut);
  - no step for 3x the baseline;
  - errors and stage processes gone.

## Takeaways

- Exactness and memory aggregation held. Speed depends on every node: one weak NIC or one noisy host sets the pace of
  the whole pipeline.
- Qualify each node under sustained two-way load before launch, and watch each stage against its own baseline.
- Balance the cut on runtime costs, not on the profile.
- Prefer datacenter fleets with ≥1 Gbps sustained per node.
