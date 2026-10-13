# 2026-10-12: seven to nine RTX 4090s in the US (MoE attempts)

No training step ran on this fleet. Every launch stopped at the coordinator's fit check, which measures each stage's
real memory before the first step. What the attempts found was fixed in the engine:
- `1cba9bd`: the fit replays the slot's real order;
- `09959e2` and `423033c`: a measured overshoot is fed back (CONTROL-I26);
- `b7fbb34`: a stage builds only its own layers (RUNTIME-I22);
- `6d996e2`: fit checks run in parallel.

The run before this one, on ten US/Canada 4090s, is in `../2026-10-12-moe9/NOTES.md`.

## Fleet

Ten 4090s were rented through the CLI after a qualification pass (`cluster.py check`):

| Node | Where | GPU |
|---|---|---|
| b1 | El Segundo, CA | 24 GB |
| b2 | same host as b1 | 24 GB |
| a3 | Tustin, CA | 48 GB |
| a4 | Boise, ID | 48 GB |
| a5 | Bay View, WI | 48 GB |
| b3 | Mesquite, TX | 24 GB |
| b4 | Royse City, TX | 24 GB |
| b5 | Davidson, NC | 24 GB |

Removed:
- a1 and a2: their network intercepts TLS, so PyPI's certificate failed under uv;
- b2: it shares b1's public address, and a pipeline takes one slot per address.

Seven nodes remained.

Links measured by the coordinator:

| Hop | Mbps (fwd / rev) | RTT ms |
|---|---|---|
| a3 ↔ b1 | 438–812 / 754–764 | 2.2 |
| b1 ↔ b3 | 545 / 523 | 8.1 |
| a5 ↔ a4 | 360–376 / 237–376 | 15 |
| b4 ↔ b5 | 225–247 / 233–251 | 27 |
| b3 ↔ a5 | 127–189 / 187–190 | 33 |
| a3 ↔ b4 | 173–182 / 184–186 | 35 |
| b1 ↔ b4 | 57–198 / 166–202 | 32 |

## Attempts

| Model | Stages | Outcome |
|---|---|---|
| MoE 19.6B (d 3072, 30 layers, 8 experts top-2) | 8 | refused: two nodes on one address |
| MoE 18.3B (28 layers) | 7 | rank 3 (24 GB, 3 MoE layers) measured 22.49 GiB against a 19.66 GiB budget; the only response then was halving a microbatch already at 1 |
| MoE 17.0B, operator cut | 7 | the chain came out reversed, so 6-layer stages landed on 24 GB cards |
| MoE 15.7B (24 layers) | 6 | stopped to trade size for speed |
| MoE 8.4B (d 2048, 28 layers), balanced, 32 microbatches | 7 | 24 GB stages OOM in the fit; calibrated schedules and moved layers did not converge in four rounds |
| same, 16 microbatches | 7 | six of seven stages fit; predicted 2.76 s/step with a 53% bubble; stopped by the operator |

## What was wrong, and what changed

1. **The fit over-counted the schedule's memory.** Under recompute it built the weight-gradient stores of every
   microbatch between F and W at once (22 on stage 0 of a 7-stage, 32-microbatch plan). The schedule holds most of
   those between F and B, with only their 4 MB input resident. It now replays the slot's pinned order (`1cba9bd`).
2. **An overshoot had no remedy but halving the microbatch.** A measured overshoot, or an out-of-memory fit counted
   as a whole-card peak, now calibrates that node's activation budget for the zero-bubble solve, then moves layers
   off it. Halving the microbatch comes last (CONTROL-I26).
3. **Every stage built the whole model on the host**: about 63 GB of host RAM for 15.7B. A stage now constructs every
   layer in order and keeps only its own: the same draws, with the kept tensors and the RNG's final state bit-identical
   to the full build (RUNTIME-I22).
4. **The planner priced a gradient transient of the whole stage.** Width-1 stages fold gradient by gradient
   (RUNTIME-I21), so it now prices one tensor.

## Why the bubble stayed above 30%

Balanced compute was 82 ms per microbatch per stage. Moving 4.2 MB (d 2048) through the 143–180 Mbps hops took about
0.19 s. The pipeline runs at the slowest hop, so a stage waits for the link most of the time. Fewer microbatches add
pipeline fill on top. See `../../README.md` for the levers that keep exactness.
