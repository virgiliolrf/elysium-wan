# 2026-10-12: 13.5B dense on two H100 NVL across the Atlantic

Commit hashes below refer to the development history before the public release; those fixes are in this
repository's first commit.

The 2026-10-12-us2 run left one open defect: about 5.5 GiB per layer above a stage's state capped a d=6144 stage at 7
layers, and with them the compute per transferred byte. This run found that defect and two more, fixed all three,
and trained the shape `advise.py` asked for (d 5632, 34 layers, 13.5B) for the first time. The pipeline now spends
about 25% of each step waiting, against 63–72% in us2. Most of that wait is the host-side optimizer, not the link.

## Result

| Run | Fleet | Cut (layers) | Mode | Steps | Step (median) | Tokens/s | Idle, stage 0 / 1 |
|---|---|---|---|---|---|---|---|
| `run13b` | Virginia → Denver | 18 \| 16 + head | 1p | 6 | 30.3 s | 1,080 | 68% / 73% (the route lost packets, below) |
| `run13b-cz` | Virginia → Czechia | 18 \| 16 + head | 1p | 28 | 12.9 s | 2,535 | 28% / 32% |
| `run13b-cz-move1` | Czechia → Virginia | 19 \| 15 + head | 1p | 25 | **12.7 s** | **2,575** | **14%** / 34% |
| `run13b-cz-auto` | Virginia → Czechia | 19 \| 15 + head | auto, bf16 fix | 31 | 13.3 s | 2,465 | 24% / 30% |

The model is d 5632, 34 layers, 44 heads, vocab 50257, seq 1024; bf16 compute and boundary, recompute, `adamw-host`,
32 microbatches of one sequence. Each run used 32 flows per hop (16 on Denver) and the lossless `planes` wire, whose
ratio on this model is 0.69 (7.9 MB per microbatch). "Idle" is the stage's `WT_TELEM` stall share; it equals 100%
minus the GPU's busy share measured from the per-op events (`*_telemetry.jsonl`), steps 2 onward.

The loss falls from 11.06 to 5.95 in 31 steps (`run13b-cz-auto`). A single H100 cannot run this model: 13.5B at 8
bytes per parameter is 108 GB.

**Bit-identical across runs.** The four runs' per-microbatch losses match, bit for bit, on every (step, microbatch)
they share: 168/168 across all four, 896/896 between `run13b-cz` and `run13b-cz-auto`. That holds across:
- two routes;
- the machines' order;
- two cuts;
- two schedule modes;
- the bf16 fix below.

## Three memory defects, found and fixed

| Defect | Commit | Measured effect |
|---|---|---|
| A deferred W's GEMM ran with grad mode on and recorded a graph that held every microbatch's dY until the step's end | `111726c` | stage 0 fit, d 5632 × 17: 77.3 GiB (`memsnap-r0.log`), was OOM; 1.4 GiB per layer above the state, was ~5.5 |
| The fit subprocess never called `setup_determinism`, so its allocator ran without the runtime's expandable segments | `08f08d9` | rank 0: 85.4 GiB in the fit, 77.3 GiB under the runtime's allocator; plans that fit were refused |
| A split linear saved the bf16 cast of its weight for backward: a recomputed backward held a bf16 copy of every weight of the stage | `abb9761`, `7fe3e63` | auto mode: 62.6 / 69.0 GiB, was 85.4 / 80.8; fits on the first try |

`7fe3e63` corrects `abb9761`. `torch.matmul` folds a non-contiguous dY only when the weight requires grad, so the
copy cast again in backward keeps the original cast's `requires_grad`. HF mamba's gate caught the difference;
the GPT runs here were unaffected (896/896 above).

`benchmarks/wan/memsnap.py` runs a stage's fit under PyTorch's allocator history and lists the blocks alive at the
peak by call site. That is how the third defect was found.

## The route that lost packets

Virginia → Denver measured 333 Mbps with 16 flows at 19:03. During `run13b` the same direction fell to about
7 Mbps:
- every flow's congestion window sat at 1–5 segments;
- about 2.2 MB was retransmitted per 10.5 MB delivered.

Denver → Virginia stayed clean. Steps 0 and 1 took about 280 s each, and the run never recovered past 114 Mbps.
Denver was replaced by the Czech node, whose path measured 895–1,097 Mbps (Virginia → Czechia) and ~2,800 Mbps
(back) with 32 flows, twice in a row.

## Where the remaining wait goes

The per-op events of `run13b-cz-move1` step 10 (Czechia is stage 0, Virginia stage 1):
- stage 0's first forward takes 2.7 s, against 70 ms for the next 31;
- stage 1 waits 4.2 s for its first activation, then runs 32 microbatches at ~265 ms each with no wait;
- 4.2 s + 32 × 265 ms = 12.7 s, the measured step.

The 2.7 s is the `adamw-host` step of the previous step. RUNTIME-I23 overlaps it with the next forward, but a
forward cannot pass layer k before layer k is updated. The update moves 16 bytes per parameter through PCIe
(7.1B parameters on stage 0). So:
- the link is not the bottleneck at these rates;
- more microbatches in flight would not help: stage 0 already runs all 32 forwards ahead.

Two changes follow (RUNTIME-I24, `optim.py`, `stage.py`); neither moves a bit:
1. **Staged copies.** The moments' copy in, update and copy out run on three CUDA streams through rotating
   device buffers, overlapping each other; a layer is released by a CUDA event its forward's stream waits on.
2. **Resident moments.** After the second optimizer step a stage moves onto the GPU the moments that fit the room
   its measured peak leaves. The bf16 fix freed about 10–16 GiB per stage, about 1.3–2B parameters' worth.

`check_host_adamw.py` measures both on a single GPU (bits against the CPU, seconds per step) before the next run.

## Other findings

- **Moving a layer** from stage 1 to stage 0 balanced one stage and unbalanced the other: +1.6% tokens/s. The fit
  allowed exactly one: stage 0 then measured 78.57 GiB of a 78.72 GiB budget.
- **The coordinator's 1-flow probe** underestimates a long hop ~10×: 44–118 Mbps against 600–1,100 with 32 flows. Its
  predicted bubble (47–52% on Virginia ↔ Czechia) was pessimistic; the measured one was 24–30%.
- **The coordinator's memory model** omitted the bf16 copies, about 2 bytes per parameter. The third fix removes
  them rather than pricing them.

## Files

- `run13b*/`: per-node stage logs (`WT_TELEM`), per-op telemetry, loss streams, specs, sysmon, coordinator logs.
- `coord-logs/`: every coordinator attempt (fit checks, calibrations, schedules).
- `job-13b*.json`: the jobs as launched; `memsnap-r0.log`: the allocation snapshot after the first fix.

## Cost

Two H100 NVL at $5.37/h (Virginia + Denver), then $5.97/h (Virginia + Czechia), about 1 h 45 min in all, ~$10.
