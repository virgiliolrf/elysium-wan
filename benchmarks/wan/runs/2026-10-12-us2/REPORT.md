# 2026-10-12: two H100 NVL on a good US link (Virginia ↔ Denver)

The goal was to show the WAN bubble falling once the link is good. The link was good, but the memory a stage could use
was not, and that, not the network, kept the bubble at about 65%.

The operator destroyed the instances after step ~12. The periodic collector had pulled only the run's start. The
step figures below were read live from the stage logs (`WT_TELEM`) during the run; `run7b/` holds the specs, job and
sysmon.

## Fleet and link

| Node | Where | GPU | $/h |
|---|---|---|---|
| va | Dulles, Virginia (Microsoft) | H100 NVL 94 GB, driver 580 | 3.03 |
| den | Denver, Colorado (H5 Data Centers) | H100 NVL 94 GB, driver 595 | 2.34 |

`cluster.py check`:
- both passed the TLS check through uv;
- both offer only reno/cubic (no BBR).

Link rates by measurement:

| Measurement | va → den | den → va |
|---|---|---|
| coordinator probe, 1 flow | 81–117 Mbps (RTT 43 ms) | — |
| `linkcheck`, 8 flows | 269 Mbps | 329 Mbps |
| `linkcheck`, 32 flows | 334 Mbps | 400 Mbps |
| during training, 16 flows (stage 0 sending, while busy) | 216–281 Mbps | — |

## What fit

`advise.py` asked for d 5632 × 34 layers (13.5B, 17 per stage, a priced 3% bubble). The fit check refused it and
three smaller plans, each out of memory on the 94 GB cards:
- 13.5B and 9.3B under the auto zero-bubble memory mode;
- 15.1B and 11.5B under `--zb-mem-mode 1p` (commit 4223dac).

6.96B fit:
- d 6144, 14 layers, 48 heads;
- partition [embed + 8 layers | 6 layers + head];
- measured peaks 68.0 and 58.5 GiB.

That leaves about 5.5 GiB per layer above the stage's state of weights and fp32 gradient. The analytic accounting of
deferred-W stores, the recompute graph and the bf16 weight copies explains less than half of it. **This is the open
defect: it caps the layers per stage, and with them the compute per transferred byte.**

## Training (6.96B dense, 32 microbatches of 1,024 tokens, adamw-host with the overlapped step, 16 flows)

| Step | Step time | Waiting on the network | Tokens/s | Loss |
|---|---|---|---|---|
| 0 | 8.5 s | 48% | 3,843 | 11.04 |
| 1 | 34.3 s | 89% | 955 | 17.50 |
| 2 | 12.2 s | 69% | 2,693 | 12.77 |
| 5 | 10.4 s | 63% | 3,157 | 8.98 |
| 6 | 10.6–15.7 s | 65–72% | 2,100–3,100 | — |
| 10 | 11.3 s | 66% | 2,903 | 6.92 |

Notes on the table:
- **Step 1** paid the one-time pinning of the 28 GB of host moments per stage, overlapped into the next step.
- **The loss spike at step 1** is AdamW's first full-size update without learning-rate warmup; the loss fell again
  from step 2.
- **Per step**, stage 0 sent 224–231 MB of planes-compressed activations: 12.6 MB raw per microbatch, about 0.55
  after the codec.
- **Per microbatch**, a stage computed in about 127 ms (F 25–33, B 71, W 31) and the hop moved its boundary in about
  200 ms.

Both GPUs were busy about a third of each step, and `nvidia-smi` sampled 0% on both at one instant.

## Conclusion

| | Compute per microbatch | Network per microbatch | Bubble |
|---|---|---|---|
| This run: 7 layers per GPU at d 6144 | ~127 ms | ~200 ms at ~280 Mbps | 63–72% measured |
| Advised: 17 layers per GPU | ~500 ms | ~200 ms | ~3–10% priced |

The network did its part: 16 flows took the hop from about 100 Mbps for one flow to about 280. The missing piece is
the per-layer memory overhead. The next step is a CUDA memory snapshot of one stage's fit (allocation by allocation,
with stacks), then the fix, then the advised shape on the same pair.

**Update (us3):** the per-layer overhead was three defects (a deferred W's autograd graph, the fit's allocator, and
bf16 weight copies held through a recomputed backward), all fixed; see `../2026-10-12-us3/REPORT.md`.
