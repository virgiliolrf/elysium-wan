# 2026-10-12: ten rented RTX 4090s (US and Canada), first multi-node attempt

The instances were destroyed after 18 training steps, before the logs were collected, so there is no REPORT here. This
records what the attempt established.

## The fleet

| Node | Where | GPU | Outcome |
|---|---|---|---|
| n0 | Richardson, TX (AT&T) | RTX 4090 48 GB | used (stage 1, coordinator) |
| n1 | Los Angeles (FASTNET) | RTX 4090 48 GB | used (stage 0); 5-37 Mbps to every other node, RTT 183 ms to n0 |
| n2 | Los Angeles (AkileCloud) | RTX 4090 48 GB | rejected: PyPI's TLS certificate fails (UnknownIssuer), an intercepting proxy |
| n3 | Tustin, CA (BSQL) | RTX 4090 48 GB | used (stage 2) |
| n4 | San Jose (PEG TECH) | RTX 4090 24 GB | used (spare) |
| n5 | Bay View, WI (Comcast) | RTX 4090 48 GB | used (stage 4) |
| n6 | Portland (Comcast) | RTX 4090 24 GB | rejected: ~10 Mbps download |
| n7 | Ashwaubenon, WI (TDS) | RTX 4090 24 GB | rejected: unreachable, then no download throughput |
| n8 | Vancouver (TELUS) | RTX 4090 24 GB | used (stage 3) |
| n9 | Montréal (Bell) | RTX 4090 24 GB | rejected: driver 570 (CUDA 12.8) cannot run the locked torch 2.14.1+cu130 |

Every node had 16 GB of disk: enough for the environment and the corpus, not for checkpoints of a 15B+ model.

## Established

- **24 GB and 48 GB RTX 4090s compute the same bits.** A GEMM and attention hash over the engine's shapes (fp32 and
  bf16) was identical on n0, n3 (48 GB) and n8 (24 GB) under drivers 595-615: same AD102, 128 SMs, 72 MB L2.
- **adamw-host (RUNTIME-I20):** 0.80 s per step per 1B parameters (moments round trip at 20 GB/s, PCIe 4.0 x16, pinned).
- **CPU fp32 sqrt is not correctly rounded** on these hosts (x86, AVX-512): 1 ulp off on 17% of elements, while CUDA's
  is exact. adamw-host now takes its sqrt in fp64 and rounds once; CUDA and CPU then agree bit for bit.
- **The links decide.** Measured by the coordinator: n0-n3 185-196 Mbps (RTT 33 ms), n3-n8 129-177 (36 ms), n8-n5
  290-300 (11 ms), n1-n0 32-37 (183 ms). A 1.0B dense model (d 2048, 16 layers, 16 microbatches) over five stages
  planned 2.95 s/step and ran 11-14 s, then 30 s per step, almost all of it waiting on the n1 hop. Loss 6.24 -> 5.97
  between steps 9 and 10.

## Fixed on the way

- `cluster.py`: Vast's login banner arrives on stderr after a command's output (it broke `discover`); `pkill -f
  "wantrain (...)"` matched its own shell and killed it before `mkdir` (workers never started).
- `cluster.py check`: qualifies each node in about a minute before any setup (driver, TLS to PyPI, download speed,
  disk, one GPU model).

## Next attempt

Rent in one metro or one provider region, check every node with `cluster.py check` before setup, and replace failures
before installing anything. Disk >= 60 GB if the run checkpoints a 15B+ model.
