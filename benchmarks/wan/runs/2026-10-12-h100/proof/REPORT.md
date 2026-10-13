# WAN training run

- Model: 2774M params (d_model 2560, 32 layers, vocab 50257, seq 1024), bf16 compute, bf16 boundary, partition [[0, 18], [18, 34]]
- Steps: 20; 16384 tokens/step (16 x 1 x 1024)

| Metric | Value |
|---|---|
| Tokens/s (median step) | 2046.6 |
| Tokens/s (wall clock) | 1725.5 |
| Step time median / p10 / p90 (ms) | 8005.4 / 5279.7 / 14812.2 |
| Engine link probe | 41.16 / 49.3 Mbps, RTT 133.83 ms |
| Loss first -> last (min) | 10.9907 -> 6.2244 (6.1405) |

| Stage | Node | F / B / W ms | Stall share | Hop MB/step | GPU peak GiB |
|---|---|---|---|---|---|
| 0 | cz | 22.34 / 59.41 / 13.38 | 78.5% | 43.65 | 29.62 |
| 1 | us | 32.92 / 67.84 / 14.1 | 71.5% | 48.96 | 30.66 |

| Node | GPU util mean (active) | Power W | NIC rx / tx Mbps mean (p95) | Moved GB |
|---|---|---|---|---|
| cz | 12.4% (48.1%) | 123.4 | 26.72 / 24.4 (96.83 / 80.64) | 2.038 |
| us | 14.2% (47.2%) | 121.9 | 23.66 / 25.26 (84.31 / 96.82) | 2.007 |
