"""Where one Block's F and F+B time goes on this GPU, under the engine's determinism settings.

    uv run --no-sync python block_prof.py [d_model] [n_head] [attn_impl] [sdpa_backend]
Prints the top CUDA kernels by total time for one F and one F+B of a bf16 Block (seq 1024, mb 1), grouped by
kernel class: GEMM, weight cast (fp32->bf16 copy), attention, elementwise/other.
"""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

from wantrain.model.gpt import Block, GPTConfig
from wantrain.model.wgrad import convert
from wantrain.runtime.determinism import setup_determinism

d = int(sys.argv[1]) if len(sys.argv) > 1 else 5632
h = int(sys.argv[2]) if len(sys.argv) > 2 else 44
impl = sys.argv[3] if len(sys.argv) > 3 else "naive"
backend = sys.argv[4] if len(sys.argv) > 4 else "math"
setup_determinism(1234)
dev = torch.device("cuda")
cfg = GPTConfig(
    vocab_size=50257,
    n_layer=1,
    n_head=h,
    d_model=d,
    seq_len=1024,
    compute_dtype="bf16",
    boundary_dtype="bf16",
    attn_impl=impl,
    sdpa_backend=backend,
)
blk = Block(cfg).to(dev)
convert(blk)
x0 = torch.randn(1, 1024, d, device=dev)
g = torch.randn(1, 1024, d, device=dev)


def run(back):
    x = x0.clone().requires_grad_(back)
    y = blk(x)
    if back:
        (y.float() * g).sum().backward()


for _ in range(3):
    run(True)
torch.cuda.synchronize()


def classify(name):
    n = name.lower()
    if "gemm" in n or "cutlass" in n or "sm90" in n or "nvjet" in n or "xmma" in n:
        return "gemm"
    if "flash" in n or "fmha" in n or "attention" in n or "softmax" in n:
        return "attention"
    if "copy" in n or "convert" in n or "cast" in n:
        return "copy/cast"
    return "other"


for back in (False, True):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            run(back)
        torch.cuda.synchronize()
    groups, tops = {}, []
    for ev in prof.key_averages():
        t = ev.device_time_total / 5 / 1e3  # ms per pass
        if t <= 0:
            continue
        c = classify(ev.key)
        groups[c] = groups.get(c, 0.0) + t
        tops.append((t, ev.key[:90], ev.count // 5))
    total = sum(groups.values())
    print(f"== {'F+B' if back else 'F'} ({impl}/{backend}): {total:.2f} ms GPU per pass")
    for c, t in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"   {c:10s} {t:7.2f} ms  {100 * t / total:5.1f}%")
    for t, k, n in sorted(tops, reverse=True)[:12]:
        print(f"     {t:7.3f} ms  x{n:<3d} {k}")
