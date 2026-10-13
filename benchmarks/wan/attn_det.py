"""Per attention implementation: is a Block's forward+backward bit-reproducible on this GPU, and how fast?

    uv run --no-sync python attn_det.py [d_model] [n_head] [reps]
Runs under the engine's setup_determinism. For each impl: REPS forward+backward passes of one Block on the same
input; every output and gradient must hash the same across passes. Prints ms per F and per F+B.
"""

import hashlib
import sys
import time

import torch

from wantrain.model.gpt import Block, GPTConfig
from wantrain.runtime.determinism import setup_determinism

d = int(sys.argv[1]) if len(sys.argv) > 1 else 5632
h = int(sys.argv[2]) if len(sys.argv) > 2 else 44
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
setup_determinism(1234)
dev = torch.device("cuda")


def digest(ts):
    m = hashlib.sha256()
    for t in ts:
        m.update(t.detach().float().cpu().numpy().tobytes())
    return m.hexdigest()[:16]


for impl, backend in (
    ("naive", "math"),
    ("sdpa", "flash"),
    ("sdpa", "cudnn"),
    ("sdpa", "efficient"),
):
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
    torch.manual_seed(0)
    blk = Block(cfg).to(dev)
    x0 = torch.randn(1, 1024, d, device=dev, dtype=torch.float32)
    g = torch.randn(1, 1024, d, device=dev, dtype=torch.float32)
    hashes, f_ms, fb_ms = set(), [], []
    try:
        for r in range(reps):
            blk.zero_grad(set_to_none=True)
            x = x0.clone().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = blk(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            (y.float() * g).sum().backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            hashes.add(digest([y, x.grad] + [p.grad for p in blk.parameters()]))
            if r:
                f_ms.append((t1 - t0) * 1e3)
                fb_ms.append((t2 - t0) * 1e3)
        f_med = sorted(f_ms)[len(f_ms) // 2]
        fb_med = sorted(fb_ms)[len(fb_ms) // 2]
        print(
            f"{impl}/{backend}: {'DETERMINISTIC' if len(hashes) == 1 else f'NONDETERMINISTIC ({len(hashes)} hashes)'} "
            f"F {f_med:.2f} ms  F+B {fb_med:.2f} ms  hash {sorted(hashes)[0]}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"{impl}/{backend}: ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
    del blk
    torch.cuda.empty_cache()
