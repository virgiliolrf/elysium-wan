"""On one GPU node: optimizer "adamw-host" (RUNTIME-I20) gives the same bits on CUDA and on the CPU, and what its
step costs (the moments' round trip through PCIe).

    uv run --no-sync python benchmarks/wan/check_host_adamw.py [--params-m 500]
Prints one JSON line: bits_equal (CUDA vs CPU over 3 steps; also with the staged copies off, and with half the
moments resident on the GPU, RUNTIME-I24), and seconds per step for --params-m million parameters: serial copies,
staged copies (the default), staged with half the moments resident.
"""

import argparse
import json
import time

import torch

from wantrain.runtime import optim


def _set(n, dev, seed):
    g = torch.Generator().manual_seed(seed)
    sizes = [n // 4, n // 4, n // 2 - 7, 7]
    return [torch.randn(s, generator=g).to(dev).requires_grad_(True) for s in sizes]


def _steps(params, k, pipeline=None, resident=0.0):
    from wantrain.runtime.spec import ADAMW

    opt = optim.HostAdamW(params, lr=3e-4, pipeline=pipeline, **ADAMW)
    for step in range(k):
        if step == 1 and resident:
            opt.promote(int(resident * 8 * sum(p.numel() for p in params)))
        g = torch.Generator().manual_seed(100 + step)
        for p in params:
            p.grad = (torch.randn(p.shape, generator=g) * 10.0 ** (-(step % 5))).to(p.device)
        opt.step()
        opt.zero_grad()
    return [p.detach().cpu() for p in params], opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-m", type=float, default=500.0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    cpu, _ = _steps(_set(3_000_001, "cpu", 1), 3)
    arms = {
        "staged": _steps(_set(3_000_001, dev, 1), 3)[0],
        "serial": _steps(_set(3_000_001, dev, 1), 3, pipeline=False)[0],
        "staged_half_resident": _steps(_set(3_000_001, dev, 1), 3, resident=0.5)[0],
    }
    equal = {k: all(torch.equal(x, y) for x, y in zip(v, cpu)) for k, v in arms.items()}

    n = int(a.params_m * 1e6)
    times = {}
    pinned = None
    for name, pipeline, resident in (
        ("serial", False, 0.0),
        ("staged", None, 0.0),
        ("staged_half_resident", None, 0.5),
    ):
        params = _set(n, dev, 2)
        for p in params:
            p.grad = torch.randn_like(p)
        opt = optim.HostAdamW(params, lr=3e-4, pipeline=pipeline)
        opt.step()  # allocates the host moments
        if pinned is None:
            pinned = all(st["exp_avg"].is_pinned() for st in opt.state.values())
        if resident:
            opt.promote(int(resident * 8 * n))
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(3):
            opt.step()
        torch.cuda.synchronize()
        times[name] = (time.perf_counter() - t) / 3
        del params, opt
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "bits_equal_cuda_cpu": equal,
                "params_m": a.params_m,
                "host_moments_pinned": pinned,
                "step_s": {k: round(v, 3) for k, v in times.items()},
                "pcie_gb_s": {k: round(16 * n / v / 1e9, 2) for k, v in times.items()},
                "gpu": torch.cuda.get_device_name(),
            }
        )
    )


if __name__ == "__main__":
    main()
