"""Hardware capability detection: measure what a node actually is.

Advertised numbers are hints; the layer partition and the precision ladder are
solved from what this box measures, running this matmul, in this process.
Compute capability is the one that is not merely a refinement: without it the
planner cannot know whether bf16 is a real dtype on this device or a slow
emulation. Works on CUDA, Apple MPS and plain CPU, so a heterogeneous fleet
(a datacenter card, a consumer card, a laptop) describes itself in one shape.

`python -m wantrain.resources.probe` prints exactly one JSON line on stdout.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time


def _sync(dev) -> None:
    import torch

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def bench_tflops(device: str = "cuda", d: int = 2048, iters: int = 8, dtype=None) -> float:
    """Sustained matmul TFLOPs on this device, warmed up. FP32 with TF32 off
    by default: the training arithmetic the reference recipe runs."""
    import torch

    dev = torch.device(device)
    dt = dtype or torch.float32
    a = torch.randn(d, d, device=dev, dtype=dt)
    b = torch.randn(d, d, device=dev, dtype=dt)
    for _ in range(3):  # warmup: context creation, autotune
        a @ b
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    _sync(dev)
    dt_s = time.perf_counter() - t0
    return 2 * d**3 * iters / max(dt_s, 1e-9) / 1e12


# At least 256 MiB per timed copy: well past every cache level, so the figure is
# DRAM bandwidth, not a cache's.
COPY_BYTES = 256 << 20


def _median_copy_gbps(dst, src, dev, reps: int = 3) -> float:
    """GB/s (1e9 bytes copied per second) of dst.copy_(src): one warm-up, then
    the median of `reps` timed copies."""
    dst.copy_(src)
    _sync(dev)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        dst.copy_(src)
        _sync(dev)
        times.append(time.perf_counter() - t0)
    times.sort()
    return src.numel() * src.element_size() / max(times[len(times) // 2], 1e-9) / 1e9


def bench_host_dram_gbps(nbytes: int = COPY_BYTES) -> float:
    """Host-DRAM copy bandwidth between two host buffers."""
    import torch

    cpu = torch.device("cpu")
    src = torch.ones(nbytes, dtype=torch.uint8)
    dst = torch.empty(nbytes, dtype=torch.uint8)
    return _median_copy_gbps(dst, src, cpu)


def bench_pcie_gbps(device: str = "cuda", nbytes: int = COPY_BYTES) -> tuple[float, float]:
    """(host->device, device->host) GB/s over PCIe, from pinned host memory."""
    import torch

    dev = torch.device(device)
    host = torch.ones(nbytes, dtype=torch.uint8).pin_memory()
    gpu = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    h2d = _median_copy_gbps(gpu, host, dev)
    d2h = _median_copy_gbps(host, gpu, dev)
    del gpu
    return h2d, d2h


def system_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_class(device: str) -> str:
    """The device class of `device` (CONTROL-I25): "cpu:<machine>:<ATen CPU
    capability>", "mps:<chip>" or "cuda:<device name>". The one function the
    HELLO's deviceClass and hfm.device_class (the gate's) both use."""
    import torch

    dev = torch.device(device)
    if dev.type == "cpu":
        return f"cpu:{platform.machine()}:{torch.backends.cpu.get_cpu_capability()}"
    if dev.type == "mps":
        import subprocess

        try:
            chip = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            chip = "unknown"
        return f"mps:{chip}"
    if dev.type == "cuda":
        # WANTRAIN_DEVICE_CLASS=arch groups CUDA devices by compute capability (one class per architecture: an
        # RTX 3060 and an RTX 3070 Laptop GPU are both sm_86) for an operator who accepts that the oracle then holds
        # per device map only (ADR-0005): cuBLAS may pick other kernels on another SM count, so such a run is
        # bit-comparable only to a run on the same map. The default class is the device name.
        if os.environ.get("WANTRAIN_DEVICE_CLASS") == "arch":
            major, minor = torch.cuda.get_device_capability(dev)
            return f"cuda:sm_{major}{minor}"
        return f"cuda:{torch.cuda.get_device_name(dev)}"
    return f"{dev.type}:unknown"


def measure(device: str | None = None, bench: bool = True, bench_size: int = 2048) -> dict:
    """-> capability record. `ok` is False only when torch itself is unusable."""
    out: dict = {
        "ok": False,
        "hostname": platform.node(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "cpuCount": os.cpu_count() or 1,
        "ramBytes": system_ram_bytes(),
    }
    if out["ramBytes"] > 0:  # 0 means unknown: leave it missing, never a measured zero
        out["hostRamBytes"] = out["ramBytes"]
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        out["error"] = f"torch import failed: {e}"
        return out

    out["torch"] = torch.__version__
    out["cudaBuild"] = torch.version.cuda
    dev_name = device or best_device()
    out["device"] = dev_name
    dev = torch.device(dev_name)
    out["deviceClass"] = device_class(dev_name)

    if dev.type == "cuda":
        idx = dev.index or 0
        props = torch.cuda.get_device_properties(idx)
        cap = torch.cuda.get_device_capability(idx)
        out.update(
            gpuName=props.name,
            gpuCount=torch.cuda.device_count(),
            computeCap=[cap[0], cap[1]],
            # total is the card; free is what this process can actually take,
            # which is what the memory budget has to be solved against.
            vramTotalBytes=int(props.total_memory),
            vramFreeBytes=int(torch.cuda.mem_get_info(idx)[0]),
        )
    elif dev.type == "mps":
        try:
            rec = int(torch.mps.recommended_max_memory())
        except Exception:  # noqa: BLE001
            rec = 0
        out.update(
            gpuName=f"apple-mps-{platform.machine()}",
            gpuCount=1,
            computeCap=None,
            vramTotalBytes=rec or out["ramBytes"],
            vramFreeBytes=rec or out["ramBytes"],
        )
    else:
        out.update(
            gpuName="cpu",
            gpuCount=0,
            computeCap=None,
            vramTotalBytes=out["ramBytes"],
            vramFreeBytes=out["ramBytes"],
        )

    if bench:
        n = bench_size
        out["tflopsFp32"] = round(bench_tflops(dev_name, d=n), 4)
        cap = out.get("computeCap")
        if dev.type == "cuda" and cap and tuple(cap) >= (8, 0):
            out["tflopsBf16"] = round(bench_tflops(dev_name, d=n, dtype=torch.bfloat16), 4)
        out["hostDramGBps"] = round(bench_host_dram_gbps(), 4)
        # On cpu and mps the device is the host: there is no PCIe hop, so the
        # PCIe fields stay absent rather than zero.
        if dev.type == "cuda":
            h2d, d2h = bench_pcie_gbps(dev_name)
            out["pcieH2dGBps"] = round(h2d, 4)
            out["pcieD2hGBps"] = round(d2h, 4)
            torch.cuda.empty_cache()
    out["ok"] = True
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="measure this node's capabilities")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--out", default=None, help="also write the JSON here")
    a = ap.parse_args(argv)
    try:
        r = measure(a.device, bench=not a.no_bench)
    except BaseException as e:  # noqa: BLE001 -- reported as data
        import traceback

        r = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-1500:],
        }
    if a.out:
        # Written before printing: a channel that dies mid-command loses stdout,
        # a file survives it.
        tmp = a.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(r, fh)
        os.replace(tmp, a.out)
    print(json.dumps(r))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
