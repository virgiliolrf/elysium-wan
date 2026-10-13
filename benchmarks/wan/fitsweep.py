"""On one GPU node of an identical fleet: the largest dense GPT whose every stage fits, measured for real.

    uv run --no-sync python benchmarks/wan/fitsweep.py --stages 9 [--moe] [--optimizer adamw-host] [--mbps 1000 --rtt 40]
For each candidate (largest first) the planner cuts the model for `--stages` copies of this GPU, then
`wantrain.resources.fit` builds stage 0, the heaviest middle stage and the last stage on this GPU with the plan's
schedule (its in-flight microbatches, recompute, one optimizer step) and reports the real peak. Prints one JSON line
per candidate and `BEST {json}` for the largest that fits everywhere. The coordinator measures every slot again at
launch; this only picks the candidate to launch.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

from wantrain.model.gpt import GPTConfig
from wantrain.scheduler.planner import Fleet, Node, Overrides, plan_job

# (d_model, n_head, n_layer), largest first: 12 d^2 L + 2 V d parameters.
CANDIDATES = [
    (5120, 40, 48),
    (5120, 40, 44),
    (5120, 40, 40),
    (5120, 40, 36),
    (4096, 32, 48),
    (4096, 32, 40),
    (4096, 32, 32),
    (3072, 24, 40),
    (2560, 32, 32),
]


# MoE (ADR-0008, the "deepseek" recipe without groups: sigmoid scores, selection by score plus a bias moved by
# the step's int64 counts instead of an auxiliary loss): (d_model, n_head, n_layer, experts, top_k, expert d_ff).
MOE_CANDIDATES = [
    (3072, 24, 32, 8, 2, 8192),
    (3072, 24, 28, 8, 2, 8192),
    (3072, 24, 24, 8, 2, 8192),
    (3072, 24, 20, 8, 2, 8192),
    (2560, 20, 24, 8, 2, 6912),
    (2048, 16, 24, 8, 2, 5632),
]
MOE_RECIPE = dict(
    norm="rmsnorm",
    mlp="swiglu",
    bias=False,
    pos="rope",
    moe_score="sigmoid",
    moe_select="score_bias",
    moe_norm=True,
    moe_bias_rate=0.001,
)


def _configs(moe: bool, seq: int):
    """(label, GPTConfig, total params, active params per token) per candidate, largest first."""
    V = 50257
    if not moe:
        for d, h, L in CANDIDATES:
            cfg = GPTConfig(
                vocab_size=V, d_model=d, n_layer=L, n_head=h, seq_len=seq, compute_dtype="bf16"
            )
            p = 12 * d * d * L + 2 * V * d
            yield f"dense d{d} L{L}", cfg, p, p
        return
    for d, h, L, E, k, f in MOE_CANDIDATES:
        cfg = GPTConfig(
            vocab_size=V,
            d_model=d,
            n_layer=L,
            n_head=h,
            seq_len=seq,
            compute_dtype="bf16",
            moe_experts=E,
            moe_top_k=k,
            moe_d_ff=f,
            **MOE_RECIPE,
        )
        attn, expert = 4 * d * d, 3 * d * f
        yield (
            f"moe d{d} L{L} E{E} k{k} f{f}",
            cfg,
            L * (attn + E * expert + d * E) + 2 * V * d,
            L * (attn + k * expert + d * E) + 2 * V * d,
        )


def _fit(spec_path, rank):
    r = subprocess.run(
        [sys.executable, "-m", "wantrain.resources.fit", "--spec", spec_path, "--rank", str(rank)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    for line in r.stdout.splitlines():
        if line.startswith("WT_FIT "):
            return json.loads(line[7:])
    return {"ok": False, "error": (r.stderr or r.stdout)[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=9)
    ap.add_argument("--optimizer", default="adamw-host")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--microbatches", type=int, default=32)
    ap.add_argument("--mbps", type=float, default=1000.0)
    ap.add_argument("--rtt", type=float, default=40.0)
    ap.add_argument(
        "--moe", action="store_true", help="MoE candidates (MOE_CANDIDATES) instead of dense"
    )
    a = ap.parse_args()

    major, minor = torch.cuda.get_device_capability()
    free, total = torch.cuda.mem_get_info()
    node = dict(
        tflops=80.0,
        vram_bytes=int(free),
        compute_cap=(major, minor),
        device_class=f"cuda:sm_{major}{minor}",
    )
    fleet = Fleet(
        nodes=[Node(f"n{i}", **node) for i in range(a.stages)],
        link_mbps=a.mbps,
        rtt_ms=a.rtt,
        duplex="per_direction",
    )
    best = None
    tmp = tempfile.mkdtemp(prefix="fitsweep-")
    for label, cfg, total, active in _configs(a.moe, a.seq):
        params_b = total / 1e9
        ov = Overrides(
            recompute=True,
            compute_dtype="bf16",
            boundary_dtype="bf16",
            mb_size=1,
            microbatches=a.microbatches,
            extras={"optimizer": a.optimizer},
        )
        try:
            out = plan_job(cfg, fleet, steps=10, device="cuda", require_margin=False, overrides=ov)
        except Exception as e:  # noqa: BLE001
            print(
                json.dumps({"model": label, "params_b": round(params_b, 2), "plan": str(e)[:200]})
            )
            continue
        spec = out["spec"]
        if spec is None:
            print(json.dumps({"model": label, "params_b": round(params_b, 2), "plan": "refused"}))
            continue
        path = os.path.join(tmp, f"spec-{label.replace(' ', '_')}.json")
        open(path, "w").write(spec.to_json())
        sizes = [b - a_ for a_, b in spec.partition]
        mid = max(range(1, spec.stages - 1), key=lambda r: sizes[r])
        rows = {r: _fit(path, r) for r in sorted({0, mid, spec.stages - 1})}
        # The coordinator's rule: the reserved peak within 85% of the memory free before the build.
        ok = all(
            x.get("ok") and x["peakReservedBytes"] <= 0.85 * x["freeBytesBefore"]
            for x in rows.values()
        )
        rec = {
            "model": label,
            "params_b": round(params_b, 2),
            "active_b": round(active / 1e9, 2),
            "partition": spec.partition,
            "ok": ok,
            "fit": {
                r: {
                    "params_m": round(x.get("params", 0) / 1e6),
                    "peak_gib": round(x.get("peakReservedBytes", 0) / 2**30, 2),
                    "budget_gib": round(0.85 * x.get("freeBytesBefore", 0) / 2**30, 2),
                    "held": x.get("concurrentGraphs"),
                    "error": (x.get("error") or "")[:160],
                }
                for r, x in rows.items()
            },
        }
        print(json.dumps(rec), flush=True)
        if ok:
            best = rec
            break
    print("BEST " + json.dumps(best))


if __name__ == "__main__":
    main()
