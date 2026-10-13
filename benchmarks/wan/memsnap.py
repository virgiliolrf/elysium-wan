"""On one GPU node: one stage's fit under the allocator's history, and what is alive at its peak, by call site.

    uv run --no-sync python benchmarks/wan/memsnap.py --d 5632 --heads 44 --layers 34 --stages 2 [--rank 0]

Plans the dense model for `--stages` copies of this GPU (as fitsweep.py does), runs wantrain.resources.fit.measure
for one rank with torch.cuda.memory._record_memory_history on, then replays the allocator trace to the moment the
allocated total was highest and groups the blocks alive then by their innermost wantrain frame. Prints
`WT_MEMSNAP {json}` and writes the raw snapshot next to it (drag it into pytorch.org/memory_viz).

The stage's state is `params x state_bytes` (8 B/param under adamw-host: fp32 weight and gradient); everything above
it, per layer, is what caps the layers a stage holds (2026-10-12: ~5.5 GiB per d=6144 layer on an H100 NVL).
PYTORCH_CUDA_ALLOC_CONF is the runtime's (expandable segments, RUNTIME setup_determinism) unless --alloc-conf says
otherwise, so reserved-minus-allocated shows the fragmentation the fit check would see.
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
from collections import defaultdict


def _site(frames):
    for f in frames:
        if "wantrain" in f.get("filename", ""):
            return f"{f['filename'].split('wantrain/')[-1]}:{f['line']} {f['name']}"
    f = frames[0] if frames else {}
    return f"{f.get('filename', '?').split('/')[-1]}:{f.get('line', 0)} {f.get('name', '?')}"


def replay(trace):
    """Allocated bytes over the trace; -> (peak bytes, {site: [bytes, count]} alive at the peak, events)."""
    live, total, peak, at_peak = {}, 0, 0, {}
    for e in trace:
        act = e["action"]
        if act == "alloc":
            live[e["addr"]] = (e["size"], e.get("frames", []))
            total += e["size"]
            if total > peak:
                peak, at_peak = total, dict(live)
        elif act == "free_completed" and e["addr"] in live:
            total -= live.pop(e["addr"])[0]
    sites = defaultdict(lambda: [0, 0])
    for size, frames in at_peak.values():
        s = sites[_site(frames)]
        s[0] += size
        s[1] += 1
    return peak, dict(sites)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--microbatches", type=int, default=32)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--optimizer", default="adamw-host")
    ap.add_argument("--alloc-conf", default="expandable_segments:True")
    ap.add_argument("--out", default=tempfile.gettempdir())
    a = ap.parse_args()
    if a.alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = a.alloc_conf

    import torch

    from wantrain.model.gpt import GPTConfig
    from wantrain.resources import fit
    from wantrain.scheduler.planner import Fleet, Node, Overrides, plan_job

    major, minor = torch.cuda.get_device_capability()
    free, _ = torch.cuda.mem_get_info()
    node = dict(
        tflops=80.0,
        vram_bytes=int(free),
        compute_cap=(major, minor),
        device_class=f"cuda:sm_{major}{minor}",
    )
    fleet = Fleet(
        nodes=[Node(f"n{i}", **node) for i in range(a.stages)],
        link_mbps=1000.0,
        rtt_ms=40.0,
        duplex="per_direction",
    )
    cfg = GPTConfig(
        vocab_size=50257,
        d_model=a.d,
        n_layer=a.layers,
        n_head=a.heads,
        seq_len=a.seq,
        compute_dtype="bf16",
    )
    ov = Overrides(
        recompute=True,
        compute_dtype="bf16",
        boundary_dtype="bf16",
        mb_size=1,
        microbatches=a.microbatches,
        extras={"optimizer": a.optimizer},
    )
    spec = plan_job(cfg, fleet, steps=10, device="cuda", require_margin=False, overrides=ov)["spec"]
    if spec is None:
        print("WT_MEMSNAP " + json.dumps({"ok": False, "error": "planner refused"}))
        return 1
    path = os.path.join(a.out, f"memsnap-d{a.d}-L{a.layers}-r{a.rank}.spec.json")
    open(path, "w").write(spec.to_json())

    torch.cuda.memory._record_memory_history(max_entries=4_000_000, stacks="python")
    try:
        r = fit.measure(path, a.rank)
    except BaseException as e:  # noqa: BLE001 -- an OOM is the answer here
        r = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    raw = os.path.join(a.out, f"memsnap-d{a.d}-L{a.layers}-r{a.rank}.pickle")
    with open(raw, "wb") as fh:
        pickle.dump(snap, fh)

    trace = snap["device_traces"][torch.cuda.current_device()]
    peak, sites = replay(trace)
    a_, b_ = spec.partition[a.rank]
    params = r.get("params") or 0
    state = params * (8 if a.optimizer == "adamw-host" else 16)
    top = sorted(sites.items(), key=lambda kv: -kv[1][0])[:25]
    out = {
        "ok": r.get("ok"),
        "error": r.get("error"),
        "partition": spec.partition,
        "rank": a.rank,
        "layers": b_ - a_,
        "params_m": round(params / 1e6),
        "alloc_conf": a.alloc_conf,
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "fit_peak_reserved_gib": round((r.get("peakReservedBytes") or 0) / 2**30, 2),
        "replay_peak_gib": round(peak / 2**30, 2),
        "state_gib": round(state / 2**30, 2),
        "over_state_per_layer_gib": round((peak - state) / max(1, b_ - a_) / 2**30, 3),
        "trace_events": len(trace),
        "trace_full": len(trace) >= 4_000_000,
        "top_sites_gib": [[k, round(v[0] / 2**30, 3), v[1]] for k, v in top],
        "snapshot": raw,
    }
    print("WT_MEMSNAP " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
