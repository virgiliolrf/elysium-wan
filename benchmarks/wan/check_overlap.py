"""On one GPU node, before a run: the adamw-host step overlapped with the next step (RUNTIME-I23), its copies staged
through three CUDA streams and its moments moved onto the GPU before the third step (RUNTIME-I24), against the
single-device reference, bit for bit.

    uv run --no-sync python benchmarks/wan/check_overlap.py [--out /tmp/check-overlap]

Runs a 3-stage gpt124m bf16 pipeline with recompute on this one GPU (every stage a process), 5 steps of 8
microbatches, and the reference on the same GPU; then the same with the last stage keeping F's graph (RUNTIME-I25). Prints `CHECK_OVERLAP {json}`: compared losses, the largest
difference (0.0 when exact), and every stage's WT_OPT lines (overlap on, moments resident).
"""

import argparse
import glob
import json
import os

from wantrain.launch.local import run_and_compare
from wantrain.model.gpt import make_config
from wantrain.runtime.spec import JobSpec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/check-overlap")
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec = JobSpec(
        model=make_config("gpt124m", compute_dtype="bf16", boundary_dtype="bf16"),
        steps=a.steps,
        microbatches=8,
        stages=3,
        recompute=True,
        optimizer="adamw-host",
        wire_codec="planes",  # the production wire: the raw codec carries no bf16 boundary
        device="cuda",
        out_dir=a.out,
        tag="check-overlap",
        port_base=29700,
        dial_timeout_s=60,
    )
    res = run_and_compare(spec, timeout=900)
    # RUNTIME-I25: the last stage keeps F's graph, against the same reference
    keep = JobSpec.from_json(spec.to_json())
    keep.recompute_stages = [True, True, False]
    keep.tag, keep.port_base, keep.out_dir = "check-overlap-keep", 29760, a.out + "-keep"
    os.makedirs(keep.out_dir, exist_ok=True)
    res_keep = run_and_compare(keep, timeout=900)
    opt = []
    for f in sorted(glob.glob(os.path.join(a.out, "*.log"))):
        opt += [line.strip() for line in open(f, errors="replace") if line.startswith("WT_OPT")]
    fields = ("ok", "compared", "max_abs_diff", "first_diff", "error")
    print(
        "CHECK_OVERLAP "
        + json.dumps(
            {
                **{k: res.get(k) for k in fields},
                "keep_graph_last": {k: res_keep.get(k) for k in fields},
                "wt_opt": opt,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
