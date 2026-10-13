"""On one GPU node, before a run: a re-cut checkpoint resumes bit for bit on CUDA (RUNTIME-I26).

    uv run --no-sync python benchmarks/wan/check_recut.py [--out /tmp/check-recut]

A 3-stage gpt124m bf16 pipeline with recompute and adamw-host on this one GPU: 6 steps uninterrupted; 3 steps with
checkpoints, the waves re-cut offline to another partition (layers moved both ways), then resumed to step 6 under
the new cut. Prints `CHECK_RECUT {json}`: the cuts, the losses compared and whether every one matched bit for bit.
"""

import argparse
import json
import os

import torch

from wantrain.fault_tolerance.checkpoint import WAVE_STATE, commit_wave, wave_dir
from wantrain.launch.local import compare, run_local
from wantrain.model.gpt import make_config
from wantrain.runtime.recut import entry_param_names, recut_states
from wantrain.runtime.spec import JobSpec


def _spec(out, tag, port, steps, partition=None):
    return JobSpec(
        model=make_config("gpt124m", compute_dtype="bf16", boundary_dtype="bf16"),
        steps=steps,
        microbatches=8,
        stages=3,
        recompute=True,
        optimizer="adamw-host",
        wire_codec="planes",  # the production wire: the raw codec carries no bf16 boundary
        device="cuda",
        out_dir=out,
        tag=tag,
        port_base=port,
        dial_timeout_s=60,
        partition=partition,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/check-recut")
    a = ap.parse_args()
    full_dir, part_dir = os.path.join(a.out, "full"), os.path.join(a.out, "part")
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(part_dir, exist_ok=True)
    out = {}
    full = _spec(full_dir, "full", 29800, 6)
    out["full"] = run_local(full, timeout=900).get("ok")
    part = _spec(part_dir, "part", 29840, 3)
    part.ckpt_every = 1
    out["part"] = run_local(part, timeout=900).get("ok")
    part.finalize()
    n = part.partition[-1][1]
    new = [[0, 2], [2, n - 4], [n - 4, n]]
    out["cut"] = {"from": part.partition, "to": new}
    states = [
        torch.load(
            os.path.join(wave_dir(part_dir, "part", r, 0, 2), WAVE_STATE),
            map_location="cpu",
            weights_only=True,
        )
        for r in range(3)
    ]
    for r, payload in enumerate(
        recut_states(states, part.partition, new, entry_param_names(part.model))
    ):
        commit_wave(part_dir, "part.cut", r, 0, 2, lambda path, x=payload: torch.save(x, path))
    resumed = _spec(part_dir, "part.cut", 29880, 6, partition=new)
    resumed.resume = True
    out["resumed"] = run_local(resumed, timeout=900).get("ok")
    r = compare(
        os.path.join(full_dir, "full_losses.json"), os.path.join(part_dir, "part.cut_losses.json")
    )
    out["compare"] = {k: r.get(k) for k in ("ok", "compared", "max_abs_diff", "first_diff")}
    print("CHECK_RECUT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
