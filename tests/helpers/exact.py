"""Shared pieces of the byte-exact suites (tests/oracle, tests/wan): the
recipe, the every-mechanism-on topology and the reference's canonical state."""

from __future__ import annotations

import json

from wantrain.launch.local import ensure_reference, links_for, reference_state_path
from wantrain.model.gpt import make_config
from wantrain.runtime.spec import JobSpec

RECIPE = dict(steps=4, microbatches=8)
# Every optional mechanism at once. Stage 1 is a wide stage of three peers
# whose training state is cut into per-tensor pieces (shard "pieces", wire
# "native"), with weighted microbatch ownership (IWRR over 1 : 0.5 : 0.25),
# streamed gradients and W emitting in forward order.
FEATURES_ON = dict(
    stages=3,
    stage_widths=[1, 3, 1],
    peer_weights=[[1.0], [1.0, 0.5, 0.25], [1.0]],
    stream_grads=True,
    recompute=True,
    wire_codec="planes",
    p2p_streams=2,
    adapt_budget_ms=8,
    wire="native",
    shard="pieces",
    emission_order=["store", "forward", "store"],
)
FEATURES_OFF = dict(stages=3)
# Ports per run: rank r peer p listens on base + 8r + p, so 3 stages use 17.
RUN_PORTS = 20


def make_spec(out_dir, tag, port_base, features, links=None, **extra) -> JobSpec:
    """links: kwargs of links_for (one profile for every link), or a list with
    one LinkCfg dict per stage."""
    spec = JobSpec(
        model=make_config("toy"),
        out_dir=str(out_dir),
        tag=tag,
        port_base=port_base,
        dial_timeout_s=60,
        **{**RECIPE, **features, **extra},
    )
    spec.links = list(links) if isinstance(links, list) else links_for(spec, **(links or {}))
    return spec


def reference_manifest(out_dir, **recipe) -> dict:
    spec = JobSpec(model=make_config("toy"), out_dir=str(out_dir), **{**RECIPE, **recipe})
    with open(reference_state_path(ensure_reference(spec, spec.out_dir))) as f:
        return json.load(f)


def reference_manifest_for(spec: JobSpec, out_dir) -> dict:
    """The reference state of an arbitrary recipe (topology fields ignored)."""
    path = ensure_reference(JobSpec.from_json(spec.to_json()), str(out_dir))
    with open(reference_state_path(path)) as f:
        return json.load(f)


def fleet_manifest(cluster, res, out_dir) -> dict:
    """The canonical state of a coordinator-run job, from the final checkpoint
    of every slot on the workers' disks. A healed slot has shards on both the
    dead and the replacement worker: the newest sidecar step wins, and every
    slot must have reached the last step. A shard without a sidecar is skipped:
    a stage killed between invalidating and rewriting its sidecar leaves one,
    and its bytes are unproven."""
    import glob
    import os
    import re
    import shutil

    from wantrain.validation.state import distributed_manifest

    with open(os.path.join(res.run_dir, "spec.json")) as f:
        spec = JobSpec.from_json(f.read())
    best: dict = {}
    for workdir in cluster.workdirs.values():
        pattern = os.path.join(workdir, "runs", res.run_id, "out", "*_ckpt_rank*p*.pt")
        for pt in glob.glob(pattern):
            m = re.search(r"_ckpt_rank(\d+)p(\d+)\.pt$", pt)
            try:
                with open(pt[:-3] + ".step.json") as f:
                    step = json.load(f)["step"]
            except FileNotFoundError:
                continue
            slot = (int(m.group(1)), int(m.group(2)))
            if slot not in best or step > best[slot][0]:
                best[slot] = (step, pt)
    os.makedirs(out_dir, exist_ok=True)
    for (r, p), (step, pt) in best.items():
        assert step == spec.steps - 1, f"slot r{r}p{p} ended at step {step}"
        shutil.copy(pt, os.path.join(out_dir, f"{spec.tag}_ckpt_rank{r}p{p}.pt"))
    shutil.copy(
        os.path.join(res.run_dir, "losses.json"), os.path.join(out_dir, f"{spec.tag}_losses.json")
    )
    spec.out_dir = str(out_dir)
    return distributed_manifest(spec)
