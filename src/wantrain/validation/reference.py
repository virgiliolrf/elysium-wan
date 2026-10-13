#!/usr/bin/env python
"""Reference trainer (the oracle): single-device deterministic baseline.

Runs the declared reference recipe in one process and records the
per-microbatch loss sequence; every distributed run is diffed against it
(success criterion: max abs diff == 0). Gradient accumulation mirrors the
engine's exactness contract: per-microbatch gradients fold into one running
sum in fixed microbatch-index order, bitwise identical to storing them all
and summing at the end.
"""

import argparse
import contextlib
import json
import os
import sys
import time

import torch

from wantrain.model.data import make_batch
from wantrain.model.gpt import build_layers, loss_fn
from wantrain.runtime import optim
from wantrain.runtime.determinism import float_hex, setup_determinism
from wantrain.runtime.spec import JobSpec
from wantrain.validation.state import aux_hex, reference_manifest, write_manifest


class OracleRefused(ValueError):
    """The oracle cannot run this recipe as the datacenter would; names why."""


def oracle_mode(device_classes, hosts=None) -> str:
    """BA-420: how the arch "hf" oracle runs a device map. device_classes[stage][peer]
    is each slot's class (hfm.device_class); hosts[stage][peer] each slot's host,
    None when one host holds the map (a local run).
    -> "vanilla": one class, HF's own ForCausalLM.forward in one process on it;
    -> "chain": a mixed map one host holds, the run's units in one process, each
       on its map device (one class per stage, one CPU class on a host);
    any other map is refused, naming the classes (MODEL-I9, CONTROL-I24)."""
    if not device_classes or any(not row for row in device_classes):
        raise OracleRefused(
            "arch='hf' oracle refused: the map has no device classes (the launch gate "
            "fills spec.device_classes)"
        )
    classes = sorted({c for row in device_classes for c in row})
    if len(classes) == 1:
        return "vanilla"
    where = sorted({h for row in hosts for h in row}) if hosts else []
    if len(where) > 1:
        raise OracleRefused(
            f"arch='hf' oracle refused: the map mixes device classes {classes} across hosts "
            f"{where}; a mixed map has an oracle only when one host holds it"
        )
    cpus = [c for c in classes if c.split(":", 1)[0] == "cpu"]
    if len(cpus) > 1:
        raise OracleRefused(
            f"arch='hf' oracle refused: the map names CPU classes {cpus}; one host has one CPU class"
        )
    for r, row in enumerate(device_classes):
        if len(set(row)) > 1:
            raise OracleRefused(
                f"arch='hf' oracle refused: stage {r} mixes device classes {sorted(set(row))}; "
                f"the unit chain runs each unit on one class"
            )
    return "chain"


def _hosts(spec: JobSpec):
    """Each slot's host, or None when the map is local (no endpoints)."""
    if not spec.endpoints:
        return None
    return [[str(e[0]) for e in row] for row in spec.endpoints]


def _refuse(flag: str, why: str):
    raise OracleRefused(f"arch='hf' oracle refused at start: {flag}: {why}")


def _assert_training(modules) -> None:
    """BA-444: every module runs in training mode, the code path the gate and
    the stages run; names the first that is not."""
    for name, m in modules:
        if not m.training:
            _refuse("training mode", f"module {name or '<root>'!r} is in eval mode")


def _hf_start_asserts(spec: JobSpec, units) -> None:
    """BA-406: after the build, the stage's start checks, each fatal naming its flag."""
    from torch import nn

    from wantrain.model import hfm
    from wantrain.runtime.determinism import check_determinism

    check_determinism()
    root = units.root
    cfg = root.config
    impl = getattr(cfg, "_attn_implementation", None)
    if impl != "eager":
        _refuse("_attn_implementation", f"{impl!r}, the oracle runs 'eager'")
    for k, v in sorted(cfg.to_dict().items()):
        dropout = "dropout" in k or k.endswith("pdrop")
        if dropout and isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0:
            _refuse("dropout", f"config.{k} = {v}")
    if getattr(cfg, "use_cache", None) is not False:
        _refuse("use_cache", f"config.use_cache is {getattr(cfg, 'use_cache', None)!r}, not False")
    if getattr(root, "is_gradient_checkpointing", False):
        _refuse("gradient_checkpointing", "HF gradient checkpointing is on in the model")
    for name, m in list(root.named_modules()) + list(units.named_modules()):
        if isinstance(m, nn.Dropout) and m.p != 0:
            _refuse("dropout", f"{name}.p = {m.p}")
        for k, v in vars(m).items():
            if "dropout" in k and isinstance(v, float) and v != 0:
                _refuse("dropout", f"{name}.{k} = {v}")
        if getattr(m, "gradient_checkpointing", False):
            _refuse(
                "gradient_checkpointing", f"HF gradient checkpointing is on in {name or '<root>'!r}"
            )
    _assert_training(list(root.named_modules()) + list(units.named_modules()))
    have = hfm.hf_code()
    if have != spec.model.hf_code:
        _refuse("hf_code", f"installed {have} != pinned {spec.model.hf_code}")


def _hf_build(spec: JobSpec):
    """BA-406, BA-420: the arch "hf" model in the stage's start order, placed by
    the oracle mode. -> (units, per-unit devices, mode)."""
    from wantrain.model import hfm

    if not spec.model.hf_buffers:
        raise OracleRefused("arch='hf' oracle refused: spec.model.hf_buffers is not pinned")
    mode = oracle_mode(spec.device_classes, _hosts(spec))
    hfm.guard_environment()
    units = build_layers(spec.model, spec.init_seed)
    hfm.apply_buffer_record(units.root, spec.model.hf_buffers)
    if mode == "vanilla":
        device = torch.device(spec.device_for(0, 0))
        units.root.to(device)
        units.to(device)
        devices = [device] * len(units)
    else:
        fin = spec if spec.partition is not None else JobSpec.from_json(spec.to_json()).finalize()
        devices = []
        for r, (a, b) in enumerate(fin.partition):
            devices += [torch.device(fin.device_for(r, 0))] * (b - a)
        for u, d in zip(units, devices):
            u.to(d)
        tied = getattr(units, "tied", None)
        if tied is not None and devices[tied[0]] != devices[tied[1]]:
            raise OracleRefused(
                f"arch='hf' oracle refused: the tied table's units {tied} are on "
                f"{devices[tied[0]]} and {devices[tied[1]]}; the chain holds one table"
            )
    _hf_start_asserts(spec, units)
    return units, devices, mode


def _hf_loss(spec: JobSpec, units, devices, mode, x, tgt, recompute):
    """BA-407..BA-410: the oracle's forward of one microbatch and its loss."""
    from wantrain.model.gpt import call_layer
    from wantrain.runtime.stage import compute_autocast

    m = spec.model
    if mode == "vanilla":
        # HF's own forward, whole: recompute moves no bit (MODEL-I9), so the
        # oracle keeps the datacenter's function as it is.
        ids = x.to(devices[0])
        with compute_autocast(m, devices[0].type, m.autocast_enabled()):
            logits = units.vanilla(ids)
    else:
        h = x
        for layer, d in zip(units, devices):
            ids, h = x.to(d), h.to(d)
            with compute_autocast(m, d.type, m.autocast_enabled()):
                if recompute:
                    h = torch.utils.checkpoint.checkpoint(
                        call_layer, layer, h, ids, use_reentrant=False
                    )
                else:
                    h = call_layer(layer, h, ids)
        logits = h
    return loss_fn(logits.float(), tgt.to(logits.device))


def _moe_key(recs) -> list:
    return [(r.layer, aux_hex(r.aux), r.counts.tolist()) for r in recs]


def run_baseline(
    spec: JobSpec,
    tag: str,
    branch_step: int | None = None,
    probe_mb: int | None = None,
    recompute: bool = False,
):
    setup_determinism(spec.init_seed)
    hf = spec.model.arch == "hf"
    # MoE layers record aux and counts in every forward (I-11); an HF model's
    # swapped blocks do too, so every hf model is walked for records.
    moe = hf or getattr(spec.model, "moe_experts", 0) > 0
    if hf:
        layers, unit_devices, mode = _hf_build(spec)
        device = unit_devices[-1]
    else:
        mode = None
        device = torch.device(spec.device)
        layers = build_layers(spec.model, spec.init_seed).to(device)
    if moe:
        from wantrain.model.arch import moe_bias_update, take_moe_records

        _assert_training(layers.named_modules())
    aux = []
    # Trainable only, in the same traversal order the stage runtime uses:
    # autograd.grad raises on a frozen tensor, and this is the sequence the
    # distributed run is compared against, so the order must match.
    if hf:
        # BA-411: the units' parameters in unit order, a tied table once.
        params = list(
            {id(p): p for u in layers for p in u.parameters() if p.requires_grad}.values()
        )
    else:
        params = [p for p in layers.parameters() if p.requires_grad]
    opt = optim.build(spec.optimizer, params, spec.lr)  # RUNTIME-I20: the stages' optimizer
    losses = []
    timings = []
    for step in range(spec.steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        # Phase 4 branching probe: deterministic replay of the base recipe up
        # to branch_step, then the probe batch size takes over
        m = spec.microbatches
        if branch_step is not None and step >= branch_step:
            if probe_mb is None:
                raise ValueError("branch_step needs probe_mb")
            m = probe_mb
        acc = None
        counts: dict = {}
        for mb in range(m):
            x, tgt = make_batch(
                spec.data_seed,
                step,
                mb,
                spec.mb_size,
                spec.model.seq_len,
                spec.model.vocab_size,
                dataset=spec.dataset_path,
                data_shards=spec.data_shards,
            )
            # A recompute replays F in the backward and an MoE layer records
            # again; torch's default early stop would end the replay at the last
            # saved tensor, before moe_forward appends its record, leaving
            # MOE-REPLAY nothing to check. Running the replay whole changes no
            # value, only how much of F is re-run. The vanilla HF path never
            # recomputes (MODEL-I9).
            replays = moe and recompute and not (hf and mode == "vanilla")
            with (
                torch.utils.checkpoint.set_checkpoint_early_stop(False)
                if replays
                else contextlib.nullcontext()
            ):
                if hf:
                    loss = _hf_loss(spec, layers, unit_devices, mode, x, tgt, recompute)
                else:
                    h = x.to(device)
                    for layer in layers:
                        if recompute:
                            # Same math re-executed in the backward, so the gradient
                            # stream is bit-identical to the plain path; only peak
                            # memory changes, which lets a card-filling model run.
                            h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
                        else:
                            h = layer(h)
                    loss = loss_fn(h, tgt.to(device))
            if moe:
                recs = take_moe_records(layers)
            gs = torch.autograd.grad(loss, params)
            if moe:
                # The replay's records must equal F's (MOE-REPLAY, as the stage
                # checks), and nothing records outside F and its replay; none
                # reaches the next microbatch.
                replay = take_moe_records(layers)
                if _moe_key(replay) != (_moe_key(recs) if replays else []):
                    raise RuntimeError(
                        f"MOE-REPLAY: step {step} mb {mb}: the recompute's records differ "
                        f"from F's ({len(replay)} replayed, {len(recs)} in F)"
                    )
                for rec in recs:
                    aux.append((step, mb, rec.layer, aux_hex(rec.aux)))
                    # Integer add in microbatch order (RUNTIME-I18).
                    c = counts.get(rec.layer)
                    counts[rec.layer] = rec.counts.clone() if c is None else c + rec.counts
            losses.append(
                {"step": step, "mb": mb, "loss_hex": float_hex(loss.item()), "loss": loss.item()}
            )
            # Fixed-index-order accumulation, in place: bitwise identical to
            # storing every microbatch's gradients and summing at the end (same
            # adds, same order, same values), but holding one gradient set.
            if acc is None:
                acc = [g.clone() for g in gs]
            else:
                for a, g in zip(acc, gs):
                    a.add_(g)
            del gs
        assert acc is not None, "a step with zero microbatches"
        for p, g in zip(params, acc):
            p.grad = g
        opt.step()
        if counts:
            # BA-453: the selection biases move after the step's AdamW, from
            # the int64 counts of every microbatch of the step.
            moe_bias_update(layers, counts, m * spec.mb_size * spec.model.seq_len)
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - t0
        tokens = m * spec.mb_size * spec.model.seq_len
        timings.append(
            {
                "step": step,
                "wall_s": round(wall, 4),
                "tokens": tokens,
                "tok_s": round(tokens / wall, 1),
            }
        )
        metric = {
            "step": step,
            "loss": losses[-1]["loss"],
            "tok_s": timings[-1]["tok_s"],
            "t": int(time.time() * 1000),
        }
        print(f"WT_METRIC {json.dumps(metric)}", flush=True)
    os.makedirs(spec.out_dir, exist_ok=True)
    # The full state (weights, AdamW state, RNG, losses) the byte-exact rule
    # compares; written before the losses file, which marks the run complete.
    write_manifest(
        os.path.join(spec.out_dir, f"{tag}_state.json"),
        reference_manifest(
            layers, opt, [(r["step"], r["mb"], r["loss_hex"]) for r in losses], aux or None
        ),
    )
    out_path = os.path.join(spec.out_dir, f"{tag}_losses.json")
    with open(out_path, "w") as f:
        json.dump({"losses": losses}, f, indent=2)
    if timings:
        # Steady-state rate: drop the first step (compile/allocator warmup),
        # which on a fresh box can be 10x the rest and would poison a
        # cost-per-token figure derived from the mean.
        steady = timings[1:] or timings
        tok_s = sum(t["tokens"] for t in steady) / sum(t["wall_s"] for t in steady)
        with open(os.path.join(spec.out_dir, f"{tag}_timing.json"), "w") as f:
            json.dump(
                {
                    "steps": timings,
                    "steady_tok_s": round(tok_s, 1),
                    "total_wall_s": round(sum(t["wall_s"] for t in timings), 2),
                },
                f,
                indent=2,
            )
        print(f"WT_TIMING steady {tok_s:.1f} tok/s over {len(steady)} steps", flush=True)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="single-device reference run of a spec")
    ap.add_argument("--spec", help="JobSpec JSON file (topology fields ignored)")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None, help="override the spec's device")
    ap.add_argument(
        "--branch-step",
        type=int,
        default=None,
        help="step at which the probe batch size starts (noise-scale probes)",
    )
    ap.add_argument("--probe-mb", type=int, default=None)
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="checkpoint each layer; bit-identical, fits big models",
    )
    args = ap.parse_args(argv)
    spec = JobSpec.load(args.spec) if args.spec else JobSpec()
    if args.out_dir:
        spec.out_dir = args.out_dir
    if args.device:
        spec.device = args.device
    path = run_baseline(spec, args.tag, args.branch_step, args.probe_mb, recompute=args.recompute)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
