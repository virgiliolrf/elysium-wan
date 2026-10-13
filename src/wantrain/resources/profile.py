"""Profile one stage's F, B and W passes on the device that will run it.

The zero-bubble scheduler is solved from measured costs (Qi et al. profile
T_F, T_B, T_W before scheduling); on a heterogeneous fleet they differ per
stage, which is exactly what the per-stage scheduler consumes. The passes are
timed the way the stage runtime executes them:

  F  forward (under no_grad when the spec recomputes activations),
  B  rebuild of the graph when recomputing + dL/d(input), with every split
     linear's dW captured instead of computed (model/wgrad.py): the critical
     path,
  W  the captured dW GEMMs, the deferrable part.

The first stage has no input gradient, so its B is the rebuild plus the
gradients of parameters outside split linears (embeddings, norms); the last
stage's B starts from the loss. Median of `reps` after one warm-up.

    python -m wantrain.resources.profile --spec spec.json --rank R
prints one `WT_PROFILE {...}` line.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback


def _sync(dev):
    import torch

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def assert_training(chunk) -> None:
    """Every module of the slot's units is in training mode (BA-444): the gate
    checked the training-mode code path, so nothing else may run."""
    for i, unit in enumerate(chunk):
        for name, mod in unit.named_modules():
            if not mod.training:
                raise ValueError(f"unit {i} module {name or '<root>'!r} is not in training mode")


def take_moe_records(model, chunk) -> None:
    """Drop the MoE records a forward left on its modules (moe_experts > 0)."""
    if getattr(model, "moe_experts", 0):
        from wantrain.model.arch import take_moe_records as take

        take(chunk)


def profile_stage(spec, rank: int, peer: int = 0, reps: int = 3) -> dict:
    import torch

    from wantrain.model import precision, wgrad
    from wantrain.model.data import make_batch
    from wantrain.model.gpt import build_layers, loss_fn
    from wantrain.runtime.determinism import setup_determinism
    from wantrain.runtime.stage import compute_autocast

    spec.finalize()
    setup_determinism(spec.init_seed)
    dev = torch.device(spec.device_for(rank, peer))
    vs = rank  # one stage per rank
    a, b = spec.partition[vs]
    n_total = spec.partition[-1][1]
    m = spec.model
    hf = m.arch == "hf"
    full = build_layers(m, spec.init_seed, span=None if hf else (a, b))  # RUNTIME-I22
    if hf:
        # Units are windows over one shared tree: index them, never slice.
        chunk = torch.nn.ModuleList([full[i] for i in range(a, b)]).to(dev)
        assert_training(chunk)
    else:
        chunk = torch.nn.ModuleList(full[a:b]).to(dev)
    del full
    params = [p for p in chunk.parameters() if p.requires_grad]
    first, last = a == 0, b == n_total
    mb, S = spec.mb_size, m.seq_len
    # An hf boundary is fp32 (GPTConfig.validate refuses any other, I-1).
    bdt = torch.float32 if hf else precision.spec(m.compute_dtype)["torch_dtype"]
    g = torch.Generator().manual_seed(spec.data_seed)

    def ctx():
        return compute_autocast(m, dev.type)

    if hf:
        from wantrain.model.gpt import call_layer

    def run_layers(x, ids=None):
        if hf:
            for layer in chunk:
                x = call_layer(layer, x, ids)
        else:
            for layer in chunk:
                x = layer(x)
        take_moe_records(m, chunk)
        return x

    tf, tb, tw = [], [], []
    mem = {"f": 0, "b": 0, "w": 0}
    for rep in range(reps + 1):
        ids = None
        if hf:
            # Every HF unit reads the microbatch's ids (BA-408), as the stage does.
            ids, tgt = make_batch(
                spec.data_seed,
                rep,
                0,
                mb,
                S,
                m.vocab_size,
                dataset=spec.dataset_path,
                data_shards=spec.data_shards,
            )
            ids, tgt = ids.to(dev), tgt.to(dev)
        if first and ids is not None:
            inp = ids
        elif first:
            inp = torch.randint(0, m.vocab_size, (mb, S), generator=g).to(dev)
        else:
            inp = torch.randn(mb, S, m.d_model, generator=g).to(dev, bdt).requires_grad_(True)
        if not hf:
            tgt = torch.randint(0, m.vocab_size, (mb, S), generator=g).to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            m0 = torch.cuda.memory_allocated(dev)

        _sync(dev)
        t0 = time.perf_counter()
        if spec.recomputes(rank):
            with torch.no_grad(), ctx():
                out = run_layers(inp, ids)
        else:
            with ctx():
                out = run_layers(inp, ids)
        _sync(dev)
        t1 = time.perf_counter()
        m1 = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0

        if spec.recomputes(rank):
            with ctx():
                out = run_layers(inp, ids)
        if last:
            if hf or (m.arch == "user" and m.autocast_enabled()):
                out = out.float()  # BA-410: the loss on fp32 logits
            target, gout = loss_fn(out, tgt), None
        else:
            target, gout = out, torch.randn_like(out)
        # The runtime's B/W split: B walks the activation chain (input
        # gradient, non-linear params), W runs the deferred linear dW GEMMs.
        owned = wgrad.owned_params(chunk)
        others = [p for p in params if id(p) not in owned]
        inputs = ([] if first else [inp]) + others
        with wgrad.capture() as store:
            if inputs:
                torch.autograd.grad(target, inputs, grad_outputs=gout, allow_unused=True)
            else:
                torch.autograd.backward(target, grad_tensors=gout)
        _sync(dev)
        t2 = time.perf_counter()
        m2 = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
        wgrad.flush(store)
        _sync(dev)
        t3 = time.perf_counter()
        del out, target, gout
        m3 = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
        if rep == 0:
            continue  # warm-up: allocator, kernels, autotune
        tf.append(t1 - t0)
        tb.append(t2 - t1)
        tw.append(t3 - t2)
        if dev.type == "cuda":
            mem = {"f": m1 - m0, "b": m2 - m1, "w": m3 - m2}
    return {
        "ok": True,
        "rank": rank,
        "peer": peer,
        "device": str(dev),
        "layers": b - a,
        "first": first,
        "last": last,
        "mbSize": mb,
        "reps": reps,
        "t_f": statistics.median(tf),
        "t_b": statistics.median(tb),
        "t_w": statistics.median(tw),
        "mem": mem if dev.type == "cuda" else None,
    }


def main(argv=None) -> int:
    from wantrain.runtime.spec import JobSpec

    ap = argparse.ArgumentParser(description="profile a stage's F/B/W passes")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--peer", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    try:
        r = profile_stage(JobSpec.load(a.spec), a.rank, a.peer, a.reps)
    except BaseException as e:  # noqa: BLE001 -- reported as data
        r = {
            "ok": False,
            "rank": a.rank,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-1200:],
        }
    print("WT_PROFILE " + json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
