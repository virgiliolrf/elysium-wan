#!/usr/bin/env python
"""Resource allocation check: does this rank's stage fit on this device? Measure it rather than predict it.

The planner's analytic memory model prices the tensors it can name, and for a
user-supplied model it describes code it has never seen (a grouped-query block
that repeat_interleaves K and V materialises extra full-width tensors per
layer). So this builds this rank's real slice at the real microbatch size,
holds as many graphs as the runtime holds, runs backward and one optimizer
step, and reports the allocator's peak. Prints one `WT_FIT {...}` line and
exits 0 even when the stage does not fit: not fitting is a measurement.
"""

import argparse
import json
import sys
import traceback


def peak_pending(order) -> int:
    """Largest number of microbatches between F and W at any point of order."""
    cur = peak = 0
    for op, _mb in order:
        cur += 1 if op == "F" else (-1 if op == "W" else 0)
        peak = max(peak, cur)
    return max(1, peak)


def measure(spec_path: str, rank: int, force: bool = False, peer: int = 0) -> dict:
    import torch

    from wantrain.model.data import make_batch
    from wantrain.model.gpt import build_layers, loss_fn
    from wantrain.resources.profile import assert_training, take_moe_records
    from wantrain.runtime import optim
    from wantrain.runtime.determinism import setup_determinism
    from wantrain.runtime.spec import JobSpec

    with open(spec_path) as fh:
        spec = JobSpec.from_json(fh.read())

    dev = torch.device(spec.device_for(rank, peer))
    # `--force` runs the whole build/forward/backward on whatever device the spec
    # names, reporting no memory numbers. It exists so this script can be
    # exercised off a GPU: a checker whose only test environment is a rented card
    # is a checker that debuts its bugs on a paid fleet.
    if dev.type != "cuda" and not force:
        return {"ok": True, "skipped": f"device is {spec.device}, nothing to measure"}
    cuda = dev.type == "cuda"

    spec.finalize()
    # The runtime's flags and allocator, before the first CUDA call: without them the caching allocator runs without
    # expandable segments and the fit measured fragmentation the stage never has (85.4 GiB here against 77.3 GiB
    # under the runtime's allocator, d=5632 on an H100 NVL, 2026-10-12), refusing a plan that fits. profile.py does
    # the same.
    setup_determinism(spec.init_seed)
    my_vs = [rank]  # one stage per rank

    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
        free_before = int(torch.cuda.mem_get_info(dev.index or 0)[0])
    else:
        free_before = 0

    m = spec.model
    hf = m.arch == "hf"
    # RUNTIME-I22: a non-hf build keeps only this rank's entries resident.
    full = build_layers(m, spec.init_seed, span=None if hf else tuple(spec.partition[rank]))
    chunks = []
    n_layers = 0
    for vs in my_vs:
        a, b = spec.partition[vs]
        if hf:
            # Units are windows over one shared tree: index them, never slice.
            chunks.append(torch.nn.ModuleList([full[i] for i in range(a, b)]).to(dev))
            assert_training(chunks[-1])
        else:
            chunks.append(torch.nn.ModuleList(full[a:b]).to(dev))
        n_layers += b - a
    params = [p for c in chunks for p in c.parameters()]
    del full

    # AdamW's state is lazily allocated on the first step, so building the
    # optimizer is not enough -- its exp_avg/exp_avg_sq are two more copies of
    # every parameter and they must be part of the peak being measured.
    # "adamw-host" (RUNTIME-I20) keeps them in host memory: its step's chunks are the peak then.
    opt = optim.build(spec.optimizer, params, spec.lr)

    S = spec.model.seq_len
    mb = spec.mb_size
    # The stage holding the last layer also holds the logits, which for a 50k
    # vocabulary is the single largest tensor in the run. Whether this rank pays
    # that cost is decided by the partition, not assumed.
    end = max(b for _, b in spec.partition)
    is_last = spec.partition[my_vs[-1]][1] == end

    # Hold what the runtime holds: the most microbatches this slot's schedule
    # ever has between their F and their W (in flight forward, or past B with
    # dW pending). Measuring one microbatch is off by exactly that factor.
    hold = {rank: peak_pending(spec.schedule_order[rank][peer])}

    # Over-counts on purpose: a deferred-W graph is smaller than a live forward
    # graph. Over-estimating costs throughput; under-estimating costs the fleet.

    if hf:
        from wantrain.model.gpt import call_layer
        from wantrain.runtime.stage import compute_autocast

    recompute_path = spec.recomputes(rank) and not hf
    if recompute_path:
        # Measure what the runtime holds under recompute, not a full graph per held microbatch: F keeps only its
        # input (the graph is rebuilt in B, one microbatch at a time), and every B defers its weight gradients
        # into the store until W, as the runtime's SplitLinear layers do. Holding `hold` live fp32 graphs instead
        # charged a 182M-parameter stage 7.5 GB on an 8 GB card and refused every plan (2026-10-12).
        _recompute_peak(spec, chunks, my_vs, rank, peer, params, dev, mb, S)
    live = []
    for ci, vs in enumerate([] if recompute_path else my_vs):
        first_here = spec.partition[vs][0] == 0
        for i in range(hold[vs]):
            if hf:
                # Every HF unit reads the microbatch's ids (BA-408), on this
                # device, inside the stage's autocast region (BA-409).
                ids, tgt = make_batch(
                    spec.data_seed,
                    0,
                    i,
                    mb,
                    S,
                    m.vocab_size,
                    dataset=spec.dataset_path,
                    data_shards=spec.data_shards,
                )
                ids, tgt = ids.to(dev), tgt.to(dev)
                if first_here:
                    x = ids
                    inp = None
                else:
                    inp = torch.randn(mb, S, m.d_model, device=dev, requires_grad=True)
                    x = inp
                with compute_autocast(m, dev.type):
                    for layer in chunks[ci]:
                        x = call_layer(layer, x, ids)
                take_moe_records(m, chunks[ci])
                live.append((x, inp, spec.partition[vs][1] == end, tgt))
                continue
            if first_here:
                x = torch.randint(0, spec.model.vocab_size, (mb, S), device=dev)
                inp = None
            else:
                inp = torch.randn(mb, S, spec.model.d_model, device=dev, requires_grad=True)
                x = inp
            for layer in chunks[ci]:
                x = layer(x)
            take_moe_records(m, chunks[ci])
            live.append((x, inp, spec.partition[vs][1] == end, None))

    # Only now unwind them, so the peak includes every graph standing together.
    for out, _inp, last, hf_tgt in live:
        if last and hf:
            loss_fn(out.float(), hf_tgt).backward()  # BA-410: the loss on fp32 logits
        elif last:
            tgt = torch.randint(0, spec.model.vocab_size, (mb, S), device=dev)
            loss_fn(out, tgt).backward()
        else:
            out.backward(torch.randn_like(out))
    del live

    # One optimizer step, so the lazily-created Adam moments are in the peak.
    opt.step()
    if cuda:
        torch.cuda.synchronize(dev)
        peak = int(torch.cuda.max_memory_allocated(dev))
        reserved = int(torch.cuda.max_memory_reserved(dev))
    else:
        peak = reserved = 0
    return {
        "ok": True,
        "rank": rank,
        "virtualStages": my_vs,
        "layers": n_layers,
        "params": sum(p.numel() for p in params),
        "mbSize": mb,
        "seqLen": S,
        "holdsHead": bool(is_last),
        "concurrentGraphs": sum(hold.values()),
        # allocated is what tensors need; reserved is what the caching allocator
        # took from the driver, and the driver is what runs out. Judge on
        # reserved: an allocator that has fragmented its pools cannot hand the
        # memory back to cuBLAS just because no tensor is holding it.
        "peakAllocatedBytes": peak,
        "peakReservedBytes": reserved,
        "freeBytesBefore": free_before,
    }


def _recompute_peak(spec, chunks, my_vs, rank, peer, params, dev, mb, S) -> None:
    """The recompute schedule's memory, replayed for real over this slot's pinned order: F runs without a graph and
    keeps its input; B rebuilds the graph of that one microbatch and defers its weight gradients into its own store
    (SplitLinear), freeing the graph; W runs that store's GEMMs and folds each gradient into the running sum. What is
    resident at every point is what the runtime holds then: the inputs of the microbatches between F and B, the
    stores of those between B and W, one rebuilt graph. Building the stores of every microbatch between F and W at
    once (the order's in-flight peak) charged a 24 GB stage with ~20 pending stores it never holds together, and
    refused an 8.4B MoE plan that fits (2026-10-12)."""
    import torch

    from wantrain.model import wgrad
    from wantrain.model.gpt import loss_fn
    from wantrain.resources.profile import take_moe_records
    from wantrain.runtime.stage import compute_autocast

    m = spec.model
    end = max(b for _, b in spec.partition)
    for c in chunks:
        wgrad.convert(c)
    ci, vs = 0, my_vs[0]
    first_here = spec.partition[vs][0] == 0
    last_here = spec.partition[vs][1] == end
    by_id = {id(p): p for p in params}
    inputs: dict = {}
    stores: dict = {}
    for op, i in spec.schedule_order[rank][peer]:
        if op == "F":
            if first_here:
                x0 = torch.randint(0, m.vocab_size, (mb, S), device=dev)
            else:
                x0 = torch.randn(mb, S, m.d_model, device=dev)
            with torch.no_grad(), compute_autocast(m, dev.type):
                x = x0
                for layer in chunks[ci]:
                    x = layer(x)
            take_moe_records(m, chunks[ci])
            del x
            inputs[i] = x0
        elif op == "B":
            x0 = inputs.pop(i)
            inp = x0 if first_here else x0.detach().requires_grad_(True)
            with compute_autocast(m, dev.type):
                x = inp
                for layer in chunks[ci]:
                    x = layer(x)
            take_moe_records(m, chunks[ci])
            with wgrad.capture() as store:
                if last_here:
                    loss_fn(x, torch.randint(0, m.vocab_size, (mb, S), device=dev)).backward()
                else:
                    x.backward(torch.randn_like(x))
            stores[i] = store
            del x, inp, x0
        elif op == "W":
            # W as the runtime folds it (RUNTIME-I21): each gradient into the running sum as it completes, B's
            # already summed into .grad by backward; never a whole gradient set beside the sum.
            for q, g in wgrad.flush_each(stores.pop(i), widen=False):
                p = by_id.get(id(q))
                if p is not None:
                    if p.grad is None:
                        p.grad = g if g.dtype == p.dtype else g.to(p.dtype)
                    else:
                        p.grad.add_(g)
                del g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="measure a stage's real peak memory")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--peer", type=int, default=0)
    ap.add_argument(
        "--force", action="store_true", help="run the build/forward/backward off-GPU, for testing"
    )
    args = ap.parse_args(argv)
    try:
        r = measure(args.spec, args.rank, force=args.force, peer=args.peer)
    except BaseException as e:  # noqa: BLE001
        # An OOM here is the answer, not a failure: it means the stage does not
        # fit, which is exactly what this was run to find out. Report it as data.
        r = {
            "ok": False,
            "rank": args.rank,
            "error": f"{type(e).__name__}: {e}",
            "oom": "out of memory" in str(e).lower() or "CUBLAS_STATUS_ALLOC_FAILED" in str(e),
            "traceback": traceback.format_exc()[-1200:],
        }
    print("WT_FIT " + json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
