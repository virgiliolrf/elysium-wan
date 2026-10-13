"""Validate a user-supplied model against what the engine can distribute.

The engine assigns contiguous slices of a flat layer list to pipeline stages,
so a model must reduce to that shape, every layer must take and return one
tensor, and the forward must be deterministic or the exactness claim is void
while the loss curve looks fine. This gate runs the user's `build(cfg)` at
tiny shapes before any GPU is rented; memory fit and the bandwidth floor are
priced later by the planner. Security: this executes untrusted code. The
hardened subprocess that calls it (stripped credentials, temp cwd, resource
limits) is defense in depth, not a sandbox; multi-tenant use needs a container.
"""

import json
import sys
import traceback

import torch
import torch.nn as nn

TINY = dict(vocab_size=256, n_layer=4, n_head=4, d_model=64, seq_len=32)


class ValidationError(Exception):
    pass


def _fail(check, detail, fix=None):
    return dict(check=check, ok=False, detail=detail, fix=fix)


def _ok(check, detail=""):
    return dict(check=check, ok=True, detail=detail)


def load_build(source: str):
    """Exec the user's source and return its `build` callable."""
    ns: dict = {}
    code = compile(source, "<user_model>", "exec")
    # validating user code is the whole point
    exec(code, ns)  # noqa: S102  # nosec B102
    build = ns.get("build")
    if build is None:
        raise ValidationError(
            "no `build(cfg)` function found. Define exactly one:\n\n"
            "    def build(cfg):\n"
            "        return [Embed(...), Block(...), ..., Head(...)]"
        )
    if not callable(build):
        raise ValidationError("`build` must be a function taking a config object")
    return build


class Cfg:
    """Attribute-style config, so user code can write cfg.d_model. It carries
    every field the real GPTConfig carries, not only the five the checks vary:
    a Llama-shaped model reads cfg.n_kv_head and cfg.d_ff, and a check that
    supplied less would answer a different question than the run will ask."""

    def __init__(self, d):
        try:
            from dataclasses import fields

            from wantrain.model.gpt import GPTConfig

            base = GPTConfig()
            for f in fields(GPTConfig):
                self.__dict__[f.name] = getattr(base, f.name)
        except Exception:  # noqa: BLE001
            # The checks still work on the five fields they vary; better a
            # narrower cfg than no validation at all.
            pass
        self.__dict__.update(d)

    def __repr__(self):
        return f"Cfg({self.__dict__})"


def check_flat_list(layers):
    if isinstance(layers, nn.Sequential):
        layers = list(layers)
    if isinstance(layers, nn.ModuleList):
        layers = list(layers)
    if not isinstance(layers, (list, tuple)):
        return None, _fail(
            "flat_layer_list",
            f"build() returned {type(layers).__name__}, which cannot be cut into pipeline stages",
            "Return a list (or nn.Sequential / nn.ModuleList) of layers. The engine "
            "assigns contiguous slices of that list to stages, so the model has to be "
            "a sequence, not a nested module graph.",
        )
    layers = list(layers)
    if len(layers) < 3:
        return None, _fail(
            "flat_layer_list",
            f"only {len(layers)} layer(s); a pipeline needs at least an embedding, "
            f"one block and a head",
        )
    bad = [i for i, l in enumerate(layers) if not isinstance(l, nn.Module)]
    if bad:
        return None, _fail(
            "flat_layer_list",
            f"positions {bad} are not nn.Module instances",
        )
    return layers, _ok("flat_layer_list", f"{len(layers)} layers")


def check_forward(layers, cfg):
    """Every layer must take one tensor and return one tensor."""
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    x = idx
    shapes = []
    for i, layer in enumerate(layers):
        try:
            x = layer(x)
        except Exception as e:
            return None, _fail(
                "single_tensor_boundaries",
                f"layer {i} ({type(layer).__name__}) raised on a "
                f"{tuple(idx.shape) if i == 0 else 'hidden-state'} input: "
                f"{type(e).__name__}: {e}",
                "Each layer must accept exactly one tensor and return exactly one "
                "tensor. Extra inputs (masks, position ids, caches) have to be "
                "computed inside the layer, because a stage boundary carries a "
                "single tensor over the network.",
            )
        if not isinstance(x, torch.Tensor):
            return None, _fail(
                "single_tensor_boundaries",
                f"layer {i} ({type(layer).__name__}) returned {type(x).__name__}, not a Tensor",
                "Return the hidden state only. A tuple cannot cross a stage "
                "boundary: the transport frames one tensor per microbatch.",
            )
        shapes.append(tuple(x.shape))
    return (x, shapes), _ok(
        "single_tensor_boundaries",
        f"logits {shapes[-1]}, boundary width {shapes[-2][-1] if len(shapes) > 1 else '?'}",
    )


def check_determinism(source, cfg_dict, seed=1234):
    """Same seed twice must give bit-identical outputs and gradients.

    This is the check that protects the exactness claim: dropout, unseeded
    initialisation and any forward-time use of the global RNG all pass a
    loss-goes-down smoke test while making the run unreproducible.
    """
    outs = []
    grads = []
    for _ in range(2):
        torch.manual_seed(seed)
        build = load_build(source)
        cfg = Cfg(cfg_dict)
        layers = build(cfg)
        if isinstance(layers, (nn.Sequential, nn.ModuleList)):
            layers = list(layers)
        torch.manual_seed(seed + 1)
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
        tgt = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
        x = idx
        for layer in layers:
            x = layer(x)
        loss = nn.functional.cross_entropy(x.reshape(-1, x.size(-1)).float(), tgt.reshape(-1))
        loss.backward()
        outs.append(loss.detach().clone())
        grads.append(
            [p.grad.detach().clone() for l in layers for p in l.parameters() if p.grad is not None]
        )

    if not torch.equal(outs[0], outs[1]):
        return _fail(
            "deterministic",
            f"two identical runs gave different losses "
            f"({outs[0].item():.10f} vs {outs[1].item():.10f})",
            "Remove dropout and any unseeded randomness from the forward pass. The "
            "engine's exactness guarantee is that a distributed run reproduces the "
            "single-device baseline bit for bit; a non-deterministic layer voids it, "
            "and nothing in the loss curve will tell you.",
        )
    if len(grads[0]) != len(grads[1]) or not all(
        torch.equal(a, b) for a, b in zip(grads[0], grads[1])
    ):
        return _fail(
            "deterministic",
            "losses matched but gradients differed between identical runs",
            "Something in the backward pass is non-deterministic. Check for "
            "scatter/index_add on duplicate indices, or atomics in a custom kernel.",
        )
    return _ok("deterministic", "loss and all gradients bit-identical across two runs")


def check_stateless_forward(layers, cfg):
    """The forward pass must not consume the global RNG.

    Seed-reproducibility is not enough: dropout reseeded identically draws the
    same masks, so a run-twice-from-one-seed check passes it. But the engine
    recomputes activations during backward at a different RNG state, so a
    dropout layer would draw a different mask than the gradients assume:
    silently wrong gradients under a loss curve that still falls. The test is
    whether two consecutive forwards, with no reseeding, agree.
    """
    # Deliberately left in train() mode: a layer that is only deterministic under
    # eval() is still unusable, because training runs in train mode.
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))

    def once():
        x = idx
        with torch.no_grad():
            for layer in layers:
                x = layer(x)
        return x

    a = once()
    b = once()
    if not torch.equal(a, b):
        diff = (a - b).abs().max().item()
        return _fail(
            "stateless_forward",
            f"two consecutive forwards on identical input disagree (max diff "
            f"{diff:.3e}), so the forward pass consumes random state",
            "Remove dropout and any other RNG use from the forward pass. The engine "
            "recomputes activations during backward to save memory, and that second "
            "forward runs at a different RNG state — so a stochastic layer produces "
            "a different mask than the gradients assume. The loss still falls; the "
            "gradients are just wrong. If you need regularisation, use weight decay "
            "or a seed derived from (step, microbatch) inside the layer.",
        )
    return _ok("stateless_forward", "forward is a pure function of its input")


def check_dead_layers(layers, cfg):
    """Every layer must receive gradient. A boundary cast that underflows to
    zero leaves the upstream layers with exactly zero gradient while
    cross-entropy still falls toward the unigram entropy and looks healthy."""
    for l in layers:
        for p in l.parameters():
            p.grad = None
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    x = idx
    for layer in layers:
        x = layer(x)
    nn.functional.cross_entropy(x.reshape(-1, x.size(-1)).float(), tgt.reshape(-1)).backward()

    dead = []
    for i, l in enumerate(layers):
        tot = sum(p.grad.abs().sum().item() for p in l.parameters() if p.grad is not None)
        if any(True for _ in l.parameters()) and tot == 0.0:
            dead.append(i)
    if dead:
        return _fail(
            "all_layers_train",
            f"layer(s) {dead} receive exactly zero gradient, so they never train",
            "A zero gradient upstream usually means a boundary cast underflowed "
            "(fp8 without a scale factor zeroes activation gradients outright) or a "
            "layer is detached from the graph. The loss will still fall — only the "
            "output head is learning.",
        )
    return _ok("all_layers_train", f"all {len(layers)} layers receive gradient")


def measure(layers, target_cfg):
    """Geometry of the tiny build, plus the boundary width the user's real
    config implies. Parameter counts at the real size come from the planner,
    which builds the actual slice."""
    tiny_params = sum(p.numel() for l in layers for p in l.parameters())
    return dict(
        tiny_params=tiny_params,
        n_layers=len(layers),
        boundary_width=int(target_cfg.get("d_model", 0)),
        scale=1.0,
    )


def validate(source: str, target: dict) -> dict:
    """Run every check. Returns a report; never raises for user-code errors."""
    results = []
    cfg_dict = dict(TINY)
    # keep the user's head/width ratios sane at tiny scale
    if target.get("n_head"):
        cfg_dict["n_head"] = min(target["n_head"], cfg_dict["d_model"])
        while cfg_dict["d_model"] % cfg_dict["n_head"]:
            cfg_dict["n_head"] -= 1

    try:
        build = load_build(source)
    except ValidationError as e:
        return dict(ok=False, results=[_fail("has_build_function", str(e))], geometry=None)
    except Exception:
        return dict(
            ok=False,
            results=[
                _fail(
                    "source_executes",
                    "the module raised while being imported:\n" + traceback.format_exc(limit=4),
                )
            ],
            geometry=None,
        )

    try:
        torch.manual_seed(1234)
        layers = build(Cfg(cfg_dict))
    except Exception:
        return dict(
            ok=False,
            results=[_fail("build_runs", "build(cfg) raised:\n" + traceback.format_exc(limit=4))],
            geometry=None,
        )
    results.append(_ok("build_runs", "build(cfg) returned successfully"))

    layers, r = check_flat_list(layers)
    results.append(r)
    if layers is None:
        return dict(ok=False, results=results, geometry=None)

    cfg = Cfg(cfg_dict)
    fwd, r = check_forward(layers, cfg)
    results.append(r)
    if fwd is None:
        return dict(ok=False, results=results, geometry=None)

    results.append(check_stateless_forward(layers, cfg))
    results.append(check_dead_layers(layers, cfg))
    results.append(check_determinism(source, cfg_dict))

    geometry = measure(layers, target)
    return dict(ok=all(r["ok"] for r in results), results=results, geometry=geometry)


def main():
    payload = json.loads(sys.stdin.read())
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        report = validate(payload["source"], payload.get("target", {}))
    except Exception:
        report = dict(
            ok=False,
            results=[_fail("validator", "validator crashed:\n" + traceback.format_exc(limit=6))],
            geometry=None,
        )
    sys.stdout.write(json.dumps(report))


if __name__ == "__main__":
    main()
