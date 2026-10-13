"""Pure piece kernels for sharded stages (`shard: "pieces"`, ADR-0006).

Every trainable tensor of a width-w stage is cut along dim 0 into w pieces;
peer k owns piece k of every tensor (fp32 master, AdamW m and v, fp32 fold
accumulator). The kernels here are what makes that exact (RUNTIME-I11): each
piece-local operation equals the matching slice of the whole-tensor reference
bit for bit.

- Boundaries are multiples of Q_t = lcm(row length, 128) elements, so a piece
  is whole rows and an elementwise kernel's vector-or-tail split falls on the
  same elements as in the whole tensor.
- The fold is the reference's left fold restricted to a piece: an fp32 copy of
  microbatch 0's widened piece, then `add_` in microbatch-index order. Never a
  clone in the wire dtype (it would round every add) and never zeros
  (+0 + -0 = +0 loses the sign the reference keeps).
- AdamW is the reference's constructor over the owned pieces; a per-unit
  update is `step()` with only that unit's gradients set, so every other
  parameter's state and step counter are untouched.

A piece is addressed as a flat element range [lo, hi) of the tensor; with
boundaries on row multiples that is the dim-0 slice t[lo // row : hi // row].
No kernel draws from the torch RNG.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from wantrain.model import wgrad

ALIGN = 128  # elements; the vector-or-tail alignment every piece boundary keeps


def row_length(shape: Sequence[int]) -> int:
    """Elements per dim-0 row: numel / shape[0], and 1 for 0-D and 1-D."""
    return max(1, math.prod(shape[1:])) if len(shape) > 1 else 1


def quantum(shape: Sequence[int]) -> int:
    """Q_t = lcm(row length, 128): the granularity of every piece boundary."""
    return math.lcm(row_length(shape), ALIGN)


def _equal_bounds(numel: int, q: int, w: int) -> tuple[int, ...]:
    # ceil: the last quantum may be partial; extra quanta go to the first pieces
    n_q = -(-numel // q)
    base, extra = divmod(n_q, w)
    return tuple(min(numel, q * (k * base + min(k, extra))) for k in range(w + 1))


def _check_bounds(name: str, b: Sequence[int], numel: int, q: int, w: int) -> tuple[int, ...]:
    b = tuple(b)
    if len(b) != w + 1 or any(type(x) is not int for x in b):
        raise ValueError(f"piece bounds of {name!r}: need {w + 1} integers, got {b!r}")
    if b[0] != 0 or b[-1] != numel:
        raise ValueError(f"piece bounds of {name!r} do not tile [0, {numel}): {b!r}")
    if any(lo > hi for lo, hi in zip(b, b[1:])):
        raise ValueError(f"piece bounds of {name!r} are not monotone: {b!r}")
    bad = [x for x in b[1:-1] if x % q and x != numel]
    if bad:
        raise ValueError(
            f"piece bounds of {name!r} are not multiples of its quantum {q} "
            f"(lcm(row length, {ALIGN})): {bad!r}"
        )
    return b


def piece_layout(
    shapes: Mapping[str, Sequence[int]],
    w: int,
    bounds: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, tuple[int, ...]]:
    """-> {name: (b_0 = 0, b_1, ..., b_w = numel)}; piece k is [b_k, b_k+1).

    By default the quanta are dealt out equally (piece sizes differ by at most
    one quantum) and a tensor with fewer quanta than peers leaves the last
    pieces empty. `bounds` overrides the split per tensor; it must start at 0,
    end at numel, be monotone and put every inner boundary on a quantum (or at
    numel), otherwise ValueError names the tensor.
    """
    if type(w) is not int or w < 1:
        raise ValueError(f"piece width must be an integer >= 1, got {w!r}")
    bounds = dict(bounds or {})
    unknown = sorted(set(bounds) - set(shapes))
    if unknown:
        raise ValueError(f"piece bounds name tensors outside the layout: {unknown!r}")
    out = {}
    for name, shape in shapes.items():
        numel, q = math.prod(shape), quantum(shape)
        if name in bounds:
            out[name] = _check_bounds(name, bounds[name], numel, q, w)
        else:
            out[name] = _equal_bounds(numel, q, w)
    return out


def piece(t: torch.Tensor, b: Sequence[int], k: int) -> torch.Tensor:
    """Piece k of `t` under boundaries `b`, as a flat view (no copy)."""
    return t.reshape(-1)[b[k] : b[k + 1]]


def exchange_units(chunk: Iterable[nn.Module]) -> list[list[tuple[str, nn.Parameter]]]:
    """A stage chunk's trainable parameters grouped into exchange units, in
    forward order. Per layer: its tensors outside split linears (the Embed's
    tables `tok` and `pos`, a block's norms) are one unit, first because they
    are used first; then one unit per SplitLinear (weight and bias) in
    registration order. A Head's norm rides in its linear's unit. Names are
    the chunk's `named_parameters` names; a parameter shared between layers
    belongs to the unit that meets it first.

    MoE layers (ADR-0008, M4; a module carrying `moe_layer`, BA-456): the
    split linears of one projection over its E routed experts (the `experts`
    child) form one unit, in expert index order, at the place of its first
    expert; every 3-D parameter (an HF expert bank: gate_up_proj, down_proj)
    is a unit of its own, after the small tensors; the selection bias
    (`sel_bias`, a buffer) joins the small-tensor unit after its parameters
    (its count lane rides that unit's GRADS frame, BA-455). A dense model's
    units are unchanged.
    """
    from wantrain.model import arch, gpt

    heads = (gpt.Head, arch.Head)
    seen: set[int] = set()
    units: list[list[tuple[str, nn.Parameter]]] = []
    for li, layer in enumerate(chunk):
        names = {id(p): f"{li}.{n}" for n, p in layer.named_parameters(remove_duplicate=False)}

        def take(params, names=names):
            got = []
            for p in params:
                if p is not None and p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    got.append((names[id(p)], p))
            return got

        linears = [m for m in layer.modules() if isinstance(m, wgrad.SplitLinear)]
        in_linears = {id(p) for m in linears for p in (m.weight, m.bias) if p is not None}
        groups = _expert_groups(layer)
        banks = [p for p in layer.parameters() if p.dim() == 3 and id(p) not in in_linears]
        in_banks = {id(p) for p in banks}
        rest = take(p for p in layer.parameters() if id(p) not in in_linears | in_banks)
        rest += _sel_biases(layer, li, seen)
        lin_units = []
        for m in linears:
            group = groups.get(id(m))
            if group is None:
                lin_units.append(take((m.weight, m.bias)))
            elif group[0] is m:
                lin_units.append(take(p for g in group for p in (g.weight, g.bias)))
        bank_units = [take([p]) for p in banks]
        if isinstance(layer, heads) and lin_units:
            lin_units[-1] = rest + lin_units[-1]
        else:
            lin_units = [rest, *bank_units, *lin_units]
        units += [u for u in lin_units if u]
    return units


def _expert_groups(layer: nn.Module) -> dict[int, list]:
    """{id(split linear): its projection's group}: under each MoE module of the
    layer (one carrying `moe_layer`), the split linears of its `experts` child
    grouped by their name inside an expert (the projection), in expert order."""
    out: dict[int, list] = {}
    for mod in layer.modules():
        experts = getattr(mod, "experts", None) if hasattr(mod, "moe_layer") else None
        if not isinstance(experts, nn.Module):
            continue
        by_proj: dict[str, list] = {}
        for qn, m in experts.named_modules():
            if isinstance(m, wgrad.SplitLinear):
                by_proj.setdefault(qn.split(".", 1)[-1], []).append(m)
        for group in by_proj.values():
            for m in group:
                out[id(m)] = group
    return out


def _sel_biases(layer: nn.Module, li: int, seen: set) -> list:
    """The MoE selection biases of a layer, (chunk name, buffer), once each."""
    got = []
    for n, b in layer.named_buffers():
        if (n == "sel_bias" or n.endswith(".sel_bias")) and id(b) not in seen:
            seen.add(id(b))
            got.append((f"{li}.{n}", b))
    return got


def cast_piece(p: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The owner's once-per-step cast of its master piece; always a new tensor,
    never an alias of the master."""
    return p.to(dtype, copy=True)


def fold_start(g: torch.Tensor) -> torch.Tensor:
    """The accumulator: an fp32 copy of microbatch 0's widened piece (an int64
    count lane stays int64, RUNTIME-I18)."""
    if g.dtype == torch.int64:
        return g.clone()
    return g.to(torch.float32, copy=True)


def fold_add(acc: torch.Tensor, g: torch.Tensor) -> None:
    """acc += widen(g), correctly rounded in fp32 (the reference's add); an
    int64 count lane adds exactly in int64."""
    if acc.dtype == torch.int64:
        if g.dtype != torch.int64:
            raise ValueError(f"a count lane folds int64 only, got {g.dtype}")
        acc.add_(g)
        return
    acc.add_(g.to(torch.float32))


def fold(gs: Iterable[torch.Tensor]) -> torch.Tensor:
    """Left fold of one piece's gradients given in microbatch-index order."""
    acc = None
    for g in gs:
        if acc is None:
            acc = fold_start(g)
        else:
            fold_add(acc, g)
    if acc is None:
        raise ValueError("a fold over zero microbatches")
    return acc


def adamw(params: Iterable[torch.Tensor], lr: float) -> torch.optim.AdamW:
    """The reference's optimizer (validation/reference.py) over owned pieces."""
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)


def step_unit(
    opt: torch.optim.Optimizer,
    params: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor],
) -> None:
    """One unit's update: only `params` carry gradients during the step, so
    every other parameter's state and step counter are left untouched."""
    if len(params) != len(grads):
        raise ValueError(f"{len(params)} params, {len(grads)} grads")
    ids = {id(p) for p in params}
    held = {id(p) for group in opt.param_groups for p in group["params"]}
    if not ids <= held:
        raise ValueError("step_unit: a parameter outside this optimizer")
    for group in opt.param_groups:
        for p in group["params"]:
            if id(p) not in ids:
                p.grad = None
    for p, g in zip(params, grads):
        p.grad = g
    opt.step()
    for p in params:
        p.grad = None
