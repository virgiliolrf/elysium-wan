"""Deferred weight gradients: the zero-bubble B/W split done properly.

Splitting the backward with two autograd calls (dL/dX, then dL/dW from the
same graph) is correct but expensive: the second call walks the whole
activation-gradient chain again, so "W" costs a full backward. Zero-bubble
implementations instead change the linear layers (sail-sg's WeightGradStore):
during B each linear computes only its input gradient and records the two
tensors its weight gradient needs (its input X and the incoming dY); W later
runs just the weight-gradient GEMMs dW = dY^T X. B then costs the activation
chain and W only the weight GEMMs, the cost structure the paper schedules
(T_B + T_W ~ 2 T_F), and the graph is freed after B: only X and dY stay
pinned until W, which is the paper's M_W instead of a whole graph.

Exactness: every path computes dW with the same function on the same saved
tensors, whether it runs inside backward (no capture active: the reference
trainer, a monolithic backward, the first stage) or later from the store (a
deferred W). A distributed run therefore stays bit-identical to the
single-device reference, which builds the same layers.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from wantrain.model import precision

# Process-wide, not thread-local: on CUDA and MPS the autograd engine runs
# backward on a device thread, which would not see a thread-local store and
# would silently return dW to an autograd call that never asked for it. A
# stage computes on one thread; the lock only guards nesting bookkeeping.
_state: dict = {"store": None}
_lock = threading.RLock()


def _active():
    return _state["store"]


@contextlib.contextmanager
def capture():
    """Within this block, linear layers defer their weight gradients into the
    yielded list instead of returning them to autograd."""
    with _lock:
        prev = _state["store"]
        store: list = []
        _state["store"] = store
    try:
        yield store
    finally:
        with _lock:
            _state["store"] = prev


def flush(store: list) -> dict:
    """Run the deferred weight-gradient GEMMs. -> {id(param): grad}, summing
    contributions of a parameter used more than once in backward order."""
    return {id(p): g for p, g in flush_each(store)}


def flush_each(store: list, widen: bool = True) -> Iterator[tuple[nn.Parameter, torch.Tensor]]:
    """`flush`, yielding (param, grad) as soon as a parameter's last
    contribution is summed (RUNTIME-I21): the same GEMMs in the same store
    order, each widened to the parameter dtype and summed in that order, so
    every yielded gradient is flush's bit for bit. Each entry's X and dY are
    released once its GEMMs ran, so a caller that folds and drops each yielded
    gradient keeps one entry's outputs and the incomplete sums resident, never
    the stage's whole gradient set.

    widen=False (RUNTIME-I28) yields a parameter with one contribution in its
    GEMM's output dtype, for a caller that widens it exactly as it adds it to
    a running sum in the parameter dtype (a bf16 to fp32 widening is exact, so
    the add sees the same operands): the widened copy, 6 bytes per parameter
    of memory traffic per microbatch, is never made. A parameter with several
    contributions is still summed in the parameter dtype."""
    left: dict = {}
    for module, _fn in store:
        for p in _deferred_params(module):
            left[id(p)] = left.get(id(p), 0) + 1
    entries = list(store)
    store.clear()
    out: dict = {}
    for i in range(len(entries)):
        module, fn = entries[i]
        entries[i] = None
        dw, db = fn()
        del fn  # the entry's X and dY
        for p, g in ((module.weight, dw), (module.bias, db)):
            if g is None or p is None or not p.requires_grad:
                continue
            k = id(p)
            if not widen and k not in out and left[k] == 1:
                left[k] = 0
                yield p, g
                continue
            g = g.to(p.dtype)
            out[k] = g if k not in out else out[k] + g
            left[k] -= 1
            if not left[k]:
                yield p, out.pop(k)
        del dw, db


FLUSH_ORDERS = ("store", "forward")


def _deferred_params(module):
    """The parameters `flush` takes a gradient for from one store entry of
    `module`, weight before bias (db is None exactly when the bias is)."""
    return [p for p in (module.weight, module.bias) if p is not None and p.requires_grad]


def flush_iter(store: list, order: str) -> Iterator[tuple[nn.Parameter, torch.Tensor]]:
    """Run the deferred weight-gradient GEMMs one store entry at a time and
    yield (param, grad) as soon as a parameter's last contribution is computed
    (MODEL-I7). Entries are grouped by module, each module's entries run
    consecutively in store order; `order` "store" keeps the modules in the
    order they first appear in the store (backward order), "forward" runs them
    reversed (forward order). A single-contribution gradient is yielded in its
    GEMM's output dtype; a multi-contribution one is widened to the parameter
    dtype and summed in store order, exactly as `flush`, so widening any
    yielded gradient to the parameter dtype gives flush's bits. The store is
    taken over and cleared; the caller drops each yielded gradient to keep at
    most one entry's GEMM outputs plus the incomplete sums resident."""
    if order not in FLUSH_ORDERS:
        raise ValueError(f"flush order {order!r} is not one of {FLUSH_ORDERS}")
    groups: dict = {}
    owner: dict = {}
    count: dict = {}
    for module, fn in store:
        groups.setdefault(id(module), []).append((module, fn))
        for p in _deferred_params(module):
            k = id(p)
            if owner.setdefault(k, id(module)) != id(module):
                # Grouping by module would reorder this parameter's sum.
                raise ValueError("a deferred parameter belongs to two modules; use flush")
            count[k] = count.get(k, 0) + 1
    seq = list(groups.values())
    if order == "forward":
        seq.reverse()
    store.clear()
    return _flush_entries(seq, count)


def _flush_entries(groups: list, count: dict) -> Iterator[tuple[nn.Parameter, torch.Tensor]]:
    sums: dict = {}
    left = dict(count)
    for entries in groups:
        for i in range(len(entries)):
            module, fn = entries[i]
            entries[i] = None
            ready = _flush_entry(module, fn, count, left, sums)
            del module, fn  # release the entry's X and dY before yielding
            while ready:
                yield ready.pop(0)


def _flush_entry(module, fn, count: dict, left: dict, sums: dict) -> list:
    """One entry's GEMMs -> the (param, grad) pairs it completes. Locals die
    on return, so no GEMM output outlives the entry unless it is a sum."""
    dw, db = fn()
    ready = []
    for p, g in ((module.weight, dw), (module.bias, db)):
        if g is None or p is None or not p.requires_grad:
            continue
        k = id(p)
        if count[k] == 1:
            ready.append((p, g))
            continue
        g = g.to(p.dtype)
        sums[k] = g if k not in sums else sums[k] + g
        left[k] -= 1
        if left[k] == 0:
            ready.append((p, sums.pop(k)))
    return ready


def gather_dtypes(module: nn.Module, compute_dtype: str, autocast: bool) -> dict[int, torch.dtype]:
    """{id(param): dtype a sharded stage gathers it in}: the recipe's compute
    dtype for parameters the store owns (SplitLinear casts them to it before
    any use) when the recipe autocasts, the parameter's own master dtype
    otherwise (LayerNorm, embeddings, tied weights, fp32 recipes)."""
    dt = precision.spec(compute_dtype)["torch_dtype"]
    owned = owned_params(module) if autocast else set()
    return {id(p): dt if id(p) in owned else p.dtype for p in module.parameters()}


@torch.no_grad()
def _weight_grads(x, g, has_bias):
    # Outside backward (a deferred W, flush) grad mode is on and the saved input still requires grad: without
    # no_grad the GEMM records a graph that keeps dY alive through the folded gradient, every microbatch's dY until
    # the step ends (~113 MB per d=6144 layer per microbatch; 2026-10-12, RUNTIME-I21). Inside backward it is a no-op.
    g2 = g.reshape(-1, g.shape[-1])
    x2 = x.reshape(-1, x.shape[-1])
    dw = g2.t().mm(x2)
    db = g2.sum(0) if has_bias else None
    return dw, db


class _SplitLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, module, param):
        # `weight` is what the GEMM reads: the parameter, or its compute-dtype cast made outside. Backward saves the
        # parameter and casts it again: saving the cast kept a copy of every weight of the stage alive through a
        # recomputed backward, 2 bytes per parameter, ~13 GiB on a 6.8B-parameter H100 stage (2026-10-12). The cast
        # is deterministic, and the copy keeps the cast's requires_grad, which picks torch.matmul's folding (a
        # non-contiguous dY folds only when the weight requires grad): backward reads the same bits through the
        # same kernel.
        ctx.save_for_backward(x, param)
        ctx.module = module
        ctx.has_bias = bias is not None
        ctx.cast = None if weight is param else (weight.dtype, weight.requires_grad)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g):
        x, param = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            weight = param
            if ctx.cast is not None:
                weight = param.to(ctx.cast[0])
                if ctx.cast[1]:
                    weight.requires_grad_()
            gx = g.matmul(weight)
            del weight
        else:
            gx = None
        mod = ctx.module
        wants_w = ctx.needs_input_grad[1] or (ctx.has_bias and ctx.needs_input_grad[2])
        if not wants_w:
            return gx, None, None, None, None
        store = _active()
        has_bias = ctx.has_bias
        if store is not None:
            store.append((mod, lambda x=x, g=g: _weight_grads(x, g, has_bias)))
            return gx, None, None, None, None
        dw, db = _weight_grads(x, g, has_bias)
        return (
            gx,
            dw if ctx.needs_input_grad[1] else None,
            db if (has_bias and ctx.needs_input_grad[2]) else None,
            None,
            None,
        )


class SplitLinear(nn.Linear):
    """nn.Linear whose backward can defer dW (see module docstring). Same
    parameters, same forward kernel; autocast handled outside the Function so
    the deferred and immediate paths cast identically."""

    def forward(self, x):
        w, b = self.weight, self.bias
        dev = x.device.type
        if torch.is_autocast_enabled(dev):
            dt = torch.get_autocast_dtype(dev)
            x, w = x.to(dt), w.to(dt)
            b = b.to(dt) if b is not None else None
        with torch.autocast(device_type=dev, enabled=False):
            return _SplitLinear.apply(x, w, b, self, self.weight)


def convert(module: nn.Module, exclude: frozenset = frozenset()) -> int:
    """Swap every nn.Linear under `module` for a SplitLinear holding the same
    parameter tensors. Left alone, their dW computed in B by autograd:
    - Linears whose weight is shared with another module (tied embeddings):
      their gradient is a sum autograd must own;
    - Linears in `exclude` (ADR-0009 D4: the unit trace's Linears whose weight
      or bias an op outside their own forward reads; a converted one would
      lose that use's gradient);
    - nn.Linear subclasses, the MoE router among them (exact type only).
    Returns how many layers were converted."""
    seen: dict = {}
    for m in module.modules():
        for p in m.parameters(recurse=False):
            seen[id(p)] = seen.get(id(p), 0) + 1
    n = 0
    for m in module.modules():
        for child in list(m.children()):
            if (
                type(child) is nn.Linear
                and seen.get(id(child.weight), 0) == 1
                and child not in exclude
            ):
                child.__class__ = SplitLinear
                n += 1
    return n


def owned_params(module: nn.Module) -> set:
    """ids of parameters whose gradient comes from the store (SplitLinear)."""
    out = set()
    for m in module.modules():
        if isinstance(m, SplitLinear):
            out.add(id(m.weight))
            if m.bias is not None:
                out.add(id(m.bias))
    return out
