"""Fixed-order incremental gradient accumulation.

The exactness contract requires the step gradient to be the iterative sum
g_0 + g_1 + ... + g_{M-1} in microbatch-index order; a pairwise tree would
change the bits. OrderedAccumulator folds each contribution into the running
sum as soon as its prefix is complete, holding an out-of-order arrival only
until the microbatches before it have been folded (per tensor, RUNTIME-I21). The op sequence (clone of
g_0, then add_ in index order) is bit-identical to summing at the step
barrier, and memory is O(1) plus the hold window instead of O(M).
"""

import torch


class OrderedAccumulator:
    """The single entry point for the step gradient, and so where an open
    fleet is defended. A contribution from an untrusted peer is quarantined:
    it keeps its place in the fold order but blocks the fold at its index
    until an auditor's recompute releases it (digest matched) or replaces it
    (digest differed; the recompute is the honest tensor, so poisoned bytes
    never touch the running sum). Quarantine delays when a fold happens,
    never the order, so the final sum is bit-identical to the all-honest run.
    The price is memory: the hold window stretches by the audit latency.

    Each tensor index folds on its own (RUNTIME-I21): tensor i's running sum is
    g_0[i].clone(), then add_ of g_1[i], g_2[i], ... in fold order, whatever the
    other indices do, so the sums are the whole-set fold's bit for bit. A part
    handed in a narrower dtype than its sum (fold_part's `dtype`, RUNTIME-I28)
    is widened by the fold itself: the first one by an exact copy into the
    sum's dtype, every later one inside add_, which widens its operand exactly
    before the same add. A slot
    may therefore hand its gradients in one at a time as they are produced
    (fold_part), B's at B and W's at W, and only a tensor whose predecessor in
    fold order has not arrived is held.
    """

    def __init__(self, order):
        self.order = list(order)  # microbatch ids in the exact fold order
        self.acc = None  # running sums, one per tensor index
        self.next = None  # per tensor index: position in `order` of its next fold
        self.parts = {}  # (mb, i) -> tensor waiting for its prefix
        self.quarantine = set()  # mb ids held pending an auditor's verdict
        self.closed = set()  # microbatches whose every tensor has arrived
        self.added = 0
        self.widen: dict = {}  # tensor index -> the dtype its running sum is kept in (fold_part's dtype)

    @property
    def pos(self) -> int:
        """How many microbatches of `order` every tensor has folded: the
        whole-set fold's position."""
        return min(self.next) if self.next else 0

    def _size(self, n):
        if self.acc is None:
            self.acc = [None] * n
            self.next = [0] * n
        assert len(self.acc) == n, f"{n} tensors where the fold holds {len(self.acc)}"

    def add(self, mb, tensors, verified: bool = True):
        assert mb not in self.closed, f"duplicate contribution for mb {mb}"
        self._size(len(tensors))
        if not verified:
            self.quarantine.add(mb)
        for i, t in enumerate(tensors):
            assert (mb, i) not in self.parts, f"duplicate contribution for mb {mb}"
            self.parts[(mb, i)] = t
        self.closed.add(mb)
        self.added += 1
        self._fold_all()

    def fold_part(self, mb, i: int, n: int, g, dtype=None) -> None:
        """Tensor i of the n of microbatch mb (RUNTIME-I21): folded now when
        every earlier microbatch's tensor i is in, held (that tensor alone)
        otherwise. end_part(mb) closes the microbatch once all n arrived.
        `dtype`: the running sum's dtype when g may come narrower (RUNTIME-I28);
        a held part then stays narrow until its fold."""
        assert mb not in self.closed and (mb, i) not in self.parts, (
            f"duplicate contribution for mb {mb} tensor {i}"
        )
        self._size(n)
        if dtype is not None:
            self.widen[i] = dtype
        self.parts[(mb, i)] = g
        self._fold(i)

    def end_part(self, mb) -> None:
        """Every tensor of microbatch mb was handed to fold_part."""
        assert mb not in self.closed, f"duplicate contribution for mb {mb}"
        self._size(0 if self.acc is None else len(self.acc))  # a slot with no trainable tensor
        self.closed.add(mb)
        self.added += 1

    def _fold(self, i):
        while self.next[i] < len(self.order):
            nxt = self.order[self.next[i]]
            if nxt in self.quarantine or (nxt, i) not in self.parts:
                return
            t = self.parts.pop((nxt, i))
            if self.next[i] == 0:
                want = self.widen.get(i)
                self.acc[i] = t.to(want, copy=True) if want is not None else t.clone()
            else:
                self.acc[i].add_(t)
            self.next[i] += 1

    def _fold_all(self):
        for i in range(len(self.acc)):
            self._fold(i)

    def release(self, mb):
        """Auditor verdict: digest matched — the held tensors are honest."""
        assert mb in self.quarantine, f"mb {mb} is not quarantined"
        self.quarantine.discard(mb)
        self._fold_all()

    def reject(self, mb, replacement):
        """Auditor verdict: digest differed. The auditor's recompute replaces
        the poisoned tensors; the sum proceeds as if the fraud never arrived."""
        assert mb in self.quarantine, f"mb {mb} is not quarantined"
        for i, t in enumerate(replacement):
            self.parts[(mb, i)] = t
        self.quarantine.discard(mb)
        self._fold_all()

    def complete(self) -> bool:
        return (
            len(self.closed) == len(self.order)
            and self.acc is not None
            and all(k == len(self.order) for k in self.next)
        )

    def result(self):
        assert self.complete(), (
            f"accumulator incomplete: {len(self.closed)}/{len(self.order)} microbatches in, "
            f"{len(self.parts)} tensors held ({len(self.quarantine)} in quarantine)"
        )
        return self.acc


class TiedRowFold:
    """The step gradient of a tied table split across stages (RUNTIME-I19,
    HANDOUT-M4 I-6), folded on the Head in the stock order without one dense
    table per microbatch in flight.

    The stock fold is acc = g_0, acc += g_i in microbatch-index order, with
    g_i = h_i + e_i (autograd's sum of the table's two uses: h_i the Head's
    dense gradient, e_i the embedding's, +0 outside the microbatch's ids T_i).
    Outside T_i, g_i = h_i + 0.0, so a row r takes h_i as soon as every earlier
    microbatch's contribution to r is folded: when h_i arrives (in index
    order), every row outside P | T_i (P: the rows held for microbatches whose
    e has not been folded yet) folds as acc + (h_i + 0.0) (the assignment
    h_i + 0.0 at i = 0); the rest of h_i is held. When e_j's rows arrive the
    oldest held microbatches resolve in index order, acc + (h_j + e_j) on their
    held rows (e_j is +0 outside T_j). Every row then sees exactly the stock
    sequence of adds, so the result is the stock fold bit for bit.

    add_head(mb, h, rows): h the (n_rows, d) gradient of the Head's use, rows
    T_mb (ascending unique ids, int64). add_embed(mb, e_rows): e_mb[T_mb], in
    ascending id. Either may come first; out-of-order microbatches wait."""

    def __init__(self, n_rows: int, order) -> None:
        self.n_rows = int(n_rows)
        self.order = list(order)
        self.pos = 0  # next position of `order` whose h is processed
        self.acc: torch.Tensor = torch.empty(0)  # allocated by the first Head gradient
        self.first = True  # no row folded yet: microbatch order[0] assigns
        self.heads: dict = {}  # mb -> (h, rows) waiting for its prefix
        self.embeds: dict = {}  # mb -> e_rows, until its held rows resolve
        self.held: list = []  # [(mb, held row ids, h on them, T_mb)], oldest first
        self.pend = torch.zeros(self.n_rows, dtype=torch.int32)  # held-row multiplicity

    def add_head(self, mb: int, h: torch.Tensor, rows: torch.Tensor) -> None:
        if mb in self.heads or self._done(mb):
            raise RuntimeError(f"tied fold: a second Head gradient for microbatch {mb}")
        if h.dim() != 2 or h.shape[0] != self.n_rows:
            raise RuntimeError(f"tied fold: Head gradient of mb {mb} has shape {tuple(h.shape)}")
        self.heads[mb] = (h, rows.to(torch.int64).cpu())
        self._advance()

    def add_embed(self, mb: int, e_rows: torch.Tensor) -> None:
        if mb in self.embeds or (self._done(mb) and not any(m == mb for m, *_ in self.held)):
            raise RuntimeError(f"tied fold: a second embedding gradient for microbatch {mb}")
        self.embeds[mb] = e_rows
        self._advance()

    def _done(self, mb) -> bool:
        return mb in self.order[: self.pos]

    def _advance(self) -> None:
        while True:
            if self.held and self.held[0][0] in self.embeds:
                self._resolve()
                continue
            if self.pos < len(self.order) and self.order[self.pos] in self.heads:
                self._take_head()
                continue
            return

    def _take_head(self) -> None:
        mb = self.order[self.pos]
        h, rows = self.heads.pop(mb)
        if self.first:
            self.acc = torch.empty_like(h)
        mask = self.pend > 0
        mask[rows] = True
        keep = mask.nonzero().reshape(-1)
        free = (~mask).nonzero().reshape(-1)
        dev = h.device
        fi, ki = free.to(dev), keep.to(dev)
        if self.first:
            self.acc[fi] = h[fi] + 0.0
        else:
            self.acc[fi] = self.acc[fi] + (h[fi] + 0.0)
        self.held.append((mb, keep, h[ki].clone(), rows, self.first))
        self.pend[keep] += 1
        self.first = False
        self.pos += 1

    def _resolve(self) -> None:
        mb, keep, hk, rows, first = self.held.pop(0)
        e_rows = self.embeds.pop(mb)
        if e_rows.dim() != 2 or e_rows.shape[0] != rows.numel() or e_rows.shape[1:] != hk.shape[1:]:
            raise RuntimeError(
                f"tied fold: embedding rows of mb {mb} have shape {tuple(e_rows.shape)}, "
                f"want ({rows.numel()}, {tuple(hk.shape[1:])})"
            )
        e_on = torch.zeros_like(hk)
        if rows.numel():
            at = torch.searchsorted(keep, rows).to(hk.device)
            e_on[at] = e_rows.to(hk.device, hk.dtype)
        ki = keep.to(hk.device)
        if first:
            self.acc[ki] = hk + e_on
        else:
            self.acc[ki] = self.acc[ki] + (hk + e_on)
        self.pend[keep] -= 1

    def complete(self) -> bool:
        return self.pos == len(self.order) and not self.held

    def result(self) -> torch.Tensor:
        if not self.complete():
            raise RuntimeError(
                f"tied fold incomplete: {self.pos}/{len(self.order)} Head gradients, "
                f"{len(self.held)} microbatch(es) waiting for embedding rows"
            )
        return self.acc


class CountFold:
    """The int64 selection counts of one step at a bias's owner (RUNTIME-I18):
    one vector per (microbatch, MoE layer), validated at arrival, folded by
    integer add in microbatch order. A vector that is not int64, not (E,), has
    a negative count, or (when `total` is given, zero1's whole vectors) does not
    sum to T_mb*k, a (mb, layer) given twice, and at result() any (mb, layer)
    missing are fatal, naming (step, mb, layer, peer)."""

    def __init__(self, step: int, n_mb: int, layers, experts: dict, total: dict | None) -> None:
        self.step = step
        self.n_mb = n_mb
        self.layers = list(layers)
        self.experts = dict(experts)  # layer -> E
        self.total = total  # layer -> T_mb*k, or None (piece lanes)
        self.got: dict = {}  # (mb, layer) -> vector

    def _name(self, mb, layer, peer) -> str:
        return f"(step {self.step}, mb {mb}, layer {layer}, peer {peer})"

    def add(self, mb, layer, vec, peer) -> None:
        name = self._name(mb, layer, peer)
        if type(mb) is not int or not 0 <= mb < self.n_mb:
            raise RuntimeError(f"MOE-COUNT {name}: microbatch outside [0, {self.n_mb})")
        if layer not in self.experts:
            raise RuntimeError(f"MOE-COUNT {name}: no selection bias of that layer here")
        if (mb, layer) in self.got:
            raise RuntimeError(f"MOE-COUNT {name}: count vector given twice")
        if not isinstance(vec, torch.Tensor) or vec.dtype != torch.int64:
            raise RuntimeError(f"MOE-COUNT {name}: not an int64 vector")
        if tuple(vec.shape) != (self.experts[layer],):
            raise RuntimeError(
                f"MOE-COUNT {name}: shape {tuple(vec.shape)}, want ({self.experts[layer]},)"
            )
        if bool((vec < 0).any()):
            raise RuntimeError(f"MOE-COUNT {name}: a negative count")
        if self.total is not None and int(vec.sum()) != self.total[layer]:
            raise RuntimeError(
                f"MOE-COUNT {name}: counts sum to {int(vec.sum())}, want T_mb*k = "
                f"{self.total[layer]}"
            )
        self.got[(mb, layer)] = vec.detach().to("cpu", copy=True)

    def missing(self) -> list:
        return [
            (mb, layer)
            for layer in self.layers
            for mb in range(self.n_mb)
            if (mb, layer) not in self.got
        ]

    def complete(self) -> bool:
        return not self.missing()

    def result(self) -> dict:
        """{layer: int64 (E,)}: the integer sum over microbatches in index order."""
        gone = self.missing()
        if gone:
            mb, layer = gone[0]
            raise RuntimeError(
                f"MOE-COUNT {self._name(mb, layer, '-')}: count vector missing at the "
                f"owner ({len(gone)} missing)"
            )
        out = {}
        for layer in self.layers:
            acc = self.got[(0, layer)].clone()
            for mb in range(1, self.n_mb):
                acc += self.got[(mb, layer)]
            out[layer] = acc
        return out
