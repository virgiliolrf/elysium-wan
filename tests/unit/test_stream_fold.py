"""A width-1 slot folds W's gradients as they are produced (RUNTIME-I21):
wgrad.flush_each is flush, op for op, yielding each parameter at its last
contribution, and OrderedAccumulator's streamed fold equals its whole-set add.

WOULD-FAIL-IF: flush_each summed a multi-use parameter out of store order or
before widening, held every entry until the end, or the streamed fold assigned
or added differently from `add` (first microbatch, held successors).
"""

import torch
import torch.nn as nn

from wantrain.model import wgrad
from wantrain.runtime.accumulate import OrderedAccumulator


def _multi_use(dtype):
    """lin runs three times, interleaved with o1 and o2: store order lin, o2,
    lin, o1, lin."""
    torch.manual_seed(11)
    m = nn.ModuleDict({"lin": nn.Linear(8, 8), "o1": nn.Linear(8, 8), "o2": nn.Linear(8, 8)})
    assert wgrad.convert(m) == 3
    x = torch.randn(4, 6, 8)
    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
        h = m["lin"](x)
        h = m["o1"](torch.tanh(h))
        h = m["lin"](torch.tanh(h))
        h = m["o2"](torch.tanh(h))
        h = m["lin"](torch.tanh(h))
    with wgrad.capture() as store:
        h.float().square().sum().backward()
    return m, store


def test_flush_each_is_flush_bit_for_bit_and_streams():
    # covers: RUNTIME-I21
    for dtype in (torch.float32, torch.bfloat16):
        m, store = _multi_use(dtype)
        ref = wgrad.flush(list(store))
        ran = []
        wrapped = []
        for mod, fn in store:
            name = next(k for k, v in m.items() if v is mod)
            wrapped.append((mod, lambda fn=fn, name=name, ran=ran: (ran.append(name), fn())[1]))
        seen = []
        for p, g in wgrad.flush_each(wrapped):
            assert g.dtype == p.dtype == torch.float32
            assert torch.equal(g, ref[id(p)])
            seen.append((id(p), len(ran)))
        assert wrapped == []
        assert {k for k, _ in seen} == set(ref)
        at = dict(seen)
        # o2 (entry 1) is yielded before entry 2 runs; lin only after its last use (entry 4)
        assert at[id(m["o2"].weight)] == 2 and at[id(m["lin"].weight)] == 5


def test_streamed_fold_equals_the_whole_set_add():
    # covers: RUNTIME-I21
    # Tensor 0 arrives at "B" in microbatch order, tensors 1 and 2 at "W" out of order (mb 2's W before mb 1's);
    # the per-tensor sums equal the whole-set add's bit for bit.
    torch.manual_seed(0)
    M, n = 5, 3
    grads = {mb: [torch.randn(17) * (mb + 1) for _ in range(n)] for mb in range(M)}
    ref = OrderedAccumulator(range(M))
    for mb in range(M):
        ref.add(mb, [g.clone() for g in grads[mb]])
    got = OrderedAccumulator(range(M))
    for mb in range(M):
        got.fold_part(mb, 0, n, grads[mb][0].clone())
    for mb in (0, 2, 1, 3, 4):
        for i in (2, 1):
            got.fold_part(mb, i, n, grads[mb][i].clone())
        if mb == 2:
            assert (2, 1) in got.parts and got.next[1] == 1  # held: mb 1's tensor 1 is not in yet
        got.end_part(mb)
    assert got.complete() and ref.complete() and not got.parts
    for a, b in zip(got.result(), ref.result(), strict=True):
        assert torch.equal(a, b)


def test_quarantine_still_orders_every_tensor():
    # covers: RUNTIME-I21
    torch.manual_seed(1)
    g = {mb: [torch.randn(9) for _ in range(2)] for mb in range(3)}
    ref = OrderedAccumulator(range(3))
    for mb in range(3):
        ref.add(mb, g[mb])
    a = OrderedAccumulator(range(3))
    a.add(0, g[0])
    a.add(1, [torch.full((9,), 1e9), torch.full((9,), 1e9)], verified=False)
    a.add(2, g[2])
    assert not a.complete() and a.next == [1, 1]
    a.reject(1, g[1])
    assert a.complete()
    for x, y in zip(a.result(), ref.result(), strict=True):
        assert torch.equal(x, y)


def test_a_deferred_w_keeps_no_graph_and_no_dy_alive():
    # covers: RUNTIME-I21
    # A deferred W runs with grad mode on, and the saved input still requires grad: the GEMM must not record a graph,
    # or the folded gradient keeps every microbatch's dY alive until the step ends (2026-10-12: ~25 GB on a 7-layer
    # d=6144 stage at 32 microbatches).
    import gc
    import weakref

    torch.manual_seed(3)
    m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
    assert wgrad.convert(m) == 2
    acc = {}
    refs = []
    for _ in range(3):
        x = torch.randn(4, 16, requires_grad=True)
        with wgrad.capture() as store:
            m(x).square().sum().backward()
        refs += [
            weakref.ref(v) for _mod, fn in store for v in fn.__defaults__ if torch.is_tensor(v)
        ]
        for p, g in wgrad.flush_each(store):
            assert g.grad_fn is None and not g.requires_grad
            acc[id(p)] = g if id(p) not in acc else acc[id(p)] + g
        del x
    gc.collect()
    assert all(r() is None for r in refs), "a deferred W's X or dY outlived its flush"


def test_a_split_linear_saves_its_parameter_not_a_compute_dtype_copy():
    # Under autocast a SplitLinear casts its fp32 weight to the compute dtype. Saving that cast for backward kept a
    # bf16 copy of every weight of a stage alive through a recomputed backward (2 B/param, ~13 GiB on a 6.8B-parameter
    # H100 stage, 2026-10-12). The parameter itself is saved and cast again in backward: same bits (the cast is
    # deterministic), and the immediate path's dW reaches the parameter widened as the cast's backward did.
    torch.manual_seed(5)
    ref = nn.Linear(16, 8)
    m = nn.Linear(16, 8)
    m.load_state_dict(ref.state_dict())
    assert wgrad.convert(nn.Sequential(m)) == 1
    x = torch.randn(4, 16)
    xr = x.clone().requires_grad_(True)
    xs = x.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        yr = ref(xr)
        ys = m(xs)
    saved = ys.grad_fn.saved_tensors
    assert any(t is m.weight or t.data_ptr() == m.weight.data_ptr() for t in saved)
    assert not any(t.dtype == torch.bfloat16 and t.shape == m.weight.shape for t in saved)
    assert torch.equal(yr, ys)
    g = torch.randn_like(yr)
    yr.backward(g)
    ys.backward(g)
    assert torch.equal(xr.grad, xs.grad)
    assert m.weight.grad.dtype == torch.float32
    assert torch.equal(ref.weight.grad, m.weight.grad) and torch.equal(ref.bias.grad, m.bias.grad)


def test_a_finished_w_task_frees_its_steps_accumulator_without_the_cycle_collector():
    # covers: RUNTIME-I27
    # WOULD-FAIL-IF: a W task kept its step's accumulator (a full set of fp32 gradients) reachable after finishing,
    # e.g. through a class built per task, which only the cyclic collector frees (2026-10-13: +1.45 GiB a step on
    # a 0.42B-parameter stage until it ran out of memory)
    import gc
    import weakref

    from wantrain.runtime.stage import StageRuntime

    torch.manual_seed(7)
    m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
    assert wgrad.convert(m) == 2
    params = [p for p in m.parameters() if p.requires_grad]

    class _Stage:
        def __init__(self):
            self.done_mbs = {}
            self.made = []

        def _step_acc(self, step, vs):
            a = OrderedAccumulator(range(1))
            self.made.append(weakref.ref(a))
            return a

        def _sim_slow_wait(self, step, t0):
            pass

        def _ev(self, *a, **k):
            pass

    stage = _Stage()
    gc.disable()
    try:
        x = torch.randn(4, 16, requires_grad=True)
        with wgrad.capture() as store:
            m(x).square().sum().backward(inputs=[x])
        task = StageRuntime._w_task(stage, 0, 0, 0, params, store)
        assert task.run(deadline=0.0) is False  # stops after its first gradient
        assert task.run() is True  # resumes and finishes
        del task, store
        assert stage.made and stage.made[0]() is None, (
            "the finished task kept its accumulator alive"
        )
    finally:
        gc.enable()


def test_a_message_is_cut_into_one_fragment_per_flow_within_bounds():
    # covers: TRANSPORT-I12
    from wantrain.runtime.stage import frag_size

    assert frag_size(2_155_101, 16) == 134_694  # 16 fragments, every flow carries one
    assert frag_size(2_155_101, 1) == 1 << 20  # one flow: the old 1 MB cut
    assert frag_size(100_000, 16) == 64 << 10  # floor: no tiny frames
    assert frag_size(64 << 20, 16) == 1 << 20  # ceiling


def test_a_narrow_gradient_folds_to_the_widened_sum_bit_for_bit():
    # covers: RUNTIME-I28
    # WOULD-FAIL-IF: folding a bf16 part straight into its fp32 sum (widened by the first fold's copy, then inside
    # add_) gives other bits than widening each part first; or the first fold keeps the narrow dtype; or a held
    # part is widened before its fold.
    import random

    from wantrain.runtime.accumulate import OrderedAccumulator

    torch.manual_seed(0)
    parts = [torch.randn(4096) * 3 for _ in range(9)]
    narrow = [p.to(torch.bfloat16) for p in parts]
    order = list(range(9))
    arrival = order[:]
    random.Random(1).shuffle(arrival)  # held parts fold when their predecessors land
    wide, fused = OrderedAccumulator(order), OrderedAccumulator(order)
    for mb in arrival:
        wide.fold_part(mb, 0, 1, narrow[mb].to(torch.float32))
        wide.end_part(mb)
        fused.fold_part(mb, 0, 1, narrow[mb], torch.float32)
        assert all(t.dtype == torch.bfloat16 for t in fused.parts.values())
        fused.end_part(mb)
    a, b = wide.result()[0], fused.result()[0]
    assert b.dtype == torch.float32 and torch.equal(a, b)


def test_flush_each_narrow_yields_one_contribution_in_the_gemm_dtype_and_sums_wide():
    # covers: RUNTIME-I28
    # WOULD-FAIL-IF: widen=False widened a single-contribution gradient (the copy it exists to skip), or yielded a
    # multi-contribution parameter's sum in the narrow dtype (a bf16 sum is other bits).
    from wantrain.model import wgrad

    torch.manual_seed(0)
    lin = wgrad.SplitLinear(8, 8)
    twice = wgrad.SplitLinear(8, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        x = torch.randn(2, 8, requires_grad=True)
        with wgrad.capture() as store:
            y = twice(lin(x))
            y = twice(y)
            y.float().sum().backward()
    ref = dict(wgrad.flush_each(list(store)))
    got = dict(wgrad.flush_each(store, widen=False))
    assert got[lin.weight].dtype == torch.bfloat16
    assert torch.equal(got[lin.weight].to(torch.float32), ref[lin.weight])
    assert got[twice.weight].dtype == torch.float32 and torch.equal(
        got[twice.weight], ref[twice.weight]
    )
