"""ADR-0008 prototype (NOT product code): the proposed MoE block, reference (one graph, immediate dW) vs a
2-stage pipeline path (graph cut, capture, deferred W run in reverse order), sha256 of weights + AdamW state + every
loss and aux hex. Local Mac, CPU, 1 thread, deterministic algorithms."""
import sys, copy, hashlib, itertools
sys.path.insert(0, "/Users/virgiliofilho/edt-sec/src")
import torch, torch.nn as nn, torch.nn.functional as F
from wantrain.model import wgrad
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

class Fanout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return tuple(x.view_as(x) for _ in range(n))
    @staticmethod
    def backward(ctx, *gs):
        acc = gs[0]
        for g in gs[1:]:
            acc = acc + g
        return acc, None

class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, src, pos):
        ctx.save_for_backward(pos); return x.index_select(0, src)
    @staticmethod
    def backward(ctx, g):
        (pos,) = ctx.saved_tensors; T, k = pos.shape
        G = g.index_select(0, pos.reshape(-1)).view(T, k, -1)
        acc = G[:, 0]
        for j in range(1, k):
            acc = acc + G[:, j]
        return acc, None, None

class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Y, w, pos, inv, order):
        T, k = pos.shape
        Ysl = Y.index_select(0, pos.reshape(-1)).view(T, k, -1).float()
        ctx.save_for_backward(Ysl, w, inv); ctx.ydt = Y.dtype; ctx.order = order
        acc = None
        for j in order:
            t = w[:, j, None] * Ysl[:, j]
            acc = t if acc is None else acc + t
        return acc
    @staticmethod
    def backward(ctx, g):
        Ysl, w, inv = ctx.saved_tensors; T, k, d = Ysl.shape
        dYsl = (w[..., None] * g[:, None, :]).to(ctx.ydt).reshape(T * k, d)
        dY = dYsl.index_select(0, inv)
        dw = (g[:, None, :] * Ysl).sum(-1)
        return dY, dw, None, None, None

class AuxInject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, aux, alpha):
        ctx.alpha = alpha; return y.view_as(y)
    @staticmethod
    def backward(ctx, g):
        return g, torch.full((), ctx.alpha, dtype=torch.float32), None

class Router(nn.Module):  # not nn.Linear: wgrad.convert leaves it; fp32 math
    def __init__(s, d, E):
        super().__init__(); s.weight = nn.Parameter(torch.randn(E, d) * 0.02)

class SwiGLU(nn.Module):
    def __init__(s, d, f):
        super().__init__(); s.gate = nn.Linear(d, f, bias=False); s.up = nn.Linear(d, f, bias=False); s.down = nn.Linear(f, d, bias=False)
    def forward(s, x): return s.down(F.silu(s.gate(x)) * s.up(x))

MUT = set()
STATS = {"boundary_ties": 0, "empty_expert_mbs": 0}
AUX = []

class MoE(nn.Module):
    def __init__(s, d, f, E, k, shared):
        super().__init__(); s.E, s.k = E, k
        s.router = Router(d, E); s.experts = nn.ModuleList(SwiGLU(d, f) for _ in range(E))
        s.shared = SwiGLU(d, f) if shared else None
        s.shared_gate = nn.Linear(d, 1, bias=False) if shared else None
    def forward(s, xn, alpha=0.01):
        T, d = xn.shape; E, k = s.E, s.k
        n = 4 if s.shared is not None else 2
        xs = Fanout.apply(xn, n) if "no_fanout" not in MUT else (xn,) * n
        with torch.autocast(xn.device.type, enabled=False):
            logits = F.linear(xs[0].float(), s.router.weight)
            probs = logits.softmax(-1)
        srt = torch.sort(probs.detach(), dim=-1, descending=True, stable=True)
        STATS["boundary_ties"] += int((srt.values[:, k - 1] == srt.values[:, k]).sum())
        sel = torch.topk(probs.detach(), k, -1).indices if "topk" in MUT else srt.indices[:, :k]
        sel = sel.sort(-1).values                                   # ascending expert index per token
        p = probs.gather(1, sel); acc = p[:, 0]
        for j in range(1, k): acc = acc + p[:, j]
        w = p / acc[:, None]
        flat_e = sel.reshape(-1); flat_t = torch.arange(T).repeat_interleave(k)
        key = flat_e * T + (T - 1 - flat_t if "desc_rows" in MUT else flat_t)
        perm = torch.sort(key, stable=True).indices                # slots grouped by expert, tokens ascending
        src = flat_t[perm]; inv_perm = torch.empty_like(perm); inv_perm[perm] = torch.arange(T * k)
        pos = inv_perm.view(T, k)                                   # slot (t, j) -> row in dispatched tensor
        counts = torch.bincount(flat_e, minlength=E)
        STATS["empty_expert_mbs"] += int((counts == 0).sum())
        Xd = Dispatch.apply(xs[1], src, pos)
        chunks = torch.split(Xd, counts.tolist())
        ys = []
        for e in range(E):
            if "skip_empty" in MUT and counts[e] == 0:
                continue
            ys.append(s.experts[e](chunks[e]))
        Y = torch.cat(ys) if ys else Xd[:0]
        order = list(range(k))[::-1] if "rev_combine" in MUT else list(range(k))
        out = Combine.apply(Y, w, pos, perm, order)
        if s.shared is not None:
            sh = torch.sigmoid(s.shared_gate(xs[3])).float() * s.shared(xs[2]).float()
            out = out + sh
        aux = (E / (T * T * k)) * (probs.sum(0) * counts.float()).sum()
        if torch.is_grad_enabled(): AUX.append(aux.item().hex())
        return AuxInject.apply(out, aux, alpha)

class Block(nn.Module):
    def __init__(s, d, f, E, k, shared, cdt):
        super().__init__(); s.ln = nn.LayerNorm(d); s.moe = MoE(d, f, E, k, shared); s.cdt = cdt
    def forward(s, x):
        b, t, d = x.shape
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=s.cdt == "bf16"):
            y = s.moe(s.ln(x).reshape(b * t, d)).view(b, t, d)
        return x + y.to(x.dtype)

def build(seed, cdt, E, k, shared, d=16, f=24, V=32, ties=True, skew=False):
    torch.manual_seed(seed)
    L = nn.ModuleList([nn.Embedding(V, d), Block(d, f, E, k, shared, cdt), Block(d, f, E, k, shared, cdt), nn.Linear(d, V, bias=False)])
    with torch.no_grad():
        for blk in (L[1], L[2]):
            W = blk.moe.router.weight
            if ties: W[1] = W[0]; W[3] = W[2]          # duplicated rows: every token ties experts (0,1) and (2,3)
            if skew: W[:] = 0; W[0, :] = 0.0           # all-zero router: every token ties all E -> experts 0..k-1
    wgrad.convert(L)
    return L

def batch(step, mb, V=32, b=2, t=3):
    g = torch.Generator().manual_seed(1000 * step + mb)
    x = torch.randint(0, V, (b, t + 1), generator=g); return x[:, :-1], x[:, 1:]

def params(L): return [p for p in L.parameters() if p.requires_grad]

def digest(L, opt):
    h = hashlib.sha256()
    for p in params(L):
        h.update(p.detach().contiguous().view(torch.uint8).numpy().tobytes())
        st = opt.state[p]
        if not st: h.update(b"nostate"); continue
        for key in ("exp_avg", "exp_avg_sq"): h.update(st[key].contiguous().view(torch.uint8).numpy().tobytes())
        h.update(str(float(st["step"])).encode())
    return h.hexdigest()

def run_reference(L, steps, M):
    P = params(L); opt = torch.optim.AdamW(P, lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0); losses = []
    for step in range(steps):
        acc = None
        for mb in range(M):
            x, tg = batch(step, mb); h = x
            for layer in L: h = layer(h)
            loss = F.cross_entropy(h.reshape(-1, h.size(-1)).float(), tg.reshape(-1)); losses.append(loss.item().hex())
            gs = torch.autograd.grad(loss, P)
            acc = [g.clone() for g in gs] if acc is None else [a.add_(g) for a, g in zip(acc, gs)]
        for p, g in zip(P, acc): p.grad = g
        opt.step(); opt.zero_grad(set_to_none=True)
    return digest(L, opt), losses

def run_pipeline(L, steps, M, cut):
    P = params(L); opt = torch.optim.AdamW(P, lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0); losses = []
    stages = [list(range(0, cut)), list(range(cut, len(L)))]
    for step in range(steps):
        Ws = []; fwd = {}
        for mb in range(M):                       # F of every microbatch on both stages first
            x, tg = batch(step, mb); h = x; outs = []
            with torch.set_grad_enabled("recompute" not in MUT):
                for si, idx in enumerate(stages):
                    inp = h if si == 0 else h.detach().requires_grad_()
                    o = inp
                    for i in idx: o = L[i](o)
                    outs.append((inp, o)); h = o
            fwd[mb] = (outs, tg)
        grads = {}
        for mb in range(M):                       # B per stage, W deferred and run in reverse microbatch order
            outs, tg = fwd[mb]; gout = None; per = {}
            if "recompute" in MUT:                    # F kept only boundaries; B rebuilds each stage's graph
                x, _ = batch(step, mb); h = x; outs = []
                for si, idx in enumerate(stages):
                    inp = h if si == 0 else h.detach().requires_grad_()
                    o = inp
                    for i in idx: o = L[i](o)
                    outs.append((inp, o)); h = o
            for si in (1, 0):
                inp, o = outs[si]; sp = [p for i in stages[si] for p in L[i].parameters() if p.requires_grad]
                owned = wgrad.owned_params(nn.ModuleList([L[i] for i in stages[si]]))
                others = [p for p in sp if id(p) not in owned]
                if si == 1:
                    loss = F.cross_entropy(o.reshape(-1, o.size(-1)).float(), tg.reshape(-1)); losses.append(loss.item().hex()); root = loss
                else:
                    root = o
                ins = ([inp] if si == 1 else []) + others
                with wgrad.capture() as store:
                    gs = list(torch.autograd.grad(root, ins, grad_outputs=gout, allow_unused=True))
                if si == 1: gout = gs.pop(0)
                per[si] = (sp, dict(zip(map(id, others), gs)), store)
            Ws.append((mb, per))
        for mb, per in reversed(Ws):
            gl = []
            for si in (0, 1):
                sp, og, store = per[si]; dws = wgrad.flush(store)
                for p in sp:
                    g = og.get(id(p)) if id(p) in og else dws.get(id(p))
                    gl.append((p, g))
            grads[mb] = gl
        acc = {}
        for mb in range(M):
            for p, g in grads[mb]:
                if g is None: continue                    # only the skip_empty control has None
                acc[id(p)] = g.clone() if id(p) not in acc else acc[id(p)].add_(g)
        for p in P: p.grad = acc.get(id(p))
        opt.step(); opt.zero_grad(set_to_none=True)
    return digest(L, opt), losses

def arm(cdt, E, k, shared, cut, mut=(), ties=True, skew=False, steps=2, M=3, mut_ref=()):
    base = build(0, cdt, E, k, shared, ties=ties, skew=skew)
    MUT.clear(); MUT.update(mut_ref); AUX.clear(); STATS.update(boundary_ties=0, empty_expert_mbs=0)
    r = run_reference(copy.deepcopy(base), steps, M); ra = list(AUX); rs = dict(STATS)
    MUT.clear(); MUT.update(mut); AUX.clear()
    c = run_pipeline(copy.deepcopy(base), steps, M, cut); ca = list(AUX)
    same = r[0] == c[0] and r[1] == c[1] and sorted(ra) == sorted(ca)
    return same, rs

if __name__ == "__main__":
    for cdt, (E, k), shared, cut in itertools.product(("fp32", "bf16"), ((8, 2), (8, 3), (6, 4)), (False, True), (1, 2, 3)):
        same, st = arm(cdt, E, k, shared, cut)
        print(f"TRUE  {cdt} E{E} k{k} shared={shared} cut={cut}: identical={same} ties={st['boundary_ties']} empty={st['empty_expert_mbs']}")
    same, st = arm("bf16", 8, 3, True, 2, skew=True)
    print(f"TRUE  skew(all-zero router) bf16 E8 k3: identical={same} ties={st['boundary_ties']} empty={st['empty_expert_mbs']}")
    for mut in ("topk", "skip_empty", "rev_combine", "desc_rows", "no_fanout"):
        for cdt in ("fp32", "bf16"):
            same, st = arm(cdt, 8, 3, True, 2, mut=(mut,), skew=(mut in ("topk", "skip_empty")))
            print(f"CTRL  {mut:12s} {cdt}: identical={same} ties={st['boundary_ties']} empty={st['empty_expert_mbs']}")
    for cdt in ("fp32", "bf16"):
        for cut in (1, 2, 3):
            same, st = arm(cdt, 8, 3, True, cut, mut=("no_fanout",), mut_ref=("no_fanout",))
            print(f"INFO  no_fanout on both paths {cdt} cut={cut}: identical={same}")
    for cdt in ("fp32", "bf16"):
        same, st = arm(cdt, 8, 3, True, 2, mut=("recompute",))
        print(f"TRUE  recompute {cdt} E8 k3 shared cut=2: identical={same}")
    print("sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
