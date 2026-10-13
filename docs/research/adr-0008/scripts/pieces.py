"""ADR-0008 prototype: ADR-0006-style per-tensor dim-0 pieces over the MoE model (every expert tensor cut in w
pieces, each piece folded in microbatch order from its slice of every microbatch's gradient and updated by its own
AdamW, then gathered back). Reference: proto.run_reference. Local Mac, CPU, 1 thread."""
import sys, copy, hashlib, itertools
sys.path.insert(0, "."); sys.argv = ["x"]
import torch, torch.nn as nn, torch.nn.functional as F
import proto as P
ALIGN = True

def run_pieces(L, steps, M, cut, w, skew_fold_from_zero=False):
    Ps = P.params(L)
    pieces = []  # (param, a, b, leaf)
    for p in Ps:
        import math
        row = p.shape[-1] if p.dim() > 1 else 1
        Q = row * 128 // math.gcd(row, 128) if ALIGN else row
        nq = -(-p.numel() // Q)
        bounds = [min(p.numel(), round(i * nq / w) * Q) for i in range(w + 1)]
        for a, b in zip(bounds, bounds[1:]):
            pieces.append((p, a, b, nn.Parameter(p.detach().reshape(-1)[a:b].clone())))
    leaves = [x[3] for x in pieces if x[3].numel()]
    opt = torch.optim.AdamW(leaves, lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    losses = []
    for step in range(steps):
        # reuse the pipeline path for per-microbatch gradients: capture them instead of folding
        grads = collect(L, step, M, cut, losses)
        for p, a, b, leaf in pieces:
            if not leaf.numel(): continue
            acc = torch.zeros_like(leaf) if skew_fold_from_zero else None
            for mb in range(M):
                g = grads[mb][id(p)].reshape(-1)[a:b]
                acc = g.clone() if acc is None else acc.add_(g)
            leaf.grad = acc
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, a, b, leaf in pieces: p.view(-1)[a:b].copy_(leaf)          # the gather: a byte copy
    h = hashlib.sha256()
    for p in Ps:
        mine = [x for x in pieces if x[0] is p and x[3].numel()]
        h.update(p.detach().contiguous().view(torch.uint8).numpy().tobytes())
        for key in ("exp_avg", "exp_avg_sq"):
            h.update(torch.cat([opt.state[x[3]][key] for x in mine]).contiguous().view(torch.uint8).numpy().tobytes())
        h.update(str(float(opt.state[mine[0][3]]["step"])).encode())
    return h.hexdigest(), losses

def collect(L, step, M, cut, losses):
    # one microbatch at a time through the 2-stage path with capture + deferred W (proto.run_pipeline's body)
    stages = [list(range(0, cut)), list(range(cut, len(L)))]
    out = {}
    for mb in range(M):
        x, tg = P.batch(step, mb); h = x; outs = []
        for si, idx in enumerate(stages):
            inp = h if si == 0 else h.detach().requires_grad_()
            o = inp
            for i in idx: o = L[i](o)
            outs.append((inp, o)); h = o
        gout = None; d = {}
        for si in (1, 0):
            inp, o = outs[si]; sp = [p for i in stages[si] for p in L[i].parameters() if p.requires_grad]
            owned = P.wgrad.owned_params(nn.ModuleList([L[i] for i in stages[si]]))
            others = [p for p in sp if id(p) not in owned]
            if si == 1:
                loss = F.cross_entropy(o.reshape(-1, o.size(-1)).float(), tg.reshape(-1)); losses.append(loss.item().hex()); root = loss
            else:
                root = o
            ins = ([inp] if si == 1 else []) + others
            with P.wgrad.capture() as store:
                gs = list(torch.autograd.grad(root, ins, grad_outputs=gout, allow_unused=True))
            if si == 1: gout = gs.pop(0)
            dws = P.wgrad.flush(store)
            for p, g in zip(others, gs): d[id(p)] = g
            for p in sp:
                if id(p) in owned: d[id(p)] = dws[id(p)]
        out[mb] = d
    return out

def ref_digest(L, steps, M):
    Ps = P.params(L); opt = torch.optim.AdamW(Ps, lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    r = P.run_reference(L, steps, M)  # builds its own optimizer; recompute digest the pieces way
    return r

if __name__ == "__main__":
    for cdt, (E, k), w, skew in itertools.product(("fp32", "bf16"), ((8, 2), (8, 3)), (2, 3), (False, True)):
        base = P.build(0, cdt, E, k, True, skew=skew)
        P.MUT.clear(); r = P.run_reference(copy.deepcopy(base), 2, 3)
        c = run_pieces(copy.deepcopy(base), 2, 3, 2, w)
        print(f"PIECES {cdt} E{E} k{k} w{w} skew={skew}: identical={r[0] == c[0] and r[1] == c[1]}")
    base = P.build(0, "bf16", 8, 3, True, skew=True)
    r = P.run_reference(copy.deepcopy(base), 2, 3); c = run_pieces(copy.deepcopy(base), 2, 3, 2, 3, skew_fold_from_zero=True)
    print(f"CTRL fold-from-zeros w3 skew: identical={r[0] == c[0] and r[1] == c[1]}")
    ALIGN = False
    for cdt in ("fp32", "bf16"):
        base = P.build(0, cdt, 8, 3, True)
        r = P.run_reference(copy.deepcopy(base), 2, 3); c = run_pieces(copy.deepcopy(base), 2, 3, 2, 3)
        print(f"CTRL row-aligned (not Q_t-aligned) pieces w3 {cdt}: identical={r[0] == c[0] and r[1] == c[1]}")
    print("sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
