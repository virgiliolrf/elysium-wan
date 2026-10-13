"""ADR-0008 round-2 probes (NOT product code). Both-paths arms the round-1 controls lacked, and cross-backend and
run-to-run observations on the local Mac (CPU and MPS). Run from this directory:
  .venv/bin/python -I docs/research/adr-0008/scripts/probe3.py <repo src dir>
Q1  torch.topk on BOTH the oracle and the 2-stage path (tie and all-tie routers, recompute).
Q2  an HF-style block (compute-dtype router, fp32 softmax, topk, renorm by .sum, loop over HIT experts, index_add_,
    no fan-out function) on BOTH paths, oracle with allow_unused and None-skipping AdamW like the stage path.
Q3  the same HF-style block looping over EVERY expert (0-row calls), so no gradient is None.
Q4  torch.topk vs the stable sort, CPU vs MPS, on P4's 20,000 bf16-rounded rows (E 64, k 8).
Q5  MPS run-to-run: index_add_ and scatter_add_ with duplicate indices, stable sort, topk (20 runs each).
Q6  torch._grouped_mm availability on this build (CPU, MPS) and per-segment equality with a per-segment mm.
Q7  MPS index_add_ with UNIQUE indices (HF's per-expert call) under deterministic algorithms."""
import sys, os, copy, hashlib, itertools
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../../../../src")
sys.path.insert(0, os.path.abspath(SRC))
from wantrain.model import wgrad  # noqa: E402  (this checkout's wgrad, cached before proto's path insert)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F  # noqa: E402
import proto  # noqa: E402

torch.set_num_threads(1); torch.use_deterministic_algorithms(True)


class HFMoE(proto.MoE):
    """transformers v4 MixtralSparseMoeBlock's eager loop, with Qwen2-MoE's sigmoid-gated shared expert."""
    def forward(s, xn, alpha=0.01):
        T, d = xn.shape; E, k = s.E, s.k
        logits = F.linear(xn, s.router.weight)                       # compute dtype under autocast (HF gate Linear)
        probs = F.softmax(logits, dim=1, dtype=torch.float)
        w, sel = torch.topk(probs, k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        w = w.to(xn.dtype)
        out = torch.zeros((T, d), dtype=xn.dtype)
        mask = F.one_hot(sel, num_classes=E).permute(2, 1, 0)
        counts = torch.bincount(sel.reshape(-1), minlength=E)
        proto.STATS["empty_expert_mbs"] += int((counts == 0).sum())
        srt = torch.sort(probs.detach(), dim=-1, descending=True, stable=True).values
        proto.STATS["boundary_ties"] += int((srt[:, k - 1] == srt[:, k]).sum())
        for e in range(E):
            idx, top_x = torch.where(mask[e])
            if "hf_all" not in proto.MUT and top_x.numel() == 0:
                continue
            cur = xn[None, top_x].reshape(-1, d)
            y = s.experts[e](cur) * w[top_x, idx, None]
            out.index_add_(0, top_x, y.to(xn.dtype))
        if s.shared is not None:
            out = out + torch.sigmoid(s.shared_gate(xn)) * s.shared(xn)
        aux = (E / (T * T * k)) * (probs.sum(0) * counts.float()).sum()
        if torch.is_grad_enabled(): proto.AUX.append(aux.item().hex())
        return proto.AuxInject.apply(out, aux, alpha)


def run_reference_unused(L, steps, M):
    """The oracle with allow_unused, folding like the stage path (None skipped, AdamW skips a None parameter)."""
    P = proto.params(L)
    opt = torch.optim.AdamW(P, lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0); losses = []
    for step in range(steps):
        acc = {}
        for mb in range(M):
            x, tg = proto.batch(step, mb); h = x
            for layer in L: h = layer(h)
            loss = F.cross_entropy(h.reshape(-1, h.size(-1)).float(), tg.reshape(-1)); losses.append(loss.item().hex())
            gs = torch.autograd.grad(loss, P, allow_unused=True)
            for p, g in zip(P, gs):
                if g is None: continue
                acc[id(p)] = g.clone() if id(p) not in acc else acc[id(p)].add_(g)
        for p in P: p.grad = acc.get(id(p))
        opt.step(); opt.zero_grad(set_to_none=True)
    return proto.digest(L, opt), losses


def arm(cdt, E, k, shared, cut, mut=(), skew=False, hf=False, steps=2, M=3):
    base = proto.build(0, cdt, E, k, shared, ties=True, skew=skew)
    if hf:
        for blk in (base[1], base[2]): blk.moe.__class__ = HFMoE
    proto.MUT.clear(); proto.MUT.update(m for m in mut if m != "recompute")
    proto.AUX.clear(); proto.STATS.update(boundary_ties=0, empty_expert_mbs=0)
    r = run_reference_unused(copy.deepcopy(base), steps, M) if hf else proto.run_reference(copy.deepcopy(base), steps, M)
    ra = list(proto.AUX); rs = dict(proto.STATS)
    proto.MUT.clear(); proto.MUT.update(mut); proto.AUX.clear()
    c = proto.run_pipeline(copy.deepcopy(base), steps, M, cut); ca = list(proto.AUX)
    return (r[0] == c[0] and r[1] == c[1] and sorted(ra) == sorted(ca)), rs


def q1_q3():
    n = {"Q1": [0, 0], "Q2": [0, 0], "Q3": [0, 0]}
    for cdt, skew, cut in itertools.product(("fp32", "bf16"), (False, True), (1, 2, 3)):
        same, st = arm(cdt, 8, 3, True, cut, mut=("topk",), skew=skew)  # proto's oracle reads MUT too: topk both paths
        n["Q1"][0] += same; n["Q1"][1] += 1
        print(f"Q1 topk both paths {cdt} skew={skew} cut={cut}: identical={same} ties={st['boundary_ties']} empty={st['empty_expert_mbs']}")
    for cdt in ("fp32", "bf16"):
        same, st = arm(cdt, 8, 3, True, 2, mut=("topk", "recompute"), skew=True)
        n["Q1"][0] += same; n["Q1"][1] += 1
        print(f"Q1 topk both paths + recompute {cdt}: identical={same}")
    for tag, mut in (("Q2", ()), ("Q3", ("hf_all",))):
        for cdt, (E, k), shared, cut in itertools.product(("fp32", "bf16"), ((8, 2), (8, 3), (6, 4)), (False, True), (1, 2, 3)):
            same, st = arm(cdt, E, k, shared, cut, mut=mut, hf=True)
            n[tag][0] += same; n[tag][1] += 1
            print(f"{tag} HF-style {cdt} E{E} k{k} shared={shared} cut={cut}: identical={same} ties={st['boundary_ties']} empty={st['empty_expert_mbs']}")
        for cdt in ("fp32", "bf16"):
            for extra in ((), ("recompute",)):
                same, st = arm(cdt, 8, 3, True, 2, mut=mut + extra, skew=True, hf=True)
                n[tag][0] += same; n[tag][1] += 1
                print(f"{tag} HF-style all-tie router {cdt} {extra}: identical={same} ties={st['boundary_ties']} empty={st['empty_expert_mbs']}")
    return n


def q4():
    torch.manual_seed(0)
    T, E, k = 20000, 64, 8
    lg = torch.randn(T, E).to(torch.bfloat16).float()
    same_set = lambda A, B: (A.sort(-1).values == B.sort(-1).values).all(-1)
    st_cpu = torch.sort(lg, dim=-1, descending=True, stable=True).indices[:, :k]
    tk_cpu = torch.topk(lg, k, -1).indices
    print("Q4 cpu topk vs cpu stable sort, set differs:", int((~same_set(tk_cpu, st_cpu)).sum()))
    if not torch.backends.mps.is_available():
        print("Q4 mps unavailable"); return
    m = lg.to("mps")
    st_mps = torch.sort(m, dim=-1, descending=True, stable=True).indices[:, :k].cpu()
    tk_mps = torch.topk(m, k, -1).indices.cpu()
    print("Q4 mps topk vs mps stable sort, set differs:", int((~same_set(tk_mps, st_mps)).sum()))
    print("Q4 topk cpu vs mps, set differs:", int((~same_set(tk_cpu, tk_mps)).sum()),
          "index order differs:", int((tk_cpu != tk_mps).any(-1).sum()))
    print("Q4 stable sort cpu vs mps, rank differs:", int((st_cpu != st_mps).any(-1).sum()))
    for dt in (torch.float32, torch.bfloat16):
        mm = m.to(dt)
        runs = {hashlib.sha256(torch.topk(mm, k, -1).indices.cpu().numpy().tobytes()).hexdigest() for _ in range(20)}
        print(f"Q4 mps topk {dt} distinct outputs over 20 runs:", len(runs))


def q5():
    if not torch.backends.mps.is_available():
        print("Q5 mps unavailable"); return
    torch.manual_seed(1)
    src = torch.randn(200000, 64)
    idx = torch.randint(0, 512, (200000,))
    for det in (True, False):
        torch.use_deterministic_algorithms(det)
        for name, fn in (
            ("index_add_", lambda s, i: torch.zeros(512, 64, device=s.device).index_add_(0, i, s)),
            ("scatter_add_", lambda s, i: torch.zeros(512, 64, device=s.device).scatter_add_(0, i[:, None].expand(-1, 64), s)),
        ):
            try:
                outs = {hashlib.sha256(fn(src.to("mps"), idx.to("mps")).cpu().numpy().tobytes()).hexdigest() for _ in range(20)}
                cpu = hashlib.sha256(fn(src, idx).numpy().tobytes()).hexdigest()
                print(f"Q5 mps {name} duplicates det={det}: distinct over 20 runs {len(outs)}; equals cpu {cpu in outs}")
            except Exception as ex:  # a refusal under deterministic mode is itself the observation
                print(f"Q5 mps {name} det={det}: raised {type(ex).__name__}: {str(ex)[:120]}")
    torch.use_deterministic_algorithms(True)
    key = torch.randn(4096, 128).to(torch.bfloat16).float().to("mps")
    runs = {hashlib.sha256(torch.sort(key, dim=-1, descending=True, stable=True).indices.cpu().numpy().tobytes()).hexdigest() for _ in range(20)}
    print("Q5 mps stable sort distinct over 20 runs:", len(runs))


def q6():
    g = getattr(torch, "_grouped_mm", None)
    print("Q6 torch._grouped_mm present:", g is not None)
    if g is None: return
    for dev in ("cpu", "mps"):
        if dev == "mps" and not torch.backends.mps.is_available(): continue
        try:
            torch.manual_seed(2)
            sizes = [0, 5, 37, 128]
            X = torch.randn(sum(sizes), 64, dtype=torch.bfloat16, device=dev)
            W = torch.randn(len(sizes), 64, 32, dtype=torch.bfloat16, device=dev)
            offs = torch.tensor(list(itertools.accumulate(sizes)), dtype=torch.int32, device=dev)
            Y = g(X, W, offs=offs)
            ref = torch.cat([X[a:b] @ W[e] for e, (a, b) in enumerate(zip([0] + offs.tolist()[:-1], offs.tolist()))])
            print(f"Q6 {dev} grouped_mm == per-segment mm: {torch.equal(Y.cpu(), ref.cpu())}")
        except Exception as ex:
            print(f"Q6 {dev} grouped_mm raised {type(ex).__name__}: {str(ex)[:160]}")


def q7():
    if not torch.backends.mps.is_available():
        print("Q7 mps unavailable"); return
    torch.use_deterministic_algorithms(True)
    src = torch.randn(300, 64, device="mps"); idx = torch.randperm(512)[:300].to("mps")
    try:
        torch.zeros(512, 64, device="mps").index_add_(0, idx, src)
        print("Q7 mps index_add_ unique indices det=True: ran")
    except Exception as ex:
        print(f"Q7 mps index_add_ unique indices det=True: raised {type(ex).__name__}: {str(ex)[:120]}")


if __name__ == "__main__":
    print("torch", torch.__version__)
    n = q1_q3()
    for kq, (a, b) in n.items(): print(f"SUMMARY {kq}: {a}/{b} identical")
    q4(); q5(); q6(); q7()
    print("sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
