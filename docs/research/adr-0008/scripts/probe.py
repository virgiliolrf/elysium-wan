"""ADR-0008 probes (local Mac, CPU, 1 thread). Re-checks the researchers' claims the ADR relies on."""
import sys, hashlib
sys.path.insert(0, "src")  # run from the repository root
import torch, torch.nn as nn, torch.nn.functional as F
from wantrain.model import wgrad
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
out = {}
# P1: an expert called on 0 rows keeps its weight in the graph; dW is +0 on both paths
lin = nn.Linear(16, 8, bias=False); wgrad.convert(nn.ModuleList([nn.Sequential(lin)]))
lin = nn.Sequential(lin)[0]
x = torch.randn(5, 16, requires_grad=True)
y = lin(x[torch.tensor([], dtype=torch.long)]).sum() + x.sum()
g = torch.autograd.grad(y, [lin.weight])[0]          # reference path: no allow_unused
out["P1_ref_grad_is_pos_zero"] = bool((g == 0).all() and not torch.signbit(g).any())
x2 = torch.randn(5, 16, requires_grad=True)
with wgrad.capture() as st:
    y = lin(x2[torch.tensor([], dtype=torch.long)]).sum() + x2.sum()
    torch.autograd.grad(y, [x2])
d = wgrad.flush(st)
gw = d.get(id(lin.weight))
out["P1_deferred_entry"] = gw is not None and bool((gw == 0).all() and not torch.signbit(gw).any())
# P2: AdamW given None vs a zero gradient
def adam(grads):
    p = nn.Parameter(torch.tensor([0.9, -0.3])); o = torch.optim.AdamW([p], lr=0.1, betas=(0.9, 0.95), weight_decay=0.0)
    for gg in grads:
        p.grad = None if gg is None else torch.tensor(gg); o.step(); o.zero_grad(set_to_none=True)
    return p.detach().clone(), o.state[p]["step"].item()
a, sa = adam([[1.0, 1.0], None]); b, sb = adam([[1.0, 1.0], [0.0, 0.0]])
out["P2_none_vs_zero_differ"] = (not torch.equal(a, b), sa, sb)
# P3: row-count dependence of a GEMM and split-K of dW (fp32)
W = torch.randn(512, 256); X = torch.randn(2048, 256)
out["P3_row_dep_M1_vs_M2048_elems_differ"] = int((X[:1] @ W.t() != (X @ W.t())[:1]).sum())
G = torch.randn(300, 512); Xd = torch.randn(300, 256)
one = G.t().mm(Xd); two = G[:150].t().mm(Xd[:150]) + G[150:].t().mm(Xd[150:])
out["P3_dW_one_vs_two_gemms_differ"] = (int((one != two).sum()), one.numel())
# P4: tie-breaking. bf16-rounded logits, E=64, k=8, 20000 tokens
T, E, k = 20000, 64, 8
lg = torch.randn(T, E).to(torch.bfloat16).float()
ref = torch.sort(lg, dim=-1, descending=True, stable=True).indices[:, :k]
# lowest-index rule by brute force: key = value desc, index asc
key = torch.argsort(-lg * 0 + torch.arange(E).float() * 0, dim=-1)  # placeholder unused
tk = torch.topk(lg, k, dim=-1).indices
same_set = lambda A, B: (A.sort(-1).values == B.sort(-1).values).all(-1)
out["P4_topk_vs_stable_sort_set_diff_tokens"] = int((~same_set(tk, ref)).sum())
# verify stable sort = lowest-index rule exactly, via an int64 composite key (order-preserving map of fp32)
bits = lg.view(torch.int32).to(torch.int64)
mono = torch.where(bits >= 0, bits, -(bits & 0x7FFFFFFF))  # -0 -> 0, order preserving for non-NaN
ck = mono * 4096 + (4095 - torch.arange(E))
kt = torch.topk(ck, k, dim=-1).indices
out["P4_stable_sort_vs_int_key_rank_diff_tokens"] = int((kt != ref).any(-1).sum())
if torch.backends.mps.is_available():
    refm = torch.sort(lg.to("mps"), dim=-1, descending=True, stable=True).indices[:, :k].cpu()
    out["P4_stable_sort_cpu_vs_mps_rank_diff_tokens"] = int((refm != ref).any(-1).sum())
ties = 0
srt = torch.sort(lg, -1, descending=True).values
out["P4_tokens_with_tie_at_boundary_pct"] = round(100 * float((srt[:, k - 1] == srt[:, k]).float().mean()), 2)
lg32 = torch.randn(T, E)
srt32 = torch.sort(lg32, -1, descending=True).values
out["P4_fp32_boundary_ties"] = int((srt32[:, k - 1] == srt32[:, k]).sum())
# P5: combine order for k=8 and k=2
for kk in (2, 3, 8):
    Y = torch.randn(T // 10, kk, 64); w = torch.rand(T // 10, kk)
    P = w[..., None] * Y
    fold = P[:, 0].clone()
    for j in range(1, kk): fold = fold + P[:, j]
    rev = P[:, kk - 1].clone()
    for j in range(kk - 2, -1, -1): rev = rev + P[:, j]
    out[f"P5_k{kk}_sum_vs_fold_tokens_pct"] = round(100 * float((P.sum(1) != fold).any(-1).float().mean()), 1)
    out[f"P5_k{kk}_rev_vs_fold_tokens_pct"] = round(100 * float((rev != fold).any(-1).float().mean()), 1)
for kq, v in out.items(): print(kq, v)
print("sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
