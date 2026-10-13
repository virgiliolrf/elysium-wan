import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
import sys, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())


import stub_inject as si
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from wantrain.model import wgrad
torch.set_num_threads(1)
def mk(fam, kw, dtype=torch.float32, attn="eager"):
    cfg = AutoConfig.for_model(fam, **kw); torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn, dtype=dtype); m.train(); return m
ids = torch.randint(0, 128, (2, 24), generator=torch.Generator().manual_seed(1))
# C1: layer-call counts per stage (did stubs really replace non-owned layers?)
m = mk("llama", si.fams["llama"]); m._wt_wrap = False
_, layers = si.find_layers(m); calls = []
for i, l in enumerate(layers): l.register_forward_pre_hook(lambda mod, a, i=i: calls.append(i))
h, k = si.run_stage(m, ids, 0, 1); c0 = list(calls); calls.clear()
h2, k2 = si.run_stage(m, ids, 1, 3, h.detach()); c1 = list(calls); calls.clear()
lg, k3 = si.run_stage(m, ids, 3, 4, h2.detach()); c2 = list(calls)
print("C1 real layer calls per stage:", c0, c1, c2, k, k2, k3)
# C2: one-ulp perturbation of the injected state must change the logits
hp = h2.detach().clone(); hp.view(-1)[5] = torch.nextafter(hp.view(-1)[5], torch.tensor(float("inf")))
lgp, _ = si.run_stage(m, ids, 3, 4, hp)
print("C2 1-ulp perturbed boundary -> logits equal?", torch.equal(lg, lgp))
# C3: eager vs sdpa (math on CPU) logits: bitwise sensitivity across attention implementations
for fam in ("llama", "gemma2"):
    a = mk(fam, si.fams[fam], attn="eager")(input_ids=ids, use_cache=False).logits
    b = mk(fam, si.fams[fam], attn="sdpa")(input_ids=ids, use_cache=False).logits
    print(f"C3 {fam} eager==sdpa logits?", torch.equal(a, b), "max|d|", (a - b).abs().max().item())
# C4: does sdpa silently drop gemma2 attention softcapping?
kw = dict(si.fams["gemma2"]); kw0 = dict(kw, attn_logit_softcapping=None)
for attn in ("eager", "sdpa"):
    a = mk("gemma2", kw, attn=attn)(input_ids=ids, use_cache=False).logits
    b = mk("gemma2", kw0, attn=attn)(input_ids=ids, use_cache=False).logits
    print(f"C4 gemma2 {attn}: softcap 50 vs None logits equal?", torch.equal(a, b))
# C5: SplitLinear surgery inside HF modules: ref (stock nn.Linear, monolithic) vs converted model with deferred W
for fam in ("llama", "qwen3", "gemma3_text", "deepseek_v3", "gpt2"):
    for dtype in (torch.float32, torch.bfloat16):
        ref = mk(fam, si.fams[fam], dtype); sur = mk(fam, si.fams[fam], dtype)
        n = wgrad.convert(sur)
        tgt = torch.randint(0, 128, (2, 24), generator=torch.Generator().manual_seed(2))
        ref(input_ids=ids, use_cache=False).logits.float().reshape(-1,128).pipe if False else None
        l1 = F.cross_entropy(ref(input_ids=ids, use_cache=False).logits.float().reshape(-1, 128), tgt.reshape(-1)); l1.backward()
        with wgrad.capture() as store:
            l2 = F.cross_entropy(sur(input_ids=ids, use_cache=False).logits.float().reshape(-1, 128), tgt.reshape(-1)); l2.backward()
        dw = wgrad.flush(store)
        bad = 0; tot = 0
        for (na, pa), (nb, pb) in zip(ref.named_parameters(), sur.named_parameters()):
            gb = dw.get(id(pb), pb.grad); tot += 1
            if pa.grad is None and gb is None: continue
            if pa.grad is None or gb is None or not torch.equal(pa.grad, gb): bad += 1
        print(f"C5 {fam:12s} {str(dtype)[6:]:9s} converted={n:3d} loss equal={torch.equal(l1,l2)} grads differ {bad}/{tot}")
# C6: query-chunked eager attention vs full (row-slice invariance of the CPU GEMM), fwd + bwd
torch.manual_seed(3)
for dtype in (torch.float32, torch.bfloat16):
  for S, D in ((256, 64), (1024, 128)):
    q = torch.randn(1, 4, S, D, dtype=dtype, requires_grad=True); kk = torch.randn(1, 4, S, D, dtype=dtype, requires_grad=True); v = torch.randn(1, 4, S, D, dtype=dtype, requires_grad=True)
    mask = torch.full((S, S), float("-inf")).triu(1).to(dtype)
    def eager(q, k, v, msk):
        w = torch.matmul(q, k.transpose(2, 3)) * D ** -0.5 + msk
        return torch.matmul(F.softmax(w, dim=-1, dtype=torch.float32).to(dtype), v)
    full = eager(q, kk, v, mask)
    for C in (64, 128):
        ch = torch.cat([eager(q[:, :, i:i+C], kk, v, mask[i:i+C]) for i in range(0, S, C)], dim=2)
        go = torch.randn_like(full)
        g1 = torch.autograd.grad(full, [q, kk, v], go, retain_graph=True); g2 = torch.autograd.grad(ch, [q, kk, v], go)
        print(f"C6 {str(dtype)[6:]:9s} S={S} D={D} chunk={C}: fwd equal {torch.equal(full, ch)}; dq,dk,dv equal {[torch.equal(a,b) for a,b in zip(g1,g2)]}")
# C7: math SDPA: is_causal vs explicit bool mask vs enable_gqa vs repeat_kv
from torch.nn.attention import sdpa_kernel, SDPBackend
q = torch.randn(2, 8, 128, 64); k = torch.randn(2, 2, 128, 64); v = torch.randn(2, 2, 128, 64)
with sdpa_kernel(SDPBackend.MATH):
    kr, vr = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
    a = F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
    b = F.scaled_dot_product_attention(q, kr, vr, attn_mask=torch.ones(128, 128, dtype=torch.bool).tril())
    c = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
print("C7 math sdpa is_causal==bool mask", torch.equal(a, b), "; enable_gqa==repeat_kv", torch.equal(a, c))
