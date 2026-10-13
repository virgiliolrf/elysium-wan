import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
import torch, hashlib
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
d, nh, hd, t = 512, 8, 64, 32
def permute(w, n_heads, dim1, dim2):
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
def inv_permute(w, n_heads, dim1, dim2):
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
def cache(t, hd, theta=10000.0):
    inv = 1.0/(theta**(torch.arange(0,hd,2,dtype=torch.float32)/hd))
    f = torch.outer(torch.arange(t,dtype=torch.float32), inv); return f
def rope_adj(x, f):
    c,s = f.cos(), f.sin(); x1,x2 = x[...,0::2], x[...,1::2]
    return torch.stack([x1*c-x2*s, x1*s+x2*c],-1).flatten(-2)
def rope_half(x, f):
    emb = torch.cat([f,f],-1); c,s = emb.cos(), emb.sin()
    h = x.shape[-1]//2; rot = torch.cat([-x[...,h:], x[...,:h]],-1)
    return x*c + rot*s
def H(x): return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()[:12]
for dt in (torch.float32, torch.bfloat16):
    x = torch.randn(1,t,d).to(dt)
    W_hf_q = (torch.randn(d,d)*0.05).to(dt); W_hf_k = (torch.randn(d,d)*0.05).to(dt)
    f = cache(t,hd)
    def qk(Wq,Wk,rope):
        q = (x@Wq.T).view(1,t,nh,hd).transpose(1,2); k=(x@Wk.T).view(1,t,nh,hd).transpose(1,2)
        q,k = rope(q,f.to(dt) if False else f), rope(k,f)
        return (q.to(dt)@k.to(dt).transpose(-1,-2))
    ref = qk(W_hf_q,W_hf_k,rope_half)                      # HF convention, HF weights
    bug = qk(W_hf_q,W_hf_k,rope_adj)                       # today: adjacent rope, unpermuted HF weights
    fix = qk(inv_permute(W_hf_q,nh,d,d),inv_permute(W_hf_k,nh,d,d),rope_adj)  # adjacent rope, inverse-permuted weights
    rt = permute(inv_permute(W_hf_q,nh,d,d),nh,d,d)
    print(dt, "bug maxdiff", (bug-ref).abs().max().item(), "| permuted maxdiff", (fix-ref).abs().max().item(),
          "bitequal", H(fix.float())==H(ref.float()), "| permute roundtrip bitequal", H(rt.float())==H(W_hf_q.float()))
    # fused kv vs separate
    Wk = W_hf_k[:128]; Wv = (torch.randn(128,d)*0.05).to(dt)
    xs = torch.randn(4*t,d).to(dt)
    fused = xs@torch.cat([Wk,Wv]).T; sep = torch.cat([xs@Wk.T, xs@Wv.T],-1)
    print(dt, "fused-kv vs separate bitequal", H(fused.float())==H(sep.float()), "maxdiff", (fused-sep).abs().max().item())
