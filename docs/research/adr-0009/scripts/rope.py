import os, sys  # added on commit (README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import sys; 
import torch
from wantrain.model.arch import apply_rope, rope_cache, RMSNorm
torch.manual_seed(0)
b,h,t,d=1,2,8,16
x=torch.randn(b,h,t,d)
cos,sin=rope_cache(t,d,10000.0,"cpu",torch.float32)
ours=apply_rope(x,cos,sin)
# HF rotate_half convention (LlamaRotaryEmbedding: emb=cat(freqs,freqs))
c2=torch.cat([cos,cos],-1)[None,None]; s2=torch.cat([sin,sin],-1)[None,None]
def rot_half(v): return torch.cat([-v[...,d//2:], v[...,:d//2]],-1)
hf=x*c2+rot_half(x)*s2
print("max|ours-hf| same weights:", (ours-hf).abs().max().item())
# the Meta->HF permutation: HF row j<d/2 = interleaved row 2j, j>=d/2 -> 2(j-d/2)+1
perm=torch.cat([torch.arange(0,d,2),torch.arange(1,d,2)])
inv=torch.argsort(perm)
ours_p=apply_rope(x[...,inv],cos,sin)[...,perm]
print("bitwise equal after un-permuting q/k rows:", torch.equal(ours_p,hf))
# q.k score invariance under permutation (what attention sees)
q=torch.randn(b,h,t,d); k=torch.randn(b,h,t,d)
sh=(apply_rope(q[...,inv],cos,sin)@apply_rope(k[...,inv],cos,sin).transpose(-1,-2))
sf=((q*c2+rot_half(q)*s2)@(k*c2+rot_half(k)*s2).transpose(-1,-2))
print("scores max diff after permutation (fp reassociation of dot):",(sh-sf).abs().max().item(), torch.equal(sh,sf))
# eps: arch RMSNorm fixed at 1e-6 vs Llama-3 rms_norm_eps=1e-5
n=RMSNorm(d); z=torch.randn(4,d)*1e-3
v=z.pow(2).mean(-1,keepdim=True); ref=z*torch.rsqrt(v+1e-5)
print("RMSNorm eps 1e-6 vs 1e-5, small-activation rel diff:", ((n(z)-ref).abs().max()/ref.abs().max()).item())
