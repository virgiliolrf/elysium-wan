import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
import json, torch, torch.nn.functional as F
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.py")).read().split("res=[]")[0]
exec(src)
FAM["deepseek_v3"]["head_dim"]=8
FAM["deepseek_v3"]["first_k_dense_replace"]=2
for attn in ("eager","sdpa"):
    try: print(json.dumps(run("deepseek_v3",attn)))
    except Exception as e: print("deepseek",attn,type(e).__name__,str(e)[:200])
# bf16 autocast: dtype at the boundary and parity
import contextlib
_run=run
def run_ac(mt,attn):
    with torch.autocast("cpu",dtype=torch.bfloat16):
        return _run(mt,attn)
for mt in ("llama","gemma3_text","qwen3","gpt_neox"):
    try: print("bf16-autocast",json.dumps(run_ac(mt,"sdpa")))
    except Exception as e: print("bf16",mt,type(e).__name__,str(e)[:200])
m,cfg=build("llama","sdpa")
_,L=layers_of(m)
with torch.autocast("cpu",dtype=torch.bfloat16):
    st,_=capture(m,torch.randint(0,128,(2,16)))
    h=st[0][0]; o=L[0](h,*st[0][1],**st[0][2])
print("autocast boundary dtypes: embed-out",h.dtype,"layer-out",o.dtype)
mb=m.to(torch.bfloat16)
st,_=capture(mb,torch.randint(0,128,(2,16))); o=L[0](st[0][0],*st[0][1],**st[0][2]); print("bf16-params layer-out",o.dtype)
# tied: per-microbatch sum then fold (reference) vs Megatron order sum(head)+sum(embed)
torch.manual_seed(3)
diff=0; n=0
for trial in range(20):
    a=[torch.randn(128,64)*1e-3 for _ in range(8)]; b=[torch.randn(128,64)*1e-3 for _ in range(8)]
    r=a[0]+b[0]
    for i in range(1,8): r=r+(a[i]+b[i])
    sa=a[0]; sb=b[0]
    for i in range(1,8): sa=sa+a[i]; sb=sb+b[i]
    alt=sa+sb
    diff+=(r!=alt).sum().item(); n+=r.numel()
print(f"tied fold order: {diff}/{n} elements differ between per-mb-sum-then-fold and fold-each-then-sum")
print("-0.0 + +0.0 =", (torch.tensor(-0.0)+torch.tensor(0.0)).item(), "signbit", torch.signbit(torch.tensor(-0.0)+torch.tensor(0.0)).item())
