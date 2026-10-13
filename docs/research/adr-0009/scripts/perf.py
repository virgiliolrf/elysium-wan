import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import sys, os, time, warnings;  os.environ["USE_HUB_KERNELS"]="0"; warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F, transformers; transformers.logging.set_verbosity_error()
from transformers import LlamaConfig, LlamaForCausalLM
from wantrain.model import gpt, wgrad
torch.set_num_threads(4)
d,ff,nl,nh,kv,V,T,mb=1024,2816,4,16,4,8192,512,2
g=gpt.GPTConfig(vocab_size=V,n_layer=nl,n_head=nh,n_kv_head=kv,d_model=d,d_ff=ff,seq_len=T,norm="rmsnorm",mlp="swiglu",pos="rope",bias=False,attn_impl="sdpa",sdpa_backend="math")
L=gpt.build_layers(g,0)
cfg=LlamaConfig(hidden_size=d,intermediate_size=ff,num_hidden_layers=nl,num_attention_heads=nh,num_key_value_heads=kv,vocab_size=V,max_position_embeddings=T)
cfg._attn_implementation="sdpa"
torch.manual_seed(0); m=LlamaForCausalLM(cfg).train(); wgrad.convert(m)
x=torch.randint(0,V,(mb,T)); t=torch.randint(0,V,(mb,T))
def nat():
    h=x
    for l in L: h=l(h)
    F.cross_entropy(h.reshape(-1,V),t.reshape(-1)).backward()
def hfm():
    lg=m(input_ids=x,use_cache=False).logits.float()
    F.cross_entropy(lg.reshape(-1,V),t.reshape(-1)).backward()
for name,f in (("native",nat),("hf",hfm),("native",nat),("hf",hfm)):
    f(); ts=[]
    for _ in range(5):
        s=time.perf_counter(); f(); ts.append(time.perf_counter()-s)
    ts.sort(); print(name,"F+B median s: %.3f"%ts[2])
