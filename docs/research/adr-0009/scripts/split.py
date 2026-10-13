import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import sys, os, copy, warnings, json

os.environ["USE_HUB_KERNELS"]="0"; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
import transformers; transformers.logging.set_verbosity_error()
from transformers import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as MAP
from wantrain.model import wgrad
torch.use_deterministic_algorithms(True); torch.set_num_threads(1)

BASE=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
          num_key_value_heads=2, vocab_size=128, max_position_embeddings=64, head_dim=16,
          rms_norm_eps=1e-6, attention_dropout=0.0, hidden_dropout=0.0, sliding_window=6)
FAM={
 "llama":{}, "mistral":{}, "qwen2":{}, "qwen3":{}, "gemma":{}, "gemma2":{"query_pre_attn_scalar":16,"final_logit_softcapping":30.0,"attn_logit_softcapping":50.0},
 "gemma3_text":{"query_pre_attn_scalar":16,"sliding_window_pattern":2}, "phi3":{"pad_token_id":0}, "olmo":{}, "olmo2":{}, "starcoder2":{"residual_dropout":0.0,"embedding_dropout":0.0},
 "granite":{"embedding_multiplier":12.0,"logits_scaling":8.0,"residual_multiplier":0.2,"attention_multiplier":0.25},
 "cohere":{"logit_scale":0.5}, "cohere2":{}, "llama4_text":{"no_rope_layers":[1,1,0,1],"attention_chunk_size":8,"num_local_experts":2,"interleave_moe_layer_step":0,"moe_layers":[], "intermediate_size_mlp":128},
 "gpt2":{"n_embd":64,"n_layer":4,"n_head":4,"n_positions":64,"resid_pdrop":0.0,"embd_pdrop":0.0,"attn_pdrop":0.0},
 "gpt_neox":{"hidden_dropout":0.0,"rotary_pct":0.5}, "phi":{"partial_rotary_factor":0.5,"resid_pdrop":0.0,"embd_pdrop":0.0},
 "deepseek_v3":{"kv_lora_rank":16,"q_lora_rank":32,"qk_rope_head_dim":8,"qk_nope_head_dim":8,"v_head_dim":16,"n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,"first_k_dense_replace":4,"n_group":1,"topk_group":1},
 "gpt_oss":{"num_local_experts":4,"num_experts_per_tok":2,"layer_types":["sliding_attention","full_attention"]*2},
 "qwen3_moe":{"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,"decoder_sparse_step":1},
}
def build(mt, attn, extra_tie=None):
    kw=dict(BASE); kw.update(FAM[mt])
    if extra_tie is not None: kw["tie_word_embeddings"]=extra_tie
    cfg=CONFIG_MAPPING[mt](**kw)
    cfg._attn_implementation=attn
    for k in ("rope_scaling",): pass
    torch.manual_seed(0)
    cls=getattr(transformers, MAP[mt]); m=cls(cfg).train()
    return m, cfg

def layers_of(m):
    n=m.config.num_hidden_layers
    best=None
    for name,mod in m.named_modules():
        if isinstance(mod,nn.ModuleList) and len(mod)==n and best is None: best=(name,mod)
    return best

class Stop(Exception): pass
class Rec(nn.Module):
    def __init__(s,i,store,inject=None,stop=False): super().__init__(); s.i=i; s.store=store; s.inject=inject; s.stop=stop
    def forward(s,h,*a,**k):
        s.store[s.i]=(h,a,k)
        if s.stop: raise Stop
        return s.inject if s.inject is not None else h

def capture(m, ids, ret_embed=False, inject=None, zeros=False):
    """Run HF's own forward with recorder layers. -> (store, out)"""
    name,L=layers_of(m); orig=list(L); store={}
    for i in range(len(L)): L[i]=Rec(i,store, inject=inject if i==0 else None, stop=ret_embed and i==0)
    try:
        if zeros:
            emb=m.get_input_embeddings()
            x=torch.zeros(ids.shape+(m.config.get_text_config().hidden_size,), dtype=emb.weight.dtype)
            out=m(inputs_embeds=x, use_cache=False)
        else:
            out=m(input_ids=ids, use_cache=False)
    except Stop: out=None
    finally:
        for i in range(len(L)): L[i]=orig[i]
    return store,out

def run(mt, attn, M=3, tie=None):
    m,cfg=build(mt,attn,tie)
    g=torch.Generator().manual_seed(1)
    V=m.get_input_embeddings().weight.shape[0]
    mbs=[torch.randint(0,V,(2,16),generator=g) for _ in range(M)]
    tg=[torch.randint(0,V,(2,16),generator=g) for _ in range(M)]
    P=[p for p in m.parameters() if p.requires_grad]
    # reference: stock HF forward per microbatch, fixed-order fold over microbatches
    ref=None; ref_logits=[]
    for x,t in zip(mbs,tg):
        lg=m(input_ids=x,use_cache=False).logits.float(); ref_logits.append(lg.detach())
        gs=torch.autograd.grad(F.cross_entropy(lg.reshape(-1,lg.size(-1)),t.reshape(-1)),P,allow_unused=True)
        gs=[torch.zeros_like(p) if q is None else q for p,q in zip(P,gs)]
        ref=gs if ref is None else [a+b for a,b in zip(ref,gs)]
    # adapter: same model object, pieces from HF's own forward, SplitLinear surgery, 2 "stages" cut at layer 2
    a=copy.deepcopy(m); nconv=wgrad.convert(a)
    PA=[p for p in a.parameters() if p.requires_grad]
    _,L=layers_of(a); n=len(L)
    acc=None; same_kw=True; lg_ok=True; tied_split=None
    for x,t,rl in zip(mbs,tg,ref_logits):
        st,_=capture(a,x)
        st0,_=capture(a,x,zeros=True)   # what a non-first stage sees (no ids, no embeddings)
        for i in range(n):
            for u,v in zip(list(st[i][1])+list(st[i][2].values()), list(st0[i][1])+list(st0[i][2].values())):
                if isinstance(u,torch.Tensor) and (u.dtype!=v.dtype or not torch.equal(u,v)): same_kw=False
                if isinstance(u,dict):
                    for kk in u:
                        if isinstance(u[kk],torch.Tensor) and not torch.equal(u[kk],v[kk]): same_kw=False
        # stage 0: embed piece + layers 0,1
        with wgrad.capture() as store:
            e,_=capture(a,x,ret_embed=True); h=e[0][0]
            for i in range(2):
                h=L[i](h,*st0[i][1],**st0[i][2]); h=h[0] if isinstance(h,tuple) else h
            b0=h.detach().requires_grad_(True)
            # stage 1: layers 2,3 + head piece
            h1=b0
            for i in range(2,n):
                h1=L[i](h1,*st0[i][1],**st0[i][2]); h1=h1[0] if isinstance(h1,tuple) else h1
            _,out=capture(a,x,inject=h1,zeros=True)
            lg=out.logits.float()
            if not torch.equal(lg,rl): lg_ok=False
            loss=F.cross_entropy(lg.reshape(-1,lg.size(-1)),t.reshape(-1))
            others=[p for p in PA if id(p) not in wgrad.owned_params(a)]
            g1=torch.autograd.grad(loss,[b0]+others,allow_unused=True)
            gb=g1[0]; go1={id(p):q for p,q in zip(others,g1[1:])}
            g0=torch.autograd.grad(h,others,grad_outputs=gb,allow_unused=True)
            go0={id(p):q for p,q in zip(others,g0)}
        dws=wgrad.flush(store)
        gs=[]
        for p in PA:
            parts=[d[id(p)] for d in (go1,go0) if d.get(id(p)) is not None]  # head-side first, then embed-side
            q=dws.get(id(p))
            if q is None: q=parts[0] if len(parts)==1 else (parts[0]+parts[1] if parts else torch.zeros_like(p))
            gs.append(q)
        acc=gs if acc is None else [u+v for u,v in zip(acc,gs)]
    bad=[i for i,(u,v) in enumerate(zip(ref,acc)) if not torch.equal(u,v)]
    tied = a.get_output_embeddings() is not None and a.get_output_embeddings().weight is a.get_input_embeddings().weight
    return dict(mt=mt,attn=attn,converted=nconv,params=len(PA),logits_equal=lg_ok,grads_equal=not bad,bad=len(bad),kwargs_data_independent=same_kw,tied=tied)

res=[]
for mt in FAM:
    for attn in ("eager","sdpa"):
        try: r=run(mt,attn)
        except Exception as e: r=dict(mt=mt,attn=attn,err=f"{type(e).__name__}: {str(e)[:160]}")
        print(json.dumps(r)); res.append(r)
