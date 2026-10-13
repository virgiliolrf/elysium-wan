import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
# ADR-0009 derivation D-overhead: host dispatch time per decoder layer (F+B) at a shape where kernels are ~free,
# HF LlamaForCausalLM (eager) with SplitLinear vs the native arch.Block stack. Local Mac CPU, 1 thread.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import sys, os, time, warnings;  os.environ["USE_HUB_KERNELS"] = "0"; warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F, transformers; transformers.logging.set_verbosity_error()
from transformers import LlamaConfig, LlamaForCausalLM
from wantrain.model import gpt, wgrad
torch.set_num_threads(1)
d, ff, nh, kv, V, T = 64, 128, 4, 2, 128, 8
def bench(nl):
    g = gpt.GPTConfig(vocab_size=V, n_layer=nl, n_head=nh, n_kv_head=kv, d_model=d, d_ff=ff, seq_len=T, norm="rmsnorm", mlp="swiglu", pos="rope", bias=False)
    L = gpt.build_layers(g, 0)
    cfg = LlamaConfig(hidden_size=d, intermediate_size=ff, num_hidden_layers=nl, num_attention_heads=nh, num_key_value_heads=kv, vocab_size=V, max_position_embeddings=T)
    cfg._attn_implementation = "eager"
    m = LlamaForCausalLM(cfg).train(); wgrad.convert(m)
    x = torch.randint(0, V, (1, T)); t = torch.randint(0, V, (1, T))
    def nat():
        h = x
        for l in L: h = l(h)
        F.cross_entropy(h.reshape(-1, V), t.reshape(-1)).backward()
    def hfm():
        F.cross_entropy(m(input_ids=x, use_cache=False).logits.float().reshape(-1, V), t.reshape(-1)).backward()
    out = {}
    for name, f in (("native", nat), ("hf", hfm)):
        for _ in range(5): f()
        ts = []
        for _ in range(41):
            s = time.perf_counter(); f(); ts.append(time.perf_counter() - s)
        ts.sort(); out[name] = ts[20]
    return out
a, b = bench(2), bench(10)
for k in ("native", "hf"):
    print(f"{k:7s} per-layer F+B host time {(b[k]-a[k])/8*1e6:7.0f} us   (2 layers {a[k]*1e3:.2f} ms, 10 layers {b[k]*1e3:.2f} ms)")
