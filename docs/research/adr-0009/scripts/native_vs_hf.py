import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import sys, os, json, warnings;  os.environ["USE_HUB_KERNELS"]="0"; warnings.filterwarnings("ignore")
import torch, transformers; transformers.logging.set_verbosity_error()
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
from wantrain.model import hf, gpt
torch.set_num_threads(1)
import tempfile; D=os.path.join(tempfile.mkdtemp(prefix="adr9-ck-"), "ck")  # checkpoints outside the repo
for name,C,M,extra in (("llama",LlamaConfig,LlamaForCausalLM,{}),("llama-yarn",LlamaConfig,LlamaForCausalLM,{"rope_parameters":{"rope_type":"llama3","rope_theta":500000.0,"factor":8.0,"low_freq_factor":1.0,"high_freq_factor":4.0,"original_max_position_embeddings":16}}),("qwen3",Qwen3Config,Qwen3ForCausalLM,{})):
    cfg=C(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,vocab_size=128,max_position_embeddings=64,head_dim=16,tie_word_embeddings=False,**extra)
    torch.manual_seed(0); m=M(cfg).eval()
    d=D+"-"+name; m.save_pretrained(d)
    hcfg=json.load(open(d+"/config.json"))
    try:
        fields=hf.config_from_hf(hcfg, seq_len=32)
        g=gpt.GPTConfig(arch="hf", hf_model=d, **fields)
        layers=hf._build_arch_layers if False else None
        L=hf.build_layers(g, 0)
        x=torch.randint(0,128,(2,32))
        with torch.no_grad():
            h=x
            for l in L: h=l(h)
            ref=m(input_ids=x).logits
        print(name,"native vs transformers: max|dlogit|=%.3e argmax agree=%.3f"%((h-ref).abs().max().item(),(h.argmax(-1)==ref.argmax(-1)).float().mean().item()))
    except Exception as e:
        print(name,"native path:",type(e).__name__,str(e)[:200])
