import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
# Derivation: today's native path (model/hf.py + arch.py) vs transformers on the same tiny random checkpoint.
import sys, os, json
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from wantrain.model import hf as whf
from wantrain.model.gpt import GPTConfig
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_llama")
for name, extra in (("plain", {}), ("llama3-rope", {"rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 16}})):
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, rms_norm_eps=1e-6, **extra)
    torch.manual_seed(0); m = LlamaForCausalLM(cfg).eval(); m.config._attn_implementation = "eager"
    m.save_pretrained(d, safe_serialization=True)
    hcfg = json.load(open(os.path.join(d, "config.json")))
    fields = whf.config_from_hf(hcfg, seq_len=24)
    gc = GPTConfig(arch="hf", hf_model=d, **fields)
    layers = whf.build_layers(gc, 0)
    ids = torch.randint(0, 128, (2, 24), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        x = ids
        for l in layers: x = l(x)
        ref = m(input_ids=ids, use_cache=False).logits
    print(f"{name}: native vs transformers logits equal {torch.equal(x, ref)}  max|d| {(x - ref).abs().max().item():.3e}  max|ref| {ref.abs().max().item():.3e}")
