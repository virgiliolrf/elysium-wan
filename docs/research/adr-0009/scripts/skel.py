import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
# ADR-0009 derivation D-skel: a skeleton with parameters on meta and buffers real on the CPU.
# Q1: does building it draw from the global CPU generator?  Q2: do its non-persistent buffers equal a real
# CPU build's bit for bit (rotary inv_freq, Gemma3 per-layer-type inv_freq, embed_scale)?  Q3 (control): the
# torchtitan path (whole model on meta, then to_empty) leaves those buffers wrong.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import os, contextlib, hashlib, warnings; os.environ["USE_HUB_KERNELS"]="0"; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, transformers; transformers.logging.set_verbosity_error()
from transformers import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as MAP
torch.set_num_threads(1)

@contextlib.contextmanager
def params_on_meta():
    orig = nn.Module.register_parameter
    def reg(self, name, param):
        if param is not None and param.device.type != "meta":
            param = nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)
        return orig(self, name, param)
    nn.Module.register_parameter = reg
    try:
        yield
    finally:
        nn.Module.register_parameter = orig

def h(t): return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]
BASE = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            vocab_size=128, max_position_embeddings=64, head_dim=16)
CASES = {"llama": dict(rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                     "original_max_position_embeddings": 16}),
         "gemma3_text": dict(sliding_window=8, sliding_window_pattern=2, query_pre_attn_scalar=16),
         "gpt2": dict(n_embd=64, n_layer=4, n_head=4, n_positions=64),
         "qwen3": dict(), "phi3": dict(partial_rotary_factor=0.5, pad_token_id=0)}
for mt, kw in CASES.items():
    cfg = CONFIG_MAPPING[mt](**{**BASE, **kw}); cls = getattr(transformers, MAP[mt])
    torch.manual_seed(0); real = cls(cfg)
    torch.manual_seed(0); s0 = torch.get_rng_state().clone()
    with params_on_meta(): sk = cls(cfg)
    drew = not torch.equal(s0, torch.get_rng_state())
    nonp = [(n, b) for n, b in sk.named_buffers() if b.device.type != "meta"]
    rb = dict(real.named_buffers())
    eq = all(torch.equal(b, rb[n]) and b.dtype == rb[n].dtype for n, b in nonp)
    on_meta_params = all(p.device.type == "meta" for p in sk.parameters())
    with torch.device("meta"): tt = cls(cfg)
    tt = tt.to_empty(device="cpu")
    tb = dict(tt.named_buffers())
    bad_tt = [n for n, b in rb.items() if not torch.equal(tb[n], b)]
    print(f"{mt:12s} rng_drawn={drew} params_meta={on_meta_params} buffers={len(nonp)} equal_to_real={eq} "
          f"to_empty_wrong={len(bad_tt)}/{len(rb)} e.g. {bad_tt[:2]}")
