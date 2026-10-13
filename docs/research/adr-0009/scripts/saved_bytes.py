import os, sys  # ADR-0009 round 2: bytes saved for backward by one eager decoder layer, per b*H*S^2 (saved_tensors_hooks)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
os.environ["USE_HUB_KERNELS"] = "0"; os.environ["HF_HUB_OFFLINE"] = "1"
import torch, warnings; warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
from transformers import LlamaConfig, LlamaForCausalLM, Gemma2Config, Gemma2ForCausalLM
torch.manual_seed(0); torch.set_num_threads(1)
def run(kind, H, S, autocast):
    common = dict(hidden_size=8*H, intermediate_size=16, num_hidden_layers=1, num_attention_heads=H,
                  num_key_value_heads=H, vocab_size=64, max_position_embeddings=S, head_dim=8)
    if kind == "llama": m = LlamaForCausalLM(LlamaConfig(**common))
    else: m = Gemma2ForCausalLM(Gemma2Config(**common, attn_logit_softcapping=50.0, final_logit_softcapping=30.0, sliding_window=S, query_pre_attn_scalar=8))
    m.config._attn_implementation = "eager"; m.train()
    layer = m.model.layers[0]
    tot = {"all": 0, "S2": 0}
    def pack(t):
        n = t.numel() * t.element_size()
        tot["all"] += n
        if t.dim() >= 2 and t.shape[-1] == S and t.shape[-2] == S and t.dim() == 4 and t.shape[1] == H: tot["S2"] += n
        return t
    ids = torch.randint(0, 64, (1, S))
    hook_on = {"v": False}
    def pre(mod, args, kwargs): hook_on["v"] = True
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else torch.autocast("cpu", enabled=False)
    with ctx:
        x = m.model.embed_tokens(ids)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = m.model(inputs_embeds=x, use_cache=False).last_hidden_state
    return tot["S2"] / (H * S * S), tot["all"] / (H * S * S)
for kind in ("llama", "gemma2"):
    for ac in (False, True):
        for H, S in ((4, 256), (4, 512)):
            s2, al = run(kind, H, S, ac)
            print(f"{kind:7s} autocast={'bf16' if ac else 'off '} H {H} S {S}: saved H*S^2-shaped bytes per element {s2:.3f}, all saved bytes per H*S^2 {al:.3f}", flush=True)
print("torch", torch.__version__, "transformers", transformers.__version__)
