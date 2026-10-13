import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
# Derivation (ADR-0009, exactness angle): can a pipeline stage run HF's OWN model.forward with the layers it
# does not own replaced by stubs (inject the received hidden state, exit with the produced one), and reproduce
# the single-device logits AND gradients bit for bit? Tiny random configs built from config classes, CPU.
import sys, os, traceback
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
print("transformers", transformers.__version__, "torch", torch.__version__)

class _Exit(Exception):
    def __init__(self, h): self.h = h

def find_layers(model):
    best = None
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and (best is None or len(m) > len(best[1])) and len(m) > 0:
            best = (name, m)
    return best

class Stub(nn.Module):
    def __init__(self, real, mode, box, wrap):
        super().__init__()
        for k, v in vars(real).items():
            if not k.startswith("_") and not isinstance(v, (nn.Module, torch.Tensor)) and not callable(v):
                setattr(self, k, v)
        self.mode, self.box, self.wrap = mode, box, wrap
    def forward(self, hidden_states, *a, **k):
        if self.mode == "exit":
            raise _Exit(hidden_states)
        h = self.box["inject"] if self.mode == "inject" else hidden_states
        return (h,) if self.wrap else h

def run_stage(model, ids, a, b, inject=None):
    lname, layers = find_layers(model)
    n = len(layers)
    real = list(layers)
    wrap = model._wt_wrap
    box = {"inject": inject}
    for i in range(n):
        if i < a:
            layers[i] = Stub(real[i], "inject" if i == a - 1 else "pass", box, wrap)
        elif i >= b:
            layers[i] = Stub(real[i], "exit", box, wrap) if i == b else real[i]
    try:
        out = model(input_ids=ids, use_cache=False).logits
        kind = "logits"
    except _Exit as e:
        out, kind = e.h, "hidden"
    finally:
        for i in range(n): layers[i] = real[i]
    return out, kind

def check(fam, cfg_kwargs, dtype, attn, cuts):
    cfg = AutoConfig.for_model(fam, **cfg_kwargs)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn, dtype=dtype)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.p = 0.0
    lname, layers = find_layers(model)
    n = len(layers)
    # does a layer return a tuple?
    got = {}
    hk = layers[0].register_forward_hook(lambda m, i, o: (got.setdefault("t", isinstance(o, tuple)), None)[1])
    S = cfg_kwargs.get("_S", 24)
    g = torch.Generator().manual_seed(1)
    V = cfg.vocab_size
    ids = torch.randint(0, V, (2, S), generator=g); tgt = torch.randint(0, V, (2, S), generator=g)
    params = [p for p in model.parameters() if p.requires_grad]
    # reference: one device, whole forward/backward
    model.zero_grad(set_to_none=True)
    ref_logits = model(input_ids=ids, use_cache=False).logits
    hk.remove(); model._wt_wrap = got["t"]
    loss = F.cross_entropy(ref_logits.float().reshape(-1, V), tgt.reshape(-1))
    loss.backward()
    ref_g = [None if p.grad is None else p.grad.clone() for p in params]
    # pipeline: stages [0,c1),[c1,c2),...,[ck,n)
    bounds = [0] + cuts + [n]
    model.zero_grad(set_to_none=True)
    outs, ins = [], []
    h = None
    for s in range(len(bounds) - 1):
        a, b = bounds[s], bounds[s + 1]
        hin = None if h is None else h.detach().clone().requires_grad_(True)
        out, kind = run_stage(model, ids, a, b, hin)
        ins.append(hin); outs.append(out); h = out
    logits = outs[-1]
    loss2 = F.cross_entropy(logits.float().reshape(-1, V), tgt.reshape(-1))
    gr = torch.autograd.grad(loss2, [ins[-1]] if ins[-1] is not None else [], retain_graph=True) if ins[-1] is not None else None
    # backward stage by stage, last to first, accumulating param grads in .grad
    loss2.backward()
    for s in range(len(outs) - 2, -1, -1):
        gh = ins[s + 1].grad
        torch.autograd.backward(outs[s], gh)
    pipe_g = [None if p.grad is None else p.grad.clone() for p in params]
    same_logits = torch.equal(ref_logits, logits)
    same_loss = loss.item() == loss2.item() and torch.equal(loss, loss2)
    nbad = sum(1 for x, y in zip(ref_g, pipe_g) if not ((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))))
    tied = model.get_output_embeddings() is not None and model.get_output_embeddings().weight is model.get_input_embeddings().weight
    return dict(logits=same_logits, loss=same_loss, grads_differ=nbad, nparams=len(params), tied=tied, wrap=model._wt_wrap)

base = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64)
fams = {
  "llama": dict(base),
  "llama-yarn": ("llama", dict(base, rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16})),
  "llama-llama3": ("llama", dict(base, rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 16})),
  "mistral": dict(base, sliding_window=8),
  "qwen2": dict(base),
  "qwen3": dict(base, head_dim=32),
  "gemma": dict(base, head_dim=16),
  "gemma2": dict(base, head_dim=16, sliding_window=8, attn_logit_softcapping=50.0, final_logit_softcapping=30.0),
  "gemma3_text": dict(base, head_dim=16, sliding_window=8, sliding_window_pattern=2),
  "phi3": dict(base, partial_rotary_factor=0.5, pad_token_id=0),
  "olmo2": dict(base),
  "starcoder2": dict(base, sliding_window=8),
  "granite": dict(base),
  "gpt_neox": dict(base, rotary_pct=0.25),
  "gpt2": dict(vocab_size=128, n_embd=64, n_layer=4, n_head=4, n_positions=64),
  "gpt_oss": dict(base, head_dim=16, num_local_experts=4, num_experts_per_tok=2, sliding_window=8),
  "llama4_text": dict(base, head_dim=16, num_local_experts=2, num_experts_per_tok=1, intermediate_size_mlp=128, no_rope_layers=[1,1,1,0], attention_chunk_size=8, interleave_moe_layer_step=2),
  "deepseek_v3": dict(base, num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16, n_routed_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1, first_k_dense_replace=1, moe_intermediate_size=32, n_shared_experts=1),
}
def main():
  only = sys.argv[1:] or list(fams)
  for name in only:
      spec = fams[name]
      fam, kw = (spec if isinstance(spec, tuple) else (name, spec))
      for dtype in (torch.float32, torch.bfloat16):
          for attn in ("eager", "sdpa"):
              try:
                  r = check(fam, kw, dtype, attn, [1, 3])
                  print(f"{name:14s} {str(dtype)[6:]:9s} {attn:6s} {r}")
              except Exception as e:
                  print(f"{name:14s} {str(dtype)[6:]:9s} {attn:6s} ERROR {type(e).__name__}: {str(e)[:160]}")

if __name__ == "__main__":
  main()
