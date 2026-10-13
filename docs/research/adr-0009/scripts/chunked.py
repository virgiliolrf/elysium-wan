import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
# Derivation: a generic query-chunked wrapper around each family's OWN eager attention function, registered in
# transformers' AttentionInterface, with per-chunk recompute. Forward/grad bits vs plain eager.
import sys, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())

import stub_inject as si
import torch, torch.nn.functional as F, torch.utils.checkpoint as ckpt
from transformers import AutoConfig, AutoModelForCausalLM, AttentionInterface
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
CH = {"C": 8}
def wt_chunked(module, query, key, value, attention_mask, **kw):
    fam_eager = sys.modules[type(module).__module__].eager_attention_forward
    S = query.shape[2]; C = CH["C"]; outs = []
    for i in range(0, S, C):
        m = None if attention_mask is None else attention_mask[:, :, i:i + C]
        f = lambda q, k, v, m=m: fam_eager(module, q, k, v, m, **kw)[0]
        outs.append(ckpt.checkpoint(f, query[:, :, i:i + C], key, value, use_reentrant=False))
    return torch.cat(outs, dim=1), None   # eager returns (b, S, h, d) after transpose
AttentionInterface.register("wt_chunked", wt_chunked)
from transformers.masking_utils import AttentionMaskInterface, eager_mask
AttentionMaskInterface.register("wt_chunked", eager_mask)
ids = torch.randint(0, 128, (2, 24), generator=torch.Generator().manual_seed(1))
tgt = torch.randint(0, 128, (2, 24), generator=torch.Generator().manual_seed(2))
def run(fam, kw, attn, dtype):
    cfg = AutoConfig.for_model(fam, **kw); torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn, dtype=dtype); m.train()
    if attn == "wt_chunked": m.config._attn_implementation = "wt_chunked"
    lg = m(input_ids=ids, use_cache=False).logits
    F.cross_entropy(lg.float().reshape(-1, 128), tgt.reshape(-1)).backward()
    return lg, [p.grad for p in m.parameters()]
for fam in ("llama", "gemma2", "gpt_oss", "llama4_text", "deepseek_v3"):
    for dtype in (torch.float32, torch.bfloat16):
        l0, g0 = run(fam, si.fams[fam], "eager", dtype)
        for C in (24, 8):
            CH["C"] = C
            try:
                l1, g1 = run(fam, si.fams[fam], "wt_chunked", dtype)
                bad = sum(1 for a, b in zip(g0, g1) if not ((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))))
                print(f"{fam:12s} {str(dtype)[6:]:9s} C={C:2d} logits equal {torch.equal(l0, l1)} grads differ {bad}/{len(g0)}")
            except Exception as e:
                print(f"{fam:12s} {str(dtype)[6:]:9s} C={C:2d} ERROR {type(e).__name__}: {str(e)[:150]}")
