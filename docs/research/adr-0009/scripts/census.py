import os, sys  # added on commit (README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
# ADR-0009 derivation D-census: every causal-LM model type of the installed transformers, default config shrunk by a
# key table (dimensions only; every flag kept), through the D-proto stage mechanism (one unit per stage, M 2, fp32,
# eager), compared bit for bit with the vanilla HF model. Outcome per family: EXACT, DIFFER, or the refusal/error.
import sys, os, json, signal
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto.py")).read().split("if __name__ != '__main__'")[0]
exec(SRC)
SHRINK = dict(hidden_size=64, n_embd=64, d_model=64, intermediate_size=128, ffn_hidden_size=128, n_inner=128,
              moe_intermediate_size=32, intermediate_size_mlp=128, shared_expert_intermediate_size=32,
              num_attention_heads=4, n_head=4, num_heads=4, head_dim=16, vocab_size=128, num_hidden_layers=4,
              n_layer=4, num_layers=4, max_position_embeddings=64, n_positions=64, num_local_experts=4, num_experts=4,
              n_routed_experts=4, num_experts_per_tok=2, sliding_window=8, attention_chunk_size=8, kv_lora_rank=16,
              q_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16, first_k_dense_replace=1,
              n_group=1, topk_group=1)
DROP = ("dropout", "pdrop", "drop_rate")
def shrunk(mt):
    d = CONFIG_MAPPING[mt]().to_dict()
    n0 = d.get("num_hidden_layers") or d.get("n_layer") or d.get("num_layers")
    out = {}
    for k, v in d.items():
        if k in SHRINK and isinstance(v, int) and not isinstance(v, bool) and v > 0: out[k] = SHRINK[k]
        elif isinstance(v, list) and n0 and len(v) == n0: out[k] = v[:4]
        elif isinstance(v, float) and any(x in k for x in DROP): out[k] = 0.0
    for k, v in d.items():
        if k.endswith("token_id") and isinstance(v, int) and v >= 128: out[k] = 1
        if k.endswith("token_id") and isinstance(v, list): out[k] = [1]
    if "head_dim" in d and d["head_dim"] is None: out["head_dim"] = 16
    if "qk_rope_head_dim" in d: out["head_dim"] = 8
    if "num_key_value_heads" in d and d["num_key_value_heads"]:
        out["num_key_value_heads"] = 2 if d["num_key_value_heads"] < (d.get("num_attention_heads") or 1) else 4
    return CONFIG_MAPPING[mt](**{**d, **out})
class TO(Exception): pass
def alarm(*a): raise TO()
signal.signal(signal.SIGALRM, alarm)
res = {}
import traceback
ONLY = sys.argv[1:]
for mt in (ONLY or sorted(MAP)):
    try:
        signal.alarm(20)
        cfg = shrunk(mt)
        r = check(mt, None, torch.float32, cfg=cfg)
        res[mt] = "EXACT" if (r["logits"] and r["differ"] == 0) else f"DIFFER logits={r['logits']} grads={r['differ']}/{r['grads']}"
    except TO: res[mt] = "TIMEOUT"
    except BaseException as e:
        res[mt] = f"ERR {type(e).__name__}: {str(e)[:90]}"
        if os.environ.get("TB"): print("\n".join(traceback.format_exc().splitlines()[-6:]))
    finally: signal.alarm(0)
    print(f"{mt:30s} {res[mt]}", flush=True)
if not ONLY: json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "census.json"), "w"), indent=1)
