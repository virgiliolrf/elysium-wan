import os, sys  # added on commit (README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
# ADR-0009 round 1 (census3): census.py over proto2.py (NaN stubs, RNG refusal, direct-read Linears unconverted).
# ADR-0009 derivation D-census: every causal-LM model type of the installed transformers, default config shrunk by a
# key table (dimensions only; every flag kept), through the D-proto stage mechanism (one unit per stage, M 2, fp32,
# eager), compared bit for bit with the vanilla HF model. Outcome per family: EXACT, DIFFER, or the refusal/error.
import sys, os, json, signal
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto2.py")).read().split("if __name__ != '__main__'")[0]
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
if not ONLY:
    # The whole census runs 3 model types per subprocess, so each batch's memory is freed. Every type in one process
    # grew past 5.7 GB and drove the 16 GB Mac to 13.9 GB of swap and 1.2 GB of free disk (the orchestrator, 2026-10-11
    # 00:04). Same per-type lines, same census3.json.
    import subprocess
    types = sorted(MAP)
    for i in range(0, len(types), 3):
        batch = types[i:i + 3]
        try:
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), *batch], capture_output=True, text=True, timeout=600)
            out, why = p.stdout, f"ERR subprocess exit {p.returncode}"
        except subprocess.TimeoutExpired as e:
            out, why = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")), "TIMEOUT (batch)"
        got = {l.split()[0]: l[30:].strip() for l in out.splitlines() if l.split() and l.split()[0] in batch}
        for mt in batch:
            res[mt] = got.get(mt, why)
            print(f"{mt:30s} {res[mt]}", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "census3.json"), "w"), indent=1)
    sys.exit(0)
class TooBig(Exception): pass
for mt in ONLY:
    try:
        signal.alarm(20)
        cfg = shrunk(mt)
        # Count parameters on the meta device first. The shrink table misses nested sub-configs (gemma4_unified's
        # towers kept their default sizes and allocated 17-20 GB on the 16 GB Mac, 2026-10-11 00:06), so a type still
        # large after shrinking is recorded, never built.
        with torch.device("meta"): _m = AutoModelForCausalLM.from_config(cfg)
        _n = sum(p.numel() for p in _m.parameters()); del _m
        if _n > 20_000_000: raise TooBig(_n)
        r = check(mt, None, torch.float32, cfg=cfg)
        res[mt] = ("EXACT" if (r["logits"] and r["differ"] == 0) else f"DIFFER logits={r['logits']} grads={r['differ']}/{r['grads']} first={r['first_bad']}") + (f" unconverted={r['direct']}" if r["direct"] else "")
    except TO: res[mt] = "TIMEOUT"
    except TooBig as e: res[mt] = f"SKIP {e.args[0] / 1e6:.0f}M params after shrinking (a sub-config the shrink table misses)"
    except BaseException as e:
        res[mt] = f"ERR {type(e).__name__}: {str(e)[:90]}"
        if os.environ.get("TB"): print("\n".join(traceback.format_exc().splitlines()[-6:]))
    finally: signal.alarm(0)
    print(f"{mt:30s} {res[mt]}", flush=True)
if not ONLY: json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "census3.json"), "w"), indent=1)
