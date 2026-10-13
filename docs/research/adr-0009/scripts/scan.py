import os, sys  # added on commit (README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import ast, sys, os, re, json, glob
root = sys.argv[1]
out = {}
for f in sorted(glob.glob(f"{root}/transformers/models/*/modeling_*.py")):
    fam = f.split("/models/")[1].split("/")[0]
    src = open(f).read()
    if not re.search(r"class \w+ForCausalLM\(", src): continue
    try: tree = ast.parse(src)
    except Exception as e: continue
    d = dict(file=os.path.basename(f))
    d["encdec"] = bool(re.search(r"encoder_hidden_states", src)) and "ForConditionalGeneration" in src
    d["ccm"] = "create_causal_mask" in src
    d["swm"] = "create_sliding_window_causal_mask" in src
    d["chunked"] = "create_chunked_causal_mask" in src
    d["update_causal_mask"] = "_update_causal_mask" in src
    d["layer_types"] = "layer_types[" in src
    d["attn_iface"] = "ALL_ATTENTION_FUNCTIONS" in src
    d["conv1d"] = "Conv1D(" in src
    d["experts3d"] = "use_experts_implementation" in src or bool(re.search(r"nn\.Parameter\(torch\.empty\(\s*self\.num_experts", src))
    d["kernel_hub"] = "use_kernel_forward_from_hub" in src or "use_kernel_func_from_hub" in src
    d["pp_plan"] = "_pp_plan" in src
    d["recurrent"] = bool(re.search(r"mamba|GatedDeltaNet|linear_attention|selective_scan|causal_conv1d|ssm_state|recurrent_gemma|RWKV|xlstm", src, re.I))
    d["softcap"] = "final_logit_softcapping" in src
    d["logits_scaling"] = "logits_scaling" in src or "logit_scale" in src
    d["emb_mult"] = "embedding_multiplier" in src or "normalizer" in src
    d["rotate_half"] = "def rotate_half" in src
    d["interleaved_rope"] = bool(re.search(r"rotate_every_two|0::2\]|::2\]", src))
    # decoder layer forward signature
    sig = None
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name.endswith("DecoderLayer"):
            for b in n.body:
                if isinstance(b, ast.FunctionDef) and b.name == "forward":
                    args = [a.arg for a in b.args.args[1:]] + [a.arg for a in b.args.kwonlyargs]
                    ret = ast.unparse(b.returns) if b.returns else None
                    sig = (n.name, args, ret)
            break
    d["layer_sig"] = sig
    out[fam] = d
json.dump(out, open(sys.argv[2], "w"), indent=0, default=str)
print(len(out))
