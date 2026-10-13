import os, sys  # added on commit (README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import ast, glob, re, sys, json, collections
root=sys.argv[1]
OK_SELF={"embed_tokens","rotary_emb","norm","layers","config","gradient_checkpointing","training","get_input_embeddings","has_sliding_layers","_update_causal_mask"}
OK_FN={"create_causal_mask","create_sliding_window_causal_mask","create_chunked_causal_mask","BaseModelOutputWithPast","MoeModelOutputWithPast","DynamicCache","ValueError","isinstance","enumerate","dict","torch.arange","len","range","get_seq_length","super","logger.warning_once","partial","copy","kwargs.get","getattr","hasattr","list","torch.zeros"}
cls_counts=collections.Counter(); rows={}
for f in sorted(glob.glob(f"{root}/transformers/models/*/modeling_*.py")):
    fam=f.split("/models/")[1].split("/")[0]; src=open(f).read()
    if not re.search(r"class \w+ForCausalLM\(", src): continue
    if "encoder_hidden_states" in src and "ForConditionalGeneration" in src: continue
    t=ast.parse(src)
    base=[n for n in t.body if isinstance(n,ast.ClassDef) and n.name.endswith("Model") and not n.name.endswith("PreTrainedModel") and any(isinstance(b,ast.FunctionDef) and b.name=="forward" for b in n.body) and "self.layers" in ast.unparse(n)]
    lm=[n for n in t.body if isinstance(n,ast.ClassDef) and n.name.endswith("ForCausalLM")]
    if not base or not lm: rows[fam]="no-base"; cls_counts["no-base"]+=1; continue
    fwd=[b for b in base[0].body if isinstance(b,ast.FunctionDef) and b.name=="forward"][0]
    selfs=set(); fns=set(); arith=False
    for n in ast.walk(fwd):
        if isinstance(n,ast.Call):
            s=ast.unparse(n.func)
            if s.startswith("self.") : selfs.add(s.split(".")[1].split("[")[0])
            else: fns.add(s)
        if isinstance(n,(ast.Assign,ast.AugAssign)):
            tg=ast.unparse(n.targets[0] if isinstance(n,ast.Assign) else n.target)
            if tg in ("hidden_states","inputs_embeds") and (isinstance(n,ast.AugAssign) or (isinstance(n.value,ast.BinOp) and isinstance(n.value.op,(ast.Mult,ast.Div)))): arith=True
    extra_self=selfs-OK_SELF
    lfwd=[b for b in lm[0].body if isinstance(b,ast.FunctionDef) and b.name=="forward"]
    lmglue=False
    if lfwd:
        for n in ast.walk(lfwd[0]):
            if isinstance(n,(ast.Assign,ast.AugAssign)):
                tg=ast.unparse(n.targets[0] if isinstance(n,ast.Assign) else n.target)
                if tg=="logits" and not (isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=="self.lm_head"): lmglue=True
    loop_generic = bool(re.search(r"for .* in (enumerate\()?self\.layers", ast.unparse(fwd)))
    k = "generic" if (not extra_self and not arith and not lmglue and loop_generic) else ("glue-only" if (not extra_self and loop_generic) else "custom")
    cls_counts[k]+=1; rows[fam]=dict(k=k, extra=sorted(extra_self), arith=arith, lmglue=lmglue)
for f in ("llama","gemma3","granite","qwen3","mistral","gpt_oss"): print(f, rows.get(f))
print(sys.argv[1].split("/")[-1], dict(cls_counts))
print("generic:", sorted(f for f,r in rows.items() if isinstance(r,dict) and r["k"]=="generic"))
print("glue-only:", sorted(f for f,r in rows.items() if isinstance(r,dict) and r["k"]=="glue-only"))
print("custom:", sorted((f, r["extra"]) for f,r in rows.items() if isinstance(r,dict) and r["k"]=="custom")[:60])
