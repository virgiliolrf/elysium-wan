import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import os, json, collections, warnings, logging
os.environ["USE_HUB_KERNELS"]="0"
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
import transformers
transformers.logging.set_verbosity_error()
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import CONFIG_MAPPING
rows=[]
for mt, cls_name in sorted(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.items()):
    r={"mt":mt}
    try:
        cfg=CONFIG_MAPPING[mt]()
        tc=cfg.get_text_config() if hasattr(cfg,"get_text_config") else cfg
        for k in ("num_hidden_layers","n_layer","num_layers"):
            if hasattr(tc,k):
                try: setattr(tc,k,2)
                except Exception: pass
        if hasattr(tc,"layer_types") and tc.layer_types is not None:
            try: tc.layer_types=tc.layer_types[:2]
            except Exception: pass
        cls=getattr(transformers, cls_name if isinstance(cls_name,str) else cls_name[0])
        with torch.device("meta"):
            m=cls(cfg)
        leaf=collections.Counter()
        for name,mod in m.named_modules():
            if any(True for _ in mod.parameters(recurse=False)):
                leaf[type(mod).__name__ if type(mod) is not nn.Linear else "nn.Linear"]+=1
        nonpers=[n for n,mod in m.named_modules() for b in getattr(mod,"_non_persistent_buffers_set",()) for n in [f"{n}.{b}"]]
        exact_linear=sum(1 for mod in m.modules() if type(mod) is nn.Linear)
        sub_linear=sum(1 for mod in m.modules() if isinstance(mod,nn.Linear) and type(mod) is not nn.Linear)
        base=getattr(m,m.base_model_prefix,None)
        has_layers=hasattr(base,"layers") or hasattr(base,"h")
        r.update(ok=True, linear=exact_linear, linear_sub=sub_linear,
                 other={k:v for k,v in leaf.items() if k not in("nn.Linear",)},
                 nonpers=sorted({x.split(".")[-1] for x in nonpers}),
                 layers_attr="layers" if hasattr(base,"layers") else ("h" if hasattr(base,"h") else None),
                 tie=bool(getattr(tc,"tie_word_embeddings",False)),
                 layer_types=sorted(set(getattr(tc,"layer_types",None) or [])) )
        del m
    except Exception as e:
        r.update(ok=False, err=f"{type(e).__name__}: {str(e)[:100]}")
    rows.append(r)
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey.json"),"w"), indent=0)
ok=[r for r in rows if r["ok"]]
print("total",len(rows),"built",len(ok))
print("no_layers_attr",[r["mt"] for r in ok if not r["layers_attr"]][:60])
oth=collections.Counter()
for r in ok:
    for k in r["other"]: oth[k]+=1
print("param-holding non-Linear module classes (by #families):", oth.most_common(60))
print("families with Linear subclasses:",[r["mt"] for r in ok if r["linear_sub"]])
print("nonpersistent buffer names:",collections.Counter(b for r in ok for b in r["nonpers"]).most_common(30))
print("tied by default:",sum(r["tie"] for r in ok))
print("failed:",[(r["mt"],r["err"]) for r in rows if not r["ok"]][:30])
