import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import os, warnings; os.environ["USE_HUB_KERNELS"]="0"; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, transformers; transformers.logging.set_verbosity_error()
from transformers import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as MAP
rows=[]
for mt,cn in sorted(MAP.items()):
    try:
        cfg=CONFIG_MAPPING[mt]()
        with torch.device("meta"): m=getattr(transformers,cn)(cfg)
        P=sum(p.numel()*p.element_size() for p in m.parameters())
        B=[(n,tuple(b.shape),b.numel()*b.element_size()) for n,b in m.named_buffers()]
        tot=sum(x[2] for x in B)
        big=max(B,key=lambda x:x[2]) if B else None
        rows.append((tot,mt,P,big,len(B)))
    except Exception as e: pass
rows.sort(reverse=True)
for r in rows[:12]: print(f"{r[1]:28s} buffers {r[0]/2**20:9.1f} MiB (n={r[4]}) params {r[2]/2**30:7.2f} GiB largest {r[3]}")
print("families with buffers > 64 MiB:",[r[1] for r in rows if r[0]>64*2**20])
