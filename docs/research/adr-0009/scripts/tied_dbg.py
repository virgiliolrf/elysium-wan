import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import os, warnings; os.environ["USE_HUB_KERNELS"]="0"; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F, transformers; transformers.logging.set_verbosity_error()
from transformers import CONFIG_MAPPING, AutoModelForCausalLM
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
cfg = CONFIG_MAPPING["gpt2"](vocab_size=128, n_embd=64, n_layer=2, n_head=4, n_positions=64, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
cfg._attn_implementation = "eager"
torch.manual_seed(0); m = AutoModelForCausalLM.from_config(cfg).train()
g = torch.Generator().manual_seed(1); x = torch.randint(0, 128, (2, 24), generator=g); t = torch.randint(0, 128, (2, 24), generator=g)
W = m.transformer.wte.weight
loss = F.cross_entropy(m(input_ids=x, use_cache=False).logits.float().reshape(-1, 128), t.reshape(-1))
G, = torch.autograd.grad(loss, [W])
# split the two uses: the head reads W through a separate leaf alias
Wh = nn.Parameter(W.detach().clone()); We = nn.Parameter(W.detach().clone())
m.lm_head.weight = Wh; m.transformer.wte.weight = We
loss2 = F.cross_entropy(m(input_ids=x, use_cache=False).logits.float().reshape(-1, 128), t.reshape(-1))
h, e = torch.autograd.grad(loss2, [Wh, We])
print("loss equal", torch.equal(loss, loss2))
print("G == h+e", torch.equal(G, h + e), " G == e+h", torch.equal(G, e + h))
d = (G != h + e); print("differing elements", d.sum().item(), "of", G.numel())
rows_touched = torch.zeros(128, dtype=torch.bool); rows_touched[x.unique()] = True
print("differing rows touched by the batch:", d.any(1)[rows_touched].sum().item(), "untouched:", d.any(1)[~rows_touched].sum().item())
idx = d.nonzero()[:3]
for r, c in idx.tolist(): print(r, c, G[r, c].item(), h[r, c].item(), e[r, c].item(), (h + e)[r, c].item())
