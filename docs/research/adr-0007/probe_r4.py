"""ADR-0007 round 4 probe: does `Module.to_empty` keep parameter sharing?

Run: .venv/bin/python -I docs/research/adr-0007/probe_r4.py
Prints, for a tied Embedding/Linear pair built on the meta device:
  - whether the pair shares one Parameter on meta, after `to_empty`, and after a re-tie
    from a sharing map taken on the skeleton before `to_empty`;
  - the parameter count after each step;
  - whether `load_state_dict(strict=True)` of a tied state accepts the re-tied module and
    leaves one tensor holding the loaded bytes.
Also checks the product modules hold no buffers (non-persistent state is what a draw-free
resume could not restore).
"""
import hashlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")


def sharing_map(mod):
    """canonical name -> first canonical name of the same Parameter object (remove_duplicate=False)."""
    first, out = {}, {}
    for name, p in mod.named_parameters(remove_duplicate=False):
        out[name] = first.setdefault(id(p), name)
    return out


def retie(mod, smap):
    params = dict(mod.named_parameters(remove_duplicate=False))
    for name, src in smap.items():
        if name != src:
            *path, leaf = name.split(".")
            owner = mod
            for part in path:
                owner = getattr(owner, part) if not part.isdigit() else owner[int(part)]
            setattr(owner, leaf, params[src])


print("torch", torch.__version__)
with torch.device("meta"):
    e = nn.Embedding(10, 4)
    h = nn.Linear(4, 10, bias=False)
    h.weight = e.weight
m = nn.ModuleList([e, h])
smap = sharing_map(m)
print("meta: tied", m[0].weight is m[1].weight, "params", len(list(m.parameters())))
m.to_empty(device="cpu")
print("to_empty: tied", m[0].weight is m[1].weight, "params", len(list(m.parameters())))
retie(m, smap)
print("re-tied: tied", m[0].weight is m[1].weight, "params", len(list(m.parameters())))

torch.manual_seed(0)
ref = nn.ModuleList([nn.Embedding(10, 4), nn.Linear(4, 10, bias=False)])
ref[1].weight = ref[0].weight
m.load_state_dict(ref.state_dict(), strict=True)
same = hashlib.sha256(m[1].weight.detach().numpy().tobytes()).hexdigest() == hashlib.sha256(
    ref[1].weight.detach().numpy().tobytes()
).hexdigest()
print("load strict ok; head bytes equal ref", same, "still tied", m[0].weight is m[1].weight)

from wantrain.model.gpt import GPTConfig, build_layers  # noqa: E402

for name, kw in (
    ("builtin", {}),
    ("general rope swiglu rmsnorm gqa tied gpt2", dict(pos="rope", mlp="swiglu", norm="rmsnorm", n_kv_head=1,
                                                       tie_embeddings=True, init_scheme="gpt2")),
):
    layers = build_layers(GPTConfig(**kw), 0)
    nb = sum(1 for _ in layers.named_buffers())
    print("buffers in", name, nb, "params", len(list(layers.parameters())))
