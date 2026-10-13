"""Low-rank adapters, applied over whatever architecture was built.

On this engine LoRA is a memory knob first: full fine-tuning costs 16 bytes per
parameter (fp32 master weight, two Adam moments, gradient), while a frozen base
costs its weight alone and optimizer state exists only for the adapters. It buys
no bandwidth: a stage boundary carries activations, the same tensors whether the
weights behind them are frozen or not. Exactness: A is drawn under a seed derived
from the adapter's position, not a global RNG walk, so its init does not depend
on which stage built it; B is zero, so the adapted model is the base at step 0.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable rank-r correction.

    y = W x + (alpha / r) * B(A(x))

    The base layer is kept as a child module rather than copied, so the loaded
    checkpoint tensor is the one being used -- there is no second copy to drift
    or to pay for.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, seed: int):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.rank = rank
        self.scale = alpha / rank
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Linear(d_in, rank, bias=False)
        self.B = nn.Linear(rank, d_out, bias=False)

        # Positional seed, not the ambient RNG state: each stage builds the whole
        # model and keeps a slice, so an A drawn from a global walk would differ
        # by how many layers the process had already constructed.
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            w = torch.empty(rank, d_in)
            # Kaiming-uniform on A, the reference initialisation. B stays zero,
            # so the adapted model is the base model at step 0 and the first loss
            # is the pretrained loss rather than pretrained-plus-noise.
            bound = math.sqrt(6.0 / d_in) if d_in else 0.0
            w.uniform_(-bound, bound, generator=g)
            self.A.weight.copy_(w)
            self.B.weight.zero_()

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scale

    def extra_repr(self):
        return f"rank={self.rank}, scale={self.scale:.4g}"


# Which projections get adapters. The default is the attention projections, which
# is where the published ablations put the returns per trainable parameter; "all"
# adds the MLP, which costs several times more adapter state for a smaller
# marginal gain and is worth having as an option rather than a default.
TARGET_SETS = {
    "attn": ("qkv", "q", "kv", "proj"),
    "attn_mlp": ("qkv", "q", "kv", "proj", "gate", "up", "down", "fc1", "fc2"),
    "all": None,  # every nn.Linear, including the output head
}


def apply(layers, rank: int, alpha: float, targets: str = "attn", seed: int = 1234) -> dict:
    """Wrap the selected Linear layers in place; freeze everything else.

    Returns a report with trainable vs frozen parameter counts and the adapted
    modules. The planner sizes the optimizer from the trainable count, so a
    wrong number is the difference between a fleet that fits and one that OOMs.
    """
    if targets not in TARGET_SETS:
        raise ValueError(f"lora_targets={targets!r} unknown; use one of {sorted(TARGET_SETS)}")
    wanted = TARGET_SETS[targets]

    # Freeze first, then unfreeze what the adapters own. Doing it in this order
    # means a Linear the wrapper missed stays frozen rather than silently
    # remaining trainable -- fail closed, so an unrecognised architecture costs
    # capability instead of blowing the memory budget.
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad_(False)

    adapted = []
    for li, layer in enumerate(layers):
        # Collect first: replacing modules while walking named_modules() mutates
        # the structure being iterated.
        hits = [
            (name, m)
            for name, m in layer.named_modules()
            if isinstance(m, nn.Linear) and (wanted is None or name.split(".")[-1] in wanted)
        ]
        for name, m in hits:
            parent = layer
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            # Seed from position, so it does not depend on build order.
            h = (li * 1_000_003 + sum(ord(c) * (i + 1) for i, c in enumerate(name))) % (2**31)
            setattr(parent, parts[-1], LoRALinear(m, rank, alpha, seed=(seed * 7919 + h) % (2**31)))
            adapted.append(f"{li}.{name}")

    if not adapted:
        raise ValueError(
            f"lora_targets={targets!r} matched no Linear layer in this model. "
            f"With every parameter frozen and no adapters, the run would have "
            f"nothing to train and the loss would sit flat while everything "
            f"else looked healthy."
        )

    trainable = sum(p.numel() for l in layers for p in l.parameters() if p.requires_grad)
    frozen = sum(p.numel() for l in layers for p in l.parameters() if not p.requires_grad)
    return dict(
        adapted=len(adapted),
        rank=rank,
        alpha=alpha,
        targets=targets,
        trainable_params=trainable,
        frozen_params=frozen,
        trainable_pct=round(100.0 * trainable / max(1, trainable + frozen), 4),
        sample=adapted[:6],
    )


def split_params(layers):
    """(trainable, frozen) parameter lists, in a fixed traversal order.

    Order matters beyond tidiness: the ordered gradient accumulator folds by
    index, and the optimizer's state is keyed positionally, so two stages that
    enumerate parameters differently would disagree about which gradient belongs
    to which weight.
    """
    train, froze = [], []
    for layer in layers:
        for p in layer.parameters():
            (train if p.requires_grad else froze).append(p)
    return train, froze
