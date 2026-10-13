"""Generalized decoder built as a flat list of layers.

The engine assigns contiguous slices of a flat layer list to stages, so every
architecture reduces to that shape: token ids -> hidden states first, hidden
-> logits last, hidden -> hidden in between. The configuration is a superset
of the original GPT block and the defaults reproduce it bit for bit; the knobs
give norm="rmsnorm", mlp="swiglu", pos="rope", n_kv_head < n_head (GQA),
bias=False and tie_embeddings. Every process builds the full layer list under
the same seed and keeps only its slice, so init is identical everywhere.
"""

import math
from types import SimpleNamespace
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- norms & mlps


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # accumulate the reduction in fp32 even under a narrow compute dtype
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps)).to(x.dtype) * self.weight


def make_norm(kind, d):
    return nn.LayerNorm(d) if kind == "layernorm" else RMSNorm(d)


class GeluMLP(nn.Module):
    def __init__(self, d, d_ff, bias):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d, bias=bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SwiGLUMLP(nn.Module):
    def __init__(self, d, d_ff, bias):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=bias)
        self.up = nn.Linear(d, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def make_mlp(kind, d, d_ff, bias):
    return (GeluMLP if kind == "gelu" else SwiGLUMLP)(d, d_ff, bias)


def default_d_ff(kind, d):
    # SwiGLU has three matrices instead of two, so the conventional width is
    # 8/3*d rounded to a multiple of 256 to keep the parameter count comparable
    if kind == "gelu":
        return 4 * d
    return int(math.ceil(8 * d / 3 / 256) * 256)


# ------------------------------------------------------------------------ rope


def rope_cache(t, head_dim, theta, device, dtype):
    inv = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(t, device=device, dtype=torch.float32)
    f = torch.outer(pos, inv)
    return torch.cos(f).to(dtype), torch.sin(f).to(dtype)


def apply_rope(x, cos, sin):
    """x: (b, heads, t, head_dim). Rotates adjacent dimension pairs."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


# ---------------------------------------------------------------------- layers


class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bnd = cfg.boundary_narrowing()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model) if cfg.pos == "learned" else None

    def forward(self, idx):
        from wantrain.model.gpt import _Boundary

        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        if self.cfg.autocast_enabled():
            x = x.to(self.cfg.compute_torch_dtype())
        return _Boundary.apply(x, *self.bnd)


class Block(nn.Module):
    def __init__(self, cfg, index: int = 0):
        super().__init__()
        self.cfg = cfg
        self.bnd = cfg.boundary_narrowing()
        d, nh = cfg.d_model, cfg.n_head
        self.nkv = cfg.n_kv_head or nh
        assert nh % self.nkv == 0, "n_head must be a multiple of n_kv_head"
        self.hd = d // nh
        d_ff = cfg.d_ff or default_d_ff(cfg.mlp, d)
        # Attribute creation order is deliberate and must not be reshuffled: it
        # fixes both the RNG draw order (so init matches the original block) and
        # the named_parameters() order that the fixed-index gradient accumulator
        # and the optimizer state rely on.
        self.ln1 = make_norm(cfg.norm, d)
        if self.nkv == nh:
            self.qkv = nn.Linear(d, 3 * d, bias=cfg.bias)
        else:
            self.q = nn.Linear(d, d, bias=cfg.bias)
            self.kv = nn.Linear(d, 2 * self.nkv * self.hd, bias=cfg.bias)
        self.proj = nn.Linear(d, d, bias=cfg.bias)
        self.ln2 = make_norm(cfg.norm, d)
        # index is the block's position in the flat list (Embed is 0), the
        # MoE block's moe_layer (I-11).
        if is_moe_layer(cfg, index):
            self.mlp = MoEMLP(cfg, index)
        else:
            self.mlp = make_mlp(cfg.mlp, d, d_ff, cfg.bias)

    def _qkv(self, h, b, t):
        if self.nkv == self.cfg.n_head:
            q, k, v = self.qkv(h).split(self.cfg.d_model, dim=2)
        else:
            q = self.q(h)
            k, v = self.kv(h).split(self.nkv * self.hd, dim=2)
        q = q.view(b, t, self.cfg.n_head, self.hd).transpose(1, 2)
        k = k.view(b, t, self.nkv, self.hd).transpose(1, 2)
        v = v.view(b, t, self.nkv, self.hd).transpose(1, 2)
        if self.nkv != self.cfg.n_head:
            rep = self.cfg.n_head // self.nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        return q, k, v

    def forward(self, x):
        from torch.nn.attention import sdpa_kernel

        from wantrain.model.gpt import _Boundary, resolve_sdpa_backend

        b, t, c = x.shape
        with torch.autocast(
            device_type=x.device.type,
            dtype=self.cfg.compute_torch_dtype(),
            enabled=self.cfg.autocast_enabled(),
        ):
            q, k, v = self._qkv(self.ln1(x), b, t)
            if self.cfg.pos == "rope":
                cos, sin = rope_cache(t, self.hd, self.cfg.rope_theta, x.device, q.dtype)
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            if self.cfg.attn_impl == "sdpa":
                with sdpa_kernel(
                    resolve_sdpa_backend(
                        self.cfg.sdpa_backend, x.device.type, self.cfg.compute_dtype
                    )
                ):
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / self.hd**0.5)
                mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
                att = F.softmax(att.masked_fill(mask, float("-inf")), dim=-1)
                y = att @ v
            y = y.transpose(1, 2).contiguous().view(b, t, c)
            x = x + self.proj(y).to(x.dtype)
            x = x + self.mlp(self.ln2(x)).to(x.dtype)
        return _Boundary.apply(x, *self.bnd)


class Head(nn.Module):
    def __init__(self, cfg, tok_embedding=None):
        super().__init__()
        self.cfg = cfg
        self.ln_f = make_norm(cfg.norm, cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            assert tok_embedding is not None, "tie_embeddings needs the Embed"
            self.out.weight = tok_embedding.weight

    def forward(self, x):
        from torch.nn.attention import sdpa_kernel  # noqa: F401  (parity)

        with torch.autocast(
            device_type=x.device.type,
            dtype=self.cfg.compute_torch_dtype(),
            enabled=self.cfg.autocast_enabled(),
        ):
            logits = self.out(self.ln_f(x))
        return logits.float()


def gpt2_draws(layer, cfg) -> list:
    """gpt2_init's normal draws over one layer, in its traversal order: [(weight, std)]."""
    std, scaled = cfg.init_std, cfg.init_std / math.sqrt(2 * cfg.n_layer)
    out = []
    for name, m in layer.named_modules():
        if isinstance(m, nn.Linear):
            # proj and mlp.down write into the residual stream
            s = scaled if name in ("proj", "down") or name.endswith((".proj", ".down")) else std
            out.append((m.weight, s))
        elif isinstance(m, nn.Embedding):
            out.append((m.weight, std))
    return out


def gpt2_init(layers, cfg, skipped: dict | None = None):
    """Standard transformer init: N(0, init_std) everywhere, with residual
    projections scaled by 1/sqrt(2*n_layer) so the residual stream variance does
    not grow with depth. Applied in a fixed traversal order under a fixed seed,
    so it is identical in every process. `skipped` (layer index -> [(shape,
    std)]): layers a span build dropped; their draws fill a scratch tensor of the
    same shape, so the generator advances exactly as in the full build (zeros_
    draws nothing)."""
    for i, layer in enumerate(layers):
        if skipped is not None and i in skipped:
            for shape, s in skipped[i]:
                torch.empty(shape).normal_(mean=0.0, std=s)
            continue
        for w, s in gpt2_draws(layer, cfg):
            nn.init.normal_(w, mean=0.0, std=s)
        for m in layer.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)


def build_layers(cfg, seed: int, span: tuple | None = None) -> nn.ModuleList:
    """Flat layer list. Construction order is fixed, so init is identical in
    every process that builds the same config under the same seed.

    `span` (a, b) (RUNTIME-I22): entries outside [a, b) are built in the same
    order, so the generator advances exactly as in the full build, and dropped
    at once (nn.Identity in their place): a stage holds one foreign layer at a
    time instead of the whole model. A tied table spans both ends and is built
    in full."""
    cfg.validate()
    torch.manual_seed(seed)
    keep = None if span is None or cfg.tie_embeddings else range(*span)
    skipped: dict = {}

    def take(i, layer):
        if keep is None or i in keep:
            return layer
        if cfg.init_scheme == "gpt2":
            skipped[i] = [(tuple(w.shape), s) for w, s in gpt2_draws(layer, cfg)]
        return nn.Identity()

    emb = Embed(cfg)
    layers: list = [take(0, emb)]
    layers += [take(i + 1, Block(cfg, i + 1)) for i in range(cfg.n_layer)]
    layers.append(take(cfg.n_layer + 1, Head(cfg, emb.tok if cfg.tie_embeddings else None)))
    if cfg.init_scheme == "gpt2":
        gpt2_init(layers, cfg, skipped)
        if cfg.tie_embeddings:  # re-tie: init overwrote out.weight
            layers[-1].out.weight = emb.tok.weight
    return nn.ModuleList(layers)


# ------------------------------------------------------------------------- MoE
#
# ADR-0008's exact MoE block (SPEC-model §2 "MoE block", MODEL-I8). Every row
# set, order and reduction is written: a sum assembled outside one kernel call
# is a left fold in a stated order, a tensor with three or more autograd
# consumers goes through one fan-out whose backward is the left fold of their
# gradients, selection is a stable descending sort (never topk), each expert is
# called exactly once per block call on exactly its rows (0 rows included), and
# nothing draws from a generator. The same function serves MoEMLP and the HF
# class-swapped blocks (moe_forward over a parameter view).

MOE_FIELDS = (
    "moe_experts",
    "moe_top_k",
    "moe_d_ff",
    "moe_shared_d_ff",
    "moe_shared_gate",
    "moe_layers",
    "moe_router_dtype",
    "moe_router_bias",
    "moe_score",
    "moe_select",
    "moe_groups",
    "moe_topk_groups",
    "moe_norm",
    "moe_norm_eps",
    "moe_scale",
    "moe_apply",
    "moe_expert",
    "moe_limit",
    "moe_alpha",
    "moe_lbl_coef",
    "moe_lbl_scope",
    "moe_z_coef",
    "moe_bias_rate",
)


class MoERecord(NamedTuple):
    """One call of a block: its layer, every aux term's value as a float hex,
    and its int64 expert counts (E,), whose sum is T·k."""

    layer: int
    aux: dict
    counts: torch.Tensor


def moe_recipe(cfg) -> SimpleNamespace:
    """The moe_* fields of a GPTConfig, or of a dict (an HF row's recipe(),
    I-11), with GPTConfig's defaults for the keys a dict leaves out."""
    from wantrain.model.gpt import GPTConfig

    base = GPTConfig()
    if isinstance(cfg, dict):
        unknown = sorted(k for k in cfg if k not in MOE_FIELDS)
        if unknown:
            raise ValueError(f"MoE recipe: unknown field {unknown[0]!r}")
        return SimpleNamespace(**{k: cfg.get(k, getattr(base, k)) for k in MOE_FIELDS})
    return SimpleNamespace(**{k: getattr(cfg, k) for k in MOE_FIELDS})


def check_moe(cfg) -> None:
    """SPEC-model §2's construction rules; each refusal names the field."""
    r = moe_recipe(cfg)
    E, k = r.moe_experts, r.moe_top_k
    if E < 0:
        raise ValueError(f"moe_experts={E} must be >= 0")
    if E == 0:
        return
    if not 1 <= k <= E:
        raise ValueError(f"moe_top_k={k} must satisfy 1 <= moe_top_k <= moe_experts={E}")
    enums = (
        ("moe_router_dtype", ("compute", "fp32")),
        ("moe_score", ("softmax", "softmax_k", "sigmoid")),
        ("moe_select", ("score", "logit", "score_bias")),
        ("moe_shared_gate", ("sigmoid", "none")),
        ("moe_apply", ("output", "input")),
        ("moe_expert", ("swiglu", "swiglu_clamped")),
        ("moe_lbl_scope", ("layer", "sequence")),
    )
    for name, allowed in enums:
        if getattr(r, name) not in allowed:
            raise ValueError(f"{name}={getattr(r, name)!r} is not one of {allowed}")
    if r.moe_apply == "input" and k != 1:
        raise ValueError(f"moe_apply='input' requires moe_top_k=1, got moe_top_k={k}")
    G, Gt = r.moe_groups, r.moe_topk_groups
    if G < 1:
        raise ValueError(f"moe_groups={G} must be >= 1")
    if G > 1:
        if E % G:
            raise ValueError(f"moe_groups={G} must divide moe_experts={E}")
        if E // G < 2:
            raise ValueError(f"moe_groups={G}: moe_experts/moe_groups={E // G} must be >= 2")
        if not 1 <= Gt <= G:
            raise ValueError(f"moe_topk_groups={Gt} must satisfy 1 <= moe_topk_groups <= {G}")
        if k > Gt * (E // G):
            raise ValueError(
                f"moe_top_k={k} exceeds moe_topk_groups*moe_experts/moe_groups={Gt * (E // G)}"
            )
    elif Gt != 1:
        raise ValueError(f"moe_topk_groups={Gt} requires moe_groups > 1")
    if r.moe_d_ff is not None and r.moe_d_ff < 1:
        raise ValueError(f"moe_d_ff={r.moe_d_ff} must be >= 1")
    if r.moe_shared_d_ff < 0:
        raise ValueError(f"moe_shared_d_ff={r.moe_shared_d_ff} must be >= 0")
    if r.moe_bias_rate < 0:
        raise ValueError(f"moe_bias_rate={r.moe_bias_rate} must be >= 0")
    if r.moe_lbl_coef < 0 or r.moe_z_coef < 0:
        raise ValueError("moe_lbl_coef and moe_z_coef must be >= 0")


def is_moe_layer(cfg, index: int) -> bool:
    """Whether flat-list position `index` (block i is index i+1) is MoE:
    every block when moe_layers is None, else the blocks it lists (0-based)."""
    if getattr(cfg, "moe_experts", 0) <= 0 or index < 1:
        return False
    layers = cfg.moe_layers
    return layers is None or (index - 1) in layers


class Router(nn.Linear):
    """The MoE router: initialized as an nn.Linear, never converted to
    SplitLinear (its type is not nn.Linear), an fp32 master; dW in B."""


class MoEExpert(nn.Module):
    """One SwiGLU expert of SplitLinears (after wgrad.convert). X is cast to
    the autocast dtype once, so gate and up record the same X: the pending W
    of an expert row is 2h + 3f values (SPEC-model §2)."""

    def __init__(self, d, f, clamped=False, limit=7.0, alpha=1.702):
        super().__init__()
        self.gate = nn.Linear(d, f, bias=clamped)
        self.up = nn.Linear(d, f, bias=clamped)
        self.down = nn.Linear(f, d, bias=clamped)
        self.clamped, self.limit, self.alpha = clamped, limit, alpha

    def forward(self, x):
        dev = x.device.type
        if torch.is_autocast_enabled(dev):
            x = x.to(torch.get_autocast_dtype(dev))
        g, u = self.gate(x), self.up(x)
        if self.clamped:
            g = g.clamp(max=self.limit)
            u = u.clamp(min=-self.limit, max=self.limit)
            return self.down((u + 1) * (g * torch.sigmoid(g * self.alpha)))
        return self.down(F.silu(g) * u)


class ExpertList(nn.ModuleList):
    """The E experts; one module call per block call runs expert e exactly
    once on its rows of the dispatched tensor (ascending token index), also
    on 0 rows, and returns their outputs concatenated in expert order."""

    def forward(self, xd, counts: list):
        return torch.cat([e(rows) for e, rows in zip(self, torch.split(xd, counts))])


class MoEMLP(nn.Module):
    """ADR-0008's block in place of a Block's MLP. Attributes in construction
    order: router, experts, shared_expert, shared_expert_gate, sel_bias."""

    def __init__(self, cfg, moe_layer: int):
        super().__init__()
        check_moe(cfg)
        r = moe_recipe(cfg)
        d, E = cfg.d_model, r.moe_experts
        f = r.moe_d_ff or cfg.d_ff or default_d_ff("swiglu", d)
        clamped = r.moe_expert == "swiglu_clamped"
        self.cfg = cfg
        self.moe_cfg = cfg
        self.moe_layer = moe_layer
        self.moe_records: list = []
        self.router = Router(d, E, bias=r.moe_router_bias)
        self.experts = ExpertList(
            MoEExpert(d, f, clamped, r.moe_limit, r.moe_alpha) for _ in range(E)
        )
        sh = r.moe_shared_d_ff
        self.shared_expert = MoEExpert(d, sh) if sh else None
        gated = sh and r.moe_shared_gate == "sigmoid"
        self.shared_expert_gate = nn.Linear(d, 1, bias=False) if gated else None
        self.register_buffer("sel_bias", torch.zeros(E, dtype=torch.float32))

    def forward(self, x):
        return moe_forward(self.moe_cfg, self, x)


class _Fanout(torch.autograd.Function):
    """n views of one tensor; backward is the left fold of their gradients in
    output order (ADR-0008 D1), never the engine's accumulation order."""

    @staticmethod
    def forward(ctx, x, n):
        return tuple(x.view_as(x) for _ in range(n))

    @staticmethod
    def backward(ctx, *gs):
        acc = gs[0]
        for g in gs[1:]:
            acc = acc + g
        return acc, None


def _fan(t, n: int) -> tuple:
    # Two consumers sum exactly in either order (a + b == b + a); three or more
    # go through the written fold.
    return _Fanout.apply(t, n) if n >= 3 else (t,) * n


class _Dispatch(torch.autograd.Function):
    """Rows of x in (expert, token) ascending slot order: one gather. Backward
    gathers a token's k slot gradients and folds them in ascending expert
    index (ADR-0008 D5)."""

    @staticmethod
    def forward(ctx, x, src, pos):
        ctx.save_for_backward(pos)
        return x.index_select(0, src)

    @staticmethod
    def backward(ctx, g):
        (pos,) = ctx.saved_tensors
        T, k = pos.shape
        G = g.index_select(0, pos.reshape(-1)).view(T, k, -1)
        acc = G[:, 0]
        for j in range(1, k):
            acc = acc + G[:, j]
        return acc, None, None


class _Permute(torch.autograd.Function):
    """A permutation of rows; backward is the inverse permutation (gathers
    only: a scatter with unique targets is a byte move)."""

    @staticmethod
    def forward(ctx, x, perm, inv):
        ctx.save_for_backward(inv)
        return x.index_select(0, perm)

    @staticmethod
    def backward(ctx, g):
        (inv,) = ctx.saved_tensors
        return g.index_select(0, inv), None, None


class _Combine(torch.autograd.Function):
    """o_t = left fold over ascending expert index of w_tj·Y_tj in fp32 (or
    the expert's row when w is None, apply 'input'). Backward: dY through the
    inverse slot permutation, dw_tj = Σ_h g_t·Y_tj in fp32 (ADR-0008 D7)."""

    @staticmethod
    def forward(ctx, Y, w, pos, perm):
        T, k = pos.shape
        Ysl = Y.index_select(0, pos.reshape(-1)).view(T, k, -1).float()
        ctx.ydt, ctx.weighted = Y.dtype, w is not None
        ctx.save_for_backward(Ysl, w if w is not None else Ysl.new_empty(0), perm)
        acc = None
        for j in range(k):
            t = Ysl[:, j] if w is None else w[:, j, None] * Ysl[:, j]
            acc = t if acc is None else acc + t
        return acc

    @staticmethod
    def backward(ctx, g):
        Ysl, w, perm = ctx.saved_tensors
        T, k, d = Ysl.shape
        if ctx.weighted:
            dYsl = (w[..., None] * g[:, None, :]).to(ctx.ydt).reshape(T * k, d)
            dw = (g[:, None, :] * Ysl).sum(-1)
        else:
            dYsl = g[:, None, :].expand(T, k, d).to(ctx.ydt).reshape(T * k, d)
            dw = None
        return dYsl.index_select(0, perm), dw, None, None


class _AuxInject(torch.autograd.Function):
    """Identity on the block output whose backward hands each injected aux
    term its coefficient as gradient (ADR-0008 D8, decision 2): the aux loss
    acts per layer and microbatch; the CE loss never carries it."""

    @staticmethod
    def forward(ctx, y, coefs, *terms):
        ctx.coefs = coefs
        return y.view_as(y)

    @staticmethod
    def backward(ctx, g):
        return (
            g,
            None,
            *(torch.full((), c, dtype=torch.float32, device=g.device) for c in ctx.coefs),
        )


def _stable_desc(key: torch.Tensor) -> torch.Tensor:
    """Indices of a stable descending sort along the last dim: larger value
    first, then lower index (ties and ±0 included)."""
    return torch.sort(key, dim=-1, descending=True, stable=True).indices


def moe_select(key: torch.Tensor, k: int, groups: int = 1, topk_groups: int = 1) -> torch.Tensor:
    """(T, E) fp32 keys -> (T, k) selected experts, ascending per token
    (ADR-0008 D3). With groups, a group's key is the sum of its top-2 keys and
    the topk_groups groups first in a stable descending sort are eligible;
    then the first k of the stable descending sort of the eligible keys. A NaN
    key is fatal; torch.topk is never used."""
    if bool(torch.isnan(key).any()):
        raise ValueError("MoE selection: a NaN router key is fatal")
    T, E = key.shape
    order = _stable_desc(key)
    if groups > 1:
        kg = key.view(T, groups, E // groups)
        top2 = torch.sort(kg, dim=-1, descending=True, stable=True).values[..., :2]
        gkey = top2[..., 0] + top2[..., 1]
        chosen = _stable_desc(gkey)[:, :topk_groups]
        allowed = torch.zeros(T, groups, dtype=torch.uint8, device=key.device)
        allowed.scatter_(1, chosen, 1)
        allowed = allowed.repeat_interleave(E // groups, dim=1)
        # Eligible experts first (stable), keeping the key order within each.
        order = order.gather(1, _stable_desc(allowed.gather(1, order)))
    return torch.sort(order[:, :k], dim=-1).values


def moe_forward(cfg, view, x: torch.Tensor) -> torch.Tensor:
    """The exact MoE function of SPEC-model §2 over a parameter view: x̂
    (b, t, h) -> (b, t, h) in x̂'s dtype. `view` holds `router` (a module whose
    call gives the logits), `experts` (one module call: (dispatched rows,
    per-expert row counts) -> outputs in expert order), `shared_expert` and
    `shared_expert_gate` (modules or None), `sel_bias` (fp32, E), `moe_layer`
    and `moe_records`, to which each call appends one MoERecord."""
    r = moe_recipe(cfg)
    E, k = r.moe_experts, r.moe_top_k
    b, t, h = x.shape
    T = b * t
    grad = torch.is_grad_enabled()
    want_lbl = grad and r.moe_lbl_coef != 0
    want_z = grad and r.moe_z_coef != 0
    sh = view.shared_expert is not None
    shg = sh and view.shared_expert_gate is not None
    xs = _fan(x.reshape(T, h), 2 + sh + shg)

    # Router: the GEMM under the recipe's autocast ('compute') or in fp32.
    if r.moe_router_dtype == "fp32":
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = view.router(xs[0].float())
    else:
        logits = view.router(xs[0])
    lf = logits.float()
    lfd = lf.detach()
    uses = (["pk", "q"] if want_lbl else ["pk"]) if r.moe_score == "softmax_k" else ["s"]
    if want_z:
        uses.append("z")
    lfs = dict(zip(uses, _fan(lf, len(uses))))

    # Scores (fp32) and q, the normalized probabilities the lbl term reads.
    if r.moe_score == "softmax_k":
        sdet = qdet = torch.softmax(lfd, -1)
    else:
        act = torch.softmax if r.moe_score == "softmax" else (lambda z, _d: torch.sigmoid(z))
        s = act(lfs["s"], -1)
        sdet = s.detach()
        qdet = sdet if r.moe_score == "softmax" else sdet / sdet.sum(-1, keepdim=True)
        suses = ["p"]
        if want_lbl:
            suses += ["q"] if r.moe_score == "softmax" else ["qn", "qd"]
        ss = dict(zip(suses, _fan(s, len(suses))))
    if want_lbl:
        if r.moe_score == "softmax_k":
            q = torch.softmax(lfs["q"], -1)
        elif r.moe_score == "softmax":
            q = ss["q"]
        else:
            q = ss["qn"] / ss["qd"].sum(-1, keepdim=True)
    else:
        q = qdet

    # Selection on fp32 keys (never differentiated).
    if r.moe_select == "logit":
        key = lfd
    elif r.moe_select == "score":
        key = sdet
    else:
        key = sdet + view.sel_bias.to(device=sdet.device, dtype=torch.float32)
    sel = moe_select(key, k, r.moe_groups, r.moe_topk_groups)

    # Weights: the score at the selected experts (never the bias), then the
    # renormalization by the left fold in ascending expert index, then scale.
    if r.moe_score == "softmax_k":
        p = torch.softmax(lfs["pk"].gather(1, sel), -1)
    else:
        p = ss["p"].gather(1, sel)
    if r.moe_norm:
        pn, pd = _fan(p, 2)
        cols = pd.unbind(1)
        den = cols[0]
        for c in cols[1:]:
            den = den + c
        p = pn / (den + r.moe_norm_eps)[:, None]
    w = p * r.moe_scale

    # Dispatch: slots sorted by (expert, token); one gather.
    flat_e = sel.reshape(-1)
    flat_t = torch.arange(T, device=x.device).repeat_interleave(k)
    perm = torch.sort(flat_e * T + flat_t, stable=True).indices
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(T * k, device=x.device)
    pos = inv.view(T, k)
    counts = torch.bincount(flat_e, minlength=E)
    xd = _Dispatch.apply(xs[1], flat_t[perm], pos)
    if r.moe_apply == "input":
        wd = _Permute.apply(w.reshape(-1), perm, inv)
        xd = (xd.float() * wd[:, None]).to(xd.dtype)
    Y = view.experts(xd, counts.tolist())
    out = _Combine.apply(Y, None if r.moe_apply == "input" else w, pos, perm)
    if sh:
        ys = view.shared_expert(xs[2]).float()
        if shg:
            ys = torch.sigmoid(view.shared_expert_gate(xs[3]).float()) * ys
        out = out + ys
    out = out.to(x.dtype).view(b, t, h)

    # Aux terms: values recorded per call; gradients injected locally.
    cf = counts.to(torch.float32)
    if r.moe_lbl_scope == "layer":
        lbl = (E / (T * T * k)) * (cf * q.sum(0)).sum()
    else:
        cs = F.one_hot(sel.view(b, t * k), E).sum(1).to(torch.float32)
        lbl = ((E / (t * t * k)) * (cs * q.view(b, t, E).sum(1)).sum(-1)).mean()
    lse = torch.logsumexp(lfs["z"] if want_z else lfd, -1)
    z = (lse * lse).mean()
    coefs, terms = [], []
    if want_lbl:
        coefs.append(float(r.moe_lbl_coef))
        terms.append(lbl)
    if want_z:
        coefs.append(float(r.moe_z_coef))
        terms.append(z)
    if terms:
        out = _AuxInject.apply(out, tuple(coefs), *terms)
    aux = {"lbl": float(lbl.detach()).hex(), "z": float(z.detach()).hex()}
    view.moe_records.append(MoERecord(view.moe_layer, aux, counts.detach().cpu()))
    return out


def take_moe_records(layers) -> list:
    """Pop every MoE record held by the modules of `layers`, sorted by
    (moe_layer, call order) (I-11, BA-462)."""
    found, seen = [], set()
    for layer in layers:
        for m in layer.modules():
            recs = getattr(m, "moe_records", None)
            if recs is None or id(m) in seen:
                continue
            seen.add(id(m))
            found += [(rec.layer, i, rec) for i, rec in enumerate(recs)]
            recs.clear()
    return [rec for _, _, rec in sorted(found, key=lambda x: (x[0], x[1]))]


def moe_bias_update(layers, counts: dict, global_tokens: int) -> None:
    """The selection-bias update after the step's AdamW (ADR-0008 D10):
    sel_bias_i += γ·sign(T_g·k − E·c_i), the sign taken in integers (0 on
    equality), a move of exactly +γ, −γ or 0. `counts` maps moe_layer to the
    int64 counts (E,) of the whole global batch, folded by integer add; a
    vector that is not int64 (E,), is negative or does not sum to T_g·k is
    fatal, naming the layer."""
    for layer in layers:
        for m in layer.modules():
            if getattr(m, "moe_records", None) is None or m.moe_layer not in counts:
                continue
            r = moe_recipe(m.moe_cfg)
            E, k, gamma = r.moe_experts, r.moe_top_k, float(r.moe_bias_rate)
            c = counts[m.moe_layer]
            if not isinstance(c, torch.Tensor) or c.dtype != torch.int64 or tuple(c.shape) != (E,):
                raise ValueError(f"MoE layer {m.moe_layer}: counts must be int64 of shape ({E},)")
            if bool((c < 0).any()) or int(c.sum()) != global_tokens * k:
                raise ValueError(
                    f"MoE layer {m.moe_layer}: counts are negative or do not sum to "
                    f"T_g*k={global_tokens * k}"
                )
            if gamma == 0:
                continue
            sgn = torch.sign(global_tokens * k - E * c.to(m.sel_bias.device))
            with torch.no_grad():
                moved = m.sel_bias + sgn.to(torch.float32) * gamma
                m.sel_bias.copy_(torch.where(sgn == 0, m.sel_bias, moved))
