"""GPT-2 style decoder as a flat layer list [Embed, Block * n_layer, Head], so
contiguous slices can be assigned to pipeline stages.

Every process builds the full list under the same seed and keeps only its
slice, so parameters are identical everywhere. No dropout: the reference recipe
is deterministic and stateless, so recomputation and cross-process replay are
exact. A boundary_dtype narrower than compute is applied inside the model at
every block boundary (scaled fp8 uses a per-token power-of-two scale, see
model/fp8.py), so the recipe is partition-independent and the wire is exact.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Pinned attention backends. The exactness claim is made per declared device
# map, so the kernel must be declared too -- letting PyTorch pick by heuristic
# would make the recipe depend on the host's kernel availability.
_SDPA_BACKENDS = {
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def resolve_sdpa_backend(name: str, device_type: str, compute_dtype: str = None):
    """Resolve "auto" against the device and the compute dtype.

    Only flash/efficient are memory-linear in sequence length and both are
    CUDA-only; on CPU the math backend materialises the score matrix. Flash
    has no fp32 path ("No available kernel"), so fp32 gets the memory-efficient
    kernel. "auto" is a pure function of (device, dtype) whose result goes into
    the run fingerprint; an explicitly named backend is never second-guessed.
    """
    if name == "auto":
        if device_type != "cuda":
            name = "math"
        else:
            name = "efficient" if (compute_dtype or "fp32") == "fp32" else "flash"
    return _SDPA_BACKENDS[name]


@dataclass
class GPTConfig:
    vocab_size: int = 256
    n_layer: int = 6
    n_head: int = 4
    d_model: int = 128
    seq_len: int = 64
    # Declared precision (see model/precision.py): compute_dtype is what the
    # block math runs in, boundary_dtype what crosses a stage boundary. Equal
    # values make the boundary native and the run lossless against the declared
    # reference recipe; a narrower boundary is a declared approximation.
    compute_dtype: str = "fp32"
    boundary_dtype: str = "fp32"
    # Attention kernel. "naive" materialises the (mb, heads, S, S) scores and
    # caps S in the low thousands; "sdpa" is memory-linear in S, which is what
    # makes long sequences reachable. sdpa is not bit-identical to naive (pure
    # reassociation), so attn_impl is part of the recipe; the baseline must match.
    attn_impl: str = "naive"  # naive | sdpa
    # "math" still materialises O(S^2); only flash/efficient are memory-linear,
    # and both are CUDA-only. "auto" resolves by device and compute dtype (flash
    # has no fp32 path) and the resolved value is recorded in the run fingerprint.
    sdpa_backend: str = "math"  # math | efficient | flash | cudnn | auto

    # Architecture knobs. Defaults reproduce the original block bit-for-bit; any
    # decoder expressible as a flat layer list can be trained by this engine.
    arch: str = "builtin"  # builtin | hf | user
    # arch="hf": a checkpoint directory on the box; each node downloads it from
    # the hub directly. hf_model_sha256 is what makes that safe: stages that
    # loaded different weights would still train and still show a falling loss.
    hf_model: str | None = None
    hf_model_sha256: str | None = None
    # arch="user": path, on the box, to a Python file defining build(cfg) ->
    # a flat nn.ModuleList. It travels in the JobSpec so every stage builds
    # from the same bytes; user_model_sha256 is checked before the file is
    # executed, so "all stages ran the same model" is verified, not assumed.
    user_model: str | None = None
    user_model_sha256: str | None = None
    # LoRA. rank 0 = off. A memory knob first: full training costs 16 bytes per
    # parameter (fp32 master + Adam's two moments + gradient); freezing the base
    # leaves optimizer state only for the adapters. It buys no bandwidth: the
    # wire carries activations either way.
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_targets: str = "attn"  # attn | attn_mlp | all
    n_kv_head: int | None = None  # None = full multi-head; < n_head = GQA
    d_ff: int | None = None  # None = 4*d_model (gelu) or 8/3*d (swiglu)
    norm: str = "layernorm"  # layernorm | rmsnorm
    mlp: str = "gelu"  # gelu | swiglu
    pos: str = "learned"  # learned | rope
    bias: bool = True
    tie_embeddings: bool = False
    rope_theta: float = 10000.0
    # Initialisation is part of the recipe. "torch_default" reproduces the
    # original baselines, but nn.Embedding defaults to N(0,1) and nn.Linear to
    # U(+-1/sqrt(d)), ~16x apart at d=256, so tie_embeddings requires "gpt2",
    # which puts both at N(0, init_std) and scales residual projections.
    init_scheme: str = "torch_default"  # torch_default | gpt2
    init_std: float = 0.02

    # arch="hf" (ADR-0009, I-1): the pinned transformers' own module tree,
    # windowed per pipeline unit (model/hfm.py). hf_config is the config.json
    # content of the pinned config class ("model_type" selects it); the
    # buffer record, census and code identity are pinned at the first launch.
    hf_config: dict | None = None
    hf_buffers: dict | None = None
    hf_census: list | None = None
    hf_code: dict | None = None
    hf_impl: str = "modules"

    # Mixture of experts (ADR-0008, SPEC-model §2; arch.MoEMLP). 0 = dense.
    moe_experts: int = 0
    moe_top_k: int = 1
    moe_d_ff: int | None = None  # None = d_ff, else the SwiGLU default width
    moe_shared_d_ff: int = 0  # 0 = no shared expert
    moe_shared_gate: str = "sigmoid"  # sigmoid | none
    moe_layers: list | None = None  # 0-based block indices; None = every block
    moe_router_dtype: str = "compute"  # compute | fp32
    moe_router_bias: bool = False
    moe_score: str = "softmax"  # softmax | softmax_k | sigmoid
    moe_select: str = "score"  # score | logit | score_bias
    moe_groups: int = 1
    moe_topk_groups: int = 1
    moe_norm: bool = False
    moe_norm_eps: float = 0.0
    moe_scale: float = 1.0
    moe_apply: str = "output"  # output | input (k = 1)
    moe_expert: str = "swiglu"  # swiglu | swiglu_clamped
    moe_limit: float = 7.0
    moe_alpha: float = 1.702
    moe_lbl_coef: float = 0.0
    moe_lbl_scope: str = "layer"  # layer | sequence
    moe_z_coef: float = 0.0
    moe_bias_rate: float = 0.0  # gamma; 0 = no selection-bias update

    def validate(self):
        from wantrain.model import precision

        if self.arch not in ("builtin", "hf", "user"):
            raise ValueError(
                f"arch={self.arch!r} is not a build path; use 'builtin' (the "
                f"built-in decoder), 'hf', or 'user' with user_model pointing "
                f"at a file defining build(cfg)"
            )
        if self.arch == "user" and not self.user_model:
            raise ValueError("arch='user' requires user_model")
        precision.spec(self.compute_dtype)
        precision.spec(self.boundary_dtype)
        if self.compute_dtype not in precision.CAN_BE_COMPUTE:
            raise ValueError(
                f"compute_dtype={self.compute_dtype!r} is not an implemented "
                f"compute path; choose from {sorted(precision.CAN_BE_COMPUTE)}"
            )
        if self.tie_embeddings and self.init_scheme != "gpt2" and self.arch != "hf":
            raise ValueError(
                "tie_embeddings=True requires init_scheme='gpt2': with torch "
                "defaults the embedding (N(0,1)) and the output projection "
                "(U(+-1/sqrt(d))) differ by ~16x in scale, so tying them makes "
                "the initial loss explode instead of starting near ln(vocab)"
            )
        if self.n_kv_head is not None and self.n_head % self.n_kv_head:
            raise ValueError("n_head must be a multiple of n_kv_head")
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if self.attn_impl not in ("naive", "sdpa"):
            raise ValueError(f"attn_impl={self.attn_impl!r}")
        if self.arch == "hf":
            self._validate_hf()
        if self.moe_experts:
            from wantrain.model import arch

            arch.check_moe(self)
            if self.lora_rank:
                raise ValueError(
                    f"lora_rank={self.lora_rank} with moe_experts={self.moe_experts}: "
                    f"LoRA with MoE is refused (SPEC-model task 3)"
                )
        return self

    def _validate_hf(self):
        """I-1's arch 'hf' rules; each refusal names the field. Never imports
        transformers: it runs in the coordinator and the planner."""
        from wantrain.model import rollback

        pieces = rollback.rolled_back_to("model-4a")
        if pieces:
            raise ValueError(
                f"arch='hf' refused: piece {pieces[0]!r} is rolled back to 'model-4a' "
                f"(model/rollback.py)"
            )
        if not isinstance(self.hf_config, dict) or not self.hf_config.get("model_type"):
            raise ValueError("arch='hf' requires hf_config, a config.json dict with model_type")
        if self.hf_impl != "modules":
            raise ValueError(f"hf_impl={self.hf_impl!r}: arch='hf' accepts only 'modules'")
        if self.lora_rank:
            raise ValueError(f"lora_rank={self.lora_rank}: LoRA is refused under arch='hf'")
        if self.fp8_boundaries or self.boundary_dtype != "fp32":
            raise ValueError(
                f"boundary_dtype={self.boundary_dtype!r}: arch='hf' cuts at HF's residual "
                f"dtype, fp32 (ADR-0009 D10)"
            )
        if self.compute_dtype not in ("fp32", "bf16"):
            raise ValueError(
                f"compute_dtype={self.compute_dtype!r}: arch='hf' accepts fp32 or bf16"
            )
        census = self.hf_census
        if census is not None:
            layers = [u for u in census if u.get("kind") == "layer"]
            mirrored = {
                "n_layer": len(layers),
                "n_head": layers[0].get("attn_heads") if layers else None,
                "tie_embeddings": any(u.get("tied") for u in census),
            }
            for name, value in mirrored.items():
                if getattr(self, name) != value:
                    raise ValueError(
                        f"{name}={getattr(self, name)!r} differs from hf_census's {value!r}"
                    )

    # Deprecated: kept so specs written before the precision ladder still load.
    # True maps to boundary_dtype="fp8_e4m3" (with the mandatory per-token scale).
    fp8_boundaries: bool = False

    def __post_init__(self):
        if self.fp8_boundaries and self.boundary_dtype == "fp32":
            self.boundary_dtype = "fp8_e4m3"

    def compute_torch_dtype(self):
        from wantrain.model import precision

        return precision.spec(self.compute_dtype)["torch_dtype"]

    def autocast_enabled(self) -> bool:
        return self.compute_dtype != "fp32"

    def boundary_narrowing(self):
        """-> (target dtype, needs_scale) or (None, False) if boundary is native."""
        from wantrain.model import precision

        if precision.classify(self.compute_dtype, self.boundary_dtype)["level"] == precision.NATIVE:
            return None, False
        s = precision.spec(self.boundary_dtype)
        return s["torch_dtype"], s["needs_scale"]

    def recipe(self):
        from wantrain.model import precision

        return precision.describe(self.compute_dtype, self.boundary_dtype, self.d_model)


# Real configs for GPU phases. Param counts are approximate (tied to the
# synthetic vocab, not GPT-2's 50257).
PRESETS: dict[str, dict] = {
    "toy": dict(),
    "gpt124m": dict(vocab_size=32768, n_layer=12, n_head=12, d_model=768, seq_len=512),
    "gpt350m": dict(vocab_size=32768, n_layer=24, n_head=16, d_model=1024, seq_len=512),
    "gpt760m": dict(vocab_size=32768, n_layer=24, n_head=16, d_model=1536, seq_len=512),
    "gpt1_5b": dict(vocab_size=32768, n_layer=48, n_head=25, d_model=1600, seq_len=512),
    # Sized to fill one H100 80GB single-device (71 GB at mb=8 with recompute).
    # Sharded, its ~70 GB of optimizer state alone needs >= 4 stages of 24 GB.
    "gpt4_4b": dict(vocab_size=32768, n_layer=34, n_head=25, d_model=3200, seq_len=1024),
}


def make_config(
    preset: str,
    fp8_boundaries: bool = False,
    compute_dtype: str = "fp32",
    boundary_dtype: str = "fp32",
) -> "GPTConfig":
    return GPTConfig(
        fp8_boundaries=fp8_boundaries,
        compute_dtype=compute_dtype,
        boundary_dtype=boundary_dtype,
        **PRESETS[preset],
    )


def narrow(x: torch.Tensor, dtype, needs_scale: bool) -> torch.Tensor:
    """Declared narrowing of a boundary tensor, in the tensor's own dtype.

    Formats whose dynamic range covers the tensor (bf16, fp16 activations) are
    a plain cast. All fp8 formats must be scaled first: e4m3's smallest
    subnormal is 2^-9 while activation gradients are ~1e-5, so an unscaled cast
    rounds them to zero and freezes every layer upstream of the output head.
    """
    if dtype is None:
        return x
    if needs_scale:
        from wantrain.model import fp8

        return fp8.roundtrip(x, "e5m2" if dtype is torch.float8_e5m2 else "e4m3")
    return x.to(dtype).to(x.dtype)


class _Boundary(torch.autograd.Function):
    """Declared narrowing applied symmetrically at a stage boundary.

    Forward narrows the activation, backward narrows the gradient, with the
    same format both ways. Applied at every block input and output so the
    recipe is partition-independent: the single-device baseline runs the
    identical narrowing at the identical points.
    """

    @staticmethod
    def forward(ctx, x, dtype, needs_scale):
        ctx.dtype, ctx.needs_scale = dtype, needs_scale
        return narrow(x, dtype, needs_scale)

    @staticmethod
    def backward(ctx, g):
        return narrow(g, ctx.dtype, ctx.needs_scale), None, None


class Embed(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.bnd = cfg.boundary_narrowing()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        # Embedding lookups are not autocast ops, so the activation stream is
        # started in the declared compute dtype here. Parameters stay fp32 --
        # standard mixed precision keeps master weights and the optimizer in
        # fp32 and carries only activations and gradients in the narrow dtype.
        if self.cfg.autocast_enabled():
            x = x.to(self.cfg.compute_torch_dtype())
        return _Boundary.apply(x, *self.bnd)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.bnd = cfg.boundary_narrowing()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Boundary op at the input as well as the output. The output-side op
        # narrows the activation sent forward but not the gradient sent back:
        # that gradient is dL/d(this layer's input), produced upstream of the
        # output-side op. The narrow is idempotent, so double crossings are safe.
        x = _Boundary.apply(x, *self.bnd)
        b, t, c = x.shape
        h = self.cfg.n_head
        # Under a narrow compute dtype the matmuls run in it while autocast
        # keeps LayerNorm and softmax in fp32 -- textbook mixed precision. With
        # compute_dtype="fp32" autocast is disabled and this is a no-op, so the
        # existing bitwise baselines are untouched.
        with torch.autocast(
            device_type=x.device.type,
            dtype=self.cfg.compute_torch_dtype(),
            enabled=self.cfg.autocast_enabled(),
        ):
            q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
            q = q.view(b, t, h, c // h).transpose(1, 2)
            k = k.view(b, t, h, c // h).transpose(1, 2)
            v = v.view(b, t, h, c // h).transpose(1, 2)
            if self.cfg.attn_impl == "sdpa":
                with sdpa_kernel(
                    resolve_sdpa_backend(
                        self.cfg.sdpa_backend, x.device.type, self.cfg.compute_dtype
                    )
                ):
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / (c // h) ** 0.5)
                mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
                att = att.masked_fill(mask, float("-inf"))
                att = F.softmax(att, dim=-1)
                y = att @ v
            y = y.transpose(1, 2).contiguous().view(b, t, c)
            x = x + self.proj(y).to(x.dtype)
            x = x + self.fc2(F.gelu(self.fc1(self.ln2(x)))).to(x.dtype)
        # One symmetric boundary op at the block output covers both directions:
        # it narrows the activation that leaves this layer, and narrows the
        # gradient w.r.t. that same tensor on the way back -- which is exactly
        # the pair of tensors that cross a stage boundary cut at this layer.
        return _Boundary.apply(x, *self.bnd)


class Head(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.bnd = cfg.boundary_narrowing()
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input-side boundary: a stage cut immediately before the head sends
        # dL/d(head input) back over the wire, and that gradient is produced here,
        # downstream of the previous layer's own boundary op.
        x = _Boundary.apply(x, *self.bnd)
        with torch.autocast(
            device_type=x.device.type,
            dtype=self.cfg.compute_torch_dtype(),
            enabled=self.cfg.autocast_enabled(),
        ):
            logits = self.out(self.ln_f(x))
        # Cross-entropy over a 50k vocab is accumulated in fp32 even in mixed
        # precision recipes; a no-op when the run is already fp32.
        return logits.float()


def call_layer(layer: nn.Module, x: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
    """layer(x, ids=ids) for a unit that takes the microbatch's input ids
    (an arch "hf" unit, I-1), else layer(x). Called only from arch "hf"
    branches, imported there at call time (BA-407); dense call sites keep
    layer(x)."""
    if getattr(layer, "takes_ids", False):
        return layer(x, ids=ids)
    return layer(x)


def build_layers(cfg: GPTConfig, seed: int, span: tuple[int, int] | None = None) -> nn.ModuleList:
    """Build the full flat layer list under a fixed seed.

    Dispatches on cfg.arch. The "builtin" path with default knobs is the
    original block, kept as its own code path so the existing bitwise baselines
    stay reproducible; any non-default knob (MoE included) routes to
    model/arch.py, and arch="hf" builds transformers' own modules windowed per
    unit (model/hfm.py; with cfg.hf_model only the units in `span` are loaded,
    BA-449). The builtin and general archs build every entry in order and keep
    only `span`'s (RUNTIME-I22): the same draws, one foreign layer resident at a
    time. LoRA (its adapters draw over the whole list) and arch "user" build in
    full. LoRA is applied last, over whichever architecture was built, which is
    why it composes with user code too.
    """
    if cfg.arch == "hf":
        from wantrain.model import hfm, wgrad

        cfg.validate()
        units = hfm.build(cfg, seed, span)
        # The trace's read-elsewhere Linears keep dW in B (ADR-0009 D4).
        wgrad.convert(units, exclude=units.exclusions)
        return units
    layers = _build_arch_layers(cfg, seed, None if cfg.lora_rank else span)
    if cfg.lora_rank:
        from wantrain.model import lora

        rep = lora.apply(layers, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets, seed=seed)
        import json as _json

        print(f"WT_LORA {_json.dumps(rep)}", flush=True)
    # Zero-bubble B/W split (model/wgrad.py): every linear can defer its dW,
    # so B costs only the activation-gradient chain and W only the weight
    # GEMMs. The reference trainer builds the same layers, so the bits agree.
    from wantrain.model import wgrad

    wgrad.convert(layers)
    return layers


def _build_arch_layers(cfg: GPTConfig, seed: int, span=None) -> nn.ModuleList:
    if cfg.arch == "user":
        return _build_user_layers(cfg, seed)
    if _uses_general_arch(cfg):
        from wantrain.model import arch

        return arch.build_layers(cfg, seed, span)
    torch.manual_seed(seed)
    keep = range(*span) if span is not None else None

    def take(i, layer):  # RUNTIME-I22: built in order, dropped at once when outside span
        return layer if keep is None or i in keep else nn.Identity()

    layers: list[nn.Module] = [take(0, Embed(cfg))]
    layers += [take(i + 1, Block(cfg)) for i in range(cfg.n_layer)]
    layers.append(take(cfg.n_layer + 1, Head(cfg)))
    return nn.ModuleList(layers)


def _build_user_layers(cfg: GPTConfig, seed: int) -> nn.ModuleList:
    """Build the operator's own model from the file the spec points at.

    The seed is set first: every stage builds the whole model and keeps its
    slice, so a build that seeds itself would give each stage a different
    model while looking healthy. The source is exec'd only after its sha256
    matches cfg.user_model_sha256, so "all stages ran the same code" is
    provable. The structural contract is enforced by control/validate.py.
    """
    import hashlib
    import os

    src_path = cfg.user_model
    if not src_path:
        raise ValueError(
            "arch='user' needs cfg.user_model to point at a Python file "
            "defining build(cfg) -> nn.ModuleList"
        )
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"user model source {src_path!r} is not on this box. The spec is "
            f"distributed to every rank, so the source file must be shipped "
            f"with it or the stages would build different models"
        )
    with open(src_path, "rb") as f:
        raw = f.read()
    if cfg.user_model_sha256:
        got = hashlib.sha256(raw).hexdigest()
        if got != cfg.user_model_sha256:
            raise ValueError(
                f"user model source does not match the spec: expected sha256 "
                f"{cfg.user_model_sha256[:12]}, this box has {got[:12]}. Refusing "
                f"to build -- stages running different code produce gradients "
                f"that are summed as if they came from one model"
            )

    ns: dict = {}
    # The operator's model is code by contract; it runs only after its sha256
    # matched the spec, so every stage provably executes the same bytes.
    exec(compile(raw.decode(), src_path, "exec"), ns)  # noqa: S102  # nosec B102
    build = ns.get("build")
    if build is None:
        raise ValueError(
            f"{src_path} defines no build(cfg). The contract is a single "
            f"function taking the config and returning a flat nn.ModuleList"
        )

    torch.manual_seed(seed)
    layers = build(cfg)
    if not isinstance(layers, nn.ModuleList):
        layers = nn.ModuleList(list(layers))
    return layers


def _uses_general_arch(cfg: GPTConfig) -> bool:
    return (
        cfg.norm != "layernorm"
        or cfg.mlp != "gelu"
        or cfg.pos != "learned"
        or not cfg.bias
        or cfg.tie_embeddings
        or (cfg.n_kv_head is not None and cfg.n_kv_head != cfg.n_head)
        or cfg.d_ff not in (None, 4 * cfg.d_model)
        or cfg.moe_experts > 0
    )


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
