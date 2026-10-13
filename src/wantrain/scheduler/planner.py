"""Job planner: turn a measured fleet into a complete, validated JobSpec.

Given the node and link measurements plus the model and corpus, return a spec
ready to ship, or refuse with the number that made it refuse; the rationale
travels with the spec. Decisions, in order: stage order (measured RTT and
bandwidth), layer partition (proportional to measured TFLOPs, exact min-max
DP), boundary precision (widest rung passing the overlap condition),
microbatch size (largest that fits the tightest node), data shards
(proportional to capacity), microbatches per step (from the bubble the
zero-bubble schedule leaves on this link). Overrides win
over all of it; one that breaks an admission check is reported, not corrected.
"""

import copy
import dataclasses
import math
import types
from dataclasses import dataclass, field
from typing import Any

from wantrain.model import precision, provision
from wantrain.model.arch import default_d_ff
from wantrain.resources.node import mixed_fleet
from wantrain.runtime import optim
from wantrain.runtime.spec import JobSpec
from wantrain.scheduler import admission, autoconfig, production, schedule
from wantrain.scheduler.exchange_sim import (
    EMISSIONS,
    Fabric,
    Unit,
    pieces_owners,
    simulate_exchange,
    stride_owners,
)
from wantrain.scheduler.partition import proportional_partition
from wantrain.scheduler.zero_bubble import (
    ZBConfig,
    analytic_costs,
    bridge_orders,
    iwrr,
    stage_memory_model,
)


@dataclass
class Node:
    """One measured worker."""

    host: str
    tflops: float  # measured sustained matmul TFLOPs
    vram_bytes: int  # usable, after driver/context overhead
    dial_port: int = 0
    listen_port: int = 0
    label: str = ""
    # CUDA compute capability as (major, minor); None when unmeasured. The
    # planner uses the fleet minimum: a pipeline runs the same recipe on every
    # stage, so the weakest node decides what dtypes are available.
    compute_cap: tuple | None = None
    # torch device this node trains on (cuda | mps | cpu); None = the job's
    # device. A fleet mixing backends gets a per-slot device map in the spec.
    device: str | None = None
    # CONTROL-I25: the node's HELLO device class (probe.device_class); "" = none.
    device_class: str = ""
    # Plan mode for wide stages (SCHEDULER-I16, BA-48): the machine and site the
    # device sits in, its probed model, and what its links carry. None = not
    # measured; `sources[field]` says measured | declared | absent | missing.
    # host_dram, pcie and hbm in GB/s, nic in Gbit/s, per direction.
    hostname: str = ""  # the machine: one RAM and one DRAM figure; "" = host
    site: str = ""
    gpu_name: str = ""
    declared: tuple = ()
    host_ram_bytes: int | None = None
    host_dram_gbps: float | None = None
    pcie_h2d_gbps: float | None = None
    pcie_d2h_gbps: float | None = None
    nic_up_gbps: float | None = None
    nic_down_gbps: float | None = None
    hbm_gbps: float | None = None
    sources: dict = field(default_factory=dict)


@dataclass
class Fleet:
    nodes: list
    link_mbps: float  # measured, per direction
    rtt_ms: float
    duplex: str = "unknown"  # per_direction | aggregate | unknown
    # Host-DRAM passes per byte per side, (value, source); None = the
    # transport's counted figure (admission.K_HOST_DRAM_COUNTED, "counted").
    k_passes: tuple | None = None

    def effective_mbps(self) -> float:
        """Per-direction budget for boundary traffic.

        Forward activations and backward gradients travel opposite ways, so
        whether they compete depends on how the cap is enforced. On the one
        WAN pair measured, the simultaneous bidirectional sum exceeded either
        direction alone (so the cap was per direction) but the reverse path
        degraded about 4x under load, so "free" would overstate it. The
        unknown case takes a 0.75 haircut from that measurement.
        """
        if self.duplex == "per_direction":
            return self.link_mbps
        if self.duplex == "aggregate":
            return self.link_mbps / 2
        return self.link_mbps * 0.75

    def min_compute_cap(self):
        """Weakest node's capability, or None if any node is unmeasured.

        Any unknown makes the whole fleet unknown on purpose: taking the minimum
        of the known ones would let one unmeasured pre-Ampere box inherit an
        Ampere neighbour's capability and get a recipe it cannot run.
        """
        caps = [n.compute_cap for n in self.nodes]
        if not caps or any(c is None for c in caps):
            return None
        return min(tuple(c) for c in caps)


@dataclass
class Overrides:
    boundary_dtype: str | None = None
    compute_dtype: str | None = None
    mb_size: int | None = None
    microbatches: int | None = None
    partition: list | None = None
    data_shards: int | None = None
    recompute: bool | None = None
    attn_impl: str | None = None
    extras: dict = field(default_factory=dict)


def params_per_layer(cfg) -> float:
    """Transformer block parameter count for the declared architecture: the
    parameters one token's compute uses (the window estimate). (M4) An MoE
    block counts its active parameters (SCHEDULER-I21); an arch "hf" layer
    the largest census layer's linear MACs per token (SCHEDULER-I22)."""
    if autoconfig.is_hf(cfg):
        units = autoconfig.hf_census(cfg)[1:-1]
        return float(max((u["linear_macs_per_token"] or 0) for u in units)) if units else 0.0
    moe = autoconfig.moe_fields(cfg)
    if moe is not None:
        return float(
            autoconfig.builtin_attn_params(cfg)
            + autoconfig.moe_layer_costs(moe, cfg.d_model, cfg.seq_len, 1)["params_active"]
        )
    return _dense_block_params(cfg)


def _dense_block_params(cfg) -> float:
    d = cfg.d_model
    d_ff = cfg.d_ff or default_d_ff(cfg.mlp, d)
    attn = autoconfig.builtin_attn_params(cfg)
    mlp = (2 if cfg.mlp == "gelu" else 3) * d * d_ff
    return attn + mlp


def largest_grad_bytes(cfg) -> float:
    """RUNTIME-I21: the largest single fp32 gradient a width-1 stage of `cfg` holds beside its running sum (it
    folds each one as it is produced): the vocabulary table, or one block's largest weight, MoE experts included."""
    d = cfg.d_model
    d_ff = cfg.d_ff or default_d_ff(cfg.mlp, d)
    moe = autoconfig.moe_fields(cfg)
    expert = 0
    if moe is not None:
        f = cfg.moe_d_ff or default_d_ff("swiglu", d)
        expert = max(f, int(cfg.moe_shared_d_ff or 0)) * d
    return 4.0 * max(cfg.vocab_size * d, 3 * d * d, d_ff * d, expert)


def vram_reserve(vram: int) -> int:
    """12% plus a 1 GB floor: what no tensor model can see (cuBLAS workspaces,
    allocator fragmentation, codec staging buffers). 8% was measured too thin."""
    return max(int(vram * 0.12), 1024 * 2**20)


def largest_mb_size(
    cfg,
    params_stage,
    vram,
    k_inflight,
    boundary,
    n_layers_stage,
    flash,
    cap=64,
    head_vocab=0,
    frozen_params_stage=0,
    span: tuple | None = None,
    n_pend: int = 1,
    state_bpp: int = autoconfig.BYTES_PER_PARAM_FP32_ADAM,
) -> int:
    """Biggest microbatch whose stage footprint still fits, powers of two.
    Judged against vram minus vram_reserve(), not raw vram: the model prices
    the tensors it knows about, and a plan that lands at 99% of the card OOMs
    at step 0. `span` (the stage's entries) and `n_pend` add an arch "hf"
    stage's eager attention and split tied table (`_hf_extras`)."""
    reserve = vram_reserve(vram)
    best = 0
    mb = 1
    while mb <= cap:
        scores, extra = _hf_extras(cfg, span, mb, n_pend)
        m = autoconfig.stage_memory_bytes(
            params_stage,
            cfg.d_model,
            cfg.seq_len,
            mb,
            k_inflight,
            boundary,
            True,
            n_layers_stage,
            cfg.n_head,
            flash,
            head_vocab=head_vocab,
            frozen_params_stage=frozen_params_stage,
            attn_scores_bytes=scores,
            extra_bytes=extra,
            state_bytes_per_param=state_bpp,
            grad_transient_bytes=largest_grad_bytes(cfg),
        )
        if m["total"] > vram - reserve:
            break
        best, mb = mb, mb * 2
    return best


def tied_split(cfg, partition) -> str | None:
    """The tied table's name when the Embed (or HF prologue) and the Head that
    reuses it fall on different stages of `partition`, else None."""
    if autoconfig.is_hf(cfg):
        census = autoconfig.hf_census(cfg)
        tied = autoconfig.hf_tied(census)
        if tied is None:
            return None
        name, ends = tied["name"], tied["users"]
    else:
        if not cfg.tie_embeddings:
            return None
        name, ends = "tok.weight", (0, cfg.n_layer + 1)
    stage = [next(s for s, (a, b) in enumerate(partition) if a <= e < b) for e in ends]
    return name if stage[0] != stage[1] else None


# The rollback pieces whose target refuses a split tied hf table (BA-467;
# ADR-0009 round6.md section 4): the planner's own (scheduler-6) and the
# runtime's (7a, 7b), whose rollback leaves no relay to plan for.
TIED_PIECES = ("scheduler-6", "runtime-7a", "runtime-7b")


def tied_rollback() -> str | None:
    """The first piece of TIED_PIECES rolled back to 'tied-split-refused',
    read from model/rollback.py at call time, or None."""
    from wantrain.model import rollback

    for piece in TIED_PIECES:
        if rollback.target(piece) == "tied-split-refused":
            return piece
    return None


def tied_rollback_text(table: str, piece: str) -> str:
    return (
        f"{table}: tied table split across stages, refused while piece {piece!r} is rolled "
        f"back to 'tied-split-refused' (model/rollback.py)"
    )


def _hf_extras(cfg, span, mb_size: int, n_pend: int) -> tuple[float | None, float]:
    """(per-layer attention score bytes, extra bytes) of an arch "hf" stage
    over entries `span` (SCHEDULER-I22): the largest eager-attention term of
    its layers, and a split tied table's charges (`autoconfig.tied_memory`:
    the Head's held rows, stage 0's working copy and transient gradient).
    (None, 0.0) for every other arch or without a span."""
    if span is None or not autoconfig.is_hf(cfg):
        return None, 0.0
    a, b = span
    census = autoconfig.hf_census(cfg)
    units = autoconfig.hf_unit_costs(
        census, cfg.seq_len, mb_size, cfg.compute_dtype, top_k=autoconfig.hf_top_k(cfg)
    )
    scores = max((u["eager_bytes"] for u in units[a:b]), default=0.0)
    extra = 0.0
    tied = autoconfig.hf_tied(census)
    users = tied["users"] if tied else (0, 0)
    if tied is not None and not (a <= users[0] < b and a <= users[1] < b):
        h = autoconfig.hf_hidden(census)
        mem = autoconfig.tied_memory(tied["elems"] // h, h, mb_size * cfg.seq_len, n_pend)
        if a <= users[1] < b:
            extra += mem["head"]
        if a <= users[0] < b:
            extra += mem["stage0"]
    return scores, extra


def stage_param_split(cfg, a: int, b: int) -> tuple[int, int]:
    """(trainable, frozen) parameters of the layer range [a, b) of the flat
    list [Embed, Block*n, Head]. Without LoRA everything trains; with LoRA the
    base is frozen and only adapters train: rank r on a d_in x d_out
    projection adds r*(d_in + d_out); the default targets are the attention
    projections, `all` adds the MLP and the head.

    (M4) An arch "hf" stage holds its census units' parameters; a tied table
    split from stage 0 is trained on the Head, so it moves there with its
    state (stage 0 keeps a working copy, `_hf_extras`; SCHEDULER-I22). An MoE
    block holds all E experts (SCHEDULER-I21)."""
    if autoconfig.is_hf(cfg):
        params = _hf_unit_params(cfg, a, b)
        return int(sum(params[a:b])), 0
    ppl = params_per_layer(cfg)
    emb = cfg.vocab_size * cfg.d_model
    n_blocks = max(0, min(b, cfg.n_layer + 1) - max(a, 1))
    base = n_blocks * ppl
    moe = autoconfig.moe_fields(cfg)
    if moe is not None:
        dense = _dense_block_params(cfg)
        moe_blk = (
            autoconfig.builtin_attn_params(cfg)
            + autoconfig.moe_layer_costs(moe, cfg.d_model, cfg.seq_len, 1)["params_total"]
        )
        base = sum(
            moe_blk if autoconfig.is_moe_block(cfg, j - 1) else dense
            for j in range(max(a, 1), min(b, cfg.n_layer + 1))
        )
    if a == 0:
        base += emb
    if b == cfg.n_layer + 2 and not cfg.tie_embeddings:
        base += emb
    lora_rank = int(getattr(cfg, "lora_rank", 0) or 0)
    if not lora_rank:
        return int(base), 0
    d = cfg.d_model
    nkv = cfg.n_kv_head or cfg.n_head
    hd = d // cfg.n_head
    d_ff = cfg.d_ff or default_d_ff(cfg.mlp, d)
    targets = getattr(cfg, "lora_targets", "attn")
    per_block = lora_rank * ((d + d) + (d + 2 * nkv * hd) + (d + d))
    if targets in ("attn_mlp", "all"):
        n_mlp = 2 if cfg.mlp == "gelu" else 3
        per_block += lora_rank * n_mlp * (d + d_ff)
    t = int(n_blocks * per_block)
    if targets == "all" and b == cfg.n_layer + 2:
        t += lora_rank * (d + cfg.vocab_size)
    return t, int(base)


def stage_fits(
    cfg,
    a: int,
    b: int,
    vram: int,
    mb_size: int = 1,
    k_inflight: int = 1,
    boundary: str = "bf16",
    flash: bool = False,
    n_pend: int = 1,
    state_bpp: int = autoconfig.BYTES_PER_PARAM_FP32_ADAM,
) -> bool:
    """Does the layer range [a, b) fit a node with `vram` bytes, by the
    planner's memory model, at this microbatch size? Unknown memory fits.
    An arch "hf" stage is priced at its cut's dtype, with its eager attention
    and, for a split tied table, `n_pend` microbatches of held rows
    (SCHEDULER-I22)."""
    if not vram:
        return True
    if autoconfig.is_hf(cfg):
        boundary = autoconfig.hf_boundary_dtype(cfg)
    trainable, frozen = stage_param_split(cfg, a, b)
    scores, extra = _hf_extras(cfg, (a, b), mb_size, n_pend)
    mem = autoconfig.stage_memory_bytes(
        trainable,
        cfg.d_model,
        cfg.seq_len,
        mb_size,
        k_inflight,
        boundary,
        True,
        b - a,
        cfg.n_head,
        flash,
        head_vocab=cfg.vocab_size if b == cfg.n_layer + 2 else 0,
        frozen_params_stage=frozen,
        attn_scores_bytes=scores,
        extra_bytes=extra,
        state_bytes_per_param=state_bpp,
        grad_transient_bytes=largest_grad_bytes(cfg),
    )
    return mem["total"] <= vram - vram_reserve(vram)


def _plan_depth(
    model_cfg,
    fleet: Fleet,
    dataset: str | None = None,
    window_s_ref: float | None = None,
    ref_layers: int | None = None,
    policy: str = "balanced",
    margin: float = 2.0,
    k_inflight: int | None = None,
    steps: int = 100,
    lr: float = 2e-4,
    overrides: Overrides = None,
    require_margin: bool = True,
    device: str = "cuda",
    bubble_target: float = 0.05,
    tokens_per_step_cap: int | None = None,
    dataset_source: str | None = None,
) -> dict:
    """One node per stage (today's planner). -> {spec, rationale, report,
    warnings}. Raises only on impossible input."""
    ov = overrides or Overrides()
    warnings = []
    n_nodes = len(fleet.nodes)
    if n_nodes < 2:
        raise ValueError("need at least two nodes for a pipeline")
    hf = autoconfig.is_hf(model_cfg)
    if hf:
        # SCHEDULER-I22: priced from the census, at HF's residual dtype; a
        # missing census or a narrower declared cut is refused, naming it
        autoconfig.hf_census(model_cfg)
        autoconfig.hf_boundary_check(model_cfg, model_cfg.boundary_dtype)
        autoconfig.hf_boundary_check(model_cfg, ov.boundary_dtype)
    bnd0 = autoconfig.hf_boundary_dtype(model_cfg) if hf else (ov.boundary_dtype or "bf16")
    # a split tied table's rows stay pending on the Head for up to one
    # microbatch per stage in flight (1F1B's bound; the order is solved later)
    n_pend = n_nodes

    # ---- 2. capacity-proportional partition -------------------------------
    # Layers are priced by their real compute (the head of a large vocabulary
    # is several blocks) and a cut is only admissible if the stage it creates
    # fits its node's memory at the smallest microbatch; the microbatch size
    # is solved later against what is left.
    speeds = [n.tflops for n in fleet.nodes]
    attn_impl = ov.attn_impl or model_cfg.attn_impl
    flash = attn_impl == "sdpa" and model_cfg.sdpa_backend in (
        "flash",
        "efficient",
        "cudnn",
        "auto",
    )

    # RUNTIME-I20: optimizer "adamw-host" (job extras) keeps 8 bytes per trained element on the device, not 16.
    state_bpp = optim.state_bytes_per_param(ov.extras.get("optimizer", "adamw"))

    def _fits(s, a, b):
        return stage_fits(
            model_cfg,
            a,
            b,
            fleet.nodes[s].vram_bytes,
            flash=flash,
            boundary=bnd0,
            n_pend=n_pend,
            state_bpp=state_bpp,
        )

    if ov.partition:
        partition = ov.partition
    else:
        try:
            partition = proportional_partition(model_cfg, speeds, fits=_fits)
        except ValueError:
            partition = proportional_partition(model_cfg, speeds)
            warnings.append(
                "no layer cut fits every node's memory even at one sample per "
                "microbatch; planning the compute-balanced cut and letting the "
                "admission and fit checks report the overflow"
            )
    layers_per_stage = [b - a for a, b in partition]

    ppl = params_per_layer(model_cfg)
    # (trainable, frozen) per stage: with LoRA the base stays resident and
    # frozen and only the adapters are optimised, which shrinks the largest
    # memory term 4x; the microbatch solver, tightest-stage choice and
    # admission all have to see the split.
    split = [stage_param_split(model_cfg, a, b) for a, b in partition]
    stage_trainable = [t for t, _ in split]
    stage_frozen = [f for _, f in split]
    lora = bool(getattr(model_cfg, "lora_rank", 0))
    stage_params = stage_frozen if lora else stage_trainable

    def _state_bytes(i):
        return stage_trainable[i] * state_bpp + stage_frozen[i] * autoconfig.BYTES_PER_PARAM_FROZEN

    tight = min(range(n_nodes), key=lambda i: fleet.nodes[i].vram_bytes - _state_bytes(i))

    # ---- compute window -------------------------------------------------
    mb_ref = ov.mb_size or 1
    if window_s_ref is None:
        # derive from measured TFLOPs: 4x forward FLOPs covers fwd + recompute
        # + backward, at an assumed 30% achieved utilisation
        d, S = model_cfg.d_model, model_cfg.seq_len
        flops_tok = 4 * layers_per_stage[tight] * (6 * ppl / d * d + 4 * d * S)
        window_s = flops_tok * mb_ref * S / (fleet.nodes[tight].tflops * 1e12 * 0.30)
    else:
        if not ref_layers:
            raise ValueError("window_s_ref needs ref_layers (the layer count it was measured on)")
        window_s = autoconfig.stage_window_s_scaled(
            layers_per_stage[tight], ref_layers, window_s_ref
        )

    # ---- 2b. in-flight depth, from the measured latency -----------------
    # k hides latency, not bandwidth, so the only inputs are the round trip,
    # the transfer, and the compute window they have to hide behind.
    link_raw = fleet.effective_mbps()
    bnd_bytes_ref = mb_ref * model_cfg.seq_len * model_cfg.d_model * precision.spec(bnd0)["bytes"]
    k_sol = schedule.solve_k_inflight(
        window_s=window_s, rtt_ms=fleet.rtt_ms, boundary_bytes=bnd_bytes_ref, link_mbps=link_raw
    )
    k_inflight = k_inflight or k_sol["k"]

    # ---- 3+4. precision and microbatch, jointly -------------------------
    link = link_raw
    if fleet.duplex == "unknown":
        warnings.append(
            "link duplex is unmeasured, so the budget takes a 0.75 haircut. "
            "Run `iperf3 -c PEER --bidir -t 60`: if the simultaneous sum exceeds "
            "either one-way rate the cap is per-direction and the full rate "
            "applies. Measured that way on one Korea<->Quebec pair (30.5 fwd, "
            "51.7 rev, 42.4+13.8 bidir), but it is per-path and worth checking"
        )

    # The microbatch must fit on every stage, not just the one with the least
    # slack after optimizer state: the last stage carries the logits (mb x S x
    # vocab in fp32, twice), which grow with mb while optimizer state does not.
    def _mb_fit(s):
        return largest_mb_size(
            model_cfg,
            stage_trainable[s],
            fleet.nodes[s].vram_bytes,
            k_inflight,
            bnd0,
            layers_per_stage[s],
            flash,
            head_vocab=model_cfg.vocab_size if s == n_nodes - 1 else 0,
            frozen_params_stage=stage_frozen[s],
            span=tuple(partition[s]),
            n_pend=n_pend,
            state_bpp=state_bpp,
        )

    mb_per_stage = [_mb_fit(s) for s in range(n_nodes)]
    mb = ov.mb_size or min(mb_per_stage)
    if mb == 0:
        worst = min(range(n_nodes), key=lambda s: mb_per_stage[s])
        raise ValueError(
            f"stage {worst} cannot fit even one microbatch: "
            f"{_state_bytes(worst) / 2**30:.1f} GB of resident weights and "
            f"optimizer state alone vs "
            f"{fleet.nodes[worst].vram_bytes / 2**30:.1f} GB of VRAM. Use more "
            f"stages, a smaller model, or bigger GPUs"
        )

    # The window above was computed for `mb_ref` samples and the transfer
    # priced against it is for `mb`; rescale, or the admission test compares
    # one microbatch of wire against one sample of compute.
    tight_scores, tight_extra = _hf_extras(model_cfg, tuple(partition[tight]), mb, n_pend)
    p = autoconfig.plan(
        model_cfg,
        stage_trainable[tight],
        layers_per_stage[tight],
        link,
        window_s * mb / mb_ref,
        fleet.nodes[tight].vram_bytes,
        mb_size=mb,
        k_inflight=k_inflight,
        margin=margin,
        policy=policy,
        compute_dtype=ov.compute_dtype or model_cfg.compute_dtype,
        require_margin=require_margin,
        boundary_override=ov.boundary_dtype,
        flash_attention=flash,
        compute_cap=fleet.min_compute_cap(),
        head_vocab=model_cfg.vocab_size if tight == n_nodes - 1 else 0,
        frozen_params_stage=stage_frozen[tight],
        attn_scores_bytes=tight_scores,
        extra_bytes=tight_extra,
    )
    if p["chosen"] is None:
        return dict(spec=None, rationale=p["reason"], report=p["report"], warnings=warnings)

    boundary = p["chosen"]["boundary"]
    compute = ov.compute_dtype or p["chosen"]["requires_compute"] or model_cfg.compute_dtype
    if not flash and model_cfg.seq_len > 4096:
        warnings.append(
            f"seq_len={model_cfg.seq_len} with attn_impl={attn_impl!r} "
            f"materialises the score matrix "
            f"({p['chosen']['attn_score_gb']:.1f} GB/layer). Long context is the "
            f"cheapest way to widen the compute window and lower the bandwidth "
            f"floor, but it needs attn_impl='sdpa' with a flash/efficient "
            f"backend to be reachable"
        )

    # ---- 5. data shards -------------------------------------------------
    # The prepared shard count, when the corpus was prepared in shards, so every slot can hold
    # and verify only its own shard files; else capacity-proportional logical shards. The
    # manifest's digest pins the tokens (model/provision.py).
    meta = None
    if dataset:
        from wantrain.model import dataset as ds

        try:
            meta = ds.read_manifest(dataset)
        except (OSError, ValueError):
            meta = None
    prepared = int(meta.get("n_shards", 1)) if meta else 1
    shards = ov.data_shards or (prepared if prepared > 1 else (n_nodes if dataset else 1))
    if dataset and not ov.data_shards and prepared > 1:
        # data_shards is a recipe pin (the mb -> shard map, hence the data order). Before MODEL-I11
        # the default was the node count, so a corpus prepared in shards re-plans to another order.
        warnings.append(
            f"data_shards {shards} = the corpus's prepared shard count (a recipe pin: it sets which "
            f"shard each microbatch reads), so each slot holds and verifies only its own shard files; "
            f"set data_shards to choose another count"
        )
    if dataset and provision.manifest_digest(dataset) is None:
        warnings.append(
            f"{dataset!r} has no manifest, so the corpus is not pinned by content and no node can "
            f"verify it (MODEL-I11); prepare it with `wantrain prepare-data`"
        )

    # The full production stack comes from one file, not from a list of
    # keyword arguments someone has to remember. Operator overrides win, and
    # any flag they turn off is reported below rather than silently honoured.
    # Resolved here because the battery and the bubble model both read it.
    flags = production.spec_fields(overrides={"recompute": ov.recompute, **ov.extras})
    for dev in production.deviations(flags):
        warnings.append(f"production flag overridden: {dev}")

    # ---- 5b. microbatches per step, from the bubble you accept ----------
    # Solved after mb_size, because the token budget per step is a product of
    # the two. The bubble is the one the zero-bubble schedule leaves for the
    # model's F:B:W ratio and this link: one boundary transfer measured in
    # forward passes of the tightest stage (a quarter of its compute window).
    from wantrain.scheduler.zero_bubble import comm_seconds

    bnd_coded = (
        mb
        * model_cfg.seq_len
        * model_cfg.d_model
        * precision.spec(boundary)["bytes"]
        / precision.spec(boundary)["ratio"]
    )
    # SCHEDULER-I22: a split tied table adds |T_i|·h·4 B per microbatch on
    # every hop between stage 0 and the Head (the Head is the last stage, so
    # every hop), at its worst case |T_i| = min(V, T_mb) (ids are unknown here)
    table = tied_split(model_cfg, partition)
    tied_rows = min(model_cfg.vocab_size, mb * model_cfg.seq_len) if table else 0
    tied_fwd, _tied_bwd = autoconfig.tied_hop_bytes(tied_rows, tied_rows, model_cfg.d_model)
    t_f = max(1e-9, window_s * mb / mb_ref / 4.0)
    comm_ratio = comm_seconds(bnd_coded + tied_fwd, link, fleet.rtt_ms) / t_f
    if ov.microbatches:
        mb_sol: dict = dict(
            microbatches=ov.microbatches,
            bubble=None,
            family="operator",
            reason=f"microbatches={ov.microbatches} set by the operator",
        )
    else:
        mb_sol = schedule.solve_microbatches(
            n_stages=n_nodes,
            bubble_target=bubble_target,
            tokens_per_step_cap=tokens_per_step_cap,
            mb_size=mb,
            seq_len=model_cfg.seq_len,
            fbw=schedule.fbw_costs(model_cfg.d_model, model_cfg.seq_len),
            comm_ratio=comm_ratio,
        )

    # ---- assemble --------------------------------------------------------
    model_cfg.compute_dtype = compute
    model_cfg.boundary_dtype = boundary
    model_cfg.attn_impl = attn_impl
    model_cfg.validate()
    # BA-446: a non-hf tied table split across stages trains two copies; the
    # plan is refused naming it (plan_job marks every candidate), and the
    # draft below only carries the recipe the candidates are priced on. An hf
    # one is refused the same way while a tied piece is rolled back (BA-467).
    rolled = tied_rollback() if hf and table is not None else None
    refused_tied = table is not None and (not hf or rolled is not None)

    spec = JobSpec(
        model=model_cfg,
        lr=lr,
        device=device,
        dataset_path=dataset,
        dataset_sha256=provision.manifest_digest(dataset) if dataset else None,
        dataset_source=dataset_source,
        data_shards=shards,
        steps=steps,
        microbatches=mb_sol["microbatches"],
        mb_size=mb,
        stages=n_nodes,
        partition=partition,
        endpoints=[[[n.host, n.dial_port, n.listen_port]] for n in fleet.nodes]
        if any(n.dial_port for n in fleet.nodes)
        else None,
        devices=(
            [[n.device or device] for n in fleet.nodes]
            if any(n.device for n in fleet.nodes)
            else None
        ),
        **flags,
    )
    if not refused_tied:
        spec.finalize()

    rationale = dict(
        partition=partition,
        layers_per_stage=layers_per_stage,
        stage_params=stage_params,
        tightest_stage=tight,
        window_s=round(window_s, 4),
        effective_link_mbps=round(link, 2),
        precision=model_cfg.recipe(),
        mb_size=mb,
        data_shards=shards,
        decision=p["reason"],
        # The schedule the system solved for, and why. Without these the
        # operator cannot tell a tuned run from a defaulted one.
        microbatches=mb_sol["microbatches"],
        bubble=None if mb_sol["bubble"] is None else round(mb_sol["bubble"], 4),
        schedule_family=mb_sol.get("family", "1f1b"),
        microbatches_reason=mb_sol["reason"],
        k_inflight=k_inflight,
        k_reason=k_sol["reason"],
        tokens_per_step=mb_sol["microbatches"] * mb * model_cfg.seq_len,
        boundary_dtype=boundary,
    )
    if refused_tied:
        rationale["decision"] = "tied: " + (
            tied_rollback_text(str(table), rolled) if rolled else TIED_SPLIT_TEXT
        )
        return dict(
            spec=None, draft=spec, rationale=rationale, report=p["report"], warnings=warnings
        )
    if table is not None:
        flops = [n.tflops * 1e12 for n in fleet.nodes]
        rationale["tied"] = _tied_terms(
            model_cfg, partition, flops, spec, fleet, fleet.nodes[-1], table
        )
    return dict(spec=spec, rationale=rationale, report=p["report"], warnings=warnings)


# ============================================================ plan mode for wide stages
# SCHEDULER task 4 (SCHEDULER-I16, I17, I19; ADR-0006): the deepest cut at width 1,
# narrower and wider stages, zero1 wide stages and pieces where I16 admits them,
# each priced by the exchange simulation; the smallest simulated step wins.

WIDTHS = (2, 4, 8)
# (shard, wire) of a wide candidate. pieces needs native (finalize contract I-7).
LAYOUTS = (("pieces", "native"), ("zero1", "native"), ("zero1", "fp32"))
NOT_ARMED = "not armed"
# Emission orders: the simulation's names -> JobSpec.emission_order (BA-43).
EMISSION_SPEC = {"backward": "store", "forward": "forward"}


def max_depth(model_cfg, n_devices: int) -> int:
    """The deepest width-1 cut on n_devices: one stage per device up to the
    layer list's n_layer + 2 entries (today's planner). When there are more
    devices than entries, one stage per block, the Embed and the Head riding
    with the end blocks (the depth candidate ADR-0006 and E7 price)."""
    if n_devices <= model_cfg.n_layer + 2:
        return n_devices
    return model_cfg.n_layer


def plan_job(
    model_cfg,
    fleet: Fleet,
    dataset: str | None = None,
    window_s_ref: float | None = None,
    ref_layers: int | None = None,
    policy: str = "balanced",
    margin: float = 2.0,
    k_inflight: int | None = None,
    steps: int = 100,
    lr: float = 2e-4,
    overrides: Overrides = None,
    require_margin: bool = True,
    device: str = "cuda",
    bubble_target: float = 0.05,
    tokens_per_step_cap: int | None = None,
    allow_pieces: bool = False,
    shard: str | None = None,
    dataset_source: str | None = None,
) -> dict:
    """-> {spec, rationale, report, warnings, stage_widths, placement, shard,
    wire, emission_order, piece_bounds, stream_grads, candidates}.

    The width-1 plan is today's (`_plan_depth`) on the deepest cut. With four
    or more devices, wide candidates are priced too (SCHEDULER task 4): each
    (S, w) stage-contiguous (and straddling sites when there are several), as
    zero1 with the fp32 or native wire, and as pieces. With allow_pieces False
    every pieces or native candidate is refused "not armed" (BA-49). `shard`
    restricts the candidates to that layout. The smallest simulated step among
    the admitted candidates wins; the rationale lists every refused one with
    the link, number, class or measurement that refused it."""
    if shard not in (None, "zero1", "pieces"):
        raise ValueError(f"shard must be zero1 or pieces, got {shard!r}")
    n = len(fleet.nodes)
    if n < 2:
        raise ValueError("need at least two nodes for a pipeline")
    mixed = mixed_fleet((nd.label or nd.host, nd.device_class) for nd in fleet.nodes)
    if mixed is not None:
        raise ValueError(mixed)
    depth = max_depth(model_cfg, n)
    base_idx = sorted(sorted(range(n), key=lambda i: -fleet.nodes[i].tflops)[:depth])
    base_fleet = Fleet(
        nodes=[fleet.nodes[i] for i in base_idx],
        link_mbps=fleet.link_mbps,
        rtt_ms=fleet.rtt_ms,
        duplex=fleet.duplex,
        k_passes=fleet.k_passes,
    )
    out = _plan_depth(
        model_cfg,
        base_fleet,
        dataset=dataset,
        window_s_ref=window_s_ref,
        ref_layers=ref_layers,
        policy=policy,
        margin=margin,
        k_inflight=k_inflight,
        steps=steps,
        lr=lr,
        overrides=overrides,
        require_margin=require_margin,
        device=device,
        bubble_target=bubble_target,
        tokens_per_step_cap=tokens_per_step_cap,
        dataset_source=dataset_source,
    )
    out.update(
        stage_widths=[1] * depth,
        placement=[[_peer_record(fleet.nodes[i], i, device)] for i in base_idx],
        shard="zero1",
        wire="fp32",
        emission_order=None,
        piece_bounds=None,
        stream_grads=False,
        candidates=[],
    )
    # a BA-446 refusal leaves a draft spec: every candidate is refused naming
    # the tied table, and no draft leaves the planner. A split tied hf table
    # whose piece is rolled back (BA-467) is planned on one stage when one
    # fits, so its candidates are priced on any fleet.
    draft = out.pop("draft", None)
    one_stage = draft is not None and autoconfig.is_hf(model_cfg)
    if (out["spec"] is None and draft is None) or (n < 4 and shard != "pieces" and not one_stage):
        return _moe_compute_rationale(model_cfg, fleet, out)
    pinned = bool(overrides and overrides.partition)
    out["draft"] = draft
    res = _choose(model_cfg, fleet, out, base_idx, device, allow_pieces, shard, pinned)
    res.pop("draft", None)
    return _moe_compute_rationale(model_cfg, fleet, res)


def gemm_params_active(cfg, a: int, b: int) -> float:
    """P^F of the entries [a, b): the GEMM parameters one token's compute
    uses (DESIGN §9; ADR-0008 Σ(P_attn + P_r + k·3hf + P_sh) per MoE block):
    each block's (params_per_layer for an MoE block, the dense block's
    otherwise) and the Head's V·h; the Embed is a lookup."""
    moe = autoconfig.moe_fields(cfg) is not None
    total = 0.0
    for j in range(max(a, 1), min(b, cfg.n_layer + 1)):
        moe_blk = moe and autoconfig.is_moe_block(cfg, j - 1)
        total += params_per_layer(cfg) if moe_blk else _dense_block_params(cfg)
    if b == cfg.n_layer + 2:
        total += cfg.vocab_size * cfg.d_model
    return total


def _moe_compute_rationale(cfg, fleet: Fleet, out: dict) -> dict:
    """SCHEDULER-I21's rationale clause: per stage of the plan holding an MoE
    block, its analytic `c_s = 8·P^F_s·T_mb/F` next to the per-microbatch
    time the plan prices it at (F + B + W of `analytic_costs`, attention
    included, recompute as the recipe runs), both in seconds at F, the
    stage's profiled rate (its peers' mean measured FLOP/s)."""
    spec = out.get("spec")
    if spec is None or autoconfig.moe_fields(cfg) is None:
        return out
    tokens = spec.mb_size * cfg.seq_len
    rows = []
    for s, (a, b) in enumerate(spec.partition):
        if not any(
            autoconfig.is_moe_block(cfg, j - 1) for j in range(max(a, 1), min(b, cfg.n_layer + 1))
        ):
            continue
        peers = out["placement"][s]
        rate = sum(fleet.nodes[p["node"]].tflops for p in peers) / len(peers) * 1e12
        ns = types.SimpleNamespace(model=cfg, recompute=spec.recompute, partition=[(a, b)])
        flops = sum(x[0] for x in analytic_costs(ns)) * tokens
        rows.append(
            dict(
                stage=s,
                c_s_analytic=8 * gemm_params_active(cfg, a, b) * tokens / rate,
                t_profiled=flops / rate,
            )
        )
    out["rationale"]["moe_compute"] = rows
    return out


# ------------------------------------------------------------ inputs


def _peer_record(node: Node, i: int, device: str) -> dict:
    return {
        "node": i,
        "host": node.host,
        "hostname": node.hostname or node.host,
        "site": node.site,
        "endpoint": [node.host, node.dial_port, node.listen_port or node.dial_port],
        "device": node.device or device,
        "gpu_name": node.gpu_name,
    }


def _gpu(node: Node, i: int, device: str) -> admission.Gpu:
    """Node -> admission.Gpu: bytes/s per direction, each with its source.
    A cpu or mps device has no PCIe hop (absent, BA-48) and folds at its
    host's DRAM rate, so its HBM figure is the host's."""
    dev = node.device or device
    src = dict(node.sources)
    host_side = dev in ("cpu", "mps") and "device" not in node.declared
    pcie = {}
    for fld in ("pcie_h2d_gbps", "pcie_d2h_gbps"):
        val = getattr(node, fld)
        if val is None and host_side and src.get(fld) in (None, "absent"):
            src[fld] = "absent"
        pcie[fld] = None if val is None else val * 1e9
    hbm, hbm_src = node.hbm_gbps, src.get("hbm_gbps")
    if hbm is None and host_side:
        hbm, hbm_src = node.host_dram_gbps, src.get("host_dram_gbps")
    sources = {
        "nic_up": src.get("nic_up_gbps"),
        "nic_down": src.get("nic_down_gbps"),
        "pcie_h2d": src.get("pcie_h2d_gbps"),
        "pcie_d2h": src.get("pcie_d2h_gbps"),
        "hbm": hbm_src,
    }
    return admission.Gpu(
        name=f"{i}:{node.label or node.host}",
        host=node.hostname or node.host,
        site=node.site,
        device=dev,
        gpu_name=node.gpu_name,
        tflops=node.tflops,
        vram_bytes=node.vram_bytes,
        nic_up=None if node.nic_up_gbps is None else node.nic_up_gbps * 1e9 / 8,
        nic_down=None if node.nic_down_gbps is None else node.nic_down_gbps * 1e9 / 8,
        pcie_h2d=pcie["pcie_h2d_gbps"],
        pcie_d2h=pcie["pcie_d2h_gbps"],
        hbm=None if hbm is None else hbm * 1e9,
        declared=frozenset(node.declared),
        sources={k: v for k, v in sources.items() if v is not None},
    )


def _hosts(nodes: list) -> dict:
    """One RAM and one DRAM figure per machine: the largest of its devices'
    measurements (each is a lower bound), with that measurement's source."""
    out: dict = {}
    for node in nodes:
        name = node.hostname or node.host
        cur = out.setdefault(name, {"ram": (None, None), "dram": (None, None)})
        for key, fld in (("ram", "host_ram_bytes"), ("dram", "host_dram_gbps")):
            val = getattr(node, fld)
            if val is not None and (cur[key][0] is None or val > cur[key][0]):
                cur[key] = (val, node.sources.get(fld))
    hosts = {}
    for name, cur in out.items():
        ram, ram_src = cur["ram"]
        dram, dram_src = cur["dram"]
        sources = {}
        if ram_src:
            sources["ram_bytes"] = ram_src
        if dram_src:
            sources["dram"] = dram_src
        hosts[name] = admission.Host(
            name,
            None if ram is None else int(ram),
            None if dram is None else dram * 1e9,
            sources,
        )
    return hosts


def _arch_priced(cfg) -> str | None:
    """The exchange units below model the builtin block (gpt.Block: ln1, qkv,
    proj, ln2, fc1, fc2, each with a bias) and its parameter list. Anything
    else has no unit model, so its wide candidates are refused, never guessed."""
    checks = (
        ("arch", "builtin"),
        ("norm", "layernorm"),
        ("mlp", "gelu"),
        ("pos", "learned"),
        ("bias", True),
        ("lora_rank", 0),
    )
    for fld, want in checks:
        if getattr(cfg, fld, want) != want:
            return f"{fld}={getattr(cfg, fld)!r}"
    if cfg.d_ff not in (None, 4 * cfg.d_model):
        return f"d_ff={cfg.d_ff}"
    return None


# ------------------------------------------------------------ one stage


def _stage_units(cfg, a: int, b: int, b_g: int, b_p: int) -> tuple[tuple, list, int]:
    """The exchange units of layers [a, b) in forward order (SCHEDULER-I17):
    the Embed's tables as one fp32 unit with no W compute, then each block's
    SplitLinear modules, then the Head's linear (tied: the Embed's). Returns
    (units, each unit's whole tensors as (elements, today's zero1 stride
    index), blocks): the Embed is tok (index 0) and pos (index 1), which the
    runtime's params[peer::w] puts on different peers. Norms and biases are
    left out: 13h of 12h^2 + 13h parameters per block.

    (M4) An MoE block (SCHEDULER-I21, SPEC-runtime task 5) has the runtime's
    units exactly (`moe_block_units`, BA-466): the small-tensor unit (norms,
    router and the selection biases, with the count lane's E·8 bytes per
    microbatch), the attention's linears, one unit per expert projection
    (gate, up, down: all E experts, compute weighted at k), and the shared
    expert's projections and gate. An arch "hf" stage (width 1 only, BA-466)
    has one unit per census unit, for its fold and AdamW time."""
    if autoconfig.is_hf(cfg):
        return _hf_stage_units(cfg, a, b, b_g, b_p)
    h, v = cfg.d_model, cfg.vocab_size
    nkv = cfg.n_kv_head or cfg.n_head
    qkv = h * (h + 2 * nkv * (h // cfg.n_head))
    mods = ((qkv, 2), (h * h, 4), (h * 4 * h, 8), (4 * h * h, 10))
    moe = autoconfig.moe_fields(cfg)
    units: list = []
    idx: list = []
    base = 0
    if a == 0:
        units.append(Unit((v + cfg.seq_len) * h, 4, 4, 0, 0.0))
        idx.append(((v * h, 0), (cfg.seq_len * h, 1)))
        base = 2
    blocks = range(max(a, 1), min(b, cfg.n_layer + 1))
    off = base
    biases: list = []  # (small-tensor unit position, E): the selection biases
    for blk in blocks:
        if moe is None or not autoconfig.is_moe_block(cfg, blk - 1):
            for p, i in mods:
                units.append(Unit(p, b_g, b_p, len(units), float(p)))
                idx.append(((p, off + i),))
            off += 12
            continue
        off = _moe_units(cfg, moe, units, idx, biases, off, b_g, b_p)
    if b == cfg.n_layer + 2:
        if not cfg.tie_embeddings:
            units.append(Unit(v * h, b_g, b_p, len(units), float(v * h)))
            idx.append(((v * h, off + 2),))
        off += 2 if cfg.tie_embeddings else 3  # the Head's norm (and linear)
    # ZeRO-1 appends the selection biases after the trainable parameters
    # (index n_params + j, HANDOUT-M4 I-12)
    for j, (u, e) in enumerate(biases):
        idx[u] = (*idx[u], (e, off + j))
    return tuple(units), idx, len(blocks)


def moe_block_units(cfg, moe: dict) -> tuple[list, list]:
    """One MoE block of the builtin attention (arch.Block with arch.MoEMLP) as
    the runtime groups it (pieces.exchange_units, BA-466): -> (units, the
    block's parameter names in named_parameters order). Each unit is a list
    of (name inside the block, elements), in exchange order: the small
    tensors (norms, router, then the selection biases `mlp.sel_bias`, a
    buffer), the attention's linears, each expert projection over all E
    experts in expert order, the shared expert's projections and its gate."""
    h, nh = cfg.d_model, cfg.n_head
    nkv = cfg.n_kv_head or nh
    E, f = int(moe["moe_experts"]), int(moe["moe_d_ff"])
    f_sh = int(moe.get("moe_shared_d_ff", 0) or 0)
    clamped = moe.get("moe_expert") == "swiglu_clamped"

    def linear(name, d_in, d_out, bias):
        return [(f"{name}.weight", d_in * d_out)] + ([(f"{name}.bias", d_out)] if bias else [])

    def norm(name):
        return [(f"{name}.weight", h)] + ([(f"{name}.bias", h)] if cfg.norm == "layernorm" else [])

    if nkv == nh:
        attn = [linear("qkv", h, 3 * h, cfg.bias)]
    else:
        attn = [linear("q", h, h, cfg.bias), linear("kv", h, 2 * nkv * (h // nh), cfg.bias)]
    proj = linear("proj", h, h, cfg.bias)
    router = linear("mlp.router", h, E, bool(moe.get("moe_router_bias")))
    shapes = (("gate", h, f), ("up", h, f), ("down", f, h))
    experts = [
        [linear(f"mlp.experts.{e}.{p}", i, o, clamped) for e in range(E)] for p, i, o in shapes
    ]
    shared = (
        [
            linear(f"mlp.shared_expert.{p}", i, o, False)
            for p, i, o in (("gate", h, f_sh), ("up", h, f_sh), ("down", f_sh, h))
        ]
        if f_sh
        else []
    )
    gated = f_sh and moe.get("moe_shared_gate", "sigmoid") == "sigmoid"
    gate = [linear("mlp.shared_expert_gate", h, 1, False)] if gated else []
    order = norm("ln1") + sum(attn, []) + proj + norm("ln2") + router
    for e in range(E):
        order += sum((experts[p][e] for p in range(3)), [])
    order += sum(shared + gate, [])
    small = norm("ln1") + norm("ln2") + router + [("mlp.sel_bias", E)]
    units = [small, *attn, proj, *(sum(g, []) for g in experts), *shared, *gate]
    return units, [n for n, _e in order]


def _moe_units(cfg, moe: dict, units: list, idx: list, biases: list, off: int, b_g, b_p) -> int:
    """Append one MoE block's units (`moe_block_units`) from its first
    parameter's index `off` in the stage's list; -> the next block's. Compute
    weights: the router and every linear at its weights, a routed expert
    projection at k of E; the selection biases join ZeRO-1's ownership list
    after the parameters (appended by the caller)."""
    E, k = int(moe["moe_experts"]), int(moe.get("moe_top_k", 1))
    block, order = moe_block_units(cfg, moe)
    pos = {name: off + i for i, name in enumerate(order)}
    for u, tensors in enumerate(block):
        params = sum(n for _name, n in tensors)
        weights = float(
            sum(
                n
                for name, n in tensors
                if name.endswith(".weight") and not name.startswith(("ln1.", "ln2."))
            )
        )
        routed = any(name.startswith("mlp.experts.") for name, _n in tensors)
        if u == 0:
            biases.append((len(units), E))
        units.append(
            Unit(
                params,
                b_g,
                b_p,
                len(units),
                k * weights / E if routed else weights,
                E * autoconfig.COUNT_LANE_BYTES if u == 0 else 0,
            )
        )
        idx.append(tuple((n, pos[name]) for name, n in tensors if name in pos))
    return off + len(order)


def _hf_unit_params(cfg, a: int, b: int) -> list:
    """Trainable parameters per census unit as a stage [a, b) holds them: a
    shared tensor counts at its first unit (I-10), except a tied table split
    across stages, which counts at the Head that trains it (SCHEDULER-I22)."""
    census = autoconfig.hf_census(cfg)
    params = [int(u["params"]) for u in census]
    tied = autoconfig.hf_tied(census)
    if tied is not None:
        first, head = tied["users"]
        if (a <= first < b) != (a <= head < b):
            params[first] -= tied["elems"]
            params[head] += tied["elems"]
    return params


def _hf_stage_units(cfg, a: int, b: int, b_g: int, b_p: int) -> tuple[tuple, list, int]:
    census = autoconfig.hf_census(cfg)
    costs = autoconfig.hf_unit_costs(
        census, cfg.seq_len, 1, cfg.compute_dtype, top_k=autoconfig.hf_top_k(cfg)
    )
    params = _hf_unit_params(cfg, a, b)
    units: list = []
    idx: list = []
    for i in range(a, b):
        if params[i] <= 0:
            continue
        first = i == 0
        units.append(
            Unit(
                params[i],
                4 if first else b_g,
                4 if first else b_p,
                len(units),
                0.0 if first else float(costs[i]["cost"]),
            )
        )
        idx.append(((params[i], len(idx)),))
    return tuple(units), idx, max(0, min(b, len(census) - 1) - max(a, 1))


class _Ctx:
    """What every candidate of one job shares."""

    def __init__(self, cfg, fleet: Fleet, base: dict, device: str):
        spec = base["spec"] if base.get("spec") is not None else base["draft"]
        self.cfg = cfg
        self.fleet = fleet
        self.device = device
        self.microbatches = spec.microbatches
        self.mb_size = spec.mb_size
        self.tokens = spec.mb_size * cfg.seq_len
        self.recompute = spec.recompute
        self.native_bytes = precision.spec(cfg.compute_dtype)["bytes"]
        self.bnd_bytes = precision.spec(cfg.boundary_dtype)["bytes"]
        self.tensor = self.tokens * cfg.d_model * self.bnd_bytes  # one boundary tensor
        counted = admission.counted_k(spec.adapt_budget_ms)
        k = fleet.k_passes or (counted, "counted")
        self.k, self.k_source = float(k[0]), str(k[1])
        if self.k_source == "counted" and self.k != counted:
            # "counted" is the transport's own figure (BA-48); any other value is a claim
            self.k_source = "declared"
        self.order = sorted(
            range(len(fleet.nodes)),
            key=lambda i: (fleet.nodes[i].site, fleet.nodes[i].hostname or fleet.nodes[i].host),
        )
        self.gpus = {i: _gpu(fleet.nodes[i], i, device) for i in range(len(fleet.nodes))}
        self.hosts = _hosts(fleet.nodes)
        self.sites = sorted({fleet.nodes[i].site for i in self.order})
        self.uplink = fleet.link_mbps * 1e6 / 8  # bytes/s per direction between sites
        multi = len(self.sites) > 1
        self.rtt = 0.0 if multi else fleet.rtt_ms / 1e3
        self.site_rtt = fleet.rtt_ms / 1e3 if multi else 0.0
        self.arch = _arch_priced(cfg)
        self.flash = cfg.attn_impl == "sdpa" and cfg.sdpa_backend in (
            "flash",
            "efficient",
            "cudnn",
            "auto",
        )
        self.k_inflight = base["rationale"]["k_inflight"]
        self.hf = autoconfig.is_hf(cfg)
        # the Embed (prologue) is the first entry and the Head the last, so
        # every cut with S > 1 splits a tied table
        if self.hf:
            table = autoconfig.hf_tied(autoconfig.hf_census(cfg))
            self.tied = table is not None
            self.tied_name = table["name"] if table else ""
        else:
            self.tied = bool(cfg.tie_embeddings)
            self.tied_name = "tok.weight"
        # BA-467: a rolled-back tied piece refuses every split of an hf table,
        # and the job is planned on one stage when one fits
        self.tied_rolled = tied_rollback() if self.hf and self.tied else None

    def costs(self, a: int, b: int) -> tuple[float, float, float]:
        """F, B, W FLOPs of one microbatch on layers [a, b) (analytic_costs,
        attention included, recompute as the recipe runs; ADR-0006 R16)."""
        ns = types.SimpleNamespace(model=self.cfg, recompute=self.recompute, partition=[(a, b)])
        f, bb, w = analytic_costs(ns)
        return f[0] * self.tokens, bb[0] * self.tokens, w[0] * self.tokens

    def stage(self, s: int, n_stages: int, ab: tuple, peers: list, shard: str, wire: str):
        a, b = ab
        b_g = self.native_bytes if wire == "native" else 4
        b_p = self.native_bytes if wire == "native" else 4
        units, idx, blocks = _stage_units(self.cfg, a, b, b_g, b_p)
        w = len(peers)
        owners = pieces_owners(len(units), w) if shard == "pieces" else stride_owners(idx, w)
        f, bb, ww = self.costs(a, b)
        last = b == self.cfg.n_layer + 2
        pending = None
        if autoconfig.moe_fields(self.cfg) is not None:
            ns = types.SimpleNamespace(model=self.cfg, mb_size=self.mb_size, recompute=True)
            pending = stage_memory_model(ns, b - a, (a, b))[1]
        return admission.Stage(
            index=s,
            units=units,
            owners=owners,
            peers=tuple(self.gpus[i] for i in peers),
            flops_mb=f + bb + ww,
            boundary_mb=self.tensor * (1 if s in (0, n_stages - 1) else 2),
            tokens_mb=self.tokens,
            d_model=self.cfg.d_model,
            n_blocks=blocks,
            boundary_bytes=self.bnd_bytes,
            logits_bytes=2 * self.tokens * self.cfg.vocab_size * admission.LOGITS_BYTES
            if last
            else 0.0,
            layout=shard,
            pending_w_mb=pending,
        )


# ------------------------------------------------------------ candidates


def _placements(ctx: _Ctx, n_stages: int, w: int) -> list:
    """[(name, [[node index per peer] per stage])]: stage-contiguous packing
    in site and host order, and with several sites, every stage straddling
    them (its peers dealt across the sites in turn). Devices left over are
    the slowest (ties: the last in site and host order), as at width 1."""

    def fastest(pool: list, k: int) -> list:
        keep = set(sorted(pool, key=lambda i: -ctx.fleet.nodes[i].tflops)[:k])
        return [i for i in pool if i in keep]

    used = fastest(ctx.order, n_stages * w)
    out = [("contiguous", [used[s * w : (s + 1) * w] for s in range(n_stages)])]
    if len(ctx.sites) > 1 and w % len(ctx.sites) == 0:
        need = n_stages * w // len(ctx.sites)
        per_site = {
            site: fastest([i for i in ctx.order if ctx.fleet.nodes[i].site == site], need)
            for site in ctx.sites
        }
        if all(len(v) >= need for v in per_site.values()):
            pos = dict.fromkeys(ctx.sites, 0)
            rows = []
            for _s in range(n_stages):
                row = []
                for q in range(w):
                    site = ctx.sites[q % len(ctx.sites)]
                    row.append(per_site[site][pos[site]])
                    pos[site] += 1
                rows.append(row)
            out.append(("straddling", rows))
    return out


def _candidates(ctx: _Ctx, base_idx: list) -> list:
    """Every candidate this fleet can field, width 1 (today's) first."""
    n = len(ctx.fleet.nodes)
    out: list[dict[str, Any]] = [
        dict(
            S=len(base_idx),
            w=1,
            shard="zero1",
            wire="fp32",
            placement="contiguous",
            rows=[[i] for i in base_idx],
        )
    ]
    if ctx.tied_rolled:
        # the fastest node that holds the whole model (BA-467: planned on one
        # stage when one fits); the fastest of all, refused MEM, when none does
        by_speed = sorted(range(n), key=lambda i: (-ctx.fleet.nodes[i].tflops, i))
        node = next((i for i in by_speed if _one_stage_fits(ctx, i)), by_speed[0])
        out.append(
            dict(S=1, w=1, shard="zero1", wire="fp32", placement="contiguous", rows=[[node]])
        )
    for w in WIDTHS:
        n_stages = max_depth(ctx.cfg, n // w)
        if n_stages < 2:
            continue
        for pname, rows in _placements(ctx, n_stages, w):
            for shard, wire in LAYOUTS:
                out.append(
                    dict(S=n_stages, w=w, shard=shard, wire=wire, placement=pname, rows=rows)
                )
    for c in out:
        tag = "" if c["placement"] == "contiguous" else f", {c['placement']}"
        lay = "" if c["w"] == 1 else f" {c['shard']} + {c['wire']}"
        c["name"] = f"S {c['S']} w {c['w']}{lay}{tag}"
        c["gpus_used"] = c["S"] * c["w"]
        c["stream_grads"] = c["w"] > 1
        c["piece_bounds"] = None
        c["emission_order"] = None
        c["classes"] = [
            [[ctx.gpus[i].device, ctx.gpus[i].gpu_name] for i in row] for row in c["rows"]
        ]
        c["tied_cross_stage"] = ctx.tied and c["S"] > 1
        c["reasons"] = []
        c["step_s"] = None
    return out


def _refusals_before_pricing(ctx: _Ctx, c: dict, allow_pieces: bool, shard: str | None) -> list:
    """Reasons that need no simulation: tied weights, arming, the requested
    layout, the class, the unit model, M, and missing measurements. The tied
    checks come first, ahead of the width-1 return (BA-446): a non-hf tied
    table split across stages at any width, and an hf one with any stage
    from stage 0 to the Head wider than 1 (BA-447)."""
    out = _tied_refusals(ctx, c)
    if c["w"] == 1:
        if shard == "pieces":
            out.append(admission.Reason("shard", "width 1 has no pieces layout"))
        if c["S"] == 1 and not out:
            out += _one_stage_memory(ctx, c)
        return out
    if not allow_pieces and (c["shard"] == "pieces" or c["wire"] == "native"):
        return [*out, admission.Reason(NOT_ARMED, f"{c['shard']} + {c['wire']}")]
    if shard is not None and c["shard"] != shard:
        return [admission.Reason("shard", f"the job asks for shard {shard}")]
    pieces = c["shard"] == "pieces"
    for s, row in enumerate(c["rows"]):
        for r in admission.check_class([ctx.gpus[i] for i in row], probed=pieces):
            out.append(dataclasses.replace(r, where=f"stage {s}: {r.where}", stage=s))
    if ctx.arch:
        out.append(admission.Reason("unpriced", f"no exchange-unit model for {ctx.arch}"))
    if ctx.microbatches < c["w"]:
        out.append(
            admission.Reason("M", "microbatches per step", float(ctx.microbatches), float(c["w"]))
        )
    return out


TIED_SPLIT_TEXT = "tok.weight: tie_embeddings with the Embed and the Head on different stages"


def tied_width_refusals(widths, table: str, head: int | None = None) -> list:
    """BA-447: a tied table split between stage 0 and the Head is relayed
    (TIED) only along a width-1 chain, so every stage from stage 0 to the
    Head (the last stage unless `head` says otherwise) at width > 1 is
    refused, naming the stage and the table."""
    last = len(widths) - 1 if head is None else head
    return [
        admission.Reason(
            "tied",
            f"stage {s} width {w}: {table} is split between stage 0 and the Head, and its "
            f"TIED relay needs width 1 on every stage between (BA-447)",
            stage=s,
        )
        for s, w in enumerate(widths[: last + 1])
        if w > 1
    ]


def _tied_refusals(ctx: _Ctx, c: dict) -> list:
    if not c["tied_cross_stage"]:
        return []
    if not ctx.hf:
        return [admission.Reason("tied", TIED_SPLIT_TEXT)]  # BA-446
    if ctx.tied_rolled:
        return [admission.Reason("tied", tied_rollback_text(ctx.tied_name, ctx.tied_rolled))]
    return tied_width_refusals([c["w"]] * c["S"], ctx.tied_name)


def _one_stage_memory(ctx: _Ctx, c: dict) -> list:
    """The one-stage candidate holds every entry on its node, at the recipe's
    microbatch (stage_fits, the width-1 memory model)."""
    i = c["rows"][0][0]
    if _one_stage_fits(ctx, i):
        return []
    return [admission.Reason("MEM", f"stage 0 peer {ctx.gpus[i].name}: the whole model", stage=0)]


def _one_stage_fits(ctx: _Ctx, i: int) -> bool:
    return stage_fits(
        ctx.cfg,
        0,
        ctx.cfg.n_layer + 2,
        ctx.gpus[i].vram_bytes,
        mb_size=ctx.mb_size,
        k_inflight=ctx.k_inflight,
        boundary=ctx.cfg.boundary_dtype,
        flash=ctx.flash,
    )


def _tied_terms(cfg, partition, flops, spec, fleet: Fleet, head_node: Node, table: str) -> dict:
    """SCHEDULER-I22's charges of an hf tied table split across stages, for
    the rationale and the step: the hops between stage 0 and the Head, their
    tied bytes, and ΔT_tied's five terms (autoconfig.delta_t_tied) at the
    worst-case rows |T_i| = |R_k| = min(V, T_mb) (the ids are unknown at
    planning), stage 0's per-microbatch times at its node's rate (`flops`,
    FLOP/s per stage), the link's effective rate and one-way latency, and the
    Head device's memory bandwidth (unmeasured: inf, those terms 0)."""
    census = autoconfig.hf_census(cfg)
    h = autoconfig.hf_hidden(census)
    tied = autoconfig.hf_tied(census)
    if tied is None:
        raise ValueError(f"{table}: no tied table in hf_census")
    vocab = tied["elems"] // h
    tokens = spec.mb_size * cfg.seq_len
    ns = types.SimpleNamespace(model=cfg, recompute=spec.recompute, partition=[partition[0]])
    t_f, t_b, t_w = (x[0] * tokens / flops[0] for x in analytic_costs(ns))
    rows = min(vocab, tokens)
    fwd, bwd = autoconfig.tied_hop_bytes(rows, rows, h)
    mem_bw = (head_node.hbm_gbps or head_node.host_dram_gbps or math.inf) * 1e9
    terms = autoconfig.delta_t_tied(
        t_f + t_b + t_w,
        t_f,
        t_b,
        [rows] * spec.microbatches,
        [rows] * spec.microbatches,
        len(partition) - 1,
        fleet.rtt_ms / 2000.0,
        fleet.effective_mbps() * 1e6 / 8,
        d_model=h,
        vocab_size=vocab,
        tokens_mb=tokens,
        mem_bw_Bps=mem_bw,
    )
    return dict(
        table=table,
        hops=[[s, s + 1] for s in range(len(partition) - 1)],
        fwd_bytes_per_mb=fwd,
        bwd_bytes_per_step=bwd,
        rows_assumed=rows,
        mem_bw_Bps=None if math.isinf(mem_bw) else mem_bw,
        delta_t_tied=terms,
    )


def _price(ctx: _Ctx, c: dict, check: bool) -> tuple[float, list, dict]:
    """Simulated step of a candidate -> (step seconds, I16 refusals, detail).

    The orders are the engine bridge's (lane-consistent, ZB-1p memory) on the
    planner's costs; wide stages run the exchange simulation per connected
    group of stages (stages that share a host or an uplink), both emission
    orders, the faster per group; step = period + max_s exposed_s + max_s
    fold and AdamW time, bounded below by every host's DRAM time and every
    uplink's boundary volume (`_exposed`)."""
    cfg, S, w, M = ctx.cfg, c["S"], c["w"], ctx.microbatches
    speeds = [sum(ctx.gpus[i].tflops for i in row) for row in c["rows"]]
    part = proportional_partition(cfg, speeds)
    # a width-1 stage folds its microbatches in the recipe's GEMM dtype
    wire = "native" if w == 1 else c["wire"]
    stages = [ctx.stage(s, S, part[s], c["rows"][s], c["shard"], wire) for s in range(S)]
    weights = [[ctx.gpus[i].tflops for i in row] for row in c["rows"]]
    owner = [iwrr(M, wt) for wt in weights]
    owned = admission.mb_owned_from(owner, stages)
    costs = [ctx.costs(a, b) for a, b in part]
    mems = [
        stage_memory_model(
            types.SimpleNamespace(model=cfg, mb_size=ctx.mb_size, recompute=ctx.recompute),
            b - a,
            (a, b),
        )
        for a, b in part
    ]
    agg = [sp * 1e12 for sp in speeds]
    comm, floor = _hops(ctx, c, stages)
    table = tied_split(cfg, part) if ctx.hf else None
    tied = None
    if table is not None:
        # SCHEDULER-I22: TIED bytes on every hop between stage 0 and the Head
        spec = types.SimpleNamespace(mb_size=ctx.mb_size, recompute=ctx.recompute, microbatches=M)
        head = ctx.fleet.nodes[c["rows"][-1][0]]
        tied = _tied_terms(cfg, part, agg, spec, ctx.fleet, head, table)
        rate = ctx.fleet.effective_mbps() * 1e6 / 8
        comm = [x + tied["fwd_bytes_per_mb"] / rate for x in comm]
    zcfg = ZBConfig(
        S,
        M,
        [x[0] / g for x, g in zip(costs, agg)],
        [x[1] / g for x, g in zip(costs, agg)],
        [x[2] / g for x, g in zip(costs, agg)],
        comm=comm,
        mem_f=[x[0] for x in mems],
        mem_b=[x[1] for x in mems],
        mem_w=[x[2] for x in mems],
    )
    orders, _sched, period = bridge_orders(zcfg, owner, [w] * S, weights, "1p")
    detail: dict = {"period_s": period, "C_s": admission.pipeline_compute(stages, M)}
    reasons: list = []
    if w == 1:
        ovh = max(
            (admission.fold_overhead(st, 0, M) for st in stages if st.peers[0].hbm), default=0.0
        )
        expo, emission = 0.0, None
    else:
        preds, emission = _exchange(ctx, zcfg, orders, weights, stages, c)
        expo = max(_exposed(p) for p in preds.values())
        ovh = max(p.overhead for p in preds.values())
        detail["E_s"] = {s: p.exposure for s, p in preds.items()}
        detail["window_s"] = {s: p.window for s, p in preds.items()}
        if check:
            pieces = c["shard"] == "pieces"
            if pieces:
                reasons += admission.s1(stages, ctx.hosts, ctx.k, _uplinks(ctx))
                reasons += admission.s2(preds, detail["C_s"])
                reasons += admission.device_memory(stages, owned, S, M)
                # SCHEDULER-I16: MEM-HOST admits pieces; zero1 keeps today's admission (task 4)
                reasons += admission.mem_host(stages, ctx.hosts, owned, M)
            else:
                reasons += _zero1_memory(ctx, c, part)
    host_t = _host_time(ctx, stages, owned, M)
    step = max(period + expo + ovh, host_t, floor)
    if tied is not None:
        step += tied["delta_t_tied"]["total"]
        detail["tied"] = tied
    detail.update(exposure_s=expo, overhead_s=ovh, host_s=host_t, uplink_floor_s=floor)
    c["emission_order"] = None if emission is None else [EMISSION_SPEC[emission]] * S
    return step, reasons, detail


def _exposed(p) -> float:
    """A stage's exchange exposure beyond the period: the tail X_s less the
    idle window_s before step t+1's forward, or the delay of that forward
    beyond its slot, which is already measured against the slot (the window
    is inside it and is not subtracted again). One definition with S2."""
    return admission.exposed_beyond_window(p)


def _uplinks(ctx: _Ctx) -> dict:
    return {(a, b): ctx.uplink for a in ctx.sites for b in ctx.sites if a != b}


def _hops(ctx: _Ctx, c: dict, stages: list) -> tuple[list, float]:
    """Per hop: one boundary tensor at the slower NIC of the two stages plus
    half the round trip; a hop between sites carries every lane over the one
    uplink (w tensors) and its round trip. -> (comm, the step's floor from the
    busiest uplink: M tensors per direction plus one round trip)."""
    comm = []
    cross = 0
    for s in range(c["S"] - 1):
        a, b = stages[s], stages[s + 1]
        if {g.site for g in a.peers} != {g.site for g in b.peers}:
            comm.append(c["w"] * ctx.tensor / ctx.uplink + ctx.site_rtt / 2)
            cross += 1
            continue
        rates = [g.nic_up for g in a.peers] + [g.nic_down for g in b.peers]
        known = [r for r in rates if r]
        rate = min(known) if known else ctx.fleet.effective_mbps() * 1e6 / 8
        comm.append(ctx.tensor / rate + ctx.rtt / 2)
    floor = (ctx.microbatches * ctx.tensor / ctx.uplink + ctx.site_rtt) if cross else 0.0
    return comm, floor


def _host_time(ctx: _Ctx, stages: list, owned: dict, M: int) -> float:
    """The busiest host's DRAM time per step: k passes per byte per side over
    every byte its devices send and receive (exchange, gathers, boundaries)."""
    load: dict = {}
    for st in stages:
        w = st.width
        for q, g in enumerate(st.peers):
            m_k = owned[st.index][q]
            byt = 2 * m_k * st.boundary_mb
            if w > 1:
                for u, row in zip(st.units, st.owners):
                    o = row[q]
                    byt += u.grad_bytes() * (o * (M - m_k) + (1 - o) * m_k)
                    byt += u.b_p * u.params * (o * (w - 1) + (1 - o))
            load[g.host] = load.get(g.host, 0.0) + ctx.k * byt
    worst = 0.0
    for h, byt in load.items():
        dram = ctx.hosts[h].dram if h in ctx.hosts else None
        if dram:
            worst = max(worst, byt / dram)
    return worst


def _exchange(ctx: _Ctx, zcfg, orders, weights, stages: list, c: dict) -> tuple[dict, str]:
    """SCHEDULER-I17 over every wide stage, both emission orders; the one with
    the smaller predicted step, one for the whole plan."""
    fabric = Fabric(
        host_dram={h: (x.dram or math.inf) for h, x in ctx.hosts.items()},
        uplinks=_uplinks(ctx),
        k_passes=ctx.k,
        rtt=ctx.rtt,
        site_rtt={(a, b): ctx.site_rtt for a in ctx.sites for b in ctx.sites if a != b},
    )
    best: dict = {}
    layouts = {st.index: admission.exchange_layout(st) for st in stages}
    for emission in EMISSIONS:
        preds = simulate_exchange(zcfg, orders, weights, layouts, fabric, emission=emission).stages
        cost = max(_exposed(p) for p in preds.values()) + max(p.overhead for p in preds.values())
        if not best or cost < best["cost"]:
            best = {"cost": cost, "preds": preds, "emission": emission}
    return best["preds"], best["emission"]


def _zero1_memory(ctx: _Ctx, c: dict, part: list) -> list:
    """zero1 keeps today's device admission (stage_fits, 16 B/param)."""
    out = []
    for s, (a, b) in enumerate(part):
        for i in c["rows"][s]:
            vram = ctx.gpus[i].vram_bytes
            if not stage_fits(
                ctx.cfg,
                a,
                b,
                vram,
                mb_size=ctx.mb_size,
                k_inflight=ctx.k_inflight,
                boundary=ctx.cfg.boundary_dtype,
                flash=ctx.flash,
            ):
                out.append(admission.Reason("MEM", f"stage {s} peer {ctx.gpus[i].name}", stage=s))
    return out


def _choose(
    cfg,
    fleet: Fleet,
    base: dict,
    base_idx: list,
    device: str,
    allow_pieces: bool,
    shard,
    pinned_partition: bool = False,
):
    ctx = _Ctx(cfg, fleet, base, device)
    cands = _candidates(ctx, base_idx)
    recipe = base["spec"] if base.get("spec") is not None else base["draft"]
    one_cut = [[0, cfg.n_layer + 2]]
    for c in cands:
        c["reasons"] = _refusals_before_pricing(ctx, c, allow_pieces, shard)
        # the job's partition override is a width-1 cut; overrides win (never
        # dropped): a wide candidate, or the one-stage candidate (BA-467) whose
        # cut is not the pinned one, is refused
        moves_cut = c["w"] > 1 or (
            c["S"] == 1 and [list(x) for x in recipe.partition or []] != one_cut
        )
        if pinned_partition and moves_cut and not c["reasons"]:
            c["reasons"] = [admission.Reason("override", "partition pinned by the job")]
        if c["reasons"]:
            continue
        if c["w"] > 1:
            stages_in = [
                ctx.stage(s, c["S"], (0, 1), row, c["shard"], c["wire"])
                for s, row in enumerate(c["rows"])
            ]
            c["reasons"] = admission.missing_inputs(
                stages_in, ctx.hosts, ctx.k_source, c["shard"] == "pieces"
            )
            if c["reasons"]:
                continue
    wide_left = [c for c in cands if c["w"] > 1 and not c["reasons"]]
    for c in cands:
        if c["reasons"] or (c["w"] == 1 and not wide_left and not (ctx.hf and ctx.tied)):
            continue
        step, reasons, detail = _price(ctx, c, check=True)
        c["step_s"], c["detail"] = step, detail
        c["reasons"] = reasons
    for c in cands:
        c["admitted"] = not c["reasons"]
        c["reasons"] = [r.as_dict() for r in c["reasons"]]
    admitted = [c for c in cands if c["admitted"]]
    base["candidates"] = [_public(c) for c in cands]
    base["rationale"]["candidates"] = [
        {k: c[k] for k in ("name", "admitted", "step_s")} for c in cands
    ]
    base["rationale"]["refused"] = {
        c["name"]: [r["text"] for r in c["reasons"]] for c in cands if not c["admitted"]
    }
    if autoconfig.moe_fields(cfg) is not None:
        # SCHEDULER-I21: ρ_s = P_s / P^F_s of every stage pieces are refused for
        base["rationale"]["moe_rho"] = {
            c["name"]: _moe_rho(ctx, c)
            for c in cands
            if c["shard"] == "pieces" and c["w"] > 1 and not c["admitted"]
        }
    if not admitted:
        base["spec"] = None
        base["rationale"]["decision"] = "no candidate is admitted: " + "; ".join(
            f"{name}: {', '.join(t)}" for name, t in base["rationale"]["refused"].items()
        )
        return base
    best = min(admitted, key=lambda c: c["step_s"] if c["step_s"] is not None else 0.0)
    base["rationale"]["chosen"] = best["name"]
    if best["S"] == 1:
        return _one_stage_spec(ctx, base, best)
    if best["w"] == 1:
        return base
    return _wide_spec(ctx, base, best)


def _moe_rho(ctx: _Ctx, c: dict) -> list:
    """ρ_s = P_s / P^F_s per stage of a candidate's cut: all parameters over
    the parameters a token uses (MoE blocks at k of E experts)."""
    cfg = ctx.cfg
    speeds = [sum(ctx.gpus[i].tflops for i in row) for row in c["rows"]]
    moe = autoconfig.moe_fields(cfg) or {}
    m = autoconfig.moe_layer_costs(moe, cfg.d_model, cfg.seq_len, 1)
    out = []
    for a, b in proportional_partition(cfg, speeds):
        total = stage_param_split(cfg, a, b)[0]
        n_moe = sum(
            autoconfig.is_moe_block(cfg, j - 1) for j in range(max(a, 1), min(b, cfg.n_layer + 1))
        )
        active = total - n_moe * (m["params_total"] - m["params_active"])
        out.append(round(total / active, 4) if active else None)
    return out


def _one_stage_spec(ctx: _Ctx, base: dict, c: dict) -> dict:
    """The one-stage candidate as a spec: the width-1 recipe (the draft's mb,
    M, flags) with every entry on the candidate's node (BA-467's target)."""
    spec = copy.deepcopy(base["spec"] if base.get("spec") is not None else base["draft"])
    i = c["rows"][0][0]
    node = ctx.fleet.nodes[i]
    spec.stages = 1
    spec.partition = [[0, ctx.cfg.n_layer + 2]]
    spec.stage_widths = [1]
    spec.peer_weights = None
    spec.mb_owner = None
    spec.schedule_order = None
    spec.links = []
    spec.emission_order = None
    spec.endpoints = (
        [[[node.host, node.dial_port, node.listen_port or node.dial_port]]]
        if node.dial_port
        else None
    )
    spec.devices = [[node.device or ctx.device]] if node.device else None
    spec.finalize()
    base.update(
        spec=spec,
        stage_widths=[1],
        placement=[[_peer_record(node, i, ctx.device)]],
        shard="zero1",
        wire="fp32",
        emission_order=None,
        piece_bounds=None,
        stream_grads=False,
    )
    base["rationale"]["partition"] = spec.partition
    base["rationale"]["decision"] = (
        f"{c['name']}: every split of the tied table is refused; the whole model on one "
        f"stage ({c['step_s']:.4g} s); the width-1 cut was: {base['rationale'].get('decision')}"
    )
    return base


def _public(c: dict) -> dict:
    return {k: v for k, v in c.items() if k != "rows"} | {"nodes": c["rows"]}


def _wide_spec(ctx: _Ctx, base: dict, c: dict) -> dict:
    """The chosen wide candidate as a spec: today's recipe fields, the
    candidate's cut, widths, peers and M2b fields (by attribute, I-7)."""
    spec = copy.deepcopy(base["spec"])
    rows = c["rows"]
    nodes = ctx.fleet.nodes
    spec.stages = c["S"]
    spec.stage_widths = [c["w"]] * c["S"]
    spec.partition = proportional_partition(
        ctx.cfg, [sum(ctx.gpus[i].tflops for i in row) for row in rows]
    )
    spec.peer_weights = [[float(ctx.gpus[i].tflops) for i in row] for row in rows]
    spec.mb_owner = None
    spec.schedule_order = None
    spec.links = []
    spec.slot_peers = None
    spec.endpoints = (
        [
            [
                [nodes[i].host, nodes[i].dial_port, nodes[i].listen_port or nodes[i].dial_port]
                for i in row
            ]
            for row in rows
        ]
        if any(nodes[i].dial_port for row in rows for i in row)
        else None
    )
    devs = [[nodes[i].device or ctx.device for i in row] for row in rows]
    spec.devices = devs if any(nodes[i].device for row in rows for i in row) else None
    spec.shard = c["shard"]
    spec.wire = c["wire"]
    spec.emission_order = c["emission_order"]
    spec.piece_bounds = None
    spec.stream_grads = True
    spec.finalize()
    base.update(
        spec=spec,
        stage_widths=list(spec.stage_widths),
        placement=[[_peer_record(nodes[i], i, ctx.device) for i in row] for row in rows],
        shard=c["shard"],
        wire=c["wire"],
        emission_order=c["emission_order"],
        piece_bounds=None,
        stream_grads=True,
    )
    base["rationale"]["partition"] = spec.partition
    # The width-1 decision text described a plan that is not the one returned.
    base["rationale"]["decision"] = (
        f"{c['name']}: the smallest simulated step of the admitted candidates "
        f"({c['step_s']:.4g} s); the width-1 cut was: {base['rationale'].get('decision')}"
    )
    return base
