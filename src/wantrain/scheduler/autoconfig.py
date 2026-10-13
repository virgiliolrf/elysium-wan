"""Automatic run planning: a measured fleet becomes a JobSpec, or a refusal.

Boundary precision is a decision variable: the planner walks the ladder from
the widest rung down and picks the first whose per-microbatch boundary
transfer fits inside one compute window on the measured link, with margin.
Every choice is overridable and every rejected rung is reported with the
number that rejected it. policy= "lossless" admits only rungs lossless vs the
declared compute dtype, "balanced" prefers them and says so when it falls
back, "max_throughput" takes the narrowest admissible rung. The floor uses
k=1: transfers serialise on bandwidth, so k hides latency, not throughput.
"""

from wantrain.model import precision

# widest (most faithful) first
LADDER = ["fp32", "bf16", "fp16", "fp8_e5m2", "fp8_e4m3"]

BYTES_PER_PARAM_FP32_ADAM = 16  # master weights 4 + grad 4 + Adam m,v 8


def activation_graph_bytes(mb_size: int, seq_len: int, d_model: int, n_layers: int = 1) -> int:
    """Live activation graph of a built-in block in fp32: about 8 tensors of
    (mb, seq, d_model) per layer."""
    return n_layers * 8 * mb_size * seq_len * d_model * 4


# A frozen parameter carries its weight and nothing else: no gradient, no
# master copy, no Adam moments. The 4-vs-16 ratio is why LoRA changes which
# models the planner can admit on small cards.
BYTES_PER_PARAM_FROZEN = 4


def boundary_bytes_per_token(d_model: int, boundary: str) -> float:
    """Wire bytes per token per direction, after lossless entropy coding."""
    s = precision.spec(boundary)
    return s["bytes"] * d_model / s["ratio"]


def tokens_per_s_ceiling(d_model: int, boundary: str, link_mbps: float) -> float:
    """Hard throughput ceiling set by the wire, per direction.

    Compute optimisations cannot exceed this. Only a narrower boundary, a better
    lossless coder, or more bandwidth moves it.
    """
    return (link_mbps * 1e6 / 8) / boundary_bytes_per_token(d_model, boundary)


def floor_mbps(d_model: int, seq_len: int, mb_size: int, boundary: str, window_s: float) -> float:
    """Bandwidth needed for one boundary transfer to fit in one compute window."""
    per_mb = boundary_bytes_per_token(d_model, boundary) * mb_size * seq_len
    return per_mb * 8 / window_s / 1e6


def stage_window_s_scaled(n_layers_stage: int, ref_layers: int, ref_window_s: float) -> float:
    """Scale a measured per-microbatch window by layer count (compute ~ layers)."""
    return ref_window_s * n_layers_stage / ref_layers


def stage_memory_bytes(
    params_stage: int,
    d_model: int,
    seq_len: int,
    mb_size: int,
    k_inflight: int,
    boundary: str,
    recompute: bool,
    n_layers_stage: int,
    n_head: int = 16,
    flash_attention: bool = False,
    head_vocab: int = 0,
    frozen_params_stage: int = 0,
    attn_scores_bytes: float | None = None,
    extra_bytes: float = 0.0,
    state_bytes_per_param: int = BYTES_PER_PARAM_FP32_ADAM,
    grad_transient_bytes: float | None = None,
) -> dict:
    """Rough memory model for one stage. Deliberately conservative.

    The attention score matrix decides whether long-sequence runs are reachable:
    materialised attention holds (mb, heads, S, S) per layer, quadratic in S,
    and dwarfs everything else past a few thousand tokens; a fused/flash kernel
    never materialises it. Long context is the cheapest way to widen the compute
    window and lower the bandwidth floor, so this term gates the lossless
    regime on a slow link.

    `attn_scores_bytes` replaces the per-layer score term where the model
    knows it (an arch "hf" layer's eager attention, `eager_attention_bytes`);
    `extra_bytes` adds what the stage holds beyond these terms (the split tied
    table's charges, `tied_memory`).
    """
    s = precision.spec(boundary)
    # params_stage counts what is trained; frozen_params_stage counts what is
    # merely resident. Charging the frozen base full optimizer price would keep
    # the planner refusing exactly the fleets LoRA exists to make possible.
    # state_bytes_per_param: 8 under optimizer "adamw-host" (RUNTIME-I20, runtime/optim.py).
    state = params_stage * state_bytes_per_param + frozen_params_stage * BYTES_PER_PARAM_FROZEN
    bnd = k_inflight * mb_size * seq_len * d_model * s["bytes"]
    per_layer = activation_graph_bytes(mb_size, seq_len, d_model)
    if attn_scores_bytes is not None:
        scores = attn_scores_bytes
    else:
        scores = 0 if flash_attention else mb_size * n_head * seq_len * seq_len * 4
    # One microbatch's backward needs the full stage graph live; recompute only
    # changes when it is built, and what it buys is not holding a graph for
    # every in-flight microbatch at once. Zero-bubble holds at least one more:
    # each deferred dL/dW closure pins its microbatch's rebuilt graph.
    graph = (per_layer + scores) * n_layers_stage
    live = graph * 2 if recompute else graph * max(2, k_inflight)
    # torch.autograd.grad materialises the entire stage's weight gradients as
    # one transient set before the ordered accumulator folds and frees them; it
    # coexists with the accumulator and the optimizer state, so it is peak.
    # RUNTIME-I21: a width-1 stage folds each gradient as it is produced, so its transient is one tensor
    # (grad_transient_bytes, the planner's bound) rather than the set; None keeps the whole-set charge.
    grad_transient = params_stage * 4 if grad_transient_bytes is None else grad_transient_bytes
    # The stage holding the Head also holds the logits: (mb, S, vocab) in fp32
    # for the forward plus the same again for their gradient. At large vocab
    # this term alone can exceed the card.
    head = 2 * mb_size * seq_len * head_vocab * 4
    return dict(
        optimizer_state=state,
        boundary_buffers=bnd,
        activations=live,
        attention_scores=scores,
        head_logits=head,
        grad_transient=grad_transient,
        extra=extra_bytes,
        total=state + bnd + live + head + grad_transient + extra_bytes,
    )


def evaluate(
    model_cfg,
    params_stage: int,
    n_layers_stage: int,
    mb_size: int,
    k_inflight: int,
    link_mbps: float,
    window_s: float,
    mem_budget_bytes: float,
    margin: float = 2.0,
    compute_dtype: str = "fp32",
    allow_compute_change: bool = True,
    flash_attention: bool = False,
    compute_cap=None,
    head_vocab: int = 0,
    frozen_params_stage: int = 0,
    attn_scores_bytes: float | None = None,
    extra_bytes: float = 0.0,
) -> list:
    """Score every priced ladder rung for this fleet (`priced_rungs`: the
    whole LADDER, or for arch "hf" its one residual dtype). Returns rows
    widest-first.

    A rung is counted lossless if the boundary is native to the compute dtype --
    either because it already matches, or because the engine can legitimately
    declare compute at that rung (standard mixed precision). Declaring a bf16
    reference recipe and shipping bf16 bytes is lossless; taking fp32 math and
    narrowing it at the socket is not, even though the wire looks identical.
    """
    rows = []
    for boundary in priced_rungs(model_cfg):
        fl = floor_mbps(model_cfg.d_model, model_cfg.seq_len, mb_size, boundary, window_s)
        mem = stage_memory_bytes(
            params_stage,
            model_cfg.d_model,
            model_cfg.seq_len,
            mb_size,
            k_inflight,
            boundary,
            True,
            n_layers_stage,
            model_cfg.n_head,
            flash_attention,
            head_vocab=head_vocab,
            frozen_params_stage=frozen_params_stage,
            attn_scores_bytes=attn_scores_bytes,
            extra_bytes=extra_bytes,
        )
        native_now = precision.classify(compute_dtype, boundary)["lossless_vs_reference"]
        # A rung is only lossless if compute can be declared at it, and that
        # needs the weakest node in the fleet to run that math; otherwise the
        # planner picks bf16 for a pre-Ampere fleet.
        runnable = precision.compute_runnable(boundary, compute_cap)
        can_declare = allow_compute_change and boundary in precision.CAN_BE_COMPUTE and runnable
        rows.append(
            dict(
                boundary=boundary,
                lossless=native_now or can_declare,
                compute_runnable=runnable,
                requires_compute=None if native_now else (boundary if can_declare else None),
                bytes_per_token=round(boundary_bytes_per_token(model_cfg.d_model, boundary), 1),
                floor_mbps=round(fl, 1),
                needed_with_margin=round(fl * margin, 1),
                link_mbps=link_mbps,
                admissible=link_mbps >= fl,
                meets_margin=link_mbps >= fl * margin,
                tokens_per_s_ceiling=round(
                    tokens_per_s_ceiling(model_cfg.d_model, boundary, link_mbps)
                ),
                stage_mem_gb=round(mem["total"] / 2**30, 2),
                attn_score_gb=round(mem["attention_scores"] / 2**30, 2),
                fits_memory=mem["total"] <= mem_budget_bytes,
                needs_scale=precision.spec(boundary)["needs_scale"],
            )
        )
    return rows


def _margin_note(r, margin):
    return (
        f"admissible but WITHOUT the {margin:g}x margin: needs "
        f"{r['needed_with_margin']:g} Mbps, link is {r['link_mbps']:g} Mbps. "
        f"Expect residual stall"
    )


def choose(rows: list, policy: str = "balanced", require_margin: bool = True, margin: float = 2.0):
    """Pick a rung under the stated objective. Returns (row, reason)."""
    fits = [r for r in rows if r["fits_memory"]]
    strong = [r for r in fits if r["meets_margin"]]
    weak = [r for r in fits if r["admissible"] and not r["meets_margin"]]
    pool = strong if require_margin else strong + weak

    def declare(r):
        if r["requires_compute"]:
            return (
                f" (requires declaring compute_dtype="
                f"'{r['requires_compute']}' so the boundary is native)"
            )
        return ""

    def pick(cands, why):
        if not cands:
            return None, None
        r = cands[0]
        return r, why + declare(r)

    if policy == "max_throughput":
        # cheapest wire, not narrowest format -- e5m2 codes better than e4m3 at
        # equal width, so ladder position is the wrong ordering here
        if strong:
            r = max(strong, key=lambda x: x["tokens_per_s_ceiling"])
            return r, (
                "highest-throughput rung clearing the "
                f"{margin:g}x margin, explicitly accepting boundary "
                "approximation" + declare(r)
            )
        if weak:
            r = max(weak, key=lambda x: x["tokens_per_s_ceiling"])
            return r, (
                "highest-throughput admissible rung" + declare(r) + " -- " + _margin_note(r, margin)
            )
        return None, "nothing is admissible on this link, at any precision"

    ll_strong = [r for r in strong if r["lossless"]]
    ll_weak = [r for r in weak if r["lossless"]]

    if ll_strong:
        return pick(ll_strong, f"widest numerically-lossless rung clearing the {margin:g}x margin")
    if policy == "lossless":
        if ll_weak and not require_margin:
            r, w = pick(ll_weak, "widest numerically-lossless rung")
            return r, w + " -- " + _margin_note(r, margin)
        hint = (
            f"the widest lossless rung needs "
            f"{[r['needed_with_margin'] for r in rows if r['lossless']][0]:g}"
            f" Mbps for the margin"
        )
        return None, (
            f"no numerically-lossless boundary clears this link ({hint}). "
            "Options: more bandwidth; a longer compute window (bigger "
            "model, longer sequence, more layers per stage); "
            "require_margin=False to run at the floor; or "
            "policy='balanced' to accept a declared approximation"
        )
    # balanced
    if ll_weak:
        r, w = pick(ll_weak, "widest numerically-lossless rung")
        return r, w + " -- " + _margin_note(r, margin)
    if pool:
        r, w = pick(
            pool,
            "NO LOSSLESS RUNG FITS. Widest admissible approximate "
            "boundary -- this run is NOT numerically lossless "
            "against the compute dtype",
        )
        return r, w
    if weak:
        r, w = pick(
            weak,
            "NO LOSSLESS RUNG FITS, and nothing clears the margin. Widest approximate boundary",
        )
        return r, w + " -- " + _margin_note(r, margin)
    return None, "nothing is admissible on this link, at any precision"


def report(rows: list, chosen, reason: str) -> str:
    """Human-readable rationale. Printed before a run starts."""
    w = (
        f"{'boundary':10s} {'lossless':>8s} {'B/token':>8s} {'floor':>7s} "
        f"{'need 2x':>8s} {'admit':>6s} {'margin':>7s} {'tok/s':>7s} "
        f"{'mem GB':>7s} {'fits':>5s}"
    )
    out = [w, "-" * len(w)]
    for r in rows:
        mark = " <-- chosen" if chosen is not None and r is chosen else ""
        out.append(
            f"{r['boundary']:10s} {str(r['lossless']):>8s} "
            f"{r['bytes_per_token']:8.0f} {r['floor_mbps']:7.1f} "
            f"{r['needed_with_margin']:8.1f} {str(r['admissible']):>6s} "
            f"{str(r['meets_margin']):>7s} {r['tokens_per_s_ceiling']:7.0f} "
            f"{r['stage_mem_gb']:7.2f} {str(r['fits_memory']):>5s}{mark}"
        )
    out.append("")
    out.append(f"decision: {reason}")
    return "\n".join(out)


def plan(
    model_cfg,
    params_stage: int,
    n_layers_stage: int,
    link_mbps: float,
    window_s: float,
    mem_budget_bytes: float,
    mb_size: int = 4,
    k_inflight: int = 4,
    margin: float = 2.0,
    policy: str = "balanced",
    compute_dtype: str = "fp32",
    require_margin: bool = True,
    boundary_override: str = None,
    flash_attention: bool = False,
    compute_cap=None,
    head_vocab: int = 0,
    frozen_params_stage: int = 0,
    attn_scores_bytes: float | None = None,
    extra_bytes: float = 0.0,
) -> dict:
    """Full plan. boundary_override skips the solver entirely: the engineer
    keeps the wheel and gets told what the override costs. For arch "hf" the
    boundary is not chosen (SPEC-scheduler §2.3): one rung is priced, and an
    override other than it is refused, naming it.

    compute_cap is the weakest node's (major, minor) CUDA compute capability,
    or None if unmeasured. It removes rungs the fleet cannot execute rather
    than admitting them and discovering it on the box.
    """
    if not precision.compute_runnable(compute_dtype, compute_cap):
        need = precision.MIN_COMPUTE_CAP[compute_dtype]
        raise ValueError(
            f"compute_dtype={compute_dtype!r} needs compute capability "
            f"{need[0]}.{need[1]}+, but the weakest node in this fleet is "
            f"sm_{compute_cap[0]}{compute_cap[1]}. Either pick a fleet of newer "
            f"GPUs or declare a compute dtype the hardware runs natively "
            f"(fp32 always works)."
        )
    if boundary_override:
        hf_boundary_check(model_cfg, boundary_override)
    rows = evaluate(
        model_cfg,
        params_stage,
        n_layers_stage,
        mb_size,
        k_inflight,
        link_mbps,
        window_s,
        mem_budget_bytes,
        margin,
        compute_dtype,
        flash_attention=flash_attention,
        head_vocab=head_vocab,
        frozen_params_stage=frozen_params_stage,
        compute_cap=compute_cap,
        attn_scores_bytes=attn_scores_bytes,
        extra_bytes=extra_bytes,
    )
    if boundary_override:
        chosen = next(r for r in rows if r["boundary"] == boundary_override)
        reason = f"boundary_dtype pinned to {boundary_override} by the operator"
        if not chosen["meets_margin"]:
            reason += " -- WARNING: " + _margin_note(chosen, margin)
        if not chosen["fits_memory"]:
            reason += (
                f" -- WARNING: estimated {chosen['stage_mem_gb']} GB per "
                f"stage exceeds the {mem_budget_bytes / 2**30:.1f} GB budget"
            )
    else:
        chosen, reason = choose(rows, policy, require_margin, margin)
    return dict(rows=rows, chosen=chosen, reason=reason, report=report(rows, chosen, reason))


# ============================================================ M4: HF and MoE costs
# SCHEDULER-I21 (ADR-0008) and SCHEDULER-I22 (ADR-0009) as pure functions over
# explicit inputs (HANDOUT-M4 I-13). The planner's layer_costs, analytic_costs
# and stage_memory_model read them through the config: an arch "hf" model from
# the builder's census (`hf_census`, I-10; never transformers), an MoE model
# from its moe_* fields. A config without either is priced as before.

# HF's residual dtype under today's recipes: parameters are fp32 masters, with
# or without autocast, so the residual stream (the cut) is fp32 (SPEC-scheduler
# §2.3; ADR-0009 decision 3). A bf16-parameter recipe is the precision ADR's.
HF_BOUNDARY = "fp32"
# AdamW of the tied table on the Head: 16 B read + 12 B written per element
# (docs/research/adr-0009/scripts/tied_terms.py).
TIED_ADAMW_BYTES = 28
COUNT_LANE_BYTES = 8  # one int64 count per expert (RUNTIME-I18)


def is_hf(cfg) -> bool:
    return getattr(cfg, "arch", "builtin") == "hf"


def hf_boundary_dtype(cfg) -> str:
    """The cut's dtype of an arch "hf" model: HF's residual dtype under the
    recipe, fp32 under fp32 parameters with or without autocast."""
    return HF_BOUNDARY


def hf_boundary_check(cfg, declared: str | None) -> None:
    """Refuse a declared boundary dtype other than HF's residual dtype for an
    arch "hf" model, naming it (SCHEDULER-I22). Nothing for other archs."""
    if not is_hf(cfg) or declared is None:
        return
    want = hf_boundary_dtype(cfg)
    if declared != want:
        raise ValueError(
            f"boundary_dtype={declared!r} declared for arch 'hf': the cut is HF's residual "
            f"dtype {want!r} ({precision.spec(want)['bytes']} bytes per element); a "
            f"narrower boundary is not HF's function (SCHEDULER-I22)"
        )


def priced_rungs(cfg) -> list:
    """The LADDER rungs the precision admission prices: all five, or for
    arch "hf" exactly HF's residual dtype (SPEC-scheduler §2.3)."""
    return [hf_boundary_dtype(cfg)] if is_hf(cfg) else list(LADDER)


def hf_census(cfg) -> list:
    """The builder's census of an arch "hf" config, checked for the fields the
    planner reads; refused naming `hf_census` when it is missing (the planner
    never imports transformers, I-10)."""
    census = getattr(cfg, "hf_census", None)
    if not census:
        raise ValueError(
            "arch 'hf' config without hf_census: the planner prices HF units only from "
            "the builder's census (HANDOUT-M4 I-10), and none was given"
        )
    n_layer = getattr(cfg, "n_layer", None)
    if n_layer is not None and len(census) != n_layer + 2:
        raise ValueError(
            f"hf_census has {len(census)} units, but n_layer + 2 = {n_layer + 2} "
            f"(prologue, one unit per layer, epilogue)"
        )
    for i, u in enumerate(census):
        for key in ("params", "linear_macs_per_token", "param_shapes"):
            if key not in u:
                raise ValueError(f"hf_census unit {i} has no {key!r}")
    return list(census)


def eager_bytes_per_elem(compute_dtype: str, softcap) -> int:
    """`s` of eager attention's `b·H·S²·s` bytes held for backward per layer:
    4 in fp32, 6 under autocast (fp32 parameters, narrower compute), plus 4
    (fp32) or 2 (autocast) with an attention softcap (SPEC-scheduler §2.3)."""
    if compute_dtype == "fp32":
        return 4 + (4 if softcap else 0)
    return 6 + (2 if softcap else 0)


def eager_attention_bytes(mb_size: int, heads: int, seq_len: int, s: int) -> int:
    """b·H·S²·s: one eager-attention layer's score bytes held for backward."""
    return mb_size * heads * seq_len * seq_len * s


def _numel(dims) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


def hf_hidden(census) -> int:
    """h: the trailing dim of the prologue's embedding table (its first 2-D
    parameter)."""
    for _name, dims in census[0]["param_shapes"]:
        if len(dims) == 2:
            return int(dims[1])
    raise ValueError("hf_census unit 0 (the prologue) has no 2-D embedding table")


def hf_tied(census) -> dict | None:
    """The tied table of an arch "hf" census, or None: {"name", "elems",
    "users": (0, i)}, i the unit that reuses the prologue's table. A unit
    whose `tied` is set uses a tensor listed once, at its first unit (I-10);
    `tied` names that tensor, True means the prologue's embedding table."""
    census = list(census)
    users = [i for i, u in enumerate(census) if u.get("tied") and i > 0]
    if not users:
        return None
    tied = census[users[-1]]["tied"]
    for v in census:
        for name, dims in v["param_shapes"]:
            if (name == tied) if isinstance(tied, str) else (v is census[0] and len(dims) == 2):
                return {"name": name, "elems": _numel(dims), "users": (0, users[-1])}
    raise ValueError(f"hf_census unit {users[-1]}: tied tensor {tied!r} is listed nowhere")


def hf_top_k(cfg) -> int | None:
    """k, the experts a token is routed to, of an arch "hf" config's
    hf_config (`num_experts_per_tok`), or None when it names none."""
    k = (getattr(cfg, "hf_config", None) or {}).get("num_experts_per_tok")
    return int(k) if k else None


def hf_unit_costs(
    census, seq_len: int, mb_size: int, compute_dtype: str, *, top_k: int | None = None
) -> list[dict]:
    """Per census unit (prologue, layers, epilogue): FLOPs per token of F, B
    and W, `cost` (F+B+W per token, layer_costs' unit), and per microbatch the
    eager-attention bytes, the pending-W bytes and the 3-D bank elements.

    F = 2·macs + 4·H·d·S', B = 2·macs + 8·H·d·S', W = 2·macs per token, with
    macs the unit's linear MACs (a tied head's GEMM included) and S' the
    window or the sequence: the paper's table that analytic_costs uses for the
    builtin block (24h² = 2·12h², 4hs). The prologue is a lookup (F = B = h).
    Pending W holds X and dY of every 2-D weight of a layer unit (in + out
    values per token, 4 bytes each), as the builtin block's 16h. Bank
    elements are the unit's 3-D parameters (an HF MoE layer's expert banks,
    [E, out, in]): no nn.Linear, so outside `macs`; a token runs `top_k` of
    the E experts, k·numel/E MACs, with dW in B (BA-459). A unit with banks
    and no `top_k` is refused naming num_experts_per_tok (SCHEDULER-I21)."""
    census = list(census)
    h = hf_hidden(census)
    last = len(census) - 1
    out = []
    for i, u in enumerate(census):
        macs = float(u.get("linear_macs_per_token") or 0)
        heads = int(u.get("attn_heads") or 0)
        hd = int(u.get("head_dim") or 0)
        window = u.get("window")
        span = min(seq_len, int(window)) if window else seq_len
        att = float(heads * hd * span)
        f, b, w = 2 * macs + 4 * att, 2 * macs + 8 * att, 2 * macs
        if i == 0:
            f, b = f + h, b + h
        shapes = u["param_shapes"]
        banks = [d for _n, d in shapes if len(d) == 3]
        bank_macs = 0.0
        if banks:
            if not top_k:
                raise ValueError(
                    f"hf_census unit {i} has expert banks, but hf_config names no "
                    f"num_experts_per_tok: the active compute is unknown (SCHEDULER-I21)"
                )
            bank_macs = float(top_k * sum(_numel(d) // int(d[0]) for d in banks))
            f, b = f + 2 * bank_macs, b + 4 * bank_macs
        layer = 0 < i < last
        pend = sum(int(d[0]) + int(d[1]) for _n, d in shapes if len(d) == 2) if layer else 0
        eager = 0
        if heads:
            s = eager_bytes_per_elem(compute_dtype, u.get("softcap"))
            eager = eager_attention_bytes(mb_size, heads, seq_len, s)
        out.append(
            dict(
                unit=i,
                params=int(u["params"]),
                f=f,
                b=b,
                w=w,
                cost=6 * (macs + bank_macs) + 12 * att + (3.0 * h if i == 0 else 0.0),
                eager_bytes=float(eager),
                pending_w_bytes=float(pend * mb_size * seq_len * 4),
                bank_elems=sum(_numel(d) for d in banks),
            )
        )
    return out


def moe_fields(cfg) -> dict | None:
    """The moe_* fields of an MoE config as moe_layer_costs' dict, or None
    for a dense one (moe_experts 0 or absent)."""
    if int(getattr(cfg, "moe_experts", 0) or 0) <= 0:
        return None
    from wantrain.model.arch import default_d_ff

    out = {
        k: getattr(cfg, k)
        for k in (
            "moe_experts",
            "moe_top_k",
            "moe_d_ff",
            "moe_shared_d_ff",
            "moe_shared_gate",
            "moe_router_bias",
            "moe_expert",
            "moe_layers",
        )
        if hasattr(cfg, k)
    }
    # the experts' width as MoEMLP builds it: moe_d_ff, else d_ff, else the
    # SwiGLU default (arch.MoEMLP; moe_d_ff None is GPTConfig's default)
    out["moe_d_ff"] = (
        out.get("moe_d_ff")
        or getattr(cfg, "d_ff", None)
        or default_d_ff("swiglu", int(cfg.d_model))
    )
    return out


def is_moe_block(cfg, j: int) -> bool:
    """Block j (0-based among the blocks) is an MoE block."""
    if moe_fields(cfg) is None:
        return False
    layers = getattr(cfg, "moe_layers", None)
    return layers is None or j in layers


def moe_layer_costs(moe: dict, d_model: int, seq_len: int, mb_size: int) -> dict:
    """One MoE block's MLP (router, E routed experts, shared expert and its
    gate; the attention is the caller's), SCHEDULER-I21: compute at the
    active parameters, bytes and memory at the total.

    `moe` holds the moe_* fields (E, k, f, f_sh, shared gate, router bias,
    expert kind) and optionally `hf_banks` (HF's 3-D expert banks, dW in B)
    and `master_bytes` (b_m, 4 under today's recipe). Per token: F = 2·P^F,
    B = 2·P^F plus the dW of what B computes (the router, never converted;
    under HF banks the k routed experts too), W the rest. Per microbatch:
    the pending W (k·(2h+3f) + 2h+3f_sh values per token, SPEC-model §2; the
    routed part 0 under HF banks) at 4 bytes, the count lane E·8 bytes per
    peer, and under HF banks the bank gradient b_m·E·3hf held from B's end to
    W's start (times m_BW, `admission.device_terms`)."""
    h = int(d_model)
    E = int(moe["moe_experts"])
    k = int(moe.get("moe_top_k", 1))
    f = int(moe["moe_d_ff"])
    f_sh = int(moe.get("moe_shared_d_ff", 0) or 0)
    if not (E > 0 and 1 <= k <= E and f > 0):
        raise ValueError(
            f"moe_layer_costs: needs E > 0, 1 <= k <= E, d_ff > 0 (E {E}, k {k}, f {f})"
        )
    banks = bool(moe.get("hf_banks", False))
    b_m = int(moe.get("master_bytes", 4))
    router = h * E + (E if moe.get("moe_router_bias") else 0)
    expert = 3 * h * f + ((2 * f + h) if moe.get("moe_expert") == "swiglu_clamped" else 0)
    shared = 3 * h * f_sh
    gate = h if f_sh and moe.get("moe_shared_gate", "sigmoid") == "sigmoid" else 0
    total = router + E * expert + shared + gate
    active = router + k * expert + shared + gate
    in_b = router + (k * expert if banks else 0)
    tokens = mb_size * seq_len
    pend = (0 if banks else k * (2 * h + 3 * f)) + ((2 * h + 3 * f_sh) if f_sh else 0)
    return dict(
        params_total=total,
        params_active=active,
        rho=total / active,
        router_params=router,
        expert_params=expert,
        shared_params=shared,
        gate_params=gate,
        f=2.0 * active,
        b=2.0 * active + 2.0 * in_b,
        w=2.0 * (active - in_b),
        pending_w_per_token=pend,
        pending_w_bytes=float(pend * tokens * 4),
        count_lane_bytes=E * COUNT_LANE_BYTES,
        bank_bytes=float(b_m * E * 3 * h * f) if banks else 0.0,
    )


def builtin_attn_params(cfg) -> int:
    """The builtin block's attention parameters (q, kv, out; no norms or
    biases, as params_per_layer)."""
    d = cfg.d_model
    nkv = cfg.n_kv_head or cfg.n_head
    hd = d // cfg.n_head
    return d * d + 2 * (nkv * hd) * d + d * d


def builtin_attn_pending(cfg) -> int:
    """Pending-W values per token of the builtin block's attention: X and dY
    of qkv and of proj (6h for full multi-head)."""
    d = cfg.d_model
    nkv = cfg.n_kv_head or cfg.n_head
    return d + (d + 2 * nkv * (d // cfg.n_head)) + 2 * d


def delta_t_tied(
    t_s: float,
    t_F: float,
    t_B: float,
    rows_T,
    rows_R,
    hops: int,
    latency_s: float,
    bandwidth_Bps: float,
    *,
    d_model: int,
    vocab_size: int,
    tokens_mb: int,
    mem_bw_Bps: float,
    adamw_bytes_per_elem: int = TIED_ADAMW_BYTES,
) -> dict:
    """ΔT_tied, the formula of record (docs/research/adr-0009/round5.md §8),
    its five terms and their sum, in seconds. rows_T[i] = |T_i| and
    rows_R[k] = |R_k| for the step's M microbatches (from make_batch's ids);
    hops = S − 1 chain hops between stage 0 and the Head, each with one-way
    latency L; B = bandwidth_Bps (bytes/s); a = T_mb·h·4 the cut's bytes per
    microbatch; t_s, t_F, t_B stage 0's per-microbatch times (the head GEMM
    in the Head's cost). mem_bw_Bps is the Head device's memory bandwidth
    (math.inf when unmeasured: the scan and the AdamW terms are then 0, a
    lower bound).

    throughput  M·max(0, (a+ē)/B − max(t_s, a/B))
    warmup      Σ_{k=1}^{S−1} max(0, |R_k|·h·4/B − t_F)
    round_trip  2(S−1)L + (|T_{M−1}|+|T_0|)·h·4/B + t_AdamW(table)
    scan        M·V·h·4/BW_mem
    back_link   max(0, Σ_{k≥1}|R_k|·h·4/B − W_idle),
                W_idle = 2(S−1)L + S·t_F + (S−1)·a/B + t_B"""
    rows_T, rows_R = list(rows_T), list(rows_R)
    M = len(rows_T)
    if M < 1 or len(rows_R) != M:
        raise ValueError(f"rows_T and rows_R need one entry per microbatch ({M} vs {len(rows_R)})")
    if hops < 0 or bandwidth_Bps <= 0 or mem_bw_Bps <= 0:
        raise ValueError("hops >= 0, bandwidth_Bps > 0 and mem_bw_Bps > 0 are required")
    S = hops + 1
    row = d_model * 4 / bandwidth_Bps
    a = tokens_mb * d_model * 4 / bandwidth_Bps
    e_bar = sum(rows_T) / M * row
    table = vocab_size * d_model
    throughput = M * max(0.0, (a + e_bar) - max(t_s, a))
    warmup = sum(max(0.0, rows_R[k] * row - t_F) for k in range(1, min(S, M)))
    round_trip = (
        2 * hops * latency_s
        + (rows_T[M - 1] + rows_T[0]) * row
        + table * adamw_bytes_per_elem / mem_bw_Bps
    )
    scan = M * table * 4 / mem_bw_Bps
    w_idle = 2 * hops * latency_s + S * t_F + (S - 1) * a + t_B
    back_link = max(0.0, sum(rows_R[1:]) * row - w_idle)
    return dict(
        throughput=throughput,
        warmup=warmup,
        round_trip=round_trip,
        scan=scan,
        back_link=back_link,
        total=throughput + warmup + round_trip + scan + back_link,
        w_idle=w_idle,
    )


def tied_memory(vocab_size: int, d_model: int, tokens_mb: int, n_pend: int) -> dict:
    """What a tied table split across stages adds beyond its parameters
    (SPEC-scheduler §2.3): the Head's held rows n_pend·min(V, n_pend·T_mb)·h·4
    (its table, optimizer state, accumulator and transient dense gradient are
    the Head's trainable parameters' 16 + 4 B), and stage 0's working copy and
    one transient dense gradient, 2·V·h·4."""
    return dict(
        head=float(n_pend * min(vocab_size, n_pend * tokens_mb) * d_model * 4),
        stage0=float(2 * vocab_size * d_model * 4),
    )


def tied_hop_bytes(rows_fwd: int, rows_bwd: int, d_model: int) -> tuple[float, float]:
    """The tied bytes on every chain hop between stage 0 and the Head:
    |T_i|·h·4 per microbatch forward (TIED frames) and |U_{t+1}|·h·4 per step
    backward."""
    return float(rows_fwd * d_model * 4), float(rows_bwd * d_model * 4)
