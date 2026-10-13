"""Fine-tuning and model construction: pretrained weights land intact or are
refused, LoRA changes only the adapters, HF decoders run as transformers' own
modules windowed per unit behind an exact gate, and the MoE block is one exact
function. The failure guarded against is not a crash but a model that loads
95% of a checkpoint, leaves the rest at random init, trains, shows a falling
loss and is worthless; or a pipeline whose bits differ from the reference."""

# WOULD-FAIL-IF: a pretrained checkpoint loads partially or under other names instead of being refused, LoRA changes the base model's numerics or trains frozen weights, flush_iter (MODEL-I7) yields gradients whose widened bits differ from flush, an HF unit chain differs from the vanilla model, or the MoE block's selection, folds, unrouted gradients or bias update leave their written rule

import base64
import contextlib
import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from wantrain.model import arch, hf, lora, rollback, wgrad
from wantrain.model.gpt import GPTConfig, build_layers, call_layer, loss_fn

# ------------------------------------------------------------------------ LoRA


# MODEL-I10's fingerprint on a Linux CUDA wheel: TORCH_RECORDS has no Linux entry yet (hfm.py: "Linux entries wait
# for the pin-bump PR", D-9), and the native-library check flags nvidia/cu13/lib/libcupti.so.13 as a RECORD file's
# name loaded from another path. These arms run on macOS, where the entry is measured; on a Linux CUDA wheel they
# are expected to fail until the guard and the entry are fixed (see the "HF fingerprint on Linux CUDA wheels" issue).
_LINUX_CUDA_FINGERPRINT = pytest.mark.xfail(
    sys.platform.startswith("linux") and torch.version.cuda is not None,
    reason="MODEL-I10 fingerprint: no Linux TORCH_RECORDS entry, FP-NATIVE false positive on nvidia libraries",
    strict=True,
)


def test_lora_leaves_the_model_numerically_untouched_at_step_zero():
    """B is zero, so the adapted model is the base model before any update."""
    # covers: MODEL-I5
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, seq_len=8)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))

    base = build_layers(cfg, seed=99)
    y0 = x
    for l in base:
        y0 = l(y0)

    adapted = build_layers(cfg, seed=99)
    lora.apply(adapted, rank=4, alpha=8, targets="attn", seed=99)
    y1 = x
    for l in adapted:
        y1 = l(y1)

    assert torch.equal(y0, y1)


def test_lora_freezes_the_base_and_trains_only_adapters():
    # covers: MODEL-I5
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, seq_len=8)
    layers = build_layers(cfg, seed=5)
    rep = lora.apply(layers, rank=4, alpha=8, targets="attn", seed=5)

    train, froze = lora.split_params(layers)
    assert train and froze
    assert all(p.requires_grad for p in train)
    assert not any(p.requires_grad for p in froze)
    # The whole economic argument: optimizer state exists only for these.
    assert rep["trainable_pct"] < 10.0
    assert rep["trainable_params"] == sum(p.numel() for p in train)

    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    y = x
    for l in layers:
        y = l(y)
    nn.functional.cross_entropy(y.reshape(-1, y.size(-1)), tgt.reshape(-1)).backward()

    assert all(p.grad is not None for p in train), "an adapter got no gradient"
    assert all(p.grad is None for p in froze), "a frozen weight received a gradient"


def test_lora_init_does_not_depend_on_how_much_was_built_before_it():
    """Adapter init must not depend on how much of the model was built before it."""
    # covers: MODEL-I5
    cfg = GPTConfig(vocab_size=64, n_layer=3, n_head=4, d_model=32, seq_len=8)

    a = build_layers(cfg, seed=11)
    lora.apply(a, rank=4, alpha=8, targets="attn", seed=11)

    b = build_layers(cfg, seed=11)
    torch.randn(1000)  # disturb the global RNG in between
    lora.apply(b, rank=4, alpha=8, targets="attn", seed=11)

    for la, lb in zip(a, b):
        for (na, pa), (nb, pb) in zip(la.named_parameters(), lb.named_parameters()):
            assert na == nb
            assert torch.equal(pa, pb), f"{na} differs between builds"


def test_lora_rank_zero_is_a_no_op():
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, seq_len=8, lora_rank=0)
    layers = build_layers(cfg, seed=3)
    assert all(p.requires_grad for l in layers for p in l.parameters())
    assert not any(isinstance(m, lora.LoRALinear) for l in layers for m in l.modules())


def test_lora_reaches_the_model_through_build_layers():
    cfg = GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        d_model=32,
        seq_len=8,
        lora_rank=4,
        lora_alpha=8,
        lora_targets="attn",
    )
    layers = build_layers(cfg, seed=3)
    assert any(isinstance(m, lora.LoRALinear) for l in layers for m in l.modules())
    train, froze = lora.split_params(layers)
    assert train and froze


def test_unknown_target_set_is_refused_not_silently_empty():
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, seq_len=8)
    layers = build_layers(cfg, seed=3)
    with pytest.raises(ValueError, match="unknown"):
        lora.apply(layers, rank=4, alpha=8, targets="attention", seed=3)


# ------------------------------------------------- memory model / planner view


def test_frozen_base_is_not_charged_for_optimizer_state():
    """The reason LoRA changes which fleets are admissible at all."""
    # covers: MODEL-I5
    from wantrain.scheduler import autoconfig as ac

    n = 1_000_000_000
    full = ac.stage_memory_bytes(n, 4096, 512, 1, 2, "bf16", True, 4)
    lo = ac.stage_memory_bytes(n // 1000, 4096, 512, 1, 2, "bf16", True, 4, frozen_params_stage=n)
    assert lo["optimizer_state"] < full["optimizer_state"] / 3.5
    # Frozen weights are still resident -- LoRA is not free, it is 4x cheaper.
    assert lo["optimizer_state"] > n * 3


# ------------------------------------------------- streamed deferred dW (MODEL-I7)

_DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _toy_stage(compute):
    """Blocks plus Head of a toy GPT: a non-first stage, input in the compute dtype."""
    cfg = GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        d_model=32,
        seq_len=8,
        compute_dtype=compute,
        boundary_dtype=compute,
    )
    stage = nn.ModuleList(list(build_layers(cfg, 7))[1:])
    g = torch.Generator().manual_seed(3)
    h0 = torch.randn(2, 8, 32, generator=g).to(_DT[compute])
    tgt = torch.randint(0, 64, (2, 8), generator=g)
    return stage, h0, tgt


def _toy_store(stage, h0, tgt):
    h0 = h0.clone().requires_grad_(True)
    h = h0
    for layer in stage:
        h = layer(h)
    loss = nn.functional.cross_entropy(h.reshape(-1, h.size(-1)), tgt.reshape(-1))
    owned = wgrad.owned_params(stage)
    others = [p for p in stage.parameters() if id(p) not in owned]
    with wgrad.capture() as store:
        torch.autograd.grad(loss, others + [h0])
    return store


def _linears_in_forward_order(stage):
    out = []
    for layer in stage:
        names = ("qkv", "proj", "fc1", "fc2") if hasattr(layer, "qkv") else ("out",)
        out += [getattr(layer, n) for n in names]
    return out


@pytest.mark.parametrize("order", ["store", "forward"])
@pytest.mark.parametrize("compute", ["fp32", "bf16", "fp16"])
def test_flush_iter_equals_flush_bitwise_on_a_toy_gpt_stage(compute, order):
    # covers: MODEL-I7
    # WOULD-FAIL-IF: a yielded gradient widened to the master dtype differs from flush, a single-contribution
    # gradient is widened before it is yielded, or the modules run in another order than the one requested
    stage, h0, tgt = _toy_stage(compute)
    ref = wgrad.flush(_toy_store(stage, h0, tgt))
    store = _toy_store(stage, h0, tgt)
    got = list(wgrad.flush_iter(store, order))
    assert store == []
    lins = _linears_in_forward_order(stage)
    if order == "store":
        lins = lins[::-1]
    expect = [p for m in lins for p in (m.weight, m.bias) if p is not None]
    assert [id(p) for p, _ in got] == [id(p) for p in expect]
    assert set(ref) == {id(p) for p in expect} == wgrad.owned_params(stage)
    for p, g in got:
        assert g.dtype == _DT[compute]  # the GEMM's output dtype, not the master's
        assert p.dtype == torch.float32
        w = g.to(p.dtype)
        assert w.dtype == ref[id(p)].dtype and torch.equal(w, ref[id(p)])


def _multi_use(compute):
    """lin is called three times in one forward, interleaved with two other
    linears: backward store order lin, o2, lin, o1, lin (entries E0..E4)."""
    torch.manual_seed(11)
    m = nn.ModuleDict({"lin": nn.Linear(8, 8), "o1": nn.Linear(8, 8), "o2": nn.Linear(8, 8)})
    assert wgrad.convert(m) == 3
    x = torch.randn(4, 6, 8)
    with torch.autocast("cpu", dtype=_DT[compute], enabled=compute != "fp32"):
        h = m["lin"](x)
        h = m["o1"](torch.tanh(h))
        h = m["lin"](torch.tanh(h))
        h = m["o2"](torch.tanh(h))
        h = m["lin"](torch.tanh(h))
    with wgrad.capture() as store:
        h.float().square().sum().backward()
    names = {id(v): k for k, v in m.items()}
    assert [names[id(mod)] for mod, _ in store] == ["lin", "o2", "lin", "o1", "lin"]
    return m, store


@pytest.mark.parametrize("compute", ["fp32", "bf16"])
def test_flush_iter_sums_a_multi_use_linear_widened_in_store_order(compute):
    # covers: MODEL-I7
    # WOULD-FAIL-IF: a multi-use parameter's contributions are summed out of store order, or summed in the GEMM
    # dtype before widening, instead of widened and summed exactly as flush does
    m, store = _multi_use(compute)
    lin = m["lin"]
    parts = [fn() for mod, fn in store if mod is lin]
    ref = wgrad.flush(list(store))
    got = {id(p): g for p, g in wgrad.flush_iter(store, "forward")}
    for i, p in enumerate((lin.weight, lin.bias)):
        c = [part[i] for part in parts]
        assert c[0].dtype == _DT[compute]
        in_order = (c[0].float() + c[1].float()) + c[2].float()
        assert got[id(p)].dtype == torch.float32
        assert torch.equal(got[id(p)], ref[id(p)]) and torch.equal(got[id(p)], in_order)
        # Controls. fp32: the reversed summation order changes the bits. bf16: widened bf16 terms (8-bit
        # significands) mostly sum exactly in fp32, so order is not observable there; summing before widening is.
        if compute == "fp32":
            assert not torch.equal((c[2] + c[1]) + c[0], in_order)
        else:
            assert not torch.equal(((c[0] + c[1]) + c[2]).float(), in_order)


@pytest.mark.parametrize("compute", ["fp32", "bf16"])
@pytest.mark.parametrize(
    "order, expect",
    [
        ("store", ["E0", "E2", "E4", "lin.w", "lin.b", "E1", "o2.w", "o2.b", "E3", "o1.w", "o1.b"]),
        (
            "forward",
            ["E3", "o1.w", "o1.b", "E1", "o2.w", "o2.b", "E0", "E2", "E4", "lin.w", "lin.b"],
        ),
    ],
)
def test_flush_iter_runs_one_entry_at_a_time_and_yields_at_the_last_contribution(
    compute, order, expect
):
    # covers: MODEL-I7
    # WOULD-FAIL-IF: flush_iter computes entries ahead of the yields, splits a module's entries, runs modules out
    # of the requested order, or keeps an earlier entry's GEMM outputs resident beyond the incomplete sums
    import weakref

    m, store = _multi_use(compute)
    name = {id(p): f"{k}.{s}" for k, v in m.items() for s, p in (("w", v.weight), ("b", v.bias))}
    lin_entries = {"E0", "E2", "E4"}
    log: list = []
    outs: list = []
    resident: list = []

    def hook(i, fn):
        # Test hook around one store entry: before its GEMMs run, count the earlier GEMM outputs still
        # alive against the allowance, lin's sums while they are open (2 between its first and last entry).
        def run():
            open_sums = 2 if any(e in log for e in lin_entries) and "lin.b" not in log else 0
            resident.append((sum(r() is not None for r in outs), open_sums))
            log.append(f"E{i}")
            dw, db = fn()
            outs.extend(weakref.ref(t) for t in (dw, db))
            return dw, db

        return run

    ref = wgrad.flush(list(store))
    store[:] = [(mod, hook(i, fn)) for i, (mod, fn) in enumerate(store)]
    bits = []
    for p, g in wgrad.flush_iter(store, order):
        log.append(name[id(p)])
        bits.append(torch.equal(g.to(p.dtype), ref[id(p)]))
        del g
    assert log == expect
    assert bits == [True] * 6
    assert len(resident) == 5 and all(live <= allowed for live, allowed in resident), resident
    # Control: the allowance is only spent while lin's sums are open; everywhere else nothing may survive.
    assert [a for _, a in resident].count(2) == 2


def test_flush_iter_refuses_unknown_orders_and_shared_parameters():
    # covers: MODEL-I7
    # WOULD-FAIL-IF: an order other than "store" or "forward" is accepted (or refused only on the first next()),
    # or a parameter shared by two modules is grouped and summed out of store order instead of refused
    _, store = _multi_use("fp32")
    for bad in ("backward", "Store", "", None):
        with pytest.raises(ValueError, match="order"):
            wgrad.flush_iter(store, bad)
    assert len(store) == 5
    assert list(wgrad.flush_iter([], "store")) == []
    a = wgrad.SplitLinear(4, 4)
    b = wgrad.SplitLinear(4, 4)
    b.weight = a.weight
    with wgrad.capture() as shared:
        b(torch.tanh(a(torch.randn(3, 4)))).sum().backward()
    with pytest.raises(ValueError, match="two modules"):
        wgrad.flush_iter(shared, "store")


@pytest.mark.parametrize("compute", ["bf16", "fp16"])
def test_flush_iter_gather_dtypes_follow_owned_params_and_the_recipe(compute):
    # covers: MODEL-I7
    # WOULD-FAIL-IF: gather_dtypes gathers a LayerNorm or embedding parameter in the compute dtype, gathers an owned
    # linear in fp32 under an autocast recipe, or narrows anything when the recipe does not autocast
    cfg = GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        d_model=32,
        seq_len=8,
        compute_dtype=compute,
        boundary_dtype=compute,
    )
    layers = build_layers(cfg, 7)
    owned = wgrad.owned_params(layers)
    params = list(layers.parameters())
    assert owned and len(owned) < len(params)
    table = wgrad.gather_dtypes(layers, compute, cfg.autocast_enabled())
    assert set(table) == {id(p) for p in params}
    assert {k for k, dt in table.items() if dt == _DT[compute]} == owned
    norms = [m for m in layers.modules() if isinstance(m, (nn.LayerNorm, nn.Embedding))]
    assert norms
    for mod in norms:
        assert all(table[id(p)] == torch.float32 for p in mod.parameters(recurse=False))
    for c, autocast in ((compute, False), ("fp32", True), ("fp32", False)):
        assert set(wgrad.gather_dtypes(layers, c, autocast).values()) == {torch.float32}
    with pytest.raises(ValueError, match="unknown precision"):
        wgrad.gather_dtypes(layers, "fp12", True)


# ======================================================= HF modules (ADR-0009)
#
# Every HF model here is a tiny, randomly initialized config of its pinned
# transformers config class (hfm.tiny_config) or a checkpoint this
# repository's writer wrote; no network, no hub, no from_pretrained.

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HF_FAMILIES = {
    "llama": ("llama", {"tie_word_embeddings": False}),
    "llama_tied": ("llama", {"tie_word_embeddings": True}),
    "qwen3": ("qwen3", {}),
    "gemma2": (
        "gemma2",
        {"sliding_window": 8, "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0},
    ),
    "granite": (
        "granite",
        {
            "embedding_multiplier": 12.0,
            "logits_scaling": 8.0,
            "residual_multiplier": 0.2,
            "attention_multiplier": 0.25,
        },
    ),
}
GATE_RECIPE = {"compute_dtype": "bf16", "seq_len": 16, "mb_size": 2}


def _hfm():
    from wantrain.model import hfm

    return hfm


def _tiny(fam: str) -> dict:
    mt, over = HF_FAMILIES[fam]
    return _hfm().tiny_config(mt, **over)


def _hf_cfg(hf_config: dict, **kw) -> GPTConfig:
    hfm = _hfm()
    fields = hfm.config_fields(hf_config)
    return GPTConfig(
        arch="hf",
        hf_config=hf_config,
        hf_buffers=hfm.buffer_record(hf_config),
        seq_len=16,
        **fields,
        **kw,
    )


def _ids(V: int, rows: int = 2, S: int = 16, seed: int = 3) -> torch.Tensor:
    return torch.randint(0, V, (rows, S), generator=torch.Generator().manual_seed(seed))


def _sha_t(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _assert_compared(rep: dict, n_layers: int = 4):
    """The gate ran its whole comparison: every cut (deepest with and without
    recompute, one stage, each single cut), fp32 and bf16, M 1 and 2."""
    assert rep["first_difference"] is None, rep["detail"]
    assert rep["dtypes"] == ["fp32", "bf16"]
    assert len(rep["cuts"]) == (n_layers + 4) * 2 * 2, rep["cuts"]
    assert any("recompute" in c for c in rep["cuts"])


def _assert_platform_cause(rep: dict):
    """BA-463: on an admitted platform only the in-process FP-PYC refusal (this
    pytest was not started with -X pycache_prefix) may remain; elsewhere the
    FP-TORCH-RECORDS refusal names the platform tag. Never skipped."""
    hfm = _hfm()
    if hfm.platform_admitted():
        assert rep["cause"] in (None, "FP-PYC"), (rep["cause"], rep["detail"])
    else:
        tag = rep["fingerprint"]["torch"]["platform_tag"]
        assert rep["cause"] == "FP-TORCH-RECORDS" and tag in rep["detail"], rep["detail"]


# ------------------------------------------------------------ the parity gate


@pytest.mark.parametrize("fam", list(HF_FAMILIES))
def test_hfm_gate_units_equal_vanilla_bitwise(fam):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: a unit window differs from HF's vanilla forward at any cut (logits, loss hex or any gradient,
    # fp32 or bf16, M 1 or 2, deferred W, recompute), the gate skips cuts or dtypes, or a non-admitted platform
    # is passed without its FP-TORCH-RECORDS refusal
    rep = _hfm().gate(_tiny(fam), GATE_RECIPE, "cpu")
    _assert_compared(rep)
    _assert_platform_cause(rep)
    assert rep["device"] == "cpu" and rep["device_class"].startswith("cpu:")
    assert 0 < rep["build_bytes"] < _hfm().GATE_BUILD_BYTES
    trig = rep["triggers"]
    assert trig["tie"] == (1 if fam in ("llama_tied", "gemma2") else 0)
    if fam == "gemma2":
        assert trig["window"] == 2 and trig["softcap"] == 2  # sliding layers 0, 2 of 4; both caps
    else:
        assert trig["window"] == 0 and trig["softcap"] == 0


@pytest.mark.parametrize(
    "control, fam, patch",
    [
        ("to_empty buffers", "llama", ("_chain_buffers", "zero")),
        ("exit one unit late", "llama", ("_exit_unit", "late")),
        ("tied use dropped", "llama_tied", ("_compose_tied", "head_only")),
    ],
)
def test_hfm_gate_refuses_a_divergent_unit_chain(monkeypatch, control, fam, patch):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: the gate admits a chain whose buffers were zeroed (to_empty), whose units stop one unit late,
    # or that drops a tied table's embedding use: a vacuous comparison passes every deliberately wrong chain
    hfm = _hfm()
    name, kind = patch

    def zero(root):
        for b in root.buffers():
            b.zero_()

    impl = {
        "zero": zero,
        "late": lambda u: u + 2,
        "head_only": lambda parts: parts[0],
    }[kind]
    monkeypatch.setattr(hfm, name, impl)
    rep = hfm.gate(_tiny(fam), GATE_RECIPE, "cpu")
    assert rep["cause"] == "GATE-DIFF", (control, rep["cause"], rep["detail"])
    assert rep["first_difference"]
    if kind == "head_only":
        assert "model.embed_tokens.weight" in rep["first_difference"]
    assert not rep["ok"]


def test_hfm_gate_refuses_bert_and_names_the_cause():
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a bidirectional model in the causal-LM mapping (bert) passes the gate; the causality probe
    # must refuse it naming GATE-CAUSAL
    hfm = _hfm()
    cfg = hfm.tiny_config("bert", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    rep = hfm.gate(cfg, GATE_RECIPE, "cpu")
    assert rep["cause"] == "GATE-CAUSAL" and not rep["ok"]


@pytest.mark.parametrize("rolled", [False, True])
def test_hfm_tiny_mixtral_without_a_row_refuses_moe_no_row(monkeypatch, rolled):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: the D8 classifier misses Mixtral's sparse-MoE block, or a block without a SUBSTITUTE row
    # (or with its rows rolled back) runs HF's own MoE forward instead of being refused naming its class
    hfm = _hfm()
    monkeypatch.setattr(hfm, "SUBSTITUTE", {})
    if rolled:
        monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset({"model-3-6a"}))
    rep = hfm.gate(hfm.tiny_config("mixtral"), {**GATE_RECIPE, "compute_dtype": "fp32"}, "cpu")
    assert rep["cause"] == "MOE-NO-ROW", rep["detail"]
    assert "MixtralSparseMoeBlock" in rep["detail"] and "model.layers.0.mlp" in rep["detail"]
    assert ("model-3-6a" in rep["detail"]) == rolled
    um = hfm.unit_map(hfm.tiny_config("mixtral"))
    assert um.moe_blocks == {i: f"model.layers.{i}.mlp" for i in range(um.n)}
    assert hfm.unit_map(_tiny("llama")).moe_blocks == {}  # dense: nothing classified


# ------------------------------------------------- build, units and the chain


@pytest.mark.parametrize("fam", ["llama", "llama_tied", "gemma2"])
def test_hfm_build_layers_units_compose_to_vanilla(fam):
    # covers: MODEL-I9
    # WOULD-FAIL-IF: build_layers for arch "hf" does not return n+2 windowed units over one tree whose chain
    # equals the vanilla logits bitwise, converts a tied or read-elsewhere Linear, or a unit holds a parameter
    # outside its own modules
    hc = _tiny(fam)
    cfg = _hf_cfg(hc)
    units = build_layers(cfg, 5)
    hfm = _hfm()
    assert isinstance(units, hfm.HFUnits) and len(units) == cfg.n_layer + 2
    assert all(getattr(u, "takes_ids", False) for u in units)
    ids = _ids(cfg.vocab_size)
    with torch.no_grad():
        ref = units.vanilla(ids)
        h = ids
        for u in units:
            h = call_layer(u, h, ids=ids)
    assert torch.equal(h, ref)
    head = units.root.lm_head
    tied = fam != "llama"
    assert (type(head) is nn.Linear) == tied and (type(head) is wgrad.SplitLinear) == (not tied)
    assert units.tied == ((0, cfg.n_layer + 1) if tied else None)
    q = units.root.model.layers[0].self_attn.q_proj
    assert type(q) is wgrad.SplitLinear
    names = [[k for k, _ in u.named_parameters()] for u in units]
    assert names[0] == ["model.embed_tokens.weight"]
    assert all(k.startswith(f"model.layers.{i}.") for i in range(cfg.n_layer) for k in names[i + 1])
    # From config: the seeded full build, identical under one seed, different under another.
    again = build_layers(cfg, 5)
    other = build_layers(cfg, 6)
    for (k, a), (_, b) in zip(units.root.named_parameters(), again.root.named_parameters()):
        assert torch.equal(a, b), k
    assert not torch.equal(
        units.root.model.embed_tokens.weight, other.root.model.embed_tokens.weight
    )


def test_hfm_skeleton_draws_nothing_and_keeps_real_buffers():
    # covers: MODEL-I9
    # WOULD-FAIL-IF: the skeleton draws from the global generator, materializes parameters, leaves a buffer on
    # meta or different from a real CPU build's, or is built in eval mode or with a non-eager attention
    hfm = _hfm()
    hc = _tiny("llama")
    torch.manual_seed(123)
    s0 = torch.get_rng_state()
    sk = hfm.skeleton(hc)
    assert torch.equal(s0, torch.get_rng_state())
    assert all(p.device.type == "meta" for p in sk.parameters())
    bufs = dict(sk.named_buffers())
    assert bufs and all(b.device.type == "cpu" for b in bufs.values())
    real = hfm._full_build(hc, 0, swap=False)
    for k, b in real.named_buffers():
        assert torch.equal(bufs[k], b), k
    assert all(m.training for m in sk.modules())
    assert sk.config._attn_implementation == "eager"


def test_hfm_buffer_record_pins_bytes_and_refuses_tampering():
    # covers: MODEL-I9
    # WOULD-FAIL-IF: a buffer record whose bytes do not hash to its sha256, that lacks or adds a buffer, or that
    # was built for another config is applied; or build runs without a pinned record
    hfm = _hfm()
    hc = _tiny("llama")
    rec = hfm.buffer_record(hc)
    name = "model.rotary_emb.inv_freq"
    assert (
        set(rec["tensors"]) == {k for k, _ in hfm.skeleton(hc).named_buffers()}
        and name in rec["tensors"]
    )
    sk = hfm.skeleton(hc)
    with torch.no_grad():
        sk.model.rotary_emb.inv_freq.zero_()
    hfm.apply_buffer_record(sk, rec)  # overwritten from the record, not trusted from the build
    assert _sha_t(sk.model.rotary_emb.inv_freq) == rec["tensors"][name]["sha256"]
    bad = copy.deepcopy(rec)
    raw = bytearray(base64.b64decode(bad["tensors"][name]["b64"]))
    raw[0] ^= 1
    bad["tensors"][name]["b64"] = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(hfm.HFRefused, match=name) as e:
        hfm.apply_buffer_record(hfm.skeleton(hc), bad)
    assert e.value.cause == "BUFFER-PIN"
    short = copy.deepcopy(rec)
    del short["tensors"][name]
    with pytest.raises(hfm.HFRefused, match=name) as e:
        hfm.apply_buffer_record(hfm.skeleton(hc), short)
    assert e.value.cause == "BUFFER-RECORD"
    cfg = _hf_cfg(hc)
    cfg.hf_buffers = None
    with pytest.raises(hfm.HFRefused) as e:
        hfm.build(cfg, 0)
    assert e.value.cause == "BUFFER-RECORD"
    cfg.hf_buffers = hfm.buffer_record(_tiny("qwen3"))
    with pytest.raises(hfm.HFRefused, match="another hf_config") as e:
        hfm.build(cfg, 0)


def test_hfm_units_refuse_eval_mode_and_in_place_stub_writes(monkeypatch):
    # covers: MODEL-I9
    # WOULD-FAIL-IF: a unit runs with a module in eval mode (the gate would check another code path than the
    # stages run), or a write into a stride-0 NaN stub goes unnoticed instead of refusing STUB-WRITE
    hfm = _hfm()
    cfg = _hf_cfg(_tiny("llama"))
    units = build_layers(cfg, 1)
    ids = _ids(cfg.vocab_size)
    with torch.no_grad():
        x = call_layer(units[1], call_layer(units[0], ids, ids=ids), ids=ids)

        def scribble(mod, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if isinstance(h, torch.Tensor) and 0 in h.stride():
                h.mul_(1.0)

        hook = torch.nn.modules.module.register_module_forward_pre_hook(
            lambda m, a: scribble(m, a, {})
        )
        try:
            with pytest.raises(hfm.HFRefused) as e:
                call_layer(units[3], x, ids=ids)  # layers 0 and 1 are stubs for unit 3
            assert e.value.cause == "STUB-WRITE"
        finally:
            hook.remove()
        call_layer(units[3], x, ids=ids)  # the same call without the write is admitted
        units.root.model.layers[1].eval()
        with pytest.raises(hfm.HFRefused, match="model.layers.1") as e:
            call_layer(units[2], x, ids=ids)
        assert e.value.cause == "EVAL-MODE"


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize(
    "change, cause, named",
    [
        ({"attn_implementation": "sdpa"}, "ATTN-IMPL", "attn_implementation"),
        ({"attention_dropout": 0.1}, "DROPOUT", "attention_dropout"),
        (
            {"rope_parameters": {"rope_type": "dynamic", "factor": 2.0, "rope_theta": 10000.0}},
            "ROPE-DYNAMIC",
            "dynamic",
        ),
        (
            {"auto_map": {"AutoModelForCausalLM": "remote.Model"}},
            "AUTO-MAP",
            "AutoModelForCausalLM",
        ),
        ({"model_type": "t5"}, "MODEL-TYPE", "t5"),
        ({"model_type": "not_a_model"}, "MODEL-TYPE", "not_a_model"),
    ],
)
def test_hfm_check_config_refusals_name_their_cause(change, cause, named):
    # covers: MODEL-I9
    # WOULD-FAIL-IF: an sdpa attention, a dropout above 0, a dynamic rope, remote code, an encoder-decoder or an
    # unpinned model_type passes check_config instead of being refused naming its cause and field
    hfm = _hfm()
    with pytest.raises(hfm.HFRefused, match=named) as e:
        hfm.check_config({**_tiny("llama"), **change}, None)
    assert e.value.cause == cause


def test_hfm_check_config_refuses_recipe_conflicts():
    # covers: MODEL-I9
    # WOULD-FAIL-IF: LoRA, a narrowed boundary or a longrope past its original window is admitted under arch hf
    hfm = _hfm()
    hc = _tiny("llama")
    for kw, cause in (({"lora_rank": 1}, "LORA"), ({"boundary_dtype": "bf16"}, "BOUNDARY-DTYPE")):
        with pytest.raises(hfm.HFRefused) as e:
            hfm.check_config(hc, GPTConfig(**kw))
        assert e.value.cause == cause
    long = {
        **hc,
        "rope_parameters": {
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "short_factor": [1.0] * 8,
            "long_factor": [1.0] * 8,
            "original_max_position_embeddings": 8,
        },
        "max_position_embeddings": 64,
    }
    with pytest.raises(hfm.HFRefused, match="original_max_position_embeddings=8") as e:
        hfm.check_config(long, GPTConfig(seq_len=16))
    assert e.value.cause == "ROPE-LONGROPE"
    hfm.check_config(long, GPTConfig(seq_len=8))  # within its window: admitted


@pytest.mark.parametrize(
    "kw, field",
    [
        ({"hf_config": None}, "hf_config"),
        ({"hf_config": {"vocab_size": 3}}, "hf_config"),
        ({"hf_impl": "native"}, "hf_impl"),
        ({"lora_rank": 1}, "lora_rank"),
        ({"boundary_dtype": "bf16"}, "boundary_dtype"),
        ({"fp8_boundaries": True}, "boundary_dtype"),
        ({"compute_dtype": "fp16", "boundary_dtype": "fp16"}, "boundary_dtype"),
        ({"compute_dtype": "fp16"}, "compute_dtype"),
    ],
)
def test_hfm_validate_refuses_hf_rules_naming_the_field(kw, field):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: GPTConfig.validate admits an arch hf config without its hf_config, with another hf_impl, LoRA,
    # a narrowed boundary or an fp16 compute, or refuses without naming the field
    base = {"arch": "hf", "hf_config": {"model_type": "llama"}}
    with pytest.raises(ValueError, match=field):
        GPTConfig(**{**base, **kw}).validate()
    GPTConfig(**base).validate()


def test_hfm_validate_refuses_mirrored_fields_that_differ_from_the_census():
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a spec whose n_layer, n_head or tie differs from the pinned census validates, so the planner
    # prices one model and the stages build another
    census = [
        {"unit": 0, "kind": "prologue", "tied": False},
        {"unit": 1, "kind": "layer", "attn_heads": 4, "tied": False},
        {"unit": 2, "kind": "layer", "attn_heads": 4, "tied": False},
        {"unit": 3, "kind": "epilogue", "tied": False},
    ]
    ok = dict(arch="hf", hf_config={"model_type": "llama"}, hf_census=census, n_layer=2, n_head=4)
    GPTConfig(**ok).validate()
    for k, v in (("n_layer", 3), ("n_head", 2), ("tie_embeddings", True)):
        with pytest.raises(ValueError, match=k):
            GPTConfig(**{**ok, k: v}).validate()


def test_hfm_validate_never_imports_transformers():
    # covers: MODEL-I10
    # WOULD-FAIL-IF: gpt.py imports transformers at import or in validate(), which the coordinator and the
    # planner run without it
    code = (
        "import sys\n"
        "from wantrain.model.gpt import GPTConfig\n"
        "GPTConfig(arch='hf', hf_config={'model_type': 'llama'}).validate()\n"
        "GPTConfig(moe_experts=4, moe_top_k=2).validate()\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'transformers'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# ------------------------------------------------------------- load and export


def _source(tmp_path, hc: dict, shard_bytes: int = 40_000, seed: int = 7):
    """A sharded checkpoint of a seeded full build, written by this
    repository's writer under HF's names (a tied table once, as HF saves it)."""
    hfm = _hfm()
    root = hfm._full_build(hc, seed, swap=False)
    tensors = {k: p.detach().clone() for k, p in root.named_parameters()}
    path = str(tmp_path / "ckpt")
    hfm.write_safetensors(tensors, path, shard_bytes)
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump(hc, fh)
    return path, tensors


def test_hfm_span_build_loads_only_its_units_and_equals_the_full_load(tmp_path):
    # covers: MODEL-I4
    # WOULD-FAIL-IF: a checkpoint span build materializes a unit outside its span, leaves an owned parameter on
    # meta or at random values, or loads it with bytes other than the full load's
    hc = _tiny("llama")
    path, src = _source(tmp_path, hc)
    with open(os.path.join(path, hf.INDEX)) as fh:
        assert len(set(json.load(fh)["weight_map"].values())) > 1  # really sharded
    cfg = _hf_cfg(hc, hf_model=path, hf_model_sha256=hf.checkpoint_digest(path))
    part = build_layers(cfg, 0, span=(1, 3))
    full = build_layers(cfg, 0)
    for u, unit in enumerate(part):
        for k, p in unit.named_parameters():
            if 1 <= u < 3:
                assert p.device.type == "cpu" and torch.equal(p, src[k]), k
                assert torch.equal(p, dict(full[u].named_parameters())[k])
            else:
                assert p.device.type == "meta", k
    for k, p in full.root.named_parameters():
        assert torch.equal(p, src[k]), k
    ids = _ids(cfg.vocab_size)
    with torch.no_grad():
        h = ids
        for u in full:
            h = call_layer(u, h, ids=ids)
        assert torch.equal(h, full.vanilla(ids))


def test_hfm_load_refuses_missing_unused_misshapen_and_converted_names(tmp_path):
    # covers: MODEL-I4
    # WOULD-FAIL-IF: a missing tensor, an unused tensor, a shape or dtype-class disagreement or a transformers
    # v5 conversion name is loaded (left at meta, ignored or renamed) instead of refused naming it
    hfm = _hfm()
    hc = _tiny("llama")
    path, src = _source(tmp_path, hc)
    cases = {
        "LOAD-MISSING": (
            {k: v for k, v in src.items() if k != "model.layers.2.mlp.up_proj.weight"},
            "model.layers.2.mlp.up_proj.weight",
        ),
        "LOAD-UNUSED": (
            {**src, "model.layers.9.extra.weight": torch.zeros(3)},
            "model.layers.9.extra.weight",
        ),
        "LOAD-SHAPE": ({**src, "model.norm.weight": torch.zeros(65)}, "model.norm.weight"),
    }
    for cause, (tensors, named) in cases.items():
        d = str(tmp_path / cause)
        hfm.write_safetensors(tensors, d, 10**9)
        with pytest.raises(hfm.HFRefused, match=named) as e:
            hfm.load(hfm.skeleton(hc), d)
        assert e.value.cause == cause
    d = str(tmp_path / "int")
    hfm.write_safetensors(
        {**src, "model.norm.weight": torch.zeros(64, dtype=torch.int32)}, d, 10**9
    )
    with pytest.raises(hfm.HFRefused, match="model.norm.weight") as e:
        hfm.load(hfm.skeleton(hc), d)
    assert e.value.cause == "LOAD-SHAPE"
    oc = hfm.tiny_config("olmoe")
    sk = hfm.skeleton(oc)
    tensors = {k: torch.zeros(p.shape) for k, p in sk.named_parameters() if "experts" not in k}
    tensors["model.layers.0.mlp.experts.0.gate_proj.weight"] = torch.zeros(2, 2)
    d = str(tmp_path / "olmoe")
    hfm.write_safetensors(tensors, d, 10**9)
    with pytest.raises(hfm.HFRefused, match="experts.0.gate_proj") as e:
        hfm.load(sk, d)
    assert e.value.cause == "LOAD-NAME"


def test_hfm_export_round_trip_keeps_every_sha256(tmp_path):
    # covers: MODEL-I4
    # WOULD-FAIL-IF: load-export-load changes any tensor's bytes (a -0 turned +0, a NaN payload canonicalized),
    # renames or drops a tied key, or the index loses a tokenizer file's sha256
    hfm = _hfm()
    hc = _tiny("llama_tied")
    path, src = _source(tmp_path, hc)
    w = src["model.embed_tokens.weight"]
    w[0, 0] = -0.0
    w.view(torch.int32)[0, 1] = 0x7FC00ABC  # a NaN with a payload
    hfm.write_safetensors(src, path, 40_000)
    with open(os.path.join(path, "tokenizer.json"), "w") as fh:
        fh.write('{"tok": 1}')
    sk = hfm.skeleton(hc)
    hfm.load(sk, path)
    assert (
        sk.lm_head.weight is sk.model.embed_tokens.weight
    )  # re-tied from the skeleton's sharing map
    out = str(tmp_path / "export")
    hfm.export(sk, out, list(src))
    idx = json.load(open(os.path.join(out, hf.INDEX)))
    assert set(idx["weight_map"]) == set(src) and "lm_head.weight" not in idx["weight_map"]
    assert idx["metadata"]["tokenizer_sha256"] == {
        "tokenizer.json": hashlib.sha256(b'{"tok": 1}').hexdigest()
    }
    back = dict(hf._iter_shards(out))
    for k, t in src.items():
        assert _sha_t(back[k]) == _sha_t(t), k
    sk2 = hfm.skeleton(hc)
    hfm.load(sk2, out)
    for k, p in sk2.named_parameters():
        assert _sha_t(p) == _sha_t(src[k]), k
    with pytest.raises(hfm.HFRefused) as e:
        hfm.export(hfm.skeleton(hc), str(tmp_path / "meta"), list(src))
    assert e.value.cause == "LOAD-MISSING"


@pytest.mark.parametrize("mt", ["mixtral", "qwen2_moe", "qwen3_moe", "deepseek_v3"])
def test_hfm_hub_expert_map_round_trips_and_equals_transformers_conversion(tmp_path, mt):
    # covers: MODEL-I4
    # WOULD-FAIL-IF: a hub per-expert tensor lands in the wrong rows of gate_up_proj / down_proj, an element is
    # written twice or not at all, the inverse on export differs, or the map differs from transformers 5.19.0's
    # own MergeModulelist + Concatenate conversion of the same tensors
    from transformers.core_model_loading import Concatenate, MergeModulelist

    hfm = _hfm()
    hc = hfm.tiny_config(mt)
    sk = hfm.skeleton(hc)
    params = dict(sk.named_parameters())
    banks = sorted(k for k in params if k.endswith(("experts.gate_up_proj", "experts.down_proj")))
    assert banks
    src, base = {}, 0
    for k, p in params.items():
        if k in banks:
            continue
        hub = (
            k.replace(".mlp.", ".block_sparse_moe.") if mt == "mixtral" and ".mlp.gate." in k else k
        )
        src[hub] = (torch.arange(p.numel(), dtype=torch.float32) + base).reshape(p.shape) / 7
        base += p.numel()
    for bank in sorted({b.rpartition(".")[0] for b in banks}):
        E, f2, h = params[bank + ".gate_up_proj"].shape
        f = f2 // 2
        for j in range(E):
            for part, shape in (("gate", (f, h)), ("up", (f, h)), ("down", (h, f))):
                t = (torch.arange(f * h, dtype=torch.float32) + base).reshape(shape) / 3
                base += f * h
                t[0, 0] = -0.0
                t.view(torch.int32)[0, 1] = 0x7FC00000 + j
                src[hfm._hub_name(mt, bank.rsplit(".experts", 1)[0], j, part)] = t
    path = str(tmp_path / mt)
    hfm.write_safetensors(src, path, 10**9)
    hfm.load(sk, path)
    loaded = dict(sk.named_parameters())
    for bank in sorted({b.rpartition(".")[0] for b in banks}):
        E = loaded[bank + ".gate_up_proj"].shape[0]
        pre = bank.rsplit(".experts", 1)[0]
        gates = [src[hfm._hub_name(mt, pre, j, "gate")] for j in range(E)]
        ups = [src[hfm._hub_name(mt, pre, j, "up")] for j in range(E)]
        downs = [src[hfm._hub_name(mt, pre, j, "down")] for j in range(E)]
        merged = MergeModulelist(dim=0).convert({"g": gates, "u": ups}, ["g", "u"], ["t"])
        ref = Concatenate(dim=1).convert(merged, ["g", "u"], ["t"])["t"]
        ref_down = MergeModulelist(dim=0).convert({"d": downs}, ["d"], ["t"])["t"]
        assert _sha_t(loaded[bank + ".gate_up_proj"]) == _sha_t(ref)
        assert _sha_t(loaded[bank + ".down_proj"]) == _sha_t(ref_down)
    out = str(tmp_path / f"{mt}-export")
    hfm.export(sk, out, list(src))
    back = dict(hf._iter_shards(out))
    assert set(back) == set(src)
    for k, t in src.items():
        assert _sha_t(back[k]) == _sha_t(t), k


def test_hfm_checkpoint_digest_changes_with_a_single_weight(tmp_path):
    # covers: MODEL-I4
    # WOULD-FAIL-IF: two checkpoints differing in one element share a digest, so stages holding different
    # weights would train together
    hfm = _hfm()
    path, src = _source(tmp_path, _tiny("llama"))
    before = hf.checkpoint_digest(path)
    assert before == hf.checkpoint_digest(path)
    src["model.norm.weight"][0] += 1e-3
    hfm.write_safetensors(src, path, 40_000)
    assert hf.checkpoint_digest(path) != before


@pytest.mark.parametrize(
    "entry, why",
    [
        ({"dtype": "F32", "shape": [-1], "data_offsets": [4, 0]}, "shape"),
        ({"dtype": "F32", "shape": [2], "data_offsets": [0, 1 << 40]}, "data_offsets"),
        ({"dtype": "F32", "shape": [2], "data_offsets": [-8, 0]}, "data_offsets"),
        ({"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}, "shape and dtype"),
        ({"dtype": "F32", "shape": ["2"], "data_offsets": [0, 8]}, "shape"),
        ("not-a-dict", "dtype"),
    ],
)
def test_hfm_safetensors_header_is_validated_before_any_read(tmp_path, entry, why):
    # covers: MODEL-I4
    # WOULD-FAIL-IF: the reader trusts a downloaded checkpoint's header: a negative shape with reversed offsets
    # reads the whole file into memory, offsets past the data or before it (into the header) are read as weights,
    # or a size that disagrees with the shape is accepted
    good = torch.tensor([1.0, -0.0])
    header = json.dumps({"t": entry}).encode()
    header += b" " * (-len(header) % 8)
    f = tmp_path / "x.safetensors"
    f.write_bytes(len(header).to_bytes(8, "little") + header + good.numpy().tobytes())
    with pytest.raises(ValueError, match=why):
        hf.checkpoint_digest(str(f))
    ok = {"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    header = json.dumps(ok).encode()
    f.write_bytes(len(header).to_bytes(8, "little") + header + good.numpy().tobytes())
    assert torch.equal(dict(hf._iter_shards(str(f)))["t"].view(torch.int32), good.view(torch.int32))


# ------------------------------------------------------------------- census


@pytest.mark.parametrize("fam", ["llama", "llama_tied", "gemma2"])
def test_hfm_census_param_shapes_match_the_units(fam):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: the census lists a shared table twice, or a name or shape differing from the unit's own
    # named_parameters, so the coordinator would price pieces for another model
    hfm = _hfm()
    hc = _tiny(fam)
    census = hfm.unit_census(hc)
    us = hfm.units(hfm.skeleton(hc), hfm.unit_map(hc))
    expect: dict = {}
    for name, p in us.named_parameters():
        u, _, k = name.partition(".")
        expect.setdefault(int(u), []).append([k, list(p.shape)])
    assert [c["unit"] for c in census] == list(range(len(us)))
    for c in census:
        assert c["param_shapes"] == expect.get(c["unit"], []), c["unit"]
        assert c["params"] == sum(torch.Size(s).numel() for _, s in c["param_shapes"])
    flat = [k for c in census for k, _ in c["param_shapes"]]
    assert len(flat) == len(set(flat))
    tied = fam != "llama"
    assert ("lm_head.weight" in flat) == (not tied)
    assert [c["kind"] for c in census] == ["prologue"] + ["layer"] * 4 + ["epilogue"]
    assert census[1]["attn_heads"] == 4 and census[1]["head_dim"] == 16
    if fam == "gemma2":
        assert [c["window"] for c in census[1:5]] == [8, None, 8, None]
        assert census[1]["softcap"] == 50.0 and census[5]["softcap"] == 30.0
    fields = hfm.config_fields(hc)
    assert fields == {
        "n_layer": 4,
        "d_model": 64,
        "n_head": 4,
        "vocab_size": 128,
        "tie_embeddings": tied,
    }


# ------------------------------------------- environment guard and fingerprint

_PROBE = (
    "import json, sys\n"
    "from wantrain.model import hfm\n"
    "import transformers\n"
    "out = {}\n"
    "try:\n"
    "    hfm.guard_environment(); out['guard'] = None\n"
    "except hfm.HFRefused as e:\n"
    "    out['guard'] = [e.cause, str(e)]\n"
    "fp = hfm.fingerprint()\n"
    "out['refusals'] = fp['refusals']; out['prefix_ok'] = fp['prefix_ok']\n"
    "out['admitted'] = hfm.platform_admitted(); out['tag'] = fp['torch']['platform_tag']\n"
    "print(json.dumps(out))\n"
)


def _probe(tmp_path, prefix=True, env_extra=None, pythonpath=None) -> dict:
    hfm = _hfm()
    env = hfm.stage_env(dict(os.environ))
    if env_extra:
        env.update(env_extra)
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(
            [pythonpath] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
    pre = str(tmp_path / "pyc")
    args = [sys.executable] + (hfm.interpreter_args(pre) if prefix else []) + ["-c", _PROBE]
    out = subprocess.run(args, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _causes(r) -> list:
    return [x.split(":")[0] for x in r["refusals"]]


@_LINUX_CUDA_FINGERPRINT
def test_hfm_guard_and_fingerprint_in_a_clean_launch(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a process started in stage_env with -X pycache_prefix is refused, or its fingerprint
    # reports a refusal other than the platform's missing TORCH_RECORDS entry (named, never skipped)
    r = _probe(tmp_path)
    assert r["guard"] is None and r["prefix_ok"] is True
    if r["admitted"]:
        assert r["refusals"] == []
    else:
        assert _causes(r) == ["FP-TORCH-RECORDS"] and r["tag"] in r["refusals"][0]


def test_hfm_guard_refuses_a_launch_without_the_pycache_prefix(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a process started without -X pycache_prefix (its bytecode cached beside the sources) passes
    # the guard or the fingerprint
    r = _probe(tmp_path, prefix=False)
    assert r["guard"][0] == "ENV-PREFIX"
    assert "FP-PYC" in _causes(r) and r["prefix_ok"] is False


def test_hfm_guard_refuses_ld_preload(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a launch with LD_PRELOAD set passes the guard (a preloaded library replaces kernels)
    r = _probe(tmp_path, env_extra={"LD_PRELOAD": str(tmp_path / "nothing.so")})
    assert r["guard"][0] == "ENV-LOADER" and "LD_PRELOAD" in r["guard"][1]


def test_hfm_guard_refuses_an_importable_probed_package(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: an importable mambapy (an algorithm switch transformers probes) passes the guard and leaves
    # the probe vector unflagged
    stub = tmp_path / "stubs" / "mambapy"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("")
    r = _probe(tmp_path, pythonpath=str(tmp_path / "stubs"))
    assert r["guard"][0] == "PROBED-PKG" and "mambapy" in r["guard"][1]
    assert "FP-PROBE" in _causes(r)


def test_hfm_fingerprint_refuses_an_edited_transformers_file(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a transformers file that differs from its RECORD sha256 (here in a symlinked copy of the
    # installed package under tmp_path, never the real environment) is not named by the fingerprint
    import importlib.metadata as md

    import transformers

    pkg = os.path.dirname(transformers.__file__)
    dist = md.distribution("transformers")
    site = tmp_path / "site"
    edited = os.path.join("transformers", "models", "llama", "configuration_llama.py")
    for dp, dns, fns in os.walk(pkg):
        dns[:] = [d for d in dns if d != "__pycache__"]
        rel = os.path.relpath(dp, os.path.dirname(pkg))
        os.makedirs(site / rel, exist_ok=True)
        for fn in fns:
            os.symlink(os.path.join(dp, fn), site / rel / fn)
    os.unlink(site / edited)
    with open(os.path.join(os.path.dirname(pkg), edited)) as fh:
        (site / edited).write_text(fh.read() + "\n# edited\n")
    info = str(dist._path)  # type: ignore[attr-defined]
    os.makedirs(site / os.path.basename(info))
    for fn in os.listdir(info):
        os.symlink(os.path.join(info, fn), site / os.path.basename(info) / fn)
    r = _probe(tmp_path, pythonpath=str(site))
    rec = [x for x in r["refusals"] if x.startswith("FP-RECORD")]
    assert len(rec) == 1 and "configuration_llama.py" in rec[0], r["refusals"]


def test_hfm_fingerprint_refuses_a_monkeypatched_transformers_function(monkeypatch):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a transformers function replaced in-process (a monkeypatch) keeps code_ok true
    hfm = _hfm()
    from transformers.models.llama import modeling_llama

    files = set(hfm._env_fingerprint()["files_of"])
    assert hfm._code_problem(files) is None
    monkeypatch.setattr(modeling_llama.LlamaMLP, "forward", lambda self, x: x)
    problem = hfm._code_problem(files)
    assert problem and "LlamaMLP.forward" in problem


@pytest.mark.parametrize("origin", ["installed", "stdlib"])
def test_hfm_fingerprint_refuses_a_monkeypatch_from_another_installed_package(monkeypatch, origin):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a transformers method or module function replaced by code of another installed
    # distribution (the way liger-style kernels patch HF) or of the standard library keeps code_ok true
    hfm = _hfm()
    import packaging.version
    from transformers.models.llama import modeling_llama

    files = set(hfm._env_fingerprint()["files_of"])
    assert hfm._code_problem(files) is None
    foreign = packaging.version.parse if origin == "installed" else json.dumps
    with monkeypatch.context() as mp:
        mp.setattr(modeling_llama.LlamaMLP, "forward", foreign)
        problem = hfm._code_problem(files)
        assert problem and "LlamaMLP.forward" in problem
    with monkeypatch.context() as mp:
        mp.setattr(modeling_llama, "rotate_half", foreign)
        problem = hfm._code_problem(files)
        assert problem and "modeling_llama.rotate_half" in problem
    assert hfm._code_problem(files) is None


def test_hfm_record_diff_names_the_first_differing_path():
    # covers: MODEL-I10
    # WOULD-FAIL-IF: record_diff misses a one-line difference, names another path, or reports equal texts as
    # different
    hfm = _hfm()
    a = "t/a.py,sha256=AAA,10\nt/b.py,sha256=BBB,20\nt/c.py,sha256=CCC,30\n"
    b = a.replace("t/b.py,sha256=BBB,20", "t/b.py,sha256=BBX,20")
    assert hfm.record_diff(a, b) == "t/b.py"
    assert hfm.record_diff(a, a) is None
    assert hfm.record_diff(a, a + "t/d.py,sha256=DDD,40\n") == "t/d.py"
    assert hfm.record_diff(a, "\n".join(a.splitlines()[:2])) == "t/c.py"
    text = hfm.record_text()
    assert text and hfm.record_diff(text, text) is None
    assert hfm.hf_code()["transformers_record_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_hfm_stage_env_and_interpreter_args():
    # covers: MODEL-I10
    # WOULD-FAIL-IF: stage_env keeps a loader variable or misses an HF_ENV entry, or interpreter_args does not
    # set the pycache prefix at interpreter start
    hfm = _hfm()
    env = hfm.stage_env(
        {
            "PATH": "/bin",
            "LD_PRELOAD": "x",
            "LD_LIBRARY_PATH": "y",
            "DYLD_INSERT_LIBRARIES": "z",
            "HF_HUB_OFFLINE": "0",
        }
    )
    assert env == {"PATH": "/bin", **hfm.HF_ENV}
    assert hfm.interpreter_args("/tmp/p") == ["-X", "pycache_prefix=/tmp/p"]


# ------------------------------------------------------------------- CLI


def test_hfm_cli_determinism_comes_before_any_build(monkeypatch, tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a CLI child (gate, record or census) builds, traces or compares before setup_determinism,
    # outside the stage's determinism regime
    from wantrain.runtime import determinism

    hfm = _hfm()
    calls: list = []
    monkeypatch.setattr(
        determinism, "setup_determinism", lambda seed: calls.append(("setup", seed))
    )
    fake_rep = {"device_class": "cpu:x", "fingerprint": {}}
    for name, ret in (
        ("gate", fake_rep),
        ("buffer_record", {}),
        ("unit_census", []),
        ("config_fields", {}),
        ("hf_code", {}),
        ("fingerprint", {}),
        ("guard_environment", None),
    ):
        monkeypatch.setattr(
            hfm, name, lambda *a, _n=name, _r=ret, **k: (calls.append((_n,)), _r)[1]
        )
    inp = tmp_path / "in.json"
    inp.write_text(
        json.dumps({"hf_config": {"model_type": "llama"}, "seed": 11, "recipe": GATE_RECIPE})
    )
    for cmd, first in (
        ("gate", "gate"),
        ("record", "guard_environment"),
        ("census", "unit_census"),
    ):
        calls.clear()
        assert hfm.main([cmd, "--in", str(inp), "--out", str(tmp_path / f"{cmd}.json")]) == 0
        assert calls[0] == ("setup", 11) and calls[1] == (first,), (cmd, calls)
        assert os.path.exists(tmp_path / f"{cmd}.json")


def test_hfm_cli_children_in_the_hf_environment(tmp_path):
    # covers: MODEL-I10, MODEL-I9
    # WOULD-FAIL-IF: the record child's output misses an I-10 key or differs from the in-process census and
    # buffer record, or the gate child (stage_env, fresh pycache prefix) fails a model the gate admits
    hfm = _hfm()
    hc = _tiny("llama")
    env = hfm.stage_env(dict(os.environ))
    outs = {}
    for cmd, payload in (
        ("record", {"hf_config": hc, "seed": 3}),
        ("gate", {"hf_config": hc, "seed": 3, "recipe": GATE_RECIPE, "devices": ["cpu"]}),
    ):
        inp, out = tmp_path / f"{cmd}.in", tmp_path / f"{cmd}.out"
        inp.write_text(json.dumps(payload))
        args = [
            sys.executable,
            *hfm.interpreter_args(str(tmp_path / f"pyc-{cmd}")),
            "-m",
            "wantrain.model.hfm",
        ]
        res = subprocess.run(
            args + [cmd, "--in", str(inp), "--out", str(out)],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert res.returncode == 0, res.stderr[-2000:]
        outs[cmd] = json.loads(out.read_text())
    rec = outs["record"]
    assert set(rec) == {
        "record",
        "census",
        "fields",
        "hf_code",
        "record_text_gz_b64",
        "fingerprint",
    }
    assert rec["census"] == hfm.unit_census(hc)
    mine = hfm.buffer_record(hc)
    assert {k: v["sha256"] for k, v in rec["record"]["tensors"].items()} == {
        k: v["sha256"] for k, v in mine["tensors"].items()
    }
    assert rec["fields"] == hfm.config_fields(hc) and rec["hf_code"] == hfm.hf_code()
    (rep,) = outs["gate"]["reports"]
    _assert_compared(rep)
    if hfm.platform_admitted():
        assert rep["ok"] and rep["cause"] is None, rep["detail"]
    else:
        assert rep["cause"] == "FP-TORCH-RECORDS"
    assert outs["gate"]["device_classes"] == [rep["device_class"]]


# ------------------------------------------------------------- rollback (BA-467)

_LLAMA_HF = {"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 4}


@pytest.mark.parametrize("piece", ["runtime-7c", "runtime-7d", "model-4b", "control-6"])
def test_hfm_rollback_model_4a_pieces_refuse_arch_hf(monkeypatch, piece):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a piece rolled back to 'model-4a' leaves arch hf reachable, or validate's refusal does not
    # name the piece
    cfg = GPTConfig(arch="hf", hf_config=dict(_LLAMA_HF))
    cfg.validate()
    assert rollback.target(piece) is None
    monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset({piece}))
    assert rollback.target(piece) == "model-4a"
    with pytest.raises(ValueError, match=piece) as e:
        cfg.validate()
    assert "model-4a" in str(e.value)
    GPTConfig().validate()  # a dense builtin config is unaffected


def test_hfm_rollback_moe_rows_refuse_hf_moe_and_keep_native_moe(monkeypatch):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: rolling back 'model-3-6a' keeps a SUBSTITUTE row swapping an HF MoE block, or disables the
    # native MoE block (an unrelated piece), or another piece's rollback touches either
    hfm = _hfm()

    class Swapped(nn.Module):
        @classmethod
        def recipe(cls, hf_config):
            return {"moe_experts": 4, "moe_top_k": 2}

    monkeypatch.setattr(hfm, "SUBSTITUTE", {("MixtralSparseMoeBlock", hfm.PIN): Swapped})
    sk = hfm.skeleton(hfm.tiny_config("mixtral"))
    assert type(sk.model.layers[0].mlp) is Swapped and sk.model.layers[1].mlp.moe_layer == 1
    monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset({"model-3-6a"}))
    with pytest.raises(hfm.HFRefused, match="model-3-6a") as e:
        hfm.units(
            hfm.skeleton(hfm.tiny_config("mixtral")), hfm.unit_map(hfm.tiny_config("mixtral"))
        )
    assert e.value.cause == "MOE-NO-ROW"
    native = GPTConfig(
        vocab_size=32,
        n_layer=2,
        n_head=2,
        d_model=16,
        seq_len=4,
        moe_experts=4,
        moe_top_k=2,
        moe_d_ff=8,
    )
    native.validate()
    layers = build_layers(native, 0)
    assert isinstance(layers[1].mlp, arch.MoEMLP)
    with pytest.raises(KeyError, match="no-such-piece"):
        rollback.target("no-such-piece")
    monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset({"runtime-7a"}))
    assert type(hfm.skeleton(hfm.tiny_config("mixtral")).model.layers[0].mlp) is Swapped


def test_hfm_rollback_table_matches_the_handout():
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a piece maps to another target than ADR-0009 round6.md section 4's, or the module imports
    # anything (a rollback must be a one-line edit with no side effect)
    assert rollback.ROLLED_BACK == frozenset()
    assert rollback.PIECES == {
        "scheduler-6": "tied-split-refused",
        "runtime-7a": "tied-split-refused",
        "runtime-7b": "tied-split-refused",
        "runtime-7c": "model-4a",
        "runtime-7d": "model-4a",
        "model-4b": "model-4a",
        "control-6": "model-4a",
        "model-3-6a": "moe-from-scratch",
    }
    src = open(rollback.__file__).read()
    assert "import " not in src


# --------------------------------------------------- convert and call_layer


def test_hfm_convert_exclusions_and_routers_stay_linear():
    # covers: MODEL-I9
    # WOULD-FAIL-IF: convert splits a Linear named in the exclusion set (its other use would lose its gradient)
    # or an MoE router, or an empty exclusion set converts anything other than what it converted before
    torch.manual_seed(0)
    m = nn.ModuleDict({"a": nn.Linear(4, 4), "b": nn.Linear(4, 4), "r": arch.Router(4, 3)})
    assert wgrad.convert(m, exclude=frozenset({m["b"]})) == 1
    assert (
        type(m["a"]) is wgrad.SplitLinear
        and type(m["b"]) is nn.Linear
        and type(m["r"]) is arch.Router
    )
    for cfg in (
        GPTConfig(vocab_size=32, n_layer=2, n_head=2, d_model=16, seq_len=4),
        GPTConfig(
            vocab_size=32,
            n_layer=2,
            n_head=2,
            d_model=16,
            seq_len=4,
            norm="rmsnorm",
            mlp="swiglu",
            tie_embeddings=True,
            init_scheme="gpt2",
        ),
    ):
        layers = build_layers(cfg, 0)
        lin = [type(x).__name__ for x in layers.modules() if isinstance(x, nn.Linear)]
        expect = 4 * 2 + 1 if cfg.mlp == "gelu" else 5 * 2
        assert lin.count("SplitLinear") == expect, lin


def test_hfm_call_layer_passes_ids_only_to_units_that_take_them():
    # covers: MODEL-I9
    # WOULD-FAIL-IF: call_layer passes ids to a dense layer (a dense call site changes) or drops them for an hf
    # unit (a unit after the first would run without the microbatch's input ids)
    class Dense(nn.Module):
        def forward(self, x):
            return x + 1

    class Unit(nn.Module):
        takes_ids = True

        def forward(self, x, ids=None):
            return x + ids

    x, ids = torch.ones(2), torch.full((2,), 5.0)
    assert torch.equal(call_layer(Dense(), x, ids=ids), torch.full((2,), 2.0))
    assert torch.equal(call_layer(Unit(), x, ids=ids), torch.full((2,), 6.0))


# ======================================================= exact MoE (ADR-0008)


def _moe_cfg(**kw) -> GPTConfig:
    base = dict(
        vocab_size=32,
        n_layer=2,
        n_head=2,
        d_model=16,
        seq_len=6,
        norm="rmsnorm",
        mlp="swiglu",
        bias=False,
        pos="rope",
        moe_experts=6,
        moe_top_k=2,
        moe_d_ff=8,
    )
    base.update(kw)
    return GPTConfig(**base)


def _ref_select(key: torch.Tensor, k: int, groups: int = 1, topk_groups: int = 1) -> list:
    """The lowest-index rule written in Python: larger key first, then lower
    index; with groups, the eligible groups by the same rule on the sum of
    each group's top-2 keys."""
    out = []
    for row in key.tolist():
        E = len(row)
        eligible = set(range(E))
        if groups > 1:
            per = E // groups
            gkeys = []
            for g in range(groups):
                top = sorted(row[g * per : (g + 1) * per], reverse=True)[:2]
                gkeys.append((-(top[0] + top[1]), g))
            chosen = [g for _, g in sorted(gkeys)[:topk_groups]]
            eligible = {e for g in chosen for e in range(g * per, (g + 1) * per)}
        ranked = sorted(eligible, key=lambda e: (-row[e], e))
        out.append(sorted(ranked[:k]))
    return out


def test_moe_selection_follows_the_lowest_index_rule_on_crafted_keys():
    # covers: MODEL-I8
    # WOULD-FAIL-IF: selection breaks ties (exact, +-0, subnormal, +-inf, group ties) by anything but the lowest
    # index, returns experts in another order than ascending, or lets a NaN key through
    sub = 1e-40  # an fp32 subnormal
    inf = float("inf")
    rows = [
        [0.5, 0.5, 0.5, 0.1, 0.5, 0.2],
        [-0.0, 0.0, -0.0, -1.0, 0.0, -2.0],
        [sub, 2 * sub, sub, 0.0, -sub, 2 * sub],
        [inf, -inf, inf, 3.0, inf, -inf],
        [-inf, -inf, -inf, -inf, -inf, -inf],
        [0.3, 0.1, 0.9, 0.9, 0.2, 0.9],
    ]
    key = torch.tensor(rows, dtype=torch.float32)
    for k in (1, 2, 3, 6):
        got = arch.moe_select(key, k).tolist()
        assert got == _ref_select(key, k), (k, got)
    assert arch.moe_select(key, 2).tolist()[0] == [0, 1]  # four-way tie: lowest indices
    assert arch.moe_select(key, 2).tolist()[1] == [0, 1]  # -0 == +0: a tie, not an order
    # Groups (E 6, G 3, top-2 groups): group keys tie between groups 0 and 2.
    gk = torch.tensor([[0.4, 0.1, 0.0, 0.3, 0.4, 0.1], [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]])
    for k in (1, 2, 3):
        got = arch.moe_select(gk, k, groups=3, topk_groups=2).tolist()
        assert got == _ref_select(gk, k, 3, 2), (k, got)
    assert arch.moe_select(gk, 3, groups=3, topk_groups=2).tolist()[0] == [0, 1, 4]
    with pytest.raises(ValueError, match="NaN"):
        arch.moe_select(torch.tensor([[0.1, float("nan"), 0.2]]), 1)
    # Control: a highest-index tie rule (a deliberately wrong implementation) is told apart by these keys.
    wrong = [sorted(sorted(range(6), key=lambda e: (-r[e], -e))[:2]) for r in key.tolist()]
    assert wrong != arch.moe_select(key, 2).tolist()


def test_moe_never_calls_topk(monkeypatch):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: the block (forward, backward, capture, flush) reaches torch.topk anywhere
    def boom(*a, **k):
        raise AssertionError("torch.topk called")

    monkeypatch.setattr(torch, "topk", boom)
    monkeypatch.setattr(torch.Tensor, "topk", boom)
    cfg = _moe_cfg(
        moe_groups=3, moe_topk_groups=2, moe_shared_d_ff=8, moe_lbl_coef=0.01, moe_z_coef=0.001
    )
    layers = build_layers(cfg, 0)
    x = torch.randint(0, 32, (2, 6))
    h = x
    for layer in layers:
        h = layer(h)
    with wgrad.capture() as store:
        loss_fn(h, x).backward()
    assert wgrad.flush(store)


@pytest.mark.parametrize(
    "kw, field",
    [
        (dict(moe_experts=8, moe_groups=8), "moe_groups"),
        (dict(moe_experts=8, moe_groups=3), "moe_groups"),
        (dict(moe_experts=8, moe_groups=4, moe_topk_groups=1, moe_top_k=3), "moe_top_k"),
        (dict(moe_experts=8, moe_groups=4, moe_topk_groups=5), "moe_topk_groups"),
        (dict(moe_experts=4, moe_top_k=5), "moe_top_k"),
        (dict(moe_experts=4, moe_top_k=2, moe_apply="input"), "moe_apply"),
        (dict(moe_score="argmax"), "moe_score"),
        (dict(lora_rank=2), "lora_rank"),
    ],
)
def test_moe_config_refusals_name_the_field(kw, field):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: a group configuration outside SPEC-model §2's rule (E/G < 2, G not dividing E, k above
    # topk_groups*E/G), an impossible k, input weighting with k > 1, an unknown score or LoRA with MoE is
    # admitted, or refused without naming the field
    with pytest.raises(ValueError, match=field):
        _moe_cfg(**kw).validate()
    _moe_cfg(moe_experts=8, moe_groups=4, moe_topk_groups=2, moe_top_k=4).validate()


def test_moe_folds_equal_their_written_left_folds():
    # covers: MODEL-I8
    # WOULD-FAIL-IF: dispatch backward, combine or the fan-out backward sum in another order than the written
    # left fold (ascending expert index; output order), which these values make observable
    big, tiny = 1.0e8, 1.0
    # Fan-out: ((g0 + g1) + g2) differs from g0 + (g1 + g2) on these values.
    gs = [torch.tensor([big]), torch.tensor([-big]), torch.tensor([tiny])]
    x = torch.zeros(1, requires_grad=True)
    outs = arch._Fanout.apply(x, 3)
    torch.autograd.backward(list(outs), gs)
    assert torch.equal(x.grad, (gs[0] + gs[1]) + gs[2])
    assert not torch.equal(x.grad, gs[0] + (gs[1] + gs[2]))
    # Dispatch: token 0 routed to experts 0, 1, 2 (k 3); its slot gradients fold in ascending expert order.
    pos = torch.tensor([[0, 1, 2]])
    xr = torch.zeros(1, 1, requires_grad=True)
    d = arch._Dispatch.apply(xr, torch.zeros(3, dtype=torch.long), pos)
    d.backward(torch.tensor([[big], [tiny], [-big]]))
    B, t1 = torch.tensor(big), torch.tensor(tiny)
    assert xr.grad.item() == ((B + t1) - B).item() == 0.0  # fp32; (big - big) + tiny would be 1.0
    # Combine: o = (w0*Y0 + w1*Y1) + w2*Y2 in fp32, ascending expert index.
    Y = torch.tensor([[big], [tiny], [-big]], requires_grad=True)
    w = torch.ones(1, 3, requires_grad=True)
    perm = torch.arange(3)
    o = arch._Combine.apply(Y, w, pos, perm)
    assert o.item() == ((B + t1) - B).item()
    o.backward(torch.tensor([[2.0]]))
    assert torch.equal(w.grad, torch.tensor([[2 * big, 2 * tiny, -2 * big]]))
    assert torch.equal(Y.grad, torch.full((3, 1), 2.0))


# Crafted routing (E 6, k 2): token 0 -> experts {0, 1}; tokens 1-4 -> {2, 3};
# token 5 -> {1, 2}. Counts 1, 2, 5, 4, 0, 0: experts with 0, 1 and many rows.
_CRAFTED = [(0, 1), (2, 3), (2, 3), (2, 3), (2, 3), (1, 2)]


def _crafted_block(compute="fp32", **kw):
    cfg = _moe_cfg(n_layer=1, compute_dtype=compute, boundary_dtype=compute, **kw)
    blk = build_layers(cfg, 0)[1].mlp
    with torch.no_grad():
        blk.router.weight.zero_()
        for e in range(cfg.moe_experts):
            blk.router.weight[e, e] = 10.0
    g = torch.Generator().manual_seed(12)
    xn = torch.randn(1, len(_CRAFTED), 16, generator=g) * 0.1
    xn[..., : cfg.moe_experts] = 0.0
    for t, pair in enumerate(_CRAFTED):
        for e in pair:
            xn[0, t, e] = 1.0
    G = torch.randn(1, len(_CRAFTED), 16, generator=g)
    return cfg, blk, xn, G


def _block_grads(blk, xn, G, compute, capture):
    """The block as a stage runs it: its input is a boundary tensor whose
    gradient B computes (so every expert's backward runs), W deferred under
    capture() when asked."""
    xn = xn.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=compute == "bf16"):
        out = blk(xn)
    loss = (out.float() * G).sum()
    params = list(blk.parameters())
    if not capture:
        return dict(zip(map(id, params), torch.autograd.grad(loss, params, allow_unused=True)))
    owned = wgrad.owned_params(blk)
    others = [p for p in params if id(p) not in owned]
    with wgrad.capture() as store:
        gs = torch.autograd.grad(loss, [xn] + others, allow_unused=True)[1:]
    entries = list(store)
    got = dict(zip(map(id, others), gs))
    got.update(wgrad.flush(store))
    return got, entries


def _run(layers, x, capture):
    """A full-model step: the immediate gradients, or those of B plus the
    deferred store's flush."""
    h = x
    for layer in layers:
        h = layer(h)
    loss = loss_fn(h, x)
    params = list(layers.parameters())
    if not capture:
        return dict(zip(map(id, params), torch.autograd.grad(loss, params)))
    owned = wgrad.owned_params(layers)
    others = [p for p in params if id(p) not in owned]
    with wgrad.capture() as store:
        gs = torch.autograd.grad(loss, others)
    out = dict(zip(map(id, others), gs))
    out.update(wgrad.flush(store))
    return out


@pytest.mark.parametrize("compute", ["fp32", "bf16"])
def test_moe_unrouted_expert_gets_an_all_plus_zero_gradient(compute):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: an expert routed no token gets no gradient (None, absent from flush) or a gradient with a
    # sign bit (-0), on the immediate path or from the deferred store
    cfg, blk, xn, G = _crafted_block(compute)
    imm = _block_grads(blk, xn, G, compute, capture=False)
    (rec,) = blk.moe_records
    assert rec.counts.tolist() == [1, 2, 5, 4, 0, 0]
    blk.moe_records.clear()
    dfr, _ = _block_grads(blk, xn, G, compute, capture=True)
    for e in (4, 5):
        for p in blk.experts[e].parameters():
            for g in (imm[id(p)], dfr.get(id(p))):
                assert g is not None and not g.any() and not torch.signbit(g).any()
    for p in blk.parameters():
        assert imm[id(p)] is not None and id(p) in dfr
        assert torch.equal(dfr[id(p)].to(imm[id(p)].dtype), imm[id(p)])


@pytest.mark.parametrize("compute", ["fp32", "bf16"])
def test_moe_capture_records_each_expert_once_on_its_rows_and_matches_immediate_dw(compute):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: an expert Linear records more or fewer than one (X, dY) per call, records rows other than
    # its routed ones (in ascending token order), or its deferred dW differs from the immediate dW for 0, 1 or
    # many rows
    cfg, blk, xn, G = _crafted_block(compute)
    imm = _block_grads(blk, xn, G, compute, capture=False)
    blk.moe_records.clear()
    _, entries = _block_grads(blk, xn, G, compute, capture=True)
    counts = blk.moe_records[0].counts.tolist()
    assert counts == [1, 2, 5, 4, 0, 0]
    rows = {e: [t for t, pair in enumerate(_CRAFTED) if e in pair] for e in range(6)}
    xf = xn.reshape(-1, 16)
    for e, ex in enumerate(blk.experts):
        for lin in (ex.gate, ex.up, ex.down):
            mine = [fn for m, fn in entries if m is lin]
            assert len(mine) == 1, (e, len(mine))
            X, _ = mine[0].__defaults__
            assert X.shape[0] == counts[e]
            if lin is ex.gate:
                assert torch.equal(X, xf[rows[e]].to(X.dtype))
            dw, _ = mine[0]()
            assert torch.equal(dw.to(torch.float32), imm[id(lin.weight)].to(torch.float32)), (
                e,
                lin,
            )


def test_moe_block_draws_nothing_and_recomputes_identically():
    # covers: MODEL-I8
    # WOULD-FAIL-IF: the block draws from a generator, or recomputing it on the same input changes its output,
    # aux values or counts (B's rebuild would then differ from F)
    cfg = _moe_cfg(
        moe_shared_d_ff=8,
        moe_lbl_coef=0.01,
        moe_z_coef=0.001,
        moe_select="score_bias",
        moe_bias_rate=0.1,
    )
    layers = build_layers(cfg, 0)
    blk = layers[1].mlp
    x = torch.randn(2, 6, 16, generator=torch.Generator().manual_seed(4))
    torch.manual_seed(9)
    s0 = torch.get_rng_state()
    a = blk(x)
    assert torch.equal(s0, torch.get_rng_state())
    b = blk(x)
    r1, r2 = arch.take_moe_records(layers)
    assert torch.equal(a, b) and r1.aux == r2.aux and torch.equal(r1.counts, r2.counts)
    assert r1.counts.dtype == torch.int64 and int(r1.counts.sum()) == 12 * 2


def test_moe_bias_update_moves_by_exactly_gamma_for_any_count_order():
    # covers: MODEL-I8
    # WOULD-FAIL-IF: the selection-bias update depends on the summation order of the counts, moves a bias by
    # anything but +gamma, -gamma or 0 (touching a bias on an exact tie), or accepts malformed counts
    gamma = 0.1
    cfg = _moe_cfg(moe_experts=4, moe_bias_rate=gamma, moe_select="score_bias")
    layers = build_layers(cfg, 0)
    blk = layers[1].mlp
    with torch.no_grad():
        blk.sel_bias.copy_(torch.tensor([-0.0, 0.3, 1e-8, -2.0]))
    before = blk.sel_bias.clone()
    per_mb = [torch.tensor([5, 1, 4, 2]), torch.tensor([1, 5, 2, 4]), torch.tensor([3, 3, 3, 3])]
    T_g = 3 * 12 // cfg.moe_top_k  # 36 slots over 3 microbatches -> T_g*k = 36, mean count 9
    results = []
    for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
        c = torch.zeros(4, dtype=torch.int64)
        for i in order:
            c = c + per_mb[i]
        with torch.no_grad():
            blk.sel_bias.copy_(before)
        arch.moe_bias_update(layers, {1: c, 2: c}, T_g)
        results.append(blk.sel_bias.clone())
    assert all(torch.equal(results[0], r) for r in results)
    total = (
        per_mb[0] + per_mb[1] + per_mb[2]
    )  # [9, 9, 9, 9] would be all ties; this is [9, 9, 9, 9]
    assert total.tolist() == [9, 9, 9, 9]
    assert torch.equal(results[0].view(torch.int32), before.view(torch.int32))  # ties: -0 stays -0
    skew = torch.tensor([12, 6, 9, 9])
    with torch.no_grad():
        blk.sel_bias.copy_(before)
    arch.moe_bias_update(layers, {1: skew}, T_g)
    assert torch.equal(
        blk.sel_bias, torch.stack([before[0] - gamma, before[1] + gamma, before[2], before[3]])
    )
    for bad in (
        torch.tensor([9, 9, 9, 8]),
        torch.tensor([9, 9, 19, -1]),
        torch.tensor([9.0, 9, 9, 9]),
        torch.tensor([18, 18]),
    ):
        with pytest.raises(ValueError, match="layer 1"):
            arch.moe_bias_update(layers, {1: bad}, T_g)


@pytest.mark.parametrize("compute", ["fp32", "bf16"])
@pytest.mark.parametrize("skew", [False, True])
def test_moe_native_pending_w_bytes_equal_tk_2h_3f(compute, skew):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: a routed expert's pending W per row is not 2h + 3f values (X stored twice for gate and up,
    # or rows padded), for a skewed (experts with 0 rows) or a random routing
    cfg, blk, xn, G = _crafted_block(compute)
    if not skew:
        xn = torch.randn(2, 6, 16, generator=torch.Generator().manual_seed(5))
        G = torch.randn(2, 6, 16, generator=torch.Generator().manual_seed(6))
    _, entries = _block_grads(blk, xn, G, compute, capture=True)
    experts = {id(lin) for ex in blk.experts for lin in (ex.gate, ex.up, ex.down)}
    seen, total = set(), 0
    for m, fn in entries:
        if id(m) not in experts:
            continue
        for t in fn.__defaults__:
            key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape))
            if key not in seen:
                seen.add(key)
                total += t.numel() * t.element_size()
    T = xn.shape[0] * xn.shape[1]
    k, hd, f = cfg.moe_top_k, cfg.d_model, cfg.moe_d_ff
    b = 4 if compute == "fp32" else 2
    assert total == T * k * (2 * hd + 3 * f) * b
    assert (0 in blk.moe_records[0].counts.tolist()) == skew


_RECIPES = [
    dict(),
    dict(moe_score="softmax_k", moe_select="logit", moe_router_bias=True),
    dict(
        moe_score="sigmoid",
        moe_select="score_bias",
        moe_norm=True,
        moe_norm_eps=1e-20,
        moe_scale=2.5,
    ),
    dict(
        moe_score="sigmoid",
        moe_groups=3,
        moe_topk_groups=2,
        moe_norm=True,
        moe_lbl_scope="sequence",
    ),
    dict(
        moe_top_k=1,
        moe_apply="input",
        moe_score="sigmoid",
        moe_select="logit",
        moe_shared_d_ff=8,
        moe_shared_gate="none",
    ),
    dict(
        moe_expert="swiglu_clamped",
        moe_limit=0.5,
        moe_alpha=1.702,
        moe_score="softmax_k",
        moe_select="logit",
    ),
    dict(moe_shared_d_ff=8, moe_router_dtype="fp32", moe_norm=True),
]


@pytest.mark.parametrize("compute", ["fp32", "bf16"])
@pytest.mark.parametrize("rk", range(len(_RECIPES)))
def test_moe_every_recipe_row_is_one_function_on_both_paths(compute, rk):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: a recipe row (scores, keys, groups, renormalization, scaling, input weighting, clamped
    # experts, shared expert and gate, router dtype, lbl scope, z) gives deferred gradients that differ from the
    # immediate ones, leaves a trainable tensor without a gradient, or records aux values other than its
    # written lbl and z
    cfg = _moe_cfg(
        compute_dtype=compute,
        boundary_dtype=compute,
        moe_lbl_coef=0.01,
        moe_z_coef=0.001,
        **_RECIPES[rk],
    )
    cfg.validate()
    layers = build_layers(cfg, 0)
    x = torch.randint(0, 32, (2, 6), generator=torch.Generator().manual_seed(6))
    arch.take_moe_records(layers)
    imm = _run(layers, x, capture=False)
    recs = arch.take_moe_records(layers)
    assert [r.layer for r in recs] == [1, 2]
    for p in layers.parameters():
        assert imm[id(p)] is not None
    layers.zero_grad(set_to_none=True)
    dfr = _run(layers, x, capture=True)
    for p in layers.parameters():
        assert torch.equal(dfr[id(p)].to(imm[id(p)].dtype), imm[id(p)])
    again = arch.take_moe_records(layers)
    assert [r.aux for r in again] == [r.aux for r in recs]
    assert all(set(r.aux) == {"lbl", "z"} for r in recs)


def test_moe_aux_values_equal_their_formulas():
    # covers: MODEL-I8
    # WOULD-FAIL-IF: the per-layer lbl is not (E/(T^2 k)) sum_i c_i S_i, the per-sequence form is not its mean
    # over sequences, or z is not (1/T) sum_t logsumexp(logits_t)^2
    for scope in ("layer", "sequence"):
        cfg = _moe_cfg(n_layer=1, moe_lbl_scope=scope)
        blk = build_layers(cfg, 0)[1].mlp
        x = torch.randn(2, 6, 16, generator=torch.Generator().manual_seed(8))
        with torch.no_grad():
            blk(x)
            (rec,) = blk.moe_records
            logits = blk.router(x.reshape(12, 16)).float()
        q = torch.softmax(logits, -1)
        E, k = 6, 2
        sel = arch.moe_select(q, k)
        c = torch.bincount(sel.reshape(-1), minlength=E).float()
        if scope == "layer":
            lbl = (E / (12 * 12 * k)) * (c * q.sum(0)).sum()
        else:
            cs = torch.nn.functional.one_hot(sel.view(2, 6 * k), E).sum(1).float()
            lbl = ((E / (6 * 6 * k)) * (cs * q.view(2, 6, E).sum(1)).sum(-1)).mean()
        lse = torch.logsumexp(logits, -1)
        assert rec.aux == {"lbl": float(lbl).hex(), "z": float((lse * lse).mean()).hex()}
        assert torch.equal(rec.counts, c.long())


def test_moe_records_are_taken_sorted_by_layer_and_call():
    # covers: MODEL-I8
    # WOULD-FAIL-IF: take_moe_records returns records out of (moe_layer, call) order, misses a layer, keeps
    # records after taking them, or keys a native block by anything but its flat-list index
    cfg = _moe_cfg(n_layer=3, moe_layers=[0, 2])
    layers = build_layers(cfg, 0)
    assert [getattr(layer.mlp, "moe_layer", None) for layer in list(layers)[1:4]] == [1, None, 3]
    xn = torch.randn(2, 6, 16, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        for blk in (layers[3].mlp, layers[1].mlp, layers[3].mlp, layers[1].mlp):
            blk(xn if blk is layers[3].mlp else xn * 2)
    recs = arch.take_moe_records(layers)
    assert [r.layer for r in recs] == [1, 1, 3, 3]
    assert recs[0].aux == recs[1].aux and recs[0].aux != recs[2].aux
    assert arch.take_moe_records(layers) == []


# ================================ M4 model-families (ADR-0009 E1, E2, E6, E7; ADR-0008 F8)
#
# The 12 families the human named and ADR-0009 E1's extra arms through the gate; E2's controls, each
# judged only after its trigger fired (a control whose trigger never fires is resized, never passed); E6's
# drift arms on a symlinked copy of the site directory under tmp_path (never .venv); E7's measurements; the
# census CLI; and the HF-MoE class-swap rows proven equal to MoEMLP (F8).

# The human's mandatory list, keyed by the pinned model type (gemma3 is gemma3_text, the causal-LM text model).
FAMILIES = {
    "llama": {},
    "mistral": {},
    "qwen2": {},
    "qwen3": {},
    "gemma": {},
    "gemma2": {},
    "gemma3_text": {},
    "phi3": {},
    "olmo2": {},
    "granite": {
        "embedding_multiplier": 12.0,
        "logits_scaling": 8.0,
        "residual_multiplier": 0.2,
        "attention_multiplier": 0.25,
    },
    "gpt_neox": {"classifier_dropout": 0.0},
    "starcoder2": {},
}
# Sliding-window layers of each family's tiny config (window 8 below the gate's S 16); 0 elsewhere.
FAMILY_WINDOWS = {"mistral": 4, "gemma2": 2, "gemma3_text": 5}
FP32_RECIPE = {**GATE_RECIPE, "compute_dtype": "fp32"}


def _gate_depth(hc: dict) -> int:
    hfm = _hfm()
    return int(hfm._config(hfm._shrunk(hc, keep_depth=False)).num_hidden_layers)


def _fired(count, what: str) -> int:
    """E2/E6: a control is judged only after its trigger fired."""
    assert count > 0, f"{what}: the control's trigger never fired; resize it, never pass it"
    return int(count)


def _refused(rep: dict, cause: str, what: str) -> None:
    assert rep["cause"] == cause and not rep["ok"], (what, rep["cause"], rep["detail"])


def _bits(t: torch.Tensor) -> tuple:
    """A tensor's dtype, shape and exact bytes (the sign of zero and NaN payloads included)."""
    t = t.detach().contiguous()
    return t.dtype, tuple(t.shape), t.reshape(-1).view(torch.uint8)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    (da, sa, ba), (db, sb, bb) = _bits(a), _bits(b)
    return da == db and sa == sb and torch.equal(ba, bb)


@pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("fam", list(FAMILIES))
def test_hf_family_gate_equals_vanilla_bitwise(fam, tie):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: a mandatory family's unit chain differs from vanilla transformers 5.19.0 at any cut (logits,
    # loss hex or any parameter gradient; fp32 or bf16; M 1 or 2; deferred W; recompute), its causality probe
    # fails, the gate compares fewer cuts than every one, a tie is missed or invented, or a sliding window that
    # fires is not counted
    hfm = _hfm()
    hc = hfm.tiny_config(fam, **FAMILIES[fam], tie_word_embeddings=tie)
    rep = hfm.gate(hc, GATE_RECIPE, "cpu")
    _assert_compared(rep, _gate_depth(hc))
    _assert_platform_cause(rep)
    assert rep["triggers"]["tie"] == int(tie), rep["triggers"]
    assert rep["triggers"]["window"] == FAMILY_WINDOWS.get(fam, 0), rep["triggers"]
    assert rep["triggers"]["moe"] == 0


_ROPE = {"rope_theta": 10000.0}
E1_ARMS = {
    "llama_rope_llama3": (
        "llama",
        {
            "rope_parameters": {
                **_ROPE,
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8,
            }
        },
    ),
    "llama_rope_yarn": (
        "llama",
        {
            "rope_parameters": {
                **_ROPE,
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 8,
            }
        },
    ),
    "llama_rope_longrope_short": (
        "llama",
        {
            "rope_parameters": {
                **_ROPE,
                "rope_type": "longrope",
                "short_factor": [1.0 + i / 10 for i in range(8)],
                "long_factor": [2.0 + i / 10 for i in range(8)],
                "original_max_position_embeddings": 64,
            }
        },
    ),
    "cohere": ("cohere", {}),
    "gpt2": (
        "gpt2",
        {"resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0, "summary_first_dropout": 0.0},
    ),
    "minicpm3": ("minicpm3", {}),
    "falcon_h1": ("falcon_h1", {}),
    "mamba": ("mamba", {}),
}


# A non-mandatory arm the gate refuses, with its documented cause. mamba under bf16 autocast: its slow path's
# `dt_proj.weight @ time_step` is a torch.matmul whose folding depends on whether its operand requires grad,
# and the autocast cast of the weight requires grad only with grad enabled, so a recompute stage's no-grad
# forward computes other bits than the reference's (MEASURED: the gate's first difference is the logits of the
# deepest cut with recompute, bf16). A real divergence the gate must and does refuse.
E1_REFUSED = {"mamba": ("GATE-DIFF", "recompute")}


@pytest.mark.parametrize("arm", list(E1_ARMS))
def test_hf_family_e1_extra_arms_pass_the_gate(arm):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: an E1 arm (llama3, yarn or short longrope rope; cohere, gpt2, minicpm3, falcon_h1, mamba)
    # differs from vanilla at any cut, or mamba's dt_proj (its weight read outside its forward) is converted to
    # SplitLinear instead of kept out by the trace's exclusion set
    hfm = _hfm()
    mt, over = E1_ARMS[arm]
    hc = hfm.tiny_config(mt, **over)
    rep = hfm.gate(hc, GATE_RECIPE, "cpu")
    if arm in E1_REFUSED:
        # Recorded with its cause: refused, never exempted (the fp32 recipe is admitted).
        cause, where = E1_REFUSED[arm]
        _refused(rep, cause, arm)
        assert where in rep["first_difference"], rep["first_difference"]
        rep = hfm.gate(hc, FP32_RECIPE, "cpu")
        assert rep["dtypes"] == ["fp32"] and len(rep["cuts"]) == (_gate_depth(hc) + 4) * 2
        assert rep["first_difference"] is None
    else:
        _assert_compared(rep, _gate_depth(hc))
    _assert_platform_cause(rep)
    if mt == "mamba":
        um = hfm.unit_map(hc)
        dt = [k for k in um.exclusions if k.endswith(".mixer.dt_proj")]
        assert len(dt) == um.n, um.exclusions
        units = hfm.units(hfm._full_build(hc, 0), um)
        wgrad.convert(units, exclude=units.exclusions)
        mixer = units.root.backbone.layers[0].mixer
        assert type(mixer.dt_proj) is nn.Linear and type(mixer.in_proj) is wgrad.SplitLinear


# --------------------------------------------------------- HF-MoE rows (F8)

MOE_HF = ("mixtral", "qwen2_moe", "qwen3_moe")


def _hf_moe(mt: str, lbl: float = 0.01, z: float = 0.001, **over):
    """A tiny swapped HF-MoE model built by transformers 5.19.0, its layer-0 block's aux coefficients set."""
    hfm = _hfm()
    hc = hfm.tiny_config(mt, **over)
    m = hfm._full_build(hc, 0, swap=True)
    blk = m.model.layers[0].mlp
    blk.moe_cfg.update(moe_lbl_coef=lbl, moe_z_coef=z)
    return hc, m, blk


def _native_twin(blk, h: int):
    """MoEMLP on the swapped block's byte-mapped parameters (gate rows first, then up rows; down)."""
    rec = dict(blk.moe_cfg)
    cfg = GPTConfig(vocab_size=8, n_layer=1, n_head=1, d_model=h, seq_len=4, **rec)
    nat = arch.MoEMLP(cfg, moe_layer=blk.moe_layer)
    f = rec["moe_d_ff"]
    with torch.no_grad():
        nat.router.weight.copy_(blk.gate.weight)
        for j, ex in enumerate(nat.experts):
            ex.gate.weight.copy_(blk.experts.gate_up_proj[j, :f])
            ex.up.weight.copy_(blk.experts.gate_up_proj[j, f:])
            ex.down.weight.copy_(blk.experts.down_proj[j])
        if nat.shared_expert is not None:
            se = blk.shared_expert
            nat.shared_expert.gate.weight.copy_(se.gate_proj.weight)
            nat.shared_expert.up.weight.copy_(se.up_proj.weight)
            nat.shared_expert.down.weight.copy_(se.down_proj.weight)
            nat.shared_expert_gate.weight.copy_(blk.shared_expert_gate.weight)
    wgrad.convert(nat)
    return nat


def _crafted_moe_input(blk, h: int, rows_seed: int = 21):
    """Routing by construction: token 0 to experts 0..k-1, tokens 1-5 to experts 1..k, so expert 0 has 1 row,
    experts 1..k-1 six, expert k five and experts above k none (0, 1 and many rows)."""
    E, k = blk.moe_cfg["moe_experts"], blk.moe_cfg["moe_top_k"]
    assert E > k + 1
    with torch.no_grad():
        blk.gate.weight.zero_()
        for e in range(E):
            blk.gate.weight[e, e] = 10.0
    g = torch.Generator().manual_seed(rows_seed)
    x = torch.randn(1, 6, h, generator=g) * 0.1
    x[..., :E] = 0.0
    for t in range(6):
        for e in range(k) if t == 0 else range(1, k + 1):
            x[0, t, e] = 1.0
    return x


def _moe_grads(mod, x, G, compute: str, capture: bool):
    """Output and every gradient of the block as a stage runs it (its input a boundary tensor); W deferred under
    capture() when asked. -> (out, {name: grad}) with "x" for the input's gradient."""
    xx = x.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=compute == "bf16"):
        out = mod(xx)
    loss = (out.float() * G).sum()
    named = list(mod.named_parameters())
    if not capture:
        gs = torch.autograd.grad(loss, [xx] + [p for _, p in named])
        return out, dict(zip(["x"] + [k for k, _ in named], gs))
    owned = wgrad.owned_params(mod)
    others = [(k, p) for k, p in named if id(p) not in owned]
    with wgrad.capture() as store:
        gs = torch.autograd.grad(loss, [xx] + [p for _, p in others])
    got = dict(zip(["x"] + [k for k, _ in others], gs))
    deferred = wgrad.flush(store)
    for k, p in named:
        if id(p) in owned:
            got[k] = deferred[id(p)]
    return out, got


def _twin_grads_as_hf(nat, g: dict, f: int) -> dict:
    """MoEMLP's gradients in the HF block's names and layouts (banks stacked from the experts)."""
    exs = list(nat.experts)
    out = {
        "x": g["x"],
        "gate.weight": g["router.weight"],
        "experts.gate_up_proj": torch.stack(
            [
                torch.cat([g[f"experts.{j}.gate.weight"], g[f"experts.{j}.up.weight"]])
                for j in range(len(exs))
            ]
        ),
        "experts.down_proj": torch.stack([g[f"experts.{j}.down.weight"] for j in range(len(exs))]),
    }
    if nat.shared_expert is not None:
        out["shared_expert.gate_proj.weight"] = g["shared_expert.gate.weight"]
        out["shared_expert.up_proj.weight"] = g["shared_expert.up.weight"]
        out["shared_expert.down_proj.weight"] = g["shared_expert.down.weight"]
        out["shared_expert_gate.weight"] = g["shared_expert_gate.weight"]
    return out


@pytest.mark.parametrize("routing", ["crafted", "random"])
@pytest.mark.parametrize("compute", ["fp32", "bf16"])
@pytest.mark.parametrize("mt", MOE_HF)
def test_moe_hf_block_equals_moemlp_bitwise(mt, compute, routing):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: a swapped transformers 5.19.0 block (Mixtral, Qwen2-MoE, Qwen3-MoE) differs from MoEMLP on
    # byte-mapped parameters in its output, aux values, counts or any gradient (sign of zero included), fp32 or
    # bf16, for 0, 1 or many rows per expert, immediate or with W deferred, or returns anything but one tensor
    hc, m, blk = _hf_moe(mt)
    h = int(m.config.hidden_size)
    if routing == "crafted":
        x = _crafted_moe_input(blk, h)
    else:
        x = torch.randn(2, 6, h, generator=torch.Generator().manual_seed(22))
    nat = _native_twin(blk, h)
    wgrad.convert(blk)  # the shared expert's Linears, as every stage converts them
    G = torch.randn(x.shape, generator=torch.Generator().manual_seed(23))
    out_h, g_h = _moe_grads(blk, x, G, compute, capture=False)
    rec_h = blk.moe_records[-1]
    out_n, g_n = _moe_grads(nat, x, G, compute, capture=False)
    rec_n = nat.moe_records[-1]
    assert isinstance(out_h, torch.Tensor) and _same_bits(out_h, out_n)
    assert rec_h.aux == rec_n.aux and torch.equal(rec_h.counts, rec_n.counts)
    assert rec_h.layer == 0
    if routing == "crafted":
        c = rec_h.counts.tolist()
        assert c[0] == 1 and max(c) == 6 and 0 in c, c
    want = _twin_grads_as_hf(nat, g_n, blk.moe_cfg["moe_d_ff"])
    assert set(want) == set(g_h)
    for k, g in want.items():
        assert _same_bits(g_h[k], g), (k, compute)
    _, g_d = _moe_grads(blk, x, G, compute, capture=True)
    for k, g in g_h.items():
        assert _same_bits(g_d[k].to(g.dtype), g), ("deferred", k)
    # The swapped decoder layers run inside HF's own model on the one-tensor contract.
    with torch.no_grad():
        logits = m(input_ids=_ids(int(m.config.vocab_size)), use_cache=False).logits
    assert logits.isfinite().all()


def test_moe_hf_two_index_slices_lose_a_constructed_minus_zero(monkeypatch):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: gate and up rows taken by two index slices of one expert view (whose backwards autograd sums
    # into one zero-filled tensor, +0 + -0 = +0) would keep every bit of the seam's one chunk(2) per view
    hfm = _hfm()
    hc, m, blk = _hf_moe("mixtral", lbl=0.0, z=0.0)
    h = int(m.config.hidden_size)
    x = _crafted_moe_input(blk, h)  # expert 0 has one row, with zeros in several columns
    G = torch.randn(x.shape, generator=torch.Generator().manual_seed(24))

    # The construction: a linear whose dW for one row is the exact outer product dY x X (a GEMM starts its
    # accumulator at +0 and never returns -0, MEASURED on this CPU), so a -0 reaches the bank's backward.
    def outer(rows, w):
        return (rows[:, None, :] * w[None]).sum(-1)

    def swiglu(rows, wg, wu, wd):
        return outer(torch.nn.functional.silu(outer(rows, wg)) * outer(rows, wu), wd)

    monkeypatch.setattr(hfm, "_swiglu", swiglu)
    _, g_ok = _moe_grads(blk, x, G, "fp32", capture=False)
    bank = g_ok["experts.gate_up_proj"]
    _fired(int(((bank == 0) & torch.signbit(bank)).sum()), "a -0 in the gate/up dW")

    def two_slices(self, xd, counts):
        f = self.gate_up_proj.shape[1] // 2
        outs = []
        for j, rows in enumerate(torch.split(xd, counts)):
            outs.append(
                hfm._swiglu(
                    rows, self.gate_up_proj[j, :f], self.gate_up_proj[j, f:], self.down_proj[j]
                )
            )
        return torch.cat(outs)

    monkeypatch.setattr(hfm._SwapExperts, "forward", two_slices)
    _, g_bad = _moe_grads(blk, x, G, "fp32", capture=False)
    assert torch.equal(g_bad["experts.gate_up_proj"], bank)  # equal as numbers ...
    assert not _same_bits(g_bad["experts.gate_up_proj"], bank)  # ... but a -0 lost its sign


@pytest.mark.parametrize("m_bw", [1, 2])
@pytest.mark.parametrize("mt", MOE_HF)
def test_moe_hf_routed_pending_w_is_zero_and_banks_hold_b_m_e_3hf(mt, m_bw):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: a routed expert of a swapped block records a deferred (X, dY) (its dW must run in B), or the
    # bank gradients held between B and W differ from b_m*E*3hf bytes per MoE layer and microbatch (F3, HF part)
    hc, m, blk = _hf_moe(mt, lbl=0.0, z=0.0)
    wgrad.convert(blk)
    h, E, f = int(m.config.hidden_size), blk.moe_cfg["moe_experts"], blk.moe_cfg["moe_d_ff"]
    banks = (blk.experts.gate_up_proj, blk.experts.down_proj)
    routed = set(map(id, blk.experts.modules()))
    held, pending = [], 0
    for mb in range(m_bw):
        x = torch.randn(2, 6, h, generator=torch.Generator().manual_seed(30 + mb))
        out = blk(x.requires_grad_(True))
        owned = wgrad.owned_params(blk)
        others = [p for p in blk.parameters() if id(p) not in owned]
        with wgrad.capture() as store:
            gs = torch.autograd.grad(out.float().pow(2).sum(), [x] + others)
        held += [g for p, g in zip(others, gs[1:]) if any(p is b for b in banks)]
        pending += sum(1 for mod, _ in store if id(mod) in routed)
        assert any(True for _ in store) == (blk.moe_cfg.get("moe_shared_d_ff", 0) > 0)
    assert pending == 0
    assert len(held) == 2 * m_bw and all(g.dtype == torch.float32 for g in held)
    assert sum(g.numel() * g.element_size() for g in held) == m_bw * 4 * E * 3 * h * f


def test_moe_hf_original_experts_forward_never_runs(monkeypatch):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: only the block is swapped and transformers' own experts forward (HF's 2f-wide GEMM with
    # index_add_) or router forward still runs inside a swapped model
    hfm = _hfm()
    from transformers.models.mixtral import modeling_mixtral as mx
    from transformers.models.qwen3_moe import modeling_qwen3_moe as q3

    built = {
        mt: hfm._full_build(hfm.tiny_config(mt), 0, swap=True) for mt in ("mixtral", "qwen3_moe")
    }
    plain = hfm._full_build(hfm.tiny_config("mixtral"), 0, swap=False)

    def boom(*a, **k):
        raise AssertionError("transformers' own MoE forward ran")

    for cls in (mx.MixtralExperts, mx.MixtralTopKRouter, q3.Qwen3MoeExperts, q3.Qwen3MoeTopKRouter):
        monkeypatch.setattr(cls, "forward", boom)
    ids = _ids(128)
    with pytest.raises(AssertionError, match="own MoE forward"):
        plain(input_ids=ids, use_cache=False)  # the patch is live: an unswapped model reaches it
    for mt, m in built.items():
        assert type(m.model.layers[0].mlp.experts).__name__.endswith("ExpertsSwap"), mt
        logits = m(input_ids=ids, use_cache=False).logits
        loss_fn(logits.float(), ids).backward()
        assert m.model.layers[0].mlp.experts.gate_up_proj.grad is not None


def _moe_recipe_cfg(**over) -> GPTConfig:
    return GPTConfig(**over)


def test_moe_hf_recipe_rows_equal_hand_written_dicts():
    # covers: MODEL-I8
    # WOULD-FAIL-IF: a row's recipe maps a config field wrongly (E from num_local_experts or num_experts, k,
    # f, norm_topk_prob, Qwen2-MoE's shared expert and sigmoid gate, the router dtype), maps router_aux_loss_coef
    # (BA-462) instead of taking lbl and z from the run's GPTConfig, or admits output_router_logits or
    # router_jitter_noise
    hfm = _hfm()
    common = {
        "moe_score": "softmax",
        "moe_select": "score",
        "moe_router_dtype": "compute",
        "moe_norm_eps": 0.0,
        "moe_scale": 1.0,
        "moe_bias_rate": 0.0,
    }
    cases = {
        "mixtral": (
            {"num_local_experts": 6, "num_experts_per_tok": 3, "intermediate_size": 40},
            hfm.MixtralSparseMoeBlockSwap,
            {"moe_experts": 6, "moe_top_k": 3, "moe_d_ff": 40, "moe_norm": True},
        ),
        "qwen2_moe": (
            {
                "num_experts": 10,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 24,
                "shared_expert_intermediate_size": 56,
                "norm_topk_prob": True,
            },
            hfm.Qwen2MoeSparseMoeBlockSwap,
            {
                "moe_experts": 10,
                "moe_top_k": 4,
                "moe_d_ff": 24,
                "moe_norm": True,
                "moe_shared_d_ff": 56,
                "moe_shared_gate": "sigmoid",
            },
        ),
        "qwen3_moe": (
            {"num_experts": 12, "num_experts_per_tok": 5, "moe_intermediate_size": 20},
            hfm.Qwen3MoeSparseMoeBlockSwap,
            {"moe_experts": 12, "moe_top_k": 5, "moe_d_ff": 20, "moe_norm": False},
        ),
    }
    for mt, (over, sub, want) in cases.items():
        hc = hfm.tiny_config(mt, router_aux_loss_coef=0.5, **over)
        assert sub.recipe(hc) == {**common, **want}, mt
        assert hfm.SUBSTITUTE[(sub.__name__.removesuffix("Swap"), hfm.PIN)] is sub
        m = hfm._full_build(hc, 0, swap=False)
        run = GPTConfig(moe_lbl_coef=0.02, moe_z_coef=0.003, moe_lbl_scope="sequence")
        hfm._swap(m, hc, run)
        mc = m.model.layers[0].mlp.moe_cfg
        assert (mc["moe_lbl_coef"], mc["moe_z_coef"], mc["moe_lbl_scope"]) == (
            0.02,
            0.003,
            "sequence",
        )
        assert 0.5 not in mc.values()  # router_aux_loss_coef is never mapped
        arch.check_moe(mc)
        with pytest.raises(hfm.HFRefused, match="output_router_logits") as e:
            sub.recipe({**hc, "output_router_logits": True})
        assert e.value.cause == "MOE-NO-ROW"
    with pytest.raises(hfm.HFRefused, match="router_jitter_noise"):
        hfm.MixtralSparseMoeBlockSwap.recipe(hfm.tiny_config("mixtral", router_jitter_noise=0.1))
    with pytest.raises(hfm.HFRefused, match="hidden_act"):
        hfm.Qwen3MoeSparseMoeBlockSwap.recipe(hfm.tiny_config("qwen3_moe", hidden_act="gelu"))
    # A router or experts row is admitted only under a block with a row.
    rows = {k: v for k, v in hfm.SUBSTITUTE.items() if k[0] != "Qwen3MoeSparseMoeBlock"}
    m = hfm._full_build(hfm.tiny_config("qwen3_moe"), 0, swap=False)
    saved = hfm.SUBSTITUTE
    try:
        hfm.SUBSTITUTE = rows
        with pytest.raises(hfm.HFRefused, match="Qwen3MoeSparseMoeBlock has none") as e:
            hfm._swap(m, hfm.tiny_config("qwen3_moe"))
        assert e.value.cause == "MOE-NO-ROW"
    finally:
        hfm.SUBSTITUTE = saved


@pytest.mark.parametrize("mt", MOE_HF)
def test_moe_hf_gate_admits_the_swapped_model(mt):
    # covers: MODEL-I8, MODEL-I10
    # WOULD-FAIL-IF: the gate refuses a swapped tiny Mixtral, Qwen2-MoE or Qwen3-MoE (its rows, the D8
    # classification, the unit trace of the swapped model, or GATE-CAUSAL from a probe that reads the
    # experts' GEMM blocking instead of the model), or compares without its substituted blocks counted,
    # or leaves the row-by-row probe on after it returns
    hfm = _hfm()
    hc = hfm.tiny_config(mt)
    rep = hfm.gate(hc, GATE_RECIPE, "cpu")
    assert rep["cause"] != "GATE-CAUSAL", rep["detail"]
    _assert_compared(rep, _gate_depth(hc))
    _assert_platform_cause(rep)
    assert rep["triggers"]["moe"] == _gate_depth(hc)
    assert hfm._ROWWISE == [False]


def _spy_rowwise(monkeypatch, hfm, fail=False):
    """Records (row-by-row probe on, rows) of every swapped expert GEMM; with
    fail, raises inside the first one the probe runs."""
    seen = []
    real = hfm._swiglu

    def spy(rows, wg, wu, wd):
        seen.append((hfm._ROWWISE[0], int(rows.shape[0])))
        if fail and hfm._ROWWISE[0]:
            raise RuntimeError("injected failure inside the row-by-row probe")
        return real(rows, wg, wu, wd)

    monkeypatch.setattr(hfm, "_swiglu", spy)
    return seen


def test_moe_hf_gate_refuses_a_non_causal_swapped_qwen3_moe_with_the_rowwise_probe(monkeypatch):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: the row-by-row probe hides a real cross-position dependence: a tiny swapped
    # Qwen3-MoE whose attention has no causal mask (every position reads the last id) is admitted,
    # or refused for anything but GATE-CAUSAL, or the probe did not run its experts one row per GEMM
    hfm = _hfm()
    from transformers.models.qwen3_moe import modeling_qwen3_moe as q3

    monkeypatch.setattr(q3, "create_causal_mask", lambda **kw: None)
    monkeypatch.setattr(q3, "create_sliding_window_causal_mask", lambda **kw: None)
    seen = _spy_rowwise(monkeypatch, hfm)
    rep = hfm.gate(hfm.tiny_config("qwen3_moe"), FP32_RECIPE, "cpu")
    _refused(rep, "GATE-CAUSAL", "attention without its causal mask")
    rowwise = [n for on, n in seen if on]
    _fired(len(rowwise), "expert GEMMs the probe ran row by row")
    # one routed row per GEMM (an expert no row reached runs on its empty split)
    assert 1 in rowwise and set(rowwise) <= {0, 1}, sorted(set(rowwise))
    assert rep["cuts"] == []  # refused at the probe, before any comparison
    assert hfm._ROWWISE == [False]


def test_moe_hf_gate_rowwise_probe_is_off_after_gate_returns_and_after_it_raises(monkeypatch):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: hfm._ROWWISE[0] stays true after hfm.gate returns or after it raises from inside
    # the probe, so training or the bitwise comparison would run one GEMM per routed row
    hfm = _hfm()
    hc = hfm.tiny_config("qwen3_moe")
    seen = _spy_rowwise(monkeypatch, hfm, fail=True)
    with pytest.raises(RuntimeError, match="injected failure inside the row-by-row probe"):
        hfm.gate(hc, FP32_RECIPE, "cpu")
    _fired(len([1 for on, _ in seen if on]), "expert GEMMs inside the probe")
    assert hfm._ROWWISE == [False]
    monkeypatch.undo()  # the real _swiglu back, then a spy that does not fail
    seen = _spy_rowwise(monkeypatch, hfm)
    rep = hfm.gate(hc, FP32_RECIPE, "cpu")
    assert rep["cause"] in (None, "FP-PYC", "FP-TORCH-RECORDS"), (rep["cause"], rep["detail"])
    assert hfm._ROWWISE == [False]
    # Outside the probe (the comparison's training passes) no expert GEMM ran row by row.
    after = [n for on, n in seen if not on]
    _fired(len(after), "expert GEMMs outside the probe")
    assert any(n > 1 for n in after)


def test_moe_hf_gate_refuses_a_chain_without_the_aux_injection(monkeypatch):
    # covers: MODEL-I8, MODEL-I10
    # WOULD-FAIL-IF: the gate compares a swapped HF-MoE model with its aux coefficients at 0 (ADR-0009 D8 (4):
    # the reference and the chain carry the same D8 injection), so a chain that drops the lbl/z injection passes
    hfm = _hfm()
    dropped: list = []

    def no_injection(root):
        for m in root.modules():
            mc = getattr(m, "moe_cfg", None)
            if isinstance(mc, dict):
                dropped.append(int(mc["moe_lbl_coef"] != 0 and mc["moe_z_coef"] != 0))
                mc.update(moe_lbl_coef=0.0, moe_z_coef=0.0)

    monkeypatch.setattr(hfm, "_chain_buffers", no_injection)
    hc = hfm.tiny_config("mixtral")
    rep = hfm.gate(hc, FP32_RECIPE, "cpu")
    _fired(sum(dropped), "swapped blocks whose injection the chain dropped")
    _refused(rep, "GATE-DIFF", "aux injection dropped")
    assert "gradient" in rep["first_difference"], rep["first_difference"]


def test_moe_hf_records_under_two_unit_spans_equal_the_swapped_vanilla():
    # covers: MODEL-I8, MODEL-I9
    # WOULD-FAIL-IF: a tiny HF Mixtral split into two unit spans (F, B and W per span through call_layer under
    # capture) records aux values or counts other than the swapped vanilla model's, keys a record by anything but
    # its HF layer index, returns them from the wrong span, or gives any gradient or logit another bit (BA-462)
    hfm = _hfm()
    hc = hfm.tiny_config("mixtral")
    cfg = _hf_cfg(hc, moe_lbl_coef=0.01, moe_z_coef=0.001)
    units = build_layers(cfg, 4)
    n = cfg.n_layer
    vanilla = copy.deepcopy(units.root)
    ids = _ids(cfg.vocab_size, seed=31)
    tgt = _ids(cfg.vocab_size, seed=32)
    arch.take_moe_records(units)
    # The oracle: the swapped vanilla model, every gradient immediate.
    logits = vanilla(input_ids=ids, use_cache=False).logits
    vparams = list(vanilla.named_parameters())
    vg = torch.autograd.grad(loss_fn(logits.float(), tgt), [p for _, p in vparams])
    want = {k: g for (k, _), g in zip(vparams, vg)}
    vrec = arch.take_moe_records([vanilla])
    assert [r.layer for r in vrec] == list(range(n))
    # Two spans: units [0, 3) and [3, n + 2).
    spans = [(0, 3), (3, n + 2)]
    fwd = []
    h = None
    for a, b in spans:
        x = None if h is None else h.detach().requires_grad_(True)
        y = x
        for u in range(a, b):
            y = call_layer(units[u], ids if u == 0 else y, ids=ids)
        fwd.append((x, y))
        h = y
    assert _same_bits(h, logits)
    recs = [arch.take_moe_records(units[a:b]) for a, b in spans]
    assert [[r.layer for r in rs] for rs in recs] == [[0, 1], list(range(2, n))]
    got: dict = {}
    names = {id(p): k for k, p in units.root.named_parameters()}
    gnext = None
    for s in (1, 0):
        x, y = fwd[s]
        mods = units[spans[s][0] : spans[s][1]]
        owned = wgrad.owned_params(mods)
        others = [p for p in mods.parameters() if id(p) not in owned]
        ins = ([x] if x is not None else []) + others
        with wgrad.capture() as store:
            if s == 1:
                gs = list(torch.autograd.grad(loss_fn(y.float(), tgt), ins))
            else:
                gs = list(torch.autograd.grad(y, ins, grad_outputs=gnext))
        if x is not None:
            gnext = gs.pop(0)
        got.update({names[id(p)]: g for p, g in zip(others, gs)})
        flushed = wgrad.flush(store)  # W of this span
        got.update({names[i]: g for i, g in flushed.items()})
    brec = arch.take_moe_records(units)  # B's recompute-free backward records nothing new
    assert brec == []
    flat = [r for rs in recs for r in rs]
    assert [(r.layer, r.aux) for r in flat] == [(r.layer, r.aux) for r in vrec]
    assert all(torch.equal(a.counts, b.counts) for a, b in zip(flat, vrec))
    assert set(got) == set(want)
    for k, g in want.items():
        assert _same_bits(got[k].to(g.dtype), g), k


@pytest.mark.parametrize("mt", MOE_HF)
def test_moe_hf_rollback_refuses_the_rows_and_keeps_native_moe(monkeypatch, mt):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: the rows stay live after 'model-3-6a' is rolled back (a swapped HF-MoE model still admits),
    # or the rollback disables native MoE (MoEMLP of the same dims no longer trains exactly against its oracle),
    # or with nothing rolled back the swapped model is refused
    hfm = _hfm()
    hc = hfm.tiny_config(mt)
    units = hfm.units(hfm.skeleton(hc), hfm.unit_map(hc))  # admitted while the rows are live
    assert len(units) == _gate_depth(hc) + 2
    recipe = hfm.SUBSTITUTE[
        (f"{type(hfm.skeleton(hc).model.layers[0].mlp).__name__}"[:-4], hfm.PIN)
    ]
    rec = recipe.recipe(hc)
    monkeypatch.setattr(rollback, "ROLLED_BACK", frozenset({"model-3-6a"}))
    with pytest.raises(hfm.HFRefused, match="model-3-6a") as e:
        hfm.units(hfm.skeleton(hc), hfm.unit_map(hc))
    assert e.value.cause == "MOE-NO-ROW"
    rep = hfm.gate(hc, FP32_RECIPE, "cpu")
    assert rep["cause"] == "MOE-NO-ROW" and "model-3-6a" in rep["detail"], rep["detail"]
    native = _moe_cfg(
        d_model=64,
        n_head=4,
        **{k: v for k, v in rec.items() if k != "moe_d_ff"},
        moe_d_ff=16,
        moe_lbl_coef=0.01,
    )
    native.validate()
    layers = build_layers(native, 0)
    x = torch.randint(0, 32, (2, 6), generator=torch.Generator().manual_seed(33))
    imm = _run(layers, x, capture=False)
    dfr = _run(layers, x, capture=True)
    for p in layers.parameters():
        assert _same_bits(dfr[id(p)].to(imm[id(p)].dtype), imm[id(p)])
    assert isinstance(layers[1].mlp, arch.MoEMLP)


# ------------------------------------------------------------- DeepSeek-V3

_DSV3 = dict(
    moe_experts=16,
    moe_top_k=4,
    moe_d_ff=8,
    moe_score="sigmoid",
    moe_select="score_bias",
    moe_groups=4,
    moe_topk_groups=2,
    moe_norm=True,
    moe_norm_eps=1e-20,
    moe_scale=2.5,
    moe_router_dtype="fp32",
    moe_shared_d_ff=8,
    moe_shared_gate="none",
    moe_bias_rate=0.001,
)


@pytest.mark.parametrize("compute", ["fp32", "bf16"])
def test_moe_deepseek_native_row_is_exact_against_its_oracle(compute):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: DeepSeek-V3's native recipe row (sigmoid score, score_bias selection, groups 4/2,
    # renormalization and scaling, fp32 router, the bias update) gives a 2-stage pipeline (boundary detached, W
    # deferred) another gradient, aux value or count than the single-process oracle, or its bias update differs
    cfg = _moe_cfg(compute_dtype=compute, boundary_dtype="fp32", n_layer=3, **_DSV3)
    cfg.validate()
    base = build_layers(cfg, 0)
    with torch.no_grad():
        for m in base.modules():
            if isinstance(m, arch.MoEMLP):
                m.sel_bias.copy_(torch.linspace(-0.05, 0.05, 16))
    x = torch.randint(0, 32, (2, 6), generator=torch.Generator().manual_seed(34))
    finals = []
    for path in ("oracle", "pipeline"):
        layers = copy.deepcopy(base)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=compute == "bf16"):
            if path == "oracle":
                g = _run(layers, x, capture=False)
            else:
                g = _two_stage(layers, x, cut=2)
        recs = arch.take_moe_records(layers)
        counts = {r.layer: r.counts for r in recs}
        T = x.numel()
        arch.moe_bias_update(layers, counts, T)
        finals.append(
            (
                [g[id(p)].to(torch.float32) for p in layers.parameters()],
                [(r.layer, r.aux) for r in recs],
                [m.sel_bias.clone() for m in layers.modules() if isinstance(m, arch.MoEMLP)],
            )
        )
    (ga, ra, ba), (gb, rb, bb) = finals
    assert ra == rb and len(ra) == 3
    assert all(_same_bits(a, b) for a, b in zip(ga, gb))
    assert all(_same_bits(a, b) for a, b in zip(ba, bb))
    assert any(not torch.equal(a, b) for a, b in zip(ba, [torch.linspace(-0.05, 0.05, 16)] * 3))


def _two_stage(layers, x, cut: int) -> dict:
    """layers[:cut] and layers[cut:] as two stages: the boundary detached, B under capture, W flushed."""
    first, second = nn.ModuleList(list(layers)[:cut]), nn.ModuleList(list(layers)[cut:])
    h0 = x
    for layer in first:
        h0 = layer(h0)
    b = h0.detach().requires_grad_(True)
    h = b
    for layer in second:
        h = layer(h)
    out: dict = {}
    gb = None
    for mods in (second, first):
        owned = wgrad.owned_params(mods)
        others = [p for p in mods.parameters() if id(p) not in owned]
        with wgrad.capture() as store:
            if mods is second:
                gb, *gs = torch.autograd.grad(loss_fn(h, x), [b] + others)
            else:
                gs = list(torch.autograd.grad(h0, others, grad_outputs=gb))
        out.update(zip(map(id, others), gs))
        out.update(wgrad.flush(store))
    return out


def test_moe_deepseek_router_selects_transformers_sets_on_tie_free_keys():
    # covers: MODEL-I8
    # WOULD-FAIL-IF: the native DeepSeek-V3 row selects another expert set than transformers 5.19.0's
    # DeepseekV3TopkRouter on tie-free keys (fidelity, not identity: ADR-0008 decision 9), with its correction
    # bias and groups
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    c = DeepseekV3Config(
        hidden_size=64,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_group=4,
        topk_group=2,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
    )
    r = DeepseekV3TopkRouter(c)
    g = torch.Generator().manual_seed(35)
    with torch.no_grad():
        r.weight.copy_(torch.randn(16, 64, generator=g) * 0.2)
        r.e_score_correction_bias.copy_(torch.randn(16, generator=g) * 0.05)
    x = torch.randn(200, 64, generator=g)
    with torch.no_grad():
        _, w_hf, idx_hf = r(x)
        s = torch.sigmoid(torch.nn.functional.linear(x.float(), r.weight.float()))
        key = s + r.e_score_correction_bias
    # Tie-free: no two keys of a token, and no two group keys, are equal.
    assert all(len(set(row)) == 16 for row in key.tolist())
    sel = arch.moe_select(key, 4, groups=4, topk_groups=2)
    assert sel.tolist() == torch.sort(idx_hf, -1).values.tolist()
    # A deliberately wrong row (no groups) selects other sets on these keys.
    assert arch.moe_select(key, 4).tolist() != sel.tolist()


def test_moe_deepseek_hf_stays_refused():
    # covers: MODEL-I8, MODEL-I10
    # WOULD-FAIL-IF: arch 'hf' deepseek_v3 (MLA, no SUBSTITUTE row) passes the gate instead of being refused
    # MOE-NO-ROW naming its sparse-MoE block
    hfm = _hfm()
    hc = hfm.tiny_config("deepseek_v3")
    blocks = hfm._classify(hfm._full_build(hfm._shrunk(hc, keep_depth=False), 0, swap=False))
    _fired(len(blocks), "deepseek_v3's sparse-MoE blocks classified")
    rep = hfm.gate(hc, FP32_RECIPE, "cpu")
    _refused(rep, "MOE-NO-ROW", "deepseek_v3")
    assert "DeepseekV3MoE" in rep["detail"]


# ------------------------------------------------------------ E2 controls


def test_hf_e2_to_empty_buffers_diverge(monkeypatch):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: a unit chain whose buffers were left as to_empty leaves them (zeroed here) passes the gate
    hfm = _hfm()
    fired = []

    def to_empty(root):
        for b in root.buffers():
            fired.append(int(bool((b != 0).any())))
            b.zero_()

    monkeypatch.setattr(hfm, "_chain_buffers", to_empty)
    rep = hfm.gate(hfm.tiny_config("gemma2"), GATE_RECIPE, "cpu")
    _fired(sum(fired), "to_empty buffers")
    _refused(rep, "GATE-DIFF", "to_empty buffers")


def _via_inputs_embeds(hfm, fired: list):
    orig = hfm._Runtime.run

    def run(self, u, x, ids):
        if u != 1:
            return orig(self, u, x, ids)
        fired.append(1)

        def leave(mod, args, kwargs):
            raise hfm._Exit(args[0] if args else kwargs["hidden_states"])

        hook = self.entry(2).register_forward_pre_hook(leave, with_kwargs=True)
        try:
            with self._autocast(x.device):
                self.root(inputs_embeds=x, use_cache=False)
        except hfm._Exit as e:
            return e.h
        finally:
            hook.remove()
        raise AssertionError("HF's forward never reached unit 2")

    return run


def test_hf_e2_granite_through_inputs_embeds_diverges(monkeypatch):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: re-entering a unit through inputs_embeds (ADR-0009's rejected alternative) instead of the
    # entry pre-hook passes the gate for granite, whose embedding_multiplier is applied after inputs_embeds
    hfm = _hfm()
    fired: list = []
    monkeypatch.setattr(hfm._Runtime, "run", _via_inputs_embeds(hfm, fired))
    hc = hfm.tiny_config("granite", **FAMILIES["granite"])
    rep = hfm.gate(hc, GATE_RECIPE, "cpu")
    _fired(len(fired) * (hc["embedding_multiplier"] != 1.0), "granite's multiplier re-applied")
    _refused(rep, "GATE-DIFF", "granite through inputs_embeds")
    # The trigger is the multiplier: llama re-entered the same way stays exact.
    llama = hfm.gate(hfm.tiny_config("llama"), GATE_RECIPE, "cpu")
    assert llama["first_difference"] is None and llama["cuts"], llama["detail"]


def test_hf_e2_tied_use_dropped_diverges(monkeypatch):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: a chain that keeps only the head's use of a tied table (gemma ties by default) passes the
    # gate, or the gate names another tensor than the tied embedding first
    hfm = _hfm()
    parts_seen: list = []

    def head_only(parts):
        parts_seen.append(len(parts))
        return parts[0]

    monkeypatch.setattr(hfm, "_compose_tied", head_only)
    rep = hfm.gate(hfm.tiny_config("gemma"), GATE_RECIPE, "cpu")
    _fired(sum(1 for n in parts_seen if n > 1), "a tied table with two uses")
    _refused(rep, "GATE-DIFF", "tied use dropped")
    assert "model.embed_tokens.weight" in rep["first_difference"]


def test_hf_e2_exit_one_unit_late_diverges(monkeypatch):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: units that stop one unit late (running the next layer twice across a cut) pass the gate
    hfm = _hfm()
    late: list = []

    def exit_late(u):
        late.append(u)
        return u + 2

    monkeypatch.setattr(hfm, "_exit_unit", exit_late)
    rep = hfm.gate(hfm.tiny_config("qwen3"), GATE_RECIPE, "cpu")
    _fired(len(late), "units exiting late")
    _refused(rep, "GATE-DIFF", "exit one unit late")


def test_hf_e2_layer_types_indexed_locally_diverge(monkeypatch):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: a stage that indexes layer_types from its own first layer (local) instead of the global
    # layer index passes the gate on gemma2, whose sliding and full layers alternate
    hfm = _hfm()
    changed: list = []

    def local(root):
        lt = list(root.config.layer_types)
        loc = [lt[0]] + lt[:-1]  # a stage starting at layer 1 that counts from 0
        for i, layer in enumerate(root.model.layers):
            changed.append(int(loc[i] != lt[i]))
            att = layer.self_attn
            att.layer_type = loc[i]
            att.sliding_window = (
                root.config.sliding_window if loc[i] == "sliding_attention" else None
            )
        root.config.layer_types = loc

    monkeypatch.setattr(hfm, "_chain_buffers", local)
    rep = hfm.gate(hfm.tiny_config("gemma2"), GATE_RECIPE, "cpu")
    _fired(sum(changed), "layers whose type moves under local indexing")
    _refused(rep, "GATE-DIFF", "layer_types indexed locally")


@pytest.mark.parametrize(
    "control, mt, over, recipe, cause, named",
    [
        ("dropout 0.1", "llama", {"attention_dropout": 0.1}, None, "DROPOUT", "attention_dropout"),
        ("sdpa on gemma2", "gemma2", {"attn_implementation": "sdpa"}, None, "ATTN-IMPL", "sdpa"),
        (
            "dynamic rope per layer type",
            "gemma3_text",
            {
                "rope_parameters": {
                    "full_attention": {"rope_type": "dynamic", "factor": 2.0, "rope_theta": 1e6},
                    "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
                }
            },
            None,
            "ROPE-DYNAMIC",
            "dynamic",
        ),
        (
            "longrope past its length",
            "llama",
            {
                "rope_parameters": {
                    "rope_type": "longrope",
                    "rope_theta": 1e4,
                    "short_factor": [1.0] * 8,
                    "long_factor": [2.0] * 8,
                    "original_max_position_embeddings": 8,
                }
            },
            {"seq_len": 16},
            "ROPE-LONGROPE",
            "original_max_position_embeddings=8",
        ),
        (
            "auto_map",
            "llama",
            {"auto_map": {"AutoModelForCausalLM": "x.Y"}},
            None,
            "AUTO-MAP",
            "auto_map",
        ),
    ],
)
def test_hf_e2_config_controls_are_refused_naming_the_field(
    control, mt, over, recipe, cause, named
):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a dropout above 0, sdpa attention, a dynamic rope in any layer type's entry, a longrope
    # whose original length is below the recipe's S, or remote code (auto_map) reaches training instead of being
    # refused before any build, naming the field
    hfm = _hfm()
    hc = {**hfm.tiny_config(mt), **over}
    trig = {
        "DROPOUT": sum(
            1 for k, v in hc.items() if "dropout" in k and isinstance(v, float) and v > 0
        ),
        "ATTN-IMPL": int(hc.get("attn_implementation") == "sdpa"),
        "ROPE-DYNAMIC": sum(
            "dynamic" in str(v.get("rope_type"))
            for v in hc["rope_parameters"].values()
            if isinstance(v, dict)
        )
        if cause == "ROPE-DYNAMIC"
        else 0,
        "ROPE-LONGROPE": int(
            cause == "ROPE-LONGROPE"
            and hc["rope_parameters"]["original_max_position_embeddings"]
            < (recipe or {}).get("seq_len", 0)
        ),
        "AUTO-MAP": int(bool(hc.get("auto_map"))),
    }[cause]
    _fired(trig, control)
    rep = hfm.gate(hc, {**GATE_RECIPE, **(recipe or {})}, "cpu")
    _refused(rep, cause, control)
    assert named in rep["detail"] and rep["cuts"] == []


def test_hf_e2_kernels_importable_is_refused(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: an importable `kernels` (hub kernels that replace HF's forwards) passes the guard or leaves
    # the probe vector unflagged
    stub = tmp_path / "stubs" / "kernels"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("")
    r = _probe(tmp_path, pythonpath=str(tmp_path / "stubs"))
    _fired(sum("kernels" in x for x in r["refusals"]), "kernels importable in the child")
    assert r["guard"][0] == "PROBED-PKG" and "kernels" in r["guard"][1]
    assert "FP-PROBE" in _causes(r)


class _ToyConfig:
    """The few config attributes the trace and the units read, for non-HF control models."""

    def __init__(self, n: int = 4, h: int = 16, V: int = 32):
        self.num_hidden_layers, self.hidden_size, self.vocab_size = n, h, V
        self.tie_word_embeddings = False

    def to_dict(self) -> dict:
        return {
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": 32,
        }


class _ToyLayer(nn.Module):
    def __init__(self, h: int, threaded: bool):
        super().__init__()
        self.lin, self.mix, self.threaded = nn.Linear(h, h), nn.Linear(h, h), threaded

    def forward(self, x, side):
        new = torch.tanh(self.mix(side)) + (side if self.threaded else 0.0)
        y = x + self.lin(x) * new
        return (y, new) if self.threaded else y


class _ToyModel(nn.Module):
    """A decoder whose layers read a side input besides the residual: the state the previous layer returned
    (threaded, ADR-0009 ctrl_thread) or the prologue's embedding output (every layer reads it)."""

    def __init__(self, threaded: bool):
        super().__init__()
        torch.manual_seed(40)
        self.config = _ToyConfig()
        self.embed = nn.Embedding(32, 16)
        self.layers = nn.ModuleList(_ToyLayer(16, threaded) for _ in range(4))
        self.head = nn.Linear(16, 32)
        self.threaded = threaded

    def forward(self, input_ids=None, use_cache=False):
        e = self.embed(input_ids)
        h, side = e, torch.zeros_like(e) if self.threaded else e
        for layer in self.layers:
            if self.threaded:
                h, side = layer(h, side)
            else:
                h = layer(h, side)
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.head(h))


def _toy_chain_diff(model, umap=None) -> str | None:
    """The gate's comparison over a non-HF control model: every single cut, fp32, M 1."""
    hfm = _hfm()
    umap = umap or hfm._traced(model, None)
    us = hfm.units(copy.deepcopy(model), umap)
    wgrad.convert(us, exclude=us.exclusions)
    batches = [(_ids(32, seed=41), _ids(32, seed=42))]
    ref = hfm._ref_pass(model, batches, "fp32", torch.device("cpu"))
    n = umap.n
    for bounds in [[0, c, n + 2] for c in range(1, n + 2)]:
        diff = hfm._first_diff(ref, hfm._chain_pass(us, bounds, batches, False), str(bounds))
        if diff:
            return diff
    return None


def test_hf_e2_threaded_state_nan_stubs_are_caught_and_replays_refused(monkeypatch):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: a layer reading state the previous layer returned (threaded state) passes the comparison
    # through a stubbed layer (NaN stubs must carry the read to the logits), or a stub replaying the trace's
    # recorded outputs is admitted instead of refused on the gate's independent batch shape (ctrl_thread)
    hfm = _hfm()
    model = _ToyModel(threaded=True)
    umap = hfm._traced(model, None)
    _fired(sum(1 for s in umap.layer_out if s), "layers returning threaded state")
    assert _toy_chain_diff(model) is not None
    # A replay: each stub returns what its layer returned in the trace (shape (bs, S) of the trace's ids).
    bs, S = umap.trace_shape
    recorded: dict = {}
    hooks = [
        layer.register_forward_hook(lambda m, a, o, i=i: recorded.setdefault(i, o))
        for i, layer in enumerate(model.layers)
    ]
    with torch.no_grad():
        model(input_ids=torch.randint(0, 32, (bs, S), generator=torch.Generator().manual_seed(43)))
    for h in hooks:
        h.remove()
    index = {id(s): i for i, s in enumerate(umap.layer_out)}
    monkeypatch.setattr(
        hfm._LayerStub, "forward", lambda self, *a, **k: recorded[index[id(self.struct)]]
    )
    with pytest.raises(RuntimeError):
        _toy_chain_diff(model, umap)


def test_hf_e2_zero_prologue_stubs_diverge(monkeypatch):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: prologue stubs returning zeros (a plausible value) instead of fresh NaN let a model whose
    # later layers read the prologue's output pass the comparison
    hfm = _hfm()
    model = _ToyModel(threaded=False)
    us = hfm.units(copy.deepcopy(model), hfm._traced(model, None))
    ids = _ids(32, seed=44)
    with torch.no_grad():
        h = call_layer(us[1], call_layer(us[0], ids, ids=ids), ids=ids)
        outs = []
        for fill in (0.0, 1.0):
            monkeypatch.setattr(
                hfm._PrologueStub,
                "forward",
                lambda self, t, *a, _f=fill, **k: torch.full(
                    tuple(t.shape) + (self.trailing,), _f, dtype=self.dt
                ),
            )
            outs.append(call_layer(us[2], h, ids=ids))
    _fired(int(not torch.equal(*outs)), "a later unit reading the prologue's output")
    monkeypatch.setattr(
        hfm._PrologueStub,
        "forward",
        lambda self, t, *a, **k: torch.zeros(tuple(t.shape) + (self.trailing,), dtype=self.dt),
    )
    assert _toy_chain_diff(model) is not None


def test_hf_e2_mamba_read_elsewhere_linear_converted_diverges(monkeypatch):
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: mamba's dt_proj (its weight read outside its own forward) converted to SplitLinear passes
    # the gate: its read-elsewhere gradient would be lost to the deferred store
    hfm = _hfm()
    hc = hfm.tiny_config("mamba")
    dt = [k for k in hfm.unit_map(hc).exclusions if k.endswith("dt_proj")]
    _fired(len(dt), "read-elsewhere Linears in mamba's exclusion set")
    monkeypatch.setattr(hfm.HFUnits, "exclusions", property(lambda self: frozenset()))
    rep = hfm.gate(hc, FP32_RECIPE, "cpu")
    _refused(rep, "GATE-DIFF", "mamba dt_proj converted")
    assert "dt_proj" in rep["first_difference"]


def test_hf_e2_phimoe_rng_router_is_refused():
    # covers: MODEL-I9, MODEL-I10
    # WOULD-FAIL-IF: phimoe's router, which draws from the generator in training mode, passes the trace
    hfm = _hfm()
    hc = hfm.tiny_config("phimoe")
    m = hfm._full_build(hfm._shrunk(hc, keep_depth=False), 0, swap=False)
    with torch.random.fork_rng(devices=[]):
        s0 = torch.get_rng_state()
        with torch.no_grad():
            m(input_ids=_ids(128), use_cache=False)
        _fired(int(not torch.equal(s0, torch.get_rng_state())), "phimoe's forward drawing")
    rep = hfm.gate(hc, FP32_RECIPE, "cpu")
    _refused(rep, "TRACE-RNG", "phimoe")


def test_hf_e2_split_alias_is_refused():
    # covers: MODEL-I9
    # WOULD-FAIL-IF: one Parameter held by two layers' units (an alias split across a cut, other than the
    # embedding/head tie) passes the trace
    hfm = _hfm()
    m = hfm._full_build(hfm.tiny_config("llama"), 0, swap=False)
    m.model.layers[2].mlp.up_proj.weight = m.model.layers[1].mlp.up_proj.weight
    shared = {id(p) for p in m.model.layers[1].parameters()} & {
        id(p) for p in m.model.layers[2].parameters()
    }
    _fired(len(shared), "a parameter shared by two layers")
    with pytest.raises(hfm.HFRefused, match="up_proj") as e:
        hfm.trace(m)
    assert e.value.cause == "TRACE-SHARED"


def test_hf_e2_bert_is_refused_by_the_causality_probe():
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a bidirectional model of the causal-LM mapping (bert) passes the gate
    hfm = _hfm()
    hc = hfm.tiny_config("bert", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    m = hfm._full_build(hfm._shrunk(hc, keep_depth=False), 0, swap=False)
    a = _ids(128, rows=1)
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 128
    with torch.no_grad():
        la, lb = m(input_ids=a).logits, m(input_ids=b).logits
    _fired(int((la[0, :-1] != lb[0, :-1]).any(-1).sum()), "earlier positions reading the last id")
    _refused(hfm.gate(hc, GATE_RECIPE, "cpu"), "GATE-CAUSAL", "bert")


@pytest.mark.parametrize("mt", ["llama4_text", "gpt_oss", "deepseek_v3", "jamba"])
def test_hf_e2_moe_without_a_row_is_refused(mt):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: D8 misses a sparse-MoE block of llama4_text (dense dispatch weighted by scores), gpt_oss,
    # deepseek_v3 or jamba, or a block without a SUBSTITUTE row runs HF's own MoE forward instead of being refused
    hfm = _hfm()
    hc = hfm.tiny_config(mt)
    blocks = hfm._classify(hfm._full_build(hfm._shrunk(hc, keep_depth=False), 0, swap=False))
    _fired(len(blocks), f"{mt}'s classified blocks")
    rep = hfm.gate(hc, FP32_RECIPE, "cpu")
    _refused(rep, "MOE-NO-ROW", mt)
    assert any(name in rep["detail"] for name in blocks.values()), rep["detail"]


def test_hf_e2_in_place_stub_write_is_refused(monkeypatch):
    # covers: MODEL-I9
    # WOULD-FAIL-IF: a write into a stride-0 NaN layer stub inside the gate goes unnoticed instead of refusing
    # STUB-WRITE
    hfm = _hfm()
    writes: list = []

    def scribble(mod, args):
        h = args[0] if args else None
        if isinstance(h, torch.Tensor) and h.dim() and 0 in h.stride() and bool(h.isnan().all()):
            writes.append(1)
            h.mul_(1.0)

    hook = torch.nn.modules.module.register_module_forward_pre_hook(scribble)
    try:
        rep = hfm.gate(hfm.tiny_config("llama"), GATE_RECIPE, "cpu")
    finally:
        hook.remove()
    _fired(len(writes), "a write into a stub")
    _refused(rep, "STUB-WRITE", "in-place stub write")


@pytest.mark.parametrize("op", ["zero_", "masked_fill_"])
def test_hf_e2_stride0_admitted_stub_write_is_refused_and_never_reused(monkeypatch, op):
    # covers: MODEL-I9
    # WOULD-FAIL-IF: a write that PyTorch admits on a stride-0 view (zero_, masked_fill_) into a layer stub's
    # cached NaN goes unrefused, or the written scalar is reused, so every later stub of the process returns 0
    # instead of NaN (threaded-state reads then pass silently)
    hfm = _hfm()
    writes: list = []

    def scribble(mod, args):
        h = args[0] if args else None
        if isinstance(h, torch.Tensor) and h.dim() and 0 in h.stride() and bool(h.isnan().all()):
            writes.append(1)
            if op == "zero_":
                h.zero_()
            else:
                h.masked_fill_(torch.ones(h.shape, dtype=torch.bool), 0.0)

    hook = torch.nn.modules.module.register_module_forward_pre_hook(scribble)
    try:
        rep = hfm.gate(hfm.tiny_config("llama"), GATE_RECIPE, "cpu")
    finally:
        hook.remove()
    _fired(len(writes), f"a stub written by {op}")
    _refused(rep, "STUB-WRITE", f"stub written by {op}")
    stub = hfm._LayerStub(None, (3, 11))(torch.zeros(2, 4, 8))
    assert bool(stub.isnan().all()), "a written stub scalar was reused"


def test_hf_e2_lora_on_hf_is_refused():
    # covers: MODEL-I10
    # WOULD-FAIL-IF: LoRA on arch 'hf' validates or passes check_config
    from types import SimpleNamespace

    hfm = _hfm()
    cfg = GPTConfig(arch="hf", hf_config=dict(_LLAMA_HF), lora_rank=2)
    _fired(cfg.lora_rank, "lora_rank")
    with pytest.raises(ValueError, match="lora_rank"):
        cfg.validate()
    with pytest.raises(hfm.HFRefused, match="lora_rank") as e:
        hfm.check_config(
            hfm.tiny_config("llama"),
            SimpleNamespace(lora_rank=2, seq_len=16, boundary_dtype="fp32", fp8_boundaries=False),
        )
    assert e.value.cause == "LORA"


def test_hf_e2_eval_mode_is_refused(monkeypatch):
    # covers: MODEL-I9
    # WOULD-FAIL-IF: a unit chain run in eval mode (another code path than the stages train) passes the gate
    hfm = _hfm()
    flipped: list = []

    def to_eval(root):
        root.eval()
        flipped.append(sum(not m.training for m in root.modules()))

    monkeypatch.setattr(hfm, "_chain_buffers", to_eval)
    rep = hfm.gate(hfm.tiny_config("qwen2"), GATE_RECIPE, "cpu")
    _fired(sum(flipped), "modules in eval mode")
    _refused(rep, "EVAL-MODE", "eval mode")


@pytest.mark.parametrize("mt", ["mamba", "falcon_h1", "rwkv"])
def test_hf_e2_classifier_admits_state_space_and_recurrent_mixers(mt):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: D8 classifies a selective scan (mamba, falcon_h1) or rwkv's recurrence as a sparse-MoE
    # block (refusing it MOE-NO-ROW), or the gate refuses them otherwise
    hfm = _hfm()
    hc = hfm.tiny_config(mt)
    m = hfm._full_build(hfm._shrunk(hc, keep_depth=False), 0, swap=False)
    assert hfm._classify(m) == {}
    rep = hfm.gate(hc, FP32_RECIPE, "cpu")
    assert rep["first_difference"] is None and len(rep["cuts"]) == (_gate_depth(hc) + 4) * 2
    _assert_platform_cause(rep)


def test_hf_e2_mixtral_without_its_block_row_is_refused(monkeypatch):
    # covers: MODEL-I8, MODEL-I10
    # WOULD-FAIL-IF: Mixtral with its block row removed from SUBSTITUTE (its router and experts rows kept) runs or
    # admits instead of being refused MOE-NO-ROW
    hfm = _hfm()
    hc = hfm.tiny_config("mixtral")
    rows = {k: v for k, v in hfm.SUBSTITUTE.items() if k[0] != "MixtralSparseMoeBlock"}
    assert len(rows) == len(hfm.SUBSTITUTE) - 1
    monkeypatch.setattr(hfm, "SUBSTITUTE", rows)
    blocks = hfm._classify(hfm._full_build(hfm._shrunk(hc, keep_depth=False), 0, swap=False))
    _fired(len(blocks), "Mixtral's classified blocks")
    rep = hfm.gate(hc, FP32_RECIPE, "cpu")
    _refused(rep, "MOE-NO-ROW", "Mixtral without its block row")
    assert "MixtralSparseMoeBlock" in rep["detail"]


def test_hf_e2_swapped_block_running_experts_inline_is_refused_by_the_trace(monkeypatch):
    # covers: MODEL-I8, MODEL-I10
    # WOULD-FAIL-IF: a swapped block that runs its experts' F.linear inline instead of calling its experts module
    # passes the unit trace (the seam rule: the block calls its router and experts as modules, so the trace sees
    # every parameter-holding module)
    hfm = _hfm()

    class Inline(hfm.MixtralSparseMoeBlockSwap):
        def forward(self, hidden_states):
            view = hfm._MoEView(self)
            experts = self.experts
            view.experts = lambda xd, counts: hfm._SwapExperts.forward(experts, xd, counts)
            return arch.moe_forward(self.moe_cfg, view, hidden_states)

    rows = dict(hfm.SUBSTITUTE)
    rows[("MixtralSparseMoeBlock", hfm.PIN)] = Inline
    monkeypatch.setattr(hfm, "SUBSTITUTE", rows)
    hc = hfm.tiny_config("mixtral")
    m = hfm._full_build(hfm._shrunk(hc, keep_depth=False), 0, swap=True)
    _fired(sum(type(x) is Inline for x in m.modules()), "blocks swapped to the inline class")
    rep = hfm.gate(hc, FP32_RECIPE, "cpu")
    _refused(rep, "TRACE-UNPLACED", "experts run inline")
    assert "mlp.experts" in rep["detail"]


def test_hf_e2_r13_where_on_the_prologue_output_is_recorded(monkeypatch, capsys):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: the gate crashes on a NaN-absorbing read of the prologue output (ADR-0009 R13: recorded,
    # not judged); the verdict is printed for the record, never asserted
    hfm = _hfm()
    reads: list = []

    def absorb(mod, args, out):
        if isinstance(mod, (nn.Embedding, hfm._PrologueStub)) and isinstance(out, torch.Tensor):
            reads.append(1)
            return torch.where(torch.isinf(out).any(), out * 0, out)
        return out

    hook = torch.nn.modules.module.register_module_forward_hook(absorb)
    try:
        rep = hfm.gate(hfm.tiny_config("llama"), GATE_RECIPE, "cpu")
    finally:
        hook.remove()
    _fired(len(reads), "where(isinf(x).any(), a, b) on the prologue output")
    print(
        "R13 where(isinf(x).any()) on the prologue output:",
        json.dumps({"cause": rep["cause"], "first_difference": rep["first_difference"]}),
    )
    assert rep["cuts"] or rep["cause"]


# ------------------------------------------------------------- E6 drift
#
# Every arm runs a child in stage_env (with -X pycache_prefix unless the arm is about the prefix) over a
# symlinked copy of the installed distributions under tmp_path; the real environment is never touched.

_E6_PROBE = (
    "import json, os, sys\n"
    "import torch\n"
    "import torch.nn.functional as F\n"
    "from wantrain.model import hfm\n"
    "import transformers\n"
    "from transformers.models.llama import modeling_llama\n"
    "PRE_OUT = None\n"
    "exec(os.environ.get('E6_PRE', ''))\n"
    "out = {'planted': [getattr(modeling_llama, 'E6_PLANTED', None), getattr(F, 'E6_PLANTED', None)],\n"
    "       'pre': PRE_OUT, 'files': [torch.__file__, transformers.__file__]}\n"
    "try:\n"
    "    hfm.guard_environment(); out['guard'] = None\n"
    "except hfm.HFRefused as e:\n"
    "    out['guard'] = [e.cause, str(e)]\n"
    "fp = hfm.fingerprint()\n"
    "out['refusals'] = fp['refusals']\n"
    "out['admitted'] = hfm.platform_admitted()\n"
    "print(json.dumps(out))\n"
)


def _site_copy(tmp_path, dists=("transformers",)):
    """A copy of each distribution (every top-level entry its RECORD lists, and its dist-info) under
    tmp_path/site: real directories, every file a symlink to the installed one, no __pycache__."""
    import importlib.metadata as md

    site = tmp_path / "site"
    for name in dists:
        d = md.distribution(name)
        info = str(d._path)  # type: ignore[attr-defined]
        root = os.path.dirname(info)
        tops = {str(f).split("/")[0] for f in d.files or [] if not str(f).startswith("..")}
        for top in sorted(tops | {os.path.basename(info)}):
            src = os.path.join(root, top)
            if os.path.isfile(src):
                site.mkdir(parents=True, exist_ok=True)
                os.symlink(src, site / top)
                continue
            for dp, dns, fns in os.walk(src):
                dns[:] = [x for x in dns if x != "__pycache__"]
                rel = os.path.relpath(dp, root)
                os.makedirs(site / rel, exist_ok=True)
                for fn in fns:
                    os.symlink(os.path.join(dp, fn), site / rel / fn)
    return site


def _e6(tmp_path, site=None, prefix=True, env_extra=None, pre="", tag="e6") -> dict:
    hfm = _hfm()
    env = hfm.stage_env(dict(os.environ))
    env.update(env_extra or {})
    env["E6_PRE"] = pre
    env["E6_TMP"] = str(tmp_path)
    if site is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(site)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
    args = [sys.executable]
    if prefix:
        args += hfm.interpreter_args(str(tmp_path / f"pyc-{tag}"))
    out = subprocess.run(
        args + ["-c", _E6_PROBE], env=env, capture_output=True, text=True, timeout=900, cwd=tmp_path
    )
    assert out.returncode == 0, out.stderr[-3000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    if site is not None:  # the copied distributions were the ones imported
        assert r["files"][1].startswith(str(site)), r["files"]
        assert r["files"][0].startswith(str(site)) == (site / "torch").exists(), r["files"]
    return r


def _drift(r: dict) -> list:
    """The refusals beyond the platform's own (a non-admitted platform's FP-TORCH-RECORDS)."""
    return [x for x in r["refusals"] if r["admitted"] or not x.startswith("FP-TORCH-RECORDS")]


def _replace_with_copy(site, rel: str, extra: str) -> None:
    p = site / rel
    text = open(os.path.realpath(p)).read()
    os.unlink(p)
    p.write_text(text + extra)


@_LINUX_CUDA_FINGERPRINT
def test_hf_e6_clean_site_copy_is_admitted(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: an unchanged copy of the installed transformers and torch (the control every drift arm is
    # judged against) is refused, so a drift refusal would prove nothing
    site = _site_copy(tmp_path, ("transformers", "torch"))
    r = _e6(tmp_path, site)
    assert r["guard"] is None and _drift(r) == [], r["refusals"]
    assert r["planted"] == [None, None]


@pytest.mark.parametrize(
    "rel",
    [
        "transformers/models/llama/modeling_llama.py",
        "transformers/activations.py",
        "transformers/utils/generic.py",
    ],
)
def test_hf_e6_edited_transformers_file_is_refused(tmp_path, rel):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: an edited transformers source (a model file, activations, utils/generic) differs from its
    # RECORD sha256 and the fingerprint does not refuse it naming the file
    site = _site_copy(tmp_path)
    _replace_with_copy(site, rel, "\n# drift\n")
    _fired(
        int(os.path.realpath(site / rel) == str(site / rel)), f"{rel} replaced by an edited copy"
    )
    r = _e6(tmp_path, site)
    rec = [x for x in r["refusals"] if x.startswith("FP-RECORD")]
    assert len(rec) == 1 and os.path.basename(rel) in rec[0], r["refusals"]


def test_hf_e6_added_and_sourceless_files_are_refused(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a file added to the transformers tree, or a sourceless .pyc beside its modules (importable
    # without any source), is admitted instead of refused as outside its RECORD
    import py_compile

    site = _site_copy(tmp_path)
    (site / "transformers/models/llama/extra_e6.py").write_text("X = 1\n")
    src = tmp_path / "e6_sourceless.py"
    src.write_text("E6_SOURCELESS = 1\n")
    py_compile.compile(str(src), cfile=str(site / "transformers/e6_sourceless.pyc"))
    pre = "import transformers.e6_sourceless as s; PRE_OUT = s.E6_SOURCELESS"
    r = _e6(tmp_path, site, pre=pre)
    _fired(r["pre"] or 0, "the sourceless .pyc imported")
    rec = [x for x in r["refusals"] if x.startswith("FP-RECORD")]
    assert any("extra_e6.py" in x for x in rec) and any("e6_sourceless.pyc" in x for x in rec), rec


@_LINUX_CUDA_FINGERPRINT
@pytest.mark.parametrize(
    "dist, rel",
    [
        ("transformers", "transformers/models/llama/modeling_llama.py"),
        ("torch", "torch/nn/functional.py"),
    ],
)
def test_hf_e6_planted_pycache_runs_only_without_the_prefix(tmp_path, dist, rel):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a planted __pycache__ .pyc carrying the source's mtime and size runs in a launch with the
    # fresh pycache prefix, or a launch without the prefix (where it does run) passes the guard or the
    # fingerprint
    import importlib._bootstrap_external as be

    site = _site_copy(tmp_path, ("transformers", "torch"))
    real = os.path.realpath(site / rel)
    st = os.stat(real)
    with open(real) as fh:
        code = compile(fh.read() + "\nE6_PLANTED = 1\n", real, "exec")
    cache = site / os.path.dirname(rel) / "__pycache__"
    cache.mkdir(exist_ok=True)
    name = f"{os.path.basename(rel)[:-3]}.{sys.implementation.cache_tag}.pyc"
    (cache / name).write_bytes(bytes(be._code_to_timestamp_pyc(code, int(st.st_mtime), st.st_size)))
    slot = 0 if dist == "transformers" else 1
    bare = _e6(tmp_path, site, prefix=False, tag="bare")
    _fired(bare["planted"][slot] or 0, f"the planted {dist} .pyc running without the prefix")
    assert bare["guard"][0] == "ENV-PREFIX" and any(
        x.startswith("FP-PYC") for x in bare["refusals"]
    )
    r = _e6(tmp_path, site, tag="prefixed")
    assert r["planted"] == [None, None] and r["guard"] is None and _drift(r) == [], r


@pytest.mark.parametrize("pkg", ["mambapy", "kernels"])
def test_hf_e6_local_probed_package_stub_is_refused(tmp_path, pkg):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a local stub of a probed package (importable, no distribution metadata; mambapy flips
    # use_mambapy's branch, kernels replaces forwards) passes the guard or the probe vector
    stub = tmp_path / "stubs" / pkg
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("")
    pre = f"import importlib.util; PRE_OUT = int(importlib.util.find_spec({pkg!r}) is not None)"
    r = _e6(tmp_path, env_extra={"PYTHONPATH": str(tmp_path / "stubs")}, pre=pre)
    _fired(r["pre"], f"{pkg} importable")
    assert r["guard"][0] == "PROBED-PKG" and pkg in r["guard"][1]
    assert any(x.startswith("FP-PROBE") and pkg in x for x in r["refusals"])


def test_hf_e6_ld_preload_is_refused(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a launch with LD_PRELOAD (a preloaded library can replace any kernel) passes the guard
    r = _e6(tmp_path, env_extra={"LD_PRELOAD": str(tmp_path / "none.so")})
    assert r["guard"][0] == "ENV-LOADER" and "LD_PRELOAD" in r["guard"][1]


@_LINUX_CUDA_FINGERPRINT
def test_hf_e6_torch_native_library_from_a_copy_is_refused(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a torch native library loaded from a copy elsewhere (its RECORD basename, another path) is
    # admitted by the native set instead of refused FP-NATIVE naming it
    lib = "libtorch_global_deps." + ("dylib" if sys.platform == "darwin" else "so")
    pre = (
        "import ctypes, shutil\n"
        f"src = os.path.join(os.path.dirname(torch.__file__), 'lib', {lib!r})\n"
        f"dst = os.path.join(os.environ['E6_TMP'], 'copy', {lib!r})\n"
        "os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)\n"
        "ctypes.CDLL(dst); PRE_OUT = 1\n"
    )
    r = _e6(tmp_path, pre=pre)
    _fired(r["pre"], "a copy of a torch native library loaded")
    nat = [x for x in r["refusals"] if x.startswith("FP-NATIVE")]
    assert nat and lib in nat[0], r["refusals"]


def test_hf_e6_monkeypatch_in_the_child_is_refused(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a transformers method replaced in the launched process keeps code_ok
    pre = "modeling_llama.LlamaMLP.forward = lambda self, x: x; PRE_OUT = 1"
    r = _e6(tmp_path, pre=pre)
    _fired(r["pre"], "LlamaMLP.forward replaced")
    assert any(x.startswith("FP-CODE") and "LlamaMLP.forward" in x for x in r["refusals"])


def test_hf_e6_torch_records_entries_admit_and_a_record_outside_them_refuses(monkeypatch):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: two test TORCH_RECORDS entries (this platform's measured records and another platform's)
    # do not admit this node, or a RECORD outside them (one distribution's sha256 changed) is admitted
    hfm = _hfm()
    tor = hfm._env_fingerprint()["torch"]
    mine = (tor["platform_tag"], tor["version"])
    other = (
        "cp312-cp312-manylinux_2_28_x86_64"
        if "mac" in mine[0]
        else "cp312-cp312-macosx_14_0_arm64",
        mine[1],
    )
    monkeypatch.setattr(
        hfm, "TORCH_RECORDS", {mine: dict(tor["records"]), other: {"torch": "0" * 64}}
    )
    assert hfm.platform_admitted()
    assert not any(x.startswith("FP-TORCH-RECORDS") for x in hfm.fingerprint()["refusals"])
    bad = dict(tor["records"])
    bad["torch"] = "f" * 64
    _fired(int(bad != tor["records"]), "a RECORD outside the entries")
    monkeypatch.setattr(hfm, "TORCH_RECORDS", {mine: bad, other: dict(tor["records"])})
    refused = [x for x in hfm.fingerprint()["refusals"] if x.startswith("FP-TORCH-RECORDS")]
    assert refused and "torch differs" in refused[0], refused


def test_hf_e6_native_sets_that_differ_compare_unequal():
    # covers: MODEL-I10
    # WOULD-FAIL-IF: two fingerprints whose OS-object sets differ compare equal on 'native' (the admission
    # refusal of that pair is control-m4's hf_admission test)
    hfm = _hfm()
    a = hfm.fingerprint()
    b = copy.deepcopy(a)
    b["native"] = sorted(b["native"] + [["os", "libextra.dylib", "A" * 43]])
    _fired(len(b["native"]) - len(a["native"]), "an extra OS object")
    assert a["native"] and a["native"] != b["native"]
    assert hfm.fingerprint()["native"] == a["native"]  # stable within one process


def _record_child(tmp_path, tag: str, site=None) -> dict:
    hfm = _hfm()
    env = hfm.stage_env(dict(os.environ))
    if site is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(site)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
    inp, out = tmp_path / f"{tag}.in", tmp_path / f"{tag}.out"
    inp.write_text(json.dumps({"hf_config": hfm.tiny_config("llama"), "seed": 3}))
    args = [
        sys.executable,
        *hfm.interpreter_args(str(tmp_path / f"pyc-{tag}")),
        "-m",
        "wantrain.model.hfm",
    ]
    res = subprocess.run(
        args + ["record", "--in", str(inp), "--out", str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(out.read_text())


def test_hf_e6_drifted_record_builder_is_refused(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a record builder run on an edited site (its fingerprint differs from the clean nodes') is
    # admitted by the same comparison that admits a clean builder
    from wantrain.coordinator import coordinator

    hfm = _hfm()
    site = _site_copy(tmp_path)
    _replace_with_copy(site, "transformers/models/llama/modeling_llama.py", "\n# drift\n")
    clean = _record_child(tmp_path, "clean")["fingerprint"]
    drifted = _record_child(tmp_path, "drifted", site)["fingerprint"]
    _fired(int(drifted != clean), "a builder fingerprint that differs from the clean node's")
    rep = {"ok": True, "device": "cpu", "device_class": "cpu:test", "cause": None, "detail": None}

    def admit(builder):
        return coordinator.hf_admission(
            {"n0": [rep]},
            {"n0": clean},
            builder,
            {},
            ["n0"],
            torch_records=hfm.TORCH_RECORDS,
            record_diff=hfm.record_diff,
        )

    base = admit(clean)
    if hfm.platform_admitted():
        assert base == [], base
    refused = [x for x in admit(drifted) if x not in base]
    assert any("record builder" in x and "modeling_llama.py" in x for x in refused), refused


def _machine_label(cmd: str) -> str:
    import platform as _pf

    chip = _pf.processor() or _pf.machine()
    if sys.platform == "darwin":
        chip = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        return f"MEASURED (local Mac: {chip}, {cmd})"
    return f"MEASURED (local {_pf.system()}: {chip}, {cmd})"


def _report(tmp_path, name: str, data: dict) -> str:
    """Write a measurement report as JSON (FACTORY_HF_REPORT_DIR, else the test's tmp_path) and print it."""
    d = os.environ.get("FACTORY_HF_REPORT_DIR") or str(tmp_path)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    print(name, json.dumps(data, sort_keys=True))
    return path


def test_hf_e6_fresh_prefix_compile_time_is_measured(tmp_path):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: a launch's fresh pycache prefix is not populated by the launch itself (bytecode compiled
    # elsewhere would be read), or the compile time is not recorded MEASURED in the JSON report
    import time

    hfm = _hfm()
    code = "import torch, transformers; from transformers.models.llama import modeling_llama"
    pre = tmp_path / "prefix"
    args = [sys.executable, *hfm.interpreter_args(str(pre)), "-c", code]
    env = hfm.stage_env(dict(os.environ))
    env.pop("PYTHONDONTWRITEBYTECODE", None)  # a launch compiles into its prefix
    times = []
    for _ in range(2):  # fresh, then warm (the same prefix)
        t0 = time.perf_counter()
        res = subprocess.run(
            args, env=env, capture_output=True, text=True, timeout=900, cwd=tmp_path
        )
        times.append(time.perf_counter() - t0)
        assert res.returncode == 0, res.stderr[-2000:]
    pycs = [os.path.join(dp, f) for dp, _, fs in os.walk(pre) for f in fs if f.endswith(".pyc")]
    _fired(len(pycs), "bytecode compiled into the fresh prefix")
    path = _report(
        tmp_path,
        "hf-e6-compile.json",
        {
            "fresh_prefix_import_s": round(times[0], 3),
            "warm_prefix_import_s": round(times[1], 3),
            "compile_s": round(times[0] - times[1], 3),
            "pyc_files": len(pycs),
            "label": _machine_label(
                " ".join(["python", "-X", "pycache_prefix=<fresh>", "-c", repr(code)])
            ),
        },
    )
    assert json.load(open(path))["label"].startswith("MEASURED")


# --------------------------------------------------------- E7 measurements
#
# Timings are written to a JSON report labelled MEASURED; the F7 bounds (ADR-0009) and F6 (ADR-0008) are
# asserted only under FACTORY_E7_BOUNDS=1, which GATES-M4 sets on a quiet machine. Byte counts are exact and
# asserted in every run.

E7_BOUNDS = os.environ.get("FACTORY_E7_BOUNDS") == "1"


def _median_s(fn, reps: int = 20) -> float:
    import statistics
    import time

    fn()  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


@contextlib.contextmanager
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


def _saved_s2_bytes(model, ids, compute: str) -> int:
    """Bytes of the b*H*S*S-element score tensors autograd saves in one forward (each storage once)."""
    b, S = ids.shape
    H = int(model.config.num_attention_heads)
    seen: dict = {}

    def pack(t):
        # (b, H, S, S), or the (b*H, S, S) view a batched matmul saves
        if t.dim() >= 3 and tuple(t.shape[-2:]) == (S, S) and t.numel() == b * H * S * S:
            key = (t.untyped_storage().data_ptr(), t.storage_offset(), t.dtype)
            seen[key] = t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=compute == "bf16"):
            model(input_ids=ids, use_cache=False)
    return sum(seen.values())


@pytest.mark.slow
@pytest.mark.parametrize(
    "fam, compute, want",
    [("llama", "fp32", 4), ("llama", "bf16", 6), ("gemma2", "fp32", 8), ("gemma2", "bf16", 8)],
)
def test_hf_e7_eager_attention_saved_bytes_per_bhs2(fam, compute, want):
    # covers: MODEL-I9
    # WOULD-FAIL-IF: eager attention saves another number of bytes per b*H*S^2 than ADR-0009's model (4 fp32, 6
    # under autocast, a softcap +4 fp32 / +2 autocast), which the planner's eager memory term charges
    hfm = _hfm()
    over = {"attn_logit_softcapping": 50.0} if fam == "gemma2" else {}
    hc = hfm.tiny_config(fam, **over)
    m = hfm._full_build(hc, 0, swap=True)
    ids = _ids(int(m.config.vocab_size), S=24)  # S equal to no dimension (head_dim is 16)
    got = _saved_s2_bytes(m, ids, compute)
    b, S = ids.shape
    per = got / (b * int(m.config.num_attention_heads) * S * S * int(m.config.num_hidden_layers))
    assert per == want, (fam, compute, per)


@pytest.mark.slow
def test_hf_e7_layer_stubs_allocate_nothing():
    # covers: MODEL-I9
    # WOULD-FAIL-IF: a layer stub allocates per call (ADR-0009 D2: stubs write 0 B); the measure itself is
    # checked against a stub that allocates a fresh NaN tensor
    from torch.profiler import ProfilerActivity, profile

    hfm = _hfm()
    res = torch.randn(2, 16, 64)
    stubs = [
        hfm._LayerStub(None, (3, 11)),
        hfm._LayerStub([("res",), ("t", "float32", ("b", "t", 64)), ("v", None)], (3, 11)),
    ]

    def allocated(fn) -> int:
        fn()
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            for _ in range(10):
                fn()
        return sum(max(int(e.cpu_memory_usage), 0) for e in prof.events())

    fresh = allocated(lambda: torch.full((2, 16, 64), float("nan")))
    _fired(fresh, "the profiler seeing a fresh NaN tensor")
    for st in stubs:
        assert allocated(lambda st=st: st(res)) == 0


def _wrapped_llama(root):
    """A0'': the user wraps HF's modules into a flat list themselves (embedding, decoder layers, norm and head),
    computing the rotary embedding and the causal mask in their own glue."""

    class Embed(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = root.model.embed_tokens

        def forward(self, ids):
            return self.m(ids)

    class Layer(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, h):
            S = h.shape[1]
            pos = torch.arange(S)[None]
            pe = root.model.rotary_emb(h, pos)
            mask = torch.full((S, S), torch.finfo(h.dtype).min).triu(1)[None, None]
            return self.layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm, self.head = root.model.norm, root.lm_head

        def forward(self, h):
            return self.head(self.norm(h))

    return nn.ModuleList([Embed(), *(Layer(x) for x in root.model.layers), Head()])


@pytest.mark.slow
@pytest.mark.parametrize("n_layers", [4, 32])
def test_hf_e7_unit_window_costs_are_measured(tmp_path, n_layers):
    # covers: MODEL-I9
    # WOULD-FAIL-IF: the unit chain's host costs are not measured (c_glue, c_pass, F+B per layer against the
    # vanilla forward, a user-wrapped flat list (A0'') and the native layers) and written MEASURED, or, under
    # FACTORY_E7_BOUNDS=1, exceed F7 (c_glue <= 1 ms, c_pass <= 10 us, HF host <= 1.5x native)
    hfm = _hfm()
    hc = hfm.tiny_config("llama", num_hidden_layers=n_layers)
    cfg = _hf_cfg(hc)
    units = build_layers(cfg, 0)
    n = n_layers
    ids, tgt = _ids(cfg.vocab_size), _ids(cfg.vocab_size, seed=4)
    native = build_layers(
        GPTConfig(
            vocab_size=cfg.vocab_size,
            n_layer=n,
            n_head=4,
            d_model=64,
            d_ff=128,
            seq_len=16,
            norm="rmsnorm",
            mlp="swiglu",
            bias=False,
            pos="rope",
        ),
        0,
    )
    wrapped = _wrapped_llama(copy.deepcopy(units.root))

    def chain_fb(layers, with_ids: bool):
        h = ids
        for layer in layers:
            h = call_layer(layer, h, ids=ids) if with_ids else layer(h)
        loss_fn(h.float(), tgt).backward()

    with _one_thread():
        with torch.no_grad():
            ins = [ids]
            for u in range(n + 1):
                ins.append(call_layer(units[u], ins[-1], ids=ids))
            t_unit = [
                _median_s(lambda u=u: call_layer(units[u], ins[u], ids=ids)) for u in range(n + 2)
            ]
            t_vanilla_f = _median_s(lambda: units.vanilla(ids))
        before = [0] + [u - 1 for u in range(1, n + 1)] + [n]
        xs = torch.tensor(before[1 : n + 1], dtype=torch.float64)
        ys = torch.tensor(t_unit[1 : n + 1], dtype=torch.float64)
        c_pass = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
        c_glue = (sum(t_unit) - t_vanilla_f - c_pass * sum(before)) / (n + 2)
        fb = {
            "units": _median_s(lambda: chain_fb(units, True)),
            "vanilla": _median_s(lambda: loss_fn(units.vanilla(ids).float(), tgt).backward()),
            "wrapped": _median_s(lambda: chain_fb(wrapped, False)),
            "native": _median_s(lambda: chain_fb(native, False)),
        }
    per_layer = {k: v / n for k, v in fb.items()}
    data = {
        "n_layers": n,
        "c_glue_s": c_glue,
        "c_pass_s": c_pass,
        "unit_forward_s": t_unit,
        "vanilla_forward_s": t_vanilla_f,
        "fb_per_layer_s": per_layer,
        "hf_host_over_native": fb["units"] / fb["native"],
        "units_over_wrapped": fb["units"] / fb["wrapped"],
        "threads": 1,
        "reps": 20,
        "label": _machine_label(f"pytest tests/unit/test_finetune.py -k 'hf_e7 and {n}'"),
    }
    _report(tmp_path, f"hf-e7-window-{n}.json", data)
    assert all(v > 0 for v in t_unit) and data["label"].startswith("MEASURED")
    if E7_BOUNDS:
        assert c_glue <= 1e-3, data
        assert c_pass <= 10e-6, data
        assert data["hf_host_over_native"] <= 1.5, data


# The v4-style HF block of docs/research/adr-0008/scripts/probe3.py (byte-pinned research, never imported):
# compute-dtype router, fp32 softmax, topk, renormalization by .sum, a loop over the hit experts and
# index_add_. Kept here only as E7a's timing baseline.
class _V4Expert(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.w1, self.w3, self.w2 = (
            nn.Linear(d, f, bias=False),
            nn.Linear(d, f, bias=False),
            nn.Linear(f, d, bias=False),
        )

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class _V4Block(nn.Module):
    def __init__(self, d, f, E, k):
        super().__init__()
        self.E, self.k = E, k
        self.router = nn.Linear(d, E, bias=False)
        self.experts = nn.ModuleList(_V4Expert(d, f) for _ in range(E))

    def forward(self, xn):
        T, d = xn.shape
        logits = torch.nn.functional.linear(xn, self.router.weight)
        probs = torch.nn.functional.softmax(logits, dim=1, dtype=torch.float)
        w, sel = torch.topk(probs, self.k, dim=-1)
        w = (w / w.sum(dim=-1, keepdim=True)).to(xn.dtype)
        out = torch.zeros((T, d), dtype=xn.dtype)
        mask = torch.nn.functional.one_hot(sel, num_classes=self.E).permute(2, 1, 0)
        for e in range(self.E):
            idx, top_x = torch.where(mask[e])
            if top_x.numel() == 0:
                continue
            y = self.experts[e](xn[None, top_x].reshape(-1, d)) * w[top_x, idx, None]
            out.index_add_(0, top_x, y.to(xn.dtype))
        return out


@pytest.mark.slow
@pytest.mark.parametrize("E, k", [(16, 2), (32, 8)], ids=["e16k2", "e32k8"])
def test_moe_e7_block_costs_are_measured(tmp_path, E, k):
    # covers: MODEL-I8
    # WOULD-FAIL-IF: MoEMLP's pending W is not T*k*(2h+3f) values, a swapped HF block's routed experts hold any
    # pending W or its banks hold other than 4*E*3hf bytes, the E7a/E7b timings are not written MEASURED, or,
    # under FACTORY_E7_BOUNDS=1, either ratio exceeds F6 (1.15x)
    from transformers import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

    hfm = _hfm()
    h, f, T = 256, 384, 1024
    g = torch.Generator().manual_seed(50)
    x = torch.randn(4, T // 4, h, generator=g)
    G = torch.randn(4, T // 4, h, generator=g)
    torch.manual_seed(51)
    cfg = GPTConfig(
        vocab_size=8,
        n_layer=1,
        n_head=4,
        d_model=h,
        seq_len=4,
        moe_experts=E,
        moe_top_k=k,
        moe_d_ff=f,
    )
    nat = arch.MoEMLP(cfg, moe_layer=1)
    wgrad.convert(nat)
    v4 = _V4Block(h, f, E, k)
    mc = MixtralConfig(
        hidden_size=h, intermediate_size=f, num_local_experts=E, num_experts_per_tok=k
    )
    hf_blk = MixtralSparseMoeBlock(mc)
    with torch.no_grad():
        for p in hf_blk.parameters():
            p.normal_(0.0, 0.02, generator=g)
    swapped = copy.deepcopy(hf_blk)
    swapped.__class__ = hfm.MixtralSparseMoeBlockSwap
    swapped.gate.__class__ = hfm.MixtralTopKRouterSwap
    swapped.experts.__class__ = hfm.MixtralExpertsSwap
    swapped.moe_layer, swapped.moe_records = 0, []
    swapped.moe_cfg = hfm.MixtralSparseMoeBlockSwap.recipe(
        {
            "model_type": "mixtral",
            "hidden_size": h,
            "intermediate_size": f,
            "num_local_experts": E,
            "num_experts_per_tok": k,
        }
    )

    def fb(mod, flat: bool, capture: bool):
        xx = x.reshape(T, h) if flat else x
        xx = xx.clone().requires_grad_(True)
        out = mod(xx)
        loss = (out.reshape(G.shape) * G).sum()
        if not capture:
            loss.backward()
            return None
        owned = wgrad.owned_params(mod)
        others = [p for p in mod.parameters() if id(p) not in owned]
        with wgrad.capture() as store:
            torch.autograd.grad(loss, [xx] + others)
        entries = list(store)
        wgrad.flush(store)
        return entries

    # Byte counts (exact, every run).
    entries = fb(nat, False, True)
    seen, pend = set(), 0
    experts = {id(lin) for ex in nat.experts for lin in (ex.gate, ex.up, ex.down)}
    for m, fn in entries:
        if id(m) in experts:
            for t in fn.__defaults__:
                key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape))
                if key not in seen:
                    seen.add(key)
                    pend += t.numel() * t.element_size()
    assert pend == T * k * (2 * h + 3 * f) * 4
    hf_entries = fb(swapped, False, True)
    assert not any(any(m is s for s in swapped.experts.modules()) for m, _ in hf_entries)
    banks = swapped.experts.gate_up_proj.numel() + swapped.experts.down_proj.numel()
    assert banks * 4 == 4 * E * 3 * h * f
    with _one_thread():
        t = {
            "moemlp_fbw": _median_s(lambda: fb(nat, False, True)),
            "v4_fb": _median_s(lambda: fb(v4, True, False)),
            "swapped_fb": _median_s(lambda: fb(swapped, False, False)),
            "hf_own_fb": _median_s(lambda: fb(hf_blk, False, False)),
        }
        empty = torch.empty(0, h)
        w = torch.randn(f, h)
        t["zero_row_linears_3E"] = _median_s(
            lambda: [torch.nn.functional.linear(empty, w) for _ in range(3 * E)]
        )
    data = {
        "shape": {"h": h, "f": f, "E": E, "k": k, "T": T},
        "seconds": t,
        "e7a_moemlp_over_v4": t["moemlp_fbw"] / t["v4_fb"],
        "e7b_swapped_over_hf": t["swapped_fb"] / t["hf_own_fb"],
        "hf_experts_implementation": str(getattr(mc, "_experts_implementation", None)),
        "pending_w_bytes_moemlp": pend,
        "bank_grad_bytes_hf": 4 * E * 3 * h * f,
        "threads": 1,
        "reps": 20,
        "label": _machine_label(f"pytest tests/unit/test_finetune.py -k 'moe_e7 and e{E}k{k}'"),
    }
    _report(tmp_path, f"moe-e7-e{E}k{k}.json", data)
    if E7_BOUNDS:
        assert data["e7a_moemlp_over_v4"] <= 1.15, data
        assert data["e7b_swapped_over_hf"] <= 1.15, data


# --------------------------------------------------------- census and E8


def test_hf_family_census_cli_runs_a_three_type_subset(tmp_path, monkeypatch):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: the census CLI does not run each type through the gate in child processes (llama EXACT,
    # bert refused GATE-CAUSAL, gpt_oss refused MOE-NO-ROW), loses a type whose child dies, counts parameters
    # after the build (no SKIP above the cap), or reports a bit difference as anything but DIFFER
    hfm = _hfm()
    inp, out = tmp_path / "census.in", tmp_path / "census.json"
    inp.write_text(json.dumps({"census_types": ["llama", "bert", "gpt_oss"], "per_child": 2}))
    env = hfm.stage_env(dict(os.environ))
    res = subprocess.run(
        [sys.executable, "-m", "wantrain.model.hfm", "census", "--in", str(inp), "--out", str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    rep = json.loads(out.read_text())
    verdicts = {r["model_type"]: r["verdict"].split()[0] for r in rep["rows"]}
    assert verdicts == {"llama": "EXACT", "bert": "GATE-CAUSAL", "gpt_oss": "MOE-NO-ROW"}, rep[
        "rows"
    ]
    assert rep["counts"] == {"EXACT": 1, "GATE-CAUSAL": 1, "MOE-NO-ROW": 1}
    assert all(r["params"] and r["params"] < hfm.CENSUS_MAX_PARAMS for r in rep["rows"])
    # In process: the cap counts on meta and skips; a deliberately wrong chain is a DIFFER row.
    monkeypatch.setattr(hfm, "CENSUS_MAX_PARAMS", 1000)
    assert hfm._census_one("llama")["verdict"].startswith("SKIP")
    monkeypatch.setattr(hfm, "CENSUS_MAX_PARAMS", 20_000_000)
    monkeypatch.setattr(hfm, "_chain_buffers", lambda root: [b.zero_() for b in root.buffers()])
    assert hfm._census_one("llama")["verdict"].startswith("DIFFER")
    # A child that dies: each of its types reruns alone and one that dies alone is an ERROR row.
    monkeypatch.setattr(hfm, "_census_child", lambda types, d, tag: None)
    rows = hfm.census(["llama", "bert"], per_child=2)["rows"]
    assert [r["verdict"] for r in rows] == ["ERROR the census child died"] * 2


@pytest.mark.mps
@pytest.mark.parametrize("fam", ["llama", "gemma2"])
def test_hf_family_e8_gate_on_mps_is_recorded(fam):
    # covers: MODEL-I10
    # WOULD-FAIL-IF: the gate cannot run on an MPS slot device at all (E8 gates nothing: its verdict is
    # recorded, never a gate input)
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    hfm = _hfm()
    rep = hfm.gate(hfm.tiny_config(fam), GATE_RECIPE, "mps")
    print("E8", fam, json.dumps({k: rep[k] for k in ("cause", "first_difference", "device_class")}))
    assert rep["device_class"].startswith("mps:")
