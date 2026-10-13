"""Any HuggingFace decoder through transformers' own modules, windowed per
pipeline unit, behind an exact launch gate (ADR-0009 D1-D9; SPEC-model §2 "HF
modules", MODEL-I4, MODEL-I9, MODEL-I10).

The model is the pinned transformers' ForCausalLM, never a re-implementation.
A trace of a tiny model of the same config classifies every parameterized
module into units: prologue (unit 0, called before layer 0), layer i (unit
i+1), epilogue (unit n+1). Unit u runs HF's whole forward with a pre-hook that
injects its input at its entry, a pre-hook that stops the call at unit u+1's
entry, and every earlier layer (and, for u != 0, every prologue module)
replaced for the call by a NaN stub, so every glue op runs where HF puts it,
once. The gate compares the unit chain with the vanilla model bit for bit at
every cut, and the fingerprint pins the code that computed it.

transformers is imported inside functions only: the coordinator and the
planner import this module's constants without it.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from wantrain.model import arch, rollback

PIN = "5.19.0"

# SPEC-model §2's environment; the guard checks it and never sets it.
HF_ENV: dict[str, str] = {
    "USE_HUB_KERNELS": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
}
_LOADER_VARS = ("LD_PRELOAD", "LD_LIBRARY_PATH")

# Names the pinned transformers probes (is_*_available / _is_package_available)
# whose presence switches an algorithm (ADR-0009 D5, round6.md §2). Importable
# is refused unless the hf extra locks it (then RECORD-fingerprinted); the hf
# extra locks none of them at the pin. unsloth is refused outright.
PROBED: tuple[str, ...] = (
    "kernels",
    "flash_attn",
    "fla",
    "mamba_ssm",
    "causal_conv1d",
    "liger_kernel",
    "mambapy",
)
REFUSED_PACKAGES: tuple[str, ...] = ("unsloth",)
PROBED_LOCKED: frozenset[str] = frozenset()

# (HF class name, transformers version) -> this repository's swap subclass
# (ADR-0008 decision 11, ADR-0009 D8). A block row's subclass has a
# classmethod recipe(hf_config) -> dict of SPEC-model §2 moe_* fields (gamma
# 0); router and experts rows are admitted only under a block with a row.
# Rows are model-families'; the mechanism is here.
SUBSTITUTE: dict[tuple[str, str], type] = {}

# (platform tag, torch version) -> {distribution: canonical RECORD sha256} of
# torch and every distribution torch requires on that platform (MODEL-I10).
# The macOS arm64 entry is MEASURED from the installed wheels (2026-10-11,
# Apple Silicon, uv 0.11.14); Linux entries wait for the pin-bump PR (D-9).
TORCH_RECORDS: dict[tuple[str, str], dict[str, str]] = {
    ("cp312-cp312-macosx_14_0_arm64", "2.14.1"): {
        "torch": "2639152c56cbc4775d19190e0103237ab44a98db012efb2159fe1c2b0291dd76",
        "filelock": "269f697654f5e033e06e479feca518522706189bd79492e57bea14c3bfa845d5",
        "fsspec": "e0f2e780521334bc614e3201d8ffcd0f650bed4bad69bcb98522555731ae79fa",
        "jinja2": "c761f3e9c42a42a54938814881033dbf14baad07e7875ea924cce2738d32884f",
        "networkx": "c94f838a57e4c0527bb950ea80ab0de52f05b5011e13783b06ecc569f94e20b1",
        "setuptools": "58459ff6817baf53960b4663ea47956933a4a444c665e63aa2f0672aa263992b",
        "sympy": "4553dcfe75c73b186c488da944a01676bdaa279d5c64122c85b4991fa5467509",
        "typing-extensions": "73026746e3fbdee22ed45b0f3601c26942b6c7ab7572045a1e7945d1ee0c42e4",
    },
}

# Config keys naming the routed-expert count (D8); none: the model has no MoE.
EXPERT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts")

# The gate's budget (MODEL-I10): a tiny build over either is refused.
GATE_BUILD_BYTES = 64 * 2**20
GATE_BUILD_S = 30.0
EXPORT_SHARD_BYTES = 2 * 2**30
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


class HFRefused(ValueError):
    """A refusal of SPEC-model §2's rules; `cause` is the handout's token and
    the message names the module, tensor, field or file."""

    def __init__(self, cause: str, detail: str = ""):
        super().__init__(f"{cause}: {detail}" if detail else cause)
        self.cause = cause
        self.detail = detail


# ----------------------------------------------------------------- environment


def stage_env(base: dict) -> dict:
    """`base` plus HF_ENV, minus LD_PRELOAD, LD_LIBRARY_PATH and every DYLD_*."""
    env = {k: v for k, v in base.items() if k not in _LOADER_VARS and not k.startswith("DYLD_")}
    env.update(HF_ENV)
    return env


def interpreter_args(prefix_dir: str) -> list[str]:
    """In force at interpreter start, before any import; never assigned
    in-process (after `import torch` it would miss every torch module)."""
    return ["-X", f"pycache_prefix={prefix_dir}"]


def _guarded(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in ("torch", "transformers", "wantrain"))


def _prefix_violation() -> str | None:
    """The first loaded torch, transformers or wantrain module whose
    __cached__ is not under sys.pycache_prefix, or the missing prefix."""
    prefix = sys.pycache_prefix
    if not prefix:
        return "sys.pycache_prefix is unset (the process was started without -X pycache_prefix)"
    root = os.path.join(os.path.abspath(prefix), "")
    for name, mod in sorted(sys.modules.items()):
        if not _guarded(name):
            continue
        try:
            cached = vars(mod).get("__cached__")
        except TypeError:
            continue
        if isinstance(cached, str) and not os.path.abspath(cached).startswith(root):
            return f"module {name} caches its bytecode at {cached}, outside {prefix}"
    return None


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return True


def guard_environment() -> None:
    """Check SPEC-model §2's environment; never set it. HFRefused on the
    first violation, naming it."""
    for k, v in HF_ENV.items():
        if os.environ.get(k) != v:
            raise HFRefused("ENV-VAR", f"{k}={os.environ.get(k)!r}, required {v!r}")
    for k in sorted(os.environ):
        if k in _LOADER_VARS or k.startswith("DYLD_"):
            raise HFRefused("ENV-LOADER", f"{k} is set")
    bad = _prefix_violation()
    if bad:
        raise HFRefused("ENV-PREFIX", bad)
    for name in PROBED:
        if name not in PROBED_LOCKED and _importable(name):
            raise HFRefused("PROBED-PKG", f"{name} is importable and not locked by the hf extra")
    for name in REFUSED_PACKAGES:
        if _importable(name):
            raise HFRefused("UNSLOTH", f"{name} is importable")


# --------------------------------------------------------------------- config


def _transformers():
    import transformers

    if transformers.__version__ != PIN:
        raise HFRefused(
            "FP-RECORD", f"transformers {transformers.__version__} is installed; the pin is {PIN}"
        )
    transformers.logging.set_verbosity_error()
    return transformers


def _causal_map() -> dict:
    _transformers()
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    return dict(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)


def _config(hf_config: dict):
    """The pinned config class of hf_config["model_type"], built from it."""
    mt = hf_config.get("model_type")
    if not isinstance(mt, str) or mt not in _causal_map():
        raise HFRefused("MODEL-TYPE", f"model_type {mt!r} is not in the pinned causal-LM mapping")
    from transformers import CONFIG_MAPPING

    kw = {k: v for k, v in hf_config.items() if k not in ("model_type", "transformers_version")}
    return CONFIG_MAPPING[mt](**kw)


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rope_entries(c) -> list:
    rp = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None)
    if not isinstance(rp, dict):
        return []
    if "rope_type" in rp or "type" in rp:
        return [rp]
    return [v for v in rp.values() if isinstance(v, dict)]


def check_config(hf_config: dict, cfg) -> None:
    """SPEC-model §2's config refusals, each naming its cause and field.
    `cfg` is the run's GPTConfig (or None for the config alone)."""
    if not isinstance(hf_config, dict) or not hf_config.get("model_type"):
        raise HFRefused("MODEL-TYPE", "hf_config is not a config dict with model_type")
    if hf_config.get("auto_map"):
        raise HFRefused("AUTO-MAP", f"auto_map {sorted(hf_config['auto_map'])} names remote code")
    c = _config(hf_config)
    if getattr(c, "is_encoder_decoder", False):
        raise HFRefused("ENCODER-DECODER", f"model_type {c.model_type!r} is_encoder_decoder")
    for key in ("attn_implementation", "_attn_implementation"):
        impl = hf_config.get(key)
        if impl not in (None, "eager"):
            raise HFRefused("ATTN-IMPL", f"{key}={impl!r}; only the family's eager attention")
    for k, v in sorted(c.to_dict().items()):
        if ("dropout" in k or "pdrop" in k) and isinstance(v, (int, float)) and v > 0:
            raise HFRefused("DROPOUT", f"{k}={v} (the recipe must set it to 0)")
    seq = getattr(cfg, "seq_len", None)
    for rp in _rope_entries(c):
        rt = str(rp.get("rope_type", rp.get("type", "")))
        if "dynamic" in rt:
            raise HFRefused("ROPE-DYNAMIC", f"rope_type {rt!r} rebuilds inv_freq in forward")
        omp = rp.get("original_max_position_embeddings")
        if rt == "longrope" and seq is not None and omp is not None and omp < seq:
            raise HFRefused(
                "ROPE-LONGROPE",
                f"longrope with original_max_position_embeddings={omp} below seq_len={seq}",
            )
    if cfg is not None:
        if getattr(cfg, "lora_rank", 0):
            raise HFRefused("LORA", f"lora_rank={cfg.lora_rank}: LoRA is refused under arch 'hf'")
        if (
            getattr(cfg, "fp8_boundaries", False)
            or getattr(cfg, "boundary_dtype", "fp32") != "fp32"
        ):
            raise HFRefused(
                "BOUNDARY-DTYPE",
                f"boundary_dtype={cfg.boundary_dtype!r}; HF's residual dtype is fp32",
            )


def config_fields(hf_config: dict) -> dict:
    """The GPTConfig fields mirrored from the HF config."""
    c = _config(hf_config)
    return {
        "n_layer": int(c.num_hidden_layers),
        "d_model": int(c.hidden_size),
        "n_head": int(c.num_attention_heads),
        "vocab_size": int(c.vocab_size),
        "tie_embeddings": bool(getattr(c, "tie_word_embeddings", False)),
    }


# Dimension keys shrunk for a tiny config; every flag, rope type, layer type,
# softcap and MoE structure is kept (MODEL-I10).
_SHRINK = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "head_dim": 16,
    "vocab_size": 128,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    # multi-head latent attention (minicpm3, deepseek_v3)
    "kv_lora_rank": 16,
    "q_lora_rank": 32,
    "qk_rope_head_dim": 8,
    "qk_nope_head_dim": 8,
    "v_head_dim": 16,
    "intermediate_size_mlp": 128,
    # state-space mixers (falcon_h1, mamba): d_ssm = n_heads * d_head
    "mamba_d_ssm": 128,
    "mamba_n_heads": 16,
    "mamba_d_head": 8,
    "mamba_d_state": 16,
    "time_step_rank": 8,
}
_WINDOW_KEYS = ("sliding_window", "attention_chunk_size", "mamba_chunk_size")
_TINY_WINDOW = 8


def _shrunk(hf_config: dict, keep_depth: bool) -> dict:
    c = _config(hf_config)
    d = c.to_dict()
    amap = getattr(type(c), "attribute_map", {}) or {}

    def put(key, value):
        real = amap.get(key, key)
        if real in d and d[real] is not None:
            d[real] = value

    heads = getattr(c, "num_attention_heads", None)
    kv = getattr(c, "num_key_value_heads", None)
    for k, v in _SHRINK.items():
        put(k, v)
    if kv is not None and heads is not None:
        put("num_key_value_heads", 2 if kv < heads else _SHRINK["num_attention_heads"])
    # Derived MLA widths follow their parts (deepseek_v3: head_dim is the rope part).
    qr = getattr(c, "qk_rope_head_dim", None)
    if isinstance(qr, int):
        if getattr(c, "head_dim", None) == qr:
            d["head_dim"] = d["qk_rope_head_dim"]
        if "qk_head_dim" in d:
            d["qk_head_dim"] = d["qk_rope_head_dim"] + d.get("qk_nope_head_dim", 0)
    n = int(c.num_hidden_layers)
    lt = d.get("layer_types")
    if not keep_depth:
        n_tiny = min(n, 4)
        if isinstance(lt, list) and lt:
            last_new = max(lt.index(v) for v in set(lt))
            n_tiny = min(n, max(n_tiny, last_new + 1))
        put("num_hidden_layers", n_tiny)
        if isinstance(lt, list):
            d["layer_types"] = lt[:n_tiny]
    for key in EXPERT_KEYS:
        e = d.get(key)
        if isinstance(e, int) and e > 4:
            d[key] = max(4, int(d.get("num_experts_per_tok") or 1) * 2)
    for key in _WINDOW_KEYS:
        w = d.get(key)
        if isinstance(w, int) and w > _TINY_WINDOW:
            d[key] = _TINY_WINDOW
    # Token ids (padding_idx, bos, eos) must index the shrunk vocabulary.
    V = d.get(amap.get("vocab_size", "vocab_size"))
    if isinstance(V, int):
        for k, v in list(d.items()):
            if not k.endswith("token_id"):
                continue
            if isinstance(v, int) and not isinstance(v, bool) and v >= V:
                d[k] = V - 1
            elif isinstance(v, list):
                d[k] = [min(x, V - 1) if isinstance(x, int) else x for x in v]
    d["model_type"] = c.model_type
    d.pop("transformers_version", None)
    return d


def tiny_config(model_type: str, **overrides) -> dict:
    """A tiny config dict of `model_type`'s pinned config class: its defaults
    with the dimension keys shrunk, then `overrides`."""
    if model_type not in _causal_map():
        raise HFRefused("MODEL-TYPE", f"model_type {model_type!r} is not in the pinned mapping")
    from transformers import CONFIG_MAPPING

    cls = CONFIG_MAPPING[model_type]
    d = _shrunk(cls().to_dict() | {"model_type": model_type}, False)
    amap = getattr(cls, "attribute_map", {}) or {}
    d.update({amap.get(k, k): v for k, v in overrides.items()})  # an alias sets its real key
    return d


def _gate_seq(hf_config: dict) -> int:
    """S above every window and chunk of the tiny config."""
    ws = [hf_config.get(k) for k in _WINDOW_KEYS]
    return max([16] + [w + 8 for w in ws if isinstance(w, int)])


# ------------------------------------------------------------------- skeleton


@contextlib.contextmanager
def _params_on_meta():
    """Every parameter registered during the block lands on meta; buffers
    stay where their constructors put them (the CPU)."""
    orig = nn.Module.register_parameter

    def reg(self, name, p):
        if p is not None and p.device.type != "meta":
            p = nn.Parameter(p.to("meta"), requires_grad=p.requires_grad)
        return orig(self, name, p)

    nn.Module.register_parameter = reg  # type: ignore[method-assign]
    try:
        yield
    finally:
        nn.Module.register_parameter = orig  # type: ignore[method-assign]


def _model_class(c):
    transformers = _transformers()
    return getattr(transformers, _causal_map()[c.model_type])


def _eager(hf_config: dict):
    """The model's config: eager attention and no KV cache, pinned here for
    every build. Every call already passes use_cache=False (SPEC-model §2), so
    pinning the attribute changes no bit; a stock config.json says true, and
    the stage and oracle start asserts read the attribute."""
    c = _config(hf_config)
    c._attn_implementation = "eager"
    c.use_cache = False
    return c


def _construct(hf_config: dict, dtype, meta: bool):
    c = _eager(hf_config)
    cls = _model_class(c)
    if meta:
        with _params_on_meta():
            return cls._from_config(c, dtype=dtype)
    return cls._from_config(c, dtype=dtype)


def _check_forwards(model) -> None:
    for name, m in model.named_modules():
        if "forward" in m.__dict__:
            raise HFRefused("FOREIGN-FORWARD", f"module {name or '<root>'} has an instance forward")


def _layer_list(model) -> tuple[str, nn.ModuleList]:
    """The unique ModuleList of num_hidden_layers modules, equal to the class's
    base_model_pp_plan layer path where it declares one."""
    n = int(model.config.num_hidden_layers)
    cands = [
        (k, v) for k, v in model.named_modules() if isinstance(v, nn.ModuleList) and len(v) == n
    ]
    plan = getattr(model.config, "base_model_pp_plan", None) or {}
    prefix = getattr(model, "base_model_prefix", "")
    planned = {f"{prefix}.{k}" if prefix else k for k in plan}
    if planned:
        pc = [(k, v) for k, v in cands if k in planned]
        if len(pc) == 1:
            return pc[0]
    if len(cands) != 1:
        raise HFRefused(
            "TRACE-UNPLACED",
            f"{len(cands)} ModuleLists of num_hidden_layers={n}: {[k for k, _ in cands]}",
        )
    return cands[0]


def _finish(model, hf_config: dict, cfg=None):
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise HFRefused("ATTN-IMPL", f"attention is {model.config._attn_implementation!r}")
    model.train()
    _check_forwards(model)
    _swap(model, hf_config, cfg)
    return model


def skeleton(hf_config: dict, dtype=torch.float32):
    """The whole ForCausalLM with parameters on meta and buffers on the CPU,
    eager attention, training mode, SUBSTITUTE swapped. Draws nothing."""
    s0 = torch.random.get_rng_state()
    m = _construct(hf_config, dtype, meta=True)
    if not torch.equal(s0, torch.random.get_rng_state()):
        raise HFRefused("SKELETON-DRAW", f"building {type(m).__name__} drew from the generator")
    return _finish(m, hf_config)


def _full_build(hf_config: dict, seed: int, cfg=None, swap: bool = True):
    """The seeded full HF build on the CPU (decision 7)."""
    torch.manual_seed(seed)
    m = _construct(hf_config, torch.float32, meta=False)
    if not swap:
        if getattr(m.config, "_attn_implementation", None) != "eager":
            raise HFRefused("ATTN-IMPL", f"attention is {m.config._attn_implementation!r}")
        _check_forwards(m.train())
        return m
    return _finish(m, hf_config, cfg)


# ---------------------------------------------------------------- MoE swap


def _row(cls) -> type | None:
    if rollback.rolled_back_to("moe-from-scratch"):
        return None
    return SUBSTITUTE.get((cls.__name__, PIN))


def _layer_index(model, module) -> int | None:
    _, L = _layer_list(model)
    for i, layer in enumerate(L):
        if any(m is module for m in layer.modules()):
            return i
    return None


def _swap(model, hf_config: dict, cfg=None) -> list[str]:
    """Class-swap every module whose (class, PIN) has a SUBSTITUTE row (ADR-0008
    decision 11): blocks first, each with moe_layer = its decoder-layer index
    (BA-462) and its row's recipe (lbl and z coefficients from the run's
    GPTConfig; router_aux_loss_coef is never mapped); a router or experts row
    only under a swapped block, else MOE-NO-ROW. -> the swapped block names."""
    blocks = []
    for name, m in model.named_modules():
        sub = _row(type(m))
        if sub is None or not hasattr(sub, "recipe"):
            continue
        idx = _layer_index(model, m)
        if idx is None:
            raise HFRefused(
                "MOE-NO-ROW", f"block {name} ({type(m).__name__}) is in no decoder layer"
            )
        recipe = dict(sub.recipe(hf_config))
        recipe["moe_bias_rate"] = 0.0
        for k in ("moe_lbl_coef", "moe_lbl_scope", "moe_z_coef"):
            if cfg is not None:
                recipe[k] = getattr(cfg, k)
        m.__class__ = sub
        m.moe_layer = idx
        m.moe_cfg = recipe
        m.moe_records = []
        blocks.append(name)
    swapped = {id(model.get_submodule(b)) for b in blocks}
    for name, m in model.named_modules():
        sub = _row(type(m))
        if sub is None or hasattr(sub, "recipe"):
            continue
        parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
        if id(parent) not in swapped:
            raise HFRefused(
                "MOE-NO-ROW",
                f"{name} ({type(m).__name__}) has a row but its block {type(parent).__name__} has none",
            )
        m.__class__ = sub
    return blocks


def _set_moe_coefs(model, cfg) -> None:
    for m in model.modules():
        if isinstance(getattr(m, "moe_cfg", None), dict):
            for k in ("moe_lbl_coef", "moe_lbl_scope", "moe_z_coef"):
                m.moe_cfg[k] = getattr(cfg, k)


# ------------------------------------------------- SUBSTITUTE rows (5.19.0)
#
# ADR-0008 decisions 9 and 11 (ADR-0009 D8): the sparse-MoE block of each
# family is class-swapped to a block whose forward is arch.moe_forward over the
# block's own children; its router (`gate`) and experts children are swapped
# too, so the block calls each once as a module and the unit trace sees every
# parameter-holding module. Names, parameters and layouts stay HF's; HF's own
# router and experts forwards never run.


def _cast_once(x: torch.Tensor) -> torch.Tensor:
    """x in the autocast dtype, cast once, so every Linear reading it records
    the same X (MoEExpert's rule; SPEC-model §2's pending W)."""
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        return x.to(torch.get_autocast_dtype(dev))
    return x


class _MoEView:
    """moe_forward's view of a swapped HF block: its gate (router) and experts
    children, the shared expert (its input cast once, as MoEMLP's) and gate."""

    sel_bias = None

    def __init__(self, block: nn.Module):
        self.router = block.gate
        self.experts = block.experts
        se = getattr(block, "shared_expert", None)
        self.shared_expert = None if se is None else (lambda x: se(_cast_once(x)))
        self.shared_expert_gate = getattr(block, "shared_expert_gate", None)
        self.moe_layer = block.moe_layer
        self.moe_records = block.moe_records


class _SwapBlock(nn.Module):
    """A swapped sparse-MoE block: one tensor in, one tensor out, the exact
    MoE function of SPEC-model §2 (arch.moe_forward)."""

    moe_layer: int
    moe_cfg: dict
    moe_records: list

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return arch.moe_forward(self.moe_cfg, _MoEView(self), hidden_states)

    @classmethod
    def _base(cls, c, E: int, f: int) -> dict:
        """The fields every row shares; refusals name the field."""
        for key in ("output_router_logits",):
            if getattr(c, key, False):
                raise HFRefused(
                    "MOE-NO-ROW",
                    f"{key}=True: HF's router-logit aux loss is not moe_forward's "
                    f"(its lbl and z come from the run's moe_lbl_coef and moe_z_coef)",
                )
        jitter = getattr(c, "router_jitter_noise", 0.0) or 0.0
        if jitter > 0:
            raise HFRefused(
                "MOE-NO-ROW",
                f"router_jitter_noise={jitter}: a block attribute moe_forward does not read",
            )
        act = getattr(c, "hidden_act", "silu")
        if act != "silu":
            raise HFRefused("MOE-NO-ROW", f"hidden_act={act!r}: the row's experts are SwiGLU")
        return {
            "moe_experts": int(E),
            "moe_top_k": int(c.num_experts_per_tok),
            "moe_d_ff": int(f),
            "moe_score": "softmax",
            "moe_select": "score",
            "moe_router_dtype": "compute",
            "moe_norm_eps": 0.0,
            "moe_scale": 1.0,
            "moe_bias_rate": 0.0,
        }


class _SwapRouter(nn.Module):
    """A swapped top-k router: the router GEMM only (logits); selection and
    weights are moe_forward's."""

    weight: nn.Parameter  # HF's (E, h) router weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)


class _SwapExperts(nn.Module):
    """Swapped fused experts: the dispatched rows (ascending expert, then
    token) and their per-expert counts -> each expert's SwiGLU output, in
    expert order. Each bank is split by one unbind(0), each expert's (2f, h)
    view by one chunk(2) into its gate and up rows, and each runs one F.linear
    on exactly its rows (0 rows included): MoEMLP's gate, up and down. dW of
    the banks is computed in B."""

    gate_up_proj: nn.Parameter  # (E, 2f, h), gate rows first
    down_proj: nn.Parameter  # (E, h, f)

    def forward(self, xd: torch.Tensor, counts: list) -> torch.Tensor:
        xd = _cast_once(xd)
        gate_up, down = self.gate_up_proj.unbind(0), self.down_proj.unbind(0)
        outs = []
        for j, rows in enumerate(torch.split(xd, counts)):
            wg, wu = gate_up[j].chunk(2, dim=0)
            if _ROWWISE[0]:
                outs += [_swiglu(r, wg, wu, down[j]) for r in rows.split(1)]
            else:
                outs.append(_swiglu(rows, wg, wu, down[j]))
        return torch.cat(outs) if outs else xd[:0]


def _swiglu(rows, wg, wu, wd):
    g, u = F.linear(rows, wg), F.linear(rows, wu)
    return F.linear(F.silu(g) * u, wd)


# The gate's causality probe only (never a training path): each expert row in
# its own GEMM. A CPU GEMM's row bits depend on its row count (MEASURED: torch
# 2.14.1, Apple M-series, 6 of 39 counts), and an exact block's per-expert
# counts move with a later token's routing, so the probe would see the GEMM's
# blocking, not a token reading the future; with one row per GEMM it sees only
# the model's own dependence (its attention).
_ROWWISE = [False]


@contextlib.contextmanager
def _rowwise_experts():
    _ROWWISE[0] = True
    try:
        yield
    finally:
        _ROWWISE[0] = False


class MixtralSparseMoeBlockSwap(_SwapBlock):
    @classmethod
    def recipe(cls, hf_config: dict) -> dict:
        c = _config(hf_config)
        r = cls._base(c, c.num_local_experts, c.intermediate_size)
        r["moe_norm"] = True
        return r


class Qwen2MoeSparseMoeBlockSwap(_SwapBlock):
    @classmethod
    def recipe(cls, hf_config: dict) -> dict:
        c = _config(hf_config)
        r = cls._base(c, c.num_experts, c.moe_intermediate_size)
        r["moe_norm"] = bool(c.norm_topk_prob)
        r["moe_shared_d_ff"] = int(c.shared_expert_intermediate_size)
        r["moe_shared_gate"] = "sigmoid"
        return r


class Qwen3MoeSparseMoeBlockSwap(_SwapBlock):
    @classmethod
    def recipe(cls, hf_config: dict) -> dict:
        c = _config(hf_config)
        r = cls._base(c, c.num_experts, c.moe_intermediate_size)
        r["moe_norm"] = bool(c.norm_topk_prob)
        return r


class MixtralTopKRouterSwap(_SwapRouter):
    pass


class Qwen2MoeTopKRouterSwap(_SwapRouter):
    pass


class Qwen3MoeTopKRouterSwap(_SwapRouter):
    pass


class MixtralExpertsSwap(_SwapExperts):
    pass


class Qwen2MoeExpertsSwap(_SwapExperts):
    pass


class Qwen3MoeExpertsSwap(_SwapExperts):
    pass


SUBSTITUTE.update(
    {
        (sub.__name__.removesuffix("Swap"), PIN): sub
        for sub in (
            MixtralSparseMoeBlockSwap,
            Qwen2MoeSparseMoeBlockSwap,
            Qwen3MoeSparseMoeBlockSwap,
            MixtralTopKRouterSwap,
            Qwen2MoeTopKRouterSwap,
            Qwen3MoeTopKRouterSwap,
            MixtralExpertsSwap,
            Qwen2MoeExpertsSwap,
            Qwen3MoeExpertsSwap,
        )
    }
)


# --------------------------------------------------------------------- trace


@dataclass
class UnitMap:
    """The unit trace of a model (ADR-0009 D2), by module names."""

    layers: str  # the layer list's path
    n: int
    prologue: list[str]  # parameterized modules called before layer 0, in call order
    epilogue: list[str]  # ... called after layer n-1; epilogue[0] is unit n+1's entry
    layer_out: list  # per layer: None (one tensor) or a tuple structure
    prologue_out: dict  # name -> [trailing size or "hidden_size", dtype name]
    exclusions: list[str]  # Linears whose parameters an op outside their forward reads
    tied: list[str] | None  # alias names of the tied table, prologue first
    moe_blocks: dict = field(default_factory=dict)  # decoder layer -> sparse-MoE block name
    trace_shape: tuple = (0, 0)


class _Exit(Exception):
    def __init__(self, h):
        super().__init__("unit exit")
        self.h = h


_SELECT_FUNCS = {"topk", "sort", "argsort", "argmax", "max", "multinomial"}


def _seq(tag) -> int:
    return tag[0]


_PY_CONVERT = {"item", "tolist", "__bool__", "__index__", "__int__"}


def _trace_shape(c) -> tuple[int, int]:
    dims = {v for v in c.to_dict().values() if isinstance(v, int) and not isinstance(v, bool)}
    bs = next(b for b in range(3, 64, 2) if b not in dims)
    S = next(s for s in range(11, 512) if s not in dims and s != bs)
    return bs, S


def _dtype_name(dt) -> str:
    return str(dt).removeprefix("torch.")


def _structure(out, bs: int, S: int, where: str):
    """A layer output's structure: None for one floating tensor, else a list
    [("res",), ("t", dtype, shape template) | ("v", value), ...]."""
    if isinstance(out, torch.Tensor):
        if not out.is_floating_point():
            raise HFRefused("TRACE-NONFLOAT", f"{where} returns a {out.dtype} tensor")
        return None
    if not isinstance(out, tuple) or not out or not isinstance(out[0], torch.Tensor):
        raise HFRefused("TRACE-NONFLOAT", f"{where} returns {type(out).__name__}, not a tensor")
    st: list = [("res",)]
    for x in out[1:]:
        if isinstance(x, torch.Tensor):
            if not x.is_floating_point():
                raise HFRefused("TRACE-NONFLOAT", f"{where} returns a {x.dtype} tensor")
            shape = tuple("b" if d == bs else ("t" if d == S else d) for d in x.shape)
            st.append(("t", _dtype_name(x.dtype), shape))
        else:
            st.append(("v", x))
    return st


def _tensors(x) -> list:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for v in x for t in _tensors(v)]
    if isinstance(x, dict):
        return [t for v in x.values() for t in _tensors(v)]
    return []


def trace(model) -> UnitMap:
    """The unit trace (ADR-0009 D2, D4): one training-mode forward of `model`
    (real parameters; its SUBSTITUTE swap applied) on ids drawn independently
    of the gate's, at a batch size and length equal to no config dimension,
    inside fork_rng. Every parameterized module must be called (in a layer, a
    module the forward never calls is in no unit: TRACE-UNPLACED, which is
    what holds a swapped block to calling its router and experts as modules).
    Its moe_blocks are D8's classification, which only an unswapped model
    shows (see _classify). Refusals name the module."""
    return _trace(model, strict=True)


def _classify(model) -> dict:
    """D8: the sparse-MoE blocks of an unswapped model, by routing structure
    on one forward (the module-call rules are the unit trace's)."""
    return _trace(model, strict=False).moe_blocks


def _trace(model, strict: bool) -> UnitMap:
    lname, L = _layer_list(model)
    n = len(L)
    c = model.config
    bs, S = _trace_shape(c)
    names = {id(m): k for k, m in model.named_modules()}
    layer_of: dict = {}
    for i, layer in enumerate(L):
        for m in layer.modules():
            if layer_of.setdefault(id(m), i) != i:
                raise HFRefused("TRACE-SHARED", f"module {names[id(m)]} is in two layers")
    L_index = {id(layer): i for i, layer in enumerate(L)}
    own = {id(m) for m in model.modules() if next(m.parameters(recurse=False), None) is not None}
    E = next((int(getattr(c, k)) for k in EXPERT_KEYS if isinstance(getattr(c, k, None), int)), 0)
    elist = {
        id(x)
        for m in model.modules()
        if isinstance(m, nn.ModuleList) and E and len(m) == E
        for x in m
    }
    st: dict = {"phase": "pro", "stack": [], "layer_calls": [], "choice": None, "seq": 0}
    calls: dict = {}
    order: list = []
    outs: dict = {}
    pro_out: dict = {}
    direct: set = set()
    tags: dict = {}
    keep: list = []
    uses: list = []
    lin: dict = {}  # parameter -> its module (a Linear's wins: the exclusion set is of Linears)
    for linear_first in (False, True):
        for m in model.modules():
            if isinstance(m, nn.Linear) == linear_first:
                for p in m.parameters(recurse=False):
                    lin[id(p)] = m

    def layer_ctx():
        for m in reversed(st["stack"]):
            if id(m) in L_index:
                return L_index[id(m)]
        return None

    def pre(m, args):
        if id(m) in L_index:
            st["layer_calls"].append(L_index[id(m)])
            st["phase"] = "mid"
        if id(m) in own:
            if id(m) not in calls:
                order.append(id(m))
            calls.setdefault(id(m), []).append((st["phase"], layer_ctx()))
        if id(m) in elist and st["choice"] is not None:
            uses.append((st["choice"], list(st["stack"]) + [m]))
        st["stack"].append(m)

    def post(m, args, out):
        if st["stack"] and st["stack"][-1] is m:
            st["stack"].pop()
        if id(m) in L_index:
            i = L_index[id(m)]
            outs.setdefault(i, _structure(out, bs, S, f"layer {i} ({names[id(m)]})"))
            if i == n - 1:
                st["phase"] = "epi"
        elif (
            id(m) in own and id(m) not in layer_of and st["phase"] == "pro" and id(m) not in pro_out
        ):
            t = out if isinstance(out, torch.Tensor) else None
            if t is None:
                raise HFRefused(
                    "TRACE-UNPLACED", f"prologue module {names[id(m)]} returns no tensor"
                )
            trail = int(t.shape[-1])
            pro_out[id(m)] = [
                "hidden_size" if trail == int(c.hidden_size) else trail,
                _dtype_name(t.dtype),
            ]

    from torch.overrides import TorchFunctionMode
    from torch.utils._pytree import tree_flatten

    class Spy(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            flat = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
            top = st["stack"][-1] if st["stack"] else None
            for t in flat:
                owner = lin.get(id(t))
                if owner is not None and owner is not top:
                    direct.add(names[id(owner)])
            fname = getattr(func, "__name__", "")
            # Tags are (selection number, its module stack); an op's inputs
            # pass on the newest selection's tag (a float input may still carry
            # an earlier layer's).
            newest = max((tags[id(t)] for t in flat if id(t) in tags), default=None, key=_seq)
            if E and fname in _PY_CONVERT and newest is not None:
                st["choice"] = newest[1]
            if E and flat:
                # A tagged value meets a parameter leading with E in one op:
                # indices or offsets choosing rows of the expert bank (an
                # index, a gather, a grouped GEMM), or routing scores weighting
                # every expert's rows of it (llama4's dense dispatch, a bmm).
                banks = [
                    t
                    for t in flat
                    if isinstance(t._base if t._base is not None else t, nn.Parameter)
                    and t.dim()
                    and t.shape[0] == E
                ]
                if banks and newest is not None:
                    uses.append((newest[1], list(st["stack"])))
            res = func(*args, **kwargs)
            if not E:
                return res
            src = newest
            outs_t = _tensors(res)  # structseqs (topk, sort, max) are tuples, not pytree nodes
            if fname in _SELECT_FUNCS and flat:
                pos = 2 if fname == "topk" else 1  # topk(x, k, dim); sort/max(x, dim)
                dim = kwargs.get(
                    "dim", args[pos] if len(args) > pos and isinstance(args[pos], int) else -1
                )
                if flat[0].dim() and flat[0].shape[dim] == E:
                    st["seq"] += 1
                    src = (st["seq"], list(st["stack"]))
                    outs_t = [t for t in outs_t if not t.is_floating_point()] or outs_t
            if src is not None:
                for t in outs_t:
                    tags[id(t)] = src
                    keep.append(t)
            return res

    lin_by_name = {k: m for k, m in model.named_modules() if isinstance(m, nn.Linear)}
    g = torch.Generator().manual_seed(0x5EED)
    ids = torch.randint(0, int(c.vocab_size), (bs, S), generator=g)
    model.train()
    from torch.nn.modules import module as tmod

    with torch.random.fork_rng(devices=[]):
        s0 = torch.random.get_rng_state()
        h1 = tmod.register_module_forward_pre_hook(pre)
        h2 = tmod.register_module_forward_hook(post)
        try:
            with torch.no_grad(), Spy():
                model(input_ids=ids, use_cache=False)
        finally:
            h1.remove()
            h2.remove()
        if not torch.equal(s0, torch.random.get_rng_state()):
            raise HFRefused(
                "TRACE-RNG", f"{type(model).__name__}'s forward draws from the generator"
            )

    lc = st["layer_calls"]
    for i in range(n):
        if lc.count(i) > 1:
            raise HFRefused("TRACE-TWICE", f"layer {i} ({lname}.{i}) is called {lc.count(i)} times")
    if lc != list(range(n)):
        raise HFRefused("TRACE-UNPLACED", f"the layer list {lname} is not called in order: {lc}")
    if strict:
        for k, m in model.named_modules():
            # A module whose parameters an op outside its forward reads
            # (mamba's dt_proj and conv1d) is placed by its layer (a Linear
            # is kept out of SplitLinear: the exclusion set); a swapped
            # router or experts module never is: the seam calls it.
            swapped = any(type(m) is sub for sub in SUBSTITUTE.values())
            read_elsewhere = k in direct and not swapped
            if id(m) in own and id(m) in layer_of and id(m) not in calls and not read_elsewhere:
                raise HFRefused(
                    "TRACE-UNPLACED",
                    f"module {k} holds parameters but layer {layer_of[id(m)]}'s forward never "
                    f"calls it",
                )
    prologue, epilogue = [], []
    for mid in order:
        nm = names[mid]
        phases = {ph for ph, _ in calls[mid]}
        ctxs = {cx for _, cx in calls[mid]}
        if mid in layer_of:
            if ctxs != {layer_of[mid]}:
                raise HFRefused(
                    "TRACE-BETWEEN", f"module {nm} is called from layers {sorted(map(str, ctxs))}"
                )
            continue
        if phases == {"pro"}:
            prologue.append(nm)
        elif phases == {"epi"}:
            epilogue.append(nm)
        elif "mid" in phases:
            raise HFRefused("TRACE-BETWEEN", f"module {nm} is called between or from layers")
        else:
            raise HFRefused("TRACE-SHARED", f"module {nm} is called before and after the layers")
    if not epilogue:
        raise HFRefused("TRACE-UNPLACED", "no parameterized module is called after the last layer")
    unit_of_mod = {}
    for nm in prologue:
        unit_of_mod[nm] = 0
    for nm in epilogue:
        unit_of_mod[nm] = n + 1
    groups: dict = {}
    for pname, p in model.named_parameters(remove_duplicate=False):
        mod, _, _ = pname.rpartition(".")
        m = model.get_submodule(mod) if mod else model
        if id(m) in layer_of:
            u = layer_of[id(m)] + 1
        else:
            u = unit_of_mod.get(mod)
        groups.setdefault(id(p), []).append((pname, u))
    tied = None
    for grp in groups.values():
        us = {u for _, u in grp if u is not None}
        if not us:
            raise HFRefused("TRACE-UNPLACED", f"parameter {grp[0][0]} is in no unit")
        if len(us) > 1:
            if us == {0, n + 1} and getattr(c, "tie_word_embeddings", False) and tied is None:
                tied = [k for k, u in sorted(grp, key=lambda x: x[1])]
            else:
                raise HFRefused(
                    "TRACE-SHARED", f"parameter {grp[0][0]} is shared by units {sorted(us)}"
                )
    excl = sorted(k for k in direct if k in lin_by_name)
    moe_blocks: dict = {}
    depth: dict = {}
    for sel_stack, use_stack in uses:
        common, d = None, 0
        for a, b in zip(sel_stack, use_stack):
            if a is not b:
                break
            common, d = a, d + 1
        # The block is the innermost common module; a shallower one (a later
        # op of the same decoder layer reading a tagged value) never wins.
        if common is not None and id(common) in layer_of:
            i = layer_of[id(common)]
            if d > depth.get(i, 0):
                moe_blocks[i], depth[i] = names[id(common)], d
    return UnitMap(
        layers=lname,
        n=n,
        prologue=prologue,
        epilogue=epilogue,
        layer_out=[outs.get(i) for i in range(n)],
        prologue_out={names[k]: v for k, v in pro_out.items()},
        exclusions=excl,
        tied=tied,
        moe_blocks=moe_blocks,
        trace_shape=(bs, S),
    )


_MAPS: dict = {}


def _traced(model, hf_config: dict, cfg=None) -> UnitMap:
    """D8 on the unswapped model, then its SUBSTITUTE swap, then the unit
    trace of the swapped model, carrying D8's blocks. Rows missing for a
    classified block leave it unswapped (units() refuses it MOE-NO-ROW)."""
    c = model.config
    if not any(isinstance(getattr(c, k, None), int) for k in EXPERT_KEYS):
        _swap(model, hf_config, cfg)  # no expert count: D8 finds nothing to swap
        return trace(model)
    pre = _trace(model, strict=False)
    _swap(model, hf_config, cfg)
    if any(not hasattr(type(model.get_submodule(b)), "recipe") for b in pre.moe_blocks.values()):
        return pre
    umap = trace(model)
    umap.moe_blocks = pre.moe_blocks
    return umap


def unit_map(hf_config: dict) -> UnitMap:
    """The trace of a tiny model of hf_config at full depth (same module
    names), built and traced under fork_rng; cached per config and per the
    SUBSTITUTE rows and rollbacks in force."""
    rows = sorted((k, id(v)) for k, v in SUBSTITUTE.items())
    key = _sha(_canon([hf_config, rows, sorted(rollback.ROLLED_BACK)]))
    if key not in _MAPS:
        with torch.random.fork_rng(devices=[]):
            shrunk = _shrunk(hf_config, keep_depth=True)
            tiny = _full_build(shrunk, 0, swap=False)
            _MAPS[key] = _traced(tiny, shrunk)
    return _MAPS[key]


# --------------------------------------------------------------------- units

_NAN = float("nan")
_NAN_SCALARS: dict = {}


def _nan_view(shape, dtype, device) -> torch.Tensor:
    """A stride-0 NaN view of `shape`: no byte allocated per call (one cached
    0-dim NaN per dtype and device; ADR-0009 D2's stubs write 0 B). An
    in-place write into it raises, or (fill_, zero_, masked_fill_ and
    index_fill_ write a stride-0 view) moves the scalar's version counter,
    which _stub_written() turns into STUB-WRITE; a written scalar is never
    reused."""
    key = (dtype, str(device))
    s = _NAN_SCALARS.get(key)
    if s is None or s._version:
        s = _NAN_SCALARS[key] = torch.full((), _NAN, dtype=dtype, device=device)
    return s.expand(shape)


def _stub_written() -> bool:
    """Whether a cached stub NaN was written in place since it was made (a
    host-side version check: no device read). Written scalars are dropped."""
    bad = [k for k, s in _NAN_SCALARS.items() if s._version]
    for k in bad:
        del _NAN_SCALARS[k]
    return bool(bad)


class _LayerStub(nn.Module):
    """A layer before the unit: stride-0 NaN views in the layer's traced
    structure (no byte written; an in-place write raises: STUB-WRITE)."""

    def __init__(self, struct, trace_shape):
        super().__init__()
        self.struct, self.trace_shape = struct, trace_shape

    def forward(self, *args, **kwargs):
        res = args[0] if args else kwargs["hidden_states"]
        h = _nan_view(res.shape, res.dtype, res.device)
        if self.struct is None:
            return h
        b, t = res.shape[0], res.shape[1]
        out: list = [h]
        for el in self.struct[1:]:
            if el[0] == "v":
                out.append(el[1])
                continue
            shape = tuple(b if d == "b" else (t if d == "t" else d) for d in el[2])
            out.append(_nan_view(shape, getattr(torch, el[1]), res.device))
        return tuple(out)


class _PrologueStub(nn.Module):
    """A prologue module on a unit u != 0: a fresh NaN tensor in its traced
    dtype and trailing size, shaped from its first argument."""

    def __init__(self, trailing: int, dtype):
        super().__init__()
        self.trailing, self.dt = trailing, dtype

    def forward(self, *args, **kwargs):
        t = args[0] if args else next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        shape = tuple(t.shape) if not t.is_floating_point() else tuple(t.shape[:-1])
        return torch.full(shape + (self.trailing,), _NAN, dtype=self.dt, device=t.device)


def _exit_unit(u: int) -> int:
    """The unit whose entry stops unit u's call (u + 1)."""
    return u + 1


def _buf_sha(t: torch.Tensor) -> str:
    return _sha(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


class _Runtime:
    """The shared tree, its unit map and the window mechanics."""

    def __init__(self, root, umap: UnitMap, compute_dtype: str):
        self.root, self.map, self.compute = root, umap, compute_dtype
        self.L = root.get_submodule(umap.layers)
        self.dev_bufs: dict = {}
        hidden = int(root.config.hidden_size)
        self.pro_stub = {
            name: (hidden if spec[0] == "hidden_size" else int(spec[0]), getattr(torch, spec[1]))
            for name, spec in umap.prologue_out.items()
        }
        ex = {root.get_submodule(k) for k in umap.exclusions}
        if umap.tied:
            for k in umap.tied:
                mod = root.get_submodule(k.rpartition(".")[0])
                if isinstance(mod, nn.Linear):
                    ex.add(mod)
        self.exclusions = frozenset(ex)

    def entry(self, u: int):
        n = self.map.n
        return self.L[u - 1] if 1 <= u <= n else self.root.get_submodule(self.map.epilogue[0])

    def _autocast(self, dev):
        from wantrain.model import precision

        dt = precision.spec(self.compute)["torch_dtype"]
        return torch.autocast(device_type=dev.type, dtype=dt, enabled=self.compute != "fp32")

    def _check_train(self):
        for name, m in self.root.named_modules():
            if not m.training:
                raise HFRefused("EVAL-MODE", f"module {name or '<root>'} is in eval mode")

    def _swap_buffers(self, dev, restore: list):
        """BA-433: the shared tree's buffers on a non-CPU unit device, copied
        once from the CPU and checked by sha256."""
        if dev.type == "cpu":
            return
        cache = self.dev_bufs.setdefault(str(dev), {})
        for mod in self.root.modules():
            for bn, buf in list(mod._buffers.items()):
                if buf is None or buf.device == dev:
                    continue
                key = (id(mod), bn)
                if key not in cache:
                    copy_ = buf.to(dev)
                    if _buf_sha(copy_) != _buf_sha(buf):
                        raise HFRefused("BUFFER-PIN", f"the {dev} copy of buffer {bn} differs")
                    cache[key] = copy_
                restore.append((mod._buffers, bn, buf))
                mod._buffers[bn] = cache[key]

    def run(self, u: int, x, ids):
        n = self.map.n
        if ids is None:
            if u != 0:
                raise ValueError(f"hf unit {u} needs the microbatch's input ids")
            ids = x
        self._check_train()
        dev = ids.device if u == 0 else x.device
        ids = ids.to(dev)
        restore: list = []
        hooks: list = []
        try:
            self._swap_buffers(dev, restore)
            for i in range(min(u - 1, n) if u >= 1 else 0):
                restore.append((self.L._modules, str(i), self.L._modules[str(i)]))
                self.L._modules[str(i)] = _LayerStub(self.map.layer_out[i], self.map.trace_shape)
            if u != 0:
                for name in self.map.prologue:
                    parent_name, _, leaf = name.rpartition(".")
                    parent = self.root.get_submodule(parent_name) if parent_name else self.root
                    restore.append((parent._modules, leaf, parent._modules[leaf]))
                    parent._modules[leaf] = _PrologueStub(*self.pro_stub[name])
            if u >= 1:

                def inject(mod, args, kwargs):
                    if args:
                        return (x,) + tuple(args[1:]), kwargs
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = x
                    return args, kwargs

                hooks.append(self.entry(u).register_forward_pre_hook(inject, with_kwargs=True))
            stop = _exit_unit(u)
            if stop <= n + 1:

                def leave(mod, args, kwargs):
                    raise _Exit(args[0] if args else kwargs["hidden_states"])

                hooks.append(self.entry(stop).register_forward_pre_hook(leave, with_kwargs=True))
            left = None
            with self._autocast(dev):
                try:
                    out = self.root(input_ids=ids, use_cache=False)
                except _Exit as e:
                    left = e.h
                except RuntimeError as e:
                    if "single memory location" in str(e):
                        raise HFRefused(
                            "STUB-WRITE", f"unit {u}: a stub was written in place: {e}"
                        ) from e
                    raise
            if _stub_written():
                raise HFRefused("STUB-WRITE", f"unit {u}: a stub's NaN was written in place")
            if left is not None:
                return left
            if u <= n:
                raise RuntimeError(f"hf unit {u}: HF's forward never reached unit {u + 1}'s entry")
            return out.logits
        finally:
            for h in hooks:
                h.remove()
            for d, k, v in reversed(restore):
                d[k] = v

    def vanilla(self, ids):
        self._check_train()
        with self._autocast(ids.device):
            return self.root(input_ids=ids, use_cache=False).logits


def _mirror(module: nn.Module, root, names: list[str]) -> None:
    """Register each named module of `root` under `module` at its own path
    (containers in between), so the unit's parameter names are HF's."""
    top = sorted(set(names))
    keep = [k for k in top if not any(k.startswith(o + ".") for o in top if o != k)]
    for name in keep:
        parts = name.split(".")
        cur: nn.Module = module
        for p in parts[:-1]:
            nxt = cur._modules.get(p)
            if nxt is None:
                nxt = nn.Module()
                cur.add_module(p, nxt)
            cur = nxt
        cur.add_module(parts[-1], root.get_submodule(name))


class HFUnit(nn.Module):
    """Unit `index` of an HF model: a window of ForCausalLM.forward. Called as
    unit(x, ids=ids); unit 0 takes the ids as x. Registers only its own
    modules, at their HF paths."""

    takes_ids = True
    _rt: _Runtime

    def __init__(self, rt: _Runtime, index: int, names: list[str]):
        super().__init__()
        object.__setattr__(self, "_rt", rt)
        self.index = index
        _mirror(self, rt.root, names)

    def forward(self, x, ids=None):
        return self._rt.run(self.index, x, ids)


class HFUnits(nn.ModuleList):
    """n+2 units over one shared tree, root in training mode (I-2)."""

    _rt: _Runtime

    def __init__(self, modules=None, rt: _Runtime | None = None):
        super().__init__(modules)
        object.__setattr__(self, "_rt", rt)

    @property
    def root(self):
        return self._rt.root

    @property
    def unit_map(self) -> UnitMap:
        return self._rt.map

    @property
    def exclusions(self) -> frozenset:
        return self._rt.exclusions

    @property
    def tied(self) -> tuple[int, int] | None:
        return (0, self._rt.map.n + 1) if self._rt.map.tied else None

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return nn.ModuleList(list(self._modules.values())[idx])
        return super().__getitem__(idx)

    def vanilla(self, ids) -> torch.Tensor:
        """The unwindowed ForCausalLM's logits on `ids` (no stubs, no hooks)."""
        return self._rt.vanilla(ids)


def _check_moe_rows(model, umap: UnitMap) -> None:
    """D8: every sparse-MoE block the classifier found must have been swapped
    by a SUBSTITUTE row; else MOE-NO-ROW naming its class (and the rollback)."""
    for idx, name in sorted(umap.moe_blocks.items()):
        try:
            m = model.get_submodule(name)
        except AttributeError as e:
            raise HFRefused("MOE-NO-ROW", f"block {name} is not in the model") from e
        if not hasattr(type(m), "recipe"):
            why = "; rolled back: model-3-6a" if rollback.rolled_back_to("moe-from-scratch") else ""
            raise HFRefused(
                "MOE-NO-ROW",
                f"sparse-MoE block {name} (layer {idx}, {type(m).__name__}) has no SUBSTITUTE "
                f"row at transformers {PIN}{why}",
            )


def units(model, umap: UnitMap, compute_dtype: str = "fp32") -> HFUnits:
    """The n+2 units of `model` (its SUBSTITUTE swap applied); a sparse-MoE
    block the trace classified without a row refuses MOE-NO-ROW."""
    _check_moe_rows(model, umap)
    rt = _Runtime(model, umap, compute_dtype)
    n = umap.n
    mods = [HFUnit(rt, 0, umap.prologue)]
    mods += [HFUnit(rt, i + 1, [f"{umap.layers}.{i}"]) for i in range(n)]
    mods.append(HFUnit(rt, n + 1, umap.epilogue))
    out = HFUnits(mods, rt)
    placed: dict = {}
    for u, unit in enumerate(mods):
        for _, p in unit.named_parameters(remove_duplicate=False):
            placed.setdefault(id(p), set()).add(u)
    for pname, p in model.named_parameters(remove_duplicate=False):
        us = placed.get(id(p), set())
        if not us:
            raise HFRefused("TRACE-UNPLACED", f"parameter {pname} is in no unit")
        if len(us) > 1 and not (umap.tied and us == {0, n + 1}):
            raise HFRefused("TRACE-SHARED", f"parameter {pname} is in units {sorted(us)}")
    return out


# ------------------------------------------------------------------- buffers


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def buffer_record(hf_config: dict) -> dict:
    """I-4: every buffer of the skeleton, its dtype, shape, sha256 and bytes."""
    m = skeleton(hf_config)
    tensors = {}
    for name, b in m.named_buffers():
        raw = _tensor_bytes(b)
        tensors[name] = {
            "dtype": _dtype_name(b.dtype),
            "shape": list(b.shape),
            "sha256": _sha(raw),
            "b64": base64.b64encode(raw).decode(),
        }
    return {
        "v": 1,
        "model_type": hf_config["model_type"],
        "hf_config_sha256": _sha(_canon(hf_config)),
        "transformers": PIN,
        "builder_fingerprint_sha256": _sha(_canon(fingerprint())),
        "tensors": tensors,
    }


def apply_buffer_record(model, record: dict) -> None:
    """Overwrite every buffer of `model` from the run's record; a record that
    does not name exactly the model's buffers, or whose bytes do not hash to
    its sha256, is refused naming the tensor."""
    if not isinstance(record, dict) or record.get("v") != 1 or record.get("transformers") != PIN:
        raise HFRefused("BUFFER-RECORD", f"not a v1 buffer record of transformers {PIN}")
    tensors = record.get("tensors") or {}
    have = dict(model.named_buffers())
    for name in sorted(set(have) ^ set(tensors)):
        side = "the model" if name in have else "the record"
        raise HFRefused("BUFFER-RECORD", f"buffer {name} is only in {side}")
    for name, b in have.items():
        ent = tensors[name]
        raw = base64.b64decode(ent["b64"])
        if _sha(raw) != ent["sha256"]:
            raise HFRefused(
                "BUFFER-PIN", f"buffer {name}: its bytes do not hash to the pinned sha256"
            )
        if ent["dtype"] != _dtype_name(b.dtype) or list(ent["shape"]) != list(b.shape):
            raise HFRefused(
                "BUFFER-RECORD", f"buffer {name}: dtype or shape differs from the model's"
            )
        dt = getattr(torch, ent["dtype"])
        t = (
            torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dt).reshape(b.shape).clone()
            if raw
            else torch.empty(b.shape, dtype=dt)
        )
        mod_name, _, bn = name.rpartition(".")
        mod = model.get_submodule(mod_name) if mod_name else model
        mod._buffers[bn] = t.to(b.device)


# --------------------------------------------------------------------- build


def build(cfg, seed: int, span: tuple[int, int] | None = None) -> HFUnits:
    """From config: the seeded full HF build on the CPU (decision 7, span
    ignored). With cfg.hf_model: the skeleton with only the units in `span`
    loaded (strict, MODEL-I4 over them), every other parameter on meta
    (BA-449). In both, every buffer is overwritten from cfg.hf_buffers."""
    hf_config = cfg.hf_config
    check_config(hf_config, cfg)
    record = cfg.hf_buffers
    if record is None:
        raise HFRefused(
            "BUFFER-RECORD", "cfg.hf_buffers is None: the run's buffer record is pinned at launch"
        )
    if record.get("hf_config_sha256") != _sha(_canon(hf_config)):
        raise HFRefused("BUFFER-RECORD", "the buffer record was built for another hf_config")
    umap = unit_map(hf_config)
    if cfg.hf_model:
        root = skeleton(hf_config)
        _set_moe_coefs(root, cfg)
    else:
        root = _full_build(hf_config, seed, cfg)
    apply_buffer_record(root, record)
    out = units(root, umap, cfg.compute_dtype)
    if cfg.hf_model:
        if cfg.hf_model_sha256:
            from wantrain.model import hf

            got = hf.checkpoint_digest(cfg.hf_model)
            if got != cfg.hf_model_sha256:
                raise HFRefused(
                    "LOAD-NAME",
                    f"checkpoint digest {got[:12]} differs from hf_model_sha256 {cfg.hf_model_sha256[:12]}",
                )
        a, b = span if span is not None else (0, umap.n + 2)
        load(out, cfg.hf_model, units=range(a, b))
    return out


# ----------------------------------------------------------- load and export


class LoadReport:
    """What was loaded, what was recomputed rather than loaded, and from where."""

    def __init__(self):
        self.loaded: list[str] = []
        self.loaded_bytes = 0
        self.recomputed: list[str] = []
        self.source = ""

    def as_dict(self) -> dict:
        return dict(
            loaded=len(self.loaded),
            loaded_bytes=self.loaded_bytes,
            recomputed=self.recomputed,
            source=self.source,
        )


# Hub per-expert names -> the 5.19.0 fused banks (ADR-0008 decision 10 (i)).
_HUB_EXPERTS = {
    "mixtral": (
        r"^(.*)\.experts\.(\d+)\.(w1|w3|w2)\.weight$",
        {"w1": "gate", "w3": "up", "w2": "down"},
    ),
    "qwen2_moe": (
        r"^(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$",
        {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    ),
}
_HUB_EXPERTS["qwen3_moe"] = _HUB_EXPERTS["qwen2_moe"]
_HUB_EXPERTS["deepseek_v3"] = _HUB_EXPERTS["qwen2_moe"]
_HUB_RENAME = {"mixtral": (".block_sparse_moe.", ".mlp.")}


def _hub_target(model_type: str, name: str):
    """A hub per-expert name -> (bank name, expert j, part) or None."""
    spec = _HUB_EXPERTS.get(model_type)
    if spec is None:
        return None
    ren = _HUB_RENAME.get(model_type)
    if ren:
        name = name.replace(*ren)
    m = re.match(spec[0], name)
    if not m:
        return None
    part = spec[1][m.group(3)]
    bank = f"{m.group(1)}.experts.{'down_proj' if part == 'down' else 'gate_up_proj'}"
    return bank, int(m.group(2)), part


def _hub_name(model_type: str, prefix: str, j: int, part: str) -> str:
    """The hub name of expert j's `part` under the fused bank's prefix."""
    if model_type == "mixtral":
        leaf = {"gate": "w1", "up": "w3", "down": "w2"}[part]
        return f"{prefix.replace('.mlp', '.block_sparse_moe')}.experts.{j}.{leaf}.weight"
    return f"{prefix}.experts.{j}.{part}_proj.weight"


def _conversion_named(model_type: str, name: str) -> str | None:
    """The transformers v5 conversion entry whose source pattern matches
    `name`, if any (refused LOAD-NAME: names must be the module tree's)."""
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    for t in get_checkpoint_conversion_mapping(model_type) or []:
        pats = getattr(t, "source_patterns", None) or []
        for p in [pats] if isinstance(pats, str) else pats:
            rxs = [re.escape(p).replace(r"\*", r"[^.]+")]
            with contextlib.suppress(re.error):
                re.compile(p)
                rxs.append(p)
            if any(re.search(rx, name) for rx in rxs):
                return p
    return None


def _root_of(model):
    return (model.root, model.unit_map) if isinstance(model, HFUnits) else (model, None)


def load(model, path: str, units=None) -> LoadReport:
    """Load a safetensors checkpoint 1:1 by HF names onto the skeleton's
    parameters of the requested units (MODEL-I4): missing, unused, shape- or
    dtype-class-mismatched and v5-conversion names refused, naming them;
    the hub-to-fused expert map is the one byte map. Persistent buffers come
    from the buffer record and are reported as recomputed."""
    from wantrain.model import hf

    _transformers()
    root, umap = _root_of(model)
    mt = root.config.model_type
    index = hf._index(path)
    rep = LoadReport()
    rep.source = path
    params = dict(root.named_parameters(remove_duplicate=False))
    alias: dict = {}
    for k, p in params.items():
        alias.setdefault(id(p), []).append(k)
    pers = set(root.state_dict(keep_vars=True)) - set(params)
    # name -> unit, over the requested units
    if isinstance(model, HFUnits):
        want = sorted(set(units)) if units is not None else list(range(len(model)))
        unit_of: dict = {}
        for u in want:
            for k, _ in model[u].named_parameters(remove_duplicate=False):
                unit_of.setdefault(k, u)
    else:
        if units is not None:
            raise ValueError("units= needs the HFUnits of the model")
        unit_of = {k: 0 for k in params}
    hub: dict = {}
    renamed: dict = {}
    for name in sorted(index):
        if name in params or name in pers:
            continue
        ren = _HUB_RENAME.get(mt)
        if ren and ren[0] in name and name.replace(*ren) in params:
            renamed[name.replace(*ren)] = name
            continue
        tgt = _hub_target(mt, name)
        if tgt is not None and tgt[0] in params:
            hub.setdefault(tgt[0], {})[(tgt[1], tgt[2])] = name
            continue
        conv = _conversion_named(mt, name)
        if conv is not None:
            raise HFRefused(
                "LOAD-NAME", f"{name} is a v5 conversion name ({conv!r}), not the module tree's"
            )
        raise HFRefused("LOAD-UNUSED", f"checkpoint tensor {name} maps onto no skeleton name")
    rep.recomputed = sorted(k for k in index if k in pers)
    done: set = set()
    for k in params:
        if k not in unit_of or k in done:
            continue
        p = params[k]
        group = alias[id(p)]
        done.update(group)
        if k in hub:
            t = _load_bank(root, k, p, hub[k], index)
        else:
            src = next((g for g in group if g in index), None)
            src = src or next((renamed[g] for g in group if g in renamed), None)
            if src is None:
                raise HFRefused("LOAD-MISSING", f"parameter {k} is not in the checkpoint")
            f, info, base = index[src]
            t = hf._read_tensor(f, info, base)
            _check_shape(k, src, t, p)
        new = nn.Parameter(t.to(p.dtype), requires_grad=p.requires_grad)
        # Every alias of the group, inside the requested units or not: a tied
        # table is one parameter, and an alias left on the old (meta) tensor
        # would untie it, so a span holding one use of a split table could
        # never be matched with the other (stage._init_tied). Nothing more is
        # read: the tensor is the one just loaded.
        for g in group:
            mod, _, leaf = g.rpartition(".")
            setattr(root.get_submodule(mod) if mod else root, leaf, new)
        rep.loaded.append(k)
        rep.loaded_bytes += new.numel() * new.element_size()
    object.__setattr__(root, "_wantrain_hf_source", path)
    return rep


def _check_shape(k: str, src: str, t: torch.Tensor, p) -> None:
    if tuple(t.shape) != tuple(p.shape):
        raise HFRefused("LOAD-SHAPE", f"{src} has shape {tuple(t.shape)}; {k} is {tuple(p.shape)}")
    if t.is_floating_point() != p.is_floating_point():
        raise HFRefused("LOAD-SHAPE", f"{src} is {t.dtype}; {k} is {p.dtype}")


def _load_bank(root, bank: str, p, parts: dict, index: dict) -> torch.Tensor:
    """gate_up_proj[j, :f] <- gate_j, [j, f:] <- up_j; down_proj[j] <- down_j;
    every element written once."""
    from wantrain.model import hf

    E = p.shape[0]
    t = torch.empty(tuple(p.shape), dtype=p.dtype)
    need = ["down"] if bank.endswith("down_proj") else ["gate", "up"]
    f = p.shape[1] // 2
    for j in range(E):
        for part in need:
            name = parts.get((j, part))
            if name is None:
                raise HFRefused(
                    "LOAD-MISSING", f"{bank}: expert {j}'s {part} tensor is not in the checkpoint"
                )
            fl, info, base = index[name]
            src = hf._read_tensor(fl, info, base)
            dst = t[j] if part == "down" else (t[j, :f] if part == "gate" else t[j, f:])
            _check_shape(f"{bank}[{j}].{part}", name, src, dst)
            dst.copy_(src)
    return t


def write_safetensors(
    tensors: dict, path: str, shard_bytes: int, metadata: dict | None = None
) -> None:
    """Sharded safetensors with an index, written by this repository's code:
    every tensor's bytes exactly (−0 and NaN payloads included)."""
    from wantrain.model import hf

    os.makedirs(path, exist_ok=True)
    shards: list = [[]]
    size = 0
    for name in sorted(tensors):
        nb = tensors[name].numel() * tensors[name].element_size()
        if shards[-1] and size + nb > shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append(name)
        size += nb
    weight_map = {}
    total = 0
    for i, names in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        header: dict = {"__metadata__": {"format": "pt"}}
        blobs, off = [], 0
        for name in names:
            t = tensors[name].detach().cpu().contiguous()
            raw = _tensor_bytes(t) if t.numel() else b""
            header[name] = {
                "dtype": hf._CODES[t.dtype],
                "shape": list(t.shape),
                "data_offsets": [off, off + len(raw)],
            }
            blobs.append(raw)
            off += len(raw)
            weight_map[name] = fname
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        with open(os.path.join(path, fname), "wb") as fh:
            fh.write(len(hb).to_bytes(8, "little"))
            fh.write(hb)
            for raw in blobs:
                fh.write(raw)
        total += off
    meta = {"total_size": total, **(metadata or {})}
    with open(os.path.join(path, hf.INDEX), "w") as fh:
        json.dump({"metadata": meta, "weight_map": weight_map}, fh, indent=1, sort_keys=True)


def export(model, path: str, source_keys) -> None:
    """Write the model's tensors under exactly the source checkpoint's names
    (its tied-key set kept; hub per-expert names re-split from the fused banks)
    as sharded safetensors with an index whose metadata records the sha256 of
    every tokenizer file beside the source checkpoint (no tokenizer loaded)."""
    root, _ = _root_of(model)
    mt = root.config.model_type
    sd = dict(root.state_dict(keep_vars=True))
    out = {}
    for key in sorted(source_keys):
        ren = _HUB_RENAME.get(mt)
        if key in sd:
            t = sd[key]
        elif ren and ren[0] in key and key.replace(*ren) in sd:
            t = sd[key.replace(*ren)]
        else:
            tgt = _hub_target(mt, key)
            if tgt is None or tgt[0] not in sd:
                raise HFRefused("LOAD-NAME", f"export: source key {key} maps onto no model tensor")
            bank, j, part = tgt
            b = sd[bank]
            f = b.shape[1] // 2
            t = b[j] if part == "down" else (b[j, :f] if part == "gate" else b[j, f:])
        if t.device.type == "meta":
            raise HFRefused("LOAD-MISSING", f"export: {key} is not materialized")
        out[key] = t.detach().clone()
    params = dict(root.named_parameters(remove_duplicate=False))
    ren = _HUB_RENAME.get(mt)
    written = {id(sd[k]) for k in source_keys if k in sd}
    written |= {id(sd[k.replace(*ren)]) for k in source_keys if ren and k.replace(*ren) in sd}
    hubbed = {t[0] for t in (_hub_target(mt, k) for k in source_keys if k not in sd) if t}
    for k, p in params.items():
        if id(p) not in written and k not in hubbed:
            raise HFRefused("LOAD-MISSING", f"export: parameter {k} has no source key")
    src = getattr(root, "_wantrain_hf_source", None)
    toks = {}
    if src:
        d = src if os.path.isdir(src) else os.path.dirname(src)
        for fn in TOKENIZER_FILES:
            fp = os.path.join(d, fn)
            if os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    toks[fn] = _sha(fh.read())
    write_safetensors(out, path, EXPORT_SHARD_BYTES, {"tokenizer_sha256": toks})


# -------------------------------------------------------------------- census


def unit_census(hf_config: dict) -> list[dict]:
    """I-10: per unit its kind, trainable parameters (a tensor shared by two
    units counted at its first), linear MACs per token, attention shape,
    softcap, window, tie and param_shapes in the unit's named_parameters order."""
    with torch.random.fork_rng(devices=[]):
        root = skeleton(hf_config)
        us = units(root, unit_map(hf_config))
    c = root.config
    n = us.unit_map.n
    lt = getattr(c, "layer_types", None)
    win = getattr(c, "sliding_window", None)
    seen: set = set()
    out = []
    for u, unit in enumerate(us):
        shapes: list[list] = []
        for k, p in unit.named_parameters():
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            shapes.append([k, list(p.shape)])
        macs = sum(
            int(m.in_features) * int(m.out_features)
            for m in unit.modules()
            if isinstance(m, nn.Linear)
        )
        kind = "prologue" if u == 0 else ("epilogue" if u == n + 1 else "layer")
        layer = kind == "layer"
        window = None
        if layer and win:
            if isinstance(lt, list):
                window = win if lt[u - 1] == "sliding_attention" else None
            elif getattr(c, "use_sliding_window", True) is not False:
                window = win
        softcap = getattr(c, "attn_logit_softcapping" if layer else "final_logit_softcapping", None)
        heads = int(c.num_attention_heads)
        out.append(
            {
                "unit": u,
                "kind": kind,
                "params": sum(math.prod(s) for _, s in shapes),
                "linear_macs_per_token": macs,
                "attn_heads": heads if layer else 0,
                "attn_kv_heads": int(getattr(c, "num_key_value_heads", None) or heads)
                if layer
                else 0,
                "head_dim": int(getattr(c, "head_dim", None) or c.hidden_size // heads)
                if layer
                else 0,
                "softcap": softcap if (kind != "prologue" and softcap) else None,
                "window": window,
                "tied": bool(us.unit_map.tied) and kind != "layer",
                "param_shapes": shapes,
            }
        )
    return out


# --------------------------------------------------------------- fingerprint


def _b64sha(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


_SKIP_INFO = ("RECORD", "INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json")


def _record_digest(text: str) -> str:
    """The canonical sha256 of a RECORD: its lines sorted, without the
    installer's own entries, installed scripts (../) and bytecode, so a
    wheel's RECORD and the installed one hash alike."""
    keep = []
    for line in text.splitlines():
        path = line.split(",")[0]
        if not line.strip() or path.startswith("../") or "__pycache__" in path:
            continue
        if ".dist-info/" in path and path.rsplit("/", 1)[-1] in _SKIP_INFO:
            continue
        keep.append(line.strip())
    return _sha("\n".join(sorted(keep)).encode())


def torch_records_entry(wheel_path: str) -> dict:
    """{distribution: canonical RECORD sha256} of one wheel file."""
    import zipfile

    with zipfile.ZipFile(wheel_path) as z:
        rec = [n for n in z.namelist() if n.endswith(".dist-info/RECORD") and n.count("/") == 1]
        if len(rec) != 1:
            raise ValueError(f"{wheel_path}: not a wheel (no single dist-info/RECORD)")
        text = z.read(rec[0]).decode()
        meta = z.read(rec[0].replace("RECORD", "METADATA")).decode()
    name = next(
        line.split(":", 1)[1].strip() for line in meta.splitlines() if line.startswith("Name:")
    )
    return {_norm(name): _record_digest(text)}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


_FILE_SHA: dict = {}


def _file_sha(path: str) -> tuple[str, int]:
    st = os.stat(path)
    key = (path, st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in _FILE_SHA:
        with open(path, "rb") as fh:
            _FILE_SHA[key] = hashlib.sha256(fh.read()).digest()
    return base64.urlsafe_b64encode(_FILE_SHA[key]).rstrip(b"=").decode(), st.st_ino


def _check_dist(name: str, problems: list, files_of: dict) -> tuple[str, Any]:
    """Verify every file of an installed distribution against its RECORD;
    record (realpath -> (dist, relpath, inode)) in files_of. -> (RECORD text,
    the distribution)."""
    import importlib.metadata as md

    d = md.distribution(name)
    text = d.read_text("RECORD") or ""
    root_dirs = set()
    listed = set()
    for f in d.files or []:
        p = os.path.realpath(str(f.locate()))
        listed.add(p)
        top = str(f).split("/")[0]
        if not top.endswith(".dist-info") and not str(f).startswith("..") and "/" in str(f):
            root_dirs.add(os.path.realpath(str(d.locate_file(top))))
        if f.hash is None:
            continue
        try:
            digest, ino = _file_sha(p)
        except OSError:
            problems.append(("FP-RECORD", f"{name}: {f} is missing"))
            continue
        if digest != f.hash.value:
            problems.append(("FP-RECORD", f"{name}: {f} differs from its RECORD sha256"))
        files_of[p] = (_norm(name), str(f), ino)
    others = None
    for top in sorted(root_dirs):
        for dp, _, fns in os.walk(top):
            for fn in fns:
                p = os.path.realpath(os.path.join(dp, fn))
                if p in listed or (fn.endswith(".pyc") and os.path.basename(dp) == "__pycache__"):
                    continue
                # A namespace package's top directory (cuda/, nvidia/ on Linux CUDA wheels) holds several
                # distributions; a file another installed distribution records is not foreign.
                if others is None:
                    others = _recorded_paths()
                if p in others:
                    continue
                problems.append(("FP-RECORD", f"{name}: {p} is outside its RECORD"))
    return text, d


def _recorded_paths() -> set:
    """The real paths every installed distribution lists in its RECORD."""
    import importlib.metadata as md

    out = set()
    for d in md.distributions():
        for f in d.files or []:
            out.add(os.path.realpath(str(f.locate())))
    return out


def _torch_deps(d) -> list[str]:
    from packaging.requirements import Requirement

    out = []
    for r in d.requires or []:
        req = Requirement(r)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            out.append(req.name)
    return sorted(set(out))


def _platform_tag(d) -> str:
    wheel = d.read_text("WHEEL") or ""
    tags = [line.split(":", 1)[1].strip() for line in wheel.splitlines() if line.startswith("Tag:")]
    return tags[0] if tags else "unknown"


_ENV_FP: dict = {}


def _env_fingerprint() -> dict:
    """The RECORD checks of transformers, torch and torch's requirements:
    computed once per process (the files do not change under a run)."""
    if _ENV_FP:
        return _ENV_FP
    import importlib.metadata as md

    import transformers

    problems: list = []
    files_of: dict = {}
    tf_text, _ = _check_dist("transformers", problems, files_of)
    _, td = _check_dist("torch", problems, files_of)
    records = {"torch": _record_digest(td.read_text("RECORD") or "")}
    for dep in _torch_deps(td):
        try:
            text, dd = _check_dist(dep, problems, files_of)
        except md.PackageNotFoundError:
            problems.append(("FP-RECORD", f"torch requires {dep}, which is not installed"))
            continue
        records[_norm(dep)] = _record_digest(text)
    _ENV_FP.update(
        transformers={
            "version": transformers.__version__,
            "record_sha256": _sha(tf_text.encode()),
            "record_gz_b64": base64.b64encode(gzip.compress(tf_text.encode(), mtime=0)).decode(),
        },
        torch={
            "version": torch.__version__,
            "git_version": torch.version.git_version,
            "platform_tag": _platform_tag(td),
            "records": records,
        },
        problems=problems,
        files_of=files_of,
    )
    return _ENV_FP


_COMPILED: dict = {}


def _qualname(co) -> str:
    # co_qualname is 3.11+; on 3.10 the name and first line still identify it.
    return getattr(co, "co_qualname", co.co_name)


def _compiled(filename: str) -> dict:
    """{(qualname, firstlineno): code} of every code object compiled from a
    source file's verified bytes."""
    if filename not in _COMPILED:
        with open(filename, "rb") as fh:
            src = fh.read()
        top = compile(src, filename, "exec", dont_inherit=True, optimize=sys.flags.optimize)
        table: dict = {}
        stack = [top]
        while stack:
            co = stack.pop()
            table[(_qualname(co), co.co_firstlineno)] = co
            stack.extend(c for c in co.co_consts if hasattr(c, "co_code"))
        _COMPILED[filename] = table
    return _COMPILED[filename]


def _functions(obj) -> list:
    # Type checks only: an attribute lookup on a lazy module would import it.
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    if isinstance(obj, property):
        return [f for f in (obj.fget, obj.fset, obj.fdel) if isinstance(f, types.FunctionType)]
    return [obj] if isinstance(obj, types.FunctionType) else []


def _installed_roots() -> tuple:
    import sysconfig

    paths = sysconfig.get_paths()
    roots = {paths[k] for k in ("stdlib", "platstdlib", "purelib", "platlib") if k in paths}
    roots |= {p for p in sys.path if p.rstrip("/").endswith("site-packages")}
    return tuple(os.path.join(os.path.realpath(r), "") for r in roots)


def _code_problem(allowed: set) -> str | None:
    """The first function of a loaded transformers module or class that is not
    the code compiled from its verified source (a monkeypatch):
    - a name the module's own verified source defines (a module-level
      function, or a method of an nn.Module class: the compute path) must
      hold, under its decorators (`__wrapped__`), exactly the code compiled
      there, whatever file a replacement comes from (config classes are
      excluded: huggingface_hub's strict dataclasses regenerate their
      methods);
    - any other function from a fingerprinted file must equal its compiled
      code, and one from a file that is neither fingerprinted nor installed
      refuses. Imported helpers and decorator-generated methods of other
      installed distributions and the standard library are not compared;
      generated code (<string>) is not checked."""
    installed = _installed_roots()
    for name, mod in sorted(sys.modules.items()):
        if not (name == "transformers" or name.startswith("transformers.")) or mod is None:
            continue
        mfile = getattr(mod, "__file__", None)
        mreal = os.path.realpath(mfile) if isinstance(mfile, str) else None
        table = _compiled(mreal) if mreal is not None and mreal in allowed else {}
        defined = {q for q, _ in table}
        objs = [(k, k, True, v) for k, v in vars(mod).items()]
        for cname, _, _, cls in list(objs):
            if isinstance(cls, type) and getattr(cls, "__module__", None) == name:
                strict = issubclass(cls, nn.Module)
                objs += [
                    (f"{cname}.{k}", f"{cls.__qualname__}.{k}", strict, v)
                    for k, v in vars(cls).items()
                ]
        for qual, src_qual, strict, obj in objs:
            for fn in _functions(obj):
                if strict and src_qual in defined:
                    # The function, or a function it wraps (__wrapped__, or
                    # held in its closure by a decorator such as
                    # huggingface_hub's), is the source's own definition.
                    chain, todo = [], [fn]
                    while todo and len(chain) < 16:
                        cur = todo.pop(0)
                        if any(cur is f for f in chain):
                            continue
                        chain.append(cur)
                        todo += [w for w in (getattr(cur, "__wrapped__", None),) if callable(w)]
                        for cell in getattr(cur, "__closure__", None) or ():
                            with contextlib.suppress(ValueError):
                                if isinstance(cell.cell_contents, types.FunctionType):
                                    todo.append(cell.cell_contents)
                    ok = any(
                        (ico := getattr(f, "__code__", None)) is not None
                        and os.path.realpath(ico.co_filename) == mreal
                        and _qualname(ico) == src_qual
                        and table.get((src_qual, ico.co_firstlineno)) == ico
                        for f in chain
                    )
                    if not ok:
                        return f"{name}.{qual} is not the code its verified source defines"
                co = fn.__code__
                fname = co.co_filename
                if fname.startswith("<"):
                    continue
                real = os.path.realpath(fname)
                if real not in allowed:
                    if real.startswith(installed):
                        continue
                    return f"{name}.{qual} is code from {fname}, outside the installed files"
                ref = _compiled(real).get((_qualname(co), co.co_firstlineno))
                if ref is None or ref != co:
                    return f"{name}.{qual} differs from the code compiled from {fname}"
    return None


def _native(files_of: dict) -> tuple[list, list]:
    """Every loaded native object: a file of a fingerprinted RECORD tied by
    path and inode to the file its check hashed, else an OS object by sha256
    (a dyld shared-cache image by cache UUID and OS build). -> (entries,
    problems)."""
    paths: list = []
    if sys.platform == "darwin":
        import ctypes

        libc = ctypes.CDLL(None)
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        for i in range(libc._dyld_image_count()):
            raw = libc._dyld_get_image_name(i)
            if raw:
                paths.append(raw.decode())
    elif os.path.exists("/proc/self/maps"):
        with open("/proc/self/maps") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 6 and "x" in parts[1] and parts[5].startswith("/"):
                    paths.append(parts[5])
    by_base: dict = {}
    for p in files_of:
        by_base.setdefault(os.path.basename(p), []).append(p)
    cache = None
    entries, problems = set(), []
    for p in sorted(set(paths)):
        real = os.path.realpath(p)
        if real in files_of:
            dist, rel, ino = files_of[real]
            if os.stat(real).st_ino != ino:
                problems.append(("FP-NATIVE", f"{p} is not the file its RECORD check hashed"))
            entries.add(("record", dist, rel))
        elif os.path.basename(real) in by_base:
            problems.append(
                ("FP-NATIVE", f"{p} has a RECORD file's name but loads from another path")
            )
        elif os.path.isfile(real):
            entries.add(("os", os.path.basename(real), _file_sha(real)[0]))
        else:
            if cache is None:
                cache = _shared_cache_id()
            entries.add(("cache", p, cache))
    return [list(e) for e in sorted(entries)], problems


def _shared_cache_id() -> str:
    import ctypes
    import subprocess

    uuid = ""
    try:
        libc = ctypes.CDLL(None)
        buf = (ctypes.c_ubyte * 16)()
        if libc._dyld_get_shared_cache_uuid(buf):
            uuid = bytes(buf).hex()
    except (AttributeError, OSError):
        uuid = "unknown"
    try:
        build = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.osversion"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        build = "unknown"
    return f"{uuid}:{build}"


def _probe_vector() -> dict:
    import importlib.metadata as md

    out = {}
    for name in PROBED + REFUSED_PACKAGES:
        imp = _importable(name)
        try:
            ver = md.version(name) if imp else None
        except md.PackageNotFoundError:
            ver = None
        out[name] = [imp, ver]
    return out


def fingerprint(model=None) -> dict:
    """I-3 (MODEL-I10): the code identity of this process. None fingerprints
    the installed environment only; a model adds its module classes.
    `refusals` lists every FP-* cause found ("CAUSE: detail")."""
    env = _env_fingerprint()
    problems = list(env["problems"])
    tor = env["torch"]
    entry = TORCH_RECORDS.get((tor["platform_tag"], tor["version"]))
    if entry is None:
        problems.append(
            (
                "FP-TORCH-RECORDS",
                f"no TORCH_RECORDS entry for platform {tor['platform_tag']} torch {tor['version']}",
            )
        )
    elif entry != tor["records"]:
        bad = sorted(
            k for k in set(entry) | set(tor["records"]) if entry.get(k) != tor["records"].get(k)
        )
        problems.append(
            ("FP-TORCH-RECORDS", f"platform {tor['platform_tag']}: {bad[0]} differs from its entry")
        )
    pv = _probe_vector()
    for name, (imp, _) in pv.items():
        if imp and name not in PROBED_LOCKED:
            problems.append(("FP-PROBE", f"{name} is importable"))
    pre = _prefix_violation()
    if pre:
        problems.append(("FP-PYC", pre))
    code = _code_problem(set(env["files_of"]))
    if code:
        problems.append(("FP-CODE", code))
    native, nprob = _native(env["files_of"])
    problems += nprob
    out = {
        "transformers": dict(env["transformers"]),
        "torch": dict(tor, records=dict(tor["records"])),
        "probe_vector": pv,
        "native": native,
        "python": {"version": sys.version.split()[0], "implementation": sys.implementation.name},
        "prefix_ok": pre is None,
        "code_ok": code is None,
        "refusals": [f"{c}: {d}" for c, d in problems],
    }
    if model is not None:
        out["model_classes"] = sorted(
            {f"{type(m).__module__}.{type(m).__qualname__}" for m in model.modules()}
        )
    return out


def platform_admitted() -> bool:
    """Whether this platform tag and torch version have a TORCH_RECORDS entry
    (BA-463)."""
    import importlib.metadata as md

    return (_platform_tag(md.distribution("torch")), torch.__version__) in TORCH_RECORDS


def record_text() -> str:
    """The installed transformers RECORD."""
    import importlib.metadata as md

    return md.distribution("transformers").read_text("RECORD") or ""


def record_diff(recorded: str, other: str) -> str | None:
    """The first RECORD path (in the recorded text's order, then the other's)
    whose line differs or is missing, or None when equal."""

    def rows(text):
        out = {}
        for line in (text or "").splitlines():
            if line.strip():
                out.setdefault(line.split(",")[0], line.strip())
        return out

    a, b = rows(recorded), rows(other)
    for path in list(a) + [p for p in b if p not in a]:
        if a.get(path) != b.get(path):
            return path
    return None


def hf_code() -> dict:
    """I-1's hf_code of the installed transformers and torch."""
    _transformers()
    import transformers

    return {
        "transformers": transformers.__version__,
        "transformers_record_sha256": _sha(record_text().encode()),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
    }


def device_class(device: str) -> str:
    """ "cpu:<machine>:<ATen capability>", "mps:<chip>" or "cuda:<name>": the
    node's HELLO device class, by the same function (CONTROL-I25)."""
    from wantrain.resources.probe import device_class as probe_device_class

    return probe_device_class(device)


# ---------------------------------------------------------------------- gate


def _compose_tied(parts: list) -> torch.Tensor:
    """A parameter's per-microbatch gradient from its stages' parts, in
    backward order (a tied table: head use, then embedding use)."""
    acc = parts[0]
    for g in parts[1:]:
        acc = acc + g
    return acc


def _chain_buffers(root) -> None:
    """The chain's buffers stand as the record left them (a control seam)."""


def _bytes_of(model) -> int:
    seen, n = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            n += t.numel() * t.element_size()
    return n


def _ref_pass(vanilla, batches, dtype: str, dev):
    """Vanilla per microbatch: logits, loss hex and every gradient, folded in
    microbatch order (the stock fold)."""
    from wantrain.model import precision
    from wantrain.model.gpt import loss_fn

    params = list(vanilla.named_parameters())
    dt = precision.spec(dtype)["torch_dtype"]
    logits, losses, acc = [], [], None
    for ids, tgt in batches:
        with torch.autocast(device_type=dev.type, dtype=dt, enabled=dtype != "fp32"):
            lg = vanilla(input_ids=ids, use_cache=False).logits
        loss = loss_fn(lg.float(), tgt)
        gs = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
        gs = [torch.zeros_like(p) if g is None else g for (_, p), g in zip(params, gs)]
        logits.append(lg.detach())
        losses.append(float(loss.detach()).hex())
        acc = [g.clone() for g in gs] if acc is None else [a.add_(g) for a, g in zip(acc, gs)]
    assert acc is not None
    return logits, losses, {k: g for (k, _), g in zip(params, acc)}


def _chain_pass(us: HFUnits, bounds: list, batches, recompute: bool):
    """The unit chain cut at `bounds` (stage i = units [bounds[i], bounds[i+1])),
    each boundary detached, W deferred under capture() and flushed per stage;
    a parameter's parts composed per microbatch, then folded in microbatch
    order."""
    from wantrain.model import wgrad
    from wantrain.model.gpt import call_layer, loss_fn

    names = {}
    for k, p in us.root.named_parameters(remove_duplicate=False):
        names.setdefault(id(p), k)
    stages = [list(range(bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]
    logits, losses, acc = [], [], None

    def run_stage(idx, inp, ids):
        h = inp
        for u in idx:
            h = call_layer(us[u], ids if u == 0 else h, ids=ids)
        return h

    for ids, tgt in batches:
        outs = []
        h = None
        with torch.set_grad_enabled(not recompute):
            for idx in stages:
                inp = None if h is None else h.detach().requires_grad_(True)
                o = run_stage(idx, inp, ids)
                outs.append((inp, o))
                h = o
        assert h is not None
        logits.append(h.detach())
        parts: dict = {}
        gnext = None
        for s in range(len(stages) - 1, -1, -1):
            inp, o = outs[s]
            if recompute:
                inp = None if inp is None else inp.detach().requires_grad_(True)
                o = run_stage(stages[s], inp, ids)
            mods = nn.ModuleList([us[u] for u in stages[s]])
            own = []
            for p in mods.parameters():
                if p.requires_grad and p.device.type != "meta":
                    own.append(p)
            owned = wgrad.owned_params(mods)
            others = [p for p in own if id(p) not in owned]
            with wgrad.capture() as store:
                if s == len(stages) - 1:
                    loss = loss_fn(o.float(), tgt)
                    losses.append(float(loss.detach()).hex())
                    gs = torch.autograd.grad(
                        loss, ([inp] if inp is not None else []) + others, allow_unused=True
                    )
                else:
                    gs = torch.autograd.grad(
                        o,
                        ([inp] if inp is not None else []) + others,
                        grad_outputs=gnext,
                        allow_unused=True,
                    )
            deferred = wgrad.flush(store)
            gs = list(gs)
            if inp is not None:
                gnext = gs.pop(0)
            for p, g in zip(others, gs):
                parts.setdefault(id(p), []).append(torch.zeros_like(p) if g is None else g)
            for p in own:
                if id(p) in owned:
                    parts.setdefault(id(p), []).append(deferred.get(id(p), torch.zeros_like(p)))
        mb = {names[k]: _compose_tied(v) for k, v in parts.items()}
        acc = (
            {k: g.clone() for k, g in mb.items()}
            if acc is None
            else {k: acc[k].add_(mb[k]) for k in acc}
        )
    return logits, losses, acc


def _first_diff(ref, got, where: str) -> str | None:
    rl, rloss, rg = ref
    gl, gloss, gg = got
    for i, (a, b) in enumerate(zip(rl, gl)):
        if a.dtype != b.dtype or not torch.equal(a, b):
            return f"logits of microbatch {i} ({where})"
    for i, (a, b) in enumerate(zip(rloss, gloss)):
        if a != b:
            return f"loss of microbatch {i}: {a} vs {b} ({where})"
    for k, g in rg.items():
        h = gg.get(k)
        if h is None or h.dtype != g.dtype or not torch.equal(h, g):
            return f"gradient {k} ({where})"
    return None


def _ablated_logits(vanilla, hf_config: dict, change: dict, ids) -> torch.Tensor:
    m = _full_build({**hf_config, **change}, 0, swap=True)
    m.load_state_dict(vanilla.state_dict())
    m.to(ids.device)
    with torch.no_grad():
        return m(input_ids=ids, use_cache=False).logits


def _triggers(hf_config: dict, umap: UnitMap, vanilla, S: int, ids) -> dict:
    """Trigger counts; a declared trigger that does not fire is refused."""
    c = vanilla.config
    with torch.no_grad():
        base = vanilla(input_ids=ids, use_cache=False).logits
    trig = {"window": 0, "chunk": 0, "nope": 0, "softcap": 0, "sink": 0, "tie": 0, "moe": 0}
    lt = getattr(c, "layer_types", None)
    win = getattr(c, "sliding_window", None)
    if isinstance(win, int) and win < S:
        if isinstance(lt, list):
            trig["window"] = sum(t == "sliding_attention" for t in lt)
        elif getattr(c, "use_sliding_window", True) is not False:
            trig["window"] = umap.n
        if trig["window"] and torch.equal(
            base, _ablated_logits(vanilla, hf_config, {"sliding_window": 4 * S}, ids)
        ):
            raise HFRefused(
                "GATE-TRIGGER", f"sliding_window={win} < S={S} does not change the logits"
            )
    chunk = getattr(c, "attention_chunk_size", None)
    if isinstance(chunk, int) and chunk < S:
        trig["chunk"] = umap.n
    nope = getattr(c, "no_rope_layers", None)
    if isinstance(nope, list):
        trig["nope"] = sum(1 for v in nope if not v)
    for key in ("attn_logit_softcapping", "final_logit_softcapping"):
        v = getattr(c, key, None)
        if v:
            trig["softcap"] += 1
            if torch.equal(base, _ablated_logits(vanilla, hf_config, {key: None}, ids)):
                raise HFRefused("GATE-TRIGGER", f"{key}={v} does not change the logits")
    trig["sink"] = sum(1 for k, _ in vanilla.named_parameters() if k.rsplit(".", 1)[-1] == "sinks")
    if getattr(c, "tie_word_embeddings", False):
        if not umap.tied:
            raise HFRefused("GATE-TRIGGER", "tie_word_embeddings is set but no table is tied")
        trig["tie"] = 1
    trig["moe"] = sum(1 for m in vanilla.modules() if hasattr(type(m), "recipe"))
    return trig


GATE_MOE_COEFS = types.SimpleNamespace(moe_lbl_coef=0.01, moe_lbl_scope="layer", moe_z_coef=0.001)


def _gate(hf_config: dict, recipe: dict, device: str, rep: dict) -> None:

    from wantrain.model import wgrad

    seq = recipe.get("seq_len")
    check_config(
        hf_config,
        types.SimpleNamespace(
            seq_len=seq, lora_rank=0, boundary_dtype="fp32", fp8_boundaries=False
        ),
    )
    tiny = _shrunk(hf_config, keep_depth=False)
    check_config(tiny, None)
    dev = torch.device(device)
    S = _gate_seq(tiny)
    t0 = time.monotonic()
    vanilla = _full_build(tiny, 0, swap=False)
    rep["build_s"] = round(time.monotonic() - t0, 3)
    rep["build_bytes"] = _bytes_of(vanilla)
    if rep["build_bytes"] > GATE_BUILD_BYTES or rep["build_s"] > GATE_BUILD_S:
        raise HFRefused(
            "GATE-BUDGET",
            f"the tiny build takes {rep['build_bytes']} bytes and {rep['build_s']} s "
            f"(caps {GATE_BUILD_BYTES} bytes, {GATE_BUILD_S} s)",
        )
    umap = _traced(vanilla, tiny)
    _check_moe_rows(vanilla, umap)
    # The reference and the chain carry the same D8 aux injection (ADR-0009
    # D8 (4)), with nonzero coefficients so it runs at every cut.
    _set_moe_coefs(vanilla, GATE_MOE_COEFS)
    chain_root = copy.deepcopy(vanilla)
    _chain_buffers(chain_root)
    vanilla.to(dev)
    V = int(vanilla.config.vocab_size)
    g = torch.Generator().manual_seed(0x6A7E)
    mb = 2
    batches = [
        (
            torch.randint(0, V, (mb, S), generator=g).to(dev),
            torch.randint(0, V, (mb, S), generator=g).to(dev),
        )
        for _ in range(2)
    ]
    rep["triggers"] = _triggers(tiny, umap, vanilla, S, batches[0][0])
    a = batches[0][0][:1].clone()
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % V
    with torch.no_grad(), _rowwise_experts():
        la = vanilla(input_ids=a, use_cache=False).logits
        lb = vanilla(input_ids=b, use_cache=False).logits
    if not torch.equal(la[:, :-1], lb[:, :-1]):
        raise HFRefused("GATE-CAUSAL", "changing the last id changes earlier positions' logits")
    n = umap.n
    cfgs = [list(range(n + 3)), [0, n + 2]] + [[0, c, n + 2] for c in range(1, n + 2)]
    dtypes = ["fp32"] + (
        [recipe["compute_dtype"]] if recipe.get("compute_dtype", "fp32") != "fp32" else []
    )
    rep["dtypes"] = dtypes
    for dt in dtypes:
        us = units(chain_root, umap, dt)
        wgrad.convert(us, exclude=us.exclusions)
        us.to(dev)
        for M in (1, 2):
            ref = _ref_pass(vanilla, batches[:M], dt, dev)
            for bounds in cfgs:
                for rc in (False, True) if bounds == cfgs[0] else (False,):
                    where = f"cut {bounds}, {dt}, M={M}" + (", recompute" if rc else "")
                    try:
                        got = _chain_pass(us, bounds, batches[:M], rc)
                    except HFRefused:
                        raise
                    except Exception as e:
                        rep["first_difference"] = f"the unit chain fails ({where})"
                        raise HFRefused(
                            "GATE-DIFF", f"the unit chain fails ({where}): {type(e).__name__}: {e}"
                        ) from e
                    rep["cuts"].append(where)
                    diff = _first_diff(ref, got, where)
                    if diff:
                        rep["first_difference"] = diff
                        raise HFRefused("GATE-DIFF", diff)


def gate(hf_config: dict, recipe: dict, device: str) -> dict:
    """MODEL-I10's launch gate on one device (SPEC-model §3; recipe = I-9's
    {compute_dtype, seq_len, mb_size}). The comparison runs on every
    platform; a fingerprint refusal (FP-*) is reported in cause without
    skipping it (BA-463). -> GateReport (I-3)."""
    t0 = time.monotonic()
    rep: dict = {
        "ok": False,
        "cause": None,
        "detail": None,
        "device": device,
        "device_class": device_class(device),
        "cuts": [],
        "dtypes": [],
        "ms": 0,
        "triggers": {},
        "build_bytes": 0,
        "build_s": 0.0,
        "first_difference": None,
        "fingerprint": None,
    }
    try:
        with torch.random.fork_rng(devices=[]):
            _gate(hf_config, recipe, device, rep)
    except HFRefused as e:
        rep["cause"], rep["detail"] = e.cause, e.detail
    fp = fingerprint()
    rep["fingerprint"] = fp
    if rep["cause"] is None and fp["refusals"]:
        cause, _, detail = fp["refusals"][0].partition(": ")
        rep["cause"], rep["detail"] = cause, detail
    rep["ok"] = rep["cause"] is None
    rep["ms"] = int((time.monotonic() - t0) * 1000)
    return rep


# -------------------------------------------------------------------- census

# A pin bump's census (SPEC-model §2 "Pin", ADR-0009 F7): every model type of
# the pinned MODEL_FOR_CAUSAL_LM_MAPPING through the gate on the CPU.
CENSUS_RECIPE = {"compute_dtype": "bf16", "seq_len": 16, "mb_size": 2}
CENSUS_MAX_PARAMS = 20_000_000
CENSUS_PER_CHILD = 10
CENSUS_CHILD_S = 1800


def _census_one(mt: str) -> dict:
    """One type's row: {"model_type", "verdict", ...}. The verdict's first
    word is EXACT (the whole comparison ran and matched; a fingerprint
    refusal is kept beside it), DIFFER (a bit differs), SKIP (above
    CENSUS_MAX_PARAMS after shrinking, counted on meta), the gate's refusal
    cause, or ERROR."""
    row: dict = {"model_type": mt}
    t0 = time.monotonic()
    try:
        hc = tiny_config(mt)
        zeroed = sorted(
            k
            for k, v in hc.items()
            if ("dropout" in k or "pdrop" in k)
            and isinstance(v, (int, float))
            and not isinstance(v, bool)
            and v > 0
        )
        for k in zeroed:
            hc[k] = 0.0  # the census recipe sets every dropout to 0 (recorded)
        row["dropout_zeroed"] = zeroed
        try:
            with torch.random.fork_rng(devices=[]):
                meta = _construct(_shrunk(hc, keep_depth=False), torch.float32, meta=True)
            row["params"] = sum(p.numel() for p in {id(p): p for p in meta.parameters()}.values())
            del meta
        except HFRefused:
            raise
        except Exception as e:  # a constructor that needs real tensors: the gate decides
            row["params"] = None
            row["meta_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        if row["params"] is not None and row["params"] > CENSUS_MAX_PARAMS:
            row["verdict"] = f"SKIP {row['params']} parameters after shrinking"
            return row
        rep = gate(hc, CENSUS_RECIPE, "cpu")
        row["cuts"] = len(rep["cuts"])
        row["triggers"] = rep["triggers"]
        cause, diff = rep["cause"], rep["first_difference"]
        if cause is None or (cause.startswith("FP-") and rep["cuts"] and diff is None):
            row["verdict"] = "EXACT" + (f" (fingerprint {cause}: {rep['detail']})" if cause else "")
        elif cause == "GATE-DIFF" and diff and not diff.startswith("the unit chain fails"):
            row["verdict"] = f"DIFFER {diff}"
        else:
            row["verdict"] = f"{cause} {rep['detail']}"
    except HFRefused as e:
        row["verdict"] = f"{e.cause} {e.detail}"
    except Exception as e:  # a type that does not build or run is a row, never a crash
        row["verdict"] = f"ERROR {type(e).__name__}: {str(e)[:400]}"
    finally:
        row["s"] = round(time.monotonic() - t0, 3)
    return row


def _census_child(types: list, workdir: str, tag: str) -> list | None:
    """One census child (a fresh interpreter in stage_env with a fresh pycache
    prefix) over `types`; None when it dies or outlives CENSUS_CHILD_S."""
    import subprocess

    inp, out = os.path.join(workdir, f"{tag}.in"), os.path.join(workdir, f"{tag}.out")
    with open(inp, "w") as fh:
        json.dump({"types": types}, fh)
    args = [sys.executable, *interpreter_args(os.path.join(workdir, f"pyc-{tag}"))]
    args += ["-m", "wantrain.model.hfm", "census", "--in", inp, "--out", out]
    try:
        res = subprocess.run(
            args,
            env=stage_env(dict(os.environ)),
            capture_output=True,
            text=True,
            timeout=CENSUS_CHILD_S,
        )
    except subprocess.TimeoutExpired:
        return None
    if res.returncode != 0 or not os.path.exists(out):
        return None
    with open(out) as fh:
        return json.load(fh)["rows"]


def census(types: list | None = None, per_child: int = CENSUS_PER_CHILD) -> dict:
    """Every model type of the pinned mapping (or `types`) through the gate,
    `per_child` types per child process; a child that dies is rerun one type
    per child, and a type whose own child dies is an ERROR row."""
    import tempfile

    todo = sorted(_causal_map()) if types is None else list(types)
    rows: list = []
    with tempfile.TemporaryDirectory(prefix="hfm-census-") as d:
        for i in range(0, len(todo), per_child):
            chunk = todo[i : i + per_child]
            got = _census_child(chunk, d, f"c{i}")
            if got is None:
                got = []
                for j, mt in enumerate(chunk):
                    one = _census_child([mt], d, f"c{i}-{j}")
                    got += one or [{"model_type": mt, "verdict": "ERROR the census child died"}]
            rows += got
    first = [r["verdict"].split()[0] for r in rows]
    return {
        "transformers": PIN,
        "torch": torch.__version__,
        "recipe": CENSUS_RECIPE,
        "rows": rows,
        "counts": {v: first.count(v) for v in sorted(set(first))},
    }


# ------------------------------------------------------------------------ CLI


def _setup(seed: int) -> None:
    from wantrain.runtime import determinism

    determinism.setup_determinism(seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wantrain.model.hfm")
    ap.add_argument("cmd", choices=("gate", "record", "census", "torch-records"))
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.inp is None and a.cmd != "census":
        ap.error(f"{a.cmd} requires --in")
    payload: dict = {}
    if a.inp is not None:
        with open(a.inp) as fh:
            payload = json.load(fh)
    if a.cmd == "census" and "hf_config" not in payload:
        if "types" in payload:  # a census child: its types in this process
            _setup(int(payload.get("seed", 0)))
            out: dict = {"rows": [_census_one(mt) for mt in payload["types"]]}
        else:  # the census: every type (or payload["census_types"]), in children
            out = census(
                payload.get("census_types"), int(payload.get("per_child", CENSUS_PER_CHILD))
            )
    elif a.cmd == "torch-records":
        entry: dict = {}
        for w in payload["wheels"]:
            entry.update(torch_records_entry(w))
        out = {"entry": entry}
    else:
        # BA-432: the determinism regime before any build, trace or compare.
        _setup(int(payload.get("seed", 0)))
        hf_config = payload["hf_config"]
        if a.cmd == "gate":
            devices = payload.get("devices") or [payload.get("device", "cpu")]
            reports = [gate(hf_config, payload["recipe"], d) for d in devices]
            out = {
                "reports": reports,
                "fingerprint": reports[0]["fingerprint"],
                "device_classes": [r["device_class"] for r in reports],
            }
        elif a.cmd == "record":
            guard_environment()
            text = record_text()
            out = {
                "record": buffer_record(hf_config),
                "census": unit_census(hf_config),
                "fields": config_fields(hf_config),
                "hf_code": hf_code(),
                "record_text_gz_b64": base64.b64encode(
                    gzip.compress(text.encode(), mtime=0)
                ).decode(),
                "fingerprint": fingerprint(),
            }
        else:
            out = {"census": unit_census(hf_config)}
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
