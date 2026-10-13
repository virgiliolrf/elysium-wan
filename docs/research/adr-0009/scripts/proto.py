import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
# ADR-0009 derivation D-proto: the proposed stage mechanism, one object per stage as separate processes would hold.
# Each stage: HF's own ForCausalLM.forward over a skeleton (params on meta, buffers real), owning only its flat
# units (prologue = modules called before layers[0], layers[i], epilogue = modules called after layers[-1]);
# unowned layers are pass stubs; a non-first stage's nn.Embedding prologue modules are zero stubs; a pre-hook on
# unit a's entry module replaces its input with the received boundary; a pre-hook on unit b's entry raises with its
# input. Linears converted to SplitLinear; W deferred under capture() and flushed after B. M microbatches folded in
# index order per parameter; a tied table's gradient is composed on its Head owner as h_i + e_i, then folded.
# Reference: the vanilla HF model (no surgery), per-microbatch autograd.grad, the same fold. Compared bit for bit.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import sys, os, contextlib, warnings

os.environ["USE_HUB_KERNELS"] = "0"; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F, transformers
transformers.logging.set_verbosity_error()
from transformers import CONFIG_MAPPING, AutoModelForCausalLM
DT = [None]; TIEKEY = [None]; OUTK = [0]; CTRL = os.environ.get('CTRL', '')
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as MAP
from wantrain.model import wgrad
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

@contextlib.contextmanager
def params_on_meta():
    orig = nn.Module.register_parameter
    def reg(self, name, p):
        if p is not None and p.device.type != "meta":
            p = nn.Parameter(p.to("meta"), requires_grad=p.requires_grad)
        return orig(self, name, p)
    nn.Module.register_parameter = reg
    try: yield
    finally: nn.Module.register_parameter = orig

def layer_list(m):
    n = m.config.num_hidden_layers
    c = [(k, v) for k, v in m.named_modules() if isinstance(v, nn.ModuleList) and len(v) == n]
    assert len(c) >= 1; return c[0]

def trace_units(ref, ids):
    """Call order of parameter-holding modules -> flat units. Returns (prologue, epilogue, entry_of_epilogue)."""
    lname, L = layer_list(ref)
    in_layers = {id(x) for l in L for x in l.modules()}
    order, state = [], {"phase": "pro"}
    hs = []
    for i, l in enumerate(L):
        hs.append(l.register_forward_pre_hook(lambda m, a, i=i: state.__setitem__("phase", "mid") if i == 0 else None))
        if i == len(L) - 1:
            hs.append(l.register_forward_hook(lambda m, a, o: state.__setitem__("phase", "epi")))
    for name, mod in ref.named_modules():
        if id(mod) in in_layers or not list(mod.parameters(recurse=False)): continue
        hs.append(mod.register_forward_pre_hook(lambda m, a, name=name: order.append((state["phase"], name))))
    hs.append(L[0].register_forward_hook(lambda m, a, o: OUTK.__setitem__(0, len(o) if isinstance(o, tuple) else 0)))
    with torch.no_grad(): ref(input_ids=ids, use_cache=False)
    for h in hs: h.remove()
    pro = [n for p, n in order if p == "pro"]; epi = [n for p, n in order if p == "epi"]
    assert not [n for p, n in order if p == "mid"], "a non-layer module runs between layers: refused"
    return lname, pro, epi

class Exit(Exception):
    def __init__(s, h): s.h = h
class Pass(nn.Module):
    def __init__(s, k=0): super().__init__(); s.k = k
    def forward(s, hidden_states, *a, **kw): return hidden_states if not s.k else (hidden_states,) + (None,) * (s.k - 1)
class ZeroEmbed(nn.Module):
    def __init__(s, d, dtype): super().__init__(); s.d, s.dt = d, dtype
    def forward(s, ids): return torch.zeros(*ids.shape, s.d, dtype=s.dt)

def first_tensor_arg(args, kwargs):
    return ("a", 0) if args else ("k", "hidden_states")

def make_stage(cls, cfg, ckpt, lname, pro, epi, a, b, n):
    """Flat units: 0 = prologue, 1..n = layers, n+1 = epilogue. Stage owns [a, b)."""
    if CTRL == "to_empty":
        with torch.device("meta"): m = AutoModelForCausalLM.from_config(cfg, dtype=DT[0])
        for mod in m.modules():
            for bn, bb in list(mod._buffers.items()):
                if bb is not None: mod._buffers[bn] = torch.zeros(bb.shape, dtype=bb.dtype)  # to_empty, zero-filled
    else:
        with params_on_meta(): m = AutoModelForCausalLM.from_config(cfg, dtype=DT[0])
    m.train()
    L = m.get_submodule(lname)
    owned = set()
    if a == 0: owned |= {p for x in pro for p in [x]}
    if b == n + 2: owned |= set(epi)
    for name, p in list(m.named_parameters(remove_duplicate=False)):
        mod_name, _, pn = name.rpartition(".")
        unit = None
        if name.startswith(lname + "."):
            unit = 1 + int(name[len(lname) + 1:].split(".")[0])
        elif mod_name in pro: unit = 0
        elif mod_name in epi: unit = n + 1
        assert unit is not None, f"unplaced parameter {name}: refused"
        if a <= unit < b:
            mod = m.get_submodule(mod_name)
            setattr(mod, pn, nn.Parameter(ckpt[name].clone()))
    # re-tie: a table shared by a prologue and an epilogue module, both owned here, is one Parameter (as HF ties it)
    if cfg.tie_word_embeddings and a == 0 and b == n + 2:
        m.get_output_embeddings().weight = m.get_input_embeddings().weight
    for i in range(n):
        if not (a <= i + 1 < b): L[i] = Pass(OUTK[0])
    if a > 0:
        for x in pro:
            mod = m.get_submodule(x)
            assert isinstance(mod, nn.Embedding), f"non-embedding prologue module {x} on a non-first stage: refused"
            d, dt = mod.embedding_dim, ckpt[x + ".weight"].dtype
            mod.forward = (lambda ids, d=d, dt=dt: torch.zeros(*ids.shape, d, dtype=dt))
    wgrad.convert(m)
    box = {}
    def entry(u):  # module whose input is unit u's input
        return L[u - 1] if 1 <= u <= n else m.get_submodule(epi[0])
    def inj(mod, args, kwargs):
        if args: return (box["h"],) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = box["h"]; return args, kwargs
    def ext(mod, args, kwargs):
        raise Exit(args[0] if args else kwargs["hidden_states"])
    if a > 0 and not (CTRL == "inputs_embeds" and a == 1): entry(a).register_forward_pre_hook(inj, with_kwargs=True)
    if b <= n + 1: entry(b).register_forward_pre_hook(ext, with_kwargs=True)
    def run(ids, h):
        box["h"] = h
        if CTRL == "inputs_embeds" and h is not None:
            try: return m(inputs_embeds=h, use_cache=False).logits
            except Exit as e: return e.h
        try: return m(input_ids=ids, use_cache=False).logits
        except Exit as e: return e.h
    return m, run

def check(mt, kw, dtype, M=2, S=24, cfg=None):
    if cfg is None: cfg = CONFIG_MAPPING[mt](**kw)
    cfg._attn_implementation = "eager"
    cls = getattr(transformers, MAP[mt])
    DT[0] = dtype
    torch.manual_seed(0); ref = AutoModelForCausalLM.from_config(cfg, dtype=dtype).train()
    for x in ref.modules():
        if isinstance(x, nn.Dropout): x.p = 0.0
    for k, v in vars(cfg).items():
        if ("dropout" in k or "pdrop" in k) and isinstance(v, float): setattr(cfg, k, 0.0)
    ckpt = {k: v.detach().clone() for k, v in ref.named_parameters(remove_duplicate=False)}
    n = cfg.num_hidden_layers; V = cfg.vocab_size
    g = torch.Generator().manual_seed(1)
    batches = [(torch.randint(0, V, (2, S), generator=g), torch.randint(0, V, (2, S), generator=g)) for _ in range(M)]
    lname, pro, epi = trace_units(ref, batches[0][0])
    # reference
    params = [(k, p) for k, p in ref.named_parameters()]
    acc, ref_logits = None, []
    for x, t in batches:
        lg = ref(input_ids=x, use_cache=False).logits; ref_logits.append(lg)
        gs = torch.autograd.grad(F.cross_entropy(lg.float().reshape(-1, V), t.reshape(-1)), [p for _, p in params])
        if acc is None: acc = [gg.clone() for gg in gs]
        else:
            for aa, gg in zip(acc, gs): aa.add_(gg)
    ref_grad = {k: a for (k, _), a in zip(params, acc)}
    # pipeline: one unit per stage (the deepest cut), M microbatches
    bounds = list(range(n + 3))
    stages = [make_stage(cls, cfg, ckpt, lname, pro, epi, bounds[s], bounds[s + 1], n) for s in range(len(bounds) - 1)]
    tied_names = set()
    if cfg.tie_word_embeddings:
        e = [k for k in ckpt if k.startswith(pro[0])][0]; hname = [k for k in ckpt if k.startswith(epi[-1])][0]
        tied_names = {e, hname}; TIEKEY[0] = e
    per_mb = []  # per microbatch: {param name: grad} from every stage
    same_logits = True
    for (x, t), rl in zip(batches, ref_logits):
        outs, ins, h = [], [], None
        for m, run in stages:
            hin = None if h is None else h.detach().clone().requires_grad_(True)
            o = run(x, hin); ins.append(hin); outs.append(o); h = o
        same_logits &= torch.equal(h, rl)
        grads = {}
        for s in range(len(stages) - 1, -1, -1):
            m, _ = stages[s]
            own = [(k, p) for k, p in m.named_parameters() if p.device.type != "meta" and p.requires_grad]
            with wgrad.capture() as store:
                if s == len(stages) - 1:
                    loss = F.cross_entropy(outs[s].float().reshape(-1, V), t.reshape(-1))
                    gin = torch.autograd.grad(loss, [ins[s]] + [p for _, p in own if id(p) not in wgrad.owned_params(m)], allow_unused=True)
                else:
                    gin = torch.autograd.grad(outs[s], ([ins[s]] if ins[s] is not None else []) + [p for _, p in own if id(p) not in wgrad.owned_params(m)], grad_outputs=gnext, allow_unused=True)
                deferred = wgrad.flush(store)  # W
            bparams = [(k, p) for k, p in own if id(p) not in wgrad.owned_params(m)]
            gb = gin[1:] if ins[s] is not None else gin
            if ins[s] is not None: gnext = gin[0]
            for (k, p), gg in zip(bparams, gb):
                grads.setdefault(k, []).append(gg if gg is not None else torch.zeros_like(p))
            for k, p in own:
                if id(p) in deferred: grads.setdefault(k, []).append(deferred[id(p)])
        per_mb.append(grads)
    # compose and fold: a tied table's two uses are h_i + e_i (head first, as autograd meets them), then the fold
    out = {}
    for grads in per_mb:
        cur = {}
        for k, parts in grads.items():
            key = TIEKEY[0] if k in tied_names else k
            for gg in (parts[:1] if (CTRL == "tie_head_only" and k in tied_names and k != TIEKEY[0]) else parts):
                if CTRL == "tie_head_only" and k in tied_names and k == TIEKEY[0]: continue
                cur[key] = gg if key not in cur else cur[key] + gg
        for key, gg in cur.items():
            out[key] = gg.clone() if key not in out else out[key].add_(gg)
    bad = [k for k, g in ref_grad.items() if k not in out or not torch.equal(out[k], g)]
    if os.environ.get("DBG") and bad:
        k = bad[0]; print("DBG", k, "parts per mb", [ {kk: len(v) for kk, v in gr.items() if kk in tied_names} for gr in per_mb ], "tied", tied_names,
              "ndiff", (out[k] != ref_grad[k]).sum().item(), "maxabs", (out[k]-ref_grad[k]).abs().max().item())
    return dict(logits=same_logits, grads=len(ref_grad), differ=len(bad), first_bad=bad[:2], stages=len(stages),
                tied=bool(tied_names), prologue=pro, epilogue=epi)

BASE = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            vocab_size=128, max_position_embeddings=64, head_dim=16)
FAM = {
 "llama": dict(rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16}),
 "qwen3": {}, "gemma2": dict(sliding_window=8, attn_logit_softcapping=50.0, final_logit_softcapping=30.0, query_pre_attn_scalar=16),
 "gemma3_text": dict(sliding_window=8, sliding_window_pattern=2, query_pre_attn_scalar=16),
 "granite": dict(embedding_multiplier=12.0, logits_scaling=8.0, residual_multiplier=0.2, attention_multiplier=0.25),
 "gpt2": dict(n_embd=64, n_layer=4, n_head=4, n_positions=64),
 "llama4_text": dict(no_rope_layers=[1, 1, 1, 0], attention_chunk_size=8, num_local_experts=2, num_experts_per_tok=1, intermediate_size_mlp=128, interleave_moe_layer_step=2),
 "gpt_oss": dict(num_local_experts=4, num_experts_per_tok=2, sliding_window=8),
 "deepseek_v3": dict(head_dim=8, num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16, n_routed_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1, first_k_dense_replace=1, moe_intermediate_size=32, n_shared_experts=1),
 "phi3": dict(partial_rotary_factor=0.5, pad_token_id=0), "starcoder2": dict(sliding_window=8),
}
if __name__ != '__main__': raise SystemExit
only = sys.argv[1:] or list(FAM)
for mt in only:
    for dt in (torch.float32, torch.bfloat16):
        try: r = check(mt, {**BASE, **FAM[mt]}, dt)
        except Exception as ex: r = f"ERROR {type(ex).__name__}: {str(ex)[:150]}"
        print(f"{mt:12s} {str(dt)[6:]:9s} {r}", flush=True)
