# Probe (not a benchmark): per-layer streamed replay of the v1 init stream vs full build.
import hashlib, sys, time, torch
sys.path.insert(0, "/Users/virgiliofilho/edt-sec/src")
from wantrain.model import arch, gpt
from wantrain.model.gpt import GPTConfig
torch.set_num_threads(1)
WRITEBACK = True

def h(t): return hashlib.sha256(t.detach().contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def classes(cfg):
    if gpt._uses_general_arch(cfg): return arch.Embed, arch.Block, arch.Head, True
    return gpt.Embed, gpt.Block, gpt.Head, False

def full_build(cfg, seed):
    torch.manual_seed(seed)
    L = gpt._build_arch_layers(cfg, seed)
    return L, torch.get_rng_state().clone()

def ctor(cfg, i, n, tok, E, B, H, is_arch):
    if i == 0: return E(cfg)
    if i < n - 1: return B(cfg)
    return H(cfg, tok) if is_arch else H(cfg)

def slice_build(cfg, seed, a, b, skip_scratch=False):
    E, B, H, is_arch = classes(cfg)
    n = cfg.n_layer + 2
    tie = is_arch and cfg.tie_embeddings
    torch.manual_seed(seed)
    kept, emb = {}, None
    for i in range(n):                      # phase 1: construction, one layer live at a time
        m = ctor(cfg, i, n, emb.tok if (tie and emb is not None) else None, E, B, H, is_arch)
        if i == 0 and tie: emb = m          # the tied table is referenced by the Head
        if a <= i < b: kept[i] = m
        del m
    if is_arch and cfg.init_scheme == "gpt2":
        for i in range(n):                  # phase 2: gpt2 pass, non-owned into a scratch layer
            if i in kept and WRITEBACK:
                with torch.device("meta"):
                    m = ctor(cfg, i, n, torch.nn.Embedding(cfg.vocab_size, cfg.d_model) if tie else None, E, B, H, is_arch)
                m = m.to_empty(device="cpu")
                with torch.no_grad():
                    for (k, dst), src_ in zip(m.state_dict(keep_vars=True).items(), kept[i].state_dict(keep_vars=True).values()):
                        dst.copy_(src_)                     # load the working layer from its home
                arch.gpt2_init([m], cfg)
                with torch.no_grad():
                    for dst, src_ in zip(kept[i].state_dict(keep_vars=True).values(), m.state_dict(keep_vars=True).values()):
                        dst.copy_(src_)                     # store it back (a tied home is one tensor: last write wins)
                del m
                continue
            if i in kept: m = kept[i]
            else:
                with torch.device("meta"):
                    m = ctor(cfg, i, n, torch.nn.Embedding(cfg.vocab_size, cfg.d_model) if tie else None, E, B, H, is_arch)
                m = m.to_empty(device="cpu")
                if tie and i == n - 1: m.out.weight = emb.tok.weight
                if tie and i == 0: m.tok = emb.tok
            if skip_scratch and i not in kept: continue   # negative control
            arch.gpt2_init([m], cfg)
            del m
    return kept, torch.get_rng_state().clone()

def meta_draws():
    torch.manual_seed(7); s0 = torch.get_rng_state().clone()
    with torch.device("meta"): arch.Block(GPTConfig(norm="rmsnorm", mlp="swiglu", bias=False))
    return torch.equal(s0, torch.get_rng_state())

cfgs = {
  "builtin": GPTConfig(n_layer=4),
  "arch_gpt2_tie_gqa": GPTConfig(n_layer=4, norm="rmsnorm", mlp="swiglu", bias=False, pos="rope",
                                 tie_embeddings=True, n_kv_head=2, init_scheme="gpt2"),
  "arch_gpt2_untied": GPTConfig(n_layer=4, norm="rmsnorm", mlp="swiglu", bias=False, init_scheme="gpt2"),
}
print("meta construction draws nothing:", meta_draws())
ok = True
for name, cfg in cfgs.items():
    full, r_full = full_build(cfg, 1234)
    n = len(full)
    for a in range(n):
        for b in range(a + 1, n + 1):
            kept, r = slice_build(cfg, 1234, a, b)
            same = torch.equal(r, r_full)
            for i in range(a, b):
                fs = {k: h(v) for k, v in full[i].state_dict().items()}
                ss = {k: h(v) for k, v in kept[i].state_dict().items()}
                same &= fs == ss
            ok &= same
            if not same: print("MISMATCH", name, a, b)
    # negative control: skipping scratch draws must be caught
    if cfg.init_scheme == "gpt2":
        kept, r = slice_build(cfg, 1234, n - 1, n, skip_scratch=True)
        caught = (not torch.equal(r, r_full)) or h(kept[n-1].out.weight) != h(full[n-1].out.weight)
        print(name, "control (skipped scratch draws) caught:", caught)
    print(name, "all contiguous slices equal (weights+rng):", ok, "layers", n)
print("RESULT", "PASS" if ok else "FAIL")
