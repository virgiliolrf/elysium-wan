# ADR-0007 round-1 probe (evidence, not a test of the product; not collected by pytest).
# Run: .venv/bin/python -I docs/research/adr-0007/probe_r1.py   (from the repository root; 1 thread, CPU)
# What it measures:
#   A. build_slice with the general home-write rule (every tensor an owned layer references lives at home; EVERY
#      working layer, owned or not, that references a home tensor loads it before its phase and stores it after; all
#      other tensors are scratch) equals build_layers(full)[a:b] and the RNG end state, for every contiguous slice of
#      four arms, including tied gpt2 (tied default init is refused by GPTConfig.validate, gpt.py:125, so it has no arm).
#   B. Controls that must be caught: (C1) tied gpt2 [0,1): the non-owned Head's write goes to scratch; (C2) tied gpt2
#      [L-1,L): the owned Head's phase-2 store is dropped; (C3) non-owned draws skipped; (C4) stream stopped
#      at b.
#   C. RNG boundary ledger (reviewer idea): set_rng_state(ledger[phase][i]) and draw only the layers that write a home
#      tensor; exact? draws saved?
#   D. Costs: sampler fingerprint probe time; hashlib sha256 throughput.
import hashlib, os, sys, time
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from wantrain.model import arch, gpt  # noqa: E402
from wantrain.model.gpt import GPTConfig  # noqa: E402

torch.set_num_threads(1)
SEED = 1234


def h(t):
    return hashlib.sha256(t.detach().contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def kinds(cfg):
    if gpt._uses_general_arch(cfg):
        return arch.Embed, arch.Block, arch.Head, True
    return gpt.Embed, gpt.Block, gpt.Head, False


def ctor(cfg, i, n, tok, K):
    E, B, H, is_arch = K
    if i == 0:
        return E(cfg)
    if i < n - 1:
        return B(cfg)
    return H(cfg, tok) if is_arch else H(cfg)


def skeleton(cfg):
    """Full model on meta: no storage, no draw. Returns layers and, per layer, [(name, key)] with key = id of the
    shared Parameter object (a tied table has one key referenced from two layers)."""
    K = kinds(cfg)
    n = cfg.n_layer + 2
    tie = K[3] and cfg.tie_embeddings
    with torch.device("meta"):
        layers = []
        for i in range(n):
            layers.append(ctor(cfg, i, n, layers[0].tok if (tie and i == n - 1) else None, K))
    refs = [[(nm, id(p)) for nm, p in L.named_parameters(remove_duplicate=False)] for L in layers]
    return layers, refs, n, tie, K


def working(cfg, i, n, tie, K):
    """A fresh CPU working layer with the full build's shapes and strides, uninitialized (later phases)."""
    with torch.device("meta"):
        m = ctor(cfg, i, n, nn.Embedding(cfg.vocab_size, cfg.d_model) if (tie and i == n - 1) else None, K)
    return m.to_empty(device="cpu")


def build_slice(cfg, a, b, ctl=None, stop_at_b=False):
    layers, refs, n, tie, K = skeleton(cfg)
    owned = set(range(a, b))
    H = {k for i in owned for _, k in refs[i]}  # home keys: referenced by an owned layer
    home = {}  # key -> tensor (the slot's home; here the CPU)
    torch.manual_seed(SEED)

    def load(i, m):
        with torch.no_grad():
            for nm, k in refs[i]:
                if k in home:
                    m.get_parameter(nm).copy_(home[k])

    def store(i, m, drop=False):
        if drop:
            return
        with torch.no_grad():
            for nm, k in refs[i]:
                if k in H:
                    t = m.get_parameter(nm)
                    if k in home:
                        home[k].copy_(t)
                    else:
                        home[k] = t.detach().clone()

    last = b if stop_at_b else n
    # phase 1: construction (the draws happen in a real working layer)
    owned_mods = {}
    for i in range(last):
        tok = None
        if tie and i == n - 1:
            tok = nn.Embedding(cfg.vocab_size, cfg.d_model, device="meta").to_empty(device="cpu")
            kt = refs[0][[nm for nm, _ in refs[0]].index("tok.weight")][1]
            if kt in home:
                with torch.no_grad():
                    tok.weight.copy_(home[kt])
        m = ctor(cfg, i, n, tok, K)
        store(i, m)
        if i in owned:
            owned_mods[i] = m  # keep the module for its non-parameter state (buffers)
    # phase 2: gpt2_init
    if K[3] and cfg.init_scheme == "gpt2":
        for i in range(last):
            if ctl == "C3" and i not in owned:
                continue
            m = working(cfg, i, n, tie, K)
            load(i, m)
            arch.gpt2_init([m], cfg)
            store(i, m, drop=(i == n - 1 and ((ctl == "C1" and i not in owned) or (ctl == "C2" and i in owned))))
    end = torch.get_rng_state().clone()
    out = {}
    for i in owned:
        out[i] = {nm: h(home[k]) for nm, k in refs[i]}
    return out, end


def full(cfg):
    torch.manual_seed(SEED)
    L = gpt._build_arch_layers(cfg, SEED)
    return L, torch.get_rng_state().clone()


def compare(cfg, L, r_full, a, b, **kw):
    out, r = build_slice(cfg, a, b, **kw)
    ok = torch.equal(r, r_full)
    for i in range(a, b):
        ok &= out[i] == {nm: h(p) for nm, p in L[i].named_parameters(remove_duplicate=False)}
    return ok


base = dict(n_layer=4)
gen = dict(norm="rmsnorm", mlp="swiglu", bias=False)
ARMS = {
    "builtin": GPTConfig(**base),
    "general_default_untied": GPTConfig(**base, **gen),
    "general_gpt2_untied": GPTConfig(**base, **gen, init_scheme="gpt2"),
    "general_gpt2_tied_gqa_rope": GPTConfig(**base, **gen, pos="rope", tie_embeddings=True, n_kv_head=2,
                                            init_scheme="gpt2"),
}
allok, nslices = True, 0
for name, cfg in ARMS.items():
    L, r_full = full(cfg)
    n = len(L)
    ok = True
    for a in range(n):
        for b in range(a + 1, n + 1):
            s = compare(cfg, L, r_full, a, b)
            nslices += 1
            if not s:
                print("MISMATCH", name, a, b)
            ok &= s
    allok &= ok
    print(f"A {name}: {n * (n + 1) // 2} slices equal (owned tensors + RNG end state): {ok}")
print(f"A total slices {nslices} all equal: {allok}")

t = ARMS["general_gpt2_tied_gqa_rope"]; Lt, rt = full(t); nt = len(Lt)
g = ARMS["general_gpt2_untied"]; Lg, rg = full(g); ng = len(Lg)
print("B C1 (tied gpt2 [0,1), non-owned Head write to scratch) caught:", not compare(t, Lt, rt, 0, 1, ctl="C1"))
print("B C2 (tied gpt2 [L-1,L), owned Head's phase-2 store dropped) caught:", not compare(t, Lt, rt, nt - 1, nt, ctl="C2"))
print("B C3 (non-owned gpt2 draws skipped, [0,1) untied) caught:", not compare(g, Lg, rg, 0, 1, ctl="C3"))
print("B C4 (stream stopped at b, [0,2) untied gpt2) caught:", not compare(g, Lg, rg, 0, 2, stop_at_b=True))


# C. RNG boundary ledger
def ledger(cfg):
    layers, refs, n, tie, K = skeleton(cfg)
    torch.manual_seed(SEED)
    led = {1: [], 2: []}
    for i in range(n):
        led[1].append(torch.get_rng_state().clone())
        tok = nn.Embedding(cfg.vocab_size, cfg.d_model, device="meta").to_empty(device="cpu") if (tie and i == n - 1) else None
        ctor(cfg, i, n, tok, K)  # a tie target built for real would draw: meta, then to_empty
    if K[3] and cfg.init_scheme == "gpt2":
        for i in range(n):
            led[2].append(torch.get_rng_state().clone())
            arch.gpt2_init([working(cfg, i, n, tie, K)], cfg)
    return led, torch.get_rng_state().clone()


def ledger_slice(cfg, a, b, led, end):
    layers, refs, n, tie, K = skeleton(cfg)
    owned = set(range(a, b))
    H = {k for i in owned for _, k in refs[i]}
    writers = [i for i in range(n) if any(k in H for _, k in refs[i])]
    home = {}
    for i in writers:
        torch.set_rng_state(led[1][i])
        tok = None
        if tie and i == n - 1:
            tok = nn.Embedding(cfg.vocab_size, cfg.d_model, device="meta").to_empty(device="cpu")
            kt = dict((nm, k) for nm, k in refs[0])["tok.weight"]
            if kt in home:
                with torch.no_grad():
                    tok.weight.copy_(home[kt])
        m = ctor(cfg, i, n, tok, K)
        with torch.no_grad():
            for nm, k in refs[i]:
                if k in H:
                    home.setdefault(k, m.get_parameter(nm).detach().clone()).copy_(m.get_parameter(nm))
    if K[3] and cfg.init_scheme == "gpt2":
        for i in writers:
            torch.set_rng_state(led[2][i])
            m = working(cfg, i, n, tie, K)
            with torch.no_grad():
                for nm, k in refs[i]:
                    if k in home:
                        m.get_parameter(nm).copy_(home[k])
            arch.gpt2_init([m], cfg)
            with torch.no_grad():
                for nm, k in refs[i]:
                    if k in H:
                        home[k].copy_(m.get_parameter(nm))
    torch.set_rng_state(end)
    return {i: {nm: h(home[k]) for nm, k in refs[i]} for i in owned}, torch.get_rng_state().clone(), len(writers)


def ledger_slice_layer_major(cfg, a, b, led, end):
    """Each writer layer runs ALL its phases back to back (construction, then gpt2) from the ledger states, so an owned
    layer is finished in one working layer and stored once. Per-tensor write order is kept (Embed before Head)."""
    layers, refs, n, tie, K = skeleton(cfg)
    owned = set(range(a, b))
    H = {k for i in owned for _, k in refs[i]}
    writers = [i for i in range(n) if any(k in H for _, k in refs[i])]
    home = {}
    kt = dict(refs[0])["tok.weight"] if tie else None
    for i in writers:
        torch.set_rng_state(led[1][i])
        tok = None
        if tie and i == n - 1:
            tok = nn.Embedding(cfg.vocab_size, cfg.d_model, device="meta").to_empty(device="cpu")
            if kt in home:
                with torch.no_grad():
                    tok.weight.copy_(home[kt])
        m = ctor(cfg, i, n, tok, K)
        if K[3] and cfg.init_scheme == "gpt2":
            torch.set_rng_state(led[2][i])
            arch.gpt2_init([m], cfg)
        with torch.no_grad():
            for nm, k in refs[i]:
                if k in H:
                    home.setdefault(k, m.get_parameter(nm).detach().clone()).copy_(m.get_parameter(nm))
        del m
    torch.set_rng_state(end)
    return {i: {nm: h(home[k]) for nm, k in refs[i]} for i in owned}, torch.get_rng_state().clone(), len(writers)


lok, lcount = True, 0
for name, cfg in ARMS.items():
    L, r_full = full(cfg)
    led, end = ledger(cfg)
    n = len(L)
    for a in range(n):
        for b in range(a + 1, n + 1):
            for fn in (ledger_slice, ledger_slice_layer_major):
                out, r, nw = fn(cfg, a, b, led, end)
                s = torch.equal(r, r_full) and all(
                    out[i] == {nm: h(p) for nm, p in L[i].named_parameters(remove_duplicate=False)} for i in range(a, b))
                lok &= s
                lcount += 1
print(f"C ledger (phase-major and layer-major): {lcount} slice builds of 4 arms equal: {lok}; "
      f"state size {led[1][0].numel()} B")

# D. costs
def fingerprint():
    out = hashlib.sha256()
    for n_ in (15, 16, 17, 65536 + 3):
        torch.manual_seed(7); out.update(h(torch.empty(n_).normal_()).encode())
        torch.manual_seed(7); out.update(h(torch.empty(n_).uniform_()).encode())
    torch.manual_seed(7); out.update(h(nn.Linear(257, 255).weight).encode())
    torch.manual_seed(7); out.update(h(nn.Embedding(97, 64).weight).encode())
    out.update(h(torch.get_rng_state()).encode())
    return out.hexdigest()


fingerprint()
t0 = time.perf_counter(); fp = fingerprint(); tf = time.perf_counter() - t0
print(f"D fingerprint {fp[:16]}... {tf * 1e3:.2f} ms; repeat equal: {fp == fingerprint()}")
buf = torch.empty(64 * 2**20, dtype=torch.float32).uniform_()
raw = buf.view(torch.uint8).numpy().tobytes()
t0 = time.perf_counter(); hashlib.sha256(raw).hexdigest(); th = time.perf_counter() - t0
print(f"D sha256 {len(raw) / th / 1e9:.2f} GB/s over {len(raw) / 2**20:.0f} MiB (hashlib, 1 thread)")
print("RESULT", "PASS" if allok and lok else "FAIL")
