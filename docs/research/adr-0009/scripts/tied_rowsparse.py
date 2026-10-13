import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
# ADR-0009 round 1, E4a: the row-sparse order-preserving tied fold against the stock order.
# Stock (oracle today): per microbatch one autograd.grad over the tied Parameter (autograd sums the two uses), then
# acc = g_0.clone(); acc.add_(g_i). Row-sparse: the head use h_i arrives n_pend microbatches before the embedding use
# e_i; rows with no unapplied earlier contribution and untouched by microbatch i fold at once as acc + (h_i + 0.0);
# the rest wait for e_i and fold in microbatch order. Also: the per-use fold (sum e) + (sum h), for the count.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import os, sys, hashlib, warnings
os.environ["USE_HUB_KERNELS"] = "0"; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F, transformers
transformers.logging.set_verbosity_error()
from transformers import CONFIG_MAPPING, AutoModelForCausalLM
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
sha = lambda t: hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

def run(mt, extra, V, dtype, M=6, S=24, b=2):
    cfg = CONFIG_MAPPING[mt](hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, vocab_size=V, max_position_embeddings=64,
                             tie_word_embeddings=True, **extra)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0); m = AutoModelForCausalLM.from_config(cfg, dtype=dtype).train()
    E, H = m.get_input_embeddings(), m.get_output_embeddings()
    assert E.weight is H.weight
    W = E.weight
    g = torch.Generator().manual_seed(7)
    # a Zipf-like id stream so some rows stay untouched
    probs = 1.0 / torch.arange(1, V + 1, dtype=torch.float64); probs /= probs.sum()
    batches = [(torch.multinomial(probs, b * S, replacement=True, generator=g).view(b, S),
                torch.randint(0, V, (b, S), generator=g)) for _ in range(M)]
    loss = lambda lg, t: F.cross_entropy(lg.float().reshape(-1, V), t.reshape(-1))
    # stock: tied parameter, autograd sums both uses
    gs = []
    for x, t in batches:
        gs.append(torch.autograd.grad(loss(m(input_ids=x, use_cache=False).logits, t), [W])[0])
    stock = gs[0].clone()
    for gg in gs[1:]: stock.add_(gg)
    # separate uses: alias leaves
    P1 = nn.Parameter(W.detach().clone()); P2 = nn.Parameter(W.detach().clone())
    E.weight = P1; H.weight = P2
    es, hs = [], []
    for x, t in batches:
        e, h = torch.autograd.grad(loss(m(input_ids=x, use_cache=False).logits, t), [P1, P2])
        es.append(e); hs.append(h)
    sum_ok = all(torch.equal(gg, h + e) for gg, h, e in zip(gs, hs, es))
    # untouched rows of e_i are +0 exactly
    plus0 = True
    for (x, _), e in zip(batches, es):
        mask = torch.ones(V, dtype=torch.bool); mask[x.unique()] = False
        u = e[mask]
        plus0 &= bool((u == 0).all() and not torch.signbit(u).any())
    # per-use fold
    se = es[0].clone(); sh = hs[0].clone()
    for e in es[1:]: se.add_(e)
    for h in hs[1:]: sh.add_(h)
    peruse = se + sh
    res = {}
    for npend in (1, 2, 3, M):
        acc = torch.empty_like(W); init = torch.zeros(V, dtype=torch.bool)
        def apply(rows, vals):
            new = rows[~init[rows]]; old = rows[init[rows]]
            if len(new): acc[new] = vals[~init[rows]]
            if len(old): acc[old] = acc[old] + vals[init[rows]]
            init[rows] = True
        pending = []  # (i, rows W_i, h_i[W_i], T_i)
        maxrows = 0
        events = []
        for i in range(M):
            events.append(("h", i))
            if i - npend + 1 >= 0: events.append(("e", i - npend + 1))
        for j in range(max(0, M - npend + 1), M): events.append(("e", j))
        arrived = {}
        for kind, i in events:
            if kind == "h":
                T = batches[i][0].unique()
                P = torch.zeros(V, dtype=torch.bool)
                for (_, R, _, _) in pending: P[R] = True
                P[T] = True
                wait = P.nonzero().flatten(); now = (~P).nonzero().flatten()
                h = hs[i]
                apply(now, h[now] + 0.0)
                pending.append((i, wait, h[wait].clone(), T)); maxrows = max(maxrows, sum(len(p[1]) for p in pending))
            else:
                arrived[i] = es[i]
                while pending and pending[0][0] in arrived:
                    j, R, hv, T = pending.pop(0)
                    e = arrived.pop(j)
                    apply(R, hv + e[R])  # e[R] is +0 on rows of R outside T_j: (h + 0.0)
        assert not pending and init.all()
        res[npend] = (torch.equal(acc, stock), maxrows)
    ndiff = int((peruse != stock).sum())
    print(f"{mt:12s} V={V:5d} {str(dtype)[6:]:8s} g_i==h_i+e_i:{sum_ok} untouched_e_plus0:{plus0} "
          f"rowsparse_equal(npend:equal,max_pending_rows):{res} stock_sha={sha(stock)} "
          f"peruse_vs_stock_differ={ndiff}/{stock.numel()}", flush=True)

for mt, extra in (("gemma2", dict(sliding_window=8, attn_logit_softcapping=50.0, final_logit_softcapping=30.0, query_pre_attn_scalar=16)),
                  ("llama", {}), ("qwen3", {})):
    for V in (128, 4096):
        for dt in (torch.float32, torch.bfloat16):
            run(mt, extra, V, dt)
