import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
# ADR-0009 round 1, A5 probe: the export-based alternative (torch.export, the front end of torch.distributed.pipelining's
# pipeline()/SplitPoint). Per family of the E1 corpus (proto.py's FAM, fp32, eager, train mode): does the whole
# ForCausalLM export, and does the exported module give the vanilla model's logits and every gradient bit for bit?
import os, sys, time
sys.argv = [sys.argv[0]]
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto2.py")).read().split("if __name__ != '__main__'")[0]
exec(SRC)
for mt in FAM:
    cfg = CONFIG_MAPPING[mt](**{**BASE, **FAM[mt]}); cfg._attn_implementation = "eager"
    for k, v in list(vars(cfg).items()):  # dropout 0 as the recipe requires (D5); the first run left gpt2's at 0.1
        if ("dropout" in k or "pdrop" in k) and isinstance(v, float): setattr(cfg, k, 0.0)
    torch.manual_seed(0); m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).train()
    V = cfg.vocab_size; g = torch.Generator().manual_seed(1)
    x = torch.randint(0, V, (2, 24), generator=g); t = torch.randint(0, V, (2, 24), generator=g)
    t0 = time.time()
    try:
        ep = torch.export.export(m, (), {"input_ids": x, "use_cache": False}, strict=False)
        em = ep.module()
    except Exception as e:
        print(f"{mt:12s} EXPORT-FAIL {type(e).__name__}: {str(e)[:110]}", flush=True); continue
    ps = [p for p in m.parameters()]
    lr = m(input_ids=x, use_cache=False).logits
    gr = torch.autograd.grad(F.cross_entropy(lr.float().reshape(-1, V), t.reshape(-1)), ps)
    try:
        le = em(input_ids=x, use_cache=False).logits
        eps_ = dict(em.named_parameters())
        names = [n for n, _ in m.named_parameters()]
        ge = torch.autograd.grad(F.cross_entropy(le.float().reshape(-1, V), t.reshape(-1)), [eps_[n] for n in names], allow_unused=True)
        nd = sum(1 for a, b in zip(gr, ge) if b is None or not torch.equal(a, b))
        x2 = torch.randint(0, V, (2, 40), generator=g)
        try: em(input_ids=x2, use_cache=False); dyn = "S40 ok"
        except Exception as e: dyn = f"S40 {type(e).__name__}"
        print(f"{mt:12s} exported in {time.time()-t0:.1f}s logits_equal={torch.equal(lr, le)} grads_differ={nd}/{len(gr)} {dyn}", flush=True)
    except Exception as e:
        print(f"{mt:12s} RUN-FAIL {type(e).__name__}: {str(e)[:110]}", flush=True)
