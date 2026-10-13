# m2b-e4 PLAN: the bf16 block's F/B/W op times when w peer processes compute at once on this machine (E4 runs w
# localhost peers, each one intra-op thread). Each process builds the same B' block as profile_fprime.py and reports
# its medians; the slowest peer sets the lockstep rate. usage: uv run python .../profile_concurrent.py <w> [reps]
import json
import multiprocessing as mp
import statistics
import sys
import time

H, V, SEQ, NHEAD = 256, 2000, 256, 4


def worker(i, reps, q, start):
    import torch
    from wantrain.model import gpt, wgrad
    from wantrain.runtime.determinism import setup_determinism
    setup_determinism(0)
    cfg = gpt.GPTConfig(vocab_size=V, n_layer=1, n_head=NHEAD, d_model=H, seq_len=SEQ,
                        compute_dtype="bf16", boundary_dtype="bf16")
    blk = gpt.build_layers(cfg, seed=0)[1]
    x0 = torch.randn(1, SEQ, H).to(torch.bfloat16)
    start.wait()
    tf, tb, tw = [], [], []
    for r in range(reps + 3):
        t0 = time.perf_counter()
        with torch.no_grad():
            blk(x0)
        t1 = time.perf_counter()
        x = x0.clone().requires_grad_(True)
        y = blk(x)
        with wgrad.capture() as store:
            torch.autograd.backward(y, torch.randn_like(y))
        st = list(store)
        t2 = time.perf_counter()
        wgrad.flush(st)
        t3 = time.perf_counter()
        if r >= 3:
            tf.append(t1 - t0); tb.append(t2 - t1); tw.append(t3 - t2)
    m = lambda a: statistics.median(a) * 1e3
    q.put(dict(peer=i, F_ms=m(tf), B_ms=m(tb), W_ms=m(tw), c_ms=m(tf) + m(tb) + m(tw)))


if __name__ == "__main__":
    w = int(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    ctx = mp.get_context("spawn"); q = ctx.Queue(); start = ctx.Barrier(w)
    ps = [ctx.Process(target=worker, args=(i, reps, q, start)) for i in range(w)]
    for p in ps:
        p.start()
    res = sorted((q.get() for _ in ps), key=lambda r: r["peer"])
    for p in ps:
        p.join()
    cs = [r["c_ms"] for r in res]
    PF = 12 * H * H
    print(json.dumps(dict(w=w, reps=reps, peers=res, c_max_ms=max(cs), c_min_ms=min(cs), skew=max(cs) / min(cs),
                          Fprime_slowest_flops=8 * PF * 256 / (max(cs) / 1e3)), indent=1))
