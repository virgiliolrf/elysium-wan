# m2b-e4 PLAN: profile F' (deterministic-kernel rate of the stage's GEMMs at the scaled shapes) and the fold rate
# (HBM'-equivalent: fp32 add_ throughput, and the bf16-widen + add the runtime folds with) on this machine's CPU.
# Uses the repository's own kernels: wantrain.model.gpt (Embed, Block, Head built by build_layers, SplitLinear via
# wgrad.convert), wgrad.capture/flush for the deferred W, and runtime.determinism.setup_determinism (1 intra-op thread,
# deterministic algorithms), exactly as a stage runs. No product file is written.
# usage: uv run python docs/experiments/m2b-e4/derive/profile_fprime.py [reps]
import json
import platform
import statistics
import sys
import time

import torch

from wantrain.model import gpt, wgrad
from wantrain.runtime.determinism import setup_determinism

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 9
WARM = 3
H, V, SEQ, TMB = 256, 2000, 256, 256  # scaled B' (round5.md section 6); T_mb' = 1 x 256 tokens
NHEAD = 4


def med(fn):
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def stage_ops(compute):
    setup_determinism(0)
    cfg = gpt.GPTConfig(vocab_size=V, n_layer=1, n_head=NHEAD, d_model=H, seq_len=SEQ,
                        compute_dtype=compute, boundary_dtype=compute)
    layers = gpt.build_layers(cfg, seed=0)
    emb, blk, head = layers[0], layers[1], layers[2]
    adt = cfg.compute_torch_dtype()
    idx = torch.randint(0, V, (1, SEQ))
    x0 = torch.randn(1, SEQ, H).to(adt)
    out = {}

    # interior block: F, B (dX chain, dW deferred), W (deferred dW GEMMs)
    def block_F():
        with torch.no_grad():
            blk(x0)

    state = {}

    def block_B():
        x = x0.clone().requires_grad_(True)
        y = blk(x)
        g = torch.randn_like(y)
        with wgrad.capture() as store:
            torch.autograd.backward(y, g)
        state["store"] = list(store)

    def block_W():
        if "store" not in state:
            block_B()
        wgrad.flush(state.pop("store"))

    tF = med(block_F)
    # B includes its own forward (recompute F + dX), as the model's c_B = c/2 assumes (8 FLOPs: F 2, B 4, W 2)
    tB = med(block_B)
    ws = []
    for _ in range(WARM):
        block_B(); block_W()
    for _ in range(REPS):
        block_B()
        t0 = time.perf_counter(); block_W(); ws.append(time.perf_counter() - t0)
    tW = (statistics.median(ws), min(ws), max(ws))
    PF = 12 * H * H  # GEMM params (weights) of one block
    c = tF[0] + tB[0] + tW[0]
    out["block"] = dict(F_ms=tF[0] * 1e3, B_incl_recompute_ms=tB[0] * 1e3, W_ms=tW[0] * 1e3,
                        c_FBW_ms=c * 1e3, PF=PF, Fprime_eff_flops=8 * PF * TMB / c,
                        split=dict(F=tF[0] / c, B=tB[0] / c, W=tW[0] / c),
                        minmax_ms=dict(F=[tF[1] * 1e3, tF[2] * 1e3], B=[tB[1] * 1e3, tB[2] * 1e3], W=[tW[1] * 1e3, tW[2] * 1e3]))

    # Embed (stage 0's tables): F + B producing the fp32 tok/pos gradients (0 FLOP-params in the model)
    def emb_FB():
        for p in emb.parameters():
            p.grad = None
        y = emb(idx)
        y.backward(torch.randn_like(y))
    out["embed_FB_ms"] = med(emb_FB)[0] * 1e3

    # Head: F + B + W (8 FLOPs per param per token in the model)
    xh = torch.randn(1, SEQ, H).to(adt)

    def head_FBW():
        x = xh.clone().requires_grad_(True)
        with wgrad.capture() as store:
            y = head(x)
            torch.autograd.backward(y, torch.randn_like(y))
        wgrad.flush(list(store))
    th = med(head_FBW)[0]
    out["head_FBW_ms"] = th * 1e3
    out["head_Fprime_eff_flops"] = 8 * V * H * TMB / th
    return out


def gemm_rate(dt):
    """pure GEMMs of the block's four SplitLinears at T_mb' 256: Y = X W^T, dX = dY W, dW = dY^T X (6 FLOPs/param/token)"""
    setup_determinism(0)
    shapes = [(H, 3 * H), (H, H), (H, 4 * H), (4 * H, H)]
    mats = [(torch.randn(TMB, i).to(dt), torch.randn(o, i).to(dt), torch.randn(TMB, o).to(dt)) for i, o in shapes]

    def run():
        for x, w, gy in mats:
            x.mm(w.t()); gy.mm(w); gy.t().mm(x)
    t = med(run)[0]
    return 6 * 12 * H * H * TMB / t, t


def fold_rates():
    setup_determinism(0)
    n = (V + SEQ) * H // 4 + 12 * H * H // 4  # one peer's piece of stage 0 at w 4 (~0.34 M elements)
    acc = torch.randn(n)
    g32 = torch.randn(n)
    g16 = torch.randn(n).to(torch.bfloat16)
    t32 = med(lambda: acc.add_(g32))[0]
    t16 = med(lambda: acc.add_(g16.float()))[0]
    p = torch.nn.Parameter(torch.randn(n)); p.grad = torch.randn(n)
    opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.0)
    ta = med(opt.step)[0]
    return dict(n=n, fp32_add_GBps=12 * n / t32 / 1e9, fp32_add_ms=t32 * 1e3,
                bf16_widen_add_GBps=18 * n / t16 / 1e9, bf16_widen_add_ms=t16 * 1e3,
                adamw_GBps_at_30B=30 * n / ta / 1e9, adamw_ms=ta * 1e3)


if __name__ == "__main__":
    res = dict(machine=platform.machine(), python=platform.python_version(), torch=torch.__version__,
               threads=None, reps=REPS, warmup=WARM, shapes=dict(h=H, V=V, seq=SEQ, T_mb=TMB, n_head=NHEAD))
    for comp in ("bf16", "fp32"):
        res[comp] = stage_ops(comp)
        g, t = gemm_rate(torch.bfloat16 if comp == "bf16" else torch.float32)
        res[comp]["pure_gemm_flops"] = g
        res[comp]["pure_gemm_ms"] = t * 1e3
    res["threads"] = torch.get_num_threads()
    res["fold"] = fold_rates()
    print(json.dumps(res, indent=1, sort_keys=True))
