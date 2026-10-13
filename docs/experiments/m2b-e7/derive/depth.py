# Depth vs width through the repository's own solver (zero_bubble.py), per-peer op time.
import importlib.util, sys
spec = importlib.util.spec_from_file_location("zb", sys.argv[1]); zb = importlib.util.module_from_spec(spec); sys.modules["zb"] = zb; spec.loader.exec_module(zb)
h = 4096; P = 12 * h * h * 32; F = 4e14; Tmb = 4096
def run(S, w, M, k):
    Ps = P / S; L = 32 // S
    c = 8 * Ps * Tmb / F
    cf, cb, cw = c / 4, c / 2, c / 4                     # per-peer, recompute on
    bnd = Tmb * h * 2; wpin = L * Tmb * 16 * h * 4         # SPEC-scheduler memory model
    cfg = zb.ZBConfig(S, M, [cf / w] * S, [cb / w] * S, [cw / w] * S, comm=[0.0] * (S - 1),
                      mem_f=[float(bnd)] * S, mem_b=[float(wpin)] * S, mem_w=[-float(bnd + wpin)] * S)
    if S == 1:
        sched = zb.zero_bubble(cfg, "1p")
    else:
        sched = zb.zero_bubble(zb._with_pending_budget(cfg, k), "given")
    owner = [zb.iwrr(M, [1.0] * w) for _ in range(S)]
    orders = zb.slot_orders(sched.orders, owner, [w] * S)
    _e, ms = zb.simulate_slots(cfg, orders, owner, [[1.0] * w for _ in range(S)])
    C = (M / w) * c
    return C, ms, 100 * (ms / C - 1), 100 * (S - 1) * w / (2 * M)
for M in (32, 128):
    for S, w in ((32, 1), (16, 2), (8, 4), (4, 8)):
        for k in (1, 2):
            C, ms, over, closed = run(S, w, M, k)
            print(f"M={M:3d} S={S:2d} w={w} ZB-{k}p: C={C*1e3:7.1f} ms makespan={ms*1e3:7.1f} ms bubble={over:5.1f}% (closed form (S-1)w/2M {closed:5.1f}%)")
print("== lane solve: one ZB schedule for (S, M/w) at per-peer op time, every lane the same")
def lane(S, w, M, k):
    Ps = P / S; L = 32 // S; c = 8 * Ps * Tmb / F; m = M // w
    bnd = Tmb * h * 2; wpin = L * Tmb * 16 * h * 4
    cfg = zb.ZBConfig(S, m, [c / 4] * S, [c / 2] * S, [c / 4] * S, comm=[0.0] * (S - 1),
                      mem_f=[float(bnd)] * S, mem_b=[float(wpin)] * S, mem_w=[-float(bnd + wpin)] * S)
    sched = zb.zero_bubble(cfg, "1p") if S == 1 else zb.zero_bubble(zb._with_pending_budget(cfg, k), "given")
    _e, ms = zb.simulate(cfg, sched.orders)
    C = m * c
    return C, ms, 100 * (ms / C - 1)
for M in (32, 128):
    for S, w in ((32, 1), (16, 2), (8, 4), (4, 8)):
        r = [lane(S, w, M, k) for k in (1, 2)]
        print(f"M={M:3d} S={S:2d} w={w}: lane ZB-1p bubble={r[0][2]:5.1f}%  ZB-2p bubble={r[1][2]:5.1f}%")
