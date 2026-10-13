# ADR-0006 round-2 cost model. Analytic; every input is HYPOTHESIS unless noted. Recompute on:
# 8 FLOPs/param/token per stage: F 2, B 4 (recompute F + dX), W 2  => c_F = c/4, c_B = c/2, c_W = c/4.
# Exchange units are SplitLinear modules in forward order: qkv 3h^2, proj h^2, fc1 4h^2, fc2 4h^2 per block
# (LayerNorm and biases ride with the next module; they are < 0.01% of the bytes).
import heapq
G9 = 1e9
DEV = {"H100": (400e12, 80e9, 50e9), "RTX4090": (50e12, 24e9, 25e9)}  # F, device bytes, PCIe B/s per direction
EPS = 0.03
h = 4096; V = 32000; NBLK = 32; P = 12 * h * h * NBLK
MOD = [("qkv", 3), ("proj", 1), ("fc1", 4), ("fc2", 4)]
F_COPY = 16 * 2**20  # copy chunk (bytes)
def reserve(mem): return max(0.12 * mem, 2**30)

def units(L):
    return [(f"b{b}.{n}", s * h * h) for b in range(L) for n, s in MOD]

def own_pieces(us, w): return [[1.0 / w] * w for _ in us]

def own_stride(L, w):
    # today's params[k::w] over the stage's 12-entry-per-block list (gpt.py:255-265); weights at 2, 4, 8, 10
    idx = {"qkv": 2, "proj": 4, "fc1": 8, "fc2": 10}
    out = []
    for b in range(L):
        for n, _ in MOD:
            row = [0.0] * w; row[(12 * b + idx[n]) % w] = 1.0; out.append(row)
    return out

def own_lpt(L, w):
    ts = sorted([(s, i) for i, (_, s) in enumerate([(n, s) for b in range(L) for n, s in MOD])], key=lambda x: (-x[0], x[1]))
    load = [0] * w; out = [[0.0] * w for _ in range(4 * L)]
    for s, i in ts:
        j = load.index(min(load)); load[j] += s; out[i][j] = 1.0
    return out

def own_flat(us, w):
    tot = sum(p for _, p in us); edges = [tot * j / w for j in range(w + 1)]; out = []; a = 0.0
    for _, p in us:
        b = a + p
        out.append([max(0.0, min(b, edges[j + 1]) - max(a, edges[j])) / p for j in range(w)]); a = b
    return out

def exch(us, own, w, cap, c, k=1, rtt=0.0, order="fwd", barrier=False, flush="module", bg=2, bp=2,
         lat=0.0, sites=None, cap_up=None):
    """Exposed step-boundary time D of one sharded stage (lockstep peers), discrete-event.
    Queues: each peer's egress and ingress and, across sites, the uplink of each direction. A transfer is served
    on each queue it crosses independently (decoupled bottlenecks: optimistic for concentrated layouts, so it
    favours the alternatives) and finishes when the last of them finishes. Each queue is non-preemptive and serves
    the released transfer whose unit is needed first in the next forward (P3-style priority; gathers before
    gradients of the same unit). k trailing W ops; a unit's gathers are released when the owner's fold of that unit
    is complete (with barrier: when every fold of the stage is complete)."""
    U = len(us); Ptot = sum(p for _, p in us)
    cW = cF = c / 4
    seq = list(range(U)) if order == "fwd" else list(range(U - 1, -1, -1))
    prod = {}
    for j in range(k):
        t = j * cW
        for u in seq:
            t += cW * us[u][1] / Ptot
            prod[(j, u)] = t if flush == "module" else (j + 1) * cW
    def queues(s, d):
        q = [("e", s), ("i", d)]
        if sites and sites[s] != sites[d]:
            q.append(("up", sites[s], sites[d]))
        return q
    def dur(q, nbytes):
        return nbytes / (cap_up if q[0] == "up" else cap)
    items = []  # [rel, prio, kind, u, s, d, j, bytes, served{q: fin}]
    pend = {}   # queue -> list of item indices not yet served there
    def add(rel, kind, u, s, d, j, nbytes):
        it = [rel, (u, kind, j, (d - s) % w, s), kind, u, s, d, j, nbytes, {}]
        items.append(it); i = len(items) - 1
        for q in queues(s, d):
            pend.setdefault(q, []).append(i)
    for (j, u), t in prod.items():
        for s in range(w):
            for d in range(w):
                if s != d and own[u][d] > 0:
                    add(t, 1, u, s, d, j, bg * us[u][1] * own[u][d])
    need_g = {(u, d): (w - 1 if own[u][d] > 0 else 0) for u in range(U) for d in range(w)}
    got = {}
    upd = {(u, d): prod[(k - 1, u)] for u in range(U) for d in range(w) if own[u][d] > 0}
    ready = {(u, d): 0.0 for u in range(U) for d in range(w)}
    owners_total = sum(1 for x in need_g if need_g[x] > 0)
    done_owners = []
    def gathers(u, o, t):
        for d in range(w):
            if d != o:
                add(t, 0, u, o, d, -1, bp * us[u][1] * own[u][o])
    free = {}
    while True:
        best = None
        for q, lst in pend.items():
            if not lst:
                continue
            rel = min(items[i][0] for i in lst)
            t = max(free.get(q, 0.0), rel)
            if best is None or t < best[0] - 1e-15:
                best = (t, q)
        if best is None:
            break
        t, q = best
        cand = [i for i in pend[q] if items[i][0] <= t + 1e-15]
        i = min(cand, key=lambda x: items[x][1])
        pend[q].remove(i)
        it = items[i]
        fin = t + dur(q, it[7]); free[q] = fin; it[8][q] = fin
        if len(it[8]) < len(queues(it[4], it[5])):
            continue
        arr = max(it[8].values()) + rtt / 2 + lat
        kind, u, s, d, j = it[2], it[3], it[4], it[5], it[6]
        if kind == 0:
            ready[(u, d)] = max(ready[(u, d)], arr)
            continue
        if j != k - 1:
            continue
        upd[(u, d)] = max(upd[(u, d)], arr)
        got[(u, d)] = got.get((u, d), 0) + 1
        if got[(u, d)] == need_g[(u, d)]:
            if not barrier:
                gathers(u, d, upd[(u, d)])
            else:
                done_owners.append((u, d))
                if len(done_owners) == owners_total:
                    tb = max(upd.values())
                    for (uu, oo) in done_owners:
                        gathers(uu, oo, tb)
    end = k * cW
    D = 0.0
    for d in range(w):
        Dd = 0.0; fcum = 0.0
        for u in range(U):
            own_ready = upd[(u, d)] if own[u][d] > 0 else 0.0
            rdy = max(ready[(u, d)], own_ready if own[u][d] > 0 else 0.0)
            if barrier:
                rdy = max(rdy, max(upd.values()))
            need = end + Dd + fcum
            if rdy > need:
                Dd += rdy - need
            fcum += cF * us[u][1] / Ptot
        D = max(D, Dd)
    return D

def lam_of(own, us, w):
    Pt = sum(p for _, p in us)
    ing = max(sum((w - 1) * own[u][j] * us[u][1] for u in range(len(us))) for j in range(w)) / Pt
    eg = max(sum((1 - own[u][j]) * us[u][1] for u in range(len(us))) for j in range(w)) / Pt
    return max(ing, eg) / ((w - 1) / w)

def stage(dev, L, w, Tmb, Mw, cap_nic, lay="pieces", k=1, rtt=0.0, order="fwd", barrier=False, flush="module",
          bg=2, bp=2, sites=None, cap_up=None, A=None, A_up=0.0, pcie=True):
    F, mem, cpcie = DEV[dev]
    us = units(L); Ps = sum(p for _, p in us)
    own = {"pieces": own_pieces(us, w), "stride": own_stride(L, w), "lpt": own_lpt(L, w), "flat": own_flat(us, w)}[lay]
    cap = min(cap_nic, cpcie) if pcie else cap_nic
    lat = 2 * F_COPY / cpcie if pcie else 0.0
    c = 8 * Ps * Tmb / F; C = Mw * c
    X = exch(us, own, w, cap, c, k, rtt, order, barrier, flush, bg, bp, lat, sites, cap_up)
    A = 4 * Tmb * h if A is None else A
    lam = lam_of(own, us, w)
    if sites:  # binding uplink: n_a*n_b/w of every unit per direction
        na = sum(1 for s in sites if s == 0); nb = w - na
        share, cb = na * nb / w, cap_up
        A = A_up
    else:
        share, cb = (w - 1) / w * lam, cap
    G = bg * Ps * share / cb
    s1 = G + A / cb <= c
    Vb = Mw * (bg * Ps * share + A) + bp * Ps * share
    linkb = Vb / cb + rtt + c / 2 + (c / 4) / len(us) - C
    return dict(c=c, C=C, X=X, G=G, s1=s1, linkb=linkb, E=max(X, linkb), Ps=Ps, lam=lam)

def tmin(dev, L, w, Tmb, cap_nic, **kw):
    """Smallest tokens per peer per step with E <= eps*C (window_s = 0, conservative). X does not depend on M/w;
    linkb is linear in M/w, so one simulation and a scan over M/w suffice."""
    r1 = stage(dev, L, w, Tmb, 1, cap_nic, **kw)
    if not r1["s1"]:
        return None, None
    r2 = stage(dev, L, w, Tmb, 2, cap_nic, **kw)
    slope = r2["linkb"] - r1["linkb"]; base = r1["linkb"] - slope
    for Mw in range(1, 10**6):
        if max(r1["X"], base + slope * Mw) <= EPS * Mw * r1["c"]:
            return Mw * Tmb, Mw
    return float("inf"), None

links = {"10G": 10e9 / 8, "100G": 100e9 / 8, "400G": 400e9 / 8}
Ps8 = P / 8
print(f"P={P/G9:.3f}B  P_s(S=8)={Ps8/G9:.4f}B")

print("== worked point A (layout comparison at fixed S 8, w 4): 7B, M 32, T_mb 4096, H100, 400G NIC, PCIe Gen5 50 GB/s")
cap = links["400G"]
rows = [
    ("today: stride, fp32, whole-stage flush, barrier", dict(lay="stride", bg=4, bp=4, flush="stage", barrier=True, order="bwd")),
    ("step 1 alone: stride, native, per-module flush, barrier", dict(lay="stride", barrier=True)),
    ("step 1 + per-unit updates (no barrier), stride", dict(lay="stride")),
    ("LPT whole tensors, native, per-unit", dict(lay="lpt")),
    ("flat element ranges, native, per-unit", dict(lay="flat")),
    ("pieces, backward-order emission (round 1)", dict(lay="pieces", order="bwd")),
    ("pieces, forward-order emission (round 2)", dict(lay="pieces")),
]
for ww in (4, 8):
    Mw = 32 // ww
    for name, kw in rows:
        r = stage("H100", 4 if ww == 4 else 4, ww, 4096, Mw, cap, **kw)
        print(f"w={ww} {name:58s} S1={r['s1']!s:5s} lam={r['lam']:.2f} X={r['X']*1e3:6.1f} linkb={r['linkb']*1e3:7.1f} E={r['E']*1e3:6.1f} ms ({100*r['E']/r['C']:5.1f}% of C={r['C']*1e3:.1f})")
r = stage("H100", 4, 4, 4096, 8, cap, pcie=False, order="bwd")
print(f"check vs round 1 (pieces, bwd, no PCIe, block units differ): X={r['X']*1e3:.1f} ms")
r = stage("H100", 4, 4, 4096, 8, cap, k=2)
print(f"pieces fwd k=2: X={r['X']*1e3:.1f} ms")

print("== synchronous host copies on the compute thread (today's .cpu() / .to(device)), per step, worked point A")
Mw, ww, bg, bp = 8, 4, 2, 2
d2h = Mw * (ww - 1) / ww * bg * Ps8 + bp * Ps8 / ww
h2d = Mw * (ww - 1) / ww * bg * Ps8 + (ww - 1) / ww * bp * Ps8
C = Mw * 8 * Ps8 * 4096 / 4e14
for name, bw in (("PCIe Gen4 25 GB/s", 25e9), ("PCIe Gen5 50 GB/s", 50e9)):
    t = (d2h + h2d) / bw
    print(f"{name}: D2H {d2h/G9:.2f} GB + H2D {h2d/G9:.2f} GB = {t*1e3:.0f} ms per step = {100*t/C:.0f}% of C ({C*1e3:.1f} ms)")
print(f"async chunked copies: added latency per hop 2*f/PCIe = {2*F_COPY/50e9*1e3:.2f} ms (Gen5)")

print("== depth vs width at 32 H100 (bubble from the lane solve, see depth.py) + exposure (pieces, fwd, 400G / 100G)")
bub = {(32, 1): 48.4, (16, 2): 46.9, (8, 4): 43.8, (4, 8): 37.5}
for ln in ("100G", "400G"):
    for (SS, ww), b in bub.items():
        LL = 32 // SS
        if ww == 1:
            print(f"{ln} S={SS:2d} w={ww}: bubble {b:5.1f}% + exposed  0.0% = {b:5.1f}%"); continue
        r = stage("H100", LL, ww, 4096, 32 // ww, links[ln])
        e = 100 * r["E"] / r["C"]
        print(f"{ln} S={SS:2d} w={ww}: bubble {b:5.1f}% + exposed {e:5.1f}% = {b+e:5.1f}%  (S1 {r['s1']})")

print("== worked point B (width forced: 7B on 128 H100, S 32 one block per stage, w 4, M 128, T_mb 4096)")
for ln in ("100G", "400G"):
    for name, kw in rows[:2] + rows[3:4] + rows[6:]:
        r = stage("H100", 1, 4, 4096, 32, links[ln], **kw)
        print(f"{ln} {name:58s} S1={r['s1']!s:5s} X={r['X']*1e3:6.1f} E={r['E']*1e3:7.1f} ms ({100*r['E']/r['C']:5.1f}% of C={r['C']*1e3:.1f})")

print("== memory per peer, device B/param")
def mem_today(w, d): return 4 + 12 / w + 4 + 4 * d
def mem_new(w, L): return 2 + 16 / w + 2 * (4 / 12) / L
for ww in (2, 4, 8, 16):
    print(f"w={ww:2d}: narrow 20.0 | today {mem_today(ww, (ww-1)/ww):5.2f} | proposed {mem_new(ww, 4):5.2f} | host worst (hold+backlog) 2*b_g*(M/w)*(w-1)/w")
print("== largest dense layer (12h^2) as its own stage, device GB = 2P + 16P/w + 2*4h^2 (today: 4P+12P/w+4P+4P(w-1)/w)")
for name, hh in (("405B", 16384), ("1T", 24576)):
    p = 12 * hh * hh
    for ww in (1, 2, 4, 8):
        today = (4 * p + 12 * p / ww + 4 * p + 4 * p * (ww - 1) / ww) if ww > 1 else 20 * p
        print(f"{name} w={ww}: proposed {(2*p + 16*p/ww + 2*4*hh*hh)/G9:6.1f} GB | today {today/G9:6.1f} GB")

print("== per-NIC T_min (tokens/peer/step, w 8, eps 3%, k 1, RTT 0, S 8): fwd order; round-1 bwd in brackets")
for d in DEV:
    for ln, cp in links.items():
        row = []
        for T in (4096, 16384, 65536):
            a, _ = tmin(d, 4, 8, T, cp)
            b, _ = tmin(d, 4, 8, T, cp, order="bwd")
            row.append("BW" if a is None else f"{a/1e6:.3f}M [{b/1e6:.3f}M]")
        print(f"{d:8s} {ln:5s}: " + " | ".join(row))

print("== cross-site: two equal sites, one 10 Gbps uplink, NICs 100G, RTT 50 ms. The uplink is shared by every")
print("   stage that straddles it: 'one' = only this stage straddles; 'all' = every stage straddles (cap_up/S each)")
cap_up = 10e9 / 8
for d, (F, memb, _) in DEV.items():
    avail = memb - reserve(memb)
    for n in (1, 2, 4):
        ww = 2 * n; sites = [0] * n + [1] * n
        for L in (1, 2, 4):
            S = NBLK // L; Ps = 12 * h * h * L
            for mode in ("one", "all"):
                cu = cap_up if mode == "one" else cap_up / S
                Au = ww * 2 * T * h if mode == "one" else 0.0
                row = []
                for T in (16384, 65536, 131072, 163840):
                    Au = ww * 2 * T * h if mode == "one" else 0.0
                    r1 = stage(d, L, ww, T, 1, links["100G"], sites=sites, cap_up=cu, rtt=0.05, A_up=Au)
                    if not r1["s1"]:
                        row.append("S1"); continue
                    tp, Mw = tmin(d, L, ww, T, links["100G"], sites=sites, cap_up=cu, rtt=0.05, A_up=Au)
                    if Mw is None:
                        row.append("E"); continue
                    infl = min(S, Mw)
                    dev_b = 2 * Ps + 16 * Ps / ww + 2 * 4 * h * h + infl * T * h * 2 + L * T * 16 * h * 4
                    host_b = 2 * 2 * Mw * (ww - 1) / ww * Ps
                    tag = "MEM" if dev_b > avail else "ok"
                    row.append(f"{tp/1e6:.2f}M({ww*tp/1e6:.1f}M){tag}:{dev_b/G9:.0f}GB,host{host_b/G9:.0f}")
                print(f"{d:8s} w={ww} L_s={L} S={S:2d} {mode:3s}: T_mb 16k/64k/128k/160k: " + "  ".join(row))
print("head stage logits at V 32k, fp32 value + grad, per microbatch: " + ", ".join(f"T_mb {T//1024}k {2*T*V*4/G9:.1f} GB" for T in (16384, 65536, 131072)))

print("== emission order where RTT matters (X in ms, fwd vs bwd; window_s = 0)")
for d, L, ww, T, cp, rtt, tag in (("H100", 4, 4, 4096, links["400G"], 0.0, "worked point A, RTT 0"),
                                  ("H100", 4, 4, 16384, links["400G"], 0.010, "H100 w4 S8 T_mb 16k, 400G, RTT 10 ms"),
                                  ("RTX4090", 4, 4, 16384, links["100G"], 0.050, "RTX4090 w4 S8 T_mb 16k, 100G, RTT 50 ms")):
    a = stage(d, L, ww, T, 8, cp, rtt=rtt); b = stage(d, L, ww, T, 8, cp, rtt=rtt, order="bwd")
    print(f"{tag}: fwd X={a['X']*1e3:.1f}  bwd X={b['X']*1e3:.1f}  (c_W={a['c']/4*1e3:.1f}, G={a['G']*1e3:.1f})")
sites = [0, 1]
for T in (16384, 65536):
    a = stage("RTX4090", 1, 2, T, 1, links["100G"], sites=sites, cap_up=10e9/8, rtt=0.05, A_up=2*2*T*h)
    b = stage("RTX4090", 1, 2, T, 1, links["100G"], sites=sites, cap_up=10e9/8, rtt=0.05, A_up=2*2*T*h, order="bwd")
    print(f"RTX4090 cross-site one-stage w2 L_s1 T_mb {T//1024}k: fwd X={a['X']*1e3:.1f}  bwd X={b['X']*1e3:.1f} ms (c_W={a['c']/4*1e3:.1f})")
print("== host worst case (owner hold + send backlog) = 2*b_g*(M/w)*(w-1)/w*P_s")
for Mw, ww, Ps in ((8, 4, Ps8), (32, 4, P / 32), (4, 8, Ps8), (256, 4, Ps8)):
    print(f"M/w={Mw} w={ww} P_s={Ps/G9:.3f}B: {2*2*Mw*(ww-1)/ww:.0f} B/param = {2*2*Mw*(ww-1)/ww*Ps/G9:.1f} GB per peer")
