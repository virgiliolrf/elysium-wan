# ADR-0006 round-3 corrections. Reuses the round-2 model (derivation.md §9, sha256 f43ab4f6...b701eb), whose
# definitions are loaded from the file given as argv[1] (everything above its `links = ` line), and adds:
#  (1) a coupled fluid exchange model "as built": per-destination send queues (need-first or FIFO), each sending
#      one chunk of <= 16 MiB at a time; egress, ingress and uplink capacity shared max-min fairly among the active
#      chunks (TCP-like), so a receiver cannot prioritize its ingress;
#  (2) the owner's device fold and per-unit AdamW as op times (gather released after them) and in T_ovh;
#  (3) the own-piece hold on the device, the per-peer host bound under weighted IWRR;
#  (4) host DRAM as a link (k passes per exchanged byte per side, g GPUs per host);
#  (5) point-B alternatives and sub-block depth.
# Every input is HYPOTHESIS: H100 HBM 3.35 TB/s, host DRAM 600 GB/s (2-socket DDR5), as in derivation.md for the rest.
import sys
src = open(sys.argv[1]).read()
exec(src[: src.index("links = ")])
links = {"10G": 10e9 / 8, "100G": 100e9 / 8, "400G": 400e9 / 8}
HBM = 3.35e12
FOLD_B = lambda bg: (bg + 4 if bg < 4 else 0) + 12  # widen (read b_g, write 4) unless fp32; add: read 4 + 4, write 4
ADAM_B = lambda bp: 4 * 7 + bp          # read p, g, m, v; write p, m, v; write the cast copy

def exch3(us, own, w, cap, c, k=1, rtt=0.0, order="fwd", barrier=False, flush="module", bg=2, bp=2, lat=0.0,
          sites=None, cap_up=None, policy="need", ops=True, chunk=F_COPY):
    U = len(us); Ptot = sum(p for _, p in us); cW = cF = c / 4
    seq = list(range(U)) if order == "fwd" else list(range(U - 1, -1, -1))
    prod = {}
    for j in range(k):
        t = j * cW
        for u in seq:
            t += cW * us[u][1] / Ptot
            prod[(j, u)] = t if flush == "module" else (j + 1) * cW
    items = []                       # [kind, u, s, d, j, nchunks_left]
    queues = {}                      # (s, d) -> list of [rel, key, rem, item]
    nseq = [0]
    def add(rel, kind, u, s, d, j, nbytes):
        items.append([kind, u, s, d, j, 0]); i = len(items) - 1
        n = max(1, -(-int(nbytes) // chunk)); rem = nbytes
        for x in range(n):
            b = min(chunk, rem); rem -= b
            key = (u, kind, j, x) if policy == "need" else (rel, nseq[0])
            nseq[0] += 1
            queues.setdefault((s, d), []).append([rel, key, b, i])
        items[i][5] = n
    for (j, u), t in prod.items():
        for s in range(w):
            for d in range(w):
                if s != d and own[u][d] > 0:
                    add(t, 1, u, s, d, j, bg * us[u][1] * own[u][d])
    need_g = {(u, d): (w - 1 if own[u][d] > 0 else 0) for u in range(U) for d in range(w)}
    got = {}; upd = {(u, d): prod[(k - 1, u)] for u in range(U) for d in range(w) if own[u][d] > 0}
    ready = {(u, d): 0.0 for u in range(U) for d in range(w)}
    owners_total = sum(1 for x in need_g if need_g[x] > 0); done_owners = []
    def optime(u, o):
        if not ops:
            return 0.0
        n = us[u][1] * own[u][o]
        return n * FOLD_B(bg) / HBM + n * ADAM_B(bp) / HBM
    def gathers(u, o, t):
        for d in range(w):
            if d != o:
                add(t, 0, u, o, d, -1, bp * us[u][1] * own[u][o])
    def res(s, d):
        r = [("e", s), ("i", d)]
        if sites and sites[s] != sites[d]:
            r.append(("up", sites[s], sites[d]))
        return r
    def capof(r):
        return cap_up if r[0] == "up" else cap
    active = {}                      # (s, d) -> chunk
    t = 0.0
    while True:
        for q, lst in queues.items():
            if q not in active and lst:
                rel = [x for x in lst if x[0] <= t + 1e-15]
                if rel:
                    x = min(rel, key=lambda y: y[1]); lst.remove(x); active[q] = x
        if not active:
            nxt = [x[0] for lst in queues.values() for x in lst]
            if not nxt:
                break
            t = min(nxt); continue
        # max-min fair rates (progressive filling)
        flows = list(active.keys()); rate = {}; left = {}
        rs = {}
        for q in flows:
            for r in res(*q):
                rs.setdefault(r, []).append(q)
        capl = {r: capof(r) for r in rs}
        unfixed = set(flows)
        while unfixed:
            best = None
            for r, fl in rs.items():
                uf = [f for f in fl if f in unfixed]
                if not uf:
                    continue
                share = capl[r] / len(uf)
                if best is None or share < best[0]:
                    best = (share, r, uf)
            share, r, uf = best
            for f in uf:
                rate[f] = share; unfixed.discard(f)
                for r2 in res(*f):
                    capl[r2] -= share
        dt = min(active[q][2] / rate[q] for q in flows)
        pend = [x[0] for lst in queues.values() for x in lst if x[0] > t + 1e-15]
        if pend and min(pend) - t < dt:
            dt = min(pend) - t
        t += dt
        for q in flows:
            active[q][2] -= rate[q] * dt
        for q in flows:
            x = active[q]
            if x[2] <= 1e-6:
                del active[q]
                it = items[x[3]]; it[5] -= 1
                if it[5] > 0:
                    continue
                arr = t + rtt / 2 + lat
                kind, u, s, d, j = it[0], it[1], it[2], it[3], it[4]
                if kind == 0:
                    ready[(u, d)] = max(ready[(u, d)], arr); continue
                if j != k - 1:
                    continue
                upd[(u, d)] = max(upd[(u, d)], arr)
                got[(u, d)] = got.get((u, d), 0) + 1
                if got[(u, d)] == need_g[(u, d)]:
                    if not barrier:
                        gathers(u, d, upd[(u, d)] + optime(u, d))
                    else:
                        done_owners.append((u, d))
                        if len(done_owners) == owners_total:
                            tb = max(upd.values()) + max(optime(uu, oo) for uu, oo in done_owners)
                            for (uu, oo) in done_owners:
                                gathers(uu, oo, tb)
    end = k * cW; D = 0.0
    for d in range(w):
        Dd = 0.0; fcum = 0.0
        for u in range(U):
            rdy = ready[(u, d)]
            if own[u][d] > 0:
                rdy = max(rdy, upd[(u, d)] + optime(u, d))
            if barrier:
                rdy = max(rdy, max(upd.values()))
            need = end + Dd + fcum
            if rdy > need:
                Dd += rdy - need
            fcum += cF * us[u][1] / Ptot
        D = max(D, Dd)
    return D

def stage3(dev, L, w, Tmb, Mw, cap_nic, lay="pieces", policy="need", ops=True, **kw):
    F, mem, cpcie = DEV[dev]
    us = units(L); Ps = sum(p for _, p in us)
    own = {"pieces": own_pieces(us, w), "stride": own_stride(L, w), "lpt": own_lpt(L, w), "flat": own_flat(us, w)}[lay]
    r2 = stage(dev, L, w, Tmb, Mw, cap_nic, lay=lay, **kw)
    cap = min(cap_nic, cpcie); lat = 2 * F_COPY / cpcie
    pk = {x: kw[x] for x in ("k", "rtt", "order", "barrier", "flush", "bg", "bp") if x in kw}
    X = exch3(us, own, w, cap, r2["c"], lat=lat, policy=policy, ops=ops, **pk)
    E = max(X, r2["linkb"])
    return dict(r2=r2, X=X, E=E, C=r2["C"], s1=r2["s1"], Ps=Ps)

rows = [
    ("today: stride, fp32, whole-stage flush, barrier", dict(lay="stride", bg=4, bp=4, flush="stage", barrier=True, order="bwd")),
    ("step 1 alone: stride, native, per-module flush, barrier", dict(lay="stride", barrier=True)),
    ("step 1 + per-unit updates (no barrier), stride", dict(lay="stride")),
    ("LPT whole tensors, native, per-unit", dict(lay="lpt")),
    ("flat element ranges, native, per-unit", dict(lay="flat")),
    ("pieces, backward emission", dict(lay="pieces", order="bwd")),
    ("pieces, forward emission", dict(lay="pieces")),
]
print("== (1)+(2) exposure E (ms, % of C): round-2 model | as built: need-first per-destination queues, fair-shared"
      " egress/ingress | FIFO per-destination queues; fold and AdamW op times included in both 'as built' columns")
for tag, L, ww, Mw, ln in (("A w4", 4, 4, 8, "400G"), ("A w8", 4, 8, 4, "400G"), ("B   ", 1, 4, 32, "400G"),
                           ("B 100G", 1, 4, 32, "100G")):
    for name, kw in rows:
        if tag.startswith("B") and name.startswith(("step 1 +", "flat", "pieces, back")):
            continue
        a = stage3("H100", L, ww, 4096, Mw, links[ln], policy="need", **kw)
        b = stage3("H100", L, ww, 4096, Mw, links[ln], policy="fifo", **kw)
        C = a["C"]
        print(f"{tag} {name:56s} r2 {a['r2']['E']*1e3:7.1f} ({100*a['r2']['E']/C:5.1f}%) | need {a['E']*1e3:7.1f} "
              f"({100*a['E']/C:5.1f}%) | fifo {b['E']*1e3:7.1f} ({100*b['E']/C:5.1f}%)  S1={a['s1']}")

print("== (2) T_ovh per step per peer: fold M*(P_s/w)*(b_g+16)/HBM (widen + add), AdamW (P_s/w)*30/HBM, event records 1 ms")
for tag, Ps, ww, M, bg in (("A (S 8, w 4) pieces", P / 8, 4, 32, 2), ("B (S 32, w 4) pieces", P / 32, 4, 128, 2),
                           ("S 32, w 1 (narrow: bf16 dW widened, folded)", P / 32, 1, 32, 2)):
    fold = M * Ps / ww * FOLD_B(bg) / HBM; adam = Ps / ww * ADAM_B(2) / HBM
    C = (M / ww) * 8 * Ps * 4096 / 4e14
    print(f"{tag:40s} fold {fold*1e3:5.1f} ms ({100*fold/C:4.1f}%)  AdamW {adam*1e3:4.1f} ms  T_ovh {(fold+adam+1e-3)*1e3:5.1f} ms "
          f"({100*(fold+adam+1e-3)/C:4.1f}% of C={C*1e3:.1f})")
print("depth vs width at 32 H100, 400G, M 32 (bubble + exposure + T_ovh), exposure from the as-built model:")
for SS, ww, b in ((32, 1, 48.4), (16, 2, 46.9), (8, 4, 43.8), (4, 8, 37.5)):
    LL = 32 // SS; Ps = P / SS; M = 32
    C = (M / ww) * 8 * Ps * 4096 / 4e14
    bg = 2
    tov = 100 * (M * Ps / ww * FOLD_B(bg) / HBM + Ps / ww * ADAM_B(2) / HBM + 1e-3) / C
    e = 0.0 if ww == 1 else 100 * stage3("H100", LL, ww, 4096, M // ww, links["400G"])["E"] / C
    print(f"  S {SS:2d} w {ww}: {b:5.1f} + {e:4.1f} + {tov:4.1f} = {b+e+tov:5.1f}%")

print("== (3) device memory per peer, B/param: state | + own-piece hold, one-step worst case b_g*(M_k)/w")
def dev_state(w, L): return 2 + 16 / w + 2 * (4 / 12) / L
for tag, ww, Mw, L in (("A w4", 4, 8, 4), ("A w8", 8, 4, 4), ("B w4", 4, 32, 1), ("w2 M/w 8", 2, 8, 4)):
    s = dev_state(ww, L); own_ = 2 * Mw / ww
    today_state = 4 + 12 / ww + 4; today_hold = 4 * (ww * Mw - 1) / ww
    print(f"{tag}: proposed {s:5.2f} | +own {s+own_:5.2f}   today (ZeRO-1, replica+owned+flush) {today_state:5.2f} | "
          f"+fp32 hold on device, one step {today_state+today_hold:6.2f}")
print("largest dense layer as its own stage (T_mb 4096, L_s 1): device GB = 2P + 16P/w + 2*4h^2 + pending W"
      " + boundaries + own hold 2*(M/w)*P/w; available 70.4 GB")
avail = 80e9 - reserve(80e9)
for name, hh in (("405B", 16384), ("1T", 24576)):
    p = 12 * hh * hh
    for ww in (2, 4, 8):
        base = 2 * p + 16 * p / ww + 2 * 4 * hh * hh + 4096 * 16 * hh * 4
        per = 2 * p / ww
        mx = 0
        for Mw in range(1, 513):
            if base + min(32, Mw) * 4096 * hh * 2 + per * Mw <= avail:
                mx = Mw
        print(f"{name} w {ww}: state+module+pending W {base/G9:5.1f} GB, own hold {per/G9:5.2f} GB per M/w; "
              f"fits with M/w <= {mx}  (M/w 8: {(base + 8*4096*hh*2 + per*8)/G9:5.1f} GB)")
print("sub-block cut, 1T: fc1 alone (4h^2) at w 1, 20 B/param:", f"{4*24576**2*20/G9:.1f} GB")

print("== (3) host bound per peer under weighted IWRR: b_g*P_s*[M_k*(w-1)/w + (M-M_k)/w]")
for M, ww, Mk in ((32, 4, 8), (32, 4, 12), (32, 4, 4)):
    hb = Mk * (ww - 1) / ww + (M - Mk) / ww; old = 2 * (M / ww) * (ww - 1) / ww
    print(f"M {M} w {ww} M_k {Mk}: {hb:5.2f} b_g*P_s (uniform bound {old:5.2f}; {100*(hb/old-1):+.0f}%); own hold M_k/w = {Mk/ww:.2f} b_g*P_s")
for tag, Ps, Mw in (("A", P / 8, 8), ("B", P / 32, 32)):
    print(f"host bound, uniform IWRR, point {tag}: {2*2*Mw*3/4*Ps/G9:.1f} GB")

print("== (4) host DRAM as a link: per GPU per step NIC egress+ingress bytes x k passes per byte per side")
for tag, Ps, Mw in (("A", P / 8, 8), ("B", P / 32, 32)):
    w_ = 4; eg = Mw * (w_ - 1) / w_ * 2 * Ps + (w_ - 1) / w_ * 2 * Ps; C = Mw * 8 * Ps * 4096 / 4e14
    for kp in (2, 4, 10):
        per = kp * 2 * eg / C
        print(f"point {tag}: NIC {eg/G9:.2f} GB/dir/step; k {kp:2d}: {per/1e9:6.1f} GB/s per GPU; 8 GPUs {8*per/1e9:7.1f} GB/s;"
              f" GPUs per host at 600 GB/s: {600e9/per:4.1f}")
    # host-bound period if 8 GPUs share one 600 GB/s host, k 10 and k 2
    for kp in (2, 10):
        Tm = 8 * kp * 2 * eg / 600e9
        print(f"   8 GPUs, k {kp}: host link needs {Tm*1e3:.0f} ms per step against C {C*1e3:.1f} ms")
print("today's transport passes per exchanged byte (counted in framing.py/serial.py): sender D2H 1 + tobytes 2 +"
      " serial join 2 + sendall concat 2 + kernel copy 2 + NIC 1 = 10; receiver NIC 1 + recv 2 + bytearray 2 + bytes() 2"
      " + np copy 2 + H2D 1 = 10 (+2 when fragmented)")

print("== (5) sub-block depth at point B (M 128): beta = (S-1)*w/(2M)")
for SS, ww in ((32, 4), (64, 2), (128, 1)):
    print(f"S {SS:3d} w {ww}: beta {100*(SS-1)*ww/256:5.1f}%")
