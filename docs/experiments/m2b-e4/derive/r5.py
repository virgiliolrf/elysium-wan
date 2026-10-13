# ADR-0006 round-5 corrections (accounting only; no mechanism changes). Loads the round-4 model (round4.md §9,
# sha256 d4838233...dd0c0, which loads round3.md §7 and derivation.md §9) and adds:
#  (1) S1 on host DRAM per GPU and per host, per stage, with the stage's own c_s = 8*P^F_s*T_mb/F (GEMM params only),
#      exactly as SCHEDULER-I16 writes S1: k*(b_g*P_s*share + A) per microbatch, share 2(w-1)/w, A both directions;
#  (2) every E7 candidate re-priced per host (stage-contiguous packing: plan GPU g on host g // gph), the host's DRAM
#      time summed over the GPUs actually on it (no averaging over hosts), pieces admitted only where S1 holds on
#      every host and on the NIC, and S2 holds;
#  (3) the one-step host-memory bound with per-unit b_g and the gather term, per peer and summed per host;
#  (4) tok and pos merged into one Embed unit (SCHEDULER-I17 as specified) at point B stage 0;
#  (5) E4 at scaled B: the exposure model is invariant under the scaling map, checked numerically.
# Every input is HYPOTHESIS as in rounds 3 and 4.
import sys
# usage: python3 -I r5.py r4.py r3.py r2.py src/wantrain/scheduler/zero_bubble.py
src4 = open(sys.argv[1]).read()
sys.argv = [sys.argv[0], sys.argv[2], sys.argv[3], sys.argv[4]]
PRE = src4[: src4.index('print("== (1)')]
exec(PRE)
DEFS = PRE[PRE.index("def block_units"):]

G9 = 1e9
def dram_rate(r, w, k, kind):
    """S1 on host DRAM for one GPU of a stage, as I16 writes it: k*(b_g*P_s*2(w-1)/w + A_both) / c_s, bytes/s."""
    if w == 1:
        per_mb = 0.0
    else:
        per_mb = sum(u[2] * u[1] for u in r["us"]) * (w - 1) / w      # one direction, per microbatch
    A = 4 * 4096 * h if kind == "mid" else 2 * 4096 * h                 # one direction, per microbatch
    return k * 2 * (per_mb + A) / r["c"]

print("== (1) S1 on host DRAM per GPU, per stage, c_s from GEMM params (P^F), 600 GB/s host")
for tag, S_, w_, M_ in (("point B (128 H100, S 32, w 4)", 32, 4, 128), ("32 H100, S 16, w 2", 16, 2, 32)):
    r = plan(S_, w_, M_, links["400G"])
    for key, x in r["stages"].items():
        cP = 8 * x["P"] * 4096 / F_H100
        row = []
        for kp in (2, 4, 6, 10):
            rt = dram_rate(x, w_, kp, key[0])
            row.append(f"k {kp:2d}: {rt/G9:6.1f} GB/s ({HOSTBW/rt:4.2f} GPUs/host)")
        kmax = HOSTBW / dram_rate(x, w_, 1, key[0])
        print(f"{tag} {key[0]:5s} L {key[1]}: c_s {x['c']*1e3:5.2f} ms (with all params {cP*1e3:5.2f}); " + "; ".join(row)
              + f"; 1-GPU host admits k <= {kmax:.2f}")

def per_peer_bytes(r, w, Mw, kind):
    """per peer j: bytes per step in, out (exchange gradients + gathers + boundaries)."""
    us, own = r["us"], r["own"]; A = (4 if kind == "mid" else 2) * 4096 * h
    out = []
    for j in range(w):
        if w == 1:
            out.append((Mw * A, Mw * A)); continue
        ing = sum(us[u][2] * us[u][1] * own[u][j] * (w - 1) for u in range(len(us)))
        eg = sum(us[u][2] * us[u][1] * (1 - own[u][j]) for u in range(len(us)))
        gin = sum(us[u][3] * us[u][1] * (1 - own[u][j]) for u in range(len(us)))
        geg = sum(us[u][3] * us[u][1] * own[u][j] * (w - 1) for u in range(len(us)))
        out.append((Mw * ing + gin + Mw * A, Mw * eg + geg + Mw * A))
    return out

def candidate(S_, w_, M_, cap, gph, kp, ngpu=None, **kw):
    r = plan(S_, w_, M_, cap, **kw)
    Mw = M_ // w_
    gpus = []   # (stage index, peer, kind, stage result)
    for s, (kind, nb) in enumerate(r["kinds"]):
        x = r["stages"][(kind, nb)]
        pb = per_peer_bytes(x, w_, Mw, kind)
        for j in range(w_):
            gpus.append((s, j, kind, x, pb[j]))
    hosts = {}
    for g, item in enumerate(gpus):
        hosts.setdefault(g // gph, []).append(item)
    host_t = 0.0; s1_host = True; worst = None
    for hid, items in hosts.items():
        t = sum(kp * (pb[0] + pb[1]) for (_, _, _, _, pb) in items) / HOSTBW
        rate = sum(dram_rate(x, w_, kp, kind) for (_, _, kind, x, _) in items)
        if t > host_t:
            host_t = t
        if rate > HOSTBW:
            s1_host = False
            if worst is None or rate > worst[1]:
                worst = (hid, rate, sorted(set(s for (s, *_r) in items)))
    s2 = r["E"] <= EPS * r["C"]
    return dict(tot=r["tot"], host_t=host_t, step=max(r["tot"], host_t), s1_nic=r["s1"], s1_host=s1_host, s2=s2,
                worst=worst, E=r["E"], C=r["C"])

print("== (2) E7 candidates, per host (stage-contiguous packing), step = max(lane + E + T_ovh, busiest host's DRAM time)")
def e7(tag, cands, cap, configs):
    for gph, kp in configs:
        best = None; lines = []
        for name, S_, w_, M_, kw in cands:
            x = candidate(S_, w_, M_, cap, gph, kp, **kw)
            pieces = kw.get("lay", "pieces") == "pieces" and w_ > 1
            adm = (x["s1_nic"] and x["s1_host"] and x["s2"]) if pieces else True
            why = "" if adm else ("refused: " + ", ".join(n for n, ok in (("S1 NIC", x["s1_nic"]), ("S1 host DRAM", x["s1_host"]), ("S2", x["s2"])) if not ok))
            if x["worst"] and pieces:
                why += f" (busiest host {x['worst'][1]/G9:.0f} GB/s, stages {x['worst'][2][:3]})"
            lines.append(f"    {name:30s}: lane+E+T_ovh {x['tot']*1e3:7.1f}, host {x['host_t']*1e3:7.1f} -> step {x['step']*1e3:7.1f} ms  {'admitted' if adm else why}")
            if adm and (best is None or x["step"] < best[0]):
                best = (x["step"], name)
        print(f"  {tag}, {gph} GPU/host, k {kp}: planner choice {best[1]} at {best[0]*1e3:.1f} ms")
        for ln in lines:
            print(ln)

Bc = [("S 32 w 4 pieces", 32, 4, 128, dict(lay="pieces")),
      ("S 32 w 4 zero1 + native", 32, 4, 128, dict(lay="stride", barrier=True)),
      ("S 32 w 4 zero1 today (fp32)", 32, 4, 128, dict(lay="stride", wire="fp32", flush="stage", barrier=True, order="bwd")),
      ("S 16 w 8 pieces", 16, 8, 128, dict(lay="pieces")),
      ("S 16 w 8 zero1 + native", 16, 8, 128, dict(lay="stride", barrier=True)),
      ("S 32 w 1 on 32 GPUs", 32, 1, 128, {})]
e7("128 H100, 400 Gbps", Bc, links["400G"], ((1, 10), (1, 6), (1, 2), (8, 10), (8, 2)))
Sc = [("S 32 w 1", 32, 1, 32, {})] + [(f"S {S_} w {w_} {nm}", S_, w_, 32, kw) for S_, w_ in ((16, 2), (8, 4), (4, 8))
                                      for nm, kw in (("pieces", dict(lay="pieces")), ("zero1 + native", dict(lay="stride", barrier=True)))]
e7("32 H100, 400 Gbps", Sc, links["400G"], ((1, 10), (2, 10), (4, 10), (8, 10), (8, 2)))
e7("32 H100, 100 Gbps", Sc, links["100G"], ((1, 10), (8, 10)))

print("== (3) one-step host-memory bound per peer, per-unit b_g, plus one step of gathers out and in: "
      "sum_u b_g,u P_u [M_k(w-1)/w + (M-M_k)/w] + 2 sum_u b_p,u P_u (w-1)/w")
def host_bound(us, w, M, Mk):
    g = sum(u[2] * u[1] for u in us) * (Mk * (w - 1) / w + (M - Mk) / w)
    p = 2 * sum(u[3] * u[1] for u in us) * (w - 1) / w
    return g, p
for tag, kind, L, w_, M_, Mk in (("A interior (S 8, w 4, M 32), M_k 8", "mid", 4, 4, 32, 8),
                                  ("A interior, M_k 12 of 32", "mid", 4, 4, 32, 12),
                                  ("B interior (S 32, w 4, M 128), M_k 32", "mid", 1, 4, 128, 32),
                                  ("B stage 0 (Embed + Block 0), M_k 32", "first", 1, 4, 128, 32),
                                  ("B last (Block 31 + Head), M_k 32", "last", 1, 4, 128, 32)):
    us = stage_units(kind, L)
    g, p = host_bound(us, w_, M_, Mk)
    print(f"  {tag:40s}: gradients {g/G9:6.2f} GB + gathers {p/G9:5.2f} GB = {(g+p)/G9:6.2f} GB per peer")
usB0 = stage_units("first", 1); usBi = stage_units("mid", 1)
g0, p0 = host_bound(usB0, 4, 128, 32); gi, pi = host_bound(usBi, 4, 128, 32)
print(f"  8-GPU host at B holding stage 0 and stage 1 (4 + 4 peers): {4*(g0+p0)/G9 + 4*(gi+pi)/G9:.1f} GB; "
      f"8 stage-0-sized peers would need {8*(g0+p0)/G9:.1f} GB")
print("  own-piece hold on the device, B/param (b_g * M_k / w) at B: bf16 units "
      f"{2*32/4:.0f}, fp32 Embed tables {4*32/4:.0f}; at A {2*8/4:.0f}")

print("== (4) point B stage 0 with tok and pos as ONE fp32 Embed unit (SCHEDULER-I17), 400 Gbps")
_base_units = stage_units
def stage_units_merged(kind, L):
    us = _base_units(kind, L)
    if kind == "first":
        tok, pos = us[0], us[1]
        us = [("embed", tok[1] + pos[1], 4, 4, 0, True, 0)] + us[2:]
    return us
rB = plan(32, 4, 128, links["400G"])
C_B = rB["C"]
saved = stage_units
for nm, fn in (("tok, pos separate (round 4)", saved), ("tok + pos one unit (as specified)", stage_units_merged)):
    stage_units = fn
    for order in ("fwd", "bwd"):
        x = stage4("first", 1, 4, 4096, 32, links["400G"], order=order, C_pipe=C_B)
        print(f"  {nm:34s} {order}: X {x['X']*1e3:6.2f} ms, E {x['E']*1e3:6.2f} ms ({100*x['E']/C_B:4.2f}% of C), eps*C {EPS*C_B*1e3:.1f}")
    for nm2, kw in (("step 1 alone", dict(lay="stride", barrier=True)), ("LPT", dict(lay="lpt"))):
        x = stage4("first", 1, 4, 4096, 32, links["400G"], C_pipe=C_B, **kw)
        print(f"     {nm2:12s}: E {x['E']*1e3:6.1f} ms")
stage_units = saved

print("== (5) E4 at scaled B: h' 256, V' 2000 (V/h 7.8125), seq' 256, T_mb' 256, M/w 32; F' 50 GFLOP/s;")
print("   cap, PCIe and HBM scaled by (F'/F)(T_mb/T_mb'), chunk by (h'/h)^2: every ratio of the model kept")
NS = {}
def scaled(hh, VV, seq, Tmb, Fp):
    g = dict(globals())
    s_bytes = (hh / 4096) ** 2; s_rate = (Fp / 400e12) * (4096 / Tmb)
    g.update(h=hh, V=VV, F_COPY=max(1, int(16 * 2**20 * s_bytes)), HBM=3.35e12 * s_rate,
             DEV={"H100": (Fp, 80e9, 50e9 * s_rate)})
    exec(DEFS.replace("SEQ * h", f"{seq} * h"), g)
    return g, s_rate
for (hh, VV, seq, Tmb, Fp) in ((4096, 32000, 4096, 4096, 400e12), (256, 2000, 256, 256, 50e9)):
    g, sr = scaled(hh, VV, seq, Tmb, Fp)
    cap = links["400G"] * sr
    us_last = g["stage_units"]("last", 1); cl = 8 * sum(u[4] for u in us_last) * Tmb / Fp
    Cp = 32 * cl
    out = []
    for nm, kw in (("pieces", dict(lay="pieces")), ("step 1 alone", dict(lay="stride", barrier=True)), ("LPT", dict(lay="lpt"))):
        x = g["stage4"]("first", 1, 4, Tmb, 32, cap, C_pipe=Cp, **kw)
        x2 = g["stage4"]("first", 1, 4, Tmb, 32, cap, **kw)
        out.append(f"{nm}: E {x['E']*1e3:7.2f} ms = {100*x['E']/Cp:5.2f}% of C_B (C_B {Cp*1e3:6.1f}); "
                   f"own-period arm E {x2['E']*1e3:7.2f} ms = {100*x2['E']/x2['C']:5.2f}% of its C {x2['C']*1e3:6.1f}; util {x['util']:.2f}")
    for wi, Mw_ in ((4, 8), (8, 4)):
        for nm, kw in (("pieces", dict(lay="pieces")), ("step 1 alone", dict(lay="stride", barrier=True)), ("LPT", dict(lay="lpt"))):
            x = g["stage4"]("mid", 4, wi, Tmb, Mw_, cap, **kw)
            out.append(f"interior L 4 w {wi}, M/w {Mw_}, {nm}: E {x['E']*1e3:7.2f} ms = {100*x['E']/x['C']:5.2f}% of C {x['C']*1e3:6.1f}")
    print(f"  h {hh}, cap {cap*8/1e9:8.3f} Gbps, HBM {g['HBM']/1e9:8.2f} GB/s, chunk {g['F_COPY']} B, c_stage0 "
          f"{8*sum(u[4] for u in g['stage_units']('first',1))*Tmb/Fp*1e3:.2f} ms:")
    for o in out:
        print("     " + o)
