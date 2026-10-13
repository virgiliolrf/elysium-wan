# m2b-e4 PLAN: E4's predictions re-pinned at the F' and op split profiled on this machine (ADR-0006 F6, handout P3).
# Loads the round-4 model exactly as round5.md section 8 does (r4.py sha256 d4838233...dd0c0, which loads r3.py
# acf1cb86...995ed6 and r2.py f43ab4f6...b701eb, and runs the repository's zero_bubble.py 22988afc...751ac8d), keeps
# every definition, and changes only what the profile measured:
#  (a) the op split: the model's c_F = c_W = c/4, c_B = c/2 becomes the measured fractions FF, FB, FW of
#      c = F + B + W (t_gap = c_B + c_W/U, derivation.md section 2, follows);
#  (b) the fold and AdamW rates: the CPU's measured rates (round5.md section 6: "where the CPU's fold rate cannot be
#      paced, the PLAN re-runs this model at the measured HBM' and pins that output"; no fold-pacing seam exists);
#  (c) two arms the round-4 code had no switch for: a shared inter-site uplink inside exch4 (round-3 exch3's resource
#      model: an uplink resource per direction shared max-min fairly with the NICs) and the F5 tree arm (residue-group
#      fold G = w: one fp32 partial per unit piece per peer, released by the last W, nothing streamed before);
#  (d) the FIFO policy is passed through to exch4 (it existed there; stage4 did not expose it).
# Step 0 checks that with the round-5 split and fold the patched code reproduces round5.md section 6 exactly.
# usage: python3 -I r6_e4.py r4.py r3.py r2.py zero_bubble.py profile_fprime.out conc_w4.out conc_w8.out
import hashlib
import json
import sys

A0 = list(sys.argv)
for p in A0[1:]:
    print(f"input {p.split('/')[-1]}: sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
PROF = json.load(open(A0[5])); C4 = json.load(open(A0[6])); C8 = json.load(open(A0[7]))
src4 = open(A0[1]).read()
sys.argv = [A0[0], A0[2], A0[3], A0[4]]
PRE = src4[: src4.index('print("== (1)')]
exec(PRE)
DEFS = PRE[PRE.index("def block_units"):]

PATCHES = [
    ("cW = cF = c / 4", "cW = c * FW; cF = c * FF"),
    ("return n * FOLD_B(us[u][2]) / HBM + n * ADAM_B(us[u][3]) / HBM",
     "return n * FOLD_B(us[u][2]) / FOLD_RATE(us[u][2]) + n * ADAM_B(us[u][3]) / ADAM_RATE"),
    ('for r in (("e", q[0]), ("i", q[1])):', "for r in RES(q[0], q[1]):"),
    ("capl = {r: cap for r in rs}", 'capl = {r: (CAP_UP if r[0] == "up" else cap) for r in rs}'),
    ('for r2 in (("e", f[0]), ("i", f[1])):', "for r2 in RES(f[0], f[1]):"),
]
PD = DEFS
for old, new in PATCHES:
    assert PD.count(old) == 1, (old, PD.count(old))
    PD = PD.replace(old, new)


def RES_factory(g):
    def RES(s, d):
        r = [("e", s), ("i", d)]
        if g["SITES"] is not None and g["SITES"][s] != g["SITES"][d]:
            r.append(("up", g["SITES"][s], g["SITES"][d]))
        return r
    return RES


def env(hh, VV, seq, Tmb, Fp, split=(0.25, 0.5, 0.25), fold=None):
    """round5.md section 6's scaling map; split = (FF, FB, FW); fold = (rate for 2-byte wire at 18 B/elem,
    rate for fp32 wire at 12 B/elem, AdamW rate at 30 B/elem) in B/s, or None for the model's HBM' (paced)."""
    g = dict(globals())
    s_bytes = (hh / 4096) ** 2; s_rate = (Fp / 400e12) * (4096 / Tmb)
    hbm = 3.35e12 * s_rate
    g.update(h=hh, V=VV, F_COPY=max(1, int(16 * 2**20 * s_bytes)), HBM=hbm, DEV={"H100": (Fp, 80e9, 50e9 * s_rate)},
             FF=split[0], FB=split[1], FW=split[2], SITES=None, CAP_UP=None, S_RATE=s_rate, TMB=Tmb, FP=Fp)
    if fold is None:
        g.update(FOLD_RATE=lambda bg: hbm, ADAM_RATE=hbm)
    else:
        g.update(FOLD_RATE=lambda bg: fold[0] if bg < 4 else fold[1], ADAM_RATE=fold[2])
    g["RES"] = RES_factory(g)
    exec(PD.replace("SEQ * h", f"{seq} * h"), g)
    return g


def e4arm(g, kind, L, w, Mw, cap_nic=None, C_pipe=None, lay="pieces", wire="native", barrier=False, flush="module",
          order="fwd", rtt=0.0, policy="need", sites=None, cap_up=None, tree=False, lat=None):
    """stage4 (round 4) with the patches: explicit cap and hop latency, sites, tree. Returns ms-free dict (seconds)."""
    Fp, _mem, cpcie = g["DEV"]["H100"]; Tmb = g["TMB"]; h_ = g["h"]
    us = g["stage_units"](kind, L)
    if wire == "fp32":
        us = [(n, p, 4, 4, fp, b, ix) for (n, p, bg, bp, fp, b, ix) in us]
    own = g["own_of"](us, w, lay)
    if cap_nic is None:
        cap_nic = links["400G"] * g["S_RATE"]
    cap = min(cap_nic, cpcie)
    lat = 2 * g["F_COPY"] / cpcie if lat is None else lat
    Pfl = sum(u[4] for u in us); c = 8 * Pfl * Tmb / Fp
    C = Mw * c if C_pipe is None else C_pipe
    A = 4 * Tmb * h_ if kind == "mid" else 2 * Tmb * h_
    U = len(us); tgap = c * g["FB"] + c * g["FW"] / U
    g["SITES"], g["CAP_UP"] = sites, cap_up
    if tree:
        ust = [(n, p, 4, bp, fp, b, ix) for (n, p, bg, bp, fp, b, ix) in us]   # one fp32 partial per unit per peer
        X = g["exch4"](ust, own, w, cap, c, rtt=rtt, order=order, barrier=barrier, flush=flush, lat=lat, policy=policy)
        g["SITES"] = None
        return dict(c=c, C=C, X=X, E=X, linkb=0.0, util=0.0, s1=True)
    X = g["exch4"](us, own, w, cap, c, rtt=rtt, order=order, barrier=barrier, flush=flush, lat=lat, policy=policy)
    g["SITES"] = None
    ing = [sum(us[u][2] * us[u][1] * own[u][j] * (w - 1) for u in range(U)) for j in range(w)]
    eg = [sum(us[u][2] * us[u][1] * (1 - own[u][j]) for u in range(U)) for j in range(w)]
    gin = [sum(us[u][3] * us[u][1] * (1 - own[u][j]) for u in range(U)) for j in range(w)]
    geg = [sum(us[u][3] * us[u][1] * own[u][j] * (w - 1) for u in range(U)) for j in range(w)]
    per_mb = max(max(ing), max(eg))
    Vmax = max(max(Mw * ing[j] + gin[j], Mw * eg[j] + geg[j]) for j in range(w)) + Mw * A
    linkb = Vmax / cap + rtt + tgap - C
    util = (per_mb + A) / cap / c; s1 = util <= 1.0
    out = dict(c=c, C=C, X=X, linkb=linkb, util=util, s1=s1, cap=cap, lat=lat, per_mb=per_mb, A=A, tgap=tgap)
    if sites is not None:
        na = sum(1 for s in sites if s == sites[0]); nb = w - na; share = na * nb / w
        Vup = Mw * sum(us[u][2] * us[u][1] * share for u in range(U)) + sum(us[u][3] * us[u][1] * share for u in range(U))
        linkb_up = Vup / cap_up + rtt + tgap - C
        util_up = sum(us[u][2] * us[u][1] * share for u in range(U)) / cap_up / c
        out.update(linkb_up=linkb_up, util_up=util_up, s1_up=util_up <= 1.0)
        linkb = max(linkb, linkb_up)
    out["E"] = max(X, linkb)
    return out


def pct(x):
    return 100 * x["E"] / x["C"]


# ---------------------------------------------------------------- step 0: reproduce round5.md section 6
print("== (0) regression: round-5 split (1:2:1) and HBM' fold at F' 50 GFLOP/s and full B reproduce round5.md section 6")
R5 = {"s0 pieces": 2.82, "s0 step1": 71.31, "s0 LPT": 20.09, "w4 pieces": 3.66, "w4 step1": 16.00, "w8 pieces": 10.03,
      "w8 LPT": 25.10}
for (hh, VV, seq, Tmb, Fp) in ((4096, 32000, 4096, 4096, 400e12), (256, 2000, 256, 256, 50e9)):
    g = env(hh, VV, seq, Tmb, Fp)
    cl = 8 * sum(u[4] for u in g["stage_units"]("last", 1)) * Tmb / Fp; CB = 32 * cl
    got = {}
    for nm, kw in (("pieces", dict(lay="pieces")), ("step1", dict(lay="stride", barrier=True)), ("LPT", dict(lay="lpt"))):
        a = e4arm(g, "first", 1, 4, 32, C_pipe=CB, **kw)
        b = g["stage4"]("first", 1, 4, Tmb, 32, links["400G"] * g["S_RATE"], C_pipe=CB, **kw)
        assert abs(a["E"] - b["E"]) < 1e-9, (nm, a["E"], b["E"])
        got["s0 " + nm] = pct(a)
    for wi, Mw_, nms in ((4, 8, ("pieces", "step1")), (8, 4, ("pieces", "LPT"))):
        for nm in nms:
            kw = {"pieces": dict(lay="pieces"), "step1": dict(lay="stride", barrier=True), "LPT": dict(lay="lpt")}[nm]
            got[f"w{wi} {nm}"] = pct(e4arm(g, "mid", 4, wi, Mw_, **kw))
    for k_, v in R5.items():
        assert abs(round(got[k_], 2) - v) < 0.006, (hh, k_, got[k_], v)
    print(f"  h {hh}: " + ", ".join(f"{k_} {got[k_]:.2f}" for k_ in R5) + f"  (C_B {CB*1e3:.1f} ms) -> reproduced")

# ---------------------------------------------------------------- the profile
PF = 12 * 256 * 256; T = 256
def split_of(conc):
    n = len(conc["peers"]); cs = [p["c_ms"] for p in conc["peers"]]
    f = [sum(p[k] / p["c_ms"] for p in conc["peers"]) / n for k in ("F_ms", "B_ms", "W_ms")]
    return tuple(x / sum(f) for x in f), max(cs) / 1e3
SPL = {}; FPR = {}
for w_, conc in ((4, C4), (8, C8)):
    sp, cmax = split_of(conc)
    SPL[w_] = sp; FPR[w_] = 8 * PF * T / cmax
fold = PROF["fold"]
FOLD = (fold["bf16_widen_add_GBps"] * 1e9, fold["fp32_add_GBps"] * 1e9, fold["adamw_GBps_at_30B"] * 1e9)
print("== (1) profile (MEASURED, this machine): bf16 recipe, one block at h' 256, T_mb' 256, 1 intra-op thread")
b1 = PROF["bf16"]["block"]
print(f"  single process: F {b1['F_ms']:.2f} B {b1['B_incl_recompute_ms']:.2f} W {b1['W_ms']:.2f} ms, c {b1['c_FBW_ms']:.2f} ms,"
      f" F'_eff {b1['Fprime_eff_flops']/1e9:.3f} GFLOP/s; pure GEMM {PROF['bf16']['pure_gemm_flops']/1e9:.3f} GFLOP/s;"
      f" fp32 recipe F'_eff {PROF['fp32']['block']['Fprime_eff_flops']/1e9:.1f}, pure GEMM {PROF['fp32']['pure_gemm_flops']/1e9:.1f}")
for w_ in (4, 8):
    print(f"  w {w_} concurrent peers: slowest c {8*PF*T/FPR[w_]*1e3:.2f} ms -> F' {FPR[w_]/1e9:.3f} GFLOP/s; split F:B:W "
          f"{SPL[w_][0]:.4f}:{SPL[w_][1]:.4f}:{SPL[w_][2]:.4f} (model 0.25:0.50:0.25)")
print(f"  fold: bf16 widen+add {FOLD[0]/1e9:.1f} GB/s (18 B/elem), fp32 add {FOLD[1]/1e9:.1f} GB/s (12 B/elem), "
      f"AdamW {FOLD[2]/1e9:.1f} GB/s (30 B/elem)")

# ---------------------------------------------------------------- pinned configuration and arms
HP, VP, SP, TP = 256, 2000, 256, 256
ARMS = []
def emit(arm, x, w_, g, f6=None, extra=None):
    d = dict(arm=arm, predicted_ms=round(x["E"] * 1e3, 2), predicted_pct_of_C=round(pct(x), 3), C_ms=round(x["C"] * 1e3, 1),
             X_ms=round(x["X"] * 1e3, 2), util=round(x.get("util", 0.0), 4), s1=bool(x.get("s1", True)), w=w_,
             f6_min_margin_ms=None, f6_min_margin_pct=None)
    if extra:
        d.update(extra)
    ARMS.append(d)
    return d

LAYS = [("today", dict(lay="stride", wire="fp32", flush="stage", barrier=True, order="bwd")),
        ("step1", dict(lay="stride", barrier=True)),
        ("lpt", dict(lay="lpt")),
        ("pieces_fwd", dict(lay="pieces", order="fwd")),
        ("pieces_bwd", dict(lay="pieces", order="bwd")),
        ("fifo_pieces_fwd", dict(lay="pieces", order="fwd", policy="fifo")),
        ("fifo_pieces_bwd", dict(lay="pieces", order="bwd", policy="fifo")),
        ("barrier_pieces_fwd", dict(lay="pieces", order="fwd", barrier=True))]
CONF = {}
for tag, kind, L, w_, Mw_ in (("stage0", "first", 1, 4, 32), ("interior_w4", "mid", 4, 4, 8), ("interior_w8", "mid", 4, 8, 4)):
    g = env(HP, VP, SP, TP, FPR[w_], SPL[w_], FOLD)
    cl = 8 * sum(u[4] for u in g["stage_units"]("last", 1)) * TP / FPR[w_]
    CB = 32 * cl if kind == "first" else None
    CONF[tag] = (g, kind, L, w_, Mw_, CB)
    for nm, kw in LAYS:
        x = e4arm(g, kind, L, w_, Mw_, C_pipe=CB, **kw)
        emit(f"{tag}.{nm}", x, w_, g)
g4 = CONF["stage0"][0]
CAP4 = links["400G"] * g4["S_RATE"]; CAP8 = links["400G"] * CONF["interior_w8"][0]["S_RATE"]
print("== (2) scaled B' pinned (h' 256, V' 2000, seq' 256, T_mb' 256, n_head 4, bf16 recipe, M/w 32 / 8 / 4)")
for w_, gg, cp in ((4, g4, CAP4), (8, CONF["interior_w8"][0], CAP8)):
    Fp, _m, cpcie = gg["DEV"]["H100"]
    print(f"  w {w_}: F' {Fp/1e9:.3f} GFLOP/s, cap' {cp*8/1e6:.2f} Mbit/s ({cp/1e6:.3f} MB/s), PCIe' {cpcie/1e6:.3f} MB/s, "
          f"HBM' {gg['HBM']/1e9:.3f} GB/s (model, paced) vs measured fold {FOLD[0]/1e9:.1f} GB/s (pinned), "
          f"chunk f' {gg['F_COPY']} B, hop latency 2f'/PCIe' {2*gg['F_COPY']/cpcie*1e3:.2f} ms, c_block {8*PF*T/Fp*1e3:.2f} ms")
print(f"  stage-0 arm period C_B' = 32 * c_last' = {CONF['stage0'][5]*1e3:.1f} ms (c_last' = (1 + V/(12h)) c_block')")

# F6 margins
def find(a):
    return next(x for x in ARMS if x["arm"] == a)
for pa, alt in (("stage0.pieces_fwd", "stage0.step1"), ("stage0.pieces_fwd", "stage0.lpt"),
                ("interior_w4.pieces_fwd", "interior_w4.step1"), ("interior_w8.pieces_fwd", "interior_w8.lpt")):
    p_, a_ = find(pa), find(alt)
    a_.setdefault("f6_vs", [])
    m_ms = a_["predicted_ms"] - p_["predicted_ms"]; m_pct = a_["predicted_pct_of_C"] - p_["predicted_pct_of_C"]
    a_["f6_min_margin_ms"] = round(m_ms / 2, 2); a_["f6_min_margin_pct"] = round(m_pct / 2, 3); a_["f6_vs"] = pa

# RTT arms (emission order), interior w4 and stage 0
for tag in ("interior_w4", "stage0"):
    g, kind, L, w_, Mw_, CB = CONF[tag]
    c = 8 * sum(u[4] for u in g["stage_units"](kind, L)) * TP / FPR[w_]; cW = c * g["FW"]
    for rname, rtt in (("rtt0", 0.0), ("rtt_2cw", 2 * cW)):
        for order in ("fwd", "bwd"):
            x = e4arm(g, kind, L, w_, Mw_, C_pipe=CB, lay="pieces", order=order, rtt=rtt)
            emit(f"{tag}.emission_{order}.{rname}", x, w_, g, extra=dict(rtt_ms=round(rtt * 1e3, 3), c_W_ms=round(cW * 1e3, 3)))

# two-site shared uplink, interior w4, sites 2 + 2, one uplink per direction at cap' shared by every cross-site flow
g, kind, L, w_, Mw_, CB = CONF["interior_w4"]
for order in ("fwd", "bwd"):
    x = e4arm(g, kind, L, w_, Mw_, lay="pieces", order=order, sites=[0, 0, 1, 1], cap_up=CAP4)
    emit(f"interior_w4.uplink_pieces_{order}", x, w_, g,
         extra=dict(cap_up_MBps=round(CAP4 / 1e6, 4), util_up=round(x["util_up"], 4), s1_up=bool(x["s1_up"]),
                    linkb_up_ms=round(x["linkb_up"] * 1e3, 2)))

# F5 tree arm against the left fold, stage 0 and interior w4 (S1 holds at both)
for tag in ("interior_w4", "stage0"):
    g, kind, L, w_, Mw_, CB = CONF[tag]
    x = e4arm(g, kind, L, w_, Mw_, C_pipe=CB, lay="pieces", order="fwd", tree=True)
    emit(f"{tag}.tree", x, w_, g, extra=dict(f5_left_fold_ms=find(f"{tag}.pieces_fwd")["predicted_ms"]))

# F3 sweep through S1: interior w4 pieces (forward emission), cap' varied (PCIe' and the hop latency held at base)
g, kind, L, w_, Mw_, CB = CONF["interior_w4"]
base = e4arm(g, kind, L, w_, Mw_, lay="pieces")
capS1 = (base["per_mb"] + base["A"]) / base["c"]
SW = []
for u_ in (0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0):
    cp = capS1 / u_
    g["DEV"] = {"H100": (g["DEV"]["H100"][0], 80e9, max(cp, g["DEV"]["H100"][2]) * 10)}   # PCIe not binding
    x = e4arm(g, kind, L, w_, Mw_, cap_nic=cp, lay="pieces", lat=base["lat"])
    SW.append((u_, cp, x))
    emit(f"sweep.util_{u_:.1f}", x, w_, g, extra=dict(cap_MBps=round(cp / 1e6, 4)))
g["DEV"] = {"H100": (FPR[4], 80e9, 50e9 * g["S_RATE"])}
def fit(pts):   # least squares E% = a + b * (1/cap), cap in MB/s
    xs = [1e6 / cp for cp, _ in pts]; ys = [pct(x) for _, x in pts]; n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((p - mx) * (q - my) for p, q in zip(xs, ys)) / sum((p - mx) ** 2 for p in xs)
    return my - b * mx, b
lo = fit([(cp, x) for u_, cp, x in SW if u_ < 1]); hi = fit([(cp, x) for u_, cp, x in SW if u_ > 1])
inv_knee = (lo[0] - hi[0]) / (hi[1] - lo[1]); cap_knee = 1e6 / inv_knee
print(f"== (3) F3 sweep (interior w4 pieces): S1 at cap' {capS1/1e6:.4f} MB/s (util 1); two-line fit of E% against 1/cap': "
      f"below S1 {lo[0]:.3f} + {lo[1]:.3f}/cap', above S1 {hi[0]:.3f} + {hi[1]:.3f}/cap'; they meet at cap' {cap_knee/1e6:.4f} MB/s"
      f" = util {capS1/cap_knee:.4f} (the predicted S1 threshold F3 compares within 25%)")
THR = dict(cap_s1_MBps=capS1 / 1e6, cap_knee_MBps=cap_knee / 1e6, util_at_knee=capS1 / cap_knee,
           fit_below=dict(a_pct=lo[0], b_pct_MBps=lo[1]), fit_above=dict(a_pct=hi[0], b_pct_MBps=hi[1]))

print("== (4) arms (SIMULATED: the round-4 model at the measured F', split and fold; % of each arm's C)")
for a in ARMS:
    f6 = "" if a["f6_min_margin_ms"] is None else f"  F6: pieces must beat this by >= {a['f6_min_margin_ms']:.2f} ms ({a['f6_min_margin_pct']:.3f} pts)"
    print(f"  {a['arm']:38s} E {a['predicted_ms']:9.2f} ms = {a['predicted_pct_of_C']:7.3f}% of C {a['C_ms']:8.1f}; X {a['X_ms']:8.2f}; "
          f"util {a['util']:.3f}{f6}")

# ---------------------------------------------------------------- invariance and sensitivity
print("== (5) scale invariance at the measured split: full B (F 400 TFLOP/s) vs B' (F' w4), stage 0 and interior w4, % of C")
for tag, kind, L, w_, Mw_ in (("stage0", "first", 1, 4, 32), ("interior_w4", "mid", 4, 4, 8), ("interior_w8", "mid", 4, 8, 4)):
    row = []
    for (hh, VV, seq, Tmb, Fp) in ((4096, 32000, 4096, 4096, 400e12), (HP, VP, SP, TP, FPR[w_])):
        gg = env(hh, VV, seq, Tmb, Fp, SPL[w_])
        cl = 8 * sum(u[4] for u in gg["stage_units"]("last", 1)) * Tmb / Fp
        CBx = 32 * cl if kind == "first" else None
        row.append([round(pct(e4arm(gg, kind, L, w_, Mw_, C_pipe=CBx, **kw)), 3) for _, kw in LAYS[:5]])
    print(f"  {tag}: full B {row[0]} | B' {row[1]}  (today, step1, lpt, pieces fwd, pieces bwd; HBM' paced)")
print("== (6) sensitivity of the F6 arms (% of C): round-5 split; HBM' paced fold; no hop latency; W fraction x0.75 / x1.25")
def f6set(split, fold_, lat0=False):
    out = []
    for tag, kind, L, w_, Mw_, alts in (("stage0", "first", 1, 4, 32, ("step1", "lpt")), ("interior_w4", "mid", 4, 4, 8, ("step1",)),
                                        ("interior_w8", "mid", 4, 8, 4, ("lpt",))):
        gg = env(HP, VP, SP, TP, FPR[w_], split(w_), fold_)
        cl = 8 * sum(u[4] for u in gg["stage_units"]("last", 1)) * TP / FPR[w_]
        CBx = 32 * cl if kind == "first" else None
        kwl = dict(LAYS)
        lat = 0.0 if lat0 else None
        p = pct(e4arm(gg, kind, L, w_, Mw_, C_pipe=CBx, lat=lat, **kwl["pieces_fwd"]))
        for a in alts:
            q = pct(e4arm(gg, kind, L, w_, Mw_, C_pipe=CBx, lat=lat, **kwl[a]))
            out.append(f"{tag} pieces {p:.2f} vs {a} {q:.2f}")
    return "; ".join(out)
def wscale(f):
    def sp(w_):
        F_, B_, W_ = SPL[w_]; W2 = W_ * f; s = F_ + B_ + W2
        return (F_ / s, B_ / s, W2 / s)
    return sp
print("  pinned (measured split, measured fold):", f6set(lambda w_: SPL[w_], FOLD))
print("  round-5 split 1:2:1, measured fold    :", f6set(lambda w_: (0.25, 0.5, 0.25), FOLD))
print("  measured split, HBM' paced fold       :", f6set(lambda w_: SPL[w_], None))
print("  measured split, no hop latency        :", f6set(lambda w_: SPL[w_], FOLD, lat0=True))
print("  measured split, W fraction x0.75      :", f6set(wscale(0.75), FOLD))
print("  measured split, W fraction x1.25      :", f6set(wscale(1.25), FOLD))

print("== (7) ratios kept (round5.md section 6), full B vs B'")
def ratios(hh, VV, seq, Tmb, Fp, split):
    gg = env(hh, VV, seq, Tmb, Fp, split)
    us0 = gg["stage_units"]("first", 1); P0 = sum(u[1] for u in us0) + 13 * hh   # + the block's 13h biases and norms
    tables = (VV + seq) * hh; tok = VV * hh
    gb = sum(u[2] * u[1] for u in us0) + 2 * 13 * hh
    fp32share = (4 * tables) / gb
    x = e4arm(gg, "first", 1, 4, 32)
    c = x["c"]
    Fp_, _m, cpcie = gg["DEV"]["H100"]
    return {"V_over_h": VV / hh, "seq_over_h": seq / hh, "Tmb_over_h": Tmb / hh, "embed_share_stage0": tables / P0,
            "tok_share_stage0": tok / P0, "fp32_share_stage0_grad_bytes": fp32share, "nic_util_stage0": x["util"],
            "M_over_w_stage0": 32, "fold_over_c_HBM_model": 18 * (sum(u[1] for u in us0) / 4) / gg["HBM"] / c,
            "chunk_time_over_c": gg["F_COPY"] / x["cap"] / c, "hop_latency_over_c": 2 * gg["F_COPY"] / cpcie / c,
            "C_B_over_c_stage0": 32 * (1 + VV / (12 * hh))}
RAT = {"full_B": ratios(4096, 32000, 4096, 4096, 400e12, SPL[4]), "B_prime": ratios(HP, VP, SP, TP, FPR[4], SPL[4])}
for k_ in RAT["full_B"]:
    a, b = RAT["full_B"][k_], RAT["B_prime"][k_]
    print(f"  {k_:32s} full B {a:10.4f}  B' {b:10.4f}  ({100*(b/a-1):+.2f}%)")

RESULT = dict(Fprime={str(k_): v for k_, v in FPR.items()}, split={str(k_): v for k_, v in SPL.items()}, fold=FOLD,
              cap_prime={"4": CAP4, "8": CAP8}, C_B_prime=CONF["stage0"][5],
              pcie_prime={"4": g4["DEV"]["H100"][2], "8": CONF["interior_w8"][0]["DEV"]["H100"][2]},
              hbm_prime_model={"4": g4["HBM"], "8": CONF["interior_w8"][0]["HBM"]},
              chunk_bytes=g4["F_COPY"], arms=ARMS, threshold=THR, ratios=RAT)
print("== (8) JSON")
print(json.dumps(RESULT, sort_keys=True))
