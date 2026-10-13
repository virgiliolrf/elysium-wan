# m2b-e7 PLAN derivation (ADR-0006 R16, F9; handout precondition P3). Re-derives round5.md section 3 (E7's
# expected planner choices) with the stage costs the planner uses instead of 8 FLOPs per GEMM parameter:
#   - the layer cut from the repository's proportional_partition (exact min-max DP over layer_costs, partition.py);
#   - per-stage F, B, W from the repository's analytic_costs (zero_bubble.py; the paper's table with attention,
#     recompute adds one F to B), which is (4/3) x layer_costs per block and per Head;
#   - c_s = (F + B + W) x T_mb / F_dev, C = (M/w) x max_s c_s, the lane solve on those F/B/W, the exposure tail
#     on the planner's c_F, c_W (t_gap = c_B + c_W/U), T_ovh unchanged, S1 (NIC, host DRAM) and S2 on that c_s.
# Everything else is round 5's model (round5.md section 8, loaded from r5.py, which loads r4, r3, r2), unchanged.
# Modes: "r5" (control: must reproduce round 5's table), "planner" (primary), "planner_s1_8p" (S1 window kept at
# 8 P^F T_mb/F as the i16-admission task pins it), "planner_adrsplit" (planner c_s, ADR's c/4, c/2, c/4 split),
# "norecompute" (c_s = layer_costs x T_mb/F). Every input is HYPOTHESIS as in rounds 3-5; every output DERIVED.
# usage: python3 -I e7.py r5.py r4.py r3.py r2.py zero_bubble.py partition.py OUT.json
import sys, json, importlib.util, types
A0 = list(sys.argv)
R5, R4, R3, R2, ZBP, PARTP, OUT = A0[1:8]
src5 = open(R5).read()
sys.argv = [A0[0], R4, R3, R2, ZBP]
exec(src5[: src5.index('print("== (1) S1 on host DRAM')])          # r4 model + r5 dram_rate (globals)
exec(src5[src5.index("def per_peer_bytes"): src5.index('print("== (2) E7')])   # r5 per_peer_bytes, candidate

_ps = importlib.util.spec_from_file_location("partition", PARTP)
part = importlib.util.module_from_spec(_ps); _ps.loader.exec_module(part)
CFG = types.SimpleNamespace(d_model=h, seq_len=SEQ, vocab_size=V, n_layer=NBLK)   # 7B as in rounds 2-5
TMB = 4096

# ---- textual patches of r4's exch4/stage4 (each must apply exactly once) ----
PATCHES = [
    ("U = len(us); Pfl = sum(u[4] for u in us); cW = cF = c / 4",
     "U = len(us); Pfl = sum(u[4] for u in us); cF, cW = c * FRAC[0], c * FRAC[2]"),
    ("Pfl = sum(u[4] for u in us); c = 8 * Pfl * Tmb / F\n",
     "Pfl = sum(u[4] for u in us); c, c_s1 = STAGE_C(kind, L, Tmb, F, Pfl)\n"),
    ("rtt + c / 2 + (c / 4) / len(us) - C", "rtt + c * FRAC[1] + (c * FRAC[2]) / len(us) - C"),
    ("s1 = (per_mb + A) / cap <= c\n", "s1 = (per_mb + A) / cap <= c_s1\n"),
    ("util=(per_mb + A) / cap / c,", "util=(per_mb + A) / cap / c_s1,"),
    ("return dict(c=c, C=C,", "return dict(c=c, c_s1=c_s1, C=C,"),
]
D = DEFS
for a, b in PATCHES:
    assert D.count(a) == 1, ("patch", a, D.count(a))
    D = D.replace(a, b)
exec(D)
r4_stage_kinds, r4_lane_makespan = stage_kinds, lane_makespan

def dram_rate(r, w, k, kind):   # r5's, on the S1 window c_s1 (identical to r5 when c_s1 == c)
    per_mb = 0.0 if w == 1 else sum(u[2] * u[1] for u in r["us"]) * (w - 1) / w
    A = 4 * 4096 * h if kind == "mid" else 2 * 4096 * h
    return k * 2 * (per_mb + A) / r["c_s1"]

# ---- the planner's per-layer costs, from the repository functions ----
def per_layer(recompute):
    spec = types.SimpleNamespace(model=CFG, recompute=recompute,
                                 partition=[(i, i + 1) for i in range(NBLK + 2)])
    f, b, w = zb.analytic_costs(spec)
    return list(zip(f, b, w))
PER_RC = per_layer(True)
PER_NR = per_layer(False)
LC = part.layer_costs(CFG)

def layers_of(kind, nb):
    blocks = list(range(1, nb + 1))          # every block costs the same; the index only selects its cost
    ls = ([0] if kind == "first" else []) + blocks + ([NBLK + 1] if kind == "last" else [])
    return ls

MODE = "r5"; HOP_COMM = None
def fbw(kind, nb, Tmb, Fdev):
    """(c_F, c_B, c_W) seconds per microbatch for a stage, by mode."""
    if MODE == "r5":
        fl = 12 * h * h * nb + (V * h if kind == "last" else 0)
        c = 8 * fl * Tmb / Fdev
        return c / 4, c / 2, c / 4
    ls = layers_of(kind, nb)
    if MODE == "norecompute":
        f = sum(PER_NR[i][0] for i in ls); b = sum(PER_NR[i][1] for i in ls); w = sum(PER_NR[i][2] for i in ls)
        tot = sum(LC[i] for i in ls); sc = tot / (f + b + w)          # c = layer_costs x T_mb / F
        return f * sc * Tmb / Fdev, b * sc * Tmb / Fdev, w * sc * Tmb / Fdev
    f = sum(PER_RC[i][0] for i in ls); b = sum(PER_RC[i][1] for i in ls); w = sum(PER_RC[i][2] for i in ls)
    if MODE == "planner_adrsplit":
        c = (f + b + w) * Tmb / Fdev
        return c / 4, c / 2, c / 4
    return f * Tmb / Fdev, b * Tmb / Fdev, w * Tmb / Fdev

FRAC = (0.25, 0.5, 0.25)
def STAGE_C(kind, L, Tmb, Fdev, Pfl):
    global FRAC
    cf, cb, cw = fbw(kind, L, Tmb, Fdev)
    c = cf + cb + cw
    FRAC = (cf / c, cb / c, cw / c)
    c_s1 = 8 * Pfl * Tmb / Fdev if MODE in ("r5", "planner_s1_8p") else c
    return c, c_s1

def stage_kinds(S):
    if MODE == "r5":
        return r4_stage_kinds(S)
    bounds = part.proportional_partition(CFG, [1.0] * S)
    out = []
    for a, b in bounds:
        nb = sum(1 for i in range(a, b) if 1 <= i <= NBLK)
        kind = "first" if a == 0 else ("last" if b == NBLK + 2 else "mid")
        out.append((kind, nb))
    return out

def lane_makespan(S, w, M, Tmb, kinds):
    m = M // w
    fs = [fbw(kd, nb, Tmb, F_H100) for kd, nb in kinds]
    cs = [sum(x) for x in fs]
    bnd = Tmb * h * 2
    wp = [nb * Tmb * 16 * h * 4 for _, nb in kinds]
    comm = HOP_COMM(S, w, Tmb) if HOP_COMM else [0.0] * (S - 1)
    cfg = zb.ZBConfig(S, m, [x[0] for x in fs], [x[1] for x in fs], [x[2] for x in fs], comm=comm,
                      mem_f=[float(bnd)] * S, mem_b=[float(x) for x in wp], mem_w=[-float(bnd + x) for x in wp])
    sched = zb.zero_bubble(cfg, "1p") if S == 1 else zb.zero_bubble(zb._with_pending_budget(cfg, 1), "given")
    _e, ms = zb.simulate(cfg, sched.orders)
    return ms, cs

# ---- candidates, admission (SCHEDULER-I16 as written: refuse on S1 NIC, S1 host DRAM, S2), reasons ----
def evaluate(name, S_, w_, M_, cap, gph, kp, kw):
    x = candidate(S_, w_, M_, cap, gph, kp, **kw)
    r = plan(S_, w_, M_, cap, **kw)
    pieces = kw.get("lay", "pieces") == "pieces" and w_ > 1
    reasons = []; struct = []
    if pieces:
        if not x["s1_nic"]:
            bad = [(k_, s["util"]) for k_, s in r["stages"].items() if not s["s1"]]
            reasons.append("S1 NIC: " + "; ".join(f"{k_[0]} stage ({k_[1]} blocks) {u:.2f}x cap*c_s" for k_, u in bad))
            for k_, u in bad:
                struct.append(dict(kind="S1 NIC", where=f"{k_[0]} stage", value=round(u, 3), bound=1.0, unit="x cap*c_s"))
        if not x["s1_host"]:
            hid, rate, sts = x["worst"]
            reasons.append(f"S1 host DRAM: host {hid} {rate/G9:.1f} GB/s > {HOSTBW/G9:.0f} GB/s (stages {sts[:4]})")
            struct.append(dict(kind="S1 host DRAM", where=f"host {hid} (stages {sts})", value=round(rate / G9, 1),
                               bound=HOSTBW / G9, unit="GB/s"))
        if not x["s2"]:
            reasons.append(f"S2: E {x['E']*1e3:.1f} ms > eps*C {EPS*x['C']*1e3:.1f} ms ({r['Etag']})")
            struct.append(dict(kind="S2", where=r["Etag"], value=round(x["E"] * 1e3, 2), bound=round(EPS * x["C"] * 1e3, 2), unit="ms"))
    return dict(name=name, S=S_, w=w_, shard=("pieces" if pieces else ("zero1" if w_ > 1 else "none")),
                wire=(kw.get("wire", "native") if w_ > 1 else "none"), lane_E_ovh_ms=round(x["tot"] * 1e3, 1),
                host_ms=round(x["host_t"] * 1e3, 1), step_ms=round(x["step"] * 1e3, 1), C_ms=round(x["C"] * 1e3, 1),
                E_ms=round(x["E"] * 1e3, 2), admitted=not reasons, refused=" | ".join(reasons), reasons=struct)

def choose(rows):
    adm = sorted([r for r in rows if r["admitted"]], key=lambda r: r["step_ms"])
    best, ru = adm[0], (adm[1] if len(adm) > 1 else None)
    return best, ru, (round(100 * (ru["step_ms"] / best["step_ms"] - 1), 2) if ru else None)

Bc = [("S 32 w 4 pieces", 32, 4, 128, dict(lay="pieces")),
      ("S 32 w 4 zero1 + native", 32, 4, 128, dict(lay="stride", barrier=True)),
      ("S 32 w 4 zero1 today (fp32)", 32, 4, 128, dict(lay="stride", wire="fp32", flush="stage", barrier=True, order="bwd")),
      ("S 16 w 8 pieces", 16, 8, 128, dict(lay="pieces")),
      ("S 16 w 8 zero1 + native", 16, 8, 128, dict(lay="stride", barrier=True)),
      ("S 32 w 1 on 32 GPUs", 32, 1, 128, {})]
Sc = [("S 32 w 1", 32, 1, 32, {})] + [(f"S {S_} w {w_} {nm}", S_, w_, 32, kw) for S_, w_ in ((16, 2), (8, 4), (4, 8))
                                      for nm, kw in (("pieces", dict(lay="pieces")), ("zero1 + native", dict(lay="stride", barrier=True)))]
CONFIGS = [("32 H100", "400G", Sc, ((1, 10), (2, 10), (4, 10), (8, 10), (8, 2))),
           ("32 H100", "100G", Sc, ((1, 10), (8, 10))),
           ("128 H100", "400G", Bc, ((1, 10), (1, 6), (1, 2), (8, 10), (8, 2))),
           ("128 H100", "100G", Bc, ((1, 10), (8, 10)))]

def run_mode(mode):
    global MODE
    MODE = mode
    res = []
    for fleet, ln, cands, hc in CONFIGS:
        for gph, kp in hc:
            rows = [evaluate(n, S_, w_, M_, links[ln], gph, kp, kw) for n, S_, w_, M_, kw in cands]
            best, ru, mg = choose(rows)
            res.append(dict(fleet=fleet, link_gbps=int(ln[:-1]), gpus_per_host=gph, k=kp, choice=best["name"],
                            step_ms=best["step_ms"], runner_up=ru["name"] if ru else None,
                            runner_up_ms=ru["step_ms"] if ru else None, margin_pct=mg, rows=rows))
    return res

OUTJ = {"modes": {}}
for mode in ("r5", "planner", "planner_s1_8p", "planner_adrsplit", "norecompute"):
    OUTJ["modes"][mode] = run_mode(mode)

# control: round 5's section-3 table must be reproduced exactly by the "r5" mode
R5_PIN = {("128 H100", 400, 1, 10): ("S 32 w 4 zero1 + native", 1721.3), ("128 H100", 400, 1, 6): ("S 32 w 4 pieces", 1124.5),
          ("128 H100", 400, 8, 10): ("S 32 w 1 on 32 GPUs", 3887.8), ("128 H100", 400, 8, 2): ("S 32 w 4 zero1 + native", 1721.3),
          ("32 H100", 400, 1, 10): ("S 16 w 2 pieces", 966.8), ("32 H100", 400, 2, 10): ("S 16 w 2 zero1 + native", 994.4),
          ("32 H100", 400, 4, 10): ("S 32 w 1", 1102.2), ("32 H100", 400, 8, 10): ("S 32 w 1", 1102.2),
          ("32 H100", 400, 8, 2): ("S 16 w 2 pieces", 966.8), ("32 H100", 100, 1, 10): ("S 32 w 1", 1102.2),
          ("32 H100", 100, 8, 10): ("S 32 w 1", 1102.2)}
for c in OUTJ["modes"]["r5"]:
    key = (c["fleet"], c["link_gbps"], c["gpus_per_host"], c["k"])
    if key in R5_PIN:
        assert (c["choice"], c["step_ms"]) == R5_PIN[key], ("round-5 control not reproduced", key, c["choice"], c["step_ms"])
print("control: round5.md section 3 reproduced by mode r5 (11 pinned cells)")

# ---- per-stage table (primary mode): c_s, C, S1 per GPU on host DRAM at k 2/6/10 ----
MODE = "planner"
stab = []
for tag, S_, w_, M_ in (("point B (128 H100, S 32, w 4)", 32, 4, 128), ("32 H100, S 16, w 2", 16, 2, 32)):
    for mode in ("r5", "planner"):
        MODE = mode
        r = plan(S_, w_, M_, links["400G"])
        for key, x in r["stages"].items():
            stab.append(dict(point=tag, mode=mode, stage=f"{key[0]} ({key[1]} blocks)", c_ms=round(x["c"] * 1e3, 2),
                             C_ms=round(r["C"] * 1e3, 1), lane_ms=round(r["ms"] * 1e3, 1),
                             E_ms=round(x["E"] * 1e3, 2), tovh_ms=round(x["tovh"] * 1e3, 1),
                             nic_util=round(x["util"], 3),
                             dram_gbps={str(kp): round(dram_rate(x, w_, kp, key[0]) / G9, 1) for kp in (2, 6, 10)},
                             gpus_per_600GBps_host_k10=round(HOSTBW / dram_rate(x, w_, 10, key[0]), 2),
                             kmax_1gpu_host=round(HOSTBW / dram_rate(x, w_, 1, key[0]), 2)))
MODE = "planner"
OUTJ["stages"] = stab
OUTJ["cut"] = {str(S_): part.proportional_partition(CFG, [1.0] * S_) for S_ in (4, 8, 16, 32)}
OUTJ["layer_costs"] = {"embed": LC[0], "block": LC[1], "head": LC[-1], "head_over_block": round(LC[-1] / LC[1], 4),
                       "r4_head_over_block": round(V * h / (12 * h * h), 4)}

# ---- refusal configurations ----
# (a) one 10 Gbps uplink between two sites of 16 H100 (derivation.md section 6, placement "all": every wide stage
#     straddles, half its peers per site, lanes inside a site); S1 on the uplink summed over every stage, k 1, share
#     n_a*n_b/w per direction, no boundary bytes on it. Planner cost c_s.
cap_up = links["10G"]; up = []
for S_, w_ in ((16, 2), (8, 4), (4, 8)):
    r = plan(S_, w_, 32, links["400G"])
    rate = 0.0
    for (kind, nb) in r["kinds"]:
        x = r["stages"][(kind, nb)]
        bgP = sum(u[2] * u[1] for u in x["us"])
        rate += bgP * (w_ / 2) * (w_ / 2) / w_ / x["c_s1"]
    bytes_dir = sum((32 * sum(u[2] * u[1] for u in r["stages"][kn]["us"]) / 4
                     + sum(u[3] * u[1] for u in r["stages"][kn]["us"]) * w_ / 4) for kn in r["kinds"])
    up.append(dict(candidate=f"S {S_} w {w_} pieces, straddling", reason=f"S1 uplink site0-site1: {rate/G9:.1f} GB/s > {cap_up/G9:.2f} GB/s ({rate/cap_up:.0f}x)",
                   reasons=[dict(kind="S1 uplink", where="site0-site1", value=round(rate / G9, 1), bound=cap_up / G9, unit="GB/s")],
                   zero1_uplink_bound_ms=round(bytes_dir / cap_up * 1e3, 0)))
#     Contiguous placement (plan GPU g on site g // 16): no wide stage straddles; one pipeline hop crosses the uplink.
#     The lane solve gets that hop's delay w*A/cap_up + RTT/2 (A = T_mb*h*2 bytes, w lanes share the uplink, fluid,
#     HYPOTHESIS), and the step is bounded below by the uplink's per-direction volume M*A/cap_up + RTT.
RTT_UP = 0.050
A_up = TMB * h * 2
def hop_up(S_, w_, Tmb):
    j = 16 // w_ - 1
    return [(w_ * A_up / cap_up + RTT_UP / 2) if i == j else 0.0 for i in range(S_ - 1)]
HOP_COMM = hop_up
rows = [evaluate(n, S_, w_, M_, links["400G"], 1, 10, kw) for n, S_, w_, M_, kw in Sc]
HOP_COMM = None
floor_up = 32 * A_up / cap_up + RTT_UP
for r_ in rows:
    r_["step_ms"] = round(max(r_["step_ms"], floor_up * 1e3), 1)
best, ru, mg = choose(rows)
OUTJ["uplink_contiguous"] = dict(choice=best["name"], step_ms=best["step_ms"], runner_up=ru["name"],
                                 runner_up_ms=ru["step_ms"], margin_pct=mg, uplink_floor_ms=round(floor_up * 1e3, 1),
                                 rows=rows)
OUTJ["uplink"] = up
# (c) mixed classes: 17 GPUs of class A then 15 of class B (1-GPU hosts, 400 Gbps, k 10, equal rates: HYPOTHESIS so
#     only the class predicate differs from the homogeneous row). Stage-contiguous: the stage holding GPUs 16 and 17
#     mixes classes at every width 2, 4, 8, so every wide candidate is refused naming the class.
mixed = []
for n, S_, w_, M_, kw in Sc:
    if w_ == 1:
        continue
    s_mix = 16 // w_
    mixed.append(dict(candidate=n, reason=f"class: stage {s_mix} peers span classes A and B (GPUs {s_mix*w_}..{s_mix*w_+w_-1})",
                      reasons=[dict(kind="class", where=f"stage {s_mix}", value="A+B", bound="one class", unit="")]))
OUTJ["mixed"] = mixed
# (b) MEM-HOST: one-step host bound per peer (round5.md section 4 formula, ownership-general), per host, at the
#     32 H100 400 Gbps 1-GPU k 10 configuration; pieces and zero1 (stride ownership; HYPOTHESIS for zero1).
def host_hold(x, w_, M_, Mk):
    us, own = x["us"], x["own"]; worst = 0.0
    for j in range(w_):
        g = sum(u[2] * u[1] * ((1 - own[i][j]) * Mk + own[i][j] * (M_ - Mk)) for i, u in enumerate(us))
        p = sum(u[3] * u[1] * (own[i][j] * (w_ - 1) + (1 - own[i][j])) for i, u in enumerate(us))
        worst = max(worst, g + p)
    return worst
mh = []
for nm, S_, w_, kw in (("S 16 w 2 pieces", 16, 2, dict(lay="pieces")), ("S 16 w 2 zero1 + native", 16, 2, dict(lay="stride", barrier=True)),
                       ("S 8 w 4 pieces", 8, 4, dict(lay="pieces")), ("S 8 w 4 zero1 + native", 8, 4, dict(lay="stride", barrier=True)),
                       ("S 4 w 8 pieces", 4, 8, dict(lay="pieces")), ("S 4 w 8 zero1 + native", 4, 8, dict(lay="stride", barrier=True))):
    r = plan(S_, w_, 32, links["400G"], **kw)
    per = {f"{k_[0]} ({k_[1]} blocks)": round(host_hold(x, w_, 32, 32 // w_) / G9, 2) for k_, x in r["stages"].items()}
    mh.append(dict(candidate=nm, per_peer_gb=per, max_gb=max(per.values())))
OUTJ["memhost"] = mh

json.dump(OUTJ, open(OUT, "w"), indent=1, sort_keys=True)

# ---- text report ----
for mode in ("r5", "planner", "planner_s1_8p", "planner_adrsplit", "norecompute"):
    print(f"== mode {mode}")
    for c in OUTJ["modes"][mode]:
        print(f"  {c['fleet']} {c['link_gbps']} Gbps, {c['gpus_per_host']} GPU/host, k {c['k']}: {c['choice']} {c['step_ms']} ms;"
              f" runner-up {c['runner_up']} {c['runner_up_ms']} (+{c['margin_pct']}%)")
        if mode == "planner":
            for r in c["rows"]:
                print(f"      {r['name']:28s} lane+E+T_ovh {r['lane_E_ovh_ms']:7.1f} host {r['host_ms']:7.1f} step {r['step_ms']:7.1f}"
                      f" C {r['C_ms']:6.1f} E {r['E_ms']:6.2f} {'admitted' if r['admitted'] else 'refused: ' + r['refused']}")
print("== stages (r5 vs planner)")
for s in stab:
    print("  ", s)
print("== cut (planner):", OUTJ["cut"])
print("== layer_costs:", OUTJ["layer_costs"])
print("== uplink:", up)
u_ = OUTJ["uplink_contiguous"]
print(f"== uplink, contiguous: {u_['choice']} {u_['step_ms']} ms; runner-up {u_['runner_up']} {u_['runner_up_ms']} (+{u_['margin_pct']}%); floor {u_['uplink_floor_ms']} ms")
for r_ in u_["rows"]:
    print(f"      {r_['name']:28s} step {r_['step_ms']:7.1f} {'admitted' if r_['admitted'] else 'refused: ' + r_['refused']}")
print("== mixed:", mixed)
print("== MEM-HOST:", mh)
