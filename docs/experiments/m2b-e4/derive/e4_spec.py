# m2b-e4, SC-4: E4's predictions re-derived with the model of the SPEC of record (SCHEDULER-I17), by an oracle
# written from the SPEC text (i17_oracle.py beside this file), independent of src/wantrain (nothing imported, no
# product source read). Inputs are exactly the granted PLAN's MEASURED records (profile_fprime.out, conc_w4.out,
# conc_w8.out) and its scaling map (round5.md section 6); nothing else is an input.
#
# What changes against the granted derivation (r6_e4.py on round 4's exch4):
#  (1) ONE simulation of every microbatch's traffic per arm, E and window_s read from it (I17), instead of
#      E = max(X_tail, V_max/cap + RTT + t_gap - C) with V_max letting a hot owner's gather egress overlap its own
#      gradient ingress (derivation.md section 2's analytic term). Under I17 a unit's gathers leave only after the
#      owner's last fold and AdamW, so ingress and gather egress of a hot owner are serial (SC-4's cause).
#  (2) The fold of every contribution is a left fold in canonical microbatch order at the measured device rates
#      ((b_g + 16) B per element at 80.98 GB/s for a 2-byte wire, 12 B at 129.22 GB/s for fp32, AdamW 30 B at
#      42.74 GB/s; I17's byte counts, the PLAN's pinned rates); the gather leaves after the last fold + AdamW.
#  (3) Host DRAM is a resource in the simulation (I17), at k 10 passes per side. In E4 every peer runs on this Mac;
#      the shim paces NIC and PCIe, not DRAM, and the DRAM load is <= 10 x 2 x 8 x 16.7 MB/s = 2.7 GB/s against
#      ~100 GB/s, so E4's hosts are unconstrained (INF): the term exists and does not bind (stated, not assumed away).
#  (4) The Embed is ONE fp32 unit (tok + pos tables, I17), released when W(mb) starts; whole-tensor layouts (stride,
#      LPT) give it fractional owners (tok and pos are different tensors).
#  (5) The stage-0 arm's padding to C_B' is placed where a pipeline puts it: peer q's i-th microbatch occupies the slot
#      [i*tau, (i+1)*tau), tau = C_B'/(M/w); its F starts the slot, its B then W end the slot (the idle is the wait
#      for the downstream backward). Stage 0's last W therefore ends at the period and window_s = 0, as round 4's
#      model assumes ("stage 0: window_s = 0", r4.py (8)). The granted model's linkb measured stage 0 against C_B'
#      with the link busy over the whole period; this is the same arm made explicit.
#  (6) A whole-stage-flush mode for the three 'today' arms (whole-tensor stride ZeRO-1, fp32 wire, every unit's
#      gradient pieces of W(mb) released at W(mb)'s END, full step barrier), so they are expressible.
# Kept from the granted PLAN: shapes, w, M/w, F', op split, fold rates (not paced), cap' = PCIe' = 2e-3 B/s per
# FLOP/s x F', f' 65,536 B, hop latency 2f'/PCIe', C_B' = 32 c_last', RTT arms (0, 2 c_W), sweep utilizations,
# uplink sites [0, 0, 1, 1] at cap', the tree's definition, boundary bytes per microbatch (load, released at F end).
# usage: python3 -I -B e4_spec.py i17_oracle.py profile_fprime.out conc_w4.out conc_w8.out > e4_spec.out
#        (writes e4_spec.json beside the output path given as the 5th argument, default ./e4_spec.json)
import hashlib
import importlib.util
import json
import sys

A = sys.argv
for p in A[1:5]:
    print(f"input {p.split('/')[-1]}: sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
_sp = importlib.util.spec_from_file_location("i17_oracle", A[1])
O = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(O)
PROF = json.load(open(A[2]))
CONC = {4: json.load(open(A[3])), 8: json.load(open(A[4]))}
OUT_JSON = A[5] if len(A) > 5 else "e4_spec.json"

# ------------------------------------------------------------------ inputs (granted PLAN, MEASURED records)
HP, VP, SP, TP = 256, 2000, 256, 256          # h', V', seq', T_mb'
PF = 12 * HP * HP                             # GEMM params of one block
F_B, T_B, H_B = 400e12, 4096, 4096            # point B
CAP_B, HBM_B, CHUNK_B = 50e9, 3.35e12, 16 * 2**20
MODS = (("qkv", 3, 2), ("proj", 1, 4), ("fc1", 4, 8), ("fc2", 4, 10))   # (name, x h^2, today's stride index)


def split_of(conc):
    n = len(conc["peers"])
    f = [sum(p[k] / p["c_ms"] for p in conc["peers"]) / n for k in ("F_ms", "B_ms", "W_ms")]
    return tuple(x / sum(f) for x in f), max(p["c_ms"] for p in conc["peers"]) / 1e3


SPL, FPR = {}, {}
for w_ in (4, 8):
    sp, cmax = split_of(CONC[w_])
    SPL[w_], FPR[w_] = sp, 8 * PF * TP / cmax
FOLD = (PROF["fold"]["bf16_widen_add_GBps"] * 1e9, PROF["fold"]["fp32_add_GBps"] * 1e9,
        PROF["fold"]["adamw_GBps_at_30B"] * 1e9)

STAGES = {"stage0": ("first", 1, 4, 32), "interior_w4": ("mid", 4, 4, 8), "interior_w8": ("mid", 4, 8, 4)}


def units_of(kind, L, h, V, seq, wire_fp32=False):
    """[(Unit, [(tensor elements, stride index)])]: the Embed as one fp32 unit (tok, pos), then 4 SplitLinears per
    block; stride indices are today's parameter list (r4.py, stage 0 base 2, 12 parameters per block)."""
    out = []
    base = 0
    if kind == "first":
        out.append((O.Unit("embed", (V + seq) * h, 4, 4, 0, 0.0), [(V * h, 0), (seq * h, 1)]))
        base = 2
    b = 4 if wire_fp32 else 2
    for blk in range(L):
        for nm, m, ix in MODS:
            p = m * h * h
            out.append((O.Unit(f"b{blk}.{nm}", p, b, b, len(out), float(p)), [(p, base + 12 * blk + ix)]))
    return out


def owners_of(us, w, lay):
    if lay == "pieces":
        return [[1.0 / w] * w for _ in us]
    rows = []
    if lay == "stride":
        for u, tensors in us:
            r = [0.0] * w
            for n, ix in tensors:
                r[ix % w] += n / u.params
            rows.append(r)
        return rows
    if lay == "lpt":                          # whole tensors, longest first onto the least loaded peer (r4.py)
        tens = [(n, ui, ti) for ui, (u, ts) in enumerate(us) for ti, (n, _ix) in enumerate(ts)]
        load = [0] * w
        rows = [[0.0] * w for _ in us]
        for n, ui, ti in sorted(tens, key=lambda x: (-x[0], x[1], x[2])):
            j = load.index(min(load))
            load[j] += n
            rows[ui][j] += n / us[ui][0].params
        return rows
    raise ValueError(lay)


class Conf:
    """One stage arm's configuration at a given F', split and fold (round5.md section 6's scaling map)."""

    def __init__(self, tag, Fp, split, fold=None, h=HP, V=VP, seq=SP, T=TP):
        kind, L, w, mw = STAGES[tag]
        self.tag, self.kind, self.L, self.w, self.mw = tag, kind, L, w, mw
        self.h, self.V, self.seq, self.T, self.Fp, self.split = h, V, seq, T, Fp, split
        self.s_rate = (Fp / F_B) * (T_B / T)
        self.cap = CAP_B * self.s_rate
        self.pcie = self.cap
        self.chunk = max(1, int(CHUNK_B * (h / H_B) ** 2))
        if fold is None:
            hbm = HBM_B * self.s_rate
            self.rates = O.Rates(hbm, hbm, hbm)
        else:
            self.rates = O.Rates(*fold)
        us = units_of(kind, L, h, V, seq)
        self.c = 8 * sum(u.flops for u, _ in us) * T / Fp
        self.c_last = 8 * (12 * h * h + V * h) * T / Fp
        self.CB = 32 * self.c_last if kind == "first" else None
        self.C = self.CB if kind == "first" else mw * self.c
        self.A = (4 if kind == "mid" else 2) * T * h


def lanes(cf):
    """Per-peer op timelines. Interior: F, B, W back to back per owned microbatch (peer q owns q, q+w, ...).
    Stage 0: slot tau = C_B'/(M/w); F at the slot start, B then W ending at the slot end."""
    cF, cB, cW = (cf.c * x for x in cf.split)
    out = []
    for q in range(cf.w):
        ops = []
        for i in range(cf.mw):
            mb = i * cf.w + q
            if cf.kind == "first":
                tau = cf.CB / cf.mw
                s0, e0 = i * tau, (i + 1) * tau
                ops += [("F", s0, s0 + cF, mb), ("B", e0 - cW - cB, e0 - cW, mb), ("W", e0 - cW, e0, mb)]
            else:
                t0 = i * cf.c
                ops += [("F", t0, t0 + cF, mb), ("B", t0 + cF, t0 + cF + cB, mb), ("W", t0 + cF + cB, t0 + cf.c, mb)]
        out.append(O.Lane(ops))
    return out


def arm(cf, lay="pieces", emission="forward", policy="need", barrier=False, flush="module", wire_fp32=False,
        rtt=0.0, sites=None, cap_up=None, tree=False, cap_nic=None):
    us = units_of(cf.kind, cf.L, cf.h, cf.V, cf.seq, wire_fp32)
    units = [u for u, _ in us]
    own = owners_of(us, cf.w, lay)
    nic = cf.cap if cap_nic is None else cap_nic
    sites = sites or [0] * cf.w
    peers = [O.Peer(f"p{q}", nic, cf.pcie, f"host{q}", f"site{sites[q]}") for q in range(cf.w)]
    ups = {}
    if len(set(sites)) > 1:
        ups = {(f"site{a}", f"site{b}"): cap_up for a in set(sites) for b in set(sites) if a != b}
    fab = O.Fabric({p.host: O.INF for p in peers}, k=10.0, uplinks=ups, rtt=rtt, site_rtt=rtt, chunk=cf.chunk)
    ln = lanes(cf)
    period = cf.CB if cf.kind == "first" else cf.mw * cf.c
    st = O.StageIn(units, own, peers, ln, period, cf.rates, boundary=cf.A, barrier=barrier, flush=flush, tree=tree,
                   M=cf.w * cf.mw)
    r = O.simulate([st], fab, emission=emission, policy=policy)[0]
    # utilization of the busiest port at c_s (S1's left side over cap, boundary included), as the granted PLAN reports
    eg = [sum(u.b_g * u.params * (1 - own[i][j]) for i, u in enumerate(units)) for j in range(cf.w)]
    ing = [sum(u.b_g * u.params * own[i][j] * (cf.w - 1) for i, u in enumerate(units)) for j in range(cf.w)]
    util = (max(max(eg), max(ing)) + cf.A) / min(nic, cf.pcie) / cf.c
    r.update(C=cf.C, util=util, pct=100 * r["E"] / cf.C, exposed_pct=100 * r["exposed"] / cf.C)
    return r


LAYS = [("today", dict(lay="stride", wire_fp32=True, flush="stage", barrier=True, emission="backward")),
        ("step1", dict(lay="stride", barrier=True)),
        ("lpt", dict(lay="lpt")),
        ("pieces_fwd", dict(lay="pieces", emission="forward")),
        ("pieces_bwd", dict(lay="pieces", emission="backward")),
        ("fifo_pieces_fwd", dict(lay="pieces", emission="forward", policy="fifo")),
        ("fifo_pieces_bwd", dict(lay="pieces", emission="backward", policy="fifo")),
        ("barrier_pieces_fwd", dict(lay="pieces", emission="forward", barrier=True))]

# ------------------------------------------------------------------ (0) round 5's configuration: round5.md section 6
print("== (0) round5.md section 6 anchor at round 5's configuration (F' 50 GFLOP/s, split 1:2:1, HBM' paced fold)")
R5_PIN = {"stage0.pieces_fwd": 2.82, "stage0.step1": 71.31, "stage0.lpt": 20.09, "interior_w4.pieces_fwd": 3.66,
          "interior_w4.step1": 16.00, "interior_w8.pieces_fwd": 10.03, "interior_w8.lpt": 25.10}
R5 = {}
for name, pin in R5_PIN.items():
    tag, lay = name.split(".")
    cf = Conf(tag, 50e9, (0.25, 0.5, 0.25), None)
    r = arm(cf, **dict(LAYS)[lay])
    R5[name] = dict(pinned_pct=pin, spec_pct=round(r["pct"], 3), rel=round(r["pct"] / pin - 1, 4),
                    E_ms=round(r["E"] * 1e3, 2), C_ms=round(r["C"] * 1e3, 1), window_ms=round(r["window"] * 1e3, 3))
    print(f"  {name:24s} granted {pin:6.2f}%  SPEC model {r['pct']:7.3f}%  ({100*(r['pct']/pin-1):+6.2f}%)  "
          f"E {r['E']*1e3:8.2f} ms of C {r['C']*1e3:7.1f}, window {r['window']*1e3:.2f} ms")

# ------------------------------------------------------------------ (1) the E4 configuration
print("== (1) E4 configuration (MEASURED inputs; % of each arm's C)")
for w_ in (4, 8):
    cf = Conf("interior_w4" if w_ == 4 else "interior_w8", FPR[w_], SPL[w_], FOLD)
    print(f"  w {w_}: F' {FPR[w_]/1e9:.3f} GFLOP/s, split {SPL[w_][0]:.4f}:{SPL[w_][1]:.4f}:{SPL[w_][2]:.4f}, cap' = PCIe' "
          f"{cf.cap/1e6:.4f} MB/s, f' {cf.chunk} B, hop latency {2*cf.chunk/cf.pcie*1e3:.3f} ms, c_block "
          f"{8*PF*TP/FPR[w_]*1e3:.2f} ms")
CONF = {tag: Conf(tag, FPR[STAGES[tag][2]], SPL[STAGES[tag][2]], FOLD) for tag in STAGES}
print(f"  stage-0 period C_B' = 32 c_last' = {CONF['stage0'].CB*1e3:.1f} ms, tau = C_B'/(M/w) = "
      f"{CONF['stage0'].CB/32*1e3:.2f} ms per microbatch slot")
ARMS = []


def emit(name, r, w_, extra=None):
    d = dict(arm=name, predicted_ms=round(r["E"] * 1e3, 2), predicted_pct_of_C=round(r["pct"], 3),
             C_ms=round(r["C"] * 1e3, 1), window_ms=round(r["window"] * 1e3, 2),
             exposed_ms=round(r["exposed"] * 1e3, 2), nic_util=round(r["util"], 4), s1_holds=bool(r["util"] <= 1.0),
             w=w_, f6_min_margin_ms=None, f6_min_margin_pct=None)
    if extra:
        d.update(extra)
    ARMS.append(d)
    return d


for tag in STAGES:
    cf = CONF[tag]
    for nm, kw in LAYS:
        emit(f"{tag}.{nm}", arm(cf, **kw), cf.w)


def find(a):
    return next(x for x in ARMS if x["arm"] == a)


F6 = (("stage0.pieces_fwd", "stage0.step1"), ("stage0.pieces_fwd", "stage0.lpt"),
      ("interior_w4.pieces_fwd", "interior_w4.step1"), ("interior_w8.pieces_fwd", "interior_w8.lpt"))
for pa, alt in F6:
    p_, a_ = find(pa), find(alt)
    a_["f6_min_margin_ms"] = round((a_["predicted_ms"] - p_["predicted_ms"]) / 2, 2)
    a_["f6_min_margin_pct"] = round((a_["predicted_pct_of_C"] - p_["predicted_pct_of_C"]) / 2, 3)
    a_["f6_vs"] = pa

for tag in ("interior_w4", "stage0"):
    cf = CONF[tag]
    cW = cf.c * cf.split[2]
    for rname, rtt in (("rtt0", 0.0), ("rtt_2cw", 2 * cW)):
        for em in ("fwd", "bwd"):
            r = arm(cf, lay="pieces", emission="forward" if em == "fwd" else "backward", rtt=rtt)
            emit(f"{tag}.emission_{em}.{rname}", r, cf.w, dict(rtt_ms=round(rtt * 1e3, 3), c_W_ms=round(cW * 1e3, 3)))

cf = CONF["interior_w4"]
for em in ("fwd", "bwd"):
    r = arm(cf, lay="pieces", emission="forward" if em == "fwd" else "backward", sites=[0, 0, 1, 1], cap_up=cf.cap)
    us = [u for u, _ in units_of("mid", 4, HP, VP, SP)]
    util_up = sum(u.b_g * u.params * (2 * 2 / 4) for u in us) / cf.cap / cf.c
    emit(f"interior_w4.uplink_pieces_{em}", r, cf.w,
         dict(cap_up_MBps=round(cf.cap / 1e6, 4), util_up=round(util_up, 4), s1_up=bool(util_up <= 1.0)))

for tag in ("interior_w4", "stage0"):
    cf = CONF[tag]
    r = arm(cf, lay="pieces", emission="forward", tree=True)
    emit(f"{tag}.tree", r, cf.w, dict(f5_left_fold_ms=find(f"{tag}.pieces_fwd")["predicted_ms"]))

# F3 sweep: interior w4 pieces (forward), NIC cap' varied, PCIe' held (so the hop latency is held at its base)
cf = CONF["interior_w4"]
us = [u for u, _ in units_of("mid", 4, HP, VP, SP)]
per_mb = sum(u.b_g * u.params * (cf.w - 1) / cf.w for u in us)
capS1 = (per_mb + cf.A) / cf.c
SW = []
for u_ in (0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0):
    cp = capS1 / u_
    r = arm(cf, lay="pieces", emission="forward", cap_nic=cp)
    SW.append((u_, cp, r))
    emit(f"sweep.util_{u_:.1f}", r, cf.w, dict(cap_MBps=round(cp / 1e6, 4)))


def fit(pts):
    xs = [1e6 / cp for cp, _ in pts]
    ys = [r["pct"] for _, r in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((p - mx) * (q - my) for p, q in zip(xs, ys)) / sum((p - mx) ** 2 for p in xs)
    return my - b * mx, b


lo = fit([(cp, r) for u_, cp, r in SW if u_ < 1])
hi = fit([(cp, r) for u_, cp, r in SW if u_ > 1])
inv = (lo[0] - hi[0]) / (hi[1] - lo[1])
knee = 1e6 / inv
THR = dict(cap_s1_MBps=round(capS1 / 1e6, 4), predicted_threshold_cap_MBps=round(knee / 1e6, 4),
           predicted_threshold_util=round(capS1 / knee, 4), fit_below=dict(a_pct=lo[0], b_pct_MBps=lo[1]),
           fit_above=dict(a_pct=hi[0], b_pct_MBps=hi[1]))
print(f"  sweep: S1 at cap' {capS1/1e6:.4f} MB/s; below {lo[0]:.3f} + {lo[1]:.3f}/cap', above {hi[0]:.3f} + {hi[1]:.3f}/cap'"
      f" -> threshold cap' {knee/1e6:.4f} MB/s (util {capS1/knee:.4f})")
for a in ARMS:
    f6 = "" if a["f6_min_margin_ms"] is None else f"  F6 half-margin {a['f6_min_margin_ms']:.2f} ms ({a['f6_min_margin_pct']:.3f} pts)"
    print(f"  {a['arm']:34s} E {a['predicted_ms']:9.2f} ms = {a['predicted_pct_of_C']:8.3f}% of C {a['C_ms']:8.1f}; "
          f"window {a['window_ms']:7.2f}; util {a['nic_util']:.3f}{f6}")

# ------------------------------------------------------------------ (2) sensitivities of the F6 arms
print("== (2) sensitivity of the F6 arms (% of C)")


def f6set(split_fn, fold, label):
    row = []
    for tag, alts in (("stage0", ("step1", "lpt")), ("interior_w4", ("step1",)), ("interior_w8", ("lpt",))):
        w_ = STAGES[tag][2]
        cf = Conf(tag, FPR[w_], split_fn(w_), fold)
        p = arm(cf, **dict(LAYS)["pieces_fwd"])["pct"]
        for a in alts:
            row.append(f"{tag} pieces {p:.2f} vs {a} {arm(cf, **dict(LAYS)[a])['pct']:.2f}")
    print(f"  {label:34s}: " + "; ".join(row))


def wscale(f):
    def sp(w_):
        a, b, c = SPL[w_]
        c2 = c * f
        s = a + b + c2
        return (a / s, b / s, c2 / s)
    return sp


f6set(lambda w_: SPL[w_], FOLD, "pinned (measured split and fold)")
f6set(lambda w_: (0.25, 0.5, 0.25), FOLD, "split 1:2:1, measured fold")
f6set(lambda w_: SPL[w_], None, "measured split, HBM' paced fold")
f6set(wscale(0.75), FOLD, "W fraction x0.75")
f6set(wscale(1.25), FOLD, "W fraction x1.25")

# scale invariance at the measured split (HBM' paced): full B against B', the F6 arms
print("== (3) scale invariance (measured split, HBM' paced): full B vs B', % of C")
for tag in STAGES:
    w_ = STAGES[tag][2]
    row = []
    for (hh, VV, seq, T, Fp) in ((4096, 32000, 4096, 4096, 400e12), (HP, VP, SP, TP, FPR[w_])):
        cf = Conf(tag, Fp, SPL[w_], None, hh, VV, seq, T)
        row.append([round(arm(cf, **kw)["pct"], 3) for _, kw in LAYS[:5]])
    print(f"  {tag}: full B {row[0]} | B' {row[1]}  (today, step1, lpt, pieces fwd, pieces bwd)")

# ------------------------------------------------------------------ (4) diagnostics: the as-built code paths
# Not predictions. Each arm re-run with ONE of two choices the as-built simulation makes differently, to attribute any
# gap between it and this oracle: (a) stage 0 packed (every op back to back from 0, the idle C_B' - (M/w) c_s at the
# end; the exposure beyond the slot is then E - window_s); (b) one fold after the last arrival instead of the left
# fold of every contribution in canonical order (the gather then leaves at max(arrivals) + 1 fold + AdamW).
print("== (4) diagnostics (not predictions): (a) stage 0 packed, E - window_s; (b) one fold after the last arrival")
_lanes_spec, _chain_spec = lanes, O._Stage._chain_end


def _packed(cf):
    cF, cB, cW = (cf.c * x for x in cf.split)
    out = []
    for q in range(cf.w):
        ops = []
        for i in range(cf.mw):
            mb, t0 = i * cf.w + q, i * cf.c
            ops += [("F", t0, t0 + cF, mb), ("B", t0 + cF, t0 + cF + cB, mb), ("W", t0 + cF + cB, t0 + cf.c, mb)]
        out.append(O.Lane(ops))
    return out


def _single(self, u, d):
    st = self.st
    unit = st.units[u]
    n = unit.params * st.owners[u][d]
    b_g = 4 if st.tree else unit.b_g
    return max(r for _k, r in self.contrib[(u, d)]) + st.rates.fold_t(n, b_g) + st.rates.adam_t(n)


DIAG = {}
for cname, mk in (("E4", lambda tag: CONF[tag]), ("round5", lambda tag: Conf(tag, 50e9, (0.25, 0.5, 0.25), None))):
    for name in ("stage0.pieces_fwd", "stage0.step1", "stage0.lpt", "interior_w4.pieces_fwd", "interior_w8.lpt"):
        tag, lay = name.split(".")
        row = {}
        for dname, ln_fn, ch_fn in (("spec", _lanes_spec, _chain_spec), ("packed_stage0", _packed, _chain_spec),
                                    ("single_fold", _lanes_spec, _single)):
            if dname == "packed_stage0" and tag != "stage0":
                continue
            lanes, O._Stage._chain_end = ln_fn, ch_fn
            r = arm(mk(tag), **dict(LAYS)[lay])
            row[dname] = round(100 * r["exposed"] / r["C"], 3)
        lanes, O._Stage._chain_end = _lanes_spec, _chain_spec
        DIAG[f"{cname}:{name}"] = row
        print(f"  {cname:6s} {name:24s} " + ", ".join(f"{k} {v:.3f}%" for k, v in row.items()))

RESULT = dict(round5_anchor=R5, arms=ARMS, threshold=THR, diagnostics=DIAG,
              Fprime={str(k): v for k, v in FPR.items()}, split={str(k): v for k, v in SPL.items()}, fold=FOLD,
              cap_prime={str(w_): CONF["interior_w4" if w_ == 4 else "interior_w8"].cap for w_ in (4, 8)},
              C_B_prime=CONF["stage0"].CB, chunk_bytes=CONF["stage0"].chunk)
json.dump(RESULT, open(OUT_JSON, "w"), indent=1, sort_keys=True)
print("== (5) JSON written:", OUT_JSON.split("/")[-1])
