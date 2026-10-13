# m2b-e7, SC-4 / SPEC SCHEDULER-I17: E7's expected planner choices re-derived with the model of the SPEC of record.
# The exchange model is the I17 oracle (docs/experiments/m2b-e4/derive/i17_oracle.py), written from the SPEC text and
# independent of src/wantrain. What the granted derivation (e7.py on round 4's exch4) did not model, and this does:
#  (1) each host's DRAM, at k passes per byte per side, is a resource shared max-min fairly INSIDE the exchange
#      simulation, together with the NICs and PCIe paths (I17), for exchange and boundary bytes alike (I16);
#  (2) a unit's gathers leave only after the owner's LAST fold of that unit (a left fold of every microbatch's
#      contribution in canonical order) and its AdamW, at HBM op times ((b_g + 16) B per folded element, 12 for fp32;
#      30 B per updated element), so a hot owner's ingress and its gather egress are serial;
#  (3) ONE simulation of every microbatch's traffic gives E_s and window_s (I17); step = period + max_s
#      max(0, E_s - window_s) + T_ovh, with T_ovh the slowest peer's M folds of its owned elements plus AdamW;
#      the plan takes the emission order (forward or backward, one for the plan) with the smaller step (I17).
# Also as the SPEC's planner states and the granted derivation did not:
#  (4) the candidate set of SCHEDULER task 4: width 1 at the deepest cut, and for w in 2, 4, 8 the deepest cut on
#      N/w devices with pieces + native, zero1 + native and zero1 + fp32 (the 128-GPU fleets gain S 32, w 2 on 64 GPUs,
#      which the granted Bc list lacked);
#  (5) the lane solve's per-hop comm from SPEC section 2.1 (one boundary tensor at the link rate plus RTT/2;
#      sensitivity "comm0" keeps the granted 0).
# Kept from the granted PLAN (inputs): the 17 fleets, k, 600 GB/s host DRAM, stage-contiguous packing, epsilon 3%,
# recompute on, the cut by layer_costs, F/B/W by analytic_costs, F 400 TFLOP/s, PCIe 50 GB/s, HBM 3.35 TB/s,
# T_mb 4096, M = the fleet's GPU count, the uplink hop pricing and floor, the class rule, the tie rule. MEM-HOST is
# charged as the SPEC states it: on pieces only (I16; task 4 keeps zero1 at today's device admission), which settles
# the granted PLAN's open question 5.
# The lane solve (the op timeline every lane runs, SCHEDULER-I13/I19) is lane_solve.py beside this file: a list
# scheduler written from the SPEC's zero-bubble rules and a closed-form lower bound P* it attains, independent of
# src/wantrain (the human's rule, 2026-10-10: nothing from src/wantrain is imported or read). The "lanes" section
# compares every priced candidate's period, E, window and step with the numbers the as-built lane solve produced
# (asbuilt_lanes.json, extracted from the earlier e7_spec.json by extract_asbuilt.py). layer_costs and analytic_costs
# are restated below from the SPEC's formulas (partition.py:164-175 and zero_bubble.py:602-618 are where the planner
# holds them), not imported.
# Modes: spec (primary, pinned); comm0 (sensitivity: the lane solve's hop comm 0, as the granted e7.py); nodram
# (sensitivity: host DRAM out of the simulation, S1 kept, which is the granted model's reading); embed_p0 (diagnostic,
# not a prediction: zero1 gives the whole Embed unit to peer 0 as the as-built planner does; the runtime's stride puts
# pos, parameter 1, on peer 1).
# usage: python3 -I -B e7_spec.py ../../m2b-e4/derive/i17_oracle.py lane_solve.py asbuilt_lanes.json OUT.json [modes] \
#            > e7_spec.out
# (stdout is the record and is identical across runs; wall times go to stderr)
import hashlib
import importlib.util
import json
import sys
import time

A = sys.argv
ORACLE, LANES, ASBUILT, OUT = A[1], A[2], A[3], A[4]
MODES = A[5].split(",") if len(A) > 5 else ["spec", "comm0", "nodram", "embed_p0"]
for p in (A[0], ORACLE, LANES, ASBUILT):
    print(f"input {p.split('/')[-1]}: sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


O = _load("i17_oracle", ORACLE)
LS = _load("lane_solve", LANES)
REC = json.load(open(ASBUILT))["rows"]

# ------------------------------------------------------------------------------------------------ model inputs
h, V, SEQ, NL, NH = 4096, 32000, 4096, 32, 32
TMB = 4096
FDEV = 400e12
PCIE = 50e9
HBM = 3.35e12
HOSTBW = 600e9
EPS = 0.03
CHUNK = 16 * 2**20
TENSOR = TMB * h * 2                   # one bf16 boundary tensor
G9 = 1e9
RATES = O.Rates(HBM, HBM, HBM)


def layer_costs():
    """partition.py:164-175: [Embed 3h, Block h(72h + 12s) x n, Head 6hV]."""
    return [3.0 * h] + [float(h * (72 * h + 12 * SEQ))] * NL + [6.0 * h * V]


def analytic(a, b, recompute=True):
    """zero_bubble.py:602-618 analytic_costs for layers [a, b): F, B, W per token; recompute adds the stage's F to B."""
    per = [(h, h, 0.0)] + [(h * (24 * h + 4 * SEQ), h * (24 * h + 8 * SEQ), 24.0 * h * h)] * NL
    per = per + [(2.0 * h * V, 2.0 * h * V, 2.0 * h * V)]
    f = sum(x[0] for x in per[a:b])
    return f, sum(x[1] for x in per[a:b]) + (f if recompute else 0.0), sum(x[2] for x in per[a:b])


def cut(S):
    """The exact min-max contiguous cut of layer_costs into S stages at equal speeds (proportional_partition)."""
    cs = layer_costs()
    n = len(cs)
    pre = [0.0]
    for x in cs:
        pre.append(pre[-1] + x)
    import functools

    @functools.lru_cache(None)
    def f(i, s):
        if s == 1:
            return pre[n] - pre[i]
        return min(max(pre[j] - pre[i], f(j, s - 1)) for j in range(i + 1, n - s + 2))
    best = f(0, S)
    out, i = [], 0
    for s in range(S, 1, -1):
        for j in range(i + 1, n - s + 2):
            if max(pre[j] - pre[i], f(j, s - 1)) <= best * (1 + 1e-12):
                out.append((i, j))
                i = j
                break
    out.append((i, n))
    return out


def stage_units(a, b, wire):
    """I17's units of layers [a, b): the Embed as one fp32 unit (no W compute; tok, pos are 2 tensors), then each
    block's SplitLinears, then the Head; with today's stride index of each tensor (planner list order)."""
    bg = 2 if wire == "native" else 4
    us = []
    base = 0
    if a == 0:
        us.append((O.Unit("embed", (V + SEQ) * h, 4, 4, 0, 0.0), [(V * h, 0), (SEQ * h, 1)]))
        base = 2
    blocks = list(range(max(a, 1), min(b, NL + 1)))
    for j, _ in enumerate(blocks):
        for nm, p, ix in (("qkv", 3 * h * h, 2), ("proj", h * h, 4), ("fc1", 4 * h * h, 8), ("fc2", 4 * h * h, 10)):
            us.append((O.Unit(nm, p, bg, bg, len(us), float(p)), [(p, base + 12 * j + ix)]))
    if b == NL + 2:
        us.append((O.Unit("head", V * h, bg, bg, len(us), float(V * h)), [(V * h, base + 12 * len(blocks) + 2)]))
    return us, len(blocks)


OWN_MODE = "spec"


def owners(us, w, shard):
    """pieces: every unit in w equal pieces. zero1: today's stride ZeRO-1 over the stage's parameter list, whole
    tensors (runtime/stage.py:183 params[peer::w]; tok and pos are two tensors, model/gpt.py:239-240). Diagnostic mode
    "embed_p0" (not a prediction): the whole Embed unit on its first tensor's peer, as the as-built planner lays it."""
    if shard == "pieces":
        return [[1.0 / w] * w for _ in us]
    rows = []
    for u, tensors in us:
        r = [0.0] * w
        if OWN_MODE == "embed_p0":
            r[tensors[0][1] % w] = 1.0
        else:
            for n, ix in tensors:
                r[ix % w] += n / u.params
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------ the 17 fleets
def cfgs():
    out = []
    for gph, k in ((1, 10), (2, 10), (4, 10), (8, 10), (8, 2)):
        out.append(dict(id=f"32xH100-400G-{gph}gph-k{k}", N=32, link=400, gph=gph, k=k))
    for gph in (1, 8):
        out.append(dict(id=f"32xH100-100G-{gph}gph-k10", N=32, link=100, gph=gph, k=10))
    for gph, k in ((1, 10), (1, 6), (1, 2), (8, 10), (8, 2)):
        out.append(dict(id=f"128xH100-400G-{gph}gph-k{k}", N=128, link=400, gph=gph, k=k))
    for gph in (1, 8):
        out.append(dict(id=f"128xH100-100G-{gph}gph-k10", N=128, link=100, gph=gph, k=10))
    out.append(dict(id="32xH100-2sites-10G-uplink", N=32, link=400, gph=1, k=10, sites=2, uplink=10, site_rtt=0.050))
    out.append(dict(id="32xGPU-mixed-classes", N=32, link=400, gph=1, k=10, mixed=17))
    out.append(dict(id="32xH100-400G-1gph-k10-memhost20", N=32, link=400, gph=1, k=10, ram=20.0))
    for c in out:
        c.setdefault("sites", 1)
    return out


def max_depth(n):
    return n if n <= NL + 2 else NL


def candidates(c):
    N = c["N"]
    out = [dict(S=max_depth(N), w=1, shard=None, wire=None, place="contiguous")]
    for w in (2, 4, 8):
        S = max_depth(N // w)
        places = ["contiguous"] + (["straddling"] if c["sites"] > 1 and w % c["sites"] == 0 else [])
        for pl in places:
            for shard, wire in (("pieces", "native"), ("zero1", "native"), ("zero1", "fp32")):
                out.append(dict(S=S, w=w, shard=shard, wire=wire, place=pl))
    for x in out:
        x["gpus_used"] = x["S"] * x["w"]
        lay = "" if x["w"] == 1 else f" {x['shard']} + {x['wire']}"
        x["name"] = f"S {x['S']} w {x['w']}{lay}" + ("" if x["place"] == "contiguous" else ", straddling")
    return out


def placement(c, x):
    """[[plan GPU per peer] per stage]: stage-contiguous in site and host order; straddling deals each stage's peers
    across the sites in turn (planner task 4, as the granted PLAN prices it)."""
    S, w = x["S"], x["w"]
    if x["place"] == "contiguous":
        return [[s * w + q for q in range(w)] for s in range(S)]
    per = c["N"] // c["sites"]
    pos = [0] * c["sites"]
    rows = []
    for _s in range(S):
        row = []
        for q in range(w):
            si = q % c["sites"]
            row.append(si * per + pos[si])
            pos[si] += 1
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ one candidate
LANE_CACHE = {}


def lanes_of(S, w, M, fbw, comm):
    key = (S, w, M, tuple(fbw), tuple(comm))
    if key in LANE_CACHE:
        return LANE_CACHE[key]
    m = M // w
    nbs = [nb for (_a, _b, nb) in fbw]
    bnd = TMB * h * 2                                # mem_f: one bf16 boundary tensor
    wp = [nb * TMB * 16 * h * 4 for nb in nbs]       # mem_b: the weight-gradient inputs B keeps for W (recompute on)
    per, ms, lb = LS.lanes([x[0] for x in fbw], list(comm), m, [float(bnd)] * S, [float(x) for x in wp])
    LANE_CACHE[key] = (per, ms)
    LANE_BOUND[key] = (ms, lb)
    return per, ms


LANE_BOUND = {}


SIM_CACHE = {}


def price(c, x, mode):
    """-> dict(step, period, C, E per stage, window per stage, overhead, emission, c_s per stage, stage meta)."""
    S, w, N = x["S"], x["w"], c["N"]
    M = N
    parts = cut(S)
    link = c["link"] * 1e9 / 8
    port = min(link, PCIE)
    meta = []
    for s, (a, b) in enumerate(parts):
        f, bb, ww = analytic(a, b)
        us, nb = stage_units(a, b, "native" if w == 1 else x["wire"])
        kind = "first" if a == 0 else ("last" if b == NL + 2 else "mid")
        meta.append(dict(a=a, b=b, kind=kind, nb=nb, f=f * TMB / FDEV, b_=bb * TMB / FDEV, w_=ww * TMB / FDEV, us=us,
                         c=(f + bb + ww) * TMB / FDEV, A=TENSOR * (1 if s in (0, S - 1) else 2)))
    rows = placement(c, x)
    sites_of = (lambda g: g // (N // c["sites"])) if c["sites"] > 1 else (lambda g: 0)
    comm = []
    cross = 0
    for s in range(S - 1):
        if {sites_of(g) for g in rows[s]} != {sites_of(g) for g in rows[s + 1]}:
            comm.append(w * TENSOR / (c["uplink"] * 1e9 / 8) + c["site_rtt"] / 2)
            cross += 1
        else:
            comm.append(0.0 if mode == "comm0" else TENSOR / link)
    floor = (M * TENSOR / (c["uplink"] * 1e9 / 8) + c["site_rtt"]) if cross else 0.0
    fbw = [((m_["f"], m_["b_"], m_["w_"]), None, m_["nb"]) for m_ in meta]
    lane, period = lanes_of(S, w, M, fbw, comm)
    cmax = max(m_["c"] for m_ in meta)
    C = (M // w) * cmax
    out = dict(period=period, C=C, cs=[m_["c"] for m_ in meta], floor=floor, meta=meta, rows=rows)
    if w == 1:
        ovh = 0.0
        for m_ in meta:
            st_units = [u for u, _ in m_["us"]]
            t = sum(M * RATES.fold_t(u.params, u.b_g) + RATES.adam_t(u.params) for u in st_units)
            ovh = max(ovh, t)
        out.update(E=[0.0] * S, window=[0.0] * S, exposed=0.0, overhead=ovh, emission=None,
                   step=max(period + ovh, floor))
        return out
    key = (c["N"], c["link"], c["gph"], c["k"], c["sites"], c.get("uplink"), x["S"], x["w"], x["shard"], x["wire"],
           x["place"], mode)
    if key in SIM_CACHE:
        out.update(SIM_CACHE[key])
        return out
    best = None
    for emission in ("forward", "backward"):
        stages = []
        hosts = {}
        for s, m_ in enumerate(meta):
            peers = []
            for q, g in enumerate(rows[s]):
                host = f"h{g // c['gph']}"
                hosts[host] = O.INF if mode == "nodram" else HOSTBW
                peers.append(O.Peer(f"g{g}", link, PCIE, host, f"site{sites_of(g)}"))
            ln = []
            for q in range(w):
                ln.append(O.Lane([(t, st, en, mb * w + q) for (t, st, en, mb) in lane[s]]))
            units = [u for u, _ in m_["us"]]
            own = owners(m_["us"], w, x["shard"])
            stages.append(O.StageIn(units, own, peers, ln, period, RATES, boundary=m_["A"],
                                    barrier=(x["shard"] == "zero1"), M=M))
        ups = {}
        if c["sites"] > 1:
            ups = {("site0", "site1"): c["uplink"] * 1e9 / 8, ("site1", "site0"): c["uplink"] * 1e9 / 8}
        fab = O.Fabric(hosts, k=float(c["k"]), uplinks=ups, rtt=0.0, site_rtt=c.get("site_rtt", 0.0), chunk=CHUNK)
        res = O.simulate(stages, fab, emission=emission)
        exposed = max(r["exposed"] for r in res)
        ovh = max(r["overhead"] for r in res)
        step = max(period + exposed + ovh, floor)
        cand = dict(E=[r["E"] for r in res], window=[r["window"] for r in res], exposed=exposed, overhead=ovh,
                    emission=emission, step=step)
        if best is None or step < best["step"]:
            best = cand
    SIM_CACHE[key] = best
    out.update(best)
    return out


def dram_rate(m_, w, k):
    """S1 on host DRAM for one GPU of the stage (I16): k (b_g P_s 2(w-1)/w + 2A) / c_s, bytes/s."""
    bgP = sum(u.b_g * u.params for u, _ in m_["us"])
    return k * (bgP * 2 * (w - 1) / w + 2 * m_["A"]) / m_["c"]


def host_hold(m_, w, M, shard):
    us = m_["us"]
    own = owners(us, w, shard)
    Mk = M // w
    worst = 0.0
    for j in range(w):
        g = sum(u.b_g * u.params * ((1 - own[i][j]) * Mk + own[i][j] * (M - Mk)) for i, (u, _) in enumerate(us))
        p = sum(u.b_p * u.params * (own[i][j] * (w - 1) + (1 - own[i][j])) for i, (u, _) in enumerate(us))
        worst = max(worst, g + p)
    return worst


def admit(c, x, r):
    """SCHEDULER-I16 refusals: S1 NIC, S1 host DRAM, S1 uplink, S2 and MEM-HOST for pieces (I16 admits shard "pieces"
    under its five conditions; SCHEDULER task 4: zero1 wide stages "keep today's admission (device memory)"; HANDOUT-M2b
    rulings: "zero1 stages keep today's FIT check"); the class for every wide candidate, as the granted PLAN charges it
    and the as-built planner checks it (admission.check_class, declared class for zero1). -> list of reasons."""
    S, w = x["S"], x["w"]
    reasons = []
    if w == 1:
        return reasons
    meta = r["meta"]
    if c.get("mixed"):
        for s, row in enumerate(r["rows"]):
            if min(row) < c["mixed"] <= max(row):
                reasons.append(dict(kind="class", where=f"stage {s}", value="A+B", bound="one class", unit=""))
        return reasons                            # refused before pricing (no S1/S2 reasons are reported)
    pieces = x["shard"] == "pieces"
    if pieces and x["place"] == "straddling":
        cap_up = c["uplink"] * 1e9 / 8
        n_a = n_b = w // 2
        rate = sum(sum(u.b_g * u.params for u, _ in m_["us"]) * n_a * n_b / w / m_["c"] for m_ in meta)
        reasons.append(dict(kind="S1 uplink", where="site0-site1", value=round(rate / G9, 1), bound=cap_up / G9,
                            unit="GB/s"))
        return reasons
    port = min(c["link"] * 1e9 / 8, PCIE)
    if pieces:
        seen = set()
        for m_ in meta:
            bgP = sum(u.b_g * u.params for u, _ in m_["us"])
            util = (bgP * (w - 1) / w + m_["A"]) / port / m_["c"]
            if util > 1.0 and m_["kind"] not in seen:
                seen.add(m_["kind"])
                reasons.append(dict(kind="S1 NIC", where=f"{m_['kind']} stage", value=round(util, 3), bound=1.0,
                                    unit="x cap*c_s"))
        hosts = {}
        for s, row in enumerate(r["rows"]):
            for g in row:
                hosts.setdefault(g // c["gph"], []).append(s)
        worst = None
        for hid in sorted(hosts):
            rate = sum(dram_rate(meta[s], w, c["k"]) for s in hosts[hid])
            if rate > HOSTBW and (worst is None or rate > worst[1]):
                worst = (hid, rate, sorted(set(hosts[hid])))
        if worst:
            reasons.append(dict(kind="S1 host DRAM", where=f"host {worst[0]} (stages {worst[2]})",
                                value=round(worst[1] / G9, 1), bound=HOSTBW / G9, unit="GB/s"))
        bad = [(r["E"][s] - r["window"][s], s) for s in range(S)]
        v, s = max(bad)
        if v > EPS * r["C"]:
            reasons.append(dict(kind="S2", where=f"stage {s} ({meta[s]['kind']})", value=round(v * 1e3, 2),
                                bound=round(EPS * r["C"] * 1e3, 2), unit="ms"))
    if pieces and c.get("ram") is not None:
        worst = max((host_hold(m_, w, c["N"], x["shard"]), m_["kind"], m_["nb"]) for m_ in meta)
        if worst[0] > c["ram"] * G9:
            reasons.append(dict(kind="MEM-HOST", where=f"host of a {worst[1]} ({worst[2]} blocks) peer",
                                value=round(worst[0] / G9, 2), bound=c["ram"], unit="GB"))
    return reasons


def bound_straddling_zero1(c, x, r):
    """A lower bound on a straddling zero1 step (uplink-bound): every gradient byte that crosses the uplink, per
    direction, at the uplink rate. It decides only that the candidate cannot be the choice or the runner-up."""
    w = x["w"]
    vol = 0.0
    for m_ in r["meta"]:
        own = owners(m_["us"], w, "zero1")
        for i, (u, _) in enumerate(m_["us"]):
            for d in range(w):
                if own[i][d] > 0:
                    senders_other_site = sum(1 for q in range(w) if q % 2 != d % 2)
                    vol += (c["N"] // w) * u.b_g * u.params * own[i][d] * senders_other_site
    return vol / 2 / (c["uplink"] * 1e9 / 8)


def run(mode):
    res = []
    for c in cfgs():
        t0 = time.time()
        rows = []
        for x in candidates(c):
            if c.get("mixed") and x["w"] > 1:
                reasons = admit(c, x, dict(rows=placement(c, x), meta=None))
                rows.append(dict(x, step=None, admitted=False, reasons=reasons))
                continue
            if x["place"] == "straddling" and x["shard"] == "zero1":
                r = price(c, dict(x, place="contiguous"), mode)      # meta only
                lb = bound_straddling_zero1(c, x, r)
                rows.append(dict(x, step=None, step_lower_bound=lb, admitted=True, reasons=[]))
                continue
            if x["place"] == "straddling" and x["shard"] == "pieces":
                r = price(c, dict(x, place="contiguous"), mode)
                r["rows"] = placement(c, x)
                rows.append(dict(x, step=None, admitted=False, reasons=admit(c, x, r)))
                continue
            r = price(c, x, mode)
            reasons = admit(c, x, r)
            S = x["S"]
            rows.append(dict(x, step=r["step"], period=r["period"], C=r["C"], exposed=r["exposed"],
                             overhead=r["overhead"], emission=r["emission"], E=r["E"], window=r["window"],
                             E_minus_window_max=max(r["E"][s] - r["window"][s] for s in range(S)),
                             binding_stage=max(range(S), key=lambda s: r["E"][s] - r["window"][s]),
                             admitted=not reasons, reasons=reasons, cs=r["cs"]))
        adm = sorted([q for q in rows if q["admitted"] and q["step"] is not None], key=lambda q: q["step"])
        best = adm[0]
        ru = adm[1] if len(adm) > 1 else None
        for q in rows:
            if q.get("step_lower_bound") is not None:
                assert q["step_lower_bound"] > (ru or best)["step"], ("bound does not exclude", c["id"], q["name"])
        margin = round(100 * (ru["step"] / best["step"] - 1), 2) if ru else None
        res.append(dict(config=c, rows=rows, choice=best["name"], step=best["step"], runner_up=ru["name"] if ru else None,
                        runner_up_step=ru["step"] if ru else None, margin_pct=margin))
        print(f"  [{mode}] {c['id']:34s} {best['name']:26s} {best['step']*1e3:8.1f} ms | runner-up "
              f"{(ru or {}).get('name')} {ru['step']*1e3 if ru else float('nan'):8.1f} (+{margin}%)")
        print(f"  [{mode}] {c['id']}: {time.time()-t0:.1f} s", file=sys.stderr)   # wall time off the record
    return res


def strip(r):
    out = []
    for c in r:
        rows = []
        for q in c["rows"]:
            d = {k: v for k, v in q.items() if k not in ("cs",)}
            for k in ("step", "period", "C", "exposed", "overhead", "E_minus_window_max", "step_lower_bound"):
                if d.get(k) is not None:
                    d[k] = round(d[k] * 1e3, 3)
            for k in ("E", "window"):
                if d.get(k) is not None:
                    d[k + "_ms"] = [round(v * 1e3, 3) for v in d.pop(k)]
            rows.append(d)
        out.append(dict(c, rows=rows, step=round(c["step"] * 1e3, 1),
                        runner_up_step=None if c["runner_up_step"] is None else round(c["runner_up_step"] * 1e3, 1)))
    return out


ALL = {}
for mode in MODES:
    OWN_MODE = "embed_p0" if mode == "embed_p0" else "spec"
    print(f"== mode {mode}" + ("  (diagnostic, not a prediction: the as-built zero1 Embed ownership)" if mode == "embed_p0" else ""))
    t0 = time.time()
    ALL[mode] = strip(run(mode))
    print(f"   ({time.time()-t0:.0f} s)", file=sys.stderr)

print("== lanes: the independent lane solve (lane_solve.py) against its closed-form bound P* and the as-built record")
print("   (asbuilt_lanes.json: the same candidates priced with src/wantrain/scheduler/zero_bubble.py, sha256 "
      + json.load(open(ASBUILT))["lane_solver_sha256"][:16] + "...)")
for key in sorted(LANE_BOUND, key=lambda k: (k[0], k[1], k[2], k[4])):
    ms, lb = LANE_BOUND[key]
    print(f"  S {key[0]:2d} w {key[1]} M {key[2]:3d} hop comm max {max(key[4] or (0.0,))*1e3:7.3f} ms: period "
          f"{ms*1e3:9.3f} ms, P* {lb*1e3:9.3f} ms ({'attained' if abs(ms - lb) <= 1e-9 * lb else 'NOT attained'})")
worst = dict(period=(0.0, None), step=(0.0, None), E=(0.0, None), window=(0.0, None))
n_cmp = 0
off = []
for mode, res in ALL.items():
    for c in res:
        for q in c["rows"]:
            if q.get("period") is None:
                continue
            ref = REC[f"{mode}|{c['config']['id']}|{q['name']}"]
            n_cmp += 1
            for k in ("period", "step"):
                d = abs(q[k] / ref[k] - 1)
                if d > worst[k][0]:
                    worst[k] = (d, f"{mode}|{c['config']['id']}|{q['name']}")
                if d > 1e-3:
                    off.append((mode, c["config"]["id"], q["name"], k, ref[k], q[k]))
            for k, kk in (("E", "E_ms"), ("window", "window_ms")):
                for a, b in zip(q.get(kk) or [], ref.get(kk) or []):
                    d = abs(a - b) / max(1.0, abs(b))
                    if d > worst[k][0]:
                        worst[k] = (d, f"{mode}|{c['config']['id']}|{q['name']}")
print(f"  {n_cmp} priced candidates compared over modes {', '.join(ALL)}:")
for k, (d, where) in worst.items():
    print(f"    largest relative difference in {k:6s}: {d:.2e}" + (f" ({where})" if where else ""))
print(f"  candidates whose period or step differs by more than 0.1%: {len(off)}")
for o in off:
    print(f"    {o[0]} {o[1]} {o[2]}: {o[3]} as-built {o[4]} -> {o[5]}")
print("== per-stage S1 on host DRAM per GPU (I16, planner c_s), and S2 at the binding stage (mode spec)")
for S_, w_ in ((32, 4), (32, 2), (16, 2), (8, 4), (4, 8), (16, 8)):
    parts = cut(S_)
    seen = set()
    for s, (a, b) in enumerate(parts):
        f, bb, ww = analytic(a, b)
        us, nb = stage_units(a, b, "native")
        kind = "first" if a == 0 else ("last" if b == NL + 2 else "mid")
        if kind in seen:
            continue
        seen.add(kind)
        m_ = dict(us=us, c=(f + bb + ww) * TMB / FDEV, A=TENSOR * (1 if s in (0, S_ - 1) else 2))
        print(f"  S {S_:2d} w {w_} {kind:5s} ({nb} blocks): c_s {m_['c']*1e3:6.2f} ms; DRAM per GPU k 2/6/10: "
              + " / ".join(f"{dram_rate(m_, w_, k)/G9:6.1f}" for k in (2, 6, 10)) + " GB/s; a 1-GPU 600 GB/s host "
              f"admits k <= {HOSTBW/dram_rate(m_, w_, 1):.2f}")
json.dump(dict(modes=ALL, cut={str(S_): cut(S_) for S_ in (4, 8, 16, 32)},
               layer_costs=dict(embed=layer_costs()[0], block=layer_costs()[1], head=layer_costs()[-1])),
          open(OUT, "w"), indent=1, sort_keys=True, default=str)
print("== JSON written:", OUT.split("/")[-1])
