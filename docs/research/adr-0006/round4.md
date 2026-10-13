# ADR-0006 round-4 corrections

Research note behind `docs/design/adr/ADR-0006-exact-fsdp-placement.md`, round 4. It decides nothing and changes no
mechanism. It corrects the round-3 accounting (`round3.md`) where the round-4 review found terms missing or
mispriced, and it supersedes `round3.md` §2 (host DRAM numbers), §4 (point B), §5 and §6 where they differ. Every number
is DERIVED by the script in §9 on this machine's CPU (no GPU, no network measured). Inputs are HYPOTHESIS as in
`round3.md`, plus: `seq_len` 4096 (a `pos` table of 4096·h), and the Head's GEMM at 8 FLOPs per param per token like
every `SplitLinear`.

## 1. Embed and Head are units of the first and last stage (review finding 3)

FACT (code, this tree): `Embed` holds `tok = nn.Embedding(V, h)` and `pos = nn.Embedding(seq_len, h)`
(`gpt.py:239-240`); they are not `SplitLinear`s, so their parameters and gradients stay fp32 and are produced in B
(`gpt.py:247-249`; SPEC-runtime §2: "Parameters outside split linears (embeddings, norms, tied weights) get their
gradients in B"). `Head.out = nn.Linear(d_model, vocab_size, bias=False)` (`gpt.py:317`) is untied by default
(`gpt.py:97`), so `wgrad.convert` makes it a `SplitLinear` (`wgrad.py:139`): a bf16 gradient under a bf16 recipe and
`V·h` of extra GEMM. Round 2 and round 3 modelled interior stages only (`derivation.md` §9 `units()`).

Correction (accounting only). The round-4 script adds `tok` and `pos` (`b_g = b_p = 4`, emitted when W(mb) starts,
forward unit index first) to stage 0 and `head` (`b_g = b_p = 2`, a W unit, forward index last) to the last stage,
cuts `[Embed, Block × 32, Head]` by exact min-max (SCHEDULER-I9), and takes C from the slowest stage. At point B
(7B, 128 H100, S 32, w 4, M 128, `T_mb` 4096) the cut is `[Embed + Block 0] … [Block 31 + Head]`: c per microbatch is
16.49 ms on stage 0 and the interior and 27.23 ms on the last stage, so **C = 32 × 27.23 = 871.4 ms** (not 527.8) and
the lane makespan (repository solver, per-stage costs) is 1041.1 ms. Per stage, 400 Gbps, exposure E in ms (% of C):

| layout | stage 0 (Embed + Block 0) | interior | last (Block 31 + Head) | step: makespan + worst E + T_ovh |
|---|---|---|---|---|
| today (stride, fp32, barrier) | 1075.7 (123.4), S1 fails | 89.5 (10.3) | 1080.4 (124.0) | 2163.3 ms |
| step 1 alone (`wire: native`, stride, barrier) | 621.3 (71.3), S1 fails | 24.6 (2.8) | 122.7 (14.1) | 1721.3 ms |
| LPT whole tensors, per-unit | 175.1 (20.1), S1 fails | 20.9 (2.4) | 38.6 (4.4) | 1275.1 ms |
| pieces, forward emission | 24.5 (2.8), S1 util 0.94 | 7.8 (0.9) | 11.9 (1.4) | 1124.5 ms |

Stage 0 binds: its window is zero (nothing upstream delays its F(t+1)), it carries 0.349B params of which 0.148B are
fp32 tables, and under a whole-tensor layout the 524 MB `tok` gradient lands on one owner. S2 at stage 0 holds for
pieces (24.5 ≤ ε·C = 26.1 ms) and fails for every alternative. `T_ovh` per stage: 53.4 ms (stage 0, fp32 fold of the
tables), 36.1 (interior), 58.9 (last). At 100 Gbps every layout fails S1 (pieces: stage 0 1191.7 ms).

## 2. F6 at point B, and LPT's device state (findings 1 and 7)

Round 3's point-B arm measured an interior stage, where pieces beat step 1 alone by 3.2% and LPT by 2.5% of the old C;
at the corrected C those are 1.9% and 1.5%, below ε, and interior exposure hides behind the fill window anyway. The
stage that binds is stage 0, where the margins are 596.8 ms (68.5% of C) over step 1 alone and 150.6 ms (17.3%) over
LPT. Memory is no reason at 7B, so F6 has no memory fallback: it refutes on time at any one arm.

LPT's device state priced per `L_s` and w (state = `b_p` + 16·largest owned share + `b_g`·largest module share):

| point | pieces | LPT (whole tensors) |
|---|---|---|
| A, w 4, `L_s` 4 | 6.17 | 6.17 (balanced) |
| A, w 8, `L_s` 4 | 4.17 | 4.50 (largest owner 7 of 48 h²) |
| B interior, w 4, `L_s` 1 | 6.67 | 8.00 (fc1 alone: 4 of 12 h²) |
| B stage 0, w 4 | 8.35 | 10.35 (`tok` alone: 0.375 of the stage) |

At `P_s` ≤ 0.35B these differ by ≤ 0.7 GB per device; no 80 GB device notices.

## 3. Depth against width, and E7's expected choice (finding 2)

32 H100, M 32, per-stage costs with Embed and Head; total = lane makespan + worst-stage exposure + `T_ovh`, ms. Pieces
are admitted only where S1 and S2 hold (SCHEDULER-I16); `zero1` candidates are priced by the same simulation:

| (S, w) | 400 Gbps: zero1 + native | pieces | 100 Gbps: zero1 + native | pieces |
|---|---|---|---|---|
| (32, 1) | 1102.2 (no exchange) | — | 1102.2 | — |
| (16, 2) | 994.4 | **966.8**, admitted (E 13.5 ≤ 21.0) | 2021.2 | 1263.7, refused (S1) |
| (8, 4) | 1013.5 | 902.5, refused (S2: E 36.0 > 18.4) | 3177.7 | 1497.2, refused |
| (4, 8) | 1163.9 | 867.8, refused (S2: E 69.5 > 17.1) | 3772.1 | 1639.7, refused |

Round 3's "the planner keeps depth" was wrong twice. Its own table already put (4, 8) 1.5 points ahead; with the Head
counted, depth pays the Head's imbalance most (slowest stage 1.65× at S 32, 1.08× at S 4), so width shortens the step
at 400 Gbps whatever the layout. **E7 expects:** 400 Gbps → (16, 2) with pieces, 966.8 ms (12.3% below depth, 2.8%
below `zero1` + native at the same width); 100 Gbps → (32, 1), 1102.2 ms. S2 refuses the two fastest plans at
400 Gbps: ε is the human's bound on exposure, not a step-time optimum (ADR R7).

## 4. The planner when SCHEDULER-I16 refuses pieces (finding 8)

128 H100 (7B, M 128), 400 Gbps; step ≥ max(makespan + E + `T_ovh`, host DRAM time), host DRAM time with every GPU's
exchange and boundary bytes spread evenly over the hosts (a lower bound for any placement), 600 GB/s per host:

| candidate | 1 GPU per host | 8 per host, k 10 | 8 per host, k 2 |
|---|---|---|---|
| S 32, w 4, pieces | **1124.5** | 3388.3 (refused: S1 host DRAM) | 1124.5 simulated, refused by S1 as written (§5) |
| S 32, w 4, `zero1` + native | 1721.3 | **3388.3** | **1721.3** |
| S 32, w 4, `zero1` today (fp32) | 2163.3 | 6099.9 | 2163.3 |
| S 32, w 1 on 32 GPUs, 96 idle | 3887.8 | 3887.8 | 3887.8 |

So a job that runs today keeps a plan: where I16 refuses pieces, the `zero1` candidates and the narrower plan remain,
and the planner takes the smallest simulated step. On 8-GPU hosts at k 10 it takes `zero1` + native (link-bound,
3388.3 ms, the same host bytes as pieces, so the refusal costs nothing there); at k 2 it also takes `zero1` + native
(1721.3 ms), because S1 as written refuses pieces there (§5). At 100 Gbps it takes S 32, w 1 on 32 of
the 128 GPUs (3887.8 ms), although pieces would simulate at 2291.7 ms: I16 refuses a link-bound pieces plan even when it
beats every admitted one (ADR R13, OPEN for the human).

## 5. Host DRAM carries boundary bytes at k passes too (finding 5)

Pipeline-boundary frames go through the same host-staged transport (`serial.py`, `framing.py`), so S1 on host DRAM is
`Σ k·(b_g·P_s·share + A) ≤ cap·c`, not `Σ k·b_g·P_s·share + A`. Interior stage, per GPU, against a 600 GB/s host:

| k per side | point A (S 8, w 4) | point B interior (S 32, w 4) | 8 GPUs at B: host time per step |
|---|---|---|---|
| 2 | 86.5 GB/s, ≤ 6.9 GPUs (was 7.3) | 91.8 GB/s, ≤ 6.5 (was 7.9) | 646 ms (was 532) |
| 4 | 172.9, ≤ 3.5 | 183.6, ≤ 3.3 | 1292 |
| 10 | 432.3, ≤ 1.4 (was 1.5) | 459.0, ≤ 1.3 (was 1.6) | 3230 (was 2658) |

Rates are per the stage's own c, as S1 is written; stage 0 carries 25.68 GB per direction per step against the
interior's 12.11. S1 in this form is conservative when stages are unequal: at point B the period is the Head stage's,
so 8-GPU hosts at k 2 need 677.7 ms of host time per 1124.5 ms step in the simulation while S1 refuses them. Using the
pipeline period in S1 is a predicate change (ADR R10, OPEN).

## 6. Sub-block depth priced alike (finding 6)

Sub-block cuts carry activations across every cut, through the same NIC and host-staged transport. 128 H100, S 128,
w 1, M 128, per GPU per direction per step: 8.6 GB with h-wide cuts (a lower bound), 23.6 GB with per-linear cuts
(boundaries 4h, h, 5h, h: the residual is carried). At k 10 on 8-GPU hosts that is 2604 and 7161 GB/s against 600
(≤ 1.8 and ≤ 0.7 GPUs per host); at k 2, ≤ 9.2 and ≤ 3.4. The NIC carries 130 and 358 Gbps. So host DRAM binds
sub-block depth as it binds pieces, and R10's transport change is needed by either path.

On time, with Embed and Head: the Head (`V·h` = 7.8 h²) cannot be cut by any sub-block scheme, so deeper cuts make it
a larger share of the slowest stage. Lane makespan at M 128 (repository solver, exact min-max cut):

| cuts | slowest stage / mean | C | lane makespan |
|---|---|---|---|
| per block, S 32, w 4 (point B) | 1.62 | 871.4 ms | 1041.1 ms (+ E 24.5 + `T_ovh` 58.9 = 1124.5) |
| halves, S 64, w 2 | 1.96 | 1055.5 | 1176.5 (+ exposure + `T_ovh`) |
| per linear, S 128, w 1 | 2.55 | 1374.4 | 1427.6 (+ `T_ovh`) |

Round 3's "sub-block cuts tie point B" (49.6% vs 49.9%) priced equal stages without Embed and Head; they are at least
303 ms (27% of pieces' step) slower per linear and ≥ 52 ms slower in halves. They still fit a 1T layer per linear at w 1 (fc1 48.3 GB).

## 7. The memory pillar priced by step time (finding 4)

The largest dense layer as its own stage, at the largest `M/w` that fits (`round3.md` §1), ZB-1p lane bubble
`(S−1)/(2·M/w)`, host one-step bound `b_g·P_s·2·(M/w)·(w−1)/w` per peer:

| layer (stages) | w, admitted M/w | bubble over C | host bound per peer |
|---|---|---|---|
| 405B (126) | w 2, 9 / w 4, 25 / w 8, 58 | 694% / 250% / 108% | 58 / 242 / 654 GB |
| 1T (138) | w 4, 4 / w 8, 14 | 1712% / 489% | 87 / 355 GB |
| 1T, per-op regather (no resident copy) | w 4, 7 / w 8, 22 | 979% / 311% | 152 / 558 GB |

A 405B layer fits at w 1 (62.3 GB of 70.4) and needs no width. A 1T layer does not fit at w 1 (133.7 GB); every
admitted width leaves a step of 6–18 C, and per-op regather does not change that. So at whole-block cuts no placement
gives a usable 1T step, with or without this ADR; the decision rests on forced width and on width at 400 Gbps (§1, §3).
The memory claims stay true as fits, not as plans.

## 8. Lane-consistent orders, and the M2 row (findings 9 and 10)

`derivation.md` §5's engine-bridge defect (a wide stage solved as one server, then split per peer: 109.4% against the
lane's 43.8% at S 8, w 4, M 32) predates this ADR and changes `zero1` runs too. The amendment no longer puts it in
SCHEDULER task 4: it stays a house finding, and every number above (lane makespans) assumes that fix lands before
M2b's planner work (ADR R14). DESIGN §8's M2 row loses "plan mode for wide stages", which SCHEDULER task 4 moved to
M2b in round 0.

## 9. Script

`python3 -I r4.py r3.py r2.py src/wantrain/scheduler/zero_bubble.py`, where `r3.py` is `round3.md` §7 (sha256
`acf1cb86…995ed6`, extracted with `awk '/^```python/{f++; next} /^```/{if(f==1){exit}} f==1' round3.md`), `r2.py` is
`derivation.md` §9 (`f43ab4f6…b701eb`) and `zero_bubble.py` is this tree's (`22988afc…751ac8d`). Exit 0, 3.7 s.
sha256 of the script as run: `d4838233e7e7be28cce19a1775c6cba2e298ac62ddebfbc0642a1affabcdd0c0`; of its output: `7ea94c7fcd5c80fdcb4d900cb3e23bdb2810c188943827c617407ba6b88ecdbe`.

```python
# ADR-0006 round-4 corrections. Loads the round-3 model (round3.md §7, sha256 acf1cb86...995ed6; it loads the
# round-2 model, derivation.md §9, sha256 f43ab4f6...b701eb) and adds, without changing any mechanism:
#  (1) Embed (tok V*h + pos seq*h, fp32 gradient produced at B, fp32 working copy) and Head (out V*h, a SplitLinear:
#      bf16 gradient, extra compute) as units of the first and last stage; per-stage c, S1, exposure and T_ovh;
#      the pipeline period from the slowest stage (layer cut by exact min-max, SCHEDULER-I9);
#  (2) depth against width at 32 H100 and point B at 128 H100 through the repository's solver (lane solve, per-stage
#      costs), 100 and 400 Gbps, with the corrected stages; candidates for E7 and for the I16-refused fallback;
#  (3) S1 with boundary bytes scaled by k on host DRAM; sub-block depth priced with the same S1;
#  (4) LPT device state per L_s and w;
#  (5) the memory pillar priced by step time (bubble at the admitted M/w) and host bound.
# Every input is HYPOTHESIS as in round 3, plus: seq_len 4096 (pos table 4096*h), Head FLOPs 8 per param per token.
import sys, importlib.util
# usage: python3 -I r4.py r3.py r2.py src/wantrain/scheduler/zero_bubble.py
src3 = open(sys.argv[1]).read(); ZB_PATH = sys.argv[3]
sys.argv = [sys.argv[0], sys.argv[2]]
exec(src3[: src3.index("rows = [")])
spec = importlib.util.spec_from_file_location("zb", ZB_PATH)
zb = importlib.util.module_from_spec(spec); sys.modules["zb"] = zb; spec.loader.exec_module(zb)

SEQ = 4096
HOSTBW = 600e9
F_H100 = 400e12

def block_units(b0, base_idx):
    # (name, params, b_g, b_p, flop_params, at_B, stride_index)
    idx = {"qkv": 2, "proj": 4, "fc1": 8, "fc2": 10}
    return [(f"b{b0}.{n}", s * h * h, 2, 2, s * h * h, False, base_idx + idx[n]) for n, s in MOD]

def stage_units(kind, L):
    us = []
    if kind == "first":
        us += [("tok", V * h, 4, 4, 0, True, 0), ("pos", SEQ * h, 4, 4, 0, True, 1)]
        for b in range(L):
            us += block_units(b, 2 + 12 * b)
    else:
        for b in range(L):
            us += block_units(b, 12 * b)
        if kind == "last":
            us += [("head", V * h, 2, 2, V * h, False, 12 * L + 2)]   # ln_f.w, ln_f.b at 12L, 12L+1
    return us

def own_of(us, w, lay):
    if lay == "pieces":
        return [[1.0 / w] * w for _ in us]
    if lay == "stride":
        out = []
        for u in us:
            r = [0.0] * w; r[u[6] % w] = 1.0; out.append(r)
        return out
    if lay == "lpt":
        ts = sorted(range(len(us)), key=lambda i: (-us[i][1], i)); load = [0] * w
        out = [[0.0] * w for _ in us]
        for i in ts:
            j = load.index(min(load)); load[j] += us[i][1]; out[i][j] = 1.0
        return out
    raise ValueError(lay)

def exch4(us, own, w, cap, c, rtt=0.0, order="fwd", barrier=False, flush="module", lat=0.0, policy="need",
          chunk=F_COPY, k=1):
    """exch3 (round3.md §7) with per-unit b_g, b_p, FLOP weight and B-produced units (Embed)."""
    U = len(us); Pfl = sum(u[4] for u in us); cW = cF = c / 4
    wunits = [i for i in range(U) if not us[i][5]]
    seq = wunits if order == "fwd" else list(reversed(wunits))
    prod = {}
    for j in range(k):
        t = j * cW
        for i in range(U):
            if us[i][5]:
                prod[(j, i)] = j * cW if flush == "module" else (j + 1) * cW   # produced at B, before W_j
        for u in seq:
            t += cW * us[u][4] / Pfl
            prod[(j, u)] = t if flush == "module" else (j + 1) * cW
    items = []; queues = {}; nseq = [0]
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
                    add(t, 1, u, s, d, j, us[u][2] * us[u][1] * own[u][d])
    need_g = {(u, d): (w - 1 if own[u][d] > 0 else 0) for u in range(U) for d in range(w)}
    got = {}; upd = {(u, d): prod[(k - 1, u)] for u in range(U) for d in range(w) if own[u][d] > 0}
    ready = {(u, d): 0.0 for u in range(U) for d in range(w)}
    owners_total = sum(1 for x in need_g if need_g[x] > 0); done_owners = []
    def optime(u, o):
        n = us[u][1] * own[u][o]
        return n * FOLD_B(us[u][2]) / HBM + n * ADAM_B(us[u][3]) / HBM
    def gathers(u, o, t):
        for d in range(w):
            if d != o:
                add(t, 0, u, o, d, -1, us[u][3] * us[u][1] * own[u][o])
    active = {}; t = 0.0
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
        flows = list(active.keys()); rate = {}; rs = {}
        for q in flows:
            for r in (("e", q[0]), ("i", q[1])):
                rs.setdefault(r, []).append(q)
        capl = {r: cap for r in rs}; unfixed = set(flows)
        while unfixed:
            best = None
            for r, fl in rs.items():
                uf = [f for f in fl if f in unfixed]
                if uf and (best is None or capl[r] / len(uf) < best[0]):
                    best = (capl[r] / len(uf), r, uf)
            share, r, uf = best
            for f in uf:
                rate[f] = share; unfixed.discard(f)
                for r2 in (("e", f[0]), ("i", f[1])):
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
            fcum += cF * us[u][4] / Pfl
        D = max(D, Dd)
    return D

def stage4(kind, L, w, Tmb, Mw, cap_nic, lay="pieces", wire="native", barrier=False, flush="module", order="fwd",
           rtt=0.0, C_pipe=None):
    F, mem, cpcie = DEV["H100"]
    us = stage_units(kind, L)
    if wire == "fp32":
        us = [(n, p, 4, 4, fp, b, ix) for (n, p, bg, bp, fp, b, ix) in us]
    own = own_of(us, w, lay)
    cap = min(cap_nic, cpcie); lat = 2 * F_COPY / cpcie
    Pfl = sum(u[4] for u in us); c = 8 * Pfl * Tmb / F
    C = Mw * c if C_pipe is None else C_pipe
    A = 4 * Tmb * h if kind == "mid" else 2 * Tmb * h        # boundary bytes per mb per direction (one side at the ends)
    X = exch4(us, own, w, cap, c, rtt=rtt, order=order, barrier=barrier, flush=flush, lat=lat) if w > 1 else 0.0
    ing = [sum(us[u][2] * us[u][1] * own[u][j] * (w - 1) for u in range(len(us))) for j in range(w)]
    eg = [sum(us[u][2] * us[u][1] * (1 - own[u][j]) for u in range(len(us))) for j in range(w)]
    gin = [sum(us[u][3] * us[u][1] * (1 - own[u][j]) for u in range(len(us))) for j in range(w)]
    geg = [sum(us[u][3] * us[u][1] * own[u][j] * (w - 1) for u in range(len(us))) for j in range(w)]
    per_mb = max(max(ing), max(eg)) if w > 1 else 0.0
    Vmax = max(max(Mw * ing[j] + gin[j], Mw * eg[j] + geg[j]) for j in range(w)) + Mw * A if w > 1 else Mw * A
    linkb = Vmax / cap + rtt + c / 2 + (c / 4) / len(us) - C if w > 1 else 0.0
    s1 = (per_mb + A) / cap <= c
    Vdir = sum(Mw * ing[j] + gin[j] for j in range(w)) / w if w > 1 else 0.0   # mean per GPU, per direction
    Adir = Mw * A
    fold = Mw * w * sum(us[u][1] / w * FOLD_B(us[u][2]) for u in range(len(us))) / HBM
    adam = sum(us[u][1] / w * ADAM_B(us[u][3]) for u in range(len(us))) / HBM
    return dict(c=c, C=C, X=X, E=max(X, linkb), s1=s1, util=(per_mb + A) / cap / c, Vdir=Vdir, Adir=Adir,
                tovh=fold + adam + 1e-3, P=sum(u[1] for u in us), us=us, own=own)

def layer_costs():
    return [0.0] + [1.0] * NBLK + [V * h / (12 * h * h)]

def minmax_cut(S):
    cs = layer_costs(); n = len(cs)
    best = {}
    import functools
    @functools.lru_cache(None)
    def f(i, s):   # min over cuts of max stage cost, layers i.., s stages, each stage >= 1 layer
        if s == 1:
            return sum(cs[i:]), (n,)
        res = None
        for j in range(i + 1, n - s + 2):
            m, rest = f(j, s - 1)
            v = max(sum(cs[i:j]), m)
            if res is None or v < res[0] - 1e-12:
                res = (v, (j,) + rest)
        return res
    v, cuts = f(0, S)
    bounds = [0] + list(cuts)
    return [(bounds[s], bounds[s + 1]) for s in range(S)]

def stage_kinds(S):
    out = []
    for (a, b) in minmax_cut(S):
        nb = sum(1 for i in range(a, b) if 1 <= i <= NBLK)
        kind = "first" if a == 0 else ("last" if b == NBLK + 2 else "mid")
        out.append((kind, nb))
    return out

def lane_makespan(S, w, M, Tmb, kinds):
    m = M // w
    cs = []
    for kind, nb in kinds:
        fl = 12 * h * h * nb + (V * h if kind == "last" else 0)
        cs.append(8 * fl * Tmb / F_H100)
    bnd = Tmb * h * 2
    wp = [nb * Tmb * 16 * h * 4 for _, nb in kinds]
    cfg = zb.ZBConfig(S, m, [c / 4 for c in cs], [c / 2 for c in cs], [c / 4 for c in cs], comm=[0.0] * (S - 1),
                      mem_f=[float(bnd)] * S, mem_b=[float(x) for x in wp], mem_w=[-float(bnd + x) for x in wp])
    sched = zb.zero_bubble(cfg, "1p") if S == 1 else zb.zero_bubble(zb._with_pending_budget(cfg, 1), "given")
    _e, ms = zb.simulate(cfg, sched.orders)
    return ms, cs

def plan(S, w, M, cap, lay="pieces", wire="native", barrier=False, flush="module", order="fwd", gpus_per_host=None,
         kpass=None):
    kinds = stage_kinds(S)
    ms, cs = lane_makespan(S, w, M, 4096, kinds)
    m = M // w
    C_pipe = m * max(cs)
    worst_E = 0.0; worst_tag = ""; tovh = 0.0; s1 = True; util = 0.0; host = 0.0
    seen = {}
    for s, (kind, nb) in enumerate(kinds):
        key = (kind, nb)
        if key not in seen:
            seen[key] = stage4(kind, nb, w, 4096, m, cap, lay=lay, wire=wire, barrier=barrier, flush=flush,
                               order=order, C_pipe=C_pipe)
        r = seen[key]
        if r["E"] > worst_E:
            worst_E, worst_tag = r["E"], f"stage {s} ({kind})"
        tovh = max(tovh, r["tovh"]); s1 = s1 and r["s1"]; util = max(util, r["util"])
        host = max(host, r["Vdir"] + r["Adir"])
    tot = ms + worst_E + tovh
    hostt = None
    if gpus_per_host:
        # every GPU's bytes per direction (exchange + boundaries), spread evenly over the hosts (a lower bound for
        # any placement), x k passes per side x 2 directions, against one host's DRAM
        allb = sum(w * (seen[key]["Vdir"] + seen[key]["Adir"]) for key in [(kd, nb) for kd, nb in kinds])
        hosts = S * w / gpus_per_host
        hostt = kpass * 2 * allb / hosts / HOSTBW if hosts >= 1 else kpass * 2 * host / HOSTBW
    return dict(ms=ms, C=C_pipe, E=worst_E, Etag=worst_tag, tovh=tovh, tot=tot, s1=s1, util=util, kinds=kinds,
                host_dir=host, hostt=hostt, cs=cs, stages=seen)

print("== (1) point B per stage: 7B, 128 H100, S 32, w 4, M 128, T_mb 4096; layers [Embed, Block x 32, Head] cut by min-max")
kB = stage_kinds(32)
print("stage kinds (first/last):", kB[0], kB[-1], "; interior", kB[1])
for ln in ("400G", "100G"):
    cap = links[ln]
    for name, kw in (("today: stride, fp32, stage flush, barrier", dict(lay="stride", wire="fp32", flush="stage", barrier=True, order="bwd")),
                     ("step 1 alone: stride, native, barrier", dict(lay="stride", barrier=True)),
                     ("LPT whole tensors, per-unit", dict(lay="lpt")),
                     ("pieces, forward emission", dict(lay="pieces")),
                     ("pieces, backward emission", dict(lay="pieces", order="bwd"))):
        r = plan(32, 4, 128, cap, **kw)
        st = r["stages"]
        row = []
        for key in (("first", 1), ("mid", 1), ("last", 1)):
            x = st[key]
            row.append(f"{key[0]} E {x['E']*1e3:7.1f} ms ({100*x['E']/r['C']:5.1f}%) S1 {x['s1']!s:5s} util {x['util']:.2f}")
        print(f"{ln} {name:42s} | " + " | ".join(row))
r = plan(32, 4, 128, links["400G"])
print(f"B: c per stage first/mid/last {r['cs'][0]*1e3:.2f}/{r['cs'][1]*1e3:.2f}/{r['cs'][-1]*1e3:.2f} ms; C (M/w * slowest) "
      f"{r['C']*1e3:.1f} ms; lane makespan {r['ms']*1e3:.1f} ms = {100*(r['ms']/r['C']-1):.1f}% over C; "
      f"interior-only C {32*r['cs'][1]*1e3:.1f} ms")
for key in (("first", 1), ("mid", 1), ("last", 1)):
    x = r["stages"][key]
    print(f"  {key[0]:5s}: P {x['P']/G9:.3f}B, T_ovh {x['tovh']*1e3:.1f} ms, NIC per GPU per dir {(x['Vdir']+x['Adir'])/G9:.2f} GB "
          f"(exchange {x['Vdir']/G9:.2f} + boundary {x['Adir']/G9:.2f})")

print("== (2) depth vs width, 32 H100, M 32: lane makespan (per-stage costs) + worst stage exposure + T_ovh, ms")
for ln in ("400G", "100G"):
    for SS, ww in ((32, 1), (16, 2), (8, 4), (4, 8)):
        r = plan(SS, ww, 32, links[ln])
        print(f"{ln} S {SS:2d} w {ww}: kinds first {r['kinds'][0]} last {r['kinds'][-1]}; makespan {r['ms']*1e3:6.1f} + E {r['E']*1e3:6.1f}"
              f" [{r['Etag']}] + T_ovh {r['tovh']*1e3:5.1f} = {r['tot']*1e3:6.1f} ms; S1 {r['s1']} (util {r['util']:.2f})")

print("== (2b) 128 H100, M 128: candidates when width is forced (E7 and the I16-refused fallback); host DRAM 600 GB/s")
for ln in ("400G",):
    cands = [("S 32 w 4 pieces", dict(S=32, w=4, M=128, lay="pieces")),
             ("S 32 w 4 zero1 + native (step 1)", dict(S=32, w=4, M=128, lay="stride", barrier=True)),
             ("S 32 w 4 zero1 today (fp32)", dict(S=32, w=4, M=128, lay="stride", wire="fp32", flush="stage", barrier=True, order="bwd")),
             ("S 32 w 1 on 32 GPUs (96 idle)", dict(S=32, w=1, M=128)),
             ("S 16 w 8 pieces", dict(S=16, w=8, M=128, lay="pieces"))]
    for name, kw in cands:
        S_, w_, M_ = kw.pop("S"), kw.pop("w"), kw.pop("M")
        for gph, kp in ((1, 10), (8, 10), (8, 2)):
            r = plan(S_, w_, M_, links[ln], gpus_per_host=gph, kpass=kp, **kw)
            step = max(r["tot"], r["hostt"])
            print(f"{ln} {name:34s} {gph} GPU/host k {kp:2d}: compute+E+T_ovh {r['tot']*1e3:7.1f} ms, host DRAM {r['hostt']*1e3:7.1f} ms"
                  f" -> step >= {step*1e3:7.1f} ms; S1(NIC) {r['s1']}")

print("== (3) host DRAM with boundary bytes scaled by k (interior stage, per GPU per direction per step)")
for tag, S_, w_, M_ in (("A", 8, 4, 32), ("B", 32, 4, 128)):
    r = plan(S_, w_, M_, links["400G"])
    x = r["stages"][("mid", 32 // S_)]
    Cm = (M_ // w_) * x["c"]
    for kp in (2, 4, 10):
        old = kp * 2 * x["Vdir"] / Cm; new = kp * 2 * (x["Vdir"] + x["Adir"]) / Cm
        print(f"point {tag} k {kp:2d}: exchange only {old/1e9:6.1f} GB/s per GPU ({HOSTBW/old:4.1f} GPUs per host) | with boundaries "
              f"{new/1e9:6.1f} GB/s ({HOSTBW/new:4.1f} GPUs per host); 8 GPUs need {8*kp*2*(x['Vdir']+x['Adir'])/HOSTBW*1e3:6.0f} ms per step (C {Cm*1e3:.1f})")
print("sub-block depth at 128 H100, S 128, w 1, M 128 (no exchange): boundary bytes per GPU per direction per step")
c128 = 8 * (P / 128) * 4096 / F_H100; C128 = 128 * c128
for tag, wid in (("h-wide cuts (lower bound)", 1.0), ("per-linear cuts: 4h, h, 5h, h (residual carried)", 2.75)):
    Ad = 128 * 2 * wid * 4096 * h * 2
    for kp in (2, 10):
        print(f"  {tag}: {Ad/G9:5.1f} GB/dir; NIC {Ad/C128*8/1e9:5.0f} Gbps; k {kp:2d}, 8 GPUs per host: {8*kp*2*Ad/C128/1e9:6.0f} GB/s"
              f" ({HOSTBW/(kp*2*Ad/C128):4.1f} GPUs per host)")

print("== (4) device state B/param, pieces vs LPT (whole tensors), state = b_p + 16*max_share + b_g*largest module share")
for tag, kind, L, ww in (("A w4", "mid", 4, 4), ("A w8", "mid", 4, 8), ("B w4 interior", "mid", 1, 4), ("B w4 stage 0", "first", 1, 4)):
    us = stage_units(kind, L); Pt = sum(u[1] for u in us)
    for lay in ("pieces", "lpt"):
        own = own_of(us, ww, lay)
        sh = max(sum(us[u][1] * own[u][j] for u in range(len(us))) for j in range(ww)) / Pt
        bp_avg = sum(us[u][3] * us[u][1] for u in range(len(us))) / Pt
        st = bp_avg + 16 * sh + max(u[2] * u[1] for u in us) / Pt
        print(f"{tag:14s} {lay:6s}: max owned share {sh:.3f} -> state {st:5.2f} B/param ({st*Pt/G9:5.2f} GB at P_s {Pt/G9:.3f}B)")

print("== (5) memory pillar priced by step time: largest dense layer per stage, ZB-1p lane bubble (S-1)/(2*M/w), host bound")
for name, hh, PT, fit in (("405B", 16384, 405e9, {2: 9, 4: 25, 8: 58}), ("1T", 24576, 1e12, {4: 4, 8: 14})):
    p = 12 * hh * hh; S_ = round(PT / p)
    for ww, mw in fit.items():
        beta = (S_ - 1) / (2 * mw)
        hostb = 2 * p * (mw * (ww - 1) / ww + mw * (ww - 1) / ww)
        print(f"{name} (S {S_}) w {ww} M/w {mw}: bubble {100*beta:6.0f}% of C (step ~{1+beta:.1f} C); host bound {hostb/G9:6.0f} GB per peer")
    avail = 80e9 - reserve(80e9)
    base1 = 2 * p + 16 * p + 2 * 4 * hh * hh + 4096 * 16 * hh * 4 + 32 * 4096 * hh * 2 - 2 * p
    print(f"{name} at w 1 (no working copy, no exchange): {base1/G9:5.1f} GB vs {avail/G9:.1f} -> {'fits' if base1 <= avail else 'does not fit'}")
p = 12 * 24576 ** 2
for ww in (4, 8):
    base = 16 * p / ww + 2 * 4 * 24576 ** 2 + 4096 * 16 * 24576 * 4
    per = 2 * p / ww; avail = 80e9 - reserve(80e9)
    mx = max(m_ for m_ in range(1, 600) if base + min(32, m_) * 4096 * 24576 * 2 + per * m_ <= avail)
    print(f"1T w {ww}, per-op regather (no resident b_p*P_s): fits M/w <= {mx}; bubble {100*137/(2*mx):.0f}%;"
          f" host bound {2*p*2*mx*(ww-1)/ww/G9:.0f} GB per peer")

print("== (6) sub-block depth at point B with Embed and Head (unit costs in h^2: qkv 3, proj 1, fc1 4, fc2 4, head V/h)")
def lane_units(costs, S, M, w):
    # exact min-max contiguous cut of unit costs (h^2) into S stages, then the lane solve at per-peer op time
    import functools
    n = len(costs); pre = [0.0]
    for x in costs: pre.append(pre[-1] + x)
    @functools.lru_cache(None)
    def f(i, s):
        if s == 1:
            return pre[n] - pre[i]
        return min(max(pre[j] - pre[i], f(j, s - 1)) for j in range(i + 1, n - s + 2))
    cmax = f(0, S); cuts = []; i = 0
    for s in range(S, 1, -1):
        for j in range(i + 1, n - s + 2):
            if max(pre[j] - pre[i], f(j, s - 1)) <= cmax + 1e-9:
                cuts.append((i, j)); i = j; break
    cuts.append((i, n))
    cs = [8 * (pre[b] - pre[a]) * h * h * 4096 / F_H100 for a, b in cuts]
    m = M // w; bnd = 4096 * h * 2
    cfg = zb.ZBConfig(S, m, [c / 4 for c in cs], [c / 2 for c in cs], [c / 4 for c in cs], comm=[0.0] * (S - 1),
                      mem_f=[float(bnd)] * S, mem_b=[float(bnd)] * S, mem_w=[-2.0 * bnd] * S)
    sched = zb.zero_bubble(zb._with_pending_budget(cfg, 1), "given")
    _e, ms = zb.simulate(cfg, sched.orders)
    return ms, m * max(cs), max(cs) / (sum(cs) / S)
hv = V / h
for tag, costs, S_, w_ in (("per block, S 32, w 4 (point B)", [0.0] + [12.0] * 32 + [hv], 32, 4),
                           ("halves (attention 4, MLP 8), S 64, w 2", [0.0] + [4.0, 8.0] * 32 + [hv], 64, 2),
                           ("per linear, S 128, w 1", [0.0] + [3.0, 1.0, 4.0, 4.0] * 32 + [hv], 128, 1)):
    ms, Cp, imb = lane_units(costs, S_, 128, w_)
    print(f"{tag:40s}: slowest stage {imb:.2f}x the mean; C {Cp*1e3:7.1f} ms; lane makespan {ms*1e3:7.1f} ms")

print("== (7) totals (lane makespan + worst-stage exposure + T_ovh, ms): point B arms and 32-H100 width with zero1")
for ln in ("400G", "100G"):
    for tag, SS, ww, M_ in (("128 H100 point B", 32, 4, 128), ("32 H100", 4, 8, 32), ("32 H100", 8, 4, 32)):
        row = []
        for name, kw in (("today", dict(lay="stride", wire="fp32", flush="stage", barrier=True, order="bwd")),
                         ("zero1+native", dict(lay="stride", barrier=True)), ("LPT", dict(lay="lpt")),
                         ("pieces", dict(lay="pieces"))):
            r = plan(SS, ww, M_, links[ln], **kw)
            row.append(f"{name} {r['tot']*1e3:7.1f} (E {r['E']*1e3:6.1f}, {100*r['E']/r['C']:5.1f}% of C)")
        print(f"{ln} {tag} S {SS} w {ww}: C {plan(SS, ww, M_, links[ln])['C']*1e3:.1f} | " + " | ".join(row))

print("== (8) S2 at the stage that binds (stage 0: window_s = 0): E <= eps*C, eps 3%, C = M/w x slowest stage")
for ln in ("400G", "100G"):
    for SS, ww, M_ in ((16, 2, 32), (8, 4, 32), (4, 8, 32), (32, 4, 128), (16, 8, 128)):
        r = plan(SS, ww, M_, links[ln])
        print(f"{ln} S {SS:2d} w {ww} M {M_:3d}: E {r['E']*1e3:6.1f} ms [{r['Etag']}] vs eps*C {EPS*r['C']*1e3:5.1f} ms -> S2 "
              f"{'holds' if r['E'] <= EPS*r['C'] else 'fails'}; S1 {r['s1']}; total {r['tot']*1e3:7.1f} ms")

print("== (9) E7 prediction, 32 H100, M 32: every candidate's total (ms); pieces admitted only where S1 and S2 hold")
for ln in ("400G", "100G"):
    best = None
    for SS, ww in ((32, 1), (16, 2), (8, 4), (4, 8)):
        arms = [("w 1", {})] if ww == 1 else [("zero1+native", dict(lay="stride", barrier=True)), ("pieces", dict(lay="pieces"))]
        for name, kw in arms:
            r = plan(SS, ww, 32, links[ln], **kw)
            adm = True if name != "pieces" else (r["s1"] and r["E"] <= EPS * r["C"])
            print(f"{ln} S {SS:2d} w {ww} {name:12s}: {r['tot']*1e3:7.1f} ms {'admitted' if adm else 'refused (S1/S2)'}")
            if adm and (best is None or r["tot"] < best[0]):
                best = (r["tot"], SS, ww, name)
    print(f"{ln}: planner choice S {best[1]} w {best[2]} {best[3]} at {best[0]*1e3:.1f} ms")
```
