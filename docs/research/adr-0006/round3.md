# ADR-0006 round-3 corrections

Research note behind `docs/design/adr/ADR-0006-exact-fsdp-placement.md`, round 3. It decides nothing and changes no
mechanism; it corrects the accounting of the round-2 model (`derivation.md`) where the round-3 review found terms
missing, and it supersedes the numbers of `derivation.md` §4 (memory) and §5 (layout tables) where they differ. Every
number is DERIVED by the script in §7 on this machine's CPU (no GPU, no network measured). Inputs are HYPOTHESIS, as in
`derivation.md`, plus: H100 HBM 3.35 TB/s; host DRAM 600 GB/s for a two-socket host (16 DDR5-4800 channels, nominal).

## 1. The owner's own pieces wait on the device (review findings 1 and 7)

FACT (code, this tree): today a peer's own-shard contribution never leaves the device: `_store_grads` calls
`_shard_add(step, mb, gw[self.peer :: self.w])` (`stage.py:940`), which moves it to the device (`stage.py:969`) and adds
it to an `OrderedAccumulator`, whose `add` keeps it in `self.hold` until every lower index is folded
(`accumulate.py:34`, `:43`). Under IWRR, peer k's microbatches interleave with the others', so an own piece waits for
the remote prefix. Round 2 charged the remote hold (host) and the send backlog (host) but no term for own pieces, and
its PCIe share `(w−1)/w` implies they stay on the device.

Correction (accounting only; the mechanism keeps own pieces where they are produced): own pieces wait on the device.
By the round-2 proof (`derivation.md` §4), no step-(t+1) piece exists before every step-t fold is complete, so a peer
holds at most its own `M_k` pieces of one step: `device_own ≤ b_g·M_k·P_s/w`, with `M_k` the microbatches peer k owns
(`M/w` under equal weights). Per stage param (bf16 gradients):

| point | state (round 2) | + own hold, one-step worst case | today's ZeRO-1: state | + its fp32 hold on the device, one step |
|---|---|---|---|---|
| A, w 4, M/w 8, L_s 4 | 6.17 | 10.17 | 11.00 | 42.00 |
| A, w 8, M/w 4 | 4.17 | 5.17 | 9.50 | 25.00 |
| B, w 4, M/w 32, L_s 1 | 6.67 | 22.67 | 11.00 | 138.00 |

Round 2's "today 14.0" counted a hold of `4·(w−1)/w` (one round); on the same one-step assumption as above, today's
hold is `4·(M−1)/w` on the device, because today folds remote and own contributions there. The like-for-like
comparison is therefore state against state (11.0 → 6.17 at w 4) and hold against hold (fp32, device, `M−1` pieces
→ `b_g`, device, `M_k` pieces; the remote part moves to the host).

The largest dense layer as its own stage (`T_mb` 4096, L_s 1; state, one module, one pending W, boundaries, own hold)
against 70.4 GB (80 GB less `vram_reserve`):

| layer | w 2 | w 4 | w 8 |
|---|---|---|---|
| 405B (h 16384) | fits with M/w ≤ 9 (65.5 GB at 8) | M/w ≤ 25 | M/w ≤ 58 |
| 1T (h 24576) | never (83.8 GB before any hold) | M/w ≤ 4 (85.4 GB at 8) | M/w ≤ 14 (56.4 GB at 8) |

So round 2's "a 1T dense layer fits 80 GB at w ≥ 4" holds only with `M/w ≤ 4` at w 4 or `M/w ≤ 14` at w 8.

**Weighted IWRR (finding 8).** With unequal weights peer k owns `M_k ≠ M/w`. Its host bytes are its send backlog
`b_g·M_k·P_s·(w−1)/w` plus the remote pieces it receives, `b_g·(M−M_k)·P_s/w`:
`host_k ≤ b_g·P_s·[M_k·(w−1)/w + (M−M_k)/w]`. At M 32, w 4: `M_k` 8 gives 12 `b_g·P_s` (the round-2 bound), `M_k` 12
gives 14 (+17%), `M_k` 4 gives 10. Admission charges `max_k` of both the host and the device term.

## 2. Host DRAM is a link (finding 2)

FACT (code): the exchange path copies each byte several times in host memory. Sender: D2H DMA write (1 pass),
`numpy().tobytes()` (`serial.py:65`, read and write: 2), `b"".join(parts)` (`serial.py:143`, 2), the
`hdr + … + payload` concatenation in `send_frame` (`framing.py:117`, 2), the kernel's socket copy (2), the NIC's DMA
read (1): 10 passes. Receiver: NIC DMA (1), `sock.recv` (2), `bytearray.extend` and `bytes(buf)` in `_recv_exact` (4),
`np.frombuffer(...).copy()` (`serial.py:79`, 2), H2D DMA (1): 10, plus 2 when fragments are joined
(`framing.py:144`). A host-staged transport cannot go below 2 per side (one DMA, one NIC DMA).

Per GPU per step the NIC carries `V_dir` bytes each way (point A 10.87 GB, point B 9.97 GB), so the host's DRAM carries
`k·2·V_dir` per GPU, summed over the host's GPUs:

| k passes per side | point A, per GPU | 8 GPUs per host | GPUs per 600 GB/s host | point B, 8 GPUs: host time per step |
|---|---|---|---|---|
| 2 (zero-copy floor) | 82.4 GB/s | 659 GB/s | 7.3 | 532 ms (C 527.8 ms) |
| 4 | 164.8 GB/s | 1318 GB/s | 3.6 | — |
| 10 (today's code) | 412.0 GB/s | 3296 GB/s | 1.5 | 2658 ms |

So the round-2 S1 link set (NIC, PCIe, uplink) omitted the binding link on dense hosts: at point B with 8 GPUs per
host, even the zero-copy floor sits at the host's capacity, and today's copy count needs 5× it. The rate scales with
`1/T_mb`. The host link is a correction to the admission predicate (a link in S1, summed over the GPUs of a host,
with the transport's counted k), not a new mechanism; lowering k or bypassing host memory (zero-copy sockets, device
memory TCP, NVLink inside a host) is a transport change for its own ADR. The round-2 rejection of the host-side fold
(64 GB per step per GPU more) stands and is now consistent: it would add to the link that already binds.

Copies do not run on the compute thread, so they are not `T_ovh`; transport threads that hold the GIL while
concatenating can delay the compute thread's Python dispatch (HYPOTHESIS, unmeasured; real-hardware item F7).

## 3. The fold and AdamW are op time (finding 3)

Each owner folds M microbatches of its `P_s/w` piece: widen `b_g → 4` (read `b_g`, write 4) then `add_` (read 4 + 4,
write 4): 18 B per element per microbatch for a bf16 wire. Per GPU per step that is `M·(P/N)·18 B` with N the device
count, the same at every placement of the same N, because each GPU folds M microbatches of its share of the params. At
7B on 32 H100 (M 32): 34.6 ms per step (6.6% of C) at S 32, w 1 (the narrow stage widens and folds the same bf16 dW),
at S 8, w 4 and at point B. AdamW reads and writes 30 B per owned element: 1.8 ms at A and at S 32, w 1; 0.5 ms at B.

`T_ovh` gains the fold, so the absolute step time rises for every candidate equally; the ranking does not move:

| 32 H100, 400 Gbps, M 32 | bubble | exposure (as built, §4) | `T_ovh` | total over C |
|---|---|---|---|---|
| S 32, w 1 | 48.4 | 0.0 | 7.1 | 55.5% |
| S 16, w 2 | 46.9 | 0.6 | 7.1 | 54.6% |
| S 8, w 4 | 43.8 | 3.7 | 7.1 | 54.6% |
| S 4, w 8 | 37.5 | 9.4 | 7.1 | 54.0% |

The review's "47.4 + 3.6 = 51.0 > 48.4" added the fold to width only; the narrow stage pays the same fold. The
per-unit fold of the last microbatch and the unit's AdamW now delay that unit's gather release in the simulation.

## 4. The exchange as it will be built (finding 4)

Round 2 simulated a non-preemptive need-first queue at every egress and every ingress. A receiver cannot order its
ingress: concurrent TCP flows from w−1 senders share it by congestion control. Today's per-destination send queues are
FIFO (`stage.py:942`, `queue.Queue`). The round-3 model is the one the specs now describe:
- each sender keeps one send queue per destination (as today) and serves it **need-first**: forward unit index, then
  gathers before gradients, then step and microbatch; one ≤ 16 MiB chunk at a time, never preempted inside a chunk
  (the priority by forward need adopted in round 1, written into SPEC-runtime task 5 this round);
- egress, ingress and uplink capacity are shared max-min fairly among the chunks in flight (fluid, coupled, so a
  receiver gets bytes only as fast as the senders send them);
- a unit's gather is released after the owner's last fold and AdamW of that unit (§3).

Exposure E in ms (% of C), 7B, `T_mb` 4096, H100, 400 Gbps; round 2's model | as built (need-first) | FIFO control:

| layout | point A, w 4 | point A, w 8 | point B (S 32, w 4, M 128) |
|---|---|---|---|
| today (stride, fp32, barrier) | 531.7 (100.7) | 433.2 (164.2) | 454.6 (86.1) |
| step 1 alone (barrier kept) | 83.5 \| 84.5 (16.0) \| 84.5 | 110.5 \| 111.5 (42.3) \| 111.5 | 23.7 \| 24.6 (4.7) \| 24.6 |
| step 1 + per-unit updates, stride | 48.6 \| 65.8 (12.5) \| 47.9 | 104.3 \| 104.3 \| 104.3 | — |
| LPT whole tensors, per-unit | 33.8 \| 35.3 (6.7) \| 31.3 | 52.4 \| 66.2 (25.1) \| 51.8 | 17.4 \| 20.9 (4.0) \| 15.7 |
| flat element ranges, per-unit | 41.3 \| 40.5 (7.7) \| 38.2 | 57.3 \| 61.0 (23.1) \| 54.0 | — |
| pieces, backward emission | 18.8 \| 18.8 (3.6) \| 33.9 | 26.8 \| 28.2 (10.7) \| 41.9 | — |
| pieces, forward emission | 19.1 \| 19.3 (3.7) \| 18.4 | 27.1 \| 26.5 (10.0) \| 26.5 | 7.6 \| 7.8 (1.5) \| 6.9 |

At 100 Gbps point B is link-bound and unchanged: today 3373.8, step 1 alone 1513.6, LPT 716.3, pieces 450.6 ms.
Pieces stay ahead of every alternative under the as-built model. Need-first matters for backward emission (18.8 vs
33.9 ms FIFO at A, w 4); with forward emission FIFO is within 1 ms. For concentrated layouts need-first under fair
sharing can be worse than FIFO (stride per-unit 65.8 vs 47.9 ms). F3 compares the measurement with this model.

## 5. Point B, honestly (finding 5)

At point B the as-built margins are: pieces over step 1 alone 3.2% of C (24.6 → 7.8 ms), about ε; pieces over LPT
2.5% of C, below ε. Step 1 alone already takes the exposure from 86.1% to 4.7%. So at point B the time case for the
layout (step 2) is about ε, and F6 at point B tests exactly that; memory is not a reason at 7B (`P_s` 0.2B).

## 6. Sub-block cut points (finding 6)

FACT: the partition unit is the layer list `[Embed, Block × n, Head]` (`arch.py:228-231`), cut by contiguous layer
bounds (`partition.py:186`); a Block holds ln1, qkv, proj, ln2, fc1, fc2 (`gpt.py:255-265`). A cut inside a Block moves
exact activation bytes only, so it is allowed by the product rule, but it is a model and partition change that does
not exist. Priced with the closed form `β = (S−1)·w/(2M)` at M 128: S 32, w 4 48.4% (+1.5% exposed); S 64, w 2 49.2%;
S 128, w 1 49.6% with no exchange. At point B, sub-block depth ties pieces on time (49.6 against 49.9 plus the same
`T_ovh`) and needs no host link. For a 1T layer split per linear, fc1 (`4h²` = 2.42B params) at 20 B/param is 48.3 GB
at w 1. Both pillars of the decision therefore hold relative to block-granular cuts; the alternative is recorded as
open for the human (ADR Risks).

## 7. Script

`r2.py` is the round-2 script exactly as printed in `derivation.md` §9 (sha256 `f43ab4f6…b701eb`, extracted with
`awk 'NR>=268 && NR<=570' derivation.md | sed '1d;$d'`). `python3 -I r3.py r2.py`: exit 0, 1.1 s. sha256 of the
script as run: `acf1cb86b8785b660299581a7d844031b899dd48ae54b58a8858c3ecdf995ed6`; of its output:
`8e336cc865a9263ec25e564fb6a421ac94de14e1bb57d6c563374887ef0258dd`.

```python
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
```
