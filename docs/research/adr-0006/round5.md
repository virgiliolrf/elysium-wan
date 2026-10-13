# ADR-0006 round-5 corrections

Research note behind `docs/design/adr/ADR-0006-exact-fsdp-placement.md`, round 5. It decides nothing and changes no
mechanism and no predicate: it corrects the round-4 accounting (`round4.md`) where the round-5 review found the
host-DRAM leg of S1 averaged or skipped, the S1 window defined two ways, the host-memory bound missing the gathers and
the per-host sum, and E4 unscaled. It supersedes `round4.md` §3 (E7 choices), §4 (fallback table) and §5 (GPUs per
host) where they differ. Every number is DERIVED by the script in §8 on this machine's CPU (no GPU, no network
measured); inputs are HYPOTHESIS as in rounds 3 and 4, plus a 600 GB/s host DRAM and stage-contiguous host packing
(plan GPU g on host `g // gpus_per_host`, a stage's peers adjacent).

## 1. One window for S1: `c_s = 8·P^F_s·T_mb/F` (review findings 2 and 7)

FACT: the ADR's model and SCHEDULER-I16's S2 use `c_s = 8·P^F_s·T_mb/F`, `P^F_s` the stage's GEMM parameters; the
round-4 script uses it too (`stage4`: `Pfl = sum(u[4] …); c = 8 * Pfl * Tmb / F`, tok and pos `flop_params` 0).
SCHEDULER-I16's S1 bullet and DESIGN §9 wrote `c = 8·P_s·T_mb/F` (every parameter). At point B's stage 0 that is
28.60 ms against 16.49 (×1.73): with it, S1 on stage 0's NIC reads 0.54 instead of 0.94, and S1 on host DRAM at k 10
admits a stage the ADR's model refuses. Correction (text): S1 uses the stage's own `c_s` as defined for S2, and with
several stages on one link it is a sum of rates, `Σ_{(s,g) on ℓ} k_ℓ·(b_g·P_s·share_{s,ℓ} + A_{s,ℓ})/c_s ≤ cap_ℓ`,
which is what "S1 uses each stage's own c" (ADR, round 4) meant; `b_g·P_s` is `Σ_u b_g,u·P_u` (the Embed's tables at
4 bytes). Interior and last stages are unchanged (their `P_s = P^F_s`).

## 2. S1 on host DRAM per stage (review findings 1, 5 and 8)

Per GPU of a stage, as I16 writes it (`k·2·(b_g·P_s·(w−1)/w + A_dir)/c_s`), against a 600 GB/s host: GB/s and the
GPUs of that stage one host admits.

| stage | `c_s` | k 2 | k 4 | k 6 | k 10 | 1-GPU host admits |
|---|---|---|---|---|---|---|
| B stage 0 (Embed + Block 0) | 16.49 ms | 189.0 (3.18) | 377.9 (1.59) | 566.9 (1.06) | 944.8 (0.64) | k ≤ 6.35 |
| B interior | 16.49 | 89.5 (6.70) | 179.0 (3.35) | 268.6 (2.23) | 447.6 (1.34) | k ≤ 13.4 |
| B last (Block 31 + Head) | 27.23 | 78.2 (7.68) | 156.3 (3.84) | 234.5 (2.56) | 390.9 (1.54) | k ≤ 15.4 |
| (16, 2) stage 0 (Embed + 2 blocks) | 32.99 | 88.8 (6.76) | 177.5 (3.38) | 266.3 (2.25) | 443.8 (1.35) | k ≤ 13.5 |
| (16, 2) interior | 32.99 | 57.0 (10.5) | 113.9 (5.27) | 170.9 (3.51) | 284.8 (2.11) | k ≤ 21.1 |
| (16, 2) last | 43.72 | 51.9 (11.6) | 103.8 (5.78) | 155.7 (3.85) | 259.5 (2.31) | k ≤ 23.1 |

Round 4's "≤ 1.3 GPUs per host at k 10, ≤ 6.5 at k 2" was the interior stage (with the per-step gathers amortized:
459.0 GB/s); stage 0 binds at 0.64 and 3.18. Round 4 §4's host-DRAM time spread every GPU's bytes over all hosts and
its admission flag checked only `min(NIC, PCIe)`, so it missed that **at today's k = 10 S1 refuses stage 0 at point B
on every host, a 1-GPU host included** (944.8 > 600 GB/s; 15.6 GB per microbatch against `cap·c_s` = 9.9 GB, 1.57×).

## 3. E7 re-derived per host (review findings 1, 5 and 8)

Each candidate: lane makespan + worst-stage exposure + `T_ovh` (round 4), and the busiest host's DRAM time
(`k·(in + out bytes per step)` summed over the GPUs on that host, per peer, gathers and boundaries included; a bound
used as `step = max(…)`, HYPOTHESIS: the as-built simulation shares host DRAM fairly instead). Pieces are admitted only
where S1 holds on every NIC and on every host and S2 holds (SCHEDULER-I16 as written). ms:

| fleet, link | hosts, k | pieces | `zero1` + native | S 32, w 1 | planner choice |
|---|---|---|---|---|---|
| 128 H100, 400 Gbps (S 32, w 4) | 1 GPU, k 10 | 1124.5, refused (stage 0 host DRAM 945 GB/s) | **1721.3** | 3887.8 | `zero1` + native, 1721.3 |
| | 1 GPU, k ≤ 6 | **1124.5** | 1721.3 | 3887.8 | pieces, 1124.5 |
| | 8 GPU, k 10 | refused (stages 0–1: 5569 GB/s) | 5038.6 (host-bound) | **3887.8** | S 32, w 1 on 32 GPUs |
| | 8 GPU, k 2 | refused (stages 0–1: 1114 GB/s) | **1721.3** | 3887.8 | `zero1` + native, 1721.3 |
| 32 H100, 400 Gbps | 1 GPU, k 10 | **(16, 2) 966.8** | (16, 2) 994.4 | 1102.2 | (16, 2) pieces |
| | 2 GPU, k 10 | (16, 2) refused (two stage-0 peers: 888 GB/s) | **(16, 2) 994.4** | 1102.2 | (16, 2) `zero1` + native |
| | 4 or 8 GPU, k 10 | refused | (16, 2) 1355.2 / 2411.0 | **1102.2** | S 32, w 1 |
| | 8 GPU, k 2 | **(16, 2) 966.8** (host 0: 520 GB/s) | 994.4 | 1102.2 | (16, 2) pieces |
| 32 H100, 100 Gbps | 1 or 8 GPU, k 10 | refused (S1 NIC, S2) | ≥ 2021.2 | **1102.2** | S 32, w 1 |

`zero1` today (fp32) at 128 H100: 2163.3 ms on 1-GPU hosts at k 10 (host 1897.4), 7696.1 on 8-GPU hosts at k 10. S 16,
w 8 pieces is refused everywhere (S2: 34.4 > 21.0 ms). Consequences: at today's k = 10 the forced-width gain of pieces
(1721.3 → 1124.5 ms) needs k ≤ 6 (the transport ADR, R10) or a host-packing change (Ideas); the gain available at
k 10 at point B is step 1 alone (`wire: native`), 2163.3 → 1721.3 ms on 1-GPU hosts. At 32 H100 pieces win at k 10
only on 1-GPU hosts (2.8% over `zero1` + native, 12.3% over depth); on 8-GPU hosts only at k 2.

## 4. Host memory: the gathers, and the sum per host (review findings 3 and 6)

**Bound.** Round 3's proof covers gradient pieces only. Gathers sit in pinned host buffers too: an owner's step-t
gather of unit u to d can wait in its send queue while it produces step-(t+1) gradient pieces of other units
(need-first orders by unit first), and a received gather waits on the host until its H2D copy. Per unit at most one
step of each is outstanding: owner o's step-(t+1) gather of u to d needs o's step-(t+1) fold of u, which needs d's
step-(t+1) piece of u, which follows d's F(t+1) of u, which consumed (after its H2D) o's step-t gather of u to d. So
outbound gathers ≤ `Σ_u b_p,u·P_u·(w−1)/w` and inbound ≤ the same, and the corrected one-step bound for peer k is

`host_k ≤ Σ_u b_g,u·P_u·[M_k·(w−1)/w + (M−M_k)/w] + 2·Σ_u b_p,u·P_u·(w−1)/w`  (plus pinned chunks in flight)

| point (M_k) | gradients | gathers | bound per peer |
|---|---|---|---|
| A interior (S 8, w 4, M 32, `M_k` 8) | 19.33 GB | 2.42 | 21.74 GB |
| A interior, `M_k` 12 of 32 | 22.55 | 2.42 | 24.96 (+15%) |
| B interior (S 32, w 4, M 128, `M_k` 32) | 19.33 | 0.60 | 19.93 |
| B stage 0 (fp32 tables at 4 B) | 47.71 | 1.49 | 49.21 |
| B last (Head) | 31.91 | 1.00 | 32.91 |

**Per host.** I16 charged each peer against the host's memory alone. Co-located peers hold their bounds at once: an
8-GPU host at B with stages 0 and 1 needs 276.5 GB; eight stage-0-sized peers 393.6 GB. Correction: MEM-HOST sums
the bound over the peers placed on a host. The own-piece device hold at B is 16 B/param for bf16 units and 32 for the
fp32 tables (`b_g·M_k/w`), not 16 throughout.

## 5. The Embed as one unit (review finding 9, minor)

SCHEDULER-I17 specifies the Embed's tables as one fp32 unit; round 4 modelled tok and pos separately. Pieces at B
stage 0, both emission orders: X = 24.53 ms either way (`pos` is 11.4% of the tables' bytes and both are released at
W(mb) start), so S2's margin (24.5 ≤ 26.1 ms) is unchanged. Step 1 alone and today's `zero1` keep tok and pos as
separate parameters on different stride owners (`stage.py:154`), so round 4's 621.3 and 1075.7 ms stand; merged they
would be 749.0 and LPT 302.7 (larger margins for pieces). Refuted for the decision.

## 6. E4 at scaled B (review finding 4)

The exposure model is invariant when every ratio below is kept, checked by running it at both scales (§8, output (5)).
The two share rows are computed from `(V+seq)·h`, `12h²` and the block's `13h` biases and norms; the `O(h)` terms move
them by ≤ 0.2% at h′, inside the ±0.5% the PLAN checks (added after the round-5 review, finding 4):

| ratio kept | full B | scaled B′ (CPU) |
|---|---|---|
| `V/h`, `seq/h`, `T_mb/h` (table : block bytes, boundary : exchange bytes) | 32000/4096, 1, 1 | h′ 256, V′ 2000, seq′ 256, `T_mb′` 256 |
| Embed share of stage 0's params (tables `(V+seq)·h` over `P_s`; `tok` alone) | 0.4234; 0.3753 | 0.4224; 0.3745 |
| fp32 share of stage 0's gradient bytes per microbatch (tables at 4 B against the block's bf16 GEMMs at 2 B) | 0.5948 | 0.5929 |
| `M/w`, w, U, emission order, need-first queues | 32, 4, 6 units | same |
| NIC utilization `Σ b_g·P/(cap·c_s)` (stage 0), `c_s` from GEMM params only | 0.94 at 400 Gbps | 0.94 at cap′ = cap·(F′/F)·(T_mb/T_mb′) |
| fold and AdamW time over `c_s` | HBM 3.35 TB/s | HBM′ = HBM·(F′/F)·(T_mb/T_mb′) |
| chunk and hop latency over `c_s` | 16 MiB, `2·f/PCIe` | f′ = f·(h′/h)², PCIe′ scaled as cap′ |
| RTT over `c_W` | 0 and > `c_W/4` | same |
| pipeline period over `c_s` | C_B = 32·c_last | the stage-0 arm pads each step to C_B′ = 32·c_last′ |

At F′ = 50 GFLOP/s (the PLAN replaces it with the profiled deterministic rate, and cap′ and HBM′ with the values that
keep the ratios; where the CPU's fold rate cannot be paced, the PLAN re-runs this model at the measured HBM′ and pins
that output): cap′ 0.8 Gbps, HBM′ 6.70 GB/s, f′ 64 KiB, `c_s′` 32.21 ms, C_B′ 1701.9 ms. Pinned predictions:

| arm | % of C (both scales) | ms at B′ | F6 refutes pieces if measured margin below |
|---|---|---|---|
| stage 0, pieces / step 1 alone / LPT | 2.82 / 71.31 / 20.09 | 47.9 / 1213.6 / 341.9 | step 1: 582.8 ms; LPT: 147.0 ms |
| interior w 4 (L 4, M/w 8), pieces / step 1 alone | 3.66 / 16.00 | 37.7 / 165.0 of C′ 1030.8 | 63.6 ms |
| interior w 8 (L 4, M/w 4), pieces / LPT | 10.03 / 25.10 | 51.7 / 129.4 of C′ 515.4 | 38.9 ms |

The interior w-4 figure is 3.66% in the round-4 model (`exch4`), against 3.6% (18.8 ms) from round 3's `exch3`; F6
uses this one. A run whose scaled configuration differs from the PLAN's is not evidence either way.

## 7. Ideas ledger, round 4 (moved verbatim from the ADR in round 5 to keep it within 400 lines)

One figure is superseded by §3: the "Lower k" row's "on 1-GPU hosts pieces still 1124.5 vs 1721.3 for `zero1`"
holds only at k ≤ 6; at k 10 S1 refuses stage 0 on a 1-GPU host and the planner takes `zero1` + native (1721.3 ms).

| Idea (source) | Verdict | Reason and number |
|---|---|---|
| Step 1 + LPT with the barrier kept, to isolate the barrier's worth (round 4) | FOLLOW-UP (E4 diagnostic) | Bounded below by LPT without the barrier: 175.1 ms at B stage 0, 6.7× ε·C, so it cannot replace RUNTIME-I13 there |
| Lower k (memoryview/sendmsg; MSG_ZEROCOPY + recv_into, io_uring zcrx, devmem TX) as an alternative to the layout (round 4, two lenses) | FOLLOW-UP (transport ADR, question 5) | Complementary: at B on 8-GPU hosts host time 3388 → 1355 (k 4) → 678 ms (k 2); on 1-GPU hosts pieces still 1124.5 vs 1721.3 for `zero1` |
| Thread-partition-free optimizer kernel or pinned `num_threads` (round 4) | FOLLOW-UP (E6 tests RUNTIME-I11 first) | AdamW 0.5–1.8 ms per step; needed only if E6 finds a thread-dependent path |
| RaggedShard (veScale-FSDP) as prior art and E4 arm (round 4) | CITED; baseline only | Its reduce-scatter order varies with w; reported 5–66% gains are not byte-exact; weighted pieces stay deferred |
| Several TCP flows per peer pair on high-BDP paths (PCCL; round 4) | FOLLOW-UP (transport; R8) | Raises measured `cap_ℓ` on lossy paths; the 10 Gbps uplink refusal (`T_mb ≥ S·40k`) is capacity, not flows |
| Step-first send key; two-step simulation (round 4, two lenses) | FOLLOW-UP (R15) | Changes the RUNTIME-I13 key; inversion unquantified by the one-step model |
| Bounded outstanding D2H per W (round 4) | FOLLOW-UP | Makes the device transient `b_g·P_mod + k·16 MiB` by construction; E3 measures today's bound |
| Host DRAM share written out in SCHEDULER-I16 (round 4) | APPLIED as text | Restates this ADR's `2·(w−1)/w`; no mechanism; rewritten with finding 5's S1 |
| Sparse touched-rows `tok` exchange with an explicit `+0` (round 4) | FOLLOW-UP (new mechanism) | ≤ 12.8% of rows at V 32k, `T_mb` 4096: 524 → ≤ 67 MB per mb; stage 0 24.5 → ≈ 7.8 ms (HYPOTHESIS) |
| Kernel TCP incast arm (dummynet) beside the shared pacers (round 4) | FOLLOW-UP (E4 control) | F3 compares a model with a shim of the same fair-sharing; decides 18.8 vs 33.9 ms orderings |
| Contiguous ownership priced against the memory pillar (round 4) | FOLLOW-UP; does not rescue it | 1T w 4 at M/w 64: host bound 1.39 TB per peer, bubble 1.07 C |

## 8. Script

`python3 -I r5.py r4.py r3.py r2.py src/wantrain/scheduler/zero_bubble.py`, with `r4.py` extracted from `round4.md` §9
(sha256 `d4838233…dd0c0`) and `r3.py`, `r2.py`, `zero_bubble.py` as in round 4 (hashes reproduced on this tree:
`acf1cb86…995ed6`, `f43ab4f6…b701eb`, `22988afc…751ac8d`; round 4's output reproduced, `7ea94c7f…ecdbe`). Exit 0,
4.5 s. sha256 of the script as run: `cc425fdef36f8ce7d9a683d1e2c9ef83cb057ed54775f1b028df3d203cce37d8`; of its output: `5ac4f6096a2929dd6df3aa860de4894dbd3daeccfa1aaeadf3d84707320250b4`.

```python
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
```
