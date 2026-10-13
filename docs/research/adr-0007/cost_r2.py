# ADR-0007 round-2 cost model (DERIVED). Run: python3 -I docs/research/adr-0007/cost_r2.py
# Same MEASURED inputs as cost_r1.py (macOS arm64, torch 2.14.1, 1 thread): t_u, t_n per fp32 element; sha256 2.3 GB/s.
# HYPOTHESIS inputs: NVMe 2 GB/s; link 1 Gbps; c 8 cores for the record; portable sampler per-sample costs
# (Philox + integer-table ziggurat in a native kernel ~3 ns; Philox + Box-Muller on correctly rounded log/sincos ~20 ns;
# Karney's exact sampler 50-200 ns). None of the portable costs is measured here.
tu, tn, sha = 1.18e-9, 7.32e-9, 2.3e9
nvme, link, cores = 2e9, 1e9 / 8, 8
STATE = 5056


def model(h, L, V, seq, kv, dff, learned_pos, bias):
    q = h * h; kvp = 2 * h * kv; o = h * h; mlp = (3 if dff != 4 * h else 2) * h * dff
    lin_block = q + kvp + o + mlp + (bias and (h + 2 * kv + h + dff + h) or 0)
    blk = lin_block + 2 * h * (2 if bias else 1)
    emb = V * h + (seq * h if learned_pos else 0)
    head = V * h + h
    P = emb + L * blk + head
    lin = L * lin_block + V * h
    return dict(P=P, blk=blk, emb=emb, head=head, lin=lin, embn=emb, L=L, Vh=V * h)


ms = {"7B": (model(4096, 32, 32000, 4096, 4096, 4 * 4096, True, True), 8),
      "70B": (model(8192, 80, 32000, 4096, 1024, 28672, False, False), 8),
      "405B": (model(16384, 126, 128256, 8192, 1024, 53248, False, False), 16)}


def t_walk(m):
    return m["lin"] * (tu + tn) + m["embn"] * 2 * tn


print("A. record production (once per recipe)")
for key, (m, S) in ms.items():
    P = m["P"]; T = t_walk(m); H = 4 * P / sha
    r1 = T + (T + H) / cores
    print(f"{key}: T_walk {T:.0f} s; hash 4P {H:.0f} s; round 1 (pass 1 + parallel pass 2) {r1:.0f} s; "
          f"round 2 (pass 1 hashes at each tensor's last write) {T + H:.0f} s on one thread, "
          f"{max(T, H):.0f} s with hashing on a second thread (HYPOTHESIS: hashlib releases the GIL)")

print("\nB. per launch, per slot (P_s = P/S)")
for key, (m, S) in ms.items():
    P = m["P"]; T = t_walk(m); Ps = P / S
    print(f"{key} S{S}: T_slice {T / S + 4 * Ps / sha:.0f} s (draws {T / S:.0f} + hash {4 * Ps / sha:.0f}); "
          f"hf skips the draws: {4 * Ps / sha:.0f} s + load; tied Embed or Head slot holds the table "
          f"4*V*h = {4 * m['Vh'] / 1e9:.2f} GB beyond one working layer {4 * max(m['blk'], m['emb'], m['head']) / 1e9:.1f} GB")

print("\nC. alternatives at 70B, S 8, k 8, with and without the record's digests")
m, S = ms["70B"]; P = m["P"]; Ps = P / S; T = t_walk(m); H = 4 * P / sha; Hs = 4 * Ps / sha
# Per launch without digests = draws (and file I/O) only; digests add the slot's own hash 4*P_s/r_sha per launch and a
# record of digests once per recipe (one hashed walk, T + H, wherever the bytes are produced sequentially).
rows = [
    ("do nothing, strongest (file-backed full build per host)", T + 8 * P / nvme, 0, Hs, T + H),
    ("full-walk replay (round 0)", T, 0, Hs, T + H),
    ("ledger replay (chosen; the ledger itself needs a record, pass 1 without hashing)", T / S, T, Hs, T + H),
    ("recipe v2, torch per-parameter generator", T / S, 0, Hs, (T + H) / cores),
]
for t_port, lab in ((3e-9, "3 ns"), (20e-9, "20 ns"), (50e-9, "50 ns"), (200e-9, "200 ns")):
    rows.append((f"recipe v2, portable sampler at {lab}/sample", Ps * t_port, 0, Hs, (P * t_port + H) / cores))
print("option | per launch without digests s | once per recipe without digests s | + per launch with digests s | "
      "once per recipe with digests s")
for name, tl, once, dh, once_d in rows:
    print(f"{name} | {tl:.0f} | {once:.0f} | +{dh:.0f} | {once_d:.0f}")
print(f"P(70B) = {P / 1e9:.2f}e9; P_s = {Ps / 1e9:.2f}e9")
