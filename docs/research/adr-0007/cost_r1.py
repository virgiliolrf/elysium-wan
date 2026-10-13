# ADR-0007 round-1 cost model (DERIVED). Run: python3 -I docs/research/adr-0007/cost_r1.py
# MEASURED inputs (this host: macOS arm64, torch 2.14.1, 1 thread): t_u, t_n (research, one run); sha256 2.3 GB/s
# (probe_r1.py, hashlib, lowest of three runs). HYPOTHESIS inputs: NVMe 2 GB/s, link 1 Gbps, 8 init cores per host.
tu, tn, sha = 1.18e-9, 7.32e-9, 2.3e9
nvme, link, cores = 2e9, 1e9 / 8, 8
STATE = 5056  # bytes of torch.get_rng_state() (MEASURED)


def model(name, h, L, V, seq, kv, dff, learned_pos, bias):
    q = h * h; kvp = 2 * h * kv; o = h * h; mlp = (3 if dff != 4 * h else 2) * h * dff
    lin_block = q + kvp + o + mlp + (bias and (h + 2 * kv + h + dff + h) or 0)
    blk = lin_block + 2 * h * (2 if bias else 1)
    emb = V * h + (seq * h if learned_pos else 0)
    head = V * h + h
    P = emb + L * blk + head
    lin = L * lin_block + V * h
    big_tensor = max(V * h, h * dff)
    return dict(name=name, P=P, blk=blk, emb=emb, head=head, lin=lin, embn=emb, L=L, big=big_tensor)


ms = {"7B": model("7B", 4096, 32, 32000, 4096, 4096, 4 * 4096, True, True),
      "70B": model("70B", 8192, 80, 32000, 4096, 1024, 28672, False, False),
      "405B": model("405B", 16384, 126, 128256, 8192, 1024, 53248, False, False)}


def t_init(m):  # gpt2 scheme: construction (uniform on Linear, normal on Embedding) + gpt2 normal on both
    return m["lin"] * (tu + tn) + m["embn"] * 2 * tn


for key, S in (("7B", 8), ("70B", 8), ("405B", 16)):
    m = ms[key]; P = m["P"]; Ps = P / S; T = t_init(m)
    Lmax = max(m["blk"], m["emb"], m["head"])
    ledger = (m["L"] + 2) * 2 * STATE
    t_slot = T / S + 4 * Ps / sha
    t_rec = T + (T + 4 * P / sha) / cores
    print(f"{key} S{S}: T_init(full walk) {T:.0f} s; ledger slot T/S {T / S:.0f} s + hash 4P_s {4 * Ps / sha:.0f} s = "
          f"{t_slot:.0f} s; record once per recipe {t_rec:.0f} s ({cores} cores); ledger {ledger / 1e6:.2f} MB; "
          f"working layer 4*Lmax {4 * Lmax / 1e9:.1f} GB")

# Alternatives at one point: 70B, S 8, w 1, k 8 slots on one host (256 GB DRAM), 1 Gbps link, local NVMe.
m = ms["70B"]; P = m["P"]; k = 8; S = 8; Ps = P / S; T = t_init(m); Lmax = max(m["blk"], m["emb"], m["head"])
rows = [
    ("today: full build per slot", k * 4 * P, 0, None, 0),
    ("do nothing, strongest: one file-backed full build per host + host lock, slots read slices",
     None, 4 * P, T + 4 * P / nvme + 4 * P / nvme, 0),
    ("seed checkpoint from a remote generator host", k * 4 * Lmax, 0, (T + (T + 4 * P / sha) / cores) + 4 * P / link, 4 * P),
    ("per-tensor interception, full walk", k * 4 * m["big"], 0, T + 4 * Ps / sha, 0),
    ("replay, full walk per slot (round 0)", k * 4 * Lmax, 0, T + 4 * Ps / sha, 0),
    ("replay with the RNG ledger (round 1)", k * 4 * Lmax, 0, T / S + 4 * Ps / sha, (m["L"] + 2) * 2 * STATE),
    ("recipe v2, per-parameter generator", k * 4 * Lmax, 0, T / S, 0),
]
print("\n70B S8 k8 host: option | host DRAM GB | local disk GB | time to first step s | network bytes")
for name, dram, disk, t, net in rows:
    print(f"{name} | {'page cache (<=256)' if dram is None else f'{dram / 1e9:.1f}'} | {disk / 1e9:.0f} | "
          f"{'cannot start' if t is None else f'{t:.0f}'} | {net / 1e9:.3g} GB")
print(f"record production (once per recipe, 70B): {T + (T + 4 * P / sha) / cores:.0f} s on {cores} cores")
