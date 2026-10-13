# ADR-0007 round-5 cost model (DERIVED). Run: python3 -I docs/research/adr-0007/cost_r5.py
# Corrects round 3 (cost_r3.py, unchanged) on three review findings; mechanism unchanged:
#  (1) the alternatives' "once per recipe" column: each option is charged its OWN record. Options that only hash a full
#      build pay T_walk + 4P/r_sha (708 s at 70B); only the ledger needs the NaN-filled two-traversal record (811 s).
#      The seed checkpoint is generated once per recipe (as torchtitan does) and shipped per launch; its NVMe-cached
#      variant reads the slices from local disk per launch after one transfer per host.
#  (2) the record's size: states base64 in JSON (5056 B -> 6744 chars), digests as (name, 64 hex) entries; the cap is a
#      function of (n_phase, L, n_tensor), not a constant.
#  (3) the S at which today's init peak 4P exceeds the resume load 12 P_s,max: 3 P_s,max < P.
# Inputs as cost_r3.py: t_u, t_n, r_sha MEASURED (one run for t_u, t_n: E4 re-measures them, 5 runs, at 2^28 elements);
# r_mem 8 GB/s, NVMe 2 GB/s, link 1 Gbps HYPOTHESIS.
tu, tn, sha, rmem, nvme, link = 1.18e-9, 7.32e-9, 2.3e9, 8e9, 2e9, 1e9 / 8
GB = 1e9


def model(h, L, V, kv, dff):          # general path, no bias, RoPE, untied (as cost_r3.py's 70B / 405B rows)
    blk_lin = h * h + 2 * h * kv + h * h + 3 * h * dff
    blk = blk_lin + 2 * h
    P = V * h + L * blk + V * h + h
    lin = L * blk_lin + V * h
    return dict(P=P, L=L, lin=lin, emb=V * h)


def t_walk(m):                        # gpt2 run on the general path
    return m["lin"] * (tu + tn) + m["emb"] * 2 * tn


m70 = model(8192, 80, 32000, 1024, 28672)
m405 = model(16384, 126, 128256, 1024, 53248)

print("A. alternatives at 70B, S 8, k 8, one 256 GB host (seconds; with digests = the slot hashes 4 P_s per launch)")
P = m70["P"]; S = 8; k = 8; Ps = P / S; T = t_walk(m70)
Hs = 4 * Ps / sha                     # per-launch digest check, 15 s
cp = 4 * Ps / rmem + 1.0              # trim copy + skeleton
own_rec = T + 4 * P / sha             # a record that only hashes a full build (no ledger, no NaN fill/scan)
led_rec = own_rec + 4 * P * (1 + 2) / rmem + 1e-3 * 2 * (m70["L"] + 2)   # cost_r3.py's completed ledger record
host_bytes = k * 4 * Ps               # what one host's slots hold at init: 276 GB at S 8
rows = [
    ("do nothing, strongest (file-backed full build per host)", T + 8 * P / nvme + Hs, own_rec, f"{4 * P / GB:.0f} GB disk"),
    ("full-walk replay / per-tensor interception", T + Hs + cp, own_rec, "0"),
    ("seed checkpoint, generated once, shipped per launch", host_bytes / link + Hs, own_rec, f"{host_bytes / GB:.0f} GB over the link"),
    ("seed checkpoint cached on the host's NVMe (one transfer per host first)", host_bytes / nvme + Hs, own_rec + host_bytes / link,
     f"{host_bytes / GB:.0f} GB disk per host"),
    ("ledger replay (chosen)", T / S + Hs + cp, led_rec, "0"),
]
for name, per, once, other in rows:
    print(f"{name} | per launch {per:.0f} | once per recipe {once:.0f} | without digests: per launch {per - Hs:.0f} | {other}")
print(f"ledger record without digests or NaN passes: still the walk, {T:.0f} s")

print("\nB. record size (bytes): states base64 6744 chars + 16 B JSON each; digest entries 64 hex + name <= 96 + 16 B")
STATE, DIG = 6744 + 16, 64 + 96 + 16
for key, m, lora in (("70B gpt2", m70, False), ("70B gpt2 + LoRA all", m70, True),
                     ("405B gpt2", m405, False), ("405B gpt2 + LoRA all", m405, True)):
    n_phase = 3 if lora else 2
    n_tensor = 1 + m["L"] * (8 + (12 if lora else 0)) + 2 + (2 if lora else 0)
    led = (n_phase * (m["L"] + 2) + 1) * STATE
    dig = n_tensor * DIG
    cap = (n_phase * (m["L"] + 2) + 1) * STATE + n_tensor * DIG + 65536
    print(f"{key}: ledger {led / 1e6:.2f} MB + digests {dig / 1e6:.2f} MB = {(led + dig) / 1e6:.2f} MB; "
          f"cap(cfg) {cap / 1e6:.2f} MB; raw-binary ledger {(n_phase * (m['L'] + 2) + 1) * 5056 / 1e6:.2f} MB")

print("\nC. today's init peak 4P against the resume load 12 P_s,max")
for S in (2, 3, 4):
    print(f"S {S} balanced: 4P/(12P/S) = {4 * S / 12:.2f} (> 1 means init binds)")
