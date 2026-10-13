# ADR-0007 round-3 cost model (DERIVED). Run: python3 -I docs/research/adr-0007/cost_r3.py
# Corrects round 2 (cost_r2.py) on four review findings; mechanism unchanged:
#  (1) the scheme a config RUNS, not the one it declares: gpt2_init runs only on the general path (gpt.py:368-376,
#      433-442; arch.py:232-233). The 7B row (bias, learned positions, d_ff 4h, no GQA, untied) is builtin-shaped,
#      so it draws torch_default whatever init_scheme says.
#  (2) host memory with the home on the host (gpu false) and with a device home, reported separately.
#  (3) the resume host peak: _resume torch.loads the whole slot payload to the CPU (stage.py:715-718, 747):
#      fp32 weights + AdamW m and v = 12*P_s, plus the slice itself when the home is the host.
#  (4) the record walk's NaN fill and scans, the slot's trim copy and the skeleton, priced by HYPOTHESIS rates.
# MEASURED inputs (macOS arm64, torch 2.14.1, 1 thread; derivation.md section 1): t_u, t_n per fp32 element; sha256.
# HYPOTHESIS inputs (to be MEASURED by E4 on the same host): r_mem 5-10 GB/s per pass over fp32 bytes (fill, isnan
# scan with its bool intermediate, copy); c_layer 1 ms per (phase, layer) of the record walk; T_skel 1 s per slot.
tu, tn, sha = 1.18e-9, 7.32e-9, 2.3e9
nvme = 2e9
GB = 1e9


def model(h, L, V, seq, kv, dff, learned_pos, bias):
    q = h * h; kvp = 2 * h * kv; o = h * h; mlp = (3 if dff != 4 * h else 2) * h * dff
    lin_block = q + kvp + o + mlp + (bias and (h + 2 * kv + h + dff + h) or 0)
    blk = lin_block + 2 * h * (2 if bias else 1)
    emb = V * h + (seq * h if learned_pos else 0)
    head = V * h + h
    P = emb + L * blk + head
    lin = L * lin_block + V * h
    general = (not bias) or (not learned_pos) or dff != 4 * h or kv != h   # _uses_general_arch, untied
    return dict(P=P, blk=blk, emb=emb, head=head, lin=lin, L=L, Vh=V * h, general=general)


def scheme_run(m, declared):
    return "gpt2" if (declared == "gpt2" and m["general"]) else "torch_default"


def t_walk(m, scheme):
    if scheme == "gpt2":
        return m["lin"] * (tu + tn) + m["emb"] * 2 * tn
    return m["lin"] * tu + m["emb"] * tn


def t_rec(m, scheme, r_mem):
    n_phase = 2 if scheme == "gpt2" else 1
    n_fill, n_scan = n_phase - 1, n_phase
    P = m["P"]
    extra = 4 * P * (n_fill + n_scan) / r_mem + 1e-3 * n_phase * (m["L"] + 2)
    return t_walk(m, scheme) + 4 * P / sha + extra, extra


def t_slice(m, scheme, S, r_mem):
    Ps = m["P"] / S
    return t_walk(m, scheme) / S + 4 * Ps / sha + 4 * Ps / r_mem + 1.0


ms = {"7B": (model(4096, 32, 32000, 4096, 4096, 4 * 4096, True, True), 8),
      "70B": (model(8192, 80, 32000, 4096, 1024, 28672, False, False), 8),
      "405B": (model(16384, 126, 128256, 8192, 1024, 53248, False, False), 16),
      "E4 250M": (model(1024, 16, 32000, 1024, 1024, 4 * 1024, False, False), 4)}

print("A. scheme run and time (declared gpt2; r_mem 8 GB/s, range 5-10)")
for key, (m, S) in list(ms.items()) + [("70B S32", (ms["70B"][0], 32))]:
    sc = scheme_run(m, "gpt2")
    T = t_walk(m, sc)
    R8, x8 = t_rec(m, sc, 8 * GB); R5, _ = t_rec(m, sc, 5 * GB); R10, _ = t_rec(m, sc, 10 * GB)
    old = T + 4 * m["P"] / sha
    sl = t_slice(m, sc, S, 8 * GB)
    full = T + 4 * (m["P"] / S) / sha
    print(f"{key}: P {m['P'] / GB:.3f}e9 path {'general' if m['general'] else 'builtin'} runs {sc}; T_walk {T:.1f} s; "
          f"T_rec round-2 model {old:.1f} s -> completed {R8:.1f} s (5..10 GB/s: {R5:.1f}..{R10:.1f}); omitted terms "
          f"{x8:.2f} s = {100 * x8 / old:.0f}% of the round-2 model; T_slice {sl:.1f} s; first launch T_rec+T_slice "
          f"{R8 + sl:.0f} s vs full walk {full:.0f} s (+{R8 + sl - full:.0f} s)")

print("\nB. host memory per slot and per k-slot host (GB, B_0 excluded)")
for key, S, k, tied in (("7B", 8, 8, False), ("70B", 8, 8, False), ("70B", 32, 8, False), ("405B", 16, 8, False)):
    m = ms[key][0]
    P = m["P"]; Ps = P / S; Lm = max(m["blk"], m["emb"], m["head"]); held = m["Vh"] if tied else 0
    build_dev = 4 * (Lm + held); build_host = build_dev + 4 * Ps
    res_dev = 12 * Ps; res_host = 16 * Ps; steady_host = 16 * Ps
    today = max(4 * P, res_dev)
    dev = max(build_dev, res_dev)
    hosth = max(build_host, res_host, steady_host)
    print(f"{key} S{S} k{k}: today slot {today / GB:.1f} host {k * today / GB:.0f} | device home: build {build_dev / GB:.1f}"
          f" resume {res_dev / GB:.1f} -> host {k * build_dev / GB:.1f} (build) {k * dev / GB:.1f} (max with resume); device"
          f" steady 16P_s {16 * Ps / GB:.1f} | home on host (gpu false): build {build_host / GB:.1f} resume {res_host / GB:.1f}"
          f" steady {steady_host / GB:.1f} -> host {k * build_host / GB:.1f} (build) {k * hosth / GB:.1f} (max)")

print("\nC. alternatives at 70B, S 8, k 8 (seconds, with digests; r_mem 8 GB/s; per launch includes the trim copy and"
      " T_skel wherever a slot cuts a slice; once per recipe = the completed record walk)")
m, S = ms["70B"]; P = m["P"]; Ps = P / S; T = t_walk(m, "gpt2"); Hs = 4 * Ps / sha; R, _ = t_rec(m, "gpt2", 8 * GB)
cp = 4 * Ps / (8 * GB) + 1.0
rows = [("do nothing, strongest (file-backed full build per host; the read is the copy)", T + 8 * P / nvme + Hs, R),
        ("full-walk replay / per-tensor interception", T + Hs + cp, R),
        ("ledger replay (chosen) / jump-ahead", t_slice(m, "gpt2", S, 8 * GB), R),
        ("recipe v2, torch per-parameter generator (record on 8 cores)", T / S + Hs + cp, R / 8)]
for t_port, lab in ((3e-9, "3 ns"), (20e-9, "20 ns"), (50e-9, "50 ns"), (200e-9, "200 ns")):
    rows.append((f"recipe v2, portable sampler at {lab}/sample", Ps * t_port + Hs + cp, (P * t_port + (R - T)) / 8))
for name, tl, once in rows:
    print(f"{name} | per launch {tl:.0f} | once per recipe {once:.0f}")
