# ADR-0009 round 4: the tied relay's cost terms (round2.md §3 corrected), evaluated. Pure arithmetic, no torch.
# t_s now includes the tied head GEMM (2·V·h FLOP per token forward, 6·V·h F+B) on a balanced partition.
def terms(name, V, h, nonemb, T_mb, S, M, L, Bbps, frac_T, frac_R, flops=4e14, bw_mem=2e12):
    B = Bbps / 8
    adamw_ms = V * h * 28 / bw_mem * 1e3     # AdamW of the table: 16 B read + 12 B written per element
    P_eq = nonemb + V * h                     # head GEMM counted once (the embedding lookup has no FLOPs)
    t_s = 6 * P_eq * T_mb / S / flops * 1e3   # ms, balanced partition
    t_F, t_B = t_s / 3, 2 * t_s / 3
    a = T_mb * h * 4 / B * 1e3                # ms per hop, fp32 cut
    e = frac_T * a                            # |T_i|·h·4/B
    r = frac_R * a                            # |R_k|·h·4/B, k >= 1
    thr = M * max(0.0, (a + e) - max(t_s, a))
    warm = sum(max(0.0, r - t_F) for _ in range(1, S))
    rt = 2 * (S - 1) * L + 2 * e + adamw_ms
    scan = M * V * h * 4 / bw_mem * 1e3
    # backward link (new): rows R_1..R_{M-1} sent after R_0, FIFO before BWD_0; exposed only past the idle window
    W_idle = 2 * (S - 1) * L + S * t_F + (S - 1) * a + t_B
    back = max(0.0, (M - 1) * r - W_idle)
    base = (M + S - 1) * t_s
    tot = thr + warm + rt + scan + back
    print(f"{name}: t_s {t_s:.1f} t_F {t_F:.1f} a/B {a:.1f} | throughput {thr:.0f} warmup {warm:.0f} "
          f"round-trip {rt:.0f} scan {scan:.0f} back-link {back:.0f} (burst {(M-1)*r:.0f} vs idle {W_idle:.0f}) "
          f"| dT_tied {tot:.0f} ms, base step {base:.0f} ms")

G2 = dict(V=256000, h=2304, nonemb=2.0e9, T_mb=4096)
terms("Gemma2-2B worst  S4 M8 L20 10G", **G2, S=4, M=8, L=20, Bbps=10e9, frac_T=1.0, frac_R=1.0)
terms("Gemma2-2B text   S4 M8 L20 10G", **G2, S=4, M=8, L=20, Bbps=10e9, frac_T=0.45, frac_R=0.15)
terms("Gemma2-2B worst  S2 M8 L0  10G", **G2, S=2, M=8, L=0, Bbps=10e9, frac_T=1.0, frac_R=1.0)
terms("Gemma2-2B worst  S4 M16 L0 10G", **G2, S=4, M=16, L=0, Bbps=10e9, frac_T=1.0, frac_R=1.0)
G3 = dict(V=262144, h=5376, nonemb=25.6e9, T_mb=4096)
terms("Gemma3-27B worst S8 M16 L20 10G", **G3, S=8, M=16, L=20, Bbps=10e9, frac_T=1.0, frac_R=1.0)
# V placement: 2a per microbatch per direction
for frac in (1.0, 0.45):
    a = 4096 * 2304 * 4 / 1.25e9 * 1e3; t_s = 6 * (2.0e9 + 256000 * 2304) * 4096 / 4 / 4e14 * 1e3
    print(f"V placement Gemma2-2B S4 M8 (relay case frac {frac}): {8 * max(0, 2 * a - max(t_s, a)):.0f} ms each way")
