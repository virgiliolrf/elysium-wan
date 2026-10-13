"""ADR-0008 derivations from public HF configs (values typed from the published config.json files; HYPOTHESIS until
E5 reads them). F = 400 TFLOP/s effective (ADR-0006's HYPOTHESIS), bf16 gradients and working copy (b_g = b_p = 2)."""
import hashlib, math
M = {  # h, heads, kv, head_dim, E, k, f, shared_f, layers
 "Mixtral-8x7B":      dict(h=4096, nh=32, kv=8,  hd=128, E=8,   k=2, f=14336, fs=0,    L=32),
 "Qwen1.5-MoE-A2.7B": dict(h=2048, nh=16, kv=16, hd=128, E=60,  k=4, f=1408,  fs=5632, L=24),
 "Qwen3-30B-A3B":     dict(h=2048, nh=32, kv=4,  hd=128, E=128, k=8, f=768,   fs=0,    L=48),
}
F, bg, bp = 400e12, 2, 2
for n, c in M.items():
    h, E, k, f, fs = c["h"], c["E"], c["k"], c["f"], c["fs"]
    attn = h * c["nh"] * c["hd"] * 2 + h * c["kv"] * c["hd"] * 2
    exp = 3 * h * f; sh = 3 * h * fs + (h if fs else 0); router = E * h
    tot = attn + router + E * exp + sh; act = attn + router + k * exp + sh
    rho = tot / act
    print(f"{n}: per layer total {tot/1e6:.1f}M active {act/1e6:.1f}M rho {rho:.2f}")
    for gbps in (400, 100, 10):
        cap = gbps * 1e9 / 8
        print(f"   S1 min T_mb at {gbps} Gbps, w 4: {bg*0.75*F/(8*cap)*rho:,.0f} (dense {bg*0.75*F/(8*cap):,.0f})")
    S = 8; Ls = c["L"] / S
    print(f"   S {S} ({Ls:g} layers/stage), w 4: device state b_p+16/w = {(bp+4)*Ls*tot/1e9:.1f} GB (dense of equal active {(bp+4)*Ls*act/1e9:.1f})")
    for T in (64, 256, 4096):
        print(f"   T_mb {T}: E[empty experts per mb per layer] = {E*(1-k/E)**T:.3g}")
    T = 4096
    print(f"   pending W, routed part, per layer per mb at T 4096: {T*k*(2*h+3*f)*2/2**20:.0f} MiB (dense MLP of width f: {T*(2*h+3*f)*2/2**20:.0f} MiB)")
    print(f"   exchange units per MoE layer: per expert 3E = {3*E}; grouped by projection: 3")
    ep = E * exp * bg / (4 * T * k * h * 2)
    print(f"   gradient bytes / EP token bytes at T 4096: {ep:.1f}; at T 16384: {E*exp*bg/(4*16384*k*h*2):.1f}")
    r = 14.8  # H100 bf16 dense 989 / fp32 67 TFLOP/s, HYPOTHESIS
    print(f"   fp32 router GEMM / layer GEMM work at r {r}: {router*r/(attn+k*exp+sh)*100:.2f}% (compute-dtype router {router/(attn+k*exp+sh)*100:.2f}%)")
    print(f"   routing byte moves / forward expert GEMM time (10 B per routed element, 3.35 TB/s): {10*F/(6*f*3.35e12)*100:.1f}%")
    cs = 8 * act * T / F * 1e3
    print(f"   c per layer per mb (recompute, 8 FLOP/param/token) at T 4096: {cs:.2f} ms; per-expert launches F+F'+B+W: {3*E*4} -> {3*E*4*5e-3:.2f} ms at 5 us")
    rtt = 40e-3
    print(f"   EP over a 40 ms WAN: 4 exchanges x RTT per MoE layer per mb = {4*rtt*1e3:.0f} ms vs {cs:.1f} ms compute per layer")
print("sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
