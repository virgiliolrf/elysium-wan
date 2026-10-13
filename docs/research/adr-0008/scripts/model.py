import math
# name: h, f_expert, E, k, layers, total params, active params (public configs)
M = {
 "Mixtral-8x7B": dict(h=4096, f=14336, E=8, k=2, L=32, tot=46.7e9, act=12.9e9),
 "Qwen1.5-MoE-A2.7B": dict(h=2048, f=1408, E=60, k=4, L=24, tot=14.3e9, act=2.7e9),
 "DeepSeek-V3": dict(h=7168, f=2048, E=256, k=8, L=58, tot=671e9, act=37e9),
}
for T in (256, 4096):
  for n, c in M.items():
    unused = c["E"] * (1 - c["k"]/c["E"])**T
    print(f"T={T:5d} {n:20s} expected experts with 0 tokens per microbatch per layer = {unused:.3g}")
print()
for n, c in M.items():
    expert_p = c["E"]*3*c["h"]*c["f"]          # per MoE layer
    active_ep = c["k"]*3*c["h"]*c["f"]
    print(f"{n:20s} params/FLOP inflation of expert part E/k = {c['E']/c['k']:.1f}; whole model tot/act = {c['tot']/c['act']:.1f}")
    for T in (4096, 16384):
        # ADR-0006 pieces: every microbatch ships its expert gradient pieces (bf16 b_g=2): bytes per mb per layer ~ expert_p*2*(w-1)/w
        wbytes = expert_p*2
        # EP: tokens move: dispatch+combine, fwd+bwd => 4 * T*k*h*2 bytes (bf16) * (w-1)/w
        abytes = 4*T*c["k"]*c["h"]*2
        print(f"   T_mb={T:5d}: weight-gradient bytes/mb/layer {wbytes/1e6:8.1f} MB vs EP token bytes {abytes/1e6:7.1f} MB -> ratio {wbytes/abytes:5.1f}")
