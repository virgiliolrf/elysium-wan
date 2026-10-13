# ADR-0007 cost model arithmetic (DERIVED; per-element costs MEASURED by research angle "systems", one run, arm64)
tu, tn = 1.18e-9, 7.32e-9
def model(name, h, L, V, seq, kv, dff, learned_pos, bias):
    q = h*h; kvp = 2*h*kv; o = h*h; mlp = (3 if dff != 4*h else 2)*h*dff
    lin_block = q + kvp + o + mlp + (bias and (h + 2*kv + h + dff + h) or 0)
    blk = lin_block + 2*h*(2 if bias else 1)
    emb = V*h + (seq*h if learned_pos else 0)
    head_lin = V*h; head = head_lin + h
    P = emb + L*blk + head
    lin = L*lin_block + head_lin
    return dict(name=name, P=P, blk=blk, emb=emb, head=head, lin=lin, embn=emb, L=L)
ms = [model("7B", 4096, 32, 32000, 4096, 4096, 4*4096, True, True),
      model("70B", 8192, 80, 32000, 4096, 1024, 28672, False, False),
      model("405B", 16384, 126, 128256, 8192, 1024, 53248, False, False)]
for m, S, k in [(ms[0], 8, 8), (ms[1], 8, 8), (ms[2], 16, 8)]:
    P = m["P"]; Ps = P / S; Lmax = max(m["blk"], m["emb"], m["head"])
    today = 4*P; new = 4*(Ps + Lmax)
    Tg = m["lin"]*(tu+tn) + m["embn"]*2*tn          # gpt2 scheme, both phases
    Td = m["lin"]*tu + m["embn"]*tn                  # default scheme
    print(f'{m["name"]}: P={P/1e9:.2f}B Lmax={Lmax/1e6:.0f}M S={S} slot today {today/1e9:.1f} GB -> {new/1e9:.1f} GB; '
          f'host k={k}: {k*today/1e9:.0f} -> {k*new/1e9:.1f} GB; T_init gpt2 {Tg:.0f} s default {Td:.0f} s; per-param gpt2 {Tg/P*1e9:.2f} ns')
for N in (32, 128, 1024):
    for tp in (8.5e-9,):
        steps = tp*N*4e14/(6*2**20)
        print(f"N={N}: T_init/T_step = {steps:.0f} steps (F_eff 4e14, T_tok 2^20, {tp*1e9} ns/param)")
