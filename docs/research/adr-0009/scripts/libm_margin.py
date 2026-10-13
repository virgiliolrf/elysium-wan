# ADR-0009 round 4: do host-libm constants reach the fp32 bits? (no torch, no transformers needed)
# For each constructor-computed Python float that a pinned forward multiplies into a tensor, take every double
# that a libm with error < 1 ULP may return at each libm call (the two doubles bracketing the exact value),
# propagate them through the expression in IEEE double (as CPython does), round each result to fp32 (the scalar's
# cast inside ATen's kernels, HYPOTHESIS from code reading) and report whether all candidates give one fp32 value.
import math, struct, itertools
from decimal import Decimal, getcontext
getcontext().prec = 60

def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]

def bracket(exact):
    d = float(exact)  # correctly rounded
    lo, hi = (d, math.nextafter(d, math.inf)) if Decimal(d) <= exact else (math.nextafter(d, -math.inf), d)
    return {lo, hi}

def ln(x):
    return Decimal(x).ln()

def check(name, cands):
    vals = {f32(c) for c in cands}
    return name, len(vals) == 1

rows = []
# scaling = head_dim ** -0.5 (one libm pow); also gemma2's query_pre_attn_scalar ** -0.5
for hd in range(1, 1025):
    rows.append(check(f"pow {hd}**-0.5", bracket(1 / Decimal(hd).sqrt())))
# yarn: 0.1*mscale*log(f) + 1.0, and the ratio of two of them (deepseek-style mscale, mscale_all_dim)
for f in [2, 4, 8, 16, 32, 40, 64, 128, 1.5, 2.5, 3, 5, 6, 10, 20]:
    for ms in [1, 0.707, 0.5]:
        cands = {0.1 * ms * l + 1.0 for l in bracket(ln(f))}
        rows.append(check(f"yarn mscale f={f} m={ms}", cands))
    num = {0.1 * 1.0 * l + 1.0 for l in bracket(ln(f))}
    den = {0.1 * 0.707 * l + 1.0 for l in bracket(ln(f))}
    rows.append(check(f"yarn ratio f={f}", {a / b for a in num for b in den}))
# longrope: sqrt(1 + log(f)/log(orig)) (sqrt is IEEE-exact; two libm logs)
for orig in [2048, 4096, 8192, 32768]:
    for mx in [32768, 65536, 131072, 262144]:
        f = mx / orig
        if f <= 1:
            continue
        cands = {math.sqrt(1 + a / b) for a in bracket(ln(f)) for b in bracket(ln(orig))}
        rows.append(check(f"longrope f={f} orig={orig}", cands))
bad = [n for n, ok in rows if not ok]
print(f"constants checked: {len(rows)}; fp32-ambiguous under a <1-ULP libm: {len(bad)}")
for n in bad:
    print("  AMBIGUOUS", n)
