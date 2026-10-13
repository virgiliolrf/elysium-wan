import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
import json, sys
S=sys.argv[1]
vs=["4.57.6","5.0.0","5.9.0","5.19.0"]
data={v:json.load(open(f"{S}/scan-{v}.json")) for v in vs}
for v in vs:
    d=data[v]; dec={k:x for k,x in d.items() if not x["encdec"]}
    keys=["ccm","swm","chunked","update_causal_mask","layer_types","attn_iface","conv1d","experts3d","kernel_hub","pp_plan","recurrent","softcap","logits_scaling","emb_mult","rotate_half","interleaved_rope"]
    print(v, "causalLM files", len(d), "non-encdec", len(dec), {k:sum(1 for x in dec.values() if x[k]) for k in keys})
    nosig=[k for k,x in dec.items() if not x["layer_sig"]]
    print("  no DecoderLayer class:", len(nosig))
    rets={}
    for k,x in dec.items():
        if x["layer_sig"]: rets[str(x["layer_sig"][2])]=rets.get(str(x["layer_sig"][2]),0)+1
    print("  layer return annotations:", sorted(rets.items(), key=lambda t:-t[1])[:6])
# signature changes between consecutive versions
for a,b in zip(vs,vs[1:]):
    common=[k for k in data[a] if k in data[b] and data[a][k]["layer_sig"] and data[b][k]["layer_sig"]]
    ch=[k for k in common if data[a][k]["layer_sig"]!=data[b][k]["layer_sig"]]
    print(f"{a}->{b}: families in both {len(common)}, decoder-layer forward signature/return changed {len(ch)}", ch[:12])
    new=[k for k in data[b] if k not in data[a]]
    print("   new causal-LM families:", len(new), new[:40])
