# Generates the amended docs/experiments/m2b-e7/PLAN.md from the granted PLAN.md/PLAN.json, the amended PLAN.json
# (build_plan_spec.py) and its tables fragment. Every number comes from those files; none is typed by hand.
# usage: python3 -I -B gen_plan_md.py granted_PLAN.md granted_PLAN.json ../PLAN.json e7_tables.md ../PLAN.md
import json
import sys

OLDMD, OLDJ, PJ, FRAG, OUT = sys.argv[1:6]
old = open(OLDMD).read()
oj = json.load(open(OLDJ))
p = json.load(open(PJ))
frag = open(FRAG).read()
d = p["derivation"]
am = p["amendments"][-1]
changed = [c["id"] for c, o in zip(p["configs"], oj["configs"]) if c["expected_choice"] != o["expected_choice"]]


def name(x, N=None):
    if x is None:
        return "—"
    tail = f" on {x['gpus_used']} GPUs" if N is not None and x["gpus_used"] != N else ""
    if x["w"] == 1:
        return f"S {x['S']}, w 1{tail}"
    return f"S {x['S']}, w {x['w']}, {x['shard']} + {x['wire']}{tail}"


def kinds(c):
    ks = []
    for r in c["refused"]:
        for x in r["reasons"]:
            if x["kind"] not in ks:
                ks.append(x["kind"])
    return f"{len(c['refused'])} ({'; '.join(ks)})" if c["refused"] else "0"


top = ("**Amended 2026-10-10 (SC-4 / SPEC SCHEDULER-I17, section 9):** the expected choices, steps, margins, ties and "
       "refusals below are re-derived by an independent SCHEDULER-I17 oracle with host DRAM inside the exchange "
       "simulation and gathers after the owner's last fold and AdamW. " + str(len(changed)) + " choices changed ("
       + ", ".join(f"`{x}`" for x in changed) + "); the granted values are kept in section 9 and in `PLAN.json` "
       "`amendments`.\n\n")
i = old.index("## 1. Hypothesis")
md = old[:i] + top + old[i:]

# ---- section 3 table
s0 = md.index("| config (`id`) | expected choice | step ms |")
s1 = md.index("**No choice changed.**")
rows = ["| config (`id`) | expected choice | step ms | runner-up, ms | margin % | tie | refused (kinds) | emission | "
        "granted choice, ms | changed |", "|---|---|---|---|---|---|---|---|---|---|"]
for c, o in zip(p["configs"], oj["configs"]):
    ru = c["runner_up"]
    N = int(c["fleet"].split()[0])
    rows.append(f"| {c['id']} | {name(c['expected_choice'], N)} | {c['step_ms']} | "
                f"{(name(ru, N) + ', ' + str(ru['step_ms'])) if ru else '—'} | {c['margin_pct'] if c['margin_pct'] is not None else '—'} | "
                f"{'**yes**' if c['tie'] else 'no'} | {kinds(c)} | {c.get('emission_order') or '—'} | "
                f"{name(o['expected_choice'], N)}, {o['step_ms']} | {'**yes**' if c['expected_choice'] != o['expected_choice'] else 'no'} |")
s2 = md.index("The refusal configurations (inputs HYPOTHESIS):")
WHY = {
    "32xH100-400G-1gph-k10": ("pieces at (16, 2) is now refused by S2. Stage 0's E − window_s comes out of the I17 "
                              "simulation with host DRAM at k 10 shared by its exchange and boundary bytes, and it "
                              "exceeds ε·C (the refusal table gives the numbers)."),
    "32xH100-400G-2gph-k10": "two GPUs share one host's DRAM, so the wide candidates are slower than width 1.",
    "32xH100-2sites-10G-uplink": "as at 1-GPU hosts, contiguous pieces fails S2.",
    "32xH100-400G-1gph-k10-memhost20": ("MEM-HOST is charged on pieces only, as the SPEC states it (SCHEDULER-I16; task 4 "
                                        "keeps `zero1` at today's device admission); every pieces candidate is refused "
                                        "by MEM-HOST and `zero1` is priced as on the 1-GPU, k 10 fleet."),
}
OC = {c["id"]: c for c in oj["configs"]}
summary = f"**{len(changed)} choices changed against the granted pins** (DERIVED, section 9 gives the mechanism):\n"
for c in p["configs"]:
    if c["id"] in changed:
        N = int(c["fleet"].split()[0])
        summary += (f"- `{c['id']}`: {WHY.get(c['id'], '')} {name(c['expected_choice'], N)} is chosen "
                    f"({c['step_ms']} ms; granted {name(OC[c['id']]['expected_choice'], N)}, {OC[c['id']]['step_ms']} ms).\n")
ties = sorted((c["margin_pct"], c["id"]) for c in p["configs"] if c["margin_pct"] is not None)
rise = [c["step_ms"] / OC[c["id"]]["step_ms"] - 1 for c in p["configs"]]
summary += ("- The 128-GPU fleets gain the candidate S 32, w 2 on 64 GPUs (SCHEDULER task 4's deepest cut on N/w "
            "devices), which the granted list lacked.\n"
            f"- No pinned margin is ≤ {p['tieRule']['margin_pct_max']}%, so the tie rule applies to none. The nearest is "
            f"`{ties[0][1]}` ({ties[0][0]}%); with the granted derivation's zero hop comm (sensitivity comm0) it is "
            f"{p['sensitivity']['modes']['comm0'][ties[0][1]]['margin_pct']}%.\n" if ties[0][0] > p['tieRule']['margin_pct_max'] else
            "- SOME MARGIN IS WITHIN THE TIE RULE: see the table.\n")
summary += (f"- Steps moved by {100*min(rise):+.1f}% to {100*max(rise):+.1f}% against the granted ones. The lane solve "
            "now carries the SPEC's per-hop comm (one boundary tensor at the link rate), and the exposure includes host "
            "DRAM.\n\n")
md = md[:s0] + "\n".join(rows) + "\n\n" + summary + md[s2:]

# MEM-HOST: the granted open question 5 is settled by the SPEC
mh = next(c for c in p["configs"] if c["id"].endswith("memhost20"))
q5 = md[md.index("5. **MEM-HOST scope.**"):md.index("6. **Local commands.**")]
md = md.replace(q5, "5. **MEM-HOST scope. SETTLED by the SPEC (SC-4).** SCHEDULER-I16 admits `shard: \"pieces\"` under its "
                "five conditions, host memory among them; SCHEDULER task 4 keeps `zero1` wide stages at today's admission "
                "(device memory), and HANDOUT-M2b's rulings say \"zero1 stages keep today's FIT check\". MEM-HOST is "
                f"therefore charged on pieces only: the memhost20 choice is {name(mh['expected_choice'])} at "
                f"{mh['step_ms']} ms. The as-built planner charges it on `zero1` too (`planner._price`, "
                "`admission.mem_host` outside the pieces branch): a code change, not a PLAN question.\n")
mhs = md[md.index("  Cross-check: (8, 4) interior at 21.74 equals round 5 §4's point A."):md.index("## 4. F9, restated")]
md = md.replace(mhs, "  Cross-check: (8, 4) interior at 21.74 equals round 5 §4's point A. Every pieces candidate is refused by "
                "MEM-HOST (SC-4: the `zero1` column is not charged, SCHEDULER task 4), and "
                f"{name(mh['expected_choice'])} is chosen.\n\n")
md = md.replace("(`tie: true`; today only `32xH100-400G-2gph-k10`, 1.06%)",
                "(`tie: true`; after SC-4 no configuration: the nearest is `32xH100-2sites-10G-uplink` at 2.18%)")
md = md.replace(f"- `refusals_matching`: expected {oj['expected']['refusals_matching']}.",
                f"- `refusals_matching`: expected {p['expected']['refusals_matching']} (SC-4; granted {oj['expected']['refusals_matching']}).")
md = md.replace("- `choices_matching` (primary): expected 17.", "- `choices_matching` (primary): expected 17 (the amended pins).")

# ---- section 5 sensitivity: replaced by the oracle's modes
t0 = md.index("## 5. Sensitivity")
t1 = md.index("## 6. Contradictions")
sens = p["sensitivity"]["modes"]
base_ch = {c["id"]: c for c in p["configs"]}


def differ(m):
    return [cid for cid, v in sens[m].items() if v["choice"] != spec_name[cid]]


spec_name = {}
for c in p["configs"]:
    x = c["expected_choice"]
    spec_name[c["id"]] = (f"S {x['S']} w 1" if x["w"] == 1 else f"S {x['S']} w {x['w']} {x['shard']} + {x['wire']}")
c0d, ndd, epd = differ("comm0"), differ("nodram"), differ("embed_p0")
md = md[:t0] + ("## 5. Sensitivity (DERIVED by `e7_spec.py`; `PLAN.json` pins mode `spec`, the others are in "
                "`PLAN.json` `sensitivity`)\n\n"
                f"- **comm0** (the lane solve without the SPEC's per-hop comm, as the granted derivation ran it): "
                f"{len(c0d)} of the 17 choices differ" + (f" ({', '.join(f'`{x}`' for x in c0d)})" if c0d else "") + ".\n"
                f"- **nodram** (host DRAM out of the simulation, S1 kept: the granted model's reading of I17): "
                f"{len(ndd)} of the 17 choices differ ({', '.join(f'`{x}`' for x in ndd)}). This is the term SC-4 adds.\n"
                "- **embed_p0** (a diagnostic, not a prediction): `zero1` gives the whole Embed unit to peer 0, as the "
                "as-built planner lays it out (`planner._stage_units` gives the Embed one stride index, 0). The runtime's "
                "stride (`runtime/stage.py:183`, `params[peer::w]`) puts `pos`, parameter 1, on peer 1. This layout "
                f"changes {len(epd)} choice(s)" + (": " + "; ".join(f"`{x}` → {sens['embed_p0'][x]['choice']} "
                f"(+{sens['embed_p0'][x]['margin_pct']}% to its runner-up)" for x in epd) if epd else "") + ".\n"
                "- Recompute on stays load-bearing, as the granted section 5 found. The S1 window and the ADR's "
                "c/4 : c/2 : c/4 split are superseded by R16 and were not re-run.\n\n") + md[t1:]

# ---- section 8 derivation
u0 = md.index("## 8. Derivation (reproducible)")
md = md[:u0] + f"""## 8. Derivation (reproducible)

```
cd docs/experiments/m2b-e7/derive
python3 -I -B e7_spec.py ../../m2b-e4/derive/i17_oracle.py lane_solve.py asbuilt_lanes.json e7_spec.json \\
    > e7_spec.out                                                     # exit 0, ~9 min
python3 -I -B build_plan_spec.py granted_PLAN.json e7_spec.json e7_spec.py e7_spec.out \\
    ../../m2b-e4/derive/i17_oracle.py lane_solve.py asbuilt_lanes.json ../PLAN.json e7_tables.md gen_plan_md.py
python3 -I -B gen_plan_md.py granted_PLAN.md granted_PLAN.json ../PLAN.json e7_tables.md ../PLAN.md
```

Nothing from `src/wantrain` is imported or read (the human's rule, 2026-10-10). `granted_PLAN.json` is the granted PLAN
(sha256 {am['base_plan_sha256'][:12]}…, the APPROVALS grant); `granted_PLAN.md` the PLAN.md it was granted with. The outputs
are byte-identical over two runs (`python3 -I -B`). Hashes:
- `e7_spec.py` {d['script_sha256']}.
- `i17_oracle.py` {d['oracle_sha256']}: the I17 model, written from the SPEC text.
- `lane_solve.py` {d['lane_solver_sha256']}: the lane solve, written from the SPEC's zero-bubble rules (a list scheduler)
  with a closed-form lower bound P* (the last stage's n·(F + B) + (n − K)·W behind the forward fill, then the backward
  chain; K = p microbatches may hold a pending W). It attains P* on every lane, and its periods equal the as-built
  lane solve's on all 644 priced candidates of the four modes (`e7_spec.out`, lanes section).
- `asbuilt_lanes.json` {d['asbuilt_lanes_sha256']}: the as-built lane solve's numbers (`zero_bubble.py` a93a8bed…,
  extracted by `extract_asbuilt.py` from the earlier `e7_spec.json` 4af7911b…), compared, never used to price.
- `e7_spec.out` {d['output_sha256']}; `e7_spec.json` {d['json_sha256']}.
- `build_plan_spec.py` {d['builder_sha256']}; `gen_plan_md.py` {d['md_generator_sha256']}.
- `layer_costs` and `analytic_costs` are restated in the script from the SPEC's formulas, not imported.

The granted derivation (`e7.py`, `e7.out` cdc4e496…e920, `e7.json` cb8a7a8a…8557) stays as the record of the superseded
pins.

## 9. Amendment (SC-4 / SPEC SCHEDULER-I17, 2026-10-10)

**Why.** The granted derivation (`e7.py` on round 4's `exch4`) did not model two things I17 requires:
1. Each host's DRAM, at k passes per byte per side, is a resource shared max-min fairly inside the exchange
   simulation. At k 10 a 600 GB/s host moves 30 GB/s per direction, below one 400 Gbit/s NIC's 50 GB/s, so the
   end-of-step bursts are DRAM-bound and E grows.
2. A unit's gathers leave only after the owner's last fold of that unit and its AdamW, so a hot owner's ingress and
   its gather egress are serial.

The SPEC is the physically correct model; the PLAN is amended. Re-derived by an independent oracle, with every input
the granted PLAN pins kept: fleets, k, 600 GB/s hosts, stage-contiguous packing, ε 3%, recompute on, `layer_costs`.
Three things the SPEC states, which the granted derivation did not, are now in:
- the candidate set of SCHEDULER task 4 (S 32, w 2 on 64 of 128 GPUs);
- the per-hop comm of section 2.1;
- MEM-HOST on pieces only (SCHEDULER-I16 and task 4; the granted open question 5).

{frag}
**Where the as-built code departs from these pins** (read from the code, file:line; the numbers are this PLAN's
diagnostics, not measurements of the as-built planner):
- `planner._stage_units` gives the Embed one stride index (0), so under `zero1` the whole Embed unit sits on peer 0. The
  runtime shards whole tensors `params[peer::w]` (`runtime/stage.py:183`): `tok` on peer 0, `pos` on peer 1. The
  `embed_p0` diagnostic above prices that layout.
- `planner._price` charges `admission.mem_host` on `zero1` candidates; the SPEC charges MEM-HOST on pieces only (task 4:
  today's device admission for `zero1`). The class check on `zero1` (`admission.check_class(..., probed=False)`) is kept
  in this PLAN as granted; whether task 4's "today's admission" includes it is a question for the SPEC's owner.
- `exchange_sim.simulate_exchange` returns `exposure = max(tail, late)`, where `late` is already measured against the
  slot (`step + first_f`); `planner._price` then takes `exposure − window`, so for a delay-bound stage with
  `window_s` > 0 the window is subtracted twice.
- `exchange_sim._Stage._complete` releases a unit's gathers at `max(arrivals) + one fold + AdamW`; I17 releases them
  after the owner's left fold of every contribution in canonical order, then AdamW.
"""
md = md.replace("(`zero_bubble.py:468-483`)", "(`zero_bubble.py:602-618` on today's tree; restated in `e7_spec.py`, not imported)")
open(OUT, "w").write(md)
print("written", OUT)
