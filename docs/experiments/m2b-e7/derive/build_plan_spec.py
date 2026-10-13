# Builds the SC-4 amended m2b-e7 PLAN.json from the granted PLAN.json and e7_spec.json (mode "spec" pins; the other
# modes are recorded as sensitivity), without hand-copied numbers, and writes the PLAN.md tables as a fragment.
# usage: python3 -I -B build_plan_spec.py granted_PLAN.json e7_spec.json e7_spec.py e7_spec.out \
#            ../../m2b-e4/derive/i17_oracle.py lane_solve.py asbuilt_lanes.json ../PLAN.json e7_tables.md gen_plan_md.py
import copy
import hashlib
import json
import sys

BASE, SPECJ, SCRIPT, OUTTXT, ORACLE, LANES, ASBUILT, OUT, MD, MDGEN = sys.argv[1:11]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


base = json.load(open(BASE))
assert not any("SC-4" in json.dumps(a) for a in base.get("amendments", [])), "base already carries SC-4"
spec = json.load(open(SPECJ))
modes = spec["modes"]
TIE = base["tieRule"]["margin_pct_max"]


def plan_of(row):
    if row is None:
        return None
    if row["w"] == 1:
        return dict(S=row["S"], w=1, shard=None, wire=None, gpus_used=row["gpus_used"])
    return dict(S=row["S"], w=row["w"], shard=row["shard"], wire=row["wire"], gpus_used=row["gpus_used"])


def pinned_name(row):
    """The granted PLAN's candidate naming (the e7 test parses it): 'S 16 w 2 pieces', 'S 16 w 2 zero1 + native',
    with ', straddling' when it straddles sites."""
    lay = "pieces" if row["shard"] == "pieces" else f"zero1 + {row['wire']}"
    return f"S {row['S']} w {row['w']} {lay}" + (", straddling" if row["place"] == "straddling" else "")


def reason_text(r):
    if r["kind"] == "S2":
        return f"S2: E - window_s {r['value']:.1f} ms > eps*C {r['bound']:.1f} ms ({r['where']})"
    if r["kind"] == "S1 NIC":
        return f"S1 NIC: {r['where']} {r['value']:.2f}x cap*c_s"
    if r["kind"] == "S1 host DRAM":
        return f"S1 host DRAM: {r['where']} {r['value']:.1f} GB/s > {r['bound']:.0f} GB/s"
    if r["kind"] == "S1 uplink":
        return f"S1 uplink {r['where']}: {r['value']:.1f} GB/s > {r['bound']:.2f} GB/s"
    if r["kind"] == "MEM-HOST":
        return f"MEM-HOST: {r['where']} {r['value']:.2f} GB > {r['bound']:.0f} GB"
    return f"{r['kind']}: {r['where']} spans classes"


by_id = {c["config"]["id"]: c for c in modes["spec"]}
new = copy.deepcopy(base)
superseded = {}
total_refusals = 0
changed_vs_granted = []
for cfg in new["configs"]:
    cid = cfg["id"]
    r = by_id[cid]
    rows = {q["name"]: q for q in r["rows"]}
    best = rows[r["choice"]]
    ru = rows[r["runner_up"]] if r["runner_up"] else None
    old = {k: copy.deepcopy(cfg.get(k)) for k in ("expected_choice", "step_ms", "runner_up", "margin_pct", "tie",
                                                    "refused", "changed", "note")}
    cfg["expected_choice"] = plan_of(best)
    cfg["step_ms"] = r["step"]
    cfg["runner_up"] = None if ru is None else dict(plan_of(ru), step_ms=r["runner_up_step"])
    cfg["margin_pct"] = r["margin_pct"]
    cfg["tie"] = r["margin_pct"] is not None and r["margin_pct"] <= TIE
    refused = []
    for q in r["rows"]:
        if q["admitted"] or q["w"] == 1:
            continue
        if q["shard"] == "zero1" and q["wire"] != "native":
            continue                       # the e7 test names pieces and zero1 + native only
        refused.append(dict(candidate=pinned_name(q), reasons=q["reasons"],
                            text=" | ".join(reason_text(x) for x in q["reasons"])))
    cfg["refused"] = refused
    total_refusals += len(refused)
    cfg["emission_order"] = best.get("emission")
    cfg["changed"] = cfg["expected_choice"] != cfg.get("adr_choice") if cfg.get("adr_choice") else False
    if cid == "32xH100-2sites-10G-uplink":
        cfg["note"] = ("ADR-0006 names only the refusal (the uplink, for every candidate whose wide stage straddles "
                       "it). Under SCHEDULER-I17 the contiguous pieces candidate is refused by S2 (host DRAM at k 10 "
                       "in the simulation), so zero1 + native is chosen.")
    if cid == "32xH100-400G-1gph-k10-memhost20":
        cfg["note"] = ("MEM-HOST is charged on pieces only, as the SPEC states it (SCHEDULER-I16 admits shard pieces "
                       "under its five conditions, host memory among them; SCHEDULER task 4: zero1 wide stages keep "
                       "today's admission, device memory). Every pieces candidate is refused by MEM-HOST; zero1 "
                       f"candidates are priced and {best['name']} is chosen. This settles the granted PLAN's open "
                       "question 5 (it charged every wide candidate).")
    for k, v in old.items():
        if v != cfg.get(k):
            superseded[f"configs[{cid}].{k}"] = v
    if old["expected_choice"] != cfg["expected_choice"]:
        changed_vs_granted.append(cid)

new["expected"] = dict(base["expected"], refusals_matching=total_refusals)
if base["expected"]["refusals_matching"] != total_refusals:
    superseded["expected.refusals_matching"] = base["expected"]["refusals_matching"]
new["costModel"] = dict(base["costModel"],
                        exchange=("SCHEDULER-I17 oracle (docs/experiments/m2b-e4/derive/i17_oracle.py): per-(sender, "
                                  "destination) need-first queues, 16 MiB chunks, egress and ingress at min(NIC, PCIe), "
                                  "host DRAM at k passes per side, uplinks, max-min fair; gathers after the owner's last "
                                  "fold and AdamW; E and window_s from one simulation of every microbatch"),
                        step="period + max_s max(0, E_s - window_s) + T_ovh (+ the uplink floor where a hop crosses it)",
                        hop_comm="one boundary tensor at the link rate + RTT/2 per hop (SPEC section 2.1); 0 in the "
                                 "granted derivation (sensitivity comm0: no choice moves)",
                        candidates=("width 1 at the deepest cut; for w in 2, 4, 8 the deepest cut on N/w devices with "
                                    "pieces + native, zero1 + native, zero1 + fp32 (SCHEDULER task 4)"),
                        zero1_ownership="stride over the parameter list, whole tensors (tok on peer 0, pos on peer 1)",
                        mem_host=("charged on pieces only (SCHEDULER-I16; task 4: zero1 keeps today's admission, "
                                  "device memory); the class rule on every wide candidate, as granted"),
                        lane_solve=("lane_solve.py: the SPEC's zero-bubble rules as a list scheduler, independent of "
                                    "src/wantrain; it attains the closed-form lower bound P* on every lane and equals "
                                    "the as-built lane periods exactly (e7_spec.out, lanes section)"))
new["derivation"] = {
    "script": "docs/experiments/m2b-e7/derive/e7_spec.py",
    "oracle": "docs/experiments/m2b-e4/derive/i17_oracle.py",
    "lane_solver": "docs/experiments/m2b-e7/derive/lane_solve.py (independent of src/wantrain)",
    "asbuilt_record": ("docs/experiments/m2b-e7/derive/asbuilt_lanes.json (the as-built lane solve's numbers, extracted "
                       "by extract_asbuilt.py; compared, never used to price)"),
    "independence": "nothing from src/wantrain is imported or read (the human's rule, 2026-10-10)",
    "command": ("cd docs/experiments/m2b-e7/derive && python3 -I -B e7_spec.py ../../m2b-e4/derive/i17_oracle.py "
                "lane_solve.py asbuilt_lanes.json e7_spec.json > e7_spec.out && python3 -I -B build_plan_spec.py "
                "granted_PLAN.json e7_spec.json e7_spec.py e7_spec.out ../../m2b-e4/derive/i17_oracle.py lane_solve.py "
                "asbuilt_lanes.json ../PLAN.json e7_tables.md gen_plan_md.py && python3 -I -B gen_plan_md.py "
                "granted_PLAN.md granted_PLAN.json ../PLAN.json e7_tables.md ../PLAN.md"),
    "script_sha256": sha(SCRIPT),
    "oracle_sha256": sha(ORACLE),
    "lane_solver_sha256": sha(LANES),
    "asbuilt_lanes_sha256": sha(ASBUILT),
    "builder_sha256": sha(sys.argv[0]),
    "md_generator_sha256": sha(MDGEN),
    "output_sha256": sha(OUTTXT),
    "json_sha256": sha(SPECJ),
    "closes": "SC-4 / SPEC SCHEDULER-I17 for E7",
    "superseded_derivation": base["derivation"],
}
sens = {}
for m, res in modes.items():
    if m == "spec":
        continue
    sens[m] = {c["config"]["id"]: dict(choice=c["choice"], step_ms=c["step"], runner_up=c["runner_up"],
                                       margin_pct=c["margin_pct"]) for c in res}
new["sensitivity"] = dict(
    note=("DERIVED by e7_spec.py; not pinned. comm0: the lane solve's hop comm 0 (the granted derivation's). nodram: "
          "host DRAM out of the simulation (S1 kept), the granted model's reading. embed_p0: a diagnostic of the "
          "as-built planner's zero1 layout (the whole Embed on peer 0), not a prediction."),
    modes=sens)
new["hypothesis"] = base["hypothesis"].replace(
    "whose expected step time the round-5 model re-derives",
    "whose expected step time the SCHEDULER-I17 oracle (SC-4) derives")
amend = {
    "id": "SC-4 / SPEC SCHEDULER-I17",
    "date": "2026-10-10",
    "reason": ("The granted derivation (e7.py on round 4's exch4) priced host DRAM only in S1 and a busiest-host bound, "
               "and let a hot owner's gathers overlap its ingress. SCHEDULER-I17, the SPEC of record, shares each "
               "host's DRAM at k passes per side inside the exchange simulation and releases a unit's gathers after "
               "the owner's last fold and AdamW; S2 and the step read E from that one simulation. Re-derived by an "
               "independent oracle written from the SPEC text, on the candidate set of SCHEDULER task 4 (the 128-GPU "
               "fleets gain S 32 w 2) with the SPEC's per-hop comm, and with MEM-HOST charged on pieces only as the SPEC "
               "states it (open question 5 settled). The lane solve is independent of src/wantrain (lane_solve.py) and "
               "equals the as-built lane periods on every candidate. Changed choices: "
               + ", ".join(changed_vs_granted) + "."),
    "script": "docs/experiments/m2b-e7/derive/e7_spec.py",
    "script_sha256": sha(SCRIPT),
    "oracle_sha256": sha(ORACLE),
    "lane_solver_sha256": sha(LANES),
    "asbuilt_lanes_sha256": sha(ASBUILT),
    "output_sha256": sha(OUTTXT),
    "json_sha256": sha(SPECJ),
    "builder_sha256": sha(sys.argv[0]),
    "md_generator_sha256": sha(MDGEN),
    "base_plan_sha256": sha(BASE),
    "superseded": superseded,
}
new["amendments"] = list(base.get("amendments", [])) + [amend]
json.dump(new, open(OUT, "w"), indent=1)
open(OUT, "a").write("\n")


def fmt_plan(p):
    if p is None:
        return "—"
    if p["w"] == 1:
        return f"S {p['S']}, w 1 ({p['gpus_used']} GPUs)"
    return f"S {p['S']}, w {p['w']}, {p['shard']} + {p['wire']}"


L = ["| config | granted choice, step ms | SC-4 choice, step ms | SC-4 runner-up, ms | margin % (granted → SC-4) | tie | "
     "refused (granted → SC-4) | choice changed |", "|---|---|---|---|---|---|---|---|"]
for cfg, o in zip(new["configs"], base["configs"]):
    ru = cfg["runner_up"]
    L.append(f"| `{cfg['id']}` | {fmt_plan(o['expected_choice'])}, {o['step_ms']} | **{fmt_plan(cfg['expected_choice'])}**, "
             f"{cfg['step_ms']} | {fmt_plan(ru) if ru else '—'}{', ' + str(ru['step_ms']) if ru else ''} | "
             f"{o['margin_pct']} → {cfg['margin_pct']} | {o['tie']} → {cfg['tie']} | {len(o['refused'])} → "
             f"{len(cfg['refused'])} | {'**yes**' if o['expected_choice'] != cfg['expected_choice'] else 'no'} |")
L += ["", f"Refusals pinned: {base['expected']['refusals_matching']} → {total_refusals}.", ""]
L += ["| config | refused candidate | reasons (kind, where, value vs bound) |", "|---|---|---|"]
for cfg in new["configs"]:
    for rf in cfg["refused"]:
        L.append(f"| `{cfg['id']}` | {rf['candidate']} | " + "; ".join(
            f"{x['kind']} at {x['where']}: {x['value']} vs {x['bound']} {x['unit']}" for x in rf["reasons"]) + " |")
L += ["", "| config | SC-4 (spec) | comm0 | nodram | embed_p0 (diagnostic) |", "|---|---|---|---|---|"]
for cfg in new["configs"]:
    cid = cfg["id"]
    cells = [f"{by_id[cid]['choice']} {by_id[cid]['step']}"]
    for m in ("comm0", "nodram", "embed_p0"):
        s = sens.get(m, {}).get(cid)
        cells.append("—" if s is None else f"{s['choice']} {s['step_ms']} (+{s['margin_pct']}%)")
    L.append(f"| `{cid}` | " + " | ".join(cells) + " |")
open(MD, "w").write("\n".join(L) + "\n")
print("PLAN.json written:", OUT, "sha256", sha(OUT))
print("changed choices vs granted:", changed_vs_granted)
print("refusals:", total_refusals, "superseded fields:", len(superseded))
