# Builds docs/experiments/m2b-e7/PLAN.json from e7.py's output (e7.json), so no pinned number is transcribed by hand.
# usage: python3 -I build_plan.py e7.json PLAN.json E7_PY_SHA E7_OUT_SHA
import sys, json, re
E, OUT, SHA_PY, SHA_OUT = sys.argv[1:5]
d = json.load(open(E))
P = d["modes"]["planner"]; R5 = {(c["fleet"], c["link_gbps"], c["gpus_per_host"], c["k"]): c for c in d["modes"]["r5"]}

def parse(name):
    m = re.match(r"S (\d+) w (\d+)(?: on (\d+) GPUs)?\s*(.*)", name)
    S, w, used, rest = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    if w == 1:
        shard, wire = None, None
    elif rest.startswith("pieces"):
        shard, wire = "pieces", "native"
    elif "today (fp32)" in rest:
        shard, wire = "zero1", "fp32"
    else:
        shard, wire = "zero1", "native"
    out = dict(S=S, w=w, shard=shard, wire=wire)
    out["gpus_used"] = int(used) if used else S * w
    return out

# ADR-0006's a-priori choices (E7 paragraph; round5.md section 3), keyed (fleet, Gbps, GPUs per host, k)
ADR = {("32 H100", 400, 1, 10): "S 16 w 2 pieces", ("32 H100", 400, 2, 10): "S 16 w 2 zero1 + native",
       ("32 H100", 400, 4, 10): "S 32 w 1", ("32 H100", 400, 8, 10): "S 32 w 1", ("32 H100", 400, 8, 2): "S 16 w 2 pieces",
       ("32 H100", 100, 1, 10): "S 32 w 1", ("32 H100", 100, 8, 10): "S 32 w 1",
       ("128 H100", 400, 1, 10): "S 32 w 4 zero1 + native", ("128 H100", 400, 1, 6): "S 32 w 4 pieces",
       ("128 H100", 400, 1, 2): "S 32 w 4 pieces", ("128 H100", 400, 8, 10): "S 32 w 1 on 32 GPUs",
       ("128 H100", 400, 8, 2): "S 32 w 4 zero1 + native", ("128 H100", 100, 1, 10): "S 32 w 1 on 32 GPUs",
       ("128 H100", 100, 8, 10): "S 32 w 1 on 32 GPUs"}

configs = []
for c in P:
    key = (c["fleet"], c["link_gbps"], c["gpus_per_host"], c["k"])
    r5 = R5[key]
    ru = next(r for r in c["rows"] if r["name"] == c["runner_up"])
    configs.append(dict(
        id=f"{c['fleet'].split()[0]}xH100-{c['link_gbps']}G-{c['gpus_per_host']}gph-k{c['k']}",
        fleet=c["fleet"], link_gbps=c["link_gbps"], gpus_per_host=c["gpus_per_host"], k=c["k"],
        host_dram_gbps=600, host_ram_avail_gb=None, packing="stage-contiguous", sites=1,
        expected_choice=parse(c["choice"]), step_ms=c["step_ms"],
        runner_up=dict(parse(c["runner_up"]), step_ms=ru["step_ms"]), margin_pct=c["margin_pct"],
        refused=[dict(candidate=r["name"], reasons=r["reasons"], text=r["refused"]) for r in c["rows"] if not r["admitted"]],
        adr_choice=parse(ADR[key]), adr_step_ms=r5["step_ms"], adr_runner_up=r5["runner_up"], adr_margin_pct=r5["margin_pct"],
        changed=parse(ADR[key]) != parse(c["choice"])))

u = d["uplink_contiguous"]
configs.append(dict(
    id="32xH100-2sites-10G-uplink", fleet="32 H100 (2 sites x 16)", link_gbps=400, uplink_gbps=10, uplink_rtt_ms=50,
    gpus_per_host=1, k=10, host_dram_gbps=600, host_ram_avail_gb=None, packing="stage-contiguous, site-major", sites=2,
    expected_choice=parse(u["choice"]), step_ms=u["step_ms"],
    runner_up=dict(parse(u["runner_up"]), step_ms=u["runner_up_ms"]), margin_pct=u["margin_pct"],
    refused=[dict(candidate=x["candidate"], reasons=x["reasons"], text=x["reason"]) for x in d["uplink"]]
    + [dict(candidate=r["name"], reasons=r["reasons"], text=r["refused"]) for r in u["rows"] if not r["admitted"]],
    adr_choice=None, adr_step_ms=None, adr_runner_up=None, adr_margin_pct=None, changed=False,
    note="ADR-0006 names only the refusal (the uplink, for every candidate whose wide stage straddles it); no choice."))

base = next(c for c in configs if c["id"] == "32xH100-400G-1gph-k10")
w1 = next(r for r in P[0]["rows"] if r["name"] == "S 32 w 1")
configs.append(dict(
    id="32xGPU-mixed-classes", fleet="32 GPUs: 17 class A then 15 class B (equal rates)", link_gbps=400, gpus_per_host=1,
    k=10, host_dram_gbps=600, host_ram_avail_gb=None, packing="stage-contiguous", sites=1,
    expected_choice=parse("S 32 w 1"), step_ms=w1["step_ms"], runner_up=None, margin_pct=None,
    refused=[dict(candidate=x["candidate"], reasons=x["reasons"], text=x["reason"]) for x in d["mixed"]],
    adr_choice=None, adr_step_ms=None, adr_runner_up=None, adr_margin_pct=None, changed=False,
    note="Every wide candidate mixes classes in the stage holding GPUs 16 and 17; width 1 is the only admitted plan."))

mh = {x["candidate"]: x for x in d["memhost"]}
refused_mh = []
base_ref = {r["candidate"]: r for r in base["refused"]}
for n, x in mh.items():
    st = max(x["per_peer_gb"], key=x["per_peer_gb"].get)
    prior = base_ref.get(n, dict(reasons=[], text=""))
    refused_mh.append(dict(candidate=n, reasons=prior["reasons"] + [dict(kind="MEM-HOST", where=f"host of a {st} peer",
                                                                          value=x["max_gb"], bound=20.0, unit="GB")],
                           text=" | ".join(t for t in (prior["text"], f"MEM-HOST: {x['max_gb']} GB one-step bound > 20.0 GB available ({st})") if t)))
configs.append(dict(
    id="32xH100-400G-1gph-k10-memhost20", fleet="32 H100", link_gbps=400, gpus_per_host=1, k=10, host_dram_gbps=600,
    host_ram_avail_gb=20.0, packing="stage-contiguous", sites=1,
    expected_choice=parse("S 32 w 1"), step_ms=w1["step_ms"], runner_up=None, margin_pct=None, refused=refused_mh,
    adr_choice=None, adr_step_ms=None, adr_runner_up=None, adr_margin_pct=None, changed=False,
    note="MEM-HOST charged on every wide candidate (pieces and zero1 + native both stage exchange bytes in host memory); "
         "if the spec exempts zero1, the choice is S 16 w 2 zero1 + native (1163.7 ms): a question for the human."))

n_ref = sum(len(c["refused"]) for c in configs)
plan = {
    "id": "m2b-e7",
    "hypothesis": ("ADR-0006 F9: on each pinned fleet, the planner (plan_job, SCHEDULER task 4, I16 as written with "
                   "epsilon 3%, k counted, R10/R13 deferred) chooses the plan pinned here, whose expected step time the "
                   "round-5 model re-derives with the planner's own attention-inclusive costs (partition.layer_costs "
                   "for the cut, zero_bubble.analytic_costs with recompute for F/B/W), names every refused candidate "
                   "with its link, number, class or MEM-HOST, and never chooses a plan whose simulated step is more "
                   "than 1% above the best admitted candidate's."),
    "expected": {"choices_matching": len(configs), "refusals_matching": n_ref, "f9_violations": 0},
    "lowerIsBetter": ["f9_violations"],
    "primary": "choices_matching",
    "tolerancePct": 1,
    "local": {
        "correctness": "uv run pytest -q -p no:randomly tests/integration/test_engine_bitwise.py -k e1_features_on",
        "performance": ("uv run pytest -q -p no:randomly -s tests/unit/test_scheduler.py -k e7_ "
                        "| grep -E '^\\{\"choices_matching\"' | tail -n 1"),
        "failure": "uv run pytest -q -p no:randomly tests/unit/test_scheduler.py -k \"plan_wide and refuse\"",
    },
    "localNotes": ("None of these tests exist yet: task plan-wide-e7 (docs/design/PLAN-M2b.json) writes the e7_ and "
                   "plan_wide tests; e1_features_on is written by task pieces-identity-e1-e2, a dependency. The e7_ test "
                   "prints one JSON line {\"choices_matching\", \"refusals_matching\", \"f9_violations\"} before it "
                   "asserts; it reads every config below from this file."),
    "cloud": None,
    "costs": "layer_costs (partition.py) for the cut; analytic_costs (zero_bubble.py, recompute on) for F/B/W",
    "costModel": dict(F_flops_per_s=400e12, T_mb=4096, seq_len=4096, d_model=4096, vocab=32000, n_layer=32,
                      recompute=True, epsilon=0.03, host_dram_GBps=600, nic="min(NIC, PCIe 50 GB/s)",
                      label="every input HYPOTHESIS (rounds 3-5); every output DERIVED"),
    "derivation": dict(script="docs/experiments/m2b-e7/derive/e7.py", script_sha256=SHA_PY, output_sha256=SHA_OUT,
                       command=("python3 -I docs/experiments/m2b-e7/derive/e7.py docs/experiments/m2b-e7/derive/r5.py "
                                "docs/experiments/m2b-e7/derive/r4.py docs/experiments/m2b-e7/derive/r3.py "
                                "docs/experiments/m2b-e7/derive/r2.py src/wantrain/scheduler/zero_bubble.py "
                                "src/wantrain/scheduler/partition.py docs/experiments/m2b-e7/derive/e7.json")),
    "tieRule": dict(margin_pct_max=2.0, within_best_pct=1.0,
                    rule=("the human's ruling, 2026-10-09: where a config's pinned margin_pct to its runner-up is at most "
                          "2%, the runner-up also counts as matching if its simulated step is within 1% of the best "
                          "admitted candidate's (F9's second clause)")),
    "configs": [dict(c, tie=(c["margin_pct"] is not None and c["margin_pct"] <= 2.0)) for c in configs],
}
json.dump(plan, open(OUT, "w"), indent=1)
print(f"configs {len(configs)}, refusals {n_ref}, changed {sum(c['changed'] for c in configs)}")
