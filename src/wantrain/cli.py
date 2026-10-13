"""`wantrain` command line.

  wantrain coordinator --port 28000 --job job.json      control plane, runs one job
  wantrain worker --coordinator HOST:28000 --coordinator-id ID --advertise-host IP --data-port 29000
  wantrain stage --spec spec.json --rank R [--peer P]    one slot, no control plane
  wantrain local --spec spec.json [--compare]            every slot on this machine
  wantrain plan --fleet fleet.json --out spec.json       planner only
  wantrain reference --spec spec.json                    single-device oracle
  wantrain compare A_losses.json B_losses.json           bitwise diff
  wantrain probe | fit | validate-model | prepare-data

A job file is {"spec": <JobSpec JSON>} (a recipe for the scheduler to place)
or {"plan": <job description>, "stages": N} (the planner decides precision,
microbatching and partition from the measured fleet).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _kv(items) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def _spec_model(d) -> dict:
    model = d.get("model") if isinstance(d, dict) else None
    return model if isinstance(model, dict) else {}


def _m4_refusal(model: dict) -> str | None:
    """The tripwire's refusal of an hf or MoE spec while M4 is unarmed (BA-464,
    I-16); None for every other spec, which starts exactly as before."""
    if model.get("arch") != "hf" and model.get("moe_experts", 0) in (0, None):
        return None
    from wantrain.coordinator.coordinator import m4_unarmed_reason

    return m4_unarmed_reason(model)


def cmd_stage(a) -> int:
    # Read from the JSON before JobSpec.load, so the refusal holds whichever
    # spec.py is on the tree (BA-52, interface I-9).
    with open(a.spec) as f:
        d = json.load(f)
    shard = d.get("shard", "zero1")
    model = _spec_model(d)
    if model.get("arch") == "hf":
        print(
            "wantrain stage: refusing model arch 'hf': a one-slot launch runs no launch gate "
            "and no admission (BA-445); launch through 'wantrain coordinator' or "
            "'wantrain local', which run the gate and admission",
            file=sys.stderr,
        )
        return 2
    reason = _m4_refusal(model)
    if reason is not None:
        print(f"wantrain stage: refusing: {reason}", file=sys.stderr)
        return 2
    if shard == "pieces":
        print(
            "wantrain stage: refusing shard 'pieces': a one-slot launch cannot verify that "
            "its peers share its device class (BA-52); use 'wantrain coordinator' or "
            "'wantrain local'",
            file=sys.stderr,
        )
        return 2
    from wantrain.worker.stage_main import run_stage

    return run_stage(a.spec, a.rank, a.peer)


def cmd_coordinator(a) -> int:
    from wantrain.coordinator.coordinator import Coordinator, CoordinatorConfig, Job, StartRefused
    from wantrain.runtime.spec import JobSpec
    from wantrain.scheduler.selection import SelectionPolicy

    cfg = CoordinatorConfig(
        state_dir=a.state_dir,
        token=a.token or os.environ.get("WANTRAIN_COORD_TOKEN"),
        probe=not a.no_probe,
        fit_check=not a.no_fit_check,
        selection=SelectionPolicy(
            group_by=a.group_by,
            distinct_hosts=not a.allow_shared_hosts,
            same_region=not a.any_region,
        ),
        replacement_match=a.replacement_match,
        max_heals=a.max_heals,
        zb_mem_mode=a.zb_mem_mode,
    )
    coord = Coordinator(a.host, a.port, cfg)
    try:
        coord.start()
    except StartRefused as e:
        print(f"wantrain coordinator: {e}", file=sys.stderr)
        return 2
    with open(a.job) as f:
        jd = json.load(f)
    if "spec" in jd:
        job = Job(spec=JobSpec.from_json(json.dumps(jd["spec"])), run_id=jd.get("run_id"))
    else:
        job = Job(plan=jd["plan"], stages=int(jd.get("stages", 2)), run_id=jd.get("run_id"))
    res = coord.run_job(job, wait_workers_s=a.wait_workers_s)
    print(
        json.dumps(
            {
                "ok": res.ok,
                "run_id": res.run_id,
                "reason": res.reason,
                "run_dir": res.run_dir,
                "attempts": res.attempts,
                "heals": res.heals,
                "restarts": res.restarts,
            },
            indent=2,
            default=str,
        )
    )
    coord.stop(shutdown_workers=not a.keep_workers)
    return 0 if res.ok else 1


def cmd_worker(a) -> int:
    from wantrain.worker.agent import WorkerAgent

    emulation = None
    if a.wan_profile:
        with open(a.wan_profile) as f:
            emulation = {"region": a.region, "profile": json.load(f)}
    declare = _kv(a.declare)
    if a.region:
        declare.setdefault("region", a.region)
    try:
        agent = WorkerAgent(
            a.coordinator,
            a.advertise_host,
            a.data_port,
            listen_port=a.listen_port,
            probe_port=a.probe_port,
            probe_listen_port=a.probe_listen_port,
            name=a.name,
            workdir=a.workdir,
            device=a.device,
            declare=declare,
            emulation=emulation,
            bench=not a.no_bench,
            coordinator_id=a.coordinator_id,
        )
    except ValueError as e:
        print(f"wantrain worker: {e}", file=sys.stderr)
        return 2
    agent.run()
    return 0


def cmd_local(a) -> int:
    with open(a.spec) as f:
        reason = _m4_refusal(_spec_model(json.load(f)))
    if reason is not None:
        print(f"wantrain local: refusing: {reason}", file=sys.stderr)
        return 2
    from wantrain.launch.local import run_and_compare, run_local
    from wantrain.runtime.spec import JobSpec

    spec = JobSpec.load(a.spec)
    if a.out_dir:
        spec.out_dir = a.out_dir
    res = run_and_compare(spec, a.timeout) if a.compare else run_local(spec, a.timeout)
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


def cmd_plan(a) -> int:
    from wantrain.scheduler.fleetplan import PlanRefused, build_spec

    with open(a.fleet) as f:
        fleet = json.load(f)
    try:
        r = build_spec(fleet)
    except PlanRefused as e:
        print(str(e), file=sys.stderr)
        return 1
    with open(a.out, "w") as f:
        f.write(r["spec_json"])
    print(r["report"], file=sys.stderr)
    for w in r["warnings"]:
        print(f"WARN {w}", file=sys.stderr)
    print("WT_PLAN " + json.dumps(r["rationale"], default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wantrain")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("coordinator", help="control plane: register, schedule, run, heal")
    # workers dial in over the WAN
    c.add_argument("--host", default="0.0.0.0")  # nosec B104
    c.add_argument("--port", type=int, default=28000)
    c.add_argument("--state-dir", default="wantrain-state")
    c.add_argument("--token", default=None)
    c.add_argument("--job", required=True)
    c.add_argument("--wait-workers-s", type=float, default=600.0)
    c.add_argument("--no-probe", action="store_true")
    c.add_argument("--no-fit-check", action="store_true")
    c.add_argument("--group-by", default="any", choices=["any", "model", "series"])
    c.add_argument(
        "--allow-shared-hosts",
        action="store_true",
        help="allow several slots on one address (single-host experiments)",
    )
    c.add_argument("--any-region", action="store_true")
    c.add_argument(
        "--zb-mem-mode",
        default="auto",
        choices=["auto", "2p", "1p", "given"],
        help="zero-bubble memory mode: auto (measured budgets), 2p, or 1p (at most one pending W per stage in "
        "flight: the least activation memory; with few stages its bubble is ~(S-1)/(M+S-1))",
    )
    c.add_argument(
        "--replacement-match", default="series", choices=["model", "series", "device", "any"]
    )
    c.add_argument("--max-heals", type=int, default=2)
    c.add_argument("--keep-workers", action="store_true")
    c.set_defaults(fn=cmd_coordinator)

    w = sub.add_parser("worker", help="node agent: register, heartbeat, run stages")
    w.add_argument("--coordinator", required=True, help="host:port")
    w.add_argument(
        "--coordinator-id",
        default=None,
        help="the coordinator's COORDINATOR_ID (default: $WANTRAIN_COORD_ID); required off loopback",
    )
    w.add_argument("--advertise-host", required=True)
    w.add_argument("--data-port", type=int, required=True)
    w.add_argument("--listen-port", type=int, default=0)
    w.add_argument("--probe-port", type=int, default=0, help="0 = any free port, -1 = none")
    w.add_argument(
        "--probe-listen-port",
        type=int,
        default=0,
        help="bind the probe server here and advertise --probe-port (behind a port map), like --listen-port",
    )
    w.add_argument("--name", default="")
    w.add_argument("--workdir", default="wantrain-work")
    w.add_argument("--device", default=None, help="cuda | mps | cpu (default: best)")
    w.add_argument("--region", default=None)
    w.add_argument("--wan-profile", default=None, help="JSON region-pair link model")
    w.add_argument("--declare", nargs="*", help="k=v resource declarations")
    w.add_argument("--no-bench", action="store_true")
    w.set_defaults(fn=cmd_worker)

    s = sub.add_parser("stage", help="run one slot of a spec")
    s.add_argument("--spec", required=True)
    s.add_argument("--rank", type=int, required=True)
    s.add_argument("--peer", type=int, default=0)
    s.set_defaults(fn=cmd_stage)

    lo = sub.add_parser("local", help="run every slot of a spec on this machine")
    lo.add_argument("--spec", required=True)
    lo.add_argument("--out-dir", default=None)
    lo.add_argument("--compare", action="store_true", help="diff against the reference")
    lo.add_argument("--timeout", type=float, default=600.0)
    lo.set_defaults(fn=cmd_local)

    p = sub.add_parser("plan", help="fleet description -> spec")
    p.add_argument("--fleet", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_plan)

    for name, mod in (
        ("reference", "wantrain.validation.reference"),
        ("probe", "wantrain.resources.probe"),
        ("fit", "wantrain.resources.fit"),
    ):
        sp = sub.add_parser(name, add_help=False)
        sp.add_argument("rest", nargs=argparse.REMAINDER)
        sp.set_defaults(fn=lambda a, m=mod: __import__(m, fromlist=["x"]).main(a.rest))

    cm = sub.add_parser("compare", help="bitwise diff of two loss files")
    cm.add_argument("a")
    cm.add_argument("b")

    def _cmp(a):
        from wantrain.validation.compare import compare

        r = compare(a.a, a.b)
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1

    cm.set_defaults(fn=_cmp)

    vm = sub.add_parser("validate-model", help="check a build(cfg) model file")
    vm.add_argument("--source", required=True)
    vm.add_argument("--target", default="{}")

    def _vm(a):
        from wantrain.validation.model_contract import validate

        with open(a.source) as f:
            r = validate(f.read(), json.loads(a.target))
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1

    vm.set_defaults(fn=_vm)

    pd = sub.add_parser("prepare-data", help="tokenize a corpus into a token stream")
    pd.add_argument("--corpus", required=True)
    pd.add_argument("--out", required=True)
    pd.add_argument("--tokenizer", default="gpt2")
    pd.add_argument("--source-type", default="auto")
    pd.add_argument("--text-key", default="text")
    pd.add_argument("--shards", type=int, default=1)
    pd.add_argument("--limit-tokens", type=int, default=None)

    def _pd(a):
        from wantrain.model import dataset as ds

        m = ds.prepare(
            a.corpus,
            a.out,
            tokenizer=a.tokenizer,
            source_type=a.source_type,
            text_key=a.text_key,
            n_shards=a.shards,
            limit_tokens=a.limit_tokens,
        )
        print("WT_CORPUS " + json.dumps(m))
        return 0

    pd.set_defaults(fn=_pd)

    a = ap.parse_args(argv)
    rc = a.fn(a)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
