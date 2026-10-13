#!/usr/bin/env python
"""Run a real distributed training job on a local, WAN-emulated cluster.

One coordinator (this process) and N worker agents (separate processes, each
supervising its own stage process) on this machine. Workers sit in two
emulated regions joined by a slow, high-latency link; one may run on the Apple
GPU (MPS) for a genuinely heterogeneous fleet. Optionally a worker is killed
mid-run to show detection, replacement and an exact resume.

    uv run python examples/wan_cluster_demo.py --workers 5 --stages 4 --kill-at 6
    uv run python examples/wan_cluster_demo.py --mps          # CPU + GPU fleet

The last line prints the comparison against the single-device reference: on
an all-CPU fleet the loss sequence is bit-identical, heal or no heal.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wantrain.coordinator.coordinator import (  # noqa: E402
    Coordinator,
    CoordinatorConfig,
    Job,
)
from wantrain.launch.local import ensure_reference  # noqa: E402
from wantrain.model.gpt import make_config  # noqa: E402
from wantrain.runtime.spec import JobSpec  # noqa: E402
from wantrain.scheduler.selection import SelectionPolicy  # noqa: E402
from wantrain.validation.compare import compare  # noqa: E402

WAN = {"intra": {"mbps": 400, "rtt_ms": 2}, "inter": {"mbps": 60, "rtt_ms": 60}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--kill-at", type=int, default=0, help="kill a worker at this step")
    ap.add_argument("--mps", action="store_true", help="put worker 0 on the Apple GPU")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="wantrain-demo-")
    cfg = CoordinatorConfig(
        state_dir=os.path.join(tmp, "coord"),
        selection=SelectionPolicy(distinct_hosts=False, same_region=False),
        probe_bulk_bytes=1 << 20,
        measure_bulk_bytes=1 << 20,
        heartbeat_timeout_s=6,
        reconnect_grace_s=4,
        strike_gap_s=2,
        stall_floor_s=20,
    )
    coord = Coordinator("127.0.0.1", 0, cfg)
    port = coord.start()
    procs = {}
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), PYTHONUNBUFFERED="1")
    for i in range(a.workers):
        wd = os.path.join(tmp, f"w{i}")
        os.makedirs(wd)
        prof = os.path.join(wd, "wan.json")
        with open(prof, "w") as f:
            json.dump(WAN, f)
        dev = "mps" if (a.mps and i == 0) else "cpu"
        cmd = [
            sys.executable,
            "-m",
            "wantrain",
            "worker",
            "--coordinator",
            f"127.0.0.1:{port}",
            "--advertise-host",
            "127.0.0.1",
            "--data-port",
            str(31000 + 16 * i),
            "--name",
            f"w{i}",
            "--workdir",
            wd,
            "--device",
            dev,
            "--region",
            "eu" if i % 2 == 0 else "us",
            "--wan-profile",
            prof,
        ]
        procs[f"w{i}"] = (subprocess.Popen(cmd, env=env, start_new_session=True), wd)
    coord.wait_for_workers(a.workers, 120)

    recipe = JobSpec(
        model=make_config("toy"),
        stages=a.stages,
        steps=a.steps,
        microbatches=8,
        recompute=True,
        wire_codec="planes",
        ckpt_every=2,
    )
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", coord.run_job(Job(spec=recipe))))
    t.start()
    if a.kill_at:
        while (
            coord.run is None
            or not coord.run.states
            or any((st.last_step() or -1) < a.kill_at for st in coord.run.states.values())
        ):
            time.sleep(0.1)
        victim = coord.workers[coord.run.slots[(1, 0)]].info.name
        print(f"\n>>> killing {victim} (agent and stage) at step >= {a.kill_at}\n", flush=True)
        subprocess.run(["pkill", "-9", "-f", procs[victim][1]], check=False)
    t.join()
    res = out["r"]
    print(
        json.dumps(
            {
                "ok": res.ok,
                "reason": res.reason,
                "attempts": res.attempts,
                "heals": res.heals,
                "restarts": res.restarts,
                "placement": res.rationale.get("placement"),
                "partition": res.rationale.get("partition"),
                "hops": [(h["mbps"], h["rtt_ms"]) for h in res.rationale.get("hops", [])],
            },
            indent=2,
            default=str,
        )
    )
    if res.ok and not a.mps:
        ref = ensure_reference(JobSpec.from_json(recipe.to_json()), os.path.join(tmp, "ref"))
        print(
            "vs single-device reference:",
            json.dumps(compare(ref, os.path.join(res.run_dir, "losses.json"))),
        )
    coord.stop()
    time.sleep(0.5)
    for p, wd in procs.values():
        subprocess.run(["pkill", "-9", "-f", wd], check=False)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
