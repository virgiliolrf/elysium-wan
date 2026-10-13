"""A real local cluster for end-to-end tests: the coordinator in-process, every
worker agent a separate OS process (and every stage a child of its agent),
all talking over TCP exactly as on a fleet. Only the addresses are local.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

from wantrain.coordinator.coordinator import Coordinator, CoordinatorConfig, Job
from wantrain.scheduler.selection import SelectionPolicy

SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"
)

FAST_FT = dict(
    heartbeat_timeout_s=6.0,
    reconnect_grace_s=4.0,
    strike_gap_s=2.0,
    stall_floor_s=8.0,
    stall_steps=6,
    first_step_timeout_s=90.0,
    poll_s=0.2,
    dead_host_ttl_s=3600.0,
)


class LocalCluster:
    def __init__(self, tmp, port_base: int, **cfg):
        self.tmp = str(tmp)
        self.port_base = port_base
        self.next_port = port_base
        sel = cfg.pop("selection", None) or SelectionPolicy(distinct_hosts=False)
        conf = dict(FAST_FT)
        conf.update(cfg)
        self.cfg = CoordinatorConfig(
            state_dir=os.path.join(self.tmp, "coord"), selection=sel, **conf
        )
        self.coord = Coordinator("127.0.0.1", 0, self.cfg)
        self.port = self.coord.start()
        self.workers: dict[str, subprocess.Popen] = {}
        # Every agent ever launched, a replaced name's included: close() reaps them all.
        self._procs: list[subprocess.Popen] = []
        self.workdirs: dict[str, str] = {}
        self.result = None
        self._thread = None

    def add_worker(
        self,
        name: str,
        device: str = "cpu",
        declare: dict | None = None,
        region: str | None = None,
        wan_profile: dict | None = None,
    ) -> None:
        port = self.next_port
        self.next_port += 8
        workdir = os.path.join(self.tmp, f"w-{name}")
        os.makedirs(workdir, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "wantrain",
            "worker",
            "--coordinator",
            f"127.0.0.1:{self.port}",
            "--advertise-host",
            "127.0.0.1",
            "--data-port",
            str(port),
            "--name",
            name,
            "--workdir",
            workdir,
            "--device",
            device,
        ]
        if declare:
            cmd += ["--declare"] + [f"{k}={json.dumps(v)}" for k, v in declare.items()]
        if region:
            cmd += ["--region", region]
        if wan_profile:
            p = os.path.join(workdir, "wan.json")
            with open(p, "w") as f:
                json.dump(wan_profile, f)
            cmd += ["--wan-profile", p]
        env = dict(os.environ, PYTHONPATH=SRC, PYTHONUNBUFFERED="1")
        # The child holds its own descriptor of the log; ours closes at once.
        with open(os.path.join(workdir, "agent.log"), "wb") as log:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        self.workers[name] = proc
        self._procs.append(proc)
        self.workdirs[name] = workdir

    def wait_registered(self, n: int, timeout: float = 120.0) -> None:
        self.coord.wait_for_workers(n, timeout)

    def run_async(self, job: Job) -> None:
        def go():
            self.result = self.coord.run_job(job, wait_workers_s=120)

        self._thread = threading.Thread(target=go, daemon=True)
        self._thread.start()

    def join(self, timeout: float = 600.0):
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "run did not finish in time"
        return self.result

    def run(self, job: Job, timeout: float = 600.0):
        self.run_async(job)
        return self.join(timeout)

    # ------------------------------------------------------------ observation
    def slot_of(self, name: str):
        run = self.coord.run
        if run is None:
            return None
        for slot, nid in run.slots.items():
            if self.coord.workers[nid].info.name == name:
                return slot
        return None

    def name_of(self, slot) -> str:
        return self.coord.workers[self.coord.run.slots[slot]].info.name

    def wait_step(self, step: int, timeout: float = 180.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            run = self.coord.run
            if run is not None and run.states:
                done = [st.last_step() for st in run.states.values()]
                if all(s is not None and s >= step for s in done):
                    return
            time.sleep(0.1)
        raise TimeoutError(f"fleet did not reach step {step}")

    # ------------------------------------------------------------ faults
    def kill_worker(self, name: str) -> None:
        """The machine dies: agent and stage, SIGKILL, no goodbye."""
        subprocess.run(["pkill", "-9", "-f", self.workdirs[name]], check=False)
        try:
            os.killpg(self.workers[name].pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def kill_stage(self, name: str) -> None:
        """Only the training process dies (OOM killer, container restart)."""
        subprocess.run(["pkill", "-9", "-f", f"stage_main.*{self.workdirs[name]}"], check=False)

    def freeze_stage(self, name: str) -> None:
        """The training process stops making progress but stays alive."""
        subprocess.run(["pkill", "-STOP", "-f", f"stage_main.*{self.workdirs[name]}"], check=False)

    def close(self) -> None:
        try:
            self.coord.stop(shutdown_workers=True)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
        for name in self.workers:
            subprocess.run(["pkill", "-9", "-f", self.workdirs[name]], check=False)
            try:
                os.killpg(self.workers[name].pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
