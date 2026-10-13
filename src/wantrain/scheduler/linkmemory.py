"""What earlier runs measured about the wire between specific machines, and
which machines recently proved broken.

Each run's rendezvous writes what it measured and each selection reads it
back, so only the first run over a bad path pays for the discovery; a host's
advertised uplink says nothing about the path to a particular peer. Not a
blocklist: a slow path is ranked down and named, never refused. Dead hosts are
the exception, remembered with a TTL because they fail against every peer.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field

# Below this a two-stage pipeline on any model small enough to want one will
# struggle. Not a cliff; the planner decides admission.
SLOW_PATH_MBPS = 100.0
# Above this the pipeline stalls on round trips regardless of bandwidth.
SLOW_PATH_RTT_MS = 80.0
# Long enough to outlive a bad evening, short enough not to outlive a repair.
DEAD_HOST_TTL_S = 6 * 3600.0


def path_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def subnet24(ip: str | None) -> str | None:
    if not ip:
        return None
    m = re.match(r"^(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}$", ip.strip())
    return m.group(1) if m else None


def shares_subnet(ips) -> str | None:
    """The /24 two machines share while their addresses differ: either one LAN
    (the best link a pipeline can get) or two port-forwards on one gateway
    (peer traffic hairpinning out and back). Reported, not scored."""
    nets = [n for n in (subnet24(i) for i in ips) if n]
    for i in range(len(nets)):
        for j in range(i + 1, len(nets)):
            if nets[i] == nets[j]:
                return nets[i]
    return None


@dataclass
class PathRecord:
    a: str
    b: str
    mbps: float  # slowest direction of the latest reading
    rtt_ms: float
    measured_at: float
    run_id: str
    samples: int = 1  # distinct runs that contributed
    run_ids: list = field(default_factory=list)
    worst_mbps: float = 0.0
    best_mbps: float = 0.0


def verdict(p: PathRecord | None) -> str | None:
    """good: every reading clears the floor; variable: some do and some do not;
    slow: none does, or the latency does not."""
    if p is None:
        return None
    if p.rtt_ms > SLOW_PATH_RTT_MS:
        return "slow"
    if p.best_mbps < SLOW_PATH_MBPS:
        return "slow"
    if p.worst_mbps < SLOW_PATH_MBPS:
        return "variable"
    return "good"


PENALTY = {"good": 0, "variable": 1, "slow": 2}


@dataclass
class DeadHost:
    node_key: str
    host: str | None
    reason: str
    at: float
    run_id: str


class LinkMemory:
    """Path and dead-host memory, persisted as JSON under `state_dir` (or held
    in memory when state_dir is None)."""

    def __init__(self, state_dir: str | None = None, dead_ttl_s: float = DEAD_HOST_TTL_S):
        self.state_dir = state_dir
        self.dead_ttl_s = dead_ttl_s
        self._lock = threading.Lock()
        self._paths: dict[str, PathRecord] = {}
        self._dead: dict[str, DeadHost] = {}
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
            for rec in self._read("link-memory.json"):
                p = PathRecord(**rec)
                self._paths[path_key(p.a, p.b)] = p
            for rec in self._read("dead-hosts.json"):
                d = DeadHost(**rec)
                self._dead[d.node_key] = d

    # ---------------------------------------------------------------- storage
    def _read(self, name):
        path = os.path.join(self.state_dir or "", name)
        try:
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _write(self, name, rows):
        if not self.state_dir:
            return
        path = os.path.join(self.state_dir, name)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=2)
        os.replace(tmp, path)

    # ---------------------------------------------------------------- paths
    def record_path(self, a: str, b: str, mbps: float, rtt_ms: float, run_id: str) -> PathRecord:
        """One reading per run per path: a replayed measurement must not turn
        into several. The latest reading is the headline and the range rides
        alongside: a path that is usually fine and sometimes 8 Mbps is not an
        average path, it is an unreliable one."""
        a, b = sorted((a, b))
        with self._lock:
            key = path_key(a, b)
            prev = self._paths.get(key)
            seen = list(prev.run_ids) if prev else []
            if prev and run_id in seen:
                return prev
            run_ids = (seen + [run_id])[-50:]
            rec = PathRecord(
                a=a,
                b=b,
                mbps=mbps,
                rtt_ms=rtt_ms,
                measured_at=time.time(),
                run_id=run_id,
                samples=len(run_ids),
                run_ids=run_ids,
                worst_mbps=min(prev.worst_mbps, mbps) if prev else mbps,
                best_mbps=max(prev.best_mbps, mbps) if prev else mbps,
            )
            self._paths[key] = rec
            self._write("link-memory.json", [asdict(p) for p in self._paths.values()])
            return rec

    def path(self, a: str, b: str) -> PathRecord | None:
        return self._paths.get(path_key(a, b))

    def paths(self) -> dict[str, PathRecord]:
        return dict(self._paths)

    # ---------------------------------------------------------------- dead hosts
    def record_dead(self, node_key: str, reason: str, run_id: str, host: str | None = None):
        now = time.time()
        with self._lock:
            self._dead = {k: d for k, d in self._dead.items() if now - d.at < self.dead_ttl_s}
            self._dead[node_key] = DeadHost(node_key, host, reason[:300], now, run_id)
            self._write("dead-hosts.json", [asdict(d) for d in self._dead.values()])

    def dead_keys(self) -> set[str]:
        now = time.time()
        return {k for k, d in self._dead.items() if now - d.at < self.dead_ttl_s}
