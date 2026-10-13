"""What the scheduler knows about one worker node.

Built from the worker's HELLO: measured capabilities (resources/probe.py) plus
the addresses peers dial, plus whatever the operator declared about the node
that cannot be measured from inside it (its region, its advertised uplink).
Declared values are recorded as declared so a plan can say which of its inputs
were measurements and which were claims.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

# Host capacities SCHEDULER-I16 admits on: NodeInfo field -> probe key. Each is
# None when not measured; a planner never assumes one.
CAPACITY_FIELDS = {
    "host_ram_bytes": "hostRamBytes",
    "host_dram_gbps": "hostDramGBps",
    "pcie_h2d_gbps": "pcieH2dGBps",
    "pcie_d2h_gbps": "pcieD2hGBps",
}
_PCIE_FIELDS = ("pcie_h2d_gbps", "pcie_d2h_gbps")


def _capacity(name: str, value) -> float | int:
    """A capacity from the HELLO, validated at the boundary: a finite
    non-negative number (a byte count is also integral), else ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: not a number ({value!r})")
    try:
        as_float = float(value)
    except OverflowError:  # an int beyond float range (JSON allows any size)
        raise ValueError(f"{name}: not a finite non-negative number (out of range)") from None
    if not math.isfinite(as_float) or as_float < 0:
        raise ValueError(f"{name}: not a finite non-negative number ({value!r})")
    if name == "host_ram_bytes":
        if value != int(value):
            raise ValueError(f"{name}: not a whole number of bytes ({value!r})")
        return int(value)
    return as_float


@dataclass
class NodeInfo:
    node_id: str
    name: str
    host: str  # address peers dial for the data plane
    data_port: int  # port peers dial
    listen_port: int = 0  # port the stage binds (NAT/port maps); 0 = data_port
    probe_port: int = 0  # link-probe echo server; 0 = none
    device: str = "cpu"  # cuda | mps | cpu
    gpu_name: str = "cpu"
    gpu_count: int = 0
    compute_cap: tuple | None = None
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    tflops_fp32: float = 0.0
    tflops_bf16: float | None = None
    torch_version: str = ""
    cuda_version: str | None = None
    hostname: str = ""
    region: str | None = None  # declared: there is no geolocation service here
    uplink_mbps: float | None = None
    downlink_mbps: float | None = None
    labels: dict = field(default_factory=dict)
    declared: list = field(default_factory=list)
    host_ram_bytes: int | None = None
    host_dram_gbps: float | None = None
    pcie_h2d_gbps: float | None = None
    pcie_d2h_gbps: float | None = None
    # CONTROL-I25: probe.device_class of `device`, from the HELLO; "" = none.
    device_class: str = ""

    @property
    def bind_port(self) -> int:
        return self.listen_port or self.data_port

    def speed(self) -> float:
        """Measured throughput the partition is cut against. bf16 when the
        silicon has real bf16, else fp32; never zero (the DP divides by it)."""
        best = self.tflops_bf16 if self.tflops_bf16 else self.tflops_fp32
        return max(1e-3, float(best or 0.0))

    def memory_bytes(self) -> int:
        """Budget for one stage: free device memory, or total when unknown."""
        return int(self.vram_free_bytes or self.vram_total_bytes or 0)

    def has_bf16(self) -> bool:
        if self.device == "cuda":
            return self.compute_cap is not None and tuple(self.compute_cap) >= (8, 0)
        return False

    def cap_int(self) -> int:
        """Compute capability as an integer (sm_86 -> 860), 0 when not CUDA."""
        if not self.compute_cap:
            return 0
        return int(self.compute_cap[0]) * 100 + int(self.compute_cap[1]) * 10

    def to_dict(self) -> dict:
        d = asdict(self)
        d["compute_cap"] = list(self.compute_cap) if self.compute_cap else None
        return d

    @staticmethod
    def from_hello(node_id: str, hello: dict) -> NodeInfo:
        probe = hello.get("probe") or {}
        decl = hello.get("declare") or {}
        cap = decl.get("compute_cap", probe.get("computeCap"))
        caps: dict[str, Any] = {}
        dclass = probe.get("deviceClass", "")
        if not isinstance(dclass, str):
            raise ValueError(f"device_class: not a string ({dclass!r})")
        for name, key in CAPACITY_FIELDS.items():
            if key in probe:
                caps[name] = _capacity(name, probe[key])
            if name in decl:
                caps[name] = _capacity(name, decl[name])
        info = NodeInfo(
            node_id=node_id,
            name=hello.get("name") or probe.get("hostname") or node_id,
            host=hello["advertise_host"],
            data_port=int(hello["data_port"]),
            listen_port=int(hello.get("listen_port") or 0),
            probe_port=int(hello.get("probe_port") or 0),
            device=decl.get("device", probe.get("device", "cpu")),
            gpu_name=decl.get("gpu_name", probe.get("gpuName", "cpu")),
            gpu_count=int(probe.get("gpuCount") or 0),
            compute_cap=tuple(cap) if cap else None,
            vram_total_bytes=int(decl.get("vram_bytes", probe.get("vramTotalBytes") or 0)),
            vram_free_bytes=int(decl.get("vram_bytes", probe.get("vramFreeBytes") or 0)),
            tflops_fp32=float(decl.get("tflops", probe.get("tflopsFp32") or 0.0)),
            tflops_bf16=(
                float(decl["tflops"])
                if "tflops" in decl and cap and tuple(cap) >= (8, 0)
                else probe.get("tflopsBf16")
            ),
            torch_version=probe.get("torch", ""),
            cuda_version=probe.get("cudaBuild"),
            hostname=probe.get("hostname", ""),
            region=decl.get("region"),
            uplink_mbps=decl.get("uplink_mbps"),
            downlink_mbps=decl.get("downlink_mbps"),
            labels=dict(decl.get("labels") or {}),
            declared=sorted(decl.keys()),
            device_class=dclass,
            **caps,
        )
        return info


MIXED_FLEET_TAIL = "only one device class is admitted until the heterogeneous track returns"


def mixed_fleet(members) -> str | None:
    """CONTROL-I25 over a device map's (name, device class) pairs: the
    MIXED-FLEET refusal of a member reporting no class, or of a map spanning
    more than one class, else None."""
    by_class: dict[str, list[str]] = {}
    for name, cls in members:
        if not cls:
            return f"MIXED-FLEET: node {name} reports no device class"
        names = by_class.setdefault(cls, [])
        if name not in names:
            names.append(name)
    if len(by_class) <= 1:
        return None
    spans = "; ".join(f"{c}: {', '.join(ns)}" for c, ns in sorted(by_class.items()))
    return (
        f"MIXED-FLEET: the device map spans {len(by_class)} device classes ({spans}); "
        + MIXED_FLEET_TAIL
    )


def class_counts(nodes) -> str:
    """'<class> x<n>, ...' of NodeInfos, sorted; a node with no class counts
    under '<none>' (the pool a MIXED-FLEET refusal names)."""
    counts: dict[str, int] = {}
    for n in nodes:
        key = n.device_class or "<none>"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{c} x{k}" for c, k in sorted(counts.items())) or "empty"


def capacity_source(info: NodeInfo, name: str) -> str:
    """Where a capacity came from: 'measured', 'declared', 'missing' or 'absent'.

    'absent' is only a PCIe field on a cpu or mps node, whose device is the host
    (no PCIe hop). It needs the device class from the probe: a declared device is
    a claim, so its missing PCIe stays 'missing'. 'missing' is any other capacity
    that should have been measured and was not (bench off, an older worker, a
    CUDA node without PCIe figures).
    """
    if name not in CAPACITY_FIELDS:
        raise ValueError(f"unknown capacity field {name!r}")
    if name in info.declared:
        return "declared"
    if getattr(info, name) is not None:
        return "measured"
    if name in _PCIE_FIELDS and info.device in ("cpu", "mps") and "device" not in info.declared:
        return "absent"
    return "missing"
