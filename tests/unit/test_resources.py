"""Resource discovery: hardware probe, node record, link measurement."""

# WOULD-FAIL-IF: the probe or link measurement reports wrong fields, duplex is misclassified, an unreachable hop hangs instead of raising, or the spec JSON loses the device map

import json
import math
import os
import threading

import pytest

from wantrain.resources import links, probe
from wantrain.resources.node import CAPACITY_FIELDS, NodeInfo, capacity_source
from wantrain.resources.probe import measure
from wantrain.runtime.spec import JobSpec


def test_probe_measures_this_node():
    # covers: CONTROL-I11
    r = measure("cpu", bench=True, bench_size=256)
    assert r["ok"] and r["device"] == "cpu"
    assert r["tflopsFp32"] > 0 and r["vramTotalBytes"] > 0 and r["cpuCount"] >= 1


def test_node_from_hello_marks_declared_fields():
    # covers: CONTROL-I11
    hello = {
        "advertise_host": "1.2.3.4",
        "data_port": 29000,
        "name": "w",
        "probe": {
            "device": "cuda",
            "gpuName": "RTX 4090",
            "computeCap": [8, 9],
            "vramTotalBytes": 24 << 30,
            "vramFreeBytes": 23 << 30,
            "tflopsFp32": 80.0,
            "tflopsBf16": 160.0,
        },
        "declare": {"region": "eu-west", "uplink_mbps": 500},
    }
    ni = NodeInfo.from_hello("x", hello)
    assert ni.has_bf16() and ni.speed() == 160.0 and ni.memory_bytes() == 23 << 30
    assert ni.region == "eu-west" and ni.declared == ["region", "uplink_mbps"]
    assert ni.cap_int() == 890


def test_duplex_classification():
    # covers: CONTROL-I11
    assert links.classify_duplex(100, 100, 95, 90)[0] == "per_direction"
    assert links.classify_duplex(100, 100, 50, 48)[0] == "aggregate"
    assert links.classify_duplex(100, 100, 1, 1)[0] == "unknown"  # broken bidir test
    assert links.classify_duplex(0, 0, 0, 0)[0] == "unknown"


def test_measure_pair_over_loopback_with_emulated_wan():
    # covers: CONTROL-I11
    stop = threading.Event()
    port = links.serve_echo(0, stop, host="127.0.0.1")
    try:
        emu = links.Emulation("eu", {"intra": {"mbps": 0}, "inter": {"mbps": 40, "rtt_ms": 30}})
        fast = links.measure_pair("127.0.0.1", port, emu.link_to("eu"), bulk_bytes=1 << 20)
        slow = links.measure_pair("127.0.0.1", port, emu.link_to("us"), bulk_bytes=1 << 20)
        assert slow["forward_mbps"] == pytest.approx(40, rel=0.25)
        assert slow["reverse_mbps"] == pytest.approx(40, rel=0.25)
        assert slow["rtt_ms"] >= 29
        assert fast["forward_mbps"] > 4 * slow["forward_mbps"]
        assert slow["duplex"] == "per_direction"
        row = links.measure_row(
            [["127.0.0.1", port, "eu"], ["127.0.0.1", port, "us"]], 0, emu, bulk_bytes=1 << 19
        )
        assert row["bw_mbps"][1] == pytest.approx(40, rel=0.3)
    finally:
        stop.set()


def test_measure_with_parallel_flows_shares_a_shaped_link_and_counts_every_flow():
    # covers: TRANSPORT-I11
    # WOULD-FAIL-IF: each of n flows were paced at the whole emulated rate (an n-fold overestimate of a shaped
    # link), or the aggregate counted one flow's bytes over n flows' time
    stop = threading.Event()
    port = links.serve_echo(0, stop, host="127.0.0.1")
    try:
        emu = links.Emulation("eu", {"intra": {"mbps": 0}, "inter": {"mbps": 80, "rtt_ms": 20}})
        one = links.measure_pair("127.0.0.1", port, emu.link_to("us"), bulk_bytes=1 << 19)
        four = links.measure_pair(
            "127.0.0.1", port, emu.link_to("us"), bulk_bytes=1 << 19, streams=4
        )
        assert one["forward_mbps"] == pytest.approx(80, rel=0.3)
        assert four["forward_mbps"] == pytest.approx(80, rel=0.3)  # the link, not 4x it
        open_ = links.measure_pair(
            "127.0.0.1", port, emu.link_to("eu"), bulk_bytes=1 << 20, streams=8
        )
        assert open_["forward_mbps"] > 4 * four["forward_mbps"]
        row = links.measure_row(
            [["127.0.0.1", port, "eu"], ["127.0.0.1", port, "us"]],
            0,
            emu,
            bulk_bytes=1 << 18,
            streams=4,
        )
        assert row["bw_mbps"][1] == pytest.approx(80, rel=0.35)
    finally:
        stop.set()


def test_unreachable_hop_raises_instead_of_hanging():
    # covers: CONTROL-I11
    s = __import__("socket").socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(ConnectionError, match="does not carry traffic"):
        links.measure_pair("127.0.0.1", port)


def test_spec_json_roundtrip_with_devices():
    # covers: CONTROL-I25
    # WOULD-FAIL-IF: JobSpec.finalize admits a devices map of two device types
    # (CPU + MPS), or a one-type map no longer round-trips
    spec = JobSpec(stages=2, devices=[["mps"], ["mps:0"]], trusted_peers=["abc"])
    back = JobSpec.from_json(spec.to_json()).finalize()
    assert back.device_for(1) == "mps:0" and back.trusted_peers == ["abc"]
    with pytest.raises(AssertionError):
        JobSpec(stages=2, devices=[["cpu"]]).finalize()
    mixed = JobSpec.from_json(JobSpec(stages=2, devices=[["cpu"], ["mps"]]).to_json())
    with pytest.raises(ValueError) as e:
        mixed.finalize()
    assert str(e.value) == (
        "MIXED-FLEET: the device map spans 2 device classes (cpu: devices[0][0]; "
        "mps: devices[1][0]); only one device class is admitted until the heterogeneous "
        "track returns"
    )


# WOULD-FAIL-IF: a capacity is assumed (zero or a default) instead of missing, the PCIe fields appear on a node whose
# device is the host, a malformed HELLO capacity is accepted, or a declared capacity is reported as measured


def _hello(probe_dict, declare=None):
    h = {"advertise_host": "10.0.0.1", "data_port": 29000, "probe": probe_dict}
    if declare is not None:
        h["declare"] = declare
    return h


def test_probe_capacities_cpu_bench_on_and_off(monkeypatch):
    # covers: SCHEDULER-I16
    seen = []
    real = probe._median_copy_gbps

    def spy(dst, src, dev, reps=3):
        seen.append((src.numel() * src.element_size(), str(dev), reps))
        return real(dst, src, dev, reps)

    monkeypatch.setattr(probe, "_median_copy_gbps", spy)
    r = measure("cpu", bench=True, bench_size=256)
    assert r["hostRamBytes"] == r["ramBytes"] > 0
    assert math.isfinite(r["hostDramGBps"]) and r["hostDramGBps"] > 0
    assert seen == [(256 << 20, "cpu", 3)]  # one host-to-host copy of 256 MiB, median of 3
    assert "pcieH2dGBps" not in r and "pcieD2hGBps" not in r  # the device is the host
    off = measure("cpu", bench=False)
    assert off["hostRamBytes"] == off["ramBytes"]
    assert not {"hostDramGBps", "pcieH2dGBps", "pcieD2hGBps"} & off.keys()


def test_probe_capacities_median_after_warmup(monkeypatch):
    # covers: SCHEDULER-I16
    import torch

    class Timed:
        """A copy whose durations are scripted: warm-up 9 s, then 3, 1, 1.5 s."""

        def __init__(self, durations):
            self.durations = durations
            self.clock = 0.0

        def copy_(self, _src):
            self.clock += self.durations.pop(0)

    t = Timed([9.0, 3.0, 1.0, 1.5])
    src = torch.empty(4_000_000_000, dtype=torch.uint8, device="meta")
    monkeypatch.setattr(probe.time, "perf_counter", lambda: t.clock)
    gbps = probe._median_copy_gbps(t, src, torch.device("cpu"))
    # 4e9 bytes / median 1.5 s; the warm-up (9), the mean (1.83), min (1) and max (3) all give other numbers
    assert gbps == pytest.approx(4 / 1.5)
    assert t.durations == []


def test_probe_capacities_one_json_line(capsys):
    # covers: SCHEDULER-I16
    assert probe.main(["--device", "cpu", "--no-bench"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    r = json.loads(lines[0])
    assert r["hostRamBytes"] > 0 and "hostDramGBps" not in r


def test_node_capacities_from_probe_declare_and_older_worker():
    # covers: SCHEDULER-I16
    p = {"device": "cpu", "hostRamBytes": 64 << 30, "hostDramGBps": 41.5}
    ni = NodeInfo.from_hello("a", _hello(p))
    assert (ni.host_ram_bytes, ni.host_dram_gbps) == (64 << 30, 41.5)
    assert ni.pcie_h2d_gbps is None and ni.pcie_d2h_gbps is None and ni.declared == []
    cuda = {
        "device": "cuda",
        "computeCap": [8, 0],
        "hostRamBytes": 1 << 40,
        "hostDramGBps": 120.0,
        "pcieH2dGBps": 24.0,
        "pcieD2hGBps": 22.0,
    }
    decl = {"host_dram_gbps": 77.0, "pcie_d2h_gbps": 13.0}
    nd = NodeInfo.from_hello("b", _hello(cuda, decl))
    assert (nd.host_ram_bytes, nd.host_dram_gbps, nd.pcie_h2d_gbps, nd.pcie_d2h_gbps) == (
        1 << 40,
        77.0,
        24.0,
        13.0,
    )
    assert nd.declared == ["host_dram_gbps", "pcie_d2h_gbps"]
    assert [capacity_source(nd, f) for f in CAPACITY_FIELDS] == [
        "measured",
        "declared",
        "measured",
        "declared",
    ]
    old = NodeInfo.from_hello("c", _hello({"device": "cuda", "ramBytes": 8 << 30}))
    assert all(getattr(old, f) is None for f in CAPACITY_FIELDS)
    d = nd.to_dict()
    assert {f: d[f] for f in CAPACITY_FIELDS} == {
        "host_ram_bytes": 1 << 40,
        "host_dram_gbps": 77.0,
        "pcie_h2d_gbps": 24.0,
        "pcie_d2h_gbps": 13.0,
    }
    back = NodeInfo(**{**d, "compute_cap": tuple(d["compute_cap"])})
    assert back == nd and json.loads(json.dumps(d)) == d


@pytest.mark.parametrize("name", list(CAPACITY_FIELDS))
@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), 10**400, "12", None, True, [1]])
@pytest.mark.parametrize("where", ["probe", "declare"])
def test_node_capacities_reject_malformed(name, bad, where):
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: a malformed capacity (incl. an int past float range, which would raise
    # OverflowError) escapes as anything but a ValueError naming the field.
    p = {"device": "cuda"}
    if where == "probe":
        h = _hello({**p, CAPACITY_FIELDS[name]: bad})
    else:
        h = _hello(p, {name: bad})
    with pytest.raises(ValueError, match=name):
        NodeInfo.from_hello("x", h)


def test_node_capacities_reject_fractional_ram_and_accept_zero():
    # covers: SCHEDULER-I16
    with pytest.raises(ValueError, match="host_ram_bytes"):
        NodeInfo.from_hello("x", _hello({"hostRamBytes": 1.5}))
    ni = NodeInfo.from_hello("x", _hello({"device": "cuda", "pcieH2dGBps": 0, "hostRamBytes": 2.0}))
    assert ni.pcie_h2d_gbps == 0.0 and capacity_source(ni, "pcie_h2d_gbps") == "measured"
    assert ni.host_ram_bytes == 2 and isinstance(ni.host_ram_bytes, int)


def test_capacities_source_cpu_absent_cuda_missing():
    # covers: SCHEDULER-I16
    cpu = NodeInfo.from_hello("c", _hello(measure("cpu", bench=True, bench_size=128)))
    assert capacity_source(cpu, "host_dram_gbps") == "measured"
    assert capacity_source(cpu, "host_ram_bytes") == "measured"
    assert (
        capacity_source(cpu, "pcie_h2d_gbps") == capacity_source(cpu, "pcie_d2h_gbps") == "absent"
    )
    mps = NodeInfo.from_hello("m", _hello({"device": "mps", "hostRamBytes": 1 << 34}))
    assert capacity_source(mps, "pcie_d2h_gbps") == "absent"
    assert capacity_source(mps, "host_dram_gbps") == "missing"  # bench off: missing, never absent
    cuda = NodeInfo.from_hello(
        "g",
        _hello(
            {"device": "cuda", "computeCap": [9, 0], "hostRamBytes": 1 << 40, "hostDramGBps": 200.0}
        ),
    )
    assert (
        capacity_source(cuda, "pcie_h2d_gbps")
        == capacity_source(cuda, "pcie_d2h_gbps")
        == "missing"
    )
    # A declared device is a claim: it cannot make a CUDA node's missing PCIe 'absent'.
    lied = NodeInfo.from_hello("l", _hello({"device": "cuda"}, {"device": "cpu"}))
    assert capacity_source(lied, "pcie_h2d_gbps") == "missing"
    # BA-31 unchanged: the declared gpu_name/device still override the probe and are listed.
    ba31 = NodeInfo.from_hello(
        "d", _hello({"device": "cuda", "gpuName": "A100"}, {"gpu_name": "H100", "device": "cuda"})
    )
    assert (ba31.gpu_name, ba31.device, ba31.declared) == ("H100", "cuda", ["device", "gpu_name"])
    with pytest.raises(ValueError, match="unknown capacity"):
        capacity_source(cuda, "uplink_mbps")


# ---------------------------------------------------------------- M4 agent (I-9, BA-423, BA-445)

_GOOD_GATE = {
    "kind": "GATE",
    "rid": "r1",
    "run_id": "run-20261011-abc123",
    "hf_config": {"model_type": "llama", "hidden_size": 64},
    "recipe": {"compute_dtype": "fp32", "seq_len": 16, "mb_size": 2},
    "devices": ["cpu", "mps"],
}


@pytest.mark.parametrize(
    "field,value,needle",
    [
        ("kind", "PROBE", "kind"),
        ("run_id", "../etc", "run_id"),
        ("run_id", "", "run_id"),
        ("run_id", 7, "run_id"),
        ("hf_config", "llama", "hf_config is a str"),
        ("hf_config", {"pad": "x" * (256 << 10)}, "above 262144"),
        ("recipe", [1], "recipe is a list"),
        ("recipe", {"compute_dtype": "fp32", "seq_len": 16}, "recipe keys"),
        ("recipe", {"compute_dtype": "fp32", "seq_len": 16, "mb_size": 2, "x": 1}, "recipe keys"),
        ("recipe", {"compute_dtype": 3, "seq_len": 16, "mb_size": 2}, "compute_dtype"),
        ("recipe", {"compute_dtype": "fp32", "seq_len": 0, "mb_size": 2}, "seq_len"),
        ("recipe", {"compute_dtype": "fp32", "seq_len": 16, "mb_size": True}, "mb_size"),
        ("devices", [], "non-empty list"),
        ("devices", "cpu", "non-empty list"),
        ("devices", ["cuda:3"], "'cuda:3' is not one of this node's"),
        ("devices", ["cpu", "cpu"], "repeats"),
    ],
)
def test_m4_agent_gate_request_each_malformed_field(field, value, needle):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: validate_gate_request admits a GATE with a wrong kind, a path-like or
    # missing run_id, an hf_config that is not an object or is above 256 KiB, a recipe with
    # missing, extra or ill-typed keys, or a device this node does not hold.
    from wantrain.worker.agent import validate_gate_request

    assert validate_gate_request(_GOOD_GATE, {"cpu", "mps"}) == []
    errs = validate_gate_request({**_GOOD_GATE, field: value}, {"cpu", "mps"})
    assert len(errs) == 1 and needle in errs[0], errs
    assert validate_gate_request(["GATE"], {"cpu"}) == ["request is a list, not an object"]


def test_m4_agent_gate_reply_bound_on_oversized_report(tmp_path):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: an oversized gate report is read into memory or sent past 1 MiB, the
    # reply's fingerprint and device classes are not taken from the reports as I-9 states, or
    # the child's output is read as anything but hfm's {"reports": [one report for the device]}.
    from wantrain.worker.agent import GATE_REPLY_MAX, gate_reply, read_gate_report

    mps = {"ok": True, "device": "mps", "device_class": "mps:Apple M4", "fingerprint": {"n": 2}}
    cpu = {
        "ok": True,
        "device": "cpu",
        "device_class": "cpu:arm64:DEFAULT",
        "fingerprint": {"n": 1},
    }
    r = gate_reply("r9", "run-a", [mps, cpu])
    assert r["kind"] == "GATE_REPORT" and (r["rid"], r["run_id"]) == ("r9", "run-a")
    assert r["fingerprint"] == {"n": 1} and r["reports"] == [mps, cpu]
    assert r["device_classes"] == ["mps:Apple M4", "cpu:arm64:DEFAULT"]
    assert gate_reply(None, "run-a", [mps])["fingerprint"] == {"n": 2}

    big = dict(cpu, detail="x" * GATE_REPLY_MAX)
    with pytest.raises(ValueError, match=r"GATE_REPORT is \d+ bytes, above 1048576"):
        gate_reply("r9", "run-a", [big])
    # Two reports each under the bound whose sum is over it are refused as well.
    half = dict(cpu, detail="y" * (GATE_REPLY_MAX // 2))
    with pytest.raises(ValueError, match="above 1048576"):
        gate_reply("r9", "run-a", [half, dict(half, device="mps")])

    # The child's output is hfm.main's gate shape: {"reports": [one GateReport], ...}.
    def cli_out(*reps):
        return json.dumps(
            {
                "reports": list(reps),
                "fingerprint": reps[0].get("fingerprint") if reps else None,
                "device_classes": [r.get("device_class") for r in reps],
            }
        )

    p = tmp_path / "out.json"
    p.write_text(cli_out(big))
    with pytest.raises(ValueError, match=r"gate report on cpu is \d+ bytes, above 1048576"):
        read_gate_report(str(p), "cpu")
    with pytest.raises(ValueError, match="wrote no report"):
        read_gate_report(str(tmp_path / "missing.json"), "cpu")
    p.write_text("{not json")
    with pytest.raises(ValueError, match="not JSON"):
        read_gate_report(str(p), "cpu")
    p.write_text(json.dumps(mps))  # a bare report is not the CLI's output
    with pytest.raises(ValueError, match="exactly one report"):
        read_gate_report(str(p), "mps")
    p.write_text(cli_out(mps, cpu))
    with pytest.raises(ValueError, match="exactly one report"):
        read_gate_report(str(p), "mps")
    p.write_text(cli_out({"ok": True, "device": "mps"}))
    with pytest.raises(ValueError, match="device_class"):
        read_gate_report(str(p), "mps")
    p.write_text(cli_out(cpu))
    with pytest.raises(ValueError, match="gate report for 'cpu', asked for 'mps'"):
        read_gate_report(str(p), "mps")
    p.write_text(cli_out(mps))
    assert read_gate_report(str(p), "mps") == mps


_CHILD = (
    "import os,sys,time;"
    "p=sys.pycache_prefix;"
    "open(sys.argv[1],'w').write(f'{os.getpid()} {p} {os.path.isdir(p)}');"
    "time.sleep(float(sys.argv[3]));"
    "sys.exit(int(sys.argv[2]))"
)


def _child_args(note, rc, sleep_s):
    import sys

    return lambda prefix: [
        sys.executable,
        "-X",
        f"pycache_prefix={prefix}",
        "-c",
        _CHILD,
        str(note),
        str(rc),
        str(sleep_s),
    ]


def _gone(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    # A zombie still answers kill(pid, 0); run_child reaps its child, so this is a live process.
    return False


@pytest.mark.parametrize("case", ["exit0", "exit1", "timeout"])
def test_m4_agent_run_child_prefix_fresh_and_removed(tmp_path, case):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: run_child reuses a pycache prefix, leaves it on disk after the child
    # exits (cleanly, with an error or on its timeout), or leaves a timed-out child running.
    import os
    import subprocess

    from wantrain.worker.agent import run_child

    root = tmp_path / "pyc"
    note = tmp_path / "note"
    rc, sleep_s, timeout = {"exit0": (0, 0, 60), "exit1": (1, 0, 60), "timeout": (0, 60, 1.5)}[case]
    if case == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            run_child(_child_args(note, rc, sleep_s), dict(os.environ), str(root), timeout)
    else:
        res = run_child(_child_args(note, rc, sleep_s), dict(os.environ), str(root), timeout)
        assert res.returncode == rc
    pid, prefix, existed = note.read_text().split()
    assert existed == "True" and os.path.dirname(prefix) == str(root)
    assert not os.path.exists(prefix) and os.listdir(root) == []
    assert _gone(int(pid))
    # A second child gets its own, fresh prefix.
    run_child(_child_args(note, 0, 0), dict(os.environ), str(root), 60)
    assert note.read_text().split()[1] != prefix and os.listdir(root) == []


def _agent(tmp_path):
    from wantrain.worker.agent import WorkerAgent

    agent = WorkerAgent("127.0.0.1:1", "127.0.0.1", 0, workdir=str(tmp_path / "w"), bench=False)
    agent.probe_info = {"device": "cpu"}
    sent: list = []

    class _Out:  # the coordinator end of the control channel (external boundary)
        def send(self, msg, body=b""):
            sent.append(msg)

    agent.ch = _Out()
    return agent, sent


def test_m4_agent_malformed_gate_answered_with_error(tmp_path):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: a malformed GATE raises out of the agent or starts a gate child instead
    # of being answered with an ERROR that names the reason.
    agent, sent = _agent(tmp_path)
    bad = {**_GOOD_GATE, "devices": ["cuda"], "recipe": {"seq_len": 4}}
    agent._guard(agent._on_gate, bad, b"")
    assert [m["kind"] for m in sent] == ["ERROR"] and sent[0]["rid"] == "r1"
    err = sent[0]["error"]
    assert err.startswith("ValueError: GATE refused:")
    assert "'cuda' is not one of this node's ['cpu']" in err and "recipe keys" in err
    assert not (tmp_path / "w" / "runs").exists()


def test_m4_agent_gate_off_reader_thread_and_abort_kills_child(tmp_path):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: GATE runs on the reader thread (PING waits behind a gate child), or ABORT
    # leaves a gate child running or its pycache prefix on disk.
    import os
    import time

    from wantrain.worker.agent import ChildAborted, run_child

    agent, sent = _agent(tmp_path)
    root = tmp_path / "gate-pyc"
    note = tmp_path / "note"
    outcome: dict = {}

    def fake_gate(msg, body, epoch=None):
        # The real handler's child shape: run_child under the agent's tracker.
        try:
            outcome["res"] = run_child(
                _child_args(note, 0, 60),
                dict(os.environ),
                str(root),
                120,
                tracker=agent._gate_children,
                epoch=epoch,
            )
        except ChildAborted as e:
            outcome["res"] = e

    agent._on_gate = fake_gate
    msgs = [({"kind": "GATE", "rid": "g"}, b""), ({"kind": "PING", "rid": "p"}, b"")]

    class _In:
        def recv(self, file_dir=None):
            if not msgs:
                agent._stop.set()
                return {"kind": "PING", "rid": "last"}, b""
            return msgs.pop(0)

    t0 = time.monotonic()
    agent._serve(_In())
    assert [m["rid"] for m in sent if m["kind"] == "PONG"] == ["p", "last"]
    assert time.monotonic() - t0 < 10  # PING was not held behind the 60 s gate child
    deadline = time.monotonic() + 30
    while not note.exists() or len(note.read_text().split()) < 3:
        assert time.monotonic() < deadline, "gate child never started"
        time.sleep(0.05)
    pid, prefix, _ = note.read_text().split()
    assert os.path.isdir(prefix)
    agent._on_abort({"kind": "ABORT", "rid": "a"}, b"")
    assert sent[-1] == {"kind": "ABORT_ACK", "rid": "a"}
    assert _gone(int(pid)) and not os.path.exists(prefix)
    deadline = time.monotonic() + 30
    while "res" not in outcome:
        assert time.monotonic() < deadline, "gate thread never returned after ABORT"
        time.sleep(0.05)
    assert outcome["res"].returncode == -9 and os.listdir(root) == []
    # A child started for the aborted epoch is killed before it runs.
    with pytest.raises(ChildAborted):
        run_child(
            _child_args(note, 0, 60),
            dict(os.environ),
            str(root),
            120,
            tracker=agent._gate_children,
            epoch=0,
        )
    assert os.listdir(root) == []


def test_m4_agent_stage_process_removes_prefix(tmp_path):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: an hf stage child's pycache prefix outlives the stage process.
    import sys

    from wantrain.worker.agent import StageProcess

    prefix = tmp_path / "pyc" / "pyc-x"
    prefix.mkdir(parents=True)
    st = StageProcess(
        [sys.executable, "-c", "print('WT_DONE')"],
        str(tmp_path / "s.log"),
        None,
        lambda line: None,
        prefix=str(prefix),
    )
    assert st.wait() == 0 and st.done_marker
    assert not prefix.exists() and (tmp_path / "pyc").exists()
    st.close()
    # Without a prefix (every non-hf stage) nothing is removed.
    keep = tmp_path / "keep"
    keep.mkdir()
    st = StageProcess(
        [sys.executable, "-c", "pass"], str(tmp_path / "t.log"), None, lambda line: None
    )
    assert st.wait() == 0 and keep.exists()
    st.close()


def test_a_stage_log_that_cannot_be_written_never_stops_the_pump(tmp_path, capsys):
    # WOULD-FAIL-IF: a failed log write (a full disk, 2026-10-13) ends the pump, so the lines after it, the stage's
    # telemetry, waves and WT_DONE, never reach on_line and the child's stdout fills; or it warns on every line.
    import sys

    from wantrain.worker.agent import StageProcess

    seen = []

    class FullDisk:
        closed = False

        def write(self, raw):
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

        def close(self):
            self.closed = True

    code = (
        "import time\ntime.sleep(0.5)\nfor i in range(200): print('WT_TELEM', i)\nprint('WT_DONE')"
    )
    st = StageProcess([sys.executable, "-c", code], str(tmp_path / "s.log"), None, seen.append)
    st._log.close()
    st._log = FullDisk()  # before the child prints anything
    assert st.wait() == 0
    st._reader.join(timeout=30)
    assert st.done_marker and len(seen) == 201 and seen[-1] == "WT_DONE"
    assert capsys.readouterr().err.count("stage log") == 1
    st.close()


def test_m4_agent_stage_cli_refuses_hf_before_jobspec(tmp_path):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: `wantrain stage` launches (or even loads through JobSpec) a spec whose
    # model arch is 'hf', which would train one slot with no launch gate and no admission.
    import subprocess
    import sys

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model": {"arch": "hf", "hf_config": {"model_type": "llama"}}}))
    code = (
        "import sys\n"
        "from wantrain.cli import main\n"
        "rc = main(['stage', '--spec', sys.argv[1], '--rank', '0'])\n"
        "print('SPEC_IMPORTED', 'wantrain.runtime.spec' in sys.modules)\n"
        "sys.exit(rc)\n"
    )
    p = subprocess.run(
        [sys.executable, "-c", code, str(spec)], capture_output=True, text=True, timeout=120
    )
    assert p.returncode == 2, p.stderr
    assert "refusing model arch 'hf'" in p.stderr and "BA-445" in p.stderr
    assert "SPEC_IMPORTED False" in p.stdout


# ------------------------------------------------- M4 agent-wire (wave 2): real hfm on this Mac's CPU
# No network and no download: every HF model is a tiny config of its pinned config class
# (hfm.tiny_config); from_pretrained, save_pretrained and the hub are never called.


@pytest.fixture
def hf_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from wantrain.model import hfm

    return hfm


_ENV_PROBE = (
    "import json, os, sys\n"
    "p = sys.pycache_prefix\n"
    "print('WT_ENV ' + json.dumps({'env': dict(os.environ), 'prefix': p,\n"
    "      'isdir': bool(p) and os.path.isdir(p), 'xopt': sys._xoptions.get('pycache_prefix')}))\n"
)


_GATE_WRAPPER = (
    "#!{python}\n"
    "import json, os, sys\n"
    "with open({note!r}, 'a') as f:\n"
    "    f.write(json.dumps({{'env': dict(os.environ), 'argv': sys.argv[1:]}}) + '\\n')\n"
    "os.execv({python!r}, [{python!r}, *sys.argv[1:]])\n"
)


def _env_seen(out) -> dict:
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("WT_ENV ")]
    assert len(lines) == 1, (out.stdout, out.stderr)
    return json.loads(lines[0][len("WT_ENV ") :])


def test_m4_agent_wire_hf_child_env_scrubbed_and_prefix_fresh(tmp_path, monkeypatch, hf_offline):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: an hf PROFILE/FIT child or a GATE child inherits LD_PRELOAD, LD_LIBRARY_PATH
    # or a DYLD_* variable, misses an HF_ENV pin, starts without -X pycache_prefix, reuses a prefix
    # or leaves it on disk after exit; or a non-hf child stops starting exactly as before (no
    # prefix, env kept).
    hfm = hf_offline
    gate_cfg = hfm.tiny_config("llama")  # built while the parent is still offline
    (tmp_path / "wt_envprobe.py").write_text(_ENV_PROBE)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("LD_PRELOAD", "/nonexistent/wt-agent-wire.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/nonexistent/wt-agent-wire-lib")
    monkeypatch.setenv("DYLD_WT_AGENT_WIRE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")  # the child's pin wins over the parent's
    agent, sent = _agent(tmp_path)
    root = tmp_path / "w" / "profile" / "pycache"
    hf_body = json.dumps({"model": {"arch": "hf"}}).encode()

    seen = [
        _env_seen(agent._run_slot_tool("profile", "wt_envprobe", ["--rank", "0"], hf_body, 60))
        for _ in range(2)
    ]
    for s in seen:
        env = s["env"]
        assert not {"LD_PRELOAD", "LD_LIBRARY_PATH"} & env.keys()
        assert [k for k in env if k.startswith("DYLD_")] == []
        assert {k: env.get(k) for k in hfm.HF_ENV} == hfm.HF_ENV
        assert env["TRANSFORMERS_OFFLINE"] == "1" and env["USE_HUB_KERNELS"] == "0"
        assert s["xopt"] == s["prefix"] and s["isdir"] is True
        assert os.path.dirname(s["prefix"]) == str(root)
        assert not os.path.exists(s["prefix"])
    assert seen[0]["prefix"] != seen[1]["prefix"] and os.listdir(root) == []

    # A non-hf spec's child starts exactly as before: no prefix, its environment untouched.
    plain = _env_seen(
        agent._run_slot_tool(
            "profile", "wt_envprobe", ["--rank", "0"], b'{"model": {"arch": "builtin"}}', 60
        )
    )
    assert plain["prefix"] is None and plain["xopt"] is None
    assert plain["env"]["LD_PRELOAD"] == "/nonexistent/wt-agent-wire.so"
    assert plain["env"]["DYLD_WT_AGENT_WIRE"] == "1" and plain["env"]["TRANSFORMERS_OFFLINE"] == "0"

    # The GATE child: started through an interpreter wrapper that records what it was handed.
    import sys

    note = tmp_path / "gate-child.jsonl"
    wrapper = tmp_path / "python-wrapper"
    wrapper.write_text(_GATE_WRAPPER.format(python=sys.executable, note=str(note)))
    wrapper.chmod(0o755)
    agent.python = str(wrapper)
    agent._guard(agent._on_gate, _gate_msg(gate_cfg, dtype="fp32"), b"")
    assert sent[-1]["kind"] == "GATE_REPORT", sent[-1]
    (child,) = [json.loads(ln) for ln in note.read_text().splitlines()]
    env, argv = child["env"], child["argv"]
    assert not {"LD_PRELOAD", "LD_LIBRARY_PATH"} & env.keys()
    assert [k for k in env if k.startswith("DYLD_")] == []
    assert {k: env.get(k) for k in hfm.HF_ENV} == hfm.HF_ENV
    assert "WANTRAIN_PEER_KEY" not in env  # the gate never proves the node's key
    gate_root = tmp_path / "w" / "runs" / "run-wire-1" / "gate"
    assert argv[0] == "-X" and argv[1].startswith("pycache_prefix=")
    prefix = argv[1][len("pycache_prefix=") :]
    assert os.path.dirname(prefix) == str(gate_root) and not os.path.exists(prefix)
    assert argv[2:5] == ["-m", "wantrain.model.hfm", "gate"]
    assert os.listdir(gate_root) == []


def _gate_msg(hc, rid="g", run_id="run-wire-1", dtype="bf16"):
    return {
        "kind": "GATE",
        "rid": rid,
        "run_id": run_id,
        "hf_config": hc,
        "recipe": {"compute_dtype": dtype, "seq_len": 16, "mb_size": 2},
        "devices": ["cpu"],
    }


def _torch_tag() -> str:
    import importlib.metadata as md

    from wantrain.model import hfm

    return hfm._platform_tag(md.distribution("torch"))


def _assert_gate_compared(rep: dict, admitted: bool) -> None:
    """A tiny llama's gate ran its whole comparison (BA-463: on every platform); its verdict
    is ok where the platform has a TORCH_RECORDS entry, else the FP-TORCH-RECORDS refusal
    naming the platform tag."""
    from wantrain.model import hfm

    # 4 layers: deepest x {recompute, not}, one stage, 5 single cuts = 8 per M, M in (1, 2),
    # dtypes fp32 and bf16: 32 compared cuts, each named.
    assert rep["dtypes"] == ["fp32", "bf16"] and len(rep["cuts"]) == 32
    assert len(set(rep["cuts"])) == 32 and rep["first_difference"] is None
    assert rep["device"] == "cpu" and rep["device_class"] == hfm.device_class("cpu")
    fp = rep["fingerprint"]
    assert fp["prefix_ok"] is True and fp["code_ok"] is True  # -X pycache_prefix was in force
    assert fp["transformers"]["version"] == hfm.PIN
    if admitted:
        assert (rep["ok"], rep["cause"], rep["detail"]) == (True, None, None)
        assert fp["refusals"] == []
    else:
        assert rep["ok"] is False and rep["cause"] == "FP-TORCH-RECORDS"
        assert _torch_tag() in rep["detail"]
        assert any(r.startswith("FP-TORCH-RECORDS: ") for r in fp["refusals"])


def test_m4_agent_wire_gate_llama_end_to_end_and_dropout_refused(tmp_path, hf_offline):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: the agent's GATE child does not run hfm's real gate on the tiny llama (no
    # comparison, a wrong device class or fingerprint in the reply), its fingerprint is taken
    # without the pycache prefix, a dropout 0.1 config is reported ok or as an ERROR instead of
    # the DROPOUT refusal naming the field, a refused platform is skipped instead of refused
    # (BA-463), or a gate prefix or work directory outlives the GATE.
    hfm = hf_offline
    agent, sent = _agent(tmp_path)
    admitted = hfm.platform_admitted()

    agent._guard(agent._on_gate, _gate_msg(hfm.tiny_config("llama")), b"")
    assert [m["kind"] for m in sent] == ["GATE_REPORT"], sent
    r = sent[0]
    assert (r["rid"], r["run_id"]) == ("g", "run-wire-1") and len(r["reports"]) == 1
    rep = r["reports"][0]
    _assert_gate_compared(rep, admitted)
    assert r["fingerprint"] == rep["fingerprint"]
    assert r["device_classes"] == [hfm.device_class("cpu")]

    agent._guard(
        agent._on_gate,
        _gate_msg(hfm.tiny_config("llama", attention_dropout=0.1), rid="d", run_id="run-wire-2"),
        b"",
    )
    assert [m["kind"] for m in sent] == ["GATE_REPORT", "GATE_REPORT"], sent[-1]
    drop = sent[1]["reports"][0]
    assert (sent[1]["rid"], drop["ok"], drop["cause"]) == ("d", False, "DROPOUT")
    assert drop["detail"] == "attention_dropout=0.1 (the recipe must set it to 0)"
    assert drop["cuts"] == [] and drop["device_class"] == hfm.device_class("cpu")
    for run in ("run-wire-1", "run-wire-2"):
        assert os.listdir(tmp_path / "w" / "runs" / run / "gate") == []


_GATE_UNADMITTED = (
    "import sys\n"
    "from wantrain.model import hfm\n"
    "hfm.TORCH_RECORDS.clear()  # this platform as a Linux host sees it: no entry\n"
    "sys.exit(hfm.main(sys.argv[1:]))\n"
)


def test_m4_agent_wire_gate_unadmitted_platform_refused_strictly(tmp_path, hf_offline):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: on a platform without a TORCH_RECORDS entry (every Linux host, BA-463) the
    # gate skips its comparison, reports ok, or refuses with another cause than FP-TORCH-RECORDS
    # naming the platform tag; this runs the Linux disposition's assertions on every platform.
    import sys

    from wantrain.worker.agent import read_gate_report, run_child

    hfm = hf_offline
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(
        json.dumps(
            {
                "hf_config": hfm.tiny_config("llama"),
                "recipe": {"compute_dtype": "bf16", "seq_len": 16, "mb_size": 2},
                "device": "cpu",
            }
        )
    )
    res = run_child(
        lambda prefix: [
            sys.executable,
            *hfm.interpreter_args(prefix),
            "-c",
            _GATE_UNADMITTED,
            "gate",
            "--in",
            str(inp),
            "--out",
            str(out),
        ],
        hfm.stage_env(dict(os.environ)),
        str(tmp_path / "pyc"),
        300,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    _assert_gate_compared(read_gate_report(str(out), "cpu"), admitted=False)


def _serve_script(agent, script):
    """The coordinator end of the control channel (external boundary): each step returns the
    next message, or waits for a condition on what the agent has sent."""

    class _In:
        def recv(self, file_dir=None):
            if not script:
                agent._stop.set()
                return {"kind": "PING", "rid": "last"}, b""
            return script.pop(0)()

    return _In()


def _wait(cond, what, timeout=240.0):
    import time

    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, what
        time.sleep(0.05)


def test_m4_agent_wire_ping_answered_while_gate_child_runs(tmp_path, hf_offline):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: GATE runs on the reader thread, so a PING sent while the gate child runs is
    # answered only after the GATE_REPORT.
    hfm = hf_offline
    agent, sent = _agent(tmp_path)
    msg = _gate_msg(hfm.tiny_config("llama"))

    def ping_while_child_runs():
        # The PING goes out once the gate child is running (or, on a reader-thread GATE, once
        # the GATE_REPORT has already been sent, which the order below then refuses).
        _wait(
            lambda: agent._gate_children._procs or any(m["kind"] == "GATE_REPORT" for m in sent),
            "the gate child never started",
        )
        return {"kind": "PING", "rid": "during"}, b""

    def gate_report_sent():
        _wait(lambda: any(m["kind"] == "GATE_REPORT" for m in sent), "no GATE_REPORT")
        return {"kind": "PING", "rid": "after"}, b""

    script = [
        lambda: (msg, b""),
        ping_while_child_runs,
        gate_report_sent,
    ]
    agent._serve(_serve_script(agent, script))
    order = [(m["kind"], m.get("rid")) for m in sent]
    assert order == [
        ("PONG", "during"),
        ("GATE_REPORT", "g"),
        ("PONG", "after"),
        ("PONG", "last"),
    ], order
    _assert_gate_compared(sent[1]["reports"][0], hfm.platform_admitted())


def test_m4_agent_wire_abort_during_gate_kills_child_and_prefix(tmp_path, hf_offline):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: an ABORT during a real GATE leaves the gate child running, its pycache
    # prefix or work directory on disk, or the GATE still answers GATE_REPORT; or GATE runs on
    # the reader thread, so the ABORT is read only after the gate finished.
    hfm = hf_offline
    agent, sent = _agent(tmp_path)
    msg = _gate_msg(hfm.tiny_config("llama"), run_id="run-wire-abort")
    held: dict = {}

    def child_running():
        procs = agent._gate_children._procs
        _wait(lambda: bool(procs), "the gate child never started")
        ((proc, prefix),) = list(procs.items())
        assert proc.poll() is None and os.path.isdir(prefix)
        held.update(proc=proc, prefix=prefix)
        return {"kind": "ABORT", "rid": "a"}, b""

    def gate_answered():
        _wait(lambda: any(m.get("rid") == "g" for m in sent), "the GATE thread never answered")
        return {"kind": "PING", "rid": "after"}, b""

    script = [lambda: (msg, b""), child_running, gate_answered]
    agent._serve(_serve_script(agent, script))
    kinds = [(m["kind"], m.get("rid")) for m in sent]
    assert kinds == [("ABORT_ACK", "a"), ("ERROR", "g"), ("PONG", "after"), ("PONG", "last")]
    assert sent[1]["error"] == "ChildAborted: GATE on cpu aborted"
    # Killed by the ABORT (SIGKILL of its group), not waited for until the gate ended.
    assert held["proc"].returncode == -9
    assert _gone(held["proc"].pid) and not os.path.exists(held["prefix"])
    assert os.listdir(tmp_path / "w" / "runs" / "run-wire-abort" / "gate") == []


def _hf_slot_spec(hfm):
    from wantrain.model.gpt import GPTConfig

    hc = hfm.tiny_config("llama")
    m = GPTConfig(
        arch="hf",
        hf_config=hc,
        hf_buffers=hfm.buffer_record(hc),
        hf_census=hfm.unit_census(hc),
        seq_len=16,
        **hfm.config_fields(hc),
    )
    spec = JobSpec(
        model=m, stages=2, steps=1, microbatches=2, mb_size=2, partition=[[0, 2], [2, 6]]
    )
    return spec, [sum(u["params"] for u in m.hf_census[a:b]) for a, b in spec.partition]


def _moe_slot_params(spec, rank: int) -> int:
    """The slot's parameter count, from a meta build (no global RNG draw, no memory)."""
    import torch

    from wantrain.model.gpt import build_layers

    a, b = spec.partition[rank]
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        layers = build_layers(spec.model, spec.init_seed)
    return sum(p.numel() for layer in list(layers)[a:b] for p in layer.parameters())


def _moe_slot_spec():
    from wantrain.model.gpt import GPTConfig

    m = GPTConfig(
        vocab_size=64,
        n_layer=3,
        n_head=2,
        d_model=16,
        seq_len=8,
        moe_experts=4,
        moe_top_k=2,
        moe_d_ff=8,
    )
    return JobSpec(
        model=m, stages=2, steps=1, microbatches=2, mb_size=2, partition=[[0, 2], [2, 5]]
    )


# Runs PROFILE and FIT of one slot in a child and reports, per MoE take, how many records it
# took, and how many records are left on any module afterwards.
_SLOT_PROBE = (
    "import json, sys\n"
    "from wantrain.model import arch, gpt\n"
    "from wantrain.resources import fit, profile\n"
    "from wantrain.runtime.spec import JobSpec\n"
    "built, taken = [], []\n"
    "real_build, real_take = gpt.build_layers, arch.take_moe_records\n"
    "def build(*a, **k):\n"
    "    layers = real_build(*a, **k)\n"
    "    built.append(layers)\n"
    "    return layers\n"
    "def take(layers):\n"
    "    recs = real_take(layers)\n"
    "    taken.append(len(recs))\n"
    "    return recs\n"
    "gpt.build_layers, arch.take_moe_records = build, take\n"
    "path, rank = sys.argv[1], int(sys.argv[2])\n"
    "out = {}\n"
    "for name, run in (\n"
    "    ('profile', lambda: profile.profile_stage(JobSpec.load(path), rank, 0, 2)),\n"
    "    ('fit', lambda: fit.measure(path, rank, force=True)),\n"
    "):\n"
    "    built.clear(); taken.clear()\n"
    "    res = run()\n"
    "    left = sum(len(m.moe_records) for ls in built for m in ls.modules()\n"
    "               if getattr(m, 'moe_records', None) is not None)\n"
    "    out[name] = {'res': res, 'taken': list(taken), 'left': left}\n"
    "print('WT_SLOT ' + json.dumps(out))\n"
)


def _slot_probe(tmp_path, spec_path, rank, hfm=None):
    import subprocess
    import sys

    from wantrain.worker.agent import run_child

    def argv(prefix):
        pre = hfm.interpreter_args(prefix) if hfm else []
        return [sys.executable, *pre, "-c", _SLOT_PROBE, str(spec_path), str(rank)]

    if hfm is not None:  # an hf child: the agent's own way (BA-423)
        res = run_child(argv, hfm.stage_env(dict(os.environ)), str(tmp_path / "pyc"), 600)
    else:
        res = subprocess.run(argv(None), capture_output=True, text=True, timeout=600)
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("WT_SLOT ")]
    assert res.returncode == 0 and len(lines) == 1, res.stderr[-3000:]
    return json.loads(lines[0][len("WT_SLOT ") :])


def _assert_profiled(p: dict, rank: int, layers: int, first: bool, last: bool) -> None:
    assert p["ok"] is True, p
    assert (p["rank"], p["layers"], p["first"], p["last"], p["device"]) == (
        rank,
        layers,
        first,
        last,
        "cpu",
    )
    for k in ("t_f", "t_b", "t_w"):
        assert isinstance(p[k], float) and math.isfinite(p[k]) and p[k] > 0, (k, p)


def _assert_fitted(f: dict, rank: int, layers: int, params: int, head: bool, graphs: int):
    assert f["ok"] is True, f
    assert (f["rank"], f["layers"], f["params"], f["holdsHead"]) == (rank, layers, params, head)
    assert (f["mbSize"], f["concurrentGraphs"], f["virtualStages"]) == (2, graphs, [rank])
    # The CPU has no allocator statistics: the peak is measured on CUDA only (gpu false here).
    assert (f["peakAllocatedBytes"], f["peakReservedBytes"], f["freeBytesBefore"]) == (0, 0, 0)


def test_m4_agent_wire_profile_and_fit_tiny_hf_llama_slot(tmp_path, hf_offline):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: an hf PROFILE or FIT cannot run a tiny llama slot (units not called with the
    # microbatch's ids, a slice of the shared tree, a module out of training mode, the agent's
    # child outside hfm's environment), or reports another slot's layers, parameters or flags.
    from wantrain.resources.fit import peak_pending

    hfm = hf_offline
    agent, sent = _agent(tmp_path)
    spec, params = _hf_slot_spec(hfm)
    path = tmp_path / "spec.json"
    path.write_text(spec.to_json())
    spec.finalize()
    assert params[0] != params[1] and min(params) > 0
    for rank, (layers, first, last) in enumerate([(2, True, False), (4, False, True)]):
        agent._guard(
            agent._on_profile,
            {"kind": "PROFILE", "rid": f"p{rank}", "rank": rank, "reps": 2},
            path.read_bytes(),
        )
        assert sent[-1]["kind"] == "PROFILE_RESULT" and sent[-1]["rid"] == f"p{rank}"
        _assert_profiled(sent[-1]["profile"], rank, layers, first, last)
        got = _slot_probe(tmp_path, path, rank, hfm)
        _assert_profiled(got["profile"]["res"], rank, layers, first, last)
        graphs = peak_pending(spec.schedule_order[rank][0])
        _assert_fitted(got["fit"]["res"], rank, layers, params[rank], last, graphs)
        assert got["profile"]["taken"] == got["fit"]["taken"] == []  # a dense hf model: no MoE
    assert os.listdir(tmp_path / "w" / "profile" / "pycache") == []
    assert os.listdir(tmp_path / "pyc") == []


def test_m4_agent_wire_profile_and_fit_tiny_native_moe_slot(tmp_path):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: a native MoE PROFILE or FIT fails, reports another slot's shape, or leaves
    # its MoE records on the modules so they accumulate across calls (every forward must take
    # exactly one record per MoE block of the slot, and none may be left at the end).
    from wantrain.resources.fit import peak_pending

    agent, sent = _agent(tmp_path)
    spec = _moe_slot_spec()
    path = tmp_path / "spec.json"
    path.write_text(spec.to_json())
    spec.finalize()
    for rank, (layers, moe_blocks, first, last) in enumerate(
        [(2, 1, True, False), (3, 2, False, True)]
    ):
        agent._guard(
            agent._on_profile,
            {"kind": "PROFILE", "rid": f"p{rank}", "rank": rank, "reps": 2},
            path.read_bytes(),
        )
        assert sent[-1]["kind"] == "PROFILE_RESULT" and sent[-1]["rid"] == f"p{rank}"
        _assert_profiled(sent[-1]["profile"], rank, layers, first, last)
        got = _slot_probe(tmp_path, path, rank)
        _assert_profiled(got["profile"]["res"], rank, layers, first, last)
        graphs = peak_pending(spec.schedule_order[rank][0])
        params = _moe_slot_params(spec, rank)
        _assert_fitted(got["fit"]["res"], rank, layers, params, last, graphs)
        # 3 profile passes (a warm-up and 2 reps) and `graphs` FIT forwards, one take each.
        assert got["profile"]["taken"] == [moe_blocks] * 3
        assert got["fit"]["taken"] == [moe_blocks] * graphs
        assert got["profile"]["left"] == got["fit"]["left"] == 0


def test_fit_measures_under_the_runtimes_allocator_and_flags(tmp_path):
    # WOULD-FAIL-IF: the fit runs without setup_determinism, so the CUDA allocator lacks the runtime's expandable
    # segments and the fit charges fragmentation the stage never has (2026-10-12: 85.4 GiB measured against 77.3 GiB
    # under the runtime's allocator refused a 13.5B plan that fits).
    import subprocess
    import sys

    from wantrain.runtime.determinism import pinned_env

    spec = _moe_slot_spec()
    path = tmp_path / "spec.json"
    path.write_text(spec.to_json())
    env = pinned_env(dict(os.environ))
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    code = (
        "import os, sys, torch\n"
        "from wantrain.resources import fit\n"
        "r = fit.measure(sys.argv[1], 0, force=True)\n"
        "print(os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), torch.are_deterministic_algorithms_enabled(), r['ok'])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.split()[-3:] == ["expandable_segments:True", "True", "True"], out.stdout[
        -400:
    ]
