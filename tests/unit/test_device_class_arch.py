"""CONTROL-I25: the operator's architecture grouping of CUDA device classes (WANTRAIN_DEVICE_CLASS=arch)."""

import torch

from wantrain.resources import probe


def _fake_cuda(monkeypatch, name, cap):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev=None: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev=None: cap)


def test_cuda_class_is_the_device_name_unless_the_operator_groups_by_architecture(monkeypatch):
    # covers: CONTROL-I25
    # WOULD-FAIL-IF: the default CUDA class is not the device name (two GPU models of one architecture would pass as
    # one class without the operator asking), or WANTRAIN_DEVICE_CLASS=arch does not give one class per compute
    # capability (an RTX 3060 and an RTX 3070 Laptop GPU, both sm_86, stay MIXED-FLEET), or it merges architectures.
    monkeypatch.delenv("WANTRAIN_DEVICE_CLASS", raising=False)
    _fake_cuda(monkeypatch, "NVIDIA GeForce RTX 3060", (8, 6))
    a = probe.device_class("cuda")
    _fake_cuda(monkeypatch, "NVIDIA GeForce RTX 3070 Laptop GPU", (8, 6))
    b = probe.device_class("cuda")
    assert (a, b) == ("cuda:NVIDIA GeForce RTX 3060", "cuda:NVIDIA GeForce RTX 3070 Laptop GPU")

    monkeypatch.setenv("WANTRAIN_DEVICE_CLASS", "arch")
    assert probe.device_class("cuda") == "cuda:sm_86"
    _fake_cuda(monkeypatch, "NVIDIA GeForce RTX 3060", (8, 6))
    assert probe.device_class("cuda") == "cuda:sm_86"
    _fake_cuda(monkeypatch, "NVIDIA GeForce RTX 4090", (8, 9))
    assert probe.device_class("cuda") == "cuda:sm_89", "another architecture is another class"
    assert probe.device_class("cpu").startswith("cpu:"), "only CUDA classes are grouped"


def test_a_worker_behind_a_port_map_binds_its_probe_where_told_and_advertises_the_public_port():
    # covers: CONTROL-I25
    # WOULD-FAIL-IF: a worker behind a container port map (bind 8080, public 28791) advertises the bound port, so the
    # coordinator's link probe dials a port nobody forwards, or a plain worker stops advertising the port it bound.
    from wantrain.worker.agent import probe_bind

    bound = []

    def serve(p):
        bound.append(p)
        return p or 41234

    assert probe_bind(28791, 8080, serve) == 28791 and bound == [8080]
    assert probe_bind(0, 0, serve) == 41234, "port 0 advertises the port the server got"
    assert probe_bind(5000, 0, serve) == 5000
    assert probe_bind(-1, 8080, serve) == -1 and bound == [8080, 0, 5000], "-1 = no probe server"


def test_a_heal_under_architecture_classes_keeps_the_dead_slots_gpu_model():
    # covers: CONTROL-I25
    # WOULD-FAIL-IF: under WANTRAIN_DEVICE_CLASS=arch a heal takes a spare of another GPU model of the same
    # architecture (the device map, and so the run's bits, would change with no refusal), or a name-class run's
    # spares are filtered by model, or admission (no gpu_name) stops grouping by architecture.
    from types import SimpleNamespace

    from wantrain.coordinator.coordinator import Coordinator

    def w(cls, gpu):
        return SimpleNamespace(info=SimpleNamespace(device_class=cls, gpu_name=gpu))

    pool = [
        w("cuda:sm_86", "NVIDIA GeForce RTX 3060"),
        w("cuda:sm_86", "NVIDIA GeForce RTX 3070 Laptop GPU"),
    ]
    c = Coordinator.__new__(Coordinator)
    c.idle_workers = lambda: pool
    names = lambda ws: [x.info.gpu_name for x in ws]  # noqa: E731
    assert names(c._class_spares("cuda:sm_86", "NVIDIA GeForce RTX 3070 Laptop GPU")) == [
        "NVIDIA GeForce RTX 3070 Laptop GPU"
    ]
    assert len(c._class_spares("cuda:sm_86")) == 2, "admission groups the architecture"
    named = [w("cuda:NVIDIA GeForce RTX 3060", "NVIDIA GeForce RTX 3060")]
    c.idle_workers = lambda: named
    assert len(c._class_spares("cuda:NVIDIA GeForce RTX 3060", "NVIDIA GeForce RTX 3060")) == 1
