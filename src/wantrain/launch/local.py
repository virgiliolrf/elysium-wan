"""Local multi-process launcher: one OS process per (rank, peer) slot on this
machine, talking over real TCP sockets, optionally through the WAN shim.

Used for the bitwise regression suite (every engine mechanism must leave the
loss sequence identical to the single-device reference) and for experiments
on one box. Each slot is a separate process, exactly as on a fleet; only the
addresses are local.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, fields

from wantrain.model.gpt import GPTConfig
from wantrain.runtime.spec import JobSpec, LinkCfg
from wantrain.validation.compare import compare

# The hf_* fields GPTConfig had before M4 (3b0d2f8): a dense digest carries
# them as it always did. Every hf_* field added since enters only an hf
# model's digest, and every moe_* field only an MoE model's (BA-465).
_PRE_M4_HF_FIELDS = frozenset({"hf_model", "hf_model_sha256"})
# The GPTConfig fields the builder child reports from the HF config (I-10).
_MIRRORED = ("n_layer", "d_model", "n_head", "vocab_size", "tie_embeddings")
# Bounds of the hfm children (BA-442): the record build and the gate.
HF_CHILD_TIMEOUT_S = 600.0


def _env() -> dict:
    from wantrain.runtime.determinism import pinned_env

    env = pinned_env(dict(os.environ))
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def recipe_model(model: GPTConfig) -> dict:
    """The model dict of the digest (BA-465): asdict(model) without the M4
    hf_* fields of a non-hf model and the moe_* fields of a dense one, so a
    dense digest is byte-identical to the one before those fields existed."""
    d = asdict(model)
    if model.arch != "hf":
        d = {k: v for k, v in d.items() if not k.startswith("hf_") or k in _PRE_M4_HF_FIELDS}
    if not getattr(model, "moe_experts", 0):
        d = {k: v for k, v in d.items() if not k.startswith("moe_")}
    return d


def recipe_digest(spec: JobSpec) -> str:
    """Baseline cache key: a hash of everything that decides numerics, not a
    hand-listed subset. A stale baseline does not fail loudly: it either
    passes (hiding a real change) or looks like a pipeline bug."""
    recipe = {
        "model": recipe_model(spec.model),
        "init_seed": spec.init_seed,
        "data_seed": spec.data_seed,
        "lr": spec.lr,
        "device": spec.device,
        "steps": spec.steps,
        "microbatches": spec.microbatches,
        "mb_size": spec.mb_size,
        "dataset_path": spec.dataset_path,
        "data_shards": spec.data_shards,
    }
    if spec.model.arch == "hf":
        # BA-422: the oracle of an hf run depends on its map (vanilla on one
        # class, the unit chain on a mixed one) and on the pinned code, which
        # rides the model dict (hf_code).
        recipe["devices"] = spec.devices
        recipe["device_classes"] = spec.device_classes
    if spec.dataset_sha256:  # the corpus's content, which the path alone does not name
        recipe["dataset_sha256"] = spec.dataset_sha256
    if spec.optimizer != "adamw":  # RUNTIME-I20: the reference builds the same optimizer
        recipe["optimizer"] = spec.optimizer
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]


def _prefix_args(prefix: str) -> list[str]:
    # hfm.interpreter_args (I-2): in force at interpreter start, before any import.
    return ["-X", f"pycache_prefix={prefix}"]


def fresh_prefix(root: str) -> str:
    """A fresh, empty pycache prefix directory under root (BA-423)."""
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix="pyc-", dir=root)


def remove_prefix(prefix: str) -> None:
    try:
        shutil.rmtree(prefix)
    except FileNotFoundError:
        pass


def run_child(
    args: list[str],
    env: dict,
    prefix_root: str,
    timeout: float | None,
    python: str | None = None,
    **kw,
) -> subprocess.CompletedProcess:
    """BA-423, BA-442: `python -X pycache_prefix=<fresh dir> *args` in env; the
    prefix is removed when the child exits, also on a failure or a timeout
    (subprocess.run kills the child and raises TimeoutExpired)."""
    prefix = fresh_prefix(prefix_root)
    try:
        return subprocess.run(
            [python or sys.executable, *_prefix_args(prefix), *args],
            env=env,
            timeout=timeout,
            **kw,
        )
    finally:
        remove_prefix(prefix)


def _hf_env() -> dict:
    from wantrain.model import hfm

    return hfm.stage_env(_env())


def _hfm_child(spec: JobSpec, cmd: str, payload: dict, python: str | None) -> dict:
    """One `python -X pycache_prefix=<fresh> -m wantrain.model.hfm <cmd>` child
    in the HF environment; -> its output JSON. A failed child is fatal, naming
    the command and its last output."""
    with tempfile.TemporaryDirectory(prefix=f"{spec.tag}_hfm_", dir=spec.out_dir) as d:
        src, dst = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        with open(src, "w") as f:
            json.dump(payload, f)
        res = run_child(
            ["-m", "wantrain.model.hfm", cmd, "--in", src, "--out", dst],
            _hf_env(),
            spec.out_dir,
            HF_CHILD_TIMEOUT_S,
            python,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0 or not os.path.exists(dst):
            tail = (res.stderr or res.stdout or "").strip().splitlines()[-5:]
            raise RuntimeError(f"hfm {cmd} child exited {res.returncode}: {' | '.join(tail)}")
        with open(dst) as f:
            return json.load(f)


def record_path(spec: JobSpec) -> str:
    """The transformers RECORD text kept with the run (BA-448)."""
    return os.path.join(spec.out_dir, f"{spec.tag}_transformers_RECORD")


def _canon(v):
    return json.loads(json.dumps(v))


def _first_tensor_diff(have: dict, want: dict) -> str:
    """The first buffer (by name) whose record entry differs between two buffer
    records, or the record's other fields when every tensor agrees."""
    a = have.get("tensors") if isinstance(have, dict) else None
    b = want.get("tensors") if isinstance(want, dict) else None
    a, b = a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {}
    for name in sorted(set(a) | set(b)):
        if _canon(a.get(name)) != _canon(b.get(name)):
            return f"tensor {name}"
    return "its fields outside the tensors"


def pin_hf_record(model: GPTConfig, out: dict) -> None:
    """Pin the builder child's record, census and hf_code (I-10) and the
    mirrored config fields in the model. A pin already present that differs is
    refused, naming the field (and, for the buffer record, the tensor); so is a
    mirrored field set to something other than its default and the builder's
    value."""
    for name, value in (
        ("hf_buffers", out["record"]),
        ("hf_census", out["census"]),
        ("hf_code", out["hf_code"]),
    ):
        have = getattr(model, name)
        if have is not None and _canon(have) != _canon(value):
            where = f" at {_first_tensor_diff(have, value)}" if name == "hf_buffers" else ""
            raise ValueError(
                f"arch='hf' refused: the spec's {name} differs from the builder's{where}"
            )
        setattr(model, name, value)
    default = {f.name: f.default for f in fields(GPTConfig)}
    for name in _MIRRORED:
        value = out["fields"][name]
        have = getattr(model, name)
        if have != value and have != default[name]:
            raise ValueError(
                f"arch='hf' refused: the spec's {name}={have!r} differs from the HF config's {value!r}"
            )
        setattr(model, name, value)


def gate_devices(spec: JobSpec) -> list[str]:
    """The CPU and every non-CPU device of the map, each gated once."""
    import torch

    rows = spec.devices if spec.devices is not None else [[spec.device]]
    other = sorted({d for row in rows for d in row if torch.device(d).type != "cpu"})
    return ["cpu", *other]


def gate_classes(reports: dict) -> dict:
    """device -> device class from the gate reports; a failed report is
    refused, naming its device and cause."""
    classes = {}
    for device, rep in reports.items():
        if not rep.get("ok"):
            raise ValueError(
                f"arch='hf' refused: the launch gate failed on {device}: "
                f"{rep.get('cause')}: {rep.get('detail')}"
            )
        classes[device] = rep["device_class"]
    return classes


def prepare_hf(spec: JobSpec, python: str | None = None) -> None:
    """BA-424, BA-448, MODEL-I10 for a local arch "hf" run. A first launch builds
    the buffer record, census, hf_code and RECORD text once, in a child, pins
    them and keeps the RECORD text with the run; a resume never rebuilds them
    and refuses an installed transformers whose RECORD differs from the kept
    text, naming the first differing path. Then the gate child runs on the CPU
    and on each non-CPU device of the map, and spec.device_classes is filled."""
    from wantrain.model import hfm

    os.makedirs(spec.out_dir, exist_ok=True)
    m = spec.model
    if spec.resume:
        missing = [n for n in ("hf_buffers", "hf_census", "hf_code") if getattr(m, n) is None]
        if missing:
            raise ValueError(f"arch='hf' resume refused: the spec has no {missing[0]} pin")
        if not os.path.exists(record_path(spec)):
            raise ValueError(
                f"arch='hf' resume refused: no kept RECORD text at {record_path(spec)}"
            )
        with open(record_path(spec)) as f:
            kept = f.read()
        path = hfm.record_diff(kept, hfm.record_text())
        if path is not None:
            raise ValueError(
                f"arch='hf' resume refused: the installed transformers differs from the run's "
                f"RECORD at {path}"
            )
    else:
        out = _hfm_child(spec, "record", {"hf_config": m.hf_config, "seed": spec.init_seed}, python)
        pin_hf_record(m, out)
        text = gzip.decompress(base64.b64decode(out["record_text_gz_b64"])).decode()
        with open(record_path(spec), "w") as f:
            f.write(text)
    spec.finalize()
    recipe = {"compute_dtype": m.compute_dtype, "seq_len": m.seq_len, "mb_size": spec.mb_size}
    reports = {}
    for device in gate_devices(spec):
        out = _hfm_child(
            spec,
            "gate",
            {"hf_config": m.hf_config, "recipe": recipe, "device": device, "devices": [device]},
            python,
        )
        # One GateReport, or a GATE_REPORT-shaped {"reports": [...]}; any
        # failed report of the device decides.
        reps = out.get("reports") or [out]
        reports[device] = next((r for r in reps if not r.get("ok")), reps[0])
    spec.device_classes = slot_classes(spec, gate_classes(reports))


def slot_classes(spec: JobSpec, classes: dict) -> list:
    """device_classes[stage][peer] from the gated devices' classes: every CPU
    slot has the CPU's class."""
    import torch

    def cls(d: str) -> str:
        return classes["cpu" if torch.device(d).type == "cpu" else d]

    return [[cls(spec.device_for(r, p)) for p in range(w)] for r, w in enumerate(spec.stage_widths)]


def ensure_reference(spec: JobSpec, out_dir: str, python: str | None = None) -> str:
    """Run (or reuse) the single-device reference for this recipe."""
    tag = f"reference_{spec.device}_{recipe_digest(spec)}"
    path = os.path.join(out_dir, f"{tag}_losses.json")
    if os.path.exists(path) and os.path.exists(reference_state_path(path)):
        return path
    os.makedirs(out_dir, exist_ok=True)
    spec_path = os.path.join(out_dir, f"{tag}_spec.json")
    with open(spec_path, "w") as f:
        f.write(spec.to_json())
    args = [
        "-m",
        "wantrain.validation.reference",
        "--spec",
        spec_path,
        "--tag",
        tag,
        "--out-dir",
        out_dir,
    ]
    if spec.model.arch == "hf":
        # BA-423: the reference runs the units' code in the stage's environment.
        res = run_child(args, _hf_env(), out_dir, None, python, stdout=subprocess.DEVNULL)
        if res.returncode != 0:
            raise subprocess.CalledProcessError(res.returncode, res.args)
        return path
    subprocess.run(
        [python or sys.executable, *args],
        check=True,
        env=_env(),
        stdout=subprocess.DEVNULL,
    )
    return path


def reference_state_path(losses_path: str) -> str:
    """The reference's canonical state manifest (validation/state.py)."""
    return losses_path[: -len("_losses.json")] + "_state.json"


def device_name(device: str) -> str:
    """The model of the device a slot resolves to: the torch device name on
    CUDA (real-hardware pending under gpu false), the device type elsewhere
    (one host's CPU is one class)."""
    import torch

    d = torch.device(device)
    if d.type == "cuda":
        return torch.cuda.get_device_name(d)
    return d.type


def check_pieces_class(spec: JobSpec) -> None:
    """BA-52: under shard "pieces" every slot of a stage computes the same
    bits only on one device model; refuse a stage whose slots resolve to
    devices with different names, naming the stage."""
    if spec.shard != "pieces":
        return
    for r, w in enumerate(spec.stage_widths):
        names = {spec.device_for(r, p): device_name(spec.device_for(r, p)) for p in range(w)}
        if len(set(names.values())) > 1:
            raise ValueError(
                f"shard='pieces' refused: stage {r} has slots on different device models {names}"
            )


def make_budget_files(spec: JobSpec) -> list:
    """BA-55: for pacing "shared", the shared-pacer budget files of the run,
    fresh under out_dir: one per peer egress, per peer ingress and per site
    pair direction. Pacing "link" creates none. -> the paths."""
    import shutil

    from wantrain.transport.shim import SharedBudget

    if spec.pacing != "shared":
        return []
    # The tree below is removed before it is made fresh: a tag that is not one
    # plain name ("", ".", "..", a path) would aim that removal at out_dir or above.
    tag = spec.tag
    if not isinstance(tag, str) or tag in ("", ".", "..") or os.path.basename(tag) != tag:
        raise ValueError(f"pacing='shared' needs a plain tag for its budget files, got {tag!r}")
    paths = spec.budget_paths()
    root = os.path.dirname(next(iter(paths.values())))
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for (kind, a, b), path in paths.items():
        if kind == "uplink":
            mbps = (spec.uplink_mbps or {})[f"{a}->{b}"]
        else:
            grid = spec.peer_egress_mbps if kind == "egress" else spec.peer_ingress_mbps
            mbps = (grid or [])[a][b]
        SharedBudget(float(mbps), path).close()
    return list(paths.values())


def jitter_seed_from_env() -> int:
    """The shim jitter seed of an E-arm (BA-22): WANTRAIN_E1_SEED when set,
    else a fresh one from os.urandom, so repeated runs see distinct arrival
    streams. Printed, so a failing run is reproducible from its log."""
    raw = os.environ.get("WANTRAIN_E1_SEED")
    seed = int(raw) if raw else int.from_bytes(os.urandom(4), "little")
    print(f"E1_SEED={seed}", flush=True)
    return seed


def run_local(
    spec: JobSpec,
    timeout: float = 600.0,
    python: str | None = None,
    log_dir: str | None = None,
    prepared: bool = False,
) -> dict:
    """Launch every slot of `spec` locally and wait. -> {ok, rc, wall_seconds}.
    An arch "hf" spec is prepared first (prepare_hf) unless `prepared` says
    run_and_compare already did, and every stage starts with a fresh pycache
    prefix in the HF environment."""
    hf = spec.model.arch == "hf"
    if hf and not prepared:
        prepare_hf(spec, python)
    spec.finalize()
    check_pieces_class(spec)
    os.makedirs(spec.out_dir, exist_ok=True)
    make_budget_files(spec)
    spec_path = os.path.join(spec.out_dir, f"{spec.tag}_spec.json")
    with open(spec_path, "w") as f:
        f.write(spec.to_json())
    log_dir = log_dir or spec.out_dir
    env = _hf_env() if hf else _env()
    procs, logs, prefixes = [], [], []
    t0 = time.time()
    rc = 1  # until every slot is waited, an exception (abort included) kills them all
    try:
        # Spawned inside the try: a launch that fails part-way (a Popen or a
        # prefix error) still kills the stages already started and removes
        # every prefix (BA-423).
        for r in range(spec.stages):
            for p in range(spec.stage_widths[r]):
                log = open(os.path.join(log_dir, f"{spec.tag}_r{r}p{p}.log"), "wb")
                logs.append(log)
                interp = []
                if hf:
                    prefixes.append(fresh_prefix(spec.out_dir))
                    interp = _prefix_args(prefixes[-1])
                procs.append(
                    subprocess.Popen(
                        [
                            python or sys.executable,
                            *interp,
                            "-m",
                            "wantrain.worker.stage_main",
                            "--spec",
                            spec_path,
                            "--rank",
                            str(r),
                            "--peer",
                            str(p),
                            "--launcher",
                            "local",
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
        res = 0
        for proc in procs:
            left = max(1.0, timeout - (time.time() - t0))
            try:
                code = proc.wait(timeout=left)
            except subprocess.TimeoutExpired:
                res = 2
                break
            if code != 0:
                res = 3
                break
        rc = res
    finally:
        if rc:
            for proc in procs:
                proc.kill()
        for proc in procs:
            proc.wait()
        for log in logs:
            log.close()
        for prefix in prefixes:
            remove_prefix(prefix)
    return {"ok": rc == 0, "rc": rc, "wall_seconds": round(time.time() - t0, 3)}


def run_and_compare(spec: JobSpec, timeout: float = 600.0, python: str | None = None) -> dict:
    """Distributed run vs the single-device reference: ok iff bit-identical."""
    hf = spec.model.arch == "hf"
    if hf:
        # The reference needs the pins and the device classes the launch fills.
        prepare_hf(spec, python)
    ref = ensure_reference(spec, spec.out_dir, python)
    res = run_local(spec, timeout, python, prepared=hf)
    if not res["ok"]:
        return dict(res, compared=0)
    cmp = compare(ref, os.path.join(spec.out_dir, f"{spec.tag}_losses.json"))
    return dict(cmp, wall_seconds=res["wall_seconds"])


def links_for(spec: JobSpec, latency_ms=0.0, bw_mbps=0.0, jitter_ms=0.0, jitter_seed=7):
    link = asdict(
        LinkCfg(
            latency_ms=latency_ms, bw_mbps=bw_mbps, jitter_ms=jitter_ms, jitter_seed=jitter_seed
        )
    )
    return [dict(link) for _ in range(spec.stages)]
