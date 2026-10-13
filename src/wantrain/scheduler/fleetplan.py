"""Fleet description -> JobSpec: the bridge between discovery and the planner.

The coordinator (or an operator with `wantrain plan --fleet fleet.json`) knows
the fleet; the planner knows how to turn it into a spec. One place plans and
the same spec bytes go to every rank: stages that disagree about the layer cut
silently train different models.

fleet = {
  "run_id": str, "peer_port": int,
  "nodes": [{"rank", "host", "dial_port", "listen_port", "tflops",
             "vram_bytes", "compute_cap", "label", "device",
             # plan mode for wide stages (SCHEDULER-I16, BA-27, BA-48), optional:
             "hostname", "site", "gpu_name", "declared", "host_ram_bytes",
             "host_dram_gbps", "pcie_h2d_gbps", "pcie_d2h_gbps" (GB/s),
             "nic_up_gbps", "nic_down_gbps" (Gbit/s), "hbm_gbps" (GB/s),
             "sources": {key: measured | declared | absent | missing}}],
  "link":  {"mbps", "rtt_ms", "duplex"},       # the bottleneck hop, measured
  "k_passes": {"value", "source"},             # host-DRAM passes per side
  "job":   {"model": preset, <GPTConfig overrides>, "dataset", "dataset_source", "policy",
            "margin", "k_inflight", "steps", "lr", "boundary_dtype",
            "compute_dtype", "mb_size", "microbatches", "data_shards",
            "recompute", "attn_impl", "require_margin", "device",
            "critical_batch_tokens", "out_dir", "ckpt_every", "tag", "shard",
            # arch "hf" and MoE (HANDOUT-M4 I-14): "hf_config", "hf_impl",
            # every moe_* field; never "hf_buffers", "hf_census", "hf_code"}
}
"""

from __future__ import annotations

import hashlib
import json
import os

from wantrain.model.arch import MOE_FIELDS
from wantrain.model.gpt import GPTConfig, make_config
from wantrain.scheduler import planner as prov

MODEL_KEYS = (
    "vocab_size",
    "n_layer",
    "n_head",
    "d_model",
    "seq_len",
    "compute_dtype",
    "boundary_dtype",
    "attn_impl",
    "sdpa_backend",
    "arch",
    "hf_model",
    "hf_model_sha256",
    "user_model",
    "user_model_sha256",
    "lora_rank",
    "lora_alpha",
    "lora_targets",
    "n_kv_head",
    "d_ff",
    "norm",
    "mlp",
    "pos",
    "bias",
    "tie_embeddings",
    "init_scheme",
    "init_std",
    "hf_config",
    "hf_impl",
    *MOE_FIELDS,
)

# The builder child's pins (HANDOUT-M4 I-10, I-14): taken from hf_pins, never
# from the plan dict, which a caller writes.
HF_PIN_KEYS = ("hf_buffers", "hf_census", "hf_code")
# The GPTConfig fields the builder mirrors from the HF config (hfm.config_fields).
HF_FIELDS = ("n_layer", "d_model", "n_head", "vocab_size", "tie_embeddings")


class PlanRefused(RuntimeError):
    """No admissible configuration exists for this fleet; message has the report."""


def model_config(job: dict, hf_pins: dict | None = None) -> tuple[GPTConfig, list[str]]:
    """The model the job declares: a preset with the job's overrides on top,
    the user model's digest taken here (never trusted from the caller), an HF
    checkpoint's config.json adopted into hf_config with its digest, the
    builder child's pins set (arch "hf"), and the corpus vocabulary
    reconciled. hf_pins = {"hf_buffers", "hf_census", "hf_code", "fields"}
    (HANDOUT-M4 I-10, I-14); an hf plan without them is refused, naming the
    census. Never imports transformers (BA-441)."""
    for k in HF_PIN_KEYS:
        if job.get(k) is not None:
            raise PlanRefused(
                f"the plan carries {k}: it is the builder child's pin (I-10), never taken "
                f"from the plan; pass it through hf_pins"
            )
    mcfg = make_config(job.get("model", "toy"))
    for k in MODEL_KEYS:
        if job.get(k) is not None:
            setattr(mcfg, k, job[k])
    notes = []

    # Bring-your-own model: hash it here on the planner rather than trusting a
    # caller-supplied value; a digest taken from the same place as the claim
    # proves nothing.
    if mcfg.arch == "user" and mcfg.user_model:
        with open(mcfg.user_model, "rb") as f:
            mcfg.user_model_sha256 = hashlib.sha256(f.read()).hexdigest()

    census_fields = _adopt_hf(mcfg, job, hf_pins, notes) if mcfg.arch == "hf" else None

    # The corpus decides the vocabulary (except for a fine-tune, where the
    # checkpoint's embedding does and a mismatch is refused).
    ds_path = job.get("dataset") or None
    if ds_path:
        from wantrain.model import dataset as ds
        from wantrain.model import provision

        if job.get("dataset_source"):
            # A coordinator that holds no corpus plans from the source's manifest.
            try:
                provision.ensure_manifest(ds_path, source=job["dataset_source"])
            except (OSError, provision.DataRefused):
                pass
        try:
            meta = ds.read_manifest(ds_path)
        except (OSError, ValueError):
            meta = None
        if meta and meta.get("vocab_size"):
            v = int(meta["vocab_size"])
            if census_fields is not None:
                census_v = int(census_fields["vocab_size"])
                if v != census_v:
                    raise PlanRefused(
                        f"the corpus was tokenized with {meta.get('tokenizer')!r} (vocab {v}), "
                        f"but the model's census fields have vocab {census_v}. "
                        f"Re-tokenize the corpus with the checkpoint's tokenizer."
                    )
            if v != mcfg.vocab_size:
                notes.append(
                    f"vocab_size {mcfg.vocab_size} -> {v}: the corpus was tokenized with "
                    f"{meta.get('tokenizer')!r}, and the model must represent every id"
                )
                mcfg.vocab_size = v
            ds.check_compatible(ds_path, mcfg)
    return mcfg, notes


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def plan_hf_config(given, hf_model):
    """The hf_config an arch "hf" plan declares: with hf_model, the
    checkpoint's config.json (a plan value that differs is refused, since the
    checkpoint decides the architecture of a fine-tune); else the plan's.
    model_config and the coordinator's record builder read it here, so both
    see the same config."""
    if not hf_model:
        return given
    with open(os.path.join(hf_model, "config.json")) as f:
        adopted = json.load(f)
    if given is not None and _canon(given) != _canon(adopted):
        raise PlanRefused(
            f"hf_config: the plan's differs from {hf_model!r}'s config.json, which "
            f"decides the architecture of a fine-tune"
        )
    return adopted


def _adopt_hf(mcfg, job: dict, hf_pins: dict | None, notes: list) -> dict:
    """arch "hf": the checkpoint's config.json (when hf_model is given) is the
    hf_config, with its digest taken here; the builder child's pins and
    mirrored fields are set; a plan-dict value that differs is refused.
    -> the census fields."""
    if mcfg.hf_model:
        from wantrain.model import hf as hfmod

        adopted = plan_hf_config(job.get("hf_config"), mcfg.hf_model)
        mcfg.hf_config = adopted
        mcfg.hf_model_sha256 = hfmod.checkpoint_digest(mcfg.hf_model)
        notes.append(
            f"adopted HF config.json of {mcfg.hf_model!r} "
            f"(model_type {adopted.get('model_type')!r})"
        )
    if hf_pins is None:
        raise PlanRefused(
            "arch 'hf' plan without hf_pins: the planner prices HF units only from the "
            "builder child's hf_census (HANDOUT-M4 I-10), and none was given"
        )
    for k in (*HF_PIN_KEYS, "fields"):
        if hf_pins.get(k) is None:
            raise PlanRefused(f"hf_pins has no {k!r} (the builder child's output, I-10)")
    fields = hf_pins["fields"]
    for k in HF_FIELDS:
        if k not in fields:
            raise PlanRefused(f"hf_pins['fields'] has no {k!r} (hfm.config_fields)")
        if job.get(k) is not None and job[k] != fields[k]:
            raise PlanRefused(
                f"{k}: the plan gives {job[k]!r}, the builder's census fields {fields[k]!r}"
            )
        setattr(mcfg, k, fields[k])
    for k in HF_PIN_KEYS:
        setattr(mcfg, k, hf_pins[k])
    return fields


_PLAN_NODE_KEYS = (
    "site",
    "gpu_name",
    "host_ram_bytes",
    "host_dram_gbps",
    "pcie_h2d_gbps",
    "pcie_d2h_gbps",
    "nic_up_gbps",
    "nic_down_gbps",
    "hbm_gbps",
)


def build_spec(fleet: dict, allow_pieces: bool = False, hf_pins: dict | None = None) -> dict:
    """-> {spec, spec_json, rationale, report, warnings, stage_widths,
    placement, shard, wire, emission_order, piece_bounds, stream_grads,
    candidates}. Raises PlanRefused. allow_pieces: see planner.plan_job
    (BA-49); job "shard" restricts the candidates to that layout. hf_pins:
    the builder child's pins of an arch "hf" job (model_config, I-14)."""
    job = fleet.get("job") or {}
    nodes_in = sorted(fleet["nodes"], key=lambda n: n["rank"])
    if len(nodes_in) < 2:
        raise PlanRefused("a pipeline needs at least two nodes")
    mcfg, notes = model_config(job, hf_pins)

    nodes = [
        prov.Node(
            host=n["host"],
            tflops=float(n["tflops"]),
            vram_bytes=int(n["vram_bytes"]),
            dial_port=int(n.get("dial_port", 0)),
            listen_port=int(n.get("listen_port", fleet.get("peer_port", 0))),
            label=n.get("label", f"rank{n['rank']}"),
            compute_cap=tuple(n["compute_cap"]) if n.get("compute_cap") else None,
            device=n.get("device"),
            device_class=n.get("device_class") or "",
            hostname=n.get("hostname") or "",
            declared=tuple(n.get("declared") or ()),
            sources=dict(n.get("sources") or {}),
            **{k: n[k] for k in _PLAN_NODE_KEYS if n.get(k) is not None},
        )
        for n in nodes_in
    ]
    link = fleet.get("link") or {}
    k_passes = fleet.get("k_passes")
    f = prov.Fleet(
        nodes=nodes,
        link_mbps=float(link.get("mbps", 20.0)),
        rtt_ms=float(link.get("rtt_ms", 200.0)),
        duplex=link.get("duplex", "unknown"),
        k_passes=(float(k_passes["value"]), str(k_passes["source"])) if k_passes else None,
    )

    ov = prov.Overrides()
    for k in (
        "boundary_dtype",
        "compute_dtype",
        "mb_size",
        "microbatches",
        "data_shards",
        "recompute",
        "attn_impl",
        "partition",
    ):
        if job.get(k) is not None:
            setattr(ov, k, job[k])
    if job.get("extras"):
        ov.extras = dict(job["extras"])

    out = prov.plan_job(
        mcfg,
        f,
        dataset=job.get("dataset") or None,
        dataset_source=job.get("dataset_source") or None,
        policy=job.get("policy", "balanced"),
        margin=float(job.get("margin", 2.0)),
        # No flat default: k is solved from the measured RTT and compute window.
        k_inflight=int(job["k_inflight"]) if job.get("k_inflight") else None,
        steps=int(job.get("steps", 20)),
        lr=float(job.get("lr", 2e-4)),
        overrides=ov,
        require_margin=bool(job.get("require_margin", False)),
        device=job.get("device", "cuda"),
        bubble_target=float(job.get("bubble_target", 0.05)),
        # Measured critical batch, tokens: past it extra microbatches only buy bubble.
        tokens_per_step_cap=(
            int(job["critical_batch_tokens"]) if job.get("critical_batch_tokens") else None
        ),
        allow_pieces=allow_pieces,
        shard=job.get("shard"),
    )
    if out["spec"] is None:
        # SCHEDULER-I16: a refused wide plan names what refused every candidate
        # (the report alone shows only the width-1 precision ladder).
        rationale = out.get("rationale")
        why = rationale.get("decision", "") if isinstance(rationale, dict) else rationale
        tail = f"\n{why}" if str(why).startswith("no candidate is admitted") else ""
        raise PlanRefused("no admissible configuration for this fleet:\n" + out["report"] + tail)

    spec = out["spec"]
    spec.out_dir = job.get("out_dir", "runs")
    spec.tag = job.get("tag") or fleet.get("run_id", "run")
    spec.ckpt_every = int(job.get("ckpt_every", 0))
    for k in ("init_seed", "data_seed", "dial_timeout_s", "p2p_streams"):
        if job.get(k) is not None:
            setattr(spec, k, job[k])
    if any(n.dial_port for n in nodes):
        spec.endpoints = [[p["endpoint"] for p in row] for row in out["placement"]]
    spec.finalize()
    return dict(
        spec=spec,
        spec_json=spec.to_json(),
        rationale=out["rationale"],
        report=out["report"],
        warnings=notes + list(out["warnings"]),
        **{k: out[k] for k in _PLAN_KEYS},
    )


_PLAN_KEYS = (
    "stage_widths",
    "placement",
    "shard",
    "wire",
    "emission_order",
    "piece_bounds",
    "stream_grads",
    "candidates",
)
