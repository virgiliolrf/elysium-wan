"""Logical-layout checkpoints of a sharded stage (RUNTIME-I12, ADR-0006).

A pieces slot writes one state.pt per wave (fault_tolerance/checkpoint.py
commit_wave), holding:

    step            the checkpoint step
    pieces          {name: {master, exp_avg, exp_avg_sq, step}}: the peer's flat
                    pieces of every trainable tensor and its AdamW step counter
    hparams         the AdamW param group (lr, betas, eps, ...)
    rng             the torch RNG state
    losses          {"step:mb": loss} up to the step
    moe_aux         (MoE only) [[step, mb, layer, hex], ...] the slot recorded
                    for the microbatches it owned (RUNTIME-I18, I-12)
    nontrainable    {name: tensor}: buffers and frozen parameters (a replica;
                    an MoE stage's selection biases among them, gathered to
                    every peer at a checkpoint step)
    working         {name: sha256 of the peer's working copy}
    pins            JobSpec.pins() of the writer
    pieces_manifest the embedded manifest (build_manifest)

The manifest names the step, the wave's width, the slot's peer index and the
full layout, and lists (name, field, offset, length, sha256 per 16 MiB chunk)
of every piece; wave.json's sha256 of state.pt binds it. Resume at any width
or split reads exactly the wave's slots p0..p(w_old - 1) (select_wave),
rebuilds the logical state (assemble) and cuts its own pieces (owned_pieces),
byte for byte. A whole-tensor ZeRO-1 wave converts both ways (from_zero1,
as_zero1). Every refusal names its reason and happens before any state is
applied. Nothing here draws from the torch RNG.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence

import torch

FIELDS = ("master", "exp_avg", "exp_avg_sq", "step")
HASH_CHUNK = 16 << 20
_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _raw(t: torch.Tensor) -> bytes:
    t = t.detach().to("cpu").contiguous().reshape(-1)
    if t.numel() == 0:
        return b""
    return t.view(_VIEW[t.element_size()]).numpy().tobytes()


def digest(t: torch.Tensor) -> str:
    """sha256 of the tensor's raw bytes."""
    return hashlib.sha256(_raw(t)).hexdigest()


def chunk_hashes(t: torch.Tensor) -> list[str]:
    """sha256 of each 16 MiB chunk of the tensor's raw bytes (one for an
    empty tensor)."""
    raw = _raw(t)
    if not raw:
        return [hashlib.sha256(b"").hexdigest()]
    return [
        hashlib.sha256(raw[i : i + HASH_CHUNK]).hexdigest() for i in range(0, len(raw), HASH_CHUNK)
    ]


def build_manifest(
    step: int, width: int, peer: int, layout: dict, tensors: dict, shapes: dict
) -> dict:
    """The manifest of one slot's pieces. layout: {name: [b_0..b_w]} (every
    trainable tensor, canonical order); tensors: {name: {field: tensor}};
    shapes: {name: shape}."""
    entries = []
    for name, b in layout.items():
        for field in FIELDS:
            t = tensors[name][field]
            off, length = (b[peer], b[peer + 1] - b[peer]) if field != "step" else (0, 1)
            entries.append(
                {
                    "name": name,
                    "field": field,
                    "offset": off,
                    "length": length,
                    "sha256": chunk_hashes(t),
                }
            )
    return {
        "v": 1,
        "step": step,
        "width": width,
        "peer": peer,
        "layout": {n: list(b) for n, b in layout.items()},
        "shapes": {n: list(shapes[n]) for n in layout},
        "entries": entries,
    }


def check_manifest(manifest, payload: dict, where: str) -> dict:
    """The payload's embedded manifest, proven to describe it: well formed,
    the payload's step, one entry per (tensor, field), each piece of its
    entry's length with its chunk hashes. Raises ValueError naming `where`."""
    if not isinstance(manifest, dict) or manifest.get("v") != 1:
        raise ValueError(f"{where}: missing manifest (a pieces payload carries pieces_manifest)")
    for k in ("step", "width", "peer"):
        if type(manifest.get(k)) is not int:
            raise ValueError(f"{where}: manifest has no integer {k}")
    if manifest["step"] != payload.get("step"):
        raise ValueError(
            f"{where}: manifest names step {manifest['step']}, the payload holds step "
            f"{payload.get('step')!r}"
        )
    layout, shapes = manifest.get("layout"), manifest.get("shapes")
    if not isinstance(layout, dict) or not isinstance(shapes, dict) or set(layout) != set(shapes):
        raise ValueError(f"{where}: manifest has no layout")
    w, k = manifest["width"], manifest["peer"]
    if not 0 <= k < w:
        raise ValueError(f"{where}: manifest peer {k} outside width {w}")
    pieces = payload.get("pieces")
    if not isinstance(pieces, dict) or set(pieces) != set(layout):
        raise ValueError(f"{where}: pieces do not match the manifest's tensors")
    seen = set()
    for e in manifest.get("entries") or []:
        name, field = e.get("name"), e.get("field")
        if name not in layout or field not in FIELDS or (name, field) in seen:
            raise ValueError(f"{where}: manifest entry {name!r}.{field!r} is unknown or repeated")
        seen.add((name, field))
        b = layout[name]
        if len(b) != w + 1:
            raise ValueError(f"{where}: layout of {name!r} has {len(b) - 1} pieces, width is {w}")
        want = (b[k], b[k + 1] - b[k]) if field != "step" else (0, 1)
        if (e.get("offset"), e.get("length")) != want:
            raise ValueError(f"{where}: {name}.{field} is not piece {k} of the layout")
        t = pieces[name].get(field)
        if not isinstance(t, torch.Tensor) or t.numel() != want[1]:
            raise ValueError(f"{where}: {name}.{field} has not the manifest's length {want[1]}")
        if chunk_hashes(t) != e.get("sha256"):
            raise ValueError(f"{where}: {name}.{field} chunk sha256 mismatch")
    if len(seen) != len(layout) * len(FIELDS):
        raise ValueError(f"{where}: manifest does not list every piece")
    return manifest


def select_wave(read_manifest: Callable[[int], dict], peers_present: Sequence[int]) -> tuple:
    """-> (w_old, layout) of the wave at the resume step, read from slot p0's
    manifest. Slots p0..p(w_old-1) must be present and their manifests name
    that step, width and layout and their own index; a present slot >= w_old
    whose manifest names that step and width w_old is an extra index (refused);
    one naming another width belongs to an older wave and is never read as
    state. read_manifest(p) returns slot p's manifest (raising if unreadable)."""
    present = set(peers_present)
    if 0 not in present:
        raise ValueError("slot p0 holds no wave at the resume step: missing peer p0")
    m0 = read_manifest(0)
    w_old, layout, step = m0["width"], m0["layout"], m0["step"]
    for p in range(w_old):
        if p not in present:
            raise ValueError(f"slot p{p} of the width-{w_old} wave at step {step} is missing")
        m = m0 if p == 0 else read_manifest(p)
        if m["width"] != w_old or m["layout"] != layout or m["step"] != step:
            raise ValueError(
                f"slot p{p}: its manifest names width {m['width']}, step {m['step']} or another "
                f"layout than slot p0's (width {w_old}, step {step})"
            )
        if m["peer"] != p:
            raise ValueError(f"slot p{p}: its manifest names peer index {m['peer']}")
    for p in sorted(x for x in present if x >= w_old):
        m = read_manifest(p)
        if m["step"] == step and m["width"] == w_old:
            raise ValueError(f"slot p{p}: an extra index in the width-{w_old} wave at step {step}")
    return w_old, layout


def _equal(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and _raw(a) == _raw(b)
        )
    return a == b


def _merge_common(items: Sequence[tuple], step: int, microbatches: int | None) -> dict:
    """Losses, RNG, non-trainable state and hyperparameters of a wave's slots:
    losses merged (two values for one (step, mb) refused; with microbatches,
    every (s, mb) up to `step` required), the rest identical on every slot."""
    losses: dict = {}
    for where, p in items:
        for key, v in (p.get("losses") or {}).items():
            if key in losses and losses[key] != v:
                raise ValueError(f"{where}: two values for (step, mb) {key}")
            losses[key] = v
    if microbatches is not None:
        for s in range(step + 1):
            for mb in range(microbatches):
                if f"{s}:{mb}" not in losses:
                    raise ValueError(f"the wave holds no loss for (step, mb) ({s}, {mb})")
    moe_aux = _merge_moe_aux(items)
    (w0, first), rest = items[0], items[1:]
    for where, p in rest:
        if not _equal(p.get("rng"), first.get("rng")):
            raise ValueError(f"{where}: its RNG state differs from {w0}'s")
        a, b = p.get("nontrainable") or {}, first.get("nontrainable") or {}
        if set(a) != set(b) or any(not _equal(a[k], b[k]) for k in a):
            raise ValueError(f"{where}: its non-trainable state differs from {w0}'s")
        if p.get("hparams") != first.get("hparams"):
            raise ValueError(f"{where}: its optimizer hyperparameters differ from {w0}'s")
    out = {
        "losses": losses,
        "rng": first.get("rng"),
        "nontrainable": dict(first.get("nontrainable") or {}),
        "hparams": first.get("hparams"),
    }
    if moe_aux is not None:
        out["moe_aux"] = moe_aux
    return out


def _merge_moe_aux(items: Sequence[tuple]) -> list | None:
    """The slots' MoE aux records (RUNTIME-I18): each (step, mb, layer) held by
    exactly one slot, the one that owned the microbatch; a triple held twice is
    refused, naming it. None when no slot holds the key (a dense wave)."""
    if not any("moe_aux" in p for _, p in items):
        return None
    seen: dict = {}
    for where, p in items:
        for row in p.get("moe_aux") or []:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                raise ValueError(f"{where}: a malformed moe_aux record {row!r}")
            key = (row[0], row[1], row[2])
            if key in seen:
                raise ValueError(
                    f"{where}: moe_aux (step, mb, layer) {key} also held by {seen[key][0]}"
                )
            seen[key] = (where, row[3])
    return [[s, mb, layer, seen[(s, mb, layer)][1]] for s, mb, layer in sorted(seen)]


def assemble(payloads: Sequence[tuple], microbatches: int | None = None) -> dict:
    """[(where, payload)] of slots p0..p(w-1) -> the logical state:
    {"step", "width", "layout", "tensors": {name: {master, exp_avg,
    exp_avg_sq, step}} (full shapes, canonical name order), "losses", "rng",
    "nontrainable", "hparams"}. Refused, naming it, before anything is
    returned: a missing manifest, a manifest step other than the payload's, a
    chunk hash mismatch, a missing peer, pieces that leave a gap or overlap,
    step counters that differ, two values for one (step, mb), a missing
    (step, mb) when microbatches is given (the final stage), and slots whose
    RNG or non-trainable state differ."""
    if not payloads:
        raise ValueError("assemble: no slot payloads")
    mans = []
    for where, p in payloads:
        if not isinstance(p, dict) or "pieces" not in p:
            raise ValueError(f"{where}: not a pieces payload")
        if "pieces_manifest" not in p:
            raise ValueError(f"{where}: pieces payload without its manifest")
        mans.append(check_manifest(p["pieces_manifest"], p, where))
    step, w = mans[0]["step"], mans[0]["width"]
    peers = {}
    for (where, _), m in zip(payloads, mans):
        if m["step"] != step or m["width"] != w:
            raise ValueError(f"{where}: manifest names step {m['step']} width {m['width']}")
        if m["peer"] in peers:
            raise ValueError(f"{where}: peer p{m['peer']} given twice")
        peers[m["peer"]] = (where, m)
    for k in range(w):
        if k not in peers:
            raise ValueError(f"missing peer p{k} of the width-{w} wave at step {step}")
    by_peer = {m["peer"]: p for (_, p), m in zip(payloads, mans)}
    names = list(mans[0]["layout"])
    tensors = {}
    for name in names:
        shape = mans[0]["shapes"][name]
        numel = math.prod(shape)
        ranges = sorted(
            (peers[k][1]["layout"][name][k], peers[k][1]["layout"][name][k + 1], k)
            for k in range(w)
        )
        pos = 0
        for lo, hi, _k in ranges:
            if lo != pos or hi < lo:
                raise ValueError(f"pieces of {name!r} leave a gap or overlap at element {pos}")
            pos = hi
        if pos != numel:
            raise ValueError(f"pieces of {name!r} cover {pos} of {numel} elements")
        full = {}
        for field in FIELDS[:3]:
            parts = [by_peer[k]["pieces"][name][field].reshape(-1) for _, _, k in ranges]
            full[field] = torch.cat(parts).reshape(shape)
        steps = [by_peer[k]["pieces"][name]["step"] for k in range(w)]
        if any(not _equal(s, steps[0]) for s in steps[1:]):
            raise ValueError(f"step counters of {name!r} differ between peers")
        full["step"] = steps[0].clone()
        tensors[name] = full
    common = _merge_common(list(payloads), step, microbatches)
    return {"step": step, "width": w, "layout": mans[0]["layout"], "tensors": tensors, **common}


def owned_pieces(logical: dict, layout: dict, peer: int) -> dict:
    """{name: {master, exp_avg, exp_avg_sq, step}}: piece `peer` of every
    logical tensor under `layout`, as flat copies."""
    out = {}
    for name, b in layout.items():
        t = logical["tensors"][name]
        lo, hi = b[peer], b[peer + 1]
        out[name] = {f: t[f].reshape(-1)[lo:hi].clone() for f in FIELDS[:3]}
        out[name]["step"] = t["step"].clone()
    return out


def as_zero1(logical: dict, names: Sequence[str], w: int, k: int) -> list:
    """ZeRO-1 shard k of a width-w stage, names[k::w], of the logical
    tensors: [(name, master, {"step", "exp_avg", "exp_avg_sq"})]."""
    out = []
    for name in list(names)[k::w]:
        t = logical["tensors"][name]
        state = {
            "step": t["step"].clone(),
            "exp_avg": t["exp_avg"].clone(),
            "exp_avg_sq": t["exp_avg_sq"].clone(),
        }
        out.append((name, t["master"].clone(), state))
    return out


def from_zero1(
    payloads: Sequence, names: Sequence[str], w: int, microbatches: int | None = None
) -> dict:
    """The logical state of a whole-tensor ZeRO-1 wave: payloads[k] is slot
    k's payload ({step, chunks: {vs: state_dict}, opt, rng, losses}) or
    (where, payload); peer k owns names[k::w]."""
    items = [p if isinstance(p, tuple) else (f"slot p{k}", p) for k, p in enumerate(payloads)]
    if len(items) != w:
        raise ValueError(f"from_zero1: {len(items)} payloads for width {w}")
    names = list(names)
    step = items[0][1].get("step")
    tensors = {}
    merged = []
    for k, (where, p) in enumerate(items):
        if not isinstance(p, dict) or "pieces" in p or "chunks" not in p or "opt" not in p:
            raise ValueError(f"{where}: not a whole-tensor zero1 payload (a wave mixing kinds)")
        if p.get("step") != step:
            raise ValueError(f"{where}: holds step {p.get('step')!r}, slot p0 holds {step!r}")
        (sd,) = p["chunks"].values()
        state = p["opt"]["state"]
        mine = names[k::w]
        if len(state) != len(mine):
            raise ValueError(
                f"{where}: optimizer state of {len(state)} tensors, shard has {len(mine)}"
            )
        for j, name in enumerate(mine):
            st = state[j]
            tensors[name] = {
                "master": sd[name].clone(),
                "exp_avg": st["exp_avg"].clone(),
                "exp_avg_sq": st["exp_avg_sq"].clone(),
                "step": st["step"].clone(),
            }
        group = {key: v for key, v in p["opt"]["param_groups"][0].items() if key != "params"}
        rest = {n: t for n, t in sd.items() if n not in set(names)}
        merged.append((where, {**p, "hparams": group, "nontrainable": rest}))
    common = _merge_common(merged, step, microbatches)
    tensors = {n: tensors[n] for n in names}
    return {"step": step, "width": w, "layout": None, "tensors": tensors, **common}
