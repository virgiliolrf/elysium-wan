"""Canonical training state: the bytes the byte-exact rule compares.

A run's state is every parameter and buffer of the model, the AdamW state of
every trainable parameter, the optimizer's hyperparameters, the global torch
RNG state and the loss of every (step, microbatch). Each entry is keyed by its
name in the full model (layers.<i>.<name>), never by where a pipeline stage or
a ZeRO-1 shard kept it, so a single-device reference and any distributed run
produce the same keys. The manifest records each tensor's dtype, shape and the
sha256 of its raw bytes: two runs left byte-identical state iff their manifests
are byte-identical, and a mismatch names the tensor that moved.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

import torch

from wantrain.model.gpt import build_layers
from wantrain.runtime.spec import JobSpec


def tensor_record(t: torch.Tensor) -> dict:
    t = t.detach().to("cpu").contiguous()
    flat = t.reshape(-1)
    raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    return {
        "dtype": str(t.dtype).removeprefix("torch."),
        "shape": list(t.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _value(v):
    if isinstance(v, torch.Tensor):
        return tensor_record(v)
    if isinstance(v, float):
        return {"float": v.hex()}
    if isinstance(v, (list, tuple)):
        return [_value(x) for x in v]
    return v


def aux_hex(aux: dict) -> str:
    """The canonical text of one MoE layer's aux values for one (step, mb)
    (RUNTIME-I18): canonical JSON of {name: float hex}, keys sorted, the text
    runtime.stage.moe_aux_hex writes into a slot's checkpoint "moe_aux" (I-12),
    so the oracle's rows and the run's compare as strings. A value is a number
    or already a float hex string (arch.moe_forward records hex); both give
    the hex of the same double."""

    def h(v) -> str:
        return float.fromhex(v).hex() if isinstance(v, str) else float(v).hex()

    return json.dumps({k: h(v) for k, v in sorted(aux.items())}, separators=(",", ":"))


def aux_section(rows) -> list:
    """BA-457: the manifest's aux section, [step, mb, layer, hex] sorted. A
    (step, mb, layer) recorded twice stays twice: diff_manifests fails on it."""
    return [[int(s), int(m), int(layer), str(h)] for s, m, layer, h in sorted(rows)]


def build_manifest(
    weights: dict, optimizer: dict, hparams: dict, rng, losses: list, aux=None
) -> dict:
    """weights: name -> tensor; optimizer: trainable name -> AdamW state dict;
    rng: a uint8 RNG state tensor, or a list of distinct ones when the
    processes of a run disagree; losses: [(step, mb, loss_hex)]; aux: None
    (no MoE layer recorded anything: the manifest has no aux key, so a dense
    manifest is byte-identical to one written before MoE existed) or
    [(step, mb, layer, hex)]."""
    m = {
        "weights": {k: tensor_record(v) for k, v in sorted(weights.items())},
        "optimizer": {
            k: {s: _value(v) for s, v in sorted(st.items())} for k, st in sorted(optimizer.items())
        },
        "hparams": {k: _value(v) for k, v in sorted(hparams.items())},
        "rng": [tensor_record(r) for r in rng] if isinstance(rng, list) else tensor_record(rng),
        "losses": [[int(s), int(m), h] for s, m, h in sorted(losses)],
    }
    if aux is not None:
        m["aux"] = aux_section(aux)
    return m


def write_manifest(path: str, manifest: dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    return path


def _hparams(opt_state: dict) -> dict:
    groups = opt_state["param_groups"]
    assert len(groups) == 1, "one AdamW param group per optimizer"
    return {k: v for k, v in groups[0].items() if k != "params"}


def _trainable_names(layers, start: int, end: int) -> list[str]:
    """Trainable parameter names of layers [start, end), in the order the
    stage runtime and the reference hand them to AdamW."""
    names, seen = [], set()
    for i in range(start, end):
        for n, p in layers[i].named_parameters():
            # A tied parameter (tie_embeddings) is one parameter, named where it
            # first appears, as parameters() and AdamW see it.
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                names.append(f"layers.{i}.{n}")
    return names


def _unit_state(layers) -> dict:
    """layers.<i>.<name> -> tensor over the units of the flat list, each
    unit's own state_dict. For a plain ModuleList this is its state_dict; HF
    units (hfm.HFUnits) share one tree, and only the units' own names count
    (BA-421), so a tied table appears under every name a unit gives it."""
    return {f"layers.{i}.{k}": v for i, u in enumerate(layers) for k, v in u.state_dict().items()}


def reference_manifest(layers, opt, losses: list, aux=None) -> dict:
    """The single-device reference's state, from its live model and optimizer."""
    sd = opt.state_dict()
    names = _trainable_names(layers, 0, len(layers))
    assert len(names) == len(sd["param_groups"][0]["params"]), "AdamW holds every trainable param"
    weights = _unit_state(layers)
    optimizer = {names[i]: st for i, st in sd["state"].items()}
    return build_manifest(weights, optimizer, _hparams(sd), torch.get_rng_state(), losses, aux)


class _HFNames:
    """BA-421: an arch "hf" run's manifest names, from its units (the same
    build the oracle runs). A parameter whose names fall on several stages (a
    tied table split between stage 0 and the Head) belongs to the last of
    them: its bytes are read only from that stage's checkpoint and recorded
    under every name the oracle's state_dict gives it; stage 0's working copy
    is never read (BA-411). Buffers are the run's buffer record (RUNTIME-I19):
    the record is applied to the build and checked by sha256, and every slot's
    copy that differs from it is a split that never compares equal. Selection
    biases change in training, so they are read from the checkpoints like
    weights."""

    def __init__(self, spec, layers):
        from wantrain.model import hfm

        record = spec.model.hf_buffers
        if not record:
            raise ValueError("arch='hf' manifest refused: spec.model.hf_buffers is not pinned")
        root = layers.root
        hfm.apply_buffer_record(root, record)
        stage_of = {}
        for r, (a, b) in enumerate(spec.partition):
            for u in range(a, b):
                stage_of[u] = r
        self.canon: dict = {}  # id(param) -> the oracle's AdamW name
        names_of: dict = {}  # id(param) -> every state_dict name
        stages_of: dict = {}  # id(param) -> the stages whose units hold it
        params = {id(p) for p in layers.parameters()}
        for u, unit in enumerate(layers):
            for n, p in unit.named_parameters():
                self.canon.setdefault(id(p), f"layers.{u}.{n}")
            for n, t in unit.state_dict(keep_vars=True).items():
                if id(t) in params:
                    names_of.setdefault(id(t), []).append(f"layers.{u}.{n}")
                    stages_of.setdefault(id(t), set()).add(stage_of[u])
        self.owner = {k: max(v) for k, v in stages_of.items()}
        self.tied = {k: names_of[k] for k, v in stages_of.items() if len(v) > 1}
        self.never_read = {
            n: self.owner[k] for k, ns in self.tied.items() for n in ns
        }  # name -> the stage whose copy is read
        root_name = {id(t): n for n, t in root.named_buffers()}
        self.record: dict = {}
        for key, t in _unit_state_vars(layers).items():
            n = root_name.get(id(t))
            if n is None or id(t) in params or n.endswith("sel_bias"):
                continue
            want = record["tensors"].get(n)
            if want is None:
                continue
            if tensor_record(t)["sha256"] != want["sha256"]:
                raise ValueError(f"arch='hf' manifest: buffer {n} differs from the buffer record")
            self.record[key] = t.detach().to("cpu").clone()
        self.layers = layers

    def stage_names(self, a: int, b: int, r: int) -> list[str]:
        """Stage r's trainable names in its AdamW order, as the oracle names
        them; a split tied table only on the stage that owns it."""
        out, seen = [], set()
        for u in range(a, b):
            for _, p in self.layers[u].named_parameters():
                if p.requires_grad and id(p) not in seen and self.owner.get(id(p), r) == r:
                    seen.add(id(p))
                    out.append(self.canon[id(p)])
        return out

    def read(self, key: str, r: int) -> bool:
        """Whether stage r's checkpoint value under `key` is read."""
        return self.never_read.get(key, r) == r

    def fill(self, weights: dict, split: dict) -> None:
        for names in self.tied.values():
            have = [n for n in names if n in weights]
            if not have:
                continue
            for n in names:
                weights[n] = weights[have[0]]
                if have[0] in split:
                    split[n] = split[have[0]]
        for key, t in self.record.items():
            if key in weights and tensor_record(weights[key]) != tensor_record(t):
                split.setdefault(key, [tensor_record(t)]).append(tensor_record(weights[key]))
            weights[key] = t


def _unit_state_vars(layers) -> dict:
    return {
        f"layers.{i}.{k}": v
        for i, u in enumerate(layers)
        for k, v in u.state_dict(keep_vars=True).items()
    }


def distributed_manifest(spec: JobSpec) -> dict:
    """A finished distributed run's state, assembled from the final checkpoint
    of every (rank, peer) slot and the loss file of the last stage.

    Every peer of a wide stage holds the stage's full parameters after the
    ZeRO-1 broadcast, and they must agree: a disagreement is recorded under the
    parameter as a list of distinct digests, so it can never compare equal to
    the reference. Optimizer state lives with the owning peer: trainable
    parameter i of a stage of width w is element i // w of peer i % w's shard.
    """
    spec.finalize()
    layers = build_layers(spec.model, spec.init_seed)
    hf = _HFNames(spec, layers) if spec.model.arch == "hf" else None
    weights: dict = {}
    split: dict = {}
    optimizer: dict = {}
    hparams: list = []
    rngs: list = []
    aux: list = []
    for r in range(spec.stages):
        a, b = spec.partition[r]
        names = hf.stage_names(a, b, r) if hf else _trainable_names(layers, a, b)
        w = spec.stage_widths[r]
        if spec.shard == "pieces":
            _pieces_stage(spec, layers, r, weights, split, optimizer, hparams, rngs, aux)
            continue
        for p in range(w):
            path = os.path.join(spec.out_dir, f"{spec.tag}_ckpt_rank{r}p{p}.pt")
            ck = torch.load(path, map_location="cpu", weights_only=True)
            # I-12: every slot keeps the aux of the microbatches it owned.
            aux.extend(ck.get("moe_aux") or [])
            for k, v in ck["chunks"][r].items():
                i, rest = k.split(".", 1)
                key = f"layers.{a + int(i)}.{rest}"
                if hf is not None and not hf.read(key, r):
                    continue
                if key not in weights:
                    weights[key] = v
                elif tensor_record(weights[key]) != tensor_record(v):
                    split.setdefault(key, [tensor_record(weights[key])]).append(tensor_record(v))
            mine = names[p::w]
            for j, st in ck["opt"]["state"].items():
                optimizer[mine[j]] = st
            hparams.append(_hparams(ck["opt"]))
            rng = ck.get("rng")
            rngs.append(rng if rng is not None else torch.zeros(0, dtype=torch.uint8))
    with open(os.path.join(spec.out_dir, f"{spec.tag}_losses.json")) as f:
        losses = [(x["step"], x["mb"], x["loss_hex"]) for x in json.load(f)["losses"]]
    distinct_rng = []
    for t in rngs:
        if not any(torch.equal(t, u) for u in distinct_rng):
            distinct_rng.append(t)
    distinct_hp = [h for i, h in enumerate(hparams) if h not in hparams[:i]]
    if hf is not None:
        hf.fill(weights, split)
    m = build_manifest(
        weights,
        optimizer,
        distinct_hp[0] if len(distinct_hp) == 1 else {"disagree": distinct_hp},
        distinct_rng[0] if len(distinct_rng) == 1 else distinct_rng,
        losses,
        aux or None,
    )
    for key, recs in split.items():
        m["weights"][key] = recs
    return m


def _pieces_stage(spec, layers, r, weights, split, optimizer, hparams, rngs, aux) -> None:
    """BA-21: a shard "pieces" stage's state. The logical trainable state is
    assembled in canonical name order from every slot's pieces
    (logical_ckpt.assemble, which refuses gaps, overlaps, hash mismatches and
    disagreeing replicas); non-trainable weights and buffers come from every
    slot's replica and the RNG from every slot (distinct values never compare
    equal); each peer's final working copy must be the cast of the logical
    master in its consumer dtype, and a peer whose digest differs is recorded
    as a split that never compares equal."""
    from torch import nn

    from wantrain.model import wgrad
    from wantrain.runtime import logical_ckpt as lc

    a, b = spec.partition[r]
    chunk = nn.ModuleList(list(layers)[a:b])
    payloads = []
    for p in range(spec.stage_widths[r]):
        path = os.path.join(spec.out_dir, f"{spec.tag}_ckpt_rank{r}p{p}.pt")
        payloads.append((path, torch.load(path, map_location="cpu", weights_only=True)))
    logical = lc.assemble(payloads)
    m = spec.model
    gd = wgrad.gather_dtypes(chunk, m.compute_dtype, m.autocast_enabled())
    canon = {}
    for n, t in chunk.named_parameters():
        canon.setdefault(id(t), n)
    for n, t in chunk.named_buffers():
        canon.setdefault(id(t), n)
    want_working = {}
    for n, t in chunk.named_parameters():
        if t.requires_grad:
            master = logical["tensors"][n]["master"]
            want_working[n] = lc.digest(master.to(gd[id(t)]))
            optimizer[f"layers.{a + int(n.split('.', 1)[0])}.{n.split('.', 1)[1]}"] = {
                f: logical["tensors"][n][f] for f in ("exp_avg", "exp_avg_sq", "step")
            }
    for key, t in chunk.state_dict(keep_vars=True).items():
        i, rest = key.split(".", 1)
        full = f"layers.{a + int(i)}.{rest}"
        n = canon.get(id(t), key)
        if n in logical["tensors"]:
            weights[full] = logical["tensors"][n]["master"]
            recs = [tensor_record(weights[full])]
            for path, pl in payloads:
                if (pl.get("working") or {}).get(n) != want_working[n]:
                    recs.append({"working_copy": path, "sha256": (pl.get("working") or {}).get(n)})
            if len(recs) > 1:
                split[full] = recs
            continue
        for _, pl in payloads:
            v = (pl.get("nontrainable") or {}).get(n)
            if v is None:
                split.setdefault(full, []).append({"missing": n})
            elif full not in weights:
                weights[full] = v
            elif tensor_record(weights[full]) != tensor_record(v):
                split.setdefault(full, [tensor_record(weights[full])]).append(tensor_record(v))
    for _, pl in payloads:
        aux.extend(pl.get("moe_aux") or [])
        hparams.append(pl.get("hparams"))
        rng = pl.get("rng")
        rngs.append(rng if rng is not None else torch.zeros(0, dtype=torch.uint8))


def _aux_diff(x, y) -> list[str]:
    """BA-457: a (step, mb, layer) recorded twice (on either side, even when
    both sides agree) or recorded on one side only fails, as does a hex that
    differs."""
    out = []
    for side, rows in (("a", x), ("b", y)):
        counts = Counter((s, m, layer) for s, m, layer, _ in rows or [])
        out += [f"aux[{side}] {k}: recorded {n} times" for k, n in sorted(counts.items()) if n > 1]
    if x is None or y is None:
        if x != y:
            out.append(
                f"aux: {'absent' if x is None else len(x)} != {'absent' if y is None else len(y)}"
            )
        return out
    hx = {(s, m, layer): h for s, m, layer, h in x}
    hy = {(s, m, layer): h for s, m, layer, h in y}
    for k in sorted(set(hx) | set(hy)):
        if k not in hx or k not in hy:
            out.append(f"aux {k}: missing in {'a' if k not in hx else 'b'}")
        elif hx[k] != hy[k]:
            out.append(f"aux {k}: {hx[k]} != {hy[k]}")
    return out


def diff_manifests(a: dict, b: dict, limit: int = 10) -> list[str]:
    """Human-readable differences, first `limit`: which entries moved."""
    out = _aux_diff(a.get("aux"), b.get("aux"))
    for section in ("weights", "optimizer", "hparams", "rng", "losses"):
        x, y = a.get(section), b.get(section)
        if x == y:
            continue
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if x.get(k) != y.get(k):
                    out.append(f"{section}.{k}: {x.get(k)} != {y.get(k)}")
        elif section == "losses" and isinstance(x, list) and isinstance(y, list):
            for i, (u, v) in enumerate(zip(x, y)):
                if u != v:
                    out.append(f"losses[{i}]: {u} != {v}")
                    break
            if len(x) != len(y):
                out.append(f"losses: {len(x)} != {len(y)} entries")
        else:
            out.append(f"{section}: {x} != {y}")
    return out[:limit]
