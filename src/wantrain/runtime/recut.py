"""Re-cut a width-1 chain's checkpoint to another partition (RUNTIME-I26).

A wave of a width-1 builtin stage holds its chunk's state_dict (keys "<local entry>.<name>"), its optimizer state
(indexed by the stage's trainable parameters, entry by entry in module order), its RNG state, the losses it holds
and the recipe pins. Moving layers between stages moves exactly those tensors: every entry keeps its parameters,
buffers and optimizer state byte for byte, only renumbered for the stage that now holds it, and the pins name the
new partition. A run resumed from the re-cut waves under the new partition therefore continues the uninterrupted
run's sequence exactly (tests/integration/test_engine_bitwise.py, test_recut_resume_*).

Refused before anything is written: another stage count, waves of different steps, a pieces or MoE payload, a
chunk that is not its rank's, an optimizer state whose parameter count does not match the stage's entries.
"""

from __future__ import annotations

import torch

from wantrain.model.gpt import build_layers


def entry_param_names(model) -> list[list[str]]:
    """Per entry ([Embed, Block*n, Head]), its trainable parameters' names in the order a stage enumerates them
    (chunk.parameters()), from the model built on the meta device: no memory, no draw."""
    with torch.device("meta"):
        layers = build_layers(model, 0, span=None)
    return [[n for n, p in e.named_parameters() if p.requires_grad] for e in layers]


def _check(states, old, new) -> int:
    if len(states) != len(old) or len(old) != len(new):
        raise ValueError(
            f"re-cut keeps the stage count: {len(states)} waves, {len(old)} -> {len(new)} stages"
        )
    for part in (old, new):
        if part[0][0] != 0 or any(part[i][1] != part[i + 1][0] for i in range(len(part) - 1)):
            raise ValueError(f"partition {part} does not cover the entries contiguously")
        if any(a >= b for a, b in part):
            raise ValueError(f"partition {part} has an empty stage")
    if old[-1][1] != new[-1][1]:
        raise ValueError(f"partitions cover {old[-1][1]} and {new[-1][1]} entries")
    steps = {int(s["step"]) for s in states}
    if len(steps) != 1:
        raise ValueError(f"waves of different steps {sorted(steps)}")
    for r, s in enumerate(states):
        if "pieces" in s or s.get("moe_aux"):
            raise ValueError(f"wave r{r}: a pieces or MoE payload is not re-cut here")
        if set(s["chunks"]) != {r}:
            raise ValueError(f"wave r{r} holds chunks {sorted(s['chunks'])}, not its own")
    return steps.pop()


def recut_states(states: list[dict], old: list, new: list, names: list[list[str]]) -> list[dict]:
    """states[r]: the wave payload of width-1 stage r under partition `old`; -> the payloads of the same step under
    partition `new` (same stage count), every tensor the same object, renumbered."""
    old = [tuple(x) for x in old]
    new = [tuple(x) for x in new]
    step = _check(states, old, new)
    tensors: dict = {}  # entry -> [(name, tensor)] in state_dict order
    moments: dict = {}  # (entry, parameter name) -> optimizer state
    for r, s in enumerate(states):
        a, b = old[r]
        for key, t in s["chunks"][r].items():
            j, rest = key.split(".", 1)
            g = a + int(j)
            if not a <= g < b:
                raise ValueError(f"wave r{r}: key {key!r} is outside its entries {a}..{b - 1}")
            tensors.setdefault(g, []).append((rest, t))
        params = [(g, n) for g in range(a, b) for n in names[g]]
        opt = s["opt"]
        listed = opt["param_groups"][0]["params"]
        if len(listed) != len(params):
            raise ValueError(
                f"wave r{r}: optimizer lists {len(listed)} parameters, its entries hold {len(params)}"
            )
        for i, key in enumerate(params):
            if i in opt["state"]:
                moments[key] = opt["state"][i]
    losses: dict = {}
    for s in states:
        losses.update(s.get("losses") or {})
    out = []
    for r, (a, b) in enumerate(new):
        chunk = {f"{g - a}.{rest}": t for g in range(a, b) for rest, t in tensors.get(g, [])}
        params = [(g, n) for g in range(a, b) for n in names[g]]
        group = dict(states[r]["opt"]["param_groups"][0])
        group["params"] = list(range(len(params)))
        opt = {
            "state": {i: moments[k] for i, k in enumerate(params) if k in moments},
            "param_groups": [group],
        }
        pins = dict(states[r]["pins"])
        pins["partition"] = [list(x) for x in new]
        payload = {
            k: v
            for k, v in states[r].items()
            if k not in ("chunks", "opt", "pins", "losses", "step")
        }
        payload.update(step=step, chunks={r: chunk}, opt=opt, pins=pins, losses=dict(losses))
        out.append(payload)
    return out
