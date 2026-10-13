"""Rollback per piece (BA-467; SPEC-model task 4, SPEC-runtime task 7, ADR-0009
round6.md section 4).

Each M4 piece reverts independently to its own target by adding its id to
ROLLED_BACK, a one-line edit. Every reader (gpt.validate, hfm, spec.finalize,
the stage's TIED receipt, the planner) reads this module's attributes at call
time, so a monkeypatch takes effect. The module imports nothing.
"""

ROLLED_BACK: frozenset[str] = frozenset()

# piece id -> the target its rollback reverts to:
# - 'model-4a': arch 'hf' refused at config validation (GPTConfig.validate);
# - 'tied-split-refused': a split tied hf table refused (spec.finalize, planner);
# - 'moe-from-scratch': hfm ignores the SUBSTITUTE rows (HF MoE refused
#   MOE-NO-ROW) while native MoE runs.
PIECES: dict[str, str] = {
    "scheduler-6": "tied-split-refused",
    "runtime-7a": "tied-split-refused",
    "runtime-7b": "tied-split-refused",
    "runtime-7c": "model-4a",
    "runtime-7d": "model-4a",
    "model-4b": "model-4a",
    "control-6": "model-4a",
    "model-3-6a": "moe-from-scratch",
}


def target(piece: str) -> str | None:
    """The target `piece` is rolled back to, or None while it is live."""
    if piece not in PIECES:
        raise KeyError(f"unknown rollback piece {piece!r}; known: {sorted(PIECES)}")
    return PIECES[piece] if piece in ROLLED_BACK else None


def rolled_back_to(goal: str) -> list[str]:
    """The pieces in ROLLED_BACK whose target is `goal`, sorted; an unknown id
    in ROLLED_BACK is a KeyError naming it, never ignored."""
    return sorted(p for p in ROLLED_BACK if target(p) == goal)
