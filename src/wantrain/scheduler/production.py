"""The production feature set: one place, imported by the planner, always on.

The pipeline itself has one execution model, zero-bubble pipeline
parallelism (runtime/stage.py); what remains configurable is how a stage
spends memory and how its boundary travels. recompute re-runs the same math
in the same order, so it is bit-identical; wire_codec 'planes' is
transport-lossless by construction. A dropped flag is a diff in this file,
not a surprise in a run log.
"""

# Fields consumed by JobSpec(**...). plan_job() merges these into every spec
# it emits; explicit operator overrides still win, and the run log reports the
# override so the deviation is on the record.
PRODUCTION = dict(
    recompute=True,
    wire_codec="planes",
    # Over a long path one TCP flow is loss- and window-limited (40-50 Mbps at 130 ms, 2026-10-12); fragments over
    # parallel flows add up until the path itself saturates (TRANSPORT-I10 tunes each flow).
    p2p_streams=8,
)

# Wide-stage flags, applied only when any stage width exceeds 1. ZeRO-1
# sharding has no flag: a stage of width w shards by construction.
PRODUCTION_WIDE: dict = {}


def spec_fields(stage_widths=None, overrides: dict | None = None) -> dict:
    """The production JobSpec fields, with operator overrides layered on top.

    overrides beat PRODUCTION on purpose: the engineer keeps the wheel. The
    caller logs any key where the override disagrees with the production
    value -- deviation is allowed, silent deviation is not.
    """
    out = dict(PRODUCTION)
    if stage_widths and any(w > 1 for w in stage_widths):
        out.update(PRODUCTION_WIDE)
    for k, v in (overrides or {}).items():
        if v is not None:
            out[k] = v
    return out


def deviations(fields: dict) -> list:
    """Which production flags this spec turned off, for the run log."""
    return [
        f"{k}={fields[k]!r} (production: {v!r})"
        for k, v in PRODUCTION.items()
        if k in fields and fields[k] != v
    ]
