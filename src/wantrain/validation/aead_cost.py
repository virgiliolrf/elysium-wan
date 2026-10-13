"""The AEAD sealing cost against the step (BA-44, RUNTIME-I16, TRANSPORT-I7).

Runs the toy 2-stage chain and a 1-2-1 wide run in local mode. Per
configuration: one discarded sealed warm-up run (it keeps the configuration's
own tag), then k pairs of the same recipe and seed: a SEALED run (production:
every data-plane socket sealed) and an UNSEALED run (the record layer swapped
for a plain passthrough, inside the slot processes this tool launches and
nowhere else). Which arm of a pair runs first is drawn from --seed, balanced
(ceil(k/2) pairs one way, floor(k/2) the other), and recorded.

    python -m wantrain.validation.aead_cost --steps 20 [--pairs 21] [--max-frac 0.01]

Run time T: a slot's step time is the interval between the ends of two
consecutive optimizer steps (`step_opt` t1 in its telemetry), so it covers the
whole step loop. The first --warmup steps (default 2) of every run are left out:
step 0 carries the process warm-up (first touch, lazy init; 60-310 ms against a
~50 ms step measured on this Mac) and step 1 is occasionally still high. A
slot's time is the median of its remaining intervals, robust to the rare gen-2
GC pause (32-48 ms; one per process in 600 steps in both arms, measured) whose
place in a short run follows the allocation history, not the sealing work (in
the 2-stage chain it falls inside a 20-step sealed run and outside the unsealed
one, +4% on a mean); T is the slowest slot's. The mean-based ratios are
reported too. Both arms start through the same shim (`_arm_stage_main`).

Gated metric (overlap-aware): per pair, ratio = (T_sealed - T_unsealed) /
T_unsealed; a configuration's cost is the median over its pairs, reported with a
distribution-free order-statistic interval of that median (coverage >= 95% from
k = 6 on, its exact coverage recorded), and overlap_cost_frac is the max of the
medians over configurations. `ok` is overlap_cost_frac < max-frac, with every
slot of every run reporting, the sealed runs sealing, the unsealed runs not, and
both arms' losses bit-identical.

Also reported, unbounded: the serialized sum (seal_ns + open_ns) / steps_wall_ns
of every slot of every measured sealed run (label MEASURED) and its max,
serialized_cost_frac.

Prints `OVERLAP_COST_FRAC=<v>`, one `OVERLAP_CI` line per configuration and
`SERIALIZED_COST_FRAC=<v>`, then one JSON object on one line, and exits 1
unless ok. The fleet is localhost on this machine's CPU, so the JSON is
labelled SIMULATED.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import runpy
import shlex
import shutil
import socket
import statistics
import sys
import tempfile

from wantrain.launch.local import run_local
from wantrain.model.gpt import make_config
from wantrain.runtime.spec import JobSpec
from wantrain.validation.compare import compare

CASES: dict[str, dict] = {
    "chain2": dict(stages=2),
    "wide_1_2_1": dict(stages=3, stage_widths=[1, 2, 1]),
}
MIN_PAIRS = 5
DEFAULT_PAIRS = 21
WARMUP_STEPS = 2
SEED = 20261009
CI_LEVEL = 0.95
SEALED, UNSEALED = "sealed", "unsealed"
_AEAD = re.compile(
    r"^WT_AEAD rank=(\d+) peer=(\d+) seal_ns=(\d+) open_ns=(\d+) bytes_out=(\d+) "
    r"bytes_in=(\d+) steps=(\d+) steps_wall_ns=(\d+)\s*$"
)
_FIELDS = ("rank", "peer", "seal_ns", "open_ns", "bytes_out", "bytes_in", "steps", "steps_wall_ns")
# A run binds port_base + 8 * stage + peer; each run gets its own block.
_BLOCK = 32
_STAGE_MAIN = "wantrain.worker.stage_main"


class PlainSocket:
    """The UNSEALED arm's stand-in for SecureSocket: the same interface over
    the raw socket, no record layer. Exists only in the slot processes this
    tool launches through `_arm_stage_main`; production never sees it."""

    def __init__(self, sock: socket.socket, cipher: str, *_keys, **_limits):
        self.cipher = cipher
        self._sock = sock

    def sendall(self, data) -> None:
        self._sock.sendall(data)

    def recv(self, bufsize: int) -> bytes:
        return self._sock.recv(bufsize)

    def recv_into(self, buffer, nbytes: int = 0) -> int:
        return self._sock.recv_into(buffer, nbytes)

    def shutdown(self, how: int) -> None:
        self._sock.shutdown(how)

    def close(self) -> None:
        self._sock.close()

    def settimeout(self, value: float | None) -> None:
        self._sock.settimeout(value)

    def setsockopt(self, *args) -> None:
        self._sock.setsockopt(*args)

    def fileno(self) -> int:
        return self._sock.fileno()

    def stats(self) -> dict[str, int]:
        # Zero seal/open time is how the tool checks the swap took effect.
        return dict(records_out=0, records_in=0, bytes_out=0, bytes_in=0, seal_ns=0, open_ns=0)


def _arm_stage_main(arm: str) -> None:
    """Entry of a measured slot: `python -c <this> -m wantrain.worker.stage_main
    --spec ... --rank r --peer p`. Both arms start through this same path, so
    the only difference between them is the swap: an UNSEALED slot replaces the
    record layer, a SEALED slot keeps production's. Then stage_main runs
    exactly as run_local would."""
    if sys.argv[1:3] != ["-m", _STAGE_MAIN] or arm not in (SEALED, UNSEALED):
        raise SystemExit(f"aead_cost shim: unexpected arm {arm!r} or argv {sys.argv!r}")
    if arm == UNSEALED:
        import wantrain.transport.secure as secure

        secure.SecureSocket = PlainSocket  # type: ignore[misc,assignment]
    sys.argv = [_STAGE_MAIN] + sys.argv[3:]
    runpy.run_module(_STAGE_MAIN, run_name="__main__", alter_sys=True)


def _arm_python(root: str, arm: str) -> str:
    """An executable that run_local launches in place of the interpreter."""
    path = os.path.join(root, f"python_{arm}")
    code = f"from wantrain.validation.aead_cost import _arm_stage_main; _arm_stage_main({arm!r})"
    with open(path, "w") as f:
        f.write(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(code)} "$@"\n')
    # Only this measurement's own process runs it, as the owner: the minimum is
    # owner read, write and execute (bandit B103 flagged the former 0o755).
    os.chmod(path, 0o700)
    return path


def _free_base(n: int) -> int:
    """A base whose next n ports bind on localhost now (best effort)."""
    rng = random.SystemRandom()
    for _ in range(200):
        base = rng.randrange(20000, 28000)
        try:
            for port in range(base, base + n):
                with socket.socket() as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", port))
        except OSError:
            continue
        return base
    raise RuntimeError(f"no block of {n} free ports found")


def slot_report(log_path: str) -> dict | None:
    """The last WT_AEAD line of one slot's log as a dict, or None."""
    found = None
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                m = _AEAD.match(line)
                if m:
                    found = dict(zip(_FIELDS, map(int, m.groups())))
    except FileNotFoundError:
        return None
    return found


def slot_ciphers(spec: JobSpec, r: int, p: int) -> set:
    """The ciphers this slot's p2p_auth events name."""
    path = os.path.join(spec.out_dir, f"{spec.tag}_rank{r}p{p}_telemetry.jsonl")
    out = set()
    try:
        with open(path) as f:
            for line in f:
                ev = json.loads(line)
                if ev.get("ev") == "p2p_auth":
                    out.add(ev.get("cipher"))
    except FileNotFoundError:
        pass
    return out


def slot_step_ms(spec: JobSpec, r: int, p: int, warmup: int) -> list[float]:
    """The slot's step times in ms: the intervals between the ends of
    consecutive optimizer steps, for the steps from `warmup` (>= 1) on."""
    path = os.path.join(spec.out_dir, f"{spec.tag}_rank{r}p{p}_telemetry.jsonl")
    ends: dict[int, float] = {}
    try:
        with open(path) as f:
            for line in f:
                ev = json.loads(line)
                if ev.get("ev") == "step_opt":
                    ends[int(ev["step"])] = float(ev["t1"])
    except FileNotFoundError:
        return []
    return [
        (ends[s] - ends[s - 1]) * 1e3 for s in sorted(ends) if s >= max(1, warmup) and s - 1 in ends
    ]


def median_ci(xs: list[float], level: float = CI_LEVEL) -> dict | None:
    """Distribution-free interval of the median: [x(j), x(k+1-j)] for the
    largest j whose coverage 1 - 2 P(Bin(k, 1/2) < j) is >= level (j = 1,
    with its lower coverage, when k is too small to reach it)."""
    k = len(xs)
    if not k:
        return None
    xs = sorted(xs)

    def coverage(j: int) -> float:
        return 1.0 - 2.0 * sum(math.comb(k, i) for i in range(j)) / 2.0**k

    j = 1
    while j + 1 <= (k + 1) // 2 and coverage(j + 1) >= level:
        j += 1
    return {"lo": xs[j - 1], "hi": xs[k - j], "coverage": coverage(j)}


def overlap_ratios(t_sealed: list[float], t_unsealed: list[float]) -> list[float]:
    """Per pair i: (T_sealed_i - T_unsealed_i) / T_unsealed_i."""
    if len(t_sealed) != len(t_unsealed):
        raise ValueError("every pair needs one sealed and one unsealed run")
    return [(s - u) / u for s, u in zip(t_sealed, t_unsealed)]


def summarize(
    configs: list,
    per_slot: list,
    missing: list,
    invalid: list,
    ciphers: set,
    hw_aes: bool,
    max_frac: float,
) -> dict:
    """configs: one {run, ratios, ...} per configuration; per_slot: the sealed
    runs' serialized rows. ok is the overlap bound on valid measurements."""
    for c in configs:
        c["median"] = statistics.median(c["ratios"]) if c["ratios"] else None
        c["overlap_ci"] = median_ci(c["ratios"])
    medians = [c["median"] for c in configs]
    overlap = max(medians) if medians and None not in medians else None
    serialized = max((s["serialized_cost_frac"] for s in per_slot), default=None)
    return {
        "label": "SIMULATED",
        "cipher": sorted(ciphers)[0] if len(ciphers) == 1 else sorted(ciphers),
        "hw_aes": hw_aes,
        "configs": configs,
        "overlap_cost_frac": overlap,
        "max_frac": max_frac,
        "per_slot": per_slot,
        "serialized_label": "MEASURED",
        "serialized_cost_frac": serialized,
        "missing": missing,
        "invalid": invalid,
        "ok": (
            not missing
            and not invalid
            and bool(per_slot)
            and overlap is not None
            and all(len(c["ratios"]) >= MIN_PAIRS for c in configs)
            and overlap < max_frac
        ),
    }


def run_arm(
    name: str,
    kw: dict,
    arm: str,
    tag: str,
    steps: int,
    warmup: int,
    out_dir: str,
    python: str,
):
    """One local run. -> ((T, mean-based T) in ms or None, per-slot reports, missing, ciphers)."""
    spec = JobSpec(
        model=make_config("toy"),
        steps=steps,
        microbatches=8,
        out_dir=out_dir,
        tag=tag,
        port_base=_free_base(_BLOCK),
        dial_timeout_s=60,
        **kw,
    )
    res = run_local(spec, timeout=600, python=python)
    reps, missing, ciphers, med, mean = [], [], set(), [], []
    for r, w in enumerate(spec.stage_widths):
        for p in range(w):
            rep = slot_report(os.path.join(out_dir, f"{tag}_r{r}p{p}.log"))
            ms = slot_step_ms(spec, r, p, warmup)
            if (
                rep is None
                or not res["ok"]
                or rep["steps_wall_ns"] <= 0
                or rep["steps"] < 1
                or len(ms) != steps - max(1, warmup)
            ):
                missing.append({"run": name, "arm": arm, "tag": tag, "rank": r, "peer": p})
                continue
            if arm == SEALED:
                ciphers |= slot_ciphers(spec, r, p)
            reps.append(rep)
            med.append(statistics.median(ms))
            mean.append(statistics.fmean(ms))
    t = None if missing or not med else (max(med), max(mean))
    return t, reps, missing, ciphers


def run_config(
    name: str,
    kw: dict,
    steps: int,
    pairs: int,
    out_dir: str,
    pythons: dict[str, str],
    seed: int = SEED,
    warmup: int = WARMUP_STEPS,
):
    """A discarded sealed warm-up run under the configuration's own tag (so its
    logs sit where they always did), then k pairs in a seeded, balanced order."""
    rng = random.Random(f"{seed}:{name}")
    sealed_first = [i < (pairs + 1) // 2 for i in range(pairs)]
    rng.shuffle(sealed_first)
    t_arm: dict[str, list[float]] = {SEALED: [], UNSEALED: []}
    t_mean: dict[str, list[float]] = {SEALED: [], UNSEALED: []}
    rows, missing, invalid, ciphers, order = [], [], [], set(), []

    def check(arm: str, tag: str, reps: list) -> None:
        for rep in reps:
            if arm == UNSEALED and rep["seal_ns"] + rep["open_ns"]:
                invalid.append({"run": name, "tag": tag, "why": "unsealed slot sealed", **rep})
            if arm == SEALED and not (rep["seal_ns"] and rep["open_ns"]):
                invalid.append({"run": name, "tag": tag, "why": "sealed slot not sealed", **rep})

    _t, reps, miss, ciph = run_arm(name, kw, SEALED, name, steps, warmup, out_dir, pythons[SEALED])
    missing += miss
    ciphers |= ciph
    check(SEALED, name, reps)
    for i in range(pairs):
        arms = (SEALED, UNSEALED) if sealed_first[i] else (UNSEALED, SEALED)
        order.append(list(arms))
        losses, t_pair = {}, {}
        for arm in arms:
            tag = f"{name}_{arm}{i}"
            t_pair[arm], reps, miss, ciph = run_arm(
                name, kw, arm, tag, steps, warmup, out_dir, pythons[arm]
            )
            missing += miss
            ciphers |= ciph
            losses[arm] = os.path.join(out_dir, f"{tag}_losses.json")
            check(arm, tag, reps)
            if arm == SEALED:
                for rep in reps:
                    rows.append(
                        {
                            "label": "MEASURED",
                            "run": name,
                            "pair": i,
                            "rank": rep["rank"],
                            "peer": rep["peer"],
                            "serialized_cost_frac": (rep["seal_ns"] + rep["open_ns"])
                            / rep["steps_wall_ns"],
                            "seal_ms": rep["seal_ns"] / 1e6,
                            "open_ms": rep["open_ns"] / 1e6,
                            "step_ms": rep["steps_wall_ns"] / 1e6 / rep["steps"],
                            "bytes_out": rep["bytes_out"],
                            "bytes_in": rep["bytes_in"],
                        }
                    )
        # A pair with a failed run measures nothing (its slots are in `missing`).
        if t_pair[SEALED] is None or t_pair[UNSEALED] is None:
            continue
        for arm in (SEALED, UNSEALED):
            t_arm[arm].append(t_pair[arm][0])
            t_mean[arm].append(t_pair[arm][1])
        cmp = compare(losses[SEALED], losses[UNSEALED])
        if not cmp["ok"]:
            invalid.append({"run": name, "pair": i, "why": "arms' losses differ", **cmp})
    config = {
        "run": name,
        "k": len(t_arm[SEALED]),
        "seed": seed,
        "warmup_steps": warmup,
        "order": order,
        "t_sealed_ms": t_arm[SEALED],
        "t_unsealed_ms": t_arm[UNSEALED],
        "ratios": overlap_ratios(t_arm[SEALED], t_arm[UNSEALED]),
        "ratios_mean": overlap_ratios(t_mean[SEALED], t_mean[UNSEALED]),
    }
    return config, rows, missing, invalid, ciphers


def _fmt(v: float | None) -> str:
    return "nan" if v is None else f"{v:.6f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    ap.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-frac", type=float, default=0.01)
    ap.add_argument("--out-dir", default=None, help="keep the runs here (default: a temp dir)")
    args = ap.parse_args(argv)
    if args.warmup < 1:
        ap.error("--warmup must be >= 1 (step 0 has no previous step end)")
    if args.steps <= args.warmup:
        ap.error("--steps must be > --warmup")
    if args.pairs < MIN_PAIRS:
        ap.error(f"--pairs must be >= {MIN_PAIRS}")
    from wantrain.transport.secure import hardware_aes

    root = args.out_dir or tempfile.mkdtemp(prefix="aead_cost_")
    os.makedirs(root, exist_ok=True)
    configs, per_slot, missing, invalid, ciphers = [], [], [], [], set()
    try:
        pythons = {arm: _arm_python(root, arm) for arm in (SEALED, UNSEALED)}
        for name, kw in CASES.items():
            cfg, rows, miss, inv, ciph = run_config(
                name,
                kw,
                args.steps,
                args.pairs,
                os.path.join(root, name),
                pythons,
                args.seed,
                args.warmup,
            )
            configs.append(cfg)
            per_slot += rows
            missing += miss
            invalid += inv
            ciphers |= ciph
    finally:
        if args.out_dir is None:
            shutil.rmtree(root, ignore_errors=True)
    out = summarize(configs, per_slot, missing, invalid, ciphers, hardware_aes(), args.max_frac)
    print(f"OVERLAP_COST_FRAC={_fmt(out['overlap_cost_frac'])}")
    for c in out["configs"]:
        ci = c["overlap_ci"] or {}
        print(
            f"OVERLAP_CI {c['run']} k={c['k']} median={_fmt(c['median'])} "
            f"lo={_fmt(ci.get('lo'))} hi={_fmt(ci.get('hi'))} coverage={_fmt(ci.get('coverage'))}"
        )
    print(f"SERIALIZED_COST_FRAC={_fmt(out['serialized_cost_frac'])}")
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
