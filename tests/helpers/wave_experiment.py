"""Discriminating experiment for M2-P2-NO-COMMON-WAVE (sampling vs seqlock).

The dead-worker heal of tests/chaos/test_recovery.py sometimes finds no checkpoint
wave held for every slot. Two hypotheses (agents-architetura/docs/findings.json):
SAMPLING -- the agent samples the shard every 0.5 s and never sees some saved waves;
SEQLOCK -- the shard/sidecar seqlock makes the shipper drop waves it did see
(race-skip). The agent prints one line per supervise pass (SEEN_RE) and one per ship
decision (SHIP_RE); this script runs the unmodified test, fleet by fleet, and
classifies every symptomatic run by the rule in VERDICT_RULE, fixed before any run.

Modes (one JSON file per arm, one row appended as soon as each run ends):
  tree arm   --until-symptoms K --max-runs N --out F   snapshot this tree, run from it
  smoke      --runs N --out F                          the instrumentation is there
  fixed      --runs N --expect-fixed [--out F]         every saved wave shipped, no gap
  base arm   --base-tree DIR --until-symptoms K --max-runs N [--out F] [--expect-symptom]
  check      --check F [--require-verdict V1,V2]       recompute from the recorded rows
Exit codes: 0 finished, 1 failed or refused, 2 unfinished (budget spent: call again
with the same arguments), 3 the base tree is missing.

Also a pytest plugin (-p helpers.wave_experiment): with HEAL_ENV set it records, at
the coordinator's first restore-and-relaunch, the store index, the common step and
the byte size of every agent and stage log, which splits the logs at the heal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SHIP_RE = re.compile(r"^\[agent (\S+)\] wave r(\d+)p(\d+) step=(\d+) (shipped|race-skip)$")
GAP_RE = re.compile(r"^\[agent (\S+)\] ship gap r(\d+)p(\d+) step=(\d+)$")
SEEN_RE = re.compile(r"^\[agent (\S+)\] wave r(\d+)p(\d+) seen=(\d+|none)$")
STEP_RE = re.compile(r"^WT_STEP (\d+)$")
STAGE_LOG_RE = re.compile(r"^stage-r(\d+)p(\d+)\.log$")
SYMPTOM = "no checkpoint wave is held for every slot"
TEST_ID = "tests/chaos/test_recovery.py::test_dead_worker_is_replaced_and_the_run_stays_exact"
# The test's recipe saves after step s when (s + 1) % ckpt_every == 0 (stage.py).
CKPT_EVERY = 2
HEAL_ENV = "WAVEEXP_HEAL_OUT"
LOGS_ENV = "WAVEEXP_LOGS_ROOT"
SCHEMA = "wave-exp/1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join(ROOT, ".factory", "runs", "M2")

VERDICT_RULE = (
    "Over symptomatic runs only (a heal with common_step None), for each step saved by "
    "every slot before the heal but not held for every slot in the coordinator's store at "
    "the heal: 'lost-after-ship' when every slot shipped it; otherwise, per slot that did "
    "not ship it, 'race' when that slot race-skipped it, 'unseen' when that slot never saw "
    "it, 'seen-unshipped' when it saw it and neither shipped nor race-skipped it; a "
    "symptomatic run with no such step is 'no-evidence'. OTHER when any lost-after-ship, "
    "seen-unshipped or no-evidence occurs; else INCONCLUSIVE when fewer than K symptomatic "
    "runs; else SAMPLING when every kind is unseen, SEQLOCK when every kind is race, MIXED "
    "when both occur. p0 = symptomatic runs / runs made."
)
OTHER_KINDS = {"lost-after-ship", "seen-unshipped", "no-evidence"}


# ---------------------------------------------------------------- pytest plugin
def pytest_configure(config):
    out = os.environ.get(HEAL_ENV)
    if not out:
        return
    import wantrain
    from wantrain.coordinator import coordinator as co

    logs_root = os.environ.get(LOGS_ENV, "")
    orig = co.Coordinator._restore_and_relaunch

    def recorded(self, run, step, spec_json):
        if not os.path.exists(out):
            rec = {
                "wantrain": os.path.abspath(wantrain.__file__),
                "slots": [_slot_key(s) for s in run.slots],
                "store_index": {_slot_key(s): v for s, v in run.store.index().items()},
                "common_step": run.store.common_step(list(run.slots)),
                "step": step,
                "heals": [dict(h) for h in run.heals],
                "restarts": run.restarts,
                "log_sizes": _log_sizes(logs_root) if logs_root else {},
            }
            tmp = out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rec, f)
            os.replace(tmp, out)
        return orig(self, run, step, spec_json)

    co.Coordinator._restore_and_relaunch = recorded


def _slot_key(slot) -> str:
    r, p = slot
    return f"r{int(r)}p{int(p)}"


def _log_paths(root: str) -> list:
    out = []
    for d, _, names in os.walk(root):
        for n in names:
            if n == "agent.log" or STAGE_LOG_RE.match(n):
                out.append(os.path.join(d, n))
    return sorted(out)


def _log_sizes(root: str) -> dict:
    sizes = {}
    for p in _log_paths(root):
        try:
            sizes[os.path.relpath(p, root)] = os.path.getsize(p)
        except OSError:
            pass
    return sizes


# ---------------------------------------------------------------- tree identity
def tree_hashes(root: str) -> dict:
    """sha256 of every <root>/src/wantrain/**/*.py, keyed by its path under root."""
    base = os.path.join(root, "src", "wantrain")
    out = {}
    for d, dirs, names in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for n in sorted(names):
            if n.endswith(".py"):
                p = os.path.join(d, n)
                with open(p, "rb") as f:
                    out[os.path.relpath(p, root)] = hashlib.sha256(f.read()).hexdigest()
    return out


def make_snapshot(src_root: str, snap: str) -> None:
    if os.path.exists(snap):
        shutil.rmtree(snap)
    os.makedirs(snap)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(src_root, d), os.path.join(snap, d), ignore=ignore)
    shutil.copy2(os.path.join(src_root, "pyproject.toml"), os.path.join(snap, "pyproject.toml"))


# ---------------------------------------------------------------- one fleet
def _kill_tree(proc, tag: str) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    # Agents run in their own sessions; their command lines name the run's temp dir.
    subprocess.run(["pkill", "-9", "-f", tag], check=False)


def run_fleet(tree: str, plugin: bool, timeout_s: float) -> tuple:
    """Run the unmodified dead-worker test once from `tree`. Returns
    (rc or None on timeout, output, temp dir, heal record or None)."""
    tmp = tempfile.mkdtemp(prefix="wave-exp-")
    bt = os.path.join(tmp, "bt")
    heal = os.path.join(tmp, "heal.json")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    paths = [os.path.join(tree, "src")] + ([os.path.join(tree, "tests")] if plugin else [])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.pop(HEAL_ENV, None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:randomly"]
    cmd += ["-p", "no:cacheprovider", "--rootdir", tree, "--basetemp", bt]
    if plugin:
        env[HEAL_ENV], env[LOGS_ENV] = heal, bt
        cmd += ["-p", "helpers.wave_experiment"]
    cmd.append(TEST_ID)
    proc = subprocess.Popen(
        cmd,
        cwd=tree,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        raw, _ = proc.communicate(timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc, tmp)
        raw, _ = proc.communicate()
        rc = None
    _kill_tree(proc, tmp)
    rec = None
    if os.path.exists(heal):
        with open(heal) as f:
            rec = json.load(f)
    return rc, raw.decode("utf-8", "replace"), tmp, rec


def _segments(path: str, size_at_heal: int | None) -> tuple:
    with open(path, "rb") as f:
        data = f.read()
    cut = 0 if size_at_heal is None else size_at_heal
    return data[:cut].decode("utf-8", "replace"), data.decode("utf-8", "replace")


def _new_slot() -> dict:
    return {
        "saved": set(),
        "seen": set(),
        "seen_none": 0,
        "shipped": set(),
        "race_skip": set(),
        "seen_lines": 0,
        "ship_lines": 0,
        "gaps": [],
    }


def parse_fleet(bt: str, rec: dict | None) -> dict:
    """Per slot: before the heal (saved, seen, shipped, race-skipped) and over the
    whole run (seen and SHIP_RE line counts, GAP_RE steps)."""
    sizes = (rec or {}).get("log_sizes", {})
    slots: dict = {}
    for p in _log_paths(bt):
        rel = os.path.relpath(p, bt)
        before, whole = _segments(p, sizes.get(rel) if rec else None)
        m = STAGE_LOG_RE.match(os.path.basename(p))
        if m:
            s = slots.setdefault(f"r{m.group(1)}p{m.group(2)}", _new_slot())
            for line in before.splitlines():
                st = STEP_RE.match(line)
                if st and (int(st.group(1)) + 1) % CKPT_EVERY == 0:
                    s["saved"].add(int(st.group(1)))
            continue
        for part, text in (("before", before), ("whole", whole)):
            for line in text.splitlines():
                g = SEEN_RE.match(line)
                if g:
                    s = slots.setdefault(f"r{g.group(2)}p{g.group(3)}", _new_slot())
                    if part == "whole":
                        s["seen_lines"] += 1
                    elif g.group(4) == "none":
                        s["seen_none"] += 1
                    else:
                        s["seen"].add(int(g.group(4)))
                    continue
                g = SHIP_RE.match(line)
                if g:
                    s = slots.setdefault(f"r{g.group(2)}p{g.group(3)}", _new_slot())
                    if part == "whole":
                        s["ship_lines"] += 1
                    else:
                        key = "shipped" if g.group(5) == "shipped" else "race_skip"
                        s[key].add(int(g.group(4)))
                    continue
                g = GAP_RE.match(line)
                if g and part == "whole":
                    s = slots.setdefault(f"r{g.group(2)}p{g.group(3)}", _new_slot())
                    s["gaps"].append(int(g.group(4)))
    out = {}
    for k, s in sorted(slots.items()):
        # A step the agent read from the sidecar was saved, whatever the stage log says.
        s["saved"] |= s["seen"]
        out[k] = {x: (sorted(v) if isinstance(v, set) else v) for x, v in s.items()}
    return out


def tree_row(i: int, tree: str, timeout_s: float, keep_dir: str) -> dict:
    load = list(os.getloadavg())
    t0 = time.time()
    rc, output, tmp, rec = run_fleet(tree, True, timeout_s)
    row: dict = {
        "run": i,
        "t_start": t0,
        "loadavg": load,
        "duration_s": round(time.time() - t0, 2),
        "rc": rc,
        "symptom_line": SYMPTOM in output,
        "heal": rec is not None,
        "common_step": None,
        "resumed_from_step": None,
        "store_index": {},
        "slots": {},
        "error": None,
    }
    try:
        bt = os.path.join(tmp, "bt")
        row["slots"] = parse_fleet(bt, rec)
        if rec is None:
            row["error"] = "timeout" if rc is None else f"no heal recorded (rc={rc})"
        elif not rec["wantrain"].startswith(os.path.join(tree, "src") + os.sep):
            row["error"] = f"wantrain imported from {rec['wantrain']}, not the measured tree"
        else:
            heals = rec["heals"]
            row["common_step"] = rec["common_step"]
            row["resumed_from_step"] = heals[-1]["resumed_from_step"] if heals else None
            row["store_index"] = rec["store_index"]
            row["run_slots"] = rec["slots"]
            if len(heals) != 1 or rec["restarts"]:
                row["error"] = f"expected one heal and no restart: {heals} {rec['restarts']}"
        if row["error"] or is_symptomatic(row, "tree"):
            _keep(tmp, output, os.path.join(keep_dir, f"run-{i:03d}"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return row


def base_row(i: int, tree: str, timeout_s: float, keep_dir: str) -> dict:
    load = list(os.getloadavg())
    t0 = time.time()
    rc, output, tmp, _ = run_fleet(tree, False, timeout_s)
    row = {
        "run": i,
        "t_start": t0,
        "loadavg": load,
        "duration_s": round(time.time() - t0, 2),
        "rc": rc,
        "symptom_line": SYMPTOM in output,
        "error": "timeout" if rc is None else None,
    }
    try:
        if row["error"] or row["symptom_line"]:
            _keep(tmp, output, os.path.join(keep_dir, f"run-{i:03d}"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return row


def _keep(tmp: str, output: str, dst: str) -> None:
    """Evidence of a symptomatic or failed run: pytest output and every log."""
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "pytest.out"), "w") as f:
        f.write(output)
    bt = os.path.join(tmp, "bt")
    for p in _log_paths(bt):
        rel = os.path.relpath(p, bt).replace(os.sep, "__")
        shutil.copy2(p, os.path.join(dst, rel))
    heal = os.path.join(tmp, "heal.json")
    if os.path.exists(heal):
        shutil.copy2(heal, os.path.join(dst, "heal.json"))


# ---------------------------------------------------------------- verdicts
def is_symptomatic(row: dict, mode: str) -> bool:
    if mode == "base":
        return bool(row.get("symptom_line"))
    return bool(row.get("heal")) and not row.get("error") and row.get("common_step") is None


def run_kinds(row: dict) -> dict:
    """step -> {slot: kind} for a symptomatic run (VERDICT_RULE); {} is no-evidence."""
    slots = row.get("run_slots") or sorted(row["slots"])
    data = {k: row["slots"].get(k, {}) for k in slots}
    saved = set.intersection(*[set(d.get("saved", [])) for d in data.values()])
    held = set.intersection(*[set(row["store_index"].get(k, [])) for k in slots])
    out: dict = {}
    for step in sorted(saved - held):
        missing = [k for k in slots if step not in data[k].get("shipped", [])]
        if not missing:
            out[str(step)] = {k: "lost-after-ship" for k in slots}
            continue
        per = out[str(step)] = {}
        for k in missing:
            if step in data[k].get("race_skip", []):
                per[k] = "race"
            elif step not in data[k].get("seen", []):
                per[k] = "unseen"
            else:
                per[k] = "seen-unshipped"
    return out


def smoke_failures(row: dict) -> list:
    if row.get("error"):
        return [row["error"]]
    slots = row.get("run_slots") or []
    if not slots:
        return ["no slots recorded"]
    bad = []
    for k in slots:
        s = row["slots"].get(k, {})
        if s.get("seen_lines", 0) < 1:
            bad.append(f"{k}: no 'seen' line")
        if s.get("ship_lines", 0) < 1:
            bad.append(f"{k}: no SHIP_RE line")
    return bad


def fixed_failures(row: dict) -> list:
    if row.get("error"):
        return [row["error"]]
    bad = []
    if row.get("common_step") is None:
        bad.append("common_step None at the heal")
    for k in row.get("run_slots") or []:
        s = row["slots"].get(k, {})
        unshipped = sorted(set(s.get("saved", [])) - set(s.get("shipped", [])))
        if unshipped:
            bad.append(f"{k}: saved before the kill, never shipped: {unshipped}")
        if s.get("gaps"):
            bad.append(f"{k}: ship gap at {s['gaps']}")
    return bad


def evaluate(doc: dict) -> dict:
    mode, params, rows = doc["mode"], doc["params"], doc["rows"]
    n = len(rows)
    sym = [r for r in rows if is_symptomatic(r, mode)]
    res: dict = {
        "verdict": None,
        "p0": (len(sym) / n) if n else 0.0,
        "runs": n,
        "symptomatic": len(sym),
        "errors": sum(1 for r in rows if r.get("error")),
    }
    if mode == "tree":
        k = params["until_symptoms"]
        kinds = set()
        res["detail"] = {}
        for r in sym:
            per_step = res["detail"][str(r["run"])] = run_kinds(r)
            kinds.update(k for per in per_step.values() for k in per.values())
            if not per_step:
                kinds.add("no-evidence")
        res["kinds"] = sorted(kinds)
        if kinds & OTHER_KINDS:
            res["verdict"] = "OTHER"
        elif len(sym) < k:
            res["verdict"] = "INCONCLUSIVE"
        elif kinds == {"unseen"}:
            res["verdict"] = "SAMPLING"
        elif kinds == {"race"}:
            res["verdict"] = "SEQLOCK"
        else:
            res["verdict"] = "MIXED"
        res["finished"] = len(sym) >= k or n >= params["max_runs"]
    elif mode == "base":
        res["verdict"] = "SYMPTOM" if sym else "NO-SYMPTOM"
        res["finished"] = len(sym) >= params["until_symptoms"] or n >= params["max_runs"]
    else:
        check = smoke_failures if mode == "smoke" else fixed_failures
        fails = {str(r["run"]): check(r) for r in rows}
        res["failures"] = {i: f for i, f in fails.items() if f}
        names = ("SMOKE-OK", "SMOKE-FAIL") if mode == "smoke" else ("FIXED", "NOT-FIXED")
        res["verdict"] = names[1] if res["failures"] else names[0]
        res["finished"] = n >= params["runs"]
    return res


# ---------------------------------------------------------------- driver
def _write(path: str, doc: dict) -> None:
    doc["result"] = evaluate(doc)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _stem(out: str) -> str:
    return out[:-5] if out.endswith(".json") else out


def drive(args) -> int:
    if args.base_tree:
        mode = "base"
        tree = os.path.abspath(args.base_tree)
        if not os.path.isdir(os.path.join(tree, "src")):
            print(f"wave_experiment: base tree {tree}/src is missing; export it first", flush=True)
            return 3
        out = args.out or tree.rstrip(os.sep) + ".base.json"
    else:
        mode = "fixed" if args.expect_fixed else ("smoke" if args.runs else "tree")
        out = args.out or os.path.join(DEFAULT_OUT_DIR, f"wave-exp-{mode}.json")
    out = os.path.abspath(out)
    if mode in ("tree", "base"):
        params = {"until_symptoms": args.until_symptoms, "max_runs": args.max_runs}
    else:
        params = {"runs": args.runs}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stem = _stem(out)
    live = tree_hashes(tree if mode == "base" else ROOT)
    doc: dict | None = None
    if os.path.exists(out):
        with open(out) as f:
            old = json.load(f)
        if old.get("mode") != mode or old.get("params") != params:
            print(
                f"wave_experiment: {out} holds another arm ({old.get('mode')}, "
                f"{old.get('params')}); refusing",
                flush=True,
            )
            return 1
        finished = evaluate(old)["finished"]
        if finished and mode in ("tree", "base"):
            print(f"wave_experiment: {out} is finished; --check it or pass another --out")
            return 1
        if not finished:
            root = old["tree"] if mode == "base" else old["snapshot"]
            if old["hashes"] != live or tree_hashes(root) != old["hashes"]:
                print(
                    f"wave_experiment: {out} was recorded on another tree; refusing to mix "
                    "runs (remove it or pass another --out)",
                    flush=True,
                )
                return 1
            doc = old
    if doc is None:
        doc = {"schema": SCHEMA, "mode": mode, "params": params, "test": TEST_ID}
        doc["verdict_rule"] = VERDICT_RULE
        if mode == "base":
            doc["tree"], doc["hashes"] = tree, live
        else:
            snap = stem + ".tree"
            make_snapshot(ROOT, snap)
            doc["snapshot"], doc["hashes"] = snap, tree_hashes(snap)
            if doc["hashes"] != live:
                print("wave_experiment: the snapshot differs from the tree it copied", flush=True)
                return 1
        doc["rows"] = []
        _write(out, doc)
    tree = doc["tree"] if mode == "base" else doc["snapshot"]
    keep = stem + ".logs"
    t0 = time.monotonic()
    while not evaluate(doc)["finished"]:
        if time.monotonic() - t0 > args.budget_s:
            r = evaluate(doc)
            print(
                f"wave_experiment: unfinished after {args.budget_s:.0f}s "
                f"({r['runs']} runs, {r['symptomatic']} symptomatic); call again",
                flush=True,
            )
            return 2
        i = len(doc["rows"])
        row = (base_row if mode == "base" else tree_row)(i, tree, args.run_timeout_s, keep)
        doc["rows"].append(row)
        _write(out, doc)
        print(
            f"wave_experiment: run {i} rc={row['rc']} symptom={is_symptomatic(row, mode)} "
            + (f"common_step={row['common_step']} " if "common_step" in row else "")
            + f"error={row['error']} "
            f"{row['duration_s']}s",
            flush=True,
        )
    res = evaluate(doc)
    print(json.dumps({k: res[k] for k in ("verdict", "p0", "runs", "symptomatic")}), flush=True)
    if mode in ("smoke", "fixed"):
        return 0 if res["verdict"] in ("SMOKE-OK", "FIXED") else 1
    if mode == "base" and args.expect_symptom and not res["symptomatic"]:
        return 1
    return 0


def check(path: str, require: list) -> int:
    try:
        with open(path) as f:
            doc = json.load(f)
        assert doc.get("schema") == SCHEMA, f"schema is {doc.get('schema')!r}"
        mode = doc["mode"]
        assert mode in ("tree", "base", "smoke", "fixed"), f"mode {mode!r}"
        assert isinstance(doc["rows"], list) and isinstance(doc["params"], dict)
        assert doc.get("verdict_rule") == VERDICT_RULE, (
            "the verdict rule differs from this script's"
        )
        keys = {"run", "loadavg", "rc", "symptom_line", "error"}
        if mode != "base":
            keys |= {"heal", "common_step", "resumed_from_step", "store_index", "slots"}
        for i, r in enumerate(doc["rows"]):
            assert r.get("run") == i and keys <= set(r), f"row {i} lacks {sorted(keys - set(r))}"
        root = doc["tree"] if mode == "base" else doc["snapshot"]
        got = tree_hashes(root)
        assert got and got == doc["hashes"], f"the hashes of {root} differ from the recorded ones"
        res = evaluate(doc)
        assert res["finished"], f"unfinished ({res['runs']} runs)"
        assert doc.get("result") == res, "the recorded result differs from the recomputed one"
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as e:
        print(f"wave_experiment: {path} is not a consistent finished file: {e}", flush=True)
        return 1
    print(json.dumps({k: res[k] for k in ("verdict", "p0", "runs", "symptomatic")}), flush=True)
    if require and res["verdict"] not in require:
        print(f"wave_experiment: verdict {res['verdict']} is not one of {require}", flush=True)
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", metavar="FILE")
    ap.add_argument("--require-verdict", default="")
    ap.add_argument("--out")
    ap.add_argument("--runs", type=int, default=0)
    ap.add_argument("--until-symptoms", type=int, default=0)
    ap.add_argument("--max-runs", type=int, default=60)
    ap.add_argument("--budget-s", type=float, default=480.0)
    # A hung fleet must not outlast the agent shell: budget + one run < 600 s.
    ap.add_argument("--run-timeout-s", type=float, default=110.0)
    ap.add_argument("--base-tree")
    ap.add_argument("--expect-symptom", action="store_true")
    ap.add_argument("--expect-fixed", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        return check(a.check, [v for v in a.require_verdict.split(",") if v])
    if a.base_tree:
        if a.until_symptoms < 1 or a.runs or a.expect_fixed:
            ap.error("--base-tree takes --until-symptoms K (>= 1) and --max-runs N")
    elif a.until_symptoms:
        if a.runs or a.expect_fixed or a.expect_symptom or not a.out:
            ap.error("the tree arm takes --until-symptoms K --max-runs N --out FILE")
    elif a.runs < 1 or a.expect_symptom or (not a.out and not a.expect_fixed):
        ap.error("pass --runs N --out FILE, --runs N --expect-fixed, or an arm")
    if a.max_runs < 1:
        ap.error("--max-runs must be >= 1")
    return drive(a)


if __name__ == "__main__":
    sys.exit(main())
