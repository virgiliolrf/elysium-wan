"""Correctness comparator: exact diff between two recorded loss sequences.

Success criterion: maximum absolute difference == 0 over the full run, checked
on the hex representation (bit-exact), with the float diff reported for
diagnostics when nonzero.
"""

import json


def compare(baseline_path: str, candidate_path: str) -> dict:
    with open(baseline_path) as f:
        a = json.load(f)["losses"]
    with open(candidate_path) as f:
        b = json.load(f)["losses"]

    def key(r):
        return (r["step"], r["mb"])

    a_map = {key(r): r for r in a}
    b_map = {key(r): r for r in b}
    if set(a_map) != set(b_map):
        return {
            "ok": False,
            "reason": "coverage mismatch",
            "missing": sorted(set(a_map) ^ set(b_map))[:10],
        }
    n_mismatch, max_abs = 0, 0.0
    first_bad = None
    for k in sorted(a_map):
        ra, rb = a_map[k], b_map[k]
        if ra["loss_hex"] != rb["loss_hex"]:
            n_mismatch += 1
            d = abs(ra["loss"] - rb["loss"])
            if d > max_abs:
                max_abs = d
            if first_bad is None:
                first_bad = {"at": k, "baseline": ra["loss_hex"], "candidate": rb["loss_hex"]}
    return {
        "ok": n_mismatch == 0,
        "compared": len(a_map),
        "mismatches": n_mismatch,
        "max_abs_diff": max_abs,
        "first_mismatch": first_bad,
    }


if __name__ == "__main__":
    import sys

    r = compare(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["ok"] else 1)
