# Extracts, from the e7_spec.json that was computed with the as-built lane solve (src/wantrain/scheduler/zero_bubble.py
# sha256 a93a8bedf977c4540bbb615147a4a7d3c1b9141a5e88bb3f6b53d9d4553e964e, loaded by path; that json's sha256
# 4af7911b0a9bb386f5ca99c1bdc3fda7f83cba0f573b5d093bb9152fecb55e4e), every priced candidate's lane period, step, E and
# window per stage, so e7_spec.py can compare its independent lane solve (lane_solve.py) against the as-built numbers
# without importing or reading src/wantrain.
# usage: python3 -I -B extract_asbuilt.py <as-built e7_spec.json> asbuilt_lanes.json
import hashlib
import json
import sys

SRC, OUT = sys.argv[1:3]
d = json.load(open(SRC))
rows = {}
for mode, res in sorted(d["modes"].items()):
    for c in res:
        for r in c["rows"]:
            if r.get("period") is None:
                continue
            rows[f"{mode}|{c['config']['id']}|{r['name']}"] = dict(
                period=r["period"], step=r["step"], E_ms=r.get("E_ms"), window_ms=r.get("window_ms"),
                admitted=r["admitted"], emission=r.get("emission"))
json.dump(dict(source_sha256=hashlib.sha256(open(SRC, "rb").read()).hexdigest(),
               lane_solver_sha256="a93a8bedf977c4540bbb615147a4a7d3c1b9141a5e88bb3f6b53d9d4553e964e", rows=rows),
          open(OUT, "w"), indent=1, sort_keys=True)
print(len(rows), "rows")
