import os, sys  # added on commit: paths resolved from this directory (docs/research/adr-0009/README.md)
# ADR-0009 round 1, D9 fingerprint by the wheel's RECORD (PEP 376/627): verify every file of the transformers and
# torch distributions against RECORD's sha256, list files present under the package but absent from RECORD, time it.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
import importlib.metadata as md, hashlib, base64, time, os, sys
for name in ("transformers", "torch"):
    t0 = time.time(); d = md.distribution(name); files = d.files or []
    n = nbytes = bad = nohash = 0; root = None
    for f in files:
        p = f.locate()
        if root is None and str(f).startswith(name + "/"): root = os.path.join(str(p)[: str(p).index(str(f))], name)
        if f.hash is None: nohash += 1; continue
        b = open(p, "rb").read(); n += 1; nbytes += len(b)
        dg = base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b"=").decode()
        bad += dg != f.hash.value
    listed = {os.path.normpath(str(f.locate())) for f in files}
    extra = [os.path.join(dp, fn) for dp, _, fns in os.walk(root) for fn in fns
             if os.path.normpath(os.path.join(dp, fn)) not in listed and not fn.endswith(".pyc")]
    print(f"{name} {d.version}: {n} files {nbytes/1e6:.1f} MB verified in {time.time()-t0:.2f}s, mismatched {bad}, "
          f"RECORD entries without hash {nohash}, files outside RECORD (non-.pyc) {len(extra)} {extra[:3]}", flush=True)
