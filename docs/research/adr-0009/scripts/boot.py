# ADR-0009 evidence boot: transformers 5.19.0 and the repo's src on sys.path, nothing else.
# ADR9_SITE (colon-separated) names site directories holding transformers==5.19.0 and its dependencies, e.g. made by
#   uv pip install --target /tmp/adr9-site transformers==5.19.0 safetensors
# Default: the uv-cache archives recorded in paths.txt (the architect's Mac, 2026-10-10).
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, "..", "..", "..", ".."))
_site = os.environ.get("ADR9_SITE")
for _p in (_site.split(":") if _site else [l.strip() for l in open(os.path.join(_here, "paths.txt")) if l.strip()]):
    if _p not in sys.path: sys.path.append(_p)
if os.path.join(_repo, "src") not in sys.path: sys.path.insert(0, os.path.join(_repo, "src"))
if _here not in sys.path: sys.path.insert(0, _here)  # sibling scripts (controls.py imports stub_inject)
