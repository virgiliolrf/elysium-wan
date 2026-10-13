# ADR-0009 round 1: run census3.py over the model types missing from out/census-r1.txt, in chunks, appending lines.
# Exists because two full runs were stopped by SIGTERM (exit 143) from outside while another checkout's gate ran.
import os, sys, subprocess
here = os.path.dirname(os.path.abspath(__file__)); exec(open(os.path.join(here, "boot.py")).read())
import warnings; warnings.filterwarnings("ignore")
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as M
out = os.path.join(here, "..", "out", "census-r1.txt")
for _ in range(40):
    done = {l.split()[0] for l in open(out) if l.strip() and not l.startswith(" ") and len(l.split()) > 1 and l.split()[1] in ("EXACT", "DIFFER", "ERR", "TIMEOUT")}
    rest = [k for k in sorted(M) if k not in done]
    print(f"done {len(done)} rest {len(rest)}", flush=True)
    if not rest: break
    with open(out, "a") as f:
        r = subprocess.run([sys.executable, "-I", os.path.join(here, "census3.py")] + rest[:20], stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
    print("chunk exit", r.returncode, flush=True)
