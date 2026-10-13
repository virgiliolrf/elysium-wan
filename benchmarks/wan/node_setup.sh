#!/bin/bash
# Prepare one rented node: uv, iperf3, the engine at /workspace/elysium (from the tarball copied next to this script),
# its locked environment, and a token corpus. Prints one "NODE {json}" line describing the node.
#   bash node_setup.sh <code.tar.gz> [corpus-tokens]
set -euo pipefail
TAR=${1:?code tarball}
CORPUS_TOKENS=${2:-20000000}
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v iperf3 >/dev/null || (apt-get update -qq >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq iperf3 iproute2 iputils-ping >/dev/null) || true
mkdir -p /workspace/elysium && tar -xzf "$TAR" -C /workspace/elysium
cd /workspace/elysium
uv sync --all-extras --locked -q
# The same corpus, byte for byte, on every node: the manifest digest is pinned by the plan (MODEL-I11).
if [ ! -f /workspace/data/corpus.bin.meta.json ]; then
  mkdir -p /workspace/data
  uv run --no-sync python - <<'PY'
import json, urllib.request
# TinyStories validation split as plain text (public, ~20 MB), one story per line
url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
with urllib.request.urlopen(url, timeout=120) as r, open("/workspace/data/corpus.txt", "wb") as f:
    f.write(r.read())
PY
  uv run --no-sync wantrain prepare-data --corpus /workspace/data/corpus.txt --source-type txt \
     --out /workspace/data/corpus.bin --tokenizer gpt2 --limit-tokens "$CORPUS_TOKENS" >/dev/null
fi
uv run --no-sync python - <<'PY'
import json, os, subprocess, urllib.request
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True).strip()
    except Exception: return ""
loc = {}
try: loc = json.load(urllib.request.urlopen("https://ipinfo.io/json", timeout=10))
except Exception: pass
ports = {k: v for k, v in os.environ.items() if k.startswith("VAST_TCP_PORT_")}
print("NODE " + json.dumps({
    "gpu": sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
    "cpus": os.cpu_count(), "public_ip": os.environ.get("PUBLIC_IPADDR") or loc.get("ip"),
    "city": loc.get("city"), "region": loc.get("region"), "country": loc.get("country"), "org": loc.get("org"),
    "ports": ports, "corpus_sha256": sh("sha256sum /workspace/data/corpus.bin.meta.json | cut -d' ' -f1"),
    "torch": sh("cd /workspace/elysium && uv run --no-sync python -c 'import torch;print(torch.__version__, torch.cuda.is_available())'"),
}))
PY
