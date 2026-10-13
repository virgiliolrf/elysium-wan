#!/bin/bash
# The same ~100M-token slice of TinyStories-train on every node: the first 400 MB of the file, cut at the last newline.
set -euo pipefail
cd /workspace/elysium
mkdir -p /workspace/data
if [ ! -f /workspace/data/big.bin.meta.json ]; then
  uv run --no-sync python - <<'PY'
import urllib.request
url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
req = urllib.request.Request(url, headers={"Range": "bytes=0-419430399"})
with urllib.request.urlopen(req, timeout=600) as r:
    data = r.read()
data = data[: data.rfind(b"\n") + 1]
open("/workspace/data/big.txt", "wb").write(data)
print("bytes", len(data))
PY
  uv run --no-sync wantrain prepare-data --corpus /workspace/data/big.txt --source-type txt --out /workspace/data/big.bin --tokenizer gpt2 >/dev/null
  rm -f /workspace/data/big.txt
fi
python3 -c "import json;m=json.load(open('/workspace/data/big.bin.meta.json'));print('BIG', m['total_tokens'], m['vocab_size'], m['dtype'])"
sha256sum /workspace/data/big.bin.meta.json | cut -c1-16
df -h / | tail -1 | awk '{print "free", $4}'
