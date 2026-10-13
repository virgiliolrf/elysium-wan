#!/bin/bash
# collect.sh <run-id> <out-dir> [name:port:host ...]: pull one run's artifacts from every node (bash 3 compatible).
RUN=$1; OUT=$2; shift 2
NODES=${*:-"us:26151:ssh6.vast.ai kr:12629:ssh5.vast.ai"}
K="-i $HOME/.ssh/elysium_vast -o BatchMode=yes"
for spec in $NODES; do
  n=${spec%%:*}; rest=${spec#*:}; port=${rest%%:*}; host=${rest#*:}; mkdir -p "$OUT/$n"
  ssh $K -p "$port" "root@$host" "cd /workspace/run && tar -czf /tmp/pull.tgz --ignore-failed-read sysmon.csv coord.log job.json \
      w/runs/$RUN/stage-*.log w/runs/$RUN/out/*.jsonl w/runs/$RUN/spec-*.json 2>/dev/null; true" 2>/dev/null
  scp -q $K -P "$port" "root@$host:/tmp/pull.tgz" "$OUT/$n/pull.tgz" && tar -xzf "$OUT/$n/pull.tgz" -C "$OUT/$n" && rm "$OUT/$n/pull.tgz"
  find "$OUT/$n/w" -type f -exec mv {} "$OUT/$n/" \; 2>/dev/null; rm -rf "$OUT/$n/w"
done
