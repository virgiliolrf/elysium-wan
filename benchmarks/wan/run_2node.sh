#!/bin/bash
# start_run.sh <token>: (re)start coordinator on A and both workers, sysmon on both. Run from the Mac.
S=$(cd "$(dirname "$0")" && pwd)
TOK=${WANTRAIN_COORD_TOKEN:?set WANTRAIN_COORD_TOKEN}
K="-i $HOME/.ssh/elysium_vast -o BatchMode=yes"
A="ssh $K -p 26151 root@ssh6.vast.ai"; B="ssh $K -p 12629 root@ssh5.vast.ai"
scp -q $K -P 26151 ${JOB:-$S/job-2node-us-kr.json} root@ssh6.vast.ai:/workspace/run/job.json
for h in "$A" "$B"; do $h 'pkill -f "wantrain (coordinator|worker|stage)"; pkill -f sysmon.sh; sleep 2; true' 2>/dev/null; done
$A "cd /workspace/elysium && (WANTRAIN_COORD_TOKEN=$TOK nohup uv run --no-sync wantrain coordinator --host 0.0.0.0 --port 10100 --job /workspace/run/job.json --state-dir /workspace/run/coord --any-region --wait-workers-s 900 > /workspace/run/coord.log 2>&1 &); sleep 10; grep -m1 COORDINATOR_ID /workspace/run/coord.log" 2>/dev/null > /tmp/wantrain_cid.txt
CID=$(awk "{print \$NF}" /tmp/wantrain_cid.txt); echo "CID=$CID"
$B "cd /workspace/elysium && (WANTRAIN_COORD_TOKEN=$TOK WANTRAIN_DEVICE_CLASS=arch nohup uv run --no-sync wantrain worker --coordinator 203.0.113.11:28789 --coordinator-id $CID --advertise-host 203.0.113.2 --data-port 31178 --listen-port 10100 --probe-port 31192 --probe-listen-port 10200 --name kr-seoul-3060 --workdir /workspace/run/w --device cuda --region ap-northeast > /workspace/run/worker.log 2>&1 &); (nohup bash /root/sysmon.sh /workspace/run/sysmon.csv >/dev/null 2>&1 &)" 2>/dev/null
$A "cd /workspace/elysium && (WANTRAIN_COORD_TOKEN=$TOK WANTRAIN_DEVICE_CLASS=arch nohup uv run --no-sync wantrain worker --coordinator 127.0.0.1:10100 --coordinator-id $CID --advertise-host 203.0.113.11 --data-port 28796 --listen-port 10200 --probe-port 28791 --probe-listen-port 8080 --name us-bellevue-3070L --workdir /workspace/run/w --device cuda --region us-west > /workspace/run/worker.log 2>&1 &); (nohup bash /root/sysmon.sh /workspace/run/sysmon.csv >/dev/null 2>&1 &)" 2>/dev/null
echo started
