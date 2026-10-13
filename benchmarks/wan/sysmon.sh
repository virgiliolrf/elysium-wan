#!/bin/bash
# Per-second system sampler for one node: GPU utilization, memory and power, and the bytes the main network interface
# moved. Writes CSV rows "t,gpu_util,gpu_mem_mb,gpu_power_w,rx_bytes,tx_bytes" until killed.
#   bash sysmon.sh <out.csv>
out=${1:-sysmon.csv}
iface=$(ip route show default 2>/dev/null | awk '{print $5; exit}')
iface=${iface:-eth0}
echo "t,gpu_util,gpu_mem_mb,gpu_power_w,rx_bytes,tx_bytes" > "$out"
while true; do
  g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  read -r rx tx < <(awk -v i="$iface:" '$1==i {print $2, $10}' /proc/net/dev)
  echo "$(date +%s.%N | cut -c1-14),${g:-,,},${rx:-0},${tx:-0}" >> "$out"
  sleep 1
done
