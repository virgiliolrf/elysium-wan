"""Turn one WAN run's artifacts into metrics.json and REPORT.md.

    python analyze.py <run-dir>
<run-dir> holds one sub-directory per node (e.g. us/, kr/) with whatever was pulled from it: stage-r*p*.log (WT_TELEM,
WT_MEM lines), *_loss_stream.jsonl, sysmon.csv, coord.log, netprobe.json, node.json. Missing pieces are skipped.
"""

import csv
import glob
import json
import os
import re
import statistics
import sys


def _telem(path):
    out = []
    for line in open(path, errors="replace"):
        if line.startswith("WT_TELEM "):
            try:
                out.append(json.loads(line[9:]))
            except ValueError:
                pass
    return out


def _mem(path):
    peaks = [
        float(m.group(1))
        for m in re.finditer(r"WT_MEM step=\d+ alloc=[\d.]+ peak=([\d.]+)", open(path).read())
    ]
    return max(peaks) if peaks else None


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None


def _sysmon(path):
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 3:
        return None
    t = [float(r["t"]) for r in rows]
    rx = [int(r["rx_bytes"] or 0) for r in rows]
    tx = [int(r["tx_bytes"] or 0) for r in rows]
    util = [float(r["gpu_util"]) for r in rows if r["gpu_util"]]
    power = [float(r["gpu_power_w"]) for r in rows if r["gpu_power_w"]]
    mem = [float(r["gpu_mem_mb"]) for r in rows if r["gpu_mem_mb"]]
    rates_rx = [
        (rx[i] - rx[i - 1]) * 8 / 1e6 / max(1e-3, t[i] - t[i - 1]) for i in range(1, len(rows))
    ]
    rates_tx = [
        (tx[i] - tx[i - 1]) * 8 / 1e6 / max(1e-3, t[i] - t[i - 1]) for i in range(1, len(rows))
    ]
    busy = [u for u in util if u > 0]
    return {
        "seconds": round(t[-1] - t[0]),
        "gpu_util_mean_pct": round(statistics.mean(util), 1) if util else None,
        "gpu_util_mean_when_active_pct": round(statistics.mean(busy), 1) if busy else None,
        "gpu_mem_peak_mb": max(mem) if mem else None,
        "gpu_power_mean_w": round(statistics.mean(power), 1) if power else None,
        "nic_rx_mbps_mean": round(statistics.mean(rates_rx), 2),
        "nic_tx_mbps_mean": round(statistics.mean(rates_tx), 2),
        "nic_rx_mbps_p95": round(_pct(rates_rx, 0.95), 2),
        "nic_tx_mbps_p95": round(_pct(rates_tx, 0.95), 2),
        "nic_total_gb": round(((rx[-1] - rx[0]) + (tx[-1] - tx[0])) / 1e9, 3),
    }


def analyze(run_dir):
    m = {"nodes": {}, "stages": {}}
    coord = None
    loss = []
    for node in sorted(os.listdir(run_dir)):
        nd = os.path.join(run_dir, node)
        if not os.path.isdir(nd):
            continue
        info = {}
        for name in ("node.json", "netprobe.json"):
            p = os.path.join(nd, name)
            if os.path.exists(p):
                info[name[:-5]] = json.load(open(p))
        p = os.path.join(nd, "sysmon.csv")
        if os.path.exists(p):
            info["sysmon"] = _sysmon(p)
        m["nodes"][node] = info
        for log in glob.glob(os.path.join(nd, "stage-r*p*.log")):
            rank = int(re.search(r"stage-r(\d+)p", log).group(1))
            tel = _telem(log)
            if not tel:
                continue
            steady = tel[3:] if len(tel) > 6 else tel  # first steps include warm-up
            step_ms = [t["step_ms"] for t in steady]
            stall = [t.get("stall_ms", 0.0) for t in steady]
            ops = {
                op: round(
                    statistics.median([t["ops"][op]["med_ms"] for t in steady if op in t["ops"]]), 2
                )
                for op in ("F", "B", "W", "OPT")
                if any(op in t["ops"] for t in steady)
            }
            hop_bytes, hop_busy = 0, 0.0
            for t in steady:
                for h in t.get("hops", {}).values():
                    hop_bytes += h["bytes"]
                    hop_busy += h["busy_ms"]
            m["stages"][rank] = {
                "node": node,
                "steps": len(tel),
                "step_ms_median": round(statistics.median(step_ms), 1),
                "step_ms_p10": round(_pct(step_ms, 0.1), 1),
                "step_ms_p90": round(_pct(step_ms, 0.9), 1),
                "stall_ms_median": round(statistics.median(stall), 1),
                "stall_share_pct": round(100 * sum(stall) / max(1e-9, sum(step_ms)), 1),
                "op_median_ms": ops,
                "hop_bytes_per_step_mb": round(hop_bytes / len(steady) / 1e6, 2),
                "hop_mbps_while_busy": round(hop_bytes * 8 / 1e6 / max(1e-9, hop_busy / 1e3), 1)
                if hop_busy
                else None,
                "gpu_mem_peak_gib": _mem(log),
            }
        for p in glob.glob(os.path.join(nd, "*_loss_stream.jsonl")):
            loss = [json.loads(x) for x in open(p) if x.strip()]
        p = os.path.join(nd, "coord.log")
        if os.path.exists(p):
            coord = open(p, errors="replace").read()

    if coord:
        g = re.search(
            r"fwd ([\d.]+) rev ([\d.]+) bidir ([\d.]+)\+([\d.]+) Mbps, RTT ([\d.]+) ms", coord
        )
        if g:
            m["engine_link_probe"] = dict(
                zip(
                    ("fwd_mbps", "rev_mbps", "bidir_a_mbps", "bidir_b_mbps", "rtt_ms"),
                    map(float, g.groups()),
                )
            )
        g = re.search(r"predicted ([\d.]+) s/step, bubble ([\d.]+)%", coord)
        if g:
            m["plan_predicted"] = {"s_per_step": float(g.group(1)), "bubble_pct": float(g.group(2))}
    if loss:
        by_step = {}
        for r in loss:
            by_step.setdefault(r["step"], []).append(r["loss"])
        steps = sorted(by_step)
        mean = {s: statistics.mean(by_step[s]) for s in steps}
        wall = loss[-1]["t"] - loss[0]["t"]
        m["loss"] = {
            "first_step": steps[0],
            "last_step": steps[-1],
            "first": round(mean[steps[0]], 4),
            "last": round(mean[steps[-1]], 4),
            "min": round(min(mean.values()), 4),
            "curve": [[s, round(mean[s], 4)] for s in steps[:: max(1, len(steps) // 40)]],
            "wall_s": round(wall, 1),
        }
    return m


def report(m, spec):
    tokens_step = spec["microbatches"] * spec["mb_size"] * spec["model"]["seq_len"]
    last = max(m["stages"]) if m["stages"] else None
    st = m["stages"].get(last, {})
    s_step = st.get("step_ms_median", 0) / 1e3
    tps = tokens_step / s_step if s_step else None
    m["throughput"] = {
        "tokens_per_step": tokens_step,
        "tokens_per_s": round(tps, 1) if tps else None,
        "steps_per_min": round(60 / s_step, 2) if s_step else None,
    }
    if "loss" in m and m["loss"]["wall_s"]:
        n = m["loss"]["last_step"] - m["loss"]["first_step"] + 1
        m["throughput"]["tokens_per_s_wall"] = round(n * tokens_step / m["loss"]["wall_s"], 1)
    lines = ["# WAN training run", ""]
    lines += [
        f"- Model: {spec['params_m']}M params (d_model {spec['model']['d_model']}, {spec['model']['n_layer']} layers, "
        f"vocab {spec['model']['vocab_size']}, seq {spec['model']['seq_len']}), {spec['model']['compute_dtype']} compute, "
        f"{spec['model']['boundary_dtype']} boundary, partition {spec['partition']}"
    ]
    lines += [
        f"- Steps: {st.get('steps')}; {tokens_step} tokens/step ({spec['microbatches']} x {spec['mb_size']} x "
        f"{spec['model']['seq_len']})",
        "",
    ]
    lines += ["| Metric | Value |", "|---|---|"]
    t = m["throughput"]
    lines += [
        f"| Tokens/s (median step) | {t.get('tokens_per_s')} |",
        f"| Tokens/s (wall clock) | {t.get('tokens_per_s_wall')} |",
        f"| Step time median / p10 / p90 (ms) | {st.get('step_ms_median')} / {st.get('step_ms_p10')} / {st.get('step_ms_p90')} |",
    ]
    if "engine_link_probe" in m:
        e = m["engine_link_probe"]
        lines += [
            f"| Engine link probe | {e['fwd_mbps']} / {e['rev_mbps']} Mbps, RTT {e['rtt_ms']} ms |"
        ]
    for node, info in m["nodes"].items():
        np_ = info.get("netprobe")
        if np_:
            lines += [
                f"| iperf3 from {node} (1 / 4 streams) | {np_.get('send_P1', {}).get('mbps')} / "
                f"{np_.get('send_P4', {}).get('mbps')} Mbps; TCP RTT median {np_['tcp_rtt_ms']['median']} ms |"
            ]
    if "loss" in m:
        lines += [
            f"| Loss first -> last (min) | {m['loss']['first']} -> {m['loss']['last']} ({m['loss']['min']}) |"
        ]
    lines += [
        "",
        "| Stage | Node | F / B / W ms | Stall share | Hop MB/step | GPU peak GiB |",
        "|---|---|---|---|---|---|",
    ]
    for r, s in sorted(m["stages"].items()):
        o = s["op_median_ms"]
        lines += [
            f"| {r} | {s['node']} | {o.get('F')} / {o.get('B')} / {o.get('W')} | {s['stall_share_pct']}% | "
            f"{s['hop_bytes_per_step_mb']} | {s['gpu_mem_peak_gib']} |"
        ]
    lines += [
        "",
        "| Node | GPU util mean (active) | Power W | NIC rx / tx Mbps mean (p95) | Moved GB |",
        "|---|---|---|---|---|",
    ]
    for node, info in m["nodes"].items():
        sm = info.get("sysmon")
        if sm:
            lines += [
                f"| {node} | {sm['gpu_util_mean_pct']}% ({sm['gpu_util_mean_when_active_pct']}%) | {sm['gpu_power_mean_w']} | "
                f"{sm['nic_rx_mbps_mean']} / {sm['nic_tx_mbps_mean']} ({sm['nic_rx_mbps_p95']} / {sm['nic_tx_mbps_p95']}) | {sm['nic_total_gb']} |"
            ]
    return "\n".join(lines) + "\n"


def main():
    run_dir = sys.argv[1]
    m = analyze(run_dir)
    p = os.path.join(run_dir, "spec.json")
    if not os.path.exists(p):
        p = sorted(glob.glob(os.path.join(run_dir, "*", "spec-*.json")))[0]
    spec = json.load(open(p))
    mdl = spec["model"]
    # transformer blocks (attention 4d^2 + MLP 8d^2) plus untied embedding and head
    spec["params_m"] = round(
        (12 * mdl["n_layer"] * mdl["d_model"] ** 2 + 2 * mdl["vocab_size"] * mdl["d_model"]) / 1e6
    )
    text = report(m, spec)
    json.dump(m, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    open(os.path.join(run_dir, "REPORT.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
