"""Link between two nodes, measured from this node: TCP connect RTT (no ICMP needed) and, with iperf3, throughput with
1 and 4 parallel streams in both directions. Prints one "NETPROBE {json}" line.

    python netprobe.py <peer-host> <peer-iperf-port> [--seconds 15]
The peer runs `iperf3 -s -p <container-port>`; <peer-iperf-port> is its public port.
"""

import json
import socket
import statistics
import subprocess
import sys
import time


def tcp_rtt(host, port, n=20):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=5):
                out.append((time.perf_counter() - t) * 1e3)
        except OSError:
            pass
        time.sleep(0.2)
    return out


def iperf(host, port, seconds, streams, reverse):
    cmd = ["iperf3", "-c", host, "-p", str(port), "-t", str(seconds), "-P", str(streams), "-J"]
    if reverse:
        cmd.append("-R")
    try:
        r = json.loads(
            subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 30).stdout
        )
        end = r["end"]
        return {
            "mbps": round(end["sum_received"]["bits_per_second"] / 1e6, 1),
            "retransmits": end.get("sum_sent", {}).get("retransmits"),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


def main():
    host, port = sys.argv[1], int(sys.argv[2])
    secs = int(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 15
    rtts = tcp_rtt(host, port)
    res = {
        "peer": f"{host}:{port}",
        "tcp_rtt_ms": {
            "n": len(rtts),
            "min": round(min(rtts), 2) if rtts else None,
            "median": round(statistics.median(rtts), 2) if rtts else None,
            "p90": round(sorted(rtts)[int(0.9 * (len(rtts) - 1))], 2) if rtts else None,
        },
    }
    for streams in (1, 4):
        res[f"send_P{streams}"] = iperf(host, port, secs, streams, False)
        res[f"recv_P{streams}"] = iperf(host, port, secs, streams, True)
    print("NETPROBE " + json.dumps(res))


if __name__ == "__main__":
    main()
