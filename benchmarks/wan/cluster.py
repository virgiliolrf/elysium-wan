"""Drive a WAN run on N rented nodes from the operator's machine (Vast.ai-style port maps).

    python cluster.py check   nodes.txt                     # pass/fail each node in ~1 min: driver, TLS, bandwidth, disk
    python cluster.py setup   nodes.txt code.tar.gz         # node_setup.sh + TinyStories slice on every node, in parallel
    python cluster.py discover nodes.txt                    # public IP, mapped ports, GPU, location -> nodes.json
    python cluster.py linkcheck nodes.json [streams]        # node-to-node Mbps over the data ports, parallel flows
    python cluster.py sustain nodes.json job.json [streams] [secs]  # every pair loaded both ways at once, per 5 s window;
                                                            # writes sustain.json; a job with "microbatches": "auto" takes its count
    python cluster.py watch   nodes.json                    # during a run: one line per step, slow stage, stall, error
    python cluster.py launch  nodes.json job.json           # coordinator on the first node, a worker on every node
    python cluster.py status  nodes.json                    # coordinator tail
    python cluster.py stop    nodes.json
    python cluster.py collect nodes.json <run-id> <out-dir>

nodes.txt: one "name ssh-port ssh-host [region]" per line. Each node needs two free mapped container ports (data,
probe) and the first node one more (coordinator); `discover` picks them from vast-capabilities (10100, 10200, 8080 when
free). WANTRAIN_COORD_TOKEN must be set for launch. Extra workers beyond the plan's stages stay registered as spares.
"""

import concurrent.futures as cf
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SSH = [
    "ssh",
    "-i",
    os.path.expanduser("~/.ssh/elysium_vast"),
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "StrictHostKeyChecking=accept-new",
]


def sh(node, cmd, timeout=3600):
    r = subprocess.run(
        SSH + ["-p", str(node["port"]), f"root@{node['host']}", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    # Vast's login banner arrives on stderr after the command's output; drop it so callers can read the last line.
    err = "\n".join(x for x in r.stderr.splitlines() if not any(b in x for b in _BANNER))
    return (r.stdout + err).strip()


_BANNER = (
    "Welcome to vast.ai",
    "If authentication fails",
    "Have fun!",
    "AI agents: READ",
    "Permanently added",
)


def scp(node, files, dest="/root/"):
    subprocess.run(
        ["scp", "-q"]
        + SSH[1:]
        + ["-P", str(node["port"])]
        + files
        + [f"root@{node['host']}:{dest}"],
        check=True,
        timeout=600,
    )


def read_txt(path):
    out = []
    for line in open(path):
        p = line.split()
        if p and not p[0].startswith("#"):
            out.append(
                {
                    "name": p[0],
                    "port": int(p[1]),
                    "host": p[2],
                    "region": p[3] if len(p) > 3 else None,
                }
            )
    return out


def parallel(nodes, fn):
    with cf.ThreadPoolExecutor(len(nodes)) as ex:
        return list(ex.map(fn, nodes))


def setup(nodes, tar):
    files = [tar] + [
        os.path.join(HERE, f)
        for f in ("node_setup.sh", "sysmon.sh", "corpus_tinystories.sh", "netprobe.py")
    ]

    def one(n):
        scp(n, files)
        a = sh(
            n,
            f"bash /root/node_setup.sh /root/{os.path.basename(tar)} 2>&1 | tail -2; bash /root/corpus_tinystories.sh 2>&1 | tail -3",
        )
        return f"== {n['name']}\n{a}"

    print("\n".join(parallel(nodes, one)))


DISCOVER = r"""python3 - <<'PY'
import json, os, subprocess, urllib.request
cap = json.loads(subprocess.run(["vast-capabilities"], capture_output=True, text=True).stdout or "{}")
listening = subprocess.run("ss -ltn", shell=True, capture_output=True, text=True).stdout
def taken(port):
    return f":{port} " in listening
free = [p for p in cap.get("instance", {}).get("open_ports", [])
        if not p.get("self_mapped") and p["container_port"] not in (22, 1111) and not taken(p["container_port"])]
loc = {}
try: loc = json.load(urllib.request.urlopen("https://ipinfo.io/json", timeout=10))
except Exception: pass
gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
print(json.dumps({"ip": os.environ.get("PUBLIC_IPADDR") or loc.get("ip"), "free_ports": [[p["container_port"], int(p["public_port"])] for p in free],
                  "gpu": gpu, "city": loc.get("city"), "country": loc.get("country"), "org": loc.get("org")}))
PY"""


CHECK = r"""python3 - <<'PY'
import json, shutil, subprocess, time, urllib.request
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return ""
out = {"gpu": sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
       "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
       "disk_gb": round(shutil.disk_usage("/").free / 1e9), "ram_gb": round(int(sh("awk '/MemTotal/{print $2}' /proc/meminfo") or 0) / 1e6)}
def speed(url, n):
    t = time.time()
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            got = len(r.read(n))
        return round(got * 8 / 1e6 / (time.time() - t))
    except Exception as e:
        return f"FAIL {type(e).__name__}: {str(e)[:80]}"
# PyPI over TLS with the system's roots: an intercepting proxy shows up as a certificate error
out["pypi_mbps"] = speed("https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl", 600000)
# One stream from a CDN, TLS and redirect included: a floor on the node's link, not the node-to-node rate the
# coordinator's probe measures. Only a node far below any usable link fails on it.
out["dl_mbps"] = speed("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt", 100_000_000)
out["cc"] = sh("cat /proc/sys/net/ipv4/tcp_available_congestion_control")
out["ip"] = sh("curl -s -m 8 https://api.ipify.org")
# uv verifies TLS against its own (Mozilla) roots, unlike urllib above with the system's: an intercepting proxy whose CA
# the image trusts passes urllib and fails here, as it failed `uv sync` on two rented nodes (2026-10-12).
uv = sh("command -v uv || ls ~/.local/bin/uv 2>/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; ls ~/.local/bin/uv)")
r = subprocess.run(f"{uv.splitlines()[-1] if uv else 'uv'} pip install --dry-run --no-deps --target /tmp/uvcheck mpmath==1.3.0",
                   shell=True, capture_output=True, text=True)
out["uv_tls"] = "ok" if r.returncode == 0 else ("FAIL " + (r.stderr.strip().splitlines() or ["?"])[-1][:100])
print(json.dumps(out))
PY"""


def check(nodes, min_driver=580, min_mbps=50, min_disk=30):
    """Pass or fail each rented node in about a minute, before any setup: one GPU model, a driver new enough for the
    locked torch (CUDA 13: >= 580), PyPI reachable over TLS without interception, download speed, free disk."""

    def one(n):
        try:
            d = json.loads(
                [x for x in sh(n, CHECK, timeout=120).splitlines() if x.startswith("{")][-1]
            )
        except Exception as e:  # noqa: BLE001
            return n["name"], {"unreachable": str(e)[:120]}, ["unreachable"]
        bad = []
        if int((d.get("driver") or "0").split(".")[0]) < min_driver:
            bad.append(f"driver {d.get('driver')} < {min_driver}")
        if not isinstance(d.get("pypi_mbps"), int):
            bad.append(f"pypi {d.get('pypi_mbps')}")
        if not isinstance(d.get("dl_mbps"), int) or d["dl_mbps"] < min_mbps:
            bad.append(f"download {d.get('dl_mbps')} Mbps < {min_mbps}")
        if d.get("disk_gb", 0) < min_disk:
            bad.append(f"disk {d.get('disk_gb')} GB < {min_disk}")
        if not str(d.get("uv_tls", "")).startswith("ok"):
            bad.append(f"uv tls {d.get('uv_tls')}")
        return n["name"], d, bad

    rows = parallel(nodes, one)
    gpus = {d.get("gpu", "").split(",")[0] for _, d, b in rows if not b}
    for name, d, bad in rows:
        print(f"{'FAIL' if bad else 'ok  '} {name} {json.dumps(d)} {'; '.join(bad)}")
    if len(gpus) > 1:
        print(
            f"WARN more than one GPU model among the passing nodes: {sorted(gpus)} (not byte-exact)"
        )
    ips: dict = {}
    for name, d, _bad in rows:
        if d.get("ip"):
            ips.setdefault(d["ip"], []).append(name)
    for ip, names in ips.items():
        if len(names) > 1:
            print(
                f"WARN {names} share the public address {ip}: a pipeline takes one slot per address"
            )
    if any("bbr" not in str(d.get("cc", "")) for _n, d, _b in rows):
        print(
            "NOTE no BBR on some nodes (containers often offer only reno/cubic): long hops rely on parallel flows"
        )


SINK = r"""nohup python3 - <<'PY' >/dev/null 2>&1 &
import socket, threading
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0", {port})); s.listen(64)
s.settimeout(120)
def drain(c):
    while c.recv(1 << 20):
        pass
try:
    while True:
        c, _ = s.accept(); threading.Thread(target=drain, args=(c,), daemon=True).start()
except Exception:
    pass
PY
sleep 1; echo sink"""

SEND = r"""python3 - <<'PY'
import json, socket, threading, time
targets = {targets}
streams, secs = {streams}, {secs}
out = {{}}
for name, host, port in targets:
    sent = [0] * streams
    stop = time.time() + secs
    def push(i):
        try:
            c = socket.create_connection((host, port), timeout=10)
            blob = b"\0" * (1 << 20)
            while time.time() < stop:
                c.sendall(blob)
                sent[i] += len(blob)
            c.close()
        except OSError:
            pass
    t0 = time.time()
    ts = [threading.Thread(target=push, args=(i,)) for i in range(streams)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    out[name] = round(sum(sent) * 8 / (time.time() - t0) / 1e6, 1)
print(json.dumps(out))
PY"""


def linkcheck(nodes, streams=8, secs=8):
    """Node-to-node rate with `streams` parallel TCP flows (the data plane's p2p_streams) over each node's discovered
    data port, before launch: the hop rates decide the bubble (SCHEDULER-I23), and a datacenter's advertised NIC speed
    says nothing about them. Prints the matrix in Mbps (row: sender). Run it before `launch` (the ports are free)."""
    parallel(nodes, lambda n: sh(n, SINK.format(port=n["data"][0])))
    rows = {}
    for n in nodes:  # one sender at a time, so each row measures its own paths
        targets = [(m["name"], m["ip"], m["data"][1]) for m in nodes if m is not n]
        line = sh(
            n,
            SEND.format(targets=repr(targets), streams=streams, secs=secs),
            timeout=60 + secs * len(nodes),
        )
        rows[n["name"]] = json.loads([x for x in line.splitlines() if x.startswith("{")][-1])
    names = [n["name"] for n in nodes]
    print("from/to  " + " ".join(f"{x:>8}" for x in names))
    for a in names:
        cells = [f"{rows[a].get(b, 0):8.0f}" if a != b else f"{'-':>8}" for b in names]
        print(f"{a:>8} " + " ".join(cells))
    return rows


SEND_WIN = r"""python3 - <<'PY'
import json, socket, threading, time
host, port, streams, secs, win = {host!r}, {port}, {streams}, {secs}, {win}
sent = [0] * streams
t0 = time.time()
stop = t0 + secs
def push(i):
    try:
        c = socket.create_connection((host, port), timeout=10)
        blob = b"\0" * (1 << 20)
        while time.time() < stop:
            c.sendall(blob)
            sent[i] += len(blob)
        c.close()
    except OSError:
        pass
ts = [threading.Thread(target=push, args=(i,), daemon=True) for i in range(streams)]
[t.start() for t in ts]
marks, last = [], 0
while time.time() < stop:
    time.sleep(win)
    now = sum(sent)
    marks.append(round((now - last) * 8 / win / 1e6, 1))
    last = now
[t.join(timeout=15) for t in ts]
print(json.dumps({{"windows": marks}}))
PY"""

# The lossless planes codec's measured ratio on bf16 GPT boundaries (2026-10-12: 0.52 at d 2560, 0.69 at d 5632-6144).
WIRE_RATIO = 0.69


def need_mbps(
    d_model, seq_len, mb_size=1, mb_ms=250.0, wire_ratio=WIRE_RATIO, margin=1.3, value_bytes=2
):
    """The rate a hop must hold, each way, for a stage computing a microbatch in `mb_ms` never to wait on it: one
    boundary (seq x d bf16 values, after the codec) per microbatch, times `margin`. A hop carries activations one way
    and gradients the other at the same time, so this is per direction under load in both."""
    boundary = mb_size * seq_len * d_model * value_bytes * wire_ratio
    return boundary * 8 / (mb_ms / 1000.0) / 1e6 * margin


# Effective dense bf16 TFLOPs a stage reaches on its layers, F+B+W per microbatch under recompute (measured: ~220 on
# an H100 SXM/NVL at d 5632-6144, 2026-10-12; ~40 on an RTX 4090 at d 1536, 85 ms per microbatch for 0.42B,
# 2026-10-13). Unknown GPUs take the 4090's, which prices the need high, not low.
GPU_TFLOPS = {"H100": 220.0, "H200": 240.0, "A100": 120.0, "RTX 4090": 40.0, "RTX 3090": 25.0}


def estimate_mb_ms(plan, stages, gpu=""):
    """A balanced stage's compute per microbatch: 8 x its parameters x tokens FLOPs under recompute (6 x without), at
    the GPU's effective rate. The need of a hop scales with it: the faster a stage, the faster its boundaries go."""
    d, layers, vocab = plan["d_model"], plan["n_layer"], plan.get("vocab_size", 50257)
    params = (12 * d * d * layers + 2 * vocab * d) / max(1, stages)
    tokens = plan["seq_len"] * plan.get("mb_size", 1)
    flops = (8 if plan.get("recompute", True) else 6) * params * tokens
    rate = next((v for k, v in GPU_TFLOPS.items() if k in gpu), GPU_TFLOPS["RTX 4090"])
    return flops / (rate * 1e12) * 1e3


def window_stats(windows):
    """{min, p10, median} Mbps over a sustained run's windows, the first one dropped (TCP's ramp)."""
    w = sorted(windows[1:] or windows)
    if not w:
        return {"min": 0.0, "p10": 0.0, "median": 0.0}
    return {"min": w[0], "p10": w[int(0.1 * (len(w) - 1))], "median": w[len(w) // 2]}


def verdict(stats, need):
    """ "ok" when even the slowest window holds `need`, "weak" when the median does but a window dips under it,
    "fail" when the median does not."""
    if stats["min"] >= need:
        return "ok"
    return "weak" if stats["median"] >= need else "fail"


def suspects(results):
    """Nodes to replace, worst first: those in a failing direction (its median below the need), ranked by failing
    directions then weak ones over every pair they are in (results: {(a, b): verdict} per direction a -> b). A node
    whose NIC is the cap fails toward every peer. A weak direction (a dip under the need, the median above it) names
    no one: recommend_microbatches answers it with fewer microbatches."""
    score = {}
    for (a, b), v in results.items():
        for n in (a, b):
            f, w = score.get(n, (0, 0))
            score[n] = (f + (v == "fail"), w + (v == "weak"))
    bad = [n for n, (f, _w) in score.items() if f]
    return sorted(bad, key=lambda n: (-score[n][0], -score[n][1], n))


def recommend_microbatches(ratios):
    """Microbatches per step for the slowest direction's sustained headroom (its min window over the need): 128 at
    2x or more, 64 at 1x or more, 32 below. More microbatches spread the step's fixed boundary (drain, optimizer,
    refill: ~4 s on four transatlantic stages) over more compute, but keep every hop loaded the whole step: on
    2026-10-12, 256 microbatches through a node that held only its need saturated its port and tripled the step."""
    worst = min(ratios) if ratios else 0.0
    if worst >= 2.0:
        return 128
    return 64 if worst >= 1.0 else 32


def sustain(nodes, job, streams=32, secs=30, win=5.0, mb_ms=None, report=None):
    """Qualify every pair under the training's own load: both directions at once, `streams` flows each, for `secs`,
    read per `win` window. A short one-way test passed a node whose NIC is a 340 Mbps port at 600-870 Mbps and the run
    then waited on it every microbatch (2026-10-12, Nuremberg); a sustained two-way load shows the cap. The rate to
    hold is need_mbps for the job's shape (`mb_ms`: a stage's compute per microbatch, estimated from the job's model
    and the fastest node's GPU unless given). Prints each direction's min / p10 / median Mbps and verdict, then the
    nodes to replace."""
    spec = json.load(open(job)) if isinstance(job, str) else job
    plan = spec["plan"]
    if mb_ms is None:
        # the fastest node's GPU: the stage that sends soonest sets the rate its hop must hold
        mb_ms = min(
            estimate_mb_ms(plan, spec.get("stages", len(nodes)), n.get("gpu", "")) for n in nodes
        )
    need = need_mbps(plan["d_model"], plan["seq_len"], plan.get("mb_size", 1), mb_ms)
    print(
        f"need {need:.0f} Mbps each way (d {plan['d_model']}, seq {plan['seq_len']}, ~{mb_ms:.0f} ms/mb per stage)"
    )
    parallel(nodes, lambda n: sh(n, SINK.format(port=n["data"][0])))
    results, rows = {}, []
    try:
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                pair = [(a, b), (b, a)]

                def run(ab):
                    src, dst = ab
                    out = sh(
                        src,
                        SEND_WIN.format(
                            host=dst["ip"], port=dst["data"][1], streams=streams, secs=secs, win=win
                        ),
                        timeout=secs + 120,
                    )
                    line = [x for x in out.splitlines() if x.startswith("{")]
                    return json.loads(line[-1])["windows"] if line else []

                for (src, dst), w in zip(pair, parallel(pair, run)):
                    st = window_stats(w)
                    v = verdict(st, need)
                    results[(src["name"], dst["name"])] = v
                    rows.append((src["name"], dst["name"], st, v))
                    print(
                        f"{src['name']:>6} -> {dst['name']:<6} min {st['min']:7.0f}  p10 {st['p10']:7.0f}  "
                        f"median {st['median']:7.0f} Mbps  {v.upper()}",
                        flush=True,
                    )
    finally:
        # The sinks stop accepting 120 s after their last connection; free the data ports for launch now.
        parallel(
            nodes,
            lambda n: sh(
                n,
                f"for p in $(ss -ltnp | grep ':{n['data'][0]} ' | grep -oE 'pid=[0-9]+' | cut -d= -f2); "
                "do kill $p; done; true",
            ),
        )
    bad = suspects(results)
    print("REPLACE " + (" ".join(bad) if bad else "none: every pair holds the rate both ways"))
    ratios = [st["min"] / need for _, _, st, _ in rows]
    mbs = recommend_microbatches(ratios)
    print(
        f"RECOMMEND microbatches={mbs} (slowest direction holds {min(ratios, default=0):.2f}x the need)"
    )
    if report:
        with open(report, "w") as fh:
            json.dump(
                {
                    "need_mbps": need,
                    "pairs": [{"src": a, "dst": b, **st, "verdict": v} for a, b, st, v in rows],
                    "replace": bad,
                    "microbatches": mbs,
                },
                fh,
                indent=1,
            )
    return rows, bad


WATCH_NODE = r"""f=$(ls -t /workspace/run/w/runs/*/stage-r*.log 2>/dev/null | head -1); [ -z "$f" ] && exit 0;
echo "DIR $(dirname $f)"; grep WT_TELEM $f | tail -3; echo "ERRS $(grep -cE 'Traceback|Error|out of memory' $f)";
grep -E 'Traceback|Error|out of memory' $f | tail -1; echo "PROCS $(pgrep -fc '[s]tage_main')" """


def stage_ms(telem):
    """A stage's compute per microbatch in one WT_TELEM summary: the median F + B + W, in ms."""
    ops = telem.get("ops", {})
    return sum(ops.get(k, {}).get("med_ms", 0.0) for k in ("F", "B", "W"))


def slow_stages(history, baseline, slow=1.3, runs=2):
    """Stages whose compute per microbatch has stayed above `slow` x their own baseline for the last `runs` steps
    (history: {name: [ms per step, ...]}; baseline: {name: ms}). A noisy host slows its stage and, through the
    pipeline, every other one (2026-10-12: one stage's backward swung 128-1,719 ms while its twin held 122)."""
    out = []
    for n, xs in history.items():
        b = baseline.get(n)
        if b and len(xs) >= runs and all(x > slow * b for x in xs[-runs:]):
            out.append((n, round(xs[-1] / b, 2)))
    return out


def watch(nodes, interval=30.0, slow=1.3, base_steps=3):
    """Follow a run from the operator's machine, one line per event (for a log or a Monitor): each step's time,
    tokens/s and idle share per stage; a stage whose compute per microbatch stays above `slow` x its baseline (the
    median of its first `base_steps` steps after the one-time pinning step); no step for 3 x the baseline step time;
    an error in a stage log; a stage process gone; the coordinator's fit, schedule and launch lines."""
    import statistics
    import time as _time

    head = nodes[0]
    plan = {}
    try:
        plan = json.loads(sh(head, "cat /workspace/run/job.json")).get("plan", {})
    except ValueError:
        pass
    tokens = plan.get("microbatches", 0) * plan.get("seq_len", 0) * plan.get("mb_size", 1)
    seen, last_step, last_t, run_dir = 0, -1, None, {}
    base, hist, base_step_s, errs, gone = {}, {}, None, {}, set()

    def say(msg):
        print(_time.strftime("%H:%M:%S") + " " + msg, flush=True)

    while True:
        log = sh(head, "cat /workspace/run/coord.log 2>/dev/null").splitlines()
        for line in log[seen:]:
            if any(
                k in line
                for k in (
                    "fit rank",
                    "does not fit",
                    "schedule:",
                    "launching",
                    "rror",
                    "lost",
                    "refus",
                )
            ):
                say("COORD " + line[:240])
        seen = len(log)
        tel = {}
        for n in nodes:
            out = sh(n, WATCH_NODE, timeout=60)
            for line in out.splitlines():
                if line.startswith("DIR "):
                    if run_dir.get(n["name"]) not in (None, line[4:]):
                        say(f"NEW RUN on {n['name']}: baselines reset")
                        last_step, last_t, base, hist, base_step_s = -1, None, {}, {}, None
                    run_dir[n["name"]] = line[4:]
                elif line.startswith("WT_TELEM "):
                    try:
                        tel[n["name"]] = json.loads(line[9:])
                    except ValueError:
                        pass
                elif line.startswith("ERRS "):
                    c = int(line.split()[1] or 0)
                    if c > errs.get(n["name"], 0):
                        say(f"ERROR in {n['name']}'s stage log ({c} lines)")
                    errs[n["name"]] = c
                elif line.startswith("PROCS ") and line.split()[1] == "0" and last_step >= 0:
                    if n["name"] not in gone:
                        gone.add(n["name"])
                        say(f"NO STAGE PROCESS on {n['name']}")
                elif line.startswith("PROCS "):
                    gone.discard(n["name"])
        if tel and len(tel) == len(nodes):
            step = min(t["step"] for t in tel.values())
            if step > last_step:
                last_step, last_t = step, _time.time()
                step_s = max(t["step_ms"] for t in tel.values()) / 1e3
                idle = " | ".join(
                    f"{k} r{t['rank']} idle {100 * t['stall_ms'] / t['step_ms']:.0f}% {stage_ms(t):.0f} ms/mb"
                    for k, t in tel.items()
                )
                rate = f"{tokens / step_s:.0f} tok/s" if tokens else ""
                say(f"STEP {step}: {step_s:.1f} s {rate} | {idle}")
                if step >= 2:  # steps 0-1 hold warm-up and the one-time pinning of host moments
                    for k, t in tel.items():
                        hist.setdefault(k, []).append(stage_ms(t))
                        if len(hist[k]) == base_steps:
                            base[k] = statistics.median(hist[k])
                    if len(hist[next(iter(hist))]) == base_steps:
                        base_step_s = step_s
                        say("BASELINE " + " ".join(f"{k} {v:.0f} ms/mb" for k, v in base.items()))
                    for k, r in slow_stages(hist, base, slow):
                        say(
                            f"SLOW STAGE {k}: {r}x its baseline compute per microbatch: replace or re-cut"
                        )
            elif last_t and _time.time() - last_t > 3 * max(base_step_s or 0, 100):
                say(f"NO STEP for {int(_time.time() - last_t)} s (last {last_step})")
                last_t = _time.time()
        _time.sleep(interval)


def discover(nodes, out):
    # The coordinator's node needs a third mapped port: Jupyter's 8080 is freed there (a Vast template default).
    sh(
        nodes[0],
        "supervisorctl stop jupyter >/dev/null 2>&1; pkill -f jupyter-notebook; sleep 2; true",
    )

    def one(n):
        d = json.loads(sh(n, DISCOVER).splitlines()[-1])
        n.update(d)
        ports = d["free_ports"]
        need = 3 if n is nodes[0] else 2
        if len(ports) < need:
            n["error"] = f"only {len(ports)} free mapped ports; need {need}"
            return n
        n["data"], n["probe"] = ports[0], ports[1]
        if n is nodes[0]:
            n["coord"] = ports[2]
        return n

    parallel(nodes, one)
    json.dump(nodes, open(out, "w"), indent=1)
    for n in nodes:
        print(
            n["name"],
            n.get("city"),
            n.get("country"),
            n.get("gpu"),
            n.get("ip"),
            n.get("data"),
            n.get("probe"),
            n.get("coord", ""),
            n.get("error", ""),
        )


def _coord_env() -> str:
    """The coordinator knobs set in the operator's environment, passed through to the coordinator's command."""
    keys = (
        "WANTRAIN_RECUT",
        "WANTRAIN_RECUT_COST_S",
        "WANTRAIN_SELECTIVE_RECOMPUTE",
        "WANTRAIN_OPT_RESIDENT",
    )
    return "".join(f"{k}={os.environ[k]} " for k in keys if os.environ.get(k))


def resolve_job(job: dict, report: dict | None) -> dict:
    """A job whose plan says "microbatches": "auto" takes the count `sustain` recommended for this fleet; refused
    without a sustain report, so no run starts on links nobody loaded."""
    plan = job["plan"]
    if plan.get("microbatches") != "auto":
        return job
    if not report:
        raise SystemExit(
            'job says "microbatches": "auto": run `cluster.py sustain nodes.json job.json` first'
        )
    if report.get("replace"):
        raise SystemExit(
            f"sustain named nodes to replace: {report['replace']}; replace them and run it again"
        )
    return {**job, "plan": {**plan, "microbatches": int(report["microbatches"])}}


def launch(nodes, job, report_path=None):
    tok = os.environ["WANTRAIN_COORD_TOKEN"]
    c = nodes[0]
    with open(job) as fh:
        spec = json.load(fh)
    report = None
    if report_path and os.path.exists(report_path):
        with open(report_path) as fh:
            report = json.load(fh)
    resolved = resolve_job(spec, report)
    if resolved is not spec:
        print(f"microbatches={resolved['plan']['microbatches']} (from {report_path})")
        job = job.rsplit(".", 1)[0] + ".resolved.json"
        with open(job, "w") as fh:
            json.dump(resolved, fh, indent=1)
    scp(c, [job], "/workspace/run_job.json")
    parallel(
        nodes,
        lambda n: sh(
            n,
            # [w]: the pattern must not match this very shell's command line, or pkill kills it first.
            'mkdir -p /workspace/run; pkill -f "[w]antrain (coordinator|worker|stage)"; pkill -f "[s]ysmon.sh"; true',
        ),
    )
    sh(
        c,
        f"mkdir -p /workspace/run && mv /workspace/run_job.json /workspace/run/job.json && cd /workspace/elysium && "
        f"({_coord_env()}WANTRAIN_COORD_TOKEN={tok} nohup uv run --no-sync wantrain coordinator --host 0.0.0.0 --port {c['coord'][0]} "
        f"--job /workspace/run/job.json --state-dir /workspace/run/coord --any-region --wait-workers-s 1200 --zb-mem-mode {os.environ.get('WANTRAIN_ZB_MEM_MODE', 'auto')} "
        f"> /workspace/run/coord.log 2>&1 &); sleep 12; true",
    )
    cid = sh(c, "grep -m1 COORDINATOR_ID /workspace/run/coord.log").split()[-1]

    def one(n):
        target = f"127.0.0.1:{c['coord'][0]}" if n is c else f"{c['ip']}:{c['coord'][1]}"
        region = n.get("region") or (n.get("country") or "x").lower()
        return sh(
            n,
            f"cd /workspace/elysium && (WANTRAIN_COORD_TOKEN={tok} nohup uv run --no-sync wantrain worker "
            f"--coordinator {target} --coordinator-id {cid} --advertise-host {n['ip']} "
            f"--data-port {n['data'][1]} --listen-port {n['data'][0]} --probe-port {n['probe'][1]} "
            f"--probe-listen-port {n['probe'][0]} --name {n['name']} --workdir /workspace/run/w --device cuda "
            f"--region {region} > /workspace/run/worker.log 2>&1 &); "
            f"(nohup bash /root/sysmon.sh /workspace/run/sysmon.csv >/dev/null 2>&1 &); echo {n['name']} up",
        )

    print(f"coordinator {cid}")
    print("\n".join(parallel(nodes, one)))


def main():
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "check":
        check(read_txt(a[0]))
    elif cmd == "linkcheck":
        linkcheck(json.load(open(a[0])), streams=int(a[1]) if len(a) > 1 else 8)
    elif cmd == "sustain":
        _, bad = sustain(
            json.load(open(a[0])),
            a[1],
            streams=int(a[2]) if len(a) > 2 else 32,
            secs=int(a[3]) if len(a) > 3 else 30,
            report=os.path.join(os.path.dirname(os.path.abspath(a[0])), "sustain.json"),
        )
        sys.exit(1 if bad else 0)
    elif cmd == "watch":
        watch(json.load(open(a[0])))
    elif cmd == "setup":
        setup(read_txt(a[0]), a[1])
    elif cmd == "discover":
        discover(read_txt(a[0]), a[0].rsplit(".", 1)[0] + ".json")
    elif cmd == "launch":
        launch(
            json.load(open(a[0])),
            a[1],
            os.path.join(os.path.dirname(os.path.abspath(a[0])), "sustain.json"),
        )
    elif cmd == "status":
        print(
            sh(json.load(open(a[0]))[0], "grep -v registered /workspace/run/coord.log | tail -15")
        )
    elif cmd == "stop":
        parallel(
            json.load(open(a[0])),
            lambda n: sh(
                n,
                'pkill -INT -f "[w]antrain (stage|worker|coordinator)"; sleep 3; pkill -f "[s]ysmon.sh"; '
                'pkill -f "[w]antrain (stage|worker|coordinator)"; true',
            ),
        )
    elif cmd == "collect":
        nodes = json.load(open(a[0]))
        specs = " ".join(f"{n['name']}:{n['port']}:{n['host']}" for n in nodes)
        subprocess.run(
            ["bash", os.path.join(HERE, "collect.sh"), a[1], a[2]] + specs.split(), check=False
        )


if __name__ == "__main__":
    main()
