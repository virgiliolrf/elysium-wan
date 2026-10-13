"""Pipeline-shaped message latency over N parallel TCP flows: does pacing (SO_MAX_PACING_RATE) or the flow count
cut a boundary message's completion time on this path?

    receiver:  python3 msgbench.py recv <listen_port>
    sender:    python3 msgbench.py send <host> <port> <flows> <pace_mbps_per_flow|0> <msg_mb> <gap_ms> <count> [idle_s]
The sender splits each message evenly over the flows, starts one every gap_ms, and times each from its first byte
sent to the receiver's ack (one control flow back): latency = transfer + one one-way trip. After `count` messages
it idles idle_s and sends three more (the step-boundary case: every flow idle for seconds).
"""

import socket
import struct
import sys
import threading
import time

SO_MAX_PACING_RATE = getattr(socket, "SO_MAX_PACING_RATE", 47)
HDR = struct.Struct("<IQ")  # msg id, part length


def recv_exact(s, n, buf=None):
    view = memoryview(buf) if buf is not None else None
    got = 0
    out = bytearray(n) if buf is None else None
    mv = memoryview(out) if out is not None else view
    while got < n:
        k = s.recv_into(mv[got:n])
        if not k:
            raise ConnectionError("closed")
        got += k
    return out


def receiver(port):
    lst = socket.socket()
    lst.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lst.bind(("0.0.0.0", port))
    lst.listen(512)
    while True:
        ctl, _ = lst.accept()
        n = struct.unpack("<I", recv_exact(ctl, 4))[0]
        flows = [lst.accept()[0] for _ in range(n)]
        lock = threading.Lock()
        parts: dict = {}
        first: dict = {}

        def pump(s, lock=lock, first=first, parts=parts, n=n, ctl=ctl):
            buf = bytearray(64 << 20)
            try:
                while True:
                    mid, ln = HDR.unpack(recv_exact(s, HDR.size))
                    t = time.monotonic()
                    with lock:
                        first.setdefault(mid, t)
                    recv_exact(s, ln, buf)
                    with lock:
                        parts[mid] = parts.get(mid, 0) + 1
                        done = parts[mid] == n
                        if done:
                            spread = time.monotonic() - first.pop(mid)
                            ctl.sendall(struct.pack("<Id", mid, spread))
            except (ConnectionError, OSError):
                return

        ts = [threading.Thread(target=pump, args=(s,), daemon=True) for s in flows]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        ctl.close()


def sender(host, port, n, pace_mbps, msg_mb, gap_ms, count, idle_s):
    def dial():
        s = socket.create_connection((host, port), timeout=20)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if pace_mbps > 0:
            s.setsockopt(
                socket.SOL_SOCKET, SO_MAX_PACING_RATE, struct.pack("Q", int(pace_mbps * 1e6 / 8))
            )
        return s

    ctl = dial()
    ctl.sendall(struct.pack("<I", n))
    flows = [dial() for _ in range(n)]
    size = int(msg_mb * 1e6)
    part = -(-size // n)
    payload = bytes(part)
    starts: dict = {}
    acks: dict = {}
    qs = [[] for _ in range(n)]
    cv = threading.Condition()

    def send_loop(i):
        s = flows[i]
        while True:
            with cv:
                while not qs[i]:
                    cv.wait()
                mid = qs[i].pop(0)
            if mid < 0:
                return
            s.sendall(HDR.pack(mid, part) + payload)

    def ack_loop():
        while True:
            try:
                mid, spread = struct.unpack("<Id", recv_exact(ctl, 12))
            except (ConnectionError, OSError):
                return
            acks[mid] = (time.monotonic(), spread)

    ts = [threading.Thread(target=send_loop, args=(i,), daemon=True) for i in range(n)]
    for t in ts:
        t.start()
    threading.Thread(target=ack_loop, daemon=True).start()

    def send(mid):
        starts[mid] = time.monotonic()
        with cv:
            for q in qs:
                q.append(mid)
            cv.notify_all()

    mid = 0
    t_next = time.monotonic()
    for _ in range(count):
        time.sleep(max(0.0, t_next - time.monotonic()))
        send(mid)
        mid += 1
        t_next += gap_ms / 1e3
    time.sleep(2.0)
    idle_ids = []
    for _ in range(3):
        time.sleep(idle_s)
        send(mid)
        idle_ids.append(mid)
        mid += 1
    time.sleep(3.0)
    lat = sorted((acks[m][0] - starts[m]) * 1e3 for m in range(count) if m in acks)
    spr = sorted(acks[m][1] * 1e3 for m in range(count) if m in acks)
    idle = [round((acks[m][0] - starts[m]) * 1e3) if m in acks else None for m in idle_ids]
    lost = count - len(lat)

    def pct(xs, q):
        return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]) if xs else None

    print(
        f"flows={n} pace={pace_mbps}Mbps/flow msg={msg_mb}MB gap={gap_ms}ms: latency p50 {pct(lat, 0.5)} p90 "
        f"{pct(lat, 0.9)} max {pct(lat, 1.0)} ms | spread p50 {pct(spr, 0.5)} p90 {pct(spr, 0.9)} | after {idle_s}s "
        f"idle {idle} ms | unacked {lost}",
        flush=True,
    )
    with cv:
        for q in qs:
            q.append(-1)
        cv.notify_all()
    for s in flows + [ctl]:
        s.close()


if __name__ == "__main__":
    if sys.argv[1] == "recv":
        receiver(int(sys.argv[2]))
    else:
        a = sys.argv[2:]
        sender(
            a[0],
            int(a[1]),
            int(a[2]),
            float(a[3]),
            float(a[4]),
            float(a[5]),
            int(a[6]),
            float(a[7]) if len(a) > 7 else 3.0,
        )
