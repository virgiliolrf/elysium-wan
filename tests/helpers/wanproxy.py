"""A lossy-WAN TCP proxy for the data plane.

TCP never hands the application a lost byte: a lost segment shows up as a
stall while it is retransmitted, with the bytes behind it held back (head of
line blocking), and a long path delivers the stream in arbitrary pieces. This
proxy reproduces exactly that between two ranks: it forwards every byte, in
order, cut into random pieces, and stalls a random fraction of the pieces.
Latency, jitter and bandwidth stay with the engine's own shim
(transport/shim.py); this adds what the shim cannot, a stream that arrives
torn and late.

Each direction runs on its own thread and propagates a half-close, so the
engine's END + shutdown(SHUT_WR) drain protocol crosses the proxy intact.
"""

from __future__ import annotations

import random
import socket
import threading
import time


class LossyLink:
    def __init__(
        self,
        listen_port: int,
        target_port: int,
        seed: int,
        piece_bytes: tuple[int, int] = (1, 4096),
        stall_p: float = 0.02,
        stall_ms: tuple[float, float] = (20.0, 200.0),
        host: str = "127.0.0.1",
    ):
        self.listen_port = listen_port
        self.target = (host, target_port)
        self.host = host
        self.seed = seed
        self.piece_bytes = piece_bytes
        self.stall_p = stall_p
        self.stall_ms = stall_ms
        self.stats = {"connections": 0, "bytes": 0, "pieces": 0, "stalls": 0}
        self._lock = threading.Lock()
        self._socks: list[socket.socket] = []
        self._server: socket.socket | None = None
        self._closed = False

    def start(self) -> LossyLink:
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.listen_port))
        srv.listen(16)
        self._server = srv
        threading.Thread(target=self._accept, daemon=True).start()
        return self

    def close(self) -> None:
        self._closed = True
        if self._server is not None:
            self._server.close()
        with self._lock:
            socks, self._socks = self._socks, []
        for s in socks:
            try:
                s.close()
            except OSError:
                pass

    def _accept(self) -> None:
        while not self._closed:
            try:
                client, _ = self._server.accept()  # type: ignore[union-attr]
            except OSError:
                return
            with self._lock:
                n = self.stats["connections"]
                self.stats["connections"] += 1
            threading.Thread(target=self._bridge, args=(client, n), daemon=True).start()

    def _bridge(self, client: socket.socket, n: int) -> None:
        # The target may not listen yet: ranks boot at different speeds, and
        # the dialer already counts this connection as made.
        deadline = time.monotonic() + 60
        while True:
            try:
                upstream = socket.create_connection(self.target, timeout=10)
                upstream.settimeout(None)
                break
            except OSError:
                if time.monotonic() > deadline or self._closed:
                    client.close()
                    return
                time.sleep(0.05)
        for s in (client, upstream):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self._lock:
            self._socks += [client, upstream]
        rngs = [random.Random(self.seed * 1_000_003 + n * 2 + d) for d in (0, 1)]
        threads = [
            threading.Thread(target=self._pump, args=(client, upstream, rngs[0]), daemon=True),
            threading.Thread(target=self._pump, args=(upstream, client, rngs[1]), daemon=True),
        ]
        for t in threads:
            t.start()

    def _pump(self, src: socket.socket, dst: socket.socket, rng: random.Random) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    dst.shutdown(socket.SHUT_WR)
                    return
                i = 0
                while i < len(data):
                    k = rng.randint(*self.piece_bytes)
                    if rng.random() < self.stall_p:
                        with self._lock:
                            self.stats["stalls"] += 1
                        time.sleep(rng.uniform(*self.stall_ms) / 1e3)
                    dst.sendall(data[i : i + k])
                    i += k
                    with self._lock:
                        self.stats["pieces"] += 1
                with self._lock:
                    self.stats["bytes"] += len(data)
        except OSError:
            return
