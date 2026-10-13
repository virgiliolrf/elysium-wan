"""TRANSPORT-I10: WAN sockets get BBR when the kernel offers it and path-sized buffers only where that raises them;
tuning never fails a connection and never touches a payload byte.

WOULD-FAIL-IF: a buffer were fixed below the kernel's autotuning ceiling (a cut, not a raise), a refused knob
raised, or the tuned connect did not reach its peer.
"""

import socket
import threading

from wantrain.transport import tuning

MB = 1 << 20


def test_buffers_are_set_only_where_they_raise_the_ceiling():
    # covers: TRANSPORT-I10
    roomy = {"wmem_max": 64 * MB, "rmem_max": 64 * MB, "tcp_wmem": 4 * MB, "tcp_rmem": 6 * MB}
    assert tuning.plan(32 * MB, roomy) == {"sndbuf": 32 * MB, "rcvbuf": 32 * MB}
    # the kernel's hard cap (212 KB by default) is below the request: autotuning (up to 4-6 MB) stays
    default = {"wmem_max": 212992, "rmem_max": 212992, "tcp_wmem": 4 * MB, "tcp_rmem": 6 * MB}
    assert tuning.plan(32 * MB, default) == {"sndbuf": None, "rcvbuf": None}
    # a request under the autotuning ceiling would cut it
    assert tuning.plan(2 * MB, roomy) == {"sndbuf": None, "rcvbuf": None}
    assert tuning.plan(32 * MB, {}) == {"sndbuf": None, "rcvbuf": None}


def test_tune_never_raises_and_the_tuned_connect_carries_bytes(monkeypatch):
    # covers: TRANSPORT-I10
    monkeypatch.setenv("WANTRAIN_TCP_CC", "no-such-algorithm")
    lst = socket.socket()
    lst.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    applied = tuning.tune(lst)
    assert set(applied) == {"cc", "sndbuf", "rcvbuf", "pace"}
    lst.bind(("127.0.0.1", 0))
    lst.listen(1)
    port = lst.getsockname()[1]
    got = []

    def serve():
        conn, _ = lst.accept()
        got.append(conn.recv(5, socket.MSG_WAITALL))
        conn.close()

    t = threading.Thread(target=serve)
    t.start()
    c, info = tuning.connect("127.0.0.1", port, timeout=5)
    c.sendall(b"hello")
    t.join(5)
    c.close()
    lst.close()
    assert got == [b"hello"] and set(info) == {"cc", "sndbuf", "rcvbuf", "pace"}
