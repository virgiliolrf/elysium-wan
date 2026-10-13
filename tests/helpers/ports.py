"""Port blocks for tests that start real TCP listeners.

A run binds `port_base + stage * 8 + peer`, so a test needs a block of
consecutive free ports. Blocks come from a fixed range below every OS's
ephemeral range (Linux 32768+, macOS and Windows 49152+): outgoing connections
take their source port from the ephemeral range, so they can never land inside
a block, and a busy machine cannot fragment the range. The range also stays
below the engine's default port_base (29000), so a local job is not hit.

Concurrent test processes (parallel agents, load campaigns, a second suite)
reserve blocks through flock'd lock files. The kernel drops a reservation when
its holder exits, so a crashed test never leaks one. Each reserved block is
still probed by binding every port: a process outside the protocol that holds
one of them makes the block skipped, not handed out.
"""

from __future__ import annotations

import fcntl
import os
import random
import socket
import tempfile

BLOCK = 64
RANGE = (20000, 28672)
LOCK_DIR = os.environ.get("WANTRAIN_TEST_PORT_LOCKS") or os.path.join(
    tempfile.gettempdir(), "wantrain-test-ports"
)


def _bindable(base: int, n: int) -> bool:
    for port in range(base, base + n):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                return False
    return True


class PortBlock:
    """`n` consecutive free ports, reserved until release() (or process exit)."""

    def __init__(self, base: int, fd: int):
        self.base = base
        self._fd: int | None = fd

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)  # closing the descriptor drops the flock
            self._fd = None


def reserve_port_block(
    n: int = BLOCK, lock_dir: str | None = None, bases: list[int] | None = None
) -> PortBlock:
    """bases: the candidate block starts (default: every block of RANGE). The
    port-block tests pass blocks they hold through the shared protocol, so
    their checks never depend on what other test processes bind elsewhere."""
    if not 0 < n <= BLOCK:
        raise ValueError(f"a port block holds 1..{BLOCK} ports, not {n}")
    lock_dir = lock_dir or LOCK_DIR
    os.makedirs(lock_dir, exist_ok=True)
    if bases is None:
        bases = list(range(RANGE[0], RANGE[1] - BLOCK + 1, BLOCK))
    count = len(bases)
    # A random start spreads concurrent processes over the range; the scan
    # still visits every block before giving up.
    start = random.SystemRandom().randrange(count) if count else 0
    for k in range(count):
        base = bases[(start + k) % count]
        fd = os.open(os.path.join(lock_dir, f"{base}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        if _bindable(base, n):
            return PortBlock(base, fd)
        os.close(fd)
    raise RuntimeError(
        f"no free block of {n} ports among {count} candidate blocks: every block is reserved by "
        f"another test process ({lock_dir}) or holds a port in use"
    )
