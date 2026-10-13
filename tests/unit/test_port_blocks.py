"""Port blocks for socket tests: concurrent test processes never share one,
blocks avoid the OS ephemeral range, and a port in use elsewhere disqualifies
its block.

WOULD-FAIL-IF: two concurrent test processes could be handed overlapping port
blocks, a block could fall inside the ephemeral range where outgoing
connections take their source ports, or a block holding a port in use were
handed out.

covers: test infrastructure (helpers/ports.py; no product invariant)
"""

import os
import socket
import subprocess
import sys
import textwrap

import pytest
from helpers.ports import BLOCK, RANGE, reserve_port_block

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def own_blocks():
    """Six blocks held through the shared lock directory, as every test
    process takes them: no cooperating process binds inside them, so the
    tests below, which reserve against a private lock_dir, see only their own
    sockets there. Scanning the whole range instead made them fail whenever
    another suite had a listener anywhere in it."""
    held = [reserve_port_block() for _ in range(6)]
    try:
        yield sorted(b.base for b in held)
    finally:
        for b in held:
            b.release()


def test_blocks_are_below_every_ephemeral_range_and_the_engine_default():
    assert RANGE[1] <= 29000 <= 32768
    block = reserve_port_block()
    try:
        assert RANGE[0] <= block.base and block.base + BLOCK <= RANGE[1]
        assert (block.base - RANGE[0]) % BLOCK == 0
    finally:
        block.release()


def test_a_block_held_by_another_process_is_never_handed_out(tmp_path, own_blocks):
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {TESTS!r})
                from helpers.ports import reserve_port_block
                blocks = [reserve_port_block(lock_dir={str(tmp_path)!r}, bases={own_blocks!r}) for _ in range(3)]
                print(" ".join(str(b.base) for b in blocks), flush=True)
                sys.stdin.read()
                """
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        held = {int(b) for b in holder.stdout.readline().split()}
        assert len(held) == 3
        mine = [reserve_port_block(lock_dir=str(tmp_path), bases=own_blocks) for _ in range(3)]
        assert not held & {b.base for b in mine}
        for b in mine:
            b.release()
    finally:
        holder.stdin.close()
        holder.wait(timeout=30)
        holder.stdout.close()
    # The holder exited without releasing: the kernel dropped its locks.
    free = [reserve_port_block(lock_dir=str(tmp_path), bases=own_blocks) for _ in own_blocks]
    assert held <= {b.base for b in free}
    for b in free:
        b.release()


def test_a_block_with_a_port_in_use_is_skipped(tmp_path, own_blocks):
    first = reserve_port_block(lock_dir=str(tmp_path), bases=own_blocks)
    base = first.base
    first.release()
    busy = socket.socket()
    try:
        busy.bind(("127.0.0.1", base + 5))
        busy.listen()
        got = [
            reserve_port_block(lock_dir=str(tmp_path), bases=own_blocks)
            for _ in range(len(own_blocks) - 1)
        ]
        assert base not in {b.base for b in got}
        with pytest.raises(RuntimeError, match="no free block"):
            reserve_port_block(lock_dir=str(tmp_path), bases=own_blocks)
        for b in got:
            b.release()
    finally:
        busy.close()


def test_block_size_is_bounded():
    with pytest.raises(ValueError):
        reserve_port_block(BLOCK + 1)
