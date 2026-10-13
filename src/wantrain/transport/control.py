"""Control-plane channel between the coordinator and worker agents.

Message = u32 header length | JSON header | u64 body length | body. The header
is a plain JSON object with a "kind"; the body is opaque bytes (a spec, a
log tail) or, for checkpoint shards, a file streamed in chunks straight to
disk on the receiving side so a multi-GB shard never sits in memory. Nothing
here deserializes code: the source used pickle on this socket, which lets any
host that can reach the coordinator's port execute code on it.

Low-rate traffic only: specs, heartbeats, progress, shards. Boundary tensors
never transit the coordinator.

TRANSPORT-I9: `control_dial` / `control_accept` run the control-domain identity
handshake (the coordinator in the coordinator role) and seal the Channel with
its keys before any message. The worker's token MAC and its session MAC are
HMACs over the handshake transcript in distinct domains; the coordinator proves
its own ed25519 key in the handshake, never anything derived from the token.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct
import tempfile
import threading
import time

from wantrain.transport import identity
from wantrain.transport.secure import SecureSocket

MAX_HEADER = 16 << 20
MAX_INLINE_BODY = 256 << 20
_CHUNK = 1 << 20


class ChannelClosed(ConnectionError):
    pass


def _recv_exact(sock: socket.socket | SecureSocket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), _CHUNK))
        if not chunk:
            raise ChannelClosed("peer closed")
        buf.extend(chunk)
    return bytes(buf)


class Channel:
    """Thread-safe for concurrent senders; one reader at a time."""

    def __init__(self, sock: socket.socket):
        self.sock: socket.socket | SecureSocket = sock
        self._send_lock = threading.Lock()
        self.closed = False
        self.sealed = False
        self._used = False
        # Last time any byte arrived: a multi-GB shard streaming over a slow
        # WAN holds the line for minutes, and that is not a silent worker.
        self.last_rx = time.time()

    def seal(
        self,
        cipher: str,
        send_key: bytes,
        recv_key: bytes,
        send_prefix: bytes,
        recv_prefix: bytes,
    ) -> None:
        """Carry every later byte of send, send_file and recv in AEAD records.
        Only before the first message: a channel that already spoke in clear
        is not a sealed channel."""
        with self._send_lock:
            raw = self.sock
            if self.sealed or self._used or isinstance(raw, SecureSocket):
                raise ValueError("a Channel is sealed once, before its first message")
            self.sock = SecureSocket(raw, cipher, send_key, recv_key, send_prefix, recv_prefix)
            self.sealed = True

    def send(self, msg: dict, body: bytes = b"") -> None:
        head = json.dumps(msg, separators=(",", ":")).encode()
        frame = struct.pack("<I", len(head)) + head + struct.pack("<Q", len(body)) + body
        with self._send_lock:
            self._used = True
            try:
                self.sock.sendall(frame)
            except OSError as e:
                self.closed = True
                raise ChannelClosed(str(e)) from e

    def send_file(self, msg: dict, path: str) -> str:
        """Stream a file as the body. Returns its sha256 (also put in the
        header as `sha256`, computed before sending so the receiver verifies)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(_CHUNK), b""):
                h.update(b)
        size = os.path.getsize(path)
        msg = dict(msg, sha256=h.hexdigest(), size=size, file=True)
        head = json.dumps(msg, separators=(",", ":")).encode()
        with self._send_lock:
            self._used = True
            try:
                self.sock.sendall(struct.pack("<I", len(head)) + head + struct.pack("<Q", size))
                with open(path, "rb") as f:
                    sent = 0
                    for b in iter(lambda: f.read(_CHUNK), b""):
                        self.sock.sendall(b)
                        sent += len(b)
                if sent != size:
                    raise ChannelClosed(f"{path} changed size while being sent")
            except OSError as e:
                self.closed = True
                raise ChannelClosed(str(e)) from e
        return msg["sha256"]

    def recv(
        self,
        file_dir: str | None = None,
        max_file: int | None = None,
        max_inline: int = MAX_INLINE_BODY,
    ):
        """-> (header dict, body). A file message is streamed to a temp file in
        `file_dir` and the body is that path; its digest is NOT verified here
        (the owner of the file decides what a mismatch means).

        Files are opt-in: without `file_dir` a file message is refused before
        anything touches the disk, so a read before admission (HELLO) cannot be
        used to fill it. `max_file` bounds an accepted file and `max_inline` an
        inline body; both are refused before their bytes are read."""
        self._used = True
        try:
            return self._recv(file_dir, max_file, max_inline)
        except ChannelClosed:
            raise
        except ConnectionError as e:  # a sealed record failed its tag, length or counter
            raise ChannelClosed(str(e)) from e

    def _recv(self, file_dir: str | None, max_file: int | None, max_inline: int):
        (n,) = struct.unpack("<I", _recv_exact(self.sock, 4))
        if n > MAX_HEADER:
            raise ChannelClosed(f"control header of {n} bytes")
        msg = json.loads(_recv_exact(self.sock, n))
        if not isinstance(msg, dict) or "kind" not in msg:
            raise ChannelClosed("control message without a kind")
        (blen,) = struct.unpack("<Q", _recv_exact(self.sock, 8))
        if msg.get("file"):
            if file_dir is None:
                raise ChannelClosed("file body where this read takes no file")
            if max_file is not None and blen > max_file:
                raise ChannelClosed(f"file body of {blen} bytes (bound {max_file})")
            os.makedirs(file_dir, exist_ok=True)
            fd, path = tempfile.mkstemp(dir=file_dir, suffix=".part")
            try:
                with os.fdopen(fd, "wb") as f:
                    left = blen
                    while left:
                        chunk = self.sock.recv(min(left, _CHUNK))
                        if not chunk:
                            raise ChannelClosed("peer closed mid-file")
                        f.write(chunk)
                        left -= len(chunk)
                        self.last_rx = time.time()
            except BaseException:
                os.unlink(path)
                raise
            self.last_rx = time.time()
            return msg, path
        if blen > max_inline:
            raise ChannelClosed(f"inline body of {blen} bytes (bound {max_inline})")
        body = _recv_exact(self.sock, blen) if blen else b""
        self.last_rx = time.time()
        return msg, body

    def close(self) -> None:
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


# ------------------------------------------------------------------ auth

_WORKER_DOMAIN = b"wantrain-control-worker|v3|"
_SESSION_DOMAIN = b"wantrain-control-session|v3|"
_MAC_HEX = 64
_HEX = frozenset("0123456789abcdef")


def control_dial(
    sock: socket.socket,
    key_path: str | None,
    coordinator_id: str | None = None,
    aes_hw: bool = False,
) -> tuple[Channel, dict]:
    """Worker side: authenticate in the control domain as the worker, refusing
    any coordinator key but `coordinator_id` when one is pinned, and return the
    Channel sealed with the handshake's keys plus the handshake info. Raises
    ConnectionError when the handshake fails."""
    info = identity.authenticate(
        sock,
        key_path,
        expect_peer=coordinator_id,
        aes_hw=aes_hw,
        domain=identity.CTL_DOMAIN,
        role="worker",
    )
    return _sealed(sock, info), info


def control_accept(
    sock: socket.socket, key_path: str | None, aes_hw: bool = False
) -> tuple[Channel, dict]:
    """Coordinator side of `control_dial`: any worker key is accepted here
    (admission is the token MAC inside the sealed channel)."""
    info = identity.authenticate(
        sock, key_path, aes_hw=aes_hw, domain=identity.CTL_DOMAIN, role="coordinator"
    )
    return _sealed(sock, info), info


def _sealed(sock: socket.socket, info: dict) -> Channel:
    ch = Channel(sock)
    ch.seal(
        info["cipher"],
        info["send_key"],
        info["recv_key"],
        info["send_prefix"],
        info["recv_prefix"],
    )
    return ch


def _mac(domain: bytes, secret: str, transcript: bytes | str) -> str:
    if not isinstance(secret, str) or not secret:
        raise ValueError("a MAC needs a non-empty secret")
    t = transcript.encode() if isinstance(transcript, str) else bytes(transcript)
    if not t:
        raise ValueError("a MAC needs a non-empty transcript")
    return hmac.new(secret.encode(), domain + t, hashlib.sha256).hexdigest()


def _check(domain: bytes, secret: str, transcript: bytes | str, mac) -> bool:
    if not isinstance(mac, str) or len(mac) != _MAC_HEX or not _HEX.issuperset(mac):
        return False
    if not isinstance(secret, str) or not secret:
        return False
    return hmac.compare_digest(_mac(domain, secret, transcript), mac)


def token_mac(token: str, transcript: bytes | str) -> str:
    """The worker's admission proof: HMAC-SHA256(token, worker domain +
    transcript of this control handshake), 64 lowercase hex."""
    return _mac(_WORKER_DOMAIN, token, transcript)


def session_mac(session: str, transcript: bytes | str) -> str:
    """A reconnecting worker's proof of its session, bound to this transcript."""
    return _mac(_SESSION_DOMAIN, session, transcript)


def check_mac(token: str, transcript: bytes | str, mac) -> bool:
    return _check(_WORKER_DOMAIN, token, transcript, mac)


def check_session_mac(session: str, transcript: bytes | str, mac) -> bool:
    return _check(_SESSION_DOMAIN, session, transcript, mac)
