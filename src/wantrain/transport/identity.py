"""Peer identity and transcript-bound handshake (TRANSPORT-I5).

Each agent holds an ed25519 keypair; its peer id is sha256(public key)[:32 hex],
the same on every host. On connect both sides exchange their public key, a fresh
16-byte challenge and an X25519 ephemeral, then each signs, under its own role,
the SHA-256 of the plane's domain and the whole transcript in role order. A
reflected challenge, a challenge relayed from another handshake of the same
process, a signature under another domain or transcript, and non-complementary
declared roles are refused. Every field is canonical lowercase hex of its exact
length. Session keys for each direction
come from HKDF-SHA256 over the X25519 secret, salted with the transcript hash.

Without an allowlist any key that proves possession is accepted (the socket is
bound to *an* identity, logged in telemetry). `trusted_peers` turns the
handshake into admission control and `expect_peer` pins one id; either refuses
another peer after the signature check, before any tensor flows. There is no
shared-token mode: without `cryptography` the handshake refuses.
"""

import hashlib
import json
import os
import struct
import threading

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
        X25519PublicKey,
    )
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    _HAVE_ED = True
except Exception:
    _HAVE_ED = False

FINGERPRINT_ALGO = "sha256"
P2P_DOMAIN = "wantrain-p2p|v2|"
CTL_DOMAIN = "wantrain-ctl|v2|"
ROLES = ("coordinator", "worker")  # role order: the coordinator first
_COMPLEMENT = {"coordinator": "worker", "worker": "coordinator"}
_CHAL_LEN = 16
_KEY_LEN = 32
_PREFIX_LEN = 4
_HEX = frozenset("0123456789abcdef")

# Challenges of every handshake this process is running. A peer that presents
# our own key with one of these challenges is another of our sockets, relayed:
# refused, so an on-path relay cannot pair two handshakes of one process into an
# authenticated self-loop. (Two processes on one host share a key, and one
# process may run both ends under two keys: both stay legitimate.) One insert,
# lookup and delete per connection.
_IN_FLIGHT: set[bytes] = set()
_IN_FLIGHT_LOCK = threading.Lock()


def fingerprint(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _send(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(struct.pack("<I", len(b)) + b)


_MAX_HANDSHAKE_BYTES = 64 * 1024


def default_key_path() -> str:
    return os.environ.get(
        "WANTRAIN_PEER_KEY", os.path.join(os.path.expanduser("~"), ".wantrain", "peer_key")
    )


def _recv(sock):
    hdr = b""
    while len(hdr) < 4:
        c = sock.recv(4 - len(hdr))
        if not c:
            raise ConnectionError("peer closed during handshake")
        hdr += c
    (n,) = struct.unpack("<I", hdr)
    if n > _MAX_HANDSHAKE_BYTES:
        raise ConnectionError("handshake message too large")
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise ConnectionError("peer closed during handshake")
        buf.extend(c)
    try:
        msg = json.loads(bytes(buf))
    except ValueError as e:
        raise ConnectionError("malformed handshake message") from e
    if not isinstance(msg, dict):
        raise ConnectionError("malformed handshake message")
    return msg


def _hex_field(msg: dict, name: str, n: int) -> bytes:
    """Exactly n bytes as 2n lowercase hex digits, else the handshake is refused
    (bytes.fromhex alone would skip whitespace and decode fewer bytes)."""
    v = msg.get(name)
    if not isinstance(v, str) or len(v) != 2 * n or not _HEX.issuperset(v):
        raise ConnectionError(
            f"handshake field {name!r} must be exactly {n} bytes of lowercase hex"
        )
    b = bytes.fromhex(v)
    if len(b) != n:
        raise ConnectionError(
            f"handshake field {name!r} must be exactly {n} bytes of lowercase hex"
        )
    return b


def load_or_create_key(path: str):
    if not _HAVE_ED:
        return None
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Ed25519PrivateKey.from_private_bytes(f.read())
    key = Ed25519PrivateKey.generate()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw = key.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
    )
    # Several stage processes (and the p2p auth threads inside one) can reach
    # this at once on one host; exactly one creates the file, the rest read it,
    # so a host never presents two identities. Owner-only: it is a private key.
    tmp = f"{path}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    try:
        os.link(tmp, path)
    except FileExistsError:
        pass
    finally:
        os.unlink(tmp)
    with open(path, "rb") as f:
        return Ed25519PrivateKey.from_private_bytes(f.read())


def peer_id(pub_bytes: bytes) -> str:
    return fingerprint(pub_bytes)[:32]


def _lp(b: bytes) -> bytes:
    return struct.pack("<H", len(b)) + b


def _body(first: dict, second: dict) -> bytes:
    """Both pubs, both challenges, both ephemerals and both AES flags, in role
    order (fixed lengths, so the concatenation is unambiguous)."""
    return (
        first["pub"]
        + second["pub"]
        + first["chal"]
        + second["chal"]
        + first["eph"]
        + second["eph"]
        + bytes([first["aes"], second["aes"]])
    )


def _transcript(domain: str, body: bytes) -> bytes:
    return hashlib.sha256(_lp(domain.encode()) + b"transcript" + body).digest()


def _signed_digest(domain: str, role: str, body: bytes) -> bytes:
    """What the side holding `role` signs: SHA-256 of domain + role + transcript body."""
    return hashlib.sha256(_lp(domain.encode()) + _lp(role.encode()) + body).digest()


def _direction(secret: bytes, transcript: bytes, domain: str, sender_role: str) -> tuple:
    okm = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_LEN + _PREFIX_LEN,
        salt=transcript,
        info=_lp(domain.encode()) + b"send|" + sender_role.encode(),
    ).derive(secret)
    return okm[:_KEY_LEN], okm[_KEY_LEN:]


def authenticate(
    sock,
    key_path: str | None = None,
    trusted_peers=None,
    expect_peer: str | None = None,
    aes_hw: bool = False,
    domain: str = P2P_DOMAIN,
    role: str | None = None,
):
    """Mutual authentication and key agreement over an established socket.
    Returns {peer_id, mode, fingerprint_algo, cipher, send_key, recv_key,
    send_prefix, recv_prefix, transcript}. Raises ConnectionError on failure."""
    if not _HAVE_ED:
        raise ConnectionError(
            "no peer authentication available: the ed25519/X25519 handshake needs `cryptography`"
        )
    if role is not None and role not in _COMPLEMENT:
        raise ValueError(f"role must be None or one of {ROLES}, got {role!r}")
    key = load_or_create_key(key_path or default_key_path())
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    eph = X25519PrivateKey.generate()
    eph_pub = eph.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    me = {"pub": pub, "chal": os.urandom(_CHAL_LEN), "eph": eph_pub, "aes": bool(aes_hw)}
    with _IN_FLIGHT_LOCK:
        _IN_FLIGHT.add(me["chal"])
    try:
        return _exchange(sock, key, eph, me, trusted_peers, expect_peer, domain, role)
    finally:
        with _IN_FLIGHT_LOCK:
            _IN_FLIGHT.discard(me["chal"])


def _exchange(sock, key, eph, me, trusted_peers, expect_peer, domain, role):
    _send(
        sock,
        {
            "mode": "ed25519",
            "pub": me["pub"].hex(),
            "chal": me["chal"].hex(),
            "eph": me["eph"].hex(),
            "aes": me["aes"],
            "role": role,
        },
    )
    msg = _recv(sock)
    if msg.get("mode") != "ed25519":
        raise ConnectionError("peer lacks ed25519 identity")
    if not isinstance(msg.get("aes"), bool):
        raise ConnectionError("handshake field 'aes' must be a boolean")
    peer = {
        "pub": _hex_field(msg, "pub", 32),
        "chal": _hex_field(msg, "chal", _CHAL_LEN),
        "eph": _hex_field(msg, "eph", 32),
        "aes": msg["aes"],
    }
    if peer["chal"] == me["chal"]:
        raise ConnectionError("peer reflected our challenge")
    with _IN_FLIGHT_LOCK:
        relayed = peer["chal"] in _IN_FLIGHT
    if relayed and peer["pub"] == me["pub"]:
        raise ConnectionError("peer challenge belongs to another handshake of this process")
    peer_role = msg.get("role")
    if role is None:
        if peer_role is not None:
            raise ConnectionError(f"peer declared role {peer_role!r}, we declared none")
        my_tag, peer_tag = ("lo", "hi") if me["chal"] < peer["chal"] else ("hi", "lo")
    else:
        if peer_role != _COMPLEMENT[role]:
            raise ConnectionError(f"peer role {peer_role!r} does not complement ours ({role!r})")
        my_tag, peer_tag = role, peer_role
    first, second = (me, peer) if my_tag in ("lo", "coordinator") else (peer, me)
    body = _body(first, second)

    _send(sock, {"sig": key.sign(_signed_digest(domain, my_tag, body)).hex()})
    peer_sig = _hex_field(_recv(sock), "sig", 64)
    try:
        Ed25519PublicKey.from_public_bytes(peer["pub"]).verify(
            peer_sig, _signed_digest(domain, peer_tag, body)
        )
    except (InvalidSignature, ValueError) as e:
        raise ConnectionError("peer signature does not cover this transcript") from e

    pid = peer_id(peer["pub"])
    if trusted_peers is not None and pid not in set(trusted_peers):
        raise ConnectionError(f"peer {pid} is not in the trusted peer set")
    if expect_peer is not None and pid != expect_peer:
        raise ConnectionError(f"peer {pid} is not the expected peer {expect_peer}")

    try:
        secret = eph.exchange(X25519PublicKey.from_public_bytes(peer["eph"]))
    except ValueError as e:  # a low-order point yields the all-zero secret
        raise ConnectionError("peer ephemeral key is invalid") from e
    transcript = _transcript(domain, body)
    send_key, send_prefix = _direction(secret, transcript, domain, my_tag)
    recv_key, recv_prefix = _direction(secret, transcript, domain, peer_tag)
    return {
        "peer_id": pid,
        "mode": "ed25519",
        "fingerprint_algo": FINGERPRINT_ALGO,
        "cipher": "aes-256-gcm" if me["aes"] and peer["aes"] else "chacha20-poly1305",
        "send_key": send_key,
        "recv_key": recv_key,
        "send_prefix": send_prefix,
        "recv_prefix": recv_prefix,
        "transcript": transcript,
    }
