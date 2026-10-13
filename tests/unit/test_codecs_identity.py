"""Lossless boundary codecs and the peer-identity handshake."""

# WOULD-FAIL-IF: a boundary codec round trip changes any bit, or the peer handshake accepts an unknown identity, a
# reflected or relayed challenge, a signature over another transcript, domain or role, or authenticates without
# ed25519

import hashlib
import inspect
import os
import socket
import struct
import threading

import pytest
import torch
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from wantrain.codec import f32_planes, planes
from wantrain.model import fp8
from wantrain.transport import identity


def _act(dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(2, 16, 64, generator=g) * 3).to(dtype)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_planes_codec_is_bit_exact(dtype):
    # covers: TRANSPORT-I4
    t = _act(dtype)
    payload, shape = planes.encode(t)
    back = planes.decode(payload, shape)
    assert back.dtype == dtype
    assert (
        torch.equal(back.view(torch.uint8), t.view(torch.uint8))
        if t.element_size() == 1
        else torch.equal(back, t)
    )


def test_planes_compresses_trained_like_fp32():
    t = torch.randn(4, 64, 256) * 0.02
    payload, _ = planes.encode(t)
    assert len(payload) < t.numel() * 4  # the exponent planes code below raw


def test_f32_planes_exact():
    # covers: TRANSPORT-I4
    t = _act(torch.float32, 1)
    p, s = f32_planes.encode(t)
    assert torch.equal(f32_planes.decode(p, s), t)


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_scaled_fp8_wire_is_exact_for_the_recipe_roundtrip(fmt):
    # covers: TRANSPORT-I4
    x = torch.randn(3, 8, 32) * 1e-4  # gradient-scale values an unscaled cast zeroes
    q = fp8.roundtrip(x, fmt)
    assert q.abs().sum() > 0
    payload, shape = planes.encode_scaled_fp8(q, fmt)
    assert torch.equal(planes.decode(payload, shape), q)
    with pytest.raises(ValueError):  # an un-narrowed tensor is a recipe mismatch
        planes.encode_scaled_fp8(x, fmt)


def _handshake(tmp_path, a=None, b=None):
    """Both ends of a socketpair run authenticate with their own key file and
    keyword arguments; a side that fails closes its end so the other sees EOF."""
    sa, sb = socket.socketpair()
    out = {}

    def side(name, sock, kw):
        try:
            out[name] = identity.authenticate(sock, str(tmp_path / f"{name}.key"), **(kw or {}))
        except Exception as e:  # noqa: BLE001
            out[name] = e
            sock.close()

    ts = [
        threading.Thread(target=side, args=("a", sa, a)),
        threading.Thread(target=side, args=("b", sb, b)),
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert not any(t.is_alive() for t in ts), "handshake hung"
    sa.close()
    sb.close()
    return out


def _pub(path):
    key = identity.load_or_create_key(str(path))
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _against_victim(tmp_path, script, victim_kw=None):
    """Run authenticate on one end and a scripted peer on the other; returns
    the victim's result or exception."""
    sv, sm = socket.socketpair()
    out = {}

    def victim():
        try:
            out["v"] = identity.authenticate(sv, str(tmp_path / "v.key"), **(victim_kw or {}))
        except Exception as e:  # noqa: BLE001
            out["v"] = e
            sv.close()

    t = threading.Thread(target=victim)
    t.start()
    try:
        script(sm)
    except ConnectionError:
        pass  # the victim refused and closed; its verdict is what the test checks
    finally:
        t.join(10)
        sm.close()
        sv.close()
    assert not t.is_alive(), "victim hung"
    return out["v"]


def _digest(domain, role, first, second):
    """The signed digest written out from the spec, independently of identity.py:
    SHA-256 of domain + signer role + both pubs, challenges, ephemerals (and AES
    flags) in role order."""

    def lp(b):
        return struct.pack("<H", len(b)) + b

    body = b"".join(first[k] + second[k] for k in ("pub", "chal", "eph"))
    body += bytes([first["aes"], second["aes"]])
    return hashlib.sha256(lp(domain.encode()) + lp(role.encode()) + body).digest()


def _scripted_peer(tmp_path, tamper):
    """A peer holding a valid key that runs the protocol by hand and signs
    the honest digest, or one changed by `tamper`."""
    key = identity.load_or_create_key(str(tmp_path / "m.key"))
    pub = _pub(tmp_path / "m.key")
    eph_priv = X25519PrivateKey.generate()
    eph_pub = eph_priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    chal = os.urandom(16)
    seen = {}

    def script(sock):
        v = identity._recv(sock)
        identity._send(
            sock,
            {
                "mode": "ed25519",
                "pub": pub.hex(),
                "chal": chal.hex(),
                "eph": eph_pub.hex(),
                "aes": False,
                "role": None,
            },
        )
        vic = {
            "pub": bytes.fromhex(v["pub"]),
            "chal": bytes.fromhex(v["chal"]),
            "eph": bytes.fromhex(v["eph"]),
            "aes": v["aes"],
        }
        me = {"pub": pub, "chal": chal, "eph": eph_pub, "aes": False}
        my_tag, other_tag = ("lo", "hi") if chal < vic["chal"] else ("hi", "lo")
        domain = identity.P2P_DOMAIN
        if tamper == "other_transcript":  # a transcript from another session
            vic = dict(vic, chal=os.urandom(16))
        elif tamper == "other_role":
            my_tag = other_tag
        elif tamper == "other_domain":
            domain = identity.CTL_DOMAIN
        first, second = (me, vic) if chal < vic["chal"] else (vic, me)
        seen.update(vic=vic, first=first, second=second, my_tag=my_tag, other_tag=other_tag)
        identity._send(sock, {"sig": key.sign(_digest(domain, my_tag, first, second)).hex()})
        identity._recv(sock)

    script.eph = eph_priv  # type: ignore[attr-defined]
    script.seen = seen  # type: ignore[attr-defined]
    return script, identity.peer_id(pub)


def test_ed25519_handshake_binds_identities(tmp_path):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: the key file is regenerated per connection or created world-readable
    out = _handshake(tmp_path)
    assert out["a"]["mode"] == "ed25519" and out["b"]["mode"] == "ed25519"
    assert out["a"]["peer_id"] != out["b"]["peer_id"]
    # stable identity across connections (the key file persists, 0600)
    again = _handshake(tmp_path)
    assert again["a"]["peer_id"] == out["a"]["peer_id"]
    assert (tmp_path / "a.key").stat().st_mode & 0o777 == 0o600


def test_handshake_binds_transcript_and_derives_mirrored_keys(tmp_path):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: the two ends derive different keys, one key serves both directions, or the keys and transcript
    # do not change with fresh challenges and ephemerals
    out = _handshake(tmp_path)
    a, b = out["a"], out["b"]
    assert a["transcript"] == b["transcript"] and len(a["transcript"]) == 32
    assert a["send_key"] == b["recv_key"] and a["recv_key"] == b["send_key"]
    assert a["send_prefix"] == b["recv_prefix"] and a["recv_prefix"] == b["send_prefix"]
    assert len(a["send_key"]) == 32 and len(a["recv_key"]) == 32
    assert len(a["send_prefix"]) == 4 and len(a["recv_prefix"]) == 4
    assert a["send_key"] != a["recv_key"] and a["send_prefix"] != a["recv_prefix"]
    again = _handshake(tmp_path)  # same identities, fresh session
    assert again["a"]["transcript"] != a["transcript"]
    assert {again["a"]["send_key"], again["a"]["recv_key"]}.isdisjoint(
        {a["send_key"], a["recv_key"]}
    )


def test_session_keys_come_from_the_x25519_secret(tmp_path):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: the session keys are derived without the X25519 shared secret (e.g. HKDF over the transcript
    # or a constant), so a passive observer of the public handshake bytes could compute every AEAD key
    script, mid = _scripted_peer(tmp_path, "honest")
    res = _against_victim(tmp_path, script)
    assert res["peer_id"] == mid
    seen = script.seen

    def lp(b):
        return struct.pack("<H", len(b)) + b

    first, second = seen["first"], seen["second"]
    body = b"".join(first[k] + second[k] for k in ("pub", "chal", "eph"))
    body += bytes([first["aes"], second["aes"]])
    dom = identity.P2P_DOMAIN.encode()
    transcript = hashlib.sha256(lp(dom) + b"transcript" + body).digest()
    assert res["transcript"] == transcript

    def direction(ikm, sender_role):
        okm = HKDF(
            algorithm=hashes.SHA256(),
            length=36,
            salt=transcript,
            info=lp(dom) + b"send|" + sender_role.encode(),
        ).derive(ikm)
        return okm[:32], okm[32:]

    secret = script.eph.exchange(X25519PublicKey.from_public_bytes(seen["vic"]["eph"]))
    # the scripted peer's send direction is the victim's recv direction, and vice versa
    assert (res["recv_key"], res["recv_prefix"]) == direction(secret, seen["my_tag"])
    assert (res["send_key"], res["send_prefix"]) == direction(secret, seen["other_tag"])
    victim_keys = {res["send_key"], res["recv_key"]}
    for public_only in (
        bytes(32),
        transcript,
        body,
    ):  # anything an observer of the wire can compute
        for tag in (seen["my_tag"], seen["other_tag"]):
            assert direction(public_only, tag)[0] not in victim_keys


def test_reflected_challenge_is_refused(tmp_path):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: a peer may answer with our own challenge (a reflection of our hello)
    def echo(sock):
        identity._send(sock, identity._recv(sock))
        identity._recv(sock)

    res = _against_victim(tmp_path, echo)
    assert isinstance(res, ConnectionError) and "reflected" in str(res)


@pytest.mark.parametrize(
    "chal",
    [
        os.urandom(15).hex(),
        os.urandom(17).hex(),
        "",
        " " * 32,  # bytes.fromhex skips whitespace: 32 characters, 0 bytes
        "aa " * 10 + "aa",  # 32 characters, 11 bytes
        "AB" * 16,  # 16 bytes, but not canonical lowercase hex
        "zz" * 16,
    ],
)
def test_challenge_must_be_exactly_16_bytes(tmp_path, chal):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: a short, long, empty, whitespace-padded or non-canonical hex challenge is accepted (the length
    # is checked on the string instead of the decoded bytes)
    def short(sock):
        v = identity._recv(sock)
        identity._send(sock, dict(v, chal=chal, pub=_pub(tmp_path / "m.key").hex()))
        identity._recv(sock)

    res = _against_victim(tmp_path, short)
    assert isinstance(res, ConnectionError) and "'chal'" in str(res)


@pytest.mark.parametrize("tamper", ["honest", "other_transcript", "other_role", "other_domain"])
def test_signature_over_another_transcript_is_refused(tmp_path, tamper):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: the verified digest omits the challenges, the signer's role or the domain (the honest arm
    # proves the scripted peer runs the real protocol, so the refusals are not an artefact of the script)
    script, mid = _scripted_peer(tmp_path, tamper)
    res = _against_victim(tmp_path, script)
    if tamper == "honest":
        assert res["peer_id"] == mid
    else:
        assert isinstance(res, ConnectionError) and "signature" in str(res)


def test_relay_between_two_handshakes_of_one_process_is_refused(tmp_path):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: an on-path relay cross-wires two concurrent role-less handshakes of one process (one key, as
    # a stage's outbound and inbound links) into an authenticated self-loop
    key = str(tmp_path / "self.key")
    own = identity.peer_id(_pub(tmp_path / "self.key"))
    (oa, ob), (ia, ib) = socket.socketpair(), socket.socketpair()
    out = {}

    def side(name, sock):
        try:
            out[name] = identity.authenticate(sock, key, trusted_peers=[own])
        except Exception as e:  # noqa: BLE001
            out[name] = e
            sock.close()

    ts = [
        threading.Thread(target=side, args=("out", oa)),
        threading.Thread(target=side, args=("in", ia)),
    ]
    for t in ts:
        t.start()
    try:
        hello_o, hello_i = (
            identity._recv(ob),
            identity._recv(ib),
        )  # both hellos sent: both in flight
        identity._send(ib, hello_o)
        identity._send(ob, hello_i)
        sig_o, sig_i = identity._recv(ob), identity._recv(ib)
        identity._send(ib, sig_o)
        identity._send(ob, sig_i)
    except (ConnectionError, OSError):
        pass  # a victim refused and closed
    finally:
        ob.close()  # a victim still waiting for a signature sees EOF
        ib.close()
        for t in ts:
            t.join(10)
        oa.close()
        ia.close()
    assert not any(t.is_alive() for t in ts), "handshake hung"
    # each side checks the peer challenge before it signs, so the side checked first always finds the other
    # in flight; the side checked second may instead see EOF. Neither ever completes.
    assert all(isinstance(out[n], ConnectionError) for n in ("out", "in")), out
    assert any("another handshake" in str(out[n]) for n in ("out", "in")), out


def test_p2p_and_control_domains_never_cross(tmp_path):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: the domain is not part of the signed digest
    crossed = _handshake(
        tmp_path, a={"domain": identity.P2P_DOMAIN}, b={"domain": identity.CTL_DOMAIN}
    )
    assert isinstance(crossed["a"], ConnectionError) and isinstance(crossed["b"], ConnectionError)
    ctl = _handshake(tmp_path, a={"domain": identity.CTL_DOMAIN}, b={"domain": identity.CTL_DOMAIN})
    p2p = _handshake(tmp_path)
    assert ctl["a"]["peer_id"] == p2p["a"]["peer_id"]
    assert ctl["a"]["send_key"] == ctl["b"]["recv_key"]
    assert ctl["a"]["transcript"] != p2p["a"]["transcript"]


@pytest.mark.parametrize(
    "ra,rb,ok",
    [
        ("worker", "coordinator", True),
        ("coordinator", "worker", True),
        ("worker", "worker", False),
        ("coordinator", "coordinator", False),
        ("worker", None, False),
        (None, "coordinator", False),
    ],
)
def test_roles_must_be_complementary(tmp_path, ra, rb, ok):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: two workers, two coordinators, or a role-declaring and a role-less side complete a handshake
    out = _handshake(tmp_path, a={"role": ra}, b={"role": rb})
    if ok:
        assert out["a"]["send_key"] == out["b"]["recv_key"]
        assert out["a"]["recv_key"] == out["b"]["send_key"]
    else:
        assert isinstance(out["a"], ConnectionError) and isinstance(out["b"], ConnectionError)


def test_expect_peer_refuses_another_key(tmp_path):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: expect_peer is ignored, or compared before the peer proved its key
    b_id = identity.peer_id(_pub(tmp_path / "b.key"))
    a_id = identity.peer_id(_pub(tmp_path / "a.key"))
    ok = _handshake(tmp_path, a={"expect_peer": b_id})
    assert ok["a"]["peer_id"] == b_id
    bad = _handshake(tmp_path, a={"expect_peer": a_id})
    assert isinstance(bad["a"], ConnectionError) and "expected" in str(bad["a"])
    # a peer holding a key that is not the expected one, even with an honest transcript
    script, mid = _scripted_peer(tmp_path, "honest")
    res = _against_victim(tmp_path, script, victim_kw={"expect_peer": b_id})
    assert mid != b_id and isinstance(res, ConnectionError)


def test_allowlist_refuses_an_unknown_peer(tmp_path):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: trusted_peers is ignored
    first = _handshake(tmp_path)
    b_id = first["a"]["peer_id"]  # what a sees of b
    ok = _handshake(tmp_path, a={"trusted_peers": [b_id]})
    assert ok["a"]["peer_id"] == b_id
    bad = _handshake(tmp_path, a={"trusted_peers": ["not-this-peer"]})
    assert isinstance(bad["a"], ConnectionError)


def test_peer_id_is_sha256_without_blake3(tmp_path):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: the id depends on an optional hash module present on some hosts only
    pub = bytes(range(32))
    assert identity.peer_id(pub) == hashlib.sha256(pub).hexdigest()[:32]
    assert identity.FINGERPRINT_ALGO == "sha256"
    assert "blake3" not in inspect.getsource(identity)
    out = _handshake(tmp_path)
    assert out["a"]["fingerprint_algo"] == "sha256"
    assert out["a"]["peer_id"] == hashlib.sha256(_pub(tmp_path / "b.key")).hexdigest()[:32]


@pytest.mark.parametrize(
    "aes_a,aes_b,cipher",
    [
        (True, True, "aes-256-gcm"),
        (True, False, "chacha20-poly1305"),
        (False, True, "chacha20-poly1305"),
        (False, False, "chacha20-poly1305"),
    ],
)
def test_cipher_is_aes_only_when_both_ends_have_it(tmp_path, aes_a, aes_b, cipher):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: one end picks AES-GCM while the other lacks hardware AES, or the two ends disagree
    out = _handshake(tmp_path, a={"aes_hw": aes_a}, b={"aes_hw": aes_b})
    assert out["a"]["cipher"] == cipher and out["b"]["cipher"] == cipher


def test_handshake_refuses_without_ed25519(monkeypatch, tmp_path):
    # covers: TRANSPORT-I5
    # WOULD-FAIL-IF: a shared-token (HMAC) mode is still reachable when cryptography is missing
    monkeypatch.setattr(identity, "_HAVE_ED", False)
    monkeypatch.setenv("WANTRAIN_SHARED_TOKEN", "s3cret")  # a token no longer opens anything
    a, b = socket.socketpair()
    try:
        with pytest.raises(ConnectionError, match="cryptography"):
            identity.authenticate(a, str(tmp_path / "k"))
        b.setblocking(False)
        with pytest.raises(BlockingIOError):  # nothing was sent before refusing
            b.recv(1)
    finally:
        a.close()
        b.close()
    assert "shared_token" not in inspect.signature(identity.authenticate).parameters
    assert "hmac" not in inspect.getsource(identity).lower()
