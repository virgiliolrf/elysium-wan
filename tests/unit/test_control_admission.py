"""Control-plane admission: with a coordinator token set, a worker joins only
by answering the challenge with the token's HMAC over the sealed control
handshake's transcript. A rented machine that does not hold the token must
never become a slot or a spare, and a worker acts only on the coordinator whose
key it pinned (CONTROL-I16)."""

# WOULD-FAIL-IF: the coordinator admits a worker whose challenge answer was computed with a wrong or missing token, or check_mac accepts a forged MAC

import glob
import hashlib
import json
import os
import socket
import stat
import threading
import time
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from helpers.cluster import LocalCluster

import wantrain.coordinator.coordinator as coord_mod
from wantrain.cli import main as cli_main
from wantrain.coordinator.coordinator import (
    Coordinator,
    CoordinatorConfig,
    Job,
    StartRefused,
    Worker,
    _Run,
)
from wantrain.fault_tolerance.checkpoint import WAVE_STATE
from wantrain.fault_tolerance.liveness import SlotState
from wantrain.resources.node import NodeInfo
from wantrain.runtime.spec import JobSpec
from wantrain.scheduler import admission
from wantrain.transport import identity
from wantrain.transport.control import (
    ChannelClosed,
    check_mac,
    control_accept,
    control_dial,
    token_mac,
)
from wantrain.worker.agent import WorkerAgent, is_loopback


def test_mac_binds_token_and_nonce_and_refuses_forgeries():
    # covers: CONTROL-I16
    mac = token_mac("s3cret", "n1")
    assert check_mac("s3cret", "n1", mac)
    assert not check_mac("other", "n1", mac)
    assert not check_mac("s3cret", "n2", mac)
    assert not check_mac("s3cret", "n1", mac[:-1] + ("0" if mac[-1] != "0" else "1"))
    assert not check_mac("s3cret", "n1", "")
    assert not check_mac("s3cret", "n1", None)


@pytest.mark.slow
def test_only_the_worker_holding_the_token_registers(tmp_path, port_base, monkeypatch):
    # covers: CONTROL-I16
    c = LocalCluster(tmp_path, port_base, token="s3cret")
    try:
        monkeypatch.setenv("WANTRAIN_COORD_TOKEN", "wrong")
        c.add_worker("intruder")
        monkeypatch.setenv("WANTRAIN_COORD_TOKEN", "s3cret")
        c.add_worker("member")
        c.wait_registered(1)
        deadline = time.time() + 60
        while time.time() < deadline and not any("bad token" in e[2] for e in c.coord.events):
            time.sleep(0.2)
        assert any("bad token" in e[2] for e in c.coord.events), c.coord.events
        time.sleep(2.0)  # the intruder may retry; it must never get in
        names = sorted(w.info.name for w in c.coord.workers.values())
        assert names == ["member"], names
    finally:
        c.close()


# ---------------------------------------------------------------- CONTROL-I20


def _i20_coordinator(tmp_path):
    """A real Coordinator and a run at attempt 1 of two slots whose rank and peer
    both differ, (0, 0) on worker n0 and (1, 1) on worker n1, plus a registered
    worker n2 that holds no slot. No socket: messages go through the real
    _conn_loop / _dispatch."""
    coord = Coordinator(config=CoordinatorConfig(state_dir=str(tmp_path / "coord")))
    os.makedirs(tmp_path / "run")
    run = _Run("run-i20", str(tmp_path / "run"))
    spec = JobSpec(steps=8, stages=2)
    run.spec, run.spec_json = spec, spec.to_json()
    run.attempt = 1
    for i in range(3):
        info = NodeInfo(node_id=f"n{i}", name=f"w{i}", host=f"10.0.0.{i}", data_port=1)
        coord.workers[info.node_id] = Worker(info, None, f"session{i}", None)
    for slot, nid in (((0, 0), "n0"), ((1, 1), "n1")):
        run.slots[slot] = nid
        run.states[slot] = SlotState(slot=slot, node_id=nid, launched=True)
    coord.run = run
    return coord, run


def _i20_feed(coord, node_id, script):
    """FAKE at the control-socket boundary, named as such: the connection of
    `node_id` delivers the scripted (msg, body) pairs in order to the real
    _conn_loop, then hangs up. Returns how many reads the loop made."""
    reads = []

    def recv(file_dir=None):
        reads.append(1)
        if len(reads) > len(script):
            raise ChannelClosed("end of script")
        return script[len(reads) - 1]

    w = coord.workers[node_id]
    ch = SimpleNamespace(recv=recv, close=lambda: None, last_rx=0.0)
    w.ch = ch
    coord._conn_loop(w, ch)
    return len(reads)


def _i20_body(path, data):
    with open(path, "wb") as f:
        f.write(data)
    return str(path), hashlib.sha256(data).hexdigest()


def _i20_wave(run, inbox, tag, rank, peer, step, data, attempt=None):
    """The two messages of a one-file wave as the agent ships it: CKPT_PART with
    the file body, then CKPT with its manifest; both name `rank`, `peer` and
    `attempt`, which the connection must hold."""
    body, sha = _i20_body(inbox / f"{tag}.part", data)
    head = {
        "run_id": run.run_id,
        "attempt": run.attempt if attempt is None else attempt,
        "rank": rank,
        "peer": peer,
        "step": step,
    }
    files = [{"name": WAVE_STATE, "sha256": sha, "bytes": len(data)}]
    return [
        (dict(head, kind="CKPT_PART", name=WAVE_STATE, sha256=sha, file=True), body),
        (dict(head, kind="CKPT", manifest={"v": 1, "step": step, "files": files}), b""),
    ]


def _i20_wave_bytes(run, slot, step):
    with open(os.path.join(run.store.wave_path(slot, step), WAVE_STATE), "rb") as f:
        return f.read()


def _i20_warnings(coord, worker_name):
    return [e[2] for e in coord.events if e[1] == "warn" and f" from {worker_name}: " in e[2]]


def test_ckpt_from_another_slots_connection_is_refused(tmp_path):
    # covers: CONTROL-I20
    # WOULD-FAIL-IF: the CKPT_PART or CKPT branch takes the slot from the message's rank/peer
    # (the mutant `slot = (int(msg["rank"]), int(msg["peer"]))`): n1's connection then commits a
    # wave as slot (0, 0)'s, and a worker holding no slot (n2) commits one at all; or a refused
    # part's body is left in the inbox
    coord, run = _i20_coordinator(tmp_path)
    inbox = tmp_path / "inbox"
    os.makedirs(inbox)
    script = (
        _i20_wave(run, inbox, "forged", 0, 0, 2, b"forged-by-n1-as-r0p0")
        + _i20_wave(run, inbox, "peer", 1, 0, 3, b"n1-names-peer-0")
        + _i20_wave(run, inbox, "own", 1, 1, 4, b"own-shard-of-r1p1-step-4")
    )
    assert _i20_feed(coord, "n1", script) == len(script) + 1  # only the hang-up ended it
    spare = _i20_wave(run, inbox, "spare", 0, 0, 5, b"sent-by-slotless-n2")
    assert _i20_feed(coord, "n2", spare) == len(spare) + 1

    assert run.store.index() == {(1, 1): [4]}, run.store.index()
    assert _i20_wave_bytes(run, (1, 1), 4) == b"own-shard-of-r1p1-step-4"
    assert os.listdir(inbox) == []  # refused bodies unlinked, the accepted one moved
    assert len([t for t in _i20_warnings(coord, "w1") if "CKPT" in t]) == 4, coord.events
    assert len([t for t in _i20_warnings(coord, "w2") if "CKPT" in t]) == 2, coord.events


def test_ckpt_of_another_attempt_is_refused(tmp_path):
    # covers: CONTROL-I20
    # WOULD-FAIL-IF: the CKPT_PART or CKPT branch does not check the attempt (the mutant drops
    # the `attempt == run.attempt` test, or coerces it with int() so True or "1" pass), or
    # accepts a step that is not a non-negative int (int() coercion stores "6" as 6, 7.5 as 7,
    # True as 1; -1 stored as a wave): any of those waves then sits in the store next to step 5
    coord, run = _i20_coordinator(tmp_path)
    inbox = tmp_path / "inbox"
    os.makedirs(inbox)
    script: list = []
    bad = [
        dict(attempt=0, step=3),  # the previous attempt
        dict(attempt=2, step=4),  # a later attempt than the coordinator's
        dict(attempt=True, step=2),  # a bool is not the attempt
        dict(attempt="1", step=0),  # nor a string
        dict(step=-1),
        dict(step=True),
        dict(step="6"),
        dict(step=7.5),
        dict(step=None),
    ]
    for i, kw in enumerate(bad):
        step = kw.pop("step")
        script += _i20_wave(run, inbox, f"bad{i}", 1, 1, step, f"bad-{i}".encode(), **kw)
    script += _i20_wave(run, inbox, "good", 1, 1, 5, b"r1p1-attempt-1-step-5")
    assert _i20_feed(coord, "n1", script) == len(script) + 1

    assert run.store.index() == {(1, 1): [5]}, run.store.index()
    assert _i20_wave_bytes(run, (1, 1), 5) == b"r1p1-attempt-1-step-5"
    assert os.listdir(inbox) == []
    assert len([t for t in _i20_warnings(coord, "w1") if "CKPT" in t]) == 2 * len(bad), coord.events


def test_forged_result_is_refused(tmp_path):
    # covers: CONTROL-I20
    # WOULD-FAIL-IF: the RESULT branch keys run.results by the message's rank/peer (the mutant
    # `run.results[(int(msg["rank"]), int(msg["peer"]))] = msg`): n1's forged losses then become
    # slot (0, 0)'s result, which _complete reports as the run's losses; or a RESULT from a
    # worker holding no slot, or of another attempt, is stored
    coord, run = _i20_coordinator(tmp_path)

    def result(rank, peer, losses, attempt=None):
        return (
            {
                "kind": "RESULT",
                "run_id": run.run_id,
                "attempt": run.attempt if attempt is None else attempt,
                "rank": rank,
                "peer": peer,
                "losses": losses,
            },
            b"",
        )

    _i20_feed(
        coord,
        "n1",
        [
            result(0, 0, [9.5, 9.25]),  # forged: names slot (0, 0)
            result(1, 1, [3.5], attempt=0),  # a stale attempt
            result(1, 1, [2.75, 1.5]),  # its own
        ],
    )
    _i20_feed(coord, "n2", [result(0, 0, [8.0])])  # holds no slot
    assert set(run.results) == {(1, 1)}, run.results
    assert run.results[(1, 1)]["losses"] == [2.75, 1.5]
    _i20_feed(coord, "n0", [result(0, 0, [4.25, 0.5])])
    assert run.results[(0, 0)]["losses"] == [4.25, 0.5]
    assert run.results[(1, 1)]["losses"] == [2.75, 1.5]
    assert len([t for t in _i20_warnings(coord, "w1") if "RESULT" in t]) == 2, coord.events
    assert len([t for t in _i20_warnings(coord, "w2") if "RESULT" in t]) == 1, coord.events


def test_reply_from_another_worker_is_ignored(tmp_path):
    # covers: CONTROL-I20
    # WOULD-FAIL-IF: _dispatch hands any message whose rid is pending to the waiting request
    # without checking the connection it came from (the mutant drops the node_id comparison):
    # n1's forged reply then completes n0's request and _request returns {"v": "forged"}; or a
    # refused reply's body is left in the inbox
    coord, run = _i20_coordinator(tmp_path)
    inbox = tmp_path / "inbox"
    os.makedirs(inbox)
    sent: list = []
    asked = threading.Event()

    def send(m, body=b""):  # FAKE at the control-socket boundary: n0's outgoing side
        sent.append(m)
        asked.set()

    w0 = coord.workers["n0"]
    w0.ch = SimpleNamespace(send=send, last_rx=0.0)
    out: dict = {}
    t = threading.Thread(
        target=lambda: out.update(reply=coord._request(w0, {"kind": "PING"}, timeout=20))
    )
    t.start()
    try:
        assert asked.wait(10)
        rid = sent[0]["rid"]
        forged, _ = _i20_body(inbox / "forged.part", b"forged reply body")
        assert _i20_feed(coord, "n1", [({"kind": "PONG", "rid": rid, "v": "forged"}, forged)]) == 2
        assert os.listdir(inbox) == []
        assert t.is_alive()  # the forged reply completed nothing
        coord._dispatch(w0, {"kind": "PONG", "rid": rid, "v": "real"}, b"")
    finally:
        t.join(20)
    assert not t.is_alive()
    assert out["reply"]["v"] == "real", out
    assert sent[0]["kind"] == "PING"
    assert [t for t in _i20_warnings(coord, "w1") if "PONG" in t], coord.events
    assert coord._pending == {}


def test_handshake_file_body_is_refused_and_unlinked(tmp_path, monkeypatch):
    # covers: CONTROL-I20
    # WOULD-FAIL-IF: _handshake reads HELLO with a file directory, or accepts a HELLO that streamed
    # a file body: the body's .part file is then left on the coordinator's disk, so each
    # unauthenticated attempt leaks a file there; or if it reads the refused body instead
    # of closing on the header (the client's send then completes instead of failing)
    import tempfile

    tmpdir = tmp_path / "tmp"
    os.makedirs(tmpdir)
    monkeypatch.setattr(tempfile, "tempdir", str(tmpdir))
    coord = Coordinator(config=CoordinatorConfig(state_dir=str(tmp_path / "coord"), token="s3cret"))
    a, b = socket.socketpair()
    t = threading.Thread(target=coord._handshake, args=(a, ("10.9.9.9", 1)))
    t.start()
    try:
        client, _ = control_dial(b, str(tmp_path / "w_key"))
        challenge, _ = client.recv()
        assert challenge["kind"] == "CHALLENGE"
        # The coordinator refuses on the header and closes without reading the
        # body (control.py _recv), so a body larger than any socketpair buffer
        # always meets the closed peer mid-stream: the send fails with EPIPE
        # (or ECONNRESET), never by chance. A 4 KiB body raced the close.
        payload = tmp_path / "payload"
        payload.write_bytes(b"\0" * (16 << 20))
        with pytest.raises(ChannelClosed) as refused:
            client.send_file({"kind": "HELLO", "mac": "00"}, str(payload))
        assert isinstance(refused.value.__cause__, (BrokenPipeError, ConnectionResetError))
        t.join(20)
        # The coordinator closed without a WELCOME (or anything else).
        with pytest.raises(ChannelClosed):
            client.recv()
    finally:
        t.join(20)
        b.close()
    assert not t.is_alive()
    assert os.listdir(tmpdir) == []
    assert not os.path.exists(tmp_path / "coord" / "incoming")
    assert coord.workers == {}
    assert any("file body" in e[2] for e in coord.events), coord.events


# ---------------------------------------------------------------- CONTROL-I16

TOKEN = "s3cret-token"


def _key_id(path) -> str:
    key = identity.load_or_create_key(str(path))
    return identity.peer_id(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def _coordinator(tmp_path, token=TOKEN, name="coord"):
    cfg = CoordinatorConfig(state_dir=str(tmp_path / name), token=token)
    coord = Coordinator("127.0.0.1", 0, cfg)
    coord.start()
    return coord


def _agent(tmp_path, name, port, token=TOKEN, coordinator_id=None, **kw):
    """A real WorkerAgent driven in-process through its own _connect (no stage,
    no probe server, no benchmark); its key is <workdir>/peer_key."""
    return WorkerAgent(
        f"127.0.0.1:{port}",
        "127.0.0.1",
        1,
        probe_port=-1,
        name=name,
        workdir=str(tmp_path / name),
        token=token,
        device="cpu",
        bench=False,
        coordinator_id=coordinator_id,
        **kw,
    )


class _FakeCoordinator:
    """FAKE at the network boundary, named as such: a listener that runs the
    real control_accept under its own key, then plays `script(ch)` (a hostile
    coordinator). Records every connection's handshake outcome and every
    message it received."""

    def __init__(self, key_path, script):
        self.key_path = str(key_path)
        self.script = script
        self.lst = socket.create_server(("127.0.0.1", 0))
        self.lst.settimeout(0.2)
        self.port = self.lst.getsockname()[1]
        self.connections = 0
        self.received: list = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                sock, _ = self.lst.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections += 1
            sock.settimeout(5)
            try:
                ch, _ = control_accept(sock, self.key_path)
                self.script(ch, self.received)
            except (OSError, ChannelClosed, ValueError):
                pass
            finally:
                sock.close()

    def close(self):
        self._stop.set()
        self.lst.close()
        self._t.join(5)


def _hostile_assign(ch, received):
    """Admit anyone and order an ASSIGN at once, whatever the worker says."""
    ch.send({"kind": "CHALLENGE", "auth": True})
    ch.send({"kind": "WELCOME", "node_id": "evil", "session": "0" * 32})
    ch.send({"kind": "ASSIGN", "run_id": "evil", "attempt": 1, "rank": 0, "peer": 0}, b"{}")
    try:
        received.append(ch.recv()[0])
    except (OSError, ChannelClosed):
        pass


def _connect_fails(agent):
    with pytest.raises((OSError, ChannelClosed)) as err:
        agent._connect()
    return str(err.value)


def test_worker_refuses_a_coordinator_with_another_key(tmp_path, capsys):
    # covers: CONTROL-I16
    # WOULD-FAIL-IF: the agent dials without expect_peer = its pinned --coordinator-id (or the
    # handshake ignores it): the hostile coordinator, holding the right token but another key,
    # then gets the HELLO and its WELCOME/ASSIGN are acted on (node_id "evil", an assignment);
    # or the agent stops at the first refusal instead of retrying within its window
    real_id = _key_id(tmp_path / "real_coord_key")
    fake = _FakeCoordinator(tmp_path / "fake_coord_key", _hostile_assign)
    try:
        agent = _agent(tmp_path, "w", fake.port, coordinator_id=real_id, reconnect_window_s=2.0)
        t = threading.Thread(target=agent.run, daemon=True)
        t.start()
        t.join(60)
        agent._stop.set()  # ends its heartbeat thread
        assert not t.is_alive()
        assert fake.connections >= 2, fake.connections  # it retried inside its window
        assert fake.received == []  # never a HELLO
        assert agent.node_id is None and agent.session is None
        assert agent.assignment is None and agent.stage is None
        out = capsys.readouterr().out
        assert f"is not the expected peer {real_id}" in out, out
        assert "giving up" in out, out
    finally:
        fake.close()

    # The same hostile listener under the pinned key is believed: the pin is what refused.
    fake = _FakeCoordinator(tmp_path / "real_coord_key", _hostile_assign)
    try:
        agent = _agent(tmp_path, "w2", fake.port, coordinator_id=real_id)
        agent._connect().close()
        assert agent.node_id == "evil"
    finally:
        fake.close()


def test_admitted_worker_cannot_pose_as_the_coordinator(tmp_path):
    # covers: CONTROL-I16
    # WOULD-FAIL-IF: a worker accepts any coordinator that holds the token (the pre-amendment
    # proof derived from the token, or no expect_peer): worker B, admitted and holding the token,
    # then serves A's control connection under its own key and A sends it a HELLO
    coord = _coordinator(tmp_path)
    fake = None
    try:
        b = _agent(tmp_path, "b", coord.port, coordinator_id=coord.coord_id)
        ch_b = b._connect()  # B is an admitted worker
        assert coord.workers[b.node_id].peer_id == _key_id(b.key_path)
        fake = _FakeCoordinator(b.key_path, _hostile_assign)
        a = _agent(tmp_path, "a", fake.port, coordinator_id=coord.coord_id)
        why = _connect_fails(a)
        assert f"is not the expected peer {coord.coord_id}" in why, why
        assert fake.received == [] and a.node_id is None
        # A, pinned to the real id, registers with the real coordinator.
        a2 = _agent(tmp_path, "a", coord.port, coordinator_id=coord.coord_id)
        ch_a = a2._connect()
        assert coord.workers[a2.node_id].peer_id == _key_id(a2.key_path)
        ch_a.close()
        ch_b.close()
    finally:
        if fake is not None:
            fake.close()
        coord.stop()


def test_relayed_hello_cannot_swap_the_worker_key(tmp_path):
    # covers: CONTROL-I16
    # WOULD-FAIL-IF: the token MAC is not bound to the handshake transcript (a MAC over a
    # coordinator nonce or over nothing): a relay that terminates the worker's handshake with
    # its own key and forwards the worker's HELLO to the coordinator then registers the relay's
    # key as the worker's
    coord = _coordinator(tmp_path)
    relay_key = str(tmp_path / "relay_key")
    outcome: dict = {}

    def relay(ch_w, received):
        ch_w.send({"kind": "CHALLENGE", "auth": True})
        hello, _ = ch_w.recv()
        received.append(hello)
        up = socket.create_connection(("127.0.0.1", coord.port), timeout=10)
        try:
            ch_c, _ = control_dial(up, relay_key, coord.coord_id)
            outcome["challenge"] = ch_c.recv()[0]
            ch_c.send(hello)
            outcome["reply"] = ch_c.recv()[0]
        finally:
            up.close()

    fake = _FakeCoordinator(relay_key, relay)
    try:
        victim = _agent(tmp_path, "v", fake.port)  # local mode: no pin on loopback
        _connect_fails(victim)
        assert fake.received and fake.received[0]["kind"] == "HELLO"
        assert outcome["challenge"]["kind"] == "CHALLENGE"
        assert outcome["reply"] == {"kind": "REJECT", "reason": "bad token"}, outcome
        assert coord.workers == {}
        # The same worker dialing directly is admitted: its MAC was valid, for its own transcript.
        direct = _agent(tmp_path, "v", coord.port, coordinator_id=coord.coord_id)
        direct._connect().close()
        assert [w.peer_id for w in coord.workers.values()] == [_key_id(direct.key_path)]
    finally:
        fake.close()
        coord.stop()


def test_worker_refuses_a_downgraded_challenge(tmp_path):
    # covers: CONTROL-I16
    # WOULD-FAIL-IF: a worker holding a token answers CHALLENGE{auth: false} (a downgrade by a
    # coordinator that wants the worker without proving the token is required): the fake, even
    # under the pinned key, then receives a HELLO
    key = tmp_path / "coord_key"

    def downgrade(ch, received):
        ch.send({"kind": "CHALLENGE", "auth": False})
        received.append(ch.recv()[0])

    fake = _FakeCoordinator(key, downgrade)
    try:
        agent = _agent(tmp_path, "w", fake.port, coordinator_id=_key_id(key))
        why = _connect_fails(agent)
        assert "downgrade" in why, why
        deadline = time.time() + 10
        while fake.connections < 1 and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        assert fake.received == [], fake.received
    finally:
        fake.close()

    def honest(ch, received):
        ch.send({"kind": "CHALLENGE", "auth": True})
        received.append(ch.recv()[0])

    fake = _FakeCoordinator(key, honest)
    try:
        agent = _agent(tmp_path, "w", fake.port, coordinator_id=_key_id(key))
        _connect_fails(agent)  # the fake never answers WELCOME
        assert [m["kind"] for m in fake.received] == ["HELLO"]
        mac = fake.received[0]["mac"]
        assert len(mac) == 64 and mac != token_mac(TOKEN, b"another transcript")
    finally:
        fake.close()


def test_coordinator_refuses_to_start_off_loopback_without_a_token(tmp_path, monkeypatch, capsys):
    # covers: CONTROL-I16
    # WOULD-FAIL-IF: start() only warns when it binds a non-loopback address without a token,
    # or the loopback test accepts 0.0.0.0 or a name that merely starts with "localhost"; or the
    # coordinator key is not 0600, not reused across restarts, or its id not logged
    for host in ("127.0.0.1", "127.5.6.7", "::1", "localhost", "LocalHost", "[::1]"):
        assert is_loopback(host), host
    for host in ("0.0.0.0", "::", "10.0.0.1", "192.168.1.2", "localhost.example", "", "lo"):
        assert not is_loopback(host), host

    for host in ("0.0.0.0", "10.0.0.1"):
        state = tmp_path / f"refused-{host}"
        coord = Coordinator(host, 0, CoordinatorConfig(state_dir=str(state)))
        with pytest.raises(StartRefused, match="without a token"):
            coord.start()
        assert coord._lst is None and coord.coord_id is None
        assert not os.path.exists(state / "coord_key")

    coord = Coordinator("0.0.0.0", 0, CoordinatorConfig(state_dir=str(tmp_path / "t"), token="x"))
    coord.start()
    coord.stop()

    state = tmp_path / "local"
    coord = Coordinator("localhost", 0, CoordinatorConfig(state_dir=str(state)))
    coord.start()
    coord.stop()
    key = state / "coord_key"
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600
    assert coord.coord_id == _key_id(key) and len(coord.coord_id) == 32
    assert any(e[2] == f"COORDINATOR_ID {coord.coord_id}" for e in coord.events), coord.events
    again = Coordinator("127.0.0.1", 0, CoordinatorConfig(state_dir=str(state)))
    again.start()
    again.stop()
    assert again.coord_id == coord.coord_id

    monkeypatch.delenv("WANTRAIN_COORD_TOKEN", raising=False)
    monkeypatch.delenv("WANTRAIN_COORD_ID", raising=False)
    capsys.readouterr()
    rc = cli_main(
        ["coordinator", "--host", "0.0.0.0", "--port", "0", "--state-dir", str(tmp_path / "cli")]
        + ["--job", str(tmp_path / "no-such-job.json")]
    )
    err = capsys.readouterr().err
    assert rc != 0 and "without a token" in err, (rc, err)

    worker = ["worker", "--advertise-host", "10.0.0.2", "--data-port", "1"]
    worker += ["--workdir", str(tmp_path / "wk")]
    rc = cli_main(worker + ["--coordinator", "10.0.0.1:28000"])
    err = capsys.readouterr().err
    assert rc != 0 and "--coordinator-id" in err, (rc, err)
    rc = cli_main(worker + ["--coordinator", "10.0.0.1:28000", "--coordinator-id", "ABC"])
    err = capsys.readouterr().err
    assert rc != 0 and "32 lowercase hex" in err, (rc, err)
    pinned = WorkerAgent(
        "10.0.0.1:28000", "10.0.0.2", 1, workdir=str(tmp_path / "wk"), coordinator_id="0" * 32
    )
    assert pinned.coordinator_id == "0" * 32  # pinned: accepted off loopback


class _Tap:
    """FAKE at the network boundary, named as such: a TCP relay that forwards
    every byte unchanged in both directions and records what crossed it."""

    def __init__(self, target_port):
        self.target = ("127.0.0.1", target_port)
        self.lst = socket.create_server(("127.0.0.1", 0))
        self.lst.settimeout(0.2)
        self.port = self.lst.getsockname()[1]
        self.wire = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._conns: list[socket.socket] = []
        threading.Thread(target=self._loop, daemon=True).start()

    def _pump(self, src, dst):
        try:
            while True:
                b = src.recv(65536)
                if not b:
                    break
                with self._lock:
                    self.wire += b
                dst.sendall(b)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _loop(self):
        while not self._stop.is_set():
            try:
                a, _ = self.lst.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with self._lock:
                self._conns.append(a)
            b = socket.create_connection(self.target)
            with self._lock:
                self._conns.append(b)
            threading.Thread(target=self._pump, args=(a, b), daemon=True).start()
            threading.Thread(target=self._pump, args=(b, a), daemon=True).start()

    def close(self):
        self._stop.set()
        self.lst.close()
        with self._lock:
            conns, self._conns = self._conns, []
        for c in conns:
            c.close()


def _wait(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def test_session_secret_never_travels_in_clear(tmp_path, monkeypatch):
    # covers: CONTROL-I16
    # WOULD-FAIL-IF: any control message travels before the channel is sealed (WELCOME's session
    # or the HELLO then appear in the captured bytes), or the reconnect HELLO carries the session
    # itself instead of session_mac (the coordinator then receives the session in a HELLO field)
    coord = _coordinator(tmp_path)
    hellos: list = []
    register = coord._register

    def spy(hello, ch, peer, transcript):  # the real _register, its input recorded
        hellos.append(json.dumps(hello))
        return register(hello, ch, peer, transcript)

    monkeypatch.setattr(coord, "_register", spy)
    tap = _Tap(coord.port)
    try:
        agent = _agent(tmp_path, "w", tap.port, coordinator_id=coord.coord_id)
        ch = agent._connect()
        nid, session = agent.node_id, agent.session
        w = coord.workers[nid]
        assert session == w.session and len(session) == 32
        ch.close()
        assert _wait(lambda: not w.connected)
        ch = agent._connect()  # the reconnect proves the session
        assert agent.node_id == nid and agent.session == session
        assert _wait(lambda: w.connected and w.ch is not None and not w.ch.closed)
        assert any("reconnected" in e[2] for e in coord.events), coord.events
        ch.close()
        assert _wait(lambda: not w.connected)
        with tap._lock:
            wire = bytes(tap.wire)
        assert len(wire) > 1000
        for secret in (session, TOKEN, "WELCOME", "HELLO", "CHALLENGE", nid):
            assert secret.encode() not in wire, secret
        assert len(hellos) == 2 and "session_mac" in hellos[1]
        assert all(session not in h for h in hellos), hellos
    finally:
        tap.close()
        coord.stop()


def test_reconnect_from_another_key_is_refused(tmp_path):
    # covers: CONTROL-I16
    # WOULD-FAIL-IF: a reconnect is accepted on the node id and session alone, not only from the
    # key the node registered with (or on a session MAC that does not verify): agent B, holding
    # the token and A's stolen node id and session, then takes over A's worker record
    coord = _coordinator(tmp_path)
    try:
        a = _agent(tmp_path, "a", coord.port, coordinator_id=coord.coord_id)
        ch_a = a._connect()
        w = coord.workers[a.node_id]
        b = _agent(tmp_path, "b", coord.port, coordinator_id=coord.coord_id)
        b.node_id, b.session = a.node_id, a.session
        with pytest.raises(PermissionError, match="not its registered key"):
            b._connect()
        assert w.ch is not None and not w.ch.closed and w.peer_id == _key_id(a.key_path)
        assert list(coord.workers) == [a.node_id]

        ch_a.close()
        assert _wait(lambda: not w.connected)
        good = a.session
        a.session = "f" * 32  # the right key with a wrong session
        with pytest.raises(PermissionError, match="session MAC"):
            a._connect()
        a.session = good
        a._connect().close()  # the right key and session: reconnected
        # the coordinator logs the reconnect on its own thread, after the handshake returns
        assert _wait(lambda: any("reconnected" in e[2] for e in coord.events)), coord.events
        assert list(coord.workers) == [a.node_id]
    finally:
        coord.stop()


@pytest.mark.slow
def test_sealed_control_channel_runs_a_job(tmp_path, port_base, monkeypatch):
    # covers: CONTROL-I16
    # WOULD-FAIL-IF: the sealed, token-protected control channel changes, drops or reorders a
    # control message or a shipped wave (ASSIGN, CKPT_PART, CKPT, RESULT): the run then fails, or
    # its losses or final state differ by one bit from the single-device reference; or a worker
    # is registered under a key other than its own <workdir>/peer_key
    from helpers.exact import fleet_manifest, reference_manifest_for

    from wantrain.launch.local import ensure_reference
    from wantrain.model.gpt import make_config
    from wantrain.validation.compare import compare
    from wantrain.validation.state import diff_manifests

    c = LocalCluster(tmp_path, port_base, token=TOKEN, probe=False)
    try:
        monkeypatch.setenv("WANTRAIN_COORD_TOKEN", TOKEN)
        monkeypatch.setenv("WANTRAIN_COORD_ID", c.coord.coord_id)
        c.add_worker("a")
        c.add_worker("b")
        c.wait_registered(2)
        for w in c.coord.workers.values():
            assert w.peer_id == _key_id(os.path.join(c.workdirs[w.info.name], "peer_key"))
        recipe = JobSpec(
            model=make_config("toy"), steps=4, microbatches=4, recompute=True, stages=2
        )
        res = c.run(Job(spec=recipe))
        assert res.ok, res.reason
        ref = ensure_reference(JobSpec.from_json(recipe.to_json()), str(tmp_path / "ref"))
        r = compare(ref, os.path.join(res.run_dir, "losses.json"))
        assert r["ok"] and r["compared"] == recipe.steps * recipe.microbatches, r
        state = fleet_manifest(c, res, tmp_path / "collected")
        reference = reference_manifest_for(recipe, tmp_path / "ref")
        assert diff_manifests(state, reference) == []
        assert state == reference
    finally:
        c.close()


def _unarmed_coordinator(tmp_path):
    """A real Coordinator with no worker. FAKE at the worker-network boundary,
    named as such: `_request` records each request (an ASSIGN would land here)."""
    coord = Coordinator(config=CoordinatorConfig(state_dir=str(tmp_path / "coord")))
    sent = []
    coord._request = lambda w, msg, *a, **k: sent.append(msg["kind"]) or {"ok": True}
    return coord, sent


def _spec_with(**fields):
    spec = JobSpec(steps=2, stages=1, stage_widths=[2])
    for k, v in fields.items():
        setattr(spec, k, v)  # the JobSpec fields may not exist yet (interface I-9)
    return spec


@pytest.mark.parametrize(
    "job, field",
    [
        (Job(spec=_spec_with(shard="pieces")), "shard 'pieces'"),
        (Job(plan={"model": {}, "shard": "pieces"}, stages=2), "shard 'pieces'"),
        (Job(spec=_spec_with(wire="native")), "wire 'native'"),
        (Job(plan={"model": {}, "wire": "native"}, stages=2), "wire 'native'"),
    ],
    ids=["spec_shard", "plan_shard", "spec_wire", "plan_wire"],
)
def test_pieces_unarmed_run_job_refuses_before_setup(tmp_path, monkeypatch, job, field):
    # covers: CONTROL-I9
    # WOULD-FAIL-IF: run_job reaches setup (selection, ASSIGN) for a job whose spec or plan
    # asks for shard 'pieces' or wire 'native' while PIECES_ARMED is False, or reads only one
    # of spec and plan, so a production job runs a mode whose legs have not all landed
    monkeypatch.setattr(coord_mod, "PIECES_ARMED", False)
    coord, sent = _unarmed_coordinator(tmp_path)

    def no_setup(*a, **k):
        raise AssertionError("setup reached")

    coord._setup = no_setup
    res = coord.run_job(job, wait_workers_s=0.0)
    assert not res.ok and field in res.reason and "not armed" in res.reason
    assert sent == []


def test_pieces_unarmed_control_zero1_fp32_and_bare_plan_pass_the_check(tmp_path, monkeypatch):
    # covers: CONTROL-I9
    # WOULD-FAIL-IF: the refusal also catches a zero1 fp32 spec or a plan without shard and
    # wire, so every existing coordinator job is refused; or arming does not lift it
    monkeypatch.setattr(coord_mod, "PIECES_ARMED", False)
    jobs = [
        Job(spec=_spec_with(shard="zero1", wire="fp32")),
        Job(spec=JobSpec(steps=2, stages=2)),
        Job(plan={"model": {}}, stages=2),
    ]
    for i, job in enumerate(jobs):
        assert coord_mod.unarmed_reason(job) is None
        coord, _ = _unarmed_coordinator(tmp_path / str(i))
        res = coord.run_job(job, wait_workers_s=0.0)
        # It went on to wait for workers, past the check, and found none.
        assert not res.ok and "workers registered" in res.reason, res.reason
    monkeypatch.setattr(coord_mod, "PIECES_ARMED", True)
    assert coord_mod.unarmed_reason(Job(spec=_spec_with(shard="pieces", wire="native"))) is None


def test_pieces_stage_cli_refuses_shard_pieces_before_launch(tmp_path, monkeypatch, capsys):
    # covers: CONTROL-I9
    # WOULD-FAIL-IF: 'wantrain stage' launches a one-slot pieces spec (it cannot verify its
    # peers' class, BA-52), or reads shard only through JobSpec.load, which may not know it
    import wantrain.worker.stage_main as stage_main

    launched = []
    monkeypatch.setattr(stage_main, "run_stage", lambda *a: launched.append(a) or 0)
    d = json.loads(JobSpec(steps=2, stages=1, stage_widths=[2]).to_json())
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(dict(d, shard="pieces")))
    rc = cli_main(["stage", "--spec", str(path), "--rank", "0", "--peer", "1"])
    assert rc == 2 and launched == []
    assert "shard 'pieces'" in capsys.readouterr().err
    # control: a zero1 spec is handed to the stage entry unchanged
    path.write_text(json.dumps(dict(d, shard="zero1")))
    assert cli_main(["stage", "--spec", str(path), "--rank", "0", "--peer", "1"]) == 0
    assert launched == [(str(path), 0, 1)]


# ---------------------------------------------------------------- M2b: plan mode for wide stages

PW_JOB = {"model": "toy", "steps": 3, "device": "cpu", "mb_size": 64, "microbatches": 8}
PW_NIC_MBPS = 16_000.0  # a loopback link probe's order of magnitude


def _pw_info(i, hostname="mac", **kw):
    """A registered cpu worker as its HELLO reports it (probed class and host
    capacities, measured)."""
    base = dict(
        node_id=f"n{i}",
        name=f"w{i}",
        host="127.0.0.1",
        data_port=7000 + 8 * i,
        device="cpu",
        gpu_name="cpu",
        vram_total_bytes=16 << 30,
        vram_free_bytes=16 << 30,
        tflops_fp32=1.0,
        hostname=hostname,
        host_ram_bytes=16 << 30,
        host_dram_gbps=34.0,
        device_class="cpu:arm64:DEFAULT",
    )
    base.update(kw)
    return NodeInfo(**base)


def _pw_coordinator(tmp_path, infos, **cfg):
    """A real Coordinator with `infos` registered. FAKE at the worker-network
    boundary, named as such: the workers have no channel, so every request
    (PROFILE, FIT) fails as to a lost worker unless a test replaces
    `_request` or `_profile`."""
    cfg.setdefault("fit_check", False)
    coord = Coordinator(config=CoordinatorConfig(state_dir=str(tmp_path / "coord"), **cfg))
    for i, info in enumerate(infos):
        coord.workers[info.node_id] = Worker(info, None, f"s{i}", None, peer_id=f"{i:032x}")
    os.makedirs(tmp_path / "run", exist_ok=True)
    return coord, _Run("run-pw", str(tmp_path / "run"))


def _pw_plan(tmp_path, infos, job=None, nic="measured", coord=None, devices=None):
    """The coordinator's plan step, called directly: (spec, rationale, groups)."""
    if coord is None:
        coord, run = _pw_coordinator(tmp_path, infos)
    else:
        run = _Run("run-pw", str(tmp_path / "run"))
    if nic == "measured":
        nic = {n.node_id: (PW_NIC_MBPS, PW_NIC_MBPS) for n in infos}
    j = Job(plan=dict(job or PW_JOB), devices=devices or len(infos))
    return coord._plan(run, j, infos, [], None, nic, {})


def _pw_unarmed_kind(c):
    return c["w"] > 1 and (c["shard"] == "pieces" or c["wire"] == "native")


def test_plan_wide_fleet_inputs_per_hostname_with_their_sources(tmp_path):
    """BA-48: the plan fleet has one host RAM and one host-DRAM figure per
    hostname (the largest of its workers' measurements), PCIe 'absent' on cpu
    nodes, the NIC from the link probes' bidirectional transfer ('measured')
    or, with probing off, the declared uplink and downlink ('declared'), and
    k the counted figure; the slots are grouped by the planner's placement."""
    # covers: CONTROL-I4
    # WOULD-FAIL-IF: the coordinator keys host capacities by node label (each worker its own
    # RAM and DRAM figure, so two workers of one machine count its RAM twice), takes the NIC
    # from anything but the probes' bidirectional transfer, marks a declared figure measured,
    # passes a k other than K_HOST_DRAM_COUNTED, or forms groups other than the placement
    ram = [8 << 30, 12 << 30, 16 << 30, 10 << 30]
    dram = [30.0, 40.0, 20.0, 25.0]
    infos = [
        _pw_info(i, hostname="hA" if i < 2 else "hB", host_ram_bytes=ram[i], host_dram_gbps=dram[i])
        for i in range(4)
    ]
    hops = [{"bidir_forward_mbps": 900.0 + i, "bidir_reverse_mbps": 700.0 + i} for i in range(3)]
    nic = coord_mod.nic_from_hops(infos, hops)
    # node i dialed i + 1: its up is the forward direction, i + 1's down
    assert nic == {
        "n0": (900.0, 700.0),
        "n1": (901.0, 900.0),
        "n2": (902.0, 901.0),
        "n3": (702.0, 902.0),
    }
    nodes = coord_mod.plan_fleet_nodes(infos, nic)

    def one_figure_per_host(nodes):
        want = {"hA": (12 << 30, 40.0), "hB": (16 << 30, 25.0)}
        return all((n["host_ram_bytes"], n["host_dram_gbps"]) == want[n["hostname"]] for n in nodes)

    assert one_figure_per_host(nodes)
    # control: keyed by node label, every worker keeps its own figures, and the check fails
    by_label = [dict(n, host_ram_bytes=ram[i], host_dram_gbps=dram[i]) for i, n in enumerate(nodes)]
    assert not one_figure_per_host(by_label)
    for n, info in zip(nodes, infos):
        assert n["sources"]["host_ram_bytes"] == n["sources"]["host_dram_gbps"] == "measured"
        assert n["pcie_h2d_gbps"] is n["pcie_d2h_gbps"] is None
        assert n["sources"]["pcie_h2d_gbps"] == n["sources"]["pcie_d2h_gbps"] == "absent"
        up, down = nic[info.node_id]
        assert (n["nic_up_gbps"], n["nic_down_gbps"]) == (up / 1e3, down / 1e3)
        assert n["sources"]["nic_up_gbps"] == n["sources"]["nic_down_gbps"] == "measured"
        assert (n["device"], n["gpu_name"], n["hostname"]) == ("cpu", "cpu", info.hostname)
        assert n["sources"]["device"] == n["sources"]["gpu_name"] == "measured"
    # probing off: the declared uplink and downlink, marked declared; none declared: missing
    off = [
        _pw_info(
            0, uplink_mbps=500.0, downlink_mbps=800.0, declared=["downlink_mbps", "uplink_mbps"]
        ),
        _pw_info(1, gpu_name="sim", declared=["gpu_name"]),
    ]
    a, b = coord_mod.plan_fleet_nodes(off, None)
    assert (a["nic_up_gbps"], a["nic_down_gbps"]) == (0.5, 0.8)
    assert a["sources"]["nic_up_gbps"] == a["sources"]["nic_down_gbps"] == "declared"
    assert b["nic_up_gbps"] is None and b["sources"]["nic_up_gbps"] == "missing"
    assert b["sources"]["gpu_name"] == "declared" and b["declared"] == ["gpu_name"]
    # through the plan step: k counted, and the slots are the planner's placement
    spec, rat, groups = _pw_plan(tmp_path, infos)
    assert rat["inputs"]["k_passes"] == {
        "value": admission.K_HOST_DRAM_COUNTED,
        "source": "counted",
    }
    assert [[n.node_id for n in g] for g in groups] == [
        [infos[p["node"]].node_id for p in row] for row in rat["plan_placement"]
    ]
    assert [len(g) for g in groups] == rat["stage_widths"] == list(spec.stage_widths)


def test_plan_wide_nic_follows_a_rerouted_hop_direction(tmp_path, monkeypatch):
    """BA-48: a hop measured in one direction and then crossed the other way
    by a re-routed chain keeps its per-direction figures with its nodes: each
    node's NIC up is what it sent, whichever end dialed the measurement. A
    worker's non-finite or non-positive figure is not a measurement."""
    # covers: CONTROL-I4
    # WOULD-FAIL-IF: _route_chain hands a re-routed hop's measurement in the dialer's
    # orientation, so nic_from_hops swaps up and down on an asymmetric link, or a NaN/negative
    # reported rate reaches the planner marked 'measured'
    infos = [_pw_info(i, probe_port=9000 + i) for i in range(3)]
    coord, _run = _pw_coordinator(tmp_path, infos)
    ws = [coord.workers[n.node_id] for n in infos]
    # Link a - b sends 1000 + 10a Mbps from a, 100 + 10b back (asymmetric, per node).
    up = {0: 1000.0, 1: 1010.0, 2: 1020.0}
    down = {0: 100.0, 1: 110.0, 2: 120.0}
    by_port = {n.probe_port: i for i, n in enumerate(infos)}

    def fake_request(w, msg, *a, **k):  # FAKE: the MEASURE a worker answers
        src, dst = ws.index(w), by_port[msg["port"]]
        f, r = min(up[src], down[dst]), min(up[dst], down[src])
        link = dict(
            mbps=min(f, r) if (src, dst) != (0, 1) else 1.0,  # 0-1 measures far below its probe
            forward_mbps=f,
            reverse_mbps=r,
            bidir_forward_mbps=f,
            bidir_reverse_mbps=r,
            rtt_ms=0.1,
            duplex="per_direction",
        )
        return {"ok": True, "link": link}

    coord._request = fake_request
    coord._probe_matrix = lambda ws: ([[0.1] * 3 for _ in range(3)], [[1e3] * 3 for _ in range(3)])
    orders = iter([[0, 1, 2], [2, 1, 0]])  # the second crosses both measured pairs reversed
    monkeypatch.setattr(coord_mod, "order_stages", lambda rtt, bw: next(orders))
    order, hops, _worst = coord._route_chain(ws, "run-pw")
    assert order == [2, 1, 0]
    nic = coord_mod.nic_from_hops([infos[i] for i in order], hops)
    for i in range(3):
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < 3]
        want = (max(min(up[i], down[j]) for j in nbrs), max(min(up[j], down[i]) for j in nbrs))
        assert nic[f"n{i}"] == want, (i, nic)
    bad = [{"bidir_forward_mbps": float("nan"), "bidir_reverse_mbps": 5.0}]
    assert coord_mod.nic_from_hops(infos[:2], bad) == {}
    bad = [{"bidir_forward_mbps": -1.0, "bidir_reverse_mbps": True}]
    assert coord_mod.nic_from_hops(infos[:2], bad) == {}


def test_plan_wide_unarmed_refuse_every_pieces_and_native_candidate(tmp_path, monkeypatch):
    """With PIECES_ARMED False the coordinator's plan step passes
    allow_pieces False: on a fleet I16 admits for pieces, every pieces and
    native candidate is refused 'not armed' in the rationale and the chosen
    plan is the best fp32 zero1 or width-1 one. Kept after the arm, with the
    flag patched off, so the unarmed path stays proven."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the coordinator passes allow_pieces True (or a constant) whatever the
    # flag says, so an unarmed coordinator plans pieces or native, or the refusal is silent
    infos = [_pw_info(i) for i in range(4)]
    monkeypatch.setattr(coord_mod, "PIECES_ARMED", True)
    armed_spec, armed, _g = _pw_plan(tmp_path / "armed", infos)
    assert armed_spec.shard == "pieces", armed["refused"]  # I16 admits this fleet for pieces
    monkeypatch.setattr(coord_mod, "PIECES_ARMED", False)
    spec, rat, _groups = _pw_plan(tmp_path / "unarmed", infos)
    cands = rat["plan_candidates"]
    assert any(_pw_unarmed_kind(c) for c in cands)
    for c in cands:
        if _pw_unarmed_kind(c):
            assert not c["admitted"], c["name"]
            assert any("not armed" in t for t in rat["refused"][c["name"]]), c["name"]
    admitted = [c for c in cands if c["admitted"]]
    assert admitted
    assert all(c["w"] == 1 or (c["shard"], c["wire"]) == ("zero1", "fp32") for c in admitted)
    best = min(admitted, key=lambda c: c["step_s"])
    assert rat["chosen"] == best["name"] and rat["chosen_step_s"] == best["step_s"]
    assert (spec.shard, spec.wire) == ("zero1", "fp32")


# The reasons SCHEDULER-I16 refuses a pieces candidate with (admission.Reason kinds).
I16_KINDS = {
    "S1 NIC",
    "S1 PCIe",
    "S1 host DRAM",
    "S1 uplink",
    "S2",
    "MEM",
    "MEM-HOST",
    "class",
    "declared",
    "missing",
    "invalid",
}


def test_plan_wide_armed_admits_or_refuses_pieces_by_i16(tmp_path, monkeypatch):
    """PIECES_ARMED True: on the fleet I16 admits, the pieces candidate is
    admitted and planned; on a fleet whose links are too slow for S2, every
    pieces candidate is refused with an I16 reason, never 'not armed'."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the armed coordinator still passes allow_pieces False (pieces refused
    # 'not armed'), or plans a pieces candidate I16 refused
    monkeypatch.setattr(coord_mod, "PIECES_ARMED", True)
    infos = [_pw_info(i) for i in range(4)]
    for sub, nic, admits in (("fast", PW_NIC_MBPS, True), ("slow", 300.0, False)):
        spec, rat, _g = _pw_plan(tmp_path / sub, infos, nic={n.node_id: (nic, nic) for n in infos})
        pieces = [c for c in rat["plan_candidates"] if c["shard"] == "pieces"]
        assert pieces
        for c in pieces:
            kinds = {r["kind"] for r in c["reasons"]}
            assert "not armed" not in kinds, c
            assert c["admitted"] or (kinds and kinds <= I16_KINDS), c
        assert any(c["admitted"] for c in pieces) is admits, rat["refused"]
        assert (spec.shard == "pieces") is admits


def test_plan_wide_rationale_lists_inputs_steps_and_refusals(tmp_path, monkeypatch):
    """The plan's rationale has every fleet key of every node with its
    source, the chosen candidate's simulated step, and every refused
    candidate with its reasons; FIT is asked of every slot of the placement."""
    # covers: CONTROL-I4
    # WOULD-FAIL-IF: an input reaches the planner without its source in the rationale, the
    # chosen step or a refused width's reason is dropped, or FIT checks one node per stage
    # instead of every slot of a wide stage
    monkeypatch.setattr(coord_mod, "PIECES_ARMED", False)
    infos = [_pw_info(i) for i in range(4)]
    coord, _run = _pw_coordinator(tmp_path, infos, fit_check=True)
    fits = []

    def fake_request(w, msg, body=b"", timeout=None, file=None, reply_kind=None):
        if msg["kind"] == "FIT":
            fits.append((msg["rank"], msg["peer"], w.node_id))
            return {"fit": {"ok": True, "peakReservedBytes": 1 << 20}}
        return {"profile": {"ok": False, "error": "no profile (fake)"}}

    coord._request = fake_request
    spec, rat, groups = _pw_plan(tmp_path, infos, coord=coord)
    assert rat["stage_widths"] == [2, 2]  # this fleet plans a wide stage
    for node in rat["inputs"]["nodes"]:
        assert set(coord_mod.PLAN_INPUT_KEYS) <= set(node), node
        for k in coord_mod.PLAN_INPUT_KEYS:
            assert node[k]["source"] in ("measured", "declared", "absent", "missing"), (k, node)
    assert rat["inputs"]["link"]["source"] == "default"
    chosen = next(c for c in rat["plan_candidates"] if c["name"] == rat["chosen"])
    assert isinstance(rat["chosen_step_s"], float) and rat["chosen_step_s"] == chosen["step_s"]
    refused = [c for c in rat["plan_candidates"] if not c["admitted"]]
    assert refused
    for c in refused:
        assert c["reasons"] and rat["refused"][c["name"]] == [r["text"] for r in c["reasons"]]
    assert sorted(fits) == sorted(
        (r, p, n.node_id) for r, g in enumerate(groups) for p, n in enumerate(g)
    )


def _pw_profiles(groups, fast=None, factor=1.8):
    """FAKE profiles at the worker-network boundary (each slot's measured
    F/B/W seconds); the slot `fast` is `factor` times faster."""
    out = []
    for r, g in enumerate(groups):
        row = []
        for p in range(len(g)):
            s = 1.0 / factor if (r, p) == fast else 1.0
            row.append({"ok": True, "t_f": 0.01 * s, "t_b": 0.02 * s, "t_w": 0.01 * s})
        out.append(row)
    return out


def test_plan_wide_lane_budgets_each_lane_gets_its_own_peer(tmp_path, monkeypatch):
    """BA-29: stage 1 at width 4 on four nodes of distinct memories; every
    lane solve of stage 1 gets its own peer's activation budget (its memory
    less vram_reserve and the stage state), never the stage's sum."""
    # covers: CONTROL-I4
    # WOULD-FAIL-IF: _zero_bubble hands the stage's summed budget (or one figure for every
    # peer) to the lane solves, so a small peer's lane is solved against its siblings' memory
    import wantrain.scheduler.zero_bubble as zb
    from wantrain.model.gpt import make_config
    from wantrain.scheduler.planner import stage_param_split, vram_reserve

    mems = [3 << 30, 4 << 30, 5 << 30, 6 << 30]
    infos = [_pw_info(0)] + [
        _pw_info(i + 1, vram_total_bytes=m, vram_free_bytes=m) for i, m in enumerate(mems)
    ]
    groups = [[infos[0]], infos[1:]]
    spec = JobSpec(model=make_config("toy"), stages=2, stage_widths=[1, 4], microbatches=8)
    spec.finalize()
    a, b = spec.partition[1]
    trainable, frozen = stage_param_split(spec.model, a, b)
    want = sorted(m - vram_reserve(m) - (trainable * 16 + frozen * 4) for m in mems)
    assert len(set(want)) == 4 and min(want) > 0

    def lane_budgets():
        seen = []
        real = zb.zero_bubble

        def spy(cfg, mode="1p"):
            if cfg.n_micro < spec.microbatches and cfg.max_mem:  # a lane of a wide stage
                seen.append(cfg.max_mem[1])
            return real(cfg, mode)

        monkeypatch.setattr(zb, "zero_bubble", spy)
        coord, _run = _pw_coordinator(tmp_path / str(len(calls)), infos)
        coord._profile = lambda s, g: _pw_profiles(g)
        info = coord._zero_bubble(JobSpec.from_json(spec.to_json()), groups, [])
        monkeypatch.setattr(zb, "zero_bubble", real)
        calls.append(info)
        return sorted(seen)

    calls: list = []
    assert lane_budgets() == want
    assert calls[0]["source"] == "profiled"
    # control: the stage's sum handed to every lane does not pass the same check
    summed = coord_mod._mem_budgets
    monkeypatch.setattr(
        coord_mod,
        "_mem_budgets",
        lambda s, g: [sum(x) if isinstance(x, list) else x for x in summed(s, g)],
    )
    assert lane_budgets() != want


def test_plan_wide_recheck_refuses_a_reweighting_past_a_bound(tmp_path, monkeypatch):
    """BA-47: after the profiled re-weighting, a pieces stage is re-checked
    with its final mb_owner and orders before FIT and ASSIGN. Eight cpu nodes
    plan S 2, w 4 pieces at 32 microbatches; node 0 (stage 0, peer 0) sits
    alone on host h0, whose RAM is sized for an equal share (8 of 32). Profiled
    1.8x faster, it owns 12 of 32 and the plan step refuses naming its host;
    with equal profiles it passes."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the coordinator admits a pieces plan on the planner's equal ownership
    # only, never re-checking device memory, MEM-HOST and S2 after schedule_for_spec moved
    # microbatches to a faster peer
    monkeypatch.setattr(coord_mod, "PIECES_ARMED", True)
    job = dict(PW_JOB, microbatches=32, shard="pieces")

    def infos_at(ram):  # node 0 alone on h0 with `ram`; the others share a large host
        # The large host's DRAM is sized like its RAM: seven peers' exchange at the counted
        # k (15 since SC-5, sealing included) needs ~39 GB/s, above the 34 GB/s default of a
        # one-worker Mac. This test is about the MEM-HOST re-check (BA-47), not S1.
        return [_pw_info(0, hostname="h0", host_ram_bytes=ram)] + [
            _pw_info(i, host_ram_bytes=64 << 30, host_dram_gbps=96.0) for i in range(1, 8)
        ]

    def plan(ram, fast):
        infos = infos_at(ram)
        coord, _run = _pw_coordinator(tmp_path / f"{ram}-{fast}", infos)
        coord._profile = lambda s, g: _pw_profiles(g, fast=fast)
        return _pw_plan(tmp_path / f"{ram}-{fast}", infos, job, coord=coord)

    spec, rat, groups = plan(64 << 30, None)
    assert rat["chosen"] == "S 2 w 4 pieces + native"
    assert [row.count(0) for row in spec.mb_owner] == [8, 8]
    assert groups[0][0].node_id == "n0"
    nic = {f"n{i}": (PW_NIC_MBPS, PW_NIC_MBPS) for i in range(8)}

    def admits(ram):
        infos = infos_at(ram)
        g = [[infos[int(n.node_id[1:])] for n in row] for row in groups]
        nodes = coord_mod._nodes_by_id(infos, nic)
        return coord_mod.pieces_admission(spec, g, nodes, True) == ([], [])

    lo, hi = 4 << 30, 64 << 30  # h0's smallest RAM that admits the equal share
    assert not admits(lo) and admits(hi)
    while hi - lo > 1 << 12:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if admits(mid) else (mid, hi)
    spec_eq, _rat, _g = plan(hi, None)  # control: 8 of 32 each, passes
    assert [row.count(0) for row in spec_eq.mb_owner] == [8, 8]
    with pytest.raises(RuntimeError, match="SCHEDULER-I16") as e:
        plan(hi, (0, 0))
    assert "MEM-HOST: host h0" in str(e.value), str(e.value)


def _pw_pieces_spec(widths=(2, 1), **kw):
    from wantrain.model.gpt import make_config

    spec = JobSpec(
        model=make_config("toy"),
        stages=len(widths),
        stage_widths=list(widths),
        microbatches=8,
        shard="pieces",
        wire="native",
        stream_grads=True,
        **kw,
    )
    return spec.finalize()


def test_pieces_spec_mode_checks_refuse_class_declared_and_memory(tmp_path):
    """BA-50: a spec-mode pieces job gets I16's class check (probed, never
    declared) and memory terms before ASSIGN, refusing naming the class, the
    peer or the host; S1 and S2 (here unevaluable: no NIC was measured) are
    warnings. One case per refusal and an admitted control."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: a spec-mode pieces job reaches ASSIGN on two device models, on a
    # declared class, on a device or a host whose memory I16 refuses, or S1/S2 refuse a
    # spec the operator placed
    spec = _pw_pieces_spec()

    def check(infos):
        coord, run = _pw_coordinator(tmp_path / f"c{len(os.listdir(tmp_path))}", infos)
        groups = [infos[:2], infos[2:]]
        run.pieces_nodes = coord_mod._nodes_by_id(infos, None)
        return coord._check_pieces(JobSpec.from_json(spec.to_json()).finalize(), groups, run)

    ok = [_pw_info(i, hostname=f"h{i}") for i in range(3)]
    warnings = check(ok)
    assert warnings and all(w.startswith("missing: ") for w in warnings), warnings
    cases = [
        ([ok[0], _pw_info(1, hostname="h1", gpu_name="cpu-b"), ok[2]], "class: stage 0"),
        (
            [ok[0], _pw_info(1, hostname="h1", declared=["gpu_name"]), ok[2]],
            "declared: stage 0: 1:w1/cpu gpu_name",
        ),
        (
            [
                ok[0],
                _pw_info(1, hostname="h1", vram_total_bytes=1 << 30, vram_free_bytes=1 << 30),
                ok[2],
            ],
            "MEM: stage 0 peer 1:w1/cpu",
        ),
        (
            [ok[0], _pw_info(1, hostname="h1", host_ram_bytes=4 << 30), ok[2]],
            "MEM-HOST: host h1",
        ),
    ]
    for infos, want in cases:
        with pytest.raises(RuntimeError, match="SCHEDULER-I16") as e:
            check(infos)
        assert want in str(e.value), (want, str(e.value))


def test_pieces_spec_mode_s2_reads_each_wide_stage_under_its_own_emission(tmp_path, monkeypatch):
    """A spec may give its wide stages different emission orders; the exchange
    simulation takes one order for every stage, so each stage's S2 must come
    from the run under that stage's own order, never the first wide stage's.
    The real admission.recheck runs; a wrapper (a test seam, not a fake of the
    check) tags one S2 per stage with the emission the call simulated."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: pieces_admission simulates every wide stage under the first wide stage's
    # emission order, or keeps an S2 of a stage from a run under another stage's order
    spec = _pw_pieces_spec(widths=(2, 2), emission_order=["forward", "store"])
    infos = [_pw_info(i, hostname=f"h{i}") for i in range(4)]
    coord, run = _pw_coordinator(tmp_path, infos)
    run.pieces_nodes = coord_mod._nodes_by_id(
        infos, {n.node_id: (PW_NIC_MBPS, PW_NIC_MBPS) for n in infos}
    )
    real = admission.recheck
    calls = []

    def tagged(stages, *a, emission, **kw):
        calls.append(emission)
        out = real(stages, *a, emission=emission, **kw)
        return out + [
            admission.Reason("S2", f"stage {st.index} via {emission}", stage=st.index)
            for st in stages
        ]

    monkeypatch.setattr(admission, "recheck", tagged)
    warnings = coord._check_pieces(spec, [infos[:2], infos[2:]], run)
    assert sorted(calls) == ["backward", "forward"], calls
    tags = sorted(w for w in warnings if " via " in w)
    assert tags == ["S2: stage 0 via forward", "S2: stage 1 via backward"], warnings


def test_plan_wide_missing_dram_refuses_naming_the_measurement(tmp_path, monkeypatch):
    """A pieces plan job whose fleet lacks one host's DRAM measurement (one
    worker's hostDramGBps withheld, its machine's only figure) is refused
    naming that missing measurement; the full fleet plans pieces."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: a missing host-DRAM measurement is filled with a default or another
    # host's figure, so a pieces stage is admitted on an input nobody measured
    from wantrain.scheduler.fleetplan import PlanRefused

    monkeypatch.setattr(coord_mod, "PIECES_ARMED", True)
    job = dict(PW_JOB, shard="pieces")
    infos = [_pw_info(i, hostname=f"h{i}") for i in range(4)]
    spec, _rat, _g = _pw_plan(tmp_path / "full", infos, job)
    assert spec.shard == "pieces"
    infos[3] = _pw_info(3, hostname="h3", host_dram_gbps=None)
    with pytest.raises(PlanRefused) as e:
        _pw_plan(tmp_path / "withheld", infos, job)
    assert "missing: host h3 dram" in str(e.value), str(e.value)


class _PrChannel:
    """FAKE of the control channel at the worker-network boundary, named as
    such: sends are recorded, nothing is transmitted."""

    last_rx = 0.0

    def __init__(self):
        self.sent = []

    def send_file(self, msg, path):
        self.sent.append(dict(msg))

    def send(self, msg, body=b""):
        self.sent.append(dict(msg))

    def close(self):
        pass


_PR_WIDTHS = [1, 3, 1]


def _pr_infos(ram=64 << 30):
    """One host per slot; the three peers of the pieces stage on h1..h3."""
    return [_pw_info(i, hostname=f"h{i}", host_ram_bytes=ram) for i in range(5)]


def _pr_run(tmp_path, infos):
    """A real Coordinator and a toy 1-3-1 pieces run launched as attempt 1.
    FAKE at the worker-network boundary, named as such: `_request` records
    (kind, message) and answers {"ok": True}; `_sync_all_connections` does
    nothing. Every coordinator method the test calls is the real one."""
    import hashlib

    coord = Coordinator(
        config=CoordinatorConfig(
            state_dir=str(tmp_path / "coord"), poll_s=0.01, replan_restart_cost_s=0.1
        )
    )
    sent = []
    coord._request = lambda w, msg, body=b"", **k: (
        sent.append((msg["kind"], dict(msg))) or {"ok": True}
    )
    coord._sync_all_connections = lambda run: None
    os.makedirs(tmp_path / "run")
    run = _Run("run-pr", str(tmp_path / "run"))
    spec = _pw_pieces_spec(_PR_WIDTHS, steps=24)
    run.spec, run.spec_json = spec, spec.to_json()
    slots = [(r, p) for r, w in enumerate(_PR_WIDTHS) for p in range(w)]
    for i, (slot, info) in enumerate(zip(slots, infos)):
        w = Worker(info, _PrChannel(), f"session{i}", None, peer_id=f"{i:032x}")
        w.run_id = run.run_id
        coord.workers[info.node_id] = w
        run.slots[slot] = info.node_id
    run.pieces_nodes = coord_mod._nodes_by_id(infos, None)
    coord.run = run
    coord._launch(run)
    for slot in run.slots:  # common wave 3, through the store's real staging path
        data = f"wave-3-r{slot[0]}p{slot[1]}|".encode() * 64
        body = os.path.join(run.run_dir, f"in-r{slot[0]}p{slot[1]}.part")
        with open(body, "wb") as f:
            f.write(data)
        sha = hashlib.sha256(data).hexdigest()
        run.store.stage_part(slot, run.attempt, 3, "state.pt", sha, body)
        manifest = {
            "v": 1,
            "step": 3,
            "files": [{"name": "state.pt", "sha256": sha, "bytes": len(data)}],
        }
        assert run.store.commit_staged(slot, run.attempt, 3, manifest) is True
    spec = coord._owned_spec(run)
    for step in range(4):  # slot (1, 0) 4x slower, through the real _dispatch
        for slot in run.slots:
            r, p = slot
            n = len(spec.owned(r, p))
            hops = {}
            for head, nb in (("next", r + 1), ("prev", r - 1)):
                if 0 <= nb < len(_PR_WIDTHS):
                    for q in range(_PR_WIDTHS[nb]):
                        hops[f"{head}{q}"] = {
                            "frames": 2 * n,
                            "bytes": 4096 * n,
                            "busy_ms": 1.0 * n,
                        }
            slow = 4.0 if slot == (1, 0) else 1.0
            summary = {
                "v": 1,
                "step": step,
                "rank": r,
                "peer": p,
                "step_ms": 100.0,
                "emit_us": 10.0,
                "ops": {
                    "F": {"n": n, "med_ms": 3.0 * slow},
                    "B": {"n": n, "med_ms": 5.0 * slow},
                    "W": {"n": n, "med_ms": 2.0 * slow},
                    "OPT": {"n": 1, "med_ms": 1.0},
                },
                "stall_ms": 0.0,
                "hops": hops,
            }
            msg = {
                "kind": "TELEM_STEP",
                "run_id": run.run_id,
                "attempt": run.attempt,
                "summary": summary,
            }
            coord._dispatch(coord.workers[run.slots[slot]], msg, b"")
    return coord, run, sent


def test_pieces_rebalance_refused_when_the_new_ownership_breaks_mem_host(tmp_path):
    """BA-44: M2's rebalance moves microbatches off a 4x slow peer of a
    width-3 pieces stage. On hosts sized for the current ownership, the new
    M_k of the other two peers breaks MEM-HOST: the rebalance is not applied,
    the refusal (naming the host) is logged and recorded with the run's
    replans, and nothing is relaunched. The same window on larger hosts is
    applied (control)."""
    # covers: SCHEDULER-I16
    # WOULD-FAIL-IF: the coordinator applies a rebalance of a pieces run without re-checking
    # I16 under the new ownership, relaunches after a refusal, or refuses silently
    from wantrain.scheduler.zero_bubble import replan

    coord, run, sent = _pr_run(tmp_path / "control", _pr_infos())
    windows = {slot: list(run.telem_window[slot]) for slot in run.slots}
    cur = coord._owned_spec(run)
    plan = replan(cur, windows)
    assert plan["decision"] and plan["owner_changed"], plan
    new = json.loads(cur.to_json())
    new.update({k: plan[k] for k in ("peer_weights", "mb_owner", "schedule_order")})
    new = JobSpec.from_json(json.dumps(new)).finalize()
    groups = [[run.slots[(r, p)] for p in range(w)] for r, w in enumerate(_PR_WIDTHS)]

    def admits(spec, ram):
        infos = _pr_infos(ram)
        by_id = {n.node_id: n for n in infos}
        g = [[by_id[n] for n in row] for row in groups]
        return (
            coord_mod.pieces_admission(spec, g, coord_mod._nodes_by_id(infos, None), False)[0] == []
        )

    lo, hi = 4 << 30, 64 << 30  # the smallest host RAM the current ownership needs
    assert not admits(cur, lo) and admits(cur, hi)
    while hi - lo > 1 << 12:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if admits(cur, mid) else (mid, hi)
    assert not admits(new, hi)

    assert coord._maybe_rebalance(run) is True  # control: 64 GiB hosts apply it
    assert run.attempt == 2 and run.replans[0]["owner_changed"] is True

    coord, run, sent = _pr_run(tmp_path / "refused", _pr_infos(hi))
    assert coord._maybe_rebalance(run) is False
    assert run.attempt == 1 and run.ownership is None
    assert [k for k, _m in sent].count("ASSIGN") == 5  # attempt 1 only, no restore
    assert not any(k in ("RESTORE", "ABORT") for k, _m in sent)
    (rec,) = run.replans
    assert rec["applied"] is False and rec["step"] == 3
    assert any(t.startswith("MEM-HOST: host h") for t in rec["refused"]), rec
    assert any("not applied" in e[2] for e in coord.events)


@pytest.mark.slow
def test_pieces_armed_spec_runs_through_the_coordinator_and_the_agent(tmp_path, port_base):
    """Armed (F1 and F8 passed on the integrated tree): a shard 'pieces' wire
    'native' spec goes through the coordinator's admission and every slot is
    launched by its agent with '--launcher coordinator' (BA-58); the run ends
    equal to the oracle."""
    # covers: CONTROL-I9
    # WOULD-FAIL-IF: the coordinator still refuses pieces after the arm, or the agent launches
    # stage_main without '--launcher coordinator' (stage_main then refuses the pieces spec and
    # the run fails), or the pieces run moves one bit of the losses or the state
    from helpers.exact import fleet_manifest, reference_manifest_for

    from wantrain.model.gpt import make_config
    from wantrain.validation.state import diff_manifests

    assert coord_mod.PIECES_ARMED is True
    c = LocalCluster(tmp_path, port_base, probe=False)
    try:
        for i in range(3):
            c.add_worker(f"a{i}")
        c.wait_registered(3)
        recipe = JobSpec(
            model=make_config("toy"),
            steps=4,
            microbatches=8,
            stages=2,
            stage_widths=[2, 1],
            shard="pieces",
            wire="native",
            stream_grads=True,
        )
        res = c.run(Job(spec=recipe))
        assert res.ok, res.reason
        for name, workdir in c.workdirs.items():
            for log in glob.glob(os.path.join(workdir, "runs", res.run_id, "stage-*.log")):
                with open(log) as f:
                    assert "WT_REFUSED" not in f.read(), (name, log)
        state = fleet_manifest(c, res, tmp_path / "collected")
        reference = reference_manifest_for(recipe, tmp_path / "ref")
        assert diff_manifests(state, reference) == []
        assert state == reference
    finally:
        c.close()


# ---------------------------------------------------------------- M4: tripwire m4-unarmed (I-16)

M4_HF_REFUSAL = "arch 'hf' is not armed: every leg of the m4 tripwire must land first"
M4_MOE_REFUSAL = "moe_experts > 0 is not armed: every leg of the m4 tripwire must land first"


def _m4_spec(**model_fields):
    spec = JobSpec(steps=2, stages=2)
    for k, v in model_fields.items():
        # moe_experts may not be a GPTConfig field yet (model-core lands it); the
        # refusal reads it with getattr (I-16).
        setattr(spec.model, k, v)
    return spec


@pytest.mark.parametrize("pieces_armed", [True, False], ids=["pieces_armed", "pieces_unarmed"])
@pytest.mark.parametrize(
    "job, want",
    [
        (Job(spec=_m4_spec(arch="hf")), M4_HF_REFUSAL),
        (Job(spec=_m4_spec(moe_experts=4)), M4_MOE_REFUSAL),
        (Job(plan={"model": "toy", "arch": "hf"}, stages=2), M4_HF_REFUSAL),
        (Job(plan={"model": "toy", "moe_experts": 2}, stages=2), M4_MOE_REFUSAL),
    ],
    ids=["spec_hf", "spec_moe", "plan_hf", "plan_moe"],
)
def test_m4_unarmed_run_job_refuses_hf_and_moe_before_any_request(
    tmp_path, monkeypatch, job, want, pieces_armed
):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: run_job reaches setup or sends any agent request for an arch 'hf' or
    # MoE job while M4_ARMED is False; or the check sits behind PIECES_ARMED (the old early
    # return skips it when pieces is armed); or one of spec and plan mode is not read
    monkeypatch.setattr(coord_mod, "M4_ARMED", False)
    monkeypatch.setattr(coord_mod, "PIECES_ARMED", pieces_armed)
    coord, sent = _unarmed_coordinator(tmp_path)

    def no_setup(*a, **k):
        raise AssertionError("setup reached")

    coord._setup = no_setup
    res = coord.run_job(job, wait_workers_s=0.0)
    assert not res.ok and res.reason == want
    assert coord_mod.unarmed_reason(job) == want
    assert sent == []


def test_m4_unarmed_reason_is_a_pure_decision_over_a_model_dict(monkeypatch):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: m4_unarmed_reason (the decision cli.py's local and stage commands call)
    # misses arch 'hf' or moe_experts > 0, refuses a dense model, ignores M4_ARMED, or takes a
    # malformed moe_experts as dense (the tripwire must fail closed)
    monkeypatch.setattr(coord_mod, "M4_ARMED", False)
    f = coord_mod.m4_unarmed_reason
    assert f({"arch": "hf", "moe_experts": 0}) == M4_HF_REFUSAL
    assert f({"arch": "builtin", "moe_experts": 8}) == M4_MOE_REFUSAL
    assert f({"arch": "builtin", "moe_experts": "8"}) == M4_MOE_REFUSAL
    assert f({"arch": "builtin", "moe_experts": 0}) is None
    assert f({"arch": "user"}) is None
    assert f({}) is None
    monkeypatch.setattr(coord_mod, "M4_ARMED", True)
    assert f({"arch": "hf", "moe_experts": 8}) is None


def test_m4_unarmed_dense_spec_and_plan_jobs_pass_the_check(tmp_path, monkeypatch):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: the m4 refusal also catches a builtin spec, a plan without arch or
    # moe keys, or a plan whose model is not a preset name, so every existing job is refused
    monkeypatch.setattr(coord_mod, "M4_ARMED", False)
    jobs = [
        Job(spec=JobSpec(steps=2, stages=2)),
        Job(spec=_m4_spec(arch="builtin", moe_experts=0)),
        Job(plan={"model": "toy"}, stages=2),
        Job(plan={"model": {}, "moe_experts": None}, stages=2),
    ]
    for i, job in enumerate(jobs):
        assert coord_mod.unarmed_reason(job) is None
        coord, sent = _unarmed_coordinator(tmp_path / str(i))
        res = coord.run_job(job, wait_workers_s=0.0)
        # It went on to wait for workers, past the check, and found none.
        assert not res.ok and "workers registered" in res.reason, res.reason


# ---------------------------------------------------------------- M4: armed (arm-m4)

M4_JOBS = [
    (Job(spec=_m4_spec(arch="hf")), M4_HF_REFUSAL),
    (Job(spec=_m4_spec(moe_experts=4)), M4_MOE_REFUSAL),
    (Job(plan={"model": "toy", "arch": "hf"}, stages=2), M4_HF_REFUSAL),
    (Job(plan={"model": "toy", "moe_experts": 2}, stages=2), M4_MOE_REFUSAL),
]
M4_JOB_IDS = ["spec_hf", "spec_moe", "plan_hf", "plan_moe"]


@pytest.mark.parametrize("pieces_armed", [True, False], ids=["pieces_armed", "pieces_unarmed"])
@pytest.mark.parametrize("job, refusal", M4_JOBS, ids=M4_JOB_IDS)
def test_m4_armed_hf_and_moe_jobs_pass_unarmed_reason(
    tmp_path, monkeypatch, job, refusal, pieces_armed
):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: M4_ARMED stays False on the tree (an hf or MoE job of either mode is
    # still refused), or PIECES_ARMED decides the m4 answer (an armed hf or MoE job refused
    # with pieces unarmed), or run_job still stops an armed job before setup
    assert coord_mod.M4_ARMED is True  # the module's own value, never patched here
    monkeypatch.setattr(coord_mod, "PIECES_ARMED", pieces_armed)
    assert coord_mod.unarmed_reason(job) is None
    coord, sent = _unarmed_coordinator(tmp_path / "armed")
    res = coord.run_job(job, wait_workers_s=0.0)
    # Past the check: it went on to wait for workers and found none.
    assert not res.ok and "workers registered" in res.reason, res.reason
    assert sent == []
    # The deliberately wrong tree (the flag off) gives the job's own refusal, which
    # PIECES_ARMED does not change either.
    monkeypatch.setattr(coord_mod, "M4_ARMED", False)
    assert coord_mod.unarmed_reason(job) == refusal


M4_CLI_MOE_MODEL = dict(
    vocab_size=32,
    n_layer=2,
    n_head=2,
    d_model=16,
    seq_len=6,
    norm="rmsnorm",
    mlp="swiglu",
    bias=False,
    pos="rope",
    moe_experts=4,
    moe_top_k=2,
    moe_d_ff=8,
)


def _m4_cli_specs(tmp_path) -> dict:
    """A tiny native MoE spec JSON and an hf spec JSON, as an operator writes them."""
    from wantrain.model.gpt import GPTConfig

    moe = JobSpec(model=GPTConfig(**M4_CLI_MOE_MODEL), steps=2, stages=2, out_dir=str(tmp_path))
    hf = json.loads(JobSpec(steps=2, stages=2, out_dir=str(tmp_path)).to_json())
    hf["model"].update(arch="hf", hf_config={"model_type": "llama"})
    paths = {"moe": str(tmp_path / "moe.json"), "hf": str(tmp_path / "hf.json")}
    with open(paths["moe"], "w") as f:
        f.write(moe.to_json())
    with open(paths["hf"], "w") as f:
        json.dump(hf, f)
    return paths


_M4_CLI_CHILD = r"""
import json, subprocess, sys
import wantrain.coordinator.coordinator as coordinator
coordinator.M4_ARMED = False
started = []
real_popen_init = subprocess.Popen.__init__
def no_child(self, *a, **k):
    started.append(str(a[0] if a else k.get("args")))
    raise AssertionError("a child started")
subprocess.Popen.__init__ = no_child
from wantrain.cli import main
try:
    rc = main(sys.argv[1:])
except SystemExit as e:
    rc = e.code
print(json.dumps({"rc": rc, "started": started,
                  "loaded": sorted(m for m in ("wantrain.launch.local", "wantrain.worker.stage_main",
                                               "wantrain.runtime.stage") if m in sys.modules)}))
"""


@pytest.mark.parametrize(
    "argv, kind",
    [
        (["local", "--spec", "{moe}"], "moe"),
        (["stage", "--spec", "{moe}", "--rank", "0"], "moe"),
        (["local", "--spec", "{hf}"], "hf"),
    ],
    ids=["local_moe", "stage_moe", "local_hf"],
)
def test_m4_unarmed_cli_operator_entries_refuse_before_any_child(tmp_path, argv, kind):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: an operator entry (`wantrain local --spec`, `wantrain stage --spec`) trains
    # an MoE or hf spec on a tree with only some legs in: with M4_ARMED False it does not exit 2
    # naming the tripwire, or it starts a child or imports the launcher or the stage first
    import subprocess
    import sys

    paths = _m4_cli_specs(tmp_path)
    args = [a.format(**paths) for a in argv]
    p = subprocess.run(
        [sys.executable, "-c", _M4_CLI_CHILD, *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    got = json.loads(p.stdout.strip().splitlines()[-1])
    want = M4_MOE_REFUSAL if kind == "moe" else M4_HF_REFUSAL
    assert got == {"rc": 2, "started": [], "loaded": []}, (got, p.stderr[-2000:])
    assert f"wantrain {argv[0]}: refusing: {want}" in p.stderr, p.stderr[-2000:]


def test_m4_armed_cli_m4_unarmed_reason_returns_none_for_both(tmp_path, monkeypatch):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: the tree is not armed, so m4_unarmed_reason (what `wantrain local` and
    # `wantrain stage` call before any child) still refuses the MoE or the hf spec JSON
    from wantrain import cli

    assert coord_mod.M4_ARMED is True
    paths = _m4_cli_specs(tmp_path)
    models = {}
    for kind, path in paths.items():
        with open(path) as f:
            models[kind] = cli._spec_model(json.load(f))
    assert models["moe"]["moe_experts"] == 4 and models["hf"]["arch"] == "hf"
    for model in models.values():
        assert coord_mod.m4_unarmed_reason(model) is None
        assert cli._m4_refusal(model) is None
    # Control: the same JSONs are refused, each with its own reason, when unarmed.
    monkeypatch.setattr(coord_mod, "M4_ARMED", False)
    assert coord_mod.m4_unarmed_reason(models["moe"]) == M4_MOE_REFUSAL
    assert coord_mod.m4_unarmed_reason(models["hf"]) == M4_HF_REFUSAL


# ---------------------------------------------------------------- M4: CONTROL-I24 admission (I-9)

M4_RECORD_A = (
    "transformers/__init__.py,sha256=AAAA,100\n"
    "transformers/models/llama/modeling_llama.py,sha256=BBBB,200\n"
    "transformers/utils/import_utils.py,sha256=CCCC,300\n"
)
# One line differs: modeling_llama.py.
M4_RECORD_B = M4_RECORD_A.replace("BBBB", "BBBX")
M4_MAC, M4_LINUX = "macosx_14_0_arm64", "manylinux_2_28_x86_64"
M4_TORCH_RECORDS = {
    (M4_MAC, "2.9.0"): {"torch": "t-mac", "filelock": "fl-1", "sympy": "sy-1"},
    (M4_LINUX, "2.9.0"): {"torch": "t-linux", "filelock": "fl-1", "sympy": "sy-1"},
}
M4_CPU_MAC, M4_MPS = "cpu:arm64:DEFAULT", "mps:Apple M2"
M4_CPU_X86 = "cpu:x86_64:AVX2"


def m4_record_diff(recorded: str, other: str):
    """Crafted record_diff (hfm's own is model-core's test): the path of the first
    differing RECORD line."""
    a, b = recorded.splitlines(), other.splitlines()
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else ""
        y = b[i] if i < len(b) else ""
        if x != y:
            return (x or y).split(",")[0]
    return None


def _m4_gz(text: str) -> str:
    import base64
    import gzip

    return base64.b64encode(gzip.compress(text.encode())).decode()


def _m4_fp(record=M4_RECORD_A, tf="5.19.0", torch="2.9.0", git="g-abc", tag=M4_MAC, **kw):
    fp = {
        "transformers": {
            "version": tf,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
            "record_gz_b64": _m4_gz(record),
        },
        "torch": {
            "version": torch,
            "git_version": git,
            "platform_tag": tag,
            "records": dict(M4_TORCH_RECORDS.get((tag, torch), {"torch": "none"})),
        },
        "probe_vector": {"kernels": [False, None], "flash_attn": [False, None]},
        "native": {"libSystem.B.dylib": "ns-mac"} if tag == M4_MAC else {"libc.so.6": "ns-lin"},
        "python": "3.12.8",
        "prefix_ok": True,
        "code_ok": True,
        "refusals": [],
    }
    for k, v in kw.items():
        if k == "records":
            fp["torch"]["records"] = v
        else:
            fp[k] = v
    return fp


def _m4_rep(device, cls, ok=True, cause=None):
    return {
        "ok": ok,
        "cause": cause,
        "detail": "x" if cause else None,
        "device": device,
        "device_class": cls,
    }


def _m4_fleet():
    """Two Macs (CPU + MPS) and one Linux CPU node, all admissible together."""
    reports = {
        "a": [_m4_rep("cpu", M4_CPU_MAC), _m4_rep("mps", M4_MPS)],
        "b": [_m4_rep("cpu", M4_CPU_MAC), _m4_rep("mps", M4_MPS)],
        "c": [_m4_rep("cpu", M4_CPU_X86)],
    }
    fps = {"a": _m4_fp(), "b": _m4_fp(), "c": _m4_fp(tag=M4_LINUX)}
    return reports, fps, _m4_fp()


def _m4_admit(reports, fps, builder, slot_map=None, pool=None, recorded=None):
    return coord_mod.hf_admission(
        reports,
        fps,
        builder,
        slot_map or {},
        sorted(reports) if pool is None else pool,
        torch_records=M4_TORCH_RECORDS,
        record_diff=m4_record_diff,
        recorded=recorded,
    )


def test_m4_admission_admits_a_mixed_platform_fleet_with_its_builder():
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: nodes of different platforms (Mac CPU+MPS beside Linux CPU, CONTROL-I5)
    # are refused for their own torch RECORD entry or their native objects, which are only
    # compared within one (platform tag, device classes)
    reports, fps, builder = _m4_fleet()
    slot_map = {
        (0, 0): {"node": "a", "device_class": M4_MPS},
        (1, 0): {"node": "b", "device_class": M4_CPU_MAC},
    }
    assert _m4_admit(reports, fps, builder) == []
    assert _m4_admit(reports, fps, builder, slot_map) == []
    # CONTROL-I25 (the human, 2026-10-11): the coordinator refuses this two-class map
    # (MPS + Mac CPU) before decision 8 runs; hf_admission's own rule above is unchanged.
    from wantrain.resources.node import mixed_fleet

    two = mixed_fleet((e["node"], e["device_class"]) for e in slot_map.values())
    assert two is not None and two.startswith("MIXED-FLEET: the device map spans 2 device classes")


@pytest.mark.parametrize(
    "mutate, want",
    [
        (
            lambda r, f, b: r["b"].__setitem__(1, _m4_rep("mps", M4_MPS, False, "GATE-DIFF")),
            "node b: gate on 'mps' failed: GATE-DIFF",
        ),
        (lambda r, f, b: f.__setitem__("b", _m4_fp(tf="5.18.0")), "node b: transformers '5.18.0'"),
        (
            lambda r, f, b: f.__setitem__("b", _m4_fp(record=M4_RECORD_B)),
            "node b: transformers RECORD differs from the record builder's at "
            "transformers/models/llama/modeling_llama.py",
        ),
        (
            lambda r, f, b: f.__setitem__("b", _m4_fp(git="g-other")),
            "node b: torch '2.9.0' ('g-other') differs",
        ),
        (
            lambda r, f, b: f.__setitem__(
                "b", _m4_fp(records={"torch": "t-mac", "filelock": "fl-2", "sympy": "sy-1"})
            ),
            f"node b: torch RECORD differs from TORCH_RECORDS[{M4_MAC!r}] at filelock",
        ),
        (
            lambda r, f, b: f.__setitem__("c", _m4_fp(tag="win_amd64")),
            "node c: no TORCH_RECORDS entry for platform 'win_amd64'",
        ),
        (
            lambda r, f, b: f.__setitem__("b", _m4_fp(native={"libSystem.B.dylib": "ns-2"})),
            "node b: native objects differ from node a's",
        ),
        (
            lambda r, f, b: b["torch"].__setitem__("records", {"torch": "t-mac"}),
            "the record builder: torch RECORD differs from TORCH_RECORDS",
        ),
        (lambda r, f, b: f.pop("b"), "node b: no fingerprint"),
        (lambda r, f, b: r.pop("c"), "node c: no gate report"),
        (
            lambda r, f, b: f.__setitem__("a", _m4_fp(prefix_ok=False)),
            "node a: fingerprint prefix_ok is false",
        ),
        (
            lambda r, f, b: f.__setitem__(
                "c", _m4_fp(tag=M4_LINUX, probe_vector={"kernels": [True, "0.4"]})
            ),
            "node c: probe vector differs",
        ),
        (
            lambda r, f, b: b.__setitem__(
                "refusals", ["FP-RECORD: torch/lib/libtorch_cpu.dylib does not hash"]
            ),
            "the record builder: fingerprint refusals",
        ),
        (lambda r, f, b: f["a"].pop("refusals"), "node a: malformed fingerprint"),
        (
            lambda r, f, b: f["c"]["torch"].__setitem__("platform_tag", [M4_LINUX]),
            "node c: malformed fingerprint",
        ),
        (
            lambda r, f, b: r["c"].__setitem__(0, _m4_rep("cpu", [M4_CPU_X86])),
            "node c: malformed gate report",
        ),
    ],
    ids=[
        "gate_failed",
        "tf_version",
        "tf_record",
        "torch_git",
        "torch_record_entry",
        "no_platform_entry",
        "native_set",
        "builder",
        "no_fingerprint",
        "no_report",
        "prefix",
        "probe",
        "builder_fp_refusals",
        "no_refusals_list",
        "unhashable_tag",
        "unhashable_class",
    ],
)
def test_m4_admission_refuses_each_cause_naming_node_and_cause(mutate, want):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: hf_admission admits a failed gate, an unequal transformers version or
    # RECORD (or names no path), an unequal torch git_version, a torch RECORD unlike its
    # platform's TORCH_RECORDS entry, a platform with no entry, unequal native sets within
    # one platform and class, a builder outside the entry, a node with no fingerprint or no
    # report, a pycache prefix outside the launch's, or an unequal probe vector; or admits
    # a builder whose fingerprint lists FP-* refusals (it runs no gate), or crashes instead
    # of refusing a peer's malformed (unhashable) platform tag or device class
    reports, fps, builder = _m4_fleet()
    mutate(reports, fps, builder)
    out = _m4_admit(reports, fps, builder, pool=["a", "b", "c"])
    assert any(want in r for r in out), out
    # Only the mutated node (or the builder) is named.
    culprit = want.split(":")[0]
    assert all(r.startswith(culprit) for r in out), out


def test_m4_admission_refuses_a_map_of_two_cpu_classes_no_node_holds():
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: a device map whose classes no single node of the pool holds (two CPU
    # ATen capabilities on two nodes) is admitted, so the run has no oracle; or a map that a
    # Mac holding CPU and MPS covers is refused; or a slot's node lacks the slot's class
    reports, fps, builder = _m4_fleet()
    two_cpus = {
        (0, 0): {"node": "a", "device_class": M4_CPU_MAC},
        (1, 0): {"node": "c", "device_class": M4_CPU_X86},
    }
    from wantrain.resources.node import mixed_fleet

    # CONTROL-I25 refuses this map first at the coordinator; decision 8 stays below it.
    assert mixed_fleet((e["node"], e["device_class"]) for e in two_cpus.values()).startswith(
        "MIXED-FLEET: the device map spans 2 device classes"
    )
    out = _m4_admit(reports, fps, builder, two_cpus)
    assert out == [
        f"device map classes {sorted([M4_CPU_MAC, M4_CPU_X86])}: no single node of the pool "
        "holds them all, so the map has no oracle (ADR-0009 decision 8)"
    ]
    cpu_mps = {
        (0, 0): {"node": "a", "device_class": M4_CPU_MAC},
        (1, 0): {"node": "b", "device_class": M4_MPS},
    }
    assert _m4_admit(reports, fps, builder, cpu_mps) == []
    # The same map with only the CPU node and Linux in the pool: no node holds both.
    out = _m4_admit(reports, fps, builder, cpu_mps, pool=["c"])
    assert len(out) == 1 and "no single node of the pool holds them all" in out[0]
    wrong = {(0, 0): {"node": "c", "device_class": M4_MPS}}
    out = _m4_admit(reports, fps, builder, wrong)
    assert f"slot (0, 0): node c does not hold the slot's device class {M4_MPS!r}" in out[0]


def test_m4_admission_resume_refuses_another_record_naming_the_path():
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: a resume admission (recorded = the run's hf_code and RECORD text) admits
    # a node whose transformers RECORD differs in one line, or refuses without naming the
    # differing path record_diff returns; or admits a node whose torch RECORD differs from
    # its slot's recorded one; or refuses a node equal to the record
    reports, fps, builder = _m4_fleet()
    recorded = {
        "hf_code": {
            "transformers": "5.19.0",
            "transformers_record_sha256": hashlib.sha256(M4_RECORD_A.encode()).hexdigest(),
            "torch": "2.9.0",
            "torch_git_version": "g-abc",
        },
        "record_text": M4_RECORD_A,
    }
    mac_records = M4_TORCH_RECORDS[(M4_MAC, "2.9.0")]
    slot_map = {
        (0, 0): {"node": "a", "device_class": M4_CPU_MAC, "torch_records": mac_records},
        (1, 0): {"node": "b", "device_class": M4_MPS, "torch_records": mac_records},
    }
    assert _m4_admit(reports, fps, builder, slot_map, recorded=recorded) == []
    fps["b"] = _m4_fp(record=M4_RECORD_B)
    out = _m4_admit(reports, fps, builder, slot_map, recorded=recorded)
    assert out == [
        "node b: transformers RECORD differs from the run's recorded hf_code's at "
        "transformers/models/llama/modeling_llama.py"
    ]
    # A Linux node healed into a slot recorded on a Mac: its torch RECORD is its platform's
    # entry (admitted as a node) but not the slot's.
    fps["b"] = _m4_fp()
    reports["c"] = [_m4_rep("cpu", M4_CPU_X86), _m4_rep("cpu", M4_CPU_MAC)]
    healed = {**slot_map, (0, 0): dict(slot_map[(0, 0)], node="c")}
    out = _m4_admit(reports, fps, builder, healed, recorded=recorded)
    assert out == [
        "slot (0, 0): node c's torch RECORD differs from the slot's recorded one at torch"
    ]


def test_m4_admission_resume_refuses_a_kept_text_edited_in_one_line():
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: the resume admission compares only the sha256 a node reports with the
    # recorded hf_code's, so a kept RECORD text that differs from every node's in one line
    # (standing for another transformers build) is admitted, or refused without the path
    reports, fps, builder = _m4_fleet()
    sha_a = hashlib.sha256(M4_RECORD_A.encode()).hexdigest()
    code = {
        "transformers": "5.19.0",
        "transformers_record_sha256": sha_a,
        "torch": "2.9.0",
        "torch_git_version": "g-abc",
    }
    slot_map = {(0, 0): {"node": "a", "device_class": M4_CPU_MAC}}
    pool = ["a", "b"]
    same = {"hf_code": code, "record_text": M4_RECORD_A}
    assert _m4_admit(reports, fps, builder, slot_map, pool, recorded=same) == []
    edited = {"hf_code": code, "record_text": M4_RECORD_B}
    out = _m4_admit(reports, fps, builder, slot_map, pool, recorded=edited)
    # The builder's fingerprint is re-checked against the run's record as a node's.
    assert out == [
        f"{n}: transformers RECORD differs from the run's recorded hf_code's at "
        "transformers/models/llama/modeling_llama.py"
        for n in ("the record builder", "node a", "node b")
    ]


def test_m4_admission_resume_fails_closed_on_an_emptied_kept_text_or_a_node_without_text():
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: the text compare is skipped when the kept RECORD text is empty (a truncated
    # run record admits on the sha256 alone) or when a node's fingerprint carries no RECORD text
    # that decodes (the node is admitted on its self-reported sha256 alone)
    reports, fps, builder = _m4_fleet()
    code = {
        "transformers": "5.19.0",
        "transformers_record_sha256": hashlib.sha256(M4_RECORD_A.encode()).hexdigest(),
        "torch": "2.9.0",
        "torch_git_version": "g-abc",
    }
    slot_map = {(0, 0): {"node": "a", "device_class": M4_CPU_MAC}}
    emptied = {"hf_code": code, "record_text": ""}
    out = _m4_admit(reports, fps, builder, slot_map, ["a"], recorded=emptied)
    assert out == [
        f"{n}: transformers RECORD differs from the run's recorded hf_code's at "
        "transformers/__init__.py"
        for n in ("the record builder", "node a")
    ]
    fps["a"]["transformers"]["record_gz_b64"] = "!"
    same = {"hf_code": code, "record_text": M4_RECORD_A}
    out = _m4_admit(reports, fps, builder, slot_map, ["a"], recorded=same)
    assert out == ["node a: transformers RECORD text missing or does not decode"]


def test_m4_admission_find_replacement_takes_only_the_slots_class():
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: find_replacement heals an hf slot with a spare whose gate does not report
    # the slot's recorded device class (another CPU capability), or refuses the spare that
    # does; without a class the rule is unchanged
    from wantrain.fault_tolerance.replacement import find_replacement

    def node(i, region="eu"):
        return NodeInfo(
            node_id=f"n{i}",
            name=f"w{i}",
            host=f"10.0.0.{i}",
            data_port=1,
            device="cpu",
            gpu_name="cpu",
            vram_total_bytes=8 << 30,
            vram_free_bytes=8 << 30,
            tflops_fp32=1.0,
            region=region,
        )

    dead, other, same = node(0), node(1, region="eu"), node(2, region="us")
    classes = {"n1": [M4_CPU_X86], "n2": [M4_CPU_MAC, M4_MPS]}
    kw = dict(match="any", device_class=M4_CPU_MAC, gate_classes=classes)
    assert find_replacement([other, same], dead, [], **kw)[0].node_id == "n2"
    assert find_replacement([other], dead, [], **kw) is None
    # A spare that was never gated holds no class.
    assert find_replacement([same], dead, [], match="any", device_class=M4_CPU_MAC) is None
    # Control: without a class the closer spare wins, as before.
    assert find_replacement([other, same], dead, [], match="any")[0].node_id == "n1"


def test_m4_admission_plan_model_mismatch_names_each_dropped_key():
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: after a plan-mode build_spec a dropped moe_experts, an hf_config whose
    # canonical sha256 differs, a dropped hf_impl or another arch passes (a dense run), or a
    # corpus vocab_size change or a reordered hf_config is flagged
    from wantrain.model.gpt import GPTConfig

    hf = {"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2}
    plan = {"model": "toy", "moe_experts": 4, "moe_top_k": 2, "vocab_size": 256}
    cfg = GPTConfig(vocab_size=512)
    # The planner dropped moe_experts: dense (0) once model-core lands the field, absent
    # before; moe_top_k came through.
    if hasattr(cfg, "moe_experts"):
        cfg.moe_experts = 0
    cfg.moe_top_k = 2
    out = coord_mod.plan_model_mismatch(plan, cfg)
    assert len(out) == 1 and out[0].startswith("moe_experts: the plan gives 4"), out
    cfg.moe_experts, cfg.moe_top_k = 4, 2
    assert coord_mod.plan_model_mismatch(plan, cfg) == []

    plan = {"arch": "hf", "hf_config": hf, "hf_impl": "modules"}
    same = {"arch": "hf", "hf_config": dict(reversed(list(hf.items()))), "hf_impl": "modules"}
    assert coord_mod.plan_model_mismatch(plan, same) == []
    other = dict(same, hf_config=dict(hf, hidden_size=32))
    out = coord_mod.plan_model_mismatch(plan, other)
    want = hashlib.sha256(
        json.dumps(hf, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    assert out == [
        f"hf_config: the plan's canonical sha256 {want} != the spec's "
        + hashlib.sha256(
            json.dumps(dict(hf, hidden_size=32), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
    ]
    out = coord_mod.plan_model_mismatch(plan, {"arch": "builtin", "hf_config": hf})
    assert out == [
        "arch: the plan gives 'hf', the spec's model has 'builtin'",
        "hf_impl: the plan gives 'modules', the spec's model has no hf_impl",
    ]


def test_m4_admission_coordinator_never_imports_transformers(tmp_path):
    # covers: CONTROL-I24
    # WOULD-FAIL-IF: a module-level import of transformers or wantrain.model.hfm in the
    # coordinator (or anything it imports), or one reached by hf_admission or
    # plan_model_mismatch; it holds whether or not transformers is installed
    import subprocess
    import sys

    child = r"""
import json, sys
seen = []
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "transformers" or name.startswith("transformers.") or name == "wantrain.model.hfm":
            seen.append(name)
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, Block())
import wantrain.coordinator.coordinator as c
fp = {"transformers": {"version": "5.19.0", "record_sha256": "x", "record_gz_b64": "!"},
      "torch": {"version": "2.9.0", "git_version": "g", "platform_tag": "t", "records": {}}}
out = c.hf_admission({"a": [{"ok": True, "device": "cpu", "device_class": "cpu:x"}]}, {"a": fp},
                     fp, {(0, 0): {"node": "a", "device_class": "cpu:x"}}, ["a"],
                     torch_records={}, record_diff=lambda a, b: None)
mm = c.plan_model_mismatch({"arch": "hf", "hf_config": {"model_type": "llama"}}, {"arch": "hf"})
print(json.dumps({"seen": seen, "refusals": len(out), "mismatch": len(mm),
                  "loaded": sorted(m for m in sys.modules
                                   if m == "transformers" or m == "wantrain.model.hfm")}))
"""
    p = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=120, cwd=tmp_path
    )
    assert p.returncode == 0, p.stderr[-2000:]
    got = json.loads(p.stdout.strip().splitlines()[-1])
    assert got["seen"] == [] and got["loaded"] == [], got
    # The crafted inputs reached both functions: no TORCH_RECORDS entry; hf_config dropped.
    assert got["refusals"] >= 1 and got["mismatch"] == 1


# ---------------------------------------------------------------- CONTROL-I25

I25_A, I25_B = "cpu:x86_64:AVX2", "cpu:x86_64:AVX512"
I25_TAIL = "only one device class is admitted until the heterogeneous track returns"


def _i25_model(arch):
    """The model of each arch CONTROL-I25 names: builtin (the toy), general (the
    builtin arch with RMSNorm, SwiGLU and RoPE: the general layer path) and hf
    (set by name: nothing hf runs before the refusal)."""
    from wantrain.model.gpt import make_config

    cfg = make_config("toy")
    if arch == "general":
        cfg.norm, cfg.mlp, cfg.pos, cfg.bias = "rmsnorm", "swiglu", "rope", False
    if arch == "hf":
        cfg.arch = "hf"
        cfg.hf_config = {"model_type": "llama", "num_hidden_layers": 2}
    return cfg


def _i25_info(i, cls, tflops=1.0):
    return _pw_info(i, hostname=f"h{i}", device_class=cls, tflops_fp32=tflops, host=f"10.0.0.{i}")


def _i25_coordinator(tmp_path, infos):
    """A real Coordinator with `infos` registered and idle. FAKE at the
    worker-network boundary, named as such: `_request` records each request
    (a GATE, PROBE or ASSIGN would land here) and answers {"ok": True}."""
    coord = Coordinator(config=CoordinatorConfig(state_dir=str(tmp_path / "coord"), probe=False))
    sent = []
    coord._request = lambda w, msg, *a, **k: sent.append((w.node_id, msg["kind"])) or {"ok": True}
    for i, info in enumerate(infos):
        coord.workers[info.node_id] = Worker(info, None, f"s{i}", None, peer_id=f"{i:032x}")
    return coord, sent


@pytest.mark.parametrize("arch", ["builtin", "general", "hf"])
def test_i25_two_class_pool_is_refused_at_launch_naming_both_classes(tmp_path, monkeypatch, arch):
    # covers: CONTROL-I25
    # WOULD-FAIL-IF: a two-class device map is admitted for a builtin/general/hf arch at launch
    # or heal: selection fields a fleet across two device classes, or the launch admission
    # (the check on the selected map) admits one selection hands it, or any request (GATE,
    # PROBE, ASSIGN) reaches a worker before the refusal
    spec = JobSpec(model=_i25_model(arch), steps=2, stages=2)
    infos = [_i25_info(0, I25_A), _i25_info(1, I25_B, tflops=2.0)]
    coord, sent = _i25_coordinator(tmp_path / "sel", infos)
    res = coord.run_job(Job(spec=spec), wait_workers_s=0.0)
    assert not res.ok and sent == [] and all(w.run_id is None for w in coord.workers.values())
    assert res.reason == (
        "RuntimeError: no fleet: MIXED-FLEET: no single device class can field 2 nodes on "
        f"distinct addresses (pool: {I25_A} x1, {I25_B} x1); {I25_TAIL}"
    )
    # The admission check itself, on a mixed selection handed past the selector.
    from wantrain.scheduler.selection import Selection

    def mixed_selection(pool, n, *a, **k):
        return Selection(ok=True, chosen=sorted(pool, key=lambda x: x.node_id), note="forced")

    monkeypatch.setattr(coord_mod, "select_nodes", mixed_selection)
    coord, sent = _i25_coordinator(tmp_path / "adm", infos)
    res = coord.run_job(Job(spec=spec), wait_workers_s=0.0)
    assert not res.ok and sent == [] and all(w.run_id is None for w in coord.workers.values())
    assert res.reason == (
        "RuntimeError: admission refused (CONTROL-I25): MIXED-FLEET: the device map spans 2 "
        f"device classes ({I25_A}: w0; {I25_B}: w1); {I25_TAIL}"
    )


@pytest.mark.parametrize("arch", ["builtin", "general", "hf"])
def test_i25_plan_job_and_place_refuse_two_classes_and_admit_one_at_two_speeds(arch):
    # covers: CONTROL-I25
    # WOULD-FAIL-IF: planner.plan_job or placement.place admits a two-class fleet for a
    # builtin/general/hf arch (or prices it before refusing), admits a node reporting no
    # class, or refuses one class at two speeds (the simulated heterogeneous fleet)
    from wantrain.scheduler import placement
    from wantrain.scheduler.planner import Fleet, Node, plan_job

    cfg = _i25_model(arch)

    def fleet(*classes):
        nodes = [
            Node(f"10.0.0.{i}", tflops=1.0 + i, vram_bytes=8 << 30, device="cpu", device_class=c)
            for i, c in enumerate(classes)
        ]
        return Fleet(nodes=nodes, link_mbps=1e6, rtt_ms=0.1, duplex="per_direction")

    with pytest.raises(ValueError) as e:
        plan_job(cfg, fleet(I25_A, I25_B), steps=1, device="cpu")
    assert str(e.value) == (
        f"MIXED-FLEET: the device map spans 2 device classes ({I25_A}: 10.0.0.0; "
        f"{I25_B}: 10.0.0.1); {I25_TAIL}"
    )
    with pytest.raises(
        ValueError, match=r"^MIXED-FLEET: node 10\.0\.0\.1 reports no device class$"
    ):
        plan_job(cfg, fleet(I25_A, ""), steps=1, device="cpu")
    spec = JobSpec(model=cfg, steps=2, stages=2)
    with pytest.raises(ValueError) as e:
        placement.place(spec, [[_i25_info(0, I25_A)], [_i25_info(1, I25_B)]])
    assert str(e.value) == (
        f"MIXED-FLEET: the device map spans 2 device classes ({I25_A}: w0; {I25_B}: w1); {I25_TAIL}"
    )
    assert spec.endpoints is None and spec.partition is None  # nothing was placed
    with pytest.raises(ValueError, match=r"^MIXED-FLEET: node w1 reports no device class$"):
        placement.place(spec, [[_i25_info(0, I25_A)], [_i25_info(1, "")]])
    # Control: one class at two speeds is admitted, and cut by speed.
    if arch != "hf":  # an hf plan prices units from the builder's census (I-10)
        out = plan_job(cfg, fleet(I25_A, I25_A), steps=1, device="cpu")
        assert out["spec"] is not None
    one = JobSpec(model=_i25_model("builtin"), steps=2, stages=2)
    r = placement.place(one, [[_i25_info(0, I25_A, 1.0)], [_i25_info(1, I25_A, 3.0)]])
    assert r["stage_tflops"] == [1.0, 3.0]
    a, b = one.partition
    assert (b[1] - b[0]) > (a[1] - a[0])


@pytest.mark.parametrize("arch", ["builtin", "general", "hf"])
def test_i25_heal_with_only_a_spare_of_another_class_fails_and_relaunches_nothing(tmp_path, arch):
    # covers: CONTROL-I25
    # WOULD-FAIL-IF: a two-class device map is admitted for a builtin/general/hf arch at launch
    # or heal: _heal takes a spare of another device class (here the faster one), gates it,
    # or relaunches; or a spare of the run's class at another speed is refused
    def setup(sub, spares):
        infos = [_i25_info(0, I25_A), _i25_info(1, I25_A)]
        coord, sent = _i25_coordinator(tmp_path / sub, infos + spares)
        os.makedirs(tmp_path / sub / "run")
        run = _Run("run-i25", str(tmp_path / sub / "run"))
        run.spec = JobSpec(model=_i25_model(arch), steps=4, stages=2)
        run.spec_json = run.spec.to_json()
        if arch == "hf":  # the hf branch would GATE every spare it considers
            run.hf = {"slots": {(0, 0): {"device_class": I25_A}, (1, 0): {"device_class": I25_A}}}
            run.hf.update(reports={}, fingerprints={}, fresh=set())
        for slot, info in (((0, 0), infos[0]), ((1, 0), infos[1])):
            run.slots[slot] = info.node_id
            coord.workers[info.node_id].run_id = run.run_id
        relaunched = []
        coord._resume_spec = lambda run, step: "resume-spec"
        coord._restore_and_relaunch = lambda run, step, spec_json: relaunched.append(step)
        coord.run = run
        return coord, run, sent, relaunched

    other = _i25_info(5, I25_B, tflops=50.0)
    coord, run, sent, relaunched = setup("other", [other])
    with pytest.raises(RuntimeError) as e:
        coord._heal(run, (1, 0), "killed")
    assert str(e.value) == (
        "slot (1, 0) died (killed) and the heal fails (CONTROL-I25); nothing relaunched: "
        f"MIXED-FLEET: no spare of the run's device class {I25_A} (pool: {I25_B} x1)"
    )
    assert run.slots[(1, 0)] == "n1" and run.heals == [] and relaunched == []
    assert {kind for _, kind in sent} <= {"ABORT"} and coord.workers["n5"].run_id is None
    # Control: a spare of the run's class at another speed takes the slot.
    if arch == "hf":
        return  # its heal gates the spare (CONTROL-I24); covered by the coordinator tests
    same = _i25_info(6, I25_A, tflops=0.5)
    coord, run, sent, relaunched = setup("same", [other, same])
    coord._heal(run, (1, 0), "killed")
    assert run.slots[(1, 0)] == "n6" and relaunched == [None]


def test_i25_resume_map_of_two_classes_relaunches_nothing(tmp_path):
    # covers: CONTROL-I25
    # WOULD-FAIL-IF: _resume_spec (heal, restart in place and rebalance all build their map
    # there) admits a slot map whose nodes span two device classes, or a node reporting none
    infos = [_i25_info(0, I25_A), _i25_info(1, I25_B), _i25_info(2, "")]
    coord, sent = _i25_coordinator(tmp_path, infos)
    run = _Run("run-i25r", str(tmp_path))
    run.spec_json = JobSpec(steps=4, stages=2).to_json()
    run.slots = {(0, 0): "n0", (1, 0): "n1"}
    with pytest.raises(RuntimeError) as e:
        coord._resume_spec(run, 2)
    assert str(e.value) == (
        "admission refused (CONTROL-I25); nothing relaunched: MIXED-FLEET: the device map "
        f"spans 2 device classes ({I25_A}: w0; {I25_B}: w1); {I25_TAIL}"
    )
    run.slots = {(0, 0): "n0", (1, 0): "n2"}
    with pytest.raises(RuntimeError, match="MIXED-FLEET: node w2 reports no device class$"):
        coord._resume_spec(run, 2)
    assert sent == []


def test_i25_hello_class_is_the_gate_class_and_a_classless_node_is_never_selected():
    # covers: CONTROL-I25
    # WOULD-FAIL-IF: the HELLO's device class is not the string hfm.device_class returns (two
    # functions that can drift), a non-string class is accepted, a node reporting no class
    # is selected, or an hf slot whose gate class differs from its node's HELLO is admitted
    from wantrain.model import hfm
    from wantrain.resources.probe import measure

    probe = measure("cpu", bench=False)
    assert probe["deviceClass"] == hfm.device_class("cpu") != ""
    hello = {"advertise_host": "10.0.0.9", "data_port": 1, "probe": probe}
    assert NodeInfo.from_hello("n9", hello).device_class == probe["deviceClass"]
    assert NodeInfo.from_hello("n9", dict(hello, probe={})).device_class == ""
    with pytest.raises(ValueError, match="device_class: not a string"):
        NodeInfo.from_hello("n9", dict(hello, probe={"deviceClass": ["cpu"]}))
    from wantrain.scheduler.selection import select_nodes

    pool = [_i25_info(0, I25_A), _i25_info(1, ""), _i25_info(2, "")]
    sel = select_nodes(pool, 2)
    assert not sel.ok and sel.reason.startswith(
        f"MIXED-FLEET: no single device class can field 2 nodes on distinct addresses "
        f"(pool: <none> x2, {I25_A} x1)"
    )
    assert select_nodes([*pool, _i25_info(3, I25_A, 9.0)], 2).ok
    slot_map = {(0, 0): {"node": "n0", "device_class": I25_B}}
    assert coord_mod.class_refusals(slot_map, {"n0": pool[0]}) == [
        f"MIXED-FLEET: slot (0, 0): node w0's gate reports device class {I25_B!r}, "
        f"its HELLO {I25_A!r}"
    ]
    assert (
        coord_mod.class_refusals({(0, 0): {"node": "n0", "device_class": I25_A}}, {"n0": pool[0]})
        == []
    )
